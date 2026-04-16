# pi-anthropic-vertex

[Pi](https://github.com/badlogic/pi-mono) extension for running Claude models on Google Vertex AI.

Pi's built-in `google-vertex` provider only supports Gemini models. This extension adds an `anthropic-vertex` provider that uses Anthropic's official [`@anthropic-ai/vertex-sdk`](https://github.com/anthropics/anthropic-sdk-typescript/tree/main/packages/vertex-sdk) to route Claude requests through your GCP project — billing goes through Vertex AI, no separate Anthropic API key needed.

## Models

| Model | ID | Context | Max Output |
|-------|----|---------|------------|
| Claude Opus 4.7 | `claude-opus-4-7` | 1M | 128k |
| Claude Opus 4.6 | `claude-opus-4-6` | 1M | 128k |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` | 1M | 64k |
| Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | 200k | 16k |

## Prerequisites

1. A GCP project with Vertex AI API enabled and Claude models available
2. Application Default Credentials configured:
   ```bash
   gcloud auth application-default login
   ```
3. Environment variables (or set `projectId`/`region` in the config file):
   ```bash
   export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
   export GOOGLE_CLOUD_LOCATION="us-east5"
   ```

## Install

From a local path:

```bash
cd /path/to/pi-anthropic-vertex && npm install
pi install /path/to/pi-anthropic-vertex
```

From git (Pi clones + runs `npm install` automatically):

```bash
pi install git:github.com/your-org/pi-anthropic-vertex
```

## Usage

```bash
# Start with Opus
pi --model anthropic-vertex/claude-opus-4-6

# Start with Sonnet
pi --model anthropic-vertex/claude-sonnet-4-6

# With a specific thinking level
pi --model anthropic-vertex/claude-opus-4-6 --thinking high
```

Switch models mid-session with `/model` or `Ctrl+L`. Cycle favorites with `Ctrl+P`.

## Configuration

Create `~/.pi/agent/anthropic-vertex.json`. See [`anthropic-vertex.example.json`](./anthropic-vertex.example.json) for a fully documented example.

Minimal config:

```json
{
  "projectId": "your-gcp-project-id",
  "region": "us-east5"
}
```

Full config with all options:

```json
{
  "projectId": "your-gcp-project-id",
  "region": "us-east5",
  "thinking": "adaptive",
  "temperature": null,
  "cache": {
    "system": { "enabled": true, "ttl": "1h" },
    "tools": { "enabled": true, "ttl": "1h" },
    "messages": { "enabled": true, "ttl": "5m" }
  }
}
```

### `projectId` / `region`

GCP project and Vertex AI region. Can also be set via `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` environment variables (env vars take precedence over config file).

### `thinking`

Controls how Claude reasons before responding. Pi has thinking levels (`minimal`, `low`, `medium`, `high`, `xhigh`, `off`) that the user selects in the UI. This config determines how those levels translate to Anthropic's API.

| Value | Behavior |
|-------|----------|
| `"adaptive"` (default) | Claude decides how deeply to think. Pi levels map to effort hints. |
| `{ minimal: N, ..., xhigh: N }` | Explicit `budget_tokens` per Pi level (deprecated for 4.6) |

Thinking can also be disabled per-session in Pi's UI by setting the thinking level to "off".

#### Adaptive (recommended)

Claude evaluates task complexity and scales thinking automatically. Pi's thinking level maps to an Anthropic effort hint:

| Pi level | Opus effort | Sonnet effort | Behavior |
|----------|-------------|---------------|----------|
| minimal | `low` | `low` | Light reasoning, skips thinking for simple tasks |
| low | `low` | `low` | Light reasoning |
| medium | `medium` | `medium` | Moderate reasoning |
| high | `high` | `high` | Deep reasoning, thinks on most tasks |
| xhigh | `max` | `high` | Maximum depth (Opus only — Sonnet caps at `high`) |

#### Explicit budget map (deprecated for 4.6)

> **Note**: Anthropic has deprecated `budget_tokens` for Opus 4.6 and Sonnet 4.6.
> It still works but may truncate reasoning mid-thought (budget too low) or waste
> tokens over-thinking simple tasks (budget too high). Use `"adaptive"` instead
> unless you need a hard cost ceiling.

```json
{
  "thinking": {
    "minimal": 16000,
    "low": 32000,
    "medium": 64000,
    "high": 96000,
    "xhigh": 128000
  }
}
```

All 5 levels required (validated on load — falls back to adaptive with a warning if invalid). Values exceeding `model.maxTokens` are capped automatically.

### `temperature`

Optional (0.0–1.0). Only applied when thinking is `"off"` in Pi's UI — Anthropic rejects the temperature parameter when thinking is enabled. `null` or omit to use Anthropic's default (1.0). Useful for creative/adversarial prompting experiments.

### `cache`

Prompt caching on Vertex AI. Omit or set to `false` to disable entirely.

Anthropic's prompt caching on Vertex AI requires explicit `cache_control` breakpoints (automatic caching is not available on Vertex). This extension injects them on three targets:

| Target | What gets cached | Recommended TTL |
|--------|-----------------|-----------------|
| `system` | System prompt (AGENTS.md, tools preamble) | `1h` — rarely changes |
| `tools` | Tool definitions (read, write, edit, bash...) | `1h` — rarely changes |
| `messages` | Conversation history (second-to-last user turn) | `5m` — prefix is stable between turns |

Each target can be toggled and configured independently:

```json
{
  "cache": {
    "system": { "enabled": true, "ttl": "1h" },
    "tools": { "enabled": true, "ttl": "1h" },
    "messages": { "enabled": true, "ttl": "5m" }
  }
}
```

Shorthand: `"cache": true` uses sensible defaults (system/tools at 1h, messages at 5m).

#### Cost impact

| | Price (per 1M tokens) |
|-|----------------------|
| Normal input | Base price |
| Cache write (first time) | Base + 25% |
| Cache read (subsequent) | Base * 10% (90% savings) |

For Opus at $5/M input: cache reads cost $0.50/M. The write premium is recouped on the second request with the same cached prefix.

#### TTL options

- `"5m"` — 5-minute cache lifetime (default for Anthropic). Good for interactive sessions where you send messages within minutes.
- `"1h"` — 1-hour cache lifetime. Costs more for cache writes but survives longer breaks between messages. Good for system prompts and tools that almost never change.

## How it works

This extension registers a custom Pi provider using `streamSimple` to handle the streaming:

1. Pi calls `streamSimple(model, context, options)` with the conversation context and thinking level
2. We create an `AnthropicVertex` client (lazy-initialized, uses GCP ADC for auth)
3. Messages are preprocessed: errored/aborted turns dropped, orphaned tool calls get synthetic results, cross-model redacted thinking filtered
4. Context is converted to Anthropic message format (toolCall→tool_use, toolResult→user, thinking signatures preserved) with cache breakpoints injected
5. Beta features (`context-1m`, `fine-grained-tool-streaming`) sent via `client.beta.messages.stream()`
6. Pi's thinking level is mapped to Anthropic's adaptive effort or explicit budget_tokens
7. Response is streamed back, mapping Anthropic events to Pi's event format (text_start/delta/end, thinking_start/delta/end, toolcall_start/delta/end)
8. Token usage (including cache read/write) and costs are tracked and reported

### Event mapping

| Anthropic | Pi | Notes |
|-----------|----|-------|
| `content_block_start` (text) | `text_start` | |
| `content_block_delta` (text_delta) | `text_delta` | |
| `content_block_stop` (text) | `text_end` | Includes `content` field |
| `content_block_start` (thinking) | `thinking_start` | Tracks `thinkingSignature` |
| `content_block_delta` (thinking_delta) | `thinking_delta` | |
| `content_block_delta` (signature_delta) | *(no event)* | Accumulated for multi-turn |
| `content_block_stop` (thinking) | `thinking_end` | Includes `content` field |
| `content_block_start` (tool_use) | `toolcall_start` | `input`→`arguments` |
| `content_block_delta` (input_json_delta) | `toolcall_delta` | JSON chunks accumulated |
| `content_block_stop` (tool_use) | `toolcall_end` | Triggers Pi's tool loop |
| `end_turn` | `stop` | |
| `tool_use` | `toolUse` | |
| `max_tokens` | `length` | |
| `refusal` | `error` | |

### Known limitations

- **xhigh thinking**: Pi's `supportsXhigh()` checks `model.api === "anthropic-messages"` ([#2040](https://github.com/badlogic/pi-mono/issues/2040)), but `xhigh` works in practice with our custom API type. The `xhigh` → `max` effort mapping is only applied for Opus 4.6 (Sonnet 4.6 caps at `high`).
- **No `transformMessages` import**: Pi's message preprocessing function isn't exported from `@mariozechner/pi-ai`. We reimplement the essential parts (error/abort filtering, synthetic tool results, cross-model thinking cleanup).
- **Model costs are hardcoded**: If Anthropic changes pricing, costs in the extension need manual update.
- **No incremental tool arguments**: During `toolcall_delta`, `arguments` stays `{}` until the block completes. Pi's built-in provider uses `parseStreamingJson` for incremental parsing, but it's not exported.

### Maintenance

This extension reimplements parts of Pi's internal Anthropic provider. Changes to Pi's message format, event contract, or the Anthropic API may require updates. Key files to monitor:
- `@mariozechner/pi-ai/dist/providers/anthropic.js` — Pi's canonical provider
- `@mariozechner/pi-ai/dist/providers/transform-messages.js` — message preprocessing
- `@mariozechner/pi-ai/dist/types.d.ts` — event and type contracts

## Development

```bash
# Clone and install
git clone <repo-url> && cd pi-anthropic-vertex
npm install

# Install locally in Pi
pi install /path/to/pi-anthropic-vertex

# Test
pi --model anthropic-vertex/claude-opus-4-6 "Hello, which model are you?"
```

The extension is loaded via [jiti](https://github.com/unjs/jiti) — TypeScript works without a build step.

## Credits

Cross-referenced against:
- [Pi's built-in Anthropic provider](https://github.com/badlogic/pi-mono/blob/main/packages/ai/src/providers/anthropic.ts)
- [basnijholt/pi-anthropic-vertex](https://github.com/basnijholt/pi-anthropic-vertex)

## License

MIT
