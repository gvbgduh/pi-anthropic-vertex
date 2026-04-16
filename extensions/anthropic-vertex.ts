import { readFileSync } from "fs";
import { join } from "path";
import { homedir } from "os";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import {
  type AssistantMessage,
  type AssistantMessageEventStream,
  type Context,
  type Model,
  type SimpleStreamOptions,
  createAssistantMessageEventStream,
} from "@mariozechner/pi-ai";

// ── Anthropic → Pi event/type mappings ──────────────────────────────
// Neither SDK uses enums — both use discriminated unions with string
// literals. These constants document the mapping in one place.

// Anthropic content block types → Pi content block types
const ContentType = {
  // Anthropic          Pi
  text: "text", //      "text"      (field: text → text)
  thinking: "thinking", //  "thinking"  (field: thinking → thinking)
  tool_use: "toolCall", //  "toolCall"  (field: input → arguments)
} as const;

// Anthropic stream events → Pi stream events
const StreamEvent = {
  text_start: "text_start",
  thinking_start: "thinking_start",
  toolcall_start: "toolcall_start",
  text_delta: "text_delta",
  thinking_delta: "thinking_delta",
  toolcall_delta: "toolcall_delta",
  text_end: "text_end",
  thinking_end: "thinking_end",
  toolcall_end: "toolcall_end",
} as const;

// Anthropic stop reasons → Pi stop reasons (Pi type: "stop"|"length"|"toolUse"|"error"|"aborted")
const StopReason: Record<string, string> = {
  end_turn: "stop",
  tool_use: "toolUse",
  max_tokens: "length",
  refusal: "error",
  pause_turn: "stop",
  stop_sequence: "stop",
  sensitive: "error",
};

// Beta features passed as body params on Vertex AI (not HTTP headers —
// Vertex rejects them as headers with "Unexpected value(s)").
// - context-1m: unlocks 1M input tokens (without it, hard 200k limit)
// - fine-grained-tool-streaming: granular tool call streaming events
const VERTEX_BETAS = [
  "context-1m-2025-08-07",
  "fine-grained-tool-streaming-2025-05-14",
];

// ── Config ──────────────────────────────────────────────────────────

interface CacheTarget {
  enabled: boolean;
  ttl: "5m" | "1h";
}

interface CacheConfig {
  system: CacheTarget;
  tools: CacheTarget;
  messages: CacheTarget;
}

// Explicit per-level budget_tokens map.
// DEPRECATED for 4.6 models — Anthropic recommends adaptive thinking instead.
// Still works and gives a hard cost ceiling, but the model may truncate
// reasoning mid-thought or over-think simple tasks. All 5 levels required.
interface ThinkingBudgetMap {
  minimal: number;
  low: number;
  medium: number;
  high: number;
  xhigh: number;
}

const ADAPTIVE = "adaptive" as const;

//  "adaptive" — Claude decides how deeply to think per task (recommended for 4.6).
//               Pi's level maps to Anthropic effort: low/medium/high/max.
//  { ... }    — explicit budget_tokens per Pi level (deprecated for 4.6, see above).
//
// Thinking can also be disabled per-session in Pi's UI (thinking level → "off").
type ThinkingConfig = typeof ADAPTIVE | ThinkingBudgetMap;

// Pi thinking level → Anthropic adaptive effort string.
// "max" is only supported on Opus 4.6 — Sonnet 4.6 caps at "high".
// Pi's built-in provider clamps xhigh per model the same way.
function mapEffort(level: string, modelId: string): string | null {
  switch (level) {
    case "minimal":
    case "low":
      return "low";
    case "medium":
      return "medium";
    case "high":
      return "high";
    case "xhigh":
      // "max" only supported on Opus models
      return modelId.includes("opus") ? "max" : "high";
    default:
      return null;
  }
}

interface Config {
  projectId: string;
  region: string;
  thinking: ThinkingConfig;
  cache: CacheConfig | null;
  // Optional temperature (0.0–1.0). Only applied when thinking is off —
  // Anthropic rejects the temperature param when thinking is enabled.
  // Useful for creative/adversarial prompting experiments.
  temperature: number | null;
}

const CONFIG_PATH = join(homedir(), ".pi", "agent", "anthropic-vertex.json");

const CACHE_DISABLED: CacheConfig = {
  system: { enabled: false, ttl: "5m" },
  tools: { enabled: false, ttl: "5m" },
  messages: { enabled: false, ttl: "5m" },
};

function loadConfig(): Config | null {
  let file: Record<string, any> = {};
  try {
    file = JSON.parse(readFileSync(CONFIG_PATH, "utf-8"));
  } catch {
    // No config file — use env vars + defaults
  }

  const projectId =
    process.env.GOOGLE_CLOUD_PROJECT ||
    process.env.ANTHROPIC_VERTEX_PROJECT_ID ||
    file.projectId;

  if (!projectId) {
    console.warn(
      "[anthropic-vertex] Skipping: set GOOGLE_CLOUD_PROJECT, " +
        "ANTHROPIC_VERTEX_PROJECT_ID, or projectId in " +
        CONFIG_PATH
    );
    return null;
  }

  const region =
    process.env.GOOGLE_CLOUD_LOCATION ||
    process.env.CLOUD_ML_REGION ||
    file.region ||
    "global";

  const thinking = resolveThinking(file.thinking);
  const cache = resolveCache(file.cache);
  const temperature =
    typeof file.temperature === "number" ? file.temperature : null;

  return { projectId, region, thinking, cache, temperature };
}

function resolveThinking(value: any): ThinkingConfig {
  if (typeof value === "object" && value !== null) {
    // Validate all 5 levels are present and numeric to avoid budget_tokens: NaN
    const required = ["minimal", "low", "medium", "high", "xhigh"] as const;
    const valid = required.every(
      (k) => typeof value[k] === "number" && value[k] > 0
    );
    if (!valid) {
      console.warn(
        "[anthropic-vertex] Invalid thinking budget map (need all 5 levels as positive numbers), falling back to adaptive"
      );
      return ADAPTIVE;
    }
    return value as ThinkingBudgetMap;
  }
  return ADAPTIVE;
}

// Build thinking-related request fields based on config mode and Pi's level.
// Returns fields to spread onto the request: { thinking, output_config? }
function resolveThinkingPayload(
  config: ThinkingConfig,
  level: string | undefined,
  maxTokens: number,
  modelId: string
): Record<string, any> | null {
  // Pi passes "off" when user disables thinking in the UI
  if (!level || level === "off") return null;

  if (config === ADAPTIVE) {
    const effort = mapEffort(level, modelId);
    if (!effort) return null;
    // Adaptive: thinking and effort are separate top-level fields
    return { thinking: { type: "adaptive" }, output_config: { effort } };
  }

  // Explicit budget mode (deprecated for 4.6, but gives hard cost ceiling).
  // max_tokens must be strictly greater than budget_tokens.
  const budget = config[level as keyof ThinkingBudgetMap];
  if (budget === undefined || budget <= 0) return null;
  return {
    thinking: {
      type: "enabled",
      budget_tokens: Math.min(budget, maxTokens - 1),
    },
  };
}

function resolveCache(fileConfig: any): CacheConfig | null {
  if (fileConfig === undefined || fileConfig === null) return null;
  if (fileConfig === false) return null;

  if (fileConfig === true) {
    return {
      system: { enabled: true, ttl: "1h" },
      tools: { enabled: true, ttl: "1h" },
      messages: { enabled: true, ttl: "5m" },
    };
  }

  return {
    system: resolveCacheTarget(fileConfig.system, { enabled: true, ttl: "1h" }),
    tools: resolveCacheTarget(fileConfig.tools, { enabled: true, ttl: "1h" }),
    messages: resolveCacheTarget(fileConfig.messages, {
      enabled: true,
      ttl: "5m",
    }),
  };
}

function resolveCacheTarget(value: any, defaults: CacheTarget): CacheTarget {
  if (value === undefined) return { ...defaults };
  if (typeof value === "boolean") return { enabled: value, ttl: defaults.ttl };
  return {
    enabled: value.enabled ?? defaults.enabled,
    ttl: value.ttl ?? defaults.ttl,
  };
}

// ── Extension entry point ───────────────────────────────────────────

/**
 * Pi extension: Claude models on Google Vertex AI
 *
 * Config: ~/.pi/agent/anthropic-vertex.json
 * Auth:   gcloud auth application-default login
 *
 * Usage:  pi --model anthropic-vertex/claude-opus-4-6
 */
export default function (pi: ExtensionAPI) {
  const config = loadConfig();
  if (!config) return;

  const cacheConfig = config.cache || CACHE_DISABLED;

  let _client: any = null;

  async function getClient() {
    if (_client) return _client;
    const { AnthropicVertex } = await import("@anthropic-ai/vertex-sdk");
    _client = new AnthropicVertex({
      region: config.region,
      projectId: config.projectId,
    });
    return _client;
  }

  pi.registerProvider("anthropic-vertex", {
    // baseUrl and apiKey are required by Pi's validation but unused —
    // the Vertex SDK handles its own endpoint URL and ADC auth.
    baseUrl: `https://${config.region}-aiplatform.googleapis.com`,
    apiKey: "<ADC>",
    // Custom API type so Pi routes through our streamSimple, not the built-in
    // anthropic handler. Trade-off: "xhigh" thinking won't auto-detect.
    // See https://github.com/badlogic/pi-mono/issues/2040
    api: "anthropic-vertex" as any,

    models: [
      {
        id: "claude-opus-4-7",
        name: "Claude Opus 4.7 (Vertex AI)",
        reasoning: true,
        input: ["text", "image"],
        cost: { input: 5.0, output: 25.0, cacheRead: 0.5, cacheWrite: 6.25 },
        contextWindow: 1000000,
        maxTokens: 128000,
      },
      {
        id: "claude-opus-4-6",
        name: "Claude Opus 4.6 (Vertex AI)",
        reasoning: true,
        input: ["text", "image"],
        cost: { input: 5.0, output: 25.0, cacheRead: 0.5, cacheWrite: 6.25 },
        contextWindow: 1000000,
        maxTokens: 128000,
      },
      {
        id: "claude-sonnet-4-6",
        name: "Claude Sonnet 4.6 (Vertex AI)",
        reasoning: true,
        input: ["text", "image"],
        cost: { input: 1.5, output: 7.5, cacheRead: 0.15, cacheWrite: 1.875 },
        contextWindow: 1000000,
        maxTokens: 64000,
      },
      {
        id: "claude-haiku-4-5-20251001",
        name: "Claude Haiku 4.5 (Vertex AI)",
        reasoning: true,
        input: ["text", "image"],
        cost: { input: 0.5, output: 2.5, cacheRead: 0.05, cacheWrite: 0.625 },
        contextWindow: 200000,
        maxTokens: 16384,
      },
    ],

    streamSimple(
      model: Model<any>,
      context: Context,
      options?: SimpleStreamOptions
    ): AssistantMessageEventStream {
      const stream = createAssistantMessageEventStream();

      (async () => {
        const output: AssistantMessage = {
          role: "assistant",
          content: [],
          api: model.api,
          provider: model.provider,
          model: model.id,
          usage: {
            input: 0,
            output: 0,
            cacheRead: 0,
            cacheWrite: 0,
            totalTokens: 0,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
          },
          stopReason: "stop",
          timestamp: Date.now(),
        };

        try {
          const client = await getClient();

          const request: Record<string, any> = {
            model: model.id,
            max_tokens: model.maxTokens,
            messages: toAnthropicMessages(context, cacheConfig, model),
          };

          const system = toAnthropicSystem(context, cacheConfig);
          if (system) request.system = system;

          const tools = toAnthropicTools(context, cacheConfig);
          if (tools.length > 0) request.tools = tools;

          // Map Pi's thinking level to Anthropic thinking config
          const reasoningLevel = (options as any)?.reasoning as
            | string
            | undefined;
          let thinkingEnabled = false;

          if (model.reasoning) {
            const thinkingFields = resolveThinkingPayload(
              config.thinking,
              reasoningLevel,
              model.maxTokens,
              model.id
            );
            if (thinkingFields) {
              Object.assign(request, thinkingFields);
              thinkingEnabled = true;
            }
          }

          // Temperature: only applied when thinking is off.
          // Anthropic rejects the temperature param when thinking is enabled.
          if (!thinkingEnabled && config.temperature !== null) {
            request.temperature = config.temperature;
          }

          stream.push({ type: "start", partial: output });

          // Use client.beta.messages.stream() to pass beta features.
          // The SDK extracts `betas` and sends them as the `anthropic-beta` HTTP header —
          // they are NOT sent in the request body (which Vertex rejects).
          const anthropicStream = client.beta.messages.stream(
            { ...request, betas: VERTEX_BETAS },
            { signal: options?.signal }
          );

          // ── Map Anthropic stream events → Pi stream events ──
          // Anthropic: content_block_start → content_block_delta → content_block_stop
          // Pi:        *_start             → *_delta             → *_end
          // Tool call arguments arrive as partial JSON chunks (input_json_delta)
          // and are accumulated in blockJsonBuffers until content_block_stop.
          const blockJsonBuffers: Map<number, string> = new Map();

          for await (const event of anthropicStream) {
            switch (event.type) {
              // ── Message start: capture initial usage (input tokens) ──
              case "message_start": {
                const ms = event as any;
                if (ms.message?.usage) {
                  output.usage.input = ms.message.usage.input_tokens || 0;
                  output.usage.cacheRead =
                    ms.message.usage.cache_read_input_tokens || 0;
                  output.usage.cacheWrite =
                    ms.message.usage.cache_creation_input_tokens || 0;
                }
                break;
              }

              // ── Block start: initialize content block in Pi format ──
              case "content_block_start": {
                const block = event.content_block;
                // Track Anthropic's block index for delta correlation.
                // Pi uses array position, so we store index temporarily.
                const idx = output.content.length;

                if (block.type === "text") {
                  output.content.push({
                    type: ContentType.text,
                    text: "",
                    _idx: event.index,
                  } as any);
                  stream.push({
                    type: StreamEvent.text_start,
                    contentIndex: idx,
                    partial: output,
                  } as any);
                } else if (block.type === "thinking") {
                  // Track thinkingSignature — Anthropic requires it on thinking blocks
                  // in multi-turn conversations (tamper-proof integrity check).
                  output.content.push({
                    type: ContentType.thinking,
                    thinking: "",
                    thinkingSignature: "",
                    _idx: event.index,
                  } as any);
                  stream.push({
                    type: StreamEvent.thinking_start,
                    contentIndex: idx,
                    partial: output,
                  } as any);
                } else if ((block as any).type === "redacted_thinking") {
                  // Redacted thinking: model chose to redact reasoning content.
                  // Store as thinking block with redacted flag for history.
                  output.content.push({
                    type: ContentType.thinking,
                    thinking: "[Reasoning redacted]",
                    thinkingSignature: (block as any).data,
                    redacted: true,
                    _idx: event.index,
                  } as any);
                  stream.push({
                    type: StreamEvent.thinking_start,
                    contentIndex: idx,
                    partial: output,
                  } as any);
                } else if (block.type === "tool_use") {
                  // Pi uses "toolCall" type with "arguments" field;
                  // Anthropic uses "tool_use" with "input" — mapped via ContentType
                  output.content.push({
                    type: ContentType.tool_use,
                    id: block.id,
                    name: block.name,
                    arguments: {},
                    _idx: event.index,
                  } as any);
                  blockJsonBuffers.set(event.index, "");
                  stream.push({
                    type: StreamEvent.toolcall_start,
                    contentIndex: idx,
                    partial: output,
                  } as any);
                }
                break;
              }

              // ── Block delta: accumulate streaming content ──
              case "content_block_delta": {
                // Find the Pi content block matching this Anthropic block index
                const idx = output.content.findIndex(
                  (b: any) => b._idx === event.index
                );
                if (idx < 0) break;
                const contentBlock = output.content[idx];

                const delta = event.delta as any;
                if (
                  delta.type === "text_delta" &&
                  contentBlock.type === ContentType.text
                ) {
                  (contentBlock as any).text += delta.text;
                  stream.push({
                    type: StreamEvent.text_delta,
                    contentIndex: idx,
                    delta: delta.text,
                    partial: output,
                  } as any);
                } else if (
                  delta.type === "thinking_delta" &&
                  contentBlock.type === ContentType.thinking
                ) {
                  (contentBlock as any).thinking += delta.thinking;
                  stream.push({
                    type: StreamEvent.thinking_delta,
                    contentIndex: idx,
                    delta: delta.thinking,
                    partial: output,
                  } as any);
                } else if (
                  delta.type === "signature_delta" &&
                  contentBlock.type === ContentType.thinking
                ) {
                  // Accumulate signature — no Pi event, just stored for multi-turn
                  (contentBlock as any).thinkingSignature += delta.signature;
                } else if (
                  delta.type === "input_json_delta" &&
                  contentBlock.type === ContentType.tool_use
                ) {
                  // Tool arguments arrive as partial JSON chunks — accumulate
                  const buf =
                    (blockJsonBuffers.get(event.index) || "") +
                    delta.partial_json;
                  blockJsonBuffers.set(event.index, buf);
                  stream.push({
                    type: StreamEvent.toolcall_delta,
                    contentIndex: idx,
                    delta: delta.partial_json,
                    partial: output,
                  } as any);
                }
                break;
              }

              // ── Block stop: finalize content block ──
              case "content_block_stop": {
                const idx = output.content.findIndex(
                  (b: any) => b._idx === event.index
                );
                if (idx < 0) break;
                const contentBlock = output.content[idx];

                // Clean up temporary index tracker
                delete (contentBlock as any)._idx;

                if (contentBlock.type === ContentType.text) {
                  // Pi's event contract requires content on end events
                  stream.push({
                    type: StreamEvent.text_end,
                    contentIndex: idx,
                    content: (contentBlock as any).text,
                    partial: output,
                  } as any);
                } else if (contentBlock.type === ContentType.thinking) {
                  stream.push({
                    type: StreamEvent.thinking_end,
                    contentIndex: idx,
                    content: (contentBlock as any).thinking,
                    partial: output,
                  } as any);
                } else if (contentBlock.type === ContentType.tool_use) {
                  // Parse accumulated JSON into final arguments object
                  const json = blockJsonBuffers.get(event.index) || "{}";
                  try {
                    (contentBlock as any).arguments = JSON.parse(json);
                  } catch {
                    (contentBlock as any).arguments = { _raw: json };
                  }
                  blockJsonBuffers.delete(event.index);
                  // toolcall_end triggers Pi's tool execution loop
                  stream.push({
                    type: StreamEvent.toolcall_end,
                    contentIndex: idx,
                    toolCall: contentBlock,
                    partial: output,
                  } as any);
                }
                break;
              }

              // ── Message delta: stop reason and output token count ──
              case "message_delta": {
                const md = event as any;
                // Map Anthropic stop reasons → Pi: "tool_use"→"toolUse", "end_turn"→"stop"
                const reason = md.delta?.stop_reason || "stop";
                output.stopReason = StopReason[reason] || reason;
                if (md.usage) {
                  output.usage.output = md.usage.output_tokens || 0;
                  // Preserve input tokens from message_start if not present here
                  if (md.usage.input_tokens != null) {
                    output.usage.input = md.usage.input_tokens;
                  }
                }
                break;
              }
            }
          }

          // ── Final usage: update from completed message if available ──
          // This supplements message_start/message_delta usage with final counts,
          // and ensures we have usage even if stream events were partial.
          const final = await anthropicStream.finalMessage();
          if (final.usage) {
            output.usage.input = final.usage.input_tokens || 0;
            output.usage.output = final.usage.output_tokens || 0;
            if ((final.usage as any).cache_read_input_tokens) {
              output.usage.cacheRead =
                (final.usage as any).cache_read_input_tokens;
            }
            if ((final.usage as any).cache_creation_input_tokens) {
              output.usage.cacheWrite =
                (final.usage as any).cache_creation_input_tokens;
            }
            // totalTokens includes all token types for consistency with Pi's built-in provider.
            // Note: input_tokens from Anthropic excludes cached tokens — cacheRead/cacheWrite
            // are reported separately and represent the cached portion of input.
            output.usage.totalTokens =
              output.usage.input +
              output.usage.output +
              output.usage.cacheRead +
              output.usage.cacheWrite;

            output.usage.cost = {
              input: (output.usage.input / 1_000_000) * model.cost.input,
              output: (output.usage.output / 1_000_000) * model.cost.output,
              cacheRead:
                (output.usage.cacheRead / 1_000_000) * model.cost.cacheRead,
              cacheWrite:
                (output.usage.cacheWrite / 1_000_000) * model.cost.cacheWrite,
              total: 0,
            };
            output.usage.cost.total =
              output.usage.cost.input +
              output.usage.cost.output +
              output.usage.cost.cacheRead +
              output.usage.cost.cacheWrite;
          }

          stream.push({
            type: "done",
            reason: output.stopReason,
            message: output,
          });
          stream.end();
        } catch (error: any) {
          output.stopReason = options?.signal?.aborted ? "aborted" : "error";
          output.errorMessage =
            error instanceof Error ? error.message : String(error);
          stream.push({
            type: "error",
            reason: output.stopReason,
            error: output,
          });
          stream.end();
        }
      })();

      return stream;
    },
  });
}

// ── Context → Anthropic format converters ────────────────────────────

function cacheMarker(ttl: "5m" | "1h") {
  return ttl === "1h"
    ? { type: "ephemeral" as const, ttl: "1h" as const }
    : { type: "ephemeral" as const };
}

// Replace unpaired Unicode surrogates with replacement char to prevent API errors.
// Can appear from binary data, broken clipboard, or cross-platform encoding.
function sanitizeSurrogates(text: string): string {
  return text.replace(/[\uD800-\uDFFF]/g, "\uFFFD");
}

// Anthropic requires tool call IDs to be alphanumeric + underscore/hyphen, max 64 chars.
// Pi or other providers may produce IDs that don't conform.
function normalizeToolCallId(id: string): string {
  return id.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64);
}

function toAnthropicMessages(
  context: Context,
  cache: CacheConfig,
  model: Model<any>
): any[] {
  // Preprocess: drop errored/aborted turns, insert synthetic tool results
  // for orphaned tool calls. This mirrors Pi's transformMessages() which
  // isn't exported from @mariozechner/pi-ai.
  const preprocessed = preprocessMessages(
    (context as any).messages || [],
    model
  );
  const messages: any[] = [];

  // Convert Pi messages to Anthropic format:
  // - toolResult messages: batch consecutive ones into a single "user" message
  // - assistant messages: convert toolCall→tool_use, preserve thinking with signatures
  // - user messages: pass through with surrogate sanitization + image filtering
  for (let i = 0; i < preprocessed.length; i++) {
    const msg = preprocessed[i];

    if (msg.role === "toolResult") {
      // Batch consecutive toolResult messages into one user message.
      // Anthropic requires all tool_results for a tool_use turn in a single message.
      const toolResults: any[] = [];
      let j = i;
      while (j < preprocessed.length && preprocessed[j].role === "toolResult") {
        const tr = preprocessed[j];
        toolResults.push({
          type: "tool_result",
          tool_use_id: normalizeToolCallId(tr.toolCallId),
          content: convertToolResultContent(tr.content),
          is_error: tr.isError,
        });
        j++;
      }
      i = j - 1; // skip batched messages
      messages.push({ role: "user", content: toolResults });
    } else if (msg.role === "assistant") {
      const blocks = toAnthropicAssistantBlocks(msg.content);
      // Skip messages that become empty after filtering
      if (blocks.length > 0) {
        messages.push({ role: "assistant", content: blocks });
      }
    } else {
      const content = sanitizeUserContent(msg.content, model);
      messages.push({ role: msg.role, content });
    }
  }

  // Anthropic requires every tool_use to have a matching tool_result in the
  // next message. Pi's conversation compaction or partial exchanges can leave
  // orphaned tool_use blocks — strip them to avoid 400 errors.
  sanitizeToolUseResults(messages);

  // Prompt caching: mark the second-to-last user turn for caching.
  // Everything before it is stable conversation history that won't change.
  // Only worth caching if conversation is long enough (>4 messages).
  if (cache.messages.enabled && messages.length >= 4) {
    const targetIdx = findSecondToLastUserIndex(messages);
    if (targetIdx >= 0) {
      messages[targetIdx] = injectCacheControl(
        messages[targetIdx],
        cache.messages.ttl
      );
    }
  }

  return messages;
}

// Convert Pi assistant content blocks to Anthropic format:
// - toolCall → tool_use (arguments → input, normalize ID)
// - thinking: redacted → redacted_thinking, with signature → preserve, without → text
// - text: sanitize surrogates, skip empty
function toAnthropicAssistantBlocks(content: any): any[] {
  if (!Array.isArray(content)) return content ? [content] : [];

  const blocks: any[] = [];
  for (const block of content) {
    if (block.type === ContentType.tool_use || block.type === "toolCall") {
      blocks.push({
        type: "tool_use",
        id: normalizeToolCallId(block.id),
        name: block.name,
        input: block.arguments || block.input || {},
      });
    } else if (block.type === "thinking") {
      // Redacted thinking: model chose to redact reasoning content
      if (block.redacted) {
        blocks.push({
          type: "redacted_thinking",
          data: block.thinkingSignature,
        });
        continue;
      }
      // Skip empty thinking blocks
      if (!block.thinking || block.thinking.trim().length === 0) continue;

      // Anthropic requires signatures on thinking blocks in multi-turn.
      // If signature is present and non-empty, preserve it.
      // If missing, convert to text to avoid 400 errors.
      if (
        block.thinkingSignature &&
        block.thinkingSignature.trim().length > 0
      ) {
        blocks.push({
          type: "thinking",
          thinking: sanitizeSurrogates(block.thinking),
          signature: block.thinkingSignature,
        });
      } else {
        blocks.push({
          type: "text",
          text: sanitizeSurrogates(block.thinking),
        });
      }
    } else if (block.type === "text") {
      // Skip empty text blocks
      if (!block.text || block.text.trim().length === 0) continue;
      blocks.push({ type: "text", text: sanitizeSurrogates(block.text) });
    } else {
      blocks.push(block);
    }
  }
  return blocks;
}

// Convert tool result content to Anthropic format
function convertToolResultContent(content: any): any {
  if (typeof content === "string") return sanitizeSurrogates(content);
  if (Array.isArray(content)) {
    return content.map((block: any) => {
      if (block.type === "text") {
        return { type: "text", text: sanitizeSurrogates(block.text || "") };
      }
      return block;
    });
  }
  return "";
}

// Sanitize user message content:
// - String: sanitize surrogates
// - Block array: sanitize text, filter images for non-vision models,
//   add placeholder if image-only message
function sanitizeUserContent(content: any, model: Model<any>): any {
  if (typeof content === "string") return sanitizeSurrogates(content);
  if (!Array.isArray(content)) return content;

  const supportsImages = model.input?.includes("image");
  let blocks = content.map((block: any) => {
    if (block.type === "text" && typeof block.text === "string") {
      return { ...block, text: sanitizeSurrogates(block.text) };
    }
    return block;
  });

  // Filter images if model doesn't support vision
  if (!supportsImages) {
    blocks = blocks.filter((b: any) => b.type !== "image");
  }

  // Add placeholder text if message is image-only (Anthropic requires text)
  const hasText = blocks.some((b: any) => b.type === "text");
  if (!hasText && blocks.length > 0) {
    blocks.unshift({ type: "text", text: "(see attached image)" });
  }

  // Filter empty text blocks
  blocks = blocks.filter(
    (b: any) => b.type !== "text" || (b.text && b.text.trim().length > 0)
  );

  return blocks;
}

function toAnthropicSystem(
  context: Context,
  cache: CacheConfig
): any[] | undefined {
  // Pi's Context type uses "systemPrompt", not "system"
  const system = (context as any).systemPrompt;
  if (!system) return undefined;

  const blocks: any[] =
    typeof system === "string"
      ? [{ type: "text", text: sanitizeSurrogates(system) }]
      : [...system];

  if (cache.system.enabled && blocks.length > 0) {
    const last = blocks[blocks.length - 1];
    blocks[blocks.length - 1] = {
      ...last,
      cache_control: cacheMarker(cache.system.ttl),
    };
  }

  return blocks;
}

function toAnthropicTools(context: Context, cache: CacheConfig): any[] {
  const rawTools = (context as any).tools || [];

  // Normalize to Anthropic tool format — Pi may use different field names
  const tools = rawTools.map((tool: any) => ({
    name: tool.name,
    description: tool.description,
    input_schema:
      tool.input_schema ||
      tool.inputSchema ||
      tool.parameters ||
      { type: "object", properties: {} },
  }));

  // Cache breakpoint on the last tool — caches the entire tools array.
  // Pi's built-in provider doesn't cache tools, but it's a free optimization.
  if (cache.tools.enabled && tools.length > 0) {
    const last = tools[tools.length - 1];
    tools[tools.length - 1] = {
      ...last,
      cache_control: cacheMarker(cache.tools.ttl),
    };
  }

  return tools;
}

// ── Message preprocessing ────────────────────────────────────────────
// Mirrors essential parts of Pi's transformMessages() (not exported from pi-ai).
// - Drops errored/aborted assistant messages (incomplete turns that cause API errors)
// - Drops cross-model redacted thinking (only valid for the same model)
// - Inserts synthetic tool results for orphaned tool calls (preserves thinking signatures)

function preprocessMessages(messages: any[], model: Model<any>): any[] {
  const result: any[] = [];
  let pendingToolCalls: any[] = [];
  const existingToolResultIds = new Set<string>();

  const insertSyntheticToolResults = () => {
    for (const tc of pendingToolCalls) {
      if (!existingToolResultIds.has(tc.id)) {
        result.push({
          role: "toolResult",
          toolCallId: tc.id,
          toolName: tc.name,
          content: [{ type: "text", text: "No result provided" }],
          isError: true,
          timestamp: Date.now(),
        });
      }
    }
    pendingToolCalls = [];
    existingToolResultIds.clear();
  };

  for (const msg of messages) {
    if (msg.role === "assistant") {
      // Insert synthetic results for any orphaned tool calls from previous turn
      insertSyntheticToolResults();

      // Skip errored/aborted assistant messages — these are incomplete turns
      // that may have partial content (reasoning without message, incomplete tool calls).
      // Replaying them causes API errors.
      if (msg.stopReason === "error" || msg.stopReason === "aborted") continue;

      // Check if this assistant message came from the same model
      const isSameModel =
        msg.provider === model.provider &&
        msg.api === model.api &&
        msg.model === model.id;

      // Filter cross-model redacted thinking (only valid for same model)
      if (!isSameModel && Array.isArray(msg.content)) {
        const filtered = msg.content.filter(
          (b: any) => !(b.type === "thinking" && b.redacted)
        );
        if (filtered.length !== msg.content.length) {
          msg = { ...msg, content: filtered };
        }
      }

      // Track tool calls for orphan detection
      const toolCalls = (msg.content || []).filter(
        (b: any) => b.type === "toolCall"
      );
      if (toolCalls.length > 0) {
        pendingToolCalls = toolCalls;
        existingToolResultIds.clear();
      }

      result.push(msg);
    } else if (msg.role === "toolResult") {
      existingToolResultIds.add(msg.toolCallId);
      result.push(msg);
    } else if (msg.role === "user") {
      // User message interrupts tool flow — insert synthetic results for orphaned calls
      insertSyntheticToolResults();
      result.push(msg);
    } else {
      result.push(msg);
    }
  }

  return result;
}

// ── Helpers ──────────────────────────────────────────────────────────

function findSecondToLastUserIndex(messages: any[]): number {
  let userCount = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user") {
      userCount++;
      if (userCount === 2) return i;
    }
  }
  return -1;
}

// Anthropic requires every tool_use to have a matching tool_result in the
// next message. Pi's conversation compaction can leave orphaned tool_use
// blocks — strip them to avoid 400 errors.
function sanitizeToolUseResults(messages: any[]): void {
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    if (msg.role !== "assistant" || !Array.isArray(msg.content)) continue;

    const toolUseIds = new Set<string>();
    for (const block of msg.content) {
      if (block.type === "tool_use" && block.id) toolUseIds.add(block.id);
    }
    if (toolUseIds.size === 0) continue;

    const next = messages[i + 1];
    if (next && Array.isArray(next.content)) {
      for (const block of next.content) {
        if (block.type === "tool_result" && block.tool_use_id) {
          toolUseIds.delete(block.tool_use_id);
        }
      }
    }

    if (toolUseIds.size > 0) {
      msg.content = msg.content.filter(
        (block: any) => block.type !== "tool_use" || !toolUseIds.has(block.id)
      );
    }
  }
}

function injectCacheControl(message: any, ttl: "5m" | "1h"): any {
  const content = message.content;
  const marker = cacheMarker(ttl);

  if (typeof content === "string") {
    return {
      ...message,
      content: [{ type: "text", text: content, cache_control: marker }],
    };
  }

  if (Array.isArray(content) && content.length > 0) {
    const blocks = [...content];
    const last = blocks[blocks.length - 1];
    blocks[blocks.length - 1] = { ...last, cache_control: marker };
    return { ...message, content: blocks };
  }

  return message;
}
