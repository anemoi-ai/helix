import { HelixUnsupportedFeatureError } from "./error-types.js";
import type { ChatCompletionRequest } from "./types.js";

const UNSUPPORTED = new Set([
  "audio", "modalities", "prediction", "parallel_tool_calls",
  "service_tier", "store", "metadata", "user", "function_call",
  "functions", "presence_penalty", "frequency_penalty",
  "best_of", "echo", "suffix",
]);

const SUPPORTED = new Set([
  "messages", "model", "temperature", "top_p", "max_tokens",
  "stop", "seed", "n", "stream", "stream_options", "tools",
  "tool_choice", "response_format", "logprobs", "top_logprobs",
  "logit_bias", "max_completion_tokens",
]);

export function buildRequest(params: Record<string, unknown>): ChatCompletionRequest {
  for (const key of Object.keys(params)) {
    if (UNSUPPORTED.has(key)) {
      throw new HelixUnsupportedFeatureError(
        `Helix does not support the '${key}' parameter. See helix-llm docs.`,
        -11, "{}"
      );
    }
  }
  const req: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(params)) {
    if (SUPPORTED.has(k) && v !== undefined && v !== null) {
      const dest = k === "max_completion_tokens" ? "max_tokens" : k;
      req[dest] = v;
    }
  }
  return req as ChatCompletionRequest;
}
