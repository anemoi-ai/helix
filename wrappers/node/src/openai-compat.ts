/**
 * OpenAI SDK compatibility shim for @helix/node.
 *
 * Usage:
 *   import { createOpenAICompat } from "@helix/node/openai-compat";
 *   const client = createOpenAICompat({ modelPath: "qwen.gguf" });
 *   const resp = await client.chat.completions.create({ model: "m", messages: [...] });
 *
 * Does NOT depend on the `openai` npm package at runtime.
 */

import { Helix, Session, Model, HelixUnsupportedFeatureError, HelixError } from "./index.js";
import type { ChatCompletionRequest } from "./index.js";

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

function buildRequest(params: Record<string, unknown>): ChatCompletionRequest {
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

class Completions {
  constructor(private session: Session) {}

  create(params: Record<string, unknown>): any {
    const { stream, ...rest } = params;
    const req = buildRequest(rest);
    if (stream) {
      return this.session.streamChatCompletions(req);
    }
    return Promise.resolve(this.session.chatCompletions(req));
  }
}

class Chat {
  completions: Completions;
  constructor(session: Session) {
    this.completions = new Completions(session);
  }
}

export interface OpenAICompatOptions {
  modelPath: string;
  alias?: string;
  runtimeOptions?: { log_level?: number };
}

export class OpenAICompatClient {
  chat: Chat;
  private runtime: Helix;
  private model: Model;
  private session: Session;

  constructor(options: OpenAICompatOptions) {
    this.runtime = new Helix(options.runtimeOptions);
    this.model = this.runtime.loadModel(options.modelPath, { alias: options.alias });
    this.session = this.model.session();
    this.chat = new Chat(this.session);
  }

  destroy(): void {
    this.session.cancel();
  }
}

export function createOpenAICompat(options: OpenAICompatOptions): OpenAICompatClient {
  return new OpenAICompatClient(options);
}
