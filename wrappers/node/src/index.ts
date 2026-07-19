/**
 * @helix/node — TypeScript API for libhelix.
 */

import { createRequire } from "module";
import * as path from "path";
import * as fs from "fs";

// ── Native binding loader ─────────────────────────────────────────────────────

function loadNative(): any {
  // 1. Prebuilds (production)
  const prebuildsDir = path.join(__dirname, "..", "prebuilds");
  if (fs.existsSync(prebuildsDir)) {
    const require_ = createRequire(import.meta?.url ?? __filename);
    try {
      return require_(path.join(prebuildsDir, `${process.platform}-${process.arch}`, "helix_node.node"));
    } catch {}
  }
  // 2. node-gyp build (development)
  const buildPath = path.join(__dirname, "..", "build", "Release", "helix_node.node");
  if (fs.existsSync(buildPath)) {
    const require_ = createRequire(import.meta?.url ?? __filename);
    return require_(buildPath);
  }
  throw new Error(
    "@helix/node: could not find native addon. Run `npm run build:native` to compile it."
  );
}

let _native: any = null;
function native(): any {
  if (!_native) _native = loadNative();
  return _native;
}

// ── Error types ───────────────────────────────────────────────────────────────

export class HelixError extends Error {
  readonly helixStatus: number;
  readonly helixErrorJson: string;
  readonly param?: string;
  readonly code?: string;

  constructor(message: string, status: number, errorJson: string) {
    super(message);
    this.name = "HelixError";
    this.helixStatus = status;
    this.helixErrorJson = errorJson;
    try {
      const parsed = JSON.parse(errorJson);
      this.param = parsed?.error?.param ?? undefined;
      this.code = parsed?.error?.code ?? undefined;
    } catch {}
  }
}

export class HelixValidationError extends HelixError { override name = "HelixValidationError"; }
export class HelixModelNotFoundError extends HelixError { override name = "HelixModelNotFoundError"; }
export class HelixModelLoadFailedError extends HelixError { override name = "HelixModelLoadFailedError"; }
export class HelixOomError extends HelixError { override name = "HelixOomError"; }
export class HelixCancelledError extends HelixError { override name = "HelixCancelledError"; }
export class HelixContextFullError extends HelixError { override name = "HelixContextFullError"; }
export class HelixUnsupportedFeatureError extends HelixError { override name = "HelixUnsupportedFeatureError"; }

function wrapNativeError(err: any): HelixError {
  const status: number = err?.helixStatus ?? 0;
  const json: string = err?.helixErrorJson ?? "{}";
  let msg: string = err?.message ?? String(err);
  try { msg = JSON.parse(json)?.error?.message ?? msg; } catch {}

  const map: Record<number, new (...a: any[]) => HelixError> = {
    [-3]: HelixValidationError,
    [-4]: HelixModelNotFoundError,
    [-5]: HelixModelLoadFailedError,
    [-6]: HelixOomError,
    [-8]: HelixContextFullError,
    [-9]: HelixCancelledError,
    [-11]: HelixUnsupportedFeatureError,
  };
  const Cls = map[status] ?? HelixError;
  return new Cls(msg, status, json);
}

// ── Request / response types ─────────────────────────────────────────────────

export interface Message {
  role: "system" | "user" | "assistant" | "tool";
  content?: string | ContentPart[];
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  reasoning_content?: string;
}

export interface ContentPart {
  type: "text" | "image_url";
  text?: string;
  image_url?: { url: string; detail?: string };
}

export interface Tool {
  type: "function";
  function: { name: string; description: string; parameters: Record<string, unknown> };
}

export interface ResponseFormat {
  type: "text" | "json_object" | "json_schema";
  json_schema?: Record<string, unknown>;
}

export interface StreamOptions {
  include_usage?: boolean;
}

export interface ChatCompletionRequest {
  model: string;
  messages: Message[];
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  stop?: string | string[];
  seed?: number;
  n?: number;
  tools?: Tool[];
  tool_choice?: string | Record<string, unknown>;
  response_format?: ResponseFormat;
  logprobs?: boolean;
  top_logprobs?: number;
  logit_bias?: Record<string, number>;
  stream_options?: StreamOptions;
}

export interface ToolCallFunction { name: string; arguments: string }
export interface ToolCall { id: string; type: string; function: ToolCallFunction }
export interface ToolCallDelta { index?: number; id?: string; type?: string; function?: { name?: string; arguments?: string } }

export interface LogprobContent {
  token: string; logprob: number; bytes?: number[];
  top_logprobs: Array<{ token: string; logprob: number; bytes?: number[] }>;
}

export interface Usage {
  prompt_tokens: number; completion_tokens: number; total_tokens: number;
  completion_tokens_details?: { reasoning_tokens?: number };
}

export interface ResponseMessage {
  role: string; content: string | null;
  tool_calls?: ToolCall[]; reasoning_content?: string;
}

export interface Choice {
  index: number; message: ResponseMessage; finish_reason: string | null;
  logprobs?: { content: LogprobContent[] };
}

export interface ChatCompletion {
  id: string; object: string; created: number; model: string;
  choices: Choice[]; usage: Usage;
}

export interface Delta {
  role?: string; content?: string; tool_calls?: ToolCallDelta[]; reasoning_content?: string;
}

export interface ChunkChoice {
  index: number; delta: Delta; finish_reason: string | null;
}

export interface ChatCompletionChunk {
  id: string; object: string; created: number; model: string;
  choices: ChunkChoice[]; usage?: Usage;
}

// ── Session ───────────────────────────────────────────────────────────────────

export class Session {
  constructor(private readonly _ptr: unknown) {}

  chatCompletions(request: ChatCompletionRequest): ChatCompletion {
    try {
      const json = native().chatCompletions(this._ptr, JSON.stringify(request));
      return JSON.parse(json) as ChatCompletion;
    } catch (e) { throw wrapNativeError(e); }
  }

  streamChatCompletions(
    request: ChatCompletionRequest,
    options?: { signal?: AbortSignal }
  ): ReadableStream<ChatCompletionChunk> {
    const ptr = this._ptr;
    const signal = options?.signal;

    return new ReadableStream<ChatCompletionChunk>({
      start(controller) {
        const abortHandler = () => {
          try { native().cancelSession(ptr); } catch {}
          controller.close();
        };
        signal?.addEventListener("abort", abortHandler, { once: true });

        native().streamChatCompletions(
          ptr,
          JSON.stringify(request),
          (chunkJson: string | null, isDone: boolean) => {
            if (isDone) {
              signal?.removeEventListener("abort", abortHandler);
              controller.close();
              return;
            }
            try {
              controller.enqueue(JSON.parse(chunkJson!) as ChatCompletionChunk);
            } catch (e) {
              controller.error(e);
            }
          }
        );
      },
      cancel() {
        try { native().cancelSession(ptr); } catch {}
      },
    });
  }

  cancel(): void {
    try { native().cancelSession(this._ptr); } catch {}
  }

  [Symbol.dispose](): void { this.cancel(); }
}

// ── Model ─────────────────────────────────────────────────────────────────────

export interface ModelOptions {
  alias?: string;
  n_gpu_layers?: number;
  n_ctx?: number;
  mmproj_path?: string;
  reasoning_format?: string;
}

export class Model {
  constructor(private readonly _ptr: unknown) {}

  session(options?: Record<string, unknown>): Session {
    try {
      const ptr = native().createSession(this._ptr, JSON.stringify(options ?? {}));
      return new Session(ptr);
    } catch (e) { throw wrapNativeError(e); }
  }

  [Symbol.dispose](): void {}
}

// ── RuntimeOptions ────────────────────────────────────────────────────────────

export interface RuntimeOptions {
  log_level?: number;
}

// ── Helix (runtime) ───────────────────────────────────────────────────────────

export class Helix {
  private readonly _ptr: unknown;

  constructor(options?: RuntimeOptions) {
    try {
      this._ptr = native().createRuntime(JSON.stringify(options ?? { log_level: 2 }));
    } catch (e) { throw wrapNativeError(e); }
  }

  loadModel(path: string, options?: ModelOptions): Model {
    const modelJson = JSON.stringify({ model_path: path, ...(options ?? {}) });
    try {
      const ptr = native().loadModel(this._ptr, modelJson);
      return new Model(ptr);
    } catch (e) { throw wrapNativeError(e); }
  }

  abiVersion(): number { return native().abiVersion(); }
  versionString(): string { return native().versionString(); }

  [Symbol.dispose](): void {}
}
