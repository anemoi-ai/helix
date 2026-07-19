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

export interface StreamOptions { include_usage?: boolean }

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

export interface Delta { role?: string; content?: string; tool_calls?: ToolCallDelta[]; reasoning_content?: string }
export interface ChunkChoice { index: number; delta: Delta; finish_reason: string | null }

export interface ChatCompletionChunk {
  id: string; object: string; created: number; model: string;
  choices: ChunkChoice[]; usage?: Usage;
}

export interface ModelOptions {
  alias?: string; n_gpu_layers?: number; n_ctx?: number;
  mmproj_path?: string; reasoning_format?: string;
  /** LoRA adapters loaded with the model (1.6). `name` defaults to the file
   *  stem; `scale` is the load-time default a session applies. */
  lora?: { path: string; name?: string; scale?: number }[];
}

export interface SpeculativeOptions {
  /** "none" (default) or "draft-mtp" */
  type?: "none" | "draft-mtp";
  /** Path to a separate MTP draft model GGUF (required for Gemma-4) */
  model_path?: string;
  n_max?: number; n_min?: number; p_min?: number;
  backend_sampling?: boolean;
  cache_type_k?: string; cache_type_v?: string;
}

export interface SessionOptions {
  n_ctx?: number; n_batch?: number; n_threads?: number; n_threads_batch?: number;
  prefix_cache?: boolean; warmup?: boolean;
  /** Context-overflow recovery (1.6): drop oldest history / evict oldest KV
   *  instead of failing at the context wall. Responses report
   *  helix.context_shifted / helix.evicted_tokens when history was lost. */
  context_shift?: boolean;
  /** LoRA adapter activation (1.6). Absent: all model adapters at load-time
   *  scales. Array: exactly the named adapters ([] = none). */
  lora?: { name: string; scale?: number }[];
  seed?: number; stream_coalesce_ms?: number; swa_full?: boolean;
  embedding?: boolean; pooling?: "none" | "mean" | "cls" | "last" | "rank";
  /** Main-context KV cache types (1.4), e.g. "q8_0". A quantized
   *  cache_type_v requires flash attention. Default "f16". */
  cache_type_k?: string; cache_type_v?: string;
  speculative?: SpeculativeOptions;
  [key: string]: unknown;
}

export interface RuntimeOptions { log_level?: number }
