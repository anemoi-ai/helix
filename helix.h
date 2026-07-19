/* helix.h — public C ABI, C99-compatible
 * This is the entire public surface of libhelix. No other header is needed. */

#ifndef HELIX_H
#define HELIX_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------
 *  Visibility
 * ------------------------------------------------------------------ */

#if defined(_WIN32) || defined(__CYGWIN__)
#  ifdef HELIX_BUILD_SHARED
#    define HELIX_API __declspec(dllexport)
#  elif defined(HELIX_SHARED)
#    define HELIX_API __declspec(dllimport)
#  else
#    define HELIX_API
#  endif
#else
#  if __GNUC__ >= 4
#    define HELIX_API __attribute__((visibility("default")))
#  else
#    define HELIX_API
#  endif
#endif

/* ------------------------------------------------------------------
 *  Versioning
 *
 *  HELIX_ABI_VERSION is encoded as (major × 65536 + minor × 256 + patch):
 *    - Major bump = backward-incompatible change (forbidden during 1.x).
 *    - Minor bump = backward-compatible additions (new symbol, status code,
 *      or JSON field). The 18-month 1.x ABI commitment allows only minor bumps.
 *    - Patch releases do not change this macro.
 *
 *  Applications should assert at startup:
 *    assert(helix_abi_version() >> 16 == HELIX_ABI_VERSION >> 16 &&
 *           helix_abi_version() >= HELIX_ABI_VERSION);
 * ------------------------------------------------------------------ */

#define HELIX_ABI_VERSION_MAJOR 1
#define HELIX_ABI_VERSION_MINOR 6
#define HELIX_ABI_VERSION_PATCH 0

/* Encoded as (major × 65536 + minor × 256 + patch). */
#define HELIX_ABI_VERSION \
    ((HELIX_ABI_VERSION_MAJOR << 16) | (HELIX_ABI_VERSION_MINOR << 8) | \
     HELIX_ABI_VERSION_PATCH)

/* Compile-time check — verifies the header is the 1.6 stable release.
 * (1.6 adds no symbols — the bump covers the context_shift session option
 * and the "helix" response extension object.) */
#if defined(__cplusplus)
static_assert(HELIX_ABI_VERSION == 0x00010600,
    "helix.h: unexpected ABI version — expected 1.6.0 stable (0x00010600)");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(HELIX_ABI_VERSION == 0x00010600,
    "helix.h: unexpected ABI version — expected 1.6.0 stable (0x00010600)");
#endif

/* Returns the ABI version the loaded library was built against.
 * Compare against HELIX_ABI_VERSION to detect runtime mismatches. */
HELIX_API uint32_t helix_abi_version(void);

/* Returns "1.0.0+llama.cpp-b<build>-<sha>" — human-readable.
 * The returned pointer is borrowed and valid for the lifetime of the process;
 * do not free it. (Copy it if you need to outlive the library being unloaded.) */
HELIX_API const char* helix_version_string(void);

/* ------------------------------------------------------------------
 *  Error codes — non-zero is always failure
 *
 *  ABI stability contract for this enum:
 *    - Existing values are never removed or renumbered across 1.x releases.
 *    - New values may be added with more-negative integers in future 1.x
 *      minor releases. Callers must default-case to handle unknown codes.
 *    - HELIX_E_INTERNAL (-99) is always the catch-all for unexpected errors.
 * ------------------------------------------------------------------ */

typedef int32_t helix_status_t;

/* Compile-time size check — also validated in helix_abi.cpp */
#if defined(__cplusplus)
static_assert(sizeof(helix_status_t) == 4, "helix_status_t must be 4 bytes");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(helix_status_t) == 4, "helix_status_t must be 4 bytes");
#else
typedef char helix_status_size_check_[sizeof(helix_status_t) == 4 ? 1 : -1];
#endif

enum {
    HELIX_OK                       =  0,
    HELIX_E_INVALID_ARG            = -1,
    HELIX_E_INVALID_JSON           = -2,
    HELIX_E_VALIDATION             = -3,  /* OpenAI-shaped 400-class */
    HELIX_E_MODEL_NOT_FOUND        = -4,
    HELIX_E_MODEL_LOAD_FAILED      = -5,
    HELIX_E_OOM                    = -6,
    HELIX_E_VRAM_EXHAUSTED         = -7,
    HELIX_E_CONTEXT_FULL           = -8,
    HELIX_E_CANCELLED              = -9,
    HELIX_E_BACKEND                = -10, /* ggml/cuda/metal failure */
    HELIX_E_UNSUPPORTED_FEATURE    = -11,
    HELIX_E_STATE_MISMATCH         = -12, /* added in 1.5: session state file
                                             from a different model, n_ctx, or
                                             incompatible library version */
    /* -13 … -98 reserved for future 1.x additions */
    HELIX_E_INTERNAL               = -99,
};

/* Thread-local. Returns a NUL-terminated, OpenAI-shaped JSON error object:
 *   {"error":{"message":..,"type":..,"param":..,"code":..}}
 * Valid until the next helix_* call on this thread.
 * Returns "{}" (never NULL) when no error is set.
 *
 * Note: the returned pointer references thread-local storage owned by libhelix.
 * It remains valid across helix_runtime_destroy / helix_model_release calls,
 * but NOT across helix_* API calls on the same thread (each call may overwrite
 * the buffer). Copy the string immediately if you need it to outlive the next
 * API call. */
HELIX_API const char* helix_last_error_json(void);

/* ------------------------------------------------------------------
 *  Runtime — process-wide lifecycle
 * ------------------------------------------------------------------ */

typedef struct helix_runtime helix_runtime_t;

/* Initialise the global ggml/llama backend, probe hardware, build the
 * default device plan. `options_json` is optional (may be NULL).
 *
 * Thread safety: concurrent calls are serialised internally, but the
 * underlying GGML backend is process-wide — create exactly ONE Runtime
 * per process, before spawning any threads that call other helix functions. */
HELIX_API helix_status_t helix_runtime_create(const char* options_json,
                                               helix_runtime_t** out_runtime);

HELIX_API void helix_runtime_destroy(helix_runtime_t* runtime);

/* Returns the JSON description of what Helix observed about this machine.
 * WARNING: returns a borrowed pointer — valid only while the Runtime lives.
 * Use helix_runtime_describe_copy() for a malloc'd copy you can outlive it.
 *
 * NULL-safe: passing NULL returns "{}". */
HELIX_API const char* helix_runtime_describe(helix_runtime_t* runtime);

/* Returns a malloc'd copy of the runtime description. Caller must helix_free().
 * May return NULL on malloc failure. NULL-safe: passing NULL returns a copy of
 * "{}" (or NULL on malloc failure). */
HELIX_API char* helix_runtime_describe_copy(helix_runtime_t* runtime);

/* ------------------------------------------------------------------
 *  Models — heavy, refcounted, shareable across sessions
 * ------------------------------------------------------------------ */

typedef struct helix_model helix_model_t;

/* `model_json` must not be NULL and must contain at least "model_path".
 * See README for full schema. Unlike `options_json` / `session_json`, there
 * is no default — a missing model path is always an error.
 *
 * This call is synchronous and may take seconds to minutes for large models.
 * There is no built-in timeout or cancellation mechanism; callers that need
 * to keep the UI responsive should call this on a background thread, or use
 * helix_model_load_ex (added in 1.4) for progress reporting + cancellation. */
HELIX_API helix_status_t helix_model_load(helix_runtime_t* runtime,
                                           const char* model_json,
                                           helix_model_t** out_model);

/* Progress callback for helix_model_load_ex. `progress` is in [0.0, 1.0]
 * (monotonically non-decreasing; 1.0 is reported before the load returns).
 * Return 0 to continue loading; return non-zero to cancel — the load then
 * fails with HELIX_E_CANCELLED and *out_model is set to NULL.
 *
 * The callback is invoked synchronously on the thread that called
 * helix_model_load_ex, during the tensor-loading phase of the model file.
 * It MUST NOT call back into libhelix (added in 1.4 — export node HELIX_1.4). */
typedef int (*helix_load_progress_cb)(void* user_data, float progress);

/* As helix_model_load, with an optional progress/cancellation callback.
 * `cb` may be NULL, making this call exactly equivalent to helix_model_load.
 * `user_data` is passed through to every `cb` invocation.
 *
 * Added in 1.4 — export node HELIX_1.4. */
HELIX_API helix_status_t helix_model_load_ex(helix_runtime_t* runtime,
                                             const char* model_json,
                                             helix_load_progress_cb cb,
                                             void* user_data,
                                             helix_model_t** out_model);

/* Releases the model. The caller owns the model's lifetime: it must NOT be
 * released while any session created on it still exists or has an in-flight
 * helix_chat_completions[_stream] call on another thread (doing so races the
 * model teardown against the session's use of it). Destroy all sessions on a
 * model before releasing the model.
 *
 * NULL-safe: passing NULL is a no-op.
 * Note: this function is also available as helix_model_destroy() for naming
 * consistency with helix_runtime_destroy / helix_session_destroy. */
HELIX_API void helix_model_release(helix_model_t* model);
#define helix_model_destroy helix_model_release

/* JSON describing the loaded model (alias, n_ctx_train, etc.).
 * WARNING: returns a borrowed pointer — valid only while the Model lives.
 * Use helix_model_describe_copy() for a malloc'd copy you can outlive it.
 *
 * NULL-safe: passing NULL returns "{}". */
HELIX_API const char* helix_model_describe(helix_model_t* model);

/* Returns a malloc'd copy of the model description. Caller must helix_free().
 * May return NULL on malloc failure. NULL-safe: passing NULL returns a copy of
 * "{}" (or NULL on malloc failure). */
HELIX_API char* helix_model_describe_copy(helix_model_t* model);

/* ------------------------------------------------------------------
 *  Sessions — own a KV cache, one inflight request at a time
 * ------------------------------------------------------------------ */

typedef struct helix_session helix_session_t;

/* `session_json` is optional (may be NULL).
 *
 * Concurrency: multiple sessions on the same model may be created and used
 * from different threads simultaneously. On CPU builds each session is fully
 * independent. On CUDA/ROCm builds all decode calls on sessions that share a
 * model are serialised through a per-model lock — concurrent calls will not
 * deadlock but they will be queued, not parallelised. */
HELIX_API helix_status_t helix_session_create(helix_model_t* model,
                                               const char* session_json,
                                               helix_session_t** out_session);

HELIX_API void helix_session_destroy(helix_session_t* session);

/* NULL-safe: passing NULL is a no-op. */

/* Returns a malloc'd JSON description of the session's effective context and
 * speculative state. Caller must helix_free() the result.
 * NULL-safe: passing NULL returns a copy of "{}" (or NULL on malloc failure).
 *
 * Added in 1.2 — export node HELIX_1.2. Example output (flash_attn and
 * cache_type_k/v added in 1.4):
 *   {
 *     "n_ctx": 4096, "n_batch": 512, "n_ubatch": 512,
 *     "n_threads": 8, "embedding": false, "swa_full": true,
 *     "flash_attn": true, "cache_type_k": "f16", "cache_type_v": "f16",
 *     "speculative": {
 *       "type": "draft-mtp", "enabled": true,
 *       "n_max": 3, "n_min": 0, "p_min": 0.0,
 *       "draft_model": true
 *     }
 *   }
 */
HELIX_API char* helix_session_describe(helix_session_t* session);

/* ------------------------------------------------------------------
 *  Chat Completions — the endpoint
 * ------------------------------------------------------------------ */

/* Streaming callback. `chunk_json` is one OpenAI chat.completion.chunk
 * object, NUL-terminated. Returning non-zero cancels generation.
 * The final invocation passes chunk_json == NULL to signal stream done.
 *
 * `chunk_json` is valid only for the duration of the callback invocation.
 * The callback is invoked synchronously on the thread that called
 * helix_chat_completions_stream.
 *
 * Callback restrictions — the callback MUST NOT:
 *   - Call helix_session_destroy() on the owning session (use-after-free:
 *     the session lock is held for the duration of the call).
 *   - Call helix_chat_completions[_stream]() on the same session (deadlock:
 *     the session is not re-entrant).
 * Calling helix_session_cancel() on the owning session IS safe. */
typedef int (*helix_stream_cb)(void* user_data, const char* chunk_json);

/* The request body follows the OpenAI Chat Completions schema. In addition,
 * Helix accepts these extension fields (all optional):
 *   "top_k"            integer >= 0; top-K filtering (default 40; 0 disables)
 *   "min_p"            [0, 1]; minimum-probability filtering (default 0.05)
 *   "repeat_penalty"   > 0; multiplicative repetition penalty over the last
 *                      64 tokens (default 1.0 = disabled, OpenAI-neutral)
 *   "reasoning_budget" integer >= -1; cap on <think> tokens for reasoning
 *                      models. -1/unset = unlimited, 0 = close the thinking
 *                      block immediately, N > 0 = force the closing tag after
 *                      N thinking tokens (see docs/USER_GUIDE.md §4.4). */

/* Synchronous form. On success, `*out_response_json` receives a malloc'd,
 * NUL-terminated JSON string; caller must free with `helix_free`.
 * On failure, `*out_response_json` is set to NULL (if the pointer itself is
 * non-NULL). Inspect helix_last_error_json() for the error details. */
HELIX_API helix_status_t helix_chat_completions(helix_session_t* session,
                                                 const char* request_json,
                                                 char** out_response_json);

/* Streaming form. Returns once the stream is complete or cancelled.
 * `on_chunk` must not be NULL; doing so produces HELIX_E_INVALID_ARG. */
HELIX_API helix_status_t helix_chat_completions_stream(helix_session_t* session,
                                                        const char* request_json,
                                                        helix_stream_cb on_chunk,
                                                        void* user_data);

/* Cooperative cancellation — safe to call from any thread.
 * Sets a flag checked at the next decode step of any in-flight call on the
 * session. NULL-safe: passing NULL is a no-op.
 *
 * What the in-flight call returns depends on where cancellation lands:
 *
 *   - Text chat completions (prefill or generation): HELIX_OK with a partial
 *     response; finish_reason is reported as "stop" (indistinguishable from
 *     natural completion in the response body).
 *   - Multimodal chat completions cancelled during media prefill:
 *     HELIX_E_CANCELLED — there is no meaningful partial response before any
 *     token has been generated. Once generation has started, behaves like
 *     text chat (HELIX_OK + partial response).
 *   - Embeddings (cancelled at an internal batch boundary):
 *     HELIX_E_CANCELLED — vectors are all-or-nothing; a partial data[] array
 *     would silently misalign with the input order.
 *
 * A cancel-and-retry caller should therefore treat both HELIX_OK-with-partial
 * and HELIX_E_CANCELLED as "request did not complete" after calling this. */
HELIX_API void helix_session_cancel(helix_session_t* session);

/* ------------------------------------------------------------------
 *  Session state persistence (added in 1.5 — export node HELIX_1.5)
 *
 *  Persist a chat session's KV cache across process restarts so the next
 *  request's prefix-cache match skips re-prefilling the saved conversation.
 *
 *  SECURITY: state files are TRUSTED input — restore decodes them directly
 *  into engine buffers. Never restore a file from an untrusted source (the
 *  same posture as GGUF model files themselves).
 * ------------------------------------------------------------------ */

/* Serialize the session's KV state and token history to `path` (the file is
 * written to `path` + ".tmp" first, then atomically renamed). The file embeds
 * the model's identity fingerprint, n_ctx, and the token history needed for
 * prefix validation after restore. Typical size: tens to hundreds of MB
 * (proportional to KV usage).
 *
 * Requires a chat session with the prefix cache enabled — restored state is
 * only reachable through prefix matching, so embedding sessions, speculative
 * (MTP) sessions, and sessions created with {"prefix_cache": false} are
 * rejected with HELIX_E_UNSUPPORTED_FEATURE. Waits for any in-flight request
 * to finish (one-inference-at-a-time rule), then holds the session for the
 * duration of the save. */
HELIX_API helix_status_t helix_session_save(helix_session_t* session,
                                            const char* path);

/* Restore state saved by helix_session_save into a session created on the
 * same model with the same n_ctx. Fails with HELIX_E_STATE_MISMATCH (-12)
 * if the file was produced by a different model, a different n_ctx, or an
 * incompatible library version — or if the file is truncated/corrupt.
 *
 * On any failure the session is left empty and usable (as if never
 * restored). On success the session behaves as if it had just processed the
 * saved conversation: the next request with a matching prompt prefix reuses
 * the restored KV instead of re-prefilling. Same session-kind restrictions
 * as helix_session_save. */
HELIX_API helix_status_t helix_session_restore(helix_session_t* session,
                                               const char* path);

/* ------------------------------------------------------------------
 *  Hidden-state extraction (Phase 3)
 *
 *  Prefill-only encode: fills `messages_json` into the session KV cache
 *  and writes the penultimate-token hidden state into `out_hidden_state`.
 *
 *  `out_hidden_state` must be pre-allocated with `helix_model_hidden_dim`
 *  floats.  Returns HELIX_OK on success.
 * ------------------------------------------------------------------ */
HELIX_API helix_status_t helix_session_encode(
    helix_session_t*  session,
    const char*       messages_json,
    float*            out_hidden_state);

/* Returns the hidden dimension of the loaded model. */
HELIX_API uint32_t helix_model_hidden_dim(helix_model_t* model);

/* ------------------------------------------------------------------
 *  Embeddings (added in 1.1 — export node HELIX_1.1)
 * ------------------------------------------------------------------ */

/**
 * Compute embeddings, following the OpenAI Embeddings API schema.
 *
 * The session MUST have been created with {"embedding": true}; calling this
 * on a chat session returns HELIX_E_UNSUPPORTED_FEATURE (and vice versa:
 * chat completions on an embedding session are rejected the same way).
 *
 * request_json:
 *   {
 *     "model": "<alias>",                     // must equal the session's model alias
 *     "input": "text" | ["text", ...],        // 1..2048 non-empty inputs
 *     "encoding_format": "float" | "base64"   // optional, default "float"
 *   }
 *
 * On HELIX_OK, *out_response_json receives a malloc-allocated, NUL-terminated
 * JSON string the caller must release with helix_free():
 *   {
 *     "object": "list",
 *     "data": [ { "object": "embedding", "index": 0,
 *                 "embedding": [ ... ] | "<base64>" }, ... ],
 *     "model": "<alias>",
 *     "usage": { "prompt_tokens": N, "total_tokens": N }
 *   }
 *
 * Vectors are L2-normalized (unit length), matching OpenAI semantics.
 * data[] order always matches input order. One inference at a time per
 * session; helix_session_cancel() aborts at the next internal batch boundary
 * with HELIX_E_CANCELLED.
 */
HELIX_API helix_status_t helix_embeddings(
    helix_session_t* session,
    const char*      request_json,
    char**           out_response_json);

/**
 * Embedding output dimension of the loaded model.
 *
 * Returns the pooled output dimension for models usable with
 * helix_embeddings. NOTE: this is not helix_model_hidden_dim() — the pooled
 * output dimension diverges from the hidden size for some architectures.
 *
 * Returns 0 when the model cannot serve embeddings on this endpoint:
 * encoder-decoder architectures, or classifier/reranker heads
 * (pooling type "rank"). Callable on any loaded model; does not require an
 * embedding session. Intended for dimension negotiation before index
 * creation (e.g. HNSW dimension locking). NULL-safe: returns 0.
 */
HELIX_API uint32_t helix_model_embedding_dim(helix_model_t* model);

/* ------------------------------------------------------------------
 *  Rerank (added in 1.5 — export node HELIX_1.5)
 * ------------------------------------------------------------------ */

/**
 * Rerank documents against a query with a reranker-head model
 * (e.g. bge-reranker, Qwen3-Reranker). Follows the de-facto /v1/rerank
 * shape (Cohere/Jina, also implemented by llama-server).
 *
 * The session MUST have been created with {"embedding": true,
 * "pooling": "rank"} on a model with a reranker head; other sessions are
 * rejected with HELIX_E_UNSUPPORTED_FEATURE (as is helix_embeddings on a
 * rerank session — rank heads produce scores, not vectors).
 *
 * request_json:
 *   {
 *     "model": "<alias>",                  // must equal the session's alias
 *     "query": "...",
 *     "documents": ["...", ...],           // 1..1024 non-empty strings
 *     "top_n": 10                          // optional, default: all
 *   }
 *
 * On HELIX_OK, *out_response_json receives a malloc'd, NUL-terminated JSON
 * string the caller must release with helix_free():
 *   {
 *     "object": "list", "model": "<alias>",
 *     "results": [ {"index": 3, "relevance_score": 4.2}, ... ],
 *     "usage": {"prompt_tokens": N, "total_tokens": N}
 *   }
 *
 * results[] is sorted by relevance_score descending (ties keep input order);
 * "index" refers to the position in the request's documents[]. Scores are
 * the model's raw head outputs (logits, may be negative), matching
 * llama-server — apply a sigmoid if you need [0, 1] probabilities.
 *
 * Each query+document pair must fit the session batch: an oversized pair
 * fails with HELIX_E_CONTEXT_FULL (param "documents"). One inference at a
 * time per session; helix_session_cancel() aborts at the next internal
 * batch boundary with HELIX_E_CANCELLED.
 */
HELIX_API helix_status_t helix_rerank(helix_session_t* session,
                                      const char* request_json,
                                      char** out_response_json);

/* ------------------------------------------------------------------
 *  Tokenizer utilities (added in 1.4 — export node HELIX_1.4)
 * ------------------------------------------------------------------ */

/* Count the tokens the given chat request would occupy after chat-template
 * rendering — i.e. the "prompt_tokens" the usage object of a real
 * helix_chat_completions call on this session would report.
 *
 * `request_json` accepts the same body as helix_chat_completions (model,
 * messages, tools, response_format, ...) and is validated the same way;
 * generation parameters (temperature, max_tokens, stream, ...) are accepted
 * but have no effect on the count.
 *
 * Pure query: does not touch the session KV cache, does not affect the
 * prefix cache, and is safe to call from any thread — including while a
 * chat completion is in flight on the same session.
 *
 * Limitations (v1): requests containing image/audio content parts return
 * HELIX_E_UNSUPPORTED_FEATURE (media patch-token counting may be added in a
 * later release). */
HELIX_API helix_status_t helix_count_tokens(helix_session_t* session,
                                            const char* request_json,
                                            uint32_t* out_token_count);

/* Raw tokenizer access. Tokenizes `text` with the model's own vocabulary and
 * writes a malloc'd JSON array of token ids ("[1,2,3]"; "[]" for empty text)
 * to *out_tokens_json; the caller must release it with helix_free().
 *
 * `add_special`  non-zero: add the model's special prefix/suffix tokens
 *                (e.g. BOS), exactly as a chat prompt would receive them.
 * `parse_special` non-zero: parse special/control token text (e.g.
 *                "<|im_start|>") into their token ids rather than tokenizing
 *                them as plain text.
 *
 * Pure query: no session required, safe to call from any thread. */
HELIX_API helix_status_t helix_tokenize(helix_model_t* model,
                                        const char* text,
                                        int add_special,
                                        int parse_special,
                                        char** out_tokens_json);

/* Free strings returned by helix_chat_completions(). Never free strings
 * returned by helix_*_describe() or helix_last_error_json().
 *
 * Always use helix_free() rather than the caller's own free() — the string
 * was allocated by libhelix's heap.  On Windows builds that use a static CRT
 * or a mismatched runtime, calling the wrong free() causes heap corruption. */
HELIX_API void helix_free(char* ptr);

/* ------------------------------------------------------------------
 *  Logging
 *
 *  ABI stability contract for helix_log_level_t:
 *    - Existing values are never removed or renumbered across 1.x releases.
 *    - Levels are always non-negative; new levels may be added with larger
 *      integers in future 1.x minor releases. Log callbacks must default-case
 *      unknown levels.
 * ------------------------------------------------------------------ */

typedef enum {
    HELIX_LOG_OFF   = 0,
    HELIX_LOG_ERROR = 1,
    HELIX_LOG_WARN  = 2,
    HELIX_LOG_INFO  = 3,
    HELIX_LOG_DEBUG = 4,
    HELIX_LOG_TRACE = 5
    /* Future levels will be appended here with larger values */
} helix_log_level_t;

#if defined(__cplusplus)
static_assert(sizeof(helix_log_level_t) == sizeof(int32_t),
    "helix_log_level_t must be 4 bytes across the C ABI boundary");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(helix_log_level_t) == sizeof(int32_t),
    "helix_log_level_t must be 4 bytes across the C ABI boundary");
#endif

typedef void (*helix_log_cb)(void* user_data,
                              helix_log_level_t level,
                              const char* msg);

/* Register a log callback. Thread-safe: may be called at any time.
 *
 * The callback/userdata/min_level triple is updated atomically, but a log
 * emission already in flight on another thread at the moment of the update may
 * still observe the previous callback and level for that one message. After
 * this call returns, subsequent emissions use the new values.
 *
 * Without a registered callback, helix writes ERROR and WARN messages to
 * stderr via fprintf(), and INFO and above to stdout.  In latency-sensitive
 * production environments this blocking I/O occurs on the decode thread and
 * adds jitter.  Register a callback (or pass HELIX_LOG_OFF as min_level) to
 * suppress the default fprintf path.
 *
 * Passing cb=NULL restores the default fprintf() behaviour described above.
 *
 * Volume note: at HELIX_LOG_DEBUG and especially HELIX_LOG_TRACE, the
 * underlying inference engine's internal logs are forwarded too — a single
 * model load at TRACE can emit tens of thousands of lines. Use INFO or lower
 * for production. */
HELIX_API void helix_set_log_callback(helix_log_cb cb,
                                       void* user_data,
                                       helix_log_level_t min_level);

#ifdef __cplusplus
}
#endif

#endif /* HELIX_H */
