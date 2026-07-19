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
 *  HELIX_ABI_VERSION is encoded as (major × 65536 + minor):
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
#define HELIX_ABI_VERSION_MINOR 0
#define HELIX_ABI_VERSION_PATCH 0

/* Encoded as (major × 65536 + minor). Patch is implicit in the build. */
#define HELIX_ABI_VERSION \
    ((HELIX_ABI_VERSION_MAJOR << 16) | HELIX_ABI_VERSION_MINOR)

/* Compile-time check — verifies the header is the 1.0 stable release. */
#if defined(__cplusplus)
static_assert(HELIX_ABI_VERSION == 0x00010000,
    "helix.h: unexpected ABI version — expected 1.0.0 stable (0x00010000)");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(HELIX_ABI_VERSION == 0x00010000,
    "helix.h: unexpected ABI version — expected 1.0.0 stable (0x00010000)");
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
typedef char _helix_status_size_check[sizeof(helix_status_t) == 4 ? 1 : -1];
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
    /* -12 … -98 reserved for future 1.x additions */
    HELIX_E_INTERNAL               = -99,
};

/* Thread-local. Returns a NUL-terminated, OpenAI-shaped JSON error object:
 *   {"error":{"message":..,"type":..,"param":..,"code":..}}
 * Valid until the next helix_* call on this thread.
 * Returns "{}" (never NULL) when no error is set. */
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
 * Use helix_runtime_describe_copy() for a malloc'd copy you can outlive it. */
HELIX_API const char* helix_runtime_describe(helix_runtime_t* runtime);

/* Returns a malloc'd copy of the runtime description. Caller must helix_free(). */
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
 * to keep the UI responsive should call this on a background thread. */
HELIX_API helix_status_t helix_model_load(helix_runtime_t* runtime,
                                           const char* model_json,
                                           helix_model_t** out_model);

/* Releases the model. The caller owns the model's lifetime: it must NOT be
 * released while any session created on it still exists or has an in-flight
 * helix_chat_completions[_stream] call on another thread (doing so races the
 * model teardown against the session's use of it). Destroy all sessions on a
 * model before releasing the model. */
HELIX_API void helix_model_release(helix_model_t* model);

/* JSON describing the loaded model (alias, n_ctx_train, etc.).
 * WARNING: returns a borrowed pointer — valid only while the Model lives.
 * Use helix_model_describe_copy() for a malloc'd copy you can outlive it. */
HELIX_API const char* helix_model_describe(helix_model_t* model);

/* Returns a malloc'd copy of the model description. Caller must helix_free(). */
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

/* Synchronous form. On success, `*out_response_json` receives a malloc'd,
 * NUL-terminated JSON string; caller must free with `helix_free`. */
HELIX_API helix_status_t helix_chat_completions(helix_session_t* session,
                                                 const char* request_json,
                                                 char** out_response_json);

/* Streaming form. Returns once the stream is complete or cancelled. */
HELIX_API helix_status_t helix_chat_completions_stream(helix_session_t* session,
                                                        const char* request_json,
                                                        helix_stream_cb on_chunk,
                                                        void* user_data);

/* Cooperative cancellation — safe to call from any thread. */
HELIX_API void helix_session_cancel(helix_session_t* session);

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
 * stderr via fprintf().  In latency-sensitive production environments this
 * blocking I/O occurs on the decode thread and adds jitter.  Register a
 * callback (or pass HELIX_LOG_OFF as min_level) to suppress the default
 * fprintf path.
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
