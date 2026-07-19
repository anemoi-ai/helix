# C API Reference

The entire public surface is declared in `helix.h`. No other header is needed.

## Versioning

```c
uint32_t helix_abi_version(void);
const char* helix_version_string(void);
```

`helix_abi_version()` returns `(major << 16) | (minor << 8) | patch` for the
library that is loaded at runtime. Compare against `HELIX_ABI_VERSION` (the
value from the header you compiled against) to detect version mismatches.

`helix_version_string()` returns a human-readable string such as
`"1.1.0+llama.cpp-abc123def456"`. The string is valid for the process lifetime.

## Error handling

```c
const char* helix_last_error_json(void);
```

Thread-local. Returns a NUL-terminated, OpenAI-shaped JSON error object:

```json
{"error":{"message":"...","type":"...","param":"...","code":"..."}}
```

Valid until the next `helix_*` call on this thread. Returns `"{}"` (never
NULL) when no error is set.

## Runtime lifecycle

```c
helix_status_t helix_runtime_create(const char* options_json,
                                     helix_runtime_t** out_runtime);
void           helix_runtime_destroy(helix_runtime_t* runtime);
const char*    helix_runtime_describe(helix_runtime_t* runtime);
```

Create exactly **one** runtime per process, before spawning threads that call
other helix functions. `options_json` may be NULL.

`helix_runtime_describe()` returns a JSON object describing the probed
hardware: CPU topology, ISA tier, memory bandwidth, GPU accelerators, and the
auto-tuned parameter plan. The string is owned by the runtime; do not free it.

## Model loading

```c
helix_status_t helix_model_load(helix_runtime_t* runtime,
                                 const char* model_json,
                                 helix_model_t** out_model);
void           helix_model_release(helix_model_t* model);
const char*    helix_model_describe(helix_model_t* model);
```

`model_json` must contain at least `"model_path"`. This call is synchronous
and may take seconds for large models — call from a background thread if
needed. Multiple models may be loaded simultaneously.

## Sessions

```c
helix_status_t helix_session_create(helix_model_t* model,
                                     const char* session_json,
                                     helix_session_t** out_session);
void           helix_session_destroy(helix_session_t* session);
void           helix_session_cancel(helix_session_t* session);
```

Each session owns a KV cache and processes one request at a time. Multiple
sessions on the same model may run from different threads simultaneously.

`helix_session_cancel()` is safe to call from any thread, including from
within a streaming callback.

## Chat completions

```c
helix_status_t helix_chat_completions(
    helix_session_t* session,
    const char* request_json,
    char** out_response_json);

helix_status_t helix_chat_completions_stream(
    helix_session_t* session,
    const char* request_json,
    helix_stream_cb on_chunk,
    void* user_data);
```

`request_json` follows the [OpenAI Chat Completions schema](json-schemas.md).
On success, `helix_chat_completions` writes a malloc'd JSON string to
`*out_response_json`; the caller must free it with `helix_free`.

The streaming callback receives one `chat.completion.chunk` object per
invocation. Returning non-zero from the callback cancels generation. The
final invocation passes `chunk_json == NULL` to signal stream end.

**Callback restrictions:** do not call `helix_session_destroy` or
`helix_chat_completions[_stream]` on the owning session from inside the
callback.

## Embeddings *(1.1, export node `HELIX_1.1`)*

```c
helix_status_t helix_embeddings(
    helix_session_t* session,
    const char* request_json,
    char** out_response_json);

uint32_t helix_model_embedding_dim(helix_model_t* model);
```

`helix_embeddings` follows the OpenAI Embeddings schema (`model`, `input` as
one string or 1..2048 strings, optional `encoding_format: "float" | "base64"`).
The session must have been created with `{"embedding": true}`; chat and
embedding sessions reject each other's requests with
`HELIX_E_UNSUPPORTED_FEATURE`. Vectors are L2-normalized and returned in input
order; the response string is freed with `helix_free`.

`helix_model_embedding_dim` returns the pooled output dimension (the length
of every returned vector), or 0 when the model cannot serve embeddings
(encoder-decoder architectures, reranker/classifier heads). NULL-safe pure
query; needs no session.

## Memory management

```c
void helix_free(char* ptr);
```

Free strings returned by `helix_chat_completions` and `helix_embeddings`.
Never use this on strings
returned by `helix_*_describe()` or `helix_last_error_json()` — those are
owned by the library. Always use `helix_free` rather than the caller's own
`free()`; on Windows with mismatched CRTs, mixing heaps causes corruption.

## Logging

```c
void helix_set_log_callback(helix_log_cb cb,
                              void* user_data,
                              helix_log_level_t min_level);
```

Register a log callback. Thread-safe; may be called at any time. Without a
callback, Helix writes ERROR and WARN messages to stderr. Pass
`HELIX_LOG_OFF` as `min_level` to silence all output.

Log levels: `HELIX_LOG_OFF=0`, `HELIX_LOG_ERROR=1`, `HELIX_LOG_WARN=2`,
`HELIX_LOG_INFO=3`, `HELIX_LOG_DEBUG=4`, `HELIX_LOG_TRACE=5`.

## Status codes

| Code | Value | Meaning |
|------|-------|---------|
| `HELIX_OK` | 0 | Success |
| `HELIX_E_INVALID_ARG` | -1 | NULL or out-of-range argument |
| `HELIX_E_INVALID_JSON` | -2 | Malformed JSON input |
| `HELIX_E_VALIDATION` | -3 | Semantically invalid request (OpenAI 400-class) |
| `HELIX_E_MODEL_NOT_FOUND` | -4 | Model path does not exist |
| `HELIX_E_MODEL_LOAD_FAILED` | -5 | Model file corrupt or unsupported format |
| `HELIX_E_OOM` | -6 | Out of system memory |
| `HELIX_E_VRAM_EXHAUSTED` | -7 | Insufficient GPU VRAM |
| `HELIX_E_CONTEXT_FULL` | -8 | Conversation exceeds model context window |
| `HELIX_E_CANCELLED` | -9 | Request cancelled via `helix_session_cancel` |
| `HELIX_E_BACKEND` | -10 | GPU backend (ggml/cuda/metal) failure |
| `HELIX_E_UNSUPPORTED_FEATURE` | -11 | Feature not available on this backend |
| `HELIX_E_INTERNAL` | -99 | Unexpected internal error |

New codes may be added in future 1.x minor releases (more-negative integers).
Always handle an unknown status code in your default case.
