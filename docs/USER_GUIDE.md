# Helix User Guide

**Embeddable, OpenAI-Isomorphic Local LLM Inference Library**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Building](#2-building)
3. [API Reference](#3-api-reference)
   - [Lifecycle](#31-lifecycle)
   - [Runtime](#32-runtime)
   - [Models](#33-models)
   - [Sessions](#34-sessions)
   - [Chat Completions](#35-chat-completions)
   - [Embeddings](#36-embeddings)
   - [Error Handling](#37-error-handling)
   - [Logging](#38-logging)
   - [Tokenizer Utilities](#39-tokenizer-utilities)
   - [Rerank](#310-rerank)
4. [Configuration](#4-configuration)
   - [Runtime Options](#41-runtime-options)
   - [Model Options](#42-model-options)
   - [Session Options](#43-session-options)
   - [Request Options](#44-request-options)
5. [Integration Patterns](#5-integration-patterns)
   - [C Integration](#51-c-integration)
   - [C++ Integration](#52-c-integration)
   - [Streaming](#53-streaming)
   - [Tool Calling](#54-tool-calling)
   - [Structured Outputs](#55-structured-outputs)
   - [Multimodal Input](#56-multimodal-input)
   - [Reasoning Extraction](#57-reasoning-extraction)
   - [Embeddings](#58-embeddings)
6. [Hardware Auto-Tuning](#6-hardware-auto-tuning)
7. [GPU Configuration](#7-gpu-configuration)
8. [Testing](#8-testing)
9. [Troubleshooting](#9-troubleshooting)
10. [Migration Guide: OpenAI → Helix](#10-migration-guide-openai--helix)

---

## 1. Overview

Helix is an embeddable C/C++ library that makes any GGUF model speak the **OpenAI Chat Completions protocol** over a tiny C ABI. The mental model is straightforward:

```
Your Application                  libhelix
╔══════════════════════╗     ╔═══════════════════════════════════════╗
║  helix_runtime_create║────→║ Probe hardware, init GGML backend   ║
║  helix_model_load    ║────→║ Load GGUF, select GPU layers         ║
║  helix_session_create║────→║ Create KV cache, auto-tune params    ║
║  helix_chat_completions║──→║ Render template → tokenize → decode  ║
║  helix_session_destroy║────→║ Free KV cache                       ║
║  helix_model_release ║────→║ Unload model                         ║
║  helix_runtime_destroy║───→║ Shutdown backends                    ║
╚══════════════════════╝     ╚═══════════════════════════════════════╝
```

There are exactly **four opaque handle types** (`helix_runtime_t`, `helix_model_t`, `helix_session_t`, and the stateless callback types) and **18 exported functions**. Everything is JSON-in, JSON-out.

---

## 2. Building

### 2.1 Requirements

| Dependency | Minimum | Notes |
|------------|---------|-------|
| C++ compiler | C++17 | GCC 11+, Clang 14+, MSVC 2022+ |
| C compiler | C11 | For the public header |
| CMake | 3.22 | — |
| llama.cpp | — | Vendored, pinned at `third_party/llama.cpp.commit` |
| nlohmann/json | — | Vendored single header |
| Google Test | 1.15.2 | Fetched by CMake at build time (tests only) |

### 2.2 Build Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `HELIX_BACKEND` | `cpu`, `cuda`, `metal`, `vulkan`, `rocm`, `omni` | `cpu` | GPU backend |
| `HELIX_BUILD_TOOLS` | `ON`, `OFF` | `ON` | Build `helix-doctor` and `helix-shim-server` |
| `HELIX_BUILD_TESTS` | `ON`, `OFF` | `ON` | Build test suite |

### 2.3 Build Commands

```bash
# CPU backend — quickest build, no GPU required
cmake -B build -DHELIX_BACKEND=cpu
cmake --build build -j$(nproc)

# CUDA backend
cmake -B build-cuda -DHELIX_BACKEND=cuda \
    -DCMAKE_CUDA_ARCHITECTURES="80;89;90"  # optional: target specific archs
cmake --build build-cuda -j$(nproc)

# Metal backend (macOS only)
cmake -B build-metal -DHELIX_BACKEND=metal
cmake --build build-metal -j$(nproc)

# Vulkan backend
cmake -B build-vulkan -DHELIX_BACKEND=vulkan
cmake --build build-vulkan -j$(nproc)

# ROCm backend
cmake -B build-rocm -DHELIX_BACKEND=rocm
cmake --build build-rocm -j$(nproc)
```

**Build outputs:**

```
build/
├── libhelix.so            # Shared library (libhelix.so.0 → libhelix.so.0.1.0)
├── libhelix.a             # Static library
├── bin/
│   ├── helix-doctor       # Diagnostic CLI tool
│   └── helix-shim-server  # HTTP shim server for integration testing
├── tests/
│   ├── helix_unit_tests
│   └── helix_integration_tests
├── helix.pc               # pkg-config file
└── compile_commands.json
```

### 2.4 Installing

```bash
cmake --install build --prefix /usr/local
# Installs: lib/libhelix.so, lib/libhelix.a, include/helix.h, lib/pkgconfig/helix.pc
```

### 2.5 Consuming with CMake

```cmake
find_package(helix REQUIRED)
target_link_libraries(my_app PRIVATE helix::helix)
```

### 2.6 Consuming with pkg-config

```bash
pkg-config --cflags --libs helix
```

---

## 3. API Reference

### 3.1 Lifecycle

The Helix object lifecycle is strictly hierarchical:

```
helix_runtime_t          (process-wide, singleton)
  └─ helix_model_t       (heavy, refcounted concept, shareable)
       └─ helix_session_t (lightweight, one KV cache per session)
```

- **Runtime**: Initialised once per process. Probes hardware, initialises GGML backends.
- **Model**: Loaded from a GGUF file. Can be shared across multiple sessions.
- **Session**: Owns a `llama_context` (KV cache). One inference at a time. Call `helix_session_cancel()` from any thread to abort.

### 3.2 Runtime

#### `helix_abi_version`

```c
uint32_t helix_abi_version(void);
```

Returns the ABI version encoded as `(major << 16) | (minor << 8) | patch`. The current version is `0x00010600` (1.6.0).

#### `helix_version_string`

```c
const char* helix_version_string(void);
```

Returns a human-readable version string: `"0.1.0+llama.cpp-b<build>-<sha>"`.

#### `helix_runtime_create`

```c
helix_status_t helix_runtime_create(
    const char* options_json,
    helix_runtime_t** out_runtime
);
```

Initialises the GGML backend, probes hardware, and builds the default device plan. `options_json` may be NULL, in which case all defaults apply.

**Thread safety:** Must be called from a single thread before any other helix function. The returned handle is safe to pass to other functions concurrently, but `helix_runtime_create` itself is not reentrant.

#### `helix_runtime_destroy`

```c
void helix_runtime_destroy(helix_runtime_t* runtime);
```

Shuts down backends and frees all resources. Safe to call with NULL. Must be called after all models and sessions have been destroyed.

#### `helix_runtime_describe`

```c
const char* helix_runtime_describe(helix_runtime_t* runtime);
```

Returns a JSON object describing the host machine. The returned string is owned by the library and must **not** be freed.

**Example output:**
```json
{
  "phase": 5,
  "backends": ["cpu", "cuda"],
  "device_preference": "auto",
  "cpu": {
    "vendor": "GenuineIntel",
    "model_name": "13th Gen Intel Core i9-13900K",
    "logical_cores": 32,
    "physical_cores": 16,
    "performance_cores": 8,
    "efficiency_cores": 16,
    "isa_tier": "avx2",
    "memory_bandwidth": "ddr5",
    "numa": false
  },
  "ram": {
    "total_bytes": 34359738368,
    "available_bytes": 25769803776
  },
  "accelerators": [
    {
      "device_index": 0,
      "device_name": "NVIDIA GeForce RTX 4090",
      "backend": "cuda",
      "total_vram_bytes": 25769803776,
      "free_vram_bytes": 21474836480,
      "compute_capability": "8.9"
    }
  ],
  "auto_tune": {
    "n_threads": 8,
    "n_threads_batch": 16,
    "n_batch": 1024,
    "n_ubatch": 256,
    "flash_attn": true
  }
}
```

### 3.3 Models

#### `helix_model_load`

```c
helix_status_t helix_model_load(
    helix_runtime_t* runtime,
    const char* model_json,
    helix_model_t** out_model
);
```

Loads a GGUF model file. `model_json` must at minimum contain a `"model_path"` field. See [§4.2 Model Options](#42-model-options) for the full schema.

Loading proceeds in two phases:
1. A lightweight **vocab-only** pass reads model metadata (layer count, context length).
2. The hardware profiler's GPU-layer fit algorithm determines how many layers to offload.
3. The **full model load** applies the selected GPU layer count.

#### `helix_model_load_ex` *(added in 1.4)*

```c
typedef int (*helix_load_progress_cb)(void* user_data, float progress);

helix_status_t helix_model_load_ex(
    helix_runtime_t* runtime,
    const char* model_json,
    helix_load_progress_cb cb,
    void* user_data,
    helix_model_t** out_model
);
```

As `helix_model_load`, with an optional progress/cancellation callback (`cb` may be NULL, making the call exactly equivalent). During the tensor-loading phase the callback receives `progress` values in `[0.0, 1.0]` (monotonically non-decreasing; a final `1.0` is always reported). Return `0` to continue; return non-zero to cancel — the load then fails with `HELIX_E_CANCELLED` and `*out_model` is NULL.

The callback runs synchronously on the loading thread, so a UI should post the value to its main thread rather than block:

```c
static int on_progress(void* user_data, float progress) {
    app_post_progress((App*)user_data, progress);   /* e.g. 43% -> dialog */
    return ((App*)user_data)->user_hit_cancel ? 1 : 0;
}

helix_model_t* model = NULL;
helix_status_t st = helix_model_load_ex(rt, opts_json, on_progress, app, &model);
if (st == HELIX_E_CANCELLED) { /* user aborted — runtime remains usable */ }
```

#### `helix_model_release`

```c
void helix_model_release(helix_model_t* model);
```

Frees the loaded model. Safe to call with NULL. All sessions using this model must be destroyed first.

#### `helix_model_describe`

```c
const char* helix_model_describe(helix_model_t* model);
```

Returns a JSON object describing the loaded model. The returned string is library-owned.

**Example output:**
```json
{
  "alias": "my-model",
  "model_path": "/models/qwen.gguf",
  "n_ctx_train": 32768,
  "n_vocab": 151936,
  "n_params": 4940329984,
  "n_layers": 64,
  "use_mmap": true,
  "use_mlock": false,
  "n_gpu_layers_requested": "auto",
  "n_gpu_layers_actual": 64,
  "phase": 7
}
```

### 3.4 Sessions

#### `helix_session_create`

```c
helix_status_t helix_session_create(
    helix_model_t* model,
    const char* session_json,
    helix_session_t** out_session
);
```

Creates an inference session bound to a loaded model. The session owns a `llama_context` (KV cache). `session_json` may be NULL; see [§4.3 Session Options](#43-session-options).

**Hardware auto-tuning** runs during session creation: thread counts, batch sizes, and flash attention are selected based on the host profile and model properties.

#### `helix_session_destroy`

```c
void helix_session_destroy(helix_session_t* session);
```

Destroys a session and frees its KV cache. Safe to call with NULL.

#### `helix_session_cancel`

```c
void helix_session_cancel(helix_session_t* session);
```

Cooperately cancels any in-flight `helix_chat_completions` or `helix_chat_completions_stream` call on this session. Safe to call from any thread. Cancellation is detected at the start of each decode step and within the prefill loop.

After cancellation, the session can be reused for a new request.

#### `helix_session_save` / `helix_session_restore` *(added in 1.5)*

```c
helix_status_t helix_session_save(helix_session_t* session, const char* path);
helix_status_t helix_session_restore(helix_session_t* session, const char* path);
```

Persist a chat session's KV cache and token history across process restarts. After a successful restore, the session behaves as if it had just processed the saved conversation: the next request with a matching prompt prefix reuses the restored KV through the prefix cache instead of re-paying the full prefill — on CPU-only hardware this turns a tens-of-seconds warm-up into milliseconds.

```c
/* On shutdown: */
helix_session_save(session, "conversation.helixstate");

/* On next launch — same model file, same n_ctx: */
helix_session_t* s = NULL;
helix_session_create(model, "{\"n_ctx\":4096}", &s);
if (helix_session_restore(s, "conversation.helixstate") != HELIX_OK) {
    /* HELIX_E_STATE_MISMATCH: different model/n_ctx/format — the session
       is left empty and fully usable; just re-send the history. */
}
```

Rules and behaviour:

- Requires a chat session with the prefix cache enabled. Embedding sessions, speculative (MTP) sessions, and `{"prefix_cache": false}` sessions are rejected with `HELIX_E_UNSUPPORTED_FEATURE`.
- Restore validates a model-identity fingerprint, `n_ctx`, and the file format version before touching the context, and fails with `HELIX_E_STATE_MISMATCH` (-12) on any mismatch or a truncated file. **On failure the session is always left empty and usable.**
- Saves are atomic: the state is written to `path` + `".tmp"` and renamed into place, so a crash mid-save never leaves a corrupt file at `path`. Files are typically tens to hundreds of MB (proportional to KV usage).
- Both calls wait for any in-flight request to finish (one-inference-at-a-time rule).

> **Security:** state files are **trusted input** — restore decodes them directly into engine buffers. Never restore a file from an untrusted source. This is the same trust posture as GGUF model files themselves.

### 3.5 Chat Completions

#### `helix_chat_completions`

```c
helix_status_t helix_chat_completions(
    helix_session_t* session,
    const char* request_json,
    char** out_response_json
);
```

Synchronous completion. On success (`HELIX_OK`), `*out_response_json` receives a `malloc`-allocated, NUL-terminated JSON string. The caller **must** free it with `helix_free()`.

The request JSON follows the [OpenAI Chat Completions schema](https://platform.openai.com/docs/api-reference/chat/create) verbatim. See [§4.4 Request Options](#44-request-options).

**Example call:**
```c
char* out = NULL;
helix_status_t st = helix_chat_completions(session,
    "{\"model\":\"qwen\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],"
    "\"temperature\":0.7,\"max_tokens\":256}",
    &out);
if (st == HELIX_OK) {
    printf("%s\n", out);
    helix_free(out);
}
```

#### `helix_chat_completions_stream`

```c
helix_status_t helix_chat_completions_stream(
    helix_session_t* session,
    const char* request_json,
    helix_stream_cb on_chunk,
    void* user_data
);
```

Streaming completion. Each chunk of the response is delivered via the `on_chunk` callback as a complete `chat.completion.chunk` JSON object. The request must include `"stream": true`.

The callback signature:
```c
typedef int (*helix_stream_cb)(void* user_data, const char* chunk_json);
```

- `chunk_json` is a NUL-terminated JSON string. Valid only during the callback invocation.
- Return **0** to continue generation, **non-zero** to cancel.
- The final invocation passes `chunk_json == NULL` to signal stream completion.
- The callback runs on the engine thread. Implementations should copy the data and return quickly.

**Example:**
```c
static int on_chunk(void* user_data, const char* chunk) {
    if (chunk == NULL) {
        puts("\n[DONE]");
        return 0;
    }
    // Parse or print the delta
    printf("%s\n", chunk);
    return 0;
}

helix_chat_completions_stream(session,
    "{\"model\":\"qwen\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],"
    "\"stream\":true}",
    on_chunk, NULL);
```

#### `helix_free`

```c
void helix_free(char* ptr);
```

Frees strings returned by `helix_chat_completions`. Must **not** be used on strings returned by `helix_*_describe()` or `helix_last_error_json()` — those are library-owned.

### 3.6 Embeddings

*Added in 1.1 (export node `HELIX_1.1`).*

#### `helix_embeddings`

```c
helix_status_t helix_embeddings(helix_session_t* session,
                                const char*      request_json,
                                char**           out_response_json);
```

Computes embeddings following the OpenAI Embeddings API schema. The session **must** have been created with `{"embedding": true}` (see [Session Options](#43-session-options)); calling this on a chat session returns `HELIX_E_UNSUPPORTED_FEATURE` with param `session` — and vice versa for chat calls on an embedding session.

Request:

```json
{
  "model": "embed-test",
  "input": "a single string, or an array of 1..2048 strings",
  "encoding_format": "float"
}
```

- `input` — required; one string or an array of 1..2048 non-empty strings. Pre-tokenized integer arrays are rejected (`HELIX_E_UNSUPPORTED_FEATURE`).
- `encoding_format` — optional; `"float"` (default) returns JSON number arrays, `"base64"` returns the raw little-endian float buffer base64-encoded.
- `dimensions` — rejected (`HELIX_E_UNSUPPORTED_FEATURE`); vectors always have the model's full output dimension.

Response (release with `helix_free`):

```json
{
  "object": "list",
  "data": [ { "object": "embedding", "index": 0, "embedding": [0.0123, "..."] } ],
  "model": "embed-test",
  "usage": { "prompt_tokens": 8, "total_tokens": 8 }
}
```

Vectors are L2-normalized (unit length), matching OpenAI semantics. `data[]` order always matches input order. A single input longer than the session batch size fails with `HELIX_E_CONTEXT_FULL` (param `input`) — there is no silent truncation. One inference at a time per session; `helix_session_cancel()` aborts at the next internal batch boundary with `HELIX_E_CANCELLED`, and the session remains usable afterwards.

Determinism: for a fixed library build, backend, device, and session geometry, repeated calls produce bit-identical vectors. Bit-equality is **not** guaranteed across backends, devices, or llama.cpp pin advances — compare cross-environment vectors with cosine similarity instead.

#### `helix_model_embedding_dim`

```c
uint32_t helix_model_embedding_dim(helix_model_t* model);
```

Pooled embedding output dimension of the loaded model — intended for dimension negotiation before index creation (e.g. HNSW dimension locking). Returns `0` when the model cannot serve embeddings on this endpoint (encoder-decoder architectures, or classifier/reranker heads). Callable on any loaded model without an embedding session; NULL-safe; pure query (no error state, safe to call concurrently). The same value appears as `n_embd_out` in `helix_model_describe()`.

Note this is the *output* dimension, which diverges from the hidden size for some architectures.

### 3.7 Error Handling

All functions return a `helix_status_t`:

| Code | Value | Meaning |
|------|-------|---------|
| `HELIX_OK` | 0 | Success |
| `HELIX_E_INVALID_ARG` | -1 | NULL pointer or invalid argument |
| `HELIX_E_INVALID_JSON` | -2 | JSON parse failure |
| `HELIX_E_VALIDATION` | -3 | OpenAI-shaped 400-level validation error |
| `HELIX_E_MODEL_NOT_FOUND` | -4 | Model file not found at the given path |
| `HELIX_E_MODEL_LOAD_FAILED` | -5 | Model could not be loaded (corrupt or incompatible GGUF) |
| `HELIX_E_OOM` | -6 | Out of memory |
| `HELIX_E_VRAM_EXHAUSTED` | -7 | GPU VRAM exhausted |
| `HELIX_E_CONTEXT_FULL` | -8 | Context window exceeded (chat: prompt + max_tokens > n_ctx, param `max_tokens`; embeddings: one input > batch size, param `input`; rerank: one pair > batch size, param `documents`) |
| `HELIX_E_CANCELLED` | -9 | Request cancelled via `helix_session_cancel` |
| `HELIX_E_BACKEND` | -10 | GGML/CUDA/Metal backend error |
| `HELIX_E_UNSUPPORTED_FEATURE` | -11 | Feature not supported |
| `HELIX_E_STATE_MISMATCH` | -12 | Session state file from a different model, n_ctx, or incompatible library version (added in 1.5) |
| `HELIX_E_INTERNAL` | -99 | Internal error (bug) |

On error, detailed information is available via:

```c
const char* helix_last_error_json(void);
```

Returns an OpenAI-shaped error JSON object. Thread-local; valid until the next `helix_*` call on the same thread. Never returns NULL (returns `"{}"` when no error is set).

```json
{
  "error": {
    "message": "model file not found: /nonexistent.gguf",
    "type": "model_not_found",
    "param": "model_path",
    "code": "helix_e_model_not_found"
  }
}
```

### 3.8 Logging

```c
typedef enum {
    HELIX_LOG_OFF   = 0,
    HELIX_LOG_ERROR = 1,
    HELIX_LOG_WARN  = 2,
    HELIX_LOG_INFO  = 3,
    HELIX_LOG_DEBUG = 4,
    HELIX_LOG_TRACE = 5
} helix_log_level_t;

typedef void (*helix_log_cb)(void* user_data,
                              helix_log_level_t level,
                              const char* msg);

void helix_set_log_callback(helix_log_cb cb,
                             void* user_data,
                             helix_log_level_t min_level);
```

By default, logs go to `stderr` (ERROR, WARN) and `stdout` (INFO, DEBUG, TRACE). Register a callback to redirect logging:

```c
static void my_logger(void* user_data, helix_log_level_t level, const char* msg) {
    FILE* f = (level <= HELIX_LOG_WARN) ? stderr : stdout;
    fprintf(f, "[my-app] %s\n", msg);
}

helix_set_log_callback(my_logger, NULL, HELIX_LOG_INFO);
```

**Thread safety:** This function is not thread-safe during concurrent logging. Set the callback before starting threads that perform inference.

### 3.9 Tokenizer Utilities *(added in 1.4)*

#### `helix_count_tokens`

```c
helix_status_t helix_count_tokens(
    helix_session_t* session,
    const char* request_json,
    uint32_t* out_token_count
);
```

Counts the tokens a chat request would occupy after chat-template rendering — exactly the `usage.prompt_tokens` a real `helix_chat_completions` call on this session would report. Use it for context-budget checks before submit, truncation UI ("this document won't fit"), and latency estimation.

- `request_json` accepts the same body as `helix_chat_completions` (messages, tools, `response_format`, …) and is validated the same way; generation parameters (`temperature`, `max_tokens`, `stream`, …) are accepted but do not affect the count.
- **Pure query:** does not touch the session KV cache or the prefix cache, and is safe to call from any thread — including while a request is in flight on the same session.
- Requests containing image/audio content parts return `HELIX_E_UNSUPPORTED_FEATURE` in this release.

```c
uint32_t n = 0;
if (helix_count_tokens(session, request_json, &n) == HELIX_OK &&
    n + max_tokens > n_ctx) {
    /* trim history before sending */
}
```

#### `helix_tokenize`

```c
helix_status_t helix_tokenize(
    helix_model_t* model,
    const char* text,
    int add_special,
    int parse_special,
    char** out_tokens_json
);
```

Raw tokenizer access using the model's own vocabulary — no second tokenizer that disagrees with the model's BPE. On success `*out_tokens_json` receives a malloc'd JSON array of token ids (`"[1,2,3]"`; `"[]"` for empty text); free it with `helix_free()`.

- `add_special` non-zero adds the model's special prefix/suffix tokens (e.g. BOS), exactly as a chat prompt would receive them.
- `parse_special` non-zero parses control-token text (e.g. `<|im_start|>`) into token ids instead of tokenizing it as plain text.
- Useful for chunking documents for `helix_embeddings`, whose per-input limit is token-based (`HELIX_E_CONTEXT_FULL`, param `input`).

### 3.10 Rerank *(added in 1.5)*

#### `helix_rerank`

```c
helix_status_t helix_rerank(
    helix_session_t* session,
    const char* request_json,
    char** out_response_json
);
```

Reranks documents against a query with a reranker-head model (bge-reranker, Qwen3-Reranker, …) — the missing third of the on-device RAG stack alongside `helix_embeddings` (retrieval) and chat (generation). The request/response shape follows the de-facto `/v1/rerank` standard (Cohere/Jina, also implemented by llama-server).

The session must be created with `{"embedding": true, "pooling": "rank"}` on a model that actually has a rank-capable head; creating a rank session on a plain model — or calling `helix_rerank`/`helix_embeddings` on the wrong session kind — returns `HELIX_E_UNSUPPORTED_FEATURE`.

**Request:**
```json
{
  "model": "my-reranker",
  "query": "What is the capital of France?",
  "documents": ["doc one ...", "doc two ...", "doc three ..."],
  "top_n": 2
}
```

**Response** (free with `helix_free`):
```json
{
  "object": "list",
  "model": "my-reranker",
  "results": [
    {"index": 1, "relevance_score": 6.53},
    {"index": 2, "relevance_score": -2.11}
  ],
  "usage": {"prompt_tokens": 57, "total_tokens": 57}
}
```

Notes:

- `results[]` is sorted by score descending; ties keep input order. `index` refers to the request's `documents[]`.
- Scores are the model's **raw head outputs** (logits, may be negative), matching llama-server. Apply a sigmoid for `[0, 1]` probabilities.
- `documents` holds 1–1024 non-empty strings; each query+document pair must fit the session batch (`HELIX_E_CONTEXT_FULL`, param `documents`, otherwise — raise `n_batch`/`n_ctx` or shorten inputs).
- `helix_session_cancel` aborts at the next internal batch boundary with `HELIX_E_CANCELLED`.
- `helix_model_embedding_dim` still returns 0 for rank-head models — they cannot serve `/v1/embeddings`.

---

## 4. Configuration

### 4.1 Runtime Options

Passed as an optional JSON string to `helix_runtime_create`. All fields are optional.

```jsonc
{
  // Device selection: "auto" | "cpu" | "cuda:N" (e.g. "cuda:0")
  "device_preference": "auto",

  // Log level: "off" | "error" | "warn" | "info" | "debug" | "trace"
  "log_level": "warn",

  // Deterministic mode: force seed=0 for reproducible results
  "deterministic": false
}
```

### 4.2 Model Options

Passed as a JSON string to `helix_model_load`. The `"model_path"` field is **required**; all others are optional.

```jsonc
{
  // Required: path to the GGUF model file
  "model_path": "/models/qwen2.5-0.5b-instruct-q4_k_m.gguf",

  // Optional alias returned in chat completions responses
  "alias": "qwen",

  // Optional mmproj path for multimodal support (vision/audio)
  "mmproj_path": "/models/mmproj-model-f16.gguf",

  // Optional chat template override (Jinja string)
  "chat_template_override": null,

  // Optional context size override (0 = use model default)
  "n_ctx": 0,

  // GPU layer offload: "auto" | "all" | integer
  "n_gpu_layers": "auto",

  // Memory-mapped model loading (default: true)
  "use_mmap": true,

  // Lock model in RAM (default: false)
  "use_mlock": false,

  // Load vocabulary only (for inspection)
  "vocab_only": false,

  // Reasoning format for <think> extraction:
  // "auto" | "deepseek-r1" | "qwq" | "none"
  "reasoning_format": "auto",

  // LoRA adapters (added in 1.6). Validated and loaded once with the model;
  // which adapters a session activates (and at what scale) is a session
  // option — the scale here is the load-time default. "name" defaults to
  // the adapter file stem and must be unique (letters, digits, '.', '_',
  // '-'). Adapter files must be LoRA GGUFs converted against this base
  // model; activated LoRAs (aLoRA) are rejected. Not combinable with
  // "vocab_only".
  "lora": [
    { "path": "/models/support-agent.lora.gguf", "scale": 1.0,
      "name": "support" }
  ]
}
```

#### `n_gpu_layers` values

| Value | Behaviour |
|-------|-----------|
| `"auto"` (default) | Run `llama_params_fit` to determine optimal offload, falling back to a heuristic VRAM estimator if the fit pass fails |
| `"all"` | Offload all layers; warns if VRAM is insufficient |
| `0` | CPU-only (no GPU offload) |
| `"24"` (integer string) | Offload exactly 24 layers |
| `42` (integer) | Same as `"42"` |

### 4.3 Session Options

Passed as an optional JSON string to `helix_session_create`. All fields are optional.

```jsonc
{
  // Context size (0 = model's training context)
  "n_ctx": 0,

  // Batch size for prompt processing (0 = auto-detect)
  "n_batch": 0,

  // Decode thread count (0 = auto-detect)
  "n_threads": 0,

  // Batch/prefill thread count (0 = auto-detect)
  "n_threads_batch": 0,

  // Enable prefix caching across requests (default: true)
  "prefix_cache": true,

  // Context-overflow recovery (added in 1.6; default: false).
  // When true, chat requests that would exceed n_ctx recover instead of
  // failing: oldest non-system history messages are dropped before prefill
  // (template-aware — messages are never split, the system prompt and the
  // latest user message are never dropped), and generation past the wall
  // evicts the oldest KV blocks and continues. Responses report
  //   "helix": {"context_shifted": true, "evicted_tokens": N}
  // when history was lost (streaming: a dedicated chunk with empty choices[]
  // before the usage chunk). Rejected at session creation for embedding/MTP
  // sessions and models whose KV cache cannot be position-shifted
  // (multi-axis rope (M-RoPE, e.g. Qwen3.5/Qwen3-VL), sliding-window, or
  // hybrid attention); image/audio requests and n > 1 are rejected per
  // request.
  "context_shift": false,

  // LoRA adapter activation (added in 1.6). Omit the option entirely to
  // activate every adapter loaded on the model at its load-time scale.
  // Provide an array to activate exactly the named adapters ([] = none);
  // "scale" overrides the load-time scale. Unknown names are rejected with
  // HELIX_E_VALIDATION. Adapters are fixed for the session's lifetime — an
  // adapter change is a new session. Sessions with any active adapter
  // reject "embedding": true, speculative (MTP), and session state
  // save/restore in this release.
  "lora": [ { "name": "support", "scale": 0.8 } ],

  // Run warmup prefill on first request (default: true)
  "warmup": true,

  // Random seed (0 = LLAMA_DEFAULT_SEED)
  "seed": 0,

  // Stream coalescing interval in milliseconds (default: 20)
  // Smaller = more responsive, larger = fewer chunks
  "stream_coalesce_ms": 20,

  // Create an embedding-mode session for helix_embeddings (default: false).
  // An embedding session rejects chat calls and vice versa.
  "embedding": false,

  // Pooling override for embedding sessions:
  // "none" | "mean" | "cls" | "last" | "rank" (default: model metadata).
  // "none" is rejected at session creation — helix_embeddings only serves
  // pooled embedding heads. "rank" (1.5) selects the helix_rerank endpoint
  // and requires a reranker-head model; a rank session cannot serve
  // helix_embeddings (and vice versa).
  "pooling": "mean",

  // Main-context KV cache types (added in 1.4; default "f16").
  // One of: f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1, f32.
  // "q8_0" roughly halves KV memory with negligible quality cost and is the
  // recommended quantized setting (q4_0 has measurable quality cost).
  // A quantized cache_type_v requires flash attention: when the hardware
  // profile disables it, session creation fails with HELIX_E_VALIDATION.
  // The effective values are reported by helix_session_describe.
  "cache_type_k": "f16",
  "cache_type_v": "f16"
}
```

For embedding sessions (`"embedding": true`) the remaining options are accepted but forced or ignored as follows:

| Option | Behaviour in embedding sessions |
|--------|---------------------------------|
| `prefix_cache` | forced `false` (KV is cleared every internal batch) |
| `warmup` | honoured (default `true`): one tiny embedding pass on first request |
| `seed` | ignored (no sampling) |
| `stream_coalesce_ms` | ignored (no streaming on this endpoint) |
| `n_batch` | also sets the physical micro-batch — non-causal attention requires `n_ubatch == n_batch`, and it bounds the longest single input |

### 4.4 Request Options

The chat completions request follows the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create) with some Helix-specific extensions.

```jsonc
{
  // ── Required ──────────────────────────────────────────────────────
  "model": "qwen",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
  ],

  // ── Sampling ──────────────────────────────────────────────────────
  "temperature": 0.7,           // [0, 2]; 0 = greedy
  "top_p": 1.0,                 // [0, 1]; nucleus sampling threshold
  "top_k": 40,                  // Helix extension; top-K filtering
  "min_p": 0.05,                // Helix extension; minimum probability
  "repeat_penalty": 1.0,        // Helix extension; > 0; 1.0 = disabled (default).
                                // Multiplicative penalty over the last 64 tokens.
  "presence_penalty": 0.0,      // [-2, 2]; penalise tokens already present
  "frequency_penalty": 0.0,     // [-2, 2]; penalise frequent tokens
  "seed": 42,                   // Deterministic generation
  "logit_bias": {               // Token-level bias map
    "2435": -100.0,             // Integer keys = token IDs
    "the": 5.0                  // String keys = tokenised text
  },

  // ── Output Control ────────────────────────────────────────────────
  "max_tokens": 256,            // Maximum tokens to generate
  "max_completion_tokens": 256, // Takes precedence over max_tokens
  "reasoning_budget": -1,       // Helix extension; cap on thinking tokens (see below).
                                // -1 = unlimited (default), 0 = no thinking
  "n": 1,                       // Number of completions [1, 128]
  "stop": ["\n\n", "END"],      // Stop strings (string or array)
  "stream": false,              // Enable streaming

  // ── Streaming Options ─────────────────────────────────────────────
  "stream_options": {
    "include_usage": false       // Emit terminal usage chunk
  },

  // ── Logprobs ──────────────────────────────────────────────────────
  "logprobs": false,            // Return per-token log probabilities
  "top_logprobs": null,         // [0, 20]; top-n alternatives

  // ── Tools ─────────────────────────────────────────────────────────
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"}
          },
          "required": ["city"]
        }
      }
    }
  ],
  "tool_choice": "auto",       // "auto" | "none" | "required" | {"type":"function","function":{"name":"..."}}
  "parallel_tool_calls": true,  // Allow multiple tool calls at once

  // ── Structured Outputs ─────────────────────────────────────────────
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "weather_response",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "temperature": {"type": "number"},
          "conditions": {"type": "string"}
        },
        "required": ["temperature", "conditions"],
        "additionalProperties": false
      }
    }
  },

  // ── Multimodal (Phase 7) ─────────────────────────────────────────
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What's in this image?"},
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,iVBORw0KGgo...",
            "detail": "auto"   // "auto" | "low" | "high"
          }
        }
      ]
    }
  ],

  // ── Deprecated v0 API Compatibility ───────────────────────────────
  "functions": [],              // Translated to "tools" if "tools" absent
  "function_call": "auto"       // Translated to "tool_choice" if "tools" absent
}
```

#### `reasoning_budget` (Helix extension)

For reasoning models that emit `<think>...</think>` blocks, `reasoning_budget`
caps the number of tokens the model may spend inside the thinking block:

- `-1` (or unset): unlimited — the model thinks for as long as it wants.
- `0`: thinking is closed immediately; the model answers directly.
- `N > 0`: after `N` thinking tokens the closing tag is forced and the model
  moves on to its answer.

Notes:

- When the budget expires mid-thought, Helix injects a short transition
  sentence (`"I've thought enough; let me give my answer."`) before the
  closing tag so the model hands off cleanly instead of producing an empty
  reply. The transition text is English regardless of conversation language,
  appears in `reasoning_content`, and counts toward the reported
  `reasoning_tokens` usage.
- The thinking tags are taken from the model's chat template; when the
  template does not report a start tag, Helix derives it from the end tag
  (`"</think>"` → `"<think>"`).
- The budget has no effect on models/templates without thinking support.

#### Supported message roles

| Role | Description |
|------|-------------|
| `"system"` | System prompt |
| `"developer"` | OpenAI v2 equivalent, mapped to `"system"` internally |
| `"user"` | User message (supports array content for multimodal) |
| `"assistant"` | Assistant response (supports `tool_calls`) |
| `"tool"` | Tool call result (requires `tool_call_id`) |

---

## 5. Integration Patterns

### 5.1 C Integration

**Complete example:**

```c
#include "helix.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    helix_runtime_t* rt  = NULL;
    helix_model_t*   m   = NULL;
    helix_session_t* s   = NULL;
    char*            out = NULL;
    helix_status_t   st;

    /* 1. Initialise runtime */
    st = helix_runtime_create(NULL, &rt);
    if (st != HELIX_OK) goto cleanup;

    /* 2. Load model */
    st = helix_model_load(rt,
        "{\"model_path\":\"/models/qwen.gguf\",\"alias\":\"qwen\"}", &m);
    if (st != HELIX_OK) goto cleanup;

    /* 3. Create session */
    st = helix_session_create(m, "{\"n_ctx\":4096}", &s);
    if (st != HELIX_OK) goto cleanup;

    /* 4. Run completion */
    st = helix_chat_completions(s,
        "{\"model\":\"qwen\","
         "\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],"
         "\"max_tokens\":64}",
        &out);
    if (st == HELIX_OK) {
        puts(out);
    } else {
        fprintf(stderr, "Error %d: %s\n", st, helix_last_error_json());
    }

cleanup:
    helix_free(out);
    helix_session_destroy(s);
    helix_model_release(m);
    helix_runtime_destroy(rt);
    return (st == HELIX_OK) ? 0 : 1;
}
```

### 5.2 C++ Integration

```cpp
#include "helix.h"
#include <iostream>
#include <memory>
#include <stdexcept>

struct HelixDeleter {
    void operator()(helix_runtime_t* p) const { helix_runtime_destroy(p); }
    void operator()(helix_model_t*   p) const { helix_model_release(p);   }
    void operator()(helix_session_t* p) const { helix_session_destroy(p); }
};

template<typename T>
using HelixPtr = std::unique_ptr<T, HelixDeleter>;

class HelixApp {
public:
    HelixApp(const char* model_path) {
        helix_runtime_t* rt;
        check(helix_runtime_create(nullptr, &rt));
        rt_.reset(rt);

        auto opts = std::string(R"({"model_path":")") + model_path + R"(","alias":"local"})";
        helix_model_t* m;
        check(helix_model_load(rt, opts.c_str(), &m));
        model_.reset(m);

        helix_session_t* s;
        check(helix_session_create(model_.get(), nullptr, &s));
        sess_.reset(s);
    }

    std::string chat(const std::string& user_msg) {
        auto req = R"({"model":"local","messages":[{"role":"user","content":")"
                  + user_msg + R"("}],"max_tokens":256})";
        char* out = nullptr;
        check(helix_chat_completions(sess_.get(), req.c_str(), &out));
        std::string result(out ? out : "");
        helix_free(out);
        return result;
    }

private:
    HelixPtr<helix_runtime_t> rt_;
    HelixPtr<helix_model_t>   model_;
    HelixPtr<helix_session_t> sess_;

    static void check(helix_status_t st) {
        if (st != HELIX_OK)
            throw std::runtime_error(helix_last_error_json());
    }
};

int main() {
    try {
        HelixApp app("/models/qwen.gguf");
        std::cout << app.chat("Hello!") << std::endl;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
```

### 5.3 Streaming

Streaming delivers each token delta as a separate JSON chunk:

```c
struct StreamState {
    FILE* out;
    std::string content;
};

static int on_chunk(void* user_data, const char* chunk_json) {
    auto* state = static_cast<StreamState*>(user_data);
    if (chunk_json == NULL) {
        // Stream complete
        fprintf(state->out, "\n[DONE]\n");
        return 0;
    }
    state->content += chunk_json; // Accumulate
    // In practice, parse with a JSON library and extract delta.content
    return 0; // Return non-zero to cancel
}

StreamState state{stdout};
helix_chat_completions_stream(session,
    "{\"model\":\"qwen\",\"messages\":[{\"role\":\"user\",\"content\":\"Count to 10.\"}],"
    "\"stream\":true}",
    on_chunk, &state);
```

**Streaming chunk format** (OpenAI-compatible):

```json
// First chunk
{"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

// Content chunks
{"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

// Tool call start
{"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_...","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}

// Tool call argument delta
{"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]},"finish_reason":null}]}

// Final chunk (finish_reason set)
{"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

// Usage chunk (when stream_options.include_usage = true)
{"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":15,"completion_tokens":42,"total_tokens":57}}
```

### 5.4 Tool Calling

Helix supports the full OpenAI tool-calling protocol with parallel tool calls and streaming extraction.

**Request with tools:**

```c
const char* req = R"({
    "model": "qwen",
    "messages": [{"role":"user","content":"What's the weather in London and Paris?"}],
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    }],
    "tool_choice": "auto",
    "parallel_tool_calls": true
})";

char* out = NULL;
helix_chat_completions(session, req, &out);
```

**Response with tool calls:**

```json
{
  "id": "chatcmpl-helix-...",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_AbCdEfGhIjKlMnOpQrStUvWx",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"city\":\"London\"}"
          }
        },
        {
          "id": "call_YzZaBbCcDdEeFfGgHhIiJjKk",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"city\":\"Paris\"}"
          }
        }
      ]
    },
    "finish_reason": "tool_calls"
  }],
  "usage": {
    "prompt_tokens": 120,
    "completion_tokens": 48,
    "total_tokens": 168
  }
}
```

### 5.5 Structured Outputs

Helix converts JSON Schema definitions to GBNF grammars, guaranteeing valid structured output.

```c
const char* req = R"({
    "model": "qwen",
    "messages": [{"role":"user","content":"Extract the name and age from: John is 30 years old."}],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "strict": true,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"],
                "additionalProperties": false
            }
        }
    }
})";
```

Three response format types are supported:

| `type` | Description |
|--------|-------------|
| `"text"` (default) | Free-form text output |
| `"json_object"` | Any valid JSON object |
| `"json_schema"` | JSON constrained to the provided schema |

### 5.6 Multimodal Input

Helix supports vision (image input) and audio input through content part arrays in user messages.

```c
const char* req = R"({
    "model": "multimodal-model",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg...",
                    "detail": "auto"
                }
            }
        ]
    }]
})";
```

**Supported image sources:**

| Source | Format | Example |
|--------|--------|---------|
| Data URI | `data:image/{png,jpeg,webp,gif,bmp};base64,...` | Embedded base64 |
| File URI | `file:///absolute/path/to/image.png` | Local file (absolute path only) |

**Supported audio format:** `input_audio` with `format: "mp3"` or `"wav"`, data as base64.

**Prerequisite:** The model must be loaded with `"mmproj_path"` pointing to the multimodal projector GGUF file.

### 5.7 Reasoning Extraction

For reasoning models like DeepSeek-R1 and QwQ, Helix automatically extracts `<think>...</think>` blocks:

- Content inside `<think>` tags is delivered as `reasoning_content` in the response.
- Content outside `<think>` tags is delivered as regular `content`.
- The `reasoning_format` model option controls extraction: `"auto"`, `"deepseek-r1"`, `"qwq"`, or `"none"`.

**Non-streaming response with reasoning:**

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "The final answer is Paris.",
      "reasoning_content": "Let me think about this... France's capital is well-known to be Paris."
    }
  }]
}
```

**Streaming chunks with reasoning:**

```json
{"choices":[{"index":0,"delta":{"reasoning_content":"Let me"}}]}
{"choices":[{"index":0,"delta":{"reasoning_content":" think about"}}]}
{"choices":[{"index":0,"delta":{"reasoning_content":" this..."}}]}
{"choices":[{"index":0,"delta":{"content":"The final"}}]}
{"choices":[{"index":0,"delta":{"content":" answer is Paris."}}]}
```

### 5.8 Embeddings

Load a small embedding model, create an `{"embedding": true}` session, and batch inputs into a single call:

```c
#include "helix.h"
#include <stdio.h>

int main(void) {
    helix_runtime_t* rt = NULL;
    helix_model_t* model = NULL;
    helix_session_t* sess = NULL;

    helix_runtime_create(NULL, &rt);
    helix_model_load(rt,
        "{\"model_path\":\"nomic-embed-text-v1.5.Q8_0.gguf\","
        "\"alias\":\"embedder\"}", &model);

    /* Lock your vector index dimension before creating sessions. */
    uint32_t dim = helix_model_embedding_dim(model);
    if (dim == 0) { /* model cannot serve embeddings */ return 1; }
    printf("embedding dimension: %u\n", dim);

    helix_session_create(model, "{\"embedding\":true}", &sess);

    char* out = NULL;
    helix_status_t st = helix_embeddings(sess,
        "{\"model\":\"embedder\","
        "\"input\":[\"first document\",\"second document\"]}", &out);
    if (st == HELIX_OK) {
        printf("%s\n", out);   /* {"object":"list","data":[...],...} */
        helix_free(out);
    } else {
        fprintf(stderr, "error: %s\n", helix_last_error_json());
    }

    helix_session_destroy(sess);
    helix_model_release(model);
    helix_runtime_destroy(rt);
    return 0;
}
```

**CUDA deployments:** an embedding session sharing a GPU-resident chat model's `helix_model_t` serializes with chat decodes behind the per-model CUDA lock (see [§7.3](#73-cuda-specific-behaviour)). Load the embedder as a **separate** `helix_model_t` — its lock is then its own, and indexing traffic never queues behind chat.

---

## 6. Hardware Auto-Tuning

On initialisation, `helix_runtime_create` probes the host machine and derives optimised inference parameters. The auto-tuning covers:

| Parameter | Derived from | Behaviour |
|-----------|-------------|-----------|
| **n_threads** | CPU topology, GPU offload ratio | Full GPU offload → 1 thread; partial → scaled to CPU-resident layers; CPU-only → performance cores |
| **n_threads_batch** | Same as above, but uses logical cores | Higher thread count for prompt processing |
| **n_batch** | GPU backend, context size, offload ratio | 2048 for Metal/full-GPU, 1024 for partial GPU, 512–2048 for CPU |
| **n_ubatch** | GPU backend, n_batch | 512 max for discrete GPU, equals n_batch for CPU/Metal |
| **flash_attn** | GPU CC, CPU ISA | Enabled for Volta+ CUDA, Apple Silicon, AVX-512/AMX CPUs |
| **n_gpu_layers** | VRAM, model size, `llama_params_fit` | Uses `common_fit_params` with fallback heuristic |

View the detected configuration:
```c
const char* profile = helix_runtime_describe(rt);
// Returns JSON with cpu, ram, accelerators, and auto_tune sections
```

---

## 7. GPU Configuration

### 7.1 Selecting a Backend

Build with `-DHELIX_BACKEND=cuda` (or `metal`, `vulkan`, `rocm`) to enable GPU inference. The `omni` backend enables **all** backends simultaneously, letting Helix choose the best available.

### 7.2 Controlling GPU Layer Offload

At model load time:

```c
// Auto-detect (default)
"n_gpu_layers": "auto"

// Full offload
"n_gpu_layers": "all"

// CPU only
"n_gpu_layers": 0

// Explicit layer count
"n_gpu_layers": 24
```

The `"auto"` strategy uses `llama_params_fit` to determine the optimal offload, considering VRAM, context size, model size, and system memory. If `llama_params_fit` is unavailable, a heuristic fallback estimates layers based on free VRAM.

### 7.3 CUDA-Specific Behaviour

- On CUDA builds, all sessions sharing the same model serialise through a per-model mutex. This is a workaround for a CUDA backend re-entrancy issue in llama.cpp.
- The `cuda_mu_` lock is acquired **before** the per-session lock to prevent deadlocks.
- CUDA compute capability is detected automatically; flash attention is enabled for Volta+ (compute capability ≥ 7.0).

---

## 8. Testing

### 8.1 Unit Tests (no model required)

```bash
# Build and run all unit tests
cmake -B build -DHELIX_BACKEND=cpu && cmake --build build -j$(nproc)
./build/tests/helix_unit_tests

# Run a specific test
./build/tests/helix_unit_tests --gtest_filter="RequestParse.*"
```

### 8.2 Integration Tests (require a model)

```bash
# Basic
HELIX_TEST_MODEL=/path/to/model.gguf ./build/tests/helix_integration_tests

# With custom alias
HELIX_TEST_MODEL=/path/to/model.gguf HELIX_TEST_ALIAS=my-model \
    ./build/tests/helix_integration_tests

# Vision tests (need vision model + mmproj)
HELIX_TEST_VISION_MODEL=/path/to/vision.gguf \
HELIX_TEST_MMPROJ=/path/to/mmproj.gguf \
    ./build-cuda/tests/helix_integration_tests --gtest_filter="HelixVision*"

# Embeddings tests (need an embedding model; alias defaults to "embed-test")
HELIX_TEST_EMBED_MODEL=/path/to/nomic-embed-text-v1.5.Q8_0.gguf \
HELIX_TEST_EMBED_ALIAS=embed-test \
    ./build/tests/helix_integration_tests --gtest_filter="HelixEmbed*"
```

---

## 9. Troubleshooting

### 9.1 Common Errors

| Error | Likely Cause | Solution |
|-------|-------------|----------|
| `HELIX_E_MODEL_NOT_FOUND` | Model path is wrong or file is missing | Verify the path exists and is readable |
| `HELIX_E_MODEL_LOAD_FAILED` | Corrupt GGUF or incompatible version | Try re-downloading the model or use a newer GGUF format |
| `HELIX_E_CONTEXT_FULL` | Prompt + max_tokens exceeds n_ctx | Increase `n_ctx` in session options, reduce `max_tokens`, or shorten the prompt |
| `HELIX_E_VRAM_EXHAUSTED` | GPU VRAM insufficient for the model + context | Reduce `n_gpu_layers`, reduce `n_ctx`, or use a smaller model |
| `HELIX_E_BACKEND` | GGML backend failure (CUDA error, etc.) | Check GPU drivers, try CPU backend for diagnosis |
| `HELIX_E_VALIDATION` | Request JSON violates OpenAI schema | Check the `param` field in the error for the offending field |

### 9.2 Performance Tips

- **First request latency**: After session creation, the first request includes a warmup prefill. This is normal. For latency-sensitive applications, disable warmup (`"warmup": false` in session options) or pre-warm with a dummy request.
- **Stream coalescing**: The default 20 ms coalescing interval balances responsiveness and throughput. Reduce to 5 ms for snappier streaming, increase to 50 ms for higher throughput.
- **Thread count**: On hybrid CPUs (P-cores + E-cores), Helix only uses P-cores for decode. Override with `n_threads` if needed.
- **Prefix caching**: Enabled by default. If your application sends many similar prompts (chat history), prefix caching significantly reduces time-to-first-token on follow-up requests.

### 9.3 Debugging

```c
// Enable verbose logging
helix_set_log_callback(my_logger, NULL, HELIX_LOG_TRACE);

// Check ABI compatibility at runtime
uint32_t ver = helix_abi_version();
printf("Helix ABI version: %d.%d.%d\n",
       ver >> 16, (ver >> 8) & 0xFF, ver & 0xFF);

// Use helix-doctor for a full diagnostic
// $ ./build/bin/helix-doctor --model /path/to/model.gguf --explain
```

---

## 10. Migration Guide: OpenAI → Helix

### 10.1 Python SDK → Helix

**OpenAI Python:**
```python
import openai
client = openai.Client(api_key="...")
resp = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=256,
)
print(resp.choices[0].message.content)
```

**Helix C (equivalent):**
```c
helix_runtime_t* rt;
helix_runtime_create(NULL, &rt);

helix_model_t* m;
helix_model_load(rt, R"({"model_path":"/models/gpt-4.gguf","alias":"gpt-4"})", &m);

helix_session_t* s;
helix_session_create(m, NULL, &s);

char* out;
helix_chat_completions(s,
    R"({"model":"gpt-4","messages":[{"role":"user","content":"Hello"}],)"
    R"("temperature":0.7,"max_tokens":256})",
    &out);
// out contains the same JSON shape as the OpenAI response
helix_free(out);
```

### 10.2 What changes

| Aspect | OpenAI | Helix |
|--------|--------|-------|
| API Endpoint | HTTP POST `https://api.openai.com/v1/chat/completions` | Function call `helix_chat_completions(session, json, &out)` |
| Auth | `Authorization: Bearer sk-...` | None (in-process) |
| Model | Cloud-hosted, specified by string name | Local GGUF file, registered alias |
| Network | Required | None |
| Cancellation | Abort HTTP request | `helix_session_cancel(session)` |
| Error format | JSON with `error.message` | Same JSON shape via `helix_last_error_json()` |
| Request JSON | Identical | Identical |
| Response JSON | Identical (`chat.completion` / `chat.completion.chunk`) | Identical |

### 10.3 Things Helix does NOT do

- **No request/response streaming over HTTP** — The streaming callback is a function pointer, not an SSE stream. Use `helix-shim-server` if you need HTTP transport.
- **No automatic model downloading** — You must provide the GGUF file.
- **No usage metering or rate limiting** — The `usage` object reflects actual token counts, not billing.
- **No multi-user or multi-tenant isolation** — One process, one runtime. Create separate sessions for separate contexts.

---

## Appendix: Build Matrix Validation

| Configuration | Unit Tests | Integration Tests | Notes |
|---------------|-----------|-------------------|-------|
| CPU, Linux x64 | ✅ All pass | ✅ 37/38 | 1 skipped: PrefixCacheSpeedup (timing) |
| CPU, macOS ARM64 | ✅ All pass | ✅ 37/38 | Same skip |
| CUDA, Linux x64 | ✅ All pass | ✅ 37/38 + vision | Requires CUDA 12.x |
| Metal, macOS ARM64 | ✅ All pass | ✅ 37/38 + vision | |

---

*For implementation details, see [RFC-0001-helix.md](RFC-0001-helix.md) and the phase-specific documents in the repository root.*
