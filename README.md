# Helix

**Embeddable, OpenAI-Isomorphic Local LLM Inference Library**

[![Release](https://img.shields.io/badge/release-v1.6.0-brightgreen)](CHANGELOG.md)
[![ABI](https://img.shields.io/badge/ABI-v1.6.0%20frozen%2018%20months-blue)](docs/abi-policy.md)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Helix is a single C/C++ shared library (`libhelix.so` / `libhelix.dylib` / `libhelix.dll`) that transforms any GGUF model into a fully in-process, **OpenAI Chat Completions–compatible** LLM endpoint. No subprocesses. No open ports. No network calls. No Python runtime.

```c
#include "helix.h"

// Initialise once, reuse everywhere
helix_runtime_t* rt;
helix_runtime_create(NULL, &rt);

helix_model_t* model;
helix_model_load(rt,
    R"({"model_path":"/models/qwen.gguf","alias":"qwen"})", &model);

helix_session_t* sess;
helix_session_create(model, NULL, &sess);

char* out;
helix_chat_completions(sess,
    R"({"model":"qwen","messages":[{"role":"user","content":"Hello!"}]})",
    &out);
// out → {"id":"chatcmpl-...","choices":[{"message":{"content":"Hi!"}}],...}
helix_free(out);
```

---

## Why Helix?

Existing approaches for local LLM inference each have a critical gap:

| Approach | Problem |
|----------|---------|
| **`llama-server`** | Spawns a child process, binds a TCP port — port collisions, firewall warnings, external `.exe` lifecycle. A server, not a library. |
| **`libllama` raw C API** | Exposes tokens, batches, sampler chains, KV caches — a 10× learning curve before the first completion. Every consumer rebuilds the same scaffolding badly. |
| **Python/npm bindings** | Require a language runtime. Not embeddable from native desktop apps (Cocoa, WinUI, Qt, Electron without Node). |

Helix solves this by offering a **tiny C ABI** (29 symbols) whose **JSON wire protocol is byte-identical to OpenAI's `/v1/chat/completions`** — streaming chunks, tool calls, structured outputs, logprobs, and usage objects all included.

---

## Features

- **OpenAI-isomorphic protocol** — Same JSON request/response shape. Use existing OpenAI SDKs, agents, and frameworks.
- **In-process inference** — No subprocesses, no sockets, no HTTP server. Just a function call.
- **Hardware auto-tuning** — On init, probes CPU topology, SIMD ISA, RAM, and GPU accelerators; auto-selects threads, batch sizes, GPU layers, and flash attention.
- **Multi-backend GPU** — CUDA, Metal, Vulkan, ROCm — selected at build time via `-DHELIX_BACKEND=`.
- **Structured outputs** — JSON Schema → GBNF grammar for guaranteed-valid structured generation.
- **Tool calling** — Full function-calling support with parallel tool calls and PEG-based streaming extraction.
- **Streaming** — Per-token SSE-shaped chunks with a cooperative cancellation callback.
- **Logprobs** — Per-token log probabilities with configurable top-n alternatives.
- **Multimodal** — Vision (image input) and audio support via `mmproj` projector models.
- **Reasoning extraction** — Automatic `<think>...</think>` block separation for DeepSeek-R1, QwQ, and similar models.
- **Prefix caching** — Automatic KV-cache reuse across requests with identical prompt prefixes.
- **Multi-token prediction (MTP)** — Speculative decoding via upstream `llama.cpp` common speculative infrastructure. 1.5-1.8x throughput speedup on MTP-trained models (e.g. Gemma-4 with a `gemma4-assistant` draft head), with full streaming/tool/reasoning compatibility.
- **Session introspection** — `helix_session_describe` returns live context geometry and speculative state as JSON (ABI v1.2).
- **Session isolation** — Each `helix_session_t` owns its own KV cache and cancellation state; multiple sessions share the same loaded model.
- **Tokenizer utilities** — `helix_count_tokens` (exact prompt-token count for a chat request) and `helix_tokenize` (raw tokenizer access) (ABI v1.4).
- **Load progress & cancellation** — `helix_model_load_ex` reports monotonic load progress and supports cooperative abort (ABI v1.4).
- **KV-cache quantization** — `cache_type_k`/`cache_type_v` session options (`q8_0` roughly halves KV VRAM) (ABI v1.4).
- **Session persistence** — `helix_session_save`/`helix_session_restore` persist a chat session's KV cache and history across process restarts (ABI v1.5).
- **Reranking** — `helix_rerank` in the de-facto `/v1/rerank` shape (Cohere/Jina compatible) on reranker-head models (ABI v1.5).
- **Context shift** — opt-in `context_shift` recovers from context overflow (template-aware history eviction + rope-shift) instead of failing (ABI v1.6).
- **LoRA adapters** — load adapter GGUFs with the base model and select per-session which adapters are active, at what scale (ABI v1.6).

---

## Quick Start

### Prerequisites

- **C++17** compiler (GCC 11+, Clang 14+, MSVC 2022+)
- **CMake** ≥ 3.22
- A **GGUF model file** (e.g., [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF))

### Build

```bash
git clone --recurse-submodules https://github.com/anemoi-ai/helix
cd helix

# CPU backend (default)
cmake -B build -DHELIX_BACKEND=cpu
cmake --build build -j$(nproc)

# CUDA backend
cmake -B build-cuda -DHELIX_BACKEND=cuda
cmake --build build-cuda -j$(nproc)
```

Outputs:
```
build/
├── libhelix.so            # Shared library
├── libhelix.a             # Static library
├── bin/
│   ├── helix-doctor       # Diagnostic tool
│   └── helix-shim-server  # HTTP test server
└── tests/
    ├── helix_unit_tests
    └── helix_integration_tests
```

### Run the diagnostic tool

```bash
./build/bin/helix-doctor --model /path/to/model.gguf
```

### Run tests

```bash
# Unit tests (no model required)
./build/tests/helix_unit_tests

# Integration tests (requires a model)
HELIX_TEST_MODEL=/path/to/model.gguf ./build/tests/helix_integration_tests
```

### Minimal C example

```c
#include "helix.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    helix_runtime_t* rt  = NULL;
    helix_model_t*   m   = NULL;
    helix_session_t* s   = NULL;
    char*            out = NULL;

    helix_runtime_create(NULL, &rt);
    helix_model_load(rt,
        "{\"model_path\":\"/models/qwen.gguf\",\"alias\":\"qwen\"}", &m);
    helix_session_create(m, "{\"n_ctx\":4096}", &s);

    helix_status_t st = helix_chat_completions(s,
        "{\"model\":\"qwen\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"What is the capital of France?\"}]"
        ",\"max_tokens\":64}",
        &out);

    if (st == HELIX_OK) {
        printf("%s\n", out);
        helix_free(out);
    } else {
        fprintf(stderr, "Error: %s\n", helix_last_error_json());
    }

    helix_session_destroy(s);
    helix_model_release(m);
    helix_runtime_destroy(rt);
    return 0;
}
```

```bash
# Build
cc -I/path/to/helix -L/path/to/build example.c -lhelix -o example
LD_LIBRARY_PATH=/path/to/build ./example
```

### Speculative decoding (MTP)

For MTP-trained models (e.g. Gemma-4), pass `speculative` in the session JSON.
Gemma-4 ships its MTP head as a separate `gemma4-assistant` GGUF — point
`model_path` at it:

```c
helix_session_t* sess;
helix_session_create(model,
    "{\"n_ctx\":4096,\"speculative\":{"
    "  \"type\":\"draft-mtp\","
    "  \"model_path\":\"/models/mtp-gemma-4-12B-it.gguf\","
    "  \"n_max\":3"
    "}}", &sess);

// Introspect the session (ABI v1.2):
char* desc = helix_session_describe(sess);
// → {"n_ctx":4096,...,"speculative":{"type":"draft-mtp","enabled":true,"n_max":3,...}}
helix_free(desc);
```

The speculative decode path drives upstream `common_speculative` draft/verify/accept,
producing observation-equivalent output under greedy sampling while delivering
**1.5-1.8x throughput** on supported models. Streaming, tool calls, reasoning
extraction, and stop strings all work unchanged under MTP. Multimodal requests
automatically fall back to the non-speculative path.

---

## Project Status

| Version | Highlights | Status |
|---------|-----------|--------|
| 1.0 | GA — chat, streaming, tools, structured outputs, logprobs, multimodal, reasoning, GPU backends, ABI freeze | ✅ Released |
| 1.1 | Frontier samplers | ✅ Released |
| 1.2 | MTP speculative decoding & session introspection | ✅ Released |
| 1.3 | Windows port (MSVC CPU/CUDA/Vulkan/ROCm) + CI matrix | ✅ Released |
| 1.4 | Tokenizer utilities, load progress, KV-cache quantization | ✅ Released |
| 1.5 | Session save/restore, reranking | ✅ Released |
| 1.6 | Context-shift overflow recovery, LoRA adapters | ✅ Released |
| — | macOS port (Apple Silicon Metal, Intel CPU) | 📋 Planned |
| — | Language wrappers (Python/Node/Go/Rust/.NET/Swift) | 🔄 In progress |

**Current:** ABI v1.6.0 — see [`CHANGELOG.md`](CHANGELOG.md) for the full history.

---

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) | Comprehensive API reference, configuration, and integration guide |
| [`helix.h`](helix.h) | Public C ABI header (the entire public surface) |
| [`docs/api-reference.md`](docs/api-reference.md) | C API reference |
| [`docs/install.md`](docs/install.md) | Installation options |
| [`docs/abi-policy.md`](docs/abi-policy.md) | ABI stability policy |
| [`docs/wrappers/index.md`](docs/wrappers/index.md) | Language wrapper overview |
| [`CHANGELOG.md`](CHANGELOG.md) | Release history |

---

## Repository Layout

```
helix/
├── helix.h                  # Public C ABI header
├── CMakeLists.txt           # Top-level build
├── src/
│   ├── abi/                 # C ABI implementation (helix_* symbols)
│   ├── engine/              # Decode loop, event sinks, logprobs, warmup
│   ├── json/                # Request/response parsing (OpenAI shape)
│   ├── session/             # Session lifecycle, KV cache, MTP speculative, cancellation
│   │   ├── session.cpp      # Context creation, MTP wiring, warmup
│   │   └── options.cpp      # SessionOptions + SpeculativeOptions parsing
│   ├── model/               # Model loading, GPU layer selection
│   ├── runtime/             # Process-wide initialisation
│   ├── chat/                # Chat template rendering, reasoning extraction
│   ├── sampling/            # Sampler chain construction, logit bias
│   ├── grammar/             # JSON Schema → GBNF grammar bridge
│   ├── hardware/            # CPU/GPU/RAM probing, auto-tuning
│   ├── multimodal/          # Image decoding, mmproj loading
│   └── internal/            # Error handling, logging
├── tests/
│   ├── unit/                # 26 unit test files (no model required)
│   └── integration/         # End-to-end integration tests
├── tools/
│   ├── helix-doctor/        # Diagnostic CLI
│   └── helix-shim-server/   # HTTP shim server for testing
└── third_party/
    ├── llama.cpp/           # Vendored inference engine (pinned)
    └── nlohmann_json.hpp    # Vendored JSON library
```

---

## Platform Support

| Platform | CPU | CUDA | Metal | Vulkan | ROCm |
|----------|-----|------|-------|--------|------|
| Linux x64 | ✓ | ✓ | — | ◐ | ◐ |
| Linux ARM64 | ◐ | — | — | ◐ | — |
| Windows x64 | ✓ | ✓ | — | ✓ | ✓ |
| macOS ARM64 | 📋 | — | 📋 | — | — |
| macOS x64 | 📋 | — | 📋 | — | — |

**Legend:** ✓ CI-tested · ◐ Buildable but untested · 📋 Planned

Linux x64 (CPU + CUDA) and Windows x64 (CPU/CUDA/Vulkan/ROCm) are tested in
CI. The build system has scaffolding for macOS (Metal, dylib versioning), but
macOS has not yet been compiled or verified.

---

## License

Helix is distributed under the terms of the Apache License 2.0. See [`LICENSE`](LICENSE) for details and [`NOTICE`](NOTICE) for third-party attributions.

`llama.cpp` is vendored under its own license (MIT). `nlohmann/json` is vendored under the MIT License.
