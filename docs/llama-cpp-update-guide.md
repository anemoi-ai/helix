# Updating Helix to the latest `llama.cpp`

Date: 2026-06-14  
Owner: Helix maintainers  
Scope: update the vendored `llama.cpp` tree used by Helix while keeping Helix's public API and behavior stable.

## Summary

Helix vendors `llama.cpp` under `third_party/llama.cpp` and records the pinned commit in `third_party/llama.cpp.commit`. The current pinned commit is:

```text
1acee6bf8939948f9bcbf4b14034e4b475f06069
```

The latest upstream commit fetched during this work is:

```text
ef8268feee28ae943958049bf3bbab4bda99c0ea
```

This document describes how to move Helix from the current vendored commit to latest upstream `llama.cpp` without accidentally changing Helix's public API, multimodal behavior, or decode semantics.

## Goals

1. Update `third_party/llama.cpp` to latest upstream `origin/master`.
2. Keep Helix's existing C ABI stable unless a separate ABI change is explicitly approved.
3. Keep Helix's current behavior unchanged by default:
   - no MTP/speculative decoding by default
   - no QAT/quantization API by default
   - same OpenAI-compatible chat, embeddings, tool, reasoning, and logprob behavior
4. Update the small number of Helix source files that directly depend on unstable `llama.cpp` helper APIs.
5. Rebuild and run regression tests before merging.

## Non-goals

This update does **not** require implementing the following:

- MTP/speculative decoding support
- QAT/quantization API support
- video input support
- new Helix public ABI symbols
- changes to request/session JSON semantics

MTP and QAT are separate design tracks and out of scope for routine updates.

## Current risk areas

The latest upstream `llama.cpp` update is expected to be mostly low-risk for Helix's direct C API usage, but there are a few known areas that require attention.

### 1. `mtmd_helper_bitmap_init_from_buf` changed

Latest upstream changed this helper from returning a raw `mtmd_bitmap *` to returning a wrapper struct:

```cpp
struct mtmd_helper_bitmap_wrapper {
    mtmd_bitmap * bitmap;
    mtmd_helper_video * video_ctx;
};

MTMD_API struct mtmd_helper_bitmap_wrapper
mtmd_helper_bitmap_init_from_buf(mtmd_context * ctx,
                                 const unsigned char * buf,
                                 size_t len,
                                 bool placeholder);
```

Helix currently calls the old API in `src/multimodal/mmproj.cpp:75`.

Required Helix change:

```cpp
auto wrapper = mtmd_helper_bitmap_init_from_buf(ctx_, raw.data(), raw.size(), false);
BitmapPtr bmp(wrapper.bitmap, bitmap_deleter);
if (wrapper.video_ctx) {
    mtmd_helper_video_free(wrapper.video_ctx);
    wrapper.video_ctx = nullptr;
}
if (!bmp) {
    throw Error(HELIX_E_VALIDATION,
                "failed to decode media payload (unsupported format or corrupt data)",
                "invalid_request_error", "messages", "helix_e_validation");
}
bitmaps.push_back(std::move(bmp));
```

Notes:

- Pass `false` for `placeholder` because Helix needs real image/audio data.
- Free `wrapper.video_ctx` if it is non-null.
- Keep `mtmd_bitmap_free` as the bitmap deleter.
- If Helix sets `MTMD_VIDEO OFF`, video context should not be returned, but the code should still handle the wrapper shape.

### 2. `mtmd_context_params_default` now defaults `batch_max_tokens` to 1024

Latest upstream `mtmd_context_params_default()` sets `batch_max_tokens = 1024`. Helix can accept this default.

If Helix wants explicit behavior, add:

```cpp
p.batch_max_tokens = 0; // or a Helix-owned default
```

Do this only if tests show the default changes multimodal batching behavior.

### 3. `MTMD_VIDEO` defaults to `ON`

Latest upstream `tools/mtmd/CMakeLists.txt` defaults:

```cmake
set(MTMD_VIDEO ON CACHE BOOL "enable video support in mtmd (requires ffmpeg binary in PATH)")
```

Helix does not currently expose video input. To avoid pulling in ffmpeg/ffprobe dependencies, set it off before adding `third_party/llama.cpp`:

```cmake
set(MTMD_VIDEO OFF CACHE BOOL "" FORCE)
```

Place this in `CMakeLists.txt` with the other vendored `llama.cpp` build flags.

### 4. Chat template and sampling helpers changed

Helix uses these upstream common helpers:

- `common_chat_templates_init`
- `common_chat_templates_apply`
- `common_chat_peg_parse`
- `llama_sampler_*`
- `llama_batch_init`
- `llama_batch_free`

Latest upstream still exposes these APIs, but some signatures and internal behavior changed. No source change is currently required for the calls Helix already makes, but regression tests are mandatory.

Affected Helix files:

- `src/chat/template.cpp`
- `src/engine/decode_loop.cpp`
- `src/engine/embed_loop.cpp`
- `src/engine/warmup.cpp`
- `src/sampling/chain.cpp`

### 5. MTP context fields are new

Latest upstream `llama_context_params` includes:

```cpp
uint32_t n_outputs_max;
struct llama_context * ctx_other;
```

These are used by upstream's MTP server path. Helix does not need to use them unless implementing MTP. Do not add MTP support as part of a routine `llama.cpp` update.

## Update procedure

### Step 1: start from a clean working tree

Run these commands from the Helix repository root:

```bash
git status --short
git -C third_party/llama.cpp status --short --branch
git -C third_party/llama.cpp fetch origin master
```

Confirm the current vendored commit:

```bash
cat third_party/llama.cpp.commit
git -C third_party/llama.cpp rev-parse HEAD
```

Expected current values before update:

```text
1acee6bf8939948f9bcbf4b14034e4b475f06069
```

Confirm the fetched latest commit:

```bash
git -C third_party/llama.cpp rev-parse origin/master
```

Expected latest value from this work:

```text
ef8268feee28ae943958049bf3bbab4bda99c0ea
```

### Step 2: update the vendored `llama.cpp` checkout

The vendored tree is a nested git repository, not a git submodule in the Helix repo. Update it directly:

```bash
git -C third_party/llama.cpp checkout origin/master
git -C third_party/llama.cpp status --short --branch
```

Then update Helix's commit marker:

```bash
git -C third_party/llama.cpp rev-parse HEAD > third_party/llama.cpp.commit
```

The file should contain exactly one SHA, with no extra whitespace:

```text
ef8268feee28ae943958049bf3bbab4bda99c0ea
```

### Step 3: add Helix CMake guard for video support

In `CMakeLists.txt`, add `MTMD_VIDEO OFF` with the other vendored `llama.cpp` build flags:

```cmake
set(LLAMA_BUILD_TOOLS    ON  CACHE BOOL "" FORCE)  # needed for the mtmd library
set(MTMD_VIDEO           OFF CACHE BOOL "" FORCE)  # Helix does not expose video input
```

Keep existing Helix backend flags unchanged:

```cmake
if(HELIX_BACKEND STREQUAL "cuda" OR HELIX_BACKEND STREQUAL "omni")
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()
if(HELIX_BACKEND STREQUAL "metal" OR HELIX_BACKEND STREQUAL "omni")
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()
if(HELIX_BACKEND STREQUAL "vulkan" OR HELIX_BACKEND STREQUAL "omni")
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()
if(HELIX_BACKEND STREQUAL "rocm" OR HELIX_BACKEND STREQUAL "omni")
    set(GGML_HIP ON CACHE BOOL "" FORCE)
endif()
```

### Step 4: update the multimodal bitmap helper

Edit `src/multimodal/mmproj.cpp` so it handles `mtmd_helper_bitmap_wrapper`.

The current code creates a `BitmapPtr` directly from `mtmd_helper_bitmap_init_from_buf`. Replace that with wrapper handling:

```cpp
auto wrapper = mtmd_helper_bitmap_init_from_buf(ctx_, raw.data(), raw.size(), false);
if (wrapper.video_ctx) {
    mtmd_helper_video_free(wrapper.video_ctx);
    wrapper.video_ctx = nullptr;
}

BitmapPtr bmp(wrapper.bitmap, bitmap_deleter);
if (!bmp) {
    throw Error(HELIX_E_VALIDATION,
                "failed to decode media payload (unsupported format or corrupt data)",
                "invalid_request_error", "messages", "helix_e_validation");
}
bitmaps.push_back(std::move(bmp));
```

Do not change the rest of `MmProj::eval_media` unless tests show a multimodal regression.

### Step 5: configure a clean build

Use a fresh build directory for the first update. This avoids stale CMake cache values from the old `llama.cpp`.

CPU:

```bash
cmake -B build-llama-update -DHELIX_BACKEND=cpu
cmake --build build-llama-update -j$(nproc)
```

CUDA, if applicable:

```bash
cmake -B build-llama-update-cuda -DHELIX_BACKEND=cuda
cmake --build build-llama-update-cuda -j$(nproc)
```

If the build fails in an existing directory, delete the build directory and reconfigure:

```bash
rm -rf build-llama-update
cmake -B build-llama-update -DHELIX_BACKEND=cpu
cmake --build build-llama-update -j$(nproc)
```

### Step 6: fix compile errors

Expect most compile errors to come from unstable helper APIs, not Helix's public C API.

Recommended order:

1. Fix `src/multimodal/mmproj.cpp` first.
2. Fix any `mtmd` include or symbol errors.
3. Fix any `common_chat_*` signature errors.
4. Fix any `llama_sampler_*` signature errors.
5. Fix any `llama_batch_init` / `llama_batch_free` errors.
6. Fix any backend/GGML build errors.

Do not silently coerce types or remove checks to make the build pass. If a helper signature changed, update the call site explicitly and add a comment only if the reason is not obvious from the code.

### Step 7: run unit tests

Run the CPU unit suite:

```bash
./build-llama-update/tests/helix_unit_tests
```

If using CTest instead:

```bash
ctest --test-dir build-llama-update --output-on-failure
```

The unit suite covers request parsing, response serialization, stop strings, sampler chain behavior, UTF-8 prefix handling, tools, hardware probing, GPU layer selection, CUDA locking, response format validation, logit bias, logprobs formatting, content parts, data URIs, reasoning extraction, and embeddings request/packing behavior.

### Step 8: run integration tests

Integration tests require a model.

```bash
HELIX_TEST_MODEL=/path/to/model.gguf \
HELIX_TEST_EMBED_MODEL=/path/to/embed-model.gguf \
./build-llama-update/tests/helix_integration_tests
```

If only one model is available, run with at least:

```bash
HELIX_TEST_MODEL=/path/to/model.gguf ./build-llama-update/tests/helix_integration_tests
```

### Step 9: run smoke tests

Run the diagnostic tool:

```bash
./build-llama-update/bin/helix-doctor --model /path/to/model.gguf
```

Run a normal chat completion smoke test through the Helix shim server or a language wrapper.

Recommended smoke coverage:

- CPU text-only chat
- CUDA text-only chat, if available
- embeddings
- tool calling
- reasoning extraction
- logprobs
- multimodal image input
- invalid model path
- invalid session options

## Verification checklist

Before merging, confirm:

- [ ] `third_party/llama.cpp` is on the intended upstream commit.
- [ ] `third_party/llama.cpp.commit` matches `git -C third_party/llama.cpp rev-parse HEAD`.
- [ ] `MTMD_VIDEO OFF` is set in `CMakeLists.txt`.
- [ ] `src/multimodal/mmproj.cpp` handles `mtmd_helper_bitmap_wrapper`.
- [ ] CPU build succeeds from a clean build directory.
- [ ] CUDA build succeeds if CUDA is part of the release target.
- [ ] `helix_unit_tests` passes.
- [ ] `helix_integration_tests` passes with the available test model.
- [ ] Chat template behavior is unchanged for supported templates.
- [ ] Sampling behavior is unchanged for deterministic requests.
- [ ] Embeddings behavior is unchanged.
- [ ] Multimodal image input works.
- [ ] No Helix public ABI symbols were added or removed unintentionally.
- [ ] `helix_version_string()` reports the new pinned `llama.cpp` short SHA.

## Rollback procedure

If the update is not ready to merge, restore the old vendored commit and commit marker.

```bash
git -C third_party/llama.cpp checkout 1acee6bf8939948f9bcbf4b14034e4b475f06069
printf '%s\n' 1acee6bf8939948f9bcbf4b14034e4b475f06069 > third_party/llama.cpp.commit
rm -rf build-llama-update
```

Then rebuild from the old build directory or recreate the normal build directory:

```bash
cmake -B build -DHELIX_BACKEND=cpu
cmake --build build -j$(nproc)
```

If `MTMD_VIDEO OFF` was added to `CMakeLists.txt`, it can remain. It only prevents optional video support from being built into `mtmd`.

## MTP and QAT notes

### MTP

Latest upstream exposes MTP through `llama_context_params`:

```cpp
LLAMA_CONTEXT_TYPE_MTP
n_rs_seq
n_outputs_max
ctx_type
ctx_other
```

Upstream server creates an MTP context from the target model and initializes the common speculative decoding manager. Helix currently owns its decode loop in `src/engine/decode_loop.cpp` and does not use upstream speculative decoding infrastructure.

Do not enable MTP during a routine `llama.cpp` update. If MTP is requested later, use a separate design and implementation plan.

### QAT

QAT is a quantization-time concern, not an inference-time option. Latest upstream exposes quantization through:

```cpp
llama_model_quantize_params
llama_model_quantize
```

Helix does not currently expose a quantization API. Do not add QAT settings to `chat_completions`, request JSON, or `SessionOptions`.

## Suggested commit structure

A clean update can be split into small commits:

1. `chore: fetch latest llama.cpp and update commit marker`
2. `build: disable MTMD video support for Helix`
3. `multimodal: adapt mtmd bitmap helper wrapper API`
4. `test: update regression expectations for latest llama.cpp`
5. `docs: document latest llama.cpp update process`

Do not combine unrelated MTP or QAT implementation work into the `llama.cpp` update commit.

## Known current state

- Helix ABI version: `1.1.0`
- Current vendored `llama.cpp`: `1acee6bf8939948f9bcbf4b14034e4b475f06069`
- Fetched latest upstream `llama.cpp`: `ef8268feee28ae943958049bf3bbab4bda99c0ea`
- Main expected source break: `mtmd_helper_bitmap_init_from_buf`
- Main expected build risk: `MTMD_VIDEO ON` upstream default
- Main behavior risk areas: chat templates, sampling, multimodal, embeddings, hardware probing
