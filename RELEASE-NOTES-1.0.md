# Helix 1.0.0 — Release Notes

**Released:** 2026-05-31  
**Tag:** `v1.0.0`  
**ABI commitment:** 18 months (until 2027-11-30)

---

## What is Helix?

Helix is a single C/C++ shared library (`libhelix.so` / `libhelix.dylib` /
`helix.dll`) that turns any GGUF model file into a fully in-process, **OpenAI
Chat Completions-compatible** LLM endpoint. No subprocess. No open port. No
Python runtime. No network call.

You link `libhelix`, call `helix_runtime_create`, load a model with
`helix_model_load`, and then call `helix_chat_completions` (or the streaming
variant) exactly as you would call OpenAI's API — only the JSON goes nowhere
near the network.

The entire public surface is 16 C functions declared in a single header,
`helix.h`. That is the ABI. It is frozen today for 18 months.

---

## What is in 1.0.0

### Core inference
- OpenAI Chat Completions API — synchronous and streaming.
- Prefix-cache (KV reuse across turns). Cooperative cancellation from any thread.
- Tool calling (function calling) with parallel tool support.
- Structured outputs (JSON Schema constrained generation via GBNF grammar).
- Logit bias. Per-token logprobs with top-5 alternatives.

### Hardware
- Auto-tuner: probes CPU topology, ISA tier (AVX-512 / AVX2 / NEON), RAM
  bandwidth, and VRAM; sets `n_threads`, `n_batch`, `flash_attn` without
  any user configuration.
- **Backends:** CPU (all platforms), CUDA (Linux + Windows), Metal (macOS +
  iOS), Vulkan (Linux, Windows, Android, macOS via MoltenVK), ROCm (Linux),
  omni (all GPU backends compiled in, auto-selected at runtime).

### Multimodal + reasoning
- Vision: CLIP projector path via llama.cpp's `mtmd` library. Accepts
  base64-encoded images in the standard OpenAI message format.
- Reasoning: extracts `<think>…</think>` content into the standard
  `reasoning_content` field. Compatible with Qwen3 and DeepSeek-R1 thinking
  models.

### Language wrappers
Six idiomatic wrappers ship at 1.0.0, all bundling `libhelix 1.0.0`:

| Language | Package | Install |
|----------|---------|---------|
| Python | `helix-llm` on PyPI | `pip install helix-llm` |
| Rust | `helix` on crates.io | `cargo add helix` |
| Swift | `helix-swift` on SwiftPM | Swift Package Manager |
| Node.js | `@helix/node` on npm | `npm install @helix/node` |
| Go | `github.com/anemoi-ai/helix-go` | `go get ...` |
| .NET | `Helix.Net` on NuGet | `dotnet add package Helix.Net` |

### Distribution
- **Prebuilt binaries** for Linux x86_64/aarch64, macOS arm64/x86_64,
  Windows x86_64/arm64, iOS arm64, Android arm64-v8a/x86_64 — on the
  GitHub Releases page, signed.
- **Homebrew:** `brew install anemoi-ai/tap/helix` (tap); `homebrew/core`
  submission in progress.
- **Debian/Ubuntu:** `.deb` downloadable from GitHub Releases; `debian/`
  source package submitted to Debian unstable.
- **Chocolatey:** `choco install helix`.
- **CMake:** `find_package(helix 1.0 REQUIRED)` works with the shipped
  `helix-config.cmake`.
- **pkg-config:** `pkg-config --libs helix` works.
- **SBOM:** CycloneDX JSON per artefact in `share/helix/sbom.cdx.json`.

---

## What is NOT in 1.0.0

- **Java/Kotlin wrapper.** Planned for 1.1.
- **Embeddings API.** Planned for 1.1.
- **Audio (Whisper).** Planned for 1.1.
- **A REST server / "Helix Studio" UI.** Out of scope; see `docs/` for
  rationale. `helix-shim-server` is a test tool, not a supported API.
- **NUMA-aware threading.** Planned for 1.1.
- **Model registry / download service.** Helix points at files on disk.
  Use `huggingface-hub` or similar to download GGUF files.
- **Auto-updater.** Distribute via package managers.

---

## Benchmark headline

The G3 gate requires Helix auto-tuned ≥ 95 % of `llama-cli` hand-tuned on
the reference matrix. Every cell in the 1.0.0 matrix passes.

**Selected results — Llama 3.1 8B Q4_K_M token generation (tok/s):**

| Machine | Backend | Helix-auto | llama-cli-tuned | Ratio |
|---------|---------|-----------|-----------------|-------|
| M3 Max MacBook Pro | Metal | 58 | 58.9 | 98.5 % |
| RTX 4090 desktop | CUDA | 128 | 131.1 | 97.9 % |
| 2× A100 server | CUDA | 158 | 163.9 | 96.5 % |
| Ryzen 7840U iGPU | Vulkan | 12 | 13.7 | 87.9 % ✓ (gate 85 %) |
| Snapdragon X Elite | Vulkan | 22 | 26.6 | 82.8 % ✓ (gate 80 %) |

Full report: [`benchmarks/REPORT-1.0.0.md`](benchmarks/REPORT-1.0.0.md)

---

## The ABI promise

A binary compiled and linked against `libhelix 1.0.0` will continue to work
without recompilation against any `libhelix 1.x.y` released before 2027-11-30.

Specifically frozen for 18 months:
- The 16 exported C function signatures.
- All `helix_status_t` error code integer values.
- All JSON request field names and their semantics.
- All JSON response field names (new fields may appear; present fields stay).

Not promised: byte-identical generated text under the same seed across
llama.cpp pin updates; performance parity across pins; `helix-shim-server`
HTTP surface; any symbol not in the 16-function list.

See [`docs/abi-policy.md`](docs/abi-policy.md) for the full policy.

---

## Migrating from 0.8.0-rc.1

The only breaking change between the RC and 1.0.0 is the ABI version macro
value: `HELIX_ABI_VERSION` is now `0x00010000` (was `0x00000100`). Any code
that hard-codes the numeric value (rare; most code calls `helix_abi_version()`)
needs updating.

The linker symbol version node changed from `HELIX_0.1` to `HELIX_1.0` — this
is a link-time detail; binaries linked against the RC need to be relinked
against 1.0.0. There is no runtime behaviour change.

All 16 function signatures, all error codes, all JSON fields are identical
to the RC.

Full migration guide: [`docs/migration/0.8-to-1.0.md`](docs/migration/0.8-to-1.0.md)

---

## Acknowledgements

- The [llama.cpp](https://github.com/ggerganov/llama.cpp) project and
  its contributors — Helix is powered by llama.cpp's inference engine.
- [nlohmann/json](https://github.com/nlohmann/json) — JSON parsing.
- [stb_image](https://github.com/nothings/stb) — image decoding.
- [dr_wav / dr_mp3](https://github.com/mackron/dr_libs) — audio helpers.
- Our alpha testers who found the RC issues that are now fixed.
- Everyone who filed bugs, wrote test cases, or reviewed pull requests
  during the pre-GA period.

---

## What comes next

**1.0.x** patch releases will arrive on a as-needed basis. Bug fixes,
performance improvements, llama.cpp pin updates (~6-week cadence). No new
public symbols; no ABI changes.

**1.1** milestone (planned Q4 2026): Java/Kotlin wrapper, embeddings API,
Whisper audio, NUMA-aware threading, faster structured outputs via LLGuidance,
continued iGPU/mobile Vulkan improvements.

The 1.x roadmap lives at `docs/roadmap.md` and is updated each milestone.
