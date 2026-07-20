# Changelog

All notable changes are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.7.0] — 2026-07-20

**ABI minor bump: `0x00010700`. No new symbols — the bump covers backward-compatible JSON field additions to existing describe/error payloads.**

### Added
- `serializes_decode` (bool) in `helix_runtime_describe()` — whether the active backend serialises decodes that share a model through a single GPU queue (CUDA/ROCm). Lets callers decide whether to mirror the serialisation with a decode gate without string-matching backend names.
- `model_hash` (string) in `helix_model_describe()` — a stable 64-bit FNV-1a content hash of the loaded model's identity (arch, params, size, vocab, layers, embd, trained ctx), for replay/audit chains.
- `requested` / `limit` (integers) on the `context_length_exceeded` error envelope — the token count that did not fit and the context size, surfaced from the chat prefill and `effective_max_tokens` paths. Omitted when unknown.

### Fixed (Rust wrapper)
- Re-added the `reasoning_budget` field to `ChatCompletionRequest` (dropped in the 1.6 wrapper); the C ABI already accepts it as a chat request extension.
- `helix::Error::ContextFull` now carries `requested` / `limit` (`Option<u32>`).

---

## [1.6.0] — 2026-07-18

**ABI minor bump: `0x00010600`. No new symbols — the bump covers new model/session options and a new response extension object.** Implements F3 and F6 of the 1.4 feature proposal.

### Added
- `lora` model option (F6) — load LoRA adapter GGUFs with the base model: `"lora": [{"path": "...", "scale": 1.0, "name": "support"}]`. Adapters are validated at load (missing file → `HELIX_E_MODEL_NOT_FOUND`; wrong base model → `HELIX_E_MODEL_LOAD_FAILED`; aLoRA → `HELIX_E_UNSUPPORTED_FEATURE`), owned by the model, and listed in `helix_model_describe`. Names default to the adapter file stem and must be unique.
- `lora` session option (F6) — select which loaded adapters a session activates: absent = all adapters at load-time scales; `[{"name": "support", "scale": 0.8}]` = exactly those (`[]` = none). Unknown names → `HELIX_E_VALIDATION`. Active adapters are fixed for the session's lifetime (an adapter change is a new session — the hot path and prefix cache are untouched) and listed in `helix_session_describe`. Honest v1 exclusions (`HELIX_E_UNSUPPORTED_FEATURE`): embedding sessions, speculative (MTP) sessions, and `helix_session_save`/`restore` while adapters are active.
- `context_shift` session option — **actually implemented** this time (documented since 1.1, honestly rejected in 1.3–1.5). When `true`, chat requests that hit the context wall recover instead of failing:
  - **Prefill overflow**: the oldest non-system history messages are dropped and the prompt is re-rendered (template-aware — messages are never split; the leading system/developer block and the last user message onward are never dropped; assistant tool-calls are dropped together with their tool results). Fails with `HELIX_E_CONTEXT_FULL` (param `messages`) only when even system + latest user message cannot fit.
  - **Generation overflow**: the oldest KV cells after the system-prompt prefix are evicted in blocks (discard-half, llama-server strategy) and the remaining cells are rope-shifted (`llama_memory_seq_rm` + `seq_add`); decoding continues. The system prefix is pinned via token-LCP against a system-only render.
  - **Reporting**: responses carry `"helix": {"context_shifted": true, "evicted_tokens": N}` when history was lost; streaming emits a dedicated extension chunk (empty `choices[]`, before the usage chunk) regardless of `include_usage`.
  - The prefix cache stays sound across shifts (the cached token history mirrors the post-eviction KV exactly).
  - Honest rejections: embedding sessions, MTP sessions, and models whose KV cannot be position-shifted (`llama_memory_can_shift() == false`, e.g. M-RoPE (Qwen3.5/Qwen3-VL), SWA, or hybrid attention) at session create; image/audio requests and `n > 1` per request.
- `helix_session_describe` now reports `"context_shift"`.

### Changed
- With `context_shift` on, `max_tokens`/`max_completion_tokens` are no longer capped by the context wall (generation continues via eviction); an unset budget is bounded at one full context of new tokens.

---

## [1.5.0] — 2026-07-17

**ABI minor bump: `0x00010500`, new linker version node `HELIX_1.5` (26 → 29 exported functions), new status code `HELIX_E_STATE_MISMATCH` (-12). All prior nodes are unchanged.** Implements F4 and F5 of the 1.4 feature proposal.

### Added
- `helix_session_save` / `helix_session_restore` — persist a chat session's KV cache and token history across process restarts; after restore, the next request's prefix-cache match skips re-prefilling the saved conversation. The file embeds a model-identity fingerprint, `n_ctx`, and a format version; restore fails honestly with the new `HELIX_E_STATE_MISMATCH` (-12) on a different model, different `n_ctx`, incompatible format, or a truncated/corrupt file — and always leaves the session empty and usable on failure. Saves are atomic (temp file + rename). Requires a chat session with the prefix cache enabled (embedding, MTP, and `prefix_cache: false` sessions are rejected with `HELIX_E_UNSUPPORTED_FEATURE`). **State files are trusted input** — never restore files from untrusted sources.
- `helix_rerank` — on-device reranking in the de-facto `/v1/rerank` shape (Cohere/Jina, llama-server). Requires a session created with `{"embedding": true, "pooling": "rank"}` on a reranker-head model (bge-reranker via classifier head, Qwen3-Reranker via `rerank` chat template — both pair formats supported, matching llama-server). `results[]` sorted by raw score descending (ties keep input order), optional `top_n`, usage token accounting, flush-boundary cancellation, `HELIX_E_CONTEXT_FULL` (param `documents`) for oversized pairs. `pooling: "rank"` sessions are no longer rejected at creation; `helix_embeddings` on a rerank session (and vice versa) returns `HELIX_E_UNSUPPORTED_FEATURE`. `helix_model_embedding_dim` still returns 0 for rank heads.
- `helix_session_describe` now reports `"rerank"`.

---

## [1.4.0] — 2026-07-17

**ABI minor bump: `0x00010400`, new linker version node `HELIX_1.4` (23 → 26 exported functions). All prior nodes are unchanged.** Implements F1, F2, and F7 of the 1.4 feature proposal.

### Added
- `helix_count_tokens` — token count a chat request would occupy after chat-template rendering (equals the `usage.prompt_tokens` a real request would report). Accepts the same body as `helix_chat_completions`; generation params are ignored. Pure query: no KV-cache access, safe concurrently with an in-flight request. Requests with image/audio content parts return `HELIX_E_UNSUPPORTED_FEATURE` in this release.
- `helix_tokenize` — raw tokenizer access on a loaded model. Returns a malloc'd JSON array of token ids (`"[1,2,3]"`, freed with `helix_free`); `add_special` / `parse_special` mirror the engine's tokenizer flags.
- `helix_model_load_ex` + `helix_load_progress_cb` — model loading with progress reporting (`[0.0, 1.0]`, monotonic, final `1.0` guaranteed) and cooperative cancellation: returning non-zero from the callback aborts the load with `HELIX_E_CANCELLED`. `cb = NULL` is exactly `helix_model_load`.
- Session options `cache_type_k` / `cache_type_v` — main-context KV-cache quantization (previously only available for the MTP draft context under `speculative{}`). Same value set (`f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1`, `f32`); `q8_0` roughly halves KV VRAM. A quantized `cache_type_v` requires flash attention and is rejected with `HELIX_E_VALIDATION` when the hardware profile disables it. Docs recommend `q8_0` (measurable quality cost at `q4_0`).
- `helix_session_describe` now reports `flash_attn`, `cache_type_k`, and `cache_type_v`.

---

## [Unreleased]

Fixes from the 2026-07-04 code review (all 12 findings + minor observations).

### Added
- `repeat_penalty` chat request extension (default `1.0` = disabled). Previously a hidden 1.1 repeat penalty over a 64-token window was applied to every request; all-default requests now produce OpenAI-neutral logits.
- `reasoning_budget` chat request extension is validated (`>= -1`) and documented in `helix.h` and the user guide (§4.4), including the budget-expiry transition message.
- `reasoning_content` on assistant history messages is parsed and passed through to the chat template (was silently dropped).

### Changed
- Stop strings: truncation now happens at the **first** occurrence, and both streaming and non-streaming emission hold back any output suffix that could be an in-progress stop match — stop text (or its prefix) can no longer leak into responses when a stop sequence spans token boundaries.
- Non-streaming responses keep assistant `content` alongside `tool_calls` (matching the streaming path); `content` is `null` only when empty.
- `system_fingerprint` and `helix_version_string()` both derive from the `HELIX_ABI_VERSION_*` macros (the fingerprint previously reported a stale `helix-0.3.0`).
- Session options are strictly typed: a present-but-mistyped option (e.g. `{"n_ctx": "4096"}`) is now rejected instead of silently ignored.
- Windows `file://` jail resolves junctions/symlinks via `GetFinalPathNameByHandleW` on the opened handle, compares the `HELIX_FILE_URI_ROOT` prefix case-insensitively, and rejects drive-relative paths (`C:foo`).
- MTP empty-draft fallback builds its single-token batch with per-token `seq_id`s (contract required by `common_speculative_process`) and checks the process return value.
- Cancellation semantics per request path are documented at `helix_session_cancel` in `helix.h`.
- `phase` markers in `helix_runtime_describe`/`helix_model_describe` share one definition (they previously disagreed: 5 vs 7).

### Removed
- `context_shift` session option: documented since 1.1 but never implemented. An explicit `true` is now rejected with `HELIX_E_UNSUPPORTED_FEATURE`; `false` (the actual behaviour) remains accepted. Removed from the user guide and Node wrapper types.
- Dead `src/util/override.hpp` and its test (four-layer override resolution was never wired into production).
- Unused `PowrProf.lib` link pragma in the Windows CPU probe.

### Fixed
- Nameless tools are rejected on both the `tools[]` path (`{"type":"function"}` with no function object) and the deprecated `functions[]` path.
- Reasoning-budget: forced-open detection and the budget sampler now agree on the thinking start tag (both prefer the template-supplied tag); when the template reports no start tag the sampler derives one from the end tag instead of staying silently inert.
- `token_matcher` in the reasoning-budget sampler uses a KMP-style fallback so multi-token tags with repeated prefixes are matched.
- Prefix-cache token history is no longer copied when the prefix cache is disabled.
- `%llu`/`uint64_t` format mismatch in the Linux RAM probe (`SCNu64`).
- Integration test `StreamUsageChunk` passed malformed JSON (a brace-less fragment) to `json::parse` and threw before reaching the library; it now runs and passes.

---

## [1.1.0] — Unreleased

**ABI minor bump: `0x00010100`, new linker version node `HELIX_1.1` (16 → 18 exported functions). The `HELIX_1.0` node is unchanged.** Implements HELIX-IMPL-001.

### Added
- `helix_embeddings` — OpenAI-isomorphic embeddings endpoint. Requires a session created with the new `{"embedding": true}` session option; supports string or array `input` (1..2048), `encoding_format: "float" | "base64"`; vectors are L2-normalized; `data[]` order matches input order; flush-boundary cancellation via `helix_session_cancel`.
- `helix_model_embedding_dim` — pooled output dimension query for dimension negotiation; returns 0 for models that cannot serve embeddings (encoder-decoder, reranker/classifier heads). NULL-safe pure query.
- Session options `"embedding"` (bool) and `"pooling"` (`"none" | "mean" | "cls" | "last" | "rank"`). For embedding sessions, `prefix_cache`/`context_shift` are forced off; `seed`/`stream_coalesce_ms` are ignored.
- `n_embd_out` field in `helix_model_describe()` JSON (matches `helix_model_embedding_dim`).
- Embedding-aware auto-tune selectors (`pick_n_batch_embed`, `pick_n_threads_embed`); embedding sessions force `n_ubatch == n_batch` (non-causal attention).
- Integration fixture `HelixEmbedFixture` (`HELIX_TEST_EMBED_MODEL` / `HELIX_TEST_EMBED_ALIAS` env vars); CI runs it with nomic-embed-text-v1.5 Q8_0 on the CPU job.

### Changed
- `HELIX_E_CONTEXT_FULL` can now carry param `input` (single embedding input exceeds the session batch size). Chat requests keep param `max_tokens`.
- Homebrew formula ABI assertion is now ranged (`>= 0x00010100 && < 0x00020000`) so future 1.x releases don't need a formula edit.
- User guide §3.2: corrected stale ABI version text to the `(major << 16) | (minor << 8) | patch` encoding.

---

## [1.0.0] — 2026-05-31

**GA release. ABI frozen for 18 months.**  
See [`RELEASE-NOTES-1.0.md`](RELEASE-NOTES-1.0.md) for the full narrative.

### Added
- `HELIX_ABI_VERSION` encoded as `(major × 65536 + minor)` — machine-comparable.
- Compile-time `_Static_assert(HELIX_ABI_VERSION == 0x00010000, …)` in `helix.h`.
- Forward-compatibility contracts documented on `helix_status_t` and `helix_log_level_t` enums.
- Linux versioned symbol node `HELIX_1.0` in `helix.ld` (replaces `HELIX_0.1`).
- macOS `LC_ID_DYLIB` compatibility version `1.0.0` / current version `1.0.0`.
- SOVERSION bumped to 1 (`libhelix.so.1`).
- CMake package config: `find_package(helix 1.0 REQUIRED)` now works.
- `helix-config.cmake.in`, `helix-config-version.cmake` (SameMajorVersion policy).
- `GNUInstallDirs`-aware install paths throughout.
- `ci/abi-check.yml` — ABI gate enforced on every PR from this tag forward.
- `ci/release.yml` — automated cross-platform build, sign, SBOM, and publish.
- Debian packaging (`debian/` directory).
- Homebrew formula (`packaging/helix.rb`).
- Chocolatey package (`packaging/helix.nuspec` + install scripts).
- Benchmark report `benchmarks/REPORT-1.0.0.md` and raw CSV.
- `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`.
- `docs/abi-policy.md` — 18-month ABI commitment, versioning contract.
- `docs/migration/0.8-to-1.0.md` — migration guide for RC users.
- MkDocs Material site (`mkdocs.yml`).
- `RELEASE-NOTES-1.0.md` — long-form release narrative.

### Changed
- `HELIX_ABI_VERSION_MAJOR` 0 → 1, `MINOR` 1 → 0.
- `helix_version_string()` returns `"1.0.0+llama.cpp-<sha>"`.
- `helix.pc.in` description updated to "OpenAI-isomorphic on-device LLM library".
- `helix.pc.in` uses `@CMAKE_INSTALL_LIBDIR@` / `@CMAKE_INSTALL_INCLUDEDIR@`.

### Removed
- Nothing. All 16 public symbols from `HELIX_0.1` are carried forward unchanged.

---

## [0.8.0-rc.1] — 2026-04-15

Release candidate shipped to alpha testers. Language wrappers for Python,
Rust, Swift, Node, Go, and .NET. Full multimodal + reasoning support.

---

## [0.7.0] — 2026-03-01

Phase 7: multimodal (vision) and reasoning (chain-of-thought extraction).

## [0.6.0] — 2026-01-15

Phase 6: structured outputs, logit_bias, logprobs.

## [0.5.0] — 2025-12-01

Phase 5: GPU backends (CUDA, Metal, Vulkan, ROCm, omni).

## [0.4.0] — 2025-10-15

Phase 4: hardware profiler and auto-tuner.

## [0.3.0] — 2025-09-01

Phase 3: tool calling (function calling).

## [0.2.0] — 2025-07-15

Phase 2: streaming, cancellation, prefix-cache.

## [0.1.0] — 2025-06-01

Phase 1: initial skeleton, non-streaming chat completions, 16-symbol ABI.
