# Refactor Plan

This plan comes from a full review of `src/`, `tools/` and the build files. It
lists duplicate and near-duplicate code worth consolidating. None of these
items change behaviour, so each one can land as its own PR, verified by the
existing unit and integration suites. Bug fixes from the same review have
already been applied separately; this document only covers structure.

Items are ordered by payoff. Payoff means how much code goes away, combined
with how likely the copies are to drift apart. Line numbers refer to the
tree at the time of the review.

## Status: all items done (2026-10-14)

Every item below has been applied. Each step was checked against golden
outputs captured before the refactor started: exact streamed chunk sequences,
non-streaming bodies, logprobs, tool calls, context shift, embeddings and
rerank on CPU (Qwen2.5-0.5B, bge-reranker-v2-m3), plus thinking, reasoning
budget, tools and MTP speculative decoding on ROCm (Qwen3.5-9B). There are no
semantic differences, and chunk sequences are byte-identical wherever
coalescing is off.

Where the work departed from the proposals:

- **1:** `TokenEmitter` lives in `decode_loop.cpp` instead of its own file,
  because it shares `route_diffs` and the tool-call ID RNG with the rest of
  that file. It has a two-phase API (`append()`, then `emit()`) because
  `decode_one` feeds each token back into the KV cache between the stop
  checks and streaming, and that order decides what a cancel leaves in the
  cache.
- **4:** `throw_unsupported()` now uses its message verbatim. It used to
  append " is not supported in this build" to messages that were already
  complete sentences. Two error `type` strings were unified:
  `HELIX_E_MODEL_LOAD_FAILED` always reports `model_load_failed` (the main
  load failure said `model_load_error`), and `HELIX_E_CANCELLED` always
  reports `request_cancelled` (two sites said `cancel`). The embedding and
  rerank `HELIX_E_CONTEXT_FULL` errors now include `requested` / `limit`.
- **3:** `state.cpp` never included `cuda_lock.hpp`, so
  `#if HELIX_CUDA_REQUIRES_GLOBAL_LOCK` was always 0 there, and save/restore
  skipped the model-wide GPU lock on CUDA/ROCm builds. `CallGuard` is defined
  in `session.hpp`, which includes that header, so this is fixed.
- **2:** While verifying this item, an existing crash turned up and was
  fixed. Embedding sessions with `n_batch` < 64 aborted the process inside
  llama.cpp, because `n_seq_max` exceeded the output capacity. It is now
  capped at `n_batch`.
- **7/8:** The shared parser is `parse_json_object()` in
  `util/json_parse.hpp`, used by requests and all three option parsers.
  Request parse errors now read "request body JSON parse error: ...", and
  model options reject non-object roots like the other parsers. The LoRA
  entry parsers were **not** merged: the model side requires `path`, derives
  the name from the file stem and restricts its charset, while the session
  side requires `name`. Only the array/scale checks overlap, and a merged
  version would need a branch for nearly every line.
- **14:** Implemented as a `helix_configure_target()` function, not an
  `INTERFACE` library, so the exported `helix` target picks up no extra
  dependency. The test target reuses `HELIX_PRIVATE_INCLUDE_DIRS` but keeps
  its own CPU-only definitions.
- **15:** The Swift header is synced. Because Windows checkouts often lack
  symlink support, `abi-check.yml` now runs `diff -u` on the two copies
  instead.

---

## P1: High payoff

### 1. Token emission shared by `decode_one` and `decode_one_speculative`

**Where:** `src/engine/decode_loop.cpp`

| Block | `decode_one` | `decode_one_speculative` |
|---|---|---|
| PEG parser setup | ~234–253 | ~623–640 |
| Per-token / coalesced content emission | ~401–483 | ~696–768 (`emit_one` lambda) |
| Final flush (PEG final parse or extractor flush + `lp_pending`) | ~486–537 | ~885–928 |
| Reasoning-token re-tokenisation | ~546–551 | ~930–935 |

About 200 lines are near-identical. The two copies already have to be kept in
sync by hand: the comments read "same as decode_one" and "see decode_one". The
streaming-logprobs fix in `52d690d` had to be applied to both.

**Proposal:** extract a `TokenEmitter` (or `GenerationState`) class into
`src/engine/token_emitter.{hpp,cpp}`. It owns:
`output`, `gen_tokens`, `logprob_entries`, `lp_pending`, `emitted_up_to`,
`stop_scan`, `last_emit`, `ReasoningExtractor`, the PEG arena/params/prev
message, `tool_calls_started` and `reasoning_text`.

```cpp
class TokenEmitter {
public:
    TokenEmitter(EventSink&, int choice, const RenderResult&, const LogprobsConfig&,
                 const std::vector<std::string>& stops, int max_new, int coalesce_ms,
                 llama_context*, std::atomic<bool>& cancel);
    /* Returns true when generation must stop (EOG / stop / length / cancel). */
    bool push(llama_token tk, const float* logits);
    DecodeResult finish();          /* final flush + reasoning token count */
};
```

After the extraction, each decoder only handles KV and batch management:
prefill, context shift, and draft/verify. `emit_one` and the inline emission
block both become `emitter.push(...)`.

### 2. Embedding and rerank pipelines

**Where:** `src/engine/embed_loop.cpp`, `run_embeddings` (~85–215) and `run_rerank` (~216–395).

Shared steps: request alias check; tokenise-per-item with the
empty-input and `> n_batch` guards; `plan_embed_flushes`; batch init plus
`BatchGuard`; a per-flush cancel check; batch building; the `rc == 2` / `rc != 0`
mapping; the per-sequence `llama_get_embeddings_seq` null check; and the
malloc'd response copy.

**Proposal:**

```cpp
/* Runs every flush; calls on_seq(input_index, const float* pooled) per sequence. */
void run_packed_flushes(Session&, const std::vector<std::vector<llama_token>>& toks,
                        const char* what /* "embedding" | "rerank" */,
                        const std::function<void(size_t, const float*)>& on_seq);
```

After this, each endpoint only builds its token lists and consumes its
outputs: normalising vectors for embeddings, reading `emb[0]` for rerank.

### 3. Session endpoint preamble (locks + in-flight marker)

**Where:** `src/session/session.cpp` (`do_chat_completions`, `embeddings`,
`rerank`) and `src/session/state.cpp` (`save_state`, `restore_state`).

Five copies of the `#if HELIX_CUDA_REQUIRES_GLOBAL_LOCK` model lock followed
by the session `mu_` lock, and three of them then construct the
`Session::InFlight` marker (which replaced the old per-function
`CancelResetter` structs). The lock order (cuda_mu → mu_ → InFlight) is a
correctness invariant, and right now each copy enforces it separately.

**Proposal:** one private RAII type that encodes the ordering once:

```cpp
class Session::CallGuard {
#if HELIX_CUDA_REQUIRES_GLOBAL_LOCK
    std::lock_guard<std::mutex> cuda_;
#endif
    std::lock_guard<std::mutex> session_;
    std::optional<InFlight>     in_flight_;   /* empty for save/restore */
};
```

### 4. Error construction boilerplate

**Where:** across `src/`. Representative counts:

- `throw Error(HELIX_E_VALIDATION, …, "invalid_request_error", key, "helix_e_validation")`
  is written out 27 times in `session/options.cpp`. `throw_validation(msg, param)`
  already exists in `internal/error.hpp` and produces the same `Error`.
- `throw Error(HELIX_E_UNSUPPORTED_FEATURE, msg, "unsupported_feature_error", …)`
  appears about 20 times (`session.cpp`, `state.cpp`, `request.cpp`, `model.cpp`).
  These can't use `throw_unsupported` because it appends
  `" is not supported in this build"`.
- `HELIX_E_BACKEND` appears about 12 times, `HELIX_E_MODEL_NOT_FOUND` 5 times and
  `HELIX_E_CANCELLED` 6 times, each with the same type/code string pair.
- `decode_loop.cpp` (~1137–1148) builds a `HELIX_E_CONTEXT_FULL` `Error` and
  sets `requested`/`limit` by hand. `throw_context_full(requested, limit, msg)`
  does exactly this, except it hard-codes `param = "max_tokens"`.

**Proposal:** add `throw_unsupported_msg(msg, param)`, `throw_backend(msg, param = "")`,
`throw_model_not_found(msg, param)` and `throw_cancelled(msg)`. Give
`throw_context_full` a `param` argument. Then replace the long-hand sites.
The status/type/code triples will then live only in `error.hpp`, so they
can't diverge.

---

## P2: Medium payoff

### 5. "Malloc a copy for the caller" pattern

**Where:** `abi/helix_abi.cpp` (`describe_copy_impl` ~115, `helix_tokenize`
~387), `engine/decode_loop.cpp` (~1446), `engine/embed_loop.cpp` (~210, ~389).

Five copies of malloc + copy + "must stay paired with helix_free" comments.

**Proposal:** add `char* helix::dup_for_caller(const std::string&)` (throws
`bad_alloc`) plus a non-throwing variant for the describe paths. Keep it in one
file next to `helix_free`, so the malloc/free pairing invariant is stated once.

### 6. Model-alias check

**Where:** `decode_loop.cpp` (~963, ~1468) and `embed_loop.cpp` (~100, ~231).
Four identical `HELIX_E_MODEL_NOT_FOUND` blocks.
**Proposal:** add `void require_model_alias(const Session&, const std::string& requested)`.

### 7. Request-body preamble

**Where:** `json/request.cpp`, `ChatRequest::from_json`,
`EmbeddingsRequest::from_json` and `RerankRequest::from_json`. Each one does
the same parse → `parse_error` → INVALID_JSON, then "must be an object", then
"model is a required string".
**Proposal:** add `static json parse_request_object(const std::string&)` and
`static std::string require_string(const json&, const char* key)`.
`require_string` would also cover `query`.

### 8. Session/model option parsing

**Where:** `session/options.cpp`, `model/model.cpp`, `runtime/runtime.cpp`.

- The `speculative` sub-object (options.cpp ~200–275) re-implements the
  typed `get_*_opt` lambdas field by field, with a hand-prefixed param name.
  Making the lambdas take `(const json& obj, const std::string& path)` would
  let both levels share them.
- LoRA entry parsing (name/scale validation, duplicate-name detection) exists
  in both `parse_model_options` (model.cpp ~93–137) and `parse_session_options`
  (options.cpp ~150–190). These could become one `parse_lora_entries(json, require_path)`.
- All three option parsers have the same "parse → object check" prologue,
  which can share the helper from item 7.

### 9. Chat-request helpers inside `run_chat_completions`

**Where:** `engine/decode_loop.cpp`.

- The "is this part media?" test (`type == "image_url" || type == "input_audio"`)
  appears 3 times (~972, ~1033, ~1476). Add `bool ContentPart::is_media() const`.
- The leading system/developer prefix (`sys_end`) is computed twice (~1102, ~1267).
  Add `size_t leading_system_count(const std::vector<Message>&)`.
- `render_prompt(...)` is called 4 times with the same 7 trailing arguments
  (~1013, ~1130, ~1274, ~1488). A small `RenderArgs` built once from the request
  and model would remove the risk of one call site diverging. That risk is
  real: `count_tokens` must render exactly as the chat path does.

### 10. Thinking-tag derivation

**Where:** `sampling/chain.cpp` (~145–160) versus
`resolve_thinking_start_tag()` in `chat/reasoning.cpp`.
`template.cpp` and the reasoning extractor now share
`resolve_thinking_start_tag()`, but the budget sampler in `chain.cpp` still
has its own copy of the "drop the `/` from the end tag" rule (it differs only
in skipping derivation when the block is forced open and in logging a
warning when there is no `/`). The comment in `template.cpp` notes that the
budget sampler and the forced-open detection "must agree".
**Proposal:** call `resolve_thinking_start_tag()` from `chain.cpp` too, keeping
the forced-open guard and the warning at the call site.

### 11. Session construction cleanup paths

**Where:** `session/session.cpp`. `llama_free(ctx_); ctx_ = nullptr;` appears on
4 constructor error paths (~295, ~314, ~359, ~462). The MTP teardown
(`spec_mtp_` → `ctx_mtp_` → `model_dft_`) is duplicated between the
constructor's catch (~455) and the destructor (~482).
**Proposal:** wrap the constructor body after context creation in one
`try { … } catch (...) { release_all(); throw; }` and add a private
`release_all()` that the destructor also uses. Holding the handles in
`std::unique_ptr` with custom deleters is the more idiomatic alternative,
and it would also make `Model` leak-proof by construction (see the
model-load leak fixed in this review).

### 12. Thread-count heuristics

**Where:** `hardware/profile.cpp`.

- The partial-GPU-offload scaling branch in `pick_n_threads` and
  `pick_n_threads_batch` is the same apart from the base core count.
  It can become `scale_for_offload(base, hw, m)`.
- `pick_n_threads_batch` contains a dead conditional:
  `if (c.efficiency_cores > 0) return c.logical_cores; return c.logical_cores;`.

### 13. `DecodeResult` cancelled early-returns

**Where:** `decode_loop.cpp`. `return {"", "stop", 0, 0, true, {}};` appears 7
times (~206–231, ~586–621).
**Proposal:** add `static DecodeResult DecodeResult::cancelled()`. This also stops
the positional brace-init from silently shifting if a field is added.

---

## P3: Build and packaging

### 14. CMake target configuration

**Where:** `CMakeLists.txt`, `tests/CMakeLists.txt`.

The include-directory list, `HELIX_LLAMACPP_SHA` / `HELIX_BACKEND_*` /
`HELIX_MAX_DECODE_THREADS` definitions, warning flags, and Windows
`NOMINMAX`/`WIN32_LEAN_AND_MEAN` are repeated for `helix`, `helix_static` and
the test target. The backend → `HELIX_HAS_*` mapping is spelled out twice.
That duplication already caused a bug, now fixed: `helix_static` was missing
`HELIX_HAS_ROCM`, so the static ROCm library compiled out the per-model
decode lock.
**Proposal:** add an `INTERFACE` library (`helix_build_settings`) carrying the
includes, definitions and flags, linked PRIVATE by all three targets. Or add a
`helix_configure_target(<tgt>)` function.

### 15. Swift wrapper header copy

**Where:** `wrappers/swift/Sources/CHelix/helix.h`. This is a hand-maintained
copy of `helix.h` and has gone stale: it is frozen at ABI 1.0
(`HELIX_ABI_VERSION_MINOR 0`) and misses every symbol added since 1.1.
**Proposal:** generate it from the root `helix.h` at package time, or make it
a symlink, and add a CI check that diffs the two.

### 16. Win32 UTF-8 ↔ UTF-16 conversion

**Where:** `multimodal/image_decode.cpp` (`widen_utf8`) and
`hardware/cpu_windows.cpp` (inline `WideCharToMultiByte`).
**Proposal:** add `src/util/win_utf8.hpp` with `widen_utf8` / `narrow_utf8`.
This review added `src/util/fs_path.hpp` for the `std::filesystem` side, and
the conversion helpers would naturally sit next to it.

---

## Suggested sequencing

1. Items 4, 5, 6, 13. Mechanical and low-risk, and they shrink the diff noise for
   everything that follows.
2. Item 3 (session call guard), then item 11 (constructor cleanup).
3. Item 1 (`TokenEmitter`). This is the largest item. Do it with the
   integration suite plus a streaming + logprobs + tools + MTP run before
   and after, and diff the emitted chunk sequences.
4. Item 2 (embed/rerank flush loop).
5. Items 7–10, 12.
6. Items 14–16 (build/packaging). These are independent of the source work.
