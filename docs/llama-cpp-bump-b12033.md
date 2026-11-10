# The llama.cpp bump to b12033: the plan, and the log of what it got wrong

**Date:** 2026-11-08
**Status:** Phases 0–5 done on `deps/llama-b12033`. The plan was written against `b12211`; the target
became `b12033` (see the log). The log directly below records every correction made while doing the
work; where it disagrees with the body, the log wins.

> ### Phase 0 log (2026-11-08)
>
> - **Decisions (§9):** all five taken as recommended. Fork branch `helix-b12033`; upstream the patch
>   after Phase 3; D2 straight after, as its own PR; no `LLAMA_LOAD_MODE_AUTO`.
> - **Target: `b12033` = `785322d17`** (2026-11-01, "requirements : fix numpy dependencies (#31508)").
>   That is the newest tag at least a week old; `b12034` is 2026-11-02. The distance from the fork's base
>   is **2,394 commits**, 178 fewer than `b12211`. It contains every upstream change this plan relies
>   on: #25544, #31374 (budget matcher), #30210, #30389 (chat API), #30882, #30922 (Gemma 4 vision),
>   `llama_load_mode`, `common_batch`, `common_fit_extra_model`, `mtmd_helper_init_opt_default`. It
>   does **not** contain #31865 (the mtmd consolidation, which is why the tip was avoided), #30141 (MTP
>   hidden-state save and restore), #31356 or #30280. So §13's gains list loses #30141.
> - **Correction: `2ee83d0c3` is *not* superseded by upstream on its own.** At both `b12033` and
>   `b12211`, upstream's `llama_memory_seq_rm` calls in draft-MTP (`speculative.cpp:1708/1807` at
>   `b12033`, `:1747/1858` at `b12211`) are guarded by **`chain_heads`** (`n_mtp_layers > 1 &&
>   !is_mem_shared`), not by `!is_mem_shared`. A **single-head, separate-KV** model, which is the
>   `else` branch of `draft()`, still writes the draft positions `pos0..pos0+k` into `ctx_dft` and
>   never removes them. Upstream moved that duty to the **caller**: `speculative-simple.cpp:211`
>   trims `ctx_dft` from `ckpt.pos_max + 1` after `common_speculative_draft()` and before verifying,
>   and `:333` trims both contexts to `n_past` after acceptance. The server does the same through
>   `common_memory::seq_rm`. helix's decode loop trims only the target (`decode_loop.cpp:723, 758`).
>   Dropping the patch with no helix change would bring the bug back on Qwythos. **Plan: drop the
>   patch anyway**, so the fork still carries one, and add the two caller-side trims of `ctx_mtp_` to
>   §4c. The §6 MTP test, two consecutive partial accepts on the separate-KV model, becomes the gate
>   that proves it.
> - **D1 done** on helix branch `fix/llama-commit-marker` (`92912a9`, not pushed). The marker is
>   corrected; configure reads the submodule's HEAD, falls back to the marker, and warns when they
>   differ; a CI job `llama-pin` checks the marker against the gitlink. Both paths were checked to
>   fail and to pass.
> - **Phase 1 done locally:** fork branch `helix-b12033` = `a0dec274e` (b12033 + the port), made in a
>   worktree of the submodule repo; not pushed. **Phases 2–4 edits are written** on helix branch
>   `deps/llama-b12033` (worktree in the session scratchpad; commits `d19805d` pin, `8695fb5` D3), and
>   not yet compiled: the build waits until the GPU baselines finish.
> - **More corrections found while porting:**
>   - `llama_model_default_params().load_mode` is now **`LLAMA_LOAD_MODE_AUTO`**. §4a lists only the
>     two sites that set `use_mmap`. `gpu_layer_select.cpp`'s fit probe never set them, so without a
>     change it would silently have gone `AUTO`. All three sites now set the mode explicitly, through
>     one `load_mode_from()` (`src/model/load_mode.hpp`).
>   - **helix has no MTP integration tests**, and **nothing reports an acceptance rate** (not
>     `helix_session_describe`, not the log). Phase 4's gate as written has nothing to run. Added
>     instead: Wren's `mtp-probe` (plain vs `draft-mtp` greedy on both layouts, text equality and
>     speedup). It links whichever DLL `HELIX_LIB_DIR` names, so it measures the old and the new
>     library. The decode loop now also logs `mtp: N rounds, D drafted, A accepted (%)` at info.
>   - **D3 resolved:** nothing includes `third_party/nlohmann_json.hpp`. Every include is
>     `"nlohmann/json.hpp"`, which resolves only to llama.cpp's `vendor/` copy, so one JSON ABI was
>     linked all along. The file is deleted (`8695fb5`).
>   - `prefill-probe`'s last round ("withdrawing the tool menu") overflows `n_ctx` 8192 on the 0.5B
>     default (8,335 tokens). The baseline is re-run at `WREN_N_CTX=16384`, on the 0.5B and on Wren's
>     12B, and the after-run must use the same setting.
> - **Phases 2–5, 2026-11-08 evening.** Branch `deps/llama-b12033` is at `0ef94df` (pin, D3, port,
>   tests, run-time fixes).
>   - **No second wave of compile errors.** CPU 331/331 and Vulkan 655/655 built on the first try.
>     Unit tests: 501 of 501 pass, 7 of them new. With the ported fork patch reverted, the GBNF test
>     fails with `root ::= "<|im_start|>assistant\n" …`, so upstream still has the bug at b12033.
>   - **Integration, CI models:** CPU 86 passed, 12 skipped. On Vulkan, `BatchEqualsSingles` fails,
>     and **it fails on the old library too** (up to 8.6e-4 against a 1e-5 tolerance). CI never runs
>     integration tests on Vulkan; this is a helix finding, not the bump.
>   - **Integration, Gemma 4 12B and Qwen3.5-9B:** the same new test binary was run against the old
>     and the new DLL (the export tables are identical: 29 symbols). The failure sets are **identical**
>     once the run-time fixes below are in. Every failure on both is a content assertion written for a
>     non-thinking 0.5B model.
>   - **Four changes compile cleanly and then fail at run time** (`0ef94df`), and none is in §4:
>     1. `load_mtp` (#26296) is opt-in and off by default. A draft-mtp session on a model with its
>        own MTP block **aborted the process**. Now set at all three load sites; `lazy_mode` is pinned
>        `OFF` at the same sites.
>     2. helix's dummy-token **warmup of the MTP context** is a `GGML_ASSERT` abort now, since an MTP
>        graph needs a state embedding. The warmup is removed.
>     3. **`mtmd_input_text.text_len`** is new, and helix left it uninitialised: `bad_alloc` on a
>        1×1 image, a segfault on a real PNG.
>     4. Gemma 4 images **cost more tokens** (#30922, no letterbox padding): 896 px went from 258 to
>        363 tokens, 1344×1008 is 588. Past `n_ubatch` (512) the decode was an abort; it is now a
>        validation error. **Open decision:** Wren's two larger sizes are refused where they used to
>        encode. Either cap image tokens (mtmd `image_max_tokens`) or raise `n_ubatch`.
>   - **MTP gate, via Wren's `mtp-probe`:** speedup 1.62× (Qwythos) and 1.79× (Gemma), against
>     1.62× and 1.81× before; acceptance 63–97% per request. **With the `ctx_mtp` trims disabled,
>     every Qwythos request fails with "inconsistent sequence positions"**, which confirms the
>     correction to `2ee83d0c3` above.
>   - **R5 cleared:** the worst per-text cosine, old against new, is 0.999971. No re-embed.
>   - **R8 is real: prefill on Wren's 12B is slower, and it reproduces.** The A/B was interleaved
>     old/new/old/new (`wren/handoff/baselines/llama-bump/ab/`). The two old runs agree to 0.2%:
>     cold 15.65 s → 19.85 s (+27%), cold long 25.08 s → 34.73 s (+38%), layout B long 5.13 s →
>     7.76 s (+51%). A reused prefix costs the same (876 against 882 ms), decode is unchanged, and
>     the 0.5B control is unchanged. The dream iteration is 16.2 s → 18.2 s, and the returning user's
>     wait 3.5 s → 4.5 s. **Ruled out:** context geometry, flash-attn mode, CPU ISA flags (AVX2 on
>     both), glslc features (coopmat on both, integer-dot on neither). **Not yet found.**
>   - **R8 localised (2026-11-09): it is upstream's Vulkan flash-attention path.** In pure llama.cpp
>     (`llama-bench`, Gemma 12B, both trees) the new library is *faster* with FA off: pp2048 373 → 405
>     tok/s, and 315 → 345 at depth 4096. With FA on it is slower: 305 → 270, and **219 → 146 at
>     depth 4096 (−33%)**. helix runs with FA on. Its policy says "Vulkan: off by default"
>     (`profile.cpp:151`), but `session.cpp` only ever sets `ENABLED` and otherwise leaves llama.cpp's
>     default `AUTO`, which resolves to enabled (*"Flash Attention was auto, set to enabled"*), while
>     helix logs `flash_attn=off`. **Setting `DISABLED` explicitly does not help helix:** at n_ctx
>     16384 with a full SWA cache, FA off is slower still (36.8 s against 23.3 s, new library), so
>     the change was reverted. The log line and the policy comment are wrong and should be fixed
>     separately. A `git bisect` of llama.cpp `ef8268fee..b12033` is running on FA prefill at depth
>     (good 225 tok/s, bad 156, threshold 190), worktree `lcbisect`.
>   - **R8 found and fixed: upstream #28105 (`fc82583e6`, sparse FA).** The bisect converged in 12
>     steps: every commit before it measures 224–225 tok/s, every one after 143. It routed every K/V
>     read in the *dense scalar* FA kernel through the sparse gather; RDNA2 has no coopmat, so it
>     runs that kernel. `GGML_VK_FA_SPARSE_DISABLE` does not help, because the cost is in the dense
>     kernel's code. Not fixed at b12211 (155.6). **The fork now carries a second patch,
>     `d007d8949`:** #28105's hunks reversed in `flash_attn.comp`, and the host never picks sparse
>     for `FA_SCALAR`, so a sparse mask there runs as masked dense attention, correct but slower for
>     DeepSeek V4 and GLM on non-coopmat GPUs. `test-backend-ops -o FLASH_ATTN_EXT -b Vulkan0`:
>     5368/5368 pass. Without the host guard, 26 of the 39 sparse cases give wrong results.
>     - llama-bench, FA on: pp2048 305 / 270 / **373**, and @d4096 219 / 146 / **263** (June base /
>       b12033 / b12033 + patch).
>     - helix cold prefill, 4,869 tokens, Gemma 12B, interleaved: **17.7 s old, 23.3 s unpatched,
>       14.7 s patched**, so 17% *faster* than before the bump.
>     - MTP fully offloaded: Gemma 2.73× → 2.74×; Qwythos 2.09× → 1.95×, and its speculative text is
>       now identical to plain greedy on all three prompts. Embedding drift and vision are unchanged.
>     - **Upstream it**: the sparse gather needs its own shader variant, not a spec-constant branch,
>       on a driver that does not fold it.
>   - **Wren gates on the patched library (2026-11-09):** headless 3/3; embed-model, chat-model and
>     three-model smokes 17/17 with no `FAIL` lines; walkthrough **102/102** and `--after-restart`,
>     after a Wren fix. `find_files` reranked the 260-character display snippet, which cut off the
>     word that decided the walkthrough's search, and it was a tie on the old library too. It now
>     reranks the chunk. Interleaved against the old library: dream iteration 2,574 → 2,202 ms, a
>     returning user's wait 73 → 38 ms, `llama-bench` pp512/tg64 with FA 596/57.5 → 733/59.8. The
>     machine was degraded for hours on 2026-11-08 (5 tok/s decode on both libraries) and recovered
>     by itself, so **no absolute number from the Phase 0 baseline files should be quoted**; only
>     interleaved pairs count. Wren's write-up is `handoff/decisions/N18-llama-cpp-b12033.md`.
>   - **Still open:** `cancel-probe` on the new library (the owner's go-ahead); the large-image
>     decision; the helix PRs, CI and release matrix; `vendor/helix` in Wren. The bisect worktree
>     `lcbisect` (+ `lcb-build`) can go once the patch is upstreamed.
>   - **The ~690 graph splits were a probe artefact, not this box.** (Decode really is about 5 tok/s
>     on this box: `llama-bench` tg64 is 5.4 with 49/49 layers and 2 splits, on both libraries.
>     That needs its own look.) A model
>     loaded without `n_ctx` gets its GPU plan fitted for the training context (262,144 for Gemma 4),
>     and helix offloads **6 of 48 layers** (`n_gpu_layers=all requested but only 6/48 layers fit`).
>     Wren's engine and the prefill, reference and vision probes pass `n_ctx` and are fully offloaded.
>     **`mtp-probe` and Wren's own `cancel-probe` do not**, so `cancel-probe`'s numbers (N14, and the
>     Phase 0 baseline) were measured on a mostly-CPU model. That is a Wren finding for N18, and a
>     helix question: should `"n_gpu_layers": "all"` with no `n_ctx` fit for the training context?
>   - **`cancel-probe` on the new library has not been run.** It is the R6 bugcheck risk, and it
>     waits for the machine owner.
> - **Wren baselines** go in `wren/handoff/baselines/llama-bump/`. `embed-probe` gained
>   `WREN_PROBE_DUMP` and `WREN_PROBE_COMPARE`, which save the raw vectors and compare them per text
>   (R5). The same-build noise floor is 1.000000, and a perturbed vector is caught (0.722).
**Supersedes for this bump:** `docs/llama-cpp-update-guide.md`, which is out of date. It describes
llama.cpp as a nested repo rather than a submodule and gives ABI 1.1. §10 updates it.

---

## 1. Summary

helix builds against a fork of llama.cpp. That fork is **upstream as of 2026-06-14 plus three helix
patches**. Upstream is now **2,572 commits ahead**, at build `b12211`.

A trial build against upstream `master` shows how big the bump is:

- **Upstream llama.cpp compiles cleanly on Windows** with helix's CMake settings. The CPU build got to
  step 277 of 283.
- **helix's own code fails in 9 files**: 31 error sites, 22 distinct errors.
  - About half are mechanical renames and signature changes.
  - The rest come from upstream refactoring its chat API twice. That is helix's tool-calling and
    reasoning path, and it is where behaviour can change without a compile error.
- **Two of the three fork patches are obsolete.** Upstream fixed both bugs in its own rewrites. The
  third (the generation-prompt prefix leaking into the grammar) is still needed. It is a one-line
  port and is already done in the trial.
- **helix's public C ABI can stay at 1.7.** Every change is internal. Wren's Rust wrapper should not
  need to change.

The work also turned up **one security improvement worth taking** (§7, D2). helix tokenizes the whole
rendered prompt with special-token parsing on. A user message, file or tool result containing a
template control token, such as Gemma's `<start_of_turn>`, is read as a real turn boundary. Upstream's
new prompt type marks which parts came from user input, which makes this fixable.

**Estimate:**
- helix: 1.5–2 days, including tests.
- Wren validation: about 1 day, mostly harness wall-clock time.
- Total: about 3 working days.

---

## 2. Where things stand

| | Value |
|---|---|
| Submodule | `third_party/llama.cpp` → `https://github.com/anemoi-ai/llama.cpp.git`, branch `helix-1.4` |
| Pinned commit | `2ee83d0c3` (`git describe`: `b9637-5-g2ee83d0c3`) |
| Upstream base of the fork | `ef8268fee` (2026-06-14) — "fix(ui): render thinking/reasoning block content as markdown (#24611)" |
| Upstream tip at time of writing | `7d7f02875` (2026-11-08), tag `b12211` — "mtmd: refactoring: remove clip.h, consolidate code into mtmd (#31865)" |
| Distance | 2,572 commits |
| `third_party/llama.cpp.commit` | `ff343dab6…`. **This is wrong**: it names the first fork patch, not the pin (see D1) |

### The three fork patches

| Fork commit | What it fixes | Upstream now | Plan |
|---|---|---|---|
| `ff343dab6` (07-17) | reasoning-budget `token_matcher` missed patterns with repeated prefixes (KMP backtracking) | Matcher replaced by Aho-Corasick over several end sequences (#25544, #31374), which handles overlapping prefixes by construction | **Drop** |
| `a5b839399` (07-19) | autoparser's generation-prompt prefix leaked into the GBNF as a required literal, forcing the model to re-emit the prompt tail under non-lazy grammars (`tool_choice: required`) | **Still present**: `common_chat_peg_builder::prefix()` still returns `literal(s)`, and `build_parser` still adds it unwrapped | **Port**: wrap in `p.gbnf(…, "")`. Done in the trial as `9a55bca57` |
| `2ee83d0c3` (07-23) | draft-MTP left speculative positions in the draft context's KV, so the catch-up decode and the next seed collided (fatal on M-RoPE models) | ~~now calls `llama_memory_seq_rm` … guarded by `!is_mem_shared`~~ **Only for `chain_heads`** (multi-head). For one head the trim is now the **caller's** job (`speculative-simple.cpp:211, 333`) | **Drop, and move the trim into helix's decode loop** (§4c). The regression test (§6) is the gate |

---

## 3. The trial build (evidence)

A scratch worktree of helix `main` at `bc71026` was used, with the submodule moved to upstream
`7d7f02875` and the one surviving patch ported. It was built with:

```
cmake -B build-cpu -G Ninja -DHELIX_BACKEND=cpu -DHELIX_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -- -k 0          # MSVC 14.44, VS 2022 x64 environment
```

The results:

- **Configure:** clean. The existing flags (`LLAMA_BUILD_COMMON`, `LLAMA_BUILD_TOOLS`, `MTMD_VIDEO OFF`,
  …) still exist, and the targets `llama`, `llama-common` and `mtmd` still have those names.
- **llama.cpp, ggml, common and mtmd:** all compiled.
- **helix:** 9 files failed (below), each built twice (`helix` and `helix_static`), three of them also
  in `helix_unit_tests`.
- **Test sources:** none failed.

MSVC reports only the first layer of errors. Fixing these will expose a second wave, mostly
downstream type mismatches, so budget for it.

Trial location, kept until the bump is done:
a scratch detached helix worktree (its submodule is on local branch
`helix-bump` = upstream `7d7f02875` + `9a55bca57`).

---

## 4. Every break, and its fix

Line numbers are helix `main` at `bc71026`.

### 4a. Mechanical (about half a day)

| Site | Error | Fix |
|---|---|---|
| `model/model.cpp:287-288`, `session/session.cpp:377-378` | `use_mmap` / `use_mlock` not members of `llama_model_params` | One field now: `enum llama_load_mode load_mode`. Map helix's two booleans exactly: both → `LLAMA_LOAD_MODE_MMAP_MLOCK`, mmap → `_MMAP`, mlock → `_MLOCK`, neither → `_NONE`. **Do not** adopt `_AUTO` in this bump: that is a behaviour change |
| `sampling/chain.cpp:68` | `llama_sampler_init_penalties` takes 5 args | New first argument `n_vocab`: pass `llama_vocab_n_tokens(vocab)` |
| `sampling/chain.cpp:180` | `common_reasoning_budget_init` wants `vector<llama_tokens>` for start and end | Now takes lists of start and end sequences. Wrap the existing ones as `{start_tokens}` / `{end_tokens}`. Better: tokenize each of the template's `thinking_end_tags` (§4b), which is what the multi-sequence matcher exists for |
| `hardware/gpu_layer_select.cpp:126` | `common_fit_params` takes 9 args | New `const common_fit_extra_model * extra` before `log_level`. Pass `nullptr`. *(Later: for MTP, `extra` lets the fit account for the draft context, `shares_model = true`)* |
| `multimodal/mmproj.cpp:76` | `mtmd_helper_bitmap_init_from_buf` takes 5 args | Add `mtmd_helper_init_opt_default()` as the last argument. The wrapper and video-context handling already there stay |
| `grammar/response_format.cpp:88` | `json_schema_to_grammar` now takes `const common_json &`; no conversion from `nlohmann::ordered_json` | Build the schema as `common_json::parse(json_schema_str)` (there is also `parse_no_throw`) instead of going through `nlohmann::ordered_json`. Keep helix's own validation error path |
| `engine/embed_loop.cpp:134, 351` | `common_batch_clear` / `common_batch_add` gone | Embeddings only need plain decode. **`llama_batch` and `llama_decode(ctx, llama_batch)` still exist**, so set `batch.n_tokens = 0` and fill `token/pos/n_seq_id/seq_id/logits` directly with a small helix-local `batch_add()` (10 lines). No need to move to `llama_batch_ext` here |
| `engine/decode_loop.cpp:594-595, 656` (multimodal and prompt batches) | same | Same helix-local helper |

### 4b. The chat API (half a day to a day; behaviour risk)

Upstream changed this in #30210 ("chat : refactor API") and #30389 ("chat: replace output string with
parts").

| Site | Error | Fix |
|---|---|---|
| `chat/template.cpp:162` | `params.prompt` is a `common_chat_prompt`, not a string | For now, `result.prompt = params.prompt.str()`. Also keep the `common_chat_prompt` in `RenderResult` as `prompt_parts` for D2 (§7) |
| `chat/template.cpp:165` | `thinking_end_tag` → `thinking_end_tags` (vector) | `RenderResult` holds the vector. `resolve_thinking_start_tag` and the forced-open detection use the first. The reasoning-budget sampler and `chat/reasoning.cpp` get all of them |
| `chat/template.cpp:198, 212`; `engine/decode_loop.cpp:312-314` | parser assigned to `std::string parser_data`; `common_peg_arena::load` gone | The arena can no longer be saved and reloaded, and doesn't need to be: it is a plain copyable value. Change `RenderResult::parser_data` (`template.hpp:28`) from `std::string` to `common_peg_arena parser` and copy it; delete the reload in `decode_loop.cpp:312` |
| `engine/decode_loop.cpp:316` | `common_chat_parser_params::reasoning_format` gone | Reasoning format is now decided when the template is applied (`common_chat_templates_inputs::reasoning_format`, which helix already sets at `template.cpp:150`). Remove the line. **Then confirm reasoning extraction still happens**: the parser built for a reasoning format now does the splitting, not a parse-time flag |
| `engine/decode_loop.cpp:392, 411` | `common_chat_peg_parse` takes `const common_chat_input &` | Wrap: `common_chat_input(text)`. The constructor is `explicit`, and marks every byte as text with no tokens |
| (new) `common_chat_parser_params::generation_prompt` is `common_chat_input` | | Constructing from `common_chat_params` does it. If helix fills it by hand, wrap the same way |

**What to check here** (it compiles easily but can still go wrong):
1. **Streaming diffs:** `common_chat_msg_diff` over the new `common_chat_msg`, with content in
   parts after #30389.
2. **Tool-call extraction**, under `tool_choice` auto, required and none, with parallel calls on and
   off.
3. **Reasoning:**
   - `reasoning_content` is split out for DeepSeek-style and Qwen forced-open templates.
   - Nothing leaks into `content` when `reasoning_format: none`.
   - Wren passes `reasoning_budget: 0`.
4. **The ported patch (a5b839399):** a `tool_choice: required` request on a template whose generation
   prompt lacks the reasoning start tag must not begin by re-emitting `<|im_start|>assistant\n`.

### 4c. Speculative decoding / MTP (half a day)

Upstream's `common/speculative.cpp` is +1,610/−471 lines since the base: draft-MTP, DFlash and EAGLE3
were reworked, and MTP hidden state is now saved and restored (#30141).

| Site | Error | Fix |
|---|---|---|
| `engine/decode_loop.cpp:665, 720, 749` | `common_speculative_process` takes `const common_batch &` | Build these three batches as `common_batch` (`common_batch b(ctx); b.clear(); b.add(id, pos, seq, output);`) and **decode them with `llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b.get())`**, as `common/common.cpp:1540` does. The speculative state reads per-token entries from the `common_batch` mirror, so the batch it is given must be the one that was decoded |
| `engine/decode_loop.cpp:716-740` | `common_batch_clear/add` on `sbatch` | same, `common_batch` methods |
| `engine/decode_loop.cpp:701-708` | (compiles but wrong) the draft-params initialiser is positional and the struct changed: `n_past` → `pos0`, plus new `result_q`, `temp`, `seed` | Assign named fields: `dp.drafting = true; dp.n_max = -1; dp.pos0 = n_past; dp.id_last = id_last; dp.prompt = &prompt_tokens; dp.result = &draft;`. Leave `result_q` null (greedy drafting, as today) |
| `session/session.cpp:427-438` | (compiles) `common_params_speculative` gained fields | Check the defaults still mean "draft-MTP, greedy, n_max from context". Read `common_speculative_n_max()` |
| `engine/decode_loop.cpp` after `common_speculative_draft` (`:709`), after the accept trim (`:758`), the fallback trim (`:723`) | (compiles, and silently wrong without it) upstream's draft-MTP no longer trims `ctx_dft` for single-head separate-KV models; the caller must, as `speculative-simple.cpp:211, 333` does | After drafting: `llama_memory_seq_rm(llama_get_memory(ctx_mtp), 0, n_past, -1)`. After acceptance, and in the fallback: trim `ctx_mtp` at the same position as `ctx`. Unguarded, as upstream's callers do it. The draft context has to reach `decode_loop`, which today sees only `spec` |

helix exposes MTP through `SessionOptions` (ABI 1.2+), so this has to keep working even though Wren
doesn't turn it on. `mtp-gemma-4-12B-it.gguf` and
`Qwythos-9B-…-MTP-Q8_0.gguf` cover the shared-KV (Gemma 4) and separate-KV paths respectively.

---

## 5. Phases

Each phase ends on something checkable. Phases 2–4 can be one PR with one commit per phase.

### Phase 0: baselines and decisions (half a day; do first)

1. **Make the decisions in §9.**
2. **Pick the target commit.** Use the newest `b`-tag at least a week old, not the tip of the day:
   today's tip is a large mtmd refactor (#31865) that landed this morning. Re-run
   `git rev-list --count` and the header diffstat against the chosen tag. If it is not `b12211`,
   update §2.
3. **Capture Wren baselines on the current DLL**, so "after" has a "before". Save the outputs under
   `handoff/baselines/llama-bump/`:
   - `cargo run -p wren-spike --bin prefill-probe` (writes its survey)
   - `cargo run -p wren-spike --bin reference-probe` (dream-loop costs; `N13`)
   - `cargo run -p wren-spike --bin cancel-probe` (`N14`: 78–108 ms mid-generation, ~1.9 s mid-prefill)
   - `cargo run -p wren-spike --bin embed-probe -- <embed model>` with a fixed `WREN_PROBE_TEXTS`
     set, keeping the **raw vectors**, not just the cosines (needed for R5)
   - helix `helix_integration_tests` with the CI models, for timings
4. **Fix D1 first** in a separate small PR: correct `llama.cpp.commit` and add the CI check (§7). Then
   the bump PR's version-string change is honest.

### Phase 1: the fork branch (an hour)

1. In `anemoi-ai/llama.cpp`, create branch **`helix-b12211`** (named for its upstream base, §9) from
   the chosen upstream tag.
2. Cherry-pick the port of `a5b839399`. Its message names the original commit and why the other two
   were not carried:

   ```
   common : keep generation-prompt prefix out of autoparser GBNF

   Port of helix-1.4 a5b839399 onto upstream b12211.
   Not carried from helix-1.4: ff343dab6 (superseded by the Aho-Corasick
   matcher, #25544) and 2ee83d0c3 (superseded by the draft-mtp rewrite,
   which now resets ctx_dft KV at speculative.cpp:1747/1858).
   ```
3. Push the branch. **Leave `helix-1.4` alone**: helix's tags up to now still need it to resolve.
4. Optional: open the patch upstream against `ggml-org/llama.cpp`. If it lands, the next bump carries
   no patches at all.

### Phase 2: mechanical ports (half a day)

On a helix branch `deps/llama-b12211`:
1. Point the submodule at the new fork commit. Set `.gitmodules` `branch = helix-b12211`. Regenerate
   `third_party/llama.cpp.commit` from the submodule.
2. Fix everything in §4a.
3. **Gate:** `helix`, `helix_static` and `helix_unit_tests` compile on CPU. Then run
   `ctest --test-dir build-cpu --output-on-failure`. Unit tests that fail here are signal: read each
   one before touching it.

### Phase 3: the chat API port (half a day to a day)

1. Make the §4b changes.
2. **Gate:** unit tests green, and **`helix_integration_tests` green** with `qwen2.5-0.5b-instruct-q4_k_m`
   and `nomic-embed-text-v1.5.Q8_0` (the CI models). Then repeat with
   `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` (Wren's model) and `Qwen3.5-9B-Q4_K_M.gguf` (a
   forced-open thinking template).
3. Add the tests in §6 that cover this path.

### Phase 4: speculative / MTP (half a day)

1. Make the §4c changes.
2. **Gate:** the MTP integration tests pass on both MTP models, and the acceptance rate reported by
   `helix_session_describe` is not lower than the baseline from Phase 0.

### Phase 5: the build matrix (half a day, mostly waiting)

1. **Vulkan locally**, since that is what Wren links. Use a **new** build directory so Wren's tree is
   untouched:
   `cmake -B build-vulkan-b12211 -DHELIX_BACKEND=vulkan …`, then
   `cmake --build build-vulkan-b12211 --config Release --target helix`.
2. **Open the helix PR.** Its CI covers linux-cpu, linux-cuda (gcc 13, CUDA 12.6), windows-cpu,
   windows-cuda (12.4, build-only), windows-vulkan, and both ABI checks.
   - The ABI checks must still say **29 symbols** and correct nodes. A failure there means a public
     export changed by accident.
3. **Run `release.yml`'s matrix before tagging.** It covers ROCm, omni, Linux Vulkan and aarch64,
   which `ci.yml` does not.

### Phase 6: helix release

1. Merge the PR. Tag following helix's usual scheme; the ABI stays 1.7.
2. `helix_version_string()` must report the new short SHA.

### Phase 7: Wren (about a day)

See §8.

---

## 6. Tests to add in helix

Each of these pins a behaviour this bump touches.

| Test | Pins |
|---|---|
| Autoparser, `tool_choice: required`, template whose generation prompt lacks the reasoning start tag: the grammar does not start with the prompt tail | fork patch `a5b839399` (unit test on the generated GBNF string) |
| Reasoning budget with an end tag repeating a prefix (`[A A B]` against `[A A A B]`) | what `ff343dab6` fixed, now supplied by upstream. Keep it so a future regression shows |
| Reasoning budget with two end tags (from `thinking_end_tags`) | the new multi-sequence path |
| MTP: two consecutive partially accepted rounds on the separate-KV model must not fail with "inconsistent sequence positions" | what `2ee83d0c3` fixed, now upstream |
| `load_mode` mapping: four combinations of mmap and mlock | §4a |
| `common_chat_peg_parse` round-trip of a streamed tool call (diffs in order, final message equal) | §4b |
| (with D2) user content containing `<start_of_turn>` / `<\|im_start\|>` tokenizes as text, not as control tokens | §7 D2 |

---

## 7. Existing defects found while planning

Problems that exist on `main` today, independent of the bump:

- **D1: the version string reports the wrong llama.cpp.**
  - `third_party/llama.cpp.commit` holds `ff343dab6`. The submodule is at `2ee83d0c3`, two fork
    commits later.
  - The marker hasn't changed since the 1.6.0 import (`c88e5ae`), while the submodule moved twice.
  - `CMakeLists.txt:66-69` reads the marker, so `helix_version_string()` names a llama.cpp that is not
    the one built.
  - **Fix:** generate it at configure time from `git -C third_party/llama.cpp rev-parse HEAD`, falling
    back to the file for source tarballs. Add a CI step that fails when the file and the submodule
    disagree.
- **D2: special tokens in user content are parsed as control tokens.**
  - Every prompt is tokenized with `parse_special = true` over the whole rendered string:
    `decode_loop.cpp:935, 983, 1117, 1338`, and the multimodal path through `mmproj->eval_media(…,
    rendered.prompt …)` at `:915`.
  - So a user message, a file, an email or a tool result containing a template control token
    (Gemma's `<start_of_turn>model`, ChatML's `<|im_start|>`) becomes a real turn boundary in the
    model's input. Wren feeds exactly that kind of third-party text to the model.
  - Upstream's new `common_chat_prompt` is a list of parts with `is_input` set on the parts that came
    from messages. **Fix:** tokenize part by part, with `parse_special = !part.is_input`.
  - The multimodal path needs the same treatment, with media markers kept as template text.
  - This is a behaviour change: output changes whenever content contained control-token text, which
    is the point. **Recommended as its own PR straight after the bump**, with its own tests (§6) and a
    Wren ADR. That keeps "the bump changed nothing" and "this closes an injection path" as separate,
    reviewable claims.
- **D3: two copies of nlohmann/json.**
  - helix vendors `third_party/nlohmann_json.hpp` at 3.11.3, but llama.cpp ships 3.12.0 and the trial
    shows helix's `response_format.cpp` compiling against 3.12.0 (`json_abi_v3_12_0`).
  - Check whether anything still includes the 3.11.3 copy. If nothing does, delete it. If something
    does, two JSON ABIs are being linked into one DLL.
- **D4: the update guide is out of date** (nested repo, not a submodule; ABI 1.1; the June commits).
  Rewrite it in Phase 6 (§10).

---

## 8. Wren: adopting the new helix

Wren links helix through `HELIX_LIB_DIR` (the prebuilt `build-vulkan` tree). The vendored-from-source
build still doesn't link (`N17`).

1. **Build alongside, don't overwrite.** Point Wren's environment at `build-vulkan-b12211` (Phase 5),
   keeping `build-vulkan` as the rollback. Bump `vendor/helix` to the new helix tag.
2. **Run every gate in `CLAUDE.md`:**
   - **Unit tests:** `cargo test` across the workspace.
   - **Headless:** tools, agent, journal.
   - **Embed model:** memory, queue, index, index-resume, promise, browse, capture, forget, remind,
     files.
   - **Chat model:** chat, loop, context, dream, egress.
   - **All three models:** watch, morning.
   - **UI:** `node ui/walkthrough.mjs` (102 checks) and `--after-restart`, with `WREN_TEMPERATURE=0`.
3. **Compare the probes against the Phase 0 baselines:**
   - `prefill-probe` and `reference-probe`. A regression in prefill hurts Wren most (`N13`/`N14`).
   - **`cancel-probe`, carefully.** On this machine's RX 6800 XT, a cancel that tore down a context
     with Vulkan work in flight hard-hung the driver into a bugcheck (`c260052`,
     `HELIX_R7_CANCEL_TEARDOWN_BUG.md`). The drain that fixes it lives in helix, not llama.cpp, but
     upstream changed context teardown and the Vulkan backend (166 commits).
     - Run `WREN_CANCEL_SECTIONS=5` first, then the whole probe.
     - Save work before running it.
   - `chat-smoke` [4]: a cancel with nothing in flight is still ignored (H5).
4. **Embeddings drift (R5).** Every vector in an existing Wren store was made by the old llama.cpp.
   - Re-run `embed-probe` on the same texts and compare **old against new vectors for the same text**.
   - Cosine ≥ 0.999: nothing to do.
   - Visibly lower: stored and query vectors now come from different functions. That moves cairn's
     0.95 duplicate line and the recall ranking. Then Wren needs a re-embed pass (the re-embed queue
     already exists) before shipping.
5. **Vision** (gated off; `WREN_MMPROJ`): `vision-probe` with `mmproj-BF16.gguf`.
   Upstream fixed Gemma 4's vision FFN activation (#30882) and image resizing (#30922), so answers
   may *improve*. Record them.
6. **Write `N18`** (the ADR): the numbers before and after, the drift result, the cancel result, and
   whether D2 shipped.

---

## 9. Decisions to make before starting

1. **Target commit.** Recommended: the newest `b`-tag at least a week old (Phase 0 step 2), not
   `b12211` itself.
2. **Fork branch name.** Recommended: `helix-b<N>` after the upstream base, since the ABI no longer
   tracks the branch (`helix-1.4` was cut at ABI 1.4 and carried 1.5–1.7).
3. **Upstream the surviving patch?** Recommended: yes, after Phase 3 proves it.
4. **D2 in this bump or straight after?** Recommended: straight after, as its own PR.
5. **`LLAMA_LOAD_MODE_AUTO`?** Recommended: no; keep explicit mapping. Revisit separately.

---

## 10. After the bump

- Rewrite `docs/llama-cpp-update-guide.md` from this plan:
  - submodule, not nested repo;
  - fork branch per upstream base;
  - the "check whether upstream fixed it" step for each fork patch;
  - the trial-build-first method in §3;
  - the commit-marker check.
- Delete this plan or mark it done, linking the PRs.

---

## 11. Risks

| # | Risk | Likelihood | Caught by |
|---|---|---|---|
| R1 | Tool-call or reasoning parsing changes without a compile error (#30210, #30389) | **high** | integration tests on 3 templates; Wren `loop-smoke`, `chat-smoke`, walkthrough |
| R2 | Reasoning-budget semantics changed (multiple end sequences; forcing ends on an end tag from the prompt, #31374) | medium | §6 budget tests; Wren dreamer thought filter (`M8`); `reasoning_budget: 0` path |
| R3 | MTP rewrite: acceptance rate or correctness | medium | Phase 4 gate on both MTP models |
| R4 | Sampling output changes (penalties sampler now vocab-sized; deterministic runs differ) | medium | helix `SeedDeterminism`; walkthrough at temperature 0 (expectations may need re-recording; read each diff) |
| R5 | Embedding numeric drift against stored vectors | low–medium | §8 step 4 |
| R6 | Vulkan teardown or driver regression on RDNA2 (bugcheck class) | low, **severe** | `cancel-probe` sections 5 then all; `dream-smoke` |
| R7 | Model load: `load_mode` mapping; GGUF or mmproj compatibility after the mtmd consolidation | low | first Wren launch; `vision-probe` |
| R8 | Prefill or decode slower on Vulkan | low | `prefill-probe`, `reference-probe` against baseline |
| R9 | Token counts change (tokenizer fixes), shifting Wren's context arithmetic and compaction points | low | `context-smoke` §6 (compaction still lands) |
| R10 | Second wave of compile errors larger than expected | medium | budgeted in Phase 2/3 |

## 12. Rollback

- **helix:** revert the submodule pointer, `.gitmodules` branch, marker and source changes (one PR).
  `helix-1.4` is untouched, so the old pin still resolves.
- **Wren:** point `HELIX_LIB_DIR`/`HELIX_DLL_DIR` back at `build-vulkan` and revert the `vendor/helix`
  bump.
- **A store re-embedded under §8 step 4** cannot be rolled back cheaply. Do the re-embed only after
  everything else has passed.

## 13. Expected gains

- **Vulkan:** 166 commits.
- **Gemma 4:** 58 commits, including vision-correctness fixes (#30882, #30922) and KV-cache fixes.
- **Fork:** one patch instead of three.

Not as large as the commit count suggests:
- The headline Gemma 4 prompt-processing gain (#31356) is for the E2B and E4B models, not Wren's 12B.
- The RDNA3/4 split-K work (#30280) doesn't apply to the RX 6800 XT (RDNA2).

Whatever Wren actually gains is what Phase 0 against Phase 7 measures. Nothing here should be claimed
before that.
