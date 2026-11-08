#include "session.hpp"
#include "../model/model.hpp"
#include "../model/load_mode.hpp"
#include "../model/cuda_lock.hpp"
#include "../runtime/runtime.hpp"
#include "../hardware/profile.hpp"
#include "../engine/warmup.hpp"
#include "../engine/decode_loop.hpp"
#include "../engine/embed_loop.hpp"
#include "../internal/error.hpp"
#include "../internal/log.hpp"
#include "../util/fs_path.hpp"
#include "../util/json_dump.hpp"
#include "llama.h"
#include "ggml.h"
#include "speculative.h"
#include "nlohmann/json.hpp"
#include <filesystem>

#include <algorithm>
#include <cassert>
#include <cstdint>

namespace helix {

/* Parallel sequences per embedding flush, capped at a sane default below
 * whatever the library allows. */
static constexpr uint32_t kEmbedSeqMax = 64;

/* Map a KV cache-type string ("f16", "q8_0", ...) to ggml_type. Matches the
 * set accepted by upstream's --cache-type-k. Throws HELIX_E_VALIDATION on an
 * unknown value; `param` names the offending option in the error (e.g.
 * "cache_type_k" or "speculative.cache_type_k"). */
static ggml_type parse_cache_type(const std::string& s, const std::string& param) {
    static const ggml_type kCacheTypes[] = {
        GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
        GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
        GGML_TYPE_IQ4_NL, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
    };
    for (ggml_type t : kCacheTypes) {
        if (ggml_type_name(t) == s) return t;
    }
    throw_validation(param + " \"" + s + "\" is not supported "
                     "(expected one of: f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1, f32)",
                     param);
}

Session::Session(Model& model, const SessionOptions& opts)
    : model_(model), opts_(opts) {

    const HardwareProfile& hw = model_.runtime().profile();

    if (opts_.embedding) {
        const llama_model* m = model.llama_model_ptr();
        if (llama_model_has_encoder(m) && llama_model_has_decoder(m)) {
            throw_unsupported("embeddings are not supported for encoder-decoder models",
                              "embedding");
        }
        /* KV is cleared every flush and nothing is generated, so the
         * prefix cache cannot apply. */
        opts_.prefix_cache  = false;
        embed_via_encode_   = llama_model_has_encoder(m);
    }

    if (opts_.context_shift) {
        if (opts_.embedding)
            throw_unsupported("context_shift is not supported in embedding sessions "
                              "(there is no conversation history to shift)",
                              "context_shift");
        if (opts_.speculative.type != SpeculativeType::None)
            throw_unsupported("context_shift is not supported with speculative (MTP) "
                              "sessions in this release", "context_shift");
    }

    /* LoRA adapter activation (1.6, F6). Resolve the active set against the
     * model's loaded adapters before any context exists, so every failure
     * here needs no cleanup. Absent option = all adapters at load scales;
     * explicit array = exactly those (empty = none). */
    std::vector<llama_adapter_lora*> lora_ptrs;
    {
        const auto& loaded = model.loras();
        std::vector<llama_adapter_lora*> ptrs;
        if (opts_.lora.has_value()) {
            for (const auto& sl : *opts_.lora) {
                const Model::LoadedLora* found = nullptr;
                for (const auto& l : loaded) {
                    if (l.name == sl.name) { found = &l; break; }
                }
                if (!found) {
                    std::string known;
                    for (const auto& l : loaded)
                        known += (known.empty() ? "\"" : ", \"") + l.name + "\"";
                    throw_validation("unknown lora adapter \"" + sl.name + "\"; "
                                     "adapters loaded on this model: " +
                                     (known.empty() ? "none" : known),
                                     "lora");
                }
                active_loras_.push_back({found->name,
                                         sl.scale.value_or(found->scale)});
                ptrs.push_back(found->adapter);
            }
        } else {
            for (const auto& l : loaded) {
                active_loras_.push_back({l.name, l.scale});
                ptrs.push_back(l.adapter);
            }
        }
        if (!active_loras_.empty()) {
            if (opts_.embedding)
                throw_unsupported("lora adapters are not supported in embedding "
                                  "sessions in this release; pass \"lora\": [] to "
                                  "create an embedding session on this model",
                                  "lora");
            if (opts_.speculative.type != SpeculativeType::None)
                throw_unsupported("lora adapters are not supported with speculative "
                                  "(MTP) sessions in this release (the draft context "
                                  "would decode without the adapters and diverge); "
                                  "pass \"lora\": [] or drop the speculative option",
                                  "lora");
        }
        lora_ptrs = std::move(ptrs);
    }

    if (opts_.speculative.type == SpeculativeType::DraftMtp) {
        if (opts_.embedding)
            throw_unsupported("speculative decoding is not supported in embedding sessions",
                              "speculative");
        /* Two MTP shapes are supported:
         *  - draft_model_path empty: the target GGUF itself carries the MTP
         *    head tensors (n_layer_nextn > 0, e.g. glm4/qwen35 family). The
         *    MTP context is created from the target model.
         *  - draft_model_path set: a separate draft model (e.g. Gemma-4's
         *    gemma4-assistant head) is loaded and the MTP context is created
         *    from it, linked to the target via ctx_other.
         * Which path applies is resolved at context creation; both are accepted
         * here. */
        opts_.prefix_cache  = false;
    }

    auto ctx_params = llama_context_default_params();

    const int training_ctx = model.n_ctx_train();
    if (opts.n_ctx > 0) {
        ctx_params.n_ctx = opts.n_ctx;
    } else if (training_ctx > 0 && training_ctx <= 8192) {
        ctx_params.n_ctx = training_ctx;
    } else {
        ctx_params.n_ctx = std::min(training_ctx > 0 ? training_ctx : 4096, 4096);
        log_warn("n_ctx not specified: capping at 4096 (model training context is "
                 + std::to_string(training_ctx)
                 + "). Set n_ctx explicitly in session options for larger context.");
    }

    ctx_params.swa_full = opts_.swa_full;

    // Build ModelInfo with GPU context for GPU-aware selector functions.
    ModelInfo mi;
    mi.n_ctx        = static_cast<uint32_t>(ctx_params.n_ctx);
    mi.n_ctx_train  = (training_ctx > 0) ? static_cast<uint32_t>(training_ctx) : 0;
    mi.n_gpu_layers = model.n_gpu_layers_actual();
    mi.n_layers     = llama_model_n_layer(model.llama_model_ptr());

    int n_batch;
    int n_threads;
    int n_threads_batch;

    if (opts_.embedding) {
        ctx_params.embeddings   = true;
        // Pooling: defer to model metadata unless the session JSON overrides.
        ctx_params.pooling_type = static_cast<enum llama_pooling_type>(opts_.pooling);

        // Batch geometry. Embedding throughput wants n_batch to hold the
        // largest input whole; non-causal attention requires ubatch == batch
        // (the whole input must be attended to in one physical pass).
        uint32_t auto_batch = pick_n_batch_embed(hw, mi);
        n_batch             = (opts.n_batch > 0) ? opts.n_batch : static_cast<int>(auto_batch);
        n_batch             = std::max(1, n_batch);
        ctx_params.n_batch  = static_cast<uint32_t>(n_batch);
        ctx_params.n_ubatch = static_cast<uint32_t>(n_batch);

        // Parallel sequences per flush; unified KV so any prompt count works.
        // Capped at n_batch: every sequence holds at least one token, and
        // llama.cpp asserts (aborting the process) when n_seq_max exceeds
        // the per-batch output capacity, which defaults to n_batch.
        ctx_params.n_seq_max  = std::min({
            static_cast<uint32_t>(llama_max_parallel_sequences()), kEmbedSeqMax,
            static_cast<uint32_t>(n_batch)});
        ctx_params.kv_unified = true;

        // No decode phase: the prefill heuristic applies to the whole call,
        // and the single-token thread count is never exercised.
        uint32_t auto_threads = pick_n_threads_embed(hw, mi);
        n_threads = (opts.n_threads > 0)
            ? std::min(opts.n_threads, static_cast<int>(hw.cpu.logical_cores))
            : static_cast<int>(auto_threads);
        n_threads = std::max(1, n_threads);
        ctx_params.n_threads = n_threads;

        n_threads_batch = (opts.n_threads_batch > 0)
            ? std::min(opts.n_threads_batch, static_cast<int>(hw.cpu.logical_cores))
            : n_threads;
        n_threads_batch = std::max(1, n_threads_batch);
        ctx_params.n_threads_batch = n_threads_batch;
    } else {
        // Batch size: explicit override or profile-derived.
        uint32_t auto_batch  = pick_n_batch(hw, mi);
        n_batch              = (opts.n_batch  > 0) ? opts.n_batch  : static_cast<int>(auto_batch);
        n_batch              = std::max(1, n_batch);
        ctx_params.n_batch   = static_cast<uint32_t>(n_batch);

        // Micro-batch (physical compute chunk).
        uint32_t auto_ubatch  = pick_n_ubatch(hw, mi, static_cast<uint32_t>(n_batch));
        ctx_params.n_ubatch   = auto_ubatch;

        // Decode threads: explicit override or profile-derived (GPU-aware).
        uint32_t auto_threads = pick_n_threads(hw, mi);
        n_threads = (opts.n_threads > 0)
            ? std::min(opts.n_threads, static_cast<int>(hw.cpu.logical_cores))
            : static_cast<int>(auto_threads);
        n_threads = std::max(1, n_threads);
        ctx_params.n_threads = n_threads;

        // Batch/prefill threads (GPU-aware).
        uint32_t auto_threads_batch = pick_n_threads_batch(hw, mi);
        n_threads_batch = (opts.n_threads_batch > 0)
            ? std::min(opts.n_threads_batch, static_cast<int>(hw.cpu.logical_cores))
            : static_cast<int>(auto_threads_batch);
        n_threads_batch = std::max(1, n_threads_batch);
        ctx_params.n_threads_batch  = n_threads_batch;
    }

    // Flash attention: auto-detect from ISA tier.
    const bool use_flash_attn = pick_flash_attn(hw);
    if (use_flash_attn) {
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    flash_attn_ = use_flash_attn;

    // Main-context KV cache quantization (1.4). Quantized V requires flash
    // attention (the FA kernels are the only path that dequantizes V);
    // reject the combination honestly instead of failing deep in ggml.
    ctx_params.type_k = parse_cache_type(opts_.cache_type_k, "cache_type_k");
    ctx_params.type_v = parse_cache_type(opts_.cache_type_v, "cache_type_v");
    if (ggml_is_quantized(ctx_params.type_v) && !use_flash_attn) {
        throw_validation("cache_type_v \"" + opts_.cache_type_v + "\" requires flash "
                         "attention, which is disabled on this hardware profile; use "
                         "\"f16\" (or \"bf16\"/\"f32\") for cache_type_v",
                         "cache_type_v");
    }

    // MTP: the target must emit logits for [id_last, draft0..draftN-1] in one
    // decode (1 + n_max positions). Size the output buffer accordingly.
    //
    // For self-speculation (Branch B) the target model carries the MTP head's
    // recurrent (nextn) state, which advances across the speculative batch and
    // must be rewound when drafts are rejected. pending_h/verify_h only roll
    // back the *draft* context's embeddings — they do nothing for the target's
    // recurrent state — so enable recurrent-state rollback snapshots sized to
    // the speculative batch (rejections roll back up to n_max positions).
    // Without this, seq_rm() on the target's recurrent memory silently fails
    // (rollback distance > n_rs_seq) and the KV position sticks ahead of n_past,
    // faulting the next decode under M-RoPE (X < Y). Arch support is gated in
    // llama_context (QWEN35/QWEN35MOE today); unsupported archs clamp it to 0.
    if (opts_.speculative.type == SpeculativeType::DraftMtp && !opts_.embedding) {
        const uint32_t want = static_cast<uint32_t>(1 + std::max(0, opts_.speculative.n_max));
        ctx_params.n_outputs_max = std::min<uint32_t>(
            ctx_params.n_batch ? ctx_params.n_batch : want, want);
        ctx_params.n_rs_seq = want;
    }

    ctx_ = llama_init_from_model(model.llama_model_ptr(), ctx_params);
    if (!ctx_) {
        throw Error(HELIX_E_BACKEND, "failed to create llama_context",
                    "context_create_error", "", "helix_e_backend");
    }

    /* A throwing constructor never runs the destructor: release everything
     * created from here on before rethrowing. */
    try {
        init_on_context(ctx_params, lora_ptrs);
    } catch (...) {
        release_all();
        throw;
    }

    log_debug("session created (n_ctx=" + std::to_string(ctx_params.n_ctx) +
              ", n_batch=" + std::to_string(n_batch) +
              ", n_threads=" + std::to_string(n_threads) +
              ", n_threads_batch=" + std::to_string(n_threads_batch) +
              ", flash_attn=" + (use_flash_attn ? "on" : "off") +
              ", swa_full=" + (ctx_params.swa_full ? "on" : "off") +
              (opts_.embedding
                   ? ", embedding=on, n_seq_max=" + std::to_string(ctx_params.n_seq_max)
                   : "") +
              (spec_mtp_ ? ", mtp=on (n_max=" + std::to_string(opts_.speculative.n_max)
                            + ", draft=" + (model_dft_ ? "external" : "from-target") + ")"
                         : "") + ")");
}

void Session::init_on_context(const llama_context_params& ctx_params,
                              std::vector<llama_adapter_lora*>& lora_ptrs) {
    /* context_shift needs rope-shiftable KV (generation-time eviction uses
     * llama_memory_seq_add). SWA and some hybrid architectures can't; reject
     * at creation rather than fail mid-generation. */
    if (opts_.context_shift &&
        !llama_memory_can_shift(llama_get_memory(ctx_))) {
        throw_unsupported("context_shift is not supported for this model: its KV "
                          "cache cannot be position-shifted (multi-axis rope "
                          "(M-RoPE), sliding-window, or hybrid attention)",
                          "context_shift");
    }

    /* Apply the resolved LoRA set (1.6, F6). Set once here and never changed:
     * an adapter change is a new session, which keeps the decode hot path and
     * the prefix-cache invariant untouched. */
    if (!lora_ptrs.empty()) {
        std::vector<float> scales;
        scales.reserve(active_loras_.size());
        for (const auto& a : active_loras_) scales.push_back(a.scale);
        if (llama_set_adapters_lora(ctx_, lora_ptrs.data(), lora_ptrs.size(),
                                    scales.data()) != 0) {
            throw Error(HELIX_E_BACKEND,
                        "failed to apply lora adapters to the context",
                        "context_create_error", "lora", "helix_e_backend");
        }
    }

    if (opts_.embedding) {
        /* Fail at creation, not per-request. */
        effective_pooling_ = llama_pooling_type(ctx_);
        if (effective_pooling_ == LLAMA_POOLING_TYPE_NONE) {
            throw_unsupported("pooling type 'none' is not supported by "
                              "helix_embeddings; load a model with a pooling head "
                              "or set the 'pooling' session option to "
                              "'mean'/'cls'/'last'", "pooling");
        }
        if (effective_pooling_ == LLAMA_POOLING_TYPE_RANK) {
            /* Rank pooling selects the rerank endpoint (1.5). The model
             * must actually carry a rank-capable head: classification
             * tensors (bge/jina reranker family — detected from tensor
             * names, since labels metadata is often absent), classifier
             * labels, or a "rerank" chat template (Qwen3-Reranker
             * family). Forcing "rank" onto a plain embedding model
             * would read garbage scores. */
            const llama_model* m2 = model_.llama_model_ptr();
            if (!model_.has_cls_head() &&
                llama_model_cls_label(m2, 0) == nullptr &&
                llama_model_chat_template(m2, "rerank") == nullptr) {
                throw_unsupported("pooling \"rank\" requires a reranker-head "
                                  "model (classification-head tensors, "
                                  "classifier labels, or a \"rerank\" chat "
                                  "template); this model has none of these",
                                  "pooling");
            }
            rerank_mode_ = true;
        }
    }

    /* ---- MTP speculative context + manager ----
     * Branch C (draft_model_path set): load a separate MTP head model
     *   (e.g. Gemma-4's gemma4-assistant) and create the MTP context from it.
     * Branch B (draft_model_path empty): create the MTP context from the
     *   target model itself, which requires the target GGUF to carry MTP head
     *   tensors (n_layer_nextn > 0). ctx_mtp_ aliases ctx_ via ctx_other, so
     *   it must be freed first. */
    if (opts_.speculative.type == SpeculativeType::DraftMtp && !opts_.embedding) {
        llama_model* mtp_model = model_.llama_model_ptr();  /* Branch B default */
        if (!opts_.speculative.draft_model_path.empty()) {
            std::error_code ec;
            if (!std::filesystem::exists(
                    utf8_path(opts_.speculative.draft_model_path), ec)) {
                throw_model_not_found("speculative draft model not found: "
                                      + opts_.speculative.draft_model_path,
                                      "speculative.model_path");
            }
            auto dft_params = llama_model_default_params();
            dft_params.n_gpu_layers = model_.n_gpu_layers_actual();
            dft_params.load_mode    = load_mode_from(model_.options().use_mmap,
                                                     model_.options().use_mlock);
            /* The draft file is an MTP model: its layers are the point. */
            dft_params.load_mtp     = true;
            dft_params.lazy_mode    = LLAMA_LAZY_MODE_OFF;
            model_.apply_device_preference(dft_params);
            log_info("loading MTP draft model: "
                     + opts_.speculative.draft_model_path);
            model_dft_ = llama_model_load_from_file(
                opts_.speculative.draft_model_path.c_str(), dft_params);
            if (!model_dft_) {
                throw_model_load_failed("failed to load MTP draft model: "
                                        + opts_.speculative.draft_model_path,
                                        "speculative.model_path");
            }
            mtp_model = model_dft_;
        }

        /* Derive the MTP context params from the target's ctx_params, the
         * same way the server's common_context_params_to_llama(params_base)
         * does, so n_ctx / n_batch / flash_attn / threads all match. Then
         * override the MTP-specific fields. */
        auto cparams_mtp = ctx_params;
        cparams_mtp.ctx_type        = LLAMA_CONTEXT_TYPE_MTP;
        cparams_mtp.type_k          = parse_cache_type(opts_.speculative.cache_type_k,
                                                       "speculative.cache_type_k");
        cparams_mtp.type_v          = parse_cache_type(opts_.speculative.cache_type_v,
                                                       "speculative.cache_type_v");
        cparams_mtp.n_rs_seq        = 0;
        cparams_mtp.n_outputs_max   = 1;           /* one draft output per step */
        cparams_mtp.ctx_other       = ctx_;
        cparams_mtp.embeddings      = false;
        cparams_mtp.pooling_type    = LLAMA_POOLING_TYPE_UNSPECIFIED;
        cparams_mtp.n_seq_max       = 1;
        cparams_mtp.kv_unified      = false;
        /* The MTP draft context borrows the target's KV for shared layers
         * (sizing is inherited from the source cache). For any unshared
         * layers, cap n_ctx so the base/SWA sub-caches don't try to
         * allocate at the model's full training context (262144 for
         * Gemma-4 => ~80 GB). The draft only ever holds a few tokens per
         * step, so a modest n_ctx is sufficient. */
        cparams_mtp.swa_full = false;
        cparams_mtp.n_ctx    = std::min<uint32_t>(ctx_params.n_ctx, 4096);

        ctx_mtp_ = llama_init_from_model(mtp_model, cparams_mtp);
        if (!ctx_mtp_) {
            throw_unsupported("failed to create MTP context - the model does not "
                              "provide an MTP head (n_layer_nextn == 0). For "
                              "Gemma-4, set speculative.model_path to the "
                              "gemma4-assistant draft file; for other architectures "
                              "use an MTP-trained GGUF.", "speculative");
        }

        common_params_speculative spec_params;
        spec_params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        spec_params.draft.ctx_tgt  = ctx_;
        spec_params.draft.ctx_dft  = ctx_mtp_;
        spec_params.draft.n_max    = opts_.speculative.n_max;
        spec_params.draft.n_min    = opts_.speculative.n_min;
        spec_params.draft.p_min    = opts_.speculative.p_min;
        spec_params.draft.backend_sampling = opts_.speculative.backend_sampling;
        spec_params.draft.cache_type_k = cparams_mtp.type_k;
        spec_params.draft.cache_type_v = cparams_mtp.type_v;

        spec_mtp_ = common_speculative_init(spec_params, /*n_seq=*/1);
        if (!spec_mtp_) {
            throw Error(HELIX_E_BACKEND,
                        "common_speculative_init failed for MTP",
                        "context_create_error", "speculative", "helix_e_backend");
        }
    }
}

Session::~Session() {
    release_all();
}

void Session::release_all() noexcept {
    /* Ordering: spec_mtp_ holds raw pointers into both contexts; ctx_mtp_
     * aliases ctx_ via ctx_other; model_dft_ is the draft model for Branch C.
     * Free in reverse-dependency order. */

    /* Drain any in-flight device work before freeing the contexts. On the
     * Vulkan/CUDA backends llama_decode() is asynchronous — it submits a batch
     * to the device queue and returns before the GPU finishes. A cooperative
     * cancel() breaks out of the decode loop immediately after such a submit,
     * leaving the last batch mid-submit; a clean (uncancelled) run only avoids
     * the same hazard by timing luck. Freeing the context — and on Vulkan the
     * underlying VkDevice/queue/command buffers — while work is still pending is
     * an out-of-order device teardown that hard-hangs the driver (TDR → bugcheck)
     * instead of faulting cleanly. See HELIX_R7_CANCEL_TEARDOWN_BUG.md.
     *
     * llama_synchronize() blocks until all submitted batches complete; on CPU it
     * is a no-op. Drain both contexts that own device work, MTP draft first
     * (ctx_mtp_ shares layers with ctx_ via ctx_other). Also runs on the
     * constructor's error path, where it is harmless. */
    if (ctx_mtp_) { llama_synchronize(ctx_mtp_); }
    if (ctx_)     { llama_synchronize(ctx_);     }

    if (spec_mtp_)  { common_speculative_free(spec_mtp_); spec_mtp_  = nullptr; }
    if (ctx_mtp_)   { llama_free(ctx_mtp_);               ctx_mtp_   = nullptr; }
    if (model_dft_) { llama_model_free(model_dft_);       model_dft_ = nullptr; }
    if (ctx_)       { llama_free(ctx_);                   ctx_       = nullptr; }
}

std::string Session::describe() const {
    /* Built with nlohmann rather than by hand: std::to_string(float) honours
     * the host's LC_NUMERIC (a "," decimal separator yields invalid JSON). */
    nlohmann::ordered_json j = nlohmann::ordered_json::object();
    if (ctx_) {
        j["n_ctx"]         = llama_n_ctx(ctx_);
        j["n_batch"]       = llama_n_batch(ctx_);
        j["n_ubatch"]      = llama_n_ubatch(ctx_);
        j["n_threads"]     = llama_n_threads(ctx_);
        j["embedding"]     = opts_.embedding;
        j["rerank"]        = rerank_mode_;
        j["swa_full"]      = opts_.swa_full;
        j["context_shift"] = opts_.context_shift;
        nlohmann::ordered_json loras = nlohmann::ordered_json::array();
        for (const auto& a : active_loras_) {
            loras.push_back({{"name", a.name}, {"scale", a.scale}});
        }
        j["lora"]          = std::move(loras);
        j["flash_attn"]    = flash_attn_;
        j["cache_type_k"]  = opts_.cache_type_k;
        j["cache_type_v"]  = opts_.cache_type_v;
    }
    nlohmann::ordered_json spec = nlohmann::ordered_json::object();
    if (opts_.speculative.type == SpeculativeType::DraftMtp) {
        spec["type"]        = "draft-mtp";
        spec["enabled"]     = spec_mtp_ != nullptr;
        spec["n_max"]       = opts_.speculative.n_max;
        spec["n_min"]       = opts_.speculative.n_min;
        spec["p_min"]       = opts_.speculative.p_min;
        spec["draft_model"] = model_dft_ != nullptr;
    } else {
        spec["type"]    = "none";
        spec["enabled"] = false;
    }
    j["speculative"] = std::move(spec);
    return dump_json(j);
}

void Session::maybe_run_warmup() {
    if (!opts_.warmup) return;
    if (warmup_done_.load(std::memory_order_relaxed)) return;
    try {
        if (opts_.embedding) {
            run_embed_warmup(*this);
        } else {
            run_warmup(ctx_, model_.n_vocab());
            /* No warmup of the MTP context. It used to get the same dummy-token
             * decode, to JIT its graph before the first speculative request.
             * Since llama.cpp b12033 an MTP graph reads the target's hidden
             * state for every token, and a plain batch carries none: the decode
             * dies on a GGML_ASSERT ("this graph requires a state embedding"),
             * which aborts the process, so no try/catch here can contain it.
             * The first speculative request pays the graph build instead. */
        }
        warmup_done_.store(true, std::memory_order_relaxed);
    } catch (...) {
        warmup_done_.store(false, std::memory_order_relaxed);
        throw;
    }
}

Session::InFlight::InFlight(Session& s) : sess(s) {
    std::lock_guard<std::mutex> lk(sess.cancel_mu_);
    sess.in_flight_ = true;
    sess.cancel_requested_.store(false, std::memory_order_relaxed);
}

Session::InFlight::~InFlight() {
    std::lock_guard<std::mutex> lk(sess.cancel_mu_);
    sess.in_flight_ = false;
    sess.cancel_requested_.store(false, std::memory_order_relaxed);
}

Session::CallGuard::CallGuard(Session& s, bool cancellable)
    :
#if HELIX_CUDA_REQUIRES_GLOBAL_LOCK
      gpu_(s.model_.cuda_mu()),
#endif
      session_(s.mu_) {
    if (cancellable) in_flight_.emplace(s);
}

void Session::require_model_alias(const std::string& requested) const {
    if (requested != model_.alias()) {
        throw_model_not_found("model '" + requested + "' not loaded in this session",
                              "model");
    }
}

void Session::cancel() {
    std::lock_guard<std::mutex> lk(cancel_mu_);
    if (in_flight_) cancel_requested_.store(true, std::memory_order_relaxed);
}

helix_status_t Session::chat_completions(const char* request_json,
                                          char** out_response_json) {
    return do_chat_completions(request_json, out_response_json, nullptr, nullptr, false);
}

helix_status_t Session::count_tokens(const char* request_json,
                                     uint32_t* out_token_count) {
    /* Deliberately no CallGuard and no warmup: rendering and
     * tokenization only read the model's chat templates and vocabulary,
     * never the KV cache, so an in-flight request on this session cannot
     * be disturbed (or disturb the count). */
    return run_count_tokens(*this, request_json, out_token_count);
}

helix_status_t Session::chat_completions_stream(const char* request_json,
                                                   helix_stream_cb on_chunk,
                                                   void* user_data) {
    /* Precondition: the ABI layer (helix_chat_completions_stream) validates
     * on_chunk and is the only caller; the check is not repeated here. */
    assert(on_chunk && "on_chunk callback must not be null");
    return do_chat_completions(request_json, nullptr, on_chunk, user_data, true);
}

helix_status_t Session::do_chat_completions(const char* request_json,
                                              char** out_response_json,
                                              helix_stream_cb on_chunk,
                                              void* user_data,
                                               bool streaming) {
    // The guard is held for the entire duration of run_chat_completions,
    // including all streaming callbacks. Callbacks must not call
    // helix_session_destroy or re-enter helix_chat_completions on this
    // session — see helix.h.
    CallGuard guard(*this, /*cancellable=*/true);

    if (opts_.embedding) {
        throw_unsupported("this session was created with {\"embedding\": true} and "
                          "only accepts helix_embeddings", "session");
    }

    maybe_run_warmup();

    return run_chat_completions(*this, request_json,
                                out_response_json, on_chunk, user_data, streaming);
}

helix_status_t Session::embeddings(const char* request_json,
                                   char** out_response_json) {
    CallGuard guard(*this, /*cancellable=*/true);

    if (!opts_.embedding) {
        throw_unsupported("this session was not created with {\"embedding\": true}",
                          "session");
    }
    if (rerank_mode_) {
        throw_unsupported("this is a rerank session (pooling \"rank\"); rank heads "
                          "produce relevance scores, not embedding vectors — use "
                          "helix_rerank", "session");
    }

    maybe_run_warmup();

    return run_embeddings(*this, request_json, out_response_json);
}

helix_status_t Session::rerank(const char* request_json,
                               char** out_response_json) {
    CallGuard guard(*this, /*cancellable=*/true);

    if (!rerank_mode_) {
        throw_unsupported("this session cannot serve helix_rerank; create it with "
                          "{\"embedding\": true, \"pooling\": \"rank\"} on a "
                          "reranker-head model", "session");
    }

    maybe_run_warmup();

    return run_rerank(*this, request_json, out_response_json);
}

} // namespace helix
