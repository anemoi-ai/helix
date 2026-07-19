#include "session.hpp"
#include "../model/model.hpp"
#include "../model/cuda_lock.hpp"
#include "../runtime/runtime.hpp"
#include "../hardware/profile.hpp"
#include "../engine/warmup.hpp"
#include "../engine/decode_loop.hpp"
#include "../engine/embed_loop.hpp"
#include "../internal/error.hpp"
#include "../internal/log.hpp"
#include "llama.h"
#include "ggml.h"
#include "speculative.h"
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
    throw Error(HELIX_E_VALIDATION,
                param + " \"" + s + "\" is not supported "
                "(expected one of: f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1, f32)",
                "invalid_request_error", param, "helix_e_validation");
}

Session::Session(Model& model, const SessionOptions& opts)
    : model_(model), opts_(opts) {

    const HardwareProfile& hw = model_.runtime().profile();

    if (opts_.embedding) {
        const llama_model* m = model.llama_model_ptr();
        if (llama_model_has_encoder(m) && llama_model_has_decoder(m)) {
            throw Error(HELIX_E_UNSUPPORTED_FEATURE,
                        "embeddings are not supported for encoder-decoder models",
                        "unsupported_feature_error", "embedding",
                        "helix_e_unsupported_feature");
        }
        /* KV is cleared every flush and nothing is generated, so the
         * prefix cache cannot apply. */
        opts_.prefix_cache  = false;
        embed_via_encode_   = llama_model_has_encoder(m);
    }

    if (opts_.context_shift) {
        if (opts_.embedding)
            throw Error(HELIX_E_UNSUPPORTED_FEATURE,
                        "context_shift is not supported in embedding sessions "
                        "(there is no conversation history to shift)",
                        "unsupported_feature_error", "context_shift",
                        "helix_e_unsupported_feature");
        if (opts_.speculative.type != SpeculativeType::None)
            throw Error(HELIX_E_UNSUPPORTED_FEATURE,
                        "context_shift is not supported with speculative (MTP) "
                        "sessions in this release",
                        "unsupported_feature_error", "context_shift",
                        "helix_e_unsupported_feature");
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
                    throw Error(HELIX_E_VALIDATION,
                                "unknown lora adapter \"" + sl.name + "\"; "
                                "adapters loaded on this model: " +
                                (known.empty() ? "none" : known),
                                "invalid_request_error", "lora",
                                "helix_e_validation");
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
                throw Error(HELIX_E_UNSUPPORTED_FEATURE,
                            "lora adapters are not supported in embedding "
                            "sessions in this release; pass \"lora\": [] to "
                            "create an embedding session on this model",
                            "unsupported_feature_error", "lora",
                            "helix_e_unsupported_feature");
            if (opts_.speculative.type != SpeculativeType::None)
                throw Error(HELIX_E_UNSUPPORTED_FEATURE,
                            "lora adapters are not supported with speculative "
                            "(MTP) sessions in this release (the draft context "
                            "would decode without the adapters and diverge); "
                            "pass \"lora\": [] or drop the speculative option",
                            "unsupported_feature_error", "lora",
                            "helix_e_unsupported_feature");
        }
        lora_ptrs = std::move(ptrs);
    }

    if (opts_.speculative.type == SpeculativeType::DraftMtp) {
        if (opts_.embedding)
            throw Error(HELIX_E_UNSUPPORTED_FEATURE,
                        "speculative decoding is not supported in embedding sessions",
                        "unsupported_feature_error", "speculative",
                        "helix_e_unsupported_feature");
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
        ctx_params.n_seq_max  = std::min<uint32_t>(
            static_cast<uint32_t>(llama_max_parallel_sequences()), kEmbedSeqMax);
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
        throw Error(HELIX_E_VALIDATION,
                    "cache_type_v \"" + opts_.cache_type_v + "\" requires flash "
                    "attention, which is disabled on this hardware profile; use "
                    "\"f16\" (or \"bf16\"/\"f32\") for cache_type_v",
                    "invalid_request_error", "cache_type_v", "helix_e_validation");
    }

    // MTP: the target must emit logits for [id_last, draft0..draftN-1] in one
    // decode (1 + n_max positions). Size the output buffer accordingly and
    // disable recurrent-state rollback snapshots (the MTP impl manages its own
    // rollback via pending_h/verify_h).
    if (opts_.speculative.type == SpeculativeType::DraftMtp && !opts_.embedding) {
        const uint32_t want = static_cast<uint32_t>(1 + std::max(0, opts_.speculative.n_max));
        ctx_params.n_outputs_max = std::min<uint32_t>(
            ctx_params.n_batch ? ctx_params.n_batch : want, want);
        ctx_params.n_rs_seq = 0;
    }

    ctx_ = llama_new_context_with_model(model.llama_model_ptr(), ctx_params);
    if (!ctx_) {
        throw Error(HELIX_E_BACKEND, "failed to create llama_context",
                    "context_create_error", "", "helix_e_backend");
    }

    /* context_shift needs rope-shiftable KV (generation-time eviction uses
     * llama_memory_seq_add). SWA and some hybrid architectures can't; reject
     * at creation rather than fail mid-generation. The constructor throws
     * before the destructor can run, so free the context on the error path. */
    if (opts_.context_shift &&
        !llama_memory_can_shift(llama_get_memory(ctx_))) {
        llama_free(ctx_);
        ctx_ = nullptr;
        throw Error(HELIX_E_UNSUPPORTED_FEATURE,
                    "context_shift is not supported for this model: its KV "
                    "cache cannot be position-shifted (multi-axis rope "
                    "(M-RoPE), sliding-window, or hybrid attention)",
                    "unsupported_feature_error", "context_shift",
                    "helix_e_unsupported_feature");
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
            llama_free(ctx_);
            ctx_ = nullptr;
            throw Error(HELIX_E_BACKEND,
                        "failed to apply lora adapters to the context",
                        "context_create_error", "lora", "helix_e_backend");
        }
    }

    if (opts_.embedding) {
        /* Fail at creation, not per-request. The constructor throws before
         * the destructor can run, so free the context on the error paths. */
        try {
            effective_pooling_ = llama_pooling_type(ctx_);
            if (effective_pooling_ == LLAMA_POOLING_TYPE_NONE) {
                throw Error(HELIX_E_UNSUPPORTED_FEATURE,
                            "pooling type 'none' is not supported by "
                            "helix_embeddings; load a model with a pooling head "
                            "or set the 'pooling' session option to "
                            "'mean'/'cls'/'last'",
                            "unsupported_feature_error", "pooling",
                            "helix_e_unsupported_feature");
            }
            if (effective_pooling_ == LLAMA_POOLING_TYPE_RANK) {
                /* Rank pooling selects the rerank endpoint (1.5). The model
                 * must actually carry a rank-capable head: classification
                 * tensors (bge/jina reranker family — detected from tensor
                 * names, since labels metadata is often absent), classifier
                 * labels, or a "rerank" chat template (Qwen3-Reranker
                 * family). Forcing "rank" onto a plain embedding model
                 * would read garbage scores. */
                const llama_model* m2 = model.llama_model_ptr();
                if (!model.has_cls_head() &&
                    llama_model_cls_label(m2, 0) == nullptr &&
                    llama_model_chat_template(m2, "rerank") == nullptr) {
                    throw Error(HELIX_E_UNSUPPORTED_FEATURE,
                                "pooling \"rank\" requires a reranker-head "
                                "model (classification-head tensors, "
                                "classifier labels, or a \"rerank\" chat "
                                "template); this model has none of these",
                                "unsupported_feature_error", "pooling",
                                "helix_e_unsupported_feature");
                }
                rerank_mode_ = true;
            }
        } catch (...) {
            llama_free(ctx_);
            ctx_ = nullptr;
            throw;
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
        try {
            llama_model* mtp_model = model.llama_model_ptr();  /* Branch B default */
            if (!opts_.speculative.draft_model_path.empty()) {
                std::error_code ec;
                if (!std::filesystem::exists(opts_.speculative.draft_model_path, ec)) {
                    throw Error(HELIX_E_MODEL_NOT_FOUND,
                                "speculative draft model not found: "
                                + opts_.speculative.draft_model_path,
                                "model_not_found", "speculative.model_path",
                                "helix_e_model_not_found");
                }
                auto dft_params = llama_model_default_params();
                dft_params.n_gpu_layers = model.n_gpu_layers_actual();
                dft_params.use_mmap     = model.options().use_mmap;
                dft_params.use_mlock    = model.options().use_mlock;
                log_info("loading MTP draft model: "
                         + opts_.speculative.draft_model_path);
                model_dft_ = llama_model_load_from_file(
                    opts_.speculative.draft_model_path.c_str(), dft_params);
                if (!model_dft_) {
                    throw Error(HELIX_E_MODEL_LOAD_FAILED,
                                "failed to load MTP draft model: "
                                + opts_.speculative.draft_model_path,
                                "model_load_failed", "speculative.model_path",
                                "helix_e_model_load_failed");
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
                throw Error(HELIX_E_UNSUPPORTED_FEATURE,
                            "failed to create MTP context - the model does not "
                            "provide an MTP head (n_layer_nextn == 0). For "
                            "Gemma-4, set speculative.model_path to the "
                            "gemma4-assistant draft file; for other architectures "
                            "use an MTP-trained GGUF.",
                            "unsupported_feature_error", "speculative",
                            "helix_e_unsupported_feature");
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
        } catch (...) {
            if (spec_mtp_)  { common_speculative_free(spec_mtp_); spec_mtp_ = nullptr; }
            if (ctx_mtp_)   { llama_free(ctx_mtp_);               ctx_mtp_  = nullptr; }
            if (model_dft_) { llama_model_free(model_dft_);       model_dft_ = nullptr; }
            llama_free(ctx_);
            ctx_ = nullptr;
            throw;
        }
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

Session::~Session() {
    /* Ordering: spec_mtp_ holds raw pointers into both contexts; ctx_mtp_
     * aliases ctx_ via ctx_other; model_dft_ is the draft model for Branch C.
     * Free in reverse-dependency order. */
    if (spec_mtp_)  { common_speculative_free(spec_mtp_); spec_mtp_ = nullptr; }
    if (ctx_mtp_)   { llama_free(ctx_mtp_);               ctx_mtp_  = nullptr; }
    if (model_dft_) { llama_model_free(model_dft_);       model_dft_ = nullptr; }
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
}

std::string Session::describe() const {
    std::string s = "{";
    if (ctx_) {
        s += "\"n_ctx\":"     + std::to_string(llama_n_ctx(ctx_));
        s += ",\"n_batch\":"  + std::to_string(llama_n_batch(ctx_));
        s += ",\"n_ubatch\":" + std::to_string(llama_n_ubatch(ctx_));
        s += ",\"n_threads\":" + std::to_string(llama_n_threads(ctx_));
        s += ",\"embedding\":";
        s += opts_.embedding ? "true" : "false";
        s += ",\"rerank\":";
        s += rerank_mode_ ? "true" : "false";
        s += ",\"swa_full\":";
        s += opts_.swa_full ? "true" : "false";
        s += ",\"context_shift\":";
        s += opts_.context_shift ? "true" : "false";
        s += ",\"lora\":[";
        for (size_t i = 0; i < active_loras_.size(); ++i) {
            if (i) s += ',';
            s += "{\"name\":\"" + active_loras_[i].name + "\",\"scale\":" +
                 std::to_string(active_loras_[i].scale) + "}";
        }
        s += "]";
        s += ",\"flash_attn\":";
        s += flash_attn_ ? "true" : "false";
        s += ",\"cache_type_k\":\"" + opts_.cache_type_k + "\"";
        s += ",\"cache_type_v\":\"" + opts_.cache_type_v + "\"";
    }
    s += ",\"speculative\":{";
    if (opts_.speculative.type == SpeculativeType::DraftMtp) {
        s += "\"type\":\"draft-mtp\"";
        s += ",\"enabled\":";
        s += spec_mtp_ ? "true" : "false";
        s += ",\"n_max\":"  + std::to_string(opts_.speculative.n_max);
        s += ",\"n_min\":"  + std::to_string(opts_.speculative.n_min);
        s += ",\"p_min\":"  + std::to_string(opts_.speculative.p_min);
        s += ",\"draft_model\":";
        s += model_dft_ ? "true" : "false";
    } else {
        s += "\"type\":\"none\",\"enabled\":false";
    }
    s += "}}";
    return s;
}

void Session::maybe_run_warmup() {
    if (!opts_.warmup) return;
    if (warmup_done_.load(std::memory_order_relaxed)) return;
    try {
        if (opts_.embedding) {
            run_embed_warmup(*this);
        } else {
            run_warmup(ctx_, model_.n_vocab());
            /* JIT the MTP head graph so the first speculative request does not
             * pay the graph-build cost. Safe to skip on failure. */
            if (ctx_mtp_) {
                try { run_warmup(ctx_mtp_, model_.n_vocab()); }
                catch (const std::exception& e) {
                    log_warn("MTP context warmup skipped: " + std::string(e.what()));
                }
            }
        }
        warmup_done_.store(true, std::memory_order_relaxed);
    } catch (...) {
        warmup_done_.store(false, std::memory_order_relaxed);
        throw;
    }
}

void Session::cancel() {
    cancel_requested_.store(true, std::memory_order_relaxed);
}

helix_status_t Session::chat_completions(const char* request_json,
                                          char** out_response_json) {
    return do_chat_completions(request_json, out_response_json, nullptr, nullptr, false);
}

helix_status_t Session::count_tokens(const char* request_json,
                                     uint32_t* out_token_count) {
    /* Deliberately no cuda_mu_/mu_ locks and no warmup: rendering and
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
    // Lock ordering: cuda_mu_ (model-wide) BEFORE mu_ (per-session).
    // mu_ is held for the entire duration of run_chat_completions, including
    // all streaming callbacks.  Callbacks must not call helix_session_destroy
    // or re-enter helix_chat_completions on this session — see helix.h.
    // On non-CUDA builds HELIX_CUDA_REQUIRES_GLOBAL_LOCK is 0 and this is a no-op.
#if HELIX_CUDA_REQUIRES_GLOBAL_LOCK
    std::lock_guard<std::mutex> cuda_guard(model_.cuda_mu());
#endif
    std::lock_guard<std::mutex> guard(mu_);

    struct CancelResetter {
        std::atomic<bool>& flag;
        ~CancelResetter() { flag.store(false, std::memory_order_relaxed); }
    } resetter{cancel_requested_};

    if (opts_.embedding) {
        throw Error(HELIX_E_UNSUPPORTED_FEATURE,
                    "this session was created with {\"embedding\": true} and "
                    "only accepts helix_embeddings",
                    "unsupported_feature_error", "session",
                    "helix_e_unsupported_feature");
    }

    maybe_run_warmup();

    return run_chat_completions(*this, request_json,
                                out_response_json, on_chunk, user_data, streaming);
}

helix_status_t Session::embeddings(const char* request_json,
                                   char** out_response_json) {
    // Lock ordering: cuda_mu_ (model-wide) BEFORE mu_ (per-session), same as
    // do_chat_completions. mu_ is held for the entire run_embeddings call —
    // one inference at a time per session.
#if HELIX_CUDA_REQUIRES_GLOBAL_LOCK
    std::lock_guard<std::mutex> cuda_guard(model_.cuda_mu());
#endif
    std::lock_guard<std::mutex> guard(mu_);

    struct CancelResetter {
        std::atomic<bool>& flag;
        ~CancelResetter() { flag.store(false, std::memory_order_relaxed); }
    } resetter{cancel_requested_};

    if (!opts_.embedding) {
        throw Error(HELIX_E_UNSUPPORTED_FEATURE,
                    "this session was not created with {\"embedding\": true}",
                    "unsupported_feature_error", "session",
                    "helix_e_unsupported_feature");
    }
    if (rerank_mode_) {
        throw Error(HELIX_E_UNSUPPORTED_FEATURE,
                    "this is a rerank session (pooling \"rank\"); rank heads "
                    "produce relevance scores, not embedding vectors — use "
                    "helix_rerank",
                    "unsupported_feature_error", "session",
                    "helix_e_unsupported_feature");
    }

    maybe_run_warmup();

    return run_embeddings(*this, request_json, out_response_json);
}

helix_status_t Session::rerank(const char* request_json,
                               char** out_response_json) {
    // Lock ordering: cuda_mu_ (model-wide) BEFORE mu_ (per-session), same as
    // the other endpoints. One inference at a time per session.
#if HELIX_CUDA_REQUIRES_GLOBAL_LOCK
    std::lock_guard<std::mutex> cuda_guard(model_.cuda_mu());
#endif
    std::lock_guard<std::mutex> guard(mu_);

    struct CancelResetter {
        std::atomic<bool>& flag;
        ~CancelResetter() { flag.store(false, std::memory_order_relaxed); }
    } resetter{cancel_requested_};

    if (!rerank_mode_) {
        throw Error(HELIX_E_UNSUPPORTED_FEATURE,
                    "this session cannot serve helix_rerank; create it with "
                    "{\"embedding\": true, \"pooling\": \"rank\"} on a "
                    "reranker-head model",
                    "unsupported_feature_error", "session",
                    "helix_e_unsupported_feature");
    }

    maybe_run_warmup();

    return run_rerank(*this, request_json, out_response_json);
}

} // namespace helix
