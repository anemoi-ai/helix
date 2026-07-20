#include "model.hpp"
#include "../runtime/runtime.hpp"
#include "../hardware/gpu_layer_select.hpp"
#include "../multimodal/mmproj.hpp"
#include "../internal/error.hpp"
#include "../internal/log.hpp"
#include "../internal/version.hpp"

#include <cstdio>

/* chat.h declares `using json = nlohmann::ordered_json` at global scope.
 * Include it before our alias so ours wins inside the helix namespace,
 * and use nj to avoid the global collision. */
#include "llama.h"
#include "gguf.h"
#include "chat.h"
#include "common.h"
#include "nlohmann/json.hpp"

#include <cctype>
#include <cstring>
#include <filesystem>
#include <stdexcept>

using nj = nlohmann::json;

namespace helix {

ModelOptions parse_model_options(const char* model_json) {
    if (!model_json || model_json[0] == '\0')
        throw_invalid_arg("model_json must not be null or empty");

    nj j;
    try {
        j = nj::parse(model_json);
    } catch (const nj::parse_error& e) {
        throw_invalid_json(std::string("model options JSON parse error: ") + e.what());
    }

    ModelOptions opts;

    if (!j.contains("model_path") || !j["model_path"].is_string())
        throw_validation("missing required field 'model_path'", "model_path");

    opts.model_path = j["model_path"].get<std::string>();

    if (j.contains("alias") && j["alias"].is_string())
        opts.alias = j["alias"].get<std::string>();
    if (j.contains("mmproj_path") && j["mmproj_path"].is_string())
        opts.mmproj_path = j["mmproj_path"].get<std::string>();
    if (j.contains("chat_template_override") && j["chat_template_override"].is_string())
        opts.chat_template_override = j["chat_template_override"].get<std::string>();
    if (j.contains("n_ctx") && j["n_ctx"].is_number_integer()) {
        opts.n_ctx = j["n_ctx"].get<int>();
        if (opts.n_ctx < 0)
            throw_validation("n_ctx must be non-negative, got " +
                             std::to_string(opts.n_ctx), "n_ctx");
    }
    if (j.contains("n_gpu_layers")) {
        const auto& v = j["n_gpu_layers"];
        if (v.is_string())
            opts.n_gpu_layers_str = v.get<std::string>();
        else if (v.is_number_integer())
            opts.n_gpu_layers_str = std::to_string(v.get<int>());
    }
    if (j.contains("use_mmap") && j["use_mmap"].is_boolean())
        opts.use_mmap = j["use_mmap"].get<bool>();
    if (j.contains("use_mlock") && j["use_mlock"].is_boolean())
        opts.use_mlock = j["use_mlock"].get<bool>();
    if (j.contains("vocab_only") && j["vocab_only"].is_boolean())
        opts.vocab_only = j["vocab_only"].get<bool>();
    if (j.contains("reasoning_format") && j["reasoning_format"].is_string()) {
        opts.reasoning_format = j["reasoning_format"].get<std::string>();
        if (opts.reasoning_format != "auto" &&
            opts.reasoning_format != "deepseek-r1" &&
            opts.reasoning_format != "qwq" &&
            opts.reasoning_format != "none") {
            throw_validation(
                "reasoning_format must be \"auto\", \"deepseek-r1\", \"qwq\", or \"none\" "
                "(got \"" + opts.reasoning_format + "\")",
                "reasoning_format");
        }
    }

    /* LoRA adapters (1.6, F6). Strict typing throughout, per the 1.3
     * session-options policy — a mistyped entry is an error, not a silently
     * ignored setting. */
    if (j.contains("lora") && !j["lora"].is_null()) {
        if (!j["lora"].is_array())
            throw_validation("lora must be an array of objects", "lora");
        for (size_t i = 0; i < j["lora"].size(); ++i) {
            const std::string param = "lora[" + std::to_string(i) + "]";
            const nj& e = j["lora"][i];
            if (!e.is_object())
                throw_validation(param + " must be an object", "lora");
            ModelLoraOptions lo;
            if (!e.contains("path") || !e["path"].is_string())
                throw_validation(param + " requires a string 'path'", "lora");
            lo.path = e["path"].get<std::string>();
            if (e.contains("scale") && !e["scale"].is_null()) {
                if (!e["scale"].is_number())
                    throw_validation(param + ".scale must be a number", "lora");
                lo.scale = e["scale"].get<float>();
            }
            if (e.contains("name") && !e["name"].is_null()) {
                if (!e["name"].is_string() || e["name"].get<std::string>().empty())
                    throw_validation(param + ".name must be a non-empty string",
                                     "lora");
                lo.name = e["name"].get<std::string>();
            } else {
                lo.name = std::filesystem::path(lo.path).stem().string();
            }
            /* Names flow into hand-built describe JSON and error messages:
             * keep them to a safe charset rather than escaping downstream. */
            for (char c : lo.name) {
                if (!std::isalnum(static_cast<unsigned char>(c)) &&
                    c != '.' && c != '_' && c != '-') {
                    throw_validation(
                        param + ": adapter name \"" + lo.name + "\" contains "
                        "unsupported characters (allowed: letters, digits, "
                        "'.', '_', '-'); set 'name' explicitly", "lora");
                }
            }
            for (const auto& prev : opts.lora) {
                if (prev.name == lo.name)
                    throw_validation("duplicate lora adapter name \"" + lo.name +
                                     "\" (names must be unique; set 'name' "
                                     "explicitly to disambiguate)", "lora");
            }
            opts.lora.push_back(std::move(lo));
        }
    }
    if (!opts.lora.empty() && opts.vocab_only)
        throw_validation("lora cannot be combined with vocab_only "
                         "(adapters attach to weight tensors that a "
                         "vocab-only load never reads)", "lora");

    return opts;
}

// Parse "auto", "all", or integer string → internal sentinel / count.
// Accepted values: "auto"/-1 (auto-fit), "all"/-2 (full offload), 0 (CPU only),
// or a positive layer count. Any other value (junk, or a negative other than
// the two sentinels) is a user error and is rejected rather than silently
// coerced to "auto".
static int parse_gpu_layers_request(const std::string& s) {
    if (s.empty() || s == "auto") return kGpuLayersAuto;
    if (s == "all")               return kGpuLayersAll;
    int v = 0;
    try {
        size_t pos = 0;
        v = std::stoi(s, &pos);
        if (pos != s.size()) throw std::invalid_argument("trailing characters");
    } catch (const std::exception&) {
        throw_validation(
            "n_gpu_layers must be \"auto\", \"all\", or an integer "
            "(0 = CPU only, -1 = auto, -2 = all, positive = layer count)",
            "n_gpu_layers");
    }
    if (v < kGpuLayersAll) { /* more negative than -2 is not a valid sentinel */
        throw_validation(
            "n_gpu_layers integer must be >= -2 "
            "(0 = CPU only, -1 = auto, -2 = all, positive = layer count)",
            "n_gpu_layers");
    }
    return v;
}

/* Adapter between llama.cpp's progress callback (return true = continue)
 * and the public helix_load_progress_cb (return non-zero = cancel). Records
 * whether the callback aborted the load so the constructor can distinguish
 * cancellation from a genuine load failure. */
struct LoadProgressState {
    helix_load_progress_cb cb;
    void*                  user_data;
    bool                   cancelled = false;
};

static bool load_progress_trampoline(float progress, void* user_data) {
    auto* st = static_cast<LoadProgressState*>(user_data);
    if (st->cb(st->user_data, progress) != 0) {
        st->cancelled = true;
        return false;
    }
    return true;
}

Model::Model(Runtime& rt, const ModelOptions& opts,
             helix_load_progress_cb progress_cb,
             void* progress_user_data) : rt_(rt), opts_(opts) {
    std::error_code ec;
    if (!std::filesystem::exists(opts.model_path, ec)) {
        throw Error(HELIX_E_MODEL_NOT_FOUND,
                    "model file not found: " + opts.model_path,
                    "model_not_found", "model_path", "helix_e_model_not_found");
    }

    if (!opts.mmproj_path.empty() &&
        !std::filesystem::exists(opts.mmproj_path, ec)) {
        throw Error(HELIX_E_MODEL_NOT_FOUND,
                    "mmproj file not found: " + opts.mmproj_path,
                    "model_not_found", "mmproj_path", "helix_e_model_not_found");
    }

    /* Adapter files are checked before the (potentially minutes-long) model
     * load so a typo'd path fails fast. */
    for (const auto& lo : opts.lora) {
        if (!std::filesystem::exists(lo.path, ec)) {
            throw Error(HELIX_E_MODEL_NOT_FOUND,
                        "lora adapter file not found: " + lo.path,
                        "model_not_found", "lora", "helix_e_model_not_found");
        }
    }

    // ------------------------------------------------------------------
    // Phase 1 of load: read layer count / training ctx from GGUF metadata.
    // A vocab-only llama probe cannot be used here: load_hparams() returns
    // before reading block_count when vocab_only is set, so
    // llama_model_n_layer() would always report 0 and the GPU layer plan
    // would silently fall back to CPU-only.
    // ------------------------------------------------------------------
    int total_layers = 0;
    int training_ctx = 0;
    {
        gguf_init_params gparams;
        gparams.no_alloc = true;
        gparams.ctx      = nullptr;
        gguf_context* gctx = gguf_init_from_file(opts.model_path.c_str(), gparams);
        if (!gctx) {
            throw Error(HELIX_E_MODEL_LOAD_FAILED,
                        "failed to probe model for metadata (GGUF read failed): " + opts.model_path,
                        "model_load_failed", "model_path", "helix_e_model_load_failed");
        }
        auto get_int = [gctx](const std::string& key) -> int {
            const int64_t idx = gguf_find_key(gctx, key.c_str());
            if (idx < 0) return 0;
            switch (gguf_get_kv_type(gctx, idx)) {
                case GGUF_TYPE_UINT32: return static_cast<int>(gguf_get_val_u32(gctx, idx));
                case GGUF_TYPE_INT32:  return static_cast<int>(gguf_get_val_i32(gctx, idx));
                case GGUF_TYPE_UINT64: return static_cast<int>(gguf_get_val_u64(gctx, idx));
                case GGUF_TYPE_INT64:  return static_cast<int>(gguf_get_val_i64(gctx, idx));
                default: return 0;
            }
        };
        const int64_t arch_idx = gguf_find_key(gctx, "general.architecture");
        if (arch_idx >= 0 && gguf_get_kv_type(gctx, arch_idx) == GGUF_TYPE_STRING) {
            const std::string arch = gguf_get_val_str(gctx, arch_idx);
            total_layers = get_int(arch + ".block_count");
            training_ctx = get_int(arch + ".context_length");
        }
        /* Rank-head detection for helix_rerank (1.5): the classification
         * head is only visible as tensors ("cls.weight" — dense head — or
         * "cls.output.weight" — direct projection). Many reranker GGUFs
         * (e.g. bge-reranker-v2-m3) carry these tensors WITHOUT the
         * classifier.output_labels metadata, so llama_model_cls_label()
         * cannot be used for this. */
        {
            const int64_t n_tensors = gguf_get_n_tensors(gctx);
            for (int64_t i = 0; i < n_tensors; ++i) {
                const char* name = gguf_get_tensor_name(gctx, i);
                if (strcmp(name, "cls.weight") == 0 ||
                    strcmp(name, "cls.output.weight") == 0) {
                    has_cls_head_ = true;
                    break;
                }
            }
        }
        gguf_free(gctx);
    }

    // ------------------------------------------------------------------
    // GPU layer selection.
    // ------------------------------------------------------------------
    const HardwareProfile& hw = rt_.profile();
    const int user_req = parse_gpu_layers_request(opts.n_gpu_layers_str);
    GpuLayerPlan plan  = pick_n_gpu_layers(hw, user_req,
                                           opts.model_path, total_layers,
                                           opts.n_ctx > 0 ? opts.n_ctx : (training_ctx > 0 ? training_ctx : 4096));
    n_gpu_layers_actual_ = plan.n_gpu_layers;
    log_info("gpu layer plan: " + plan.rule);

    // ------------------------------------------------------------------
    // Full model load with chosen n_gpu_layers.
    // ------------------------------------------------------------------
    auto params         = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers_actual_;
    params.use_mmap     = opts.use_mmap;
    params.use_mlock    = opts.use_mlock;
    params.vocab_only   = opts.vocab_only;

    LoadProgressState progress_state{progress_cb, progress_user_data};
    if (progress_cb) {
        params.progress_callback           = load_progress_trampoline;
        params.progress_callback_user_data = &progress_state;
    }

    log_info("loading model: " + opts.model_path +
             " (gpu_layers=" + std::to_string(n_gpu_layers_actual_) + ")");
    model_ = llama_model_load_from_file(opts.model_path.c_str(), params);
    if (!model_) {
        if (progress_state.cancelled) {
            throw Error(HELIX_E_CANCELLED,
                        "model load cancelled by progress callback: " + opts.model_path,
                        "cancel", "", "helix_e_cancelled");
        }
        throw_model_load_failed(opts.model_path);
    }

    /* common_chat_templates_init takes a const std::string& — pass empty string
     * (not nullptr) when there is no override. */
    auto tmpls_ptr = common_chat_templates_init(model_,
                                                opts.chat_template_override);
    if (!tmpls_ptr) {
        throw Error(HELIX_E_MODEL_LOAD_FAILED,
                    "failed to initialise chat templates from model: " + opts.model_path,
                    "model_load_failed", "model_path", "helix_e_model_load_failed");
    }
    tmpls_.reset(tmpls_ptr.release());

    /* Load mmproj if provided. */
    if (!opts.mmproj_path.empty()) {
        mmproj_ = std::make_unique<MmProj>(
            opts.mmproj_path.c_str(),
            model_,
            n_gpu_layers_actual_ > 0,
            0 /* n_threads: 0 = auto */);
    }

    /* LoRA adapters (1.6, F6). Handles are owned by llama and freed with the
     * model. A constructor throw skips the destructor, so free the model on
     * the error paths (which also frees any adapters already loaded). */
    for (const auto& lo : opts.lora) {
        try {
            log_info("loading lora adapter \"" + lo.name + "\": " + lo.path);
            llama_adapter_lora* adapter =
                llama_adapter_lora_init(model_, lo.path.c_str());
            if (!adapter) {
                throw Error(HELIX_E_MODEL_LOAD_FAILED,
                            "failed to load lora adapter \"" + lo.name + "\" (" +
                            lo.path + "): not a LoRA GGUF, or its tensors do "
                            "not match this base model",
                            "model_load_failed", "lora",
                            "helix_e_model_load_failed");
            }
            /* Activated LoRAs (aLoRA) only apply after their invocation-token
             * sequence appears in the context — semantics the decode loop
             * does not implement. Honest rejection over silent misbehaviour. */
            if (llama_adapter_get_alora_n_invocation_tokens(adapter) > 0) {
                throw Error(HELIX_E_UNSUPPORTED_FEATURE,
                            "lora adapter \"" + lo.name + "\" is an activated "
                            "LoRA (aLoRA), which is not supported in this "
                            "release",
                            "unsupported_feature_error", "lora",
                            "helix_e_unsupported_feature");
            }
            loras_.push_back({lo.name, lo.path, lo.scale, adapter});
        } catch (...) {
            mmproj_.reset();
            tmpls_.reset();
            llama_model_free(model_);
            model_ = nullptr;
            throw;
        }
    }

    build_describe();
    log_info("model loaded: " + opts.alias);
}

Model::~Model() {
    mmproj_.reset();
    tmpls_.reset();
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
}

bool Model::supports_vision() const { return mmproj_ && mmproj_->supports_vision(); }
bool Model::supports_audio()  const { return mmproj_ && mmproj_->supports_audio();  }

int Model::n_ctx_train() const {
    return llama_model_n_ctx_train(model_);
}

int Model::n_vocab() const {
    return llama_vocab_n_tokens(llama_model_get_vocab(model_));
}

uint32_t Model::embedding_dim() const {
    /* 0 = this model cannot serve embeddings on helix_embeddings:
     * encoder-decoder architectures, or classifier/reranker heads.
     * llama_model_n_embd_out (not the deprecated llama_n_embd) because the
     * pooled output dimension diverges from the hidden size for some
     * architectures, and dimension negotiation must see the output. */
    if (llama_model_has_encoder(model_) && llama_model_has_decoder(model_)) return 0;
    /* Classifier/reranker detection: classifier models carry output labels
     * in their GGUF metadata. llama_model_n_cls_out() cannot be used here —
     * it defaults to 1 for every model and is only meaningful when labels
     * are present. */
    if (llama_model_cls_label(model_, 0) != nullptr) return 0;

    const int32_t d = llama_model_n_embd_out(model_);
    return d > 0 ? static_cast<uint32_t>(d) : 0;
}

std::string Model::tokenize_json(const char* text,
                                 bool add_special,
                                 bool parse_special) const {
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    const std::vector<llama_token> toks =
        common_tokenize(vocab, text, add_special, parse_special);

    std::string out = "[";
    for (size_t i = 0; i < toks.size(); ++i) {
        if (i) out += ',';
        out += std::to_string(toks[i]);
    }
    out += ']';
    return out;
}

uint64_t Model::state_fingerprint() const {
    /* FNV-1a, 64-bit. */
    uint64_t h = 1469598103934665603ull;
    auto mix_bytes = [&h](const void* data, size_t n) {
        const unsigned char* p = static_cast<const unsigned char*>(data);
        for (size_t i = 0; i < n; ++i) {
            h ^= p[i];
            h *= 1099511628211ull;
        }
    };
    auto mix_u64 = [&](uint64_t v) { mix_bytes(&v, sizeof v); };

    char desc[128] = {0};
    llama_model_desc(model_, desc, sizeof(desc));
    mix_bytes(desc, strnlen(desc, sizeof(desc)));
    mix_u64(llama_model_n_params(model_));
    mix_u64(llama_model_size(model_));
    mix_u64(static_cast<uint64_t>(n_vocab()));
    mix_u64(static_cast<uint64_t>(llama_model_n_layer(model_)));
    mix_u64(static_cast<uint64_t>(llama_model_n_embd(model_)));
    mix_u64(static_cast<uint64_t>(n_ctx_train()));
    return h;
}

void Model::build_describe() {
    /* A stable content hash of the loaded model's identity (arch, params, size,
     * vocab, layers, embd, trained ctx), for replay/audit chains that need to
     * confirm a completion was produced by the same model (helix ABI 1.7). */
    char hash_hex[17] = {0};
    std::snprintf(hash_hex, sizeof(hash_hex), "%016llx",
                  static_cast<unsigned long long>(state_fingerprint()));

    nj j = {
        {"alias",        opts_.alias},
        {"model_path",   opts_.model_path},
        {"model_hash",   hash_hex},
        {"n_ctx_train",  n_ctx_train()},
        {"n_vocab",      n_vocab()},
        {"n_params",     llama_model_n_params(model_)},
        {"n_layers",     llama_model_n_layer(model_)},
        /* Matches helix_model_embedding_dim(): 0 when the model cannot
         * serve embeddings on helix_embeddings. */
        {"n_embd_out",   embedding_dim()},
        {"use_mmap",     opts_.use_mmap},
        {"use_mlock",    opts_.use_mlock},
        {"n_gpu_layers_requested", opts_.n_gpu_layers_str},
        {"n_gpu_layers_actual",    n_gpu_layers_actual_},
        {"phase",        HELIX_PROGRAM_PHASE}
    };
    nj loras_j = nj::array();
    for (const auto& l : loras_) {
        loras_j.push_back({{"name",  l.name},
                           {"path",  l.path},
                           {"scale", l.scale}});
    }
    j["lora"] = loras_j;
    describe_json_ = j.dump(2);
}

} // namespace helix
