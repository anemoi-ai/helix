#pragma once
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "helix.h"
#include "chat.h"

/* Forward-declare llama types to avoid leaking them. */
struct llama_model;
struct llama_adapter_lora;

namespace helix {

class MmProj;
class Runtime;

/* One LoRA adapter entry in the model-load options (1.6, F6). */
struct ModelLoraOptions {
    std::string path;
    std::string name;          /* default: adapter file stem */
    float       scale = 1.0f;  /* default activation scale */
};

struct ModelOptions {
    std::string alias         = "local-helix";
    std::string model_path;
    std::string mmproj_path;
    std::string chat_template_override;
    /* reasoning_format: "auto" | "deepseek-r1" | "qwq" | "none"
     * Controls how <think> blocks are extracted from model output. */
    std::string reasoning_format = "auto";
    // n_gpu_layers_str: "auto" (default), "all", or an integer string.
    // Parsed in the constructor into the internal n_gpu_layers int.
    std::string n_gpu_layers_str = "auto";
    int         n_ctx         = 0;     /* 0 = model training ctx */
    bool        use_mmap      = true;
    bool        use_mlock     = false;
    bool        vocab_only    = false;
    /* LoRA adapters (1.6, F6): validated and loaded with the model, owned by
     * it (llama frees loaded adapters with the model). Which adapters a
     * session activates — and at what scale — is a session option; the
     * scale here is the default. Rejected with vocab_only. */
    std::vector<ModelLoraOptions> lora;
};

class Model {
public:
    /* progress_cb (optional, may be null) is invoked on this thread during
     * the tensor-loading phase; a non-zero return cancels the load and the
     * constructor throws HELIX_E_CANCELLED. Backs helix_model_load_ex. */
    Model(Runtime& rt, const ModelOptions& opts,
          helix_load_progress_cb progress_cb = nullptr,
          void* progress_user_data = nullptr);
    ~Model();

    llama_model* llama_model_ptr() const { return model_; }

    const std::string&  alias()           const { return opts_.alias; }
    const ModelOptions& options()         const { return opts_; }
    // describe_json_ is set once in the constructor and never mutated afterwards,
    // so the c_str() pointer returned by helix_model_describe() is stable for
    // the lifetime of the Model object.
    const std::string&  describe()        const { return describe_json_; }
    Runtime&            runtime()         const { return rt_; }
    int                 n_gpu_layers_actual() const { return n_gpu_layers_actual_; }

    // Per-model CUDA lock (acquired before the per-session lock on CUDA builds).
    std::mutex& cuda_mu() { return cuda_mu_; }

    /* Chat templates wrapper (opaque pointer to common_chat_templates). */
    const common_chat_templates* chat_templates() const { return tmpls_.get(); }

    /* LoRA adapters loaded with the model (1.6, F6). Adapter handles are
     * owned by llama and freed with the model; `scale` is the load-time
     * default a session applies unless its own lora option overrides it. */
    struct LoadedLora {
        std::string         name;
        std::string         path;
        float               scale;
        llama_adapter_lora* adapter;
    };
    const std::vector<LoadedLora>& loras() const { return loras_; }

    /* Phase 7: multimodal projector (null when no mmproj_path set). */
    MmProj* mmproj()         const { return mmproj_.get(); }
    bool    supports_vision() const;
    bool    supports_audio()  const;

    int n_ctx_train() const;
    int n_vocab()     const;

    /* Pooled embedding output dimension; 0 when the model cannot serve
     * embeddings (encoder-decoder, or reranker/classifier head). Backs
     * helix_model_embedding_dim(). */
    uint32_t embedding_dim() const;

    /* True when the GGUF carries classification-head tensors ("cls.weight"
     * or "cls.output.weight") — the graph-level requirement for rank
     * pooling. Detected from tensor names at load (labels metadata is
     * absent in many reranker GGUFs). */
    bool has_cls_head() const { return has_cls_head_; }

    /* Tokenize `text` with the model's vocabulary and return the token ids
     * as a JSON array string ("[1,2,3]"). Backs helix_tokenize(). Pure
     * query: thread-safe, no state mutated. */
    std::string tokenize_json(const char* text,
                              bool add_special,
                              bool parse_special) const;

    /* Stable identity fingerprint for session state files (1.5): FNV-1a over
     * architecture description, parameter count, file size, vocab size,
     * layer count, embedding width, and training context. Cheap to compute
     * (metadata only — no file hashing) yet strong enough that state from a
     * different model cannot pass the restore check. */
    uint64_t state_fingerprint() const;

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

private:
    Runtime&     rt_;
    ModelOptions opts_;
    llama_model* model_ = nullptr;
    int          n_gpu_layers_actual_ = 0;
    bool         has_cls_head_ = false;
    std::mutex   cuda_mu_;

    std::unique_ptr<common_chat_templates, common_chat_templates_deleter> tmpls_;

    std::unique_ptr<MmProj> mmproj_;

    std::vector<LoadedLora> loras_;

    std::string describe_json_;

    void build_describe();
};

ModelOptions parse_model_options(const char* model_json);

} // namespace helix
