#pragma once
#include <atomic>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include "helix.h"

struct llama_context;
struct llama_model;
struct common_speculative;
typedef int32_t llama_token;

namespace helix {

class Model;

enum class SpeculativeType {
    None,
    DraftMtp,
};

struct SpeculativeOptions {
    SpeculativeType type = SpeculativeType::None;

    /* Reserved for future separate-draft-model support. Empty means create
     * the MTP context from the target model (Branch B, requires the target
     * GGUF to carry MTP head tensors). Set to a separate draft model path
     * (Branch C, e.g. Gemma-4's gemma4-assistant head) to load and link a
     * standalone MTP head. */
    std::string draft_model_path;

    int32_t n_max = 3;       /* max drafted tokens per iteration */
    int32_t n_min = 0;       /* drafts shorter than this are discarded */
    float   p_min = 0.0f;    /* greedy cutoff probability for drafting */
    bool    backend_sampling = true;

    /* KV cache types as strings ("f16", "q8_0", ...), parsed to ggml_type at
     * context creation. Mirrors Helix's n_gpu_layers string convention. */
    std::string cache_type_k = "f16";
    std::string cache_type_v = "f16";
};

/* One entry in the session's lora option (1.6, F6): activate the named
 * adapter loaded on the model, optionally overriding its load-time scale. */
struct SessionLoraOptions {
    std::string          name;
    std::optional<float> scale;   /* absent = adapter's load-time scale */
};

struct SessionOptions {
    int  n_ctx              = 0;    /* 0 = model default */
    int  n_batch            = 0;    /* 0 = auto: profile-derived */
    int  n_threads          = 0;    /* 0 = auto: profile-derived */
    int  n_threads_batch    = 0;    /* 0 = auto: profile-derived */
    bool prefix_cache       = true;
    bool warmup             = true;
    /* Context-overflow handling (1.6). When true, chat requests that hit the
     * context wall recover instead of failing: prefill overflow drops the
     * oldest non-system history messages before rendering (template-aware),
     * and generation overflow evicts the oldest non-system KV cells in blocks
     * and continues decoding. Responses report {"helix": {"context_shifted":
     * true, "evicted_tokens": N}} when it happened. Rejected at session
     * create for embedding/MTP sessions and for models whose KV cache cannot
     * be position-shifted (llama_memory_can_shift() == false). */
    bool context_shift      = false;
    unsigned int seed       = 0;    /* 0 = LLAMA_DEFAULT_SEED */
    int  stream_coalesce_ms = 20;
    /* Embedding-mode session: accepts helix_embeddings, rejects chat (and
     * vice versa). Forces prefix_cache off; seed and stream_coalesce_ms are
     * ignored (no sampling, no streaming). */
    bool embedding          = false;
    /* llama_pooling_type as int to keep llama.h out of this header.
     * -1 = UNSPECIFIED (defer to model metadata). Parsed from the "pooling"
     * session option: "none" | "mean" | "cls" | "last" | "rank". */
    int  pooling            = -1;
    /* Full-size SWA (sliding-window attention) KV cache. When true (the
     * llama.cpp default), SWA layers store KV for all n_ctx positions. When
     * false, SWA layers only store the sliding window (n_swa tokens, typically
     * 1024), reducing memory dramatically for large-context models like
     * Gemma-4. Set to false if the KV cache is too large. */
    bool swa_full            = true;
    /* Main-context KV cache types ("f16", "q8_0", ...), parsed to ggml_type
     * at context creation like the speculative pair below. A quantized
     * cache_type_v requires flash attention (rejected at session create with
     * HELIX_E_VALIDATION when the hardware profile disables it). */
    std::string cache_type_k = "f16";
    std::string cache_type_v = "f16";
    /* Multi-token-prediction speculative decoding. When type != None, the
     * session creates an MTP context (from the target model or a separate
     * draft model) and drives draft verification in the decode loop.
     * Forces prefix_cache off. */
    SpeculativeOptions speculative;
    /* LoRA adapter activation (1.6, F6). Absent: every adapter loaded on the
     * model is active at its load-time scale. Present: exactly the named
     * adapters are active (empty array = none). Unknown names are rejected
     * at session create with HELIX_E_VALIDATION; sessions with any active
     * adapter reject embedding and speculative (MTP) modes in this release. */
    std::optional<std::vector<SessionLoraOptions>> lora;
};

class Session {
public:
    Session(Model& model, const SessionOptions& opts);
    ~Session();

    helix_status_t chat_completions(const char* request_json,
                                    char** out_response_json);

    helix_status_t chat_completions_stream(const char* request_json,
                                           helix_stream_cb on_chunk,
                                           void* user_data);

    helix_status_t embeddings(const char* request_json,
                              char** out_response_json);

    /* Rerank documents against a query. Requires a rerank session
     * ({"embedding": true, "pooling": "rank"} on a reranker-head model).
     * Backs helix_rerank. */
    helix_status_t rerank(const char* request_json,
                          char** out_response_json);

    /* Session state persistence (1.5). Chat sessions with the prefix cache
     * enabled only; both take the session mutex (wait for any in-flight
     * request). Restore leaves the session empty and usable on failure.
     * Backs helix_session_save / helix_session_restore. */
    helix_status_t save_state(const char* path);
    helix_status_t restore_state(const char* path);

    /* Token count a chat request would occupy after template rendering.
     * Pure query: no KV-cache access, no locks taken — safe concurrently
     * with an in-flight request on this session. Backs helix_count_tokens. */
    helix_status_t count_tokens(const char* request_json,
                                uint32_t* out_token_count);

    void cancel();

    llama_context* ctx()        const { return ctx_; }
    const Model&         model()      const { return model_; }
    std::atomic<bool>& cancel_flag()        { return cancel_requested_; }

    bool embedding_mode()    const { return opts_.embedding; }
    /* True when the session serves helix_rerank (embedding session whose
     * effective pooling resolved to "rank"). Mutually exclusive with
     * helix_embeddings. */
    bool rerank_mode()       const { return rerank_mode_; }
    /* Effective llama_pooling_type resolved at context creation (embedding
     * sessions only; -1 otherwise). */
    int  effective_pooling() const { return effective_pooling_; }
    /* Encoder-only models (BERT family) embed via llama_encode. */
    bool embed_via_encode()  const { return embed_via_encode_; }

    /* Prefix-cache state. INVARIANT: last_tokens_ either is empty or exactly
     * mirrors the token contents of KV seq 0. It is mutated only from within
     * run_chat_completions, which runs under mu_ (held for the entire call,
     * see do_chat_completions). These accessors are therefore NOT internally
     * synchronised — callers must hold mu_. Do not read last_tokens_ from
     * another thread (e.g. telemetry) without taking mu_ first. */
    const std::vector<llama_token>& last_tokens() const { return last_tokens_; }
    void set_last_tokens(std::vector<llama_token> t)    { last_tokens_ = std::move(t); }
    bool prefix_cache_enabled() const { return opts_.prefix_cache; }
    bool context_shift_enabled() const { return opts_.context_shift; }
    int  stream_coalesce_ms()   const { return opts_.stream_coalesce_ms; }

    /* MTP speculative state. Non-null only when the session was created with
     * speculative.type == DraftMtp and the MTP context/manager initialised
     * successfully. ctx_mtp() is the draft/MTP context; spec_mtp() is the
     * upstream common_speculative manager that drives draft generation and
     * accept. Both are owned by the Session and valid until destruction. */
    bool has_speculative() const { return spec_mtp_ != nullptr; }
    common_speculative* spec_mtp() const { return spec_mtp_; }
    llama_context*      ctx_mtp()  const { return ctx_mtp_; }

    /* JSON description for helix_session_describe (1.2). */
    std::string describe() const;

    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

private:
    Model&         model_;
    SessionOptions opts_;
    llama_context* ctx_ = nullptr;
    std::mutex     mu_;
    std::atomic<bool> cancel_requested_{false};
    std::atomic<bool> warmup_done_{false};

    int  effective_pooling_ = -1;
    bool embed_via_encode_  = false;
    bool flash_attn_        = false;
    bool rerank_mode_       = false;

    /* Adapters active on ctx_ (resolved at creation, immutable after; an
     * adapter change is a new session). Kept for describe(). */
    struct ActiveLora { std::string name; float scale; };
    std::vector<ActiveLora> active_loras_;

    std::vector<llama_token> last_tokens_;

    /* MTP speculative state. Ordering matters for destruction: spec_mtp_
     * before ctx_mtp_ before model_dft_ before ctx_. model_dft_ is null for
     * the Branch B (MTP-from-target) path. ctx_other on ctx_mtp_ aliases ctx_,
     * so ctx_mtp_ must be freed before ctx_. */
    llama_model*         model_dft_ = nullptr;
    llama_context*       ctx_mtp_   = nullptr;
    common_speculative*  spec_mtp_  = nullptr;

    void maybe_run_warmup();

    /* Throws HELIX_E_UNSUPPORTED_FEATURE unless this session's state can be
     * persisted (chat session, no MTP, prefix cache on). `what` names the
     * calling API in the error message. */
    void require_persistable(const char* what) const;

    helix_status_t do_chat_completions(const char* request_json,
                                       char** out_response_json,
                                       helix_stream_cb on_chunk,
                                       void* user_data,
                                       bool streaming);
};

SessionOptions parse_session_options(const char* session_json);

} // namespace helix
