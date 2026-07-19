#include "session.hpp"
#include "../internal/error.hpp"
#include "llama.h"
#include "nlohmann/json.hpp"

namespace helix {

using nj = nlohmann::json;

SessionOptions parse_session_options(const char* session_json) {
    SessionOptions opts;
    if (!session_json || session_json[0] == '\0') return opts;

    nj j;
    try {
        j = nj::parse(session_json);
    } catch (const nj::parse_error& e) {
        throw_invalid_json(std::string("session options JSON parse error: ") + e.what());
    }

    /* Strict typing: a present-but-mistyped option (e.g. {"n_ctx": "4096"})
     * is an error, matching the request parser, rather than a silently
     * ignored setting. null is treated as absent throughout. */
    auto get_int_opt = [&](const char* key) -> const nj* {
        auto it = j.find(key);
        if (it == j.end() || it->is_null()) return nullptr;
        if (!it->is_number_integer())
            throw helix::Error(HELIX_E_VALIDATION,
                std::string(key) + " must be an integer",
                "invalid_request_error", key, "helix_e_validation");
        return &*it;
    };
    auto get_bool_opt = [&](const char* key) -> const nj* {
        auto it = j.find(key);
        if (it == j.end() || it->is_null()) return nullptr;
        if (!it->is_boolean())
            throw helix::Error(HELIX_E_VALIDATION,
                std::string(key) + " must be a boolean",
                "invalid_request_error", key, "helix_e_validation");
        return &*it;
    };
    auto get_str_opt = [&](const char* key) -> const nj* {
        auto it = j.find(key);
        if (it == j.end() || it->is_null()) return nullptr;
        if (!it->is_string())
            throw helix::Error(HELIX_E_VALIDATION,
                std::string(key) + " must be a string",
                "invalid_request_error", key, "helix_e_validation");
        return &*it;
    };

    if (const nj* v = get_int_opt("n_ctx")) {
        opts.n_ctx = v->get<int>();
        if (opts.n_ctx < 0)
            throw helix::Error(HELIX_E_VALIDATION,
                "n_ctx must be non-negative, got " + std::to_string(opts.n_ctx),
                "invalid_request_error", "n_ctx", "helix_e_validation");
    }
    if (const nj* v = get_int_opt("n_batch")) {
        opts.n_batch = v->get<int>();
        if (opts.n_batch < 0)
            throw helix::Error(HELIX_E_VALIDATION,
                "n_batch must be non-negative, got " + std::to_string(opts.n_batch),
                "invalid_request_error", "n_batch", "helix_e_validation");
    }
    if (const nj* v = get_int_opt("n_threads")) {
        opts.n_threads = v->get<int>();
        if (opts.n_threads < 0)
            throw helix::Error(HELIX_E_VALIDATION,
                "n_threads must be non-negative, got " + std::to_string(opts.n_threads),
                "invalid_request_error", "n_threads", "helix_e_validation");
    }
    if (const nj* v = get_int_opt("n_threads_batch")) {
        opts.n_threads_batch = v->get<int>();
        if (opts.n_threads_batch < 0)
            throw helix::Error(HELIX_E_VALIDATION,
                "n_threads_batch must be non-negative, got " + std::to_string(opts.n_threads_batch),
                "invalid_request_error", "n_threads_batch", "helix_e_validation");
    }
    if (const nj* v = get_bool_opt("prefix_cache"))
        opts.prefix_cache = v->get<bool>();
    /* context_shift was documented since 1.1 but only implemented in 1.6
     * (an explicit true was rejected in 1.3–1.5). Capability validation —
     * embedding/MTP exclusion, llama_memory_can_shift — happens at session
     * creation where the model is known. */
    if (const nj* v = get_bool_opt("context_shift"))
        opts.context_shift = v->get<bool>();
    if (const nj* v = get_bool_opt("swa_full"))
        opts.swa_full = v->get<bool>();
    /* Main-context KV cache types (1.4). The value set is validated against
     * ggml at session creation (parse_cache_type), where an unknown name or
     * a quantized V without flash attention is rejected. */
    if (const nj* v = get_str_opt("cache_type_k"))
        opts.cache_type_k = v->get<std::string>();
    if (const nj* v = get_str_opt("cache_type_v"))
        opts.cache_type_v = v->get<std::string>();
    if (const nj* v = get_bool_opt("warmup"))
        opts.warmup = v->get<bool>();
    if (const nj* v = get_int_opt("seed")) {
        int64_t seed_val = v->get<int64_t>();
        if (seed_val < 0) {
            throw helix::Error(HELIX_E_VALIDATION,
                "seed must be non-negative, got " + std::to_string(seed_val),
                "invalid_request_error", "seed", "helix_e_validation");
        }
        if (seed_val > static_cast<int64_t>(UINT32_MAX)) {
            throw helix::Error(HELIX_E_VALIDATION,
                "seed must be <= " + std::to_string(UINT32_MAX) +
                ", got " + std::to_string(seed_val),
                "invalid_request_error", "seed", "helix_e_validation");
        }
        opts.seed = static_cast<unsigned int>(seed_val);
    }
    if (const nj* v = get_int_opt("stream_coalesce_ms")) {
        opts.stream_coalesce_ms = v->get<int>();
        if (opts.stream_coalesce_ms < 0)
            throw helix::Error(HELIX_E_VALIDATION,
                "stream_coalesce_ms must be non-negative, got " + std::to_string(opts.stream_coalesce_ms),
                "invalid_request_error", "stream_coalesce_ms", "helix_e_validation");
    }
    if (const nj* v = get_bool_opt("embedding"))
        opts.embedding = v->get<bool>();
    if (j.contains("pooling") && !j["pooling"].is_null()) {
        if (!j["pooling"].is_string())
            throw helix::Error(HELIX_E_VALIDATION,
                "pooling must be a string",
                "invalid_request_error", "pooling", "helix_e_validation");
        const std::string p = j["pooling"].get<std::string>();
        if      (p == "none") opts.pooling = LLAMA_POOLING_TYPE_NONE;
        else if (p == "mean") opts.pooling = LLAMA_POOLING_TYPE_MEAN;
        else if (p == "cls")  opts.pooling = LLAMA_POOLING_TYPE_CLS;
        else if (p == "last") opts.pooling = LLAMA_POOLING_TYPE_LAST;
        else if (p == "rank") opts.pooling = LLAMA_POOLING_TYPE_RANK;
        else
            throw helix::Error(HELIX_E_VALIDATION,
                "pooling must be \"none\", \"mean\", \"cls\", \"last\", or "
                "\"rank\" (got \"" + p + "\")",
                "invalid_request_error", "pooling", "helix_e_validation");
    }

    /* LoRA adapter activation (1.6, F6). Names are resolved against the
     * model's loaded adapters at session creation, where the model is known.
     * An explicit empty array means "no adapters active" — distinct from an
     * absent option (all adapters at load-time scales). */
    if (j.contains("lora") && !j["lora"].is_null()) {
        if (!j["lora"].is_array())
            throw helix::Error(HELIX_E_VALIDATION,
                "lora must be an array of objects",
                "invalid_request_error", "lora", "helix_e_validation");
        std::vector<SessionLoraOptions> entries;
        for (size_t i = 0; i < j["lora"].size(); ++i) {
            const std::string param = "lora[" + std::to_string(i) + "]";
            const nj& e = j["lora"][i];
            if (!e.is_object())
                throw helix::Error(HELIX_E_VALIDATION,
                    param + " must be an object",
                    "invalid_request_error", "lora", "helix_e_validation");
            SessionLoraOptions sl;
            if (!e.contains("name") || !e["name"].is_string() ||
                e["name"].get<std::string>().empty())
                throw helix::Error(HELIX_E_VALIDATION,
                    param + " requires a non-empty string 'name'",
                    "invalid_request_error", "lora", "helix_e_validation");
            sl.name = e["name"].get<std::string>();
            if (e.contains("scale") && !e["scale"].is_null()) {
                if (!e["scale"].is_number())
                    throw helix::Error(HELIX_E_VALIDATION,
                        param + ".scale must be a number",
                        "invalid_request_error", "lora", "helix_e_validation");
                sl.scale = e["scale"].get<float>();
            }
            for (const auto& prev : entries) {
                if (prev.name == sl.name)
                    throw helix::Error(HELIX_E_VALIDATION,
                        "duplicate lora adapter name \"" + sl.name + "\"",
                        "invalid_request_error", "lora", "helix_e_validation");
            }
            entries.push_back(std::move(sl));
        }
        opts.lora = std::move(entries);
    }

    if (j.contains("speculative") && !j["speculative"].is_null()) {
        if (!j["speculative"].is_object())
            throw helix::Error(HELIX_E_VALIDATION,
                "speculative must be an object",
                "invalid_request_error", "speculative", "helix_e_validation");
        const auto& s = j["speculative"];
        if (s.contains("type") && !s["type"].is_null()) {
            if (!s["type"].is_string())
                throw helix::Error(HELIX_E_VALIDATION,
                    "speculative.type must be a string",
                    "invalid_request_error", "speculative.type", "helix_e_validation");
            const std::string t = s["type"].get<std::string>();
            if      (t == "none")      opts.speculative.type = SpeculativeType::None;
            else if (t == "draft-mtp") opts.speculative.type = SpeculativeType::DraftMtp;
            else
                throw helix::Error(HELIX_E_VALIDATION,
                    "speculative.type must be \"none\" or \"draft-mtp\" (got \""
                    + t + "\")",
                    "invalid_request_error", "speculative.type", "helix_e_validation");
        }
        if (s.contains("model_path") && !s["model_path"].is_null()) {
            if (!s["model_path"].is_string())
                throw helix::Error(HELIX_E_VALIDATION,
                    "speculative.model_path must be a string",
                    "invalid_request_error", "speculative.model_path", "helix_e_validation");
            opts.speculative.draft_model_path = s["model_path"].get<std::string>();
        }
        if (s.contains("n_max") && !s["n_max"].is_null()) {
            if (!s["n_max"].is_number_integer())
                throw helix::Error(HELIX_E_VALIDATION,
                    "speculative.n_max must be an integer",
                    "invalid_request_error", "speculative.n_max", "helix_e_validation");
            opts.speculative.n_max = s["n_max"].get<int32_t>();
            if (opts.speculative.n_max < 0)
                throw helix::Error(HELIX_E_VALIDATION,
                    "speculative.n_max must be non-negative, got "
                    + std::to_string(opts.speculative.n_max),
                    "invalid_request_error", "speculative.n_max", "helix_e_validation");
        }
        if (s.contains("n_min") && !s["n_min"].is_null()) {
            if (!s["n_min"].is_number_integer())
                throw helix::Error(HELIX_E_VALIDATION,
                    "speculative.n_min must be an integer",
                    "invalid_request_error", "speculative.n_min", "helix_e_validation");
            opts.speculative.n_min = s["n_min"].get<int32_t>();
            if (opts.speculative.n_min < 0)
                throw helix::Error(HELIX_E_VALIDATION,
                    "speculative.n_min must be non-negative, got "
                    + std::to_string(opts.speculative.n_min),
                    "invalid_request_error", "speculative.n_min", "helix_e_validation");
        }
        if (s.contains("p_min") && !s["p_min"].is_null()) {
            if (!s["p_min"].is_number())
                throw helix::Error(HELIX_E_VALIDATION,
                    "speculative.p_min must be a number",
                    "invalid_request_error", "speculative.p_min", "helix_e_validation");
            opts.speculative.p_min = s["p_min"].get<float>();
            if (opts.speculative.p_min < 0.0f || opts.speculative.p_min > 1.0f)
                throw helix::Error(HELIX_E_VALIDATION,
                    "speculative.p_min must be in [0.0, 1.0], got "
                    + std::to_string(opts.speculative.p_min),
                    "invalid_request_error", "speculative.p_min", "helix_e_validation");
        }
        if (s.contains("backend_sampling") && !s["backend_sampling"].is_null()) {
            if (!s["backend_sampling"].is_boolean())
                throw helix::Error(HELIX_E_VALIDATION,
                    "speculative.backend_sampling must be a boolean",
                    "invalid_request_error", "speculative.backend_sampling",
                    "helix_e_validation");
            opts.speculative.backend_sampling = s["backend_sampling"].get<bool>();
        }
        if (s.contains("cache_type_k") && !s["cache_type_k"].is_null()) {
            if (!s["cache_type_k"].is_string())
                throw helix::Error(HELIX_E_VALIDATION,
                    "speculative.cache_type_k must be a string",
                    "invalid_request_error", "speculative.cache_type_k", "helix_e_validation");
            opts.speculative.cache_type_k = s["cache_type_k"].get<std::string>();
        }
        if (s.contains("cache_type_v") && !s["cache_type_v"].is_null()) {
            if (!s["cache_type_v"].is_string())
                throw helix::Error(HELIX_E_VALIDATION,
                    "speculative.cache_type_v must be a string",
                    "invalid_request_error", "speculative.cache_type_v", "helix_e_validation");
            opts.speculative.cache_type_v = s["cache_type_v"].get<std::string>();
        }
    }

    return opts;
}

} // namespace helix
