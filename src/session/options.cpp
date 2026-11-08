#include "session.hpp"
#include "../internal/error.hpp"
#include "../util/json_parse.hpp"
#include "llama.h"
#include "nlohmann/json.hpp"

namespace helix {

using nj = nlohmann::json;

namespace {

enum class OptKind { Integer, Number, Boolean, String };

/* Strict typing: a present-but-mistyped option (e.g. {"n_ctx": "4096"}) is
 * an error, matching the request parser, rather than a silently ignored
 * setting. null is treated as absent throughout. `prefix` is prepended to
 * the reported name ("speculative." for the sub-object). */
const nj* typed_opt(const nj& obj, const char* key, const std::string& prefix,
                    OptKind kind) {
    auto it = obj.find(key);
    if (it == obj.end() || it->is_null()) return nullptr;
    bool ok = false;
    const char* expected = "";
    switch (kind) {
        case OptKind::Integer: ok = it->is_number_integer(); expected = "an integer"; break;
        case OptKind::Number:  ok = it->is_number();         expected = "a number";   break;
        case OptKind::Boolean: ok = it->is_boolean();        expected = "a boolean";  break;
        case OptKind::String:  ok = it->is_string();         expected = "a string";   break;
    }
    if (!ok) {
        const std::string param = prefix + key;
        throw_validation(param + " must be " + expected, param);
    }
    return &*it;
}

/* Reads an optional non-negative integer option into `out`. */
template <typename Int>
void non_negative_int_opt(const nj& obj, const char* key, const std::string& prefix,
                          Int& out) {
    if (const nj* v = typed_opt(obj, key, prefix, OptKind::Integer)) {
        out = v->get<Int>();
        if (out < 0) {
            const std::string param = prefix + key;
            throw_validation(param + " must be non-negative, got " + std::to_string(out),
                             param);
        }
    }
}

} // namespace

SessionOptions parse_session_options(const char* session_json) {
    SessionOptions opts;
    if (!session_json || session_json[0] == '\0') return opts;

    nj j = parse_json_object<nj>(session_json, "session options",
                                 /*null_is_empty=*/true);

    auto opt = [&](const char* key, OptKind kind) { return typed_opt(j, key, "", kind); };

    non_negative_int_opt(j, "n_ctx",           "", opts.n_ctx);
    non_negative_int_opt(j, "n_batch",         "", opts.n_batch);
    non_negative_int_opt(j, "n_threads",       "", opts.n_threads);
    non_negative_int_opt(j, "n_threads_batch", "", opts.n_threads_batch);
    if (const nj* v = opt("prefix_cache", OptKind::Boolean))
        opts.prefix_cache = v->get<bool>();
    /* context_shift was documented since 1.1 but only implemented in 1.6
     * (an explicit true was rejected in 1.3–1.5). Capability validation —
     * embedding/MTP exclusion, llama_memory_can_shift — happens at session
     * creation where the model is known. */
    if (const nj* v = opt("context_shift", OptKind::Boolean))
        opts.context_shift = v->get<bool>();
    if (const nj* v = opt("swa_full", OptKind::Boolean))
        opts.swa_full = v->get<bool>();
    /* Main-context KV cache types (1.4). The value set is validated against
     * ggml at session creation (parse_cache_type), where an unknown name or
     * a quantized V without flash attention is rejected. */
    if (const nj* v = opt("cache_type_k", OptKind::String))
        opts.cache_type_k = v->get<std::string>();
    if (const nj* v = opt("cache_type_v", OptKind::String))
        opts.cache_type_v = v->get<std::string>();
    if (const nj* v = opt("warmup", OptKind::Boolean))
        opts.warmup = v->get<bool>();
    if (const nj* v = opt("seed", OptKind::Integer)) {
        int64_t seed_val = v->get<int64_t>();
        if (seed_val < 0) {
            throw_validation("seed must be non-negative, got " + std::to_string(seed_val),
                             "seed");
        }
        if (seed_val > static_cast<int64_t>(UINT32_MAX)) {
            throw_validation("seed must be <= " + std::to_string(UINT32_MAX) +
                             ", got " + std::to_string(seed_val), "seed");
        }
        opts.seed = static_cast<unsigned int>(seed_val);
    }
    non_negative_int_opt(j, "stream_coalesce_ms", "", opts.stream_coalesce_ms);
    if (const nj* v = opt("embedding", OptKind::Boolean))
        opts.embedding = v->get<bool>();
    if (const nj* v = opt("pooling", OptKind::String)) {
        const std::string p = v->get<std::string>();
        if      (p == "none") opts.pooling = LLAMA_POOLING_TYPE_NONE;
        else if (p == "mean") opts.pooling = LLAMA_POOLING_TYPE_MEAN;
        else if (p == "cls")  opts.pooling = LLAMA_POOLING_TYPE_CLS;
        else if (p == "last") opts.pooling = LLAMA_POOLING_TYPE_LAST;
        else if (p == "rank") opts.pooling = LLAMA_POOLING_TYPE_RANK;
        else
            throw_validation("pooling must be \"none\", \"mean\", \"cls\", \"last\", or "
                             "\"rank\" (got \"" + p + "\")", "pooling");
    }

    /* LoRA adapter activation (1.6, F6). Names are resolved against the
     * model's loaded adapters at session creation, where the model is known.
     * An explicit empty array means "no adapters active" — distinct from an
     * absent option (all adapters at load-time scales). */
    if (j.contains("lora") && !j["lora"].is_null()) {
        if (!j["lora"].is_array())
            throw_validation("lora must be an array of objects", "lora");
        std::vector<SessionLoraOptions> entries;
        for (size_t i = 0; i < j["lora"].size(); ++i) {
            const std::string param = "lora[" + std::to_string(i) + "]";
            const nj& e = j["lora"][i];
            if (!e.is_object())
                throw_validation(param + " must be an object", "lora");
            SessionLoraOptions sl;
            if (!e.contains("name") || !e["name"].is_string() ||
                e["name"].get<std::string>().empty())
                throw_validation(param + " requires a non-empty string 'name'", "lora");
            sl.name = e["name"].get<std::string>();
            if (e.contains("scale") && !e["scale"].is_null()) {
                if (!e["scale"].is_number())
                    throw_validation(param + ".scale must be a number", "lora");
                sl.scale = e["scale"].get<float>();
            }
            for (const auto& prev : entries) {
                if (prev.name == sl.name)
                    throw_validation("duplicate lora adapter name \"" + sl.name + "\"",
                                     "lora");
            }
            entries.push_back(std::move(sl));
        }
        opts.lora = std::move(entries);
    }

    if (j.contains("speculative") && !j["speculative"].is_null()) {
        if (!j["speculative"].is_object())
            throw_validation("speculative must be an object", "speculative");
        const nj& s = j["speculative"];
        const std::string sp = "speculative.";
        auto spec_opt = [&](const char* key, OptKind kind) {
            return typed_opt(s, key, sp, kind);
        };
        if (const nj* v = spec_opt("type", OptKind::String)) {
            const std::string t = v->get<std::string>();
            if      (t == "none")      opts.speculative.type = SpeculativeType::None;
            else if (t == "draft-mtp") opts.speculative.type = SpeculativeType::DraftMtp;
            else
                throw_validation("speculative.type must be \"none\" or \"draft-mtp\" (got \""
                                 + t + "\")", "speculative.type");
        }
        if (const nj* v = spec_opt("model_path", OptKind::String))
            opts.speculative.draft_model_path = v->get<std::string>();
        non_negative_int_opt(s, "n_max", sp, opts.speculative.n_max);
        non_negative_int_opt(s, "n_min", sp, opts.speculative.n_min);
        if (const nj* v = spec_opt("p_min", OptKind::Number)) {
            opts.speculative.p_min = v->get<float>();
            if (opts.speculative.p_min < 0.0f || opts.speculative.p_min > 1.0f)
                throw_validation("speculative.p_min must be in [0.0, 1.0], got "
                                 + std::to_string(opts.speculative.p_min),
                                 "speculative.p_min");
        }
        if (const nj* v = spec_opt("backend_sampling", OptKind::Boolean))
            opts.speculative.backend_sampling = v->get<bool>();
        if (const nj* v = spec_opt("cache_type_k", OptKind::String))
            opts.speculative.cache_type_k = v->get<std::string>();
        if (const nj* v = spec_opt("cache_type_v", OptKind::String))
            opts.speculative.cache_type_v = v->get<std::string>();
    }

    return opts;
}

} // namespace helix
