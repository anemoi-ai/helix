#include "response.hpp"
#include "base64.hpp"    /* from llama.cpp/common */
#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace helix {

static json logprob_entry_to_json(const TokenLogprobEntry& e) {
    json top = json::array();
    for (const auto& a : e.top_logprobs) {
        top.push_back({{"token", a.token}, {"logprob", a.logprob}, {"bytes", json(a.bytes)}});
    }
    return {{"token", e.token}, {"logprob", e.logprob},
            {"bytes", json(e.bytes)}, {"top_logprobs", top}};
}

static json choice_logprobs_to_json(const ChoiceLogprobs& lp) {
    json arr = json::array();
    for (const auto& e : lp.content) arr.push_back(logprob_entry_to_json(e));
    return {{"content", arr}, {"refusal", nullptr}};
}

static json tool_call_to_json(const ToolCall& tc) {
    return {
        {"id",   tc.id},
        {"type", tc.type.empty() ? "function" : tc.type},
        {"function", {
            {"name",      tc.function_name},
            {"arguments", tc.function_arguments}
        }}
    };
}

static json usage_to_json(const Usage& usage) {
    json j = {
        {"prompt_tokens",     usage.prompt_tokens},
        {"completion_tokens", usage.completion_tokens},
        {"total_tokens",      usage.total_tokens}
    };
    if (usage.completion_tokens_details &&
        usage.completion_tokens_details->reasoning_tokens > 0) {
        j["completion_tokens_details"] = {
            {"reasoning_tokens", usage.completion_tokens_details->reasoning_tokens}
        };
    }
    return j;
}

std::string ChatResponse::to_json() const {
    json choices_j = json::array();
    for (const auto& c : choices) {
        json msg = {{"role", c.message.role}};
        /* OpenAI permits assistant text alongside tool_calls (models often
         * emit commentary before calling a tool), and the streaming path
         * already delivers it — keep both response modes consistent. content
         * is null only when there is none. */
        if (!c.message.tool_calls.empty()) {
            msg["content"] = c.message.content.empty()
                                 ? json(nullptr)
                                 : json(c.message.content);
            json tcs = json::array();
            for (const auto& tc : c.message.tool_calls) tcs.push_back(tool_call_to_json(tc));
            msg["tool_calls"] = tcs;
        } else {
            msg["content"] = c.message.content;
        }
        if (c.message.reasoning_content) {
            msg["reasoning_content"] = *c.message.reasoning_content;
        }
        choices_j.push_back({
            {"index",         c.index},
            {"message",       msg},
            {"finish_reason", c.finish_reason},
            {"logprobs",      c.logprobs ? choice_logprobs_to_json(*c.logprobs) : json(nullptr)}
        });
    }

    json j = {
        {"id",                 id},
        {"object",             object},
        {"created",            created},
        {"model",              model},
        {"system_fingerprint", system_fingerprint},
        {"choices",            choices_j},
    };
    j["usage"] = usage_to_json(usage);
    if (context_shifted) {
        j["helix"] = {
            {"context_shifted", true},
            {"evicted_tokens",  evicted_tokens},
        };
    }
    return j.dump();
}

std::string ChatChunk::to_json() const {
    json choices_j = json::array();
    for (const auto& c : choices) {
        json delta = json::object();
        if (c.delta.role)    delta["role"]    = *c.delta.role;
        if (c.delta.content) delta["content"] = *c.delta.content;
        if (!c.delta.tool_calls.empty()) {
            json tcs = json::array();
            for (const auto& tc : c.delta.tool_calls) {
                json tc_j = json::object();
                tc_j["index"] = tc.index;
                if (!tc.id.empty()) {
                    /* First chunk for this tool call: include id, type, name */
                    tc_j["id"]   = tc.id;
                    tc_j["type"] = tc.type.empty() ? "function" : tc.type;
                    tc_j["function"] = {
                        {"name",      tc.function_name},
                        {"arguments", tc.function_arguments}
                    };
                } else {
                    /* Subsequent argument-delta chunk: only arguments */
                    tc_j["function"] = {{"arguments", tc.function_arguments}};
                }
                tcs.push_back(tc_j);
            }
            delta["tool_calls"] = tcs;
        }
        if (c.delta.reasoning_content) {
            delta["reasoning_content"] = *c.delta.reasoning_content;
        }

        json choice = {
            {"index",    c.index},
            {"delta",    delta},
            {"logprobs", c.logprobs ? choice_logprobs_to_json(*c.logprobs) : json(nullptr)}
        };
        if (!c.finish_reason.empty()) {
            choice["finish_reason"] = c.finish_reason;
        } else {
            choice["finish_reason"] = nullptr;
        }
        choices_j.push_back(choice);
    }

    json j = {
        {"id",                 id},
        {"object",             object},
        {"created",            created},
        {"model",              model},
        {"system_fingerprint", system_fingerprint},
        {"choices",            choices_j}
    };
    if (usage) {
        j["usage"] = usage_to_json(*usage);
    }
    return j.dump();
}

std::string EmbeddingsResponse::to_json() const {
    json data = json::array();
    for (size_t i = 0; i < vectors.size(); ++i) {
        json item = {
            {"object", "embedding"},
            {"index",  static_cast<int>(i)}
        };
        if (encoding == EmbeddingsRequest::Encoding::Base64) {
            /* Raw little-endian IEEE-754 float buffer, matching OpenAI's
             * base64 encoding of the same vector. */
            const auto& v = vectors[i];
            item["embedding"] = base64::encode(
                reinterpret_cast<const char*>(v.data()),
                v.size() * sizeof(float));
        } else {
            item["embedding"] = vectors[i];
        }
        data.push_back(std::move(item));
    }

    json j = {
        {"object", "list"},
        {"data",   data},
        {"model",  model},
        {"usage",  {
            {"prompt_tokens", prompt_tokens},
            {"total_tokens",  prompt_tokens}
        }}
    };
    return j.dump();
}

std::string make_chunk_json(const std::string& id,
                             int64_t created,
                             const std::string& model,
                             const std::string& fingerprint,
                             int choice_index,
                             const DeltaContent& delta,
                             const std::string& finish_reason,
                             const ChoiceLogprobs* logprobs) {
    ChatChunk chunk;
    chunk.id                = id;
    chunk.created           = created;
    chunk.model             = model;
    chunk.system_fingerprint = fingerprint;

    ChunkChoice c;
    c.index         = choice_index;
    c.delta         = delta;
    c.finish_reason = finish_reason;
    if (logprobs) c.logprobs = *logprobs;
    chunk.choices.push_back(c);

    return chunk.to_json();
}

std::string make_usage_chunk_json(const std::string& id,
                                   int64_t created,
                                   const std::string& model,
                                   const std::string& fingerprint,
                                   const Usage& usage) {
    ChatChunk chunk;
    chunk.id                 = id;
    chunk.created            = created;
    chunk.model              = model;
    chunk.system_fingerprint = fingerprint;
    /* choices is empty — terminal usage chunk */
    chunk.usage              = usage;
    return chunk.to_json();
}

std::string make_shift_chunk_json(const std::string& id,
                                  int64_t created,
                                  const std::string& model,
                                  const std::string& fingerprint,
                                  int evicted_tokens) {
    json j = {
        {"id",                 id},
        {"object",             "chat.completion.chunk"},
        {"created",            created},
        {"model",              model},
        {"system_fingerprint", fingerprint},
        {"choices",            json::array()},
        {"helix",              {{"context_shifted", true},
                                {"evicted_tokens",  evicted_tokens}}},
    };
    return j.dump();
}

} // namespace helix
