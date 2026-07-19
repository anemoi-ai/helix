#include "request.hpp"
#include "../internal/error.hpp"
#include "../multimodal/image_decode.hpp"
#include "../util/strings.hpp"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace helix {

static std::string get_str(const json& j, const char* key, const std::string& def = "") {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return def;
    if (!it->is_string()) throw_validation("field must be a string", key);
    return it->get<std::string>();
}

static ContentPart parse_content_part(const json& j, size_t part_idx) {
    ContentPart p;
    p.type = get_str(j, "type");

    const std::string part_path =
        "messages[].content[" + std::to_string(part_idx) + "]";

    if (p.type == "text") {
        p.text = get_str(j, "text");

    } else if (p.type == "image_url") {
        if (!j.contains("image_url") || !j["image_url"].is_object()) {
            throw_validation("image_url part missing 'image_url' object", part_path);
        }
        const auto& iu = j["image_url"];
        const std::string url = get_str(iu, "url");
        if (url.empty()) {
            throw_validation("image_url.url is empty", part_path + ".image_url.url");
        }
        if (starts_with(url, "http://") || starts_with(url, "https://")) {
            throw_validation(
                "HTTP/HTTPS image URLs are not supported — helix does not fetch "
                "external resources (pass the image as a data URI instead)",
                part_path + ".image_url.url");
        }
        p.image_detail = get_str(iu, "detail", "auto");
        if (p.image_detail != "auto" && p.image_detail != "low" &&
            p.image_detail != "high") {
            throw_validation(
                "image_url.detail must be \"auto\", \"low\", or \"high\" "
                "(got \"" + p.image_detail + "\")",
                part_path + ".image_url.detail");
        }

        if (starts_with(url, "data:image/")) {
            p.media_raw = decode_image_data_uri(url);
        } else if (starts_with(url, "file://")) {
            p.media_raw = load_file_uri(url);
        } else {
            throw_validation(
                "image_url.url must be a data URI (data:image/...) or a "
                "file URI (file:///path)",
                part_path + ".image_url.url");
        }

    } else if (p.type == "input_audio") {
        if (!j.contains("input_audio") || !j["input_audio"].is_object()) {
            throw_validation("input_audio part missing 'input_audio' object", part_path);
        }
        const auto& ia = j["input_audio"];
        p.audio_format = get_str(ia, "format");
        const std::string data = get_str(ia, "data");
        p.media_raw = decode_audio_base64(data, p.audio_format);

    } else {
        throw_validation(
            "unsupported content part type '" + p.type +
            "'; supported: text, image_url, input_audio",
            part_path + ".type");
    }
    return p;
}

static ToolCall parse_tool_call(const json& j, const char* path = "messages") {
    ToolCall tc;
    tc.id   = get_str(j, "id");
    if (tc.id.empty()) {
        throw_validation(
            std::string(path) + ".tool_calls[].id is required and must not be empty",
            std::string(path) + ".tool_calls[].id");
    }
    tc.type = get_str(j, "type", "function");
    if (j.contains("function") && j["function"].is_object()) {
        tc.function_name = get_str(j["function"], "name");
        const auto& fn = j["function"];
        if (fn.contains("arguments")) {
            if (fn["arguments"].is_object()) {
                throw_validation(
                    std::string(path) + ".tool_calls[].function.arguments must be a "
                    "JSON-encoded string, not an object — pass the arguments as a "
                    "serialised JSON string, e.g. \"{\\\"key\\\":\\\"value\\\"}\"",
                    std::string(path) + ".tool_calls[].function.arguments");
            }
            tc.function_arguments = get_str(fn, "arguments");
            if (!tc.function_arguments.empty()) {
                try {
                    (void)json::parse(tc.function_arguments);
                } catch (const json::parse_error&) {
                    throw_validation(
                        std::string(path) + ".tool_calls[].function.arguments must be "
                        "valid JSON",
                        std::string(path) + ".tool_calls[].function.arguments");
                }
            }
        }
    }
    return tc;
}

static Message parse_message(const json& j, size_t msg_index) {
    if (!j.is_object()) throw_validation("each message must be an object", "messages");
    Message m;
    m.role = get_str(j, "role");
    if (m.role.empty()) throw_validation("message missing required 'role' field", "messages");

    if (j.contains("content")) {
        const auto& c = j["content"];
        if (c.is_string()) {
            m.content_str = c.get<std::string>();
        } else if (c.is_array()) {
            /* 'tool' results are a single string per the OpenAI schema; reject
             * array form. 'assistant'/'system'/'developer'/'user' may use the
             * content-part array, but only 'user' turns may carry media — an
             * assistant turn is text the model previously produced. */
            if (m.role == "tool") {
                throw_validation("'tool' messages cannot have array content",
                                 "messages");
            }
            /* An empty content array renders to an empty message, which the
             * model reads as "no input" and answers with a hallucinated
             * completion — reject it as a malformed request rather than
             * silently degrade. Checking here (where the JSON shape is known)
             * also lets the rest of the code treat "has content parts" and
             * "content was an array" as the same condition. */
            if (c.empty()) {
                throw_validation("message content array cannot be empty",
                                 "messages[" + std::to_string(msg_index) +
                                     "].content");
            }
            size_t pidx = 0;
            for (const auto& part : c) {
                ContentPart cp = parse_content_part(part, pidx++);
                if (m.role != "user" && cp.type != "text") {
                    throw_validation(
                        "'" + m.role + "' messages may only contain text content parts",
                        "messages");
                }
                m.content_parts.push_back(std::move(cp));
            }
        } else if (!c.is_null()) {
            throw_validation("message 'content' must be a string or array", "messages");
        }
    }

    if (j.contains("tool_calls") && j["tool_calls"].is_array()) {
        for (const auto& tc : j["tool_calls"]) {
            m.tool_calls.push_back(parse_tool_call(tc, "messages"));
        }
    }

    if (j.contains("tool_call_id")) {
        m.tool_call_id = get_str(j, "tool_call_id");
    }

    /* Reasoning content from a prior assistant turn (phase 7). Carried through
     * to the chat template — reasoning templates may re-render it. */
    if (j.contains("reasoning_content") && !j["reasoning_content"].is_null()) {
        m.reasoning_content = get_str(j, "reasoning_content");
    }

    return m;
}

static FunctionDef parse_function_def(const json& j) {
    FunctionDef f;
    f.name        = get_str(j, "name");
    f.description = get_str(j, "description");
    if (j.contains("parameters")) {
        if (!j["parameters"].is_object()) {
            throw_validation("function.parameters must be a JSON object",
                             "tools[].function.parameters");
        }
        f.parameters = j["parameters"].dump();
    }
    if (j.contains("strict") && j["strict"].is_boolean()) {
        f.strict = j["strict"].get<bool>();
    }
    return f;
}

static Tool parse_tool(const json& j) {
    Tool t;
    t.type = get_str(j, "type", "function");
    if (j.contains("function") && j["function"].is_object()) {
        t.function = parse_function_def(j["function"]);
    }
    /* Checked outside the contains() guard so a tool with no 'function'
     * object at all is rejected too, not parsed into a nameless tool. */
    if (t.function.name.empty()) {
        throw_validation("tool function name must not be empty",
                         "tools[].function.name");
    }
    return t;
}

ChatRequest ChatRequest::from_json(const std::string& json_str) {
    json j;
    try {
        j = json::parse(json_str);
    } catch (const json::parse_error& e) {
        throw_invalid_json(std::string("JSON parse error: ") + e.what());
    }

    if (!j.is_object()) throw_invalid_json("request body must be a JSON object");

    ChatRequest r;

    /* Required */
    if (!j.contains("model") || j["model"].is_null())
        throw_validation("missing required field 'model'", "model");
    if (!j["model"].is_string())
        throw_validation("'model' must be a string", "model");
    r.model = j["model"].get<std::string>();

    if (!j.contains("messages") || !j["messages"].is_array())
        throw_validation("missing required field 'messages'", "messages");
    for (size_t mi = 0; mi < j["messages"].size(); ++mi) {
        r.messages.push_back(parse_message(j["messages"][mi], mi));
    }

    /* Sampling */
    auto get_float = [&](const char* key) -> std::optional<float> {
        auto it = j.find(key);
        if (it == j.end() || it->is_null()) return std::nullopt;
        if (!it->is_number()) throw_validation(std::string("'") + key + "' must be a number", key);
        return it->get<float>();
    };
    auto get_int = [&](const char* key) -> std::optional<int> {
        auto it = j.find(key);
        if (it == j.end() || it->is_null()) return std::nullopt;
        if (!it->is_number_integer()) throw_validation(std::string("'") + key + "' must be an integer", key);
        return it->get<int>();
    };

    r.temperature        = get_float("temperature");
    r.top_p              = get_float("top_p");
    r.top_k              = get_int("top_k");
    r.min_p              = get_float("min_p");
    r.repeat_penalty     = get_float("repeat_penalty");
    r.presence_penalty   = get_float("presence_penalty");
    r.frequency_penalty  = get_float("frequency_penalty");

    if (auto it = j.find("seed"); it != j.end() && !it->is_null()) {
        if (!it->is_number_integer())
            throw_validation("'seed' must be an integer", "seed");
        int64_t seed_val = it->get<int64_t>();
        if (seed_val < 0) {
            throw_validation("'seed' must be non-negative (got " +
                             std::to_string(seed_val) + ")", "seed");
        }
        if (seed_val > static_cast<int64_t>(UINT32_MAX)) {
            throw_validation("'seed' must be <= " + std::to_string(UINT32_MAX) +
                             " (got " + std::to_string(seed_val) + ")", "seed");
        }
        r.seed = static_cast<uint32_t>(seed_val);
    }

    if (j.contains("logit_bias") && j["logit_bias"].is_object()) {
        for (auto& [k, v] : j["logit_bias"].items()) {
            if (!v.is_number()) throw_validation("logit_bias values must be numbers", "logit_bias");
            r.logit_bias[k] = v.get<float>();
        }
    }

    /* Output control */
    r.max_tokens             = get_int("max_tokens");
    r.max_completion_tokens  = get_int("max_completion_tokens");
    r.reasoning_budget       = get_int("reasoning_budget");
    r.n                      = get_int("n");

    if (j.contains("stop")) {
        const auto& s = j["stop"];
        if (s.is_string()) {
            r.stop.push_back(s.get<std::string>());
        } else if (s.is_array()) {
            for (const auto& item : s) {
                if (!item.is_string()) throw_validation("'stop' array elements must be strings", "stop");
                r.stop.push_back(item.get<std::string>());
            }
        } else if (!s.is_null()) {
            throw_validation("'stop' must be a string or array of strings", "stop");
        }
    }

    if (j.contains("stream") && !j["stream"].is_null()) {
        if (!j["stream"].is_boolean()) throw_validation("'stream' must be a boolean", "stream");
        r.stream = j["stream"].get<bool>();
    }

    if (j.contains("stream_options") && j["stream_options"].is_object()) {
        StreamOptions so;
        if (j["stream_options"].contains("include_usage") &&
            j["stream_options"]["include_usage"].is_boolean()) {
            so.include_usage = j["stream_options"]["include_usage"].get<bool>();
        }
        r.stream_options = so;
    }

    /* Logprobs */
    if (j.contains("logprobs") && j["logprobs"].is_boolean()) {
        r.logprobs = j["logprobs"].get<bool>();
    }
    r.top_logprobs = get_int("top_logprobs");

    /* Tools */
    if (j.contains("tools") && j["tools"].is_array()) {
        for (const auto& t : j["tools"]) {
            r.tools.push_back(parse_tool(t));
        }
    }

    if (j.contains("tool_choice") && !j["tool_choice"].is_null()) {
        const auto& tc = j["tool_choice"];
        if (tc.is_string()) {
            const auto& s = tc.get<std::string>();
            if (s != "auto" && s != "none" && s != "required") {
                throw_validation(
                    "tool_choice must be \"auto\", \"none\", or \"required\" "
                    "(got \"" + s + "\")", "tool_choice");
            }
            r.tool_choice = s;
        } else if (tc.is_object()) {
            ToolChoiceFunction tcf;
            if (tc.contains("function") && tc["function"].is_object()) {
                tcf.name = get_str(tc["function"], "name");
            }
            r.tool_choice = tcf;
        }
    }

    if (j.contains("parallel_tool_calls") && j["parallel_tool_calls"].is_boolean()) {
        r.parallel_tool_calls = j["parallel_tool_calls"].get<bool>();
    }

    /* response_format */
    if (j.contains("response_format") && j["response_format"].is_object()) {
        const auto& rf = j["response_format"];
        ResponseFormat fmt;
        fmt.type = get_str(rf, "type", "text");
        if (fmt.type != "text" && fmt.type != "json_object" &&
            fmt.type != "json_schema") {
            throw_validation(
                "response_format.type must be \"text\", \"json_object\", or "
                "\"json_schema\" (got \"" + fmt.type + "\")",
                "response_format.type");
        }
        if (fmt.type == "json_schema" && rf.contains("json_schema")) {
            fmt.json_schema = rf["json_schema"].dump();
            if (rf["json_schema"].contains("strict") && rf["json_schema"]["strict"].is_boolean()) {
                fmt.strict = rf["json_schema"]["strict"].get<bool>();
            }
        }
        r.response_format = fmt;
    }

    /* Misc */
    if (j.contains("user") && j["user"].is_string()) {
        r.user = j["user"].get<std::string>();
    }
    if (j.contains("store") && j["store"].is_boolean()) {
        r.store = j["store"].get<bool>();
    }

    /* Deprecated v0 API: translate functions[] → tools[] and function_call → tool_choice.
     * Only applied when tools[] is not already present. */
    if (r.tools.empty() && j.contains("functions") && j["functions"].is_array()) {
        for (const auto& fn : j["functions"]) {
            if (!fn.is_object()) continue;
            Tool t;
            t.type = "function";
            t.function = parse_function_def(fn);
            /* Same rule as parse_tool: the deprecated path must not admit a
             * nameless tool either. */
            if (t.function.name.empty()) {
                throw_validation("function name must not be empty",
                                 "functions[].name");
            }
            r.tools.push_back(std::move(t));
        }
        if (!r.tools.empty() && j.contains("function_call") && !j["function_call"].is_null()) {
            const auto& fc = j["function_call"];
            if (fc.is_string()) {
                const auto& s = fc.get<std::string>();
                if (s == "none" || s == "auto") {
                    r.tool_choice = s;
                } else {
                    ToolChoiceFunction tcf;
                    tcf.name = s;
                    r.tool_choice = tcf;
                }
            } else if (fc.is_object() && fc.contains("name")) {
                ToolChoiceFunction tcf;
                tcf.name = get_str(fc, "name");
                r.tool_choice = tcf;
            }
        }
    }

    return r;
}

void ChatRequest::validate() const {
    if (model.empty())    throw_validation("'model' cannot be empty",    "model");
    if (messages.empty()) throw_validation("'messages' cannot be empty", "messages");

    for (size_t i = 0; i < messages.size(); ++i) {
        const auto& msg = messages[i];
        if (msg.role != "system" && msg.role != "developer" &&
            msg.role != "user"   && msg.role != "assistant"  &&
            msg.role != "tool") {
            throw_validation("unknown message role '" + msg.role + "'", "messages");
        }
        if (msg.role == "tool" && msg.tool_call_id.empty()) {
            throw_validation(
                "messages[" + std::to_string(i) + "] with role 'tool' requires 'tool_call_id'",
                "messages[" + std::to_string(i) + "].tool_call_id");
        }
        /* An empty text part is a no-op at render time; reject it so clients
         * get a clear error instead of a silently-dropped part. */
        for (size_t pi = 0; pi < msg.content_parts.size(); ++pi) {
            if (msg.content_parts[pi].type == "text" &&
                msg.content_parts[pi].text.empty()) {
                throw_validation(
                    "text content part cannot be empty",
                    "messages[" + std::to_string(i) + "].content[" +
                        std::to_string(pi) + "].text");
            }
        }
    }

    /* A tool_choice that pins a function must name one; an empty name would
     * otherwise fall through to "match any tool" semantics. */
    if (tool_choice && std::holds_alternative<ToolChoiceFunction>(*tool_choice)) {
        if (std::get<ToolChoiceFunction>(*tool_choice).name.empty()) {
            throw_validation("tool_choice.function.name cannot be empty",
                             "tool_choice.function.name");
        }
    }

    if (n && *n < 1) throw_validation("'n' must be >= 1", "n");
    if (n && *n > 128) throw_validation("'n' must be <= 128", "n");

    if (temperature && (*temperature < 0.0f || *temperature > 2.0f))
        throw_validation("'temperature' must be in [0, 2]", "temperature");
    if (top_p && (*top_p < 0.0f || *top_p > 1.0f))
        throw_validation("'top_p' must be in [0, 1]", "top_p");
    /* 0 is meaningful: llama.cpp treats top_k <= 0 as "keep all tokens". */
    if (top_k && *top_k < 0)
        throw_validation("'top_k' must be >= 0 (0 disables top-k)", "top_k");
    if (min_p && (*min_p < 0.0f || *min_p > 1.0f))
        throw_validation("'min_p' must be in [0, 1]", "min_p");
    if (repeat_penalty && *repeat_penalty <= 0.0f)
        throw_validation("'repeat_penalty' must be > 0 (1.0 disables it)",
                         "repeat_penalty");
    if (presence_penalty && (*presence_penalty < -2.0f || *presence_penalty > 2.0f))
        throw_validation("'presence_penalty' must be in [-2, 2]", "presence_penalty");
    if (frequency_penalty && (*frequency_penalty < -2.0f || *frequency_penalty > 2.0f))
        throw_validation("'frequency_penalty' must be in [-2, 2]", "frequency_penalty");
    if (max_tokens && *max_tokens < 1)
        throw_validation("'max_tokens' must be >= 1", "max_tokens");
    if (max_completion_tokens && *max_completion_tokens < 1)
        throw_validation("'max_completion_tokens' must be >= 1", "max_completion_tokens");
    if (reasoning_budget && *reasoning_budget < -1)
        throw_validation("'reasoning_budget' must be >= -1 "
                         "(-1 = unlimited, 0 = close the thinking block immediately)",
                         "reasoning_budget");

    /* Phase 6 conflict rules. */
    if (top_logprobs && *top_logprobs > 0 && !logprobs)
        throw_validation("'top_logprobs' requires 'logprobs: true'", "top_logprobs");
    if (top_logprobs && *top_logprobs < 0)
        throw_validation("'top_logprobs' must be >= 0", "top_logprobs");
    if (top_logprobs && *top_logprobs > 20)
        throw_validation("'top_logprobs' must be <= 20", "top_logprobs");

    for (size_t i = 0; i < stop.size(); ++i) {
        if (stop[i].empty())
            throw_validation("'stop' strings must not be empty", "stop");
    }
    if (stop.size() > 4)
        throw_validation("'stop' must contain at most 4 sequences", "stop");

    if (response_format && response_format->type == "json_object" && !tools.empty()) {
        // json_object + tools + tool_choice "auto" (or default) is rejected per OpenAI.
        bool tc_auto = !tool_choice.has_value(); // default when tools present
        if (tool_choice.has_value() && std::holds_alternative<std::string>(*tool_choice)) {
            const auto& s = std::get<std::string>(*tool_choice);
            tc_auto = (s == "auto" || s.empty());
        }
        if (tc_auto) {
            throw_validation(
                "response_format=json_object cannot be combined with tools when "
                "tool_choice is 'auto' (use tool_choice='none' to suppress tool calls)",
                "response_format");
        }
    }
}

/* The OpenAI schema accepts up to 2048 inputs per request; there is no
 * pre-existing ABI-wide size guard, so this cap is the only bound on the
 * request fan-out (HELIX-IMPL-001 §5.1). */
static constexpr size_t k_max_embedding_inputs = 2048;

EmbeddingsRequest EmbeddingsRequest::from_json(const std::string& json_str) {
    json j;
    try {
        j = json::parse(json_str);
    } catch (const json::parse_error& e) {
        throw_invalid_json(std::string("JSON parse error: ") + e.what());
    }

    if (!j.is_object()) throw_invalid_json("request body must be a JSON object");

    EmbeddingsRequest r;

    if (!j.contains("model") || j["model"].is_null())
        throw_validation("missing required field 'model'", "model");
    if (!j["model"].is_string())
        throw_validation("'model' must be a string", "model");
    r.model = j["model"].get<std::string>();

    if (!j.contains("input") || j["input"].is_null())
        throw_validation("missing required field 'input'", "input");
    const auto& in = j["input"];
    if (in.is_string()) {
        r.inputs.push_back(in.get<std::string>());
    } else if (in.is_array()) {
        if (in.empty())
            throw_validation("'input' array cannot be empty", "input");
        if (in.size() > k_max_embedding_inputs)
            throw_validation("'input' array cannot exceed " +
                             std::to_string(k_max_embedding_inputs) +
                             " elements (got " + std::to_string(in.size()) + ")",
                             "input");
        for (const auto& item : in) {
            if (item.is_number_integer() || item.is_array()) {
                throw Error(HELIX_E_UNSUPPORTED_FEATURE,
                            "pre-tokenized 'input' (integer token arrays) is "
                            "not supported; pass text strings",
                            "unsupported_feature_error", "input",
                            "helix_e_unsupported_feature");
            }
            if (!item.is_string())
                throw_validation("'input' array elements must be strings", "input");
            r.inputs.push_back(item.get<std::string>());
        }
    } else {
        throw_validation("'input' must be a string or an array of strings",
                         "input");
    }

    if (auto it = j.find("encoding_format"); it != j.end() && !it->is_null()) {
        if (!it->is_string())
            throw_validation("'encoding_format' must be a string", "encoding_format");
        const auto& s = it->get<std::string>();
        if (s == "float") {
            r.encoding = Encoding::Float;
        } else if (s == "base64") {
            r.encoding = Encoding::Base64;
        } else {
            throw_validation("'encoding_format' must be \"float\" or \"base64\" "
                             "(got \"" + s + "\")", "encoding_format");
        }
    }

    if (j.contains("dimensions") && !j["dimensions"].is_null()) {
        throw Error(HELIX_E_UNSUPPORTED_FEATURE,
                    "'dimensions' is not supported; vectors always have the "
                    "model's full output dimension (see helix_model_embedding_dim)",
                    "unsupported_feature_error", "dimensions",
                    "helix_e_unsupported_feature");
    }

    /* Unknown fields are ignored, matching ChatRequest leniency. */
    return r;
}

void EmbeddingsRequest::validate() const {
    if (model.empty()) throw_validation("'model' cannot be empty", "model");
    /* Empty strings would otherwise surface confusingly after tokenization. */
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].empty())
            throw_validation("input[" + std::to_string(i) +
                             "] must be a non-empty string", "input");
    }
}

/* Matches llama-server's rerank document cap; keeps a single request's
 * fan-out bounded like k_max_embedding_inputs does for embeddings. */
static constexpr size_t k_max_rerank_documents = 1024;

RerankRequest RerankRequest::from_json(const std::string& json_str) {
    json j;
    try {
        j = json::parse(json_str);
    } catch (const json::parse_error& e) {
        throw_invalid_json(std::string("JSON parse error: ") + e.what());
    }

    if (!j.is_object()) throw_invalid_json("request body must be a JSON object");

    RerankRequest r;

    if (!j.contains("model") || j["model"].is_null())
        throw_validation("missing required field 'model'", "model");
    if (!j["model"].is_string())
        throw_validation("'model' must be a string", "model");
    r.model = j["model"].get<std::string>();

    if (!j.contains("query") || j["query"].is_null())
        throw_validation("missing required field 'query'", "query");
    if (!j["query"].is_string())
        throw_validation("'query' must be a string", "query");
    r.query = j["query"].get<std::string>();

    if (!j.contains("documents") || j["documents"].is_null())
        throw_validation("missing required field 'documents'", "documents");
    const auto& docs = j["documents"];
    if (!docs.is_array())
        throw_validation("'documents' must be an array of strings", "documents");
    if (docs.empty())
        throw_validation("'documents' array cannot be empty", "documents");
    if (docs.size() > k_max_rerank_documents)
        throw_validation("'documents' array cannot exceed " +
                         std::to_string(k_max_rerank_documents) +
                         " elements (got " + std::to_string(docs.size()) + ")",
                         "documents");
    for (const auto& d : docs) {
        if (!d.is_string())
            throw_validation("'documents' array elements must be strings",
                             "documents");
        r.documents.push_back(d.get<std::string>());
    }

    if (auto it = j.find("top_n"); it != j.end() && !it->is_null()) {
        if (!it->is_number_integer())
            throw_validation("'top_n' must be an integer", "top_n");
        r.top_n = it->get<int>();
        if (r.top_n < 1)
            throw_validation("'top_n' must be >= 1, got " +
                             std::to_string(r.top_n), "top_n");
    }

    /* Unknown fields are ignored, matching ChatRequest leniency. */
    return r;
}

void RerankRequest::validate() const {
    if (model.empty()) throw_validation("'model' cannot be empty", "model");
    if (query.empty()) throw_validation("'query' cannot be empty", "query");
    for (size_t i = 0; i < documents.size(); ++i) {
        if (documents[i].empty())
            throw_validation("documents[" + std::to_string(i) +
                             "] must be a non-empty string", "documents");
    }
}

int ChatRequest::effective_max_tokens(int ctx_size, int prompt_tokens) const {
    int ceiling = ctx_size - prompt_tokens;
    if (ceiling <= 0) throw_context_full();
    if (max_completion_tokens) return std::min(*max_completion_tokens, ceiling);
    if (max_tokens)            return std::min(*max_tokens,            ceiling);
    return ceiling; /* unlimited — run to context */
}

} // namespace helix
