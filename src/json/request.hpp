#pragma once
#include <optional>
#include <string>
#include <vector>
#include <variant>
#include <map>
#include <cstdint>

namespace helix {

/* Mirrors the OpenAI message content-part array element.
 * Uses a flat struct rather than a variant for simplicity — only one
 * of text, image, or audio fields is populated per part depending on type. */
struct ContentPart {
    std::string type;  /* "text" | "image_url" | "input_audio" */
    std::string text;  /* for type=="text" */

    /* for type=="image_url" */
    std::string image_detail;    /* "low" | "high" | "auto" (default "auto") */
    /* for type=="input_audio" */
    std::string audio_format;    /* "mp3" | "wav" */

    /* raw media bytes:
     *   image_url:   decoded PNG/JPEG/WebP bytes (from data URI or file URI)
     *   input_audio: decoded WAV/MP3 bytes (from base64 audio.data field)
     * Empty for text parts. */
    std::vector<uint8_t> media_raw;
};

struct ToolCall {
    int         index = -1;         /* streaming: position in tool_calls array */
    std::string id;
    std::string type; /* always "function" */
    std::string function_name;
    std::string function_arguments; /* raw JSON string */
};

struct Message {
    std::string role;   /* system | developer | user | assistant | tool */
    /* content is either a plain string or an array of content parts;
     * parse rejects empty arrays, so a non-empty content_parts is
     * exactly "content was an array" */
    std::string content_str;
    std::vector<ContentPart> content_parts;
    /* assistant turn fields */
    std::vector<ToolCall> tool_calls;
    /* tool turn fields */
    std::string tool_call_id;
    /* reasoning content (phase 7) — parsed from assistant history and passed
     * through to the chat template */
    std::optional<std::string> reasoning_content;
};

struct FunctionDef {
    std::string name;
    std::string description;
    std::string parameters; /* raw JSON string */
    bool strict = false;
};

struct Tool {
    std::string type; /* "function" */
    FunctionDef function;
};

struct ToolChoiceFunction {
    std::string name;
};

/* tool_choice: "auto" | "none" | "required" | {type:"function", function:{name:...}} */
using ToolChoice = std::variant<std::string, ToolChoiceFunction>;

struct ResponseFormat {
    std::string type; /* "text" | "json_object" | "json_schema" */
    std::string json_schema; /* raw JSON string when type=="json_schema" */
    bool strict = false;
};

struct StreamOptions {
    bool include_usage = false;
};

/* Full OpenAI Chat Completions request.
 * Fields are parsed permissively; validate() enforces semantics. */
struct ChatRequest {
    /* Required */
    std::string model;
    std::vector<Message> messages;

    /* Sampling */
    std::optional<float> temperature;        /* default 1.0 */
    std::optional<float> top_p;              /* default 1.0 */
    std::optional<int>   top_k;              /* Helix extension; default 40 */
    std::optional<float> min_p;              /* Helix extension; default 0.05 */
    std::optional<float> repeat_penalty;     /* Helix extension; default 1.0 (disabled, OpenAI-neutral) */
    std::optional<float> presence_penalty;   /* default 0.0 */
    std::optional<float> frequency_penalty;  /* default 0.0 */
    std::optional<uint32_t> seed;
    std::map<std::string, float> logit_bias; /* token_id → bias */

    /* Output control */
    std::optional<int> max_tokens;
    std::optional<int> max_completion_tokens; /* takes precedence over max_tokens */
    std::optional<int> reasoning_budget;      /* Helix extension: cap <think> tokens, then force the
                                               * end tag. -1/unset = unlimited; 0 = close the think
                                               * block immediately; < -1 rejected by validate(). */
    std::optional<int> n;                    /* default 1 */
    std::vector<std::string> stop;
    bool stream = false;
    std::optional<StreamOptions> stream_options;

    /* Logprobs (phase 6) — carried but ignored in phase 1 */
    bool logprobs = false;
    std::optional<int> top_logprobs;

    /* Tools (phase 3) — carried but rejected in phase 1 */
    std::vector<Tool> tools;
    std::optional<ToolChoice> tool_choice;
    bool parallel_tool_calls = true;

    /* Structured output (phase 6) */
    std::optional<ResponseFormat> response_format;

    /* Misc OpenAI fields — carried but ignored */
    std::optional<std::string> user;
    bool store = false;

    /* Effective max tokens to generate (resolved from max_completion_tokens or max_tokens). */
    int effective_max_tokens(int ctx_size, int prompt_tokens) const;

    /* Parse from JSON string. Throws helix::Error on parse failure. */
    static ChatRequest from_json(const std::string& json_str);

    /* Semantic validation. Throws helix::Error with OpenAI-shaped errors. */
    void validate() const;
};

/* Rerank request (Cohere/Jina-style /v1/rerank, as served by llama-server).
 * Fields are parsed permissively; validate() enforces semantics. */
struct RerankRequest {
    /* Required */
    std::string model;
    std::string query;
    std::vector<std::string> documents; /* 1..1024 non-empty strings */

    /* Optional; 0 = unset (return all documents). */
    int top_n = 0;

    /* Parse from JSON string. Throws helix::Error on parse failure. */
    static RerankRequest from_json(const std::string& json_str);

    /* Semantic validation. Throws helix::Error with OpenAI-shaped errors. */
    void validate() const;
};

/* OpenAI Embeddings request.
 * Fields are parsed permissively; validate() enforces semantics. */
struct EmbeddingsRequest {
    enum class Encoding { Float, Base64 };

    /* Required */
    std::string model;
    /* "input" normalised: a single string becomes a 1-element vector.
     * Pre-tokenized integer arrays and "dimensions" are rejected at parse. */
    std::vector<std::string> inputs;

    Encoding encoding = Encoding::Float; /* "encoding_format", default "float" */

    /* Parse from JSON string. Throws helix::Error on parse failure. */
    static EmbeddingsRequest from_json(const std::string& json_str);

    /* Semantic validation. Throws helix::Error with OpenAI-shaped errors. */
    void validate() const;
};

} // namespace helix
