#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include "request.hpp" /* for ToolCall */

namespace helix {

// Phase 6 — per-token log-probability data (OpenAI logprobs shape).
struct TokenLogprobAlt {
    std::string          token;
    float                logprob = 0.0f;
    std::vector<uint8_t> bytes;
};

struct TokenLogprobEntry {
    std::string                  token;
    float                        logprob = 0.0f;
    std::vector<uint8_t>         bytes;
    std::vector<TokenLogprobAlt> top_logprobs;
};

struct ChoiceLogprobs {
    std::vector<TokenLogprobEntry> content;
};

struct CompletionTokensDetails {
    int reasoning_tokens = 0;
};

struct Usage {
    int prompt_tokens     = 0;
    int completion_tokens = 0;
    int total_tokens      = 0;
    std::optional<CompletionTokensDetails> completion_tokens_details;
};

struct ResponseMessage {
    std::string role = "assistant";
    std::string content;
    std::vector<ToolCall> tool_calls;
    std::optional<std::string> reasoning_content; /* phase 7 */
};

struct Choice {
    int            index        = 0;
    ResponseMessage message;
    std::string    finish_reason; /* "stop" | "length" | "tool_calls" | "content_filter" */
    std::optional<ChoiceLogprobs> logprobs; /* phase 6 */
};

struct ChatResponse {
    std::string id;       /* "chatcmpl-helix-..." */
    std::string object = "chat.completion";
    int64_t     created  = 0;
    std::string model;
    std::string system_fingerprint;
    std::vector<Choice> choices;
    Usage usage;

    /* Helix extension (1.6): present only when context_shift evicted
     * history during this request — serialised as
     * {"helix": {"context_shifted": true, "evicted_tokens": N}}. */
    bool context_shifted = false;
    int  evicted_tokens  = 0;

    /* Serialise to JSON string (OpenAI-shape). */
    std::string to_json() const;
};

/* A single streaming chunk. */
struct DeltaContent {
    std::optional<std::string> role;
    std::optional<std::string> content;
    std::vector<ToolCall> tool_calls;
    std::optional<std::string> reasoning_content;
};

struct ChunkChoice {
    int          index = 0;
    DeltaContent delta;
    std::string  finish_reason; /* empty until final chunk; see make_chunk_json() default */
    std::optional<ChoiceLogprobs> logprobs; /* phase 6 */
};

struct ChatChunk {
    std::string id;
    std::string object = "chat.completion.chunk";
    int64_t     created = 0;
    std::string model;
    std::string system_fingerprint;
    std::vector<ChunkChoice> choices;
    std::optional<Usage> usage; /* only in terminal usage chunk */

    std::string to_json() const;
};

/* OpenAI Embeddings response ("object": "list"). data[] order always
 * matches input order; usage.total_tokens == usage.prompt_tokens (there
 * are no completion tokens on this endpoint). */
struct EmbeddingsResponse {
    std::string model;                       /* session alias */
    std::vector<std::vector<float>> vectors; /* [input index] → embedding */
    int32_t prompt_tokens = 0;
    EmbeddingsRequest::Encoding encoding = EmbeddingsRequest::Encoding::Float;

    /* Serialise to JSON string (OpenAI-shape). */
    std::string to_json() const;
};

/* Build helpers */
std::string make_chunk_json(const std::string& id,
                             int64_t created,
                             const std::string& model,
                             const std::string& fingerprint,
                             int choice_index,
                             const DeltaContent& delta,
                             const std::string& finish_reason = "",
                             const ChoiceLogprobs* logprobs = nullptr);

std::string make_usage_chunk_json(const std::string& id,
                                   int64_t created,
                                   const std::string& model,
                                   const std::string& fingerprint,
                                   const Usage& usage);

/* Extension chunk carrying the context-shift report (1.6): empty choices[]
 * (like the usage chunk) plus {"helix": {"context_shifted": true,
 * "evicted_tokens": N}}. Emitted before the usage chunk / [DONE]. */
std::string make_shift_chunk_json(const std::string& id,
                                  int64_t created,
                                  const std::string& model,
                                  const std::string& fingerprint,
                                  int evicted_tokens);

} // namespace helix
