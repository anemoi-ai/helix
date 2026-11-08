#pragma once
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include "../json/request.hpp"

struct common_chat_templates;
class common_peg_arena;

namespace helix {

/* Opaque grammar trigger — mirrors common_grammar_trigger without pulling in llama.h. */
struct GrammarTrigger {
    /* Mirrors common_grammar_trigger_type: Token=0, Word=1, Pattern=2, PatternFull=3 */
    int         type  = 1; /* Word by default */
    std::string value;
    int32_t     token = -1; /* LLAMA_TOKEN_NULL for non-token triggers */
};

struct RenderResult {
    std::string              prompt;
    std::vector<std::string> stop_strings; /* from chat template + request */

    /* Phase 3: grammar / autoparser fields (empty when no tools) */
    std::string              grammar;               /* GBNF string */
    bool                     grammar_lazy  = false;
    std::vector<GrammarTrigger> grammar_triggers;
    /* The autoparser's arena. llama.cpp b12033 dropped its save/load: the
     * arena is a plain copyable value, so it is shared rather than
     * serialised and reparsed per request. Null when there is no parser. */
    std::shared_ptr<const common_peg_arena> parser;
    std::string              generation_prompt;     /* for common_chat_parser_params */
    int                      chat_format   = 0;     /* common_chat_format enum value; 0=CONTENT_ONLY */
    bool                     has_tools     = false; /* true when tool-call parsing is active */
    bool                     supports_thinking  = false; /* model may emit <think>...</think> */
    std::string              thinking_start_tag;         /* e.g. "<think>" */
    std::string              thinking_end_tag;           /* e.g. "</think>"; the first of thinking_end_tags */
    /* Every end tag the template accepts (llama.cpp b12033 made this a list).
     * The reasoning-budget sampler watches for all of them; the forced-open
     * check and helix's own extractor use the first. */
    std::vector<std::string> thinking_end_tags;
    bool                     extract_reasoning  = false; /* reasoning_format != "none" */
    bool                     thinking_forced_open = false; /* prompt ends with an open think tag */
};

/* Convert helix::Message vector + tools to a rendered prompt string using the
 * model's Jinja chat template.  Throws helix::Error on unsupported content. */
RenderResult render_prompt(const std::vector<Message>& messages,
                           const std::vector<std::string>& request_stop,
                           const common_chat_templates* tmpls,
                           const std::vector<Tool>& tools = {},
                           const std::optional<ToolChoice>& tool_choice = {},
                           bool parallel_tool_calls = true,
                           bool enable_thinking = true,
                           const std::string& reasoning_format = "auto");

} // namespace helix
