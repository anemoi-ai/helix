#pragma once
#include <string>
#include <string_view>

namespace helix {

struct ReasoningOutput {
    std::string reasoning;
    std::string content;
};

/* Opening tag of a reasoning block: the template-supplied start tag, else the
 * end tag with its '/' removed ("</think>" -> "<think>", "[/THINK]" ->
 * "[THINK]"), else "<think>". */
std::string resolve_thinking_start_tag(const std::string& start_tag,
                                       const std::string& end_tag);

/* Streaming state machine that splits model output into reasoning vs content.
 *
 * Scans for the thinking tags (default <think>...</think>; the chat template's
 * tags when given) in the token stream.  Text inside the tags is routed to
 * `reasoning`; text outside is routed to `content`. An empty tag selects the
 * default.
 *
 * Some chat templates (e.g. Qwen3 thinking mode) pre-open the think block
 * inside the rendered prompt, so generation starts mid-thought and only the
 * closing tag appears in the output. Pass start_in_think=true for those.
 *
 * Call push() for each new text fragment during generation.
 * Call flush() once generation ends to drain any buffered partial-tag text.
 *
 * Not thread-safe — one instance per decode call. */
class ReasoningExtractor {
public:
    explicit ReasoningExtractor(bool enabled = true, bool start_in_think = false,
                                std::string open_tag  = "<think>",
                                std::string close_tag = "</think>");

    ReasoningOutput push(std::string_view text);
    ReasoningOutput flush();

    bool enabled() const { return enabled_; }

private:
    bool        enabled_;
    std::string open_;
    std::string close_;
    enum class State { NORMAL, IN_THINK };
    State       state_   = State::NORMAL;
    std::string pending_;

    ReasoningOutput process();
};

} // namespace helix
