#pragma once
#include <string>
#include <string_view>

namespace helix {

struct ReasoningOutput {
    std::string reasoning;
    std::string content;
};

/* Streaming state machine that splits model output into reasoning vs content.
 *
 * Scans for <think>...</think> tags in the token stream.  Text inside the
 * tags is routed to `reasoning`; text outside is routed to `content`.
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
    explicit ReasoningExtractor(bool enabled = true, bool start_in_think = false);

    ReasoningOutput push(std::string_view text);
    ReasoningOutput flush();

    bool enabled() const { return enabled_; }

private:
    static constexpr std::string_view kOpen  = "<think>";   /* 7 chars */
    static constexpr std::string_view kClose = "</think>";  /* 8 chars */

    bool        enabled_;
    enum class State { NORMAL, IN_THINK };
    State       state_   = State::NORMAL;
    std::string pending_;

    ReasoningOutput process();
};

} // namespace helix
