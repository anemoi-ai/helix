#include "reasoning.hpp"

namespace helix {

/* Move a split position left until it no longer lands inside a multi-byte
 * UTF-8 sequence. The lookahead split below is a fixed byte offset, so it
 * can cut a character in half; emitting that would hand the sink invalid
 * UTF-8 (which the JSON serialiser rejects). Tags are ASCII, so tag-boundary
 * splits never need this. */
static size_t utf8_safe_cut(const std::string& s, size_t pos, size_t floor) {
    while (pos > floor && (static_cast<unsigned char>(s[pos]) & 0xC0) == 0x80) {
        --pos;
    }
    return pos;
}

ReasoningExtractor::ReasoningExtractor(bool enabled, bool start_in_think)
    : enabled_(enabled) {
    if (enabled && start_in_think) {
        state_ = State::IN_THINK;
    }
}

/* Split pending_ at the current state's tag boundary.
 * Returns whatever can be safely emitted and leaves only the uncertain
 * suffix (partial tag) in pending_. */
ReasoningOutput ReasoningExtractor::process() {
    ReasoningOutput out;
    if (!enabled_) {
        out.content = std::move(pending_);
        pending_.clear();
        return out;
    }

    /* Scan with a consumed-prefix offset and erase once at the end, instead
     * of substr()+erase() per match — this runs per pushed token, so the
     * allocation churn adds up on long generations. */
    size_t start = 0;
    for (;;) {
        if (state_ == State::NORMAL) {
            /* Look for <think> in the unconsumed tail. */
            auto pos = pending_.find(kOpen, start);
            if (pos != std::string::npos) {
                /* Emit everything before the tag as content, consume the tag,
                 * switch to IN_THINK, and continue scanning. */
                out.content.append(pending_, start, pos - start);
                start  = pos + kOpen.size();
                state_ = State::IN_THINK;
            } else {
                /* Tag not found; emit safe prefix (keep last 6 bytes as
                 * lookahead, backed off to a UTF-8 character boundary). */
                const size_t keep = kOpen.size() - 1;
                if (pending_.size() - start > keep) {
                    size_t cut = utf8_safe_cut(pending_,
                                               pending_.size() - keep, start);
                    out.content.append(pending_, start, cut - start);
                    start = cut;
                }
                break;
            }
        } else { /* IN_THINK */
            /* Look for </think> in the unconsumed tail. */
            auto pos = pending_.find(kClose, start);
            if (pos != std::string::npos) {
                out.reasoning.append(pending_, start, pos - start);
                start  = pos + kClose.size();
                state_ = State::NORMAL;
            } else {
                /* Tag not found; emit safe prefix (keep last 7 bytes as
                 * lookahead, backed off to a UTF-8 character boundary). */
                const size_t keep = kClose.size() - 1;
                if (pending_.size() - start > keep) {
                    size_t cut = utf8_safe_cut(pending_,
                                               pending_.size() - keep, start);
                    out.reasoning.append(pending_, start, cut - start);
                    start = cut;
                }
                break;
            }
        }
    }
    pending_.erase(0, start);
    return out;
}

ReasoningOutput ReasoningExtractor::push(std::string_view text) {
    pending_.append(text);
    return process();
}

// flush() drains any remaining buffered text and returns it. Calling flush()
// more than once is safe but the second call always returns empty output —
// call it exactly once at end of generation.
ReasoningOutput ReasoningExtractor::flush() {
    /* Drain any buffered partial-tag text at end of generation. */
    ReasoningOutput out;
    if (state_ == State::IN_THINK) {
        /* Unclosed <think> block: emit remainder as reasoning. */
        out.reasoning = std::move(pending_);
    } else {
        out.content = std::move(pending_);
    }
    pending_.clear();
    return out;
}

} // namespace helix
