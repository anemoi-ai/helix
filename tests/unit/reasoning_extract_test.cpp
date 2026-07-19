#include <gtest/gtest.h>
#include "src/chat/reasoning.hpp"

using namespace helix;

/* Helper: push all text at once and return the result. */
static ReasoningOutput push_all(ReasoningExtractor& ex, const std::string& text) {
    return ex.push(text);
}

/* ---- disabled extractor ---- */

TEST(ReasoningExtract, DisabledPassesThrough) {
    ReasoningExtractor ex(false);
    auto out = ex.push("<think>this should not be extracted</think>content");
    EXPECT_EQ(out.content, "<think>this should not be extracted</think>content");
    EXPECT_TRUE(out.reasoning.empty());
}

/* ---- no think tags ---- */

TEST(ReasoningExtract, PlainContentNoTags) {
    ReasoningExtractor ex;
    auto out = ex.push("Hello, world!");
    auto flush = ex.flush();
    EXPECT_EQ(out.content + flush.content, "Hello, world!");
    EXPECT_TRUE(out.reasoning.empty());
    EXPECT_TRUE(flush.reasoning.empty());
}

/* ---- full think block in one push ---- */

TEST(ReasoningExtract, FullThinkBlock) {
    ReasoningExtractor ex;
    auto out = ex.push("<think>my reasoning</think>answer");
    auto flush = ex.flush();
    EXPECT_EQ(out.reasoning, "my reasoning");
    EXPECT_EQ(out.content + flush.content, "answer");
}

/* ---- think block split across multiple pushes ---- */

TEST(ReasoningExtract, ThinkBlockSplit) {
    ReasoningExtractor ex;

    /* Push the open tag spread across two calls. */
    auto o1 = ex.push("<thi");
    auto o2 = ex.push("nk>reasoning here</think>content");
    auto flush = ex.flush();

    std::string all_reasoning = o1.reasoning + o2.reasoning + flush.reasoning;
    std::string all_content   = o1.content   + o2.content   + flush.content;

    EXPECT_EQ(all_reasoning, "reasoning here");
    EXPECT_EQ(all_content, "content");
}

/* ---- close tag split across pushes ---- */

TEST(ReasoningExtract, CloseTagSplit) {
    ReasoningExtractor ex;

    auto o1 = ex.push("<think>think</");
    auto o2 = ex.push("think>final");
    auto flush = ex.flush();

    std::string all_reasoning = o1.reasoning + o2.reasoning + flush.reasoning;
    std::string all_content   = o1.content   + o2.content   + flush.content;

    EXPECT_EQ(all_reasoning, "think");
    EXPECT_EQ(all_content, "final");
}

/* ---- unclosed think block (generation ended before </think>) ---- */

TEST(ReasoningExtract, UnclosedThinkBlock) {
    ReasoningExtractor ex;
    auto out = ex.push("<think>dangling");
    auto flush = ex.flush();

    std::string all_reasoning = out.reasoning + flush.reasoning;
    EXPECT_EQ(all_reasoning, "dangling");
    EXPECT_TRUE(out.content.empty());
    EXPECT_TRUE(flush.content.empty());
}

/* ---- content before and after think block ---- */

TEST(ReasoningExtract, ContentBeforeAndAfter) {
    ReasoningExtractor ex;
    auto out = ex.push("preamble<think>inner</think>epilogue");
    auto flush = ex.flush();

    std::string all_content   = out.content   + flush.content;
    std::string all_reasoning = out.reasoning + flush.reasoning;

    EXPECT_EQ(all_content, "preambleepilogue");
    EXPECT_EQ(all_reasoning, "inner");
}

/* ---- '<' that is NOT the start of a think tag ---- */

TEST(ReasoningExtract, AngledBracketInContent) {
    ReasoningExtractor ex;
    /* "<b>bold</b>" has no <think> — should all be content. */
    auto out = ex.push("<b>bold</b>");
    auto flush = ex.flush();

    std::string all_content = out.content + flush.content;
    EXPECT_EQ(all_content, "<b>bold</b>");
    EXPECT_TRUE(out.reasoning.empty());
    EXPECT_TRUE(flush.reasoning.empty());
}

/* ---- multiple think blocks (unusual but should work) ---- */

TEST(ReasoningExtract, MultipleThinkBlocks) {
    ReasoningExtractor ex;
    auto out = ex.push("<think>r1</think>c1<think>r2</think>c2");
    auto flush = ex.flush();

    std::string all_reasoning = out.reasoning + flush.reasoning;
    std::string all_content   = out.content   + flush.content;

    EXPECT_EQ(all_reasoning, "r1r2");
    EXPECT_EQ(all_content, "c1c2");
}

/* ---- forced-open think block (template pre-opened <think> in the prompt) ---- */

TEST(ReasoningExtract, ForcedOpenClosingTagOnly) {
    ReasoningExtractor ex(true, /*start_in_think=*/true);
    auto out = ex.push("step one\nstep two</think>the answer");
    auto flush = ex.flush();

    std::string all_reasoning = out.reasoning + flush.reasoning;
    std::string all_content   = out.content   + flush.content;

    EXPECT_EQ(all_reasoning, "step one\nstep two");
    EXPECT_EQ(all_content, "the answer");
}

TEST(ReasoningExtract, ForcedOpenNeverClosed) {
    ReasoningExtractor ex(true, /*start_in_think=*/true);
    auto out = ex.push("endless pondering");
    auto flush = ex.flush();

    std::string all_reasoning = out.reasoning + flush.reasoning;
    EXPECT_EQ(all_reasoning, "endless pondering");
    EXPECT_TRUE(out.content.empty());
    EXPECT_TRUE(flush.content.empty());
}

TEST(ReasoningExtract, ForcedOpenIgnoredWhenDisabled) {
    ReasoningExtractor ex(false, /*start_in_think=*/true);
    auto out = ex.push("raw</think>text");
    EXPECT_EQ(out.content, "raw</think>text");
    EXPECT_TRUE(out.reasoning.empty());
}

/* ---- multi-byte UTF-8 must never be split by the lookahead cut ---- */

static bool is_valid_utf8(const std::string& s) {
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = s[i];
        size_t len = (c < 0x80) ? 1 : (c >> 5) == 0x6 ? 2
                   : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
        if (len == 0 || i + len > s.size()) return false;
        for (size_t j = 1; j < len; ++j) {
            if ((static_cast<unsigned char>(s[i + j]) & 0xC0) != 0x80) return false;
        }
        i += len;
    }
    return true;
}

TEST(ReasoningExtract, Utf8NeverSplitAcrossEmits) {
    /* "17 × 23 ≈ ２３９" mixes 1-, 2- and 3-byte characters. Feed it byte by
     * byte so the lookahead cut lands at every possible offset. */
    const std::string input = u8"think 17 × 23 ≈ ３９１ done</think>answer: ３９１ × ok";
    ReasoningExtractor ex(true, /*start_in_think=*/true);

    std::string all_reasoning, all_content;
    for (char c : input) {
        auto out = ex.push(std::string_view(&c, 1));
        EXPECT_TRUE(is_valid_utf8(out.reasoning)) << "invalid reasoning fragment";
        EXPECT_TRUE(is_valid_utf8(out.content))   << "invalid content fragment";
        all_reasoning += out.reasoning;
        all_content   += out.content;
    }
    auto flush = ex.flush();
    EXPECT_TRUE(is_valid_utf8(flush.reasoning));
    EXPECT_TRUE(is_valid_utf8(flush.content));
    all_reasoning += flush.reasoning;
    all_content   += flush.content;

    EXPECT_EQ(all_reasoning, u8"think 17 × 23 ≈ ３９１ done");
    EXPECT_EQ(all_content, u8"answer: ３９１ × ok");
}

/* ---- token-by-token feeding (simulates streaming) ---- */

TEST(ReasoningExtract, TokenByToken) {
    ReasoningExtractor ex;

    const std::string input = "<think>think step</think>answer text";
    std::string all_reasoning, all_content;

    /* Feed one character at a time. */
    for (char c : input) {
        auto out = ex.push(std::string_view(&c, 1));
        all_reasoning += out.reasoning;
        all_content   += out.content;
    }
    auto flush = ex.flush();
    all_reasoning += flush.reasoning;
    all_content   += flush.content;

    EXPECT_EQ(all_reasoning, "think step");
    EXPECT_EQ(all_content, "answer text");
}
