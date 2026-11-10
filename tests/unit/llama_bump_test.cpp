/* Behaviours the llama.cpp b12033 bump touches, pinned (docs/llama-cpp-bump-plan-b12211.md §6).
 *
 * Each test names what it holds in place. Several hold a fix that used to be
 * a helix fork patch and is now supplied by upstream, so that a future
 * upstream regression shows up here rather than in a user's output. */

#include <gtest/gtest.h>

#include "model/load_mode.hpp"

#include "chat.h"
#include "llama.h"
#include "reasoning-budget.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef HELIX_LLAMACPP_TEMPLATES
#error "HELIX_LLAMACPP_TEMPLATES must name llama.cpp's models/templates directory"
#endif

namespace {

/* ---------------------------------------------------------------- load_mode */

/* helix's use_mmap/use_mlock booleans are public JSON; llama.cpp replaced its
 * own pair with one enum. All four combinations must map exactly, and none to
 * AUTO (which would change load behaviour under an unchanged request). */
TEST(LlamaBump, LoadModeMapsAllFourCombinations) {
    EXPECT_EQ(helix::load_mode_from(true,  true),  LLAMA_LOAD_MODE_MMAP_MLOCK);
    EXPECT_EQ(helix::load_mode_from(true,  false), LLAMA_LOAD_MODE_MMAP);
    EXPECT_EQ(helix::load_mode_from(false, true),  LLAMA_LOAD_MODE_MLOCK);
    EXPECT_EQ(helix::load_mode_from(false, false), LLAMA_LOAD_MODE_NONE);
}

/* The helix defaults (use_mmap=true, use_mlock=false) load as llama.cpp's old
 * defaults did — not as its new default, AUTO. */
TEST(LlamaBump, LoadModeDefaultIsNotAuto) {
    EXPECT_NE(helix::load_mode_from(true, false), LLAMA_LOAD_MODE_AUTO);
    EXPECT_EQ(llama_model_default_params().load_mode, LLAMA_LOAD_MODE_AUTO)
        << "upstream's default changed; revisit why helix sets load_mode explicitly";
}

/* --------------------------------------------------------- reasoning budget */

using common_tokens = std::vector<llama_token>;

common_reasoning_budget_state feed(llama_sampler* s, const common_tokens& toks) {
    for (llama_token t : toks) llama_sampler_accept(s, t);
    return common_reasoning_budget_get_state(s);
}

/* What helix-1.4 fork patch ff343dab6 fixed, now supplied by upstream's
 * Aho-Corasick matcher (#25544): an end sequence whose prefix repeats
 * ([A A B]) must still be found when the stream overlaps it ([A A A B]).
 * A matcher that resets on mismatch without backtracking misses it, and the
 * budget then forces an end the model had already written. */
TEST(LlamaBump, ReasoningBudgetFindsEndSequenceWithRepeatedPrefix) {
    const llama_token A = 11, B = 12;
    llama_sampler* s = common_reasoning_budget_init(
        nullptr, /*start*/ {}, /*end*/ {{A, A, B}}, /*forced*/ {B},
        /*budget*/ 100, REASONING_BUDGET_COUNTING);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(feed(s, {A, A, A, B}), REASONING_BUDGET_DONE);
    llama_sampler_free(s);
}

TEST(LlamaBump, ReasoningBudgetIgnoresAPartialEndSequence) {
    const llama_token A = 11, B = 12, X = 13;
    llama_sampler* s = common_reasoning_budget_init(
        nullptr, {}, {{A, A, B}}, {B}, 100, REASONING_BUDGET_COUNTING);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(feed(s, {A, A, X, B}), REASONING_BUDGET_COUNTING);
    llama_sampler_free(s);
}

/* The multi-sequence path (thinking_end_tags): any one of several end tags
 * ends the block naturally. */
TEST(LlamaBump, ReasoningBudgetEndsOnAnyOfSeveralEndSequences) {
    const llama_token E1a = 21, E1b = 22, E2 = 23, X = 24;
    for (const common_tokens& closing : {common_tokens{E1a, E1b}, common_tokens{E2}}) {
        llama_sampler* s = common_reasoning_budget_init(
            nullptr, {}, {{E1a, E1b}, {E2}}, {E1a, E1b}, 100, REASONING_BUDGET_COUNTING);
        ASSERT_NE(s, nullptr);
        common_tokens stream{X, X};
        stream.insert(stream.end(), closing.begin(), closing.end());
        EXPECT_EQ(feed(s, stream), REASONING_BUDGET_DONE);
        llama_sampler_free(s);
    }
}

/* ------------------------------------------------------------- autoparser */

std::string read_template(const char* name) {
    std::ifstream f(std::string(HELIX_LLAMACPP_TEMPLATES) + "/" + name, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

common_chat_params apply_with_tool(const std::string& tmpl, common_chat_tool_choice choice) {
    common_chat_templates_ptr tmpls(common_chat_templates_init(nullptr, tmpl));
    common_chat_templates_inputs in;
    common_chat_msg user;
    user.role    = "user";
    user.content = "What is the weather in Paris?";
    in.messages              = {user};
    in.add_generation_prompt = true;
    in.use_jinja             = true;
    in.tool_choice           = choice;
    common_chat_tool tool;
    tool.name        = "get_weather";
    tool.description = "Current weather for a city";
    tool.parameters  = R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})";
    in.tools = {tool};
    return common_chat_templates_apply(tmpls.get(), in);
}

/* The surviving fork patch (a5b839399, ported as helix-b12033 a0dec274e): the
 * autoparser aligns its parser with the generation-prompt tail, but that
 * prefix must not reach the GBNF. The sampler grammar sees generated tokens
 * only, so a literal "<|im_start|>assistant\n" there forces the model to
 * re-emit the prompt before any constrained output — under tool_choice
 * required, the grammar is eager and that is the very first thing sampled. */
TEST(LlamaBump, RequiredToolGrammarCarriesNoPromptTail) {
    const std::string tmpl = read_template("Qwen-Qwen2.5-7B-Instruct.jinja");
    ASSERT_FALSE(tmpl.empty()) << "template file not found under " HELIX_LLAMACPP_TEMPLATES;
    const auto params = apply_with_tool(tmpl, COMMON_CHAT_TOOL_CHOICE_REQUIRED);
    ASSERT_FALSE(params.grammar.empty());
    EXPECT_FALSE(params.grammar_lazy) << "tool_choice required must give an eager grammar";
    EXPECT_EQ(params.grammar.find("im_start"), std::string::npos)
        << "the generation prompt leaked into the grammar:\n" << params.grammar;
}

/* §4b: a streamed tool call, parsed prefix by prefix through
 * common_chat_peg_parse (now over common_chat_input), yields diffs that
 * arrive in order and rebuild exactly the final message. */
TEST(LlamaBump, StreamedToolCallDiffsRebuildTheFinalMessage) {
    const std::string tmpl = read_template("Qwen-Qwen2.5-7B-Instruct.jinja");
    ASSERT_FALSE(tmpl.empty());
    const auto params = apply_with_tool(tmpl, COMMON_CHAT_TOOL_CHOICE_AUTO);
    ASSERT_FALSE(params.parser.empty()) << "expected the autoparser for this template";

    const std::string output =
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
    common_chat_parser_params pp(params);

    common_chat_msg prev;
    std::string content;
    std::string args;
    std::string name;
    for (size_t n = 1; n <= output.size(); ++n) {
        common_chat_msg cur;
        try {
            cur = common_chat_peg_parse(params.parser, common_chat_input(output.substr(0, n)),
                                        /*is_partial=*/true, pp);
        } catch (const std::exception&) {
            continue; /* a partial the parser cannot place yet */
        }
        for (const auto& d : common_chat_msg_diff::compute_diffs(prev, cur)) {
            content += d.content_delta;
            if (d.tool_call_index != std::string::npos) {
                EXPECT_EQ(d.tool_call_index, 0u);
                name += d.tool_call_delta.name;
                args += d.tool_call_delta.arguments;
            }
        }
        prev = std::move(cur);
    }
    const common_chat_msg final_msg =
        common_chat_peg_parse(params.parser, common_chat_input(output), /*is_partial=*/false, pp);
    for (const auto& d : common_chat_msg_diff::compute_diffs(prev, final_msg)) {
        content += d.content_delta;
        if (d.tool_call_index != std::string::npos) {
            name += d.tool_call_delta.name;
            args += d.tool_call_delta.arguments;
        }
    }

    ASSERT_EQ(final_msg.tool_calls.size(), 1u);
    EXPECT_EQ(final_msg.tool_calls[0].name, "get_weather");
    EXPECT_EQ(name, final_msg.tool_calls[0].name);
    EXPECT_EQ(args, final_msg.tool_calls[0].arguments);
    EXPECT_EQ(content, final_msg.content);
    EXPECT_NE(final_msg.tool_calls[0].arguments.find("Paris"), std::string::npos);
}

} // namespace
