#include <gtest/gtest.h>
#include "src/sampling/chain.hpp"
#include "src/json/request.hpp"

/* These tests verify that build_sampler_chain doesn't crash and returns
 * a non-null chain.  Full sampler-order testing requires a live llama context;
 * that's covered by the integration tests. */

using namespace helix;

static ChatRequest make_req(const std::string& json_str) {
    return ChatRequest::from_json(json_str);
}

TEST(SamplerChain, DefaultsDoNotCrash) {
    auto req   = make_req(R"({"model":"m","messages":[{"role":"user","content":"x"}]})");
    auto chain = build_sampler_chain(req);
    EXPECT_NE(chain.get(), nullptr);
}

TEST(SamplerChain, GreedyDoesNotCrash) {
    auto req   = make_req(R"({"model":"m","messages":[{"role":"user","content":"x"}],"temperature":0})");
    auto chain = build_sampler_chain(req);
    EXPECT_NE(chain.get(), nullptr);
}

TEST(SamplerChain, WithSeedDoesNotCrash) {
    auto req   = make_req(R"({"model":"m","messages":[{"role":"user","content":"x"}],"seed":12345})");
    auto chain = build_sampler_chain(req);
    EXPECT_NE(chain.get(), nullptr);
}

TEST(SamplerChain, SeedOverrideDoesNotCrash) {
    auto req   = make_req(R"({"model":"m","messages":[{"role":"user","content":"x"}]})");
    auto chain = build_sampler_chain(req, /*seed_override=*/999);
    EXPECT_NE(chain.get(), nullptr);
}

TEST(SamplerChain, PenaltiesDoNotCrash) {
    auto req   = make_req(R"({
        "model":"m","messages":[{"role":"user","content":"x"}],
        "presence_penalty":0.5,"frequency_penalty":0.3
    })");
    auto chain = build_sampler_chain(req);
    EXPECT_NE(chain.get(), nullptr);
}

TEST(SamplerChain, TopPDoesNotCrash) {
    auto req   = make_req(R"({
        "model":"m","messages":[{"role":"user","content":"x"}],
        "top_p":0.9
    })");
    auto chain = build_sampler_chain(req);
    EXPECT_NE(chain.get(), nullptr);
}

TEST(SamplerChain, TopPAtOneSkipsSampler) {
    auto req   = make_req(R"({
        "model":"m","messages":[{"role":"user","content":"x"}],
        "top_p":1.0
    })");
    auto chain = build_sampler_chain(req);
    EXPECT_NE(chain.get(), nullptr);
}

TEST(SamplerChain, TopKDoesNotCrash) {
    auto req   = make_req(R"({
        "model":"m","messages":[{"role":"user","content":"x"}],
        "top_k":10
    })");
    auto chain = build_sampler_chain(req);
    EXPECT_NE(chain.get(), nullptr);
}

TEST(SamplerChain, MinPDoesNotCrash) {
    auto req   = make_req(R"({
        "model":"m","messages":[{"role":"user","content":"x"}],
        "min_p":0.1
    })");
    auto chain = build_sampler_chain(req);
    EXPECT_NE(chain.get(), nullptr);
}

TEST(SamplerChain, MinPAtZeroSkipsSampler) {
    auto req   = make_req(R"({
        "model":"m","messages":[{"role":"user","content":"x"}],
        "min_p":0.0
    })");
    auto chain = build_sampler_chain(req);
    EXPECT_NE(chain.get(), nullptr);
}

TEST(SamplerChain, SeedOverrideTakesPrecedence) {
    auto req   = make_req(R"({
        "model":"m","messages":[{"role":"user","content":"x"}],
        "seed":12345
    })");
    auto chain = build_sampler_chain(req, /*seed_override=*/999);
    EXPECT_NE(chain.get(), nullptr);
}

TEST(SamplerChain, AllSamplingParamsCombined) {
    auto req   = make_req(R"({
        "model":"m","messages":[{"role":"user","content":"x"}],
        "temperature":0.7,
        "top_p":0.9,
        "top_k":50,
        "min_p":0.05,
        "presence_penalty":0.1,
        "frequency_penalty":0.2,
        "seed":42
    })");
    auto chain = build_sampler_chain(req);
    EXPECT_NE(chain.get(), nullptr);
}

TEST(SamplerChain, MaxPenaltiesDoNotCrash) {
    auto req   = make_req(R"({
        "model":"m","messages":[{"role":"user","content":"x"}],
        "presence_penalty":2.0,"frequency_penalty":2.0
    })");
    auto chain = build_sampler_chain(req);
    EXPECT_NE(chain.get(), nullptr);
}

TEST(SamplerChain, MinPenaltiesDoNotCrash) {
    auto req   = make_req(R"({
        "model":"m","messages":[{"role":"user","content":"x"}],
        "presence_penalty":-2.0,"frequency_penalty":-2.0
    })");
    auto chain = build_sampler_chain(req);
    EXPECT_NE(chain.get(), nullptr);
}

TEST(SamplerChain, TemperatureAtMaxValid) {
    auto req   = make_req(R"({
        "model":"m","messages":[{"role":"user","content":"x"}],
        "temperature":2.0
    })");
    auto chain = build_sampler_chain(req);
    EXPECT_NE(chain.get(), nullptr);
}

TEST(SamplerChain, WithLogitBiasNoVocab) {
    auto req   = make_req(R"({
        "model":"m","messages":[{"role":"user","content":"x"}],
        "logit_bias":{"100":5.0}
    })");
    auto chain = build_sampler_chain(req);
    EXPECT_NE(chain.get(), nullptr);
}

TEST(SamplerChain, WithGrammar) {
    auto req   = make_req(R"({"model":"m","messages":[{"role":"user","content":"x"}]})");
    auto chain = build_sampler_chain(req, 0, nullptr, "root ::= \"hello\"", false);
    EXPECT_NE(chain.get(), nullptr);
}

TEST(SamplerChain, WithGrammarLazyAndTriggers) {
    auto req   = make_req(R"({"model":"m","messages":[{"role":"user","content":"x"}]})");
    std::vector<GrammarTrigger> triggers = {
        {1, "tool_call", -1},
        {2, "\\d+", -1}
    };
    auto chain = build_sampler_chain(req, 0, nullptr, "root ::= \"hello\"", true, triggers);
    EXPECT_NE(chain.get(), nullptr);
}

TEST(SamplerChain, WithGrammarTriggerTypeToken) {
    auto req   = make_req(R"({"model":"m","messages":[{"role":"user","content":"x"}]})");
    std::vector<GrammarTrigger> triggers = {
        {0, "", 42}
    };
    auto chain = build_sampler_chain(req, 0, nullptr, "root ::= \"hello\"", true, triggers);
    EXPECT_NE(chain.get(), nullptr);
}

TEST(SamplerChain, WithGrammarTriggerTypePatternFull) {
    auto req   = make_req(R"({"model":"m","messages":[{"role":"user","content":"x"}]})");
    std::vector<GrammarTrigger> triggers = {
        {3, "tool_call", -1},
        {3, "^\\[TOOL_CALL\\]$", -1}
    };
    auto chain = build_sampler_chain(req, 0, nullptr, "root ::= \"hello\"", true, triggers);
    EXPECT_NE(chain.get(), nullptr);
}

TEST(SamplerChain, GreedyWithPenalties) {
    auto req   = make_req(R"({
        "model":"m","messages":[{"role":"user","content":"x"}],
        "temperature":0,
        "presence_penalty":1.5
    })");
    auto chain = build_sampler_chain(req);
    EXPECT_NE(chain.get(), nullptr);
}

TEST(SamplerChain, EmptyGrammarNoVocab) {
    auto req   = make_req(R"({"model":"m","messages":[{"role":"user","content":"x"}]})");
    auto chain = build_sampler_chain(req, 0, nullptr, "", false);
    EXPECT_NE(chain.get(), nullptr);
}

TEST(SamplerChain, RepeatPenaltyDoesNotCrash) {
    auto req   = make_req(R"({
        "model":"m","messages":[{"role":"user","content":"x"}],
        "repeat_penalty":1.1
    })");
    auto chain = build_sampler_chain(req);
    EXPECT_NE(chain.get(), nullptr);
}

/* Reasoning budget plumbing: without a vocab the budget sampler cannot be
 * constructed (tags cannot be tokenized), but the chain must still build for
 * every budget value the request layer can produce. */
TEST(SamplerChain, ReasoningBudgetNoVocabDoesNotCrash) {
    auto req = make_req(R"({
        "model":"m","messages":[{"role":"user","content":"x"}],
        "reasoning_budget":64
    })");
    auto chain = build_sampler_chain(req, 0, nullptr, "", false, {}, {},
                                     req.reasoning_budget.value_or(-1),
                                     "<think>", "</think>",
                                     /*thinking_forced_open=*/false);
    EXPECT_NE(chain.get(), nullptr);
}

TEST(SamplerChain, ReasoningBudgetZeroNoVocabDoesNotCrash) {
    auto req = make_req(R"({
        "model":"m","messages":[{"role":"user","content":"x"}],
        "reasoning_budget":0
    })");
    auto chain = build_sampler_chain(req, 0, nullptr, "", false, {}, {},
                                     req.reasoning_budget.value_or(-1),
                                     "<think>", "</think>",
                                     /*thinking_forced_open=*/true);
    EXPECT_NE(chain.get(), nullptr);
}

TEST(SamplerChain, ReasoningBudgetUnlimitedDoesNotCrash) {
    auto req = make_req(R"({
        "model":"m","messages":[{"role":"user","content":"x"}]
    })");
    auto chain = build_sampler_chain(req, 0, nullptr, "", false, {}, {},
                                     req.reasoning_budget.value_or(-1),
                                     "", "</think>",
                                     /*thinking_forced_open=*/false);
    EXPECT_NE(chain.get(), nullptr);
}

TEST(SamplerChain, ReasoningBudgetEmptyStartTagDoesNotCrash) {
    /* Template supplies no start tag: the chain derives one from the end tag
     * rather than leaving the budget inert. Must not crash without a vocab. */
    auto req = make_req(R"({
        "model":"m","messages":[{"role":"user","content":"x"}],
        "reasoning_budget":16
    })");
    auto chain = build_sampler_chain(req, 0, nullptr, "", false, {}, {},
                                     req.reasoning_budget.value_or(-1),
                                     "", "</think>",
                                     /*thinking_forced_open=*/false);
    EXPECT_NE(chain.get(), nullptr);
}

TEST(SamplerChain, SamplerDeleterFreesChain) {
    auto req   = make_req(R"({"model":"m","messages":[{"role":"user","content":"x"}]})");
    {
        auto chain = build_sampler_chain(req);
        EXPECT_NE(chain.get(), nullptr);
    }
}
