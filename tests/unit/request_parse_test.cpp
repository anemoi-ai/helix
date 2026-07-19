#include <gtest/gtest.h>
#include "src/json/request.hpp"
#include "src/internal/error.hpp"

using namespace helix;

/* ---- Parsing success ---- */

TEST(RequestParse, MinimalValid) {
    auto r = ChatRequest::from_json(R"({"model":"gpt-4","messages":[{"role":"user","content":"hi"}]})");
    EXPECT_EQ(r.model, "gpt-4");
    ASSERT_EQ(r.messages.size(), 1u);
    EXPECT_EQ(r.messages[0].role, "user");
    EXPECT_EQ(r.messages[0].content_str, "hi");
}

TEST(RequestParse, AllSamplingFields) {
    auto r = ChatRequest::from_json(R"({
        "model": "m",
        "messages": [{"role":"user","content":"x"}],
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "min_p": 0.03,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "seed": 42,
        "max_tokens": 100,
        "n": 3,
        "stop": ["stop1", "stop2"]
    })");
    EXPECT_FLOAT_EQ(*r.temperature, 0.7f);
    EXPECT_FLOAT_EQ(*r.top_p, 0.9f);
    EXPECT_EQ(*r.top_k, 50);
    EXPECT_FLOAT_EQ(*r.min_p, 0.03f);
    EXPECT_FLOAT_EQ(*r.presence_penalty, 0.1f);
    EXPECT_FLOAT_EQ(*r.frequency_penalty, 0.2f);
    EXPECT_EQ(*r.seed, 42u);
    EXPECT_EQ(*r.max_tokens, 100);
    EXPECT_EQ(*r.n, 3);
    ASSERT_EQ(r.stop.size(), 2u);
    EXPECT_EQ(r.stop[0], "stop1");
    EXPECT_EQ(r.stop[1], "stop2");
}

TEST(RequestParse, StopCanBeString) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"stop":"END"})");
    ASSERT_EQ(r.stop.size(), 1u);
    EXPECT_EQ(r.stop[0], "END");
}

TEST(RequestParse, MaxCompletionTokensTakesPrecedence) {
    auto r = ChatRequest::from_json(R"({
        "model":"m","messages":[{"role":"user","content":"x"}],
        "max_tokens":100,"max_completion_tokens":50
    })");
    EXPECT_EQ(*r.max_tokens, 100);
    EXPECT_EQ(*r.max_completion_tokens, 50);
    EXPECT_EQ(r.effective_max_tokens(1024, 10), 50);
}

TEST(RequestParse, UnknownFieldsIgnored) {
    EXPECT_NO_THROW(
        ChatRequest::from_json(R"({
            "model":"m","messages":[{"role":"user","content":"x"}],
            "future_field_we_dont_know": true,
            "another_unknown": 42
        })")
    );
}

TEST(RequestParse, DeveloperRoleAccepted) {
    auto r = ChatRequest::from_json(R"({
        "model":"m",
        "messages":[{"role":"developer","content":"be helpful"}]
    })");
    EXPECT_EQ(r.messages[0].role, "developer");
}

TEST(RequestParse, ArrayContentParsedAsArray) {
    auto r = ChatRequest::from_json(R"({
        "model":"m",
        "messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]
    })");
    ASSERT_EQ(r.messages[0].content_parts.size(), 1u);
    EXPECT_EQ(r.messages[0].content_parts[0].type, "text");
}

TEST(RequestParse, LogitBias) {
    auto r = ChatRequest::from_json(R"({
        "model":"m","messages":[{"role":"user","content":"x"}],
        "logit_bias":{"100":5.0,"200":-5.0}
    })");
    EXPECT_FLOAT_EQ(r.logit_bias.at("100"), 5.0f);
    EXPECT_FLOAT_EQ(r.logit_bias.at("200"), -5.0f);
}

/* ---- Parsing failures ---- */

TEST(RequestParse, MissingModel) {
    EXPECT_THROW(
        ChatRequest::from_json(R"({"messages":[{"role":"user","content":"x"}]})"),
        Error
    );
}

TEST(RequestParse, MissingMessages) {
    EXPECT_THROW(
        ChatRequest::from_json(R"({"model":"m"})"),
        Error
    );
}

TEST(RequestParse, InvalidJSON) {
    EXPECT_THROW(ChatRequest::from_json("{bad json"), Error);
}

TEST(RequestParse, TemperatureNotNumber) {
    EXPECT_THROW(
        ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"temperature":"hot"})"),
        Error
    );
}

/* ---- Validation ---- */

TEST(RequestValidate, TemperatureRange) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"temperature":3.0})");
    EXPECT_THROW(r.validate(), Error);
}

TEST(RequestValidate, NMustBePositive) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"n":0})");
    EXPECT_THROW(r.validate(), Error);
}

TEST(RequestValidate, ValidPassesWithoutThrow) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}]})");
    EXPECT_NO_THROW(r.validate());
}

TEST(RequestValidate, ContextFullCheck) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"max_tokens":100})");
    /* 200 prompt tokens fills the entire ctx_size=200 context → no room */
    EXPECT_THROW(r.effective_max_tokens(200, 200), Error);
    /* 150 prompt tokens in a ctx of 200 → 50 slots; smaller than max_tokens=100, so returns 50. */
    EXPECT_EQ(r.effective_max_tokens(200, 150), 50);
}

/* ---- n bounds ---- */

TEST(RequestValidate, NExceeds128) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"n":129})");
    EXPECT_THROW(r.validate(), Error);
}

TEST(RequestValidate, NAt128IsValid) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"n":128})");
    EXPECT_NO_THROW(r.validate());
}

TEST(RequestValidate, NAt1IsValid) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"n":1})");
    EXPECT_NO_THROW(r.validate());
}

/* ---- temperature boundaries ---- */

TEST(RequestValidate, TemperatureZeroValid) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"temperature":0.0})");
    EXPECT_NO_THROW(r.validate());
}

TEST(RequestValidate, TemperatureTwoValid) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"temperature":2.0})");
    EXPECT_NO_THROW(r.validate());
}

TEST(RequestValidate, TemperatureNegativeRejected) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"temperature":-0.1})");
    EXPECT_THROW(r.validate(), Error);
}

TEST(RequestValidate, TemperatureAboveTwoRejected) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"temperature":2.1})");
    EXPECT_THROW(r.validate(), Error);
}

/* ---- penalty boundaries ---- */

TEST(RequestValidate, PresencePenaltyAtNegTwoValid) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"presence_penalty":-2.0})");
    EXPECT_NO_THROW(r.validate());
}

TEST(RequestValidate, PresencePenaltyAtTwoValid) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"presence_penalty":2.0})");
    EXPECT_NO_THROW(r.validate());
}

TEST(RequestValidate, PresencePenaltyAboveTwoRejected) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"presence_penalty":2.1})");
    EXPECT_THROW(r.validate(), Error);
}

TEST(RequestValidate, FrequencyPenaltyBelowNegTwoRejected) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"frequency_penalty":-2.1})");
    EXPECT_THROW(r.validate(), Error);
}

TEST(RequestValidate, FrequencyPenaltyAtTwoValid) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"frequency_penalty":2.0})");
    EXPECT_NO_THROW(r.validate());
}

/* ---- top_logprobs ---- */

TEST(RequestValidate, TopLogprobsZeroValidWithoutLogprobs) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"top_logprobs":0})");
    EXPECT_NO_THROW(r.validate());
}

TEST(RequestValidate, TopLogprobsPositiveRequiresLogprobs) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"top_logprobs":5})");
    EXPECT_THROW(r.validate(), Error);
}

TEST(RequestValidate, TopLogprobsWithLogprobsTrueValid) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"logprobs":true,"top_logprobs":5})");
    EXPECT_NO_THROW(r.validate());
}

TEST(RequestValidate, TopLogprobsNegativeRejected) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"top_logprobs":-1})");
    EXPECT_THROW(r.validate(), Error);
}

TEST(RequestValidate, TopLogprobsAboveTwentyRejected) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"logprobs":true,"top_logprobs":21})");
    EXPECT_THROW(r.validate(), Error);
}

TEST(RequestValidate, TopLogprobsAtTwentyValid) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"logprobs":true,"top_logprobs":20})");
    EXPECT_NO_THROW(r.validate());
}

/* ---- seed bounds ---- */

TEST(RequestValidate, SeedNegativeRejected) {
    EXPECT_THROW(
        ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"seed":-1})"),
        Error);
}

TEST(RequestValidate, SeedAboveUint32MaxRejected) {
    EXPECT_THROW(
        ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"seed":4294967296})"),
        Error);
}

TEST(RequestValidate, SeedAtUint32MaxValid) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"seed":4294967295})");
    EXPECT_NO_THROW(r.validate());
    EXPECT_EQ(*r.seed, 4294967295u);
}

TEST(RequestValidate, SeedZeroValid) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"seed":0})");
    EXPECT_NO_THROW(r.validate());
    EXPECT_EQ(*r.seed, 0u);
}

/* ---- top_k / min_p bounds ---- */

TEST(RequestValidate, TopKNegativeRejected) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"top_k":-1})");
    EXPECT_THROW(r.validate(), Error);
}

TEST(RequestValidate, TopKZeroValid) {
    /* 0 disables top-k filtering (llama.cpp convention). */
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"top_k":0})");
    EXPECT_NO_THROW(r.validate());
    EXPECT_EQ(*r.top_k, 0);
}

TEST(RequestValidate, TopKPositiveValid) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"top_k":40})");
    EXPECT_NO_THROW(r.validate());
    EXPECT_EQ(*r.top_k, 40);
}

TEST(RequestValidate, MinPNegativeRejected) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"min_p":-0.1})");
    EXPECT_THROW(r.validate(), Error);
}

TEST(RequestValidate, MinPAboveOneRejected) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"min_p":1.5})");
    EXPECT_THROW(r.validate(), Error);
}

TEST(RequestValidate, MinPAtBoundsValid) {
    auto r0 = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"min_p":0.0})");
    EXPECT_NO_THROW(r0.validate());
    auto r1 = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"min_p":1.0})");
    EXPECT_NO_THROW(r1.validate());
}

/* ---- reasoning_content in assistant history ---- */

TEST(RequestParse, ReasoningContentParsed) {
    auto r = ChatRequest::from_json(R"({
        "model":"m",
        "messages":[
            {"role":"user","content":"q"},
            {"role":"assistant","content":"a","reasoning_content":"thought"}
        ]
    })");
    ASSERT_EQ(r.messages.size(), 2u);
    ASSERT_TRUE(r.messages[1].reasoning_content.has_value());
    EXPECT_EQ(*r.messages[1].reasoning_content, "thought");
}

TEST(RequestParse, ReasoningContentAbsentByDefault) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"assistant","content":"a"}]})");
    EXPECT_FALSE(r.messages[0].reasoning_content.has_value());
}

/* ---- repeat_penalty (Helix extension) ---- */

TEST(RequestParse, RepeatPenaltyParsed) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"repeat_penalty":1.1})");
    ASSERT_TRUE(r.repeat_penalty.has_value());
    EXPECT_FLOAT_EQ(*r.repeat_penalty, 1.1f);
}

TEST(RequestParse, RepeatPenaltyAbsentByDefault) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}]})");
    EXPECT_FALSE(r.repeat_penalty.has_value());
}

TEST(RequestValidate, RepeatPenaltyZeroRejected) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"repeat_penalty":0.0})");
    EXPECT_THROW(r.validate(), Error);
}

TEST(RequestValidate, RepeatPenaltyNegativeRejected) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"repeat_penalty":-1.0})");
    EXPECT_THROW(r.validate(), Error);
}

TEST(RequestValidate, RepeatPenaltyOneValid) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"repeat_penalty":1.0})");
    EXPECT_NO_THROW(r.validate());
}

TEST(RequestParse, RepeatPenaltyNotNumberRejected) {
    EXPECT_THROW(
        ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"repeat_penalty":"high"})"),
        Error);
}

/* ---- reasoning_budget (Helix extension) ---- */

TEST(RequestParse, ReasoningBudgetParsed) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"reasoning_budget":128})");
    ASSERT_TRUE(r.reasoning_budget.has_value());
    EXPECT_EQ(*r.reasoning_budget, 128);
}

TEST(RequestParse, ReasoningBudgetAbsentByDefault) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}]})");
    EXPECT_FALSE(r.reasoning_budget.has_value());
}

TEST(RequestParse, ReasoningBudgetNotIntegerRejected) {
    EXPECT_THROW(
        ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"reasoning_budget":1.5})"),
        Error);
}

TEST(RequestValidate, ReasoningBudgetMinusOneValid) {
    /* -1 is the explicit "unlimited" sentinel. */
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"reasoning_budget":-1})");
    EXPECT_NO_THROW(r.validate());
}

TEST(RequestValidate, ReasoningBudgetZeroValid) {
    /* 0 means "close the thinking block immediately". */
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"reasoning_budget":0})");
    EXPECT_NO_THROW(r.validate());
}

TEST(RequestValidate, ReasoningBudgetBelowMinusOneRejected) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"reasoning_budget":-5})");
    EXPECT_THROW(r.validate(), Error);
}

/* ---- stop sequences ---- */

TEST(RequestValidate, EmptyStopStringRejected) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"stop":""})");
    EXPECT_THROW(r.validate(), Error);
}

TEST(RequestValidate, EmptyStopInArrayRejected) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"stop":["","x"]})");
    EXPECT_THROW(r.validate(), Error);
}

TEST(RequestValidate, StopMoreThanFourRejected) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"stop":["a","b","c","d","e"]})");
    EXPECT_THROW(r.validate(), Error);
}

TEST(RequestValidate, StopExactlyFourValid) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"stop":["a","b","c","d"]})");
    EXPECT_NO_THROW(r.validate());
}

/* ---- stream_options ---- */

TEST(RequestParse, StreamOptionsIncludeUsage) {
    auto r = ChatRequest::from_json(R"({
        "model":"m","messages":[{"role":"user","content":"x"}],
        "stream":true,
        "stream_options":{"include_usage":true}
    })");
    ASSERT_TRUE(r.stream_options.has_value());
    EXPECT_TRUE(r.stream_options->include_usage);
}

TEST(RequestParse, StreamOptionsIncludeUsageFalse) {
    auto r = ChatRequest::from_json(R"({
        "model":"m","messages":[{"role":"user","content":"x"}],
        "stream_options":{"include_usage":false}
    })");
    ASSERT_TRUE(r.stream_options.has_value());
    EXPECT_FALSE(r.stream_options->include_usage);
}

TEST(RequestParse, StreamOptionsAbsentByDefault) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}]})");
    EXPECT_FALSE(r.stream_options.has_value());
}

/* ---- top_p ---- */

TEST(RequestValidate, TopPAboveOneRejected) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"top_p":1.1})");
    EXPECT_THROW(r.validate(), Error);
}

TEST(RequestValidate, TopPZeroValid) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"top_p":0.0})");
    EXPECT_NO_THROW(r.validate());
}

TEST(RequestValidate, TopPAtOneValid) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"top_p":1.0})");
    EXPECT_NO_THROW(r.validate());
}

/* ---- max_tokens ---- */

TEST(RequestValidate, MaxTokensZeroRejected) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"max_tokens":0})");
    EXPECT_THROW(r.validate(), Error);
}

TEST(RequestValidate, MaxTokensNegativeRejected) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[{"role":"user","content":"x"}],"max_tokens":-1})");
    EXPECT_THROW(r.validate(), Error);
}

/* ---- tool validation ---- */

TEST(RequestValidate, ToolRoleWithoutToolCallIdRejected) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[
        {"role":"tool","content":"result"}
    ]})");
    EXPECT_THROW(r.validate(), Error);
}

TEST(RequestValidate, ToolRoleWithToolCallIdValid) {
    auto r = ChatRequest::from_json(R"({"model":"m","messages":[
        {"role":"tool","tool_call_id":"call_123","content":"result"}
    ]})");
    EXPECT_NO_THROW(r.validate());
}

/* ---- response_format + tools conflict ---- */

TEST(RequestValidate, JsonObjectWithToolsAndAutoToolChoiceRejected) {
    auto r = ChatRequest::from_json(R"({
        "model":"m",
        "messages":[{"role":"user","content":"x"}],
        "tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],
        "response_format":{"type":"json_object"}
    })");
    EXPECT_THROW(r.validate(), Error);
}

TEST(RequestValidate, JsonObjectWithToolsAndNoneToolChoiceValid) {
    auto r = ChatRequest::from_json(R"({
        "model":"m",
        "messages":[{"role":"user","content":"x"}],
        "tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],
        "tool_choice":"none",
        "response_format":{"type":"json_object"}
    })");
    EXPECT_NO_THROW(r.validate());
}

/* ---- tool_choice enum ---- */

TEST(RequestParse, ToolChoiceAutoValid) {
    auto r = ChatRequest::from_json(R"({
        "model":"m","messages":[{"role":"user","content":"x"}],
        "tool_choice":"auto"
    })");
    ASSERT_TRUE(r.tool_choice.has_value());
    EXPECT_EQ(std::get<std::string>(*r.tool_choice), "auto");
}

TEST(RequestParse, ToolChoiceNoneValid) {
    auto r = ChatRequest::from_json(R"({
        "model":"m","messages":[{"role":"user","content":"x"}],
        "tool_choice":"none"
    })");
    ASSERT_TRUE(r.tool_choice.has_value());
    EXPECT_EQ(std::get<std::string>(*r.tool_choice), "none");
}

TEST(RequestParse, ToolChoiceRequiredValid) {
    auto r = ChatRequest::from_json(R"({
        "model":"m","messages":[{"role":"user","content":"x"}],
        "tool_choice":"required"
    })");
    ASSERT_TRUE(r.tool_choice.has_value());
    EXPECT_EQ(std::get<std::string>(*r.tool_choice), "required");
}

TEST(RequestParse, ToolChoiceInvalidStringRejected) {
    EXPECT_THROW(
        ChatRequest::from_json(R"({
            "model":"m","messages":[{"role":"user","content":"x"}],
            "tool_choice":"requried"
        })"),
        Error);
}
