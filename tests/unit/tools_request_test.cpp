#include <gtest/gtest.h>
#include "json/request.hpp"
#include "internal/error.hpp"

using namespace helix;

/* ------------------------------------------------------------------ */
/*  Helpers                                                             */
/* ------------------------------------------------------------------ */

static ChatRequest parse(const char* json) {
    auto r = ChatRequest::from_json(json);
    return r;
}

/* ------------------------------------------------------------------ */
/*  Tools[] parsing                                                     */
/* ------------------------------------------------------------------ */

TEST(ToolsRequest, ToolWithoutFunctionObjectRejected) {
    /* {"type":"function"} with no function object must not parse into a
     * nameless tool. */
    EXPECT_THROW(parse(R"({
        "model":"m","messages":[{"role":"user","content":"x"}],
        "tools":[{"type":"function"}]
    })"), Error);
}

TEST(ToolsRequest, ToolWithEmptyNameRejected) {
    EXPECT_THROW(parse(R"({
        "model":"m","messages":[{"role":"user","content":"x"}],
        "tools":[{"type":"function","function":{"name":"","parameters":{}}}]
    })"), Error);
}

TEST(ToolsRequest, DeprecatedFunctionWithoutNameRejected) {
    /* The functions[] -> tools[] translation applies the same empty-name
     * rule as parse_tool. */
    EXPECT_THROW(parse(R"({
        "model":"m","messages":[{"role":"user","content":"x"}],
        "functions":[{"description":"no name","parameters":{}}]
    })"), Error);
}

TEST(ToolsRequest, ParseSingleTool) {
    auto r = parse(R"({
        "model": "m",
        "messages": [{"role":"user","content":"hi"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {"type":"object","properties":{"location":{"type":"string"}}}
            }
        }]
    })");
    ASSERT_EQ(r.tools.size(), 1u);
    EXPECT_EQ(r.tools[0].type, "function");
    EXPECT_EQ(r.tools[0].function.name, "get_weather");
    EXPECT_EQ(r.tools[0].function.description, "Get current weather");
    EXPECT_FALSE(r.tools[0].function.parameters.empty());
}

TEST(ToolsRequest, ParseMultipleTools) {
    auto r = parse(R"({
        "model": "m",
        "messages": [{"role":"user","content":"hi"}],
        "tools": [
            {"type":"function","function":{"name":"tool_a","description":"","parameters":{}}},
            {"type":"function","function":{"name":"tool_b","description":"","parameters":{}}}
        ]
    })");
    ASSERT_EQ(r.tools.size(), 2u);
    EXPECT_EQ(r.tools[0].function.name, "tool_a");
    EXPECT_EQ(r.tools[1].function.name, "tool_b");
}

TEST(ToolsRequest, ToolChoiceAuto) {
    auto r = parse(R"({"model":"m","messages":[{"role":"user","content":"x"}],
        "tools":[{"type":"function","function":{"name":"f","description":"","parameters":{}}}],
        "tool_choice":"auto"})");
    ASSERT_TRUE(r.tool_choice.has_value());
    ASSERT_TRUE(std::holds_alternative<std::string>(*r.tool_choice));
    EXPECT_EQ(std::get<std::string>(*r.tool_choice), "auto");
}

TEST(ToolsRequest, ToolChoiceNone) {
    auto r = parse(R"({"model":"m","messages":[{"role":"user","content":"x"}],
        "tools":[{"type":"function","function":{"name":"f","description":"","parameters":{}}}],
        "tool_choice":"none"})");
    EXPECT_EQ(std::get<std::string>(*r.tool_choice), "none");
}

TEST(ToolsRequest, ToolChoiceRequired) {
    auto r = parse(R"({"model":"m","messages":[{"role":"user","content":"x"}],
        "tools":[{"type":"function","function":{"name":"f","description":"","parameters":{}}}],
        "tool_choice":"required"})");
    EXPECT_EQ(std::get<std::string>(*r.tool_choice), "required");
}

TEST(ToolsRequest, ToolChoiceSpecificFunction) {
    auto r = parse(R"({"model":"m","messages":[{"role":"user","content":"x"}],
        "tools":[{"type":"function","function":{"name":"f","description":"","parameters":{}}}],
        "tool_choice":{"type":"function","function":{"name":"get_weather"}}})");
    ASSERT_TRUE(r.tool_choice.has_value());
    ASSERT_TRUE(std::holds_alternative<ToolChoiceFunction>(*r.tool_choice));
    EXPECT_EQ(std::get<ToolChoiceFunction>(*r.tool_choice).name, "get_weather");
}

TEST(ToolsRequest, ParallelToolCallsDefault) {
    auto r = parse(R"({"model":"m","messages":[{"role":"user","content":"x"}]})");
    EXPECT_TRUE(r.parallel_tool_calls);
}

TEST(ToolsRequest, ParallelToolCallsFalse) {
    auto r = parse(R"({"model":"m","messages":[{"role":"user","content":"x"}],
        "parallel_tool_calls":false})");
    EXPECT_FALSE(r.parallel_tool_calls);
}

/* ------------------------------------------------------------------ */
/*  Tool-role messages                                                  */
/* ------------------------------------------------------------------ */

TEST(ToolsRequest, ParseToolRoleMessage) {
    auto r = parse(R"({
        "model": "m",
        "messages": [
            {"role":"user","content":"What's the weather?"},
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\"}"}}
            ]},
            {"role":"tool","content":"sunny and 20C","tool_call_id":"call_abc"}
        ]
    })");
    ASSERT_EQ(r.messages.size(), 3u);
    EXPECT_EQ(r.messages[1].role, "assistant");
    ASSERT_EQ(r.messages[1].tool_calls.size(), 1u);
    EXPECT_EQ(r.messages[1].tool_calls[0].id, "call_abc");
    EXPECT_EQ(r.messages[1].tool_calls[0].function_name, "get_weather");
    EXPECT_EQ(r.messages[1].tool_calls[0].function_arguments, "{\"location\":\"Paris\"}");
    EXPECT_EQ(r.messages[2].role, "tool");
    EXPECT_EQ(r.messages[2].tool_call_id, "call_abc");
    EXPECT_EQ(r.messages[2].content_str, "sunny and 20C");
}

TEST(ToolsRequest, ToolRoleRequiresToolCallId) {
    /* tool-role message without tool_call_id → validation error */
    EXPECT_THROW({
        auto r = parse(R"({"model":"m","messages":[
            {"role":"tool","content":"result"}
        ]})");
        r.validate();
    }, helix::Error);
}

TEST(ToolsRequest, ArgumentsObjectRejected) {
    /* arguments as object → parse error with descriptive message */
    EXPECT_THROW({
        parse(R"({
            "model":"m","messages":[
                {"role":"assistant","tool_calls":[
                    {"id":"x","type":"function","function":{
                        "name":"f","arguments":{"key":"val"}
                    }}
                ]}
            ]
        })");
    }, helix::Error);
}

/* ------------------------------------------------------------------ */
/*  Deprecated functions→tools translation                             */
/* ------------------------------------------------------------------ */

TEST(ToolsRequest, FunctionsV0Translated) {
    auto r = parse(R"({
        "model":"m",
        "messages":[{"role":"user","content":"x"}],
        "functions":[{
            "name":"get_weather",
            "description":"Get weather",
            "parameters":{"type":"object","properties":{"loc":{"type":"string"}}}
        }],
        "function_call":"auto"
    })");
    ASSERT_EQ(r.tools.size(), 1u);
    EXPECT_EQ(r.tools[0].function.name, "get_weather");
    ASSERT_TRUE(r.tool_choice.has_value());
    EXPECT_EQ(std::get<std::string>(*r.tool_choice), "auto");
}

TEST(ToolsRequest, FunctionCallNoneTranslated) {
    auto r = parse(R"({
        "model":"m",
        "messages":[{"role":"user","content":"x"}],
        "functions":[{"name":"f","description":"","parameters":{}}],
        "function_call":"none"
    })");
    EXPECT_EQ(std::get<std::string>(*r.tool_choice), "none");
}

TEST(ToolsRequest, FunctionCallSpecificTranslated) {
    auto r = parse(R"({
        "model":"m",
        "messages":[{"role":"user","content":"x"}],
        "functions":[{"name":"get_weather","description":"","parameters":{}}],
        "function_call":{"name":"get_weather"}
    })");
    ASSERT_TRUE(std::holds_alternative<ToolChoiceFunction>(*r.tool_choice));
    EXPECT_EQ(std::get<ToolChoiceFunction>(*r.tool_choice).name, "get_weather");
}

TEST(ToolsRequest, FunctionsNotOverrideTools) {
    /* When tools[] is already present, functions[] is ignored. */
    auto r = parse(R"({
        "model":"m",
        "messages":[{"role":"user","content":"x"}],
        "tools":[{"type":"function","function":{"name":"t1","description":"","parameters":{}}}],
        "functions":[{"name":"t2","description":"","parameters":{}}]
    })");
    ASSERT_EQ(r.tools.size(), 1u);
    EXPECT_EQ(r.tools[0].function.name, "t1");
}
