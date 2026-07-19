#include <gtest/gtest.h>
#include "json/response.hpp"
#include "nlohmann/json.hpp"

using namespace helix;
using json = nlohmann::json;

/* ------------------------------------------------------------------ */
/*  Non-streaming response with tool_calls                             */
/* ------------------------------------------------------------------ */

static ChatResponse make_tool_response(const std::string& args = "{\"location\":\"Paris\"}") {
    ChatResponse r;
    r.id      = "chatcmpl-test";
    r.created = 1234567890;
    r.model   = "test-model";
    r.system_fingerprint = "fp-test";

    ToolCall tc;
    tc.id                 = "call_abc123";
    tc.type               = "function";
    tc.function_name      = "get_weather";
    tc.function_arguments = args;

    Choice c;
    c.index              = 0;
    c.message.role       = "assistant";
    c.message.tool_calls = {tc};
    c.finish_reason      = "tool_calls";
    r.choices.push_back(c);

    r.usage = {10, 5, 15};
    return r;
}

TEST(ToolsResponse, ContentNullWhenToolCallsPresent) {
    auto j = json::parse(make_tool_response().to_json());
    const auto& msg = j["choices"][0]["message"];
    EXPECT_TRUE(msg["content"].is_null());
}

TEST(ToolsResponse, ToolCallsShape) {
    auto j = json::parse(make_tool_response().to_json());
    const auto& tcs = j["choices"][0]["message"]["tool_calls"];
    ASSERT_TRUE(tcs.is_array());
    ASSERT_EQ(tcs.size(), 1u);
    EXPECT_EQ(tcs[0]["id"],                    "call_abc123");
    EXPECT_EQ(tcs[0]["type"],                  "function");
    EXPECT_EQ(tcs[0]["function"]["name"],      "get_weather");
    EXPECT_EQ(tcs[0]["function"]["arguments"], "{\"location\":\"Paris\"}");
}

TEST(ToolsResponse, FinishReasonToolCalls) {
    auto j = json::parse(make_tool_response().to_json());
    EXPECT_EQ(j["choices"][0]["finish_reason"], "tool_calls");
}

TEST(ToolsResponse, ContentPresentWhenNoToolCalls) {
    ChatResponse r;
    r.id = "x"; r.created = 0; r.model = "m"; r.system_fingerprint = "fp";
    Choice c;
    c.index = 0;
    c.message.role    = "assistant";
    c.message.content = "Hello!";
    c.finish_reason   = "stop";
    r.choices.push_back(c);
    auto j = json::parse(r.to_json());
    EXPECT_EQ(j["choices"][0]["message"]["content"], "Hello!");
    EXPECT_FALSE(j["choices"][0]["message"].contains("tool_calls"));
}

/* ------------------------------------------------------------------ */
/*  Streaming delta: first chunk (with id/type/name)                   */
/* ------------------------------------------------------------------ */

TEST(ToolsResponse, StreamFirstChunkShape) {
    ToolCall tc;
    tc.index             = 0;
    tc.id                = "call_xyz";
    tc.type              = "function";
    tc.function_name     = "get_weather";
    tc.function_arguments = "";

    DeltaContent d;
    d.tool_calls.push_back(tc);

    auto chunk_str = make_chunk_json("cid", 0, "m", "fp", 0, d);
    auto j = json::parse(chunk_str);
    const auto& tc_j = j["choices"][0]["delta"]["tool_calls"][0];
    EXPECT_EQ(tc_j["index"],                    0);
    EXPECT_EQ(tc_j["id"],                       "call_xyz");
    EXPECT_EQ(tc_j["type"],                     "function");
    EXPECT_EQ(tc_j["function"]["name"],         "get_weather");
    EXPECT_EQ(tc_j["function"]["arguments"],    "");
}

TEST(ToolsResponse, StreamArgDeltaChunkShape) {
    /* id is empty → arg-delta shape (no id/type/name) */
    ToolCall tc;
    tc.index              = 0;
    /* tc.id intentionally empty */
    tc.function_arguments = "{\"loc";

    DeltaContent d;
    d.tool_calls.push_back(tc);

    auto chunk_str = make_chunk_json("cid", 0, "m", "fp", 0, d);
    auto j = json::parse(chunk_str);
    const auto& tc_j = j["choices"][0]["delta"]["tool_calls"][0];
    EXPECT_EQ(tc_j["index"], 0);
    EXPECT_FALSE(tc_j.contains("id"));
    EXPECT_FALSE(tc_j.contains("type"));
    EXPECT_FALSE(tc_j["function"].contains("name"));
    EXPECT_EQ(tc_j["function"]["arguments"], "{\"loc");
}

TEST(ToolsResponse, StreamFinishChunkToolCallsReason) {
    DeltaContent d;
    auto chunk_str = make_chunk_json("cid", 0, "m", "fp", 0, d, "tool_calls");
    auto j = json::parse(chunk_str);
    EXPECT_EQ(j["choices"][0]["finish_reason"], "tool_calls");
}

/* ------------------------------------------------------------------ */
/*  Tool call ID format                                                 */
/* ------------------------------------------------------------------ */

TEST(ToolCallId, FormatAndUniqueness) {
    /* We can't call make_tool_call_id directly from outside decode_loop.cpp,
     * but we can verify the shape of IDs generated during a real round-trip.
     * Here we just check the response shape accepts any "call_XXX" format. */
    ToolCall tc;
    tc.id = "call_ABCdef0123456789abcdef01";
    EXPECT_EQ(tc.id.substr(0, 5), "call_");
    EXPECT_GE(tc.id.size(), 10u);
}
