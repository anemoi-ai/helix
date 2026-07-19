#include <gtest/gtest.h>
#include "src/json/request.hpp"
#include "src/internal/error.hpp"

using namespace helix;

static ChatRequest make_base_req() {
    ChatRequest r;
    r.model = "test";
    Message m;
    m.role = "user";
    m.content_str = "hi";
    r.messages.push_back(m);
    return r;
}

TEST(ConflictValidation, TopLogprobsWithoutLogprobsThrows) {
    auto r = make_base_req();
    r.logprobs    = false;
    r.top_logprobs = 3;
    EXPECT_THROW(r.validate(), helix::Error);
}

TEST(ConflictValidation, TopLogprobsWithLogprobsOk) {
    auto r = make_base_req();
    r.logprobs    = true;
    r.top_logprobs = 3;
    EXPECT_NO_THROW(r.validate());
}

TEST(ConflictValidation, TopLogprobsTooHighThrows) {
    auto r = make_base_req();
    r.logprobs    = true;
    r.top_logprobs = 21;
    EXPECT_THROW(r.validate(), helix::Error);
}

TEST(ConflictValidation, TopLogprobsNegativeThrows) {
    auto r = make_base_req();
    r.logprobs    = true;
    r.top_logprobs = -1;
    EXPECT_THROW(r.validate(), helix::Error);
}

TEST(ConflictValidation, JsonObjectPlusToolsAutoThrows) {
    auto r = make_base_req();
    ResponseFormat rf;
    rf.type = "json_object";
    r.response_format = rf;
    Tool t;
    t.type = "function";
    t.function.name = "foo";
    r.tools.push_back(t);
    // No tool_choice → defaults to "auto" → should throw.
    EXPECT_THROW(r.validate(), helix::Error);
}

TEST(ConflictValidation, JsonObjectPlusToolsNoneOk) {
    auto r = make_base_req();
    ResponseFormat rf;
    rf.type = "json_object";
    r.response_format = rf;
    Tool t;
    t.type = "function";
    t.function.name = "foo";
    r.tools.push_back(t);
    r.tool_choice = std::string("none");
    EXPECT_NO_THROW(r.validate());
}

TEST(ConflictValidation, JsonSchemaWithToolsAutoOk) {
    // json_schema + tools + auto is allowed (schema for content, tools for tool calls).
    auto r = make_base_req();
    ResponseFormat rf;
    rf.type = "json_schema";
    rf.json_schema = R"({"schema":{"type":"object","additionalProperties":false}})";
    r.response_format = rf;
    Tool t;
    t.type = "function";
    t.function.name = "foo";
    r.tools.push_back(t);
    EXPECT_NO_THROW(r.validate());
}
