#include <gtest/gtest.h>
#include "src/json/response.hpp"
#include "nlohmann/json.hpp"

using namespace helix;
using json = nlohmann::json;

static ChatResponse make_simple_response() {
    ChatResponse r;
    r.id                 = "chatcmpl-helix-test";
    r.created            = 1748000000;
    r.model              = "local-helix";
    r.system_fingerprint = "helix-0.1.0";

    Choice c;
    c.index               = 0;
    c.message.role        = "assistant";
    c.message.content     = "Hello!";
    c.finish_reason       = "stop";
    r.choices.push_back(c);

    r.usage.prompt_tokens     = 10;
    r.usage.completion_tokens = 5;
    r.usage.total_tokens      = 15;
    return r;
}

TEST(ResponseSerialize, BasicShape) {
    auto resp = make_simple_response();
    auto j    = json::parse(resp.to_json());

    EXPECT_EQ(j["object"], "chat.completion");
    EXPECT_EQ(j["id"],     "chatcmpl-helix-test");
    EXPECT_EQ(j["model"],  "local-helix");
    ASSERT_TRUE(j["choices"].is_array());
    ASSERT_EQ(j["choices"].size(), 1u);
    EXPECT_EQ(j["choices"][0]["message"]["content"], "Hello!");
    EXPECT_EQ(j["choices"][0]["message"]["role"],    "assistant");
    EXPECT_EQ(j["choices"][0]["finish_reason"],      "stop");
    EXPECT_EQ(j["usage"]["prompt_tokens"],           10);
    EXPECT_EQ(j["usage"]["completion_tokens"],       5);
    EXPECT_EQ(j["usage"]["total_tokens"],            15);
}

TEST(ResponseSerialize, LogprobsNullPresent) {
    auto resp = make_simple_response();
    auto j    = json::parse(resp.to_json());
    EXPECT_TRUE(j["choices"][0]["logprobs"].is_null());
}

TEST(ResponseSerialize, SystemFingerprintPresent) {
    auto resp = make_simple_response();
    auto j    = json::parse(resp.to_json());
    EXPECT_EQ(j["system_fingerprint"], "helix-0.1.0");
}

TEST(ResponseSerialize, MultipleChoices) {
    ChatResponse resp;
    resp.id = "chatcmpl-helix-multi";
    resp.created = 1;
    resp.model = "m";
    resp.system_fingerprint = "fp";

    for (int i = 0; i < 3; ++i) {
        Choice c;
        c.index           = i;
        c.message.content = "choice " + std::to_string(i);
        c.finish_reason   = "stop";
        resp.choices.push_back(c);
    }
    resp.usage = {5, 9, 14};

    auto j = json::parse(resp.to_json());
    ASSERT_EQ(j["choices"].size(), 3u);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(j["choices"][i]["index"], i);
    }
}

TEST(ChunkSerialize, RoleChunk) {
    DeltaContent d;
    d.role = "assistant";
    d.content = "";
    std::string s = make_chunk_json("id", 1, "m", "fp", 0, d);
    auto j = json::parse(s);

    EXPECT_EQ(j["object"], "chat.completion.chunk");
    EXPECT_EQ(j["choices"][0]["delta"]["role"], "assistant");
    EXPECT_TRUE(j["choices"][0]["finish_reason"].is_null());
}

TEST(ChunkSerialize, ContentDelta) {
    DeltaContent d;
    d.content = "Hello";
    std::string s = make_chunk_json("id", 1, "m", "fp", 0, d);
    auto j = json::parse(s);
    EXPECT_EQ(j["choices"][0]["delta"]["content"], "Hello");
}

TEST(ChunkSerialize, FinalChunkHasFinishReason) {
    DeltaContent d;
    std::string s = make_chunk_json("id", 1, "m", "fp", 0, d, "stop");
    auto j = json::parse(s);
    EXPECT_EQ(j["choices"][0]["finish_reason"], "stop");
}

TEST(ChunkSerialize, UsageChunk) {
    Usage u;
    u.prompt_tokens = 10; u.completion_tokens = 5; u.total_tokens = 15;
    std::string s = make_usage_chunk_json("id", 1, "m", "fp", u);
    auto j = json::parse(s);
    EXPECT_TRUE(j["choices"].is_array());
    EXPECT_EQ(j["choices"].size(), 0u);
    EXPECT_EQ(j["usage"]["total_tokens"], 15);
}
