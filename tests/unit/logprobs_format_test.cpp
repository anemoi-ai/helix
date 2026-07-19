#include <gtest/gtest.h>
#include "src/json/response.hpp"
#include "nlohmann/json.hpp"

using namespace helix;
using json = nlohmann::json;

static ChatResponse make_response_with_logprobs() {
    TokenLogprobAlt alt;
    alt.token  = "Hi";
    alt.logprob = -7.5f;
    alt.bytes  = {72, 105};

    TokenLogprobEntry entry;
    entry.token  = "Hello";
    entry.logprob = -0.002f;
    entry.bytes  = {72, 101, 108, 108, 111};
    entry.top_logprobs.push_back(alt);

    ChoiceLogprobs lp;
    lp.content.push_back(entry);

    ChatResponse resp;
    resp.id    = "test-id";
    resp.model = "test-model";
    resp.system_fingerprint = "fp";

    Choice c;
    c.index         = 0;
    c.finish_reason = "stop";
    c.logprobs      = lp;
    resp.choices.push_back(c);
    return resp;
}

TEST(LogprobsFormat, SerializesLogprobsContent) {
    auto resp = make_response_with_logprobs();
    std::string js = resp.to_json();
    auto j = json::parse(js);

    auto& lp = j["choices"][0]["logprobs"];
    ASSERT_FALSE(lp.is_null());
    ASSERT_TRUE(lp.contains("content"));
    ASSERT_EQ(lp["content"].size(), 1u);

    auto& entry = lp["content"][0];
    EXPECT_EQ(entry["token"].get<std::string>(), "Hello");
    EXPECT_NEAR(entry["logprob"].get<float>(), -0.002f, 1e-5f);
    EXPECT_EQ(entry["bytes"].size(), 5u);
}

TEST(LogprobsFormat, SerializesTopLogprobs) {
    auto resp = make_response_with_logprobs();
    auto j = json::parse(resp.to_json());

    auto& top = j["choices"][0]["logprobs"]["content"][0]["top_logprobs"];
    ASSERT_EQ(top.size(), 1u);
    EXPECT_EQ(top[0]["token"].get<std::string>(), "Hi");
    EXPECT_NEAR(top[0]["logprob"].get<float>(), -7.5f, 1e-5f);
}

TEST(LogprobsFormat, NullWhenNoLogprobs) {
    ChatResponse resp;
    resp.id = "x";
    resp.model = "m";
    resp.system_fingerprint = "fp";
    Choice c;
    c.finish_reason = "stop";
    resp.choices.push_back(c);

    auto j = json::parse(resp.to_json());
    EXPECT_TRUE(j["choices"][0]["logprobs"].is_null());
}

TEST(LogprobsFormat, ChunkSerializesLogprobs) {
    TokenLogprobEntry entry;
    entry.token  = "Hi";
    entry.logprob = -0.5f;
    entry.bytes  = {72, 105};
    ChoiceLogprobs lp;
    lp.content.push_back(entry);

    DeltaContent d;
    d.content = "Hi";

    std::string chunk = make_chunk_json("id", 0, "m", "fp", 0, d, "", &lp);
    auto j = json::parse(chunk);
    auto& lp_j = j["choices"][0]["logprobs"];
    ASSERT_FALSE(lp_j.is_null());
    EXPECT_EQ(lp_j["content"][0]["token"].get<std::string>(), "Hi");
}
