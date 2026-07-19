#include <gtest/gtest.h>
#include "base64.hpp"    /* from llama.cpp/common */
#include "src/json/request.hpp"
#include "src/json/response.hpp"
#include "src/internal/error.hpp"
#include "nlohmann/json.hpp"

#include <cstring>

using namespace helix;
using json = nlohmann::json;

/* Run `fn`, assert it throws helix::Error with the given status and param. */
template <typename Fn>
static void expect_error(Fn&& fn, helix_status_t status, const std::string& param) {
    try {
        fn();
        FAIL() << "expected helix::Error(status=" << status << ", param=" << param << ")";
    } catch (const Error& e) {
        EXPECT_EQ(e.status, status);
        EXPECT_EQ(e.param, param);
    }
}

/* ---- Parsing success ---- */

TEST(EmbeddingsParse, SingleStringNormalizedToOneVec) {
    auto r = EmbeddingsRequest::from_json(R"({"model":"m","input":"hello"})");
    EXPECT_EQ(r.model, "m");
    ASSERT_EQ(r.inputs.size(), 1u);
    EXPECT_EQ(r.inputs[0], "hello");
    EXPECT_EQ(r.encoding, EmbeddingsRequest::Encoding::Float);
}

TEST(EmbeddingsParse, ArrayInputPreservesOrder) {
    auto r = EmbeddingsRequest::from_json(R"({"model":"m","input":["a","b","c"]})");
    ASSERT_EQ(r.inputs.size(), 3u);
    EXPECT_EQ(r.inputs[0], "a");
    EXPECT_EQ(r.inputs[1], "b");
    EXPECT_EQ(r.inputs[2], "c");
}

TEST(EmbeddingsParse, EncodingFormatFloat) {
    auto r = EmbeddingsRequest::from_json(
        R"({"model":"m","input":"x","encoding_format":"float"})");
    EXPECT_EQ(r.encoding, EmbeddingsRequest::Encoding::Float);
}

TEST(EmbeddingsParse, EncodingFormatBase64) {
    auto r = EmbeddingsRequest::from_json(
        R"({"model":"m","input":"x","encoding_format":"base64"})");
    EXPECT_EQ(r.encoding, EmbeddingsRequest::Encoding::Base64);
}

TEST(EmbeddingsParse, UnknownFieldsIgnored) {
    EXPECT_NO_THROW(EmbeddingsRequest::from_json(R"({
        "model":"m","input":"x",
        "future_field_we_dont_know": true,
        "user": "abc"
    })"));
}

TEST(EmbeddingsParse, MaxInputArrayAccepted) {
    json j = {{"model", "m"}};
    j["input"] = json::array();
    for (int i = 0; i < 2048; ++i) j["input"].push_back("t");
    auto r = EmbeddingsRequest::from_json(j.dump());
    EXPECT_EQ(r.inputs.size(), 2048u);
}

/* ---- Parsing failures ---- */

TEST(EmbeddingsParse, BadJsonRejected) {
    EXPECT_THROW(EmbeddingsRequest::from_json("{bad json"), Error);
}

TEST(EmbeddingsParse, MissingModelRejected) {
    expect_error([] { EmbeddingsRequest::from_json(R"({"input":"x"})"); },
                 HELIX_E_VALIDATION, "model");
}

TEST(EmbeddingsParse, NonStringModelRejected) {
    expect_error([] { EmbeddingsRequest::from_json(R"({"model":3,"input":"x"})"); },
                 HELIX_E_VALIDATION, "model");
}

TEST(EmbeddingsParse, MissingInputRejected) {
    expect_error([] { EmbeddingsRequest::from_json(R"({"model":"m"})"); },
                 HELIX_E_VALIDATION, "input");
}

TEST(EmbeddingsParse, EmptyInputArrayRejected) {
    expect_error([] { EmbeddingsRequest::from_json(R"({"model":"m","input":[]})"); },
                 HELIX_E_VALIDATION, "input");
}

TEST(EmbeddingsParse, OversizeInputArrayRejected) {
    json j = {{"model", "m"}};
    j["input"] = json::array();
    for (int i = 0; i < 2049; ++i) j["input"].push_back("t");
    const std::string body = j.dump();
    expect_error([&] { EmbeddingsRequest::from_json(body); },
                 HELIX_E_VALIDATION, "input");
}

TEST(EmbeddingsParse, NumericInputRejected) {
    expect_error([] { EmbeddingsRequest::from_json(R"({"model":"m","input":42})"); },
                 HELIX_E_VALIDATION, "input");
}

TEST(EmbeddingsParse, TokenIdArrayRejectedAsUnsupported) {
    expect_error([] {
        EmbeddingsRequest::from_json(R"({"model":"m","input":[1,2,3]})");
    }, HELIX_E_UNSUPPORTED_FEATURE, "input");
}

TEST(EmbeddingsParse, NestedTokenIdArraysRejectedAsUnsupported) {
    expect_error([] {
        EmbeddingsRequest::from_json(R"({"model":"m","input":[[1,2],[3]]})");
    }, HELIX_E_UNSUPPORTED_FEATURE, "input");
}

TEST(EmbeddingsParse, MixedArrayRejected) {
    expect_error([] {
        EmbeddingsRequest::from_json(R"({"model":"m","input":["a",false]})");
    }, HELIX_E_VALIDATION, "input");
}

TEST(EmbeddingsParse, BadEncodingFormatRejected) {
    expect_error([] {
        EmbeddingsRequest::from_json(
            R"({"model":"m","input":"x","encoding_format":"int8"})");
    }, HELIX_E_VALIDATION, "encoding_format");
}

TEST(EmbeddingsParse, DimensionsRejectedAsUnsupported) {
    expect_error([] {
        EmbeddingsRequest::from_json(
            R"({"model":"m","input":"x","dimensions":256})");
    }, HELIX_E_UNSUPPORTED_FEATURE, "dimensions");
}

/* ---- validate() ---- */

TEST(EmbeddingsParse, EmptyModelRejectedByValidate) {
    auto r = EmbeddingsRequest::from_json(R"({"model":"","input":"x"})");
    expect_error([&] { r.validate(); }, HELIX_E_VALIDATION, "model");
}

TEST(EmbeddingsParse, EmptyStringMemberRejectedWithIndex) {
    auto r = EmbeddingsRequest::from_json(
        R"({"model":"m","input":["a","b","c",""]})");
    try {
        r.validate();
        FAIL() << "expected validation error";
    } catch (const Error& e) {
        EXPECT_EQ(e.status, HELIX_E_VALIDATION);
        EXPECT_EQ(e.param, "input");
        EXPECT_NE(e.message.find("input[3]"), std::string::npos)
            << "message should name the offending index: " << e.message;
    }
}

TEST(EmbeddingsParse, EmptySingleStringRejected) {
    auto r = EmbeddingsRequest::from_json(R"({"model":"m","input":""})");
    expect_error([&] { r.validate(); }, HELIX_E_VALIDATION, "input");
}

/* ---- Serialization ---- */

static EmbeddingsResponse make_response() {
    EmbeddingsResponse resp;
    resp.model         = "embed-test";
    resp.vectors       = {{1.0f, 0.0f}, {0.0f, 1.0f}, {0.5f, -0.5f}};
    resp.prompt_tokens = 7;
    return resp;
}

TEST(EmbeddingsSerialize, ListShapeAndOrder) {
    auto j = json::parse(make_response().to_json());
    EXPECT_EQ(j["object"], "list");
    EXPECT_EQ(j["model"], "embed-test");
    ASSERT_EQ(j["data"].size(), 3u);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(j["data"][i]["object"], "embedding");
        EXPECT_EQ(j["data"][i]["index"], i);
        ASSERT_TRUE(j["data"][i]["embedding"].is_array());
        EXPECT_EQ(j["data"][i]["embedding"].size(), 2u);
    }
    EXPECT_FLOAT_EQ(j["data"][2]["embedding"][0].get<float>(), 0.5f);
    EXPECT_FLOAT_EQ(j["data"][2]["embedding"][1].get<float>(), -0.5f);
}

TEST(EmbeddingsSerialize, UsageTotalEqualsPrompt) {
    auto j = json::parse(make_response().to_json());
    EXPECT_EQ(j["usage"]["prompt_tokens"], 7);
    EXPECT_EQ(j["usage"]["total_tokens"], 7);
}

TEST(EmbeddingsSerialize, Base64RoundTrip) {
    auto resp = make_response();
    resp.encoding = EmbeddingsRequest::Encoding::Base64;
    auto j = json::parse(resp.to_json());
    ASSERT_EQ(j["data"].size(), 3u);

    for (size_t i = 0; i < resp.vectors.size(); ++i) {
        ASSERT_TRUE(j["data"][i]["embedding"].is_string());
        const std::string decoded =
            base64::decode(j["data"][i]["embedding"].get<std::string>());
        ASSERT_EQ(decoded.size(), resp.vectors[i].size() * sizeof(float));
        std::vector<float> back(resp.vectors[i].size());
        std::memcpy(back.data(), decoded.data(), decoded.size());
        for (size_t k = 0; k < back.size(); ++k) {
            /* bitwise comparison — base64 must not perturb the payload */
            EXPECT_EQ(std::memcmp(&back[k], &resp.vectors[i][k], sizeof(float)), 0)
                << "vector " << i << " element " << k;
        }
    }
}

TEST(EmbeddingsSerialize, EmptyResponseIsValidList) {
    EmbeddingsResponse resp;
    resp.model = "m";
    auto j = json::parse(resp.to_json());
    EXPECT_EQ(j["object"], "list");
    EXPECT_TRUE(j["data"].is_array());
    EXPECT_EQ(j["data"].size(), 0u);
    EXPECT_EQ(j["usage"]["prompt_tokens"], 0);
}
