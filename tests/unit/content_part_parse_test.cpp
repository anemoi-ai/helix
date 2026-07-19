#include <gtest/gtest.h>
#include "src/json/request.hpp"
#include "src/internal/error.hpp"

using namespace helix;

/* Helper: parse a full chat request JSON string. */
static ChatRequest parse(const std::string& json) {
    return ChatRequest::from_json(json);
}

/* ---- text-only content (existing behaviour) ---- */

TEST(ContentPartParse, StringContentStillWorks) {
    auto r = parse(R"({"model":"m","messages":[{"role":"user","content":"hello"}]})");
    ASSERT_EQ(r.messages.size(), 1u);
    EXPECT_EQ(r.messages[0].content_str, "hello");
    EXPECT_TRUE(r.messages[0].content_parts.empty());
}

/* ---- array content ---- */

TEST(ContentPartParse, ArrayWithTextPart) {
    auto r = parse(R"({
        "model":"m",
        "messages":[{"role":"user","content":[
            {"type":"text","text":"What is this?"}
        ]}]
    })");
    ASSERT_EQ(r.messages.size(), 1u);
    const auto& m = r.messages[0];
    ASSERT_EQ(m.content_parts.size(), 1u);
    EXPECT_EQ(m.content_parts[0].type, "text");
    EXPECT_EQ(m.content_parts[0].text, "What is this?");
}

TEST(ContentPartParse, ArrayWithImageUrlDataUri) {
    /* Minimal valid PNG base64 (1×1 transparent pixel). */
    const std::string tiny_png_b64 =
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
    std::string json = R"({"model":"m","messages":[{"role":"user","content":[
        {"type":"image_url","image_url":{"url":"data:image/png;base64,)" +
        tiny_png_b64 + R"(","detail":"auto"}}
    ]}]})";
    auto r = parse(json);
    ASSERT_EQ(r.messages[0].content_parts.size(), 1u);
    const auto& p = r.messages[0].content_parts[0];
    EXPECT_EQ(p.type, "image_url");
    EXPECT_EQ(p.image_detail, "auto");
    EXPECT_FALSE(p.media_raw.empty());
}

TEST(ContentPartParse, ArrayMixedTextAndImage) {
    const std::string tiny_png_b64 =
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
    std::string json = R"({"model":"m","messages":[{"role":"user","content":[
        {"type":"text","text":"Describe:"},
        {"type":"image_url","image_url":{"url":"data:image/png;base64,)" +
        tiny_png_b64 + R"("}}
    ]}]})";
    auto r = parse(json);
    const auto& parts = r.messages[0].content_parts;
    ASSERT_EQ(parts.size(), 2u);
    EXPECT_EQ(parts[0].type, "text");
    EXPECT_EQ(parts[1].type, "image_url");
}

/* ---- rejection cases ---- */

TEST(ContentPartParse, HttpUrlRejected) {
    std::string json = R"({"model":"m","messages":[{"role":"user","content":[
        {"type":"image_url","image_url":{"url":"http://example.com/img.png"}}
    ]}]})";
    EXPECT_THROW(parse(json), helix::Error);
}

TEST(ContentPartParse, HttpsUrlRejected) {
    std::string json = R"({"model":"m","messages":[{"role":"user","content":[
        {"type":"image_url","image_url":{"url":"https://example.com/img.png"}}
    ]}]})";
    EXPECT_THROW(parse(json), helix::Error);
}

TEST(ContentPartParse, UnknownPartTypeRejected) {
    std::string json = R"({"model":"m","messages":[{"role":"user","content":[
        {"type":"video_url","video_url":{"url":"data:video/mp4;base64,abc"}}
    ]}]})";
    EXPECT_THROW(parse(json), helix::Error);
}

TEST(ContentPartParse, AssistantTextArrayAccepted) {
    /* OpenAI allows assistant content as an array of text parts. */
    std::string json = R"({"model":"m","messages":[{"role":"assistant","content":[
        {"type":"text","text":"hi"}
    ]}]})";
    auto r = parse(json);
    ASSERT_EQ(r.messages[0].content_parts.size(), 1u);
    EXPECT_EQ(r.messages[0].content_parts[0].type, "text");
    EXPECT_EQ(r.messages[0].content_parts[0].text, "hi");
}

TEST(ContentPartParse, AssistantMediaArrayRejected) {
    /* Media parts are only valid on user turns. */
    std::string json = R"({"model":"m","messages":[{"role":"assistant","content":[
        {"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}}
    ]}]})";
    EXPECT_THROW(parse(json), helix::Error);
}

TEST(ContentPartParse, ToolArrayContentRejected) {
    std::string json = R"({"model":"m","messages":[
        {"role":"tool","tool_call_id":"x","content":[{"type":"text","text":"hi"}]}
    ]})";
    EXPECT_THROW(parse(json), helix::Error);
}

/* ---- input_audio ---- */

TEST(ContentPartParse, InputAudioWithValidBase64) {
    std::string json = R"({"model":"m","messages":[{"role":"user","content":[
        {"type":"input_audio","input_audio":{"format":"mp3","data":"AAAA"}}
    ]}]})";
    auto r = parse(json);
    ASSERT_EQ(r.messages[0].content_parts.size(), 1u);
    const auto& p = r.messages[0].content_parts[0];
    EXPECT_EQ(p.type, "input_audio");
    EXPECT_EQ(p.audio_format, "mp3");
    EXPECT_FALSE(p.media_raw.empty());
}

TEST(ContentPartParse, InputAudioWavFormat) {
    std::string json = R"({"model":"m","messages":[{"role":"user","content":[
        {"type":"input_audio","input_audio":{"format":"wav","data":"AAAA"}}
    ]}]})";
    auto r = parse(json);
    ASSERT_EQ(r.messages[0].content_parts.size(), 1u);
    EXPECT_EQ(r.messages[0].content_parts[0].audio_format, "wav");
}

TEST(ContentPartParse, InputAudioMissingObjectRejected) {
    std::string json = R"({"model":"m","messages":[{"role":"user","content":[
        {"type":"input_audio"}
    ]}]})";
    EXPECT_THROW(parse(json), helix::Error);
}

TEST(ContentPartParse, InputAudioInvalidFormatRejected) {
    std::string json = R"({"model":"m","messages":[{"role":"user","content":[
        {"type":"input_audio","input_audio":{"format":"ogg","data":"AAAA"}}
    ]}]})";
    EXPECT_THROW(parse(json), helix::Error);
}

TEST(ContentPartParse, InputAudioEmptyDataRejected) {
    std::string json = R"({"model":"m","messages":[{"role":"user","content":[
        {"type":"input_audio","input_audio":{"format":"mp3","data":""}}
    ]}]})";
    EXPECT_THROW(parse(json), helix::Error);
}

/* ---- empty content array / empty text part ---- */

TEST(ContentPartParse, EmptyContentArrayRejected) {
    /* Rejected at parse time: an empty array is a malformed request. */
    std::string json = R"({"model":"m","messages":[{"role":"user","content":[]}]})";
    EXPECT_THROW(parse(json), helix::Error);
}

TEST(ContentPartParse, EmptyTextPartRejected) {
    std::string json = R"({"model":"m","messages":[{"role":"user","content":[
        {"type":"text","text":""}
    ]}]})";
    auto r = parse(json);
    EXPECT_THROW(r.validate(), helix::Error);
}

TEST(ContentPartParse, NonEmptyTextPartPasses) {
    std::string json = R"({"model":"m","messages":[{"role":"user","content":[
        {"type":"text","text":"hello"}
    ]}]})";
    auto r = parse(json);
    EXPECT_NO_THROW(r.validate());
}

/* ---- file:// URL scheme ---- */

TEST(ContentPartParse, FileUriNonExistentPathRejected) {
    std::string json = R"({"model":"m","messages":[{"role":"user","content":[
        {"type":"image_url","image_url":{"url":"file:///nonexistent/path/image.png"}}
    ]}]})";
    EXPECT_THROW(parse(json), helix::Error);
}

/* ---- invalid base64 ---- */

TEST(ContentPartParse, InvalidBase64InDataUriRejected) {
    std::string json = R"({"model":"m","messages":[{"role":"user","content":[
        {"type":"image_url","image_url":{"url":"data:image/png;base64,!!!invalid!!!"}}
    ]}]})";
    EXPECT_THROW(parse(json), helix::Error);
}

/* ---- oversized media payload ---- */

TEST(ContentPartParse, InvalidImageDetailRejected) {
    std::string json = R"({"model":"m","messages":[{"role":"user","content":[
        {"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo=","detail":"ultra"}}
    ]}]})";
    EXPECT_THROW(parse(json), helix::Error);
}

TEST(ContentPartParse, ImageDetailLowAccepted) {
    const std::string tiny_png_b64 =
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
    std::string json = R"({"model":"m","messages":[{"role":"user","content":[
        {"type":"image_url","image_url":{"url":"data:image/png;base64,)" +
        tiny_png_b64 + R"(","detail":"low"}}
    ]}]})";
    auto r = parse(json);
    EXPECT_EQ(r.messages[0].content_parts[0].image_detail, "low");
}

TEST(ContentPartParse, ImageDetailHighAccepted) {
    const std::string tiny_png_b64 =
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
    std::string json = R"({"model":"m","messages":[{"role":"user","content":[
        {"type":"image_url","image_url":{"url":"data:image/png;base64,)" +
        tiny_png_b64 + R"(","detail":"high"}}
    ]}]})";
    auto r = parse(json);
    EXPECT_EQ(r.messages[0].content_parts[0].image_detail, "high");
}
