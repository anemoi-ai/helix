#include <gtest/gtest.h>
#include "src/multimodal/image_decode.hpp"
#include "src/internal/error.hpp"

using namespace helix;

/* Minimal 1×1 transparent PNG, base64-encoded. */
static const std::string kTinyPngB64 =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

TEST(DataUri, DecodePng) {
    auto bytes = decode_image_data_uri("data:image/png;base64," + kTinyPngB64);
    EXPECT_FALSE(bytes.empty());
    /* PNG magic: \x89PNG */
    ASSERT_GE(bytes.size(), 4u);
    EXPECT_EQ(bytes[0], 0x89u);
    EXPECT_EQ(bytes[1], 0x50u); /* 'P' */
    EXPECT_EQ(bytes[2], 0x4Eu); /* 'N' */
    EXPECT_EQ(bytes[3], 0x47u); /* 'G' */
}

TEST(DataUri, JpegMimePrefixAccepted) {
    auto bytes = decode_image_data_uri("data:image/jpeg;base64," + kTinyPngB64);
    EXPECT_FALSE(bytes.empty());
}

TEST(DataUri, WebpMimePrefixAccepted) {
    auto bytes = decode_image_data_uri("data:image/webp;base64," + kTinyPngB64);
    EXPECT_FALSE(bytes.empty());
}

TEST(DataUri, UnsupportedMimeRejected) {
    EXPECT_THROW(decode_image_data_uri("data:image/tiff;base64,abc"), helix::Error);
}

TEST(DataUri, NonDataUriRejected) {
    EXPECT_THROW(decode_image_data_uri("https://example.com/img.png"), helix::Error);
}

TEST(DataUri, MissingBase64MarkerRejected) {
    EXPECT_THROW(decode_image_data_uri("data:image/png,notbase64"), helix::Error);
}

TEST(DataUri, EmptyDataRejected) {
    /* No actual base64 payload after ";base64," */
    /* An empty payload decodes to empty bytes — validate as empty. */
    EXPECT_THROW(decode_image_data_uri("data:image/png;base64,"), helix::Error);
}

TEST(AudioDecode, DecodeBase64Mp3) {
    /* A minimal MP3 header — just enough to pass base64 decode. */
    /* We only test that the function accepts "mp3" format without throwing on valid b64. */
    /* "AAAA" decodes to 3 zero bytes — not a valid MP3 but tests the decode path. */
    /* We don't expect a parse error from audio format validation (that's mtmd's job). */
    auto bytes = decode_audio_base64("AAAA", "mp3");
    ASSERT_EQ(bytes.size(), 3u);
}

TEST(AudioDecode, UnsupportedFormatRejected) {
    EXPECT_THROW(decode_audio_base64("AAAA", "ogg"), helix::Error);
}

TEST(AudioDecode, EmptyDataRejected) {
    EXPECT_THROW(decode_audio_base64("", "mp3"), helix::Error);
}
