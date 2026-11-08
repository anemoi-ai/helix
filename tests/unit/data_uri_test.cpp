#include <gtest/gtest.h>
#include "src/multimodal/image_decode.hpp"
#include "src/internal/error.hpp"

#include <filesystem>
#include <fstream>

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

/* ---- file:// percent-decoding ---- */

TEST(FileUri, EncodedNulRejected) {
    try {
#ifdef _WIN32
        load_file_uri("file://C:/tmp/a%00b.png");
#else
        load_file_uri("file:///tmp/a%00b.png");
#endif
        FAIL() << "expected HELIX_E_VALIDATION";
    } catch (const Error& e) {
        EXPECT_EQ(e.status, HELIX_E_VALIDATION);
        EXPECT_NE(e.message.find("NUL"), std::string::npos) << e.message;
    }
}

TEST(FileUri, EncodedTraversalRejected) {
    try {
#ifdef _WIN32
        load_file_uri("file://C:/tmp/%2e%2e/secret");
#else
        load_file_uri("file:///tmp/%2e%2e/secret");
#endif
        FAIL() << "expected HELIX_E_VALIDATION";
    } catch (const Error& e) {
        EXPECT_EQ(e.status, HELIX_E_VALIDATION);
        EXPECT_NE(e.message.find(".."), std::string::npos) << e.message;
    }
}

#ifdef _WIN32
TEST(FileUri, BackslashTraversalRejected) {
    try {
        load_file_uri("file://C:\\tmp\\..\\secret");
        FAIL() << "expected HELIX_E_VALIDATION";
    } catch (const Error& e) {
        EXPECT_EQ(e.status, HELIX_E_VALIDATION);
        EXPECT_NE(e.message.find(".."), std::string::npos) << e.message;
    }
}
#endif

/* A missing file and an existing-but-rejected file must be indistinguishable
 * to the caller (no existence oracle). */
TEST(FileUri, FilesystemFailuresShareOneMessage) {
    const auto tmp = std::filesystem::temp_directory_path() /
                     "helix_file_uri_oracle_test.bin";
    {
        std::ofstream f(tmp, std::ios::binary);
        f << "0123456789";
    }
    const std::string existing = "file://" + tmp.generic_u8string();
    const std::string missing = existing + ".does-not-exist";

    std::string msg_missing, msg_too_big;
    try { load_file_uri(missing); FAIL() << "expected an error"; }
    catch (const Error& e) { msg_missing = e.message; }
    try { load_file_uri(existing, /*max_bytes=*/1); FAIL() << "expected an error"; }
    catch (const Error& e) { msg_too_big = e.message; }

    std::error_code ec;
    std::filesystem::remove(tmp, ec);

    EXPECT_FALSE(msg_missing.empty());
    EXPECT_EQ(msg_missing, msg_too_big);
    EXPECT_EQ(msg_missing.find("helix_file_uri_oracle_test"), std::string::npos)
        << "message must not echo the resolved path: " << msg_missing;
}
