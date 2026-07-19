#include <gtest/gtest.h>
#include "engine/utf8.hpp"

using helix::valid_utf8_prefix_len;

/* ASCII: entire string is a valid prefix. */
TEST(Utf8Prefix, AllAscii) {
    const char* s = "hello";
    EXPECT_EQ(valid_utf8_prefix_len(s, 5), 5u);
}

TEST(Utf8Prefix, Empty) {
    EXPECT_EQ(valid_utf8_prefix_len("", 0), 0u);
}

/* 2-byte sequence: "é" = 0xC3 0xA9. */
TEST(Utf8Prefix, TwoByteComplete) {
    const char s[] = {'\xC3', '\xA9', 'x'};
    EXPECT_EQ(valid_utf8_prefix_len(s, 3), 3u);
}

/* Truncated 2-byte sequence: only the lead byte present. */
TEST(Utf8Prefix, TwoByteTruncated) {
    const char s[] = {'a', '\xC3'};           /* 'a' + lead of é */
    EXPECT_EQ(valid_utf8_prefix_len(s, 2), 1u);
}

/* 3-byte sequence: "€" = 0xE2 0x82 0xAC. */
TEST(Utf8Prefix, ThreeByteComplete) {
    const char s[] = {'\xE2', '\x82', '\xAC'};
    EXPECT_EQ(valid_utf8_prefix_len(s, 3), 3u);
}

/* Truncated 3-byte: lead + one continuation. */
TEST(Utf8Prefix, ThreeByteTruncatedTwo) {
    const char s[] = {'x', '\xE2', '\x82'};
    EXPECT_EQ(valid_utf8_prefix_len(s, 3), 1u);
}

/* 4-byte sequence: U+1F600 = 0xF0 0x9F 0x98 0x80. */
TEST(Utf8Prefix, FourByteComplete) {
    const char s[] = {'\xF0', '\x9F', '\x98', '\x80'};
    EXPECT_EQ(valid_utf8_prefix_len(s, 4), 4u);
}

/* Truncated 4-byte: three bytes of a 4-byte sequence. */
TEST(Utf8Prefix, FourByteTruncatedThree) {
    const char s[] = {'\xF0', '\x9F', '\x98'};
    EXPECT_EQ(valid_utf8_prefix_len(s, 3), 0u);
}

/* ASCII followed by a truncated multi-byte. */
TEST(Utf8Prefix, AsciiThenTruncated) {
    const char s[] = {'a', 'b', '\xE2', '\x82'};  /* "ab" + 2/3 of € */
    EXPECT_EQ(valid_utf8_prefix_len(s, 4), 2u);
}

/* Safe prefix ends at last complete char even with trailing continuation. */
TEST(Utf8Prefix, CompleteCharsThenTruncated) {
    const char s[] = {'\xC3', '\xA9',   /* é (complete) */
                      '\xE2', '\x82'};  /* truncated € */
    EXPECT_EQ(valid_utf8_prefix_len(s, 4), 2u);
}

/* Single byte: ASCII. */
TEST(Utf8Prefix, SingleByte) {
    const char s[] = {'Z'};
    EXPECT_EQ(valid_utf8_prefix_len(s, 1), 1u);
}

/* Single lead byte only (incomplete). */
TEST(Utf8Prefix, SingleLeadOnly) {
    const char s[] = {'\xC3'};
    EXPECT_EQ(valid_utf8_prefix_len(s, 1), 0u);
}
