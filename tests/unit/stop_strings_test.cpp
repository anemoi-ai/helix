#include <gtest/gtest.h>
#include "engine/utf8.hpp"
#include "engine/stop_strings.hpp"

using helix::valid_utf8_prefix_len;
using helix::check_stops;
using helix::partial_stop_len;

TEST(StopStrings, ASCIIStop) {
    std::string out = "Hello world\n\nNext";
    size_t safe = valid_utf8_prefix_len(out.data(), out.size());
    auto pos = check_stops(out, safe, {"\n\n"});
    ASSERT_NE(pos, std::string::npos);
    out.resize(pos);
    EXPECT_EQ(out, "Hello world");
}

TEST(StopStrings, MultipleStopsFirstMatch) {
    std::string out = "done END here";
    size_t safe = valid_utf8_prefix_len(out.data(), out.size());
    auto pos = check_stops(out, safe, {"END", "here"});
    ASSERT_NE(pos, std::string::npos);
    EXPECT_EQ(out.substr(0, pos), "done ");
}

TEST(StopStrings, NoMatchReturnsNpos) {
    std::string out = "hello world";
    size_t safe = valid_utf8_prefix_len(out.data(), out.size());
    EXPECT_EQ(check_stops(out, safe, {"STOP"}), std::string::npos);
}

TEST(StopStrings, UTF8FullSequenceIsValid) {
    const char jp[] = "\xE6\x97\xA5";
    EXPECT_EQ(valid_utf8_prefix_len(jp, 3), 3u);
}

TEST(StopStrings, IncompleteUTF8ExcludedFromSafe) {
    const char partial[] = "\xE6\x97";
    size_t safe = valid_utf8_prefix_len(partial, 2);
    EXPECT_LT(safe, 2u);
}

TEST(StopStrings, EmptyStopIgnored) {
    std::string out = "hello";
    size_t safe = valid_utf8_prefix_len(out.data(), out.size());
    EXPECT_EQ(check_stops(out, safe, {""}), std::string::npos);
}

TEST(StopStrings, StopAtBeginning) {
    std::string out = "ENDsometext";
    size_t safe = valid_utf8_prefix_len(out.data(), out.size());
    auto pos = check_stops(out, safe, {"END"});
    ASSERT_NE(pos, std::string::npos);
    EXPECT_EQ(pos, 0u);
}

TEST(StopStrings, ValidUTF8PrefixAsciiOnly) {
    std::string s = "hello";
    EXPECT_EQ(valid_utf8_prefix_len(s.data(), s.size()), 5u);
}

TEST(StopStrings, EmptyString) {
    EXPECT_EQ(valid_utf8_prefix_len("", 0), 0u);
}

/* A single token piece can complete two occurrences at once; truncation must
 * happen at the FIRST one, not the last. */
TEST(StopStrings, FirstOccurrenceWins) {
    std::string out = "|yes|";
    size_t safe = valid_utf8_prefix_len(out.data(), out.size());
    auto pos = check_stops(out, safe, {"|"});
    ASSERT_NE(pos, std::string::npos);
    EXPECT_EQ(pos, 0u);
}

TEST(StopStrings, FirstOccurrenceWinsMidString) {
    std::string out = "a|yes|b";
    size_t safe = valid_utf8_prefix_len(out.data(), out.size());
    auto pos = check_stops(out, safe, {"|"});
    ASSERT_NE(pos, std::string::npos);
    EXPECT_EQ(out.substr(0, pos), "a");
}

/* Incremental scanning: per-token calls with a caller-maintained offset must
 * find a stop that spans a token boundary. */
TEST(StopStrings, IncrementalScanAcrossTokens) {
    const std::vector<std::string> stops = {"END"};
    std::string out;
    size_t scan = 0;
    out += "Hello EN";
    EXPECT_EQ(check_stops(out, out.size(), stops, &scan), std::string::npos);
    out += "D!";
    auto pos = check_stops(out, out.size(), stops, &scan);
    ASSERT_NE(pos, std::string::npos);
    EXPECT_EQ(out.substr(0, pos), "Hello ");
}

TEST(StopStrings, IncrementalScanDoesNotSkipPending) {
    const std::vector<std::string> stops = {"\n\nUser:", "END"};
    std::string out;
    size_t scan = 0;
    out += "text\n\nUse";
    EXPECT_EQ(check_stops(out, out.size(), stops, &scan), std::string::npos);
    out += "r";
    EXPECT_EQ(check_stops(out, out.size(), stops, &scan), std::string::npos);
    out += ":";
    auto pos = check_stops(out, out.size(), stops, &scan);
    ASSERT_NE(pos, std::string::npos);
    EXPECT_EQ(out.substr(0, pos), "text");
}

/* Holdback: a suffix that is a prefix of a stop string must not be emitted. */
TEST(StopStrings, PartialStopSuffixHeldBack) {
    std::string out = "Hello EN";
    EXPECT_EQ(partial_stop_len(out, out.size(), {"END"}), 2u);
}

TEST(StopStrings, NoPartialSuffixNothingHeld) {
    std::string out = "Hello ENQ";
    EXPECT_EQ(partial_stop_len(out, out.size(), {"END"}), 0u);
}

TEST(StopStrings, PartialSuffixLongestAcrossStops) {
    std::string out = "abc\n\nUse";
    EXPECT_EQ(partial_stop_len(out, out.size(), {"END", "\n\nUser:"}), 5u);
}

TEST(StopStrings, PartialSuffixCappedByStopLength) {
    /* A full match is check_stops' job; partial_stop_len only ever reports
     * proper prefixes (at most stop.size() - 1 bytes). */
    std::string out = "END";
    EXPECT_EQ(partial_stop_len(out, out.size(), {"END"}), 0u);
}

TEST(StopStrings, PartialSuffixEmptyStops) {
    std::string out = "hello";
    EXPECT_EQ(partial_stop_len(out, out.size(), {}), 0u);
    EXPECT_EQ(partial_stop_len(out, out.size(), {""}), 0u);
}
