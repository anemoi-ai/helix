#include <gtest/gtest.h>
#include "src/sampling/logit_bias.hpp"
#include "src/internal/error.hpp"

using namespace helix;

TEST(LogitBias, NullVocabReturnsEmpty) {
    std::map<std::string, float> raw = {{"42", -1.0f}};
    auto result = resolve_logit_bias(raw, nullptr);
    EXPECT_TRUE(result.empty());
}

TEST(LogitBias, EmptyMapReturnsEmpty) {
    auto result = resolve_logit_bias({}, nullptr);
    EXPECT_TRUE(result.empty());
}

TEST(LogitBias, NullVocabAndEmptyMapReturnsEmpty) {
    auto result = resolve_logit_bias({}, nullptr);
    EXPECT_TRUE(result.empty());
}

TEST(LogitBias, ResolvedBiasFields) {
    ResolvedBias rb{42, -5.0f};
    EXPECT_EQ(rb.token, 42);
    EXPECT_FLOAT_EQ(rb.bias, -5.0f);
}

TEST(LogitBias, MultipleKeysNullVocab) {
    std::map<std::string, float> raw = {{"42", -1.0f}, {"100", 2.0f}, {"hello", 0.5f}};
    auto result = resolve_logit_bias(raw, nullptr);
    EXPECT_TRUE(result.empty());
}

TEST(LogitBias, ZeroBiasNullVocab) {
    std::map<std::string, float> raw = {{"42", 0.0f}};
    auto result = resolve_logit_bias(raw, nullptr);
    EXPECT_TRUE(result.empty());
}

TEST(LogitBias, NegativeTokenIdKeyNullVocab) {
    std::map<std::string, float> raw = {{"-1", 5.0f}};
    auto result = resolve_logit_bias(raw, nullptr);
    EXPECT_TRUE(result.empty());
}

TEST(LogitBias, StringKeyNullVocab) {
    std::map<std::string, float> raw = {{"hello", 1.0f}};
    auto result = resolve_logit_bias(raw, nullptr);
    EXPECT_TRUE(result.empty());
}

TEST(LogitBias, LargeIntegerKeyNullVocab) {
    std::map<std::string, float> raw = {{"999999", -10.0f}};
    auto result = resolve_logit_bias(raw, nullptr);
    EXPECT_TRUE(result.empty());
}

TEST(LogitBias, ResolvedBiasZeroBias) {
    ResolvedBias rb{0, 0.0f};
    EXPECT_EQ(rb.token, 0);
    EXPECT_FLOAT_EQ(rb.bias, 0.0f);
}

TEST(LogitBias, ResolvedBiasMaxPositiveBias) {
    ResolvedBias rb{100, 100.0f};
    EXPECT_EQ(rb.token, 100);
    EXPECT_FLOAT_EQ(rb.bias, 100.0f);
}

TEST(LogitBias, ResolvedBiasMaxNegativeBias) {
    ResolvedBias rb{200, -100.0f};
    EXPECT_EQ(rb.token, 200);
    EXPECT_FLOAT_EQ(rb.bias, -100.0f);
}
