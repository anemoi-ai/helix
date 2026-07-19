#include <gtest/gtest.h>
#include "engine/lcp.hpp"
#include <vector>

using helix::compute_lcp;

TEST(Lcp, BothEmpty) {
    EXPECT_EQ(compute_lcp(std::vector<int>{}, std::vector<int>{}), 0u);
}

TEST(Lcp, OneEmpty) {
    EXPECT_EQ(compute_lcp(std::vector<int>{1, 2, 3}, std::vector<int>{}), 0u);
    EXPECT_EQ(compute_lcp(std::vector<int>{}, std::vector<int>{1, 2, 3}), 0u);
}

TEST(Lcp, Identical) {
    EXPECT_EQ(compute_lcp(std::vector<int>{1, 2, 3}, std::vector<int>{1, 2, 3}), 3u);
}

TEST(Lcp, CompletelyDifferent) {
    EXPECT_EQ(compute_lcp(std::vector<int>{1, 2, 3}, std::vector<int>{4, 5, 6}), 0u);
}

TEST(Lcp, FirstIsPrefixOfSecond) {
    EXPECT_EQ(compute_lcp(std::vector<int>{1, 2}, std::vector<int>{1, 2, 3, 4}), 2u);
}

TEST(Lcp, SecondIsPrefixOfFirst) {
    EXPECT_EQ(compute_lcp(std::vector<int>{1, 2, 3, 4}, std::vector<int>{1, 2}), 2u);
}

TEST(Lcp, DivergeMidway) {
    EXPECT_EQ(compute_lcp(std::vector<int>{1, 2, 3, 9}, std::vector<int>{1, 2, 3, 4}), 3u);
}

TEST(Lcp, OffByOne_DifferAtLastPosition) {
    EXPECT_EQ(compute_lcp(std::vector<int>{1, 2, 3}, std::vector<int>{1, 2, 4}), 2u);
}

TEST(Lcp, OffByOne_DifferAtFirst) {
    EXPECT_EQ(compute_lcp(std::vector<int>{2, 2, 3}, std::vector<int>{1, 2, 3}), 0u);
}

TEST(Lcp, SingleElementMatch) {
    EXPECT_EQ(compute_lcp(std::vector<int>{42}, std::vector<int>{42}), 1u);
}

TEST(Lcp, SingleElementNoMatch) {
    EXPECT_EQ(compute_lcp(std::vector<int>{42}, std::vector<int>{43}), 0u);
}
