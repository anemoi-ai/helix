#include <gtest/gtest.h>
#include "src/engine/embed_packing.hpp"

#include <numeric>

using helix::plan_embed_flushes;

/* Every plan must partition 0..n-1 in order — concatenating the groups
 * reproduces the input sequence exactly. */
static void expect_ordered_partition(
        const std::vector<std::vector<size_t>>& flushes, size_t n) {
    std::vector<size_t> flat;
    for (const auto& g : flushes) {
        EXPECT_FALSE(g.empty());
        flat.insert(flat.end(), g.begin(), g.end());
    }
    std::vector<size_t> expected(n);
    std::iota(expected.begin(), expected.end(), 0u);
    EXPECT_EQ(flat, expected);
}

TEST(EmbedPacking, EmptyInputNoFlushes) {
    EXPECT_TRUE(plan_embed_flushes({}, 512, 8).empty());
}

TEST(EmbedPacking, SingleInputSingleFlush) {
    auto f = plan_embed_flushes({10}, 512, 8);
    ASSERT_EQ(f.size(), 1u);
    EXPECT_EQ(f[0], std::vector<size_t>{0});
}

TEST(EmbedPacking, PacksUpToTokenBudget) {
    /* 100+200+212 = 512 fits exactly; the next input flushes. */
    auto f = plan_embed_flushes({100, 200, 212, 1}, 512, 8);
    ASSERT_EQ(f.size(), 2u);
    EXPECT_EQ(f[0], (std::vector<size_t>{0, 1, 2}));
    EXPECT_EQ(f[1], std::vector<size_t>{3});
    expect_ordered_partition(f, 4);
}

TEST(EmbedPacking, FlushTriggersAtTokenOverflow) {
    auto f = plan_embed_flushes({300, 300}, 512, 8);
    ASSERT_EQ(f.size(), 2u);
    expect_ordered_partition(f, 2);
}

TEST(EmbedPacking, FlushTriggersAtSeqMax) {
    /* 5 tiny inputs, n_seq_max 2 → groups of 2,2,1. */
    auto f = plan_embed_flushes({1, 1, 1, 1, 1}, 512, 2);
    ASSERT_EQ(f.size(), 3u);
    EXPECT_EQ(f[0].size(), 2u);
    EXPECT_EQ(f[1].size(), 2u);
    EXPECT_EQ(f[2].size(), 1u);
    expect_ordered_partition(f, 5);
}

TEST(EmbedPacking, FullBatchInputAlwaysAlone) {
    auto f = plan_embed_flushes({512, 512, 5}, 512, 8);
    ASSERT_EQ(f.size(), 3u);
    EXPECT_EQ(f[0], std::vector<size_t>{0});
    EXPECT_EQ(f[1], std::vector<size_t>{1});
    EXPECT_EQ(f[2], std::vector<size_t>{2});
}

TEST(EmbedPacking, OrderPreservedAcrossManyFlushes) {
    /* Mixed lengths across several flush boundaries. */
    std::vector<size_t> lens;
    for (int i = 0; i < 100; ++i) lens.push_back(1 + (i * 37) % 200);
    auto f = plan_embed_flushes(lens, 256, 4);
    expect_ordered_partition(f, lens.size());
    /* No group exceeds either budget. */
    for (const auto& g : f) {
        EXPECT_LE(g.size(), 4u);
        size_t tokens = 0;
        for (size_t idx : g) tokens += lens[idx];
        EXPECT_LE(tokens, 256u);
    }
}
