#include <gtest/gtest.h>
#include "hardware/cpu.hpp"
#include "hardware/profile.hpp"

// ── ISA tier names ──────────────────────────────────────────────────────────

TEST(IsaTier, Names) {
    using namespace helix;
    EXPECT_STREQ(isa_tier_name(IsaTier::Generic),    "generic");
    EXPECT_STREQ(isa_tier_name(IsaTier::Avx2),       "avx2");
    EXPECT_STREQ(isa_tier_name(IsaTier::Avx512),     "avx512");
    EXPECT_STREQ(isa_tier_name(IsaTier::Amx),        "amx");
    EXPECT_STREQ(isa_tier_name(IsaTier::Neon),       "neon");
    EXPECT_STREQ(isa_tier_name(IsaTier::NeonFp16),   "neon_fp16");
    EXPECT_STREQ(isa_tier_name(IsaTier::NeonBf16),   "neon_bf16");
    EXPECT_STREQ(isa_tier_name(IsaTier::Sve),        "sve");
    EXPECT_STREQ(isa_tier_name(IsaTier::Sve2I8mm),   "sve2_i8mm");
    EXPECT_STREQ(isa_tier_name(IsaTier::AppleM1Plus),"apple_m1plus");
}

TEST(MemBwTier, Names) {
    using namespace helix;
    EXPECT_STREQ(mem_bw_tier_name(MemBandwidthTier::Slow),      "slow");
    EXPECT_STREQ(mem_bw_tier_name(MemBandwidthTier::DDR4),      "ddr4");
    EXPECT_STREQ(mem_bw_tier_name(MemBandwidthTier::DDR5),      "ddr5");
    EXPECT_STREQ(mem_bw_tier_name(MemBandwidthTier::LPDDR5x),   "lpddr5x");
    EXPECT_STREQ(mem_bw_tier_name(MemBandwidthTier::OnPackage), "on_package");
}

// ── Memory bandwidth table ──────────────────────────────────────────────────

// Forward declaration from mem_bw_table.cpp
namespace helix { MemBandwidthTier infer_mem_bw_tier(const std::string& model_name); }

TEST(MemBwTable, AppleSilicon) {
    using namespace helix;
    EXPECT_EQ(infer_mem_bw_tier("Apple M1"),         MemBandwidthTier::OnPackage);
    EXPECT_EQ(infer_mem_bw_tier("Apple M2 Pro"),     MemBandwidthTier::OnPackage);
    EXPECT_EQ(infer_mem_bw_tier("Apple M3 Ultra"),   MemBandwidthTier::OnPackage);
    EXPECT_EQ(infer_mem_bw_tier("Apple M4 Max"),     MemBandwidthTier::OnPackage);
}

TEST(MemBwTable, SnapdragonX) {
    using namespace helix;
    EXPECT_EQ(infer_mem_bw_tier("Snapdragon X Elite"),  MemBandwidthTier::LPDDR5x);
    EXPECT_EQ(infer_mem_bw_tier("Snapdragon X Plus"),   MemBandwidthTier::LPDDR5x);
}

TEST(MemBwTable, IntelGeneration) {
    using namespace helix;
    EXPECT_EQ(infer_mem_bw_tier("13th Gen Intel(R) Core(TM) i7-13700K"),  MemBandwidthTier::DDR5);
    EXPECT_EQ(infer_mem_bw_tier("12th Gen Intel(R) Core(TM) i9-12900K"),  MemBandwidthTier::DDR5);
    EXPECT_EQ(infer_mem_bw_tier("11th Gen Intel(R) Core(TM) i7-11700K"),  MemBandwidthTier::DDR4);
    EXPECT_EQ(infer_mem_bw_tier("Intel(R) Core Ultra 9 185H"),             MemBandwidthTier::DDR5);
}

TEST(MemBwTable, AmdRyzen) {
    using namespace helix;
    EXPECT_EQ(infer_mem_bw_tier("AMD Ryzen 9 7950X"),   MemBandwidthTier::DDR5);
    EXPECT_EQ(infer_mem_bw_tier("AMD Ryzen 9 5950X"),   MemBandwidthTier::DDR4);
    EXPECT_EQ(infer_mem_bw_tier("AMD Ryzen 7 5800X"),   MemBandwidthTier::DDR4);
    EXPECT_EQ(infer_mem_bw_tier("AMD Ryzen 5 7600X"),   MemBandwidthTier::DDR5);
}

TEST(MemBwTable, UnknownFallbackToDDR4) {
    using namespace helix;
    EXPECT_EQ(infer_mem_bw_tier("QEMU Virtual CPU version 2.5+"), MemBandwidthTier::DDR4);
    EXPECT_EQ(infer_mem_bw_tier(""),                              MemBandwidthTier::DDR4);
}

// ── probe_hardware smoke test ────────────────────────────────────────────────

TEST(ProbeHardware, Sanity) {
    using namespace helix;
    HardwareProfile hw = probe_hardware();

    EXPECT_GE(hw.cpu.logical_cores,     1u);
    EXPECT_GE(hw.cpu.physical_cores,    1u);
    EXPECT_GE(hw.cpu.performance_cores, 1u);
    EXPECT_LE(hw.cpu.performance_cores, hw.cpu.logical_cores);
    EXPECT_LE(hw.cpu.physical_cores,    hw.cpu.logical_cores);

    // ISA tier must be a valid enum value.
    const char* tier = isa_tier_name(hw.cpu.isa_tier);
    EXPECT_NE(tier, nullptr);
    EXPECT_STRNE(tier, "");

    // RAM: total must be positive.
    EXPECT_GT(hw.ram.total_bytes, 0u);
    EXPECT_LE(hw.ram.available_bytes, hw.ram.total_bytes + (1ULL << 30)); // allow small drift
}

TEST(ProbeHardware, ThreadPickIsReasonable) {
    using namespace helix;
    HardwareProfile hw = probe_hardware();

    uint32_t n  = pick_n_threads(hw);
    uint32_t nb = pick_n_threads_batch(hw);

    EXPECT_GE(n,  1u);
    EXPECT_GE(nb, 1u);
    EXPECT_LE(n,  hw.cpu.logical_cores);
    EXPECT_LE(nb, hw.cpu.logical_cores);
}
