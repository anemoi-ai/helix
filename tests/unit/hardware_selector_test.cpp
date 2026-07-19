#include <gtest/gtest.h>
#include "hardware/cpu.hpp"
#include "hardware/profile.hpp"
#include "ggml.h"

// Helpers to build canned HardwareProfile values.
static helix::HardwareProfile make_profile(
    uint32_t logical, uint32_t physical, uint32_t perf, uint32_t eff,
    helix::IsaTier isa, uint64_t total_ram_gib = 16, uint64_t avail_ram_gib = 14)
{
    using namespace helix;
    HardwareProfile hw;
    hw.cpu.logical_cores     = logical;
    hw.cpu.physical_cores    = physical;
    hw.cpu.performance_cores = perf;
    hw.cpu.efficiency_cores  = eff;
    hw.cpu.isa_tier          = isa;
    hw.ram.total_bytes     = total_ram_gib  * 1024ULL * 1024 * 1024;
    hw.ram.available_bytes = avail_ram_gib  * 1024ULL * 1024 * 1024;
    return hw;
}

// ── pick_n_threads ───────────────────────────────────────────────────────────

TEST(PickNThreads, HomogeneousX86_8Physical) {
    using namespace helix;
    // 8 physical, 16 logical (HT), no E-cores.
    auto hw = make_profile(16, 8, 8, 0, IsaTier::Avx2);
    EXPECT_EQ(pick_n_threads(hw), 8u);
}

TEST(PickNThreads, HybridX86_8P_8E) {
    using namespace helix;
    // i7-13700K style: 8P + 8E = 16 physical, 24 logical.
    auto hw = make_profile(24, 16, 8, 8, IsaTier::Avx512);
    EXPECT_EQ(pick_n_threads(hw), 8u); // P-cores only
}

TEST(PickNThreads, AppleSilicon_8P_2E) {
    using namespace helix;
    auto hw = make_profile(10, 10, 8, 2, IsaTier::AppleM1Plus);
    EXPECT_EQ(pick_n_threads(hw), 8u);
}

TEST(PickNThreads, LowCoreCount_4Physical) {
    using namespace helix;
    // 4-core machine: leave one for OS.
    auto hw = make_profile(8, 4, 4, 0, IsaTier::Avx2);
    EXPECT_EQ(pick_n_threads(hw), 3u);
}

TEST(PickNThreads, ServerCap_64Cores) {
    using namespace helix;
    // 64-core server: hard cap at 32.
    auto hw = make_profile(128, 64, 64, 0, IsaTier::Avx512);
    EXPECT_LE(pick_n_threads(hw), 32u);
}

// ── pick_n_threads_batch ─────────────────────────────────────────────────────

TEST(PickNThreadsBatch, HybridX86_UsesAllLogical) {
    using namespace helix;
    auto hw = make_profile(24, 16, 8, 8, IsaTier::Avx512);
    EXPECT_EQ(pick_n_threads_batch(hw), 24u);
}

TEST(PickNThreadsBatch, AppleM_UsesPerf) {
    using namespace helix;
    auto hw = make_profile(10, 10, 8, 2, IsaTier::AppleM1Plus);
    EXPECT_EQ(pick_n_threads_batch(hw), 8u);
}

TEST(PickNThreadsBatch, HomogeneousX86_AllLogical) {
    using namespace helix;
    auto hw = make_profile(16, 8, 8, 0, IsaTier::Avx2);
    EXPECT_EQ(pick_n_threads_batch(hw), 16u);
}

// ── pick_n_batch ─────────────────────────────────────────────────────────────

TEST(PickNBatch, Default) {
    using namespace helix;
    auto hw = make_profile(8, 4, 4, 0, IsaTier::Avx2);
    ModelInfo mi; mi.n_ctx = 4096;
    EXPECT_EQ(pick_n_batch(hw, mi), 512u);
}

TEST(PickNBatch, LongContext) {
    using namespace helix;
    auto hw = make_profile(8, 4, 4, 0, IsaTier::Avx2);
    ModelInfo mi; mi.n_ctx = 32768;
    EXPECT_EQ(pick_n_batch(hw, mi), 2048u);
}

// ── pick_flash_attn ──────────────────────────────────────────────────────────

TEST(PickFlashAttn, AvxDisabled) {
    using namespace helix;
    auto hw = make_profile(8, 4, 4, 0, IsaTier::Avx2);
    EXPECT_FALSE(pick_flash_attn(hw));
}

TEST(PickFlashAttn, Avx512Enabled) {
    using namespace helix;
    auto hw = make_profile(8, 4, 4, 0, IsaTier::Avx512);
    EXPECT_TRUE(pick_flash_attn(hw));
}

TEST(PickFlashAttn, AmxEnabled) {
    using namespace helix;
    auto hw = make_profile(8, 4, 4, 0, IsaTier::Amx);
    EXPECT_TRUE(pick_flash_attn(hw));
}

TEST(PickFlashAttn, NeonDisabled) {
    using namespace helix;
    auto hw = make_profile(8, 8, 8, 0, IsaTier::Neon);
    EXPECT_FALSE(pick_flash_attn(hw));
}

// ── pick_kv_cache_type ───────────────────────────────────────────────────────

TEST(PickKvCacheType, AmpleRam_Default_F16) {
    using namespace helix;
    auto hw = make_profile(8, 4, 4, 0, IsaTier::Avx2, 32, 28);
    EXPECT_EQ(pick_kv_cache_type(hw, false), GGML_TYPE_F16);
}

TEST(PickKvCacheType, TightRam_NoTools_Q4) {
    using namespace helix;
    auto hw = make_profile(8, 4, 4, 0, IsaTier::Avx2, 8, 6);
    EXPECT_EQ(pick_kv_cache_type(hw, false), GGML_TYPE_Q4_0);
}

TEST(PickKvCacheType, TightRam_WithTools_Q8) {
    using namespace helix;
    auto hw = make_profile(8, 4, 4, 0, IsaTier::Avx2, 8, 6);
    EXPECT_EQ(pick_kv_cache_type(hw, true), GGML_TYPE_Q8_0);
}

// ── ram_budget ───────────────────────────────────────────────────────────────

TEST(RamBudget, Calculation) {
    using namespace helix;
    RamInfo ram;
    ram.available_bytes = 16ULL * 1024 * 1024 * 1024; // 16 GiB
    uint64_t budget = ram_budget(ram);
    // 16 GiB - 1.5 GiB host - 256 MiB helix = 14.25 GiB
    EXPECT_GT(budget, 14ULL * 1024 * 1024 * 1024);
    EXPECT_LT(budget, 15ULL * 1024 * 1024 * 1024);
}

TEST(RamBudget, LowMemoryReturnsZero) {
    using namespace helix;
    RamInfo ram;
    ram.available_bytes = 512ULL * 1024 * 1024; // 512 MiB — less than reservations
    EXPECT_EQ(ram_budget(ram), 0u);
}

// ── pick_n_threads with ModelInfo (GPU offload) ──────────────────────────────

TEST(PickNThreads, FullGpuOffloadReturnsOne) {
    using namespace helix;
    auto hw = make_profile(16, 8, 8, 0, IsaTier::Avx2);
    AcceleratorInfo a;
    a.backend            = AccelBackend::Cuda;
    a.device_index       = 0;
    a.total_vram_bytes   = 24ULL * 1024 * 1024 * 1024;
    a.free_vram_bytes    = 20ULL * 1024 * 1024 * 1024;
    hw.accelerators.push_back(a);

    ModelInfo mi;
    mi.n_layers     = 32;
    mi.n_gpu_layers = 32;
    EXPECT_EQ(pick_n_threads(hw, mi), 1u);
}

TEST(PickNThreads, PartialGpuOffloadScalesDown) {
    using namespace helix;
    auto hw = make_profile(16, 8, 8, 0, IsaTier::Avx2);
    AcceleratorInfo a;
    a.backend            = AccelBackend::Cuda;
    a.device_index       = 0;
    a.total_vram_bytes   = 24ULL * 1024 * 1024 * 1024;
    a.free_vram_bytes    = 20ULL * 1024 * 1024 * 1024;
    hw.accelerators.push_back(a);

    ModelInfo mi;
    mi.n_layers     = 32;
    mi.n_gpu_layers = 16;
    uint32_t n = pick_n_threads(hw, mi);
    EXPECT_GT(n, 1u);
    EXPECT_LT(n, 8u);
}

TEST(PickNThreads, CpuOnlyUsesPhysicalCores) {
    using namespace helix;
    auto hw = make_profile(16, 8, 8, 0, IsaTier::Avx2);
    ModelInfo mi;
    mi.n_layers     = 32;
    mi.n_gpu_layers = 0;
    EXPECT_EQ(pick_n_threads(hw, mi), 8u);
}

// ── pick_n_threads_batch with ModelInfo ──────────────────────────────────────

TEST(PickNThreadsBatch, FullGpuOffloadReturnsOne) {
    using namespace helix;
    auto hw = make_profile(16, 8, 8, 0, IsaTier::Avx2);
    AcceleratorInfo a;
    a.backend            = AccelBackend::Cuda;
    a.device_index       = 0;
    a.total_vram_bytes   = 24ULL * 1024 * 1024 * 1024;
    a.free_vram_bytes    = 20ULL * 1024 * 1024 * 1024;
    hw.accelerators.push_back(a);

    ModelInfo mi;
    mi.n_layers     = 32;
    mi.n_gpu_layers = 32;
    EXPECT_EQ(pick_n_threads_batch(hw, mi), 1u);
}

TEST(PickNThreadsBatch, PartialGpuOffloadScalesDown) {
    using namespace helix;
    auto hw = make_profile(16, 8, 8, 0, IsaTier::Avx2);
    AcceleratorInfo a;
    a.backend            = AccelBackend::Cuda;
    a.device_index       = 0;
    a.total_vram_bytes   = 24ULL * 1024 * 1024 * 1024;
    a.free_vram_bytes    = 20ULL * 1024 * 1024 * 1024;
    hw.accelerators.push_back(a);

    ModelInfo mi;
    mi.n_layers     = 32;
    mi.n_gpu_layers = 16;
    uint32_t n = pick_n_threads_batch(hw, mi);
    EXPECT_GT(n, 1u);
    EXPECT_LT(n, 16u);
}

// ── pick_n_batch with GPU ────────────────────────────────────────────────────

TEST(PickNBatch, MetalBackendReturns2048) {
    using namespace helix;
    auto hw = make_profile(10, 10, 8, 2, IsaTier::AppleM1Plus);
    AcceleratorInfo a;
    a.backend            = AccelBackend::Metal;
    a.device_index       = 0;
    a.total_vram_bytes   = 32ULL * 1024 * 1024 * 1024;
    a.free_vram_bytes    = 28ULL * 1024 * 1024 * 1024;
    hw.accelerators.push_back(a);

    ModelInfo mi;
    mi.n_ctx = 4096;
    EXPECT_EQ(pick_n_batch(hw, mi), 2048u);
}

TEST(PickNBatch, FullGpuOffloadReturns2048) {
    using namespace helix;
    auto hw = make_profile(16, 8, 8, 0, IsaTier::Avx2);
    AcceleratorInfo a;
    a.backend            = AccelBackend::Cuda;
    a.device_index       = 0;
    a.total_vram_bytes   = 24ULL * 1024 * 1024 * 1024;
    a.free_vram_bytes    = 20ULL * 1024 * 1024 * 1024;
    hw.accelerators.push_back(a);

    ModelInfo mi;
    mi.n_ctx        = 4096;
    mi.n_layers     = 32;
    mi.n_gpu_layers = 32;
    EXPECT_EQ(pick_n_batch(hw, mi), 2048u);
}

TEST(PickNBatch, PartialGpuOffloadReturns1024) {
    using namespace helix;
    auto hw = make_profile(16, 8, 8, 0, IsaTier::Avx2);
    AcceleratorInfo a;
    a.backend            = AccelBackend::Cuda;
    a.device_index       = 0;
    a.total_vram_bytes   = 24ULL * 1024 * 1024 * 1024;
    a.free_vram_bytes    = 20ULL * 1024 * 1024 * 1024;
    hw.accelerators.push_back(a);

    ModelInfo mi;
    mi.n_ctx        = 4096;
    mi.n_layers     = 32;
    mi.n_gpu_layers = 16;
    EXPECT_EQ(pick_n_batch(hw, mi), 1024u);
}

// ── pick_n_ubatch ────────────────────────────────────────────────────────────

TEST(PickNUbatch, CpuOnlyReturnsNBatch) {
    using namespace helix;
    auto hw = make_profile(8, 4, 4, 0, IsaTier::Avx2);
    ModelInfo mi;
    mi.n_ctx = 4096;
    EXPECT_EQ(pick_n_ubatch(hw, mi, 512), 512u);
    EXPECT_EQ(pick_n_ubatch(hw, mi, 2048), 2048u);
}

TEST(PickNUbatch, MetalReturnsNBatch) {
    using namespace helix;
    auto hw = make_profile(10, 10, 8, 2, IsaTier::AppleM1Plus);
    AcceleratorInfo a;
    a.backend            = AccelBackend::Metal;
    a.device_index       = 0;
    hw.accelerators.push_back(a);

    ModelInfo mi;
    mi.n_ctx = 4096;
    EXPECT_EQ(pick_n_ubatch(hw, mi, 2048), 2048u);
}

TEST(PickNUbatch, FullGpuOffloadCapsAt512) {
    using namespace helix;
    auto hw = make_profile(16, 8, 8, 0, IsaTier::Avx2);
    AcceleratorInfo a;
    a.backend            = AccelBackend::Cuda;
    a.device_index       = 0;
    hw.accelerators.push_back(a);

    ModelInfo mi;
    mi.n_layers     = 32;
    mi.n_gpu_layers = 32;
    EXPECT_EQ(pick_n_ubatch(hw, mi, 2048), 512u);
}

TEST(PickNUbatch, PartialGpuOffloadCapsAt256) {
    using namespace helix;
    auto hw = make_profile(16, 8, 8, 0, IsaTier::Avx2);
    AcceleratorInfo a;
    a.backend            = AccelBackend::Cuda;
    a.device_index       = 0;
    hw.accelerators.push_back(a);

    ModelInfo mi;
    mi.n_layers     = 32;
    mi.n_gpu_layers = 16;
    EXPECT_EQ(pick_n_ubatch(hw, mi, 2048), 256u);
}

// ── pick_flash_attn with GPU backends ────────────────────────────────────────

TEST(PickFlashAttn, CudaVoltaEnabled) {
    using namespace helix;
    auto hw = make_profile(16, 8, 8, 0, IsaTier::Avx2);
    AcceleratorInfo a;
    a.backend             = AccelBackend::Cuda;
    a.device_index        = 0;
    a.compute_capability  = "7.0";
    hw.accelerators.push_back(a);
    EXPECT_TRUE(pick_flash_attn(hw));
}

TEST(PickFlashAttn, CudaPascalDisabled) {
    using namespace helix;
    auto hw = make_profile(16, 8, 8, 0, IsaTier::Avx2);
    AcceleratorInfo a;
    a.backend             = AccelBackend::Cuda;
    a.device_index        = 0;
    a.compute_capability  = "6.1";
    hw.accelerators.push_back(a);
    EXPECT_FALSE(pick_flash_attn(hw));
}

TEST(PickFlashAttn, CudaAmpereEnabled) {
    using namespace helix;
    auto hw = make_profile(16, 8, 8, 0, IsaTier::Avx2);
    AcceleratorInfo a;
    a.backend             = AccelBackend::Cuda;
    a.device_index        = 0;
    a.compute_capability  = "8.9";
    hw.accelerators.push_back(a);
    EXPECT_TRUE(pick_flash_attn(hw));
}

TEST(PickFlashAttn, CudaNoComputeCapabilityDisabled) {
    using namespace helix;
    auto hw = make_profile(16, 8, 8, 0, IsaTier::Avx2);
    AcceleratorInfo a;
    a.backend             = AccelBackend::Cuda;
    a.device_index        = 0;
    hw.accelerators.push_back(a);
    EXPECT_FALSE(pick_flash_attn(hw));
}

TEST(PickFlashAttn, RocmGfx11Enabled) {
    using namespace helix;
    auto hw = make_profile(16, 8, 8, 0, IsaTier::Avx2);
    AcceleratorInfo a;
    a.backend             = AccelBackend::Rocm;
    a.device_index        = 0;
    a.compute_capability  = "gfx1100";
    hw.accelerators.push_back(a);
    EXPECT_TRUE(pick_flash_attn(hw));
}

TEST(PickFlashAttn, RocmGfx10Disabled) {
    using namespace helix;
    auto hw = make_profile(16, 8, 8, 0, IsaTier::Avx2);
    AcceleratorInfo a;
    a.backend             = AccelBackend::Rocm;
    a.device_index        = 0;
    a.compute_capability  = "gfx1030";
    hw.accelerators.push_back(a);
    EXPECT_FALSE(pick_flash_attn(hw));
}

TEST(PickFlashAttn, MetalEnabled) {
    using namespace helix;
    auto hw = make_profile(10, 10, 8, 2, IsaTier::AppleM1Plus);
    AcceleratorInfo a;
    a.backend             = AccelBackend::Metal;
    a.device_index        = 0;
    hw.accelerators.push_back(a);
    EXPECT_TRUE(pick_flash_attn(hw));
}

TEST(PickFlashAttn, VulkanDisabled) {
    using namespace helix;
    auto hw = make_profile(16, 8, 8, 0, IsaTier::Avx2);
    AcceleratorInfo a;
    a.backend             = AccelBackend::Vulkan;
    a.device_index        = 0;
    hw.accelerators.push_back(a);
    EXPECT_FALSE(pick_flash_attn(hw));
}

// ── pick_kv_cache_type with GPU ──────────────────────────────────────────────

TEST(PickKvCacheType, GpuHighVramF16) {
    using namespace helix;
    auto hw = make_profile(8, 4, 4, 0, IsaTier::Avx2, 32, 28);
    AcceleratorInfo a;
    a.backend            = AccelBackend::Cuda;
    a.device_index       = 0;
    a.total_vram_bytes   = 24ULL * 1024 * 1024 * 1024;
    a.free_vram_bytes    = 20ULL * 1024 * 1024 * 1024;
    hw.accelerators.push_back(a);
    EXPECT_EQ(pick_kv_cache_type(hw, false), GGML_TYPE_F16);
}

TEST(PickKvCacheType, GpuLowVramFallsBackToRamBudget) {
    using namespace helix;
    auto hw = make_profile(8, 4, 4, 0, IsaTier::Avx2, 8, 6);
    AcceleratorInfo a;
    a.backend            = AccelBackend::Cuda;
    a.device_index       = 0;
    a.total_vram_bytes   = 4ULL * 1024 * 1024 * 1024;
    a.free_vram_bytes    = 512ULL * 1024 * 1024;
    hw.accelerators.push_back(a);
    EXPECT_EQ(pick_kv_cache_type(hw, false), GGML_TYPE_Q4_0);
    EXPECT_EQ(pick_kv_cache_type(hw, true), GGML_TYPE_Q8_0);
}

// ── ram_budget edge cases ────────────────────────────────────────────────────

TEST(RamBudget, ExactReservationBytesReturnsZero) {
    using namespace helix;
    RamInfo ram;
    ram.available_bytes = kHostReservationBytes + kHelixWorkingBytes;
    EXPECT_EQ(ram_budget(ram), 0u);
}

TEST(RamBudget, SlightlyAboveReservation) {
    using namespace helix;
    RamInfo ram;
    ram.available_bytes = kHostReservationBytes + kHelixWorkingBytes + 1024;
    EXPECT_EQ(ram_budget(ram), 1024u);
}

// ── pick_n_batch_embed / pick_n_threads_embed ────────────────────────────────

TEST(PickNBatchEmbed, CpuDefaultBucketKeepsFloor) {
    using namespace helix;
    auto hw = make_profile(8, 4, 4, 0, IsaTier::Avx2);
    ModelInfo mi; mi.n_ctx = 2048; mi.n_ctx_train = 2048;
    // CPU default bucket is 512; floor is 512.
    EXPECT_EQ(pick_n_batch_embed(hw, mi), 512u);
}

TEST(PickNBatchEmbed, TrainingWindowCapsMetalBucket) {
    using namespace helix;
    auto hw = make_profile(10, 10, 8, 2, IsaTier::AppleM1Plus);
    AcceleratorInfo a;
    a.backend          = AccelBackend::Metal;
    a.device_index     = 0;
    a.total_vram_bytes = 16ULL * 1024 * 1024 * 1024;
    a.free_vram_bytes  = 12ULL * 1024 * 1024 * 1024;
    hw.accelerators.push_back(a);

    ModelInfo mi; mi.n_ctx = 1024; mi.n_ctx_train = 1024;
    // Metal bucket is 2048; the 1024-token training window caps it.
    EXPECT_EQ(pick_n_batch_embed(hw, mi), 1024u);
}

TEST(PickNBatchEmbed, FloorAppliesBelowTinyTrainingWindow) {
    using namespace helix;
    auto hw = make_profile(8, 4, 4, 0, IsaTier::Avx2);
    ModelInfo mi; mi.n_ctx = 128; mi.n_ctx_train = 128;
    // The 512 floor keeps small-window models batchable.
    EXPECT_EQ(pick_n_batch_embed(hw, mi), 512u);
}

TEST(PickNBatchEmbed, UnknownTrainingWindowSkipsCap) {
    using namespace helix;
    auto hw = make_profile(10, 10, 8, 2, IsaTier::AppleM1Plus);
    AcceleratorInfo a;
    a.backend          = AccelBackend::Metal;
    a.device_index     = 0;
    a.total_vram_bytes = 16ULL * 1024 * 1024 * 1024;
    a.free_vram_bytes  = 12ULL * 1024 * 1024 * 1024;
    hw.accelerators.push_back(a);

    ModelInfo mi; mi.n_ctx = 8192; mi.n_ctx_train = 0; // unknown
    EXPECT_EQ(pick_n_batch_embed(hw, mi), 2048u);
}

TEST(PickNBatchEmbed, AlwaysWithinClampRange) {
    using namespace helix;
    auto cpu_hw = make_profile(8, 4, 4, 0, IsaTier::Avx2);
    ModelInfo big; big.n_ctx = 65536; big.n_ctx_train = 1u << 20;
    EXPECT_GE(pick_n_batch_embed(cpu_hw, big), 512u);
    EXPECT_LE(pick_n_batch_embed(cpu_hw, big), 8192u);
}

TEST(PickNThreadsEmbed, MatchesBatchHeuristic) {
    using namespace helix;
    auto hw = make_profile(24, 16, 8, 8, IsaTier::Avx512);
    ModelInfo mi; mi.n_ctx = 2048;
    EXPECT_EQ(pick_n_threads_embed(hw, mi), pick_n_threads_batch(hw, mi));
}

TEST(PickNThreadsEmbed, FullGpuOffloadReturnsOne) {
    using namespace helix;
    auto hw = make_profile(16, 8, 8, 0, IsaTier::Avx2);
    AcceleratorInfo a;
    a.backend          = AccelBackend::Cuda;
    a.device_index     = 0;
    a.total_vram_bytes = 24ULL * 1024 * 1024 * 1024;
    a.free_vram_bytes  = 20ULL * 1024 * 1024 * 1024;
    hw.accelerators.push_back(a);

    ModelInfo mi;
    mi.n_layers     = 12;
    mi.n_gpu_layers = 12;
    EXPECT_EQ(pick_n_threads_embed(hw, mi), 1u);
}
