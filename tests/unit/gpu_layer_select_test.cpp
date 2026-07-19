#include <gtest/gtest.h>
#include "hardware/cpu.hpp"
#include "hardware/gpu_layer_select.hpp"

using namespace helix;

// Helper: build a HardwareProfile with one synthetic GPU.
static HardwareProfile make_gpu_profile(uint64_t free_vram_bytes,
                                        AccelBackend backend = AccelBackend::Cuda,
                                        bool is_primary = false) {
    HardwareProfile hw;
    hw.cpu.logical_cores   = 16;
    hw.cpu.physical_cores  = 8;
    hw.cpu.performance_cores = 8;
    hw.ram.total_bytes     = 64ULL * 1024 * 1024 * 1024;
    hw.ram.available_bytes = 32ULL * 1024 * 1024 * 1024;

    AcceleratorInfo a;
    a.backend             = backend;
    a.device_index        = 0;
    a.device_name         = "Test GPU";
    a.total_vram_bytes    = 24ULL * 1024 * 1024 * 1024;
    a.free_vram_bytes     = free_vram_bytes;
    a.is_primary_display  = is_primary;
    a.compute_capability  = "8.9";
    hw.accelerators.push_back(a);
    return hw;
}

// No accelerators → always 0 layers, regardless of user_request.
TEST(GpuLayerSelect, NoAccelerators_AlwaysZero) {
    HardwareProfile hw;  // no accelerators
    auto plan = pick_n_gpu_layers(hw, kGpuLayersAuto, "", 32, 4096);
    EXPECT_EQ(plan.n_gpu_layers, 0);
    EXPECT_FALSE(plan.full_offload);

    plan = pick_n_gpu_layers(hw, kGpuLayersAll, "", 32, 4096);
    EXPECT_EQ(plan.n_gpu_layers, 0);
}

// user_request=0 → CPU only even when GPU present.
TEST(GpuLayerSelect, UserRequestZero_CpuOnly) {
    auto hw = make_gpu_profile(20ULL * 1024 * 1024 * 1024);
    auto plan = pick_n_gpu_layers(hw, 0, "", 32, 4096);
    EXPECT_EQ(plan.n_gpu_layers, 0);
    EXPECT_FALSE(plan.full_offload);
}

// Explicit positive request is clamped to n_layers + 1 (the +1 also
// offloads the output layer, llama.cpp convention).
TEST(GpuLayerSelect, UserRequestExplicit_Clamps) {
    auto hw = make_gpu_profile(20ULL * 1024 * 1024 * 1024);
    // request > n_layers
    auto plan = pick_n_gpu_layers(hw, 999, "", 32, 4096);
    EXPECT_EQ(plan.n_gpu_layers, 33);
    EXPECT_TRUE(plan.full_offload);

    // request < n_layers
    plan = pick_n_gpu_layers(hw, 16, "", 32, 4096);
    EXPECT_EQ(plan.n_gpu_layers, 16);
    EXPECT_FALSE(plan.full_offload);
}

// Heuristic fallback when model_path is empty and no fit pass.
TEST(GpuLayerSelect, HeuristicFallback_InsufficientVram) {
    // Only 200 MiB free — less than primary GPU margin (1.5 GiB).
    auto hw = make_gpu_profile(200ULL * 1024 * 1024, AccelBackend::Cuda, /*primary=*/true);
    auto plan = pick_n_gpu_layers(hw, kGpuLayersAuto, "", 32, 4096);
    EXPECT_EQ(plan.n_gpu_layers, 0);
    EXPECT_FALSE(plan.full_offload);
}

TEST(GpuLayerSelect, HeuristicFallback_AmpleVram) {
    // 20 GiB free — well above the 1.5 GiB margin. With 100 MiB/layer heuristic,
    // ~185 layers fit → clamped to n_layers=32 → full offload.
    auto hw = make_gpu_profile(20ULL * 1024 * 1024 * 1024, AccelBackend::Cuda,
                                /*primary=*/true);
    auto plan = pick_n_gpu_layers(hw, kGpuLayersAuto, "", 32, 4096);
    EXPECT_EQ(plan.n_gpu_layers, 32);
    EXPECT_TRUE(plan.full_offload);
}

// Secondary GPU uses smaller margin (512 MiB).
TEST(GpuLayerSelect, SecondaryGpuSmallerMargin) {
    // 600 MiB free on a secondary GPU. Primary margin = 1.5 GiB would give 0;
    // secondary margin = 512 MiB leaves ~88 MiB → 0 layers with 100 MiB/layer.
    // With 512 MiB margin and 600 MiB free → usable = ~88 MiB → 0 layers.
    auto hw = make_gpu_profile(600ULL * 1024 * 1024, AccelBackend::Cuda,
                                /*primary=*/false);
    auto plan = pick_n_gpu_layers(hw, kGpuLayersAuto, "", 32, 4096);
    // 600 - 512 = 88 MiB usable, 88 / 100 = 0 layers
    EXPECT_EQ(plan.n_gpu_layers, 0);
}

// Rule string is not empty.
TEST(GpuLayerSelect, RuleStringPopulated) {
    auto hw = make_gpu_profile(20ULL * 1024 * 1024 * 1024);
    auto plan = pick_n_gpu_layers(hw, kGpuLayersAuto, "", 32, 4096);
    EXPECT_FALSE(plan.rule.empty());
}

// n_layers <= 0 produces a zero-layer plan with an error message.
TEST(GpuLayerSelect, InvalidNLayersZero) {
    auto hw = make_gpu_profile(20ULL * 1024 * 1024 * 1024);
    auto plan = pick_n_gpu_layers(hw, kGpuLayersAuto, "", 0, 4096);
    EXPECT_EQ(plan.n_gpu_layers, 0);
    EXPECT_FALSE(plan.full_offload);
    EXPECT_NE(plan.rule.find("invalid"), std::string::npos);
}

TEST(GpuLayerSelect, InvalidNLayersNegative) {
    auto hw = make_gpu_profile(20ULL * 1024 * 1024 * 1024);
    auto plan = pick_n_gpu_layers(hw, kGpuLayersAuto, "", -5, 4096);
    EXPECT_EQ(plan.n_gpu_layers, 0);
    EXPECT_FALSE(plan.full_offload);
}

// kGpuLayersAll (-2) with no model file falls back to heuristic.
TEST(GpuLayerSelect, AllLayersHeuristicAmpleVram) {
    auto hw = make_gpu_profile(20ULL * 1024 * 1024 * 1024, AccelBackend::Cuda, false);
    auto plan = pick_n_gpu_layers(hw, kGpuLayersAll, "", 32, 4096);
    EXPECT_TRUE(plan.full_offload);
    EXPECT_NE(plan.rule.find("full offload"), std::string::npos);
}

TEST(GpuLayerSelect, AllLayersHeuristicInsufficientVram) {
    auto hw = make_gpu_profile(200ULL * 1024 * 1024, AccelBackend::Cuda, true);
    auto plan = pick_n_gpu_layers(hw, kGpuLayersAll, "", 32, 4096);
    EXPECT_EQ(plan.n_gpu_layers, 0);
    EXPECT_NE(plan.rule.find("user requested"), std::string::npos);
}

// Multi-GPU: two accelerators with different VRAM.
TEST(GpuLayerSelect, MultiGpuUsesFirstAccelerator) {
    HardwareProfile hw;
    hw.cpu.logical_cores   = 16;
    hw.cpu.physical_cores  = 8;
    hw.cpu.performance_cores = 8;
    hw.ram.total_bytes     = 64ULL * 1024 * 1024 * 1024;
    hw.ram.available_bytes = 32ULL * 1024 * 1024 * 1024;

    AcceleratorInfo a1;
    a1.backend             = AccelBackend::Cuda;
    a1.device_index        = 0;
    a1.device_name         = "GPU 1";
    a1.total_vram_bytes    = 24ULL * 1024 * 1024 * 1024;
    a1.free_vram_bytes     = 20ULL * 1024 * 1024 * 1024;
    a1.is_primary_display  = true;
    a1.compute_capability  = "8.9";
    hw.accelerators.push_back(a1);

    AcceleratorInfo a2;
    a2.backend             = AccelBackend::Cuda;
    a2.device_index        = 1;
    a2.device_name         = "GPU 2";
    a2.total_vram_bytes    = 12ULL * 1024 * 1024 * 1024;
    a2.free_vram_bytes     = 10ULL * 1024 * 1024 * 1024;
    a2.is_primary_display  = false;
    a2.compute_capability  = "8.6";
    hw.accelerators.push_back(a2);

    auto plan = pick_n_gpu_layers(hw, kGpuLayersAuto, "", 32, 4096);
    EXPECT_GT(plan.n_gpu_layers, 0);
    EXPECT_TRUE(plan.full_offload);
}

TEST(GpuLayerSelect, MultiGpuDifferentBackends) {
    HardwareProfile hw;
    hw.cpu.logical_cores   = 16;
    hw.cpu.physical_cores  = 8;
    hw.cpu.performance_cores = 8;
    hw.ram.total_bytes     = 64ULL * 1024 * 1024 * 1024;
    hw.ram.available_bytes = 32ULL * 1024 * 1024 * 1024;

    AcceleratorInfo a1;
    a1.backend             = AccelBackend::Cuda;
    a1.device_index        = 0;
    a1.device_name         = "NVIDIA RTX";
    a1.total_vram_bytes    = 24ULL * 1024 * 1024 * 1024;
    a1.free_vram_bytes     = 20ULL * 1024 * 1024 * 1024;
    a1.is_primary_display  = true;
    hw.accelerators.push_back(a1);

    AcceleratorInfo a2;
    a2.backend             = AccelBackend::Vulkan;
    a2.device_index        = 1;
    a2.device_name         = "Intel Arc";
    a2.total_vram_bytes    = 12ULL * 1024 * 1024 * 1024;
    a2.free_vram_bytes     = 10ULL * 1024 * 1024 * 1024;
    a2.is_primary_display  = false;
    hw.accelerators.push_back(a2);

    auto plan = pick_n_gpu_layers(hw, kGpuLayersAuto, "", 32, 4096);
    EXPECT_GT(plan.n_gpu_layers, 0);
}

// Explicit request with single layer.
TEST(GpuLayerSelect, UserRequestSingleLayer) {
    auto hw = make_gpu_profile(20ULL * 1024 * 1024 * 1024);
    auto plan = pick_n_gpu_layers(hw, 1, "", 32, 4096);
    EXPECT_EQ(plan.n_gpu_layers, 1);
    EXPECT_FALSE(plan.full_offload);
}

// Explicit request matching n_layers exactly.
TEST(GpuLayerSelect, UserRequestExactMatch) {
    auto hw = make_gpu_profile(20ULL * 1024 * 1024 * 1024);
    auto plan = pick_n_gpu_layers(hw, 32, "", 32, 4096);
    EXPECT_EQ(plan.n_gpu_layers, 32);
    EXPECT_TRUE(plan.full_offload);
}
