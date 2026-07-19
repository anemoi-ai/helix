#include <gtest/gtest.h>
#include "hardware/cpu.hpp"
#include "hardware/profile.hpp"

// accel_backend_name ─────────────────────────────────────────────────────────

TEST(AccelBackend, Names) {
    using namespace helix;
    EXPECT_STREQ(accel_backend_name(AccelBackend::Cuda),    "cuda");
    EXPECT_STREQ(accel_backend_name(AccelBackend::Metal),   "metal");
    EXPECT_STREQ(accel_backend_name(AccelBackend::Vulkan),  "vulkan");
    EXPECT_STREQ(accel_backend_name(AccelBackend::Rocm),    "rocm");
    EXPECT_STREQ(accel_backend_name(AccelBackend::Unknown), "unknown");
}

// AcceleratorInfo defaults ────────────────────────────────────────────────────

TEST(AcceleratorInfo, Defaults) {
    helix::AcceleratorInfo a;
    EXPECT_EQ(a.backend,          helix::AccelBackend::Unknown);
    EXPECT_EQ(a.device_index,     0);
    EXPECT_EQ(a.total_vram_bytes, 0u);
    EXPECT_EQ(a.free_vram_bytes,  0u);
    EXPECT_FALSE(a.is_primary_display);
    EXPECT_FALSE(a.compute_capability.has_value());
}

// HardwareProfile contains accelerators ──────────────────────────────────────

TEST(HardwareProfile, AcceleratorsField) {
    helix::HardwareProfile hw;
    EXPECT_TRUE(hw.accelerators.empty());

    helix::AcceleratorInfo a;
    a.backend          = helix::AccelBackend::Cuda;
    a.device_index     = 0;
    a.device_name      = "NVIDIA GeForce RTX 4090";
    a.total_vram_bytes = 24ULL * 1024 * 1024 * 1024;
    a.free_vram_bytes  = 22ULL * 1024 * 1024 * 1024;
    a.compute_capability = "8.9";
    hw.accelerators.push_back(a);

    ASSERT_EQ(hw.accelerators.size(), 1u);
    EXPECT_EQ(hw.accelerators[0].backend, helix::AccelBackend::Cuda);
    EXPECT_EQ(hw.accelerators[0].compute_capability, "8.9");
}

// probe_accelerators() — smoke test on current machine ───────────────────────
// On a CPU-only machine this returns empty; on a GPU machine it must not crash.

TEST(AcceleratorProbe, DoesNotCrash) {
    auto accels = helix::probe_accelerators();
    for (const auto& a : accels) {
        EXPECT_GE(a.device_index, 0);
        EXPECT_FALSE(a.device_name.empty());
        EXPECT_TRUE(a.backend == helix::AccelBackend::Cuda   ||
                    a.backend == helix::AccelBackend::Metal  ||
                    a.backend == helix::AccelBackend::Vulkan ||
                    a.backend == helix::AccelBackend::Rocm   ||
                    a.backend == helix::AccelBackend::Unknown);
    }
}

// Multi-backend scenarios — construct profiles with different backends
// and verify the profile struct works correctly.

TEST(AcceleratorInfo, MetalBackendFields) {
    helix::AcceleratorInfo a;
    a.backend          = helix::AccelBackend::Metal;
    a.device_name      = "Apple M2 Max";
    a.total_vram_bytes = 32ULL * 1024 * 1024 * 1024;
    a.free_vram_bytes  = 28ULL * 1024 * 1024 * 1024;
    a.backend_name     = "Metal";
    EXPECT_EQ(a.backend, helix::AccelBackend::Metal);
    EXPECT_STREQ(helix::accel_backend_name(a.backend), "metal");
}

TEST(AcceleratorInfo, VulkanBackendFields) {
    helix::AcceleratorInfo a;
    a.backend          = helix::AccelBackend::Vulkan;
    a.device_name      = "Intel Arc A770";
    a.total_vram_bytes = 16ULL * 1024 * 1024 * 1024;
    a.free_vram_bytes  = 12ULL * 1024 * 1024 * 1024;
    a.backend_name     = "Vulkan";
    EXPECT_EQ(a.backend, helix::AccelBackend::Vulkan);
    EXPECT_STREQ(helix::accel_backend_name(a.backend), "vulkan");
}

TEST(AcceleratorInfo, RocmBackendFields) {
    helix::AcceleratorInfo a;
    a.backend             = helix::AccelBackend::Rocm;
    a.device_name         = "AMD Radeon RX 7900 XTX";
    a.total_vram_bytes    = 24ULL * 1024 * 1024 * 1024;
    a.free_vram_bytes     = 20ULL * 1024 * 1024 * 1024;
    a.compute_capability  = "gfx1100";
    a.backend_name        = "ROCm";
    EXPECT_EQ(a.backend, helix::AccelBackend::Rocm);
    EXPECT_STREQ(helix::accel_backend_name(a.backend), "rocm");
    ASSERT_TRUE(a.compute_capability.has_value());
    EXPECT_EQ(*a.compute_capability, "gfx1100");
}

TEST(HardwareProfile, MultiBackendAccelerators) {
    helix::HardwareProfile hw;

    helix::AcceleratorInfo cuda;
    cuda.backend         = helix::AccelBackend::Cuda;
    cuda.device_index    = 0;
    cuda.device_name     = "RTX 4090";
    cuda.compute_capability = "8.9";
    hw.accelerators.push_back(cuda);

    helix::AcceleratorInfo vulkan;
    vulkan.backend        = helix::AccelBackend::Vulkan;
    vulkan.device_index   = 1;
    vulkan.device_name    = "Intel Arc";
    hw.accelerators.push_back(vulkan);

    ASSERT_EQ(hw.accelerators.size(), 2u);
    EXPECT_EQ(hw.accelerators[0].backend, helix::AccelBackend::Cuda);
    EXPECT_EQ(hw.accelerators[1].backend, helix::AccelBackend::Vulkan);
    EXPECT_EQ(hw.accelerators[0].device_index, 0);
    EXPECT_EQ(hw.accelerators[1].device_index, 1);
}

TEST(AcceleratorInfo, BackendDeviceIndexAssignment) {
    helix::AcceleratorInfo a;
    a.backend              = helix::AccelBackend::Cuda;
    a.device_index         = 2;
    a.backend_device_index = 1;
    EXPECT_EQ(a.device_index, 2);
    EXPECT_EQ(a.backend_device_index, 1);
}

TEST(AcceleratorInfo, PcieFieldsOptional) {
    helix::AcceleratorInfo a;
    EXPECT_FALSE(a.pcie_link_gen.has_value());
    EXPECT_FALSE(a.pcie_link_width.has_value());

    a.pcie_link_gen   = 4;
    a.pcie_link_width = 16;
    EXPECT_EQ(*a.pcie_link_gen, 4);
    EXPECT_EQ(*a.pcie_link_width, 16);
}

TEST(AccelBackend, AllNamesNonNull) {
    using namespace helix;
    EXPECT_NE(accel_backend_name(AccelBackend::Cuda), nullptr);
    EXPECT_NE(accel_backend_name(AccelBackend::Metal), nullptr);
    EXPECT_NE(accel_backend_name(AccelBackend::Vulkan), nullptr);
    EXPECT_NE(accel_backend_name(AccelBackend::Rocm), nullptr);
    EXPECT_NE(accel_backend_name(AccelBackend::Unknown), nullptr);
}
