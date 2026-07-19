#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace helix {

enum class IsaTier {
    Generic,    // SSE4.2 or below (x86), or unknown
    Avx2,       // AVX2 (x86)
    Avx512,     // AVX-512F+BW+VL (x86)
    Amx,        // AMX (x86, Intel Sapphire Rapids+)
    Neon,       // NEON baseline (aarch64)
    NeonFp16,   // NEON + FP16
    NeonBf16,   // NEON + BF16
    Sve,        // SVE (aarch64)
    Sve2I8mm,   // SVE2 + I8MM (aarch64)
    AppleM1Plus, // Apple M-series unified memory
};

enum class MemBandwidthTier {
    Slow,      // < DDR4 speeds (embedded, old hardware)
    DDR4,      // ~50 GB/s
    DDR5,      // ~80 GB/s
    LPDDR5x,   // ~120 GB/s (Snapdragon X Elite etc.)
    OnPackage, // Apple Silicon unified memory (~200+ GB/s)
};

struct CpuInfo {
    std::string vendor;             // "GenuineIntel", "AuthenticAMD", "Apple", ...
    std::string model_name;
    uint32_t logical_cores    = 1;
    uint32_t physical_cores   = 1;
    uint32_t performance_cores = 1;
    uint32_t efficiency_cores  = 0;
    bool numa_available        = false;
    IsaTier isa_tier           = IsaTier::Generic;
    MemBandwidthTier bw_tier   = MemBandwidthTier::DDR4;
};

struct RamInfo {
    uint64_t total_bytes     = 0;
    uint64_t available_bytes = 0;
};

// Identifies which GPU stack powers an accelerator.
enum class AccelBackend { Cuda, Metal, Vulkan, Rocm, Unknown };

struct AcceleratorInfo {
    AccelBackend backend              = AccelBackend::Unknown;
    int          device_index         = 0;  // global helix accelerator index
    int          backend_device_index = 0;  // backend-local index (e.g. CUDA device 0, 1…)
    std::string  device_name;
    std::string  backend_name;      // raw ggml backend registry name
    uint64_t     total_vram_bytes   = 0;
    uint64_t     free_vram_bytes    = 0;
    bool         is_primary_display = false;

    // Optional extras (populated where the backend provides them):
    std::optional<std::string> compute_capability; // "12.0" (CUDA), "gfx1100" (ROCm)
    std::optional<int>         pcie_link_gen;       // 3, 4, 5
    std::optional<int>         pcie_link_width;     // 1, 4, 8, 16
};

struct HardwareProfile {
    CpuInfo cpu;
    RamInfo ram;
    std::vector<AcceleratorInfo> accelerators;
};

HardwareProfile probe_hardware();
CpuInfo probe_cpu();
IsaTier detect_isa_tier();

// Returns detected GPU accelerators (empty when none are available).
std::vector<AcceleratorInfo> probe_accelerators();

const char* isa_tier_name(IsaTier t);
const char* mem_bw_tier_name(MemBandwidthTier t);
const char* accel_backend_name(AccelBackend b);

} // namespace helix
