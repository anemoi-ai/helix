#include "profile.hpp"
#include "../internal/log.hpp"
#include <algorithm>

namespace helix {

// Hard cap on decode threads — beyond ~32, coordination overhead dominates on
// server CPUs (§9.4). Power users on many-core parts (e.g. 64-core EPYC) can
// raise this at build time with -DHELIX_MAX_DECODE_THREADS=N.
#ifndef HELIX_MAX_DECODE_THREADS
#define HELIX_MAX_DECODE_THREADS 32
#endif
static constexpr uint32_t kMaxDecodeThreads = HELIX_MAX_DECODE_THREADS;
static_assert(kMaxDecodeThreads >= 1, "HELIX_MAX_DECODE_THREADS must be >= 1");

// VRAM threshold above which we prefer f16 KV cache.
static constexpr uint64_t kHighVramThresholdBytes = 16ULL * 1024 * 1024 * 1024; // 16 GiB

static bool has_gpu(const HardwareProfile& hw) {
    return !hw.accelerators.empty();
}

static bool is_full_gpu_offload(const HardwareProfile& hw, const ModelInfo& m) {
    return has_gpu(hw) && m.n_layers > 0 && m.n_gpu_layers >= m.n_layers;
}

uint32_t pick_n_threads(const HardwareProfile& hw, const ModelInfo& m) {
    // Full GPU offload: 1 CPU thread is optimal — it only launches kernels.
    if (is_full_gpu_offload(hw, m)) return 1;

    // Partial offload: scale proportionally to CPU-resident layers.
    if (has_gpu(hw) && m.n_layers > 0 && m.n_gpu_layers > 0) {
        const double cpu_frac = 1.0 - static_cast<double>(m.n_gpu_layers) / m.n_layers;
        const CpuInfo& c = hw.cpu;
        const uint32_t full_cpu = std::min(c.performance_cores, kMaxDecodeThreads);
        const uint32_t scaled   = static_cast<uint32_t>(full_cpu * cpu_frac + 0.5);
        return std::max(1u, scaled);
    }

    const CpuInfo& c = hw.cpu;
    uint32_t n = c.performance_cores;

    if (c.efficiency_cores > 0) {
        n = c.performance_cores;
        if (n < 4) n = std::min(c.physical_cores, 8u);
    } else {
        n = c.physical_cores;
    }

    // On small CPUs (<= 4 physical cores), reserve one core for the OS and the
    // caller's own threads so decode does not starve the rest of the process.
    if (c.physical_cores <= 4) n = std::max(1u, n - 1);

    return std::min(n, kMaxDecodeThreads);
}

// Legacy overload for callers that don't have ModelInfo yet (runtime describe).
uint32_t pick_n_threads(const HardwareProfile& hw) {
    return pick_n_threads(hw, ModelInfo{});
}

uint32_t pick_n_threads_batch(const HardwareProfile& hw, const ModelInfo& m) {
    // Full GPU offload: 1 thread (kernel launches only).
    if (is_full_gpu_offload(hw, m)) return 1;

    // Partial offload: scale to CPU-resident share.
    if (has_gpu(hw) && m.n_layers > 0 && m.n_gpu_layers > 0) {
        const double cpu_frac = 1.0 - static_cast<double>(m.n_gpu_layers) / m.n_layers;
        const CpuInfo& c = hw.cpu;
        const uint32_t full_cpu = c.logical_cores;
        const uint32_t scaled   = static_cast<uint32_t>(full_cpu * cpu_frac + 0.5);
        return std::max(1u, scaled);
    }

    const CpuInfo& c = hw.cpu;
    if (c.isa_tier == IsaTier::AppleM1Plus) return c.performance_cores;
    if (c.efficiency_cores > 0)             return c.logical_cores;
    return c.logical_cores;
}

uint32_t pick_n_threads_batch(const HardwareProfile& hw) {
    return pick_n_threads_batch(hw, ModelInfo{});
}

uint32_t pick_n_batch(const HardwareProfile& hw, const ModelInfo& m) {
    if (has_gpu(hw)) {
        if (hw.accelerators[0].backend == AccelBackend::Metal) {
            // Apple Silicon: 2048/2048 (unified memory, no PCIe bottleneck).
            return 2048;
        }
        if (is_full_gpu_offload(hw, m)) return 2048; // discrete GPU, full offload
        return 1024; // partial offload
    }
    if (m.n_ctx >= 32768) return 2048;
    return 512;
}

uint32_t pick_n_ubatch(const HardwareProfile& hw, const ModelInfo& m, uint32_t n_batch) {
    if (has_gpu(hw)) {
        if (hw.accelerators[0].backend == AccelBackend::Metal) {
            return n_batch; // Apple Silicon: match n_batch
        }
        if (is_full_gpu_offload(hw, m)) {
            // Discrete GPU full offload: 512 physical micro-batch.
            return std::min(n_batch, 512u);
        }
        return std::min(n_batch, 256u); // partial offload
    }
    return n_batch;
}

uint32_t pick_n_batch_embed(const HardwareProfile& hw, const ModelInfo& m) {
    // Start from the chat bucket, then cap at the model's training window —
    // an embedding input can never usefully exceed it. The 512 floor keeps
    // small-window models batchable; the 8192 ceiling bounds activation
    // memory (n_ubatch == n_batch on this path, so there is no micro-batch
    // cap to fall back on).
    uint32_t n = pick_n_batch(hw, m);
    if (m.n_ctx_train > 0) n = std::min(n, m.n_ctx_train);
    return std::clamp(n, 512u, 8192u);
}

uint32_t pick_n_threads_embed(const HardwareProfile& hw, const ModelInfo& m) {
    return pick_n_threads_batch(hw, m);
}

bool pick_flash_attn(const HardwareProfile& hw) {
    if (!hw.accelerators.empty()) {
        const AccelBackend b = hw.accelerators[0].backend;
        if (b == AccelBackend::Metal) return true; // all Apple Silicon
        if (b == AccelBackend::Cuda) {
            // Volta+ (compute cap >= 7.0).
            const auto& cc = hw.accelerators[0].compute_capability;
            if (cc) {
                try {
                    float v = std::stof(*cc);
                    return v >= 7.0f;
                } catch (...) {
                    log_warn("pick_flash_attn: could not parse compute capability '" +
                             *cc + "', disabling flash attention by default");
                }
            }
            /* Unknown capability: be conservative. Pascal (CC 6.x) and older
             * lack the flash-attention kernels llama.cpp needs, and requesting
             * the feature there is a hard error at context creation. Default
             * off; Volta+ users can opt in via a session option. */
            return false;
        }
        if (b == AccelBackend::Rocm) {
            // RDNA3+ (gfx1100+) with rocWMMA — conservative: off by default,
            // on when compute_capability starts with "gfx11".
            const auto& cc = hw.accelerators[0].compute_capability;
            if (cc && cc->rfind("gfx11", 0) == 0) return true;
            return false;
        }
        // Vulkan: off by default (driver heterogeneity; per-driver allow-list TBD).
        return false;
    }

    // CPU path (phase 4 logic).
    switch (hw.cpu.isa_tier) {
        case IsaTier::Avx512:
        case IsaTier::Amx:
            return true;
        default:
            return false;
    }
}

ggml_type pick_kv_cache_type(const HardwareProfile& hw, bool tools_likely) {
    if (has_gpu(hw)) {
        const uint64_t free_vram = hw.accelerators[0].free_vram_bytes;
        if (free_vram > kHighVramThresholdBytes) {
            return GGML_TYPE_F16;
        }
    }

    const uint64_t budget = ram_budget(hw.ram);
    if (budget < kLowRamThresholdBytes) {
        return tools_likely ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_0;
    }
    return GGML_TYPE_F16;
}

uint64_t ram_budget(const RamInfo& ram) {
    if (ram.available_bytes < kHostReservationBytes + kHelixWorkingBytes) {
        return 0;
    }
    return ram.available_bytes - kHostReservationBytes - kHelixWorkingBytes;
}

} // namespace helix
