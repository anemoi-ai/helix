#include "cpu.hpp"
#include "ggml-backend.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <map>

namespace helix {

const char* accel_backend_name(AccelBackend b) {
    switch (b) {
        case AccelBackend::Cuda:    return "cuda";
        case AccelBackend::Metal:   return "metal";
        case AccelBackend::Vulkan:  return "vulkan";
        case AccelBackend::Rocm:    return "rocm";
        default:                    return "unknown";
    }
}

// Map the GGML backend registry name to our enum.  The strings are
// lower-cased first so they're case-insensitive.
static AccelBackend classify_backend(const char* reg_name) {
    if (!reg_name) return AccelBackend::Unknown;
    std::string s(reg_name);
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (s.find("cuda") != std::string::npos)   return AccelBackend::Cuda;
    if (s.find("metal") != std::string::npos)  return AccelBackend::Metal;
    if (s.find("vulkan") != std::string::npos) return AccelBackend::Vulkan;
    if (s.find("hip") != std::string::npos)    return AccelBackend::Rocm;
    if (s.find("rocm") != std::string::npos)   return AccelBackend::Rocm;
    return AccelBackend::Unknown;
}

#if defined(HELIX_HAS_CUDA)
// Forward declaration — defined in accel_cuda.cpp (CUDA builds only).
void enrich_cuda_device(AcceleratorInfo& info);
#endif

std::vector<AcceleratorInfo> probe_accelerators() {
    std::vector<AcceleratorInfo> result;

    const size_t n = ggml_backend_dev_count();
    int gpu_index = 0;
    // Per-backend device counter so each backend's index starts at 0.
    std::map<AccelBackend, int> backend_counters;

    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;

        const auto dtype = ggml_backend_dev_type(dev);
        if (dtype != GGML_BACKEND_DEVICE_TYPE_GPU &&
            dtype != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }

        AcceleratorInfo info;
        info.device_index = gpu_index++;

        const char* dev_desc = ggml_backend_dev_description(dev);
        if (dev_desc) info.device_name = dev_desc;

        // The registry name identifies the backend (cuda, metal, vulkan…)
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (reg) {
            const char* reg_name = ggml_backend_reg_name(reg);
            if (reg_name) {
                info.backend_name = reg_name;
                info.backend = classify_backend(reg_name);
            }
        }

        // Assign backend-local device index (0, 1, … per backend).
        info.backend_device_index = backend_counters[info.backend]++;

        size_t free_b = 0, total_b = 0;
        ggml_backend_dev_memory(dev, &free_b, &total_b);
        info.total_vram_bytes = static_cast<uint64_t>(total_b);
        info.free_vram_bytes  = static_cast<uint64_t>(free_b);

        // Backend-specific enrichment (name, CC, PCIe).
#if defined(HELIX_HAS_CUDA)
        if (info.backend == AccelBackend::Cuda) {
            enrich_cuda_device(info);
        }
#endif

        result.push_back(std::move(info));
    }

    return result;
}

} // namespace helix
