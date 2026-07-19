#pragma once
#include "cpu.hpp"
#include <cstdint>
#include <string>

namespace helix {

// Sentinel: caller wants automatic GPU-layer selection.
inline constexpr int kGpuLayersAuto = -1;
// Sentinel: caller wants full offload (all layers).
inline constexpr int kGpuLayersAll  = -2;

// Result of pick_n_gpu_layers().
struct GpuLayerPlan {
    int         n_gpu_layers = 0;          // actual layers to offload (0 = CPU only)
    bool        full_offload = false;       // all transformer layers fit on GPU
    std::string rule;                       // human-readable rationale for --explain
};

// Decide how many layers to put on the GPU.
//
// user_request:
//   kGpuLayersAuto (-1) → run the fit pass / heuristic
//   kGpuLayersAll  (-2) → request full offload; return VRAM headroom warning
//   >= 0            → explicit; accept as-is (validation happens at load time)
//
// model_path is needed for common_fit_params (which does a no-alloc model load).
// n_layers is the total transformer layer count (llama_model_n_layer).
// n_ctx    is the context size the session will use.
GpuLayerPlan pick_n_gpu_layers(const HardwareProfile& hw,
                                int    user_request,
                                const  std::string& model_path,
                                int    n_layers,
                                int    n_ctx);

} // namespace helix
