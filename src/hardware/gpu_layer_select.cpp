#include "gpu_layer_select.hpp"
#include "../internal/log.hpp"

#include "llama.h"
#include "common.h"
#include "fit.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <system_error>
#include <vector>

namespace helix {

// ---------------------------------------------------------------------------
// Heuristic fallback when common_fit_params is unavailable or the fit model
// load itself fails (unlikely, but defensive).  Uses the "oobabooga formula":
//
//   layers_fit = (free_vram_after_margin - kv_estimate) / per_layer_size_bytes
//
// where per_layer_size_bytes is a rough per-quant estimate.
// ---------------------------------------------------------------------------

static constexpr uint64_t kPrimaryGpuMarginBytes   = 1536ULL * 1024 * 1024; // 1.5 GiB
static constexpr uint64_t kSecondaryGpuMarginBytes =  512ULL * 1024 * 1024; // 0.5 GiB
// Last-resort per-layer VRAM estimate (~100 MiB/layer, Q4_K_M on a 7B model)
// used only when the model file size is unavailable. Real per-layer size varies
// ~3x across model sizes/quants, so heuristic_fit prefers a size-derived figure.
static constexpr uint64_t kDefaultLayerSizeBytes   =  100ULL * 1024 * 1024;
// Rough KV cache estimate per layer per context token: 2 (K+V) * 2 bytes (fp16).
// Multiplied by hidden_dim and n_ctx at heuristic time.
static constexpr uint64_t kKvCacheBytesPerLayerPerToken = 4ULL;

static GpuLayerPlan heuristic_fit(const HardwareProfile& hw,
                                   const std::string& model_path,
                                   int n_layers, int n_ctx) {
    if (hw.accelerators.empty()) {
        return GpuLayerPlan{0, false, "no accelerators found"};
    }

    const AcceleratorInfo& primary = hw.accelerators[0];
    const uint64_t margin  = primary.is_primary_display
                             ? kPrimaryGpuMarginBytes
                             : kSecondaryGpuMarginBytes;
    if (primary.free_vram_bytes <= margin) {
        return GpuLayerPlan{0, false,
            "insufficient free VRAM (heuristic: free VRAM < margin)"};
    }

    /* Per-layer estimate: prefer (GGUF file size / n_layers) over the fixed
     * fallback. The repeating transformer blocks dominate the file, so this
     * tracks the true per-layer cost far better across model sizes and quants.
     * It slightly over-counts (the divisor excludes non-layer tensors like the
     * embedding/output matrices), which is the safe direction for a VRAM fit. */
    uint64_t per_layer = kDefaultLayerSizeBytes;
    if (n_layers > 0 && !model_path.empty()) {
        std::error_code ec;
        const auto fsize = std::filesystem::file_size(model_path, ec);
        if (!ec && fsize > 0) {
            per_layer = static_cast<uint64_t>(fsize) / static_cast<uint64_t>(n_layers);
            if (per_layer == 0) per_layer = kDefaultLayerSizeBytes;
        }
    }

    const uint64_t usable = primary.free_vram_bytes - margin;

    // Subtract a rough KV cache estimate so we don't over-count VRAM.
    // Uses 2 (K+V) * fp16 per layer per token; hidden_dim unknown here,
    // so estimate 4096 as a conservative mid-range value.
    const int estimated_n_embd = 4096;
    const uint64_t kv_estimate = static_cast<uint64_t>(n_layers > 0 ? n_layers : 32)
                                 * static_cast<uint64_t>(n_ctx > 0 ? n_ctx : 4096)
                                 * static_cast<uint64_t>(estimated_n_embd)
                                 * kKvCacheBytesPerLayerPerToken;
    const uint64_t usable_after_kv = (usable > kv_estimate) ? (usable - kv_estimate) : 0;
    const int layers_fit  = static_cast<int>(usable_after_kv / per_layer);
    const int clamped     = std::min(layers_fit, n_layers);

    if (clamped <= 0) {
        return GpuLayerPlan{0, false, "heuristic: no layers fit within VRAM margin"};
    }

    bool full = (clamped >= n_layers);
    char buf[128];
    std::snprintf(buf, sizeof(buf),
        "heuristic: %d/%d layers fit (%s)",
        clamped, n_layers, full ? "full offload" : "partial offload");
    return GpuLayerPlan{clamped, full, buf};
}

// ---------------------------------------------------------------------------
// common_fit_params integration
// ---------------------------------------------------------------------------

static GpuLayerPlan fit_pass(const HardwareProfile& hw,
                              const std::string& model_path,
                              int n_layers,
                              int n_ctx) {
    if (hw.accelerators.empty()) {
        return GpuLayerPlan{0, false, "no accelerators found"};
    }

    const size_t max_dev      = llama_max_devices();
    const size_t max_buft_ov  = llama_max_tensor_buft_overrides();

    std::vector<float>  tensor_split(max_dev, 0.0f);
    std::vector<llama_model_tensor_buft_override> buft_overrides(max_buft_ov);
    std::memset(buft_overrides.data(), 0,
                max_buft_ov * sizeof(llama_model_tensor_buft_override));

    // Per-device margins.
    std::vector<size_t> margins(max_dev, kSecondaryGpuMarginBytes);
    for (size_t i = 0; i < hw.accelerators.size() && i < max_dev; ++i) {
        margins[i] = hw.accelerators[i].is_primary_display
                     ? kPrimaryGpuMarginBytes
                     : kSecondaryGpuMarginBytes;
    }

    auto mparams = llama_model_default_params();
    auto cparams = llama_context_default_params();
    cparams.n_ctx   = static_cast<uint32_t>(n_ctx > 0 ? n_ctx : 4096);

    const auto status = common_fit_params(
        model_path.c_str(),
        &mparams,
        &cparams,
        tensor_split.data(),
        buft_overrides.data(),
        margins.data(),
        /*n_ctx_min=*/ 512,
        GGML_LOG_LEVEL_WARN);

    if (status != COMMON_PARAMS_FIT_STATUS_SUCCESS) {
        log_warn("common_fit_params failed (status=" +
                 std::to_string(static_cast<int>(status)) +
                 "); falling back to heuristic");
        return heuristic_fit(hw, model_path, n_layers, n_ctx);
    }

    /* common_fit_params leaves n_gpu_layers at -1 when the whole model
     * fits (llama.cpp convention for "offload everything"). */
    const int chosen = (mparams.n_gpu_layers < 0) ? n_layers + 1
                                                  : mparams.n_gpu_layers;
    bool full        = (chosen >= n_layers);
    char buf[128];
    std::snprintf(buf, sizeof(buf),
        "llama_params_fit: %d/%d layers fit (%s)",
        std::min(chosen, n_layers), n_layers,
        full ? "full offload" : "partial offload");

    /* llama.cpp convention: n_gpu_layers = n_layers + 1 also offloads the
     * output layer, so allow one above the repeating-block count. */
    return GpuLayerPlan{std::min(chosen, n_layers + 1), full, buf};
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

GpuLayerPlan pick_n_gpu_layers(const HardwareProfile& hw,
                                int    user_request,
                                const  std::string& model_path,
                                int    n_layers,
                                int    n_ctx) {
    if (n_layers <= 0) {
        return GpuLayerPlan{0, false, "invalid model: n_layers <= 0"};
    }

    if (user_request == 0) {
        return GpuLayerPlan{0, false, "user requested CPU-only (n_gpu_layers=0)"};
    }

    if (user_request > 0) {
        // Explicit user value — accept unconditionally.
        // The actual offload is handled by llama.cpp; if the values are
        // unrealistic (e.g. > n_layers) it will be silently clamped downstream.
        // n_layers + 1 also offloads the output layer (llama.cpp convention).
        int clamped = std::min(user_request, n_layers + 1);
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "user-specified: %d layers", clamped);
        return GpuLayerPlan{clamped, (clamped >= n_layers), buf};
    }

    if (hw.accelerators.empty()) {
        return GpuLayerPlan{0, false, "no accelerators"};
    }

    // kGpuLayersAll: request full offload, run fit to validate.
    if (user_request == kGpuLayersAll) {
        GpuLayerPlan plan = fit_pass(hw, model_path, n_layers, n_ctx);
        if (!plan.full_offload) {
            log_warn("n_gpu_layers=all requested but only " +
                     std::to_string(plan.n_gpu_layers) + "/" +
                     std::to_string(n_layers) + " layers fit; using partial offload");
        }
        plan.rule = "user requested full offload; " + plan.rule;
        return plan;
    }

    // kGpuLayersAuto: run fit pass (falls back to heuristic on failure).
    if (!model_path.empty()) {
        return fit_pass(hw, model_path, n_layers, n_ctx);
    }
    return heuristic_fit(hw, model_path, n_layers, n_ctx);
}

} // namespace helix
