#pragma once
#include "cpu.hpp"
#include "ggml.h"   // for ggml_type
#include <cstdint>
#include <optional>

namespace helix {

// Lightweight view of model characteristics relevant to the selector functions.
struct ModelInfo {
    uint64_t n_params     = 0;
    uint32_t n_ctx        = 4096;
    uint32_t n_ctx_train  = 0;     // model training context (0 = unknown)
    bool     tools_likely = false;
    // How many transformer layers are on GPU (0 = CPU only, == n_layers → full GPU).
    int      n_gpu_layers = 0;
    int      n_layers     = 0;
};

// RAM reservations (§5.2.4)
static constexpr uint64_t kHostReservationBytes = 1536ULL * 1024 * 1024;       // 1.5 GiB
static constexpr uint64_t kHelixWorkingBytes     =  256ULL * 1024 * 1024;       // 256 MiB
static constexpr uint64_t kLowRamThresholdBytes  =   12ULL * 1024 * 1024 * 1024; // 12 GiB

// Thread count: decode (single-token, memory-bound).
// Model-aware overload: adjusts for GPU offload fraction.
uint32_t pick_n_threads(const HardwareProfile& hw, const ModelInfo& m);
uint32_t pick_n_threads(const HardwareProfile& hw); // legacy: ModelInfo{}

// Thread count: prefill (batch, more compute-bound).
uint32_t pick_n_threads_batch(const HardwareProfile& hw, const ModelInfo& m);
uint32_t pick_n_threads_batch(const HardwareProfile& hw); // legacy

// Logical batch size (max tokens per llama_decode call).
uint32_t pick_n_batch(const HardwareProfile& hw, const ModelInfo& m);

// Physical micro-batch size (internal compute chunk).
uint32_t pick_n_ubatch(const HardwareProfile& hw, const ModelInfo& m, uint32_t n_batch);

// Logical batch size for embedding sessions. Non-causal attention requires
// n_ubatch == n_batch, so this value is used for both; pick_n_ubatch is not
// consulted for embedding sessions.
uint32_t pick_n_batch_embed(const HardwareProfile& hw, const ModelInfo& m);

// Thread count for embedding sessions. There is no decode phase, so the
// prefill (batch) heuristic applies to the whole call.
uint32_t pick_n_threads_embed(const HardwareProfile& hw, const ModelInfo& m);

// Whether to enable flash attention.
bool pick_flash_attn(const HardwareProfile& hw);

// KV cache quantization type.
ggml_type pick_kv_cache_type(const HardwareProfile& hw, bool tools_likely);

// Memory budget available to the model (total available minus reservations).
uint64_t ram_budget(const RamInfo& ram);

} // namespace helix
