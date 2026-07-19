#pragma once
#include <cstdint>

struct llama_context;

namespace helix {

struct WarmupResult {
    float prefill_tokens_per_sec = 0.0f;  // 0 = warmup skipped or failed
    bool  ran = false;
};

// Run a 256-token dummy prefill to measure throughput and prime the CPU.
// Clears the KV cache afterwards. Safe to call multiple times (idempotent result).
WarmupResult run_warmup(llama_context* ctx, int n_vocab);

} // namespace helix
