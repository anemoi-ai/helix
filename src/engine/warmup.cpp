#include "warmup.hpp"
#include "llama.h"
#include "../internal/log.hpp"

#include <algorithm>
#include <chrono>
#include <vector>

namespace helix {

WarmupResult run_warmup(llama_context* ctx, int n_vocab) {
    WarmupResult result;
    if (!ctx || n_vocab <= 0) return result;

    const int WARMUP_TOKENS = 256;
    const int n_batch        = static_cast<int>(llama_n_batch(ctx));
    const int n_tokens       = std::min(WARMUP_TOKENS, n_batch);

    // Use token ID 1 for all positions (a neutral, always-valid token in any vocab).
    const llama_token dummy_tok = std::min(1, n_vocab - 1);

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[i]     = dummy_tok;
        batch.pos[i]       = static_cast<llama_pos>(i);
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == n_tokens - 1) ? 1 : 0;
    }
    batch.n_tokens = n_tokens;

    auto t0 = std::chrono::steady_clock::now();
    int rc = llama_decode(ctx, batch);
    auto t1 = std::chrono::steady_clock::now();

    llama_batch_free(batch);

    // Clear KV so the warmup doesn't pollute the first real request.
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, 0, 0, -1);

    if (rc != 0) {
        log_warn("warmup decode failed (rc=" + std::to_string(rc) + ") — skipping");
        return result;
    }

    float elapsed_s = std::chrono::duration<float>(t1 - t0).count();
    result.prefill_tokens_per_sec = (elapsed_s > 0) ? n_tokens / elapsed_s : 0.0f;
    result.ran = true;

    log_debug("warmup: " + std::to_string(n_tokens) + " tokens in " +
              std::to_string(static_cast<int>(elapsed_s * 1000)) + " ms (" +
              std::to_string(static_cast<int>(result.prefill_tokens_per_sec)) + " t/s prefill)");

    return result;
}

} // namespace helix
