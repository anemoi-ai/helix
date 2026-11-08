#pragma once
#include "llama.h"
#include "ggml.h"

#include <vector>

namespace helix {

/* Fill helpers for a plain llama_batch (llama_batch_init / llama_decode).
 *
 * llama.cpp's common_batch_clear / common_batch_add operated on llama_batch
 * and were removed upstream (b12033) when `common_batch` became a wrapper over
 * llama_batch_ext. The plain batch and llama_decode remain, and the paths that
 * use these helpers — embeddings, multimodal step decode, the MTP prefill —
 * need nothing more, so they are reproduced here exactly rather than moving
 * every caller onto llama_batch_ext.
 *
 * The speculative paths cannot use these: common_speculative_process now reads
 * per-token entries from a `common_batch`, so the batch it is shown must be
 * the one that was decoded (see decode_loop.cpp). */

inline void batch_clear(llama_batch & batch) noexcept {
    batch.n_tokens = 0;
}

inline void batch_add(llama_batch & batch,
                      llama_token id,
                      llama_pos pos,
                      const std::vector<llama_seq_id> & seq_ids,
                      bool logits) {
    /* The same capacity check upstream made: llama_batch_init leaves a null
     * sentinel one past the last seq_id slot. */
    GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");

    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = static_cast<int32_t>(seq_ids.size());
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;

    batch.n_tokens++;
}

} // namespace helix
