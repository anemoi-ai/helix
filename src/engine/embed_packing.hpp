#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace helix {

/* Plans how embedding inputs pack into llama batches: greedy and
 * insertion-ordered, flushing when the next input would overflow n_batch
 * tokens or n_seq_max sequences. Insertion order is what makes embedding
 * vectors bit-deterministic for a fixed session geometry (HELIX-IMPL-001
 * §12), so do not reorder inputs to pack tighter.
 *
 * Precondition: every length is in (0, n_batch] — oversize inputs are
 * rejected before planning.
 *
 * Returns groups of input indices; each group is one llama_decode/encode
 * call, with the position within a group becoming the llama_seq_id. */
inline std::vector<std::vector<size_t>> plan_embed_flushes(
        const std::vector<size_t>& lengths,
        uint32_t n_batch,
        uint32_t n_seq_max) {
    std::vector<std::vector<size_t>> flushes;
    std::vector<size_t> live;
    size_t used_tokens = 0;

    for (size_t i = 0; i < lengths.size(); ++i) {
        if (!live.empty() &&
            (used_tokens + lengths[i] > n_batch || live.size() >= n_seq_max)) {
            flushes.push_back(std::move(live));
            live.clear();
            used_tokens = 0;
        }
        live.push_back(i);
        used_tokens += lengths[i];
    }
    if (!live.empty()) flushes.push_back(std::move(live));
    return flushes;
}

} // namespace helix
