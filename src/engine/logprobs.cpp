#include "logprobs.hpp"
#include "llama.h"
#include "common.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

namespace helix {

TokenLogprobEntry compute_logprob_entry(
    const float* logits,
    int n_vocab,
    llama_token chosen_token,
    llama_context* ctx,
    float temperature,
    int top_n)
{
    if (!logits || n_vocab <= 0) return {};

    // At temperature=0 (greedy) use 1.0 so logprobs remain informative.
    const float temp = temperature > 0.0f ? temperature : 1.0f;

    // Numerically stable log-softmax with temperature scaling.
    float max_logit = *std::max_element(logits, logits + n_vocab);
    std::vector<float> lp(n_vocab);
    float sum_exp = 0.0f;
    for (int i = 0; i < n_vocab; ++i) {
        lp[i] = (logits[i] - max_logit) / temp;
        sum_exp += std::exp(lp[i]);
    }
    const float log_sum = std::log(sum_exp);
    for (int i = 0; i < n_vocab; ++i) lp[i] -= log_sum;

    auto token_info = [&](llama_token t) -> std::pair<std::string, std::vector<uint8_t>> {
        std::string s = common_token_to_piece(ctx, t, /*special=*/false);
        return {s, std::vector<uint8_t>(s.begin(), s.end())};
    };

    if (chosen_token < 0 || chosen_token >= n_vocab) return {};

    auto [ch_str, ch_bytes] = token_info(chosen_token);
    TokenLogprobEntry entry;
    entry.token  = std::move(ch_str);
    entry.logprob = lp[chosen_token];
    entry.bytes  = std::move(ch_bytes);

    if (top_n > 0) {
        int actual_top = std::min(top_n, n_vocab);
        std::vector<int> idx(n_vocab);
        std::iota(idx.begin(), idx.end(), 0);
        // Partial sort descending by log-prob; tiebreak by token id ascending.
        std::partial_sort(idx.begin(), idx.begin() + actual_top, idx.end(),
            [&](int a, int b) {
                return lp[a] != lp[b] ? lp[a] > lp[b] : a < b;
            });
        entry.top_logprobs.reserve(actual_top);
        for (int k = 0; k < actual_top; ++k) {
            auto [s, b] = token_info(static_cast<llama_token>(idx[k]));
            TokenLogprobAlt alt;
            alt.token  = std::move(s);
            alt.logprob = lp[idx[k]];
            alt.bytes  = std::move(b);
            entry.top_logprobs.push_back(std::move(alt));
        }
    }

    return entry;
}

} // namespace helix
