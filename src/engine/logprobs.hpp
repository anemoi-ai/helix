#pragma once
#include "../json/response.hpp"
#include "llama.h"

namespace helix {

// Compute a TokenLogprobEntry for the chosen token given the raw logits
// from the most recently decoded position.
//
// logits:       raw float logits for the position (not copied; must stay
//               valid for the duration of the call — i.e. no llama_decode
//               between capture and this call)
// n_vocab:      number of entries in logits
// chosen_token: the token that was actually sampled
// ctx:          context (for common_token_to_piece)
// temperature:  request temperature; 0 is treated as 1.0 for logprob scaling
// top_n:        number of top alternatives to include (0 = chosen token only)
TokenLogprobEntry compute_logprob_entry(
    const float* logits,
    int n_vocab,
    llama_token chosen_token,
    llama_context* ctx,
    float temperature,
    int top_n);

} // namespace helix
