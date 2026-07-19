#pragma once
#include "llama.h"
#include <map>
#include <string>
#include <vector>

namespace helix {

struct ResolvedBias {
    llama_token token;
    float       bias;
};

// Convert a raw logit_bias map (string key → bias) to resolved token+bias pairs.
// Integer string keys are treated as token IDs directly.
// Non-integer string keys are tokenised; the bias is applied to each resulting token.
// Throws HELIX_E_VALIDATION if an integer key is out of range for the vocabulary.
std::vector<ResolvedBias> resolve_logit_bias(
    const std::map<std::string, float>& raw,
    const llama_vocab* vocab);

} // namespace helix
