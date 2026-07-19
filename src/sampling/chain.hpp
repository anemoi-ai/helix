#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "../chat/template.hpp"  /* for GrammarTrigger */
#include "logit_bias.hpp"         /* for ResolvedBias */

struct llama_sampler;
struct llama_context;
struct llama_vocab;
typedef int32_t llama_token;
namespace helix { struct ChatRequest; }

namespace helix {

struct SamplerDeleter {
    void operator()(llama_sampler* s) const;
};
using SamplerPtr = std::unique_ptr<llama_sampler, SamplerDeleter>;

/* Build a llama_sampler chain from the request's sampling parameters.
 * vocab + grammar_cfg are optional: pass nullptr/empty to skip grammar.
 * Grammar slot is inserted between penalties and top_k, per RFC §6.6.3.
 * logit_biases (phase 6) are applied after grammar, before top_k. */
SamplerPtr build_sampler_chain(const ChatRequest& req,
                                unsigned int seed_override = 0,
                                const llama_vocab* vocab = nullptr,
                                const std::string& grammar = "",
                                bool grammar_lazy = false,
                                const std::vector<GrammarTrigger>& grammar_triggers = {},
                                const std::vector<ResolvedBias>& logit_biases = {},
                                /* Reasoning budget (cap <think> length, then force the end tag).
                                 * < 0 disables it. Thinking tags + forced-open come from the
                                 * chat render (RenderResult). */
                                int reasoning_budget = -1,
                                const std::string& thinking_start_tag = "",
                                const std::string& thinking_end_tag = "",
                                bool thinking_forced_open = false);

/* Speculative-decoding verification: sample one token at each batch position
 * in `idxs`, accepting draft tokens that match the target's choice. Returns
 * the accepted token sequence (always length >= 1). If all `draft` tokens
 * match, a bonus token is appended (length == draft.size() + 1).
 *
 * `idxs` must have length draft.size() + 1. idxs[i] is the batch position
 * whose logits verify draft token i; idxs.back() verifies the bonus token.
 *
 * This is a raw-llama_sampler port of upstream's
 * common_sampler_sample_and_accept_n (common/sampling.cpp), avoiding the need
 * to convert Helix's sampler chain to common_sampler. */
std::vector<llama_token> helix_sample_and_accept_n(
    llama_sampler*            sampler,
    llama_context*            ctx,
    const std::vector<int>&   idxs,
    const std::vector<llama_token>& draft);

} // namespace helix
