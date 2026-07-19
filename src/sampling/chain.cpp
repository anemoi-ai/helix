#include "chain.hpp"
#include "../internal/log.hpp"
#include "../json/request.hpp"

#include "llama.h"
#include "common.h"
#include "reasoning-budget.h"

#include <regex>
#include <string>
#include <vector>

namespace helix {

/* Local copy of llama.cpp's regex_escape() so we don't depend on its
 * presence in common.h across llama.cpp version updates. */
static std::string regex_escape(const std::string& s) {
    static const std::regex kSpecial(R"([.^$|()*+?\[\]{}\\])");
    return std::regex_replace(s, kSpecial, "\\$&");
}

void SamplerDeleter::operator()(llama_sampler* s) const {
    if (s) llama_sampler_free(s);
}

SamplerPtr build_sampler_chain(const ChatRequest& req,
                                unsigned int seed_override,
                                const llama_vocab* vocab,
                                const std::string& grammar,
                                bool grammar_lazy,
                                const std::vector<GrammarTrigger>& grammar_triggers,
                                const std::vector<ResolvedBias>& logit_biases,
                                int reasoning_budget,
                                const std::string& thinking_start_tag,
                                const std::string& thinking_end_tag,
                                bool thinking_forced_open) {
    const float temperature       = req.temperature.value_or(1.0f);
    const float top_p             = req.top_p.value_or(1.0f);
    const int   top_k             = req.top_k.value_or(40);
    const float min_p             = req.min_p.value_or(0.05f);
    /* Default 1.0 = disabled: an all-default request must produce the same
     * logits an OpenAI-compatible backend would (no repetition penalty unless
     * asked for). The repeat_penalty request field is a Helix extension. */
    const float repeat_penalty    = req.repeat_penalty.value_or(1.0f);
    const float presence_penalty  = req.presence_penalty.value_or(0.0f);
    const float frequency_penalty = req.frequency_penalty.value_or(0.0f);

    static constexpr int32_t kDefaultPenaltyWindow = 64;

    uint32_t seed;
    if (seed_override != 0) {
        seed = seed_override;
    } else if (req.seed) {
        seed = *req.seed;
    } else {
        seed = LLAMA_DEFAULT_SEED;
    }

    auto chain_params = llama_sampler_chain_default_params();
    llama_sampler* chain = llama_sampler_chain_init(chain_params);

    /* Order follows §6.6.3 of the RFC:
     *   penalties → grammar → logit_bias → top_k → top_p → min_p → temp → dist */

    /* 1. Penalties */
    llama_sampler_chain_add(chain,
        llama_sampler_init_penalties(
            kDefaultPenaltyWindow,
            repeat_penalty,
            /*penalty_freq=*/    frequency_penalty,
            /*penalty_present=*/ presence_penalty));

    /* 2. Grammar slot (Phase 3) */
    if (vocab && !grammar.empty()) {
        if (!grammar_lazy || grammar_triggers.empty()) {
            /* Eager grammar: active from token 0 (tool_choice: "required") */
            llama_sampler_chain_add(chain,
                llama_sampler_init_grammar(vocab, grammar.c_str(), "root"));
        } else {
            /* Lazy grammar: activates on trigger pattern (tool_choice: "auto") */
            std::vector<std::string> pattern_strs;
            std::vector<const char*> patterns;
            std::vector<llama_token> tokens;

            for (const auto& trig : grammar_triggers) {
                /* Mirror the logic in llama.cpp/common/sampling.cpp:
                 * 0=Token, 1=Word (regex-escaped literal), 2=Pattern, 3=PatternFull */
                switch (trig.type) {
                    case 0: /* Token */
                        if (trig.token != -1)
                            tokens.push_back(static_cast<llama_token>(trig.token));
                        break;
                    case 1: /* Word — must be regex-escaped */
                        pattern_strs.push_back(regex_escape(trig.value));
                        break;
                    case 2: /* Pattern — use as-is */
                        pattern_strs.push_back(trig.value);
                        break;
                    case 3: /* PatternFull — anchor to full match */
                    {
                        const auto& p = trig.value;
                        std::string anchored = "^$";
                        if (!p.empty()) {
                            anchored = (p.front() != '^' ? "^" : "")
                                     + p
                                     + (p.back()  != '$' ? "$" : "");
                        }
                        pattern_strs.push_back(anchored);
                        break;
                    }
                    default: break;
                }
            }
            for (const auto& s : pattern_strs) patterns.push_back(s.c_str());

            llama_sampler_chain_add(chain,
                llama_sampler_init_grammar_lazy_patterns(
                    vocab, grammar.c_str(), "root",
                    patterns.empty()  ? nullptr : patterns.data(),  patterns.size(),
                    tokens.empty()    ? nullptr : tokens.data(),    tokens.size()));
        }
    }

    /* 3. Logit bias (phase 6) — applied after grammar, before stochastic truncation. */
    if (vocab && !logit_biases.empty()) {
        std::vector<llama_logit_bias> biases;
        biases.reserve(logit_biases.size());
        for (const auto& rb : logit_biases)
            biases.push_back({rb.token, rb.bias});
        llama_sampler_chain_add(chain,
            llama_sampler_init_logit_bias(
                llama_vocab_n_tokens(vocab),
                static_cast<int32_t>(biases.size()),
                biases.data()));
    }

    /* 3a. Reasoning budget (Helix extension) — cap the <think> block at
     * `reasoning_budget` tokens, then force the closing tag so the model moves
     * on to its answer. Passthrough until the budget is hit, so non-budgeted
     * calls (reasoning_budget < 0) skip it entirely. */
    if (vocab && reasoning_budget >= 0 && !thinking_end_tag.empty()) {
        /* When the template supplies no start tag and the prompt is not
         * forced-open, an empty start matcher would leave the sampler IDLE
         * forever and silently ignore the budget — yet the model may well
         * emit the tag on its own. Derive the opening tag from the end tag
         * ("</think>" -> "<think>") as a fallback so the budget stays live. */
        std::string start_tag = thinking_start_tag;
        if (start_tag.empty() && !thinking_forced_open) {
            const auto slash = thinking_end_tag.find('/');
            if (slash != std::string::npos) {
                start_tag = thinking_end_tag;
                start_tag.erase(slash, 1);
                log_debug("reasoning_budget: template supplied no thinking start "
                          "tag; watching for \"" + start_tag + "\" derived from "
                          "the end tag");
            } else {
                log_warn("reasoning_budget: no thinking start tag is available "
                         "and none can be derived from the end tag \"" +
                         thinking_end_tag + "\"; the budget will only apply if "
                         "the prompt pre-opens a thinking block");
            }
        }
        const auto start_tokens = start_tag.empty()
            ? std::vector<llama_token>{}
            : common_tokenize(vocab, start_tag, false, true);
        const auto end_tokens = common_tokenize(vocab, thinking_end_tag, false, true);
        /* When the budget expires, inject a short transition message before the
         * closing tag so the model cleanly hands off from interrupted thinking to
         * its answer. Forcing the bare end tag alone can leave the model with no
         * formed conclusion, yielding an empty reply. */
        static const std::string kBudgetMessage =
            "\n\nI've thought enough; let me give my answer.\n";
        const auto forced_tokens =
            common_tokenize(vocab, kBudgetMessage + thinking_end_tag, false, true);
        /* Qwen-style templates pre-open <think> in the prompt, so we are already
         * inside the block at token 0 — start counting immediately. Otherwise
         * watch for the opening tag. */
        const auto initial_state = thinking_forced_open
            ? REASONING_BUDGET_COUNTING
            : REASONING_BUDGET_IDLE;
        llama_sampler* rbudget = common_reasoning_budget_init(
            vocab, start_tokens, end_tokens, forced_tokens, reasoning_budget, initial_state);
        if (rbudget) llama_sampler_chain_add(chain, rbudget);
    }

    if (temperature == 0.0f) {
        /* Greedy: collapse the whole tail. */
        if (req.top_k || req.top_p || req.min_p) {
            log_debug("temperature=0 selects greedy sampling; "
                      "top_k/top_p/min_p are ignored");
        }
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    } else {
        /* 4. top_k */
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
        /* 5. top_p */
        if (top_p < 1.0f) {
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, /*min_keep=*/1));
        }
        /* 6. min_p */
        if (min_p > 0.0f) {
            llama_sampler_chain_add(chain, llama_sampler_init_min_p(min_p, /*min_keep=*/1));
        }
        /* 7. temperature */
        llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature));
        /* 8. distribution sampler (applies the seed) */
        llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
    }

    return SamplerPtr(chain);
}

std::vector<llama_token> helix_sample_and_accept_n(
        llama_sampler*            sampler,
        llama_context*            ctx,
        const std::vector<int>&   idxs,
        const std::vector<llama_token>& draft) {
    /* Port of common_sampler_sample_and_accept_n (common/sampling.cpp) over
     * the raw llama_sampler chain. idxs.size() == draft.size() + 1. */
    std::vector<llama_token> result;
    result.reserve(idxs.size());
    size_t i = 0;
    for (; i < draft.size(); ++i) {
        llama_token id = llama_sampler_sample(sampler, ctx, idxs[i]);
        llama_sampler_accept(sampler, id);
        result.push_back(id);
        if (draft[i] != id) break;       /* mismatch - reject the rest */
    }
    if (i == draft.size()) {             /* all drafts matched - sample bonus */
        llama_token id = llama_sampler_sample(sampler, ctx, idxs[i]);
        llama_sampler_accept(sampler, id);
        result.push_back(id);
    }
    return result;
}

} // namespace helix
