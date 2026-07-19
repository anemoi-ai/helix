#include "logit_bias.hpp"
#include "../internal/error.hpp"
#include "../internal/log.hpp"

#include "llama.h"

#include <cstdlib>
#include <string>

namespace helix {

/* logit_bias numeric keys are parsed as long long and narrowed to llama_token.
 * The range check below guarantees the value fits, but assert the token width
 * so the narrowing stays well-defined if llama_token ever changes type. */
static_assert(sizeof(llama_token) == 4, "logit_bias assumes 32-bit llama_token");

std::vector<ResolvedBias> resolve_logit_bias(
    const std::map<std::string, float>& raw,
    const llama_vocab* vocab)
{
    std::vector<ResolvedBias> result;
    if (!vocab || raw.empty()) return result;

    const int n_vocab = llama_vocab_n_tokens(vocab);

    for (const auto& [key, bias] : raw) {
        // Try to parse the key as an integer token ID.
        char* end = nullptr;
        long long id = std::strtoll(key.c_str(), &end, 10);
        if (*end == '\0' && end != key.c_str()) {
            // Pure integer key.
            if (id < 0 || id >= static_cast<long long>(n_vocab)) {
                throw_validation(
                    "logit_bias: token id " + key + " is out of range [0, " +
                    std::to_string(n_vocab) + ")",
                    "logit_bias");
            }
            result.push_back({static_cast<int32_t>(id), bias});
            continue;
        }

        // String key: tokenise it, apply bias to each resulting token.
        std::vector<llama_token> tokens(64);
        int n = llama_tokenize(vocab, key.c_str(), static_cast<int32_t>(key.size()),
                               tokens.data(), static_cast<int32_t>(tokens.size()),
                               /*add_special=*/false, /*parse_special=*/false);
        if (n < 0) {
            tokens.resize(static_cast<size_t>(-n));
            n = llama_tokenize(vocab, key.c_str(), static_cast<int32_t>(key.size()),
                               tokens.data(), static_cast<int32_t>(tokens.size()),
                               false, false);
        }
        if (n <= 0) {
            log_warn("logit_bias: key '" + key + "' did not tokenize, ignoring");
            continue;
        }
        tokens.resize(static_cast<size_t>(n));

        if (tokens.size() > 1) {
            log_debug("logit_bias: key '" + key + "' tokenized to " +
                      std::to_string(tokens.size()) + " tokens, biasing each");
        }
        for (llama_token t : tokens) {
            result.push_back({t, bias});
        }
    }
    return result;
}

} // namespace helix
