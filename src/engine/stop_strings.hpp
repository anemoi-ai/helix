#pragma once
#include <algorithm>
#include <string>
#include <vector>

namespace helix {

/* Find the earliest completed stop-string match in output[0, safe_len).
 * Returns the byte offset where the match begins (the truncation point),
 * or npos when no stop has fully matched yet.
 *
 * scan_from (optional): caller-maintained incremental scan offset. On entry
 * it must be the offset returned through it by the previous call (0 for the
 * first call); on return it is advanced past every position that has been
 * definitively checked, so repeated per-token calls do not rescan the whole
 * output. A match that ends beyond safe_len (incomplete UTF-8 tail) is left
 * for a later call. */
inline size_t check_stops(const std::string& output,
                           size_t safe_len,
                           const std::vector<std::string>& stops,
                           size_t* scan_from = nullptr) {
    const size_t start = scan_from ? *scan_from : 0;
    size_t best    = std::string::npos;
    size_t max_len = 0;
    for (const auto& stop : stops) {
        if (stop.empty()) continue;
        max_len = std::max(max_len, stop.size());
        /* Forward find: the FIRST occurrence. A later occurrence can never
         * complete within safe_len if an earlier one of the same stop
         * doesn't, so one find per stop suffices. */
        size_t pos = output.find(stop, start);
        if (pos != std::string::npos && pos + stop.size() <= safe_len) {
            best = std::min(best, pos);
        }
    }
    if (scan_from && max_len > 0) {
        /* Positions p with p + max_len <= safe_len are decided for every
         * stop; a still-pending (partial or past-safe) match must start at
         * safe_len - max_len + 1 or later. */
        const size_t decided = safe_len >= max_len ? safe_len - max_len + 1 : 0;
        *scan_from = std::max(*scan_from, decided);
    }
    return best;
}

/* Length of the longest suffix of output[0, safe_len) that is a proper
 * prefix of any stop string. These bytes must be held back from emission:
 * they may be the start of a stop match that completes on a later token,
 * and emitted content cannot be retracted. Held-back bytes that turn out
 * not to complete a stop are released by a later call or the final flush. */
inline size_t partial_stop_len(const std::string& output,
                                size_t safe_len,
                                const std::vector<std::string>& stops) {
    size_t held = 0;
    for (const auto& stop : stops) {
        if (stop.empty()) continue;
        const size_t k_max = std::min(stop.size() - 1, safe_len);
        for (size_t k = k_max; k > held; --k) {
            if (output.compare(safe_len - k, k, stop, 0, k) == 0) {
                held = k;
                break;
            }
        }
    }
    return held;
}

} // namespace helix
