#pragma once
#include <algorithm>
#include <cstddef>
#include <vector>

namespace helix {

template <typename T>
inline size_t compute_lcp(const std::vector<T>& a, const std::vector<T>& b) {
    size_t lcp   = 0;
    size_t limit = std::min(a.size(), b.size());
    while (lcp < limit && a[lcp] == b[lcp]) ++lcp;
    return lcp;
}

} // namespace helix
