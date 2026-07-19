#pragma once
#include <string_view>

namespace helix {

/* C++17 stand-in for C++20 std::string_view::starts_with. */
inline bool starts_with(std::string_view s, std::string_view prefix) {
    return s.size() >= prefix.size() &&
           s.compare(0, prefix.size(), prefix) == 0;
}

} // namespace helix
