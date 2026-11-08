#pragma once
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>

namespace helix {

/* Every string handed to the caller for release with helix_free() is
 * allocated here, with the C library's malloc(), because helix_free() calls
 * free(). Keep the two paired: crossing the malloc/new heaps (especially
 * across a mismatched Windows CRT) corrupts the heap. */

/* NUL-terminated copy of `s`, or nullptr when allocation fails. */
inline char* dup_for_caller_nothrow(const std::string& s) noexcept {
    char* p = static_cast<char*>(std::malloc(s.size() + 1));
    if (p) std::memcpy(p, s.c_str(), s.size() + 1);
    return p;
}

/* NUL-terminated copy of `s`; throws std::bad_alloc when allocation fails. */
inline char* dup_for_caller(const std::string& s) {
    char* p = dup_for_caller_nothrow(s);
    if (!p) throw std::bad_alloc();
    return p;
}

} // namespace helix
