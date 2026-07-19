#pragma once
#include <cstddef>

namespace helix {

/* Returns the length of the longest valid UTF-8 prefix in [data, data+len).
 * Scans back at most 3 bytes from the end to detect a truncated multi-byte
 * sequence and excludes it from the safe prefix. */
inline size_t valid_utf8_prefix_len(const char* data, size_t len) {
    if (len == 0) return 0;
    size_t i = len;
    size_t check = (len >= 3) ? len - 3 : 0;
    for (size_t j = check; j < len; ++j) {
        unsigned char c = static_cast<unsigned char>(data[j]);
        bool is_lead = (c < 0x80) || (c >= 0xC2 && c <= 0xF4);
        if (is_lead) {
            int total;
            if      (c < 0x80) total = 1;
            else if (c < 0xE0) total = 2;
            else if (c < 0xF0) total = 3;
            else               total = 4;

            size_t end = j + total;
            if (end > len) {
                i = j;
                break;
            }
            bool ok = true;
            for (int k = 1; k < total; ++k) {
                if ((static_cast<unsigned char>(data[j + k]) & 0xC0) != 0x80) {
                    ok = false; break;
                }
            }
            if (ok) i = j + total;
        }
    }
    return i;
}

} // namespace helix
