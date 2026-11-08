#pragma once
#include <filesystem>
#include <string>

namespace helix {

/* Paths cross the ABI as UTF-8. std::filesystem::path(std::string) decodes
 * narrow strings with the active ANSI code page on Windows, so a non-ASCII
 * UTF-8 path would name a different (usually nonexistent) file there.
 * u8path is the C++17 spelling of an explicit UTF-8 decode. */
inline std::filesystem::path utf8_path(const std::string& s) {
    return std::filesystem::u8path(s);
}

} // namespace helix
