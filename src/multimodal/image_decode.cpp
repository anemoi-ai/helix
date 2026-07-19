#include "image_decode.hpp"
#include "../internal/error.hpp"
#include "../util/strings.hpp"

#include "base64.hpp"    /* from llama.cpp/common */

#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#  include <sys/stat.h>
#  include <limits.h>
#  include <stdlib.h>
#endif

#if defined(_WIN32)
#  include <stdlib.h>   // _fullpath, _MAX_PATH
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>  // CreateFileW, GetFinalPathNameByHandleW
#endif

namespace helix {

#if defined(_WIN32)
/* UTF-8 → UTF-16 for Win32 wide-char APIs. Returns empty on failure (callers
 * treat an empty result for non-empty input as an error). */
static std::wstring widen_utf8(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    if (n <= 0) return {};
    std::wstring w(static_cast<size_t>(n) - 1, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    return w;
}
#endif

/* ------------------------------------------------------------------ */
/*  Data URI — image                                                   */
/* ------------------------------------------------------------------ */

std::vector<uint8_t> decode_image_data_uri(const std::string& uri) {
    /* Expected prefix: "data:image/<type>;base64," */
    static const std::string kPrefix = "data:image/";
    if (!starts_with(uri, kPrefix)) {
        throw_validation("image_url must start with 'data:image/' or 'file://'",
                         "messages");
    }

    /* Find ";base64," separator */
    const auto sep = uri.find(";base64,", kPrefix.size());
    if (sep == std::string::npos) {
        throw_validation(
            "data URI must use base64 encoding (expected ';base64,' separator)",
            "messages");
    }

    /* Validate media type */
    const std::string mime = uri.substr(kPrefix.size(), sep - kPrefix.size());
    if (mime != "png"  && mime != "jpeg" &&
        mime != "webp" && mime != "gif"  && mime != "bmp") {
        throw_validation(
            "unsupported image type '" + mime +
            "'; supported: png, jpeg, webp, gif, bmp",
            "messages");
    }

    const std::string b64 = uri.substr(sep + 8 /* len(";base64,") */);
    if (b64.empty()) {
        throw_validation("data URI has empty base64 payload", "messages");
    }

    std::string raw;
    try {
        raw = base64::decode(b64);
    } catch (const std::exception& e) {
        throw_validation(
            std::string("base64 decode failed: ") + e.what(), "messages");
    }

    if (raw.empty()) {
        throw_validation("image data URI decoded to zero bytes", "messages");
    }
    if (raw.size() > kMaxImageBytes) {
        throw_validation(
            "image exceeds max size of " +
            std::to_string(kMaxImageBytes / (1024 * 1024)) + " MiB",
            "messages");
    }

    return {raw.begin(), raw.end()};
}

/* ------------------------------------------------------------------ */
/*  File URI                                                           */
/* ------------------------------------------------------------------ */

std::vector<uint8_t> load_file_uri(const std::string& uri, size_t max_bytes) {
    static const std::string kFileScheme = "file://";
    /* `<=` (not `<`): a bare "file://" with no path is also invalid. */
    if (uri.size() <= kFileScheme.size() || !starts_with(uri, kFileScheme)) {
        throw_validation("file URI must start with 'file://'", "messages");
    }

    std::string path = uri.substr(kFileScheme.size());

    /* Percent-decode common escapes (%20 → space, etc.) */
    std::string decoded;
    decoded.reserve(path.size());
    for (size_t i = 0; i < path.size(); ++i) {
        if (path[i] == '%' && i + 2 < path.size()) {
            char hex[3] = { path[i+1], path[i+2], '\0' };
            char* end;
            long val = std::strtol(hex, &end, 16);
            if (end == hex + 2) {
                decoded += static_cast<char>(val);
                i += 2;
                continue;
            }
        }
        decoded += path[i];
    }
    path = std::move(decoded);

    // Absolute-path requirement. Windows drive/UNC paths are accepted on
    // Windows; POSIX absolute paths ("/...") everywhere else.
    const bool is_absolute =
#ifdef _WIN32
        /* Drive paths need a separator after the colon: "C:foo" is
         * drive-RELATIVE (resolves against the drive's current directory). */
        (path.size() >= 3 && path[1] == ':' &&
         (path[2] == '/' || path[2] == '\\')) ||
        (path.size() >= 2 && path[0] == '\\' && path[1] == '\\');
#else
        !path.empty() && path[0] == '/';
#endif
    if (!is_absolute) {
        throw_validation(
            "file:// URI must use an absolute path (got: " + path + ")",
            "messages");
    }

    // Reject '..' path components to prevent path traversal attacks.
    if (path.find("/../") != std::string::npos ||
        (path.size() >= 3 && path.compare(path.size() - 3, 3, "/..") == 0)) {
        throw_validation(
            "file:// URI must not contain '..' path traversal components",
            "messages");
    }

#if defined(__unix__) || defined(__APPLE__)
    char* resolved = realpath(path.c_str(), nullptr);
    if (!resolved) {
        throw_validation("cannot resolve file path: " + path, "messages");
    }
    std::string canonical_path(resolved);
    free(resolved);

    static const std::string kAllowedBase = []() -> std::string {
        const char* env = std::getenv("HELIX_FILE_URI_ROOT");
        if (!env || env[0] == '\0') return {};
        std::string s(env);
        if (s.back() != '/') s += '/';
        return s;
    }();
    if (!kAllowedBase.empty()) {
        if (canonical_path != kAllowedBase.substr(0, kAllowedBase.size() - 1) &&
            !starts_with(canonical_path, kAllowedBase)) {
            throw_validation(
                "file:// URI resolves outside allowed base directory",
                "messages");
        }
    }
    path = std::move(canonical_path);
#elif defined(_WIN32)
    // Canonicalise via _fullpath (resolves "."/".." lexically), open the
    // file, then enforce the HELIX_FILE_URI_ROOT jail on the handle's FINAL
    // path (GetFinalPathNameByHandleW resolves junctions and symlinks — a
    // reparse point placed inside the allowed root must not escape it, which
    // pure lexical checks cannot guarantee). Reading through the same handle
    // that was jail-checked also avoids a check/open race. The prefix compare
    // is case-insensitive to match the filesystem.
    char resolved[_MAX_PATH];
    if (!_fullpath(resolved, path.c_str(), _MAX_PATH)) {
        throw_validation("cannot resolve file path: " + path, "messages");
    }

    const std::wstring wide_path = widen_utf8(resolved);
    if (wide_path.empty()) {
        throw_validation("cannot resolve file path: " + path, "messages");
    }

    struct HandleGuard {
        HANDLE h;
        ~HandleGuard() { if (h != INVALID_HANDLE_VALUE) CloseHandle(h); }
    } guard{CreateFileW(wide_path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                        nullptr)};
    if (guard.h == INVALID_HANDLE_VALUE) {
        throw_validation("cannot open file: " + path, "messages");
    }

    static const std::wstring kAllowedBase = []() -> std::wstring {
        const char* env = std::getenv("HELIX_FILE_URI_ROOT");
        if (!env || env[0] == '\0') return {};
        std::wstring w = widen_utf8(env);
        for (auto& c : w) {
            if (c == L'/') c = L'\\';
        }
        if (!w.empty() && w.back() != L'\\') w += L'\\';
        return w;
    }();
    if (!kAllowedBase.empty()) {
        DWORD need = GetFinalPathNameByHandleW(guard.h, nullptr, 0,
                                               FILE_NAME_NORMALIZED);
        if (need == 0) {
            throw_validation("cannot resolve final file path: " + path,
                             "messages");
        }
        std::wstring final_path(need, L'\0');
        DWORD len = GetFinalPathNameByHandleW(guard.h, &final_path[0], need,
                                              FILE_NAME_NORMALIZED);
        if (len == 0 || len >= need) {
            throw_validation("cannot resolve final file path: " + path,
                             "messages");
        }
        final_path.resize(len);
        /* Strip the extended-length prefix: "\\?\C:\..." / "\\?\UNC\srv\..." */
        if (final_path.rfind(LR"(\\?\UNC\)", 0) == 0) {
            final_path = L"\\\\" + final_path.substr(8);
        } else if (final_path.rfind(LR"(\\?\)", 0) == 0) {
            final_path = final_path.substr(4);
        }
        const bool inside =
            final_path.size() >= kAllowedBase.size() &&
            CompareStringOrdinal(final_path.c_str(),
                                 static_cast<int>(kAllowedBase.size()),
                                 kAllowedBase.c_str(),
                                 static_cast<int>(kAllowedBase.size()),
                                 /*bIgnoreCase=*/TRUE) == CSTR_EQUAL;
        if (!inside) {
            throw_validation(
                "file:// URI resolves outside allowed base directory",
                "messages");
        }
    }

    /* Read via the already-vetted handle. */
    LARGE_INTEGER fsize;
    if (!GetFileSizeEx(guard.h, &fsize) || fsize.QuadPart < 0) {
        throw_validation("cannot stat file: " + path, "messages");
    }
    if (static_cast<uint64_t>(fsize.QuadPart) > max_bytes) {
        throw_validation(
            "file exceeds max size of " +
            std::to_string(max_bytes / (1024 * 1024)) + " MiB: " + path,
            "messages");
    }
    std::vector<uint8_t> wbuf(static_cast<size_t>(fsize.QuadPart));
    size_t off = 0;
    while (off < wbuf.size()) {
        const DWORD chunk = static_cast<DWORD>(
            std::min<size_t>(wbuf.size() - off, 1u << 20));
        DWORD got = 0;
        if (!ReadFile(guard.h, wbuf.data() + off, chunk, &got, nullptr) ||
            got == 0) {
            throw_validation("failed to read file: " + path, "messages");
        }
        off += got;
    }
    return wbuf;
#endif

#if !defined(_WIN32)
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw_validation("cannot open file: " + path, "messages");
    }

    const std::streamsize size = f.tellg();
    if (size < 0) {
        throw_validation("cannot stat file: " + path, "messages");
    }
    if (static_cast<size_t>(size) > max_bytes) {
        throw_validation(
            "file exceeds max size of " +
            std::to_string(max_bytes / (1024 * 1024)) + " MiB: " + path,
            "messages");
    }

    f.seekg(0);
    std::vector<uint8_t> buf(static_cast<size_t>(size));
    if (!f.read(reinterpret_cast<char*>(buf.data()), size)) {
        throw_validation("failed to read file: " + path, "messages");
    }
    return buf;
#endif
}

/* ------------------------------------------------------------------ */
/*  Audio base64                                                       */
/* ------------------------------------------------------------------ */

std::vector<uint8_t> decode_audio_base64(const std::string& b64_data,
                                          const std::string& format) {
    if (format != "mp3" && format != "wav") {
        throw_validation(
            "unsupported audio format '" + format + "'; supported: mp3, wav",
            "messages");
    }
    if (b64_data.empty()) {
        throw_validation("input_audio.data is empty", "messages");
    }

    static constexpr size_t kMaxAudio = 25u * 1024u * 1024u; /* 25 MiB */

    std::string raw;
    try {
        raw = base64::decode(b64_data);
    } catch (const std::exception& e) {
        throw_validation(
            std::string("audio base64 decode failed: ") + e.what(), "messages");
    }

    if (raw.size() > kMaxAudio) {
        throw_validation("audio payload exceeds 25 MiB limit", "messages");
    }

    return {raw.begin(), raw.end()};
}

} // namespace helix
