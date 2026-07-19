#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace helix {

/* Maximum decoded image size (bytes) before we reject. Default 256 MiB. */
static constexpr size_t kMaxImageBytes    = size_t{256} * 1024 * 1024;

/* Maximum raw media payload size (encoded bytes) — 32 MiB matches OpenAI. */
static constexpr size_t kMaxMediaRawBytes = size_t{ 32} * 1024 * 1024;

/* Decode a data:image/<type>;base64,<payload> URI into raw image bytes.
 * Accepted types: png, jpeg, webp, gif.
 * Throws helix::Error (HELIX_E_VALIDATION) on any parse or decode error. */
std::vector<uint8_t> decode_image_data_uri(const std::string& uri);

/* Read a file:// URI into raw bytes.
 * Validates the path is absolute. Rejects non-existent or unreadable files.
 * max_bytes caps the allocation; throws HELIX_E_VALIDATION if exceeded. */
std::vector<uint8_t> load_file_uri(const std::string& uri,
                                    size_t max_bytes = kMaxMediaRawBytes);

/* Decode a base64-encoded audio payload (from input_audio.data) into raw audio bytes.
 * format must be "mp3" or "wav" (validated here; others rejected).
 * Returns raw compressed audio bytes — mtmd_helper_bitmap_init_from_buf decodes them. */
std::vector<uint8_t> decode_audio_base64(const std::string& b64_data,
                                          const std::string& format);

} // namespace helix
