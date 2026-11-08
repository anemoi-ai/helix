#pragma once
#include <string>

namespace helix {

/* Serialise a nlohmann::json / ordered_json value, replacing invalid UTF-8
 * with U+FFFD instead of throwing. Model output (partial-UTF-8 token pieces
 * in logprobs), hardware strings, and parse-error messages that echo the
 * offending input bytes are not guaranteed to be valid UTF-8, and the
 * default strict handler throws type_error 316 — which, on the error path,
 * would escape an extern "C" boundary. */
template <typename Json>
inline std::string dump_json(const Json& j, int indent = -1) {
    return j.dump(indent, ' ', false, Json::error_handler_t::replace);
}

} // namespace helix
