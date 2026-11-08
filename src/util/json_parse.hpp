#pragma once
#include <string>
#include "../internal/error.hpp"

namespace helix {

/* Parses `text` as the root object of a request body or options document.
 * Malformed input and non-object roots throw HELIX_E_INVALID_JSON, naming
 * the document as `what` ("request body", "session options", ...). A null
 * root yields an empty object when `null_is_empty` (options documents treat
 * it as "all defaults") and is rejected otherwise. */
template <typename Json>
Json parse_json_object(const std::string& text, const std::string& what,
                       bool null_is_empty) {
    Json j;
    try {
        j = Json::parse(text);
    } catch (const typename Json::parse_error& e) {
        throw_invalid_json(what + " JSON parse error: " + e.what());
    }
    if (j.is_null() && null_is_empty) return Json::object();
    if (!j.is_object()) throw_invalid_json(what + " must be a JSON object");
    return j;
}

} // namespace helix
