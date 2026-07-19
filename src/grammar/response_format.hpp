#pragma once
#include <string>
#include <vector>

namespace helix {

enum class ResponseFormatKind { Text, JsonObject, JsonSchema };

struct ResponseFormatResolved {
    ResponseFormatKind       kind = ResponseFormatKind::Text;
    std::string              grammar_gbnf;   // empty for Text
    bool                     strict  = false;
    std::vector<std::string> warnings;
};

// Resolve a response_format request into a GBNF grammar string (empty for text mode).
// rf_type:         "text" | "json_object" | "json_schema"
// json_schema_str: raw JSON string of the json_schema sub-object (when type=="json_schema")
// strict:          strict flag — if true, unsupported schema features cause HELIX_E_VALIDATION
ResponseFormatResolved resolve_response_format(
    const std::string& rf_type,
    const std::string& json_schema_str,
    bool strict);

} // namespace helix
