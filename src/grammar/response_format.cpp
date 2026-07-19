#include "response_format.hpp"
#include "../internal/error.hpp"

#include "nlohmann/json.hpp"
#include "json-schema-to-grammar.h"

namespace helix {

// Embedded json.gbnf: any valid JSON document.
static const char kJsonObjectGbnf[] = R"GBNF(root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws

ws ::= | " " | "\n" [ \t]{0,20}
)GBNF";

ResponseFormatResolved resolve_response_format(
    const std::string& rf_type,
    const std::string& json_schema_str,
    bool strict)
{
    ResponseFormatResolved r;
    r.strict = strict;

    if (rf_type.empty() || rf_type == "text") {
        r.kind = ResponseFormatKind::Text;
        return r;
    }

    if (rf_type == "json_object") {
        r.kind = ResponseFormatKind::JsonObject;
        r.grammar_gbnf = kJsonObjectGbnf;
        return r;
    }

    if (rf_type == "json_schema") {
        r.kind = ResponseFormatKind::JsonSchema;

        if (json_schema_str.empty()) {
            throw_validation(
                "response_format.json_schema requires a 'schema' object",
                "response_format");
        }

        nlohmann::ordered_json js_obj;
        try {
            js_obj = nlohmann::ordered_json::parse(json_schema_str);
        } catch (const std::exception& e) {
            throw_validation(
                std::string("response_format.json_schema is not valid JSON: ") + e.what(),
                "response_format");
        }

        // The json_schema object contains {name?, description?, schema, strict?}.
        // Extract the actual schema. Accept direct schema objects too (some clients omit nesting).
        nlohmann::ordered_json schema;
        if (js_obj.contains("schema") && js_obj["schema"].is_object()) {
            schema = js_obj["schema"];
        } else if (js_obj.contains("type") || js_obj.contains("properties") ||
                   js_obj.contains("anyOf") || js_obj.contains("oneOf")) {
            schema = js_obj;
        } else {
            throw_validation(
                "response_format.json_schema must contain a 'schema' object",
                "response_format");
        }

        try {
            r.grammar_gbnf = json_schema_to_grammar(schema);
        } catch (const std::exception& e) {
            if (strict) {
                throw_validation(
                    std::string("schema-to-grammar conversion failed (strict=true): ") + e.what(),
                    "response_format");
            }
            r.warnings.push_back(
                std::string("schema-to-grammar conversion failed, falling back to "
                            "json_object grammar: ") + e.what());
            r.grammar_gbnf = kJsonObjectGbnf;
        }
        return r;
    }

    throw_validation(
        "response_format.type must be 'text', 'json_object', or 'json_schema'",
        "response_format");
}

} // namespace helix
