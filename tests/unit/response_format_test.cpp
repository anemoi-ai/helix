#include <gtest/gtest.h>
#include "src/grammar/response_format.hpp"
#include "src/internal/error.hpp"

using namespace helix;

TEST(ResponseFormat, TextReturnsEmpty) {
    auto r = resolve_response_format("text", "", false);
    EXPECT_EQ(r.kind, ResponseFormatKind::Text);
    EXPECT_TRUE(r.grammar_gbnf.empty());
}

TEST(ResponseFormat, EmptyTypeIsText) {
    auto r = resolve_response_format("", "", false);
    EXPECT_EQ(r.kind, ResponseFormatKind::Text);
}

TEST(ResponseFormat, JsonObjectGrammarContainsRoot) {
    auto r = resolve_response_format("json_object", "", false);
    EXPECT_EQ(r.kind, ResponseFormatKind::JsonObject);
    EXPECT_FALSE(r.grammar_gbnf.empty());
    EXPECT_NE(r.grammar_gbnf.find("root"), std::string::npos);
    EXPECT_NE(r.grammar_gbnf.find("object"), std::string::npos);
}

TEST(ResponseFormat, JsonSchemaSimpleRecord) {
    const std::string schema_json = R"({
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age":  {"type": "integer"}
            },
            "required": ["name", "age"],
            "additionalProperties": false
        }
    })";
    auto r = resolve_response_format("json_schema", schema_json, false);
    EXPECT_EQ(r.kind, ResponseFormatKind::JsonSchema);
    EXPECT_FALSE(r.grammar_gbnf.empty());
    EXPECT_TRUE(r.warnings.empty());
}

TEST(ResponseFormat, JsonSchemaDirectObject) {
    // Some clients send the schema directly (without the nested "schema" key).
    const std::string schema_json = R"({
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "additionalProperties": false
    })";
    auto r = resolve_response_format("json_schema", schema_json, false);
    EXPECT_EQ(r.kind, ResponseFormatKind::JsonSchema);
    EXPECT_FALSE(r.grammar_gbnf.empty());
}

TEST(ResponseFormat, JsonSchemaEmptyBodyThrows) {
    EXPECT_THROW(resolve_response_format("json_schema", "", false), helix::Error);
}

TEST(ResponseFormat, InvalidTypeThrows) {
    EXPECT_THROW(resolve_response_format("xml", "", false), helix::Error);
}

TEST(ResponseFormat, JsonSchemaStrictFlagPropagated) {
    const std::string schema_json = R"({
        "schema": {"type": "object", "additionalProperties": false}
    })";
    auto r = resolve_response_format("json_schema", schema_json, true);
    EXPECT_TRUE(r.strict);
}

TEST(ResponseFormat, JsonSchemaBadJsonThrows) {
    EXPECT_THROW(resolve_response_format("json_schema", "{not valid json", false), helix::Error);
}

TEST(ResponseFormat, JsonSchemaFallbackOnBadSchema) {
    auto r = resolve_response_format("json_schema",
        R"({"schema":{"type":"object","properties":{"x":{"type":"recursive_self_ref"}}}})", false);
    EXPECT_EQ(r.kind, ResponseFormatKind::JsonSchema);
    EXPECT_FALSE(r.grammar_gbnf.empty());
    if (!r.warnings.empty()) {
        EXPECT_NE(r.warnings[0].find("falling back"), std::string::npos);
    }
}

TEST(ResponseFormat, JsonSchemaStrictBadSchemaThrows) {
    EXPECT_THROW(
        resolve_response_format("json_schema", R"({"schema":123})", true),
        helix::Error);
}

TEST(ResponseFormat, JsonSchemaWithNestedSchemaKey) {
    const std::string schema_json = R"({
        "name": "test",
        "schema": {"type": "object", "properties": {"x": {"type": "string"}}, "additionalProperties": false}
    })";
    auto r = resolve_response_format("json_schema", schema_json, false);
    EXPECT_EQ(r.kind, ResponseFormatKind::JsonSchema);
    EXPECT_FALSE(r.grammar_gbnf.empty());
    EXPECT_TRUE(r.warnings.empty());
}

TEST(ResponseFormat, JsonSchemaMissingSchemaKeyThrows) {
    const std::string schema_json = R"({"name": "test"})";
    EXPECT_THROW(resolve_response_format("json_schema", schema_json, false), helix::Error);
}

TEST(ResponseFormat, JsonSchemaWithAnyOfDirect) {
    const std::string schema_json = R"({
        "anyOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    })";
    auto r = resolve_response_format("json_schema", schema_json, false);
    EXPECT_EQ(r.kind, ResponseFormatKind::JsonSchema);
    EXPECT_FALSE(r.grammar_gbnf.empty());
}

TEST(ResponseFormat, JsonSchemaWithOneOfDirect) {
    const std::string schema_json = R"({
        "oneOf": [
            {"type": "string"},
            {"type": "null"}
        ]
    })";
    auto r = resolve_response_format("json_schema", schema_json, false);
    EXPECT_EQ(r.kind, ResponseFormatKind::JsonSchema);
    EXPECT_FALSE(r.grammar_gbnf.empty());
}
