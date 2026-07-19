#include <gtest/gtest.h>
#include "src/internal/error.hpp"
#include "nlohmann/json.hpp"

using namespace helix;
using json = nlohmann::json;

TEST(ErrorEnvelope, SetAndRead) {
    set_last_error(HELIX_E_OOM, "out of memory", "memory_error", "", "helix_e_oom");
    const char* s = last_error_json_ptr();
    ASSERT_TRUE(s != nullptr);
    auto j = json::parse(s);

    EXPECT_TRUE(j.contains("error"));
    EXPECT_EQ(j["error"]["message"], "out of memory");
    EXPECT_EQ(j["error"]["type"],    "memory_error");
    EXPECT_EQ(j["error"]["code"],    "helix_e_oom");
}

TEST(ErrorEnvelope, EmptyAfterClear) {
    set_last_error(HELIX_E_INTERNAL, "test", "x", "", "y");
    clear_last_error();
    EXPECT_STREQ(last_error_json_ptr(), "{}");
}

TEST(ErrorEnvelope, ParamNullWhenEmpty) {
    set_last_error(HELIX_E_BACKEND, "fail", "backend_error", "", "code");
    auto j = json::parse(last_error_json_ptr());
    EXPECT_TRUE(j["error"]["param"].is_null());
}

TEST(ErrorEnvelope, ParamPopulatedWhenSet) {
    set_last_error(HELIX_E_VALIDATION, "bad field", "invalid_request_error",
                   "temperature", "helix_e_validation");
    auto j = json::parse(last_error_json_ptr());
    EXPECT_EQ(j["error"]["param"], "temperature");
}

TEST(ErrorEnvelope, AllStatusCodesHaveCode) {
    const struct { helix_status_t st; const char* expected_code; } cases[] = {
        {HELIX_E_INVALID_ARG,         "helix_e_invalid_arg"},
        {HELIX_E_INVALID_JSON,        "helix_e_invalid_json"},
        {HELIX_E_VALIDATION,          "helix_e_validation"},
        {HELIX_E_MODEL_NOT_FOUND,     "helix_e_model_not_found"},
        {HELIX_E_MODEL_LOAD_FAILED,   "helix_e_model_load_failed"},
        {HELIX_E_OOM,                 "helix_e_oom"},
        {HELIX_E_VRAM_EXHAUSTED,      "helix_e_vram_exhausted"},
        {HELIX_E_CONTEXT_FULL,        "helix_e_context_full"},
        {HELIX_E_CANCELLED,           "helix_e_cancelled"},
        {HELIX_E_BACKEND,             "helix_e_backend"},
        {HELIX_E_UNSUPPORTED_FEATURE, "helix_e_unsupported_feature"},
        {HELIX_E_INTERNAL,            "helix_e_internal"},
    };
    for (const auto& c : cases) {
        set_last_error(c.st, "msg", "type", "", "explicit_code");
        auto j = json::parse(last_error_json_ptr());
        EXPECT_EQ(j["error"]["code"], "explicit_code")
            << "status=" << c.st << " expected_code=" << c.expected_code;
        EXPECT_NE(j["error"]["code"], c.expected_code)
            << "code field should reflect what was set, not derive from status";
    }
}

/* Verify the Error exception type is correctly wired. */
TEST(ErrorException, ThrowAndCatch) {
    helix_status_t captured = HELIX_OK;
    try {
        throw_unsupported("streaming", "stream");
    } catch (const Error& e) {
        captured = e.status;
        EXPECT_EQ(e.param, "stream");
    }
    EXPECT_EQ(captured, HELIX_E_UNSUPPORTED_FEATURE);
}
