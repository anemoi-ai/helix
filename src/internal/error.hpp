#pragma once
#include <stdexcept>
#include <string>
#include "helix.h"

namespace helix {

/* Internal exception type — caught at every extern "C" boundary. */
struct Error : std::exception {
    helix_status_t status;
    std::string    message;
    std::string    type;   /* OpenAI error type string */
    std::string    param;  /* offending field name, or "" */
    std::string    code;   /* machine-readable code string */

    Error(helix_status_t s,
          std::string msg,
          std::string t   = "internal_error",
          std::string p   = "",
          std::string c   = "helix_e_internal")
        : status(s), message(std::move(msg)), type(std::move(t)),
          param(std::move(p)), code(std::move(c)) {}

    const char* what() const noexcept override { return message.c_str(); }
};

void set_last_error(helix_status_t status,
                    const std::string& message,
                    const std::string& type  = "internal_error",
                    const std::string& param = "",
                    const std::string& code  = "helix_e_internal");

void set_last_error(const Error& e);
void clear_last_error();
const char* last_error_json_ptr();

/* Throw helpers */
[[noreturn]] inline void throw_invalid_arg(const std::string& msg, const std::string& param = "") {
    throw Error(HELIX_E_INVALID_ARG, msg, "invalid_request_error", param, "helix_e_invalid_arg");
}
[[noreturn]] inline void throw_invalid_json(const std::string& msg) {
    throw Error(HELIX_E_INVALID_JSON, msg, "invalid_request_error", "", "helix_e_invalid_json");
}
[[noreturn]] inline void throw_validation(const std::string& msg, const std::string& param = "") {
    throw Error(HELIX_E_VALIDATION, msg, "invalid_request_error", param, "helix_e_validation");
}
[[noreturn]] inline void throw_unsupported(const std::string& feature, const std::string& param = "") {
    throw Error(HELIX_E_UNSUPPORTED_FEATURE, feature + " is not supported in this build",
                "unsupported_feature_error", param, "helix_e_unsupported_feature");
}
[[noreturn]] inline void throw_context_full() {
    throw Error(HELIX_E_CONTEXT_FULL, "prompt + max_tokens exceeds context size",
                "context_length_exceeded", "max_tokens", "helix_e_context_full");
}
[[noreturn]] inline void throw_model_load_failed(const std::string& path) {
    throw Error(HELIX_E_MODEL_LOAD_FAILED, "failed to load model: " + path,
                "model_load_error", "model_path", "helix_e_model_load_failed");
}

} // namespace helix
