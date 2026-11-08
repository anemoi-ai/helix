#include "error.hpp"
#include <cstring>
#include <string>

/* nlohmann/json — JSON-only TU, never leaks into headers. */
#include "nlohmann/json.hpp"
#include "../util/json_dump.hpp"

namespace helix {

static thread_local std::string g_last_error;
/* Set when building the envelope itself failed (allocation): the error
 * path runs inside HELIX_CATCH, so it must never throw. */
static thread_local bool g_last_error_oom = false;
static const char kOomErrorJson[] =
    "{\"error\":{\"message\":\"out of memory\",\"type\":\"memory_error\","
    "\"param\":null,\"code\":\"helix_e_oom\"}}";

static std::string make_error_json(helix_status_t /*status*/,
                                   const std::string& message,
                                   const std::string& type,
                                   const std::string& param,
                                   const std::string& code,
                                   int64_t requested,
                                   int64_t limit) {
    nlohmann::json err = {
        {"message", message},
        {"type",    type},
        {"param",   param.empty() ? nlohmann::json(nullptr) : nlohmann::json(param)},
        {"code",    code}
    };
    /* Context-length detail (helix ABI 1.7): additive, omitted when unknown. */
    if (requested >= 0) err["requested"] = requested;
    if (limit     >= 0) err["limit"]     = limit;
    nlohmann::json j = {{"error", std::move(err)}};
    /* Compact (un-indented) dump: the OpenAI error envelope is a wire format,
     * not human-facing output. Keep this compact even if the *_describe paths
     * switch to j.dump(2) for readability. */
    return dump_json(j);
}

void set_last_error(helix_status_t status,
                    const std::string& message,
                    const std::string& type,
                    const std::string& param,
                    const std::string& code) {
    try {
        g_last_error = make_error_json(status, message, type, param, code, -1, -1);
        g_last_error_oom = false;
    } catch (...) {
        g_last_error.clear();
        g_last_error_oom = true;
    }
}

void set_last_error(const Error& e) {
    try {
        g_last_error = make_error_json(e.status, e.message, e.type, e.param,
                                       e.code, e.requested, e.limit);
        g_last_error_oom = false;
    } catch (...) {
        g_last_error.clear();
        g_last_error_oom = true;
    }
}

void clear_last_error() {
    g_last_error.clear();
    g_last_error_oom = false;
}

const char* last_error_json_ptr() {
    if (g_last_error_oom) return kOomErrorJson;
    return g_last_error.empty() ? "{}" : g_last_error.c_str();
}

} // namespace helix
