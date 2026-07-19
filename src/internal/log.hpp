#pragma once
#include "helix.h"
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <atomic>

namespace helix {

struct Logger {
    mutable std::mutex mu;
    helix_log_cb       cb       = nullptr;
    void*              userdata = nullptr;
    std::atomic<helix_log_level_t> min_level{HELIX_LOG_WARN};

    static Logger& instance();

    void log(helix_log_level_t level, const char* msg) const {
        // Snapshot fields under the lock, then invoke the callback outside it
        // so we never hold the lock while calling unknown user code.
        helix_log_cb      local_cb;
        void*             local_ud;
        helix_log_level_t local_min;
        {
            std::lock_guard<std::mutex> lk(mu);
            local_min = min_level.load(std::memory_order_relaxed);
            local_cb  = cb;
            local_ud  = userdata;
        }
        if (level > local_min) return;
        if (local_cb) {
            local_cb(local_ud, level, msg);
        } else {
            /* Default: stderr for ERROR/WARN, stdout for the rest. */
            FILE* f = (level <= HELIX_LOG_WARN) ? stderr : stdout;
            const char* lvl = "?";
            switch (level) {
                case HELIX_LOG_ERROR: lvl = "ERROR"; break;
                case HELIX_LOG_WARN:  lvl = "WARN";  break;
                case HELIX_LOG_INFO:  lvl = "INFO";  break;
                case HELIX_LOG_DEBUG: lvl = "DEBUG"; break;
                case HELIX_LOG_TRACE: lvl = "TRACE"; break;
                default: break;
            }
            fprintf(f, "[helix/%s] %s\n", lvl, msg);
        }
    }
};

inline void log_error(const char* msg) { Logger::instance().log(HELIX_LOG_ERROR, msg); }
inline void log_warn (const char* msg) { Logger::instance().log(HELIX_LOG_WARN,  msg); }
inline void log_info (const char* msg) { Logger::instance().log(HELIX_LOG_INFO,  msg); }
inline void log_debug(const char* msg) { Logger::instance().log(HELIX_LOG_DEBUG, msg); }
inline void log_trace(const char* msg) { Logger::instance().log(HELIX_LOG_TRACE, msg); }

/* Convenience wrappers that accept std::string. */
inline void log_error(const std::string& s) { log_error(s.c_str()); }
inline void log_warn (const std::string& s) { log_warn (s.c_str()); }
inline void log_info (const std::string& s) { log_info (s.c_str()); }
inline void log_debug(const std::string& s) { log_debug(s.c_str()); }
inline void log_trace(const std::string& s) { log_trace(s.c_str()); }

} // namespace helix
