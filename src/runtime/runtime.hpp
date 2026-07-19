#pragma once
#include <string>
#include "helix.h"
#include "../internal/log.hpp"
#include "../hardware/cpu.hpp"

namespace helix {

struct RuntimeOptions {
    std::string device_preference = "auto";
    helix_log_level_t log_level   = HELIX_LOG_WARN;
    bool deterministic            = false;
};

class Runtime {
public:
    explicit Runtime(const RuntimeOptions& opts);
    ~Runtime();

    const RuntimeOptions&  options() const { return opts_; }
    const std::string&     describe() const { return describe_json_; }
    const HardwareProfile& profile() const { return profile_; }

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

private:
    RuntimeOptions  opts_;
    HardwareProfile profile_;
    std::string     describe_json_;

    void build_describe();
};

RuntimeOptions parse_runtime_options(const char* options_json);

} // namespace helix
