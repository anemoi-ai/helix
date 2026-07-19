#include "runtime.hpp"
#include "../hardware/profile.hpp"
#include "../internal/error.hpp"
#include "../internal/version.hpp"
#include "llama.h"
#include "nlohmann/json.hpp"

#include <mutex>

using nj = nlohmann::json;

static std::mutex s_backend_init_mu;
static int s_runtime_count = 0;

namespace helix {

RuntimeOptions parse_runtime_options(const char* options_json) {
    RuntimeOptions opts;
    if (!options_json || options_json[0] == '\0') return opts;

    nj j;
    try {
        j = nj::parse(options_json);
    } catch (const nj::parse_error& e) {
        throw_invalid_json(std::string("runtime options JSON parse error: ") + e.what());
    }

    if (j.contains("device_preference") && j["device_preference"].is_string())
        opts.device_preference = j["device_preference"].get<std::string>();

    if (j.contains("log_level") && j["log_level"].is_string()) {
        const auto& lvl = j["log_level"].get<std::string>();
        if      (lvl == "off")   opts.log_level = HELIX_LOG_OFF;
        else if (lvl == "error") opts.log_level = HELIX_LOG_ERROR;
        else if (lvl == "warn")  opts.log_level = HELIX_LOG_WARN;
        else if (lvl == "info")  opts.log_level = HELIX_LOG_INFO;
        else if (lvl == "debug") opts.log_level = HELIX_LOG_DEBUG;
        else if (lvl == "trace") opts.log_level = HELIX_LOG_TRACE;
        else {
            log_warn("unknown log_level '" + lvl + "'; ignoring");
        }
    }

    if (j.contains("deterministic") && j["deterministic"].is_boolean())
        opts.deterministic = j["deterministic"].get<bool>();

    return opts;
}

static void llama_log_redirect(ggml_log_level level, const char* text, void* /*userdata*/) {
    if (!text || text[0] == '\0') return;

    std::string msg(text);
    if (!msg.empty() && msg.back() == '\n') msg.pop_back();

    helix_log_level_t hl;
    switch (level) {
        case GGML_LOG_LEVEL_ERROR: hl = HELIX_LOG_ERROR; break;
        case GGML_LOG_LEVEL_WARN:  hl = HELIX_LOG_WARN;  break;
        case GGML_LOG_LEVEL_INFO:  hl = HELIX_LOG_INFO;  break;
        case GGML_LOG_LEVEL_DEBUG: hl = HELIX_LOG_DEBUG; break;
        default:                   hl = HELIX_LOG_TRACE;  break;
    }
    Logger::instance().log(hl, msg.c_str());
}

Runtime::Runtime(const RuntimeOptions& opts) : opts_(opts) {
    std::lock_guard<std::mutex> init_guard(s_backend_init_mu);
    if (s_runtime_count == 0) {
        Logger::instance().min_level.store(opts.log_level, std::memory_order_relaxed);
        llama_log_set(llama_log_redirect, nullptr);
        llama_backend_init();
    }
    ++s_runtime_count;
    profile_ = probe_hardware();
    build_describe();
    log_info("helix runtime initialised (phase 5, " +
             std::to_string(profile_.accelerators.size()) + " accelerator(s))");
}

Runtime::~Runtime() {
    std::lock_guard<std::mutex> init_guard(s_backend_init_mu);
    --s_runtime_count;
    if (s_runtime_count == 0) {
        llama_backend_free();
    }
}

void Runtime::build_describe() {
    const CpuInfo& cpu = profile_.cpu;
    const RamInfo& ram = profile_.ram;

    // Accelerator array.
    nj accel_arr = nj::array();
    for (const auto& a : profile_.accelerators) {
        nj entry = {
            {"device_index",    a.device_index},
            {"device_name",     a.device_name},
            {"backend",         accel_backend_name(a.backend)},
            {"backend_name",    a.backend_name},
            {"total_vram_bytes", a.total_vram_bytes},
            {"free_vram_bytes",  a.free_vram_bytes},
            {"is_primary_display", a.is_primary_display},
        };
        if (a.compute_capability) entry["compute_capability"] = *a.compute_capability;
        if (a.pcie_link_gen)      entry["pcie_link_gen"]      = *a.pcie_link_gen;
        if (a.pcie_link_width)    entry["pcie_link_width"]    = *a.pcie_link_width;
        accel_arr.push_back(std::move(entry));
    }

    // Build active backend list.
    nj backends_arr = nj::array();
    backends_arr.push_back("cpu");
    for (const auto& a : profile_.accelerators) {
        std::string name = accel_backend_name(a.backend);
        bool already = false;
        for (const auto& b : backends_arr) {
            if (b.get<std::string>() == name) { already = true; break; }
        }
        if (!already) backends_arr.push_back(name);
    }

    nj j = {
        {"phase",             HELIX_PROGRAM_PHASE},
        {"backends",          backends_arr},
        {"device_preference", opts_.device_preference},
        {"cpu", {
            {"vendor",             cpu.vendor.empty() ? "unknown" : cpu.vendor},
            {"model_name",         cpu.model_name.empty() ? "unknown" : cpu.model_name},
            {"logical_cores",      cpu.logical_cores},
            {"physical_cores",     cpu.physical_cores},
            {"performance_cores",  cpu.performance_cores},
            {"efficiency_cores",   cpu.efficiency_cores},
            {"isa_tier",           isa_tier_name(cpu.isa_tier)},
            {"memory_bandwidth",   mem_bw_tier_name(cpu.bw_tier)},
            {"numa",               cpu.numa_available},
        }},
        {"ram", {
            {"total_bytes",     ram.total_bytes},
            {"available_bytes", ram.available_bytes},
        }},
        {"accelerators", accel_arr},
    };

    const uint32_t desc_batch = pick_n_batch(profile_, ModelInfo{});
    j["auto_tune"] = {
        {"n_threads",       pick_n_threads(profile_)},
        {"n_threads_batch", pick_n_threads_batch(profile_)},
        {"n_batch",         desc_batch},
        {"n_ubatch",        pick_n_ubatch(profile_, ModelInfo{}, desc_batch)},
        {"flash_attn",      pick_flash_attn(profile_)},
    };

    describe_json_ = j.dump(2);
}

} // namespace helix
