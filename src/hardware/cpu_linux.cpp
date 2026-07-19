#include "cpu.hpp"

#ifdef __linux__

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace helix {

static int safe_stoi(const std::string& s, int default_val = 0) {
    try {
        return std::stoi(s);
    } catch (...) {
        return default_val;
    }
}

// Read a sysfs file as a string, return empty on failure.
static std::string read_sysfs(const char* path) {
    std::ifstream f(path);
    if (!f) return {};
    std::string s;
    std::getline(f, s);
    return s;
}

// Parse /proc/cpuinfo into key→value pairs per CPU.
struct CpuInfoEntry {
    int processor_id = -1;
    std::string vendor_id;
    std::string model_name;
    int siblings     = 0;   // logical cores per package
    int cpu_cores    = 0;   // physical cores per package
};

static std::vector<CpuInfoEntry> parse_proc_cpuinfo() {
    std::ifstream f("/proc/cpuinfo");
    std::vector<CpuInfoEntry> entries;
    if (!f) return entries;

    CpuInfoEntry cur;
    auto flush = [&]() {
        if (cur.processor_id >= 0) entries.push_back(cur);
        cur = {};
    };

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) { flush(); continue; }

        auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = line.substr(0, colon);
        std::string val = line.substr(colon + 1);

        // Trim whitespace.
        auto trim = [](std::string& s) {
            auto a = s.find_first_not_of(" \t\r\n");
            auto b = s.find_last_not_of(" \t\r\n");
            s = (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
        };
        trim(key); trim(val);

        if (key == "processor")  cur.processor_id = safe_stoi(val);
        else if (key == "vendor_id")   cur.vendor_id   = val;
        else if (key == "model name")  cur.model_name  = val;
        else if (key == "siblings")    cur.siblings     = safe_stoi(val);
        else if (key == "cpu cores")   cur.cpu_cores   = safe_stoi(val);
    }
    flush();
    return entries;
}

// Detect P/E core split on hybrid CPUs by comparing max CPU frequencies.
// Returns {performance_count, efficiency_count}.
static std::pair<uint32_t,uint32_t> detect_hybrid_cores(uint32_t logical_cores) {
    std::map<uint64_t, uint32_t> freq_count;

    for (uint32_t i = 0; i < logical_cores; ++i) {
        char path[128];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%u/cpufreq/scaling_max_freq", i);
        std::string val = read_sysfs(path);
        if (val.empty()) continue;
        try {
            uint64_t f = std::stoull(val);
            freq_count[f]++;
        } catch (...) {}
    }

    if (freq_count.size() < 2) {
        // Non-hybrid or can't detect.
        return {0, 0};
    }

    // Highest frequency group = P-cores.
    auto it = freq_count.rbegin();
    uint32_t p = it->second;
    ++it;
    uint32_t e = 0;
    while (it != freq_count.rend()) { e += it->second; ++it; }
    return {p, e};
}

static bool detect_numa() {
    std::ifstream f("/sys/devices/system/node/online");
    if (!f) return false;
    std::string s;
    std::getline(f, s);
    // e.g. "0" = no NUMA; "0-1" or "0,1" = NUMA
    return s.find('-') != std::string::npos || s.find(',') != std::string::npos;
}

CpuInfo probe_cpu() {
    CpuInfo info;
    info.isa_tier = detect_isa_tier();

    auto entries = parse_proc_cpuinfo();
    if (entries.empty()) {
        // Fallback
        info.logical_cores     = std::max(1u, static_cast<uint32_t>(std::thread::hardware_concurrency()));
        info.physical_cores    = std::max(1u, info.logical_cores / 2);
        info.performance_cores = info.physical_cores;
        return info;
    }

    info.logical_cores = static_cast<uint32_t>(entries.size());
    if (!entries[0].vendor_id.empty())  info.vendor     = entries[0].vendor_id;
    if (!entries[0].model_name.empty()) info.model_name = entries[0].model_name;

    // Physical core count: each entry reports siblings (logical) and cpu cores (physical)
    // per package. Ratio: cpu_cores / siblings = fraction of physical cores.
    if (entries[0].siblings > 0 && entries[0].cpu_cores > 0) {
        double ratio = static_cast<double>(entries[0].cpu_cores) /
                       static_cast<double>(entries[0].siblings);
        info.physical_cores = std::max(1u,
            static_cast<uint32_t>(info.logical_cores * ratio + 0.5));
    } else {
        info.physical_cores = std::max(1u, info.logical_cores / 2);
    }

    // Hybrid detection.
    auto [p, e] = detect_hybrid_cores(info.logical_cores);
    if (p > 0 && e > 0) {
        info.performance_cores = p;
        info.efficiency_cores  = e;
    } else {
        info.performance_cores = info.physical_cores;
        info.efficiency_cores  = 0;
    }

    info.numa_available = detect_numa();

    return info;
}

} // namespace helix

#else // not Linux — use fallback

#include <thread>

namespace helix {

CpuInfo probe_cpu() {
    CpuInfo info;
    info.isa_tier          = detect_isa_tier();
    info.logical_cores     = std::max(1u, static_cast<uint32_t>(std::thread::hardware_concurrency()));
    info.physical_cores    = std::max(1u, info.logical_cores / 2);
    info.performance_cores = info.physical_cores;
    return info;
}

} // namespace helix

#endif // __linux__
