#include "cpu.hpp"

#if defined(__linux__)
#  include <cinttypes>
#  include <cstdio>
#  include <fstream>
#  include <string>
#elif defined(__APPLE__)
#  include <mach/mach.h>
#  include <sys/sysctl.h>
#elif defined(_WIN32)
#  include <windows.h>
#endif

namespace helix {

RamInfo probe_ram() {
    RamInfo info;

#if defined(__linux__)
    std::ifstream f("/proc/meminfo");
    if (!f) return info;
    std::string line;
    while (std::getline(f, line)) {
        uint64_t val = 0;
        char key[64] = {};
        if (std::sscanf(line.c_str(), "%63s %" SCNu64 " kB", key, &val) == 2) {
            if (std::string(key) == "MemTotal:")     info.total_bytes     = val * 1024ULL;
            else if (std::string(key) == "MemAvailable:") info.available_bytes = val * 1024ULL;
        }
    }

#elif defined(__APPLE__)
    uint64_t total = 0;
    size_t len = sizeof(total);
    sysctlbyname("hw.memsize", &total, &len, nullptr, 0);
    info.total_bytes = total;

    vm_statistics64_data_t vmstat = {};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    kern_return_t kr = host_statistics64(mach_host_self(), HOST_VM_INFO64,
                      reinterpret_cast<host_info64_t>(&vmstat), &count);
    uint64_t page = vm_page_size;
    if (kr == KERN_SUCCESS) {
        info.available_bytes = (vmstat.free_count + vmstat.inactive_count) * page;
    }

#elif defined(_WIN32)
    MEMORYSTATUSEX ms = {};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        info.total_bytes     = ms.ullTotalPhys;
        info.available_bytes = ms.ullAvailPhys;
    }
#endif

    return info;
}

} // namespace helix
