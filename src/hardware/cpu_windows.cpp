#include "cpu.hpp"

#ifdef _WIN32

#include <windows.h>
#include <intrin.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace helix {

// Vendor string from CPUID leaf 0 (EBX, EDX, ECX order).
static std::string get_vendor_from_cpuid() {
    int info[4];
    __cpuid(info, 0);
    char vendor[13] = {};
    std::memcpy(vendor + 0, &info[1], 4);   // EBX
    std::memcpy(vendor + 4, &info[3], 4);   // EDX
    std::memcpy(vendor + 8, &info[2], 4);   // ECX
    return {vendor};                         // "GenuineIntel" / "AuthenticAMD"
}

// Model name from the registry (populated by the HAL at boot).
static std::string get_model_name_from_registry() {
    HKEY hKey;
    if (RegOpenKeyExW(HKEY_LOCAL_MACHINE,
            L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
            0, KEY_READ, &hKey) != ERROR_SUCCESS) {
        return {};
    }
    WCHAR buf[256] = {};
    DWORD len = sizeof(buf);
    DWORD type = 0;
    LSTATUS rc = RegQueryValueExW(hKey, L"ProcessorNameString",
                                  nullptr, &type,
                                  reinterpret_cast<LPBYTE>(buf), &len);
    RegCloseKey(hKey);
    if (rc != ERROR_SUCCESS || type != REG_SZ) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, buf, -1, nullptr, 0, nullptr, nullptr);
    std::string result(n > 0 ? n - 1 : 0, '\0');
    if (n > 0)
        WideCharToMultiByte(CP_UTF8, 0, buf, -1, result.data(), n, nullptr, nullptr);
    return result;
}

struct CoreTopology {
    uint32_t logical = 0, physical = 0, perf = 0, efficiency = 0;
    bool numa = false;
};

// Topology via GetLogicalProcessorInformationEx (RelationAll). Counts logical
// CPUs across all processor groups (correct on >64-thread boxes), P/E cores via
// EfficiencyClass, and NUMA nodes.
static CoreTopology probe_topology() {
    CoreTopology topo;
    topo.logical = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);

    DWORD len = 0;
    if (!GetLogicalProcessorInformationEx(RelationAll, nullptr, &len) &&
        GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
        return topo;
    }
    std::vector<BYTE> buf(len);
    if (!GetLogicalProcessorInformationEx(RelationAll,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()),
            &len)) {
        return topo;
    }

    bool hybrid = false;
    uint32_t numa_nodes = 0;
    for (DWORD off = 0; off < len; ) {
        auto* e = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data() + off);
        switch (e->Relationship) {
            case RelationProcessorCore: {
                ++topo.physical;
                // EfficiencyClass: 0 = E-core, >0 = P-core (Win10 1507+).
                if (e->Processor.EfficiencyClass > 0) { ++topo.perf; hybrid = true; }
                else                                  { ++topo.efficiency; }
                break;
            }
            case RelationNumaNode: ++numa_nodes; break;
            default: break;
        }
        off += e->Size;   // records are variable-length
    }
    if (!hybrid) { topo.perf = topo.physical; topo.efficiency = 0; }
    topo.numa = (numa_nodes > 1);
    return topo;
}

CpuInfo probe_cpu() {
    CpuInfo info;
    info.isa_tier   = detect_isa_tier();          // shared isa.cpp (x86 MSVC path)
    info.vendor     = get_vendor_from_cpuid();
    info.model_name = get_model_name_from_registry();

    auto topo = probe_topology();
    info.logical_cores     = std::max(1u, topo.logical);
    info.physical_cores    = std::max(1u, topo.physical);
    info.performance_cores = std::max(1u, topo.perf);
    info.efficiency_cores  = topo.efficiency;
    info.numa_available    = topo.numa;

    // bw_tier is inferred centrally from model_name in hardware.cpp.
    return info;
}

} // namespace helix

#endif // _WIN32
