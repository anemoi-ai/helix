#include "cpu.hpp"

// Forward declarations from platform-specific and table TUs.
namespace helix {
    CpuInfo probe_cpu();
    RamInfo probe_ram();
    MemBandwidthTier infer_mem_bw_tier(const std::string& model_name);
}

namespace helix {

HardwareProfile probe_hardware() {
    HardwareProfile hw;
    hw.cpu = probe_cpu();
    hw.ram = probe_ram();
    hw.cpu.bw_tier = infer_mem_bw_tier(hw.cpu.model_name);
    // probe_accelerators() uses the GGML backend registry, which is populated
    // by llama_backend_init() before probe_hardware() is called.
    hw.accelerators = probe_accelerators();
    return hw;
}

} // namespace helix
