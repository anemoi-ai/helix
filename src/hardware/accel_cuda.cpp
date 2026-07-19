#include "cpu.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>

namespace helix {

// ---------------------------------------------------------------------------
// PCIe link speed/width from Linux sysfs.
// The PCI address is formatted as "%04x:%02x:%02x.0" (domain:bus:device.fn).
// ---------------------------------------------------------------------------

static std::optional<int> pcie_link_gen_from_speed(const std::string& speed_str) {
    // speed_str examples: "2.5 GT/s PCIe", "5.0 GT/s PCIe", "8.0 GT/s PCIe",
    //                     "16.0 GT/s PCIe", "32.0 GT/s PCIe", "64.0 GT/s PCIe"
    char* end = nullptr;
    double speed = std::strtod(speed_str.c_str(), &end);
    if (end == speed_str.c_str()) return std::nullopt;
    if (speed >= 63.0) return 6;
    if (speed >= 31.0) return 5;
    if (speed >= 15.0) return 4;
    if (speed >= 7.0)  return 3;
    if (speed >= 4.0)  return 2;
    if (speed >= 2.0)  return 1;
    return std::nullopt;
}

static void read_pcie_info_linux(int pci_domain, int pci_bus, int pci_device,
                                  std::optional<int>& out_gen,
                                  std::optional<int>& out_width) {
    char pci_addr[32];
    std::snprintf(pci_addr, sizeof(pci_addr), "%04x:%02x:%02x.0",
                  pci_domain, pci_bus, pci_device);

    // Link speed → PCIe generation.
    {
        std::string path = std::string("/sys/bus/pci/devices/") + pci_addr
                           + "/current_link_speed";
        std::ifstream f(path);
        if (f) {
            std::string line;
            std::getline(f, line);
            out_gen = pcie_link_gen_from_speed(line);
        }
    }

    // Link width (x1, x4, x8, x16 …).
    {
        std::string path = std::string("/sys/bus/pci/devices/") + pci_addr
                           + "/current_link_width";
        std::ifstream f(path);
        if (f) {
            int w = 0;
            if (f >> w) out_width = w;
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point called from accelerators.cpp for each CUDA device.
// ---------------------------------------------------------------------------

void enrich_cuda_device(AcceleratorInfo& info) {
    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, info.backend_device_index) != cudaSuccess)
        return;

    // Real device name (e.g. "NVIDIA GeForce RTX 5070 Ti").
    if (props.name[0] != '\0')
        info.device_name = props.name;

    // Compute capability as a dotted string ("12.0", "8.9", …).
    char cc[16];
    std::snprintf(cc, sizeof(cc), "%d.%d", props.major, props.minor);
    info.compute_capability = cc;

    // PCIe link info via sysfs (Linux only).
#if defined(__linux__)
    read_pcie_info_linux(props.pciDomainID, props.pciBusID, props.pciDeviceID,
                          info.pcie_link_gen, info.pcie_link_width);
#endif
}

} // namespace helix
