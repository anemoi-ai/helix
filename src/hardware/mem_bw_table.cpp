#include "cpu.hpp"
#include <cstring>
#include <string>

// Maps CPU model_name substrings to memory bandwidth tier.
// First match wins; order matters (more specific patterns first).

namespace helix {

struct MemBwEntry {
    const char* pattern;       // substring of model_name (case-sensitive)
    MemBandwidthTier tier;
};

static const MemBwEntry kTable[] = {
    // Apple Silicon — unified memory, on-package bandwidth
    { "Apple M",                       MemBandwidthTier::OnPackage },

    // Qualcomm Snapdragon X Elite / Plus — LPDDR5X
    { "Snapdragon X Elite",            MemBandwidthTier::LPDDR5x },
    { "Snapdragon X Plus",             MemBandwidthTier::LPDDR5x },

    // Intel 12th/13th/14th/15th gen — mostly DDR5 platforms
    { "12th Gen Intel",                MemBandwidthTier::DDR5 },
    { "13th Gen Intel",                MemBandwidthTier::DDR5 },
    { "14th Gen Intel",                MemBandwidthTier::DDR5 },
    { "15th Gen Intel",                MemBandwidthTier::DDR5 },
    // Arrow Lake and Lunar Lake
    { "Core Ultra",                    MemBandwidthTier::DDR5 },

    // AMD Ryzen 7000/8000/9000 series — DDR5
    { "Ryzen 9 7",                     MemBandwidthTier::DDR5 },
    { "Ryzen 9 8",                     MemBandwidthTier::DDR5 },
    { "Ryzen 9 9",                     MemBandwidthTier::DDR5 },
    { "Ryzen 7 7",                     MemBandwidthTier::DDR5 },
    { "Ryzen 7 8",                     MemBandwidthTier::DDR5 },
    { "Ryzen 7 9",                     MemBandwidthTier::DDR5 },
    { "Ryzen 5 7",                     MemBandwidthTier::DDR5 },
    { "Ryzen 5 8",                     MemBandwidthTier::DDR5 },
    { "Ryzen 5 9",                     MemBandwidthTier::DDR5 },
    // Threadripper PRO 7000 series
    { "Threadripper PRO 7",            MemBandwidthTier::DDR5 },
    // EPYC Genoa/Bergamo
    { "EPYC 9",                        MemBandwidthTier::DDR5 },

    // AMD Ryzen 5000 series — DDR4
    { "Ryzen 9 5",                     MemBandwidthTier::DDR4 },
    { "Ryzen 7 5",                     MemBandwidthTier::DDR4 },
    { "Ryzen 5 5",                     MemBandwidthTier::DDR4 },
    // AMD Ryzen 6000 series (Rembrandt, Zen 3+) — LPDDR5/DDR5
    { "Ryzen 9 6",                     MemBandwidthTier::DDR5 },
    { "Ryzen 7 6",                     MemBandwidthTier::DDR5 },
    { "Ryzen 5 6",                     MemBandwidthTier::DDR5 },

    // Intel 10th/11th gen — DDR4
    { "10th Gen Intel",                MemBandwidthTier::DDR4 },
    { "11th Gen Intel",                MemBandwidthTier::DDR4 },

    // Intel Xeon Scalable 3rd gen+ (Ice Lake/Sapphire Rapids) — DDR4/DDR5
    { "Xeon Platinum 8",               MemBandwidthTier::DDR5 },
    { "Xeon Gold 6",                   MemBandwidthTier::DDR4 },
    { "Xeon Silver",                   MemBandwidthTier::DDR4 },

    // Generic Xeon / Opteron fallback
    { "Xeon",                          MemBandwidthTier::DDR4 },
    { "EPYC",                          MemBandwidthTier::DDR4 },
};

MemBandwidthTier infer_mem_bw_tier(const std::string& model_name) {
    for (const auto& entry : kTable) {
        if (model_name.find(entry.pattern) != std::string::npos) {
            return entry.tier;
        }
    }
    return MemBandwidthTier::DDR4; // conservative default
}

} // namespace helix
