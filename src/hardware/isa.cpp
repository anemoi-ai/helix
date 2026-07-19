#include "cpu.hpp"

// Platform-specific ISA detection.

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  define HELIX_ARCH_X86 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#  define HELIX_ARCH_ARM64 1
#endif

#ifdef HELIX_ARCH_X86
#  if defined(_MSC_VER)
#    include <intrin.h>
#  else
#    include <cpuid.h>
#  endif
#  include <cstdint>

static void cpuid(uint32_t leaf, uint32_t subleaf,
                  uint32_t& eax, uint32_t& ebx, uint32_t& ecx, uint32_t& edx) {
#if defined(_MSC_VER)
    int info[4];
    __cpuidex(info, static_cast<int>(leaf), static_cast<int>(subleaf));
    eax = info[0]; ebx = info[1]; ecx = info[2]; edx = info[3];
#else
    __cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
#endif
}

static uint64_t xgetbv(uint32_t idx) {
#if defined(_MSC_VER)
    return _xgetbv(idx);
#elif defined(__GNUC__) || defined(__clang__)
    uint32_t lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(idx));
    return (static_cast<uint64_t>(hi) << 32) | lo;
#else
    (void)idx; return 0;
#endif
}
#endif // HELIX_ARCH_X86

#ifdef HELIX_ARCH_ARM64
#  ifdef __linux__
#    include <sys/auxv.h>
#    include <asm/hwcap.h>
#  endif
#endif

namespace helix {

IsaTier detect_isa_tier() {
#ifdef HELIX_ARCH_X86
    uint32_t eax, ebx, ecx, edx;

    // Check max supported leaf.
    cpuid(0, 0, eax, ebx, ecx, edx);
    uint32_t max_leaf = eax;

    if (max_leaf < 1) return IsaTier::Generic;

    // Leaf 1: SSE / AVX baseline.
    cpuid(1, 0, eax, ebx, ecx, edx);
    bool has_avx  = (ecx & (1u << 28)) != 0; // AVX
    bool has_xsave = (ecx & (1u << 27)) != 0; // OSXSAVE

    if (!has_avx || !has_xsave) return IsaTier::Generic;

    // OS must have enabled AVX state (XSAVE).
    uint64_t xcr0 = xgetbv(0);
    bool avx_enabled = (xcr0 & 0x6u) == 0x6u;   // XMM + YMM
    if (!avx_enabled) return IsaTier::Generic;

    if (max_leaf < 7) return IsaTier::Generic;

    // Leaf 7 sub-leaf 0: AVX2, AVX-512, AMX.
    cpuid(7, 0, eax, ebx, ecx, edx);
    bool has_avx2   = (ebx & (1u << 5)) != 0;
    bool has_avx512f = (ebx & (1u << 16)) != 0;
    bool has_avx512bw = (ebx & (1u << 30)) != 0;
    bool has_avx512vl = (ebx & (1u << 31)) != 0;
    bool has_amx_tile = (edx & (1u << 24)) != 0;

    if (!has_avx2) return IsaTier::Generic;

    // AVX-512: need F + BW + VL for useful matmul.
    bool avx512_ok = has_avx512f && has_avx512bw && has_avx512vl;
    if (avx512_ok) {
        // Verify OS enabled AVX-512 state (bits 5+6 in XCR0: opmask + ZMM_hi).
        bool avx512_enabled = (xcr0 & 0xE0u) == 0xE0u;
        if (avx512_enabled) return IsaTier::Avx512;
    }

    // AMX: needs XSAVE state support (bits 17+18 in XCR0).
    bool amx_enabled = (xcr0 & 0x60000u) == 0x60000u;
    if (has_amx_tile && amx_enabled) return IsaTier::Amx;

    return IsaTier::Avx2;

#elif defined(HELIX_ARCH_ARM64)

#  if defined(__APPLE__)
    // Apple: treat all M-series as AppleM1Plus (sysctl check in cpu_macos.cpp).
    return IsaTier::AppleM1Plus;
#  elif defined(__linux__)
    unsigned long hwcap  = getauxval(AT_HWCAP);
    unsigned long hwcap2 = getauxval(AT_HWCAP2);

    // HWCAP2_SVE2 = 1<<1, HWCAP2_I8MM = 1<<8 (kernel 5.x).
    bool has_sve  = (hwcap & HWCAP_SVE)   != 0;
    bool has_sve2 = (hwcap2 & (1UL << 1)) != 0;  // HWCAP2_SVE2
    bool has_i8mm = (hwcap2 & (1UL << 8)) != 0;  // HWCAP2_I8MM

    // HWCAP_FPHP = half-precision FP; HWCAP_ASIMDHP = NEON FP16
    bool has_fp16 = (hwcap & HWCAP_ASIMDHP) != 0;

    // BF16 is HWCAP2_BF16 = 1<<14
    bool has_bf16 = (hwcap2 & (1UL << 14)) != 0;

    if (has_sve2 && has_i8mm) return IsaTier::Sve2I8mm;
    if (has_sve)               return IsaTier::Sve;
    if (has_bf16)              return IsaTier::NeonBf16;
    if (has_fp16)              return IsaTier::NeonFp16;
    return IsaTier::Neon;
#  else
    return IsaTier::Neon;
#  endif

#else
    return IsaTier::Generic;
#endif
}

const char* isa_tier_name(IsaTier t) {
    switch (t) {
        case IsaTier::Generic:    return "generic";
        case IsaTier::Avx2:      return "avx2";
        case IsaTier::Avx512:    return "avx512";
        case IsaTier::Amx:       return "amx";
        case IsaTier::Neon:      return "neon";
        case IsaTier::NeonFp16:  return "neon_fp16";
        case IsaTier::NeonBf16:  return "neon_bf16";
        case IsaTier::Sve:       return "sve";
        case IsaTier::Sve2I8mm:  return "sve2_i8mm";
        case IsaTier::AppleM1Plus: return "apple_m1plus";
    }
    return "unknown";
}

const char* mem_bw_tier_name(MemBandwidthTier t) {
    switch (t) {
        case MemBandwidthTier::Slow:      return "slow";
        case MemBandwidthTier::DDR4:      return "ddr4";
        case MemBandwidthTier::DDR5:      return "ddr5";
        case MemBandwidthTier::LPDDR5x:   return "lpddr5x";
        case MemBandwidthTier::OnPackage: return "on_package";
    }
    return "unknown";
}

} // namespace helix
