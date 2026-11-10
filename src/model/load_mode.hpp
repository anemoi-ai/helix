#pragma once
#include "llama.h"

namespace helix {

/* helix's ModelOptions keeps two booleans (use_mmap, use_mlock) — they are
 * public ABI JSON keys. llama.cpp replaced its own pair with one enum
 * (llama_model_params::load_mode, upstream b12033). Map the four
 * combinations exactly, so a load behaves as it did before the bump.
 *
 * LLAMA_LOAD_MODE_AUTO ("detect from device capabilities") is deliberately
 * never produced: adopting it would change load behaviour under an
 * unchanged request, and is its own decision. */
constexpr llama_load_mode load_mode_from(bool use_mmap, bool use_mlock) noexcept {
    if (use_mmap && use_mlock) return LLAMA_LOAD_MODE_MMAP_MLOCK;
    if (use_mmap)              return LLAMA_LOAD_MODE_MMAP;
    if (use_mlock)             return LLAMA_LOAD_MODE_MLOCK;
    return LLAMA_LOAD_MODE_NONE;
}

} // namespace helix
