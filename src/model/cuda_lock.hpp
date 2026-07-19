#pragma once

// HELIX_CUDA_REQUIRES_GLOBAL_LOCK controls whether sessions on the same CUDA
// model serialise through cuda_mu_. Set to 0 once upstream llama.cpp resolves
// the CUDA backend re-entrancy issue; the lock plumbing stays but becomes a
// no-op so the exit path compiles cleanly.
#if defined(HELIX_HAS_CUDA) || defined(HELIX_HAS_ROCM)
#  define HELIX_CUDA_REQUIRES_GLOBAL_LOCK 1
#else
#  define HELIX_CUDA_REQUIRES_GLOBAL_LOCK 0
#endif

namespace helix {
// Helpers to query whether the current build has a backend that needs the
// per-model lock. Used in unit tests and helix-doctor.
bool cuda_lock_required();
} // namespace helix
