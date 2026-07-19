#include <gtest/gtest.h>
#include "model/cuda_lock.hpp"

// cuda_lock_required() reflects compile-time backend flags.
TEST(CudaLock, RequiredReflectsBuildFlags) {
#if HELIX_CUDA_REQUIRES_GLOBAL_LOCK
    EXPECT_TRUE(helix::cuda_lock_required());
#else
    EXPECT_FALSE(helix::cuda_lock_required());
#endif
}

// On a CPU build the lock is not required.
TEST(CudaLock, CpuBuildNoLockRequired) {
#if !defined(HELIX_HAS_CUDA) && !defined(HELIX_HAS_ROCM)
    EXPECT_FALSE(helix::cuda_lock_required());
#endif
}
