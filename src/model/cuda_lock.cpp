#include "cuda_lock.hpp"

namespace helix {

bool cuda_lock_required() {
#if HELIX_CUDA_REQUIRES_GLOBAL_LOCK
    return true;
#else
    return false;
#endif
}

} // namespace helix
