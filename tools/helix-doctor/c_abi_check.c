/* Compile-time canary: this translation unit is compiled as C (not C++) so the
 * build breaks if helix.h ever stops being valid C99. helix-doctor's main logic
 * lives in main.cpp (it parses describe() JSON with nlohmann), so without this
 * file nothing would exercise the C path of the public header any more.
 *
 * It references the public types, macros, and a function pointer signature
 * without calling anything. helix_doctor_c_abi_check() is invoked once from
 * main() so the linker keeps it. */
#include "helix.h"
#include <stddef.h> /* NULL */

void helix_doctor_c_abi_check(void);

void helix_doctor_c_abi_check(void) {
    helix_status_t    st = HELIX_OK;
    helix_log_level_t lv = HELIX_LOG_INFO;
    uint32_t          v  = (uint32_t)HELIX_ABI_VERSION;
    helix_stream_cb   cb = NULL; /* C function-pointer typedef must parse */
    (void)st; (void)lv; (void)v; (void)cb;
}
