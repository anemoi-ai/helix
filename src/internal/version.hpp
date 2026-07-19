#pragma once
#include "helix.h"

/* Human-readable "major.minor.patch" derived from the ABI macros in helix.h,
 * so helix_version_string() and the chat-response system_fingerprint can
 * never drift from HELIX_ABI_VERSION. */
#define HELIX_VERSION_STR_(x) #x
#define HELIX_VERSION_STR(x) HELIX_VERSION_STR_(x)
#define HELIX_VERSION_STRING                       \
    HELIX_VERSION_STR(HELIX_ABI_VERSION_MAJOR) "." \
    HELIX_VERSION_STR(HELIX_ABI_VERSION_MINOR) "." \
    HELIX_VERSION_STR(HELIX_ABI_VERSION_PATCH)

/* Informational program-phase marker reported by the describe() JSON of
 * Runtime and Model. Single definition so the two can never disagree; keep
 * in sync with the phase documented in README.md when it advances. */
#define HELIX_PROGRAM_PHASE 7
