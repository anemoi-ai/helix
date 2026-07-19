# ABI Policy

**Commitment window:** 2026-05-31 to 2027-11-30 (18 months)

## What is frozen

From the `v1.0.0` tag, the following are stable across all 1.x releases:

1. **The 16 exported C function signatures frozen at v1.0.0** — names,
   parameter types, return types, and calling conventions. None will be
   removed, renamed, or have their parameters reordered. Minor releases may
   add functions under new linker version nodes; v1.1.0 added
   `helix_embeddings` and `helix_model_embedding_dim` under `HELIX_1.1`,
   frozen under the same terms from that tag.

2. **`helix_status_t` error code integer values** — existing codes keep their
   numeric values. New codes may be added with more-negative integers in 1.x
   minor releases. Callers must default-case unknown values.

3. **`helix_log_level_t` integer values** — existing levels are stable. Future
   levels may be appended with larger integers. Log callbacks must
   default-case unknown values.

4. **JSON request field semantics** — a request that worked at 1.0.0 will
   continue to work unchanged in all 1.x versions.

5. **JSON response field names** — fields present in 1.0.0 responses will
   remain present in 1.x responses. New fields may appear; absent fields
   stay absent.

6. **`HELIX_ABI_VERSION` encoding** — `(major << 16) | (minor << 8) | patch`.
   The major component will remain `1` for the duration of the commitment
   window.

## What is NOT promised

- Byte-identical generated text under the same seed across llama.cpp pin
  updates. Model quantisation and sampling behaviour can shift between pins.
- Performance parity across llama.cpp pin updates. Improvements and
  regressions are both possible; regressions > 10 % on the reference matrix
  are treated as release blockers.
- The `helix-shim-server` HTTP surface (a test tool, not an API).
- Any symbol not in the 16-function list. We have none today, but if an
  internal symbol leaked it would not be protected by this policy.
- Swift, Rust, Python, Node, Go, or .NET wrapper API shapes beyond the
  semantics they derive from the C ABI.

## Versioning mechanics

### HELIX_ABI_VERSION

```c
#define HELIX_ABI_VERSION  ((major << 16) | minor)
```

At 1.0.0: `0x00010000`.  
At 1.1.0 (hypothetical): `0x00010001`.  
At 2.0.0 (major bump): `0x00020000`.

Applications should assert at startup:

```c
uint32_t rt = helix_abi_version();
assert((rt >> 16) == (HELIX_ABI_VERSION >> 16));  /* same major */
assert(rt >= HELIX_ABI_VERSION);                   /* >= compiled-against minor */
```

### Linux soname

`libhelix.so.1` — the soname is tied to the major version. All 1.x releases
share this soname; the runtime linker will transparently use any 1.x patch
or minor release in place of 1.0.0.

### macOS install name

`@rpath/libhelix.1.dylib` with compatibility version `1.0.0` and current
version `1.x.y`. Bumping current version is allowed in patch/minor releases;
bumping compatibility version is forbidden during 1.x.

### Linux versioned symbols

All 16 symbols are tagged with the `HELIX_1.0` version node. Future 1.x
additions will appear in `HELIX_1.1`, `HELIX_1.2`, … each inheriting from
`HELIX_1.0`. This lets downstream tools detect the exact feature level
available at link time.

## Adding new symbols (1.x minor releases)

A backward-compatible addition — new function, new status code, new JSON
field — increments the minor version and gets its own `HELIX_1.N` version
node. The process:

1. Open an RFC-style issue describing the new symbol's purpose and signature.
2. Two senior engineers approve the design.
3. PR adds the symbol to `helix.h`, `helix.ld`, `helix.exp`, `helix.def`,
   all six wrappers, and the documentation.
4. `abi-compliance-checker` confirms the change is additive (no CI failure).
5. Merge; tag `v1.N.0`; update `HELIX_ABI_VERSION_MINOR`.

## The major-version escape (breaking changes)

A 2.0 is permitted only if:
- A security vulnerability requires a fundamentally different function
  signature, or
- The upstream ecosystem (OpenAI wire format, llama.cpp foundations) makes
  1.x untenable to maintain.

Procedure:
1. Apply `[abi-break]` label to the PR. Two senior engineers approve.
2. A dedicated notification workflow fires, alerting the maintainer list.
3. Release `v2.0.0`. Continue `v1.x` security patches for at least 6 months.
4. Document migration in `docs/migration/1.x-to-2.0.md`.

This escape has never been used. The goal is to never use it in the 18-month
window.

## Enforcement

`ci/abi-check.yml` runs on every pull request that touches `helix.h`, source
files, or `CMakeLists.txt`. It downloads the `v1.0.0` baseline, builds the
PR, and runs `abi-compliance-checker --strict`. A failing check blocks merge.

The only way to merge a breaking change is the `[abi-break]` label + two
senior approvals. The CI job is aware of this label and bypasses the strict
check when it is present, while simultaneously requiring the two approvals
via branch protection rules.
