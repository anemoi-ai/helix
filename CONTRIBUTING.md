# Contributing to Helix

Thank you for your interest in contributing. This document covers the
practicalities of getting a change into the project.

## Before you start

- Check the issue tracker to see if your bug or feature is already tracked.
- For non-trivial changes, open an issue first so we can discuss approach
  before you invest time in an implementation.
- By contributing you agree that your changes will be licensed under
  Apache 2.0 (the same license as the project).

## Development setup

```sh
git clone --recurse-submodules https://github.com/anemoi-ai/helix
cd helix
cmake -B build -DHELIX_BACKEND=cpu -DHELIX_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

All tests must pass on the CPU backend before submitting a PR. The CI
matrix runs additional backends; you don't need GPU hardware locally.

## Pull request checklist

- [ ] All existing tests pass (`ctest --test-dir build`).
- [ ] New behaviour is covered by at least one unit or integration test.
- [ ] `helix.h` is not modified without a corresponding update to all six
      wrappers and a note in the PR description.
- [ ] Any public API change is discussed in the issue tracker before
      implementation — the 1.x ABI is frozen; new symbols require a minor
      version bump and two senior approvals.
- [ ] The `[abi-break]` label is applied and two senior engineers have
      approved if the change would fail `abi-compliance-checker`.
- [ ] Commit messages are imperative, present tense, ≤ 72 characters on
      the subject line.

## ABI policy

The public C ABI (`helix.h`, the 29 exported symbols) is frozen for 18 months
from v1.0.0. PRs that would change function signatures, struct layouts, or
remove enum values will be rejected unless they carry `[abi-break]` with the
required approvals. See `docs/abi-policy.md` for the full policy.

## Testing

```sh
# Unit tests only (fast)
ctest --test-dir build -R unit --output-on-failure

# Integration tests (need a GGUF model)
export HELIX_TEST_MODEL=/path/to/model.gguf
ctest --test-dir build -R integration --output-on-failure

# Wrapper conformance (Python example)
cd wrappers/python && pip install -e ".[dev]" && pytest
```

## Style

- C/C++ code follows the existing style (4-space indent, braces on the same
  line for functions, `snake_case` for identifiers).
- No new comments that describe *what* the code does — only *why* when the
  reason is non-obvious.
- No new dependencies without a discussion in the issue tracker.

## Releasing (maintainers only)

See `.github/workflows/release.yml` and `docs/ci-build-setup.md`. Releases are always
tagged; never pushed manually. The CI workflow handles signing, packaging,
and wrapper publication.
