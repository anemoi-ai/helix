# AGENTS.md

> Orientation for AI coding agents (and humans) starting a new session on this
> repository. Read this first.

## What Helix is

Helix is a single C/C++ shared library (`libhelix.so` / `.dylib` / `.dll`) that
turns any GGUF model into an in-process, OpenAI Chat Completions–compatible
endpoint. Tiny C ABI (29 symbols as of the 1.5 bump), JSON wire
protocol byte-identical to OpenAI. No subprocesses, no sockets, no Python.

- **Language/toolchain:** C++17, CMake ≥ 3.22, GCC 11+/Clang 14+/MSVC 2022+.
- **Backend (build-time):** `-DHELIX_BACKEND=cpu|cuda|metal|vulkan|rocm|omni`.
- **Vendored inference:** `third_party/llama.cpp` (pinned; SHA in
  `third_party/llama.cpp.commit`).

## Read these first (in order)

1. **`README.md`** — project overview, quick start, platform-support table.
2. **`CONTRIBUTING.md`** — dev setup, PR checklist, ABI rules.

## Repository layout (the parts that matter most)

```
helix.h                       Public C ABI (26 symbols). Touching it = ABI bump.
helix.ld / helix.exp / helix.def   Linux / macOS / Windows export control.
CMakeLists.txt                Build; backend selection; per-platform export wiring.
tests/CMakeLists.txt          Unit-test source list (mirrors the CPU-probe conditional).
src/hardware/
  cpu.hpp                     CpuInfo / AcceleratorInfo / HardwareProfile structs.
  cpu_linux.cpp               Linux probe + generic #else stub.
  cpu_windows.cpp             Windows probe.
  accel_cuda.cpp              CUDA enrichment — already cross-platform.
  accelerators.cpp            GGML backend registry → AcceleratorInfo.
  isa.cpp  ram.cpp  mem_bw_table.cpp   Already cross-platform; mostly unchanged.
src/abi/helix_abi.cpp         Every extern "C" export + version string.
src/multimodal/image_decode.cpp   file:// URI jail (platform-aware path handling).
.github/workflows/  ci.yml  release.yml  abi-check.yml
```

## How to build / test (Linux, default)

```sh
cmake -B build -DHELIX_BACKEND=cpu -DHELIX_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Run lint/typecheck-equivalent before considering work done: there is no
separate lint target — `cmake --build` with `-Wall -Wextra` (GCC/Clang) or
`/W4` (MSVC) is the gate. The CI gate is `ctest` + `abi-check.yml`.

## Rules that are easy to violate by accident

- **ABI is frozen for the 1.x window.** New symbols = a new `HELIX_1.x` node
  in `helix.ld`, an entry in `helix.exp` (leading `_`) and `helix.def`, a bump
  of `HELIX_ABI_VERSION` in `helix.h`, the version string in
  `helix_abi.cpp`, *and* the expected surface in `.github/workflows/abi-check.yml`.
  Miss any one and CI fails.
- **No comments in code unless asked.** Match surrounding style.
- **`__attribute__` / POSIX-only headers / `ssize_t` are banned** in `src/`
  (MSVC compatibility — the tree currently has zero hits; keep it that way).
  All threading via `std::thread`; all `snprintf` via `<cstdio>`.
- **Don't commit untracked artefacts** (`build*/`, `build-llama-update/`).
  Stage only the files your change actually touches.
