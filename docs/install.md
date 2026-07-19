# Installation

## Requirements

- A GGUF model file (download from Hugging Face or use `helix-doctor --download-test-model`)
- For GPU backends: appropriate drivers (CUDA ≥ 12.0, ROCm ≥ 6.0, Vulkan ≥ 1.3)

## Package managers

### macOS — Homebrew

```sh
# Helix tap (available immediately at GA)
brew tap anemoi-ai/tap
brew install helix

# Once homebrew/core review completes (a few weeks post-GA):
brew install helix
```

### Windows — Chocolatey

```powershell
choco install helix
```

### Debian/Ubuntu

```sh
# Download from https://github.com/anemoi-ai/helix/releases/latest
sudo dpkg -i libhelix1_1.0.0-1_amd64.deb
sudo dpkg -i libhelix-dev_1.0.0-1_amd64.deb   # headers + cmake + pkg-config
sudo dpkg -i helix-tools_1.0.0-1_amd64.deb    # helix-doctor, helix-bench
```

### Language wrappers (bundled libhelix)

The wrappers bundle `libhelix` — no separate native install needed.

| Language | Command |
|----------|---------|
| Python | `pip install helix-llm` |
| Rust | `cargo add helix` |
| Node | `npm install @helix/node` |
| .NET | `dotnet add package Helix.Net` |
| Go | `go get github.com/anemoi-ai/helix-go` |

## Manual install from tarball

Download from [GitHub Releases](https://github.com/anemoi-ai/helix/releases):

```sh
# Linux x86_64, CPU backend
curl -fsSL -o helix.tar.xz \
  https://github.com/anemoi-ai/helix/releases/download/v1.0.0/libhelix-1.0.0-linux-x86_64-cpu.tar.xz

# Verify checksum
sha256sum -c <(grep linux-x86_64-cpu SHA256SUMS)

tar xJf helix.tar.xz
sudo cp -r dist/include/* /usr/local/include/
sudo cp -r dist/lib/*     /usr/local/lib/
sudo ldconfig
```

## CMake integration

After installing, downstream projects use:

```cmake
find_package(helix 1.0 REQUIRED)
target_link_libraries(my_app PRIVATE helix::helix)
```

Or with pkg-config:

```sh
gcc main.c $(pkg-config --cflags --libs helix) -o my_app
```

## Verify installation

```sh
helix-doctor
# Prints runtime summary and ABI version
# Pass --explain for the full hardware profile and auto-tune plan
```

## Backends

| Backend | Platforms | Flag |
|---------|-----------|------|
| cpu | All | `-DHELIX_BACKEND=cpu` |
| cuda | Linux, Windows | `-DHELIX_BACKEND=cuda` |
| metal | macOS, iOS | `-DHELIX_BACKEND=metal` |
| vulkan | Linux, Windows, Android, macOS | `-DHELIX_BACKEND=vulkan` |
| rocm | Linux | `-DHELIX_BACKEND=rocm` |
| omni | Linux, Windows | `-DHELIX_BACKEND=omni` |

The `omni` backend compiles all GPU backends in and selects the best one at
runtime. It produces a larger binary but eliminates the need to choose at
build time.

For macOS, the prebuilt `metal` artefact includes a CPU fallback — no separate
CPU build is needed.
