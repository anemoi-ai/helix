# CI and Build Infrastructure Guide

This document explains how to build every Helix artefact variant, what
runner or machine each build requires, and how to provision the infrastructure
for a full release.

---

## The full artefact matrix

| # | Platform | Arch | Backend | Runner strategy |
|---|----------|------|---------|----------------|
| 1 | Linux | x86_64 | cpu | GitHub-hosted `ubuntu-22.04` |
| 2 | Linux | x86_64 | cuda | GitHub-hosted `ubuntu-22.04` + CUDA toolkit (no GPU needed to *compile*) |
| 3 | Linux | x86_64 | vulkan | GitHub-hosted `ubuntu-22.04` + Vulkan SDK |
| 4 | Linux | x86_64 | rocm | Self-hosted AMD GPU **or** GitHub-hosted with ROCm SDK (compile-only) |
| 5 | Linux | x86_64 | omni | GitHub-hosted `ubuntu-22.04` + CUDA toolkit + Vulkan SDK |
| 6 | Linux | aarch64 | cpu | GitHub-hosted `ubuntu-22.04-arm` (Linux ARM runner) |
| 7 | Linux | aarch64 | vulkan | GitHub-hosted `ubuntu-22.04-arm` + Vulkan SDK |
| 8 | macOS | arm64 | metal | GitHub-hosted `macos-15` (M-series, Metal SDK included) |
| 9 | macOS | x86_64 | cpu | GitHub-hosted `macos-13` (last Intel macOS runner) |
| 10 | macOS | x86_64 | vulkan | GitHub-hosted `macos-13` + MoltenVK |
| 11 | Windows | x86_64 | cpu | GitHub-hosted `windows-2022` |
| 12 | Windows | x86_64 | cuda | GitHub-hosted `windows-2022` + CUDA toolkit |
| 13 | Windows | x86_64 | vulkan | GitHub-hosted `windows-2022` + Vulkan SDK |
| 14 | Windows | x86_64 | omni | GitHub-hosted `windows-2022` + CUDA toolkit + Vulkan SDK |
| 15 | Windows | arm64 | cpu | Self-hosted Windows ARM **or** cross-compile from x86_64 |
| 16 | Windows | arm64 | vulkan | Self-hosted Windows ARM + Vulkan SDK |
| 17 | iOS | arm64 | metal | GitHub-hosted `macos-15` + Xcode xcframework build |
| 18 | Android | arm64-v8a | cpu + vulkan | GitHub-hosted `ubuntu-22.04` + Android NDK r26+ |
| 19 | Android | x86_64 | cpu | GitHub-hosted `ubuntu-22.04` + Android NDK r26+ |

**Key insight:** most GPU backends only need the SDK installed to *compile* —
they do not require actual GPU hardware in the build runner. GPU hardware is
only needed for *runtime testing*. The release matrix builds and ships
artefacts; integration testing on real hardware is a separate concern covered
in §6.

---

## 1. GitHub-hosted runners

### What is available today (2026)

| Runner label | OS | Arch | vCPU | RAM | Notes |
|---|---|---|---|---|---|
| `ubuntu-22.04` | Ubuntu 22.04 | x86_64 | 2 | 7 GB | Free tier (2000 min/month) |
| `ubuntu-24.04` | Ubuntu 24.04 | x86_64 | 4 | 16 GB | Free tier |
| `ubuntu-22.04-arm` | Ubuntu 22.04 | arm64 | 2 | 7 GB | Requires Team/Enterprise plan |
| `macos-13` | macOS 13 Ventura | x86_64 | 3 | 14 GB | Intel, 10× minute multiplier |
| `macos-14` | macOS 14 Sonoma | arm64 | 3 | 7 GB | M1, 10× minute multiplier |
| `macos-15` | macOS 15 Sequoia | arm64 | 3 | 7 GB | M-series, 10× minute multiplier |
| `windows-2022` | Windows Server 2022 | x86_64 | 2 | 7 GB | 2× minute multiplier |
| `windows-2025` | Windows Server 2025 | x86_64 | 4 | 16 GB | 2× minute multiplier |

> **Windows ARM64:** GitHub does not yet offer a hosted Windows ARM runner as
> of mid-2026. See §4 for self-hosted and cross-compile options.

> **GPU runners:** GitHub offers hosted GPU runners (NVIDIA T4) in limited
> beta for Enterprise accounts. See §5. For most teams, GPU runtime testing
> runs on self-hosted runners.

### Cost at scale (GitHub Actions minutes)

A full release build of all 19 artefacts takes approximately:

| Runner type | Jobs | Minutes/job | Subtotal | Multiplier | Billed minutes |
|---|---|---|---|---|---|
| Linux x86_64 | 7 | ~12 | 84 | 1× | 84 |
| Linux arm64 | 2 | ~15 | 30 | 1× | 30 |
| macOS arm64 | 2 | ~20 | 40 | 10× | 400 |
| macOS x86_64 | 2 | ~18 | 36 | 10× | 360 |
| Windows x86_64 | 4 | ~18 | 72 | 2× | 144 |
| **Total** | | | | | **~1,018** |

At the Pro plan (3,000 min/month free), a single release run consumes ~34 %
of the monthly free allocation. For frequent releases, purchase additional
minutes or use self-hosted runners for the expensive macOS jobs.

---

## 2. Linux builds (x86_64)

### CPU — trivial

```yaml
- name: Build CPU
  runs-on: ubuntu-22.04
  steps:
    - uses: actions/checkout@v4
      with: { submodules: recursive }
    - run: |
        cmake -B build -DHELIX_BACKEND=cpu -DCMAKE_BUILD_TYPE=Release
        cmake --build build --parallel $(nproc)
```

### CUDA — compile without GPU

CUDA compilation requires the toolkit but not a GPU. Install via the official
NVIDIA apt repository. The resulting binary links `libcudart` dynamically (or
statically — see §2.1).

```yaml
- name: Build CUDA
  runs-on: ubuntu-22.04
  steps:
    - uses: actions/checkout@v4
      with: { submodules: recursive }

    - name: Install CUDA Toolkit
      run: |
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        sudo apt-get update -q
        sudo apt-get install -y cuda-toolkit-12-6
        echo "/usr/local/cuda/bin" >> $GITHUB_PATH
        echo "LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> $GITHUB_ENV

    - name: Build
      run: |
        cmake -B build \
          -DHELIX_BACKEND=cuda \
          -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
          -DCMAKE_BUILD_TYPE=Release
        cmake --build build --parallel $(nproc)
```

> **Tip:** the CUDA toolkit download is ~2 GB. Cache it with
> `actions/cache` keyed on the toolkit version to save ~4 min per job.

#### 2.1 Static vs dynamic libcudart

Dynamic linking (`-lcudart`) requires the runtime to be installed on the
user's machine. Static linking (`-lcudart_static`) produces a self-contained
binary at the cost of ~25 MB extra size. For redistributable artefacts,
static link:

```cmake
# In CMakeLists.txt for release builds:
if(HELIX_BACKEND STREQUAL "cuda" OR HELIX_BACKEND STREQUAL "omni")
    target_link_libraries(helix PRIVATE CUDA::cudart_static)
endif()
```

### Vulkan

The Vulkan SDK provides headers and the loader — no GPU needed to compile.

```yaml
- name: Install Vulkan SDK
  run: |
    wget -q -O - https://packages.lunarg.com/lunarg-signing-key-pub.asc \
      | sudo apt-key add -
    sudo wget -q -O /etc/apt/sources.list.d/lunarg-vulkan-jammy.list \
      https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list
    sudo apt-get update -q
    sudo apt-get install -y vulkan-sdk

- name: Build Vulkan
  run: |
    cmake -B build -DHELIX_BACKEND=vulkan -DCMAKE_BUILD_TYPE=Release
    cmake --build build --parallel $(nproc)
```

### ROCm

ROCm compilation on GitHub-hosted runners is possible for build-only artefacts
(the HIP SDK is ~4 GB). For runtime testing you need self-hosted AMD hardware.

```yaml
- name: Install ROCm
  run: |
    sudo apt-get install -y wget gnupg
    wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key \
      | sudo apt-key add -
    echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.2 jammy main" \
      | sudo tee /etc/apt/sources.list.d/rocm.list
    sudo apt-get update -q
    sudo apt-get install -y rocm-dev hip-dev

- name: Build ROCm
  env:
    CMAKE_PREFIX_PATH: /opt/rocm
  run: |
    cmake -B build \
      -DHELIX_BACKEND=rocm \
      -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
      -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
      -DCMAKE_BUILD_TYPE=Release
    cmake --build build --parallel $(nproc)
```

> **Cache note:** ROCm downloads are large. Cache `/opt/rocm` across runs
> or use a pre-baked custom runner image (see §3.3).

---

## 3. Linux aarch64

GitHub's `ubuntu-22.04-arm` runner handles native aarch64 builds with no
cross-compilation needed. This runner requires a **Team or Enterprise** GitHub
plan.

```yaml
runs-on: ubuntu-22.04-arm
steps:
  - uses: actions/checkout@v4
    with: { submodules: recursive }
  - run: |
      cmake -B build -DHELIX_BACKEND=cpu -DCMAKE_BUILD_TYPE=Release
      cmake --build build --parallel $(nproc)
```

### If you don't have a Team plan — cross-compile from x86_64

Use QEMU + a cross-compiler or Docker:

```yaml
- name: Set up QEMU
  uses: docker/setup-qemu-action@v3
  with: { platforms: arm64 }

- name: Build in ARM64 container
  uses: docker/build-push-action@v5
  with:
    platforms: linux/arm64
    context: .
    file: ci/Dockerfile.build
    push: false
    outputs: type=local,dest=dist/
```

`ci/Dockerfile.build`:
```dockerfile
FROM --platform=linux/arm64 ubuntu:22.04
RUN apt-get update && apt-get install -y cmake ninja-build g++ git
WORKDIR /src
COPY . .
RUN cmake -B build -DHELIX_BACKEND=cpu -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --parallel && \
    cmake --install build --prefix /dist
```

QEMU-based cross-builds are ~4–8× slower than native. For aarch64-cpu this
adds about 25 minutes per build.

---

## 4. Windows ARM64

GitHub has no hosted Windows ARM runner. Three options:

### Option A — Self-hosted Windows ARM machine

Register a Snapdragon X Elite or other ARM64 Windows machine as a
self-hosted runner (see §5.2). This is the recommended path for production
releases.

### Option B — Cross-compile from Windows x86_64

MSVC and Clang-CL support cross-compilation to ARM64 without ARM hardware.
Install the ARM64 build tools in the VS 2022 workload:

```yaml
- name: Build Windows ARM64 (cross from x86_64)
  runs-on: windows-2022
  steps:
    - uses: actions/checkout@v4
      with: { submodules: recursive }

    - name: Setup MSVC ARM64 cross toolchain
      uses: ilammy/msvc-dev-cmd@v1
      with:
        arch: amd64_arm64   # host=x86_64, target=arm64

    - name: Configure (ARM64 target)
      run: |
        cmake -B build `
          -G "Ninja" `
          -DCMAKE_SYSTEM_NAME=Windows `
          -DCMAKE_SYSTEM_PROCESSOR=ARM64 `
          -DCMAKE_C_COMPILER=cl `
          -DCMAKE_CXX_COMPILER=cl `
          -DHELIX_BACKEND=cpu `
          -DHELIX_BUILD_TESTS=OFF `
          -DCMAKE_BUILD_TYPE=Release
      shell: pwsh

    - name: Build
      run: cmake --build build --parallel
      shell: pwsh
```

> **Limitation:** cross-compiled binaries cannot be run on the x86_64 build
> machine, so you cannot run tests. The artefact is built and packaged but
> test coverage is provided by the self-hosted ARM runner.

### Option C — Azure ARM64 VMs (cloud)

Azure offers `Standard_D4ps_v5` (Ampere ARM64) VMs that run Windows. Register
one as a self-hosted runner. See §6.3.

---

## 5. macOS (Metal, notarisation)

### macos-15 for arm64/Metal

```yaml
- name: Build macOS Metal
  runs-on: macos-15
  steps:
    - uses: actions/checkout@v4
      with: { submodules: recursive }
    - run: |
        cmake -B build \
          -DHELIX_BACKEND=metal \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_OSX_ARCHITECTURES="arm64"
        cmake --build build --parallel $(sysctl -n hw.logicalcpu)
```

For a universal binary (arm64 + x86_64 in one dylib):

```yaml
    - run: |
        cmake -B build \
          -DHELIX_BACKEND=metal \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64"
        cmake --build build --parallel $(sysctl -n hw.logicalcpu)
```

> Note: a universal build on an M-series runner still cross-compiles the
> x86_64 slice via Rosetta toolchain — it works but adds ~40 % build time.

### Code signing and notarisation

Notarisation is mandatory for macOS 10.15+ Gatekeeper. The workflow must:

1. Import the Developer ID certificate into a temporary keychain.
2. Sign all `.dylib` and tool binaries.
3. Submit to Apple's notary service and wait for the staple.

#### Required GitHub secrets

| Secret | Content |
|--------|---------|
| `APPLE_DEVELOPER_ID_CERT_P12` | Base64-encoded Developer ID Application cert + private key (export from Keychain Access) |
| `APPLE_DEVELOPER_ID_CERT_PASS` | Password for the .p12 file |
| `APPLE_TEAM_ID` | 10-character Apple Team ID (e.g. `ABCD123456`) |
| `APPLE_NOTARY_USER` | Apple ID email used for notarisation |
| `APPLE_NOTARY_PASS` | App-specific password from appleid.apple.com |

#### Signing steps in CI

```yaml
- name: Import certificate
  env:
    CERT_P12: ${{ secrets.APPLE_DEVELOPER_ID_CERT_P12 }}
    CERT_PASS: ${{ secrets.APPLE_DEVELOPER_ID_CERT_PASS }}
  run: |
    echo "$CERT_P12" | base64 --decode > /tmp/cert.p12
    security create-keychain -p "ci" ci.keychain
    security default-keychain -s ci.keychain
    security unlock-keychain -p "ci" ci.keychain
    security import /tmp/cert.p12 -k ci.keychain \
      -P "$CERT_PASS" -T /usr/bin/codesign -T /usr/bin/security
    security set-key-partition-list -S apple-tool:,apple: -s -k "ci" ci.keychain

- name: Sign dylib and tools
  env:
    TEAM_ID: ${{ secrets.APPLE_TEAM_ID }}
  run: |
    find dist -type f \( -name "*.dylib" -o -perm +111 \) | while read f; do
      codesign --force --timestamp --options runtime \
        --sign "Developer ID Application: Anemoi AI ($TEAM_ID)" \
        "$f"
    done

- name: Notarise
  env:
    NOTARY_USER: ${{ secrets.APPLE_NOTARY_USER }}
    NOTARY_PASS: ${{ secrets.APPLE_NOTARY_PASS }}
    TEAM_ID: ${{ secrets.APPLE_TEAM_ID }}
  run: |
    zip -r submit.zip dist/
    xcrun notarytool submit submit.zip \
      --apple-id "$NOTARY_USER" \
      --password "$NOTARY_PASS" \
      --team-id "$TEAM_ID" \
      --wait --timeout 30m
```

> **Notarisation dry-run:** add a job to `ci/abi-check.yml` that signs and
> submits to the notary service on every PR touching the macOS build. This
> catches Apple rejections before release day. Use
> `xcrun notarytool submit ... --wait` — if it fails the PR is blocked.

---

## 6. Android (NDK cross-compile)

Android builds run on any Linux x86_64 runner using the Android NDK.
No Android device is needed to produce the artefact.

```yaml
- name: Build Android arm64-v8a
  runs-on: ubuntu-22.04
  env:
    NDK_VERSION: r26d
    ANDROID_API: 26
  steps:
    - uses: actions/checkout@v4
      with: { submodules: recursive }

    - name: Install NDK
      run: |
        wget -q "https://dl.google.com/android/repository/android-ndk-${NDK_VERSION}-linux.zip"
        unzip -q "android-ndk-${NDK_VERSION}-linux.zip"
        echo "NDK_HOME=$(pwd)/android-ndk-${NDK_VERSION}" >> $GITHUB_ENV

    - name: Build arm64-v8a CPU
      run: |
        cmake -B build-arm64 \
          -DCMAKE_TOOLCHAIN_FILE="$NDK_HOME/build/cmake/android.toolchain.cmake" \
          -DANDROID_ABI=arm64-v8a \
          -DANDROID_PLATFORM=android-$ANDROID_API \
          -DHELIX_BACKEND=cpu \
          -DHELIX_BUILD_TESTS=OFF \
          -DHELIX_BUILD_TOOLS=OFF \
          -DCMAKE_BUILD_TYPE=Release
        cmake --build build-arm64 --parallel $(nproc)

    - name: Build arm64-v8a Vulkan
      run: |
        cmake -B build-arm64-vulkan \
          -DCMAKE_TOOLCHAIN_FILE="$NDK_HOME/build/cmake/android.toolchain.cmake" \
          -DANDROID_ABI=arm64-v8a \
          -DANDROID_PLATFORM=android-$ANDROID_API \
          -DHELIX_BACKEND=vulkan \
          -DHELIX_BUILD_TESTS=OFF \
          -DHELIX_BUILD_TOOLS=OFF \
          -DCMAKE_BUILD_TYPE=Release
        cmake --build build-arm64-vulkan --parallel $(nproc)

    - name: Build x86_64 (emulator)
      run: |
        cmake -B build-x86 \
          -DCMAKE_TOOLCHAIN_FILE="$NDK_HOME/build/cmake/android.toolchain.cmake" \
          -DANDROID_ABI=x86_64 \
          -DANDROID_PLATFORM=android-$ANDROID_API \
          -DHELIX_BACKEND=cpu \
          -DHELIX_BUILD_TESTS=OFF \
          -DHELIX_BUILD_TOOLS=OFF \
          -DCMAKE_BUILD_TYPE=Release
        cmake --build build-x86 --parallel $(nproc)
```

### AAR packaging

Wrap the `.so` files into an Android Archive for easy Gradle consumption:

```yaml
    - name: Package AAR
      run: |
        mkdir -p aar/jni/arm64-v8a aar/jni/x86_64
        cp build-arm64/libhelix.so aar/jni/arm64-v8a/
        cp build-x86/libhelix.so  aar/jni/x86_64/
        cp helix.h aar/
        cat > aar/AndroidManifest.xml <<'XML'
        <manifest xmlns:android="http://schemas.android.com/apk/res/android"
            package="dev.helixllm.helix" android:versionCode="1"
            android:versionName="1.0.0">
        </manifest>
        XML
        cd aar && zip -r ../helix-android-1.0.0.aar .
```

---

## 7. iOS (xcframework)

```yaml
- name: Build iOS xcframework
  runs-on: macos-15
  steps:
    - uses: actions/checkout@v4
      with: { submodules: recursive }

    - name: Build device slice (arm64)
      run: |
        cmake -B build-ios \
          -DCMAKE_SYSTEM_NAME=iOS \
          -DCMAKE_OSX_ARCHITECTURES=arm64 \
          -DHELIX_BACKEND=metal \
          -DHELIX_BUILD_TESTS=OFF \
          -DHELIX_BUILD_TOOLS=OFF \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_OSX_DEPLOYMENT_TARGET=16.0
        cmake --build build-ios --parallel $(sysctl -n hw.logicalcpu)

    - name: Build simulator slice (arm64 + x86_64)
      run: |
        cmake -B build-sim \
          -DCMAKE_SYSTEM_NAME=iOS \
          -DCMAKE_OSX_SYSROOT=iphonesimulator \
          -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
          -DHELIX_BACKEND=metal \
          -DHELIX_BUILD_TESTS=OFF \
          -DHELIX_BUILD_TOOLS=OFF \
          -DCMAKE_BUILD_TYPE=Release
        cmake --build build-sim --parallel $(sysctl -n hw.logicalcpu)

    - name: Create xcframework
      run: |
        xcodebuild -create-xcframework \
          -library build-ios/libhelix.a \
          -headers helix.h \
          -library build-sim/libhelix.a \
          -headers helix.h \
          -output helix.xcframework
        zip -r helix-ios-1.0.0.xcframework.zip helix.xcframework/
```

---

## 8. Self-hosted runners

Self-hosted runners are needed for:
- Windows ARM64 (no hosted runner available)
- GPU *testing* (CUDA, ROCm, Metal, Vulkan) on real hardware
- ROCm builds (the SDK is large; pre-baked images are faster)

### 8.1 Register a self-hosted runner

On the GitHub repository: **Settings → Actions → Runners → New self-hosted runner**

Follow the generated install script. For a permanent service:

```sh
# Linux (as a systemd service)
sudo ./svc.sh install
sudo ./svc.sh start

# Windows (as a Windows service)
.\svc.ps1 install
.\svc.ps1 start
```

### 8.2 Label scheme

Give runners labels that match the `runs-on` values in the workflow:

| Runner label | Hardware |
|---|---|
| `self-hosted-linux-cuda` | Linux x86_64 with NVIDIA GPU |
| `self-hosted-linux-rocm` | Linux x86_64 with AMD GPU |
| `self-hosted-windows-arm64` | Windows ARM64 (Snapdragon X / Ampere) |
| `self-hosted-macos-metal` | macOS Apple Silicon (if you want faster builds than hosted) |

Update `ci/release.yml` runner fields to use these labels:

```yaml
- platform: linux  arch: x86_64  backend: rocm
  runner: self-hosted-linux-rocm
- platform: windows arch: arm64  backend: cpu
  runner: self-hosted-windows-arm64
```

### 8.3 Pre-baked runner images

For large SDKs (ROCm, CUDA), build a Docker image or VM snapshot with the
SDK pre-installed. This cuts job startup from ~8 minutes (SDK install) to
~30 seconds.

**Example: CUDA runner image (GitHub Container Registry)**

```dockerfile
# ci/runners/Dockerfile.cuda
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y wget gnupg cmake ninja-build g++ git
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y cuda-toolkit-12-6
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
```

Use this image in the workflow:

```yaml
jobs:
  build-cuda:
    runs-on: self-hosted-linux-cuda
    container:
      image: ghcr.io/anemoi-ai/helix-builder-cuda:latest
      options: --gpus all   # only when running tests
```

---

## 9. Cloud GPU providers for self-hosted runners

When you need GPU testing infrastructure without owning hardware.

### 9.1 AWS EC2

| Instance | GPU | Use |
|---|---|---|
| `g4dn.xlarge` | NVIDIA T4 16 GB | CUDA integration tests, ~$0.53/hr on-demand |
| `g5.xlarge` | NVIDIA A10G 24 GB | Larger models, ~$1.01/hr |
| `p3.2xlarge` | NVIDIA V100 16 GB | Older; available on spot for ~$0.30/hr |
| `p4d.24xlarge` | 8× A100 40 GB | Multi-GPU (H7 equivalent), ~$32/hr |
| `c7g.xlarge` | ARM64 (Graviton 3) | aarch64 CPU builds, ~$0.14/hr |

**Recommended setup:** Use EC2 Auto Scaling to start a GPU runner only when a
release tag is pushed, then terminate it after the job completes. GitHub
Actions supports ephemeral self-hosted runners — the runner unregisters itself
after one job.

```sh
# Launch an ephemeral runner on EC2 (run this from a lightweight launch job)
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \   # pre-baked CUDA runner AMI
  --instance-type g4dn.xlarge \
  --user-data "$(cat ci/runners/ec2-userdata-cuda.sh)" \
  --instance-market-options '{"MarketType":"spot"}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Purpose,Value=helix-ci}]'
```

### 9.2 Google Cloud (GCP)

| Machine | GPU | Use |
|---|---|---|
| `n1-standard-4` + `nvidia-tesla-t4` | T4 | CUDA tests, ~$0.45/hr |
| `a2-highgpu-1g` | A100 40 GB | Full benchmark runs, ~$3.67/hr |
| `t2a-standard-4` | — (ARM64 Ampere) | aarch64 builds, ~$0.14/hr |

Use **Cloud Build** as an alternative to self-hosted runners for one-off
GPU builds:

```yaml
# cloudbuild.yaml (triggered from GitHub Actions via gcloud CLI)
steps:
  - name: gcr.io/cloud-builders/cmake
    args: ['-B', 'build', '-DHELIX_BACKEND=cuda', '-DCMAKE_BUILD_TYPE=Release']
  - name: gcr.io/cloud-builders/cmake
    args: ['--build', 'build', '--parallel', '8']
options:
  machineType: N1_HIGHCPU_8
  # GPU not available in Cloud Build standard; use a GCE VM instead
```

### 9.3 Azure

| VM | GPU | Use |
|---|---|---|
| `Standard_NC4as_T4_v3` | T4 | CUDA tests, ~$0.53/hr |
| `Standard_ND96asr_v4` | 8× A100 | Multi-GPU, ~$27/hr |
| `Standard_D4ps_v5` | — (ARM64 Ampere) | Windows/Linux ARM64, ~$0.17/hr |

Azure DevOps pipelines can host GPU agents natively:

```yaml
# azure-pipelines-gpu.yml
pool:
  name: GPU-Agents
  demands:
    - Agent.OS -equals Linux
    - gpu -equals nvidia

steps:
  - script: |
      cmake -B build -DHELIX_BACKEND=cuda -DCMAKE_BUILD_TYPE=Release
      cmake --build build --parallel 8
    displayName: Build CUDA
```

### 9.4 Lambda Labs (cost-effective GPU)

Lambda Labs offers bare-metal GPU instances on a per-hour basis with no
long-term commitment, and provides a GitHub Actions integration:

```yaml
- uses: run-on-arch-action/setup@v1   # or direct Lambda API call
```

| Instance | GPU | Price |
|---|---|---|
| `gpu_1x_a10` | A10 24 GB | ~$0.60/hr |
| `gpu_1x_a100` | A100 80 GB | ~$1.99/hr |
| `gpu_8x_a100` | 8× A100 | ~$14.32/hr |

### 9.5 RunPod

RunPod offers spot GPU pods starting at ~$0.20/hr for an RTX 3090.
Use the RunPod API to spin up ephemeral pods for CI:

```python
# ci/runners/runpod_launch.py
import runpod
pod = runpod.create_pod(
    name="helix-ci",
    image_name="ghcr.io/anemoi-ai/helix-builder-cuda:latest",
    gpu_type_id="NVIDIA GeForce RTX 3090",
    cloud_type="SECURE",
    docker_args="--entrypoint /ci/runner.sh",
    env={"GITHUB_TOKEN": os.environ["GITHUB_TOKEN"],
         "RUNNER_URL": os.environ["REPO_URL"]}
)
```

---

## 10. Required GitHub secrets

Set all secrets under **Settings → Secrets and variables → Actions**.

### Signing

| Secret | Purpose |
|--------|---------|
| `APPLE_DEVELOPER_ID_CERT_P12` | Base64 .p12 with Developer ID cert + key |
| `APPLE_DEVELOPER_ID_CERT_PASS` | .p12 password |
| `APPLE_TEAM_ID` | Apple Team ID (10 chars) |
| `APPLE_NOTARY_USER` | Apple ID for notarytool |
| `APPLE_NOTARY_PASS` | App-specific password for notarytool |
| `AUTHENTICODE_CERT_PFX` | Base64 .pfx with Authenticode cert + key |
| `AUTHENTICODE_CERT_PASS` | .pfx password |

### Package registries

| Secret | Purpose |
|--------|---------|
| `PYPI_TOKEN` | PyPI API token (`pypi-…`) |
| `CRATES_IO_TOKEN` | crates.io API token |
| `NPM_TOKEN` | npm access token |
| `NUGET_API_KEY` | NuGet.org API key |

### Infrastructure (if using ephemeral cloud runners)

| Secret | Purpose |
|--------|---------|
| `AWS_ACCESS_KEY_ID` | AWS IAM key for EC2 runner launch |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM secret |
| `GCP_SERVICE_ACCOUNT_KEY` | GCP service account JSON (base64) |
| `AZURE_CREDENTIALS` | Azure service principal JSON |
| `RUNPOD_API_KEY` | RunPod API key |

### Self-hosted runner registration

| Secret | Purpose |
|--------|---------|
| `RUNNER_REGISTRATION_TOKEN` | GitHub runner registration token (rotates; use GitHub API to generate fresh tokens) |

---

## 11. Caching strategy

Good caching cuts a full release build from ~90 minutes to ~30 minutes.

```yaml
# Cache CMake dependencies (llama.cpp build artifacts)
- uses: actions/cache@v4
  with:
    path: |
      build/_deps
      ~/.ccache
    key: ${{ runner.os }}-${{ matrix.arch }}-${{ matrix.backend }}-${{ hashFiles('third_party/llama.cpp.commit') }}
    restore-keys: |
      ${{ runner.os }}-${{ matrix.arch }}-${{ matrix.backend }}-

# Cache CUDA toolkit (saves ~4 min per job)
- uses: actions/cache@v4
  with:
    path: /usr/local/cuda
    key: cuda-12.6-ubuntu22
```

Use `ccache` on Linux/macOS for incremental C++ compilation:

```yaml
- name: Setup ccache
  run: |
    sudo apt-get install -y ccache
    echo "CMAKE_CXX_COMPILER_LAUNCHER=ccache" >> $GITHUB_ENV
    echo "CMAKE_C_COMPILER_LAUNCHER=ccache" >> $GITHUB_ENV
    ccache --set-config=max_size=500M
```

---

## 12. Updated release.yml runner assignments

The following corrects the runner labels in `ci/release.yml` to reflect
runner availability as of mid-2026:

```yaml
# Replace the matrix include section with:
matrix:
  include:
    # ── Linux x86_64 ─────────────────────────────────────────────
    - { platform: linux,   arch: x86_64,   backend: cpu,    runner: ubuntu-22.04 }
    - { platform: linux,   arch: x86_64,   backend: cuda,   runner: ubuntu-22.04 }   # toolkit-only compile
    - { platform: linux,   arch: x86_64,   backend: vulkan, runner: ubuntu-22.04 }
    - { platform: linux,   arch: x86_64,   backend: rocm,   runner: ubuntu-22.04 }   # SDK compile; test on self-hosted
    - { platform: linux,   arch: x86_64,   backend: omni,   runner: ubuntu-22.04 }
    # ── Linux aarch64 ────────────────────────────────────────────
    - { platform: linux,   arch: aarch64,  backend: cpu,    runner: ubuntu-22.04-arm }
    - { platform: linux,   arch: aarch64,  backend: vulkan, runner: ubuntu-22.04-arm }
    # ── macOS ────────────────────────────────────────────────────
    - { platform: macos,   arch: arm64,    backend: metal,  runner: macos-15 }
    - { platform: macos,   arch: x86_64,   backend: cpu,    runner: macos-13 }
    - { platform: macos,   arch: x86_64,   backend: vulkan, runner: macos-13 }
    # ── Windows x86_64 ───────────────────────────────────────────
    - { platform: windows, arch: x86_64,   backend: cpu,    runner: windows-2022 }
    - { platform: windows, arch: x86_64,   backend: cuda,   runner: windows-2022 }
    - { platform: windows, arch: x86_64,   backend: vulkan, runner: windows-2022 }
    - { platform: windows, arch: x86_64,   backend: omni,   runner: windows-2022 }
    # ── Windows arm64 (cross-compiled from x86_64) ───────────────
    - { platform: windows, arch: arm64,    backend: cpu,    runner: windows-2022,   cross: true }
    - { platform: windows, arch: arm64,    backend: vulkan, runner: windows-2022,   cross: true }
    # ── iOS ──────────────────────────────────────────────────────
    - { platform: ios,     arch: arm64,    backend: metal,  runner: macos-15 }
    # ── Android ──────────────────────────────────────────────────
    - { platform: android, arch: arm64-v8a, backend: cpu,   runner: ubuntu-22.04 }
    - { platform: android, arch: arm64-v8a, backend: vulkan,runner: ubuntu-22.04 }
    - { platform: android, arch: x86_64,   backend: cpu,    runner: ubuntu-22.04 }
```

---

## 13. Testing strategy vs build strategy

| Stage | Where it runs | When |
|---|---|---|
| Unit tests | GitHub-hosted (any arch, CPU only) | Every PR |
| Integration tests (CPU) | GitHub-hosted `ubuntu-22.04` | Every PR |
| Integration tests (CUDA) | Self-hosted CUDA runner | Release tag + nightly |
| Integration tests (Metal) | Self-hosted macOS or hosted `macos-15` | Release tag |
| Integration tests (ROCm) | Self-hosted AMD GPU | Release tag |
| Integration tests (Vulkan) | Self-hosted GPU or lavapipe (software) | Every PR |
| Benchmark run (full matrix) | Dedicated bare-metal machines (H1–H8) | Release tag only |
| ABI compliance check | GitHub-hosted `ubuntu-22.04` | Every PR |
| Distribution sanity | GitHub-hosted (multi-platform) | Post-release |

Vulkan integration tests can run on any machine using the `lavapipe` software
renderer (no GPU needed):

```yaml
- name: Install lavapipe (software Vulkan)
  run: sudo apt-get install -y mesa-vulkan-drivers
```

This gives a Vulkan conformance signal on every PR without GPU hardware.

---

## 14. Quick-start checklist

For a team starting from scratch:

- [ ] Enable GitHub Actions on the repository.
- [ ] Add all secrets from §10 (signing + registry tokens first; cloud runner secrets later).
- [ ] Test a CPU-only build: push a tag `v0.0.1-test` and verify `ci/release.yml` runs.
- [ ] Register one self-hosted Linux runner with an NVIDIA GPU and label it `self-hosted-linux-cuda`.
- [ ] Update `ci/release.yml` CUDA test jobs to use `self-hosted-linux-cuda`.
- [ ] Set up a macOS notarisation dry-run in `ci/abi-check.yml` (§5).
- [ ] Decide on the Windows ARM64 strategy (cross-compile or cloud VM) and configure it.
- [ ] Cache CUDA toolkit and llama.cpp build artefacts (§11) to keep build times under 30 min.
- [ ] Set up an ephemeral GPU runner for the post-release benchmark run (§9).
