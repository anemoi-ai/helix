# Helix 1.0.0 — Reference Benchmark Report

**Date:** 2026-05-31  
**Tag:** v1.0.0  
**llama.cpp pin:** see `third_party/llama.cpp.commit`  
**Reproduced with:** `helix-bench --reproduce REPORT-1.0.0`

---

## Methodology

### Reference machines

| ID | System | CPU | GPU | RAM | OS |
|----|--------|-----|-----|-----|----|
| H1 | Apple M2 MacBook Air | Apple M2 (8P+2E) | Unified 8-core GPU | 16 GB | macOS 15.4 |
| H2 | Apple M3 Max MacBook Pro 14" | Apple M3 Max (12P+4E) | Unified 30-core GPU | 36 GB | macOS 15.4 |
| H3 | Custom desktop | Intel i7-13700K | NVIDIA RTX 4070 12 GB | 64 GB DDR5-6000 | Ubuntu 22.04 |
| H4 | Lenovo ThinkPad X1 Carbon (2024) | AMD Ryzen 7840U | AMD Radeon 780M iGPU | 32 GB LPDDR5 | Ubuntu 22.04 |
| H5 | Custom workstation | AMD Ryzen 5950X | NVIDIA RTX 3090 24 GB | 128 GB DDR4-3600 | Ubuntu 22.04 |
| H6 | Custom workstation | AMD Ryzen 7800X3D | NVIDIA RTX 4090 24 GB | 64 GB DDR5-6000 | Ubuntu 22.04 |
| H7 | Dual-socket server | Intel Xeon 8480+ (2×) | 2× NVIDIA A100 80 GB | 1 TB DDR5 | Ubuntu 22.04 |
| H8 | Microsoft Surface Pro 11 | Snapdragon X Elite X1E-80-100 | Adreno X1 | 64 GB LPDDR5x | Windows 11 ARM |

### Reference models

| Model | Size | Quantisation | Source |
|-------|------|--------------|--------|
| Qwen 2.5 0.5B | 0.5B | Q8_0 | Qwen/Qwen2.5-0.5B-Instruct-GGUF |
| Llama 3.2 1B | 1B | Q8_0 | meta-llama/Llama-3.2-1B-Instruct-GGUF |
| Llama 3.2 3B | 3B | Q6_K | meta-llama/Llama-3.2-3B-Instruct-GGUF |
| Llama 3.1 8B | 8B | Q4_K_M | meta-llama/Meta-Llama-3.1-8B-Instruct-GGUF |
| Qwen 2.5 14B | 14B | Q4_K_M | Qwen/Qwen2.5-14B-Instruct-GGUF |

### Metrics

| Metric | Definition |
|--------|-----------|
| **Cold PP** | Prompt-processing throughput (tok/s) on the first request (cold KV cache) |
| **Warm PP** | Prompt-processing throughput (tok/s) with prefix-cache active |
| **TG** | Token-generation throughput (tok/s), sustained over 256 output tokens |
| **TTFT** | Time-to-first-token (ms), warm cache |
| **RSS** | Resident set size (MB) after model load |
| **VRAM** | GPU VRAM used (MB); 0 for CPU/iGPU unified-memory builds |
| **Helix-auto/tuned** | Helix auto-tuned TG ÷ Helix hand-tuned TG (%) |
| **Helix-auto/llama-cli** | Helix auto-tuned TG ÷ llama-cli hand-tuned TG (%) — the G3 gate |

### Procedure

- Each cell = median of 5 runs; outliers >2σ discarded and re-run.
- 30-second thermal cooldown between runs.
- CPU/GPU governor set to `performance` for the duration.
- CUDA driver: 560.35.03. ROCm: 6.2.4. Vulkan SDK: 1.3.290.
- Helix built with `-DCMAKE_BUILD_TYPE=Release` and the appropriate `-DHELIX_BACKEND=`.
- llama-cli comparison: `llama-cli -m <model> -p <prompt> -n 256 -ngl 99 -t <tuned-threads>`.

---

## G3 acceptance gates (RFC §4 and §8.4)

| Machine | Canonical model | Gate | Result |
|---------|-----------------|------|--------|
| H1–H3, H5–H7 | Llama 3.1 8B TG + PP | ≥ 95 % of llama-cli-tuned | **PASS** |
| H4 (Radeon 780M iGPU) | Llama 3.1 8B TG | ≥ 85 % (§8 exception) | **PASS** |
| H8 (Snapdragon Adreno) | Llama 3.1 8B TG | ≥ 80 % (§8 exception) | **PASS** |

All cells pass their respective gates. No release blockers.

---

## Results table — Token generation (TG, tok/s)

_Bold = Helix-auto ≥ 99 % of hand-tuned (effectively indistinguishable)._

| Machine | Backend | Qwen 0.5B | Llama 1B | Llama 3B | **Llama 8B** | Qwen 14B |
|---------|---------|-----------|----------|----------|-------------|---------|
| H1 M2 | Metal | **92** | **78** | **51** | **32** | **18** |
| H2 M3 Max | Metal | **148** | **128** | **88** | **58** | **34** |
| H3 i7 + RTX 4070 | CUDA | **198** | **172** | **118** | **78** | **42** |
| H4 Ryzen 7840U | Vulkan | **42** | **36** | **22** | 12 | 6.4 |
| H5 5950X + RTX 3090 | CUDA | **228** | **196** | **138** | **92** | **52** |
| H6 7800X3D + RTX 4090 | CUDA | **312** | **272** | **188** | **128** | **72** |
| H7 2× A100 | CUDA | **382** | **328** | **228** | **158** | **88** |
| H8 Snapdragon X | Vulkan | **72** | **62** | **38** | 22 | 11 |

## Results table — Prompt processing (Cold PP, tok/s)

| Machine | Backend | Qwen 0.5B | Llama 1B | Llama 3B | **Llama 8B** | Qwen 14B |
|---------|---------|-----------|----------|----------|-------------|---------|
| H1 M2 | Metal | 1820 | 1540 | 820 | 410 | 210 |
| H2 M3 Max | Metal | 3140 | 2680 | 1480 | 760 | 390 |
| H3 i7 + RTX 4070 | CUDA | 4820 | 3960 | 2140 | 1120 | 520 |
| H4 Ryzen 7840U | Vulkan | 520 | 440 | 240 | 120 | 58 |
| H5 5950X + RTX 3090 | CUDA | 5640 | 4820 | 2640 | 1380 | 640 |
| H6 7800X3D + RTX 4090 | CUDA | 7820 | 6640 | 3680 | 1920 | 920 |
| H7 2× A100 | CUDA | 9240 | 7840 | 4320 | 2240 | 1080 |
| H8 Snapdragon X | Vulkan | 1240 | 1040 | 580 | 298 | 140 |

## Results table — Time to first token (TTFT, ms, warm cache)

| Machine | Backend | Qwen 0.5B | Llama 1B | Llama 3B | **Llama 8B** | Qwen 14B |
|---------|---------|-----------|----------|----------|-------------|---------|
| H1 M2 | Metal | 38 | 44 | 72 | 118 | 210 |
| H2 M3 Max | Metal | 22 | 27 | 42 | 68 | 128 |
| H3 i7 + RTX 4070 | CUDA | 14 | 18 | 28 | 48 | 92 |
| H4 Ryzen 7840U | Vulkan | 88 | 104 | 168 | 312 | 620 |
| H5 5950X + RTX 3090 | CUDA | 12 | 15 | 24 | 40 | 78 |
| H6 7800X3D + RTX 4090 | CUDA | 9 | 11 | 18 | 28 | 52 |
| H7 2× A100 | CUDA | 8 | 9 | 14 | 21 | 38 |
| H8 Snapdragon X | Vulkan | 48 | 58 | 94 | 172 | 360 |

---

## Discussion

### Where we are proud: H2 (M3 Max)

The M3 Max achieves 58 tok/s on Llama 3.1 8B — 99.0 % of hand-tuned —
because the Metal backend's unified memory eliminates PCIe transfer overhead
entirely. The auto-tuner correctly identifies the 30-core GPU and sets
`n_gpu_layers=99`, `flash_attn=true`.

### Where we meet expectations: H7 (dual A100)

The two-A100 configuration shows near-linear scaling from single-GPU on
smaller models (0.5B, 1B) and expected diminishing returns on the 14B model
where activation transfer overhead dominates. The omni backend's tensor-parallel
path handles this correctly; Helix-auto hits 96.3 % of `llama-cli` tuned.

### Known gap: H4 (Radeon 780M) and H8 (Adreno)

Integrated GPU Vulkan paths (AMD 780M, Snapdragon Adreno) land in the 82–91 %
range vs hand-tuned. The gap is architectural: the auto-tuner's n_gpu_layers
heuristic conservatively offloads fewer layers than an expert would to avoid
iGPU VRAM pressure on the 780M, and the Adreno Vulkan driver lacks certain
subgroup extensions that GGML uses for optimal performance. Both paths pass
their lower G3 gates (85 % and 80 % respectively). Continued improvement is
planned for 1.1.

---

## Reproduction

```
helix-bench --reproduce REPORT-1.0.0
```

This command reads the report's machine spec, checks the local hardware
matches within acceptable tolerance, and reruns the same workload. The output
is formatted identically to the CSV above for direct comparison.

For a single cell:
```
helix-bench \
  --model /path/to/llama-3.1-8b-q4_k_m.gguf \
  --backend cuda \
  --runs 5 \
  --output-format csv
```

Raw data: [`REPORT-1.0.0.csv`](REPORT-1.0.0.csv)
