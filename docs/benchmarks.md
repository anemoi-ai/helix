# Benchmarks

Reference performance numbers for Helix 1.0.0.

Full report: [`benchmarks/REPORT-1.0.0.md`](../benchmarks/REPORT-1.0.0.md)  
Raw data CSV: [`benchmarks/REPORT-1.0.0.csv`](../benchmarks/REPORT-1.0.0.csv)

## G3 acceptance gate

The G3 gate (from RFC §4) requires Helix auto-tuned throughput ≥ 95 % of
`llama-cli` hand-tuned on all reference machines. Lower thresholds apply to
integrated-GPU paths (85 % for AMD 780M, 80 % for Snapdragon Adreno).

**All cells pass at 1.0.0.**

## Token generation — Llama 3.1 8B Q4_K_M

| Machine | Backend | tok/s | vs llama-cli |
|---------|---------|-------|-------------|
| Apple M3 Max | Metal | 58 | 98.5 % |
| RTX 4090 | CUDA | 128 | 97.9 % |
| RTX 3090 | CUDA | 92 | 97.3 % |
| RTX 4070 | CUDA | 78 | 97.1 % |
| 2× A100 80 GB | CUDA | 158 | 96.5 % |
| Apple M2 | Metal | 32 | 97.3 % |
| Snapdragon X Elite | Vulkan | 22 | 82.8 % ✓ |
| Ryzen 7840U (iGPU) | Vulkan | 12 | 87.9 % ✓ |

## Reproduce

```sh
helix-bench --reproduce REPORT-1.0.0
```

Reads the report's machine spec, verifies the local hardware matches, and
reruns the workload. Output is formatted identically to the published CSV.
