# AGENTS.md – guide for AI coding agents (and humans)

Three small **llama.cpp `ggml-vulkan` patches** that speed up prompt processing on **AMD GCN APUs** (Vega iGPU,
**gfx90c** – Renoir/Cezanne/Lucienne, Ryzen 4000/5000G; likely also Raven/Picasso) with **Mesa RADV**: +27 % to +62 %
pp512, token generation unchanged. Benchmarks and usage in `README.md`.

Keywords: llama.cpp, ggml, ggml-vulkan, Vulkan, RADV, Mesa, AMD GCN, Vega iGPU, gfx90c, APU, UMA, Ryzen 5600G,
flash attention, matmul tile, prompt processing, llama-bench, test-backend-ops.

## Layout

| Path | What |
|---|---|
| `patches/0001-*` | flash attention prefill: keep 8-row blocks on GCN APUs also for long KV |
| `patches/0002-*` | dequantizing matmul: 64×128 two-wave tile as the large tile (batches > 64 tokens) |
| `patches/0004-*` | the same large tile for f32/f16/bf16 matmuls (depends on 0002) |

All three only take effect for `AMD GCN && UMA && RADV`; other GPUs keep the upstream path.

## Rules

- Apply in order (0004 builds on 0002) on the llama.cpp base named in `README.md`; check with `test-backend-ops`
  (FLASH_ATTN_EXT, MUL_MAT) before benchmarking.
- Benchmark each build from its own build directory – copied binaries keep the RUNPATH of their build folder and load
  the wrong `libggml-vulkan.so`.
- Measured on **one machine** (Ryzen 5 5600G, Vega 7, DDR4-3200, Mesa 26.2.7). Use Mesa ≥ 26.3.1 for Vulkan compute on
  this GPU (an older NIR bug miscompiles a 16-bit integer pattern, see https://github.com/Styx85/gfx90c-rocm).
- **llama.cpp does not accept AI-written pull requests, issues or comments.** These patches were made with an AI coding
  assistant; do not open anything upstream from this repository. A human may submit them in their own words.

## Related

ROCm/HIP on the same GPU (HIP backend patches, rocBLAS, kernel fix): https://github.com/Styx85/gfx90c-rocm

## Contributing

Benchmarks from other GCN APUs are welcome as issues. AI assistance is fine here – please say so.
