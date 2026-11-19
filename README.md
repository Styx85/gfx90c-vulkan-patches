# llama.cpp Vulkan patches for GCN APUs (gfx90c / Vega iGPU)

Three small patches for `ggml-vulkan` that speed up prompt processing on AMD GCN APUs
(tested: Ryzen 5 5600G, Vega 7 iGPU, gfx90c, Mesa RADV). Token generation is unchanged.

They only take effect on **AMD GCN + UMA (integrated GPU) + RADV**. On any other GPU the code path is unchanged.

> **How this was made:** these patches are the result of an AI-assisted trial-and-error tuning session
> (Claude Code agents measuring on real hardware). Every change was benchmarked and checked with
> `test-backend-ops`, but only on one device. They are shared as-is in case they help others with the same hardware.
> Not submitted upstream.

## Patches

| File | Change |
|---|---|
| `0001` | Flash attention prefill: keep 8-row blocks on GCN APUs also for long KV (upstream switches to 16 rows at n_kv ≥ 1024, which is slower here) |
| `0002` | Dequantizing matmul: 64×128 tile computed by two waves as the *large* tile, used for batches > 64 tokens |
| `0004` | Same large tile for f32/f16/bf16 matmuls (no effect on quantized models) |

A fourth variant (one wave per FA workgroup for head size ≥ 256) was dropped: it made Gemma 4 prefill much slower at long context.

## Results (gfx90c, RADV / Mesa 26.2.7, `llama-bench -ngl 99 -fa 1`, t/s)

**gemma-4-12b-it QAT q4_0**, two interleaved rounds, spread ≈ 1 %:

| Build | pp512 | tg128 | pp512 @ 4096 ctx | tg128 @ 4096 ctx |
|---|---:|---:|---:|---:|
| llama.cpp 41d8297, unpatched | 61.7 | 5.6 | 39.2 | 4.9 |
| **+ 0001, 0002, 0004** | **78.7 (+27 %)** | 5.6 | **63.5 (+62 %)** | 5.1 |

**Small models** (Q4_K_M, unpatched → patched):

| Model, context | pp512 | tg128 |
|---|---|---|
| Qwen3-0.6B, 0 | 1136 → **1343** | 101 → 100 |
| Qwen3-0.6B, 4096 | 313 → **425** | 41 → 41 |
| Qwen3-1.7B, 0 | 441 → **547** | 39 → 38 |
| Qwen3-1.7B, 4096 | 224 → **293** | 25 → 25 |
| gemma-3-1b, 0 | 820 → **1018** | 45 → 45 |
| gemma-3-1b, 4096 | 723 → **885** | 43 → 43 |

Single op (`test-backend-ops perf`, MUL_MAT m=4096 n=512 k=14336, TFLOPS): q4_0 1.43 → 1.93, q4_K 1.39 → 1.88,
q6_K 1.26 → 1.65. Correctness: `test-backend-ops` FLASH_ATTN_EXT 2931/2931, MUL_MAT 2604/2604.

Token generation is memory-bandwidth bound on this APU (~45 GB/s), so the patches don't change it.

## Use

Base: llama.cpp `41d8297` (19 Nov 2026); they also apply to tag `b12542`. Apply one after another (0004 builds on 0002):

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout b12542
for p in /path/to/patches/000{1,2,4}-*.patch; do git apply "$p"; done
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server llama-bench
```

When comparing builds, run each from its own build directory (or set `LD_LIBRARY_PATH`): copied binaries keep an
absolute RUNPATH to the original build folder and silently load the wrong `libggml-vulkan.so`.

Other GCN APUs (Raven/Picasso, Renoir/Cezanne/Lucienne) will probably behave similarly but are untested.
Feedback with numbers from other devices is welcome via issues.

## License

MIT, like llama.cpp.
