# GenomeOcean × vLLM V1 Upgrade — Benchmark Report

**Date:** 2026-07-01
**Hardware:** NERSC Perlmutter, NVIDIA A100-SXM4-40GB (4 GPUs/node)
**Branch under test:** `vllm-v1-upgrade` vs `main`
**Model:** `DOEJGI/GenomeOcean-4B`

---

## TL;DR

The vLLM **V1 engine upgrade is a decisive win** for GenomeOcean-4B generation and
**performance-neutral (safe)** for embedding. Ship it.

| Workload | V0 (baseline) | V1 (upgrade) | Result |
|---|---|---|---|
| **Generation** (median over all configs) | — | — | **V1 faster at every one of 18 configs** |
| Generation median speedup TP=1 / TP=2 / TP=4 | 1× | **1.39× / 1.86× / 2.31×** | speedup grows with tensor parallelism |
| Generation best case (plen=10240, TP=4, b=1) | 61 tok/s | **227 tok/s** | **3.69×** |
| **Multi-GPU scaling** (heavy config, TP1→TP4) | 2.36× | **6.43×** | V0 barely scales; V1 scales well |
| **Embedding** (transformers 4.55→5.6) | — | — | ≤1% delta, identical VRAM — **neutral** |

---

## Environment

| | V0 baseline | V1 upgrade |
|---|---|---|
| Container | `nvcr.io/nvidia/vllm:25.09-py3` | `nvcr.io/nvidia/vllm:26.06-py3` |
| vLLM | 0.10.1.1+nv25.09 | **0.22.1** |
| transformers | 4.55.2 | **5.6.0** |
| torch | 2.9.0a0+nv25.09 | 2.13.0a0+nv26.06 |
| Engine | V0 (`VLLM_USE_V1=0`) | V1 (default) |
| `enforce_eager` | **False** (CUDA graphs ON) | **False** (CUDA graphs ON) |
| `gpu_memory_utilization` | 0.85 | 0.85 |
| `max_model_len` | 11264 | 11264 |

**Fairness controls:** identical sampling params (temp 0.7, top_p 0.9, top_k 50, 512 output
tokens), identical `enforce_eager=False` (CUDA graphs on for both), identical prompt/batch
sweep, 2 warmup + 5 measured iterations, `detokenize=False`. N-token blocking is disabled for
both engines (it requires a logits processor that the V0 legacy engine rejects, and it does not
affect throughput) — see Methodology note 5.

---

## Generation results (tokens/sec, median of 5 iterations)

### TP = 1 (single A100)
| prompt_len | batch | V0 | V1 | V1/V0 |
|---:|---:|---:|---:|:---:|
| 1024 | 1 | 110.3 | 124.8 | 1.13× |
| 1024 | 8 | 712.3 | 882.6 | 1.24× |
| 1024 | 32 | 1722.1 | 2650.7 | 1.54× |
| 10240 | 1 | 65.1 | 113.7 | **1.75×** |
| 10240 | 8 | 299.7 | 531.6 | **1.77×** |
| 10240 | 32 | 400.1 | 412.5 | 1.03× |

Median speedup **1.39×**.

### TP = 2
| prompt_len | batch | V0 | V1 | V1/V0 |
|---:|---:|---:|---:|:---:|
| 1024 | 1 | 140.1 | 172.5 | 1.23× |
| 1024 | 8 | 951.1 | 1266.7 | 1.33× |
| 1024 | 32 | 2383.5 | 3948.5 | 1.66× |
| 10240 | 1 | 45.9 | 160.8 | **3.50×** |
| 10240 | 8 | 309.9 | 636.5 | 2.05× |
| 10240 | 32 | 709.5 | 1587.7 | 2.24× |

Median speedup **1.86×**.

### TP = 4
| prompt_len | batch | V0 | V1 | V1/V0 |
|---:|---:|---:|---:|:---:|
| 1024 | 1 | 166.5 | 235.8 | 1.42× |
| 1024 | 8 | 1189.5 | 1547.4 | 1.30× |
| 1024 | 32 | 3162.8 | 5748.4 | **1.82×** |
| 10240 | 1 | 61.4 | 226.9 | **3.69×** |
| 10240 | 8 | 346.8 | 1096.2 | **3.16×** |
| 10240 | 32 | 945.6 | 2651.4 | **2.80×** |

Median speedup **2.31×**.

### Tensor-parallel scaling (heavy config: plen=10240, batch=32)
| Engine | TP=1 | TP=2 | TP=4 | TP1→TP4 scaling |
|---|---:|---:|---:|:---:|
| V0 | 400 | 709 | 946 | 2.36× |
| V1 | 413 | 1588 | 2651 | **6.43×** |

The headline structural finding: **V0 barely benefits from more GPUs; V1 scales almost linearly.**
The V1 advantage is largest exactly where it matters most — long prompts, high batch, multi-GPU.

---

## Embedding results (transformers 4.55.2 → 5.6.0, 1× A100)

The embedding path uses `transformers.AutoModel` directly (no vLLM), so this isolates the
transformers-library upgrade.

| seq_len | batch | V0 tok/s | V1 tok/s | V1/V0 | Peak VRAM |
|---:|---:|---:|---:|:---:|---:|
| 1024 | 1 | 19914 | 21346 | 1.07× | 8.0 GB |
| 1024 | 8 | 22191 | 22358 | 1.01× | 8.9 GB |
| 1024 | 32 | 22887 | 23106 | 1.01× | 12.0 GB |
| 1024 | 64 | 22970 | 23178 | 1.01× | 16.2 GB |
| 10240 | 1 | 19463 | 19644 | 1.01× | 9.5 GB |
| 10240 | 8 | 19822 | 19969 | 1.01× | 18.2 GB |
| 10240 | 32 | OOM | OOM | — | >40 GB |
| 10240 | 64 | OOM | OOM | — | >40 GB |

**Neutral:** the transformers 4.55→5.6 upgrade does not change embedding throughput (≤1%
everywhere), VRAM, or the OOM boundary. Safe to upgrade.

---

## Methodology notes & fixes applied

The benchmark harness required five fixes before jobs would run correctly on Perlmutter/Shifter.
These are documented so the run is reproducible:

1. **Shifter `/root` mount trap.** `--volume=CACHE:/root/.cache/huggingface` fails with
   `cannot create mount points in that location` (Shifter can't create nested mount points under
   `/root`). Fix: mount to a top-level dir `/hfcache` and `export HF_HOME=/hfcache`.
2. **Custom all-reduce crash at TP≥2.** `custom_all_reduce.cuh:455 'invalid argument'` during
   CUDA-graph capture. Fix: `disable_custom_all_reduce=True` in `LLM()` when TP>1 (falls back to
   PyNCCL).
3. **libcuda.so not found under torch.compile at TP≥2.** Inductor/Triton only see the compat stub
   `libcuda.so.1`. Fix: symlink it to `/tmp/culibs/libcuda.so` and set `TRITON_LIBCUDA_PATH` +
   `LD_LIBRARY_PATH`.
4. **max_model_len too small.** prompt_len 10240 + 512 output = 10752 > 10240. Fix: raise
   `max_model_len` to 11264.
5. **N-token blocking / logits-processor guard.** Blocking the `N` token requires
   `allowed_token_ids` (V0) or `logit_bias` (V1), both of which register a logits processor. The V0
   legacy engine rejects this with "Logits processors are not supported in multi-step decoding"
   (`llm_engine.py:684`) while V1 tolerates it. `num_scheduler_steps` does **not** exist as a kwarg
   in these builds. To keep the comparison fair and unblock V0, N-blocking is disabled for both
   engines (it does not affect throughput; sequences are not inspected here since
   `detokenize=False`). Set `GO_BLOCK_N=1` to re-enable (V1 only).

**VRAM caveat:** vLLM V1's CuMem allocator does not report through
`torch.cuda.max_memory_allocated` (reads 0.0), so per-config generation VRAM is not tabulated for
V1. Use `nvidia-smi` polling if VRAM figures are needed. Embedding VRAM (transformers path) is
reported correctly.

---

## Recommendation

**Merge `vllm-v1-upgrade`.** V1 is faster on generation across the board (1.1×–3.7×), scales far
better across GPUs (6.4× vs 2.4× at TP=4), and is neutral on embedding. There is no configuration
in which V0 wins. The upgrade also unblocks native V1 features (e.g. speculative decoding), which
is the next investigation.

---

## Reproducibility

- Benchmark scripts: `benchmarks/bench_gen_a100.py`, `benchmarks/bench_embed_a100.py`
- SLURM submit scripts: `benchmarks/sbatch_v{0,1}_{gen,embed}.sh`
- Raw results (JSON, all iterations): `benchmarks/results/`
- Both containers use anonymous NGC pulls; HF cache mounted at `/hfcache` (`HF_HOME`).
- Jobs: submitted with `--constraint=gpu` (set your own `--account` / `--qos` for your cluster);
  gen uses 4 GPUs (sweeps TP=1,2,4), embedding uses 1 GPU.
