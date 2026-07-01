#!/usr/bin/env python3
"""
GenomeOcean V0/V1 generation benchmark — Perlmutter A100 edition.

Designed to run inside a shifter container (vllm 0.x for V0, vllm 0.24+ for V1).
Reads the genomeocean package from a mounted repo, runs prompt_len x batch sweep,
writes JSON results.

Env vars:
  GO_ENGINE       "v0" or "v1" (label only)
  GO_MODEL        HF model id (default: pGenomeOcean/GenomeOcean-4B)
  GO_TP           tensor_parallel_size (default 1)
  GO_GPU_MEM      gpu_memory_utilization (default 0.85)
  GO_MAX_LEN      max_model_len (default 10240)
  GO_OUT_TOK      output tokens per request (default 512)
  GO_WARMUP       warmup iters (default 2)
  GO_MEASURE      measure iters (default 5)
  GO_OUT          output JSON path
  GO_PROMPTS      comma list of prompt lengths (default "1024,10240")
  GO_BATCHES      comma list of batch sizes (default "1,8,32")
  GO_REPO         path to genomeocean repo (default /repo)
  GO_ENFORCE_EAGER  "1" or "0" (default "0" on A100 — enable CUDA graphs)
"""
import gc, json, os, statistics, sys, time, random, traceback


def main():
    import torch

    REPO = os.environ.get("GO_REPO", "/repo")
    sys.path.insert(0, REPO)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    import vllm, transformers
    from vllm import LLM, SamplingParams
    from transformers import PreTrainedTokenizerFast

    ENGINE      = os.environ.get("GO_ENGINE", "v?")
    MODEL       = os.environ.get("GO_MODEL", "DOEJGI/GenomeOcean-4B")
    TP          = int(os.environ.get("GO_TP", "1"))
    GPU_MEM     = float(os.environ.get("GO_GPU_MEM", "0.85"))
    MAX_LEN     = int(os.environ.get("GO_MAX_LEN", "10240"))
    OUT_TOK     = int(os.environ.get("GO_OUT_TOK", "512"))
    WARMUP      = int(os.environ.get("GO_WARMUP", "2"))
    MEASURE     = int(os.environ.get("GO_MEASURE", "5"))
    PROMPT_LENS = [int(x) for x in os.environ.get("GO_PROMPTS", "1024,10240").split(",")]
    BATCHES     = [int(x) for x in os.environ.get("GO_BATCHES", "1,8,32").split(",")]
    ENFORCE_EAGER = os.environ.get("GO_ENFORCE_EAGER", "0") == "1"
    slug        = MODEL.split("/")[-1].lower()
    OUT         = os.environ.get("GO_OUT", f"/bench/results_{ENGINE}_{slug}_tp{TP}.json")

    LOG = f"[bench {ENGINE} {slug} tp{TP}]"
    print(f"{LOG} START at {time.strftime('%F %T')}", flush=True)
    print(f"{LOG} python={sys.version.split()[0]} torch={torch.__version__} vllm={vllm.__version__} transformers={transformers.__version__}", flush=True)
    print(f"{LOG} model={MODEL} tp={TP} gpu_mem={GPU_MEM} max_len={MAX_LEN} out_tok={OUT_TOK} enforce_eager={ENFORCE_EAGER}", flush=True)
    print(f"{LOG} PROMPTS={PROMPT_LENS} BATCHES={BATCHES} WARMUP={WARMUP} MEASURE={MEASURE}", flush=True)
    print(f"{LOG} writing to {OUT}", flush=True)
    print(f"{LOG} GPU count: {torch.cuda.device_count()}", flush=True)
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"{LOG}   GPU {i}: {p.name}  {p.total_memory/1024**3:.1f}GB", flush=True)

    torch.cuda.reset_peak_memory_stats()
    def vram_gb():
        # Aggregate across all visible GPUs
        return sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 1024**3

    print(f"{LOG} building LLM...", flush=True)
    t0 = time.time()

    llm_kwargs = dict(
        model=MODEL,
        trust_remote_code=False,
        seed=0,
        dtype=torch.bfloat16,
        max_model_len=MAX_LEN,
        gpu_memory_utilization=GPU_MEM,
        enforce_eager=ENFORCE_EAGER,
        tensor_parallel_size=TP,
    )
    # vLLM V1 (>=0.20) needs skip_tokenizer_init for GenomeOcean's mistral configs
    # under transformers >= 5; safe to add for v1 always.
    needs_skip_tok = ENGINE == "v1" or vllm.__version__.startswith(("0.2", "0.3"))
    if needs_skip_tok:
        llm_kwargs["skip_tokenizer_init"] = True
    # BUG FIX 1: vLLM's custom all-reduce kernel throws
    # 'custom_all_reduce.cuh:455 invalid argument' during CUDA-graph capture on this
    # A100/NCCL topology at TP>=2. Disable it (falls back to PyNCCL) for multi-GPU.
    if TP > 1:
        llm_kwargs["disable_custom_all_reduce"] = True

    try:
        llm = LLM(**llm_kwargs)
    except Exception as e:
        print(f"{LOG} LLM init FAILED: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

    if needs_skip_tok:
        tok = PreTrainedTokenizerFast.from_pretrained(MODEL)
        vocab_size = tok.vocab_size
    else:
        tok = llm.get_tokenizer()
        vocab_size = tok.vocab_size if hasattr(tok, 'vocab_size') else 4096

    print(f"{LOG} LLM built in {time.time()-t0:.1f}s  vocab={vocab_size}", flush=True)

    random.seed(42)
    SAFE_IDS = [i for i in range(9, min(vocab_size - 3, 4093)) if i != 8]

    def make_prompt_ids(length):
        return [1] + random.choices(SAFE_IDS, k=length - 1)

    # N-token blocking (id 8). This requires a logits processor, which the V0 legacy
    # engine (llm_engine.py:684) rejects ("Logits processors are not supported in
    # multi-step decoding"), while V1 tolerates it. To keep the V0-vs-V1 comparison
    # FAIR, N-blocking is applied IDENTICALLY to both engines and defaults OFF:
    #   - It does not affect throughput (tokens/sec) at all.
    #   - detokenize=False means we never inspect the generated sequences here.
    # Set GO_BLOCK_N=1 to force it on (V1 only — V0 will error, by design).
    BLOCK_N = os.environ.get("GO_BLOCK_N", "0") == "1"

    def make_sp():
        base = dict(
            temperature=0.7, top_p=0.9, top_k=50,
            max_tokens=OUT_TOK, min_tokens=min(OUT_TOK, 8),
            stop_token_ids=[2], detokenize=False,
        )
        if not BLOCK_N:
            return SamplingParams(**base)
        if needs_skip_tok:
            return SamplingParams(**base, logit_bias={8: float('-inf')})
        else:
            allowed = [i for i in range(vocab_size) if i != 8]
            return SamplingParams(**base, allowed_token_ids=allowed)

    results = {
        "engine": ENGINE,
        "model": MODEL,
        "tensor_parallel_size": TP,
        "vllm_version": vllm.__version__,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "gpu_memory_utilization": GPU_MEM,
        "max_model_len": MAX_LEN,
        "out_tokens": OUT_TOK,
        "warmup": WARMUP,
        "measure": MEASURE,
        "enforce_eager": ENFORCE_EAGER,
        "gpu_count": torch.cuda.device_count(),
        "configs": []
    }
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    for plen in PROMPT_LENS:
        for batch in BATCHES:
            print(f"{LOG} === config plen={plen} batch={batch} out_tok={OUT_TOK} === ({time.strftime('%T')})", flush=True)
            prompt_inputs = [{"prompt_token_ids": make_prompt_ids(plen)} for _ in range(batch)]
            sp = make_sp()
            cfg = {"prompt_len": plen, "batch": batch, "out_tokens": OUT_TOK}

            try:
                print(f"{LOG}   warmup ({WARMUP} iters)...", flush=True)
                for _ in range(WARMUP):
                    llm.generate(prompt_inputs, sp)

                print(f"{LOG}   measure ({MEASURE} iters)...", flush=True)
                raw = []
                torch.cuda.reset_peak_memory_stats()
                for i in range(MEASURE):
                    t_start = time.perf_counter()
                    outs = llm.generate(prompt_inputs, sp)
                    wall = time.perf_counter() - t_start
                    gen_toks = sum(len(o.token_ids) for out in outs for o in out.outputs)
                    raw.append({
                        "iter": i, "wall_s": wall, "gen_tokens": gen_toks,
                        "tok_per_s": gen_toks / wall, "peak_vram_gb": vram_gb()
                    })

                walls = [r["wall_s"] for r in raw]
                tps   = [r["tok_per_s"] for r in raw]
                vrams = [r["peak_vram_gb"] for r in raw]
                summ = {
                    "n_iters": len(raw),
                    "wall_s_median": statistics.median(walls),
                    "wall_s_mean": statistics.mean(walls),
                    "wall_s_stdev": statistics.stdev(walls) if len(walls) > 1 else 0.0,
                    "wall_s_min": min(walls), "wall_s_max": max(walls),
                    "tok_per_s_median": statistics.median(tps),
                    "tok_per_s_mean": statistics.mean(tps),
                    "gen_tokens_median": statistics.median([r["gen_tokens"] for r in raw]),
                    "peak_vram_gb_max": max(vrams),
                }
                cfg.update({"status": "ok", "raw": raw, "summary": summ})
                print(f"{LOG}   -> median wall={summ['wall_s_median']:.3f}s tok/s={summ['tok_per_s_median']:.1f} vram={summ['peak_vram_gb_max']:.2f}GB", flush=True)

            except torch.cuda.OutOfMemoryError as e:
                print(f"{LOG}   OOM at plen={plen} batch={batch}: {e}", flush=True)
                cfg["status"] = "oom"
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"{LOG}   ERROR at plen={plen} batch={batch}: {e}", flush=True)
                traceback.print_exc()
                cfg["status"] = f"error: {type(e).__name__}: {e}"

            results["configs"].append(cfg)
            with open(OUT, "w") as f:
                json.dump(results, f, indent=2)

    print(f"{LOG} DONE at {time.strftime('%F %T')} — wrote {len(results['configs'])} configs to {OUT}", flush=True)
    print(f"\n{LOG} === SUMMARY ===", flush=True)
    print(f"{'plen':>6} {'batch':>6} {'wall(s)':>10} {'tok/s':>10} {'VRAM(GB)':>10} status", flush=True)
    for c in results["configs"]:
        s = c.get("summary")
        if s:
            print(f"{c['prompt_len']:>6} {c['batch']:>6} {s['wall_s_median']:>10.3f} {s['tok_per_s_median']:>10.1f} {s['peak_vram_gb_max']:>10.2f} ok", flush=True)
        else:
            print(f"{c['prompt_len']:>6} {c['batch']:>6} {'':>10} {'':>10} {'':>10} {c.get('status','?')}", flush=True)


if __name__ == "__main__":
    main()
