#!/usr/bin/env python3
"""
Speculative decoding benchmark — Experiment A: draft-model spec decode.
GenomeOcean-100M drafts for GenomeOcean-4B on vLLM V1. temp>=0.7 (DNA diversity).

Compares baseline (no spec) vs draft_model spec at several K (num_speculative_tokens).
Measures: end-to-end wallclock, output tok/s, and (when available) mean acceptance length.

Env vars:
  GO_TARGET     target model (default DOEJGI/GenomeOcean-4B)
  GO_DRAFT      draft model  (default DOEJGI/GenomeOcean-100M)
  GO_KS         comma list of num_speculative_tokens to test (default "0,2,3,4"; 0=baseline no-spec)
  GO_TEMP       sampling temperature (default 0.7)
  GO_GPU_MEM    gpu_memory_utilization (default 0.85)
  GO_MAX_LEN    max_model_len (default 4608)
  GO_OUT_TOK    output tokens per request (default 512)
  GO_WARMUP     warmup iters (default 1)
  GO_MEASURE    measure iters (default 3)
  GO_PROMPTS    comma prompt lens (default "1024")
  GO_BATCHES    comma batch sizes (default "1,8")
  GO_OUT        output JSON path
"""
import gc, json, os, statistics, sys, time, random, traceback


def build_llm(target, draft, k, gpu_mem, max_len):
    """Build an LLM. k=0 => no spec decode (baseline). k>=1 => draft_model spec."""
    import torch
    from vllm import LLM
    kwargs = dict(
        model=target, trust_remote_code=False, seed=0, dtype=torch.bfloat16,
        max_model_len=max_len, gpu_memory_utilization=gpu_mem,
        enforce_eager=False, tensor_parallel_size=1, skip_tokenizer_init=True,
    )
    if k >= 1:
        kwargs["speculative_config"] = {
            "method": "draft_model",
            "model": draft,
            "num_speculative_tokens": k,
        }
    return LLM(**kwargs)


def run_config(llm, sp_factory, plen, batch, out_tok, warmup, measure, make_prompt):
    import torch
    prompts = [{"prompt_token_ids": make_prompt(plen)} for _ in range(batch)]
    for _ in range(warmup):
        llm.generate(prompts, sp_factory())
    raw = []
    for i in range(measure):
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp_factory())
        wall = time.perf_counter() - t0
        gen = sum(len(o.token_ids) for out in outs for o in out.outputs)
        raw.append({"iter": i, "wall_s": wall, "gen_tokens": gen, "tok_per_s": gen / wall})
    # try to pull acceptance metrics from vLLM engine
    accept = None
    try:
        m = llm.llm_engine.get_metrics()  # V1 metrics API
        for metric in m:
            name = getattr(metric, "name", "")
            if "acceptance" in name.lower() or "accepted" in name.lower():
                accept = {name: getattr(metric, "value", None)}
    except Exception:
        pass
    walls = [r["wall_s"] for r in raw]
    tps = [r["tok_per_s"] for r in raw]
    return {
        "n_iters": len(raw), "raw": raw,
        "wall_s_median": statistics.median(walls),
        "wall_s_mean": statistics.mean(walls),
        "wall_s_stdev": statistics.stdev(walls) if len(walls) > 1 else 0.0,
        "tok_per_s_median": statistics.median(tps),
        "tok_per_s_mean": statistics.mean(tps),
        "gen_tokens_median": statistics.median([r["gen_tokens"] for r in raw]),
        "acceptance": accept,
    }


def main():
    import torch
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import vllm, transformers
    from vllm import SamplingParams
    from transformers import PreTrainedTokenizerFast

    TARGET  = os.environ.get("GO_TARGET", "DOEJGI/GenomeOcean-4B")
    DRAFT   = os.environ.get("GO_DRAFT", "DOEJGI/GenomeOcean-100M")
    KS      = [int(x) for x in os.environ.get("GO_KS", "0,2,3,4").split(",")]
    TEMP    = float(os.environ.get("GO_TEMP", "0.7"))
    GPU_MEM = float(os.environ.get("GO_GPU_MEM", "0.85"))
    MAX_LEN = int(os.environ.get("GO_MAX_LEN", "4608"))
    OUT_TOK = int(os.environ.get("GO_OUT_TOK", "512"))
    WARMUP  = int(os.environ.get("GO_WARMUP", "1"))
    MEASURE = int(os.environ.get("GO_MEASURE", "3"))
    PLENS   = [int(x) for x in os.environ.get("GO_PROMPTS", "1024").split(",")]
    BATCHES = [int(x) for x in os.environ.get("GO_BATCHES", "1,8").split(",")]
    OUT     = os.environ.get("GO_OUT", "/bench/specdec_results.json")

    LOG = "[specdec]"
    print(f"{LOG} START {time.strftime('%F %T')} vllm={vllm.__version__} tf={transformers.__version__}", flush=True)
    print(f"{LOG} target={TARGET} draft={DRAFT} KS={KS} temp={TEMP} out_tok={OUT_TOK}", flush=True)
    print(f"{LOG} PLENS={PLENS} BATCHES={BATCHES} max_len={MAX_LEN}", flush=True)

    tok = PreTrainedTokenizerFast.from_pretrained(TARGET)
    vocab = tok.vocab_size
    random.seed(42)
    SAFE = [i for i in range(9, min(vocab - 3, 4093)) if i != 8]
    def make_prompt(n):
        return [1] + random.choices(SAFE, k=n - 1)
    def sp_factory():
        return SamplingParams(temperature=TEMP, top_p=0.9, top_k=50,
                              max_tokens=OUT_TOK, min_tokens=min(OUT_TOK, 8),
                              stop_token_ids=[2], detokenize=False)

    results = {
        "target": TARGET, "draft": DRAFT, "temp": TEMP, "vllm_version": vllm.__version__,
        "transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "max_model_len": MAX_LEN, "out_tokens": OUT_TOK, "warmup": WARMUP, "measure": MEASURE,
        "gpu_memory_utilization": GPU_MEM, "configs": [],
    }
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    # baseline throughput per (plen,batch) to compute speedup
    baseline = {}
    for k in KS:
        label = "baseline(no-spec)" if k == 0 else f"draft_model K={k}"
        print(f"{LOG} ===== building engine: {label} =====", flush=True)
        t0 = time.time()
        try:
            llm = build_llm(TARGET, DRAFT, k, GPU_MEM, MAX_LEN)
        except Exception as e:
            print(f"{LOG} BUILD FAILED for K={k}: {e}", flush=True)
            traceback.print_exc()
            results["configs"].append({"k": k, "status": f"build_error: {type(e).__name__}: {e}"})
            with open(OUT, "w") as f:
                json.dump(results, f, indent=2)
            continue
        print(f"{LOG} engine built in {time.time()-t0:.1f}s", flush=True)

        for plen in PLENS:
            for batch in BATCHES:
                print(f"{LOG} K={k} plen={plen} batch={batch} ({time.strftime('%T')})", flush=True)
                try:
                    summ = run_config(llm, sp_factory, plen, batch, OUT_TOK, WARMUP, MEASURE, make_prompt)
                    tps = summ["tok_per_s_median"]
                    if k == 0:
                        baseline[(plen, batch)] = tps
                        speedup = 1.0
                    else:
                        b = baseline.get((plen, batch))
                        speedup = (tps / b) if b else None
                    cfg = {"k": k, "label": label, "prompt_len": plen, "batch": batch,
                           "status": "ok", "speedup_vs_baseline": speedup, "summary": summ}
                    acc = summ.get("acceptance")
                    print(f"{LOG}   -> tok/s={tps:.1f} speedup={speedup if speedup else '-'} accept={acc}", flush=True)
                except Exception as e:
                    print(f"{LOG}   ERROR: {e}", flush=True)
                    traceback.print_exc()
                    cfg = {"k": k, "prompt_len": plen, "batch": batch,
                           "status": f"error: {type(e).__name__}: {e}"}
                results["configs"].append(cfg)
                with open(OUT, "w") as f:
                    json.dump(results, f, indent=2)

        # free engine before next K
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(3)

    print(f"{LOG} DONE {time.strftime('%F %T')} wrote {len(results['configs'])} configs to {OUT}", flush=True)
    print(f"\n{LOG} === SUMMARY ===", flush=True)
    print(f"{'K':>3} {'plen':>6} {'batch':>6} {'tok/s':>10} {'speedup':>9} status", flush=True)
    for c in results["configs"]:
        s = c.get("summary")
        if s:
            sp = c.get("speedup_vs_baseline")
            print(f"{c['k']:>3} {c['prompt_len']:>6} {c['batch']:>6} {s['tok_per_s_median']:>10.1f} "
                  f"{(f'{sp:.2f}x' if sp else '-'):>9} ok", flush=True)
        else:
            print(f"{c.get('k','?'):>3} {c.get('prompt_len',''):>6} {c.get('batch',''):>6} "
                  f"{'':>10} {'':>9} {c.get('status','?')}", flush=True)


if __name__ == "__main__":
    main()
