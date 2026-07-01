#!/usr/bin/env python3
"""
GenomeOcean embedding benchmark — Perlmutter A100 edition.

transformers AutoModel forward pass on GO-4B. Compares transformers 4.x vs 5.x.
Embedding doesn't shard well across GPUs — runs on 1 visible GPU.

Env vars:
  GO_LABEL        label for output (e.g. "v0_embed" or "v1_embed")
  GO_MODEL        HF model id (default: pGenomeOcean/GenomeOcean-4B)
  GO_BATCHES      comma list (default "1,8,32,64")
  GO_SEQ_LENS     comma list (default "1024,10240")
  GO_WARMUP       (default 2)
  GO_MEASURE      (default 3)
  GO_OUT          output JSON path
"""
import gc, json, os, statistics, sys, time, random, traceback


def main():
    import torch
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import transformers
    from transformers import AutoModel, PreTrainedTokenizerFast

    LABEL    = os.environ.get("GO_LABEL", "v?")
    MODEL    = os.environ.get("GO_MODEL", "DOEJGI/GenomeOcean-4B")
    BATCHES  = [int(x) for x in os.environ.get("GO_BATCHES", "1,8,32,64").split(",")]
    SEQ_LENS = [int(x) for x in os.environ.get("GO_SEQ_LENS", "1024,10240").split(",")]
    WARMUP   = int(os.environ.get("GO_WARMUP", "2"))
    MEASURE  = int(os.environ.get("GO_MEASURE", "3"))
    OUT      = os.environ.get("GO_OUT", f"/bench/embed_{LABEL}.json")

    LOG = f"[embed {LABEL}]"
    print(f"{LOG} START at {time.strftime('%F %T')} model={MODEL}", flush=True)
    print(f"{LOG} python={sys.version.split()[0]} torch={torch.__version__} transformers={transformers.__version__}", flush=True)
    print(f"{LOG} BATCHES={BATCHES} SEQ_LENS={SEQ_LENS} WARMUP={WARMUP} MEASURE={MEASURE}", flush=True)
    print(f"{LOG} GPU count: {torch.cuda.device_count()}", flush=True)

    print(f"{LOG} loading model...", flush=True)
    t0 = time.time()
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16, trust_remote_code=False).to("cuda")
    model.config.use_cache = False
    model.eval()
    tok = PreTrainedTokenizerFast.from_pretrained(MODEL)
    print(f"{LOG} loaded in {time.time()-t0:.1f}s  vocab={tok.vocab_size}", flush=True)

    random.seed(42)
    vocab_size = tok.vocab_size
    SAFE_IDS = [i for i in range(9, min(vocab_size - 3, 4093)) if i != 8]

    results = {
        "label": LABEL,
        "model": MODEL,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "warmup": WARMUP,
        "measure": MEASURE,
        "configs": []
    }
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    for seq_len in SEQ_LENS:
        for batch in BATCHES:
            print(f"{LOG} === seq_len={seq_len} batch={batch} === ({time.strftime('%T')})", flush=True)
            cfg = {"seq_len": seq_len, "batch": batch, "tokens_per_pass": seq_len * batch}

            try:
                input_ids = torch.tensor(
                    [[1] + random.choices(SAFE_IDS, k=seq_len - 1) for _ in range(batch)],
                    device="cuda"
                )
                attn_mask = torch.ones_like(input_ids)

                with torch.no_grad():
                    for _ in range(WARMUP):
                        _ = model(input_ids=input_ids, attention_mask=attn_mask)
                torch.cuda.synchronize()

                raw = []
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    for i in range(MEASURE):
                        torch.cuda.synchronize()
                        t_start = time.perf_counter()
                        out = model(input_ids=input_ids, attention_mask=attn_mask)
                        torch.cuda.synchronize()
                        wall = time.perf_counter() - t_start
                        toks = seq_len * batch
                        raw.append({
                            "iter": i, "wall_s": wall, "tokens": toks,
                            "tok_per_s": toks / wall,
                            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3
                        })

                walls = [r["wall_s"] for r in raw]
                tps = [r["tok_per_s"] for r in raw]
                vrams = [r["peak_vram_gb"] for r in raw]
                summ = {
                    "n_iters": len(raw),
                    "wall_s_median": statistics.median(walls),
                    "wall_s_mean": statistics.mean(walls),
                    "wall_s_stdev": statistics.stdev(walls) if len(walls) > 1 else 0.0,
                    "tok_per_s_median": statistics.median(tps),
                    "tok_per_s_mean": statistics.mean(tps),
                    "peak_vram_gb_max": max(vrams),
                }
                cfg.update({"status": "ok", "raw": raw, "summary": summ})
                print(f"{LOG}   -> median wall={summ['wall_s_median']:.3f}s tok/s={summ['tok_per_s_median']:.1f} vram={summ['peak_vram_gb_max']:.2f}GB", flush=True)

            except torch.cuda.OutOfMemoryError:
                print(f"{LOG}   OOM seq_len={seq_len} batch={batch}", flush=True)
                cfg["status"] = "oom"
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"{LOG}   ERROR: {e}", flush=True)
                traceback.print_exc()
                cfg["status"] = f"error: {type(e).__name__}: {e}"

            results["configs"].append(cfg)
            with open(OUT, "w") as f:
                json.dump(results, f, indent=2)

    print(f"{LOG} DONE at {time.strftime('%F %T')} — wrote to {OUT}", flush=True)


if __name__ == "__main__":
    main()
