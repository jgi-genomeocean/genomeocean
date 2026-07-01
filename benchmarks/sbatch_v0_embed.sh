#!/bin/bash
#SBATCH --job-name=gov0emb
#SBATCH --account=m342_g
#SBATCH --qos=debug
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/pscratch/sd/z/zhwang/genomeocean-vllm-v1/bench/v0_embed_%j.log
#SBATCH --error=/pscratch/sd/z/zhwang/genomeocean-vllm-v1/bench/v0_embed_%j.err

set -e
BENCH=/pscratch/sd/z/zhwang/genomeocean-vllm-v1/bench
REPO=/pscratch/sd/z/zhwang/genomeocean-vllm-v1/main
HF_CACHE=/pscratch/sd/z/zhwang/.cache/huggingface

echo "=== V0 embedding benchmark — $(date) ==="
echo "Node: $(hostname)  Job: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

shifter --image=nvcr.io/nvidia/vllm:25.09-py3 \
  --volume="${HF_CACHE}:/hfcache" \
  --volume="${REPO}:/repo" \
  --volume="${BENCH}:/bench" \
  -- bash -c "
    export HF_HOME=/hfcache
    cd /repo && pip install -e . -q --no-deps 2>&1 | tail -3
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    GO_LABEL=v0 \
    GO_MODEL=DOEJGI/GenomeOcean-4B \
    GO_BATCHES=1,8,32,64 \
    GO_SEQ_LENS=1024,10240 \
    GO_WARMUP=2 GO_MEASURE=3 \
    GO_OUT=/bench/embed_v0.json \
    python3 /bench/bench_embed_a100.py
  "

echo ""
echo "=== V0 embedding done — $(date) ==="
ls -la $BENCH/embed_v0.json
