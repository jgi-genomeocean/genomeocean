#!/bin/bash
#SBATCH --job-name=gospec
#SBATCH --account=<YOUR_SLURM_ACCOUNT>
#SBATCH --qos=<YOUR_SLURM_QOS>
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=spec_%j.log
#SBATCH --error=spec_%j.err

set -e
BENCH=${GO_BENCH_DIR:-$PWD}
REPO=${GO_REPO_DIR:-$PWD/..}
HF_CACHE=${HF_HOME:-$HOME/.cache/huggingface}

echo "=== Speculative decoding benchmark (draft_model) — $(date) ==="
echo "Node: $(hostname)  Job: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

shifter --image=nvcr.io/nvidia/vllm:26.06-py3 \
  --volume="${HF_CACHE}:/hfcache" \
  --volume="${REPO}:/repo" \
  --volume="${BENCH}:/bench" \
  -- bash -c "
    export HF_HOME=/hfcache
    mkdir -p /tmp/culibs
    REAL_LIBCUDA=\$(ls /usr/local/cuda/compat/lib/libcuda.so.1 /usr/lib64/libcuda.so.1 2>/dev/null | head -1)
    ln -sf \"\$REAL_LIBCUDA\" /tmp/culibs/libcuda.so
    export TRITON_LIBCUDA_PATH=/tmp/culibs
    export LD_LIBRARY_PATH=/tmp/culibs:\$LD_LIBRARY_PATH
    cd /repo && pip install -e . -q --no-deps 2>&1 | tail -3
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    GO_TARGET=DOEJGI/GenomeOcean-4B \
    GO_DRAFT=DOEJGI/GenomeOcean-100M \
    GO_KS=0,2,3,4 \
    GO_TEMP=0.7 \
    GO_GPU_MEM=0.85 \
    GO_MAX_LEN=4608 \
    GO_OUT_TOK=512 \
    GO_WARMUP=1 GO_MEASURE=3 \
    GO_PROMPTS=1024 \
    GO_BATCHES=1,8 \
    GO_OUT=/bench/specdec_results.json \
    python3 /bench/bench_specdec_a100.py
  "

echo ""
echo "=== spec decode done — $(date) ==="
ls -la $BENCH/specdec_results.json
