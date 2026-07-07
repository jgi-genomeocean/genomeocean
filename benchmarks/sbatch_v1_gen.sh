#!/bin/bash
#SBATCH --job-name=gov1gen
#SBATCH --account=<YOUR_SLURM_ACCOUNT>
#SBATCH --qos=<YOUR_SLURM_QOS>
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --time=00:30:00
#SBATCH --output=v1_gen_%j.log
#SBATCH --error=v1_gen_%j.err

set -e
BENCH=${GO_BENCH_DIR:-$PWD}
REPO=${GO_REPO_DIR:-$PWD/..}
HF_CACHE=${HF_HOME:-$HOME/.cache/huggingface}

echo "=== V1 generation benchmark — $(date) ==="
echo "Node: $(hostname)  Job: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

for TP in 1 2 4; do
  echo ""
  echo "=== V1 / GO-4B / tp=$TP at $(date) ==="
  shifter --image=nvcr.io/nvidia/vllm:26.06-py3 \
    --volume="${HF_CACHE}:/hfcache" \
    --volume="${REPO}:/repo" \
    --volume="${BENCH}:/bench" \
    -- bash -c "
      export HF_HOME=/hfcache
      # BUG FIX 2b: same libcuda symlink for TP>=2 torch.compile robustness.
      mkdir -p /tmp/culibs
      REAL_LIBCUDA=\$(ls /usr/local/cuda/compat/lib/libcuda.so.1 /usr/lib64/libcuda.so.1 2>/dev/null | head -1)
      ln -sf \"\$REAL_LIBCUDA\" /tmp/culibs/libcuda.so
      export TRITON_LIBCUDA_PATH=/tmp/culibs
      export LD_LIBRARY_PATH=/tmp/culibs:\$LD_LIBRARY_PATH
      cd /repo && pip install -e . -q --no-deps 2>&1 | tail -3
      export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      GO_ENGINE=v1 \
      GO_MODEL=DOEJGI/GenomeOcean-4B \
      GO_TP=${TP} \
      GO_GPU_MEM=0.85 \
      GO_MAX_LEN=11264 \
      GO_OUT_TOK=512 \
      GO_WARMUP=2 GO_MEASURE=5 \
      GO_PROMPTS=1024,10240 \
      GO_BATCHES=1,8,32 \
      GO_ENFORCE_EAGER=0 \
      GO_OUT=/bench/results_v1_genomeocean-4b_tp${TP}.json \
      python3 /bench/bench_gen_a100.py
    "
  # Brief pause + kill any zombie EngineCore processes
  sleep 10
  fuser /dev/nvidia* 2>/dev/null | xargs -r kill -9 2>/dev/null || true
  sleep 5
done

echo ""
echo "=== V1 generation done — $(date) ==="
ls -la $BENCH/results_v1_*.json
