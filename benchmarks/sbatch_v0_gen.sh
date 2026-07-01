#!/bin/bash
#SBATCH --job-name=gov0gen
#SBATCH --account=m342_g
#SBATCH --qos=debug
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --time=00:30:00
#SBATCH --output=/pscratch/sd/z/zhwang/genomeocean-vllm-v1/bench/v0_gen_%j.log
#SBATCH --error=/pscratch/sd/z/zhwang/genomeocean-vllm-v1/bench/v0_gen_%j.err

set -e
BENCH=/pscratch/sd/z/zhwang/genomeocean-vllm-v1/bench
REPO=/pscratch/sd/z/zhwang/genomeocean-vllm-v1/main
HF_CACHE=/pscratch/sd/z/zhwang/.cache/huggingface

echo "=== V0 generation benchmark — $(date) ==="
echo "Node: $(hostname)  Job: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Run V0 on TP=1, TP=2, TP=4 sequentially in the same job
# Each TP config loads the model fresh; KV cache memory cleaned between
for TP in 1 2 4; do
  echo ""
  echo "=== V0 / GO-4B / tp=$TP at $(date) ==="
  shifter --image=nvcr.io/nvidia/vllm:25.09-py3 \
    --volume="${HF_CACHE}:/hfcache" \
    --volume="${REPO}:/repo" \
    --volume="${BENCH}:/bench" \
    -- bash -c "
      export HF_HOME=/hfcache
      # BUG FIX 2b: Triton/Inductor (torch.compile) can't find libcuda.so at TP>=2,
      # only the compat stub libcuda.so.1 exists. Symlink it and point Triton at it.
      mkdir -p /tmp/culibs
      REAL_LIBCUDA=\$(ls /usr/local/cuda/compat/lib/libcuda.so.1 /usr/lib64/libcuda.so.1 2>/dev/null | head -1)
      ln -sf \"\$REAL_LIBCUDA\" /tmp/culibs/libcuda.so
      export TRITON_LIBCUDA_PATH=/tmp/culibs
      export LD_LIBRARY_PATH=/tmp/culibs:\$LD_LIBRARY_PATH
      cd /repo && pip install -e . -q --no-deps 2>&1 | tail -3
      export VLLM_USE_V1=0
      export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      GO_ENGINE=v0 \
      GO_MODEL=DOEJGI/GenomeOcean-4B \
      GO_TP=${TP} \
      GO_GPU_MEM=0.85 \
      GO_MAX_LEN=11264 \
      GO_OUT_TOK=512 \
      GO_WARMUP=2 GO_MEASURE=5 \
      GO_PROMPTS=1024,10240 \
      GO_BATCHES=1,8,32 \
      GO_ENFORCE_EAGER=0 \
      GO_OUT=/bench/results_v0_genomeocean-4b_tp${TP}.json \
      python3 /bench/bench_gen_a100.py
    "
  # Brief pause to let GPU memory release between TP configs
  sleep 10
done

echo ""
echo "=== V0 generation done — $(date) ==="
ls -la $BENCH/results_v0_*.json
