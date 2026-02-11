# Use official NVIDIA vLLM container for CUDA 13 and Blackwell support
FROM nvcr.io/nvidia/vllm:25.09-py3

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/bin:$PATH" \
    CUDA_HOME="/usr/local/cuda" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_USE_V1=0 \
    HF_HOME="/workspace/model_cache" \
    TMPDIR="/workspace/tmp" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"





# Install system dependencies (basic ones, base image has most)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git curl bzip2 libx11-6 build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create directories
RUN mkdir -p /workspace/model_cache /workspace/tmp /workspace/build_temp
ENV TMPDIR="/workspace/build_temp"

# Copy the local refactored GenomeOcean code instead of cloning from GitHub
COPY . /genomeocean
WORKDIR /genomeocean

# Install Python dependencies into SYSTEM python (inheriting torch/vllm)
# Fix PEP 668 "externally managed" error on newer Debian/Ubuntu
RUN rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED && \
    pip install uv --break-system-packages && \
    # Install dependencies WITHOUT upgrading torch/vllm/transformers which are provided by the container
    uv pip install --system --no-cache-dir peft accelerate bitsandbytes pandas biopython scikit-learn pyrodigal \
        cloudpickle msgspec outlines sentencepiece pydantic ray psutil && \
    uv pip install --system --no-deps -e .




# Clean up
RUN rm -rf /workspace/build_temp

# Set default command
CMD ["/bin/bash"]
