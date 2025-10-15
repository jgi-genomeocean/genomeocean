# Use the CUDA developer base image for Ubuntu 24.04
FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04


# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/bin:$PATH" \
    CUDA_HOME="/usr/local/cuda" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    HF_HOME="/workspace/model_cache" \
    VENV_DIR="/genomeocean/GO" \
    PATH="/genomeocean/GO/bin:$PATH" \
    TMPDIR="/workspace/tmp"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    curl \
    bzip2 \
    libx11-6 \
    build-essential \
    python3.13 \
    python3.13-dev \
    python3.13-venv \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3.13 /usr/bin/python

# Create directories for model cache, runtime temporary files, and a build-time temporary directory
RUN mkdir -p /workspace/model_cache /workspace/tmp /workspace/build_temp
# Set TMPDIR to use the new build-time temporary directory
ENV TMPDIR="/workspace/build_temp"


# Clone GenomeOcean repository
WORKDIR /
RUN git clone https://github.com/jgi-genomeocean/genomeocean.git
WORKDIR /genomeocean

# Create venv and install Python dependencies
RUN python3.13 -m venv GO
ENV PATH="/genomeocean/GO/bin:$PATH"
RUN . /genomeocean/GO/bin/activate && \
    pip install uv && \
    uv pip install --no-cache-dir --upgrade transformers[torch] peft accelerate bitsandbytes pandas vllm biopython scikit-learn pyrodigal && \
    MAX_JOBS=4 uv pip install flash-attn --no-cache-dir --no-build-isolation && \
    uv pip install -e .

# Clean up the build directory
RUN rm -rf /workspace/build_temp

# Set default command
CMD ["/bin/bash"]