# Use the CUDA developer base image for Ubuntu 20.04
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/bin:$PATH" \
    CUDA_HOME="/usr/local/cuda" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_USE_V1=0

ENV TRANSFORMERS_CACHE=/workspace/model_cache

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    curl \
    bzip2 \
    libx11-6 \
    build-essential \
    python3.11 \
    python3.11-venv \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python

# Clone GenomeOcean repository
WORKDIR /workspace
RUN git clone https://github.com/jgi-genomeocean/genomeocean.git
WORKDIR /workspace/genomeocean

# Install Python dependencies
RUN pip install -r requirements.txt  \
    && pip install --no-build-isolation flash_attn \
    && pip install git+https://github.com/jgi-genomeocean/genomeocean@update-torch-vllm

# Set default command
CMD ["/bin/bash"]