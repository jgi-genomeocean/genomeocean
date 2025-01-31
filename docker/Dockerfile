# Use the CUDA developer base image for Ubuntu 20.04
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/conda/bin:$PATH" \
    CUDA_HOME="/usr/local/cuda"

ENV TRANSFORMERS_CACHE=/workspace/model_cache

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    curl \
    bzip2 \
    libx11-6 \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda init bash

# Create and activate conda environment and install Python
RUN conda create -n GO python=3.11 -y && \
    echo "conda activate GO" >> ~/.bashrc && \
    conda clean -a -y
ENV CONDA_DEFAULT_ENV=GO
ENV PATH="/opt/conda/envs/GO/bin:$PATH"

# Clone GenomeOcean repository
WORKDIR /workspace
RUN git clone https://github.com/jgi-genomeocean/genomeocean.git
WORKDIR /workspace/genomeocean

# Install Python dependencies
RUN pip install --no-cache-dir torch==2.4.0 

# Install GenomeOcean
RUN pip install genomeocean

# Set default command
CMD ["/bin/bash"]