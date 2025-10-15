# [GenomeOcean: An Efficient Genome Foundation Model Trained on Large-Scale Metagenomic Assemblies](https://www.biorxiv.org/content/10.1101/2025.01.30.635558v1)

![Figure 1](figures/Overview.jpeg)


## 1. Installation

### 1.1 Docker or Apptainer
We provide a container images for GenomeOcean. See `docker/` and `/apptainer/` for more information.


### 1.2 uv

#### Pre-requisites: Python 3.13 (tested), older version should work as well
```bash
# Create a new virtual environment
uv venv GO --python 3.13
source GO/bin/activate
# install required packages
uv pip install --no-cache-dir --upgrade transformers[torch]==4.57.1 peft accelerate bitsandbytes pandas vllm==0.11.0 biopython scikit-learn pyrodigal
MAX_JOBS=4 uv pip install --no-build-isolation flash_attn==2.8.3
```

#### Install GenomeOcean package

from source:
```bash
# clone the repo
git clone https://github.com/jgi-genomeocean/genomeocean
cd genomeocean/
export VLLM_WORKER_MULTIPROC_METHOD=spawn
uv pip install -e .
python -m unittest unittests.py
```

## 2. Usage

GenomeOcean is compatible with all the standard HuggingFace APIs. We publish the following checkpoints on HuggingFace:

| Checkpoint                                   | Description                                                  |
| -------------------------------------------- | ------------------------------------------------------------ |
| [pGenomeOcean/GenomeOcean-100M](https://huggingface.co/DOEJGI/GenomeOcean-100M)                  | The base model with 100M parameters. Support maximum sequence length of 1024 tokens (~5,100 bp). |
| [pGenomeOcean/GenomeOcean-500M](https://huggingface.co/DOEJGI/GenomeOcean-500M)                  | The base model with 500M parameters. Support maximum sequence length of 1024 tokens (~5,100 bp). |
| [pGenomeOcean/GenomeOcean-4B](https://huggingface.co/DOEJGI/GenomeOcean-4B)                    | The base model with 4B parameters. Support maximum sequence length of 10240 tokens (~51,000 bp). |
| [pGenomeOcean/GenomeOcean-4B-bgcFM](https://huggingface.co/DOEJGI/GenomeOcean-4B-bgcFM)             | The `GenomeOcean-4B` model finetuned on 11M biosynthetic gene clusters (BGC) sequences. Support maximum sequence length of 10240 tokens (~51,000 bp). |
| [pGenomeOcean/GenomeOcean-4B-Artificial-Detector](https://huggingface.co/DOEJGI/GenomeOcean-4B-Artificial-Detector)   | The `GenomeOcean-4B` model finetuned to detected GenomeOcean-generated sequences. A binary classifier where label `0` indicate artificial sequences. |

We recommend using the `GenomeOcean-4B` model for general-purpose genome sequence analysis. The `GenomeOcean-4B-bgcFM` model is fine-tuned on BGC sequences and can be used for BGC-related tasks. The smaller models (`GenomeOcean-100M` and `GenomeOcean-500M`) can be used when GPU memory is limited, but the performance may be compromised.

### 2.1. Sequence Embedding
```bash
cd genomeocean/examples
python embedding_sequences.py \
    --model_dir DOEJGI/GenomeOcean-4B \
    --sequence_file ../sample_data/dna_sequences.txt \
    --model_max_length 10240 \
    --batch_size 10 \
    --output_file outputs/embeddings.npy
```

### 2.2. Sequence Generation
```bash
cd genomeocean/examples
python generate_sequences.py \
    --model_dir DOEJGI/GenomeOcean-4B \
    --promptfile ../sample_data/dna_sequences.txt \
    --out_prefix outputs/generated \
    --out_format fa \
    --num 10 \
    --min_seq_len 100 \
    --max_seq_len 100 \
    --temperature 1.3 \
    --top_k -1 \
    --top_p 0.7 \
    --max_repeats 100 \
    --presence_penalty 0.5 \
    --frequency_penalty 0.5 \
    --repetition_penalty 1.0 \
    --seed 123 \
    --sort_by_orf_length
```

### 2.3. more examples

please see the folder `examples/`


## 3. Contribute

Please submit pull requests and issues to the main branch.

## 4. Citation

Zhou, Zhihan, et al. "GenomeOcean: An Efficient Genome Foundation Model Trained on Large-Scale Metagenomic Assemblies." bioRxiv (2025): 2025-01. (https://www.biorxiv.org/content/10.1101/2025.01.30.635558v1.full)

```
@article{zhou2025genomeocean,
  title={GenomeOcean: An Efficient Genome Foundation Model Trained on Large-Scale Metagenomic Assemblies},
  author={Zhou, Zhihan and Riley, Robert and Kautsar, Satria and Wu, Weimin and Egan, Rob and Hofmeyr, Steven and Goldhaber-Gordon, Shira and Yu, Mutian and Ho, Harrison and Liu, Fengchen and others},
  journal={bioRxiv},
  pages={2025--01},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```

## 5. Copyright Notice

genomeocean: a pretrained microbial genome foundational model (genomeoceanLLM) ” Copyright (c) 2025, The Regents of the
University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy) and Northwestern University. All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.


