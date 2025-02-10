# [GenomeOcean: An Efficient Genome Foundation Model Trained on Large-Scale Metagenomic Assemblies](https://www.biorxiv.org/content/10.1101/2025.01.30.635558v1)

![Figure 1](figures/Overview.jpeg)


## 1. Installation

### 1.1 Docker
We provide a Docker image for GenomeOcean. See `docker/` for more information.


### 1.2 Conda/pip

#### Pre-requisites: Python 3.11 tested
```bash
# Create a new conda environment
conda create -n GO python=3.11
conda activate GO
```

#### Install GenomeOcean package

from pip:
```bash
pip install genomeocean
```

from source:
```bash
pip install git+https://github.com/jgi-genomeocean/genomeocean
```

## 2. Usage

### 2.1. Sequence Generation

Generate 100 BGCs in zero-shot mode
```bash
go_generate.py --model bgc --zero_shot --preset creative_long --num 100
```
Autocomplete a known siderophore BGC from MIBiG (https://mibig.secondarymetabolites.org/repository/BGC0000947.5)
```bash
go_generate.py --model bgc --prompts CGTTTAAAAATAGTGGAAGATCGGCGGTTCATGCATCAAGAAGTTTTGATGGGAGATGTTCCACGCATAGGCGCCCGCTAAGGTAAACACAACGTAATCACCAATATCGACGTGGGCGACGTGTTGATTGCGTGCCAGCACATCTTTGGGCGTACAAAGCTGGCCGACAAAGGTCGCTTGTGCATGCTGAATTGGATGAGAAATCTCATGGTGTTGCTCGCTTTTCAGGATGACGAATGGATGATCATGGCTCTGCGCCGCTGGCGTGCGGAAGTGGTGAGTTCCTCCGCGCGCGATGACAAAGTTTTCGCCAAGATTTTGTTTAATATCCAGTACTTCCATCACGTAATAACCGCATGCCGCCGTGATGAAACGTCCGCATTCAAAACGCAGTGTCCAGTCTTGCACTTGCTCTTTGGCGATGAGGAATTCCAACTTGTCGCAAAACTCCATCCACGGAAAGTGTTGCTCAGGGTTTTGGTAGTTAATCCCCATACCGCCGCCCAAGTTGATCATCAGCTCACTGAGTTTGAACTCCTGCTGCCACGTTTTCACGACTTGGAAATAGCGCTGCATTAACGCTAAGTGACGTTCAACATCGAGCTGATGCGACATCAAGTGAAAATGGAAGCCTTTGAGTGACACTTGCGGGAAATCGCGCAGCAACATCAAAGCATTACTCAGCTCGGACTCGTCCAAGCCAAATGGAGTCGGCTTGCCACCCATCGCCAACTTGCTGAGCGTGATGTCGCCAATATCGATGTTCATACGCA --preset conservative_long --num 100
```

### 2.2. Sequence Embedding
```python
from genomeocean.llm_utils import LLMUtils

sequences = [
    "GCCGCTAAAAAGCGACCAGAATGATCCAAAAAAGAAGGCAGGCCAGCACCATCCGTTTTTTACAGCTCCAGAACTTCCTTT", 
    "CAGTCAGTGGCTAGCATGCTAGCATCGATCGATCGATCGATCGATCGATCGATCGGTGCATGCTAGCATCGATCGATCGAA"
]
llm = LLMUtils('pGenomeOcean/GenomeOcean-4B')
embeddings = llm.predict(sequences, batch_size=2, do_embedding=True) # batch_size can be adjusted based on GPU memory and sequence length
print(embeddings.shape)  # (2, 3072)
print(type(embeddings)) # numpy.ndarray

```

### 2.3. Loss scores scanning
```python

```

### 2.4. more examples

please see the folder `examples/`

### 2.5. HuggingFace

GenomeOcean is compatible with all the standard HuggingFace APIs. We publish the following checkpoints on HuggingFace:

| Checkpoint                                   | Description                                                  |
| -------------------------------------------- | ------------------------------------------------------------ |
| pGenomeOcean/GenomeOcean-4B                  | The base model with 4B parameters. Support maximum sequence length of 10240 tokens (~51,000 bp). |
| pGenomeOcean/GenomeOcean-4B-bgcFM            | The `GenomeOcean-4B` model finetuned on 11M biosynthetic gene clusters (BGC) sequences. Support maximum sequence length of 10240 tokens (~51,000 bp). |
| pGenomeOcean/GenomeOcean-Artificial-Detector | The `GenomeOcean-4B` model finetuned to detected GenomeOcean-generated sequences. A binary classifier where label `0` indicate artificial sequences. |


## 3. Contribute

Please submit pull requests and issues to the main branch.

## 4. Citation

Zhou, Zhihan, et al. "GenomeOcean: An Efficient Genome Foundation Model Trained on Large-Scale Metagenomic Assemblies." bioRxiv (2025): 2025-01. (https://www.biorxiv.org/content/10.1101/2025.01.30.635558v1.full)

### Copyright Notice

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


