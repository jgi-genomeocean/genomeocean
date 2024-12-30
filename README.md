# GenomeOcean: A metagenomics foundation model

A Python library for GenomeOcean inference.

## 1. Installation

```bash
conda create -n GO python=3.11
conda activate GO
pip install torch==2.4.0 # need to do this first since other packages depend on it
pip install -r requirements.txt
pip install .
```


## 2. Usage

GenomeOcean is compatible with all the standard HuggingFace APIs. Our implement further wraps it with vLLM for generation efficiency and quality. 

### 2.1 Our implementation (Recommended)

#### 2.1.1 Sequence Generation
```python
from genomeocean.generation import SequenceGenerator

sequences = [
    "GCCGCTAAAAAGCGACCAGAATGATCCAAAAAAGAAGGCAGGCCAGCACCATCCGTTTTTTACAGCTCCAGAACTTCCTTT", 
    "CAGTCAGTGGCTAGCATGCTAGCATCGATCGATCGATCGATCGATCGATCGATCGGTGCATGCTAGCATCGATCGATCGAA"
]
seq_gen = SequenceGenerator(
    model_dir='pGenomeOcean/GenomeOcean-4B', # model_dir can also be the path to a local copy of the model
    prompts=sequences, # Provide a list of DNA sequences as prompts
    promptfile='', # or provide a file contains DNA sequences as prompts
    num=10, # number of sequences to generate for each prompt
    min_seq_len=100, # minimum length of generated sequences in token, set it as expected bp length // 4 (e.g., set it as 1000 for 4kb)
    max_seq_len=100, # maximum length of generated sequences in token, max value is 10240
    temperature=1.3, # temperature for sampling
    top_k=-1, # top_k for sampling
    top_p=0.7, # top_p for sampling
    presence_penalty=0.5, # presence penalty for sampling
    frequency_penalty=0.5, # frequency penalty for sampling
    repetition_penalty=1.0, # repetition penalty for sampling
    seed=123, # random seed for sampling
)
all_generated = seq_gen.generate_sequences(
    prepend_prompt_to_output=True, # set to False to only save the generated sequence
    max_repeats=0, # set to k to remove sequences with more than k% simple repeats, set to 0 to return all the generated sequences
)
seq_gen.save_sequences(
    all_generated, 
    out_prefix='debug/seqs', # output file prefix, the final output file will be named as path/to/output.txt or path/to/output.fa
    out_format='txt' # or 'fa' for fasta format,
)
```

#### 2.1.2 Sequence Embedding
```python
from genomeocean.llm_utils import LLMUtils

sequences = [
    "GCCGCTAAAAAGCGACCAGAATGATCCAAAAAAGAAGGCAGGCCAGCACCATCCGTTTTTTACAGCTCCAGAACTTCCTTT", 
    "CAGTCAGTGGCTAGCATGCTAGCATCGATCGATCGATCGATCGATCGATCGATCGGTGCATGCTAGCATCGATCGATCGAA"
]
llm = LLMUtils('pGenomeOcean/GenomeOcean-4B')
embeddings = llm.embedding(sequences, batch_size=2) # batch_size can be adjusted based on GPU memory and sequence length
print(embeddings.shape)  # (2, 3072)
print(type(embeddings)) # numpy.ndarray

```

### 2.2 HuggingFace API
```python
# Load model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "pGenomeOcean/GenomeOcean-4B",
    trust_remote_code=True,
    padding_side="left",
)
model = AutoModelForCausalLM.from_pretrained(
    "pGenomeOcean/GenomeOcean-4B",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2",
).to("cuda") 

# Embedding
sequences = [
    "GCCGCTAAAAAGCGACCAGAATGATCCAAAAAAGAAGGCAGGCCAGCACCATCCGTTTTTTACAGCTCCAGAACTTCCTTT", 
    "CAGTCAGTGGCTAGCATGCTAGCATCGATCGATCGATCGATCGATCGATCGATCGGTGCATGCTAGCATCGATCGATCGAA"
]
output = tokenizer.batch_encode_plus(
    sequences,
    max_length=10240,
    return_tensors='pt',
    padding='longest',
    truncation=True
)
input_ids = output['input_ids'].cuda()
attention_mask = output['attention_mask'].cuda()
model_output = model.forward(input_ids=input_ids, attention_mask=attention_mask)[0].detach().cpu()
attention_mask = attention_mask.unsqueeze(-1).detach().cpu()
embedding = torch.sum(model_output * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
print(f"Shape: {embedding.shape}") # (2, 3072)

# Generation
sequences = [
    "GCCGCTAAAAAGCGACCAGAATGATCCAAAAAAGAAGGCAGGCCAGCACCATCCGTTTTTTACAGCTCCAGAACTTCCTTT", 
    "CAGTCAGTGGCTAGCATGCTAGCATCGATCGATCGATCGATCGATCGATCGATCGGTGCATGCTAGCATCGATCGATCGAA"
]
input_ids = tokenizer(sequence, return_tensors='pt', padding=True)["input_ids"]
input_ids = input_ids[:, :-1].to("cuda")   # remove the [SEP] token at the end
model_output = model.generate(
    input_ids=input_ids,
    min_new_tokens=10,
    max_new_tokens=10,
    do_sample=True,
    top_p=0.9,
    temperature=1.0,
    num_return_sequences=1,
)
generated = tokenizer.decode(model_output[0]).replace(" ", "")[5+len(sequence):]
print(f"Generated sequence: {generated}")

```



## 3. Contribute

Please submit pull requests to the main branch.

## 4. Citation

### `LICENSE`
Choose a suitable license for your library (e.g., MIT, Apache 2.0).

