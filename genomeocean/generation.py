""" class for sequence generation

Usage:
    from genomeocean.generation import SequenceGenerator
    sequences = [
        "GCCGCTAAAAAGCGACCAGAATGATCCAAAAAAGAAGGCAGGCCAGCACCATCCGTTTTTTACAGCTCCAGAACTTCCTTT", 
        "CAGTCAGTGGCTAGCATGCTAGCATCGATCGATCGATCGATCGATCGATCGATCGGTGCATGCTAGCATCGATCGATCGAA"
    ]
    seq_gen = SequenceGenerator(
        model_dir='pGenomeOcean/GenomeOcean-4B', 
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
"""
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'  # to avoid error in multi-gpu generation

from genomeocean.dnautils import find_tandem_repeats_percentage
from genomeocean.llm_utils import LLMUtils
import pandas as pd
from Bio import SeqIO
import textwrap

class SequenceGenerator:
    def __init__(
        self, 
        model_dir='', 
        promptfile='', 
        prompts=[],
        num=100, 
        min_seq_len=1024, 
        max_seq_len=10240,
        temperature=1.3,
        top_k=-1,
        top_p=0.7,
        presence_penalty=0.5,
        frequency_penalty=0.5,
        repetition_penalty=1.0,
        seed=123,
        ):
        assert promptfile or prompts, "prompts (A list of str) or promptfile (A file contains DNA sequences) must be provided"
        if prompts and promptfile:
            print("+++Warning: Both prompts and promptfile are provided, only prompts will be used")
        self.model_dir = model_dir
        self.promptfile = promptfile
        self.prompts = prompts
        self.num = num
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.repetition_penalty = repetition_penalty
        self.seed = seed

    def _load_prompts(self):
        if self.prompts:
            return self.prompts

        allowed_promptfile_types = ["txt", "fa", "fasta", "csv", "tsv"]
        assert self.promptfile.split(".")[-1] in allowed_promptfile_types, f"Prompt file must be one of {allowed_promptfile_types}"
        
        if self.promptfile.endswith('.fa') or self.promptfile.endswith('.fasta'):
            return [str(r.seq) for r in SeqIO.parse(self.promptfile, 'fasta')]
        else:
            return list(pd.read_csv(self.promptfile, header=None, delimiter="\t" if self.promptfile.endswith('.tsv') else None)[0])



    def generate_sequences(self, prepend_prompt_to_output=False, max_repeats=0):
        prompts = self._load_prompts()
        llm = LLMUtils(model_dir=self.model_dir)
        
        print(f"======First Prompt {prompts[0]}")
        generated = llm.generate(        
            prompts=prompts, 
            num_generation_from_each_prompt=self.num,
            temperature=self.temperature,
            min_length=self.min_seq_len,
            max_length=self.max_seq_len,
            top_k=self.top_k,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=self.repetition_penalty,
            seed=self.seed  # change it to avoid getting the same results
        )
        all_generated = pd.DataFrame(generated, columns=['seq'])
        all_generated['id'] = all_generated.index // self.num # use prompt index as id
        if prepend_prompt_to_output:
            # concatenate prompts to generated sequences, apply to all 
            all_generated['seq'] = all_generated.apply(lambda x: prompts[x.id] + x.seq, axis=1)
        
        if max_repeats > 0:  # remove those containing mostly simple repeats
            # remove identical duplicates
            all_generated = all_generated.drop_duplicates(subset='seq')
            original_len = len(all_generated)
            all_generated['TRF'] = all_generated['seq'].apply(find_tandem_repeats_percentage)
            non_repetitive_sequences = all_generated[all_generated['TRF'] <= max_repeats]
            print(f"Kept {non_repetitive_sequences.shape[0]} out of {original_len} sequences with <= {max_repeats}% simple repeats")
            return non_repetitive_sequences
        return all_generated

    def save_sequences(self, all_generated, out_prefix='generated', out_format="txt"):
        print(f"Saving generated sequences to {out_prefix}.{out_format}")
        out_prefix = out_prefix.rstrip("/")
        if "/" in out_prefix:
            out_dir = "/".join(out_prefix.split("/")[:-1])
            os.makedirs(out_dir, exist_ok=True)

        assert out_format in ["txt", "fa"], "can only output .txt or .fa files, choose from [txt, fa]"

        if out_format == "txt":
            with open(f"{out_prefix}.txt", "w") as f:
                for i, row in all_generated.iterrows():
                    f.write(row['seq'] + '\n')
            print(f"Generated {all_generated.shape[0]} final sequences written to {out_prefix}.txt")

        else:
            with open(f"{out_prefix}.fa", "w") as f:
                for i, row in all_generated.iterrows():
                    f.write(f">{row['id']}_{i}\n")
                    f.write('\n'.join(textwrap.wrap(row['seq'], 80)) + '\n')
            print(f"Generated {all_generated.shape[0]} final sequences written to {out_prefix}.fa")

