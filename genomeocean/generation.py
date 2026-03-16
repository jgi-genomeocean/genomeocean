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
    # The generation parameters also accept:
    # filter_compression=True, compression_threshold=1/3 (filters out sequences with low info via gzip)
    # filter_loss=True, loss_threshold=3.5 (filters out sequences with high NLL loss)
    seq_gen.save_sequences(
        all_generated, 
        out_prefix='debug/seqs', # output file prefix, the final output file will be named as path/to/output.txt or path/to/output.fa
        out_format='txt' # or 'fa' for fasta format,
    )
"""
import os
import gzip
import concurrent.futures

from genomeocean.dnautils import find_tandem_repeats_percentage
from genomeocean.llm_utils import LLMUtils
import pandas as pd
from Bio import SeqIO
import textwrap

def _compute_compression_ratio(seq: str) -> float:
    """Return the gzip compressed size divided by original byte size."""
    raw = seq.encode('utf-8')
    if not len(raw):
        return 1.0
    return len(gzip.compress(raw)) / len(raw)

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
        gpu_memory_utilization=0.6,
        filter_compression=False,
        compression_threshold=1/3,
        filter_loss=False,
        loss_threshold=3.5,
        loss_batch_size=8,
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
        self.gpu_memory_utilization = gpu_memory_utilization
        self.filter_compression = filter_compression
        self.compression_threshold = compression_threshold
        self.filter_loss = filter_loss
        self.loss_threshold = loss_threshold
        self.loss_batch_size = loss_batch_size
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = LLMUtils(
                model_dir=self.model_dir, 
                model_max_length=self.max_seq_len, 
                gpu_memory_utilization=self.gpu_memory_utilization
            )
        return self._llm

    def _load_prompts(self):
        if self.prompts:
            return self.prompts

        allowed_promptfile_types = ["txt", "fa", "fasta", "csv", "tsv"]
        assert self.promptfile.split(".")[-1] in allowed_promptfile_types, f"Prompt file must be one of {allowed_promptfile_types}"
        
        if self.promptfile.endswith('.fa') or self.promptfile.endswith('.fasta'):
            return [str(r.seq) for r in SeqIO.parse(self.promptfile, 'fasta')]
        else:
            return list(pd.read_csv(self.promptfile, header=None, delimiter="\t" if self.promptfile.endswith('.tsv') else None)[0])


    def _print_generate_parameters(self):
        print("Parameters used for sequence generation:")
        print(f"Number of generations from each prompt: {self.num}")
        print(f"Temperature: {self.temperature}")
        print(f"Minimum length of generated tokens: {self.min_seq_len}")
        print(f"Maximum length of generated tokens: {self.max_seq_len}")
        print(f"Top-k sampling: {self.top_k}")
        print(f"Top-p sampling: {self.top_p}")
        print(f"Presence penalty: {self.presence_penalty}")
        print(f"Frequency penalty: {self.frequency_penalty}")
        print(f"Repetition penalty: {self.repetition_penalty}")
        print(f"Seed: {self.seed}")
        print(f"Filter compression: {self.filter_compression} (threshold: {self.compression_threshold})")
        print(f"Filter loss: {self.filter_loss} (threshold: {self.loss_threshold})")

    def _filter_by_compression(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        
        print(f"Applying compression filter (threshold={self.compression_threshold})...")
        original_len = len(df)
        
        # Parallel execution of compression
        with concurrent.futures.ThreadPoolExecutor() as executor:
            ratios = list(executor.map(_compute_compression_ratio, df['seq']))
            
        df['compression_ratio'] = ratios
        filtered_df = df[df['compression_ratio'] >= self.compression_threshold].copy()
        
        print(f"Kept {len(filtered_df)} out of {original_len} sequences with compression ratio >= {self.compression_threshold:.3f}")
        return filtered_df
        
    def _filter_by_loss(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
            
        print(f"Applying mean loss filter (threshold={self.loss_threshold})...")
        original_len = len(df)
        
        # Compute losses via GPU
        losses = self.llm.compute_mean_loss_batch(df['seq'].tolist(), batch_size=self.loss_batch_size)
        
        df['mean_loss'] = losses
        filtered_df = df[df['mean_loss'] >= self.loss_threshold].copy()
        
        print(f"Kept {len(filtered_df)} out of {original_len} sequences with mean loss >= {self.loss_threshold:.3f}")
        return filtered_df

    
    def generate_sequences(self, prepend_prompt_to_output=False, max_repeats=0):
        prompts = self._load_prompts()
        llm = self.llm
        
        print(f"======First Prompt {prompts[0]}")
        self._print_generate_parameters()

        final_sequences = []
        prompts_to_process = {i: prompt for i, prompt in enumerate(prompts)}
        
        # If any filters are active, over-generate to guarantee target number (1.5x, then 1.5x fallback)
        multipliers = [1.5, 1.5] if (self.filter_compression or max_repeats > 0 or self.filter_loss) else [1.0]
        
        for attempt, multiplier in enumerate(multipliers):
            if not prompts_to_process:
                break
                
            current_prompts = list(prompts_to_process.values())
            current_ids = list(prompts_to_process.keys())
            
            # Calculate target sequences per prompt for this generation pass
            target_num = int(self.num * multiplier)
            if attempt > 0:
                print(f"\nAttempt {attempt + 1}: Escalating generation to {target_num} sequences per prompt for {len(current_prompts)} prompts...")
            elif multiplier > 1.0:
                print(f"Filters active: Over-generating to {target_num} sequences per prompt (Attempt 1)...")
            
            generated = llm.generate(
                prompts=current_prompts, 
                num_generation_from_each_prompt=target_num,
                temperature=self.temperature,
                min_length=self.min_seq_len,
                max_length=self.max_seq_len,
                top_k=self.top_k,
                top_p=self.top_p,
                presence_penalty=self.presence_penalty,
                frequency_penalty=self.frequency_penalty,
                repetition_penalty=self.repetition_penalty,
                seed=self.seed + attempt  # slightly alter seed across passes
            )
            
            # Re-map generated results back to their correct prompt origin IDs
            records = []
            for i, seq in enumerate(generated):
                prompt_idx = i // target_num
                original_id = current_ids[prompt_idx]
                records.append({'seq': seq, 'id': original_id})
                
            batch_df = pd.DataFrame(records)
            
            # Apply enabled filters BEFORE prepending the prompt, so we evaluate
            # only the generated portion (not the fixed prompt sequence).
            if self.filter_compression:
                batch_df = self._filter_by_compression(batch_df)
                
            if max_repeats > 0 and not batch_df.empty:
                batch_df = batch_df.drop_duplicates(subset='seq')
                original_len = len(batch_df)
                batch_df['TRF'] = batch_df['seq'].apply(find_tandem_repeats_percentage)
                batch_df = batch_df[batch_df['TRF'] <= max_repeats]
                print(f"Kept {batch_df.shape[0]} out of {original_len} sequences with <= {max_repeats}% simple repeats")
                
            if self.filter_loss:
                batch_df = self._filter_by_loss(batch_df)

            if prepend_prompt_to_output:
                batch_df['seq'] = batch_df.apply(lambda x: prompts[x.id] + x.seq, axis=1)
                
            final_sequences.append(batch_df)
            
            # Evaluate fulfillment
            combined_so_far = pd.concat(final_sequences, ignore_index=True) if final_sequences else pd.DataFrame(columns=['seq', 'id'])
            counts = combined_so_far['id'].value_counts()
            
            still_needed = {}
            for pid in current_ids:
                if counts.get(pid, 0) < self.num:
                    still_needed[pid] = prompts_to_process[pid]
                    
            prompts_to_process = still_needed

        # Final assembly, truncation, and warnings
        all_generated = pd.concat(final_sequences, ignore_index=True) if final_sequences else pd.DataFrame(columns=['seq', 'id'])
        
        results = []
        for i, prompt in enumerate(prompts):
            prompt_seqs = all_generated[all_generated['id'] == i]
            
            if len(prompt_seqs) < self.num:
                print(f"WARNING: Prompt {i} only produced {len(prompt_seqs)} valid sequences (requested {self.num}) after all fallback attempts.")
            
            # Standardize length backwards to perfectly hit N sequences per prompt
            results.append(prompt_seqs.head(self.num))
            
        final_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=['seq', 'id'])
        return final_df

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

