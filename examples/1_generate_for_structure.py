
"""
This script, the first in a two-part pipeline, generates DNA sequences from a given prompt. 
The prompt can either be provided as a raw DNA sequence or fetched by a gene ID. The script 
can also generate mutated versions of the prompt to explore sequence variations.

The output of this script is a CSV file containing the generated sequences, which will be 
used as input for the second part of the pipeline (lddt_scoring.py).

Example usage:

# Generate sequences for GMP synthetase
python 1_generate_for_structure.py \
    --gen_id NZ_JAYXHC010000003.1 \
    --start 157 \
    --end 1698 \
    --strand -1 \
    --prompt_start 0 \
    --prompt_end 600 \
    --model_dir pGenomeOcean/GenomeOcean-4B \
    --num 100 \
    --min_seq_len 250 \
    --max_seq_len 300 \
    --output_prefix outputs/gmp

# Generate sequences for a TRAP-like protein and explore mutations in the prompt
python 1_generate_for_structure.py \
    --gen_id OY729418.1 \
    --start 1675256 \
    --end 1676176 \
    --strand -1 \
    --prompt_start 0 \
    --prompt_end 450 \
    --mutate_prompt 1 \
    --model_dir pGenomeOcean/GenomeOcean-4B \
    --num 200 \
    --min_seq_len 250 \
    --max_seq_len 300 \
    --output_prefix outputs/trapl_wt_mutations
"""
import pandas as pd
import os
import sys
import argparse

from genomeocean.dnautils import get_nuc_seq_by_id, introduce_mutations, reverse_complement
from genomeocean.generation import SequenceGenerator
from Bio.Seq import Seq, translate

def get_largest_orf(seq):
    """
    Finds the longest Open Reading Frame (ORF) in a given DNA sequence.
    """
    seq = Seq(seq)
    orfs = []
    for frame in range(3):
        for i in range(frame, len(seq), 3):
            if seq[i:i+3] in ['ATG', 'GTG']:
                orfs.append(str(seq[i:].translate(to_stop=True)))
        orfs.append(str(seq[frame:].translate(to_stop=True)))
    return max(orfs, key=len) if orfs else ''

def generate_sequences(
    gen_id=None, 
    start=0, 
    end=0, 
    sequence=None,
    prompt_start=0, 
    prompt_end=0,
    mutate_prompt=False,
    strand=1,
    backward=False,
    model_dir='',
    **kwargs,
):
    """
    Generates sequences from a given prompt and filters them based on ORF length.
    """
    if sequence:
        gene = sequence
    elif gen_id:
        gene = get_nuc_seq_by_id(gen_id, start=start, end=end)
        if gene is None:
            print(f'Failed to retrieve gene sequence {gen_id} from {start} to {end}')
            sys.exit(1)
    else:
        print("Either --gen_id or --sequence must be provided.")
        sys.exit(1)

    if strand == -1:
        gene = reverse_complement(gene)
    
    if backward:
        gene = reverse_complement(gene)  

    prompts = [gene[prompt_start:prompt_end]] 
    if mutate_prompt:
        orf_prompt = gene[prompt_start:prompt_end]
        for mutation_rate in range(10, 50, 10):
            mutated = introduce_mutations(orf_prompt, mutation_percentage=mutation_rate, mutation_type='synonymous')
            prompts.append(mutated)
        for mutation_rate in range(10, 50, 10):
            mutated = introduce_mutations(orf_prompt, mutation_percentage=mutation_rate, mutation_type='nonsynonymous')
            prompts.append(mutated)  

    with open('tmp_prompts.csv', 'w') as f:
        for p in prompts:
            f.write(p + '\n')

    seq_gen = SequenceGenerator(
        model_dir=model_dir, 
        promptfile='tmp_prompts.csv', 
        num=kwargs.get('num', 200),
        min_seq_len=kwargs.get('min_seq_len', 250),
        max_seq_len=kwargs.get('max_seq_len', 300),
        temperature=1.0,
        presence_penalty=0.5, 
        frequency_penalty=0.5, 
        repetition_penalty=1.0, 
        seed=1234
    )
    g_seqs = seq_gen.generate_sequences(prepend_prompt_to_output=True, max_repeats=100)
    print(f'Total {g_seqs.shape[0]} sequences were generated.')  
    os.remove('tmp_prompts.csv')

    if backward:
        g_seqs['seq'] = g_seqs['seq'].apply(reverse_complement)
    
    g_seqs['protein'] = g_seqs['seq'].apply(get_largest_orf)  
    g_seqs['orf_len'] = g_seqs['protein'].apply(lambda x: 3*len(x))
    g_seqs['length'] = g_seqs['seq'].apply(len)
    g_seqs = g_seqs[g_seqs['orf_len'] >= len(gene) - 100].copy()
    print(f'Total {g_seqs.shape[0]} sequences have longer ORFs than the original minus 100bp.')
    
    return g_seqs

def main():
    parser = argparse.ArgumentParser(description="Generate DNA sequences from a prompt.")
    # Arguments for specifying the prompt
    parser.add_argument("--gen_id", help="Gene ID to fetch the sequence from.")
    parser.add_argument("--sequence", help="Raw DNA sequence to use as a prompt.")
    parser.add_argument("--start", type=int, default=0, help="Start position for fetching the gene sequence.")
    parser.add_argument("--end", type=int, default=0, help="End position for fetching the gene sequence.")
    parser.add_argument("--strand", type=int, default=1, help="Strand of the gene (1 for forward, -1 for reverse).")
    
    # Arguments for defining the prompt and generation direction
    parser.add_argument("--prompt_start", type=int, default=0, help="Start position of the prompt within the gene sequence.")
    parser.add_argument("--prompt_end", type=int, default=0, help="End position of the prompt within the gene sequence.")
    parser.add_argument("--direction", type=int, default=1, help="Set to -1 to generate sequences in the reverse direction.")

    # Arguments for controlling sequence generation
    parser.add_argument("--mutate_prompt", type=int, default=0, help="If set to 1, introduces synonymous and non-synonymous mutations to the prompt.")
    parser.add_argument("--model_dir", required=True, help="Directory of the language model.")
    parser.add_argument("--num", type=int, default=200, help="Number of sequences to generate.")
    parser.add_argument("--min_seq_len", type=int, default=250, help="Minimum sequence length.")
    parser.add_argument("--max_seq_len", type=int, default=300, help="Maximum sequence length.")
    
    # Argument for the output file
    parser.add_argument("--output_prefix", default='generated', help="Prefix for the output CSV file.")
    
    args = parser.parse_args()

    print(f"Parameters: {args}")

    generated_sequences = generate_sequences(
        gen_id=args.gen_id,
        sequence=args.sequence,
        start=args.start, 
        end=args.end, 
        prompt_start=args.prompt_start, 
        prompt_end=args.prompt_end,
        mutate_prompt=bool(args.mutate_prompt),
        strand=args.strand,
        backward=(args.direction == -1),
        model_dir=args.model_dir,
        num=args.num,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
    )

    output_filename = args.output_prefix + '.csv'
    generated_sequences.to_csv(output_filename, sep='\t', index=False)
    print(f"Generated sequences saved to {output_filename}")

if __name__ == '__main__':
    main()
