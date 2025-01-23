"""
Generate sequences using a prompt file (csv or txt format)
Example usage: 
```
# this may be needed to avoid issues with multiprocessing on some systems
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python generate_sequences.py \
    --model_dir pGenomeOcean/GenomeOcean-4B \
    --promptfile ../sample_data/dna_sequences.txt \
    --out_prefix outputs/generated \
    --out_format fa \
    --num 10 \
    --min_seq_len 100 \
    --max_seq_len 100 \
    --temperature 1.3 \
    --top_k=-1 \
    --top_p=0.7 \
    --max_repeats 100 \
    --presence_penalty=0.5 \
    --frequency_penalty=0.5 \
    --repetition_penalty=1.0 \
    --seed 123 \
    --sort_by_orf_length
```
"""
from genomeocean.generation import SequenceGenerator
from genomeocean.dnautils import find_tandem_repeats_percentage
import pandas as pd
import os
import sys
import argparse
import pyrodigal
from Bio import SeqIO
import textwrap

def generate_sequences(
            model_dir, 
            promptfile, 
            out_prefix, 
            num=100, 
            min_seq_len=1024, 
            max_seq_len=10240, 
            max_repeats=100,
            temperature=1.3,
            top_k=-1,
            top_p=0.7,
            presence_penalty=0.5,
            frequency_penalty=0.5,
            repetition_penalty=1.0,
            seed=123,
            sort_by_orf_length=False,
            out_format='fa',
            ):
    # generate sequences using a prompt file
    if max_seq_len >10240:
        print("Warning: max_seq_len is set to bigger than 10kb, which may exceed the maximum allowed by the model")
        print("Use the continuous generation mode")
        all_generated = generate_sequences_long(
                                model_dir=model_dir,
                                promptfile=promptfile,
                                num=num,
                                max_seq_len=max_seq_len,
                                max_repeats=max_repeats,
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                                presence_penalty=presence_penalty,
                                frequency_penalty=frequency_penalty,
                                repetition_penalty=repetition_penalty,
                                seed=seed,
                                )
    else:
        seq_gen = SequenceGenerator(
            model_dir=model_dir, 
            promptfile=promptfile, 
            num=num, 
            min_seq_len=min_seq_len, 
            max_seq_len=max_seq_len,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )
        all_generated = seq_gen.generate_sequences(prepend_prompt_to_output=True, max_repeats=max_repeats)
    seq_gen.save_sequences(all_generated, out_prefix=out_prefix, out_format=out_format)

    if sort_by_orf_length:
        # translate the sequences using prodigal
        orf_finder = pyrodigal.GeneFinder(meta=True)
        all_orfs = {}
        for r in SeqIO.parse(f'{out_prefix}.fa', 'fasta'):
            sid = r.id 
            for i, pred in enumerate(orf_finder.find_genes(bytes(str(r.seq), encoding='utf8'))):
                seq = pred.translate().upper()[:-1] # need to remove the last stop codon
                if len(seq) >= 100:
                    all_orfs[str(sid) + '_' + str(i)] = seq

        # make a fasta file longest ORFs first
        with open(f'{out_prefix}.faa', "w") as f:
            # sort by length in reverse order
            all_orfs = {k: v for k, v in sorted(all_orfs.items(), key=lambda item: len(item[1]), reverse=True)}
            # write to file
            for i, (sid, seq) in enumerate(all_orfs.items()):
                f.write(f">{sid}\n")
                f.write('\n'.join(textwrap.wrap(seq, 80)) + '\n')   

def generate_sequences_long(
            model_dir, 
            promptfile, 
            num=2, 
            max_seq_len=10240, 
            max_repeats=100,
            temperature=1.3,
            top_k=-1,
            top_p=0.7,
            presence_penalty=0.5,
            frequency_penalty=0.5,
            repetition_penalty=1.0,
            seed=123,
            ):
    # generate the first round
    seq_gen = SequenceGenerator(
        model_dir=model_dir, 
        promptfile=promptfile, 
        num=num, 
        min_seq_len=10230, 
        max_seq_len=10240,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repetition_penalty=repetition_penalty,
        seed=seed,
    )
    g_seqs = seq_gen.generate_sequences(add_prompt=True)
    # destroy the SequenceGenerator object to release memory
    del seq_gen
    for i in range(int(max_seq_len/1024)):
        # generate the next round, use the last 10000 bases as prompt
        prompts = g_seqs['seq'].apply(lambda x:  x[-10000:]).to_csv('prompts_c.csv', index=False, header=None)
        seq_gen = SequenceGenerator(
            model_dir=model_dir, 
            promptfile='prompts_c.csv', 
            num=1, 
            min_seq_len=10230, 
            max_seq_len=10240,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )
        c_seqs = seq_gen.generate_sequences(prepend_prompt_to_output=False)
        del seq_gen
        # append the new sequences to the previous ones
        g_seqs['seq'] = g_seqs['seq'] + c_seqs['seq']
    # remove sequences wth repeats
    if (max_repeats > 0) and (max_repeats < 100):
        g_seqs['repeats'] = g_seqs['seq'].apply(find_tandem_repeats_percentage)
        g_seqs = g_seqs[g_seqs['repeats'] < max_repeats]
    os.remove('prompts_c.csv')
    return g_seqs

  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="Directory containing the model")
    parser.add_argument("--promptfile", help="Prompt file in csv or fasta format")
    parser.add_argument("--out_prefix", default='./SCAN', help="Output prefix")
    parser.add_argument("--out_format", default='fa', help="Output format")
    parser.add_argument("--num", default=100, type=int, help="Number of sequences to generate from each prompt")
    parser.add_argument("--min_seq_len", type=int, default=1024, help="minimum length of generated sequences in token, set it as expected bp length // 4 (e.g., set it as 1000 for 4kb)")
    parser.add_argument("--max_seq_len", type=int, default=10240, help="maximum length of generated sequences in token, max value is 10240")
    parser.add_argument("--temperature", type=float, default=1.3, help="temperature for sampling")
    parser.add_argument("--top_k", type=int, default=-1, help="top_k for sampling")
    parser.add_argument("--top_p", type=float, default=0.7, help="top_p for sampling")
    parser.add_argument("--presence_penalty", type=float, default=0.5, help="presence penalty for sampling")
    parser.add_argument("--frequency_penalty", type=float, default=0.5, help="frequency penalty for sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="repetition penalty for sampling")
    parser.add_argument("--seed", type=int, default=123, help="random seed for sampling")
    parser.add_argument("--max_repeats", type=int, default=0, help="Maximum percentage of repeats")
    parser.add_argument("--sort_by_orf_length", action='store_true', help="Sort the sequences by ORF length")
    args = parser.parse_args()
    
    # print out the arguments to standard output
    print(f'Parameters: {args}')
    generate_sequences(
        model_dir=args.model_dir,
        promptfile=args.promptfile,
        out_prefix=args.out_prefix,
        num=args.num,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        max_repeats=args.max_repeats,
        sort_by_orf_length=args.sort_by_orf_length,
        out_format=args.out_format,
    )