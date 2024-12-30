"""
Generate sequences using a prompt file (csv format)
Example usage: 
```
# this may be needed to avoid issues with multiprocessing on some systems
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export model=/path/to/meta_hmp_seq1024
export promptfile=../test_data/CoA_prompts.csv
export outprefix=../test_data/CoA

python generate_sequences.py --model_dir $model --promptfile $promptfile --out_prefix $outprefix --num 100 --min_seq_len 1024 --max_seq_len 10240

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

def generate_sequences(model_dir, promptfile, out_prefix, num=100, min_seq_len=1024, max_seq_len=10240, max_repeats=100):
    # generate sequences using a prompt file
    if max_seq_len >10240:
        print("Warning: max_seq_len is set to bigger than 10kb, which may exceed the maximum allowed by the model")
        print("Use the continuous generation mode")
        all_generated = generate_sequences_long(model_dir, promptfile, num=num, max_seq_len=max_seq_len, max_repeats=max_repeats)
    else:
        seq_gen = SequenceGenerator(
            model_dir=model_dir, 
            promptfile=promptfile, 
            num=num, 
            min_seq_len=min_seq_len, 
            max_seq_len=max_seq_len,)
        all_generated = seq_gen.generate_sequences(add_prompt=True, max_repeats=max_repeats)
    seq_gen.save_sequences(all_generated, out_prefix=out_prefix)

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

def generate_sequences_long(model_dir, promptfile, num=2, max_seq_len=10240, max_repeats=100):
    # generate the first round
    seq_gen = SequenceGenerator(
        model_dir=model_dir, 
        promptfile=promptfile, 
        num=num, 
        min_seq_len=10230, 
        max_seq_len=10240,)
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
            max_seq_len=10240,)
        c_seqs = seq_gen.generate_sequences(add_prompt=False)
        del seq_gen
        # append the new sequences to the previous ones
        g_seqs['seq'] = g_seqs['seq'] + c_seqs['seq']
    # remove sequences wth repeats
    if (max_repeats > 0) and (max_repeats < 100):
        g_seqs['repeats'] = g_seqs['seq'].apply(find_tandem_repeats_percentage)
        g_seqs = g_seqs[g_seqs['repeats'] < max_repeats]
    os.remove('prompts_c.csv')
    return g_seqs

  


def main(model_dir, promptfile, out_prefix, num=100, min_seq_len=1024, max_seq_len=10240, max_repeats=0):
    generate_sequences(model_dir, promptfile, out_prefix, num=num, min_seq_len=min_seq_len, max_seq_len=max_seq_len)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="Directory containing the model")
    parser.add_argument("--promptfile", help="Prompt file in csv or fasta format")
    parser.add_argument("--out_prefix", default='./SCAN', help="Output prefix")
    parser.add_argument("--num", default=100, type=int, help="Number of sequences to generate from each prompt")
    parser.add_argument("--min_seq_len", type=int, default=1024, help="Minimum sequence length")
    parser.add_argument("--max_seq_len", type=int, default=10240, help="Maximum sequence length")
    parser.add_argument("--max_repeats", type=int, default=100, help="Maximum percentage of repeats")
    args = parser.parse_args()
    model_dir = args.model_dir
    promptfile = args.promptfile
    out_prefix = args.out_prefix
    num = int(args.num)
    min_seq_len = int(args.min_seq_len)
    max_seq_len = int(args.max_seq_len)
    max_repeats = int(args.max_repeats)
    # print out the arguments to standard output
    print(f'Parameters: {args}')
    main(model_dir, promptfile, out_prefix, num=num, min_seq_len=min_seq_len, max_seq_len=max_seq_len, max_repeats=max_repeats)