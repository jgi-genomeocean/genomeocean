"""
scan a genome with a language model to compute per-token scores
Example usage: 
```
python scan_genome.py \
    --model_dir pGenomeOcean/GenomeOcean-4B \
    --model_max_length 10000 \
    --genomefile ../sample_data/sample_genomes.fna.gz \
    --out_prefix outputs/scan/ \
    --out_postfix scores_meta.csv
```
"""
from genomeocean.genomescan import GenomeWideScanUtility
from genomeocean.llm_utils import LLMUtils
import pandas as pd

import sys, os
import argparse

def genomescan(model_dir, genomefile, out_prefix, out_postfix, overlap=500, model_max_length=10000):
    # allow 500bp overlap between adjacent segments, and limit the maximum sequence length to 10kb
    wgs = GenomeWideScanUtility(genome_file=genomefile, overlap=overlap, model_max_length=model_max_length)
    wgs.segment_genome(out_prefix=out_prefix)
    print(f"Genome segmented into segments with max length {wgs.model_max_length} and overlap {wgs.overlap} in directory {out_prefix}")

    # calculate per chromosome scores
    llm = LLMUtils(model_dir=model_dir)
    for chrom in wgs.chromosomes:
        input1 = f"{out_prefix}_{chrom}"
        output1 = f"{out_prefix}_{chrom}_{out_postfix}"
        if os.path.exists(output1):
            print(f"Scores for chromosome {chrom} already computed, skipping")
            continue
        # skip empty input files
        if os.path.getsize(input1) == 0:
            print(f"Empty file for chromosome {chrom}, skipping")
            continue
        input1 = pd.read_csv(input1, sep=',', header=None)
        scores = llm.compute_token_perplexity(list(input1[0]))
        pd.DataFrame(scores).to_csv(output1, index=False, header=False)
        print(f"Scores for chromosome {chrom} written to {output1}")

def main(model_dir, genomefile, out_prefix, out_postfix, overlap=500, model_max_length=10000):
    if model_max_length > 10000:
        print("Warning: model_max_length is set to bigger than 10kb, which may exceed the maximum allowed by the model")
    genomescan(model_dir, genomefile, out_prefix, out_postfix, overlap=overlap, model_max_length=model_max_length)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="Directory containing the model")
    parser.add_argument("--genomefile", help="Genome file")
    parser.add_argument("--out_prefix", default='./SCAN', help="Output prefix")
    parser.add_argument("--out_postfix", default='scores.csv', help="Output postfix")
    parser.add_argument("--overlap", type=int, default=500, help="Overlap between segments")
    parser.add_argument("--model_max_length", type=int, default=10000, help="Max sequence length for each segment")
    args = parser.parse_args()
     # print out the arguments to standard output
    print(f'Parameters: {args}')   
    main(args.model_dir, args.genomefile, args.out_prefix, args.out_postfix, overlap=args.overlap, model_max_length=args.model_max_length)


