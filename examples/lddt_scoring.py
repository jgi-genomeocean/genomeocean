import pandas as pd
import numpy as np
import os
import sys
import argparse
import requests
import subprocess
from genomeocean.dnautils import fasta2pdb_api, LDDT_scoring

def score_generated_sequences(generated_seqs, ref_pdb, structure_start, structure_end, foldmason_path=''):
    g_seqs = pd.read_csv(generated_seqs)
    g_seqs['lddt_score'] = g_seqs['protein'].apply(lambda x: LDDT_scoring(x[structure_start:structure_end], ref_pdb, foldmason_path=foldmason_path))
    return g_seqs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_seqs", help="Generated sequences")
    parser.add_argument("--ref_pdb", default='', help="reference pdb file")
    parser.add_argument("--structure_start", type=int, default=0, help="start position of the structure")
    parser.add_argument("--structure_end", type=int, default=0, help="end position of the structure")
    parser.add_argument("--foldmason_path", default='', help="foldmason path")
    parser.add_argument("--output_prefix", default='generated', help="output prefix")
    args = parser.parse_args()
    g_seqs = score_generated_sequences(args.generated_seqs, args.ref_pdb, args.structure_start, args.structure_end, args.foldmason_path)
    g_seqs.to_csv(args.output_prefix + '.csv', sep='\t', index=False)
if __name__ == '__main__':
    main()
