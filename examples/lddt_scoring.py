import pandas as pd
import numpy as np
import os
import sys
import argparse
import requests
import subprocess
from genomeocean.dnautils import LDDT_scoring_parallel

def score_generated_sequences_parallel(queries, ref_pdb, foldmason_path='', max_workers=None):
    """
    Score a list of protein sequences against a reference PDB structure in parallel.

    Args:
        queries (list): A list of protein sequences.
        ref_pdb (str): Path to the reference PDB file.
        foldmason_path (str): Path to the foldmason executable.
        max_workers (int): Maximum number of threads for parallel execution.

    Returns:
        A dictionary with sequences as keys and their lDDT scores as values.
    """
    scores = LDDT_scoring_parallel(
        queries,
        ref_pdb,
        foldmason_path=foldmason_path,
        method='foldmason',
        max_workers=max_workers
    )
    return scores

def main():
    parser = argparse.ArgumentParser(description="Score protein sequences against a reference PDB structure using FoldMason.")
    parser.add_argument("--generated_seqs_csv", help="CSV file with a 'protein' column containing sequences to score.")
    parser.add_argument("--ref_pdb", required=True, help="Reference PDB file.")
    parser.add_argument("--structure_start", type=int, default=0, help="Start position of the structure to be scored.")
    parser.add_argument("--structure_end", type=int, help="End position of the structure. If not provided, scores the full length.")
    parser.add_argument("--foldmason_path", required=True, help="Path to the FoldMason executable.")
    parser.add_argument("--output_prefix", default='generated', help="Output prefix for the results file.")
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum number of parallel workers.")
    args = parser.parse_args()

    # Define a default list of sequences for demonstration
    sample_queries = [
        "MKLFWLLFTIGLCVSAQYGDVLENAEQGDFATDDYKDDDDKSPAGS",
        "MKLFWLLFTIGLCVSAQYGDVLENAEQGDFATDDYKDDDDKSPAGS",
        "MKLFWLLFTIGLCVSAQYGDVLENAEQGDFATDDYKDDDDKSPAGS",
    ]

    if args.generated_seqs_csv:
        g_seqs_df = pd.read_csv(args.generated_seqs_csv)
        queries = g_seqs_df['protein'].tolist()
    else:
        print("No CSV file provided. Using sample queries for demonstration.")
        queries = sample_queries
    
    # Trim sequences if start/end positions are specified
    if args.structure_end is not None:
        trimmed_queries = [q[args.structure_start:args.structure_end] for q in queries]
    else:
        trimmed_queries = [q[args.structure_start:] for q in queries]

    # Score sequences in parallel
    scores = score_generated_sequences_parallel(trimmed_queries, args.ref_pdb, args.foldmason_path, args.max_workers)

    # Create a DataFrame with the results and save to a CSV file
    results_df = pd.DataFrame({'protein': queries, 'lddt_score': [scores.get(q, 'N/A') for q in trimmed_queries]})
    output_filename = f"{args.output_prefix}_scores.csv"
    results_df.to_csv(output_filename, sep='\t', index=False)
    print(f"Scoring complete. Results saved to {output_filename}")

if __name__ == '__main__':
    main()
