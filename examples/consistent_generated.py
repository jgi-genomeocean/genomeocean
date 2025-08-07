
import argparse
import logging
import os
import random
import subprocess
import sys
import tempfile
from collections import Counter

import numpy as np
import pandas as pd
from Bio import SeqIO

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_dependencies():
    """Checks if required external dependencies (prodigal, cd-hit) are installed."""
    for cmd in ['prodigal', 'cd-hit']:
        if subprocess.run(['which', cmd], capture_output=True, text=True).returncode != 0:
            logging.error(f"Dependency not found: {cmd}. Please install it and ensure it's in your PATH.")
            sys.exit(1)

def read_sequences(input_file):
    """Reads sequences from a CSV or FASTA file."""
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file, header=0)
        sequences = []
        for i, row in df.iterrows():
            sequences.append((f"seq_{i}", row['sequence']))
        return sequences
    elif input_file.endswith('.fa') or input_file.endswith('.fasta'):
        return [(record.id, str(record.seq)) for record in SeqIO.parse(input_file, "fasta")]
    else:
        raise ValueError("Unsupported input file format. Please use .csv or .fasta")

def run_prodigal(sequences, temp_dir):
    """Runs prodigal to predict genes and proteins."""
    fasta_path = os.path.join(temp_dir, "temp.fasta")
    genes_path = os.path.join(temp_dir, "genes.fa")
    proteins_path = os.path.join(temp_dir, "proteins.faa")

    with open(fasta_path, "w") as f:
        for seq_id, seq_data in sequences:
            f.write(f">{seq_id}\n{seq_data}\n")

    logging.info("Running prodigal...")
    subprocess.run(
        ["prodigal", "-i", fasta_path, "-d", genes_path, "-a", proteins_path, "-p", "meta"],
        check=True, capture_output=True, text=True
    )

    genes = {record.id: str(record.seq) for record in SeqIO.parse(genes_path, "fasta")}
    proteins = {record.id: str(record.seq) for record in SeqIO.parse(proteins_path, "fasta")}
    
    return genes, proteins


def filter_orfs(genes, proteins, entropy_threshold):
    """Filters ORFs based on length, start codon, and entropy."""
    
    # Initial filtering and creation of DataFrame
    data = []
    for prot_id, prot_seq in proteins.items():
        if len(prot_seq) >= 100 and prot_id in genes:
            data.append({'id': prot_id, 'gene': genes[prot_id], 'ORF': prot_seq})
    
    if not data:
        return pd.DataFrame()

    df_orfs = pd.DataFrame(data)

    # Filter by start codon
    df_orfs = df_orfs[df_orfs['ORF'].str.startswith('M')]
    
    # Filter by entropy
    df_orfs['entropy'] = df_orfs['ORF'].apply(calculate_entropy)
    df_orfs = df_orfs[df_orfs['entropy'] >= entropy_threshold]

    logging.info(f"Filtered ORFs: Retained {len(df_orfs)} sequences.")
    return df_orfs


def calculate_entropy(sequence):
    """Calculates the Shannon entropy of a sequence."""
    counts = Counter(sequence)
    length = len(sequence)
    if length == 0:
        return 0
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * np.log2(p)
    return entropy


def run_cd_hit(df_orfs, temp_dir, c, n):
    """Runs cd-hit to cluster similar proteins."""
    orfs_fasta_path = os.path.join(temp_dir, "orfs.fasta")
    cdhit_output_path = os.path.join(temp_dir, "orfs_cdhit")

    with open(orfs_fasta_path, "w") as f:
        for _, row in df_orfs.iterrows():
            f.write(f">{row['id']}\n{row['ORF']}\n")

    logging.info("Running cd-hit...")
    subprocess.run(
        ["cd-hit", "-i", orfs_fasta_path, "-o", cdhit_output_path, "-c", str(c), "-n", str(n)],
        check=True, capture_output=True, text=True
    )

    clusters = {}
    with open(f"{cdhit_output_path}.clstr", "r") as f:
        cluster_id = None
        for line in f:
            if line.startswith(">"):
                cluster_id = line.strip().split(" ")[1]
                clusters[cluster_id] = []
            else:
                member_id = line.split(">")[1].split("...")[0]
                clusters[cluster_id].append(member_id)

    return {k: v for k, v in clusters.items() if len(v) >= 2}


def select_genes(df_orfs, clusters, min_len, max_len):
    """Randomly selects a gene from each large cluster within a specified length range."""
    selected_genes_info = []
    for _, members in clusters.items():
        valid_members = [
            member_id for member_id in members
            if min_len <= len(df_orfs.loc[df_orfs['id'] == member_id, 'ORF'].iloc[0]) <= max_len
        ]

        if valid_members:
            selected_member_id = random.choice(valid_members)
            selected_genes_info.append(df_orfs[df_orfs['id'] == selected_member_id])

    if selected_genes_info:
        return pd.concat(selected_genes_info)
    return pd.DataFrame()


def main():
    """Main function to run the gene selection pipeline."""
    parser = argparse.ArgumentParser(description="Select consistently generated genes from a set of sequences.")
    parser.add_argument("-i", "--input", required=True, help="Input file with generated sequences (.csv or .fasta)")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file for selected genes.")
    parser.add_argument("--cdhit_c", type=float, default=0.7, help="CD-HIT sequence identity threshold.")
    parser.add_argument("--cdhit_n", type=int, default=5, help="CD-HIT word size.")
    parser.add_argument("--min_len", type=int, default=100, help="Minimum ORF length for final selection.")
    parser.add_argument("--max_len", type=int, default=400, help="Maximum ORF length for final selection.")
    parser.add_argument("--entropy_threshold", type=float, default=2.5, help="Entropy threshold for filtering low-complexity sequences.")
    args = parser.parse_args()

    check_dependencies()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Step 1: Read Sequences
        sequences = read_sequences(args.input)
        
        # Step 2: Run Prodigal
        genes, proteins = run_prodigal(sequences, temp_dir)
        
        # Step 3: Filter ORFs
        df_orfs = filter_orfs(genes, proteins, args.entropy_threshold)
        if df_orfs.empty:
            logging.warning("No ORFs passed the initial filtering. Exiting.")
            return

        # Step 4: Run CD-HIT
        large_clusters = run_cd_hit(df_orfs, temp_dir, args.cdhit_c, args.cdhit_n)
        if not large_clusters:
            logging.warning("No clusters with at least two members were found. Exiting.")
            return
            
        logging.info(f"Found {len(large_clusters)} clusters with at least two members.")

        # Step 5: Select Genes
        df_selected = select_genes(df_orfs, large_clusters, args.min_len, args.max_len)

        # Step 6: Save Results
        if not df_selected.empty:
            df_selected.to_csv(args.output, index=False)
            logging.info(f"Selected {len(df_selected)} genes written to {args.output}")
        else:
            logging.warning("No genes found in the specified length range for final selection.")


if __name__ == "__main__":
    main()
