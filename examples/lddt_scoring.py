"""
This script, the second in a two-part pipeline, scores generated protein sequences 
against a reference structure. It takes a CSV file of generated sequences (produced by 
1_generate_for_structure.py) and a reference gene. 

The script first creates a reference PDB structure from the gene sequence. Then, it uses 
FoldMason to calculate the lDDT score for each generated protein sequence against the 
reference PDB. The final output is a CSV file with the original data and the new lDDT scores.

Example usage:

# Score the generated sequences for GMP synthetase
python lddt_scoring.py \
    --generated_seqs_csv outputs/gmp.csv \
    --gen_id NZ_JAYXHC010000003.1 \
    --start 157 \
    --end 1698 \
    --strand -1 \
    --structure_start 150 \
    --structure_end 500 \
    --foldmason_path ~/bin/foldmason \
    --output_prefix outputs/gmp

# Score the generated sequences for the TRAP-like protein
python lddt_scoring.py \
    --generated_seqs_csv outputs/trapl_wt_mutations.csv \
    --gen_id OY729418.1 \
    --start 1675256 \
    --end 1676176 \
    --strand -1 \
    --structure_start 0 \
    --structure_end 341 \
    --foldmason_path ~/bin/foldmason \
    --output_prefix outputs/trapl_wt_mutations
"""
import pandas as pd
import os
import sys
import argparse
from Bio.Seq import Seq, translate

from genomeocean.dnautils import LDDT_scoring_parallel, get_nuc_seq_by_id, reverse_complement, fasta2pdb_api

def get_reference_protein_sequence(
    gene_id=None,
    sequence=None,
    start=0,
    end=0,
    strand=1,
    structure_start=0,
    structure_end=None
):
    """
    Creates a reference PDB file from a gene sequence.
    """
    if sequence:
        gene = sequence
    elif gene_id:
        gene = get_nuc_seq_by_id(gene_id, start=start, end=end)
        if gene is None:
            print(f'Failed to retrieve gene sequence {gene_id} from {start} to {end}')
            sys.exit(1)
    else:
        print("Either --gene_id or --sequence must be provided for the reference gene.")
        sys.exit(1)

    if strand == -1:
        gene = reverse_complement(gene)

    protein_sequence = translate(Seq(gene), to_stop=True)

    if structure_end is None:
        structure_end = len(protein_sequence)

    structure_segment = protein_sequence[structure_start:structure_end]

    if not structure_segment:
        print("Error: The specified structure segment is empty.")
        sys.exit(1)

    return structure_segment

def main():
    parser = argparse.ArgumentParser(description="Score protein sequences against a reference PDB structure.")
    
    # Arguments for input and output files
    parser.add_argument("--generated_seqs_csv", required=True, help="CSV file with a 'protein' column containing sequences to score.")
    parser.add_argument("--output_prefix", required=True, help="Prefix for the output files (including the scores CSV and the reference PDB).")

    # Arguments for defining the reference gene and structure
    parser.add_argument("--gene_id", help="Gene ID for the reference sequence.")
    parser.add_argument("--sequence", help="Raw DNA sequence for the reference.")
    parser.add_argument("--start", type=int, default=0, help="Start position for fetching the gene sequence.")
    parser.add_argument("--end", type=int, default=0, help="End position for fetching the gene sequence.")
    parser.add_argument("--strand", type=int, default=1, help="Strand of the gene (1 for forward, -1 for reverse).")
    parser.add_argument("--structure_start", type=int, default=0, help="Start position of the protein structure to be scored.")
    parser.add_argument("--structure_end", type=int, help="End position of the protein structure. If not provided, scores the full length.")

    # Arguments for the scoring tool
    parser.add_argument("--foldmason_path", required=True, help="Path to the FoldMason executable.")
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum number of parallel workers for scoring.")
    parser.add_argument("--truncate-sequences", default=True, action=argparse.BooleanOptionalAction, help="Truncate query and reference to same length for scoring.")

    args = parser.parse_args()

    # 1. Get reference protein sequence
    ref_protein_seq = get_reference_protein_sequence(
        gene_id=args.gene_id,
        sequence=args.sequence,
        start=args.start,
        end=args.end,
        strand=args.strand,
        structure_start=args.structure_start,
        structure_end=args.structure_end
    )

    # 2. Load the generated sequences
    try:
        g_seqs_df = pd.read_csv(args.generated_seqs_csv)
        if 'protein' in g_seqs_df.columns:
            queries = g_seqs_df['protein'].tolist()
        elif 'ORF' in g_seqs_df.columns:
            queries = g_seqs_df['ORF'].tolist()
        else:
            print("Error: CSV file must contain a 'protein' or 'ORF' column.")
            sys.exit(1)
    except FileNotFoundError:
        print(f"Error: The file {args.generated_seqs_csv} was not found.")
        sys.exit(1)

    # 3. Trim the query sequences to the specified structure region
    if args.structure_end is not None:
        trimmed_queries = [q[args.structure_start:args.structure_end] for q in queries]
    else:
        trimmed_queries = [q[args.structure_start:] for q in queries]


    final_ref_seq = ref_protein_seq
    final_queries = trimmed_queries

    if args.truncate_sequences:
        min_len = len(ref_protein_seq)
        for q in trimmed_queries:
            min_len = min(min_len, len(q))

        if min_len < len(ref_protein_seq) or any(len(q) > min_len for q in trimmed_queries):
             print(f"Truncating reference and query sequences to length {min_len}")
             final_ref_seq = ref_protein_seq[:min_len]
             final_queries = [q[:min_len] for q in trimmed_queries]

    # 4. Create the reference PDB file
    if args.truncate_sequences:
        ref_pdb_path = args.output_prefix + f'_ref_len{len(final_ref_seq)}.pdb'
    else:
        ref_pdb_path = args.output_prefix + '_ref.pdb'

    if os.path.exists(ref_pdb_path):
        print(f'PDB file {ref_pdb_path} already exists, using it as reference.')
    else:
        fasta2pdb_api(final_ref_seq, ref_pdb_path)
        print(f"Reference PDB file created at {ref_pdb_path}")


    # 5. Score the sequences in parallel
    scores = LDDT_scoring_parallel(
        queries=final_queries,
        target_pdb=ref_pdb_path,
        foldmason_path=args.foldmason_path,
        method='foldmason',
        max_workers=args.max_workers
    )

    # 6. Save the results
    g_seqs_df['lddt_score'] = [scores.get(q, 'N/A') for q in final_queries]
    output_filename = f"{args.output_prefix}_scores.csv"
    g_seqs_df.to_csv(output_filename, sep='\t', index=False)
    print(f"Scoring complete. Results saved to {output_filename}")

if __name__ == '__main__':
    main()
