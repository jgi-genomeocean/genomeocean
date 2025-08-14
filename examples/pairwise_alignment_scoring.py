
"""
This script calculates pairwise alignment scores for protein sequences.
"""
import pandas as pd
import argparse
from Bio.Seq import Seq, translate
from Bio import Align
from genomeocean.dnautils import get_nuc_seq_by_id, reverse_complement

def get_reference_protein(gene_id, start, end, strand, sequence=None, structure_start=0, structure_end=None):
    """
    Retrieves and translates a reference gene sequence.
    """
    if sequence:
        gene = sequence
    elif gene_id:
        gene = get_nuc_seq_by_id(gene_id, start=start, end=end)
        if gene is None:
            raise ValueError(f"Failed to retrieve gene sequence {gene_id} from {start} to {end}")
    else:
        raise ValueError("Either --gene_id or --sequence must be provided.")

    if strand == -1:
        gene = reverse_complement(gene)

    protein_sequence = translate(Seq(gene), to_stop=True)
    
    if structure_end is None:
        structure_end = len(protein_sequence)
        
    return protein_sequence[structure_start:structure_end]

def main():
    parser = argparse.ArgumentParser(description="Calculate pairwise alignment scores for protein sequences.")
    parser.add_argument("--generated_seqs_csv", required=True, help="CSV file with a 'protein' column.")
    parser.add_argument("--output_prefix", required=True, help="Prefix for the output scores CSV file.")
    parser.add_argument("--gene_id", help="Gene ID for the reference sequence.")
    parser.add_argument("--sequence", help="Raw DNA sequence for the reference.")
    parser.add_argument("--start", type=int, help="Start position for the gene sequence.")
    parser.add_argument("--end", type=int, help="End position for the gene sequence.")
    parser.add_argument("--strand", type=int, choices=[-1, 1], help="Strand of the gene.")
    parser.add_argument("--structure_start", type=int, default=0, help="Start position of the protein structure to be scored.")
    parser.add_argument("--structure_end", type=int, help="End position of the protein structure. If not provided, scores the full length.")

    args = parser.parse_args()

    try:
        ref_protein = get_reference_protein(
            args.gene_id, args.start, args.end, args.strand,
            args.sequence, args.structure_start, args.structure_end
        )
        
        g_seqs_df = pd.read_csv(args.generated_seqs_csv, sep=None, engine='python')
        queries = g_seqs_df['protein'].tolist()

        if args.structure_end is not None:
            trimmed_queries = [q[args.structure_start:args.structure_end] for q in queries]
        else:
            trimmed_queries = [q[args.structure_start:] for q in queries]

        aligner = Align.PairwiseAligner()
        aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")

        max_score = aligner.score(ref_protein, ref_protein)
        if max_score == 0:
            print("Warning: Max score for reference protein is 0. Cannot normalize scores.")
            scores = ['N/A'] * len(trimmed_queries)
        else:
            scores = [aligner.score(ref_protein, q) / max_score for q in trimmed_queries]

        g_seqs_df['score'] = scores
        output_filename = f"{args.output_prefix}.csv"
        g_seqs_df.to_csv(output_filename, index=False)
        
        print(f"Pairwise alignment scoring complete. Results saved to {output_filename}")

    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == '__main__':
    main()
