
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

def load_sequences(filepath):
    """Loads sequences from a CSV file."""
    try:
        df = pd.read_csv(filepath, sep=',', engine='python')
        return df
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {filepath}")
        exit(1)
    except Exception as e:
        print(f"Error reading or parsing CSV file {filepath}: {e}")
        exit(1)

def trim_sequences(df, structure_start, structure_end):
    """Extracts and trims query sequences from the dataframe."""
    try:
        queries = df['ORF'].tolist()
        if structure_end is not None:
            trimmed = [q[structure_start:structure_end] for q in queries]
        else:
            trimmed = [q[structure_start:] for q in queries]
        return trimmed
    except KeyError:
        print(f"Error: 'ORF' column not found in the input CSV.")
        exit(1)
    except TypeError as e:
        print(f"Error processing sequences from 'ORF' column. Make sure they are valid sequences. Details: {e}")
        exit(1)

def calculate_scores(ref_protein, queries):
    """Calculates alignment scores for query sequences against a reference."""
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
    
    try:
        max_score = aligner.score(ref_protein, ref_protein)
        if max_score == 0:
            print("Warning: Max score for reference protein is 0. Cannot normalize scores.")
            return ['N/A'] * len(queries)
        else:
            return [aligner.score(ref_protein, q) / max_score if len(q) > 0 else 0 for q in queries]
    except Exception as e:
        print(f"An error occurred during sequence alignment: {e}")
        exit(1)

def save_scores(df, scores, output_prefix):
    """Saves the scores to a CSV file."""
    try:
        df['score'] = scores
        output_filename = f"{output_prefix}.csv"
        df.to_csv(output_filename, index=False)
        print(f"Pairwise alignment scoring complete. Results saved to {output_filename}")
    except Exception as e:
        print(f"Error saving results to {output_filename}: {e}")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Calculate pairwise alignment scores for protein sequences.")
    parser.add_argument("--generated_seqs_csv", required=True, help="CSV file with a 'ORF' column.")
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
    except ValueError as e:
        print(f"Error getting reference protein: {e}")
        exit(1)
        
    g_seqs_df = load_sequences(args.generated_seqs_csv)
    trimmed_queries = trim_sequences(g_seqs_df, args.structure_start, args.structure_end)
    scores = calculate_scores(ref_protein, trimmed_queries)
    save_scores(g_seqs_df, scores, args.output_prefix)

if __name__ == '__main__':
    main()
