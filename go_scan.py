#!/usr/bin/env python

from genomeocean.llm_utils import LLMUtils
from Bio import SeqIO
import gzip
import os
import numpy as np
import pickle
import argparse

def read_genome(file_path):
    """
    Reads a genome file in FASTA, GenBank, or plain text format (one sequence per line),
    optionally gzipped, and yields nucleotide sequences.
    
    :param file_path: Path to the genome file.
    :return: A generator yielding nucleotide sequences.
    """
    is_gzipped = file_path.endswith('.gz')
    file_path_base, file_extension = os.path.splitext(file_path[:-3] if is_gzipped else file_path)
    
    if file_extension in ['.fa', '.fasta', '.fna', '.ffn', '.faa', '.frn']:
        file_format = 'fasta'
    elif file_extension in ['.gb', '.gbk', '.genbank']:
        file_format = 'genbank'
    elif file_extension in ['.txt']:
        file_format = 'txt'
    else:
        raise ValueError("Unsupported file extension")
    
    open_func = gzip.open if is_gzipped else open
    with open_func(file_path, 'rt') as handle:
        if file_format in ['fasta', 'genbank']:
            for record in SeqIO.parse(handle, file_format):
                yield (record.id, str(record.seq))
        elif file_format == 'txt':
            i = 0
            for line in handle:
                yield (str(i), line.strip())
                i += 1

def split_into_overlapping_segments(sequence, segment_size, overlap_size):
    """
    Splits the input sequence into overlapping segments.
    
    :param sequence: The input nucleotide sequence as a string.
    :param segment_size: The size of each segment.
    :param overlap_size: The size of the overlap between consecutive segments.
    :return: A generator yielding overlapping segments.
    """
    if segment_size <= overlap_size:
        raise ValueError("Segment size must be greater than overlap size.")
    
    step_size = segment_size - overlap_size
    for i in range(0, len(sequence), step_size):
        yield sequence[i:i + segment_size]

def reverse_complement(sequence):
    """
    Returns the reverse complement of the given nucleotide sequence.
    
    :param sequence: The input nucleotide sequence as a string.
    :return: The reverse complement of the sequence.
    """
    complement = str.maketrans('ATCGatcg', 'TAGCtagc')
    return sequence.translate(complement)[::-1]

def calculate_loss(genome_file_path, model='bgc', segment_size=None, overlap_size=None, model_dir=None, use_reverse=False):
    """
    Calculates the loss by segmenting the genome sequences based on the model type and parameters.
    
    :param genome_file_path: Path to the genome file.
    :param model: The type of model ('base', 'bgc', 'bgc_substracted', or custom file path). Default is 'bgc'.
    :param segment_size: The size of each segment (optional, required for custom model).
    :param overlap_size: The size of the overlap between consecutive segments (optional, required for custom model).
    :param model_dir: Directory of the model (optional, required for custom model).
    :param use_reverse: Whether to use the combined scores of forward and reverse strands. Default is False.
    :return: A generator yielding lists of perplexity scores for each sequence.
    """
    presets = {
        'base': ("pGenomeOcean/GenomeOcean-4B", 50000, 5000),
        'bgc': ("pGenomeOcean/GenomeOcean-4B-bgcFM", 50000, 5000)
    }
    
    if model in presets:
        model_dir, segment_size, overlap_size = presets[model]
    elif model == 'custom':
        if model_dir is None or segment_size is None or overlap_size is None:
            raise ValueError("For custom model, model_dir, segment_size, and overlap_size must be provided.")
    elif model == 'bgc_substracted':
        use_reverse = True
    else:
        raise ValueError("Unsupported model type. Use 'base', 'bgc', 'bgc_substracted', or 'custom'.")

    # Initiate the model
    llm = LLMUtils(model_dir=model_dir)
    
    for seq_id, sequence in read_genome(genome_file_path):
        print(f"Scanning forward strand for sequence {seq_id}...")
        # Forward strand
        segments_forward = list(split_into_overlapping_segments(sequence, segment_size, overlap_size))
        scores_forward = llm.compute_token_perplexity(segments_forward)
        scores_concat_forward = scores_forward.pop(0)
        for score_segment in scores_forward:
            scores_concat_forward.extend(score_segment[overlap_size:])
        scores_concat_forward = np.array(scores_concat_forward).astype(np.float16)

        if use_reverse:
            print(f"Scanning reverse strand for sequence {seq_id}...")
            # Reverse strand
            reverse_sequence = reverse_complement(sequence)
            segments_reverse = list(split_into_overlapping_segments(reverse_sequence, segment_size, overlap_size))
            scores_reverse = llm.compute_token_perplexity(segments_reverse)
            scores_concat_reverse = scores_reverse.pop(0)
            for score_segment in scores_reverse:
                scores_concat_reverse.extend(score_segment[overlap_size:])
            scores_concat_reverse = np.array(scores_concat_reverse).astype(np.float16)
            
            combined_scores = np.minimum(scores_concat_forward, scores_concat_reverse)
        else:
            combined_scores = scores_concat_forward

        yield (seq_id, combined_scores)

def calculate_bgc_substracted_loss(genome_file_path, segment_size, overlap_size):
    """
    Calculates the bgc_substracted loss by segmenting the genome sequences and using both 'base' and 'bgc' models.
    
    :param genome_file_path: Path to the genome file.
    :param segment_size: The size of each segment.
    :param overlap_size: The size of the overlap between consecutive segments.
    :return: A generator yielding lists of subtracted perplexity scores for each sequence.
    """
    base_scores = {seq_id: scores for seq_id, scores in calculate_loss(
        genome_file_path, model='base', segment_size=segment_size, overlap_size=overlap_size, use_reverse=True
    )}
    
    bgc_scores = {seq_id: scores for seq_id, scores in calculate_loss(
        genome_file_path, model='bgc', segment_size=segment_size, overlap_size=overlap_size, use_reverse=True
    )}
    
    for seq_id in base_scores:
        yield (seq_id, base_scores[seq_id] - bgc_scores[seq_id])

def main():
    parser = argparse.ArgumentParser(description="Calculate loss for genome sequences using specified model. The output will be given a *.pkl extension.")
    parser.add_argument('genome_file', type=str, help="Path to the genome file.")
    parser.add_argument('output_prefix', type=str, help="Prefix for the output file.")
    parser.add_argument('--model', type=str, choices=['base', 'bgc', 'bgc_substracted', 'custom'], default='bgc', help="Type of model to use. Default is 'bgc'.")
    parser.add_argument('--segment_size', type=int, help="Size of each segment (required for custom model).")
    parser.add_argument('--overlap_size', type=int, help="Size of the overlap between consecutive segments (required for custom model).")
    parser.add_argument('--model_dir', type=str, help="Directory of the model (required for custom model).")
    parser.add_argument('--use_reverse', action='store_true', help="Use the combined scores of forward and reverse strands.")
    args = parser.parse_args()

    output_file = args.output_prefix + ".pkl"

    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Exiting without processing.")
        return

    if args.model == 'bgc_substracted':
        results = {seq_id: scores for seq_id, scores in calculate_bgc_substracted_loss(
            args.genome_file, segment_size=args.segment_size, overlap_size=args.overlap_size
        )}
    else:
        results = {seq_id: scores for seq_id, scores in calculate_loss(
            args.genome_file, 
            model=args.model, 
            segment_size=args.segment_size, 
            overlap_size=args.overlap_size, 
            model_dir=args.model_dir,
            use_reverse=args.use_reverse
        )}

    with open(output_file, "wb") as file:
        pickle.dump(results, file)

if __name__ == "__main__":
    main()