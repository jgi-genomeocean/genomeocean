"""Example of embedding sequences using a pre-trained model.

This script demonstrates how to embed sequences using a pre-trained model.
The script takes as input a model path, a gzipped sequence file with one sequence perline, and a model max length.
It returns the embeddings of the sequences in the sequence file.


"""

from genomeocean.llm_utils import calculate_llm_embedding
import pandas as pd
import numpy as np
import argparse

def embeding_sequences(model_path, sequence_file, model_max_length, batch_size=50):
    dna = pd.read_csv(sequence_file, sep='\t', header=None, compression='gzip')
    sequences = dna[0].tolist()
    embeddings = calculate_llm_embedding(
        sequences, 
        model_name_or_path=model_path, 
        model_max_length=model_max_length,
        batch_size=50 # on A100-40G
    )
    return embeddings

def main():
    parser = argparse.ArgumentParser(description='Embedding sequences')
    parser.add_argument('--model_path', type=str, help='Path to the model')
    parser.add_argument('--sequence_file', type=str, help='Path to the sequence file')
    parser.add_argument('--model_max_length', type=int, help='Max length of the model')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
    parser.add_argument('--output_file', type=str, help='Path to the output file')
    args = parser.parse_args()
    # ensure tat model_max_length does not exceed the model's max length: 10240
    assert args.model_max_length <= 10240
    embeddings = embeding_sequences(args.model_path, args.sequence_file, args.model_max_length, batch_size=args.batch_size)
    np.save(args.output_file, embeddings)