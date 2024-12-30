"""Example of embedding sequences using a pre-trained model.

This script demonstrates how to embed sequences using a pre-trained model.

python embedding_sequences.py \
    --model_path pGenomeOcean/GenomeOcean-4B \
    --sequence_file ../sample_data/dna_sequences.txt \
    --model_max_length 1024 \
    --batch_size 10 \
    --output_file outputs/embeddings.npy

"""

from genomeocean.llm_utils import LLMUtils
import os
import pandas as pd
import numpy as np
import argparse

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

    with open(args.sequence_file, "r") as f:
        sequences = f.read().splitlines()
    print(f"Get {len(sequences)} sequences from {args.sequence_file} with max length {np.max([len(seq) for seq in sequences])}")


    llm = LLMUtils(
        model_dir=args.model_path, 
        model_max_length=args.model_max_length,
    )
    embeddings = llm.embedding(sequences, batch_size=args.batch_size)
    
    print(f"Save embedding to {args.output_file}")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    np.save(args.output_file, embeddings)

if __name__ == "__main__":
    main()