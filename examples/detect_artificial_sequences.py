"""Example of detecting artificial sequences using a fine-tuned GenomeOcean model.

python detect_artificial_sequences.py \
    --model_dir pGenomeOcean/GenomeOcean-Artificial-Detector \
    --sequence_file ../sample_data/dna_sequences.txt \
    --model_max_length 1024 \
    --batch_size 10 \
    --output_dir outputs/artificial_test

"""

from genomeocean.llm_utils import LLMUtils
import os
import pandas as pd
import numpy as np
import argparse

def detect_artificial(sequence, model_dir, model_max_length, batch_size):
    llm = LLMUtils(
        model_dir=model_dir, 
        model_max_length=model_max_length,
        is_classification_model=True,
    )
    logits = llm.predict(sequence, batch_size=batch_size, do_embedding=False)
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1)[:, None]
    preds = np.argmax(probs, axis=1)
    return probs[:, 0], preds
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect artificial sequences from GenomeOcean model')
    parser.add_argument('--model_dir', type=str, help='Path to the model')
    parser.add_argument('--sequence_file', type=str, help='Path to the sequence file')
    parser.add_argument('--model_max_length', type=int, help='Max length of the model, set as sequence length in bp / 4, e.g. 256 for 1024bp sequence')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
    parser.add_argument('--output_dir', type=str, help='Path to the output file')
    args = parser.parse_args()
    # ensure tat model_max_length does not exceed the model's max length: 10240
    assert args.model_max_length <= 10240

    with open(args.sequence_file, "r") as f:
        sequences = f.read().splitlines()
    print(f"Get {len(sequences)} sequences from {args.sequence_file} with max length {np.max([len(seq) for seq in sequences])}")

    probs, preds = detect_artificial(sequences, args.model_dir, args.model_max_length, args.batch_size)
    print(f"Label 0: Artificial, Label 1: Natural")
    print(f"{len(np.where(preds == 0)[0])} / {len(preds)} sequences are predicted as artificial")
    
    print(f"Save detailed results to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "probs.npy"), probs)
    np.save(os.path.join(args.output_dir, "preds.npy"), preds)