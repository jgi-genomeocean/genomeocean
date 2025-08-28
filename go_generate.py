#!/usr/bin/env python

import argparse
import random
import os
from genomeocean.generation import SequenceGenerator

def validate_nucleotide_sequence(seq):
    valid_nucleotides = {'A', 'C', 'G', 'T'}
    if not all(nucleotide in valid_nucleotides for nucleotide in seq):
        raise argparse.ArgumentTypeError(f"Invalid nucleotide sequence: {seq}. Only 'A', 'C', 'G', 'T' are allowed.")
    return seq

def main():
    parser = argparse.ArgumentParser(description="Generate DNA sequences using GenomeOcean model.")

    # Create a mutually exclusive group for model_dir and model
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('--model_dir', type=str, help='Model directory or path to a local copy of the model.')
    model_group.add_argument('--model', type=str, choices=['base', 'bgc'], help='Predefined model to use (base or bgc).')

    # Create a mutually exclusive group for prompts and promptfile
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument('--prompts', type=lambda s: [validate_nucleotide_sequence(seq) for seq in s.split(',')], help="Comma-separated list of DNA sequences as prompts.")
    prompt_group.add_argument('--promptfile', type=str, help='File containing DNA sequences as prompts.')
    prompt_group.add_argument('--zero_shot', action='store_true', help='Use no prompt (zero-shot generation).')

    parser.add_argument('--num', type=int, default=10, help='Number of sequences to generate for each prompt.')
    parser.add_argument('--min_seq_len', type=int, help='Minimum length of generated sequences in tokens.')
    parser.add_argument('--max_seq_len', type=int, default=10240, help='Maximum length of generated sequences in tokens.')
    parser.add_argument('--temperature', type=float, help='Temperature for sampling.')
    parser.add_argument('--top_k', type=int, default=-1, help='Top_k for sampling.')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top_p for sampling.')
    parser.add_argument('--presence_penalty', type=float, default=0, help='Presence penalty for sampling.')
    parser.add_argument('--frequency_penalty', type=float, default=0, help='Frequency penalty for sampling.')
    parser.add_argument('--repetition_penalty', type=float, help='Repetition penalty for sampling.')
    parser.add_argument('--seed', type=int, default=random.randint(0, 999999999), help='Random seed for sampling.')
    parser.add_argument('--prepend_prompt_to_output', type=bool, default=True, help='Prepend prompt to output sequences.')
    parser.add_argument('--max_repeats', type=float, default=100, help='Remove sequences with more than k% simple repeats.')

    # Generate a default output prefix with a randomized number
    default_out_prefix = os.path.join(os.getcwd(), f"go_seq_{random.randint(1000, 9999)}")
    parser.add_argument('--out_prefix', type=str, default=default_out_prefix, help='Output file prefix.')

    parser.add_argument('--out_format', type=str, choices=['fa', 'txt'], default='fa', help='Output format (txt or fa).')

    parser.add_argument('--preset', type=str, choices=['conservative', 'conservative_long', 'creative', 'creative_long'], default='conservative', help='Preset configuration for generation parameters.')

    args = parser.parse_args()

    # if zero_shot
    if args.zero_shot:
        args.prompts = [""]

    # Set the default model if neither --model_dir nor --model is provided
    if args.model_dir is None and args.model is None:
        args.model = 'base'

    # Determine the model directory based on the provided arguments
    model_dir = args.model_dir
    if args.model:
        if args.model == 'base':
            model_dir = 'pGenomeOcean/GenomeOcean-4B'  # Example path for the base model
        elif args.model == 'bgc':
            model_dir = 'pGenomeOcean/GenomeOcean-4B-bgcFM'  # Example path for the BGC model

    # Apply preset configurations
    if args.preset:
        if args.preset == 'conservative':
            min_seq_len = 1024
            temperature = 0.7
            repetition_penalty = 1.0
            top_k = -1
            top_p = 0.9
        elif args.preset == 'conservative_long':
            min_seq_len = 9600
            temperature = 0.7
            repetition_penalty = 1.0
            top_k = -1
            top_p = 0.9
        elif args.preset == 'creative':
            min_seq_len = 1024
            temperature = 0.9
            repetition_penalty = 1.2
            top_k = -1
            top_p = 0.9
        elif args.preset == 'creative_long':
            min_seq_len = 9600
            temperature = 0.9
            repetition_penalty = 1.2
            top_k = -1
            top_p = 0.9
        else:
            raise argparse.ArgumentTypeError(f"Please select between presets [conservative, conservative_long, creative, creative_long].")

    # Override preset values with user-provided values if they exist
    if args.min_seq_len is not None:
        min_seq_len = args.min_seq_len
    if args.temperature is not None:
        temperature = args.temperature
    if args.repetition_penalty is not None:
        repetition_penalty = args.repetition_penalty
    if args.top_k != -1:
        top_k = args.top_k
    if args.top_p != 0.9:
        top_p = args.top_p

    # Initialize the SequenceGenerator with the provided arguments
    seq_gen = SequenceGenerator(
        model_dir=model_dir,
        prompts=args.prompts,
        promptfile=args.promptfile,
        num=args.num,
        min_seq_len=min_seq_len,
        max_seq_len=args.max_seq_len,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        repetition_penalty=repetition_penalty,
        seed=args.seed,
    )

    # Generate sequences
    all_generated = seq_gen.generate_sequences(
        prepend_prompt_to_output=args.prepend_prompt_to_output,
        max_repeats=args.max_repeats,
    )

    # Save the generated sequences to the specified output file
    seq_gen.save_sequences(
        all_generated,
        out_prefix=args.out_prefix,
        out_format=args.out_format,
    )

if __name__ == '__main__':
    main()