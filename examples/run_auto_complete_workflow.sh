#!/bin/bash

# Check for the correct number of arguments
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <config_file> <model_dir> <num_sequences> <output_prefix> <scoring_method>"
    exit 1
fi

CONFIG_FILE="$1"
MODEL_DIR="$2"
NUM_SEQS="$3"
OUTPUT_PREFIX="$4"
SCORING_METHOD="$5"

# Read variables from the config file
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found!"
    exit 1
fi

source "$CONFIG_FILE"

# Stage 1: Generate sequences
python ../1_generate_for_structure.py \
  --gen_id "$gene" --start "$start" --end "$end" --strand "$strand" \
  --prompt_start "$pstart" --prompt_end "$pend" \
  --num "$NUM_SEQS" --min_seq_len "$min" --max_seq_len "$max" \
  --output_prefix "$OUTPUT_PREFIX" \
  --model_dir "$MODEL_DIR"

# Check if the first script was successful
if [ $? -ne 0 ]; then
    echo "Error: Sequence generation failed."
    exit 1
fi

# Stage 2: Scoring
if [ "$SCORING_METHOD" = "lddt" ]; then
    python ../lddt_scoring.py \
        --generated_seqs_csv "$OUTPUT_PREFIX.csv" \
        --gen_id "$gene" \
        --start "$start" \
        --end "$end" \
        --strand "$strand" \
        --structure_start "$sstart" \
        --structure_end "$send" \
        --foldmason_path "$foldmason" \
        --output_prefix "$OUTPUT_PREFIX"
elif [ "$SCORING_METHOD" = "pairwise" ]; then
    python ../pairwise_alignment_scoring.py \
        --generated_seqs_csv "$OUTPUT_PREFIX.csv" \
        --gen_id "$gene" \
        --start "$start" \
        --end "$end" \
        --strand "$strand" \
        --structure_start "$sstart" \
        --structure_end "$send" \
        --output_prefix "$OUTPUT_PREFIX"
else
    echo "Error: Invalid scoring method. Choose 'lddt' or 'pairwise'."
    exit 1
fi

# Check if the scoring script was successful
if [ $? -ne 0 ]; then
    echo "Error: Scoring failed."
    exit 1
fi

echo "Workflow completed successfully."