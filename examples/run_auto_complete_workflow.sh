#!/bin/bash

# Check for the correct number of arguments
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <config_file> <model_dir> <num_sequences> <output_prefix>"
    exit 1
fi

CONFIG_FILE="$1"
MODEL_DIR="$2"
NUM_SEQS="$3"
OUTPUT_PREFIX="$4"

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

# Stage 2: lDDT scoring
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

# Check if the second script was successful
if [ $? -ne 0 ]; then
    echo "Error: lDDT scoring failed."
    exit 1
fi

echo "Workflow completed successfully."