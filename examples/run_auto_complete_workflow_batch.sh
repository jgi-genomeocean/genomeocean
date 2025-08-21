#!/bin/bash

# Check for the correct number of arguments
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <tasks_file> <model_dir> <num_sequences> <output_prefix> <scoring_method>"
    exit 1
fi

TASKS_FILE="$1"
MODEL_DIR="$2"
NUM_SEQS="$3"
OUTPUT_PREFIX="$4"
SCORING_METHOD="$5"

# Stage 1: Generate sequences in batch
python ./1_generate_for_structure_batch.py \\
  --tasks_file "$TASKS_FILE" \\
  --model_dir "$MODEL_DIR" \\
  --num "$NUM_SEQS" \\
  --min_seq_len 1000 \\
  --max_seq_len 1024 \\
  --output_prefix "$OUTPUT_PREFIX"

# Check if the first script was successful
if [ $? -ne 0 ]; then
    echo "Error: Batch sequence generation failed."
    exit 1
fi

# Read tasks file and process each task
while IFS=, read -r gene_id sequence start end strand pstart pend sstart send task_output_prefix; do
    # Skip header
    if [[ "$gene_id" == "gene_id" ]]; then
        continue
    fi
    
    GENERATED_SEQS_CSV="$task_output_prefix.csv"
    CONSISTENT_SEQS_CSV="$task_output_prefix.consistent.csv"
    
    # Stage 2: Filter generated sequences for consistency
    python ./consistent_generated.py \\
        --input "$GENERATED_SEQS_CSV" \\
        --output "$CONSISTENT_SEQS_CSV"

    # Check if the consistency filter script was successful
    if [ $? -ne 0 ]; then
        echo "Error: Filtering for consistency failed for $task_output_prefix."
        continue
    fi

    # Stage 3: Scoring
    if [ "$SCORING_METHOD" = "pairwise" ]; then
        python ./pairwise_alignment_scoring.py \\
            --generated_seqs_csv "$CONSISTENT_SEQS_CSV" \\
            --sequence "$sequence" \\
            --start "$start" \\
            --end "$end" \\
            --strand "$strand" \\
            --structure_start "$sstart" \\
            --structure_end "$send" \\
            --output_prefix "$task_output_prefix"
    else
        echo "Error: Invalid scoring method. Only 'pairwise' is implemented for batch mode."
        continue
    fi

    # Check if the scoring script was successful
    if [ $? -ne 0 ]; then
        echo "Error: Scoring failed for $task_output_prefix."
        continue
    fi

done < <(tail -n +2 "$TASKS_FILE") # tail -n +2 to skip header

echo "Batch workflow completed successfully."
