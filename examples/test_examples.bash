#!/bin/bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "testing $1"

if [ $1 == "autocomplete" ]; then
    python autocomplete_structure.py \
        --gen_id NZ_JAYXHC010000003.1 \
        --start 157 \
        --end 1698 \
        --strand -1 \
        --prompt_start 0 \
        --prompt_end 600 \
        --structure_start 150 \
        --structure_end 500 \
        --model_dir pGenomeOcean/GenomeOcean-4B \
        --num 10 \
        --min_seq_len 250 \
        --max_seq_len 300 \
        --foldmason_path ~/bin/foldmason \
        --output_prefix outputs/gmp

elif [ $1 == "prompt_mutation" ]; then
    python autocomplete_structure.py \
        --gen_id OY729418.1 \
        --start 1675256 \
        --end 1676176 \
        --strand -1 \
        --prompt_start 0 \
        --prompt_end 450 \
        --mutate_prompt 1 \
        --structure_start 0 \
        --structure_end 341 \
        --model_dir pGenomeOcean/GenomeOcean-4B \
        --num 20 \
        --min_seq_len 250 \
        --max_seq_len 300 \
        --foldmason_path ~/bin/foldmason \
        --output_prefix outputs/trapl_wt_mutations

elif [ $1 == "generation" ]; then
    python generate_sequences.py \
        --model_dir pGenomeOcean/GenomeOcean-4B \
        --promptfile ../sample_data/dna_sequences.txt \
        --out_prefix outputs/generated \
        --out_format fa \
        --num 10 \
        --min_seq_len 100 \
        --max_seq_len 100 \
        --temperature 1.3 \
        --top_k -1 \
        --top_p 0.7 \
        --max_repeats 100 \
        --presence_penalty 0.5 \
        --frequency_penalty 0.5 \
        --repetition_penalty 1.0 \
        --seed 123 \
        --sort_by_orf_length

elif [ $1 == "genomescan" ]; then
    python scan_genome.py \
        --model_dir pGenomeOcean/GenomeOcean-4B \
        --model_max_length 10000 \
        --genomefile ../sample_data/sample_genomes.fna.gz \
        --out_prefix outputs/scan/ \
        --out_postfix scores_meta.csv

else
    echo "Invalid command"
fi
