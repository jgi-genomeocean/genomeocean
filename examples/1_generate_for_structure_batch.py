
import pandas as pd
import os
import sys
import argparse
import atexit

from genomeocean.dnautils import get_nuc_seq_by_id, reverse_complement
from genomeocean.generation import SequenceGenerator

def generate_sequences_batch(
    tasks_file,
    model_dir,
    num,
    min_seq_len,
    max_seq_len,
    output_prefix
):
    """
    Generates sequences in batch from a list of tasks.
    """
    try:
        tasks_df = pd.read_csv(tasks_file)
    except FileNotFoundError:
        print(f"Error: Tasks file not found at {tasks_file}")
        sys.exit(1)

    prompts = []
    task_info = []

    for index, row in tasks_df.iterrows():
        gene_id = row.get('gene_id')
        sequence = row.get('sequence')
        start = row.get('start', 0)
        end = row.get('end', 0)
        strand = row.get('strand', 1)
        prompt_start = row.get('prompt_start', 0)
        prompt_end = row.get('prompt_end', 0)
        
        if pd.isna(sequence):
            sequence = None
        if pd.isna(gene_id):
            gene_id = None

        if sequence:
            gene = sequence
        elif gene_id:
            gene = get_nuc_seq_by_id(gene_id, start=start, end=end)
            if gene is None:
                print(f"Failed to retrieve gene sequence {gene_id} from {start} to {end}")
                continue
        else:
            print("Either gene_id or sequence must be provided for each task.")
            continue

        if strand == -1:
            gene = reverse_complement(gene)
        
        prompt = gene[prompt_start:prompt_end]
        prompts.append(prompt)
        task_info.append({
            'output_prefix': row['output_prefix'],
            'gene_id': gene_id,
            'sequence': sequence,
            'start': start,
            'end': end,
            'strand': strand,
            'prompt_start': prompt_start,
            'prompt_end': prompt_end
        })

    prompts_file = f"{output_prefix}_prompts.txt"
    with open(prompts_file, 'w') as f:
        for p in prompts:
            f.write(p + '\n')

    seq_gen = SequenceGenerator(
        model_dir=model_dir,
        promptfile=prompts_file,
        num=num,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
        top_k=30,
        top_p=0.95,
        temperature=1.0,
        presence_penalty=0.5,
        frequency_penalty=0.5,
        repetition_penalty=1.0,
        seed=1234
    )
    
    g_seqs = seq_gen.generate_sequences(prepend_prompt_to_output=True, max_repeats=-1)
    print(f"Total {g_seqs.shape[0]} sequences were generated.")
    
    # Split the generated sequences back according to the original tasks
    sequences_per_prompt = num
    for i, info in enumerate(task_info):
        start_index = i * sequences_per_prompt
        end_index = (i + 1) * sequences_per_prompt
        task_seqs = g_seqs.iloc[start_index:end_index]
        
        output_filename = info['output_prefix'] + '.csv'
        output_dir = os.path.dirname(output_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        task_seqs.to_csv(output_filename, sep='\t', index=False)
        print(f"Generated sequences for task {i} saved to {output_filename}")
    # Register cleanup function to remove temporary prompts file
    atexit.register(os.remove, prompts_file)        

def main():
    parser = argparse.ArgumentParser(description="Generate DNA sequences in batch from a list of prompts.")
    parser.add_argument("--tasks_file", required=True, help="CSV file with tasks. Each row should define a generation task.")
    parser.add_argument("--model_dir", required=True, help="Directory of the language model.")
    parser.add_argument("--num", type=int, default=20, help="Number of sequences to generate per task.")
    parser.add_argument("--min_seq_len", type=int, default=1000, help="Minimum sequence length.")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length.")
    parser.add_argument("--output_prefix", default='batch_generated', help="Prefix for temporary and summary files.")
    
    args = parser.parse_args()

    print(f"Parameters: {args}")

    generate_sequences_batch(
        tasks_file=args.tasks_file,
        model_dir=args.model_dir,
        num=args.num,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        output_prefix=args.output_prefix,
    )

if __name__ == '__main__':
    main()
