import argparse
import subprocess
import pandas as pd
import os
import logging
import numpy as np

def run_autocomplete_batch(tasks_file, model_dir, num_sequences, output_prefix, scoring_method, log_file):
    """Runs the autocomplete workflow in batch mode."""
    command = [
        'bash',
        './run_auto_complete_workflow_batch.sh',
        tasks_file,
        model_dir,
        str(num_sequences),
        output_prefix,
        scoring_method
    ]
    logging.info(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"Error running autocomplete batch for {output_prefix}:")
        logging.error(f"The command was: {' '.join(command)}")
        logging.error(result.stderr)
        return False
    logging.info(f"Successfully ran autocomplete batch for {output_prefix}")
    return True

def analyze_results(output_dir, genes_df):
    """
    Analyzes the results to find the best prompt length for each gene.
    Returns a DataFrame where each row is a gene and each column is a prompt percentage.
    """
    gene_results = []
    prompt_percentages = list(range(5, 51, 5))

    for index, row in genes_df.iterrows():
        gene_id = row['id']
        gene_data = {'gene_id': gene_id}
        
        scores_for_gene = {}

        for percent in prompt_percentages:
            result_file = f"{output_dir}/prompt_{percent}/gene_{gene_id}.csv"
            score = np.nan # Default to NaN if no score
            if os.path.exists(result_file):
                try:
                    scores_df = pd.read_csv(result_file)
                    if (not scores_df.empty) and ('score' in scores_df.columns):
                        top_scores = scores_df['score'].nlargest(3)
                        if len(top_scores) > 0:
                            score = top_scores.mean()
                except pd.errors.EmptyDataError:
                    logging.warning(f"File {result_file} is empty.")
            
            gene_data[f'prompt_{percent}%'] = score
            scores_for_gene[percent] = score
        
        # Find best score and prompt for the current gene
        valid_scores = {p: s for p, s in scores_for_gene.items() if not np.isnan(s)}
        if valid_scores:
            best_prompt = max(valid_scores, key=valid_scores.get)
            best_score = valid_scores[best_prompt]
            gene_data['best_prompt_percent'] = best_prompt
            gene_data['best_score'] = best_score
        else:
            gene_data['best_prompt_percent'] = np.nan
            gene_data['best_score'] = np.nan

        gene_results.append(gene_data)

    results_df = pd.DataFrame(gene_results)
    
    # Reorder columns
    cols = ['gene_id'] + [f'prompt_{p}%' for p in prompt_percentages] + ['best_prompt_percent', 'best_score']
    results_df = results_df[cols]
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description="Optimize prompt length for gene autocompletion.")
    parser.add_argument("--model_dir", required=True, help="Directory of the model.")
    parser.add_argument("--genes_file", required=True, help="CSV file with genes.")
    parser.add_argument("--output_prefix", default="optimize_prompt", help="Prefix for saving results.")
    parser.add_argument("--debug", action="store_true", help="Keep intermediate files.")
    parser.add_argument("--max_genes", type=int, default=None, help="Maximum number of genes to process. If not set, all genes will be processed.")
    parser.add_argument("--num_sequences", type=int, default=20, help="Number of sequences to generate for each task.")
    args = parser.parse_args()

    # Create output directory
    output_dir = os.path.dirname(args.output_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_file = f"{args.output_prefix}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Read the gene file
    try:
        genes_df = pd.read_csv(args.genes_file)
    except FileNotFoundError as e:
        logging.error(f"Error: {e.filename} not found.")
        return

    # Limit number of genes
    if args.max_genes is not None and len(genes_df) > args.max_genes:
        logging.info(f"Sampling {args.max_genes} genes from {args.genes_file}")
        genes_df = genes_df.sample(n=args.max_genes, random_state=42)

    for prompt_percent in range(5, 51, 5):
        logging.info(f"Processing with prompt length: {prompt_percent}%")
        
        tasks = []
        for index, row in genes_df.iterrows():
            gene_id = row['id']
            gene_seq = row['gene']
            orf_len = len(row['ORF'])
            gene_len = len(gene_seq)
            
            prompt_end = int(gene_len * prompt_percent / 100)
            structure_start = int(prompt_end / 3)
            
            task_output_prefix = f"{args.output_prefix}/prompt_{prompt_percent}/gene_{gene_id}"
            os.makedirs(os.path.dirname(task_output_prefix), exist_ok=True)

            tasks.append({
                'gene_id': gene_id,
                'sequence': gene_seq,
                'start': 0,
                'end': gene_len - 1,
                'strand': 1,
                'prompt_start': 0,
                'prompt_end': prompt_end,
                'structure_start': structure_start,
                'structure_end': orf_len,
                'output_prefix': task_output_prefix
            })

        tasks_df = pd.DataFrame(tasks)
        tasks_filename = f"temp_tasks_prompt_{prompt_percent}.csv"
        tasks_df.to_csv(tasks_filename, index=False)

        run_autocomplete_batch(
            tasks_filename,
            args.model_dir,
            args.num_sequences,
            f"{args.output_prefix}/prompt_{prompt_percent}",
            "pairwise",
            log_file
        )

        if not args.debug:
            os.remove(tasks_filename)

    logging.info("Analyzing results...")
    results_df = analyze_results(args.output_prefix, genes_df)
    
    if not results_df.empty:
        results_df.to_csv(f"{args.output_prefix}_summary.csv", index=False)
        logging.info(f"Summary of results saved to {args.output_prefix}_summary.csv")
        
        best_prompt_overall = results_df['best_prompt_percent'].mode()
        if not best_prompt_overall.empty:
            logging.info(f"Most frequent best prompt percentage: {best_prompt_overall.iloc[0]}%")
    else:
        logging.warning("No results to analyze.")

    logging.info("Optimization finished.")

if __name__ == "__main__":
    main()