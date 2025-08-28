import argparse
import subprocess
import pandas as pd
import os
import logging
from matplotlib_venn import venn2
import matplotlib.pyplot as plt

def run_autocomplete_batch(tasks_file, model_dir, num_sequences, output_prefix, scoring_method, log_file, top_k, top_p, temperature):
    """Runs the autocomplete workflow in batch mode."""
    command = [
        'bash',
        './run_auto_complete_workflow_batch.sh',
        tasks_file,
        model_dir,
        str(num_sequences),
        output_prefix,
        scoring_method,
        str(top_k),
        str(top_p),
        str(temperature)
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

def main():
    parser = argparse.ArgumentParser(description="Battle-bot for gene autocompletion.")
    parser.add_argument("--model1_dir", required=True, help="Directory of model1.")
    parser.add_argument("--model2_dir", required=True, help="Directory of model2.")
    parser.add_argument("--model1_genes", required=True, help="CSV file with genes for model1.")
    parser.add_argument("--model2_genes", required=True, help="CSV file with genes for model2.")
    parser.add_argument("--output_prefix", default="battle-bot", help="Prefix for saving results.")
    parser.add_argument("--debug", action="store_true", help="Keep intermediate files.")
    parser.add_argument("--max_genes", type=int, default=100, help="Maximum number of genes to process for each model.")
    parser.add_argument("--top_k", type=int, default=30, help="Top-k sampling.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling.")
    args = parser.parse_args()

    # Create output directory
    output_dir = os.path.dirname(args.output_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_file = f"{args.output_prefix}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Read the gene files
    try:
        model1_genes_df = pd.read_csv(args.model1_genes)
        model2_genes_df = pd.read_csv(args.model2_genes)
    except FileNotFoundError as e:
        logging.error(f"Error: {e.filename} not found.")
        return

    # Limit number of genes
    if len(model1_genes_df) > args.max_genes:
        logging.info(f"Sampling {args.max_genes} genes from {args.model1_genes}")
        model1_genes_df = model1_genes_df.sample(n=args.max_genes, random_state=42)
    if len(model2_genes_df) > args.max_genes:
        logging.info(f"Sampling {args.max_genes} genes from {args.model2_genes}")
        model2_genes_df = model2_genes_df.sample(n=args.max_genes, random_state=42)

    def process_genes_batch(genes_df, genes_name, model_dir, output_prefix_template):
        tasks = []
        for index, row in genes_df.iterrows():
            gene_id = row['id']
            gene_seq = row['gene']
            orf_len = len(row['ORF'])
            gene_len = len(gene_seq)
            
            task_output_prefix = output_prefix_template.format(gene_id=gene_id)

            tasks.append({
                'gene_id': gene_id,
                'sequence': gene_seq,
                'start': 0,
                'end': gene_len - 1,
                'strand': 1,
                'prompt_start': 0,
                'prompt_end': 150,
                'structure_start': 50,
                'structure_end': orf_len,
                'output_prefix': task_output_prefix
            })

        tasks_df = pd.DataFrame(tasks)
        tasks_filename = f"temp_tasks_{genes_name}_{os.path.basename(model_dir)}.csv"
        tasks_df.to_csv(tasks_filename, index=False)

        logging.info(f"Running autocomplete in batch for {genes_name} with model {model_dir}")
        run_autocomplete_batch(
            tasks_filename,
            model_dir,
            100,
            f"{args.output_prefix}/{genes_name}_model_{os.path.basename(model_dir)}",
            "pairwise",
            log_file,
            args.top_k,
            args.top_p,
            args.temperature
        )

        if not args.debug:
            os.remove(tasks_filename)

    # Create subdirectories for outputs
    os.makedirs(f"{args.output_prefix}/model1_genes_model1", exist_ok=True)
    os.makedirs(f"{args.output_prefix}/model2_genes_model2", exist_ok=True)
    os.makedirs(f"{args.output_prefix}/model2_genes_model1", exist_ok=True)
    os.makedirs(f"{args.output_prefix}/model1_genes_model2", exist_ok=True)

    # Run the four scenarios in batch
    logging.info("Starting autocompletion runs...")
    process_genes_batch(model1_genes_df, "model1_genes", args.model1_dir, f"{args.output_prefix}/model1_genes_model1/gene_{{gene_id}}")
    process_genes_batch(model2_genes_df, "model2_genes", args.model2_dir, f"{args.output_prefix}/model2_genes_model2/gene_{{gene_id}}")
    process_genes_batch(model2_genes_df, "model2_genes", args.model1_dir, f"{args.output_prefix}/model2_genes_model1/gene_{{gene_id}}")
    process_genes_batch(model1_genes_df, "model1_genes", args.model2_dir, f"{args.output_prefix}/model1_genes_model2/gene_{{gene_id}}")
    logging.info("Finished autocompletion runs.")

def get_autocompleted_genes(output_dir, genes_df, threshold):
    autocompleted_genes = set()
    for index, row in genes_df.iterrows():
        gene_id = row['id']
        result_file = f"{output_dir}/gene_{gene_id}.csv"
        if os.path.exists(result_file):
            try:
                scores_df = pd.read_csv(result_file)
                if not scores_df.empty and scores_df['score'].max() >= threshold:
                    autocompleted_genes.add(gene_id)
            except pd.errors.EmptyDataError:
                logging.warning(f"File {result_file} is empty.")
    return autocompleted_genes

    # Analyze results
    logging.info("Analyzing results...")
    
    summaries = []
    thresholds = [0.8, 0.6, 0.4]

    for threshold in thresholds:
        logging.info(f"Analyzing with threshold: {threshold}")

        model1_genes_m1_completed = get_autocompleted_genes(f"{args.output_prefix}/model1_genes_model1", model1_genes_df, threshold)
        model1_genes_m2_completed = get_autocompleted_genes(f"{args.output_prefix}/model1_genes_model2", model1_genes_df, threshold)
        model2_genes_m1_completed = get_autocompleted_genes(f"{args.output_prefix}/model2_genes_model1", model2_genes_df, threshold)
        model2_genes_m2_completed = get_autocompleted_genes(f"{args.output_prefix}/model2_genes_model2", model2_genes_df, threshold)

        # Summary statistics
        summary = {
            "threshold": threshold,
            "model1_genes_total": len(model1_genes_df),
            "model1_genes_model1_completed": len(model1_genes_m1_completed),
            "model1_genes_model2_completed": len(model1_genes_m2_completed),
            "model1_genes_overlap": len(model1_genes_m1_completed.intersection(model1_genes_m2_completed)),
            "model2_genes_total": len(model2_genes_df),
            "model2_genes_model1_completed": len(model2_genes_m1_completed),
            "model2_genes_model2_completed": len(model2_genes_m2_completed),
            "model2_genes_overlap": len(model2_genes_m1_completed.intersection(model2_genes_m2_completed))
        }
        summaries.append(summary)
        logging.info(summary)

        # Create Venn diagrams
        plt.figure()
        venn2([model1_genes_m1_completed, model1_genes_m2_completed], set_labels=('Model 1', 'Model 2'))
        plt.title(f"Overlap for model1_genes (threshold >= {threshold})")
        plt.savefig(f"{args.output_prefix}_model1_genes_venn_threshold_{threshold}.pdf")
        logging.info(f"Venn diagram for model1_genes with threshold {threshold} saved.")

        plt.figure()
        venn2([model2_genes_m1_completed, model2_genes_m2_completed], set_labels=('Model 1', 'Model 2'))
        plt.title(f"Overlap for model2_genes (threshold >= {threshold})")
        plt.savefig(f"{args.output_prefix}_model2_genes_venn_threshold_{threshold}.pdf")
        logging.info(f"Venn diagram for model2_genes with threshold {threshold} saved.")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(f"{args.output_prefix}_summary.csv", index=False)
    logging.info("Summary statistics saved.")

    logging.info("Battle-bot finished.")

if __name__ == "__main__":
    main()
