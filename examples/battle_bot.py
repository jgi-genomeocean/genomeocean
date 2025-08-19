import argparse
import subprocess
import pandas as pd
import os
import logging
from matplotlib_venn import venn2
import matplotlib.pyplot as plt

def run_autocomplete(config_file, model_dir, genes_file, num_sequences, output_prefix, scoring_method, log_file):
    """Runs the autocomplete workflow."""
    command = [
        'bash',
        './run_auto_complete_workflow.sh',
        config_file,
        model_dir,
        genes_file,
        str(num_sequences),
        output_prefix,
        scoring_method
    ]
    logging.info(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"Error running autocomplete for {output_prefix}:")
        logging.error(f"The command was: {' '.join(command)}")
        logging.error(result.stderr)
        return False
    logging.info(f"Successfully ran autocomplete for {output_prefix}")
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

    def process_genes(genes_df, genes_name, model_dir, output_prefix_template):
        for index, row in genes_df.iterrows():
            gene_id = row['id']
            gene_seq = row['gene']
            orf_len = len(row['ORF'])
            gene_len = len(gene_seq)

            config_content = f"""
sequence="{gene_seq}"
start=0
end={gene_len - 1}
strand=1
pstart=0
pend=300
min=1000
max=1024
sstart=100
send={orf_len}
"""
            config_filename = f"temp_config_{genes_name}_{gene_id}.txt"
            with open(config_filename, "w") as f:
                f.write(config_content)

            output_prefix = output_prefix_template.format(gene_id=gene_id)
            
            logging.info(f"Running autocomplete for {genes_name} gene {gene_id} with model {model_dir}")
            run_autocomplete(
                config_filename,
                model_dir,
                f"temp_{genes_name}.csv",
                20,
                output_prefix,
                "pairwise",
                log_file
            )

            if not args.debug:
                os.remove(config_filename)

    # Create temporary gene files
    model1_genes_df.to_csv("temp_model1_genes.csv", index=False)
    model2_genes_df.to_csv("temp_model2_genes.csv", index=False)

    # Create subdirectories for outputs
    os.makedirs(f"{args.output_prefix}/model1_genes_model1", exist_ok=True)
    os.makedirs(f"{args.output_prefix}/model2_genes_model2", exist_ok=True)
    os.makedirs(f"{args.output_prefix}/model2_genes_model1", exist_ok=True)
    os.makedirs(f"{args.output_prefix}/model1_genes_model2", exist_ok=True)

    # Run the four scenarios
    logging.info("Starting autocompletion runs...")
    process_genes(model1_genes_df, "model1_genes", args.model1_dir, f"{args.output_prefix}/model1_genes_model1/gene_{{gene_id}}")
    process_genes(model2_genes_df, "model2_genes", args.model2_dir, f"{args.output_prefix}/model2_genes_model2/gene_{{gene_id}}")
    process_genes(model2_genes_df, "model2_genes", args.model1_dir, f"{args.output_prefix}/model2_genes_model1/gene_{{gene_id}}")
    process_genes(model1_genes_df, "model1_genes", args.model2_dir, f"{args.output_prefix}/model1_genes_model2/gene_{{gene_id}}")
    logging.info("Finished autocompletion runs.")

    if not args.debug:
        os.remove("temp_model1_genes.csv")
        os.remove("temp_model2_genes.csv")

    def get_autocompleted_genes(output_dir, genes_df):
        autocompleted_genes = set()
        for index, row in genes_df.iterrows():
            gene_id = row['id']
            result_file = f"{output_dir}/gene_{gene_id}.scores.csv"
            if os.path.exists(result_file):
                try:
                    scores_df = pd.read_csv(result_file)
                    if not scores_df.empty and scores_df['score'].max() >= 0.80:
                        autocompleted_genes.add(gene_id)
                except pd.errors.EmptyDataError:
                    logging.warning(f"File {result_file} is empty.")
        return autocompleted_genes

    # Analyze results
    logging.info("Analyzing results...")
    model1_genes_m1_completed = get_autocompleted_genes(f"{args.output_prefix}/model1_genes_model1", model1_genes_df)
    model1_genes_m2_completed = get_autocompleted_genes(f"{args.output_prefix}/model1_genes_model2", model1_genes_df)
    model2_genes_m1_completed = get_autocompleted_genes(f"{args.output_prefix}/model2_genes_model1", model2_genes_df)
    model2_genes_m2_completed = get_autocompleted_genes(f"{args.output_prefix}/model2_genes_model2", model2_genes_df)

    # Summary statistics
    summary = {
        "model1_genes_total": len(model1_genes_df),
        "model1_genes_model1_completed": len(model1_genes_m1_completed),
        "model1_genes_model2_completed": len(model1_genes_m2_completed),
        "model1_genes_overlap": len(model1_genes_m1_completed.intersection(model1_genes_m2_completed)),
        "model2_genes_total": len(model2_genes_df),
        "model2_genes_model1_completed": len(model2_genes_m1_completed),
        "model2_genes_model2_completed": len(model2_genes_m2_completed),
        "model2_genes_overlap": len(model2_genes_m1_completed.intersection(model2_genes_m2_completed))
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(f"{args.output_prefix}_summary.csv", index=False)
    logging.info("Summary statistics saved.")
    logging.info(summary)

    # Create Venn diagrams
    plt.figure()
    venn2([model1_genes_m1_completed, model1_genes_m2_completed], set_labels=('Model 1', 'Model 2'))
    plt.title("Overlap for model1_genes")
    plt.savefig(f"{args.output_prefix}_model1_genes_venn.pdf")
    logging.info("Venn diagram for model1_genes saved.")

    plt.figure()
    venn2([model2_genes_m1_completed, model2_genes_m2_completed], set_labels=('Model 1', 'Model 2'))
    plt.title("Overlap for model2_genes")
    plt.savefig(f"{args.output_prefix}_model2_genes_venn.pdf")
    logging.info("Venn diagram for model2_genes saved.")

    logging.info("Battle-bot finished.")

if __name__ == "__main__":
    main()
