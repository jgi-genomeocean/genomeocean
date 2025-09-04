"""
Optimize generation parameters to minimize tandem repeats in generated sequences.

This script uses Bayesian optimization to find the best set of hyperparameters for
sequence generation, with the goal of minimizing the percentage of tandem repeats.

Example usage:
```
# You may need to install dependencies first:
# pip install scikit-optimize matplotlib pydustmasker

# this may be needed to avoid issues with multiprocessing on some systems
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python examples/optimize_generation_parameters.py \
    --model_dir pGenomeOcean/GenomeOcean-4B \
    --promptfile sample_data/dna_sequences.txt \
    --n_calls 50 \
    --num_seqs_per_call 5 \
    --output_dir outputs/optimization
```
"""
import os
import argparse
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import matplotlib.pyplot as plt
from skopt.plots import plot_convergence, plot_objective

from genomeocean.generation import SequenceGenerator
from genomeocean.dnautils import find_tandem_repeats_percentage

# Global variable for the sequence generator to avoid reloading the model
seq_gen = None

# --- Objective Function ---
def evaluate_params(temperature, top_p, presence_penalty, frequency_penalty, repetition_penalty, min_seq_len, max_seq_len, num, max_repeats):
    """
    Objective function for Bayesian optimization.
    Generates sequences with a given set of parameters and returns the mean
    tandem repeat percentage.
    """
    global seq_gen
    
    print(f"Testing parameters: temperature={temperature:.3f}, top_p={top_p:.3f}, "
          f"presence_penalty={presence_penalty:.3f}, frequency_penalty={frequency_penalty:.3f}, "
          f"repetition_penalty={repetition_penalty:.3f}")

    # Set new parameters for the global generator instance
    # This assumes SequenceGenerator stores these as public attributes.
    seq_gen.temperature = temperature
    seq_gen.top_p = top_p
    seq_gen.presence_penalty = presence_penalty
    seq_gen.frequency_penalty = frequency_penalty
    seq_gen.repetition_penalty = repetition_penalty
    seq_gen.min_seq_len = min_seq_len
    seq_gen.max_seq_len = max_seq_len
    seq_gen.num = num
    
    # Generate sequences
    try:
        generated_seqs_df = seq_gen.generate_sequences(prepend_prompt_to_output=False, max_repeats=max_repeats)
        
        if generated_seqs_df.empty:
            print("Warning: No sequences were generated. Returning a high penalty.")
            return 100.0

        # Calculate tandem repeat percentage for each sequence
        trf_percentages = generated_seqs_df['seq'].apply(find_tandem_repeats_percentage)
        
        mean_trf = np.mean(trf_percentages)
        print(f"--> Mean Tandem Repeat Percentage: {mean_trf:.2f}%")
        
        return mean_trf
    except Exception as e:
        print(f"An error occurred during sequence generation or evaluation: {e}")
        # Return a high value to penalize failing parameter sets
        return 100.0

def main():
    parser = argparse.ArgumentParser(description="Optimize generation parameters using Bayesian Optimization.")
    parser.add_argument("--model_dir", required=True, help="Directory containing the model")
    parser.add_argument("--promptfile", required=True, help="Prompt file in csv or fasta format")
    parser.add_argument("--output_dir", default="outputs/optimization", help="Directory to save optimization results and plots")
    parser.add_argument("--n_calls", type=int, default=50, help="Number of iterations for Bayesian optimization")
    parser.add_argument("--num_seqs_per_call", type=int, default=10, help="Number of sequences to generate for each parameter evaluation")
    parser.add_argument("--min_seq_len", type=int, default=1024, help="Minimum length of generated sequences in tokens")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum length of generated sequences in tokens")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for optimization")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Initialize Sequence Generator ---
    global seq_gen
    # We instantiate it once to avoid reloading the model in every optimization step.
    seq_gen = SequenceGenerator(
        model_dir=args.model_dir,
        promptfile=args.promptfile,
        num=args.num_seqs_per_call,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        top_k=-1, # Keep top_k fixed while optimizing top_p
        seed=args.seed
    )

    # --- Define Search Space ---
    search_space = [
        Real(0.5, 1.5, name='temperature'),
        Real(0.5, 1.0, name='top_p'),
        Real(0.0, 1.0, name='presence_penalty'),
        Real(0.0, 1.0, name='frequency_penalty'),
        Real(1.0, 1.5, name='repetition_penalty'),
    ]

    # Wrapper for the objective function to pass fixed arguments from argparse
    @use_named_args(search_space)
    def objective_wrapper(**params):
        return evaluate_params(
            min_seq_len=args.min_seq_len,
            max_seq_len=args.max_seq_len,
            num=args.num_seqs_per_call,
            max_repeats=100, # Do not filter by repeats during optimization
            **params
        )

    # --- Run Bayesian Optimization ---
    print("Starting Bayesian optimization...")
    result = gp_minimize(
        func=objective_wrapper,
        dimensions=search_space,
        n_calls=args.n_calls,
        random_state=args.seed,
        verbose=True
    )

    # --- Results ---
    print("\nOptimization finished.")
    print(f"Best score (mean TRF %): {result.fun:.2f}%")
    best_parameters = {dim.name: val for dim, val in zip(search_space, result.x)}
    print("Best parameters:")
    for name, value in best_parameters.items():
        print(f"  {name}: {value}")


    # Save results to a file
    with open(os.path.join(args.output_dir, "best_params.txt"), "w") as f:
        f.write(f"Best score (mean TRF %): {result.fun:.2f}%\n")
        f.write("Best parameters found:\n")
        for name, value in best_parameters.items():
            f.write(f"  {name}: {value}\n")

    # --- Visualization ---
    print("Generating plots...")

    # Convergence plot
    plot_convergence(result)
    plt.title("Convergence Plot")
    plt.savefig(os.path.join(args.output_dir, "convergence.png"))
    plt.close()

    # Objective plot
    _ = plot_objective(result, n_points=20)
    plt.savefig(os.path.join(args.output_dir, "objective_plot.png"))
    plt.close()

    print(f"Results and plots saved in {args.output_dir}")

if __name__ == '__main__':
    main()
