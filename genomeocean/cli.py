#!/usr/bin/env python
"""
go-infer — Unified GenomeOcean Inference CLI
=============================================

Sub-commands
------------
  generate      De-novo or prompt-conditioned DNA sequence generation
  autocomplete  Complete a partial protein-coding gene (+ optional structure scoring)
  embed         Compute embedding vectors for a list of sequences
  score         Compute NLL loss for sequences or a whole-genome scan

Quick examples
--------------
# generate 10 sequences from prompts in a file
go-infer generate \\
    --model_dir pGenomeOcean/GenomeOcean-4B \\
    --promptfile my_prompts.fa \\
    --num 10 --out_prefix outputs/gen

# embed sequences, save as .npy
go-infer embed \\
    --model_dir pGenomeOcean/GenomeOcean-4B \\
    --sequence_file my_seqs.txt \\
    --model_max_length 256 \\
    --out_file outputs/emb.npy

# per-sequence NLL loss
go-infer score \\
    --model_dir pGenomeOcean/GenomeOcean-4B \\
    --sequence_file my_seqs.txt \\
    --out_prefix outputs/scores

# genome-wide scan (produces outputs/scan.pkl)
go-infer score \\
    --model_dir pGenomeOcean/GenomeOcean-4B \\
    --genome_file my_genome.fa.gz \\
    --mode genome \\
    --use_reverse \\
    --out_prefix outputs/scan
"""

import argparse
import os
import sys


# ---------------------------------------------------------------------------
# Sub-command: generate
# ---------------------------------------------------------------------------

def _cmd_generate(args):
    from genomeocean.inference import GenomeOceanInference
    import numpy as np

    go = GenomeOceanInference(
        model_dir=args.model_dir,
        model_max_length=args.max_seq_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    df = go.generate(
        prompts=None if args.promptfile else ([""] if args.prompts is None else args.prompts),
        promptfile=args.promptfile,
        num=args.num,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        max_repeats=args.max_repeats,
        filter_compression=args.filter_compression,
        compression_threshold=args.compression_threshold,
        filter_loss=args.filter_loss,
        loss_threshold=args.loss_threshold,
        prepend_prompt=not args.no_prepend_prompt,
    )

    go.save_sequences(df, out_prefix=args.out_prefix, out_format=args.out_format)

    if args.sort_by_orf_length:
        import pyrodigal
        from Bio import SeqIO
        import textwrap

        orf_finder = pyrodigal.GeneFinder(meta=True)
        all_orfs = {}
        for r in SeqIO.parse(f"{args.out_prefix}.fa", "fasta"):
            for i, pred in enumerate(orf_finder.find_genes(bytes(str(r.seq), encoding="utf8"))):
                seq = pred.translate().upper()[:-1]
                if len(seq) >= 100:
                    all_orfs[f"{r.id}_{i}"] = seq
        with open(f"{args.out_prefix}.faa", "w") as fh:
            for sid, seq in sorted(all_orfs.items(), key=lambda x: len(x[1]), reverse=True):
                fh.write(f">{sid}\n")
                fh.write("\n".join(textwrap.wrap(seq, 80)) + "\n")
        print(f"ORF-sorted protein sequences → {args.out_prefix}.faa")


# ---------------------------------------------------------------------------
# Sub-command: autocomplete
# ---------------------------------------------------------------------------

def _cmd_autocomplete(args):
    from genomeocean.inference import GenomeOceanInference

    go = GenomeOceanInference(
        model_dir=args.model_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    df = go.autocomplete(
        gen_id=args.gen_id,
        start=args.start,
        end=args.end,
        prompt_start=args.prompt_start,
        prompt_end=args.prompt_end,
        mutate_prompt=args.mutate_prompt,
        strand=args.strand,
        backward=(args.direction == -1),
        ref_pdb=args.ref_pdb,
        structure_start=args.structure_start,
        structure_end=args.structure_end,
        foldmason_path=args.foldmason_path,
        num=args.num,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        max_repeats=args.max_repeats,
        filter_compression=args.filter_compression,
        compression_threshold=args.compression_threshold,
        filter_loss=args.filter_loss,
        loss_threshold=args.loss_threshold,
    )

    out_csv = args.out_prefix + ".csv"
    df.to_csv(out_csv, sep="\t", index=False)
    print(f"Autocomplete results → {out_csv}")


# ---------------------------------------------------------------------------
# Sub-command: embed
# ---------------------------------------------------------------------------

def _cmd_embed(args):
    import numpy as np
    from genomeocean.inference import GenomeOceanInference

    go = GenomeOceanInference(
        model_dir=args.model_dir,
        model_max_length=args.model_max_length,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    embeddings = go.embed(
        sequences=None,
        sequence_file=args.sequence_file,
        model_max_length=args.model_max_length,
        batch_size=args.batch_size,
    )

    out_path = args.out_prefix if args.out_prefix.endswith(".npy") else args.out_prefix + ".npy"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.save(out_path, embeddings)
    print(f"Embeddings {embeddings.shape} → {out_path}")


# ---------------------------------------------------------------------------
# Sub-command: score
# ---------------------------------------------------------------------------

def _cmd_score(args):
    from genomeocean.inference import GenomeOceanInference

    go = GenomeOceanInference(
        model_dir=args.model_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    results = go.score(
        sequences=None,
        sequence_file=args.sequence_file if hasattr(args, "sequence_file") else None,
        genome_file=args.genome_file if hasattr(args, "genome_file") else None,
        mode=args.mode,
        use_perplexity=args.use_perplexity,
        use_token_scores=args.use_token_scores,
        segment_size=args.segment_size,
        overlap_size=args.overlap_size,
        use_reverse=args.use_reverse,
        out_prefix=args.out_prefix,
    )

    if args.out_prefix is None:
        # Print summary to stdout when no output prefix given
        for k, v in results.items():
            try:
                import numpy as np
                mean_val = float(np.mean(v))
            except Exception:
                mean_val = v
            print(f"{k}\t{mean_val:.4f}")


# ---------------------------------------------------------------------------
# Argument parsers
# ---------------------------------------------------------------------------

def _parser_generate(sub):
    p = sub.add_parser(
        "generate",
        help="Generate DNA sequences (de-novo or prompt-conditioned).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_dir", required=True, help="Model name or path")
    p.add_argument("--promptfile", default=None,
                   help="Prompt file (.txt / .fa / .csv). Omit for de-novo generation.")
    p.add_argument("--prompts", nargs="+", default=None,
                   help="One or more inline prompt sequences (DNA strings). "
                        "Ignored when --promptfile is given.")
    p.add_argument("--out_prefix", default="outputs/generated", help="Output file prefix")
    p.add_argument("--out_format", default="fa", choices=["fa", "txt"],
                   help="Output format")
    p.add_argument("--num", type=int, default=100,
                   help="Sequences to generate per prompt")
    p.add_argument("--min_seq_len", type=int, default=1024,
                   help="Min token length (≈ bp/4)")
    p.add_argument("--max_seq_len", type=int, default=10240,
                   help="Max token length (max 10240; larger triggers chained mode)")
    p.add_argument("--temperature", type=float, default=1.3)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--top_p", type=float, default=0.7)
    p.add_argument("--presence_penalty", type=float, default=0.5)
    p.add_argument("--frequency_penalty", type=float, default=0.5)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max_repeats", type=int, default=0,
                   help="Discard sequences with >N%% tandem repeats (0=keep all)")
    p.add_argument("--filter_compression", action="store_true",
                   help="Filter low-complexity sequences by gzip ratio")
    p.add_argument("--compression_threshold", type=float, default=1/3)
    p.add_argument("--filter_loss", action="store_true",
                   help="Filter sequences below --loss_threshold")
    p.add_argument("--loss_threshold", type=float, default=3.5)
    p.add_argument("--no_prepend_prompt", action="store_true",
                   help="Do not prepend the prompt to the output sequences")
    p.add_argument("--sort_by_orf_length", action="store_true",
                   help="Also write an ORF-sorted .faa file (requires pyrodigal)")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    p.set_defaults(func=_cmd_generate)
    return p


def _parser_autocomplete(sub):
    p = sub.add_parser(
        "autocomplete",
        help="Complete a partial protein-coding gene; optionally score structure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_dir", required=True)
    p.add_argument("--gen_id", required=True,
                   help="NCBI accession / contig ID")
    p.add_argument("--start", type=int, required=True,
                   help="Gene start position (bp, 0-based)")
    p.add_argument("--end", type=int, required=True,
                   help="Gene end position (bp, 0-based)")
    p.add_argument("--prompt_start", type=int, default=0)
    p.add_argument("--prompt_end", type=int, default=0)
    p.add_argument("--mutate_prompt", action="store_true",
                   help="Also generate from synonymous / non-synonymous mutants")
    p.add_argument("--strand", type=int, default=1, choices=[1, -1])
    p.add_argument("--direction", type=int, default=1, choices=[1, -1],
                   help="-1 to generate in the reverse (3′→5′) direction")
    p.add_argument("--ref_pdb", default="",
                   help="Reference PDB (auto-predicted via ESMFold if omitted)")
    p.add_argument("--structure_start", type=int, default=0)
    p.add_argument("--structure_end", type=int, default=0)
    p.add_argument("--foldmason_path", default="",
                   help="Path to foldmason binary (structure scoring skipped if empty)")
    p.add_argument("--num", type=int, default=200)
    p.add_argument("--min_seq_len", type=int, default=250)
    p.add_argument("--max_seq_len", type=int, default=300)
    p.add_argument("--max_repeats", type=int, default=0)
    p.add_argument("--filter_compression", action="store_true")
    p.add_argument("--compression_threshold", type=float, default=0.33)
    p.add_argument("--filter_loss", action="store_true")
    p.add_argument("--loss_threshold", type=float, default=3.5)
    p.add_argument("--out_prefix", default="outputs/autocomplete")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    p.set_defaults(func=_cmd_autocomplete)
    return p


def _parser_embed(sub):
    p = sub.add_parser(
        "embed",
        help="Compute mean-pool embedding vectors for DNA sequences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_dir", required=True)
    p.add_argument("--sequence_file", required=True,
                   help="Path to sequences (.txt / .fa / .csv)")
    p.add_argument("--model_max_length", type=int, default=256,
                   help="Token context length (set to bp_length / 4, max 10240)")
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--out_prefix", default="outputs/embeddings",
                   help="Output prefix (.npy extension added automatically)")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    p.set_defaults(func=_cmd_embed)
    return p


def _parser_score(sub):
    p = sub.add_parser(
        "score",
        help="Compute NLL loss for sequences or a whole-genome scan.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_dir", required=True)

    # Input — mutually exclusive groups
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--sequence_file",
                     help="Path to sequences (.txt / .fa / .csv) — use with --mode sequence")
    inp.add_argument("--genome_file",
                     help="Path to genome FASTA / FASTA.gz — use with --mode genome")

    p.add_argument("--mode", default="sequence", choices=["sequence", "genome"],
                   help="'sequence': per-sequence loss; 'genome': whole-genome scan")
    p.add_argument("--use_perplexity", action="store_true",
                   help="Return perplexity (exp(NLL)) instead of NLL [sequence mode]")
    p.add_argument("--use_token_scores", action="store_true",
                   help="Return per-token loss arrays instead of scalars [sequence mode]")
    p.add_argument("--segment_size", type=int, default=50000,
                   help="Segment size in bp [genome mode]")
    p.add_argument("--overlap_size", type=int, default=5000,
                   help="Overlap between segments in bp [genome mode]")
    p.add_argument("--use_reverse", action="store_true",
                   help="Also score reverse-complement strand [genome mode]")
    p.add_argument("--out_prefix", default=None,
                   help="Output prefix; results saved as <prefix>.pkl. "
                        "Omit to print summary to stdout.")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    p.set_defaults(func=_cmd_score)
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="go-infer",
        description=(
            "GenomeOcean Unified Inference CLI\n"
            "Supports sequence generation, gene autocomplete,\n"
            "embedding, and NLL loss scoring."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    _parser_generate(sub)
    _parser_autocomplete(sub)
    _parser_embed(sub)
    _parser_score(sub)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
