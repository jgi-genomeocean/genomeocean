"""
genomeocean.inference
=====================
Unified OOP interface for all GenomeOcean inference modes.

Modes
-----
generate      – de-novo or prompt-conditioned DNA sequence generation
autocomplete  – complete a partial protein-coding gene and score against a
                reference structure (requires foldmason)
embed         – compute mean-pool embedding vectors for a list of sequences
score         – compute per-sequence NLL loss (scalar) or per-token NLL
                loss (vector) for sequences or a whole genome

Typical Python usage
--------------------
>>> from genomeocean.inference import GenomeOceanInference
>>> go = GenomeOceanInference(model_dir="pGenomeOcean/GenomeOcean-4B")
>>> df = go.generate(prompts=["ATGCGT..."], num=10)
>>> emb = go.embed(sequences=["ATGCGT..."])
>>> scores = go.score(sequences=["ATGCGT..."])
"""

from __future__ import annotations

import os
import gzip
import pickle
import textwrap
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Lazy imports: heavy ML stacks are only pulled when actually used.
# ---------------------------------------------------------------------------

__all__ = ["GenomeOceanInference"]


# ---------------------------------------------------------------------------
# Helper: read sequences from disk
# ---------------------------------------------------------------------------

def _load_sequences_from_file(path: str) -> List[str]:
    """Load DNA sequences from .txt (one per line), .fa/.fasta, .csv, or .gz."""
    from Bio import SeqIO

    is_gz = path.endswith(".gz")
    base = path[:-3] if is_gz else path
    ext = os.path.splitext(base)[1].lower()

    open_fn = gzip.open if is_gz else open
    with open_fn(path, "rt") as fh:
        if ext in (".fa", ".fasta", ".fna", ".ffn", ".faa"):
            return [str(r.seq) for r in SeqIO.parse(fh, "fasta")]
        elif ext in (".txt",):
            return [line.strip() for line in fh if line.strip()]
        elif ext in (".csv",):
            import io
            content = fh.read()
            return list(pd.read_csv(io.StringIO(content), header=None)[0])
        else:
            raise ValueError(
                f"Unsupported sequence file extension '{ext}'. "
                "Use .txt, .fa/.fasta, or .csv (optionally .gz compressed)."
            )


def _load_genome_sequences(path: str):
    """Generator: yield (seq_id, seq_str) pairs from FASTA/GenBank/txt."""
    from Bio import SeqIO

    is_gz = path.endswith(".gz")
    base = path[:-3] if is_gz else path
    ext = os.path.splitext(base)[1].lower()

    if ext in (".fa", ".fasta", ".fna", ".ffn"):
        file_format = "fasta"
    elif ext in (".gb", ".gbk", ".genbank"):
        file_format = "genbank"
    elif ext in (".txt",):
        file_format = "txt"
    else:
        raise ValueError(f"Unsupported genome file extension '{ext}'.")

    open_fn = gzip.open if is_gz else open
    with open_fn(path, "rt") as fh:
        if file_format in ("fasta", "genbank"):
            for rec in SeqIO.parse(fh, file_format):
                yield rec.id, str(rec.seq)
        else:
            for i, line in enumerate(fh):
                yield str(i), line.strip()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GenomeOceanInference:
    """Unified interface to all GenomeOcean inference modes.

    Parameters
    ----------
    model_dir : str
        HuggingFace model name or local path
        (e.g. ``"pGenomeOcean/GenomeOcean-4B"``).
    model_max_length : int
        Maximum token length the model accepts (default 10240).
    gpu_memory_utilization : float
        Fraction of GPU VRAM available to vLLM (default 0.6).
    """

    def __init__(
        self,
        model_dir: str,
        model_max_length: int = 10240,
        gpu_memory_utilization: float = 0.6,
    ):
        self.model_dir = model_dir
        self.model_max_length = model_max_length
        self.gpu_memory_utilization = gpu_memory_utilization

        # Cached internal objects – created on first use
        self._llm_utils: Optional[Any] = None
        self._seq_gen: Optional[Any] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_llm_utils(self, model_max_length: Optional[int] = None) -> Any:
        """Return a cached LLMUtils instance."""
        from genomeocean.llm_utils import LLMUtils

        max_len = model_max_length or self.model_max_length
        # Re-create if the requested length differs from cached
        if self._llm_utils is None or self._llm_utils.model_max_length != max_len:
            self._llm_utils = LLMUtils(
                model_dir=self.model_dir,
                model_max_length=max_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
            )
        return self._llm_utils

    def _build_seq_generator(self, prompts: List[str], **kwargs) -> Any:
        """Return a freshly configured SequenceGenerator."""
        from genomeocean.generation import SequenceGenerator

        return SequenceGenerator(
            model_dir=self.model_dir,
            prompts=prompts,
            gpu_memory_utilization=self.gpu_memory_utilization,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # 1. Generate
    # ------------------------------------------------------------------

    def generate(
        self,
        prompts: Optional[List[str]] = None,
        promptfile: Optional[str] = None,
        num: int = 100,
        min_seq_len: int = 1024,
        max_seq_len: int = 10240,
        temperature: float = 1.3,
        top_k: int = -1,
        top_p: float = 0.7,
        presence_penalty: float = 0.5,
        frequency_penalty: float = 0.5,
        repetition_penalty: float = 1.0,
        seed: int = 123,
        max_repeats: int = 0,
        filter_compression: bool = False,
        compression_threshold: float = 1 / 3,
        filter_loss: bool = False,
        loss_threshold: float = 3.5,
        prepend_prompt: bool = True,
    ) -> pd.DataFrame:
        """Generate DNA sequences.

        For sequences longer than 10 240 tokens the method automatically
        chains multiple 10 240-token generation passes (continuous-generation
        mode).

        Parameters
        ----------
        prompts : list of str, optional
            DNA sequences used as generation prompts.  Provide either
            *prompts* or *promptfile*, not both.
        promptfile : str, optional
            Path to a .txt, .fa, or .csv file of prompts.
        num : int
            Number of sequences to generate from each prompt.
        min_seq_len, max_seq_len : int
            Token-length range for generated sequences.  Multiply by ~4 to
            get approximate base-pair length.
        temperature : float
            Sampling temperature (higher → more random).
        top_k, top_p : int, float
            Top-k / nucleus sampling parameters.
        presence_penalty, frequency_penalty, repetition_penalty : float
            vLLM sampling penalties.
        seed : int
            Random seed.
        max_repeats : int
            Discard sequences with more than this percentage of simple
            tandem repeats (0 = keep all).
        filter_compression : bool
            Discard low-information sequences via gzip compression ratio.
        compression_threshold : float
            Minimum compression ratio to keep a sequence.
        filter_loss : bool
            Discard sequences with mean NLL loss below *loss_threshold*.
        loss_threshold : float
            Minimum mean NLL loss to keep a sequence.
        prepend_prompt : bool
            If True, prepend the original prompt to each generated sequence
            in the output DataFrame.

        Returns
        -------
        pd.DataFrame with columns ``seq`` and ``id``.
        """
        if prompts is None and promptfile is None:
            # De-novo generation: use a single empty-string prompt
            prompts = [""]
        if promptfile is not None and prompts is None:
            prompts = _load_sequences_from_file(promptfile)

        if max_seq_len > 10240:
            return self._generate_long(
                prompts=prompts,
                num=num,
                max_seq_len=max_seq_len,
                max_repeats=max_repeats,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repetition_penalty=repetition_penalty,
                seed=seed,
            )

        sg = self._build_seq_generator(
            prompts=prompts,
            num=num,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
            seed=seed,
            filter_compression=filter_compression,
            compression_threshold=compression_threshold,
            filter_loss=filter_loss,
            loss_threshold=loss_threshold,
        )
        return sg.generate_sequences(
            prepend_prompt_to_output=prepend_prompt,
            max_repeats=max_repeats,
        )

    def _generate_long(
        self,
        prompts: List[str],
        num: int,
        max_seq_len: int,
        max_repeats: int,
        **sampling_kwargs,
    ) -> pd.DataFrame:
        """Chain multiple generation passes to exceed 10 240 token limit."""
        from genomeocean.dnautils import find_tandem_repeats_percentage

        sg = self._build_seq_generator(
            prompts=prompts,
            num=num,
            min_seq_len=10230,
            max_seq_len=10240,
            **sampling_kwargs,
        )
        g_seqs = sg.generate_sequences(prepend_prompt_to_output=True)
        del sg

        # Extend in 10 240-token windows
        tmp_file = "prompts_c.csv"
        for _ in range(int(max_seq_len / 10240)):
            g_seqs["seq"].apply(lambda x: x[-10000:]).to_csv(
                tmp_file, index=False, header=None
            )
            sg = self._build_seq_generator(
                prompts=None,  # will be overridden below
                num=1,
                min_seq_len=10230,
                max_seq_len=10240,
                **sampling_kwargs,
            )
            # Re-build using promptfile
            from genomeocean.generation import SequenceGenerator

            sg2 = SequenceGenerator(
                model_dir=self.model_dir,
                promptfile=tmp_file,
                num=1,
                min_seq_len=10230,
                max_seq_len=10240,
                gpu_memory_utilization=self.gpu_memory_utilization,
                **sampling_kwargs,
            )
            c_seqs = sg2.generate_sequences(prepend_prompt_to_output=False)
            del sg2
            g_seqs["seq"] = g_seqs["seq"] + c_seqs["seq"]

        if os.path.exists(tmp_file):
            os.remove(tmp_file)

        if 0 < max_repeats < 100:
            g_seqs["repeats"] = g_seqs["seq"].apply(find_tandem_repeats_percentage)
            g_seqs = g_seqs[g_seqs["repeats"] < max_repeats]

        return g_seqs

    @staticmethod
    def save_sequences(
        df: pd.DataFrame,
        out_prefix: str = "generated",
        out_format: str = "fa",
    ) -> None:
        """Write a generation DataFrame to a FASTA or plain-text file.

        Parameters
        ----------
        df : pd.DataFrame
            Output of :meth:`generate` or :meth:`autocomplete`.
        out_prefix : str
            Path prefix; the appropriate extension will be appended.
        out_format : str
            ``"fa"`` for FASTA or ``"txt"`` for one-sequence-per-line.
        """
        from genomeocean.generation import SequenceGenerator

        # SequenceGenerator.save_sequences is a method on an instance; call
        # it on a minimal instance created only for its save helper.
        # Alternatively, replicate the logic here to avoid the overhead.
        assert out_format in ("fa", "txt"), "out_format must be 'fa' or 'txt'"
        out_prefix = out_prefix.rstrip("/")
        if "/" in out_prefix:
            out_dir = "/".join(out_prefix.split("/")[:-1])
            os.makedirs(out_dir, exist_ok=True)

        if out_format == "txt":
            with open(f"{out_prefix}.txt", "w") as fh:
                for _, row in df.iterrows():
                    fh.write(row["seq"] + "\n")
            print(f"Saved {len(df)} sequences → {out_prefix}.txt")
        else:
            with open(f"{out_prefix}.fa", "w") as fh:
                for i, (_, row) in enumerate(df.iterrows()):
                    fh.write(f">{row['id']}_{i}\n")
                    fh.write("\n".join(textwrap.wrap(row["seq"], 80)) + "\n")
            print(f"Saved {len(df)} sequences → {out_prefix}.fa")

    # ------------------------------------------------------------------
    # 2. Autocomplete
    # ------------------------------------------------------------------

    def autocomplete(
        self,
        gen_id: str,
        start: int,
        end: int,
        prompt_start: int = 0,
        prompt_end: int = 0,
        mutate_prompt: bool = False,
        strand: int = 1,
        backward: bool = False,
        ref_pdb: str = "",
        structure_start: int = 0,
        structure_end: int = 0,
        foldmason_path: str = "",
        num: int = 200,
        min_seq_len: int = 250,
        max_seq_len: int = 300,
        max_repeats: int = 0,
        filter_compression: bool = False,
        compression_threshold: float = 0.33,
        filter_loss: bool = False,
        loss_threshold: float = 3.5,
    ) -> pd.DataFrame:
        """Autocomplete a partial protein-coding gene and score its structure.

        Retrieves the gene sequence from NCBI (using *gen_id*, *start*, *end*),
        generates continuations conditioned on the supplied prompt region, then
        optionally evaluates structural similarity to the reference using
        FoldMason + ESMFold (via the ESMFold API).

        Parameters
        ----------
        gen_id : str
            NCBI accession or contig ID (e.g. ``"NZ_JAYXHC010000003.1"``).
        start, end : int
            0-based start / end positions (bp) of the gene on the contig.
        prompt_start, prompt_end : int
            Slice of the gene (in bp) to use as generation prompt.
        mutate_prompt : bool
            If True, also generate from synonymous / non-synonymous mutants
            of the prompt region.
        strand : int
            ``1`` for forward, ``-1`` for reverse-complement.
        backward : bool
            Generate in the 3′→5′ direction (special mode).
        ref_pdb : str
            Path to a reference PDB file.  If empty, the reference structure
            is predicted from the gene translation via the ESMFold API.
        structure_start, structure_end : int
            AA indices of the structure region to score.
        foldmason_path : str
            Path to the FoldMason binary.
        num : int
            Number of sequences to generate.
        min_seq_len, max_seq_len : int
            Token-length range for the generated region.
        max_repeats : int
            Discard sequences exceeding this tandem-repeat %.
        filter_compression, compression_threshold : bool, float
            Gzip-ratio quality filter.
        filter_loss, loss_threshold : bool, float
            NLL loss quality filter.

        Returns
        -------
        pd.DataFrame with columns ``seq``, ``protein``, ``orf_len``,
        ``length``, and (if foldmason_path is provided) ``lddt_score``.
        """
        from genomeocean.dnautils import (
            get_nuc_seq_by_id,
            introduce_mutations,
            fasta2pdb_api,
            reverse_complement,
            LDDT_scoring,
        )
        from genomeocean.generation import SequenceGenerator
        from Bio.Seq import Seq, translate

        gene = get_nuc_seq_by_id(gen_id, start=start, end=end)
        if gene is None:
            raise RuntimeError(
                f"Failed to retrieve gene sequence {gen_id} [{start}:{end}]"
            )

        if strand == -1:
            gene = reverse_complement(gene)

        # Optionally cap structure window at 400 aa (API limit)
        if structure_end - structure_start > 400:
            structure_end = structure_start + 400
            print(
                "Structure prediction limited to 400 aa; "
                f"structure_end adjusted to {structure_end}."
            )

        # Predict / load reference structure
        if ref_pdb == "":
            ref_pdb = "ref_tmp.pdb"
            if os.path.exists(ref_pdb):
                os.remove(ref_pdb)
            fasta2pdb_api(
                translate(gene, to_stop=True)[structure_start:structure_end],
                ref_pdb,
            )

        if backward:
            gene = reverse_complement(gene)

        # Build prompts
        prompts: List[str] = [gene[prompt_start:prompt_end]]
        if mutate_prompt:
            orf_prompt = gene[prompt_start:prompt_end]
            for rate in range(10, 50, 10):
                prompts.append(
                    introduce_mutations(orf_prompt, mutation_percentage=rate,
                                       mutation_type="synonymous")
                )
            for rate in range(10, 50, 10):
                prompts.append(
                    introduce_mutations(orf_prompt, mutation_percentage=rate,
                                       mutation_type="nonsynonymous")
                )

        tmp_prompts = "tmp_prompts.csv"
        pd.DataFrame(prompts).to_csv(tmp_prompts, sep="\t", header=None, index=False)

        sg = SequenceGenerator(
            model_dir=self.model_dir,
            promptfile=tmp_prompts,
            num=num,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            temperature=1.0,
            presence_penalty=0.5,
            frequency_penalty=0.5,
            repetition_penalty=1.0,
            seed=1234,
            gpu_memory_utilization=self.gpu_memory_utilization,
            filter_compression=filter_compression,
            compression_threshold=compression_threshold,
            filter_loss=filter_loss,
            loss_threshold=loss_threshold,
        )
        g_seqs = sg.generate_sequences(
            prepend_prompt_to_output=True, max_repeats=max_repeats
        )
        print(f"Total {len(g_seqs)} sequences generated.")
        if os.path.exists(tmp_prompts):
            os.remove(tmp_prompts)

        if backward:
            g_seqs["seq"] = g_seqs["seq"].apply(reverse_complement)

        # Find largest ORF per sequence
        def _largest_orf(seq: str) -> str:
            s = Seq(seq)
            orfs: List[str] = []
            for frame in range(3):
                for i in range(frame, len(s), 3):
                    if str(s[i : i + 3]) in ("ATG", "GTG"):
                        orfs.append(str(s[i:].translate(to_stop=True)))
                orfs.append(str(s[frame:].translate(to_stop=True)))
            return max(orfs, key=len)

        g_seqs["protein"] = g_seqs["seq"].apply(_largest_orf)
        g_seqs["orf_len"] = g_seqs["protein"].apply(lambda x: 3 * len(x))
        g_seqs["length"] = g_seqs["seq"].apply(len)
        g_seqs = g_seqs[g_seqs["orf_len"] >= len(gene) - 100].copy()
        print(f"Total {len(g_seqs)} sequences have ORF ≥ len(gene) - 100.")

        # Structure scoring (optional)
        if foldmason_path:
            g_seqs["lddt_score"] = g_seqs["protein"].apply(
                lambda x: LDDT_scoring(
                    x[structure_start:structure_end],
                    ref_pdb,
                    foldmason_path=foldmason_path,
                )
            )

        if os.path.exists("ref_tmp.pdb"):
            os.remove("ref_tmp.pdb")

        return g_seqs

    # ------------------------------------------------------------------
    # 3. Embed
    # ------------------------------------------------------------------

    def embed(
        self,
        sequences: Optional[List[str]] = None,
        sequence_file: Optional[str] = None,
        model_max_length: int = 1024,
        batch_size: int = 50,
    ) -> np.ndarray:
        """Compute mean-pool embedding vectors for DNA sequences.

        Parameters
        ----------
        sequences : list of str, optional
            Raw DNA sequences.  Provide either *sequences* or
            *sequence_file*, not both.
        sequence_file : str, optional
            Path to a file of sequences (one per line, or FASTA).
        model_max_length : int
            Token context length.  Set as (bp length / 4).  Max 10 240.
        batch_size : int
            Number of sequences per GPU batch.

        Returns
        -------
        np.ndarray of shape ``(N, hidden_size)``.
        """
        assert model_max_length <= 10240, (
            f"model_max_length {model_max_length} exceeds model maximum 10240"
        )

        if sequences is None and sequence_file is None:
            raise ValueError("Provide either 'sequences' or 'sequence_file'.")
        if sequence_file is not None and sequences is None:
            sequences = _load_sequences_from_file(sequence_file)

        llm = self._get_llm_utils(model_max_length=model_max_length)
        print(
            f"Embedding {len(sequences)} sequences "
            f"(max_length={model_max_length}, batch_size={batch_size})"
        )
        embeddings = llm.predict(sequences, batch_size=batch_size, do_embedding=True)
        return embeddings

    # ------------------------------------------------------------------
    # 4. Score (loss)
    # ------------------------------------------------------------------

    def score(
        self,
        sequences: Optional[List[str]] = None,
        sequence_file: Optional[str] = None,
        genome_file: Optional[str] = None,
        mode: str = "sequence",
        use_perplexity: bool = False,
        use_token_scores: bool = False,
        segment_size: int = 50000,
        overlap_size: int = 5000,
        use_reverse: bool = False,
        batch_size: int = 8,
        out_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute NLL loss scores for sequences or a whole genome.

        Two operating modes are supported:

        **sequence mode** (``mode="sequence"``)
            Each sequence is scored independently.  Returns a dict
            ``{seq_id: score}`` where *score* is:

            * a scalar ``float`` (mean NLL loss) when
              ``use_token_scores=False`` and ``use_perplexity=False``, or
            * perplexity (``exp(mean_NLL)``) when ``use_perplexity=True``, or
            * a 1-D ``np.ndarray`` of per-token NLL values when
              ``use_token_scores=True``.

        **genome mode** (``mode="genome"``)
            The genome is split into overlapping segments, scored, and
            re-assembled.  Returns a dict ``{chrom_id: per_base_scores}``.
            Optionally writes a pickle file if *out_prefix* is provided.

        Parameters
        ----------
        sequences : list of str, optional
        sequence_file : str, optional
            Path to sequences (txt / fa / csv).
        genome_file : str, optional
            Path to genome FASTA / FASTA.gz (for genome mode).
        mode : {"sequence", "genome"}
        use_perplexity : bool
            Return exp(mean_NLL) instead of mean_NLL (sequence mode only).
        use_token_scores : bool
            Return per-token NLL arrays instead of scalars (sequence mode).
        segment_size : int
            Segment length (bp) for genome scanning.
        overlap_size : int
            Overlap between consecutive segments (bp).
        use_reverse : bool
            Also score the reverse-complement strand and take the minimum
            per base (genome mode).
        batch_size : int
            Batch size for batched loss computation.
        out_prefix : str, optional
            If provided, write results to ``{out_prefix}.pkl``.

        Returns
        -------
        dict mapping sequence / chromosome id → score(s).
        """
        if mode == "sequence":
            return self._score_sequences(
                sequences=sequences,
                sequence_file=sequence_file,
                use_perplexity=use_perplexity,
                use_token_scores=use_token_scores,
                out_prefix=out_prefix,
            )
        elif mode == "genome":
            return self._score_genome(
                genome_file=genome_file,
                segment_size=segment_size,
                overlap_size=overlap_size,
                use_reverse=use_reverse,
                out_prefix=out_prefix,
            )
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'sequence' or 'genome'.")

    def _score_sequences(
        self,
        sequences: Optional[List[str]],
        sequence_file: Optional[str],
        use_perplexity: bool,
        use_token_scores: bool,
        out_prefix: Optional[str],
    ) -> Dict[str, Any]:
        if sequences is None and sequence_file is None:
            raise ValueError("Provide 'sequences' or 'sequence_file'.")
        if sequence_file is not None and sequences is None:
            sequences = _load_sequences_from_file(sequence_file)

        if not sequences:
            raise ValueError("No sequences found in the input file.")

        llm = self._get_llm_utils()

        if use_token_scores:
            raw = llm.compute_token_perplexity(sequences)
            results = {str(i): np.array(s, dtype=np.float32) for i, s in enumerate(raw)}
        else:
            raw = llm.compute_sequence_perplexity(
                sequences, use_ppl=use_perplexity
            )
            results = {str(i): float(s) for i, s in enumerate(raw)}

        if out_prefix:
            _pkl_path = f"{out_prefix}.pkl"
            with open(_pkl_path, "wb") as fh:
                pickle.dump(results, fh)
            print(f"Scores saved → {_pkl_path}")

        return results

    def _score_genome(
        self,
        genome_file: Optional[str],
        segment_size: int,
        overlap_size: int,
        use_reverse: bool,
        out_prefix: Optional[str],
    ) -> Dict[str, np.ndarray]:
        if genome_file is None:
            raise ValueError("'genome_file' is required for genome mode.")

        from genomeocean.dnautils import reverse_complement

        llm = self._get_llm_utils()

        def _split(seq: str):
            step = segment_size - overlap_size
            for i in range(0, len(seq), step):
                yield seq[i : i + segment_size]

        results: Dict[str, np.ndarray] = {}
        for seq_id, sequence in _load_genome_sequences(genome_file):
            print(f"Scanning forward strand: {seq_id}")
            fwd_segs = list(_split(sequence))
            fwd_raw = llm.compute_token_perplexity(fwd_segs)

            # Stitch segments
            stitched_fwd = list(fwd_raw[0])
            for seg in fwd_raw[1:]:
                stitched_fwd.extend(seg[overlap_size:])
            fwd_arr = np.array(stitched_fwd, dtype=np.float16)

            if use_reverse:
                print(f"Scanning reverse strand: {seq_id}")
                rc_seq = reverse_complement(sequence)
                rev_segs = list(_split(rc_seq))
                rev_raw = llm.compute_token_perplexity(rev_segs)
                stitched_rev = list(rev_raw[0])
                for seg in rev_raw[1:]:
                    stitched_rev.extend(seg[overlap_size:])
                rev_arr = np.array(stitched_rev, dtype=np.float16)
                combined = np.minimum(fwd_arr, rev_arr)
            else:
                combined = fwd_arr

            results[seq_id] = combined

        if out_prefix:
            _pkl_path = f"{out_prefix}.pkl"
            with open(_pkl_path, "wb") as fh:
                pickle.dump(results, fh)
            print(f"Genome scores saved → {_pkl_path}")

        return results
