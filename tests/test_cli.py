"""
tests/test_cli.py
=================
CPU-safe unit tests for the ``go-infer`` CLI.

These tests exercise argument parsing and edge-case validation only.
No model weights are loaded and no GPU is required, so they run on
standard GitHub Actions ubuntu-latest runners.
"""

import subprocess
import sys
import os
import tempfile
import pytest
import argparse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_cli(*args, **kwargs):
    """Run ``go-infer <args>`` as a subprocess and return CompletedProcess."""
    return subprocess.run(
        [sys.executable, "-m", "genomeocean.cli", *args],
        capture_output=True,
        text=True,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. go-infer --help
# ---------------------------------------------------------------------------

class TestHelp:
    """``go-infer --help`` must exit 0 and mention all sub-commands."""

    def test_help_exit_code(self):
        result = run_cli("--help")
        assert result.returncode == 0, (
            f"Expected exit 0 from --help, got {result.returncode}.\n"
            f"stderr: {result.stderr}"
        )

    def test_help_lists_subcommands(self):
        result = run_cli("--help")
        output = result.stdout + result.stderr
        for cmd in ("generate", "autocomplete", "embed", "score"):
            assert cmd in output, f"Sub-command '{cmd}' not mentioned in --help output."

    def test_subcommand_help_generate(self):
        result = run_cli("generate", "--help")
        assert result.returncode == 0

    def test_subcommand_help_score(self):
        result = run_cli("score", "--help")
        assert result.returncode == 0

    def test_subcommand_help_embed(self):
        result = run_cli("embed", "--help")
        assert result.returncode == 0

    def test_subcommand_help_autocomplete(self):
        result = run_cli("autocomplete", "--help")
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# 2. Parser unit tests (import-level, no subprocess overhead)
# ---------------------------------------------------------------------------

class TestParserDirect:
    """Test the argparse layer directly without spawning a subprocess.

    We import cli.py and exercise its parsers in-process so we can verify
    that ``args.<attr>`` is set correctly after parsing.
    """

    @pytest.fixture(autouse=True)
    def _import_cli(self):
        """Import cli module; skip whole class if it fails (e.g. missing dep)."""
        try:
            from genomeocean import cli as _cli
            self.cli = _cli
        except ImportError as exc:
            pytest.skip(f"Could not import genomeocean.cli: {exc}")

    def _build_parser(self):
        """Re-create the top-level parser (mirrors cli.main())."""
        parser = argparse.ArgumentParser(prog="go-infer")
        sub = parser.add_subparsers(dest="command", metavar="COMMAND")
        sub.required = True
        self.cli._parser_generate(sub)
        self.cli._parser_autocomplete(sub)
        self.cli._parser_embed(sub)
        self.cli._parser_score(sub)
        return parser

    # -- generate --

    def test_generate_prompts_attribute_exists(self):
        """Regression: --prompts used to cause AttributeError in _cmd_generate."""
        parser = self._build_parser()
        args = parser.parse_args([
            "generate",
            "--model_dir", "fake/model",
            "--prompts", "ATCG", "GCTA",
        ])
        # args.prompts must be a list, not missing
        assert hasattr(args, "prompts"), "args.prompts attribute missing after parse"
        assert args.prompts == ["ATCG", "GCTA"]

    def test_generate_prompts_default_is_none(self):
        """When --prompts is omitted, args.prompts should be None."""
        parser = self._build_parser()
        args = parser.parse_args([
            "generate",
            "--model_dir", "fake/model",
        ])
        assert args.prompts is None

    def test_generate_out_prefix_default(self):
        parser = self._build_parser()
        args = parser.parse_args(["generate", "--model_dir", "m"])
        assert args.out_prefix == "outputs/generated"

    def test_generate_out_prefix_custom(self):
        parser = self._build_parser()
        args = parser.parse_args([
            "generate", "--model_dir", "m", "--out_prefix", "my/custom/prefix"
        ])
        assert args.out_prefix == "my/custom/prefix"

    def test_generate_no_prepend_prompt_flag(self):
        parser = self._build_parser()
        args = parser.parse_args([
            "generate", "--model_dir", "m", "--no_prepend_prompt"
        ])
        assert args.no_prepend_prompt is True

    def test_generate_promptfile_sets_attribute(self):
        parser = self._build_parser()
        args = parser.parse_args([
            "generate", "--model_dir", "m", "--promptfile", "seqs.fa"
        ])
        assert args.promptfile == "seqs.fa"

    # -- score --

    def test_score_out_prefix_default_none(self):
        parser = self._build_parser()
        args = parser.parse_args([
            "score", "--model_dir", "m", "--sequence_file", "seqs.txt"
        ])
        assert args.out_prefix is None

    def test_score_out_prefix_custom(self):
        parser = self._build_parser()
        args = parser.parse_args([
            "score", "--model_dir", "m",
            "--sequence_file", "seqs.txt",
            "--out_prefix", "results/scores",
        ])
        assert args.out_prefix == "results/scores"

    def test_score_sequence_file_sets_attribute(self):
        parser = self._build_parser()
        args = parser.parse_args([
            "score", "--model_dir", "m", "--sequence_file", "path/to/seqs.fa"
        ])
        assert args.sequence_file == "path/to/seqs.fa"
        assert not hasattr(args, "genome_file") or args.genome_file is None

    def test_score_genome_file_sets_attribute(self):
        parser = self._build_parser()
        args = parser.parse_args([
            "score", "--model_dir", "m",
            "--genome_file", "genome.fa", "--mode", "genome"
        ])
        assert args.genome_file == "genome.fa"
        assert args.mode == "genome"

    def test_score_mutually_exclusive_inputs_rejects_both(self):
        """Providing both --sequence_file and --genome_file must fail."""
        parser = self._build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([
                "score", "--model_dir", "m",
                "--sequence_file", "s.txt",
                "--genome_file", "g.fa",
            ])
        assert exc_info.value.code != 0

    # -- embed --

    def test_embed_out_prefix_default(self):
        parser = self._build_parser()
        args = parser.parse_args([
            "embed", "--model_dir", "m", "--sequence_file", "s.txt"
        ])
        assert args.out_prefix == "outputs/embeddings"

    def test_embed_out_prefix_custom(self):
        parser = self._build_parser()
        args = parser.parse_args([
            "embed", "--model_dir", "m",
            "--sequence_file", "s.txt",
            "--out_prefix", "emb/run1",
        ])
        assert args.out_prefix == "emb/run1"

    def test_func_attribute_set_for_each_subcommand(self):
        """Every sub-command must bind args.func so the dispatch works."""
        from genomeocean.cli import (
            _cmd_generate, _cmd_autocomplete, _cmd_embed, _cmd_score
        )
        parser = self._build_parser()
        cases = [
            (["generate", "--model_dir", "m"], _cmd_generate),
            (["embed", "--model_dir", "m", "--sequence_file", "s.txt"], _cmd_embed),
            (["score", "--model_dir", "m", "--sequence_file", "s.txt"], _cmd_score),
        ]
        for argv, expected_func in cases:
            args = parser.parse_args(argv)
            assert args.func is expected_func, (
                f"For argv={argv!r}: expected func={expected_func.__name__}, "
                f"got {args.func.__name__}"
            )


# ---------------------------------------------------------------------------
# 3. score with an empty sequence file → ValueError
# ---------------------------------------------------------------------------

class TestScoreEmptyFile:
    """
    Regression test: passing an empty sequence file to ``go.score()`` must
    raise ``ValueError('No sequences found ...')``, not crash in the tokenizer.

    We monkey-patch ``GenomeOceanInference.__init__`` so no model is loaded.
    """

    def test_empty_txt_raises_value_error(self, tmp_path):
        """_load_sequences_from_file on an empty .txt file returns [].
        score() must then raise ValueError before touching the tokenizer.
        """
        try:
            from genomeocean.inference import GenomeOceanInference
        except ImportError as exc:
            pytest.skip(f"Cannot import GenomeOceanInference: {exc}")

        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")  # truly empty

        # Patch __init__ so we never try to load weights
        original_init = GenomeOceanInference.__init__

        def _fake_init(self, model_dir=None, **kwargs):
            self.model_dir = model_dir
            self.model = None
            self.tokenizer = None

        GenomeOceanInference.__init__ = _fake_init
        try:
            go = GenomeOceanInference(model_dir="fake/model")
            with pytest.raises(ValueError, match="No sequences found"):
                go.score(sequence_file=str(empty_file))
        finally:
            GenomeOceanInference.__init__ = original_init

    def test_empty_fasta_raises_value_error(self, tmp_path):
        """An empty .fa file (no records) must also raise ValueError."""
        try:
            from genomeocean.inference import GenomeOceanInference
        except ImportError as exc:
            pytest.skip(f"Cannot import GenomeOceanInference: {exc}")

        empty_fa = tmp_path / "empty.fa"
        empty_fa.write_text("")

        original_init = GenomeOceanInference.__init__

        def _fake_init(self, model_dir=None, **kwargs):
            self.model_dir = model_dir
            self.model = None
            self.tokenizer = None

        GenomeOceanInference.__init__ = _fake_init
        try:
            go = GenomeOceanInference(model_dir="fake/model")
            with pytest.raises(ValueError, match="No sequences found"):
                go.score(sequence_file=str(empty_fa))
        finally:
            GenomeOceanInference.__init__ = original_init


# ---------------------------------------------------------------------------
# 4. _load_sequences_from_file unit tests (pure I/O, no model)
# ---------------------------------------------------------------------------

class TestLoadSequencesFromFile:
    """Unit tests for the private helper that reads sequence files."""

    @pytest.fixture(autouse=True)
    def _import(self):
        try:
            from genomeocean.inference import _load_sequences_from_file
            self.load = _load_sequences_from_file
        except ImportError as exc:
            pytest.skip(f"Cannot import _load_sequences_from_file: {exc}")

    def test_txt_file(self, tmp_path):
        f = tmp_path / "seqs.txt"
        f.write_text("ATCG\nGCTA\n")
        seqs = self.load(str(f))
        assert seqs == ["ATCG", "GCTA"]

    def test_fasta_file(self, tmp_path):
        f = tmp_path / "seqs.fa"
        f.write_text(">seq1\nATCG\n>seq2\nGCTA\n")
        seqs = self.load(str(f))
        assert seqs == ["ATCG", "GCTA"]

    def test_empty_txt_returns_empty_list(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        seqs = self.load(str(f))
        assert seqs == []

    def test_empty_fasta_returns_empty_list(self, tmp_path):
        f = tmp_path / "empty.fa"
        f.write_text("")
        seqs = self.load(str(f))
        assert seqs == []

    def test_unsupported_extension_raises(self, tmp_path):
        f = tmp_path / "seqs.xyz"
        f.write_text("ATCG\n")
        with pytest.raises(ValueError, match="Unsupported"):
            self.load(str(f))

    def test_txt_blank_lines_skipped(self, tmp_path):
        f = tmp_path / "seqs.txt"
        f.write_text("ATCG\n\nGCTA\n\n")
        seqs = self.load(str(f))
        assert seqs == ["ATCG", "GCTA"]
