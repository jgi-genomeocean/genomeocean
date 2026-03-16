"""
Unified protein structure prediction backend.

Supports three backends:
- 'esmfold'   : Local ESMFold (facebookresearch/esm). Loaded once as a Python object.
- 'omegafold' : Local OmegaFold (HeliXonProtein/OmegaFold). Called via CLI subprocess.
- 'colabfold' : Local ColabFold (YoshitakaMo/localcolabfold). Called via CLI subprocess.
- 'api'       : ESMFold web API (legacy fallback).

Example:
    from genomeocean.folding import FoldingBackend
    folder = FoldingBackend(backend='esmfold')
    result = folder.predict("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG")
    # result = {'pdb': '...pdb string...', 'plddt': 88.3}
"""

import os
import time
import tempfile
import subprocess
from pathlib import Path

# Default model weight cache on scratch space (avoids home quota)
SCRATCH_CACHE = os.environ.get(
    'GENOMEOCEAN_MODEL_CACHE',
    '/pscratch/sd/z/zhwang/model_cache'
)

# Point torch hub and omegafold caches to scratch
os.environ.setdefault('TORCH_HOME', os.path.join(SCRATCH_CACHE, 'torch_hub'))
os.environ.setdefault('OMEGAFOLD_CACHE', os.path.join(SCRATCH_CACHE, 'omegafold'))

COLABFOLD_BIN = os.environ.get(
    'COLABFOLD_BIN',
    '/pscratch/sd/z/zhwang/model_cache/localcolabfold/conda_env/bin/colabfold_batch'
)


def _plddt_from_pdb(pdb_str: str) -> float:
    """Extract mean pLDDT (stored in B-factor column) from a PDB string."""
    bfactors = []
    for line in pdb_str.splitlines():
        if line.startswith(('ATOM', 'HETATM')):
            try:
                bfactors.append(float(line[60:66].strip()))
            except (ValueError, IndexError):
                pass
    return float(sum(bfactors) / len(bfactors)) if bfactors else 0.0


class FoldingBackend:
    """Lazy-loading, single-sequence structure prediction wrapper.

    Parameters
    ----------
    backend : str
        One of 'esmfold', 'omegafold', 'colabfold', 'api'.
    device : str
        PyTorch device string (e.g. 'cuda', 'cpu'). Only used by ESMFold.
    chunk_size : int or None
        ESMFold attention chunk size (128/64/32). Lower = less VRAM, slower.
        Set None to disable chunking (fastest but most memory).
    omegafold_model : int
        OmegaFold model release (1 = default, 2 = high-accuracy).
    colabfold_bin : str
        Path to the colabfold_batch binary.
    """

    def __init__(
        self,
        backend: str = 'esmfold',
        device: str = 'cuda',
        chunk_size: int = 128,
        omegafold_model: int = 2,
        colabfold_bin: str = COLABFOLD_BIN,
    ):
        assert backend in ('esmfold', 'omegafold', 'colabfold', 'api'), \
            f"Unknown backend '{backend}'. Choose: esmfold, omegafold, colabfold, api"
        self.backend = backend
        self.device = device
        self.chunk_size = chunk_size
        self.omegafold_model = omegafold_model
        self.colabfold_bin = colabfold_bin
        self._esmfold_model = None  # lazy-loaded

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def predict(self, seq: str) -> dict:
        """Predict structure for a single amino-acid sequence.

        Returns
        -------
        dict with keys:
            pdb   : str  — PDB-format structure string (empty on failure)
            plddt : float — mean pLDDT score (0.0 on failure)
            time_s : float — wall-clock seconds taken
        """
        t0 = time.time()
        try:
            if self.backend == 'esmfold':
                pdb = self._predict_esmfold(seq)
            elif self.backend == 'omegafold':
                pdb = self._predict_omegafold(seq)
            elif self.backend == 'colabfold':
                pdb = self._predict_colabfold(seq)
            else:  # api
                pdb = self._predict_api(seq)
        except Exception as e:
            print(f"[FoldingBackend:{self.backend}] Error predicting sequence (len={len(seq)}): {e}")
            pdb = ''

        elapsed = time.time() - t0
        plddt = _plddt_from_pdb(pdb) if pdb else 0.0
        
        # HuggingFace EsmForProteinFolding outputs b-factors in [0, 1]. Scale to [0, 100].
        if self.backend == 'esmfold' and 0.0 < plddt <= 1.0:
            plddt *= 100.0
            
        return {'pdb': pdb, 'plddt': plddt, 'time_s': elapsed}

    def predict_to_file(self, seq: str, outfile: str) -> bool:
        """Predict and write PDB to file. Returns True on success."""
        result = self.predict(seq)
        if result['pdb']:
            Path(outfile).parent.mkdir(parents=True, exist_ok=True)
            with open(outfile, 'w') as f:
                f.write(result['pdb'])
            return True
        return False

    def vram_used_gb(self) -> float:
        """Return current GPU VRAM usage in GB (after model load)."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_reserved() / 1e9
        except ImportError:
            pass
        return 0.0

    # -------------------------------------------------------------------------
    # ESMFold
    # -------------------------------------------------------------------------

    @property
    def _esmfold(self):
        if self._esmfold_model is None:
            import torch
            from transformers import EsmForProteinFolding, AutoTokenizer
            hf_model_id = 'facebook/esmfold_v1'
            cache = os.path.join(os.environ['TORCH_HOME'], 'hf_models')
            print(f"[ESMFold] Loading HuggingFace model '{hf_model_id}' (cache: {cache})...")
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id, cache_dir=cache)
            model = EsmForProteinFolding.from_pretrained(
                hf_model_id,
                low_cpu_mem_usage=True,
                cache_dir=cache,
            )
            model = model.to(self.device)
            model.esm = model.esm.half()  # Use fp16 for ESM trunk to save VRAM
            if self.chunk_size is not None:
                model.trunk.set_chunk_size(self.chunk_size)
            model.eval()
            self._esmfold_model = (model, tokenizer)
            print(f"[ESMFold] Model loaded. VRAM: {self.vram_used_gb():.1f} GB")
        return self._esmfold_model

    def _predict_esmfold(self, seq: str) -> str:
        import torch
        model, tokenizer = self._esmfold
        tokenized = tokenizer(
            [seq], return_tensors='pt', add_special_tokens=False
        ).to(self.device)
        with torch.no_grad():
            outputs = model(**tokenized)
        pdb_str = model.output_to_pdb(outputs)[0]
        return pdb_str

    # -------------------------------------------------------------------------
    # OmegaFold
    # -------------------------------------------------------------------------

    def _predict_omegafold(self, seq: str) -> str:
        omegafold_bin = self._find_omegafold_bin()
        cache_dir = os.environ['OMEGAFOLD_CACHE']
        os.makedirs(cache_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = os.path.join(tmpdir, 'input.fasta')
            out_dir = os.path.join(tmpdir, 'output')
            os.makedirs(out_dir, exist_ok=True)

            with open(fasta_path, 'w') as f:
                f.write(f'>seq\n{seq}\n')

            cmd = [
                omegafold_bin,
                fasta_path,
                out_dir,
                f'--model={self.omegafold_model}',
                '--subbatch_size=32',
            ]
            # Let OmegaFold use its default cache dir (Helixon downloads into ~/.cache/omegafold_ckpt)
            env = os.environ.copy()
            # OmegaFold doesn't have a standard env var for cache; weights go to ~/.cache/omegafold_ckpt
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, env=env
            )
            if result.returncode != 0:
                print(f"[OmegaFold] Error (exit={result.returncode}): {result.stdout.decode()[-1500:]}")
                return ''

            pdb_files = list(Path(out_dir).glob('*.pdb'))
            if not pdb_files:
                print(f"[OmegaFold] No PDB output found in {out_dir}")
                print(f"[OmegaFold] stdout: {result.stdout.decode()[-1000:]}")
                return ''
            return pdb_files[0].read_text()

    def _find_omegafold_bin(self) -> str:
        # Try venv bin first, then PATH
        candidates = [
            os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), 'GO', 'bin', 'omegafold'),
            '/global/homes/z/zhwang/pscratch/genomeocean/GO/bin/omegafold',
            'omegafold',
        ]
        for c in candidates:
            if os.path.isfile(c) and os.access(c, os.X_OK):
                return c
        # fall back to PATH
        import shutil
        which = shutil.which('omegafold')
        if which:
            return which
        raise FileNotFoundError(
            "omegafold binary not found. Install with: "
            "pip install git+https://github.com/HeliXonProtein/OmegaFold.git"
        )

    # -------------------------------------------------------------------------
    # ColabFold
    # -------------------------------------------------------------------------

    def _predict_colabfold(self, seq: str) -> str:
        if not os.path.isfile(self.colabfold_bin):
            raise FileNotFoundError(
                f"colabfold_batch not found at {self.colabfold_bin}. "
                "Install localcolabfold: https://github.com/YoshitakaMo/localcolabfold"
            )

        colabfold_cache = os.path.join(SCRATCH_CACHE, 'colabfold')
        os.makedirs(colabfold_cache, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = os.path.join(tmpdir, 'input.fasta')
            out_dir = os.path.join(tmpdir, 'output')
            os.makedirs(out_dir, exist_ok=True)

            with open(fasta_path, 'w') as f:
                f.write(f'>seq\n{seq}\n')

            cmd = [
                self.colabfold_bin,
                fasta_path,
                out_dir,
                '--msa-mode', 'single_sequence',   # skip MSA (de novo seqs)
                '--num-recycle', '1',               # faster, 1 recycle for benchmark
                '--model-type', 'alphafold2',
            ]
            env = os.environ.copy()
            env['XDG_CACHE_HOME'] = colabfold_cache

            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=False, env=env
            )
            if result.returncode != 0:
                print(f"[ColabFold] Error: {result.stderr.decode()[:500]}")
                return ''

            pdb_files = sorted(Path(out_dir).glob('*.pdb'))
            if not pdb_files:
                print(f"[ColabFold] No PDB output found in {out_dir}")
                return ''
            # ColabFold names the best model with _relaxed_ or _unrelaxed_rank_001
            best = sorted(pdb_files, key=lambda p: ('rank_001' in p.name), reverse=True)[0]
            return best.read_text()

    # -------------------------------------------------------------------------
    # Legacy API
    # -------------------------------------------------------------------------

    def _predict_api(self, seq: str) -> str:
        """Call ESMFold web API (legacy fallback with retry)."""
        import subprocess as sp
        import time as t
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as f:
            outfile = f.name

        for attempt in range(1, 4):
            sp.run(
                f'curl -s -X POST --data "{seq}" '
                f'https://api.esmatlas.com/foldSequence/v1/pdb/ -o {outfile}',
                shell=True
            )
            if os.path.exists(outfile) and os.path.getsize(outfile) > 0:
                content = open(outfile).read(500)
                if 'ATOM' in content or 'MODEL' in content:
                    pdb = open(outfile).read()
                    os.unlink(outfile)
                    return pdb
            if attempt < 3:
                t.sleep(10)

        if os.path.exists(outfile):
            os.unlink(outfile)
        return ''
