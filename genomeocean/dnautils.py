import random
import time
import pandas as pd
import numpy as np
import gzip
import glob
import re
import requests
from Bio.Seq import Seq
from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
import subprocess
import os, sys


def get_nuc_seq_by_id(uid, start=0, end=0, db='nuccore'):
    # retrive nucleotide sequence from NCBI by id
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db={db}&id={uid}&rettype=fasta&retmode=text"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        if response.status_code == 200:
            sequence = ''.join((response.text).split("\n")[1:])
            if start>0:
                if end>0:
                    return sequence[start-1:end] # NCBI is 1-based
                else:
                    return sequence[start-1:]
            else:
                return sequence 
        else:
            print("Error: ", response.status_code)
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching from NCBI: {e}")
        return None

def fetch_genome(genbank_id="U00096.3"):
    """fetch genome from NCBI by id, default to E. coli K-12 MG1655"""
    Entrez.email = "your_email@example.com"  # Replace with your email
    handle = Entrez.efetch(db="nucleotide", id=genbank_id, rettype="gb", retmode="text")
    record = SeqIO.read(handle, "genbank")
    handle.close()
    return record


def extract_gene_sequences(genome_record, num_genes=12, seed=42, upstream=1000, downstream=100):
    """randomly select genes and extract sequences, both nucleotide and protein"""
    gene_features = [f for f in genome_record.features if f.type == "gene" and "gene" in f.qualifiers]
    
    # Randomly select genes
    random.seed(seed)
    selected_genes = random.sample(gene_features, num_genes)
    nucleotide_sequences = []
    protein_sequences = []
    
    for gene in selected_genes:
        gene_name = gene.qualifiers["gene"][0]
        
        # Find corresponding CDS feature for protein translation
        cds_feature = next((f for f in genome_record.features if f.type == "CDS" and "gene" in f.qualifiers and f.qualifiers["gene"][0] == gene_name), None)
        
        if cds_feature and "translation" in cds_feature.qualifiers:
            # Get the protein translation from the CDS feature
            protein_seq = cds_feature.qualifiers["translation"][0]
            
            # Create a SeqRecord for the protein sequence
            prot_record = SeqRecord(Seq(protein_seq), id=gene_name, description=f"{gene_name} - protein sequence from GenBank")
            protein_sequences.append(prot_record)
        else:
            print(f"No CDS feature found for {gene_name}")
            continue
        
        # Get the start position and strand orientation for the nucleotide sequence
        start = int(cds_feature.location.start)  # Change to CDS start position
        strand = gene.location.strand
        
        if strand == 1:
            # Positive strand: 1kb upstream and 100bp downstream of the start codon
            upstream_start = max(0, start - upstream)
            upstream_seq = genome_record.seq[upstream_start:start]
            downstream_seq = genome_record.seq[start:start + downstream]
            sequence = upstream_seq + downstream_seq
        elif strand == -1:
            # Negative strand: 1kb downstream and 100bp upstream (relative to gene orientation), reverse complement
            downstream_end = min(len(genome_record), start + upstream)
            downstream_seq = genome_record.seq[start:start + downstream]
            upstream_seq = genome_record.seq[start + downstream:start + upstream + downstream]
            sequence = (downstream_seq + upstream_seq).reverse_complement()
        
        # Create SeqRecord for the extracted nucleotide sequence
        nuc_record = SeqRecord(sequence, id=gene_name, description=f"{gene_name} - extracted {upstream}bp upstream + {downstream}bp downstream of start codon")
        nucleotide_sequences.append(nuc_record)
    
    return nucleotide_sequences, protein_sequences

def reverse_complement(sequence):
    # Create a translation table for complementing bases
    complement = str.maketrans('ATCGN', 'TAGCN')
    # Reverse the complemented sequence
    reverse_complemented = sequence.translate(complement)[::-1]    
    return reverse_complemented

def process_gbk_dir(data_dir, num_seqs=100, min_seq_len=5000, max_seq_len=40000, overlap=0, rc=True):
    """load sequences from a directory of GBK files process them into fragments"""

    seqs = {}
    counter = 0
    for f in glob.glob(data_dir + '/*.gbk'):
        with open(f, 'rt') as FA:
            for s in SeqIO.parse(FA, 'gb'):
                if counter >= num_seqs:
                    break
                if len(s.seq) < min_seq_len:
                    continue
                segments = segmentation(s, min_seq_len=min_seq_len, max_seq_len=max_seq_len, overlap=overlap, rc=rc)
                seqs = {**seqs, **segments}
            counter +=1        
    seqs = pd.DataFrame.from_dict(seqs, orient='index', columns=['seq'])
    return seqs

def process_genome(genome_file, min_seq_len=5000, max_seq_len=40000, overlap=0, rc=True):
    """load sequences from a genome gzip file, process it into fragments
    
    Todo: a long strecth of 'N's will be the worst case, 1token=1base. 
    Set the max_seq_len to 10240 (model size). a better solution would be skip calculating the losses of Ns
    This is rather common in draft genome assemblies (large gaps)
    Need to workaround that (break the sequences at 'N's, keep track of their original positions)
    
    """
    genome = {}
    with gzip.open(genome_file, 'rt') as G:
        for record in SeqIO.parse(G, 'fasta'):
            segments = segmentation(record, min_seq_len=min_seq_len, max_seq_len=max_seq_len, overlap=overlap, rc=rc)
            genome = {**genome, **segments}
    genome = pd.DataFrame.from_dict(genome, orient='index', columns=['seq'])
    return genome


def find_tandem_repeats_percentage(sequence, max_period_size=100):
    """
    Find the percentage of the sequence that consists of perfect tandem repeats.
    
    Parameters:
    sequence (str): DNA sequence to search for tandem repeats.
    max_period_size (int): Maximum allowed period size for repeats (default is 7).
    
    Returns:
    float: Percentage of the sequence that is made up of tandem repeats.
    """
    sequence_length = len(sequence)
    repeats = []
    covered_positions = set()  # To track positions already counted

    # Iterate over possible period sizes from 1 to max_period_size
    for period_size in range(1, max_period_size + 1):
        # Use regex to match a tandem repeat pattern: (x){2,}, where x is any sequence of period_size length, 3 or more times
        pattern = r'((\w{%d}))\2{2,}' % period_size  # Adjust regex for the period size

        # Find all matches in the sequence
        for match in re.finditer(pattern, sequence):
            start = match.start()
            end = match.end()

            # Only consider new, non-overlapping positions
            for i in range(start, end):
                if i not in covered_positions:
                    covered_positions.add(i)
    
    # Calculate the percentage of the sequence that is part of a repeat
    repeat_length = len(covered_positions)
    percentage = (repeat_length / sequence_length) * 100
    
    return percentage

def fasta2pdb_api(seq, outfile, max_retries=3, retry_delay=10, backend='api'):
    """Predict protein structure and write PDB to outfile.

    Parameters
    ----------
    seq : str
        Amino-acid sequence to fold.
    outfile : str
        Path to write the PDB output.
    max_retries : int
        Number of retries (only used by 'api' backend).
    retry_delay : int
        Seconds between retries (only used by 'api' backend).
    backend : str
        'api'       — ESMFold web API (legacy, unreliable).
        'esmfold'   — Local ESMFold (recommended).
        'omegafold' — Local OmegaFold.
        'colabfold' — Local ColabFold.

    Returns True on success, False on failure.
    """
    if backend != 'api':
        # Use local FoldingBackend
        try:
            from genomeocean.folding import FoldingBackend
            folder = FoldingBackend(backend=backend)
            return folder.predict_to_file(seq, outfile)
        except Exception as e:
            print(f"[fasta2pdb_api] Local backend '{backend}' failed: {e}")
            return False

    # Legacy API path with retry + PDB validation
    for attempt in range(1, max_retries + 1):
        cmd = f'curl -m 30 -s -X POST --data "{seq}" https://api.esmatlas.com/foldSequence/v1/pdb/ -o {outfile}'
        subprocess.run(cmd, shell=True)
        
        # Validate: file must exist, be non-empty, and contain valid PDB records
        if os.path.exists(outfile) and os.path.getsize(outfile) > 0:
            with open(outfile, 'r') as f:
                content = f.read(10000)  # read first 10k chars to bypass long headers
            if 'ATOM' in content or 'MODEL' in content:
                return True
            else:
                print(f"ESMfold attempt {attempt}/{max_retries}: returned invalid PDB (no ATOM records). Content preview: {content[:100]}")
        else:
            print(f"ESMfold attempt {attempt}/{max_retries}: empty or missing output file for outfile={outfile}")
        
        if attempt < max_retries:
            print(f"Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
    
    print(f"ESMfold API failed after {max_retries} attempts for sequence (len={len(seq)}).")
    return False
    
def LDDT_scoring(query, target_pdb, foldmason_path='', debug=False, backend='esmfold'):
    """Compute structural lDDT by folding a query sequence and comparing to target_pdb.

    Parameters
    ----------
    query : str
        Amino-acid sequence to fold and compare.
    target_pdb : str
        Path to the reference PDB file.
    foldmason_path : str
        Path to the foldmason binary.
    debug : bool
        If True, return the raw subprocess result instead of the lDDT score.
    backend : str
        Folding backend to use: 'esmfold' (default), 'omegafold', 'colabfold', 'api'.
    """
    if not foldmason_path:
        print("Foldmason path not provided")
        return None
    if not os.path.exists(foldmason_path):
        print("Foldmason path does not exist")
        return None
    # create temp directory
    temp_dir = 'lddt_temp/'
    subprocess.run(f'rm -rf {temp_dir} && mkdir -p {temp_dir}', shell=True)
    pdb_dir = 'lddt_pdbs/'
    
    subprocess.run(f'rm -rf {pdb_dir} && mkdir -p {pdb_dir}', shell=True)
    # predict structure using the specified backend
    success = fasta2pdb_api(query, pdb_dir+'query.pdb', backend=backend)
    if not success:
        subprocess.run(f'rm -rf {temp_dir} {pdb_dir}', shell=True)
        return None
    subprocess.run(['cp', target_pdb, pdb_dir])
    cmd = f'{foldmason_path} easy-msa {pdb_dir} results.m8 {temp_dir} --match-ratio 0.51 --filter-msa 1 --gap-open aa:10,nucl:10 --gap-extend aa:1,nucl:1 --report-paths 0 --report-mode 2'
    # run the command and capture the results, check if the command is successful
    try:
        results = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if results.returncode != 0:
            print(f"Foldmason Warning: non-zero exit ({results.returncode}). Output: {results.stderr.decode()}")
            return None
    except Exception as e:
        print(f"Error executing foldmason: {e}")
        return None
    
    subprocess.run('rm -rf tmp/ pdbs/ results.m8*', shell=True)
    score = 0.0
    for l in results.stdout.decode().split('\n'):
        if 'Average MSA LDDT:' in l:
            score = float(l.split(':')[1].strip())
    if debug:
        return results
    # remove temp files
    subprocess.run(f'rm -rf {temp_dir} {pdb_dir}', shell=True)
    return score
        
def introduce_mutations(orf_sequence, mutation_percentage=5, mutation_type='synonymous', codontable=11):
    codon_table = CodonTable.unambiguous_dna_by_id[codontable]
    codons = [orf_sequence[i:i+3] for i in range(0, len(orf_sequence), 3)]
    
    num_codons_to_mutate = int(len(codons) * mutation_percentage / 100)
    codons_to_mutate = random.sample(range(len(codons)), num_codons_to_mutate)
    for index in codons_to_mutate:
        original_codon = codons[index]
        amino_acid = codon_table.forward_table.get(original_codon, None)
        if amino_acid:
            if mutation_type=='synonymous':
                candidate_codons = [codon for codon, aa in codon_table.forward_table.items() if aa == amino_acid and codon != original_codon]
            else:
                candidate_codons = [codon for codon, aa in codon_table.forward_table.items() if aa != amino_acid]
            if candidate_codons:
                codons[index] = random.choice(candidate_codons)
    
    mutated_sequence = ''.join(codons)
    return mutated_sequence
