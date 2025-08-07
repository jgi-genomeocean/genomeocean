import random
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
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_nuc_seq_by_id(uid, start=0, end=0, db='nuccore'):
    # retrive nucleotide sequence from NCBI by id
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db={db}&id={uid}&rettype=fasta&retmode=text"
    response = requests.get(url)
    if response.status_code == 200:
        sequence = ''.join((response.text).split("\n")[1:])
        if start > 0:
            if end > 0:
                return sequence[start-1:end] # NCBI is 1-based
            else:
                return sequence[start-1:]
        else:
            return sequence 
    else:
        print("Error: ", response.status_code)
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

def fasta2pdb_api(seq, outfile):
    # use esmfold web api for predicting protein structure
    # use subprocess to run curl command
    cmd = f'curl -X POST --data "{seq}" https://api.esmatlas.com/foldSequence/v1/pdb/ > {outfile} 2>/dev/null'
    subprocess.run(cmd, shell=True) 
    # check if query is successful, outfile should have non-zero size
    if os.path.exists(outfile) and os.path.getsize(outfile) > 0:
        return True
    else:
        raise Exception("Error in cmd '{cmd}': failed to generate pdb file")
    
def LDDT_scoring(query, target_pdb, foldmason_path='', reseek_path='', method='foldmason', debug=False):
    """
    Compute structure alignment score using foldmason or reseek.
    This function is now thread-safe.
    """
    temp_dir = tempfile.mkdtemp()
    score = None
    results_to_return = None

    try:
        query_pdb = os.path.join(temp_dir, 'query.pdb')
        fasta2pdb_api(query, query_pdb)

        if method == 'foldmason':
            if not foldmason_path or not os.path.exists(foldmason_path):
                print("Foldmason path not provided or does not exist")
            else:
                pdb_dir = os.path.join(temp_dir, 'pdbs/')
                subprocess.run(['mkdir', '-p', pdb_dir], check=True)
                subprocess.run(['cp', query_pdb, pdb_dir], check=True)
                subprocess.run(['cp', target_pdb, pdb_dir], check=True)
                
                cmd = [
                    foldmason_path, 'easy-msa', pdb_dir, 
                    os.path.join(temp_dir, "results.m8"),
                    os.path.join(temp_dir, "tmp"),
                    '--match-ratio', '0.51', '--filter-msa', '1',
                    '--gap-open', 'aa:10,nucl:10', '--gap-extend', 'aa:1,nucl:1',
                    '--report-paths', '0', '--report-mode', '2'
                ]
                try:
                    results = subprocess.run(cmd, stdout=subprocess.PIPE, check=True, text=True)
                    if debug:
                        results_to_return = results
                    
                    current_score = 0.0
                    for line in results.stdout.split('\n'):
                        if 'Average MSA LDDT:' in line:
                            current_score = float(line.split(':')[1].strip())
                    score = current_score
                except subprocess.CalledProcessError as e:
                    print(f"Error running foldmason: {e}")

        elif method == 'reseek':
            if not reseek_path or not os.path.exists(reseek_path):
                print("Reseek path not provided or does not exist")
            else:
                aln_file = os.path.join(temp_dir, 'alignment.txt')
                cmd = [
                    reseek_path, '-alignpair', query_pdb, '-input2', target_pdb,
                    '-aln', aln_file, '-columns', 'aq'
                ]
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if os.path.exists(aln_file):
                        with open(aln_file, 'r') as f:
                            for line in f:
                                if line.strip().startswith('AQ'):
                                    score = float(line.strip().split()[1].split(',')[0])
                                    break
                except subprocess.CalledProcessError as e:
                    print(f"Error running reseek: {e.stderr}")
        else:
            print("Invalid method specified. Choose 'foldmason' or 'reseek'.")
    
    finally:
        shutil.rmtree(temp_dir)

    return results_to_return if debug and results_to_return else score

def LDDT_scoring_parallel(queries, target_pdb, foldmason_path='', reseek_path='', method='foldmason', debug=False, max_workers=None):
    """
    Run LDDT_scoring for multiple query sequences in parallel.

    Args:
        queries (list): A list of query sequences.
        target_pdb (str): Path to the target PDB file.
        foldmason_path (str): Path to foldmason executable.
        reseek_path (str): Path to reseek executable.
        method (str): 'foldmason' or 'reseek'.
        debug (bool): Debug flag.
        max_workers (int): Maximum number of threads to use. If None, it will default to os.cpu_count().

    Returns:
        A dictionary mapping query sequences to their scores.
    """
    scores = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_query = {
            executor.submit(LDDT_scoring, query, target_pdb, foldmason_path, reseek_path, method, debug): query 
            for query in queries
        }
        for future in as_completed(future_to_query):
            query = future_to_query[future]
            try:
                score = future.result()
                scores[query] = score
            except Exception as exc:
                print(f'{query!r} generated an exception: {exc}')
                scores[query] = None
    return scores
        
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
