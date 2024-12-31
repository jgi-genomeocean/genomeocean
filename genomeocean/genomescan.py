""" genome-wide scan utility

# Example usage
out_prefix = 'path/to/output'
wgs = GenomeWideScanUtility(genome_file='path/to/genome.fasta.gz', overlap=500, model_max_length=10240)
# Produce input files to be used by the model
wgs.segment_genome(out_prefix=out_prefix)

# Run the model to preduce score

# Read the output file from the model
scores = {}
for chrom in wgs.chromosomes:
    score_file = f"{out_prefix}_{chrom}_scores.csv"
    if os.path.exists(score_file):
        score = wgs.process_score(score_file, window=2000)
        scores[chrom] = score

# make wiggle files
wgs.score_to_wiggle(scores, out_prefix=out_prefix, name='my track')

"""

from Bio import SeqIO
from genomeocean.dnautils import reverse_complement
import pandas as pd
import numpy as np
import gzip
import csv
import os

class GenomeWideScanUtility:
    def __init__(self, genome_file='', min_seq_len=5000, model_max_length=40000, overlap=0, rc=True):
        self.min_seq_len = min_seq_len
        self.model_max_length = model_max_length
        self.overlap = overlap
        self.rc = rc
        self.genome_file = genome_file
        self.chromosomes=[]

    def segmentation(self, s, s_id):
        """segment a sequence into overlapping fragments
        s is a string of DNA sequence
        allow fragments to overlap
        set min_seq_len to 0 to scan the whole sequence
        set rc=True to include reverse complement of each fragment
        return a pandas dataframe with id and seq columns
        """
        seqs = {}
        s = s.upper()
        for i in range(0, len(s), self.model_max_length - self.overlap):
            if (len(s) - i) < (self.min_seq_len - self.overlap):  # skip if the remaining sequence is too short
                break
            segment = s[i:(i+self.model_max_length)]
            s_id_segment = f'{s_id}:{i}-{i + len(segment)}'
            seqs[s_id_segment] = segment
            if self.rc:
                seqs[s_id_segment + '_rc'] = reverse_complement(segment)
        seqs = pd.DataFrame(seqs.items(), columns=['id', 'seq'])

        return seqs

    def segment_genome(self, out_prefix=''):
        """load sequences from a genome gzip file, process it into fragments, each chromosome is saved into a separate file

        TODO: a long stretch of 'N's will be the worst case, 1token=1base.
        Set the model_max_length to 10240 (model size). A better solution would be to skip calculating the losses of Ns
        This is rather common in draft genome assemblies (large gaps)
        Need to workaround that (break the sequences at 'N's, keep track of their original positions)

        Taking care of the gaps will also take care of models that produce scores at 'N' tokens

        """
        with gzip.open(self.genome_file, 'rt') as G:
            for record in SeqIO.parse(G, 'fasta'):
                chrom = str(record.id)
                self.chromosomes.append(chrom)
                segments = self.segmentation(str(record.seq), chrom)
                # the first column needs to be sequence for the model
                save_path = out_prefix+'_'+chrom
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                segments[['seq', 'id']].to_csv(save_path, index=False, header=None)


    def process_score(self, score_file, window=2000):
        """given a list of scores (with rc), combine them into a single score

        TODO: Some tokens are either uncertain (model limitation) or unimportant (evolution) for function. Need to toss out high losses
        E.g., since the loss values follow Gaussian, we can simply discard all values larger than two SDs

        The rolling mean method loses resolution. We can apply some Bayesian methods. In a BGC region, the low scores won't be thrown off
        by a few outliers. E.g., from a low score region, the next high-value won't be counted, and the next next high-value will be discounted.

        set window to 1 to disable rolling mean
        """
        # read the model output
        with open(score_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            scores = [row for row in reader]

        score = scores[0]

        if self.rc:
            score_rc = scores[1]
            for i in range(2, len(scores) - 2, 2):
                score += scores[i][self.overlap:]  # the first overlap is not trustworthy, no context
                score_rc = score_rc[:len(score_rc)-self.overlap] + scores[i+1]  # the last overlap is not trustworthy, no context
        else:
            for i in range(1, len(scores)):
                score += scores[i][self.overlap:]  # the first overlap is not trustworthy, no context
        # take the smaller of the two strands, need to be BGC for at least one strand
        score = np.array([score, score_rc], dtype=np.float16).min(axis=0) 

        score = pd.DataFrame(score)
        if window > 1:
            score = score.rolling(window).mean()  # rolling average to suppress noise, lower for better resolution
        score = score.fillna(0)
        return score

    def score_to_wiggle(self, scores, out_prefix, name='my_score'):
        """convert scores to a wiggle file
        scores is a dictionary with chromosome as key and a pandas dataframe as value
        """
        for chrom in scores.keys():
            score = scores[chrom]
            with open(out_prefix+'_'+str(chrom)+'.wig', 'w') as OUT:
                OUT.write('track type=wiggle_0 name="{name}"\nfixedStep chrom={chrom} start=1 step=1\n'.format(name=name, chrom=chrom))
                for row in score.iterrows():
                    OUT.write("{score}\n".format(score=str(row[1][0])))
        
        return None