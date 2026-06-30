import unittest
import torch
import os
import sys

# Ensure the package is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



class TestGenomeOceanCapabilities(unittest.TestCase):
    def test_embedding(self):
        from genomeocean.embedding import llm_embed_sequences
        print("\nTesting Embedding...")

        model_name = "pGenomeOcean/GenomeOcean-100M" # Use a smaller model for testing if available
        # Check if environment variable for model path is set, otherwise use default
        if "GENOMEOCEAN_MODEL_PATH" in os.environ:
             model_name = os.environ["GENOMEOCEAN_MODEL_PATH"]
        
        sequences = ["ACGTACGT", "TGCATGCA"]
        embeddings = llm_embed_sequences(sequences, model_name, batch_size=2)
        print(f"Embeddings shape: {embeddings.shape}")
        self.assertEqual(embeddings.shape[0], 2)
        self.assertTrue(embeddings.shape[1] > 0)

    def test_generation(self):
        from genomeocean.generation import SequenceGenerator
        print("\nTesting Generation...")

        model_name = "pGenomeOcean/GenomeOcean-100M"
        if "GENOMEOCEAN_MODEL_PATH" in os.environ:
             model_name = os.environ["GENOMEOCEAN_MODEL_PATH"]

        prompts = ["ACGT"]
        generator = SequenceGenerator(model_dir=model_name, prompts=prompts, num=1, min_seq_len=10, max_seq_len=20)
        generated = generator.generate_sequences()
        print(f"Generated sequences: {generated}")
        self.assertTrue(len(generated) > 0)
        self.assertTrue(len(generated.iloc[0]['seq']) >= 10)


if __name__ == '__main__':
    # Try embedding first as it doesn't depend on vllm
    suite = unittest.TestSuite()
    suite.addTest(TestGenomeOceanCapabilities('test_embedding'))
    suite.addTest(TestGenomeOceanCapabilities('test_generation'))
    unittest.TextTestRunner().run(suite)

