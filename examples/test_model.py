# Check if the model works correctly by embedding sequences and generating new sequences

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from genomeocean.generation import SequenceGenerator


# Step 1: Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "DOEJGI/GenomeOcean-4B",
    trust_remote_code=True,
    padding_side="left",
)

# Step 2: Load the model
model = AutoModel.from_pretrained(
    "DOEJGI/GenomeOcean-4B",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to("cuda")

print("Model and tokenizer downloaded successfully!")

# Step 3: Embedding sequences
sequences = [
    "GCCGCTAAAAAGCGACCAGAATGATCCAAAAAAGAAGGCAGGCCAGCACCATCCGTTTTTTACAGCTCCAGAACTTCCTTT", 
    "CAGTCAGTGGCTAGCATGCTAGCATCGATCGATCGATCGATCGATCGATCGATCGGTGCATGCTAGCATCGATCGATCGAA"
]
output = tokenizer.batch_encode_plus(
    sequences,
    max_length=10240,
    return_tensors="pt",
    padding="longest",
    truncation=True
)
input_ids = output["input_ids"].cuda()
attention_mask = output["attention_mask"].cuda()
model_output = model.forward(input_ids=input_ids, attention_mask=attention_mask)[0].detach().cpu()
attention_mask = attention_mask.unsqueeze(-1).detach().cpu()
embedding = torch.sum(model_output * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
print(f"Embedding Shape: {embedding.shape}")  # Should be (2, 3072)

assert embedding.shape == (2, 3072), "Embedding shape is not (2, 3072), sth. is wrong!"

# Step 4: Sequence generation
seq_gen = SequenceGenerator(
    model_dir='DOEJGI/GenomeOcean-4B',
    prompts=[sequences[0]],
    num=1,
    min_seq_len=10,
    max_seq_len=10,
    temperature=1.0,
    top_p=0.9,
    seed=123,
)
all_generated = seq_gen.generate_sequences(
    prepend_prompt_to_output=False
)
generated = all_generated['seq'][0]
print(f"Generated sequence: {generated}")

assert len(generated) >= 10, "Generated sequence length is too short, sth. is wrong!"
