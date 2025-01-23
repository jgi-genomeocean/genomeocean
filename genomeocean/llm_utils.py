# common LLM Utils as a class
# Author: Zhong Wang
# Date: 2024-09-12
"""
Example usage:
from genomeocean.llm_utils import LLMUtils

"""
import os
import numpy as np
import transformers
import torch
import torch.utils.data as util_data
import torch.nn as nn
import tqdm
import pandas as pd
from vllm import LLM, SamplingParams

from sklearn.preprocessing import normalize

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def reorder_sequences(sequences, seed=0):
    """ reorder the sequences by length
    process sequences with similar lengths in the same batch can greatly speed up the computation
    need to adjust batch_size according to the GPU memory
    use all GPUs on a node"""

    lengths = [len(seq) for seq in sequences]
    idx = np.argsort(lengths)
    return [sequences[i] for i in idx], idx 


def bad_word_processor(token_ids, logits):
    # To suppress 'N's from being generated:	
    logits[8] = float("-inf")
    return logits

def max_divisor_of_12(number):
    """Return the maximum gpu number within [1, number] that divides 12 (attention head) evenly."""
    max_divisor = None
    for i in range(1, number + 1):
        if 12 % i == 0:
            max_divisor = i
    return max_divisor   

# main class
class LLMUtils:
    def __init__(self, model_dir, model_max_length=10240, is_classification_model=False):
        self.model_dir = model_dir # model name or path
        self.tokenizer =  transformers.AutoTokenizer.from_pretrained(
                model_dir,
                cache_dir=None,
                model_max_length=model_max_length,
                padding_side="left",
                use_fast=True,
                trust_remote_code=True,
            )
        MODEL_CLASS = transformers.AutoModel if not is_classification_model else transformers.AutoModelForSequenceClassification
        self.model = MODEL_CLASS.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
        )
        self.gpus = torch.cuda.device_count()
        if self.gpus > 1: # ensure we can use all GPUs on a node allowed by max heades (12)
            self.gpus = max_divisor_of_12(self.gpus)
        self.model_max_length = model_max_length
        self.model_dir = model_dir

    
    def predict(self, dna_sequences, batch_size=25, do_embedding=True):
        """Embedding sequences using the LLM model
        adjust the batch_size according to the GPU memory
        """
 
        dna_sequences, idx = reorder_sequences(dna_sequences)
        tokenizer = self.tokenizer
        model = self.model
        if self.gpus > 1:
            model = nn.DataParallel(model)
        model.to("cuda")
        train_loader = util_data.DataLoader(dna_sequences, batch_size=batch_size*self.gpus, shuffle=False, num_workers=2*self.gpus, prefetch_factor=2)
  
        for j, batch in enumerate(tqdm.tqdm(train_loader)):
            with torch.no_grad():
                token_feat = tokenizer.batch_encode_plus(
                        batch,
                        max_length=self.model_max_length,
                        return_tensors='pt',
                        padding='longest',
                        truncation=True
                    )
                input_ids = token_feat['input_ids'].cuda()
                attention_mask = token_feat['attention_mask'].cuda()
                model_output = model.forward(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)

                if do_embedding: # use the last hidden state as the embedding
                    model_output = model_output.last_hidden_state.detach().cpu()
                    attention_mask = attention_mask.unsqueeze(-1).detach().cpu()
                    model_output = torch.sum(model_output * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
                else: # take the logits (Used for fine-tuned classification or regression models)
                    model_output = model_output.logits.detach().cpu()

                if j == 0:
                    outputs = model_output
                else:
                    outputs = torch.cat((outputs, model_output), dim=0)

        outputs = np.array(outputs.detach().float().cpu())

        # reorder the embeddings according to the original order
        outputs = outputs[np.argsort(idx)]
        return outputs


    def compute_sequence_perplexity(
        self,
        dna_sequences, 
        use_ppl=False,  # by default return log loss
        stride=512):
        """ Compute perplexity for a list of sequences, scores are in log space per seqeunce
        """
        
        tokenizer = self.tokenizer
        model = transformers.AutoModelForCausalLM.from_pretrained( 
            self.model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
        )
        
        model.to("cuda")
        
        encodings = tokenizer(dna_sequences, 
                            padding=False,)

        max_length = self.model_max_length # current max length of the model
        
        perplexities = np.zeros(len(dna_sequences))

        for i, sample in tqdm.tqdm(enumerate(encodings["input_ids"])):
            nlls = []
            prev_end_loc = 0
            sample = torch.tensor(sample).unsqueeze(0)
            seq_len = sample.size(1)
            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
                input_ids = sample[:, begin_loc:end_loc].to("cuda")
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                with torch.no_grad():
                    outputs = model(input_ids, labels=target_ids)

                    # loss is calculated using CrossEntropyLoss which averages over valid labels
                    # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                    # to the left by 1.
                    neg_log_likelihood = outputs.loss

                nlls.append(neg_log_likelihood)

                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break
            
            losses = torch.stack(nlls).mean().item()
            score = torch.exp(losses).item() if use_ppl else losses
            perplexities[i] = score
        
        return perplexities

    def compute_token_perplexity(self, dna_sequences):
        # This function computes the perplexity of a list of DNA sequences using a model in model_dir
        # Returns a numpy array of perplexities
        # print(f"Getting perplexity for {len(dna_sequences)} sequences")
        # print(f"Model directory: {model_dir}")
        
        tokenizer = self.tokenizer
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
        )
        
        model.to("cuda")
        
        encodings = tokenizer(dna_sequences, 
                            padding=False,)
        
        all_losses = []
        
        for i, sample in tqdm.tqdm(enumerate(encodings["input_ids"])):
            losses = []
            sample = torch.tensor(sample).unsqueeze(0)
            seq_len = sample.size(1)
            if seq_len > self.model_max_length:
                raise ValueError(f"Sequence length {seq_len} is greater than max_length {self.model_max_length}")
            
            input_ids = sample.to("cuda")
            target_ids = input_ids.clone()

            with torch.no_grad():
                outputs = model(input_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = target_ids[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss(reduction="none")
                shift_logits = shift_logits.view(-1, model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels).cpu().detach().numpy().tolist()
                losses.extend(loss)
            
            #print(np.mean(losses))
            all_losses.append(losses)
        
        # transfer the per token loss to
        per_token_losses = []
        for loss, sample in zip(all_losses, encodings["input_ids"]):
            sample = sample[1:-1]
            loss = loss[:-1]
            assert len(sample) == len(loss), f"Sample length {len(sample)} does not match loss length {len(loss)}"
            
            per_token_loss = []
            for token, token_loss in zip(sample, loss):
                token_length = len(tokenizer.decode([token]))
                per_token_loss.extend([token_loss] * token_length)
                
            per_token_losses.append(per_token_loss)
   
        return per_token_losses

    def generate(
        self, 
        prompts,
        num_generation_from_each_prompt=100,
        temperature=0.7,
        min_length=128,
        max_length=1024, 
        top_k=50,
        top_p=0.95,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        seed=0,
    ):
        """
        presence_penalty: Float that penalizes new tokens based on whether they appear in the generated text
        so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to 
        repeat tokens.

        frequency_penalty: Float that penalizes new tokens based on their frequency in the generated text 
        so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to 
        repeat tokens.

        repetition_penalty: Float that penalizes new tokens based on whether they appear in the prompt and 
        the generated text so far. Values > 1 encourage the model to use new tokens, while values < 1 
        encourage the model to repeat tokens. max 2.0

        temperature:    Float that controls the randomness of the sampling. Lower values make the model more 
        deterministic, while higher values make the model more random. Zero means greedy sampling.

        """  
        num_gpus = self.gpus

        # Initialize the LLM model using vllm package
        llm = LLM(
            model=self.model_dir,
            tokenizer=self.model_dir,
            tokenizer_mode="slow",
            trust_remote_code=True,
            seed=seed,
            dtype=torch.bfloat16,
            gpu_memory_utilization=0.9, # default is 0.9
            tensor_parallel_size=num_gpus,
        )

        # Initialize the tokenizer separately
        tokenizer = self.tokenizer
        prompts = ["[CLS]"+p for p in prompts]
        prompt_token_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
        
        sampling_params = SamplingParams(
            n=num_generation_from_each_prompt,
            temperature=temperature, 
            top_k=top_k,
            top_p=top_p,
            stop_token_ids=[2],
            max_tokens=max_length,
            min_tokens=min_length,
            detokenize=False,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
            logits_processors=[bad_word_processor], #To suppress 'N's from being generated, remove it if not needed
        )
        
        # Generate sequences using prompt_token_ids
        generated_sequences = []
        if num_generation_from_each_prompt >= 100:
            # If num_generation_from_each_prompt is greater than 100, generate sequences for each prompt one by one to optimize speed
            for prompt in prompt_token_ids:
                all_outputs = llm.generate(
                    prompts=None, 
                    prompt_token_ids=prompt,
                    sampling_params=sampling_params,
                )

                for outputs in all_outputs:
                    for output in outputs.outputs:
                        text = tokenizer.decode(output.token_ids, skip_special_tokens=True).replace(" ", "").replace("\n", "")
                        generated_sequences.append(text)
                        
        else:
            # Otherwise, directly use the vllm generate function
            all_outputs = llm.generate(
                prompts=None, 
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
            )
            
            for outputs in all_outputs:
                for output in outputs.outputs:
                    text = tokenizer.decode(output.token_ids, skip_special_tokens=True).replace(" ", "").replace("\n", "")
                    generated_sequences.append(text)

        print(f"Generated {len(generated_sequences)} sequences")
        
        return generated_sequences
    
    

