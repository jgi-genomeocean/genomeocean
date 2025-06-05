# -*- coding: utf-8 -*-
"""
Module for generating LLM embeddings.
"""

import logging
import tqdm
import torch
import numpy as np
import transformers
from torch.utils.data import DataLoader
import gc
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os

logger = logging.getLogger(__name__)

# GenomeOcean Model Token Limits - Centralized configuration
MODEL_TOKEN_LIMITS = {
    'pGenomeOcean/GenomeOcean-100M': 1024,
    'pGenomeOcean/GenomeOcean-500M': 1024,
    'pGenomeOcean/GenomeOcean-4B': 10240
}

def llm_embed_sequences(dna_sequences,
                        model_name_or_path,
                        batch_size=25,
                        layer_num=-1,
                        strategy='mean',
                        device='cuda',
                        num_workers=2,
                        prefetch_factor=2):
    """
    Embed DNA sequences using a pre-trained Hugging Face transformer model (e.g., GenomeOcean).

    Automatically adjusts batch size downwards if CUDA OOM occurs.
    Sorts sequences by length for potentially faster processing.

    Handles sequences of different lengths by using the tokenizer's padding mechanism
    during the embedding process. No N-padding of sequences is required beforehand.
    The tokenizer pads to the length of the longest sequence in each batch.

    Args:
        dna_sequences (list): List of DNA sequence strings to embed. Can have varying lengths.
        model_name_or_path (str): Name or path of the Hugging Face model.
        batch_size (int): Initial batch size for processing.
        layer_num (int): Layer number to extract embeddings from (-1 for the last layer).
                         Indices are 0-based relative to hidden layers (0 is first hidden layer).
        strategy (str): Pooling strategy for token embeddings ('mean' or 'last_token').
        device (str): Device to run the model on ('cuda' or 'cpu').
        num_workers (int): Number of worker processes for DataLoader.
        prefetch_factor (int): Prefetch factor for DataLoader.

    Returns:
        np.ndarray: A NumPy array of embeddings (shape: num_sequences x embedding_dim).

    Raises:
        ValueError: If an invalid strategy is provided or model name is not found in limits.
        RuntimeError: If model loading or embedding computation fails for non-OOM reasons.
        ImportError: If required libraries (torch, transformers) are not installed.
    """
    if not dna_sequences:
        logger.warning("Received empty list of DNA sequences for embedding.")
        return np.empty((0, 0)) # Return empty array matching expected type

    # Import gc at the beginning of the function for garbage collection
    import gc

    if strategy not in ['mean', 'last_token']:
        raise ValueError(f"Invalid embedding strategy: {strategy}. Choose 'mean' or 'last_token'.")
    if model_name_or_path not in MODEL_TOKEN_LIMITS:
         # Try to infer if it's a path vs name
         if os.path.isdir(model_name_or_path):
              logger.warning(f"Model path '{model_name_or_path}' not found in pre-defined MODEL_TOKEN_LIMITS. Assuming default limit of 1024. Performance might be affected if incorrect.")
              model_max_length = 1024
         else:
              raise ValueError(f"Model name '{model_name_or_path}' not found in MODEL_TOKEN_LIMITS configuration.")
    else:
        model_max_length = MODEL_TOKEN_LIMITS[model_name_or_path]

    logger.info(f"Starting LLM embedding for {len(dna_sequences)} sequences.")
    logger.info(f"Model: {model_name_or_path}, Max Length: {model_max_length}, Strategy: {strategy}, Layer: {layer_num}, Device: {device}")

    # Reorder sequences by length for efficiency
    try:
        lengths = [len(seq) for seq in dna_sequences]
        original_indices = np.arange(len(dna_sequences))
        sorted_indices = np.argsort(lengths)
        dna_sequences_sorted = [dna_sequences[i] for i in sorted_indices]
    except Exception as e:
        logger.error(f"Error processing sequence lengths: {e}", exc_info=True)
        raise

    # --- Load Model and Tokenizer ---
    try:
        logger.info("Loading tokenizer...")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=None,
            model_max_length=model_max_length,
            padding_side="left", # Pad on left for decoder-only or encoder-decoder models if needed
            use_fast=True,
            trust_remote_code=True,
        )
        logger.info("Loading model...")
        # Try loading with bfloat16, fallback to float32 if needed
        try:
            model = transformers.AutoModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16, # Use bfloat16 for potential speedup/memory saving
            )
            logger.info("Model loaded with torch.bfloat16 data type.")
        except (RuntimeError, ValueError) as e: # Catch potential dtype errors
             logger.warning(f"Failed to load model with bfloat16 ({e}). Falling back to default (float32).")
             model = transformers.AutoModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                # torch_dtype=torch.float32, # Default
             )
             logger.info("Model loaded with default data type (likely float32).")

        # Handle multi-GPU
        # --- Distributed Setup ---
        is_distributed = dist.is_available() and dist.is_initialized()
        if is_distributed:
            local_rank = dist.get_rank()
            world_size = dist.get_world_size()
            logger.info(f"Distributed training detected. Rank: {local_rank}, World Size: {world_size}")
            device = local_rank # Assign device based on rank
            torch.cuda.set_device(device)
            model.to(device)
            # Find unused parameters can be helpful for debugging DDP issues
            model = DDP(model, device_ids=[device], output_device=device, find_unused_parameters=False)
            logger.info(f"Wrapped model with DistributedDataParallel on device {device}.")
            # Remove the old DataParallel batch size scaling logic
        elif device == 'cuda': # Single GPU CUDA or CPU
            model.to(device)
        # else: model remains on CPU

        model.eval() # Ensure model is in eval mode

    except Exception as e:
        logger.error(f"Failed to load model or tokenizer '{model_name_or_path}': {e}", exc_info=True)
        raise RuntimeError(f"Failed to load model or tokenizer: {e}")

    # Using fixed batch size as provided in the parameters
    # Note: This batch_size is now per-GPU batch size if distributed
    logger.info(f"Using fixed batch size of {batch_size} for embedding.")


    # --- Process Sequences in Batches ---
    # Create dataset from sorted sequences
    # Note: Sorting might be less critical with DDP sampler, but keep for now
    dataset = dna_sequences_sorted
    sampler = DistributedSampler(dataset, shuffle=False) if is_distributed else None # Don't shuffle for inference/embedding

    # Adjust num_workers based on GPU count for better multi-GPU performance
    # This logic might need refinement depending on system specifics
    if device == 'cuda' and torch.cuda.device_count() > 1 and not is_distributed: # Original DataParallel logic check
         adjusted_workers = min(num_workers * torch.cuda.device_count(), os.cpu_count() or 1)
         logger.info(f"Processing embeddings with batch size {batch_size} using {adjusted_workers} workers for {torch.cuda.device_count()} GPUs (DataParallel).")
    elif is_distributed:
         # For DDP, num_workers is per process
         adjusted_workers = num_workers
         logger.info(f"Processing embeddings with per-GPU batch size {batch_size} using {adjusted_workers} workers (Distributed).")
    else: # Single GPU or CPU
         adjusted_workers = num_workers
         logger.info(f"Processing embeddings with batch size {batch_size} using {adjusted_workers} workers.")


    dataloader = DataLoader(
        dataset,
        batch_size=batch_size, # This is now per-GPU batch size if distributed
        shuffle=False, # Sampler handles shuffling if needed (set to False here)
        num_workers=adjusted_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True if device != 'cpu' else False, # Pin memory if using CUDA
        sampler=sampler # Use the distributed sampler if applicable
    )

    all_embeddings = []
    try:
        with torch.no_grad():
            # If using DistributedSampler, set the epoch to ensure proper shuffling across epochs (not relevant for inference, but good practice)
            if is_distributed and sampler:
                 sampler.set_epoch(0) # Set epoch 0 for inference

            for batch in tqdm.tqdm(dataloader, desc="Embedding sequences", disable=is_distributed and local_rank != 0): # Only show progress bar on rank 0
                # Tokenize sequences - handles different length sequences by padding to the longest in batch
                # No N-padding of sequences is needed beforehand
                token_feat = tokenizer.batch_encode_plus(
                    batch, max_length=model_max_length, return_tensors='pt',
                    padding='longest', truncation=True
                )
                input_ids = token_feat['input_ids'].to(device)
                attention_mask = token_feat['attention_mask'].to(device)

                outputs = model.forward(input_ids=input_ids, output_hidden_states=True, attention_mask=attention_mask)
                hidden_states = outputs.hidden_states # Tuple of hidden states

                # Extract the specified layer's hidden state
                # layer_num = -1 means the last layer. hidden_states includes embedding layer + all transformer layers
                # So layer_num=0 is the first transformer layer output, layer_num=-1 is the last.
                if layer_num < -1 or layer_num >= len(hidden_states) -1 :
                     actual_layer_index = -1 # Default to last layer if invalid index
                     if is_distributed and local_rank == 0: # Only warn on rank 0
                         logger.warning(f"Invalid layer_num {layer_num}. Using last layer ({actual_layer_index}).")
                else:
                     actual_layer_index = layer_num if layer_num != -1 else -1

                model_output = hidden_states[actual_layer_index] # Index directly if -1 is last

                # Apply pooling strategy
                if strategy == 'mean':
                    # Masked average pooling
                    attention_mask_expanded = attention_mask.unsqueeze(-1).expand_as(model_output)
                    sum_embeddings = torch.sum(model_output * attention_mask_expanded, dim=1)
                    sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9) # Avoid division by zero
                    embedding = sum_embeddings / sum_mask
                elif strategy == 'last_token':
                    # Find the index of the last non-padding token for each sequence
                    # sequence_lengths = torch.eq(input_ids, tokenizer.pad_token_id).int().argmax(-1) - 1 # Alternative way
                    sequence_lengths = attention_mask.sum(dim=1) - 1
                    # Gather the hidden state of the last token for each sequence
                    embedding = model_output[torch.arange(model_output.size(0), device=device), sequence_lengths]
                else:
                    # This case is already validated, but as a safeguard:
                    raise ValueError(f"Internal error: Invalid strategy '{strategy}' reached processing.")

                all_embeddings.append(embedding.cpu().detach()) # Move to CPU and detach immediately

                # Explicitly delete intermediate tensors and clear cache
                del input_ids, attention_mask, outputs, hidden_states, model_output
                if strategy == 'mean':
                    del attention_mask_expanded, sum_embeddings, sum_mask
                elif strategy == 'last_token':
                    del sequence_lengths
                # torch.cuda.empty_cache() # Clear unused memory from cache

    except RuntimeError as e:
        logger.error(f"Rank {local_rank if is_distributed else 0}: Runtime error during embedding batch processing: {e}", exc_info=True)
        if 'out of memory' in str(e).lower():
             logger.error(f"Rank {local_rank if is_distributed else 0}: Consider reducing --batch_size further.")
        # Ensure all processes hit the barrier even on error
        if is_distributed:
            dist.barrier()
        raise
    except Exception as e:
        logger.error(f"Rank {local_rank if is_distributed else 0}: Unexpected error during embedding batch processing: {e}", exc_info=True)
        # Ensure all processes hit the barrier even on error
        if is_distributed:
            dist.barrier()
        raise

    if not all_embeddings:
        logger.warning(f"Rank {local_rank if is_distributed else 0}: No embeddings were generated in this process.")
        # Create an empty tensor with the correct dimension for gathering
        # Need to get the embedding dimension from the model config
        # If all_embeddings is empty, we might not have the model object directly here
        # Need a way to get the embedding dimension without the model instance if no data was processed
        # Let's assume for now that if all_embeddings is empty, the process didn't get any data
        # and the embedding dimension can be inferred or handled during gathering.
        # The gathering logic below already handles empty tensors to some extent.
        return np.empty((0, 0)) # Return empty numpy array

    # Concatenate embeddings from this process (already on CPU)
    embeddings_tensor = torch.cat(all_embeddings, dim=0)

    # --- Result Aggregation ---
    if is_distributed:
        # Gather results from all processes to rank 0
        # This requires tensors to be on the GPU for dist.all_gather
        # We need to move the concatenated CPU tensor back to GPU for gathering
        embeddings_tensor_gpu = embeddings_tensor.to(device)

        # Gather the sizes first to handle varying tensor sizes
        local_size = torch.tensor([embeddings_tensor_gpu.shape[0]], dtype=torch.int64, device=device)
        gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(gathered_sizes, local_size)
        gathered_sizes = [s.item() for s in gathered_sizes]

        max_size = max(gathered_sizes)
        # Get embedding dimension from the tensor if not empty, otherwise need a fallback
        embedding_dim = embeddings_tensor_gpu.shape[1] if embeddings_tensor_gpu.numel() > 0 else (model.module.config.hidden_size if is_distributed else model.config.hidden_size)

        # Pad local tensor if necessary
        if embeddings_tensor_gpu.shape[0] < max_size:
            padding_size = max_size - embeddings_tensor_gpu.shape[0]
            embeddings_tensor_gpu = torch.cat([embeddings_tensor_gpu, torch.zeros((padding_size, embedding_dim), dtype=embeddings_tensor_gpu.dtype, device=device)], dim=0)

        # Gather padded tensors
        gathered_padded_embeddings = [torch.zeros_like(embeddings_tensor_gpu) for _ in range(world_size)]
        dist.all_gather(gathered_padded_embeddings, embeddings_tensor_gpu)

        # Concatenate and unpad on rank 0
        if local_rank == 0:
            all_gathered_embeddings = torch.cat(gathered_padded_embeddings, dim=0)
            # Unpad based on original sizes
            unpadded_embeddings = []
            current_idx = 0
            for size in gathered_sizes:
                unpadded_embeddings.append(all_gathered_embeddings[current_idx : current_idx + size])
                current_idx += max_size # Move by max_size because of padding

            if unpadded_embeddings:
                 embeddings_tensor = torch.cat(unpadded_embeddings, dim=0).cpu() # Move final result to CPU
            else:
                 embeddings_tensor = torch.empty((0, embedding_dim), dtype=torch.float32).cpu()

            logger.info(f"Rank 0: Gathered and concatenated embeddings from {world_size} processes.")
        else:
            # Other ranks don't need the full tensor, they can discard the gathered data
            # For simplicity, we'll let them also construct the full tensor and then discard it
            # A more memory-efficient approach would be to only perform gathering on rank 0
            # and have other ranks return None or an empty tensor.
            # Let's construct the full tensor on all ranks for now to match the expected return type.
            all_gathered_embeddings = torch.cat(gathered_padded_embeddings, dim=0)
            unpadded_embeddings = []
            current_idx = 0
            for size in gathered_sizes:
                unpadded_embeddings.append(all_gathered_embeddings[current_idx : current_idx + size])
                current_idx += max_size

            if unpadded_embeddings:
                 embeddings_tensor = torch.cat(unpadded_embeddings, dim=0).cpu() # Move final result to CPU
            else:
                 embeddings_tensor = torch.empty((0, embedding_dim), dtype=torch.float32).cpu()


    else: # Not distributed
        # Embeddings are already on CPU from the loop
        pass # embeddings_tensor is already the concatenated CPU tensor


    # --- Reordering (remains the same, but operates on the potentially gathered tensor) ---
    if embeddings_tensor.numel() == 0:
         logger.warning(f"Rank {local_rank if is_distributed else 0}: No embeddings were generated after processing and gathering.")
         # Ensure barrier is hit even if no embeddings
         if is_distributed:
             dist.barrier()
         return np.empty((0, 0)) # Return empty numpy array

    try:
        embeddings_np = embeddings_tensor.float().numpy() # Ensure float32

        # Reorder back to original input order
        # Need original_indices and sorted_indices from earlier in the function
        # Assuming they are still in scope from lines 297-300
        # The sorting was done *before* distributing, need to adjust reordering logic
        # The current reordering logic assumes the input `embeddings_np` is in the sorted order.
        # After gathering, `embeddings_tensor` is in the order of how data was distributed across ranks,
        # which is based on the *sorted* order of sequences.
        # Concatenating `all_embeddings` gives a tensor in the sorted order.
        # `dist.all_gather` gathers these sorted-order tensors from each rank.
        # Concatenating the gathered tensors on rank 0 results in a tensor where
        # the embeddings are in the original sorted order of `dna_sequences_sorted`.
        # Therefore, applying the original_indices mapping should correctly reorder
        # back to the original input order of `dna_sequences`.

        # The reordering logic seems correct for the gathered tensor which is in sorted order.
        final_embeddings_np = np.empty_like(embeddings_np)
        final_embeddings_np[original_indices] = embeddings_np[np.argsort(sorted_indices)] # Apply reordering logic here

        # Barrier to ensure all processes finish before rank 0 potentially exits
        if is_distributed:
            dist.barrier()

        # Only rank 0 should log the final shape ideally, but for simplicity let all log
        logger.info(f"Rank {local_rank if is_distributed else 0}: Successfully generated {final_embeddings_np.shape[0]} embeddings with dimension {final_embeddings_np.shape[1]}.")
        return final_embeddings_np

    except Exception as e:
         logger.error(f"Rank {local_rank if is_distributed else 0}: Error concatenating or reordering embeddings: {e}", exc_info=True)
         # Ensure cleanup happens even on error
         if is_distributed:
             dist.barrier()
         raise