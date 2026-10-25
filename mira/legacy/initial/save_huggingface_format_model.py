#!/usr/bin/env python
# encoding: utf-8
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoProcessor
import torch
import fire
from glob import glob
from collections import defaultdict
import os
import re


def main(fsdp_checkpoint_path, huggingface_model_path, output_path, dtype="float16"):
    """Merge FSDP checkpoint shards into Hugging Face format.

    Args:
        fsdp_checkpoint_path: Local directory containing distributed checkpoint shards.
        huggingface_model_path: Base model used to load the configuration and tokenizer.
        output_path: Destination directory.
        dtype: Requested model dtype: float16, float32, or bfloat16."""
    # Import distributed tensor support when loading DTensor checkpoints.
    import torch.distributed.tensor as dtensor

    state_dict = defaultdict(list)

    # Infer the distributed world size.
    checkpoint_files = glob(f"{fsdp_checkpoint_path}/model_world_size_*_rank_*.pt")
    if not checkpoint_files:
        # Try alternative checkpoint filename patterns.
        checkpoint_files = glob(f"{fsdp_checkpoint_path}/*_rank_*.pt")
        if not checkpoint_files:
            raise ValueError(f"No checkpoint files found in {fsdp_checkpoint_path}")

    # Extract world size from the checkpoint filename.
    world_size = 0
    rank_files = {}

    for filepath in checkpoint_files:
        # Try the supported filename patterns.
        patterns = [
            r'model_world_size_(\d+)_rank_(\d+)\.pt',
            r'checkpoint_(\d+)_rank_(\d+)\.pt',
            r'rank_(\d+)_of_(\d+)\.pt',
            r'model_rank_(\d+)_world_size_(\d+)\.pt'
        ]

        match = None
        for pattern in patterns:
            match = re.search(pattern, os.path.basename(filepath))
            if match:
                break

        if match:
            if 'world_size' in filepath or 'of' in filepath:
                found_world_size = int(match.group(1))
                found_rank = int(match.group(2))
            else:
                found_world_size = int(match.group(2))
                found_rank = int(match.group(1))

            if world_size == 0:
                world_size = found_world_size
            elif world_size != found_world_size:
                print(f"Warning: inconsistent world sizes: {world_size} and {found_world_size}")

            rank_files[found_rank] = filepath

    if world_size == 0:
        # Fall back to the number of shard files if filenames do not encode world size.
        world_size = len(checkpoint_files)
        print(f"Warning: could not infer world size from filenames; assuming {world_size}")
        rank_files = {i: checkpoint_files[i] for i in range(world_size)}

    print(f"Detected world size: {world_size}")

    # Load shards in rank order.
    for rank in sorted(rank_files.keys()):
        filepath = rank_files[rank]
        print(f'Loading rank {rank}: {filepath}')

        # Allow distributed tensor objects when loading trusted local checkpoints.
        try:
            this_state_dict = torch.load(filepath, weights_only=False, map_location='cpu')
        except Exception as e:
            print(f"Loading failed: {e}")
            raise

        for key, value in this_state_dict.items():
            # Convert DTensor values to local tensors.
            if hasattr(value, 'to_local'):
                value = value.to_local()

            # Move the values to CPU.
            if value.device != torch.device('cpu'):
                value = value.cpu()

            state_dict[key].append(value)

    # Check the number of shards for each parameter.
    for key, values in list(state_dict.items()):
        if len(values) != world_size:
            print(f"Warning: parameter {key} has {len(values)} shards, expected world size {world_size}")

    # Merge parameter shards.
    merged_state_dict = {}
    for key in list(state_dict.keys()):
        values = state_dict[key]
        if len(values) > 1:
            # Infer the concatenation dimension.
            # Linear-layer weights commonly concatenate along dimension zero.
            # Bias vectors also commonly concatenate along dimension zero.
            # Handle convolution weights according to their tensor dimensions.

            # Check whether shard shapes agree.
            shapes = [v.shape for v in values]
            if all(s == shapes[0] for s in shapes):
                # Equal shapes can indicate replicated parameters, such as normalization weights.
                # Keep one copy of a replicated parameter.
                merged_state_dict[key] = values[0]
                print(f"Parameter {key}: equal shard shapes {shapes[0]}; keeping the first copy")
            else:
                # First try concatenating along dimension zero.
                try:
                    merged_state_dict[key] = torch.cat(values, dim=0)
                    print(f"Parameter {key}: concatenated along dimension zero, shape {merged_state_dict[key].shape}")
                except RuntimeError as e:
                    # If that fails, try dimension one.
                    try:
                        merged_state_dict[key] = torch.cat(values, dim=1)
                        print(f"Parameter {key}: concatenated along dimension one, shape {merged_state_dict[key].shape}")
                    except RuntimeError:
                        print(f"Warning: could not concatenate parameter {key}; shard shapes: {shapes}")
                        # Keep the unmerged shards if no concatenation is valid.
                        for i, v in enumerate(values):
                            merged_state_dict[f"{key}_shard_{i}"] = v
        else:
            merged_state_dict[key] = values[0]

    # Load the model configuration.
    print(f"Loading configuration from {huggingface_model_path}...")
    config = AutoConfig.from_pretrained(huggingface_model_path)

    # Select the requested dtype.
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "float32":
        torch_dtype = torch.float32
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Use the automatic model loader for Qwen2.5-VL.
    print("Creating the Qwen2.5-VL model...")
    try:
        # Try loading the original model.
        model = AutoModel.from_pretrained(
            huggingface_model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True
        )

        # Load the merged state dictionary.
        missing_keys, unexpected_keys = model.load_state_dict(merged_state_dict, strict=False)

        if missing_keys:
            print(f"Warning: missing keys ({len(missing_keys)}):")
            for i, key in enumerate(missing_keys[:10]):  # Display only the first ten keys.
                print(f"  {key}")
            if len(missing_keys) > 10:
                print(f"  ... and {len(missing_keys) - 10} more")

        if unexpected_keys:
            print(f"Warning: unexpected keys ({len(unexpected_keys)}):")
            for i, key in enumerate(unexpected_keys[:10]):  # Display only the first ten keys.
                print(f"  {key}")
            if len(unexpected_keys) > 10:
                print(f"  ... and {len(unexpected_keys) - 10} more")

    except Exception as e:
        print(f"Automatic model loading failed: {e}")
        print("Attempting to create the model directly from its configuration...")

        # Fall back to constructing a model from its configuration.
        model = AutoModel.from_config(config, torch_dtype=torch_dtype)
        missing_keys, unexpected_keys = model.load_state_dict(merged_state_dict, strict=False)

        if missing_keys:
            print(f"Warning: missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: unexpected keys: {unexpected_keys}")

    # Save the merged model.
    print(f"Saving the model to {output_path}...")
    model.save_pretrained(
        output_path,
        max_shard_size="10GB",
        safe_serialization=True
    )
    print(f"Model saved to: {output_path}")

    # Save the tokenizer and processor.
    print("Saving tokenizer and processor...")
    try:
        # Vision-language models typically require a processor.
        processor = AutoProcessor.from_pretrained(huggingface_model_path, trust_remote_code=True)
        processor.save_pretrained(output_path)
        print(f"Processor saved to: {output_path}")
    except:
        # If processor loading fails, save the tokenizer only.
        tokenizer = AutoTokenizer.from_pretrained(huggingface_model_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_path)
        print(f"Tokenizer saved to: {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
