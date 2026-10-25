import argparse
from pathlib import Path
from datasets import load_dataset, Dataset, concatenate_datasets
import copy
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser(description="Create clean counterparts of augmented samples and append them to the dataset.")
    parser.add_argument("--input_path", type=str, required=True, help="Input Parquet path, for example data/train.parquet.")
    parser.add_argument("--output_path", type=str, required=True, help="Output Parquet path, for example data/train_augmented.parquet.")
    return parser.parse_args()

def process_dataset(input_path: str, output_path: str):
    print(f"[*] Loading dataset: {input_path}")
    # Load a single Parquet file.
    dataset = load_dataset("parquet", data_files=input_path, split="train")
    
    print(f"[*] Original sample count: {len(dataset)}")
    
    # Count the original sample types.
    types = [item.get("extra_info", {}).get("type", -1) for item in dataset]
    counter = Counter(types)
    print("\n[-] Original sample-type distribution:")
    for k, v in counter.items():
        print(f"    Type {k}: {v} samples")

    # Select augmented samples to copy as clean counterparts.
    type_1_samples = dataset.filter(lambda x: x.get("extra_info", {}).get("type") == 1)
    print(f"\n[*] Creating clean counterparts for {len(type_1_samples)} augmented samples...")

    def convert_to_type_0(example):
        # Deep-copy the sample to avoid modifying the original.
        new_example = copy.deepcopy(example)
        
        # Remove the final assistant turn to restore a clean prompt.
        prompt = new_example["prompt"]
        if prompt and prompt[-1]["role"] == "assistant":
            prompt = prompt[:-1]
        new_example["prompt"] = prompt
        
        # Clear metadata associated with misleading captions.
        extra_info = new_example.get("extra_info", {})
        extra_info["type"] = 0
        extra_info["caption_modified"] = None
        extra_info["caption_correct"] = None
        extra_info["modify_info"] = None
        new_example["extra_info"] = extra_info
        
        return new_example

    # Convert each copied augmented sample to a clean sample.
    augmented_samples = type_1_samples.map(convert_to_type_0, batched=False, desc="Converting samples")

    # Append the clean counterparts to the original dataset.
    final_dataset = concatenate_datasets([dataset, augmented_samples])
    
    # Shuffle so related samples do not remain adjacent.
    print("[*] Shuffling the augmented dataset...")
    final_dataset = final_dataset.shuffle(seed=42)

    # Count the resulting sample types.
    new_types = [item.get("extra_info", {}).get("type", -1) for item in final_dataset]
    new_counter = Counter(new_types)
    print("\n[-] Resulting sample-type distribution:")
    for k, v in new_counter.items():
        print(f"    Type {k}: {v} samples")
        
    print(f"\n[*] Resulting sample count: {len(final_dataset)}")

    # Save the output.
    print(f"[*] Saving to: {output_path}")
    final_dataset.to_parquet(output_path)
    print("[OK] Processing complete.")

if __name__ == "__main__":
    args = parse_args()
    process_dataset(args.input_path, args.output_path)
