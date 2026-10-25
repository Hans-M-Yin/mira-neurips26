import argparse
import os
import pandas as pd
from datasets import Dataset, load_dataset

def process_ablation(example):
    """Move the misleading caption into the user message."""
    # Extract the required fields.
    extra_info = example.get('extra_info', {})
    _type = extra_info.get('type', 0)
    caption = extra_info.get('caption_modified', None)
    
    # Keep normal samples and samples without a caption unchanged.
    if _type == 0 or not caption:
        return example

    # Remove the initial reasoning tag and any following newline.
    caption = str(caption).strip()
    if caption.startswith("<think>"):
        caption = caption[len("<think>"):].strip()

    # Read the question from metadata, falling back to the prompt.
    question = extra_info.get('question', '')
    
    # Choose the prompt language based on the presence of Chinese characters.
    def has_chinese(text):
        return any('\u4e00' <= char <= '\u9fff' for char in str(text))
        
    if has_chinese(question):
        ablation_text = f"以上图片展示的内容是：\n{caption}\n\n图片描述可能有错，请严格根据图片内容回答问题：\n{question}"
    else:
        ablation_text = f"The content shown in the image above is:\n{caption}\n\nThe image description might contain errors. Please answer the question strictly based on the visual content:\n{question}"
    
    # Remove assistant turns previously inserted into the prompt.
    prompt = [m for m in example['prompt'] if m['role'] != 'assistant']
    
    # Update the user message.
    for message in prompt:
        if message['role'] == 'user':
            if isinstance(message['content'], str):
                message['content'] = ablation_text
            else:
                for content_item in message['content']:
                    if content_item['type'] == 'text':
                        content_item['text'] = ablation_text
                        break
    
    example['prompt'] = prompt
    # Reset the sample type so the agent loop does not insert the caption again.
    example['extra_info']['type'] = 0
    return example

def main():
    parser = argparse.ArgumentParser(description="Ablation Study: Move corrupted caption to user prompt.")
    parser.add_argument("--input_parquet", type=str, required=True, help="Path to the original train.parquet")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the ablated parquet")
    args = parser.parse_args()

    print(f"Loading dataset from: {args.input_parquet}")
    # Load the input Parquet file.
    dataset = load_dataset("parquet", data_files=args.input_parquet, split="train")
    
    print(f"Total samples: {len(dataset)}")
    
    print("Applying ablation prompt modifications...")
    # Apply the prompt transformation.
    ablated_dataset = dataset.map(process_ablation, num_proc=8)
    
    # Save the transformed dataset.
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "train.parquet")
    ablated_dataset.to_parquet(out_path)
    
    print(f"✅ Ablation dataset saved to: {out_path}")
    
    # Display one transformed sample.
    for i in range(len(ablated_dataset)):
        if ablated_dataset[i]['extra_info']['type'] == 1:
            print("\n🔍 Example Ablated Prompt (Type=1):")
            user_msg = [m for m in ablated_dataset[i]['prompt'] if m['role'] == 'user'][0]
            for c in user_msg['content']:
                if c['type'] == 'text':
                    print(c['text'])
                    break
            break

if __name__ == "__main__":
    main()
    print("KILL MYSELF")
