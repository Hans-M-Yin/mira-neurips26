import argparse
import os
import pandas as pd
from datasets import Dataset, load_dataset

def process_ablation(example):
    """
    修改 user 端的 prompt，将错 caption 作为用户输入的一部分。
    """
    # 提取需要的字段
    extra_info = example.get('extra_info', {})
    _type = extra_info.get('type', 0)
    caption = extra_info.get('caption_modified', None)
    
    # 如果是生成失败的（type=0）或没有 caption 的，保持原样
    if _type == 0 or not caption:
        return example

    # 移除开头的 <think> 标签及其可能带的换行
    caption = str(caption).strip()
    if caption.startswith("<think>"):
        caption = caption[len("<think>"):].strip()

    # 提取问题 (从 extra_info 提取最安全，或者从 prompt 提取)
    question = extra_info.get('question', '')
    
    # 简单的中英文检测（检查是否包含中文字符）
    def has_chinese(text):
        return any('\u4e00' <= char <= '\u9fff' for char in str(text))
        
    if has_chinese(question):
        ablation_text = f"以上图片展示的内容是：\n{caption}\n\n图片描述可能有错，请严格根据图片内容回答问题：\n{question}"
    else:
        ablation_text = f"The content shown in the image above is:\n{caption}\n\nThe image description might contain errors. Please answer the question strictly based on the visual content:\n{question}"
    
    # 获取原来的 prompt，并剔除所有已经被强插进去的 assistant 消息
    prompt = [m for m in example['prompt'] if m['role'] != 'assistant']
    
    # 修改 prompt 中的 user text
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
    # 关键修改：重置 type 为 0，欺骗 agent_loop，避免它再次把 caption 作为 assistant 强插进去
    example['extra_info']['type'] = 0
    return example

def main():
    parser = argparse.ArgumentParser(description="Ablation Study: Move corrupted caption to user prompt.")
    parser.add_argument("--input_parquet", type=str, required=True, help="Path to the original train.parquet")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the ablated parquet")
    args = parser.parse_args()

    print(f"Loading dataset from: {args.input_parquet}")
    # 读取原始 parquet
    dataset = load_dataset("parquet", data_files=args.input_parquet, split="train")
    
    print(f"Total samples: {len(dataset)}")
    
    print("Applying ablation prompt modifications...")
    # Map 操作修改 prompt
    ablated_dataset = dataset.map(process_ablation, num_proc=8)
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "train.parquet")
    ablated_dataset.to_parquet(out_path)
    
    print(f"✅ Ablation dataset saved to: {out_path}")
    
    # 打印一条 sample 检查一下
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
