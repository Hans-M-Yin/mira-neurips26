import argparse
from pathlib import Path
from datasets import load_dataset, Dataset, concatenate_datasets
import copy
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser(description="统计并增强预处理后的数据集，复制 Type 1 样本作为无 Caption 对照组 (Type 0)")
    parser.add_argument("--input_path", type=str, required=True, help="原始 Parquet 文件的路径，例如 ./data/train.parquet")
    parser.add_argument("--output_path", type=str, required=True, help="保存的新 Parquet 文件的路径，例如 ./data/train_augmented.parquet")
    return parser.parse_args()

def process_dataset(input_path: str, output_path: str):
    print(f"[*] 正在加载数据集: {input_path}")
    # 这里处理单文件 parquet 格式
    dataset = load_dataset("parquet", data_files=input_path, split="train")
    
    print(f"[*] 原始数据集总数: {len(dataset)}")
    
    # 统计原有类别分布
    types = [item.get("extra_info", {}).get("type", -1) for item in dataset]
    counter = Counter(types)
    print("\n[-] 增强前的数据类别分布:")
    for k, v in counter.items():
        print(f"    Type {k}: {v} 条")

    # 抽取出来需要转换为 Type 0 的 Type 1 数据
    type_1_samples = dataset.filter(lambda x: x.get("extra_info", {}).get("type") == 1)
    print(f"\n[*] 准备复制并转换 {len(type_1_samples)} 条 Type 1 样本...")

    def convert_to_type_0(example):
        # 深拷贝以防意外修改
        new_example = copy.deepcopy(example)
        
        # 1. 调整 prompt (摘掉最后一个 assistant role，即强制置空的 Zero-shot 状态)
        prompt = new_example["prompt"]
        if prompt and prompt[-1]["role"] == "assistant":
            prompt = prompt[:-1]
        new_example["prompt"] = prompt
        
        # 2. 修改 extra_info，抹掉与错 caption 相关的指引字段
        extra_info = new_example.get("extra_info", {})
        extra_info["type"] = 0
        extra_info["caption_modified"] = None
        extra_info["caption_correct"] = None
        extra_info["modify_info"] = None
        new_example["extra_info"] = extra_info
        
        return new_example

    # 对所有 Type 1 执行映射清洗，化身 Type 0
    augmented_samples = type_1_samples.map(convert_to_type_0, batched=False, desc="转换进度")

    # 把衍生出来的无 Caption 数据拼接回原数据集中
    final_dataset = concatenate_datasets([dataset, augmented_samples])
    
    # 重新洗牌打乱顺序，防止同源数据挨在一块产生局部过拟合
    print("[*] 正在打乱增强后的数据集...")
    final_dataset = final_dataset.shuffle(seed=42)

    # 统计最新类别分布
    new_types = [item.get("extra_info", {}).get("type", -1) for item in final_dataset]
    new_counter = Counter(new_types)
    print("\n[-] 增强后的数据类别分布:")
    for k, v in new_counter.items():
        print(f"    Type {k}: {v} 条")
        
    print(f"\n[*] 增强后数据集总数: {len(final_dataset)}")

    # 保存
    print(f"[*] 正在保存到: {output_path}")
    final_dataset.to_parquet(output_path)
    print("[OK] 处理完成！")

if __name__ == "__main__":
    args = parse_args()
    process_dataset(args.input_path, args.output_path)
