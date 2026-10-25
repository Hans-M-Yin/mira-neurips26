import pandas as pd
import argparse
import os
import json
import random
import io
import copy
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Corrupt Image Detection Logic (from check_corrupt_images.py)
# ─────────────────────────────────────────────────────────────────────────────

def extract_image(images_field) -> Image.Image | None:
    if images_field is None: return None
    imgs = images_field
    if not isinstance(imgs, (list, np.ndarray)): imgs = [imgs]
    if len(imgs) == 0: return None
    img = imgs[0]
    # bytes dict (HuggingFace datasets.Image serialization)
    if isinstance(img, dict) and 'bytes' in img and img['bytes']:
        try: return Image.open(io.BytesIO(img['bytes'])).convert('RGB')
        except: return None
    # PIL.Image
    if hasattr(img, 'convert'):
        try: return img.convert('RGB')
        except: return None
    # raw bytes
    if isinstance(img, (bytes, bytearray)) and len(img) > 0:
        try: return Image.open(io.BytesIO(img)).convert('RGB')
        except: return None
    return None

def is_corrupt(img: Image.Image, std_threshold: float = 5.0) -> bool:
    """Detect if image is pure black, pure white, or near-uniform."""
    gray = np.array(img.convert('L'), dtype=np.float32)
    std  = float(gray.std())
    mean = float(gray.mean())
    if std < std_threshold:
        if mean < 5 or mean > 250: return True # Pure Black/White
        return True # Near uniform
    return False

"""
python data/mix_ablation_datasets.py \
    --path_a ./ds_gpt4o/train.parquet \
    --path_b ./ds_qwen30b/train.parquet \
    --path_c ./ds_local_model/train.parquet \
    --ratios 0.0,0.2,0.4,0.6,0.8,1.0 \
    --output_dir ./per_ratio_experiment

"""

def get_source_idx(extra_info):
    if isinstance(extra_info, str):
        extra_info = json.loads(extra_info)
    if not isinstance(extra_info, dict):
        return -1
    # Current `data_preprocess.py` writes `idx`, while older variants may write
    # `source_idx`. Support both to avoid silently dropping all samples.
    return extra_info.get("source_idx", extra_info.get("idx", -1))

def get_type(extra_info):
    if isinstance(extra_info, str):
        extra_info = json.loads(extra_info)
    if not isinstance(extra_info, dict):
        return 0
    return extra_info.get("type", 0)

def validate_ratios(ratios):
    for ratio in ratios:
        if ratio < 0 or ratio > 1:
            raise ValueError(f"Ratio must be in [0, 1], but got {ratio}")

def count_types(samples):
    counts = {}
    for sample in samples:
        sample_type = get_type(sample.get("extra_info", {}))
        counts[sample_type] = counts.get(sample_type, 0) + 1
    return counts

def convert_to_normal(sample):
    """将样本降级为普通样本(Type 0)"""
    new_sample = copy.deepcopy(sample)
    ex = new_sample['extra_info']
    if isinstance(ex, str):
        ex = json.loads(ex)
    
    # 强制修改状态
    ex['type'] = 0
    # 保留 caption_correct 便于追溯，但把 wrong-caption 相关信息清空，避免语义不一致
    ex['caption_modified'] = None
    ex['modify_info'] = None
    ex['num_modify'] = 0
    new_sample['extra_info'] = ex
    
    # 兼容历史顶层字段；如果存在也一起清掉，避免 schema 脏数据
    if 'caption_modified' in new_sample:
        new_sample['caption_modified'] = None
    if 'modify_info' in new_sample:
        new_sample['modify_info'] = None
    if 'num_modify' in new_sample:
        new_sample['num_modify'] = 0
    
    # 还原 Prompt (如果是 assistant prefill 模式，需要把插入的文本删掉)
    prompt = new_sample.get('prompt', [])
    if isinstance(prompt, list):
        new_sample['prompt'] = [copy.deepcopy(m) for m in prompt if m.get('role') != 'assistant']
    
    return new_sample

def main():
    parser = argparse.ArgumentParser(description="Merge 3 datasets (A->B->C) and generate proportional ablations.")
    parser.add_argument("--path_a", type=str, required=True, help="Priority 1 dataset (e.g. GPT-4o-mini version)")
    parser.add_argument("--path_b", type=str, required=True, help="Priority 2 dataset")
    parser.add_argument("--path_c", type=str, required=True, help="Priority 3 dataset")
    parser.add_argument("--ratios", type=str, default="0.1,0.3,0.5,0.7,1.0", help="Comma separated float list")
    parser.add_argument("--output_dir", type=str, default="./ablation_mix")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ratios = [float(r) for r in args.ratios.split(",")]
    validate_ratios(ratios)
    random.seed(args.seed)
    
    print("Loading datasets...")
    ds_list = [
        load_dataset("parquet", data_files=p, split="train").to_pandas() 
        for p in [args.path_a, args.path_b, args.path_c]
    ]

    # 1. 建立基于 source_idx 的查找字典
    print("Building lookup index and filtering corrupt images...")
    lookup = [{}, {}, {}]
    all_source_indices = set()
    corrupt_counts = [0, 0, 0]
    
    for i, df in enumerate(ds_list):
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Indexing & Filtering DS {i}"):
            # ── 过滤损坏图片 ──
            img = extract_image(row.get('images'))
            if img is None or is_corrupt(img):
                corrupt_counts[i] += 1
                continue

            s_idx = get_source_idx(row['extra_info'])
            if s_idx != -1:
                row_dict = row.to_dict()
                # ── 关键修复：确保存储的是解码后的 PIL 对象列表 ──
                # transformers 期待 [PIL.Image, ...]，而不是 HuggingFace 的镜像字典
                row_dict['images'] = [img] 
                lookup[i][s_idx] = row_dict
                all_source_indices.add(s_idx)
    
    for i, count in enumerate(corrupt_counts):
        if count > 0:
            print(f"  - Dataset {['A','B','C'][i]}: Filtered {count} corrupt/missing images.")

    if not all_source_indices:
        raise RuntimeError(
            "No valid source indices were found. "
            "Please check whether your parquet files contain `extra_info.idx` or `extra_info.source_idx`."
        )

    # 2. 优先级合并核心逻辑
    print(f"Merging with Priority A -> B -> C (Total unique samples: {len(all_source_indices)})")
    golden_wrong_pool = []
    
    stats = {"from_a": 0, "from_b": 0, "from_c": 0, "failed": 0}
    
    for s_idx in tqdm(sorted(list(all_source_indices)), desc="Matching Priorities"):
        target_sample = None
        # 按 A -> B -> C 顺序找第一个 type=1 的
        for i in range(3):
            candidate = lookup[i].get(s_idx)
            if candidate and get_type(candidate['extra_info']) == 1:
                target_sample = candidate
                stats[f"from_{['a','b','c'][i]}"] += 1
                break
        
        if target_sample:
            golden_wrong_pool.append(target_sample)
        else:
            stats["failed"] += 1

    print("\n" + "="*40)
    print("GOLDEN POOL STATISTICS (Samples with at least one type=1 candidate)")
    print("="*40)
    print(f"Total Source Indices  : {len(all_source_indices)}")
    print(f"Successfully Merged   : {len(golden_wrong_pool)}")
    print(f"  - From Dataset A    : {stats['from_a']}")
    print(f"  - From Dataset B    : {stats['from_b']}")
    print(f"  - From Dataset C    : {stats['from_c']}")
    print(f"Discarded (All Normal): {stats['failed']}")
    print("="*40)

    if not golden_wrong_pool:
        raise RuntimeError(
            "The merged wrong-caption pool is empty. "
            "That usually means none of the three datasets contains `type == 1` samples "
            "for the shared indices."
        )

    # 3. 按比例生成
    os.makedirs(args.output_dir, exist_ok=True)
    shuffled_pool = copy.deepcopy(golden_wrong_pool)
    random.shuffle(shuffled_pool) # 随机打乱以保证不同比例间的样本包含关系是平滑的

    for r in ratios:
        print(f"\nGenerating dataset for ratio: {r:.1%}")
        num_wrong = int(len(shuffled_pool) * r)
        
        final_list = []
        for i, sample in enumerate(shuffled_pool):
            if i < num_wrong:
                # 保持为 Wrong Caption
                final_list.append(copy.deepcopy(sample))
            else:
                # 转换为 Normal
                final_list.append(convert_to_normal(sample))
        
        # 转换为 HF Dataset 以保存为 Parquet
        out_ds = Dataset.from_list(final_list)
        
        # 修复特征类型冲突（如果保存报错，通常是因为 None 值导致推断错误）
        # 这里统一将 extra_info 转回 dict
        def finalize(example):
            if isinstance(example['extra_info'], str):
                example['extra_info'] = json.loads(example['extra_info'])
            
            # 确保 images 是列表格式且不包含残留的 HF 字典
            if example.get('images') is not None:
                # 如果是单张图，包装成列表
                if not isinstance(example['images'], list):
                    example['images'] = [example['images']]
            return example
        out_ds = out_ds.map(finalize)

        save_subdir = os.path.join(args.output_dir, f"ratio_{int(r*100):03d}")
        os.makedirs(save_subdir, exist_ok=True)
        out_ds.to_parquet(os.path.join(save_subdir, "train.parquet"))
        actual_type_counts = count_types(final_list)
        actual_wrong = actual_type_counts.get(1, 0)
        actual_normal = actual_type_counts.get(0, 0)
        stats_payload = {
            "ratio": r,
            "expected_wrong": num_wrong,
            "expected_normal": len(final_list) - num_wrong,
            "actual_type_counts": actual_type_counts,
            "total_samples": len(final_list),
        }
        with open(os.path.join(save_subdir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats_payload, f, ensure_ascii=False, indent=2)

        print(f"  -> Saved {len(final_list)} items to {save_subdir}")
        print(f"  -> Dist: expected {num_wrong} Wrong / {len(final_list)-num_wrong} Normal")
        print(f"  -> Dist: actual   {actual_wrong} Wrong / {actual_normal} Normal")
        if actual_wrong != num_wrong:
            raise RuntimeError(
                f"Type distribution mismatch for ratio={r}: expected {num_wrong} wrong samples, "
                f"but found {actual_wrong} after conversion."
            )

if __name__ == "__main__":
    main()
