"""
统计 mix_ablation_datasets.py 生成的 dataset 中各 type 的数据数量。

用法示例:
  # 统计单个 parquet 文件
  python data/count_dataset_types.py --path ./ablation_mix/ratio_050/train.parquet

  # 统计整个输出目录下所有子目录的 parquet 文件（批量）
  python data/count_dataset_types.py --dir ./ablation_mix

  # 打印每种 type 的详细信息
  python data/count_dataset_types.py --path ./ablation_mix/ratio_050/train.parquet --verbose
"""

import argparse
import json
import os
import sys
from collections import Counter

import pandas as pd
from datasets import load_dataset

# type 含义映射（与 mix_ablation_datasets.py 保持一致）
TYPE_LABELS = {
    0: "Normal (Type 0)",
    1: "Wrong Caption / Adversarial (Type 1)",
}


def get_type(extra_info) -> int:
    """从 extra_info 字段提取 type 值。"""
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except json.JSONDecodeError:
            return -1
    if isinstance(extra_info, dict):
        return extra_info.get("type", -1)
    return -1


def _extract_prompt_text(prompt_field) -> str:
    """从 prompt 列表（messages 格式）中提取 user 消息的文本。"""
    if isinstance(prompt_field, list):
        for msg in prompt_field:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content.strip()
    if isinstance(prompt_field, str):
        return prompt_field.strip()
    return str(prompt_field)


def print_sample_previews(df: pd.DataFrame, n: int = 5, target_type: int | None = None) -> None:
    """打印前 n 个样本（可按 type 过滤）的 type 和 prompt 内容（截断显示）。

    Args:
        df: 完整的 DataFrame。
        n:  最多显示几个样本。
        target_type: 若指定，则只看 type == target_type 的行；None 表示不过滤。
    """

    if target_type is None:
        subset = df.head(n)
        title = f"SAMPLE PREVIEW (first {min(n, len(df))} samples)"
    else:
        # 过滤出目标 type 的行
        mask = df["extra_info"].apply(lambda x: get_type(x) == target_type)
        filtered = df[mask]
        found = len(filtered)
        subset = filtered.head(n)
        label = TYPE_LABELS.get(target_type, f"Type {target_type}")
        title = f"SAMPLE PREVIEW — {label} (type={target_type})"
        if found < n:
            title += f"  [WARNING: only {found} sample(s) found, fewer than {n}]"
        else:
            title += f"  (first {n} of {found})"

    print(f"\n{'─' * 60}")
    print(title)
    print(f"{'─' * 60}")

    if len(subset) == 0:
        print("  (no samples to display)")
        print(f"{'─' * 60}")
        return

    for i, (_, row) in enumerate(subset.iterrows()):
        sample_type = get_type(row.get("extra_info", {}))
        type_label  = TYPE_LABELS.get(sample_type, f"Unknown (Type {sample_type})")
        prompt_raw  = row.get("prompt", row.get("question", ""))
        prompt_text = _extract_prompt_text(prompt_raw)
        print(f"\n[Sample {i}]  type={sample_type} ({type_label})")
        print(f"  Prompt: {prompt_text}")
    print(f"{'─' * 60}")


def count_types_in_parquet(parquet_path: str, verbose: bool = False) -> Counter:
    """加载 parquet 文件并统计各 type 的数量。"""
    print(f"\n{'='*60}")
    print(f"File : {parquet_path}")
    print(f"{'='*60}")

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"  [ERROR] Failed to load file: {e}", file=sys.stderr)
        return Counter()

    total = len(df)
    print(f"Total samples: {total}")

    if "extra_info" not in df.columns:
        print("  [ERROR] Column 'extra_info' not found in dataset.", file=sys.stderr)
        return Counter()

    type_counter: Counter = Counter()
    for _, row in df.iterrows():
        t = get_type(row["extra_info"])
        type_counter[t] += 1

    # 打印结果
    print(f"\n{'Type':<10} {'Label':<45} {'Count':>8} {'Ratio':>8}")
    print("-" * 75)
    for t in sorted(type_counter.keys()):
        label = TYPE_LABELS.get(t, f"Unknown (Type {t})")
        count = type_counter[t]
        ratio = count / total if total > 0 else 0.0
        print(f"{t:<10} {label:<45} {count:>8,} {ratio:>8.2%}")
    print("-" * 75)
    print(f"{'TOTAL':<56} {total:>8,}")

    if verbose:
        print("\n[Verbose] type value distribution (raw Counter):")
        for t, c in sorted(type_counter.items()):
            print(f"  type={t!r}: {c}")

    print_sample_previews(df)
    print_sample_previews(df, n=5, target_type=0)
    return type_counter


def count_types_in_dir(directory: str, verbose: bool = False):
    """遍历目录下所有子目录，查找 train.parquet 并批量统计。"""
    found = []
    for root, dirs, files in os.walk(directory):
        for fname in files:
            if fname.endswith(".parquet"):
                found.append(os.path.join(root, fname))

    if not found:
        print(f"[WARNING] No .parquet files found under: {directory}", file=sys.stderr)
        return

    found.sort()
    print(f"Found {len(found)} parquet file(s) under '{directory}'.")

    summary_rows = []
    for path in found:
        counter = count_types_in_parquet(path, verbose=verbose)
        total = sum(counter.values())
        summary_rows.append({"path": path, "counter": counter, "total": total})

    # 汇总表格
    print(f"\n\n{'='*60}")
    print("SUMMARY ACROSS ALL FILES")
    print(f"{'='*60}")

    # 收集所有出现过的 type
    all_types = sorted(set(t for row in summary_rows for t in row["counter"].keys()))

    header = f"{'File':<35}" + "".join(
        f"  {'T'+str(t):>8}" for t in all_types
    ) + f"  {'Total':>8}"
    print(header)
    print("-" * len(header))

    for row in summary_rows:
        fname = os.path.basename(os.path.dirname(row["path"])) + "/" + os.path.basename(row["path"])
        line = f"{fname:<35}"
        for t in all_types:
            line += f"  {row['counter'].get(t, 0):>8,}"
        line += f"  {row['total']:>8,}"
        print(line)


def main():
    parser = argparse.ArgumentParser(
        description="统计 mix_ablation_datasets.py 生成的 dataset 中各 type 的数据数量。"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--path",
        type=str,
        help="单个 .parquet 文件路径",
    )
    group.add_argument(
        "--dir",
        type=str,
        help="包含多个 ratio_xxx 子目录的输出目录路径（批量统计）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印每个文件的原始 type Counter",
    )
    args = parser.parse_args()

    if args.path:
        if not os.path.isfile(args.path):
            print(f"[ERROR] File not found: {args.path}", file=sys.stderr)
            sys.exit(1)
        count_types_in_parquet(args.path, verbose=args.verbose)
    else:
        if not os.path.isdir(args.dir):
            print(f"[ERROR] Directory not found: {args.dir}", file=sys.stderr)
            sys.exit(1)
        count_types_in_dir(args.dir, verbose=args.verbose)


if __name__ == "__main__":
    main()
