"""
check_corrupt_images.py — 检测 parquet 数据集中损坏的图片（纯白/纯黑/近似纯色）

判断逻辑：
  - 将图片转为灰度，计算像素标准差 (std)
  - std < STD_THRESHOLD（默认 5）视为"近似纯色"（包括纯白/纯黑/纯灰）
  - 同时检测极端均值：mean < 5（极暗/纯黑）或 mean > 250（极亮/纯白）

用法：
  python check_corrupt_images.py --parquet_path ./train.parquet
  python check_corrupt_images.py --parquet_path ./train.parquet --std_threshold 8 --save_indices bad_indices.txt
"""

import argparse
import io
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import PIL.Image as Image
except ImportError:
    raise ImportError("请安装 Pillow: pip install Pillow")


# ─────────────────────────────────────────────────────────────────────────────
# 图片提取
# ─────────────────────────────────────────────────────────────────────────────

def extract_image(images_field) -> Image.Image | None:
    """从 parquet row 的 images 字段中提取第一张 PIL.Image。"""
    if images_field is None:
        return None

    imgs = images_field
    if not isinstance(imgs, (list, np.ndarray)):
        imgs = [imgs]
    if len(imgs) == 0:
        return None

    img = imgs[0]

    # bytes dict（HuggingFace datasets.Image 序列化格式）
    if isinstance(img, dict) and 'bytes' in img and img['bytes']:
        try:
            return Image.open(io.BytesIO(img['bytes'])).convert('RGB')
        except Exception:
            return None

    # 已经是 PIL.Image
    if hasattr(img, 'convert'):
        try:
            return img.convert('RGB')
        except Exception:
            return None

    # raw bytes
    if isinstance(img, (bytes, bytearray)) and len(img) > 0:
        try:
            return Image.open(io.BytesIO(img)).convert('RGB')
        except Exception:
            return None

    return None


# ─────────────────────────────────────────────────────────────────────────────
# 损坏判断
# ─────────────────────────────────────────────────────────────────────────────

def is_corrupt(img: Image.Image, std_threshold: float = 5.0) -> tuple[bool, str]:
    """
    判断图片是否损坏（近似纯色）。
    返回 (is_bad, reason)。
    """
    gray = np.array(img.convert('L'), dtype=np.float32)
    std  = float(gray.std())
    mean = float(gray.mean())

    if std < std_threshold and mean < 5:
        return True, f"pure_black (mean={mean:.1f}, std={std:.1f})"
    if std < std_threshold and mean > 250:
        return True, f"pure_white (mean={mean:.1f}, std={std:.1f})"
    if std < std_threshold:
        return True, f"near_uniform (mean={mean:.1f}, std={std:.1f})"
    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet_path', type=str, required=True,
                        help='待检测的 parquet 文件路径（train.parquet 或 test.parquet）')
    parser.add_argument('--std_threshold', type=float, default=5.0,
                        help='像素灰度标准差低于此值视为损坏（默认 5）')
    parser.add_argument('--save_indices', type=str, default=None,
                        help='将损坏样本的行号保存到此文件（可选）')
    parser.add_argument('--show_examples', type=int, default=10,
                        help='打印前 N 个损坏样本的详情')
    args = parser.parse_args()

    print(f"Loading: {args.parquet_path}")
    df = pd.read_parquet(args.parquet_path)
    total = len(df)
    print(f"Total samples: {total}\n")

    bad_rows = []   # [(row_idx, reason, question_snippet)]
    no_image = 0

    for i, row in tqdm(df.iterrows(), total=total, desc="Checking images"):
        img = extract_image(row.get('images'))

        if img is None:
            no_image += 1
            bad_rows.append((i, "no_image", ""))
            continue

        corrupt, reason = is_corrupt(img, std_threshold=args.std_threshold)
        if corrupt:
            # 尝试获取题目片段方便排查
            extra = row.get('extra_info', {})
            if isinstance(extra, str):
                try:
                    extra = json.loads(extra)
                except Exception:
                    extra = {}
            q = str(extra.get('question', ''))[:60] if isinstance(extra, dict) else ''
            bad_rows.append((i, reason, q))

    # ── 统计 ──
    n_bad = len(bad_rows)
    n_no_img = sum(1 for _, r, _ in bad_rows if r == "no_image")
    n_black  = sum(1 for _, r, _ in bad_rows if 'black' in r)
    n_white  = sum(1 for _, r, _ in bad_rows if 'white' in r)
    n_uniform= sum(1 for _, r, _ in bad_rows if 'uniform' in r)

    print(f"\n{'='*55}")
    print(f"  RESULT SUMMARY")
    print(f"{'='*55}")
    print(f"  Total samples          : {total}")
    print(f"  Corrupted (total)      : {n_bad}  ({n_bad/total:.2%})")
    print(f"    - No image           : {n_no_img}")
    print(f"    - Pure black         : {n_black}")
    print(f"    - Pure white         : {n_white}")
    print(f"    - Near-uniform color : {n_uniform}")
    print(f"  Std threshold used     : {args.std_threshold}")
    print(f"{'='*55}")

    if bad_rows:
        print(f"\nExamples (first {args.show_examples}):")
        for idx, reason, q in bad_rows[:args.show_examples]:
            print(f"  row {idx:5d} | {reason:40s} | Q: {q}")

    if args.save_indices and bad_rows:
        with open(args.save_indices, 'w') as f:
            for idx, reason, _ in bad_rows:
                f.write(f"{idx}\t{reason}\n")
        print(f"\nCorrupt row indices saved to: {args.save_indices}")

    print(f"\n✅ Clean samples: {total - n_bad} ({(total - n_bad)/total:.2%})")


if __name__ == '__main__':
    main()
