"""Detect nearly uniform images in a Parquet dataset.

Convert each image to grayscale and compare its pixel standard deviation with
the configured threshold (default: 5). Also detect extreme mean intensities
below 5 or above 250.

Examples:
    python -m mira.data.preparation.check_dataset_corrupt --parquet_path train.parquet
    python -m mira.data.preparation.check_dataset_corrupt --parquet_path train.parquet --std_threshold 8 --save_indices bad_indices.txt"""

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
    raise ImportError("Install Pillow with: pip install Pillow")


# ─────────────────────────────────────────────────────────────────────────────
# Image extraction.
# ─────────────────────────────────────────────────────────────────────────────

def extract_image(images_field) -> Image.Image | None:
    """Extract the first PIL image from a Parquet row image field."""
    if images_field is None:
        return None

    imgs = images_field
    if not isinstance(imgs, (list, np.ndarray)):
        imgs = [imgs]
    if len(imgs) == 0:
        return None

    img = imgs[0]

    # Serialized Hugging Face image dictionary.
    if isinstance(img, dict) and 'bytes' in img and img['bytes']:
        try:
            return Image.open(io.BytesIO(img['bytes'])).convert('RGB')
        except Exception:
            return None

    # Already decoded as a PIL image.
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
# Image corruption checks.
# ─────────────────────────────────────────────────────────────────────────────

def is_corrupt(img: Image.Image, std_threshold: float = 5.0) -> tuple[bool, str]:
    """Return whether an image is nearly uniform and the reason for that decision."""
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
                        help='Parquet file to inspect, such as train.parquet or test.parquet.')
    parser.add_argument('--std_threshold', type=float, default=5.0,
                        help='Flag images with a grayscale standard deviation below this threshold (default: 5).')
    parser.add_argument('--save_indices', type=str, default=None,
                        help='Optional output path for indices of flagged samples.')
    parser.add_argument('--show_examples', type=int, default=10,
                        help='Display details for the first N flagged samples.')
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
            # Include a question excerpt to help identify the sample.
            extra = row.get('extra_info', {})
            if isinstance(extra, str):
                try:
                    extra = json.loads(extra)
                except Exception:
                    extra = {}
            q = str(extra.get('question', ''))[:60] if isinstance(extra, dict) else ''
            bad_rows.append((i, reason, q))

    # Summary statistics.
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
