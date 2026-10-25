"""Validate that image placeholders match the number of stored images.

Inspect each prompt and report samples whose image placeholder count differs
from the image count.

Examples:
    python -m mira.data.preparation.check_placeholder --parquet train.parquet
    python -m mira.data.preparation.check_placeholder --parquet train.parquet --verbose"""

import argparse
import json
import re
import sys
from collections import Counter

import pandas as pd
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def count_image_placeholders(prompt_field) -> int:
    """Count image placeholders across all prompt messages."""
    if not isinstance(prompt_field, list):
        return 0
    total = 0
    for msg in prompt_field:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            total += content.count("<image>")
        elif isinstance(content, list):
            # Multimodal content contains image and text blocks.
            # Embedded image blocks do not require textual image placeholders.
            # Also count image placeholders inside text blocks.
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    total += block.get("text", "").count("<image>")
    return total


def count_actual_images(images_field) -> int:
    """Count stored images across the supported image representations."""
    if images_field is None:
        return 0
    if isinstance(images_field, list):
        # Remove missing and empty image values.
        return sum(1 for img in images_field if img is not None)
    return 1  # Treat a non-list value as a single image.


def get_source_idx(extra_info) -> int:
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except Exception:
            return -1
    if isinstance(extra_info, dict):
        return extra_info.get("source_idx", extra_info.get("idx", -1))
    return -1


def get_type(extra_info) -> int:
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except Exception:
            return -1
    if isinstance(extra_info, dict):
        return extra_info.get("type", -1)
    return -1


# ─────────────────────────────────────────────────────────────────────────────
# Main check
# ─────────────────────────────────────────────────────────────────────────────
def check_parquet(parquet_path: str, verbose: bool = False):
    print(f"\nLoading: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    total = len(df)
    print(f"Total samples: {total:,}")

    mismatches = []       # (row_idx, n_placeholders, n_images, type, source_idx)
    placeholder_dist = Counter()   # n_placeholders -> count
    image_dist = Counter()         # n_images -> count

    for row_idx, row in tqdm(df.iterrows(), total=total, desc="Checking"):
        prompt   = row.get("prompt", [])
        images   = row.get("images", [])
        extra    = row.get("extra_info", {})

        n_ph  = count_image_placeholders(prompt)
        n_img = count_actual_images(images)

        placeholder_dist[n_ph]  += 1
        image_dist[n_img]        += 1

        if n_ph != n_img:
            mismatches.append({
                "row_idx":    int(row_idx),
                "source_idx": get_source_idx(extra),
                "type":       get_type(extra),
                "n_placeholders": n_ph,
                "n_images":   n_img,
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"CONSISTENCY CHECK RESULTS")
    print(f"{'='*60}")
    print(f"  Total samples        : {total:,}")
    print(f"  Consistent samples   : {total - len(mismatches):,}")
    print(f"  MISMATCHED samples   : {len(mismatches):,}  ({len(mismatches)/total:.2%})")

    # <image> placeholder distribution
    print(f"\n<image> placeholder count distribution (in prompt):")
    for k in sorted(placeholder_dist.keys()):
        print(f"  {k} placeholder(s): {placeholder_dist[k]:,} samples")

    # actual image count distribution
    print(f"\nActual image count distribution (in 'images' field):")
    for k in sorted(image_dist.keys()):
        print(f"  {k} image(s): {image_dist[k]:,} samples")

    # mismatch breakdown
    if mismatches:
        print(f"\nMismatch breakdown:")
        pattern_counter: Counter = Counter()
        type_counter: Counter = Counter()
        for m in mismatches:
            pattern_counter[(m["n_placeholders"], m["n_images"])] += 1
            type_counter[m["type"]] += 1

        print("  (n_placeholders, n_images) -> count")
        for (ph, img), cnt in sorted(pattern_counter.items()):
            print(f"    ({ph} placeholder, {img} image): {cnt:,} samples")

        print("  By sample type:")
        for t, cnt in sorted(type_counter.items()):
            print(f"    type={t}: {cnt:,} samples")

        if verbose:
            print(f"\nMismatched sample details (first 20):")
            print(f"  {'row_idx':>8}  {'source_idx':>12}  {'type':>6}  {'#ph':>4}  {'#img':>5}")
            print(f"  {'-'*8}  {'-'*12}  {'-'*6}  {'-'*4}  {'-'*5}")
            for m in mismatches[:20]:
                print(
                    f"  {m['row_idx']:>8}  {m['source_idx']:>12}  "
                    f"{m['type']:>6}  {m['n_placeholders']:>4}  {m['n_images']:>5}"
                )
            if len(mismatches) > 20:
                print(f"  ... and {len(mismatches) - 20} more.")
    else:
        print("\n✅ All samples are consistent!")

    print(f"{'='*60}\n")
    return mismatches


def main():
    parser = argparse.ArgumentParser(
        description="Check <image> placeholder vs actual image count consistency in a parquet dataset."
    )
    parser.add_argument("--parquet", type=str, required=True, help="Path to .parquet file")
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-sample details for mismatched entries"
    )
    args = parser.parse_args()

    mismatches = check_parquet(args.parquet, verbose=args.verbose)
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
