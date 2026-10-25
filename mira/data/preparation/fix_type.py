"""Remove trailing assistant turns from clean samples in a mixed dataset.

Inspect each sample whose metadata type is zero, remove trailing assistant
messages, verify the repair, and write the resulting Parquet file. In-place
writes create a backup; an explicit output path writes a separate file.

Examples:
    python -m mira.data.preparation.fix_type --parquet train.parquet
    python -m mira.data.preparation.fix_type --parquet train.parquet --output fixed.parquet
    python -m mira.data.preparation.fix_type --parquet train.parquet --dry-run"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter

import datasets
import pandas as pd
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_type(extra_info) -> int:
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except Exception:
            return -1
    if isinstance(extra_info, dict):
        return extra_info.get("type", -1)
    return -1


def has_trailing_assistant(prompt) -> bool:
    """Return whether the prompt ends with an assistant message."""
    if not isinstance(prompt, list) or len(prompt) == 0:
        return False
    last = prompt[-1]
    if isinstance(last, dict):
        return last.get("role") == "assistant"
    return False


def strip_trailing_assistant(prompt) -> list:
    """Remove every trailing assistant message from the prompt."""
    if not isinstance(prompt, list):
        return prompt
    result = list(prompt)
    while result and isinstance(result[-1], dict) and result[-1].get("role") == "assistant":
        result.pop()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Core fix logic (operates on a HuggingFace Dataset row)
# ─────────────────────────────────────────────────────────────────────────────
def fix_row(example: dict) -> dict:
    sample_type = get_type(example.get("extra_info", {}))
    if sample_type != 0:
        return example

    prompt = example.get("prompt", [])
    if has_trailing_assistant(prompt):
        example["prompt"] = strip_trailing_assistant(prompt)
    return example


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Remove residual assistant turns from type=0 samples in a parquet dataset."
    )
    parser.add_argument("--parquet",  type=str, required=True,
                        help="Input parquet file path")
    parser.add_argument("--output",   type=str, default=None,
                        help="Output parquet path (default: overwrite input with .bak backup)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Only count affected samples, do not write any file")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip creating .bak backup when overwriting input file")
    args = parser.parse_args()

    if not os.path.isfile(args.parquet):
        print(f"[ERROR] File not found: {args.parquet}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or args.parquet

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading: {args.parquet}")
    ds = datasets.load_dataset("parquet", data_files=args.parquet, split="train")
    total = len(ds)
    print(f"Total samples: {total:,}")

    # ── Pre-scan: count how many type=0 samples have trailing assistant ────────
    print("\nScanning for type=0 samples with trailing assistant turns...")
    needs_fix = 0
    type0_total = 0
    by_msg_count: Counter = Counter()   # message count before fix → occurrences

    for example in tqdm(ds, total=total, desc="Scanning"):
        if get_type(example.get("extra_info", {})) != 0:
            continue
        type0_total += 1
        prompt = example.get("prompt", [])
        if has_trailing_assistant(prompt):
            needs_fix += 1
            by_msg_count[len(prompt)] += 1

    print(f"\n{'='*55}")
    print(f"SCAN RESULTS")
    print(f"{'='*55}")
    print(f"  type=0 samples total      : {type0_total:,}")
    print(f"  type=0 with assistant turn: {needs_fix:,}  ({needs_fix/type0_total:.2%} of type=0)" if type0_total else "  (no type=0 samples)")
    if by_msg_count:
        print(f"  Message-count breakdown (before fix):")
        for k in sorted(by_msg_count):
            print(f"    {k} messages: {by_msg_count[k]:,} samples")
    print(f"{'='*55}")

    if needs_fix == 0:
        print("\n✅ No samples need fixing. Exiting.")
        return

    if args.dry_run:
        print("\n[dry-run] No file written.")
        return

    # ── Apply fix ─────────────────────────────────────────────────────────────
    print(f"\nApplying fix to {needs_fix:,} samples...")
    fixed_ds = ds.map(fix_row, desc="Fixing type=0 prompts")

    # ── Post-validation ───────────────────────────────────────────────────────
    print("Validating fixed dataset...")
    remaining = sum(
        1 for ex in tqdm(fixed_ds, total=total, desc="Validating")
        if get_type(ex.get("extra_info", {})) == 0 and has_trailing_assistant(ex.get("prompt", []))
    )
    if remaining > 0:
        print(f"[ERROR] {remaining} type=0 samples STILL have assistant turns after fix!", file=sys.stderr)
        sys.exit(1)
    print(f"✅ Validation passed: 0 type=0 samples have trailing assistant turns.")

    # ── Save ──────────────────────────────────────────────────────────────────
    if output_path == args.parquet and not args.no_backup:
        backup_path = args.parquet + ".bak"
        shutil.copy2(args.parquet, backup_path)
        print(f"Backup saved to: {backup_path}")

    print(f"Saving fixed dataset to: {output_path}")
    fixed_ds.to_parquet(output_path)
    print(f"✅ Done. {needs_fix:,} samples fixed and saved to {output_path}")


if __name__ == "__main__":
    main()
