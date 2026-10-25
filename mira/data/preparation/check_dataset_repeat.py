"""Check for overlapping questions between two Parquet datasets.

Example:
    python -m mira.data.preparation.check_dataset_repeat --batch1 batch1/train.parquet --batch2 batch2/train.parquet"""
import argparse
import hashlib
import json
import pandas as pd
from collections import defaultdict


def get_question(row):
    extra = row.get('extra_info', {})
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except Exception:
            extra = {}
    q = extra.get('question', '')
    if not q:
        # Fall back to the user message in the prompt.
        prompt = row.get('prompt', [])
        for msg in (prompt if isinstance(prompt, list) else []):
            if isinstance(msg, dict) and msg.get('role') == 'user':
                q = str(msg.get('content', ''))
                break
    return str(q).strip()


def question_hash(q: str) -> str:
    """Use an MD5 digest as a lightweight key instead of storing long question strings."""
    return hashlib.md5(q.encode('utf-8')).hexdigest()


def load_hashes(parquet_path):
    df = pd.read_parquet(parquet_path)
    hashes = {}  # Map each hash to a question excerpt for display.
    for _, row in df.iterrows():
        q = get_question(row)
        if q:
            h = question_hash(q)
            hashes[h] = q[:80]  # Display only the first 80 characters.
    return hashes, len(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch1', required=True, help='First Parquet dataset path.')
    parser.add_argument('--batch2', required=True, help='Second Parquet dataset path.')
    parser.add_argument('--show_examples', type=int, default=5, help='Number of overlapping examples to display.')
    args = parser.parse_args()

    print(f"Loading batch1: {args.batch1}")
    hashes1, n1 = load_hashes(args.batch1)
    print(f"  {n1} rows, {len(hashes1)} unique questions")

    print(f"Loading batch2: {args.batch2}")
    hashes2, n2 = load_hashes(args.batch2)
    print(f"  {n2} rows, {len(hashes2)} unique questions")

    overlap_hashes = set(hashes1.keys()) & set(hashes2.keys())
    only_in_1 = set(hashes1.keys()) - set(hashes2.keys())
    only_in_2 = set(hashes2.keys()) - set(hashes1.keys())

    print(f"\n{'='*50}")
    print(f"  Overlapping questions: {len(overlap_hashes)}")
    print(f"  Only in batch 1: {len(only_in_1)}")
    print(f"  Only in batch 2: {len(only_in_2)}")
    print(f"  Overlap relative to batch 1: {len(overlap_hashes)/len(hashes1):.2%}")
    print(f"  Overlap relative to batch 2: {len(overlap_hashes)/len(hashes2):.2%}")
    print(f"{'='*50}")

    if overlap_hashes:
        print(f"\nFirst {args.show_examples} overlapping questions:")
        for h in list(overlap_hashes)[:args.show_examples]:
            print(f"  Q: {hashes1[h]}")
    else:
        print("\nThe two datasets contain no overlapping questions.")


if __name__ == '__main__':
    main()
