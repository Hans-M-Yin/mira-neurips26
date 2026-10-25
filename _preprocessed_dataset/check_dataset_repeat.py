"""
check_overlap.py  —  检查两批 parquet 数据是否有重合题目

用法：
  python check_overlap.py \
      --batch1 ./batch1/train.parquet \
      --batch2 ./batch2/train.parquet
"""
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
        # fallback: 从 prompt 里取 user 消息
        prompt = row.get('prompt', [])
        for msg in (prompt if isinstance(prompt, list) else []):
            if isinstance(msg, dict) and msg.get('role') == 'user':
                q = str(msg.get('content', ''))
                break
    return str(q).strip()


def question_hash(q: str) -> str:
    """用 MD5 做轻量 hash，避免直接存长字符串"""
    return hashlib.md5(q.encode('utf-8')).hexdigest()


def load_hashes(parquet_path):
    df = pd.read_parquet(parquet_path)
    hashes = {}  # hash -> question snippet (便于展示)
    for _, row in df.iterrows():
        q = get_question(row)
        if q:
            h = question_hash(q)
            hashes[h] = q[:80]  # 只保留前80字符供展示
    return hashes, len(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch1', required=True, help='第一批 parquet 路径')
    parser.add_argument('--batch2', required=True, help='第二批 parquet 路径')
    parser.add_argument('--show_examples', type=int, default=5, help='展示几个重合示例')
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
    print(f"  重合题目数  : {len(overlap_hashes)}")
    print(f"  仅在batch1  : {len(only_in_1)}")
    print(f"  仅在batch2  : {len(only_in_2)}")
    print(f"  重合比例(vs batch1): {len(overlap_hashes)/len(hashes1):.2%}")
    print(f"  重合比例(vs batch2): {len(overlap_hashes)/len(hashes2):.2%}")
    print(f"{'='*50}")

    if overlap_hashes:
        print(f"\n重合题目示例（前{args.show_examples}个）：")
        for h in list(overlap_hashes)[:args.show_examples]:
            print(f"  Q: {hashes1[h]}")
    else:
        print("\n✅ 两批数据无重合题目")


if __name__ == '__main__':
    main()
