"""Validate deceptive-context training data.

Checks include blind-test answer flips, correct-caption sufficiency, edits to
facts explicitly stated in the question, image-aware oracle susceptibility,
and sample/edit/length distributions.

Example:
    python -m mira.data.preparation.validate_dataset --parquet_path train.parquet --api_url https://api.openai.com/v1 --sample_n 100 --run_oracle"""

import argparse
import json
import re
import random
import os
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize_answer(ans: str) -> str:
    """Normalize an answer for a coarse equivalence comparison."""
    if ans is None:
        return ""
    ans = ans.strip().lower()
    # Extract an option letter from surrounding brackets.
    m = re.search(r'\(?([a-d])\)?', ans)
    if m:
        return m.group(1)
    return ans[:20]  # Compare the first 20 characters as a coarse fallback.


def extract_answer_tag(text: str) -> str | None:
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def blind_test(client, model, question, caption):
    """Run a text-only blind test with the caption and question, without the image."""
    prompt = f"""I have a question related to an image. The image is not available, but I have a caption describing the image.
Please answer the question based ONLY on the caption below.
Think step by step, then give your final answer in <answer></answer> tags.

[Caption]: {caption}
[Question]: {question}
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.0
    )
    return resp.choices[0].message.content


def oracle_test(client, model, question, caption, image_b64):
    """Test whether an image-aware oracle follows the misleading caption."""
    prompt = f"""You are given an image and a caption that describes the image.
The caption may or may not contain errors compared to the actual image.

Your task:
1. Answer the question based on BOTH the image and the caption.
2. In <answer></answer> tags, give your final answer.
3. In <caption_trust></caption_trust> tags, say whether you trusted the caption or relied more on the image (and why).

[Caption]: {caption}
[Question]: {question}
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                },
                {"type": "text", "text": prompt}
            ]
        }],
        max_tokens=1024,
        temperature=0.0
    )
    return resp.choices[0].message.content


# ─────────────────────────────────────────────────────────────────────────────
# Check 1: blind-test answer flip rate.
# ─────────────────────────────────────────────────────────────────────────────

def check_blind_tests(samples, client, model, max_workers=8):
    """Run blind tests on augmented samples.

    The correct caption should lead to the correct answer, while the misleading
    caption should change the answer. Return the results for each sample."""
    results = []

    def process(sample):
        extra = sample.get('extra_info', {})
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except Exception:
                extra = {}

        question = extra.get('question', '')
        gt_answer = str(extra.get('answer', '')).strip().lower()
        caption_correct = extra.get('caption_correct', '')
        caption_modified = extra.get('caption_modified', '')
        sample_type = extra.get('type', -1)

        result = {
            'type': sample_type,
            'question': question[:80],
            'gt_answer': gt_answer,
            'flip_passed': None,
            'sufficiency_passed': None,
            'blind_correct_answer': None,
            'blind_modified_answer': None,
        }

        if sample_type == 1 and caption_correct and caption_modified:
            # Blind test with the correct caption.
            try:
                resp_correct = blind_test(client, model, question, caption_correct)
                ans_correct = extract_answer_tag(resp_correct)
                result['blind_correct_answer'] = ans_correct
                result['sufficiency_passed'] = (
                    normalize_answer(ans_correct) == normalize_answer(gt_answer)
                )
            except Exception as e:
                result['sufficiency_passed'] = None

            # Blind test with the misleading caption.
            try:
                resp_modified = blind_test(client, model, question, caption_modified)
                ans_modified = extract_answer_tag(resp_modified)
                result['blind_modified_answer'] = ans_modified
                result['flip_passed'] = (
                    normalize_answer(ans_modified) != normalize_answer(gt_answer)
                )
            except Exception as e:
                result['flip_passed'] = None

        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process, s): s for s in samples}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Blind tests"):
            results.append(f.result())

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Check 2: violations of known-condition protection.
# ─────────────────────────────────────────────────────────────────────────────

def _to_python_list(obj):
    """Recursively convert NumPy arrays and scalars to native Python values."""
    if isinstance(obj, np.ndarray):
        return [_to_python_list(x) for x in obj.tolist()]
    if isinstance(obj, (np.generic,)):  # numpy scalar
        return obj.item()
    if isinstance(obj, list):
        return [_to_python_list(x) for x in obj]
    return obj


def check_rule0_violations(samples):
    """Flag edits to numeric facts already present in the question.

    Normalize Parquet-loaded modification metadata before checking whether the
    original numbers appear in the question text."""
    violations = []
    clean = []

    for sample in samples:
        extra = sample.get('extra_info', {})
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except Exception:
                extra = {}

        if extra.get('type') != 1:
            continue

        question = str(extra.get('question', ''))
        modify_info = extra.get('modify_info', None)

        # Normalize arrays and missing values to native Python objects.
        if modify_info is None:
            continue
        # Parquet loading can convert missing values to floating-point NaN.
        try:
            if isinstance(modify_info, float):  # NaN
                continue
        except Exception:
            pass
        modify_info = _to_python_list(modify_info)
        if not isinstance(modify_info, list) or len(modify_info) == 0:
            continue

        violated = False
        for mod in modify_info:
            mod = _to_python_list(mod)
            if not isinstance(mod, (list, tuple)) or len(mod) < 2:
                continue
            before_text = str(mod[0])
            after_text = str(mod[1])
            # Treat edits to numbers explicitly present in the question as possible rule violations.
            numbers_in_before = re.findall(r'\d+\.?\d*', before_text)
            for num in numbers_in_before:
                if num in question:
                    violated = True
                    violations.append({
                        'question': question[:80],
                        'before': before_text,
                        'after': after_text,
                        'matched_num': num
                    })
                    break
            if violated:
                break

        if not violated:
            clean.append(sample)

    return violations, clean


# ─────────────────────────────────────────────────────────────────────────────
# Check 3: image-aware oracle evaluation of misleading captions.
# ─────────────────────────────────────────────────────────────────────────────

def check_oracle_deception(samples, client, oracle_model, max_workers=4):
    """Evaluate misleading captions with an image-aware oracle.

    An incorrect answer indicates that the visual error remains deceptive even
    with the image available. A corrected answer indicates a more obvious error.
    Blind-test failures that also mislead the oracle are the strongest examples;
    samples corrected by the oracle can still provide a recovery signal."""
    import base64
    import io

    results = []

    def process(sample):
        extra = sample.get('extra_info', {})
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except Exception:
                extra = {}

        question = extra.get('question', '')
        gt_answer = str(extra.get('answer', '')).strip().lower()
        caption_modified = extra.get('caption_modified', '')

        # Extract images.
        images = sample.get('images', [])
        if not images:
            return None

        try:
            # Images may already be decoded or stored as bytes.
            img = images[0] if isinstance(images, list) else images
            if hasattr(img, 'save'):
                buf = io.BytesIO()
                img.save(buf, format='JPEG')
                image_b64 = base64.b64encode(buf.getvalue()).decode()
            elif isinstance(img, dict) and 'bytes' in img:
                image_b64 = base64.b64encode(img['bytes']).decode()
            else:
                return None

            resp = oracle_test(client, oracle_model, question, caption_modified, image_b64)
            ans = extract_answer_tag(resp)
            trust_match = re.search(r'<caption_trust>(.*?)</caption_trust>', resp, re.DOTALL)
            caption_trust = trust_match.group(1).strip() if trust_match else ""

            oracle_fooled = (normalize_answer(ans) != normalize_answer(gt_answer))

            return {
                'question': question[:80],
                'gt_answer': gt_answer,
                'oracle_answer': ans,
                'oracle_fooled': oracle_fooled,
                'caption_trust': caption_trust[:100],
            }
        except Exception as e:
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process, s) for s in samples]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Oracle tests"):
            r = f.result()
            if r:
                results.append(r)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Check 4: distribution statistics.
# ─────────────────────────────────────────────────────────────────────────────

def analyze_distribution(df):
    """Summarize sample-type, edit-count, and caption-length distributions."""
    stats = {}

    # Sample-type distribution.
    def get_type(row):
        extra = row.get('extra_info', {})
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except Exception:
                extra = {}
        return extra.get('type', -1)

    types = df.apply(get_type, axis=1)
    type_counts = Counter(types)
    stats['type_distribution'] = dict(type_counts)
    stats['total_samples'] = len(df)
    stats['type1_ratio'] = type_counts.get(1, 0) / len(df)
    stats['type0_ratio'] = type_counts.get(0, 0) / len(df)

    # Number-of-edits distribution.
    def get_num_modify(row):
        extra = row.get('extra_info', {})
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except Exception:
                extra = {}
        return extra.get('num_modify', None)

    num_modifies = df.apply(get_num_modify, axis=1).dropna()
    k_counts = Counter(num_modifies)
    stats['k_distribution'] = dict(k_counts)

    # Caption-length statistics.
    def get_caption_lengths(row):
        extra = row.get('extra_info', {})
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except Exception:
                extra = {}
        correct = extra.get('caption_correct', '') or ''
        modified = extra.get('caption_modified', '') or ''
        return len(correct), len(modified)

    type1_rows = df[types == 1]
    if len(type1_rows) > 0:
        lengths = type1_rows.apply(get_caption_lengths, axis=1)
        correct_lens = [x[0] for x in lengths]
        modified_lens = [x[1] for x in lengths]
        stats['avg_caption_correct_len'] = sum(correct_lens) / len(correct_lens) if correct_lens else 0
        stats['avg_caption_modified_len'] = sum(modified_lens) / len(modified_lens) if modified_lens else 0

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", type=str, required=True, help="Path to train.parquet")
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--api_url", type=str, default="https://yunwu.ai/v1")
    parser.add_argument("--blind_model", type=str, default="gpt-4.1-mini",
                        help="Text-only model for blind tests")
    parser.add_argument("--oracle_model", type=str, default="gpt-4.1",
                        help="Image-aware oracle model; uses additional image tokens.")
    parser.add_argument("--sample_n", type=int, default=100,
                        help="Number of type=1 samples to validate")
    parser.add_argument("--run_oracle", action="store_true",
                        help="Run oracle (VLM+image) deception test (slower)")
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", type=str, default="validation_report.json",
                        help="Save detailed results to this file")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading dataset from: {args.parquet_path}")
    df = pd.read_parquet(args.parquet_path)

    client = OpenAI(api_key=args.api_key, base_url=args.api_url)

    report = {}

    # ── Distribution Analysis ──────────────────────────────────────────────
    print_section("Check 1: Distribution Analysis")
    dist_stats = analyze_distribution(df)
    report['distribution'] = dist_stats
    print(f"  Total samples       : {dist_stats['total_samples']}")
    print(f"  Type distribution   : {dist_stats['type_distribution']}")
    print(f"  Type-1 ratio        : {dist_stats['type1_ratio']:.2%}")
    print(f"  Type-0 ratio        : {dist_stats['type0_ratio']:.2%}")
    print(f"  K distribution      : {dist_stats.get('k_distribution', {})}")
    print(f"  Avg correct cap len : {dist_stats.get('avg_caption_correct_len', 0):.0f} chars")
    print(f"  Avg modified cap len: {dist_stats.get('avg_caption_modified_len', 0):.0f} chars")

    # Sample augmented rows.
    all_rows = df.to_dict('records')

    def get_type_from_row(row):
        extra = row.get('extra_info', {})
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except Exception:
                extra = {}
        return extra.get('type', -1)

    type1_rows = [r for r in all_rows if get_type_from_row(r) == 1]
    sample_n = min(args.sample_n, len(type1_rows))
    sampled = random.sample(type1_rows, sample_n)
    print(f"\n  Sampled {sample_n} type=1 samples for detailed validation.")

    # ── RULE 0 Violation Check ─────────────────────────────────────────────
    print_section("Check 2: RULE 0 Violation Detection")
    violations, clean_samples = check_rule0_violations(sampled)
    violation_rate = len(violations) / sample_n if sample_n > 0 else 0
    report['rule0'] = {
        'sample_n': sample_n,
        'violations': len(violations),
        'violation_rate': violation_rate,
        'examples': violations[:5]  # Keep the first five violation examples.
    }
    print(f"  Sampled: {sample_n} | Violations: {len(violations)} | Rate: {violation_rate:.2%}")
    if violations:
        print(f"\n  Example violations:")
        for v in violations[:3]:
            print(f"    Q: {v['question']}")
            print(f"    Before: {v['before'][:60]}")
            print(f"    After:  {v['after'][:60]}")
            print(f"    Matched num in question: '{v['matched_num']}'")
            print()

    # ── Blind Test ─────────────────────────────────────────────────────────
    print_section("Check 3: Blind Test (Answer Flip Rate + Caption Sufficiency)")
    blind_results = check_blind_tests(sampled, client, args.blind_model, max_workers=args.max_workers)

    type1_results = [r for r in blind_results if r.get('type') == 1]
    flip_results = [r for r in type1_results if r.get('flip_passed') is not None]
    suff_results = [r for r in type1_results if r.get('sufficiency_passed') is not None]

    flip_rate = sum(1 for r in flip_results if r['flip_passed']) / len(flip_results) if flip_results else 0
    suff_rate = sum(1 for r in suff_results if r['sufficiency_passed']) / len(suff_results) if suff_results else 0

    # Count examples passing both caption sufficiency and answer-flip checks.
    both_passed = [r for r in type1_results
                   if r.get('flip_passed') and r.get('sufficiency_passed')]
    both_rate = len(both_passed) / len(type1_results) if type1_results else 0

    report['blind_test'] = {
        'sample_n': len(type1_results),
        'flip_rate': flip_rate,
        'sufficiency_rate': suff_rate,
        'both_passed_rate': both_rate,
        'details': blind_results[:20]  # Keep the first 20 detailed results.
    }

    print(f"  Samples tested          : {len(type1_results)}")
    print(f"  Answer flip rate        : {flip_rate:.2%}  (modified caption → wrong answer)")
    print(f"  Caption sufficiency rate: {suff_rate:.2%}  (correct caption → right answer)")
    print(f"  Both passed (gold)      : {both_rate:.2%}  (ideal samples for training)")

    # Display selected failure examples.
    failed_flip = [r for r in flip_results if not r['flip_passed']][:3]
    if failed_flip:
        print(f"\n  ⚠ Flip FAIL examples (modification had no effect):")
        for r in failed_flip:
            print(f"    Q: {r['question']}")
            print(f"    GT: {r['gt_answer']} | Blind modified answer: {r['blind_modified_answer']}")

    failed_suff = [r for r in suff_results if not r['sufficiency_passed']][:3]
    if failed_suff:
        print(f"\n  ⚠ Sufficiency FAIL examples (correct caption insufficient):")
        for r in failed_suff:
            print(f"    Q: {r['question']}")
            print(f"    GT: {r['gt_answer']} | Blind correct answer: {r['blind_correct_answer']}")

    # Optional oracle evaluation.
    if args.run_oracle:
        print_section("Check 4: Oracle Deception Test (VLM + image)")
        # Evaluate at most 30 blind-test successes with the image-aware oracle.
        oracle_sample_n = min(30, sample_n)
        oracle_samples = random.sample(sampled, oracle_sample_n)
        oracle_results = check_oracle_deception(
            oracle_samples, client, args.oracle_model, max_workers=args.max_workers
        )

        fooled_count = sum(1 for r in oracle_results if r.get('oracle_fooled'))
        oracle_fool_rate = fooled_count / len(oracle_results) if oracle_results else 0

        report['oracle_test'] = {
            'sample_n': len(oracle_results),
            'fooled_count': fooled_count,
            'oracle_fool_rate': oracle_fool_rate,
            'details': oracle_results[:10]
        }
        print(f"  Oracle samples tested: {len(oracle_results)}")
        print(f"  Oracle fooled rate   : {oracle_fool_rate:.2%}")
        print(f"  (Higher = errors are more subtle/plausible)")

        print(f"\n  Examples where Oracle was FOOLED (ideal):")
        for r in [x for x in oracle_results if x.get('oracle_fooled')][:2]:
            print(f"    Q: {r['question']}")
            print(f"    GT: {r['gt_answer']} | Oracle ans: {r['oracle_answer']}")

        print(f"\n  Examples where Oracle SELF-CORRECTED (modification too obvious):")
        for r in [x for x in oracle_results if not x.get('oracle_fooled')][:2]:
            print(f"    Q: {r['question']}")
            print(f"    GT: {r['gt_answer']} | Oracle ans: {r['oracle_answer']}")
            print(f"    Trust note: {r['caption_trust']}")

    # ── Final Summary ──────────────────────────────────────────────────────
    print_section("SUMMARY")
    print(f"  Type-1 ratio (in dataset)  : {dist_stats['type1_ratio']:.2%}")
    print(f"  RULE0 violation rate       : {violation_rate:.2%}  (target: <5%)")
    print(f"  Answer flip rate           : {flip_rate:.2%}  (target: >70%)")
    print(f"  Caption sufficiency rate   : {suff_rate:.2%}  (target: >60%)")
    print(f"  Gold sample rate (both)    : {both_rate:.2%}  (target: >50%)")
    if args.run_oracle:
        print(f"  Oracle fooled rate         : {oracle_fool_rate:.2%}  (target: >30%)")

    print(f"\n  Interpretation:")
    if flip_rate > 0.7:
        print(f"  ✅ Modifications are effective (answer changes when caption wrong)")
    else:
        print(f"  ❌ Low flip rate — many modifications don't affect the answer at all")

    if suff_rate > 0.6:
        print(f"  ✅ Correct captions are informative enough to answer questions")
    else:
        print(f"  ⚠️  Low sufficiency — correct captions may lack key visual details")

    if violation_rate < 0.05:
        print(f"  ✅ RULE0 well-respected — modifications avoid given conditions")
    else:
        print(f"  ❌ High RULE0 violations — model is modifying question-given facts")

    # ── Save Report ────────────────────────────────────────────────────────
    output_path = args.output_json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Detailed report saved to: {output_path}")


if __name__ == '__main__':
    main()
