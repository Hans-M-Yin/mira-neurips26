#!/usr/bin/env python3
"""Select medium-difficulty visual reasoning samples from ViRL-39K and MMK12.

Pipeline
--------
1. Download the two public Hugging Face datasets (the datasets cache handles
   reuse automatically) and normalize their schemas to ``problem/answer/images``.
2. Run Qwen3-VL-8B ``--rollouts-per-sample`` times for every sample.
3. Ask an OpenAI-compatible judge whether each rollout answer is correct.
4. Keep samples answered correctly 0--4 times out of 5, filtering only the
   consistently easy samples answered correctly in all five rollouts.

The output directory contains a ``dataset/`` saved with
``datasets.Dataset.save_to_disk``; pass that directory to
``data_preprocess_v2.py --dataset_path OUTPUT_DIR/dataset``. It also contains
``selected.parquet`` for inspection. The canonical ``problem`` field
contains exactly one ``<image>`` placeholder per stored image.  During rollout,
however, images are passed as structured chat-template content; do not manually
insert image tokens into the vLLM prompt.

Example
-------
CUDA_VISIBLE_DEVICES=0,1 python mira/data/preparation/select_virl_mmk_by_rollout.py \
  --rollout-model ../models/qwen3-vl-8b --tensor-parallel-size 2 \
  --judge-api-url http://HOST:PORT/v1 --judge-api-key EMPTY \
  --judge-model Qwen3-VL-32B \
  --output-dir mira/data/preparation/virl39k_mmk12_qwen3vl8b_n5_medium

Use ``--max-samples 100 --batch-size 16`` for a smoke test first.  Add
``--resume`` after an interruption; the script reuses ``rollout_judgements.jsonl``
in the output directory.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from datasets import Dataset, Image as HFImage, load_dataset
from openai import OpenAI
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


MMK12_REPO = "FanqingM/MMK12"
VIRL39K_REPO = "PAPOGalaxy/PAPO_ViRL39K_train"
IMAGE_PLACEHOLDER_RE = re.compile(r"<image(?:_\d+)?>", flags=re.IGNORECASE)
SYSTEM_PROMPT = (
    "Solve the visual reasoning problem carefully. First reason in <think></think>, "
    "then give the final answer in <answer></answer>."
)
JUDGE_SYSTEM_PROMPT = """You are a strict answer-equivalence judge for visual reasoning.
Given a question, its ground-truth answer, and a model response, decide whether the
model's final answer is semantically equivalent to the ground truth. Ignore reasoning
quality and formatting. For multiple choice questions, accept the option letter only
when it identifies the ground-truth option. Output exactly one token: CORRECT or
INCORRECT."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rollout-model", default="../models/qwen3-vl-8b")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=14000)
    parser.add_argument("--rollouts-per-sample", type=int, default=5)
    parser.add_argument("--keep-min-correct", type=int, default=0)
    parser.add_argument("--keep-max-correct", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap after merging and shuffling.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--staging-dir",
        default=None,
        help=("Local writable staging directory. Defaults to "
              "/tmp/verl_difficulty_filter_staging/<output-dir-name>. "
              "JSONL is appended here; final artifacts are copied to --output-dir."),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--judge-api-url", required=True)
    parser.add_argument("--judge-api-key", default=os.getenv("JUDGE_API_KEY", "EMPTY"))
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-parallel", type=int, default=16)
    parser.add_argument("--judge-max-retries", type=int, default=4)
    return parser.parse_args()


def as_pil(value: Any) -> Image.Image:
    """Decode common datasets.Image representations into an RGB PIL image."""
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        raw = value.get("bytes") or value.get("data")
        if raw is not None:
            return Image.open(io.BytesIO(raw)).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    raise TypeError(f"Unsupported image value: {type(value)!r}")


def normalize_images(value: Any) -> list[Image.Image]:
    raw_images = value if isinstance(value, (list, tuple)) else [value]
    images = [as_pil(image) for image in raw_images if image is not None]
    if not images:
        raise ValueError("sample contains no decodable images")
    return images


def strip_image_placeholders(text: str) -> str:
    text = IMAGE_PLACEHOLDER_RE.sub("", text or "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def canonical_problem(text: str, num_images: int) -> str:
    """The raw schema consumed by data_preprocess_v2: N explicit placeholders."""
    return ("<image>\n" * num_images + strip_image_placeholders(text)).strip()


def normalize_mmk12(row: dict[str, Any], split: str, source_idx: int) -> dict[str, Any]:
    images = normalize_images(row["image"])
    return {
        "id": str(row.get("id", source_idx)),
        "problem": canonical_problem(str(row["question"]), len(images)),
        "answer": str(row["answer"]),
        "images": images,
        "source_dataset": "MMK12",
        "source_split": split,
        "source_idx": source_idx,
        "subject": str(row.get("subject", "")),
    }


def normalize_virl39k(row: dict[str, Any], split: str, source_idx: int) -> dict[str, Any]:
    images = normalize_images(row["images"])
    return {
        "id": str(row.get("id", source_idx)),
        "problem": canonical_problem(str(row["problem"]), len(images)),
        "answer": str(row["answer"]),
        "images": images,
        "source_dataset": "ViRL39K",
        "source_split": split,
        "source_idx": source_idx,
        "subject": "",
    }


def load_sources(args: argparse.Namespace) -> tuple[dict[tuple[str, str], tuple[Any, Any]], list[dict[str, Any]]]:
    """Load Arrow-backed datasets but defer image decoding until each batch."""
    sources: dict[tuple[str, str], tuple[Any, Any]] = {}
    references: list[dict[str, Any]] = []
    # Load all advertised splits so that "MMK12" means the complete public
    # release, while retaining split provenance in every output row.
    for repo, normalizer in ((MMK12_REPO, normalize_mmk12), (VIRL39K_REPO, normalize_virl39k)):
        dataset_dict = load_dataset(repo, cache_dir=args.cache_dir)
        for split_name, split in dataset_dict.items():
            print(f"Loading {repo}:{split_name} ({len(split)} rows)")
            source_name = "MMK12" if repo == MMK12_REPO else "ViRL39K"
            sources[(source_name, split_name)] = (split, normalizer)
            references.extend(
                {"source_dataset": source_name, "source_split": split_name, "source_idx": source_idx}
                for source_idx in range(len(split))
            )
    random.Random(args.seed).shuffle(references)
    if args.max_samples is not None:
        references = references[: args.max_samples]
    print(f"Merged source references: {len(references)}")
    return sources, references


def materialize_record(reference: dict[str, Any], sources: dict[tuple[str, str], tuple[Any, Any]]) -> dict[str, Any]:
    split, normalizer = sources[(reference["source_dataset"], reference["source_split"])]
    return normalizer(dict(split[reference["source_idx"]]), reference["source_split"], reference["source_idx"])


def build_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    # Structured content is important: AutoProcessor adds exactly one image
    # token sequence for each image.  Passing literal <image> text here would
    # duplicate or desynchronize visual placeholders.
    content = [{"type": "image", "image": image} for image in record["images"]]
    content.append({"type": "text", "text": strip_image_placeholders(record["problem"]) + "\n" + SYSTEM_PROMPT})
    return [{"role": "user", "content": content}]


def build_vllm_input(record: dict[str, Any], processor: Any) -> dict[str, Any]:
    messages = build_messages(record)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    multi_modal_data: dict[str, Any] = {}
    if image_inputs is not None:
        multi_modal_data["image"] = image_inputs
    if video_inputs is not None:
        multi_modal_data["video"] = video_inputs
    return {"prompt": prompt, "multi_modal_data": multi_modal_data}


def judge_one(
    client: OpenAI,
    judge_model: str,
    question: str,
    ground_truth: str,
    response: str,
    retries: int,
) -> bool:
    user_prompt = (
        f"[Question]\n{question}\n\n[Ground truth]\n{ground_truth}\n\n"
        f"[Model response]\n{response}\n\nVerdict:"
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=judge_model,
                temperature=0.0,
                max_tokens=8,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            verdict = (completion.choices[0].message.content or "").strip().upper()
            if "INCORRECT" in verdict:
                return False
            if "CORRECT" in verdict:
                return True
            raise ValueError(f"unexpected judge verdict: {verdict!r}")
        except Exception as exc:
            last_error = exc
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"judge failed after {retries} attempts: {last_error}")


def judge_rollouts(record: dict[str, Any], outputs: list[str], args: argparse.Namespace, client: OpenAI) -> list[bool]:
    results: list[bool | None] = [None] * len(outputs)
    with ThreadPoolExecutor(max_workers=args.judge_parallel) as executor:
        futures = {
            executor.submit(
                judge_one, client, args.judge_model, strip_image_placeholders(record["problem"]),
                record["answer"], output, args.judge_max_retries,
            ): index
            for index, output in enumerate(outputs)
        }
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
    return [bool(value) for value in results]


def load_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as file:
        for line in file:
            try:
                completed.add(json.loads(line)["record_key"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def record_key(record: dict[str, Any]) -> str:
    return f"{record['source_dataset']}:{record['source_split']}:{record['source_idx']}"


def merge_remote_checkpoint(remote_path: Path, local_path: Path) -> int:
    """Merge a previously written HDFS JSONL checkpoint into local staging."""
    if not remote_path.exists():
        return 0
    existing = load_completed(local_path)
    copied = 0
    with remote_path.open(encoding="utf-8") as source, local_path.open("a", encoding="utf-8") as target:
        for line in source:
            try:
                key = json.loads(line)["record_key"]
            except (json.JSONDecodeError, KeyError):
                continue
            if key not in existing:
                target.write(line if line.endswith("\n") else line + "\n")
                existing.add(key)
                copied += 1
    return copied


def promote_staging_directory(staging_dir: Path, output_dir: Path) -> None:
    """Copy completed local artifacts to HDFS, then remove the local staging copy."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in staging_dir.iterdir():
        destination = output_dir / source.name
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)
    # This is a local delete only, after every artifact has been copied to HDFS;
    # it implements the requested move while keeping the active run recoverable.
    shutil.rmtree(staging_dir)
    print(f"Promoted completed artifacts from local staging to {output_dir}")


def serializable_record(record: dict[str, Any], outputs: list[str], correct: list[bool]) -> dict[str, Any]:
    return {
        "record_key": record_key(record),
        "id": record["id"],
        "problem": record["problem"],
        "answer": record["answer"],
        "source_dataset": record["source_dataset"],
        "source_split": record["source_split"],
        "source_idx": record["source_idx"],
        "subject": record["subject"],
        "rollout_outputs": outputs,
        "judge_correct": correct,
        "num_correct": sum(correct),
    }


def write_final_dataset(
    references: list[dict[str, Any]],
    sources: dict[tuple[str, str], tuple[Any, Any]],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    result_path = output_dir / "rollout_judgements.jsonl"
    retained_by_key: dict[str, dict[str, Any]] = {}
    with result_path.open(encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            if args.keep_min_correct <= item["num_correct"] <= args.keep_max_correct:
                retained_by_key[item["record_key"]] = item

    reference_by_key = {
        f"{reference['source_dataset']}:{reference['source_split']}:{reference['source_idx']}": reference
        for reference in references
    }
    final_rows: list[dict[str, Any]] = []
    for key, judgement in retained_by_key.items():
        reference = reference_by_key.get(key)
        if reference is None:
            continue
        try:
            source = materialize_record(reference, sources)
        except Exception as exc:
            print(f"[skip final write] {key}: {type(exc).__name__}: {exc}")
            continue
        final_rows.append({
            # data_preprocess_v2 consumes these three raw fields.
            "problem": source["problem"],
            "answer": source["answer"],
            "images": source["images"],
            "id": source["id"],
            "source_dataset": source["source_dataset"],
            "source_split": source["source_split"],
            "source_idx": source["source_idx"],
            "subject": source["subject"],
            "rollout_outputs": judgement["rollout_outputs"],
            "judge_correct": judgement["judge_correct"],
            "num_correct": judgement["num_correct"],
        })

    if not final_rows:
        raise RuntimeError("No samples passed the difficulty filter; no dataset was written.")
    random.Random(args.seed).shuffle(final_rows)
    dataset = Dataset.from_list(final_rows)
    features = dataset.features.copy()
    features["images"] = [HFImage()]
    dataset = dataset.cast(features)
    dataset.save_to_disk(str(output_dir / "dataset"))
    dataset.to_parquet(str(output_dir / "selected.parquet"))
    with (output_dir / "selection_metadata.json").open("w", encoding="utf-8") as file:
        json.dump({
            "sources": [MMK12_REPO, VIRL39K_REPO],
            "rollout_model": args.rollout_model,
            "rollouts_per_sample": args.rollouts_per_sample,
            "keep_correct_range": [args.keep_min_correct, args.keep_max_correct],
            "num_retained": len(dataset),
            "seed": args.seed,
        }, file, indent=2, ensure_ascii=False)
    print(f"Saved {len(dataset)} selected samples to {output_dir / 'dataset'}")
    print("Next step: python mira/data/preparation/data_preprocess_v2.py --dataset_path " f"{output_dir / 'dataset'} ...")


def main() -> None:
    args = parse_args()
    if args.rollouts_per_sample != 5:
        print(f"[warning] requested {args.rollouts_per_sample} rollouts; the default protocol uses 5.")
    if not (0 <= args.keep_min_correct <= args.keep_max_correct <= args.rollouts_per_sample):
        raise ValueError("Invalid keep-correct range")
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = (
        Path(args.staging_dir).expanduser()
        if args.staging_dir
        else Path("/tmp/verl_difficulty_filter_staging") / output_dir.name
    )
    staging_dir.mkdir(parents=True, exist_ok=True)
    remote_result_path = output_dir / "rollout_judgements.jsonl"
    result_path = staging_dir / "rollout_judgements.jsonl"
    if args.resume:
        copied = merge_remote_checkpoint(remote_result_path, result_path)
        if copied:
            print(f"Imported {copied} completed records from HDFS checkpoint into {staging_dir}")
    print(f"Using local staging directory: {staging_dir}")
    sources, references = load_sources(args)
    completed = load_completed(result_path) if args.resume else set()
    pending = [reference for reference in references if (
        f"{reference['source_dataset']}:{reference['source_split']}:{reference['source_idx']}" not in completed
    )]
    print(f"Pending samples: {len(pending)}; resumed samples: {len(completed)}")

    processor = AutoProcessor.from_pretrained(args.rollout_model, trust_remote_code=True)
    llm = LLM(
        model=args.rollout_model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 8},
    )
    sampling_params = SamplingParams(
        n=args.rollouts_per_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    judge_client = OpenAI(api_key=args.judge_api_key, base_url=args.judge_api_url)

    with result_path.open("a", encoding="utf-8") as result_file:
        for start in tqdm(range(0, len(pending), args.batch_size), desc="Rollout batches"):
            batch_references = pending[start : start + args.batch_size]
            batch: list[dict[str, Any]] = []
            for reference in batch_references:
                try:
                    batch.append(materialize_record(reference, sources))
                except Exception as exc:
                    print(
                        f"[skip image/schema error] {reference['source_dataset']}:"
                        f"{reference['source_split']}:{reference['source_idx']}: {type(exc).__name__}: {exc}"
                    )
            if not batch:
                continue
            inputs = [build_vllm_input(record, processor) for record in batch]
            generated = llm.generate(inputs, sampling_params)
            for record, request_output in zip(batch, generated, strict=True):
                outputs = [candidate.text for candidate in request_output.outputs]
                if len(outputs) != args.rollouts_per_sample:
                    raise RuntimeError(
                        f"{record_key(record)} returned {len(outputs)} rollouts, expected {args.rollouts_per_sample}"
                    )
                try:
                    correct = judge_rollouts(record, outputs, args, judge_client)
                except Exception as exc:
                    # Do not write failed judge calls: --resume will retry them.
                    print(f"[judge failed; will retry on resume] {record_key(record)}: {exc}")
                    continue
                result_file.write(json.dumps(serializable_record(record, outputs, correct), ensure_ascii=False) + "\n")
                result_file.flush()

    write_final_dataset(references, sources, staging_dir, args)
    promote_staging_directory(staging_dir, output_dir)


if __name__ == "__main__":
    main()
