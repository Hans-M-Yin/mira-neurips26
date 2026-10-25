"""Convert an augmented verl parquet into a strict vanilla-GRPO training parquet.

Every row is retained, but any synthetic assistant caption prefix is removed
from ``prompt`` and the augmentation metadata is reset to type=0.  Images,
questions, answers, reward-model ground truth, and all ordinary prompt turns
are preserved.  This makes the Vanilla GRPO control use the same examples as
the augmented run without exposing injected wrong/correct descriptions.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--local-cache-dir", default="/tmp/verl_vanilla_cache",
                        help="Local cache for Dataset.map; do not place it on HDFS.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Reserved for reproducibility; row order is preserved.")
    return parser.parse_args()


def as_extra_info(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def to_vanilla(example: dict[str, Any]) -> dict[str, Any]:
    prompt = example.get("prompt") or []
    # Preprocessed MIRA rows only use assistant turns as injected prefixes.
    # Preserve system and user turns verbatim, including multimodal content.
    example["prompt"] = [copy.deepcopy(message) for message in prompt if message.get("role") != "assistant"]

    extra = as_extra_info(example.get("extra_info"))
    extra["type"] = 0
    extra["caption_correct"] = None
    extra["caption_modified"] = None
    extra["modify_info"] = None
    extra["num_modify"] = 0
    # These fields are internal preprocessing diagnostics, not supervision.
    extra.pop("augmentation_attempted", None)
    extra.pop("downgrade_reason", None)
    example["extra_info"] = extra

    # Support older variants that stored augmentation fields at top level.
    for key in ("caption_correct", "caption_modified", "modify_info"):
        if key in example:
            example[key] = None
    if "num_modify" in example:
        example["num_modify"] = 0
    if "augmentation_attempted" in example:
        example["augmentation_attempted"] = False
    if "downgrade_reason" in example:
        example["downgrade_reason"] = ""
    return example


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.local_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep both HF's builder cache and Dataset.map cache off HDFS.  Some HDFS
    # mounts reject the temporary-file/flush pattern used by datasets.
    dataset = load_dataset(
        "parquet",
        data_files=args.input_parquet,
        split="train",
        cache_dir=str(cache_dir),
    )
    original_types = Counter(as_extra_info(value).get("type", "unknown") for value in dataset["extra_info"])
    print(f"Input rows: {len(dataset):,}; original types: {dict(sorted(original_types.items(), key=str))}")

    converted = dataset.map(
        to_vanilla,
        load_from_cache_file=False,
        cache_file_name=str(cache_dir / f"vanilla_{os.getpid()}.arrow"),
        desc="Removing augmented assistant prefixes",
    )
    remaining_assistant = sum(
        any(message.get("role") == "assistant" for message in (prompt or []))
        for prompt in converted["prompt"]
    )
    final_types = Counter(as_extra_info(value).get("type", "unknown") for value in converted["extra_info"])
    if remaining_assistant or final_types != Counter({0: len(converted)}):
        raise RuntimeError(
            f"Vanilla conversion validation failed: assistant_prefixes={remaining_assistant}, "
            f"types={dict(final_types)}"
        )

    converted.to_parquet(str(output_path))
    print(f"Saved strict vanilla parquet ({len(converted):,} rows) to {output_path}")


if __name__ == "__main__":
    main()
