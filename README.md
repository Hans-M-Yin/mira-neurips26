# MIRA evaluation release candidate

This is a clean, evaluation-focused candidate package for the accepted paper. It contains the unified local-model inference/evaluation entry point and the judge modules imported by that entry point. It intentionally excludes training runs, GRPO baselines, exploratory analyses, raw experiment outputs, checkpoints, and datasets.

## Scope to confirm before publication

- Add the final paper citation, authors, and public model links.
- Confirm the exact main-experiment and Qwen3-VL checkpoints and the benchmark list reported in the paper.
- Add only verified published aggregate scores (for example, a compact CSV/table); do not copy raw run JSON or logs.
- Confirm licenses/terms for every benchmark and model linked here.

## Environment

Use a Python environment with a vLLM/Transformers combination that supports the selected checkpoint. Install `requirements-evaluation.txt`, then set the judge credentials and endpoint in a local `.env` or export them in the shell. The real `.env` is ignored by Git.

```bash
cp .env.example .env
# Fill in CORRECTION_PROJECT_API_KEY and, if needed, CORRECTION_PROJECT_API_URL.
```

The evaluator loads public benchmark datasets from their configured sources. For FREAK, set `FREAK_DATASET_ID` to the authorized dataset repository before selecting that benchmark.

## Run local checkpoint inference and judging

From `perception/evaluation/`, pass a locally accessible checkpoint or an authorized Hugging Face model ID. Select only benchmarks reported by the paper.

```bash
python evaluation_all.py \
  --checkpoint /path/to/checkpoint \
  --benchmarks mathvista,mathverse,mathvision,wemath \
  --out-dir /path/to/local/results
```

The command generates predictions and invokes the configured judge pipeline. Outputs are local artifacts and are excluded from version control. The benchmark list above is illustrative and must be checked against the final paper before release.

## Contents

- `perception/evaluation/evaluation_all.py`: shared inference, benchmark loading, and judge orchestration.
- `perception/evaluation/judge_*.py`: judge modules imported by that entry point. Keep these together unless the evaluator is refactored to load judges lazily.
- `.env.example`: names of optional runtime settings without secrets.

This folder is a local review candidate. It has not been published to GitHub and is not yet a complete, version-pinned reproduction package.
