# MIRA: Reinforcing Multimodal Reasoning via Deceptive Contextual Augmentation

This repository contains the evaluation inference entry point and benchmark judge scripts accompanying the NeurIPS 2026 paper. It is a code-only release: model checkpoints, datasets, run-level outputs, and training or ablation configurations are not included. Published aggregate results are reported in the paper; no private experiment JSON or logs are distributed here.

## Contents

- `perception/evaluation/evaluation_all.py`: local-model inference, benchmark loading, and judge orchestration.
- `perception/evaluation/judge_*.py`: benchmark judge modules imported by the evaluation entry point.
- `requirements-evaluation.txt`: direct Python dependencies for this evaluation subset.
- `.env.example`: names of optional runtime settings, without credentials.

## Setup

Use a Python environment with a compatible vLLM and Transformers installation for the selected model. Install the listed direct dependencies, then provide judge credentials through environment variables or a local `.env` file. The real `.env` is excluded by `.gitignore`.

```bash
pip install -r requirements-evaluation.txt
cp .env.example .env
# Fill in CORRECTION_PROJECT_API_KEY and, if needed, CORRECTION_PROJECT_API_URL.
```

The evaluation code loads benchmark data from its configured public or authorized sources. For FREAK, set `FREAK_DATASET_ID` to the authorized dataset repository before selecting that benchmark.

## Run evaluation

From `perception/evaluation/`, provide a locally accessible checkpoint or an authorized Hugging Face model ID. Select benchmarks according to the paper's evaluation protocol.

```bash
python evaluation_all.py \
  --checkpoint /path/to/checkpoint \
  --benchmarks mathvista,mathverse,mathvision,wemath \
  --out-dir /path/to/local/results
```

The benchmark list is an example; use the exact benchmark subset and settings for the result you intend to reproduce. Predictions and judge outputs are written locally and are not tracked in this repository.

## Security and release scope

Do not commit API keys, private dataset identifiers, local paths, model weights, or per-example experiment outputs. Keep credentials in `.env` or environment variables. This repository intentionally does not provide private checkpoints or raw experiment results.

## License

See `LICENSE` for the license applying to the code in this repository. Benchmark datasets and model checkpoints remain subject to their own licenses and access terms.
