# MIRA: Reinforcing Multimodal Reasoning via Deceptive Contextual Augmentation

Official evaluation-code release for the NeurIPS 2026 paper **MIRA: Reinforcing Multimodal Reasoning via Deceptive Contextual Augmentation**.

[Paper page (NeurIPS 2026)](https://neurips.cc/virtual/2026/poster/153690)

## Abstract

Reinforcement learning with verifiable rewards (RLVR) can improve multimodal reasoning, but a model may still over-rely on an incorrect visual interpretation and allow that error to compound during reasoning. MIRA (Misleading-context Robustness Augmentation) trains a vision-language model to recover from such errors. It constructs deceptive visual contexts by perturbing image descriptions, filters examples where the perturbation does not affect the model's answer, and uses a group-wise reflection trigger during reinforcement-learning rollouts to encourage the model to revisit visual evidence. The method uses automatically generated data rather than human-annotated chains of thought. The paper evaluates MIRA on mathematical reasoning and vision-centric benchmarks.

## Method at a glance

1. Generate a description of the image content relevant to a question.
2. Create a plausible but misleading perturbation and retain examples that change the model's answer.
3. Train with a mixture of clean and augmented examples. Within rollout groups, some augmented examples receive a reflection trigger while others do not, giving the policy a relative learning signal for self-correction.
4. Evaluate answer accuracy and visual reasoning using the benchmark pipeline in this repository.

## Main results

The paper reports results using Qwen2.5-VL-7B as the backbone. The table below reproduces the main comparison from the manuscript; scores are accuracy (%). The reported average covers the seven listed benchmarks.

| Model | MathVista | MathVerse | MathVision | WeMath | MMMU-Pro | HallusionBench | FREAK | Average |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 68.2 | 46.7 | 24.3 | 61.4 | 38.5 | 65.7 | 30.0 | 47.8 |
| MIRA | 75.2 | 53.8 | 29.0 | 70.7 | 41.0 | 69.8 | 30.6 | 52.9 |

MIRA improves the four mathematical benchmarks by an average of about 7 percentage points over the base model. The full paper contains the complete baseline comparison, ablations, and evaluation details.

## Repository contents

- `perception/evaluation/evaluation_all.py`: shared local-model inference, benchmark loading, and judge orchestration.
- `perception/evaluation/judge_*.py`: benchmark-specific scoring and judging modules imported by the runner.
- `requirements-evaluation.txt`: direct Python dependencies for this evaluation subset.
- `.env.example`: optional runtime setting names, without credentials.

This is an evaluation-code release. It does not include model checkpoints, datasets, training jobs, ablation configurations, per-example predictions, raw experiment JSON, or logs. Checkpoint availability and dataset access are governed by their respective repositories and licenses.

## Setup

Use an environment with vLLM and Transformers versions compatible with the selected model and your CUDA stack. Install the listed direct dependencies, configure the judge API if needed, and make the environment variables available to the process:

```bash
pip install -r requirements-evaluation.txt
cp .env.example .env
# Edit .env locally, then load it into the current shell:
set -a
source .env
set +a
```

Never commit the populated `.env`. For FREAK evaluation, set `FREAK_DATASET_ID` to an authorized dataset repository.

## Run evaluation

From `perception/evaluation/`, provide a locally accessible checkpoint or an authorized Hugging Face model ID. The benchmark names below match the main-paper evaluation; the two MMMU-Pro subsets are selected separately by this runner.

```bash
python evaluation_all.py \
  --checkpoint /path/to/checkpoint \
  --benchmarks mathvista,mathverse,mathvision,wemath,mmmu_pro_10,mmmu_pro_vision,hallusionbench,freak \
  --out-dir /path/to/local/results
```

The runner writes predictions and judge outputs to the selected output directory. These local artifacts are excluded from version control. Use the exact benchmark splits and settings specified in the paper when reproducing a reported result.

## Citation

```text
MIRA: Reinforcing Multimodal Reasoning via Deceptive Contextual Augmentation.
Advances in Neural Information Processing Systems (NeurIPS), 2026.
```

See the [NeurIPS paper page](https://neurips.cc/virtual/2026/poster/153690) for the final author list and citation metadata.

## License

See `LICENSE` for this repository's code license. Benchmark datasets and model checkpoints remain subject to their own licenses and access terms.
