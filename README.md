# MIRA: Reinforcing Multimodal Reasoning via Deceptive Contextual Augmentation

**Official code repository · NeurIPS 2026**

[Paper](https://neurips.cc/virtual/2026/poster/153690) · [Method](#method) · [Results](#main-results) · [Training](#training) · [Evaluation](#evaluation)

MIRA (**MIsleading-context Robustness Augmentation**) trains vision-language models to recover from misleading visual interpretations during reasoning. This repository includes the complete VERL-based training framework, MIRA's data-construction and reward code, and benchmark inference/scoring scripts.

## Abstract

Reinforcement learning with verifiable rewards has advanced multimodal reasoning, but errors in visual interpretation can still propagate through subsequent reasoning and produce incorrect conclusions. MIRA addresses this perception–reasoning gap by training models with filtered, deceptive visual contexts. These contexts simulate plausible initial misinterpretations and create challenging examples in which the model must recover from misleading information. A group-wise reflection-triggering mechanism encourages the policy to revisit visual evidence and resolve inconsistencies between its initial context and the problem. The resulting training signal helps the model internalize visual self-correction without human-annotated reasoning traces or an additional supervised cold-start stage. The paper evaluates MIRA on mathematical reasoning and vision-centric benchmarks, reporting an average improvement of approximately seven percentage points across the four mathematical benchmarks over the base model.

## Method

1. **Construct deceptive context.** Generate a question-relevant image description, then introduce plausible errors in visual details.
2. **Filter ineffective perturbations.** Retain misleading descriptions that affect answer correctness, so the augmented examples provide a useful recovery signal.
3. **Train visual self-correction.** Mix clean and augmented examples. Within an augmented example's rollout group, insert reflection triggers into a subset of continuations and optimize their relative rewards.
4. **Evaluate generalization.** Measure performance on mathematical reasoning and vision-centric tasks using the included inference and scoring pipeline.

The implementation includes answer correctness, correction of misleading details, formatting, and repetition terms. The main reward implementation and Qwen3 formatting adaptation are documented under [`mira/rewards/`](mira/rewards).

## Main results

Paper-reported accuracy (%) with Qwen2.5-VL-7B as the backbone:

| Model | MathVista | MathVerse | MathVision | WeMath | MMMU-Pro | HallusionBench | FREAK | Average |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 68.2 | 46.7 | 24.3 | 61.4 | 38.5 | 65.7 | 30.0 | 47.8 |
| **MIRA** | **75.2** | **53.8** | **29.0** | **70.7** | **41.0** | **69.8** | **30.6** | **52.9** |

The average column follows the paper's reported aggregate. See the paper for the complete comparisons, ablations, and benchmark settings. Qwen3-VL-4B and 8B training entry points are included below.

## Repository structure

```text
mira/
  training/          Main and Qwen3-VL launchers; reflection triggers
  rewards/           Correction reward and Qwen3 formatting adaptation
  data/              Adaptive sampler and data preparation
  evaluation/        Benchmark inference and scoring
  configs/           Static trigger phrases
  legacy/            Historical research implementations and configurations
verl/                Complete training framework, workers, rollouts, and utilities
tests/               Framework tests
examples/            Framework examples
docker/              Environment definitions
scripts/             Framework utilities
docs/                Installation and framework documentation
```

Current training and evaluation use the functional packages under `mira/`. Historical implementations remain in [`mira/legacy/`](mira/legacy/README.md) for reference. The full `verl/` source is bundled and can be installed directly from this checkout.

## Installation

Use a CUDA environment with compatible PyTorch, vLLM, Transformers, Ray, and FlashAttention builds. See the included [installation guide](docs/start/install.rst) and [`docker/`](docker) for environment definitions. In an already compatible environment:

```bash
git clone https://github.com/Hans-M-Yin/mira-neurips26.git
cd mira-neurips26
pip install -r requirements-training.txt
pip install --no-deps -e .
```

Install the appropriate vLLM build separately for your CUDA stack. The local software versions inspected during packaging are recorded in the [release notes](RELEASE_NOTES.md); they are not a lockfile for every experiment in the paper.

For optional API-based evaluation or data construction, copy `.env.example` to a local `.env`, fill in the relevant settings, and load them explicitly:

```bash
set -a
source .env
set +a
```

## Data preparation

The data-construction pipeline includes caption generation, visual-detail perturbation, verification, filtering, and conversion to VERL training Parquet. See [`mira/data/preparation/README.md`](mira/data/preparation/README.md) for input modes, service configuration, and commands.

Prepared training/validation data and model checkpoints are not bundled with this code repository. Their availability and redistribution terms are separate from the code license. Raw experiment JSON, predictions, metrics records, logs, and credentials are excluded from the public source tree.

## Training

Set the paths to prepared training and validation Parquet files:

```bash
export TRAIN_FILE=/path/to/train.parquet
export VAL_FILE=/path/to/validation.parquet
```

Run the main Qwen2.5-VL-7B recipe:

```bash
bash mira/training/train_mira.sh
```

Or select a Qwen3-VL recipe:

```bash
bash mira/training/train_qwen3_vl_4b.sh
bash mira/training/train_qwen3_vl_8b.sh
```

The main recipe defaults to `Qwen/Qwen2.5-VL-7B-Instruct`, a `Qwen/Qwen3-VL-32B-Instruct` reward model, and eight GPUs. The 8B Qwen3 recipe uses one epoch and a 3000-token response budget. Override resources and paths through environment variables such as `MODEL_PATH`, `RM_MODEL_PATH`, `N_GPUS`, `OUTPUT_DIR`, `TRAIN_BATCH_SIZE`, and `MAX_RESPONSE_LENGTH`.

Append Hydra overrides when needed. For example, inspect a configuration without starting a training job:

```bash
bash mira/training/train_mira.sh --cfg job
```

Set `DATASET_ADAPTIVE_RATIO=true` to enable the optional adaptive sampler. Logging defaults to the console; configure W&B locally and set `TRAINER_LOGGER='[console,wandb]'` to enable it. Checkpoints and rollout artifacts are written under `outputs/` by the current launchers.

## Evaluation

Install `requirements-evaluation.txt` in a compatible inference environment, then run from the repository root:

```bash
python -m mira.evaluation.evaluation_all \
  --checkpoint /path/to/checkpoint \
  --benchmarks mathvista,mathverse,mathvision,wemath,mmmu_pro_10,mmmu_pro_vision,hallusionbench,freak \
  --out-dir /path/to/local/results
```

The runner shares one local model instance across benchmarks and runs the corresponding scoring pipeline. Set `JUDGE_API_KEY` and `JUDGE_API_BASE_URL` for API-based answer extraction/judging. For FREAK, set `FREAK_DATASET_ID` to an authorized dataset repository. Use the paper's benchmark splits and inference settings when comparing against reported results.

## Citation

```bibtex
@inproceedings{mira2026,
  title={MIRA: Reinforcing Multimodal Reasoning via Deceptive Contextual Augmentation},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026},
  url={https://neurips.cc/virtual/2026/poster/153690}
}
```

See the [paper page](https://neurips.cc/virtual/2026/poster/153690) for the final author list and citation metadata.

## Acknowledgments and license

MIRA builds on [VERL](https://github.com/verl-project/verl). The full framework and upstream copyright/license notices are retained; its original overview is available [here](docs/UPSTREAM_VERL_README.md).

Code is licensed under [Apache-2.0](LICENSE). Datasets, checkpoints, and benchmark assets retain their respective licenses and access terms.
