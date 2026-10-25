# MIRA: Reinforcing Multimodal Reasoning via Deceptive Contextual Augmentation

**Official implementation · NeurIPS 2026**

[Paper](https://neurips.cc/virtual/2026/poster/153690) · [Method](#method) · [Results](#main-results) · [Installation](#installation) · [Data](#data-preparation) · [Training](#training) · [Evaluation](#evaluation) · [Citation](#citation)

MIRA (**MIsleading-context Robustness Augmentation**) trains vision-language models to recover from misleading visual interpretations during reasoning. It combines deceptive context construction, group-wise reflection triggers, and correction-aware reinforcement learning, without human-annotated reasoning traces or an additional supervised cold-start stage.

This repository includes the **complete VERL-based training framework**, data-construction tools, and inference/scoring scripts, with training recipes for **Qwen2.5-VL-7B**, **Qwen3-VL-4B**, and **Qwen3-VL-8B**.

## Method

1. **Construct deceptive context.** Generate question-relevant image descriptions and edit selected visual facts, retaining the correct/incorrect fact pairs.
2. **Verify the perturbation.** Keep contexts for which a stronger verifier answers correctly with the correct description and incorrectly with the deceptive description.
3. **Train visual self-correction.** Present the deceptive context as an assistant prefill, add reflection triggers to a subset of each augmented example's rollout group, and optimize the policy with GRPO.
4. **Reward successful recovery.** Combine answer correctness, explicit correction, formatting, and repetition terms. In the released implementation, the correction contribution is gated by final-answer correctness.

The main implementation is in [`mira/rewards/correction.py`](mira/rewards/correction.py), with a [Qwen3 formatting adapter](mira/rewards/qwen3_vl.py). Reflection triggers are implemented in [`mira/training/triggers.py`](mira/training/triggers.py) and integrated into the [PPO trainer](verl/trainer/ppo/ray_trainer.py).

At inference time, the trained policy answers the original image/question input. Data-construction services and the training reward model are not required for generation; benchmark scoring can use a separate API judge.

## Main results

Paper-reported accuracy (%) with Qwen2.5-VL-7B as the backbone. MIRA improves the mean accuracy across the four mathematical benchmarks by approximately **7 percentage points** over the base model.

| Model | MathVista | MathVerse | MathVision | WeMath | MMMU-Pro | HallusionBench | FREAK | Average |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 68.2 | 46.7 | 24.3 | 61.4 | 38.5 | 65.7 | 30.0 | 47.8 |
| **MIRA** | **75.2** | **53.8** | **29.0** | **70.7** | **41.0** | **69.8** | **30.6** | **52.9** |

The Average column follows the paper's reported aggregate. The runner scores the two MMMU-Pro subsets separately. See the [paper](https://neurips.cc/virtual/2026/poster/153690) for aggregation details, comparisons, and ablations. Qwen3-VL launchers are included as separate experiment recipes.

## Repository structure

```text
mira/
  training/          Main and Qwen3-VL launchers; reflection triggers
  rewards/           Answer/correction reward and Qwen3 format adapter
  data/              Data construction, selection, and adaptive sampling
  evaluation/        Benchmark inference, answer extraction, and scoring
  configs/           Reflection-trigger phrases
  legacy/            Historical research implementations
verl/                Complete framework, workers, rollouts, and utilities
tests/               Framework tests
examples/            Framework examples
scripts/             Framework utilities
docker/              CUDA/backend environment definitions
docs/                Framework documentation
```

Use the functional packages under `mira/` for the workflows below. Earlier development versions are documented in [`mira/legacy/README.md`](mira/legacy/README.md). The bundled framework includes MIRA's trainer integration and is installed directly from this checkout.

## Installation

Use Python **3.10+**, NVIDIA GPUs, and compatible CUDA, PyTorch, vLLM, Transformers, and FlashAttention builds. Install the CUDA-dependent packages first, following the [installation guide](docs/start/install.rst) or [`docker/`](docker) definitions. Software versions observed during release preparation are recorded in the [release notes](RELEASE_NOTES.md).

```bash
git clone https://github.com/Hans-M-Yin/mira-neurips26.git
cd mira-neurips26
python -m pip install -r requirements-training.txt
python -m pip install --no-deps -e .
```

This installs both `verl` and `mira`. For a separate evaluation environment, install `requirements-evaluation.txt` and the editable package instead. Run the commands below from the repository root so that relative configuration and reward paths resolve correctly.

Configure local paths and services using the environment template:

```bash
cp .env.example .env
# Edit .env for your paths, models, and credentials.
set -a
source .env
set +a
```

The scripts do not load `.env` automatically. Training uses `TRAIN_FILE`, `VAL_FILE`, `MODEL_PATH`, and `RM_MODEL_PATH`; evaluation uses `JUDGE_API_KEY` and `JUDGE_API_BASE_URL`. Data construction uses a caption service plus `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT`. Populated `.env` files are ignored by Git.

## Data preparation

### Prepared data

Set the paths to your prepared MIRA Parquet files:

```bash
export TRAIN_FILE=/absolute/path/to/train.parquet
export VAL_FILE=/absolute/path/to/test.parquet
```

The builder names its held-out validation file `test.parquet`; this is separate from the final benchmark test sets. Training expects VERL's multimodal fields (`prompt`, `images`, and `reward_model.ground_truth`) plus MIRA's `extra_info`: sample type, question, captions, `modify_info`, and `num_modify`. Preserve the assistant prefill and fact-edit metadata when converting data for the correction reward.

### Construct deceptive-context data

The [data builder](mira/data/preparation/build_training_data.py) accepts a Hugging Face dataset identifier or a local `Dataset.save_to_disk` directory with `problem`, `answer`, and `images` fields. It generates captions with an OpenAI-compatible multimodal service, then edits and verifies them using a configured Azure deployment.

After starting your caption service and configuring Azure in `.env`:

```bash
python -m mira.data.preparation.build_training_data \
  --dataset_path /absolute/path/to/source_dataset \
  --local-api-url http://localhost:8000/v1 \
  --target_model YOUR_SERVED_CAPTION_MODEL \
  --model YOUR_AZURE_DEPLOYMENT \
  --save_dir ./outputs/data \
  --experiment_name mira \
  --seed 42 \
  --test_size 0.02
```

The script prints a timestamped directory containing `train.parquet`, `test.parquet`, and construction metadata. Point the training variables to those files. Use `--help` for resuming, prepared-Parquet input, concurrency, and rate limits.

For Qwen3 experiments, [`select_virl_mmk_by_rollout.py`](mira/data/preparation/select_virl_mmk_by_rollout.py) selects samples from ViRL-39K/MMK12 using repeated rollouts and a correctness judge. See the [data preparation guide](mira/data/preparation/README.md) for the available utilities. Optional selection, adaptive sampling, and `augment_dataset.py` change the training mixture; the latter adds clean counterparts of augmented samples.

## Training

Set `TRAIN_FILE` and `VAL_FILE`, then choose a recipe. Explicit model settings below override any different `MODEL_PATH` loaded from `.env`.

**Main experiment: Qwen2.5-VL-7B**

```bash
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct \
EXP_NAME=mira-qwen2.5-vl-7b \
bash mira/training/train_mira.sh
```

**Qwen3-VL-4B**

```bash
MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct \
EXP_NAME=mira-qwen3-vl-4b \
bash mira/training/train_qwen3_vl_4b.sh
```

**Qwen3-VL-8B**

```bash
MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct \
EXP_NAME=mira-qwen3-vl-8b \
bash mira/training/train_qwen3_vl_8b.sh
```

| Recipe | Epochs | Maximum response tokens | Reward module |
|---|---:|---:|---|
| Qwen2.5-VL-7B | 2 | 1500 | `correction.py` |
| Qwen3-VL-4B | 2 | 1500 | `qwen3_vl.py` |
| Qwen3-VL-8B | 1 | 3000 | `qwen3_vl.py` |

The shared launcher defaults to one node with **8 GPUs**, FSDP2, asynchronous vLLM rollouts, and `Qwen/Qwen3-VL-32B-Instruct` as the training reward model. Other defaults include batch size `128`, `8` rollouts per prompt, learning rate `1e-6`, and KL coefficient `0.005`.

Override settings through environment variables such as `N_GPUS`, `RM_MODEL_PATH`, `TRAIN_BATCH_SIZE`, `MAX_RESPONSE_LENGTH`, `TENSOR_PARALLEL_SIZE`, and `RM_TENSOR_PARALLEL_SIZE`. Resource settings must accommodate both the actor and the reward model. Environment variables take precedence over recipe defaults; additional Hydra overrides can be appended to the command.

Inspect the resolved configuration without starting training, after setting both data paths:

```bash
bash mira/training/train_mira.sh --cfg job
```

Checkpoints and saved training/validation records are written under `outputs/$EXP_NAME/`. Set `OUTPUT_DIR` to change the location, `DATASET_ADAPTIVE_RATIO=true` to enable the optional adaptive sampler, or `TRAINER_LOGGER='[console,wandb]'` to add locally configured W&B logging. See the [shared launcher](mira/training/train_mira.sh) for all parameters.

## Checkpoint export

Merge an FSDP actor checkpoint into a Hugging Face model directory before evaluation:

```bash
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /absolute/path/to/checkpoints/global_step_120/actor \
  --target_dir ./outputs/merged_mira
```

Choose a saved training step. If you already have merged Hugging Face weights, use them directly. Distributed training state is needed for resuming training; the merged directory is the inference artifact. Other export options are described in the [checkpoint guide](docs/advance/checkpoint.rst).

## Evaluation

The unified runner shares one vLLM instance across benchmarks and runs inference followed by answer extraction and scoring. Configure `JUDGE_API_KEY` and `JUDGE_API_BASE_URL` for the API judge, independently of training's reward-model settings.

```bash
python -m mira.evaluation.evaluation_all \
  --checkpoint ./outputs/merged_mira \
  --benchmarks mathvista,mathverse,mathvision,wemath \
  --tensor-parallel-size 2 \
  --judge-model-name gpt-4o \
  --out-dir ./outputs/evaluation
```

`--checkpoint` also accepts an accessible Hugging Face model identifier or merged Qwen3-VL checkpoint. For the complete main-paper evaluation, use this benchmark list and set `FREAK_DATASET_ID` to a dataset repository you can access **before starting the process**:

```text
mathvista,mathverse,mathvision,wemath,mmmu_pro_10,mmmu_pro_vision,hallusionbench,freak
```

The [benchmark registry](mira/evaluation/evaluation_all.py) defines dataset configurations and splits. Additional utilities support `logicvista`, `mmstar`, `pope`, and `vstar`; V*Bench requires local assets via `--vstar-data-dir`.

Current generation defaults are greedy decoding with up to **4096 generated tokens**, a 20,000-token context limit, and vLLM memory utilization `0.85`. Generation settings are defined in `build_sampling_params()` and `build_shared_llm()`. Match the paper's prompts, dataset splits, decoding, and judge settings when comparing results; launcher defaults can differ across experiment versions.

Outputs include benchmark predictions, scoring records, and `evaluation_all_summary.json` under `--out-dir`. Inspect each benchmark's status and sample count before using its accuracy. Add `--continue-on-error` to process later benchmarks after a failure, or use `--help` for all runner options.

To score saved MathVista responses without rerunning inference:

```bash
python -m mira.evaluation.judge_mathvista \
  --eval_file ./outputs/evaluation/mathvista.json \
  --model_name gpt-4o \
  --parallel_num 10
```

Other [`judge_*.py` scripts](mira/evaluation) provide the corresponding benchmark scorers; consult their `--help` for input conventions. Judge credentials are read from the environment, and scoring records are written beside the input predictions.

## Model and data availability

| Artifact | Hugging Face repository | Access |
|---|---|---|
| MIRA Qwen2.5-VL-7B checkpoint | [hansQAQ/checkpoint-main-private](https://huggingface.co/hansQAQ/checkpoint-main-private) | Private |
| Training data archive | [hansQAQ/dataset-archive-private](https://huggingface.co/datasets/hansQAQ/dataset-archive-private) | Private |

Both repositories are currently private and require an authorized Hugging Face account. The main training file is `training/main_augmented/train.parquet` in the data archive. Dataset and model access terms are governed by their providers.

Full experiment JSON/JSONL, predictions, metrics records, logs, checkpoints, datasets, and credentials are excluded from public source control. See the [release notes](RELEASE_NOTES.md) and [file manifest](RELEASE_MANIFEST.txt) for the release scope.

## Citation

If you use MIRA in your research, please cite the paper:

```bibtex
@inproceedings{mira2026,
  title     = {{MIRA}: Reinforcing Multimodal Reasoning via Deceptive Contextual Augmentation},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://neurips.cc/virtual/2026/poster/153690}
}
```

See the [official paper page](https://neurips.cc/virtual/2026/poster/153690) for the final author list and citation metadata.

## Acknowledgments and license

MIRA builds on [VERL](https://github.com/verl-project/verl), Qwen vision-language models, and the datasets and benchmarks acknowledged in the paper. Upstream notices are retained; the original framework overview is available [here](docs/UPSTREAM_VERL_README.md).

Code is licensed under [Apache-2.0](LICENSE), with the upstream license retained as [`LICENSE-VERL`](LICENSE-VERL). Weights, datasets, and benchmark assets retain their respective licenses and access terms.
