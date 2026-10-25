# MIRA: Reinforcing Multimodal Reasoning via Deceptive Contextual Augmentation

**Official implementation · NeurIPS 2026**

[Paper](https://neurips.cc/virtual/2026/poster/153690) · [Overview](#overview) · [Results](#main-results) · [Installation](#installation) · [Data](#data-preparation) · [Training](#training) · [Evaluation](#evaluation) · [Citation](#citation)

MIRA (**MIsleading-context Robustness Augmentation**) trains vision-language models to recover from misleading visual interpretations during reasoning. It combines deceptive context construction, group-wise reflection triggers, and reinforcement learning with correction-aware rewards. The method requires neither human-annotated reasoning traces nor an additional supervised cold-start stage.

This repository provides the **complete VERL-based training framework**, the MIRA implementation, data-construction tools, checkpoint conversion, and benchmark inference and scoring. Training recipes cover **Qwen2.5-VL-7B**, **Qwen3-VL-4B**, and **Qwen3-VL-8B**.

## Contents

- [Overview](#overview)
- [Main results](#main-results)
- [Release contents and artifact availability](#release-contents-and-artifact-availability)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Environment configuration](#environment-configuration)
- [Data preparation](#data-preparation)
- [Training](#training)
- [Exporting a checkpoint](#exporting-a-checkpoint)
- [Evaluation](#evaluation)
- [Reproducibility notes](#reproducibility-notes)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Acknowledgments and license](#acknowledgments-and-license)

## Overview

### Motivation

An incorrect visual interpretation can propagate through an otherwise coherent reasoning chain. Rewarding only the final answer does not explicitly teach a model to recognize and repair the visual error that caused the failure. MIRA creates training situations in which recovery from a misleading interpretation is necessary to solve the original problem.

### How MIRA works

1. **Generate question-relevant visual context.** Describe the image details needed to answer the question.
2. **Construct deceptive context.** Edit selected visual facts to create plausible but incorrect descriptions, retaining the corresponding correct/incorrect fact pairs.
3. **Verify the perturbation.** Use a stronger verifier to retain contexts for which the correct description supports a correct answer and the deceptive description induces an incorrect answer.
4. **Train recovery with group-wise reflection.** Present deceptive context as an assistant prefill. Within an augmented example's rollout group, add reflection triggers to a subset of continuations so that the policy can compare different recovery behaviors under GRPO.
5. **Reward successful correction.** Combine answer correctness with correction, format, and repetition terms. The correction contribution is gated by final-answer correctness in the released implementation.

At inference time, the trained policy answers the original image/question input. Data-construction services and the training reward model are not part of benchmark inference; some benchmark scoring procedures use a separate API judge.

### Reward implemented in this release

The main reward implementation is [`mira/rewards/correction.py`](mira/rewards/correction.py):

```text
reward = accuracy
       + FORMAT_COEF × format_score
       + CORRECTION_COEF × accuracy × correction_score
       + REPETITION_PENALTY × repetition_score
```

| Component | Implementation behavior |
|---|---|
| `accuracy` | Binary answer-correctness reward, using answer matching and a model-assisted fallback. |
| `format_score` | `0` for a valid format and `-1` for a format violation; accounts for the assistant prefill on augmented samples. |
| `correction_score` | Judge-based assessment of explicit correction of the recorded misleading facts; applies to deceptive-context samples. Its contribution is zero when the final answer is incorrect. |
| `repetition_score` | A nonpositive penalty for excessive repetition. |

The default coefficients are `FORMAT_COEF=0.4`, `CORRECTION_COEF=0.2`, and `REPETITION_PENALTY=0.8`. The [Qwen3 adapter](mira/rewards/qwen3_vl.py) shares the correctness and correction logic while adapting format validation to Qwen3-VL outputs.

The reflection schedule is implemented in [`mira/training/triggers.py`](mira/training/triggers.py) and the [PPO trainer](verl/trainer/ppo/ray_trainer.py). Trigger phrases are stored in [`mira/configs/triggers.json`](mira/configs/triggers.json).

## Main results

**Paper-reported accuracy (%) with Qwen2.5-VL-7B as the backbone.** MIRA improves the mean accuracy across the four mathematical benchmarks by approximately **7 percentage points** over the base model.

| Model | MathVista | MathVerse | MathVision | WeMath | MMMU-Pro | HallusionBench | FREAK | Average |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 68.2 | 46.7 | 24.3 | 61.4 | 38.5 | 65.7 | 30.0 | 47.8 |
| **MIRA** | **75.2** | **53.8** | **29.0** | **70.7** | **41.0** | **69.8** | **30.6** | **52.9** |

The Average column reproduces the paper's reported aggregate. MMMU-Pro is evaluated using the 10-option and vision subsets; the runner exposes their scores separately. Consult the [paper](https://neurips.cc/virtual/2026/poster/153690) for aggregation details, comparisons with other methods, and ablations. The Qwen3-VL recipes below are separate experiment configurations; the table above describes the Qwen2.5-VL main experiment.

## Release contents and artifact availability

| Component | Availability |
|---|---|
| MIRA training, reward, reflection, and data code | Included under [`mira/`](mira). |
| Complete VERL framework | Included under [`verl/`](verl), with framework examples, tests, utilities, and environment definitions. |
| Main and Qwen3-VL training recipes | Included under [`mira/training/`](mira/training). |
| Benchmark inference and scoring | Included under [`mira/evaluation/`](mira/evaluation). |
| Trained MIRA weights | Distributed separately from the source code; public download links have not yet been published in this repository. |
| Prepared MIRA training/validation data | Distributed separately from the source code; public download links have not yet been published in this repository. |
| Public source datasets and benchmarks | Dataset identifiers are used by the corresponding loaders; access and licenses are governed by their providers. |

Full experiment JSON/JSONL, per-example predictions, saved metrics, training logs, checkpoints, datasets, and credentials are excluded from the public source tree. The scripts generate their own local outputs when run. See [`RELEASE_NOTES.md`](RELEASE_NOTES.md) for the release scope and [`RELEASE_MANIFEST.txt`](RELEASE_MANIFEST.txt) for the source file list.

## Repository structure

```text
.
├── mira/
│   ├── training/
│   │   ├── train_mira.sh                # Qwen2.5-VL-7B main recipe
│   │   ├── train_qwen3_vl_4b.sh         # Qwen3-VL-4B recipe
│   │   ├── train_qwen3_vl_8b.sh         # Qwen3-VL-8B recipe
│   │   └── triggers.py                 # Group-wise reflection triggers
│   ├── rewards/
│   │   ├── correction.py               # Main answer/correction reward
│   │   └── qwen3_vl.py                 # Qwen3 format adaptation
│   ├── data/
│   │   ├── adaptive_dataset.py         # Optional adaptive clean/augmented sampling
│   │   └── preparation/                # Construction, selection, and validation
│   ├── evaluation/
│   │   ├── evaluation_all.py           # Shared-model inference/scoring runner
│   │   └── judge_*.py                  # Benchmark answer extraction and judging
│   ├── configs/triggers.json
│   └── legacy/                         # Historical research implementations
├── verl/                               # Complete training framework
├── tests/                              # Framework tests
├── examples/                           # Upstream framework examples
├── docker/                             # CUDA/backend environment definitions
├── scripts/                            # Framework utilities
├── docs/                               # Framework documentation
├── .env.example                        # Local configuration template
├── requirements-training.txt
└── requirements-evaluation.txt
```

Use the current functional packages under `mira/` for the workflows below. Earlier development versions are documented in [`mira/legacy/README.md`](mira/legacy/README.md). The bundled framework is installed directly from this checkout; MIRA's trainer integration and runtime resources are already included.

## Installation

### Hardware and software

Training uses NVIDIA GPUs, FSDP2, asynchronous vLLM rollouts, and a model-based reward service managed by VERL. The main launcher defaults to **one node with eight GPUs** and a **Qwen3-VL-32B-Instruct reward model**. GPU memory requirements depend on the actor, reward model, sequence lengths, batch sizes, and parallelism settings.

Use Python **3.10 or newer** and mutually compatible CUDA, PyTorch, vLLM, Transformers, and FlashAttention builds. The included [installation guide](docs/start/install.rst) and [`docker/`](docker) describe CUDA/backend setup. Install the CUDA-dependent packages before the commands below.

The environment inspected during release preparation used these versions:

| Package | Observed version |
|---|---|
| Python | 3.11 |
| PyTorch | 2.8.0 |
| Transformers | 4.57.3 |
| vLLM | 0.11.0 |
| Ray | 2.54.0 |
| TensorDict | 0.10.0 |

These versions document the inspected environment; they are not a historical lockfile for every paper experiment.

### Install the repository

Run the following inside your CUDA-compatible Python environment:

```bash
git clone https://github.com/Hans-M-Yin/mira-neurips26.git
cd mira-neurips26

python -m pip install -r requirements-training.txt
python -m pip install --no-deps -e .
```

The editable installation includes both `verl` and `mira`. Run subsequent commands from the repository root so that relative configuration and reward-module paths resolve correctly.

For a separate inference/evaluation environment, install its CUDA-compatible vLLM stack, then:

```bash
python -m pip install -r requirements-evaluation.txt
python -m pip install --no-deps -e .
python -m mira.evaluation.evaluation_all --help
```

## Environment configuration

Create a local configuration file from the checked-in template:

```bash
cp .env.example .env
# Edit .env for your dataset paths, models, and services.
set -a
source .env
set +a
```

The scripts read environment variables; they do not automatically load `.env`.

| Variable | Used for |
|---|---|
| `TRAIN_FILE`, `VAL_FILE` | Prepared training and validation Parquet files. Required by all training launchers. |
| `MODEL_PATH` | Actor model identifier or local Hugging Face model directory. |
| `RM_MODEL_PATH` | Local training judge/reward model; defaults to `Qwen/Qwen3-VL-32B-Instruct`. |
| `JUDGE_API_KEY`, `JUDGE_API_BASE_URL` | OpenAI-compatible answer extraction and judging for evaluation. |
| `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` | Azure deployment used for data editing and verification. |
| `FREAK_DATASET_ID` | Dataset repository to use for the FREAK benchmark. Set before starting the evaluation process. |
| `N_GPUS`, `TRAINER_LOGGER` | Training GPU count and logging backends. |

Training's `RM_MODEL_PATH` is independent of evaluation's API judge settings. Data construction additionally uses an OpenAI-compatible caption-model service configured through command-line arguments. A populated `.env` is ignored by Git; `.env.example` contains the shareable configuration template.

## Data preparation

### Use prepared training data

If you already have prepared MIRA data, set the two Parquet paths and proceed to [training](#training):

```bash
export TRAIN_FILE=/absolute/path/to/train.parquet
export VAL_FILE=/absolute/path/to/test.parquet
```

Here `VAL_FILE` is held-out training validation data. In the data builder's output it is named `test.parquet`; it is separate from the benchmark test sets used for final evaluation.

The launchers expect the multimodal VERL schema produced by the data builder:

| Field | Purpose |
|---|---|
| `prompt` | Chat messages, including the image/question and an assistant prefill for context-augmented samples. |
| `images` | List of images in the Hugging Face image representation. |
| `data_source`, `ability` | Dataset/task routing metadata. |
| `reward_model.ground_truth` | Reference answer used for correctness verification. |
| `extra_info.type` | `0` for clean samples, `1` for deceptive-context samples; historical variants also support `2` for correct-context prefills. |
| `extra_info.question` | Original question for reward evaluation. |
| `extra_info.modify_info`, `extra_info.num_modify` | Correct/incorrect fact pairs and their count for the correction judge. |
| Other `extra_info` fields | Captions, source indices, split information, and construction metadata retained by the builder. |

Preserve augmentation metadata and the prompt structure when converting your own data. A plain image/question/answer Parquet file still needs preprocessing before it can be used with the MIRA correction reward.

### Construct deceptive-context data

[`build_training_data.py`](mira/data/preparation/build_training_data.py) generates captions, edits visual details, verifies the misleading contexts, and exports training/validation Parquet files. It uses two services:

- An **OpenAI-compatible multimodal caption service**, specified by `--local-api-url` and `--target_model`.
- An **Azure OpenAI deployment** for editing and verification, configured through `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, and `--model`. `--guide_model` can select a separate editing/verification deployment.

Supply a Hugging Face dataset identifier or a local `Dataset.save_to_disk` directory containing `problem`, `answer`, and `images` fields. For example, after starting the caption service and configuring Azure in your local environment:

```bash
python -m mira.data.preparation.build_training_data \
  --dataset_path /absolute/path/to/source_dataset \
  --local-api-url http://localhost:8000/v1 \
  --target_model YOUR_SERVED_CAPTION_MODEL \
  --model YOUR_AZURE_DEPLOYMENT \
  --save_dir ./outputs/data \
  --experiment_name mira \
  --seed 42 \
  --test_size 0.02 \
  --parallel_num 16 \
  --qpm 120
```

The script prints a timestamped output directory containing `train.parquet`, `test.parquet`, construction metadata, and a log. Point `TRAIN_FILE` and `VAL_FILE` to the files inside that directory. `--qpm` limits aggregate Azure requests across workers, including retries; supported values are 1–120.

Additional input modes are available through `--help`. `--from_parquet` regenerates contexts from a prepared Parquet file; `--resume_from` skips source indices already present in the supplied training Parquet file. Choose caption/expert models and filtering settings for the experiment you intend to reproduce.

### Optional Qwen3-VL sample selection

The Qwen3 data-selection tool combines ViRL-39K and MMK12, generates five responses per sample by default, and keeps samples answered correctly zero to four times, removing samples answered correctly in all five rollouts.

With an OpenAI-compatible correctness judge configured:

```bash
python -m mira.data.preparation.select_virl_mmk_by_rollout \
  --rollout-model Qwen/Qwen3-VL-8B-Instruct \
  --tensor-parallel-size 2 \
  --rollouts-per-sample 5 \
  --keep-min-correct 0 \
  --keep-max-correct 4 \
  --judge-api-url "$JUDGE_API_BASE_URL" \
  --judge-model YOUR_SERVED_JUDGE_MODEL \
  --output-dir ./outputs/selected_data
```

The judge key is read from `JUDGE_API_KEY`. Use `./outputs/selected_data/dataset` as the data builder's `--dataset_path`; `selected.parquet` and selection records are also written. `--max-samples` limits a trial run and `--resume` reuses saved judgments.

Further conversion and validation utilities are described in the [data preparation README](mira/data/preparation/README.md). In particular, `augment_dataset.py` **adds clean counterparts of augmented samples** as a separate postprocessing operation, changing the training mixture. Apply it only when that operation belongs to the intended experiment recipe.

## Training

### Select a recipe

Set `TRAIN_FILE` and `VAL_FILE` before invoking a launcher. The following commands explicitly select the actor and experiment name, including when `.env` contains a different `MODEL_PATH`.

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

| Recipe | Reward implementation | Epochs | Maximum response tokens |
|---|---|---:|---:|
| Qwen2.5-VL-7B | `mira/rewards/correction.py` | 2 | 1500 |
| Qwen3-VL-4B | `mira/rewards/qwen3_vl.py` | 2 | 1500 |
| Qwen3-VL-8B | `mira/rewards/qwen3_vl.py` | 1 | 3000 |

These are current launcher defaults. Environment variables take precedence over launcher defaults, including in the Qwen3 wrappers.

### Configure resources and optimization

All recipes use `train_mira.sh` as the shared launcher. Its principal defaults are:

| Environment variable | Default | Meaning |
|---|---|---|
| `N_GPUS` | `8` | GPUs on the training node. |
| `RM_MODEL_PATH` | `Qwen/Qwen3-VL-32B-Instruct` | Model used by the training reward loop. |
| `TRAIN_BATCH_SIZE` | `128` | Training prompt batch size. |
| `ROLLOUT_N` | `8` | Responses sampled per prompt. |
| `PPO_MINI_BATCH_SIZE` | `64` | Actor optimization minibatch size. |
| `PPO_MICRO_BATCH_SIZE` | `16` | Actor microbatch size per GPU. |
| `MAX_PROMPT_LENGTH` | `5500` | Maximum prompt length. |
| `LR` | `1e-6` | Actor learning rate. |
| `KL_COEF` | `0.005` | KL loss coefficient. |
| `TENSOR_PARALLEL_SIZE` | `2` | Actor rollout tensor parallelism. |
| `RM_TENSOR_PARALLEL_SIZE` | `4` | Reward-model tensor parallelism. |
| `GPU_MEMORY_UTIL` | `0.65` | Actor rollout vLLM memory utilization. |
| `RM_GPU_MEMORY_UTIL` | `0.7` | Reward-model vLLM memory utilization. |
| `TRIGGER_WARMUP_RATIO` | `0.3` | Reflection schedule warmup fraction. |
| `TRIGGER_START_RATIO`, `TRIGGER_FINAL_RATIO` | `0.6`, `0.4` | Initial/final reflection-trigger ratios. |
| `SAVE_FREQ`, `TEST_FREQ` | `120`, `3` | Checkpoint and validation intervals in training steps. |
| `OUTPUT_DIR` | `outputs/$EXP_NAME` | Root directory for checkpoints and generated records. |
| `TRAINER_LOGGER` | `[console]` | Logging backends. |

Adjust GPU count, parallelism, sequence lengths, and batch sizes together for your hardware. The 32B reward model also consumes GPU memory. Inspect the [launcher](mira/training/train_mira.sh) for the full set of overrides.

Additional Hydra overrides can be appended to a launcher command. To inspect the resolved configuration without starting training:

```bash
TRAIN_FILE=/absolute/path/to/train.parquet \
VAL_FILE=/absolute/path/to/test.parquet \
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct \
bash mira/training/train_mira.sh --cfg job
```

Set `DATASET_ADAPTIVE_RATIO=true` to enable the optional [adaptive dataset](mira/data/adaptive_dataset.py). Logging defaults to the console; after configuring W&B locally, use `TRAINER_LOGGER='[console,wandb]'` to add W&B logging.

### Training outputs

The default output layout for the main recipe is:

```text
outputs/mira-qwen2.5-vl-7b/
├── checkpoints/
│   └── global_step_<N>/
│       └── actor/                      # Distributed actor checkpoint
├── train/                              # Saved rollout records
└── val/                                # Saved validation records
```

Checkpoint contents depend on the configured backend. Use the actor checkpoint for [export](#exporting-a-checkpoint); optimizer and distributed training state are needed when resuming training.

## Exporting a checkpoint

Convert an FSDP actor checkpoint into a Hugging Face model directory before passing it to the evaluation runner:

```bash
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /absolute/path/to/checkpoints/global_step_120/actor \
  --target_dir ./outputs/merged_mira
```

Choose a step that exists in your training output. The exported directory is the inference artifact; it is distinct from the distributed checkpoint used for training recovery. If you already have a complete merged Hugging Face model, use that directory directly.

See the included [checkpoint documentation](docs/advance/checkpoint.rst) and `python -m verl.model_merger merge --help` for other backends and export options.

## Evaluation

### Run inference and scoring

The unified runner loads one vLLM model instance, generates responses, and runs the matching benchmark scoring pipeline. Configure `JUDGE_API_KEY` and `JUDGE_API_BASE_URL` for the API-scored benchmarks below.

For example, evaluate the exported model on the four mathematical benchmarks:

```bash
python -m mira.evaluation.evaluation_all \
  --checkpoint ./outputs/merged_mira \
  --benchmarks mathvista,mathverse,mathvision,wemath \
  --tensor-parallel-size 2 \
  --judge-model-name gpt-4o \
  --judge-parallel-num 10 \
  --judge-temperature 0 \
  --seed 0 \
  --out-dir ./outputs/evaluation/math
```

`--checkpoint` also accepts an accessible Hugging Face model identifier. To evaluate all main-paper benchmarks, set `FREAK_DATASET_ID` in your environment and use:

```bash
python -m mira.evaluation.evaluation_all \
  --checkpoint ./outputs/merged_mira \
  --benchmarks mathvista,mathverse,mathvision,wemath,mmmu_pro_10,mmmu_pro_vision,hallusionbench,freak \
  --tensor-parallel-size 2 \
  --judge-model-name gpt-4o \
  --out-dir ./outputs/evaluation/main
```

The same entry point accepts merged Qwen3-VL checkpoints. Match prompt and decoding settings to the experiment being evaluated when comparing backbones.

### Supported benchmarks

These identifiers and splits follow the current [benchmark registry](mira/evaluation/evaluation_all.py):

| CLI name | Dataset source | Configuration / split | Scoring |
|---|---|---|---|
| `mathvista` | `AI4Math/MathVista` | `testmini` | API answer extraction/judging. |
| `mathverse` | `AI4Math/MathVerse` | Config `testmini`, split `testmini` | API answer extraction/judging. |
| `mathvision` | `MathLLMs/MathVision` | `test` | API answer extraction/judging. |
| `wemath` | `We-Math/We-Math` | `testmini` | API answer extraction/judging. |
| `mmmu_pro_10` | `MMMU/MMMU_Pro` | Config `standard (10 options)`, split `test` | API answer extraction/judging. |
| `mmmu_pro_vision` | `MMMU/MMMU_Pro` | Config `vision`, split `test` | API answer extraction/judging. |
| `hallusionbench` | `lmms-lab/HallusionBench` | `image` | API answer extraction/judging. |
| `freak` | Environment variable `FREAK_DATASET_ID` | `test` | API answer extraction/judging. |
| `logicvista` | `lscpku/LogicVista` | `test` | API answer extraction/judging. |
| `mmstar` | `Lin-Chen/MMStar` | `val` | Local evaluator. |
| `pope` | `lmms-lab/POPE` or matching local official assets | `test`; random/popular/adversarial | Local evaluator. |
| `vstar` | Local V*Bench assets | `--vstar-data-dir` | Local evaluator. |

POPE accepts `--pope-annotation-dir` and `--pope-image-dir` together for local assets; otherwise its loader uses the dataset repository. V*Bench requires `--vstar-data-dir`. Additional benchmarks are evaluation utilities and are separate from the main results table.

The runner currently constructs an API client even for selections using only local evaluators. For such runs, `JUDGE_API_KEY=EMPTY` satisfies client initialization; the local evaluators do not make judge requests.

### Generation settings and saved results

The current unified runner uses greedy generation (`temperature=0`, `top_k=1`, one response) with up to **4096 generated tokens**. Its vLLM model configuration uses a 20,000-token context limit and `gpu_memory_utilization=0.85`. These settings are defined in `build_sampling_params()` and `build_shared_llm()`; they are not exposed as generation CLI flags.

Each run writes benchmark-specific prediction files, scoring records, and `evaluation_all_summary.json` under `--out-dir`. The summary contains success/failure status, sample counts, accuracy, elapsed time, and output paths. Inspect the status for each benchmark before using its score. `--continue-on-error` allows later benchmarks to run after a failure.

### Score saved responses

The standalone [`judge_*.py` scripts](mira/evaluation) can score existing responses without rerunning model inference. For example, to score a MathVista prediction file produced by the unified runner:

```bash
python -m mira.evaluation.judge_mathvista \
  --eval_file ./outputs/evaluation/math/mathvista.json \
  --model_name gpt-4o \
  --parallel_num 10
```

This command reads the judge API credentials from the environment and writes scoring records beside the prediction file. Each judge script has its own input conventions; consult its `--help` when scoring externally generated responses. To list all unified-runner options:

```bash
python -m mira.evaluation.evaluation_all --help
```

## Reproducibility notes

The main results table reports measurements from the paper. Launcher defaults and the observed release environment are documented separately above; some settings differ across experiment versions. For a comparable run, retain the chosen source revision, dataset construction/split, actor and reward-model revisions, resolved training configuration, and evaluation prompts, splits, decoding, and judge settings.

In particular, match the clean/augmented data mixture and preprocessing operations to the target experiment. Running optional selection, adaptive sampling, or clean-counterpart augmentation changes that recipe. The two MMMU-Pro subsets are reported individually by the runner, so use the paper's aggregation convention when comparing its table.

CLI help and Hydra configuration inspection can be used without starting a GPU training job. They check the command/configuration interface; full training and model-quality reproduction require the corresponding GPU environment, datasets, and model artifacts.

## Troubleshooting

| Symptom | What to check |
|---|---|
| Launcher asks for `TRAIN_FILE` or `VAL_FILE` | Set both Parquet paths before invocation, including for `--cfg job`. If using `.env`, load it explicitly. |
| Qwen3 launcher selects an unexpected backbone | An exported `MODEL_PATH` overrides the wrapper default. Use the explicit recipe commands above. |
| `mira` or `verl` cannot be imported | Install this checkout with `python -m pip install --no-deps -e .` in the active environment and run from its root. |
| vLLM/CUDA import or attention-kernel errors | Check the CUDA, PyTorch, vLLM, and FlashAttention builds against the included installation guide. |
| Out-of-memory during training | Review actor/reward parallelism, batch sizes, prompt/response lengths, and both vLLM memory allocations together. |
| Evaluation judge authentication fails | Check the evaluation API key, base URL, and `--judge-model-name`. Azure data-construction settings configure a different service. |
| FREAK fails before generation | Set `FREAK_DATASET_ID` to a dataset repository you can access before starting the runner. |
| A benchmark has no valid score | Inspect its `status` and error in `evaluation_all_summary.json`, dataset access, and judge outputs. |

For reproducible bug reports, open a [GitHub issue](https://github.com/Hans-M-Yin/mira-neurips26/issues) with the source revision, software versions, redacted command/configuration, and relevant error message.

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

See the [official paper page](https://neurips.cc/virtual/2026/poster/153690) for the final author list and full citation metadata.

## Acknowledgments and license

MIRA builds on [VERL](https://github.com/verl-project/verl), Qwen vision-language models, and the datasets and benchmarks acknowledged in the paper. We thank their authors and maintainers. The bundled framework retains its upstream copyright notices, licenses, examples, and documentation; its original overview is available in [`docs/UPSTREAM_VERL_README.md`](docs/UPSTREAM_VERL_README.md).

The code is licensed under [Apache-2.0](LICENSE); the upstream framework license is also retained as [`LICENSE-VERL`](LICENSE-VERL). Model weights, datasets, and benchmark assets are subject to their respective licenses and access terms.
