# MIRA: Reinforcing Multimodal Reasoning via Deceptive Contextual Augmentation

Official code release for the NeurIPS 2026 paper [MIRA](https://neurips.cc/virtual/2026/poster/153690).

MIRA improves multimodal reasoning by training vision-language models to recover from misleading visual context. The training pipeline combines clean and context-perturbed examples, group-wise reflection triggers, and a reward that scores answer correctness, output format, repetition, and correction of misleading details.

## Contents

- `perception/training/reward_function.py`: reward function and model-assisted correction judge used by the main training recipe.
- `perception/training/train_mira.sh`: Qwen2.5-VL-7B GRPO training entry point; adjacent launchers cover Qwen3-VL-4B/8B.
- `perception/training/adaptive_dataset.py`: optional dataset sampler for balancing clean and augmented examples.
- `perception/utils/triggers.json`: reflection trigger phrases used during training.
- `verl/`: the complete local VERL fork, including trainers, workers, FSDP/Megatron backends, rollout engines, model support, checkpoint merging, configuration, and utilities.
- `perception/utils/utils.py`: MIRA's reflection-trigger insertion and scheduling, imported by the PPO trainer.
- `perception/` and `perception_2/`: reward implementations, training configurations, model serving/conversion tools, and evaluation code.
- `_preprocessed_dataset/`: data-construction and preprocessing source, without generated datasets or run artifacts.
- `perception/evaluation/`: local inference and benchmark judge scripts.
- `tests/`, `examples/`, `docker/`, `scripts/`, and `docs/`: framework tests, examples, environments, utilities, and documentation.

The full framework source is bundled here; there is no need to copy patches into a separate VERL checkout. Training metric computation and logging implementations are included. Generated metrics, checkpoints, private datasets, per-example evaluation outputs, raw experiment JSON, logs, credentials, and local analysis scripts are excluded. `RELEASE_MANIFEST.txt` lists the published files, and `RELEASE_NOTES.md` explains the exclusions.

## Environment

Training requires a CUDA environment with compatible PyTorch, Transformers, vLLM, Ray, and FlashAttention builds. The included [installation guide](docs/start/install.rst) and `docker/` definitions describe the framework environments. For an already compatible environment:

```bash
git clone https://github.com/Hans-M-Yin/mira-neurips26.git
cd mira-neurips26
pip install -r requirements-training.txt
pip install --no-deps -e .
```

vLLM is installed separately for your CUDA stack; the local environment inspected during packaging uses PyTorch 2.8.0, Transformers 4.57.3, and vLLM 0.11.0. These are packaging-environment observations, not a pinned record of every paper experiment. The default training recipe uses eight GPUs. See `requirements*.txt`, `setup.py`, and [VERL's original overview](docs/UPSTREAM_VERL_README.md) for optional backends.

The training reward uses the configured local reward-model service supplied by VERL. Set any service credentials in your shell or a local `.env` file that is not committed. The reward endpoint and judge model are passed by the training framework; no API key is embedded in this repository.

## Train

Prepare the paper's training and validation Parquet files in the expected VERL format, then run:

```bash
TRAIN_FILE=/path/to/train.parquet \
VAL_FILE=/path/to/validation.parquet \
bash perception/training/train_mira.sh
```

The default actor is `Qwen/Qwen2.5-VL-7B-Instruct`; the default reward model is `Qwen/Qwen3-VL-32B-Instruct`. Override paths and resource settings through environment variables, for example `MODEL_PATH`, `RM_MODEL_PATH`, `N_GPUS`, `OUTPUT_DIR`, `TRAIN_BATCH_SIZE`, and `MAX_RESPONSE_LENGTH`. Append Hydra overrides after the script name. Set `DATASET_ADAPTIVE_RATIO=true` to enable the optional adaptive sampler. The script stores rollout artifacts under `outputs/`, which is ignored by Git.

For the Qwen3-VL recipes, use the same dataset-path variables with:

```bash
bash perception/training/train_qwen3_vl_4b.sh
# Or the 8B recipe (one epoch and a 3000-token response budget):
bash perception/training/train_qwen3_vl_8b.sh
```

Logging defaults to the console. Set `TRAINER_LOGGER='[console,wandb]'` and configure W&B locally if needed. To inspect the composed training configuration without starting a job, append `--cfg job` to a launcher.

The release script captures the main model and optimization settings. Exact reproduction also depends on the paper's data construction, dataset versions, model access, hardware, and service configuration.

## Prepare data

Data-construction, filtering, and Parquet conversion code is included under `_preprocessed_dataset/`; see its [README](_preprocessed_dataset/README.md). It includes the original caption/edit/verification prompts. The data files and the private Hugging Face archives created during project backup are not published by this code release.

## Evaluate

Install evaluation dependencies from `requirements-evaluation.txt`, make authorized benchmark datasets available, and run from `perception/evaluation/`:

```bash
python evaluation_all.py \
  --checkpoint /path/to/checkpoint \
  --benchmarks mathvista,mathverse,mathvision,wemath,mmmu_pro_10,mmmu_pro_vision,hallusionbench,freak \
  --out-dir /path/to/local/results
```

Set `JUDGE_API_KEY` and `JUDGE_API_BASE_URL` in the process environment for API-based judges. Evaluation outputs are written to the selected output directory and are not part of the release.

Use `.env.example` as a template if desired. The scripts read the process environment; load a local `.env` explicitly with `set -a; source .env; set +a`. For FREAK evaluation, set `FREAK_DATASET_ID` to the dataset repository you can access.

## Results

The paper reports the following main comparison for Qwen2.5-VL-7B (accuracy, %):

| Model | MathVista | MathVerse | MathVision | WeMath | MMMU-Pro | HallusionBench | FREAK | Average |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 68.2 | 46.7 | 24.3 | 61.4 | 38.5 | 65.7 | 30.0 | 47.8 |
| MIRA | 75.2 | 53.8 | 29.0 | 70.7 | 41.0 | 69.8 | 30.6 | 52.9 |

See the paper for full baseline comparisons, ablations, and evaluation details.

## Citation

```bibtex
@inproceedings{mira2026,
  title={MIRA: Reinforcing Multimodal Reasoning via Deceptive Contextual Augmentation},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## License

Code is released under the Apache-2.0 license in `LICENSE`. This repository builds on [VERL](https://github.com/verl-project/verl); upstream copyright and license notices are retained. Datasets, checkpoints, and benchmark assets are governed by their own licenses and terms.
