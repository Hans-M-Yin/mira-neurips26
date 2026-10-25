# MIRA: Reinforcing Multimodal Reasoning via Deceptive Contextual Augmentation

Official code release for the NeurIPS 2026 paper [MIRA](https://neurips.cc/virtual/2026/poster/153690).

MIRA improves multimodal reasoning by training vision-language models to recover from misleading visual context. The training pipeline combines clean and context-perturbed examples, group-wise reflection triggers, and a reward that scores answer correctness, output format, repetition, and correction of misleading details.

## Contents

- `perception/training/reward_function.py`: reward function and model-assisted correction judge used by the main training recipe.
- `perception/training/train_mira.sh`: Qwen2.5-VL-7B GRPO training entry point.
- `perception/training/adaptive_dataset.py`: optional dataset sampler for balancing clean and augmented examples.
- `perception/utils/triggers.json`: reflection trigger phrases used during training.
- `verl/trainer/ppo/rollout_corr_helper.py`: rollout-correction implementation used by this VERL fork.
- `perception/evaluation/`: local inference and benchmark judge scripts.

This repository contains code and configuration only. It does not include model checkpoints, datasets, per-example evaluation outputs, raw experiment JSON, logs, or credentials. Obtain the training data and checkpoints from the authorized sources described by the paper and their respective licenses.

## Environment

Training requires a CUDA environment with the versions of PyTorch, Transformers, vLLM, Ray, and other packages supported by this VERL revision. `requirements-training.txt` lists the direct Python dependencies. The complete VERL framework is available from the [VERL project](https://github.com/volcengine/verl); apply the MIRA-specific rollout-correction helper and configuration in this repository to a compatible VERL checkout. A multi-GPU setup is expected for the paper recipe.

The training reward uses the configured local reward-model service supplied by VERL. Set any service credentials in your shell or a local `.env` file that is not committed. The reward endpoint and judge model are passed by the training framework; no API key is embedded in this repository.

## Train

Prepare the paper's training and validation Parquet files in the expected VERL format, then run:

```bash
TRAIN_FILE=/path/to/train.parquet \
VAL_FILE=/path/to/validation.parquet \
bash perception/training/train_mira.sh
```

The default actor is `Qwen/Qwen2.5-VL-7B-Instruct`; the default reward model is `Qwen/Qwen3-VL-32B-Instruct`. Override paths and resource settings through environment variables, for example `MODEL_PATH`, `RM_MODEL_PATH`, `N_GPUS`, `OUTPUT_DIR`, `TRAIN_BATCH_SIZE`, and `MAX_RESPONSE_LENGTH`. Append Hydra overrides after the script name. Set `DATASET_ADAPTIVE_RATIO=true` to enable the optional adaptive sampler. The script stores rollout artifacts under `outputs/`, which is ignored by Git.

The release script captures the main model and optimization settings. Exact reproduction also depends on the paper's data construction, dataset versions, model access, hardware, and service configuration.

## Evaluate

Install evaluation dependencies from `requirements-evaluation.txt`, make authorized benchmark datasets available, and run from `perception/evaluation/`:

```bash
python evaluation_all.py \
  --checkpoint /path/to/checkpoint \
  --benchmarks mathvista,mathverse,mathvision,wemath,mmmu_pro_10,mmmu_pro_vision,hallusionbench,freak \
  --out-dir /path/to/local/results
```

Set `JUDGE_API_KEY` and `JUDGE_API_BASE_URL` in the process environment for API-based judges. Evaluation outputs are written to the selected output directory and are not part of the release.

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

Code is released under the license in `LICENSE`. Datasets, checkpoints, and benchmark assets are governed by their own licenses and terms.
