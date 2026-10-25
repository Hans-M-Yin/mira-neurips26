# MIRA implementation

| Package | Purpose |
|---|---|
| `training/` | Main/Qwen3-VL launchers and the group-wise reflection-trigger helper. |
| `rewards/` | Answer/correction reward and Qwen3 answer-format adaptation. |
| `data/` | Optional adaptive sampler and data-construction/preprocessing source. |
| `evaluation/` | Benchmark inference, answer extraction, and scoring. |
| `configs/` | Static reflection-trigger phrases. |
| `legacy/` | Earlier research code and historical experiment configurations. |

The bundled `verl/` framework imports `mira.training.triggers` for rollout augmentation.
Current launchers select `mira/rewards/correction.py` or the Qwen3 formatting adapter,
and optionally `mira/data/adaptive_dataset.py`. See the repository root README for
installation and runnable commands.
