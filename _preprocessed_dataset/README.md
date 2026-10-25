# Data construction and preprocessing

This directory contains the project's data-processing source. Generated datasets,
sample images, metadata, selection decisions, and validation reports are excluded.

`data_preprocess_v2.py` generates image descriptions, edits relevant visual details,
verifies misleading descriptions, and writes training/validation Parquet files in
the schema expected by the MIRA training code. Its caption model uses a local
OpenAI-compatible service; its editing and verification model uses a configured
Azure OpenAI deployment. Configure your own endpoint and keys through environment
variables. The original internal gateway and routing identifiers are removed.

For example, after starting an authorized caption-model service:

```bash
export AZURE_OPENAI_API_KEY=...       # Set locally; do not commit credentials.
export AZURE_OPENAI_ENDPOINT=...      # Your deployment endpoint.
python _preprocessed_dataset/data_preprocess_v2.py \
  --dataset_path /path/to/input_dataset \
  --local-api-url http://localhost:8000/v1 \
  --target_model YOUR_SERVED_CAPTION_MODEL \
  --model YOUR_AZURE_DEPLOYMENT \
  --save_dir /path/to/private_output
```

Use `--help` for input modes, resuming a run, and other settings. `--part` optionally
reads excluded indices from the path supplied through `PREPROCESS_EXCLUDE_INDICES`.
API credentials are removed from the metadata written by the preprocessing script.

`select_virl_mmk_by_rollout.py` contains the ViRL-39K/MMK12 selection pipeline used
for the Qwen3 experiments. `augment_dataset.py`, `make_vanilla_from_augmented.py`,
and the other conversion/validation utilities are retained with their original
behavior. For example, `augment_dataset.py` adds clean counterparts of augmented
samples; it is a separate postprocessing operation from the caption pipeline.

The retained scripts include historical dataset variants. Use the construction
and filtering settings specified for the experiment you intend to reproduce.
