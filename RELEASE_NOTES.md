# Public release scope

The repository bundles the complete local VERL framework: all 458 source/configuration
files under `verl/` are present. It also includes framework tests, examples, Docker
definitions, utilities, installation metadata, documentation, MIRA training code,
and data-preparation source.

The PPO trainer's dependency on `mira.training.triggers` is included, together with
the reflection-trigger JSON files. Computation and logging code for training metrics
remains part of the framework. The release can be installed directly from this
checkout; it does not require applying fragments to another VERL revision.

## Source organization

The current paper implementation is organized under `mira/training`, `mira/rewards`,
`mira/data`, `mira/evaluation`, and `mira/configs`. The earlier development directories
have been moved into `mira/legacy/initial` and `mira/legacy/revised`; see their README
for the development history. All source files from the preceding release are retained,
and all 458 framework files remain present.

Training imports and launcher paths follow the current package layout. The Qwen3
format adapter imports the shared correction reward directly. Module-style and
direct-file evaluation commands are both supported. Moving these files does not
change the reward, sampler, or trigger function bodies.

## Annotation and filename cleanup

Code comments, Python docstrings, command-line help, and diagnostic messages are
written in English. Temporary development remarks have been removed, and ambiguous
or misspelled historical filenames have been corrected together with their references.
Historical experiment identifiers and optimization settings are retained.

Chinese text that forms part of model prompts, answer-matching expressions,
reflection triggers, or multilingual examples remains unchanged. The upstream
Chinese documentation is also retained as documentation, separately from code
annotations. The cleanup does not change these model inputs or the training logic.

## Software environment observed during packaging

The local packaging/validation environment uses Python 3.11, PyTorch 2.8.0,
Transformers 4.57.3, vLLM 0.11.0, Ray 2.54.0, and TensorDict 0.10.0. These versions
describe the environment inspected for this release; they are not a historical
lockfile for every paper experiment. Select compatible CUDA and optional backend
builds as described in the included installation documentation.

## Files excluded from public source control

- Checkpoints, merged weights, datasets, data samples, generated metadata, and caches.
- Per-example predictions, full experiment JSON/JSONL, saved rollout records, metrics
  records/plots, logs, and W&B run directories.
- Credentials, populated environment files, machine notes, private archive tools,
  and local resource-management scripts.
- Experiment-specific analysis scripts, summary tools, and their launchers.
- MIRA's historical vanilla GRPO comparison configurations, as requested for this
  paper release. Upstream framework training algorithms and examples are retained.

The six retained JSON files are static trigger/tool/model configurations. They do
not contain experiment results. Preprocessing scripts previously ignored along with
the datasets are included individually; their generated output remains ignored.

## Portability changes

- Internal service locations, routing identifiers, and machine-specific paths are
  removed or replaced with environment settings/documentation examples.
- The data-preprocessing pipeline accepts a caller-configured Azure deployment and
  excludes both Azure and local API keys from saved run metadata.
- Historical Python 3.12 f-string logging syntax is made compatible with Python 3.11
  without changing reward calculations or trigger behavior.
- New convenience launchers cover the main Qwen2.5-VL-7B recipe and Qwen3-VL-4B/8B.
  Historical training source is also retained. The adaptive dataset's import identity
  follows its actual module path for serialization across workers.
- Package data declarations include nested training configuration, trigger JSON,
  and launcher scripts. Package discovery includes namespace directories so their
  runtime modules are installed along with the rest of the framework.
- Upstream CI definitions are retained in `.github/upstream-workflows/` for reference;
  they do not run against this repository automatically.

## Validation

- All included Python files parse, and all included shell scripts pass `bash -n`.
- Main, adaptive-sampler, Qwen3-VL-4B, and Qwen3-VL-8B launchers compose their Hydra
  configuration without starting a training job.
- Key trainer, worker, reward, sampler, and model-merger modules import from this checkout.
- The existing rollout-correction tests pass: 11 tests, with one pre-existing pytest
  warning about a test returning a value.
- A local MIRA trigger check preserves clean samples and inserts triggers into the
  expected subset of augmented rollouts, synchronizing their raw prompts.
- The data-preprocessing command exposes its help without requiring a private service.
- A wheel builds locally, and its contents include the framework, nested configuration,
  MIRA helpers, trigger JSON, and the new training launchers.
- The selected public files are scanned for credential formats and original internal
  paths, hosts, and identifiers before publication.

Full GPU training, benchmark inference, and external judge requests were not rerun as
part of packaging. Access to the exact data/checkpoints and compatible hardware is
still required to reproduce the paper's experiments.
