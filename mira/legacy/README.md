# Historical research implementations

The supported paper-release entry points are documented in the repository's root
README and live in `mira/training`, `mira/rewards`, `mira/data`, and `mira/evaluation`.
They do not import the historical package to implement the current training recipe.

This directory preserves earlier research source and configurations:

| Directory | Origin and purpose |
|---|---|
| `initial/` | The original `perception/` development code: earlier reward variants, dated training launchers, and model serving/conversion helpers. |
| `revised/` | The original `perception_2/` development code: subsequent reward variants, main-run configurations, adaptive-data experiments, and rebuttal launchers. |

The two original names recorded development iterations; they did not represent two
independent components of the paper. The active code has been placed in functional
packages, while these historical implementations are retained for reference.

Historical scripts preserve their experiment settings and require the corresponding
user-provided data/model paths. For current training, use the root README's launchers.
Private data, checkpoints, outputs, and local analysis utilities remain excluded.
