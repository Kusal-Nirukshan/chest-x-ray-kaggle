# Experiment Configurations

Every training run should be driven by a YAML config in this folder, not by
hardcoded values scattered through notebooks or scripts.

## Files

- `efficientnet_b0_lung_attention.yaml` - T25 EfficientNet-B0 lung-region
  attention ablation config. It uses `reduction=16` for the EfficientNet cost
  target; override `module.lambda_att` per sweep point instead of editing the
  baseline config.

- `baseline.yaml` — the reference configuration. Treat it as read-only; don't
  edit it to run a one-off experiment.

## Adding a new experiment config

1. Copy `baseline.yaml` to a new file named after what changed, e.g.:
   - `densenet121_baseline.yaml` — same experiment, different model
   - `resnet50_aug_v2.yaml` — same model, new augmentation pipeline
   - `resnet50_lr_1e-3.yaml` — a learning-rate sweep point
2. Change only the fields relevant to that experiment.
3. If you change `dataset.split_version` or `augmentation.version`, make sure
   the corresponding split manifest / augmentation code actually exists —
   these fields are labels that must match real, reproducible artifacts.
4. Never add `experiment.seed` values ad hoc. Use `42` for the standard
   baseline seed, and only deviate when deliberately testing seed
   sensitivity (see `docs/experiment_policy.md`).

## Loading a config

```python
from src.utils import load_config

config = load_config("configs/baseline.yaml")
```

`load_config` just parses YAML into a plain nested `dict` — no hidden
magic, so it's easy to read, override, and log to W&B.

To override a handful of values without creating a new file (e.g. in a
notebook cell while iterating):

```python
from src.utils.config import merge_overrides

config = merge_overrides(config, {"training.batch_size": 64, "model.name": "densenet121"})
```

## What must never go in a config file

API keys, tokens, passwords, or any other secret. These belong in
environment variables or platform secret managers — see
`docs/wandb_setup.md`.
