"""Kaggle runner for T25: EfficientNet-B0 + Lung-Region Attention A0-A5.

Typical Kaggle setup and usage from notebook cells:

    %cd /kaggle/working
    !git clone https://github.com/Kusal-Nirukshan/chest-x-ray-kaggle.git
    %cd chest-x-ray-kaggle
    !pip install -q timm grad-cam wandb PyYAML scikit-learn pandas matplotlib seaborn

    !python scripts/kaggle_t25_efficientnet_b0_lung_attention.py \
      --repo-root /kaggle/working/chest-x-ray-kaggle \
      --data-dir "/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset" \
      --arms A0_vanilla A1_gate_only A4_guidance_only A2_full A3_multiply A5_cbam \
      --cam-subset-size 1000

For a quick smoke test:

    !python scripts/kaggle_t25_efficientnet_b0_lung_attention.py \
      --repo-root /kaggle/working/chest-x-ray-kaggle \
      --data-dir "/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset" \
      --quick \
      --arms A0_vanilla A2_full

Outputs are written under:
    artifacts/T25_efficientnet_b0_lung_attention/

The script creates:
    - runs/<arm>/ phase histories, metrics, confusion matrices, checkpoints
    - runs/<arm>/per_image_predictions.csv
    - T25_comparison_table.csv
    - acceptance_criteria_A1_A4.json
    - T25_efficiency.csv
    - T25_efficientnet_b0_lung_attention_results.zip
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
import torch
import torch.nn as nn


ARM_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "A0_vanilla": {
        "module.use_attention": False,
        "module.attention": "lung",
        "module.gate_mode": "none",
        "module.lambda_att": 0.0,
    },
    "A1_gate_only": {
        "module.use_attention": True,
        "module.attention": "lung",
        "module.gate_mode": "residual",
        "module.lambda_att": 0.0,
    },
    "A4_guidance_only": {
        "module.use_attention": True,
        "module.attention": "lung",
        "module.gate_mode": "none",
        "module.lambda_att": 1.0,
    },
    "A2_full": {
        "module.use_attention": True,
        "module.attention": "lung",
        "module.gate_mode": "residual",
        "module.lambda_att": 1.0,
    },
    "A3_multiply": {
        "module.use_attention": True,
        "module.attention": "lung",
        "module.gate_mode": "multiply",
        "module.lambda_att": 1.0,
    },
    "A5_cbam": {
        "module.use_attention": True,
        "module.attention": "cbam",
        "module.gate_mode": "multiply",
        "module.lambda_att": 0.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run T25 EfficientNet-B0 lung-attention ablation on Kaggle.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--arms", nargs="+", default=list(ARM_OVERRIDES.keys()), choices=list(ARM_OVERRIDES.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cam-subset-size", type=int, default=1000)
    parser.add_argument("--skip-cam", action="store_true", help="Skip Grad-CAM per-image EIL columns.")
    parser.add_argument("--skip-efficiency", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B via src.utils initialize_wandb.")
    parser.add_argument("--quick", action="store_true", help="Short smoke-test schedule; not for final results.")
    return parser.parse_args()


def apply_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    for dotted_key, value in overrides.items():
        target = cfg
        parts = dotted_key.split(".")
        for key in parts[:-1]:
            target = target[key]
        target[parts[-1]] = value
    return cfg


def import_repo(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def load_checkpoint_state(path: Path) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    return ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt


def build_model_from_cfg(cfg: Dict[str, Any]):
    from src.modules import build_model

    m = cfg["module"]
    return build_model(
        num_classes=cfg["model"]["num_classes"],
        use_attention=m["use_attention"],
        attention=m.get("attention", "lung"),
        gate_mode=m.get("gate_mode", "residual"),
        reduction=m.get("reduction", 16),
        backbone_name=cfg["model"]["name"],
        pretrained=False,
        drop_rate=cfg["model"].get("drop_rate", 0.0),
    )


def write_module_card(output_dir: Path, config: Dict[str, Any], comparison: pd.DataFrame | None) -> None:
    lines = [
        "# T25 EfficientNet-B0 Lung-Region Attention Module Card",
        "",
        f"Seed: `{config['experiment']['seed']}`",
        f"Backbone: `{config['model']['name']}`",
        f"Reduction: `{config['module']['reduction']}`",
        f"Drop rate: `{config['model'].get('drop_rate', 0.0)}`",
        "",
        "This artifact was generated by `scripts/kaggle_t25_efficientnet_b0_lung_attention.py`.",
        "",
    ]
    if comparison is not None:
        lines.extend(["## Comparison Table", "", "```csv", comparison.to_csv(index=False).strip(), "```", ""])
    (output_dir / "T25_module_card.md").write_text("\n".join(lines), encoding="utf-8")


def run_efficiency(output_dir: Path, cfg: Dict[str, Any], arm_cfgs: Dict[str, Dict[str, Any]]) -> None:
    from notebooks.efficiency import benchmark_model, results_to_table
    from src.modules import LogitsOnly

    rows = []
    for arm_name, arm_cfg in arm_cfgs.items():
        ckpt_path = output_dir / "runs" / arm_name / f"efficientnet_b0_{arm_name}.pt"
        model = build_model_from_cfg(arm_cfg)
        if ckpt_path.exists():
            model.load_state_dict(load_checkpoint_state(ckpt_path), strict=True)
        rows.append(
            benchmark_model(
                LogitsOnly(model),
                model_name="EfficientNet-B0",
                module_status=arm_name,
                checkpoint_path=ckpt_path if ckpt_path.exists() else None,
                cpu_runs=20,
                gpu_runs=50,
            )
        )
    results_to_table(rows, output_dir / "T25_efficiency.csv")


def main() -> None:
    args = parse_args()
    import_repo(args.repo_root)

    import timm
    from src.datasets import build_dataloaders, compute_class_weights
    from src.modules import (
        build_comparison_table,
        build_per_image_predictions,
        check_acceptance_criteria,
        run_full_arm,
        stratified_cam_subset,
    )
    from src.utils import load_config, set_seed

    config_path = args.config or args.repo_root / "configs" / "efficientnet_b0_lung_attention.yaml"
    output_dir = args.output_dir or args.repo_root / "artifacts" / "T25_efficientnet_b0_lung_attention"
    manifest_path = args.repo_root / "artifacts" / "splits" / "split_manifest_v1.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path)
    cfg["experiment"]["seed"] = args.seed
    if args.quick:
        cfg["training"]["phase1_epochs"] = 1
        cfg["training"]["phase2_epochs"] = 1
        cfg["training"]["patience"] = 1
        args.cam_subset_size = min(args.cam_subset_size, 16)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Repo root: {args.repo_root.resolve()}")
    print(f"Data dir: {args.data_dir.resolve()}")
    print(f"Output dir: {output_dir.resolve()}")

    audit_model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=cfg["model"]["num_classes"])
    print(f"EfficientNet-B0 num_features according to this timm version: {audit_model.num_features}")
    del audit_model

    train_loader, val_loader, test_loader, class_names, train_targets, datasets = build_dataloaders(
        data_dir=args.data_dir,
        img_size=cfg["dataset"]["image_size"],
        batch_size=cfg["training"]["batch_size"],
        seed=args.seed,
        num_workers=args.num_workers,
        split_manifest_path=manifest_path,
    )
    print(f"Classes: {class_names}")
    print(f"Split sizes: train={len(datasets['train'])}, val={len(datasets['val'])}, test={len(datasets['test'])}")

    class_weights = compute_class_weights(train_targets, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    arm_cfgs: Dict[str, Dict[str, Any]] = {}
    arm_results: Dict[str, Dict[str, Any]] = {}
    for arm_name in args.arms:
        arm_cfg = apply_overrides(cfg, ARM_OVERRIDES[arm_name])
        arm_cfg["experiment"]["seed"] = args.seed
        arm_cfgs[arm_name] = arm_cfg

        print("\n" + "=" * 90)
        print(f"Running {arm_name}: {ARM_OVERRIDES[arm_name]}")
        print("=" * 90)
        arm_output = output_dir / "runs" / arm_name
        result = run_full_arm(
            arm_name=arm_name,
            arm_cfg=arm_cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_names=class_names,
            criterion=criterion,
            device=device,
            output_dir=arm_output,
            backbone_name="efficientnet_b0",
            wandb_enabled=args.wandb,
        )
        arm_results[arm_name] = result

    cam_subset = None
    if not args.skip_cam:
        cam_subset = stratified_cam_subset(datasets["test"], n=args.cam_subset_size, seed=args.seed)
        with (output_dir / "cam_subset_indices.json").open("w", encoding="utf-8") as f:
            json.dump([int(x) for x in cam_subset], f, indent=2)

    comparison_input: Dict[str, Dict[str, Any]] = {}
    for arm_name, arm_cfg in arm_cfgs.items():
        ckpt_path = output_dir / "runs" / arm_name / f"efficientnet_b0_{arm_name}.pt"
        model = build_model_from_cfg(arm_cfg)
        model.load_state_dict(load_checkpoint_state(ckpt_path), strict=True)
        model = model.to(device)

        per_image = build_per_image_predictions(
            model=model,
            test_dataset=datasets["test"],
            class_names=class_names,
            device=device,
            cam_subset=cam_subset,
            batch_size=cfg["training"]["batch_size"],
        )
        per_image.to_csv(output_dir / "runs" / arm_name / "per_image_predictions.csv", index=False)
        comparison_input[arm_name] = {
            "gate_mode": arm_cfg["module"]["gate_mode"],
            "lambda_att": arm_cfg["module"]["lambda_att"],
            "evaluate_results": arm_results[arm_name],
            "per_image_df": per_image,
        }
        model = model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    comparison = build_comparison_table(
        comparison_input,
        reference_arm="A0_vanilla" if "A0_vanilla" in comparison_input else None,
        output_csv=output_dir / "T25_comparison_table.csv",
    )
    print("\nT25 comparison table:")
    print(comparison.to_string(index=False))

    if "A2_full" in comparison["arm"].values:
        acceptance = check_acceptance_criteria(comparison, headline_arm="A2_full")
        with (output_dir / "acceptance_criteria_A1_A4.json").open("w", encoding="utf-8") as f:
            json.dump(acceptance, f, indent=2)

    if not args.skip_efficiency:
        run_efficiency(output_dir, cfg, arm_cfgs)

    write_module_card(output_dir, cfg, comparison)

    zip_base = output_dir.parent / "T25_efficientnet_b0_lung_attention_results"
    zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=output_dir)
    print(f"\nZipped results: {zip_path}")
    print("Download this zip from Kaggle Output, then extract/copy it back into the repo artifacts folder.")


if __name__ == "__main__":
    main()
