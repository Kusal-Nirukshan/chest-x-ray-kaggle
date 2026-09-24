"""Kaggle runner for the trustworthiness baseline-vs-guided comparison.

This is the final research framing runner:

    vanilla baseline
        vs.
    anatomically/lung-guided model

for one CNN backbone at a time. The older A0-A5 scripts remain available as
method-development ablations and are not overwritten by this runner.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline-vs-lung-guided trustworthiness pair.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--backbone", choices=["densenet121", "resnet50", "efficientnet_b0"], required=True)
    parser.add_argument("--guided-config", type=Path, default=None)
    parser.add_argument("--guided-mode", choices=["residual", "multiply", "none"], default=None)
    parser.add_argument("--lambda-att", type=float, default=None)
    parser.add_argument("--lambda-bg", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cam-subset-size", type=int, default=1000)
    parser.add_argument("--skip-cam", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-counterfactual", action="store_true")
    parser.add_argument("--skip-efficiency", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Short smoke-test schedule; not final results.")
    return parser.parse_args()


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


def _flatten_efficiency(row: Dict[str, Any]) -> Dict[str, Any]:
    aliases = {
        "params": ("params_total", "params", "parameters", "num_params"),
        "gflops": ("gflops", "flops_g", "GFLOPs"),
        "cpu_latency_ms": ("cpu_latency_ms", "cpu_ms", "latency_cpu_ms"),
        "gpu_latency_ms": ("gpu_latency_ms", "gpu_ms", "latency_gpu_ms"),
    }
    out = {}
    for target, keys in aliases.items():
        out[target] = next((row[k] for k in keys if k in row), float("nan"))
    return out


def run_efficiency(model, version: str, ckpt_path: Path) -> Dict[str, Any]:
    from notebooks.efficiency import benchmark_model
    from src.modules import LogitsOnly

    row = benchmark_model(
        LogitsOnly(model),
        model_name=model.backbone_name,
        module_status=version,
        checkpoint_path=ckpt_path,
        cpu_runs=20,
        gpu_runs=50,
    )
    return _flatten_efficiency(row)


def main() -> None:
    args = parse_args()
    import_repo(args.repo_root)

    from src.datasets import build_dataloaders, compute_class_weights
    from src.modules import (
        LogitsOnly,
        build_master_row,
        build_statistics_report,
        build_per_image_predictions,
        calibration_report,
        default_config_path,
        evaluate_counterfactual_robustness,
        load_guided_config,
        make_pair_configs,
        resolve_guided_hparams,
        run_epoch,
        run_full_arm,
        seed_output_dir,
        stratified_cam_subset,
        summarize_multiseed_master,
        upsert_master_comparison,
        validate_backbone_config,
        validation_selection_row,
        write_master_comparison,
        write_validation_selection_table,
    )
    from src.utils import load_config, set_seed

    if args.quick:
        print("SMOKE TEST ONLY - NOT FINAL RESULTS")

    config_path = args.config or default_config_path(args.repo_root, args.backbone)
    base_output_dir = args.output_dir or args.repo_root / "artifacts" / "trustworthiness"
    manifest_path = args.repo_root / "artifacts" / "splits" / "split_manifest_v1.csv"

    cfg = load_config(config_path)
    validate_backbone_config(cfg, args.backbone, config_path)
    if args.quick:
        cfg["training"]["phase1_epochs"] = 1
        cfg["training"]["phase2_epochs"] = 1
        cfg["training"]["patience"] = 1
        args.cam_subset_size = min(args.cam_subset_size, 16)

    selected_guided = load_guided_config(args.guided_config) if args.guided_config else None
    guided_hparams = resolve_guided_hparams(
        selected_guided,
        guided_mode=args.guided_mode,
        lambda_att=args.lambda_att,
        lambda_bg=args.lambda_bg,
    )
    seeds = args.seeds or [args.seed]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Backbone: {args.backbone}")
    print(f"Config: {config_path.resolve()}")
    print(f"Guided hparams: {guided_hparams}")

    for seed in seeds:
        output_dir = seed_output_dir(base_output_dir, args.backbone, seed)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSeed {seed} output dir: {output_dir.resolve()}")

        pair_cfgs = make_pair_configs(
            cfg,
            backbone=args.backbone,
            guided_mode=guided_hparams["guided_mode"],
            lambda_att=guided_hparams["lambda_att"],
            lambda_bg=guided_hparams["lambda_bg"],
            seed=seed,
        )

        set_seed(seed)
        train_loader, val_loader, test_loader, class_names, train_targets, datasets = build_dataloaders(
            data_dir=args.data_dir,
            img_size=cfg["dataset"]["image_size"],
            batch_size=cfg["training"]["batch_size"],
            seed=seed,
            num_workers=args.num_workers,
            split_manifest_path=manifest_path,
        )
        print(f"Classes: {class_names}")
        print(f"Split sizes: train={len(datasets['train'])}, val={len(datasets['val'])}, test={len(datasets['test'])}")

        criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train_targets, len(class_names)).to(device))

        eval_results: Dict[str, Dict[str, Any]] = {}
        master_rows = []
        validation_rows = []

        for version in ("baseline", "guided"):
            version_dir = output_dir / version
            arm_name = version
            arm_cfg = pair_cfgs[version]
            print("\n" + "=" * 90)
            print(f"Running {version}: module={arm_cfg['module']}")
            print("=" * 90)

            eval_results[version] = run_full_arm(
                arm_name=arm_name,
                arm_cfg=arm_cfg,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                class_names=class_names,
                criterion=criterion,
                device=device,
                output_dir=version_dir,
                backbone_name=args.backbone,
                wandb_enabled=args.wandb,
            )

            ckpt_path = version_dir / f"{args.backbone}_{arm_name}.pt"
            model = build_model_from_cfg(arm_cfg)
            model.load_state_dict(load_checkpoint_state(ckpt_path), strict=True)
            model = model.to(device)

            cam_subset = None
            if not args.skip_cam:
                cam_subset = stratified_cam_subset(datasets["test"], n=args.cam_subset_size, seed=seed)
                with (version_dir / "cam_subset_indices.json").open("w", encoding="utf-8") as f:
                    json.dump([int(x) for x in cam_subset], f, indent=2)

            per_image = build_per_image_predictions(
                model=model,
                test_dataset=datasets["test"],
                class_names=class_names,
                device=device,
                cam_subset=cam_subset,
                batch_size=cfg["training"]["batch_size"],
            )
            per_image.to_csv(version_dir / "per_image_predictions.csv", index=False)

            if args.skip_counterfactual:
                cf_summary = pd.DataFrame(columns=["mode", "mean_stability", "pred_flip_rate"])
            else:
                cf_summary = evaluate_counterfactual_robustness(
                    model=model,
                    loader=test_loader,
                    device=device,
                    seed=seed,
                    arm_name=version,
                    output_csv=version_dir / "counterfactual_summary.csv",
                    per_image_csv=version_dir / "counterfactual_per_image.csv",
                )

            if args.skip_calibration:
                calibration = {
                    "temperature": float("nan"),
                    "before_scaling": {"ece": float("nan"), "brier": float("nan")},
                    "after_scaling": {"ece": float("nan"), "brier": float("nan")},
                }
            else:
                calibration = calibration_report(
                    LogitsOnly(model),
                    val_loader=val_loader,
                    test_loader=test_loader,
                    class_names=class_names,
                    device=device,
                    output_dir=version_dir / "calibration",
                    model_name=f"{args.backbone}_{version}",
                    make_plots=not args.quick,
                )

            efficiency = None if args.skip_efficiency else run_efficiency(model, version, ckpt_path)

            master_rows.append(build_master_row(
                backbone=args.backbone,
                version=version,
                seed=seed,
                eval_results=eval_results[version],
                per_image_df=per_image,
                cf_summary=cf_summary,
                calibration=calibration,
                efficiency=efficiency,
                external=None,
            ))

            if version == "guided":
                val_metrics = run_epoch(
                    model, val_loader, criterion, optimizer=None, device=device, train=False,
                    lambda_att=arm_cfg["module"]["lambda_att"],
                    lambda_bg=arm_cfg["module"].get("lambda_bg", 0.0),
                    desc=f"{version} validation selection metrics",
                )
                validation_rows.append(validation_selection_row(
                    backbone=args.backbone,
                    guided_mode=guided_hparams["guided_mode"],
                    lambda_att=guided_hparams["lambda_att"],
                    lambda_bg=guided_hparams["lambda_bg"],
                    val_metrics=val_metrics,
                    cf_summary=None,
                ))

            model = model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        master = write_master_comparison(master_rows, output_dir / "master_comparison.csv")
        global_master = upsert_master_comparison(
            master_rows, args.repo_root / "artifacts" / "trustworthiness" / "master_comparison.csv"
        )
        summarize_multiseed_master(
            args.repo_root / "artifacts" / "trustworthiness" / "master_comparison.csv",
            args.repo_root / "artifacts" / "trustworthiness" / "multiseed_summary.csv",
        )
        write_validation_selection_table(validation_rows, output_dir / "validation_selection.csv")
        if (output_dir / "baseline" / "per_image_predictions.csv").exists() and (
            output_dir / "guided" / "per_image_predictions.csv"
        ).exists():
            build_statistics_report(
                output_dir / "baseline" / "per_image_predictions.csv",
                output_dir / "guided" / "per_image_predictions.csv",
                output_dir / "statistics.json",
                baseline_counterfactual_csv=output_dir / "baseline" / "counterfactual_per_image.csv",
                guided_counterfactual_csv=output_dir / "guided" / "counterfactual_per_image.csv",
            )

        print("\nTrustworthiness comparison:")
        print(master.to_string(index=False))

        zip_base = output_dir.parent / f"{args.backbone}_seed{seed}_trustworthiness_pair_results"
        zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=output_dir)
        print(f"\nZipped results: {zip_path}")


if __name__ == "__main__":
    main()
