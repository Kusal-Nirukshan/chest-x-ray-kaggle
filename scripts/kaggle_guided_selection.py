"""Validation-only guided-method selection for the trustworthiness study.

This script is deliberately separate from the final pair runner. It may train
many guided candidates, but it never evaluates the held-out test split and
never writes test metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-only guided configuration selection.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--backbone", choices=["densenet121", "resnet50", "efficientnet_b0"], required=True)
    parser.add_argument("--guided-modes", nargs="+", choices=["residual", "multiply", "none"], default=["residual", "multiply"])
    parser.add_argument("--lambda-att-values", nargs="+", type=float, default=[0.3, 0.5, 1.0])
    parser.add_argument("--lambda-bg-values", nargs="+", type=float, default=[0.0, 0.1, 0.3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-cam-subset-size", type=int, default=300)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def import_repo(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def validation_attention_metrics(model, loader, device: torch.device) -> Dict[str, float]:
    from src.modules import attention_dice, attention_iou, ilar

    model.eval()
    vals = {"ilar": [], "dice": [], "iou": []}
    with torch.no_grad():
        for images, _labels, masks in loader:
            images = images.to(device)
            masks = masks.to(device).float()
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            _logits, att, _att_logits = model(images)
            if att is None:
                continue
            vals["ilar"].append(ilar(att.float(), masks).cpu())
            vals["dice"].append(attention_dice(att.float(), masks).cpu())
            vals["iou"].append(attention_iou(att.float(), masks).cpu())
    out = {}
    for key, tensors in vals.items():
        out[key] = float(torch.cat(tensors).mean()) if tensors else float("nan")
    return out


def main() -> None:
    args = parse_args()
    import_repo(args.repo_root)

    from src.datasets import build_dataloaders, compute_class_weights
    from src.modules import (
        build_model,
        build_optimizer,
        build_scheduler,
        build_per_image_predictions,
        default_config_path,
        evaluate_counterfactual_robustness,
        freeze_backbone,
        make_pair_configs,
        run_epoch,
        seed_output_dir,
        select_guided_candidate,
        stratified_cam_subset,
        train_phase,
        unfreeze_final_blocks,
        validate_backbone_config,
        validation_selection_row,
        write_validation_selection_table,
    )
    from src.utils import load_config, set_seed

    if args.quick:
        print("SMOKE TEST ONLY - NOT FINAL RESULTS")

    config_path = args.config or default_config_path(args.repo_root, args.backbone)
    cfg = load_config(config_path)
    validate_backbone_config(cfg, args.backbone, config_path)
    if args.quick:
        cfg["training"]["phase1_epochs"] = 1
        cfg["training"]["phase2_epochs"] = 1
        cfg["training"]["patience"] = 1
        args.val_cam_subset_size = min(args.val_cam_subset_size, 16)

    base_output = args.output_dir or args.repo_root / "artifacts" / "trustworthiness"
    output_dir = seed_output_dir(base_output, args.backbone, args.seed) / "selection"
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _test_loader, class_names, train_targets, datasets = build_dataloaders(
        data_dir=args.data_dir,
        img_size=cfg["dataset"]["image_size"],
        batch_size=cfg["training"]["batch_size"],
        seed=args.seed,
        num_workers=args.num_workers,
        split_manifest_path=args.repo_root / "artifacts" / "splits" / "split_manifest_v1.csv",
    )
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train_targets, len(class_names)).to(device))
    val_cam_subset = stratified_cam_subset(datasets["val"], n=args.val_cam_subset_size, seed=args.seed)
    with (output_dir / "validation_cam_subset_indices.json").open("w", encoding="utf-8") as f:
        json.dump([int(x) for x in val_cam_subset], f, indent=2)

    rows = []
    candidate_specs = [("baseline", "none", 0.0, 0.0)]
    candidate_specs.extend(
        (f"{guided_mode}_att{lambda_att}_bg{lambda_bg}".replace(".", "p"), guided_mode, lambda_att, lambda_bg)
        for guided_mode in args.guided_modes
        for lambda_att in args.lambda_att_values
        for lambda_bg in args.lambda_bg_values
    )

    for candidate_name, guided_mode, lambda_att, lambda_bg in candidate_specs:
        candidate_dir = output_dir / "candidates" / candidate_name
        pair_cfgs = make_pair_configs(cfg, args.backbone, guided_mode, lambda_att, lambda_bg, args.seed)
        arm_cfg = pair_cfgs["baseline"] if candidate_name == "baseline" else pair_cfgs["guided"]
        m = arm_cfg["module"]
        t = arm_cfg["training"]

        print("\n" + "=" * 80)
        print(f"Validation candidate: {candidate_name}")
        print("=" * 80)

        set_seed(args.seed)
        model = build_model(
            num_classes=arm_cfg["model"]["num_classes"],
            use_attention=m["use_attention"],
            attention=m.get("attention", "lung"),
            gate_mode=m.get("gate_mode", "residual"),
            reduction=m.get("reduction", 16),
            backbone_name=args.backbone,
            pretrained=arm_cfg["model"]["pretrained"],
            drop_rate=arm_cfg["model"].get("drop_rate", 0.0),
        ).to(device)

        freeze_backbone(model)
        opt1 = build_optimizer(model, t["optimizer"], t["phase1_lr"], t["weight_decay"])
        sched1 = build_scheduler(opt1, arm_cfg["scheduler"]["name"], t["phase1_epochs"])
        model = train_phase(
            model, train_loader, val_loader, criterion, opt1, sched1, device,
            epochs=t["phase1_epochs"], patience=t["patience"], phase_name="phase1_frozen",
            output_dir=candidate_dir, lambda_att=lambda_att, lambda_bg=lambda_bg,
            wandb_enabled=args.wandb,
        )

        unfreeze_final_blocks(model, t["unfreeze_blocks"])
        opt2 = build_optimizer(model, t["optimizer"], t["phase2_lr"], t["weight_decay"])
        sched2 = build_scheduler(opt2, arm_cfg["scheduler"]["name"], t["phase2_epochs"])
        model = train_phase(
            model, train_loader, val_loader, criterion, opt2, sched2, device,
            epochs=t["phase2_epochs"], patience=t["patience"], phase_name="phase2_finetune",
            output_dir=candidate_dir, lambda_att=lambda_att, lambda_bg=lambda_bg,
            wandb_enabled=args.wandb,
        )

        val_metrics = run_epoch(
            model, val_loader, criterion, optimizer=None, device=device, train=False,
            lambda_att=lambda_att, lambda_bg=lambda_bg, desc=f"{candidate_name} val",
        )
        att_metrics = validation_attention_metrics(model, val_loader, device)
        val_per_image = build_per_image_predictions(
            model=model,
            test_dataset=datasets["val"],
            class_names=class_names,
            device=device,
            cam_subset=val_cam_subset,
            batch_size=cfg["training"]["batch_size"],
        )
        val_per_image.to_csv(candidate_dir / "validation_per_image_predictions.csv", index=False)
        val_eil_post = val_per_image["eil_post"].dropna()
        val_eil_pre = val_per_image["eil_pre"].dropna()
        cf_summary = evaluate_counterfactual_robustness(
            model, val_loader, device=device, seed=args.seed, arm_name=candidate_name,
            output_csv=candidate_dir / "validation_counterfactual_summary.csv",
            per_image_csv=candidate_dir / "validation_counterfactual_per_image.csv",
        )
        row = validation_selection_row(
            args.backbone, guided_mode, lambda_att, lambda_bg, val_metrics, cf_summary
        )
        row.update({
            "candidate": candidate_name,
            "version": "baseline" if candidate_name == "baseline" else "guided",
            "val_attention_dice": att_metrics["dice"],
            "val_attention_iou": att_metrics["iou"],
            "val_ilar": att_metrics["ilar"],
            "val_eil_post": float(val_eil_post.mean()) if len(val_eil_post) else float("nan"),
            "val_eil_pre": float(val_eil_pre.mean()) if len(val_eil_pre) else float("nan"),
        })
        rows.append(row)
        model = model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    table = write_validation_selection_table(rows, output_dir / "validation_selection.csv")
    selected = select_guided_candidate(table, output_dir / "selected_guided_config.json")
    print("\nValidation selection table:")
    print(table.to_string(index=False))
    print("\nSelected config:")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
