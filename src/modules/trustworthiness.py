"""Trustworthiness comparison helpers.

This module keeps the final research framing separate from the older T18
A0-A5 ablation helpers. A0-A5 remains a method-development study; the
functions here support the cross-backbone baseline-vs-lung-guided comparison.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


CNN_BACKBONES = {"densenet121", "resnet50", "efficientnet_b0"}


def make_pair_configs(
    base_cfg: Dict[str, Any],
    backbone: str,
    guided_mode: str,
    lambda_att: float,
    lambda_bg: float,
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    """Return configs for the controlled baseline-vs-guided experiment."""
    if backbone not in CNN_BACKBONES:
        raise ValueError(
            f"'{backbone}' is not supported by the CNN lung-attention runner. "
            "ViT needs a patch-token lung-guidance adapter, not the Conv2D gate."
        )
    if guided_mode not in {"residual", "multiply", "none"}:
        raise ValueError(f"unknown guided_mode: {guided_mode}")

    baseline = copy.deepcopy(base_cfg)
    guided = copy.deepcopy(base_cfg)
    for cfg in (baseline, guided):
        cfg["experiment"]["seed"] = seed
        cfg["model"]["name"] = backbone
        cfg["module"]["attention"] = "lung"

    baseline["module"].update({
        "use_attention": False,
        "gate_mode": "none",
        "lambda_att": 0.0,
        "lambda_bg": 0.0,
    })
    guided["module"].update({
        "use_attention": True,
        "gate_mode": guided_mode,
        "lambda_att": float(lambda_att),
        "lambda_bg": float(lambda_bg),
    })
    if backbone == "efficientnet_b0":
        guided["module"]["reduction"] = 16
        baseline["module"]["reduction"] = 16

    return {"baseline": baseline, "guided": guided}


def validation_selection_row(
    backbone: str,
    guided_mode: str,
    lambda_att: float,
    lambda_bg: float,
    val_metrics: Dict[str, Any],
    cf_summary: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Record validation-only model-selection inputs.

    The test split is intentionally absent from this schema.
    """
    row = {
        "backbone": backbone,
        "guided_mode": guided_mode,
        "lambda_att": float(lambda_att),
        "lambda_bg": float(lambda_bg),
        "val_accuracy": val_metrics.get("accuracy"),
        "val_macro_f1": val_metrics.get("f1"),
        "val_auc": val_metrics.get("auroc"),
        "val_ilar": val_metrics.get("ilar"),
    }
    if cf_summary is not None and not cf_summary.empty:
        for _, cf in cf_summary.iterrows():
            mode = cf["mode"]
            row[f"val_cf_{mode}_stability"] = cf.get("mean_stability")
            row[f"val_cf_{mode}_flip_rate"] = cf.get("pred_flip_rate")
    return row


def write_validation_selection_table(rows: Iterable[Dict[str, Any]], output_csv) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def build_master_row(
    backbone: str,
    version: str,
    seed: int,
    eval_results: Dict[str, Any],
    per_image_df: pd.DataFrame,
    cf_summary: pd.DataFrame,
    calibration: Dict[str, Any],
    efficiency: Optional[Dict[str, Any]] = None,
    external: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    report = eval_results["classification_report"]
    eil_post = per_image_df["eil_post"].dropna() if "eil_post" in per_image_df else pd.Series(dtype=float)
    eil_pre = per_image_df["eil_pre"].dropna() if "eil_pre" in per_image_df else pd.Series(dtype=float)

    row: Dict[str, Any] = {
        "backbone": backbone,
        "version": version,
        "seed": seed,
        "test_accuracy": report["accuracy"],
        "test_macro_f1": report["macro avg"]["f1-score"],
        "test_macro_auc": eval_results.get("roc_auc_macro"),
        "eil": float(eil_post.mean()) if len(eil_post) else float("nan"),
        "eil_pre": float(eil_pre.mean()) if len(eil_pre) else float("nan"),
        "eil_post": float(eil_post.mean()) if len(eil_post) else float("nan"),
        "attention_ilar": eval_results.get("attention_ilar", float("nan")),
        "attention_dice": eval_results.get("attention_dice", float("nan")),
        "attention_iou": eval_results.get("attention_iou", float("nan")),
        "ece_before": calibration["before_scaling"]["ece"],
        "ece_after": calibration["after_scaling"]["ece"],
        "brier_before": calibration["before_scaling"]["brier"],
        "brier_after": calibration["after_scaling"]["brier"],
        "temperature": calibration["temperature"],
    }

    for mode in ("zero", "shuffle", "noise"):
        match = cf_summary[cf_summary["mode"] == mode]
        if match.empty:
            row[f"cf_{mode}_stability"] = float("nan")
            row[f"cf_{mode}_flip_rate"] = float("nan")
            row[f"cf_{mode}_confidence_change"] = float("nan")
        else:
            cf = match.iloc[0]
            row[f"cf_{mode}_stability"] = cf.get("mean_stability")
            row[f"cf_{mode}_flip_rate"] = cf.get("pred_flip_rate")
            row[f"cf_{mode}_confidence_change"] = cf.get("mean_abs_confidence_change")

    efficiency = efficiency or {}
    for key in ("params", "gflops", "cpu_latency_ms", "gpu_latency_ms"):
        row[key] = efficiency.get(key, float("nan"))

    external = external or {}
    row["external_accuracy"] = external.get("external_accuracy", float("nan"))
    row["external_auc"] = external.get("external_auc", float("nan"))
    row["ood_drop"] = external.get("ood_drop", float("nan"))
    return row


def write_master_comparison(rows: Iterable[Dict[str, Any]], output_csv) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df
