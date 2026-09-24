"""Trustworthiness comparison helpers.

This module keeps the final research framing separate from the older T18
A0-A5 ablation helpers. A0-A5 remains a method-development study; the
functions here support the cross-backbone baseline-vs-lung-guided comparison.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


CNN_BACKBONES = {"densenet121", "resnet50", "efficientnet_b0"}

BACKBONE_CONFIGS = {
    "densenet121": "configs/densenet121_lung_attention.yaml",
    "resnet50": "configs/resnet50_lung_attention.yaml",
    "efficientnet_b0": "configs/efficientnet_b0_lung_attention.yaml",
}


def default_config_path(repo_root: Path, backbone: str) -> Path:
    if backbone not in BACKBONE_CONFIGS:
        raise ValueError(f"No default config registered for backbone '{backbone}'")
    return repo_root / BACKBONE_CONFIGS[backbone]


def validate_backbone_config(config: Dict[str, Any], backbone: str, config_path: Path | str = "<config>") -> None:
    cfg_backbone = config.get("model", {}).get("name")
    if cfg_backbone != backbone:
        raise ValueError(
            f"Config/backbone mismatch: {config_path} has model.name={cfg_backbone!r}, "
            f"but --backbone is {backbone!r}."
        )


def seed_output_dir(base_output_dir: Path, backbone: str, seed: int) -> Path:
    return base_output_dir / backbone / f"seed{seed}"


def load_guided_config(path: Path | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    required = {"guided_mode", "lambda_att", "lambda_bg"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"guided config missing keys: {sorted(missing)}")
    return data


def resolve_guided_hparams(
    guided_config: Optional[Dict[str, Any]],
    guided_mode: Optional[str],
    lambda_att: Optional[float],
    lambda_bg: Optional[float],
    defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """CLI override > selected config > defaults."""
    defaults = defaults or {"guided_mode": "multiply", "lambda_att": 1.0, "lambda_bg": 0.0}
    source = {**defaults, **(guided_config or {})}
    if guided_mode is not None:
        source["guided_mode"] = guided_mode
    if lambda_att is not None:
        source["lambda_att"] = lambda_att
    if lambda_bg is not None:
        source["lambda_bg"] = lambda_bg
    return {
        "guided_mode": source["guided_mode"],
        "lambda_att": float(source["lambda_att"]),
        "lambda_bg": float(source["lambda_bg"]),
    }


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
            row[f"val_cf_{mode}_confidence_change"] = cf.get("mean_abs_confidence_change")
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


def upsert_master_comparison(rows: Iterable[Dict[str, Any]], output_csv) -> pd.DataFrame:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(list(rows))
    key = ["backbone", "version", "seed"]
    if output_csv.exists():
        old_df = pd.read_csv(output_csv)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.drop_duplicates(subset=key, keep="last")
    combined = combined.sort_values(key).reset_index(drop=True)
    combined.to_csv(output_csv, index=False)
    return combined


def summarize_multiseed_master(master_csv, output_csv) -> pd.DataFrame:
    df = pd.read_csv(master_csv)
    metrics = [
        "test_accuracy", "test_macro_f1", "test_macro_auc", "eil_post",
        "cf_zero_stability", "cf_zero_flip_rate",
        "cf_shuffle_stability", "cf_shuffle_flip_rate",
        "cf_noise_stability", "cf_noise_flip_rate",
        "ece_before", "ece_after", "brier_before", "brier_after",
        "params", "gflops", "cpu_latency_ms", "gpu_latency_ms",
        "external_accuracy", "external_auc", "ood_drop",
    ]
    rows = []
    for (backbone, version), group in df.groupby(["backbone", "version"], dropna=False):
        row = {"backbone": backbone, "version": version, "n_seeds": int(group["seed"].nunique())}
        for metric in metrics:
            if metric in group:
                row[f"{metric}_mean"] = group[metric].mean(skipna=True)
                row[f"{metric}_std"] = group[metric].std(skipna=True)
        rows.append(row)
    out = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def select_guided_candidate(selection_df: pd.DataFrame, output_json=None) -> Dict[str, Any]:
    """Conservative validation-only selection.

    Returns no clear winner unless at least one candidate has non-degraded
    macro-F1 versus the validation baseline and improves zero-background
    stability over that baseline. This avoids selecting the least-bad guided
    candidate when every guided candidate is worse than vanilla.
    """
    if selection_df.empty:
        result = {"selected": False, "reason": "no candidates", "selection_rule_version": "v1"}
    else:
        df = selection_df.copy()
        if "version" not in df:
            df["version"] = "guided"
        baseline = df[df["version"] == "baseline"]
        guided = df[df["version"] != "baseline"]
        if baseline.empty:
            result = {"selected": False, "reason": "missing validation baseline", "selection_rule_version": "v2"}
            if output_json is not None:
                output_json = Path(output_json)
                output_json.parent.mkdir(parents=True, exist_ok=True)
                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
            return result

        baseline_row = baseline.iloc[0]
        f1_floor = baseline_row["val_macro_f1"] - 0.01
        stability_col = "val_cf_zero_stability"
        if stability_col in guided and pd.notna(baseline_row.get(stability_col)):
            candidates = guided[(guided["val_macro_f1"] >= f1_floor) & (guided[stability_col] > baseline_row[stability_col])]
        else:
            candidates = guided[guided["val_macro_f1"] >= f1_floor]
        if candidates.empty:
            result = {"selected": False, "reason": "no clear winner versus validation baseline", "selection_rule_version": "v2"}
        else:
            sort_cols = [c for c in [stability_col, "val_eil_post", "val_attention_dice", "val_macro_f1"] if c in candidates]
            best = candidates.sort_values(sort_cols, ascending=False).iloc[0].to_dict()
            result = {"selected": True, "selection_rule_version": "v2", **best}
    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    return result
