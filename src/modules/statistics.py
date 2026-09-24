"""Paired statistical helpers for final baseline-vs-guided comparisons."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def _try_scipy():
    try:
        from scipy.stats import binomtest, wilcoxon
        return binomtest, wilcoxon
    except Exception:
        return None, None


def mcnemar_exact_from_correctness(baseline_correct, guided_correct) -> Dict[str, Any]:
    binomtest, _wilcoxon = _try_scipy()
    b_wrong_g_right = int(((~baseline_correct) & guided_correct).sum())
    b_right_g_wrong = int((baseline_correct & (~guided_correct)).sum())
    n_discordant = b_wrong_g_right + b_right_g_wrong
    p_value = None
    if binomtest is not None and n_discordant > 0:
        p_value = float(binomtest(min(b_wrong_g_right, b_right_g_wrong), n_discordant, 0.5).pvalue)
    return {
        "b_wrong_g_right": b_wrong_g_right,
        "b_right_g_wrong": b_right_g_wrong,
        "n_discordant": n_discordant,
        "p_value": p_value,
        "test": "mcnemar_exact_binomial",
    }


def paired_wilcoxon(x, y, metric: str) -> Dict[str, Any]:
    _binomtest, wilcoxon = _try_scipy()
    result = {"metric": metric, "test": "wilcoxon_signed_rank", "n": int(len(x)), "statistic": None, "p_value": None}
    if wilcoxon is not None and len(x) > 0:
        try:
            stat = wilcoxon(x, y)
            result["statistic"] = float(stat.statistic)
            result["p_value"] = float(stat.pvalue)
        except ValueError as exc:
            result["error"] = str(exc)
    return result


def _add_counterfactual_statistics(report: Dict[str, Any], baseline_cf_csv, guided_cf_csv) -> None:
    if baseline_cf_csv is None or guided_cf_csv is None:
        return
    if not Path(baseline_cf_csv).exists() or not Path(guided_cf_csv).exists():
        return
    baseline = pd.read_csv(baseline_cf_csv)
    guided = pd.read_csv(guided_cf_csv)
    merged = baseline.merge(guided, on=["image_path", "mode"], suffixes=("_baseline", "_guided"))
    for mode in sorted(merged["mode"].dropna().unique()):
        mode_df = merged[merged["mode"] == mode]
        if {"stability_baseline", "stability_guided"}.issubset(mode_df.columns):
            clean = mode_df[["stability_baseline", "stability_guided"]].dropna()
            report[f"cf_{mode}_stability_wilcoxon"] = paired_wilcoxon(
                clean["stability_baseline"], clean["stability_guided"], f"cf_{mode}_stability"
            )
        if {"prediction_flipped_baseline", "prediction_flipped_guided"}.issubset(mode_df.columns):
            b_flip = mode_df["prediction_flipped_baseline"].astype(bool)
            g_flip = mode_df["prediction_flipped_guided"].astype(bool)
            report[f"cf_{mode}_flip_paired_counts"] = {
                "baseline_flip_guided_not": int((b_flip & ~g_flip).sum()),
                "guided_flip_baseline_not": int((~b_flip & g_flip).sum()),
                "both_flip": int((b_flip & g_flip).sum()),
                "neither_flip": int((~b_flip & ~g_flip).sum()),
            }


def build_statistics_report(
    baseline_per_image_csv,
    guided_per_image_csv,
    output_json,
    baseline_counterfactual_csv=None,
    guided_counterfactual_csv=None,
) -> Dict[str, Any]:
    baseline = pd.read_csv(baseline_per_image_csv)
    guided = pd.read_csv(guided_per_image_csv)
    merged = baseline.merge(guided, on="image_path", suffixes=("_baseline", "_guided"))

    report: Dict[str, Any] = {}
    if {"true_label_baseline", "pred_label_baseline", "pred_label_guided"}.issubset(merged.columns):
        b_correct = merged["true_label_baseline"] == merged["pred_label_baseline"]
        g_correct = merged["true_label_baseline"] == merged["pred_label_guided"]
        report["classification_mcnemar"] = mcnemar_exact_from_correctness(b_correct, g_correct)
    if {"eil_post_baseline", "eil_post_guided"}.issubset(merged.columns):
        eil = merged[["eil_post_baseline", "eil_post_guided"]].dropna()
        report["eil_post_wilcoxon"] = paired_wilcoxon(eil["eil_post_baseline"], eil["eil_post_guided"], "eil_post")
    if {"eil_pre_baseline", "eil_pre_guided"}.issubset(merged.columns):
        eil = merged[["eil_pre_baseline", "eil_pre_guided"]].dropna()
        report["eil_pre_wilcoxon"] = paired_wilcoxon(eil["eil_pre_baseline"], eil["eil_pre_guided"], "eil_pre")

    _add_counterfactual_statistics(report, baseline_counterfactual_csv, guided_counterfactual_csv)

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
