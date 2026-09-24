from __future__ import annotations

import pandas as pd

from src.modules.statistics import build_statistics_report, mcnemar_exact_from_correctness


def test_mcnemar_counts_discordant_pairs():
    baseline = pd.Series([True, True, False, False])
    guided = pd.Series([True, False, True, False])
    result = mcnemar_exact_from_correctness(baseline, guided)
    assert result["b_wrong_g_right"] == 1
    assert result["b_right_g_wrong"] == 1
    assert result["n_discordant"] == 2


def test_build_statistics_report_from_per_image_predictions(tmp_path):
    baseline = pd.DataFrame({
        "image_path": ["a.png", "b.png"],
        "true_label": ["Normal", "COVID"],
        "pred_label": ["Normal", "Normal"],
        "eil_post": [0.5, 0.6],
    })
    guided = pd.DataFrame({
        "image_path": ["a.png", "b.png"],
        "true_label": ["Normal", "COVID"],
        "pred_label": ["Normal", "COVID"],
        "eil_post": [0.7, 0.8],
    })
    b_path = tmp_path / "baseline.csv"
    g_path = tmp_path / "guided.csv"
    baseline.to_csv(b_path, index=False)
    guided.to_csv(g_path, index=False)
    report = build_statistics_report(b_path, g_path, tmp_path / "statistics.json")
    assert "classification_mcnemar" in report
    assert "eil_post_wilcoxon" in report
    assert (tmp_path / "statistics.json").exists()
