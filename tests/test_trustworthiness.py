from __future__ import annotations

import pandas as pd
import pytest

from src.modules.trustworthiness import (
    build_master_row,
    default_config_path,
    resolve_guided_hparams,
    make_pair_configs,
    seed_output_dir,
    select_guided_candidate,
    summarize_multiseed_master,
    upsert_master_comparison,
    validate_backbone_config,
    validation_selection_row,
    write_master_comparison,
)


def _base_cfg():
    return {
        "experiment": {"seed": 42},
        "model": {"name": "densenet121", "num_classes": 4, "pretrained": False, "drop_rate": 0.0},
        "module": {
            "attention": "lung",
            "use_attention": True,
            "reduction": 8,
            "gate_mode": "residual",
            "lambda_att": 1.0,
            "lambda_bg": 0.0,
        },
    }


def test_make_pair_configs_creates_controlled_baseline_and_guided_configs():
    pair = make_pair_configs(
        _base_cfg(),
        backbone="efficientnet_b0",
        guided_mode="multiply",
        lambda_att=0.3,
        lambda_bg=0.1,
        seed=123,
    )

    baseline = pair["baseline"]
    guided = pair["guided"]
    assert baseline["model"]["name"] == guided["model"]["name"] == "efficientnet_b0"
    assert baseline["experiment"]["seed"] == guided["experiment"]["seed"] == 123
    assert baseline["module"]["use_attention"] is False
    assert baseline["module"]["lambda_att"] == 0.0
    assert baseline["module"]["lambda_bg"] == 0.0
    assert guided["module"]["use_attention"] is True
    assert guided["module"]["gate_mode"] == "multiply"
    assert guided["module"]["lambda_att"] == 0.3
    assert guided["module"]["lambda_bg"] == 0.1
    assert guided["module"]["reduction"] == 16


def test_make_pair_configs_rejects_vit_cnn_gate_compatibility():
    with pytest.raises(ValueError, match="patch-token"):
        make_pair_configs(_base_cfg(), "vit_base_patch16_224", "multiply", 1.0, 0.0, 42)


def test_validation_selection_row_contains_no_test_metrics():
    row = validation_selection_row(
        backbone="densenet121",
        guided_mode="residual",
        lambda_att=1.0,
        lambda_bg=0.3,
        val_metrics={"accuracy": 0.8, "f1": 0.7, "auroc": 0.9, "ilar": 0.6},
    )
    assert row["val_macro_f1"] == 0.7
    assert not any(key.startswith("test_") for key in row)


def test_validation_selection_row_includes_counterfactual_confidence_change():
    cf = pd.DataFrame([{
        "mode": "zero",
        "mean_stability": 0.9,
        "pred_flip_rate": 0.1,
        "mean_abs_confidence_change": 0.02,
    }])
    row = validation_selection_row("densenet121", "residual", 1.0, 0.0, {"f1": 0.7}, cf)
    assert row["val_cf_zero_stability"] == 0.9
    assert row["val_cf_zero_confidence_change"] == 0.02


def test_build_master_row_writes_expected_trustworthiness_columns(tmp_path):
    eval_results = {
        "classification_report": {
            "accuracy": 0.8,
            "macro avg": {"f1-score": 0.75},
        },
        "roc_auc_macro": 0.9,
        "attention_ilar": None,
        "attention_dice": None,
        "attention_iou": None,
    }
    per_image = pd.DataFrame({"eil_post": [0.6, 0.8], "eil_pre": [0.5, 0.7]})
    cf = pd.DataFrame([
        {"mode": "zero", "mean_stability": 0.9, "pred_flip_rate": 0.1, "mean_abs_confidence_change": 0.03},
        {"mode": "shuffle", "mean_stability": 0.8, "pred_flip_rate": 0.2, "mean_abs_confidence_change": 0.04},
    ])
    calibration = {
        "temperature": 1.2,
        "before_scaling": {"ece": 0.1, "brier": 0.2},
        "after_scaling": {"ece": 0.05, "brier": 0.18},
    }

    row = build_master_row("densenet121", "baseline", 42, eval_results, per_image, cf, calibration)
    assert row["eil_post"] == pytest.approx(0.7)
    assert row["cf_zero_stability"] == 0.9
    assert row["cf_noise_flip_rate"] != row["cf_noise_flip_rate"]  # NaN until implemented/run
    assert row["ece_after"] == 0.05

    out = write_master_comparison([row], tmp_path / "master.csv")
    assert out.iloc[0]["backbone"] == "densenet121"
    assert (tmp_path / "master.csv").exists()


def test_default_config_path_and_backbone_mismatch_detection(tmp_path):
    assert str(default_config_path(tmp_path, "efficientnet_b0")).endswith("configs/efficientnet_b0_lung_attention.yaml")
    validate_backbone_config({"model": {"name": "resnet50"}}, "resnet50", "cfg.yaml")
    with pytest.raises(ValueError, match="mismatch"):
        validate_backbone_config({"model": {"name": "efficientnet_b0"}}, "densenet121", "cfg.yaml")


def test_seed_output_dir_is_seed_safe(tmp_path):
    assert seed_output_dir(tmp_path, "efficientnet_b0", 123) == tmp_path / "efficientnet_b0" / "seed123"


def test_master_upsert_replaces_existing_key_without_duplicates(tmp_path):
    path = tmp_path / "master.csv"
    upsert_master_comparison([
        {"backbone": "densenet121", "version": "baseline", "seed": 42, "test_accuracy": 0.8},
        {"backbone": "densenet121", "version": "guided", "seed": 42, "test_accuracy": 0.81},
    ], path)
    out = upsert_master_comparison([
        {"backbone": "densenet121", "version": "baseline", "seed": 42, "test_accuracy": 0.82},
    ], path)
    assert len(out) == 2
    row = out[(out["backbone"] == "densenet121") & (out["version"] == "baseline") & (out["seed"] == 42)].iloc[0]
    assert row["test_accuracy"] == 0.82


def test_multiseed_summary_groups_mean_std(tmp_path):
    master = tmp_path / "master.csv"
    pd.DataFrame([
        {"backbone": "densenet121", "version": "baseline", "seed": 42, "test_accuracy": 0.8},
        {"backbone": "densenet121", "version": "baseline", "seed": 123, "test_accuracy": 0.9},
    ]).to_csv(master, index=False)
    summary = summarize_multiseed_master(master, tmp_path / "summary.csv")
    assert summary.iloc[0]["test_accuracy_mean"] == pytest.approx(0.85)
    assert summary.iloc[0]["n_seeds"] == 2


def test_guided_config_precedence_cli_overrides_config():
    resolved = resolve_guided_hparams(
        {"guided_mode": "residual", "lambda_att": 0.3, "lambda_bg": 0.1},
        guided_mode="multiply",
        lambda_att=None,
        lambda_bg=0.5,
    )
    assert resolved == {"guided_mode": "multiply", "lambda_att": 0.3, "lambda_bg": 0.5}


def test_select_guided_candidate_requires_baseline_reference(tmp_path):
    df = pd.DataFrame([
        {"version": "guided", "guided_mode": "residual", "lambda_att": 0.3, "lambda_bg": 0.0, "val_macro_f1": 0.8, "val_cf_zero_stability": 0.7},
    ])
    result = select_guided_candidate(df, tmp_path / "selected.json")
    assert result["selected"] is False
    assert result["reason"] == "missing validation baseline"


def test_select_guided_candidate_uses_baseline_not_best_guided(tmp_path):
    df = pd.DataFrame([
        {"version": "baseline", "candidate": "baseline", "val_macro_f1": 0.92, "val_cf_zero_stability": 0.80},
        {"version": "guided", "guided_mode": "residual", "lambda_att": 0.3, "lambda_bg": 0.0, "val_macro_f1": 0.86, "val_cf_zero_stability": 0.90},
        {"version": "guided", "guided_mode": "multiply", "lambda_att": 1.0, "lambda_bg": 0.1, "val_macro_f1": 0.87, "val_cf_zero_stability": 0.91},
    ])
    result = select_guided_candidate(df, tmp_path / "selected.json")
    assert result["selected"] is False


def test_select_guided_candidate_outputs_no_test_metrics(tmp_path):
    df = pd.DataFrame([
        {"version": "baseline", "candidate": "baseline", "val_macro_f1": 0.80, "val_cf_zero_stability": 0.70},
        {"version": "guided", "guided_mode": "multiply", "lambda_att": 1.0, "lambda_bg": 0.1, "val_macro_f1": 0.805, "val_cf_zero_stability": 0.80, "val_eil_post": 0.75},
    ])
    result = select_guided_candidate(df, tmp_path / "selected.json")
    assert result["selected"] is True
    assert not any(k.startswith("test_") for k in result)
    assert (tmp_path / "selected.json").exists()
