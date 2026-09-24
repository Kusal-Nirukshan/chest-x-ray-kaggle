"""Tests for src/modules/counterfactual.py -- background-perturbation
robustness tests.

Uses tiny, deterministic, parameter-free toy models (never the real
DenseNet121 backbone) so these tests are fast, need no dataset, and their
expected numbers can be worked out by hand rather than just "runs without
crashing" -- the whole point of this module is that it must actually
DETECT background reliance when present and NOT flag it when absent, so
both directions get an explicit test.
"""
from __future__ import annotations

import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.modules import counterfactual_stability, evaluate_counterfactual_robustness, perturb_background


class _RegionMeanModel(nn.Module):
    """Deterministic, parameter-free 2-class model: logits = [0, scale *
    mean(x[:, :, rows, cols])]. Used to build models whose prediction
    depends ONLY on a chosen spatial region, so counterfactual_stability's
    ability to detect (or correctly not detect) reliance on that region
    can be checked directly, without any real network or training."""

    def __init__(self, row_slice: slice, col_slice: slice, scale: float = 10.0):
        super().__init__()
        self.row_slice = row_slice
        self.col_slice = col_slice
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        region = x[:, :, self.row_slice, self.col_slice]
        signal = region.mean(dim=(1, 2, 3)) * self.scale
        return torch.stack([torch.zeros_like(signal), signal], dim=1)


class _TupleWrapModel(nn.Module):
    """Mimics this repo's DenseNetLungAttention forward contract: returns
    (logits, attention, attention_logits) instead of a plain tensor."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor):
        return self.inner(x), None, None


def _synthetic_batch(batch: int = 2, size: int = 224, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(batch, 3, size, size, generator=g)
    masks = torch.zeros(batch, 1, size, size)
    masks[:, :, :, : size // 2] = 1.0  # left half is lung
    return images, masks


# ---------------------------------------------------------------------------
# perturb_background
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["zero", "noise", "shuffle"])
def test_perturb_background_leaves_lung_pixels_unchanged(mode):
    images, masks = _synthetic_batch()
    out = perturb_background(images, masks, mode=mode)
    lung = masks.bool().expand_as(images)
    torch.testing.assert_close(out[lung], images[lung])


def test_perturb_background_zero_sets_background_to_zero():
    images, masks = _synthetic_batch()
    out = perturb_background(images, masks, mode="zero")
    bg = (~masks.bool()).expand_as(images)
    assert torch.all(out[bg] == 0.0)


def test_perturb_background_noise_changes_background_only():
    images, masks = _synthetic_batch()
    torch.manual_seed(0)
    out = perturb_background(images, masks, mode="noise", noise_std=1.0)
    lung = masks.bool().expand_as(images)
    bg = ~lung
    torch.testing.assert_close(out[lung], images[lung])
    assert not torch.allclose(out[bg], images[bg])


def test_perturb_background_shuffle_preserves_value_multiset():
    images, masks = _synthetic_batch(batch=1)
    out = perturb_background(images, masks, mode="shuffle")
    bg = (~masks.bool()).expand_as(images)
    orig_sorted, _ = torch.sort(images[bg])
    out_sorted, _ = torch.sort(out[bg])
    torch.testing.assert_close(out_sorted, orig_sorted)


def test_perturb_background_shuffle_is_noop_on_all_lung_mask():
    images = torch.randn(1, 3, 64, 64)
    masks = torch.ones(1, 1, 64, 64)
    out = perturb_background(images, masks, mode="shuffle")
    torch.testing.assert_close(out, images)


def test_perturb_background_raises_on_unknown_mode():
    images, masks = _synthetic_batch()
    with pytest.raises(ValueError):
        perturb_background(images, masks, mode="bogus")


# ---------------------------------------------------------------------------
# counterfactual_stability
# ---------------------------------------------------------------------------

def test_counterfactual_stability_perfect_for_background_invariant_model():
    """A model that only reads the lung region must be COMPLETELY
    unaffected by any background perturbation -- stability exactly 1.0."""
    images, masks = _synthetic_batch(batch=4)
    model = _RegionMeanModel(row_slice=slice(0, 224), col_slice=slice(0, 112))
    for mode in ("zero", "noise", "shuffle"):
        result = counterfactual_stability(model, images, masks, mode=mode)
        torch.testing.assert_close(result["stability"], torch.ones(4), atol=1e-5, rtol=0)
        assert not bool(result["pred_changed"].bool().any())


def test_counterfactual_stability_detects_background_dependent_model():
    """A model that reads ONLY the background must show a large stability
    drop and a flipped prediction once the background is zeroed out --
    this is the core claim the whole module exists to test."""
    images = torch.zeros(2, 3, 224, 224)
    images[:, :, :, 112:] = 5.0  # background (right half) carries a strong constant signal
    masks = torch.zeros(2, 1, 224, 224)
    masks[:, :, :, :112] = 1.0

    model = _RegionMeanModel(row_slice=slice(0, 224), col_slice=slice(112, 224), scale=10.0)
    result = counterfactual_stability(model, images, masks, mode="zero")
    assert result["stability"].mean().item() < 0.6
    assert bool(result["pred_changed"].all())


def test_counterfactual_stability_ignores_background_for_lung_only_model():
    """No false positives: a lung-only model must show ZERO reaction to any
    background perturbation, even a strong one."""
    images = torch.zeros(2, 3, 224, 224)
    images[:, :, :, :112] = 3.0  # lung region carries a strong constant signal
    masks = torch.zeros(2, 1, 224, 224)
    masks[:, :, :, :112] = 1.0

    model = _RegionMeanModel(row_slice=slice(0, 224), col_slice=slice(0, 112), scale=10.0)
    for mode in ("zero", "noise", "shuffle"):
        result = counterfactual_stability(model, images, masks, mode=mode)
        torch.testing.assert_close(result["stability"], torch.ones(2), atol=1e-4, rtol=0)
        assert not bool(result["pred_changed"].bool().any())


def test_counterfactual_stability_handles_tuple_model_output():
    """Must work directly on this repo's (logits, attention, attention_logits)
    model contract, without requiring the caller to wrap it in LogitsOnly."""
    images, masks = _synthetic_batch(batch=2)
    inner = _RegionMeanModel(row_slice=slice(0, 224), col_slice=slice(0, 112))
    wrapped = _TupleWrapModel(inner)

    result_tuple = counterfactual_stability(wrapped, images, masks, mode="zero")
    result_plain = counterfactual_stability(inner, images, masks, mode="zero")
    torch.testing.assert_close(result_tuple["stability"], result_plain["stability"])
    torch.testing.assert_close(result_tuple["pred_changed"], result_plain["pred_changed"])


# ---------------------------------------------------------------------------
# evaluate_counterfactual_robustness
# ---------------------------------------------------------------------------

def test_evaluate_counterfactual_robustness_multiple_modes_one_row_each():
    images, masks = _synthetic_batch(batch=2)
    loader = [(images, torch.zeros(2, dtype=torch.long), masks)]
    model = _RegionMeanModel(row_slice=slice(0, 224), col_slice=slice(0, 112))

    df = evaluate_counterfactual_robustness(
        model, loader, device=torch.device("cpu"), modes=("zero", "noise", "shuffle"),
    )
    assert list(df["mode"]) == ["zero", "noise", "shuffle"]
    assert len(df) == 3
    assert (df["mean_stability"] > 0.999).all()  # lung-only model, as above


def test_evaluate_counterfactual_robustness_writes_and_appends_csv(tmp_path):
    images = torch.zeros(4, 3, 224, 224)
    images[:, :, :, 112:] = 5.0
    masks = torch.zeros(4, 1, 224, 224)
    masks[:, :, :, :112] = 1.0
    loader = [(images, torch.zeros(4, dtype=torch.long), masks)]
    model = _RegionMeanModel(row_slice=slice(0, 224), col_slice=slice(112, 224), scale=10.0)

    csv_path = tmp_path / "counterfactual_robustness.csv"
    df1 = evaluate_counterfactual_robustness(
        model, loader, device=torch.device("cpu"), modes=("zero",), arm_name="A2", output_csv=csv_path,
    )
    assert set(df1.columns) >= {"arm", "mode", "n_images", "mean_stability", "pred_flip_rate"}
    assert df1.iloc[0]["n_images"] == 4
    assert df1.iloc[0]["pred_flip_rate"] == 1.0

    evaluate_counterfactual_robustness(
        model, loader, device=torch.device("cpu"), modes=("zero",), arm_name="A0", output_csv=csv_path,
    )
    on_disk = pd.read_csv(csv_path)
    assert len(on_disk) == 2  # appended, not overwritten
    assert set(on_disk["arm"]) == {"A2", "A0"}


def test_evaluate_counterfactual_robustness_writes_per_image_csv(tmp_path):
    images = torch.zeros(3, 3, 224, 224)
    images[:, :, :, 112:] = 5.0
    masks = torch.zeros(3, 1, 224, 224)
    masks[:, :, :, :112] = 1.0
    labels = torch.tensor([0, 1, 1])
    loader = [(images, labels, masks)]
    model = _RegionMeanModel(row_slice=slice(0, 224), col_slice=slice(112, 224), scale=10.0)

    per_image_csv = tmp_path / "counterfactual_per_image.csv"
    evaluate_counterfactual_robustness(
        model,
        loader,
        device=torch.device("cpu"),
        modes=("zero", "noise"),
        arm_name="guided",
        per_image_csv=per_image_csv,
    )

    rows = pd.read_csv(per_image_csv)
    assert len(rows) == 6
    assert set(rows["mode"]) == {"zero", "noise"}
    assert set(rows.columns) >= {
        "image_path",
        "true_label",
        "original_prediction",
        "perturbed_prediction",
        "stability",
        "prediction_flipped",
        "original_confidence",
        "perturbed_confidence",
        "confidence_change",
    }
