"""Tests for src/modules/gradcam.py -- the Grad-CAM faithfulness harness.

Requires `pip install grad-cam` (pytorch_grad_cam). All models built with
pretrained=False -- these test the harness's wiring, not learned weights,
and avoid a network weight download in CI.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.modules import build_model, energy_inside_lung
from src.modules.gradcam import cam_for, get_taps

RECT = (slice(64, 160), slice(48, 176))


def _rect_mask(batch: int = 2) -> torch.Tensor:
    m = torch.zeros(batch, 1, 224, 224)
    m[:, :, RECT[0], RECT[1]] = 1.0
    return m


class _DummyFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm5 = nn.Identity()


class _DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = _DummyFeatures()
        self.bn2 = nn.Identity()
        self.layer4 = nn.Sequential(nn.Identity(), nn.Identity())


class _DummyLungAttentionModel(nn.Module):
    def __init__(self, backbone_name: str):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = _DummyBackbone()
        self.post_attn = nn.Identity()


def test_get_taps_selects_architecture_specific_pre_gate_layer():
    dense = _DummyLungAttentionModel("densenet121")
    eff = _DummyLungAttentionModel("efficientnet_b0")
    res = _DummyLungAttentionModel("resnet50")

    assert get_taps(dense) == (dense.post_attn, dense.backbone.features.norm5)
    assert get_taps(eff) == (eff.post_attn, eff.backbone.bn2)
    assert get_taps(res) == (res.post_attn, res.backbone.layer4[-1])


def test_cam_for_output_contract():
    model = build_model(num_classes=4, use_attention=True, pretrained=False)
    images = torch.randn(2, 3, 224, 224)
    tap_post, _ = get_taps(model)

    cam, preds = cam_for(model, images, tap_post, torch.device("cpu"))
    assert cam.shape == (2, 1, 224, 224)
    assert torch.all(cam >= 0) and torch.all(cam <= 1)
    assert preds.shape == (2,)
    assert preds.device.type == "cpu"


def test_energy_inside_lung_from_real_cam_is_in_unit_interval():
    model = build_model(num_classes=4, use_attention=True, pretrained=False)
    images = torch.randn(2, 3, 224, 224)
    tap_post, _ = get_taps(model)
    cam, _ = cam_for(model, images, tap_post, torch.device("cpu"))

    eil = energy_inside_lung(cam, _rect_mask())
    assert eil.shape == (2,)
    assert torch.all(eil >= 0) and torch.all(eil <= 1)


def test_a0_arm_taps_are_exactly_identical():
    """Arm A0 has no gate at all -- post_attn and features.norm5 capture the
    SAME tensor, so EIL_post must equal EIL_pre exactly. This is the WBS
    section 6.3 self-check that the harness is wired to the right layers."""
    model = build_model(num_classes=4, use_attention=False, pretrained=False)
    images = torch.randn(2, 3, 224, 224)
    tap_post, tap_pre = get_taps(model)

    cam_post, preds_post = cam_for(model, images, tap_post, torch.device("cpu"))
    cam_pre, preds_pre = cam_for(model, images, tap_pre, torch.device("cpu"))

    assert torch.equal(preds_post, preds_pre)
    torch.testing.assert_close(cam_post, cam_pre, rtol=0, atol=1e-4)


def test_uniform_gate_taps_are_also_identical():
    """At init (zero-init conv2 -> att uniformly 0.5 everywhere), the
    residual gate f*(1+att) is a UNIFORM scalar rescale, not a spatial
    transform. Grad-CAM's ReLU + min-max normalisation is invariant to a
    uniform positive rescale of the activations, so the two taps should
    still agree even though a (trivial) gate now sits between them. This
    independently corroborates the zero-init design (WBS section 4.3) --
    it really does start as a pure identity, verified via Grad-CAM too."""
    model = build_model(num_classes=4, use_attention=True, gate_mode="residual", pretrained=False)
    images = torch.randn(2, 3, 224, 224)
    tap_post, tap_pre = get_taps(model)

    cam_post, _ = cam_for(model, images, tap_post, torch.device("cpu"))
    cam_pre, _ = cam_for(model, images, tap_pre, torch.device("cpu"))
    torch.testing.assert_close(cam_post, cam_pre, rtol=0, atol=1e-3)


def test_nonuniform_gate_taps_genuinely_diverge():
    """Once the gate is spatially non-uniform (as it will be after training),
    pre-gate and post-gate evidence must differ -- otherwise the harness
    couldn't answer H3 ("did the backbone change, or did the module just
    multiply by a lung mask?", WBS section 6.3) at all."""
    model = build_model(num_classes=4, use_attention=True, gate_mode="residual", pretrained=False)

    def forced_forward(feat):
        att = torch.zeros(feat.shape[0], 1, 7, 7, device=feat.device)
        att[:, :, :, :4] = 0.95
        att[:, :, :, 4:] = 0.05
        att_logits = torch.logit(att, eps=1e-6)
        return feat * (1.0 + att), att, att_logits

    model.attn.forward = forced_forward

    images = torch.randn(2, 3, 224, 224)
    tap_post, tap_pre = get_taps(model)
    cam_post, preds_post = cam_for(model, images, tap_post, torch.device("cpu"))
    cam_pre, preds_pre = cam_for(model, images, tap_pre, torch.device("cpu"))

    assert torch.equal(preds_post, preds_pre)  # the gate doesn't change the prediction path
    max_diff = (cam_post - cam_pre).abs().max().item()
    assert max_diff > 1e-2, f"expected clear divergence with a strongly non-uniform gate, got {max_diff}"
