from __future__ import annotations

import json

import pytest

from src.external_eval import evaluate_external_dataset, load_external_mapping


def test_external_mapping_required_to_avoid_silent_rsna_mapping():
    with pytest.raises(ValueError, match="explicit class-mapping"):
        load_external_mapping(None)


def test_external_mapping_contract_validates_required_keys(tmp_path):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps({"dataset": "RSNA"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing keys"):
        load_external_mapping(path)


def test_external_eval_leaves_values_pending_after_valid_mapping(tmp_path):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps({
        "dataset": "RSNA",
        "source_labels": ["Normal", "Pneumonia"],
        "target_labels": ["Normal", "Lung Opacity"],
        "mapping": {"Normal": "Normal", "Pneumonia": "Lung Opacity"},
        "limitation": "Binary RSNA labels are not equivalent to the four-class primary task.",
    }), encoding="utf-8")
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        evaluate_external_dataset(mapping_config=path)
