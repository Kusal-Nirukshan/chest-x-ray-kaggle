from __future__ import annotations

from pathlib import Path


def test_guided_selection_script_does_not_call_test_evaluation():
    source = Path("scripts/kaggle_guided_selection.py").read_text(encoding="utf-8")
    assert "run_full_arm(" not in source
    assert "evaluate(" not in source
    assert "test_results" not in source


def test_guided_selection_records_baseline_and_validation_eil():
    source = Path("scripts/kaggle_guided_selection.py").read_text(encoding="utf-8")
    assert '("baseline", "none", 0.0, 0.0)' in source
    assert "validation_cam_subset_indices.json" in source
    assert "val_eil_post" in source
    assert "val_eil_pre" in source
