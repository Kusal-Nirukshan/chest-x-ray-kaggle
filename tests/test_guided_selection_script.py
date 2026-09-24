from __future__ import annotations

from pathlib import Path


def test_guided_selection_script_does_not_call_test_evaluation():
    source = Path("scripts/kaggle_guided_selection.py").read_text(encoding="utf-8")
    assert "run_full_arm(" not in source
    assert "evaluate(" not in source
    assert "test_results" not in source
