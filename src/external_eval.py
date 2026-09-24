"""External/OOD evaluation interface.

RSNA is not the same four-class task as the COVID-19 Radiography Database.
This module therefore requires an explicit class-mapping config instead of
inventing a mapping silently.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_external_mapping(path: Optional[str | Path]) -> Dict[str, Any]:
    if path is None:
        raise ValueError(
            "External evaluation requires an explicit class-mapping JSON. "
            "RSNA is binary pneumonia/normal and cannot be silently mapped to "
            "the four COVID-19 Radiography classes."
        )
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    required = {"dataset", "source_labels", "target_labels", "mapping", "limitation"}
    missing = required - set(mapping)
    if missing:
        raise ValueError(f"External mapping config is missing keys: {sorted(missing)}")
    return mapping


def evaluate_external_dataset(*args, mapping_config: Optional[str | Path] = None, **kwargs) -> Dict[str, float]:
    """Placeholder-safe external evaluation entry point.

    The implementation intentionally starts by validating the mapping contract.
    Dataset-specific loading can be added behind this interface without
    changing callers or pretending unavailable OOD numbers exist.
    """
    load_external_mapping(mapping_config)
    raise NotImplementedError(
        "External dataset loading/evaluation is not implemented yet. The mapping "
        "contract exists so OOD fields can remain NaN until a predefined, reviewed "
        "RSNA experiment setting is added."
    )
