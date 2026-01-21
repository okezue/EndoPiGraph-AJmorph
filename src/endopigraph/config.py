from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


DEFAULTS: Dict[str, Any] = {
    "pixel_size_um": None,
    "segmentation": {
        "method": "cellpose",
        "cellpose": {
            "model_type": "cyto2",
            "diameter": 30,
            "channels": {"cyto": None, "nuclei": None},
            "flow_threshold": 0.4,
            "cellprob_threshold": 0.0,
        },
        "watershed": {
            "nuclei": None,
            "membrane": None,
            "min_cell_area_px": 200,
        },
    },
    "junction_markers": {
        "AJ": {
            "channel": None,
            "threshold": "otsu",
            "min_occupancy": 0.05,
            "dilate_px": 2,
        }
    },
    "graph": {
        "min_contact_px": 10,
    },
    "qc": {
        "make_figures": True,
        "max_images": None,
        "random_seed": 0,
    },
}


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must parse to a dict")

    # Merge defaults (shallow + nested)
    merged = _deep_merge(DEFAULTS, cfg)

    # Required keys
    for k in ("manifest_csv", "output_dir"):
        if k not in merged or merged[k] in (None, ""):
            raise ValueError(f"Missing required config key: {k}")

    return merged


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in base.items():
        if k in override:
            if isinstance(v, dict) and isinstance(override[k], dict):
                out[k] = _deep_merge(v, override[k])
            else:
                out[k] = override[k]
        else:
            out[k] = v
    for k, v in override.items():
        if k not in out:
            out[k] = v
    return out
