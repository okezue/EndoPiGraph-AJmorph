from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from .io import read_image


IMAGE_EXTS = {".tif", ".tiff", ".ome.tif", ".ome.tiff"}


def iter_image_files(root: str | Path) -> Iterable[Path]:
    root = Path(root)
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        # handle compound suffixes like .ome.tif
        suffix = p.suffix.lower()
        suffix2 = "".join([s.lower() for s in p.suffixes[-2:]]) if len(p.suffixes) >= 2 else None
        if suffix in IMAGE_EXTS or (suffix2 in IMAGE_EXTS):
            yield p


def make_manifest(input_dir: str | Path, out_csv: str | Path) -> pd.DataFrame:
    """Scan a directory for images and write a manifest CSV.

    The manifest is a simple table with columns:
      - image_id
      - path
      - channel_names_json

    The pipeline uses this to know which images to process.
    """
    input_dir = Path(input_dir)
    out_csv = Path(out_csv)
    rows: List[dict] = []

    files = sorted(iter_image_files(input_dir))
    if not files:
        raise FileNotFoundError(f"No image files found under {input_dir}")

    for idx, f in enumerate(files):
        image_id = f.stem
        try:
            _arr, ch_names = read_image(f)
        except Exception:
            ch_names = []
        rows.append(
            {
                "image_id": image_id,
                "path": str(f.resolve()),
                "channel_names_json": pd.io.json.dumps(ch_names),
            }
        )

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
