#!/usr/bin/env python3
"""EndoPiGraph–AJmorph v1 (all-in-one)

This single-file script implements a practical, end-to-end pipeline for:

  1) Downloading/curating BioImage Archive datasets (focused on S-BIAD1540).
  2) Segmenting endothelial cells (and optionally nuclei + Golgi).
  3) Extracting cell-cell interface masks.
  4) Computing adherens junction (AJ / VE-cadherin) morphology features per
     interface and classifying into common morphological classes.
  5) Building an endothelial pi-graph / typed multigraph representation.
  6) Exporting tables, graphs, and publication-ready QC figures.

The implementation is designed to be:
  - self-contained (one file),
  - easy to troubleshoot (verbose logs, deterministic outputs),
  - able to run with or without Cellpose.

Important scientific note
-------------------------
This code provides:
  - a *feature extraction* pipeline that is largely deterministic; and
  - two classification options:
      (A) a transparent heuristic baseline (fast, no labels), and
      (B) a supervised ML classifier trained from your manual labels.

For any serious scientific claim, you should use (B) and report performance
on held-out images. The heuristic baseline is included mainly as a bootstrap.

Data/model scope
----------------
S-BIAD1540 (BioImage Archive) is the dataset referenced by the Polarity-JaM
paper and contains multi-channel microscopy images of endothelial monolayers.
This script can read OME-TIFF/TIFF and uses a manifest CSV/JSON to map
conditions (e.g., shear stress) and channel semantics.

License
-------
This file is provided under the MIT License (same permissive spirit as the
Polarity-JaM repository). If you incorporate code from other sources, ensure
license compatibility.

Quick start (typical workflow)
------------------------------
1) Create a run folder and fetch dataset links:

    python EndoPiGraph_AJmorph_v1_allinone.py bia-info --acc S-BIAD1540 --out runs/S-BIAD1540

2) Download the dataset (recommended: use wget as instructed by BioImage Archive).
   After download, ensure you have the dataset root locally, e.g.:

    data/S-BIAD1540/Files/bioimage_archive.json
    data/S-BIAD1540/Files/*.tif

3) Build a manifest (CSV) from bioimage_archive.json or by scanning for TIFFs:

    python EndoPiGraph_AJmorph_v1_allinone.py make-manifest \
        --dataset-root data/S-BIAD1540 \
        --out runs/S-BIAD1540/manifest.csv

4) Inspect a few images to confirm channel mapping:

    python EndoPiGraph_AJmorph_v1_allinone.py inspect \
        --manifest runs/S-BIAD1540/manifest.csv \
        --n 3 --out runs/S-BIAD1540/inspect

5) Run the pipeline on the manifest:

    python EndoPiGraph_AJmorph_v1_allinone.py run \
        --manifest runs/S-BIAD1540/manifest.csv \
        --out runs/S-BIAD1540/out \
        --aj-channel "VE-cadherin" \
        --cell-channel "VE-cadherin" \
        --nuc-channel "DAPI" \
        --golgi-channel "GM130" \
        --use-cellpose 0

6) (Optional) Export interface patches for labeling:

    python EndoPiGraph_AJmorph_v1_allinone.py export-patches \
        --edges-csv runs/S-BIAD1540/out/all_edges.csv \
        --out runs/S-BIAD1540/patches

7) (Optional) Train an AJ morphology classifier from your labels:

    python EndoPiGraph_AJmorph_v1_allinone.py train \
        --edges-csv runs/S-BIAD1540/out/all_edges.csv \
        --labels-csv runs/S-BIAD1540/labels.csv \
        --out runs/S-BIAD1540/models/ajmorph_rf.joblib

8) Re-run with the trained model:

    python EndoPiGraph_AJmorph_v1_allinone.py run \
        --manifest runs/S-BIAD1540/manifest.csv \
        --out runs/S-BIAD1540/out_ml \
        --ajmorph-model runs/S-BIAD1540/models/ajmorph_rf.joblib

"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as _dt
import json
import math
import os
import re
import shutil
import sys
import textwrap
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Core scientific stack
import tifffile

from scipy import ndimage as ndi

from skimage import exposure
from skimage import filters
from skimage import measure
from skimage import morphology
from skimage import segmentation
from skimage import util

import networkx as nx

# plotting (matplotlib is available)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# optional deps
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
except Exception:  # pragma: no cover
    RandomForestClassifier = None
    train_test_split = None
    classification_report = None
    confusion_matrix = None

# Cellpose is optional. We import lazily in functions.


# -----------------------------
# Utilities
# -----------------------------


def log(msg: str) -> None:
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def die(msg: str, code: int = 1) -> None:
    log("ERROR: " + msg)
    raise SystemExit(code)


def ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def now_slug() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def read_json(path: Union[str, Path]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: Union[str, Path], indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)


def write_text(s: str, path: Union[str, Path]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)


def normalize01(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Percentile normalize to [0,1] per image."""
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    out = (img - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    return out


def maybe_tqdm(it: Iterable, total: Optional[int] = None, desc: str = ""):
    if tqdm is None:
        return it
    return tqdm(it, total=total, desc=desc)


# -----------------------------
# BioStudies / BioImage Archive helpers
# -----------------------------


def bia_study_info_url(acc: str) -> str:
    return f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{acc}/info"


def bia_study_json_url(acc: str) -> str:
    return f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{acc}"


def http_get_json(url: str, timeout: int = 60) -> Any:
    """HTTP GET JSON (requests is imported lazily to keep import surface small)."""
    try:
        import requests
    except Exception as e:  # pragma: no cover
        raise RuntimeError("requests is required for network operations. pip install requests") from e

    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def bia_fetch_info(acc: str, out_dir: Union[str, Path]) -> Dict[str, Any]:
    """Fetch /info JSON and write it to out_dir."""
    out_dir = ensure_dir(out_dir)
    info = http_get_json(bia_study_info_url(acc))
    write_json(info, out_dir / f"{acc}_info.json")

    # convenience text
    ftp = info.get("ftpLink")
    http = info.get("httpLink")
    cols = info.get("columns", [])

    lines = [
        f"Accession: {acc}",
        f"ftpLink: {ftp}",
        f"httpLink: {http}",
        f"nFiles: {info.get('nFiles')}",
        "columns:",
        *[f"  - {c}" for c in cols],
        "",
        "Typical download (BioImage Archive):",
        f"  wget -m -np -nH --cut-dirs=4 -R 'index.html*' {http}/",
    ]
    write_text("\n".join(lines) + "\n", out_dir / f"{acc}_links.txt")
    return info


# -----------------------------
# Manifest handling
# -----------------------------


MANIFEST_COLUMNS = [
    "image_id",
    "path",
    "dataset",
    "shear_stress",
    "supplementation",
    "replicate",
    "section",
    "channel_1",
    "channel_2",
    "channel_3",
    "channel_4",
]


def _flatten_manifest_json(obj: Any) -> List[Dict[str, Any]]:
    """Attempt to flatten various plausible bioimage_archive.json schemas.

    The BioImage Archive 'bioimage_archive.json' schema can vary. This function
    tries a few common patterns and returns a list of row dicts.
    """
    if obj is None:
        return []

    # If it's already a list of dict rows.
    if isinstance(obj, list):
        if all(isinstance(x, dict) for x in obj):
            return obj  # type: ignore
        return []

    if isinstance(obj, dict):
        # Common: {'data': [...]} or {'files': [...]} or {'rows': [...]}
        for key in ("data", "files", "rows", "entries", "objects"):
            if key in obj and isinstance(obj[key], list):
                lst = obj[key]
                if all(isinstance(x, dict) for x in lst):
                    return lst  # type: ignore

        # Sometimes nested: {'section': {'files': [...]}}
        # Fall back: search all values for list-of-dicts.
        for v in obj.values():
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                return v  # type: ignore

    return []


def load_bia_manifest_json(dataset_root: Union[str, Path]) -> Optional[pd.DataFrame]:
    """Load BioImage Archive manifest if present under Files/bioimage_archive.json."""
    dataset_root = Path(dataset_root)
    candidates = [
        dataset_root / "Files" / "bioimage_archive.json",
        dataset_root / "files" / "bioimage_archive.json",
        dataset_root / "bioimage_archive.json",
        dataset_root / "bioimage_archive" / "bioimage_archive.json",
    ]
    for c in candidates:
        if c.exists():
            obj = read_json(c)
            rows = _flatten_manifest_json(obj)
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["__manifest_path"] = str(c)
            return df
    return None


def _guess_file_path_from_row(row: pd.Series) -> Optional[str]:
    # possible keys for file path
    for k in (
        "File",
        "file",
        "filePath",
        "filepath",
        "path",
        "Name",
        "name",
        "RelativePath",
        "relpath",
    ):
        if k in row and isinstance(row[k], str) and row[k].strip():
            return row[k].strip()
    return None


def _guess_channel_name(row: pd.Series, idx: int) -> Optional[str]:
    # expected keys like "Channel 1" or "Channel_1" etc
    candidates = [f"Channel {idx}", f"Channel_{idx}", f"channel_{idx}", f"channel {idx}"]
    for k in candidates:
        if k in row and isinstance(row[k], str) and row[k].strip():
            return row[k].strip()
    return None


def build_manifest(
    dataset_root: Union[str, Path],
    out_csv: Union[str, Path],
    dataset_name: str = "S-BIAD1540",
    prefer_bia_json: bool = True,
    glob_pattern: str = "**/*.tif*",
) -> pd.DataFrame:
    """Create a manifest CSV.

    Priority:
      1) If bioimage_archive.json exists and prefer_bia_json=True: parse it.
      2) Else: scan dataset_root for TIFF files.

    The produced manifest is intentionally simple and can be edited.
    """
    dataset_root = Path(dataset_root)
    out_csv = Path(out_csv)
    ensure_dir(out_csv.parent)

    rows_out: List[Dict[str, Any]] = []

    df_bia = load_bia_manifest_json(dataset_root) if prefer_bia_json else None

    if df_bia is not None:
        log(f"Found bioimage_archive.json with {len(df_bia)} rows. Building manifest from it.")

        # Normalize column names
        def get_any(row: pd.Series, keys: Sequence[str]) -> Any:
            for k in keys:
                if k in row and pd.notna(row[k]):
                    return row[k]
            return None

        for idx, row in df_bia.iterrows():
            fpath = _guess_file_path_from_row(row)
            if not fpath:
                continue

            # Skip non-image files by extension.
            if not re.search(r"\.(tif|tiff|ome\.tif|ome\.tiff)$", str(fpath), flags=re.IGNORECASE):
                continue

            image_id = Path(str(fpath)).stem
            # Many BIA manifests store paths like "Files/xxx.tif".
            # Make path relative to dataset_root.
            rel = str(fpath)
            if rel.startswith(str(dataset_root)):
                rel = str(Path(rel).relative_to(dataset_root))
            # common case: rel already includes 'Files/' prefix.

            rows_out.append(
                {
                    "image_id": image_id,
                    "path": rel,
                    "dataset": dataset_name,
                    "shear_stress": str(get_any(row, ["Sheer Stress", "Shear Stress", "shear_stress", "shear", "Shear"]) or ""),
                    "supplementation": str(get_any(row, ["Supplementation", "supplement", "supplementation"]) or ""),
                    "replicate": str(get_any(row, ["replication", "replicate", "Replicate"]) or ""),
                    "section": str(get_any(row, ["Section", "section"]) or ""),
                    "channel_1": str(_guess_channel_name(row, 1) or ""),
                    "channel_2": str(_guess_channel_name(row, 2) or ""),
                    "channel_3": str(_guess_channel_name(row, 3) or ""),
                    "channel_4": str(_guess_channel_name(row, 4) or ""),
                }
            )

        if not rows_out:
            log("bioimage_archive.json was present but no TIFF rows were parsed; falling back to filesystem scan.")

    if not rows_out:
        log(f"Scanning {dataset_root} for images with pattern '{glob_pattern}'")
        for p in sorted(dataset_root.glob(glob_pattern)):
            if p.is_dir():
                continue
            if not re.search(r"\.(tif|tiff)$", p.name, flags=re.IGNORECASE):
                continue
            rel = str(p.relative_to(dataset_root))
            rows_out.append(
                {
                    "image_id": p.stem,
                    "path": rel,
                    "dataset": dataset_name,
                    "shear_stress": "",
                    "supplementation": "",
                    "replicate": "",
                    "section": "",
                    "channel_1": "",
                    "channel_2": "",
                    "channel_3": "",
                    "channel_4": "",
                }
            )

    if not rows_out:
        die(f"No images found under {dataset_root}")

    df_out = pd.DataFrame(rows_out)
    # Deduplicate paths
    df_out = df_out.drop_duplicates(subset=["path"]).reset_index(drop=True)

    df_out.to_csv(out_csv, index=False)
    log(f"Wrote manifest with {len(df_out)} images to {out_csv}")
    return df_out


# -----------------------------
# Image I/O and channel handling
# -----------------------------


@dataclass
class ImageRecord:
    image_id: str
    path: Path
    dataset: str = ""
    shear_stress: str = ""
    supplementation: str = ""
    replicate: str = ""
    section: str = ""
    channel_1: str = ""
    channel_2: str = ""
    channel_3: str = ""
    channel_4: str = ""


def load_manifest_csv(manifest_csv: Union[str, Path], dataset_root: Union[str, Path]) -> List[ImageRecord]:
    df = pd.read_csv(manifest_csv)
    dataset_root = Path(dataset_root)
    recs: List[ImageRecord] = []
    for _, r in df.iterrows():
        rel = str(r.get("path", ""))
        if not rel:
            continue
        recs.append(
            ImageRecord(
                image_id=str(r.get("image_id", Path(rel).stem)),
                path=(dataset_root / rel).resolve(),
                dataset=str(r.get("dataset", "")),
                shear_stress=str(r.get("shear_stress", "")),
                supplementation=str(r.get("supplementation", "")),
                replicate=str(r.get("replicate", "")),
                section=str(r.get("section", "")),
                channel_1=str(r.get("channel_1", "")),
                channel_2=str(r.get("channel_2", "")),
                channel_3=str(r.get("channel_3", "")),
                channel_4=str(r.get("channel_4", "")),
            )
        )
    return recs


def _extract_ome_channel_names(tif: tifffile.TiffFile) -> List[str]:
    """Try to extract channel names from OME-XML metadata.

    This is best-effort; if metadata is missing or parsing fails, return [].
    """
    try:
        ome = tif.ome_metadata
        if ome is None:
            return []
        # Very lightweight XML parsing by regex (avoid heavy XML deps).
        # Look for Channel Name="..." patterns.
        names = re.findall(r"Channel[^>]*Name=\"([^\"]+)\"", ome)
        # Deduplicate while preserving order.
        out = []
        seen = set()
        for n in names:
            if n not in seen:
                out.append(n)
                seen.add(n)
        return out
    except Exception:
        return []


def read_image_any(path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read TIFF/OME-TIFF into array and metadata.

    Returns:
      arr: ndarray
      meta: dict with at least keys: 'shape', 'dtype', 'channel_names'

    The returned array is in one of these forms:
      - (H, W)
      - (C, H, W)
      - (Z, C, H, W)
      - (T, Z, C, H, W)

    The pipeline downstream will try to squeeze/choose a 2D or (C,H,W)
    representation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    with tifffile.TiffFile(str(path)) as tif:
        arr = tif.asarray()
        meta: Dict[str, Any] = {
            "shape": tuple(arr.shape),
            "dtype": str(arr.dtype),
            "is_ome": tif.is_ome,
            "channel_names": _extract_ome_channel_names(tif),
        }
    return arr, meta


def squeeze_to_cyx(arr: np.ndarray) -> np.ndarray:
    """Convert various TIFF shapes to (C,H,W) when possible.

    Heuristics:
      - If 2D: return (1,H,W)
      - If 3D and first dim small: assume (C,H,W)
      - If 3D and last dim small: assume (H,W,C) and transpose
      - If 4D: try (Z,C,H,W) -> max over Z
      - If 5D: try (T,Z,C,H,W) -> max over T and Z

    If the shape is ambiguous, we do best-effort and warn via meta.
    """
    if arr.ndim == 2:
        return arr[None, ...]

    if arr.ndim == 3:
        # (C,H,W)
        if arr.shape[0] <= 8 and arr.shape[1] > 16 and arr.shape[2] > 16:
            return arr
        # (H,W,C)
        if arr.shape[2] <= 8 and arr.shape[0] > 16 and arr.shape[1] > 16:
            return np.transpose(arr, (2, 0, 1))
        # otherwise, treat first dim as channels
        return arr

    if arr.ndim == 4:
        # (Z,C,H,W)
        if arr.shape[1] <= 8 and arr.shape[2] > 16 and arr.shape[3] > 16:
            return np.max(arr, axis=0)
        # (C,Z,H,W)
        if arr.shape[0] <= 8 and arr.shape[2] > 16 and arr.shape[3] > 16:
            return np.max(arr, axis=1)
        # (H,W,C,Z) uncommon
        # fallback: flatten leading dims into channels
        c = int(np.prod(arr.shape[:-2]))
        return arr.reshape((c, arr.shape[-2], arr.shape[-1]))

    if arr.ndim == 5:
        # (T,Z,C,H,W)
        if arr.shape[2] <= 8:
            return np.max(arr, axis=(0, 1))
        # (Z,T,C,H,W)
        if arr.shape[2] <= 8 and arr.shape[0] < 32:
            return np.max(arr, axis=(0, 1))
        c = int(np.prod(arr.shape[:-2]))
        return arr.reshape((c, arr.shape[-2], arr.shape[-1]))

    # Higher dims: flatten to channels
    c = int(np.prod(arr.shape[:-2]))
    return arr.reshape((c, arr.shape[-2], arr.shape[-1]))


def channel_index_from_semantics(
    rec: Optional[ImageRecord],
    arr_cyx: np.ndarray,
    channel_names: Sequence[str],
    want: str,
    fallback: Optional[int] = None,
) -> Optional[int]:
    """Map a desired semantic channel name to an index.

    This tries, in order:
      1) If rec has channel_1..channel_4 strings, match 'want' as substring.
      2) Match against OME channel_names as substring.
      3) If want is an integer string, use that.
      4) fallback.

    Matching is case-insensitive and ignores non-alphanumerics.
    """

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", s.lower())

    want_n = norm(want)

    # (1) match manifest channel_* fields
    if rec is not None:
        chan_fields = [rec.channel_1, rec.channel_2, rec.channel_3, rec.channel_4]
        for i, label in enumerate(chan_fields):
            if not label:
                continue
            if want_n in norm(label) or norm(label) in want_n:
                if i < arr_cyx.shape[0]:
                    return i

    # (2) match OME channel names
    for i, label in enumerate(channel_names):
        if not label:
            continue
        if want_n in norm(label) or norm(label) in want_n:
            if i < arr_cyx.shape[0]:
                return i

    # (3) parse as integer
    if re.fullmatch(r"\d+", want.strip()):
        idx = int(want.strip())
        if 0 <= idx < arr_cyx.shape[0]:
            return idx

    return fallback


# -----------------------------
# Segmentation
# -----------------------------


@dataclass
class SegmentationConfig:
    use_cellpose: bool = False
    cellpose_model_cells: str = "cyto"  # 'cyto', 'cyto2', etc
    cellpose_model_nuclei: str = "nuclei"
    cellpose_diameter: Optional[float] = None
    cellpose_flow_threshold: float = 0.4
    cellpose_cellprob_threshold: float = 0.0
    # watershed fallback params
    ws_sigma: float = 1.5
    ws_min_distance: int = 10


def segment_cells_cellpose(img: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    """Segment cells with Cellpose (if installed)."""
    try:
        from cellpose import models
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Cellpose is not installed. Install with: pip install cellpose"
        ) from e

    # Cellpose 4.x uses CellposeModel instead of Cellpose
    model = models.CellposeModel(model_type=cfg.cellpose_model_cells, gpu=False)
    # Cellpose expects 2D or 3D images, channels handling is separate.
    masks, flows, styles = model.eval(
        img,
        diameter=cfg.cellpose_diameter,
        channels=[0, 0],
        flow_threshold=cfg.cellpose_flow_threshold,
        cellprob_threshold=cfg.cellpose_cellprob_threshold,
    )
    masks = masks.astype(np.int32)
    return masks


def segment_nuclei_cellpose(img: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    """Segment nuclei with Cellpose nuclei model."""
    try:
        from cellpose import models
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Cellpose is not installed. Install with: pip install cellpose"
        ) from e

    # Cellpose 4.x uses CellposeModel instead of Cellpose
    model = models.CellposeModel(model_type=cfg.cellpose_model_nuclei, gpu=False)
    masks, flows, styles = model.eval(
        img,
        diameter=cfg.cellpose_diameter,
        channels=[0, 0],
        flow_threshold=cfg.cellpose_flow_threshold,
        cellprob_threshold=cfg.cellpose_cellprob_threshold,
    )
    return masks.astype(np.int32)


def segment_cells_watershed(img: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    """Fallback cell segmentation using watershed on a smoothed image."""
    img = normalize01(img)
    sm = filters.gaussian(img, sigma=cfg.ws_sigma)
    thr = filters.threshold_otsu(sm)
    bw = sm > thr
    bw = morphology.remove_small_objects(bw, 64)
    bw = morphology.binary_closing(bw, morphology.disk(2))

    dist = ndi.distance_transform_edt(bw)
    peaks = morphology.local_maxima(dist)
    # Ensure some markers exist
    markers = measure.label(peaks)
    if markers.max() < 2:
        # fallback: use h-maxima
        peaks = morphology.h_maxima(dist, 0.2 * float(dist.max() or 1.0))
        markers = measure.label(peaks)

    labels = segmentation.watershed(-dist, markers, mask=bw)
    labels = labels.astype(np.int32)

    # Relabel to compact
    labels = measure.label(labels > 0, connectivity=1)
    return labels.astype(np.int32)


def segment_nuclei_simple(img: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    img = normalize01(img)
    sm = filters.gaussian(img, sigma=1.0)
    thr = filters.threshold_otsu(sm)
    bw = sm > thr
    bw = morphology.remove_small_objects(bw, 32)
    bw = morphology.binary_opening(bw, morphology.disk(1))
    dist = ndi.distance_transform_edt(bw)
    peaks = morphology.local_maxima(dist)
    markers = measure.label(peaks)
    labels = segmentation.watershed(-dist, markers, mask=bw)
    labels = labels.astype(np.int32)
    return labels


def segment_cells(img: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    if cfg.use_cellpose:
        return segment_cells_cellpose(img, cfg)
    return segment_cells_watershed(img, cfg)


def segment_nuclei(img: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    if cfg.use_cellpose:
        return segment_nuclei_cellpose(img, cfg)
    return segment_nuclei_simple(img, cfg)


# -----------------------------
# Region properties / polarity
# -----------------------------


@dataclass
class CellProps:
    label: int
    area: float
    perimeter: float
    centroid_y: float
    centroid_x: float
    eccentricity: float
    orientation_rad: float
    major_axis_length: float
    minor_axis_length: float


def compute_cell_props(labels: np.ndarray) -> List[CellProps]:
    props = []
    for r in measure.regionprops(labels):
        if r.label == 0:
            continue
        props.append(
            CellProps(
                label=int(r.label),
                area=float(r.area),
                perimeter=float(getattr(r, "perimeter", 0.0)),
                centroid_y=float(r.centroid[0]),
                centroid_x=float(r.centroid[1]),
                eccentricity=float(getattr(r, "eccentricity", 0.0)),
                orientation_rad=float(getattr(r, "orientation", 0.0)),
                major_axis_length=float(getattr(r, "major_axis_length", 0.0)),
                minor_axis_length=float(getattr(r, "minor_axis_length", 0.0)),
            )
        )
    return props


def assign_nucleus_to_cells(cell_labels: np.ndarray, nuc_labels: np.ndarray) -> Dict[int, Optional[int]]:
    """Assign each cell to the nucleus label with max overlap."""
    out: Dict[int, Optional[int]] = {}
    if cell_labels.shape != nuc_labels.shape:
        raise ValueError("cell_labels and nuc_labels must have same shape")

    # For each cell label, find nucleus label with maximum overlap.
    max_cell = int(cell_labels.max())
    for cid in range(1, max_cell + 1):
        mask = cell_labels == cid
        if not mask.any():
            continue
        nucs = nuc_labels[mask]
        nucs = nucs[nucs > 0]
        if nucs.size == 0:
            out[cid] = None
            continue
        vals, counts = np.unique(nucs, return_counts=True)
        out[cid] = int(vals[int(np.argmax(counts))])
    return out


def centroid_of_label(mask_labels: np.ndarray, label_id: int) -> Optional[Tuple[float, float]]:
    if label_id <= 0:
        return None
    coords = np.argwhere(mask_labels == label_id)
    if coords.size == 0:
        return None
    yx = coords.mean(axis=0)
    return float(yx[0]), float(yx[1])


def estimate_golgi_centroid_per_cell(cell_labels: np.ndarray, golgi_img: np.ndarray) -> Dict[int, Optional[Tuple[float, float]]]:
    """Very simple Golgi centroid estimate per cell using Otsu threshold.

    Steps:
      - Threshold golgi_img with Otsu.
      - For each cell, take golgi pixels within the cell mask.
      - If there are multiple components, pick the largest.
      - Return centroid.

    This is an intentionally conservative implementation; you may replace it
    with a more faithful Golgi segmentation if needed.
    """
    golgi = normalize01(golgi_img)
    thr = filters.threshold_otsu(golgi) if np.any(golgi > 0) else 1.0
    bw = golgi > thr
    bw = morphology.remove_small_objects(bw, 8)

    out: Dict[int, Optional[Tuple[float, float]]] = {}
    max_cell = int(cell_labels.max())
    for cid in range(1, max_cell + 1):
        cm = cell_labels == cid
        gmask = bw & cm
        if not gmask.any():
            out[cid] = None
            continue
        lab = measure.label(gmask)
        if lab.max() > 1:
            # pick largest component
            sizes = [(i, int((lab == i).sum())) for i in range(1, lab.max() + 1)]
            i_best = max(sizes, key=lambda t: t[1])[0]
            gmask = lab == i_best
        coords = np.argwhere(gmask)
        yx = coords.mean(axis=0)
        out[cid] = (float(yx[0]), float(yx[1]))
    return out


def angle_deg_from_vec(dy: float, dx: float) -> float:
    # angle in degrees in [-180, 180]
    return float(np.degrees(np.arctan2(dy, dx)))


def wrap_angle_deg(angle: float) -> float:
    # map to [0, 360)
    a = angle % 360.0
    if a < 0:
        a += 360.0
    return a


def compute_polarity_features(
    cell_props: List[CellProps],
    nuc_centroids: Dict[int, Optional[Tuple[float, float]]],
    golgi_centroids: Dict[int, Optional[Tuple[float, float]]],
    flow_dir_deg: float = 0.0,
) -> pd.DataFrame:
    """Compute basic polarity features per cell.

    Returns DataFrame with columns:
      cell_id, nuc_y, nuc_x, golgi_y, golgi_x, polarity_dx, polarity_dy,
      polarity_angle_deg, polarity_alignment_deg, polarity_length

    flow_dir_deg is direction of external cue (e.g., flow) in degrees.
    """
    rows = []
    for cp in cell_props:
        cid = cp.label
        nuc = nuc_centroids.get(cid)
        gol = golgi_centroids.get(cid)
        if nuc is None or gol is None:
            rows.append(
                {
                    "cell_id": cid,
                    "nuc_y": np.nan,
                    "nuc_x": np.nan,
                    "golgi_y": np.nan,
                    "golgi_x": np.nan,
                    "polarity_dx": np.nan,
                    "polarity_dy": np.nan,
                    "polarity_angle_deg": np.nan,
                    "polarity_alignment_deg": np.nan,
                    "polarity_length": np.nan,
                }
            )
            continue
        dy = gol[0] - nuc[0]
        dx = gol[1] - nuc[1]
        ang = wrap_angle_deg(angle_deg_from_vec(dy, dx))
        # alignment: smallest absolute difference to flow direction (mod 180)
        d = abs(ang - flow_dir_deg) % 360.0
        d = min(d, 360.0 - d)
        # undirected alignment
        d_undirected = min(d, abs(d - 180.0))
        rows.append(
            {
                "cell_id": cid,
                "nuc_y": nuc[0],
                "nuc_x": nuc[1],
                "golgi_y": gol[0],
                "golgi_x": gol[1],
                "polarity_dx": dx,
                "polarity_dy": dy,
                "polarity_angle_deg": ang,
                "polarity_alignment_deg": d_undirected,
                "polarity_length": float(math.hypot(dx, dy)),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------
# Interface extraction
# -----------------------------


@dataclass
class Interface:
    cell_i: int
    cell_j: int
    coords: np.ndarray  # (N,2) yx coords of boundary pixels belonging to contact
    contact_px: int


def compute_interfaces(labels: np.ndarray, connectivity: int = 1) -> List[Interface]:
    """Compute cell-cell interfaces from a labeled mask.

    We identify boundary pixels of each cell and find adjacent labels.

    Returns list of Interface objects with boundary coordinates.
    """
    labels = labels.astype(np.int32)
    # Create adjacency by dilation
    selem = morphology.disk(1) if connectivity == 1 else morphology.disk(2)

    interfaces: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    # boundary pixels: label changes in neighborhood
    # We'll compute for each pixel, its neighbors (4-connected) and store pairs.
    H, W = labels.shape
    # pad for safe neighbor checks
    lab = labels

    # We'll iterate over pixels where label>0 and check right/down neighbors to avoid double counting
    for y in range(H):
        for x in range(W):
            a = int(lab[y, x])
            if a <= 0:
                continue
            # right
            if x + 1 < W:
                b = int(lab[y, x + 1])
                if b > 0 and b != a:
                    i, j = (a, b) if a < b else (b, a)
                    interfaces.setdefault((i, j), []).append((y, x))
                    interfaces.setdefault((i, j), []).append((y, x + 1))
            # down
            if y + 1 < H:
                b = int(lab[y + 1, x])
                if b > 0 and b != a:
                    i, j = (a, b) if a < b else (b, a)
                    interfaces.setdefault((i, j), []).append((y, x))
                    interfaces.setdefault((i, j), []).append((y + 1, x))

    out: List[Interface] = []
    for (i, j), pts in interfaces.items():
        if not pts:
            continue
        coords = np.array(list({p for p in pts}), dtype=np.int32)
        out.append(Interface(cell_i=i, cell_j=j, coords=coords, contact_px=int(coords.shape[0])))

    return out


def interface_mask_from_coords(shape: Tuple[int, int], coords: np.ndarray, dilate: int = 2) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if coords.size == 0:
        return mask
    mask[coords[:, 0], coords[:, 1]] = True
    if dilate > 0:
        mask = morphology.binary_dilation(mask, morphology.disk(dilate))
    return mask


# -----------------------------
# AJ morphology features
# -----------------------------


AJ_CLASSES = ["straight", "thick", "thick_to_reticular", "reticular", "fingers", "other"]


@dataclass
class AJFeatures:
    mean_intensity: float
    max_intensity: float
    std_intensity: float
    occupancy: float
    intensity_per_interface: float
    cluster_count: int
    cluster_area_mean: float
    skeleton_len: float
    linearity_index: float
    thickness_proxy: float


def compute_interface_threshold(img: np.ndarray, interface_masks: List[np.ndarray]) -> float:
    """Compute a single AJ threshold for an image based on interface pixels.

    Strategy:
      - collect pixels within all interface masks (downsample if huge)
      - compute Otsu threshold on those pixels
    """
    if not interface_masks:
        return float(filters.threshold_otsu(img))

    vals = []
    for m in interface_masks:
        v = img[m]
        if v.size == 0:
            continue
        vals.append(v.ravel())
    if not vals:
        return float(filters.threshold_otsu(img))
    v = np.concatenate(vals)
    if v.size > 200000:
        # random subsample for speed
        rng = np.random.default_rng(0)
        v = rng.choice(v, size=200000, replace=False)
    if np.all(v == v.flat[0]):
        return float(v.flat[0])
    return float(filters.threshold_otsu(v))


def extract_aj_features_for_interface(
    aj_img: np.ndarray,
    iface_mask: np.ndarray,
    thr: float,
) -> AJFeatures:
    """Compute per-interface AJ features."""
    vals = aj_img[iface_mask]
    if vals.size == 0:
        return AJFeatures(
            mean_intensity=0.0,
            max_intensity=0.0,
            std_intensity=0.0,
            occupancy=0.0,
            intensity_per_interface=0.0,
            cluster_count=0,
            cluster_area_mean=0.0,
            skeleton_len=0.0,
            linearity_index=0.0,
            thickness_proxy=0.0,
        )

    mean_int = float(vals.mean())
    max_int = float(vals.max())
    std_int = float(vals.std())

    # intensity mask (junction protein region)
    bw = np.zeros_like(iface_mask, dtype=bool)
    bw[iface_mask] = aj_img[iface_mask] >= thr

    iface_area = float(iface_mask.sum())
    int_area = float(bw.sum())
    occupancy = float(int_area / iface_area) if iface_area > 0 else 0.0

    # intensity per interface area (similar to Polarity-JaM concept)
    intensity_sum = float(aj_img[bw].sum()) if int_area > 0 else 0.0
    intensity_per_interface = float(intensity_sum / iface_area) if iface_area > 0 else 0.0

    # clusters
    lab = measure.label(bw)
    cluster_count = int(lab.max())
    if cluster_count > 0:
        areas = [int((lab == k).sum()) for k in range(1, cluster_count + 1)]
        cluster_area_mean = float(np.mean(areas)) if areas else 0.0
    else:
        cluster_area_mean = 0.0

    # skeleton-based metrics
    sk = morphology.skeletonize(bw)
    skeleton_len = float(sk.sum())

    # linearity index: skeleton length / straight line distance between endpoints
    linearity_index = 0.0
    if skeleton_len > 0:
        endpoints = _skeleton_endpoints(sk)
        if endpoints.shape[0] >= 2:
            # choose farthest pair
            dmax = 0.0
            p0 = endpoints[0]
            p1 = endpoints[1]
            for a in endpoints:
                for b in endpoints:
                    d = float(np.hypot(a[0] - b[0], a[1] - b[1]))
                    if d > dmax:
                        dmax = d
                        p0, p1 = a, b
            if dmax > 0:
                linearity_index = float(skeleton_len / dmax)
            else:
                linearity_index = 0.0
        else:
            linearity_index = 0.0

    # thickness proxy: cluster area / skeleton length (roughly proportional to thickness)
    thickness_proxy = float(int_area / skeleton_len) if skeleton_len > 0 else 0.0

    return AJFeatures(
        mean_intensity=mean_int,
        max_intensity=max_int,
        std_intensity=std_int,
        occupancy=occupancy,
        intensity_per_interface=intensity_per_interface,
        cluster_count=cluster_count,
        cluster_area_mean=cluster_area_mean,
        skeleton_len=skeleton_len,
        linearity_index=linearity_index,
        thickness_proxy=thickness_proxy,
    )


def _skeleton_endpoints(sk: np.ndarray) -> np.ndarray:
    """Return endpoints of a skeleton as an array of yx coords."""
    if sk.dtype != bool:
        sk = sk.astype(bool)
    # Count neighbors in 8-neighborhood
    kernel = np.array(
        [[1, 1, 1],
         [1, 10, 1],
         [1, 1, 1]],
        dtype=np.uint8,
    )
    conv = ndi.convolve(sk.astype(np.uint8), kernel, mode="constant", cval=0)
    # For skeleton pixels, neighbor count = (conv-10)
    # endpoints have exactly 1 neighbor.
    endpoints = np.argwhere((sk) & (conv == 11))
    return endpoints.astype(np.int32)


def ajmorph_heuristic_label(feat: AJFeatures) -> str:
    """Heuristic AJ morphology label.

    This is *not* a validated classifier. It is only a baseline.

    Intuition:
      - straight: low tortuosity (linearity close to 1), low thickness, few clusters
      - thick: high thickness and occupancy
      - reticular: many clusters and/or high tortuosity
      - thick_to_reticular: between thick and reticular
      - fingers: sparse occupancy + high tortuosity or fragmented clusters
    """
    if feat.occupancy < 0.02 and feat.mean_intensity < 0.05:
        return "other"

    # Normalize a few signals
    tort = feat.linearity_index  # >=1 for non-straight; near 1 for straight

    if feat.cluster_count >= 6 and feat.thickness_proxy < 6.0:
        return "reticular"

    if feat.thickness_proxy >= 10.0 and feat.occupancy >= 0.25:
        # thick junctions
        if feat.cluster_count >= 4 or tort >= 1.5:
            return "thick_to_reticular"
        return "thick"

    if tort <= 1.25 and feat.cluster_count <= 3 and feat.thickness_proxy < 10.0:
        return "straight"

    # fingers: fragmented + elongated
    if feat.occupancy < 0.18 and (tort >= 1.6 or feat.cluster_count >= 4):
        return "fingers"

    # intermediate
    if feat.cluster_count >= 4:
        return "thick_to_reticular" if feat.thickness_proxy >= 7.0 else "reticular"

    return "other"


def feature_dict(feat: AJFeatures) -> Dict[str, Any]:
    return dataclasses.asdict(feat)


# -----------------------------
# Graph building
# -----------------------------


def build_pi_graph(
    cell_props: List[CellProps],
    edges_df: pd.DataFrame,
    include_self_loops: bool = False,
) -> nx.MultiGraph:
    """Build a MultiGraph with per-edge AJ attributes.

    Nodes: cell IDs
    Edges: one per interface (cell_i, cell_j)

    Node attributes:
      - centroid_x, centroid_y, area, ...

    Edge attributes:
      - contact_px
      - AJ features
      - aj_morph (label)
    """
    G = nx.MultiGraph()

    for cp in cell_props:
        G.add_node(
            cp.label,
            centroid_x=cp.centroid_x,
            centroid_y=cp.centroid_y,
            area=cp.area,
            perimeter=cp.perimeter,
            eccentricity=cp.eccentricity,
            orientation_rad=cp.orientation_rad,
            major_axis_length=cp.major_axis_length,
            minor_axis_length=cp.minor_axis_length,
        )

    for _, r in edges_df.iterrows():
        i = int(r["cell_i"])
        j = int(r["cell_j"])
        if (i == j) and (not include_self_loops):
            continue
        attrs = r.to_dict()
        # Remove redundant columns
        for k in ["cell_i", "cell_j"]:
            attrs.pop(k, None)
        G.add_edge(i, j, **attrs)

    return G


def graph_to_json(G: nx.MultiGraph) -> Dict[str, Any]:
    nodes = []
    for n, d in G.nodes(data=True):
        nodes.append({"id": int(n), **d})

    edges = []
    for u, v, k, d in G.edges(keys=True, data=True):
        edges.append({"u": int(u), "v": int(v), "key": int(k), **d})

    return {"nodes": nodes, "edges": edges}


# -----------------------------
# Visualization
# -----------------------------


def save_segmentation_qc(
    img: np.ndarray,
    labels: np.ndarray,
    out_path: Union[str, Path],
    title: str = "Segmentation QC",
) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    img_n = normalize01(img)
    rgb = np.stack([img_n, img_n, img_n], axis=-1)
    # boundaries
    b = segmentation.find_boundaries(labels, mode="outer")
    rgb[b] = np.array([1.0, 0.0, 0.0])

    plt.figure(figsize=(6, 6), dpi=200)
    plt.imshow(rgb)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_graph_qc(
    aj_img: np.ndarray,
    G: nx.MultiGraph,
    out_path: Union[str, Path],
    title: str = "AJ pi-graph",
) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    img_n = normalize01(aj_img)
    plt.figure(figsize=(6, 6), dpi=200)
    plt.imshow(img_n, cmap="gray")

    # nodes
    xs = []
    ys = []
    for n, d in G.nodes(data=True):
        xs.append(d.get("centroid_x", 0.0))
        ys.append(d.get("centroid_y", 0.0))
    plt.scatter(xs, ys, s=6)

    # edges colored by aj_morph
    color_map = {
        "straight": "#1f77b4",
        "thick": "#d62728",
        "thick_to_reticular": "#ff7f0e",
        "reticular": "#2ca02c",
        "fingers": "#9467bd",
        "other": "#7f7f7f",
    }

    for u, v, d in G.edges(data=True):
        du = G.nodes[u]
        dv = G.nodes[v]
        x0, y0 = du.get("centroid_x", 0.0), du.get("centroid_y", 0.0)
        x1, y1 = dv.get("centroid_x", 0.0), dv.get("centroid_y", 0.0)
        lab = str(d.get("aj_morph", "other"))
        c = color_map.get(lab, "#7f7f7f")
        plt.plot([x0, x1], [y0, y1], linewidth=1.0, alpha=0.8, color=c)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_summary_barplot(
    edges_df: pd.DataFrame,
    out_path: Union[str, Path],
    group_cols: List[str],
    title: str,
) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    if edges_df.empty:
        return

    df = edges_df.copy()
    if "aj_morph" not in df.columns:
        return

    # Group
    grp = df.groupby(group_cols + ["aj_morph"]).size().reset_index(name="count")

    # Pivot
    piv = grp.pivot_table(index=group_cols, columns="aj_morph", values="count", fill_value=0)
    piv = piv.reindex(columns=[c for c in AJ_CLASSES if c in piv.columns], fill_value=0)

    plt.figure(figsize=(10, 4), dpi=200)
    piv.plot(kind="bar", stacked=True)
    plt.title(title)
    plt.ylabel("# interfaces")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_polarity_rose_plot(
    polarity_df: pd.DataFrame,
    out_path: Union[str, Path],
    title: str,
    bins: int = 18,
) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    if polarity_df.empty or "polarity_angle_deg" not in polarity_df.columns:
        return

    angles = polarity_df["polarity_angle_deg"].dropna().values
    if angles.size == 0:
        return

    # Convert to radians
    theta = np.deg2rad(angles)

    plt.figure(figsize=(6, 6), dpi=200)
    ax = plt.subplot(111, projection="polar")
    ax.hist(theta, bins=bins)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -----------------------------
# Patch export and ML training
# -----------------------------


def export_interface_patches(
    edges_df: pd.DataFrame,
    out_dir: Union[str, Path],
    patch_px: int = 96,
    aj_channel_idx_col: str = "aj_channel_idx",
    aj_channel_override: Optional[int] = None,
) -> None:
    """Export interface patches for manual labeling.

    Requirements:
      edges_df must include columns:
        - image_path
        - cell_i, cell_j
        - iface_cy, iface_cx (interface centroid)

    Output:
      out_dir/patches/<image_id>__i_<cell_i>__j_<cell_j>.png
      out_dir/patch_index.csv

    Patches are extracted from the AJ channel.
    """
    out_dir = ensure_dir(out_dir)
    patch_dir = ensure_dir(out_dir / "patches")

    required = ["image_id", "image_path", "cell_i", "cell_j", "iface_cy", "iface_cx"]
    for c in required:
        if c not in edges_df.columns:
            die(f"export-patches requires column '{c}' in edges_df")

    rows_index = []

    for idx, r in maybe_tqdm(list(edges_df.iterrows()), total=len(edges_df), desc="export patches"):
        img_path = Path(str(r["image_path"]))
        if not img_path.exists():
            continue
        image_id = str(r["image_id"])
        i = int(r["cell_i"])
        j = int(r["cell_j"])
        cy = int(float(r["iface_cy"]))
        cx = int(float(r["iface_cx"]))

        arr, meta = read_image_any(img_path)
        cyx = squeeze_to_cyx(arr)        # Choose AJ channel
        if aj_channel_override is not None:
            aj_idx = int(aj_channel_override)
        elif aj_channel_idx_col in edges_df.columns and (not pd.isna(r.get(aj_channel_idx_col))):
            aj_idx = int(r[aj_channel_idx_col])
        else:
            aj_idx = 0
        if aj_idx < 0 or aj_idx >= cyx.shape[0]:
            aj_idx = 0
        aj = cyx[aj_idx]
        aj_n = normalize01(aj)

        H, W = aj_n.shape
        half = patch_px // 2
        y0 = max(0, cy - half)
        y1 = min(H, cy + half)
        x0 = max(0, cx - half)
        x1 = min(W, cx + half)
        patch = aj_n[y0:y1, x0:x1]

        # pad to fixed size
        pad_y0 = 0
        pad_y1 = patch_px - patch.shape[0]
        pad_x0 = 0
        pad_x1 = patch_px - patch.shape[1]
        if pad_y1 < 0:
            pad_y1 = 0
        if pad_x1 < 0:
            pad_x1 = 0
        patch = np.pad(patch, ((pad_y0, pad_y1), (pad_x0, pad_x1)), mode="constant")

        fname = f"{image_id}__i_{i}__j_{j}.png"
        fpath = patch_dir / fname
        plt.figure(figsize=(2, 2), dpi=200)
        plt.imshow(patch, cmap="gray")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(fpath)
        plt.close()

        rows_index.append(
            {
                "image_id": image_id,
                "cell_i": i,
                "cell_j": j,
                "patch_path": str(fpath),
                "aj_morph": "",  # to be filled by user
            }
        )

    pd.DataFrame(rows_index).to_csv(out_dir / "patch_index.csv", index=False)
    log(f"Exported {len(rows_index)} patches to {patch_dir}")


def train_ajmorph_classifier(
    edges_csv: Union[str, Path],
    labels_csv: Union[str, Path],
    out_model: Union[str, Path],
    test_size: float = 0.2,
    random_state: int = 0,
) -> None:
    if RandomForestClassifier is None or joblib is None:
        die("Training requires scikit-learn and joblib. pip install scikit-learn joblib")

    edges = pd.read_csv(edges_csv)
    labels = pd.read_csv(labels_csv)

    # Expect labels file has columns: image_id, cell_i, cell_j, aj_morph
    required = ["image_id", "cell_i", "cell_j", "aj_morph"]
    for c in required:
        if c not in labels.columns:
            die(f"labels_csv must contain column '{c}'")

    # Merge
    key = ["image_id", "cell_i", "cell_j"]
    df = edges.merge(labels[key + ["aj_morph"]], on=key, how="inner", suffixes=("", "_lab"))
    df = df[df["aj_morph"].notna()]
    df = df[df["aj_morph"].astype(str).str.len() > 0]

    if df.empty:
        die("No labeled rows after merging edges and labels")

    # Feature columns: all numeric aj_* features
    feat_cols = [c for c in df.columns if c.startswith("aj_") and c not in ("aj_morph",)]
    if not feat_cols:
        die("No feature columns found (expected columns starting with 'aj_')")

    X = df[feat_cols].values
    y = df["aj_morph"].astype(str).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    rep = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))

    log("\n" + rep)
    log("Confusion matrix (rows=true, cols=pred):")
    log(str(cm))

    out_model = Path(out_model)
    ensure_dir(out_model.parent)
    joblib.dump({"model": clf, "feature_cols": feat_cols, "classes": np.unique(y).tolist()}, out_model)
    write_text(rep, out_model.with_suffix(".report.txt"))
    log(f"Saved model to {out_model}")


def load_ajmorph_model(path: Union[str, Path]) -> Dict[str, Any]:
    if joblib is None:
        die("joblib is required to load a trained model. pip install joblib")
    obj = joblib.load(path)
    if not isinstance(obj, dict) or "model" not in obj:
        die("Invalid model file")
    return obj


def predict_ajmorph(df_edges: pd.DataFrame, model_obj: Dict[str, Any]) -> pd.Series:
    model = model_obj["model"]
    feat_cols = model_obj["feature_cols"]
    missing = [c for c in feat_cols if c not in df_edges.columns]
    if missing:
        die(f"Edges table missing required feature columns: {missing}")
    X = df_edges[feat_cols].values
    pred = model.predict(X)
    return pd.Series(pred, index=df_edges.index)


# -----------------------------
# Pipeline per image
# -----------------------------


@dataclass
class RunConfig:
    dataset_root: Path
    out_dir: Path
    aj_channel: str
    cell_channel: str
    nuc_channel: Optional[str] = None
    golgi_channel: Optional[str] = None
    flow_dir_deg: float = 0.0
    seg: SegmentationConfig = dataclasses.field(default_factory=SegmentationConfig)
    ajmorph_model_path: Optional[Path] = None
    interface_dilate: int = 2


def process_one_image(rec: ImageRecord, cfg: RunConfig) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], nx.MultiGraph]:
    """Process one image and write outputs to cfg.out_dir/image_id/.

    Returns:
      cells_df, edges_df, polarity_df, graph
    """
    image_out = ensure_dir(cfg.out_dir / rec.image_id)

    arr, meta = read_image_any(rec.path)
    cyx = squeeze_to_cyx(arr)
    channel_names = meta.get("channel_names", [])

    # Map channels
    aj_idx = channel_index_from_semantics(rec, cyx, channel_names, cfg.aj_channel, fallback=0)
    cell_idx = channel_index_from_semantics(rec, cyx, channel_names, cfg.cell_channel, fallback=aj_idx or 0)
    nuc_idx = None
    golgi_idx = None
    if cfg.nuc_channel:
        nuc_idx = channel_index_from_semantics(rec, cyx, channel_names, cfg.nuc_channel, fallback=None)
    if cfg.golgi_channel:
        golgi_idx = channel_index_from_semantics(rec, cyx, channel_names, cfg.golgi_channel, fallback=None)

    if aj_idx is None:
        die(f"Could not resolve AJ channel '{cfg.aj_channel}' for {rec.image_id}")
    if cell_idx is None:
        die(f"Could not resolve cell channel '{cfg.cell_channel}' for {rec.image_id}")

    aj = normalize01(cyx[int(aj_idx)])
    cell_img = normalize01(cyx[int(cell_idx)])

    # Segmentation
    labels = segment_cells(cell_img, cfg.seg)

    # Basic props
    cps = compute_cell_props(labels)
    cells_df = pd.DataFrame([dataclasses.asdict(cp) for cp in cps])
    cells_df = cells_df.rename(columns={"label": "cell_id"})

    # Interfaces
    ifaces = compute_interfaces(labels)

    iface_masks = [interface_mask_from_coords(labels.shape, iface.coords, dilate=cfg.interface_dilate) for iface in ifaces]
    thr = compute_interface_threshold(aj, iface_masks)

    edge_rows = []
    for iface, m in zip(ifaces, iface_masks):
        feat = extract_aj_features_for_interface(aj, m, thr)
        row = {
            "image_id": rec.image_id,
            "image_path": str(rec.path),
            "cell_i": iface.cell_i,
            "cell_j": iface.cell_j,
            "contact_px": iface.contact_px,
            "iface_cy": float(iface.coords[:, 0].mean()) if iface.coords.size else np.nan,
            "iface_cx": float(iface.coords[:, 1].mean()) if iface.coords.size else np.nan,
            "aj_threshold": thr,
            "aj_channel_idx": int(aj_idx),
            "cell_channel_idx": int(cell_idx),
            "nuc_channel_idx": int(nuc_idx) if nuc_idx is not None else None,
            "golgi_channel_idx": int(golgi_idx) if golgi_idx is not None else None,
            # prefix AJ features
            **{f"aj_{k}": v for k, v in feature_dict(feat).items()},
        }
        # label (heuristic for now)
        row["aj_morph"] = ajmorph_heuristic_label(feat)
        edge_rows.append(row)

    edges_df = pd.DataFrame(edge_rows)

    # apply ML model if provided
    if cfg.ajmorph_model_path is not None:
        model_obj = load_ajmorph_model(cfg.ajmorph_model_path)
        edges_df["aj_morph"] = predict_ajmorph(edges_df, model_obj)

    # polarity features (optional)
    polarity_df = None
    if nuc_idx is not None:
        nuc_img = normalize01(cyx[int(nuc_idx)])
        nuc_labels = segment_nuclei(nuc_img, cfg.seg)
        nuc_assign = assign_nucleus_to_cells(labels, nuc_labels)
        nuc_centroids = {cid: centroid_of_label(nuc_labels, nid) if nid is not None else None for cid, nid in nuc_assign.items()}

        if golgi_idx is not None:
            golgi_img = normalize01(cyx[int(golgi_idx)])
            golgi_centroids = estimate_golgi_centroid_per_cell(labels, golgi_img)
        else:
            golgi_centroids = {cid: None for cid in nuc_centroids.keys()}

        polarity_df = compute_polarity_features(cps, nuc_centroids, golgi_centroids, flow_dir_deg=cfg.flow_dir_deg)

        # Save nuclei QC
        save_segmentation_qc(nuc_img, nuc_labels, image_out / "qc_nuclei.png", title="Nuclei segmentation")

    # Build graph
    G = build_pi_graph(cps, edges_df)

    # Write outputs
    cells_df.to_csv(image_out / "cells.csv", index=False)
    edges_df.to_csv(image_out / "edges.csv", index=False)
    nx.write_graphml(G, image_out / "graph.graphml")
    write_json(graph_to_json(G), image_out / "graph.json")

    save_segmentation_qc(cell_img, labels, image_out / "qc_cells.png", title="Cell segmentation")
    save_graph_qc(aj, G, image_out / "qc_graph.png", title="AJ pi-graph (edges colored by aj_morph)")

    if polarity_df is not None:
        polarity_df.to_csv(image_out / "polarity.csv", index=False)
        save_polarity_rose_plot(polarity_df, image_out / "polarity_rose.png", title="Polarity angle distribution")

    # add condition metadata to edges/cells for global concat
    for col, val in [
        ("dataset", rec.dataset),
        ("shear_stress", rec.shear_stress),
        ("supplementation", rec.supplementation),
        ("replicate", rec.replicate),
        ("section", rec.section),
    ]:
        cells_df[col] = val
        edges_df[col] = val
        if polarity_df is not None:
            polarity_df[col] = val

    return cells_df, edges_df, polarity_df, G


def run_pipeline(
    manifest_csv: Union[str, Path],
    dataset_root: Union[str, Path],
    out_dir: Union[str, Path],
    aj_channel: str,
    cell_channel: str,
    nuc_channel: Optional[str],
    golgi_channel: Optional[str],
    flow_dir_deg: float,
    use_cellpose: bool,
    cellpose_diameter: Optional[float],
    ajmorph_model: Optional[Union[str, Path]],
    limit: Optional[int] = None,
) -> None:
    dataset_root = Path(dataset_root)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    recs = load_manifest_csv(manifest_csv, dataset_root)
    if limit is not None:
        recs = recs[: int(limit)]

    segcfg = SegmentationConfig(
        use_cellpose=bool(use_cellpose),
        cellpose_diameter=cellpose_diameter,
    )

    cfg = RunConfig(
        dataset_root=dataset_root,
        out_dir=out_dir,
        aj_channel=aj_channel,
        cell_channel=cell_channel,
        nuc_channel=nuc_channel,
        golgi_channel=golgi_channel,
        flow_dir_deg=float(flow_dir_deg),
        seg=segcfg,
        ajmorph_model_path=Path(ajmorph_model) if ajmorph_model else None,
    )

    all_cells = []
    all_edges = []
    all_polarity = []

    log(f"Processing {len(recs)} images...")

    for rec in maybe_tqdm(recs, total=len(recs), desc="run"):
        try:
            if not rec.path.exists():
                log(f"WARNING: missing file {rec.path} (skipping)")
                continue
            cells_df, edges_df, pol_df, G = process_one_image(rec, cfg)
            all_cells.append(cells_df)
            all_edges.append(edges_df)
            if pol_df is not None:
                all_polarity.append(pol_df)
        except Exception as e:
            log(f"FAILED on {rec.image_id}: {e}")
            traceback.print_exc()
            continue

    if not all_edges:
        die("No images processed successfully; check logs")

    df_cells = pd.concat(all_cells, ignore_index=True) if all_cells else pd.DataFrame()
    df_edges = pd.concat(all_edges, ignore_index=True) if all_edges else pd.DataFrame()
    df_cells.to_csv(out_dir / "all_cells.csv", index=False)
    df_edges.to_csv(out_dir / "all_edges.csv", index=False)

    if all_polarity:
        df_pol = pd.concat(all_polarity, ignore_index=True)
        df_pol.to_csv(out_dir / "all_polarity.csv", index=False)

    # Summary plots
    save_summary_barplot(
        df_edges,
        out_dir / "summary_ajmorph_by_shear.png",
        group_cols=["shear_stress"],
        title="AJ morphology counts by shear_stress",
    )

    if all_polarity:
        save_polarity_rose_plot(df_pol, out_dir / "polarity_rose_all.png", title="Polarity angles (all cells)")

    # Simple report
    report = {
        "created": now_slug(),
        "n_images": len(recs),
        "n_cells": int(df_cells.shape[0]) if not df_cells.empty else 0,
        "n_edges": int(df_edges.shape[0]) if not df_edges.empty else 0,
        "aj_classes": AJ_CLASSES,
        "config": {
            "aj_channel": aj_channel,
            "cell_channel": cell_channel,
            "nuc_channel": nuc_channel,
            "golgi_channel": golgi_channel,
            "flow_dir_deg": flow_dir_deg,
            "use_cellpose": use_cellpose,
            "cellpose_diameter": cellpose_diameter,
            "ajmorph_model": str(ajmorph_model) if ajmorph_model else None,
        },
    }
    write_json(report, out_dir / "run_report.json")

    log(f"Done. Outputs written to {out_dir}")


# -----------------------------
# Inspect helper
# -----------------------------


def inspect_images(
    manifest_csv: Union[str, Path],
    dataset_root: Union[str, Path],
    out_dir: Union[str, Path],
    n: int = 3,
) -> None:
    dataset_root = Path(dataset_root)
    out_dir = ensure_dir(out_dir)
    recs = load_manifest_csv(manifest_csv, dataset_root)

    rows = []
    for rec in recs[:n]:
        arr, meta = read_image_any(rec.path)
        cyx = squeeze_to_cyx(arr)
        ch_names = meta.get("channel_names", [])

        rows.append(
            {
                "image_id": rec.image_id,
                "path": str(rec.path),
                "raw_shape": meta.get("shape"),
                "cyx_shape": tuple(cyx.shape),
                "dtype": meta.get("dtype"),
                "is_ome": meta.get("is_ome"),
                "channel_names": ";".join(ch_names) if ch_names else "",
                "manifest_channel_1": rec.channel_1,
                "manifest_channel_2": rec.channel_2,
                "manifest_channel_3": rec.channel_3,
                "manifest_channel_4": rec.channel_4,
            }
        )

        # save per-channel thumbnails
        thumb_dir = ensure_dir(out_dir / rec.image_id)
        for c in range(min(cyx.shape[0], 8)):
            im = normalize01(cyx[c])
            plt.figure(figsize=(4, 4), dpi=150)
            plt.imshow(im, cmap="gray")
            plt.axis("off")
            title = f"{rec.image_id} channel {c}"
            if c < len(ch_names) and ch_names[c]:
                title += f" ({ch_names[c]})"
            plt.title(title)
            plt.tight_layout()
            plt.savefig(thumb_dir / f"ch{c}.png")
            plt.close()

    pd.DataFrame(rows).to_csv(out_dir / "inspect.csv", index=False)
    log(f"Wrote inspect outputs to {out_dir}")


# -----------------------------
# CLI
# -----------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="EndoPiGraph_AJmorph_v1_allinone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    # bia-info
    sp = sub.add_parser("bia-info", help="Fetch BioStudies /info JSON and write links")
    sp.add_argument("--acc", required=True, help="BioImage Archive accession (e.g., S-BIAD1540)")
    sp.add_argument("--out", required=True, help="Output directory")

    # make-manifest
    sp = sub.add_parser("make-manifest", help="Create a manifest CSV from dataset root")
    sp.add_argument("--dataset-root", required=True, help="Local dataset root (contains Files/)")
    sp.add_argument("--out", required=True, help="Output CSV path")
    sp.add_argument("--dataset-name", default="S-BIAD1540", help="Dataset name")
    sp.add_argument("--no-bia-json", action="store_true", help="Do not parse bioimage_archive.json, just scan")
    sp.add_argument("--glob", default="**/*.tif*", help="Glob pattern for scanning")

    # inspect
    sp = sub.add_parser("inspect", help="Inspect a few images and write channel thumbnails")
    sp.add_argument("--manifest", required=True, help="Manifest CSV")
    sp.add_argument("--dataset-root", required=True, help="Dataset root")
    sp.add_argument("--n", type=int, default=3, help="# images to inspect")
    sp.add_argument("--out", required=True, help="Output directory")

    # run
    sp = sub.add_parser("run", help="Run EndoPiGraph–AJmorph pipeline")
    sp.add_argument("--manifest", required=True, help="Manifest CSV")
    sp.add_argument("--dataset-root", required=True, help="Dataset root")
    sp.add_argument("--out", required=True, help="Output directory")
    sp.add_argument("--aj-channel", required=True, help="AJ channel semantic name or index (e.g., 'VE-cadherin' or '0')")
    sp.add_argument("--cell-channel", required=True, help="Cell segmentation channel semantic name or index")
    sp.add_argument("--nuc-channel", default=None, help="Nucleus channel semantic name or index (optional)")
    sp.add_argument("--golgi-channel", default=None, help="Golgi channel semantic name or index (optional)")
    sp.add_argument("--flow-dir-deg", type=float, default=0.0, help="Flow/cue direction in degrees (0 = +x)")
    sp.add_argument("--use-cellpose", type=int, default=0, help="1 to use Cellpose, 0 to use watershed fallback")
    sp.add_argument("--cellpose-diameter", type=float, default=None, help="Cellpose diameter (pixels)")
    sp.add_argument("--ajmorph-model", default=None, help="Optional path to trained ajmorph model (.joblib)")
    sp.add_argument("--limit", type=int, default=None, help="Limit # images for a quick run")

    # export patches
    sp = sub.add_parser("export-patches", help="Export interface patches for manual labeling")
    sp.add_argument("--edges-csv", required=True, help="Path to all_edges.csv")
    sp.add_argument("--out", required=True, help="Output dir")
    sp.add_argument("--patch-px", type=int, default=96, help="Patch size (pixels)")
    sp.add_argument("--aj-channel-idx", type=int, default=None, help="Override AJ channel index for patch extraction (otherwise uses aj_channel_idx in edges.csv if present)")

    # train
    sp = sub.add_parser("train", help="Train an AJ morphology classifier")
    sp.add_argument("--edges-csv", required=True, help="Path to all_edges.csv")
    sp.add_argument("--labels-csv", required=True, help="CSV with columns image_id,cell_i,cell_j,aj_morph")
    sp.add_argument("--out", required=True, help="Output model path (.joblib)")
    sp.add_argument("--test-size", type=float, default=0.2, help="Hold-out fraction")

    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    if args.cmd == "bia-info":
        bia_fetch_info(args.acc, args.out)
        log("Done")
        return

    if args.cmd == "make-manifest":
        build_manifest(
            dataset_root=args.dataset_root,
            out_csv=args.out,
            dataset_name=args.dataset_name,
            prefer_bia_json=not args.no_bia_json,
            glob_pattern=args.glob,
        )
        return

    if args.cmd == "inspect":
        inspect_images(args.manifest, args.dataset_root, args.out, n=int(args.n))
        return

    if args.cmd == "run":
        run_pipeline(
            manifest_csv=args.manifest,
            dataset_root=args.dataset_root,
            out_dir=args.out,
            aj_channel=args.aj_channel,
            cell_channel=args.cell_channel,
            nuc_channel=args.nuc_channel,
            golgi_channel=args.golgi_channel,
            flow_dir_deg=float(args.flow_dir_deg),
            use_cellpose=bool(int(args.use_cellpose)),
            cellpose_diameter=args.cellpose_diameter,
            ajmorph_model=args.ajmorph_model,
            limit=args.limit,
        )
        return

    if args.cmd == "export-patches":
        df = pd.read_csv(args.edges_csv)
        export_interface_patches(df, args.out, patch_px=int(args.patch_px), aj_channel_override=(args.aj_channel_idx if args.aj_channel_idx is not None else None))
        return

    if args.cmd == "train":
        train_ajmorph_classifier(
            edges_csv=args.edges_csv,
            labels_csv=args.labels_csv,
            out_model=args.out,
            test_size=float(args.test_size),
        )
        return

    die(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
