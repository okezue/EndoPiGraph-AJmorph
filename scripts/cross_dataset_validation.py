#!/usr/bin/env python3
"""
Cross-Dataset Validation Framework for EndoPiGraph-AJmorph

This script performs comprehensive validation across multiple datasets:
1. Lymphatic EC junction morphology (Zenodo 13880404)
2. HUVEC imaging screen (Scientific Data)
3. S-BIAD463 vascular remodeling (BioImage Archive)
4. S-BIAD1540 (existing reference dataset)

Tests include:
- Cross-dataset transfer (train on A, test on B)
- Leave-one-dataset-out validation
- Parameter stability analysis
- Comparison with Junction Mapper

Author: EndoPiGraph-AJmorph validation suite
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import hashlib

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import tifffile

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from endopigraph.pipeline import process_one_image, PipelineError
from endopigraph.ajmorph import compute_threshold, compute_interface_features, infer_ajmorph_label_heuristic
from endopigraph.segmentation import segment_cells
from endopigraph.interfaces import extract_interfaces
from endopigraph.io import read_image
from endopigraph.utils import ensure_dir


# ============================================================================
# Dataset Configuration
# ============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a validation dataset."""
    name: str
    data_dir: Path
    channel_mapping: Dict[str, int]  # e.g., {"VE-cadherin": 0, "DAPI": 1}
    pixel_size_um: Optional[float] = None
    cell_type: str = "endothelial"
    source: str = "unknown"
    notes: str = ""

    # Processing parameters (can be overridden for stability tests)
    cellpose_diameter: int = 30
    min_contact_px: int = 10
    threshold_method: str = "otsu"


def get_dataset_configs(data_root: Path) -> Dict[str, DatasetConfig]:
    """Return configurations for all validation datasets."""
    return {
        "sbiad1540": DatasetConfig(
            name="S-BIAD1540",
            data_dir=data_root / "S-BIAD1540",
            channel_mapping={"VE-cadherin": 2, "DAPI": 1, "GM130": 0},
            pixel_size_um=0.325,
            cell_type="HUVEC (EGM2)",
            source="BioImage Archive",
            notes="Reference dataset with shear stress conditions",
        ),
        "lymphatic_ec": DatasetConfig(
            name="Lymphatic EC",
            data_dir=data_root / "lymphatic_ec",
            channel_mapping={"VE-cadherin": 1, "LYVE1": 0},  # C2=VE-cadherin, C1=LYVE1
            pixel_size_um=None,  # Need to extract from metadata
            cell_type="Lymphatic EC (dermal capillary)",
            source="Zenodo 13880404",
            notes="Button/curvilinear/zipper junction types",
        ),
        "huvec_screen": DatasetConfig(
            name="HUVEC Screen",
            data_dir=data_root / "huvec_screen",
            channel_mapping={"VE-cadherin": 1, "DAPI": 0, "ch2": 2, "ch3": 3},  # Assumed mapping
            pixel_size_um=None,
            cell_type="HUVEC",
            source="High-content screen",
            notes="siRNA screen with VE-cadherin staining",
        ),
        "sbiad463": DatasetConfig(
            name="S-BIAD463",
            data_dir=data_root / "sbiad463",
            channel_mapping={"ch0": 0, "ch1": 1, "ch2": 2},  # Generic 3-channel
            pixel_size_um=None,
            cell_type="HEV (lymph node)",
            source="BioImage Archive",
            notes="HEV vascular remodeling in tumor-draining LN",
        ),
    }


# ============================================================================
# Data Loaders
# ============================================================================

def load_lymphatic_image(image_folder: Path) -> Tuple[np.ndarray, List[str]]:
    """Load lymphatic EC image by combining C1 and C2 channels."""
    c1_files = list(image_folder.glob("C1-*.tif"))
    c2_files = list(image_folder.glob("C2-*.tif"))

    if not c1_files or not c2_files:
        raise FileNotFoundError(f"Cannot find C1/C2 files in {image_folder}")

    # Filter out RGB files (-1 variants) and B variants, prefer grayscale
    def select_best_file(files):
        # Prefer files without "-1" (RGB) and without "B" in name
        candidates = [f for f in files if '-1' not in f.stem and 'B' not in f.stem]
        if not candidates:
            candidates = [f for f in files if '-1' not in f.stem]  # Allow B but not RGB
        if not candidates:
            candidates = files  # Fallback to any
        return sorted(candidates)[0]

    c1_file = select_best_file(c1_files)
    c2_file = select_best_file(c2_files)

    c1 = tifffile.imread(str(c1_file)).astype(np.float32)
    c2 = tifffile.imread(str(c2_file)).astype(np.float32)

    # Handle RGB images by taking first channel or converting to grayscale
    if c1.ndim == 3 and c1.shape[-1] == 3:
        c1 = c1[:, :, 0]  # Take first channel
    if c2.ndim == 3 and c2.shape[-1] == 3:
        c2 = c2[:, :, 0]  # Take first channel

    # Stack into (C, H, W)
    if c1.ndim == 2 and c2.ndim == 2:
        arr = np.stack([c1, c2], axis=0)
    else:
        raise ValueError(f"Unexpected shapes: C1={c1.shape}, C2={c2.shape}")

    return arr, ["LYVE1", "VE-cadherin"]


def load_zarr_image(zarr_path: Path) -> Tuple[np.ndarray, List[str]]:
    """Load image from zarr format (S-BIAD463)."""
    import zarr
    z = zarr.open(str(zarr_path), 'r')
    # Access highest resolution: 0/0
    arr = z['0']['0'][:]
    # Shape is (T, C, Z, Y, X) -> extract (C, Y, X)
    if arr.ndim == 5:
        arr = arr[0, :, 0, :, :]  # First timepoint, all channels, first Z
    channel_names = [f"ch{i}" for i in range(arr.shape[0])]
    return arr.astype(np.float32), channel_names


def discover_images(config: DatasetConfig, max_images: Optional[int] = None) -> List[Dict]:
    """Discover images in a dataset and return list of image info dicts."""
    images = []

    if config.name == "Lymphatic EC":
        # Structure: timepoint/animal/image folders
        for timepoint in ["3w", "5w", "25w"]:
            tp_dir = config.data_dir / timepoint
            if not tp_dir.exists():
                continue
            for animal_dir in tp_dir.iterdir():
                if not animal_dir.is_dir() or animal_dir.name.startswith('.'):
                    continue
                for img_dir in animal_dir.iterdir():
                    if not img_dir.is_dir() or not img_dir.name.startswith('image'):
                        continue
                    # Check for C1/C2 files
                    if list(img_dir.glob("C1-*.tif")) and list(img_dir.glob("C2-*.tif")):
                        image_id = f"{timepoint}_{animal_dir.name}_{img_dir.name}"
                        images.append({
                            "image_id": image_id,
                            "path": img_dir,
                            "timepoint": timepoint,
                            "animal": animal_dir.name,
                            "condition": timepoint,
                            "loader": "lymphatic",
                        })
        # Also check tissue-specific folders
        for timepoint in ["25w"]:
            tp_dir = config.data_dir / timepoint
            for tissue in ["diaphragm", "trachea"]:
                tissue_dir = tp_dir / tissue
                if not tissue_dir.exists():
                    continue
                for img_dir in tissue_dir.iterdir():
                    if not img_dir.is_dir() or not img_dir.name.startswith('image'):
                        continue
                    if list(img_dir.glob("C1-*.tif")) and list(img_dir.glob("C2-*.tif")):
                        image_id = f"{timepoint}_{tissue}_{img_dir.name}"
                        images.append({
                            "image_id": image_id,
                            "path": img_dir,
                            "timepoint": timepoint,
                            "tissue": tissue,
                            "condition": f"{timepoint}_{tissue}",
                            "loader": "lymphatic",
                        })

    elif config.name == "HUVEC Screen":
        # Structure: replicate/plate/images
        for replicate_dir in config.data_dir.iterdir():
            if not replicate_dir.is_dir():
                continue
            replicate_name = replicate_dir.name  # A1, A2, G1, G2
            for plate_dir in replicate_dir.iterdir():
                if not plate_dir.is_dir():
                    continue
                for tif_file in plate_dir.glob("*.tif"):
                    # Parse well position from filename (e.g., 004007000_Field_2.tif)
                    image_id = f"{replicate_name}_{tif_file.stem}"
                    images.append({
                        "image_id": image_id,
                        "path": tif_file,
                        "replicate": replicate_name,
                        "condition": replicate_name[:1],  # A or G
                        "loader": "tiff",
                    })

    elif config.name == "S-BIAD463":
        # Structure: uuid folders with .zarr inside
        for uuid_dir in config.data_dir.iterdir():
            if not uuid_dir.is_dir() or uuid_dir.name.endswith('.png'):
                continue
            zarr_path = uuid_dir / f"{uuid_dir.name}.zarr"
            if zarr_path.exists():
                images.append({
                    "image_id": uuid_dir.name,
                    "path": zarr_path,
                    "condition": "HEV",
                    "loader": "zarr",
                })

    elif config.name == "S-BIAD1540":
        # Use existing manifest if available
        manifest_path = config.data_dir / "manifest.csv"
        if manifest_path.exists():
            df = pd.read_csv(manifest_path)
            for _, row in df.iterrows():
                images.append({
                    "image_id": row["image_id"],
                    "path": Path(row["path"]),
                    "condition": row.get("shear_stress", "unknown"),
                    "loader": "tiff",
                })
        else:
            # Discover TIFFs
            for tif_file in config.data_dir.rglob("*.tif"):
                image_id = tif_file.stem
                images.append({
                    "image_id": image_id,
                    "path": tif_file,
                    "condition": "unknown",
                    "loader": "tiff",
                })

    if max_images:
        images = images[:max_images]

    return images


# ============================================================================
# Processing Functions
# ============================================================================

def process_single_image(
    image_info: Dict,
    config: DatasetConfig,
    output_dir: Path,
    params: Optional[Dict] = None,
    use_cellpose: bool = False,
) -> Optional[pd.DataFrame]:
    """Process a single image and return edge features DataFrame."""
    params = params or {}

    try:
        # Load image based on loader type
        if image_info["loader"] == "lymphatic":
            arr, channel_names = load_lymphatic_image(image_info["path"])
        elif image_info["loader"] == "zarr":
            arr, channel_names = load_zarr_image(image_info["path"])
        else:  # tiff
            arr, channel_names = read_image(image_info["path"])

        # Get VE-cadherin channel
        ve_cad_idx = config.channel_mapping.get("VE-cadherin", 0)
        if ve_cad_idx >= arr.shape[0]:
            ve_cad_idx = 0  # Fallback to first channel

        # Segment cells - use watershed by default (faster), cellpose optional
        if use_cellpose:
            seg_cfg = {
                "method": "cellpose",
                "cellpose": {
                    "model_type": "cyto2",
                    "diameter": params.get("diameter", config.cellpose_diameter),
                },
            }
        else:
            # Watershed is faster for validation
            seg_cfg = {
                "method": "watershed",
                "watershed": {
                    "nuclei": {"channel_index": min(1, arr.shape[0]-1)},  # Try DAPI-like channel
                    "membrane": {"channel_index": ve_cad_idx},
                },
            }
        labels = segment_cells(arr, channel_names, seg_cfg)

        if labels.max() < 2:
            print(f"  Warning: Only {labels.max()} cells found in {image_info['image_id']}")
            return None

        # Extract interfaces
        iface = extract_interfaces(labels)
        edges_df = iface.edges.copy()

        min_contact = params.get("min_contact_px", config.min_contact_px)
        edges_df = edges_df[edges_df["contact_px"] >= min_contact].reset_index(drop=True)

        if len(edges_df) == 0:
            print(f"  Warning: No edges found in {image_info['image_id']}")
            return None

        # Compute AJ features
        marker = arr[ve_cad_idx].astype(np.float32)
        boundary_values = marker[iface.all_boundary_mask]

        threshold_method = params.get("threshold_method", config.threshold_method)
        thr = compute_threshold(boundary_values, threshold_method)

        dilate_px = params.get("dilate_px", 2)

        feats_rows = []
        ajmorph_labels = []

        from endopigraph.interfaces import interface_mask_from_coords

        for _, erow in edges_df.iterrows():
            i, j = int(erow["cell_i"]), int(erow["cell_j"])
            coords = iface.boundary_coords.get((min(i, j), max(i, j)))
            if coords is None:
                coords = np.zeros((0, 2), dtype=int)
            mask = interface_mask_from_coords(coords, labels.shape, dilate_px=dilate_px)
            feats = compute_interface_features(marker, mask, thr)
            feats_rows.append(feats)
            ajmorph_labels.append(infer_ajmorph_label_heuristic(feats))

        feats_df = pd.DataFrame(feats_rows)
        edges_df = pd.concat([edges_df, feats_df], axis=1)
        edges_df["aj_morph_label"] = ajmorph_labels
        edges_df["image_id"] = image_info["image_id"]
        edges_df["dataset"] = config.name
        edges_df["condition"] = image_info.get("condition", "unknown")

        return edges_df

    except Exception as e:
        print(f"  Error processing {image_info['image_id']}: {e}")
        return None


def process_dataset(
    config: DatasetConfig,
    output_dir: Path,
    max_images: Optional[int] = None,
    params: Optional[Dict] = None,
    use_cellpose: bool = False,
) -> pd.DataFrame:
    """Process all images in a dataset and return combined edge features."""
    print(f"\nProcessing dataset: {config.name}")
    print(f"  Data dir: {config.data_dir}")
    print(f"  Segmentation: {'cellpose' if use_cellpose else 'watershed (fast)'}")

    images = discover_images(config, max_images)
    print(f"  Found {len(images)} images")

    if not images:
        return pd.DataFrame()

    all_edges = []
    for i, img_info in enumerate(images):
        print(f"  [{i+1}/{len(images)}] {img_info['image_id']}")
        edges_df = process_single_image(img_info, config, output_dir, params, use_cellpose)
        if edges_df is not None and len(edges_df) > 0:
            all_edges.append(edges_df)

    if all_edges:
        combined = pd.concat(all_edges, ignore_index=True)
        print(f"  Total edges: {len(combined)}")
        return combined
    return pd.DataFrame()


# ============================================================================
# Cross-Dataset Transfer Tests
# ============================================================================

def train_test_transfer(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "aj_morph_label",
) -> Dict:
    """Train on one dataset, test on another."""
    # Remove rows with missing features or labels
    train_df = train_df.dropna(subset=feature_cols + [target_col])
    test_df = test_df.dropna(subset=feature_cols + [target_col])

    if len(train_df) < 10 or len(test_df) < 10:
        return {"error": "Insufficient data", "train_n": len(train_df), "test_n": len(test_df)}

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    return {
        "train_dataset": train_df["dataset"].iloc[0] if "dataset" in train_df.columns else "unknown",
        "test_dataset": test_df["dataset"].iloc[0] if "dataset" in test_df.columns else "unknown",
        "train_n": len(train_df),
        "test_n": len(test_df),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "classes": list(np.unique(np.concatenate([y_train, y_test]))),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def leave_one_dataset_out(
    all_data: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    target_col: str = "aj_morph_label",
) -> List[Dict]:
    """Leave-one-dataset-out cross-validation."""
    results = []
    dataset_names = list(all_data.keys())

    for held_out in dataset_names:
        # Combine all except held_out for training
        train_dfs = [df for name, df in all_data.items() if name != held_out and len(df) > 0]
        test_df = all_data[held_out]

        if not train_dfs or len(test_df) == 0:
            continue

        train_df = pd.concat(train_dfs, ignore_index=True)

        result = train_test_transfer(train_df, test_df, feature_cols, target_col)
        result["held_out_dataset"] = held_out
        result["training_datasets"] = [n for n in dataset_names if n != held_out]
        results.append(result)

    return results


# ============================================================================
# Stability Analysis
# ============================================================================

def compute_effect_sizes(
    df: pd.DataFrame,
    group_col: str,
    metric_cols: List[str],
) -> pd.DataFrame:
    """Compute effect sizes (Cohen's d, rank-biserial r) for metrics across groups."""
    results = []
    groups = df[group_col].unique()

    if len(groups) < 2:
        return pd.DataFrame()

    for i, g1 in enumerate(groups):
        for g2 in groups[i+1:]:
            d1 = df[df[group_col] == g1]
            d2 = df[df[group_col] == g2]

            for col in metric_cols:
                v1 = d1[col].dropna().values
                v2 = d2[col].dropna().values

                if len(v1) < 2 or len(v2) < 2:
                    continue

                # Cohen's d
                pooled_std = np.sqrt(((len(v1)-1)*v1.std()**2 + (len(v2)-1)*v2.std()**2) / (len(v1)+len(v2)-2))
                cohens_d = (v1.mean() - v2.mean()) / pooled_std if pooled_std > 0 else 0

                # Mann-Whitney U and rank-biserial r
                try:
                    stat, pval = stats.mannwhitneyu(v1, v2, alternative='two-sided')
                    # Rank-biserial correlation
                    r = 1 - (2*stat) / (len(v1) * len(v2))
                except:
                    stat, pval, r = np.nan, np.nan, np.nan

                results.append({
                    "group1": g1,
                    "group2": g2,
                    "metric": col,
                    "mean1": v1.mean(),
                    "mean2": v2.mean(),
                    "std1": v1.std(),
                    "std2": v2.std(),
                    "n1": len(v1),
                    "n2": len(v2),
                    "cohens_d": cohens_d,
                    "mann_whitney_U": stat,
                    "p_value": pval,
                    "rank_biserial_r": r,
                })

    return pd.DataFrame(results)


def stability_test_parameters(
    config: DatasetConfig,
    output_dir: Path,
    sample_images: int = 5,
) -> pd.DataFrame:
    """Test stability under different parameter settings."""
    print(f"\nRunning stability tests for {config.name}")

    images = discover_images(config, sample_images)
    if not images:
        return pd.DataFrame()

    # Parameter variations to test
    param_sets = [
        {"name": "baseline", "diameter": 30, "threshold_method": "otsu", "dilate_px": 2},
        {"name": "small_cells", "diameter": 20, "threshold_method": "otsu", "dilate_px": 2},
        {"name": "large_cells", "diameter": 40, "threshold_method": "otsu", "dilate_px": 2},
        {"name": "percentile_95", "diameter": 30, "threshold_method": "percentile:95", "dilate_px": 2},
        {"name": "percentile_90", "diameter": 30, "threshold_method": "percentile:90", "dilate_px": 2},
        {"name": "dilate_1", "diameter": 30, "threshold_method": "otsu", "dilate_px": 1},
        {"name": "dilate_3", "diameter": 30, "threshold_method": "otsu", "dilate_px": 3},
    ]

    all_results = []

    for ps in param_sets:
        params = {k: v for k, v in ps.items() if k != "name"}
        print(f"  Testing params: {ps['name']}")

        for img_info in images:
            edges_df = process_single_image(img_info, config, output_dir, params)
            if edges_df is not None and len(edges_df) > 0:
                edges_df["param_set"] = ps["name"]
                all_results.append(edges_df)

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


def intensity_perturbation_test(
    config: DatasetConfig,
    output_dir: Path,
    sample_images: int = 3,
) -> pd.DataFrame:
    """Test robustness to intensity scaling and noise."""
    print(f"\nRunning intensity perturbation tests for {config.name}")

    images = discover_images(config, sample_images)
    if not images:
        return pd.DataFrame()

    # Perturbation functions
    def scale_intensity(arr, factor):
        return np.clip(arr * factor, 0, arr.max()).astype(arr.dtype)

    def add_gaussian_noise(arr, sigma):
        noise = np.random.normal(0, sigma * arr.std(), arr.shape)
        return np.clip(arr + noise, 0, arr.max()).astype(arr.dtype)

    def gaussian_blur(arr, sigma):
        from scipy.ndimage import gaussian_filter
        return np.stack([gaussian_filter(arr[c], sigma) for c in range(arr.shape[0])])

    perturbations = [
        ("original", lambda x: x),
        ("scale_0.8", lambda x: scale_intensity(x, 0.8)),
        ("scale_1.2", lambda x: scale_intensity(x, 1.2)),
        ("noise_low", lambda x: add_gaussian_noise(x, 0.05)),
        ("noise_high", lambda x: add_gaussian_noise(x, 0.1)),
        ("blur_1px", lambda x: gaussian_blur(x, 1.0)),
        ("blur_2px", lambda x: gaussian_blur(x, 2.0)),
    ]

    all_results = []

    for img_info in images:
        try:
            # Load original image
            if img_info["loader"] == "lymphatic":
                arr_orig, channel_names = load_lymphatic_image(img_info["path"])
            elif img_info["loader"] == "zarr":
                arr_orig, channel_names = load_zarr_image(img_info["path"])
            else:
                arr_orig, channel_names = read_image(img_info["path"])

            for pert_name, pert_fn in perturbations:
                print(f"  {img_info['image_id']} - {pert_name}")
                arr = pert_fn(arr_orig.copy())

                # Process with perturbed image
                ve_cad_idx = config.channel_mapping.get("VE-cadherin", 0)
                if ve_cad_idx >= arr.shape[0]:
                    ve_cad_idx = 0

                seg_cfg = {
                    "method": "cellpose",
                    "cellpose": {"model_type": "cyto2", "diameter": config.cellpose_diameter},
                }
                labels = segment_cells(arr, channel_names, seg_cfg)

                if labels.max() < 2:
                    continue

                iface = extract_interfaces(labels)
                edges_df = iface.edges.copy()
                edges_df = edges_df[edges_df["contact_px"] >= config.min_contact_px].reset_index(drop=True)

                if len(edges_df) == 0:
                    continue

                marker = arr[ve_cad_idx].astype(np.float32)
                boundary_values = marker[iface.all_boundary_mask]
                thr = compute_threshold(boundary_values, config.threshold_method)

                from endopigraph.interfaces import interface_mask_from_coords

                feats_rows = []
                for _, erow in edges_df.iterrows():
                    i, j = int(erow["cell_i"]), int(erow["cell_j"])
                    coords = iface.boundary_coords.get((min(i, j), max(i, j)))
                    if coords is None:
                        coords = np.zeros((0, 2), dtype=int)
                    mask = interface_mask_from_coords(coords, labels.shape, dilate_px=2)
                    feats = compute_interface_features(marker, mask, thr)
                    feats_rows.append(feats)

                feats_df = pd.DataFrame(feats_rows)
                edges_df = pd.concat([edges_df, feats_df], axis=1)
                edges_df["image_id"] = img_info["image_id"]
                edges_df["perturbation"] = pert_name
                all_results.append(edges_df)

        except Exception as e:
            print(f"  Error: {e}")
            continue

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


# ============================================================================
# Main Validation Runner
# ============================================================================

def run_full_validation(
    data_root: Path,
    output_dir: Path,
    max_images_per_dataset: int = 20,
    use_cellpose: bool = False,
) -> Dict:
    """Run the complete cross-dataset validation suite."""
    ensure_dir(output_dir)

    configs = get_dataset_configs(data_root)
    results = {
        "datasets_processed": {},
        "cross_dataset_transfer": [],
        "leave_one_out": [],
        "stability_tests": {},
        "effect_sizes": {},
    }

    # Feature columns for classification
    feature_cols = [
        "occupancy", "mean_intensity", "max_intensity", "std_intensity",
        "cluster_count", "cluster_density", "skeleton_len", "thickness_proxy",
    ]

    # 1. Process all datasets
    print("=" * 60)
    print("PHASE 1: Processing Datasets")
    print("=" * 60)

    all_data = {}
    for name, config in configs.items():
        if not config.data_dir.exists():
            print(f"Skipping {name}: data directory not found")
            continue

        edges_df = process_dataset(config, output_dir, max_images_per_dataset, use_cellpose=use_cellpose)
        if len(edges_df) > 0:
            all_data[name] = edges_df
            edges_df.to_csv(output_dir / f"{name}_edges.csv", index=False)
            results["datasets_processed"][name] = {
                "n_images": edges_df["image_id"].nunique(),
                "n_edges": len(edges_df),
                "conditions": edges_df["condition"].unique().tolist() if "condition" in edges_df.columns else [],
            }

    # 2. Cross-dataset transfer tests
    print("\n" + "=" * 60)
    print("PHASE 2: Cross-Dataset Transfer Tests")
    print("=" * 60)

    dataset_names = list(all_data.keys())
    for i, train_name in enumerate(dataset_names):
        for test_name in dataset_names[i+1:]:
            print(f"\nTrain: {train_name} -> Test: {test_name}")
            result = train_test_transfer(
                all_data[train_name], all_data[test_name],
                [c for c in feature_cols if c in all_data[train_name].columns and c in all_data[test_name].columns],
            )
            results["cross_dataset_transfer"].append(result)

            # Reverse direction
            print(f"Train: {test_name} -> Test: {train_name}")
            result_rev = train_test_transfer(
                all_data[test_name], all_data[train_name],
                [c for c in feature_cols if c in all_data[train_name].columns and c in all_data[test_name].columns],
            )
            results["cross_dataset_transfer"].append(result_rev)

    # 3. Leave-one-dataset-out validation
    if len(all_data) >= 3:
        print("\n" + "=" * 60)
        print("PHASE 3: Leave-One-Dataset-Out Validation")
        print("=" * 60)

        common_cols = set(feature_cols)
        for df in all_data.values():
            common_cols &= set(df.columns)
        common_cols = list(common_cols)

        if common_cols:
            results["leave_one_out"] = leave_one_dataset_out(all_data, common_cols)

    # 4. Stability tests (on a sample)
    print("\n" + "=" * 60)
    print("PHASE 4: Stability Analysis")
    print("=" * 60)

    for name, config in configs.items():
        if name not in all_data:
            continue

        # Parameter stability
        param_df = stability_test_parameters(config, output_dir, sample_images=3)
        if len(param_df) > 0:
            results["stability_tests"][f"{name}_params"] = param_df.to_dict(orient="records")

            # Compute effect sizes across parameter settings
            metric_cols = [c for c in feature_cols if c in param_df.columns]
            effect_df = compute_effect_sizes(param_df, "param_set", metric_cols)
            if len(effect_df) > 0:
                results["effect_sizes"][f"{name}_params"] = effect_df.to_dict(orient="records")

        # Intensity perturbation
        pert_df = intensity_perturbation_test(config, output_dir, sample_images=2)
        if len(pert_df) > 0:
            results["stability_tests"][f"{name}_perturbation"] = pert_df.to_dict(orient="records")

            metric_cols = [c for c in feature_cols if c in pert_df.columns]
            effect_df = compute_effect_sizes(pert_df, "perturbation", metric_cols)
            if len(effect_df) > 0:
                results["effect_sizes"][f"{name}_perturbation"] = effect_df.to_dict(orient="records")

    # 5. Save results
    results_path = output_dir / "cross_dataset_validation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {results_path}")

    return results


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cross-dataset validation for EndoPiGraph-AJmorph")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parent.parent / "data",
                        help="Root directory containing datasets")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent.parent / "runs" / "cross_dataset_validation",
                        help="Output directory for results")
    parser.add_argument("--max-images", type=int, default=20,
                        help="Maximum images to process per dataset")
    parser.add_argument("--use-cellpose", action="store_true",
                        help="Use Cellpose instead of watershed (slower but more accurate)")

    args = parser.parse_args()

    results = run_full_validation(args.data_root, args.output_dir, args.max_images, args.use_cellpose)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nDatasets processed:")
    for name, info in results["datasets_processed"].items():
        print(f"  {name}: {info['n_images']} images, {info['n_edges']} edges")

    print("\nCross-dataset transfer accuracy:")
    for r in results["cross_dataset_transfer"]:
        if "error" not in r:
            print(f"  {r['train_dataset']} -> {r['test_dataset']}: {r['accuracy']:.3f} (F1={r['f1_macro']:.3f})")

    if results["leave_one_out"]:
        print("\nLeave-one-out validation:")
        for r in results["leave_one_out"]:
            if "error" not in r:
                print(f"  Held out {r['held_out_dataset']}: {r['accuracy']:.3f} (F1={r['f1_macro']:.3f})")
