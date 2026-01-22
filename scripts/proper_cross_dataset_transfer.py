#!/usr/bin/env python3
"""
Proper Cross-Dataset Transfer Validation for EndoPiGraph-AJmorph

This script performs VALID cross-dataset transfer tests by:
1. Using EXPERIMENTAL CONDITIONS as labels (not heuristic morphology classes)
2. Training on one dataset, testing on another
3. Leave-one-dataset-out validation

Why this is valid:
- Experimental conditions (static vs shear, timepoint, treatment) are ground truth
- We test if junction FEATURES predict experimental conditions
- Transfer = does this feature→condition mapping generalize across datasets?

This avoids the circular reasoning of using heuristic labels on both sides.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
import tifffile

from endopigraph.interfaces import extract_interfaces, interface_mask_from_coords
from endopigraph.ajmorph import interface_marker_features, compute_threshold


# =============================================================================
# Data Loading with Condition Labels
# =============================================================================

def load_sbiad1540_with_conditions(data_dir: Path, max_images: int = None) -> pd.DataFrame:
    """
    Load S-BIAD1540 with experimental condition labels.

    Conditions based on filename patterns:
    - "static" → static culture
    - "dyn" or "flow" → shear stress / flow
    """
    images_dir = data_dir / "images_egm2"
    if not images_dir.exists():
        images_dir = data_dir

    tiffs = list(images_dir.glob("*.tif"))
    if max_images:
        tiffs = tiffs[:max_images]

    all_edges = []

    for tiff_path in tiffs:
        fname = tiff_path.stem.lower()

        # Determine condition from filename
        if "static" in fname:
            condition = "static"
        elif "dyn" in fname or "flow" in fname:
            condition = "flow"
        else:
            condition = "unknown"

        if condition == "unknown":
            continue

        try:
            raw = tifffile.imread(tiff_path)
            if raw.ndim == 3:
                marker = raw[0].astype(np.float32)  # VE-cadherin channel
            else:
                marker = raw.astype(np.float32)

            # Simple threshold-based segmentation for speed
            from skimage.filters import threshold_otsu
            from skimage.measure import label
            from skimage.morphology import binary_opening, disk

            thresh = threshold_otsu(marker)
            binary = marker > thresh * 0.5
            binary = binary_opening(binary, disk(2))
            labels = label(binary)

            if labels.max() < 2:
                continue

            iface = extract_interfaces(labels)
            edges_df = iface.edges.copy()
            edges_df = edges_df[edges_df["contact_px"] >= 10].reset_index(drop=True)

            if len(edges_df) == 0:
                continue

            boundary_vals = marker[iface.all_boundary_mask]
            thr = compute_threshold(boundary_vals, "otsu")

            feats_rows = []
            for _, erow in edges_df.iterrows():
                i, j = int(erow["cell_i"]), int(erow["cell_j"])
                coords = iface.boundary_coords.get((min(i, j), max(i, j)))
                if coords is None or len(coords) == 0:
                    continue
                mask = interface_mask_from_coords(coords, labels.shape, dilate_px=2)
                feats = interface_marker_features(marker, mask, thr)
                feats["condition"] = condition
                feats["dataset"] = "S-BIAD1540"
                feats["image_id"] = tiff_path.stem
                feats_rows.append(feats)

            all_edges.extend(feats_rows)
            print(f"  {tiff_path.name}: {len(feats_rows)} edges ({condition})")

        except Exception as e:
            print(f"  Error {tiff_path.name}: {e}")
            continue

    return pd.DataFrame(all_edges)


def load_lymphatic_with_conditions(data_dir: Path, max_images: int = None) -> pd.DataFrame:
    """
    Load lymphatic EC dataset with timepoint conditions.

    Conditions: 3w, 5w, 25w (weeks of age - developmental timepoints)
    """
    all_edges = []
    image_count = 0

    for timepoint in ["3w", "5w", "25w"]:
        tp_dir = data_dir / timepoint
        if not tp_dir.exists():
            continue

        # Find all image directories
        for animal_dir in tp_dir.iterdir():
            if not animal_dir.is_dir() or animal_dir.name.startswith('.'):
                continue

            for img_dir in animal_dir.iterdir():
                if not img_dir.is_dir():
                    continue

                if max_images and image_count >= max_images:
                    break

                # Look for C2 (VE-cadherin) files
                c2_files = list(img_dir.glob("C2-*.tif"))
                if not c2_files:
                    continue

                # Select non-RGB file
                c2_file = None
                for f in c2_files:
                    if '-1' not in f.stem:  # Avoid RGB versions
                        c2_file = f
                        break
                if c2_file is None:
                    c2_file = c2_files[0]

                try:
                    marker = tifffile.imread(c2_file).astype(np.float32)
                    if marker.ndim == 3:
                        marker = marker[:, :, 0] if marker.shape[-1] == 3 else marker[0]

                    from skimage.filters import threshold_otsu
                    from skimage.measure import label
                    from skimage.morphology import binary_opening, disk

                    thresh = threshold_otsu(marker)
                    binary = marker > thresh * 0.5
                    binary = binary_opening(binary, disk(2))
                    labels = label(binary)

                    if labels.max() < 2:
                        continue

                    iface = extract_interfaces(labels)
                    edges_df = iface.edges.copy()
                    edges_df = edges_df[edges_df["contact_px"] >= 10].reset_index(drop=True)

                    if len(edges_df) == 0:
                        continue

                    boundary_vals = marker[iface.all_boundary_mask]
                    thr = compute_threshold(boundary_vals, "otsu")

                    feats_rows = []
                    for _, erow in edges_df.iterrows():
                        i, j = int(erow["cell_i"]), int(erow["cell_j"])
                        coords = iface.boundary_coords.get((min(i, j), max(i, j)))
                        if coords is None or len(coords) == 0:
                            continue
                        mask = interface_mask_from_coords(coords, labels.shape, dilate_px=2)
                        feats = interface_marker_features(marker, mask, thr)
                        feats["condition"] = timepoint
                        feats["dataset"] = "Lymphatic_EC"
                        feats["image_id"] = f"{timepoint}_{animal_dir.name}_{img_dir.name}"
                        feats_rows.append(feats)

                    all_edges.extend(feats_rows)
                    image_count += 1
                    print(f"  {timepoint}/{animal_dir.name}/{img_dir.name}: {len(feats_rows)} edges")

                except Exception as e:
                    print(f"  Error: {e}")
                    continue

            if max_images and image_count >= max_images:
                break

    return pd.DataFrame(all_edges)


def load_benchmark_with_conditions(benchmark_dir: Path) -> pd.DataFrame:
    """
    Load benchmark dataset with animal/timepoint conditions.
    """
    manifest_path = benchmark_dir / "manifest.csv"
    if not manifest_path.exists():
        return pd.DataFrame()

    manifest = pd.read_csv(manifest_path)
    all_edges = []

    for _, row in manifest.iterrows():
        benchmark_id = row["benchmark_id"]

        # Parse condition from benchmark_id
        # Format: "3w_animal1_...", "25w_animal2_...", "sbiad1540_..."
        if benchmark_id.startswith("sbiad"):
            # Parse flow condition from filename
            if "static" in benchmark_id.lower():
                condition = "static"
            elif "dyn" in benchmark_id.lower():
                condition = "flow"
            else:
                condition = "sbiad_other"
        elif benchmark_id.startswith(("3w", "5w", "25w")):
            # Lymphatic timepoint
            condition = benchmark_id.split("_")[0]
        else:
            condition = "other"

        img_path = benchmark_dir / "images" / f"{benchmark_id}.tif"
        mask_path = benchmark_dir / "masks" / f"{benchmark_id}_mask.tif"
        ann_path = benchmark_dir / "annotations" / f"{benchmark_id}.json"

        if not all(p.exists() for p in [img_path, mask_path, ann_path]):
            continue

        try:
            image = tifffile.imread(img_path).astype(np.float32)
            masks = tifffile.imread(mask_path)

            with open(ann_path) as f:
                ann = json.load(f)

            iface = extract_interfaces(masks)
            boundary_vals = image[iface.all_boundary_mask]
            thresh = compute_threshold(boundary_vals, "otsu")

            for edge in ann["edges"]:
                cell_i, cell_j = edge["cell_i"], edge["cell_j"]
                key = (min(cell_i, cell_j), max(cell_i, cell_j))
                coords = iface.boundary_coords.get(key)

                if coords is None or len(coords) == 0:
                    continue

                mask = interface_mask_from_coords(coords, masks.shape, dilate_px=2)
                feats = interface_marker_features(image, mask, thresh)
                feats["condition"] = condition
                feats["dataset"] = "Benchmark"
                feats["image_id"] = benchmark_id
                all_edges.append(feats)

            print(f"  {benchmark_id}: {len(ann['edges'])} edges ({condition})")

        except Exception as e:
            print(f"  Error {benchmark_id}: {e}")
            continue

    return pd.DataFrame(all_edges)


# =============================================================================
# Transfer Learning Tests
# =============================================================================

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get numeric feature columns for classification."""
    exclude = ["condition", "dataset", "image_id", "cell_i", "cell_j", "contact_px"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in exclude and not c.startswith("_")]


def train_condition_classifier(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "condition",
) -> Tuple[RandomForestClassifier, LabelEncoder, float]:
    """Train a classifier to predict experimental condition from features."""

    # Clean data
    train_clean = train_df.dropna(subset=feature_cols + [target_col])

    if len(train_clean) < 20:
        return None, None, 0.0

    X = train_clean[feature_cols].values
    y = train_clean[target_col].values

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # Cross-validation score on training data
    if len(np.unique(y_encoded)) > 1:
        cv_scores = cross_val_score(clf, X, y_encoded, cv=min(5, len(np.unique(y_encoded))), scoring='accuracy')
        cv_acc = cv_scores.mean()
    else:
        cv_acc = 0.0

    clf.fit(X, y_encoded)

    return clf, le, cv_acc


def evaluate_transfer(
    clf: RandomForestClassifier,
    le: LabelEncoder,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "condition",
) -> Dict:
    """Evaluate trained classifier on test dataset."""

    test_clean = test_df.dropna(subset=feature_cols + [target_col])

    if len(test_clean) < 10:
        return {"error": "Insufficient test data", "n": len(test_clean)}

    X_test = test_clean[feature_cols].values
    y_test = test_clean[target_col].values

    # Check if test labels are in training label set
    known_labels = set(le.classes_)
    test_labels = set(y_test)
    common_labels = known_labels & test_labels

    if not common_labels:
        return {
            "error": "No overlapping conditions",
            "train_conditions": list(known_labels),
            "test_conditions": list(test_labels),
        }

    # Filter to common labels only
    mask = np.isin(y_test, list(common_labels))
    X_test = X_test[mask]
    y_test = y_test[mask]

    if len(y_test) < 10:
        return {"error": "Too few samples with common labels", "n": len(y_test)}

    y_test_encoded = le.transform(y_test)
    y_pred = clf.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test_encoded, y_pred)),
        "f1_macro": float(f1_score(y_test_encoded, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test_encoded, y_pred, average="weighted", zero_division=0)),
        "n_test": len(y_test),
        "common_conditions": list(common_labels),
        "confusion_matrix": confusion_matrix(y_test_encoded, y_pred).tolist(),
        "classification_report": classification_report(y_test_encoded, y_pred, target_names=le.classes_, output_dict=True, zero_division=0),
    }


def cross_dataset_transfer_test(
    datasets: Dict[str, pd.DataFrame],
    feature_cols: List[str],
) -> List[Dict]:
    """
    Proper cross-dataset transfer: train on A, test on B.

    Labels = experimental conditions (ground truth, not heuristic).
    """
    results = []
    dataset_names = list(datasets.keys())

    for train_name in dataset_names:
        train_df = datasets[train_name]

        if len(train_df) < 20:
            continue

        # Train classifier on this dataset
        clf, le, cv_acc = train_condition_classifier(train_df, feature_cols)

        if clf is None:
            continue

        for test_name in dataset_names:
            if test_name == train_name:
                continue

            test_df = datasets[test_name]

            if len(test_df) < 10:
                continue

            result = evaluate_transfer(clf, le, test_df, feature_cols)
            result["train_dataset"] = train_name
            result["test_dataset"] = test_name
            result["train_cv_accuracy"] = cv_acc
            result["train_n"] = len(train_df)
            result["train_conditions"] = list(train_df["condition"].unique())
            result["test_conditions"] = list(test_df["condition"].unique())

            results.append(result)

    return results


def leave_one_dataset_out(
    datasets: Dict[str, pd.DataFrame],
    feature_cols: List[str],
) -> List[Dict]:
    """Leave-one-dataset-out cross-validation."""
    results = []
    dataset_names = list(datasets.keys())

    for held_out in dataset_names:
        # Combine all except held_out for training
        train_dfs = [df for name, df in datasets.items() if name != held_out and len(df) > 0]
        test_df = datasets[held_out]

        if not train_dfs or len(test_df) < 10:
            continue

        train_df = pd.concat(train_dfs, ignore_index=True)

        clf, le, cv_acc = train_condition_classifier(train_df, feature_cols)

        if clf is None:
            continue

        result = evaluate_transfer(clf, le, test_df, feature_cols)
        result["held_out"] = held_out
        result["training_datasets"] = [n for n in dataset_names if n != held_out]
        result["train_cv_accuracy"] = cv_acc
        result["train_n"] = len(train_df)

        results.append(result)

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("PROPER CROSS-DATASET TRANSFER VALIDATION")
    print("=" * 80)
    print("""
This test uses EXPERIMENTAL CONDITIONS as labels (not heuristic morphology classes).
This avoids circular validation where both train/test labels come from the same source.

Ground truth labels:
- S-BIAD1540: static vs flow (shear stress conditions)
- Lymphatic EC: 3w, 5w, 25w (developmental timepoints)
- Benchmark: mixed conditions from both sources
""")

    data_dir = Path(__file__).parent.parent / "data"
    benchmark_dir = Path(__file__).parent.parent / "benchmark"
    output_dir = Path(__file__).parent.parent / "runs" / "proper_transfer_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {}

    # Load S-BIAD1540
    print("\n" + "-" * 60)
    print("Loading S-BIAD1540 (static vs flow conditions)")
    print("-" * 60)
    sbiad_dir = data_dir / "S-BIAD1540"
    if sbiad_dir.exists():
        df = load_sbiad1540_with_conditions(sbiad_dir, max_images=15)
        if len(df) > 0:
            datasets["S-BIAD1540"] = df
            print(f"Loaded {len(df)} edges")
            print(f"Conditions: {df['condition'].value_counts().to_dict()}")

    # Load Lymphatic EC
    print("\n" + "-" * 60)
    print("Loading Lymphatic EC (developmental timepoints)")
    print("-" * 60)
    lymph_dir = data_dir / "lymphatic_ec"
    if lymph_dir.exists():
        df = load_lymphatic_with_conditions(lymph_dir, max_images=10)
        if len(df) > 0:
            datasets["Lymphatic_EC"] = df
            print(f"Loaded {len(df)} edges")
            print(f"Conditions: {df['condition'].value_counts().to_dict()}")

    # Load Benchmark
    print("\n" + "-" * 60)
    print("Loading Benchmark dataset")
    print("-" * 60)
    if benchmark_dir.exists():
        df = load_benchmark_with_conditions(benchmark_dir)
        if len(df) > 0:
            datasets["Benchmark"] = df
            print(f"Loaded {len(df)} edges")
            print(f"Conditions: {df['condition'].value_counts().to_dict()}")

    if len(datasets) < 2:
        print("\nError: Need at least 2 datasets for transfer validation")
        return

    # Get common feature columns
    all_cols = None
    for df in datasets.values():
        cols = set(get_feature_columns(df))
        all_cols = cols if all_cols is None else all_cols & cols

    feature_cols = list(all_cols)
    print(f"\nUsing {len(feature_cols)} features: {feature_cols[:5]}...")

    # Run transfer tests
    print("\n" + "=" * 80)
    print("CROSS-DATASET TRANSFER (Train on A → Test on B)")
    print("=" * 80)
    print("\nTask: Predict experimental condition from junction features")
    print("Labels are GROUND TRUTH (experimental conditions), not heuristic.\n")

    transfer_results = cross_dataset_transfer_test(datasets, feature_cols)

    for r in transfer_results:
        if "error" in r:
            print(f"{r['train_dataset']} → {r['test_dataset']}: {r['error']}")
        else:
            print(f"{r['train_dataset']} → {r['test_dataset']}:")
            print(f"  Train conditions: {r['train_conditions']}")
            print(f"  Test conditions: {r['test_conditions']}")
            print(f"  Common conditions: {r.get('common_conditions', [])}")
            print(f"  Accuracy: {r['accuracy']:.3f}")
            print(f"  F1 (macro): {r['f1_macro']:.3f}")
            print(f"  N test: {r['n_test']}")
            print()

    # Leave-one-out
    print("\n" + "=" * 80)
    print("LEAVE-ONE-DATASET-OUT VALIDATION")
    print("=" * 80)

    loo_results = leave_one_dataset_out(datasets, feature_cols)

    for r in loo_results:
        if "error" in r:
            print(f"Held out {r['held_out']}: {r['error']}")
        else:
            print(f"Held out {r['held_out']}:")
            print(f"  Training datasets: {r['training_datasets']}")
            print(f"  Accuracy: {r['accuracy']:.3f}")
            print(f"  F1 (macro): {r['f1_macro']:.3f}")
            print()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    valid_transfers = [r for r in transfer_results if "error" not in r]
    if valid_transfers:
        avg_acc = np.mean([r["accuracy"] for r in valid_transfers])
        avg_f1 = np.mean([r["f1_macro"] for r in valid_transfers])
        print(f"\nCross-dataset transfer (condition prediction):")
        print(f"  Average accuracy: {avg_acc:.3f}")
        print(f"  Average F1: {avg_f1:.3f}")
        print(f"  N valid transfers: {len(valid_transfers)}")

    valid_loo = [r for r in loo_results if "error" not in r]
    if valid_loo:
        avg_acc = np.mean([r["accuracy"] for r in valid_loo])
        avg_f1 = np.mean([r["f1_macro"] for r in valid_loo])
        print(f"\nLeave-one-out validation:")
        print(f"  Average accuracy: {avg_acc:.3f}")
        print(f"  Average F1: {avg_f1:.3f}")

    print("""
INTERPRETATION:
- These accuracies measure whether junction FEATURES predict EXPERIMENTAL CONDITIONS
- High accuracy = junction morphology differs between conditions (good!)
- Low accuracy = junction features don't discriminate conditions well
- Transfer accuracy = do these differences generalize across datasets?

This is NOT the same as "classification accuracy" for morphology types.
""")

    # Save results
    all_results = {
        "transfer_tests": transfer_results,
        "leave_one_out": loo_results,
        "datasets": {name: {"n_edges": len(df), "conditions": df["condition"].value_counts().to_dict()}
                     for name, df in datasets.items()},
        "feature_columns": feature_cols,
    }

    with open(output_dir / "proper_transfer_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save detailed report
    report = f"""# Proper Cross-Dataset Transfer Validation

## Methodology

This validation uses **experimental conditions as ground truth labels**, not heuristic morphology classes.

### Why This Is Valid

The circular validation problem:
- Training labels: heuristic(features_A) → morphology_A
- Test labels: heuristic(features_B) → morphology_B
- High accuracy just means the heuristic is deterministic

The proper approach:
- Training labels: experimental condition (static/flow, timepoint)
- Test labels: experimental condition
- High accuracy means features predict conditions AND this generalizes

### Ground Truth Labels Used

| Dataset | Condition Labels | Source |
|---------|------------------|--------|
| S-BIAD1540 | static, flow | Experimental shear stress conditions |
| Lymphatic EC | 3w, 5w, 25w | Developmental timepoints |

## Results

### Cross-Dataset Transfer (Train A → Test B)

| Train | Test | Accuracy | F1 | Interpretation |
|-------|------|----------|----|----|
"""

    for r in transfer_results:
        if "error" in r:
            report += f"| {r['train_dataset']} | {r['test_dataset']} | - | - | {r['error']} |\n"
        else:
            report += f"| {r['train_dataset']} | {r['test_dataset']} | {r['accuracy']:.3f} | {r['f1_macro']:.3f} | Valid |\n"

    report += """
### Leave-One-Dataset-Out

| Held Out | Accuracy | F1 |
|----------|----------|----|\n"""

    for r in loo_results:
        if "error" in r:
            report += f"| {r['held_out']} | - | {r['error']} |\n"
        else:
            report += f"| {r['held_out']} | {r['accuracy']:.3f} | {r['f1_macro']:.3f} |\n"

    report += """
## Interpretation

### What High Accuracy Means
- Junction features differ between experimental conditions
- These differences are consistent across datasets
- The feature→condition mapping generalizes

### What Low Accuracy Means
- Junction features don't strongly predict experimental conditions
- OR conditions are too different between datasets to transfer
- OR sample size is too small

### Key Difference from Previous "Transfer" Test

Previous test trained on heuristic labels, tested on heuristic labels.
That was circular - it only showed the heuristic is consistent.

This test trains on experimental conditions (ground truth), tests on experimental conditions.
This is proper transfer learning validation.

## Caveats

1. **Condition overlap required**: Transfer only works if train and test share some conditions
2. **Class imbalance**: Some conditions may be underrepresented
3. **Confounders**: Other factors (imaging, cell density) may affect features

## Files

- `proper_transfer_results.json`: Full results
- This report: `PROPER_TRANSFER_REPORT.md`
"""

    with open(output_dir / "PROPER_TRANSFER_REPORT.md", "w") as f:
        f.write(report)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
