#!/usr/bin/env python3
"""
B3: Typed vs Untyped Edge Features for Condition Prediction

This experiment tests whether typed edge features (AJ morphology, junction
quantification) improve condition classification compared to untyped graph
statistics.

Hypothesis: If typed edge features outperform untyped graph statistics,
it demonstrates the value of the typed π-graph representation.

Features compared:
1. Untyped graph stats: n_cells, n_edges, mean_degree, degree_std
2. Typed edge features: AJ occupancy, cluster count, intensity, morph proportions
3. Combined: Both feature sets

Usage:
    python scripts/typed_vs_untyped_experiment.py runs/sbiad1540_full/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


def extract_image_features(run_dir: Path, manifest_path: Path = None) -> pd.DataFrame:
    """Extract per-image features for condition prediction."""

    # Try to find manifest in multiple locations
    manifest_candidates = [
        manifest_path,
        run_dir / "manifest_egm2_local.csv",
        Path("data/S-BIAD1540/manifest_subset.csv"),
        Path("data/S-BIAD1540/manifest_egm2_full.csv"),
    ]

    manifest = None
    for candidate in manifest_candidates:
        if candidate is not None and candidate.exists():
            manifest = pd.read_csv(candidate)
            print(f"Using manifest: {candidate}")
            break

    if manifest is None:
        raise FileNotFoundError("No manifest found")

    all_features = []

    for subdir in sorted(run_dir.iterdir()):
        if not subdir.is_dir():
            continue

        cells_file = subdir / "cells.csv"
        edges_file = subdir / "edges.csv"

        if not cells_file.exists() or not edges_file.exists():
            continue

        try:
            cells = pd.read_csv(cells_file)
            edges = pd.read_csv(edges_file)
        except Exception:
            continue

        # Skip images with no edges
        if len(edges) == 0 or len(cells) == 0:
            continue

        image_id = subdir.name

        # Get condition from manifest
        match = manifest[manifest['image_id'] == image_id]
        if len(match) == 0:
            continue

        shear = match['shear_stress'].iloc[0]
        if 'static' in str(shear).lower():
            condition = 'static'
        elif '6dyn' in str(shear).lower():
            condition = '6dyne'
        elif '18' in str(shear) or '20dyn' in str(shear).lower():
            condition = 'high_shear'
        else:
            continue

        features = {'image_id': image_id, 'condition': condition}

        # === UNTYPED GRAPH FEATURES ===
        n_cells = len(cells)
        n_edges = len(edges)

        features['n_cells'] = n_cells
        features['n_edges'] = n_edges
        features['mean_degree'] = 2 * n_edges / n_cells if n_cells > 0 else 0
        features['edge_density'] = n_edges / (n_cells * (n_cells - 1) / 2) if n_cells > 1 else 0

        # Cell area statistics
        if 'area' in cells.columns:
            features['mean_cell_area'] = cells['area'].mean()
            features['std_cell_area'] = cells['area'].std()
            features['cv_cell_area'] = features['std_cell_area'] / features['mean_cell_area'] if features['mean_cell_area'] > 0 else 0

        # Interface length statistics
        if 'interface_length' in edges.columns:
            features['mean_interface_len'] = edges['interface_length'].mean()
            features['std_interface_len'] = edges['interface_length'].std()

        # === TYPED EDGE FEATURES (AJ-specific) ===
        if 'aj_occupancy' in edges.columns:
            features['mean_aj_occupancy'] = edges['aj_occupancy'].mean()
            features['std_aj_occupancy'] = edges['aj_occupancy'].std()
            features['median_aj_occupancy'] = edges['aj_occupancy'].median()

        if 'aj_cluster_count' in edges.columns:
            features['mean_aj_clusters'] = edges['aj_cluster_count'].mean()
            features['std_aj_clusters'] = edges['aj_cluster_count'].std()
            features['max_aj_clusters'] = edges['aj_cluster_count'].max()

        if 'aj_mean_intensity' in edges.columns:
            features['mean_aj_intensity'] = edges['aj_mean_intensity'].mean()
            features['std_aj_intensity'] = edges['aj_mean_intensity'].std()

        if 'aj_linearity_index' in edges.columns:
            features['mean_linearity'] = edges['aj_linearity_index'].mean()
            features['std_linearity'] = edges['aj_linearity_index'].std()

        if 'aj_thickness_proxy' in edges.columns:
            features['mean_thickness'] = edges['aj_thickness_proxy'].mean()
            features['std_thickness'] = edges['aj_thickness_proxy'].std()

        if 'aj_skeleton_len' in edges.columns:
            features['mean_skeleton_len'] = edges['aj_skeleton_len'].mean()
            features['std_skeleton_len'] = edges['aj_skeleton_len'].std()

        # AJ morphology proportions (categorical → numeric)
        if 'aj_morph' in edges.columns:
            morph_counts = edges['aj_morph'].value_counts(normalize=True)
            for morph_class in ['reticular', 'straight', 'fingers', 'other', 'thick', 'thick_to_reticular']:
                features[f'prop_{morph_class}'] = morph_counts.get(morph_class, 0)

        all_features.append(features)

    return pd.DataFrame(all_features)


def define_feature_sets(df: pd.DataFrame) -> dict:
    """Define untyped vs typed feature sets."""

    # Untyped: basic graph statistics (no AJ-specific info)
    untyped_cols = [
        'n_cells', 'n_edges', 'mean_degree', 'edge_density',
        'mean_cell_area', 'std_cell_area', 'cv_cell_area',
        'mean_interface_len', 'std_interface_len'
    ]
    untyped_cols = [c for c in untyped_cols if c in df.columns]

    # Typed: AJ-specific features
    typed_cols = [
        'mean_aj_occupancy', 'std_aj_occupancy', 'median_aj_occupancy',
        'mean_aj_clusters', 'std_aj_clusters', 'max_aj_clusters',
        'mean_aj_intensity', 'std_aj_intensity',
        'mean_linearity', 'std_linearity',
        'mean_thickness', 'std_thickness',
        'mean_skeleton_len', 'std_skeleton_len',
        'prop_reticular', 'prop_straight', 'prop_fingers', 'prop_other'
    ]
    typed_cols = [c for c in typed_cols if c in df.columns]

    # Combined: all features
    combined_cols = untyped_cols + typed_cols

    return {
        'untyped': untyped_cols,
        'typed': typed_cols,
        'combined': combined_cols
    }


def run_classification_experiment(df: pd.DataFrame, feature_sets: dict) -> dict:
    """Run classification experiment comparing feature sets."""

    y = df['condition'].values
    results = {}

    print("="*70)
    print("CONDITION PREDICTION: TYPED vs UNTYPED FEATURES")
    print("="*70)
    print(f"\nDataset: {len(df)} images")
    print(f"Conditions: {np.unique(y)}")
    print(f"Distribution: {pd.Series(y).value_counts().to_dict()}")

    # Use Leave-One-Out CV (small dataset)
    loo = LeaveOneOut()

    for feature_type, feature_cols in feature_sets.items():
        print(f"\n{'='*50}")
        print(f"Feature set: {feature_type.upper()}")
        print(f"Features ({len(feature_cols)}): {feature_cols}")
        print("="*50)

        X = df[feature_cols].values

        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Random Forest
        clf_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        y_pred_rf = cross_val_predict(clf_rf, X_scaled, y, cv=loo)
        acc_rf = accuracy_score(y, y_pred_rf)
        f1_rf = f1_score(y, y_pred_rf, average='macro')

        # Logistic Regression (simpler model)
        clf_lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        y_pred_lr = cross_val_predict(clf_lr, X_scaled, y, cv=loo)
        acc_lr = accuracy_score(y, y_pred_lr)
        f1_lr = f1_score(y, y_pred_lr, average='macro')

        print(f"\nRandom Forest:       Accuracy={acc_rf:.3f}, Macro-F1={f1_rf:.3f}")
        print(f"Logistic Regression: Accuracy={acc_lr:.3f}, Macro-F1={f1_lr:.3f}")

        # Detailed report for RF
        print(f"\nRandom Forest Classification Report:")
        print(classification_report(y, y_pred_rf))

        results[feature_type] = {
            'n_features': len(feature_cols),
            'features': feature_cols,
            'rf_accuracy': float(acc_rf),
            'rf_f1_macro': float(f1_rf),
            'lr_accuracy': float(acc_lr),
            'lr_f1_macro': float(f1_lr),
            'rf_predictions': y_pred_rf.tolist(),
            'lr_predictions': y_pred_lr.tolist()
        }

    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"\n{'Feature Set':<15} {'N Features':<12} {'RF Acc':<10} {'RF F1':<10} {'LR Acc':<10} {'LR F1':<10}")
    print("-"*70)
    for ft, res in results.items():
        print(f"{ft:<15} {res['n_features']:<12} {res['rf_accuracy']:<10.3f} {res['rf_f1_macro']:<10.3f} {res['lr_accuracy']:<10.3f} {res['lr_f1_macro']:<10.3f}")

    # Determine winner
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    typed_f1 = results['typed']['rf_f1_macro']
    untyped_f1 = results['untyped']['rf_f1_macro']
    combined_f1 = results['combined']['rf_f1_macro']

    if typed_f1 > untyped_f1:
        improvement = (typed_f1 - untyped_f1) / untyped_f1 * 100
        print(f"\nTyped features OUTPERFORM untyped by {improvement:.1f}% (F1: {typed_f1:.3f} vs {untyped_f1:.3f})")
        print("This demonstrates the value of AJ-typed edge representation!")
    elif untyped_f1 > typed_f1:
        print(f"\nUntyped features perform better (F1: {untyped_f1:.3f} vs {typed_f1:.3f})")
        print("Basic graph structure is sufficient for this task.")
    else:
        print(f"\nFeature sets perform equally (F1: {typed_f1:.3f})")

    if combined_f1 > max(typed_f1, untyped_f1):
        print(f"\nCombined features achieve best performance (F1: {combined_f1:.3f})")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Pipeline run directory")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir

    # Extract features
    print("Extracting per-image features...")
    df = extract_image_features(run_dir)
    print(f"Extracted features for {len(df)} images")

    # Save features
    df.to_csv(output_dir / "image_features_for_classification.csv", index=False)
    print(f"Saved: {output_dir / 'image_features_for_classification.csv'}")

    # Define feature sets
    feature_sets = define_feature_sets(df)

    # Run experiment
    results = run_classification_experiment(df, feature_sets)

    # Save results
    with open(output_dir / "typed_vs_untyped_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_dir / 'typed_vs_untyped_results.json'}")


if __name__ == "__main__":
    main()
