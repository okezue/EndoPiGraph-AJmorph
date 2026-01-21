#!/usr/bin/env python3
"""
Combine existing sbiad1540_full dataset with new egm2_full images
and run B3 experiment on combined data.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


def extract_image_features(run_dir: Path, manifest_paths: list) -> pd.DataFrame:
    """Extract per-image features for condition prediction."""

    # Load and combine manifests
    manifest_dfs = []
    for mp in manifest_paths:
        if Path(mp).exists():
            manifest_dfs.append(pd.read_csv(mp))
    manifest = pd.concat(manifest_dfs, ignore_index=True)
    print(f"Combined manifest: {len(manifest)} entries")

    all_features = []

    for subdir in sorted(run_dir.iterdir()):
        if not subdir.is_dir():
            continue

        cells_file = subdir / "cells.csv"
        edges_file = subdir / "edges.csv"

        if not cells_file.exists() or not edges_file.exists():
            continue

        cells = pd.read_csv(cells_file)
        edges = pd.read_csv(edges_file)

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

        if 'area' in cells.columns:
            features['mean_cell_area'] = cells['area'].mean()
            features['std_cell_area'] = cells['area'].std()
            features['cv_cell_area'] = features['std_cell_area'] / features['mean_cell_area'] if features['mean_cell_area'] > 0 else 0

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

        if 'aj_morph' in edges.columns:
            morph_counts = edges['aj_morph'].value_counts(normalize=True)
            for morph_class in ['reticular', 'straight', 'fingers', 'other']:
                features[f'prop_{morph_class}'] = morph_counts.get(morph_class, 0)

        all_features.append(features)

    return pd.DataFrame(all_features)


def main():
    # Combine both run directories
    run_dirs = [
        Path("runs/sbiad1540_full"),
        Path("runs/egm2_full")
    ]

    manifest_paths = [
        "data/S-BIAD1540/manifest_subset.csv",
        "runs/egm2_full/manifest_egm2_local.csv"
    ]

    # Collect features from both directories
    all_features = []
    for run_dir in run_dirs:
        if run_dir.exists():
            print(f"\nProcessing {run_dir}...")
            df = extract_image_features(run_dir, manifest_paths)
            all_features.append(df)
            print(f"  Found {len(df)} images")

    df = pd.concat(all_features, ignore_index=True)

    # Remove duplicates (in case any image appears in both)
    df = df.drop_duplicates(subset='image_id')

    print(f"\n{'='*70}")
    print(f"Combined dataset: {len(df)} images")
    print(f"Conditions: {df['condition'].value_counts().to_dict()}")
    print(f"{'='*70}")

    # Define feature sets
    untyped_cols = [c for c in ['n_cells', 'n_edges', 'mean_degree', 'edge_density',
                                'mean_cell_area', 'std_cell_area', 'cv_cell_area'] if c in df.columns]

    typed_cols = [c for c in ['mean_aj_occupancy', 'std_aj_occupancy', 'median_aj_occupancy',
                              'mean_aj_clusters', 'std_aj_clusters', 'max_aj_clusters',
                              'mean_aj_intensity', 'std_aj_intensity',
                              'mean_linearity', 'std_linearity',
                              'mean_thickness', 'std_thickness',
                              'mean_skeleton_len', 'std_skeleton_len',
                              'prop_reticular', 'prop_straight', 'prop_fingers', 'prop_other'] if c in df.columns]

    combined_cols = untyped_cols + typed_cols

    feature_sets = {
        'untyped': untyped_cols,
        'typed': typed_cols,
        'combined': combined_cols
    }

    y = df['condition'].values
    results = {}

    loo = LeaveOneOut()

    for feature_type, feature_cols in feature_sets.items():
        print(f"\n{'='*50}")
        print(f"Feature set: {feature_type.upper()}")
        print(f"Features ({len(feature_cols)}): {feature_cols[:5]}...")
        print("="*50)

        X = df[feature_cols].values
        X = np.nan_to_num(X, nan=0.0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        y_pred_rf = cross_val_predict(clf_rf, X_scaled, y, cv=loo)
        acc_rf = accuracy_score(y, y_pred_rf)
        f1_rf = f1_score(y, y_pred_rf, average='macro')

        clf_lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        y_pred_lr = cross_val_predict(clf_lr, X_scaled, y, cv=loo)
        acc_lr = accuracy_score(y, y_pred_lr)
        f1_lr = f1_score(y, y_pred_lr, average='macro')

        print(f"\nRandom Forest:       Accuracy={acc_rf:.3f}, Macro-F1={f1_rf:.3f}")
        print(f"Logistic Regression: Accuracy={acc_lr:.3f}, Macro-F1={f1_lr:.3f}")

        results[feature_type] = {
            'n_features': len(feature_cols),
            'rf_accuracy': float(acc_rf),
            'rf_f1_macro': float(f1_rf),
            'lr_accuracy': float(acc_lr),
            'lr_f1_macro': float(f1_lr)
        }

    # Summary
    print("\n" + "="*70)
    print("COMBINED DATASET SUMMARY")
    print("="*70)
    print(f"\n{'Feature Set':<15} {'N Features':<12} {'RF Acc':<10} {'RF F1':<10}")
    print("-"*50)
    for ft, res in results.items():
        print(f"{ft:<15} {res['n_features']:<12} {res['rf_accuracy']:<10.3f} {res['rf_f1_macro']:<10.3f}")

    # Save results
    output_file = Path("runs/combined_typed_vs_untyped_results.json")
    with open(output_file, "w") as f:
        json.dump({
            'n_images': len(df),
            'conditions': df['condition'].value_counts().to_dict(),
            'results': results
        }, f, indent=2)
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
