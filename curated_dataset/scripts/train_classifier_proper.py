#!/usr/bin/env python3
"""
Proper AJmorph classifier training with image-level splits.

CRITICAL NOTE:
The current labels are HEURISTIC-DERIVED from the same features used for training.
This means high accuracy indicates the RF learned the threshold rules, NOT validated
biological classification. Manual annotation is required for true validation.

Per Polarity-JaM paper: "advanced classifier translating junction features into
the 5 manual AJ classes was not ready and requires manual training data."

Usage:
    python scripts/train_classifier_proper.py runs/sbiad1540_full/all_edges.csv models/
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict


# Features used for classification (must NOT include label columns)
FEATURE_COLS = [
    'aj_mean_intensity', 'aj_max_intensity', 'aj_std_intensity',
    'aj_occupancy', 'aj_intensity_per_interface', 'aj_cluster_count',
    'aj_cluster_area_mean', 'aj_skeleton_len', 'aj_linearity_index',
    'aj_thickness_proxy'
]


def train_and_evaluate(edges_csv: Path, output_dir: Path, n_splits: int = 5):
    """Train classifier with proper image-level cross-validation."""

    # Load data
    edges = pd.read_csv(edges_csv)

    # Filter rare classes into "other"
    rare_threshold = 10
    class_counts = edges['aj_morph'].value_counts()
    rare_classes = class_counts[class_counts < rare_threshold].index.tolist()
    edges['aj_morph'] = edges['aj_morph'].apply(
        lambda x: 'other' if x in rare_classes else x
    )

    df = edges[edges['aj_morph'].notna()].copy()

    # Prepare features and labels
    X = df[FEATURE_COLS].values
    y = df['aj_morph'].values
    groups = df['image_id'].values

    n_samples = len(X)
    n_images = len(np.unique(groups))
    classes = sorted(np.unique(y))

    print("="*60)
    print("CLASSIFIER TRAINING WITH IMAGE-LEVEL SPLITS")
    print("="*60)
    print(f"\nData: {n_samples} edges from {n_images} images")
    print(f"Features: {len(FEATURE_COLS)}")
    print(f"Classes: {classes}")
    print(f"\nClass distribution:")
    print(pd.Series(y).value_counts())

    # Cross-validation
    print(f"\n{n_splits}-Fold GroupKFold Cross-Validation (by image_id)")
    print("-"*60)

    gkf = GroupKFold(n_splits=n_splits)
    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )

    # Get CV predictions
    y_pred = cross_val_predict(clf, X, y, groups=groups, cv=gkf)

    # Classification report
    report = classification_report(y_true=y, y_pred=y_pred, output_dict=True)
    report_str = classification_report(y_true=y, y_pred=y_pred)
    print("\nClassification Report:")
    print(report_str)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred, labels=classes)
    print("\nConfusion Matrix:")
    print(f"Labels: {classes}")
    print(cm)

    # Per-fold results
    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        train_images = list(np.unique(groups[train_idx]))
        test_images = list(np.unique(groups[test_idx]))

        clf_fold = RandomForestClassifier(
            n_estimators=400, random_state=42,
            class_weight='balanced', n_jobs=-1
        )
        clf_fold.fit(X[train_idx], y[train_idx])
        y_fold_pred = clf_fold.predict(X[test_idx])
        acc = (y_fold_pred == y[test_idx]).mean()

        fold_results.append({
            'fold': fold_idx + 1,
            'train_images': train_images,
            'test_images': test_images,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'accuracy': float(acc)
        })

        print(f"\nFold {fold_idx+1}: Test images={test_images}, Acc={acc:.3f}")

    # Train final model on all data
    print("\n" + "-"*60)
    print("Training final model on all data...")
    clf.fit(X, y)

    # Feature importance
    feat_importance = sorted(
        zip(FEATURE_COLS, clf.feature_importances_),
        key=lambda x: -x[1]
    )
    print("\nFeature Importances:")
    for feat, imp in feat_importance:
        print(f"  {feat}: {imp:.3f}")

    # Save model with metadata
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get version info
    import sklearn
    model_artifact = {
        'classifier': clf,
        'feature_cols': FEATURE_COLS,
        'classes': classes,
        'metadata': {
            'created': datetime.now().isoformat(),
            'source_data': str(edges_csv),
            'n_samples': n_samples,
            'n_images': n_images,
            'n_folds': n_splits,
            'sklearn_version': sklearn.__version__,
            'numpy_version': np.__version__,
            'python_version': sys.version,
            'evaluation': {
                'method': 'GroupKFold by image_id',
                'macro_f1': report['macro avg']['f1-score'],
                'weighted_f1': report['weighted avg']['f1-score'],
                'accuracy': report['accuracy'],
                'fold_results': fold_results
            },
            'WARNING': (
                "Labels are HEURISTIC-DERIVED from threshold rules on the same "
                "features used for training. High accuracy indicates the RF "
                "learned the rules, NOT validated biological classification. "
                "Manual annotation required for true validation."
            )
        }
    }

    model_path = output_dir / 'ajmorph_classifier_v2.joblib'
    joblib.dump(model_artifact, model_path)
    print(f"\nSaved model to {model_path}")

    # Save evaluation report
    eval_report = {
        'created': datetime.now().isoformat(),
        'data': {
            'source': str(edges_csv),
            'n_samples': n_samples,
            'n_images': n_images,
            'class_distribution': pd.Series(y).value_counts().to_dict()
        },
        'evaluation': {
            'method': f'{n_splits}-fold GroupKFold by image_id',
            'metrics': {k: v for k, v in report.items() if k not in ('accuracy',)},
            'accuracy': report['accuracy'],
            'confusion_matrix': cm.tolist(),
            'confusion_labels': classes,
            'fold_results': fold_results
        },
        'feature_importance': dict(feat_importance),
        'warning': model_artifact['metadata']['WARNING']
    }

    eval_path = output_dir / 'ajmorph_evaluation_report.json'
    with open(eval_path, 'w') as f:
        json.dump(eval_report, f, indent=2)
    print(f"Saved evaluation report to {eval_path}")

    # Print warning
    print("\n" + "="*60)
    print("CRITICAL WARNING")
    print("="*60)
    print(model_artifact['metadata']['WARNING'])

    return model_artifact


def main():
    parser = argparse.ArgumentParser(
        description="Train AJmorph classifier with proper image-level splits"
    )
    parser.add_argument("edges_csv", help="Path to all_edges.csv")
    parser.add_argument("output_dir", help="Output directory for model and reports")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()

    train_and_evaluate(
        Path(args.edges_csv),
        Path(args.output_dir),
        args.n_splits
    )


if __name__ == "__main__":
    main()
