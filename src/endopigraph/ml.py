from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def train_ajmorph_classifier(
    features_csv: str | Path,
    labels_csv: str | Path,
    out_dir: str | Path,
    label_col: str = "aj_label",
    id_cols: Optional[List[str]] = None,
) -> Path:
    """Train a simple AJ morphology classifier on exported edge features.

    ... Requires optional dependency: scikit-learn (pip install .[ml])
    """
    if id_cols is None:
        id_cols = ["image_id", "cell_i", "cell_j"]

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    features = pd.read_csv(features_csv)
    labels = pd.read_csv(labels_csv)

    # normalize pair ordering in labels
    labels = labels.copy()
    _a = labels["cell_i"].to_numpy()
    _b = labels["cell_j"].to_numpy()
    labels["cell_i"] = np.minimum(_a, _b)
    labels["cell_j"] = np.maximum(_a, _b)

    df = features.merge(labels, on=id_cols, how="inner")
    if df.empty:
        raise ValueError("No rows after merging features and labels. Check id columns.")

    y = df[label_col].astype(str)
    X = df.drop(columns=[label_col])

    # choose numeric feature columns only
    num_cols = [c for c in X.columns if c not in id_cols and pd.api.types.is_numeric_dtype(X[c])]
    if not num_cols:
        raise ValueError("No numeric feature columns found to train on.")

    X_num = X[num_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X_num, y, test_size=0.2, random_state=0, stratify=y if y.nunique() > 1 else None
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "ajmorph_random_forest.joblib"
    joblib.dump({"pipeline": pipeline, "feature_columns": num_cols}, model_path)

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return model_path
