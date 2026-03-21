from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


FEATURE_COLS = [
    "occupancy",
    "skeleton_len",
    "mean",
    "std",
    "cluster_count",
    "skeleton_components",
    "skeleton_endpoints",
    "skeleton_branch_points",
    "thickness_proxy",
]

BLUR_ROBUST_COLS = [
    "occupancy",
    "skeleton_len",
    "mean",
]

CANONICAL_NAMES = {
    "high_occ_low_complexity": "straight",
    "high_occ_high_complexity": "reticular",
    "med_occ_thick": "thick",
    "med_occ_transitional": "thick_to_reticular",
    "low_occ_elongated": "fingers",
    "low_occ_sparse": "discontinuous",
    "very_low_occ": "minimal",
    "low_occ_compact": "punctate",
}


def _get_feature_matrix(df: pd.DataFrame, prefix: str, cols: List[str]) -> Tuple[np.ndarray, List[str]]:
    avail = []
    for c in cols:
        col = f"{prefix}{c}"
        if col in df.columns:
            avail.append(col)
    if not avail:
        raise ValueError(f"No feature columns found with prefix '{prefix}'")
    X = df[avail].values.astype(np.float64)
    mask = np.isfinite(X).all(axis=1)
    return X, avail, mask


def cluster_junctions_gmm(
    df: pd.DataFrame,
    prefix: str = "AJ_",
    k_range: Tuple[int, int] = (3, 9),
    blur_robust: bool = True,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    cols = BLUR_ROBUST_COLS if blur_robust else FEATURE_COLS
    X_raw, used_cols, valid_mask = _get_feature_matrix(df, prefix, cols)
    X_valid = X_raw[valid_mask]
    if len(X_valid) < 10:
        raise ValueError(f"Too few valid rows ({len(X_valid)}) for clustering")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    best_bic = np.inf
    best_k = k_range[0]
    bic_scores = {}
    for k in range(k_range[0], k_range[1]+1):
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              n_init=5, random_state=random_state)
        gmm.fit(X_scaled)
        bic = gmm.bic(X_scaled)
        bic_scores[k] = float(bic)
        if bic < best_bic:
            best_bic = bic
            best_k = k

    gmm_best = GaussianMixture(n_components=best_k, covariance_type="full",
                                n_init=10, random_state=random_state)
    gmm_best.fit(X_scaled)
    cluster_ids = gmm_best.predict(X_scaled)
    probs = gmm_best.predict_proba(X_scaled)

    centroids_scaled = gmm_best.means_
    centroids_raw = scaler.inverse_transform(centroids_scaled)

    occ_idx = None
    sk_idx = None
    for i, c in enumerate(used_cols):
        if c.endswith("occupancy"):
            occ_idx = i
        elif c.endswith("skeleton_len"):
            sk_idx = i
        elif c.endswith("mean"):
            pass

    name_map = {}
    used_names = {}
    for cid in range(best_k):
        occ = centroids_raw[cid, occ_idx] if occ_idx is not None else 0.5
        sk = centroids_raw[cid, sk_idx] if sk_idx is not None else 0
        base = _assign_canonical_name(occ, sk, cid, best_k)
        if base in used_names:
            used_names[base] += 1
            name_map[cid] = f"{base}_{used_names[base]}"
        else:
            used_names[base] = 1
            name_map[cid] = base

    out = df.copy()
    labels = np.full(len(df), "unknown", dtype=object)
    confidence = np.full(len(df), np.nan)
    cluster_col = np.full(len(df), -1, dtype=int)

    valid_idx = np.where(valid_mask)[0]
    for i, idx in enumerate(valid_idx):
        cid = int(cluster_ids[i])
        labels[idx] = name_map[cid]
        confidence[idx] = float(probs[i, cid])
        cluster_col[idx] = cid

    out[f"{prefix}morph_label"] = labels
    out[f"{prefix}morph_confidence"] = confidence
    out[f"{prefix}morph_cluster"] = cluster_col

    meta = {
        "method": "GMM",
        "optimal_k": best_k,
        "bic_scores": bic_scores,
        "features_used": used_cols,
        "blur_robust": blur_robust,
        "n_samples": int(len(X_valid)),
        "cluster_centroids": {name_map[i]: centroids_raw[i].tolist()
                              for i in range(best_k)},
        "cluster_sizes": {name_map[i]: int((cluster_ids==i).sum())
                          for i in range(best_k)},
        "name_mapping": name_map,
    }
    return out, meta


def _assign_canonical_name(occ: float, sk: float, cid: int, k: int) -> str:
    if occ < 0.15:
        if sk > 30:
            return "fingers"
        return "minimal"
    elif occ < 0.3:
        if sk > 20:
            return "discontinuous"
        return "punctate"
    elif occ < 0.5:
        if sk > 40:
            return "thick_to_reticular"
        return "thick"
    elif occ < 0.7:
        if sk > 50:
            return "reticular"
        return "thick"
    else:
        if sk > 30:
            return "reticular"
        return "straight"


def evaluate_cluster_stability(
    df: pd.DataFrame,
    prefix: str = "AJ_",
    n_bootstrap: int = 50,
    blur_robust: bool = True,
    random_state: int = 0,
) -> Dict[str, Any]:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import adjusted_rand_score

    cols = BLUR_ROBUST_COLS if blur_robust else FEATURE_COLS
    X_raw, used_cols, valid_mask = _get_feature_matrix(df, prefix, cols)
    X_valid = X_raw[valid_mask]
    n = len(X_valid)
    if n < 20:
        return {"error": "too few samples", "n": n}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    gmm_ref = GaussianMixture(n_components=5, covariance_type="full",
                               n_init=10, random_state=random_state)
    ref_labels = gmm_ref.fit_predict(X_scaled)

    rng = np.random.RandomState(random_state)
    ari_scores = []
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        X_boot = X_scaled[idx]
        gmm_b = GaussianMixture(n_components=5, covariance_type="full",
                                 n_init=3, random_state=random_state+b+1)
        boot_labels = gmm_b.fit_predict(X_boot)
        ref_sub = ref_labels[idx]
        ari = adjusted_rand_score(ref_sub, boot_labels)
        ari_scores.append(float(ari))

    return {
        "mean_ari": float(np.mean(ari_scores)),
        "std_ari": float(np.std(ari_scores)),
        "median_ari": float(np.median(ari_scores)),
        "n_bootstrap": n_bootstrap,
        "n_samples": n,
        "interpretation": (
            "stable" if np.mean(ari_scores) > 0.6
            else "moderate" if np.mean(ari_scores) > 0.4
            else "unstable"
        ),
    }


def cross_validate_clustering(
    df: pd.DataFrame,
    prefix: str = "AJ_",
    image_col: str = "image_id",
    n_folds: int = 5,
    blur_robust: bool = True,
    random_state: int = 0,
) -> Dict[str, Any]:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import adjusted_rand_score
    from sklearn.model_selection import GroupKFold

    cols = BLUR_ROBUST_COLS if blur_robust else FEATURE_COLS
    X_raw, used_cols, valid_mask = _get_feature_matrix(df, prefix, cols)
    X_valid = X_raw[valid_mask]
    df_valid = df[valid_mask].reset_index(drop=True)

    if image_col not in df_valid.columns:
        groups = np.arange(len(df_valid))
    else:
        groups = df_valid[image_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    n_unique = len(np.unique(groups))
    actual_folds = min(n_folds, n_unique)
    if actual_folds < 2:
        return {"error": "not enough groups for CV", "n_groups": n_unique}

    gkf = GroupKFold(n_splits=actual_folds)
    fold_aris = []

    for train_idx, test_idx in gkf.split(X_scaled, groups=groups):
        X_train = X_scaled[train_idx]
        X_test = X_scaled[test_idx]

        gmm = GaussianMixture(n_components=5, covariance_type="full",
                               n_init=5, random_state=random_state)
        gmm.fit(X_train)
        gmm.predict(X_train)
        test_labels = gmm.predict(X_test)

        gmm_all = GaussianMixture(n_components=5, covariance_type="full",
                                   n_init=5, random_state=random_state)
        gmm_all.fit(X_scaled)
        all_labels_test = gmm_all.predict(X_test)

        ari = adjusted_rand_score(all_labels_test, test_labels)
        fold_aris.append(float(ari))

    return {
        "mean_ari": float(np.mean(fold_aris)),
        "std_ari": float(np.std(fold_aris)),
        "n_folds": actual_folds,
        "fold_aris": fold_aris,
    }
