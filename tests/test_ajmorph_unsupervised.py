import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from endopigraph.ajmorph_unsupervised import (
    cluster_junctions_gmm,
    evaluate_cluster_stability,
    _assign_canonical_name,
    BLUR_ROBUST_COLS,
)


def _make_synthetic_edges(n=200, seed=42):
    rng = np.random.RandomState(seed)
    q = n // 4
    rem = n - 4 * q
    occ = np.concatenate([
        rng.uniform(0.0, 0.15, q),
        rng.uniform(0.3, 0.5, q),
        rng.uniform(0.5, 0.7, q),
        rng.uniform(0.7, 1.0, q + rem),
    ])
    return pd.DataFrame({
        "AJ_occupancy": occ,
        "AJ_skeleton_len": rng.uniform(0, 80, n),
        "AJ_mean": rng.uniform(50, 200, n),
        "AJ_std": rng.uniform(10, 50, n),
        "AJ_cluster_count": rng.randint(0, 10, n),
        "AJ_skeleton_components": rng.randint(0, 5, n),
        "AJ_skeleton_endpoints": rng.randint(0, 10, n),
        "AJ_skeleton_branch_points": rng.randint(0, 8, n),
        "AJ_thickness_proxy": rng.uniform(0.5, 5.0, n),
        "image_id": [f"img_{i//max(n//10,1)}" for i in range(n)],
    })


class TestClusterJunctionsGMM:
    def test_basic(self):
        df = _make_synthetic_edges(200)
        out, meta = cluster_junctions_gmm(df, prefix="AJ_", blur_robust=True)
        assert "AJ_morph_label" in out.columns
        assert "AJ_morph_confidence" in out.columns
        assert meta["method"] == "GMM"
        assert meta["optimal_k"] >= 3

    def test_labels_valid(self):
        df = _make_synthetic_edges(200)
        out, meta = cluster_junctions_gmm(df)
        base_valid = {"straight", "thick", "thick_to_reticular", "reticular",
                      "fingers", "discontinuous", "minimal", "punctate", "unknown"}
        for lbl in out["AJ_morph_label"].unique():
            base = lbl.rsplit("_", 1)[0] if "_" in lbl and lbl.rsplit("_",1)[1].isdigit() else lbl
            assert base in base_valid, f"Unexpected label: {lbl}"

    def test_confidence_range(self):
        df = _make_synthetic_edges(200)
        out, _ = cluster_junctions_gmm(df)
        conf = out["AJ_morph_confidence"].dropna()
        assert (conf >= 0).all()
        assert (conf <= 1).all()

    def test_too_few_raises(self):
        df = _make_synthetic_edges(5)
        with pytest.raises(ValueError, match="Too few"):
            cluster_junctions_gmm(df)

    def test_full_features(self):
        df = _make_synthetic_edges(200)
        out, meta = cluster_junctions_gmm(df, blur_robust=False)
        assert len(meta["features_used"]) > len(BLUR_ROBUST_COLS)

    def test_k_range(self):
        df = _make_synthetic_edges(200)
        _, meta = cluster_junctions_gmm(df, k_range=(2, 5))
        assert 2 <= meta["optimal_k"] <= 5

    def test_cluster_sizes_sum(self):
        df = _make_synthetic_edges(200)
        out, meta = cluster_junctions_gmm(df)
        total = sum(meta["cluster_sizes"].values())
        assert total == meta["n_samples"]


class TestEvaluateClusterStability:
    def test_basic(self):
        df = _make_synthetic_edges(200)
        result = evaluate_cluster_stability(df, n_bootstrap=10)
        assert "mean_ari" in result
        assert result["mean_ari"] >= 0

    def test_too_few(self):
        df = _make_synthetic_edges(10)
        result = evaluate_cluster_stability(df)
        assert "error" in result


class TestAssignCanonicalName:
    def test_low_occ(self):
        assert _assign_canonical_name(0.05, 5, 0, 5) == "minimal"
        assert _assign_canonical_name(0.05, 50, 0, 5) == "fingers"

    def test_high_occ(self):
        assert _assign_canonical_name(0.85, 10, 0, 5) == "straight"
        assert _assign_canonical_name(0.85, 50, 0, 5) == "reticular"

    def test_mid_occ(self):
        name = _assign_canonical_name(0.4, 30, 0, 5)
        assert name in {"thick", "thick_to_reticular"}
