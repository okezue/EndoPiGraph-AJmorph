import numpy as np
import pytest
from endopigraph.ajmorph import (
    compute_threshold,
    interface_marker_features,
    heuristic_ajmorph_class,
    count_clusters_robust,
    count_skeleton_components,
    compute_skeleton_complexity,
    AJMORPH_CLASSES,
)


class TestComputeThreshold:
    def test_otsu(self):
        v = np.array([0, 0, 0, 100, 100, 100], dtype=float)
        t = compute_threshold(v, "otsu")
        assert 0 < t < 100

    def test_percentile(self):
        v = np.arange(100, dtype=float)
        t = compute_threshold(v, "percentile:50")
        assert abs(t - 49.5) < 1.0

    def test_numeric(self):
        v = np.array([1.0, 2.0])
        t = compute_threshold(v, "42.5")
        assert t == 42.5

    def test_empty(self):
        v = np.array([], dtype=float)
        t = compute_threshold(v, "otsu")
        assert np.isnan(t)

    def test_bad_method(self):
        with pytest.raises(ValueError):
            compute_threshold(np.array([1.0]), "bad_method")

    def test_nan_handling(self):
        v = np.array([1.0, np.nan, 2.0, np.nan, 3.0])
        t = compute_threshold(v, "percentile:50")
        assert np.isfinite(t)


class TestInterfaceMarkerFeatures:
    def test_basic(self, marker_image):
        mask = np.zeros_like(marker_image, dtype=bool)
        mask[8:12, 8:12] = True
        feats = interface_marker_features(marker_image, mask, threshold=100.0)
        assert np.isfinite(feats["mean"])
        assert np.isfinite(feats["occupancy"])
        assert 0 <= feats["occupancy"] <= 1

    def test_empty_mask(self, marker_image):
        mask = np.zeros_like(marker_image, dtype=bool)
        feats = interface_marker_features(marker_image, mask, threshold=100.0)
        assert np.isnan(feats["mean"])
        assert feats["cluster_count"] == 0

    def test_full_occupancy(self):
        img = np.full((10, 10), 200.0, dtype=np.float32)
        mask = np.ones((10, 10), dtype=bool)
        feats = interface_marker_features(img, mask, threshold=50.0)
        assert feats["occupancy"] == 1.0

    def test_zero_occupancy(self):
        img = np.full((10, 10), 10.0, dtype=np.float32)
        mask = np.ones((10, 10), dtype=bool)
        feats = interface_marker_features(img, mask, threshold=100.0)
        assert feats["occupancy"] == 0.0

    def test_skeleton_len_nonneg(self, marker_image):
        mask = np.ones_like(marker_image, dtype=bool)
        feats = interface_marker_features(marker_image, mask, threshold=50.0)
        assert feats["skeleton_len"] >= 0

    def test_returns_all_keys(self, marker_image):
        mask = np.ones_like(marker_image, dtype=bool)
        feats = interface_marker_features(marker_image, mask, threshold=100.0)
        expected = {"mean", "median", "max", "std", "occupancy",
                    "cluster_count", "cluster_count_cc", "cluster_density",
                    "cluster_area_mean", "skeleton_len",
                    "skeleton_components", "skeleton_endpoints",
                    "skeleton_branch_points", "complexity_score",
                    "thickness_proxy"}
        assert expected == set(feats.keys())

    def test_robust_vs_cc(self, marker_image):
        mask = np.ones_like(marker_image, dtype=bool)
        feats_r = interface_marker_features(marker_image, mask, threshold=50.0, use_robust_clustering=True)
        feats_c = interface_marker_features(marker_image, mask, threshold=50.0, use_robust_clustering=False)
        assert feats_r["cluster_count_cc"] == feats_c["cluster_count_cc"]


class TestCountClustersRobust:
    def test_no_mask(self):
        img = np.ones((10, 10), dtype=float)
        mask = np.zeros((10, 10), dtype=bool)
        assert count_clusters_robust(img, mask) == 0

    def test_single_cluster(self):
        img = np.zeros((20, 20), dtype=float)
        mask = np.zeros((20, 20), dtype=bool)
        img[5:15, 5:15] = 100.0
        mask[5:15, 5:15] = True
        n = count_clusters_robust(img, mask)
        assert n >= 1

    def test_uniform_intensity(self):
        img = np.full((10, 10), 50.0)
        mask = np.ones((10, 10), dtype=bool)
        n = count_clusters_robust(img, mask)
        assert n >= 1


class TestCountSkeletonComponents:
    def test_empty(self):
        assert count_skeleton_components(np.zeros((10, 10), dtype=bool)) == 0

    def test_single_line(self, binary_mask):
        n = count_skeleton_components(binary_mask)
        assert n >= 1


class TestComputeSkeletonComplexity:
    def test_empty(self):
        r = compute_skeleton_complexity(np.zeros((10, 10), dtype=bool))
        assert r["skeleton_components"] == 0
        assert r["endpoints"] == 0

    def test_line(self):
        m = np.zeros((20, 20), dtype=bool)
        m[10, 2:18] = True
        r = compute_skeleton_complexity(m)
        assert r["skeleton_components"] >= 1
        assert r["endpoints"] == 2

    def test_branch(self):
        m = np.zeros((30, 30), dtype=bool)
        m[15, 5:25] = True
        m[5:25, 15] = True
        r = compute_skeleton_complexity(m)
        assert r["branch_points"] >= 1


class TestHeuristicAjmorphClass:
    def test_minimal(self):
        f = {"occupancy": 0.01, "cluster_count": 0, "skeleton_len": 0, "thickness_proxy": float("nan")}
        assert heuristic_ajmorph_class(f) == "minimal"

    def test_straight(self):
        f = {"occupancy": 0.9, "cluster_count": 1, "skeleton_len": 50, "thickness_proxy": 2.0}
        assert heuristic_ajmorph_class(f) == "straight"

    def test_unknown_nan(self):
        f = {"occupancy": float("nan")}
        assert heuristic_ajmorph_class(f) == "unknown"

    def test_returns_valid_class(self):
        for occ in [0.05, 0.2, 0.5, 0.8]:
            for ncl in [0, 2, 5, 10]:
                f = {"occupancy": occ, "cluster_count": ncl,
                     "skeleton_len": 20, "thickness_proxy": 2.0}
                c = heuristic_ajmorph_class(f)
                assert c in AJMORPH_CLASSES

    def test_blur_robust_mode(self):
        f = {"occupancy": 0.5, "skeleton_components": 8,
             "skeleton_branch_points": 5, "complexity_score": 12,
             "skeleton_len": 50, "thickness_proxy": 2.0, "cluster_count": 3}
        c = heuristic_ajmorph_class(f, blur_robust=True)
        assert c in AJMORPH_CLASSES
