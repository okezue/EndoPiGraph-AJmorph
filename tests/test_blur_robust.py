import numpy as np
import pytest
from endopigraph.blur_robust import (
    estimate_blur_score,
    detect_blur,
    correct_blur,
    compute_blur_robust_features,
    blur_robust_classifier,
    get_recommended_metrics,
    STABLE_METRICS,
)


class TestEstimateBlurScore:
    def test_sharp_image(self):
        img = np.random.RandomState(0).rand(100, 100).astype(np.float32) * 255
        score = estimate_blur_score(img)
        assert score > 0

    def test_blurry_lower(self):
        rng = np.random.RandomState(42)
        sharp = rng.rand(100, 100).astype(np.float32) * 255
        from scipy.ndimage import gaussian_filter
        blurry = gaussian_filter(sharp, sigma=5)
        s_sharp = estimate_blur_score(sharp)
        s_blurry = estimate_blur_score(blurry)
        assert s_sharp > s_blurry

    def test_3d_raises(self):
        with pytest.raises(ValueError):
            estimate_blur_score(np.zeros((10, 10, 3)))


class TestDetectBlur:
    def test_returns_tuple(self):
        img = np.random.rand(50, 50).astype(np.float32) * 255
        result = detect_blur(img)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)


class TestCorrectBlur:
    def test_same_shape(self):
        img = np.random.rand(50, 50).astype(np.float32) * 255
        out = correct_blur(img)
        assert out.shape == img.shape

    def test_same_dtype(self):
        img = (np.random.rand(50, 50) * 255).astype(np.float32)
        out = correct_blur(img)
        assert out.dtype == img.dtype


class TestComputeBlurRobustFeatures:
    def test_basic(self):
        marker = np.random.rand(20, 20).astype(np.float32) * 200
        mask = np.ones((20, 20), dtype=bool)
        feats = compute_blur_robust_features(marker, mask, threshold=100.0)
        assert "mean_intensity" in feats
        assert "occupancy" in feats
        assert "skeleton_len" in feats

    def test_empty_mask(self):
        marker = np.random.rand(20, 20).astype(np.float32) * 200
        mask = np.zeros((20, 20), dtype=bool)
        feats = compute_blur_robust_features(marker, mask, threshold=100.0)
        assert np.isnan(feats["mean_intensity"])

    def test_occupancy_range(self):
        marker = np.full((10, 10), 200.0, dtype=np.float32)
        mask = np.ones((10, 10), dtype=bool)
        feats = compute_blur_robust_features(marker, mask, threshold=100.0)
        assert 0 <= feats["occupancy"] <= 1


class TestBlurRobustClassifier:
    def test_minimal(self):
        f = {"occupancy": 0.01, "skeleton_len": 0, "total_area": 0}
        assert blur_robust_classifier(f) == "minimal"

    def test_straight(self):
        f = {"occupancy": 0.8, "skeleton_len": 10, "total_area": 500}
        assert blur_robust_classifier(f) == "straight"

    def test_unknown_nan(self):
        f = {"occupancy": float("nan")}
        assert blur_robust_classifier(f) == "unknown"

    def test_all_classes_valid(self):
        valid = {"minimal", "punctate", "fingers", "discontinuous",
                 "thick_to_reticular", "thick", "reticular", "straight", "unknown"}
        for occ in [0.05, 0.15, 0.3, 0.6, 0.8]:
            for sk_d in [0.05, 0.15, 0.25, 0.35]:
                area = 100
                sk = int(sk_d * area)
                f = {"occupancy": occ, "skeleton_len": sk, "total_area": area}
                c = blur_robust_classifier(f)
                assert c in valid


class TestGetRecommendedMetrics:
    def test_sharp(self):
        m = get_recommended_metrics(150)
        assert all(s in m for s in STABLE_METRICS)
        assert "cluster_count" in m

    def test_moderate(self):
        m = get_recommended_metrics(75)
        assert all(s in m for s in STABLE_METRICS)
        assert "cluster_count" not in m

    def test_heavy_blur(self):
        m = get_recommended_metrics(10)
        assert m == STABLE_METRICS
