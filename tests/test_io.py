
import numpy as np
import pytest
import tifffile
from endopigraph.io import read_image, channel_index


class TestReadImage:
    def test_2d(self, tmp_path):
        img = np.random.rand(50, 50).astype(np.float32)
        p = tmp_path / "test.tif"
        tifffile.imwrite(str(p), img)
        arr, ch = read_image(p)
        assert arr.shape == (1, 50, 50)
        assert len(ch) == 1

    def test_3d_chw(self, tmp_path):
        img = np.random.rand(3, 50, 50).astype(np.float32)
        p = tmp_path / "test.tif"
        tifffile.imwrite(str(p), img)
        arr, ch = read_image(p)
        assert arr.shape == (3, 50, 50)
        assert len(ch) == 3

    def test_3d_hwc(self, tmp_path):
        img = np.random.rand(50, 50, 3).astype(np.float32)
        p = tmp_path / "test.tif"
        tifffile.imwrite(str(p), img)
        arr, ch = read_image(p)
        assert arr.shape[0] == 3
        assert arr.shape[1] == 50

    def test_float32_output(self, tmp_path):
        img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        p = tmp_path / "test.tif"
        tifffile.imwrite(str(p), img)
        arr, _ = read_image(p)
        assert arr.dtype == np.float32


class TestChannelIndex:
    def test_int(self):
        assert channel_index(0, ["a", "b"]) == 0
        assert channel_index(1, ["a", "b"]) == 1

    def test_name(self):
        assert channel_index("VE-cadherin", ["DAPI", "VE-cadherin"]) == 1

    def test_not_found(self):
        with pytest.raises(ValueError):
            channel_index("missing", ["a", "b"])

    def test_none_raises(self):
        with pytest.raises(ValueError):
            channel_index(None, ["a"])
