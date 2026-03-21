import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def simple_labels():
    lab = np.zeros((20, 20), dtype=np.int32)
    lab[0:10, 0:10] = 1
    lab[0:10, 10:20] = 2
    lab[10:20, 0:10] = 3
    lab[10:20, 10:20] = 4
    return lab


@pytest.fixture
def single_cell_labels():
    lab = np.zeros((20, 20), dtype=np.int32)
    lab[2:18, 2:18] = 1
    return lab


@pytest.fixture
def marker_image():
    img = np.zeros((20, 20), dtype=np.float32)
    img[9:11, :] = 200.0
    img[:, 9:11] = 200.0
    img[5:15, 5:15] += 50.0
    return img


@pytest.fixture
def binary_mask():
    m = np.zeros((30, 30), dtype=bool)
    m[5:25, 10:20] = True
    return m


@pytest.fixture
def cells_df():
    return pd.DataFrame({
        "cell_id": [1, 2, 3, 4],
        "area_px": [100, 100, 100, 100],
        "cy": [5.0, 5.0, 15.0, 15.0],
        "cx": [5.0, 15.0, 5.0, 15.0],
    })


@pytest.fixture
def edges_df():
    return pd.DataFrame({
        "cell_i": [1, 1, 2, 3],
        "cell_j": [2, 3, 4, 4],
        "contact_px": [10, 10, 10, 10],
        "has_AJ": [True, False, True, True],
    })


@pytest.fixture
def semantic_cornea_mask():
    m = np.full((100, 100), 3, dtype=np.uint8)
    m[10:45, 10:45] = 1
    m[10:45, 55:90] = 1
    m[55:90, 10:45] = 1
    m[55:90, 55:90] = 1
    m[10:45, 45:55] = 2
    m[55:90, 45:55] = 2
    m[45:55, 10:90] = 2
    return m
