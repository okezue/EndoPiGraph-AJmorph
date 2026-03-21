import numpy as np
import pytest
from endopigraph.interfaces import (
    compute_interfaces,
    build_interface_mask_from_coords,
    local_interface_mask,
    _pair_code,
)


class TestPairCode:
    def test_basic(self):
        codes = _pair_code(np.array([1, 2]), np.array([3, 4]))
        assert codes[0] == (1 << 32) | 3
        assert codes[1] == (2 << 32) | 4

    def test_single(self):
        c = _pair_code(np.array([5]), np.array([10]))
        assert c[0] == (5 << 32) | 10


class TestComputeInterfaces:
    def test_four_cells(self, simple_labels):
        iface = compute_interfaces(simple_labels, min_contact_px=1)
        edges = iface.edges
        assert len(edges) > 0
        pairs = set(zip(edges["cell_i"], edges["cell_j"]))
        assert (1, 2) in pairs
        assert (1, 3) in pairs
        assert (2, 4) in pairs
        assert (3, 4) in pairs

    def test_no_diagonal(self, simple_labels):
        iface = compute_interfaces(simple_labels, min_contact_px=1)
        pairs = set(zip(iface.edges["cell_i"], iface.edges["cell_j"]))
        assert (1, 4) not in pairs
        assert (2, 3) not in pairs

    def test_single_cell_no_edges(self, single_cell_labels):
        iface = compute_interfaces(single_cell_labels)
        assert len(iface.edges) == 0

    def test_empty_mask(self):
        lab = np.zeros((10, 10), dtype=np.int32)
        iface = compute_interfaces(lab)
        assert len(iface.edges) == 0

    def test_min_contact_filter(self, simple_labels):
        iface_low = compute_interfaces(simple_labels, min_contact_px=1)
        iface_high = compute_interfaces(simple_labels, min_contact_px=100)
        assert len(iface_low.edges) >= len(iface_high.edges)

    def test_boundary_coords_populated(self, simple_labels):
        iface = compute_interfaces(simple_labels, min_contact_px=1)
        for _, row in iface.edges.iterrows():
            key = (int(row["cell_i"]), int(row["cell_j"]))
            assert key in iface.boundary_coords
            assert len(iface.boundary_coords[key]) > 0

    def test_boundary_mask_shape(self, simple_labels):
        iface = compute_interfaces(simple_labels, min_contact_px=1)
        assert iface.all_boundary_mask.shape == simple_labels.shape

    def test_2d_required(self):
        with pytest.raises(ValueError):
            compute_interfaces(np.zeros((5, 5, 5), dtype=np.int32))

    def test_contact_px_positive(self, simple_labels):
        iface = compute_interfaces(simple_labels, min_contact_px=1)
        assert (iface.edges["contact_px"] > 0).all()

    def test_symmetry(self, simple_labels):
        iface = compute_interfaces(simple_labels, min_contact_px=1)
        for _, row in iface.edges.iterrows():
            assert row["cell_i"] < row["cell_j"]


class TestBuildInterfaceMask:
    def test_basic(self):
        coords = np.array([[5, 5], [5, 6], [5, 7]], dtype=np.int32)
        mask = build_interface_mask_from_coords((20, 20), coords, dilate_px=0)
        assert mask[5, 5] and mask[5, 6] and mask[5, 7]
        assert not mask[0, 0]

    def test_dilation(self):
        coords = np.array([[10, 10]], dtype=np.int32)
        m0 = build_interface_mask_from_coords((20, 20), coords, dilate_px=0)
        m2 = build_interface_mask_from_coords((20, 20), coords, dilate_px=2)
        assert m2.sum() > m0.sum()

    def test_empty_coords(self):
        coords = np.zeros((0, 2), dtype=np.int32)
        mask = build_interface_mask_from_coords((10, 10), coords)
        assert mask.sum() == 0


class TestLocalInterfaceMask:
    def test_basic(self):
        coords = np.array([[10, 10], [10, 11], [10, 12]], dtype=np.int32)
        bbox, m = local_interface_mask(coords, margin=2, shape=(20, 20))
        r0, r1, c0, c1 = bbox
        assert r0 <= 10 and r1 > 10
        assert c0 <= 10 and c1 > 12
        assert m.any()

    def test_empty(self):
        coords = np.zeros((0, 2), dtype=np.int32)
        bbox, m = local_interface_mask(coords, margin=2, shape=(20, 20))
        assert m.size == 0
