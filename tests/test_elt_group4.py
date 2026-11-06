# tests/test_elt_group4.py
"""Unit tests for ELT Group 4:
- ELT-12 (#30): SAR dynamic polarimetric decompositions (Sinclair, Pauli, Freeman-Durden)
- ELT-13 (#31): Spatial filtering & edge enhancement
- ELT-14 (#32): SAR spatial despeckle filters (Lee, Frost, Kuan, Boxcar)
- ELT-19 (#37): Pan-sharpening preview engine (Brovey, Mean)
- ELT-21 (#39): Graticule coordinate grid overlay
- ELT-22 (#40): Target bookmark & waypoint manager
- ELT-23 (#41): Vector markup & annotation layer
"""

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

if QApplication.instance() is None:
    _app = QApplication(["--platform", "offscreen"])
else:
    _app = QApplication.instance()

from grdk.widgets._pol_utils import (
    compute_sinclair_decomposition,
    compute_pauli_decomposition,
    compute_freeman_durden_decomposition,
)
from grdk.viewers.filters import (
    gaussian_blur,
    sharpen_unsharp_mask,
    sobel_gradient,
    laplacian_edge,
    lee_filter,
    frost_filter,
    kuan_filter,
    boxcar_filter,
)
from grdk.viewers.pan_sharpen import brovey_transform, simple_mean_fusion
from grdk.viewers.graticule import compute_graticule_lines
from grdk.viewers.bookmarks import BookmarkManagerWidget, Bookmark
from grdk.viewers.vector_markup import VectorMarkupLayer, MarkupItem


class TestPolarimetricDecompositions:
    """Tests for ELT-12 (#30): SAR Decompositions."""

    def test_sinclair_decomposition(self):
        s_hh = np.ones((20, 20), dtype=np.complex64) * 2.0
        s_hv = np.ones((20, 20), dtype=np.complex64) * 0.5
        s_vv = np.ones((20, 20), dtype=np.complex64) * 1.0

        rgb = compute_sinclair_decomposition(s_hh, s_hv, s_vv)
        assert rgb.shape == (3, 20, 20)
        # R = |HH - VV| = |2 - 1| = 1
        assert np.isclose(rgb[0, 0, 0], 1.0, atol=1e-3)
        # G = 2 * |HV| = 2 * 0.5 = 1
        assert np.isclose(rgb[1, 0, 0], 1.0, atol=1e-3)
        # B = |HH + VV| = |2 + 1| = 3
        assert np.isclose(rgb[2, 0, 0], 3.0, atol=1e-3)

    def test_pauli_decomposition(self):
        s_hh = np.ones((10, 10), dtype=np.complex64)
        s_hv = np.ones((10, 10), dtype=np.complex64) * 0.5
        s_vv = np.ones((10, 10), dtype=np.complex64)

        rgb = compute_pauli_decomposition(s_hh, s_hv, s_vv)
        assert rgb.shape == (3, 10, 10)
        # HH == VV, so R = |HH - VV|/sqrt(2) = 0
        assert np.isclose(rgb[0, 0, 0], 0.0, atol=1e-3)
        # B = |HH + VV|/sqrt(2) = 2/sqrt(2) = sqrt(2) ≈ 1.414
        assert np.isclose(rgb[2, 0, 0], np.sqrt(2.0), atol=1e-2)

    def test_freeman_durden_decomposition(self):
        s_hh = np.ones((10, 10), dtype=np.complex64) * 1.5
        s_hv = np.ones((10, 10), dtype=np.complex64) * 0.3
        s_vv = np.ones((10, 10), dtype=np.complex64) * 1.2

        rgb = compute_freeman_durden_decomposition(s_hh, s_hv, s_vv)
        assert rgb.shape == (3, 10, 10)
        assert np.all(rgb >= 0.0)


class TestSpatialAndSARFilters:
    """Tests for ELT-13 (#31) and ELT-14 (#32): Spatial and SAR filters."""

    def test_spatial_filters(self):
        arr = np.random.rand(30, 30).astype(np.float32)
        blurred = gaussian_blur(arr, sigma=1.0)
        assert blurred.shape == arr.shape

        sharpened = sharpen_unsharp_mask(arr, amount=1.5)
        assert sharpened.shape == arr.shape

        grad = sobel_gradient(arr)
        assert grad.shape == arr.shape

        lap = laplacian_edge(arr)
        assert lap.shape == arr.shape

    def test_sar_despeckle_filters(self):
        # Image with synthetic speckle (multiplicative noise)
        clean = np.ones((25, 25), dtype=np.float32) * 50.0
        speckle = np.random.exponential(1.0, size=(25, 25)).astype(np.float32)
        sar_img = clean * speckle

        lee = lee_filter(sar_img, window_size=5)
        assert lee.shape == sar_img.shape

        frost = frost_filter(sar_img, window_size=5)
        assert frost.shape == sar_img.shape

        kuan = kuan_filter(sar_img, window_size=5)
        assert kuan.shape == sar_img.shape

        box = boxcar_filter(sar_img, window_size=3)
        assert box.shape == sar_img.shape


class TestPanSharpening:
    """Tests for ELT-19 (#37): Pan-Sharpening."""

    def test_brovey_transform(self):
        pan = np.ones((40, 40), dtype=np.float32) * 2.0
        ms = np.ones((3, 40, 40), dtype=np.float32) * 1.0

        fused = brovey_transform(pan, ms)
        assert fused.shape == (3, 40, 40)
        # Ratio is 2.0 / 1.0 = 2.0, so fused = 1.0 * 2.0 = 2.0
        assert np.allclose(fused, 2.0, atol=1e-3)

    def test_simple_mean_fusion(self):
        pan = np.ones((20, 20), dtype=np.float32) * 4.0
        ms = np.ones((3, 20, 20), dtype=np.float32) * 2.0

        fused = simple_mean_fusion(pan, ms)
        assert fused.shape == (3, 20, 20)
        # Mean of 4.0 and 2.0 is 3.0
        assert np.allclose(fused, 3.0, atol=1e-3)


class TestGraticuleLines:
    """Tests for ELT-21 (#39): Graticule coordinate grid."""

    def test_compute_graticule_lines(self):
        # Bounding box around (lat: 30 to 35, lon: -110 to -105)
        grid = compute_graticule_lines(
            min_lat=30.0, max_lat=35.0,
            min_lon=-110.0, max_lon=-105.0,
            step_deg=1.0,
        )
        assert len(grid.lat_lines) >= 5
        assert len(grid.lon_lines) >= 5
        assert grid.lat_lines[0] == 30.0
        assert grid.lon_lines[0] == -110.0


class TestBookmarkManager:
    """Tests for ELT-22 (#40): Target bookmarks."""

    def test_bookmark_manager_crud(self, tmp_path):
        mgr = BookmarkManagerWidget()
        bm = Bookmark(name="Target Alpha", lat=34.5, lon=-118.2, row=100, col=200, zoom=2.5, notes="Facility check")
        mgr.add_bookmark(bm)
        assert len(mgr.bookmarks) == 1
        assert mgr.bookmarks[0].name == "Target Alpha"

        # Export and import JSON
        json_file = tmp_path / "targets.json"
        mgr.export_bookmarks(str(json_file))
        assert json_file.exists()

        mgr.clear()
        assert len(mgr.bookmarks) == 0

        mgr.import_bookmarks(str(json_file))
        assert len(mgr.bookmarks) == 1
        assert mgr.bookmarks[0].lat == 34.5


class TestVectorMarkup:
    """Tests for ELT-23 (#41): Vector markup."""

    def test_vector_markup_items(self):
        layer = VectorMarkupLayer()
        item = MarkupItem(
            shape_type="callout",
            coordinates=[(10.0, 10.0), (30.0, 40.0)],
            label="SAM Site A",
            color="#FF0000",
            line_width=2,
        )
        layer.add_item(item)
        assert len(layer.items) == 1
        assert layer.items[0].label == "SAM Site A"
        assert layer.items[0].shape_type == "callout"
