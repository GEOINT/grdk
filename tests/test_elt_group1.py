# tests/test_elt_group1.py
"""Unit tests for ELT Group 1:
- ELT-01 (#19): Async image loading worker
- ELT-02 (#20): Close image action & state reset
- ELT-03 (#21): 3-Band RGB color gun mapping
- ELT-06 (#24): False-color RGB presets
- ELT-24 (#42): Color inversion / negative display
- ELT-25 (#43): Expanded colormap suite
"""

import numpy as np
import pytest
from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QApplication

if QApplication.instance() is None:
    _app = QApplication(["--platform", "offscreen"])
else:
    _app = QApplication.instance()

from grdk.viewers.image_canvas import (
    DisplaySettings,
    normalize_array,
    AVAILABLE_COLORMAPS,
    _get_colormaps,
)
from grdk.viewers.band_info import BandInfo


class TestRGBColorGunsAndPresets:
    """Tests for ELT-03 (#21) and ELT-06 (#24)."""

    def test_rgb_bands_mapping_3d_array(self):
        # 10 bands with varying data
        data = np.zeros((10, 20, 20), dtype=np.float32)
        for i in range(10):
            data[i] = (i + 1) * 10.0 + np.linspace(0, 10, 20)

        # Select RGB guns: R=band 8, G=band 4, B=band 1
        settings = DisplaySettings(rgb_bands=(8, 4, 1))
        out = normalize_array(data, settings)

        assert out.ndim == 3
        # Channels-first (3, 20, 20)
        assert out.shape == (3, 20, 20)
        assert np.max(out[0]) > 0
        assert np.max(out[1]) > 0
        assert np.max(out[2]) > 0

    def test_rgb_bands_out_of_bounds_clamped(self):
        data = np.random.rand(5, 10, 10).astype(np.float32)
        settings = DisplaySettings(rgb_bands=(0, 2, 99))  # 99 should clamp to 4
        out = normalize_array(data, settings)
        assert out.ndim == 3
        assert out.shape == (3, 10, 10)


class TestColormapsAndInversion:
    """Tests for ELT-24 (#42) and ELT-25 (#43)."""

    def test_expanded_colormaps_exist(self):
        for cmap in ['jet', 'turbo', 'magma', 'bone', 'coolwarm']:
            assert cmap in AVAILABLE_COLORMAPS
            lut = _get_colormaps().get(cmap)
            assert lut is not None
            assert lut.shape == (256, 3)

    def test_invert_colormap_grayscale(self):
        data = np.array([[0, 255]], dtype=np.uint8)
        settings_normal = DisplaySettings(invert_colormap=False)
        settings_inverted = DisplaySettings(invert_colormap=True)

        out_normal = normalize_array(data, settings_normal)
        out_inverted = normalize_array(data, settings_inverted)

        assert out_normal[0, 0] == 0
        assert out_normal[0, 1] == 255
        assert out_inverted[0, 0] == 255
        assert out_inverted[0, 1] == 0

    def test_invert_colormap_rgb(self):
        data = np.linspace(0, 1, 300, dtype=np.float32).reshape(3, 10, 10)
        settings_normal = DisplaySettings(invert_colormap=False)
        settings_inverted = DisplaySettings(invert_colormap=True)

        out_normal = normalize_array(data, settings_normal)
        out_inverted = normalize_array(data, settings_inverted)

        # Inverted should invert colors (sum = 255)
        diff = np.abs(out_normal.astype(int) + out_inverted.astype(int) - 255)
        assert np.max(diff) <= 2


class TestCloseImageAction:
    """Tests for ELT-02 (#20): Close image action."""

    def test_geo_viewer_close_image(self):
        from grdk.viewers.geo_viewer import GeoImageViewer

        viewer = GeoImageViewer()
        dummy_arr = np.zeros((10, 10), dtype=np.float32)
        viewer.open_array(dummy_arr)
        assert viewer.canvas._source is not None

        viewer.close_image()
        assert viewer.canvas._source is None
        assert viewer.reader is None
        assert viewer.geolocation is None

    def test_dual_viewer_close_image(self):
        from grdk.viewers.dual_viewer import DualGeoViewer

        viewer = DualGeoViewer()
        dummy_arr = np.zeros((10, 10), dtype=np.float32)
        viewer.open_array(dummy_arr, pane=0)
        assert viewer._left_viewer.canvas._source is not None

        viewer.close_image(pane=0)
        assert viewer._left_viewer.canvas._source is None
        assert viewer._left_viewer.reader is None


class TestAsyncImageLoader:
    """Tests for ELT-01 (#19): Async loading worker."""

    def test_worker_instantiation_and_signals(self, qtbot):
        from grdk.viewers.main_window import ImageLoaderWorker

        worker = ImageLoaderWorker("dummy_nonexistent_path.tif", pane=0)
        assert isinstance(worker, QThread)
        assert hasattr(worker, "progress")
        assert hasattr(worker, "loaded")
        assert hasattr(worker, "failed")
