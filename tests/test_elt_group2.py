# tests/test_elt_group2.py
"""Unit tests for ELT Group 2:
- ELT-04 (#22): Spectral Profile Tool
- ELT-05 (#23): Interactive DRA & Histogram Stretch Widget
- ELT-11 (#29): Comprehensive HUD & DMS Coordinates
- ELT-18 (#36): Spectral index and band math calculator
"""

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

if QApplication.instance() is None:
    _app = QApplication(["--platform", "offscreen"])
else:
    _app = QApplication.instance()

from grdk.viewers.coordinate_bar import CoordinateBar
from grdk.viewers.spectral_profile import SpectralProfileWidget, extract_spectral_profile
from grdk.viewers.histogram_widget import HistogramStretchWidget, compute_histogram_bins
from grdk.viewers.band_math import compute_spectral_index, BandMathDialog


class TestSpectralProfileTool:
    """Tests for ELT-04 (#22): Spectral Profile Tool."""

    def test_extract_spectral_profile_3d_array(self):
        # 50-band HSI mock
        cube = np.random.rand(50, 20, 20).astype(np.float32)
        wavelengths = [400.0 + i * 10.0 for i in range(50)]

        profile = extract_spectral_profile(cube, row=5, col=10, wavelengths=wavelengths)
        assert len(profile.values) == 50
        assert len(profile.wavelengths) == 50
        assert profile.row == 5
        assert profile.col == 10
        assert profile.wavelengths[0] == 400.0
        assert profile.wavelengths[-1] == 890.0

    def test_spectral_profile_widget_plot(self):
        widget = SpectralProfileWidget()
        cube = np.random.rand(30, 10, 10).astype(np.float32)
        widget.set_data(cube, wavelengths=list(range(400, 700, 10)))
        widget.probe_pixel(row=2, col=3)

        assert widget.current_profile is not None
        assert len(widget.current_profile.values) == 30
        assert widget.get_peak_wavelength() is not None


class TestHistogramStretchWidget:
    """Tests for ELT-05 (#23): Histogram & DRA."""

    def test_compute_histogram_bins(self):
        arr = np.linspace(0, 100, 1000).reshape(10, 100).astype(np.float32)
        counts, bin_edges = compute_histogram_bins(arr, num_bins=50)
        assert len(counts) == 50
        assert len(bin_edges) == 51
        assert np.sum(counts) == 1000

    def test_histogram_widget_presets(self):
        widget = HistogramStretchWidget()
        arr = np.linspace(10, 200, 500).reshape(50, 10).astype(np.float32)
        widget.set_array(arr)

        # Apply 2% - 98% preset
        low, high = widget.get_percentile_preset(2.0, 98.0)
        assert low > 10.0
        assert high < 200.0
        assert low < high


class TestCoordinateHUD:
    """Tests for ELT-11 (#29): Comprehensive HUD & DMS Coordinates."""

    def test_dd_to_dms_conversion(self):
        # 38.8977° N, 77.0365° W (Washington DC)
        lat_dms = CoordinateBar.dd_to_dms(38.8977, is_lat=True)
        lon_dms = CoordinateBar.dd_to_dms(-77.0365, is_lat=False)

        assert "38°" in lat_dms and "N" in lat_dms
        assert "77°" in lon_dms and "W" in lon_dms

    def test_coordinate_bar_hud_multiband_formatting(self):
        bar = CoordinateBar()
        multiband_val = np.array([120.5, 45.2, 88.9], dtype=np.float32)
        bar._format_value(multiband_val)
        assert "120.5" in bar._value_label.text()
        assert "45.2" in bar._value_label.text()
        assert "88.9" in bar._value_label.text()


class TestBandMathAndIndices:
    """Tests for ELT-18 (#36): Spectral index and band math."""

    def test_compute_ndvi(self):
        # Mock NIR (band 3) and Red (band 2)
        cube = np.zeros((4, 10, 10), dtype=np.float32)
        cube[2] = 0.2  # Red
        cube[3] = 0.6  # NIR

        ndvi = compute_spectral_index(cube, index_type="NDVI", nir_idx=3, red_idx=2)
        # NDVI = (0.6 - 0.2) / (0.6 + 0.2) = 0.4 / 0.8 = 0.5
        assert np.allclose(ndvi, 0.5, atol=1e-4)

    def test_compute_ndwi(self):
        cube = np.zeros((4, 10, 10), dtype=np.float32)
        cube[1] = 0.5  # Green
        cube[3] = 0.1  # NIR

        ndwi = compute_spectral_index(cube, index_type="NDWI", green_idx=1, nir_idx=3)
        # NDWI = (0.5 - 0.1) / (0.5 + 0.1) = 0.4 / 0.6 = 0.6667
        assert np.allclose(ndwi, 0.6667, atol=1e-3)
