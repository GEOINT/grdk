# -*- coding: utf-8 -*-
"""
Tests for grdk.viewers.image_canvas — DisplaySettings, normalize_array,
ImageCanvas, and ImageCanvasThumbnail.

Pure-function tests (normalize_array) require only numpy.
Qt-dependent tests (ImageCanvas, array_to_qimage) are skipped when
no display is available.

Author
------
Claude Code (Anthropic)

Contributor
-----------
Steven Siebert

Created
-------
2026-02-06
"""

import numpy as np
import pytest

from grdk.viewers.image_canvas import (
    DisplaySettings,
    normalize_array,
    _get_colormaps,
)


# ---------------------------------------------------------------------------
# DisplaySettings
# ---------------------------------------------------------------------------

class TestDisplaySettings:
    def test_defaults(self):
        s = DisplaySettings()
        assert s.window_min is None
        assert s.window_max is None
        assert s.percentile_low == 0.0
        assert s.percentile_high == 100.0
        assert s.brightness == 0.0
        assert s.contrast == 1.0
        assert s.colormap == 'grayscale'
        assert s.band_index is None
        assert s.gamma == 1.0


# ---------------------------------------------------------------------------
# normalize_array — pure function tests (no Qt)
# ---------------------------------------------------------------------------

class TestNormalizeArray:
    def test_2d_minmax(self):
        """2D float array should map min→0, max→255."""
        arr = np.array([[0.0, 50.0], [100.0, 200.0]])
        result = normalize_array(arr)
        assert result.dtype == np.uint8
        assert result.shape == (2, 2)
        assert result[0, 0] == 0
        assert result[1, 1] == 255

    def test_complex_input(self):
        """Complex arrays should be converted via np.abs()."""
        arr = np.array([[3 + 4j, 0 + 0j]], dtype=np.complex64)
        result = normalize_array(arr)
        assert result.dtype == np.uint8
        assert result[0, 0] == 255  # |3+4j| = 5 → max → 255
        assert result[0, 1] == 0    # |0+0j| = 0 → min → 0

    def test_3d_rgb(self):
        """3-band array should pass through as RGB."""
        arr = np.zeros((4, 4, 3), dtype=np.float32)
        arr[:, :, 0] = 100  # R channel
        result = normalize_array(arr)
        assert result.dtype == np.uint8
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_3d_band_select(self):
        """band_index should select a specific band → 2D output."""
        arr = np.zeros((4, 4, 5), dtype=np.float32)
        arr[:, :, 2] = np.arange(16).reshape(4, 4).astype(np.float32)
        settings = DisplaySettings(band_index=2)
        result = normalize_array(arr, settings)
        assert result.ndim == 2  # Single band → grayscale
        assert result.max() == 255
        assert result.min() == 0

    def test_percentile_clipping(self):
        """Percentile windowing should clip outliers."""
        # Create a gradient with outliers at both ends
        arr = np.linspace(0, 100, 10000).reshape(100, 100).astype(np.float32)
        arr[0, 0] = 10000  # High outlier
        arr[0, 1] = -10000  # Low outlier

        settings = DisplaySettings(percentile_low=5, percentile_high=95)
        result = normalize_array(arr, settings)
        # Outliers should be clipped to 0/255
        assert result[0, 0] == 255
        assert result[0, 1] == 0
        # Mid-range pixel should be somewhere in the middle
        center = result[50, 50]
        assert 50 < center < 200

    def test_contrast_brightness(self):
        """Contrast > 1 should expand range, brightness shifts."""
        arr = np.array([[100.0, 200.0]], dtype=np.float64)
        # Default: no adjustment
        default_result = normalize_array(arr, DisplaySettings())
        # High contrast
        high_contrast = normalize_array(
            arr, DisplaySettings(contrast=2.0)
        )
        # The difference between pixels should be larger with higher contrast
        default_diff = abs(int(default_result[0, 1]) - int(default_result[0, 0]))
        contrast_diff = abs(int(high_contrast[0, 1]) - int(high_contrast[0, 0]))
        assert contrast_diff >= default_diff

    def test_gamma(self):
        """Gamma > 1 brightens midtones (power < 1), gamma < 1 darkens."""
        arr = np.array([[128.0]], dtype=np.float64)

        # gamma=2.0 → power=0.5 → brightens midtones
        bright = normalize_array(arr, DisplaySettings(
            gamma=2.0, window_min=0, window_max=255,
        ))
        # gamma=0.5 → power=2.0 → darkens midtones
        dark = normalize_array(arr, DisplaySettings(
            gamma=0.5, window_min=0, window_max=255,
        ))
        assert bright[0, 0] > dark[0, 0]

    def test_manual_window(self):
        """Manual window_min/max should clip to that range."""
        arr = np.array([[0.0, 50.0, 100.0, 200.0]])
        settings = DisplaySettings(window_min=50.0, window_max=100.0)
        result = normalize_array(arr, settings)
        assert result[0, 0] == 0    # Below window → 0
        assert result[0, 1] == 0    # At window_min → 0
        assert result[0, 2] == 255  # At window_max → 255
        assert result[0, 3] == 255  # Above window → 255

    def test_constant_array(self):
        """Constant array should not cause division by zero."""
        arr = np.full((4, 4), 42.0)
        result = normalize_array(arr)
        assert result.dtype == np.uint8
        # All zeros since vmax == vmin
        assert np.all(result == 0)

    def test_none_settings_uses_defaults(self):
        """Passing None for settings should use defaults."""
        arr = np.array([[0.0, 255.0]])
        result = normalize_array(arr, None)
        assert result.dtype == np.uint8
        assert result[0, 0] == 0
        assert result[0, 1] == 255


# ---------------------------------------------------------------------------
# Colormap tests
# ---------------------------------------------------------------------------

class TestColormaps:
    def test_grayscale_identity(self):
        """Grayscale colormap should produce 2D uint8 output."""
        arr = np.arange(256, dtype=np.float64).reshape(16, 16)
        result = normalize_array(arr, DisplaySettings(colormap='grayscale'))
        assert result.ndim == 2
        assert result.dtype == np.uint8

    def test_viridis_shape(self):
        """Viridis colormap should produce (H, W, 3) RGB output."""
        arr = np.arange(256, dtype=np.float64).reshape(16, 16)
        result = normalize_array(arr, DisplaySettings(colormap='viridis'))
        assert result.ndim == 3
        assert result.shape == (16, 16, 3)
        assert result.dtype == np.uint8

    def test_inferno_produces_rgb(self):
        """Inferno colormap should produce RGB."""
        arr = np.linspace(0, 100, 64).reshape(8, 8)
        result = normalize_array(arr, DisplaySettings(colormap='inferno'))
        assert result.shape == (8, 8, 3)

    def test_colormap_lut_shapes(self):
        """All colormaps should be 256×3 uint8 LUTs."""
        colormaps = _get_colormaps()
        for name, lut in colormaps.items():
            assert lut.shape == (256, 3), f"{name} LUT shape wrong"
            assert lut.dtype == np.uint8, f"{name} LUT dtype wrong"


# ---------------------------------------------------------------------------
# Qt-dependent tests (skip if no display)
# ---------------------------------------------------------------------------

try:
    from PySide6.QtGui import QImage
    from grdk.viewers.image_canvas import array_to_qimage, ImageCanvasThumbnail
    _QT_SKIP = False
except (ImportError, RuntimeError):
    _QT_SKIP = True


@pytest.mark.skipif(_QT_SKIP, reason="Qt not available")
class TestArrayToQImage:
    def test_returns_qimage(self):
        arr = np.random.rand(32, 32).astype(np.float32)
        qimg = array_to_qimage(arr)
        assert isinstance(qimg, QImage)
        assert qimg.width() == 32
        assert qimg.height() == 32

    def test_rgb_qimage(self):
        arr = np.random.rand(16, 16, 3).astype(np.float32)
        qimg = array_to_qimage(arr)
        assert isinstance(qimg, QImage)
        assert qimg.width() == 16


@pytest.mark.skipif(_QT_SKIP, reason="Qt not available")
class TestImageCanvasThumbnail:
    def test_set_array(self):
        thumb = ImageCanvasThumbnail(size=64)
        arr = np.random.rand(32, 32).astype(np.float32)
        thumb.set_array(arr)
        assert thumb._source is not None

    def test_fixed_size(self):
        thumb = ImageCanvasThumbnail(size=96)
        assert thumb.width() == 96
        assert thumb.height() == 96
