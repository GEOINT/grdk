# -*- coding: utf-8 -*-
"""
Tests for grdk.viewers.dual_viewer — SyncController, SyncBar, DualGeoViewer,
and overlap computation utilities.

Pure-function tests require only numpy.  Qt-dependent tests are skipped
when no display is available.

Author
------
Claude Code (Anthropic)

Created
-------
2026-02-20
"""

from typing import Any, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from grdk.viewers.dual_viewer import compute_geo_bounds, compute_overlap


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

class SyntheticReader:
    """Minimal ImageReader-like object backed by a numpy array."""

    def __init__(self, rows: int, cols: int, dtype: type = np.float32) -> None:
        self._arr = np.random.rand(rows, cols).astype(dtype) * 255
        self.metadata = {'rows': rows, 'cols': cols, 'dtype': str(dtype)}

    def read_chip(
        self, row_start: int, row_end: int, col_start: int, col_end: int,
        bands: Any = None,
    ) -> np.ndarray:
        return self._arr[row_start:row_end, col_start:col_end].copy()

    def get_shape(self) -> Tuple[int, ...]:
        return self._arr.shape

    def get_dtype(self) -> np.dtype:
        return self._arr.dtype

    def read_full(self, bands: Any = None) -> np.ndarray:
        return self._arr.copy()

    def close(self) -> None:
        pass


class MultibandSyntheticReader:
    """Multiband reader that returns channels-first (C, H, W) data.

    Simulates a multi-polarization SAR image with optional complex dtype.
    get_shape() returns (rows, cols, bands) per grdl convention.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        bands: int = 4,
        complex_dtype: bool = False,
    ) -> None:
        dtype = np.complex64 if complex_dtype else np.float32
        if complex_dtype:
            self._arr = (
                np.random.rand(bands, rows, cols).astype(np.float32)
                + 1j * np.random.rand(bands, rows, cols).astype(np.float32)
            )
        else:
            self._arr = np.random.rand(bands, rows, cols).astype(np.float32)
        self._bands = bands
        self._rows = rows
        self._cols = cols
        self.metadata = {
            'rows': rows, 'cols': cols, 'bands': bands,
            'dtype': str(dtype),
        }

    def read_chip(
        self, row_start: int, row_end: int, col_start: int, col_end: int,
        bands: Any = None,
    ) -> np.ndarray:
        return self._arr[:, row_start:row_end, col_start:col_end].copy()

    def get_shape(self) -> Tuple[int, ...]:
        return (self._rows, self._cols, self._bands)

    def get_dtype(self) -> np.dtype:
        return self._arr.dtype

    def read_full(self, bands: Any = None) -> np.ndarray:
        return self._arr.copy()

    def close(self) -> None:
        pass


class MockGeolocation:
    """Mock geolocation that maps pixel coords to lat/lon linearly.

    Maps (row=0, col=0) to (lat_start, lon_start) and
    (row=rows-1, col=cols-1) to (lat_end, lon_end).
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        lat_start: float,
        lat_end: float,
        lon_start: float,
        lon_end: float,
    ) -> None:
        self._rows = rows
        self._cols = cols
        self._lat_start = lat_start
        self._lat_end = lat_end
        self._lon_start = lon_start
        self._lon_end = lon_end

    def image_to_latlon(
        self, row: float, col: float,
    ) -> Tuple[float, float]:
        lat = self._lat_start + (self._lat_end - self._lat_start) * row / max(1, self._rows - 1)
        lon = self._lon_start + (self._lon_end - self._lon_start) * col / max(1, self._cols - 1)
        return (lat, lon)

    def latlon_to_image(
        self, lat: float, lon: float,
    ) -> Tuple[float, float]:
        row = (lat - self._lat_start) / (self._lat_end - self._lat_start) * (self._rows - 1)
        col = (lon - self._lon_start) / (self._lon_end - self._lon_start) * (self._cols - 1)
        return (row, col)


# ---------------------------------------------------------------------------
# compute_geo_bounds (no Qt)
# ---------------------------------------------------------------------------

class TestComputeGeoBounds:
    def test_basic(self):
        geo = MockGeolocation(100, 200, 30.0, 31.0, -90.0, -89.0)
        bounds = compute_geo_bounds(geo, 100, 200)
        assert bounds is not None
        lat_min, lat_max, lon_min, lon_max = bounds
        assert abs(lat_min - 30.0) < 0.01
        assert abs(lat_max - 31.0) < 0.01
        assert abs(lon_min - (-90.0)) < 0.01
        assert abs(lon_max - (-89.0)) < 0.01

    def test_returns_none_on_error(self):
        geo = MagicMock()
        geo.image_to_latlon.side_effect = Exception("transform error")
        assert compute_geo_bounds(geo, 100, 100) is None


# ---------------------------------------------------------------------------
# compute_overlap (no Qt)
# ---------------------------------------------------------------------------

class TestComputeOverlap:
    def test_full_overlap(self):
        """Two identical geo extents should produce full overlap."""
        left_geo = MockGeolocation(100, 100, 30.0, 31.0, -90.0, -89.0)
        right_geo = MockGeolocation(100, 100, 30.0, 31.0, -90.0, -89.0)
        overlap = compute_overlap(left_geo, (100, 100), right_geo, (100, 100))
        assert overlap is not None
        lat_min, lat_max, lon_min, lon_max = overlap
        assert abs(lat_min - 30.0) < 0.01
        assert abs(lat_max - 31.0) < 0.01

    def test_partial_overlap(self):
        """Partially overlapping extents."""
        left_geo = MockGeolocation(100, 100, 30.0, 31.0, -90.0, -89.0)
        right_geo = MockGeolocation(100, 100, 30.5, 31.5, -89.5, -88.5)
        overlap = compute_overlap(left_geo, (100, 100), right_geo, (100, 100))
        assert overlap is not None
        lat_min, lat_max, lon_min, lon_max = overlap
        assert abs(lat_min - 30.5) < 0.01
        assert abs(lat_max - 31.0) < 0.01
        assert abs(lon_min - (-89.5)) < 0.01
        assert abs(lon_max - (-89.0)) < 0.01

    def test_no_overlap(self):
        """Non-overlapping extents should return None."""
        left_geo = MockGeolocation(100, 100, 30.0, 31.0, -90.0, -89.0)
        right_geo = MockGeolocation(100, 100, 40.0, 41.0, -80.0, -79.0)
        assert compute_overlap(
            left_geo, (100, 100), right_geo, (100, 100),
        ) is None

    def test_none_geolocation(self):
        """None geolocation should return None."""
        geo = MockGeolocation(100, 100, 30.0, 31.0, -90.0, -89.0)
        assert compute_overlap(None, (100, 100), geo, (100, 100)) is None
        assert compute_overlap(geo, (100, 100), None, (100, 100)) is None


# ---------------------------------------------------------------------------
# Qt-dependent tests
# ---------------------------------------------------------------------------

try:
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from grdk.viewers.dual_viewer import (
        DualGeoViewer,
        SyncBar,
        SyncController,
    )
    from grdk.viewers.tiled_canvas import TiledImageCanvas

    _QT_SKIP = False
    if QApplication.instance() is None:
        _app = QApplication([])
except (ImportError, RuntimeError):
    _QT_SKIP = True


@pytest.mark.skipif(_QT_SKIP, reason="Qt not available")
class TestSyncController:
    def test_init(self):
        ctrl = SyncController()
        assert ctrl.sync_mode == "pixel"
        assert ctrl.enabled is True

    def test_set_sync_mode(self):
        ctrl = SyncController()
        ctrl.set_sync_mode("geo")
        assert ctrl.sync_mode == "geo"
        ctrl.set_sync_mode("none")
        assert ctrl.enabled is False

    def test_invalid_mode_raises(self):
        ctrl = SyncController()
        with pytest.raises(ValueError):
            ctrl.set_sync_mode("invalid")

    def test_toggle_enabled(self):
        ctrl = SyncController()
        ctrl.set_enabled(False)
        assert ctrl.sync_mode == "none"
        ctrl.set_enabled(True)
        assert ctrl.sync_mode == "pixel"

    def test_set_canvases(self):
        ctrl = SyncController()
        left = TiledImageCanvas()
        right = TiledImageCanvas()
        ctrl.set_canvases(left, right)
        # Should not raise

    def test_reentrancy_guard(self):
        """Sync should not trigger infinite loops."""
        ctrl = SyncController()
        left = TiledImageCanvas()
        right = TiledImageCanvas()
        ctrl.set_canvases(left, right)

        # Load small arrays so the canvases have content
        arr = np.random.rand(50, 50).astype(np.float32)
        left.set_array(arr)
        right.set_array(arr)

        # Trigger viewport change — should not hang or recurse
        left.viewport_changed.emit()
        right.viewport_changed.emit()

    def test_geo_overlap_detection(self):
        ctrl = SyncController()
        left_geo = MockGeolocation(100, 100, 30.0, 31.0, -90.0, -89.0)
        right_geo = MockGeolocation(100, 100, 30.5, 31.5, -89.5, -88.5)
        ctrl.set_geolocations(left_geo, (100, 100), right_geo, (100, 100))
        assert ctrl.get_overlap() is not None

    def test_no_geo_no_overlap(self):
        ctrl = SyncController()
        ctrl.set_geolocations(None, (100, 100), None, (100, 100))
        assert ctrl.get_overlap() is None

    def test_geo_mode_fallback_without_overlap(self):
        """Geo mode should fall back to pixel if no overlap."""
        ctrl = SyncController()
        ctrl.set_sync_mode("geo")
        left_geo = MockGeolocation(100, 100, 30.0, 31.0, -90.0, -89.0)
        right_geo = MockGeolocation(100, 100, 50.0, 51.0, -70.0, -69.0)
        ctrl.set_geolocations(left_geo, (100, 100), right_geo, (100, 100))
        assert ctrl.sync_mode == "pixel"  # Fell back


@pytest.mark.skipif(_QT_SKIP, reason="Qt not available")
class TestSyncBar:
    def test_init(self):
        bar = SyncBar()
        assert bar.width() == 32

    def test_set_overlap_available(self):
        bar = SyncBar()
        bar.set_overlap_available(True)
        assert bar._crop_btn.isEnabled()
        bar.set_overlap_available(False)
        assert not bar._crop_btn.isEnabled()

    def test_set_cropped(self):
        bar = SyncBar()
        bar.set_cropped(True)
        # Use isVisibleTo(parent) since the bar is not shown on screen
        assert not bar._crop_btn.isVisibleTo(bar)
        assert bar._reset_btn.isVisibleTo(bar)
        bar.set_cropped(False)
        assert bar._crop_btn.isVisibleTo(bar)
        assert not bar._reset_btn.isVisibleTo(bar)


@pytest.mark.skipif(_QT_SKIP, reason="Qt not available")
class TestDualGeoViewer:
    def test_starts_in_single_mode(self):
        viewer = DualGeoViewer()
        assert viewer.mode == "single"
        assert viewer.active_pane == 0

    def test_switch_to_dual(self):
        viewer = DualGeoViewer()
        viewer.set_mode("dual")
        assert viewer.mode == "dual"
        # Use isVisibleTo since the viewer is not shown on screen
        assert viewer.right_viewer.isVisibleTo(viewer)

    def test_switch_back_to_single(self):
        viewer = DualGeoViewer()
        viewer.set_mode("dual")
        viewer.set_mode("single")
        assert viewer.mode == "single"
        assert not viewer.right_viewer.isVisibleTo(viewer)

    def test_invalid_mode_raises(self):
        viewer = DualGeoViewer()
        with pytest.raises(ValueError):
            viewer.set_mode("triple")

    def test_set_array_left(self):
        viewer = DualGeoViewer()
        arr = np.random.rand(50, 50).astype(np.float32)
        viewer.set_array(arr, pane=0)
        assert viewer.left_viewer.canvas.source_array is not None

    def test_set_array_right(self):
        viewer = DualGeoViewer()
        viewer.set_mode("dual")
        arr = np.random.rand(50, 50).astype(np.float32)
        viewer.set_array(arr, pane=1)
        assert viewer.right_viewer.canvas.source_array is not None

    def test_active_canvas_property(self):
        viewer = DualGeoViewer()
        # In single mode, active_canvas should be left canvas
        assert viewer.active_canvas is viewer.left_viewer.canvas

    def test_canvas_backward_compat(self):
        """viewer.canvas should alias active_canvas."""
        viewer = DualGeoViewer()
        assert viewer.canvas is viewer.active_canvas

    def test_load_vector(self):
        """load_vector should not raise (even without geolocation)."""
        import json
        import os
        import tempfile

        viewer = DualGeoViewer()
        arr = np.random.rand(50, 50).astype(np.float32)
        viewer.set_array(arr, pane=0)

        data = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [10, 20]},
                "properties": {},
            }],
        }
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.geojson', delete=False,
        ) as f:
            json.dump(data, f)
            tmppath = f.name

        try:
            viewer.load_vector(tmppath, pane=0)
        finally:
            os.unlink(tmppath)

    def test_clear_vectors(self):
        viewer = DualGeoViewer()
        viewer.clear_vectors()  # Should not raise

    def test_export_view(self):
        import os
        import tempfile

        viewer = DualGeoViewer()
        arr = np.random.rand(50, 50).astype(np.float32)
        viewer.set_array(arr, pane=0)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            tmppath = f.name

        try:
            viewer.export_view(tmppath, pane=0)
            assert os.path.exists(tmppath)
        finally:
            os.unlink(tmppath)

    def test_set_mode_single_resets_active_pane(self):
        """Switching to single should reset active pane to 0."""
        viewer = DualGeoViewer()
        viewer.set_mode("dual")
        viewer._set_active_pane(1)
        assert viewer.active_pane == 1
        viewer.set_mode("single")
        assert viewer.active_pane == 0


@pytest.mark.skipif(_QT_SKIP, reason="Qt not available")
class TestMultibandPrompt:
    """Test the multiband dual-display prompt in ViewerMainWindow."""

    def test_multiband_prompt_yes(self):
        """Accepting the multiband prompt should switch to dual mode."""
        from unittest.mock import patch
        from grdk.viewers.main_window import ViewerMainWindow

        window = ViewerMainWindow()
        arr = np.random.rand(3, 50, 50).astype(np.float32)
        window.set_array(arr, pane=0)

        # Accept the dialog and mock open_file on the DualGeoViewer so
        # the second open (pane=1) doesn't try to read a fake path.
        with patch(
            "grdk.viewers.main_window.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ), patch.object(
            window._viewer, "open_file",
        ):
            window._offer_dual_for_multiband("/fake/path.tif")

        assert window._viewer.mode == "dual"

    def test_multiband_prompt_no(self):
        """Declining the multiband prompt should stay in single mode."""
        from unittest.mock import patch
        from grdk.viewers.main_window import ViewerMainWindow

        window = ViewerMainWindow()
        arr = np.random.rand(3, 50, 50).astype(np.float32)
        window.set_array(arr, pane=0)

        with patch(
            "grdk.viewers.main_window.QMessageBox.question",
            return_value=QMessageBox.StandardButton.No,
        ):
            window._offer_dual_for_multiband("/fake/path.tif")

        assert window._viewer.mode == "single"

    def test_single_band_no_prompt(self):
        """Single-band images should not trigger the prompt."""
        from unittest.mock import patch
        from grdk.viewers.main_window import ViewerMainWindow

        window = ViewerMainWindow()
        arr = np.random.rand(50, 50).astype(np.float32)
        window.set_array(arr, pane=0)

        with patch(
            "grdk.viewers.main_window.QMessageBox.question",
        ) as mock_q:
            window._offer_dual_for_multiband("/fake/path.tif")
            mock_q.assert_not_called()

    def test_already_dual_no_prompt(self):
        """If already in dual mode, no prompt should appear."""
        from unittest.mock import patch
        from grdk.viewers.main_window import ViewerMainWindow

        window = ViewerMainWindow()
        arr = np.random.rand(3, 50, 50).astype(np.float32)
        window.set_array(arr, pane=0)
        window._viewer.set_mode("dual")

        with patch(
            "grdk.viewers.main_window.QMessageBox.question",
        ) as mock_q:
            window._offer_dual_for_multiband("/fake/path.tif")
            mock_q.assert_not_called()

    def test_multiband_prompt_sets_band_selectors(self):
        """Accepting prompt should set band 0/1 on left/right controls."""
        from unittest.mock import patch
        from grdk.viewers.main_window import ViewerMainWindow

        window = ViewerMainWindow()
        arr = np.random.rand(3, 50, 50).astype(np.float32)
        window.set_array(arr, pane=0)

        with patch(
            "grdk.viewers.main_window.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ), patch.object(
            window._viewer, "open_file",
        ):
            window._offer_dual_for_multiband("/fake/path.tif")

        # Left canvas should have band_index=0
        left_canvas = window._viewer.left_viewer.canvas
        assert left_canvas.display_settings.band_index == 0

        # Left band combo should show band 0 (not Auto)
        left_controls = window._left_display_dock.widget()
        left_combo = left_controls.findChild(type(left_controls), "")
        # Access the band combo via the controls dict pattern
        from PyQt6.QtWidgets import QComboBox
        left_combos = left_controls.findChildren(QComboBox)
        # The last combo should be the band selector
        band_combo = [c for c in left_combos if c.findText("Auto") >= 0]
        assert len(band_combo) > 0
        assert band_combo[0].currentData() == 0  # band 0 selected


@pytest.mark.skipif(_QT_SKIP, reason="Qt not available")
class TestDisplayControlsSync:
    """Test display controls set_band_index, set_colormap, update_band_info."""

    def test_set_band_index(self):
        """set_band_index should update combo and push to canvas."""
        from grdk.viewers.main_window import ViewerMainWindow
        from grdk.viewers.band_info import BandInfo

        window = ViewerMainWindow()
        arr = np.random.rand(3, 50, 50).astype(np.float32)
        window.set_array(arr, pane=0)

        controls = window._left_display_dock.widget()

        # Populate band info first
        band_info = [
            BandInfo(0, "HH"), BandInfo(1, "HV"), BandInfo(2, "VV"),
        ]
        controls.update_band_info(band_info)

        # Programmatically select band 2
        controls.set_band_index(2)
        assert window._viewer.left_viewer.canvas.display_settings.band_index == 2

        # Switch to Auto
        controls.set_band_index(None)
        assert window._viewer.left_viewer.canvas.display_settings.band_index is None

    def test_set_colormap(self):
        """set_colormap should update combo and push to canvas."""
        from grdk.viewers.main_window import ViewerMainWindow

        window = ViewerMainWindow()
        arr = np.random.rand(50, 50).astype(np.float32)
        window.set_array(arr, pane=0)

        controls = window._left_display_dock.widget()
        controls.set_colormap("viridis")
        assert window._viewer.left_viewer.canvas.display_settings.colormap == "viridis"

    def test_update_band_info_preserves_selection(self):
        """update_band_info should preserve the canvas's current band_index."""
        from grdk.viewers.main_window import ViewerMainWindow
        from grdk.viewers.band_info import BandInfo

        window = ViewerMainWindow()
        arr = np.random.rand(3, 50, 50).astype(np.float32)
        window.set_array(arr, pane=0)

        controls = window._left_display_dock.widget()
        band_info = [
            BandInfo(0, "HH"), BandInfo(1, "HV"), BandInfo(2, "VV"),
        ]

        # Set band 1 on the canvas first
        controls.set_band_index(1)

        # Now update band info — should keep band 1 selected
        controls.update_band_info(band_info)

        from PyQt6.QtWidgets import QComboBox
        band_combos = [
            c for c in controls.findChildren(QComboBox)
            if c.findText("Auto") >= 0
        ]
        assert len(band_combos) > 0
        assert band_combos[0].currentData() == 1

    def test_right_dock_independent(self):
        """Right dock controls should drive the right canvas independently."""
        from grdk.viewers.main_window import ViewerMainWindow

        window = ViewerMainWindow()
        arr = np.random.rand(50, 50).astype(np.float32)
        window._viewer.set_mode("dual")
        window.set_array(arr, pane=0)
        window._viewer.set_array(arr, pane=1)

        left_controls = window._left_display_dock.widget()
        right_controls = window._right_display_dock.widget()

        # Set different colormaps on each pane
        left_controls.set_colormap("viridis")
        right_controls.set_colormap("inferno")

        left_cmap = window._viewer.left_viewer.canvas.display_settings.colormap
        right_cmap = window._viewer.right_viewer.canvas.display_settings.colormap
        assert left_cmap == "viridis"
        assert right_cmap == "inferno"

    def test_pane_band_info_routes_to_correct_dock(self):
        """band_info_changed from right pane should update right dock."""
        from grdk.viewers.main_window import ViewerMainWindow
        from grdk.viewers.band_info import BandInfo

        window = ViewerMainWindow()
        window._viewer.set_mode("dual")

        # Load multiband array into right pane (active pane is still 0)
        arr = np.random.rand(3, 50, 50).astype(np.float32)
        window._viewer.set_array(arr, pane=1)

        # The right dock should have received band info via
        # pane_band_info_changed signal, even though active pane is 0
        from PyQt6.QtWidgets import QComboBox
        right_controls = window._right_display_dock.widget()
        band_combos = [
            c for c in right_controls.findChildren(QComboBox)
            if c.findText("Auto") >= 0
        ]
        # Should have Auto + 3 bands = 4 items
        assert len(band_combos) > 0
        assert band_combos[0].count() >= 4


@pytest.mark.skipif(_QT_SKIP, reason="Qt not available")
class TestSARDisplayFixes:
    """Test fixes for SAR display controls: auto band selection and sync."""

    def test_complex_multiband_auto_selects_band0(self):
        """Multi-band complex data should auto-select band 0, not RGB."""
        from grdk.viewers.geo_viewer import GeoImageViewer

        viewer = GeoImageViewer()
        reader = MultibandSyntheticReader(50, 50, bands=4, complex_dtype=True)
        viewer.open_reader(reader)

        # _apply_auto_settings should have set band_index=0
        assert viewer.canvas.display_settings.band_index == 0

    def test_complex_multiband_auto_percentile(self):
        """Complex SAR data should get 2-98% percentile stretch."""
        from grdk.viewers.geo_viewer import GeoImageViewer

        viewer = GeoImageViewer()
        reader = MultibandSyntheticReader(50, 50, bands=4, complex_dtype=True)
        viewer.open_reader(reader)

        settings = viewer.canvas.display_settings
        assert settings.percentile_low == 2.0
        assert settings.percentile_high == 98.0

    def test_real_multiband_no_auto_band_override(self):
        """Non-complex multi-band data should keep Auto band selection."""
        from grdk.viewers.geo_viewer import GeoImageViewer

        viewer = GeoImageViewer()
        reader = MultibandSyntheticReader(50, 50, bands=3, complex_dtype=False)
        viewer.open_reader(reader)

        # Regular multi-band data (like RGB) should NOT force band 0
        assert viewer.canvas.display_settings.band_index is None

    def test_sync_from_settings_percentile(self):
        """sync_from_settings should update percentile spinboxes."""
        from grdk.viewers.main_window import ViewerMainWindow

        window = ViewerMainWindow()
        reader = MultibandSyntheticReader(50, 50, bands=4, complex_dtype=True)
        window.open_reader(reader, pane=0)

        # After loading a complex SAR image, the display controls
        # should reflect the auto-applied percentile 2/98
        from PyQt6.QtWidgets import QDoubleSpinBox
        controls = window._left_display_dock.widget()
        spinboxes = controls.findChildren(QDoubleSpinBox)
        pct_spinboxes = [s for s in spinboxes if s.suffix() == "%"]

        # Should have two percentile spinboxes
        assert len(pct_spinboxes) == 2

        # They should show 2.0 and 98.0 (from auto-settings), not 0/100
        values = sorted(s.value() for s in pct_spinboxes)
        assert abs(values[0] - 2.0) < 0.1
        assert abs(values[1] - 98.0) < 0.1

    def test_sync_from_settings_band(self):
        """sync_from_settings should update band combo to match canvas."""
        from grdk.viewers.main_window import ViewerMainWindow

        window = ViewerMainWindow()
        reader = MultibandSyntheticReader(50, 50, bands=4, complex_dtype=True)
        window.open_reader(reader, pane=0)

        # After loading, the band combo should show band 0 (not Auto)
        from PyQt6.QtWidgets import QComboBox
        controls = window._left_display_dock.widget()
        band_combos = [
            c for c in controls.findChildren(QComboBox)
            if c.findText("Auto") >= 0
        ]
        assert len(band_combos) > 0
        assert band_combos[0].currentData() == 0  # band 0, not -1 (Auto)

    def test_controls_work_after_sar_load(self):
        """After loading SAR data, changing controls should update canvas."""
        from grdk.viewers.main_window import ViewerMainWindow

        window = ViewerMainWindow()
        reader = MultibandSyntheticReader(50, 50, bands=4, complex_dtype=True)
        window.open_reader(reader, pane=0)

        canvas = window._viewer.left_viewer.canvas
        controls = window._left_display_dock.widget()

        # Change colormap via controls
        controls.set_colormap("viridis")
        assert canvas.display_settings.colormap == "viridis"

        # Percentile should be preserved (not reset to 0/100)
        assert canvas.display_settings.percentile_low == 2.0
        assert canvas.display_settings.percentile_high == 98.0

        # Band index should be preserved
        assert canvas.display_settings.band_index == 0

    def test_colormap_applies_with_band_selected(self):
        """Colormap should be applied when a specific band is selected."""
        from grdk.viewers.image_canvas import normalize_array, DisplaySettings

        arr = np.random.rand(4, 50, 50).astype(np.float32)
        settings = DisplaySettings(
            band_index=0,
            colormap="viridis",
            percentile_low=2.0,
            percentile_high=98.0,
        )
        result = normalize_array(arr, settings)

        # With band 0 selected, output should be RGB from viridis colormap
        assert result.ndim == 3
        assert result.shape[2] == 3  # viridis produces (H, W, 3)

    def test_remap_applies_with_band_selected(self):
        """Remap function should work when data is 2D (band selected)."""
        from grdk.viewers.image_canvas import normalize_array, DisplaySettings

        # Create a simple remap function
        def double_remap(arr):
            return np.clip(arr * 2, 0, 255).astype(np.uint8)

        arr = np.random.rand(4, 50, 50).astype(np.float32) * 100
        settings = DisplaySettings(
            band_index=0,
            remap_function=double_remap,
        )
        result = normalize_array(arr, settings)

        # With band 0 selected (2D input to remap), remap should work
        assert result.dtype == np.uint8
        assert result.ndim == 2  # Single band, grayscale

    def test_decline_dual_controls_still_work(self):
        """Declining dual-view for multiband SAR should not break controls."""
        from unittest.mock import patch
        from grdk.viewers.main_window import ViewerMainWindow

        window = ViewerMainWindow()

        # Simulate loading a complex multiband SAR image
        reader = MultibandSyntheticReader(50, 50, bands=4, complex_dtype=True)

        # Mock open_any to return our reader, and create_geolocation to
        # return None
        with patch(
            "grdk.viewers.geo_viewer.open_any", return_value=reader,
        ), patch(
            "grdk.viewers.geo_viewer.create_geolocation", return_value=None,
        ), patch(
            "grdk.viewers.main_window.QMessageBox.question",
            return_value=QMessageBox.StandardButton.No,
        ):
            window._open_fresh("/fake/sar_image.nitf")

        canvas = window._viewer.left_viewer.canvas
        controls = window._left_display_dock.widget()

        # Canvas should have auto-settings applied
        assert canvas.display_settings.band_index == 0
        assert canvas.display_settings.percentile_low == 2.0
        assert canvas.display_settings.percentile_high == 98.0
        assert canvas.display_settings.colormap == "grayscale"

        # Changing colormap should work (not show false-color RGB)
        controls.set_colormap("hot")
        assert canvas.display_settings.colormap == "hot"
        # Percentile should be preserved
        assert canvas.display_settings.percentile_low == 2.0
        assert canvas.display_settings.percentile_high == 98.0

    def test_toggle_dual_populates_right_pane(self):
        """User-initiated dual toggle should try to re-open in right pane.

        Auto-population only works with file-backed readers (has filepath).
        With a synthetic reader (no filepath), the right pane stays empty
        — this is correct because sharing reader objects caused tile-loading
        failures when one pane's reader was closed.
        """
        from grdk.viewers.main_window import ViewerMainWindow

        window = ViewerMainWindow()
        reader = MultibandSyntheticReader(50, 50, bands=4, complex_dtype=True)
        window.open_reader(reader, pane=0)

        # Right pane should be empty initially
        right_canvas = window._viewer.right_viewer.canvas
        assert right_canvas.source_array is None

        # User-initiated toggle calls _on_toggle_dual, which tries
        # to re-open from filepath.  Synthetic readers have no filepath,
        # so the right pane stays empty (safe behavior).
        window._on_toggle_dual(True)

        # Verify dual mode is active
        assert window._viewer.mode == "dual"

        # Explicit open_reader into right pane should still work
        reader2 = MultibandSyntheticReader(50, 50, bands=4, complex_dtype=True)
        window.open_reader(reader2, pane=1)
        assert right_canvas.source_array is not None

        # Right display controls should have band info populated
        from PyQt6.QtWidgets import QComboBox
        right_controls = window._right_display_dock.widget()
        band_combos = [
            c for c in right_controls.findChildren(QComboBox)
            if c.findText("Auto") >= 0
        ]
        assert len(band_combos) > 0
        # Should have Auto + 4 bands = 5 items
        assert band_combos[0].count() >= 5

    def test_toggle_dual_right_not_overwritten(self):
        """Enabling dual mode should NOT overwrite existing right pane content."""
        from grdk.viewers.main_window import ViewerMainWindow

        window = ViewerMainWindow()
        window._viewer.set_mode("dual")

        # Load different arrays in each pane
        left_arr = np.random.rand(50, 50).astype(np.float32)
        right_arr = np.random.rand(30, 30).astype(np.float32)
        window.set_array(left_arr, pane=0)
        window._viewer.set_array(right_arr, pane=1)

        # Switch to single, then back to dual via user toggle
        window._viewer.set_mode("single")
        window._on_toggle_dual(True)

        # Right pane should still have its own content (30x30), not left's
        right_source = window._viewer.right_viewer.canvas.source_array
        assert right_source is not None
        assert right_source.shape == (30, 30)


@pytest.mark.skipif(_QT_SKIP, reason="Qt not available")
class TestAutoSettingsNoRerender:
    """Test that _apply_auto_settings doesn't trigger stale tile re-render."""

    def test_auto_settings_assigns_directly(self):
        """_apply_auto_settings should set _settings without re-rendering."""
        from grdk.viewers.geo_viewer import GeoImageViewer
        from unittest.mock import patch

        viewer = GeoImageViewer()
        reader = MultibandSyntheticReader(50, 50, bands=4, complex_dtype=True)

        # Track if set_display_settings was called on the canvas
        called = []
        original = viewer.canvas.set_display_settings

        def spy(settings):
            called.append(True)
            original(settings)

        with patch.object(viewer.canvas, 'set_display_settings', side_effect=spy):
            viewer._apply_auto_settings(reader)

        # set_display_settings should NOT have been called — we use
        # direct assignment to avoid re-rendering stale tiles
        assert len(called) == 0

        # But settings should still be applied
        assert viewer.canvas.display_settings.band_index == 0
        assert viewer.canvas.display_settings.percentile_low == 2.0


@pytest.mark.skipif(_QT_SKIP, reason="Qt not available")
class TestContrastBrightnessSpinboxes:
    """Test that contrast/brightness sliders have linked spinboxes."""

    def test_contrast_spinbox_exists(self):
        """Display controls should have a contrast spinbox."""
        from grdk.viewers.main_window import ViewerMainWindow
        from PyQt6.QtWidgets import QSpinBox

        window = ViewerMainWindow()
        controls = window._left_display_dock.widget()
        spinboxes = controls.findChildren(QSpinBox)
        # Should have at least 2 spinboxes (contrast + brightness)
        assert len(spinboxes) >= 2

    def test_contrast_spinbox_linked_to_slider(self):
        """Changing contrast slider should update spinbox and vice versa."""
        from grdk.viewers.main_window import ViewerMainWindow
        from PyQt6.QtWidgets import QSlider, QSpinBox

        window = ViewerMainWindow()
        arr = np.random.rand(50, 50).astype(np.float32)
        window.set_array(arr, pane=0)

        controls = window._left_display_dock.widget()
        sliders = controls.findChildren(QSlider)
        spinboxes = controls.findChildren(QSpinBox)

        # Find contrast slider (range 0-300)
        contrast_slider = None
        for s in sliders:
            if s.maximum() == 300:
                contrast_slider = s
                break
        assert contrast_slider is not None

        # Find contrast spinbox (range 0-300)
        contrast_spin = None
        for s in spinboxes:
            if s.maximum() == 300:
                contrast_spin = s
                break
        assert contrast_spin is not None

        # Change slider → spinbox should follow
        contrast_slider.setValue(200)
        assert contrast_spin.value() == 200

        # Change spinbox → slider should follow
        contrast_spin.setValue(150)
        assert contrast_slider.value() == 150

        # Canvas should reflect the change
        canvas = window._viewer.left_viewer.canvas
        assert abs(canvas.display_settings.contrast - 1.5) < 0.01

    def test_brightness_spinbox_linked_to_slider(self):
        """Changing brightness slider should update spinbox and vice versa."""
        from grdk.viewers.main_window import ViewerMainWindow
        from PyQt6.QtWidgets import QSlider, QSpinBox

        window = ViewerMainWindow()
        arr = np.random.rand(50, 50).astype(np.float32)
        window.set_array(arr, pane=0)

        controls = window._left_display_dock.widget()
        sliders = controls.findChildren(QSlider)
        spinboxes = controls.findChildren(QSpinBox)

        # Find brightness slider (range -100 to 100)
        brightness_slider = None
        for s in sliders:
            if s.minimum() == -100:
                brightness_slider = s
                break
        assert brightness_slider is not None

        # Find brightness spinbox (range -100 to 100)
        brightness_spin = None
        for s in spinboxes:
            if s.minimum() == -100:
                brightness_spin = s
                break
        assert brightness_spin is not None

        # Change slider → spinbox should follow
        brightness_slider.setValue(50)
        assert brightness_spin.value() == 50

        # Change spinbox → slider should follow
        brightness_spin.setValue(-30)
        assert brightness_slider.value() == -30

        # Canvas should reflect the change
        canvas = window._viewer.left_viewer.canvas
        assert abs(canvas.display_settings.brightness - (-0.3)) < 0.01

    def test_sync_from_settings_updates_spinboxes(self):
        """sync_from_settings should update both sliders and spinboxes."""
        from dataclasses import replace
        from grdk.viewers.main_window import ViewerMainWindow
        from PyQt6.QtWidgets import QSpinBox

        window = ViewerMainWindow()
        arr = np.random.rand(50, 50).astype(np.float32)
        window.set_array(arr, pane=0)

        canvas = window._viewer.left_viewer.canvas
        # Set contrast and brightness directly on canvas
        settings = replace(
            canvas.display_settings, contrast=2.0, brightness=0.5,
        )
        canvas._settings = settings

        # Sync controls from settings
        controls = window._left_display_dock.widget()
        controls.sync_from_settings()

        spinboxes = controls.findChildren(QSpinBox)
        # Find contrast spinbox (range 0-300)
        contrast_spin = None
        for s in spinboxes:
            if s.maximum() == 300:
                contrast_spin = s
                break
        assert contrast_spin is not None
        assert contrast_spin.value() == 200  # 2.0 * 100

        # Find brightness spinbox (range -100 to 100)
        brightness_spin = None
        for s in spinboxes:
            if s.minimum() == -100:
                brightness_spin = s
                break
        assert brightness_spin is not None
        assert brightness_spin.value() == 50  # 0.5 * 100


@pytest.mark.skipif(_QT_SKIP, reason="Qt not available")
class TestColorBarWidget:
    """Test the ColorBarWidget."""

    def test_colorbar_exists_on_viewer(self):
        """GeoImageViewer should have a colorbar widget."""
        from grdk.viewers.geo_viewer import GeoImageViewer

        viewer = GeoImageViewer()
        assert hasattr(viewer, 'colorbar')
        assert viewer.colorbar is not None

    def test_colorbar_initially_hidden(self):
        """Colorbar should be hidden by default."""
        from grdk.viewers.geo_viewer import GeoImageViewer

        viewer = GeoImageViewer()
        assert viewer.colorbar.isHidden()

    def test_colorbar_set_colormap(self):
        """Setting colormap should update the colorbar."""
        from grdk.widgets.colorbar import ColorBarWidget

        bar = ColorBarWidget()
        bar.set_colormap("viridis")
        assert bar._colormap_name == "viridis"

    def test_colorbar_set_range(self):
        """Setting range should update labels."""
        from grdk.widgets.colorbar import ColorBarWidget

        bar = ColorBarWidget()
        bar.set_range(10.0, 200.0)
        assert bar._vmin == 10.0
        assert bar._vmax == 200.0

    def test_colorbar_update_from_settings(self):
        """update_from_settings should set colormap and range."""
        from grdk.viewers.image_canvas import DisplaySettings
        from grdk.widgets.colorbar import ColorBarWidget

        bar = ColorBarWidget()
        settings = DisplaySettings(
            colormap="inferno",
            window_min=5.0,
            window_max=100.0,
        )
        bar.update_from_settings(settings)
        assert bar._colormap_name == "inferno"
        assert bar._vmin == 5.0
        assert bar._vmax == 100.0


@pytest.mark.skipif(_QT_SKIP, reason="Qt not available")
class TestColorBarToggle:
    """Test the colorbar toggle checkbox in display controls."""

    def test_colorbar_checkbox_exists(self):
        """Display controls should have a colorbar checkbox."""
        from grdk.viewers.main_window import ViewerMainWindow
        from PyQt6.QtWidgets import QCheckBox

        window = ViewerMainWindow()
        controls = window._left_display_dock.widget()
        cb = getattr(controls, 'colorbar_checkbox', None)
        assert cb is not None
        assert isinstance(cb, QCheckBox)

    def test_colorbar_checkbox_toggles_visibility(self):
        """Checking the colorbar checkbox should show/hide the colorbar."""
        from grdk.viewers.main_window import ViewerMainWindow

        window = ViewerMainWindow()
        arr = np.random.rand(50, 50).astype(np.float32)
        window.set_array(arr, pane=0)

        controls = window._left_display_dock.widget()
        cb = controls.colorbar_checkbox
        colorbar = window._viewer.left_viewer.colorbar

        # Use isHidden() — isVisible() requires the parent window to be shown
        assert colorbar.isHidden()
        cb.setChecked(True)
        assert not colorbar.isHidden()
        cb.setChecked(False)
        assert colorbar.isHidden()

    def test_colorbar_disabled_for_rgb(self):
        """Colorbar checkbox should be disabled for RGB (auto-band 3+ bands)."""
        from grdk.viewers.main_window import ViewerMainWindow

        window = ViewerMainWindow()
        arr = np.random.rand(3, 50, 50).astype(np.float32)
        window.set_array(arr, pane=0)

        controls = window._left_display_dock.widget()
        cb = controls.colorbar_checkbox

        # With 3 bands and no explicit band selection → RGB → disabled
        assert not cb.isEnabled()

    def test_colorbar_enabled_for_scalar(self):
        """Colorbar checkbox should be enabled for single-band data."""
        from grdk.viewers.main_window import ViewerMainWindow

        window = ViewerMainWindow()
        arr = np.random.rand(50, 50).astype(np.float32)
        window.set_array(arr, pane=0)

        controls = window._left_display_dock.widget()
        cb = controls.colorbar_checkbox

        # Single band → scalar → enabled
        assert cb.isEnabled()

    def test_colorbar_enabled_when_band_selected(self):
        """Colorbar should be enabled when a specific band is selected."""
        from grdk.viewers.main_window import ViewerMainWindow

        window = ViewerMainWindow()
        reader = MultibandSyntheticReader(50, 50, bands=4, complex_dtype=True)
        window.open_reader(reader, pane=0)

        controls = window._left_display_dock.widget()
        cb = controls.colorbar_checkbox

        # Complex multi-band auto-selects band 0 → scalar → enabled
        assert cb.isEnabled()


@pytest.mark.skipif(_QT_SKIP, reason="Qt not available")
class TestTiledLoading:
    """Integration tests for the tiled image loading pipeline.

    Verifies that tiles are loaded asynchronously, rendered on the
    main thread, and placed in the scene.
    """

    def test_tiled_loading_produces_tiles(self):
        """Tiles should be loaded and rendered for large images."""
        import time
        from unittest.mock import patch
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import QThreadPool
        from grdk.viewers.tiled_canvas import TiledImageCanvas
        from grdk.viewers.tile_cache import TILE_THRESHOLD

        # Create a reader just above the tiling threshold
        side = 200  # 200 x 200 = 40000 pixels
        reader = SyntheticReader(side, side)

        canvas = TiledImageCanvas()

        # Lower the threshold so our small image triggers tiling
        with patch('grdk.viewers.tiled_canvas.needs_tiling', return_value=True):
            canvas.set_reader(reader)

        assert canvas._tiled_mode is True
        assert canvas._tile_cache is not None

        # Force a tile update (normally triggered by timer after show)
        canvas._update_visible_tiles()

        # Wait for worker threads to complete
        QThreadPool.globalInstance().waitForDone(5000)

        # Process queued signals (tile_data_ready → _on_tile_data)
        for _ in range(50):
            QApplication.processEvents()
            time.sleep(0.01)

        # Tiles should have been placed in the scene
        assert len(canvas._tile_items) > 0, (
            "No tiles placed in scene after loading"
        )

        # Tile cache should have raw data
        assert len(canvas._tile_cache._raw_cache) > 0, (
            "No raw tiles in cache after loading"
        )

        # Tile cache should have rendered pixmaps
        assert len(canvas._tile_cache._pixmap_cache) > 0, (
            "No rendered pixmaps in cache after loading"
        )

        # No tiles should be pending
        assert not canvas._tile_cache.has_pending, (
            "Tiles still pending after waitForDone + processEvents"
        )

    def test_tiled_loading_signal_chain(self):
        """The tile_data_ready → _on_tile_data → tile_ready chain works."""
        import time
        from unittest.mock import patch
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import QThreadPool
        from grdk.viewers.tile_cache import TileCache, TileKey

        reader = SyntheticReader(200, 200)
        cache = TileCache(reader, tile_size=128)

        # Track tile_ready emissions
        ready_tiles = []
        cache.tile_ready.connect(
            lambda l, r, c: ready_tiles.append(TileKey(l, r, c))
        )

        # Request a single tile
        key = TileKey(0, 0, 0)
        cache.request_visible(0, [key])

        # Wait for worker to complete
        QThreadPool.globalInstance().waitForDone(5000)

        # Process queued signals
        for _ in range(50):
            QApplication.processEvents()
            time.sleep(0.01)

        # tile_ready should have fired
        assert key in ready_tiles, (
            f"tile_ready not emitted for {key}; "
            f"raw_cache={list(cache._raw_cache.keys())}, "
            f"pending={cache._pending}"
        )

        # Raw data should be cached
        assert key in cache._raw_cache

        # Pixmap should be rendered
        assert key in cache._pixmap_cache

    def test_tile_worker_failure_clears_pending(self):
        """Failed tile workers should clear pending so busy cursor unblocks."""
        import time
        from unittest.mock import patch
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import QThreadPool
        from grdk.viewers.tile_cache import TileCache, TileKey

        reader = SyntheticReader(200, 200)
        cache = TileCache(reader, tile_size=128)

        # Make read_chip always raise to simulate a closed reader
        def _failing_read_chip(*args, **kwargs):
            raise RuntimeError("Reader closed")

        reader.read_chip = _failing_read_chip

        # Request a tile — the worker should fail
        key = TileKey(0, 0, 0)
        cache.request_visible(0, [key])

        # Wait for worker to complete
        QThreadPool.globalInstance().waitForDone(5000)

        # Process queued signals
        for _ in range(50):
            QApplication.processEvents()
            time.sleep(0.01)

        # The key should NOT be in pending (worker failure clears it)
        assert key not in cache._pending, (
            f"Failed tile {key} still in _pending — busy cursor would stick"
        )

        # has_pending should be False
        assert not cache.has_pending, (
            "has_pending still True after failed tile load"
        )
