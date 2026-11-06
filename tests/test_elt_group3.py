# tests/test_elt_group3.py
"""Unit tests for ELT Group 3:
- ELT-07 (#25): Geodetic mensuration tool
- ELT-08 (#26): Roaming magnifier / loupe
- ELT-09 (#27): Image rotation & flip
- ELT-10 (#28): Georeferenced chip exporter
- ELT-15 (#33): Overview / mini-map locator
- ELT-16 (#34): Dual-pane split swipe
- ELT-17 (#35): Automated flicker comparator
- ELT-20 (#38): Synchronized conjugate cursors
"""

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

if QApplication.instance() is None:
    _app = QApplication(["--platform", "offscreen"])
else:
    _app = QApplication.instance()

from grdk.viewers.mensuration import (
    haversine_distance,
    geodesic_azimuth,
    geodesic_polygon_area,
    MensurationResult,
)
from grdk.viewers.loupe import MagnifierLoupe
from grdk.viewers.image_canvas import ImageCanvas
from grdk.viewers.chip_exporter import export_image_chip
from grdk.viewers.overview_map import OverviewMapWidget
from grdk.viewers.dual_viewer import DualGeoViewer, FlickerController


class TestGeodeticMensuration:
    """Tests for ELT-07 (#25): Mensuration."""

    def test_haversine_distance(self):
        # Distance between (0, 0) and (0, 1) deg lon at equator (~111.19 km)
        dist = haversine_distance(0.0, 0.0, 0.0, 1.0)
        assert 110_000 < dist < 112_000

    def test_geodesic_azimuth(self):
        # Due north
        az_north = geodesic_azimuth(0.0, 0.0, 1.0, 0.0)
        assert np.isclose(az_north, 0.0, atol=1e-2)

        # Due east
        az_east = geodesic_azimuth(0.0, 0.0, 0.0, 1.0)
        assert np.isclose(az_east, 90.0, atol=1e-2)

    def test_geodesic_polygon_area(self):
        # ~1 degree x 1 degree equatorial square (~111km x 111km ≈ 1.23e10 m^2)
        coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        area = geodesic_polygon_area(coords)
        assert 1.2e10 < area < 1.3e10


class TestMagnifierLoupe:
    """Tests for ELT-08 (#26): Magnifier Loupe."""

    def test_loupe_chip_extraction(self):
        arr = np.arange(10000, dtype=np.float32).reshape(100, 100)
        loupe = MagnifierLoupe(zoom_factor=2.0, radius=20)
        chip = loupe.extract_zoomed_chip(arr, center_row=50, center_col=50)
        assert chip.shape == (40, 40)
        assert chip[20, 20] == arr[50, 50]


class TestImageRotationAndFlip:
    """Tests for ELT-09 (#27): Image rotation & flip."""

    def test_canvas_rotation_and_flip(self):
        canvas = ImageCanvas()
        arr = np.zeros((10, 10), dtype=np.uint8)
        canvas.set_array(arr)

        assert canvas.rotation_angle == 0.0
        canvas.rotate_by(90.0)
        assert canvas.rotation_angle == 90.0

        canvas.flip_horizontal()
        assert canvas.is_flipped_h is True

        canvas.flip_vertical()
        assert canvas.is_flipped_v is True

        canvas.reset_orientation()
        assert canvas.rotation_angle == 0.0
        assert canvas.is_flipped_h is False
        assert canvas.is_flipped_v is False


class TestChipExporter:
    """Tests for ELT-10 (#28): Chip exporter."""

    def test_export_chip_numpy_roundtrip(self, tmp_path):
        out_tif = tmp_path / "chip_test.tif"
        data = np.ones((3, 50, 50), dtype=np.float32) * 42.0

        export_image_chip(
            data,
            str(out_tif),
            row=10,
            col=10,
            height=20,
            width=20,
        )
        assert out_tif.exists()
        assert out_tif.stat().st_size > 0


class TestOverviewMapLocator:
    """Tests for ELT-15 (#33): Overview mini-map."""

    def test_overview_map_init_and_rect(self):
        overview = OverviewMapWidget()
        arr = np.zeros((200, 200), dtype=np.uint8)
        overview.set_full_image(arr)
        overview.update_viewport_rect(row=20, col=20, height=40, width=40)
        assert overview.viewport_rect == (20, 20, 40, 40)


class TestDualPaneComparative:
    """Tests for ELT-16 (#34), ELT-17 (#35), ELT-20 (#38)."""

    def test_flicker_controller(self):
        viewer = DualGeoViewer()
        flicker = FlickerController(viewer, interval_ms=100)
        assert flicker.is_running is False
        flicker.start()
        assert flicker.is_running is True
        flicker.toggle_state()
        flicker.stop()
        assert flicker.is_running is False

    def test_swipe_position(self):
        viewer = DualGeoViewer()
        viewer.set_swipe_mode(True)
        assert viewer.is_swipe_mode is True
        viewer.set_swipe_position(0.65)
        assert viewer.swipe_position == 0.65
        viewer.set_swipe_mode(False)
        assert viewer.is_swipe_mode is False

    def test_sync_conjugate_cursor(self):
        viewer = DualGeoViewer()
        # Verify sync cursor helper exists and doesn't crash on hover
        viewer.set_cursor_sync(True)
        assert viewer.is_cursor_sync is True
        viewer._on_pane_cursor_hover(pane=0, row=50, col=50)

