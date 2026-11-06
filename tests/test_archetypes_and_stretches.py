# -*- coding: utf-8 -*-
"""
Tests for Image Archetypes, Dynamic Contrast Stretches, Close Image Actions, and Points/Lines vector geometry.
"""

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from grdk.viewers.archetypes import ArchetypeManager, ImageArchetype, ArchetypeDisplayDefaults
from grdk.viewers.image_canvas import DisplaySettings, normalize_array
from grdk.viewers.main_window import ViewerMainWindow


def test_archetype_manager_builtins():
    mgr = ArchetypeManager()
    assert "General / Standard" in mgr.archetypes
    assert "SAR Polarimetric" in mgr.archetypes
    assert "Optical / Multispectral" in mgr.archetypes
    assert "Hyperspectral Exploitation" in mgr.archetypes


def test_archetype_detection():
    mgr = ArchetypeManager()
    
    class MockSARReader:
        pass
    class MockHSIReader:
        metadata = type("Meta", (), {"channel_metadata": [None]*23})()
    class MockMSIReader:
        metadata = type("Meta", (), {"channel_metadata": [None]*4})()

    sar_arch = mgr.detect_archetype_for_reader(MockSARReader())
    assert sar_arch.name == "SAR Polarimetric"

    hsi_arch = mgr.detect_archetype_for_reader(MockHSIReader())
    assert hsi_arch.name == "Hyperspectral Exploitation"

    msi_arch = mgr.detect_archetype_for_reader(MockMSIReader())
    assert msi_arch.name == "Optical / Multispectral"


def test_contrast_stretches():
    arr = np.linspace(10, 1000, 100, dtype=np.float32).reshape(10, 10)

    # 1. Linear MinMax
    s_minmax = DisplaySettings(stretch_mode="minmax")
    out_minmax = normalize_array(arr, s_minmax)
    assert out_minmax.min() == 0
    assert out_minmax.max() == 255

    # 2. StdDev stretch
    s_std = DisplaySettings(stretch_mode="stddev", stddev_factor=2.0)
    out_std = normalize_array(arr, s_std)
    assert out_std.dtype == np.uint8

    # 3. Logarithmic stretch
    s_log = DisplaySettings(stretch_mode="log")
    out_log = normalize_array(arr, s_log)
    assert out_log.dtype == np.uint8

    # 4. Equalize stretch
    s_eq = DisplaySettings(stretch_mode="equalize")
    out_eq = normalize_array(arr, s_eq)
    assert out_eq.dtype == np.uint8


def test_viewer_close_actions():
    window = ViewerMainWindow()
    test_arr = np.ones((50, 50), dtype=np.float32)
    window.set_array(test_arr, pane=0)

    assert window._viewer.left_viewer.canvas.source_array is not None
    assert window._close_image_action is not None
    assert window._close_all_images_action is not None

    # Trigger close image
    window._close_image_action.trigger()
    assert window._viewer.left_viewer.canvas.source_array is None
    window.close()


def test_geometry_modes():
    window = ViewerMainWindow()
    assert hasattr(window, "_draw_point_action")
    assert hasattr(window, "_draw_line_action")
    assert hasattr(window, "_draw_polygon_action")
    assert hasattr(window, "_archetype_combo")

    # Toggle point mode
    window._draw_point_action.setChecked(True)
    window._draw_point_action.triggered.emit(True)
    canvas = window._viewer.left_viewer.canvas
    assert canvas._polygon_state.active is True
    assert canvas._polygon_state.mode == "point"

    # Toggle line mode
    window._draw_line_action.setChecked(True)
    window._draw_line_action.triggered.emit(True)
    assert canvas._polygon_state.active is True
    assert canvas._polygon_state.mode == "line"

    window.close()
