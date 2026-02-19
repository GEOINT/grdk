# -*- coding: utf-8 -*-
"""
Viewers Module - Embeddable Qt viewer components for GRDK widgets.

Provides interactive image display via ``ImageCanvas`` (pan, zoom,
contrast, colormaps), tiled rendering for large images, geospatial
coordinate display, GeoJSON vector overlays, napari-based stack
viewing, chip thumbnail galleries, and polygon drawing tools.

Components
----------
- ``image_canvas`` — Base interactive image viewer (ImageCanvas,
  ImageCanvasThumbnail, DisplaySettings, normalize_array)
- ``tile_cache`` — LOD tile pyramid with async loading and LRU eviction
- ``tiled_canvas`` — Tiled rendering extension of ImageCanvas
- ``coordinate_bar`` — Pixel + lat/lon status bar
- ``vector_overlay`` — GeoJSON vector rendering over images
- ``geo_viewer`` — Single-pane geospatial viewer (composite widget)
- ``main_window`` — Standalone viewer application window
- ``stack_viewer`` — napari-based multi-image stack viewer
- ``chip_gallery`` — Scrollable chip thumbnail grid with labels
- ``polygon_tools`` — Polygon-based chip extraction utilities

Dependencies
------------
napari (optional, for stack viewer)

Author
------
Claude Code (Anthropic)

Contributor
-----------
Steven Siebert

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-06

Modified
--------
2026-02-18
"""

from grdk.viewers.band_info import BandInfo, get_band_info
from grdk.viewers.tile_cache import TileCache, TileKey, needs_tiling
from grdk.viewers.tiled_canvas import TiledImageCanvas
from grdk.viewers.coordinate_bar import CoordinateBar
from grdk.viewers.vector_overlay import VectorOverlayLayer
from grdk.viewers.geo_viewer import GeoImageViewer, open_any, create_geolocation
from grdk.viewers.main_window import ViewerMainWindow


def show(data, *, geolocation=None, title=None, block=True):
    """Display image data in the GRDK viewer.

    Dispatches on input type:

    - ``np.ndarray`` — displays the array directly
    - ``str`` or ``Path`` — opens file with auto-detection
    - ImageReader — displays via reader with metadata/geolocation

    Parameters
    ----------
    data : np.ndarray or ImageReader or str or Path
        Image data, reader, or file path to display.
    geolocation : optional
        Geolocation model.  Ignored for file paths (auto-detected).
        For readers, if ``None``, attempts auto-detection via
        ``create_geolocation()``.
    title : str, optional
        Window title.  If ``None``, auto-generated from input.
    block : bool
        If ``True`` (default), block until the viewer window is
        closed.  If ``False``, return immediately.

    Returns
    -------
    ViewerMainWindow
        The viewer window instance.
    """
    import numpy as np
    from pathlib import Path as _Path

    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication.instance()
    created_app = False
    if app is None:
        app = QApplication(sys.argv)
        created_app = True

    window = ViewerMainWindow()

    if isinstance(data, np.ndarray):
        window.set_array(data, geolocation=geolocation, title=title)
    elif isinstance(data, (str, _Path)):
        window.open_file(str(data))
        if title:
            window.setWindowTitle(f"GRDK Viewer \u2014 {title}")
    else:
        # Duck-typed ImageReader
        if geolocation is None:
            geolocation = create_geolocation(data)
        window.open_reader(data, geolocation=geolocation)
        if title:
            window.setWindowTitle(f"GRDK Viewer \u2014 {title}")

    window.show()

    if block:
        if created_app:
            app.exec()
        else:
            from PyQt6.QtCore import QEventLoop
            loop = QEventLoop()
            original_close = window.closeEvent

            def _on_close(event):
                original_close(event)
                loop.quit()

            window.closeEvent = _on_close
            loop.exec()

    return window


def imshow(arr, *, geolocation=None, title=None, block=True):
    """Display a numpy array in the GRDK viewer.

    Convenience alias for ``show()`` restricted to numpy arrays.

    Parameters
    ----------
    arr : np.ndarray
        Image data (2D, 3D channels-first, or complex).
    geolocation : optional
        Geolocation model for coordinate display.
    title : str, optional
        Window title.
    block : bool
        If ``True`` (default), block until closed.

    Returns
    -------
    ViewerMainWindow
        The viewer window instance.
    """
    import numpy as np

    if not isinstance(arr, np.ndarray):
        raise TypeError(
            f"imshow() expects a numpy array, got {type(arr).__name__}. "
            "Use show() for readers or file paths."
        )
    return show(arr, geolocation=geolocation, title=title, block=block)
