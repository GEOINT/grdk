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
