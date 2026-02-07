# -*- coding: utf-8 -*-
"""
Viewers Module - Embeddable Qt viewer components for GRDK widgets.

Provides interactive image display via ``ImageCanvas`` (pan, zoom,
contrast, colormaps), napari-based stack viewing with polygon
interaction, chip thumbnail galleries, and polygon drawing tools.

Components
----------
- ``image_canvas`` — Base interactive image viewer (ImageCanvas,
  ImageCanvasThumbnail, DisplaySettings, normalize_array)
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
2026-02-06
"""
