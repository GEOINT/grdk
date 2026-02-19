# -*- coding: utf-8 -*-
"""
TiledImageCanvas - Tiled rendering extension of ImageCanvas for large images.

Extends ImageCanvas to support level-of-detail tiled rendering for images
that are too large to fit in memory.  Small images delegate to the base
class single-pixmap path.  Large images are rendered via TileCache with
on-demand tile loading and LOD selection based on zoom level.

Dependencies
------------
PyQt6

Author
------
Claude Code (Anthropic)

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-17

Modified
--------
2026-02-17
"""

# Standard library
import math
from typing import Any, Dict, Optional, Tuple

# Third-party
import numpy as np

try:
    from PyQt6.QtWidgets import QApplication, QGraphicsPixmapItem
    from PyQt6.QtCore import QPointF, Qt, QTimer, pyqtSignal as Signal
    from PyQt6.QtGui import QPixmap

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False

from grdk.viewers.image_canvas import DisplaySettings, ImageCanvas
from grdk.viewers.tile_cache import TileCache, TileKey, needs_tiling


if _QT_AVAILABLE:

    class TiledImageCanvas(ImageCanvas):
        """Interactive image viewer with tiled rendering for large images.

        Extends ``ImageCanvas`` to handle images too large for in-memory
        rendering.  When ``set_reader()`` is called with a large image,
        a ``TileCache`` manages a level-of-detail tile pyramid.  Only
        tiles visible at the current viewport and zoom level are loaded.

        Small images (below the tiling threshold) use the base class
        single-pixmap rendering path via ``set_array()``.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget.

        Signals
        -------
        viewport_changed()
            Emitted on pan, zoom, or resize.  Designed for
            ``SyncController`` in dual-view mode (Phase 3).
        """

        viewport_changed = Signal()

        def __init__(self, parent: Optional[Any] = None) -> None:
            super().__init__(parent)

            self._tile_cache: Optional[TileCache] = None
            self._tiled_mode = False
            self._reader: Optional[Any] = None

            # Map of TileKey -> QGraphicsPixmapItem for active tiles
            self._tile_items: Dict[TileKey, QGraphicsPixmapItem] = {}

            # Current LOD level being displayed
            self._current_level = 0

            # Deferred fit-in-view: fitInView called before the widget is
            # shown produces a wrong transform because the viewport has no
            # real size yet.  We store a pending flag and execute on first
            # showEvent / resizeEvent when the viewport is valid.
            self._fit_pending = False
            self._has_valid_size = False

            # Debounce timer for viewport updates
            self._viewport_timer = QTimer(self)
            self._viewport_timer.setSingleShot(True)
            self._viewport_timer.setInterval(16)  # ~60 Hz
            self._viewport_timer.timeout.connect(self._update_visible_tiles)

            # Busy cursor: shown after 1 second of outstanding tile loads
            self._busy_timer = QTimer(self)
            self._busy_timer.setSingleShot(True)
            self._busy_timer.setInterval(1000)
            self._busy_timer.timeout.connect(self._show_busy_cursor)
            self._busy_active = False

        # --- Public API ---

        def set_reader(self, reader: Any) -> None:
            """Set an ImageReader for tiled lazy rendering.

            If the image is below the tiling threshold, the entire
            image is loaded via the base class ``set_array()`` path.

            Parameters
            ----------
            reader : ImageReader
                grdl reader with ``read_chip()`` and ``get_shape()``.
            """
            self._clear_tiles()
            self._reader = reader

            shape = reader.get_shape()
            rows, cols = shape[0], shape[1]

            if not needs_tiling(rows, cols):
                # Small image — load fully into base class
                self._tiled_mode = False
                arr = reader.read_full()
                super().set_array(arr)
                self._request_fit_in_view()
                return

            # Large image — tiled mode
            self._tiled_mode = True
            self._source = None  # No in-memory source array
            self._pixmap_item.setVisible(False)

            self._tile_cache = TileCache(reader, parent=self)
            self._tile_cache.set_display_settings(self._settings)
            self._tile_cache.tile_ready.connect(self._on_tile_ready)

            # Set scene rect to full image dimensions
            self._scene.setSceneRect(0, 0, cols, rows)

            # Fit will be executed when the widget is shown/resized
            self._request_fit_in_view()

        def set_array(self, arr: np.ndarray) -> None:
            """Set the source image array.

            Overrides base class to handle large arrays via tiling.

            Parameters
            ----------
            arr : np.ndarray
                Image data: ``(H, W)`` for single-band or ``(C, H, W)``
                for multi-band (channels-first).
            """
            self._clear_tiles()

            # Extract spatial dimensions — channels-first convention
            if arr.ndim == 2:
                rows, cols = arr.shape
            else:
                rows, cols = arr.shape[1], arr.shape[2]

            if needs_tiling(rows, cols):
                # Large in-memory array — create a tile cache from a
                # lightweight array-backed reader
                self._tiled_mode = True
                self._source = None
                self._pixmap_item.setVisible(False)

                reader = _ArrayReader(arr)
                self._reader = reader
                self._tile_cache = TileCache(reader, parent=self)
                self._tile_cache.set_display_settings(self._settings)
                self._tile_cache.tile_ready.connect(self._on_tile_ready)

                self._scene.setSceneRect(0, 0, cols, rows)
                self._request_fit_in_view()
            else:
                self._tiled_mode = False
                self._reader = None
                super().set_array(arr)
                self._request_fit_in_view()

        def set_display_settings(self, settings: DisplaySettings) -> None:
            """Update display settings and re-render.

            Parameters
            ----------
            settings : DisplaySettings
                New display parameters.
            """
            if self._tiled_mode and self._tile_cache is not None:
                self._settings = settings
                self._tile_cache.set_display_settings(settings)
                # Refresh all visible tile pixmap items
                for key, item in self._tile_items.items():
                    pixmap = self._tile_cache.get_pixmap(key)
                    if pixmap is not None:
                        item.setPixmap(pixmap)
                # Force scene repaint so updated pixmaps are displayed
                self._scene.update()
                self.display_settings_changed.emit(settings)
            else:
                super().set_display_settings(settings)

        # --- Viewport control (for Phase 3 SyncController) ---

        def get_viewport_center(self) -> Tuple[float, float]:
            """Return the center of the current viewport in scene coords.

            Returns
            -------
            Tuple[float, float]
                (row, col) — center position in source image pixel space.
            """
            center = self.mapToScene(self.viewport().rect().center())
            return (center.y(), center.x())

        def center_on(self, row: float, col: float) -> None:
            """Center the viewport on the given scene coordinates.

            Parameters
            ----------
            row : float
                Row in source image pixel space.
            col : float
                Column in source image pixel space.
            """
            self.centerOn(QPointF(col, row))

        def get_zoom(self) -> float:
            """Return the current zoom factor.

            Returns
            -------
            float
                Zoom level (1.0 = 100%).
            """
            return self.transform().m11()

        # --- Event overrides ---

        def zoom_undo(self) -> None:
            """Revert zoom and schedule tile update."""
            super().zoom_undo()
            if self._tiled_mode:
                self._schedule_tile_update()
                self.viewport_changed.emit()

        def wheelEvent(self, event: Any) -> None:
            """Zoom and schedule tile update."""
            super().wheelEvent(event)
            if self._tiled_mode:
                self._schedule_tile_update()
            self.viewport_changed.emit()

        def mouseMoveEvent(self, event: Any) -> None:
            """Emit pixel coordinates using tile cache data in tiled mode."""
            if self._zoom_box_active:
                super().mouseMoveEvent(event)
                return
            if not self._tiled_mode:
                super().mouseMoveEvent(event)
                return

            # Call QGraphicsView base for drag handling (skip ImageCanvas
            # which reads from _source)
            from PyQt6.QtWidgets import QGraphicsView
            QGraphicsView.mouseMoveEvent(self, event)

            scene_pos = self.mapToScene(event.pos())
            col = int(scene_pos.x())
            row = int(scene_pos.y())

            if self._tile_cache is None:
                return

            rows, cols = self._tile_cache.image_shape
            if 0 <= row < rows and 0 <= col < cols:
                # Try to get value from the level-0 tile cache
                ts = self._tile_cache.tile_size
                tile_row = row // ts
                tile_col = col // ts
                key = TileKey(0, tile_row, tile_col)
                raw = self._tile_cache.get_raw(key)
                if raw is not None:
                    local_r = row - tile_row * ts
                    local_c = col - tile_col * ts
                    if raw.ndim == 2:
                        h, w = raw.shape
                    else:
                        h, w = raw.shape[1], raw.shape[2]
                    if 0 <= local_r < h and 0 <= local_c < w:
                        if raw.ndim == 2:
                            value = raw[local_r, local_c]
                        else:
                            value = raw[:, local_r, local_c]
                        self.pixel_hovered.emit(row, col, value)
                        return

                # Level-0 tile not cached — read single pixel from reader
                if self._reader is not None:
                    try:
                        chip = self._reader.read_chip(row, row + 1, col, col + 1)
                        if chip.ndim == 2:
                            value = chip[0, 0]
                        else:
                            value = chip[:, 0, 0]
                        self.pixel_hovered.emit(row, col, value)
                        return
                    except Exception:
                        pass

                self.pixel_hovered.emit(row, col, None)

        def scrollContentsBy(self, dx: int, dy: int) -> None:
            """Schedule tile update on pan."""
            super().scrollContentsBy(dx, dy)
            if self._tiled_mode:
                self._schedule_tile_update()
            self.viewport_changed.emit()

        def showEvent(self, event: Any) -> None:
            """Execute deferred fit-in-view on first show."""
            super().showEvent(event)
            self._has_valid_size = True
            if self._fit_pending:
                self._fit_pending = False
                QTimer.singleShot(0, self.fit_in_view)

        def resizeEvent(self, event: Any) -> None:
            """Schedule tile update on resize."""
            super().resizeEvent(event)
            self._has_valid_size = True
            if self._fit_pending:
                self._fit_pending = False
                QTimer.singleShot(0, self.fit_in_view)
            elif self._tiled_mode:
                self._schedule_tile_update()
            self.viewport_changed.emit()

        def fit_in_view(self) -> None:
            """Zoom to fit the entire image in the viewport."""
            if not self._has_valid_size:
                self._fit_pending = True
                return
            if self._tiled_mode and self._tile_cache is not None:
                self.fitInView(self._scene.sceneRect(),
                               Qt.AspectRatioMode.KeepAspectRatio)
                self._update_zoom_level()
                self._schedule_tile_update()
            else:
                super().fit_in_view()

        def _request_fit_in_view(self) -> None:
            """Request a fit-in-view, deferring if the widget isn't shown yet."""
            if self._has_valid_size:
                self.fit_in_view()
            else:
                self._fit_pending = True

        def mouseReleaseEvent(self, event: Any) -> None:
            """Schedule tile update after zoom box selection."""
            was_zoom_box = self._zoom_box_active
            super().mouseReleaseEvent(event)
            if was_zoom_box and self._tiled_mode:
                self._schedule_tile_update()
                self.viewport_changed.emit()

        def mouseDoubleClickEvent(self, event: Any) -> None:
            """Fit to view on double-click."""
            self.fit_in_view()

        # --- Internal ---

        def _schedule_tile_update(self) -> None:
            """Debounce tile updates to avoid excessive loading."""
            self._viewport_timer.start()

        def _select_lod_level(self) -> int:
            """Pick LOD level where ~1 source pixel ≈ 1 viewport pixel.

            Returns
            -------
            int
                Pyramid level index.
            """
            if self._tile_cache is None:
                return 0

            zoom = abs(self.transform().m11())
            if zoom <= 0:
                return self._tile_cache.num_levels - 1

            # At zoom=1.0, 1 source pixel = 1 viewport pixel → level 0
            # At zoom=0.5, 2 source pixels per viewport pixel → level 1
            # At zoom=0.25, 4 source pixels per viewport pixel → level 2
            level = max(0, int(math.log2(max(1.0, 1.0 / zoom))))
            return min(level, self._tile_cache.num_levels - 1)

        def _update_visible_tiles(self) -> None:
            """Compute which tiles are visible and request them."""
            if self._tile_cache is None:
                return

            level = self._select_lod_level()
            self._current_level = level

            # Get viewport rectangle in scene coordinates
            viewport_rect = self.mapToScene(self.viewport().rect()).boundingRect()

            factor = 1 << level
            ts = self._tile_cache.tile_size

            # Source pixel range visible in viewport
            col_start = max(0, int(viewport_rect.left()))
            col_end = min(self._tile_cache.image_shape[1], int(viewport_rect.right()) + 1)
            row_start = max(0, int(viewport_rect.top()))
            row_end = min(self._tile_cache.image_shape[0], int(viewport_rect.bottom()) + 1)

            # Tile range at this LOD level
            tile_col_start = col_start // (ts * factor)
            tile_col_end = math.ceil(col_end / (ts * factor))
            tile_row_start = row_start // (ts * factor)
            tile_row_end = math.ceil(row_end / (ts * factor))

            # Build list of needed tile keys
            needed_keys = []
            for tr in range(tile_row_start, tile_row_end):
                for tc in range(tile_col_start, tile_col_end):
                    needed_keys.append(TileKey(level, tr, tc))

            # Remove tiles from other levels
            stale_keys = [k for k in self._tile_items if k.level != level]
            for k in stale_keys:
                item = self._tile_items.pop(k)
                self._scene.removeItem(item)

            # Remove tiles at current level that are no longer visible
            needed_set = set(needed_keys)
            offscreen = [k for k in self._tile_items if k not in needed_set]
            for k in offscreen:
                item = self._tile_items.pop(k)
                self._scene.removeItem(item)

            # Place already-cached tiles, request missing ones
            self._tile_cache.request_visible(level, needed_keys)

            for key in needed_keys:
                if key in self._tile_items:
                    continue
                pixmap = self._tile_cache.get_pixmap(key)
                if pixmap is not None:
                    self._place_tile(key, pixmap)

            # Start busy cursor timer if tiles are still loading
            if self._tile_cache.has_pending:
                if not self._busy_timer.isActive() and not self._busy_active:
                    self._busy_timer.start()
            else:
                self._hide_busy_cursor()

        def _on_tile_ready(self, level: int, tile_row: int, tile_col: int) -> None:
            """Handle a tile that finished loading."""
            key = TileKey(level, tile_row, tile_col)

            # Only place tiles at the current display level
            if level != self._current_level:
                return

            if key in self._tile_items:
                # Update existing item
                pixmap = self._tile_cache.get_pixmap(key)
                if pixmap is not None:
                    self._tile_items[key].setPixmap(pixmap)
            else:
                pixmap = self._tile_cache.get_pixmap(key)
                if pixmap is not None:
                    self._place_tile(key, pixmap)

            # Hide busy cursor when all pending tiles are loaded
            if self._tile_cache is not None and not self._tile_cache.has_pending:
                self._hide_busy_cursor()

        def _place_tile(self, key: TileKey, pixmap: QPixmap) -> None:
            """Add a tile pixmap item to the scene at the correct position."""
            factor = 1 << key.level
            ts = self._tile_cache.tile_size

            item = QGraphicsPixmapItem(pixmap)

            # Position in source image pixel coordinates
            scene_x = key.tile_col * ts * factor
            scene_y = key.tile_row * ts * factor
            item.setPos(scene_x, scene_y)

            # Scale the tile pixmap to cover the correct source pixel region
            if factor > 1:
                item.setScale(factor)

            item.setZValue(-1)  # Behind overlays
            self._scene.addItem(item)
            self._tile_items[key] = item

        def _show_busy_cursor(self) -> None:
            """Show busy cursor after the delay timer fires."""
            if self._tile_cache is not None and self._tile_cache.has_pending:
                if not self._busy_active:
                    QApplication.setOverrideCursor(
                        Qt.CursorShape.BusyCursor,
                    )
                    self._busy_active = True

        def _hide_busy_cursor(self) -> None:
            """Remove the busy cursor override."""
            self._busy_timer.stop()
            if self._busy_active:
                QApplication.restoreOverrideCursor()
                self._busy_active = False

        def _clear_tiles(self) -> None:
            """Remove all tile items and reset tiled state."""
            self._hide_busy_cursor()

            for item in self._tile_items.values():
                self._scene.removeItem(item)
            self._tile_items.clear()

            if self._tile_cache is not None:
                self._tile_cache.clear()
                self._tile_cache = None

            self._tiled_mode = False
            self._reader = None
            self._current_level = 0
            self._pixmap_item.setVisible(True)


    class _ArrayReader:
        """Minimal ImageReader-like wrapper around an in-memory numpy array.

        Used by ``TiledImageCanvas.set_array()`` when a large array is
        passed directly.  Provides ``read_chip()`` and ``get_shape()``
        without requiring the full ``ImageReader`` ABC.

        Arrays are expected in channels-first format: ``(H, W)`` for
        single-band or ``(C, H, W)`` for multi-band.
        """

        def __init__(self, arr: np.ndarray) -> None:
            self._arr = arr

        def read_chip(
            self,
            row_start: int,
            row_end: int,
            col_start: int,
            col_end: int,
            bands: Any = None,
        ) -> np.ndarray:
            if self._arr.ndim == 2:
                return self._arr[row_start:row_end, col_start:col_end]
            # Channels-first: (C, H, W)
            return self._arr[:, row_start:row_end, col_start:col_end]

        def get_shape(self) -> Tuple[int, ...]:
            # Return grdl convention: (rows, cols) or (rows, cols, bands)
            if self._arr.ndim == 2:
                return self._arr.shape
            # Channels-first (C, H, W) → report as (H, W, C) for get_shape
            return (self._arr.shape[1], self._arr.shape[2], self._arr.shape[0])

        def get_dtype(self) -> np.dtype:
            return self._arr.dtype

        def read_full(self, bands: Any = None) -> np.ndarray:
            return self._arr

        def close(self) -> None:
            pass

else:

    class TiledImageCanvas:  # type: ignore[no-redef]
        """Stub when Qt is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("Qt is required for TiledImageCanvas")
