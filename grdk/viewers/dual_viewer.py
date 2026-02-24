# -*- coding: utf-8 -*-
"""
DualGeoViewer - Dual-pane geospatial image viewer with synchronized navigation.

Provides side-by-side image comparison with synchronized pan/zoom
(pixel or geographic), independent display controls per pane, and
optional crop-to-overlap for partially overlapping geolocated images.

Components
----------
- ``SyncController`` — mediates pan/zoom sync between two canvases
- ``SyncBar`` — compact vertical control strip between panes
- ``DualGeoViewer`` — top-level dual-pane composite widget

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
2026-02-20

Modified
--------
2026-02-20
"""

# Standard library
import logging
from typing import Any, Optional, Tuple

# Third-party
import numpy as np

_log = logging.getLogger("grdk.dual_viewer")

try:
    from PyQt6.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QPushButton,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )
    from PyQt6.QtCore import QEvent, QObject, Qt, pyqtSignal as Signal

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False

from grdk.viewers.image_canvas import DisplaySettings


# ---------------------------------------------------------------------------
# Overlap utilities (no Qt dependency)
# ---------------------------------------------------------------------------

def compute_geo_bounds(
    geolocation: Any,
    rows: int,
    cols: int,
) -> Optional[Tuple[float, float, float, float]]:
    """Compute geographic bounding box from a geolocation model.

    Transforms the four image corners to lat/lon and returns the
    axis-aligned bounding box.

    Parameters
    ----------
    geolocation : Geolocation
        grdl geolocation with ``image_to_latlon(row, col)``.
    rows : int
        Image height in pixels.
    cols : int
        Image width in pixels.

    Returns
    -------
    Optional[Tuple[float, float, float, float]]
        ``(lat_min, lat_max, lon_min, lon_max)``, or ``None`` on error.
    """
    corners = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]
    lats, lons = [], []
    for r, c in corners:
        try:
            result = geolocation.image_to_latlon(r, c)
            if isinstance(result, tuple) and len(result) >= 2:
                lats.append(float(result[0]))
                lons.append(float(result[1]))
        except Exception:
            return None
    if len(lats) < 4:
        return None
    return (min(lats), max(lats), min(lons), max(lons))


def compute_overlap(
    left_geo: Any,
    left_shape: Tuple[int, int],
    right_geo: Any,
    right_shape: Tuple[int, int],
) -> Optional[Tuple[float, float, float, float]]:
    """Compute the geographic bounding box intersection of two images.

    Parameters
    ----------
    left_geo : Geolocation
        Left image geolocation.
    left_shape : Tuple[int, int]
        ``(rows, cols)`` of left image.
    right_geo : Geolocation
        Right image geolocation.
    right_shape : Tuple[int, int]
        ``(rows, cols)`` of right image.

    Returns
    -------
    Optional[Tuple[float, float, float, float]]
        ``(lat_min, lat_max, lon_min, lon_max)`` of overlap, or
        ``None`` if no overlap or geolocation unavailable.
    """
    if left_geo is None or right_geo is None:
        return None

    left_bounds = compute_geo_bounds(left_geo, *left_shape)
    right_bounds = compute_geo_bounds(right_geo, *right_shape)
    if left_bounds is None or right_bounds is None:
        return None

    lat_min = max(left_bounds[0], right_bounds[0])
    lat_max = min(left_bounds[1], right_bounds[1])
    lon_min = max(left_bounds[2], right_bounds[2])
    lon_max = min(left_bounds[3], right_bounds[3])

    if lat_min >= lat_max or lon_min >= lon_max:
        return None

    return (lat_min, lat_max, lon_min, lon_max)


# ---------------------------------------------------------------------------
# SyncController
# ---------------------------------------------------------------------------

if _QT_AVAILABLE:
    from grdk.viewers.tiled_canvas import TiledImageCanvas

    class SyncController(QObject):
        """Mediates synchronized pan/zoom between two TiledImageCanvas instances.

        Supports pixel-based and geolocation-based viewport sync.  Uses
        a re-entrancy guard to prevent infinite sync loops.

        Parameters
        ----------
        parent : QObject, optional
            Parent object.

        Signals
        -------
        sync_mode_changed(str)
            Emitted when the sync mode changes.
        overlap_changed(bool)
            Emitted when geographic overlap availability changes.
        """

        sync_mode_changed = Signal(str)
        overlap_changed = Signal(bool)

        def __init__(self, parent: Optional[QObject] = None) -> None:
            super().__init__(parent)

            self._left_canvas: Optional[TiledImageCanvas] = None
            self._right_canvas: Optional[TiledImageCanvas] = None
            self._left_geo: Optional[Any] = None
            self._right_geo: Optional[Any] = None
            self._left_shape: Tuple[int, int] = (0, 0)
            self._right_shape: Tuple[int, int] = (0, 0)

            self._sync_mode: str = "pixel"  # "pixel" | "geo" | "none"
            self._syncing: bool = False  # Re-entrancy guard
            self._enabled: bool = True

        # --- Configuration ---

        def set_canvases(
            self,
            left: TiledImageCanvas,
            right: TiledImageCanvas,
        ) -> None:
            """Connect to two canvases for viewport synchronization.

            Parameters
            ----------
            left : TiledImageCanvas
                Left pane canvas.
            right : TiledImageCanvas
                Right pane canvas.
            """
            # Disconnect previous if any
            if self._left_canvas is not None:
                self._left_canvas.viewport_changed.disconnect(
                    self._on_left_viewport_changed
                )
            if self._right_canvas is not None:
                self._right_canvas.viewport_changed.disconnect(
                    self._on_right_viewport_changed
                )

            self._left_canvas = left
            self._right_canvas = right

            left.viewport_changed.connect(self._on_left_viewport_changed)
            right.viewport_changed.connect(self._on_right_viewport_changed)
            _log.debug("SyncController: canvases connected")

        def set_geolocations(
            self,
            left_geo: Optional[Any],
            left_shape: Tuple[int, int],
            right_geo: Optional[Any],
            right_shape: Tuple[int, int],
        ) -> None:
            """Update geolocation models and recompute overlap.

            Parameters
            ----------
            left_geo : Optional[Geolocation]
                Left image geolocation.
            left_shape : Tuple[int, int]
                ``(rows, cols)`` of left image.
            right_geo : Optional[Geolocation]
                Right image geolocation.
            right_shape : Tuple[int, int]
                ``(rows, cols)`` of right image.
            """
            self._left_geo = left_geo
            self._left_shape = left_shape
            self._right_geo = right_geo
            self._right_shape = right_shape

            has_overlap = self.get_overlap() is not None
            _log.info(
                "SyncController: geolocations updated, left=%s, right=%s, overlap=%s",
                type(left_geo).__name__ if left_geo else None,
                type(right_geo).__name__ if right_geo else None,
                has_overlap,
            )
            self.overlap_changed.emit(has_overlap)

            # If geo mode but no overlap, fall back to pixel
            if self._sync_mode == "geo" and not has_overlap:
                _log.info("SyncController: no overlap, falling back to pixel sync")
                self.set_sync_mode("pixel")

        def set_sync_mode(self, mode: str) -> None:
            """Set the synchronization mode.

            Parameters
            ----------
            mode : str
                One of ``"pixel"``, ``"geo"``, ``"none"``.
            """
            if mode not in ("pixel", "geo", "none"):
                raise ValueError(f"Invalid sync mode: {mode!r}")
            _log.info("SyncController: sync mode -> %s", mode)
            self._sync_mode = mode
            self.sync_mode_changed.emit(mode)

        @property
        def sync_mode(self) -> str:
            """Current sync mode."""
            return self._sync_mode

        @property
        def enabled(self) -> bool:
            """Whether sync is active (mode != 'none')."""
            return self._sync_mode != "none"

        def set_enabled(self, enabled: bool) -> None:
            """Toggle sync on/off.

            Parameters
            ----------
            enabled : bool
                If False, sets mode to "none".  If True, restores
                to "pixel" (or "geo" if overlap exists).
            """
            if enabled:
                if self.get_overlap() is not None:
                    self.set_sync_mode("geo")
                else:
                    self.set_sync_mode("pixel")
            else:
                self.set_sync_mode("none")

        def get_overlap(self) -> Optional[Tuple[float, float, float, float]]:
            """Compute the current geographic overlap, if any.

            Returns
            -------
            Optional[Tuple[float, float, float, float]]
                ``(lat_min, lat_max, lon_min, lon_max)`` or ``None``.
            """
            return compute_overlap(
                self._left_geo, self._left_shape,
                self._right_geo, self._right_shape,
            )

        # --- Internal sync handlers ---

        def _on_left_viewport_changed(self) -> None:
            """Sync right canvas to match left viewport."""
            self._sync_viewport(
                self._left_canvas, self._right_canvas,
                self._left_geo, self._right_geo,
            )

        def _on_right_viewport_changed(self) -> None:
            """Sync left canvas to match right viewport."""
            self._sync_viewport(
                self._right_canvas, self._left_canvas,
                self._right_geo, self._left_geo,
            )

        def _sync_viewport(
            self,
            source: Optional[TiledImageCanvas],
            target: Optional[TiledImageCanvas],
            source_geo: Optional[Any],
            target_geo: Optional[Any],
        ) -> None:
            """Sync target canvas viewport to match source.

            Uses a re-entrancy guard to prevent infinite loops.
            """
            if self._syncing or self._sync_mode == "none":
                return
            if source is None or target is None:
                return

            self._syncing = True
            try:
                src_row, src_col = source.get_viewport_center()
                src_zoom = source.get_zoom()

                if (
                    self._sync_mode == "geo"
                    and source_geo is not None
                    and target_geo is not None
                ):
                    try:
                        result = source_geo.image_to_latlon(src_row, src_col)
                        if isinstance(result, tuple) and len(result) >= 2:
                            lat, lon = float(result[0]), float(result[1])
                            tgt = target_geo.latlon_to_image(lat, lon)
                            if isinstance(tgt, tuple) and len(tgt) >= 2:
                                tgt_row, tgt_col = float(tgt[0]), float(tgt[1])
                            else:
                                tgt_row, tgt_col = src_row, src_col
                        else:
                            tgt_row, tgt_col = src_row, src_col
                    except Exception:
                        tgt_row, tgt_col = src_row, src_col
                else:
                    tgt_row, tgt_col = src_row, src_col

                target.center_on(tgt_row, tgt_col)
                target.zoom_to(src_zoom)
            finally:
                self._syncing = False


    # -------------------------------------------------------------------
    # SyncBar
    # -------------------------------------------------------------------

    class SyncBar(QWidget):
        """Compact vertical control strip between dual viewer panes.

        Provides sync toggle, pixel/geo mode switch, and
        crop-to-overlap button.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget.

        Signals
        -------
        sync_toggled(bool)
            Emitted when the sync toggle changes.
        mode_changed(str)
            Emitted when the mode switch changes ("pixel" or "geo").
        crop_requested()
            Emitted when the crop-to-overlap button is clicked.
        reset_requested()
            Emitted when the reset (undo crop) button is clicked.
        """

        sync_toggled = Signal(bool)
        mode_changed = Signal(str)
        crop_requested = Signal()
        reset_requested = Signal()

        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)

            self.setFixedWidth(32)

            layout = QVBoxLayout(self)
            layout.setContentsMargins(2, 4, 2, 4)
            layout.setSpacing(4)

            # Sync toggle
            self._sync_btn = QPushButton("\u26d3", self)  # ⛓ chain
            self._sync_btn.setCheckable(True)
            self._sync_btn.setChecked(True)
            self._sync_btn.setToolTip("Toggle pan/zoom sync")
            self._sync_btn.setFixedSize(28, 28)
            self._sync_btn.toggled.connect(self._on_sync_toggled)
            layout.addWidget(self._sync_btn)

            # Mode toggle: Pixel / Geo
            self._mode_btn = QPushButton("P", self)
            self._mode_btn.setToolTip("Sync mode: Pixel")
            self._mode_btn.setFixedSize(28, 28)
            self._mode_btn.clicked.connect(self._on_mode_clicked)
            self._current_mode = "pixel"
            layout.addWidget(self._mode_btn)

            # Crop to overlap
            self._crop_btn = QPushButton("\u2702", self)  # ✂ scissors
            self._crop_btn.setToolTip("Crop to geographic overlap")
            self._crop_btn.setFixedSize(28, 28)
            self._crop_btn.setEnabled(False)
            self._crop_btn.clicked.connect(self._on_crop_clicked)
            layout.addWidget(self._crop_btn)

            # Reset (undo crop)
            self._reset_btn = QPushButton("\u21ba", self)  # ↺ reset
            self._reset_btn.setToolTip("Reset to full images")
            self._reset_btn.setFixedSize(28, 28)
            self._reset_btn.setEnabled(False)
            self._reset_btn.setVisible(False)
            self._reset_btn.clicked.connect(self._on_reset_clicked)
            layout.addWidget(self._reset_btn)

            layout.addStretch(1)

        # --- Public API ---

        def set_geo_available(self, available: bool) -> None:
            """Enable or disable geo-mode and crop buttons.

            Parameters
            ----------
            available : bool
                Whether both images have geolocation and overlap.
            """
            if not available and self._current_mode == "geo":
                self._set_mode("pixel")
            self._mode_btn.setEnabled(available or self._current_mode == "pixel")
            self._crop_btn.setEnabled(available)

        def set_overlap_available(self, available: bool) -> None:
            """Enable or disable the crop-to-overlap button.

            Parameters
            ----------
            available : bool
                Whether geographic overlap exists.
            """
            self._crop_btn.setEnabled(available)

        def set_cropped(self, cropped: bool) -> None:
            """Update UI state after crop/reset.

            Parameters
            ----------
            cropped : bool
                Whether images are currently cropped to overlap.
            """
            self._crop_btn.setVisible(not cropped)
            self._reset_btn.setVisible(cropped)
            self._reset_btn.setEnabled(cropped)

        # --- Internal ---

        def _on_sync_toggled(self, checked: bool) -> None:
            if checked:
                self._sync_btn.setText("\u26d3")  # chain
                self._sync_btn.setToolTip("Sync enabled — click to disable")
            else:
                self._sync_btn.setText("\u26a0")  # warning/broken
                self._sync_btn.setToolTip("Sync disabled — click to enable")
            self.sync_toggled.emit(checked)

        def _on_mode_clicked(self) -> None:
            if self._current_mode == "pixel":
                self._set_mode("geo")
            else:
                self._set_mode("pixel")

        def _set_mode(self, mode: str) -> None:
            self._current_mode = mode
            if mode == "geo":
                self._mode_btn.setText("G")
                self._mode_btn.setToolTip("Sync mode: Geographic")
            else:
                self._mode_btn.setText("P")
                self._mode_btn.setToolTip("Sync mode: Pixel")
            self.mode_changed.emit(mode)

        def _on_crop_clicked(self) -> None:
            self.crop_requested.emit()

        def _on_reset_clicked(self) -> None:
            self.reset_requested.emit()


    # -------------------------------------------------------------------
    # DualGeoViewer
    # -------------------------------------------------------------------

    class DualGeoViewer(QWidget):
        """Dual-pane geospatial image viewer with synchronized navigation.

        Wraps two ``GeoImageViewer`` instances in a QSplitter with a
        shared ``CoordinateBar``, a ``SyncBar`` between the panes, and
        a ``SyncController`` for synchronized pan/zoom.

        Supports single-pane mode (right pane hidden) and dual-pane
        mode.  In single mode the left pane fills the full width.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget.

        Signals
        -------
        active_pane_changed(int)
            Emitted when the active pane changes (0=left, 1=right).
        band_info_changed(list)
            Forwarded from the active pane.
        mode_changed(str)
            Emitted on ``"single"`` / ``"dual"`` switch.
        """

        active_pane_changed = Signal(int)
        band_info_changed = Signal(list)
        pane_band_info_changed = Signal(int, list)  # (pane_index, band_info)
        mode_changed = Signal(str)

        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)

            from grdk.viewers.geo_viewer import GeoImageViewer
            from grdk.viewers.coordinate_bar import CoordinateBar

            self._mode: str = "single"
            self._active_pane: int = 0
            self._cropped: bool = False

            # Store original readers/geolocations for crop reset
            self._left_reader: Optional[Any] = None
            self._left_orig_geo: Optional[Any] = None
            self._right_reader: Optional[Any] = None
            self._right_orig_geo: Optional[Any] = None

            # --- Panes ---
            self._left_viewer = GeoImageViewer(self)
            self._right_viewer = GeoImageViewer(self)

            # Hide per-viewer coordinate bars (we use a shared one)
            self._left_viewer.coord_bar.hide()
            self._right_viewer.coord_bar.hide()

            # --- Sync ---
            self._sync_controller = SyncController(self)
            self._sync_controller.set_canvases(
                self._left_viewer.canvas,
                self._right_viewer.canvas,
            )
            self._sync_bar = SyncBar(self)

            # --- Shared coordinate bar ---
            self._coord_bar = CoordinateBar(self)

            # Connect pixel_hovered from both canvases
            self._left_viewer.canvas.pixel_hovered.connect(
                self._on_left_pixel_hovered,
            )
            self._right_viewer.canvas.pixel_hovered.connect(
                self._on_right_pixel_hovered,
            )

            # --- Layout ---
            self._splitter = QSplitter(Qt.Orientation.Horizontal, self)
            self._splitter.addWidget(self._left_viewer)
            self._splitter.addWidget(self._right_viewer)
            self._splitter.setChildrenCollapsible(False)

            top_layout = QHBoxLayout()
            top_layout.setContentsMargins(0, 0, 0, 0)
            top_layout.setSpacing(0)
            top_layout.addWidget(self._splitter, 1)
            top_layout.addWidget(self._sync_bar, 0)

            main_layout = QVBoxLayout(self)
            main_layout.setContentsMargins(0, 0, 0, 0)
            main_layout.setSpacing(0)
            main_layout.addLayout(top_layout, 1)
            main_layout.addWidget(self._coord_bar, 0)

            # --- Wire signals ---
            self._left_viewer.band_info_changed.connect(
                self._on_left_band_info,
            )
            self._right_viewer.band_info_changed.connect(
                self._on_right_band_info,
            )
            self._sync_bar.sync_toggled.connect(self._on_sync_toggled)
            self._sync_bar.mode_changed.connect(self._on_sync_mode_changed)
            self._sync_bar.crop_requested.connect(self.crop_to_overlap)
            self._sync_bar.reset_requested.connect(self.reset_crop)
            self._sync_controller.overlap_changed.connect(
                self._sync_bar.set_overlap_available,
            )

            # --- Active pane tracking via event filters ---
            self._left_viewer.installEventFilter(self)
            self._right_viewer.installEventFilter(self)

            # Start in single mode
            self._right_viewer.hide()
            self._sync_bar.hide()

        # --- Properties ---

        @property
        def left_viewer(self) -> Any:
            """Left-pane GeoImageViewer."""
            return self._left_viewer

        @property
        def right_viewer(self) -> Any:
            """Right-pane GeoImageViewer."""
            return self._right_viewer

        @property
        def active_pane(self) -> int:
            """Index of the active pane (0=left, 1=right)."""
            return self._active_pane

        @property
        def active_viewer(self) -> Any:
            """The active GeoImageViewer."""
            if self._active_pane == 1 and self._mode == "dual":
                return self._right_viewer
            return self._left_viewer

        @property
        def active_canvas(self) -> TiledImageCanvas:
            """The active pane's TiledImageCanvas."""
            return self.active_viewer.canvas

        @property
        def canvas(self) -> TiledImageCanvas:
            """Alias for active_canvas (backward compatibility)."""
            return self.active_canvas

        @property
        def sync_controller(self) -> SyncController:
            """The SyncController instance."""
            return self._sync_controller

        @property
        def metadata(self) -> Optional[Any]:
            """Metadata from the active viewer."""
            return self.active_viewer.metadata

        @property
        def geolocation(self) -> Optional[Any]:
            """Geolocation from the active viewer."""
            return self.active_viewer.geolocation

        # --- Mode ---

        def set_mode(self, mode: str) -> None:
            """Switch between single and dual pane mode.

            Parameters
            ----------
            mode : str
                ``"single"`` or ``"dual"``.
            """
            if mode not in ("single", "dual"):
                raise ValueError(f"Invalid mode: {mode!r}")
            if mode == self._mode:
                return

            _log.info("DualGeoViewer: mode -> %s", mode)
            self._mode = mode

            if mode == "dual":
                self._right_viewer.show()
                self._sync_bar.show()
                # Equal split
                total = self._splitter.width()
                self._splitter.setSizes([total // 2, total // 2])
                # Show active pane indicator
                if self._active_pane == 0:
                    self._left_viewer.setStyleSheet(self._ACTIVE_BORDER)
                    self._right_viewer.setStyleSheet(self._INACTIVE_BORDER)
                else:
                    self._left_viewer.setStyleSheet(self._INACTIVE_BORDER)
                    self._right_viewer.setStyleSheet(self._ACTIVE_BORDER)
            else:
                self._right_viewer.hide()
                self._sync_bar.hide()
                # Clear pane borders in single mode
                self._left_viewer.setStyleSheet("")
                self._right_viewer.setStyleSheet("")
                # Ensure active pane is left in single mode
                if self._active_pane == 1:
                    self._active_pane = 0
                    self.active_pane_changed.emit(0)

            self.mode_changed.emit(mode)

        @property
        def mode(self) -> str:
            """Current mode: ``"single"`` or ``"dual"``."""
            return self._mode

        # --- Loading API ---

        def open_file(self, filepath: str, pane: int = 0) -> None:
            """Open an image file in the specified pane.

            Parameters
            ----------
            filepath : str
                Path to image file or directory.
            pane : int
                0 for left pane, 1 for right pane.
            """
            _log.info("DualGeoViewer: open_file(%r, pane=%d)", filepath, pane)
            viewer = self._left_viewer if pane == 0 else self._right_viewer
            viewer.open_file(filepath)
            self._update_after_load(pane)

        def open_reader(
            self,
            reader: Any,
            geolocation: Optional[Any] = None,
            pane: int = 0,
        ) -> None:
            """Open an image from a grdl reader in the specified pane.

            Parameters
            ----------
            reader : ImageReader
                grdl reader instance.
            geolocation : Geolocation, optional
                Geolocation model.
            pane : int
                0 for left pane, 1 for right pane.
            """
            _log.info(
                "DualGeoViewer: open_reader(%s, geo=%s, pane=%d)",
                type(reader).__name__, type(geolocation).__name__ if geolocation else None, pane,
            )
            viewer = self._left_viewer if pane == 0 else self._right_viewer
            viewer.open_reader(reader, geolocation=geolocation)
            self._update_after_load(pane)

        def set_array(
            self,
            arr: Any,
            geolocation: Optional[Any] = None,
            pane: int = 0,
        ) -> None:
            """Display a numpy array in the specified pane.

            Parameters
            ----------
            arr : np.ndarray
                Image data.
            geolocation : Geolocation, optional
                Geolocation model.
            pane : int
                0 for left pane, 1 for right pane.
            """
            viewer = self._left_viewer if pane == 0 else self._right_viewer
            viewer.set_array(arr, geolocation=geolocation)
            self._update_after_load(pane)

        def _update_after_load(self, pane: int) -> None:
            """Update sync controller and overlap state after loading."""
            # Store reader/geo references for crop reset
            if pane == 0:
                self._left_reader = self._left_viewer._reader
                self._left_orig_geo = self._left_viewer.geolocation
            else:
                self._right_reader = self._right_viewer._reader
                self._right_orig_geo = self._right_viewer.geolocation

            # Reset crop state
            if self._cropped:
                self._cropped = False
                self._sync_bar.set_cropped(False)

            # Update geolocations for sync
            left_shape = self._get_image_shape(self._left_viewer)
            right_shape = self._get_image_shape(self._right_viewer)

            self._sync_controller.set_geolocations(
                self._left_viewer.geolocation, left_shape,
                self._right_viewer.geolocation, right_shape,
            )

            # Update coordinate bar geolocation for active pane
            self._coord_bar.set_geolocation(self.active_viewer.geolocation)

        @staticmethod
        def _get_image_shape(viewer: Any) -> Tuple[int, int]:
            """Extract (rows, cols) from a viewer's loaded image."""
            if viewer._reader is not None:
                try:
                    shape = viewer._reader.get_shape()
                    return (shape[0], shape[1])
                except Exception:
                    pass
            canvas = viewer.canvas
            if canvas.source_array is not None:
                arr = canvas.source_array
                if arr.ndim == 2:
                    return arr.shape
                return (arr.shape[1], arr.shape[2])
            return (0, 0)

        # --- Vector overlays ---

        def load_vector(self, filepath: str, pane: Optional[int] = None) -> None:
            """Load a GeoJSON vector overlay.

            Parameters
            ----------
            filepath : str
                Path to GeoJSON file.
            pane : int, optional
                Target pane.  If None, uses the active pane.
            """
            if pane is None:
                pane = self._active_pane
            viewer = self._left_viewer if pane == 0 else self._right_viewer
            viewer.load_vector(filepath)

        def clear_vectors(self, pane: Optional[int] = None) -> None:
            """Clear vector overlays.

            Parameters
            ----------
            pane : int, optional
                Target pane.  If None, clears both panes.
            """
            if pane is None:
                self._left_viewer.clear_vectors()
                self._right_viewer.clear_vectors()
            else:
                viewer = (
                    self._left_viewer if pane == 0 else self._right_viewer
                )
                viewer.clear_vectors()

        # --- Export ---

        def export_view(self, filepath: str, pane: Optional[int] = None) -> None:
            """Export the current view from a pane.

            Parameters
            ----------
            filepath : str
                Output file path.
            pane : int, optional
                Target pane.  If None, uses the active pane.
            """
            if pane is None:
                pane = self._active_pane
            viewer = self._left_viewer if pane == 0 else self._right_viewer
            viewer.export_view(filepath)

        # --- Crop to overlap ---

        def crop_to_overlap(self) -> None:
            """Crop both panes to show only the geographic overlap region.

            Requires both images to be loaded with geolocation and to
            have a geographic overlap.  Does nothing otherwise.
            """
            overlap = self._sync_controller.get_overlap()
            if overlap is None:
                _log.warning("crop_to_overlap: no geographic overlap")
                return

            _log.info("crop_to_overlap: overlap=%s", overlap)

            lat_min, lat_max, lon_min, lon_max = overlap

            for viewer, reader, geo in [
                (self._left_viewer, self._left_reader, self._left_orig_geo),
                (self._right_viewer, self._right_reader, self._right_orig_geo),
            ]:
                if reader is None or geo is None:
                    continue

                try:
                    # Convert geographic bounds to pixel bounds
                    corners_lat = [lat_min, lat_min, lat_max, lat_max]
                    corners_lon = [lon_min, lon_max, lon_min, lon_max]
                    pixel_rows, pixel_cols = [], []
                    for lat, lon in zip(corners_lat, corners_lon):
                        result = geo.latlon_to_image(lat, lon)
                        if isinstance(result, tuple) and len(result) >= 2:
                            pixel_rows.append(int(result[0]))
                            pixel_cols.append(int(result[1]))

                    if len(pixel_rows) < 4:
                        continue

                    r0 = max(0, min(pixel_rows))
                    r1 = max(pixel_rows)
                    c0 = max(0, min(pixel_cols))
                    c1 = max(pixel_cols)

                    # Read the overlap chip
                    chip = reader.read_chip(r0, r1, c0, c1)
                    viewer.set_array(chip, geolocation=geo)
                except Exception:
                    continue

            self._cropped = True
            self._sync_bar.set_cropped(True)

        def reset_crop(self) -> None:
            """Restore full images after crop-to-overlap."""
            if not self._cropped:
                return
            _log.info("reset_crop: restoring full images")

            if self._left_reader is not None:
                self._left_viewer.open_reader(
                    self._left_reader,
                    geolocation=self._left_orig_geo,
                )
            if self._right_reader is not None:
                self._right_viewer.open_reader(
                    self._right_reader,
                    geolocation=self._right_orig_geo,
                )

            self._cropped = False
            self._sync_bar.set_cropped(False)

        # --- Coordinate bar ---

        def _on_left_pixel_hovered(
            self, row: int, col: int, value: Any,
        ) -> None:
            """Forward left canvas hover to shared coordinate bar."""
            if self._active_pane == 0:
                self._coord_bar._on_pixel_hovered(row, col, value)

        def _on_right_pixel_hovered(
            self, row: int, col: int, value: Any,
        ) -> None:
            """Forward right canvas hover to shared coordinate bar."""
            if self._active_pane == 1:
                self._coord_bar._on_pixel_hovered(row, col, value)

        # --- Band info forwarding ---

        def _on_left_band_info(self, info: list) -> None:
            self.pane_band_info_changed.emit(0, info)
            if self._active_pane == 0:
                self.band_info_changed.emit(info)

        def _on_right_band_info(self, info: list) -> None:
            self.pane_band_info_changed.emit(1, info)
            if self._active_pane == 1:
                self.band_info_changed.emit(info)

        # --- Sync bar handlers ---

        def _on_sync_toggled(self, enabled: bool) -> None:
            self._sync_controller.set_enabled(enabled)

        def _on_sync_mode_changed(self, mode: str) -> None:
            self._sync_controller.set_sync_mode(mode)

        # --- Active pane tracking ---

        def eventFilter(self, obj: QObject, event: QEvent) -> bool:
            """Track which pane the cursor is over."""
            if event.type() == QEvent.Type.Enter:
                if obj is self._left_viewer:
                    self._set_active_pane(0)
                elif obj is self._right_viewer:
                    self._set_active_pane(1)
            return super().eventFilter(obj, event)

        _ACTIVE_BORDER = (
            "border: 2px solid #4A90D9; border-radius: 2px;"
        )
        _INACTIVE_BORDER = "border: none;"

        def _set_active_pane(self, pane: int) -> None:
            """Update the active pane and notify listeners."""
            if pane == self._active_pane:
                return
            _log.debug("Active pane -> %d", pane)
            self._active_pane = pane
            # Update shared coordinate bar geolocation
            self._coord_bar.set_geolocation(self.active_viewer.geolocation)
            # Visual indicator: highlight active pane with a border
            if self._mode == "dual":
                if pane == 0:
                    self._left_viewer.setStyleSheet(self._ACTIVE_BORDER)
                    self._right_viewer.setStyleSheet(self._INACTIVE_BORDER)
                else:
                    self._left_viewer.setStyleSheet(self._INACTIVE_BORDER)
                    self._right_viewer.setStyleSheet(self._ACTIVE_BORDER)
            self.active_pane_changed.emit(pane)

else:

    class SyncController:  # type: ignore[no-redef]
        """Stub when Qt is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("Qt is required for SyncController")

    class SyncBar:  # type: ignore[no-redef]
        """Stub when Qt is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("Qt is required for SyncBar")

    class DualGeoViewer:  # type: ignore[no-redef]
        """Stub when Qt is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("Qt is required for DualGeoViewer")
