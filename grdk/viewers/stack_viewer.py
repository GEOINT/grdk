# -*- coding: utf-8 -*-
"""
Napari Stack Viewer - Embeddable Qt widget for multi-image viewing.

Wraps napari's ViewerModel and QtViewer to provide a pan/zoom image
viewer with layer switching for each image in a stack. Includes
polygon drawing via napari's shapes layer.

Dependencies
------------
napari

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

# Standard library
from typing import Any, Callable, Dict, List, Optional

# Third-party
import numpy as np

try:
    from napari import Viewer
    from napari.qt import QtViewer
    from napari.layers import Shapes
    _NAPARI_AVAILABLE = True
except ImportError:
    _NAPARI_AVAILABLE = False

try:
    from PyQt6.QtWidgets import QVBoxLayout, QWidget
    from PyQt6.QtCore import pyqtSignal as Signal, QObject
    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False


def _percentile_stretch(img: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Percentile-stretch *img* to ``[0, 1]`` float32, per-band."""
    img = img.astype(np.float32)
    if img.ndim == 2:
        lo, hi = np.nanpercentile(img, low), np.nanpercentile(img, high)
        if hi > lo:
            img = (img - lo) / (hi - lo)
        return np.clip(img, 0.0, 1.0)
    # Multi-band: stretch each band independently
    out = np.empty_like(img, dtype=np.float32)
    # Work band-last; if CYX was already converted to YXC, last axis is bands
    for b in range(img.shape[-1]):
        band = img[..., b]
        lo, hi = np.nanpercentile(band, low), np.nanpercentile(band, high)
        if hi > lo:
            out[..., b] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
        else:
            out[..., b] = 0.0
    return out


class NapariStackViewer:
    """Embeddable napari-based image stack viewer.

    Manages a napari Viewer instance with image layers for each
    image in the stack, plus a shapes layer for polygon drawing.

    Parameters
    ----------
    parent : Optional[QWidget]
        Parent Qt widget to embed into.
    on_polygon_added : Optional[Callable]
        Callback when a polygon is completed.
        Signature: on_polygon_added(vertices: np.ndarray)
        where vertices is shape (N, 2) in (row, col) format.
    """

    def __init__(
        self,
        parent: Optional[Any] = None,
        on_polygon_added: Optional[Callable] = None,
    ) -> None:
        if not _NAPARI_AVAILABLE:
            raise ImportError(
                "napari is required for the stack viewer. "
                "Install with: pip install napari[pyqt6]"
            )

        self._on_polygon_added = on_polygon_added
        self._viewer = Viewer(show=False)
        self._shapes_layer: Optional[Shapes] = None
        self._polygon_count = 0

        # Create Qt widget wrapper
        self._qt_viewer = QtViewer(self._viewer)
        if parent is not None and _QT_AVAILABLE:
            layout = parent.layout()
            if layout is None:
                layout = QVBoxLayout(parent)
                parent.setLayout(layout)
            layout.addWidget(self._qt_viewer)

        # Add shapes layer for polygon drawing
        self._shapes_layer = self._viewer.add_shapes(
            name="Polygons",
            edge_color="yellow",
            edge_width=2,
            face_color=[1, 1, 0, 0.1],
        )
        self._shapes_layer.mode = 'add_polygon'

        # Connect shape change events
        self._shapes_layer.events.data.connect(self._on_shapes_changed)

    @property
    def widget(self) -> Any:
        """The Qt widget for embedding in layouts."""
        return self._qt_viewer

    @property
    def viewer(self) -> Any:
        """The underlying napari Viewer."""
        return self._viewer

    def load_stack(
        self,
        images: List[np.ndarray],
        names: Optional[List[str]] = None,
    ) -> None:
        """Load a stack of images as napari layers.

        Parameters
        ----------
        images : List[np.ndarray]
            Image arrays. Accepted layouts:

            * ``(H, W)`` — single band greyscale
            * ``(H, W, C)`` — multi-band in YXC order (napari native)
            * ``(C, H, W)`` — multi-band in CYX order (GRDL convention);
              automatically transposed to ``(H, W, C)``

            Float32 arrays in ``[0, 1]`` are passed directly to napari.
            Other numeric types are normalised via percentile stretch so
            they display sensibly.
        names : Optional[List[str]]
            Display names for each image.
        """
        # Remove existing image layers (keep shapes)
        for layer in list(self._viewer.layers):
            if layer is not self._shapes_layer:
                self._viewer.layers.remove(layer)

        for i, img in enumerate(images):
            name = names[i] if names and i < len(names) else f"Image {i}"

            # ── Axis-order normalisation ────────────────────────────────
            # Convert CYX → YXC so napari gets the correct layout.
            # Heuristic: the leading dimension is channels when it is
            # small (≤ 16) and both trailing dimensions are larger.
            if (img.ndim == 3
                    and img.shape[0] <= 16
                    and img.shape[1] > img.shape[0]
                    and img.shape[2] > img.shape[0]):
                img = np.moveaxis(img, 0, -1)  # (C, H, W) → (H, W, C)

            # ── Complex SAR imagery ─────────────────────────────────────
            if np.iscomplexobj(img):
                img = np.abs(img).astype(np.float32)

            # ── Intensity normalisation ─────────────────────────────────
            # float32 already in [0,1] (e.g. Pauli RGB): pass through.
            # Everything else: per-band percentile stretch → [0,1].
            if img.dtype != np.float32 or img.max() > 1.0 or img.min() < 0.0:
                img = _percentile_stretch(img)

            # ── Add to napari ───────────────────────────────────────────
            if img.ndim == 3 and img.shape[2] >= 3:
                self._viewer.add_image(img[:, :, :3], name=name, rgb=True)
            elif img.ndim == 3:
                self._viewer.add_image(img[:, :, 0], name=name)
            else:
                self._viewer.add_image(img, name=name)

        # Reset view
        self._viewer.reset_view()

    def get_polygons(self) -> List[np.ndarray]:
        """Get all drawn polygons.

        Returns
        -------
        List[np.ndarray]
            List of polygon vertex arrays, each shape (N, 2)
            in (row, col) format.
        """
        if self._shapes_layer is None:
            return []
        return [np.array(shape) for shape in self._shapes_layer.data]

    def clear_polygons(self) -> None:
        """Remove all drawn polygons."""
        if self._shapes_layer is not None:
            self._shapes_layer.data = []
            self._polygon_count = 0

    def set_polygon_mode(self) -> None:
        """Switch to polygon drawing mode."""
        if self._shapes_layer is not None:
            self._shapes_layer.mode = 'add_polygon'

    def _on_shapes_changed(self, event: Any) -> None:
        """Handle polygon additions."""
        current_count = len(self._shapes_layer.data)
        if current_count > self._polygon_count:
            self._polygon_count = current_count
            if self._on_polygon_added:
                new_polygon = np.array(
                    self._shapes_layer.data[-1]
                )
                self._on_polygon_added(new_polygon)

    def close(self) -> None:
        """Close the viewer and release resources."""
        try:
            self._viewer.close()
        except Exception:
            pass
