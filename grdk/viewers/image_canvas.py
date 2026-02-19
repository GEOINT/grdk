# -*- coding: utf-8 -*-
"""
ImageCanvas - Interactive image viewer with pan, zoom, and display controls.

Provides a QGraphicsView-based image canvas that renders numpy arrays
with configurable display settings (contrast, brightness, gamma,
window/level, colormaps). Designed as the shared base for all GRDK
image display components.

Dependencies
------------
PyQt6

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
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

# Third-party
import numpy as np

try:
    from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QRubberBand
    from PyQt6.QtGui import QImage, QPixmap, QPainter
    from PyQt6.QtCore import QPoint, QRect, QSize, Qt, pyqtSignal as Signal

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Colormap LUTs (256 × 3 uint8, sampled from matplotlib published tables)
# ---------------------------------------------------------------------------

def _make_viridis_lut() -> np.ndarray:
    """Generate viridis colormap LUT (256 entries)."""
    # Key control points sampled from matplotlib viridis
    points = np.array([
        [68, 1, 84], [72, 35, 116], [64, 67, 135], [52, 94, 141],
        [41, 120, 142], [32, 144, 140], [34, 167, 132], [68, 190, 112],
        [121, 209, 81], [189, 222, 38], [253, 231, 37],
    ], dtype=np.float64)
    indices = np.linspace(0, len(points) - 1, 256)
    lut = np.zeros((256, 3), dtype=np.uint8)
    for c in range(3):
        lut[:, c] = np.interp(indices, np.arange(len(points)), points[:, c])
    return lut


def _make_inferno_lut() -> np.ndarray:
    """Generate inferno colormap LUT (256 entries)."""
    points = np.array([
        [0, 0, 4], [22, 11, 57], [66, 10, 104], [106, 23, 110],
        [147, 38, 103], [186, 54, 85], [221, 81, 58], [243, 118, 27],
        [249, 166, 10], [240, 215, 66], [252, 255, 164],
    ], dtype=np.float64)
    indices = np.linspace(0, len(points) - 1, 256)
    lut = np.zeros((256, 3), dtype=np.uint8)
    for c in range(3):
        lut[:, c] = np.interp(indices, np.arange(len(points)), points[:, c])
    return lut


def _make_plasma_lut() -> np.ndarray:
    """Generate plasma colormap LUT (256 entries)."""
    points = np.array([
        [13, 8, 135], [75, 3, 161], [126, 3, 168], [168, 34, 150],
        [199, 63, 125], [224, 100, 97], [241, 140, 73], [248, 181, 48],
        [241, 222, 36], [240, 249, 33],
    ], dtype=np.float64)
    indices = np.linspace(0, len(points) - 1, 256)
    lut = np.zeros((256, 3), dtype=np.uint8)
    for c in range(3):
        lut[:, c] = np.interp(indices, np.arange(len(points)), points[:, c])
    return lut


def _make_hot_lut() -> np.ndarray:
    """Generate hot colormap LUT (256 entries)."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        # Red ramps first, then green, then blue
        r = min(1.0, t * 2.5)
        g = max(0.0, min(1.0, (t - 0.4) * 2.5))
        b = max(0.0, min(1.0, (t - 0.8) * 5.0))
        lut[i] = [int(r * 255), int(g * 255), int(b * 255)]
    return lut


# Lazy-initialized colormap registry
_COLORMAPS: Optional[dict] = None


def _get_colormaps() -> dict:
    """Return colormap LUT registry, building on first access."""
    global _COLORMAPS
    if _COLORMAPS is None:
        _COLORMAPS = {
            'viridis': _make_viridis_lut(),
            'inferno': _make_inferno_lut(),
            'plasma': _make_plasma_lut(),
            'hot': _make_hot_lut(),
        }
    return _COLORMAPS


AVAILABLE_COLORMAPS = ('grayscale', 'viridis', 'inferno', 'plasma', 'hot')


# ---------------------------------------------------------------------------
# DisplaySettings
# ---------------------------------------------------------------------------

@dataclass
class DisplaySettings:
    """Visual rendering parameters for image display.

    These settings control how a source numpy array is mapped to
    display pixels. They do NOT modify the source data.

    Parameters
    ----------
    window_min : Optional[float]
        Manual minimum value for window/level. None = auto from data.
    window_max : Optional[float]
        Manual maximum value for window/level. None = auto from data.
    percentile_low : float
        Lower percentile for auto windowing (0-100). Default 0.0.
    percentile_high : float
        Upper percentile for auto windowing (0-100). Default 100.0.
    brightness : float
        Additive brightness offset in [-1.0, 1.0]. Default 0.0.
    contrast : float
        Multiplicative contrast scale in [0.0, 3.0]. Default 1.0.
    colormap : str
        Colormap name. One of: 'grayscale', 'viridis', 'inferno',
        'plasma', 'hot'. Default 'grayscale'.
    band_index : Optional[int]
        For multi-band arrays, which band to display. None = auto
        (3-band as RGB, otherwise first band as grayscale).
    gamma : float
        Gamma correction in [0.1, 5.0]. Default 1.0 (linear).
    """

    window_min: Optional[float] = None
    window_max: Optional[float] = None
    percentile_low: float = 0.0
    percentile_high: float = 100.0
    brightness: float = 0.0
    contrast: float = 1.0
    colormap: str = 'grayscale'
    band_index: Optional[int] = None
    gamma: float = 1.0
    remap_function: Optional[Callable] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Pure functions (no Qt dependency)
# ---------------------------------------------------------------------------

def normalize_array(
    arr: np.ndarray,
    settings: Optional[DisplaySettings] = None,
) -> np.ndarray:
    """Convert a numpy array to display-ready uint8.

    Pure function with no Qt dependency. Applies the full display
    pipeline: complex handling, band selection, window/level,
    contrast/brightness, gamma, and colormap.

    Parameters
    ----------
    arr : np.ndarray
        Source image. 2D (H, W), 3D (H, W, C), or complex.
    settings : Optional[DisplaySettings]
        Display parameters. None uses defaults.

    Returns
    -------
    np.ndarray
        uint8 array. Shape (H, W) for grayscale or (H, W, 3) for
        RGB (either from 3-band input or colormap application).
    """
    if settings is None:
        settings = DisplaySettings()

    # 1. Complex → magnitude
    if np.iscomplexobj(arr):
        arr = np.abs(arr)

    # 2. Band selection — channels-first convention: (C, H, W)
    if arr.ndim == 3:
        num_bands = arr.shape[0]
        if settings.band_index is not None:
            idx = min(settings.band_index, num_bands - 1)
            arr = arr[idx]  # (H, W)
        elif num_bands == 3:
            pass  # Keep as (3, H, W) → RGB
        elif num_bands >= 3:
            arr = arr[:3]  # First 3 bands as RGB
        else:
            arr = arr[0]  # Single band → (H, W)

    is_rgb = arr.ndim == 3

    # Remap path: SAR-specific remap functions (from grdl_sartoolbox)
    # replace the standard window/level/percentile pipeline but
    # contrast, brightness, and gamma are still applied on top.
    # Remap functions expect 2D (H, W) input.
    if settings.remap_function is not None and arr.ndim == 2:
        try:
            arr = settings.remap_function(arr)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            # Apply contrast, brightness, gamma on the remap output
            arr = arr.astype(np.float64) / 255.0
            if settings.contrast != 1.0 or settings.brightness != 0.0:
                arr = settings.contrast * (arr - 0.5) + 0.5 + settings.brightness
            if settings.gamma != 1.0:
                arr = np.clip(arr, 0.0, 1.0)
                arr = np.power(arr, 1.0 / settings.gamma)
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            # Apply colormap if grayscale
            if settings.colormap != 'grayscale':
                colormaps = _get_colormaps()
                lut = colormaps.get(settings.colormap)
                if lut is not None:
                    arr = lut[arr]
            return arr
        except Exception:
            pass  # Fall through to standard pipeline on error

    # 3. Window/level
    arr = arr.astype(np.float64)

    if settings.window_min is not None and settings.window_max is not None:
        vmin = float(settings.window_min)
        vmax = float(settings.window_max)
    else:
        if settings.percentile_low > 0 or settings.percentile_high < 100:
            vmin = float(np.nanpercentile(arr, settings.percentile_low))
            vmax = float(np.nanpercentile(arr, settings.percentile_high))
        else:
            vmin = float(np.nanmin(arr))
            vmax = float(np.nanmax(arr))

    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    else:
        arr = np.zeros_like(arr)

    # 4. Contrast and brightness
    if settings.contrast != 1.0 or settings.brightness != 0.0:
        arr = settings.contrast * (arr - 0.5) + 0.5 + settings.brightness

    # 5. Gamma
    if settings.gamma != 1.0:
        arr = np.clip(arr, 0.0, 1.0)
        arr = np.power(arr, 1.0 / settings.gamma)

    # 6. Scale to uint8
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    # 7. Colormap (only for grayscale images)
    if not is_rgb and settings.colormap != 'grayscale':
        colormaps = _get_colormaps()
        lut = colormaps.get(settings.colormap)
        if lut is not None:
            arr = lut[arr]  # (H, W) uint8 → (H, W, 3) uint8

    return arr


def array_to_qimage(
    arr: np.ndarray,
    settings: Optional[DisplaySettings] = None,
) -> Any:
    """Convert a numpy array to a QImage using display settings.

    Parameters
    ----------
    arr : np.ndarray
        Source image array.
    settings : Optional[DisplaySettings]
        Display parameters. None uses defaults.

    Returns
    -------
    QImage
        Ready-to-display Qt image.
    """
    if not _QT_AVAILABLE:
        raise ImportError("Qt is required for array_to_qimage")

    display = normalize_array(arr, settings)

    # Channels-first (C, H, W) → channels-last (H, W, C) for QImage.
    # Distinguish from colormap output (H, W, 3) by checking dim sizes.
    if (display.ndim == 3
            and display.shape[0] < display.shape[1]
            and display.shape[0] < display.shape[2]):
        display = np.transpose(display, (1, 2, 0))

    display = np.ascontiguousarray(display)

    if display.ndim == 2:
        h, w = display.shape
        return QImage(display.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    elif display.ndim == 3 and display.shape[2] == 3:
        h, w, _ = display.shape
        bpl = 3 * w
        return QImage(display.data, w, h, bpl, QImage.Format.Format_RGB888).copy()
    else:
        # Fallback: first channel
        band = display[:, :, 0] if display.ndim == 3 else display
        h, w = band.shape
        band = np.ascontiguousarray(band)
        return QImage(band.data, w, h, w, QImage.Format.Format_Grayscale8).copy()


# ---------------------------------------------------------------------------
# ImageCanvas — interactive QGraphicsView-based image viewer
# ---------------------------------------------------------------------------

if _QT_AVAILABLE:

    class ImageCanvas(QGraphicsView):
        """Interactive image viewer with pan, zoom, and display controls.

        Renders numpy arrays via QGraphicsView with configurable display
        settings. Supports mouse-drag panning, scroll-wheel zooming,
        and pixel inspection on hover.

        Parameters
        ----------
        parent : Optional[QWidget]
            Parent widget.

        Signals
        -------
        pixel_hovered(int, int, object)
            Emitted on mouse move: row, col, raw value(s) at that pixel.
        zoom_changed(float)
            Emitted when zoom level changes.
        display_settings_changed(object)
            Emitted when display settings are modified.
        """

        pixel_hovered = Signal(int, int, object)
        zoom_changed = Signal(float)
        display_settings_changed = Signal(object)

        _ZOOM_FACTOR = 1.15

        def __init__(self, parent: Optional[Any] = None) -> None:
            super().__init__(parent)

            self._source: Optional[np.ndarray] = None
            self._settings = DisplaySettings()
            self._zoom_level = 1.0
            self._zoom_history: list = []  # Stack of QTransform for undo

            # Scene setup
            self._scene = QGraphicsScene(self)
            self.setScene(self._scene)
            self._pixmap_item = QGraphicsPixmapItem()
            self._scene.addItem(self._pixmap_item)

            # Interaction
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            self.setMouseTracking(True)
            self.setTransformationAnchor(
                QGraphicsView.ViewportAnchor.AnchorUnderMouse
            )
            self.setResizeAnchor(
                QGraphicsView.ViewportAnchor.AnchorUnderMouse
            )

            # Visual
            self.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            self.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            self.setBackgroundBrush(Qt.GlobalColor.darkGray)

            # Zoom box (Ctrl + left-drag)
            self._zoom_box_active = False
            self._zoom_box_origin = QPoint()
            self._zoom_rubber_band = QRubberBand(
                QRubberBand.Shape.Rectangle, self,
            )

        def set_array(self, arr: np.ndarray) -> None:
            """Set the source image array and refresh display.

            Parameters
            ----------
            arr : np.ndarray
                Image data (2D, 3D, or complex).
            """
            self._source = arr
            self._refresh_display()

        def set_display_settings(self, settings: DisplaySettings) -> None:
            """Update display settings and re-render.

            Parameters
            ----------
            settings : DisplaySettings
                New display parameters.
            """
            self._settings = settings
            self._refresh_display()
            self.display_settings_changed.emit(settings)

        @property
        def display_settings(self) -> DisplaySettings:
            """Current display settings."""
            return self._settings

        @property
        def source_array(self) -> Optional[np.ndarray]:
            """The source image array, or None."""
            return self._source

        def fit_in_view(self) -> None:
            """Zoom to fit the entire image in the viewport."""
            if self._source is not None:
                self.fitInView(
                    self._pixmap_item,
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
                self._update_zoom_level()

        def zoom_to(self, factor: float) -> None:
            """Set absolute zoom factor.

            Parameters
            ----------
            factor : float
                Zoom level (1.0 = 100%).
            """
            if factor <= 0:
                return
            current = self.transform().m11()
            if current > 0:
                scale = factor / current
                self.scale(scale, scale)
            self._zoom_level = factor
            self.zoom_changed.emit(self._zoom_level)

        def reset_view(self) -> None:
            """Reset pan, zoom, and display settings to defaults."""
            self._settings = DisplaySettings()
            self.resetTransform()
            self._zoom_level = 1.0
            self._refresh_display()
            self.fit_in_view()
            self.display_settings_changed.emit(self._settings)

        # --- Zoom history ---

        def _push_zoom_state(self) -> None:
            """Save the current view transform for right-click undo."""
            self._zoom_history.append(self.transform())

        def zoom_undo(self) -> None:
            """Revert to the previous zoom/pan state.

            Does nothing if the history stack is empty.
            """
            if not self._zoom_history:
                return
            xform = self._zoom_history.pop()
            self.setTransform(xform)
            self._update_zoom_level()

        # --- Event overrides ---

        def mousePressEvent(self, event: Any) -> None:
            """Start zoom box on Ctrl+left-click, else default drag."""
            if (event.modifiers() & Qt.KeyboardModifier.ControlModifier
                    and event.button() == Qt.MouseButton.LeftButton):
                self._zoom_box_active = True
                self._zoom_box_origin = event.pos()
                self._zoom_rubber_band.setGeometry(
                    QRect(self._zoom_box_origin, QSize()),
                )
                self._zoom_rubber_band.show()
                event.accept()
                return
            if event.button() == Qt.MouseButton.RightButton:
                self.zoom_undo()
                event.accept()
                return
            super().mousePressEvent(event)

        def mouseReleaseEvent(self, event: Any) -> None:
            """Finish zoom box and fit the selected region in view."""
            if (self._zoom_box_active
                    and event.button() == Qt.MouseButton.LeftButton):
                self._zoom_box_active = False
                self._zoom_rubber_band.hide()
                rect = QRect(
                    self._zoom_box_origin, event.pos(),
                ).normalized()
                # Ignore tiny accidental drags
                if rect.width() > 5 and rect.height() > 5:
                    self._push_zoom_state()
                    scene_rect = self.mapToScene(rect).boundingRect()
                    self.fitInView(
                        scene_rect,
                        Qt.AspectRatioMode.KeepAspectRatio,
                    )
                    self._update_zoom_level()
                event.accept()
                return
            super().mouseReleaseEvent(event)

        def wheelEvent(self, event: Any) -> None:
            """Zoom in/out on scroll wheel."""
            delta = event.angleDelta().y()
            if delta > 0:
                factor = self._ZOOM_FACTOR
            elif delta < 0:
                factor = 1.0 / self._ZOOM_FACTOR
            else:
                return

            self._push_zoom_state()
            self.scale(factor, factor)
            self._update_zoom_level()

        def mouseDoubleClickEvent(self, event: Any) -> None:
            """Fit to view on double-click."""
            self._push_zoom_state()
            self.fit_in_view()

        def mouseMoveEvent(self, event: Any) -> None:
            """Emit pixel coordinates and value on hover, or update zoom box."""
            if self._zoom_box_active:
                self._zoom_rubber_band.setGeometry(
                    QRect(self._zoom_box_origin, event.pos()).normalized(),
                )
                event.accept()
                return
            super().mouseMoveEvent(event)
            if self._source is None:
                return

            scene_pos = self.mapToScene(event.pos())
            col = int(scene_pos.x())
            row = int(scene_pos.y())

            if self._source.ndim >= 2:
                if self._source.ndim == 2:
                    h, w = self._source.shape
                else:
                    # Channels-first: (C, H, W)
                    h, w = self._source.shape[1], self._source.shape[2]
                if 0 <= row < h and 0 <= col < w:
                    if self._source.ndim == 2:
                        value = self._source[row, col]
                    else:
                        value = self._source[:, row, col]
                    self.pixel_hovered.emit(row, col, value)

        # --- Internal ---

        def _refresh_display(self) -> None:
            """Re-render the source array with current settings."""
            if self._source is None:
                self._pixmap_item.setPixmap(QPixmap())
                return

            qimg = array_to_qimage(self._source, self._settings)
            self._pixmap_item.setPixmap(QPixmap.fromImage(qimg))
            self._scene.setSceneRect(self._pixmap_item.boundingRect())

        def _update_zoom_level(self) -> None:
            """Read current transform and emit zoom_changed."""
            self._zoom_level = self.transform().m11()
            self.zoom_changed.emit(self._zoom_level)

    class ImageCanvasThumbnail(ImageCanvas):
        """Non-interactive fixed-size thumbnail variant of ImageCanvas.

        Drop-in replacement for QLabel + QPixmap in gallery grids.
        Disables all user interaction (pan, zoom, hover).

        Parameters
        ----------
        size : int
            Fixed size in pixels (width and height).
        parent : Optional[QWidget]
            Parent widget.
        """

        def __init__(
            self,
            size: int = 128,
            parent: Optional[Any] = None,
        ) -> None:
            super().__init__(parent)

            self._thumb_size = size
            self.setFixedSize(size, size)

            # Disable interactivity
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            self.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            self.setMouseTracking(False)
            self.setInteractive(False)

            # Clean appearance
            self.setFrameShape(QGraphicsView.Shape.NoFrame)
            self.setBackgroundBrush(Qt.GlobalColor.transparent)

        def set_array(self, arr: np.ndarray) -> None:
            """Set image and auto-fit to thumbnail size."""
            super().set_array(arr)
            self.fit_in_view()

        def resizeEvent(self, event: Any) -> None:
            """Re-fit on resize."""
            super().resizeEvent(event)
            if self._source is not None:
                self.fit_in_view()

        def wheelEvent(self, event: Any) -> None:
            """Ignore scroll — no zoom on thumbnails."""
            event.ignore()

        def mouseDoubleClickEvent(self, event: Any) -> None:
            """Ignore double-click."""
            event.ignore()

        def mouseMoveEvent(self, event: Any) -> None:
            """Skip pixel hover for thumbnails."""
            pass

else:
    # Stubs when Qt is not available
    class ImageCanvas:  # type: ignore[no-redef]
        """Stub when Qt is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("Qt is required for ImageCanvas")

    class ImageCanvasThumbnail:  # type: ignore[no-redef]
        """Stub when Qt is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("Qt is required for ImageCanvasThumbnail")
