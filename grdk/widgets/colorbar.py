# -*- coding: utf-8 -*-
"""
ColorBarWidget - Horizontal colorbar showing the active colormap gradient.

Displays the current colormap as a horizontal gradient bar with min/max
value labels.  Designed to sit below the image canvas in GeoImageViewer.
One colorbar per pane; initially hidden, toggled via a checkbox in the
display controls panel.  Greyed out / hidden for RGB display.

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

from typing import Optional

try:
    from PyQt6.QtWidgets import QWidget
    from PyQt6.QtGui import QColor, QImage, QPainter, QPixmap
    from PyQt6.QtCore import Qt

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False

if _QT_AVAILABLE:
    import numpy as np
    from grdk.viewers.image_canvas import _get_colormaps

    class ColorBarWidget(QWidget):
        """Horizontal colorbar showing the active colormap gradient.

        Renders a 256-step gradient using the selected colormap LUT,
        with min/max value labels at either end.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget.
        """

        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)

            self._colormap_name: str = 'grayscale'
            self._vmin: float = 0.0
            self._vmax: float = 255.0
            self._gradient_pixmap: Optional[QPixmap] = None

            self.setFixedHeight(32)
            self.hide()

        def set_colormap(self, name: str) -> None:
            """Set the colormap to display.

            Parameters
            ----------
            name : str
                Colormap name (e.g., 'grayscale', 'viridis').
            """
            if name == self._colormap_name:
                return
            self._colormap_name = name
            self._gradient_pixmap = None  # invalidate cache
            self.update()

        def set_range(self, vmin: float, vmax: float) -> None:
            """Set the data value range for labels.

            Parameters
            ----------
            vmin : float
                Minimum value.
            vmax : float
                Maximum value.
            """
            self._vmin = vmin
            self._vmax = vmax
            self.update()

        def update_from_settings(self, settings: object) -> None:
            """Update colorbar from a DisplaySettings instance.

            Parameters
            ----------
            settings : DisplaySettings
                Current display settings.
            """
            cmap = getattr(settings, 'colormap', 'grayscale')
            if cmap != self._colormap_name:
                self._colormap_name = cmap
                self._gradient_pixmap = None

            wmin = getattr(settings, 'window_min', None)
            wmax = getattr(settings, 'window_max', None)
            if wmin is not None and wmax is not None:
                self._vmin = wmin
                self._vmax = wmax

            self.update()

        def _build_gradient(self, width: int) -> QPixmap:
            """Build a QPixmap of the colormap gradient at the given width."""
            colormaps = _get_colormaps()
            lut = colormaps.get(self._colormap_name)

            # Create a 256x1 image then scale
            img = QImage(256, 1, QImage.Format.Format_RGB888)
            for i in range(256):
                if lut is not None:
                    r, g, b = int(lut[i, 0]), int(lut[i, 1]), int(lut[i, 2])
                else:
                    r = g = b = i
                img.setPixelColor(i, 0, QColor(r, g, b))

            pixmap = QPixmap.fromImage(
                img.scaled(width, 1, Qt.AspectRatioMode.IgnoreAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
            )
            return pixmap

        def paintEvent(self, event: object) -> None:
            """Paint the colorbar gradient and labels."""
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            w = self.width()
            h = self.height()

            margin_left = 50
            margin_right = 50
            bar_left = margin_left
            bar_width = w - margin_left - margin_right
            bar_top = 2
            bar_height = h - 16

            if bar_width <= 0 or bar_height <= 0:
                painter.end()
                return

            # Build/cache gradient pixmap
            if (self._gradient_pixmap is None
                    or self._gradient_pixmap.width() != bar_width):
                self._gradient_pixmap = self._build_gradient(bar_width)

            # Draw gradient bar
            painter.drawPixmap(bar_left, bar_top, bar_width, bar_height,
                               self._gradient_pixmap)

            # Draw border
            painter.setPen(QColor(128, 128, 128))
            painter.drawRect(bar_left, bar_top, bar_width - 1, bar_height)

            # Draw min/max labels
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)
            painter.setPen(self.palette().color(self.foregroundRole()))

            label_y = bar_top + bar_height + 11

            min_text = f"{self._vmin:.1f}"
            painter.drawText(2, label_y, min_text)

            max_text = f"{self._vmax:.1f}"
            fm = painter.fontMetrics()
            painter.drawText(
                w - fm.horizontalAdvance(max_text) - 2, label_y, max_text,
            )

            painter.end()

        def resizeEvent(self, event: object) -> None:
            """Invalidate gradient cache on resize."""
            self._gradient_pixmap = None
            super().resizeEvent(event)

else:

    class ColorBarWidget:  # type: ignore[no-redef]
        """Stub when Qt is unavailable."""

        def __init__(self, *args, **kwargs):
            raise ImportError("Qt is required for ColorBarWidget")
