# -*- coding: utf-8 -*-
"""
MagnifierLoupe - Roaming inset magnifier tool for fine inspection.

Provides real-time 2x, 4x, 8x magnification overlay over the cursor position.

Author: Steven Siebert
License: MIT License
"""

from typing import Any, Optional
import numpy as np

try:
    from PyQt6.QtWidgets import QWidget
    from PyQt6.QtGui import QPainter, QPixmap, QPen, QColor
    from PyQt6.QtCore import Qt, QPoint
    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False


class MagnifierLoupe:
    """Core magnifier calculation helper."""

    def __init__(self, zoom_factor: float = 2.0, radius: int = 40) -> None:
        self.zoom_factor = zoom_factor
        self.radius = radius

    def extract_zoomed_chip(
        self,
        arr: np.ndarray,
        center_row: int,
        center_col: int
    ) -> np.ndarray:
        """Extract a bounding box chip centered at (center_row, center_col)."""
        r0 = max(0, center_row - self.radius)
        r1 = min(arr.shape[-2], center_row + self.radius)
        c0 = max(0, center_col - self.radius)
        c1 = min(arr.shape[-1], center_col + self.radius)

        if arr.ndim == 3:
            chip = arr[:, r0:r1, c0:c1]
        else:
            chip = arr[r0:r1, c0:c1]

        return chip


if _QT_AVAILABLE:

    class LoupeOverlayWidget(QWidget):
        """Floating circular loupe window attached to canvas cursor."""

        def __init__(self, parent: Optional[QWidget] = None, size: int = 180) -> None:
            super().__init__(parent)
            self.size = size
            self.zoom_factor: float = 2.0
            self._pixmap: Optional[QPixmap] = None
            self.setFixedSize(size, size)
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            self.hide()

        def set_chip_pixmap(self, pixmap: QPixmap) -> None:
            self._pixmap = pixmap
            self.update()

        def set_zoom(self, zoom: float) -> None:
            self.zoom_factor = zoom
            self.update()

        def paintEvent(self, event) -> None:
            if self._pixmap is None:
                return
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

            # Circular clip
            radius = self.size / 2.0
            painter.setPen(QPen(QColor(255, 215, 0), 3))
            painter.drawEllipse(1, 1, self.size - 2, self.size - 2)

            scaled = self._pixmap.scaled(
                int(self.size * self.zoom_factor),
                int(self.size * self.zoom_factor),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
            # Center cropped
            sx = (scaled.width() - self.size) // 2
            sy = (scaled.height() - self.size) // 2
            painter.drawPixmap(0, 0, scaled, sx, sy, self.size, self.size)

            # Crosshairs
            painter.setPen(QPen(QColor(255, 0, 0, 180), 1))
            painter.drawLine(int(radius), int(radius - 10), int(radius), int(radius + 10))
            painter.drawLine(int(radius - 10), int(radius), int(radius + 10), int(radius))
