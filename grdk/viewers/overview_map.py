# -*- coding: utf-8 -*-
"""
OverviewMapWidget - Thumbnail overview locator pane for navigation.

Displays full scene extent with an interactive dynamic bounding box showing
the current viewport zoom and roam state.

Author: Steven Siebert
License: MIT License
"""

from typing import Any, Optional, Tuple
import numpy as np

try:
    from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
    from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QPixmap, QImage
    from PyQt6.QtCore import Qt, QRectF, pyqtSignal as Signal
    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False


if _QT_AVAILABLE:

    class OverviewMapWidget(QWidget):
        """Mini-map overview widget showing scene context and viewport rectangle."""

        viewport_clicked = Signal(int, int)  # center_row, center_col

        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            self._full_image: Optional[np.ndarray] = None
            self._pixmap: Optional[QPixmap] = None
            self.viewport_rect: Optional[Tuple[int, int, int, int]] = None  # (row, col, h, w)
            self.setMinimumSize(120, 120)
            self.setMaximumSize(300, 300)

        def set_full_image(self, arr: np.ndarray) -> None:
            self._full_image = arr
            if arr.ndim == 3:
                # Use first band or RGB
                display = arr[0]
            else:
                display = arr

            h, w = display.shape
            # Downsample if large
            max_dim = 250
            if max(h, w) > max_dim:
                scale = max_dim / float(max(h, w))
                sh, sw = int(h * scale), int(w * scale)
                # Subsample
                step_r = max(1, h // sh)
                step_c = max(1, w // sw)
                display = display[::step_r, ::step_c]

            vmin = float(np.min(display))
            vmax = float(np.max(display))
            if vmax > vmin:
                norm = ((display - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
            else:
                norm = np.zeros_like(display, dtype=np.uint8)

            norm = np.ascontiguousarray(norm)
            qimg = QImage(norm.data, norm.shape[1], norm.shape[0], norm.shape[1], QImage.Format.Format_Grayscale8)
            self._pixmap = QPixmap.fromImage(qimg)
            self.update()

        def update_viewport_rect(self, row: int, col: int, height: int, width: int) -> None:
            self.viewport_rect = (row, col, height, width)
            self.update()

        def paintEvent(self, event) -> None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            rect = self.rect()

            painter.fillRect(rect, QColor(20, 20, 20))

            if self._pixmap is None or self._full_image is None:
                painter.setPen(QColor(120, 120, 120))
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Overview Map")
                return

            scaled = self._pixmap.scaled(
                rect.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            ox = (rect.width() - scaled.width()) // 2
            oy = (rect.height() - scaled.height()) // 2
            painter.drawPixmap(ox, oy, scaled)

            # Draw viewport bounding box
            if self.viewport_rect is not None and self._full_image is not None:
                full_h = self._full_image.shape[-2]
                full_w = self._full_image.shape[-1]
                r, c, h, w = self.viewport_rect

                bx = ox + (c / float(full_w)) * scaled.width()
                by = oy + (r / float(full_h)) * scaled.height()
                bw = max(4.0, (w / float(full_w)) * scaled.width())
                bh = max(4.0, (h / float(full_h)) * scaled.height())

                painter.setPen(QPen(QColor(255, 60, 60), 2))
                painter.setBrush(QBrush(QColor(255, 60, 60, 40)))
                painter.drawRect(QRectF(bx, by, bw, bh))

        def mousePressEvent(self, event) -> None:
            if self._full_image is None or self._pixmap is None:
                return
            rect = self.rect()
            scaled = self._pixmap.scaled(
                rect.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )
            ox = (rect.width() - scaled.width()) // 2
            oy = (rect.height() - scaled.height()) // 2

            mx = event.pos().x() - ox
            my = event.pos().y() - oy
            if 0 <= mx <= scaled.width() and 0 <= my <= scaled.height():
                full_h = self._full_image.shape[-2]
                full_w = self._full_image.shape[-1]
                target_col = int((mx / float(scaled.width())) * full_w)
                target_row = int((my / float(scaled.height())) * full_h)
                self.viewport_clicked.emit(target_row, target_col)

