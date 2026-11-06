# -*- coding: utf-8 -*-
"""
VectorMarkup - Interactive tactical vector markup and annotation layer.

Supports drawing lines, polylines, polygons, arrows, and text callouts
with customizable stroke color, fill, opacity, and line width.

Author: Steven Siebert
License: MIT License
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple
import numpy as np

try:
    from PyQt6.QtWidgets import QWidget
    from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QFont
    from PyQt6.QtCore import Qt, QPointF
    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False


@dataclass
class MarkupItem:
    shape_type: str  # "point", "line", "polygon", "arrow", "callout"
    coordinates: List[Tuple[float, float]]  # [(row, col), ...]
    label: str = ""
    color: str = "#FF0000"
    line_width: int = 2
    fill_opacity: float = 0.2


class VectorMarkupLayer:
    """Stores and manages vector markup annotations."""

    def __init__(self) -> None:
        self.items: List[MarkupItem] = []

    def add_item(self, item: MarkupItem) -> None:
        self.items.append(item)

    def remove_item(self, index: int) -> None:
        if 0 <= index < len(self.items):
            self.items.pop(index)

    def clear(self) -> None:
        self.items.clear()


if _QT_AVAILABLE:

    class VectorMarkupOverlay(QWidget):
        """Overlay widget on top of image canvas rendering vector markup."""

        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            self.layer = VectorMarkupLayer()
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        def add_item(self, item: MarkupItem) -> None:
            self.layer.add_item(item)
            self.update()

        def clear(self) -> None:
            self.layer.clear()
            self.update()

        def paintEvent(self, event) -> None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            for item in self.layer.items:
                col = QColor(item.color)
                painter.setPen(QPen(col, item.line_width))

                pts = [QPointF(c[1], c[0]) for c in item.coordinates]
                if not pts:
                    continue

                if item.shape_type == "point":
                    painter.setBrush(QBrush(col))
                    painter.drawEllipse(pts[0], 4, 4)
                elif item.shape_type == "line":
                    for i in range(len(pts) - 1):
                        painter.drawLine(pts[i], pts[i+1])
                elif item.shape_type == "polygon":
                    col_fill = QColor(col)
                    col_fill.setAlphaF(item.fill_opacity)
                    painter.setBrush(QBrush(col_fill))
                    painter.drawPolygon(pts)
                elif item.shape_type == "callout":
                    # Draw line to anchor
                    if len(pts) >= 2:
                        painter.drawLine(pts[0], pts[1])
                    painter.setBrush(QBrush(QColor(0, 0, 0, 180)))
                    painter.drawRect(int(pts[-1].x()), int(pts[-1].y() - 20), len(item.label) * 8 + 10, 20)
                    painter.setPen(QColor(255, 255, 255))
                    painter.drawText(int(pts[-1].x() + 5), int(pts[-1].y() - 5), item.label)
