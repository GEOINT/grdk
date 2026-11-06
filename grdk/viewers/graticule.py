# -*- coding: utf-8 -*-
"""
Graticule - Coordinate grid overlay with latitude/longitude lines.

Author: Steven Siebert
License: MIT License
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
import numpy as np

try:
    from PyQt6.QtWidgets import QWidget
    from PyQt6.QtGui import QPainter, QPen, QColor, QFont
    from PyQt6.QtCore import Qt, QPointF
    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False


@dataclass
class GraticuleGrid:
    lat_lines: List[float]
    lon_lines: List[float]


def compute_graticule_lines(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    step_deg: float = 1.0,
) -> GraticuleGrid:
    """Compute grid coordinates at step_deg intervals."""
    # Round to step
    start_lat = np.floor(min_lat / step_deg) * step_deg
    end_lat = np.ceil(max_lat / step_deg) * step_deg
    lats = list(np.arange(start_lat, end_lat + step_deg * 0.5, step_deg))

    start_lon = np.floor(min_lon / step_deg) * step_deg
    end_lon = np.ceil(max_lon / step_deg) * step_deg
    lons = list(np.arange(start_lon, end_lon + step_deg * 0.5, step_deg))

    return GraticuleGrid(lat_lines=lats, lon_lines=lons)


if _QT_AVAILABLE:

    class GraticuleOverlay(QWidget):
        """Graticule grid overlay on top of QGraphicsView canvas."""

        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            self._geolocation: Optional[Any] = None
            self._visible: bool = False
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        def set_geolocation(self, geo: Optional[Any]) -> None:
            self._geolocation = geo
            self.update()

        def set_graticule_visible(self, visible: bool) -> None:
            self._visible = visible
            self.setVisible(visible)
            self.update()
