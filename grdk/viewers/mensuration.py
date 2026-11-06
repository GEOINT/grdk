# -*- coding: utf-8 -*-
"""
Mensuration - Geodetic measurement tool for ELT exploitation.

Provides Vincenty/Haversine distance, azimuth/bearing, and geodesic polygon
area & perimeter calculations across coordinate systems.

Author: Steven Siebert
License: MIT License
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
import numpy as np

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QComboBox, QTableWidget, QTableWidgetItem
    )
    from PyQt6.QtCore import Qt, pyqtSignal as Signal
    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False


# WGS-84 ellipsoid constants
WGS84_A = 6378137.0  # semi-major axis (meters)
WGS84_F = 1.0 / 298.257223563  # flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)  # semi-minor axis


@dataclass
class MensurationResult:
    distance_m: float
    azimuth_deg: float
    back_azimuth_deg: float
    perimeter_m: Optional[float] = None
    area_m2: Optional[float] = None


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two lat/lon coordinates in meters."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(WGS84_A * c)


def geodesic_azimuth(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute forward azimuth (bearing) from pt1 to pt2 in degrees [0, 360)."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_lambda = np.radians(lon2 - lon1)

    y = np.sin(delta_lambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)
    bearing = np.degrees(np.arctan2(y, x))
    return float((bearing + 360.0) % 360.0)


def geodesic_polygon_area(coords: List[Tuple[float, float]]) -> float:
    """Compute spherical polygon area in square meters for lat/lon coordinates."""
    if len(coords) < 3:
        return 0.0

    # Ensure closed polygon
    pts = list(coords)
    if pts[0] != pts[-1]:
        pts.append(pts[0])

    # Spherical excess formulation
    total = 0.0
    for i in range(len(pts) - 1):
        lat1, lon1 = pts[i]
        lat2, lon2 = pts[i + 1]
        p1 = np.radians(lat1)
        p2 = np.radians(lat2)
        d_lambda = np.radians(lon2 - lon1)
        total += d_lambda * (2.0 + np.sin(p1) + np.sin(p2))

    area = abs(total * (WGS84_A ** 2) / 2.0)
    return float(area)


if _QT_AVAILABLE:

    class MensurationToolWidget(QWidget):
        """Interactive widget reporting ground distance, azimuth, and area."""

        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            self._points: List[Tuple[float, float]] = []  # lat, lon pairs

            layout = QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)

            self.dist_label = QLabel("Distance: —", self)
            self.az_label = QLabel("Bearing: —", self)
            self.area_label = QLabel("Area: —", self)

            layout.addWidget(self.dist_label)
            layout.addWidget(self.az_label)
            layout.addWidget(self.area_label)

            btn_row = QHBoxLayout()
            self.clear_btn = QPushButton("Clear Measurements", self)
            self.clear_btn.clicked.connect(self.clear)
            btn_row.addWidget(self.clear_btn)
            layout.addLayout(btn_row)

        def add_point(self, lat: float, lon: float) -> None:
            self._points.append((lat, lon))
            self._update_readouts()

        def clear(self) -> None:
            self._points.clear()
            self.dist_label.setText("Distance: —")
            self.az_label.setText("Bearing: —")
            self.area_label.setText("Area: —")

        def _update_readouts(self) -> None:
            if len(self._points) < 2:
                return

            p1 = self._points[-2]
            p2 = self._points[-1]
            dist_m = haversine_distance(p1[0], p1[1], p2[0], p2[1])
            az_deg = geodesic_azimuth(p1[0], p1[1], p2[0], p2[1])

            if dist_m > 1000.0:
                self.dist_label.setText(f"Distance: {dist_m/1000.0:.3f} km ({dist_m:.1f} m)")
            else:
                self.dist_label.setText(f"Distance: {dist_m:.2f} m")

            self.az_label.setText(f"Bearing: {az_deg:.1f}°")

            if len(self._points) >= 3:
                area_m2 = geodesic_polygon_area(self._points)
                if area_m2 > 1_000_000.0:
                    self.area_label.setText(f"Area: {area_m2/1_000_000.0:.3f} km²")
                else:
                    self.area_label.setText(f"Area: {area_m2:.1f} m²")

