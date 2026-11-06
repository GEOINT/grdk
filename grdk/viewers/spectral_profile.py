# -*- coding: utf-8 -*-
"""
SpectralProfileWidget - Interactive spectral profile probe and curve viewer.

Supports querying pixel and ROI spectral signatures across multi-band
and hyperspectral cubes (reflectance/radiance vs wavelength).

Author: Steven Siebert
License: MIT License
"""

from dataclasses import dataclass
from typing import Any, List, Optional
import numpy as np

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QTableWidget, QTableWidgetItem, QFileDialog, QGroupBox
    )
    from PyQt6.QtGui import QPainter, QPen, QColor, QBrush
    from PyQt6.QtCore import Qt, QPointF
    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False


@dataclass
class SpectralProfile:
    row: int
    col: int
    values: np.ndarray
    wavelengths: List[float]


def extract_spectral_profile(
    source: Any,
    row: int,
    col: int,
    wavelengths: Optional[List[float]] = None
) -> SpectralProfile:
    """Extract spectral curve from an in-memory array or grdl ImageReader."""
    if isinstance(source, np.ndarray):
        if source.ndim == 3:
            values = source[:, row, col].astype(np.float64)
        elif source.ndim == 2:
            values = np.array([source[row, col]], dtype=np.float64)
        else:
            raise ValueError(f"Unsupported array shape {source.shape}")
    elif hasattr(source, "read_chip"):
        try:
            chip = source.read_chip(row, row + 1, col, col + 1)
        except Exception:
            chip = source.read_chip(row, col, 1, 1)
        if chip.ndim == 3:
            values = chip[:, 0, 0].astype(np.float64)
        elif chip.ndim == 2:
            values = np.array([chip[0, 0]], dtype=np.float64)
        else:
            values = chip.flatten().astype(np.float64)
    else:
        raise ValueError("Source must be a numpy ndarray or grdl ImageReader")

    num_bands = len(values)
    if wavelengths is None or len(wavelengths) != num_bands:
        wavelengths = [float(i + 1) for i in range(num_bands)]

    return SpectralProfile(row=row, col=col, values=values, wavelengths=wavelengths)


if _QT_AVAILABLE:

    class SpectralPlotCanvas(QWidget):
        """Simple lightweight 2D spectral curve renderer using QPainter."""

        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            self.profile: Optional[SpectralProfile] = None
            self.setMinimumHeight(180)
            self.setMinimumWidth(320)

        def set_profile(self, profile: Optional[SpectralProfile]) -> None:
            self.profile = profile
            self.update()

        def paintEvent(self, event) -> None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            rect = self.rect()

            # Background
            painter.fillRect(rect, QColor(30, 30, 30))

            if self.profile is None or len(self.profile.values) == 0:
                painter.setPen(QColor(180, 180, 180))
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Click image canvas to probe spectral curve")
                return

            margin_left = 45
            margin_right = 20
            margin_top = 25
            margin_bottom = 35

            plot_w = rect.width() - margin_left - margin_right
            plot_h = rect.height() - margin_top - margin_bottom

            if plot_w <= 10 or plot_h <= 10:
                return

            vals = self.profile.values
            wls = np.array(self.profile.wavelengths)

            ymin = float(np.min(vals))
            ymax = float(np.max(vals))
            if ymax <= ymin:
                ymax = ymin + 1.0

            xmin = float(wls[0])
            xmax = float(wls[-1])
            if xmax <= xmin:
                xmax = xmin + 1.0

            # Grid lines
            painter.setPen(QPen(QColor(60, 60, 60), 1, Qt.PenStyle.DashLine))
            for i in range(5):
                y = margin_top + plot_h * (i / 4.0)
                painter.drawLine(int(margin_left), int(y), int(margin_left + plot_w), int(y))
                val_label = f"{ymax - i * (ymax - ymin)/4.0:.2f}"
                painter.setPen(QColor(140, 140, 140))
                painter.drawText(5, int(y + 4), val_label)
                painter.setPen(QPen(QColor(60, 60, 60), 1, Qt.PenStyle.DashLine))

            # Axes
            painter.setPen(QPen(QColor(200, 200, 200), 1))
            painter.drawLine(margin_left, margin_top, margin_left, margin_top + plot_h)
            painter.drawLine(margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h)

            # X labels
            painter.drawText(margin_left, margin_top + plot_h + 18, f"{xmin:.1f}")
            painter.drawText(margin_left + plot_w - 40, margin_top + plot_h + 18, f"{xmax:.1f} nm")

            # Plot curve
            pts = []
            for i in range(len(vals)):
                px = margin_left + ((wls[i] - xmin) / (xmax - xmin)) * plot_w
                py = margin_top + plot_h - ((vals[i] - ymin) / (ymax - ymin)) * plot_h
                pts.append(QPointF(px, py))

            painter.setPen(QPen(QColor(0, 220, 255), 2))
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i+1])

            # Draw points
            painter.setBrush(QBrush(QColor(255, 180, 0)))
            for pt in pts:
                painter.drawEllipse(pt, 2, 2)

            # Title
            painter.setPen(QColor(240, 240, 240))
            painter.drawText(margin_left, margin_top - 8, f"Pixel ({self.profile.row}, {self.profile.col}) — {len(vals)} bands")


    class SpectralProfileWidget(QWidget):
        """Interactive dockable widget to inspect spectral profiles."""

        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            self._source: Any = None
            self._wavelengths: Optional[List[float]] = None
            self.current_profile: Optional[SpectralProfile] = None

            layout = QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)

            self.plot_canvas = SpectralPlotCanvas(self)
            layout.addWidget(self.plot_canvas)

            info_row = QHBoxLayout()
            self.summary_label = QLabel("No pixel selected", self)
            info_row.addWidget(self.summary_label)

            self.export_btn = QPushButton("Export CSV", self)
            self.export_btn.clicked.connect(self._export_csv)
            info_row.addWidget(self.export_btn)
            layout.addLayout(info_row)

        def set_data(self, source: Any, wavelengths: Optional[List[float]] = None) -> None:
            self._source = source
            self._wavelengths = wavelengths

        def probe_pixel(self, row: int, col: int) -> None:
            if self._source is None:
                return
            profile = extract_spectral_profile(self._source, row, col, self._wavelengths)
            self.current_profile = profile
            self.plot_canvas.set_profile(profile)

            peak_idx = int(np.argmax(profile.values))
            peak_wl = profile.wavelengths[peak_idx]
            peak_val = profile.values[peak_idx]
            self.summary_label.setText(
                f"Peak: {peak_wl:.1f} nm ({peak_val:.3g}) | Min: {np.min(profile.values):.3g} | Mean: {np.mean(profile.values):.3g}"
            )

        def get_peak_wavelength(self) -> Optional[float]:
            if self.current_profile is None or len(self.current_profile.values) == 0:
                return None
            return float(self.current_profile.wavelengths[int(np.argmax(self.current_profile.values))])

        def _export_csv(self) -> None:
            if self.current_profile is None:
                return
            path, _ = QFileDialog.getSaveFileName(self, "Export Spectral Profile", "spectral_profile.csv", "CSV Files (*.csv)")
            if not path:
                return
            with open(path, "w", encoding="utf-8") as f:
                f.write("Band,Wavelength_nm,Value\n")
                for i, (wl, v) in enumerate(zip(self.current_profile.wavelengths, self.current_profile.values)):
                    f.write(f"{i+1},{wl:.3f},{v:.6f}\n")
