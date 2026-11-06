# -*- coding: utf-8 -*-
"""
HistogramStretchWidget - Interactive Dynamic Range Adjustment (DRA) & Histogram tool.

Supports real-time histogram calculation, cumulative percentile stretch presets,
and interactive window/level adjustments.

Author: Steven Siebert
License: MIT License
"""

from typing import Any, Optional, Tuple
import numpy as np

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QSlider, QSpinBox, QDoubleSpinBox, QGroupBox
    )
    from PyQt6.QtGui import QPainter, QPen, QColor, QBrush
    from PyQt6.QtCore import Qt, pyqtSignal as Signal
    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False


def compute_histogram_bins(
    arr: np.ndarray,
    num_bins: int = 256,
    sample_limit: int = 500_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute histogram counts and bin edges with downsampling for performance."""
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return np.zeros(num_bins, dtype=np.int64), np.linspace(0, 1, num_bins + 1)

    if len(valid) > sample_limit:
        step = len(valid) // sample_limit
        valid = valid[::step]

    vmin = float(np.min(valid))
    vmax = float(np.max(valid))
    if vmin == vmax:
        vmax = vmin + 1.0

    counts, bin_edges = np.histogram(valid, bins=num_bins, range=(vmin, vmax))
    return counts, bin_edges


if _QT_AVAILABLE:

    class HistogramPlot(QWidget):
        """Custom QPainter-based histogram chart with stretch markers."""

        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            self.counts: Optional[np.ndarray] = None
            self.bin_edges: Optional[np.ndarray] = None
            self.low_pct: float = 0.0
            self.high_pct: float = 100.0
            self.setMinimumHeight(140)
            self.setMinimumWidth(260)

        def set_histogram(self, counts: np.ndarray, bin_edges: np.ndarray) -> None:
            self.counts = counts
            self.bin_edges = bin_edges
            self.update()

        def set_percentiles(self, low: float, high: float) -> None:
            self.low_pct = low
            self.high_pct = high
            self.update()

        def paintEvent(self, event) -> None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            rect = self.rect()

            painter.fillRect(rect, QColor(25, 25, 25))

            if self.counts is None or len(self.counts) == 0:
                painter.setPen(QColor(160, 160, 160))
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "No image data loaded")
                return

            margin = 10
            w = rect.width() - 2 * margin
            h = rect.height() - 2 * margin

            # Log scale for high dynamic range
            log_counts = np.log1p(self.counts.astype(np.float64))
            max_log = float(np.max(log_counts))
            if max_log <= 0:
                max_log = 1.0

            num_bins = len(self.counts)
            bin_w = w / float(num_bins)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(70, 130, 180, 200)))

            for i in range(num_bins):
                bh = (log_counts[i] / max_log) * h
                bx = margin + i * bin_w
                by = margin + h - bh
                painter.drawRect(int(bx), int(by), int(max(1.0, bin_w)), int(bh))

            # Draw low & high cut markers
            low_x = margin + (self.low_pct / 100.0) * w
            high_x = margin + (self.high_pct / 100.0) * w

            painter.setPen(QPen(QColor(0, 255, 128), 2))
            painter.drawLine(int(low_x), margin, int(low_x), margin + h)

            painter.setPen(QPen(QColor(255, 64, 64), 2))
            painter.drawLine(int(high_x), margin, int(high_x), margin + h)


    class HistogramStretchWidget(QWidget):
        """Interactive widget providing DRA presets and black/white sliders."""

        stretch_changed = Signal(float, float)  # (low_percentile, high_percentile)

        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            self._arr: Optional[np.ndarray] = None

            layout = QVBoxLayout(self)
            layout.setContentsMargins(6, 6, 6, 6)

            self.plot = HistogramPlot(self)
            layout.addWidget(self.plot)

            # Quick presets
            preset_row = QHBoxLayout()
            for label, (lp, hp) in [
                ("Min-Max", (0.0, 100.0)),
                ("1% - 99%", (1.0, 99.0)),
                ("2% - 98%", (2.0, 98.0)),
                ("5% - 95%", (5.0, 95.0)),
            ]:
                btn = QPushButton(label, self)
                btn.clicked.connect(lambda _, l=lp, h=hp: self.apply_preset(l, h))
                preset_row.addWidget(btn)
            layout.addLayout(preset_row)

            # Sliders
            slider_row = QHBoxLayout()
            slider_row.addWidget(QLabel("Black Point:", self))
            self.low_slider = QSlider(Qt.Orientation.Horizontal, self)
            self.low_slider.setRange(0, 1000)
            self.low_slider.setValue(0)
            self.low_slider.valueChanged.connect(self._on_slider_changed)
            slider_row.addWidget(self.low_slider)

            slider_row.addWidget(QLabel("White Point:", self))
            self.high_slider = QSlider(Qt.Orientation.Horizontal, self)
            self.high_slider.setRange(0, 1000)
            self.high_slider.setValue(1000)
            self.high_slider.valueChanged.connect(self._on_slider_changed)
            slider_row.addWidget(self.high_slider)
            layout.addLayout(slider_row)

        def set_array(self, arr: np.ndarray) -> None:
            self._arr = arr
            counts, bin_edges = compute_histogram_bins(arr)
            self.plot.set_histogram(counts, bin_edges)

        def apply_preset(self, low: float, high: float) -> None:
            self.low_slider.blockSignals(True)
            self.high_slider.blockSignals(True)
            self.low_slider.setValue(int(low * 10))
            self.high_slider.setValue(int(high * 10))
            self.low_slider.blockSignals(False)
            self.high_slider.blockSignals(False)
            self.plot.set_percentiles(low, high)
            self.stretch_changed.emit(low, high)

        def get_percentile_preset(self, low: float, high: float) -> Tuple[float, float]:
            if self._arr is None:
                return (0.0, 100.0)
            vmin = float(np.nanpercentile(self._arr, low))
            vmax = float(np.nanpercentile(self._arr, high))
            return (vmin, vmax)

        def _on_slider_changed(self) -> None:
            low = self.low_slider.value() / 10.0
            high = self.high_slider.value() / 10.0
            if low >= high:
                low = max(0.0, high - 0.1)
                self.low_slider.blockSignals(True)
                self.low_slider.setValue(int(low * 10))
                self.low_slider.blockSignals(False)

            self.plot.set_percentiles(low, high)
            self.stretch_changed.emit(low, high)

