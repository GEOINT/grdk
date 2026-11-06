# -*- coding: utf-8 -*-
"""
Filters - Spatial filtering and SAR despeckle processing engine.

Reuses GRDL image processor components (SobelFilter, LaplacianFilter,
UnsharpMaskFilter, HighPassFilter, GaussianFilter, MeanFilter, MedianFilter,
LeeFilter, FrostFilter, KuanFilter, BoxcarFilter) for in-viewer exploitation
and automated execution.

Author: Steven Siebert
License: MIT License
"""

from typing import Any, Optional
import numpy as np

# GRDL Processors
from grdl.image_processing.filters.edge import (
    HighPassFilter,
    LaplacianFilter,
    SobelFilter,
    UnsharpMaskFilter,
)
from grdl.image_processing.filters.linear import GaussianFilter, MeanFilter
from grdl.image_processing.filters.rank import MedianFilter
from grdl.image_processing.filters.sar_speckle import (
    BoxcarFilter,
    FrostFilter,
    KuanFilter,
)
from grdl.image_processing.filters.standard_lee import LeeFilter

try:
    from PyQt6.QtWidgets import (
        QComboBox,
        QDialog,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
    )
    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Spatial & Edge Filtering (Optical / PAN / Multi-band)
# ---------------------------------------------------------------------------

def gaussian_blur(arr: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian smoothing filter via GRDL GaussianFilter."""
    return GaussianFilter(sigma=float(sigma)).apply(arr)


def sharpen_unsharp_mask(arr: np.ndarray, amount: float = 1.5, radius: float = 1.0) -> np.ndarray:
    """Sharpen image using unsharp masking via GRDL UnsharpMaskFilter."""
    return UnsharpMaskFilter(radius=float(radius), amount=float(amount)).apply(arr)


def sobel_gradient(arr: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude using GRDL SobelFilter."""
    return SobelFilter(output_type="magnitude").apply(arr)


def laplacian_edge(arr: np.ndarray) -> np.ndarray:
    """Compute second-derivative edge detector via GRDL LaplacianFilter."""
    return LaplacianFilter().apply(arr)


def high_pass_filter(arr: np.ndarray, cutoff: float = 2.0) -> np.ndarray:
    """High-pass filter retaining fine spatial details via GRDL HighPassFilter."""
    return HighPassFilter(cutoff=float(cutoff)).apply(arr)


# ---------------------------------------------------------------------------
# SAR Despeckle Filtering
# ---------------------------------------------------------------------------

def boxcar_filter(arr: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Simple moving average filter via GRDL BoxcarFilter."""
    return BoxcarFilter(window_size=int(window_size)).apply(arr)


def median_filter(arr: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Non-linear median rank filter via GRDL MedianFilter."""
    return MedianFilter(kernel_size=int(window_size)).apply(arr)


def lee_filter(arr: np.ndarray, window_size: int = 5, num_looks: int = 1) -> np.ndarray:
    """Lee adaptive speckle filter via GRDL LeeFilter."""
    enl = float(max(1, num_looks))
    return LeeFilter(kernel_size=int(window_size), enl=enl).apply(arr).astype(arr.dtype)


def kuan_filter(arr: np.ndarray, window_size: int = 5, num_looks: int = 1) -> np.ndarray:
    """Kuan adaptive speckle filter via GRDL KuanFilter."""
    return KuanFilter(window_size=int(window_size), num_looks=int(num_looks)).apply(arr)


def frost_filter(arr: np.ndarray, window_size: int = 5, damping_factor: float = 2.0) -> np.ndarray:
    """Frost exponentially damped adaptive speckle filter via GRDL FrostFilter."""
    return FrostFilter(window_size=int(window_size), damping_factor=float(damping_factor)).apply(arr)


if _QT_AVAILABLE:

    class ImageFilterDialog(QDialog):
        """Dialog to select and configure image spatial filters."""

        def __init__(self, parent: Optional[Any] = None) -> None:
            super().__init__(parent)
            self.setWindowTitle("Spatial & SAR Despeckle Filters")
            self.resize(320, 200)

            layout = QVBoxLayout(self)

            row_f = QHBoxLayout()
            row_f.addWidget(QLabel("Filter Type:", self))
            self.filter_combo = QComboBox(self)
            self.filter_combo.addItems([
                "Sharpen (Unsharp Mask)",
                "Gaussian Blur",
                "Sobel Gradient",
                "Laplacian Edge",
                "Lee Despeckle (SAR)",
                "Frost Despeckle (SAR)",
                "Kuan Despeckle (SAR)",
                "Median Filter",
                "Boxcar Filter",
            ])
            row_f.addWidget(self.filter_combo)
            layout.addLayout(row_f)

            row_w = QHBoxLayout()
            row_w.addWidget(QLabel("Window Size / Radius:", self))
            self.win_spin = QSpinBox(self)
            self.win_spin.setRange(1, 31)
            self.win_spin.setSingleStep(2)
            self.win_spin.setValue(5)
            row_w.addWidget(self.win_spin)
            layout.addLayout(row_w)

            btn_row = QHBoxLayout()
            self.apply_btn = QPushButton("Apply Filter", self)
            self.apply_btn.clicked.connect(self.accept)
            self.cancel_btn = QPushButton("Cancel", self)
            self.cancel_btn.clicked.connect(self.reject)
            btn_row.addWidget(self.apply_btn)
            btn_row.addWidget(self.cancel_btn)
            layout.addLayout(btn_row)
