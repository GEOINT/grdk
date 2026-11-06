# -*- coding: utf-8 -*-
"""
Filters - Spatial filtering and SAR despeckle processing engine.

Provides edge enhancement, sharpening, and standard radar speckle reduction
filters (Lee, Frost, Kuan, Gamma MAP, Median, Boxcar).

Author: Steven Siebert
License: MIT License
"""

from typing import Any, Optional
import numpy as np
from scipy import ndimage

try:
    from PyQt6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
        QSpinBox, QDoubleSpinBox, QPushButton, QMessageBox
    )
    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Spatial & Edge Filtering (Optical / PAN / Multi-band)
# ---------------------------------------------------------------------------

def gaussian_blur(arr: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian smoothing filter."""
    if arr.ndim == 3:
        out = np.empty_like(arr)
        for i in range(arr.shape[0]):
            out[i] = ndimage.gaussian_filter(arr[i], sigma=sigma)
        return out
    return ndimage.gaussian_filter(arr, sigma=sigma)


def sharpen_unsharp_mask(arr: np.ndarray, amount: float = 1.5, radius: float = 1.0) -> np.ndarray:
    """Sharpen image using unsharp masking: arr + amount * (arr - blurred)."""
    blurred = gaussian_blur(arr, sigma=radius)
    sharpened = arr + amount * (arr - blurred)
    return sharpened.astype(arr.dtype)


def sobel_gradient(arr: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude using 3x3 Sobel operators."""
    if arr.ndim == 3:
        out = np.empty_like(arr)
        for i in range(arr.shape[0]):
            out[i] = sobel_gradient(arr[i])
        return out
    gx = ndimage.sobel(arr, axis=1)
    gy = ndimage.sobel(arr, axis=0)
    return np.hypot(gx, gy).astype(np.float32)


def laplacian_edge(arr: np.ndarray) -> np.ndarray:
    """Compute second-derivative edge detector via discrete Laplacian."""
    if arr.ndim == 3:
        out = np.empty_like(arr)
        for i in range(arr.shape[0]):
            out[i] = laplacian_edge(arr[i])
        return out
    return np.abs(ndimage.laplace(arr)).astype(np.float32)


def high_pass_filter(arr: np.ndarray, cutoff: float = 2.0) -> np.ndarray:
    """High-pass filter retaining fine spatial details: arr - gaussian_blur(arr)."""
    return arr - gaussian_blur(arr, sigma=cutoff)


# ---------------------------------------------------------------------------
# SAR Despeckle Filtering
# ---------------------------------------------------------------------------

def boxcar_filter(arr: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Simple moving average filter."""
    return ndimage.uniform_filter(arr, size=window_size)


def median_filter(arr: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Non-linear median rank filter."""
    return ndimage.median_filter(arr, size=window_size)


def lee_filter(arr: np.ndarray, window_size: int = 5, num_looks: int = 1) -> np.ndarray:
    """Lee adaptive speckle filter based on local coefficient of variation."""
    img = arr.astype(np.float64)
    local_mean = ndimage.uniform_filter(img, size=window_size)
    local_sq_mean = ndimage.uniform_filter(img ** 2, size=window_size)
    local_var = np.maximum(0.0, local_sq_mean - local_mean ** 2)

    # Noise variance estimate for multiplicative speckle
    noise_var = (local_mean ** 2) / float(max(1, num_looks))
    weight = local_var / np.maximum(1e-7, local_var + noise_var)
    weight = np.clip(weight, 0.0, 1.0)

    filtered = local_mean + weight * (img - local_mean)
    return filtered.astype(arr.dtype)


def kuan_filter(arr: np.ndarray, window_size: int = 5, num_looks: int = 1) -> np.ndarray:
    """Kuan adaptive speckle filter (exact MMSE formulation)."""
    img = arr.astype(np.float64)
    local_mean = ndimage.uniform_filter(img, size=window_size)
    local_sq_mean = ndimage.uniform_filter(img ** 2, size=window_size)
    local_var = np.maximum(0.0, local_sq_mean - local_mean ** 2)

    cu = 1.0 / np.sqrt(float(max(1, num_looks)))
    ci = np.sqrt(local_var) / np.maximum(1e-7, local_mean)
    weight = np.maximum(0.0, 1.0 - (cu ** 2) / np.maximum(1e-7, ci ** 2)) / (1.0 + cu ** 2)
    weight = np.clip(weight, 0.0, 1.0)

    filtered = local_mean + weight * (img - local_mean)
    return filtered.astype(arr.dtype)


def frost_filter(arr: np.ndarray, window_size: int = 5, damping_factor: float = 2.0) -> np.ndarray:
    """Frost exponentially damped adaptive speckle filter."""
    img = arr.astype(np.float64)
    local_mean = ndimage.uniform_filter(img, size=window_size)
    local_sq_mean = ndimage.uniform_filter(img ** 2, size=window_size)
    local_var = np.maximum(0.0, local_sq_mean - local_mean ** 2)

    ci = np.sqrt(local_var) / np.maximum(1e-7, local_mean)
    # Exponential weighting matrix
    weight = np.exp(-damping_factor * ci)
    filtered = local_mean * weight + img * (1.0 - weight)
    return filtered.astype(arr.dtype)


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
            row_w.addWidget(QLabel("Window Size:", self))
            self.win_spin = QSpinBox(self)
            self.win_spin.setRange(3, 15)
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
