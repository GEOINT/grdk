# -*- coding: utf-8 -*-
"""
PanSharpen - Panchromatic image fusion engine.

Reuses GRDL fusion processors (BroveyPanSharpenTransform, SimpleMeanFusionTransform)
to sharpen multispectral and hyperspectral imagery using high-resolution PAN.

Author: Steven Siebert
License: MIT License
"""

from typing import Any, Optional
import numpy as np

# GRDL Processors
from grdl.image_processing.fusion import (
    BroveyPanSharpenTransform,
    SimpleMeanFusionTransform,
    brovey_pan_sharpen,
    simple_mean_fusion as grdl_mean_fusion,
)

try:
    from PyQt6.QtWidgets import (
        QComboBox,
        QDialog,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QVBoxLayout,
    )
    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False


def brovey_transform(pan: np.ndarray, ms: np.ndarray) -> np.ndarray:
    """Brovey Transform Pan-sharpening delegating to GRDL."""
    if ms.ndim != 3:
        raise ValueError("MS image must be 3D (C, H, W)")
    if pan.ndim != 2:
        raise ValueError("PAN image must be 2D (H, W)")
    return brovey_pan_sharpen(pan, ms)


def simple_mean_fusion(pan: np.ndarray, ms: np.ndarray) -> np.ndarray:
    """Simple additive fusion delegating to GRDL."""
    if ms.ndim != 3 or pan.ndim != 2:
        raise ValueError("MS must be (C, H, W) and PAN must be (H, W)")
    return grdl_mean_fusion(pan, ms, pan_weight=0.5)


if _QT_AVAILABLE:

    class PanSharpenDialog(QDialog):
        """Dialog to configure and execute pan-sharpening."""

        def __init__(self, parent: Optional[Any] = None) -> None:
            super().__init__(parent)
            self.setWindowTitle("Pan-Sharpening Fusion")
            self.resize(320, 180)

            layout = QVBoxLayout(self)

            row_m = QHBoxLayout()
            row_m.addWidget(QLabel("Method:", self))
            self.method_combo = QComboBox(self)
            self.method_combo.addItems(["Brovey Transform", "Simple Mean Fusion"])
            row_m.addWidget(self.method_combo)
            layout.addLayout(row_m)

            btn_row = QHBoxLayout()
            self.apply_btn = QPushButton("Fuse", self)
            self.apply_btn.clicked.connect(self.accept)
            self.cancel_btn = QPushButton("Cancel", self)
            self.cancel_btn.clicked.connect(self.reject)
            btn_row.addWidget(self.apply_btn)
            btn_row.addWidget(self.cancel_btn)
            layout.addLayout(btn_row)
