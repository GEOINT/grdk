# -*- coding: utf-8 -*-
"""
BandMath - Spectral index and band arithmetic engine.

Computes common remote sensing indices (NDVI, NDWI, NDBI, SAVI) and
arbitrary band math for multispectral and hyperspectral imagery.

Author: Steven Siebert
License: MIT License
"""

from typing import Any, List, Optional
import numpy as np

try:
    from PyQt6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
        QPushButton, QDoubleSpinBox, QMessageBox
    )
    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False


def compute_spectral_index(
    data: np.ndarray,
    index_type: str = "NDVI",
    nir_idx: int = 3,
    red_idx: int = 2,
    green_idx: int = 1,
    swir_idx: int = 5,
    savi_l: float = 0.5,
) -> np.ndarray:
    """Compute standard spectral index from a 3D (C, H, W) array."""
    if data.ndim != 3:
        raise ValueError("Data must be 3D (C, H, W) for spectral index calculation")

    num_bands = data.shape[0]

    def _get_band(idx: int) -> np.ndarray:
        return data[min(idx, num_bands - 1)].astype(np.float32)

    index_type = index_type.upper()
    eps = 1e-7

    if index_type == "NDVI":
        nir = _get_band(nir_idx)
        red = _get_band(red_idx)
        denom = nir + red
        denom[denom == 0] = eps
        return (nir - red) / denom

    elif index_type == "NDWI":
        green = _get_band(green_idx)
        nir = _get_band(nir_idx)
        denom = green + nir
        denom[denom == 0] = eps
        return (green - nir) / denom

    elif index_type == "NDBI":
        swir = _get_band(swir_idx)
        nir = _get_band(nir_idx)
        denom = swir + nir
        denom[denom == 0] = eps
        return (swir - nir) / denom

    elif index_type == "SAVI":
        nir = _get_band(nir_idx)
        red = _get_band(red_idx)
        denom = nir + red + savi_l
        denom[denom == 0] = eps
        return ((nir - red) / denom) * (1.0 + savi_l)

    elif index_type == "RATIO":
        band_a = _get_band(nir_idx)
        band_b = _get_band(red_idx)
        denom = band_b.copy()
        denom[denom == 0] = eps
        return band_a / denom

    else:
        raise ValueError(f"Unknown spectral index type: {index_type}")


if _QT_AVAILABLE:

    class BandMathDialog(QDialog):
        """Dialog to configure and execute band math / index calculations."""

        def __init__(self, band_names: List[str], parent: Optional[Any] = None) -> None:
            super().__init__(parent)
            self.setWindowTitle("Spectral Index & Band Math")
            self.resize(360, 240)

            self.band_names = band_names
            self.selected_index: str = "NDVI"
            self.selected_band_a: int = min(3, len(band_names) - 1)
            self.selected_band_b: int = min(2, len(band_names) - 1)

            layout = QVBoxLayout(self)

            idx_row = QHBoxLayout()
            idx_row.addWidget(QLabel("Index:", self))
            self.index_combo = QComboBox(self)
            self.index_combo.addItems(["NDVI (Vegetation)", "NDWI (Water)", "NDBI (Built-up)", "SAVI (Soil-Adjusted)", "Custom Ratio (A / B)"])
            self.index_combo.currentIndexChanged.connect(self._on_index_changed)
            idx_row.addWidget(self.index_combo)
            layout.addLayout(idx_row)

            self.a_row = QHBoxLayout()
            self.a_label = QLabel("NIR Band:", self)
            self.a_row.addWidget(self.a_label)
            self.a_combo = QComboBox(self)
            self.a_combo.addItems(band_names)
            if len(band_names) > 3:
                self.a_combo.setCurrentIndex(3)
            self.a_row.addWidget(self.a_combo)
            layout.addLayout(self.a_row)

            self.b_row = QHBoxLayout()
            self.b_label = QLabel("Red Band:", self)
            self.b_row.addWidget(self.b_label)
            self.b_combo = QComboBox(self)
            self.b_combo.addItems(band_names)
            if len(band_names) > 2:
                self.b_combo.setCurrentIndex(2)
            self.b_row.addWidget(self.b_combo)
            layout.addLayout(self.b_row)

            btn_row = QHBoxLayout()
            self.apply_btn = QPushButton("Calculate", self)
            self.apply_btn.clicked.connect(self.accept)
            self.cancel_btn = QPushButton("Cancel", self)
            self.cancel_btn.clicked.connect(self.reject)
            btn_row.addWidget(self.apply_btn)
            btn_row.addWidget(self.cancel_btn)
            layout.addLayout(btn_row)

        def _on_index_changed(self, idx: int) -> None:
            text = self.index_combo.currentText()
            if "NDVI" in text:
                self.selected_index = "NDVI"
                self.a_label.setText("NIR Band:")
                self.b_label.setText("Red Band:")
            elif "NDWI" in text:
                self.selected_index = "NDWI"
                self.a_label.setText("Green Band:")
                self.b_label.setText("NIR Band:")
            elif "NDBI" in text:
                self.selected_index = "NDBI"
                self.a_label.setText("SWIR Band:")
                self.b_label.setText("NIR Band:")
            elif "SAVI" in text:
                self.selected_index = "SAVI"
                self.a_label.setText("NIR Band:")
                self.b_label.setText("Red Band:")
            else:
                self.selected_index = "RATIO"
                self.a_label.setText("Numerator (Band A):")
                self.b_label.setText("Denominator (Band B):")
