# -*- coding: utf-8 -*-
"""
OWPreview Widget - Real-time GPU-accelerated chip preview grid.

Displays a grid of before/after chip pairs showing the effect of
the current processing pipeline on all chips in the set.

Dependencies
------------
orange-widget-base

Author
------
Duane Smalley, PhD
duane.d.smalley@gmail.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-06

Modified
--------
2026-02-06
"""

# Standard library
from typing import Any, Dict, List, Optional

# Third-party
import numpy as np
from orangewidget import gui
from orangewidget.widget import OWBaseWidget, Input, Msg

from AnyQt.QtWidgets import (
    QGridLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from AnyQt.QtGui import QImage, QPixmap
from AnyQt.QtCore import Qt

# GRDK internal
from grdk.core.gpu import GpuBackend
from grdk.widgets._signals import ChipSetSignal, ProcessingPipelineSignal


PREVIEW_THUMB = 160


def _array_to_pixmap(arr: np.ndarray, size: int = PREVIEW_THUMB) -> QPixmap:
    """Convert numpy array to QPixmap thumbnail."""
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    arr = arr.astype(np.float64)
    vmin, vmax = np.nanmin(arr), np.nanmax(arr)
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin) * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        h, w = arr.shape
        qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = np.ascontiguousarray(arr[:, :, :3])
        h, w, _ = arr.shape
        qimg = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    else:
        band = arr[:, :, 0] if arr.ndim == 3 else arr
        h, w = band.shape
        qimg = QImage(band.data, w, h, w, QImage.Format.Format_Grayscale8)

    pixmap = QPixmap.fromImage(qimg)
    return pixmap.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio)


class OWPreview(OWBaseWidget):
    """Real-time GPU preview of pipeline on all chips.

    Shows a scrollable grid of before/after image pairs for every
    chip in the connected ChipSet, processed through the connected
    pipeline.
    """

    name = "Preview"
    description = "Real-time GPU preview of pipeline on chips"
    icon = "icons/preview.svg"
    category = "GEODEV"
    priority = 75

    class Inputs:
        chip_set = Input("Chip Set", ChipSetSignal)
        pipeline = Input("Pipeline", ProcessingPipelineSignal)

    class Warning(OWBaseWidget.Warning):
        no_chips = Msg("No chips connected.")
        no_pipeline = Msg("No pipeline connected.")

    want_main_area = True

    def __init__(self) -> None:
        super().__init__()

        self._chip_set: Optional[Any] = None
        self._pipeline: Optional[Any] = None
        self._gpu = GpuBackend()
        self._processors: Dict[str, Any] = {}

        # Discover processors
        try:
            import grdl.image_processing as ip
            import inspect
            for name, obj in inspect.getmembers(ip, inspect.isclass):
                if hasattr(obj, 'apply') and not inspect.isabstract(obj):
                    self._processors[name] = obj
        except ImportError:
            pass

        # --- Control area ---
        box = gui.vBox(self.controlArea, "Info")
        self._info_label = QLabel("No data", self)
        box.layout().addWidget(self._info_label)

        gpu_info = "GPU" if self._gpu.gpu_available else "CPU only"
        box.layout().addWidget(QLabel(f"Compute: {gpu_info}", self))

        # --- Main area ---
        self._scroll = QScrollArea(self.mainArea)
        self._scroll.setWidgetResizable(True)
        self._container = QWidget()
        self._grid = QGridLayout(self._container)
        self._grid.setSpacing(8)
        self._scroll.setWidget(self._container)
        self.mainArea.layout().addWidget(self._scroll)

    @Inputs.chip_set
    def set_chip_set(self, signal: Optional[ChipSetSignal]) -> None:
        """Receive chip set."""
        if signal is None or signal.chip_set is None:
            self._chip_set = None
            self.Warning.no_chips()
        else:
            self._chip_set = signal.chip_set
            self.Warning.no_chips.clear()
        self._refresh()

    @Inputs.pipeline
    def set_pipeline(self, signal: Optional[ProcessingPipelineSignal]) -> None:
        """Receive processing pipeline."""
        if signal is None or signal.workflow is None:
            self._pipeline = None
            self.Warning.no_pipeline()
        else:
            self._pipeline = signal.workflow
            self.Warning.no_pipeline.clear()
        self._refresh()

    def _refresh(self) -> None:
        """Refresh the preview grid."""
        # Clear grid
        while self._grid.count():
            child = self._grid.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if self._chip_set is None or len(self._chip_set) == 0:
            self._info_label.setText("No chips")
            return

        n_chips = len(self._chip_set)
        has_pipeline = (
            self._pipeline is not None and len(self._pipeline.steps) > 0
        )
        self._info_label.setText(
            f"Chips: {n_chips}, "
            f"Pipeline: {len(self._pipeline.steps) if has_pipeline else 0} steps"
        )

        # Header row
        self._grid.addWidget(QLabel("Source"), 0, 0)
        self._grid.addWidget(QLabel("Before"), 0, 1)
        if has_pipeline:
            self._grid.addWidget(QLabel("After"), 0, 2)

        for i, chip in enumerate(self._chip_set.chips):
            row = i + 1
            source = chip.image_data

            # Source info
            info = QLabel(chip.source_image_name)
            info.setWordWrap(True)
            info.setMaximumWidth(120)
            self._grid.addWidget(info, row, 0)

            # Before image
            before_lbl = QLabel()
            before_lbl.setPixmap(_array_to_pixmap(source))
            before_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._grid.addWidget(before_lbl, row, 1)

            # After image (run pipeline)
            if has_pipeline:
                after_lbl = QLabel()
                result = self._run_pipeline(source.copy())
                after_lbl.setPixmap(_array_to_pixmap(result))
                after_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self._grid.addWidget(after_lbl, row, 2)

    def _run_pipeline(self, source: np.ndarray) -> np.ndarray:
        """Run the pipeline on a single chip."""
        if self._pipeline is None:
            return source

        result = source
        for step in self._pipeline.steps:
            proc_class = self._processors.get(step.processor_name)
            if proc_class is None:
                continue
            try:
                proc = proc_class()
                result = self._gpu.apply_transform(proc, result, **step.params)
            except Exception:
                break

        return result
