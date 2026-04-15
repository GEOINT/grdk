# -*- coding: utf-8 -*-
"""
OWPauliDecomposer Widget - Pauli-basis decomposition from a coherency matrix.

Accepts a :class:`~grdk.widgets._signals.CovarianceMatrixSignal` containing a
T3 coherency matrix and produces a Pauli-basis RGB composite via the
T3 diagonal:

    - Red   = T3[1,1]  → double-bounce / dihedral scattering
    - Green = T3[2,2]  → volume / depolarization scattering
    - Blue  = T3[0,0]  → surface / odd-bounce scattering

The output ``"Image Stack"`` signal is a single-reader
:class:`~grdk.widgets._signals.ImageStack` compatible with
:class:`OWStackViewer`.  Spatial metadata and geolocation are forwarded
from the upstream matrix node so that the Stack Viewer can display
correct coordinate information.

Typical canvas chain
--------------------

    OWImageLoader → OWCovarianceMatrix → OWPauliDecomposer → OWStackViewer

A C3 covariance matrix may be connected but will trigger a warning; the
diagonal of C3 has a different physical interpretation (lexicographic powers)
and produces a valid but non-standard RGB.

Author
------
Ava Courtney
courtney-ava@zai.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-04-15

Modified
-------
2026-04-15
"""

# Standard library
import logging
from pathlib import Path
from typing import List, Optional

# Third-party
import numpy as np
from orangewidget import gui
from orangewidget.settings import Setting
from orangewidget.widget import OWBaseWidget, Input, Output, Msg

from PyQt6.QtCore import QThread, pyqtSignal as Signal
from PyQt6.QtWidgets import (
    QComboBox,
    QLabel,
    QPushButton,
    QTextEdit,
)

# GRDK internal
from grdk.widgets._signals import CovarianceMatrixSignal, ImageStack

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal in-memory reader wrapping a pre-computed RGB array
# ---------------------------------------------------------------------------

class _RGBArrayReader:
    """Duck-typed reader exposing a ``(3, rows, cols)`` float32 RGB array.

    Implements the interface expected by downstream GRDK widgets
    (``read_chip``, ``read_full``, context manager).
    """

    def __init__(self, rgb: np.ndarray, metadata, source_name: str = 'pauli_rgb') -> None:
        assert rgb.ndim == 3 and rgb.shape[0] == 3, \
            f"Expected (3, H, W) RGB array, got {rgb.shape}"
        self._rgb = rgb  # (3, H, W) float32

        self.filepath = Path(source_name)
        self.metadata = metadata

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        chip = self._rgb[:, row_start:row_end, col_start:col_end]
        if bands is not None:
            chip = chip[bands]
        return chip

    def read_full(self, bands: Optional[List[int]] = None) -> np.ndarray:
        if bands is not None:
            return self._rgb[bands]
        return self._rgb

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __repr__(self) -> str:
        return f"_RGBArrayReader(shape={self._rgb.shape})"


# ---------------------------------------------------------------------------
# Background worker thread
# ---------------------------------------------------------------------------

class _PauliWorker(QThread):
    """Compute the Pauli RGB in a background thread.

    Emits ``log_message`` for heartbeat status updates and
    ``result_ready`` on success or ``error_occurred`` on failure.
    """

    log_message = Signal(str)
    result_ready = Signal(object)    # ImageStack
    error_occurred = Signal(str)

    def __init__(
        self,
        matrix_signal: CovarianceMatrixSignal,
        representation: str,
        percentile_low: float,
        percentile_high: float,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._matrix_signal = matrix_signal
        self._representation = representation
        self._percentile_low = percentile_low
        self._percentile_high = percentile_high

    def run(self) -> None:
        try:
            T3 = self._matrix_signal.matrix
            self.log_message.emit("Initializing Pauli decomposition…")
            self.log_message.emit(
                f"  Input: {self._matrix_signal.matrix_type} matrix  "
                f"shape={T3.shape}  dtype={T3.dtype}"
            )

            # Step 1 — extract T3 diagonal (spatially-averaged Pauli powers)
            self.log_message.emit("Extracting T3 diagonal (Pauli powers)…")
            surface_pwr = T3[0, 0].real   # Blue  — <|S_HH + S_VV|²>/2
            db_pwr      = T3[1, 1].real   # Red   — <|S_HH - S_VV|²>/2
            volume_pwr  = T3[2, 2].real   # Green — 2·<|S_HV|²>

            self.log_message.emit(
                f"  surface ∈ [{surface_pwr.min():.3g}, {surface_pwr.max():.3g}]  "
                f"double-bounce ∈ [{db_pwr.min():.3g}, {db_pwr.max():.3g}]  "
                f"volume ∈ [{volume_pwr.min():.3g}, {volume_pwr.max():.3g}]"
            )

            # Step 2 — apply representation conversion
            self.log_message.emit(
                f"Applying '{self._representation}' representation…"
            )
            r = self._stretch(self._convert(db_pwr))
            g = self._stretch(self._convert(volume_pwr))
            b = self._stretch(self._convert(surface_pwr))

            # Step 3 — composite RGB
            self.log_message.emit(
                "Compositing RGB (R=double-bounce, G=volume, B=surface)…"
            )
            rgb = np.stack([r, g, b], axis=0)  # (3, H, W) float32
            self.log_message.emit(
                f"  RGB: shape={rgb.shape}  "
                f"range=[{rgb.min():.4f}, {rgb.max():.4f}]"
            )

            # Step 4 — wrap as ImageStack for OWStackViewer
            self.log_message.emit("Wrapping result as Image Stack…")
            from grdl.image_processing.decomposition.pauli import PauliDecomposition
            from grdl.IO.models.base import ImageMetadata
            rgb_metadata = ImageMetadata(
                format='PauliRGB',
                rows=int(rgb.shape[1]),
                cols=int(rgb.shape[2]),
                dtype=str(rgb.dtype),
                bands=3,
                axis_order='CYX',
                channel_metadata=PauliDecomposition.rgb_channel_metadata(),
            )
            reader = _RGBArrayReader(rgb, rgb_metadata, source_name='Pauli RGB')
            output = ImageStack(
                readers=[reader],
                names=["Pauli RGB"],
                geolocation=self._matrix_signal.geolocation,
                metadata={
                    **self._matrix_signal.source_metadata,
                    'decomposition': 'pauli',
                    'representation': self._representation,
                    'window_size': self._matrix_signal.window_size,
                    'matrix_type': self._matrix_signal.matrix_type,
                },
            )
            self.log_message.emit(
                "Done — emitted Pauli RGB as Image Stack."
            )
            self.result_ready.emit(output)

        except Exception as exc:
            logger.exception("Pauli decomposition failed")
            self.error_occurred.emit(str(exc))

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------

    def _convert(self, power: np.ndarray) -> np.ndarray:
        """Apply representation conversion to a real-valued power map."""
        power = power.astype(np.float32)
        if self._representation == 'db':
            floor = np.finfo(np.float32).tiny
            return 10.0 * np.log10(np.maximum(power, floor))
        if self._representation == 'magnitude':
            return np.sqrt(np.maximum(power, 0.0))
        return power  # 'power'

    def _stretch(self, data: np.ndarray) -> np.ndarray:
        """Percentile-stretch *data* to ``[0, 1]`` float32."""
        v_lo = float(np.nanpercentile(data, self._percentile_low))
        v_hi = float(np.nanpercentile(data, self._percentile_high))
        if v_hi > v_lo:
            out = (data - v_lo) / (v_hi - v_lo)
        else:
            out = np.zeros_like(data)
        return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# OWPauliDecomposer widget
# ---------------------------------------------------------------------------

class OWPauliDecomposer(OWBaseWidget):
    """Pauli-basis quad-pol decomposition from a coherency matrix.

    Accepts a :class:`~grdk.widgets._signals.CovarianceMatrixSignal` carrying
    a T3 coherency matrix and emits an ``"Image Stack"`` containing the
    Pauli-basis RGB composite compatible with
    :class:`~grdk.widgets.geodev.ow_stack_viewer.OWStackViewer`.

    The three Pauli components are mapped to display channels as follows:

    ===============  ==================  =========
    Component        T3 element          Display
    ===============  ==================  =========
    Double-bounce    T3[1,1]             Red
    Volume           T3[2,2]             Green
    Surface          T3[0,0]             Blue
    ===============  ==================  =========

    Connect after :class:`~grdk.widgets.geodev.ow_covariance_matrix.OWCovarianceMatrix`
    and before :class:`~grdk.widgets.geodev.ow_stack_viewer.OWStackViewer`.

    Computation runs in a background thread with heartbeat log messages
    streamed to the main area.
    """

    name = "Pauli Decomposer"
    description = (
        "Pauli-basis decomposition from T3 coherency matrix "
        "(T3 diagonal → R/G/B composite)"
    )
    icon = "icons/pauli.svg"
    category = "GEODEV"
    priority = 35

    class Inputs:
        cov_matrix = Input(
            "Covariance Matrix", CovarianceMatrixSignal, auto_summary=False
        )

    class Outputs:
        image_stack = Output("Image Stack", ImageStack, auto_summary=False)

    class Warning(OWBaseWidget.Warning):
        no_input = Msg("No covariance matrix received.")
        not_coherency = Msg(
            "Input is a {} matrix; T3 coherency matrix expected for "
            "standard Pauli decomposition.  The result is valid but "
            "uses a non-Pauli physical interpretation."
        )

    class Error(OWBaseWidget.Error):
        decomposition_failed = Msg("Decomposition failed: {}")

    # Persisted settings
    representation: str = Setting("db")
    percentile_low: float = Setting(2.0)
    percentile_high: float = Setting(98.0)

    want_main_area = True

    def __init__(self) -> None:
        super().__init__()

        self._input_signal: Optional[CovarianceMatrixSignal] = None
        self._worker: Optional[_PauliWorker] = None

        # --- Control area ---
        box = gui.vBox(self.controlArea, "Decomposition Settings")

        box.layout().addWidget(QLabel("Intensity representation:"))
        self._repr_combo = QComboBox(self)
        self._repr_combo.addItem("Decibels (dB)", "db")
        self._repr_combo.addItem("Magnitude", "magnitude")
        self._repr_combo.addItem("Power", "power")
        idx = self._repr_combo.findData(self.representation)
        if idx >= 0:
            self._repr_combo.setCurrentIndex(idx)
        self._repr_combo.currentIndexChanged.connect(self._on_repr_changed)
        box.layout().addWidget(self._repr_combo)

        # Input status
        status_box = gui.vBox(self.controlArea, "Input Status")
        self._status_label = QLabel("No data")
        self._status_label.setWordWrap(True)
        status_box.layout().addWidget(self._status_label)

        # Run button
        self._btn_run = QPushButton("Run Decomposition", self)
        self._btn_run.clicked.connect(self._on_run)
        self._btn_run.setEnabled(False)
        self.controlArea.layout().addWidget(self._btn_run)

        # --- Main area: log ---
        self._log_text = QTextEdit(self.mainArea)
        self._log_text.setReadOnly(True)
        self.mainArea.layout().addWidget(self._log_text)

    # ------------------------------------------------------------------
    # Orange signal handlers
    # ------------------------------------------------------------------

    @Inputs.cov_matrix
    def set_cov_matrix(
        self, signal: Optional[CovarianceMatrixSignal]
    ) -> None:
        """Receive a covariance/coherency matrix signal."""
        self._input_signal = signal
        self.Warning.no_input.clear()
        self.Warning.not_coherency.clear()
        self.Error.decomposition_failed.clear()
        self._log_text.clear()

        if signal is None or signal.matrix is None:
            self.Warning.no_input()
            self._status_label.setText("No data")
            self._btn_run.setEnabled(False)
            self.Outputs.image_stack.send(None)
            return

        mat = signal.matrix
        if mat.ndim != 4 or mat.shape[0] < 3 or mat.shape[1] < 3:
            self.Error.decomposition_failed(
                f"Expected (N, N, rows, cols) matrix with N ≥ 3, "
                f"got {mat.shape}"
            )
            self._btn_run.setEnabled(False)
            self.Outputs.image_stack.send(None)
            return

        if signal.matrix_type != 'T3':
            self.Warning.not_coherency(signal.matrix_type)

        rows, cols = mat.shape[2], mat.shape[3]
        self._status_label.setText(
            f"Type: {signal.matrix_type}\n"
            f"Size: {rows}×{cols}\n"
            f"Window: {signal.window_size}px"
        )
        self._btn_run.setEnabled(True)

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _on_repr_changed(self, _index: int) -> None:
        self.representation = self._repr_combo.currentData()

    def _on_run(self) -> None:
        """Dispatch background Pauli decomposition."""
        if self._input_signal is None or self._input_signal.matrix is None:
            return

        if self._worker is not None and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()

        self.Error.decomposition_failed.clear()
        self._log_text.clear()
        self._btn_run.setEnabled(False)

        self._worker = _PauliWorker(
            matrix_signal=self._input_signal,
            representation=self.representation,
            percentile_low=self.percentile_low,
            percentile_high=self.percentile_high,
            parent=self,
        )
        self._worker.log_message.connect(self._on_log)
        self._worker.result_ready.connect(self._on_worker_result)
        self._worker.error_occurred.connect(self._on_worker_error)
        self._worker.finished.connect(
            lambda: self._btn_run.setEnabled(
                self._input_signal is not None
                and self._input_signal.matrix is not None
            )
        )
        self._worker.start()

    def _on_log(self, message: str) -> None:
        self._log_text.append(message)

    def _on_worker_result(self, stack: ImageStack) -> None:
        self.Outputs.image_stack.send(stack)

    def _on_worker_error(self, message: str) -> None:
        self.Error.decomposition_failed(message)
        self.Outputs.image_stack.send(None)

    def onDeleteWidget(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()
        super().onDeleteWidget()
