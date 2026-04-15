# -*- coding: utf-8 -*-
"""
OWCovarianceMatrix Widget - Polarimetric matrix computation for SAR imagery.

Accepts a quad-pol :class:`~grdk.widgets._signals.ImageStack` (HH, HV, VH,
VV) sourced from NISAR or BIOMASS readers and outputs a
:class:`~grdk.widgets._signals.CovarianceMatrixSignal` carrying the
spatially-averaged polarimetric matrix.

Two matrix types are supported:

**T3 — Coherency Matrix (Pauli basis)**
    For quad-pol reciprocal scatter, the Pauli target vector is:
    k_P = [(S_HH + S_VV), (S_HH - S_VV), 2·S_HV]^T / √2
    T3 = <k_P · k_P^H>  (spatially averaged with a boxcar window)
    The diagonal T3[i,i] gives the spatially-averaged Pauli component power.
    **Use T3 as input to the Pauli Decomposer.**

**C3 — Covariance Matrix (lexicographic basis)**
    k_L = [S_HH, √2·S_HV, S_VV]^T
    C3 = <k_L · k_L^H>

Both outputs are ``(3, 3, rows, cols)`` complex64 arrays with
``axis_order='CCYX'``.  Spatial metadata and polarimetric tags are
forwarded in the emitted :class:`~grdk.widgets._signals.CovarianceMatrixSignal`
so the downstream Pauli Decomposer can validate compatibility and the
Stack Viewer can display correct geolocation coordinates.

Typical canvas chain
--------------------

    OWImageLoader → OWCovarianceMatrix → OWPauliDecomposer → OWStackViewer

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
"""

# Standard library
import logging
from typing import Optional

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
    QSpinBox,
    QTextEdit,
)

# GRDK internal
from grdk.widgets._signals import CovarianceMatrixSignal, ImageStack
from grdk.widgets._pol_utils import (
    extract_quad_pol_arrays_strided,
    get_polarimetric_mode,
    is_quad_pol,
    _native_dims,
    _reader_polarization,
    _reader_quad_pol_channels,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background worker thread
# ---------------------------------------------------------------------------

class _MatrixWorker(QThread):
    """Compute the polarimetric matrix in a background thread.

    Emits ``log_message`` for heartbeat status updates and
    ``result_ready`` on success or ``error_occurred`` on failure.
    """

    log_message = Signal(str)
    result_ready = Signal(object)    # CovarianceMatrixSignal
    error_occurred = Signal(str)

    def __init__(
        self,
        stack: ImageStack,
        matrix_type: str,
        window_size: int,
        max_pixels: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._stack = stack
        self._matrix_type = matrix_type
        self._window_size = window_size
        self._max_pixels = max_pixels

    def run(self) -> None:  # noqa: max-complexity
        try:
            self.log_message.emit("Initializing matrix computation…")

            # Step 1 — read quad-pol channels
            self.log_message.emit("Reading quad-pol channels (HH, HV, VH, VV)…")
            shh, shv, svh, svv = extract_quad_pol_arrays_strided(
                self._stack, max_pixels=self._max_pixels
            )

            rows_full, cols_full = _native_dims(self._stack)
            self.log_message.emit(f"  Shape: {shh.shape[0]}\u00d7{shh.shape[1]}")

            # Step 2 — build the matrix
            if self._matrix_type == 'T3':
                from grdl.image_processing.decomposition.pol_matrix import (
                    CoherencyMatrix,
                )
                self.log_message.emit(
                    f"Building T3 coherency matrix "
                    f"(window={self._window_size})…"
                )
                mat = CoherencyMatrix(window_size=self._window_size).compute(
                    shh, shv, svh, svv
                )
            else:
                from grdl.image_processing.decomposition.pol_matrix import (
                    CovarianceMatrix,
                )
                self.log_message.emit(
                    f"Building C3 covariance matrix "
                    f"(window={self._window_size})…"
                )
                mat = CovarianceMatrix(window_size=self._window_size).compute(
                    shh, shv, svh, svv
                )

            self.log_message.emit(
                f"Matrix computed. Shape: {mat.shape}  dtype={mat.dtype}"
            )

            # Step 3 — package the result
            self.log_message.emit("Packaging result and propagating metadata…")
            signal = CovarianceMatrixSignal(
                matrix=mat,
                matrix_type=self._matrix_type,
                window_size=self._window_size,
                source_metadata={
                    **self._stack.metadata,
                    'matrix_type': self._matrix_type,
                    'window_size': self._window_size,
                    'original_rows': rows_full,
                    'original_cols': cols_full,
                    'downsampled_rows': shh.shape[0],
                    'downsampled_cols': shh.shape[1],
                },
                geolocation=self._stack.geolocation,
            )
            self.log_message.emit(
                f"Done — emitting {self._matrix_type} "
                f"({mat.shape[2]}×{mat.shape[3]} pixels)."
            )
            self.result_ready.emit(signal)

        except Exception as exc:
            logger.exception("Matrix computation failed")
            self.error_occurred.emit(str(exc))


# ---------------------------------------------------------------------------
# OWCovarianceMatrix widget
# ---------------------------------------------------------------------------

class OWCovarianceMatrix(OWBaseWidget):
    """Polarimetric covariance/coherency matrix computation.

    Accepts a quad-pol :class:`~grdk.widgets._signals.ImageStack` (HH, HV,
    VH, VV) and emits a :class:`~grdk.widgets._signals.CovarianceMatrixSignal`
    carrying either:

    - **T3** — the Pauli coherency matrix (default, required for
      :class:`OWPauliDecomposer`).
    - **C3** — the lexicographic covariance matrix.

    Both are ``(3, 3, rows, cols)`` complex64, spatially averaged with a
    boxcar window of configurable size.

    Computation runs in a background thread so the UI remains responsive
    during processing. Status updates are streamed to the log panel.
    """

    name = "Covariance Matrix"
    description = (
        "Compute the quad-pol polarimetric matrix "
        "(T3 coherency or C3 covariance) from an image stack."
    )
    icon = "icons/covariance_matrix.svg"
    category = "GEODEV"
    priority = 32

    class Inputs:
        image_stack = Input("Image Stack", ImageStack, auto_summary=False)

    class Outputs:
        cov_matrix = Output(
            "Covariance Matrix", CovarianceMatrixSignal, auto_summary=False
        )

    class Warning(OWBaseWidget.Warning):
        incomplete_quad_pol = Msg(
            "Stack is not quad-pol. Missing channels: {}"
        )
        no_input = Msg("No image stack received.")

    class Error(OWBaseWidget.Error):
        computation_failed = Msg("Matrix computation failed: {}")

    # Persisted settings
    matrix_type: str = Setting("T3")
    window_size: int = Setting(7)
    max_pixels: int = Setting(4096 * 4096)

    want_main_area = True

    def __init__(self) -> None:
        super().__init__()

        self._input_stack: Optional[ImageStack] = None
        self._worker: Optional[_MatrixWorker] = None

        # --- Control area ---
        box = gui.vBox(self.controlArea, "Matrix Settings")

        box.layout().addWidget(QLabel("Matrix type:"))
        self._type_combo = QComboBox(self)
        self._type_combo.addItem("T3 — Coherency (Pauli basis)", "T3")
        self._type_combo.addItem("C3 — Covariance (lexicographic)", "C3")
        idx = self._type_combo.findData(self.matrix_type)
        if idx >= 0:
            self._type_combo.setCurrentIndex(idx)
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        box.layout().addWidget(self._type_combo)

        box.layout().addWidget(QLabel("Averaging window size:"))
        self._window_spin = QSpinBox(self)
        self._window_spin.setMinimum(3)
        self._window_spin.setMaximum(63)
        self._window_spin.setSingleStep(2)
        self._window_spin.setValue(self.window_size)
        self._window_spin.setToolTip(
            "Boxcar spatial averaging window (pixels, odd integer). "
            "Larger windows reduce speckle but blur spatial detail."
        )
        self._window_spin.valueChanged.connect(self._on_window_changed)
        box.layout().addWidget(self._window_spin)

        box.layout().addWidget(QLabel("Resolution cap (MP, 0 = full):"))
        self._max_mp_spin = QSpinBox(self)
        self._max_mp_spin.setMinimum(0)
        self._max_mp_spin.setMaximum(500)
        self._max_mp_spin.setValue(self.max_pixels // (1024 * 1024))
        self._max_mp_spin.setToolTip(
            "Downsample input to at most this many megapixels before "
            "computing the matrix. 0 = full resolution."
        )
        self._max_mp_spin.valueChanged.connect(self._on_max_mp_changed)
        box.layout().addWidget(self._max_mp_spin)

        # Polarimetric status
        pol_box = gui.vBox(self.controlArea, "Polarimetric Status")
        self._status_label = QLabel("No data")
        self._status_label.setWordWrap(True)
        pol_box.layout().addWidget(self._status_label)

        # Compute button
        self._btn_compute = QPushButton("Compute Matrix", self)
        self._btn_compute.clicked.connect(self._on_compute)
        self._btn_compute.setEnabled(False)
        self.controlArea.layout().addWidget(self._btn_compute)

        # --- Main area: log ---
        self._log_text = QTextEdit(self.mainArea)
        self._log_text.setReadOnly(True)
        self.mainArea.layout().addWidget(self._log_text)

    # ------------------------------------------------------------------
    # Orange signal handlers
    # ------------------------------------------------------------------

    @Inputs.image_stack
    def set_image_stack(self, stack: Optional[ImageStack]) -> None:
        """Receive an image stack and validate polarimetric completeness."""
        self._input_stack = stack
        self.Warning.no_input.clear()
        self.Warning.incomplete_quad_pol.clear()
        self.Error.computation_failed.clear()
        self._log_text.clear()

        if stack is None or len(stack) == 0:
            self.Warning.no_input()
            self._status_label.setText("No data")
            self._btn_compute.setEnabled(False)
            self.Outputs.cov_matrix.send(None)
            return

        self._update_status_label(stack)

        if is_quad_pol(stack):
            self._btn_compute.setEnabled(True)
        else:
            missing = self._missing_channels(stack)
            self.Warning.incomplete_quad_pol(', '.join(sorted(missing)))
            self._btn_compute.setEnabled(False)
            self.Outputs.cov_matrix.send(None)

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _on_type_changed(self, _index: int) -> None:
        self.matrix_type = self._type_combo.currentData()

    def _on_window_changed(self, value: int) -> None:
        if value % 2 == 0:
            value += 1
            self._window_spin.blockSignals(True)
            self._window_spin.setValue(value)
            self._window_spin.blockSignals(False)
        self.window_size = value

    def _on_max_mp_changed(self, value: int) -> None:
        self.max_pixels = value * 1024 * 1024 if value > 0 else 0

    def _on_compute(self) -> None:
        """Dispatch background matrix computation."""
        if self._input_stack is None or not is_quad_pol(self._input_stack):
            return

        # Cancel any running worker
        if self._worker is not None and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()

        self.Error.computation_failed.clear()
        self._log_text.clear()
        self._btn_compute.setEnabled(False)

        self._worker = _MatrixWorker(
            stack=self._input_stack,
            matrix_type=self.matrix_type,
            window_size=self.window_size,
            max_pixels=self.max_pixels,
            parent=self,
        )
        self._worker.log_message.connect(self._on_log)
        self._worker.result_ready.connect(self._on_worker_result)
        self._worker.error_occurred.connect(self._on_worker_error)
        self._worker.finished.connect(
            lambda: self._btn_compute.setEnabled(
                is_quad_pol(self._input_stack)
                if self._input_stack is not None else False
            )
        )
        self._worker.start()

    def _on_log(self, message: str) -> None:
        self._log_text.append(message)

    def _on_worker_result(self, signal: CovarianceMatrixSignal) -> None:
        self.Outputs.cov_matrix.send(signal)

    def _on_worker_error(self, message: str) -> None:
        self.Error.computation_failed(message)
        self.Outputs.cov_matrix.send(None)

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _update_status_label(self, stack: ImageStack) -> None:
        pol_mode = get_polarimetric_mode(stack)
        pols = set()
        for reader in stack.readers:
            mb = _reader_quad_pol_channels(reader)
            if mb:
                pols.update(mb.keys())
            else:
                p = _reader_polarization(reader)
                if p:
                    pols.add(p)

        mode_str = (
            pol_mode.value.replace('_', '-').upper()
            if pol_mode else 'unknown'
        )
        channels_str = ', '.join(sorted(pols)) if pols else '(none found)'
        self._status_label.setText(
            f"Mode: {mode_str}\nChannels: {channels_str}"
        )

    @staticmethod
    def _missing_channels(stack: ImageStack) -> set:
        present = set()
        for reader in stack.readers:
            mb = _reader_quad_pol_channels(reader)
            if mb:
                present.update(mb.keys())
            else:
                p = _reader_polarization(reader)
                if p:
                    present.add(p)
        return {'HH', 'HV', 'VH', 'VV'} - present

    def onDeleteWidget(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()
        super().onDeleteWidget()
