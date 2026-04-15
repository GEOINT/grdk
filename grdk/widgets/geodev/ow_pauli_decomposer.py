# -*- coding: utf-8 -*-
"""
OWPauliDecomposer Widget - Quad-pol Pauli-basis decomposition for SAR imagery.

Accepts a quad-pol :class:`~grdk.widgets._signals.ImageStack` (HH, HV, VH,
VV) and produces a Pauli-basis RGB composite via the two-step pipeline:

1. **CoherencyMatrix** — spatially-averaged T3 (speckle-reduced Pauli powers)
2. **Pauli RGB from T3 diagonal**:
   - Red   = T3[1,1]  (double-bounce / dihedral)
   - Green = T3[2,2]  (volume / depolarization)
   - Blue  = T3[0,0]  (surface / odd-bounce)

Both NISAR (per-polarization readers) and BIOMASS (single multi-band reader)
stacks are handled transparently.

The output ``"Image Stack"`` signal carries the Pauli RGB result as a
single in-memory reader.  Connect this widget between OWImageLoader and
OWStackViewer to replace the currently viewed data with the Pauli RGB:

    OWImageLoader  →  OWPauliDecomposer  →  OWStackViewer

Dependencies
------------
orange-widget-base, grdl, grdl-tools

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
--------
2026-04-15
"""

# Standard library
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party
import numpy as np
from orangewidget import gui
from orangewidget.settings import Setting
from orangewidget.widget import OWBaseWidget, Input, Output, Msg

from PyQt6.QtWidgets import (
    QComboBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QTextEdit,
)

# GRDK internal
from grdk.widgets._signals import ImageStack
from grdk.widgets._pol_utils import (
    extract_quad_pol_arrays,
    extract_quad_pol_arrays_strided,
    get_polarimetric_mode,
    is_quad_pol,
    _native_dims,
    _reader_polarization,
    _reader_quad_pol_channels,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal in-memory reader wrapping a pre-computed RGB array
# ---------------------------------------------------------------------------

class _RGBArrayReader:
    """Duck-typed reader exposing a ``(3, rows, cols)`` float32 RGB array.

    Implements the interface expected by downstream GRDK widgets
    (``read_chip``, ``read_full``, context manager).
    """

    def __init__(self, rgb: np.ndarray, source_name: str = 'pauli_rgb') -> None:
        from grdl.IO.models.base import ChannelMetadata, ImageMetadata

        assert rgb.ndim == 3 and rgb.shape[0] == 3, \
            f"Expected (3, H, W) RGB array, got {rgb.shape}"
        self._rgb = rgb  # (3, H, W) float32

        self.filepath = Path(source_name)
        self.metadata = ImageMetadata(
            format='PauliRGB',
            rows=int(rgb.shape[1]),
            cols=int(rgb.shape[2]),
            dtype=str(rgb.dtype),
            bands=3,
            axis_order='CYX',
            channel_metadata=[
                ChannelMetadata(
                    index=0, name='double_bounce', role='decomposition',
                    source_indices=[0, 3],
                    extras={'pauli_component': 'double_bounce',
                            'formula': 'T3[1,1] = <|S_HH-S_VV|²>/2',
                            'display': 'Red'},
                ),
                ChannelMetadata(
                    index=1, name='volume', role='decomposition',
                    source_indices=[1, 2],
                    extras={'pauli_component': 'volume',
                            'formula': 'T3[2,2] = 2·<|S_HV|²>',
                            'display': 'Green'},
                ),
                ChannelMetadata(
                    index=2, name='surface', role='decomposition',
                    source_indices=[0, 3],
                    extras={'pauli_component': 'surface',
                            'formula': 'T3[0,0] = <|S_HH+S_VV|²>/2',
                            'display': 'Blue'},
                ),
            ],
        )

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
# OWPauliDecomposer widget
# ---------------------------------------------------------------------------

class OWPauliDecomposer(OWBaseWidget):
    """Pauli-basis quad-pol decomposition.

    Accepts a quad-pol :class:`~grdk.widgets._signals.ImageStack` (HH, HV,
    VH, VV) and emits an ``"Image Stack"`` containing a Pauli-basis RGB
    composite.  Insert between OWImageLoader and OWStackViewer to replace
    the viewed image with the Pauli decomposition result.

    Supports both NISAR (four separate readers) and BIOMASS (single
    multi-band reader) loading patterns.
    """

    name = "Pauli Decomposer"
    description = (
        "Pauli-basis quad-pol decomposition "
        "(HH/HV/VH/VV → Coherency Matrix T3 → RGB)"
    )
    icon = "icons/pauli.svg"
    category = "GEODEV"
    priority = 35

    class Inputs:
        image_stack = Input("Image Stack", ImageStack, auto_summary=False)

    class Outputs:
        image_stack = Output("Image Stack", ImageStack, auto_summary=False)

    class Warning(OWBaseWidget.Warning):
        incomplete_quad_pol = Msg(
            "Stack is not quad-pol. Missing channels: {}"
        )
        no_input = Msg("No image stack received.")

    class Error(OWBaseWidget.Error):
        decomposition_failed = Msg("Decomposition failed: {}")

    # Persisted settings
    representation: str = Setting("db")
    window_size: int = Setting(7)
    percentile_low: float = Setting(2.0)
    percentile_high: float = Setting(98.0)
    max_pixels: int = Setting(4096 * 4096)  # pixel cap for preview mode

    want_main_area = True

    def __init__(self) -> None:
        super().__init__()

        self._input_stack: Optional[ImageStack] = None
        self._pol_mode = None

        # --- Control area -----------------------------------------------
        box = gui.vBox(self.controlArea, "Decomposition Settings")

        # Averaging window
        box.layout().addWidget(QLabel("Coherency window size:"))
        self._window_spin = QSpinBox(self)
        self._window_spin.setMinimum(3)
        self._window_spin.setMaximum(63)
        self._window_spin.setSingleStep(2)   # keep it odd
        self._window_spin.setValue(self.window_size)
        self._window_spin.valueChanged.connect(self._on_window_changed)
        box.layout().addWidget(self._window_spin)

        # Output resolution cap
        box.layout().addWidget(QLabel("Output resolution (MP, 0 = full):"))
        self._max_mp_spin = QSpinBox(self)
        self._max_mp_spin.setMinimum(0)
        self._max_mp_spin.setMaximum(500)
        self._max_mp_spin.setValue(self.max_pixels // (1024 * 1024))
        self._max_mp_spin.setToolTip(
            "Downsample input to at most this many megapixels before "
            "computing the coherency matrix.  Use 0 for full resolution "
            "(may crash on large images).  Default: 16 MP."
        )
        self._max_mp_spin.valueChanged.connect(self._on_max_mp_changed)
        box.layout().addWidget(self._max_mp_spin)

        # Representation selector
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

        # Polarimetric status
        pol_box = gui.vBox(self.controlArea, "Polarimetric Status")
        self._status_label = QLabel("No data")
        self._status_label.setWordWrap(True)
        pol_box.layout().addWidget(self._status_label)

        # Run button
        self._btn_run = QPushButton("Run Decomposition", self)
        self._btn_run.clicked.connect(self._on_run)
        self._btn_run.setEnabled(False)
        self.controlArea.layout().addWidget(self._btn_run)

        # --- Main area: log output --------------------------------------
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
        self.Error.decomposition_failed.clear()
        self._log_text.clear()

        if stack is None or len(stack) == 0:
            self.Warning.no_input()
            self._status_label.setText("No data")
            self._btn_run.setEnabled(False)
            self.Outputs.image_stack.send(None)
            return

        self._pol_mode = get_polarimetric_mode(stack)
        self._update_status_label(stack)

        if is_quad_pol(stack):
            self._btn_run.setEnabled(True)
        else:
            missing = self._missing_channels(stack)
            self.Warning.incomplete_quad_pol(', '.join(sorted(missing)))
            self._btn_run.setEnabled(False)
            self.Outputs.image_stack.send(None)

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _on_window_changed(self, value: int) -> None:
        # Enforce odd window size
        if value % 2 == 0:
            value += 1
            self._window_spin.blockSignals(True)
            self._window_spin.setValue(value)
            self._window_spin.blockSignals(False)
        self.window_size = value

    def _on_repr_changed(self, _index: int) -> None:
        self.representation = self._repr_combo.currentData()

    def _on_max_mp_changed(self, value: int) -> None:
        self.max_pixels = value * 1024 * 1024 if value > 0 else 0

    def _on_run(self) -> None:
        """Execute Pauli decomposition and emit the RGB stack."""
        if self._input_stack is None or not is_quad_pol(self._input_stack):
            return

        self.Error.decomposition_failed.clear()
        self._log_text.clear()

        try:
            from grdl_tools.sar import PauliDecompositionProcessor
        except ImportError:
            self.Error.decomposition_failed(
                "grdl-tools is required. Install with: pip install grdl-tools"
            )
            return

        try:
            self._log_text.append("Reading quad-pol channels…")
            shh, shv, svh, svv = extract_quad_pol_arrays_strided(
                self._input_stack, max_pixels=self.max_pixels
            )
            rows_full, cols_full = _native_dims(self._input_stack)
            if (rows_full and cols_full
                    and self.max_pixels > 0
                    and rows_full * cols_full > self.max_pixels):
                import math
                stride = math.ceil(math.sqrt(rows_full * cols_full / self.max_pixels))
                self._log_text.append(
                    f"  Center crop {shh.shape[0]}\u00d7{shh.shape[1]} of "
                    f"full {rows_full}\u00d7{cols_full} (stride={stride}, "
                    f"cap={self.max_pixels//1_000_000} MP)"
                )
            self._log_text.append(
                f"  HH: {shh.shape} {shh.dtype}  "
                f"HV: {shv.shape}  VH: {svh.shape}  VV: {svv.shape}"
            )

            proc = PauliDecompositionProcessor(
                window_size=self.window_size,
                representation=self.representation,
                percentile_low=self.percentile_low,
                percentile_high=self.percentile_high,
            )
            self._log_text.append(
                f"Running {proc!r}…"
            )

            # Stack to (4, H, W) for the processor's execute() API
            from grdl.IO.models.base import ImageMetadata, ChannelMetadata
            source = np.stack([shh, shv, svh, svv], axis=0)  # (4, H, W)

            input_meta = ImageMetadata(
                format=getattr(
                    getattr(self._input_stack.readers[0], 'metadata', None),
                    'format', 'unknown'
                ),
                rows=int(source.shape[1]),
                cols=int(source.shape[2]),
                dtype=str(source.dtype),
                bands=4,
                axis_order='CYX',
                channel_metadata=[
                    ChannelMetadata(index=0, name='HH', role='measurement',
                                    polarization='HH'),
                    ChannelMetadata(index=1, name='HV', role='measurement',
                                    polarization='HV'),
                    ChannelMetadata(index=2, name='VH', role='measurement',
                                    polarization='VH'),
                    ChannelMetadata(index=3, name='VV', role='measurement',
                                    polarization='VV'),
                ],
            )

            rgb, _ = proc.execute(input_meta, source)
            # rgb: (3, H, W) float32 [0, 1]

            self._log_text.append(
                f"RGB composite: {rgb.shape} {rgb.dtype}  "
                f"range=[{rgb.min():.4f}, {rgb.max():.4f}]"
            )

            reader = _RGBArrayReader(rgb, source_name='Pauli RGB')
            output = ImageStack(
                readers=[reader],
                names=["Pauli RGB"],
                geolocation=self._input_stack.geolocation,
                metadata={
                    **self._input_stack.metadata,
                    'polarimetric_mode': 'quad_pol',
                    'decomposition': 'pauli',
                    'representation': self.representation,
                    'window_size': self.window_size,
                },
            )
            self.Outputs.image_stack.send(output)
            self._log_text.append(
                "Done — emitted Pauli RGB as Image Stack."
            )

        except Exception as exc:
            logger.exception("Pauli decomposition failed")
            self.Error.decomposition_failed(str(exc))

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _update_status_label(self, stack: ImageStack) -> None:
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
            self._pol_mode.value.replace('_', '-').upper()
            if self._pol_mode else 'unknown'
        )
        n_readers = len(stack.readers)
        if n_readers == 1:
            bands = getattr(
                getattr(stack.readers[0], 'metadata', None), 'bands', None
            )
            reader_note = (
                f'1 reader ({bands} bands)' if bands and bands > 1
                else '1 reader'
            )
        else:
            reader_note = f'{n_readers} readers'
        channels_str = ', '.join(sorted(pols)) if pols else '(none found)'
        self._status_label.setText(
            f"Mode: {mode_str}\n"
            f"Channels: {channels_str}\n"
            f"Readers: {reader_note}"
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

