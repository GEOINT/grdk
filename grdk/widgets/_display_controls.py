# -*- coding: utf-8 -*-
"""
Display Controls - Convenience UI builder for ImageCanvas display settings.

Builds a QGroupBox with sliders, spinboxes, and combo boxes that drive
an ImageCanvas's DisplaySettings. Follows the same pattern as
``_param_controls.py``.

Dependencies
------------
PyQt6

Author
------
Claude Code (Anthropic)

Contributor
-----------
Steven Siebert

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
from dataclasses import replace
from typing import Any, Dict, Optional, Sequence

# Third-party
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import Qt

from grdk.viewers.image_canvas import (
    AVAILABLE_COLORMAPS,
    DisplaySettings,
    ImageCanvas,
)


ALL_CONTROLS = (
    'window', 'percentile', 'contrast', 'brightness',
    'gamma', 'colormap', 'band',
)


def build_display_controls(
    parent: QWidget,
    canvas: ImageCanvas,
    show: Optional[Sequence[str]] = None,
) -> QGroupBox:
    """Build a QGroupBox with display adjustment controls for an ImageCanvas.

    Parameters
    ----------
    parent : QWidget
        Parent widget for the group box.
    canvas : ImageCanvas
        The canvas whose display_settings will be driven by the controls.
    show : Optional[Sequence[str]]
        Which controls to include. Subset of:
        ``('window', 'percentile', 'contrast', 'brightness',
          'gamma', 'colormap', 'band')``.
        None (default) shows all.

    Returns
    -------
    QGroupBox
        Group box containing the selected controls.
    """
    visible = set(show) if show is not None else set(ALL_CONTROLS)

    group = QGroupBox("Display", parent)
    layout = QVBoxLayout(group)

    controls: Dict[str, Any] = {}

    def _update() -> None:
        """Read all controls and push a new DisplaySettings to the canvas."""
        s = canvas.display_settings

        if 'window' in controls:
            auto_cb = controls['window_auto']
            if auto_cb.isChecked():
                s = replace(s, window_min=None, window_max=None)
            else:
                s = replace(
                    s,
                    window_min=controls['window_min'].value(),
                    window_max=controls['window_max'].value(),
                )

        if 'percentile' in controls:
            s = replace(
                s,
                percentile_low=controls['percentile_low'].value(),
                percentile_high=controls['percentile_high'].value(),
            )

        if 'contrast' in controls:
            s = replace(s, contrast=controls['contrast'].value() / 100.0)

        if 'brightness' in controls:
            s = replace(s, brightness=controls['brightness'].value() / 100.0)

        if 'gamma' in controls:
            s = replace(s, gamma=controls['gamma'].value())

        if 'colormap' in controls:
            s = replace(s, colormap=controls['colormap'].currentData())

        if 'band' in controls:
            val = controls['band'].value()
            s = replace(s, band_index=val if val >= 0 else None)

        canvas.set_display_settings(s)

    # --- Window/Level ---
    if 'window' in visible:
        row = QHBoxLayout()
        auto_cb = QCheckBox("Auto", group)
        auto_cb.setChecked(True)
        row.addWidget(auto_cb)

        win_min = QDoubleSpinBox(group)
        win_min.setRange(-1e9, 1e9)
        win_min.setDecimals(2)
        win_min.setPrefix("Min: ")
        win_min.setEnabled(False)
        row.addWidget(win_min)

        win_max = QDoubleSpinBox(group)
        win_max.setRange(-1e9, 1e9)
        win_max.setDecimals(2)
        win_max.setValue(255.0)
        win_max.setPrefix("Max: ")
        win_max.setEnabled(False)
        row.addWidget(win_max)

        def _on_auto_toggled(checked: bool) -> None:
            win_min.setEnabled(not checked)
            win_max.setEnabled(not checked)
            _update()

        auto_cb.toggled.connect(_on_auto_toggled)
        win_min.valueChanged.connect(lambda _: _update())
        win_max.valueChanged.connect(lambda _: _update())

        layout.addLayout(row)
        controls['window'] = True
        controls['window_auto'] = auto_cb
        controls['window_min'] = win_min
        controls['window_max'] = win_max

    # --- Percentile ---
    if 'percentile' in visible:
        row = QHBoxLayout()
        row.addWidget(QLabel("Percentile:", group))

        pct_low = QDoubleSpinBox(group)
        pct_low.setRange(0.0, 100.0)
        pct_low.setValue(0.0)
        pct_low.setSuffix("%")
        pct_low.setSingleStep(1.0)
        row.addWidget(pct_low)

        pct_high = QDoubleSpinBox(group)
        pct_high.setRange(0.0, 100.0)
        pct_high.setValue(100.0)
        pct_high.setSuffix("%")
        pct_high.setSingleStep(1.0)
        row.addWidget(pct_high)

        pct_low.valueChanged.connect(lambda _: _update())
        pct_high.valueChanged.connect(lambda _: _update())

        layout.addLayout(row)
        controls['percentile'] = True
        controls['percentile_low'] = pct_low
        controls['percentile_high'] = pct_high

    # --- Contrast ---
    if 'contrast' in visible:
        row = QHBoxLayout()
        row.addWidget(QLabel("Contrast:", group))

        contrast_slider = QSlider(Qt.Orientation.Horizontal, group)
        contrast_slider.setRange(0, 300)
        contrast_slider.setValue(100)
        contrast_slider.valueChanged.connect(lambda _: _update())
        row.addWidget(contrast_slider)

        layout.addLayout(row)
        controls['contrast'] = contrast_slider

    # --- Brightness ---
    if 'brightness' in visible:
        row = QHBoxLayout()
        row.addWidget(QLabel("Brightness:", group))

        brightness_slider = QSlider(Qt.Orientation.Horizontal, group)
        brightness_slider.setRange(-100, 100)
        brightness_slider.setValue(0)
        brightness_slider.valueChanged.connect(lambda _: _update())
        row.addWidget(brightness_slider)

        layout.addLayout(row)
        controls['brightness'] = brightness_slider

    # --- Gamma ---
    if 'gamma' in visible:
        row = QHBoxLayout()
        row.addWidget(QLabel("Gamma:", group))

        gamma_spin = QDoubleSpinBox(group)
        gamma_spin.setRange(0.1, 5.0)
        gamma_spin.setValue(1.0)
        gamma_spin.setSingleStep(0.1)
        gamma_spin.valueChanged.connect(lambda _: _update())
        row.addWidget(gamma_spin)

        layout.addLayout(row)
        controls['gamma'] = gamma_spin

    # --- Colormap ---
    if 'colormap' in visible:
        row = QHBoxLayout()
        row.addWidget(QLabel("Colormap:", group))

        cmap_combo = QComboBox(group)
        for name in AVAILABLE_COLORMAPS:
            cmap_combo.addItem(name.title(), name)
        cmap_combo.currentIndexChanged.connect(lambda _: _update())
        row.addWidget(cmap_combo)

        layout.addLayout(row)
        controls['colormap'] = cmap_combo

    # --- Band selector ---
    if 'band' in visible:
        row = QHBoxLayout()
        row.addWidget(QLabel("Band:", group))

        band_spin = QSpinBox(group)
        band_spin.setRange(-1, 255)
        band_spin.setValue(-1)
        band_spin.setSpecialValueText("Auto")
        band_spin.valueChanged.connect(lambda _: _update())
        row.addWidget(band_spin)

        layout.addLayout(row)
        controls['band'] = band_spin

    return group
