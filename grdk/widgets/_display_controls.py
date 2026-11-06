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
    'remap', 'window', 'percentile', 'contrast', 'brightness',
    'gamma', 'colormap', 'colorbar', 'band',
)

# SAR remap operators from grdl.contrast.
# Each value is a contrast class instance; .apply is stored as remap_function.
_REMAP_FUNCTIONS: dict = {}
try:
    from grdl.contrast import (
        MangisDensity, Brighter, Darker, HighContrast, PEDF,
        NRLStretch, LogStretch, LinearStretch, ToDecibels,
    )
    _REMAP_FUNCTIONS = {
        'Density':       MangisDensity(),
        'Brighter':      Brighter(),
        'Darker':        Darker(),
        'High Contrast': HighContrast(),
        'PEDF':          PEDF(),
        'NRL':           NRLStretch(),
        'Log':           LogStretch(),
        'Decibels':      ToDecibels(),
        'Linear':        LinearStretch(),
    }
except ImportError:
    pass


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

    group = QGroupBox(parent)
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

        if 'stretch_mode' in controls:
            mode_data = controls['stretch_mode'].currentData()
            s = replace(s, stretch_mode=mode_data)

        if 'remap' in controls:
            remap_name = controls['remap'].currentData()
            if remap_name and remap_name in _REMAP_FUNCTIONS:
                s = replace(s, remap_function=_REMAP_FUNCTIONS[remap_name].apply)
            else:
                s = replace(s, remap_function=None)

        if 'colormap' in controls:
            s = replace(s, colormap=controls['colormap'].currentData())

        if 'invert' in controls:
            s = replace(s, invert_colormap=controls['invert'].isChecked())

        if controls.get('display_mode') and controls['display_mode'].currentText() == "RGB Composite":
            r_idx = controls['rgb_r'].currentData()
            g_idx = controls['rgb_g'].currentData()
            b_idx = controls['rgb_b'].currentData()
            if r_idx is not None and g_idx is not None and b_idx is not None and r_idx >= 0 and g_idx >= 0 and b_idx >= 0:
                s = replace(s, rgb_bands=(r_idx, g_idx, b_idx), band_index=None)
            else:
                s = replace(s, rgb_bands=None)
        elif 'band' in controls:
            val = controls['band'].currentData()
            s = replace(s, band_index=val if val is not None and val >= 0 else None, rgb_bands=None)

        canvas.set_display_settings(s)

    # --- SAR Remap ---
    if 'remap' in visible:
        row = QHBoxLayout()
        remap_label = QLabel("Remap:", group)
        row.addWidget(remap_label)

        remap_combo = QComboBox(group)
        remap_combo.addItem("None (Standard)", "")
        for name in _REMAP_FUNCTIONS:
            remap_combo.addItem(name, name)
        remap_combo.currentIndexChanged.connect(lambda _: _update())
        row.addWidget(remap_combo)

        # Disabled by default — enabled when a SAR image is loaded
        remap_label.setEnabled(False)
        remap_combo.setEnabled(False)

        layout.addLayout(row)
        controls['remap'] = remap_combo
        controls['remap_label'] = remap_label

    # --- Contrast Stretch Mode ---
    stretch_row = QHBoxLayout()
    stretch_row.addWidget(QLabel("Stretch Mode:", group))
    stretch_combo = QComboBox(group)
    stretch_combo.addItem("Linear (Min / Max)", "minmax")
    stretch_combo.addItem("Percentile Clip", "percentile")
    stretch_combo.addItem("Standard Deviation (±2σ)", "stddev")
    stretch_combo.addItem("Logarithmic", "log")
    stretch_combo.addItem("Histogram Equalization", "equalize")
    stretch_combo.addItem("Manual (Window / Level)", "manual")
    stretch_combo.currentIndexChanged.connect(lambda _: _update())
    stretch_row.addWidget(stretch_combo)
    layout.addLayout(stretch_row)
    controls['stretch_mode'] = stretch_combo

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
        row.addWidget(contrast_slider)

        contrast_spin = QSpinBox(group)
        contrast_spin.setRange(0, 300)
        contrast_spin.setValue(100)
        contrast_spin.setFixedWidth(55)
        row.addWidget(contrast_spin)

        # Bidirectional link: slider ↔ spinbox
        contrast_slider.valueChanged.connect(contrast_spin.setValue)
        contrast_spin.valueChanged.connect(contrast_slider.setValue)
        # _update triggers via slider.valueChanged
        contrast_slider.valueChanged.connect(lambda _: _update())

        layout.addLayout(row)
        controls['contrast'] = contrast_slider
        controls['contrast_spin'] = contrast_spin

    # --- Brightness ---
    if 'brightness' in visible:
        row = QHBoxLayout()
        row.addWidget(QLabel("Brightness:", group))

        brightness_slider = QSlider(Qt.Orientation.Horizontal, group)
        brightness_slider.setRange(-100, 100)
        brightness_slider.setValue(0)
        row.addWidget(brightness_slider)

        brightness_spin = QSpinBox(group)
        brightness_spin.setRange(-100, 100)
        brightness_spin.setValue(0)
        brightness_spin.setFixedWidth(55)
        row.addWidget(brightness_spin)

        # Bidirectional link: slider ↔ spinbox
        brightness_slider.valueChanged.connect(brightness_spin.setValue)
        brightness_spin.valueChanged.connect(brightness_slider.setValue)
        # _update triggers via slider.valueChanged
        brightness_slider.valueChanged.connect(lambda _: _update())

        layout.addLayout(row)
        controls['brightness'] = brightness_slider
        controls['brightness_spin'] = brightness_spin

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

    # --- Colorbar toggle ---
    if 'colorbar' in visible:
        colorbar_cb = QCheckBox("Show Colorbar", group)
        colorbar_cb.setChecked(False)
        layout.addWidget(colorbar_cb)
        controls['colorbar'] = colorbar_cb

    # --- Display Inversion ---
    invert_row = QHBoxLayout()
    invert_cb = QCheckBox("Invert Display", group)
    invert_cb.setChecked(getattr(canvas.display_settings, 'invert_colormap', False))
    invert_cb.toggled.connect(lambda _: _update())
    invert_row.addWidget(invert_cb)
    layout.addLayout(invert_row)
    controls['invert'] = invert_cb

    # --- Band selector & RGB Guns ---
    if 'band' in visible:
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:", group))
        mode_combo = QComboBox(group)
        mode_combo.addItems(["Single Band", "RGB Composite"])
        mode_row.addWidget(mode_combo)
        layout.addLayout(mode_row)
        controls['display_mode'] = mode_combo

        # Single band widget
        single_row = QHBoxLayout()
        single_row.addWidget(QLabel("Band:", group))
        band_combo = QComboBox(group)
        band_combo.addItem("Auto", -1)
        band_combo.currentIndexChanged.connect(lambda _: _update())
        single_row.addWidget(band_combo)
        layout.addLayout(single_row)
        controls['band'] = band_combo
        controls['single_row'] = single_row

        # RGB gun widgets
        rgb_widget = QWidget(group)
        rgb_layout = QVBoxLayout(rgb_widget)
        rgb_layout.setContentsMargins(0, 0, 0, 0)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:", rgb_widget))
        preset_combo = QComboBox(rgb_widget)
        preset_combo.addItems([
            "Custom",
            "Natural Color (RGB)",
            "Color Infrared (CIR)",
            "SWIR Agriculture",
            "Geology / Mineral",
            "Urban / Atmospheric",
        ])
        preset_row.addWidget(preset_combo)
        rgb_layout.addLayout(preset_row)
        controls['rgb_preset'] = preset_combo

        # Red Gun
        r_row = QHBoxLayout()
        r_row.addWidget(QLabel("Red Gun:", rgb_widget))
        r_combo = QComboBox(rgb_widget)
        r_combo.currentIndexChanged.connect(lambda _: _update())
        r_row.addWidget(r_combo)
        rgb_layout.addLayout(r_row)
        controls['rgb_r'] = r_combo

        # Green Gun
        g_row = QHBoxLayout()
        g_row.addWidget(QLabel("Green Gun:", rgb_widget))
        g_combo = QComboBox(rgb_widget)
        g_combo.currentIndexChanged.connect(lambda _: _update())
        g_row.addWidget(g_combo)
        rgb_layout.addLayout(g_row)
        controls['rgb_g'] = g_combo

        # Blue Gun
        b_row = QHBoxLayout()
        b_row.addWidget(QLabel("Blue Gun:", rgb_widget))
        b_combo = QComboBox(rgb_widget)
        b_combo.currentIndexChanged.connect(lambda _: _update())
        b_row.addWidget(b_combo)
        rgb_layout.addLayout(b_row)
        controls['rgb_b'] = b_combo

        layout.addWidget(rgb_widget)
        controls['rgb_widget'] = rgb_widget
        rgb_widget.hide()

        def _on_mode_changed(idx: int):
            is_rgb = (idx == 1)
            band_combo.setVisible(not is_rgb)
            single_row.itemAt(0).widget().setVisible(not is_rgb)
            rgb_widget.setVisible(is_rgb)
            _update()

        mode_combo.currentIndexChanged.connect(_on_mode_changed)

        def _apply_preset(name: str):
            count = r_combo.count()
            if count < 3:
                return
            if name == "Natural Color (RGB)":
                r_combo.setCurrentIndex(min(0, count-1))
                g_combo.setCurrentIndex(min(1, count-1))
                b_combo.setCurrentIndex(min(2, count-1))
            elif name == "Color Infrared (CIR)":
                r_combo.setCurrentIndex(min(3 if count > 3 else 0, count-1))
                g_combo.setCurrentIndex(min(0, count-1))
                b_combo.setCurrentIndex(min(1, count-1))
            elif name == "SWIR Agriculture":
                r_combo.setCurrentIndex(min(5 if count > 5 else 0, count-1))
                g_combo.setCurrentIndex(min(3 if count > 3 else 1, count-1))
                b_combo.setCurrentIndex(min(2, count-1))
            elif name == "Geology / Mineral":
                r_combo.setCurrentIndex(min(6 if count > 6 else 0, count-1))
                g_combo.setCurrentIndex(min(4 if count > 4 else 1, count-1))
                b_combo.setCurrentIndex(min(1, count-1))
            elif name == "Urban / Atmospheric":
                r_combo.setCurrentIndex(min(6 if count > 6 else 0, count-1))
                g_combo.setCurrentIndex(min(3 if count > 3 else 1, count-1))
                b_combo.setCurrentIndex(min(0, count-1))
            _update()

        preset_combo.currentTextChanged.connect(_apply_preset)

    # Attach a method to update the band combo when band info changes.
    # Callers can use:  group.update_band_info(list_of_BandInfo)
    def _update_band_combo(band_info_list: list) -> None:
        combo = controls.get('band')
        if combo is None:
            return
        # Preserve the canvas's current band_index so the combo reflects it
        current_band = canvas.display_settings.band_index
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Auto", -1)
        for info in band_info_list:
            label = info.name
            if info.description:
                label = f"{info.name} \u2014 {info.description}"
            combo.addItem(label, info.index)

        # Update RGB gun combos if present
        for gun_key in ['rgb_r', 'rgb_g', 'rgb_b']:
            gun_combo = controls.get(gun_key)
            if gun_combo is not None:
                gun_combo.blockSignals(True)
                gun_combo.clear()
                for info in band_info_list:
                    lbl = info.name
                    if info.description:
                        lbl = f"{info.name} - {info.description}"
                    gun_combo.addItem(lbl, info.index)
                gun_combo.blockSignals(False)

        # Default assignments for RGB guns
        if len(band_info_list) >= 3:
            if controls.get('rgb_r') and controls['rgb_r'].count() >= 3:
                controls['rgb_r'].blockSignals(True)
                controls['rgb_g'].blockSignals(True)
                controls['rgb_b'].blockSignals(True)
                controls['rgb_r'].setCurrentIndex(0)
                controls['rgb_g'].setCurrentIndex(1)
                controls['rgb_b'].setCurrentIndex(2)
                controls['rgb_r'].blockSignals(False)
                controls['rgb_g'].blockSignals(False)
                controls['rgb_b'].blockSignals(False)
        # Select the item matching the canvas's current band_index
        if current_band is not None:
            for idx in range(combo.count()):
                if combo.itemData(idx) == current_band:
                    combo.setCurrentIndex(idx)
                    break
        combo.blockSignals(False)

    group.update_band_info = _update_band_combo  # type: ignore[attr-defined]

    def _set_band_index(band_index: Optional[int]) -> None:
        """Programmatically select a band in the combo and update canvas."""
        combo = controls.get('band')
        if combo is None:
            return
        target = band_index if band_index is not None else -1
        combo.blockSignals(True)
        for idx in range(combo.count()):
            if combo.itemData(idx) == target:
                combo.setCurrentIndex(idx)
                break
        combo.blockSignals(False)
        _update()

    group.set_band_index = _set_band_index  # type: ignore[attr-defined]

    def _set_colormap(name: str) -> None:
        """Programmatically select a colormap in the combo and update canvas."""
        combo = controls.get('colormap')
        if combo is None:
            return
        combo.blockSignals(True)
        for idx in range(combo.count()):
            if combo.itemData(idx) == name:
                combo.setCurrentIndex(idx)
                break
        combo.blockSignals(False)
        _update()

    group.set_colormap = _set_colormap  # type: ignore[attr-defined]

    def _set_remap_enabled(enabled: bool) -> None:
        combo = controls.get('remap')
        if combo is None:
            return
        combo.setEnabled(enabled)
        label = controls.get('remap_label')
        if label is not None:
            label.setEnabled(enabled)
        if not enabled:
            combo.blockSignals(True)
            combo.setCurrentIndex(0)  # Reset to "None (Standard)"
            combo.blockSignals(False)

    group.set_remap_enabled = _set_remap_enabled  # type: ignore[attr-defined]

    def _set_colorbar_enabled(enabled: bool) -> None:
        """Enable or disable the colorbar checkbox (greyed out for RGB)."""
        cb = controls.get('colorbar')
        if cb is None:
            return
        cb.setEnabled(enabled)
        if not enabled:
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)

    group.set_colorbar_enabled = _set_colorbar_enabled  # type: ignore[attr-defined]

    # Expose the colorbar checkbox for external signal wiring
    group.colorbar_checkbox = controls.get('colorbar')  # type: ignore[attr-defined]

    def _sync_from_settings() -> None:
        """Sync all control widgets to match the canvas's current settings.

        Call after programmatic changes to the canvas (e.g. auto-settings
        applied during file load) so the UI reflects the actual state.
        Blocks all signals during the sync to prevent feedback loops.
        """
        s = canvas.display_settings

        if 'window' in controls:
            auto = s.window_min is None and s.window_max is None
            controls['window_auto'].blockSignals(True)
            controls['window_auto'].setChecked(auto)
            controls['window_auto'].blockSignals(False)
            controls['window_min'].setEnabled(not auto)
            controls['window_max'].setEnabled(not auto)
            if not auto:
                controls['window_min'].blockSignals(True)
                controls['window_min'].setValue(s.window_min or 0.0)
                controls['window_min'].blockSignals(False)
                controls['window_max'].blockSignals(True)
                controls['window_max'].setValue(s.window_max or 255.0)
                controls['window_max'].blockSignals(False)

        if 'percentile' in controls:
            controls['percentile_low'].blockSignals(True)
            controls['percentile_low'].setValue(s.percentile_low)
            controls['percentile_low'].blockSignals(False)
            controls['percentile_high'].blockSignals(True)
            controls['percentile_high'].setValue(s.percentile_high)
            controls['percentile_high'].blockSignals(False)

        if 'contrast' in controls:
            controls['contrast'].blockSignals(True)
            controls['contrast'].setValue(int(s.contrast * 100))
            controls['contrast'].blockSignals(False)
            if 'contrast_spin' in controls:
                controls['contrast_spin'].blockSignals(True)
                controls['contrast_spin'].setValue(int(s.contrast * 100))
                controls['contrast_spin'].blockSignals(False)

        if 'brightness' in controls:
            controls['brightness'].blockSignals(True)
            controls['brightness'].setValue(int(s.brightness * 100))
            controls['brightness'].blockSignals(False)
            if 'brightness_spin' in controls:
                controls['brightness_spin'].blockSignals(True)
                controls['brightness_spin'].setValue(int(s.brightness * 100))
                controls['brightness_spin'].blockSignals(False)

        if 'gamma' in controls:
            controls['gamma'].blockSignals(True)
            controls['gamma'].setValue(s.gamma)
            controls['gamma'].blockSignals(False)

        if 'colormap' in controls:
            combo = controls['colormap']
            combo.blockSignals(True)
            for idx in range(combo.count()):
                if combo.itemData(idx) == s.colormap:
                    combo.setCurrentIndex(idx)
                    break
            combo.blockSignals(False)

        if 'band' in controls:
            combo = controls['band']
            target = s.band_index if s.band_index is not None else -1
            combo.blockSignals(True)
            for idx in range(combo.count()):
                if combo.itemData(idx) == target:
                    combo.setCurrentIndex(idx)
                    break
            combo.blockSignals(False)

    group.sync_from_settings = _sync_from_settings  # type: ignore[attr-defined]

    return group
