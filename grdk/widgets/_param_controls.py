# -*- coding: utf-8 -*-
"""
Parameter Control Builder - Auto-generate Qt controls from ParamSpec.

Provides functions to create Qt widgets (sliders, spinboxes, comboboxes,
checkboxes) from GRDL ParamSpec declarations. Used by multiple
GEODEV widgets to build parameter editing UIs dynamically.

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
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QGroupBox,
        QLabel,
        QLineEdit,
        QSlider,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
    from PyQt6.QtCore import Qt

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False


def build_param_controls(
    specs: tuple,
    parent: Optional[Any] = None,
    on_changed: Optional[Callable] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Build a QGroupBox with controls for each ParamSpec.

    Parameters
    ----------
    specs : tuple
        Tuple of ParamSpec instances.
    parent : Optional[QWidget]
        Parent widget.
    on_changed : Optional[Callable]
        Callback invoked when any parameter value changes.
        Signature: on_changed(param_name: str, value: Any)

    Returns
    -------
    Tuple[QGroupBox, Dict[str, QWidget]]
        (group_box, control_map) where control_map maps parameter
        names to their Qt control widgets.
    """
    if not _QT_AVAILABLE:
        raise ImportError(
            "Qt is required for parameter controls. "
            "Install with: pip install orange3"
        )

    group = QGroupBox("Parameters", parent)
    layout = QFormLayout(group)
    controls: Dict[str, Any] = {}

    for spec in specs:
        label = spec.name.replace('_', ' ').title()

        if spec.param_type is bool:
            widget = QCheckBox(group)
            default = spec.default if not spec.required else False
            widget.setChecked(bool(default))
            if on_changed:
                widget.toggled.connect(
                    lambda val, n=spec.name: on_changed(n, val)
                )

        elif spec.choices is not None:
            widget = QComboBox(group)
            for choice in spec.choices:
                widget.addItem(str(choice), choice)
            if not spec.required and spec.default in spec.choices:
                idx = spec.choices.index(spec.default)
                widget.setCurrentIndex(idx)
            if on_changed:
                widget.currentIndexChanged.connect(
                    lambda idx, w=widget, n=spec.name: on_changed(
                        n, w.itemData(idx)
                    )
                )

        elif spec.param_type is int:
            widget = QSpinBox(group)
            if spec.min_value is not None:
                widget.setMinimum(int(spec.min_value))
            else:
                widget.setMinimum(-999999)
            if spec.max_value is not None:
                widget.setMaximum(int(spec.max_value))
            else:
                widget.setMaximum(999999)
            if not spec.required:
                widget.setValue(int(spec.default) if spec.default is not None else 0)
            if on_changed:
                widget.valueChanged.connect(
                    lambda val, n=spec.name: on_changed(n, val)
                )

        elif spec.param_type is float:
            widget = QDoubleSpinBox(group)
            widget.setDecimals(4)
            if spec.min_value is not None:
                widget.setMinimum(float(spec.min_value))
            else:
                widget.setMinimum(-999999.0)
            if spec.max_value is not None:
                widget.setMaximum(float(spec.max_value))
            else:
                widget.setMaximum(999999.0)
            # Derive step from range if available
            if spec.min_value is not None and spec.max_value is not None:
                step = (float(spec.max_value) - float(spec.min_value)) / 100.0
                widget.setSingleStep(max(step, 0.0001))
            else:
                widget.setSingleStep(0.01)
            if not spec.required:
                widget.setValue(float(spec.default) if spec.default is not None else 0.0)
            elif spec.default is not None:
                widget.setValue(float(spec.default))
            if on_changed:
                widget.valueChanged.connect(
                    lambda val, n=spec.name: on_changed(n, val)
                )

        elif spec.param_type is str:
            widget = QLineEdit(group)
            if spec.default is not None:
                widget.setText(str(spec.default))
            elif not spec.required:
                widget.setText("")
            if on_changed:
                widget.textChanged.connect(
                    lambda text, n=spec.name: on_changed(n, text)
                )

        else:
            # Unknown type — use a label as placeholder
            type_name = getattr(spec.param_type, '__name__', str(spec.param_type))
            widget = QLabel(f"({type_name})", group)

        layout.addRow(f"{label}:", widget)
        controls[spec.name] = widget

    return group, controls


def get_param_values(
    specs: tuple,
    controls: Dict[str, Any],
) -> Dict[str, Any]:
    """Read current values from parameter controls.

    Parameters
    ----------
    specs : tuple
        ParamSpec instances.
    controls : Dict[str, QWidget]
        Control widgets from build_param_controls().

    Returns
    -------
    Dict[str, Any]
        Parameter name → current value.
    """
    values: Dict[str, Any] = {}
    for spec in specs:
        widget = controls.get(spec.name)
        if widget is None:
            continue

        if spec.param_type is bool:
            values[spec.name] = widget.isChecked()
        elif spec.choices is not None:
            values[spec.name] = widget.currentData()
        elif spec.param_type is int:
            values[spec.name] = widget.value()
        elif spec.param_type is float:
            values[spec.name] = widget.value()
        elif spec.param_type is str and hasattr(widget, 'text'):
            values[spec.name] = widget.text()

    return values
