# -*- coding: utf-8 -*-
"""
Tests for grdk.widgets._param_controls — build_param_controls & get_param_values.

Author
------
Claude Code (Anthropic)

Contributor
-----------
Steven Siebert

Created
-------
2026-02-06
"""

from unittest.mock import MagicMock

import pytest

try:
    from PyQt6.QtWidgets import QApplication
    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False


pytestmark = pytest.mark.skipif(not _QT_AVAILABLE, reason="Qt not available")


def _make_spec(name, param_type=float, default=None, required=False,
               min_value=None, max_value=None, choices=None):
    """Create a mock TunableParameterSpec."""
    spec = MagicMock()
    spec.name = name
    spec.param_type = param_type
    spec.default = default
    spec.required = required
    spec.min_value = min_value
    spec.max_value = max_value
    spec.choices = choices
    return spec


class TestBuildParamControls:
    def test_float_control(self, qapp):
        from grdk.widgets._param_controls import build_param_controls
        spec = _make_spec("sigma", float, default=1.5, min_value=0.0, max_value=10.0)
        group, controls = build_param_controls((spec,))
        assert "sigma" in controls
        assert abs(controls["sigma"].value() - 1.5) < 0.01

    def test_int_control(self, qapp):
        from grdk.widgets._param_controls import build_param_controls
        spec = _make_spec("kernel_size", int, default=3, min_value=1, max_value=31)
        group, controls = build_param_controls((spec,))
        assert "kernel_size" in controls
        assert controls["kernel_size"].value() == 3

    def test_bool_control(self, qapp):
        from grdk.widgets._param_controls import build_param_controls
        spec = _make_spec("normalize", bool, default=True)
        group, controls = build_param_controls((spec,))
        assert "normalize" in controls
        assert controls["normalize"].isChecked() is True

    def test_str_control(self, qapp):
        from grdk.widgets._param_controls import build_param_controls
        spec = _make_spec("label", str, default="test")
        group, controls = build_param_controls((spec,))
        assert "label" in controls
        assert controls["label"].text() == "test"

    def test_choices_control(self, qapp):
        from grdk.widgets._param_controls import build_param_controls
        spec = _make_spec("method", str, default="bilinear",
                          choices=["nearest", "bilinear", "bicubic"])
        group, controls = build_param_controls((spec,))
        assert "method" in controls

    def test_callback_fires(self, qapp):
        from grdk.widgets._param_controls import build_param_controls
        spec = _make_spec("sigma", float, default=1.0, min_value=0.0, max_value=10.0)
        called = {}
        def on_changed(name, value):
            called[name] = value
        group, controls = build_param_controls((spec,), on_changed=on_changed)
        controls["sigma"].setValue(5.0)
        assert called.get("sigma") == 5.0

    def test_multiple_specs(self, qapp):
        from grdk.widgets._param_controls import build_param_controls
        specs = (
            _make_spec("sigma", float, default=1.0),
            _make_spec("iterations", int, default=3),
            _make_spec("invert", bool, default=False),
        )
        group, controls = build_param_controls(specs)
        assert len(controls) == 3


class TestGetParamValues:
    def test_read_float(self, qapp):
        from grdk.widgets._param_controls import build_param_controls, get_param_values
        spec = _make_spec("sigma", float, default=2.0, min_value=0.0, max_value=10.0)
        group, controls = build_param_controls((spec,))
        controls["sigma"].setValue(7.5)
        values = get_param_values((spec,), controls)
        assert abs(values["sigma"] - 7.5) < 0.01

    def test_read_int(self, qapp):
        from grdk.widgets._param_controls import build_param_controls, get_param_values
        spec = _make_spec("size", int, default=5, min_value=1, max_value=99)
        group, controls = build_param_controls((spec,))
        controls["size"].setValue(42)
        values = get_param_values((spec,), controls)
        assert values["size"] == 42

    def test_read_bool(self, qapp):
        from grdk.widgets._param_controls import build_param_controls, get_param_values
        spec = _make_spec("flag", bool, default=False)
        group, controls = build_param_controls((spec,))
        controls["flag"].setChecked(True)
        values = get_param_values((spec,), controls)
        assert values["flag"] is True

    def test_read_str(self, qapp):
        from grdk.widgets._param_controls import build_param_controls, get_param_values
        spec = _make_spec("name", str, default="hello")
        group, controls = build_param_controls((spec,))
        controls["name"].setText("world")
        values = get_param_values((spec,), controls)
        assert values["name"] == "world"
