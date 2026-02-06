# -*- coding: utf-8 -*-
"""
OWProcessor Widget - Single processing component viewer.

Displays a single GRDL image processor with its tunable parameter
controls, allowing interactive parameter adjustment and preview of
the transform on a single image chip.

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
from typing import Any, Dict, Optional

# Third-party
from orangewidget import gui
from orangewidget.settings import Setting
from orangewidget.widget import OWBaseWidget, Input, Output, Msg

from AnyQt.QtWidgets import (
    QComboBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

# GRDK internal
from grdk.core.workflow import ProcessingStep
from grdk.widgets._signals import ProcessingPipelineSignal


def _discover_processors() -> Dict[str, Any]:
    """Discover all available GRDL ImageTransform and ImageDetector classes.

    Returns
    -------
    Dict[str, Any]
        Mapping of display name → class object.
    """
    processors: Dict[str, Any] = {}
    try:
        import grdl.image_processing as ip
        import inspect

        for name, obj in inspect.getmembers(ip, inspect.isclass):
            # Look for classes with an `apply` method (ImageTransform/Detector)
            if hasattr(obj, 'apply') and not inspect.isabstract(obj):
                processors[name] = obj
    except ImportError:
        pass
    return processors


class OWProcessor(OWBaseWidget):
    """Single processing component with tunable parameters.

    Allows selecting a GRDL processor and adjusting its tunable
    parameters. Emits the configured step as a ProcessingPipeline
    signal for chaining.
    """

    name = "Processor"
    description = "Configure a single image processing component"
    icon = "icons/processor.svg"
    category = "GEODEV"
    priority = 65

    class Outputs:
        pipeline = Output("Pipeline", ProcessingPipelineSignal)

    selected_processor: str = Setting("")

    want_main_area = False

    def __init__(self) -> None:
        super().__init__()

        self._processors = _discover_processors()
        self._param_controls: Dict[str, Any] = {}
        self._param_group: Optional[QWidget] = None

        # --- Control area ---
        box = gui.vBox(self.controlArea, "Processor")

        self._combo = QComboBox(self)
        self._combo.addItem("(select processor)", None)
        for proc_name in sorted(self._processors.keys()):
            self._combo.addItem(proc_name, proc_name)
        self._combo.currentIndexChanged.connect(self._on_processor_changed)
        box.layout().addWidget(self._combo)

        # Version info
        self._version_label = QLabel("", self)
        box.layout().addWidget(self._version_label)

        # Parameter controls container
        self._params_container = QWidget(self)
        self._params_container.setLayout(QVBoxLayout())
        box.layout().addWidget(self._params_container)

        # Restore selection
        if self.selected_processor:
            idx = self._combo.findData(self.selected_processor)
            if idx >= 0:
                self._combo.setCurrentIndex(idx)

    def _on_processor_changed(self, index: int) -> None:
        """Handle processor selection change."""
        proc_name = self._combo.itemData(index)
        if proc_name is None:
            return

        self.selected_processor = proc_name
        self._clear_param_controls()

        proc_class = self._processors.get(proc_name)
        if proc_class is None:
            return

        # Show version
        version = getattr(proc_class, '_processor_version', '')
        self._version_label.setText(f"Version: {version}" if version else "")

        # Build parameter controls from TunableParameterSpec
        specs = getattr(proc_class, 'tunable_parameter_specs', ())
        if specs:
            from grdk.widgets._param_controls import build_param_controls
            group, self._param_controls = build_param_controls(
                specs, self._params_container, on_changed=self._on_param_changed
            )
            self._param_group = group
            self._params_container.layout().addWidget(group)

    def _on_param_changed(self, param_name: str, value: Any) -> None:
        """Handle parameter value changes — emit updated step."""
        self._emit_step()

    def _emit_step(self) -> None:
        """Emit current processor configuration as a pipeline signal."""
        proc_name = self._combo.currentData()
        if proc_name is None:
            return

        proc_class = self._processors.get(proc_name)
        version = getattr(proc_class, '_processor_version', '') if proc_class else ''

        params = {}
        if self._param_controls:
            from grdk.widgets._param_controls import get_param_values
            specs = getattr(proc_class, 'tunable_parameter_specs', ())
            params = get_param_values(specs, self._param_controls)

        from grdk.core.workflow import WorkflowDefinition
        wf = WorkflowDefinition(name="Single Processor")
        wf.add_step(ProcessingStep(
            processor_name=proc_name,
            processor_version=version,
            params=params,
        ))

        self.Outputs.pipeline.send(ProcessingPipelineSignal(wf))

    def _clear_param_controls(self) -> None:
        """Remove existing parameter controls."""
        if self._param_group is not None:
            self._param_group.setParent(None)
            self._param_group.deleteLater()
            self._param_group = None
        self._param_controls.clear()
