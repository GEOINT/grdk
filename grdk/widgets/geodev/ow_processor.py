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
from typing import Any, Dict, Optional

# Third-party
from orangewidget import gui
from orangewidget.settings import Setting
from orangewidget.widget import OWBaseWidget, Input, Output, Msg

from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

# GRDK internal
from grdk.core.discovery import (
    discover_processors, get_processor_tags,
    get_all_modalities, get_all_categories,
)
from grdk.core.workflow import ProcessingStep
from grdk.widgets._signals import ProcessingPipelineSignal


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

        self._processors = discover_processors()
        self._param_controls: Dict[str, Any] = {}
        self._param_group: Optional[QWidget] = None

        # --- Control area ---
        box = gui.vBox(self.controlArea, "Processor")

        # Modality filter
        self._modality_combo = QComboBox(self)
        self._modality_combo.addItem("All Modalities", None)
        for mod in sorted(get_all_modalities()):
            self._modality_combo.addItem(mod, mod)
        self._modality_combo.currentIndexChanged.connect(self._on_filter_changed)
        box.layout().addWidget(self._modality_combo)

        # Category filter
        self._category_combo = QComboBox(self)
        self._category_combo.addItem("All Categories", None)
        for cat in sorted(get_all_categories()):
            self._category_combo.addItem(cat.replace('_', ' ').title(), cat)
        self._category_combo.currentIndexChanged.connect(self._on_filter_changed)
        box.layout().addWidget(self._category_combo)

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

        # Show version and tags
        version = getattr(proc_class, '_processor_version', '')
        tags = get_processor_tags(proc_class)
        tag_parts = []
        if version:
            tag_parts.append(f"v{version}")
        if tags.get('category'):
            tag_parts.append(tags['category'].replace('_', ' '))
        mods = tags.get('modalities', ())
        if mods:
            tag_parts.append(', '.join(mods))
        gpu = getattr(proc_class, '__gpu_compatible__', None)
        if gpu is True:
            tag_parts.append("GPU")
        self._version_label.setText(' | '.join(tag_parts) if tag_parts else "")

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

    def _on_filter_changed(self, _index: int) -> None:
        """Rebuild processor combo when modality/category filters change."""
        modality = self._modality_combo.currentData()
        category = self._category_combo.currentData()

        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItem("(select processor)", None)

        for proc_name in sorted(self._processors.keys()):
            proc_cls = self._processors[proc_name]
            tags = get_processor_tags(proc_cls)
            if modality and modality not in tags.get('modalities', ()):
                continue
            if category and tags.get('category') != category:
                continue
            self._combo.addItem(proc_name, proc_name)

        self._combo.blockSignals(False)

    def _clear_param_controls(self) -> None:
        """Remove existing parameter controls."""
        if self._param_group is not None:
            self._param_group.setParent(None)
            self._param_group.deleteLater()
            self._param_group = None
        self._param_controls.clear()
