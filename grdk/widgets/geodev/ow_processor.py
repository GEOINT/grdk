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
2026-02-16
"""

# Standard library
from typing import Any, Dict, Optional

# Third-party
from orangewidget import gui
from orangewidget.settings import Setting
from orangewidget.widget import OWBaseWidget, Input, Output, Msg

from PyQt6.QtWidgets import (
    QComboBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

# GRDK internal
from grdl_rt.execution.discovery import (
    filter_processors_for_connection,
    filter_processors_for_modality,
    get_processor_tags,
    get_all_modalities,
    get_all_categories,
)
from grdl_rt.execution.workflow import ProcessingStep
from grdk.widgets._signals import (
    ChipSetSignal,
    ImageStack,
    ProcessingPipelineSignal,
    get_modality_hint,
)


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

    class Inputs:
        chip_set = Input("Chip Set", ChipSetSignal, auto_summary=False)
        upstream_pipeline = Input("Upstream Pipeline", ProcessingPipelineSignal, auto_summary=False)

    class Outputs:
        pipeline = Output("Pipeline", ProcessingPipelineSignal, auto_summary=False)

    selected_processor: str = Setting("")

    want_main_area = False

    def __init__(self) -> None:
        super().__init__()

        self._modality: Optional[Any] = None  # ImageModality | None
        self._upstream_output_type: Optional[str] = None
        self._param_controls: Dict[str, Any] = {}
        self._param_group: Optional[QWidget] = None

        # --- Control area ---
        box = gui.vBox(self.controlArea, "Processor")

        # Modality filter (manual override; auto-set from incoming signal)
        self._modality_combo = QComboBox(self)
        self._modality_combo.addItem("Auto (from input)", "auto")
        self._modality_combo.addItem("All Modalities", None)
        for mod in sorted(get_all_modalities(), key=lambda m: m.value):
            self._modality_combo.addItem(mod.value, mod)
        self._modality_combo.currentIndexChanged.connect(self._on_filter_changed)
        box.layout().addWidget(self._modality_combo)

        self._combo = QComboBox(self)
        self._combo.addItem("(select processor)", None)
        self._combo.currentIndexChanged.connect(self._on_processor_changed)
        box.layout().addWidget(self._combo)

        # Version info
        self._version_label = QLabel("", self)
        box.layout().addWidget(self._version_label)

        # Parameter controls container
        self._params_container = QWidget(self)
        self._params_container.setLayout(QVBoxLayout())
        box.layout().addWidget(self._params_container)

        self._rebuild_combo()

        # Restore selection
        if self.selected_processor:
            idx = self._combo.findData(self.selected_processor)
            if idx >= 0:
                self._combo.setCurrentIndex(idx)

    @Inputs.chip_set
    def set_chip_set(self, signal: Optional[ChipSetSignal]) -> None:
        self._modality = get_modality_hint(signal) if signal is not None else None
        self._rebuild_combo()

    @Inputs.upstream_pipeline
    def set_upstream_pipeline(self, signal: Optional[ProcessingPipelineSignal]) -> None:
        self._upstream_output_type = (
            signal.output_port_type if signal is not None else None
        )
        self._rebuild_combo()

    def _effective_modality(self) -> Optional[str]:
        """Return the modality string to filter by, respecting manual override."""
        choice = self._modality_combo.currentData()
        if choice == "auto":
            m = self._modality
            return m.value if m is not None and hasattr(m, "value") else (str(m) if m else None)
        if choice is None:
            return None
        return choice.value if hasattr(choice, "value") else str(choice)

    def _rebuild_combo(self) -> None:
        """Repopulate processor combo using filter_processors_for_connection."""
        modality = self._effective_modality()
        # Use connection-aware filter: respects port types + implicit ANY
        processors = filter_processors_for_connection(
            self._upstream_output_type,
            modality,
        )
        # Fallback when no catalog entries exist (dev environment)
        if not processors:
            processors = filter_processors_for_modality(modality)

        prev = self._combo.currentData()
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItem("(select processor)", None)
        for name in sorted(processors.keys()):
            self._combo.addItem(name, name)
        self._combo.blockSignals(False)

        # Restore previous selection if still available
        idx = self._combo.findData(prev)
        if idx >= 0:
            self._combo.setCurrentIndex(idx)
        elif prev:
            self._on_processor_changed(0)

    def _on_processor_changed(self, index: int) -> None:
        """Handle processor selection change."""
        proc_name = self._combo.itemData(index)
        if proc_name is None:
            return

        self.selected_processor = proc_name
        self._clear_param_controls()

        # Resolve from current filtered set
        modality = self._effective_modality()
        processors = filter_processors_for_connection(self._upstream_output_type, modality)
        if not processors:
            processors = filter_processors_for_modality(modality)
        proc_class = processors.get(proc_name)
        if proc_class is None:
            return

        # Show version and tags
        version = getattr(proc_class, '__processor_version__', '')
        tags = get_processor_tags(proc_class)
        tag_parts = []
        if version:
            tag_parts.append(f"v{version}")
        if tags.get('category'):
            tag_parts.append(tags['category'].value.replace('_', ' '))
        mods = tags.get('modalities', ())
        if mods:
            tag_parts.append(', '.join(m.value for m in mods))
        gpu = getattr(proc_class, '__gpu_compatible__', None)
        if gpu is True:
            tag_parts.append("GPU")
        self._version_label.setText(' | '.join(tag_parts) if tag_parts else "")

        # Build parameter controls from __param_specs__
        specs = getattr(proc_class, '__param_specs__', ())
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

        modality = self._effective_modality()
        processors = filter_processors_for_connection(self._upstream_output_type, modality)
        if not processors:
            processors = filter_processors_for_modality(modality)
        proc_class = processors.get(proc_name)
        version = getattr(proc_class, '__processor_version__', '') if proc_class else ''

        params = {}
        if self._param_controls:
            from grdk.widgets._param_controls import get_param_values
            specs = getattr(proc_class, '__param_specs__', ())
            params = get_param_values(specs, self._param_controls)

        from grdl_rt.execution.workflow import WorkflowDefinition
        wf = WorkflowDefinition(name="Single Processor")
        wf.add_step(ProcessingStep(
            processor_name=proc_name,
            processor_version=version,
            params=params,
        ))

        self.Outputs.pipeline.send(ProcessingPipelineSignal(wf))

    def _on_filter_changed(self, _index: int) -> None:
        """Rebuild processor combo when modality override changes."""
        self._rebuild_combo()

    def _clear_param_controls(self) -> None:
        """Remove existing parameter controls."""
        if self._param_group is not None:
            self._param_group.setParent(None)
            self._param_group.deleteLater()
            self._param_group = None
        self._param_controls.clear()
