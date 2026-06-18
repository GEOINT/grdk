# -*- coding: utf-8 -*-
"""
OWOrchestrator Widget - Drag-drop workflow builder with real-time preview.

The heart of GEODEV mode. Provides a component palette (left),
workflow step list (center), and real-time GPU-accelerated chip
preview (right) for building image processing workflows.

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
import logging
import traceback
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Third-party
import numpy as np
from orangewidget import gui
from orangewidget.widget import OWBaseWidget, Input, Output, Msg

from PyQt6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer

# GRDK internal
from grdl_rt.execution.discovery import discover_processors, get_processor_tags
from grdl_rt.execution.gpu import GpuBackend
from grdl_rt.execution.workflow import ProcessingStep, WorkflowDefinition
from grdk.widgets._signals import ChipSetSignal, ProcessingPipelineSignal
from grdk.viewers.image_canvas import ImageCanvasThumbnail


class OWOrchestrator(OWBaseWidget):
    """Drag-drop workflow builder with real-time GPU preview.

    Three-panel layout:
    - Left: Component palette (discovered GRDL processors)
    - Center: Workflow step list with parameter controls
    - Right: Real-time before/after chip preview
    """

    name = "Orchestrator"
    description = "Build image processing workflows with real-time preview"
    icon = "icons/orchestrator.svg"
    category = "GEODEV"
    priority = 70

    class Inputs:
        chip_set = Input("Chip Set", ChipSetSignal, auto_summary=False)

    class Outputs:
        pipeline = Output("Pipeline", ProcessingPipelineSignal, auto_summary=False)

    class Warning(OWBaseWidget.Warning):
        no_chips = Msg("No chips connected for preview.")

    want_main_area = True

    def __init__(self) -> None:
        super().__init__()

        self._processors = discover_processors()
        self._workflow = WorkflowDefinition(name="New Workflow")
        self._chip_set: Optional[Any] = None
        self._selected_chip_index = 0
        self._gpu = GpuBackend()
        self._step_param_controls: List[Dict[str, Any]] = []
        self._preview_running = False  # Execution lock
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(50)  # 50ms debounce
        self._preview_timer.timeout.connect(self._update_preview)

        # Undo/redo stacks (command pattern)
        self._undo_stack: List[Dict[str, Any]] = []
        self._redo_stack: List[Dict[str, Any]] = []

        # --- Control area ---
        box = gui.vBox(self.controlArea, "Workflow")
        self._workflow_name_label = QLabel("Steps: 0", self)
        box.layout().addWidget(self._workflow_name_label)

        # Undo/Redo buttons
        undo_redo_row = QWidget()
        undo_redo_row.setLayout(QHBoxLayout())
        self._btn_undo = QPushButton("Undo")
        self._btn_undo.clicked.connect(self._on_undo)
        self._btn_undo.setEnabled(False)
        self._btn_redo = QPushButton("Redo")
        self._btn_redo.clicked.connect(self._on_redo)
        self._btn_redo.setEnabled(False)
        undo_redo_row.layout().addWidget(self._btn_undo)
        undo_redo_row.layout().addWidget(self._btn_redo)
        box.layout().addWidget(undo_redo_row)

        btn_emit = QPushButton("Emit Pipeline", self)
        btn_emit.clicked.connect(self._on_emit)
        box.layout().addWidget(btn_emit)

        # GPU status indicator
        self._gpu_status_label = QLabel()
        self._update_gpu_status_display()
        box.layout().addWidget(self._gpu_status_label)

        # --- Main area: 3-panel splitter ---
        splitter = QSplitter(Qt.Orientation.Horizontal, self.mainArea)
        self.mainArea.layout().addWidget(splitter)

        # Left panel: Component palette
        left = QWidget()
        left.setLayout(QVBoxLayout())
        left.layout().addWidget(QLabel("Component Palette"))

        # Search filter
        self._palette_filter = QLineEdit()
        self._palette_filter.setPlaceholderText("Filter processors...")
        self._palette_filter.textChanged.connect(self._on_filter_palette)
        left.layout().addWidget(self._palette_filter)

        self._palette_list = QListWidget()
        self._all_processor_names = sorted(self._processors.keys())
        for name in self._all_processor_names:
            item = QListWidgetItem(name)
            self._palette_list.addItem(item)
        left.layout().addWidget(self._palette_list)

        btn_add = QPushButton("Add to Workflow >>")
        btn_add.clicked.connect(self._on_add_step)
        left.layout().addWidget(btn_add)
        splitter.addWidget(left)

        # Center panel: Workflow steps
        center = QWidget()
        center.setLayout(QVBoxLayout())

        # Workflow name editor
        name_row = QWidget()
        name_row.setLayout(QHBoxLayout())
        name_row.layout().addWidget(QLabel("Name:"))
        self._workflow_name_edit = QLineEdit("New Workflow")
        self._workflow_name_edit.textChanged.connect(
            lambda t: setattr(self._workflow, 'name', t)
        )
        name_row.layout().addWidget(self._workflow_name_edit)
        center.layout().addWidget(name_row)

        center.layout().addWidget(QLabel("Workflow Steps"))
        self._steps_list = QListWidget()
        self._steps_list.currentRowChanged.connect(self._on_step_selected)
        center.layout().addWidget(self._steps_list)

        step_btns = QWidget()
        step_btns.setLayout(QHBoxLayout())
        btn_up = QPushButton("Up")
        btn_up.clicked.connect(self._on_move_up)
        btn_down = QPushButton("Down")
        btn_down.clicked.connect(self._on_move_down)
        btn_remove = QPushButton("Remove")
        btn_remove.clicked.connect(self._on_remove_step)
        step_btns.layout().addWidget(btn_up)
        step_btns.layout().addWidget(btn_down)
        step_btns.layout().addWidget(btn_remove)
        center.layout().addWidget(step_btns)

        # Parameter panel for selected step
        self._param_container = QWidget()
        self._param_container.setLayout(QVBoxLayout())
        center.layout().addWidget(self._param_container)
        splitter.addWidget(center)

        # Right panel: Preview
        right = QWidget()
        right.setLayout(QVBoxLayout())
        right.layout().addWidget(QLabel("Preview"))

        # Chip selector
        chip_row = QWidget()
        chip_row.setLayout(QHBoxLayout())
        chip_row.layout().addWidget(QLabel("Chip:"))
        self._chip_spinbox = QSpinBox()
        self._chip_spinbox.setMinimum(0)
        self._chip_spinbox.setMaximum(0)
        self._chip_spinbox.valueChanged.connect(self._on_chip_selected)
        chip_row.layout().addWidget(self._chip_spinbox)
        right.layout().addWidget(chip_row)

        self._before_label = QLabel("Before")
        self._before_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._before_image = ImageCanvasThumbnail(size=256)
        self._before_image.setMinimumSize(128, 128)

        self._after_label = QLabel("After Pipeline")
        self._after_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._after_image = ImageCanvasThumbnail(size=256)
        self._after_image.setMinimumSize(128, 128)

        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: red;")
        self._error_label.setWordWrap(True)

        right.layout().addWidget(self._before_label)
        right.layout().addWidget(self._before_image)
        right.layout().addWidget(self._after_label)
        right.layout().addWidget(self._after_image)
        right.layout().addWidget(self._error_label)
        right.layout().addStretch()
        splitter.addWidget(right)

        splitter.setSizes([200, 300, 300])

    @Inputs.chip_set
    def set_chip_set(self, signal: Optional[ChipSetSignal]) -> None:
        """Receive chip set for preview."""
        if signal is None or signal.chip_set is None or len(signal.chip_set) == 0:
            self._chip_set = None
            self.Warning.no_chips()
            return

        self.Warning.no_chips.clear()
        self._chip_set = signal.chip_set
        self._selected_chip_index = 0
        self._chip_spinbox.setMaximum(max(0, len(signal.chip_set) - 1))
        self._chip_spinbox.setValue(0)
        self._schedule_preview()

    def _on_add_step(self) -> None:
        """Add selected palette item as a workflow step."""
        item = self._palette_list.currentItem()
        if item is None:
            return

        proc_name = item.text()
        proc_class = self._processors.get(proc_name)
        version = getattr(proc_class, '__processor_version__', '') if proc_class else ''

        step = ProcessingStep(
            processor_name=proc_name,
            processor_version=version,
        )
        idx = len(self._workflow.steps)
        self._workflow.add_step(step)
        self._steps_list.addItem(f"{len(self._workflow.steps)}. {proc_name}")
        self._workflow_name_label.setText(f"Steps: {len(self._workflow.steps)}")

        # Record command for undo
        self._record_command({
            'action': 'add_step',
            'index': idx,
            'step': step,
        })

        self._schedule_preview()

    def _on_remove_step(self) -> None:
        """Remove the selected workflow step."""
        row = self._steps_list.currentRow()
        if row < 0 or row >= len(self._workflow.steps):
            return

        step = self._workflow.steps[row]

        # Record command for undo
        self._record_command({
            'action': 'remove_step',
            'index': row,
            'step': step,
        })

        self._workflow.remove_step(row)
        self._rebuild_steps_list()
        self._schedule_preview()

    def _on_move_up(self) -> None:
        """Move the selected step up."""
        row = self._steps_list.currentRow()
        if row <= 0:
            return

        # Record command for undo
        self._record_command({
            'action': 'move_step',
            'from_index': row,
            'to_index': row - 1,
        })

        self._workflow.move_step(row, row - 1)
        self._rebuild_steps_list()
        self._steps_list.setCurrentRow(row - 1)
        self._schedule_preview()

    def _on_move_down(self) -> None:
        """Move the selected step down."""
        row = self._steps_list.currentRow()
        if row < 0 or row >= len(self._workflow.steps) - 1:
            return

        # Record command for undo
        self._record_command({
            'action': 'move_step',
            'from_index': row,
            'to_index': row + 1,
        })

        self._workflow.move_step(row, row + 1)
        self._rebuild_steps_list()
        self._steps_list.setCurrentRow(row + 1)
        self._schedule_preview()

    def _rebuild_steps_list(self) -> None:
        """Refresh the steps list widget from the workflow."""
        self._steps_list.clear()
        for i, step in enumerate(self._workflow.steps):
            self._steps_list.addItem(f"{i + 1}. {step.processor_name}")
        self._workflow_name_label.setText(f"Steps: {len(self._workflow.steps)}")

    def _on_step_selected(self, row: int) -> None:
        """Show parameter controls for the selected step."""
        # Clear existing controls
        layout = self._param_container.layout()
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if row < 0 or row >= len(self._workflow.steps):
            return

        step = self._workflow.steps[row]
        proc_class = self._processors.get(step.processor_name)
        if proc_class is None:
            layout.addWidget(QLabel(f"Processor '{step.processor_name}' not found"))
            return

        specs = getattr(proc_class, '__param_specs__', ())
        if not specs:
            layout.addWidget(QLabel("No tunable parameters"))
            return

        from grdk.widgets._param_controls import build_param_controls

        def on_param_changed(name: str, value: Any) -> None:
            step.params[name] = value
            self._schedule_preview()

        group, controls = build_param_controls(
            specs, self._param_container, on_changed=on_param_changed
        )

        # Set existing param values
        for spec in specs:
            if spec.name in step.params:
                widget = controls.get(spec.name)
                if widget is None:
                    continue
                val = step.params[spec.name]
                if spec.param_type is bool and hasattr(widget, 'setChecked'):
                    widget.setChecked(bool(val))
                elif spec.choices is not None and hasattr(widget, 'setCurrentIndex'):
                    idx = list(spec.choices).index(val) if val in spec.choices else 0
                    widget.setCurrentIndex(idx)
                elif hasattr(widget, 'setValue'):
                    widget.setValue(val)

        layout.addWidget(group)

    def _on_filter_palette(self, text: str) -> None:
        """Filter the processor palette by search text.

        Matches against processor name, category, and modalities so
        users can type e.g. "SAR" or "contrast" to find relevant processors.
        """
        self._palette_list.clear()
        query = text.lower()
        for name in self._all_processor_names:
            if query in name.lower():
                self._palette_list.addItem(QListWidgetItem(name))
                continue
            # Also match against processor tags
            proc_cls = self._processors.get(name)
            if proc_cls is not None:
                tags = get_processor_tags(proc_cls)
                cat = (tags.get('category').value if tags.get('category') else '').lower()
                mods = ' '.join(m.value for m in tags.get('modalities', ())).lower()
                if query in cat or query in mods:
                    self._palette_list.addItem(QListWidgetItem(name))

    def _on_chip_selected(self, index: int) -> None:
        """Handle chip selector change."""
        self._selected_chip_index = index
        self._schedule_preview()

    def _schedule_preview(self) -> None:
        """Debounced preview update."""
        if not self._preview_running:
            self._preview_timer.start()

    def _update_preview(self) -> None:
        """Run the current pipeline on the selected chip and display result."""
        if self._preview_running:
            return
        self._preview_running = True
        self._error_label.setText("")

        if self._chip_set is None or len(self._chip_set) == 0:
            self._preview_running = False
            return

        chip = self._chip_set[self._selected_chip_index]
        source = chip.image_data.copy()

        # Show before
        self._before_image.set_array(source)

        if not self._workflow.steps:
            self._after_image.set_array(source)
            self._preview_running = False
            return

        # Run pipeline
        result = source.copy()
        gpu_used = False
        try:
            for step in self._workflow.steps:
                proc_class = self._processors.get(step.processor_name)
                if proc_class is None:
                    continue
                proc = proc_class()
                # Check if GPU was used
                is_compatible = getattr(proc_class, '__gpu_compatible__', True)
                if is_compatible and self._gpu.gpu_available:
                    gpu_used = True
                result = self._gpu.apply_transform(proc, result, **step.params)

            # Update GPU status based on actual usage
            self._update_gpu_status_display(active=gpu_used)

        except Exception as e:
            logger.error("Preview pipeline failed: %s", e)
            self._error_label.setText(f"Error: {e}")
            self._after_image.set_array(source)
            self._preview_running = False
            self._update_gpu_status_display(active=False)
            return

        self._after_image.set_array(result)
        self._preview_running = False

    def _on_emit(self) -> None:
        """Emit the current workflow as a ProcessingPipeline signal."""
        self.Outputs.pipeline.send(ProcessingPipelineSignal(self._workflow))

    # -----------------------------------------------------------------------
    # Undo/Redo Support (Command Pattern)
    # -----------------------------------------------------------------------

    def _record_command(self, command: Dict[str, Any]) -> None:
        """Record a workflow modification command for undo.

        Parameters
        ----------
        command : dict
            Command dict with 'action' key and action-specific data.
            Actions: 'add_step', 'remove_step', 'move_step', 'modify_param'
        """
        self._undo_stack.append(command)
        self._redo_stack.clear()  # Clear redo on new action
        self._update_undo_redo_buttons()

    def _update_undo_redo_buttons(self) -> None:
        """Enable/disable undo/redo buttons based on stack state."""
        self._btn_undo.setEnabled(len(self._undo_stack) > 0)
        self._btn_redo.setEnabled(len(self._redo_stack) > 0)

    def _on_undo(self) -> None:
        """Undo the last workflow modification."""
        if not self._undo_stack:
            return

        command = self._undo_stack.pop()
        action = command['action']

        if action == 'add_step':
            # Undo: remove the step that was added
            idx = command['index']
            step = self._workflow.steps[idx]
            self._workflow.remove_step(idx)
            # Record reverse command for redo
            self._redo_stack.append({
                'action': 'remove_step',
                'index': idx,
                'step': step,
            })

        elif action == 'remove_step':
            # Undo: re-add the step that was removed
            idx = command['index']
            step = command['step']
            self._workflow.steps.insert(idx, step)
            self._redo_stack.append({
                'action': 'add_step',
                'index': idx,
                'step': step,
            })

        elif action == 'move_step':
            # Undo: move back
            from_idx = command['from_index']
            to_idx = command['to_index']
            self._workflow.move_step(to_idx, from_idx)
            self._redo_stack.append({
                'action': 'move_step',
                'from_index': to_idx,
                'to_index': from_idx,
            })

        self._rebuild_steps_list()
        self._update_undo_redo_buttons()
        self._schedule_preview()

    def _on_redo(self) -> None:
        """Redo a previously undone workflow modification."""
        if not self._redo_stack:
            return

        command = self._redo_stack.pop()
        action = command['action']

        if action == 'add_step':
            # Redo: re-add the step
            idx = command['index']
            step = command['step']
            self._workflow.steps.insert(idx, step)
            self._undo_stack.append({
                'action': 'add_step',
                'index': idx,
                'step': step,
            })

        elif action == 'remove_step':
            # Redo: remove the step again
            idx = command['index']
            step = self._workflow.steps[idx]
            self._workflow.remove_step(idx)
            self._undo_stack.append({
                'action': 'remove_step',
                'index': idx,
                'step': step,
            })

        elif action == 'move_step':
            # Redo: move forward
            from_idx = command['from_index']
            to_idx = command['to_index']
            self._workflow.move_step(from_idx, to_idx)
            self._undo_stack.append({
                'action': 'move_step',
                'from_index': from_idx,
                'to_index': to_idx,
            })

        self._rebuild_steps_list()
        self._update_undo_redo_buttons()
        self._schedule_preview()

