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
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer

# GRDK internal
from grdl_rt.execution.discovery import (
    discover_processors,
    get_processor_tags,
    filter_processors_for_connection,
)
from grdl_rt.execution.gpu import GpuBackend
from grdl_rt.execution.graph import types_compatible, WorkflowGraph
from grdl_rt.execution.workflow import ProcessingStep, TapOutStepDef, WorkflowDefinition
from grdk.widgets._signals import (
    ChipSetSignal,
    ProcessingPipelineSignal,
    get_modality_hint,
)


def _array_to_pixmap(arr: np.ndarray, size: int = 256) -> QPixmap:
    """Convert numpy array to QPixmap for preview display."""
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    arr = arr.astype(np.float64)
    vmin, vmax = np.nanmin(arr), np.nanmax(arr)
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin) * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        h, w = arr.shape
        qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = np.ascontiguousarray(arr[:, :, :3])
        h, w, _ = arr.shape
        qimg = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    else:
        band = arr[:, :, 0] if arr.ndim == 3 else arr
        h, w = band.shape
        qimg = QImage(band.data, w, h, w, QImage.Format.Format_Grayscale8)

    pixmap = QPixmap.fromImage(qimg)
    return pixmap.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio)


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
        self._graph = WorkflowGraph()
        self._chip_set: Optional[Any] = None
        self._detected_modality: Optional[Any] = None  # ImageModality | None
        self._selected_chip_index = 0
        self._gpu = GpuBackend()
        self._step_param_controls: List[Dict[str, Any]] = []
        self._preview_running = False  # Execution lock
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(50)  # 50ms debounce
        self._preview_timer.timeout.connect(self._update_preview)

        # --- Control area ---
        box = gui.vBox(self.controlArea, "Workflow")
        self._workflow_name_label = QLabel("Steps: 0", self)
        box.layout().addWidget(self._workflow_name_label)

        btn_emit = QPushButton("Emit Pipeline", self)
        btn_emit.clicked.connect(self._on_emit)
        box.layout().addWidget(btn_emit)

        self._compat_label = QLabel("", self)
        self._compat_label.setStyleSheet("color: orange; font-size: 10px;")
        self._compat_label.setWordWrap(True)
        box.layout().addWidget(self._compat_label)

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
        btn_tapout = QPushButton("Tapout…")
        btn_tapout.setToolTip("Insert a save-to-disk tapout after the selected step")
        btn_tapout.clicked.connect(self._on_mark_tapout)
        step_btns.layout().addWidget(btn_up)
        step_btns.layout().addWidget(btn_down)
        step_btns.layout().addWidget(btn_remove)
        step_btns.layout().addWidget(btn_tapout)
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
        self._before_image = QLabel()
        self._before_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._before_image.setMinimumSize(128, 128)

        self._after_label = QLabel("After Pipeline")
        self._after_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._after_image = QLabel()
        self._after_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
        self._detected_modality = get_modality_hint(signal)
        self._selected_chip_index = 0
        self._chip_spinbox.setMaximum(max(0, len(signal.chip_set) - 1))
        self._chip_spinbox.setValue(0)
        self._refresh_palette_compat()
        self._schedule_preview()

    def _on_add_step(self) -> None:
        """Add selected palette item as a workflow step, with type validation."""
        item = self._palette_list.currentItem()
        if item is None:
            return

        proc_name = item.text()
        proc_class = self._processors.get(proc_name)
        version = getattr(proc_class, '__processor_version__', '') if proc_class else ''

        # WorkflowGraph type validation
        tags = get_processor_tags(proc_class) if proc_class else {}
        input_type = tags.get('input_type')
        output_type = tags.get('output_type')
        input_type_str = (input_type.value if hasattr(input_type, 'value') else str(input_type)) if input_type else None
        output_type_str = (output_type.value if hasattr(output_type, 'value') else str(output_type)) if output_type else None

        step_id = f"step_{len(self._workflow.steps) + 1}_{proc_name}"
        self._graph.add_node(step_id, proc_name, input_type_str, output_type_str)

        if len(self._workflow.steps) > 0:
            prev_step = self._workflow.steps[-1]
            prev_id = f"step_{len(self._workflow.steps)}_{prev_step.processor_name}"
            try:
                self._graph.connect(prev_id, step_id)
                self._compat_label.setText("")
            except ValueError as exc:
                self._compat_label.setText(f"Type mismatch: {exc}")
                # Roll back the node we just added
                try:
                    self._graph.remove_node(step_id)
                except Exception:
                    pass
                return

        step = ProcessingStep(
            processor_name=proc_name,
            processor_version=version,
        )
        self._workflow.add_step(step)
        self._steps_list.addItem(f"{len(self._workflow.steps)}. {proc_name}")
        self._workflow_name_label.setText(f"Steps: {len(self._workflow.steps)}")
        self._refresh_palette_compat()
        self._schedule_preview()

    def _on_mark_tapout(self) -> None:
        """Insert a TapOutStepDef after the selected workflow step."""
        from PyQt6.QtWidgets import QFileDialog

        row = self._steps_list.currentRow()
        if row < 0:
            # No step selected — append after all steps
            insert_after = len(self._workflow.steps) - 1
        else:
            insert_after = row

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Tapout Output Path",
            "",
            "NumPy (*.npy);;GeoTIFF (*.tif *.tiff);;All Files (*)",
        )
        if not path:
            return

        tapout = TapOutStepDef(
            path=path,
            id=f"tapout_{insert_after + 1}",
        )
        # Insert after the selected step
        self._workflow.steps.insert(insert_after + 1, tapout)
        self._rebuild_steps_list()
        self._workflow_name_label.setText(f"Steps: {len(self._workflow.steps)}")

    def _on_remove_step(self) -> None:
        """Remove the selected workflow step."""
        row = self._steps_list.currentRow()
        if row < 0 or row >= len(self._workflow.steps):
            return

        self._workflow.remove_step(row)
        self._rebuild_steps_list()
        self._refresh_palette_compat()
        self._schedule_preview()

    def _on_move_up(self) -> None:
        """Move the selected step up."""
        row = self._steps_list.currentRow()
        if row <= 0:
            return

        self._workflow.move_step(row, row - 1)
        self._rebuild_steps_list()
        self._steps_list.setCurrentRow(row - 1)
        self._schedule_preview()

    def _on_move_down(self) -> None:
        """Move the selected step down."""
        row = self._steps_list.currentRow()
        if row < 0 or row >= len(self._workflow.steps) - 1:
            return

        self._workflow.move_step(row, row + 1)
        self._rebuild_steps_list()
        self._steps_list.setCurrentRow(row + 1)
        self._schedule_preview()

    def _rebuild_steps_list(self) -> None:
        """Refresh the steps list widget from the workflow."""
        self._steps_list.clear()
        for i, step in enumerate(self._workflow.steps):
            if isinstance(step, TapOutStepDef):
                label = f"  ⬇ tapout → {step.path}"
                item = QListWidgetItem(label)
                item.setForeground(QColor(80, 160, 80))
            else:
                item = QListWidgetItem(f"{i + 1}. {step.processor_name}")
            self._steps_list.addItem(item)
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
        self._before_image.setPixmap(_array_to_pixmap(source))

        if not self._workflow.steps:
            self._after_image.setPixmap(_array_to_pixmap(source))
            self._preview_running = False
            return

        # Run pipeline
        result = source.copy()
        try:
            for step in self._workflow.steps:
                proc_class = self._processors.get(step.processor_name)
                if proc_class is None:
                    continue
                proc = proc_class()
                result = self._gpu.apply_transform(proc, result, **step.params)
        except Exception as e:
            logger.error("Preview pipeline failed: %s", e)
            self._error_label.setText(f"Error: {e}")
            self._after_image.setPixmap(_array_to_pixmap(source))
            self._preview_running = False
            return

        self._after_image.setPixmap(_array_to_pixmap(result))
        self._preview_running = False

    def _on_emit(self) -> None:
        """Emit the current workflow as a ProcessingPipeline signal."""
        output_type = self._get_last_output_type()
        self.Outputs.pipeline.send(
            ProcessingPipelineSignal(self._workflow, output_port_type=output_type)
        )

    def _get_last_output_type(self) -> Optional[str]:
        """Return the DataPortType string produced by the last workflow step.

        Reads ``__processor_tags__['output_type']`` from the last step's
        processor class.  Falls back to reading ``input_type`` as a proxy
        (many transforms are type-preserving).  Returns ``None`` if the
        workflow is empty or the type is undeclared.
        """
        if not self._workflow.steps:
            return None
        last_step = self._workflow.steps[-1]
        proc_class = self._processors.get(last_step.processor_name)
        if proc_class is None:
            return None
        tags = get_processor_tags(proc_class)
        out = tags.get('output_type')
        if out is not None:
            return out.value if hasattr(out, 'value') else str(out)
        # Type-preserving processors: output == input
        inp = tags.get('input_type')
        if inp is not None:
            return inp.value if hasattr(inp, 'value') else str(inp)
        return None

    def _refresh_palette_compat(self) -> None:
        """Recolour palette items incompatible with current last-step output or modality.

        - Port-type incompatible items are greyed out.
        - Modality-incompatible items (where the chip modality is known and the
          processor declares non-empty modalities that don't include it) are
          hidden entirely.
        - Items with no ``input_type`` or empty ``modalities`` (implicit ANY)
          remain enabled.
        """
        last_out = self._get_last_output_type()
        self._compat_label.setText("" if last_out is None else f"Last output: {last_out}")

        modality_str: Optional[str] = None
        if self._detected_modality is not None:
            m = self._detected_modality
            modality_str = m.value if hasattr(m, 'value') else str(m)

        # Build a modality-filtered compatible set for quick lookup
        compat_names: Optional[set] = None
        if modality_str is not None:
            try:
                compat_map = filter_processors_for_connection(last_out, modality_str)
                compat_names = set(compat_map.keys())
            except Exception:
                compat_names = None

        for i in range(self._palette_list.count()):
            item = self._palette_list.item(i)
            if item is None:
                continue
            proc_name = item.text()
            proc_class = self._processors.get(proc_name)
            if proc_class is None:
                continue
            tags = get_processor_tags(proc_class)

            # Port-type compatibility (affects colour)
            proc_in = tags.get('input_type')
            proc_in_str = (proc_in.value if hasattr(proc_in, 'value') else str(proc_in)) if proc_in else None
            port_ok = types_compatible(last_out, proc_in_str)

            # Modality compatibility (affects visibility)
            if compat_names is not None:
                modality_ok = proc_name in compat_names
            else:
                modality_ok = True

            item.setHidden(not modality_ok)
            item.setForeground(
                item.foreground() if port_ok else QColor(150, 150, 150)
            )
