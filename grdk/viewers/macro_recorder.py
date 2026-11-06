# -*- coding: utf-8 -*-
"""
Macro Recorder - In-memory image processing workflow builder and executor.

Records user image processing actions (filters, enhancements, band math,
decompositions) into an in-memory `grdl_rt.execution.workflow.WorkflowDefinition`
DAG. Enables interactive execution on the active image via `grdl-runtime`,
exporting to YAML or Python DSL, registration in the catalog, and direct
interoperability with GRDK Orange Canvas.

Author
------
Steven Siebert

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.
"""

from __future__ import annotations

import logging
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Union
import yaml

import numpy as np

# grdl-runtime execution engine
from grdl_rt.execution.workflow import ProcessingStep, WorkflowDefinition, WorkflowState
from grdl_rt.execution.dag_executor import DAGExecutor
from grdl_rt.execution.dsl import DslCompiler
from grdl_rt.execution.result import WorkflowResult

try:
    from PyQt6.QtCore import Qt, pyqtSignal as Signal
    from PyQt6.QtWidgets import (
        QDialog,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QMessageBox,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False

_log = logging.getLogger("grdk.viewers.macro")


class WorkflowMacroRecorder:
    """Manages an in-memory grdl-runtime WorkflowDefinition.

    Allows recording steps, executing them on numpy image arrays,
    serializing to YAML, and handing off to Orange Canvas.
    """

    def __init__(
        self,
        name: str = "Image Exploitation Macro",
        version: str = "1.0.0",
        description: str = "Recorded image exploitation processing chain",
    ) -> None:
        self._name = name
        self._version = version
        self._description = description
        self._workflow = WorkflowDefinition(
            name=name,
            version=version,
            description=description,
            steps=[],
            state=WorkflowState.DRAFT,
        )

    @property
    def workflow(self) -> WorkflowDefinition:
        """Return the underlying WorkflowDefinition."""
        return self._workflow

    @property
    def steps(self) -> list[ProcessingStep]:
        """Return the list of processing steps."""
        return self._workflow.steps

    def step_count(self) -> int:
        """Return the number of steps in the macro."""
        return len(self._workflow.steps)

    def is_empty(self) -> bool:
        """Check if macro contains any steps."""
        return len(self._workflow.steps) == 0

    def clear(self) -> None:
        """Reset macro steps."""
        self._workflow.steps.clear()

    def add_step(
        self,
        processor_name_or_class: str | type,
        params: dict[str, Any] | None = None,
        step_id: str | None = None,
    ) -> ProcessingStep:
        """Append an image processing step to the in-memory macro DAG.

        Parameters
        ----------
        processor_name_or_class : str or type
            FQN or class of the image processor.
        params : dict, optional
            Tunable parameter values.
        step_id : str, optional
            Identifier for this step; auto-generated if omitted.

        Returns
        -------
        ProcessingStep
            The created step.
        """
        if isinstance(processor_name_or_class, type):
            proc_name = f"{processor_name_or_class.__module__}.{processor_name_or_class.__qualname__}"
            short_name = processor_name_or_class.__name__
        else:
            proc_name = str(processor_name_or_class)
            short_name = proc_name.split(".")[-1]

        if not step_id:
            step_id = f"{short_name}_{len(self._workflow.steps) + 1}"

        step = ProcessingStep(
            id=step_id,
            processor_name=proc_name,
            params=dict(params or {}),
        )
        self._workflow.add_step(step)
        _log.info("Macro step added: %s (%s)", step.id, step.processor_name)
        return step

    def remove_step(self, index: int) -> ProcessingStep:
        """Remove a step by index and maintain DAG integrity."""
        return self._workflow.remove_step(index)

    def execute(
        self,
        source_array: np.ndarray,
        progress_callback: Callable[[float], None] | None = None,
    ) -> WorkflowResult:
        """Execute the in-memory macro on an image array using DAGExecutor.

        Parameters
        ----------
        source_array : np.ndarray
            Input image chip or full raster.
        progress_callback : callable, optional
            Progress update hook.

        Returns
        -------
        WorkflowResult
            Execution result containing the transformed numpy array.
        """
        if self.is_empty():
            raise ValueError("Cannot execute empty macro: no steps recorded.")

        executor = DAGExecutor(self._workflow)
        res = executor.execute(source_array, progress_callback=progress_callback)
        return res

    def to_yaml(self) -> str:
        """Serialize macro to YAML string."""
        d = self._workflow.to_dict()
        return yaml.dump(d, sort_keys=False)

    def to_python(self) -> str:
        """Compile macro to Python DSL code."""
        compiler = DslCompiler()
        return compiler.to_python(self._workflow)

    def save_to_yaml_file(self, filepath: str | Path) -> Path:
        """Save macro to a YAML workflow file on disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml(), encoding="utf-8")
        _log.info("Macro saved to YAML: %s", path)
        return path

    def load_from_yaml(self, source: str | Path | dict) -> WorkflowDefinition:
        """Load a workflow from YAML string, file, or dict into this macro."""
        if isinstance(source, (str, Path)) and Path(str(source)).exists():
            data = yaml.safe_load(Path(str(source)).read_text(encoding="utf-8"))
        elif isinstance(source, str):
            data = yaml.safe_load(source)
        elif isinstance(source, dict):
            data = source
        else:
            raise TypeError(f"Invalid workflow source: {type(source)}")

        self._workflow = WorkflowDefinition.from_dict(data)
        _log.info("Loaded macro workflow: %s with %d steps", self._workflow.name, len(self._workflow.steps))
        return self._workflow

    def register_in_catalog(self, catalog: Any = None) -> Any:
        """Register this macro into the local catalog as a grdk_workflow artifact.

        Makes the macro immediately accessible to Orange Canvas and Catalog Runner.
        """
        from grdl_rt.catalog.models import Artifact
        if catalog is None:
            from grdk.catalog.manager import CatalogManager
            mgr = CatalogManager()
            catalog = mgr.get_catalog()

        compiler = DslCompiler()
        yaml_text = self.to_yaml()
        python_dsl = compiler.to_python(self._workflow)

        artifact = Artifact(
            name=self._workflow.name,
            version=self._workflow.version,
            artifact_type="grdk_workflow",
            description=self._workflow.description,
            yaml_definition=yaml_text,
            python_dsl=python_dsl,
            tags=self._workflow.tags.to_dict(),
        )
        catalog.add_artifact(artifact)
        _log.info("Registered macro in catalog: %s v%s", artifact.name, artifact.version)
        return artifact

    def open_in_orange_canvas(self, save_path: str | Path | None = None) -> subprocess.Popen:
        """Export workflow and launch GRDK Orange Canvas to edit/refine as a graph.

        Parameters
        ----------
        save_path : str or Path, optional
            File path to save the workflow YAML before launching Orange.
        """
        if save_path is None:
            import tempfile
            save_path = Path(tempfile.gettempdir()) / f"grdk_macro_{self._workflow.name.lower().replace(' ', '_')}.yaml"

        saved = self.save_to_yaml_file(save_path)
        # Also ensure registered in catalog so Orange's OWWorkflowManager sees it
        try:
            self.register_in_catalog()
        except Exception as e:
            _log.warning("Could not register in catalog during launch: %s", e)

        # Launch Orange Canvas
        cmd = [sys.executable, "-m", "grdk._launcher"]
        _log.info("Launching Orange Canvas: %s (workflow at %s)", cmd, saved)
        proc = subprocess.Popen(cmd)
        return proc


if _QT_AVAILABLE:

    class WorkflowMacroDialog(QDialog):
        """Dialog to inspect, execute, export, and launch Orange Canvas for the macro."""

        macro_executed = Signal(np.ndarray, str)  # (result_array, label)

        def __init__(
            self,
            recorder: WorkflowMacroRecorder,
            source_provider: Callable[[], Optional[np.ndarray]],
            parent: Optional[QWidget] = None,
        ) -> None:
            super().__init__(parent)
            self.setWindowTitle("Processing Workflow (Macro)")
            self.resize(760, 440)

            self._recorder = recorder
            self._source_provider = source_provider

            main_layout = QVBoxLayout(self)

            # Table of Steps
            table_group = QGroupBox(f"Workflow Steps ({self._recorder.workflow.name})", self)
            table_layout = QVBoxLayout(table_group)

            self._table = QTableWidget(0, 4, self)
            self._table.setHorizontalHeaderLabels(["Step ID", "Processor", "Parameters", "Depends On"])
            self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
            table_layout.addWidget(self._table)
            main_layout.addWidget(table_group)

            # Bottom Status & Action Buttons
            status_row = QHBoxLayout()
            self._status_label = QLabel(self)
            status_row.addWidget(self._status_label)
            status_row.addStretch()
            main_layout.addLayout(status_row)

            btn_row = QHBoxLayout()

            self._run_btn = QPushButton("Run Macro on Active Image", self)
            self._run_btn.setStyleSheet("font-weight: bold; padding: 6px 12px;")
            self._run_btn.clicked.connect(self._on_run)
            btn_row.addWidget(self._run_btn)

            btn_remove = QPushButton("Remove Step", self)
            btn_remove.clicked.connect(self._on_remove)
            btn_row.addWidget(btn_remove)

            btn_clear = QPushButton("Clear All", self)
            btn_clear.clicked.connect(self._on_clear)
            btn_row.addWidget(btn_clear)

            btn_export = QPushButton("Export YAML...", self)
            btn_export.clicked.connect(self._on_export_yaml)
            btn_row.addWidget(btn_export)

            btn_import = QPushButton("Load YAML...", self)
            btn_import.clicked.connect(self._on_import_yaml)
            btn_row.addWidget(btn_import)

            btn_orange = QPushButton("Open in Orange Canvas", self)
            btn_orange.setStyleSheet("background-color: #ff8c00; color: white; font-weight: bold; padding: 6px 12px;")
            btn_orange.clicked.connect(self._on_open_orange)
            btn_row.addWidget(btn_orange)

            btn_close = QPushButton("Close", self)
            btn_close.clicked.connect(self.accept)
            btn_row.addWidget(btn_close)

            main_layout.addLayout(btn_row)

            self._populate_table()

        def _populate_table(self) -> None:
            self._table.setRowCount(0)
            steps = self._recorder.steps
            self._table.setRowCount(len(steps))

            for r, step in enumerate(steps):
                self._table.setItem(r, 0, QTableWidgetItem(step.id or f"step_{r}"))
                self._table.setItem(r, 1, QTableWidgetItem(step.processor_name))
                self._table.setItem(r, 2, QTableWidgetItem(str(step.params)))
                deps = ", ".join(step.depends_on) if step.depends_on else "(initial)"
                self._table.setItem(r, 3, QTableWidgetItem(deps))

            count = len(steps)
            self._status_label.setText(f"{count} step{'s' if count != 1 else ''} recorded.")
            self._run_btn.setEnabled(count > 0)

        def _on_run(self) -> None:
            source = self._source_provider()
            if source is None:
                QMessageBox.warning(self, "No Image", "Please load an image before running the macro.")
                return

            if self._recorder.is_empty():
                QMessageBox.information(self, "Empty Macro", "No processing steps in macro to run.")
                return

            self._status_label.setText("Executing macro via grdl-runtime...")
            self._run_btn.setEnabled(False)
            try:
                result = self._recorder.execute(source)
                self.macro_executed.emit(result.result, f"Macro ({self._recorder.workflow.name})")
                self._status_label.setText("Macro executed successfully.")
                QMessageBox.information(
                    self,
                    "Execution Success",
                    f"Workflow macro executed successfully.\nOutput shape: {result.result.shape}\nSteps: {len(result.step_results)}"
                )
            except Exception as e:
                _log.error("Macro execution failed: %s", e, exc_info=True)
                QMessageBox.critical(self, "Execution Failed", f"Failed to execute workflow macro:\n{e}")
                self._status_label.setText("Execution failed.")
            finally:
                self._run_btn.setEnabled(True)

        def _on_remove(self) -> None:
            row = self._table.currentRow()
            if row >= 0:
                self._recorder.remove_step(row)
                self._populate_table()

        def _on_clear(self) -> None:
            if self._recorder.is_empty():
                return
            reply = QMessageBox.question(
                self, "Clear Macro", "Are you sure you want to clear all steps in this macro?"
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._recorder.clear()
                self._populate_table()

        def _on_export_yaml(self) -> None:
            if self._recorder.is_empty():
                QMessageBox.warning(self, "Empty Macro", "No steps to export.")
                return
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Save Workflow Macro YAML", f"{self._recorder.workflow.name.lower().replace(' ', '_')}.yaml", "YAML Files (*.yaml *.yml)"
            )
            if filepath:
                self._recorder.save_to_yaml_file(filepath)
                QMessageBox.information(self, "Exported", f"Saved workflow macro to:\n{filepath}")

        def _on_import_yaml(self) -> None:
            filepath, _ = QFileDialog.getOpenFileName(
                self, "Open Workflow YAML", "", "YAML Files (*.yaml *.yml);;All Files (*)"
            )
            if filepath:
                try:
                    self._recorder.load_from_yaml(filepath)
                    self._populate_table()
                    QMessageBox.information(self, "Loaded", f"Loaded workflow '{self._recorder.workflow.name}'.")
                except Exception as e:
                    QMessageBox.critical(self, "Load Error", f"Failed to load workflow YAML:\n{e}")

        def _on_open_orange(self) -> None:
            try:
                proc = self._recorder.open_in_orange_canvas()
                QMessageBox.information(
                    self,
                    "Orange Canvas Launched",
                    f"Launched GRDK Orange Canvas.\nYour workflow '{self._recorder.workflow.name}' was registered and saved."
                )
            except Exception as e:
                QMessageBox.critical(self, "Launch Error", f"Could not launch Orange Canvas:\n{e}")
