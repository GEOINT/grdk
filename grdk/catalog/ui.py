# -*- coding: utf-8 -*-
"""
Catalog Management & Execution GUI Dialogs for GRDK Viewer.

Contains:
1. ``CatalogManagementDialog``: Add/remove/enable catalog backends and set preferences.
2. ``CatalogRunnerDialog``: Search, inspect, configure, and execute catalog components.
3. ``CatalogSyncWorker``: Background thread for non-blocking cache validation.

Author
------
Claude Code / Steven Siebert

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal as Signal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from grdk.catalog.manager import CatalogEntry, CatalogManager, CatalogPreferencesManager

_log = logging.getLogger("grdk.catalog.ui")


class CatalogSyncWorker(QThread):
    """Background worker that tests for catalog changes and refreshes cache."""
    sync_finished = Signal(bool, int)  # (was_updated, total_items)

    def __init__(self, manager: CatalogManager) -> None:
        super().__init__()
        self._manager = manager

    def run(self) -> None:
        try:
            if self._manager.is_cache_stale():
                artifacts = self._manager.refresh_cache(force=True)
                self.sync_finished.emit(True, len(artifacts))
            else:
                artifacts = self._manager.load_cached_artifacts()
                self.sync_finished.emit(False, len(artifacts))
        except Exception as e:
            _log.warning("Background catalog sync failed: %s", e)
            self.sync_finished.emit(False, 0)


class CatalogManagementDialog(QDialog):
    """Dialog for configuring multiple catalog sources and execution preferences."""
    catalogs_changed = Signal()

    def __init__(self, manager: CatalogManager, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Catalog Management")
        self.resize(720, 480)
        self._manager = manager

        layout = QVBoxLayout(self)

        # Catalogs table
        table_group = QGroupBox("Registered Catalogs")
        table_layout = QVBoxLayout(table_group)

        self._table = QTableWidget(0, 5, self)
        self._table.setHorizontalHeaderLabels(["Enabled", "Name", "Type", "Path", "Items"])
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table_layout.addWidget(self._table)

        # Buttons under table
        btn_layout = QHBoxLayout()
        self._add_sqlite_btn = QPushButton("Add SQLite (.db)...", self)
        self._add_sqlite_btn.clicked.connect(self._on_add_sqlite)
        btn_layout.addWidget(self._add_sqlite_btn)

        self._add_yaml_btn = QPushButton("Add YAML (.yaml)...", self)
        self._add_yaml_btn.clicked.connect(self._on_add_yaml)
        btn_layout.addWidget(self._add_yaml_btn)

        self._remove_btn = QPushButton("Remove Selected", self)
        self._remove_btn.clicked.connect(self._on_remove_selected)
        btn_layout.addWidget(self._remove_btn)

        self._reload_btn = QPushButton("Reload All Now", self)
        self._reload_btn.clicked.connect(self._on_force_reload)
        btn_layout.addWidget(self._reload_btn)

        table_layout.addLayout(btn_layout)
        layout.addWidget(table_group)

        # Preferences Group
        pref_group = QGroupBox("Execution & Output Destination")
        pref_layout = QVBoxLayout(pref_group)

        self._radio_replace = QRadioButton("Replace active pane (clean up previous memory)", self)
        self._radio_dual = QRadioButton("Open result in right pane (Dual Mode comparison)", self)

        curr_mode = CatalogPreferencesManager.get_output_destination_mode()
        if curr_mode == "dual":
            self._radio_dual.setChecked(True)
        else:
            self._radio_replace.setChecked(True)

        self._radio_replace.toggled.connect(self._on_pref_mode_changed)
        pref_layout.addWidget(self._radio_replace)
        pref_layout.addWidget(self._radio_dual)
        layout.addWidget(pref_group)

        # Dialog Buttons
        bbox = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        bbox.rejected.connect(self.accept)
        layout.addWidget(bbox)

        self._populate_table()

    def _populate_table(self) -> None:
        self._table.setRowCount(0)
        catalogs = self._manager.catalogs
        self._table.setRowCount(len(catalogs))

        for row, cat in enumerate(catalogs):
            # Enabled checkbox
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            chk_item.setCheckState(Qt.CheckState.Checked if cat.enabled else Qt.CheckState.Unchecked)
            self._table.setItem(row, 0, chk_item)

            self._table.setItem(row, 1, QTableWidgetItem(cat.name))
            self._table.setItem(row, 2, QTableWidgetItem(cat.backend_type.upper()))
            self._table.setItem(row, 3, QTableWidgetItem(cat.path))
            self._table.setItem(row, 4, QTableWidgetItem(str(cat.item_count)))

        self._table.itemChanged.connect(self._on_item_changed)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == 0:
            row = item.row()
            enabled = (item.checkState() == Qt.CheckState.Checked)
            self._manager.toggle_catalog(row, enabled)
            self.catalogs_changed.emit()

    def _on_add_sqlite(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SQLite Catalog File", "", "SQLite Databases (*.db *.sqlite);;All Files (*)"
        )
        if path:
            name = Path(path).stem.replace("_", " ").title()
            self._manager.add_catalog(name=name, path=path, backend_type="sqlite")
            self._table.itemChanged.disconnect(self._on_item_changed)
            self._populate_table()
            self.catalogs_changed.emit()

    def _on_add_yaml(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YAML Catalog File", "", "YAML Catalogs (*.yaml *.yml);;All Files (*)"
        )
        if path:
            name = Path(path).stem.replace("_", " ").title()
            self._manager.add_catalog(name=name, path=path, backend_type="yaml")
            self._table.itemChanged.disconnect(self._on_item_changed)
            self._populate_table()
            self.catalogs_changed.emit()

    def _on_remove_selected(self) -> None:
        row = self._table.currentRow()
        if row >= 0:
            self._table.itemChanged.disconnect(self._on_item_changed)
            self._manager.remove_catalog(row)
            self._populate_table()
            self.catalogs_changed.emit()

    def _on_force_reload(self) -> None:
        artifacts = self._manager.refresh_cache(force=True)
        self._table.itemChanged.disconnect(self._on_item_changed)
        self._populate_table()
        self.catalogs_changed.emit()
        QMessageBox.information(
            self, "Catalogs Reloaded", f"Successfully reloaded all catalogs. Found {len(artifacts)} total artifacts."
        )

    def _on_pref_mode_changed(self) -> None:
        mode = "dual" if self._radio_dual.isChecked() else "replace"
        CatalogPreferencesManager.set_output_destination_mode(mode)


class CatalogRunnerDialog(QDialog):
    """Browse, inspect, configure, and execute catalog components and workflows."""

    def __init__(
        self,
        manager: CatalogManager,
        source_array_provider: Callable[[], Optional[Any]],
        execution_callback: Callable[[Any, str], None],
        macro_recorder: Optional[Any] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Run from Catalog")
        self.resize(880, 560)

        self._manager = manager
        self._source_array_provider = source_array_provider
        self._execution_callback = execution_callback
        self._macro_recorder = macro_recorder

        self._artifacts: list[Any] = []
        self._param_inputs: dict[str, Any] = {}

        main_layout = QVBoxLayout(self)

        # Top Filter Bar
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Filter Type:"))
        self._type_combo = QComboBox(self)
        self._type_combo.addItems(["All", "Processors (grdl_processor)", "Workflows (grdk_workflow)"])
        self._type_combo.currentIndexChanged.connect(self._apply_filter)
        top_bar.addWidget(self._type_combo)

        top_bar.addWidget(QLabel("Search:"))
        self._search_input = QLineEdit(self)
        self._search_input.setPlaceholderText("Filter by name, description, tags...")
        self._search_input.textChanged.connect(self._apply_filter)
        top_bar.addWidget(self._search_input)

        btn_reload = QPushButton("Reload", self)
        btn_reload.clicked.connect(self._reload_artifacts)
        top_bar.addWidget(btn_reload)

        main_layout.addLayout(top_bar)

        # Splitter: Left List, Right Details & Parameters
        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # Left list
        self._list = QListWidget(self)
        self._list.currentRowChanged.connect(self._on_artifact_selected)
        splitter.addWidget(self._list)

        # Right pane
        right_widget = QWidget(self)
        right_layout = QVBoxLayout(right_widget)

        self._details_view = QTextEdit(self)
        self._details_view.setReadOnly(True)
        self._details_view.setMaximumHeight(160)
        right_layout.addWidget(self._details_view)

        # Parameter Form Area
        self._param_group = QGroupBox("Parameters", self)
        self._param_layout = QFormLayout(self._param_group)
        self._param_scroll = QScrollArea(self)
        self._param_scroll.setWidgetResizable(True)
        self._param_scroll.setWidget(self._param_group)
        right_layout.addWidget(self._param_scroll)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter)

        # Action Buttons
        btn_box = QHBoxLayout()
        self._status_label = QLabel(self)
        btn_box.addWidget(self._status_label)
        btn_box.addStretch()

        self._run_btn = QPushButton("Run on Active Image", self)
        self._run_btn.setStyleSheet("font-weight: bold; padding: 6px 14px;")
        self._run_btn.clicked.connect(self._on_run)
        btn_box.addWidget(self._run_btn)

        close_btn = QPushButton("Close", self)
        close_btn.clicked.connect(self.accept)
        btn_box.addWidget(close_btn)

        main_layout.addLayout(btn_box)

        self._load_artifacts()

    def _reload_artifacts(self) -> None:
        self._manager.refresh_cache(force=True)
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        self._artifacts = self._manager.load_cached_artifacts()
        self._apply_filter()

    def _apply_filter(self) -> None:
        self._list.clear()
        query = self._search_input.text().lower().strip()
        type_idx = self._type_combo.currentIndex()

        for art in self._artifacts:
            # Type filter
            if type_idx == 1 and art.artifact_type != "grdl_processor":
                continue
            if type_idx == 2 and art.artifact_type != "grdk_workflow":
                continue

            # Query filter
            full_text = f"{art.name} {art.description or ''} {art.processor_class or ''}".lower()
            if query and query not in full_text:
                continue

            type_label = "[Proc]" if art.artifact_type == "grdl_processor" else "[Workflow]"
            item = QListWidgetItem(f"{type_label} {art.name} (v{art.version})")
            item.setData(Qt.ItemDataRole.UserRole, art)
            self._list.addItem(item)

        if self._list.count() > 0:
            self._list.setCurrentRow(0)
        else:
            self._details_view.setPlainText("No artifacts match the current filter.")
            self._clear_param_form()

    def _clear_param_form(self) -> None:
        while self._param_layout.count() > 0:
            item = self._param_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self._param_inputs.clear()

    def _on_artifact_selected(self, row: int) -> None:
        self._clear_param_form()
        if row < 0:
            return

        item = self._list.item(row)
        if not item:
            return
        art = item.data(Qt.ItemDataRole.UserRole)
        if not art:
            return

        # Render description
        details = [
            f"<b>{art.name}</b> (Version: {art.version})",
            f"Type: {art.artifact_type}",
            f"Description: {art.description or 'No description provided.'}",
        ]
        if art.processor_class:
            details.append(f"Class: <code>{art.processor_class}</code>")
        if art.author:
            details.append(f"Author: {art.author}")
        self._details_view.setHtml("<br>".join(details))

        # Build dynamic parameters
        schema = {}
        if art.param_schema:
            try:
                schema = json.loads(art.param_schema) if isinstance(art.param_schema, str) else art.param_schema
            except Exception:
                pass

        if schema and isinstance(schema, dict):
            props = schema.get("properties", {})
            for param_name, meta in props.items():
                p_type = meta.get("type", "string")
                default_val = meta.get("default", None)
                label_text = meta.get("title", param_name)

                if p_type == "integer":
                    sb = QSpinBox(self)
                    sb.setRange(meta.get("minimum", -999999), meta.get("maximum", 999999))
                    if default_val is not None:
                        sb.setValue(int(default_val))
                    self._param_inputs[param_name] = (sb, "int")
                    self._param_layout.addRow(label_text, sb)
                elif p_type == "number":
                    dsb = QDoubleSpinBox(self)
                    dsb.setRange(meta.get("minimum", -999999.0), meta.get("maximum", 999999.0))
                    dsb.setDecimals(4)
                    if default_val is not None:
                        dsb.setValue(float(default_val))
                    self._param_inputs[param_name] = (dsb, "float")
                    self._param_layout.addRow(label_text, dsb)
                elif p_type == "boolean":
                    chk = QRadioButton(self)
                    if default_val:
                        chk.setChecked(True)
                    self._param_inputs[param_name] = (chk, "bool")
                    self._param_layout.addRow(label_text, chk)
                else:
                    le = QLineEdit(self)
                    if default_val is not None:
                        le.setText(str(default_val))
                    self._param_inputs[param_name] = (le, "str")
                    self._param_layout.addRow(label_text, le)
        else:
            self._param_layout.addRow(QLabel("<i>No configurable parameters defined.</i>"))

    def _collect_parameters(self) -> dict[str, Any]:
        params = {}
        for param_name, (widget, p_type) in self._param_inputs.items():
            if p_type == "int":
                params[param_name] = widget.value()
            elif p_type == "float":
                params[param_name] = widget.value()
            elif p_type == "bool":
                params[param_name] = widget.isChecked()
            else:
                txt = widget.text().strip()
                params[param_name] = txt
        return params

    def _on_run(self) -> None:
        source_arr = self._source_array_provider()
        if source_arr is None:
            QMessageBox.warning(
                self, "No Image Loaded", "Please load an image in the viewer before executing a catalog tool."
            )
            return

        row = self._list.currentRow()
        if row < 0:
            return
        item = self._list.item(row)
        art = item.data(Qt.ItemDataRole.UserRole)
        params = self._collect_parameters()

        self._status_label.setText("Executing...")
        self._run_btn.setEnabled(False)

        try:
            if art.artifact_type == "grdl_processor":
                from grdl_rt.execution.workflow import ProcessingStep, WorkflowDefinition
                from grdl_rt.execution.executor import WorkflowExecutor

                step = ProcessingStep(
                    id=f"{art.name}_step",
                    processor_name=art.name,
                    params=params,
                )
                wf = WorkflowDefinition(
                    name=f"run_{art.name}",
                    version="1.0.0",
                    description=art.description or "",
                    steps=[step],
                )
                executor = WorkflowExecutor(wf)
                res = executor.execute(source_arr)
                result_data = res.result
            else:
                # Workflow
                import yaml
                from grdl_rt.execution.workflow import WorkflowDefinition
                from grdl_rt.execution.executor import WorkflowExecutor

                raw_def = art.yaml_definition
                if not raw_def:
                    raise ValueError(f"Workflow '{art.name}' does not contain a yaml_definition.")
                wf_dict = yaml.safe_load(raw_def)
                wf = WorkflowDefinition.from_dict(wf_dict)
                executor = WorkflowExecutor(wf)
                res = executor.execute(source_arr)
                result_data = res.result

            if self._macro_recorder is not None:
                try:
                    if art.artifact_type == "grdl_processor":
                        proc_name = art.processor_class or art.name
                        self._macro_recorder.add_step(proc_name, params=params)
                except Exception as ex:
                    _log.warning("Could not record macro step: %s", ex)

            self._execution_callback(result_data, art.name)
            self._status_label.setText(f"Finished {art.name} successfully.")
            self.accept()
        except Exception as e:
            _log.error("Execution failed: %s", e, exc_info=True)
            QMessageBox.critical(self, "Execution Error", f"Failed to execute '{art.name}':\n{e}")
            self._status_label.setText("Execution failed.")
        finally:
            self._run_btn.setEnabled(True)
