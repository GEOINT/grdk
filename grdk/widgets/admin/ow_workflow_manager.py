# -*- coding: utf-8 -*-
"""
OWWorkflowManager Widget - Manage workflow definitions.

Lists all registered workflow artifacts with import/export, tag
management, and version comparison.

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
from pathlib import Path
from typing import List, Optional

# Third-party
from orangewidget import gui
from orangewidget.widget import OWBaseWidget, Msg

from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

# GRDK internal
from grdk.catalog.models import Artifact


class OWWorkflowManager(OWBaseWidget):
    """Manage registered workflow definitions.

    Lists all grdk_workflow artifacts, with tabs for viewing YAML
    and Python DSL source. Supports import/export to files.
    """

    name = "Workflow Manager"
    description = "Manage workflow definitions"
    icon = "icons/workflow_manager.svg"
    category = "GRDK Admin"
    priority = 30

    class Error(OWBaseWidget.Error):
        catalog_error = Msg("Failed to open catalog: {}")

    class Information(OWBaseWidget.Information):
        imported = Msg("Workflow imported successfully.")
        exported = Msg("Workflow exported successfully.")

    want_main_area = True

    def __init__(self) -> None:
        super().__init__()

        self._catalog = None
        self._workflows: List[Artifact] = []

        # --- Control area ---
        box = gui.vBox(self.controlArea, "Actions")

        btn_refresh = QPushButton("Refresh", self)
        btn_refresh.clicked.connect(self._on_refresh)
        box.layout().addWidget(btn_refresh)

        btn_import = QPushButton("Import YAML...", self)
        btn_import.clicked.connect(self._on_import)
        box.layout().addWidget(btn_import)

        btn_export = QPushButton("Export Selected...", self)
        btn_export.clicked.connect(self._on_export)
        box.layout().addWidget(btn_export)

        btn_remove = QPushButton("Remove Selected", self)
        btn_remove.clicked.connect(self._on_remove)
        box.layout().addWidget(btn_remove)

        self._count_label = QLabel("Workflows: 0", self)
        box.layout().addWidget(self._count_label)

        # --- Main area ---
        splitter = QSplitter(Qt.Orientation.Horizontal, self.mainArea)
        self.mainArea.layout().addWidget(splitter)

        # Left: list
        self._list_widget = QListWidget()
        self._list_widget.currentRowChanged.connect(self._on_selected)
        splitter.addWidget(self._list_widget)

        # Right: tabs for YAML and Python DSL
        right = QWidget()
        right.setLayout(QVBoxLayout())

        self._info_label = QLabel("", right)
        self._info_label.setWordWrap(True)
        right.layout().addWidget(self._info_label)

        self._tabs = QTabWidget(right)

        self._yaml_view = QPlainTextEdit()
        self._yaml_view.setReadOnly(True)
        self._tabs.addTab(self._yaml_view, "YAML")

        self._python_view = QPlainTextEdit()
        self._python_view.setReadOnly(True)
        self._tabs.addTab(self._python_view, "Python DSL")

        right.layout().addWidget(self._tabs)
        splitter.addWidget(right)

        splitter.setSizes([250, 450])

        self._open_catalog()
        self._on_refresh()

    def _open_catalog(self) -> None:
        """Open the catalog database."""
        try:
            from grdk.catalog.database import ArtifactCatalog
            from grdk.catalog.resolver import resolve_catalog_path

            path = resolve_catalog_path()
            self._catalog = ArtifactCatalog(db_path=path)
            self.Error.catalog_error.clear()
        except Exception as e:
            self.Error.catalog_error(str(e))

    def _on_refresh(self) -> None:
        """Refresh workflow list from catalog."""
        if self._catalog is None:
            return

        self._workflows = self._catalog.list_artifacts(
            artifact_type="grdk_workflow"
        )
        self._list_widget.clear()
        for wf in self._workflows:
            self._list_widget.addItem(f"{wf.name} v{wf.version}")
        self._count_label.setText(f"Workflows: {len(self._workflows)}")

    def _on_selected(self, row: int) -> None:
        """Show details for the selected workflow."""
        if row < 0 or row >= len(self._workflows):
            return

        wf = self._workflows[row]
        self._info_label.setText(
            f"Name: {wf.name}\n"
            f"Version: {wf.version}\n"
            f"Description: {wf.description}\n"
            f"Author: {wf.author}"
        )
        self._yaml_view.setPlainText(wf.yaml_definition or "(no YAML)")
        self._python_view.setPlainText(wf.python_dsl or "(no Python DSL)")

    def _on_import(self) -> None:
        """Import a workflow from a YAML file."""
        if self._catalog is None:
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Import Workflow YAML", "",
            "YAML (*.yaml *.yml);;All Files (*)"
        )
        if not path:
            return

        try:
            from grdk.core.dsl import DslCompiler

            compiler = DslCompiler()
            wf_def = compiler.compile_yaml(Path(path))
            yaml_text = Path(path).read_text(encoding='utf-8')

            artifact = Artifact(
                name=wf_def.name,
                version=wf_def.version,
                artifact_type="grdk_workflow",
                description=wf_def.description,
                yaml_definition=yaml_text,
                python_dsl=compiler.to_python(wf_def),
                tags=wf_def.tags.to_dict(),
            )
            self._catalog.add_artifact(artifact)
            self.Information.imported()
            self._on_refresh()
        except Exception:
            pass

    def _on_export(self) -> None:
        """Export selected workflow to files."""
        row = self._list_widget.currentRow()
        if row < 0 or row >= len(self._workflows):
            return

        wf = self._workflows[row]
        dir_path = QFileDialog.getExistingDirectory(
            self, "Export workflow files"
        )
        if not dir_path:
            return

        base = Path(dir_path)
        func_name = wf.name.lower().replace(' ', '_').replace('-', '_')

        if wf.yaml_definition:
            (base / f"{func_name}.yaml").write_text(
                wf.yaml_definition, encoding='utf-8'
            )
        if wf.python_dsl:
            (base / f"{func_name}.py").write_text(
                wf.python_dsl, encoding='utf-8'
            )

        self.Information.exported()

    def _on_remove(self) -> None:
        """Remove the selected workflow from catalog."""
        row = self._list_widget.currentRow()
        if row < 0 or row >= len(self._workflows) or self._catalog is None:
            return

        wf = self._workflows[row]
        self._catalog.remove_artifact(wf.name, wf.version)
        self._on_refresh()

    def onDeleteWidget(self) -> None:
        """Close catalog on widget removal."""
        if self._catalog is not None:
            self._catalog.close()
        super().onDeleteWidget()
