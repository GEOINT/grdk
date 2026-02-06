# -*- coding: utf-8 -*-
"""
OWArtifactEditor Widget - Add, edit, and remove artifact metadata.

Form-based interface for managing artifact entries in the catalog
database.

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
from typing import Optional

# Third-party
from orangewidget import gui
from orangewidget.widget import OWBaseWidget, Msg

from AnyQt.QtWidgets import (
    QComboBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

# GRDK internal
from grdk.catalog.models import Artifact


class OWArtifactEditor(OWBaseWidget):
    """Add, edit, and remove artifacts in the catalog.

    Form-based interface for creating new artifact entries or editing
    existing ones. Supports both GRDL processor and GRDK workflow
    artifact types.
    """

    name = "Artifact Editor"
    description = "Add, edit, and remove artifact metadata"
    icon = "icons/artifact_editor.svg"
    category = "GRDK Admin"
    priority = 20

    class Error(OWBaseWidget.Error):
        catalog_error = Msg("Failed to open catalog: {}")
        save_error = Msg("Failed to save artifact: {}")

    class Information(OWBaseWidget.Information):
        saved = Msg("Artifact saved successfully.")
        removed = Msg("Artifact removed successfully.")

    want_main_area = True

    def __init__(self) -> None:
        super().__init__()

        self._catalog = None

        # --- Control area ---
        box = gui.vBox(self.controlArea, "Actions")

        btn_save = QPushButton("Save Artifact", self)
        btn_save.clicked.connect(self._on_save)
        box.layout().addWidget(btn_save)

        btn_clear = QPushButton("Clear Form", self)
        btn_clear.clicked.connect(self._on_clear)
        box.layout().addWidget(btn_clear)

        btn_remove = QPushButton("Remove Artifact", self)
        btn_remove.clicked.connect(self._on_remove)
        box.layout().addWidget(btn_remove)

        # --- Main area: form ---
        form = self.mainArea
        layout = form.layout()

        # Basic metadata
        layout.addWidget(QLabel("Name:"))
        self._name_edit = QLineEdit(self)
        layout.addWidget(self._name_edit)

        layout.addWidget(QLabel("Version:"))
        self._version_edit = QLineEdit("0.1.0", self)
        layout.addWidget(self._version_edit)

        layout.addWidget(QLabel("Type:"))
        self._type_combo = QComboBox(self)
        self._type_combo.addItem("GRDL Processor", "grdl_processor")
        self._type_combo.addItem("GRDK Workflow", "grdk_workflow")
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        layout.addWidget(self._type_combo)

        layout.addWidget(QLabel("Description:"))
        self._desc_edit = QLineEdit(self)
        layout.addWidget(self._desc_edit)

        layout.addWidget(QLabel("Author:"))
        self._author_edit = QLineEdit(self)
        layout.addWidget(self._author_edit)

        # Package info
        layout.addWidget(QLabel("PyPI Package:"))
        self._pypi_edit = QLineEdit(self)
        layout.addWidget(self._pypi_edit)

        layout.addWidget(QLabel("Conda Package:"))
        self._conda_edit = QLineEdit(self)
        layout.addWidget(self._conda_edit)

        layout.addWidget(QLabel("Conda Channel:"))
        self._channel_edit = QLineEdit("conda-forge", self)
        layout.addWidget(self._channel_edit)

        # Processor-specific
        self._proc_label = QLabel("Processor Class:")
        layout.addWidget(self._proc_label)
        self._proc_class_edit = QLineEdit(self)
        layout.addWidget(self._proc_class_edit)

        self._proc_type_label = QLabel("Processor Type:")
        layout.addWidget(self._proc_type_label)
        self._proc_type_combo = QComboBox(self)
        self._proc_type_combo.addItem("transform")
        self._proc_type_combo.addItem("detector")
        self._proc_type_combo.addItem("decomposition")
        layout.addWidget(self._proc_type_combo)

        # Workflow-specific
        self._yaml_label = QLabel("YAML Definition:")
        layout.addWidget(self._yaml_label)
        self._yaml_edit = QPlainTextEdit(self)
        self._yaml_edit.setMaximumHeight(150)
        layout.addWidget(self._yaml_edit)

        self._dsl_label = QLabel("Python DSL:")
        layout.addWidget(self._dsl_label)
        self._dsl_edit = QPlainTextEdit(self)
        self._dsl_edit.setMaximumHeight(150)
        layout.addWidget(self._dsl_edit)

        # Initialize catalog
        self._open_catalog()
        self._on_type_changed(0)

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

    def _on_type_changed(self, _index: int) -> None:
        """Toggle processor vs workflow fields based on type."""
        is_processor = self._type_combo.currentData() == "grdl_processor"

        self._proc_label.setVisible(is_processor)
        self._proc_class_edit.setVisible(is_processor)
        self._proc_type_label.setVisible(is_processor)
        self._proc_type_combo.setVisible(is_processor)

        self._yaml_label.setVisible(not is_processor)
        self._yaml_edit.setVisible(not is_processor)
        self._dsl_label.setVisible(not is_processor)
        self._dsl_edit.setVisible(not is_processor)

    def _on_save(self) -> None:
        """Save the artifact to the catalog."""
        if self._catalog is None:
            return

        name = self._name_edit.text().strip()
        version = self._version_edit.text().strip()
        if not name or not version:
            return

        artifact_type = self._type_combo.currentData()

        try:
            artifact = Artifact(
                name=name,
                version=version,
                artifact_type=artifact_type,
                description=self._desc_edit.text(),
                author=self._author_edit.text(),
                pypi_package=self._pypi_edit.text() or None,
                conda_package=self._conda_edit.text() or None,
                conda_channel=self._channel_edit.text() or None,
                processor_class=(
                    self._proc_class_edit.text() or None
                    if artifact_type == "grdl_processor"
                    else None
                ),
                processor_type=(
                    self._proc_type_combo.currentText()
                    if artifact_type == "grdl_processor"
                    else None
                ),
                yaml_definition=(
                    self._yaml_edit.toPlainText() or None
                    if artifact_type == "grdk_workflow"
                    else None
                ),
                python_dsl=(
                    self._dsl_edit.toPlainText() or None
                    if artifact_type == "grdk_workflow"
                    else None
                ),
            )
            self._catalog.add_artifact(artifact)
            self.Information.saved()
            self.Error.save_error.clear()
        except Exception as e:
            self.Error.save_error(str(e))

    def _on_clear(self) -> None:
        """Clear all form fields."""
        self._name_edit.clear()
        self._version_edit.setText("0.1.0")
        self._desc_edit.clear()
        self._author_edit.clear()
        self._pypi_edit.clear()
        self._conda_edit.clear()
        self._channel_edit.setText("conda-forge")
        self._proc_class_edit.clear()
        self._yaml_edit.clear()
        self._dsl_edit.clear()

    def _on_remove(self) -> None:
        """Remove the artifact from the catalog."""
        if self._catalog is None:
            return

        name = self._name_edit.text().strip()
        version = self._version_edit.text().strip()
        if not name or not version:
            return

        if self._catalog.remove_artifact(name, version):
            self.Information.removed()
        self._on_clear()

    def onDeleteWidget(self) -> None:
        """Close catalog on widget removal."""
        if self._catalog is not None:
            self._catalog.close()
        super().onDeleteWidget()
