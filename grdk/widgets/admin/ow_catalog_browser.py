# -*- coding: utf-8 -*-
"""
OWCatalogBrowser Widget - Search and discover artifacts in the catalog.

Provides full-text search over GRDL processors and GRDK workflows,
with filtering by type and tags, and a detail panel showing artifact
metadata.

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
from typing import List, Optional

# Third-party
from orangewidget import gui
from orangewidget.widget import OWBaseWidget, Msg

from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

# GRDK internal
from grdl_rt.catalog.models import Artifact


class OWCatalogBrowser(OWBaseWidget):
    """Browse and search the artifact catalog.

    Full-text search over artifact names and descriptions, with
    type filtering and a detail panel showing full metadata.
    """

    name = "Catalog Browser"
    description = "Search and discover GRDL/GRDK artifacts"
    icon = "icons/catalog_browser.svg"
    category = "GRDK Admin"
    priority = 10

    class Error(OWBaseWidget.Error):
        catalog_error = Msg("Failed to open catalog: {}")

    want_main_area = True

    def __init__(self) -> None:
        super().__init__()

        self._artifacts: List[Artifact] = []
        self._catalog = None

        # --- Control area ---
        box = gui.vBox(self.controlArea, "Search")

        self._search_edit = QLineEdit(self)
        self._search_edit.setPlaceholderText("Search artifacts...")
        self._search_edit.returnPressed.connect(self._on_search)
        box.layout().addWidget(self._search_edit)

        btn_search = QPushButton("Search", self)
        btn_search.clicked.connect(self._on_search)
        box.layout().addWidget(btn_search)

        # Type filter
        box.layout().addWidget(QLabel("Filter by type:"))
        self._type_combo = QComboBox(self)
        self._type_combo.addItem("All", None)
        self._type_combo.addItem("GRDL Processors", "grdl_processor")
        self._type_combo.addItem("GRDK Workflows", "grdk_workflow")
        self._type_combo.currentIndexChanged.connect(self._on_filter_changed)
        box.layout().addWidget(self._type_combo)

        # Refresh
        btn_refresh = QPushButton("Refresh", self)
        btn_refresh.clicked.connect(self._on_refresh)
        box.layout().addWidget(btn_refresh)

        self._count_label = QLabel("Artifacts: 0", self)
        box.layout().addWidget(self._count_label)

        # --- Main area: list + detail splitter ---
        splitter = QSplitter(Qt.Orientation.Horizontal, self.mainArea)
        self.mainArea.layout().addWidget(splitter)

        # Left: artifact list
        self._list_widget = QListWidget()
        self._list_widget.currentRowChanged.connect(self._on_artifact_selected)
        splitter.addWidget(self._list_widget)

        # Right: detail panel
        detail = QWidget()
        detail.setLayout(QVBoxLayout())
        self._detail_text = QPlainTextEdit()
        self._detail_text.setReadOnly(True)
        detail.layout().addWidget(self._detail_text)
        splitter.addWidget(detail)

        splitter.setSizes([300, 400])

        # Load catalog on init
        self._open_catalog()
        self._on_refresh()

    def _open_catalog(self) -> None:
        """Open the artifact catalog database."""
        try:
            from grdl_rt.catalog.database import ArtifactCatalog
            from grdl_rt.catalog.resolver import resolve_catalog_path

            path = resolve_catalog_path()
            self._catalog = ArtifactCatalog(db_path=path)
            self.Error.catalog_error.clear()
        except Exception as e:
            self.Error.catalog_error(str(e))
            self._catalog = None

    def _on_search(self) -> None:
        """Execute a search query."""
        if self._catalog is None:
            return

        query = self._search_edit.text().strip()
        if query:
            self._artifacts = self._catalog.search(query)
        else:
            self._artifacts = self._catalog.list_artifacts()

        # Apply type filter
        type_filter = self._type_combo.currentData()
        if type_filter:
            self._artifacts = [
                a for a in self._artifacts if a.artifact_type == type_filter
            ]

        self._update_list()

    def _on_filter_changed(self, _index: int) -> None:
        """Re-run search with new type filter."""
        self._on_search()

    def _on_refresh(self) -> None:
        """Refresh the full artifact list."""
        if self._catalog is None:
            return

        type_filter = self._type_combo.currentData()
        self._artifacts = self._catalog.list_artifacts(artifact_type=type_filter)
        self._update_list()

    def _update_list(self) -> None:
        """Rebuild the list widget from current artifacts."""
        self._list_widget.clear()
        for artifact in self._artifacts:
            text = f"{artifact.name} v{artifact.version} ({artifact.artifact_type})"
            self._list_widget.addItem(text)
        self._count_label.setText(f"Artifacts: {len(self._artifacts)}")

    def _on_artifact_selected(self, row: int) -> None:
        """Show detail for the selected artifact."""
        if row < 0 or row >= len(self._artifacts):
            self._detail_text.clear()
            return

        a = self._artifacts[row]
        lines = [
            f"Name: {a.name}",
            f"Version: {a.version}",
            f"Type: {a.artifact_type}",
            f"Description: {a.description}",
            f"Author: {a.author}",
            f"License: {a.license}",
            "",
        ]

        if a.artifact_type == 'grdl_processor':
            lines.extend([
                "--- Processor Info ---",
                f"Class: {a.processor_class or 'N/A'}",
                f"Processor Version: {a.processor_version or 'N/A'}",
                f"Processor Type: {a.processor_type or 'N/A'}",
                "",
            ])

        if a.pypi_package:
            lines.append(f"PyPI: {a.pypi_package}")
        if a.conda_package:
            channel = a.conda_channel or 'conda-forge'
            lines.append(f"Conda: {a.conda_package} ({channel})")

        if a.tags:
            lines.extend(["", "--- Tags ---"])
            for key, values in a.tags.items():
                lines.append(f"  {key}: {', '.join(values)}")

        if a.yaml_definition:
            lines.extend([
                "",
                "--- YAML Definition ---",
                a.yaml_definition[:500],
            ])

        if a.python_dsl:
            lines.extend([
                "",
                "--- Python DSL ---",
                a.python_dsl[:500],
            ])

        self._detail_text.setPlainText('\n'.join(lines))

    def onDeleteWidget(self) -> None:
        """Close catalog on widget removal."""
        if self._catalog is not None:
            self._catalog.close()
        super().onDeleteWidget()
