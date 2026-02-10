# -*- coding: utf-8 -*-
"""
OWGrdkProject Widget - Create, open, and save GRDK projects.

Entry point widget for the GEODEV workflow. Allows users to create
new projects or open existing ones, edit project-level tags, and
emit the project as a signal for downstream widgets.

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
from typing import Optional

# Third-party
from orangewidget import gui
from orangewidget.settings import Setting
from orangewidget.widget import OWBaseWidget, Output, Msg

from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

# GRDK internal
from grdl_rt.execution.project import GrdkProject
from grdl_rt.execution.tags import ProjectTags
from grdk.widgets._signals import GrdkProjectSignal


class OWGrdkProject(OWBaseWidget):
    """Create or open a GRDK project.

    Provides a file-browser interface for creating new projects or
    opening existing ones. Emits the project as a signal for downstream
    widgets in the GEODEV workflow.
    """

    name = "GRDK Project"
    description = "Create or open a GRDK project"
    icon = "icons/project.svg"
    category = "GEODEV"
    priority = 10

    class Outputs:
        project = Output("Project", GrdkProjectSignal)

    class Warning(OWBaseWidget.Warning):
        no_project = Msg("No project loaded.")

    # Persisted settings
    recent_path: str = Setting("")
    project_name: str = Setting("")
    intended_target: str = Setting("")

    want_main_area = False

    def __init__(self) -> None:
        super().__init__()

        self._project: Optional[GrdkProject] = None

        # --- Control area ---
        box = gui.vBox(self.controlArea, "Project")

        # Project name
        self._name_edit = QLineEdit(self)
        self._name_edit.setPlaceholderText("Project name")
        if self.project_name:
            self._name_edit.setText(self.project_name)
        box.layout().addWidget(QLabel("Name:"))
        box.layout().addWidget(self._name_edit)

        # Target tag
        self._target_edit = QLineEdit(self)
        self._target_edit.setPlaceholderText("e.g., vehicle, building, ship")
        if self.intended_target:
            self._target_edit.setText(self.intended_target)
        box.layout().addWidget(QLabel("Intended Target:"))
        box.layout().addWidget(self._target_edit)

        # Buttons
        btn_new = QPushButton("New Project...", self)
        btn_new.clicked.connect(self._on_new_project)
        box.layout().addWidget(btn_new)

        btn_open = QPushButton("Open Project...", self)
        btn_open.clicked.connect(self._on_open_project)
        box.layout().addWidget(btn_open)

        btn_save = QPushButton("Save Project", self)
        btn_save.clicked.connect(self._on_save_project)
        box.layout().addWidget(btn_save)

        # Status label
        self._status_label = QLabel("No project loaded", self)
        box.layout().addWidget(self._status_label)

        # Load recent project if available
        if self.recent_path:
            self._try_load(Path(self.recent_path))

    def _on_new_project(self) -> None:
        """Create a new project via directory picker."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Choose project directory"
        )
        if not dir_path:
            return

        name = self._name_edit.text() or "Untitled Project"
        tags = ProjectTags(
            intended_target=self._target_edit.text() or None
        )

        try:
            self._project = GrdkProject.create(
                Path(dir_path), name=name, tags=tags
            )
            self._update_after_load()
        except FileExistsError:
            # Project already exists — open it instead
            self._try_load(Path(dir_path))

    def _on_open_project(self) -> None:
        """Open an existing project via directory picker."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Open GRDK project directory"
        )
        if dir_path:
            self._try_load(Path(dir_path))

    def _on_save_project(self) -> None:
        """Save the current project."""
        if self._project is None:
            return

        self._project.name = self._name_edit.text()
        target = self._target_edit.text()
        self._project.tags.intended_target = target if target else None
        self._project.save()
        self._status_label.setText(f"Saved: {self._project.project_dir}")

    def _try_load(self, project_dir: Path) -> None:
        """Attempt to load a project from a directory."""
        try:
            self._project = GrdkProject.load(project_dir)
            self._update_after_load()
        except FileNotFoundError:
            self.Warning.no_project()
            self._status_label.setText(f"No project found at {project_dir}")

    def _update_after_load(self) -> None:
        """Update UI and emit signal after loading/creating a project."""
        if self._project is None:
            return

        self.Warning.no_project.clear()
        self._name_edit.setText(self._project.name)
        if self._project.tags.intended_target:
            self._target_edit.setText(self._project.tags.intended_target)

        self.recent_path = str(self._project.project_dir)
        self.project_name = self._project.name
        self.intended_target = self._target_edit.text()

        n_images = len(self._project.image_paths)
        n_workflows = len(self._project.workflows)
        self._status_label.setText(
            f"Loaded: {self._project.project_dir}\n"
            f"Images: {n_images}, Workflows: {n_workflows}"
        )

        self.Outputs.project.send(GrdkProjectSignal(self._project))
