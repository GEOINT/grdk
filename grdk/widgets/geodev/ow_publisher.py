# -*- coding: utf-8 -*-
"""
OWPublisher Widget - Publish workflow as Python DSL and YAML.

Generates both Python DSL and YAML representations of a processing
pipeline, with tag editing and catalog registration.

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
from typing import Any, Dict, List, Optional

# Third-party
from orangewidget import gui
from orangewidget.settings import Setting
from orangewidget.widget import OWBaseWidget, Input, Output, Msg

from AnyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# GRDK internal
from grdk.core.dsl import DslCompiler
from grdk.core.tags import (
    DetectionType,
    ImageModality,
    SegmentationType,
    WorkflowTags,
)
from grdk.core.workflow import WorkflowDefinition, WorkflowState
from grdk.widgets._signals import (
    GrdkProjectSignal,
    ProcessingPipelineSignal,
    WorkflowArtifactSignal,
)


class OWPublisher(OWBaseWidget):
    """Publish a processing pipeline as a workflow definition.

    Generates Python DSL and YAML representations from the connected
    pipeline. Allows editing workflow metadata and tags before
    publishing to file or catalog.
    """

    name = "Publisher"
    description = "Publish workflow as Python DSL and YAML"
    icon = "icons/publisher.svg"
    category = "GEODEV"
    priority = 80

    class Inputs:
        pipeline = Input("Pipeline", ProcessingPipelineSignal)
        project = Input("Project", GrdkProjectSignal)

    class Outputs:
        artifact = Output("Workflow Artifact", WorkflowArtifactSignal)

    class Warning(OWBaseWidget.Warning):
        no_pipeline = Msg("No pipeline connected.")

    class Information(OWBaseWidget.Information):
        published = Msg("Workflow published successfully.")

    # Persisted settings
    workflow_name: str = Setting("My Workflow")
    workflow_version: str = Setting("0.1.0")
    workflow_description: str = Setting("")

    want_main_area = True

    def __init__(self) -> None:
        super().__init__()

        self._pipeline: Optional[WorkflowDefinition] = None
        self._project: Optional[Any] = None
        self._compiler = DslCompiler()

        # --- Control area ---
        box = gui.vBox(self.controlArea, "Workflow Metadata")

        box.layout().addWidget(QLabel("Name:"))
        self._name_edit = QLineEdit(self.workflow_name, self)
        box.layout().addWidget(self._name_edit)

        box.layout().addWidget(QLabel("Version:"))
        self._version_edit = QLineEdit(self.workflow_version, self)
        box.layout().addWidget(self._version_edit)

        box.layout().addWidget(QLabel("Description:"))
        self._desc_edit = QLineEdit(self.workflow_description, self)
        box.layout().addWidget(self._desc_edit)

        # Tags
        tag_box = gui.vBox(self.controlArea, "Tags")

        tag_box.layout().addWidget(QLabel("Modalities:"))
        self._modality_checks: Dict[str, QCheckBox] = {}
        for mod in ImageModality:
            cb = QCheckBox(mod.value, self)
            self._modality_checks[mod.value] = cb
            tag_box.layout().addWidget(cb)

        tag_box.layout().addWidget(QLabel("Detection Type:"))
        self._detection_combo = QComboBox(self)
        self._detection_combo.addItem("(none)", None)
        for dt in DetectionType:
            self._detection_combo.addItem(dt.value, dt)
        tag_box.layout().addWidget(self._detection_combo)

        self._day_check = QCheckBox("Day Capable", self)
        self._day_check.setChecked(True)
        self._night_check = QCheckBox("Night Capable", self)
        tag_box.layout().addWidget(self._day_check)
        tag_box.layout().addWidget(self._night_check)

        # Actions
        act_box = gui.vBox(self.controlArea, "Actions")

        btn_generate = QPushButton("Generate", self)
        btn_generate.clicked.connect(self._on_generate)
        act_box.layout().addWidget(btn_generate)

        btn_export = QPushButton("Export to Files...", self)
        btn_export.clicked.connect(self._on_export)
        act_box.layout().addWidget(btn_export)

        btn_publish = QPushButton("Publish to Catalog", self)
        btn_publish.clicked.connect(self._on_publish_catalog)
        act_box.layout().addWidget(btn_publish)

        # --- Main area: tab views ---
        self._tabs = QTabWidget(self.mainArea)
        self.mainArea.layout().addWidget(self._tabs)

        self._python_view = QPlainTextEdit()
        self._python_view.setReadOnly(True)
        self._tabs.addTab(self._python_view, "Python DSL")

        self._yaml_view = QPlainTextEdit()
        self._yaml_view.setReadOnly(True)
        self._tabs.addTab(self._yaml_view, "YAML")

    @Inputs.pipeline
    def set_pipeline(self, signal: Optional[ProcessingPipelineSignal]) -> None:
        """Receive pipeline signal."""
        if signal is None or signal.workflow is None:
            self._pipeline = None
            self.Warning.no_pipeline()
        else:
            self._pipeline = signal.workflow
            self.Warning.no_pipeline.clear()

    @Inputs.project
    def set_project(self, signal: Optional[GrdkProjectSignal]) -> None:
        """Receive project signal (for catalog integration)."""
        self._project = signal.project if signal else None

    def _build_workflow(self) -> WorkflowDefinition:
        """Build a WorkflowDefinition from current UI state."""
        modalities = [
            ImageModality(val)
            for val, cb in self._modality_checks.items()
            if cb.isChecked()
        ]
        detection_types = []
        dt = self._detection_combo.currentData()
        if dt is not None:
            detection_types.append(dt)

        tags = WorkflowTags(
            modalities=modalities,
            day_capable=self._day_check.isChecked(),
            night_capable=self._night_check.isChecked(),
            detection_types=detection_types,
        )

        # Copy steps from the connected pipeline
        steps = list(self._pipeline.steps) if self._pipeline else []

        return WorkflowDefinition(
            name=self._name_edit.text(),
            version=self._version_edit.text(),
            description=self._desc_edit.text(),
            steps=steps,
            tags=tags,
            state=WorkflowState.PUBLISHED,
        )

    def _on_generate(self) -> None:
        """Generate Python DSL and YAML from the current pipeline."""
        wf = self._build_workflow()

        python_src = self._compiler.to_python(wf)
        yaml_src = self._compiler.to_yaml(wf)

        self._python_view.setPlainText(python_src)
        self._yaml_view.setPlainText(yaml_src)

        # Persist settings
        self.workflow_name = self._name_edit.text()
        self.workflow_version = self._version_edit.text()
        self.workflow_description = self._desc_edit.text()

        # Emit artifact signal
        self.Outputs.artifact.send(WorkflowArtifactSignal(
            python_dsl=python_src,
            yaml_definition=yaml_src,
            metadata=wf.tags.to_dict(),
        ))

    def _on_export(self) -> None:
        """Export generated files to disk."""
        wf = self._build_workflow()
        python_src = self._compiler.to_python(wf)
        yaml_src = self._compiler.to_yaml(wf)

        dir_path = QFileDialog.getExistingDirectory(
            self, "Export workflow files"
        )
        if not dir_path:
            return

        base = Path(dir_path)
        func_name = wf.name.lower().replace(' ', '_').replace('-', '_')

        py_path = base / f"{func_name}.py"
        yaml_path = base / f"{func_name}.yaml"

        py_path.write_text(python_src, encoding='utf-8')
        yaml_path.write_text(yaml_src, encoding='utf-8')

        self._python_view.setPlainText(python_src)
        self._yaml_view.setPlainText(yaml_src)
        self.Information.published()

    def _on_publish_catalog(self) -> None:
        """Register workflow in the artifact catalog."""
        wf = self._build_workflow()
        python_src = self._compiler.to_python(wf)
        yaml_src = self._compiler.to_yaml(wf)

        try:
            from grdk.catalog.database import ArtifactCatalog
            from grdk.catalog.models import Artifact
            from grdk.catalog.resolver import resolve_catalog_path

            catalog_path = resolve_catalog_path()
            with ArtifactCatalog(db_path=catalog_path) as catalog:
                artifact = Artifact(
                    name=wf.name,
                    version=wf.version,
                    artifact_type="grdk_workflow",
                    description=wf.description,
                    yaml_definition=yaml_src,
                    python_dsl=python_src,
                    tags=wf.tags.to_dict(),
                )
                catalog.add_artifact(artifact)

            self.Information.published()
        except Exception:
            pass

        self._python_view.setPlainText(python_src)
        self._yaml_view.setPlainText(yaml_src)

        self.Outputs.artifact.send(WorkflowArtifactSignal(
            python_dsl=python_src,
            yaml_definition=yaml_src,
            metadata=wf.tags.to_dict(),
        ))
