# -*- coding: utf-8 -*-
"""
Integration tests for catalog preferences, multi-catalog management, and viewer execution.

Author
------
Claude Code / Steven Siebert

License
-------
MIT License
Copyright (c) 2026 geoint.org
"""

import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from PyQt6.QtWidgets import QApplication

# Ensure QApplication exists for UI tests
if QApplication.instance() is None:
    _app = QApplication(["--platform", "offscreen"])
else:
    _app = QApplication.instance()

from grdk.catalog.manager import (
    CatalogEntry,
    CatalogPreferencesManager,
    CatalogManager,
)
from grdl_rt.catalog.database import SqliteArtifactCatalog
from grdl_rt.catalog.yaml_catalog import YamlArtifactCatalog
from grdl_rt.catalog.models import Artifact


@pytest.fixture
def temp_grdl_env(monkeypatch, tmp_path):
    """Sets up an isolated ~/.grdl config and cache directory in tmp_path."""
    config_dir = tmp_path / ".grdl"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.json"
    cache_file = config_dir / ".catalog_cache.json"

    monkeypatch.setattr("grdk.catalog.manager._CONFIG_DIR", config_dir)
    monkeypatch.setattr("grdk.catalog.manager._CONFIG_FILE", config_file)
    monkeypatch.setattr("grdk.catalog.manager._CACHE_FILE", cache_file)

    return config_dir


@pytest.fixture
def sample_processor_artifact():
    return Artifact(
        name="test_invert_filter",
        version="1.0.0",
        artifact_type="grdl_processor",
        description="Simple test inverter",
        author="Test Author",
        processor_class="tests.test_catalog_integration.MockProcessor",
        processor_type="transform",
        param_schema={
            "type": "object",
            "properties": {
                "factor": {
                    "type": "number",
                    "default": 1.0,
                    "description": "Scaling factor",
                    "minimum": 0.0,
                    "maximum": 10.0,
                }
            },
        },
    )


def test_catalog_preferences_defaults(temp_grdl_env):
    """Test reading and writing preferences in isolated config."""
    config = CatalogPreferencesManager.load_config()
    assert config == {}

    # Output destination defaults to 'replace'
    assert CatalogPreferencesManager.get_output_destination_mode() == "replace"

    # Set and read back
    CatalogPreferencesManager.set_output_destination_mode("dual")
    assert CatalogPreferencesManager.get_output_destination_mode() == "dual"


def test_catalog_manager_add_toggle_remove(temp_grdl_env, tmp_path):
    """Test adding, toggling, and removing catalog sources."""
    mgr = CatalogManager()
    initial_count = len(mgr.catalogs)

    # Create dummy catalog file
    dummy_db = tmp_path / "custom.db"
    dummy_db.write_text("dummy")

    entry = mgr.add_catalog(name="Custom Catalog", path=str(dummy_db), backend_type="sqlite")
    assert len(mgr.catalogs) == initial_count + 1
    assert entry.name == "Custom Catalog"
    assert entry.enabled is True

    # Toggle disabled
    mgr.toggle_catalog(initial_count, False)
    assert mgr.catalogs[initial_count].enabled is False

    # Remove
    mgr.remove_catalog(initial_count)
    assert len(mgr.catalogs) == initial_count


def test_catalog_manager_sqlite_and_yaml_federation(temp_grdl_env, tmp_path, sample_processor_artifact):
    """Test federating SQLite and YAML catalogs with caching and staleness detection."""
    # 1. Create a SQLite catalog with sample processor
    db_path = tmp_path / "sqlite_cat.db"
    sqlite_cat = SqliteArtifactCatalog(db_path)
    sqlite_cat.add_artifact(sample_processor_artifact)
    sqlite_cat.close()

    # 2. Create a YAML catalog with another workflow artifact
    yaml_path = tmp_path / "yaml_cat.yaml"
    yaml_content = """artifacts:
  - name: test_yaml_workflow
    version: "2.0.0"
    artifact_type: grdk_workflow
    description: "Workflow in YAML"
    author: "Test Author"
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")

    # Configure CatalogManager
    mgr = CatalogManager()
    # Clear any default
    mgr._catalogs = []
    mgr.add_catalog("Test SQLite", str(db_path), "sqlite")
    mgr.add_catalog("Test YAML", str(yaml_path), "yaml")

    # Refresh and verify artifacts
    artifacts = mgr.refresh_cache(force=True)
    names = [a.name for a in artifacts]
    assert "test_invert_filter" in names
    assert "test_yaml_workflow" in names
    assert len(artifacts) == 2

    # Verify disk cache file exists
    cache_file = temp_grdl_env / ".catalog_cache.json"
    assert cache_file.exists()

    # Verify cache is NOT stale initially
    assert not mgr.is_cache_stale()

    # Modify YAML file mtime and verify cache detects staleness
    time.sleep(0.05)
    new_yaml_content = yaml_content.replace("Workflow in YAML", "Updated Workflow in YAML")
    yaml_path.write_text(new_yaml_content, encoding="utf-8")
    assert mgr.is_cache_stale()

    # Refresh again
    mgr.refresh_cache()
    assert not mgr.is_cache_stale()
    art = mgr.get_artifact("test_yaml_workflow")
    assert art is not None
    assert art.description == "Updated Workflow in YAML"

    # Verify get_catalog returns active federated catalog
    cat = mgr.get_catalog()
    assert cat is not None
    assert len(cat.list_artifacts()) == 2


def test_catalog_management_dialog(temp_grdl_env, tmp_path):
    """Test CatalogManagementDialog initialization and table population."""
    from grdk.catalog.ui import CatalogManagementDialog

    mgr = CatalogManager()
    db_path = tmp_path / "test.db"
    db_path.write_text("data")
    mgr.add_catalog("Test Source", str(db_path), "sqlite")

    dlg = CatalogManagementDialog(manager=mgr)
    assert dlg.windowTitle() == "Catalog Management"
    assert dlg._table.rowCount() >= 1
    dlg.close()


def test_catalog_runner_dialog_population(temp_grdl_env, tmp_path, sample_processor_artifact):
    """Test CatalogRunnerDialog list display and schema controls."""
    from grdk.catalog.ui import CatalogRunnerDialog

    db_path = tmp_path / "cat.db"
    cat = SqliteArtifactCatalog(db_path)
    cat.add_artifact(sample_processor_artifact)
    cat.close()

    mgr = CatalogManager()
    mgr._catalogs = [CatalogEntry("Cat", str(db_path), "sqlite", enabled=True)]
    mgr.refresh_cache(force=True)

    test_image = np.ones((64, 64), dtype=np.float32)
    dlg = CatalogRunnerDialog(
        manager=mgr,
        source_array_provider=lambda: test_image,
        execution_callback=MagicMock(),
    )

    # Verify artifact appears in the list
    assert dlg._list.count() >= 1
    item = dlg._list.item(0)
    assert "test_invert_filter" in item.text()

    # Select it and verify controls built
    dlg._list.setCurrentRow(0)
    assert "factor" in dlg._param_inputs
    dlg.close()


def test_viewer_catalog_result_replace_and_memory_cleanup(temp_grdl_env, monkeypatch):
    """Verify that when a catalog workflow/processor result arrives,
    the active pane is replaced and previous resources/memory are cleaned up.
    """
    from grdk.viewers.main_window import ViewerMainWindow

    # Set mode to 'replace'
    CatalogPreferencesManager.set_output_destination_mode("replace")

    window = ViewerMainWindow()
    # Mock status bar to avoid unneeded UI calls
    window.statusBar = MagicMock()

    # Initial image
    initial_array = np.zeros((100, 100), dtype=np.float32)
    window.set_array(initial_array)
    active_v = window._viewer.active_viewer
    assert window._get_source_array(active_v) is not None

    # Track garbage collection call
    gc_mock = MagicMock()
    monkeypatch.setattr(gc, "collect", gc_mock)

    # Deliver catalog execution result
    new_array = np.ones((50, 50), dtype=np.uint8) * 200
    window._on_catalog_result_ready(new_array, "TestProcessor")

    # Verify active array was replaced
    src = window._get_source_array(window._viewer.active_viewer)
    assert src is not None
    assert src.shape == (50, 50)
    assert np.all(src == 200)

    # Verify memory cleanup was invoked
    assert gc_mock.called

    window.close()


def test_viewer_catalog_result_dual_mode(temp_grdl_env):
    """Verify that when mode is 'dual', execution triggers dual viewer view."""
    from grdk.viewers.main_window import ViewerMainWindow

    # Set mode to 'dual'
    CatalogPreferencesManager.set_output_destination_mode("dual")

    window = ViewerMainWindow()
    initial_array = np.zeros((100, 100), dtype=np.float32)
    window.set_array(initial_array)

    new_array = np.ones((100, 100), dtype=np.float32)
    window._on_catalog_result_ready(new_array, "TestWorkflow")

    # In dual mode, dual viewer split should be invoked
    assert window._viewer.mode == "dual"
    assert window._viewer.right_viewer.canvas.source_array is not None

    window.close()
