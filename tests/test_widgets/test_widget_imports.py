# -*- coding: utf-8 -*-
"""
Widget Import Smoke Tests - Verify all 15 widget modules import cleanly.

Tests that each widget module can be imported without errors, catching
missing dependencies, syntax errors, and import cycle issues early.

Author
------
Claude Code (Anthropic)

Contributor
-----------
Steven Siebert

Created
-------
2026-02-06
"""

import importlib

import pytest


# All GRDK widget modules
_GEODEV_WIDGETS = [
    "grdk.widgets.geodev.ow_chipper",
    "grdk.widgets.geodev.ow_coregister",
    "grdk.widgets.geodev.ow_image_loader",
    "grdk.widgets.geodev.ow_labeler",
    "grdk.widgets.geodev.ow_orchestrator",
    "grdk.widgets.geodev.ow_preview",
    "grdk.widgets.geodev.ow_processor",
    "grdk.widgets.geodev.ow_project",
    "grdk.widgets.geodev.ow_publisher",
    "grdk.widgets.geodev.ow_stack_viewer",
]

_ADMIN_WIDGETS = [
    "grdk.widgets.admin.ow_artifact_editor",
    "grdk.widgets.admin.ow_catalog_browser",
    "grdk.widgets.admin.ow_update_monitor",
    "grdk.widgets.admin.ow_workflow_manager",
]


@pytest.mark.parametrize("module_path", _GEODEV_WIDGETS)
def test_geodev_widget_imports(module_path):
    """Each GEODEV widget module should import without error."""
    try:
        importlib.import_module(module_path)
    except ImportError as e:
        # Allow missing optional deps like Qt, but fail on actual code errors
        msg = str(e)
        if "PySide6" in msg or "orangewidget" in msg or "Qt" in msg:
            pytest.skip(f"Qt not available: {e}")
        raise


@pytest.mark.parametrize("module_path", _ADMIN_WIDGETS)
def test_admin_widget_imports(module_path):
    """Each Admin widget module should import without error."""
    try:
        importlib.import_module(module_path)
    except ImportError as e:
        msg = str(e)
        if "PySide6" in msg or "orangewidget" in msg or "Qt" in msg:
            pytest.skip(f"Qt not available: {e}")
        raise
