# -*- coding: utf-8 -*-
"""
Conftest for widget smoke tests.

Provides a QApplication fixture and mock catalog for testing
Orange widgets without a full Orange environment.

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

import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="session")
def qapp():
    """Create a QApplication for the test session."""
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("Qt not available")
        return

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def mock_catalog():
    """Mock ArtifactCatalog for widget tests."""
    catalog = MagicMock()
    catalog.list_artifacts.return_value = []
    catalog.search.return_value = []
    catalog.search_by_tags.return_value = []
    return catalog
