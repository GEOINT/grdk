# -*- coding: utf-8 -*-
"""
Pytest configuration and session fixtures for GRDK.
"""

import sys
import pytest

try:
    from PyQt6.QtWidgets import QApplication
    _HAS_QT = True
except ImportError:
    _HAS_QT = False


@pytest.fixture(scope="session", autouse=True)
def qapp():
    """Ensure a single persistent QApplication instance exists for all GUI tests."""
    if not _HAS_QT:
        yield None
        return

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    yield app
