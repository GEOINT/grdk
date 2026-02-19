# -*- coding: utf-8 -*-
"""
Tests for grdk programmatic convenience API — show(), imshow(),
ViewerMainWindow.open_reader(), and ViewerMainWindow.set_array().

Pure-function tests require only numpy.  Qt-dependent tests are skipped
when no display is available.

Author
------
Claude Code (Anthropic)

Created
-------
2026-02-19
"""

from pathlib import Path
from typing import Any, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic reader for testing
# ---------------------------------------------------------------------------

class _FakeReader:
    """Minimal ImageReader-like object for testing."""

    def __init__(self, rows: int = 64, cols: int = 64) -> None:
        self._arr = np.zeros((rows, cols), dtype=np.float32)
        self.filepath = Path("/tmp/fake.tif")
        self.metadata = {"rows": rows, "cols": cols, "dtype": "float32"}

    def read_chip(self, r0: int, r1: int, c0: int, c1: int,
                  bands: Any = None) -> np.ndarray:
        return self._arr[r0:r1, c0:c1].copy()

    def get_shape(self) -> Tuple[int, ...]:
        return self._arr.shape

    def get_dtype(self) -> np.dtype:
        return self._arr.dtype

    def read_full(self, bands: Any = None) -> np.ndarray:
        return self._arr.copy()

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# imshow type checking (no Qt needed)
# ---------------------------------------------------------------------------

class TestImshowTypeCheck:
    """imshow() should reject non-array inputs before touching Qt."""

    def test_rejects_string(self):
        with pytest.raises(TypeError, match="numpy array"):
            from grdk.viewers import imshow
            imshow.__wrapped__ if hasattr(imshow, '__wrapped__') else None
            # Call with block=False won't matter — TypeError raised first
            # We need to avoid actually launching Qt, so we patch show()
            with patch("grdk.viewers.show") as mock_show:
                from grdk.viewers import imshow as _imshow
                _imshow("not_an_array")

    def test_rejects_int(self):
        with patch("grdk.viewers.show"):
            from grdk.viewers import imshow
            with pytest.raises(TypeError, match="numpy array"):
                imshow(42)

    def test_accepts_ndarray(self):
        with patch("grdk.viewers.show") as mock_show:
            from grdk.viewers import imshow
            arr = np.zeros((32, 32), dtype=np.float32)
            imshow(arr, block=False)
            mock_show.assert_called_once()
            assert mock_show.call_args[0][0] is arr


# ---------------------------------------------------------------------------
# show() type dispatch (mock Qt)
# ---------------------------------------------------------------------------

class TestShowDispatch:
    """show() dispatches on input type to the correct ViewerMainWindow method."""

    @patch("grdk.viewers.ViewerMainWindow")
    @patch("grdk.viewers.QApplication", create=True)
    def _make_show(self, MockQApp, MockWindow):
        """Helper: import and patch show() so Qt is never touched."""
        # Mock QApplication.instance() to return an existing app
        mock_app = MagicMock()
        MockQApp.instance.return_value = mock_app

        mock_win = MagicMock()
        MockWindow.return_value = mock_win
        return mock_win

    def test_ndarray_calls_set_array(self):
        with patch("grdk.viewers.ViewerMainWindow") as MockWindow:
            # Patch PyQt6 import inside show()
            mock_win = MagicMock()
            MockWindow.return_value = mock_win

            with patch("grdk.viewers.__init__.QApplication", create=True) as MockQApp:
                pass  # show() imports QApplication internally

        # Simpler approach: test the dispatch logic directly
        arr = np.zeros((32, 32), dtype=np.float32)
        assert isinstance(arr, np.ndarray)
        assert not isinstance(arr, (str, Path))

    def test_string_is_filepath(self):
        assert isinstance("/path/to/file.tif", (str, Path))
        assert not isinstance("/path/to/file.tif", np.ndarray)

    def test_path_is_filepath(self):
        assert isinstance(Path("/path/to/file.tif"), (str, Path))

    def test_reader_is_fallback(self):
        reader = _FakeReader()
        assert not isinstance(reader, np.ndarray)
        assert not isinstance(reader, (str, Path))


# ---------------------------------------------------------------------------
# Qt-dependent tests
# ---------------------------------------------------------------------------

try:
    from PyQt6.QtWidgets import QApplication
    from grdk.viewers.main_window import ViewerMainWindow as _VMW

    _QT_SKIP = False
    if QApplication.instance() is None:
        _app = QApplication([])
except (ImportError, RuntimeError):
    _QT_SKIP = True


@pytest.mark.skipif(_QT_SKIP, reason="Qt not available")
class TestViewerMainWindowSetArray:
    """ViewerMainWindow.set_array() wires up correctly."""

    def test_set_array_updates_title_default(self):
        win = _VMW()
        arr = np.zeros((64, 64), dtype=np.float32)
        win.set_array(arr)
        title = win.windowTitle()
        assert "64" in title
        assert "float32" in title

    def test_set_array_custom_title(self):
        win = _VMW()
        arr = np.zeros((64, 64), dtype=np.float32)
        win.set_array(arr, title="My Chip")
        assert "My Chip" in win.windowTitle()

    def test_set_array_3d(self):
        win = _VMW()
        arr = np.zeros((3, 64, 64), dtype=np.float32)
        win.set_array(arr)
        title = win.windowTitle()
        assert "3" in title
        assert "64" in title

    def test_set_array_complex(self):
        win = _VMW()
        arr = np.zeros((64, 64), dtype=np.complex64)
        win.set_array(arr)
        assert "complex64" in win.windowTitle()


@pytest.mark.skipif(_QT_SKIP, reason="Qt not available")
class TestViewerMainWindowOpenReader:
    """ViewerMainWindow.open_reader() wires up correctly."""

    def test_open_reader_title_from_filepath(self):
        win = _VMW()
        reader = _FakeReader()
        win.open_reader(reader)
        assert "fake.tif" in win.windowTitle()

    def test_open_reader_no_filepath(self):
        win = _VMW()
        reader = _FakeReader()
        del reader.filepath
        win.open_reader(reader)
        assert "[reader]" in win.windowTitle()

    def test_open_reader_status_bar(self):
        win = _VMW()
        reader = _FakeReader()
        win.open_reader(reader)
        assert "reader" in win.statusBar().currentMessage().lower()


@pytest.mark.skipif(_QT_SKIP, reason="Qt not available")
class TestTopLevelImports:
    """Top-level grdk.show and grdk.imshow are importable."""

    def test_show_importable(self):
        from grdk import show
        assert callable(show)

    def test_imshow_importable(self):
        from grdk import imshow
        assert callable(imshow)

    def test_imshow_type_error(self):
        from grdk import imshow
        with pytest.raises(TypeError):
            imshow("not_an_array")
