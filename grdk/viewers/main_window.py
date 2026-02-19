# -*- coding: utf-8 -*-
"""
ViewerMainWindow - Top-level application window for the geospatial viewer.

Standalone QMainWindow assembling GeoImageViewer with menu bar, toolbar,
display controls dock, and metadata dock.  Provides a ``main()`` entry
point for command-line invocation.

Dependencies
------------
PyQt6

Author
------
Claude Code (Anthropic)

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-17

Modified
--------
2026-02-17
"""

# Standard library
import argparse
import sys
from typing import Any, Optional

try:
    from PyQt6.QtWidgets import (
        QApplication,
        QDockWidget,
        QFileDialog,
        QGroupBox,
        QHeaderView,
        QMainWindow,
        QMessageBox,
        QTableWidget,
        QTableWidgetItem,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )
    from PyQt6.QtGui import QAction, QKeySequence
    from PyQt6.QtCore import Qt

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for grdk-viewer."""
    parser = argparse.ArgumentParser(
        prog="grdk-viewer",
        description="GRDK geospatial image viewer — display and inspect "
        "SAR, EO/IR, multispectral, and other grdl-supported imagery.",
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Image file or directory to open (GeoTIFF, NITF/SICD, HDF5, "
        "JPEG2000, Sentinel-1 .SAFE, BIOMASS product dir, etc.)",
    )
    return parser


if _QT_AVAILABLE:
    from grdk.viewers.geo_viewer import GeoImageViewer
    from grdk.widgets._display_controls import build_display_controls

    # File filter for the open dialog
    _IMAGE_FILTER = (
        "All Supported (*.tif *.tiff *.geotiff *.nitf *.ntf *.nsf "
        "*.h5 *.he5 *.hdf5 *.hdf *.jp2 *.j2k);;"
        "GeoTIFF (*.tif *.tiff *.geotiff);;"
        "NITF / SICD (*.nitf *.ntf *.nsf);;"
        "HDF5 (*.h5 *.he5 *.hdf5 *.hdf);;"
        "JPEG2000 (*.jp2 *.j2k);;"
        "All Files (*)"
    )

    _VECTOR_FILTER = "GeoJSON (*.geojson *.json);;All Files (*)"

    _EXPORT_FILTER = (
        "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;All Files (*)"
    )

    class ViewerMainWindow(QMainWindow):
        """Top-level geospatial image viewer application window.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget.
        """

        def __init__(self, parent: Optional[Any] = None) -> None:
            super().__init__(parent)

            self.setWindowTitle("GRDK Viewer")
            self.resize(1200, 800)

            # Central widget
            self._viewer = GeoImageViewer(self)
            self.setCentralWidget(self._viewer)

            # Build UI (docks + toolbar before menus so toggleViewAction is available)
            self._create_actions()
            self._create_display_dock()
            self._create_metadata_dock()
            self._create_toolbar()
            self._create_menus()

            # Wire band info updates to display controls
            self._viewer.band_info_changed.connect(self._on_band_info_changed)

            # Status bar
            self.statusBar().showMessage("Ready")

        # --- Public API ---

        def open_file(self, filepath: str) -> None:
            """Open an image file.

            Parameters
            ----------
            filepath : str
                Path to the image file.
            """
            try:
                self._viewer.open_file(filepath)
                self.setWindowTitle(f"GRDK Viewer — {filepath}")
                self._update_metadata_table()
                self._update_remap_state()
                self.statusBar().showMessage(f"Opened: {filepath}")
            except Exception as e:
                QMessageBox.critical(
                    self, "Open Error",
                    f"Could not open file:\n{filepath}\n\n{e}",
                )

        def open_reader(
            self,
            reader: Any,
            geolocation: Optional[Any] = None,
        ) -> None:
            """Open an image from a grdl ImageReader.

            Parameters
            ----------
            reader : ImageReader
                grdl reader instance.
            geolocation : Geolocation, optional
                Geolocation model.  If ``None``, the viewer operates
                in pixel-only mode.
            """
            try:
                self._viewer.open_reader(reader, geolocation=geolocation)
                title = getattr(reader, 'filepath', None)
                if title is not None:
                    self.setWindowTitle(f"GRDK Viewer \u2014 {title}")
                else:
                    self.setWindowTitle("GRDK Viewer \u2014 [reader]")
                self._update_metadata_table()
                self._update_remap_state()
                self.statusBar().showMessage("Opened reader")
            except Exception as e:
                QMessageBox.critical(
                    self, "Open Error",
                    f"Could not open reader:\n{e}",
                )

        def set_array(
            self,
            arr: Any,
            geolocation: Optional[Any] = None,
            title: Optional[str] = None,
        ) -> None:
            """Display a pre-loaded numpy array.

            Parameters
            ----------
            arr : np.ndarray
                Image data (2D, 3D channels-first, or complex).
            geolocation : Geolocation, optional
                Geolocation model for coordinate display.
            title : str, optional
                Window title override.  If ``None``, shows array
                shape and dtype.
            """
            try:
                self._viewer.set_array(arr, geolocation=geolocation)
                if title:
                    self.setWindowTitle(f"GRDK Viewer \u2014 {title}")
                else:
                    self.setWindowTitle(
                        f"GRDK Viewer \u2014 ndarray {arr.shape} {arr.dtype}"
                    )
                self._update_metadata_table()
                self._update_remap_state()
                self.statusBar().showMessage(
                    f"Displaying array: {arr.shape} {arr.dtype}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Display Error",
                    f"Could not display array:\n{e}",
                )

        # --- Actions ---

        def _create_actions(self) -> None:
            """Create menu/toolbar actions."""
            self._open_action = QAction("&Open Image...", self)
            self._open_action.setShortcut(QKeySequence.StandardKey.Open)
            self._open_action.triggered.connect(self._on_open)

            self._open_dir_action = QAction("Open &Directory...", self)
            self._open_dir_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
            self._open_dir_action.triggered.connect(self._on_open_dir)

            self._load_vector_action = QAction("Load &GeoJSON...", self)
            self._load_vector_action.setShortcut(QKeySequence("Ctrl+G"))
            self._load_vector_action.triggered.connect(self._on_load_vector)

            self._export_action = QAction("&Export View...", self)
            self._export_action.setShortcut(QKeySequence("Ctrl+E"))
            self._export_action.triggered.connect(self._on_export)

            self._exit_action = QAction("E&xit", self)
            self._exit_action.setShortcut(QKeySequence.StandardKey.Quit)
            self._exit_action.triggered.connect(self.close)

            self._fit_action = QAction("&Fit in View", self)
            self._fit_action.setShortcut(QKeySequence("Ctrl+0"))
            self._fit_action.triggered.connect(
                lambda: self._viewer.canvas.fit_in_view()
            )

            self._zoom_in_action = QAction("Zoom &In", self)
            self._zoom_in_action.setShortcut(QKeySequence.StandardKey.ZoomIn)
            self._zoom_in_action.triggered.connect(
                lambda: self._viewer.canvas.zoom_to(
                    self._viewer.canvas.get_zoom() * 1.5
                )
            )

            self._zoom_out_action = QAction("Zoom &Out", self)
            self._zoom_out_action.setShortcut(QKeySequence.StandardKey.ZoomOut)
            self._zoom_out_action.triggered.connect(
                lambda: self._viewer.canvas.zoom_to(
                    self._viewer.canvas.get_zoom() / 1.5
                )
            )

            self._clear_vectors_action = QAction("&Clear Vectors", self)
            self._clear_vectors_action.triggered.connect(
                self._viewer.clear_vectors
            )

        # --- Menus ---

        def _create_menus(self) -> None:
            """Build the menu bar."""
            file_menu = self.menuBar().addMenu("&File")
            file_menu.addAction(self._open_action)
            file_menu.addAction(self._open_dir_action)
            file_menu.addAction(self._load_vector_action)
            file_menu.addSeparator()
            file_menu.addAction(self._export_action)
            file_menu.addSeparator()
            file_menu.addAction(self._exit_action)

            view_menu = self.menuBar().addMenu("&View")
            view_menu.addAction(self._fit_action)
            view_menu.addAction(self._zoom_in_action)
            view_menu.addAction(self._zoom_out_action)
            view_menu.addSeparator()
            view_menu.addAction(self._clear_vectors_action)
            view_menu.addSeparator()
            view_menu.addAction(self._toolbar.toggleViewAction())
            view_menu.addAction(self._display_dock.toggleViewAction())
            view_menu.addAction(self._metadata_dock.toggleViewAction())

        # --- Toolbar ---

        def _create_toolbar(self) -> None:
            """Build the main toolbar (hidden by default)."""
            self._toolbar = QToolBar("Main", self)
            self._toolbar.setMovable(False)
            self.addToolBar(self._toolbar)

            self._toolbar.addAction(self._open_action)
            self._toolbar.addAction(self._open_dir_action)
            self._toolbar.addAction(self._load_vector_action)
            self._toolbar.addSeparator()
            self._toolbar.addAction(self._fit_action)
            self._toolbar.addAction(self._zoom_in_action)
            self._toolbar.addAction(self._zoom_out_action)
            self._toolbar.addSeparator()
            self._toolbar.addAction(self._export_action)

            self._toolbar.hide()

        # --- Dock widgets ---

        def _create_display_dock(self) -> None:
            """Create the display controls dock widget."""
            self._display_dock = QDockWidget("Display", self)
            self._display_dock.setAllowedAreas(
                Qt.DockWidgetArea.LeftDockWidgetArea
                | Qt.DockWidgetArea.RightDockWidgetArea
            )

            controls = build_display_controls(
                self._display_dock, self._viewer.canvas,
            )
            self._display_dock.setWidget(controls)
            self.addDockWidget(
                Qt.DockWidgetArea.RightDockWidgetArea, self._display_dock,
            )

        def _create_metadata_dock(self) -> None:
            """Create the metadata display dock widget."""
            self._metadata_dock = QDockWidget("Metadata", self)
            self._metadata_dock.setAllowedAreas(
                Qt.DockWidgetArea.LeftDockWidgetArea
                | Qt.DockWidgetArea.RightDockWidgetArea
                | Qt.DockWidgetArea.BottomDockWidgetArea
            )

            self._metadata_table = QTableWidget(0, 2, self._metadata_dock)
            self._metadata_table.setHorizontalHeaderLabels(["Property", "Value"])
            self._metadata_table.horizontalHeader().setStretchLastSection(True)
            self._metadata_table.setEditTriggers(
                QTableWidget.EditTrigger.NoEditTriggers
            )

            self._metadata_dock.setWidget(self._metadata_table)
            self.addDockWidget(
                Qt.DockWidgetArea.RightDockWidgetArea, self._metadata_dock,
            )

        def _update_metadata_table(self) -> None:
            """Populate the metadata table from the current image."""
            self._metadata_table.setRowCount(0)
            meta = self._viewer.metadata
            if meta is None:
                return

            # Collect key-value pairs from metadata
            pairs = []
            for attr in ('format', 'rows', 'cols', 'bands', 'dtype', 'crs'):
                val = meta.get(attr) if hasattr(meta, 'get') else getattr(meta, attr, None)
                if val is not None:
                    pairs.append((attr, str(val)))

            # Add extras if available
            extras = getattr(meta, 'extras', None)
            if isinstance(extras, dict):
                for k, v in extras.items():
                    pairs.append((k, str(v)[:200]))

            # Add geolocation info
            geo = self._viewer.geolocation
            if geo is not None:
                pairs.append(("geolocation", type(geo).__name__))
                try:
                    bounds = geo.get_bounds()
                    if bounds:
                        pairs.append(("bounds", str(bounds)))
                except Exception:
                    pass

            self._metadata_table.setRowCount(len(pairs))
            for i, (key, val) in enumerate(pairs):
                self._metadata_table.setItem(i, 0, QTableWidgetItem(key))
                self._metadata_table.setItem(i, 1, QTableWidgetItem(val))

        # --- Slots ---

        def _on_open(self) -> None:
            """Handle File > Open Image."""
            filepath, _ = QFileDialog.getOpenFileName(
                self, "Open Image", "", _IMAGE_FILTER,
            )
            if filepath:
                self.open_file(filepath)

        def _on_open_dir(self) -> None:
            """Handle File > Open Directory (Sentinel .SAFE, BIOMASS, etc.)."""
            dirpath = QFileDialog.getExistingDirectory(
                self, "Open Image Directory", "",
                QFileDialog.Option.ShowDirsOnly,
            )
            if dirpath:
                self.open_file(dirpath)

        def _on_band_info_changed(self, band_info: list) -> None:
            """Update band selector when band info changes."""
            controls = self._display_dock.widget()
            if hasattr(controls, 'update_band_info'):
                controls.update_band_info(band_info)

        def _update_remap_state(self) -> None:
            """Enable remap controls only for SAR imagery."""
            controls = self._display_dock.widget()
            if not hasattr(controls, 'set_remap_enabled'):
                return
            reader = self._viewer._reader
            if reader is None:
                controls.set_remap_enabled(False)
                return
            is_sar = False
            try:
                from grdl.IO.sar.sicd import SICDReader
                if isinstance(reader, SICDReader):
                    is_sar = True
            except ImportError:
                pass
            try:
                from grdl.IO.sar.biomass import BIOMASSL1Reader
                if isinstance(reader, BIOMASSL1Reader):
                    is_sar = True
            except ImportError:
                pass
            try:
                from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader
                if isinstance(reader, Sentinel1SLCReader):
                    is_sar = True
            except ImportError:
                pass
            try:
                from grdl.IO.sar.sidd import SIDDReader
                if isinstance(reader, SIDDReader):
                    is_sar = True
            except ImportError:
                pass
            # Also check for complex data as a general SAR indicator
            if not is_sar and reader is not None:
                try:
                    import numpy as np
                    dtype = reader.get_dtype()
                    if np.issubdtype(dtype, np.complexfloating):
                        is_sar = True
                except Exception:
                    pass
            controls.set_remap_enabled(is_sar)

        def _on_load_vector(self) -> None:
            """Handle File > Load GeoJSON."""
            filepath, _ = QFileDialog.getOpenFileName(
                self, "Load GeoJSON", "", _VECTOR_FILTER,
            )
            if filepath:
                try:
                    self._viewer.load_vector(filepath)
                    self.statusBar().showMessage(f"Loaded vector: {filepath}")
                except Exception as e:
                    QMessageBox.critical(
                        self, "Vector Error",
                        f"Could not load GeoJSON:\n{filepath}\n\n{e}",
                    )

        def _on_export(self) -> None:
            """Handle File > Export View."""
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Export View", "", _EXPORT_FILTER,
            )
            if filepath:
                try:
                    self._viewer.export_view(filepath)
                    self.statusBar().showMessage(f"Exported: {filepath}")
                except Exception as e:
                    QMessageBox.critical(
                        self, "Export Error",
                        f"Could not export view:\n{filepath}\n\n{e}",
                    )


    def main() -> None:
        """Entry point for the grdk-viewer command."""
        args = _build_arg_parser().parse_args()

        app = QApplication(sys.argv)
        window = ViewerMainWindow()

        if args.file is not None:
            window.open_file(args.file)

        window.show()
        sys.exit(app.exec())

else:

    class ViewerMainWindow:  # type: ignore[no-redef]
        """Stub when Qt is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("Qt is required for ViewerMainWindow")

    def main() -> None:
        """Stub entry point."""
        _build_arg_parser().parse_args()
        print("Error: PyQt6 is required. Install with: pip install PyQt6")
        sys.exit(1)
