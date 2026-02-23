# -*- coding: utf-8 -*-
"""
ViewerMainWindow - Top-level application window for the geospatial viewer.

Standalone QMainWindow assembling DualGeoViewer with menu bar, toolbar,
per-pane display controls docks, and metadata dock.  Provides a ``main()``
entry point for command-line invocation.

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
2026-02-20
"""

# Standard library
import argparse
import logging
import sys
from typing import Any, Optional

_log = logging.getLogger("grdk.main_window")

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
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: WARNING).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help="Write log output to a file (in addition to stderr).",
    )
    return parser


if _QT_AVAILABLE:
    from grdk.viewers.dual_viewer import DualGeoViewer
    from grdk.viewers.image_canvas import DisplaySettings
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

        Supports single and dual (side-by-side) pane modes with
        independent display controls per pane and synchronized
        pan/zoom navigation.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget.
        """

        def __init__(self, parent: Optional[Any] = None) -> None:
            super().__init__(parent)

            self.setWindowTitle("GRDK Viewer")
            self.resize(1200, 800)

            # Central widget — dual viewer (starts in single mode)
            self._viewer = DualGeoViewer(self)
            self.setCentralWidget(self._viewer)

            # Build UI (docks + toolbar before menus so toggleViewAction works)
            self._create_actions()
            self._create_left_display_dock()
            self._create_right_display_dock()
            self._create_metadata_dock()
            self._create_toolbar()
            self._create_menus()

            # Guard for polarization swaps (prevents re-entrant loops)
            self._switching_pol: bool = False

            # Per-pane polarization names — tracks band/pol identity for
            # arrays that no longer have a reader (e.g. after ortho).
            # {pane: ['HH', 'HV', 'VH', 'VV']} or {pane: None}
            self._pane_pol_names: dict = {0: None, 1: None}

            # Wire signals
            self._viewer.band_info_changed.connect(
                self._on_band_info_changed,
            )
            self._viewer.pane_band_info_changed.connect(
                self._on_pane_band_info_changed,
            )
            self._viewer.active_pane_changed.connect(
                self._on_active_pane_changed,
            )
            self._viewer.mode_changed.connect(self._on_mode_changed)

            # Update status bar when display settings change in dual mode
            self._viewer.left_viewer.canvas.display_settings_changed.connect(
                lambda _: self._update_dual_status(),
            )
            self._viewer.right_viewer.canvas.display_settings_changed.connect(
                lambda _: self._update_dual_status(),
            )

            # Polarization swap: when user selects a different pol in
            # the band combo, swap the reader to load that polarization
            self._viewer.left_viewer.canvas.display_settings_changed.connect(
                lambda s: self._on_pol_swap_check(0, s),
            )
            self._viewer.right_viewer.canvas.display_settings_changed.connect(
                lambda s: self._on_pol_swap_check(1, s),
            )

            # Status bar
            self.statusBar().showMessage("Ready")

        # --- Public API ---

        def open_file(self, filepath: str, pane: int = 0) -> None:
            """Open an image file.

            Parameters
            ----------
            filepath : str
                Path to the image file.
            pane : int
                Target pane (0=left, 1=right).
            """
            _log.info("open_file(%r, pane=%d)", filepath, pane)
            try:
                self._viewer.open_file(filepath, pane=pane)
                self.setWindowTitle(f"GRDK Viewer \u2014 {filepath}")
                self._update_metadata_table()
                self._update_remap_state()
                self._sync_display_controls(pane)
                self._update_colorbar_state(pane)
                self._update_tools_state()
                self._update_pane_pol_names(pane)
                self.statusBar().showMessage(f"Opened: {filepath}")
                _log.info("open_file: success")
            except Exception as e:
                _log.error("open_file failed: %s", e, exc_info=True)
                QMessageBox.critical(
                    self, "Open Error",
                    f"Could not open file:\n{filepath}\n\n{e}",
                )

        def open_reader(
            self,
            reader: Any,
            geolocation: Optional[Any] = None,
            pane: int = 0,
        ) -> None:
            """Open an image from a grdl ImageReader.

            Parameters
            ----------
            reader : ImageReader
                grdl reader instance.
            geolocation : Geolocation, optional
                Geolocation model.  If ``None``, the viewer operates
                in pixel-only mode.
            pane : int
                Target pane (0=left, 1=right).
            """
            _log.info("open_reader(%s, pane=%d)", type(reader).__name__, pane)
            try:
                self._viewer.open_reader(
                    reader, geolocation=geolocation, pane=pane,
                )
                title = getattr(reader, 'filepath', None)
                if title is not None:
                    self.setWindowTitle(f"GRDK Viewer \u2014 {title}")
                else:
                    self.setWindowTitle("GRDK Viewer \u2014 [reader]")
                self._update_metadata_table()
                self._update_remap_state()
                self._sync_display_controls(pane)
                self._update_colorbar_state(pane)
                self._update_tools_state()
                self._update_pane_pol_names(pane)
                self.statusBar().showMessage("Opened reader")
            except Exception as e:
                _log.error("open_reader failed: %s", e, exc_info=True)
                QMessageBox.critical(
                    self, "Open Error",
                    f"Could not open reader:\n{e}",
                )

        def set_array(
            self,
            arr: Any,
            geolocation: Optional[Any] = None,
            title: Optional[str] = None,
            pane: int = 0,
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
            pane : int
                Target pane (0=left, 1=right).
            """
            _log.info("set_array(shape=%s, dtype=%s, pane=%d)", arr.shape, arr.dtype, pane)
            try:
                self._viewer.set_array(
                    arr, geolocation=geolocation, pane=pane,
                )
                if title:
                    self.setWindowTitle(f"GRDK Viewer \u2014 {title}")
                else:
                    self.setWindowTitle(
                        f"GRDK Viewer \u2014 ndarray {arr.shape} {arr.dtype}"
                    )
                self._update_metadata_table()
                self._update_remap_state()
                self._update_tools_state()
                self._pane_pol_names[pane] = None
                self.statusBar().showMessage(
                    f"Displaying array: {arr.shape} {arr.dtype}"
                )
            except Exception as e:
                _log.error("set_array failed: %s", e, exc_info=True)
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

            self._open_left_action = QAction(
                "Open in &Left Pane...", self,
            )
            self._open_left_action.setShortcut(QKeySequence("Ctrl+Shift+L"))
            self._open_left_action.triggered.connect(self._on_open_left)
            self._open_left_action.setEnabled(False)

            self._open_left_dir_action = QAction(
                "Open Directory in Left Pane...", self,
            )
            self._open_left_dir_action.triggered.connect(
                self._on_open_left_dir,
            )
            self._open_left_dir_action.setEnabled(False)

            self._open_right_action = QAction(
                "Open in &Right Pane...", self,
            )
            self._open_right_action.setShortcut(QKeySequence("Ctrl+Shift+R"))
            self._open_right_action.triggered.connect(self._on_open_right)
            self._open_right_action.setEnabled(False)

            self._open_right_dir_action = QAction(
                "Open Directory in Right Pane...", self,
            )
            self._open_right_dir_action.triggered.connect(
                self._on_open_right_dir,
            )
            self._open_right_dir_action.setEnabled(False)

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
                lambda: self._viewer.active_canvas.fit_in_view()
            )

            self._zoom_in_action = QAction("Zoom &In", self)
            self._zoom_in_action.setShortcut(QKeySequence.StandardKey.ZoomIn)
            self._zoom_in_action.triggered.connect(
                lambda: self._viewer.active_canvas.zoom_to(
                    self._viewer.active_canvas.get_zoom() * 1.5
                )
            )

            self._zoom_out_action = QAction("Zoom &Out", self)
            self._zoom_out_action.setShortcut(
                QKeySequence.StandardKey.ZoomOut,
            )
            self._zoom_out_action.triggered.connect(
                lambda: self._viewer.active_canvas.zoom_to(
                    self._viewer.active_canvas.get_zoom() / 1.5
                )
            )

            self._clear_vectors_action = QAction("&Clear Vectors", self)
            self._clear_vectors_action.triggered.connect(
                self._viewer.clear_vectors
            )

            # Dual view toggle
            self._toggle_dual_action = QAction("&Dual View", self)
            self._toggle_dual_action.setShortcut(QKeySequence("Ctrl+D"))
            self._toggle_dual_action.setCheckable(True)
            self._toggle_dual_action.setChecked(False)
            self._toggle_dual_action.toggled.connect(self._on_toggle_dual)

            # Tools
            self._ortho_action = QAction("&Orthorectify", self)
            self._ortho_action.setShortcut(QKeySequence("Ctrl+T"))
            self._ortho_action.setEnabled(False)
            self._ortho_action.triggered.connect(self._on_orthorectify)

            self._rgb_action = QAction("Combine to &RGB", self)
            self._rgb_action.setShortcut(QKeySequence("Ctrl+Shift+D"))
            self._rgb_action.setEnabled(False)
            self._rgb_action.triggered.connect(self._on_combine_rgb)

        # --- Menus ---

        def _create_menus(self) -> None:
            """Build the menu bar."""
            file_menu = self.menuBar().addMenu("&File")
            file_menu.addAction(self._open_action)
            file_menu.addAction(self._open_dir_action)
            file_menu.addSeparator()
            file_menu.addAction(self._open_left_action)
            file_menu.addAction(self._open_left_dir_action)
            file_menu.addSeparator()
            file_menu.addAction(self._open_right_action)
            file_menu.addAction(self._open_right_dir_action)
            file_menu.addSeparator()
            file_menu.addAction(self._load_vector_action)
            file_menu.addSeparator()
            file_menu.addAction(self._export_action)
            file_menu.addSeparator()
            file_menu.addAction(self._exit_action)

            tools_menu = self.menuBar().addMenu("&Tools")
            tools_menu.addAction(self._ortho_action)
            tools_menu.addAction(self._rgb_action)

            view_menu = self.menuBar().addMenu("&View")
            view_menu.addAction(self._fit_action)
            view_menu.addAction(self._zoom_in_action)
            view_menu.addAction(self._zoom_out_action)
            view_menu.addSeparator()
            view_menu.addAction(self._toggle_dual_action)
            view_menu.addSeparator()
            view_menu.addAction(self._clear_vectors_action)
            view_menu.addSeparator()
            view_menu.addAction(self._toolbar.toggleViewAction())
            view_menu.addAction(self._left_display_dock.toggleViewAction())
            view_menu.addAction(self._right_display_dock.toggleViewAction())
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
            self._toolbar.addSeparator()
            self._toolbar.addAction(self._toggle_dual_action)

            self._toolbar.hide()

        # --- Dock widgets ---

        def _create_left_display_dock(self) -> None:
            """Create the left-pane display controls dock widget."""
            self._left_display_dock = QDockWidget("Display", self)
            self._left_display_dock.setAllowedAreas(
                Qt.DockWidgetArea.LeftDockWidgetArea
                | Qt.DockWidgetArea.RightDockWidgetArea
            )

            controls = build_display_controls(
                self._left_display_dock,
                self._viewer.left_viewer.canvas,
            )
            self._left_display_dock.setWidget(controls)
            self.addDockWidget(
                Qt.DockWidgetArea.RightDockWidgetArea,
                self._left_display_dock,
            )

            # Wire colorbar checkbox to left viewer's colorbar
            cb = getattr(controls, 'colorbar_checkbox', None)
            if cb is not None:
                cb.toggled.connect(
                    self._viewer.left_viewer.colorbar.setVisible,
                )

        def _create_right_display_dock(self) -> None:
            """Create the right-pane display controls dock widget."""
            self._right_display_dock = QDockWidget("Display (Right)", self)
            self._right_display_dock.setAllowedAreas(
                Qt.DockWidgetArea.LeftDockWidgetArea
                | Qt.DockWidgetArea.RightDockWidgetArea
            )

            controls = build_display_controls(
                self._right_display_dock,
                self._viewer.right_viewer.canvas,
            )
            self._right_display_dock.setWidget(controls)
            self.addDockWidget(
                Qt.DockWidgetArea.RightDockWidgetArea,
                self._right_display_dock,
            )

            # Wire colorbar checkbox to right viewer's colorbar
            cb = getattr(controls, 'colorbar_checkbox', None)
            if cb is not None:
                cb.toggled.connect(
                    self._viewer.right_viewer.colorbar.setVisible,
                )

            # Hidden by default (single mode)
            self._right_display_dock.hide()

        def _create_metadata_dock(self) -> None:
            """Create the metadata display dock widget."""
            self._metadata_dock = QDockWidget("Metadata", self)
            self._metadata_dock.setAllowedAreas(
                Qt.DockWidgetArea.LeftDockWidgetArea
                | Qt.DockWidgetArea.RightDockWidgetArea
                | Qt.DockWidgetArea.BottomDockWidgetArea
            )

            self._metadata_table = QTableWidget(0, 2, self._metadata_dock)
            self._metadata_table.setHorizontalHeaderLabels(
                ["Property", "Value"],
            )
            self._metadata_table.horizontalHeader().setStretchLastSection(True)
            self._metadata_table.setEditTriggers(
                QTableWidget.EditTrigger.NoEditTriggers
            )

            self._metadata_dock.setWidget(self._metadata_table)
            self.addDockWidget(
                Qt.DockWidgetArea.RightDockWidgetArea, self._metadata_dock,
            )

        def _update_metadata_table(self) -> None:
            """Populate the metadata table from the active viewer."""
            self._metadata_table.setRowCount(0)
            meta = self._viewer.metadata
            if meta is None:
                return

            # Collect key-value pairs from metadata
            pairs = []
            for attr in ('format', 'rows', 'cols', 'bands', 'dtype', 'crs'):
                val = (
                    meta.get(attr)
                    if hasattr(meta, 'get')
                    else getattr(meta, attr, None)
                )
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

        # --- Multiband prompt ---

        def _offer_dual_for_multiband(self, filepath: str) -> None:
            """Prompt the user to show multiband or multi-pol images in dual mode.

            Handles two cases:
            1. Multi-band readers (e.g. BIOMASS) — both bands come from the
               same reader, selected via ``band_index``.
            2. Multi-polarization SAR (Sentinel-1, TerraSAR-X) — each
               polarization requires a separate reader instance.

            Does nothing if already in dual mode or if fewer than 2
            bands/polarizations are available.

            Parameters
            ----------
            filepath : str
                Path to the image file (for re-opening in right pane).
            """
            if self._viewer.mode == "dual":
                return

            # Check for multi-polarization SAR (separate reader per pol)
            reader = self._viewer.left_viewer._reader
            available_pols = self._get_available_polarizations(reader)
            if available_pols is not None and len(available_pols) > 1:
                self._offer_dual_for_multipol(filepath, reader, available_pols)
                return

            # Standard multi-band (same reader, different band_index)
            band_info = self._viewer.left_viewer.band_info
            if len(band_info) < 2:
                return

            band0_name = band_info[0].name
            band1_name = band_info[1].name

            result = QMessageBox.question(
                self,
                "Multiband Image",
                f"This image has {len(band_info)} bands.\n\n"
                f"Show as dual display with "
                f"{band0_name} and {band1_name} side by side?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if result != QMessageBox.StandardButton.Yes:
                return

            # Switch to dual mode and open same file in right pane
            self._viewer.set_mode("dual")
            self._viewer.open_file(filepath, pane=1)

            # Sync right dock controls to match auto-settings from loading
            self._update_remap_state()
            self._sync_display_controls(1)
            self._update_colorbar_state(0)
            self._update_colorbar_state(1)

            # Use set_band_index on the display controls — this sets the
            # combo selection AND pushes DisplaySettings to the canvas
            # atomically, so the UI and canvas stay in sync.
            left_controls = self._left_display_dock.widget()
            right_controls = self._right_display_dock.widget()
            if hasattr(left_controls, 'set_band_index'):
                left_controls.set_band_index(0)
            if hasattr(right_controls, 'set_band_index'):
                right_controls.set_band_index(1)

            self._update_metadata_table()
            self.statusBar().showMessage(
                f"Dual view: {band0_name} | {band1_name}",
            )

        @staticmethod
        def _get_available_polarizations(reader: Any) -> Optional[list]:
            """Return available polarizations for multi-pol SAR readers.

            Returns None for non-SAR or single-pol readers.
            """
            if reader is None:
                return None
            get_pols = getattr(reader, 'get_available_polarizations', None)
            if get_pols is None:
                return None
            try:
                pols = get_pols()
                return pols if len(pols) > 1 else None
            except Exception:
                return None

        def _offer_dual_for_multipol(
            self,
            filepath: str,
            reader: Any,
            pols: list,
        ) -> None:
            """Offer dual view for multi-polarization SAR products.

            Each polarization requires a separate reader instance.
            Opens the first two polarizations side by side.

            Parameters
            ----------
            filepath : str
                Path to the SAR product.
            reader : ImageReader
                Current (left pane) reader.
            pols : list
                Available polarization strings (e.g. ['VV', 'VH']).
            """
            # Determine which pol the left pane currently has
            current_pol = self._get_reader_polarization(reader)
            if current_pol and current_pol in pols:
                # Offer the first OTHER polarization
                other_pols = [p for p in pols if p != current_pol]
                second_pol = other_pols[0] if other_pols else pols[1]
            else:
                current_pol = pols[0]
                second_pol = pols[1]

            result = QMessageBox.question(
                self,
                "Multi-Polarization SAR",
                f"This product has {len(pols)} polarizations: "
                f"{', '.join(pols)}.\n\n"
                f"Show {current_pol} and {second_pol} side by side?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if result != QMessageBox.StandardButton.Yes:
                return

            _log.info(
                "Multi-pol dual view: %s | %s for %s",
                current_pol, second_pol, filepath,
            )

            # Switch to dual mode
            self._viewer.set_mode("dual")

            # Create a new reader for the second polarization
            right_reader = self._create_reader_for_pol(
                filepath, reader, second_pol,
            )
            if right_reader is None:
                _log.warning(
                    "Could not create reader for polarization %s",
                    second_pol,
                )
                return

            from grdk.viewers.geo_viewer import create_geolocation
            geo = create_geolocation(right_reader)
            self._viewer.open_reader(right_reader, geolocation=geo, pane=1)

            self._update_remap_state()
            self._sync_display_controls(1)
            self._update_colorbar_state(0)
            self._update_colorbar_state(1)
            self._update_metadata_table()
            self.statusBar().showMessage(
                f"Dual view: {current_pol} | {second_pol}",
            )

        @staticmethod
        def _get_reader_polarization(reader: Any) -> Optional[str]:
            """Extract the current polarization from a SAR reader."""
            # TerraSAR-X
            pol = getattr(reader, '_requested_polarization', None)
            if pol:
                return pol
            # Sentinel-1 SLC
            meta = getattr(reader, 'metadata', None)
            if meta is not None:
                swath_info = (
                    meta.get('swath_info')
                    if hasattr(meta, 'get')
                    else getattr(meta, 'swath_info', None)
                )
                if swath_info:
                    return getattr(swath_info, 'polarization', None)
            return None

        @staticmethod
        def _create_reader_for_pol(
            filepath: str, reader: Any, polarization: str,
        ) -> Any:
            """Create a new reader instance for a specific polarization.

            Parameters
            ----------
            filepath : str
                Path to the SAR product.
            reader : ImageReader
                Existing reader (used to determine reader class).
            polarization : str
                Target polarization (e.g. 'VH', 'VV').

            Returns
            -------
            ImageReader or None
                New reader for the requested polarization.
            """
            try:
                from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader
                if isinstance(reader, Sentinel1SLCReader):
                    return Sentinel1SLCReader(
                        filepath, polarization=polarization,
                    )
            except (ImportError, Exception):
                pass

            try:
                from grdl.IO.sar.terrasar import TerraSARReader
                if isinstance(reader, TerraSARReader):
                    return TerraSARReader(
                        filepath, polarization=polarization,
                    )
            except (ImportError, Exception):
                pass

            return None

        # --- Slots ---

        def _on_open(self) -> None:
            """Handle File > Open Image.

            Resets to single mode, clears the right pane, loads into the
            left pane, then offers dual view for multiband images.
            """
            filepath, _ = QFileDialog.getOpenFileName(
                self, "Open Image", "", _IMAGE_FILTER,
            )
            if filepath:
                self._open_fresh(filepath)

        def _on_open_dir(self) -> None:
            """Handle File > Open Directory.

            Resets to single mode, clears the right pane, loads into the
            left pane, then offers dual view for multiband images.
            """
            dirpath = QFileDialog.getExistingDirectory(
                self, "Open Image Directory", "",
                QFileDialog.Option.ShowDirsOnly,
            )
            if dirpath:
                self._open_fresh(dirpath)

        def _open_fresh(self, filepath: str) -> None:
            """Reset to single mode, clear right pane, and open a file.

            After loading, offers the multiband dual-view prompt.
            """
            _log.info("_open_fresh(%r)", filepath)
            # Reset to single mode (clears right pane visibility)
            if self._viewer.mode == "dual":
                self._viewer.set_mode("single")
            self.open_file(filepath, pane=0)
            self._offer_dual_for_multiband(filepath)

        def _on_open_left(self) -> None:
            """Handle File > Open in Left Pane."""
            filepath, _ = QFileDialog.getOpenFileName(
                self, "Open Image (Left Pane)", "", _IMAGE_FILTER,
            )
            if filepath:
                self.open_file(filepath, pane=0)

        def _on_open_left_dir(self) -> None:
            """Handle File > Open Directory in Left Pane."""
            dirpath = QFileDialog.getExistingDirectory(
                self, "Open Image Directory (Left Pane)", "",
                QFileDialog.Option.ShowDirsOnly,
            )
            if dirpath:
                self.open_file(dirpath, pane=0)

        def _on_open_right(self) -> None:
            """Handle File > Open in Right Pane."""
            filepath, _ = QFileDialog.getOpenFileName(
                self, "Open Image (Right Pane)", "", _IMAGE_FILTER,
            )
            if filepath:
                self.open_file(filepath, pane=1)

        def _on_open_right_dir(self) -> None:
            """Handle File > Open Directory in Right Pane."""
            dirpath = QFileDialog.getExistingDirectory(
                self, "Open Image Directory (Right Pane)", "",
                QFileDialog.Option.ShowDirsOnly,
            )
            if dirpath:
                self.open_file(dirpath, pane=1)

        def _on_toggle_dual(self, checked: bool) -> None:
            """Handle View > Dual View toggle."""
            _log.info("Toggle dual view: %s", checked)
            self._viewer.set_mode("dual" if checked else "single")
            if checked:
                self._populate_right_pane_if_empty()

        def _on_mode_changed(self, mode: str) -> None:
            """Handle dual viewer mode changes."""
            _log.info("Mode changed: %s", mode)
            is_dual = mode == "dual"
            self._open_left_action.setEnabled(is_dual)
            self._open_left_dir_action.setEnabled(is_dual)
            self._open_right_action.setEnabled(is_dual)
            self._open_right_dir_action.setEnabled(is_dual)
            if is_dual:
                self._left_display_dock.setWindowTitle("Display (Left)")
                self._right_display_dock.show()
            else:
                self._left_display_dock.setWindowTitle("Display")
                self._right_display_dock.hide()
            # Keep toggle in sync (if mode changed programmatically)
            self._toggle_dual_action.blockSignals(True)
            self._toggle_dual_action.setChecked(is_dual)
            self._toggle_dual_action.blockSignals(False)
            self._update_tools_state()

        def _populate_right_pane_if_empty(self) -> None:
            """Open the left pane's file in the right pane when entering dual mode.

            If the left pane has a loaded image and the right pane is
            empty, re-opens the file from disk into the right pane.
            This avoids sharing reader objects between panes, which would
            cause tile-loading failures when one pane's reader is closed.
            """
            left = self._viewer.left_viewer
            right = self._viewer.right_viewer

            # Skip if the right pane already has content
            if right._reader is not None or right.canvas.source_array is not None:
                return

            # Re-open the file from disk (never share reader objects)
            if left._reader is not None:
                filepath = getattr(left._reader, 'filepath', None)
                _log.info("_populate_right_pane_if_empty: filepath=%s", filepath)
                if filepath is not None:
                    try:
                        self._viewer.open_file(str(filepath), pane=1)
                        self._update_remap_state()
                        self._sync_display_controls(1)
                        self._update_colorbar_state(1)
                    except Exception:
                        pass

        def _on_active_pane_changed(self, pane: int) -> None:
            """Handle active pane switch — refresh metadata and remap."""
            self._update_metadata_table()
            self._update_remap_state()
            self._update_tools_state()

        def _update_dual_status(self) -> None:
            """Update status bar with current band names in dual mode."""
            if self._viewer.mode != "dual":
                return

            def _band_label(viewer: Any) -> str:
                settings = viewer.canvas.display_settings
                idx = settings.band_index
                info = viewer.band_info
                if idx is not None and idx < len(info):
                    return info[idx].name
                return "Auto"

            left = _band_label(self._viewer.left_viewer)
            right = _band_label(self._viewer.right_viewer)
            self.statusBar().showMessage(f"Dual view: {left} | {right}")

        def _on_band_info_changed(self, band_info: list) -> None:
            """Update band selector when active pane's band info changes."""
            # Legacy signal — routed to active pane's dock
            self._on_pane_band_info_changed(
                self._viewer.active_pane, band_info,
            )

        def _on_pane_band_info_changed(
            self, pane: int, band_info: list,
        ) -> None:
            """Update band selector for a specific pane's dock."""
            dock = (
                self._left_display_dock if pane == 0
                else self._right_display_dock
            )
            controls = dock.widget()
            if hasattr(controls, 'update_band_info'):
                controls.update_band_info(band_info)
            self._update_colorbar_state(pane)

        def _sync_display_controls(self, pane: int = 0) -> None:
            """Sync display control widgets to match the canvas settings.

            Called after loading a file so that auto-applied settings
            (e.g., 2-98% percentile for SAR, band 0 for multi-band SAR)
            are reflected in the UI.  Without this, the first user
            interaction with any control would overwrite the auto-settings.

            Parameters
            ----------
            pane : int
                Which pane's dock to sync (0=left, 1=right).
            """
            dock = (
                self._left_display_dock if pane == 0
                else self._right_display_dock
            )
            controls = dock.widget()
            if hasattr(controls, 'sync_from_settings'):
                controls.sync_from_settings()

        def _update_colorbar_state(self, pane: int = 0) -> None:
            """Enable or disable colorbar based on RGB vs scalar display.

            The colorbar is only meaningful for scalar (single-band) data
            with a colormap applied.  For RGB display (band_index=None
            with 3+ bands), the colorbar is disabled and hidden.

            Parameters
            ----------
            pane : int
                Which pane to update (0=left, 1=right).
            """
            viewer = (
                self._viewer.left_viewer if pane == 0
                else self._viewer.right_viewer
            )
            dock = (
                self._left_display_dock if pane == 0
                else self._right_display_dock
            )
            controls = dock.widget()
            if not hasattr(controls, 'set_colorbar_enabled'):
                return

            # Determine if the current settings produce RGB output
            settings = viewer.canvas.display_settings
            band_info = viewer.band_info
            is_rgb = (
                settings.band_index is None
                and len(band_info) >= 3
            )
            controls.set_colorbar_enabled(not is_rgb)

        def _update_remap_state(self) -> None:
            """Enable remap controls only for SAR imagery."""
            # Update left pane remap state
            self._update_remap_for_dock(
                self._left_display_dock,
                self._viewer.left_viewer._reader,
            )
            # Update right pane remap state
            self._update_remap_for_dock(
                self._right_display_dock,
                self._viewer.right_viewer._reader,
            )

        @staticmethod
        def _update_remap_for_dock(dock: Any, reader: Any) -> None:
            """Update remap enabled state for a display dock."""
            controls = dock.widget()
            if not hasattr(controls, 'set_remap_enabled'):
                return
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
            if not is_sar and reader is not None:
                try:
                    import numpy as np
                    dtype = reader.get_dtype()
                    if np.issubdtype(dtype, np.complexfloating):
                        is_sar = True
                except Exception:
                    pass
            controls.set_remap_enabled(is_sar)

        def _on_pol_swap_check(self, pane: int, settings: Any) -> None:
            """Swap the reader when the user selects a different polarization.

            Multi-pol SAR readers (Sentinel-1, TerraSAR-X) produce one
            band per reader instance.  The band combo lists all
            available polarizations.  When the user selects a pol
            different from the one currently loaded, this method creates
            a new reader for that pol and reloads the pane.

            Parameters
            ----------
            pane : int
                Pane that changed (0=left, 1=right).
            settings : DisplaySettings
                New display settings from the canvas.
            """
            if self._switching_pol:
                return

            viewer = (
                self._viewer.left_viewer if pane == 0
                else self._viewer.right_viewer
            )
            reader = viewer._reader
            if reader is None:
                return

            # Only handle Sentinel-1 SLC and TerraSAR-X
            is_multipol_sar = False
            try:
                from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader
                if isinstance(reader, Sentinel1SLCReader):
                    is_multipol_sar = True
            except ImportError:
                pass
            if not is_multipol_sar:
                try:
                    from grdl.IO.sar.terrasar import TerraSARReader
                    if isinstance(reader, TerraSARReader):
                        is_multipol_sar = True
                except ImportError:
                    pass
            if not is_multipol_sar:
                return

            # Check if selected band_index maps to a different pol
            band_index = settings.band_index
            if band_index is None:
                return

            band_info = viewer.band_info
            if band_index >= len(band_info):
                return

            target_pol = band_info[band_index].name
            current_pol = self._get_reader_polarization(reader)
            if target_pol == current_pol:
                return  # Same pol, no swap needed

            filepath = getattr(reader, 'filepath', None)
            if filepath is None:
                return

            _log.info(
                "Pol swap: pane %d, %s -> %s", pane, current_pol, target_pol,
            )

            self._switching_pol = True
            try:
                new_reader = self._create_reader_for_pol(
                    str(filepath), reader, target_pol,
                )
                if new_reader is None:
                    _log.warning(
                        "Could not create reader for pol %s", target_pol,
                    )
                    return

                from grdk.viewers.geo_viewer import create_geolocation
                geo = create_geolocation(new_reader)
                self._viewer.open_reader(
                    new_reader, geolocation=geo, pane=pane,
                )
                self._update_remap_state()
                self._sync_display_controls(pane)
                self._update_colorbar_state(pane)
                self._update_tools_state()
            except Exception as e:
                _log.error("Pol swap failed: %s", e, exc_info=True)
            finally:
                self._switching_pol = False

        @staticmethod
        def _get_source_array(viewer: Any) -> Any:
            """Retrieve the source array from a viewer (if any).

            Handles both small arrays (canvas.source_array) and large
            tiled arrays (canvas._reader._arr from internal _ArrayReader).

            Returns
            -------
            np.ndarray or None
            """
            arr = viewer.canvas.source_array
            if arr is not None:
                return arr
            # Tiled canvas stores large arrays in an _ArrayReader wrapper
            internal = getattr(viewer.canvas, '_reader', None)
            if internal is not None:
                return getattr(internal, '_arr', None)
            return None

        def _update_pane_pol_names(self, pane: int) -> None:
            """Read and cache polarization names from the pane's reader."""
            viewer = (
                self._viewer.left_viewer if pane == 0
                else self._viewer.right_viewer
            )
            reader = viewer._reader
            if reader is None:
                self._pane_pol_names[pane] = None
                return
            # BIOMASS: native multi-pol reader
            pols = getattr(reader, 'polarizations', None)
            if pols and len(pols) >= 2:
                self._pane_pol_names[pane] = list(pols)
                return
            # Single-pol SAR (TerraSAR-X, Sentinel-1)
            pol = self._get_reader_polarization(reader)
            if pol:
                self._pane_pol_names[pane] = [pol]
                return
            self._pane_pol_names[pane] = None

        # --- Tools state management ---

        def _update_tools_state(self) -> None:
            """Enable/disable tool actions based on current data."""
            # Ortho: enabled when any pane has data AND geolocation
            ortho_ok = False
            for v in (self._viewer.left_viewer,
                      self._viewer.right_viewer):
                has_data = (
                    v._reader is not None
                    or self._get_source_array(v) is not None
                )
                if has_data and v._geolocation is not None:
                    ortho_ok = True
                    break
            self._ortho_action.setEnabled(ortho_ok)

            # RGB: enabled when at least 2 distinct bands are available
            self._rgb_action.setEnabled(
                self._count_available_bands() >= 2,
            )

        # --- RGB Combine ---

        def _count_available_bands(self) -> int:
            """Count distinct band/polarization names across all sources.

            Considers both panes: reader polarizations and cached
            ``_pane_pol_names`` (which survive orthorectification).

            Returns
            -------
            int
                Number of distinct band names available for RGB
                combination.
            """
            names: set = set()

            for pane_idx in (0, 1):
                viewer = (
                    self._viewer.left_viewer if pane_idx == 0
                    else self._viewer.right_viewer
                )
                reader = viewer._reader

                # Multi-band reader (BIOMASS)
                pols = getattr(reader, 'polarizations', None)
                if pols:
                    names.update(pols)
                elif reader is not None:
                    # Single-pol reader (TerraSAR-X, Sentinel-1)
                    pol = self._get_reader_polarization(reader)
                    if pol:
                        names.add(pol)

                # Cached pol names (post-ortho arrays)
                cached = self._pane_pol_names.get(pane_idx)
                if cached:
                    names.update(cached)

            return len(names)

        def _gather_rgb_bands(
            self,
        ) -> tuple:
            """Load all available bands as magnitude arrays.

            Scans both panes for band data from readers and post-ortho
            arrays.  Complex data is converted to magnitude.

            Returns
            -------
            tuple of (dict, geolocation, int)
                ``band_data`` maps band name → 2-D float32 array.
                ``geolocation`` from the active pane.
                ``pane`` index of the active pane.

            Raises
            ------
            RuntimeError
                If fewer than 2 bands are found.
            """
            import numpy as np

            pane = self._viewer.active_pane
            band_data: dict = {}
            min_rows = None
            min_cols = None

            for pane_idx in (0, 1):
                viewer = (
                    self._viewer.left_viewer if pane_idx == 0
                    else self._viewer.right_viewer
                )
                reader = viewer._reader

                # --- Multi-band reader (BIOMASS) ---
                pols = getattr(reader, 'polarizations', None)
                if pols and len(pols) >= 2:
                    shape = reader.get_shape()
                    rows, cols = shape[0], shape[1]
                    for i, pol_name in enumerate(pols):
                        if pol_name not in band_data:
                            chip = reader.read_chip(
                                0, rows, 0, cols, bands=[i],
                            )
                            arr = np.abs(chip).astype(np.float32)
                            del chip
                            band_data[pol_name] = arr
                            if min_rows is None:
                                min_rows, min_cols = arr.shape
                            else:
                                min_rows = min(min_rows, arr.shape[0])
                                min_cols = min(min_cols, arr.shape[1])
                    continue

                # --- Single-pol reader (TerraSAR-X, Sentinel-1) ---
                if reader is not None:
                    pol_name = self._get_reader_polarization(reader)
                    if pol_name and pol_name not in band_data:
                        shape = reader.get_shape()
                        rows = shape[0]
                        cols = shape[1] if len(shape) > 1 else shape[0]
                        chip = reader.read_chip(0, rows, 0, cols)
                        arr = np.abs(chip).astype(np.float32)
                        del chip
                        band_data[pol_name] = arr
                        if min_rows is None:
                            min_rows, min_cols = arr.shape[-2:]
                        else:
                            min_rows = min(min_rows, arr.shape[-2])
                            min_cols = min(min_cols, arr.shape[-1])
                    continue

                # --- Post-ortho array with cached pol names ---
                cached = self._pane_pol_names.get(pane_idx)
                source = self._get_source_array(viewer)
                if cached and source is not None:
                    for i, pol_name in enumerate(cached):
                        if pol_name not in band_data:
                            if source.ndim == 3 and i < source.shape[0]:
                                arr = source[i].astype(np.float32)
                            elif source.ndim == 2 and len(cached) == 1:
                                arr = source.astype(np.float32)
                            else:
                                continue
                            band_data[pol_name] = arr
                            if min_rows is None:
                                min_rows, min_cols = arr.shape
                            else:
                                min_rows = min(min_rows, arr.shape[0])
                                min_cols = min(min_cols, arr.shape[1])

            if len(band_data) < 2:
                raise RuntimeError(
                    f"Need at least 2 bands for RGB, found "
                    f"{len(band_data)}: {list(band_data.keys())}",
                )

            # Crop all bands to common shape if sizes differ
            if min_rows is not None and min_cols is not None:
                for name in band_data:
                    arr = band_data[name]
                    if (arr.shape[0] != min_rows
                            or arr.shape[1] != min_cols):
                        band_data[name] = arr[:min_rows, :min_cols]

            active_viewer = (
                self._viewer.left_viewer if pane == 0
                else self._viewer.right_viewer
            )
            geo = active_viewer._geolocation

            return band_data, geo, pane

        def _on_combine_rgb(self) -> None:
            """Handle Tools > Combine to RGB.

            Shows a dialog for the user to select which band or band
            combination to assign to each RGB channel, then computes
            and displays the result.
            """
            import numpy as np
            from PyQt6.QtCore import Qt as _Qt
            from PyQt6.QtWidgets import (
                QComboBox,
                QDialog,
                QDialogButtonBox,
                QFormLayout,
                QVBoxLayout,
            )

            # --- 1. Gather band data ---
            QApplication.setOverrideCursor(_Qt.CursorShape.WaitCursor)
            try:
                band_data, geo, pane = self._gather_rgb_bands()
            except Exception as e:
                QApplication.restoreOverrideCursor()
                _log.error("Combine to RGB failed: %s", e, exc_info=True)
                QMessageBox.critical(
                    self, "Combine to RGB",
                    f"Could not gather band data:\n{e}",
                )
                return
            QApplication.restoreOverrideCursor()

            band_names = sorted(band_data.keys())
            _log.info(
                "Combine to RGB: bands=%s, shapes=%s",
                band_names,
                {k: v.shape for k, v in band_data.items()},
            )

            # --- 2. Build channel options ---
            options = list(band_names)
            if 'HH' in band_data and 'VV' in band_data:
                options.append('HH + VV')
                options.append('HH \u2212 VV')
            if 'HV' in band_data and 'VH' in band_data:
                options.append('HV + VH')

            # --- 3. Show channel selection dialog ---
            dlg = QDialog(self)
            dlg.setWindowTitle("Combine Bands to RGB")
            layout = QVBoxLayout(dlg)

            form = QFormLayout()
            combos = {}
            for i, color in enumerate(('Red', 'Green', 'Blue')):
                combo = QComboBox()
                combo.addItems(options)
                # Set default selections to spread across options
                if i < len(options):
                    combo.setCurrentIndex(i)
                combos[color] = combo
                form.addRow(f"{color}:", combo)
            layout.addLayout(form)

            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok
                | QDialogButtonBox.StandardButton.Cancel,
            )
            buttons.accepted.connect(dlg.accept)
            buttons.rejected.connect(dlg.reject)
            layout.addWidget(buttons)

            if dlg.exec() != QDialog.DialogCode.Accepted:
                return

            r_sel = combos['Red'].currentText()
            g_sel = combos['Green'].currentText()
            b_sel = combos['Blue'].currentText()
            _log.info(
                "Combine to RGB: R=%s, G=%s, B=%s", r_sel, g_sel, b_sel,
            )

            # --- 4. Compute RGB ---
            QApplication.setOverrideCursor(_Qt.CursorShape.WaitCursor)
            try:
                r = self._resolve_rgb_channel(r_sel, band_data)
                g = self._resolve_rgb_channel(g_sel, band_data)
                b = self._resolve_rgb_channel(b_sel, band_data)

                rgb = np.stack([
                    self._percentile_stretch(r),
                    self._percentile_stretch(g),
                    self._percentile_stretch(b),
                ], axis=0)
                del r, g, b

                self._display_rgb_result(rgb, geo, pane, "RGB Composite")

            except Exception as e:
                _log.error(
                    "Combine to RGB failed: %s", e, exc_info=True,
                )
                QMessageBox.critical(
                    self, "Combine to RGB Error",
                    f"RGB composite failed:\n{e}",
                )
            finally:
                QApplication.restoreOverrideCursor()

        @staticmethod
        def _resolve_rgb_channel(
            selection: str,
            band_data: dict,
        ) -> 'np.ndarray':
            """Resolve a channel selection string to a 2-D array.

            Parameters
            ----------
            selection : str
                Band name (e.g. ``'HH'``) or combination
                (``'HH + VV'``, ``'HH \u2212 VV'``, ``'HV + VH'``).
            band_data : dict
                Maps band name → 2-D float array.

            Returns
            -------
            np.ndarray
                2-D float array for the selected channel.
            """
            if selection == 'HH + VV':
                return band_data['HH'] + band_data['VV']
            if selection == 'HH \u2212 VV':
                return band_data['HH'] - band_data['VV']
            if selection == 'HV + VH':
                return band_data['HV'] + band_data['VH']
            return band_data[selection]

        # --- Orthorectification ---

        def _on_orthorectify(self) -> None:
            """Handle Tools > Orthorectify.

            In dual mode, orthorectifies both panes together:

            - **BIOMASS multi-band**: reads all bands from the reader,
              orthos as a single ``(bands, rows, cols)`` array, and
              displays the result in both panes with preserved band
              selections.
            - **Separate readers per pane** (Sentinel-1, TerraSAR-X):
              orthos both panes using a shared output grid so they
              remain geo-synced.

            In single mode, orthos the active pane only.
            """
            import numpy as np

            pane = self._viewer.active_pane
            viewer = (
                self._viewer.left_viewer if pane == 0
                else self._viewer.right_viewer
            )
            reader = viewer._reader
            geo = viewer._geolocation

            # Also accept source_array (e.g. post-decomposition RGB)
            source_arr = self._get_source_array(viewer)

            if reader is None and source_arr is None:
                QMessageBox.critical(
                    self, "Orthorectify",
                    "No image loaded in the active pane.",
                )
                return
            if geo is None:
                QMessageBox.critical(
                    self, "Orthorectify",
                    "No geolocation available for the active pane.\n"
                    "Orthorectification requires geolocation metadata.",
                )
                return

            # --- Dual-mode: ortho both panes together ---
            if self._viewer.mode == "dual":
                left_v = self._viewer.left_viewer
                right_v = self._viewer.right_viewer
                left_r = left_v._reader
                right_r = right_v._reader

                # BIOMASS multi-band: both panes share the same reader
                # type with multiple bands — ortho all bands at once.
                try:
                    from grdl.IO.sar.biomass import BIOMASSL1Reader
                    if (isinstance(left_r, BIOMASSL1Reader)
                            and isinstance(right_r, BIOMASSL1Reader)):
                        self._ortho_biomass_all_bands()
                        return
                except ImportError:
                    pass

                # Separate readers per pane: ortho both with shared
                # output grid for geo-sync.
                if (left_r is not None and right_r is not None
                        and left_v._geolocation is not None
                        and right_v._geolocation is not None):
                    self._ortho_dual_separate()
                    return

            # --- Single pane (or fallback) ---
            _log.info(
                "Orthorectify: pane=%d, reader=%s, source_arr=%s",
                pane,
                type(reader).__name__ if reader else None,
                source_arr.shape if source_arr is not None else None,
            )

            pol_names = self._pane_pol_names.get(pane)

            from PyQt6.QtCore import Qt as _Qt
            QApplication.setOverrideCursor(_Qt.CursorShape.WaitCursor)
            try:
                from grdl.image_processing.ortho import OrthoPipeline

                if source_arr is not None and reader is None:
                    _log.info(
                        "Orthorectify: source array %s %s",
                        source_arr.shape, source_arr.dtype,
                    )
                    result = (
                        OrthoPipeline()
                        .with_source_array(source_arr)
                        .with_geolocation(geo)
                        .with_interpolation('bilinear')
                        .run()
                    )
                else:
                    is_complex = False
                    try:
                        dtype = reader.get_dtype()
                        if np.issubdtype(dtype, np.complexfloating):
                            is_complex = True
                    except Exception:
                        pass

                    if is_complex:
                        shape = reader.get_shape()
                        rows = shape[0]
                        cols = shape[1] if len(shape) > 1 else shape[0]
                        _log.info(
                            "Orthorectify: reading complex data %dx%d",
                            rows, cols,
                        )
                        data = reader.read_chip(0, rows, 0, cols)
                        mag = np.abs(data).astype(np.float32)
                        del data

                        result = (
                            OrthoPipeline()
                            .with_source_array(mag)
                            .with_metadata(reader.metadata)
                            .with_geolocation(geo)
                            .with_interpolation('bilinear')
                            .run()
                        )
                        del mag
                    else:
                        result = (
                            OrthoPipeline()
                            .with_reader(reader)
                            .with_geolocation(geo)
                            .with_interpolation('bilinear')
                            .run()
                        )

                _log.info("Orthorectify: result shape=%s", result.data.shape)

                self._viewer.set_array(
                    result.data,
                    geolocation=result.output_grid,
                    pane=pane,
                )
                self._pane_pol_names[pane] = pol_names
                self._sync_display_controls(pane)
                self._update_colorbar_state(pane)
                self._update_remap_state()
                self._update_tools_state()
                self.statusBar().showMessage("Orthorectification complete")

            except Exception as e:
                _log.error("Orthorectify failed: %s", e, exc_info=True)
                QMessageBox.critical(
                    self, "Orthorectify Error",
                    f"Orthorectification failed:\n{e}",
                )
            finally:
                QApplication.restoreOverrideCursor()

        def _ortho_biomass_all_bands(self) -> None:
            """Orthorectify all BIOMASS bands at once for both panes.

            Reads all bands from the BIOMASS reader, converts complex
            data to magnitude, orthorectifies the full ``(bands, rows,
            cols)`` array, and displays the result in both panes with
            their original band selections preserved.
            """
            import numpy as np
            from PyQt6.QtCore import Qt as _Qt

            # Use left viewer's reader (both panes share the same file)
            left_v = self._viewer.left_viewer
            reader = left_v._reader
            geo = left_v._geolocation
            pol_names = list(reader.polarizations)

            # Remember current band selections for each pane
            left_canvas = left_v.canvas
            right_canvas = self._viewer.right_viewer.canvas
            left_band = getattr(
                left_canvas.display_settings, 'band_index', 0,
            )
            right_band = getattr(
                right_canvas.display_settings, 'band_index', 1,
            )
            # Default to 0/1 if None
            if left_band is None:
                left_band = 0
            if right_band is None:
                right_band = 1

            QApplication.setOverrideCursor(_Qt.CursorShape.WaitCursor)
            try:
                from grdl.image_processing.ortho import OrthoPipeline

                shape = reader.get_shape()
                rows, cols = shape[0], shape[1]
                _log.info(
                    "Orthorectify BIOMASS all bands: %dx%d, pols=%s",
                    rows, cols, pol_names,
                )

                # Read ALL bands → (bands, rows, cols) complex
                data = reader.read_chip(0, rows, 0, cols)
                mag = np.abs(data).astype(np.float32)
                del data

                result = (
                    OrthoPipeline()
                    .with_source_array(mag)
                    .with_metadata(reader.metadata)
                    .with_geolocation(geo)
                    .with_interpolation('bilinear')
                    .run()
                )
                del mag

                _log.info(
                    "Orthorectify BIOMASS: result shape=%s",
                    result.data.shape,
                )

                # Display in both panes
                for p in (0, 1):
                    self._viewer.set_array(
                        result.data,
                        geolocation=result.output_grid,
                        pane=p,
                    )
                    self._pane_pol_names[p] = pol_names
                    self._sync_display_controls(p)
                    self._update_colorbar_state(p)

                # Restore band selections via display controls
                left_controls = self._left_display_dock.widget()
                right_controls = self._right_display_dock.widget()
                if hasattr(left_controls, 'set_band_index'):
                    left_controls.set_band_index(left_band)
                if hasattr(right_controls, 'set_band_index'):
                    right_controls.set_band_index(right_band)

                self._update_remap_state()
                self._update_tools_state()
                self.statusBar().showMessage(
                    "Orthorectification complete (all bands)",
                )

            except Exception as e:
                _log.error(
                    "Orthorectify BIOMASS failed: %s", e, exc_info=True,
                )
                QMessageBox.critical(
                    self, "Orthorectify Error",
                    f"Orthorectification failed:\n{e}",
                )
            finally:
                QApplication.restoreOverrideCursor()

        def _ortho_dual_separate(self) -> None:
            """Orthorectify both panes (separate readers) with shared grid.

            Each pane has its own reader (e.g. Sentinel-1 VV in left,
            VH in right).  The first pane is orthorectified to compute
            the output grid, then the second pane reuses that same grid
            so both results are geo-synced.
            """
            import numpy as np
            from PyQt6.QtCore import Qt as _Qt

            QApplication.setOverrideCursor(_Qt.CursorShape.WaitCursor)
            try:
                from grdl.image_processing.ortho import OrthoPipeline

                results = {}
                output_grid = None

                for pane_idx, viewer in [
                    (0, self._viewer.left_viewer),
                    (1, self._viewer.right_viewer),
                ]:
                    reader = viewer._reader
                    geo = viewer._geolocation
                    pol_names = self._pane_pol_names.get(pane_idx)

                    is_complex = False
                    try:
                        dtype = reader.get_dtype()
                        if np.issubdtype(dtype, np.complexfloating):
                            is_complex = True
                    except Exception:
                        pass

                    if is_complex:
                        shape = reader.get_shape()
                        rows = shape[0]
                        cols = shape[1] if len(shape) > 1 else shape[0]
                        _log.info(
                            "Orthorectify dual pane %d: reading complex "
                            "%dx%d", pane_idx, rows, cols,
                        )
                        data = reader.read_chip(0, rows, 0, cols)
                        mag = np.abs(data).astype(np.float32)
                        del data

                        pipeline = (
                            OrthoPipeline()
                            .with_source_array(mag)
                            .with_metadata(reader.metadata)
                            .with_geolocation(geo)
                            .with_interpolation('bilinear')
                        )
                        if output_grid is not None:
                            pipeline.with_output_grid(output_grid)

                        result = pipeline.run()
                        del mag
                    else:
                        pipeline = (
                            OrthoPipeline()
                            .with_reader(reader)
                            .with_geolocation(geo)
                            .with_interpolation('bilinear')
                        )
                        if output_grid is not None:
                            pipeline.with_output_grid(output_grid)

                        result = pipeline.run()

                    # Capture the output grid from the first pane
                    if output_grid is None:
                        output_grid = result.output_grid

                    results[pane_idx] = (result, pol_names)

                # Display results in both panes
                for pane_idx, (result, pol_names) in results.items():
                    self._viewer.set_array(
                        result.data,
                        geolocation=result.output_grid,
                        pane=pane_idx,
                    )
                    self._pane_pol_names[pane_idx] = pol_names
                    self._sync_display_controls(pane_idx)
                    self._update_colorbar_state(pane_idx)

                self._update_remap_state()
                self._update_tools_state()
                self.statusBar().showMessage(
                    "Orthorectification complete (both panes)",
                )

            except Exception as e:
                _log.error(
                    "Orthorectify dual failed: %s", e, exc_info=True,
                )
                QMessageBox.critical(
                    self, "Orthorectify Error",
                    f"Orthorectification failed:\n{e}",
                )
            finally:
                QApplication.restoreOverrideCursor()

        # --- Shared helpers for RGB results ---

        def _display_rgb_result(
            self,
            rgb_chw: Any,
            geolocation: Any,
            pane: int,
            label: str,
        ) -> None:
            """Display a pre-stretched RGB result and reset display settings.

            Parameters
            ----------
            rgb_chw : np.ndarray
                RGB image, shape ``(3, rows, cols)``, float32 [0, 1].
            geolocation : Geolocation or OutputGrid
                Geolocation for coordinate display.
            pane : int
                Target pane (0=left, 1=right).
            label : str
                Status bar label.
            """
            self._viewer.set_array(
                rgb_chw, geolocation=geolocation, pane=pane,
            )

            # Reset display settings — output is already contrast-stretched
            reset = DisplaySettings(
                percentile_low=0.0,
                percentile_high=100.0,
            )
            canvas = (
                self._viewer.left_viewer.canvas if pane == 0
                else self._viewer.right_viewer.canvas
            )
            canvas.set_display_settings(reset)

            self._sync_display_controls(pane)
            self._update_colorbar_state(pane)
            self._update_remap_state()
            self._update_tools_state()
            self.statusBar().showMessage(label)

        @staticmethod
        def _percentile_stretch(
            arr: Any,
            low: float = 2.0,
            high: float = 98.0,
        ) -> Any:
            """Percentile-stretch a real array to [0, 1] float32.

            Parameters
            ----------
            arr : np.ndarray
                Input array.
            low : float
                Lower percentile.
            high : float
                Upper percentile.

            Returns
            -------
            np.ndarray
                Stretched array, float32, values in [0, 1].
            """
            import numpy as np
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return np.zeros_like(arr, dtype=np.float32)
            vmin = np.percentile(finite, low)
            vmax = np.percentile(finite, high)
            span = vmax - vmin
            if span < np.finfo(np.float32).eps:
                return np.zeros_like(arr, dtype=np.float32)
            return np.clip(
                (arr - vmin) / span, 0.0, 1.0,
            ).astype(np.float32)

        def _on_load_vector(self) -> None:
            """Handle File > Load GeoJSON."""
            filepath, _ = QFileDialog.getOpenFileName(
                self, "Load GeoJSON", "", _VECTOR_FILTER,
            )
            if filepath:
                try:
                    self._viewer.load_vector(filepath)
                    self.statusBar().showMessage(
                        f"Loaded vector: {filepath}",
                    )
                except Exception as e:
                    QMessageBox.critical(
                        self, "Vector Error",
                        f"Could not load GeoJSON:\n{filepath}\n\n{e}",
                    )

        def _on_export(self) -> None:
            """Handle File > Export View.

            In dual mode, prompts the user to choose which pane to
            export.  Ensures the file path has an appropriate extension
            based on the selected format filter.
            """
            import os

            # In dual mode, ask which pane to export
            pane: Optional[int] = None
            if self._viewer.mode == "dual":
                from PyQt6.QtWidgets import QInputDialog
                labels = ["Left pane", "Right pane"]
                choice, ok = QInputDialog.getItem(
                    self, "Export View",
                    "Select pane to export:",
                    labels, 0, False,
                )
                if not ok:
                    return
                pane = 0 if choice == "Left pane" else 1

            filepath, selected_filter = QFileDialog.getSaveFileName(
                self, "Export View", "", _EXPORT_FILTER,
            )
            if not filepath:
                return

            # Ensure file has an extension matching the selected filter
            _, ext = os.path.splitext(filepath)
            if not ext:
                filter_ext_map = {
                    'PNG': '.png',
                    'JPEG': '.jpg',
                    'BMP': '.bmp',
                }
                for key, default_ext in filter_ext_map.items():
                    if key in selected_filter:
                        filepath += default_ext
                        break
                else:
                    # Default to PNG if filter is "All Files"
                    filepath += '.png'

            try:
                self._viewer.export_view(filepath, pane=pane)
                self.statusBar().showMessage(f"Exported: {filepath}")
            except Exception as e:
                _log.error("Export failed: %s", e, exc_info=True)
                QMessageBox.critical(
                    self, "Export Error",
                    f"Could not export view:\n{filepath}\n\n{e}",
                )

        # --- Cleanup ---

        def closeEvent(self, event: Any) -> None:
            """Ensure tile caches and busy cursors are cleaned up on close."""
            # Clear tiles in both canvases (restores busy cursor, stops workers)
            for viewer in (self._viewer.left_viewer, self._viewer.right_viewer):
                canvas = viewer.canvas
                if hasattr(canvas, '_clear_tiles'):
                    canvas._clear_tiles()
            super().closeEvent(event)

        # --- Backward compatibility ---

        @property
        def _display_dock(self) -> Any:
            """Alias for left display dock (backward compatibility)."""
            return self._left_display_dock


    def main() -> None:
        """Entry point for the grdk-viewer command."""
        args = _build_arg_parser().parse_args()

        # Configure logging
        log_level = getattr(logging, args.log_level, logging.WARNING)
        log_fmt = "%(asctime)s %(name)s %(levelname)s: %(message)s"
        log_datefmt = "%H:%M:%S"

        handlers: list = [logging.StreamHandler()]
        if args.log_file is not None:
            handlers.append(logging.FileHandler(args.log_file))

        logging.basicConfig(
            level=log_level,
            format=log_fmt,
            datefmt=log_datefmt,
            handlers=handlers,
        )

        _log = logging.getLogger("grdk.viewer")
        _log.info("grdk-viewer starting, log level=%s", args.log_level)

        app = QApplication(sys.argv)
        window = ViewerMainWindow()

        if args.file is not None:
            window._open_fresh(args.file)

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
