# -*- coding: utf-8 -*-
"""
OWImageLoader Widget - Load images into an ordered stack.

Provides a file-browser interface for loading one or more images
(NITF, TIFF, any GRDL-supported format) into an ImageStack signal.
Supports drag-reorder and format auto-detection.

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
2026-02-16
"""

# Standard library
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Third-party
from orangewidget import gui
from orangewidget.settings import Setting
from orangewidget.widget import OWBaseWidget, Input, Output, Msg

from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)
from PyQt6.QtCore import Qt

# GRDK internal
from grdk.widgets._signals import GrdkProjectSignal, ImageStack


def _try_open_reader(filepath: str):
    """Attempt to open a single image reader for *filepath*.

    Tries multiple reader strategies in order:
    1. GRDL open_sar() for SAR formats (SICD, CPHD, CRSD, SIDD, Sentinel-1)
    2. GRDL BIOMASSL1Reader for BIOMASS directories
    3. GRDL open_image() for generic formats (GeoTIFF, NITF, HDF5, JPEG2000)

    Parameters
    ----------
    filepath : str

    Returns
    -------
    ImageReader or None
    """
    path = Path(filepath)

    # Try SAR formats first
    try:
        from grdl.IO.sar import open_sar
        return open_sar(str(path))
    except Exception:
        pass

    # Try BIOMASS (directory product)
    try:
        from grdl.IO.sar import BIOMASSL1Reader
        return BIOMASSL1Reader(str(path))
    except Exception:
        pass

    # Generic fallback (GeoTIFF, NITF, HDF5, JPEG2000)
    try:
        from grdl.IO import open_image
        return open_image(str(path))
    except Exception:
        pass

    return None


def _try_open_readers(filepath: str) -> List[Tuple[Any, str]]:
    """Open readers for *filepath*, returning a list of ``(reader, name)`` pairs.

    Handles two special cases beyond the basic single-reader path:

    **BIOMASS measurement TIFF redirect**
        When the user drops an ``*_abs.tiff`` or ``*_phase.tiff`` from inside
        a ``measurement/`` sub-directory of a BIOMASS product, the function
        redirects to ``BIOMASSL1Reader`` on the parent product directory.
        This is required because the individual TIFF is magnitude-only;
        the complex scattering matrix needed for polarimetric decomposition
        requires both the ``_abs`` and ``_phase`` bands read together.

    **NISAR HDF5 multi-polarization expansion**
        A single NISAR ``.h5`` file can contain multiple polarizations.
        ``open_sar()`` only opens the first available polarization, which
        would leave the stack incomplete.  When ``available_polarizations``
        has more than one entry, the function creates one ``NISARReader``
        per polarization, labelled ``filename.h5 [HH]``, etc.

    Parameters
    ----------
    filepath : str
        Path to a file or directory.

    Returns
    -------
    list of (reader, display_name)
        Empty list if the file cannot be opened.
    """
    path = Path(filepath)

    # ── BIOMASS measurement TIFF redirect ────────────────────────────────────
    # Pattern: .../BIO_S3_.../measurement/*_abs.tiff (or *_phase.tiff / *.vrt)
    if path.suffix.lower() in ('.tiff', '.tif', '.vrt') and path.parent.name == 'measurement':
        product_dir = path.parent.parent
        if (product_dir / 'annotation').is_dir():
            try:
                from grdl.IO.sar import BIOMASSL1Reader
                reader = BIOMASSL1Reader(str(product_dir))
                logger.info(
                    "Redirected '%s' → BIOMASS product dir '%s'",
                    path.name, product_dir.name,
                )
                return [(reader, product_dir.name)]
            except Exception as exc:
                logger.debug("BIOMASS product-dir redirect failed: %s", exc)

    # ── Standard single-reader open ──────────────────────────────────────────
    reader = _try_open_reader(filepath)
    if reader is None:
        return []

    # ── NISAR multi-pol upgrade ───────────────────────────────────────────────
    # When the file exposes multiple polarizations, upgrade to a single CYX
    # cube reader (NISARReader with polarizations='all') instead of creating
    # N separate per-pol readers.  The CYX reader returns (C, rows, cols)
    # from read_chip/read_full, with all channels described in channel_metadata.
    avail_pols = getattr(
        getattr(reader, 'metadata', None), 'available_polarizations', None
    )
    if avail_pols and len(avail_pols) > 1:
        try:
            from grdl.IO.sar.nisar import NISARReader
            if isinstance(reader, NISARReader):
                reader.close()
                reader = NISARReader(str(path), polarizations='all')
                logger.info(
                    "Upgraded NISAR '%s' → CYX multi-pol reader (%s)",
                    path.name, reader.metadata.available_polarizations,
                )
        except Exception as exc:
            logger.debug("NISAR multi-pol upgrade failed: %s", exc)

    return [(reader, path.name)]


class OWImageLoader(OWBaseWidget):
    """Load one or more images into an ordered stack.

    Images can be loaded from files (NITF, TIFF, etc.) using GRDL
    readers. The stack ordering can be adjusted by dragging items
    in the list. Optional project input auto-loads project images.
    """

    name = "Image Loader"
    description = "Load images into an ordered stack"
    icon = "icons/image_loader.svg"
    category = "GEODEV"
    priority = 20

    class Inputs:
        project = Input("Project", GrdkProjectSignal, default=True, auto_summary=False)

    class Outputs:
        image_stack = Output("Image Stack", ImageStack, auto_summary=False)

    class Warning(OWBaseWidget.Warning):
        no_images = Msg("No images loaded.")
        load_failed = Msg("Failed to open: {}")

    # Persisted
    file_paths: List[str] = Setting([])

    want_main_area = False

    def __init__(self) -> None:
        super().__init__()

        self._readers: list = []
        self._names: List[str] = []

        # --- Control area ---
        box = gui.vBox(self.controlArea, "Image Stack")

        # File list
        self._list_widget = QListWidget(self)
        self._list_widget.setDragDropMode(
            QAbstractItemView.DragDropMode.InternalMove
        )
        self._list_widget.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._list_widget.model().rowsMoved.connect(self._on_reorder)
        box.layout().addWidget(self._list_widget)

        # Buttons
        btn_add = QPushButton("Add Images...", self)
        btn_add.clicked.connect(self._on_add_images)
        box.layout().addWidget(btn_add)

        btn_remove = QPushButton("Remove Selected", self)
        btn_remove.clicked.connect(self._on_remove_selected)
        box.layout().addWidget(btn_remove)

        btn_clear = QPushButton("Clear All", self)
        btn_clear.clicked.connect(self._on_clear)
        box.layout().addWidget(btn_clear)

        # Info label
        self._info_label = QLabel("No images loaded", self)
        box.layout().addWidget(self._info_label)

        # Restore from settings
        if self.file_paths:
            self._load_files(self.file_paths)

    @Inputs.project
    def set_project(self, signal: Optional[GrdkProjectSignal]) -> None:
        """Receive a project signal and load its images."""
        if signal is not None and signal.project is not None:
            project = signal.project
            if project.image_paths:
                self._load_files(project.image_paths)

    def _on_add_images(self) -> None:
        """Add images via file dialog."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select images",
            "",
            "Images (*.tif *.tiff *.ntf *.nitf *.hdf5 *.h5);;All Files (*)",
        )
        if files:
            self._load_files(files)

    def _on_remove_selected(self) -> None:
        """Remove selected images from the stack."""
        indices = sorted(
            [i.row() for i in self._list_widget.selectedIndexes()],
            reverse=True,
        )
        for idx in indices:
            self._list_widget.takeItem(idx)
            if idx < len(self._readers):
                reader = self._readers.pop(idx)
                self._names.pop(idx)
                try:
                    reader.close()
                except Exception:
                    pass
        self._emit_stack()

    def _on_clear(self) -> None:
        """Clear all images."""
        for reader in self._readers:
            try:
                reader.close()
            except Exception:
                pass
        self._readers.clear()
        self._names.clear()
        self._list_widget.clear()
        self._emit_stack()

    def _on_reorder(self) -> None:
        """Handle drag-drop reorder."""
        new_order = []
        new_names = []
        for i in range(self._list_widget.count()):
            item = self._list_widget.item(i)
            path = item.data(Qt.ItemDataRole.UserRole)
            for j, name in enumerate(self._names):
                if name == path or self._names[j] == item.text():
                    new_order.append(self._readers[j])
                    new_names.append(self._names[j])
                    break
        if len(new_order) == len(self._readers):
            self._readers = new_order
            self._names = new_names
        self._emit_stack()

    def _load_files(self, paths: List[str]) -> None:
        """Load a list of image file paths."""
        for filepath in paths:
            entries = _try_open_readers(filepath)

            if not entries:
                self.Warning.load_failed(Path(filepath).name)
                continue

            for reader, display_name in entries:
                # Avoid loading the same display name twice (e.g., settings
                # restore on second call, or NISAR pol already expanded).
                if display_name in self._names:
                    try:
                        reader.close()
                    except Exception:
                        pass
                    continue

                self._readers.append(reader)
                self._names.append(display_name)

                item = QListWidgetItem(display_name)
                item.setData(Qt.ItemDataRole.UserRole, filepath)

                try:
                    shape = reader.get_shape()
                    dtype = reader.get_dtype()
                    item.setToolTip(
                        f"{filepath}\nShape: {shape}\nDtype: {dtype}"
                    )
                except Exception:
                    item.setToolTip(filepath)

                self._list_widget.addItem(item)

        self.file_paths = [
            self._list_widget.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self._list_widget.count())
        ]
        self._emit_stack()

    def _emit_stack(self) -> None:
        """Emit the current image stack signal."""
        if not self._readers:
            self.Warning.no_images()
            self._info_label.setText("No images loaded")
            self.Outputs.image_stack.send(None)
            return

        self.Warning.no_images.clear()
        self.Warning.load_failed.clear()
        self._info_label.setText(f"{len(self._readers)} images loaded")

        stack = ImageStack(
            readers=list(self._readers),
            names=list(self._names),
        )
        self.Outputs.image_stack.send(stack)

    def onDeleteWidget(self) -> None:
        """Clean up readers on widget deletion."""
        for reader in self._readers:
            try:
                reader.close()
            except Exception:
                pass
        super().onDeleteWidget()
