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
2026-02-06
"""

# Standard library
from pathlib import Path
from typing import List, Optional

# Third-party
from orangewidget import gui
from orangewidget.settings import Setting
from orangewidget.widget import OWBaseWidget, Input, Output, Msg

from AnyQt.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

# GRDK internal
from grdk.widgets._signals import GrdkProjectSignal, ImageStack


def _try_open_reader(filepath: str):
    """Attempt to open an image using GRDL readers.

    Tries multiple reader strategies in order:
    1. GRDL open_sar() for SAR formats
    2. GRDL BIOMASSL1Reader for BIOMASS
    3. Rasterio fallback for GeoTIFF/NITF

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

    # Try BIOMASS
    try:
        from grdl.IO.biomass import BIOMASSL1Reader
        return BIOMASSL1Reader(str(path))
    except Exception:
        pass

    # Rasterio fallback (GeoTIFF, NITF, etc.)
    try:
        from grdl.IO.sar import GRDReader
        return GRDReader(str(path))
    except Exception:
        pass

    return None


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
        project = Input("Project", GrdkProjectSignal, default=True)

    class Outputs:
        image_stack = Output("Image Stack", ImageStack)

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
        self._list_widget.setDefaultDropAction(1)  # MoveAction
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
            path = item.data(256)  # Qt.UserRole
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
            if filepath in self._names:
                continue

            reader = _try_open_reader(filepath)
            if reader is not None:
                self._readers.append(reader)
                name = Path(filepath).name
                self._names.append(name)
                item = QListWidgetItem(name)
                item.setData(256, filepath)  # Qt.UserRole

                # Add shape/dtype info as tooltip
                try:
                    shape = reader.get_shape()
                    dtype = reader.get_dtype()
                    item.setToolTip(f"{filepath}\nShape: {shape}\nDtype: {dtype}")
                except Exception:
                    item.setToolTip(filepath)

                self._list_widget.addItem(item)
            else:
                self.Warning.load_failed(Path(filepath).name)

        self.file_paths = [
            self._list_widget.item(i).data(256)
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
