# -*- coding: utf-8 -*-
"""
OWChipLoader Widget - Load a directory of pre-extracted chip images.

For workflows where the user arrives with chips already extracted from
a scene (e.g. from a previous run or an external tool). Scans a
directory for supported image files, loads each as a Chip, and checks
for a companion ``.geojson`` sidecar next to each file.

Sidecar format: standard GeoJSON FeatureCollection (see
:mod:`grdl.vector.io`).  Features are attached to ``chip.annotations``.
The FeatureCollection's ``properties.modality`` field (if present) is
stored on ``chip.modality`` and propagated to the output signal via
``Chip.modality``, enabling downstream modality-aware palette filtering
via :func:`~grdk.widgets._signals.get_modality_hint`.

Author
------
Steven Siebert

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-04-20
"""

# Standard library
import logging
from pathlib import Path
from typing import List, Optional

# Third-party
import numpy as np
from orangewidget import gui
from orangewidget.settings import Setting
from orangewidget.widget import OWBaseWidget, Output, Msg

from PyQt6.QtWidgets import (
    QFileDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# GRDK internal
from grdl_rt.execution.chip import Chip, ChipLabel, ChipSet, PolygonRegion
from grdk.widgets._signals import ChipSetSignal

logger = logging.getLogger(__name__)

# Image extensions supported by grdl's open_any
_IMAGE_EXTENSIONS = {
    ".nitf", ".ntf", ".tif", ".tiff", ".geotiff",
    ".h5", ".hdf5", ".nc", ".img", ".bip", ".bsq",
    ".png", ".jpg", ".jpeg",
}


def _load_chip_from_file(path: Path) -> Optional[Chip]:
    """Load a single chip image file.

    Returns a :class:`~grdl_rt.execution.chip.Chip` whose
    ``image_data`` is the first band array, ``modality`` is set from
    any companion ``.geojson`` sidecar, and ``annotations`` holds the
    sidecar Features.

    Returns ``None`` on read failure.
    """
    try:
        from grdl.IO.generic import open_any
    except ImportError:
        logger.error("grdl is not installed; cannot load chip images.")
        return None

    try:
        reader = open_any(str(path))
    except Exception as exc:
        logger.warning("Cannot open %s: %s", path, exc)
        return None

    try:
        arr = reader.read()
    except Exception as exc:
        logger.warning("Cannot read %s: %s", path, exc)
        return None

    if arr is None or arr.size == 0:
        return None

    # Full-image bounding box as PolygonRegion
    if arr.ndim == 2:
        rows, cols = arr.shape
    else:
        rows, cols = arr.shape[:2]
    region = PolygonRegion(
        vertices=np.array([[0, 0], [0, cols], [rows, cols], [rows, 0]], dtype=np.float64)
    )

    # Attempt to resolve modality from reader metadata
    modality = None
    try:
        from grdl.discovery import extract_modality
        meta = getattr(reader, "metadata", None)
        if meta is not None:
            modality_str = extract_modality(meta)
            if modality_str is not None:
                from grdl.vocabulary import ImageModality
                try:
                    modality = ImageModality(modality_str)
                except ValueError:
                    pass
    except ImportError:
        pass

    # Load companion GeoJSON sidecar if present
    annotations: list = []
    sidecar_path = path.with_suffix(".geojson")
    if sidecar_path.exists():
        try:
            from grdl.vector.io import VectorReader
            feature_set = VectorReader.read(sidecar_path)
            annotations = list(feature_set)

            # Sidecar modality takes precedence over reader-derived modality
            sidecar_modality_str = feature_set.metadata.get("modality")
            if sidecar_modality_str is not None:
                try:
                    from grdl.vocabulary import ImageModality
                    modality = ImageModality(str(sidecar_modality_str).upper())
                except (ValueError, ImportError):
                    pass
        except Exception as exc:
            logger.warning("Could not load sidecar %s: %s", sidecar_path, exc)

    return Chip(
        image_data=arr,
        source_image_index=0,
        source_image_name=path.name,
        polygon_region=region,
        label=ChipLabel.UNKNOWN,
        annotations=annotations,
        modality=modality,
    )


class OWChipLoader(OWBaseWidget):
    """Load a directory of pre-extracted chip images into a ChipSet.

    Scans the chosen directory for supported image files and loads each
    as a :class:`~grdl_rt.execution.chip.Chip`.  Companion ``.geojson``
    sidecars are loaded automatically via
    :class:`grdl.vector.io.VectorReader` — features are stored on
    ``chip.annotations`` and collection-level ``modality`` is propagated
    to ``chip.modality`` so that downstream widgets (OWOrchestrator,
    OWProcessor) can auto-detect modality via
    :func:`~grdk.widgets._signals.get_modality_hint`.
    """

    name = "Chip Loader"
    description = "Load pre-extracted chip images from a directory"
    icon = "icons/chip_loader.svg"
    category = "GEODEV"
    priority = 45

    class Outputs:
        chip_set = Output("Chip Set", ChipSetSignal, auto_summary=False)

    class Warning(OWBaseWidget.Warning):
        no_chips = Msg("No supported image files found in the selected directory.")

    class Error(OWBaseWidget.Error):
        load_failed = Msg("Failed to load chips: {}")

    last_directory: str = Setting("")

    want_main_area = False

    def __init__(self) -> None:
        super().__init__()

        self._chip_paths: List[Path] = []

        box = gui.vBox(self.controlArea, "Chip Directory")

        btn_browse = QPushButton("Browse Directory…", self)
        btn_browse.clicked.connect(self._on_browse)
        box.layout().addWidget(btn_browse)

        self._dir_label = QLabel(self.last_directory or "(no directory selected)", self)
        self._dir_label.setWordWrap(True)
        box.layout().addWidget(self._dir_label)

        self._file_list = QListWidget(self)
        self._file_list.setMaximumHeight(160)
        box.layout().addWidget(self._file_list)

        self._status_label = QLabel("", self)
        box.layout().addWidget(self._status_label)

        btn_load = QPushButton("Load Chips", self)
        btn_load.clicked.connect(self._on_load)
        box.layout().addWidget(btn_load)

        if self.last_directory:
            self._scan_directory(Path(self.last_directory))

    def _on_browse(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Chip Directory",
            self.last_directory or "",
        )
        if not directory:
            return
        self.last_directory = directory
        self._dir_label.setText(directory)
        self._scan_directory(Path(directory))

    def _scan_directory(self, directory: Path) -> None:
        self._chip_paths = sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        self._file_list.clear()
        for p in self._chip_paths:
            has_sidecar = p.with_suffix(".geojson").exists()
            label = p.name + (" [sidecar]" if has_sidecar else "")
            self._file_list.addItem(QListWidgetItem(label))
        self._status_label.setText(f"{len(self._chip_paths)} file(s) found")

        if not self._chip_paths:
            self.Warning.no_chips()
        else:
            self.Warning.no_chips.clear()

    def _on_load(self) -> None:
        self.Error.load_failed.clear()
        if not self._chip_paths:
            return

        chips = []
        for i, path in enumerate(self._chip_paths):
            chip = _load_chip_from_file(path)
            if chip is not None:
                chip.source_image_index = i
                chips.append(chip)

        if not chips:
            self.Error.load_failed("All files failed to load.")
            return

        chip_set = ChipSet(chips=chips)
        self._status_label.setText(f"Loaded {len(chips)} chip(s)")
        self.Outputs.chip_set.send(ChipSetSignal(chip_set))
