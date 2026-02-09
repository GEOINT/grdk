# -*- coding: utf-8 -*-
"""
OWChipper Widget - Standalone chip extraction from polygon definitions.

For users entering the workflow with pre-defined polygons (GeoJSON or
pixel coordinates). Extracts chips from all images in the stack at
the specified polygon locations.

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
import json
from pathlib import Path
from typing import List, Optional

# Third-party
import numpy as np
from orangewidget import gui
from orangewidget.settings import Setting
from orangewidget.widget import OWBaseWidget, Input, Output, Msg

from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
)

# GRDK internal
from grdk.core.chip import ChipSet
from grdk.viewers.polygon_tools import chip_stack_at_polygons
from grdk.widgets._signals import ImageStack, ChipSetSignal


def _parse_geojson_polygons(geojson: dict) -> List[np.ndarray]:
    """Extract polygon coordinates from a GeoJSON FeatureCollection.

    Parameters
    ----------
    geojson : dict
        GeoJSON object (Feature or FeatureCollection).

    Returns
    -------
    List[np.ndarray]
        List of polygon vertex arrays, each (N, 2) in (row, col) format.
        Note: GeoJSON coordinates are (lon, lat) or (x, y) — we treat
        them as (col, row) and swap to (row, col).
    """
    polygons = []
    features = []

    if geojson.get('type') == 'FeatureCollection':
        features = geojson.get('features', [])
    elif geojson.get('type') == 'Feature':
        features = [geojson]
    elif geojson.get('type') == 'Polygon':
        features = [{'geometry': geojson}]

    for feature in features:
        geometry = feature.get('geometry', {})
        if geometry.get('type') != 'Polygon':
            continue
        coords = geometry.get('coordinates', [])
        if not coords:
            continue
        # Use outer ring, swap (x, y) → (row, col)
        ring = np.array(coords[0])
        vertices = ring[:, ::-1]  # (x, y) → (y, x) = (row, col)
        polygons.append(vertices)

    return polygons


class OWChipper(OWBaseWidget):
    """Extract chips from image stack at polygon locations.

    Accepts polygon definitions from file (GeoJSON) or manual pixel
    coordinate entry. Extracts bounding-box chips from all images
    in the connected stack.
    """

    name = "Chipper"
    description = "Extract image chips at polygon locations"
    icon = "icons/chipper.svg"
    category = "GEODEV"
    priority = 50

    class Inputs:
        image_stack = Input("Image Stack", ImageStack)

    class Outputs:
        chip_set = Output("Chip Set", ChipSetSignal)

    class Warning(OWBaseWidget.Warning):
        no_stack = Msg("No image stack connected.")
        no_polygons = Msg("No polygons defined.")

    class Error(OWBaseWidget.Error):
        parse_error = Msg("Failed to parse polygon file: {}")

    want_main_area = False

    def __init__(self) -> None:
        super().__init__()

        self._image_stack: Optional[ImageStack] = None
        self._polygons: List[np.ndarray] = []

        # --- Control area ---
        box = gui.vBox(self.controlArea, "Polygon Source")

        btn_load = QPushButton("Load GeoJSON...", self)
        btn_load.clicked.connect(self._on_load_geojson)
        box.layout().addWidget(btn_load)

        box.layout().addWidget(QLabel("Or enter pixel coordinates (row,col per line):"))

        self._coord_edit = QPlainTextEdit(self)
        self._coord_edit.setPlaceholderText(
            "100,200\n100,300\n200,300\n200,200"
        )
        self._coord_edit.setMaximumHeight(120)
        box.layout().addWidget(self._coord_edit)

        btn_add = QPushButton("Add Polygon from Text", self)
        btn_add.clicked.connect(self._on_add_text_polygon)
        box.layout().addWidget(btn_add)

        # Status
        self._status_label = QLabel("Polygons: 0", self)
        box.layout().addWidget(self._status_label)

        # Normalization (uses GRDL data_prep.Normalizer when available)
        box_norm = gui.vBox(self.controlArea, "Normalization")
        self._norm_combo = QComboBox(self)
        self._norm_combo.addItem("None", None)
        try:
            from grdl.data_prep import Normalizer  # noqa: F401
            self._norm_combo.addItem("Min-Max [0,1]", "minmax")
            self._norm_combo.addItem("Z-Score", "zscore")
            self._norm_combo.addItem("Percentile (2-98%)", "percentile")
        except ImportError:
            pass
        box_norm.layout().addWidget(self._norm_combo)

        # Chip button
        box2 = gui.vBox(self.controlArea, "Actions")

        btn_chip = QPushButton("Extract Chips", self)
        btn_chip.clicked.connect(self._on_extract)
        box2.layout().addWidget(btn_chip)

        btn_clear = QPushButton("Clear Polygons", self)
        btn_clear.clicked.connect(self._on_clear)
        box2.layout().addWidget(btn_clear)

        self._result_label = QLabel("", self)
        box2.layout().addWidget(self._result_label)

    @Inputs.image_stack
    def set_image_stack(self, stack: Optional[ImageStack]) -> None:
        """Receive image stack signal."""
        self._image_stack = stack
        if stack is None or not stack.readers:
            self.Warning.no_stack()
        else:
            self.Warning.no_stack.clear()

    def _on_load_geojson(self) -> None:
        """Load polygon definitions from a GeoJSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load GeoJSON", "",
            "GeoJSON (*.geojson *.json);;All Files (*)"
        )
        if not path:
            return

        try:
            with open(path, 'r') as f:
                geojson = json.load(f)
            polys = _parse_geojson_polygons(geojson)
            self._polygons.extend(polys)
            self.Error.parse_error.clear()
            self._update_status()
        except Exception as e:
            self.Error.parse_error(str(e))

    def _on_add_text_polygon(self) -> None:
        """Parse polygon from the text coordinate entry."""
        text = self._coord_edit.toPlainText().strip()
        if not text:
            return

        try:
            vertices = []
            for line in text.strip().split('\n'):
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    row = float(parts[0].strip())
                    col = float(parts[1].strip())
                    vertices.append([row, col])

            if len(vertices) >= 3:
                self._polygons.append(np.array(vertices))
                self._coord_edit.clear()
                self._update_status()
        except ValueError:
            pass

    def _on_extract(self) -> None:
        """Extract chips at all polygon locations."""
        if self._image_stack is None or not self._image_stack.readers:
            self.Warning.no_stack()
            return

        if not self._polygons:
            self.Warning.no_polygons()
            return

        self.Warning.no_polygons.clear()
        timestamps = self._image_stack.metadata.get('timestamps')

        chip_set = chip_stack_at_polygons(
            readers=self._image_stack.readers,
            names=self._image_stack.names,
            polygons=self._polygons,
            timestamps=timestamps,
        )

        # Apply optional normalization via GRDL data_prep.Normalizer
        norm_method = self._norm_combo.currentData()
        if norm_method is not None and len(chip_set) > 0:
            try:
                from grdl.data_prep import Normalizer
                normalizer = Normalizer(method=norm_method)
                for chip in chip_set.chips:
                    chip.image_data = normalizer.normalize(
                        chip.image_data.astype(float)
                    )
            except ImportError:
                pass

        self._result_label.setText(
            f"Extracted {len(chip_set)} chips from "
            f"{len(self._polygons)} polygons"
        )
        self.Outputs.chip_set.send(ChipSetSignal(chip_set))

    def _on_clear(self) -> None:
        """Clear all polygon definitions."""
        self._polygons.clear()
        self._update_status()
        self._result_label.setText("")

    def _update_status(self) -> None:
        """Update the polygon count label."""
        self._status_label.setText(f"Polygons: {len(self._polygons)}")
