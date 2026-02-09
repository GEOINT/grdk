# -*- coding: utf-8 -*-
"""
OWStackViewer Widget - Interactive napari-based image stack viewer.

Embeds the napari viewer for pan/zoom navigation of co-registered image
stacks with polygon drawing for ROI selection. When a polygon is completed,
chips are extracted from all images at that location and emitted as a
ChipSet signal.

Dependencies
------------
orange-widget-base, napari

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
from typing import List, Optional

# Third-party
import numpy as np
from orangewidget import gui
from orangewidget.widget import OWBaseWidget, Input, Output, Msg

from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# GRDK internal
from grdk.core.chip import ChipSet
from grdk.viewers.polygon_tools import chip_stack_at_polygons
from grdk.widgets._signals import ImageStack, ChipSetSignal


class OWStackViewer(OWBaseWidget):
    """Interactive image stack viewer with polygon drawing.

    Embeds napari for pan/zoom image viewing across all images in a
    co-registered stack. Drawing polygons triggers chip extraction from
    all images, emitting the result as a ChipSet signal.
    """

    name = "Stack Viewer"
    description = "View image stack and draw polygons for chipping"
    icon = "icons/stack_viewer.svg"
    category = "GEODEV"
    priority = 40

    class Inputs:
        image_stack = Input("Image Stack", ImageStack)

    class Outputs:
        chip_set = Output("Chip Set", ChipSetSignal)

    class Warning(OWBaseWidget.Warning):
        no_images = Msg("No image stack connected.")
        napari_missing = Msg("napari not available. Install with: pip install napari[pyside6]")

    want_main_area = True

    def __init__(self) -> None:
        super().__init__()

        self._image_stack: Optional[ImageStack] = None
        self._viewer = None
        self._polygons: List[np.ndarray] = []

        # --- Control area ---
        box = gui.vBox(self.controlArea, "Polygon Tools")

        btn_polygon = QPushButton("Draw Polygon", self)
        btn_polygon.clicked.connect(self._on_draw_polygon)
        box.layout().addWidget(btn_polygon)

        btn_chip = QPushButton("Chip All Polygons", self)
        btn_chip.clicked.connect(self._on_chip_polygons)
        box.layout().addWidget(btn_chip)

        btn_clear = QPushButton("Clear Polygons", self)
        btn_clear.clicked.connect(self._on_clear_polygons)
        box.layout().addWidget(btn_clear)

        self._polygon_count_label = QLabel("Polygons: 0", self)
        box.layout().addWidget(self._polygon_count_label)

        self._chip_count_label = QLabel("Chips: 0", self)
        box.layout().addWidget(self._chip_count_label)

        # --- Main area ---
        self._viewer_container = QWidget(self.mainArea)
        self.mainArea.layout().addWidget(self._viewer_container)
        self._viewer_container.setLayout(QVBoxLayout())

        self._init_viewer()

    def _init_viewer(self) -> None:
        """Initialize the napari viewer."""
        try:
            from grdk.viewers.stack_viewer import NapariStackViewer

            self._viewer = NapariStackViewer(
                parent=self._viewer_container,
                on_polygon_added=self._on_polygon_added,
            )
            self.Warning.napari_missing.clear()
        except ImportError:
            self.Warning.napari_missing()
            lbl = QLabel(
                "napari is not installed.\n"
                "Install with: pip install napari[pyside6]",
                self._viewer_container,
            )
            self._viewer_container.layout().addWidget(lbl)

    @Inputs.image_stack
    def set_image_stack(self, stack: Optional[ImageStack]) -> None:
        """Receive image stack signal."""
        self._image_stack = stack
        self._polygons.clear()
        self._polygon_count_label.setText("Polygons: 0")

        if stack is None or not stack.readers:
            self.Warning.no_images()
            return

        self.Warning.no_images.clear()

        if self._viewer is not None:
            images = []
            for reader in stack.readers:
                try:
                    images.append(reader.read_full())
                except Exception:
                    continue
            self._viewer.load_stack(images, names=stack.names)

    def _on_polygon_added(self, vertices: np.ndarray) -> None:
        """Callback when a polygon is completed in napari."""
        self._polygons.append(vertices)
        self._polygon_count_label.setText(f"Polygons: {len(self._polygons)}")

    def _on_draw_polygon(self) -> None:
        """Switch napari to polygon drawing mode."""
        if self._viewer is not None:
            self._viewer.set_polygon_mode()

    def _on_chip_polygons(self) -> None:
        """Extract chips from all images at all polygon locations."""
        if self._image_stack is None or not self._polygons:
            return

        polygons = (
            self._viewer.get_polygons()
            if self._viewer is not None
            else self._polygons
        )

        if not polygons:
            return

        timestamps = self._image_stack.metadata.get('timestamps')

        chip_set = chip_stack_at_polygons(
            readers=self._image_stack.readers,
            names=self._image_stack.names,
            polygons=polygons,
            registration_results=self._image_stack.registration_results or None,
            timestamps=timestamps,
        )

        self._chip_count_label.setText(f"Chips: {len(chip_set)}")
        self.Outputs.chip_set.send(ChipSetSignal(chip_set))

    def _on_clear_polygons(self) -> None:
        """Clear all drawn polygons."""
        self._polygons.clear()
        if self._viewer is not None:
            self._viewer.clear_polygons()
        self._polygon_count_label.setText("Polygons: 0")

    def onDeleteWidget(self) -> None:
        """Clean up napari viewer on widget removal."""
        if self._viewer is not None:
            self._viewer.close()
        super().onDeleteWidget()
