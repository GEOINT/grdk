# -*- coding: utf-8 -*-
"""
OWLabeler Widget - Chip labeling interface.

Displays chip thumbnails grouped by polygon region with click-to-cycle
labeling (unknown → positive → negative). Shows label summary statistics.

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
from typing import Optional

# Third-party
from orangewidget import gui
from orangewidget.widget import OWBaseWidget, Input, Output, Msg

from PyQt6.QtWidgets import (
    QLabel,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# GRDK internal
from grdl_rt.execution.chip import ChipLabel, ChipSet
from grdk.viewers.chip_gallery import ChipGalleryWidget
from grdk.widgets._signals import ChipSetSignal


class OWLabeler(OWBaseWidget):
    """Label chips as positive, negative, or unknown.

    Displays a tabbed gallery of chip thumbnails grouped by polygon
    region. Clicking a chip cycles through labels. Emits the updated
    ChipSet when labeling is complete.
    """

    name = "Labeler"
    description = "Label chips as positive/negative/unknown"
    icon = "icons/labeler.svg"
    category = "GEODEV"
    priority = 60

    class Inputs:
        chip_set = Input("Chip Set", ChipSetSignal, auto_summary=False)

    class Outputs:
        chip_set = Output("Chip Set", ChipSetSignal, auto_summary=False)

    class Warning(OWBaseWidget.Warning):
        no_chips = Msg("No chips received.")

    want_main_area = True

    def __init__(self) -> None:
        super().__init__()

        self._chip_set: Optional[ChipSet] = None

        # --- Control area ---
        box = gui.vBox(self.controlArea, "Label Summary")
        self._positive_label = QLabel("Positive: 0", self)
        self._negative_label = QLabel("Negative: 0", self)
        self._unknown_label = QLabel("Unknown: 0", self)
        box.layout().addWidget(self._positive_label)
        box.layout().addWidget(self._negative_label)
        box.layout().addWidget(self._unknown_label)

        # Bulk action buttons
        box2 = gui.vBox(self.controlArea, "Bulk Actions")

        btn_all_pos = QPushButton("Mark All Positive", self)
        btn_all_pos.clicked.connect(lambda: self._set_all_labels(ChipLabel.POSITIVE))
        box2.layout().addWidget(btn_all_pos)

        btn_all_neg = QPushButton("Mark All Negative", self)
        btn_all_neg.clicked.connect(lambda: self._set_all_labels(ChipLabel.NEGATIVE))
        box2.layout().addWidget(btn_all_neg)

        btn_all_unk = QPushButton("Reset All to Unknown", self)
        btn_all_unk.clicked.connect(lambda: self._set_all_labels(ChipLabel.UNKNOWN))
        box2.layout().addWidget(btn_all_unk)

        # Send button
        btn_send = QPushButton("Send Labeled Chips", self)
        btn_send.clicked.connect(self._on_send)
        box2.layout().addWidget(btn_send)

        # --- Main area ---
        self._tabs = QTabWidget(self.mainArea)
        self.mainArea.layout().addWidget(self._tabs)
        self._galleries = []

    @Inputs.chip_set
    def set_chip_set(self, signal: Optional[ChipSetSignal]) -> None:
        """Receive chip set signal."""
        if signal is None or signal.chip_set is None:
            self._chip_set = None
            self.Warning.no_chips()
            self._clear_galleries()
            return

        self.Warning.no_chips.clear()
        self._chip_set = signal.chip_set
        self._build_galleries()
        self._update_summary()

    def _build_galleries(self) -> None:
        """Build tabbed galleries grouped by polygon region."""
        self._clear_galleries()

        if self._chip_set is None:
            return

        regions = self._chip_set.polygon_regions
        if not regions:
            # Single gallery for all chips
            gallery = ChipGalleryWidget(
                chips=list(self._chip_set.chips),
                on_label_changed=self._on_label_changed,
            )
            self._tabs.addTab(gallery, "All Chips")
            self._galleries.append(gallery)
        else:
            for i, region in enumerate(regions):
                chips = self._chip_set.chips_for_region(region)
                if not chips:
                    continue
                tab_name = region.name or f"Region {i + 1}"
                gallery = ChipGalleryWidget(
                    chips=chips,
                    on_label_changed=self._on_label_changed,
                )
                self._tabs.addTab(gallery, tab_name)
                self._galleries.append(gallery)

    def _clear_galleries(self) -> None:
        """Remove all gallery tabs."""
        for gallery in self._galleries:
            gallery.clear()
        self._galleries.clear()
        self._tabs.clear()

    def _on_label_changed(self, index: int, label: ChipLabel) -> None:
        """Handle label change from gallery click."""
        self._update_summary()

    def _set_all_labels(self, label: ChipLabel) -> None:
        """Set all chips to the specified label."""
        if self._chip_set is None:
            return

        for chip in self._chip_set.chips:
            chip.label = label

        # Rebuild galleries to reflect changes
        self._build_galleries()
        self._update_summary()

    def _update_summary(self) -> None:
        """Update the label count summary."""
        if self._chip_set is None:
            return

        counts = self._chip_set.label_counts
        self._positive_label.setText(f"Positive: {counts.get('positive', 0)}")
        self._negative_label.setText(f"Negative: {counts.get('negative', 0)}")
        self._unknown_label.setText(f"Unknown: {counts.get('unknown', 0)}")

    def _on_send(self) -> None:
        """Emit the labeled chip set."""
        if self._chip_set is not None:
            self.Outputs.chip_set.send(ChipSetSignal(self._chip_set))

    def onDeleteWidget(self) -> None:
        """Clean up galleries on widget removal."""
        self._clear_galleries()
        super().onDeleteWidget()
