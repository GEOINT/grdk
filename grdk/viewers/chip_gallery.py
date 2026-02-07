# -*- coding: utf-8 -*-
"""
Chip Gallery Widget - Thumbnail grid for chip viewing and labeling.

Provides a scrollable grid of chip thumbnails with color-coded borders
indicating labels (positive=green, negative=red, unknown=gray).
Clicking a chip cycles through labels.

Dependencies
------------
PyQt5 (via Orange)

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
from typing import Any, Callable, Dict, List, Optional

# Third-party
import numpy as np

try:
    from AnyQt.QtWidgets import (
        QFrame,
        QGridLayout,
        QLabel,
        QScrollArea,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )
    from AnyQt.QtCore import Qt, pyqtSignal

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False

from grdk.viewers.image_canvas import ImageCanvasThumbnail

# GRDK internal
from grdk.core.chip import Chip, ChipLabel


_LABEL_COLORS = {
    ChipLabel.POSITIVE: "#22c55e",   # green
    ChipLabel.NEGATIVE: "#ef4444",   # red
    ChipLabel.UNKNOWN: "#9ca3af",    # gray
}

_LABEL_CYCLE = [ChipLabel.UNKNOWN, ChipLabel.POSITIVE, ChipLabel.NEGATIVE]

THUMB_SIZE = 128


class ChipThumbnail(QFrame):
    """A single chip thumbnail with label indicator.

    Parameters
    ----------
    chip : Chip
    index : int
    on_label_changed : Optional[Callable]
    parent : Optional[QWidget]
    """

    def __init__(
        self,
        chip: Chip,
        index: int,
        on_label_changed: Optional[Callable] = None,
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self._chip = chip
        self._index = index
        self._on_label_changed = on_label_changed

        self.setFrameShape(QFrame.Shape.Box)
        self.setLineWidth(3)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_border()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Thumbnail image via ImageCanvas
        self._canvas = ImageCanvasThumbnail(size=THUMB_SIZE, parent=self)
        if _QT_AVAILABLE:
            self._canvas.set_array(chip.image_data)
        layout.addWidget(self._canvas)

        # Info text
        info = chip.source_image_name
        if chip.timestamp:
            info += f"\n{chip.timestamp}"
        text_label = QLabel(info, self)
        text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text_label.setWordWrap(True)
        text_label.setStyleSheet("font-size: 10px;")
        layout.addWidget(text_label)

        # Label indicator
        self._label_text = QLabel(chip.label.value.upper(), self)
        self._label_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label_text.setStyleSheet("font-weight: bold; font-size: 11px;")
        layout.addWidget(self._label_text)

    def mousePressEvent(self, event: Any) -> None:
        """Cycle through labels on click."""
        current_idx = _LABEL_CYCLE.index(self._chip.label)
        next_idx = (current_idx + 1) % len(_LABEL_CYCLE)
        self._chip.label = _LABEL_CYCLE[next_idx]
        self._update_border()
        self._label_text.setText(self._chip.label.value.upper())
        if self._on_label_changed:
            self._on_label_changed(self._index, self._chip.label)

    def _update_border(self) -> None:
        """Update border color based on current label."""
        color = _LABEL_COLORS[self._chip.label]
        self.setStyleSheet(f"ChipThumbnail {{ border: 3px solid {color}; }}")


class ChipGalleryWidget(QScrollArea):
    """Scrollable grid of chip thumbnails with click-to-label.

    Parameters
    ----------
    chips : List[Chip]
    columns : int
    on_label_changed : Optional[Callable]
    parent : Optional[QWidget]
    """

    def __init__(
        self,
        chips: Optional[List[Chip]] = None,
        columns: int = 4,
        on_label_changed: Optional[Callable] = None,
        parent: Optional[Any] = None,
    ) -> None:
        if not _QT_AVAILABLE:
            raise ImportError("Qt is required for ChipGalleryWidget")

        super().__init__(parent)
        self._columns = columns
        self._on_label_changed = on_label_changed
        self._thumbnails: List[ChipThumbnail] = []

        self.setWidgetResizable(True)

        self._container = QWidget()
        self._grid = QGridLayout(self._container)
        self._grid.setSpacing(8)
        self.setWidget(self._container)

        if chips:
            self.set_chips(chips)

    def set_chips(self, chips: List[Chip]) -> None:
        """Replace displayed chips.

        Parameters
        ----------
        chips : List[Chip]
        """
        # Clear existing
        for thumb in self._thumbnails:
            thumb.setParent(None)
            thumb.deleteLater()
        self._thumbnails.clear()

        for i, chip in enumerate(chips):
            thumb = ChipThumbnail(
                chip, i,
                on_label_changed=self._on_label_changed,
                parent=self._container,
            )
            row = i // self._columns
            col = i % self._columns
            self._grid.addWidget(thumb, row, col)
            self._thumbnails.append(thumb)

    def clear(self) -> None:
        """Remove all thumbnails."""
        for thumb in self._thumbnails:
            thumb.setParent(None)
            thumb.deleteLater()
        self._thumbnails.clear()
