# -*- coding: utf-8 -*-
"""
Bookmarks - Target bookmark and waypoint manager.

Allows analysts to catalog, name, jump to, and export/import Targets of Interest (TOIs).

Author: Steven Siebert
License: MIT License
"""

from dataclasses import dataclass, asdict
from typing import Any, List, Optional
import json

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog
    )
    from PyQt6.QtCore import Qt, pyqtSignal as Signal
    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False


@dataclass
class Bookmark:
    name: str
    lat: float
    lon: float
    row: int
    col: int
    zoom: float = 1.0
    notes: str = ""


if _QT_AVAILABLE:

    class BookmarkManagerWidget(QWidget):
        """Dockable target bookmark manager."""

        bookmark_selected = Signal(object)  # Bookmark

        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            self.bookmarks: List[Bookmark] = []

            layout = QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)

            self.table = QTableWidget(0, 4, self)
            self.table.setHorizontalHeaderLabels(["Name", "Lat", "Lon", "Notes"])
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.table.itemDoubleClicked.connect(self._on_item_double_clicked)
            layout.addWidget(self.table)

            btn_row = QHBoxLayout()
            self.add_btn = QPushButton("Add Target", self)
            self.del_btn = QPushButton("Delete", self)
            self.export_btn = QPushButton("Export", self)
            self.import_btn = QPushButton("Import", self)

            self.del_btn.clicked.connect(self._delete_selected)
            self.export_btn.clicked.connect(self._export_dialog)
            self.import_btn.clicked.connect(self._import_dialog)

            btn_row.addWidget(self.add_btn)
            btn_row.addWidget(self.del_btn)
            btn_row.addWidget(self.export_btn)
            btn_row.addWidget(self.import_btn)
            layout.addLayout(btn_row)

        def add_bookmark(self, bm: Bookmark) -> None:
            self.bookmarks.append(bm)
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(bm.name))
            self.table.setItem(row, 1, QTableWidgetItem(f"{bm.lat:.4f}"))
            self.table.setItem(row, 2, QTableWidgetItem(f"{bm.lon:.4f}"))
            self.table.setItem(row, 3, QTableWidgetItem(bm.notes))

        def clear(self) -> None:
            self.bookmarks.clear()
            self.table.setRowCount(0)

        def export_bookmarks(self, filepath: str) -> None:
            data = [asdict(b) for b in self.bookmarks]
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        def import_bookmarks(self, filepath: str) -> None:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            for d in data:
                self.add_bookmark(Bookmark(**d))

        def _delete_selected(self) -> None:
            row = self.table.currentRow()
            if 0 <= row < len(self.bookmarks):
                self.bookmarks.pop(row)
                self.table.removeRow(row)

        def _on_item_double_clicked(self, item) -> None:
            row = self.table.row(item)
            if 0 <= row < len(self.bookmarks):
                self.bookmark_selected.emit(self.bookmarks[row])

        def _export_dialog(self) -> None:
            path, _ = QFileDialog.getSaveFileName(self, "Export Targets", "targets.json", "JSON (*.json)")
            if path:
                self.export_bookmarks(path)

        def _import_dialog(self) -> None:
            path, _ = QFileDialog.getOpenFileName(self, "Import Targets", "", "JSON (*.json)")
            if path:
                self.import_bookmarks(path)
