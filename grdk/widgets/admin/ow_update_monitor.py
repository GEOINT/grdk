# -*- coding: utf-8 -*-
"""
OWUpdateMonitor Widget - Check PyPI/conda for artifact updates.

Dashboard showing artifacts with available updates, with background
checking via ThreadExecutorPool and one-click update capability.

Dependencies
------------
orange-widget-base

Author
------
Duane Smalley, PhD
duane.d.smalley@gmail.com

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
from orangewidget import gui
from orangewidget.widget import OWBaseWidget, Msg

from AnyQt.QtWidgets import (
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)
from AnyQt.QtCore import Qt, QTimer

# GRDK internal
from grdk.catalog.models import UpdateResult


class OWUpdateMonitor(OWBaseWidget):
    """Monitor artifacts for available updates.

    Checks PyPI and conda for newer versions of all artifacts in
    the catalog. Results are displayed in a table with update status.
    """

    name = "Update Monitor"
    description = "Check for artifact updates from PyPI/conda"
    icon = "icons/update_monitor.svg"
    category = "GRDK Admin"
    priority = 40

    class Error(OWBaseWidget.Error):
        catalog_error = Msg("Failed to open catalog: {}")
        check_error = Msg("Update check failed: {}")

    class Information(OWBaseWidget.Information):
        checking = Msg("Checking for updates...")
        done = Msg("Update check complete.")

    want_main_area = True

    def __init__(self) -> None:
        super().__init__()

        self._catalog = None
        self._pool = None
        self._results: List[UpdateResult] = []
        self._future = None

        # --- Control area ---
        box = gui.vBox(self.controlArea, "Update Check")

        btn_check = QPushButton("Check for Updates", self)
        btn_check.clicked.connect(self._on_check)
        box.layout().addWidget(btn_check)

        self._status_label = QLabel("No check performed", self)
        box.layout().addWidget(self._status_label)

        self._updates_label = QLabel("", self)
        self._updates_label.setStyleSheet("font-weight: bold;")
        box.layout().addWidget(self._updates_label)

        # --- Main area: results table ---
        self._table = QTableWidget(0, 5, self.mainArea)
        self._table.setHorizontalHeaderLabels([
            "Artifact", "Current", "Latest", "Source", "Status"
        ])
        header = self._table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.mainArea.layout().addWidget(self._table)

        # Timer for polling background future
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(500)
        self._poll_timer.timeout.connect(self._poll_future)

        self._open_catalog()

    def _open_catalog(self) -> None:
        """Open catalog and pool."""
        try:
            from grdk.catalog.database import ArtifactCatalog
            from grdk.catalog.pool import ThreadExecutorPool
            from grdk.catalog.resolver import resolve_catalog_path

            path = resolve_catalog_path()
            self._catalog = ArtifactCatalog(db_path=path)
            self._pool = ThreadExecutorPool(max_workers=2)
            self.Error.catalog_error.clear()
        except Exception as e:
            self.Error.catalog_error(str(e))

    def _on_check(self) -> None:
        """Start background update check."""
        if self._catalog is None or self._pool is None:
            return

        self.Information.checking()
        self._status_label.setText("Checking...")

        from grdk.catalog.updater import ArtifactUpdateWorker
        worker = ArtifactUpdateWorker(self._catalog, timeout=10.0)
        self._future = self._pool.submit_update_check(worker)
        self._poll_timer.start()

    def _poll_future(self) -> None:
        """Poll the background future for completion."""
        if self._future is None or not self._future.done():
            return

        self._poll_timer.stop()
        self.Information.checking.clear()

        try:
            self._results = self._future.result()
            self._update_table()
            self.Information.done()

            n_updates = sum(1 for r in self._results if r.update_available)
            self._status_label.setText(
                f"Checked {len(self._results)} artifacts"
            )
            if n_updates > 0:
                self._updates_label.setText(
                    f"{n_updates} update(s) available!"
                )
                self._updates_label.setStyleSheet(
                    "font-weight: bold; color: #22c55e;"
                )
            else:
                self._updates_label.setText("All artifacts up to date")
                self._updates_label.setStyleSheet("font-weight: bold;")
        except Exception as e:
            self.Error.check_error(str(e))
            self._status_label.setText("Check failed")

        self._future = None

    def _update_table(self) -> None:
        """Refresh the results table."""
        self._table.setRowCount(len(self._results))

        for i, result in enumerate(self._results):
            name_item = QTableWidgetItem(result.artifact.name)
            current_item = QTableWidgetItem(result.current_version)
            latest_item = QTableWidgetItem(result.latest_version or "N/A")
            source_item = QTableWidgetItem(result.source)

            if result.update_available:
                status_item = QTableWidgetItem("UPDATE AVAILABLE")
                status_item.setForeground(Qt.GlobalColor.green)
            elif result.error:
                status_item = QTableWidgetItem(result.error)
                status_item.setForeground(Qt.GlobalColor.red)
            else:
                status_item = QTableWidgetItem("Up to date")

            self._table.setItem(i, 0, name_item)
            self._table.setItem(i, 1, current_item)
            self._table.setItem(i, 2, latest_item)
            self._table.setItem(i, 3, source_item)
            self._table.setItem(i, 4, status_item)

    def onDeleteWidget(self) -> None:
        """Clean up on widget removal."""
        self._poll_timer.stop()
        if self._pool is not None:
            self._pool.shutdown(wait=False)
        if self._catalog is not None:
            self._catalog.close()
        super().onDeleteWidget()
