# -*- coding: utf-8 -*-
"""
OWMetadataInspector Widget - CPHD/CRSD phase history metadata viewer.

Displays critical phase-history parameters for CPHD (Compensated Phase
History Data) and CRSD (Compensated Received Signal Data) formats.
Validates required fields and highlights suspicious values.

Dependencies
------------
orange-widget-base

Author
------
Claude Code (Anthropic)

License
-------
MIT License
Copyright (c) 2026 geoint.org

Created
-------
2026-06-18
"""

# Standard library
import json
from typing import Any, Dict, List, Optional

# Third-party
from orangewidget import gui
from orangewidget.widget import OWBaseWidget, Input, Msg

from PyQt6.QtWidgets import (
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

# GRDK internal
from grdk.widgets._signals import ImageStack


class OWMetadataInspector(OWBaseWidget):
    """CPHD/CRSD phase history metadata inspector.

    Displays pulse parameters, timeline metadata, channel descriptions,
    and validates critical fields for phase-history formats. Highlights
    suspicious values (e.g., negative pulse widths, excessive PRF).
    """

    name = "Metadata Inspector"
    description = "Inspect and validate CPHD/CRSD phase history metadata"
    icon = "icons/metadata_inspector.svg"
    category = "GEODEV"
    priority = 25

    class Inputs:
        image_stack = Input("Image Stack", ImageStack, auto_summary=False)

    class Warning(OWBaseWidget.Warning):
        missing_field = Msg("Required field missing: {}")
        suspicious_value = Msg("Suspicious value: {}")

    want_main_area = True

    def __init__(self) -> None:
        super().__init__()

        self._stack: Optional[ImageStack] = None
        self._metadata: Dict[str, Any] = {}
        self._warnings: List[str] = []

        # --- Control area ---
        box = gui.vBox(self.controlArea, "Inspector")

        self._format_label = QLabel("Format: (no data)")
        box.layout().addWidget(self._format_label)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self._on_refresh)
        box.layout().addWidget(btn_refresh)

        btn_export = QPushButton("Export JSON...")
        btn_export.clicked.connect(self._on_export)
        box.layout().addWidget(btn_export)

        # --- Main area ---
        main_layout = QVBoxLayout()
        self.mainArea.setLayout(main_layout)

        # Metadata table
        main_layout.addWidget(QLabel("Phase History Parameters"))
        self._metadata_table = QTableWidget()
        self._metadata_table.setColumnCount(3)
        self._metadata_table.setHorizontalHeaderLabels(
            ["Parameter", "Value", "Status"]
        )
        self._metadata_table.horizontalHeader().setStretchLastSection(False)
        self._metadata_table.setColumnWidth(0, 250)
        self._metadata_table.setColumnWidth(1, 200)
        self._metadata_table.setColumnWidth(2, 100)
        main_layout.addWidget(self._metadata_table)

        # Validation warnings panel
        main_layout.addWidget(QLabel("Validation Warnings"))
        self._warnings_text = QTextEdit()
        self._warnings_text.setReadOnly(True)
        self._warnings_text.setMaximumHeight(150)
        main_layout.addWidget(self._warnings_text)

        # Full metadata dump
        main_layout.addWidget(QLabel("Full Metadata (JSON)"))
        self._json_text = QTextEdit()
        self._json_text.setReadOnly(True)
        self._json_text.setFont(self.font())
        main_layout.addWidget(self._json_text)

    @Inputs.image_stack
    def set_image_stack(self, signal: Optional[ImageStack]) -> None:
        """Receive image stack and extract metadata."""
        self.Warning.missing_field.clear()
        self.Warning.suspicious_value.clear()

        if signal is None or not signal.readers:
            self._stack = None
            self._clear_display()
            return

        self._stack = signal
        self._extract_metadata()
        self._validate_metadata()
        self._update_display()

    def _extract_metadata(self) -> None:
        """Extract metadata from the first reader in the stack."""
        if not self._stack or not self._stack.readers:
            self._metadata = {}
            return

        reader = self._stack.readers[0]
        meta = getattr(reader, 'metadata', None)
        if meta is None:
            self._metadata = {}
            return

        # Extract all metadata attributes as a dict
        self._metadata = {}
        for attr in dir(meta):
            if attr.startswith('_'):
                continue
            try:
                value = getattr(meta, attr)
                if callable(value):
                    continue
                self._metadata[attr] = value
            except Exception:
                pass

        # Determine format
        fmt = self._metadata.get('format', 'Unknown')
        self._format_label.setText(f"Format: {fmt}")

    def _validate_metadata(self) -> None:
        """Validate CPHD/CRSD metadata and collect warnings."""
        self._warnings = []

        fmt = self._metadata.get('format', '').upper()
        if 'CPHD' not in fmt and 'CRSD' not in fmt:
            # Not a phase-history format
            return

        # Required fields for CPHD
        required_fields = [
            'rows',
            'cols',
            'dtype',
        ]

        for field in required_fields:
            if field not in self._metadata:
                self._warnings.append(f"Required field missing: {field}")
                self.Warning.missing_field(field)

        # Validate numeric ranges
        self._validate_pulse_params()
        self._validate_timeline_params()

    def _validate_pulse_params(self) -> None:
        """Validate pulse-related parameters."""
        # Pulse width
        pulse_width = self._metadata.get('pulse_width')
        if pulse_width is not None:
            if pulse_width <= 0:
                msg = f"pulse_width ({pulse_width}) is negative or zero"
                self._warnings.append(msg)
                self.Warning.suspicious_value(msg)
            elif pulse_width > 1e-3:  # >1ms is unusual
                msg = f"pulse_width ({pulse_width:.6f}s) exceeds 1ms (unusual)"
                self._warnings.append(msg)

        # Bandwidth
        bandwidth = self._metadata.get('bandwidth')
        if bandwidth is not None:
            if bandwidth <= 0:
                msg = f"bandwidth ({bandwidth}) is negative or zero"
                self._warnings.append(msg)
                self.Warning.suspicious_value(msg)

        # PRF (Pulse Repetition Frequency)
        prf = self._metadata.get('prf') or self._metadata.get('PRF')
        if prf is not None:
            if prf <= 0:
                msg = f"PRF ({prf}) is negative or zero"
                self._warnings.append(msg)
                self.Warning.suspicious_value(msg)
            elif prf > 10000:  # >10kHz is unusual
                msg = f"PRF ({prf:.1f} Hz) exceeds 10kHz (unusual for SAR)"
                self._warnings.append(msg)

    def _validate_timeline_params(self) -> None:
        """Validate timeline and acquisition parameters."""
        # Number of vectors/pulses
        num_vectors = self._metadata.get('num_vectors')
        if num_vectors is not None:
            if num_vectors <= 0:
                msg = f"num_vectors ({num_vectors}) is negative or zero"
                self._warnings.append(msg)
                self.Warning.suspicious_value(msg)

        # Check for timeline consistency
        start_time = self._metadata.get('collection_start')
        end_time = self._metadata.get('collection_end')
        if start_time and end_time:
            try:
                from datetime import datetime
                if isinstance(start_time, str):
                    start_time = datetime.fromisoformat(start_time)
                if isinstance(end_time, str):
                    end_time = datetime.fromisoformat(end_time)

                if end_time <= start_time:
                    msg = "collection_end is before or equal to collection_start"
                    self._warnings.append(msg)
                    self.Warning.suspicious_value(msg)
            except Exception:
                pass

    def _update_display(self) -> None:
        """Update UI with extracted metadata and validation results."""
        # Update metadata table
        self._metadata_table.setRowCount(0)

        # Priority fields to show first
        priority_fields = [
            'format',
            'rows',
            'cols',
            'dtype',
            'bands',
            'pulse_width',
            'bandwidth',
            'prf',
            'PRF',
            'num_vectors',
            'num_samples',
            'collection_start',
            'collection_end',
            'center_frequency',
            'polarization',
        ]

        # Add priority fields first
        for field in priority_fields:
            if field in self._metadata:
                self._add_table_row(field, self._metadata[field])

        # Add remaining fields
        for field, value in sorted(self._metadata.items()):
            if field not in priority_fields:
                self._add_table_row(field, value)

        # Update warnings panel
        if self._warnings:
            self._warnings_text.setPlainText('\n'.join(self._warnings))
            self._warnings_text.setStyleSheet("color: red;")
        else:
            self._warnings_text.setPlainText("No validation issues found.")
            self._warnings_text.setStyleSheet("color: green;")

        # Update JSON dump
        try:
            json_str = json.dumps(self._metadata, indent=2, default=str)
            self._json_text.setPlainText(json_str)
        except Exception as e:
            self._json_text.setPlainText(f"Error serializing metadata: {e}")

    def _add_table_row(self, param: str, value: Any) -> None:
        """Add a row to the metadata table."""
        row = self._metadata_table.rowCount()
        self._metadata_table.insertRow(row)

        # Parameter name
        item_param = QTableWidgetItem(param)
        self._metadata_table.setItem(row, 0, item_param)

        # Value (convert to string, handle nested structures)
        value_str = str(value)
        if len(value_str) > 100:
            value_str = value_str[:97] + "..."
        item_value = QTableWidgetItem(value_str)
        self._metadata_table.setItem(row, 1, item_value)

        # Status indicator
        status = "OK"
        status_color = QColor(200, 255, 200)  # Light green

        # Check if this parameter is in warnings
        for warning in self._warnings:
            if param in warning.lower():
                status = "WARNING"
                status_color = QColor(255, 200, 200)  # Light red
                break

        item_status = QTableWidgetItem(status)
        item_status.setBackground(status_color)
        self._metadata_table.setItem(row, 2, item_status)

    def _clear_display(self) -> None:
        """Clear all display elements."""
        self._format_label.setText("Format: (no data)")
        self._metadata_table.setRowCount(0)
        self._warnings_text.setPlainText("No data connected.")
        self._warnings_text.setStyleSheet("")
        self._json_text.setPlainText("")

    def _on_refresh(self) -> None:
        """Refresh metadata extraction and validation."""
        if self._stack:
            self._extract_metadata()
            self._validate_metadata()
            self._update_display()

    def _on_export(self) -> None:
        """Export metadata as JSON file."""
        from PyQt6.QtWidgets import QFileDialog

        if not self._metadata:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Metadata",
            "metadata.json",
            "JSON Files (*.json);;All Files (*)",
        )

        if not filepath:
            return

        try:
            with open(filepath, 'w') as f:
                json.dump(self._metadata, f, indent=2, default=str)
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self, "Export Error", f"Failed to export metadata:\n{e}"
            )
