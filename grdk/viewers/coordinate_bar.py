# -*- coding: utf-8 -*-
"""
CoordinateBar - Status bar displaying pixel position and geographic coordinates.

Shows pixel (row, col), optional lat/lon via grdl Geolocation, and the raw
pixel value under the cursor.  Designed to connect to any ImageCanvas via
the ``pixel_hovered`` signal.

Dependencies
------------
PyQt6

Author
------
Claude Code (Anthropic)

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-17

Modified
--------
2026-02-17
"""

# Standard library
from typing import Any, Optional

# Third-party
import numpy as np

try:
    from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget
    from PyQt6.QtCore import QTimer

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False


if _QT_AVAILABLE:

    class CoordinateBar(QWidget):
        """Status bar showing pixel coordinates, lat/lon, and pixel value.

        Connect to an ``ImageCanvas`` via ``connect_canvas()`` to receive
        live cursor position updates.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget.
        """

        _THROTTLE_MS = 33  # ~30 Hz max geolocation update rate

        def __init__(self, parent: Optional[Any] = None) -> None:
            super().__init__(parent)

            self._geolocation: Optional[Any] = None

            # Pending geolocation lookup
            self._pending_row: Optional[int] = None
            self._pending_col: Optional[int] = None
            self._throttle_timer = QTimer(self)
            self._throttle_timer.setSingleShot(True)
            self._throttle_timer.setInterval(self._THROTTLE_MS)
            self._throttle_timer.timeout.connect(self._do_geo_lookup)

            # Labels
            self._pixel_label = QLabel("Pixel: —")
            self._geo_label = QLabel("")
            self._value_label = QLabel("Value: —")

            layout = QHBoxLayout(self)
            layout.setContentsMargins(4, 2, 4, 2)
            layout.addWidget(self._pixel_label)
            layout.addWidget(self._geo_label)
            layout.addWidget(self._value_label)
            layout.addStretch(1)

            self.setFixedHeight(24)

        def set_geolocation(self, geo: Optional[Any]) -> None:
            """Set the geolocation model for lat/lon display.

            Parameters
            ----------
            geo : Optional[Geolocation]
                grdl Geolocation instance, or None to disable lat/lon.
            """
            self._geolocation = geo
            if geo is None:
                self._geo_label.setText("")

        def connect_canvas(self, canvas: Any) -> None:
            """Connect to an ImageCanvas's pixel_hovered signal.

            Parameters
            ----------
            canvas : ImageCanvas
                Canvas whose cursor position drives this bar.
            """
            canvas.pixel_hovered.connect(self._on_pixel_hovered)

        def _on_pixel_hovered(self, row: int, col: int, value: Any) -> None:
            """Handle cursor position update from canvas."""
            self._pixel_label.setText(f"Pixel: ({row}, {col})")
            self._format_value(value)

            if self._geolocation is not None:
                self._pending_row = row
                self._pending_col = col
                if not self._throttle_timer.isActive():
                    self._throttle_timer.start()
            else:
                self._geo_label.setText("")

        def _do_geo_lookup(self) -> None:
            """Perform the geolocation lookup (throttled)."""
            if self._pending_row is None or self._geolocation is None:
                return

            try:
                result = self._geolocation.image_to_latlon(
                    self._pending_row, self._pending_col
                )
                if isinstance(result, tuple) and len(result) >= 2:
                    lat, lon = result[0], result[1]
                    self._geo_label.setText(
                        f"Lat: {lat:.6f}\u00b0  Lon: {lon:.6f}\u00b0"
                    )
                else:
                    self._geo_label.setText("")
            except Exception:
                self._geo_label.setText("Lat/Lon: —")

        def _format_value(self, value: Any) -> None:
            """Format the pixel value for display."""
            if value is None:
                self._value_label.setText("Value: —")
                return

            # Numpy array (includes 0-d scalars from array indexing)
            if isinstance(value, np.ndarray):
                if np.iscomplexobj(value):
                    if value.ndim == 0:
                        self._value_label.setText(
                            f"Value: {self._fmt_complex(value)}"
                        )
                    else:
                        parts = [self._fmt_complex(v) for v in value.flat]
                        self._value_label.setText(
                            f"Value: [{', '.join(parts)}]"
                        )
                elif value.ndim == 0:
                    self._value_label.setText(
                        f"Value: {float(value):.4g}"
                    )
                else:
                    parts = [f"{float(v):.4g}" for v in value.flat]
                    self._value_label.setText(
                        f"Value: [{', '.join(parts)}]"
                    )
            # Numpy complex scalars (np.complex64, np.complex128, etc.)
            # np.complex64 does NOT inherit from Python complex, so
            # check np.complexfloating before the builtin complex check.
            elif isinstance(value, (np.complexfloating, complex)):
                self._value_label.setText(
                    f"Value: {self._fmt_complex(value)}"
                )
            elif isinstance(value, (int, float, np.integer, np.floating)):
                self._value_label.setText(f"Value: {float(value):.4g}")
            else:
                self._value_label.setText(f"Value: {value}")

        @staticmethod
        def _fmt_complex(val: Any) -> str:
            """Format a single complex value as magnitude∠phase."""
            mag = float(np.abs(val))
            phase = float(np.angle(val, deg=True))
            return f"{mag:.4g}\u2220{phase:.1f}\u00b0"

else:

    class CoordinateBar:  # type: ignore[no-redef]
        """Stub when Qt is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("Qt is required for CoordinateBar")
