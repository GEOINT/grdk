# -*- coding: utf-8 -*-
"""
VectorOverlayLayer - GeoJSON vector rendering over image canvas.

Renders GeoJSON features (Points, LineStrings, Polygons) as QGraphicsItems
positioned in source image pixel coordinates.  Geographic coordinates are
transformed via grdl Geolocation.  Without geolocation, coordinates are
interpreted as pixel (col, row) directly.

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
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Third-party
import numpy as np

try:
    from PyQt6.QtWidgets import (
        QGraphicsEllipseItem,
        QGraphicsPathItem,
        QGraphicsPolygonItem,
        QGraphicsScene,
    )
    from PyQt6.QtGui import QBrush, QColor, QPainterPath, QPen, QPolygonF
    from PyQt6.QtCore import QPointF

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False


if _QT_AVAILABLE:

    # Default styling
    _DEFAULT_STROKE = QColor(255, 255, 0, 230)       # yellow
    _DEFAULT_FILL = QColor(255, 255, 0, 50)           # translucent yellow
    _DEFAULT_POINT_FILL = QColor(255, 80, 80, 200)    # red
    _DEFAULT_STROKE_WIDTH = 2.0
    _DEFAULT_POINT_RADIUS = 5.0

    class VectorOverlayLayer:
        """Renders GeoJSON features as QGraphicsItems in a scene.

        Features are positioned in source image pixel coordinates.
        If a ``Geolocation`` object is provided, geographic (lon, lat)
        coordinates are transformed to pixel (row, col).  Without
        geolocation, coordinates are interpreted as (col, row) directly.

        Parameters
        ----------
        scene : QGraphicsScene
            The graphics scene to add overlay items to.
        geolocation : optional
            grdl Geolocation instance for coordinate transforms.
        """

        def __init__(
            self,
            scene: QGraphicsScene,
            geolocation: Optional[Any] = None,
        ) -> None:
            self._scene = scene
            self._geolocation = geolocation
            self._items: List[Any] = []
            self._visible = True

            # Style
            self._stroke_color = _DEFAULT_STROKE
            self._fill_color = _DEFAULT_FILL
            self._stroke_width = _DEFAULT_STROKE_WIDTH
            self._point_radius = _DEFAULT_POINT_RADIUS

        @property
        def feature_count(self) -> int:
            """Number of rendered feature items."""
            return len(self._items)

        def set_geolocation(self, geolocation: Optional[Any]) -> None:
            """Update the geolocation model.

            Parameters
            ----------
            geolocation : optional
                grdl Geolocation instance, or None.
            """
            self._geolocation = geolocation

        def load_geojson(self, filepath: str) -> None:
            """Load and render features from a GeoJSON file.

            Parameters
            ----------
            filepath : str
                Path to a GeoJSON file (.geojson or .json).

            Raises
            ------
            FileNotFoundError
                If the file does not exist.
            ValueError
                If the file is not valid GeoJSON.
            """
            path = Path(filepath)
            if not path.exists():
                raise FileNotFoundError(f"GeoJSON file not found: {filepath}")

            with open(path, 'r') as f:
                data = json.load(f)

            if data.get('type') == 'FeatureCollection':
                features = data.get('features', [])
            elif data.get('type') == 'Feature':
                features = [data]
            elif data.get('type') in (
                'Point', 'MultiPoint', 'LineString', 'MultiLineString',
                'Polygon', 'MultiPolygon',
            ):
                features = [{'type': 'Feature', 'geometry': data, 'properties': {}}]
            else:
                raise ValueError(f"Unrecognized GeoJSON type: {data.get('type')}")

            self.load_features(features)

        def load_features(self, features: List[Dict]) -> None:
            """Render a list of GeoJSON Feature dicts.

            Parameters
            ----------
            features : List[Dict]
                GeoJSON Feature objects with 'geometry' and 'properties'.
            """
            for feature in features:
                geom = feature.get('geometry')
                props = feature.get('properties', {})
                if geom is None:
                    continue
                self._render_geometry(geom, props)

        def set_visible(self, visible: bool) -> None:
            """Show or hide all overlay items.

            Parameters
            ----------
            visible : bool
            """
            self._visible = visible
            for item in self._items:
                item.setVisible(visible)

        def set_style(
            self,
            stroke_color: Optional[QColor] = None,
            fill_color: Optional[QColor] = None,
            stroke_width: Optional[float] = None,
        ) -> None:
            """Update rendering style for future and existing items.

            Parameters
            ----------
            stroke_color : QColor, optional
            fill_color : QColor, optional
            stroke_width : float, optional
            """
            if stroke_color is not None:
                self._stroke_color = stroke_color
            if fill_color is not None:
                self._fill_color = fill_color
            if stroke_width is not None:
                self._stroke_width = stroke_width

            # Update existing items
            pen = QPen(self._stroke_color, self._stroke_width)
            brush = QBrush(self._fill_color)
            for item in self._items:
                item.setPen(pen)
                if hasattr(item, 'setBrush'):
                    item.setBrush(brush)

        def clear(self) -> None:
            """Remove all overlay items from the scene."""
            for item in self._items:
                self._scene.removeItem(item)
            self._items.clear()

        # --- Internal rendering ---

        def _render_geometry(self, geom: Dict, props: Dict) -> None:
            """Dispatch geometry rendering by type."""
            gtype = geom.get('type', '')
            coords = geom.get('coordinates', [])

            if gtype == 'Point':
                self._render_point(coords, props)
            elif gtype == 'MultiPoint':
                for pt in coords:
                    self._render_point(pt, props)
            elif gtype == 'LineString':
                self._render_linestring(coords, props)
            elif gtype == 'MultiLineString':
                for line in coords:
                    self._render_linestring(line, props)
            elif gtype == 'Polygon':
                self._render_polygon(coords, props)
            elif gtype == 'MultiPolygon':
                for poly in coords:
                    self._render_polygon(poly, props)

        def _geo_to_pixel(
            self, coords: Sequence,
        ) -> List[QPointF]:
            """Convert geographic or raw coordinates to scene QPointFs.

            Parameters
            ----------
            coords : Sequence
                List of [lon, lat] or [col, row] coordinate pairs.

            Returns
            -------
            List[QPointF]
                Points in scene (pixel) coordinates.
            """
            if not coords:
                return []

            coords_arr = np.array(coords, dtype=np.float64)
            if coords_arr.ndim == 1:
                coords_arr = coords_arr.reshape(1, -1)

            if self._geolocation is not None:
                # GeoJSON coordinates are [longitude, latitude]
                lons = coords_arr[:, 0]
                lats = coords_arr[:, 1]
                try:
                    result = self._geolocation.latlon_to_image(lats, lons)
                    if isinstance(result, tuple) and len(result) >= 2:
                        rows, cols = result[0], result[1]
                    else:
                        return []
                except Exception:
                    return []

                if isinstance(rows, (int, float)):
                    return [QPointF(float(cols), float(rows))]

                return [
                    QPointF(float(c), float(r))
                    for r, c in zip(rows, cols)
                ]
            else:
                # No geolocation — interpret as (col, row)
                return [
                    QPointF(float(c[0]), float(c[1]))
                    for c in coords_arr
                ]

        def _render_point(self, coords: Sequence, props: Dict) -> None:
            """Render a single Point."""
            points = self._geo_to_pixel([coords])
            if not points:
                return

            pt = points[0]
            r = self._point_radius
            item = QGraphicsEllipseItem(
                pt.x() - r, pt.y() - r, r * 2, r * 2,
            )
            pen = QPen(self._stroke_color, self._stroke_width)
            item.setPen(pen)
            item.setBrush(QBrush(_DEFAULT_POINT_FILL))
            item.setZValue(10)  # Above image tiles
            item.setToolTip(self._format_tooltip(props))
            item.setVisible(self._visible)

            self._scene.addItem(item)
            self._items.append(item)

        def _render_linestring(self, coords: Sequence, props: Dict) -> None:
            """Render a LineString."""
            points = self._geo_to_pixel(coords)
            if len(points) < 2:
                return

            path = QPainterPath()
            path.moveTo(points[0])
            for pt in points[1:]:
                path.lineTo(pt)

            item = QGraphicsPathItem(path)
            pen = QPen(self._stroke_color, self._stroke_width)
            item.setPen(pen)
            item.setZValue(10)
            item.setToolTip(self._format_tooltip(props))
            item.setVisible(self._visible)

            self._scene.addItem(item)
            self._items.append(item)

        def _render_polygon(self, rings: Sequence, props: Dict) -> None:
            """Render a Polygon (exterior ring only for now)."""
            if not rings:
                return

            # Exterior ring is the first ring
            points = self._geo_to_pixel(rings[0])
            if len(points) < 3:
                return

            polygon = QPolygonF(points)
            item = QGraphicsPolygonItem(polygon)

            pen = QPen(self._stroke_color, self._stroke_width)
            item.setPen(pen)
            item.setBrush(QBrush(self._fill_color))
            item.setZValue(10)
            item.setToolTip(self._format_tooltip(props))
            item.setVisible(self._visible)

            self._scene.addItem(item)
            self._items.append(item)

        @staticmethod
        def _format_tooltip(props: Dict) -> str:
            """Format feature properties as a tooltip string."""
            if not props:
                return ""
            lines = [f"{k}: {v}" for k, v in props.items() if v is not None]
            return "\n".join(lines[:10])  # Cap at 10 lines

else:

    class VectorOverlayLayer:  # type: ignore[no-redef]
        """Stub when Qt is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("Qt is required for VectorOverlayLayer")
