# -*- coding: utf-8 -*-
"""
Polygon Drawing & Vector Geometry Tools - Interactive point, line, and polygon drawing.

Manages vector drawing state for ImageCanvas, including points, polylines,
polygons, vertex collection, rubber-band preview, and completed geometry rendering.

Dependencies
------------
PyQt6, numpy

Author
------
Claude Code / Steven Siebert

License
-------
MIT License
Copyright (c) 2026 geoint.org
"""

from __future__ import annotations

# Standard library
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

# Third-party
import numpy as np

try:
    from PyQt6.QtWidgets import (
        QGraphicsEllipseItem,
        QGraphicsLineItem,
        QGraphicsPolygonItem,
        QGraphicsPathItem,
        QGraphicsItem,
    )
    from PyQt6.QtGui import QBrush, QColor, QPainterPath, QPen, QPolygonF
    from PyQt6.QtCore import QPointF, Qt

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False


if _QT_AVAILABLE:

    @dataclass
    class PolygonDrawingState:
        """State for interactive geometry drawing (points, lines, polygons)."""

        active: bool = False
        mode: str = "polygon"  # "polygon", "point", "line"
        vertices: List[Tuple[float, float]] = field(default_factory=list)
        completed_polygons: List[np.ndarray] = field(default_factory=list)
        annotations: List[str] = field(default_factory=list)
        geometry_types: List[str] = field(default_factory=list)  # "polygon", "point", "line"
        vertex_items: List[QGraphicsEllipseItem] = field(default_factory=list)
        rubber_band_item: Optional[QGraphicsLineItem] = None
        polygon_items: List[QGraphicsItem] = field(default_factory=list)
        deleted_stack: List[Tuple[np.ndarray, QGraphicsItem, Any, str, str]] = field(default_factory=list)

        def clear_active_drawing(self) -> None:
            """Clear the current in-progress geometry."""
            self.vertices.clear()
            for item in self.vertex_items:
                scene = item.scene()
                if scene is not None:
                    scene.removeItem(item)
            self.vertex_items.clear()
            if self.rubber_band_item is not None:
                scene = self.rubber_band_item.scene()
                if scene is not None:
                    scene.removeItem(self.rubber_band_item)
                self.rubber_band_item = None

        def clear_all_polygons(self) -> None:
            """Clear all completed geometries and their visual items."""
            self.completed_polygons.clear()
            for item in self.polygon_items:
                scene = item.scene()
                if scene is not None:
                    scene.removeItem(item)
            self.polygon_items.clear()
            self.annotations.clear()
            self.geometry_types.clear()
            self.deleted_stack.clear()

        def remove_last_polygon(self) -> bool:
            """Remove the most recently added geometry (for undo)."""
            if not self.completed_polygons:
                return False

            vertices = self.completed_polygons.pop()
            annotation = self.annotations.pop() if self.annotations else ""
            geom_type = self.geometry_types.pop() if self.geometry_types else "polygon"

            if self.polygon_items:
                item = self.polygon_items.pop()
                scene = item.scene()
                if scene is not None:
                    scene.removeItem(item)
                self.deleted_stack.append((vertices, item, scene, annotation, geom_type))

            return True

        def redo_last_deletion(self) -> bool:
            """Restore the most recently deleted geometry (for redo)."""
            if not self.deleted_stack:
                return False

            deleted = self.deleted_stack.pop()
            if len(deleted) == 5:
                vertices, item, scene, annotation, geom_type = deleted
            elif len(deleted) == 4:
                vertices, item, scene, annotation = deleted
                geom_type = "polygon"
            else:
                vertices, item, scene = deleted
                annotation = ""
                geom_type = "polygon"

            self.completed_polygons.append(vertices)
            self.annotations.append(annotation)
            self.geometry_types.append(geom_type)
            self.polygon_items.append(item)

            if scene is not None and item.scene() is None:
                scene.addItem(item)

            return True

        def remove_polygon_at_index(self, index: int) -> bool:
            """Remove a specific geometry by index."""
            if index < 0 or index >= len(self.completed_polygons):
                return False

            vertices = self.completed_polygons.pop(index)
            annotation = self.annotations.pop(index) if index < len(self.annotations) else ""
            geom_type = self.geometry_types.pop(index) if index < len(self.geometry_types) else "polygon"

            if index < len(self.polygon_items):
                item = self.polygon_items.pop(index)
                scene = item.scene()
                if scene is not None:
                    scene.removeItem(item)
                self.deleted_stack.append((vertices, item, scene, annotation, geom_type))

            return True

        def set_polygons_selectable(self, selectable: bool) -> None:
            """Enable or disable geometry selection."""
            for item in self.polygon_items:
                item.setFlag(
                    QGraphicsItem.GraphicsItemFlag.ItemIsSelectable,
                    selectable,
                )


    def create_vertex_marker(scene: Any, x: float, y: float) -> QGraphicsEllipseItem:
        """Create a visual marker for a vertex."""
        radius = 5.0
        item = QGraphicsEllipseItem(x - radius, y - radius, radius * 2, radius * 2)
        item.setPen(QPen(QColor(255, 255, 0), 2))
        item.setBrush(QBrush(QColor(255, 255, 0, 200)))
        item.setZValue(1000)
        scene.addItem(item)
        return item


    def create_rubber_band(scene: Any, x1: float, y1: float, x2: float, y2: float) -> QGraphicsLineItem:
        """Create or update a rubber-band preview line."""
        item = QGraphicsLineItem(x1, y1, x2, y2)
        pen = QPen(QColor(255, 255, 0, 255), 3, Qt.PenStyle.DashLine)
        item.setPen(pen)
        item.setZValue(1001)
        scene.addItem(item)
        return item


    def create_polygon_item(scene: Any, vertices: np.ndarray) -> QGraphicsPolygonItem:
        """Create a visual polygon from vertices."""
        points = [QPointF(float(v[1]), float(v[0])) for v in vertices]
        polygon = QPolygonF(points)

        item = QGraphicsPolygonItem(polygon)
        item.setPen(QPen(QColor(255, 255, 0, 230), 2))
        item.setBrush(QBrush(QColor(255, 255, 0, 50)))
        item.setZValue(998)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)

        scene.addItem(item)
        return item


    def create_point_item(scene: Any, point: Tuple[float, float]) -> QGraphicsEllipseItem:
        """Create a visual point marker."""
        radius = 6.0
        row, col = point
        item = QGraphicsEllipseItem(col - radius, row - radius, radius * 2, radius * 2)
        item.setPen(QPen(QColor(0, 255, 255, 255), 2))  # Cyan stroke
        item.setBrush(QBrush(QColor(0, 255, 255, 180)))
        item.setZValue(999)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        scene.addItem(item)
        return item


    def create_line_item(scene: Any, vertices: np.ndarray) -> QGraphicsPathItem:
        """Create a visual polyline from vertices."""
        path = QPainterPath()
        if len(vertices) > 0:
            path.moveTo(float(vertices[0, 1]), float(vertices[0, 0]))
            for v in vertices[1:]:
                path.lineTo(float(v[1]), float(v[0]))

        item = QGraphicsPathItem(path)
        item.setPen(QPen(QColor(0, 255, 128, 240), 2, Qt.PenStyle.SolidLine))  # Spring green
        item.setZValue(998)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        scene.addItem(item)
        return item


else:
    @dataclass
    class PolygonDrawingState:
        pass

    def create_vertex_marker(*args: Any, **kwargs: Any) -> None:
        pass

    def create_rubber_band(*args: Any, **kwargs: Any) -> None:
        pass

    def create_polygon_item(*args: Any, **kwargs: Any) -> None:
        pass

    def create_point_item(*args: Any, **kwargs: Any) -> None:
        pass

    def create_line_item(*args: Any, **kwargs: Any) -> None:
        pass
