# -*- coding: utf-8 -*-
"""
Polygon Drawing - Interactive polygon drawing state and rendering.

Manages polygon drawing state for ImageCanvas, including vertex collection,
rubber-band preview, and completed polygon rendering.

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
2026-06-23

Modified
--------
2026-06-23
"""

# Standard library
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

# Third-party
import numpy as np

try:
    from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsPolygonItem
    from PyQt6.QtGui import QBrush, QColor, QPen, QPolygonF
    from PyQt6.QtCore import QPointF, Qt

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False


if _QT_AVAILABLE:

    @dataclass
    class PolygonDrawingState:
        """State for interactive polygon drawing on a canvas.

        Attributes
        ----------
        active : bool
            Whether drawing mode is currently enabled.
        vertices : List[Tuple[float, float]]
            Current polygon vertices in scene coordinates (col, row).
        completed_polygons : List[np.ndarray]
            Stored completed polygons as (N, 2) arrays in (row, col) format.
        annotations : List[str]
            Annotation labels for each completed polygon (parallel to completed_polygons).
        vertex_items : List[QGraphicsEllipseItem]
            Visual markers for vertices being drawn.
        rubber_band_item : Optional[QGraphicsLineItem]
            Preview line from last vertex to mouse cursor.
        polygon_items : List[QGraphicsPolygonItem]
            Rendered completed polygons.
        deleted_stack : List[Tuple[np.ndarray, QGraphicsPolygonItem, Any, str]]
            Stack of deleted polygons for redo (vertices, item, scene, annotation).
        """

        active: bool = False
        vertices: List[Tuple[float, float]] = field(default_factory=list)
        completed_polygons: List[np.ndarray] = field(default_factory=list)
        annotations: List[str] = field(default_factory=list)
        vertex_items: List[QGraphicsEllipseItem] = field(default_factory=list)
        rubber_band_item: Optional[QGraphicsLineItem] = None
        polygon_items: List[QGraphicsPolygonItem] = field(default_factory=list)
        deleted_stack: List[Tuple[np.ndarray, QGraphicsPolygonItem, Any, str]] = field(default_factory=list)

        def clear_active_drawing(self) -> None:
            """Clear the current in-progress polygon."""
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
            """Clear all completed polygons and their visual items."""
            self.completed_polygons.clear()
            for item in self.polygon_items:
                scene = item.scene()
                if scene is not None:
                    scene.removeItem(item)
            self.polygon_items.clear()
            self.annotations.clear()
            self.deleted_stack.clear()  # Clear redo stack
        
        def remove_last_polygon(self) -> bool:
            """Remove the most recently added polygon (for undo).
            
            Returns
            -------
            bool
                True if a polygon was removed, False if there were no polygons.
            """
            if not self.completed_polygons:
                return False
            
            # Remove from storage and save for redo
            vertices = self.completed_polygons.pop()
            annotation = self.annotations.pop() if self.annotations else ""
            
            # Remove visual item and save for redo
            if self.polygon_items:
                item = self.polygon_items.pop()
                # Get scene reference BEFORE removing item
                scene = item.scene()
                if scene is not None:
                    scene.removeItem(item)
                # Store item, scene, and annotation for redo
                self.deleted_stack.append((vertices, item, scene, annotation))
            
            return True
        
        def redo_last_deletion(self) -> bool:
            """Restore the most recently deleted polygon (for redo).
            
            Returns
            -------
            bool
                True if a polygon was restored, False if nothing to redo.
            """
            if not self.deleted_stack:
                return False
            
            # Pop from redo stack (handle both old and new tuple formats)
            deleted = self.deleted_stack.pop()
            if len(deleted) == 4:
                vertices, item, scene, annotation = deleted
            else:
                # Legacy format without annotation
                vertices, item, scene = deleted
                annotation = ""
            
            # Restore to storage and visual items
            self.completed_polygons.append(vertices)
            self.annotations.append(annotation)
            self.polygon_items.append(item)
            
            # Re-add to scene
            if scene is not None:
                scene.addItem(item)
            
            return True
        
        def remove_polygon_at_index(self, index: int) -> bool:
            """Remove a specific polygon by index (for delete selected).
            
            Parameters
            ----------
            index : int
                Index of the polygon to remove.
            
            Returns
            -------
            bool
                True if polygon was removed, False if index was invalid.
            """
            if index < 0 or index >= len(self.completed_polygons):
                return False
            
            # Remove from storage and save for redo
            vertices = self.completed_polygons.pop(index)
            annotation = self.annotations.pop(index) if index < len(self.annotations) else ""
            
            # Remove visual item and save for redo
            if index < len(self.polygon_items):
                item = self.polygon_items.pop(index)
                # Get scene reference BEFORE removing item
                scene = item.scene()
                if scene is not None:
                    scene.removeItem(item)
                # Store item, scene, and annotation for redo
                self.deleted_stack.append((vertices, item, scene, annotation))
            
            return True
        
        def set_polygons_selectable(self, selectable: bool) -> None:
            """Enable or disable polygon selection.
            
            Parameters
            ----------
            selectable : bool
                True to make polygons selectable, False to disable selection.
            """
            for item in self.polygon_items:
                item.setFlag(
                    QGraphicsPolygonItem.GraphicsItemFlag.ItemIsSelectable,
                    selectable
                )


    def create_vertex_marker(scene: Any, x: float, y: float) -> QGraphicsEllipseItem:
        """Create a visual marker for a polygon vertex.

        Parameters
        ----------
        scene : QGraphicsScene
            Scene to add the marker to.
        x : float
            X coordinate (column) in scene space.
        y : float
            Y coordinate (row) in scene space.

        Returns
        -------
        QGraphicsEllipseItem
            The created vertex marker.
        """
        radius = 5.0
        item = QGraphicsEllipseItem(x - radius, y - radius, radius * 2, radius * 2)
        item.setPen(QPen(QColor(255, 255, 0), 2))  # Yellow outline
        item.setBrush(QBrush(QColor(255, 255, 0, 200)))  # Yellow fill
        item.setZValue(1000)  # Above image
        scene.addItem(item)
        return item


    def create_rubber_band(scene: Any, x1: float, y1: float, x2: float, y2: float) -> QGraphicsLineItem:
        """Create or update a rubber-band preview line.

        Parameters
        ----------
        scene : QGraphicsScene
            Scene to add the line to.
        x1, y1 : float
            Start point (last vertex).
        x2, y2 : float
            End point (current mouse position).

        Returns
        -------
        QGraphicsLineItem
            The rubber-band line.
        """
        item = QGraphicsLineItem(x1, y1, x2, y2)
        pen = QPen(QColor(255, 255, 0, 255), 3, Qt.PenStyle.DashLine)  # Bright yellow, thicker, dashed
        item.setPen(pen)
        item.setZValue(1001)  # Above vertices
        scene.addItem(item)
        return item


    def create_polygon_item(scene: Any, vertices: np.ndarray) -> QGraphicsPolygonItem:
        """Create a visual polygon from vertices.

        Parameters
        ----------
        scene : QGraphicsScene
            Scene to add the polygon to.
        vertices : np.ndarray
            Polygon vertices, shape (N, 2) in (row, col) format.

        Returns
        -------
        QGraphicsPolygonItem
            The rendered polygon.
        """
        # Convert (row, col) to QPointF (col, row) for Qt
        points = [QPointF(float(v[1]), float(v[0])) for v in vertices]
        polygon = QPolygonF(points)

        item = QGraphicsPolygonItem(polygon)
        item.setPen(QPen(QColor(255, 255, 0, 230), 2))  # Yellow stroke
        item.setBrush(QBrush(QColor(255, 255, 0, 50)))  # Translucent fill
        item.setZValue(998)  # Below vertices but above image
        
        # Polygons start as NOT selectable (enabled when exiting drawing mode)
        item.setFlag(QGraphicsPolygonItem.GraphicsItemFlag.ItemIsSelectable, False)
        
        scene.addItem(item)
        return item


else:
    # No-op stubs when Qt is not available
    @dataclass
    class PolygonDrawingState:  # type: ignore[no-redef]
        pass


    def create_vertex_marker(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        pass


    def create_rubber_band(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        pass


    def create_polygon_item(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        pass
