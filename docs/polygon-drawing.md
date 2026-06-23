# Interactive Polygon Drawing Feature

## Overview

GRDK now supports interactive polygon drawing on the viewer canvas for creating regions of interest (ROIs) and training labels for algorithm development. Polygons can be exported to GeoJSON format with full geolocation metadata.

## Usage

### GUI Workflow (ViewerMainWindow)

1. **Load an Image**
   - Open any supported image format (SAR, EO/IR, etc.) using File → Open or the toolbar button

2. **Enter Drawing Mode**
   - Click the "Draw Polygon" button in the toolbar (shortcut: `D`)
   - The cursor changes to a crosshair
   - Status bar shows: "Drawing Mode: Click to add vertex | Double-click or Enter to close | Esc to cancel"

3. **Draw a Polygon**
   - **Left-click** to add each vertex
   - A yellow circle marks each vertex
   - A dashed line previews from the last vertex to your cursor
   - **Double-click** or press **Enter** to close the polygon
   - Press **Escape** to cancel and discard the current polygon

4. **Export to GeoJSON**
   - Tools → "Export Polygons as GeoJSON..."
   - Polygons are automatically converted to geographic coordinates (lat/lon) if geolocation is available
   - Falls back to pixel coordinates if no geolocation is available

5. **Clear Polygons**
   - Tools → "Clear Polygons" to remove all drawn polygons from the active pane

### Important Notes

- **GeoJSON Import Not Yet Implemented**: The current version only supports **export**. To edit existing polygons, you'll need to redraw them manually. GeoJSON import is planned for a future release.
- **Double-click to close**: If double-click doesn't work reliably, use the **Enter** key instead to close the polygon
- **Rubber-band preview**: A dashed yellow line should show from your last vertex to the cursor position while drawing

### Programmatic Usage

```python
from grdk.viewers.image_canvas import ImageCanvas
from grdk.viewers.geojson_export import export_polygons_to_geojson
import numpy as np

# Create a canvas and load an image
canvas = ImageCanvas()
canvas.set_array(my_image_array)

# Enter drawing mode
canvas.enter_polygon_mode()

# ... user draws polygons interactively ...

# Get completed polygons
polygons = canvas.get_completed_polygons()  # List[np.ndarray]

# Export to GeoJSON
export_polygons_to_geojson(
    polygons=polygons,
    reader=my_reader,
    geolocation=my_geolocation,  # or None for pixel coords
    output_path="rois.geojson",
    label_class="roi",
)

# Clear all polygons
canvas.clear_all_polygons()

# Exit drawing mode
canvas.exit_polygon_mode()
```

## GeoJSON Schema

Exported GeoJSON files follow this structure:

```json
{
  "type": "FeatureCollection",
  "crs": {
    "type": "name",
    "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
  },
  "features": [
    {
      "type": "Feature",
      "id": 0,
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[lon1, lat1], [lon2, lat2], ...]]
      },
      "properties": {
        "label_class": "roi",
        "source_image_id": "sentinel1_slc_20260622_HH",
        "sensor": "Sentinel-1",
        "polarization": "HH",
        "acquisition_time": "2026-06-22T12:34:56Z",
        "creation_timestamp": "2026-06-22T15:00:00Z",
        "pixel_vertices": [[row1, col1], [row2, col2], ...],
        "coordinate_system": "geographic",
        "grdk_version": "0.1.0"
      }
    }
  ]
}
```

### Property Fields

- **`label_class`**: Classification label (default: "roi")
- **`source_image_id`**: Image filename or product ID
- **`sensor`**: Sensor type (e.g., "Sentinel-1", "SICD SAR", "NISAR")
- **`polarization`**: SAR polarization (e.g., "HH", "VV", "HV")
- **`acquisition_time`**: Image acquisition timestamp
- **`creation_timestamp`**: When the polygon was drawn
- **`pixel_vertices`**: Original pixel coordinates (backup for round-trip editing)
- **`coordinate_system`**: "geographic" or "image_pixel"
- **`grdk_version`**: GRDK version used for export

## Coordinate Systems

### Geographic Coordinates (with Geolocation)

When a `Geolocation` object is available, polygon vertices are converted from image pixel coordinates `(row, col)` to geographic coordinates `(longitude, latitude)` using:

```python
lat, lon = geolocation.image_to_latlon(row, col)
```

This works automatically for:
- SICD SAR
- Sentinel-1 SLC
- NISAR
- BIOMASS
- TerraSAR-X
- GeoTIFF with GCPs or affine transform

### Pixel Coordinates (no Geolocation)

If no geolocation is available (e.g., generic TIFF without georeferencing), polygons are exported in image pixel space with `coordinate_system: "image_pixel"`.

## Architecture

### Components

1. **`polygon_drawing.py`**
   - `PolygonDrawingState`: Manages active and completed polygons
   - Helper functions for creating Qt graphics items (vertices, rubber-band, polygons)

2. **`geojson_export.py`**
   - `export_polygons_to_geojson()`: Converts polygons to GeoJSON with metadata enrichment
   - Handles coordinate transformation via GRDL geolocation

3. **`image_canvas.py`** (extended)
   - `enter_polygon_mode()` / `exit_polygon_mode()`: Toggle drawing state
   - `get_completed_polygons()`: Retrieve all drawn polygons
   - `clear_all_polygons()`: Remove all polygons
   - `polygon_completed` signal: Emitted when a polygon is closed
   - Mouse/keyboard event handlers for interactive drawing

4. **`main_window.py`** (extended)
   - "Draw Polygon" toolbar action (checkable)
   - "Export Polygons as GeoJSON..." menu action
   - "Clear Polygons" menu action

### Design Principles

- **Simple State Machine**: Drawing mode is a boolean flag; polygons are stored as numpy arrays
- **Pure Pixel Storage**: Polygons are stored in image pixel coordinates `(row, col)`; geolocation conversion happens on-demand during export
- **Qt Graphics Overlay**: Polygons render as `QGraphicsPolygonItem` above the image pixmap
- **Geolocation Agnostic**: Works with or without geolocation; falls back to pixel coordinates gracefully
- **Minimal UI Clutter**: All drawing controls are in the toolbar/menu; no separate polygon management panel

## Future Enhancements

- Polygon editing (drag vertices, delete individual polygons)
- GeoJSON import (load existing polygons and render on canvas)
- Multiple label classes (dropdown to set label per polygon)
- Polygon list panel (view, select, rename polygons)
- Integration with Orange workflow (PolygonSetSignal → OWChipper)
- Automatic chip extraction on polygon completion
