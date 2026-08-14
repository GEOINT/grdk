# -*- coding: utf-8 -*-
"""
GeoJSON Import - Load polygons from GeoJSON with strict validation.

Imports polygon features from GeoJSON files with geographic coordinate
conversion and bounds checking to ensure polygons align with the loaded
image.

Dependencies
------------
json, numpy, pathlib, GRDL readers and geolocation

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
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Third-party
import numpy as np

_log = logging.getLogger(__name__)


class GeoJSONImportError(Exception):
    """Raised when GeoJSON import fails validation."""
    pass


def import_polygons_from_geojson(
    geojson_path: str,
    reader: Optional[Any],
    geolocation: Optional[Any],
    image_shape: Tuple[int, int],
) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """Import polygons from GeoJSON with strict validation.
    
    Parameters
    ----------
    geojson_path : str
        Path to the GeoJSON file.
    reader : Any, optional
        GRDL reader instance (for metadata validation).
    geolocation : Any, optional
        Geolocation model for coordinate conversion.
    image_shape : tuple of (int, int)
        Image dimensions (rows, cols) for bounds checking.
    
    Returns
    -------
    tuple of (list of np.ndarray, list of str, list of str)
        - Imported polygons as (N, 2) arrays in (row, col) format
        - Imported annotations (parallel to polygons list)
        - List of warning messages about skipped polygons
    
    Raises
    ------
    GeoJSONImportError
        If validation fails or no polygons can be imported.
    """
    # 1. Load and validate GeoJSON structure
    try:
        with open(geojson_path, 'r', encoding='utf-8') as f:
            geojson = json.load(f)
    except json.JSONDecodeError as e:
        raise GeoJSONImportError(f"Invalid JSON format: {e}")
    except FileNotFoundError:
        raise GeoJSONImportError(f"File not found: {geojson_path}")
    except Exception as e:
        raise GeoJSONImportError(f"Could not read file: {e}")
    
    if geojson.get('type') != 'FeatureCollection':
        raise GeoJSONImportError(
            "Invalid GeoJSON: Expected 'FeatureCollection', "
            f"got '{geojson.get('type')}'"
        )
    
    features = geojson.get('features', [])
    if not features:
        raise GeoJSONImportError("GeoJSON contains no features")
    
    # 2. Determine coordinate conversion strategy
    if geolocation is not None:
        use_geographic = True
        _log.info("Using geographic coordinate conversion")
    else:
        # Check if pixel backup is available
        has_pixel_backup = any(
            'pixel_vertices' in f.get('properties', {})
            for f in features
        )
        if not has_pixel_backup:
            raise GeoJSONImportError(
                "Cannot import: Image has no geolocation and GeoJSON "
                "contains no pixel coordinate backup.\n\n"
                "This GeoJSON was likely exported from a different image "
                "with geolocation data."
            )
        use_geographic = False
        _log.info("Using pixel coordinate backup (no geolocation)")
    
    # 3. Bounds check (if using geographic coords)
    if use_geographic:
        image_bounds = _get_image_geographic_bounds(reader, geolocation, image_shape)
        geojson_bounds = _get_geojson_bounds(geojson)
        
        if not _bounds_overlap(image_bounds, geojson_bounds):
            raise GeoJSONImportError(
                "Cannot import: Polygon coordinates are completely outside "
                "the image area.\n\n"
                f"Image bounds:\n  {_format_bounds(image_bounds)}\n\n"
                f"GeoJSON bounds:\n  {_format_bounds(geojson_bounds)}\n\n"
                "This GeoJSON appears to be from a different location."
            )
    
    # 4. Convert and validate each polygon
    imported = []
    imported_annotations = []
    skipped = []
    
    for i, feature in enumerate(features):
        feature_id = feature.get('id', i)
        
        try:
            # Extract coordinates
            if use_geographic:
                coords = feature.get('geometry', {}).get('coordinates', [[]])[0]
                if not coords:
                    skipped.append(f"Feature {feature_id}: empty coordinates")
                    continue
                
                # Convert lat/lon to pixel row/col
                pixels = _convert_geographic_to_pixels(coords, geolocation)
            else:
                # Use pixel backup
                pixels = feature.get('properties', {}).get('pixel_vertices')
                if pixels is None:
                    skipped.append(f"Feature {feature_id}: missing pixel_vertices")
                    continue
                pixels = np.array(pixels, dtype=np.float64)
            
            # Validate polygon has at least 3 vertices
            if len(pixels) < 3:
                skipped.append(f"Feature {feature_id}: fewer than 3 vertices")
                continue
            
            # Check if at least ONE vertex is inside image bounds
            if not _any_vertex_in_image(pixels, image_shape):
                skipped.append(f"Feature {feature_id}: outside image bounds")
                continue
            
            # Extract annotation from properties
            annotation = feature.get('properties', {}).get('annotation', '')
            
            imported.append(pixels)
            imported_annotations.append(annotation)
            
        except Exception as e:
            _log.warning("Failed to import feature %s: %s", feature_id, e)
            skipped.append(f"Feature {feature_id}: conversion error ({e})")
            continue
    
    # 5. Validate results
    if not imported:
        if skipped:
            raise GeoJSONImportError(
                f"No polygons could be imported.\n\n"
                f"Skipped {len(skipped)} polygon(s):\n" +
                "\n".join(f"  • {s}" for s in skipped[:10]) +
                ("\n  ..." if len(skipped) > 10 else "")
            )
        else:
            raise GeoJSONImportError("GeoJSON contains no valid polygon features")
    
    _log.info("Imported %d polygon(s), skipped %d", len(imported), len(skipped))
    
    return imported, imported_annotations, skipped


def _get_image_geographic_bounds(
    reader: Any,
    geolocation: Any,
    image_shape: Tuple[int, int],
) -> Dict[str, float]:
    """Get geographic bounding box of image (only 4 corner points).
    
    Parameters
    ----------
    reader : Any
        GRDL reader (unused, kept for future metadata checks).
    geolocation : Any
        Geolocation model.
    image_shape : tuple of (int, int)
        Image dimensions (rows, cols).
    
    Returns
    -------
    dict
        Bounding box with keys: min_lat, max_lat, min_lon, max_lon
    """
    rows, cols = image_shape
    
    # Sample 4 corners
    corners = [
        (0, 0),           # top-left
        (0, cols - 1),    # top-right
        (rows - 1, 0),    # bottom-left
        (rows - 1, cols - 1)  # bottom-right
    ]
    
    lats = []
    lons = []
    
    for row, col in corners:
        try:
            result = geolocation.image_to_latlon(float(row), float(col))
            if isinstance(result, (tuple, list, np.ndarray)):
                lat, lon = result[0], result[1]
                lats.append(lat)
                lons.append(lon)
        except Exception as e:
            _log.warning("Failed to geolocate corner (%d, %d): %s", row, col, e)
    
    if not lats:
        raise GeoJSONImportError(
            "Could not determine image geographic bounds "
            "(geolocation failed for all corners)"
        )
    
    return {
        'min_lat': min(lats),
        'max_lat': max(lats),
        'min_lon': min(lons),
        'max_lon': max(lons),
    }


def _get_geojson_bounds(geojson: Dict) -> Dict[str, float]:
    """Extract bounding box from GeoJSON coordinates.
    
    Parameters
    ----------
    geojson : dict
        Parsed GeoJSON FeatureCollection.
    
    Returns
    -------
    dict
        Bounding box with keys: min_lat, max_lat, min_lon, max_lon
    """
    lats = []
    lons = []
    
    for feature in geojson.get('features', []):
        coords = feature.get('geometry', {}).get('coordinates', [[]])[0]
        for lon, lat in coords:
            lats.append(lat)
            lons.append(lon)
    
    if not lats:
        raise GeoJSONImportError("GeoJSON contains no coordinates")
    
    return {
        'min_lat': min(lats),
        'max_lat': max(lats),
        'min_lon': min(lons),
        'max_lon': max(lons),
    }


def _bounds_overlap(bounds1: Dict[str, float], bounds2: Dict[str, float]) -> bool:
    """Check if two bounding boxes overlap.
    
    Parameters
    ----------
    bounds1, bounds2 : dict
        Bounding boxes with keys: min_lat, max_lat, min_lon, max_lon
    
    Returns
    -------
    bool
        True if boxes overlap, False otherwise.
    """
    # Boxes don't overlap if one is completely to the side of the other
    return not (
        bounds2['max_lat'] < bounds1['min_lat'] or
        bounds2['min_lat'] > bounds1['max_lat'] or
        bounds2['max_lon'] < bounds1['min_lon'] or
        bounds2['min_lon'] > bounds1['max_lon']
    )


def _format_bounds(bounds: Dict[str, float]) -> str:
    """Format bounds dict as human-readable string.
    
    Parameters
    ----------
    bounds : dict
        Bounding box with lat/lon keys.
    
    Returns
    -------
    str
        Formatted string like "40.7°N to 40.8°N, 74.0°W to 73.9°W"
    """
    def fmt_lat(lat):
        return f"{abs(lat):.3f}°{'N' if lat >= 0 else 'S'}"
    
    def fmt_lon(lon):
        return f"{abs(lon):.3f}°{'E' if lon >= 0 else 'W'}"
    
    return (
        f"{fmt_lat(bounds['min_lat'])} to {fmt_lat(bounds['max_lat'])}, "
        f"{fmt_lon(bounds['min_lon'])} to {fmt_lon(bounds['max_lon'])}"
    )


def _convert_geographic_to_pixels(
    geo_coords: List,
    geolocation: Any,
) -> np.ndarray:
    """Convert GeoJSON lat/lon coordinates to pixel row/col.
    
    Parameters
    ----------
    geo_coords : list
        List of [lon, lat] pairs (GeoJSON format).
    geolocation : Any
        Geolocation model with from_latlon() method.
    
    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with (row, col) pixel coordinates.
    """
    pixel_coords = []
    
    for lon, lat in geo_coords:
        try:
            result = geolocation.latlon_to_image(lat, lon)
            if isinstance(result, (tuple, list, np.ndarray)) and len(result) >= 2:
                row, col = result[0], result[1]
                pixel_coords.append((row, col))
            else:
                _log.warning("Geolocation returned unexpected result: %s", result)
        except Exception as e:
            _log.warning("Failed to convert (%f, %f): %s", lat, lon, e)
    
    if not pixel_coords:
        raise ValueError("Could not convert any coordinates to pixels")
    
    return np.array(pixel_coords, dtype=np.float64)


def _any_vertex_in_image(
    pixels: np.ndarray,
    image_shape: Tuple[int, int],
) -> bool:
    """Check if at least one vertex is inside image bounds.
    
    Parameters
    ----------
    pixels : np.ndarray
        Polygon vertices, shape (N, 2) in (row, col) format.
    image_shape : tuple of (int, int)
        Image dimensions (rows, cols).
    
    Returns
    -------
    bool
        True if any vertex is inside image bounds.
    """
    rows, cols = image_shape
    
    for row, col in pixels:
        if 0 <= row < rows and 0 <= col < cols:
            return True
    
    return False
