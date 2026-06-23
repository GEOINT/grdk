# -*- coding: utf-8 -*-
"""
GeoJSON Export - Export polygon geometries to GeoJSON format.

Converts image pixel coordinates to geographic coordinates (if geolocation
is available) and serializes polygons as GeoJSON FeatureCollection with
metadata enrichment from reader provenance.

Dependencies
------------
None (stdlib only)

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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party
import numpy as np

_log = logging.getLogger("grdk.geojson_export")


def export_polygons_to_geojson(
    polygons: List[np.ndarray],
    reader: Any,
    geolocation: Optional[Any],
    output_path: str,
    label_class: str = "roi",
) -> None:
    """Export polygons to GeoJSON FeatureCollection.

    Converts pixel (row, col) coordinates to geographic (lon, lat) if
    geolocation is available. Enriches features with reader metadata
    (sensor, polarization, acquisition time, etc.).

    Parameters
    ----------
    polygons : List[np.ndarray]
        List of polygon vertex arrays, each shape (N, 2) in (row, col) format.
    reader : ImageReader
        GRDL reader instance for metadata extraction.
    geolocation : Optional[Geolocation]
        GRDL geolocation instance for coordinate transformation, or None.
    output_path : str
        Path to write the GeoJSON file (.geojson or .json).
    label_class : str, optional
        Label class name for all polygons (default: "roi").

    Raises
    ------
    IOError
        If the output file cannot be written.
    """
    features = []

    # Extract reader metadata once
    reader_meta = _extract_reader_metadata(reader)

    for i, vertices in enumerate(polygons):
        if len(vertices) < 3:
            _log.warning(f"Skipping polygon {i}: fewer than 3 vertices")
            continue

        feature = _polygon_to_feature(
            vertices=vertices,
            geolocation=geolocation,
            reader_meta=reader_meta,
            label_class=label_class,
            feature_id=i,
        )
        features.append(feature)

    # Build FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    # Add CRS only if we have geographic coordinates
    if geolocation is not None:
        geojson["crs"] = {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        }

    # Write to file
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path_obj, 'w') as f:
        json.dump(geojson, f, indent=2)

    _log.info(f"Exported {len(features)} polygon(s) to {output_path}")


def _extract_reader_metadata(reader: Any) -> Dict[str, Any]:
    """Extract provenance metadata from a reader.

    Parameters
    ----------
    reader : ImageReader
        GRDL reader instance.

    Returns
    -------
    Dict[str, Any]
        Metadata dictionary with keys: sensor, polarization, acquisition_time,
        source_image_id.
    """
    meta = {}

    # Source image ID (filename or product ID)
    if hasattr(reader, 'filepath'):
        meta['source_image_id'] = Path(reader.filepath).name
    elif hasattr(reader, 'product_id'):
        meta['source_image_id'] = reader.product_id
    else:
        meta['source_image_id'] = type(reader).__name__

    # Sensor type
    reader_class = type(reader).__name__
    sensor_map = {
        'SICDReader': 'SICD SAR',
        'Sentinel1SLCReader': 'Sentinel-1',
        'NISARReader': 'NISAR',
        'BIOMASSReader': 'BIOMASS',
        'TerraSARXReader': 'TerraSAR-X',
    }
    meta['sensor'] = sensor_map.get(reader_class, reader_class)

    # Polarization (use canonical pattern from _pol_utils)
    try:
        from grdk.widgets._pol_utils import _reader_polarization
        pol = _reader_polarization(reader)
        if pol:
            meta['polarization'] = pol
    except ImportError:
        pass

    # Acquisition time
    if hasattr(reader, 'metadata'):
        reader_metadata = reader.metadata
        if hasattr(reader_metadata, 'acquisition_time'):
            meta['acquisition_time'] = str(reader_metadata.acquisition_time)
        elif hasattr(reader_metadata, 'start_time'):
            meta['acquisition_time'] = str(reader_metadata.start_time)
        elif isinstance(reader_metadata, dict):
            meta['acquisition_time'] = reader_metadata.get('acquisition_time') or reader_metadata.get('start_time')

    return meta


def _polygon_to_feature(
    vertices: np.ndarray,
    geolocation: Optional[Any],
    reader_meta: Dict[str, Any],
    label_class: str,
    feature_id: int,
) -> Dict[str, Any]:
    """Convert a polygon to a GeoJSON Feature.

    Parameters
    ----------
    vertices : np.ndarray
        Polygon vertices, shape (N, 2) in (row, col) format.
    geolocation : Optional[Geolocation]
        GRDL geolocation for coordinate transform, or None.
    reader_meta : Dict[str, Any]
        Reader metadata from _extract_reader_metadata.
    label_class : str
        Label class name.
    feature_id : int
        Unique feature ID.

    Returns
    -------
    Dict[str, Any]
        GeoJSON Feature dict.
    """
    # Store original pixel coordinates as backup
    pixel_coords = vertices.tolist()

    # Convert to geographic if geolocation is available
    if geolocation is not None:
        try:
            geo_coords = _vertices_to_geographic(vertices, geolocation)
            coordinate_system = "geographic"
        except Exception as e:
            _log.warning(f"Geolocation failed for polygon {feature_id}: {e}, using pixel coordinates")
            geo_coords = [[float(v[1]), float(v[0])] for v in vertices]  # (col, row) as (x, y)
            coordinate_system = "image_pixel"
    else:
        # No geolocation — use pixel coords as (x, y)
        geo_coords = [[float(v[1]), float(v[0])] for v in vertices]
        coordinate_system = "image_pixel"

    # Close the ring if not already closed
    if len(geo_coords) > 0 and geo_coords[0] != geo_coords[-1]:
        geo_coords.append(geo_coords[0])

    # Build Feature
    feature = {
        "type": "Feature",
        "id": feature_id,
        "geometry": {
            "type": "Polygon",
            "coordinates": [geo_coords],  # Exterior ring
        },
        "properties": {
            "label_class": label_class,
            "creation_timestamp": datetime.utcnow().isoformat() + 'Z',
            "pixel_vertices": pixel_coords,
            "coordinate_system": coordinate_system,
            "grdk_version": "0.1.0",
        },
    }

    # Merge reader metadata into properties
    feature["properties"].update(reader_meta)

    return feature


def _vertices_to_geographic(vertices: np.ndarray, geolocation: Any) -> List[List[float]]:
    """Convert pixel vertices to geographic coordinates.

    Parameters
    ----------
    vertices : np.ndarray
        Polygon vertices, shape (N, 2) in (row, col) format.
    geolocation : Geolocation
        GRDL geolocation with image_to_latlon method.

    Returns
    -------
    List[List[float]]
        List of [longitude, latitude] pairs (GeoJSON order).

    Raises
    ------
    ValueError
        If geolocation transform fails.
    """
    geo_coords = []

    for row, col in vertices:
        result = geolocation.image_to_latlon(float(row), float(col))
        if isinstance(result, tuple) and len(result) >= 2:
            lat, lon = float(result[0]), float(result[1])
            geo_coords.append([lon, lat])  # GeoJSON: [longitude, latitude]
        else:
            raise ValueError(f"Geolocation returned invalid result: {result}")

    return geo_coords
