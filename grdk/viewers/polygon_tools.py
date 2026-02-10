# -*- coding: utf-8 -*-
"""
Polygon Tools - Utilities for polygon-based image chipping.

Provides functions for extracting image chips at polygon locations,
computing polygon bounding boxes, and converting between polygon
representations.

Author
------
Claude Code (Anthropic)

Contributor
-----------
Steven Siebert

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
from typing import Any, Dict, List, Optional, Tuple

# Third-party
import numpy as np

# GRDK internal
from grdl_rt.execution.chip import Chip, ChipLabel, ChipSet, PolygonRegion


def polygon_bounding_box(
    vertices: np.ndarray,
) -> Tuple[int, int, int, int]:
    """Compute the axis-aligned bounding box for a polygon.

    Parameters
    ----------
    vertices : np.ndarray
        Polygon vertices, shape (N, 2) where columns are (row, col).

    Returns
    -------
    Tuple[int, int, int, int]
        (row_start, row_end, col_start, col_end) suitable for
        array slicing (start inclusive, end exclusive).
    """
    row_min = int(np.floor(vertices[:, 0].min()))
    row_max = int(np.ceil(vertices[:, 0].max()))
    col_min = int(np.floor(vertices[:, 1].min()))
    col_max = int(np.ceil(vertices[:, 1].max()))
    return row_min, row_max, col_min, col_max


def chip_stack_at_polygon(
    readers: list,
    names: List[str],
    polygon: np.ndarray,
    registration_results: Optional[list] = None,
    timestamps: Optional[List[str]] = None,
) -> List[Chip]:
    """Extract chips from all images in a stack at a polygon location.

    Chips are extracted using the polygon's bounding box in pixel space.
    If registration results are available, the polygon coordinates are
    transformed to each image's native space before chipping.

    Parameters
    ----------
    readers : list
        GRDL ImageReader instances.
    names : List[str]
        Display names for each reader.
    polygon : np.ndarray
        Polygon vertices, shape (N, 2) in (row, col) format.
    registration_results : Optional[list]
        Registration results (one per reader, None for reference).
    timestamps : Optional[List[str]]
        Acquisition timestamps per image.

    Returns
    -------
    List[Chip]
        One chip per image in the stack.
    """
    region = PolygonRegion(vertices=polygon)
    row_start, row_end, col_start, col_end = polygon_bounding_box(polygon)

    chips = []
    for i, reader in enumerate(readers):
        try:
            # Transform polygon to this image's native space if registered
            img_polygon = polygon
            if registration_results and i < len(registration_results):
                result = registration_results[i]
                if result is not None and hasattr(result, 'transform_matrix'):
                    H = result.transform_matrix
                    try:
                        H_inv = np.linalg.inv(
                            H if H.shape == (3, 3) else np.vstack([H, [0, 0, 1]])
                        )
                        # Transform (row, col) vertices through inverse homography
                        ones = np.ones((polygon.shape[0], 1))
                        # polygon is (N, 2) as (row, col) → homogeneous (col, row, 1)
                        pts_h = np.hstack([
                            polygon[:, 1:2], polygon[:, 0:1], ones
                        ])  # (N, 3) as (x, y, 1)
                        warped = (H_inv @ pts_h.T).T
                        warped = warped[:, :2] / warped[:, 2:3]
                        img_polygon = np.column_stack([warped[:, 1], warped[:, 0]])
                    except np.linalg.LinAlgError:
                        pass  # Use original polygon if inversion fails

            local_rs, local_re, local_cs, local_ce = polygon_bounding_box(img_polygon)

            shape = reader.get_shape()
            # Clamp to image bounds
            rs = max(0, local_rs)
            re = min(shape[0], local_re)
            cs = max(0, local_cs)
            ce = min(shape[1], local_ce)

            if re <= rs or ce <= cs:
                continue

            chip_data = reader.read_chip(rs, re, cs, ce)

            chip = Chip(
                image_data=chip_data,
                source_image_index=i,
                source_image_name=names[i] if i < len(names) else f"Image {i}",
                polygon_region=region,
                label=ChipLabel.UNKNOWN,
                timestamp=timestamps[i] if timestamps and i < len(timestamps) else None,
            )
            chips.append(chip)
        except Exception:
            # Skip images where chipping fails (e.g., out of bounds)
            continue

    return chips


def chip_stack_at_polygons(
    readers: list,
    names: List[str],
    polygons: List[np.ndarray],
    registration_results: Optional[list] = None,
    timestamps: Optional[List[str]] = None,
) -> ChipSet:
    """Extract chips from all images at multiple polygon locations.

    Parameters
    ----------
    readers : list
        GRDL ImageReader instances.
    names : List[str]
        Display names.
    polygons : List[np.ndarray]
        Polygon vertex arrays.
    registration_results : Optional[list]
    timestamps : Optional[List[str]]

    Returns
    -------
    ChipSet
    """
    all_chips: List[Chip] = []
    regions: List[PolygonRegion] = []

    for poly in polygons:
        region = PolygonRegion(vertices=poly)
        regions.append(region)

        chips = chip_stack_at_polygon(
            readers, names, poly, registration_results, timestamps
        )
        # Override the region to share the same instance
        for chip in chips:
            chip.polygon_region = region
        all_chips.extend(chips)

    return ChipSet(chips=all_chips, polygon_regions=regions)
