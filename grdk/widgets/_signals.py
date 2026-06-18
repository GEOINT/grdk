# -*- coding: utf-8 -*-
"""
Custom Orange Signal Types - Typed data channels for GRDK widgets.

Defines the signal types used to pass data between Orange widgets in
the GEODEV and Admin workflows. These replace Orange's default Table
signals with GEOINT-specific data structures.

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
from typing import Any, Dict, List, Optional

# Third-party
import numpy as np


class ImageStack:
    """Signal type: an ordered collection of co-registered images.

    Represents a stack of images that share a common pixel coordinate
    space (after co-registration). Each image is accessible via its
    GRDL ImageReader.

    Parameters
    ----------
    readers : list
        Ordered list of GRDL ImageReader instances.
    names : List[str]
        Display names for each image in the stack.
    geolocation : optional
        GRDL Geolocation object for the reference image.
    registration_results : optional
        List of RegistrationResult objects (one per non-reference image).
    metadata : Dict[str, Any]
        Stack-level metadata (e.g., sensor type, acquisition dates).
    reader_metadata : List[Dict[str, Any]], optional
        Per-reader metadata list (one dict per reader). Each dict should
        contain keys like 'polarization', 'swath_id', 'acquisition_time',
        'sensor' for provenance tracking. If None, an empty list is created.
    """

    def __init__(
        self,
        readers: Optional[list] = None,
        names: Optional[List[str]] = None,
        geolocation: Optional[Any] = None,
        registration_results: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
        reader_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.readers = readers or []
        self.names = names or []
        self.geolocation = geolocation
        self.registration_results = registration_results or []
        self.metadata = metadata or {}
        self.reader_metadata = reader_metadata or []

    def __len__(self) -> int:
        return len(self.readers)


class ChipSetSignal:
    """Signal type: a collection of chips with labels and polygon metadata.

    Wraps the core ChipSet for use as an Orange signal type.

    Parameters
    ----------
    chip_set : optional
        A grdl_rt.execution.chip.ChipSet instance.
    """

    def __init__(self, chip_set: Optional[Any] = None) -> None:
        self.chip_set = chip_set


class ProcessingPipelineSignal:
    """Signal type: an ordered list of processing steps.

    Wraps the core WorkflowDefinition for use as an Orange signal type.

    Parameters
    ----------
    workflow : optional
        A grdl_rt.execution.workflow.WorkflowDefinition instance.
    """

    def __init__(self, workflow: Optional[Any] = None) -> None:
        self.workflow = workflow


class WorkflowArtifactSignal:
    """Signal type: a published workflow definition.

    Contains both the Python DSL and YAML representations of a
    published workflow, along with its metadata and tags.

    Parameters
    ----------
    python_dsl : str
        Generated Python DSL source code.
    yaml_definition : str
        Generated YAML workflow definition.
    metadata : Dict[str, Any]
        Workflow metadata including tags.
    """

    def __init__(
        self,
        python_dsl: str = "",
        yaml_definition: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.python_dsl = python_dsl
        self.yaml_definition = yaml_definition
        self.metadata = metadata or {}


class GrdkProjectSignal:
    """Signal type: a full GRDK project reference.

    Wraps the core GrdkProject for use as an Orange signal type.

    Parameters
    ----------
    project : optional
        A grdl_rt.execution.project.GrdkProject instance.
    """

    def __init__(self, project: Optional[Any] = None) -> None:
        self.project = project


class CovarianceMatrixSignal:
    """Signal type: a spatially-averaged polarimetric matrix.

    Carries the T3 coherency matrix or C3 covariance matrix computed
    from quad-pol SAR data, along with the metadata required for
    downstream Pauli decomposition and spatial coordinate display.

    Parameters
    ----------
    matrix : np.ndarray
        Complex array of shape ``(N, N, rows, cols)`` (axis_order='CCYX').
        N is 3 for quad-pol, 2 for dual-pol.
    matrix_type : str
        ``'T3'`` for the Pauli coherency matrix or ``'C3'`` for the
        lexicographic covariance matrix.  Downstream decomposition nodes
        use this tag to validate compatibility.
    window_size : int
        Spatial averaging window used when computing the matrix.
    source_metadata : Dict[str, Any]
        Stack-level metadata inherited from the originating
        ``ImageStack`` (polarimetric mode, sensor, timestamps, etc.).
    geolocation : optional
        GRDL Geolocation object for spatial coordinate display in the
        Stack Viewer.
    """

    def __init__(
        self,
        matrix: Optional['np.ndarray'] = None,
        matrix_type: str = 'T3',
        window_size: int = 7,
        source_metadata: Optional[Dict[str, Any]] = None,
        geolocation: Optional[Any] = None,
    ) -> None:
        self.matrix = matrix
        self.matrix_type = matrix_type
        self.window_size = window_size
        self.source_metadata = source_metadata or {}
        self.geolocation = geolocation


# ---------------------------------------------------------------------------
# Validation Functions
# ---------------------------------------------------------------------------

def validate_image_stack(stack: ImageStack) -> List[str]:
    """Validate an ImageStack for spatial consistency and return warnings.

    Checks for common issues that can compromise processing quality:
    - Mismatched dimensions (rows/cols) across readers
    - Mismatched pixel spacing (resolution) beyond coregistration tolerance
    - Missing or inconsistent polarization metadata
    - Time gaps that may indicate incompatible acquisitions

    Parameters
    ----------
    stack : ImageStack
        The image stack to validate.

    Returns
    -------
    List[str]
        List of warning messages. Empty list indicates no issues found.

    Examples
    --------
    >>> warnings = validate_image_stack(my_stack)
    >>> if warnings:
    ...     for w in warnings:
    ...         print(f"Warning: {w}")
    """
    warnings = []

    if not stack.readers:
        warnings.append("Stack contains no readers")
        return warnings

    # Extract dimensions and pixel spacing from all readers
    dimensions = []
    pixel_spacings = []
    polarizations = []
    acquisition_times = []

    for i, reader in enumerate(stack.readers):
        meta = getattr(reader, 'metadata', None)
        if meta is None:
            warnings.append(f"Reader {i} has no metadata attribute")
            continue

        # Check dimensions
        rows = getattr(meta, 'rows', None)
        cols = getattr(meta, 'cols', None)
        if rows is not None and cols is not None:
            dimensions.append((rows, cols, i))
        else:
            warnings.append(f"Reader {i} missing rows/cols metadata")

        # Check pixel spacing
        pixel_spacing_x = getattr(meta, 'pixel_spacing_x', None)
        pixel_spacing_y = getattr(meta, 'pixel_spacing_y', None)
        if pixel_spacing_x is not None and pixel_spacing_y is not None:
            pixel_spacings.append((pixel_spacing_x, pixel_spacing_y, i))

        # Extract polarization from reader_metadata if available
        if i < len(stack.reader_metadata):
            reader_meta = stack.reader_metadata[i]
            pol = reader_meta.get('polarization')
            if pol:
                polarizations.append((pol, i))

            acq_time = reader_meta.get('acquisition_time')
            if acq_time:
                acquisition_times.append((acq_time, i))

    # Validate dimension consistency
    if len(dimensions) > 1:
        ref_dims = dimensions[0][:2]
        for rows, cols, idx in dimensions[1:]:
            if (rows, cols) != ref_dims:
                warnings.append(
                    f"Dimension mismatch: reader {idx} has {rows}×{cols}, "
                    f"but reader 0 has {ref_dims[0]}×{ref_dims[1]}"
                )

    # Validate pixel spacing consistency (within 1% tolerance)
    if len(pixel_spacings) > 1:
        ref_spacing = pixel_spacings[0][:2]
        for spacing_x, spacing_y, idx in pixel_spacings[1:]:
            x_diff = abs(spacing_x - ref_spacing[0]) / ref_spacing[0]
            y_diff = abs(spacing_y - ref_spacing[1]) / ref_spacing[1]
            if x_diff > 0.01 or y_diff > 0.01:
                warnings.append(
                    f"Pixel spacing mismatch: reader {idx} has "
                    f"({spacing_x:.3f}, {spacing_y:.3f}), but reader 0 has "
                    f"({ref_spacing[0]:.3f}, {ref_spacing[1]:.3f}) "
                    f"(>{1}% difference)"
                )

    # Validate polarization diversity (warn if duplicate polarizations)
    if polarizations:
        pol_set = set(pol for pol, _ in polarizations)
        if len(pol_set) < len(polarizations):
            # Find duplicates
            pol_counts = {}
            for pol, idx in polarizations:
                if pol not in pol_counts:
                    pol_counts[pol] = []
                pol_counts[pol].append(idx)
            for pol, indices in pol_counts.items():
                if len(indices) > 1:
                    warnings.append(
                        f"Duplicate polarization '{pol}' in readers {indices}"
                    )

    # Validate acquisition time consistency (warn if >1 minute apart)
    if len(acquisition_times) > 1:
        from datetime import timedelta
        times = [t for t, _ in acquisition_times]
        min_time = min(times)
        max_time = max(times)
        time_diff = max_time - min_time
        if time_diff > timedelta(minutes=1):
            warnings.append(
                f"Large acquisition time gap: {time_diff} between images "
                "(may indicate incompatible data)"
            )

    return warnings

