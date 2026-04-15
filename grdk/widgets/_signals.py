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
    """

    def __init__(
        self,
        readers: Optional[list] = None,
        names: Optional[List[str]] = None,
        geolocation: Optional[Any] = None,
        registration_results: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.readers = readers or []
        self.names = names or []
        self.geolocation = geolocation
        self.registration_results = registration_results or []
        self.metadata = metadata or {}

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
