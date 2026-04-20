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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Third-party
import numpy as np


@dataclass
class StackMetadata:
    """Typed stack-level metadata for an ImageStack.

    Carries only inter-image, stack-level state that has no counterpart
    in a single ``ImageReader.metadata``.  Sensor-specific information
    (bands, polarization, CRS, transform) must be read from each
    reader's ``metadata`` attribute directly.

    Parameters
    ----------
    reference_image_index : int
        Index of the reference image in the stack (default 0).
    registration_quality : List[float], optional
        Per-image registration quality score (0-1).  One entry per
        non-reference image.  ``None`` when co-registration has not
        been run.
    acquisition_timestamps : List[str], optional
        ISO 8601 acquisition timestamps, one per image.
    extras : Dict[str, Any]
        Catch-all for any additional stack-level metadata that does
        not yet have a typed field.
    """

    reference_image_index: int = 0
    registration_quality: Optional[List[float]] = None
    acquisition_timestamps: Optional[List[str]] = None
    extras: Dict[str, Any] = field(default_factory=dict)


class ImageStack:
    """Signal type: an ordered collection of co-registered images.

    Represents a stack of images that share a common pixel coordinate
    space (after co-registration). Each image is accessible via its
    GRDL ImageReader.  Sensor-specific metadata (bands, polarization,
    CRS, transform) must be read from each ``reader.metadata``
    directly; only stack-level state lives here.

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
    stack_metadata : StackMetadata, optional
        Typed stack-level metadata.
    """

    def __init__(
        self,
        readers: Optional[list] = None,
        names: Optional[List[str]] = None,
        geolocation: Optional[Any] = None,
        registration_results: Optional[list] = None,
        stack_metadata: Optional[StackMetadata] = None,
    ) -> None:
        self.readers = readers or []
        self.names = names or []
        self.geolocation = geolocation
        self.registration_results = registration_results or []
        self.stack_metadata = stack_metadata if stack_metadata is not None else StackMetadata()

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
    output_port_type : str, optional
        The ``DataPortType`` value (e.g. ``'binary_mask'``, ``'raster'``)
        that the last step in *workflow* produces.  ``None`` means the
        type is unknown or unconstrained (implicit ANY).  Receiving
        widgets use this to validate compatibility and show warnings
        when the incoming pipeline type does not match what they expect.
    """

    def __init__(
        self,
        workflow: Optional[Any] = None,
        output_port_type: Optional[str] = None,
    ) -> None:
        self.workflow = workflow
        self.output_port_type = output_port_type


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
