# -*- coding: utf-8 -*-
"""
Workflow Executor - Headless and interactive workflow execution.

Executes a compiled WorkflowDefinition by instantiating GRDL processors
and running them in sequence over input imagery. Supports both single-image
and batch execution with optional GPU acceleration.

Author
------
Duane Smalley, PhD
duane.d.smalley@gmail.com

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
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

# Third-party
import numpy as np

# GRDK internal
from grdk.core.gpu import GpuBackend
from grdk.core.workflow import ProcessingStep, WorkflowDefinition


def _resolve_processor_class(processor_name: str) -> Type:
    """Resolve a processor class name to the actual class.

    Supports both short names (e.g., "PauliDecomposition") resolved
    by scanning grdl.image_processing, and fully-qualified names
    (e.g., "grdl.image_processing.decomposition.pauli.PauliDecomposition").

    Parameters
    ----------
    processor_name : str
        Short or fully-qualified processor class name.

    Returns
    -------
    Type
        The processor class.

    Raises
    ------
    ImportError
        If the processor class cannot be found.
    """
    # Try fully-qualified import first
    if '.' in processor_name:
        module_path, class_name = processor_name.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    # Try scanning grdl.image_processing
    try:
        ip_module = importlib.import_module('grdl.image_processing')
        if hasattr(ip_module, processor_name):
            return getattr(ip_module, processor_name)
    except ImportError:
        pass

    raise ImportError(
        f"Cannot resolve processor class '{processor_name}'. "
        f"Use a fully-qualified name (e.g., 'grdl.image_processing.ortho.ortho.Orthorectifier')."
    )


class WorkflowExecutor:
    """Execute a compiled workflow headlessly or interactively.

    Instantiates each processor in the workflow and runs them in
    sequence on input imagery.

    Parameters
    ----------
    workflow : WorkflowDefinition
        Compiled workflow to execute.
    gpu : Optional[GpuBackend]
        GPU backend for acceleration. If None, CPU only.
    """

    def __init__(
        self,
        workflow: WorkflowDefinition,
        gpu: Optional[GpuBackend] = None,
    ) -> None:
        self._workflow = workflow
        self._gpu = gpu or GpuBackend(prefer_gpu=False)

    def execute(
        self,
        source: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """Run the full pipeline on a single image.

        Parameters
        ----------
        source : np.ndarray
            Input image array.
        **kwargs
            Additional arguments passed to each processor.

        Returns
        -------
        np.ndarray
            Result after all processing steps.
        """
        current = source
        for step in self._workflow.steps:
            current = self._execute_step(step, current, **kwargs)
        return current

    def execute_batch(
        self,
        sources: List[np.ndarray],
        **kwargs: Any,
    ) -> List[np.ndarray]:
        """Run the pipeline on multiple images.

        Parameters
        ----------
        sources : List[np.ndarray]
            List of input image arrays.
        **kwargs
            Additional arguments passed to each processor.

        Returns
        -------
        List[np.ndarray]
            List of results.
        """
        return [self.execute(src, **kwargs) for src in sources]

    def execute_step(
        self,
        step_index: int,
        source: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """Execute a single step from the workflow.

        Useful for interactive preview of individual steps.

        Parameters
        ----------
        step_index : int
            Index of the step to execute.
        source : np.ndarray
            Input image array.
        **kwargs
            Additional arguments.

        Returns
        -------
        np.ndarray
            Result of this step.
        """
        step = self._workflow.steps[step_index]
        return self._execute_step(step, source, **kwargs)

    def _execute_step(
        self,
        step: ProcessingStep,
        source: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """Execute a single processing step.

        Parameters
        ----------
        step : ProcessingStep
        source : np.ndarray
        **kwargs

        Returns
        -------
        np.ndarray
        """
        processor_cls = _resolve_processor_class(step.processor_name)
        processor = processor_cls()

        # Merge step params with kwargs (step params take precedence)
        merged_kwargs = {**kwargs, **step.params}

        return self._gpu.apply_transform(processor, source, **merged_kwargs)
