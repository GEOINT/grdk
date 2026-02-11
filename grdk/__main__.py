# -*- coding: utf-8 -*-
"""
GRDK CLI - Headless workflow execution.

Usage::

    python -m grdk workflow.yaml --input image.tif --output result.tif
    python -m grdk workflow.yaml --input image.tif --output result.tif --no-gpu

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

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="grdk",
        description="GRDK — Execute image processing workflows headlessly.",
    )
    parser.add_argument(
        "workflow",
        type=Path,
        help="Path to a YAML workflow definition file.",
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        dest="input_path",
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        dest="output_path",
        help="Path to write the output image.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=True,
        help="Enable GPU acceleration (default).",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration, use CPU only.",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.workflow.exists():
        print(f"Error: workflow file not found: {args.workflow}", file=sys.stderr)
        return 1
    if not args.input_path.exists():
        print(f"Error: input file not found: {args.input_path}", file=sys.stderr)
        return 1

    # Compile workflow
    from grdl_rt.execution.dsl import DslCompiler
    compiler = DslCompiler()
    workflow_def = compiler.compile_yaml(args.workflow)

    # Set up GPU backend
    from grdl_rt.execution.gpu import GpuBackend
    prefer_gpu = args.gpu and not args.no_gpu
    gpu = GpuBackend(prefer_gpu=prefer_gpu)

    # Load input image
    import numpy as np
    try:
        from grdl.IO.base import ImageReader
        # Try GRDL readers first
        from grdl.IO import open_image
        reader = open_image(str(args.input_path))
        source = reader.read_full()
    except (ImportError, Exception):
        # Fall back to numpy load for .npy files
        if args.input_path.suffix == '.npy':
            source = np.load(str(args.input_path))
        else:
            print(
                f"Error: cannot read input file: {args.input_path}",
                file=sys.stderr,
            )
            return 1

    # Execute workflow
    from grdl_rt.execution.executor import WorkflowExecutor
    executor = WorkflowExecutor(workflow=workflow_def, gpu=gpu)
    wr = executor.execute(source)

    # Write output
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.output_path.suffix == '.npy':
        np.save(str(args.output_path), wr.result)
    else:
        try:
            from grdl.IO import open_writer
            writer = open_writer(str(args.output_path))
            writer.write(wr.result)
        except (ImportError, Exception):
            np.save(str(args.output_path), wr.result)

    print(f"Output written to: {args.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
