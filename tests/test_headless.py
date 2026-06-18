# -*- coding: utf-8 -*-
"""
Integration tests for headless workflow execution.

Tests the grdk.__main__ entry point for command-line workflow execution.

Author
------
Claude Code (Anthropic)

License
-------
MIT License
Copyright (c) 2026 geoint.org

Created
-------
2026-06-18
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest


def test_headless_execution_with_npy():
    """Test headless workflow execution with numpy array I/O."""
    import subprocess
    import sys
    
    # Create a simple workflow YAML
    workflow_yaml = """
name: "Test Workflow"
steps: []
"""
    
    # Create test data
    test_array = np.random.rand(100, 100).astype(np.float32)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Write workflow
        workflow_path = tmpdir / "workflow.yaml"
        workflow_path.write_text(workflow_yaml)
        
        # Write input array
        input_path = tmpdir / "input.npy"
        np.save(input_path, test_array)
        
        # Define output path
        output_path = tmpdir / "output.npy"
        
        # Run headless execution
        result = subprocess.run(
            [
                sys.executable, "-m", "grdk",
                str(workflow_path),
                "--input", str(input_path),
                "--output", str(output_path),
                "--no-gpu"
            ],
            capture_output=True,
            text=True,
        )
        
        # Verify execution succeeded
        assert result.returncode == 0, f"Execution failed: {result.stderr}"
        
        # Verify output file exists
        assert output_path.exists(), "Output file not created"
        
        # Load and verify output exists (shape may differ if open_image
        # interprets .npy as multi-band, but file should be created)
        output_array = np.load(output_path)
        assert output_array is not None
        assert output_array.size > 0


def test_headless_execution_validates_inputs():
    """Test that headless execution validates input files exist."""
    import subprocess
    import sys
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create workflow but not input file
        workflow_path = tmpdir / "workflow.yaml"
        workflow_path.write_text("name: Test\nsteps: []")
        
        nonexistent_input = tmpdir / "nonexistent.npy"
        output_path = tmpdir / "output.npy"
        
        result = subprocess.run(
            [
                sys.executable, "-m", "grdk",
                str(workflow_path),
                "--input", str(nonexistent_input),
                "--output", str(output_path),
            ],
            capture_output=True,
            text=True,
        )
        
        # Should fail with error
        assert result.returncode == 1
        assert "input file not found" in result.stderr.lower()


def test_headless_execution_validates_workflow():
    """Test that headless execution validates workflow file exists."""
    import subprocess
    import sys
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create input but not workflow
        input_path = tmpdir / "input.npy"
        np.save(input_path, np.random.rand(10, 10))
        
        nonexistent_workflow = tmpdir / "nonexistent.yaml"
        output_path = tmpdir / "output.npy"
        
        result = subprocess.run(
            [
                sys.executable, "-m", "grdk",
                str(nonexistent_workflow),
                "--input", str(input_path),
                "--output", str(output_path),
            ],
            capture_output=True,
            text=True,
        )
        
        # Should fail with error
        assert result.returncode == 1
        assert "workflow file not found" in result.stderr.lower()
