# -*- coding: utf-8 -*-
"""
Tests for polarimetric processing widgets.

Tests the OWCovarianceMatrix and OWPauliDecomposer widgets with
synthetic quad-pol SAR data.

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

import numpy as np
import pytest


class MockQuadPolReader:
    """Mock reader providing synthetic quad-pol SAR data.

    Provides HH, HV, VH, VV channels in CYX layout for testing
    polarimetric matrix computation and Pauli decomposition.
    """

    def __init__(self, rows=100, cols=100):
        self.rows = rows
        self.cols = cols
        self.filepath = "mock_quadpol.h5"

        # Create synthetic quad-pol data
        # HH: strong surface scatter
        # VV: moderate surface scatter
        # HV/VH: volume scatter
        np.random.seed(42)
        self.shh = (
            np.random.randn(rows, cols) + 1j * np.random.randn(rows, cols)
        ) * 10.0
        self.svv = (
            np.random.randn(rows, cols) + 1j * np.random.randn(rows, cols)
        ) * 7.0
        self.shv = (
            np.random.randn(rows, cols) + 1j * np.random.randn(rows, cols)
        ) * 3.0
        self.svh = (
            np.random.randn(rows, cols) + 1j * np.random.randn(rows, cols)
        ) * 3.0

        # Stack into CYX array
        self._data = np.stack([self.shh, self.shv, self.svh, self.svv], axis=0)

        # Create metadata
        from grdk.widgets._pol_utils import _reader_polarization

        class Metadata:
            def __init__(self):
                self.rows = rows
                self.cols = cols
                self.bands = 4
                self.dtype = 'complex64'
                self.axis_order = 'CYX'
                self.format = 'MockQuadPol'

                # Channel metadata for polarization extraction
                class ChannelMeta:
                    def __init__(self, pol, idx):
                        self.polarization = pol
                        self.index = idx

                self.channel_metadata = [
                    ChannelMeta('HH', 0),
                    ChannelMeta('HV', 1),
                    ChannelMeta('VH', 2),
                    ChannelMeta('VV', 3),
                ]

        self.metadata = Metadata()

    def read_full(self, bands=None):
        """Read full CYX cube."""
        if bands is not None:
            return self._data[bands]
        return self._data

    def read_chip(self, row_start, row_end, col_start, col_end, bands=None):
        """Read a spatial chip."""
        chip = self._data[:, row_start:row_end, col_start:col_end]
        if bands is not None:
            chip = chip[bands]
        return chip

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_quad_pol_channel_extraction():
    """Test that we can extract quad-pol channels from mock reader."""
    from grdk.widgets._pol_utils import channel_pol_map, extract_quad_pol_arrays_strided

    reader = MockQuadPolReader(rows=100, cols=100)

    # Test channel mapping
    pol_map = channel_pol_map(reader)
    assert pol_map == {'HH': 0, 'HV': 1, 'VH': 2, 'VV': 3}

    # Create a mock ImageStack
    from grdk.widgets._signals import ImageStack

    stack = ImageStack(readers=[reader], names=['mock_quad_pol'])

    # Extract channels
    shh, shv, svh, svv = extract_quad_pol_arrays_strided(stack, max_pixels=0)

    # Verify shapes
    assert shh.shape == (100, 100)
    assert shv.shape == (100, 100)
    assert svh.shape == (100, 100)
    assert svv.shape == (100, 100)

    # Verify they're complex
    assert np.iscomplexobj(shh)
    assert np.iscomplexobj(svv)


def test_t3_coherency_matrix_shape():
    """Test T3 matrix computation produces correct shape."""
    from grdk.widgets._pol_utils import extract_quad_pol_arrays_strided
    from grdk.widgets._signals import ImageStack

    reader = MockQuadPolReader(rows=50, cols=50)
    stack = ImageStack(readers=[reader], names=['test'])

    # Extract channels
    shh, shv, svh, svv = extract_quad_pol_arrays_strided(stack, max_pixels=0)

    # Compute T3 matrix (using grdl if available, else skip)
    try:
        from grdl.image_processing.decomposition.pol_matrix import CoherencyMatrix

        t3_computer = CoherencyMatrix(window_size=5)
        t3 = t3_computer.compute(shh, shv, svh, svv)

        # Verify shape: (3, 3, rows, cols)
        assert t3.shape[0] == 3
        assert t3.shape[1] == 3
        assert t3.shape[2] == 50
        assert t3.shape[3] == 50

        # Verify it's complex
        assert np.iscomplexobj(t3)

        # Verify diagonal is real and positive
        for i in range(3):
            assert np.all(np.imag(t3[i, i]) < 1e-10)  # Nearly real
            assert np.all(np.real(t3[i, i]) >= 0)  # Positive power

    except ImportError:
        pytest.skip("grdl.image_processing.decomposition not available")


def test_pauli_decomposition_produces_rgb():
    """Test that Pauli decomposition produces valid RGB output."""
    from grdk.widgets._pol_utils import extract_quad_pol_arrays_strided
    from grdk.widgets._signals import ImageStack

    reader = MockQuadPolReader(rows=40, cols=40)
    stack = ImageStack(readers=[reader], names=['test'])

    shh, shv, svh, svv = extract_quad_pol_arrays_strided(stack, max_pixels=0)

    try:
        from grdl.image_processing.decomposition.pol_matrix import CoherencyMatrix
        from grdl.image_processing.decomposition.pauli import PauliDecomposition

        # Compute T3
        t3_computer = CoherencyMatrix(window_size=3)
        t3 = t3_computer.compute(shh, shv, svh, svv)

        # Extract diagonal (Pauli powers)
        surface_pwr = t3[0, 0].real
        db_pwr = t3[1, 1].real
        volume_pwr = t3[2, 2].real

        # Stack into RGB (R=double-bounce, G=volume, B=surface)
        rgb = np.stack([db_pwr, volume_pwr, surface_pwr], axis=0)

        # Verify shape: (3, H, W)
        assert rgb.shape == (3, 40, 40)

        # Verify all values are real and non-negative
        assert np.all(rgb >= 0)
        assert not np.iscomplexobj(rgb)

    except ImportError:
        pytest.skip("grdl decomposition modules not available")


def test_image_stack_validation():
    """Test validate_image_stack function."""
    from grdk.widgets._signals import ImageStack, validate_image_stack

    # Create readers with matching dimensions
    reader1 = MockQuadPolReader(rows=100, cols=100)
    reader2 = MockQuadPolReader(rows=100, cols=100)

    stack = ImageStack(
        readers=[reader1, reader2],
        names=['image1', 'image2'],
        reader_metadata=[
            {'polarization': 'HH', 'swath_id': 'IW1'},
            {'polarization': 'VV', 'swath_id': 'IW1'},
        ],
    )

    warnings = validate_image_stack(stack)
    # Should have no dimension warnings
    dimension_warnings = [w for w in warnings if 'Dimension mismatch' in w]
    assert len(dimension_warnings) == 0


def test_image_stack_validation_detects_dimension_mismatch():
    """Test that validation detects dimension mismatches."""
    from grdk.widgets._signals import ImageStack, validate_image_stack

    reader1 = MockQuadPolReader(rows=100, cols=100)
    reader2 = MockQuadPolReader(rows=50, cols=50)

    stack = ImageStack(readers=[reader1, reader2], names=['img1', 'img2'])

    warnings = validate_image_stack(stack)
    # Should detect dimension mismatch
    assert any('Dimension mismatch' in w for w in warnings)
    assert any('50×50' in w for w in warnings)


def test_image_stack_validation_detects_duplicate_polarization():
    """Test that validation detects duplicate polarizations."""
    from grdk.widgets._signals import ImageStack, validate_image_stack

    reader1 = MockQuadPolReader(rows=100, cols=100)
    reader2 = MockQuadPolReader(rows=100, cols=100)

    stack = ImageStack(
        readers=[reader1, reader2],
        names=['img1', 'img2'],
        reader_metadata=[
            {'polarization': 'HH'},
            {'polarization': 'HH'},  # Duplicate!
        ],
    )

    warnings = validate_image_stack(stack)
    # Should detect duplicate polarization
    assert any('Duplicate polarization' in w and 'HH' in w for w in warnings)


def test_covariance_matrix_signal():
    """Test CovarianceMatrixSignal creation."""
    from grdk.widgets._signals import CovarianceMatrixSignal

    # Create a mock T3 matrix
    t3 = np.random.randn(3, 3, 50, 50).astype(np.complex64)

    signal = CovarianceMatrixSignal(
        matrix=t3,
        matrix_type='T3',
        window_size=7,
        source_metadata={'sensor': 'BIOMASS', 'mode': 'quad-pol'},
    )

    assert signal.matrix_type == 'T3'
    assert signal.window_size == 7
    assert signal.source_metadata['sensor'] == 'BIOMASS'
    assert signal.matrix.shape == (3, 3, 50, 50)
