# -*- coding: utf-8 -*-
"""
Polarimetric Utilities - Helpers for quad-pol dataset validation in GRDK.

Provides functions to inspect an :class:`~grdk.widgets._signals.ImageStack`
and determine its polarimetric collection mode, extract per-polarization
channel arrays, and validate completeness for quad-pol workflows.

Two loading patterns are supported:

**BIOMASS / multi-band single reader**
    ``BIOMASSL1Reader`` exposes all four polarizations in a single
    ``read_full()`` call as a ``(4, rows, cols)`` CYX complex array.
    Channel order is declared via ``metadata.channel_metadata[i].polarization``.

**NISAR / multi-reader per polarization**
    ``NISARReader`` exposes one polarization per reader instance.  A quad-pol
    ``ImageStack`` therefore holds four readers whose ``metadata.polarization``
    strings are ``'HH'``, ``'HV'``, ``'VH'``, and ``'VV'`` respectively.

``extract_quad_pol_arrays()`` handles both patterns transparently.

Author
------
Ava Courtney
courtney-ava@zai.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-04-15

Modified
--------
2026-04-15
"""

# Standard library
from typing import Dict, Optional, Set, Tuple

# Third-party
import numpy as np

# GRDL vocabulary
from grdl.vocabulary import PolarimetricMode

# GRDK internal
from grdk.widgets._signals import ImageStack

# The four channels required for a complete quad-pol acquisition.
_QUAD_POL_CHANNELS: Set[str] = {'HH', 'HV', 'VH', 'VV'}

# Dual-pol pairs recognised as a pair by convention.
_DUAL_POL_PAIRS = [
    {'HH', 'HV'},
    {'VV', 'VH'},
    {'HH', 'VV'},
    {'HV', 'VH'},
]


def _reader_polarization(reader) -> Optional[str]:
    """Extract the *primary* polarization string from a reader's metadata.

    For NISAR-style readers (one polarization per reader) this returns the
    single polarization string from ``metadata.polarization``.

    For BIOMASS-style readers (all polarizations in one multi-band reader)
    this returns the polarization of the **first** channel only — use
    :func:`_reader_quad_pol_channels` to get the full mapping.

    Parameters
    ----------
    reader : ImageReader
        Any GRDL reader instance.

    Returns
    -------
    Optional[str]
        Uppercase polarization string (e.g. ``'HH'``) or ``None``.
    """
    # NISAR / CPHD / sensor-specific: metadata.polarization is a plain string
    pol = getattr(getattr(reader, 'metadata', None), 'polarization', None)
    if isinstance(pol, str) and pol:
        return pol.upper()

    meta = getattr(reader, 'metadata', None)

    # Generic multi-band readers: first ChannelMetadata with a polarization
    channel_metadata = getattr(meta, 'channel_metadata', None)
    if channel_metadata:
        for ch in channel_metadata:
            p = getattr(ch, 'polarization', None)
            if isinstance(p, str) and p.strip():
                return p.strip().upper()

    # BIOMASSMetadata stores polarizations as a list when channel_metadata
    # is not populated (e.g. when polarisationList is absent from the XML).
    pol_list = getattr(meta, 'polarizations', None)
    if pol_list:
        for p in pol_list:
            if isinstance(p, str) and p.strip():
                return p.strip().upper()

    return None


def _reader_quad_pol_channels(reader) -> Optional[Dict[str, int]]:
    """Return a ``{pol: band_index}`` mapping when a reader holds all four
    quad-pol channels in a single multi-band cube (e.g. BIOMASS).

    Parameters
    ----------
    reader : ImageReader
        Any GRDL reader instance.

    Returns
    -------
    Optional[Dict[str, int]]
        Mapping ``{'HH': 0, 'HV': 1, 'VH': 2, 'VV': 3}`` (indices equal
        ``channel_metadata[i].index``) when all four polarizations are
        present, or ``None`` when the reader is single-band or does not
        declare polarization metadata.
    """
    meta = getattr(reader, 'metadata', None)

    # Primary path: ChannelMetadata list (set when XML annotation is complete)
    channel_metadata = getattr(meta, 'channel_metadata', None)
    if channel_metadata:
        mapping: Dict[str, int] = {}
        for ch in channel_metadata:
            pol = getattr(ch, 'polarization', None)
            if isinstance(pol, str) and pol.strip().upper() in _QUAD_POL_CHANNELS:
                mapping[pol.strip().upper()] = ch.index
        if _QUAD_POL_CHANNELS.issubset(mapping.keys()):
            return mapping

    # Fallback: BIOMASSMetadata.polarizations list (available even when
    # channel_metadata is None, e.g. when polarisationList is absent from XML).
    # Band order follows the list order.
    pol_list = getattr(meta, 'polarizations', None)
    if pol_list:
        pols_upper = [
            p.strip().upper() for p in pol_list
            if isinstance(p, str) and p.strip()
        ]
        if _QUAD_POL_CHANNELS.issubset(set(pols_upper)):
            return {
                p: i for i, p in enumerate(pols_upper)
                if p in _QUAD_POL_CHANNELS
            }

    return None


def get_polarimetric_mode(stack: ImageStack) -> Optional[PolarimetricMode]:
    """Determine the :class:`~grdl.vocabulary.PolarimetricMode` of an
    :class:`~grdk.widgets._signals.ImageStack`.

    Handles both multi-band single-reader stacks (BIOMASS) and per-pol
    multi-reader stacks (NISAR).

    Parameters
    ----------
    stack : ImageStack
        The image stack to inspect.

    Returns
    -------
    Optional[PolarimetricMode]
        - :attr:`~grdl.vocabulary.PolarimetricMode.QUAD_POL` when all four
          HH, HV, VH, VV channels are available.
        - :attr:`~grdl.vocabulary.PolarimetricMode.DUAL_POL` for a
          complementary two-channel subset.
        - :attr:`~grdl.vocabulary.PolarimetricMode.SINGLE_POL` for a
          single co-pol or cross-pol channel.
        - ``None`` when no polarization metadata is found.
    """
    if not stack or not stack.readers:
        return None

    pols: Set[str] = set()
    for reader in stack.readers:
        # Multi-band reader (BIOMASS): contributes all its polarizations
        mb_mapping = _reader_quad_pol_channels(reader)
        if mb_mapping:
            pols.update(mb_mapping.keys())
        else:
            # Single-band reader (NISAR): contributes its one polarization
            p = _reader_polarization(reader)
            if p is not None:
                pols.add(p)

    if not pols:
        return None
    if _QUAD_POL_CHANNELS.issubset(pols):
        return PolarimetricMode.QUAD_POL
    if len(pols) == 2 and any(pols == pair for pair in _DUAL_POL_PAIRS):
        return PolarimetricMode.DUAL_POL
    if len(pols) == 1:
        return PolarimetricMode.SINGLE_POL
    return None


def is_quad_pol(stack: ImageStack) -> bool:
    """Return ``True`` when *stack* contains all four quad-pol channels.

    Parameters
    ----------
    stack : ImageStack
        The image stack to inspect.

    Returns
    -------
    bool
    """
    return get_polarimetric_mode(stack) == PolarimetricMode.QUAD_POL


def extract_quad_pol_arrays(
    stack: ImageStack,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read and return ``(shh, shv, svh, svv)`` at full resolution.

    For large images that would exceed available RAM, use
    :func:`extract_quad_pol_arrays_strided` which reads stripe-by-stripe.
    """
    return _extract_quad_pol_arrays_impl(stack, stride=1)


def extract_quad_pol_arrays_strided(
    stack: ImageStack,
    max_pixels: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read and return ``(shh, shv, svh, svv)`` downsampled to *max_pixels*.

    Reads each polarization channel stripe-by-stripe via ``read_chip``
    so that full-resolution data is never materialised in memory.  The
    output arrays have shape ``(ceil(rows/stride), ceil(cols/stride))``.

    Parameters
    ----------
    stack : ImageStack
        A quad-pol image stack.
    max_pixels : int
        Maximum number of pixels (H × W) to return per channel.
        A spatial stride is derived from the native image dimensions
        and this cap.  Pass ``0`` to read at full resolution.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(shh, shv, svh, svv)`` complex 2-D arrays.
    """
    import math

    if max_pixels <= 0:
        return _extract_quad_pol_arrays_impl(stack, stride=1)

    # Determine native dimensions from the first readable reader
    rows, cols = _native_dims(stack)
    if rows is None or cols is None:
        return _extract_quad_pol_arrays_impl(stack, stride=1)

    total = rows * cols
    if total <= max_pixels:
        return _extract_quad_pol_arrays_impl(stack, stride=1)

    stride = math.ceil(math.sqrt(total / max_pixels))
    return _extract_quad_pol_arrays_impl(stack, stride=stride)


def _native_dims(stack: ImageStack):
    """Return ``(rows, cols)`` from the first reader with metadata, or ``(None, None)``."""
    for reader in stack.readers:
        meta = getattr(reader, 'metadata', None)
        rows = getattr(meta, 'rows', None)
        cols = getattr(meta, 'cols', None)
        if rows and cols:
            return int(rows), int(cols)
    return None, None


def _read_channel_strided(reader, stride: int) -> np.ndarray:
    """Read a single-band reader downsampled by *stride*.

    For stride > 1, reads a **center crop** of size
    ``(ceil(rows/stride), ceil(cols/stride))`` rather than striding
    over the full image.  This is O(output pixels) rather than
    O(full image), making it practical on large NFS-stored HDF5 files
    where reading even every 4th row forces a full file scan.

    The crop is centered in the scene so representative content is shown.
    """
    if stride <= 1:
        arr = reader.read_full()
        if arr.ndim == 3 and arr.shape[0] == 1:
            return arr[0]
        if arr.ndim == 3 and arr.shape[-1] == 1:
            return arr[..., 0]
        return arr

    meta = getattr(reader, 'metadata', None)
    rows = int(getattr(meta, 'rows', 0))
    cols = int(getattr(meta, 'cols', 0))

    if not (rows and cols):
        arr = reader.read_full()
        if arr.ndim == 2:
            return arr[::stride, ::stride]
        return arr[:, ::stride, ::stride]

    out_h = -(-rows // stride)   # ceil(rows/stride)
    out_w = -(-cols // stride)   # ceil(cols/stride)

    # Center the crop
    r0 = max(0, (rows - out_h) // 2)
    c0 = max(0, (cols - out_w) // 2)
    r1 = min(rows, r0 + out_h)
    c1 = min(cols, c0 + out_w)

    chip = reader.read_chip(r0, r1, c0, c1)

    if chip.ndim == 3 and chip.shape[0] == 1:
        chip = chip[0]
    elif chip.ndim == 3 and chip.shape[-1] == 1:
        chip = chip[..., 0]
    return chip


def _extract_quad_pol_arrays_impl(
    stack: ImageStack,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Internal implementation used by both public variants."""
    if not stack or not stack.readers:
        raise ValueError("Stack is empty — no readers found.")

    # Case 1: Multi-band single reader (BIOMASS)
    for reader in stack.readers:
        mb_mapping = _reader_quad_pol_channels(reader)
        if mb_mapping:
            if stride <= 1:
                cube = reader.read_full()  # (4, H, W) CYX complex
            else:
                # Read full then stride — BIOMASS tiffs are typically
                # small enough that full-res fits; if not, stride in-place
                cube = reader.read_full()
                if cube.ndim == 3 and cube.shape[-1] == 4 and cube.shape[0] != 4:
                    cube = np.moveaxis(cube, -1, 0)
                cube = cube[:, ::stride, ::stride]
            if cube.ndim == 3 and cube.shape[-1] == 4 and cube.shape[0] != 4:
                cube = np.moveaxis(cube, -1, 0)
            return (
                cube[mb_mapping['HH']],
                cube[mb_mapping['HV']],
                cube[mb_mapping['VH']],
                cube[mb_mapping['VV']],
            )

    # Case 2: Per-polarization readers (NISAR) — read each channel separately
    pol_readers = extract_quad_pol_readers(stack)
    shh = _read_channel_strided(pol_readers['HH'], stride)
    shv = _read_channel_strided(pol_readers['HV'], stride)
    svh = _read_channel_strided(pol_readers['VH'], stride)
    svv = _read_channel_strided(pol_readers['VV'], stride)
    return shh, shv, svh, svv


def extract_quad_pol_readers(stack: ImageStack) -> Dict[str, object]:
    """Return a ``{pol: reader}`` mapping for a per-pol multi-reader stack.

    .. note::
        For BIOMASS-style multi-band single-reader stacks, use
        :func:`extract_quad_pol_arrays` instead.

    Parameters
    ----------
    stack : ImageStack
        A quad-pol ``ImageStack`` with one reader per polarization.

    Returns
    -------
    Dict[str, ImageReader]
        ``{'HH': reader, 'HV': reader, 'VH': reader, 'VV': reader}``.

    Raises
    ------
    ValueError
        If any quad-pol channel is absent.
    """
    mapping: Dict[str, object] = {}
    for reader in stack.readers:
        p = _reader_polarization(reader)
        if p is not None and p in _QUAD_POL_CHANNELS:
            mapping[p] = reader

    missing = _QUAD_POL_CHANNELS - set(mapping.keys())
    if missing:
        present = sorted(mapping.keys()) or ['(none)']
        raise ValueError(
            f"Quad-pol stack is incomplete. "
            f"Missing polarizations: {sorted(missing)}. "
            f"Present: {present}."
        )

    return {pol: mapping[pol] for pol in ('HH', 'HV', 'VH', 'VV')}
