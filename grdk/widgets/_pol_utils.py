# -*- coding: utf-8 -*-
"""
Polarimetric Utilities - Helpers for quad-pol dataset validation in GRDK.

Provides functions to inspect an :class:`~grdk.widgets._signals.ImageStack`
and determine its polarimetric collection mode, extract per-polarization
channel arrays, and validate completeness for quad-pol workflows.

Both BIOMASS and NISAR stacks are handled through a unified CYX single-reader
path.  NISAR files are opened with ``polarizations='all'`` by
:func:`~grdk.widgets.geodev.ow_image_loader._try_open_readers`, which causes
the reader to return a ``(C, rows, cols)`` cube from ``read_chip``/
``read_full`` with all polarizations declared in
``channel_metadata[i].polarization``.

``extract_quad_pol_arrays_strided()`` handles both reader types transparently
via ``channel_metadata``.

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


def _reader_all_channel_pols(reader) -> Optional[Dict[str, int]]:
    """Return ``{pol: band_index}`` for *every* polarization channel in *reader*.

    Unlike :func:`_reader_quad_pol_channels`, this works for dual-pol
    (2-channel) CYX readers as well as full quad-pol (4-channel) readers.
    Returns ``None`` when the reader is single-channel or carries no
    polarization metadata.
    """
    meta = getattr(reader, 'metadata', None)

    channel_metadata = getattr(meta, 'channel_metadata', None)
    if channel_metadata:
        mapping: Dict[str, int] = {}
        for ch in channel_metadata:
            pol = getattr(ch, 'polarization', None)
            if isinstance(pol, str) and pol.strip():
                mapping[pol.strip().upper()] = int(ch.index)
        if len(mapping) >= 2:
            return mapping

    pol_list = getattr(meta, 'polarizations', None)
    if pol_list:
        pols_upper = [
            p.strip().upper() for p in pol_list
            if isinstance(p, str) and p.strip()
        ]
        if len(pols_upper) >= 2:
            return {p: i for i, p in enumerate(pols_upper)}

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
        # Full quad-pol multi-band reader (BIOMASS, quad-pol NISAR)
        mb_mapping = _reader_quad_pol_channels(reader)
        if mb_mapping:
            pols.update(mb_mapping.keys())
        else:
            # Dual-pol multi-band reader (2-channel CYX NISAR)
            all_mapping = _reader_all_channel_pols(reader)
            if all_mapping:
                pols.update(all_mapping.keys())
            else:
                # Single-band reader: contributes its one polarization
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


def _split_copol_crosspol(
    pol_arrays: Dict[str, 'np.ndarray'],
) -> Tuple['np.ndarray', 'np.ndarray']:
    """Split a 2-polarization dict into ``(co_pol, cross_pol)`` arrays.

    Co-pol channels are same-character pairs (HH, VV); cross-pol are
    mixed-character pairs (HV, VH).  When both channels share the same
    type (both co-pol or both cross-pol), the two arrays are returned in
    alphabetical polarization order.

    Parameters
    ----------
    pol_arrays : dict
        Maps polarization string \u2192 2-D complex array.  Must have
        exactly 2 entries.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(co_pol, cross_pol)``.
    """
    pol_names = list(pol_arrays.keys())
    if len(pol_names) != 2:
        raise ValueError(
            f"Expected exactly 2 polarization channels, got {pol_names}"
        )

    def _is_like(p: str) -> bool:
        return len(p) == 2 and p[0] == p[1]

    like = [p for p in pol_names if _is_like(p)]
    cross = [p for p in pol_names if not _is_like(p)]

    if len(like) == 1 and len(cross) == 1:
        return pol_arrays[like[0]], pol_arrays[cross[0]]
    # Ambiguous (both co-pol or both cross-pol): sort alphabetically
    pol_names.sort()
    return pol_arrays[pol_names[0]], pol_arrays[pol_names[1]]


def extract_dual_pol_arrays_strided(
    stack: ImageStack,
    max_pixels: int,
) -> Tuple['np.ndarray', 'np.ndarray']:
    """Read and return ``(s_co, s_cross)`` from a dual-pol *stack*.

    Automatically determines which channel is co-pol and which is
    cross-pol from the reader's ``channel_metadata`` polarization tags.

    Parameters
    ----------
    stack : ImageStack
        A dual-pol image stack (exactly 2 polarization channels).
    max_pixels : int
        Maximum pixels per channel.  Pass ``0`` for full resolution.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(s_co, s_cross)`` complex 2-D arrays.
    """
    import math

    if not stack or not stack.readers:
        raise ValueError("Stack is empty \u2014 no readers found.")

    reader = stack.readers[0]
    all_mapping = _reader_all_channel_pols(reader)

    if all_mapping and len(all_mapping) >= 2:
        # CYX single multi-band reader (NISAR opened with polarizations='all')
        rows, cols = _native_dims(stack)
        if rows is None or cols is None:
            stride = 1
        elif max_pixels > 0 and rows * cols > max_pixels:
            stride = math.ceil(math.sqrt(rows * cols / max_pixels))
        else:
            stride = 1

        cube = _read_cyx_chip(reader, stride)  # (C, H\u2019, W\u2019)
        pol_arrays: Dict[str, np.ndarray] = {}
        for pol, idx in all_mapping.items():
            if idx < cube.shape[0]:
                pol_arrays[pol] = cube[idx]

        if len(pol_arrays) < 2:
            raise ValueError(
                f"Only {len(pol_arrays)} channel(s) readable; need 2."
            )
        # Keep only 2 channels (trim extra pols from a quad-pol cube)
        if len(pol_arrays) > 2:
            pol_arrays = dict(
                list(sorted(pol_arrays.items()))[:2]
            )
        return _split_copol_crosspol(pol_arrays)

    # Fallback: two separate single-pol readers
    if len(stack.readers) >= 2:
        shape_r: Optional[Tuple[int, int]] = None
        stride = 1
        pol_arrays_fb: Dict[str, np.ndarray] = {}
        for r in stack.readers[:2]:
            p = _reader_polarization(r)
            if p is None:
                continue
            if shape_r is None:
                meta = getattr(r, 'metadata', None)
                rows_r = int(getattr(meta, 'rows', 0) or 0)
                cols_r = int(getattr(meta, 'cols', 0) or 0)
                if rows_r and cols_r:
                    shape_r = (rows_r, cols_r)
                    total = rows_r * cols_r
                    if max_pixels > 0 and total > max_pixels:
                        stride = math.ceil(
                            math.sqrt(total / max_pixels)
                        )
            if shape_r is None:
                continue
            rows_r, cols_r = shape_r
            data = r.read_chip(0, rows_r, 0, cols_r)
            if stride > 1:
                data = data[..., ::stride, ::stride]
            if data.ndim == 3:
                data = data[0]
            pol_arrays_fb[p] = data

        if len(pol_arrays_fb) == 2:
            return _split_copol_crosspol(pol_arrays_fb)

    raise ValueError(
        "Could not find 2 polarization channels for dual-pol decomposition. "
        "Expected a CYX multi-band reader with 2 channels declared in "
        "channel_metadata, or 2 separate single-pol readers in the stack."
    )


def extract_quad_pol_arrays_strided(
    stack: ImageStack,
    max_pixels: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read and return ``(shh, shv, svh, svv)`` downsampled to *max_pixels*.

    Uses a center-crop chip read so that full-resolution data is never
    materialised in memory.  The output arrays have shape
    ``(ceil(rows/stride), ceil(cols/stride))``.

    Requires a single CYX multi-band reader in the stack (BIOMASS or
    NISAR opened with ``polarizations='all'``).

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


def _read_cyx_chip(reader, stride: int) -> np.ndarray:
    """Read a CYX cube from *reader*, downsampled by *stride* via center crop.

    For ``stride > 1``, reads a chip of shape
    ``(C, ceil(rows/stride), ceil(cols/stride))`` using ``read_chip``.
    This is O(output pixels) and avoids loading the full image into memory
    — critical for large NISAR scenes (e.g. 54 720 × 26 610 pixels,
    4 polarizations, ~47 GB at full resolution).

    For ``stride == 1``, calls ``read_full()``.
    """
    if stride <= 1:
        cube = reader.read_full()
    else:
        meta = getattr(reader, 'metadata', None)
        rows = int(getattr(meta, 'rows', 0) or 0)
        cols = int(getattr(meta, 'cols', 0) or 0)
        if rows and cols:
            out_h = -(-rows // stride)   # ceil(rows / stride)
            out_w = -(-cols // stride)   # ceil(cols / stride)
            r0 = max(0, (rows - out_h) // 2)
            c0 = max(0, (cols - out_w) // 2)
            r1 = min(rows, r0 + out_h)
            c1 = min(cols, c0 + out_w)
            cube = reader.read_chip(r0, r1, c0, c1)
        else:
            cube = reader.read_full()

    # Normalise to CYX.  Trust axis_order from metadata — grdl CYX readers
    # (NISAR polarizations='all', BIOMASS) explicitly set axis_order='CYX'.
    # Only transpose to CYX when axis_order explicitly says 'YXC'.
    if cube.ndim == 3:
        axis_order = getattr(
            getattr(reader, 'metadata', None), 'axis_order', None
        )
        if axis_order == 'YXC':
            cube = np.moveaxis(cube, -1, 0)  # YXC → CYX

    return cube


def _extract_quad_pol_arrays_impl(
    stack: ImageStack,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract HH, HV, VH, VV from a CYX multi-band reader.

    Handles both BIOMASS (multi-band GeoTIFF) and NISAR (HDF5 opened
    with ``polarizations='all'``).  For ``stride > 1``, reads a center-crop
    chip via ``_read_cyx_chip`` to avoid loading full large images.
    """
    if not stack or not stack.readers:
        raise ValueError("Stack is empty — no readers found.")

    for reader in stack.readers:
        mb_mapping = _reader_quad_pol_channels(reader)
        if mb_mapping:
            cube = _read_cyx_chip(reader, stride)
            return (
                cube[mb_mapping['HH']],
                cube[mb_mapping['HV']],
                cube[mb_mapping['VH']],
                cube[mb_mapping['VV']],
            )

    raise ValueError(
        "No quad-pol reader found in stack. "
        "Expected a multi-band CYX reader (axis_order='CYX') with "
        "HH, HV, VH, VV channels declared in channel_metadata."
    )
