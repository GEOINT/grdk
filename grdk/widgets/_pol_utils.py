# -*- coding: utf-8 -*-
"""
Polarimetric Utilities - Stack-level helpers for the GRDK widget layer.

This module owns the I/O side of polarimetric processing: reading channels
from grdl readers, strided downsampling for display, and channel routing.
All signal-processing mathematics lives in grdl.  The boundary is the
array handoff: grdk reads + strides, grdl decomposes.

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
"""

# Standard library
import math
from typing import Dict, Optional, Set, Tuple

# Third-party
import numpy as np

# GRDK internal
from grdk.widgets._signals import ImageStack

# The four channels required for a complete quad-pol acquisition.
_QUAD_POL_CHANNELS: Set[str] = {'HH', 'HV', 'VH', 'VV'}


# ---------------------------------------------------------------------------
# Reader I/O helpers (grdk-owned; live here because they serve display needs)
# ---------------------------------------------------------------------------

def channel_pol_map(reader) -> Dict[str, int]:
    """Return ``{polarization: band_index}`` for all channels in *reader*.

    Reads ``reader.metadata.channel_metadata``.  Returns an empty dict
    when that attribute is absent or contains no polarization strings.
    """
    meta = getattr(reader, 'metadata', None)
    channel_metadata = getattr(meta, 'channel_metadata', None)
    if not channel_metadata:
        return {}
    result: Dict[str, int] = {}
    for ch in channel_metadata:
        pol = getattr(ch, 'polarization', None)
        if isinstance(pol, str) and pol.strip():
            result[pol.strip().upper()] = int(ch.index)
    return result


def read_cyx_with_stride(reader, max_pixels: int) -> np.ndarray:
    """Read a CYX cube from *reader*, centre-crop strided to *max_pixels*.

    Downsamples spatially so the viewer stays responsive at full-scene
    scale.  Pass ``max_pixels=0`` to read at full resolution.

    Parameters
    ----------
    reader : ImageReader
        Any grdl reader.
    max_pixels : int
        Maximum H\u00d7W pixels per channel.  ``0`` means no cap.

    Returns
    -------
    np.ndarray
        ``(C, rows, cols)`` array in CYX layout.
    """
    meta = getattr(reader, 'metadata', None)
    rows = int(getattr(meta, 'rows', 0) or 0)
    cols = int(getattr(meta, 'cols', 0) or 0)

    if max_pixels <= 0 or not (rows and cols) or rows * cols <= max_pixels:
        cube = reader.read_full()
    else:
        stride = math.ceil(math.sqrt(rows * cols / max_pixels))
        out_h = -(-rows // stride)   # ceil(rows / stride)
        out_w = -(-cols // stride)   # ceil(cols / stride)
        r0 = max(0, (rows - out_h) // 2)
        c0 = max(0, (cols - out_w) // 2)
        r1 = min(rows, r0 + out_h)
        c1 = min(cols, c0 + out_w)
        cube = reader.read_chip(r0, r1, c0, c1)

    if cube.ndim == 3:
        axis_order = getattr(meta, 'axis_order', None)
        if axis_order == 'YXC':
            cube = np.moveaxis(cube, -1, 0)
    return cube


def split_copol_crosspol(
    pol_map: Dict[str, int],
    cube: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(s_co, s_cross)`` from a 2-polarization CYX cube.

    Co-pol channels share both characters (HH, VV); cross-pol channels
    have mixed characters (HV, VH).  For compact-pol where both channels
    are the same type, alphabetical order is used.

    Parameters
    ----------
    pol_map : dict
        ``{polarization: band_index}`` for the two channels.
    cube : np.ndarray
        CYX array containing at least those bands.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(s_co, s_cross)`` 2-D complex arrays.
    """
    def _is_copol(pol: str) -> bool:
        return len(pol) == 2 and pol[0] == pol[1]

    pol_items = sorted(pol_map.items())[:2]
    pol_names = [p for p, _ in pol_items]
    like = [p for p in pol_names if _is_copol(p)]
    cross = [p for p in pol_names if not _is_copol(p)]
    if len(like) == 1 and len(cross) == 1:
        return cube[pol_map[like[0]]], cube[pol_map[cross[0]]]
    # Ambiguous (compact-pol): alphabetical order
    return cube[pol_items[0][1]], cube[pol_items[1][1]]


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
    meta = getattr(reader, 'metadata', None)

    # NISAR / CPHD / sensor-specific: metadata.polarization is a plain string
    pol = getattr(meta, 'polarization', None)
    if isinstance(pol, str) and pol:
        return pol.upper()

    # Sentinel-1 SLC: polarization stored inside swath_info sub-object
    swath_info = (
        meta.get('swath_info') if hasattr(meta, 'get')
        else getattr(meta, 'swath_info', None)
    )
    if swath_info is not None:
        p = getattr(swath_info, 'polarization', None)
        if isinstance(p, str) and p.strip():
            return p.strip().upper()

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

    # TerraSAR-X: polarization stored as a semi-private reader attribute
    # (no public metadata property for the currently-loaded channel).
    p = getattr(reader, '_requested_polarization', None)
    if isinstance(p, str) and p.strip():
        return p.strip().upper()

    return None


def _reader_quad_pol_channels(reader) -> Optional[Dict[str, int]]:
    """Return a ``{pol: band_index}`` mapping when a reader holds all four
    quad-pol channels in a single multi-band cube (e.g. BIOMASS, NISAR
    opened with ``polarizations='all'``).

    Parameters
    ----------
    reader : ImageReader
        Any GRDL reader instance.

    Returns
    -------
    Optional[Dict[str, int]]
        Mapping ``{'HH': 0, 'HV': 1, 'VH': 2, 'VV': 3}`` when all four
        polarizations are present, or ``None`` otherwise.
    """
    meta = getattr(reader, 'metadata', None)

    channel_metadata = getattr(meta, 'channel_metadata', None)
    if channel_metadata:
        mapping: Dict[str, int] = {}
        for ch in channel_metadata:
            pol = getattr(ch, 'polarization', None)
            if isinstance(pol, str) and pol.strip().upper() in _QUAD_POL_CHANNELS:
                mapping[pol.strip().upper()] = ch.index
        if _QUAD_POL_CHANNELS.issubset(mapping.keys()):
            return mapping

    # Fallback: BIOMASSMetadata.polarizations list
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


def is_quad_pol(stack: ImageStack) -> bool:
    """Return ``True`` when *stack* exposes all four quad-pol channels.

    Delegates mode detection to
    :func:`grdl.image_processing.decomposition.pipeline.polarimetric_mode_from_reader`
    for the fast CYX single-reader case (NISAR opened with
    ``polarizations='all'``, BIOMASS).  Falls back to aggregating across
    per-pol readers for legacy multi-reader stacks.

    Parameters
    ----------
    stack : ImageStack
        The image stack to inspect.

    Returns
    -------
    bool
    """
    if not stack or not stack.readers:
        return False

    from grdl.vocabulary import PolarimetricMode

    # Fast path: single multi-pol CYX reader declares all 4 pols
    for reader in stack.readers:
        if PolarimetricMode.from_reader(reader) == PolarimetricMode.QUAD_POL:
            return True

    # Aggregate path: 4 separate single-pol readers
    pols: Set[str] = set()
    for reader in stack.readers:
        mb = _reader_quad_pol_channels(reader)
        if mb:
            pols.update(mb.keys())
        else:
            p = _reader_polarization(reader)
            if p:
                pols.add(p)
    return _QUAD_POL_CHANNELS.issubset(pols)


def extract_quad_pol_arrays_strided(
    stack: ImageStack,
    max_pixels: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read and return ``(shh, shv, svh, svv)`` downsampled to *max_pixels*.

    Delegates all striding and channel extraction to
    :func:`grdl.image_processing.decomposition.pipeline.extract_quad_pol_arrays`.
    This function's sole responsibility is locating the multi-pol CYX
    reader within the stack.

    Parameters
    ----------
    stack : ImageStack
        A quad-pol image stack.  Requires a single CYX multi-band reader
        (BIOMASS or NISAR opened with ``polarizations='all'``).
    max_pixels : int
        Maximum number of pixels (H \u00d7 W) to return per channel.
        Pass ``0`` to read at full resolution.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(shh, shv, svh, svv)`` complex 2-D arrays.
    """
    if not stack or not stack.readers:
        raise ValueError("Stack is empty \u2014 no readers found.")

    for reader in stack.readers:
        if _reader_quad_pol_channels(reader):
            pol_map = channel_pol_map(reader)
            cube = read_cyx_with_stride(reader, max_pixels)
            return (
                cube[pol_map['HH']],
                cube[pol_map['HV']],
                cube[pol_map['VH']],
                cube[pol_map['VV']],
            )

    raise ValueError(
        "No quad-pol reader found in stack. "
        "Expected a multi-band CYX reader (axis_order='CYX') with "
        "HH, HV, VH, VV channels declared in channel_metadata."
    )


def _native_dims(stack: ImageStack) -> Tuple[Optional[int], Optional[int]]:
    """Return ``(rows, cols)`` from the first reader with shape metadata.

    Used by :class:`~grdk.widgets.geodev.ow_covariance_matrix._MatrixWorker`
    to record original dimensions in the output signal metadata.

    Returns ``(None, None)`` when no reader exposes shape metadata.
    """
    for reader in stack.readers:
        meta = getattr(reader, 'metadata', None)
        rows = getattr(meta, 'rows', None)
        cols = getattr(meta, 'cols', None)
        if rows and cols:
            return int(rows), int(cols)
    return None, None
