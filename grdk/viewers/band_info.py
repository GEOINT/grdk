# -*- coding: utf-8 -*-
"""
Band Info - Extract named band metadata from any grdl ImageReader.

Provides a universal ``get_band_info()`` function that dispatches by reader
type to extract human-readable band names (polarizations, spectral band
IDs, etc.) for use in the viewer's band selector UI.

Dependencies
------------
None (pure Python, lazy-imports grdl readers)

Author
------
Claude Code (Anthropic)

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-18

Modified
--------
2026-02-18
"""

from dataclasses import dataclass
from typing import Any, List


@dataclass
class BandInfo:
    """Description of a single band in an image reader.

    Attributes
    ----------
    index : int
        0-based band index for ``read_chip(bands=[index])``.
    name : str
        Short display name, e.g. ``"HH"``, ``"B04"``, ``"Band 0"``.
    description : str
        Longer description, e.g. ``"Polarization HH"``, ``"Red (665 nm)"``.
    """

    index: int
    name: str
    description: str = ""


def get_band_info(reader: Any) -> List[BandInfo]:
    """Extract named band information from any grdl reader.

    Dispatches by reader class to extract sensor-specific band names.
    Falls back to generic ``Band 0``, ``Band 1``, etc. for unknown
    reader types.

    Parameters
    ----------
    reader : ImageReader
        Any grdl reader instance.

    Returns
    -------
    List[BandInfo]
        One entry per band, ordered by band index.
    """
    # --- BIOMASS L1: polarization channels (HH, HV, VH, VV) ---
    try:
        from grdl.IO.sar.biomass import BIOMASSL1Reader
        if isinstance(reader, BIOMASSL1Reader):
            pols = getattr(reader, 'polarizations', None)
            if pols:
                return [
                    BandInfo(i, pol, f"Polarization {pol}")
                    for i, pol in enumerate(pols)
                ]
    except ImportError:
        pass

    # --- Sentinel-1 SLC: all available polarizations ---
    try:
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader
        if isinstance(reader, Sentinel1SLCReader):
            # List all available polarizations so the combo shows them all
            all_pols = []
            try:
                all_pols = reader.get_available_polarizations()
            except Exception:
                pass
            swath_info = reader.metadata.get('swath_info')
            swath = getattr(swath_info, 'swath', None) if swath_info else None

            if len(all_pols) > 1:
                return [
                    BandInfo(
                        i, pol,
                        f"{swath} {pol}" if swath else f"Polarization {pol}",
                    )
                    for i, pol in enumerate(all_pols)
                ]
            # Single pol fallback
            if swath_info:
                pol = getattr(swath_info, 'polarization', None)
                name = pol or "Complex"
                desc = f"{swath} {pol}" if swath and pol else name
                return [BandInfo(0, name, desc)]
            return [BandInfo(0, "Complex", "SAR SLC")]
    except ImportError:
        pass

    # --- SICD: single complex band ---
    try:
        from grdl.IO.sar.sicd import SICDReader
        if isinstance(reader, SICDReader):
            return [BandInfo(0, "Complex", "SAR complex data")]
    except ImportError:
        pass

    # --- CPHD: single complex band ---
    try:
        from grdl.IO.sar.cphd import CPHDReader
        if isinstance(reader, CPHDReader):
            return [BandInfo(0, "Complex", "Compensated phase history")]
    except ImportError:
        pass

    # --- CRSD: single complex band ---
    try:
        from grdl.IO.sar.crsd import CRSDReader
        if isinstance(reader, CRSDReader):
            return [BandInfo(0, "Complex", "Compensated received signal")]
    except ImportError:
        pass

    # --- TerraSAR-X / TanDEM-X: all available polarizations ---
    try:
        from grdl.IO.sar.terrasar import TerraSARReader
        if isinstance(reader, TerraSARReader):
            # List all available polarizations so the combo shows them all
            all_pols = []
            try:
                all_pols = reader.get_available_polarizations()
            except Exception:
                pass
            if len(all_pols) > 1:
                return [
                    BandInfo(i, pol, f"Polarization {pol}")
                    for i, pol in enumerate(all_pols)
                ]
            pol = getattr(reader, '_requested_polarization', None)
            if pol:
                return [BandInfo(0, pol, f"Polarization {pol}")]
            return [BandInfo(0, "Complex", "SAR")]
    except ImportError:
        pass

    # --- SIDD: single detected band ---
    try:
        from grdl.IO.sar.sidd import SIDDReader
        if isinstance(reader, SIDDReader):
            return [BandInfo(0, "Detected", "SAR detected image")]
    except ImportError:
        pass

    # --- Sentinel-2: spectral bands with wavelength info ---
    try:
        from grdl.IO.eo.sentinel2 import Sentinel2Reader
        if isinstance(reader, Sentinel2Reader):
            band_id = reader.metadata.get('band_id')
            wl = reader.metadata.get('wavelength_center')
            if band_id:
                desc = f"{wl:.0f} nm" if wl else ""
                return [BandInfo(0, band_id, desc)]
            # TCI or other non-spectral product — check filename
            fname = reader.filepath.stem.upper()
            if "_TCI_" in fname:
                return [BandInfo(0, "TCI", "True Color Image (RGB)")]
            return [BandInfo(0, "Band 0", "Sentinel-2")]
    except ImportError:
        pass

    # --- Generic fallback: use metadata.bands count ---
    num_bands = _get_num_bands(reader)
    return [BandInfo(i, f"Band {i}", "") for i in range(num_bands)]


def _get_num_bands(reader: Any) -> int:
    """Get band count from a reader using available metadata."""
    # Try metadata.bands
    meta = getattr(reader, 'metadata', None)
    if meta is not None:
        bands = meta.get('bands') if hasattr(meta, 'get') else getattr(meta, 'bands', None)
        if bands is not None and isinstance(bands, int) and bands > 0:
            return bands

    # Try get_shape() — (rows, cols) or (rows, cols, bands)
    try:
        shape = reader.get_shape()
        if len(shape) == 3:
            # Channels-first: (C, H, W) if shape[0] < shape[1] and shape[0] < shape[2]
            # Channels-last: (H, W, C) otherwise
            # grdl uses channels-last for get_shape but channels-first for read_chip
            return shape[2]
        return 1
    except Exception:
        return 1


__all__ = ["BandInfo", "get_band_info"]
