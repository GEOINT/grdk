# -*- coding: utf-8 -*-
"""
Band Info - Extract named band metadata from any grdl ImageReader.

Provides a universal ``get_band_info()`` function that dispatches by reader
type and reader channel metadata to extract human-readable band names
(wavelengths, polarizations, spectral band IDs, etc.) for use in the viewer's
band selector UI.

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
"""

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class BandInfo:
    """Description of a single band in an image reader.

    Attributes
    ----------
    index : int
        0-based band index for ``read_chip(bands=[index])``.
    name : str
        Display name, e.g. ``"Band 1 (503.0 nm)"``, ``"HH"``, ``"B04"``.
    description : str
        Longer description, e.g. ``"Green (503.0 nm)"``, ``"Polarization HH"``.
    wavelength: Optional[float]
        Center wavelength in nanometers (if optical/spectral).
    polarization: Optional[str]
        Polarization code (if SAR).
    """

    index: int
    name: str
    description: str = ""
    wavelength: Optional[float] = None
    polarization: Optional[str] = None


def _wavelength_to_color_label(nm: float) -> str:
    """Return a descriptive spectral regime/color label for a wavelength in nm."""
    if nm < 380:
        return "UV"
    elif 380 <= nm < 450:
        return "Violet"
    elif 450 <= nm < 495:
        return "Blue"
    elif 495 <= nm < 570:
        return "Green"
    elif 570 <= nm < 590:
        return "Yellow"
    elif 590 <= nm < 620:
        return "Orange"
    elif 620 <= nm < 750:
        return "Red"
    elif 750 <= nm < 1100:
        return "NIR"
    elif 1100 <= nm < 3000:
        return "SWIR"
    elif 3000 <= nm < 5000:
        return "MWIR"
    elif 5000 <= nm <= 14000:
        return "LWIR"
    return ""


def get_band_info(reader: Any) -> List[BandInfo]:
    """Extract named band information from any grdl reader.

    Dispatches by reader metadata (channel_metadata, channels, bands)
    and reader class to extract sensor- and modality-specific band names.
    Falls back to generic ``Band 1``, etc. for unknown reader types.

    Parameters
    ----------
    reader : ImageReader
        Any grdl reader instance.

    Returns
    -------
    List[BandInfo]
        One entry per band, ordered by band index.
    """
    # 1. First priority: Check canonical reader.metadata.channel_metadata
    meta = getattr(reader, 'metadata', None)
    channel_meta = getattr(meta, 'channel_metadata', None) if meta is not None else None
    if channel_meta and len(channel_meta) > 0:
        band_infos: List[BandInfo] = []
        for i, ch in enumerate(channel_meta):
            idx = getattr(ch, 'index', i)
            extras = getattr(ch, 'extras', {}) or {}
            raw_wl = extras.get('wavelength') or getattr(ch, 'wavelength', None)
            wl: Optional[float] = None
            if raw_wl is not None:
                try:
                    wl = float(raw_wl)
                except (ValueError, TypeError):
                    wl = None

            pol = getattr(ch, 'polarization', None)
            name_attr = getattr(ch, 'name', '') or f'band_{idx + 1}'

            # Formatting
            if wl is not None:
                color_name = _wavelength_to_color_label(wl)
                display_name = f"Band {idx + 1} ({wl:.1f} nm)"
                desc = f"{color_name} ({wl:.1f} nm)" if color_name else f"{wl:.1f} nm"
                band_infos.append(BandInfo(index=idx, name=display_name, description=desc, wavelength=wl, polarization=pol))
            elif pol:
                display_name = f"Band {idx + 1} ({pol})"
                desc = f"Polarization {pol}"
                band_infos.append(BandInfo(index=idx, name=display_name, description=desc, wavelength=None, polarization=pol))
            else:
                display_name = f"Band {idx + 1} ({name_attr})"
                band_infos.append(BandInfo(index=idx, name=display_name, description=name_attr, wavelength=None, polarization=None))
        return band_infos

    # --- BIOMASS L1: polarization channels (HH, HV, VH, VV) ---
    try:
        from grdl.IO.sar.biomass import BIOMASSL1Reader
        if isinstance(reader, BIOMASSL1Reader):
            pols = getattr(reader, 'polarizations', None)
            if pols:
                return [
                    BandInfo(i, f"Band {i} ({pol})", f"Polarization {pol}", polarization=pol)
                    for i, pol in enumerate(pols)
                ]
    except ImportError:
        pass

    # --- Sentinel-1 SLC: all available polarizations ---
    try:
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader
        if isinstance(reader, Sentinel1SLCReader):
            all_pols = []
            try:
                all_pols = reader.get_available_polarizations()
            except Exception:
                pass
            swath_info = reader.metadata.get('swath_info') if hasattr(reader.metadata, 'get') else getattr(reader.metadata, 'swath_info', None)
            swath = getattr(swath_info, 'swath', None) if swath_info else None

            if len(all_pols) > 1:
                return [
                    BandInfo(
                        i,
                        f"Band {i} ({pol})",
                        f"{swath} {pol}".strip() if swath else f"Polarization {pol}",
                        polarization=pol
                    )
                    for i, pol in enumerate(all_pols)
                ]
            pol = getattr(reader.metadata, 'polarization', None) or "Complex"
            desc = f"{swath} {pol}".strip() if swath else pol
            return [BandInfo(0, f"Band 1 ({pol})", desc, polarization=pol)]
    except ImportError:
        pass

    # --- SICD: single complex band ---
    try:
        from grdl.IO.sar.sicd import SICDReader
        if isinstance(reader, SICDReader):
            all_pols = []
            try:
                all_pols = reader.get_available_polarizations()
            except Exception:
                pass
            freq = getattr(reader.metadata, 'frequency', None)
            freq_label = f"Freq {freq}" if freq else ""
            if len(all_pols) > 1:
                return [
                    BandInfo(i, f"Band {i} ({pol})", f"{freq_label} {pol}".strip(), polarization=pol)
                    for i, pol in enumerate(all_pols)
                ]
            pol = getattr(reader.metadata, 'polarization', None) or "Complex"
            desc = f"{freq_label} {pol}".strip() if freq_label else pol
            return [BandInfo(0, f"Band 1 ({pol})", desc, polarization=pol)]
    except ImportError:
        pass

    # --- SIDD: single detected band ---
    try:
        from grdl.IO.sar.sidd import SIDDReader
        if isinstance(reader, SIDDReader):
            return [BandInfo(0, "Band 1 (Detected)", "SAR detected image")]
    except ImportError:
        pass

    # --- Sentinel-2: spectral bands with wavelength info ---
    try:
        from grdl.IO.eo.sentinel2 import Sentinel2Reader
        if isinstance(reader, Sentinel2Reader):
            band_id = reader.metadata.get('band_id') if hasattr(reader.metadata, 'get') else getattr(reader.metadata, 'band_id', None)
            wl = reader.metadata.get('wavelength_center') if hasattr(reader.metadata, 'get') else getattr(reader.metadata, 'wavelength_center', None)
            if band_id:
                desc = f"{wl:.0f} nm" if wl else ""
                color_name = _wavelength_to_color_label(wl) if wl else ""
                if color_name:
                    desc = f"{color_name} ({desc})"
                return [BandInfo(0, f"{band_id} ({wl:.0f} nm)" if wl else band_id, desc, wavelength=wl)]
            fname = reader.filepath.stem.upper()
            if "_TCI_" in fname:
                return [BandInfo(0, "TCI (RGB)", "True Color Image (RGB)")]
            return [BandInfo(0, "Band 1", "Sentinel-2")]
    except ImportError:
        pass

    # --- Generic fallback: use metadata.bands count ---
    num_bands = _get_num_bands(reader)
    return [BandInfo(i, f"Band {i}", "") for i in range(num_bands)]


def _get_num_bands(reader: Any) -> int:
    """Get band count from a reader using available metadata."""
    meta = getattr(reader, 'metadata', None)
    if meta is not None:
        bands = meta.get('bands') if hasattr(meta, 'get') else getattr(meta, 'bands', None)
        if bands is not None and isinstance(bands, int) and bands > 0:
            return bands

    try:
        shape = reader.get_shape()
        if len(shape) == 3:
            return shape[2]
        return 1
    except Exception:
        return 1


__all__ = ["BandInfo", "get_band_info"]
