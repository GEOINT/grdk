# -*- coding: utf-8 -*-
"""
Image Archetypes - Role- and Modality-Based Visualization Layout Presets for GRDK ELT.

Supports predefined and custom layout archetypes:
- SAR Polarimetric Archetype (Pauli, Lee filter, polarimetric decomposition docks)
- Optical / Multispectral Archetype (RGB gun assignment, DRA histogram, spectral stretches)
- Hyperspectral Archetype (Spectral profile probe, cube navigation, band math)
- General / Default Archetype

Presets store dock configurations, default display stretches, colormaps,
and suggested tool visibility, persistable to ~/.grdk/archetypes.json.

Author
------
Claude Code / Steven Siebert

License
-------
MIT License
Copyright (c) 2026 geoint.org
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_log = logging.getLogger("grdk.archetypes")

_CONFIG_DIR = Path.home() / ".grdk"
_ARCHETYPES_FILE = _CONFIG_DIR / "archetypes.json"


@dataclass
class ArchetypeDisplayDefaults:
    colormap: str = "grayscale"
    stretch_mode: str = "minmax"
    percentile_low: float = 1.0
    percentile_high: float = 99.0
    gamma: float = 1.0
    contrast: float = 1.0
    brightness: float = 0.0
    invert: bool = False


@dataclass
class ImageArchetype:
    name: str
    modality: str  # "SAR", "HSI", "MSI", "PAN", "GENERAL"
    description: str
    display: ArchetypeDisplayDefaults = field(default_factory=ArchetypeDisplayDefaults)
    visible_docks: List[str] = field(default_factory=lambda: ["Display", "Metadata"])
    dock_positions: Dict[str, str] = field(default_factory=dict)
    suggested_processors: List[str] = field(default_factory=list)
    custom: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ImageArchetype:
        display_data = data.get("display", {})
        display = ArchetypeDisplayDefaults(**display_data) if isinstance(display_data, dict) else ArchetypeDisplayDefaults()
        return cls(
            name=data.get("name", "Default"),
            modality=data.get("modality", "GENERAL"),
            description=data.get("description", ""),
            display=display,
            visible_docks=data.get("visible_docks", ["Display", "Metadata"]),
            dock_positions=data.get("dock_positions", {}),
            suggested_processors=data.get("suggested_processors", []),
            custom=data.get("custom", False),
        )


BUILTIN_ARCHETYPES: Dict[str, ImageArchetype] = {
    "General / Standard": ImageArchetype(
        name="General / Standard",
        modality="GENERAL",
        description="Standard single/multi-band display with display controls and metadata.",
        display=ArchetypeDisplayDefaults(colormap="grayscale", stretch_mode="minmax"),
        visible_docks=["Display", "Metadata"],
        suggested_processors=[],
    ),
    "SAR Polarimetric": ImageArchetype(
        name="SAR Polarimetric",
        modality="SAR",
        description="Optimized for complex/detected SAR polarimetry, Pauli decomposition, and speckle reduction.",
        display=ArchetypeDisplayDefaults(
            colormap="grayscale",
            stretch_mode="percentile",
            percentile_low=2.0,
            percentile_high=98.0,
            gamma=1.0,
        ),
        visible_docks=["Display", "Metadata", "Polarimetric"],
        suggested_processors=["PauliDecomposition", "LeeFilter", "FrostFilter", "DegreeOfPolarization"],
    ),
    "Optical / Multispectral": ImageArchetype(
        name="Optical / Multispectral",
        modality="MSI",
        description="Optimized for multispectral imagery with natural RGB presets and DRA histogram stretching.",
        display=ArchetypeDisplayDefaults(
            colormap="grayscale",
            stretch_mode="percentile",
            percentile_low=1.0,
            percentile_high=99.0,
        ),
        visible_docks=["Display", "Metadata", "Histogram & DRA"],
        suggested_processors=["PanSharpeningFusion", "NDVI", "NDWI"],
    ),
    "Hyperspectral Exploitation": ImageArchetype(
        name="Hyperspectral Exploitation",
        modality="HSI",
        description="Optimized for HSI datacubes with spectral profiling, narrow-band selection, and band math.",
        display=ArchetypeDisplayDefaults(
            colormap="grayscale",
            stretch_mode="percentile",
            percentile_low=1.0,
            percentile_high=99.0,
        ),
        visible_docks=["Display", "Metadata", "Spectral Profile", "Histogram & DRA"],
        suggested_processors=["SpectralAngleMapper", "BandMath", "ContinuumRemoval"],
    ),
}


class ArchetypeManager:
    """Manages built-in and user-saved image archetypes."""

    def __init__(self) -> None:
        self._archetypes: Dict[str, ImageArchetype] = dict(BUILTIN_ARCHETYPES)
        self.load_user_archetypes()

    @property
    def archetypes(self) -> Dict[str, ImageArchetype]:
        return self._archetypes

    def get_archetype(self, name: str) -> Optional[ImageArchetype]:
        return self._archetypes.get(name)

    def detect_archetype_for_reader(self, reader: Any) -> ImageArchetype:
        """Infer best matching archetype from reader metadata or class type."""
        meta = getattr(reader, "metadata", None)
        reader_name = type(reader).__name__.lower()

        # Check for SAR
        if "sar" in reader_name or "sicd" in reader_name or "sidd" in reader_name or "sentinel1" in reader_name or "biomass" in reader_name:
            return self._archetypes["SAR Polarimetric"]

        # Check channel metadata or band count for HSI
        channels = getattr(meta, "channel_metadata", None) if meta is not None else None
        if channels and len(channels) >= 16:
            return self._archetypes["Hyperspectral Exploitation"]

        # Check for MSI
        if channels and len(channels) >= 4:
            return self._archetypes["Optical / Multispectral"]

        # Fallback to general
        return self._archetypes["General / Standard"]

    def save_user_archetype(self, archetype: ImageArchetype) -> None:
        archetype.custom = True
        self._archetypes[archetype.name] = archetype
        self._persist_user_archetypes()

    def delete_user_archetype(self, name: str) -> bool:
        if name in self._archetypes and self._archetypes[name].custom:
            del self._archetypes[name]
            self._persist_user_archetypes()
            return True
        return False

    def load_user_archetypes(self) -> None:
        if not _ARCHETYPES_FILE.exists():
            return
        try:
            with open(_ARCHETYPES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data.get("archetypes", []):
                arch = ImageArchetype.from_dict(item)
                arch.custom = True
                self._archetypes[arch.name] = arch
        except Exception as e:
            _log.warning("Could not load user archetypes from %s: %s", _ARCHETYPES_FILE, e)

    def _persist_user_archetypes(self) -> None:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        custom_list = [arch.to_dict() for arch in self._archetypes.values() if arch.custom]
        try:
            with open(_ARCHETYPES_FILE, "w", encoding="utf-8") as f:
                json.dump({"archetypes": custom_list}, f, indent=2)
        except Exception as e:
            _log.error("Failed to save user archetypes to %s: %s", _ARCHETYPES_FILE, e)


__all__ = [
    "ArchetypeDisplayDefaults",
    "ImageArchetype",
    "ArchetypeManager",
    "BUILTIN_ARCHETYPES",
]
