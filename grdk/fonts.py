# -*- coding: utf-8 -*-
"""
Font management and offline fallback font configuration for GRDK.

Provides automatic discovery and configuration of font directories for Qt:
1. Bundled DejaVu TrueType fonts (Bitstream Vera / Free font license)
   located under ``grdk/resources/fonts`` (offline capable).
2. Host system font directories (e.g. C:/Windows/Fonts on Windows,
   fontconfig paths on Linux).
3. Configures ``QT_QPA_FONTDIR`` if not already explicitly set in the environment.
4. Registers bundled fonts with ``QFontDatabase`` once a ``QApplication`` exists.

Author
------
Claude Code / Steven Siebert

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

_log = logging.getLogger("grdk.fonts")

_BUNDLED_FONTS_DIR = Path(__file__).resolve().parent / "resources" / "fonts"


def get_bundled_fonts_dir() -> Optional[Path]:
    """Return path to bundled fonts directory if it exists and contains TTF files."""
    if _BUNDLED_FONTS_DIR.is_dir():
        ttf_files = list(_BUNDLED_FONTS_DIR.glob("*.ttf"))
        if ttf_files:
            return _BUNDLED_FONTS_DIR
    return None


def get_system_font_dir() -> Optional[Path]:
    """Return platform system font directory if available."""
    if sys.platform == "win32":
        win_fonts = Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts"
        if win_fonts.is_dir():
            return win_fonts
    elif sys.platform == "darwin":
        mac_fonts = Path("/System/Library/Fonts")
        if mac_fonts.is_dir():
            return mac_fonts
    else:
        linux_fonts = Path("/usr/share/fonts")
        if linux_fonts.is_dir():
            return linux_fonts
    return None


def configure_qt_font_environment() -> None:
    """Ensure QT_QPA_FONTDIR is configured before Qt QPA platform initialization.

    If ``QT_QPA_FONTDIR`` is not set, checks for bundled fonts or system fonts
    and sets ``QT_QPA_FONTDIR`` to ensure Qt's font engine (offscreen or standard)
    finds valid TrueType fonts without warnings.
    """
    if "QT_QPA_FONTDIR" in os.environ:
        return

    # Check bundled font directory first for reproducible offline behavior
    bundled = get_bundled_fonts_dir()
    if bundled is not None:
        os.environ["QT_QPA_FONTDIR"] = str(bundled)
        _log.debug("Set QT_QPA_FONTDIR to bundled fonts: %s", bundled)
        return

    # Fallback to system fonts
    system_font_dir = get_system_font_dir()
    if system_font_dir is not None:
        os.environ["QT_QPA_FONTDIR"] = str(system_font_dir)
        _log.debug("Set QT_QPA_FONTDIR to system fonts: %s", system_font_dir)


def register_application_fonts() -> List[int]:
    """Register bundled fonts into QFontDatabase.

    Should be called after ``QApplication`` or ``QGuiApplication`` has been instantiated.
    Returns list of font IDs added.
    """
    bundled = get_bundled_fonts_dir()
    if bundled is None:
        return []

    try:
        from PyQt6.QtGui import QFontDatabase
    except ImportError:
        return []

    added_ids = []
    for font_file in sorted(bundled.glob("*.ttf")):
        font_id = QFontDatabase.addApplicationFont(str(font_file))
        if font_id >= 0:
            added_ids.append(font_id)
            _log.debug("Registered font %s (id: %d)", font_file.name, font_id)
        else:
            _log.warning("Failed to register font %s", font_file.name)

    return added_ids


# Automatically set QT_QPA_FONTDIR on module import
configure_qt_font_environment()
