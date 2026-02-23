# -*- coding: utf-8 -*-
"""
GeoImageViewer - Single-pane geospatial image viewer.

Composite widget combining TiledImageCanvas, CoordinateBar, and
VectorOverlayLayer into a self-contained viewer pane.  Provides
``open_any()`` and ``create_geolocation()`` as public module-level
functions for reuse by DualGeoViewer and scripts.

Dependencies
------------
PyQt6

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
2026-02-17

Modified
--------
2026-02-18
"""

# Standard library
import contextlib
import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

_log = logging.getLogger("grdk.geo_viewer")

# Third-party
import numpy as np

try:
    from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget
    from PyQt6.QtCore import Qt, pyqtSignal as Signal

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False

from grdk.viewers.band_info import BandInfo, get_band_info
from grdk.viewers.image_canvas import DisplaySettings
from grdk.viewers.tiled_canvas import TiledImageCanvas
from grdk.viewers.coordinate_bar import CoordinateBar
from grdk.viewers.vector_overlay import VectorOverlayLayer
from grdk.widgets.colorbar import ColorBarWidget


# ---------------------------------------------------------------------------
# Public module-level utilities (reusable by DualGeoViewer, scripts, etc.)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _suppress_stderr():
    """Temporarily silence C-level stderr (e.g. GDAL TXTFMT warnings)."""
    stderr_fd = 2
    try:
        old_fd = os.dup(stderr_fd)
    except OSError:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        yield
    finally:
        os.dup2(old_fd, stderr_fd)
        os.close(old_fd)


def _find_biomass_product_dir(path: Path) -> Optional[Path]:
    """Locate the BIOMASS product directory containing annotation/.

    BIOMASS products may be nested: the user-provided directory may
    contain a single child directory (often with the same name) that
    holds the actual ``annotation/`` and ``measurement/`` folders.

    Returns the product directory path, or None if not found.
    """
    if (path / "annotation").is_dir():
        return path
    # Check immediate subdirectories for nested structure
    for child in path.iterdir():
        if child.is_dir() and (child / "annotation").is_dir():
            return child
    return None


def _find_sentinel2_band_file(safe_dir: Path) -> Optional[Path]:
    """Find the best JP2 band file inside a Sentinel-2 .SAFE directory.

    Prefers TCI (True Color Image) at 10m, then B04 (Red) at 10m,
    then any spectral band JP2 at the highest resolution available.

    Returns the JP2 file path, or None if not found.
    """
    # Locate IMG_DATA directory inside GRANULE
    granule_dir = safe_dir / "GRANULE"
    if not granule_dir.is_dir():
        return None

    img_data_dirs = list(granule_dir.glob("*/IMG_DATA"))
    if not img_data_dirs:
        return None
    img_data = img_data_dirs[0]

    # Search in resolution order: 10m > 20m > 60m
    for res_dir_name in ("R10m", "R20m", "R60m"):
        res_dir = img_data / res_dir_name
        if not res_dir.is_dir():
            continue
        jp2_files = list(res_dir.glob("*.jp2"))
        if not jp2_files:
            continue

        # Prefer TCI (True Color Image — pre-composed RGB)
        for f in jp2_files:
            if "_TCI_" in f.name:
                return f

        # Prefer B04 (Red band)
        for f in jp2_files:
            if "_B04_" in f.name:
                return f

        # Prefer any spectral band (B01–B12, B8A)
        for f in jp2_files:
            if "_B" in f.name:
                return f

        # Any JP2 in this resolution tier
        return jp2_files[0]

    # Fallback: any JP2 under IMG_DATA (flat structure)
    jp2_files = list(img_data.glob("*.jp2"))
    if jp2_files:
        for f in jp2_files:
            if "_TCI_" in f.name:
                return f
        for f in jp2_files:
            if "_B04_" in f.name:
                return f
        return jp2_files[0]

    return None


def open_any(filepath: Union[str, Path]) -> Any:
    """Open any supported grdl imagery file or directory.

    Tries all grdl modality openers in priority order: directory-based
    formats first (BIOMASS, Sentinel-2 .SAFE), then SAR (so NITF files
    containing SICD complex data are handled correctly), then generic
    formats, then EO, IR, and multispectral.

    Parameters
    ----------
    filepath : str or Path
        Path to the image file or directory.

    Returns
    -------
    ImageReader
        grdl reader instance.

    Raises
    ------
    ValueError
        If no opener can handle the file.
    """
    path = Path(filepath)
    errors = []
    _log.info("open_any: trying %s", filepath)

    # Suppress C-level GDAL warnings (e.g. "TXTFMT: Invalid field value")
    # that are emitted to stderr when reading NITF/SICD files.
    with _suppress_stderr():
        # 0. Directory-based formats
        if path.is_dir():
            # BIOMASS — directory name contains 'BIO' and product type
            if 'BIO' in path.name.upper():
                product_dir = _find_biomass_product_dir(path)
                if product_dir is not None:
                    try:
                        from grdl.IO import open_biomass
                        return open_biomass(product_dir)
                    except (ValueError, ImportError, Exception) as e:
                        errors.append(f"BIOMASS: {e}")

            # Sentinel-2 .SAFE directory
            if path.name.upper().endswith('.SAFE'):
                band_file = _find_sentinel2_band_file(path)
                if band_file is not None:
                    try:
                        from grdl.IO.eo.sentinel2 import Sentinel2Reader
                        return Sentinel2Reader(band_file)
                    except (ValueError, ImportError, Exception) as e:
                        errors.append(f"Sentinel-2 SAFE: {e}")

        # 1. SAR first — NITF files may be SICD (complex SAR), not generic NITF
        try:
            from grdl.IO import open_sar
            reader = open_sar(path)
            _log.info("open_any: opened via open_sar → %s", type(reader).__name__)
            return reader
        except (ValueError, ImportError, Exception) as e:
            _log.debug("open_any: open_sar failed: %s", e)
            errors.append(f"SAR: {e}")

        # 2. Generic formats (GeoTIFF, NITF, HDF5, JP2)
        try:
            from grdl.IO import open_image
            reader = open_image(path)
            _log.info("open_any: opened via open_image → %s", type(reader).__name__)
            return reader
        except (ValueError, ImportError, Exception) as e:
            _log.debug("open_any: open_image failed: %s", e)
            errors.append(f"Image: {e}")

        # 3. EO-specific (Sentinel-2, etc.)
        try:
            from grdl.IO import open_eo
            reader = open_eo(path)
            _log.info("open_any: opened via open_eo → %s", type(reader).__name__)
            return reader
        except (ValueError, ImportError, Exception) as e:
            _log.debug("open_any: open_eo failed: %s", e)
            errors.append(f"EO: {e}")

        # 4. IR
        try:
            from grdl.IO import open_ir
            reader = open_ir(path)
            _log.info("open_any: opened via open_ir → %s", type(reader).__name__)
            return reader
        except (ValueError, ImportError, Exception) as e:
            _log.debug("open_any: open_ir failed: %s", e)
            errors.append(f"IR: {e}")

        # 5. Multispectral
        try:
            from grdl.IO import open_multispectral
            reader = open_multispectral(path)
            _log.info(
                "open_any: opened via open_multispectral → %s",
                type(reader).__name__,
            )
            return reader
        except (ValueError, ImportError, Exception) as e:
            _log.debug("open_any: open_multispectral failed: %s", e)
            errors.append(f"MSI: {e}")

    _log.error("open_any: all openers failed for %s", filepath)
    raise ValueError(
        f"Could not open {filepath}. Tried all grdl openers:\n"
        + "\n".join(f"  - {e}" for e in errors)
    )


def create_geolocation(reader: Any) -> Optional[Any]:
    """Create the appropriate Geolocation from a reader type.

    Dispatches by reader class to the matching grdl Geolocation
    factory.  Returns None if geolocation cannot be determined
    (the viewer will operate in pixel-only mode).

    Parameters
    ----------
    reader : ImageReader
        grdl reader instance.

    Returns
    -------
    Optional[Geolocation]
        Geolocation instance, or None.
    """
    _log.debug("create_geolocation: reader type = %s", type(reader).__name__)

    # Lazy imports to avoid pulling in optional dependencies at module level
    try:
        from grdl.IO.sar.sicd import SICDReader
        if isinstance(reader, SICDReader):
            from grdl.geolocation import SICDGeolocation
            geo = SICDGeolocation.from_reader(reader)
            _log.info("create_geolocation: SICDGeolocation created")
            return geo
    except ImportError:
        pass

    try:
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader
        if isinstance(reader, Sentinel1SLCReader):
            from grdl.geolocation import Sentinel1SLCGeolocation
            geo = Sentinel1SLCGeolocation.from_reader(reader)
            _log.info("create_geolocation: Sentinel1SLCGeolocation created")
            return geo
    except ImportError:
        pass

    try:
        from grdl.IO.sar.biomass import BIOMASSL1Reader
        if isinstance(reader, BIOMASSL1Reader):
            gcps = reader.metadata.get('gcps')
            if gcps:
                from grdl.geolocation import GCPGeolocation
                geo = GCPGeolocation.from_dict(
                    {'gcps': gcps, 'crs': reader.metadata.get('crs', 'WGS84')},
                    reader.metadata,
                )
                _log.info("create_geolocation: GCPGeolocation created (BIOMASS)")
                return geo
    except ImportError:
        pass

    try:
        from grdl.IO.geotiff import GeoTIFFReader
        if isinstance(reader, GeoTIFFReader):
            transform = reader.metadata.get('transform')
            crs = reader.metadata.get('crs')
            if transform and crs:
                from grdl.geolocation import AffineGeolocation
                geo = AffineGeolocation.from_reader(reader)
                _log.info("create_geolocation: AffineGeolocation created (GeoTIFF)")
                return geo
    except ImportError:
        pass

    try:
        from grdl.IO.nitf import NITFReader
        if isinstance(reader, NITFReader):
            transform = reader.metadata.get('transform')
            crs = reader.metadata.get('crs')
            if transform and crs:
                from grdl.geolocation import AffineGeolocation
                geo = AffineGeolocation.from_reader(reader)
                _log.info("create_geolocation: AffineGeolocation created (NITF)")
                return geo
    except ImportError:
        pass

    _log.info(
        "create_geolocation: no geolocation for %s (pixel-only mode)",
        type(reader).__name__,
    )
    return None


# ---------------------------------------------------------------------------
# GeoImageViewer widget
# ---------------------------------------------------------------------------

if _QT_AVAILABLE:

    class GeoImageViewer(QWidget):
        """Single-pane geospatial image viewer.

        Composes a ``TiledImageCanvas`` with ``CoordinateBar`` and
        ``VectorOverlayLayer`` into a self-contained viewer pane.
        Supports opening files via grdl readers, accepting pre-loaded
        arrays, and overlaying GeoJSON vector data.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget.

        Signals
        -------
        band_info_changed(list)
            Emitted when band info changes (e.g., after opening a file).
            Payload is a ``List[BandInfo]``.
        """

        band_info_changed = Signal(list)

        def __init__(self, parent: Optional[Any] = None) -> None:
            super().__init__(parent)

            self._reader: Optional[Any] = None
            self._geolocation: Optional[Any] = None
            self._band_info: list = []
            self._metadata: Optional[Any] = None

            # Core widgets
            self._canvas = TiledImageCanvas(self)
            self._coord_bar = CoordinateBar(self)
            self._coord_bar.connect_canvas(self._canvas)
            self._colorbar = ColorBarWidget(self)

            # Update colorbar when display settings change
            self._canvas.display_settings_changed.connect(
                self._colorbar.update_from_settings,
            )

            # Vector overlay (operates on canvas scene)
            self._vector_overlay = VectorOverlayLayer(self._canvas._scene)

            # Layout
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            layout.addWidget(self._canvas, 1)
            layout.addWidget(self._colorbar, 0)
            layout.addWidget(self._coord_bar, 0)

        # --- Public properties ---

        @property
        def canvas(self) -> TiledImageCanvas:
            """The underlying image canvas."""
            return self._canvas

        @property
        def coord_bar(self) -> CoordinateBar:
            """The coordinate bar widget."""
            return self._coord_bar

        @property
        def display_settings(self) -> DisplaySettings:
            """Current display settings (delegates to canvas)."""
            return self._canvas.display_settings

        @display_settings.setter
        def display_settings(self, settings: DisplaySettings) -> None:
            self._canvas.set_display_settings(settings)

        @property
        def geolocation(self) -> Optional[Any]:
            """Current geolocation model, or None."""
            return self._geolocation

        @property
        def metadata(self) -> Optional[Any]:
            """Current image metadata, or None."""
            return self._metadata

        @property
        def band_info(self) -> list:
            """Current band info list (List[BandInfo])."""
            return self._band_info

        @property
        def colorbar(self) -> ColorBarWidget:
            """The colorbar widget."""
            return self._colorbar

        @property
        def vector_overlay(self) -> VectorOverlayLayer:
            """The vector overlay layer."""
            return self._vector_overlay

        # --- Loading ---

        def open_file(self, filepath: str) -> None:
            """Open an image file with auto-detection.

            Uses ``open_any()`` to try all grdl openers, then creates
            the appropriate geolocation model.

            Parameters
            ----------
            filepath : str
                Path to the image file.

            Raises
            ------
            ValueError
                If the file cannot be opened.
            """
            _log.info("GeoImageViewer.open_file: %s", filepath)
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
            try:
                reader = open_any(filepath)
                geo = create_geolocation(reader)
                self.open_reader(reader, geolocation=geo)
            finally:
                QApplication.restoreOverrideCursor()

        def open_reader(
            self,
            reader: Any,
            geolocation: Optional[Any] = None,
        ) -> None:
            """Open an image from a grdl ImageReader.

            Parameters
            ----------
            reader : ImageReader
                grdl reader instance.
            geolocation : Geolocation, optional
                Geolocation model.  If None, the viewer operates in
                pixel-only mode (no lat/lon display, no geographic
                vector transforms).
            """
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
            try:
                # Close previous reader
                if self._reader is not None:
                    try:
                        self._reader.close()
                    except Exception:
                        pass

                self._reader = reader
                self._geolocation = geolocation
                self._metadata = getattr(reader, 'metadata', None)

                shape = reader.get_shape()
                dtype = getattr(reader, 'get_dtype', lambda: 'unknown')()
                _log.info(
                    "open_reader: %s, shape=%s, dtype=%s, geo=%s",
                    type(reader).__name__, shape, dtype,
                    type(geolocation).__name__ if geolocation else "None",
                )

                # Auto-detect SICD and apply 2-98% contrast stretch
                self._apply_auto_settings(reader)

                # Update coordinate bar
                self._coord_bar.set_geolocation(geolocation)

                # Update vector overlay geolocation
                self._vector_overlay.set_geolocation(geolocation)

                # Extract and emit band info
                self._band_info = get_band_info(reader)
                _log.debug(
                    "open_reader: band_info = %s",
                    [(b.index, b.name) for b in self._band_info],
                )
                self.band_info_changed.emit(self._band_info)

                # Load into canvas
                self._canvas.set_reader(reader)

                # Fit image in view on first load
                self._canvas.fit_in_view()
            finally:
                QApplication.restoreOverrideCursor()

        def set_array(
            self,
            arr: np.ndarray,
            geolocation: Optional[Any] = None,
        ) -> None:
            """Display a pre-loaded numpy array.

            Parameters
            ----------
            arr : np.ndarray
                Image data (2D, 3D, or complex).
            geolocation : Geolocation, optional
                Geolocation model for coordinate display.
            """
            if self._reader is not None:
                try:
                    self._reader.close()
                except Exception:
                    pass
                self._reader = None

            self._geolocation = geolocation
            self._metadata = None
            self._coord_bar.set_geolocation(geolocation)
            self._vector_overlay.set_geolocation(geolocation)

            # Emit band info for the array
            if arr.ndim == 3:
                # Channels-first: (C, H, W)
                num_bands = arr.shape[0]
                self._band_info = [
                    BandInfo(i, f"Band {i}", "") for i in range(num_bands)
                ]
            else:
                self._band_info = [BandInfo(0, "Band 0", "")]
            self.band_info_changed.emit(self._band_info)

            self._canvas.set_array(arr)

            # Fit image in view on load
            self._canvas.fit_in_view()

        # --- Auto settings ---

        def _apply_auto_settings(self, reader: Any) -> None:
            """Apply sensible default display settings based on reader type.

            For SAR/complex data:
            - Applies a 2-98% percentile contrast stretch (standard for SAR).
            - For multi-band data, selects band 0 to ensure single-band
              display.  Without this, multi-band SAR data is incorrectly
              displayed as false-color RGB, which disables remap functions
              and colormap application.
            """
            from dataclasses import replace

            is_sar = False
            try:
                from grdl.IO.sar.sicd import SICDReader
                if isinstance(reader, SICDReader):
                    is_sar = True
            except ImportError:
                pass
            if not is_sar:
                try:
                    from grdl.IO.sar.biomass import BIOMASSL1Reader
                    if isinstance(reader, BIOMASSL1Reader):
                        is_sar = True
                except ImportError:
                    pass
            if not is_sar:
                try:
                    from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader
                    if isinstance(reader, Sentinel1SLCReader):
                        is_sar = True
                except ImportError:
                    pass
            if not is_sar:
                try:
                    from grdl.IO.sar.sidd import SIDDReader
                    if isinstance(reader, SIDDReader):
                        is_sar = True
                except ImportError:
                    pass
            if not is_sar:
                try:
                    dtype = reader.get_dtype()
                    if np.issubdtype(dtype, np.complexfloating):
                        is_sar = True
                except Exception:
                    pass

            if not is_sar:
                _log.debug("_apply_auto_settings: not SAR, skipping")
                return

            _log.info("_apply_auto_settings: SAR detected (%s)", type(reader).__name__)
            settings = self._canvas.display_settings

            # 2-98% contrast stretch for SAR
            settings = replace(
                settings,
                percentile_low=2.0,
                percentile_high=98.0,
            )

            # Multi-band SAR: select band 0 to avoid false-color RGB
            # display.  SAR polarization bands (HH, HV, etc.) are not
            # RGB channels — displaying them as RGB disables remap and
            # colormap, and produces misleading colors.
            try:
                shape = reader.get_shape()
                if len(shape) >= 3 and shape[2] > 1:
                    # get_shape returns (rows, cols, bands)
                    settings = replace(settings, band_index=0)
                    _log.info(
                        "_apply_auto_settings: multi-band SAR, selected band 0"
                    )
                elif len(shape) == 2:
                    pass  # Single band, no override needed
            except Exception:
                pass

            # Multi-pol SAR (Sentinel-1, TerraSAR-X): each reader
            # only loads one polarization, but band_info lists all
            # available pols.  Set band_index to the loaded pol's
            # position so the combo shows the correct selection.
            try:
                all_pols = reader.get_available_polarizations()
                if len(all_pols) > 1:
                    current_pol = None
                    # TerraSAR-X
                    current_pol = getattr(
                        reader, '_requested_polarization', None,
                    )
                    if current_pol is None:
                        # Sentinel-1
                        meta = getattr(reader, 'metadata', None)
                        if meta is not None:
                            si = (
                                meta.get('swath_info')
                                if hasattr(meta, 'get')
                                else getattr(meta, 'swath_info', None)
                            )
                            if si:
                                current_pol = getattr(
                                    si, 'polarization', None,
                                )
                    if current_pol and current_pol in all_pols:
                        pol_index = all_pols.index(current_pol)
                        settings = replace(settings, band_index=pol_index)
                        _log.info(
                            "_apply_auto_settings: multi-pol SAR, "
                            "selected %s at index %d",
                            current_pol, pol_index,
                        )
            except (AttributeError, Exception):
                pass

            _log.debug("_apply_auto_settings: settings = %s", settings)

            # Assign directly to avoid triggering a re-render on the
            # stale tile cache.  The old reader has already been closed
            # (in open_reader above) but canvas.set_reader() hasn't been
            # called yet, so set_display_settings() would try to refresh
            # tiles using the closed reader — causing partial loads or
            # blank displays.
            self._canvas._settings = settings  # type: ignore[attr-defined]

        # --- Vector overlays ---

        def load_vector(self, filepath: str) -> None:
            """Load a GeoJSON vector overlay.

            Parameters
            ----------
            filepath : str
                Path to a GeoJSON file.
            """
            self._vector_overlay.load_geojson(filepath)

        def clear_vectors(self) -> None:
            """Remove all vector overlay features."""
            self._vector_overlay.clear()

        # --- Export ---

        def export_view(self, filepath: str) -> None:
            """Save the current viewport as an image file.

            Parameters
            ----------
            filepath : str
                Output path.  Format determined by extension
                (e.g., .png, .jpg, .bmp).

            Raises
            ------
            RuntimeError
                If the save operation fails.
            """
            import os

            pixmap = self._canvas.grab()

            # Determine format from extension for reliable saving
            ext = os.path.splitext(filepath)[1].lower()
            fmt_map = {
                '.png': 'PNG',
                '.jpg': 'JPEG',
                '.jpeg': 'JPEG',
                '.bmp': 'BMP',
            }
            fmt = fmt_map.get(ext)

            if fmt:
                ok = pixmap.save(filepath, fmt)
            else:
                ok = pixmap.save(filepath)

            if not ok:
                raise RuntimeError(
                    f"Failed to save image (format={fmt or 'auto'})."
                )

else:

    class GeoImageViewer:  # type: ignore[no-redef]
        """Stub when Qt is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("Qt is required for GeoImageViewer")
