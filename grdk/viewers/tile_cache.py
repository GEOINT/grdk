# -*- coding: utf-8 -*-
"""
TileCache - Level-of-detail tile pyramid for lazy rendering of large images.

Manages a multi-level tile pyramid for images too large to fit in memory.
Tiles are loaded asynchronously via grdl ImageReader.read_chip() and cached
with LRU eviction.  Raw numpy tiles are cached pre-display-pipeline so
DisplaySettings changes re-render without re-reading from disk.

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
2026-02-17
"""

# Standard library
import math
from collections import OrderedDict
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

# Third-party
import numpy as np

try:
    from PyQt6.QtCore import QMutex, QObject, QRunnable, QThreadPool, pyqtSignal as Signal
    from PyQt6.QtGui import QPixmap

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False

from grdk.viewers.image_canvas import DisplaySettings, array_to_qimage

# ---------------------------------------------------------------------------
# Tile pyramid size threshold
# ---------------------------------------------------------------------------

#: Images with total pixels at or below this threshold load fully into memory
#: rather than using tiled rendering.  4096 * 4096 = ~16M pixels.
TILE_THRESHOLD = 4096 * 4096


class TileKey(NamedTuple):
    """Identifier for a single tile in the LOD pyramid."""

    level: int
    tile_row: int
    tile_col: int


def needs_tiling(rows: int, cols: int) -> bool:
    """Return True if an image is large enough to require tiled rendering.

    Parameters
    ----------
    rows : int
        Image height in pixels.
    cols : int
        Image width in pixels.

    Returns
    -------
    bool
    """
    return rows * cols > TILE_THRESHOLD


def compute_num_levels(rows: int, cols: int, tile_size: int = 512) -> int:
    """Compute the number of LOD levels for a tile pyramid.

    Level 0 is full resolution.  Each successive level halves the
    resolution (2x downsample).  Levels continue until the entire
    image fits in a single tile.

    Parameters
    ----------
    rows : int
        Image height in pixels.
    cols : int
        Image width in pixels.
    tile_size : int
        Tile edge length in pixels.

    Returns
    -------
    int
        Number of pyramid levels (always >= 1).
    """
    max_dim = max(rows, cols)
    if max_dim <= tile_size:
        return 1
    return max(1, math.ceil(math.log2(max_dim / tile_size)) + 1)


# ---------------------------------------------------------------------------
# Async tile loading worker
# ---------------------------------------------------------------------------

if _QT_AVAILABLE:

    class _TileLoadWorker(QRunnable):
        """Load a single tile from an ImageReader in a thread pool.

        The worker reads a chip from the reader, optionally decimates it
        for lower LOD levels, and stores the raw numpy result.  The
        parent TileCache is notified via a signal proxy.
        """

        def __init__(
            self,
            key: TileKey,
            reader: Any,
            row_start: int,
            row_end: int,
            col_start: int,
            col_end: int,
            factor: int,
            mutex: QMutex,
            callback: Any,
        ) -> None:
            super().__init__()
            self.key = key
            self.reader = reader
            self.row_start = row_start
            self.row_end = row_end
            self.col_start = col_start
            self.col_end = col_end
            self.factor = factor
            self.mutex = mutex
            self.callback = callback
            self.setAutoDelete(True)

        def run(self) -> None:
            """Execute the tile load in a worker thread."""
            try:
                self.mutex.lock()
                try:
                    chip = self.reader.read_chip(
                        self.row_start, self.row_end,
                        self.col_start, self.col_end,
                    )
                finally:
                    self.mutex.unlock()

                # Decimate for lower LOD levels
                # Data convention: (H, W) or (C, H, W) — channels-first
                if self.factor > 1:
                    if chip.ndim == 2:
                        chip = chip[:: self.factor, :: self.factor]
                    else:
                        chip = chip[:, :: self.factor, :: self.factor]

                self.callback(self.key, chip)
            except Exception:
                # Tile load failure is non-fatal; tile stays missing
                pass

    # -------------------------------------------------------------------
    # Signal proxy (QRunnable cannot emit signals directly)
    # -------------------------------------------------------------------

    class _SignalProxy(QObject):
        """Proxy to emit tile_ready from worker threads to the main thread."""

        tile_ready = Signal(int, int, int)

    # -------------------------------------------------------------------
    # TileCache
    # -------------------------------------------------------------------

    class TileCache(QObject):
        """LOD tile pyramid with async loading and LRU eviction.

        Manages a multi-resolution tile pyramid for a single
        ``ImageReader``.  Tiles are loaded asynchronously via
        ``QThreadPool`` and cached as raw numpy arrays.  When display
        settings change, cached raw tiles are re-rendered to QPixmaps
        without re-reading from disk.

        Global image statistics are sampled once from a low-resolution
        overview so that all tiles share the same window/level, preventing
        visible seams at tile boundaries.

        Parameters
        ----------
        reader : ImageReader
            grdl reader providing ``read_chip()`` and ``get_shape()``.
        tile_size : int
            Tile edge length in pixels.  Default 512.
        max_memory_mb : int
            Approximate memory budget for raw tile cache in megabytes.
            Default 512.
        parent : QObject, optional
            Qt parent.

        Signals
        -------
        tile_ready(int, int, int)
            Emitted when a tile is loaded and ready.  Arguments are
            (level, tile_row, tile_col).
        """

        tile_ready = Signal(int, int, int)

        def __init__(
            self,
            reader: Any,
            tile_size: int = 512,
            max_memory_mb: int = 512,
            parent: Optional[QObject] = None,
        ) -> None:
            super().__init__(parent)

            self._reader = reader
            self._tile_size = tile_size
            self._max_bytes = max_memory_mb * 1024 * 1024

            shape = reader.get_shape()
            self._rows = shape[0]
            self._cols = shape[1]
            self._num_levels = compute_num_levels(self._rows, self._cols, tile_size)

            self._settings = DisplaySettings()

            # Global image statistics for consistent window/level across tiles
            self._global_min: Optional[float] = None
            self._global_max: Optional[float] = None
            self._sample_overview()

            # LRU cache: TileKey -> raw np.ndarray
            self._raw_cache: OrderedDict[TileKey, np.ndarray] = OrderedDict()
            self._cache_bytes = 0

            # Rendered pixmap cache: TileKey -> QPixmap
            self._pixmap_cache: Dict[TileKey, QPixmap] = {}

            # Track in-flight tile requests to avoid duplicates
            self._pending: set = set()

            # Thread safety for reader access
            self._mutex = QMutex()

            # Signal proxy for cross-thread notification
            self._proxy = _SignalProxy()
            self._proxy.tile_ready.connect(self.tile_ready)

            # Thread pool (shared Qt pool)
            self._pool = QThreadPool.globalInstance()

        @property
        def num_levels(self) -> int:
            """Number of LOD levels in the pyramid."""
            return self._num_levels

        @property
        def image_shape(self) -> Tuple[int, int]:
            """Source image dimensions as (rows, cols)."""
            return (self._rows, self._cols)

        @property
        def tile_size(self) -> int:
            """Tile edge length in pixels."""
            return self._tile_size

        def tiles_at_level(self, level: int) -> Tuple[int, int]:
            """Return (num_tile_rows, num_tile_cols) at the given LOD level.

            Parameters
            ----------
            level : int
                Pyramid level (0 = full resolution).

            Returns
            -------
            Tuple[int, int]
                (tile_rows, tile_cols) grid dimensions.
            """
            factor = 1 << level
            effective_rows = math.ceil(self._rows / factor)
            effective_cols = math.ceil(self._cols / factor)
            return (
                math.ceil(effective_rows / self._tile_size),
                math.ceil(effective_cols / self._tile_size),
            )

        def request_visible(self, level: int, keys: List[TileKey]) -> None:
            """Request tiles to be loaded.

            Already-cached tiles are immediately available via
            ``get_pixmap()``.  Missing tiles are queued for async
            loading; ``tile_ready`` is emitted when each completes.

            Parameters
            ----------
            level : int
                Pyramid level.
            keys : List[TileKey]
                Tiles to ensure are loaded.
            """
            for key in keys:
                if key in self._raw_cache:
                    # Promote in LRU
                    self._raw_cache.move_to_end(key)
                    # Ensure pixmap is rendered with current settings
                    if key not in self._pixmap_cache:
                        self._render_pixmap(key)
                    continue

                if key in self._pending:
                    continue

                self._enqueue_load(key)

        def get_pixmap(self, key: TileKey) -> Optional[QPixmap]:
            """Return the rendered QPixmap for a tile, or None if not loaded.

            Parameters
            ----------
            key : TileKey
                Tile identifier.

            Returns
            -------
            Optional[QPixmap]
            """
            return self._pixmap_cache.get(key)

        def get_raw(self, key: TileKey) -> Optional[np.ndarray]:
            """Return the raw numpy array for a tile, or None if not loaded.

            Parameters
            ----------
            key : TileKey
                Tile identifier.

            Returns
            -------
            Optional[np.ndarray]
            """
            arr = self._raw_cache.get(key)
            if arr is not None:
                self._raw_cache.move_to_end(key)
            return arr

        def set_display_settings(self, settings: DisplaySettings) -> None:
            """Update display settings and re-render all cached pixmaps.

            Raw tile data is preserved; only the rendering pass is
            repeated.

            Parameters
            ----------
            settings : DisplaySettings
                New display parameters.
            """
            self._settings = settings
            self._pixmap_cache.clear()
            for key in self._raw_cache:
                self._render_pixmap(key)

        @property
        def has_pending(self) -> bool:
            """True if any tile loads are in flight."""
            return len(self._pending) > 0

        def clear(self) -> None:
            """Discard all cached tiles and pixmaps."""
            self._raw_cache.clear()
            self._pixmap_cache.clear()
            self._cache_bytes = 0
            self._pending.clear()

        # --- Internal ---

        def _sample_overview(self) -> None:
            """Read a low-resolution sample of the image to compute global stats.

            Reads several spatially-distributed chips (corners + center)
            to approximate global statistics without loading the full image.
            The raw sample is stored so exact percentiles can be computed
            on demand for any display settings, ensuring all tiles render
            with identical window/level.
            """
            samples = []
            chip_size = 512

            # Sample positions: center + corners + mid-edges
            positions = [
                (self._rows // 2 - chip_size // 2, self._cols // 2 - chip_size // 2),
                (0, 0),
                (0, max(0, self._cols - chip_size)),
                (max(0, self._rows - chip_size), 0),
                (max(0, self._rows - chip_size), max(0, self._cols - chip_size)),
                (self._rows // 2 - chip_size // 2, 0),
                (self._rows // 2 - chip_size // 2, max(0, self._cols - chip_size)),
                (0, self._cols // 2 - chip_size // 2),
                (max(0, self._rows - chip_size), self._cols // 2 - chip_size // 2),
            ]

            for r0, c0 in positions:
                try:
                    r0 = max(0, r0)
                    c0 = max(0, c0)
                    r1 = min(self._rows, r0 + chip_size)
                    c1 = min(self._cols, c0 + chip_size)
                    chip = self._reader.read_chip(r0, r1, c0, c1)
                    # For multi-band (C, H, W), use first band for stats
                    if chip.ndim == 3:
                        chip = chip[0]
                    if np.iscomplexobj(chip):
                        chip = np.abs(chip)
                    samples.append(chip.astype(np.float64).ravel())
                except Exception:
                    continue

            if not samples:
                return

            # Store the combined sample for on-demand percentile computation
            self._overview_sample = np.concatenate(samples)
            self._global_min = float(np.nanmin(self._overview_sample))
            self._global_max = float(np.nanmax(self._overview_sample))
            # Pre-compute the mean for remap functions that need it
            finite = self._overview_sample[np.isfinite(self._overview_sample)]
            self._global_mean = float(np.mean(finite)) if len(finite) > 0 else None
            # Cache for already-computed percentile pairs
            self._percentile_cache: Dict[Tuple[float, float], Tuple[float, float]] = {}

            # Pre-compute stats for NRL remap (99th percentile)
            self._global_p99 = float(np.nanpercentile(self._overview_sample, 99))

            # Pre-compute stats for log remap
            g_mean = self._global_mean or 1.0
            log_A = self._overview_sample.copy()
            if g_mean > 0:
                log_A = 10.0 * log_A / g_mean
            self._log_shift = max(1.0 - float(np.nanmin(log_A)), 0.0)
            log_A = log_A + self._log_shift
            log_x = 20.0 * np.log10(np.maximum(log_A, 1e-10))
            self._log_rcent = 10.0 * np.log10(
                max(float(np.nanmean(log_A ** 2)), 1e-10),
            )
            self._log_dmin = max(
                float(np.nanmin(log_x)), self._log_rcent - 25.0,
            )
            self._log_dmax = min(
                float(np.nanmax(log_x)), self._log_rcent + 25.0,
            )

        def _get_global_percentiles(self, lo: float, hi: float) -> Optional[Tuple[float, float]]:
            """Compute exact percentiles from the stored overview sample.

            Results are cached for repeated lookups with the same values.
            """
            if not hasattr(self, '_overview_sample') or self._overview_sample is None:
                return None
            key = (lo, hi)
            if key not in self._percentile_cache:
                vmin = float(np.nanpercentile(self._overview_sample, lo))
                vmax = float(np.nanpercentile(self._overview_sample, hi))
                self._percentile_cache[key] = (vmin, vmax)
            return self._percentile_cache[key]

        @property
        def global_mean(self) -> Optional[float]:
            """Global mean of the image sample (magnitude for complex data)."""
            return getattr(self, '_global_mean', None)

        def _resolve_settings(self, settings: DisplaySettings) -> DisplaySettings:
            """Resolve percentile-based settings to explicit window_min/max.

            Uses pre-computed global image statistics so all tiles share
            identical window/level values, preventing visible tile seams.
            """
            if settings.window_min is not None and settings.window_max is not None:
                return settings

            result = self._get_global_percentiles(
                settings.percentile_low, settings.percentile_high
            )
            if result is None:
                return settings

            vmin, vmax = result
            from dataclasses import replace
            return replace(settings, window_min=vmin, window_max=vmax)

        def _enqueue_load(self, key: TileKey) -> None:
            """Submit a tile load job to the thread pool."""
            self._pending.add(key)

            factor = 1 << key.level
            ts = self._tile_size

            # Source region in full-resolution pixel coordinates
            row_start = key.tile_row * ts * factor
            col_start = key.tile_col * ts * factor
            row_end = min(row_start + ts * factor, self._rows)
            col_end = min(col_start + ts * factor, self._cols)

            worker = _TileLoadWorker(
                key=key,
                reader=self._reader,
                row_start=row_start,
                row_end=row_end,
                col_start=col_start,
                col_end=col_end,
                factor=factor,
                mutex=self._mutex,
                callback=self._on_tile_loaded,
            )
            self._pool.start(worker)

        def _on_tile_loaded(self, key: TileKey, data: np.ndarray) -> None:
            """Handle a completed tile load (called from worker thread)."""
            self._pending.discard(key)

            # Evict old tiles if over budget
            tile_bytes = data.nbytes
            while self._cache_bytes + tile_bytes > self._max_bytes and self._raw_cache:
                evict_key, evict_arr = self._raw_cache.popitem(last=False)
                self._cache_bytes -= evict_arr.nbytes
                self._pixmap_cache.pop(evict_key, None)

            # Store raw tile
            self._raw_cache[key] = data
            self._cache_bytes += tile_bytes

            # Render to pixmap
            self._render_pixmap(key)

            # Notify main thread
            self._proxy.tile_ready.emit(key.level, key.tile_row, key.tile_col)

        def _render_pixmap(self, key: TileKey) -> None:
            """Render a cached raw tile to QPixmap with current settings.

            Uses globally-resolved window/level to ensure consistent
            rendering across all tiles (no visible seams).  For remap
            functions that compute per-tile statistics (e.g. data_mean),
            wraps them with the global mean so all tiles render identically.
            """
            raw = self._raw_cache.get(key)
            if raw is None:
                return
            resolved = self._resolve_settings(self._settings)

            # For remap functions, wrap with global mean to prevent
            # per-tile statistical variation causing visible seams.
            # Density-based remaps (density, brighter, darker, highcontrast,
            # pedf) internally call amplitude_to_density which computes
            # data_mean per-tile.  We replace them with a version that uses
            # the precomputed global mean.
            if resolved.remap_function is not None and self._global_mean is not None:
                resolved = self._wrap_remap_with_global_stats(resolved)

            qimg = array_to_qimage(raw, resolved)
            self._pixmap_cache[key] = QPixmap.fromImage(qimg)

        def _wrap_remap_with_global_stats(
            self, settings: DisplaySettings,
        ) -> DisplaySettings:
            """Replace remap functions with versions using global statistics.

            Density-based SAR remaps compute per-tile data_mean which causes
            visible tile seams.  This wraps them to use the precomputed
            global mean instead.
            """
            fn = settings.remap_function
            if fn is None:
                return settings

            global_mean = self._global_mean
            wrapped = None

            try:
                from grdl_sartoolbox.visualization.remap import (
                    amplitude_to_density,
                    density_remap,
                    brighter_remap,
                    darker_remap,
                    high_contrast_remap,
                    pedf_remap,
                    log_remap,
                    nrl_remap,
                )

                # Map each density-variant remap to its a2d parameters
                if fn is density_remap:
                    def wrapped(data: np.ndarray) -> np.ndarray:
                        return np.clip(amplitude_to_density(
                            data, dmin=30, mmult=40, data_mean=global_mean
                        ), 0, 255).astype(np.uint8)
                elif fn is brighter_remap:
                    def wrapped(data: np.ndarray) -> np.ndarray:
                        return np.clip(amplitude_to_density(
                            data, dmin=60, mmult=40, data_mean=global_mean
                        ), 0, 255).astype(np.uint8)
                elif fn is darker_remap:
                    def wrapped(data: np.ndarray) -> np.ndarray:
                        return np.clip(amplitude_to_density(
                            data, dmin=0, mmult=40, data_mean=global_mean
                        ), 0, 255).astype(np.uint8)
                elif fn is high_contrast_remap:
                    def wrapped(data: np.ndarray) -> np.ndarray:
                        return np.clip(amplitude_to_density(
                            data, dmin=30, mmult=4, data_mean=global_mean
                        ), 0, 255).astype(np.uint8)
                elif fn is pedf_remap:
                    def wrapped(data: np.ndarray) -> np.ndarray:
                        D = amplitude_to_density(
                            data, data_mean=global_mean,
                        )
                        D[D > 128] = 0.5 * (D[D > 128] + 128.0)
                        return np.clip(D, 0, 255).astype(np.uint8)
                elif fn is log_remap:
                    # Use fully precomputed global log stats
                    g_mean = global_mean
                    g_shift = self._log_shift
                    g_ldmin = self._log_dmin
                    g_ldmax = self._log_dmax
                    def wrapped(data: np.ndarray) -> np.ndarray:
                        A = np.abs(data).astype(np.float32)
                        if g_mean > 0:
                            A = 10.0 * A / g_mean
                        A = A + g_shift
                        x = 20.0 * np.log10(np.maximum(A, 1e-10))
                        if g_ldmax > g_ldmin:
                            out = 255.0 * (x - g_ldmin) / (g_ldmax - g_ldmin)
                        else:
                            out = np.zeros_like(x)
                        return np.clip(out, 0, 255).astype(np.uint8)
                elif fn is nrl_remap:
                    g_amin = self._global_min
                    g_amax = self._global_max
                    g_p99 = self._global_p99
                    def wrapped(data: np.ndarray) -> np.ndarray:
                        A = np.abs(data).astype(np.float32)
                        knee = 1.0 * g_p99  # a=1.0 default
                        if knee <= g_amin or g_amax <= g_amin:
                            return np.zeros(data.shape, dtype=np.uint8)
                        c = 220.0
                        log_d = np.log10(
                            max((g_amax - g_amin) / (knee - g_amin), 1e-10),
                        )
                        b = (255.0 - c) / max(log_d, 1e-10)
                        out = np.zeros(data.shape, dtype=np.float64)
                        linear_mask = A <= knee
                        out[linear_mask] = (
                            (A[linear_mask] - g_amin) * c / (knee - g_amin)
                        )
                        log_mask = ~linear_mask
                        ratio = (A[log_mask] - g_amin) / (knee - g_amin)
                        out[log_mask] = (
                            b * np.log10(np.maximum(ratio, 1e-10)) + c
                        )
                        return np.clip(out, 0, 255).astype(np.uint8)
                # linear_remap doesn't need global stats (just magnitude)
            except ImportError:
                pass

            if wrapped is not None:
                from dataclasses import replace
                return replace(settings, remap_function=wrapped)
            return settings

else:

    class TileCache:  # type: ignore[no-redef]
        """Stub when Qt is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("Qt is required for TileCache")
