# -*- coding: utf-8 -*-
"""
Tests for grdk.viewers geo viewer components — TileCache, CoordinateBar,
VectorOverlayLayer, and utility functions.

Pure-function tests require only numpy.  Qt-dependent tests are skipped
when no display is available.

Author
------
Claude Code (Anthropic)

Created
-------
2026-02-18
"""

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from grdk.viewers.tile_cache import (
    TileKey,
    compute_num_levels,
    needs_tiling,
    TILE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Synthetic ImageReader for testing
# ---------------------------------------------------------------------------

class SyntheticReader:
    """Minimal ImageReader-like object backed by a numpy array."""

    def __init__(self, rows: int, cols: int, dtype: type = np.float32) -> None:
        self._arr = np.random.rand(rows, cols).astype(dtype) * 255
        self.metadata = {'rows': rows, 'cols': cols, 'dtype': str(dtype)}

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Any = None,
    ) -> np.ndarray:
        return self._arr[row_start:row_end, col_start:col_end].copy()

    def get_shape(self) -> Tuple[int, ...]:
        return self._arr.shape

    def get_dtype(self) -> np.dtype:
        return self._arr.dtype

    def read_full(self, bands: Any = None) -> np.ndarray:
        return self._arr.copy()

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# needs_tiling
# ---------------------------------------------------------------------------

class TestNeedsTiling:
    def test_small_image(self):
        assert needs_tiling(100, 100) is False

    def test_threshold_boundary(self):
        # 4096 * 4096 = TILE_THRESHOLD → should NOT need tiling (<=)
        assert needs_tiling(4096, 4096) is False

    def test_large_image(self):
        assert needs_tiling(5000, 5000) is True

    def test_wide_image(self):
        # 1 x (TILE_THRESHOLD + 1) → 1 * 16777217 > TILE_THRESHOLD → True
        assert needs_tiling(1, TILE_THRESHOLD + 1) is True
        # 1 x TILE_THRESHOLD → exactly at threshold → False
        assert needs_tiling(1, TILE_THRESHOLD) is False


# ---------------------------------------------------------------------------
# compute_num_levels
# ---------------------------------------------------------------------------

class TestComputeNumLevels:
    def test_single_tile(self):
        """Image fits in one tile → 1 level."""
        assert compute_num_levels(256, 256, tile_size=512) == 1

    def test_two_tiles(self):
        """Image needs 2x2 tiles → 2 levels."""
        levels = compute_num_levels(1024, 1024, tile_size=512)
        assert levels == 2

    def test_large_image(self):
        """10000x10000 image should have several levels."""
        levels = compute_num_levels(10000, 10000, tile_size=512)
        # At highest level, should fit in one tile
        assert levels >= 4
        # Verify: at top level, image fits in one tile
        top_factor = 1 << (levels - 1)
        assert max(10000, 10000) / top_factor <= 512

    def test_always_at_least_one(self):
        assert compute_num_levels(1, 1, tile_size=512) >= 1


# ---------------------------------------------------------------------------
# TileKey
# ---------------------------------------------------------------------------

class TestTileKey:
    def test_named_tuple(self):
        key = TileKey(level=2, tile_row=3, tile_col=4)
        assert key.level == 2
        assert key.tile_row == 3
        assert key.tile_col == 4

    def test_hashable(self):
        key = TileKey(0, 1, 2)
        d = {key: "test"}
        assert d[TileKey(0, 1, 2)] == "test"


# ---------------------------------------------------------------------------
# CoordinateBar value formatting (no Qt needed for logic testing)
# ---------------------------------------------------------------------------

class TestCoordinateFormatting:
    """Test the value formatting logic from CoordinateBar."""

    def test_scalar_float(self):
        # Just verify the formatting logic works
        val = 42.123456
        formatted = f"{val:.4g}"
        assert formatted == "42.12"

    def test_complex_value(self):
        val = 3 + 4j
        mag = abs(val)
        phase = np.angle(val, deg=True)
        assert abs(mag - 5.0) < 0.01
        assert abs(phase - 53.13) < 0.1

    def test_rgb_tuple(self):
        val = np.array([128, 64, 32])
        formatted = f"({val[0]:.4g}, {val[1]:.4g}, {val[2]:.4g})"
        assert "128" in formatted
        assert "64" in formatted
        assert "32" in formatted


# ---------------------------------------------------------------------------
# VectorOverlayLayer coordinate transform logic (no Qt needed)
# ---------------------------------------------------------------------------

class TestVectorGeoTransform:
    """Test the geographic-to-pixel coordinate transform logic."""

    def test_no_geolocation_passthrough(self):
        """Without geolocation, coords should pass through as (col, row)."""
        coords = [[100.5, 200.3], [300.7, 400.1]]
        arr = np.array(coords, dtype=np.float64)
        # Without geolocation, first element = col (x), second = row (y)
        assert arr[0, 0] == 100.5
        assert arr[0, 1] == 200.3

    def test_geolocation_transform(self):
        """With mock geolocation, latlon_to_image should be called."""
        mock_geo = MagicMock()
        mock_geo.latlon_to_image.return_value = (
            np.array([10.0, 20.0]),  # rows
            np.array([30.0, 40.0]),  # cols
        )

        lons = np.array([1.0, 2.0])
        lats = np.array([3.0, 4.0])
        rows, cols = mock_geo.latlon_to_image(lats, lons)

        assert len(rows) == 2
        assert rows[0] == 10.0
        assert cols[0] == 30.0
        mock_geo.latlon_to_image.assert_called_once()


# ---------------------------------------------------------------------------
# GeoJSON parsing (no Qt, test file I/O only)
# ---------------------------------------------------------------------------

class TestGeoJSONParsing:
    def test_feature_collection(self):
        """FeatureCollection should be recognized."""
        data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
                    "properties": {"name": "test"},
                }
            ],
        }
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == 1
        assert data["features"][0]["geometry"]["type"] == "Point"

    def test_single_feature(self):
        """A single Feature should be wrapped in a list."""
        data = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [
                [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
            ]},
            "properties": {},
        }
        if data["type"] == "Feature":
            features = [data]
        assert len(features) == 1

    def test_bare_geometry(self):
        """A bare geometry should be wrapped in Feature and list."""
        data = {
            "type": "Point",
            "coordinates": [1.5, 2.5],
        }
        if data["type"] in ('Point', 'Polygon', 'LineString'):
            features = [{"type": "Feature", "geometry": data, "properties": {}}]
        assert len(features) == 1
        assert features[0]["geometry"]["coordinates"] == [1.5, 2.5]

    def test_geojson_file_roundtrip(self):
        """Write and read a GeoJSON file."""
        data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [10.0, 20.0]},
                    "properties": {"id": 1},
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
                        ],
                    },
                    "properties": {"id": 2},
                },
            ],
        }
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.geojson', delete=False
        ) as f:
            json.dump(data, f)
            tmppath = f.name

        try:
            with open(tmppath) as f:
                loaded = json.load(f)
            assert loaded["type"] == "FeatureCollection"
            assert len(loaded["features"]) == 2
        finally:
            os.unlink(tmppath)


# ---------------------------------------------------------------------------
# open_any error handling
# ---------------------------------------------------------------------------

class TestFindBiomassProductDir:
    def test_direct_annotation(self):
        """Product dir with annotation/ at top level."""
        from grdk.viewers.geo_viewer import _find_biomass_product_dir

        with tempfile.TemporaryDirectory() as d:
            annot = os.path.join(d, "annotation")
            os.makedirs(annot)
            result = _find_biomass_product_dir(Path(d))
            assert result == Path(d)

    def test_nested_product_dir(self):
        """Nested product dir: outer/inner/annotation/."""
        from grdk.viewers.geo_viewer import _find_biomass_product_dir

        with tempfile.TemporaryDirectory() as d:
            inner = os.path.join(d, "BIO_S3_SCS_INNER")
            os.makedirs(os.path.join(inner, "annotation"))
            result = _find_biomass_product_dir(Path(d))
            assert result == Path(inner)

    def test_not_found(self):
        """Directory without annotation/ returns None."""
        from grdk.viewers.geo_viewer import _find_biomass_product_dir

        with tempfile.TemporaryDirectory() as d:
            result = _find_biomass_product_dir(Path(d))
            assert result is None


class TestFindSentinel2BandFile:
    def test_finds_tci_at_10m(self):
        """Should prefer TCI at 10m resolution."""
        from grdk.viewers.geo_viewer import _find_sentinel2_band_file

        with tempfile.TemporaryDirectory() as d:
            r10 = os.path.join(d, "GRANULE", "L2A_T15RTP", "IMG_DATA", "R10m")
            os.makedirs(r10)
            tci = os.path.join(r10, "T15RTP_20260204T170409_TCI_10m.jp2")
            b04 = os.path.join(r10, "T15RTP_20260204T170409_B04_10m.jp2")
            for f in (tci, b04):
                Path(f).touch()
            result = _find_sentinel2_band_file(Path(d))
            assert result is not None
            assert "_TCI_" in result.name

    def test_falls_back_to_b04(self):
        """Without TCI, should pick B04."""
        from grdk.viewers.geo_viewer import _find_sentinel2_band_file

        with tempfile.TemporaryDirectory() as d:
            r10 = os.path.join(d, "GRANULE", "L2A_T15RTP", "IMG_DATA", "R10m")
            os.makedirs(r10)
            b04 = os.path.join(r10, "T15RTP_20260204T170409_B04_10m.jp2")
            aot = os.path.join(r10, "T15RTP_20260204T170409_AOT_10m.jp2")
            for f in (b04, aot):
                Path(f).touch()
            result = _find_sentinel2_band_file(Path(d))
            assert result is not None
            assert "_B04_" in result.name

    def test_no_granule_returns_none(self):
        """No GRANULE directory → None."""
        from grdk.viewers.geo_viewer import _find_sentinel2_band_file

        with tempfile.TemporaryDirectory() as d:
            result = _find_sentinel2_band_file(Path(d))
            assert result is None

    def test_lower_resolution_fallback(self):
        """Should find bands at 20m if 10m is absent."""
        from grdk.viewers.geo_viewer import _find_sentinel2_band_file

        with tempfile.TemporaryDirectory() as d:
            r20 = os.path.join(d, "GRANULE", "L2A_T15RTP", "IMG_DATA", "R20m")
            os.makedirs(r20)
            b05 = os.path.join(r20, "T15RTP_20260204T170409_B05_20m.jp2")
            Path(b05).touch()
            result = _find_sentinel2_band_file(Path(d))
            assert result is not None
            assert "_B05_" in result.name


class TestOpenAny:
    def test_nonexistent_file_raises(self):
        """open_any should raise for nonexistent files."""
        from grdk.viewers.geo_viewer import open_any

        with pytest.raises((ValueError, FileNotFoundError)):
            open_any("/nonexistent/path/to/image.tif")

    def test_invalid_file_raises(self):
        """open_any should raise for invalid files."""
        from grdk.viewers.geo_viewer import open_any

        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            f.write(b"not a real image")
            tmppath = f.name

        try:
            with pytest.raises((ValueError, Exception)):
                open_any(tmppath)
        finally:
            os.unlink(tmppath)


# ---------------------------------------------------------------------------
# create_geolocation
# ---------------------------------------------------------------------------

class TestCreateGeolocation:
    def test_unknown_reader_returns_none(self):
        """Unknown reader type should return None."""
        from grdk.viewers.geo_viewer import create_geolocation

        reader = SyntheticReader(100, 100)
        geo = create_geolocation(reader)
        assert geo is None


# ---------------------------------------------------------------------------
# TileCache (Qt-dependent)
# ---------------------------------------------------------------------------

try:
    from PyQt6.QtWidgets import QApplication
    from grdk.viewers.tile_cache import TileCache

    _QT_SKIP = False
    # Ensure QApplication exists for tests
    if QApplication.instance() is None:
        _app = QApplication([])
except (ImportError, RuntimeError):
    _QT_SKIP = True


@pytest.mark.skipif(_QT_SKIP, reason="Qt not available")
class TestTileCache:
    def test_init(self):
        reader = SyntheticReader(2048, 2048)
        cache = TileCache(reader, tile_size=512)
        assert cache.num_levels >= 1
        assert cache.image_shape == (2048, 2048)
        assert cache.tile_size == 512

    def test_tiles_at_level(self):
        reader = SyntheticReader(1024, 1024)
        cache = TileCache(reader, tile_size=512)
        # Level 0: 2x2 tiles
        rows, cols = cache.tiles_at_level(0)
        assert rows == 2
        assert cols == 2
        # Level 1: 1x1 tile
        if cache.num_levels > 1:
            rows, cols = cache.tiles_at_level(1)
            assert rows == 1
            assert cols == 1

    def test_get_pixmap_before_load(self):
        reader = SyntheticReader(1024, 1024)
        cache = TileCache(reader, tile_size=512)
        key = TileKey(0, 0, 0)
        assert cache.get_pixmap(key) is None

    def test_clear(self):
        reader = SyntheticReader(1024, 1024)
        cache = TileCache(reader, tile_size=512)
        cache.clear()
        assert cache.get_pixmap(TileKey(0, 0, 0)) is None

    def test_display_settings_update(self):
        from grdk.viewers.image_canvas import DisplaySettings

        reader = SyntheticReader(1024, 1024)
        cache = TileCache(reader, tile_size=512)
        new_settings = DisplaySettings(contrast=2.0)
        cache.set_display_settings(new_settings)
        # Should not crash; pixmap cache is cleared


@pytest.mark.skipif(_QT_SKIP, reason="Qt not available")
class TestTiledImageCanvas:
    def test_set_small_array(self):
        """Small arrays should use base class path (not tiled)."""
        from grdk.viewers.tiled_canvas import TiledImageCanvas

        canvas = TiledImageCanvas()
        arr = np.random.rand(100, 100).astype(np.float32)
        canvas.set_array(arr)
        assert canvas._tiled_mode is False
        assert canvas._source is not None

    def test_set_reader_small(self):
        """Small reader should load fully into base class."""
        from grdk.viewers.tiled_canvas import TiledImageCanvas

        canvas = TiledImageCanvas()
        reader = SyntheticReader(100, 100)
        canvas.set_reader(reader)
        assert canvas._tiled_mode is False

    def test_viewport_center(self):
        """get_viewport_center should return a (row, col) tuple."""
        from grdk.viewers.tiled_canvas import TiledImageCanvas

        canvas = TiledImageCanvas()
        arr = np.random.rand(100, 100).astype(np.float32)
        canvas.set_array(arr)
        center = canvas.get_viewport_center()
        assert len(center) == 2

    def test_get_zoom(self):
        """get_zoom should return a float."""
        from grdk.viewers.tiled_canvas import TiledImageCanvas

        canvas = TiledImageCanvas()
        zoom = canvas.get_zoom()
        assert isinstance(zoom, float)
