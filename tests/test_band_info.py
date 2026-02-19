# -*- coding: utf-8 -*-
"""Tests for grdk.viewers.band_info — band info extraction from readers."""

import numpy as np
import pytest

from grdk.viewers.band_info import BandInfo, get_band_info


class _MockReader:
    """Minimal mock reader for testing fallback path."""

    def __init__(self, bands=1):
        self.metadata = _DictMeta(bands=bands)

    def get_shape(self):
        return (100, 100) if self.metadata.get('bands') == 1 else (100, 100, self.metadata.get('bands'))


class _DictMeta:
    """Minimal metadata with dict-like access."""

    def __init__(self, **kwargs):
        self._data = kwargs

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return self._data.get(name)


class TestBandInfo:
    def test_dataclass(self):
        info = BandInfo(index=0, name="HH", description="Polarization HH")
        assert info.index == 0
        assert info.name == "HH"
        assert info.description == "Polarization HH"

    def test_default_description(self):
        info = BandInfo(index=1, name="Band 1")
        assert info.description == ""


class TestGetBandInfo:
    def test_single_band_fallback(self):
        reader = _MockReader(bands=1)
        result = get_band_info(reader)
        assert len(result) == 1
        assert result[0].index == 0
        assert result[0].name == "Band 0"

    def test_multi_band_fallback(self):
        reader = _MockReader(bands=4)
        result = get_band_info(reader)
        assert len(result) == 4
        for i, info in enumerate(result):
            assert info.index == i
            assert info.name == f"Band {i}"

    def test_returns_list_of_bandinfo(self):
        reader = _MockReader(bands=3)
        result = get_band_info(reader)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, BandInfo)
