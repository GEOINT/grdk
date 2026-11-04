# -*- coding: utf-8 -*-
"""
Catalog Client & Cache Manager for GRDK.

Manages user-configured catalog entries stored in ``~/.grdl/config.json``,
federates queries across multiple backends (SQLite, YAML), and provides
transparent in-memory and disk caching with automatic background staleness checks.

Shared across GRDK components (image viewer and Orange canvas admin widgets).

Author
------
Claude Code / Steven Siebert

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_log = logging.getLogger("grdk.catalog")

_CONFIG_DIR = Path.home() / ".grdl"
_CONFIG_FILE = _CONFIG_DIR / "config.json"
_CACHE_FILE = _CONFIG_DIR / ".catalog_cache.json"


@dataclass
class CatalogEntry:
    """Descriptor for a configured catalog source."""
    name: str
    path: str
    backend_type: str = "sqlite"  # 'sqlite' or 'yaml'
    enabled: bool = True
    last_mtime: float = 0.0
    item_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CatalogEntry:
        return cls(
            name=data.get("name", "Unnamed Catalog"),
            path=data.get("path", ""),
            backend_type=data.get("backend_type", "sqlite"),
            enabled=data.get("enabled", True),
            last_mtime=data.get("last_mtime", 0.0),
            item_count=data.get("item_count", 0),
        )


class CatalogPreferencesManager:
    """Reads and writes user catalog configurations in ``~/.grdl/config.json``."""

    @classmethod
    def load_config(cls) -> dict[str, Any]:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if _CONFIG_FILE.exists():
            try:
                with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                _log.warning("Could not load %s: %s", _CONFIG_FILE, e)
        return {}

    @classmethod
    def save_config(cls, data: dict[str, Any]) -> None:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            _log.error("Failed to write %s: %s", _CONFIG_FILE, e)

    @classmethod
    def get_configured_catalogs(cls) -> list[CatalogEntry]:
        config = cls.load_config()
        raw_entries = config.get("catalogs")

        # Default fallback if no catalogs configured yet
        if not raw_entries:
            from grdl_rt.catalog.resolver import resolve_catalog_path
            default_path = resolve_catalog_path()
            default_entry = CatalogEntry(
                name="Default Catalog",
                path=str(default_path),
                backend_type="sqlite",
                enabled=True,
            )
            raw_entries = [default_entry.to_dict()]
            config["catalogs"] = raw_entries
            cls.save_config(config)

        return [CatalogEntry.from_dict(d) for d in raw_entries]

    @classmethod
    def save_configured_catalogs(cls, catalogs: list[CatalogEntry]) -> None:
        config = cls.load_config()
        config["catalogs"] = [c.to_dict() for c in catalogs]
        cls.save_config(config)

    @classmethod
    def get_output_destination_mode(cls) -> str:
        """Returns 'replace' (default) or 'dual'."""
        config = cls.load_config()
        return config.get("catalog_execution_output_mode", "replace")

    @classmethod
    def set_output_destination_mode(cls, mode: str) -> None:
        config = cls.load_config()
        config["catalog_execution_output_mode"] = mode
        cls.save_config(config)


class CatalogManager:
    """High-level catalog manager with disk caching and background invalidation."""

    def __init__(self) -> None:
        self._catalogs: list[CatalogEntry] = []
        self._cached_artifacts: list[Any] = []
        self._last_sync_time: float = 0.0
        self._active_catalog: Optional[Any] = None
        self.reload_config()

    def reload_config(self) -> None:
        """Reload configuration entries from disk."""
        self._catalogs = CatalogPreferencesManager.get_configured_catalogs()

    @property
    def catalogs(self) -> list[CatalogEntry]:
        return self._catalogs

    def add_catalog(self, name: str, path: str, backend_type: str = "sqlite") -> CatalogEntry:
        entry = CatalogEntry(
            name=name.strip(),
            path=str(Path(path).expanduser().resolve()),
            backend_type=backend_type.lower(),
            enabled=True,
        )
        self._catalogs.append(entry)
        CatalogPreferencesManager.save_configured_catalogs(self._catalogs)
        self.refresh_cache(force=True)
        return entry

    def remove_catalog(self, index: int) -> None:
        if 0 <= index < len(self._catalogs):
            self._catalogs.pop(index)
            CatalogPreferencesManager.save_configured_catalogs(self._catalogs)
            self.refresh_cache(force=True)

    def toggle_catalog(self, index: int, enabled: bool) -> None:
        if 0 <= index < len(self._catalogs):
            self._catalogs[index].enabled = enabled
            CatalogPreferencesManager.save_configured_catalogs(self._catalogs)
            self.refresh_cache(force=True)

    def is_cache_stale(self) -> bool:
        """Check if any active catalog file mtime differs from cached state."""
        for cat in self._catalogs:
            if not cat.enabled:
                continue
            p = Path(cat.path)
            if not p.exists():
                continue
            try:
                mtime = p.stat().st_mtime
                if mtime != cat.last_mtime:
                    return True
            except OSError:
                pass
        return False

    def load_cached_artifacts(self) -> list[Any]:
        """Fast-load artifacts from disk cache if present and valid."""
        if not _CACHE_FILE.exists():
            return self.refresh_cache()

        try:
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            from grdl_rt.catalog.models import Artifact
            artifacts = [Artifact(**d) for d in data.get("artifacts", [])]
            self._cached_artifacts = artifacts
            self._last_sync_time = data.get("sync_time", 0.0)
            return artifacts
        except Exception as e:
            _log.warning("Cache load failed, refreshing: %s", e)
            return self.refresh_cache()

    def refresh_cache(self, force: bool = False) -> list[Any]:
        """Synchronously query all active backends, deduplicate, and persist cache."""
        from grdl_rt.catalog.database import SqliteArtifactCatalog
        from grdl_rt.catalog.yaml_catalog import YamlArtifactCatalog
        from grdl_rt.catalog.federated import FederatedArtifactCatalog
        from grdl_rt.execution.discovery import init_discovery

        backend_instances = []
        updated_catalogs = []

        for cat in self._catalogs:
            p = Path(cat.path)
            current_mtime = 0.0
            count = 0
            if p.exists():
                try:
                    current_mtime = p.stat().st_mtime
                except OSError:
                    pass

            if cat.enabled and p.exists():
                try:
                    if cat.backend_type == "yaml" or p.suffix in (".yaml", ".yml"):
                        backend = YamlArtifactCatalog(p)
                    else:
                        backend = SqliteArtifactCatalog(p)
                    backend_instances.append(backend)
                    count = len(backend.list_artifacts())
                except Exception as e:
                    _log.warning("Failed opening catalog %s: %s", cat.name, e)

            cat.last_mtime = current_mtime
            cat.item_count = count
            updated_catalogs.append(cat)

        self._catalogs = updated_catalogs
        CatalogPreferencesManager.save_configured_catalogs(self._catalogs)

        if not backend_instances:
            self._cached_artifacts = []
            self._active_catalog = None
            return []

        if len(backend_instances) == 1:
            active_catalog = backend_instances[0]
        else:
            active_catalog = FederatedArtifactCatalog(backend_instances)

        self._active_catalog = active_catalog

        # Wire into grdl-runtime's discovery system so WorkflowExecutor finds processors
        init_discovery(active_catalog)

        all_artifacts = active_catalog.list_artifacts()
        self._cached_artifacts = all_artifacts
        self._last_sync_time = time.time()

        # Write to disk cache
        try:
            cache_payload = {
                "sync_time": self._last_sync_time,
                "artifacts": [vars(a) for a in all_artifacts],
            }
            with open(_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache_payload, f, indent=2)
        except Exception as e:
            _log.warning("Failed to write catalog cache: %s", e)

        return all_artifacts

    def get_catalog(self) -> Optional[Any]:
        """Return the active (single or federated) catalog instance."""
        if self._active_catalog is None:
            self.refresh_cache()
        return self._active_catalog

    def get_artifact(self, name: str, version: Optional[str] = None) -> Optional[Any]:
        for art in self._cached_artifacts:
            if art.name == name:
                if version is None or art.version == version:
                    return art
        return None
