# Changelog

All notable changes to GRDK are documented in this file.

## [Unreleased]

### Added

#### Architectural Refactor — Polarization, Reader Dispatch & Pauli Idempotency
- `_pol_utils._reader_polarization()` now handles all sensor families in one place:
  Sentinel-1 SLC (`swath_info.polarization`), TerraSAR-X (`_requested_polarization`
  fallback), NISAR / CPHD (`metadata.polarization`), generic multi-band
  (`channel_metadata`), and BIOMASS (`metadata.polarizations` list).
- `create_geolocation()` in `geo_viewer.py` replaced 6-branch isinstance waterfall
  with a declarative `_GEO_REGISTRY` list; new sensor types are registered in one
  line without touching the dispatch logic.
- `_decomp_state` per-pane dict added to `ViewerMainWindow`: tracks whether a pane
  is currently displaying a Pauli or H/Alpha decomposition result (`None | 'pauli' |
  'halpha'`).  State is reset on `open_file`, `open_reader`, and `set_array`.
- Pauli idempotency guard: `_on_pauli_decomp()` checks `_decomp_state` before
  launching a worker; if the pane already shows a decomposition it prompts the user
  to confirm a refresh rather than silently re-running.
- `_on_pauli_finished()` now displays before cleanup so `_update_tools_state()`
  evaluates the post-display state (reader = None) and correctly disables the
  Pauli action.
- `_gather_rgb_bands()` now handles NISAR opened with `polarizations='all'` (CYX
  multi-pol) via `channel_pol_map()` as the first dispatch path; the fallback BIOMASS
  `polarizations` attribute is only consulted when `channel_metadata` is absent.
- RGB Combine dialog defaults to Pauli-like channel assignment for quad-pol data
  (R = HH−VV, G = HV+VH, B = HH+VV); prior behaviour was to spread across alphabetical
  order which produced non-physical composites.

#### ImageCanvas Base Viewer (`grdk/viewers/image_canvas.py`)
- `DisplaySettings` dataclass — window/level, percentile clipping, contrast, brightness, gamma, colormap, band selection
- `normalize_array()` pure function — 7-step rendering pipeline (complex→abs, band select, window/level, contrast/brightness, gamma, colormap LUT, uint8)
- `ImageCanvas(QGraphicsView)` — interactive image viewer with pan (drag), zoom (scroll wheel), pixel hover inspector, fit-to-view (Ctrl+0 / double-click)
- `ImageCanvasThumbnail(ImageCanvas)` — fixed-size non-interactive subclass for use in grids and galleries
- `array_to_qimage()` — thin wrapper converting numpy arrays to QImage via normalize_array
- Built-in colormaps (grayscale, viridis, inferno, plasma, hot) as 256x3 uint8 numpy LUTs — no matplotlib dependency

#### Display Controls (`grdk/widgets/_display_controls.py`)
- `build_display_controls()` convenience function — builds QGroupBox with window/level, percentile, contrast, brightness, gamma, colormap, and band controls
- Follows existing `build_param_controls()` pattern
- Supports `show` parameter to select subset of controls

#### GRDL v2 API Integration
- Processor discovery via `inspect.getmembers()` scanning `grdl.image_processing` and `grdl.coregistration` — no hardcoded processor list
- GPU dispatch with `__gpu_compatible__` flag checking and automatic CPU fallback
- Tag filtering via `__processor_tags__` dict — modality and category-based processor palette filtering
- Progress callback forwarding with per-step rescaling to overall pipeline progress
- Exception handling with `grdl.exceptions.GrdlError` graceful fallback
- Data prep delegation via optional `grdl.data_prep.Normalizer` in OWChipper

#### Widgets
- `ow_processor.py` — single-step processor configuration with dynamic parameter UI
- `ow_orchestrator.py` — multi-step pipeline builder with drag-and-drop reordering
- `ow_chipper.py` — chip extraction with region selection and optional GRDL normalization
- `ow_preview.py` — real-time GPU-accelerated before/after chip preview grid

#### Tests
- 19 tests for ImageCanvas (DisplaySettings defaults, normalize_array edge cases, colormaps, Qt widget tests)
- 5 tests for GRDL integration (discovery, GPU flags, tag filtering, progress callbacks, exception handling)
- 7 tests for core config
- Full suite: 176 passed, 0 failures

#### Documentation
- `README.md` — full project documentation with architecture, quick start, and file inventory
- `docs/architecture.md` — system architecture, layer boundaries, data flow, signal types, GRDL integration
- `docs/image-canvas.md` — ImageCanvas API reference and extensibility guide
- `CLAUDE.md` — updated development guide with all modules, patterns, and standards

### Changed
- `_pol_utils._reader_polarization()` — unified sensor coverage; replaces all direct
  private-attribute accesses (`reader._requested_polarization`,
  `reader.metadata.swath_info`) that were scattered across `main_window.py`.
- `ViewerMainWindow._on_pol_swap_check()` — removed 12-line isinstance cascade;
  uses `_get_available_polarizations()` (generic public API) to determine whether
  a pol-swap is applicable.
- `ViewerMainWindow._update_remap_for_dock()` — removed 5-reader isinstance cascade;
  replaced with dtype check (`np.issubdtype(complexfloating)`) plus
  `PolarimetricMode.from_reader()` for magnitude SAR (e.g. SIDD).
- `ViewerMainWindow._update_pane_pol_names()` — now uses `channel_pol_map()` as the
  primary source (catches NISAR CYX multi-pol); BIOMASS `polarizations` attribute
  consulted as fallback.
- `create_geolocation()` (`geo_viewer.py`) — refactored from 6-branch isinstance
  waterfall to declarative registry (`_GEO_REGISTRY`); extending to a new sensor
  requires a single list entry.
- `grdk/viewers/chip_gallery.py` — replaced `_array_to_pixmap()` with `ImageCanvasThumbnail`
- `grdk/widgets/geodev/ow_preview.py` — replaced `_array_to_pixmap()` with `ImageCanvasThumbnail`
- `grdk/viewers/__init__.py` — updated module docstring to reference ImageCanvas components

### Removed
- `ViewerMainWindow._get_reader_polarization()` — duplicate of `_pol_utils._reader_polarization()`;
  all six call-sites now use the canonical function from `_pol_utils`.
- `grdk/core/` — empty directory; execution logic lives in `grdl_rt.execution` (grdl-runtime).
- `grdk/catalog/` — empty directory; catalog logic lives in `grdl_rt.catalog` (grdl-runtime).
- Stale CHANGELOG entries for `grdk/core/discovery.py`, `grdk/core/gpu.py`,
  `grdk/core/executor.py`, `grdk/core/config.py` — these modules were never created
  in grdk (they exist in grdl-runtime).
- `_array_to_pixmap()` from `chip_gallery.py` — duplicated min-max normalization replaced by shared `normalize_array()`
- `_array_to_pixmap()` from `ow_preview.py` — same deduplication

## [0.1.0] — Initial Release

### Added
- Orange Data Mining add-on plugin architecture
- GEODEV mode: image loading, co-registration, stack viewing, chipping, labeling, workflow orchestration, publishing, project management
- Administrative mode: catalog browser, artifact editor, workflow manager, update monitor
- Core models: Chip, ChipSet, ChipLabel, WorkflowDefinition, WorkflowStep, GrdkProject
- DSL compiler: Python decorator syntax and YAML serialization with bidirectional conversion
- Catalog: SQLite + FTS5 artifact database with path resolution and update checking
- Viewers: NapariStackViewer, ChipGalleryWidget, polygon tools
- Custom signal types: ImageStack, ChipSetSignal, ProcessingPipelineSignal, WorkflowArtifactSignal, GrdkProjectSignal
- 134 tests (104 passed, 30 skipped for Qt-dependent tests)
