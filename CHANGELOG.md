# Changelog

All notable changes to GRDK are documented in this file.

## [Unreleased]

### Added

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

#### Core Modules
- `grdk/core/discovery.py` — `discover_processors()` for dynamic GRDL processor scanning
- `grdk/core/gpu.py` — `GpuBackend` with CuPy GPU dispatch and CPU fallback
- `grdk/core/executor.py` — `WorkflowExecutor` with progress callbacks and exception wrapping
- `grdk/core/config.py` — `GrdkConfig` dataclass loaded from `~/.grdl/grdk_config.json`

#### Widgets
- `ow_processor.py` — single-step processor configuration with dynamic parameter UI
- `ow_orchestrator.py` — multi-step pipeline builder with drag-and-drop reordering
- `ow_chipper.py` — chip extraction with region selection and optional GRDL normalization
- `ow_preview.py` — real-time GPU-accelerated before/after chip preview grid

#### Tests
- 19 tests for ImageCanvas (DisplaySettings defaults, normalize_array edge cases, colormaps, Qt widget tests)
- 5 tests for GRDL integration (discovery, GPU flags, tag filtering, progress callbacks, exception handling)
- 7 tests for core config
- Full suite: 149 passed, 30 skipped, 0 failures

#### Documentation
- `README.md` — full project documentation with architecture, quick start, and file inventory
- `docs/architecture.md` — system architecture, layer boundaries, data flow, signal types, GRDL integration
- `docs/image-canvas.md` — ImageCanvas API reference and extensibility guide
- `CLAUDE.md` — updated development guide with all modules, patterns, and standards

### Changed
- `grdk/viewers/chip_gallery.py` — replaced `_array_to_pixmap()` with `ImageCanvasThumbnail`
- `grdk/widgets/geodev/ow_preview.py` — replaced `_array_to_pixmap()` with `ImageCanvasThumbnail`
- `grdk/viewers/__init__.py` — updated module docstring to reference ImageCanvas components

### Removed
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
