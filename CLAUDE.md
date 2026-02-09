# GRDK Development Guide

## Project Overview

GRDK (GEOINT Rapid Development Kit) is a GUI toolkit for CUDA-optimized image processing workflow orchestration. It is built as a set of Orange Data Mining add-on plugins on top of the GRDL library.

Two modes:
- **GEODEV Mode**: Interactive GEOINT development — image loading, co-registration, chipping, labeling, workflow orchestration with real-time GPU preview, and workflow publishing.
- **Administrative Mode**: Artifact catalog management — search/discover/download GRDL processors and GRDK workflows.

## Architecture

### Layers

1. **`grdk/core/`** — Non-GUI business logic (project model, workflow model, DSL compiler, GPU backend, tag taxonomy, executor, discovery, config). **No Qt imports.**
2. **`grdk/catalog/`** — SQLite catalog database, path resolution, update checking, thread pool. **No Qt imports.**
3. **`grdk/viewers/`** — Embeddable Qt widgets (ImageCanvas, NapariStackViewer, ChipGalleryWidget, polygon tools). Requires PySide6.
4. **`grdk/widgets/`** — Orange3 OWWidget subclasses. Depends on `core/`, `catalog/`, and `viewers/`.

### Key Relationships

- GRDK depends on **GRDL** (`grdl` package) for all image I/O, geolocation, and processing operations.
- GRDK widgets use **custom signal types** (`grdk/widgets/_signals.py`) instead of Orange's default Table signals.
- GPU acceleration uses **CuPy** (numpy-compatible) for image transforms and **PyTorch** for ML model inference.
- The **DSL** supports both Python decorator syntax and YAML serialization, with bidirectional conversion.
- **One GRDK widget per GRDL component type** — widgets dynamically discover available GRDL classes via `discover_processors()`, not a hardcoded list.

### Image Display

All image display flows through `grdk/viewers/image_canvas.py`:

- **`normalize_array(arr, DisplaySettings) → uint8`** — pure function (no Qt). Steps: complex→abs, band selection, window/level, contrast/brightness, gamma, colormap LUT, clip to uint8.
- **`ImageCanvas(QGraphicsView)`** — interactive viewer: pan, zoom, hover, pixel inspector. Uses QGraphicsScene for future overlays/annotations.
- **`ImageCanvasThumbnail(ImageCanvas)`** — fixed-size non-interactive subclass replacing all QLabel+QPixmap thumbnail patterns.
- **`build_display_controls()`** in `grdk/widgets/_display_controls.py` — convenience UI builder for display settings (contrast, brightness, gamma, colormap, window/level).

Colormaps are implemented as 256×3 uint8 numpy LUTs (grayscale, viridis, inferno, plasma, hot) — no matplotlib dependency.

### GRDL Integration Points

Six integration surfaces (see `docs/architecture.md` for details):

1. **Processor Discovery** (`grdk/core/discovery.py`) — scans `grdl.image_processing` and `grdl.coregistration` via `inspect.getmembers()`.
2. **GPU Dispatch** (`grdk/core/gpu.py`) — wraps processor execution with optional CuPy acceleration, checks `__gpu_compatible__` flag.
3. **Tag Filtering** (`grdk/core/discovery.py`) — reads `__processor_tags__` dict for modality/category filtering.
4. **Progress Callbacks** (`grdk/core/executor.py`) — forwards `progress_callback` to each processor's `apply()`, rescales per-step progress.
5. **Exception Handling** (`grdk/core/executor.py`) — imports `grdl.exceptions.GrdlError` with graceful fallback, wraps in `RuntimeError`.
6. **Data Prep Delegation** (`grdk/widgets/geodev/ow_chipper.py`) — optionally uses `grdl.data_prep.Normalizer` for post-extraction normalization.

## File Inventory

### `grdk/core/` (no Qt)

| File | Purpose |
|------|---------|
| `chip.py` | Chip, ChipSet, ChipLabel models |
| `config.py` | GrdkConfig dataclass (loaded from `~/.grdl/grdk_config.json`) |
| `discovery.py` | `discover_processors()` — scans GRDL for transform/coregistration classes |
| `dsl.py` | DSL compiler (Python decorator ↔ YAML) |
| `executor.py` | WorkflowExecutor — runs pipeline steps with progress callbacks |
| `gpu.py` | GpuBackend — CuPy GPU dispatch with CPU fallback |
| `project.py` | Project model (directory structure, manifest) |
| `tags.py` | Tag taxonomy and processor filtering |
| `workflow.py` | WorkflowDefinition, WorkflowStep models |

### `grdk/catalog/` (no Qt)

| File | Purpose |
|------|---------|
| `database.py` | ArtifactCatalog — SQLite + FTS5 |
| `models.py` | Artifact, ArtifactType models |
| `pool.py` | ThreadExecutorPool for background installs |
| `resolver.py` | Catalog path resolution (env → config → default) |
| `updater.py` | UpdateChecker — PyPI/Conda repodata polling |

### `grdk/viewers/` (Qt widgets, no Orange)

| File | Purpose |
|------|---------|
| `image_canvas.py` | DisplaySettings, normalize_array, ImageCanvas, ImageCanvasThumbnail |
| `stack_viewer.py` | NapariStackViewer — napari-based multi-image viewer |
| `chip_gallery.py` | ChipGalleryWidget — scrollable chip grid with click-to-label |
| `polygon_tools.py` | Polygon drawing/extraction utilities |

### `grdk/widgets/` (Orange OWWidgets)

| File | Purpose |
|------|---------|
| `_signals.py` | Signal types (ImageStack, ChipSetSignal, ProcessingPipelineSignal, etc.) |
| `_param_controls.py` | `build_param_controls()` — dynamic UI from processor params |
| `_display_controls.py` | `build_display_controls()` — ImageCanvas display settings UI |
| `geodev/ow_image_loader.py` | OWImageLoader — load image stacks from disk |
| `geodev/ow_stack_viewer.py` | OWStackViewer — napari viewer with polygon chip extraction |
| `geodev/ow_coregister.py` | OWCoRegister — image co-registration |
| `geodev/ow_chipper.py` | OWChipper — chip extraction with optional GRDL normalization |
| `geodev/ow_processor.py` | OWProcessor — single-step processor configuration |
| `geodev/ow_orchestrator.py` | OWOrchestrator — multi-step pipeline builder |
| `geodev/ow_preview.py` | OWPreview — real-time GPU preview grid |
| `geodev/ow_labeler.py` | OWLabeler — chip labeling interface |
| `geodev/ow_publisher.py` | OWPublisher — workflow publishing to catalog |
| `geodev/ow_project.py` | OWProject — project directory management |
| `admin/ow_catalog_browser.py` | OWCatalogBrowser — search/browse artifacts |
| `admin/ow_artifact_editor.py` | OWArtifactEditor — edit artifact metadata |
| `admin/ow_workflow_manager.py` | OWWorkflowManager — import/export workflows |
| `admin/ow_update_monitor.py` | OWUpdateMonitor — check for updates |

## Development Environment

Use the same Python environment as GRDL. Install GRDK in editable mode:

```bash
pip install -e ".[dev]"
```

For GUI development:
```bash
pip install -e ".[gui,dev]"
```

For GPU support:
```bash
pip install -e ".[gpu,dev]"
```

## Standards

Follow GRDL's development standards (see `C:\projects\grdl\CLAUDE.md`):

- **PEP 8/257/484** — naming, docstrings, type hints
- **NumPy-style docstrings** — Parameters, Returns, Raises sections
- **File headers** — encoding, title, dependencies, author, license, dates
- **Imports** — three groups (stdlib, third-party, GRDK/GRDL internal)
- **No global state** — no singletons, no module-level side effects
- **Fail fast** — clear ImportError for missing dependencies
- **Vectorized numpy** — for all array operations

### Additional GRDK Standards

- **core/ and catalog/ must not import Qt** — keep GUI-free for headless execution.
- **Qt imports use PySide6 directly** — do not use AnyQt or PyQt5. Import from `PySide6.QtWidgets`, `PySide6.QtCore`, `PySide6.QtGui`. Use `Signal` (not `pyqtSignal`).
- **Orange widgets** follow the `ow_<name>.py` naming convention.
- **Signal types** are defined in `_signals.py` and wrap core models.
- **Tests** use pytest, synthetic data, no real imagery files.
- **Image display** goes through ImageCanvas — never create standalone QLabel+QPixmap thumbnail code.
- **Display enhancements** are pure functions — `normalize_array()` doesn't modify source data.
- **Viewers are standalone Qt widgets** — no Orange dependency. They can be embedded in any Qt app.
- **Remote GUI support** — GUI must work over X11 forwarding. Avoid OpenGL-dependent rendering; use `QT_QPA_PLATFORM=xcb` and `QT_QUICK_BACKEND=software` for container compatibility.

## Catalog Path Resolution

Priority order:
1. `GRDK_CATALOG_PATH` environment variable
2. `~/.grdl/config.json` → `"catalog_path"` field
3. `~/.grdl/catalog.db` (default)

## Configuration

`GrdkConfig` dataclass loaded from `~/.grdl/grdk_config.json`:

| Setting | Default | Purpose |
|---------|---------|---------|
| `thumb_size` | 128 | Thumbnail pixel size |
| `preview_thumb` | 160 | Preview thumbnail size |
| `debounce_ms` | 50 | UI debounce interval |
| `update_timeout` | 10.0 | Network timeout (seconds) |
| `max_workers` | 4 | Thread pool size |

## Testing

```bash
pytest tests/ -v
```

Tests are in `tests/` following `test_<module>.py` naming. Widget tests are in `tests/test_widgets/`.

| Layer | Test Approach | Qt Required |
|-------|---------------|-------------|
| `core/` | Unit tests with synthetic numpy arrays | No |
| `catalog/` | Unit tests with temp SQLite databases | No |
| `viewers/image_canvas` | Pure function tests (normalize_array) + Qt widget tests | Partial |
| `widgets/` | Import smoke tests, skip if no display | Yes (skipped) |
| GRDL integration | Mock-based tests for discovery, GPU flag, callbacks | No |

## Documentation

- `README.md` — project overview and quick start
- `docs/architecture.md` — system architecture, layer boundaries, data flow, GRDL integration
- `docs/image-canvas.md` — ImageCanvas API reference and extensibility guide
- `CHANGELOG.md` — version history

## Git Practices

Same as GRDL: imperative commit messages, one change per commit, `<type>/<description>` branches.
