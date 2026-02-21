# GRDK Development Guide

## Project Overview

GRDK (GEOINT Rapid Development Kit) is a GUI toolkit for CUDA-optimized image processing workflow orchestration. It is built as a set of Orange Data Mining add-on plugins on top of the GRDL library.

Two modes:
- **GEODEV Mode**: Interactive GEOINT development — image loading, co-registration, chipping, labeling, workflow orchestration with real-time GPU preview, and workflow publishing.
- **Administrative Mode**: Artifact catalog management — search/discover/download GRDL processors and GRDK workflows.

## Architecture

### Layers

```
grdl  (processing primitives — no framework awareness)
  ↓
grdl-runtime  (execution framework — no GUI)
  ├── grdl_rt.execution/  — workflow engine, builder, GPU backend, discovery, DSL
  └── grdl_rt.catalog/    — artifact storage, search, updates
  ↓
grdk  (Qt/Orange GUI — uses grdl-runtime for execution)
  ├── grdk/viewers/  — embeddable Qt widgets (no Orange dependency)
  └── grdk/widgets/  — Orange3 OWWidget subclasses
```

1. **`grdl-runtime`** (`grdl_rt` package) — Non-GUI execution framework: workflow models, DSL compiler, GPU backend, tag taxonomy, executor, discovery, config, project model, artifact catalog (SQLite + FTS5), path resolution, update checking, thread pool. **No Qt imports.** See `../grdl-runtime/CLAUDE.md` for full details.
2. **`grdk/viewers/`** — Embeddable Qt widgets (ImageCanvas, NapariStackViewer, ChipGalleryWidget, polygon tools). Requires PyQt6. No Orange dependency.
3. **`grdk/widgets/`** — Orange3 OWWidget subclasses. Depends on `grdl_rt`, `grdl`, and `viewers/`.

### Key Relationships

- GRDK depends on **GRDL** (`grdl` package) for all image I/O, geolocation, and processing operations.
- GRDK depends on **grdl-runtime** (`grdl_rt` package) for workflow execution, discovery, GPU dispatch, catalog, and project management.
- GRDK widgets use **custom signal types** (`grdk/widgets/_signals.py`) instead of Orange's default Table signals.
- GPU acceleration uses **CuPy** (numpy-compatible) for image transforms and **PyTorch** for ML model inference.
- The **DSL** supports both Python decorator syntax and YAML serialization, with bidirectional conversion (via `grdl_rt.execution.dsl`).
- **One GRDK widget per GRDL component type** — widgets dynamically discover available GRDL classes via `grdl_rt.execution.discover_processors()`, not a hardcoded list.

### Image Display

All image display flows through `grdk/viewers/image_canvas.py`:

- **`normalize_array(arr, DisplaySettings) → uint8`** — pure function (no Qt). Steps: complex→abs, band selection, window/level, contrast/brightness, gamma, colormap LUT, clip to uint8.
- **`ImageCanvas(QGraphicsView)`** — interactive viewer: pan, zoom, hover, pixel inspector. Uses QGraphicsScene for future overlays/annotations.
- **`ImageCanvasThumbnail(ImageCanvas)`** — fixed-size non-interactive subclass replacing all QLabel+QPixmap thumbnail patterns.
- **`build_display_controls()`** in `grdk/widgets/_display_controls.py` — convenience UI builder for display settings (contrast, brightness, gamma, colormap, window/level).

Colormaps are implemented as 256×3 uint8 numpy LUTs (grayscale, viridis, inferno, plasma, hot) — no matplotlib dependency.

### GRDL / grdl-runtime Integration Points

Six integration surfaces (see `docs/architecture.md` for details):

1. **Processor Discovery** (`grdl_rt.execution.discovery`) — scans `grdl.image_processing` and `grdl.coregistration` via `inspect.getmembers()`.
2. **GPU Dispatch** (`grdl_rt.execution.gpu`) — wraps processor execution with optional CuPy acceleration, checks `__gpu_compatible__` flag.
3. **Tag Filtering** (`grdl_rt.execution.discovery`) — reads `__processor_tags__` dict for modality/category filtering.
4. **Progress Callbacks** (`grdl_rt.execution.executor`) — forwards `progress_callback` to each processor's `apply()`, rescales per-step progress.
5. **Exception Handling** (`grdl_rt.execution.executor`) — imports `grdl.exceptions.GrdlError` with graceful fallback, wraps in `RuntimeError`.
6. **Data Prep Delegation** (`grdk/widgets/geodev/ow_chipper.py`) — optionally uses `grdl.data_prep.Normalizer` for post-extraction normalization.

## File Inventory

### `grdl_rt.execution` (from grdl-runtime, no Qt)

These modules live in the `grdl-runtime` package and are imported by GRDK widgets as `from grdl_rt.execution.<module> import ...`:

| Module | Key Types Used by Widgets |
|--------|--------------------------|
| `discovery` | `discover_processors()`, `get_processor_tags()`, `filter_processors()` |
| `gpu` | `GpuBackend` |
| `workflow` | `WorkflowDefinition`, `ProcessingStep`, `WorkflowState` |
| `dsl` | `DslCompiler` |
| `chip` | `Chip`, `ChipSet`, `ChipLabel`, `PolygonRegion` |
| `tags` | `WorkflowTags`, `ProjectTags` |
| `project` | `GrdkProject` |
| `config` | `GrdkConfig`, `load_config()` |
| `executor` | `WorkflowExecutor` |

### `grdl_rt.catalog` (from grdl-runtime, no Qt)

These modules live in the `grdl-runtime` package and are imported by GRDK widgets as `from grdl_rt.catalog.<module> import ...`:

| Module | Key Types Used by Widgets |
|--------|--------------------------|
| `database` | `ArtifactCatalog` (SQLite + FTS5) |
| `models` | `Artifact`, `UpdateResult` |
| `resolver` | `resolve_catalog_path()` |
| `updater` | `ArtifactUpdateWorker` |
| `pool` | `ThreadExecutorPool` |

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

Follow GRDL's development standards (see `../grdl/CLAUDE.md`):

- **PEP 8/257/484** — naming, docstrings, type hints
- **NumPy-style docstrings** — Parameters, Returns, Raises sections
- **File headers** — encoding, title, dependencies, author, license, dates
- **Imports** — three groups (stdlib, third-party, GRDK/GRDL internal)
- **No global state** — no singletons, no module-level side effects
- **Fail fast** — clear ImportError for missing dependencies
- **Vectorized numpy** — for all array operations

### Additional GRDK Standards

- **Execution and catalog logic lives in grdl-runtime** — GRDK widgets import from `grdl_rt.execution` and `grdl_rt.catalog`, not from local `grdk/core/` or `grdk/catalog/` directories (these do not exist).
- **Qt imports use PyQt6 directly** — do not use AnyQt or PyQt5. Import from `PyQt6.QtWidgets`, `PyQt6.QtCore`, `PyQt6.QtGui`. Use `from PyQt6.QtCore import pyqtSignal as Signal` for signal declarations.
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
| `grdl_rt.execution` | Tests in grdl-runtime repo | No |
| `grdl_rt.catalog` | Tests in grdl-runtime repo | No |
| `viewers/image_canvas` | Pure function tests (normalize_array) + Qt widget tests | Partial |
| `widgets/` | Import smoke tests, skip if no display | Yes (skipped) |
| GRDL integration | Mock-based tests for discovery, GPU flag, callbacks | No |

## Dependency Management

### Source of Truth: `pyproject.toml`

**`pyproject.toml` is the single source of truth** for all dependencies. All package metadata, dependencies, and optional extras are defined here. This file drives PyPI publication and is read by build tools.

### Keeping Files in Sync

Three files must be kept synchronized:

| File | Purpose | How to Update |
|------|---------|---------------|
| `pyproject.toml` | **Source of truth** — package metadata, all dependencies, extras | Edit directly; this is the authoritative definition |
| `requirements.txt` (if it exists) | Development convenience — pinned versions for reproducible environments | `pip freeze > requirements.txt` after updating dependencies in `pyproject.toml` and installing |
| `.github/workflows/publish.yml` | PyPI publication — **DO NOT EDIT this file manually** (it extracts version from `pyproject.toml` automatically) | No action needed; the workflow reads `version` from `pyproject.toml` |

**Workflow:**
1. Update dependencies in `pyproject.toml` (add new packages, change versions, create/rename extras)
2. Install dependencies: `pip install -e ".[all,dev]"` (or appropriate extras for your work)
3. If `requirements.txt` exists in this project, regenerate it: `pip freeze > requirements.txt`
4. Commit both files
5. When creating a release, bump the `version` field in `pyproject.toml` (semantic versioning: `major.minor.patch`)
6. Create a git tag (e.g., `v0.2.0`) and push — the publish workflow triggers automatically

### Versioning for PyPI

- Versions follow **semantic versioning**: `major.minor.patch` (e.g., `0.1.0`, `1.2.3`)
- Update `version = "X.Y.Z"` in `pyproject.toml` before creating a release
- The publish workflow extracts the version automatically — no manual version extraction needed

## Documentation

- `README.md` — project overview and quick start
- `docs/architecture.md` — system architecture, layer boundaries, data flow, GRDL integration
- `docs/image-canvas.md` — ImageCanvas API reference and extensibility guide
- `CHANGELOG.md` — version history

## Git Practices

Same as GRDL: imperative commit messages, one change per commit, `<type>/<description>` branches.
