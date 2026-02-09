# GRDK Architecture

## System Context

```
Orange Data Mining Canvas
    |
    v
GRDK Widgets (ow_*.py)          <-- GUI layer (PySide6 + Orange)
    |
    +-- grdk/viewers/            <-- Embeddable Qt components
    |       ImageCanvas, NapariStackViewer, ChipGallery
    |
    +-- grdk/core/               <-- Business logic (no Qt)
    |       WorkflowExecutor, GpuBackend, DslCompiler
    |
    +-- grdk/catalog/            <-- Data management (no Qt)
    |       ArtifactCatalog, UpdateChecker
    |
    v
GRDL Library (grdl package)     <-- Image processing engine
        ImageTransform, CoRegistration, ImageReader
```

## Layer Boundaries

### Rule 1: No Qt in core/ or catalog/

These layers must remain importable without PySide6 installed. This enables:
- Headless workflow execution (`python -m grdk workflow.yaml`)
- Server-side batch processing
- Unit testing without a display

### Rule 2: Viewers are standalone Qt widgets

`grdk/viewers/` components (ImageCanvas, NapariStackViewer, ChipGalleryWidget) are pure Qt widgets with no Orange dependency. They can be embedded in any Qt application — GRDK's Orange widgets are just one consumer.

### Rule 3: Widgets compose, don't implement

Orange widgets (`ow_*.py`) compose viewers and core logic. They handle signal routing, Orange settings persistence, and widget metadata — but delegate all real work to the lower layers.

## Data Flow

### GEODEV Workflow

```
ImageLoader ──ImageStack──> StackViewer ──ChipSetSignal──> Labeler
                                |                            |
                                v                            v
                           CoRegister                    Publisher
                                |
                                v
Processor ──PipelineSignal──> Orchestrator ──PipelineSignal──> Preview
                                |
                                v
Chipper ──ChipSetSignal──> Labeler ──ChipSetSignal──> Project
```

### Signal Types

| Signal | Carries | Producers | Consumers |
|--------|---------|-----------|-----------|
| `ImageStack` | List of ImageReaders + metadata | ImageLoader, CoRegister | StackViewer, Chipper, CoRegister |
| `ChipSetSignal` | ChipSet with labels and regions | StackViewer, Chipper, Labeler | Labeler, Preview, Project |
| `ProcessingPipelineSignal` | WorkflowDefinition (step sequence) | Processor, Orchestrator | Preview |
| `WorkflowArtifactSignal` | Published workflow (DSL + metadata) | Publisher | WorkflowManager |
| `GrdkProjectSignal` | Project directory reference | Project | (external consumers) |

## GRDL Integration Points

GRDK interacts with GRDL through six integration surfaces:

### 1. Processor Discovery (`grdk/core/discovery.py`)

Scans `grdl.image_processing` and `grdl.coregistration` using `inspect.getmembers()`. Finds concrete classes with `apply()` (transforms) or `estimate()` (co-registration) methods. No hardcoded processor list — new GRDL processors appear automatically.

### 2. GPU Dispatch (`grdk/core/gpu.py`)

Wraps processor execution with optional CuPy acceleration:
1. Check `__gpu_compatible__` flag — skip GPU if `False`
2. Transfer array to GPU via `cupy.asarray()`
3. Run `processor.apply()` on GPU array
4. On failure, fall back to CPU
5. Transfer result back via `cupy.asnumpy()`

### 3. Tag Filtering (`grdk/core/discovery.py`)

Reads `__processor_tags__` dict from processor classes:
```python
{'modalities': ('SAR', 'PAN'), 'category': 'spatial_filter', 'description': '...'}
```
Used by OWProcessor and OWOrchestrator to filter the processor palette by modality and category.

### 4. Progress Callbacks (`grdk/core/executor.py`)

WorkflowExecutor forwards `progress_callback` to each processor's `apply()` via kwargs. Rescales per-step progress to overall pipeline progress:
```
step_callback = lambda f: overall_callback(base + f * scale)
```

### 5. Exception Handling (`grdk/core/executor.py`)

Imports `grdl.exceptions.GrdlError` with graceful fallback. Distinguishes GRDL-specific errors from general Python errors in logging, then wraps both in `RuntimeError` for pipeline callers.

### 6. Data Prep Delegation (`grdk/widgets/geodev/ow_chipper.py`)

OWChipper optionally uses `grdl.data_prep.Normalizer` for post-extraction normalization (minmax, zscore, percentile). Gracefully degrades if GRDL data_prep module is unavailable.

## Image Display Architecture

All image display flows through the `ImageCanvas` component:

```
Source numpy array
    |
    v
normalize_array(arr, DisplaySettings)     <-- Pure function, no Qt
    |  1. Complex → abs
    |  2. Band selection
    |  3. Window/level
    |  4. Contrast/brightness
    |  5. Gamma correction
    |  6. Colormap LUT
    |  7. Clip to uint8
    v
QImage → QPixmap → QGraphicsPixmapItem    <-- Qt rendering
    |
    +-- ImageCanvas (interactive: pan, zoom, hover)
    +-- ImageCanvasThumbnail (fixed-size, non-interactive)
```

### Before ImageCanvas

Previously, two separate `_array_to_pixmap()` functions existed (in `chip_gallery.py` and `ow_preview.py`) with duplicated min-max normalization and no interactive controls. These have been replaced by the shared ImageCanvas infrastructure.

## Catalog Architecture

```
UpdateChecker ──checks──> PyPI / Conda repodata
    |
    v
ArtifactCatalog (SQLite + FTS5)
    |
    +-- schema_version table (migration framework)
    +-- artifacts table (name, type, version, tags, metadata)
    +-- artifacts_fts table (full-text search)
    +-- remote_versions table (latest available versions)
    |
    v
ThreadExecutorPool ──pip install──> local environment
```

### Path Resolution

1. `GRDK_CATALOG_PATH` environment variable (highest priority)
2. `~/.grdl/config.json` → `catalog_path` field
3. `~/.grdl/catalog.db` (default fallback)

## Configuration

`GrdkConfig` dataclass loaded from `~/.grdl/grdk_config.json`:

| Setting | Default | Purpose |
|---------|---------|---------|
| `thumb_size` | 128 | Thumbnail pixel size |
| `preview_thumb` | 160 | Preview thumbnail size |
| `debounce_ms` | 50 | UI debounce interval |
| `update_timeout` | 10.0 | Network timeout (seconds) |
| `max_workers` | 4 | Thread pool size |

## Testing Strategy

| Layer | Test Approach | Qt Required |
|-------|---------------|-------------|
| `core/` | Unit tests with synthetic numpy arrays | No |
| `catalog/` | Unit tests with temp SQLite databases | No |
| `viewers/image_canvas` | Pure function tests (normalize_array) + Qt widget tests | Partial |
| `widgets/` | Import smoke tests, skip if no display | Yes (skipped) |
| GRDL integration | Mock-based tests for discovery, GPU flag, callbacks | No |
