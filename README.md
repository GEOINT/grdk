# GRDK — GEOINT Rapid Development Kit

GUI toolkit for CUDA-optimized image processing workflow orchestration. Built as [Orange Data Mining](https://orangedatamining.com/) add-on plugins on top of the [GRDL](../grdl) library.

## What It Does

GRDK turns GRDL's image processing algorithms into a visual drag-and-drop workflow builder. Load multi-band satellite/aerial imagery, co-register image stacks, chip regions of interest, label training data, build processing pipelines with real-time GPU preview, and publish reproducible workflows — all without writing code.

### Two Modes

| Mode | Purpose | Widgets |
|------|---------|---------|
| **GEODEV** | Interactive GEOINT development | Image Loader, Stack Viewer, Co-Register, Processor, Orchestrator, Preview, Chipper, Labeler, Project, Publisher |
| **Admin** | Catalog management | Catalog Browser, Artifact Editor, Workflow Manager, Update Monitor |

## Quick Start

```bash
# Install in editable mode (core only)
pip install -e ".[dev]"

# With GUI support (Orange + napari)
pip install -e ".[gui,dev]"

# With GPU acceleration (CuPy + PyTorch)
pip install -e ".[gpu,dev]"
```

Launch Orange Canvas and find GRDK widgets in the **GEODEV** and **GRDK Admin** categories.

### Headless Execution

```bash
python -m grdk workflow.yaml --input image.tif --output result.tif
```

## Architecture

```
grdk/
  core/           # Business logic (no Qt) — workflow, executor, GPU, DSL, tags, config
  catalog/        # SQLite artifact catalog (no Qt) — database, resolver, updater, pool
  viewers/        # Embeddable Qt widgets — ImageCanvas, napari stack viewer, chip gallery
  widgets/
    _signals.py   # Custom Orange signal types (ImageStack, ChipSetSignal, etc.)
    _param_controls.py    # Dynamic parameter UI builder
    _display_controls.py  # Image display settings UI builder
    geodev/       # 10 GEODEV workflow widgets (ow_*.py)
    admin/        # 4 Admin catalog widgets (ow_*.py)
tests/            # pytest suite (149+ tests)
docs/             # Architecture and API documentation
```

### Layer Rules

- **`core/`** and **`catalog/`** must not import Qt — keeps headless execution clean
- **`viewers/`** are standalone Qt widgets, embeddable in any parent
- **`widgets/`** are Orange `OWBaseWidget` subclasses that compose `viewers/` and `core/`

### Key Dependencies

| Dependency | Required | Purpose |
|-----------|----------|---------|
| numpy, scipy | Yes | Array operations, image processing |
| pyyaml | Yes | Workflow DSL serialization |
| requests, packaging | Yes | Catalog update checking |
| orange3, orange-widget-base | GUI | Widget framework |
| napari[pyqt5] | GUI | Stack viewer with polygon drawing |
| cupy-cuda12x | GPU | CUDA-accelerated array operations |
| torch | GPU | ML model inference |

## Image Viewer (ImageCanvas)

All image display in GRDK is built on a shared `ImageCanvas` component — a `QGraphicsView`-based interactive viewer with:

- **Pan** (mouse drag), **Zoom** (scroll wheel, double-click to fit)
- **Contrast/Brightness** adjustment (does not modify source data)
- **Dynamic range windowing** (manual min/max or percentile-based)
- **Gamma correction**
- **Colormaps** (grayscale, viridis, inferno, plasma, hot)
- **Band selection** for multi-band imagery
- **Pixel inspector** (hover to read raw values)

Two variants:
- `ImageCanvas` — full interactive viewer for main display areas
- `ImageCanvasThumbnail` — non-interactive fixed-size variant for grids and galleries

See [docs/image-canvas.md](docs/image-canvas.md) for the API reference.

## GRDL Integration

GRDK automatically discovers and wraps GRDL processors:

- **Processor discovery**: Scans `grdl.image_processing` and `grdl.coregistration` for concrete classes
- **Tag filtering**: Processors with `__processor_tags__` can be filtered by modality (SAR, PAN, MSI, ...) and category (spatial_filter, contrast_enhancement, ...)
- **GPU dispatch**: Respects `__gpu_compatible__` flags — skips futile GPU attempts for scipy-dependent processors
- **Progress callbacks**: Forwards `progress_callback` through the pipeline with per-step rescaling
- **Exception handling**: Distinguishes `GrdlError` subtypes from general Python errors in logging
- **Normalization**: Optional GRDL `data_prep.Normalizer` integration in the Chipper widget

## Testing

```bash
# Full suite
pytest tests/ -v

# Specific module
pytest tests/test_image_canvas.py -v

# With coverage
pytest tests/ --cov=grdk --cov-report=term-missing
```

**149+ tests** across 15 test modules. Widget and Qt tests auto-skip when no display is available.

## Project Structure (All Files)

<details>
<summary>Click to expand full file listing</summary>

### Core (`grdk/core/`)
| File | Purpose |
|------|---------|
| `chip.py` | Chip, ChipSet, ChipLabel, PolygonRegion models |
| `config.py` | GrdkConfig dataclass, load/save from ~/.grdl/grdk_config.json |
| `discovery.py` | Processor discovery, tag filtering, class resolution |
| `dsl.py` | Workflow DSL — Python decorator syntax + YAML serialization |
| `executor.py` | WorkflowExecutor — headless pipeline execution with progress callbacks |
| `gpu.py` | GpuBackend — CuPy/PyTorch dispatch with CPU fallback |
| `project.py` | GrdkProject — project directory management with atomic saves |
| `tags.py` | ImageModality, DetectionType, ProjectTags, WorkflowTags |
| `workflow.py` | ProcessingStep, WorkflowDefinition, WorkflowState |

### Catalog (`grdk/catalog/`)
| File | Purpose |
|------|---------|
| `database.py` | ArtifactCatalog — SQLite FTS5 database with schema migrations |
| `models.py` | CatalogArtifact, ArtifactType, SearchResult models |
| `pool.py` | ThreadExecutorPool — background package install/download |
| `resolver.py` | Catalog path resolution (env var, config, default) |
| `updater.py` | UpdateChecker — PyPI/conda version monitoring |

### Viewers (`grdk/viewers/`)
| File | Purpose |
|------|---------|
| `image_canvas.py` | ImageCanvas, ImageCanvasThumbnail, DisplaySettings, normalize_array |
| `stack_viewer.py` | NapariStackViewer — multi-layer image viewer with polygon drawing |
| `chip_gallery.py` | ChipGalleryWidget — scrollable thumbnail grid with click-to-label |
| `polygon_tools.py` | chip_stack_at_polygons — polygon-based chip extraction |

### Widgets — GEODEV (`grdk/widgets/geodev/`)
| Widget | Name | Purpose |
|--------|------|---------|
| `ow_image_loader.py` | Image Loader | Load TIFF/NITF/HDF5 into image stack |
| `ow_stack_viewer.py` | Stack Viewer | Napari viewer with polygon drawing |
| `ow_coregister.py` | Co-Register | Affine/projective/feature-match registration |
| `ow_processor.py` | Processor | Single GRDL processor with tunable params |
| `ow_orchestrator.py` | Orchestrator | Multi-step pipeline builder |
| `ow_preview.py` | Preview | Real-time before/after GPU preview |
| `ow_chipper.py` | Chipper | Extract chips from polygon definitions |
| `ow_labeler.py` | Labeler | Click-to-label chip classification |
| `ow_project.py` | Project | Save/load GRDK project directories |
| `ow_publisher.py` | Publisher | Publish workflows to catalog |

### Widgets — Admin (`grdk/widgets/admin/`)
| Widget | Name | Purpose |
|--------|------|---------|
| `ow_catalog_browser.py` | Catalog Browser | Search/discover artifacts |
| `ow_artifact_editor.py` | Artifact Editor | Edit catalog artifact metadata |
| `ow_workflow_manager.py` | Workflow Manager | Import/export workflow files |
| `ow_update_monitor.py` | Update Monitor | Check for package updates |

### Shared Widget Infrastructure (`grdk/widgets/`)
| File | Purpose |
|------|---------|
| `_signals.py` | ImageStack, ChipSetSignal, ProcessingPipelineSignal, WorkflowArtifactSignal, GrdkProjectSignal |
| `_param_controls.py` | build_param_controls() — auto-generates Qt controls from TunableParameterSpec |
| `_display_controls.py` | build_display_controls() — contrast/brightness/gamma/colormap/window UI |

</details>

## License

MIT License. Copyright (c) 2026 geoint.org. See [LICENSE](LICENSE).
