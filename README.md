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
# Install in editable mode
pip install -e .

# With dev tools (pytest, black, mypy)
pip install -e ".[dev]"
```

### Launching the Canvas

GRDK provides the `grdk-canvas` command, which launches the [Orange Canvas](https://orangedatamining.com/) with the PyQt6 backend pre-configured and GRDK widgets registered:

```bash
grdk-canvas
```

This is the primary way to use GRDK interactively. The canvas loads two widget categories:

- **GEODEV** — Image Loader, Stack Viewer, Co-Register, Processor, Orchestrator, Preview, Chipper, Labeler, Project, Publisher
- **GRDK Admin** — Catalog Browser, Artifact Editor, Workflow Manager, Update Monitor

Drag widgets onto the canvas, connect them with signal wires, and build image processing workflows visually.

> **Note:** `grdk-canvas` sets `QT_API=pyqt6` and configures the AnyQt backend before importing Orange. Always use `grdk-canvas` instead of `orange-canvas` to ensure the correct Qt backend.

### Headless Execution

Run workflows without a GUI:

```bash
python -m grdk workflow.yaml --input image.tif --output result.tif
python -m grdk workflow.yaml --input image.tif --output result.tif --no-gpu
```

## Architecture

```
grdl              (processing primitives — no framework awareness)
  ↓
grdl-runtime      (execution framework — no GUI)
  ├── grdl_rt.execution/  — workflow engine, GPU backend, discovery, DSL
  └── grdl_rt.catalog/    — artifact storage, search, updates
  ↓
grdk              (Qt/Orange GUI)
  ├── viewers/    — embeddable Qt widgets (ImageCanvas, napari viewer, chip gallery)
  ├── widgets/
  │   ├── _signals.py          — custom Orange signal types
  │   ├── _param_controls.py   — dynamic parameter UI builder
  │   ├── _display_controls.py — image display settings UI builder
  │   ├── geodev/              — 10 GEODEV workflow widgets (ow_*.py)
  │   └── admin/               — 4 Admin catalog widgets (ow_*.py)
  ├── _launcher.py             — grdk-canvas entry point
  └── _pyqt6_bootstrap.py     — PyQt6/AnyQt backend setup
tests/            — pytest suite
docs/             — architecture and API documentation
```

### Layer Rules

- **Execution and catalog logic lives in [grdl-runtime](../grdl-runtime)** (`grdl_rt` package) — GRDK widgets import from `grdl_rt.execution` and `grdl_rt.catalog`, not from local directories
- **`viewers/`** are standalone Qt widgets, embeddable in any Qt parent — no Orange dependency
- **`widgets/`** are Orange `OWBaseWidget` subclasses that compose `viewers/` and `grdl_rt`

### Key Dependencies

| Dependency | Purpose |
|-----------|---------|
| grdl-runtime | Execution framework (workflow engine, GPU backend, catalog, discovery) |
| grdl | Image I/O, processing algorithms, coregistration |
| numpy, scipy | Array operations |
| PyQt6 | Qt6 widget toolkit |
| orange3, orange-widget-base | Orange Canvas widget framework |
| napari | Stack viewer with polygon drawing |

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

### Execution & Catalog (from [grdl-runtime](../grdl-runtime))

These modules are imported by GRDK widgets from the `grdl_rt` package:

| Package | Key Types Used by Widgets |
|---------|--------------------------|
| `grdl_rt.execution.discovery` | `discover_processors()`, `get_processor_tags()` |
| `grdl_rt.execution.gpu` | `GpuBackend` |
| `grdl_rt.execution.workflow` | `WorkflowDefinition`, `ProcessingStep`, `WorkflowState` |
| `grdl_rt.execution.dsl` | `DslCompiler` |
| `grdl_rt.execution.chip` | `Chip`, `ChipSet`, `ChipLabel`, `PolygonRegion` |
| `grdl_rt.execution.tags` | `WorkflowTags`, `ProjectTags` |
| `grdl_rt.execution.project` | `GrdkProject` |
| `grdl_rt.execution.executor` | `WorkflowExecutor` |
| `grdl_rt.catalog.database` | `ArtifactCatalog` |
| `grdl_rt.catalog.models` | `Artifact`, `UpdateResult` |
| `grdl_rt.catalog.resolver` | `resolve_catalog_path()` |
| `grdl_rt.catalog.updater` | `ArtifactUpdateWorker` |
| `grdl_rt.catalog.pool` | `ThreadExecutorPool` |

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
| `ow_coregister.py` | Co-Register | ORB/SIFT feature-match registration (affine/homography) |
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

## Remote GUI Visualization

GRDK's GUI can be displayed on a remote machine (via SSH X11 forwarding) or from inside a Docker/Podman container. PyQt6 (Qt6) requires the X11/XCB platform plugin and associated libraries on the remote/container side, and a running X server on the host side.

### Prerequisites (Remote / Container Side)

Install the X11/XCB runtime libraries required by Qt6:

```bash
# Debian / Ubuntu
apt-get update && apt-get install -y \
    libxcb1 libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
    libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xfixes0 \
    libxcb-xinerama0 libxcb-xkb1 libxkbcommon-x11-0 libxkbcommon0 \
    libegl1 libgl1-mesa-glx libglib2.0-0 libfontconfig1 libdbus-1-3 \
    x11-utils

# RHEL / Fedora / Rocky
dnf install -y \
    libxcb xcb-util-cursor xcb-util-image xcb-util-keysyms \
    xcb-util-renderutil xcb-util-wm libxkbcommon-x11 libxkbcommon \
    mesa-libEGL mesa-libGL glib2 fontconfig dbus-libs xorg-x11-utils
```

### Environment Variables

Set these before launching GRDK:

```bash
# Force the XCB platform plugin (required for X11 forwarding)
export QT_QPA_PLATFORM=xcb

# Disable OpenGL for pure software rendering over X11 (avoids GLX errors)
export QT_QUICK_BACKEND=software
export LIBGL_ALWAYS_SOFTWARE=1

# If you see "Could not connect to display", verify DISPLAY is set:
echo $DISPLAY   # Should show e.g. :0 or localhost:10.0
```

### Option A: Native Linux — SSH X11 Forwarding

From your **local machine** (the one with the display):

```bash
# Connect with X11 forwarding enabled
ssh -X user@remote-host

# Or for trusted forwarding (faster, less restrictive):
ssh -Y user@remote-host
```

On the **remote host**:

```bash
# Verify DISPLAY is set (ssh -X sets it automatically)
echo $DISPLAY

# Set Qt platform and launch
export QT_QPA_PLATFORM=xcb
export QT_QUICK_BACKEND=software

# Install and run
pip install -e .
python -c "from PyQt6.QtWidgets import QApplication; print('PyQt6 OK')"  # Quick test
grdk-canvas  # Launch GRDK GUI
```

### Option B: Docker Container with X11 Forwarding

**Host setup** (run once per session on your local machine):

```bash
# Allow local Docker containers to access the X server
xhost +local:docker
```

**Run the container** with X11 socket and DISPLAY forwarded:

```bash
docker run -it \
    -e DISPLAY=$DISPLAY \
    -e QT_QPA_PLATFORM=xcb \
    -e QT_QUICK_BACKEND=software \
    -e LIBGL_ALWAYS_SOFTWARE=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --network=host \
    your-grdk-image:latest \
    bash
```

**Example Dockerfile**:

```dockerfile
FROM python:3.12-slim

# Install X11/XCB runtime dependencies for Qt6
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
    libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xfixes0 \
    libxcb-xinerama0 libxcb-xkb1 libxkbcommon-x11-0 libxkbcommon0 \
    libegl1 libgl1-mesa-glx libglib2.0-0 libfontconfig1 libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*

# Set Qt environment for X11 forwarding
ENV QT_QPA_PLATFORM=xcb
ENV QT_QUICK_BACKEND=software
ENV LIBGL_ALWAYS_SOFTWARE=1

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .

CMD ["grdk-canvas"]
```

Build and run:

```bash
docker build -t grdk-gui .
xhost +local:docker
docker run -it \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --network=host \
    grdk-gui
```

### Option C: Podman Container

Podman works similarly but uses `--userns=keep-id` for rootless operation:

```bash
xhost +local:
podman run -it \
    -e DISPLAY=$DISPLAY \
    -e QT_QPA_PLATFORM=xcb \
    -e QT_QUICK_BACKEND=software \
    -e LIBGL_ALWAYS_SOFTWARE=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --userns=keep-id \
    --network=host \
    grdk-gui
```

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Could not connect to display` | `DISPLAY` not set or X server not reachable | Verify `echo $DISPLAY`, use `ssh -X`, mount X11 socket |
| `qt.qpa.xcb: could not connect` | Missing xcb libraries or no X server | Install `libxcb*` packages (see Prerequisites) |
| `Could not load the Qt platform plugin "xcb"` | Missing Qt XCB platform dependencies | Install all `libxcb*` and `libxkbcommon*` packages |
| `GLX/OpenGL errors` | Hardware GL not available over X11 | Set `QT_QUICK_BACKEND=software` and `LIBGL_ALWAYS_SOFTWARE=1` |
| `Authorization required` | xhost not configured | Run `xhost +local:docker` on host |
| `No protocol specified` | Container user mismatch | Use `xhost +local:` or pass `--userns=keep-id` (Podman) |
| Blank/frozen window | Compositor issue over forwarding | Try `export QT_X11_NO_MITSHM=1` |
| Slow rendering | Network-bound X11 pixel transfer | Use SSH compression (`ssh -XC`) or consider VNC |

### Security Note

`xhost +local:docker` permits any local process running under the `docker` user to access your X server. For production or multi-user environments, prefer X11 cookie-based authentication:

```bash
# More secure alternative: share the X authority cookie
docker run -it \
    -e DISPLAY=$DISPLAY \
    -e XAUTHORITY=/tmp/.Xauthority \
    -v $XAUTHORITY:/tmp/.Xauthority:ro \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --network=host \
    grdk-gui
```

## License

MIT License. Copyright (c) 2026 geoint.org. See [LICENSE](LICENSE).
