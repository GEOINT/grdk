# GRDK Development Guide

## Project Overview

GRDK (GEOINT Rapid Development Kit) is a GUI toolkit for CUDA-optimized image processing workflow orchestration. It is built as a set of Orange Data Mining add-on plugins on top of the GRDL library.

This project has two modes:
- **GEODEV Mode**: Interactive GEOINT development — image loading, co-registration, chipping, labeling, workflow orchestration with real-time GPU preview, and workflow publishing.
- **Administrative Mode**: Artifact catalog management — search/discover/download GRDL processors and GRDK workflows.

## Architecture

### Layers

1. **`grdk/core/`** — Non-GUI business logic (project model, workflow model, DSL compiler, GPU backend, tag taxonomy, executor). No Qt imports.
2. **`grdk/catalog/`** — SQLite catalog database, path resolution, update checking, thread pool. No Qt imports.
3. **`grdk/viewers/`** — Embeddable Qt widgets (napari stack viewer, chip gallery, polygon tools). Requires PyQt5.
4. **`grdk/widgets/`** — Orange3 OWWidget subclasses. Depends on `core/`, `catalog/`, and `viewers/`.

### Key Relationships

- GRDK depends on **GRDL** (`grdl` package) for all image I/O, geolocation, and processing operations.
- GRDK widgets use **custom signal types** (`grdk/widgets/_signals.py`) instead of Orange's default Table signals.
- GPU acceleration uses **CuPy** (numpy-compatible) for image transforms and **PyTorch** for ML model inference.
- The **DSL** supports both Python decorator syntax and YAML serialization, with bidirectional conversion.

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
- **Orange widgets** follow the `ow_<name>.py` naming convention.
- **Signal types** are defined in `_signals.py` and wrap core models.
- **Tests** use pytest, synthetic data, no real imagery files.

## Catalog Path Resolution

Priority order:
1. `GRDK_CATALOG_PATH` environment variable
2. `~/.grdl/config.json` → `"catalog_path"` field
3. `~/.grdl/catalog.db` (default)

## Testing

```bash
pytest tests/ -v
```

Tests are in `tests/` following `test_<module>.py` naming. Widget tests are in `tests/test_widgets/`.

## Git Practices

Same as GRDL: imperative commit messages, one change per commit, `<type>/<description>` branches.
