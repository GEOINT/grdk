# ImageCanvas API Reference

`grdk.viewers.image_canvas` — Interactive image viewer built on `QGraphicsView`.

## Overview

ImageCanvas is the shared base for all image display in GRDK. It renders numpy arrays with configurable visual enhancements (contrast, brightness, gamma, colormaps, dynamic range windowing) without modifying source data. All enhancements are applied as a pure rendering pipeline at display time.

Two classes are provided:

- **`ImageCanvas`** — Full interactive viewer with pan, zoom, and pixel hover
- **`ImageCanvasThumbnail`** — Non-interactive fixed-size variant for thumbnail grids

## DisplaySettings

Dataclass controlling how source arrays are mapped to display pixels.

```python
from grdk.viewers.image_canvas import DisplaySettings

settings = DisplaySettings(
    window_min=None,        # Manual min (None = auto from data)
    window_max=None,        # Manual max (None = auto from data)
    percentile_low=0.0,     # Auto percentile lower bound (0-100)
    percentile_high=100.0,  # Auto percentile upper bound (0-100)
    brightness=0.0,         # Additive offset [-1.0, 1.0]
    contrast=1.0,           # Multiplicative scale [0.0, 3.0]
    colormap='grayscale',   # 'grayscale', 'viridis', 'inferno', 'plasma', 'hot'
    band_index=None,        # Band for multi-band (None = auto RGB/gray)
    gamma=1.0,              # Gamma correction [0.1, 5.0]
)
```

### Window/Level Modes

- **Auto (default)**: `window_min=None, window_max=None` — uses full data range
- **Percentile**: `percentile_low=2, percentile_high=98` — clips outliers
- **Manual**: `window_min=100, window_max=500` — fixed display range

### Band Selection

For multi-band arrays (shape `(H, W, C)`):

- `band_index=None` — auto: 3-band as RGB, otherwise first band as grayscale
- `band_index=2` — display band 2 only (grayscale)

## normalize_array()

Pure function (no Qt dependency) that converts any numpy array to display-ready uint8.

```python
from grdk.viewers.image_canvas import normalize_array, DisplaySettings

arr = np.random.rand(512, 512).astype(np.float32)
display = normalize_array(arr, DisplaySettings(contrast=1.5, gamma=0.8))
# Returns: np.ndarray, dtype=uint8, shape (512, 512) or (512, 512, 3)
```

### Processing Pipeline

1. Complex arrays → `np.abs()`
2. Band selection (3D arrays)
3. Window/level normalization to [0, 1]
4. Contrast/brightness: `pixel = contrast * (pixel - 0.5) + 0.5 + brightness`
5. Gamma: `pixel = pixel ^ (1/gamma)`
6. Colormap application (grayscale → RGB via 256-entry LUT)
7. Scale to uint8 [0, 255]

### Supported Input Types

| Input | Handling |
|-------|----------|
| `float32/64` (H, W) | Direct normalization |
| `uint8/16` (H, W) | Cast to float, then normalize |
| `complex64/128` (H, W) | `np.abs()` → magnitude |
| `float` (H, W, 3) | RGB passthrough (no colormap) |
| `float` (H, W, C) where C > 3 | Auto: first 3 bands as RGB, or `band_index` |
| `float` (H, W, 1) | Squeeze to grayscale |

## ImageCanvas

Interactive `QGraphicsView` subclass.

```python
from grdk.viewers.image_canvas import ImageCanvas, DisplaySettings

canvas = ImageCanvas(parent=some_widget)
canvas.set_array(my_numpy_array)
canvas.set_display_settings(DisplaySettings(contrast=1.5))
canvas.fit_in_view()
```

### Methods

| Method | Description |
|--------|-------------|
| `set_array(arr)` | Set source numpy array and refresh display |
| `set_display_settings(s)` | Update display settings and re-render |
| `fit_in_view()` | Zoom to fit entire image in viewport |
| `zoom_to(factor)` | Set absolute zoom level (1.0 = 100%) |
| `reset_view()` | Reset pan, zoom, and settings to defaults |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `display_settings` | `DisplaySettings` | Current rendering settings |
| `source_array` | `Optional[np.ndarray]` | The source data, or None |

### Signals

| Signal | Arguments | When |
|--------|-----------|------|
| `pixel_hovered` | `(int, int, object)` | Mouse moves over image — emits (row, col, raw_value) |
| `zoom_changed` | `(float,)` | Zoom level changes |
| `display_settings_changed` | `(object,)` | Display settings updated |

### Mouse Interaction

| Input | Action |
|-------|--------|
| Drag | Pan the view |
| Scroll wheel | Zoom in/out (1.15x per step, centered on cursor) |
| Double-click | Fit image to viewport |

## ImageCanvasThumbnail

Non-interactive fixed-size subclass for use in thumbnail grids.

```python
from grdk.viewers.image_canvas import ImageCanvasThumbnail

thumb = ImageCanvasThumbnail(size=128, parent=gallery_widget)
thumb.set_array(chip_data)
```

- Fixed size: `setFixedSize(size, size)`
- No pan, zoom, or mouse interaction
- Auto-fits image on `set_array()` and `resizeEvent()`
- Transparent background, no frame
- Drop-in replacement for `QLabel + QPixmap`

## build_display_controls()

Convenience function that builds a pre-wired QGroupBox of display adjustment controls.

```python
from grdk.widgets._display_controls import build_display_controls

# In an Orange widget's __init__:
controls = build_display_controls(
    parent=self.controlArea,
    canvas=self._canvas,
    show=['contrast', 'brightness', 'gamma', 'colormap'],
)
self.controlArea.layout().addWidget(controls)
```

### Available Controls

| Name | Widget | Range |
|------|--------|-------|
| `window` | Two QDoubleSpinBox + Auto checkbox | Any float |
| `percentile` | Two QDoubleSpinBox | 0-100% |
| `contrast` | QSlider | 0.0 - 3.0 |
| `brightness` | QSlider | -1.0 - 1.0 |
| `gamma` | QDoubleSpinBox | 0.1 - 5.0 |
| `colormap` | QComboBox | grayscale, viridis, inferno, plasma, hot |
| `band` | QSpinBox | -1 (auto) to 255 |

The `show` parameter selects which controls to include. Default is all.

## Colormaps

Five built-in colormaps implemented as 256x3 uint8 lookup tables (no matplotlib dependency):

| Name | Description |
|------|-------------|
| `grayscale` | Identity mapping (no LUT applied) |
| `viridis` | Perceptually uniform blue-green-yellow |
| `inferno` | Dark purple-red-yellow |
| `plasma` | Purple-pink-orange-yellow |
| `hot` | Black-red-yellow-white |

Applied via numpy indexing: `rgb = lut[grayscale_array]`.

## Extensibility

ImageCanvas is built on `QGraphicsScene`, making future additions straightforward:

| Feature | Implementation |
|---------|---------------|
| Histogram overlay | `QGraphicsItem` subclass added to scene |
| Measurement tools | `QGraphicsLineItem` with distance labels |
| Annotations | `QGraphicsTextItem` / `QGraphicsPathItem` |
| Crosshair inspector | `QGraphicsLineItem` pair following mouse |
| ROI selection | `QGraphicsRectItem` with rubber band |
| Layer compositing | Multiple `QGraphicsPixmapItem`s with opacity |

None of these require changes to the ImageCanvas core — they plug into the scene's item model.
