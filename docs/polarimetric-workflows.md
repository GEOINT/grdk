# Polarimetric SAR Processing Workflows

This guide covers quad-pol SAR processing in GRDK using the Orange Canvas interface, from raw scattering matrices to Pauli RGB composites.

## Overview

Polarimetric SAR (PolSAR) captures the full scattering matrix at each pixel, revealing physical properties of scatterers through multi-polarization measurements. GRDK provides two widgets for PolSAR processing:

1. **OWCovarianceMatrix** — computes spatially-averaged polarimetric matrices (T3 or C3)
2. **OWPauliDecomposer** — generates Pauli-basis RGB composites from T3 matrices

## Prerequisites

### Data Requirements

- **Quad-pol SAR data** with all four polarizations: HH, HV, VH, VV
- Supported formats:
  - **BIOMASS L1** HDF5 products (ESA P-band SAR)
  - **NISAR RSLC** with `polarizations='all'` mode
  - **Custom multi-band readers** with `channel_metadata` containing polarization tags

### Assumptions

- **Reciprocal scattering** (monostatic radar): S_HV = S_VH
- **Single-look complex (SLC)** data preserving phase information
- **Spatial stationarity** within the averaging window

## Theory: Coherency Matrix and Pauli Decomposition

### The Pauli Target Vector

For quad-pol reciprocal scatter, the Pauli target vector is:

```
k_P = [(S_HH + S_VV), (S_HH - S_VV), 2·S_HV]^T / √2
```

This basis decomposes scatter into three physical mechanisms:

- **k_P[0]** — Odd-bounce (surface scattering from rough surfaces)
- **k_P[1]** — Double-bounce (dihedral scattering from vertical structures)
- **k_P[2]** — Volume scattering (depolarization from vegetation canopy)

### The T3 Coherency Matrix

The coherency matrix is the spatially-averaged outer product:

```
T3 = <k_P · k_P^H>
```

where `< · >` denotes spatial averaging via a boxcar window.

**T3 is 3×3 complex Hermitian** with structure:

```
T3 = | T11   T12   T13 |
     | T12*  T22   T23 |
     | T13*  T23*  T33 |
```

The diagonal elements are **real-valued powers**:

- `T3[0,0]` — Surface scatter power
- `T3[1,1]` — Double-bounce scatter power
- `T3[2,2]` — Volume scatter power

### Pauli RGB Composite

The Pauli decomposition assigns diagonal powers to color channels:

- **Red** = T3[1,1] (double-bounce / dihedral)
- **Green** = T3[2,2] (volume / depolarization)
- **Blue** = T3[0,0] (surface / odd-bounce)

**Physical interpretation:**
- Urban areas (buildings, corner reflectors) → bright red
- Forests (volume scatter) → green
- Bare fields, water (surface scatter) → blue

## Workflow: Quad-Pol to Pauli RGB

### Step 1: Load Quad-Pol Data

1. Add **Image Loader** widget to canvas
2. Click "Add Images..." and select your quad-pol product:
   - BIOMASS: select the HDF5 file or product directory
   - NISAR: select RSLC HDF5 with all polarizations
3. Verify the loader shows 4 images (HH, HV, VH, VV) or 1 multi-band image

**Validation:** Check that all four polarizations are present and dimensions match.

### Step 2: Compute T3 Coherency Matrix

1. Add **Covariance Matrix** widget
2. Connect Image Loader → Covariance Matrix
3. Configure parameters:
   - **Matrix Type:** T3 (required for Pauli decomposition)
   - **Window Size:** 7 (default), adjust based on speckle vs. resolution trade-off
     - Smaller (3-5): preserves spatial detail, higher speckle
     - Larger (9-15): smoother, loss of fine detail
4. Click "Compute"

The widget will:
- Extract HH, HV, VH, VV channels
- Optionally downsample for faster processing
- Compute the 3×3 complex T3 matrix at each pixel
- Package result as `CovarianceMatrixSignal`

**Processing time:** ~10-30 seconds for 1000×1000 pixel scene (CPU), faster with GPU.

### Step 3: Generate Pauli RGB

1. Add **Pauli Decomposer** widget
2. Connect Covariance Matrix → Pauli Decomposer
3. Configure representation:
   - **Power** (default): linear power scale
   - **Amplitude**: square root of power
   - **Decibel (dB)**: 10·log₁₀(power)
4. Adjust percentile clipping:
   - **Low**: 2 (clip dark outliers)
   - **High**: 98 (clip bright outliers)
5. Click "Decompose"

Output: RGB composite as an `ImageStack` signal (single-reader stack)

### Step 4: Visualize

1. Add **Stack Viewer** widget
2. Connect Pauli Decomposer → Stack Viewer
3. The RGB composite displays automatically
4. Use display controls to adjust:
   - Contrast/brightness
   - Gamma correction
   - Save as PNG or GeoTIFF

## YAML Workflow Example

For headless batch processing, save the workflow as YAML:

```yaml
name: "Pauli RGB from Quad-Pol"
description: "BIOMASS or NISAR quad-pol → T3 → Pauli decomposition"

steps:
  - processor: CoherencyMatrix
    version: "1.0"
    params:
      matrix_type: T3
      window_size: 7

  - processor: PauliDecomposition
    version: "1.0"
    params:
      representation: power
      percentile_low: 2.0
      percentile_high: 98.0
```

Run via command line:

```bash
python -m grdk pauli_workflow.yaml \
  --input biomass_product.h5 \
  --output pauli_rgb.tif
```

## Validation Checklist

Before running the workflow, verify:

- [ ] **All four polarizations present** (HH, HV, VH, VV)
- [ ] **Dimensions match** across all channels (use `validate_image_stack()`)
- [ ] **Complex data** (not magnitude-detected — check `dtype`)
- [ ] **SLC data** (not multi-looked GRD products)
- [ ] **Reciprocal scatter assumption** valid (monostatic radar)
- [ ] **Window size appropriate** for scene characteristics:
  - Agricultural fields: 5-7 (preserve field boundaries)
  - Forests: 9-15 (smooth speckle, stable statistics)
  - Urban: 3-5 (preserve building edges)

## Common Issues

### "Not quad-pol" Error

**Symptom:** OWCovarianceMatrix rejects input with "Stack is not quad-pol"

**Cause:** Missing polarization channels or reader doesn't expose `channel_metadata`

**Solution:**
1. Check Image Loader — should show 4 readers or 1 reader with 4 bands
2. Verify polarization tags: use `channel_pol_map(reader)` in Python console
3. For NISAR, ensure you opened with `polarizations='all'` mode

### Excessive Speckle in Output

**Symptom:** Pauli RGB looks noisy, difficult to interpret

**Solution:**
- Increase `window_size` in OWCovarianceMatrix (try 11 or 15)
- Apply spatial filtering to T3 before decomposition
- Use decibel representation in Pauli Decomposer (compresses dynamic range)

### Washed-Out Colors

**Symptom:** RGB composite lacks color saturation

**Solution:**
- Adjust percentile clipping (try 5-95 instead of 2-98)
- Switch to amplitude representation (gentler than power)
- Manually set contrast/brightness in Stack Viewer

### Dimension Mismatch

**Symptom:** Error during T3 computation about shape incompatibility

**Solution:**
- Run `validate_image_stack(stack)` before processing
- Check for co-registration issues if using multi-temporal data
- Ensure all channels use same imaging mode (same swath for Sentinel-1)

## Advanced: C3 vs T3

GRDK supports two matrix types:

| Matrix | Basis | Use Case |
|--------|-------|----------|
| **T3** | Pauli (k_P) | Physical decomposition (surface, double-bounce, volume) |
| **C3** | Lexicographic (k_L) | Scattering model fitting, compact-pol simulation |

**For Pauli RGB, always use T3.** C3 diagonal elements don't correspond to the same physical mechanisms.

## References

- **Cloude, S.R. and Pottier, E.** (1996). "A review of target decomposition theorems in radar polarimetry." *IEEE TGRS* 34(2):498-518.
- **Lee, J.S. and Pottier, E.** (2009). *Polarimetric Radar Imaging: From Basics to Applications.* CRC Press.
- **Moreira, A., et al.** (2013). "A tutorial on synthetic aperture radar." *IEEE GRSL* 1(1):6-43.

## See Also

- [Image Canvas API](image-canvas.md) — Display controls and color mapping
- [Architecture Guide](architecture.md) — Signal flow and widget layer boundaries
- `grdk/widgets/geodev/ow_covariance_matrix.py` — T3/C3 computation implementation
- `grdk/widgets/geodev/ow_pauli_decomposer.py` — Pauli RGB generation
