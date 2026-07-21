# Data Directory

This directory is excluded from version control (see `.gitignore`).

## Expected Structure

```
data/
├── tensors/          # Scene-level .npz patches assembled by pipeline_complete.ipynb
│   ├── train/
│   └── val/
└── inference_output/ # GeoTIFF exports from symgate_infer_export.ipynb
```

## Tensor Format

Each `.npz` file contains:
- `arr` — shape `(4, H, W)`, float32
  - Channel 0: `dem_bic` — bicubic-resampled FABDEM (10 m)
  - Channel 1: `lidar_raw` — sparse ICESat-2 ATL08 elevations
  - Channel 2: `mask` — binary photon mask (1 = valid ATL08 photon)
  - Channel 3: `gt_dem` — ground truth bare-earth DEM
- `geotransform` — GDAL 6-tuple
- `epsg` — integer EPSG code

## Data Sources

- **FABDEM**: [fabdem.space](https://fabdem.space) — Forest And Buildings removed Copernicus DEM
- **ICESat-2 ATL08**: [NSIDC](https://nsidc.org/data/atl08) — Land and Vegetation Height product
- **Ground Truth DEM**: HMA DSM with WhiteboxTools bare-earth extraction

## Download Link

> Add your Google Drive / Earthdata link here
