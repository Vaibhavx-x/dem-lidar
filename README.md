# LiDAR-Guided DEM Super-Resolution for High Mountain Asia

> **ICESat-2 ATL08 photon anchoring + deep learning to sharpen coarse FABDEM terrain models over the Himalaya.**

---

## Overview

This project improves the spatial resolution and vertical accuracy of freely available 30 m FABDEM digital elevation models (DEMs) by fusing them with sparse ICESat-2 ATL08 LiDAR photon measurements using a dual-stream convolutional neural network.

**Key idea:** FABDEM provides smooth, continuous terrain coverage but loses fine-scale topographic detail in complex mountain terrain. ICESat-2 provides millimetre-precision elevation at discrete photon tracks. This network learns to *anchor* super-resolved DEM predictions to the known photon elevations while in-painting between tracks.

---

## Model Architecture: SymGate-VDSR

```
Input (4 channels)
  ├─ dem_bic      : Bicubic-resampled FABDEM (10 m)
  ├─ lidar_raw    : Sparse ICESat-2 ATL08 elevations
  ├─ mask         : Binary photon mask
  └─ edt          : Euclidean distance transform to nearest photon

       ┌──────────────────────────────────────────────────────┐
       │                 SymGate-VDSR                         │
       │                                                      │
       │  Stream A (Geomorphology Backbone)                   │
       │  ─── 18-layer residual VDSR ───────────────────────► │
       │                                  SymmetricGated      │
       │  Stream B (Anchor Radiator)       Fusion ──────────► │ Predicted DEM
       │  ─── sparse LiDAR encoder ────► (per-channel) ────► │
       │                                                      │
       │  CSPNRefine: iterative propagation + soft anchor pin │
       └──────────────────────────────────────────────────────┘
```

**Key design decisions:**
- `SymmetricGatedFusion`: per-channel additive gating at multiple depths — prevents gradient starvation in the sparse LiDAR branch
- `CSPNRefine`: propagates the coarse prediction using a learned affinity matrix, then softly blends with a confidence-weighted anchor at ATL08 photon locations
- `DistanceGatedTopoLoss`: combines L1 elevation, slope/curvature smoothness penalties, and a pin loss that applies extra weight at photon locations, weighted by distance via EDT

---

## Repository Layout

```
dem-lidar-hma/
│
├── src/                        # Reusable Python modules
│   ├── dataloader.py           # HMATensorDataset — 6-channel 32×32 patch loading
│   ├── dataloader_baseline.py  # Simple no-anchor baseline dataloader
│   ├── hann_merger.py          # HannStreamMerger — smooth tile stitching
│   ├── model_vdsr_baseline.py  # v0 — single-stream VDSR (no LiDAR)
│   ├── model_dg_vdsr.py        # v1 — DG-VDSR dual-stream with scalar trust gate
│   └── model_symgatevdsr.py    # v2 — SymGate-VDSR (current best)
│
├── notebooks/
│   ├── 01_data_pipeline/       # ICESat-2 download, FABDEM resampling, tensor assembly
│   ├── 02_data_preparation/    # Validation patch extraction
│   ├── 03_training/            # Training notebooks (main: train_symgate_vdsr_new.ipynb)
│   ├── 04_inference/           # Full-scene inference + GeoTIFF export
│   └── 05_analysis/            # Model diagnostics, trust gate analysis
│
├── scripts/
│   ├── combine_predictions.py  # Batch inference → GeoTIFF export
│   └── symgate_infer_export.py # Full-scene SymGate inference pipeline
│
├── experiments/                # Archived earlier architectures and training variants
│   ├── model_symgatevdsr_next.py   # In-progress improved model (WIP)
│   ├── model_resunet.py            # Alternative ResUNet architecture
│   └── *.ipynb                     # Earlier training notebooks (DG-VDSR, curriculum)
│
├── data/                       # Gitignored — see data/README.md for format & download
├── checkpoints/                # Gitignored — see checkpoints/README.md for best model link
├── requirements.txt
└── .gitignore
```

---

## Pipeline

```
1. Download ICESat-2 ATL08       notebooks/01_data_pipeline/ICESat-2_Downloader.ipynb
2. Assemble 4-channel tensors    notebooks/01_data_pipeline/pipeline_complete.ipynb
3. Extract validation patches    notebooks/02_data_preparation/extract_val_patches.ipynb
4. Train model                   notebooks/03_training/train_symgate_vdsr_new.ipynb
5. Run inference                 notebooks/04_inference/symgate_infer_export.ipynb
6. (Optional) Batch export       scripts/symgate_infer_export.py
```

---

## Setup

```bash
git clone https://github.com/<your-username>/dem-lidar-hma.git
cd dem-lidar-hma
pip install -r requirements.txt
```

Download data and best checkpoint — see [`data/README.md`](data/README.md) and [`checkpoints/README.md`](checkpoints/README.md).

---

## Quick Inference

```python
import sys
sys.path.insert(0, "src")

import torch
import numpy as np
from model_symgatevdsr import SymGateVDSR
from hann_merger import HannStreamMerger

# Load model
model = SymGateVDSR()
ckpt = torch.load("checkpoints/symgate_vdsr_best.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Load scene
scene = np.load("data/tensors/val/scene_001.npz")
# See notebooks/04_inference/symgate_infer_export.ipynb for full tiling pipeline
```

---

## Model Evolution

| Version | File | Key Change |
|---|---|---|
| v0 Baseline | `model_vdsr_baseline.py` | Single-stream VDSR, no LiDAR input |
| v1 DG-VDSR | `model_dg_vdsr.py` | Dual-stream + scalar trust gate `alpha` |
| v2 SymGate-VDSR | `model_symgatevdsr.py` | Per-channel symmetric gating + CSPNRefine |
| v3 (WIP) | `experiments/model_symgatevdsr_next.py` | In progress |

---

## Data Sources

| Source | Description |
|---|---|
| [FABDEM](https://fabdem.space) | 30 m global bare-earth DEM (Forest/Building removed) |
| [ICESat-2 ATL08](https://nsidc.org/data/atl08) | Land & Vegetation Height — sparse photon elevations |
| HMA DSM | High Mountain Asia DSM, geoid-corrected + WBT bare-earth |
