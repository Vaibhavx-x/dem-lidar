#!/usr/bin/env python3
"""
SymGate-VDSR inference + GeoTIFF export for NPZ scene files.

This script implements the user's post-model export / combine contract:

Input per sample (.npz)
-----------------------
Keys:
  tensor:       float32 (4, H, W)
                Ch1 = baseline / bicubic DEM
                Ch2 = raw ICESat-2 / LiDAR elevations (0 where absent)
                Ch3 = photon mask (0/1)
                Ch4 = GT DEM (ignored at inference)
  geotransform: float64 (6,) in GDAL order (x0, dx, rx, y0, ry, -dy)
  epsg:         integer EPSG code

Pipeline
--------
1. Load scene from .npz.
2. Reflect-pad the scene.
3. Tile into overlapping patches.
4. For each patch:
      - zero-center Ch1 by its patch mean
      - build lidar_delta = mask * (lidar_raw - dem_bic)
      - build distance transform in metres from the mask
5. Run SymGateVDSR.
6. Stitch centered predictions with Hann overlap-add.
7. Un-center via stored patch means inside HannStreamMerger.
8. Crop padding back to original extent.
9. Write a single-band GeoTIFF with the input scene georeferencing.

Notes
-----
- Ch4 (GT) is never fed into the network.
- geotransform / epsg are used only for GeoTIFF export.
- Output filename is <stem>_pred.tif by default.
- The checkpoint may be either:
    * a wrapped training checkpoint with key 'model_state_dict', or
    * a raw state_dict.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

import torch
from rasterio.crs import CRS
from rasterio.transform import Affine
import rasterio

from model_symgatevdsr_old import SymGateVDSR
from hann_merge import HannStreamMerger


DEFAULT_EMPTY_DISTANCE_METRES = 500.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SymGate-VDSR on NPZ scene files and export GeoTIFF predictions."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to .pt/.pth checkpoint (wrapped checkpoint or raw state_dict).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to one .npz file or a directory containing .npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write <stem>_pred.tif outputs.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Sliding-window patch size used for inference and Hann merge.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=192,
        help="Overlap in pixels between adjacent inference patches.",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=None,
        help="Reflect padding on each side before tiling. Default: patch_size // 2.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of patches per inference batch.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Load checkpoint state_dict strictly. Default is non-strict=False for robustness.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples whose output GeoTIFF already exists.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use torch autocast on CUDA for faster inference.",
    )
    return parser.parse_args()


def collect_npz_files(path: Path) -> List[Path]:
    if path.is_file():
        if path.suffix.lower() != ".npz":
            raise ValueError(f"Input file must be .npz, got: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(path.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under: {path}")
    return files


def gdal_to_affine(geotransform: Sequence[float]) -> Affine:
    gt = np.asarray(geotransform, dtype=np.float64).reshape(6)
    return Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3])


def infer_pixel_size_metres(geotransform: Sequence[float]) -> float:
    gt = np.asarray(geotransform, dtype=np.float64).reshape(6)
    dx = abs(float(gt[1]))
    dy = abs(float(gt[5]))
    if dx > 0 and dy > 0:
        return float((dx + dy) / 2.0)
    if dx > 0:
        return dx
    if dy > 0:
        return dy
    return 10.0


def load_scene(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path) as z:
        tensor = z["tensor"].astype(np.float32)
        geotransform = z["geotransform"].astype(np.float64)
        epsg = int(z["epsg"])

    if tensor.ndim != 3 or tensor.shape[0] < 3:
        raise ValueError(
            f"Expected tensor with shape (4, H, W) or at least 3 channels, got {tensor.shape}"
        )

    return {
        "tensor": tensor,
        "geotransform": geotransform,
        "epsg": epsg,
    }


def build_padded_scene(
    tensor: np.ndarray,
    geotransform: Sequence[float],
    pad: int,
    empty_distance_metres: float = DEFAULT_EMPTY_DISTANCE_METRES,
) -> Dict[str, np.ndarray]:
    """
    Create reflect-padded scene arrays and a full-scene EDT in metres.
    """
    if tensor.ndim != 3:
        raise ValueError(f"Expected tensor shape (C, H, W), got {tensor.shape}")

    data_padded = np.pad(tensor, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    mask_padded = data_padded[2:3]

    if np.sum(mask_padded) == 0:
        dist_map_padded = np.full_like(mask_padded, empty_distance_metres, dtype=np.float32)
    else:
        pixel_size = infer_pixel_size_metres(geotransform)
        dist_px = distance_transform_edt(1.0 - mask_padded[0]).astype(np.float32)
        dist_map_padded = (dist_px * np.float32(pixel_size))[np.newaxis, :, :]

    return {
        "data_padded": data_padded.astype(np.float32, copy=False),
        "dist_map_padded": dist_map_padded.astype(np.float32, copy=False),
        "pad": int(pad),
    }


def make_coords(height: int, width: int, patch_size: int, overlap: int) -> List[Tuple[int, int]]:
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    if overlap < 0 or overlap >= patch_size:
        raise ValueError("overlap must satisfy 0 <= overlap < patch_size")

    stride = patch_size - overlap
    y_starts = list(range(0, height - patch_size + 1, stride))
    x_starts = list(range(0, width - patch_size + 1, stride))

    if not y_starts or y_starts[-1] + patch_size < height:
        y_starts.append(height - patch_size)
    if not x_starts or x_starts[-1] + patch_size < width:
        x_starts.append(width - patch_size)

    return [(y, x) for y in y_starts for x in x_starts]


def iter_patch_batches(
    data_padded: np.ndarray,
    dist_map_padded: np.ndarray,
    coords: Sequence[Tuple[int, int]],
    patch_size: int,
    batch_size: int,
) -> Iterator[Dict[str, torch.Tensor]]:
    dem_bic_padded = data_padded[0]
    lidar_raw_padded = data_padded[1]
    mask_padded = data_padded[2]

    n_patches = len(coords)
    n_batches = math.ceil(n_patches / batch_size)

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_patches)
        batch_coords = coords[start:end]

        dem_bic_list = []
        lidar_delta_list = []
        mask_list = []
        dist_map_list = []
        patch_mean_list = []
        coord_list = []

        for y, x in batch_coords:
            dem_slice = dem_bic_padded[y : y + patch_size, x : x + patch_size]
            lidar_raw_slice = lidar_raw_padded[y : y + patch_size, x : x + patch_size]
            mask_slice = mask_padded[y : y + patch_size, x : x + patch_size]
            dist_slice = dist_map_padded[0, y : y + patch_size, x : x + patch_size]

            patch_mean = float(dem_slice.mean())
            dem_centered = dem_slice - patch_mean
            lidar_delta = mask_slice * (lidar_raw_slice - dem_slice)

            dem_bic_list.append(dem_centered[np.newaxis])
            lidar_delta_list.append(lidar_delta[np.newaxis])
            mask_list.append(mask_slice[np.newaxis])
            dist_map_list.append(dist_slice[np.newaxis])
            patch_mean_list.append(patch_mean)
            coord_list.append([y, x])

        yield {
            "dem_bic": torch.from_numpy(np.stack(dem_bic_list).astype(np.float32)),
            "lidar_delta": torch.from_numpy(np.stack(lidar_delta_list).astype(np.float32)),
            "mask": torch.from_numpy(np.stack(mask_list).astype(np.float32)),
            "dist_map": torch.from_numpy(np.stack(dist_map_list).astype(np.float32)),
            "patch_mean": torch.tensor(patch_mean_list, dtype=torch.float32),
            "coords": torch.tensor(coord_list, dtype=torch.int32),
        }


def strip_dataparallel_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if state_dict and all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def load_model(checkpoint_path: Path, device: str = "cpu", strict: bool = False) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    if not isinstance(state_dict, dict):
        raise TypeError(
            "Checkpoint does not contain a valid state_dict or model_state_dict dictionary."
        )

    state_dict = strip_dataparallel_prefix(state_dict)

    model = SymGateVDSR()
    incompat = model.load_state_dict(state_dict, strict=strict)

    if not strict:
        missing = list(getattr(incompat, "missing_keys", []))
        unexpected = list(getattr(incompat, "unexpected_keys", []))
        if missing:
            print(f"[WARN] Missing keys while loading checkpoint: {missing}")
        if unexpected:
            print(f"[WARN] Unexpected keys while loading checkpoint: {unexpected}")

    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def predict_scene(
    model: torch.nn.Module,
    tensor: np.ndarray,
    geotransform: Sequence[float],
    patch_size: int = 256,
    overlap: int = 192,
    pad: int | None = None,
    batch_size: int = 16,
    device: str = "cpu",
    amp: bool = False,
) -> np.ndarray:
    if pad is None:
        pad = patch_size // 2

    _, h_orig, w_orig = tensor.shape
    padded = build_padded_scene(tensor=tensor, geotransform=geotransform, pad=pad)
    data_padded = padded["data_padded"]
    dist_map_padded = padded["dist_map_padded"]

    _, h_pad, w_pad = data_padded.shape
    coords = make_coords(h_pad, w_pad, patch_size=patch_size, overlap=overlap)

    merger = HannStreamMerger(
        canvas_shape=(h_pad, w_pad),
        patch_size=patch_size,
        device=device,
        pad=pad,
        original_shape=(h_orig, w_orig),
    )

    use_amp = amp and str(device).startswith("cuda")
    autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"

    for batch in iter_patch_batches(
        data_padded=data_padded,
        dist_map_padded=dist_map_padded,
        coords=coords,
        patch_size=patch_size,
        batch_size=batch_size,
    ):
        dem_bic = batch["dem_bic"].to(device, non_blocking=True)
        lidar_delta = batch["lidar_delta"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        dist_map = batch["dist_map"].to(device, non_blocking=True)

        with torch.autocast(device_type=autocast_device, enabled=use_amp):
            outputs = model(dem_bic, lidar_delta, mask, dist_map)
            pred_centered = outputs["dem_pred"]

        merger.add_batch(
            preds_centered=pred_centered,
            patch_means=batch["patch_mean"],
            coords=batch["coords"],
        )

    pred = merger.get_final_dem(as_tensor=False).astype(np.float32)
    if pred.shape != (h_orig, w_orig):
        raise RuntimeError(
            f"Merged prediction shape {pred.shape} does not match original scene {(h_orig, w_orig)}"
        )
    return pred


def write_geotiff(
    out_path: Path,
    pred: np.ndarray,
    geotransform: Sequence[float],
    epsg: int,
    compress: str = "deflate",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    transform = gdal_to_affine(geotransform)
    h, w = pred.shape

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(int(epsg)),
        transform=transform,
        compress=compress,
    ) as dst:
        dst.write(pred.astype(np.float32), 1)


def output_path_for(npz_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{npz_path.stem}_pred.tif"


def run_one(
    npz_path: Path,
    model: torch.nn.Module,
    output_dir: Path,
    patch_size: int,
    overlap: int,
    pad: int | None,
    batch_size: int,
    device: str,
    amp: bool,
    skip_existing: bool,
) -> Path:
    out_path = output_path_for(npz_path, output_dir)
    if skip_existing and out_path.exists():
        print(f"[SKIP] {out_path.name}")
        return out_path

    sample = load_scene(npz_path)
    tensor = sample["tensor"]
    geotransform = sample["geotransform"]
    epsg = sample["epsg"]

    print(
        f"[RUN] {npz_path.name} | scene_shape={tuple(tensor.shape)} | "
        f"epsg={epsg} | out={out_path.name}"
    )

    pred = predict_scene(
        model=model,
        tensor=tensor,
        geotransform=geotransform,
        patch_size=patch_size,
        overlap=overlap,
        pad=pad,
        batch_size=batch_size,
        device=device,
        amp=amp,
    )
    write_geotiff(out_path, pred, geotransform=geotransform, epsg=epsg)
    return out_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device=args.device, strict=args.strict)
    npz_files = collect_npz_files(args.input)

    print(f"[INFO] Found {len(npz_files)} NPZ file(s)")
    print(f"[INFO] Device: {args.device}")
    print(f"[INFO] Patch size: {args.patch_size}, overlap: {args.overlap}, pad: {args.pad}")

    written = []
    for npz_path in npz_files:
        out_path = run_one(
            npz_path=npz_path,
            model=model,
            output_dir=output_dir,
            patch_size=args.patch_size,
            overlap=args.overlap,
            pad=args.pad,
            batch_size=args.batch_size,
            device=args.device,
            amp=args.amp,
            skip_existing=args.skip_existing,
        )
        written.append(out_path)

    print(f"[DONE] Wrote {len(written)} GeoTIFF file(s) to: {output_dir}")


if __name__ == "__main__":
    main()
