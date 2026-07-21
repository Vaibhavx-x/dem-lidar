#!/usr/bin/env python3
"""
combine_predictions.py
======================
Post-model export / combine script  --  follows the agreed dataset contract.

INPUT  (per patch):  tensors/train/{NNNNN}_{tile}_pXXXXX.npz
    keys:
      tensor       : float32 (4, H, W)   Ch1 FABDEM, Ch2 ICESat-2, Ch3 mask, Ch4 GT
      geotransform : float64 (6,)        GDAL (x0, dx, 0, y0, 0, -dy)  [georef ONLY]
      epsg         : int                 e.g. 32643                    [georef ONLY]

SIBLING (visual only):  gt_visual/{NNNNN}_{tile}_pXXXXX.tif  (filled GT DEM)

OUTPUT : out_dir/{NNNNN}_{tile}_pXXXXX_pred.tif     (georeferenced prediction)
         out_dir/{NNNNN}_{tile}_pXXXXX_diff.tif     (|pred - gt|, with --write-diff)

ALIGNMENT RULES (from the contract)
  * The model NEVER sees Ch5/Ch6 (geotransform/epsg). It sees only the spatial
    channels you select (default Ch1 for the VDSR; Ch1-Ch3 per the contract).
  * Output georef comes ONLY from the .npz:  CRS = EPSG(epsg),
    transform = Affine(dx, 0, x0,  0, -dy, y0)  built from geotransform.
  * Variable H,W (do NOT assume 256). Output shape == tensor.shape[1:].
  * Pairing key = basename string only.

Examples
--------
  # VDSR (single-channel Ch1, per-patch zero-centering)
  python combine_predictions.py \
      --tensors-dir tensors/train --gt-visual-dir gt_visual --out-dir out \
      --weights weights/vdsr.pth --model-type vdsr \
      --in-channels 0 --norm vdsr --write-diff --device cuda

  # Contract 3-channel variant (Ch1-Ch3 input, no normalization)
  python combine_predictions.py \
      --tensors-dir tensors/train --out-dir out \
      --weights weights/net.pth --model-type generic \
      --model-module mymodel --model-class DEMNet \
      --in-channels 0,1,2 --norm none --device cuda

Note on EGM2008: the elevation VALUES are orthometric (EGM2008) by virtue of
your training data; that lives in the numbers, not in Ch5/Ch6. Put the EPSG
that describes the CRS -- e.g. 32643 (UTM) or a compound code like 9518
(WGS84 + EGM2008 height) -- straight into the .npz's `epsg` field.
"""
import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import torch
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS

# make local model modules (model_vdsr.py) importable
sys.path.insert(0, str(Path(__file__).resolve().parent))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(args):
    if args.model_type == "vdsr":
        from model_vdsr import BaselineVDSR
        net = BaselineVDSR(num_layers=args.num_layers, num_features=args.num_features)
    elif args.model_type == "generic":
        mod = __import__(args.model_module, fromlist=[args.model_class])
        net = getattr(mod, args.model_class)()
    else:
        raise ValueError(f"unknown model-type {args.model_type}")

    sd = torch.load(args.weights, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    net.load_state_dict(sd)
    return net.eval().to(args.device)


def predict_patch(model, tensor, args):
    """Run one patch -> (H,W) float32 prediction. Honors --in-channels / --norm."""
    in_channels = [int(x) for x in args.in_channels.split(",") if x != ""]
    in_ch = tensor[np.array(in_channels, dtype=int)].astype(np.float32)  # (Cin,H,W)
    dev = args.device

    if args.norm == "vdsr":
        assert len(in_channels) == 1, "vdsr uses a single input channel (Ch1)"
        pm = float(in_ch.mean())
        xc = (in_ch - pm)[None]                           # [1,1,H,W]  (in_ch is (1,H,W))
        with torch.no_grad():
            out = model(torch.from_numpy(xc).to(dev))
        out = np.squeeze(out.detach().cpu().numpy())
        if out.ndim == 3:
            out = out[0]
        return (out + pm).astype(np.float32)

    if args.norm == "none":
        x = in_ch[None]                                  # [1,Cin,H,W]
        with torch.no_grad():
            out = model(torch.from_numpy(x).to(dev))
        out = np.squeeze(out.detach().cpu().numpy())
        if out.ndim == 3:
            out = out[0]
        return out.astype(np.float32)

    if args.norm == "fixed":
        mean = np.array(args.in_mean, dtype=np.float32)[:, None, None]
        std = np.array(args.in_std, dtype=np.float32)[:, None, None]
        x = (in_ch - mean) / (std + 1e-8)
        x = x[None]                                      # [1,Cin,H,W]
        with torch.no_grad():
            out = model(torch.from_numpy(x).to(dev))
        out = np.squeeze(out.detach().cpu().numpy())
        if out.ndim == 3:
            out = out[0]
        return (out * args.out_std + args.out_mean).astype(np.float32)

    raise ValueError(f"unknown norm {args.norm}")


# --------------------------------------------------------------------------- #
# GeoTIFF I/O
# --------------------------------------------------------------------------- #
def write_geotiff(pred, path, geotransform, epsg):
    gt = np.asarray(geotransform, dtype=np.float64)
    # GDAL (x0,dx,0, y0,0,-dy) -> rasterio Affine(a,b,c, d,e,f)
    #   x = a*col + b*row + c = dx*col + x0
    #   y = d*col + e*row + f = -dy*row + y0
    transform = Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3])
    H, W = pred.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=H, width=W,
        count=1, dtype="float32", crs=CRS.from_epsg(int(epsg)),
        transform=transform, compress="deflate", tiled=True,
    ) as dst:
        dst.write(pred.astype(np.float32), 1)
        dst.set_band_description(1, "elevation_m")


def photon_count_of(stem):
    m = re.match(r"(\d+)_", stem)
    return int(m.group(1)) if m else -1


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Combine model predictions -> georeferenced GeoTIFFs")
    ap.add_argument("--tensors-dir", required=True)
    ap.add_argument("--gt-visual-dir", default=None, help="optional, for --write-diff")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model-type", default="vdsr", choices=["vdsr", "generic"])
    ap.add_argument("--model-module", default="model_vdsr")
    ap.add_argument("--model-class", default="BaselineVDSR")
    ap.add_argument("--num-layers", type=int, default=20)
    ap.add_argument("--num-features", type=int, default=64)
    ap.add_argument("--in-channels", default="0", help="comma list, e.g. '0' or '0,1,2'")
    ap.add_argument("--norm", default="vdsr", choices=["vdsr", "none", "fixed"])
    ap.add_argument("--in-mean", default=None, help="comma list for --norm fixed")
    ap.add_argument("--in-std", default=None, help="comma list for --norm fixed")
    ap.add_argument("--out-mean", type=float, default=0.0)
    ap.add_argument("--out-std", type=float, default=1.0)
    ap.add_argument("--min-photons", type=int, default=None)
    ap.add_argument("--max-photons", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="process at most N patches")
    ap.add_argument("--write-diff", action="store_true", help="also write |pred-gt| tif + MAE/RMSE")
    args = ap.parse_args()

    if args.in_mean:
        args.in_mean = [float(x) for x in args.in_mean.split(",")]
    if args.in_std:
        args.in_std = [float(x) for x in args.in_std.split(",")]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = Path(args.gt_visual_dir) if args.gt_visual_dir else None

    model = build_model(args)
    npz_files = sorted(glob.glob(str(Path(args.tensors_dir) / "*.npz")))
    print(f"Found {len(npz_files)} .npz files; model inputs = channels {args.in_channels}, norm={args.norm}")

    done = 0
    for npz_path in npz_files:
        stem = Path(npz_path).stem
        pc = photon_count_of(stem)
        if args.min_photons is not None and pc < args.min_photons:
            continue
        if args.max_photons is not None and pc > args.max_photons:
            continue

        z = np.load(npz_path)
        tensor = z["tensor"].astype(np.float32)
        geotransform = z["geotransform"]
        epsg = int(z["epsg"])
        H, W = tensor.shape[1], tensor.shape[2]

        pred = predict_patch(model, tensor, args)            # (H, W) float32

        out_path = out_dir / f"{stem}_pred.tif"
        write_geotiff(pred, out_path, geotransform, epsg)
        done += 1

        if gt_dir is not None and args.write_diff:
            gt_path = gt_dir / f"{stem}.tif"
            if gt_path.exists():
                with rasterio.open(gt_path) as ds:
                    gt = ds.read(1).astype(np.float32)
                diff = np.abs(pred - gt)
                write_geotiff(diff, out_dir / f"{stem}_diff.tif", geotransform, epsg)
                valid = np.isfinite(gt) & np.isfinite(pred)
                mae = float(np.abs(pred[valid] - gt[valid]).mean()) if valid.any() else float("nan")
                rmse = float(np.sqrt(((pred[valid] - gt[valid]) ** 2).mean())) if valid.any() else float("nan")
                print(f"  {stem}: H={H} W={W} EPSG={epsg}  MAE={mae:.3f}  RMSE={rmse:.3f}")
        else:
            print(f"  {stem}: wrote {out_path.name}  shape=({H},{W})  EPSG={epsg}")

        if args.limit and done >= args.limit:
            break

    print(f"Done. Wrote {done} prediction GeoTIFFs to {out_dir}")


if __name__ == "__main__":
    main()
