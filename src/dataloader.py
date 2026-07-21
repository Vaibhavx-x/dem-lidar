import os
import math
import random
import torch
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from torch.utils.data import Dataset, DataLoader

# Global statistics for elevation normalization. 
# These should be updated with the actual training set mean and std.
GLOBAL_MEAN = 1321.6211
GLOBAL_STD = 453.2252

def load_hma_npy(path):
    """Load one HMA tensor file (.npy or .npz), tolerant of:

    Legacy .npy: a plain (4, H, W) float array -- dem_bic, lidar_raw, mask, gt_dem.

    Extended .npy: the same 4 channels plus per-file georeferencing metadata
    appended after them --
        Ch5: GDAL geotransform, shape (6,) float64, (x0, dx, 0, y0, 0, -dy)
        Ch6: EPSG code, single int.
    Since these extra fields don't share the (H, W) shape of the image
    channels, they're packed as a heterogeneous container (object-dtype
    array wrapping a dict, or a list/tuple of 6 items).

    .npz: an NpzFile of named arrays -- either 4 separate channel arrays
    (dem_bic/lidar_raw/mask/gt_dem, or Ch1..Ch4) plus optional geotransform/
    epsg, or a single combined (4,H,W) array under a common key name.

    Returns
    -------
    image : (4, H, W) float32 ndarray -- dem_bic, lidar_raw, mask, gt_dem.
    geotransform : (6,) float64 ndarray, or None if not present.
    epsg : int, or None if not present.
    """
    path_str = str(path)

    if path_str.endswith('.npz'):
        with np.load(path, allow_pickle=True) as npz:
            obj = {k: npz[k] for k in npz.files}
    else:
        try:
            raw = np.load(path, allow_pickle=False)
        except ValueError:
            # Legacy numpy raises ValueError here when the file actually needs
            # pickle support (i.e. it's the extended, object-dtype format).
            raw = np.load(path, allow_pickle=True)

        # ---- Legacy format: plain (4, H, W) numeric array, no metadata. ----
        if raw.dtype != object and raw.ndim == 3 and raw.shape[0] == 4:
            return raw.astype(np.float32, copy=False), None, None

        # ---- Extended format: object-dtype container (dict, or list/tuple
        #      of 6 items packed positionally). ----
        obj = raw
        if isinstance(raw, np.ndarray) and raw.dtype == object:
            obj = raw.item() if raw.shape == () else raw

    if isinstance(obj, dict):
        def _get(*names):
            for n in names:
                if n in obj:
                    return obj[n]
            return None

        # Some .npz exports may stash all 4 channels combined under one key
        # instead of 4 separate keys -- check that first.
        combined = _get('data', 'image', 'stack', 'tensor', 'arr_0')
        if combined is not None and np.asarray(combined).ndim == 3 and np.asarray(combined).shape[0] == 4:
            combined = np.asarray(combined, dtype=np.float32)
            dem_bic, lidar_raw, mask, gt_dem = combined[0], combined[1], combined[2], combined[3]
        else:
            dem_bic   = _get('dem_bic', 'Ch1', 'ch1')
            lidar_raw = _get('lidar_raw', 'Ch2', 'ch2')
            mask      = _get('mask', 'Ch3', 'ch3')
            gt_dem    = _get('gt_dem', 'Ch4', 'ch4')

        geotransform = _get('geotransform', 'gt', 'affine', 'Ch5', 'ch5')
        epsg         = _get('epsg', 'EPSG', 'Ch6', 'ch6')
    elif isinstance(obj, (list, tuple, np.ndarray)) and len(obj) >= 4:
        dem_bic, lidar_raw, mask, gt_dem = obj[0], obj[1], obj[2], obj[3]
        geotransform = obj[4] if len(obj) > 4 else None
        epsg = obj[5] if len(obj) > 5 else None
    else:
        raise ValueError(
            f'Unrecognized container for {path}: type={type(obj)}, '
            f'keys={list(obj.keys()) if isinstance(obj, dict) else "n/a"}. '
            f'Expected a plain (4,H,W) array, a dict, or a list/tuple of '
            f'image channels (+ optional geotransform/epsg).'
        )

    if dem_bic is None or lidar_raw is None or mask is None or gt_dem is None:
        raise ValueError(
            f'Could not find all 4 channels in {path}. '
            f'Keys present: {list(obj.keys()) if isinstance(obj, dict) else "n/a"} -- '
            f'edit the _get(...) name lists in load_hma_npy() to match your actual key names.'
        )

    image = np.stack([
        np.asarray(dem_bic, dtype=np.float32),
        np.asarray(lidar_raw, dtype=np.float32),
        np.asarray(mask, dtype=np.float32),
        np.asarray(gt_dem, dtype=np.float32),
    ], axis=0)

    geotransform = np.asarray(geotransform, dtype=np.float64) if geotransform is not None else None
    epsg = int(epsg) if epsg is not None else None

    return image, geotransform, epsg


class HMATensorDataset(Dataset):

    def __init__(
        self,
        data_paths,
        mode="train",
        train_crop=128,
        val_crop=256,
        val_overlap=64,
        val_pad=None
    ):
        self.mode = mode
        self.train_crop = train_crop
        self.val_crop = val_crop
        self.val_overlap = val_overlap
        self.val_pad = val_crop // 2 if val_pad is None else val_pad
        
        if isinstance(data_paths, str):
            data_paths = [data_paths]
            
        self.filepaths = []
        for path in data_paths:
            for root, _, files in os.walk(path):
                for f in files:
                    if f.endswith('.npy') or f.endswith('.npz'):
                        self.filepaths.append(os.path.join(root, f))
                        
        if self.mode == "train":
            random.shuffle(self.filepaths)
            print(f"[{self.mode.upper()} DATASET] Loaded {len(self.filepaths)} training files.")
            print("   -> Generating 2048x2048 Spatial Noise Cache in CPU RAM...")
            raw_noise = np.random.normal(0, 1.0, (2048, 2048)).astype(np.float32)
            smoothed_noise = gaussian_filter(raw_noise, sigma=2.0)
            self.noise_cache = (smoothed_noise / np.max(np.abs(smoothed_noise))) * 1.5

        elif self.mode == "val":
            self.filepaths = sorted(self.filepaths)
            print(f"[{self.mode.upper()} DATASET] Loaded {len(self.filepaths)} validation files.")

    def __len__(self):
        return len(self.filepaths)

    def _package_dg_vdsr(self, patch_data, compute_edt=True):
        """Transforms a raw (4, H, W) numpy slice into the standard DG-VDSR 6-pack."""
        dem_bic   = patch_data[0:1, :, :]  # Ch1: 10m Bicubic DEM
        lidar_raw = patch_data[1:2, :, :]  # Ch2: Raw ATL08 Elevation
        mask      = patch_data[2:3, :, :]  # Ch3: Photon existence mask
        gt_dem    = patch_data[3:4, :, :]  # Ch4: 10m Ground Truth DEM

        # 1. Zero-Center the dense Topo
        patch_mean = np.mean(dem_bic)
        dem_bic_centered = dem_bic - patch_mean
        gt_dem_centered  = gt_dem - patch_mean

        # Normalize patch mean for the confidence head (elevation context)
        norm_patch_mean = (patch_mean - GLOBAL_MEAN) / GLOBAL_STD

        # 2. LiDAR Delta (Invariant to mean subtraction)
        lidar_delta = mask * (lidar_raw - dem_bic)

        # 3. Channel 5: Euclidean Distance Transform (EDT)
        if compute_edt:
            if np.sum(mask) == 0:
                dist_map = np.full_like(dem_bic, 500.0, dtype=np.float32)
            else:
                dist_map = distance_transform_edt(1.0 - mask[0]).astype(np.float32)
                dist_map *= np.float32(10.0)
                dist_map = dist_map[np.newaxis, :, :]
        else:
            # Bypass wasted compute during validation overlapping patches
            dist_map = np.zeros_like(dem_bic)

        # .copy() prevents PyTorch "negative stride" warnings caused by np.flip()
        return {
            "dem_bic": torch.from_numpy(dem_bic_centered.copy()).float(),
            "lidar_delta": torch.from_numpy(lidar_delta.copy()).float(),
            "mask": torch.from_numpy(mask.copy()).float(),
            "dist_map": torch.from_numpy(dist_map.copy()).float(),
            "gt_dem": torch.from_numpy(gt_dem_centered.copy()).float(),
            "patch_mean": torch.tensor(patch_mean, dtype=torch.float32),
            "norm_patch_mean": torch.tensor(norm_patch_mean, dtype=torch.float32)
        }

    def __getitem__(self, idx):
        tensor_path = self.filepaths[idx]
        # geotransform/epsg (only present in the extended file format) aren't
        # needed for training -- the loader still strips them out cleanly so
        # `data` is always the (4, H, W) image array either way.
        data, _geotransform, _epsg = load_hma_npy(tensor_path)
        
        # ======================================================================
        # TRAIN MODE: Single dynamic augmented crop
        # ======================================================================
        if self.mode == "train":
            if data.ndim == 3:
                c, h, w = data.shape
            else:
                h, w = data.shape
                c = 1
            
            # Safely handle inputs smaller than train_crop to prevent ValueError
            pad_h = max(0, self.train_crop - h)
            pad_w = max(0, self.train_crop - w)
            if pad_h > 0 or pad_w > 0:
                data = np.pad(data, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
                _, h, w = data.shape
                
            max_x, max_y = w - self.train_crop, h - self.train_crop
            x0, y0 = np.random.randint(0, max_x + 1), np.random.randint(0, max_y + 1)
            
            data = data[:, y0:y0 + self.train_crop, x0:x0 + self.train_crop]
            
            k = np.random.randint(0, 4)
            if k > 0:
                data = np.rot90(data, k=k, axes=(1, 2)).copy()
            if np.random.rand() > 0.5: 
                data = np.flip(data, axis=2).copy()
            if np.random.rand() > 0.5: 
                data = np.flip(data, axis=1).copy()                          

            nx0, ny0 = np.random.randint(0, 2048 - self.train_crop), np.random.randint(0, 2048 - self.train_crop)
            data[0] += self.noise_cache[ny0:ny0+self.train_crop, nx0:nx0+self.train_crop]

            return self._package_dg_vdsr(data)
            
        # ======================================================================
        # VAL MODE: Return raw padded arrays + metadata only.
        # Patches are generated lazily by val_patch_generator() to avoid
        # materialising all ~2000+ patches at once (~2.6 GB per file).
        # ======================================================================
        elif self.mode == "val":
            c, h_orig, w_orig = data.shape
            pad = self.val_pad

            data_padded = np.pad(
                data,
                ((0, 0), (pad, pad), (pad, pad)),
                mode="reflect"
            )

            # Compute EDT on the padded photon mask once for the whole file
            mask_padded = data_padded[2:3]
            if np.sum(mask_padded) == 0:
                dist_map_padded = np.full_like(mask_padded, 500.0, dtype=np.float32)
            else:
                dist_map_padded = distance_transform_edt(1.0 - mask_padded[0]).astype(np.float32)
                dist_map_padded *= 10.0
                dist_map_padded = dist_map_padded[np.newaxis]

            _, h_pad, w_pad = data_padded.shape
            stride = self.val_crop - self.val_overlap

            y_starts = list(range(0, h_pad - self.val_crop + 1, stride))
            x_starts = list(range(0, w_pad - self.val_crop + 1, stride))
            if not y_starts or y_starts[-1] + self.val_crop < h_pad:
                y_starts.append(h_pad - self.val_crop)
            if not x_starts or x_starts[-1] + self.val_crop < w_pad:
                x_starts.append(w_pad - self.val_crop)

            # Build flat coordinate list (no patch tensors allocated yet)
            coords_list = [
                [y, x] for y in y_starts for x in x_starts
            ]

            return {
                # Raw padded arrays (numpy, not torch) -- kept in RAM once
                "data_padded":      data_padded,        # (4, H_pad, W_pad) float32
                "dist_map_padded":  dist_map_padded,    # (1, H_pad, W_pad) float32
                "coords_list":      coords_list,        # list of [y, x] int pairs
                # Scalars / small tensors
                "canvas_shape":     [h_pad, w_pad],
                "original_shape":   [h_orig, w_orig],
                "pad":              pad,
                "val_crop":         self.val_crop,
                # GT full image for final metric (1-D float32 numpy array)
                "gt_canvas_full":   data[3].copy(),     # (H_orig, W_orig)
            }

def create_dataloaders(train_dirs, val_dirs, batch_size=4, num_workers=8, prefetch_factor=4, pin_memory=True):
    train_dataset = HMATensorDataset(train_dirs, mode="train", train_crop=128)

    # TRAIN LOADER
    train_prefetch = prefetch_factor if num_workers > 0 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=train_prefetch,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    # VAL: expose dataset for direct iteration (no DataLoader collation needed)
    val_dataset = HMATensorDataset(val_dirs, mode="val", val_crop=256, val_overlap=192)

    return train_loader, val_dataset


def val_patch_generator(file_item, patch_batch_size=32):
    """Lazily yield mini-batches of patches from one validation file item.

    ``file_item`` is the dict returned by HMATensorDataset.__getitem__ in val
    mode.  Only ``patch_batch_size`` patches are ever live in RAM at once.

    Yields dicts with keys:
        dem_bic, lidar_delta, mask, dist_map, gt_dem  -- (B,1,256,256) float32 torch
        patch_mean                                    -- (B,)           float32 torch
        coords                                        -- (B,2)          int32   torch
    plus per-file scalar metadata on the first yield:
        canvas_shape, original_shape, pad, gt_canvas_full (passed as extra keys
        on every batch for simplicity; consumers only read them once).
    """
    data_padded     = file_item["data_padded"]      # (4, H_pad, W_pad)
    dist_map_padded = file_item["dist_map_padded"]  # (1, H_pad, W_pad)
    coords_list     = file_item["coords_list"]      # list of [y, x]
    val_crop        = file_item["val_crop"]

    n_patches = len(coords_list)
    n_batches = math.ceil(n_patches / patch_batch_size)

    # Pre-extract reused channels
    dem_bic_padded     = data_padded[0]          # (H_pad, W_pad)
    lidar_raw_padded   = data_padded[1]          # (H_pad, W_pad)
    mask_padded_2d     = data_padded[2]          # (H_pad, W_pad)
    gt_dem_padded      = data_padded[3]          # (H_pad, W_pad)

    for b in range(n_batches):
        start = b * patch_batch_size
        end   = min(start + patch_batch_size, n_patches)
        batch_coords = coords_list[start:end]
        bs = len(batch_coords)

        dem_bic_list     = []
        lidar_delta_list = []
        mask_list        = []
        dist_map_list    = []
        gt_dem_list      = []
        patch_mean_list  = []
        norm_patch_mean_list = []
        coord_tensors    = []

        for y, x in batch_coords:
            # Slice views (no copy until .copy() call below)
            dem_slice  = dem_bic_padded[y:y+val_crop, x:x+val_crop]
            mask_slice = mask_padded_2d[y:y+val_crop, x:x+val_crop]
            gt_slice   = gt_dem_padded[y:y+val_crop, x:x+val_crop]
            dist_slice = dist_map_padded[0, y:y+val_crop, x:x+val_crop]
            lidar_raw_slice = lidar_raw_padded[y:y+val_crop, x:x+val_crop]

            patch_mean    = float(dem_slice.mean())
            dem_centered  = dem_slice  - patch_mean
            gt_centered   = gt_slice   - patch_mean
            lidar_delta   = mask_slice * (lidar_raw_slice - dem_slice)
            
            # Normalize patch mean for the confidence head
            norm_patch_mean = (patch_mean - GLOBAL_MEAN) / GLOBAL_STD

            dem_bic_list.append(dem_centered[np.newaxis])       # (1,H,W)
            lidar_delta_list.append(lidar_delta[np.newaxis])
            mask_list.append(mask_slice[np.newaxis])
            dist_map_list.append(dist_slice[np.newaxis])
            gt_dem_list.append(gt_centered[np.newaxis])
            patch_mean_list.append(patch_mean)
            norm_patch_mean_list.append(norm_patch_mean)
            coord_tensors.append([y, x])

        yield {
            "dem_bic":       torch.from_numpy(np.stack(dem_bic_list).astype(np.float32)),
            "lidar_delta":   torch.from_numpy(np.stack(lidar_delta_list).astype(np.float32)),
            "mask":          torch.from_numpy(np.stack(mask_list).astype(np.float32)),
            "dist_map":      torch.from_numpy(np.stack(dist_map_list).astype(np.float32)),
            "gt_dem":        torch.from_numpy(np.stack(gt_dem_list).astype(np.float32)),
            "patch_mean":    torch.tensor(patch_mean_list, dtype=torch.float32),
            "norm_patch_mean": torch.tensor(norm_patch_mean_list, dtype=torch.float32),
            "coords":        torch.tensor(coord_tensors, dtype=torch.int32),
            # per-file metadata (same on every batch)
            "canvas_shape":  file_item["canvas_shape"],
            "original_shape": file_item["original_shape"],
            "pad":           file_item["pad"],
            "n_patches":     n_patches,
        }