import os
import random
import torch
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from torch.utils.data import Dataset, DataLoader

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
                    if f.endswith('.npy'):
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
            "patch_mean": torch.tensor(patch_mean, dtype=torch.float32)
        }

    def __getitem__(self, idx):
        tensor_path = self.filepaths[idx]
        data = np.load(tensor_path).astype(np.float32)
        
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
        # VAL MODE: Deterministic overlapping patch generator
        # ======================================================================
        elif self.mode == "val":
            c, h_orig, w_orig = data.shape
            pad = self.val_pad
            
            # ------------------------------------------------------------------
            # PADDING KEPT INTACT: Needed for seamless border merging
            # ------------------------------------------------------------------
            data_padded = np.pad(
                data,
                ((0, 0), (pad, pad), (pad, pad)),
                mode="reflect"
            )
            
            # Extract the padded photon mask
            mask_padded = data_padded[2:3]
            
            # Compute EDT on the padded mask globally once
            if np.sum(mask_padded) == 0:
                dist_map_padded = np.full_like(mask_padded, 500.0, dtype=np.float32)
            else:
                dist_map_padded = distance_transform_edt(1.0 - mask_padded[0]).astype(np.float32)
                dist_map_padded *= 10.0
                dist_map_padded = dist_map_padded[np.newaxis]

            _, h, w = data_padded.shape
            stride = self.val_crop - self.val_overlap

            y_starts = list(range(0, h - self.val_crop + 1, stride))
            x_starts = list(range(0, w - self.val_crop + 1, stride))

            if not y_starts or y_starts[-1] + self.val_crop < h:
                y_starts.append(h - self.val_crop)
            if not x_starts or x_starts[-1] + self.val_crop < w:
                x_starts.append(w - self.val_crop)

            packaged_patches = []
            coords = []

            for y in y_starts:
                for x in x_starts:
                    patch_slice = data_padded[:, y:y+self.val_crop, x:x+self.val_crop]
                    dist_slice = dist_map_padded[:, y:y+self.val_crop, x:x+self.val_crop]

                    patch = self._package_dg_vdsr(patch_slice, compute_edt=False)
                    
                    # Overwrite placeholder distance map with the pre-calculated one
                    patch["dist_map"] = torch.from_numpy(dist_slice.copy()).float()
                    
                    packaged_patches.append(patch)
                    coords.append(torch.tensor([y, x], dtype=torch.int32))

            return {
                "dem_bic": torch.stack([p["dem_bic"] for p in packaged_patches]),
                "lidar_delta": torch.stack([p["lidar_delta"] for p in packaged_patches]),
                "mask": torch.stack([p["mask"] for p in packaged_patches]),
                "dist_map": torch.stack([p["dist_map"] for p in packaged_patches]),
                "gt_dem": torch.stack([p["gt_dem"] for p in packaged_patches]),
                "patch_mean": torch.stack([p["patch_mean"] for p in packaged_patches]),
                "coords": torch.stack(coords),
                "canvas_shape": torch.tensor([h, w], dtype=torch.int32),
                "original_shape": torch.tensor([h_orig, w_orig], dtype=torch.int32),
                "pad": torch.tensor(pad, dtype=torch.int32),  # Pad integer returned for the inference script
                "gt_canvas_full": torch.from_numpy(data[3].copy())
            }


def create_dataloaders(train_dirs, val_dirs, batch_size=4, num_workers=8, prefetch_factor=4, pin_memory=True):
    train_dataset = HMATensorDataset(train_dirs, mode="train", train_crop=128)
    val_dataset   = HMATensorDataset(val_dirs,   mode="val",   val_crop=256, val_overlap=192)

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
        persistent_workers=(num_workers > 0)  # Must be False when num_workers=0
    )
    
    # VAL LOADER
    val_loader = DataLoader(
        val_dataset,   
        batch_size=1,          
        shuffle=False, 
        num_workers=0,       
        prefetch_factor=None,
        pin_memory=False       
    )

    return train_loader, val_loader