import os
import random
import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset, DataLoader

class HMATensorDataset(Dataset):
    def __init__(self, data_paths, mode="train", crop_size=128):
        self.mode = mode
        self.crop_size = crop_size
        
        # Ensure it can handle a list of directories (e.g., [Kl_path, SG_path])
        if isinstance(data_paths, str):
            data_paths = [data_paths]
            
        # Gather all .npy files recursively
        self.filepaths = []
        for path in data_paths:
            for root, _, files in os.walk(path):
                for f in files:
                    if f.endswith('.npy'):
                        self.filepaths.append(os.path.join(root, f))
                        
        if self.mode == "train":
            random.shuffle(self.filepaths) # Break sorted order from subfolders
            print(f"[{self.mode.upper()} DATASET] Loaded {len(self.filepaths)} training files.")

            print("   -> Generating 2048x2048 Spatial Noise Cache in CPU RAM...")
            raw_noise = np.random.normal(0, 1.0, (2048, 2048)).astype(np.float32)
            smoothed_noise = gaussian_filter(raw_noise, sigma=2.0)
            
            max_val = np.max(np.abs(smoothed_noise))
            self.noise_cache = (smoothed_noise / max_val) * 1.5

        elif self.mode == "val":
            print(f"[{self.mode.upper()} DATASET] Loaded {len(self.filepaths)} validation files.")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        # Load the 4-channel tensor from disk
        tensor_path = self.filepaths[idx]
        data = np.load(tensor_path).astype(np.float32)
        
        if self.mode == "train":
            c, h, w = data.shape
            
            # 1. Dynamic Cropping
            max_x = w - self.crop_size
            max_y = h - self.crop_size
            x0 = np.random.randint(0, max_x + 1)
            y0 = np.random.randint(0, max_y + 1)
            data = data[:, y0:y0 + self.crop_size, x0:x0 + self.crop_size]
            
            # 2. Rotations and Flips
            k = np.random.randint(0, 4)
            if k > 0:
                data = np.rot90(data, k=k, axes=(1, 2)).copy()
            if np.random.rand() > 0.5:
                data = np.flip(data, axis=2).copy()
            if np.random.rand() > 0.5:
                data = np.flip(data, axis=1).copy()

            # 3. Z-Shift (Since we ignore ATL08, we only shift C1 and C4)
            z_shift = np.random.uniform(-150.0, 150.0)
            data[0] = data[0] + z_shift                          
            data[3] = data[3] + z_shift                          

            # 4. Sensor Noise Injection
            nx0 = np.random.randint(0, 2048 - self.crop_size)
            ny0 = np.random.randint(0, 2048 - self.crop_size)
            noise_crop = self.noise_cache[ny0:ny0+self.crop_size, nx0:nx0+self.crop_size]
            data[0] = data[0] + noise_crop

        # =========================================================
        # SLICING FOR NORMAL VDSR (Ignore ATL08 Channels 2 and 3)
        # =========================================================
        lr_dem = data[0:1, :, :]  # Channel 1 (Input)
        hr_gt = data[3:4, :, :]   # Channel 4 (Target)
        
        # =========================================================
        # CRITICAL DEM NORMALIZATION: Local Patch Zero-Centering
        # =========================================================
        patch_mean = np.mean(lr_dem)
        
        lr_dem = lr_dem - patch_mean
        hr_gt = hr_gt - patch_mean

        # Return exactly the 3 elements the Jupyter training loop expects
        return (
            torch.from_numpy(np.copy(lr_dem)), 
            torch.from_numpy(np.copy(hr_gt)),
            torch.tensor(patch_mean, dtype=torch.float32)
        )

# ==============================================================================
# Helper function for Jupyter Notebook Import
# ==============================================================================
def create_dataloaders(train_dirs, val_dirs):
    train_dataset = HMATensorDataset(data_paths=train_dirs, mode="train", crop_size=128)
    val_dataset = HMATensorDataset(data_paths=val_dirs, mode="val", crop_size=256)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=0,   # 0 prevents Jupyter Windows crashes
        pin_memory=True,
        drop_last=True 
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )

    return train_loader, val_loader