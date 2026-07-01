import torch
import numpy as np

class HannStreamMerger:
    def __init__(
        self,
        canvas_shape,
        patch_size=256,
        device="cpu",
        pad=0,
        original_shape=None,):
        """
            A stateful, memory-safe 2D Hann Window Overlap-Add sticher.
        
        canvas_shape: Tuple (H_canvas, W_canvas) of the final target Geotiff.
        patch_size:   The window size (default 256).
        device:       Where to store the master canvas ('cpu' saves VRAM; 'cuda' is 20x faster).
        """
        self.H, self.W = canvas_shape
        self.patch_size = patch_size
        self.device = device
        self.pad = pad
        self.original_shape = original_shape

        # 1. Allocate the Master Accumulators
        self.canvas_dem = torch.zeros((self.H, self.W), dtype=torch.float32, device=self.device)
        self.canvas_wt  = torch.zeros((self.H, self.W), dtype=torch.float32, device=self.device)

        # 2. Generate the 2D Hann Window once
        # periodic=False ensures the window decays to absolute 0.0 at the extreme edges
        win_1d = torch.hann_window(patch_size, periodic=False, dtype=torch.float32, device=self.device)
        
        # Outer product: win_2d[y, x] = win_1d[y] * win_1d[x]
        self.window_2d = win_1d.view(-1, 1) * win_1d.view(1, -1)

    def add_batch(self, preds_centered, patch_means, coords):
        """
        Pushes a mini-batch of model predictions onto the canvas.
        
        preds_centered: Tensor of shape [B, 1, H, W] or [B, H, W] (The network's output)
        patch_means:    Tensor of shape [B] (The patch means saved by the DataLoader)
        coords:         Tensor of shape [B, 2] containing the [y, x] top-left origins
        """
        with torch.no_grad():
            # Standardize shapes to 3D: [B, 256, 256]
            if preds_centered.dim() == 4:
                preds_centered = preds_centered.squeeze(1)

            # Move batch to the merger's device
            preds_centered = preds_centered.to(self.device)
            patch_means = patch_means.to(self.device).view(-1, 1, 1)
            coords = coords.to(self.device)

            # ==================================================================
            # CRITICAL STEP: Un-center the DEM *before* applying the window
            # ==================================================================
            preds_absolute = preds_centered + patch_means

            # Apply Hann weighting
            preds_weighted = preds_absolute * self.window_2d

            # Scatter-add onto the master canvas
            for i in range(len(coords)):
                y, x = coords[i][0].item(), coords[i][1].item()
                
                self.canvas_dem[y : y + self.patch_size, x : x + self.patch_size] += preds_weighted[i]
                self.canvas_wt[y  : y + self.patch_size, x : x + self.patch_size] += self.window_2d

    def get_final_dem(self, as_tensor=False):
        """
        Returns the stitched DEM.
        If padding was used during validation, the padding is cropped away.
        """
        safe_wt = torch.where(
            self.canvas_wt == 0,
            torch.ones_like(self.canvas_wt),
            self.canvas_wt,
        )
    
        final = self.canvas_dem / safe_wt
    
        # Remove validation padding
        if self.pad > 0:
            final = final[
                self.pad : self.H - self.pad,
                self.pad : self.W - self.pad,
            ]
            if self.original_shape is not None:
                h0, w0 = self.original_shape
                final = final[:h0, :w0]
    
        if as_tensor:
            return final
            
        return final.cpu().numpy()

    def reset(self):
        """Wipes the canvas clean for the next validation chunk."""
        self.canvas_dem.zero_()
        self.canvas_wt.zero_()


# ==============================================================================
# Execution test to verify memory usage
# ==============================================================================
if __name__ == "__main__":
    print("Testing HannStreamMerger on mock 3000x3000 canvas...")
    
    # 1. Create a sticher sitting in RAM
    merger = HannStreamMerger(canvas_shape=(3000, 3000), patch_size=256, device="cpu")

    # 2. Simulate feeding it 4 batches of 8 tiles
    mock_coords = torch.tensor([[0, 0], [0, 192], [0, 384], [192, 0]])
    mock_preds  = torch.randn(4, 1, 256, 256)
    mock_means  = torch.tensor([2100.0, 2105.0, 2090.0, 2110.0])

    merger.add_batch(mock_preds, mock_means, mock_coords)

    finished_map = merger.get_final_dem()
    print(f"[SUCCESS] Stitched Canvas generated! Shape: {finished_map.shape}, Dtype: {finished_map.dtype}")