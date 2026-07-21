# hann_merge.py
import torch
import numpy as np


class HannStreamMerger:
    def __init__(
        self,
        canvas_shape,
        patch_size=256,
        device="cpu",
        pad=0,
        original_shape=None,
    ):
        """
        Stateful 2D Hann overlap-add merger.

        Args:
            canvas_shape:   (H_canvas, W_canvas) of padded canvas
            patch_size:     tile size
            device:         usually 'cpu' to save VRAM
            pad:            symmetric validation padding to crop off later
            original_shape: original unpadded (H, W)
        """
        self.H, self.W = canvas_shape
        self.patch_size = patch_size
        self.device = device
        self.pad = pad
        self.original_shape = original_shape

        self.canvas_dem = torch.zeros(
            (self.H, self.W), dtype=torch.float32, device=self.device
        )
        self.canvas_wt = torch.zeros(
            (self.H, self.W), dtype=torch.float32, device=self.device
        )

        win_1d = torch.hann_window(
            patch_size, periodic=False, dtype=torch.float32, device=self.device
        )
        self.window_2d = win_1d.view(-1, 1) * win_1d.view(1, -1)

    @torch.no_grad()
    def add_batch(self, preds_centered, patch_means, coords):
        """
        Args:
            preds_centered: [B, 1, H, W] or [B, H, W]
            patch_means:    [B]
            coords:         [B, 2] with [y, x]
        """
        if preds_centered.dim() == 4:
            preds_centered = preds_centered.squeeze(1)

        preds_centered = preds_centered.to(self.device, dtype=torch.float32)
        patch_means = patch_means.to(self.device, dtype=torch.float32).view(-1, 1, 1)
        coords = coords.to(self.device)

        preds_absolute = preds_centered + patch_means
        preds_weighted = preds_absolute * self.window_2d

        for i in range(coords.shape[0]):
            y, x = coords[i].tolist()

            self.canvas_dem[y:y + self.patch_size, x:x + self.patch_size] += preds_weighted[i]
            self.canvas_wt[y:y + self.patch_size, x:x + self.patch_size] += self.window_2d

    def get_final_dem(self, as_tensor=False):
        safe_wt = torch.where(
            self.canvas_wt == 0,
            torch.ones_like(self.canvas_wt),
            self.canvas_wt,
        )

        final = self.canvas_dem / safe_wt

        if self.pad > 0:
            final = final[
                self.pad:self.H - self.pad,
                self.pad:self.W - self.pad,
            ]

        if self.original_shape is not None:
            h0, w0 = self.original_shape
            final = final[:h0, :w0]

        if as_tensor:
            return final

        return final.cpu().numpy()

    def reset(self):
        self.canvas_dem.zero_()
        self.canvas_wt.zero_()


if __name__ == "__main__":
    print("Testing HannStreamMerger on mock 3000x3000 canvas...")

    merger = HannStreamMerger(
        canvas_shape=(3000, 3000),
        patch_size=256,
        device="cpu",
    )

    mock_coords = torch.tensor([
        [0, 0],
        [0, 192],
        [0, 384],
        [192, 0],
    ], dtype=torch.long)

    mock_preds = torch.randn(4, 1, 256, 256)
    mock_means = torch.tensor([2100.0, 2105.0, 2090.0, 2110.0], dtype=torch.float32)

    merger.add_batch(mock_preds, mock_means, mock_coords)
    finished_map = merger.get_final_dem()

    print(f"[SUCCESS] Stitched Canvas generated! Shape: {finished_map.shape}, Dtype: {finished_map.dtype}")