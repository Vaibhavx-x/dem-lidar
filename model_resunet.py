import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
    The building block of ResUNet. Replaces plain convolutions with 
    a dual-conv residual pathway + InstanceNorm 'shock absorbers'.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)

        # FIX: Avoid 1x1 convs with stride=2 (which drops 75% of features).
        # Use AvgPool for downsampling, then a 1x1 conv to map channel depths.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2) if stride != 1 else nn.Identity(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        # Notice: Residual addition happens BEFORE the block's final ReLU
        return self.relu(out + identity)


class ResUNetDEM(nn.Module):
    def __init__(self, base_features=64):
        """
        ResUNet optimized for unbounded, single-channel topography grids.
        Expects input tensor shape: [Batch, 1, H, W] (already interpolated to target HR grid).
        Automatically pads dimensions that are not divisible by 8 (e.g. 2700x2700).
        """
        super(ResUNetDEM, self).__init__()

        # ==================== ENCODER (Captures Macro-Geomorphology) ====================
        # Level 1: [B, 1, H, W] -> [B, 64, H, W]
        self.enc1 = ResBlock(in_channels=1, out_channels=base_features, stride=1)
        
        # Level 2: Downsamples to [B, 128, H/2, W/2]
        self.enc2 = ResBlock(in_channels=base_features, out_channels=base_features * 2, stride=2)
        
        # Level 3: Downsamples to [B, 256, H/4, W/4]
        self.enc3 = ResBlock(in_channels=base_features * 2, out_channels=base_features * 4, stride=2)

        # ==================== BOTTLENECK (The Watershed View) ====================
        # Receptive field now covers the entire patch at [B, 512, H/8, W/8]
        self.bridge = ResBlock(in_channels=base_features * 4, out_channels=base_features * 8, stride=2)

        # ==================== DECODER (Precise Topology Reconstruction) ====================
        # Level 3 Up: [B, 512, H/8, W/8] -> [B, 256, H/4, W/4]
        self.up3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, kernel_size=2, stride=2, bias=False)
        self.dec3 = ResBlock(in_channels=base_features * 8, out_channels=base_features * 4, stride=1)

        # Level 2 Up: [B, 256, H/4, W/4] -> [B, 128, H/2, W/2]
        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2, bias=False)
        self.dec2 = ResBlock(in_channels=base_features * 4, out_channels=base_features * 2, stride=1)

        # Level 1 Up: [B, 128, H/2, W/2] -> [B, 64, H, W]
        self.up1 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2, bias=False)
        self.dec1 = ResBlock(in_channels=base_features * 2, out_channels=base_features, stride=1)

        # ==================== FINAL RECONSTRUCTION LAYER ====================
        # FIX: bias=True is required here so the network can learn global elevation shifts 
        # that InstanceNorm removed in the earlier layers.
        self.reconstruction_layer = nn.Conv2d(
            in_channels=base_features, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True
        )

        # Optimized Kaiming He initialization for continuous geomorphology layers
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # FIX: Dynamically pad arbitrary shapes (like 2700x2700) to be divisible by 8
        _, _, H, W = x.shape
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            # Pad on the Right and Bottom using edge replication to avoid cliffs
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            x_padded = x

        # x shape: [Batch, 1, H', W'] (Coarse DEM upscaled via bicubic/bilinear interpolation)

        # --- Encoder Path (Stashing outputs for skip connections) ---
        e1 = self.enc1(x_padded)         # Shape: [B, 64, H', W']
        e2 = self.enc2(e1)               # Shape: [B, 128, H'/2, W'/2]
        e3 = self.enc3(e2)               # Shape: [B, 256, H'/4, W'/4]

        # --- Deep Watershed Bottleneck ---
        b = self.bridge(e3)              # Shape: [B, 512, H'/8, W'/8]

        # --- Decoder Path (Stitching spatial superhighways) ---
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)  # Stacking along channel axis: 256 + 256 = 512
        d3 = self.dec3(d3)               # Fused back down to 256

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)  # 128 + 128 = 256
        d2 = self.dec2(d2)               # Fused back down to 128

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)  # 64 + 64 = 128
        d1 = self.dec1(d1)               # Fused back down to 64

        # Output the single-channel high-frequency elevation residual
        residual = self.reconstruction_layer(d1)

        # Global Skip Connection: Add predicted high-frequency ridges back to the base geometry
        out = x_padded + residual
        
        # Crop back to the original dimensions before returning
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]
            
        return out


class GeomorphologicalLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=2.0):
        super(GeomorphologicalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.l1 = nn.L1Loss()

        sobel_x = torch.tensor([[ -1.,  0.,  1.],
                                [ -2.,  0.,  2.],
                                [ -1.,  0.,  1.]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0

        sobel_y = torch.tensor([[ -1., -2., -1.],
                                [  0.,  0.,  0.],
                                [  1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0

        laplacian = torch.tensor([[  0.,  1.,  0.],
                                  [  1., -4.,  1.],
                                  [  0.,  1.,  0.]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('laplacian', laplacian)

    def _pad_and_conv(self, tensor, kernel):
        # FIX: Pad 1 pixel on Left, Right, Top, Bottom using 'replicate' 
        # to prevent artificial "1000m cliffs" at the boundaries of the image patch.
        padded = F.pad(tensor, (1, 1, 1, 1), mode='replicate')
        return F.conv2d(padded, kernel, padding=0)

    def forward(self, pred, gt):
        base_loss = self.l1(pred, gt)

        # FIX: Safely snap the buffers to whatever device/dtype the incoming image lives on
        # This prevents crashes if the user forgets to send the loss function `.to('cuda')`
        s_x = self.sobel_x.to(device=pred.device, dtype=pred.dtype)
        s_y = self.sobel_y.to(device=pred.device, dtype=pred.dtype)
        lap = self.laplacian.to(device=pred.device, dtype=pred.dtype)

        pred_dx = self._pad_and_conv(pred, s_x)
        pred_dy = self._pad_and_conv(pred, s_y)
        gt_dx = self._pad_and_conv(gt, s_x)
        gt_dy = self._pad_and_conv(gt, s_y)
        
        slope_loss = self.l1(pred_dx, gt_dx) + self.l1(pred_dy, gt_dy)

        pred_curve = self._pad_and_conv(pred, lap)
        gt_curve = self._pad_and_conv(gt, lap)
        
        curve_loss = self.l1(pred_curve, gt_curve)

        return (self.alpha * base_loss) + (self.beta * slope_loss) + (self.gamma * curve_loss)


# =============================================================================
# Jupyter Cell Execution Verification
# =============================================================================
if __name__ == "__main__":
    # Test tensor representing 4 patches of 128x128 continuous elevation grids
    sample_input = torch.randn(4, 1, 128, 128)
    
    # Initialize the new ResUNet DEM Super-Resolution model
    model = ResUNetDEM(base_features=64)
    loss_fn = GeomorphologicalLoss(alpha=1.0, beta=1.0, gamma=2.0)
    
    with torch.no_grad():
        sample_output = model(sample_input)
        dummy_gt = torch.randn(4, 1, 128, 128)
        loss = loss_fn(sample_output, dummy_gt)
        
    print("✅ ResUNet Architecture loaded and forward pass successful.")
    print(f"   Input Tensor Shape:  {sample_input.shape}")
    print(f"   Output Tensor Shape: {sample_output.shape}")
    print(f"   Test Geomorphological Loss Value: {loss.item():.4f}")
    
    # Test for the Dynamic Pad/Crop Feature:
    big_input = torch.randn(1, 1, 2700, 2700)
    with torch.no_grad():
        big_output = model(big_input)
    print(f"\n✅ Dynamic Padding handled arbitrary dimensions safely.")
    print(f"   Original Big Input Shape: {big_input.shape}")
    print(f"   Returned Big Output Shape: {big_output.shape}")