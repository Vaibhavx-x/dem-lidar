import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. STREAM A: THE GEOMORPHOLOGY BACKBONE (Standard VDSR)
# ==============================================================================
class ConvPReLU(nn.Module):
    """Standard VDSR building block: 3x3 Conv -> PReLU"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.prelu = nn.PReLU(channels)

    def forward(self, x):
        return self.prelu(self.conv(x))

class TopoStream(nn.Module):
    def __init__(self, num_layers=18, features=64):
        super().__init__()
        # Input Layer (1 channel -> 64)
        self.entry = nn.Sequential(
            nn.Conv2d(1, features, kernel_size=3, padding=1, bias=False),
            nn.PReLU(features)
        )
        
        # Body (16 intermediate layers)
        layers = [ConvPReLU(features) for _ in range(num_layers - 2)]
        self.body = nn.Sequential(*layers)
        
        # Output Layer (64 -> 1 channel residual)
        self.exit = nn.Conv2d(features, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.entry(x)
        x = self.body(x)
        return self.exit(x) # Outputs R_topo


# ==============================================================================
# 2. STREAM B: THE ANCHOR RADIATOR (Dilated Net)
# ==============================================================================
class DilatedAnchorNet(nn.Module):
    def __init__(self):
        super().__init__()

        # L1: RF = 3×3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, dilation=1)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        # L2: RF = 7×7
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        # L3: RF = 15×15
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=4, dilation=4)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        # Output:
        # channel 0 -> R_anchor
        # channel 1 -> alpha_raw
        self.conv4 = nn.Conv2d(
            64,
            2,
            kernel_size=3,
            padding=8,
            dilation=8,
            bias=True
        )

        # ---- Initialize gate bias ----
        with torch.no_grad():
            nn.init.zeros_(self.conv4.bias)

            # bias[0] -> R_anchor output
            # bias[1] -> alpha output
            self.conv4.bias[1] = -2.0
            # sigmoid(-2) ≈ 0.119

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))

        out = self.conv4(x)

        r_anchor = out[:, 0:1, :, :]
        alpha_raw = out[:, 1:2, :, :]

        alpha = torch.sigmoid(alpha_raw)

        return r_anchor, alpha

# ==============================================================================
# 3. THE MASTER MODEL: DG-VDSR
# ==============================================================================
class DistanceGatedGeoVDSR(nn.Module):
    def __init__(self, topo_layers=18, features=64):
        super().__init__()
        self.stream_a = TopoStream(num_layers=topo_layers, features=features)
        self.stream_b = DilatedAnchorNet()
    
    def forward(self, dem_bic, lidar_delta, mask, dist_map):
        dem_bic = dem_bic.float()
        lidar_delta = lidar_delta.float()
        mask = mask.float()
        dist_map = dist_map.float()
        """
        Expected Input Shapes: All [Batch, 1, Height, Width]
        """
        # 1. Geomorphology guess
        r_topo = self.stream_a(dem_bic)

        # 2. Anchor Radiator guess
        sparse_package = torch.cat([lidar_delta, mask, dist_map], dim=1)
        r_anchor, alpha = self.stream_b(sparse_package)

        # 3. THE TRUST GATE
        r_final = ((1.0 - alpha) * r_topo) + (alpha * r_anchor)

        dem_pred = dem_bic + r_final

        # We return the intermediate pieces too! This allows you to inspect 
        # what the gate is doing in TensorBoard.
        return dem_pred, alpha, r_topo, r_anchor


# ==============================================================================
# 4. THE LOSS FUNCTION
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistanceGatedTopoLoss(nn.Module):
    def __init__(
        self,
        alpha=1.0,
        beta=2.0,
        gamma=5.0,
        lambda_pin=4.0,
        pin_beta=2.0,
        decay_radius=20.0,
        buffer_size=5,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_pin = lambda_pin
        self.pin_beta = pin_beta
        self.decay_radius = decay_radius
        self.buffer_size = buffer_size

        sobel_x = torch.tensor(
            [[-1.,  0.,  1.],
             [-2.,  0.,  2.],
             [-1.,  0.,  1.]],
            dtype=torch.float32
        ).view(1, 1, 3, 3) / 8.0

        sobel_y = torch.tensor(
            [[-1., -2., -1.],
             [ 0.,  0.,  0.],
             [ 1.,  2.,  1.]],
            dtype=torch.float32
        ).view(1, 1, 3, 3) / 8.0

        laplacian = torch.tensor(
            [[ 0.,  1.,  0.],
             [ 1., -4.,  1.],
             [ 0.,  1.,  0.]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("laplacian", laplacian)

    def _safe_conv(self, tensor, kernel):
        padded = F.pad(tensor, pad=(1, 1, 1, 1), mode="replicate")
        return F.conv2d(padded, kernel, padding=0)

    def _make_buffered_mask(self, mask):
        """
        Expands photon supervision to a local neighborhood.
        mask: [B, 1, H, W]
        returns: float tensor [B, 1, H, W]
        """
        mask = mask.float()
        pad = self.buffer_size // 2
        buffered = F.max_pool2d(mask, kernel_size=self.buffer_size, stride=1, padding=pad)
        return (buffered > 0).float()

    def forward(self, pred_dem, gt_dem, lidar_raw, mask, dist_map):
        """
        pred_dem : [B, 1, H, W]
        gt_dem   : [B, 1, H, W]
        lidar_raw: [B, 1, H, W]
        mask     : [B, 1, H, W]
        dist_map : [B, 1, H, W]  (EDT distance in meters)
        """
        mask = mask.float()
        dist_map = dist_map.float()

        # Base elevation loss
        base_loss = F.l1_loss(pred_dem, gt_dem)

        # Slope loss
        pred_dx = self._safe_conv(pred_dem, self.sobel_x)
        pred_dy = self._safe_conv(pred_dem, self.sobel_y)
        gt_dx   = self._safe_conv(gt_dem, self.sobel_x)
        gt_dy   = self._safe_conv(gt_dem, self.sobel_y)
        slope_loss = F.l1_loss(pred_dx, gt_dx) + F.l1_loss(pred_dy, gt_dy)

        # Curvature loss
        pred_curve = self._safe_conv(pred_dem, self.laplacian)
        gt_curve   = self._safe_conv(gt_dem, self.laplacian)
        curve_loss = F.l1_loss(pred_curve, gt_curve)

        # Buffered anchor region
        buffered_mask = self._make_buffered_mask(mask)

        # Distance weight: strong near photons, weaker farther away
        dist_weight = torch.exp(-dist_map / self.decay_radius)

        # Combine the two
        anchor_weight = buffered_mask * dist_weight

        # SmoothL1 anchor loss, weighted spatially
        pin_pixel_loss = F.smooth_l1_loss(
            pred_dem,
            lidar_raw,
            beta=self.pin_beta,
            reduction="none"
        )

        pin_loss = (anchor_weight * pin_pixel_loss).sum() / (anchor_weight.sum() + 1e-8)

        total_loss = (
            self.alpha * base_loss
            + self.beta * slope_loss
            + self.gamma * curve_loss
            + self.lambda_pin * pin_loss
        )

        return {
            "total": total_loss,
            "base": base_loss,
            "slope": slope_loss,
            "curve": curve_loss,
            "pin": pin_loss,
        }
# ==============================================================================
# SANITY CHECK: Execute this script to test your GPU/CPU tensor flow
# ==============================================================================
if __name__ == "__main__":
    print("Initializing DG-VDSR Sanity Test...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate
    model = DistanceGatedGeoVDSR().to(device)
    criterion = DistanceGatedTopoLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Create mock batch of 8 tiles (128x128)
    B, H, W = 8, 128, 128
    
    mock_dem_bic    = torch.randn(B, 1, H, W, device=device) * 200 + 2000 # ~2000m elevation
    mock_lidar_del  = torch.randn(B, 1, H, W, device=device) * 5          # +/- 5m residuals
    mock_mask = (torch.rand(B,1,H,W,device=device) < 0.01).float()
    mock_dist_map   = torch.rand(B, 1, H, W, device=device) * 150.0       # 0 to 150m away
    
    mock_gt_dem     = mock_dem_bic + torch.randn(B, 1, H, W, device=device)*2
    mock_lidar_raw  = mock_dem_bic + mock_lidar_del
  
    # Forward
    optimizer.zero_grad()
    dem_pred, alpha, r_topo, r_anchor = model(
        mock_dem_bic,
        mock_lidar_del,
        mock_mask,
        mock_dist_map
    )
    
    # Loss
    loss_dict = criterion(
        dem_pred,
        mock_gt_dem,
        mock_lidar_raw,
        mock_mask,
        mock_dist_map
    )
    
    # Backward
    loss_dict["total"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    print("\n[SUCCESS] Sanity check passed!")
    print(f"Mean alpha: {alpha.mean().item():.4f}")
    print(f"Base loss :  {loss_dict['base'].item():.4f}")
    print(f"Slope loss:  {loss_dict['slope'].item():.4f}")
    print(f"Curve loss:  {loss_dict['curve'].item():.4f}")
    print(f"Pin loss  :  {loss_dict['pin'].item():.4f}")
    print(f"-> Output DEM shape:   {dem_pred.shape}")
    print(f"-> Trust Alpha shape:  {alpha.shape}")
    print(f"-> Sample Trust range: [{alpha.min().item():.4f} to {alpha.max().item():.4f}]")
    print(f"-> Total Loss generated: {loss_dict['total'].item():.4f}")