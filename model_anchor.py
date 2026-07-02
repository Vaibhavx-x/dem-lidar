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
    """
    Rubber-sheet loss for DG-VDSR.

    Key fix vs. previous version:
    - base_loss is halo-masked: relaxed near photon pixels so the network
      is free to deviate from the optical GT where LiDAR corrections land.
    - slope_loss and curve_loss are split into two zones:
        OUTSIDE buffer ring: supervised against optical gt_dem (correct terrain shape)
        INSIDE buffer ring but off-photon: UNSUPERVISED smoothness on pred_dem itself
      This prevents the network exploiting single-pixel spikes to simultaneously
      satisfy pin_loss (few pixels) while leaving slope_loss unchanged (diluted mean).
    - pin_loss is strictly at photon pixels only, normalized by photon count.
    - buffer_size is in PIXELS; converted to metres internally (buffer_size * 10.0).
    """
    def __init__(
        self,
        alpha=1.0,
        beta=1.5,
        gamma=0.5,
        lambda_pin=1.0,       # Start low; ramped to 5.0 over first 15 epochs externally
        pin_beta=1.0,         # SmoothL1 transition point in metres
        decay_radius=15.0,    # Halo decay width in metres (~one ATL08 along-track spacing)
        buffer_size=3,        # Buffer ring width in PIXELS (3px = 30m at 10m/px)
    ):
        super().__init__()
        self.alpha        = alpha
        self.beta         = beta
        self.gamma        = gamma
        self.lambda_pin   = lambda_pin
        self.pin_beta     = pin_beta
        self.decay_radius = decay_radius
        # Convert buffer from pixels to metres for comparison against dist_map (metres)
        self.buffer_metres = buffer_size * 10.0

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

        self.register_buffer("sobel_x",   sobel_x)
        self.register_buffer("sobel_y",   sobel_y)
        self.register_buffer("laplacian", laplacian)

    def _safe_conv(self, tensor, kernel):
        padded = F.pad(tensor, pad=(1, 1, 1, 1), mode="replicate")
        return F.conv2d(padded, kernel, padding=0)

    def forward(self, pred_dem, gt_dem, lidar_raw, mask, dist_map):
        """
        All inputs: [B, 1, H, W] float32
          pred_dem : model output (zero-centred)
          gt_dem   : HMA DTM ch3 (zero-centred by same patch_mean)
          lidar_raw: raw ATL08 elevation ch1 (zero-centred by same patch_mean)
          mask     : binary photon mask ch2
          dist_map : EDT in metres (computed in _package_dg_vdsr; 500m for no-photon patches)
        """
        mask     = mask.float()
        dist_map = dist_map.float()

        # ── 1. Spatial Halo ──────────────────────────────────────────────────
        # w_halo → 1.0 at photon (dist=0), → 0.0 far away
        w_halo = torch.exp(-dist_map / self.decay_radius)

        # ── 2. Base Elevation Loss (optical, relaxed near photons) ───────────
        base_pixel_loss = F.l1_loss(pred_dem, gt_dem, reduction="none")
        base_loss = ((1.0 - w_halo) * base_pixel_loss).mean()

        # ── 3. Zone Masks for Differential Losses ────────────────────────────
        # inner_zone: all pixels within buffer_metres of any photon, INCLUDING
        # the photon pixel itself (dist_map=0 at mask==1 by EDT construction,
        # so the old '& (mask < 0.5)' exclusion was wrong — spikes at photon
        # pixels still escaped the localized term via mean-dilution over 65k px).
        # outside_buffer: normal optical-supervised zone.
        inner_zone     = (dist_map <= self.buffer_metres).float()
        outside_buffer = 1.0 - inner_zone

        # ── 4. Differential Losses (Slope + Curvature) ───────────────────────
        pred_dx = self._safe_conv(pred_dem, self.sobel_x)
        pred_dy = self._safe_conv(pred_dem, self.sobel_y)
        gt_dx   = self._safe_conv(gt_dem,   self.sobel_x)
        gt_dy   = self._safe_conv(gt_dem,   self.sobel_y)

        pred_curve = self._safe_conv(pred_dem, self.laplacian)
        gt_curve   = self._safe_conv(gt_dem,   self.laplacian)

        # Outside inner zone: supervised against optical GT (correct terrain shape)
        slope_sup = (outside_buffer * (
            F.l1_loss(pred_dx, gt_dx, reduction="none") +
            F.l1_loss(pred_dy, gt_dy, reduction="none")
        )).mean()
        curve_sup = (outside_buffer * F.l1_loss(pred_curve, gt_curve, reduction="none")).mean()

        # Inside inner zone (incl. photon pixels): UNSUPERVISED smoothness.
        # Penalises pred_dem's own gradient/curvature magnitude — the rubber-sheet
        # tension that prevents spikes. Normalised per inner-zone pixel so scale
        # is invariant to photon density.
        n_inner      = inner_zone.sum() + 1e-8
        slope_smooth = (inner_zone * (pred_dx.abs() + pred_dy.abs())).sum() / n_inner
        curve_smooth = (inner_zone * pred_curve.abs()).sum() / n_inner

        slope_loss = slope_sup + slope_smooth
        curve_loss = curve_sup + curve_smooth

        # ── 5. Anchor / Pin Loss (strictly at photon pixels) ──────────────────
        pin_pixel_loss = F.smooth_l1_loss(
            pred_dem,
            lidar_raw,
            beta=self.pin_beta,
            reduction="none"
        )
        num_photons = mask.sum() + 1e-8
        pin_loss = (mask * pin_pixel_loss).sum() / num_photons

        # ── 6. Total ──────────────────────────────────────────────────────────
        total_loss = (
            self.alpha      * base_loss  +
            self.beta       * slope_loss +
            self.gamma      * curve_loss +
            self.lambda_pin * pin_loss
        )

        return {
            "total": total_loss,
            "base":  base_loss,
            "slope": slope_loss,
            "curve": curve_loss,
            "pin":   pin_loss,
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