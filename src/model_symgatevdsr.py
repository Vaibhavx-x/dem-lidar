"""
model_symgate_vdsr.py
======================
Revision of DG-VDSR after the failure analysis on Lower Himalaya tiles.
This file is written as a design log as much as a model file, because the
fixes only make sense with the failure modes next to them.

WHAT WENT WRONG LAST TIME, AND WHAT CHANGES HERE BECAUSE OF IT
----------------------------------------------------------------

1. Gradient starvation in Stream B (the sparse/anchor branch).
   The old model fused streams with a single learned scalar trust gate:
       r_final = (1 - alpha) * r_topo + alpha * r_anchor
   alpha was produced by Stream B itself and initialized biased toward
   distrust (bias=-2.0, sigmoid≈0.12). d(r_final)/d(r_anchor) = alpha,
   so once alpha drifted lower during training, the gradient reaching
   EVERY parameter in Stream B was scaled down by that same collapsing
   number. That is exactly the g_B crash (16-26 -> ~3.0 while g_A held
   steady): the gate did not just express low trust, it throttled the
   one branch responsible for ATL08 fidelity. Textbook modality
   laziness / over-alignment collapse.

   Fix: fusion now happens as additive, per-channel, SYMMETRIC gating at
   three depths (SymmetricGatedFusion), not one late scalar blend. Each
   branch keeps its own identity path to the output no matter what the
   gate is doing:
       dense_out  = dense_feat  + g_dense  * sparse_feat
       sparse_out = sparse_feat + g_sparse * dense_feat
   so d(sparse_out)/d(sparse_feat) = 1 always. A closed gate mutes
   cross-talk between branches, it can never zero out a branch's own
   gradient the way the scalar blend did. Gate biases are initialized
   at 0 (sigmoid=0.5, neutral) instead of pre-committing to distrust.

2. Pin loss vs. smoothness loss fighting at the same pixels.
   The old loss applied an unsupervised curvature/slope penalty INSIDE
   the buffer ring, including the photon pixel itself, at the same time
   pin_loss demanded an exact match to a possibly very different LiDAR
   elevation at that same pixel. Any real local relief the ATL08 point
   captured got fought by the smoothness term. lambda_pin=5.0 vs.
   beta=1.5/gamma=0.5 just produced a stalemate instead of either term
   winning - which is exactly the "Base/Slope/Curve barely move over
   400 epochs" gridlock you logged.

   Fix: the anchor pixel is now handled structurally, not through loss
   weighting. CSPNRefine hard-resets masked (ATL08) pixels to their raw
   elevation on every propagation iteration, so there is no free
   parameter at that exact pixel for pin_loss and curvature_loss to
   disagree about. The new loss explicitly EXCLUDES dist_map==0 pixels
   from the slope/curve terms rather than including them with a diluted
   weight. pin_loss is scored against the pre-refinement coarse output
   only (scoring it against the final, hard-anchored output would be a
   constant zero with no gradient) and no longer needs an aggressive
   ramp to carry the whole burden of anchor fidelity.

   One side effect worth knowing: "anchor MAE" as a metric is no longer
   informative post-refinement, because dem_pred == lidar_raw at masked
   pixels by construction. Track aux_mae (Stream B's own prediction
   quality, see below) and slope/curve RMSE in the halo/far zones
   instead - those are the numbers that will actually move now that
   they are not gridlocked against pin_loss.

3. LR schedule hit its floor for reasons unrelated to real convergence.
   LR_A decayed to 1e-6 by epoch 140, LR_B by epoch 272, after which the
   optimizer was taking steps too small to do anything on a flat-looking
   (but actually gridlocked, not converged) loss surface. See
   `build_recommended_scheduler` below for warm restarts instead of a
   monotonic decay, and `branch_grad_norms` for the same kind of
   diagnostic you were already running manually - now split by
   dense/sparse/fusion/refine/head so a starving branch shows up in the
   first few epochs instead of at epoch 200.

WHAT DIDN'T CHANGE, AND WHY
----------------------------
- GroupNorm + PReLU in Stream A: this is exactly what stopped Stream A's
  gradients from dying in the earlier plain-ReLU version, on terrain
  with far more elevation variance per patch than flat-terrain DEM-SR
  benchmarks assume. Nothing here touches that; it is kept as-is.
- Stream B's own encoder (dilated conv, 4 layers): this part was healthy
  before it collapsed (g_B was fine at 16-26 in early epochs) - the
  failure was in how its output was fused, not in the encoder itself.
  The dilated conv stack is preserved, just reorganized into 3 stages so
  it has fusion tap points. GroupNorm is added here too, but that is a
  precaution for the deeper multi-stage structure, not a response to a
  diagnosed problem in the old 4-layer version.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Shared building block
# ---------------------------------------------------------------------------
class ResidualConvBlock(nn.Module):
    """Conv -> GroupNorm -> PReLU with a skip connection."""

    def __init__(self, channels, dilation=1, num_groups=8):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=dilation, dilation=dilation, bias=False)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.act = nn.PReLU(channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return x + out


def _make_dilated_stage(channels, dilations, num_groups=8):
    return nn.ModuleList([
        ResidualConvBlock(channels, dilation=d, num_groups=num_groups) for d in dilations
    ])


def run_stage(stage, x):
    for blk in stage:
        x = blk(x)
    return x


# ---------------------------------------------------------------------------
# 2. Stream A - dense bicubic DEM encoder
# ---------------------------------------------------------------------------
class DenseTerrainEncoder(nn.Module):
    """
    Encodes Ch1 (bicubic DEM). Three stages of dilated residual blocks so
    the receptive field is large enough to infer a plausible correction
    in tiles with little or no ATL08 coverage - local propagation from
    anchors alone can't reach those pixels, the correction has to come
    from learned terrain context instead.
    """

    def __init__(self, features=64, num_groups=8):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(1, features, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=features),
            nn.PReLU(features),
        )
        self.stage1 = _make_dilated_stage(features, [1, 2, 4, 8], num_groups)
        self.stage2 = _make_dilated_stage(features, [1, 2, 4, 8], num_groups)
        self.stage3 = _make_dilated_stage(features, [1, 2, 4, 8], num_groups)


# ---------------------------------------------------------------------------
# 3. Stream B - sparse ATL08 encoder
# ---------------------------------------------------------------------------
class SparseAnchorEncoder(nn.Module):
    """
    Encodes [lidar_delta, mask, dist_map]. Smaller dilations than Stream A
    - this stream's job is to characterize the neighborhood of a real
    measurement, not to see across the whole tile.

    aux_head predicts the ATL08 delta directly from Stream B's own final
    features, tapped BEFORE the last fusion step so it reflects what
    Stream B can explain on its own. The additive fusion design already
    guarantees Stream B's own gradient can't be gated to zero the way the
    old scalar trust gate did - this head isn't load-bearing for that
    fix, it's a cheap, always-on diagnostic and a bit of extra training
    signal (watch its MAE the way you watched alpha before).
    """

    def __init__(self, features=64, num_groups=8):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(3, features, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=features),
            nn.PReLU(features),
        )
        self.stage1 = _make_dilated_stage(features, [1, 1, 2], num_groups)
        self.stage2 = _make_dilated_stage(features, [1, 2, 4], num_groups)
        self.stage3 = _make_dilated_stage(features, [1, 2, 4], num_groups)
        self.aux_head = nn.Conv2d(features, 1, kernel_size=1, bias=True)


# ---------------------------------------------------------------------------
# 4. Symmetric gated fusion (replaces the single scalar trust gate)
# ---------------------------------------------------------------------------
class SymmetricGatedFusion(nn.Module):
    """
    G_dense  = sigmoid(conv(dense_feat))  -> how much sparse_feat gets
                                              injected into the dense path
    G_sparse = sigmoid(conv(sparse_feat)) -> how much dense_feat gets
                                              injected into the sparse path

    Both are additive residuals on top of each branch's own features,
    never a full replacement, so a closed gate only mutes cross-talk -
    it never scales down a branch's own gradient the way the old
    (1-alpha)/alpha blend did. Gate conv biases start at 0 (sigmoid=0.5):
    neutral, no pre-committed distrust of either stream at init.
    """

    def __init__(self, channels):
        super().__init__()
        self.gate_from_dense = nn.Conv2d(channels, channels, kernel_size=1)
        self.gate_from_sparse = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.gate_from_dense.bias)
        nn.init.zeros_(self.gate_from_sparse.bias)

    def forward(self, dense_feat, sparse_feat):
        g_dense = torch.sigmoid(self.gate_from_dense(dense_feat))
        g_sparse = torch.sigmoid(self.gate_from_sparse(sparse_feat))
        dense_out = dense_feat + g_dense * sparse_feat
        sparse_out = sparse_feat + g_sparse * dense_feat
        return dense_out, sparse_out, g_sparse


# ---------------------------------------------------------------------------
# 5. CSPN refinement - hard-anchors ATL08 pixels, diffuses via learned affinity
# ---------------------------------------------------------------------------
class CSPNRefine(nn.Module):
    """
    Predicts a 9-way affinity (self + 8 neighbors) from the fused dense
    features, then iteratively propagates the coarse prediction while
    forcing masked (ATL08) pixels back to their true raw elevation on
    every iteration. This is the structural fix for the pin-vs-smoothness
    fight: the anchor pixel becomes a boundary condition, not a free
    variable two loss terms have to negotiate over.
    """

    NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),           (0, 1),
                        (1, -1), (1, 0), (1, 1)]

    def __init__(self, channels, num_iters=6):
        super().__init__()
        self.num_iters = num_iters
        # channel 0 = self-affinity, channels 1..8 = the 8 neighbors above
        self.affinity_head = nn.Conv2d(channels, 9, kernel_size=3, padding=1)

    def forward(self, coarse_dem, dense_feat, lidar_raw, mask):
        affinity = F.softmax(self.affinity_head(dense_feat), dim=1)
        H, W = coarse_dem.shape[2], coarse_dem.shape[3]

        x = torch.where(mask > 0.5, lidar_raw, coarse_dem)

        for _ in range(self.num_iters):
            propagated = affinity[:, 0:1] * x
            padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
            for i, (dy, dx) in enumerate(self.NEIGHBOR_OFFSETS, start=1):
                shifted = padded[:, :, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
                propagated = propagated + affinity[:, i:i + 1] * shifted
            x = torch.where(mask > 0.5, lidar_raw, propagated)

        return x, affinity


# ---------------------------------------------------------------------------
# 6. Fusion head
# ---------------------------------------------------------------------------
class FusionHead(nn.Module):
    """Combines the final dense+sparse features into a single-channel
    residual to add back onto the bicubic DEM."""

    def __init__(self, features=64, num_groups=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=features),
            nn.PReLU(features),
        )
        self.out = nn.Conv2d(features, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, dense_feat, sparse_feat):
        x = torch.cat([dense_feat, sparse_feat], dim=1)
        x = self.conv(x)
        return self.out(x)


# ---------------------------------------------------------------------------
# 7. Master model
# ---------------------------------------------------------------------------
class SymGateVDSR(nn.Module):
    """
    Expected input shapes: all [Batch, 1, Height, Width].
    dem_bic      : Ch1, bicubic-upsampled DEM
    lidar_delta  : Ch2 (raw ATL08 elevation) minus dem_bic, precomputed
    mask         : Ch3, binary photon-existence mask
    dist_map     : Euclidean distance transform of mask, in metres
                   (0 at photon pixels; matches the existing data pipeline)
    """

    def __init__(self, features=64, num_groups=8, cspn_iters=6):
        super().__init__()
        self.dense_encoder = DenseTerrainEncoder(features=features, num_groups=num_groups)
        self.sparse_encoder = SparseAnchorEncoder(features=features, num_groups=num_groups)
        self.fusion1 = SymmetricGatedFusion(features)
        self.fusion2 = SymmetricGatedFusion(features)
        self.fusion3 = SymmetricGatedFusion(features)
        self.head = FusionHead(features=features, num_groups=num_groups)
        self.refine = CSPNRefine(channels=features, num_iters=cspn_iters)

    def forward(self, dem_bic, lidar_delta, mask, dist_map):
        dem_bic = dem_bic.float()
        lidar_delta = lidar_delta.float()
        mask = mask.float()
        dist_map = dist_map.float()

        d = self.dense_encoder.entry(dem_bic)
        s = self.sparse_encoder.entry(torch.cat([lidar_delta, mask, dist_map], dim=1))

        d = run_stage(self.dense_encoder.stage1, d)
        s = run_stage(self.sparse_encoder.stage1, s)
        d, s, _ = self.fusion1(d, s)

        d = run_stage(self.dense_encoder.stage2, d)
        s = run_stage(self.sparse_encoder.stage2, s)
        d, s, _ = self.fusion2(d, s)

        d = run_stage(self.dense_encoder.stage3, d)
        s = run_stage(self.sparse_encoder.stage3, s)
        r_anchor_aux = self.sparse_encoder.aux_head(s)  # Stream B's own opinion, pre-fusion3
        d, s, g_sparse_final = self.fusion3(d, s)

        r_coarse = self.head(d, s)
        dem_coarse = dem_bic + r_coarse

        lidar_raw = dem_bic + lidar_delta
        dem_refined, affinity = self.refine(dem_coarse, d, lidar_raw, mask)

        return {
            "dem_pred": dem_refined,      # final output, hard-anchored at ATL08 pixels
            "dem_coarse": dem_coarse,     # pre-refinement, use this for pin_loss
            "r_anchor_aux": r_anchor_aux, # Stream B's own delta prediction, for aux_loss
            "gate_sparse": g_sparse_final,
            "affinity": affinity,
        }


# ---------------------------------------------------------------------------
# 8. Loss function
# ---------------------------------------------------------------------------
class SymGateTopoLoss(nn.Module):
    """
    Rubber-sheet loss, redesigned around the pin-vs-smoothness conflict.

    Zones by distance to nearest photon (dist_map, metres):
      anchor zone (dist_map == 0): excluded entirely from slope/curve -
        CSPNRefine already hard-anchors these pixels, there's nothing
        left for a smoothness term to fight there.
      halo ring   (0 < dist_map <= buffer_metres): light UNsupervised
        smoothness only, to keep the CSPN diffusion from ringing. Weighted
        down (0.25x) relative to the far zone's supervised term.
      far zone    (dist_map > buffer_metres): standard optical-GT
        supervised slope/curve loss, same as before.

    base_loss keeps the same halo-relaxation as before (unchanged, it
    wasn't implicated in the gridlock).

    pin_loss is scored against dem_coarse (pre-refinement), not dem_pred.
    Scoring it against dem_pred would be a constant zero with no gradient,
    since dem_pred already equals lidar_raw at masked pixels by
    construction. pin_loss's job now is just to help the coarse network
    body get close, not to single-handedly enforce anchor fidelity - so
    it no longer needs an aggressive ramp to lambda_pin=5.0.

    aux_loss supervises r_anchor_aux (Stream B's own head) directly
    against lidar_delta. It doesn't compete with anything else - it's an
    isolated diagnostic/training signal, not a term any other loss can
    fight.
    """

    def __init__(self, alpha=1.0, beta=1.5, gamma=0.5,
                 lambda_pin=1.0, lambda_aux=0.4, pin_beta=1.0,
                 decay_radius=15.0, buffer_size=3, halo_weight=0.25):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_pin = lambda_pin
        self.lambda_aux = lambda_aux
        self.pin_beta = pin_beta
        self.decay_radius = decay_radius
        self.buffer_metres = buffer_size * 10.0
        self.halo_weight = halo_weight

        sobel_x = torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor(
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 8.0
        laplacian = torch.tensor(
            [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("laplacian", laplacian)

    def _safe_conv(self, tensor, kernel):
        padded = F.pad(tensor, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(padded, kernel, padding=0)

    def forward(self, outputs, gt_dem, lidar_raw, lidar_delta, mask, dist_map):
        """
        outputs   : dict returned by SymGateVDSR.forward
        gt_dem    : ch4, ground-truth 10m DEM, zero-centred consistently with pred
        lidar_raw : dem_bic + lidar_delta (true elevation at ATL08 pixels)
        lidar_delta: ch2 - dem_bic, the raw delta target for aux_loss
        mask      : ch3, binary photon mask
        dist_map  : EDT in metres, 0 at photon pixels
        """
        dem_pred = outputs["dem_pred"]
        dem_coarse = outputs["dem_coarse"]
        r_anchor_aux = outputs["r_anchor_aux"]

        mask = mask.float()
        dist_map = dist_map.float()

        anchor_zone = (dist_map <= 1e-6).float()
        halo_zone = ((dist_map > 1e-6) & (dist_map <= self.buffer_metres)).float()
        far_zone = (dist_map > self.buffer_metres).float()

        # 1. base loss - halo-relaxed, same mechanism as before
        w_halo = torch.exp(-dist_map / self.decay_radius)
        base_pixel = F.l1_loss(dem_pred, gt_dem, reduction="none")
        base_loss = ((1.0 - w_halo) * base_pixel).mean()

        # 2. slope / curvature on the FINAL refined output - the CSPN
        #    affinity head needs gradient from these to learn a sensible
        #    propagation field, not just from base_loss.
        pred_dx = self._safe_conv(dem_pred, self.sobel_x)
        pred_dy = self._safe_conv(dem_pred, self.sobel_y)
        gt_dx = self._safe_conv(gt_dem, self.sobel_x)
        gt_dy = self._safe_conv(gt_dem, self.sobel_y)
        pred_curve = self._safe_conv(dem_pred, self.laplacian)
        gt_curve = self._safe_conv(gt_dem, self.laplacian)

        n_far = far_zone.sum() + 1e-8
        slope_far = (far_zone * (
            F.l1_loss(pred_dx, gt_dx, reduction="none") +
            F.l1_loss(pred_dy, gt_dy, reduction="none")
        )).sum() / n_far
        curve_far = (far_zone * F.l1_loss(pred_curve, gt_curve, reduction="none")).sum() / n_far

        n_halo = halo_zone.sum() + 1e-8
        slope_halo = (halo_zone * (pred_dx.abs() + pred_dy.abs())).sum() / n_halo
        curve_halo = (halo_zone * pred_curve.abs()).sum() / n_halo

        # anchor_zone is deliberately absent from both terms below.
        slope_loss = slope_far + self.halo_weight * slope_halo
        curve_loss = curve_far + self.halo_weight * curve_halo

        # 3. pin loss - pre-refinement coarse output only
        pin_pixel = F.smooth_l1_loss(dem_coarse, lidar_raw, beta=self.pin_beta, reduction="none")
        num_photons = mask.sum() + 1e-8
        pin_loss = (mask * pin_pixel).sum() / num_photons

        # 4. auxiliary anchor loss - Stream B's own, isolated signal
        aux_pixel = F.smooth_l1_loss(r_anchor_aux, lidar_delta, beta=self.pin_beta, reduction="none")
        aux_loss = (mask * aux_pixel).sum() / num_photons

        total_loss = (
            self.alpha * base_loss +
            self.beta * slope_loss +
            self.gamma * curve_loss +
            self.lambda_pin * pin_loss +
            self.lambda_aux * aux_loss
        )

        return {
            "total": total_loss,
            "base": base_loss,
            "slope": slope_loss,
            "curve": curve_loss,
            "pin": pin_loss,
            "aux": aux_loss,
        }


# ---------------------------------------------------------------------------
# 9. Training-loop utilities: the LR schedule and grad-norm diagnostics
#    that the previous run's post-mortem was crying out for.
# ---------------------------------------------------------------------------
def build_recommended_scheduler(optimizer, warm_restart_epochs=40, min_lr_floor=1e-5):
    """
    Cosine annealing with warm restarts instead of a monotonic decay to a
    hard floor. The previous schedule hit 1e-6 by epoch 140-272 and then
    "vibrated in place" for the rest of training on what looked like a
    plateau but was actually loss-term gridlock, not convergence. Warm
    restarts periodically kick the LR back up so the optimizer keeps
    exploring instead of freezing the first time the (new) loss surface
    looks flat. min_lr_floor is set an order of magnitude above the old
    1e-6 floor deliberately - if the model needs steps smaller than that
    to keep improving, that's worth seeing directly rather than assuming.
    """
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=warm_restart_epochs, T_mult=1, eta_min=min_lr_floor
    )


def branch_grad_norms(model):
    """
    Call after loss.backward(), before optimizer.step(). Returns the L2
    grad norm per branch so a starving branch shows up in the first few
    epochs, the way g_B's collapse only became visible ~200 epochs in
    last time.
    """
    def _norm(params):
        total = 0.0
        for p in params:
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        return total ** 0.5

    return {
        "dense_encoder": _norm(model.dense_encoder.parameters()),
        "sparse_encoder": _norm(model.sparse_encoder.parameters()),
        "fusion": _norm(list(model.fusion1.parameters()) +
                         list(model.fusion2.parameters()) +
                         list(model.fusion3.parameters())),
        "refine": _norm(model.refine.parameters()),
        "head": _norm(model.head.parameters()),
    }


# ==============================================================================
# SANITY CHECK
# ==============================================================================
if __name__ == "__main__":
    print("Initializing SymGate-VDSR sanity test...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SymGateVDSR().to(device)
    criterion = SymGateTopoLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    B, H, W = 8, 128, 128
    mock_dem_bic = torch.randn(B, 1, H, W, device=device) * 200 + 2000
    mock_lidar_delta = torch.randn(B, 1, H, W, device=device) * 5
    mock_mask = (torch.rand(B, 1, H, W, device=device) < 0.01).float()
    mock_dist_map = torch.rand(B, 1, H, W, device=device) * 150.0
    mock_dist_map = torch.where(mock_mask > 0.5, torch.zeros_like(mock_dist_map), mock_dist_map)
    mock_gt_dem = mock_dem_bic + torch.randn(B, 1, H, W, device=device) * 2
    mock_lidar_raw = mock_dem_bic + mock_lidar_delta

    optimizer.zero_grad()
    outputs = model(mock_dem_bic, mock_lidar_delta, mock_mask, mock_dist_map)
    loss_dict = criterion(
        outputs, mock_gt_dem, mock_lidar_raw, mock_lidar_delta, mock_mask, mock_dist_map
    )
    loss_dict["total"].backward()

    grad_norms = branch_grad_norms(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    print("\n[SUCCESS] Sanity check passed!")
    for k in ["total", "base", "slope", "curve", "pin", "aux"]:
        print(f"  {k:>6}: {loss_dict[k].item():.4f}")

    print("\nPer-branch gradient norms (dense_encoder and sparse_encoder")
    print("should stay in the same order of magnitude - unlike the old")
    print("g_A/g_B split where sparse collapsed to ~3.0 while dense held ~12):")
    for k, v in grad_norms.items():
        print(f"  {k:16s}: {v:.4f}")

    anchor_diff = (outputs["dem_pred"] - mock_lidar_raw).abs()
    anchor_fidelity = (mock_mask * anchor_diff).sum() / (mock_mask.sum() + 1e-8)
    print(f"\nOutput DEM shape: {tuple(outputs['dem_pred'].shape)}")
    print(f"Anchor fidelity (mean abs error at ATL08 pixels, should be ~0 by construction): "
          f"{anchor_fidelity.item():.8f}")