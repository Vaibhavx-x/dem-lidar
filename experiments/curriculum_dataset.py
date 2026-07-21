"""
curriculum_dataset.py
=====================
Curriculum training dataloader for DG-VDSR.

Rationale
---------
When both streams train simultaneously from epoch 1, Stream B (DilatedAnchorNet)
receives early gradient through pin_loss while the trust gate alpha is still open
(~0.119 at init). This dilutes Stream A's slope/curve gradient ownership, leaving
the TopoStream without a chance to build a strong morphological prior, causing
val_slope_rmse to stagnate while val_anchor_mae keeps dropping.

Fix: teach Stream A first on the FULL dataset, then introduce photon buckets.

Stage 1  (epochs   1-30 ): FULL unbucketed dataset, lambda_pin 0.0,      stream_b FROZEN
Phase 1  (epochs  31-60 ): 0-10 + 11-25,             lambda_pin 0.0 → 1.0, stream_b unfreezes
Phase 2  (epochs  61-100): 0-10 … 51-100,            lambda_pin 1.0 → 3.0
Phase 3  (epochs 101+   ): all 5 buckets,            lambda_pin 3.0 → 5.0 (ramp ends ~ep150)

Why Stage 1 is the full dataset (not the 0-10 bucket)
----------------------------------------------------
The original Phase 0 pretrained Stream A on the 0-10 photon-count bucket alone. That
bucket turned out to be a biased subset: in this domain a low photon count tracks
canopy obstruction (dense forest), not "easy/flat" terrain, so ~50 epochs on it
plateaued SlopeRMSE at ~1.10-1.15 — worse than joint full-dataset training (~1.03-1.05).
Stage 1 therefore pretrains Stream A on the full 1649-tile pool with plain shuffle
(no WeightedRandomSampler, no bucket subfolders), keeping the freeze/unfreeze +
lambda_pin curriculum around it. The dense-photon buckets still drive Phase 1+ once
Stream B unfreezes, so CurriculumDataset / the bucket-merging logic is retained.

Validation cadence ("train several epochs, then validate; dial back to 1:1")
---------------------------------------------------------------------------
VAL_EVERY dials the train:val ratio from 4:1 down to 1:1 as training matures:

    Stage 1 → validate every 4 epochs
    Phase 1 → validate every 3 epochs
    Phase 2 → validate every 2 epochs
    Phase 3 → validate every 1 epoch   (1:1)

Usage in train_lidar_curriculum.ipynb
--------------------------------------
    from curriculum_dataset import (
        get_curriculum_phase,          # epoch -> dict (pure function)
        interp_lambda_pin,             # (phase_dict, epoch) -> float
        create_curriculum_dataloader,  # phase_dict -> DataLoader
        count_full_training_set,       # -> int (fixed sampler length)
        set_stream_trainable,          # freeze / unfreeze Stream B
        is_phase_boundary,             # epoch in {1, 31, 61, 101}
        should_validate,               # (phase_dict, epoch, total_epochs) -> bool
        PHASE_START_EPOCHS,
        STREAM_B_UNFREEZE_EPOCH,
        BUCKET_NAMES,
    )
    from dataloader_anchor import HMATensorDataset   # val loader only (unchanged)

Validation loader: keep HMATensorDataset(mode="val") on the fixed
validation_contiguous folders unchanged across ALL phases.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataloader_anchor import HMATensorDataset


# ==============================================================================
# 1. CURRICULUM SCHEDULE
# ==============================================================================

BUCKET_NAMES = ["0-10", "11-25", "26-50", "51-100", "100_plus"]

# ── Phase table ────────────────────────────────────────────────────────────────
# Each entry:
#   membership_end   : last epoch (inclusive) that still belongs to this phase.
#                      Governs DataLoader rebuilds and "last epoch of phase" logic.
#   active_buckets   : which bucket subdirs to sample this phase.
#   bucket_weights   : desired probability mass per active bucket (need not sum to 1;
#                      normalised inside CurriculumDataset).
#   (lp_lo, lp_hi)   : lambda_pin endpoints, linearly interpolated within the phase.
#   lambda_ramp_end  : epoch at which lambda_pin reaches lp_hi. For finite phases this
#                      equals membership_end; Phase 3 is open-ended, so we give it a
#                      nominal ramp window (ends ~ep150) and hold at lp_hi afterwards.
#   val_every        : validate once every N epochs during this phase.
# ──────────────────────────────────────────────────────────────────────────────
CURRICULUM_PHASES = [
    # Stage 1 (epochs 1-30): Pure topo pretraining on the FULL unbucketed dataset.
    # stream_b FROZEN, pin loss = 0. use_full_dataset=True → plain-shuffle DataLoader
    # with NO WeightedRandomSampler and NO bucket subfolders (active_buckets is unused).
    dict(membership_end=30,   use_full_dataset=True,
         active_buckets=None, bucket_weights=None,
         lp_lo=0.0, lp_hi=0.0, lambda_ramp_end=30,  val_every=4),

    # Phase 1 (epochs 31-60): Sparse photons introduced. stream_b unfreezes, pin ramps 0 → 1.
    dict(membership_end=60,   use_full_dataset=False,
         active_buckets=["0-10", "11-25"],
         bucket_weights=[0.4, 0.6],
         lp_lo=0.0, lp_hi=1.0, lambda_ramp_end=60,  val_every=3),

    # Phase 2 (epochs 61-100): Moderate photons, pin ramps 1 → 3. (buckets/weights unchanged)
    dict(membership_end=100,  use_full_dataset=False,
         active_buckets=["0-10", "11-25", "26-50", "51-100"],
         bucket_weights=[0.1, 0.2, 0.4, 0.3],
         lp_lo=1.0, lp_hi=3.0, lambda_ramp_end=100, val_every=2),

    # Phase 3 (epochs 101+): All buckets, pin at operating range 3 → 5 (open-ended
    # membership, nominal lambda ramp finishing at epoch 150). (buckets/weights unchanged)
    dict(membership_end=9999, use_full_dataset=False,
         active_buckets=list(BUCKET_NAMES),
         bucket_weights=[0.05, 0.1, 0.3, 0.35, 0.2],
         lp_lo=3.0, lp_hi=5.0, lambda_ramp_end=150, val_every=1),
]

# First epoch of each stage/phase (triggers DataLoader rebuild + freeze/unfreeze logic).
PHASE_START_EPOCHS = [1, 31, 61, 101]

# Stream B unfreezes at the start of Phase 1.
STREAM_B_UNFREEZE_EPOCH = PHASE_START_EPOCHS[1]   # 31


# ── Pure schedule helpers (functions of epoch only) ─────────────────────────────

def get_phase_index(epoch: int) -> int:
    """Return the 0-based curriculum phase index for *epoch*."""
    for i, phase in enumerate(CURRICULUM_PHASES):
        if epoch <= phase["membership_end"]:
            return i
    return len(CURRICULUM_PHASES) - 1


def get_curriculum_phase(epoch: int) -> dict:
    """
    Pure function: return the full curriculum configuration for *epoch*.

    Because this depends ONLY on the epoch number, a resumed run can reconstruct
    the exact phase / lambda_pin / freeze-state from ckpt['epoch'] with no extra
    persisted state — see the resume path in train_lidar_curriculum.ipynb.

    Returns a dict with:
        phase_idx          : int, 0-based phase index (for logging).
        use_full_dataset   : bool, True for Stage 1 (full unbucketed pool, plain shuffle).
        active_buckets     : list[str] | None, bucket subdirs sampled this phase
                             (None when use_full_dataset is True).
        bucket_weights     : list[float] | None, per-bucket probability mass (unnormalised OK).
        lambda_pin_lo/hi   : float, endpoints of the pin-loss ramp for this phase.
        phase_start_epoch  : int, first epoch of this phase.
        phase_end_epoch    : int, last epoch (inclusive) of this phase (9999 = open).
        lambda_ramp_end    : int, epoch at which lambda_pin reaches lambda_pin_hi.
        val_every          : int, validate once every N epochs this phase.
    """
    idx   = get_phase_index(epoch)
    phase = CURRICULUM_PHASES[idx]
    return {
        "phase_idx":         idx,
        "use_full_dataset":  bool(phase.get("use_full_dataset", False)),
        "active_buckets":    list(phase["active_buckets"]) if phase["active_buckets"] else None,
        "bucket_weights":    list(phase["bucket_weights"]) if phase["bucket_weights"] else None,
        "lambda_pin_lo":     float(phase["lp_lo"]),
        "lambda_pin_hi":     float(phase["lp_hi"]),
        "phase_start_epoch": PHASE_START_EPOCHS[idx],
        "phase_end_epoch":   int(phase["membership_end"]),
        "lambda_ramp_end":   int(phase["lambda_ramp_end"]),
        "val_every":         int(phase["val_every"]),
    }


def interp_lambda_pin(phase: dict, epoch: int) -> float:
    """
    Linearly interpolate lambda_pin from lambda_pin_lo → lambda_pin_hi using the
    epoch's position within the phase's ramp window. Clamped to [lo, hi].
    """
    start    = phase["phase_start_epoch"]
    ramp_end = phase["lambda_ramp_end"]
    lo, hi   = phase["lambda_pin_lo"], phase["lambda_pin_hi"]

    duration = ramp_end - start
    if duration <= 0 or lo == hi:
        return float(lo)

    t = (epoch - start) / duration
    t = min(max(t, 0.0), 1.0)
    return float(lo + (hi - lo) * t)


def is_phase_boundary(epoch: int) -> bool:
    """True if *epoch* is the first epoch of a new curriculum phase (1/51/101/151)."""
    return epoch in PHASE_START_EPOCHS


def should_validate(phase: dict, epoch: int, total_epochs: int) -> bool:
    """
    Decide whether to run validation this epoch under the variable cadence.

    Validate when EITHER:
      • epoch is a multiple of the phase's val_every, OR
      • it is the last epoch of the phase (so the model entering the next phase
        is always scored — keeps scheduler / best-checkpoint state coherent), OR
      • it is the final training epoch.
    """
    if epoch % phase["val_every"] == 0:
        return True
    if epoch == phase["phase_end_epoch"]:
        return True
    if epoch == total_epochs:
        return True
    return False


# ==============================================================================
# 2. CURRICULUM DATASET
# ==============================================================================

class CurriculumDataset(HMATensorDataset):
    """
    Training dataset that draws .npy tiles from a chosen subset of photon-density
    buckets, merging both regions (Kl + SG) into one pool per bucket.

    Subclasses HMATensorDataset so the crop / rotate / flip / noise augmentation and
    ``_package_dg_vdsr`` packaging are BYTE-FOR-BYTE identical to the standard
    training path — the only thing that changes is *which files* are in the pool and
    how they are weighted. We deliberately bypass the parent __init__ (it walks whole
    directories and shuffles); everything the inherited train-mode __getitem__ needs
    (mode, train_crop, filepaths, noise_cache) is set up here instead.

    Expected directory layout (one base dir per region)::

        <base_train_dir>/
            0-10/  11-25/  26-50/  51-100/  100_plus/   ← *.npy tiles

    Parameters
    ----------
    base_train_dirs : list[str]
        One entry per region, e.g. ["…/Kl/tensors/train", "…/SG/tensors/train"].
    active_buckets  : list[str]
        Bucket names to include this phase.
    bucket_weights  : list[float]
        Desired probability mass per active bucket (need not sum to 1).
    train_crop      : int
        Spatial crop size in pixels (default 128).
    """

    def __init__(self, base_train_dirs, active_buckets, bucket_weights, train_crop: int = 128):
        # NB: intentionally NOT calling super().__init__ — see class docstring.
        if len(active_buckets) != len(bucket_weights):
            raise ValueError(
                f"active_buckets ({len(active_buckets)}) and "
                f"bucket_weights ({len(bucket_weights)}) must have equal length."
            )
        if isinstance(base_train_dirs, str):
            base_train_dirs = [base_train_dirs]

        # Attributes the inherited train-mode __getitem__ relies on
        self.mode        = "train"
        self.train_crop  = train_crop
        self.val_crop    = 256
        self.val_overlap = 64
        self.val_pad     = 128

        # Normalise bucket weights so they sum to 1 (WeightedRandomSampler renormalises
        # anyway, but this keeps per_file weights interpretable).
        total_w = float(sum(bucket_weights))
        norm_w  = [w / total_w for w in bucket_weights]

        # ── Collect files and per-file sampling weights ────────────────────────
        # Weighting is at the BUCKET level: each active bucket gets total mass
        # `bucket_weight`, split equally over its files → per_file = weight / n_files.
        # This makes sampling independent of how many tiles a bucket happens to hold.
        self.filepaths     = []
        self.sample_weights = []
        self.bucket_stats  = {}      # bucket -> {region_dir -> count, "TOTAL" -> merged}

        print(f"[CurriculumDataset] active_buckets = {active_buckets}")
        print(f"   {'bucket':>10s} | " + " | ".join(f"{os.path.basename(os.path.dirname(os.path.dirname(d))):>6s}" for d in base_train_dirs) + " |  merged")
        print(f"   {'-' * 10}-+-" + "-+-".join("-" * 6 for _ in base_train_dirs) + "-+--------")

        for bucket, bw in zip(active_buckets, norm_w):
            merged_files = []
            per_region   = {}
            for base_dir in base_train_dirs:
                bucket_dir = os.path.join(base_dir, bucket)
                region_files = []
                if os.path.isdir(bucket_dir):
                    for root, _, files in os.walk(bucket_dir):
                        for f in sorted(files):            # sorted for reproducibility
                            if f.endswith(".npy"):
                                region_files.append(os.path.join(root, f))
                else:
                    print(f"   [WARN] missing bucket dir: {bucket_dir}")
                per_region[base_dir] = len(region_files)
                merged_files.extend(region_files)

            n = len(merged_files)
            self.bucket_stats[bucket] = {**per_region, "TOTAL": n}

            region_counts = " | ".join(f"{per_region[d]:6d}" for d in base_train_dirs)
            print(f"   {bucket:>10s} | {region_counts} | {n:6d}")

            if n == 0:
                print(f"   [WARN] no .npy in bucket '{bucket}' — skipped.")
                continue

            per_file_weight = bw / n                      # bucket-level weighting
            self.filepaths.extend(merged_files)
            self.sample_weights.extend([per_file_weight] * n)

        print(f"   {'-' * 10}-+-" + "-+-".join("-" * 6 for _ in base_train_dirs) + "-+--------")
        print(f"   {'ACTIVE':>10s} | pool size = {len(self.filepaths)} tiles\n")

        if len(self.filepaths) == 0:
            raise RuntimeError("CurriculumDataset built an empty pool — check base dirs / buckets.")

        # ── Spatial noise cache (identical construction to HMATensorDataset) ────
        from scipy.ndimage import gaussian_filter
        raw_noise = np.random.normal(0, 1.0, (2048, 2048)).astype(np.float32)
        smoothed  = gaussian_filter(raw_noise, sigma=2.0)
        self.noise_cache = (smoothed / np.max(np.abs(smoothed))) * 1.5

    # __len__ and __getitem__ (train branch) and _package_dg_vdsr are inherited.


# ==============================================================================
# 3. DATALOADER FACTORY
# ==============================================================================

def count_full_training_set(base_train_dirs) -> int:
    """
    Count every .npy tile under the train dirs across ALL buckets and regions.

    This fixed number is used as ``num_samples`` for the WeightedRandomSampler so the
    epoch length (step count) stays roughly constant across phases regardless of how
    small the currently-active bucket pool is.
    """
    if isinstance(base_train_dirs, str):
        base_train_dirs = [base_train_dirs]
    total = 0
    for base_dir in base_train_dirs:
        for root, _, files in os.walk(base_dir):
            total += sum(1 for f in files if f.endswith(".npy"))
    return total


def create_curriculum_dataloader(
    phase:            dict,
    base_train_dirs,
    batch_size:       int,
    num_workers:      int,
    full_train_size:  int,
    prefetch_factor:  int  = 4,
    pin_memory:       bool = True,
    train_crop:       int  = 128,
) -> DataLoader:
    """
    Build a training DataLoader for the given curriculum *phase* dict.

    Two paths:

    • Stage 1 (phase["use_full_dataset"] is True): build a standard DataLoader over the
      FULL unbucketed HMATensorDataset (train mode) with plain shuffle=True and NO
      sampler. This reuses the exact augmentation / packaging of the single-phase
      train_lidar.ipynb, so Stage 1 is a like-for-like full-dataset run. full_train_size
      is unused here (shuffle already sees every tile once per epoch).

    • Phase 1+ (bucketed): wrap CurriculumDataset in a WeightedRandomSampler with
      replacement=True and ``num_samples=full_train_size`` (the size of the FULL
      unbucketed training set), so every phase runs approximately the same number of
      optimisation steps even when only a tiny bucket is active.

    Call this once at epoch 1 and again at each phase boundary (is_phase_boundary()).
    """
    # ── Stage 1: full unbucketed dataset, plain shuffle (no WeightedRandomSampler) ──
    if phase.get("use_full_dataset"):
        dataset = HMATensorDataset(base_train_dirs, mode="train", train_crop=train_crop)
        prefetch = prefetch_factor if num_workers > 0 else None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,                  # plain uniform shuffle over all 1649 tiles
            num_workers=num_workers,
            prefetch_factor=prefetch,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=(num_workers > 0),
        )

    # ── Phase 1+: bucketed pool + WeightedRandomSampler (existing path) ──
    dataset = CurriculumDataset(
        base_train_dirs=base_train_dirs,
        active_buckets=phase["active_buckets"],
        bucket_weights=phase["bucket_weights"],
        train_crop=train_crop,
    )

    weights = torch.tensor(dataset.sample_weights, dtype=torch.float64)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=full_train_size,   # fixed → constant epoch length across phases
        replacement=True,
    )

    prefetch = prefetch_factor if num_workers > 0 else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,               # WeightedRandomSampler — replaces shuffle=True
        num_workers=num_workers,
        prefetch_factor=prefetch,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    return loader


# ==============================================================================
# 4. STREAM FREEZE / UNFREEZE HELPER
# ==============================================================================

def set_stream_trainable(module: nn.Module, trainable: bool) -> None:
    """
    Freeze or unfreeze all parameters of *module* (Stream B in practice).

    trainable=False (Phase 0 freeze):
      - backward() never populates .grad for these params.
      - optimizer.step() and clip_grad_norm_() are safe no-ops on grad=None — NO
        special-casing needed in the train loop; just keep calling them.
      - forward still runs, so alpha stays near its sigmoid(-2) ≈ 0.119 init,
        letting Stream A own the topography gradient during pretraining.

    trainable=True (unfreeze at epoch 31):
      - gradients resume from the next backward call.
      - call BEFORE the first train_one_epoch of epoch 31, and pair with the
        optimizer_b LR re-warmup to avoid a gradient spike.
    """
    for p in module.parameters():
        p.requires_grad = trainable
    label = "UNFROZEN (trainable)" if trainable else "FROZEN  (no grad)"
    print(f"  [Curriculum] Stream B → {label}")


# ==============================================================================
# 5. SANITY CHECK
# ==============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 78)
    print("curriculum_dataset.py  —  Schedule Sanity Check")
    print("=" * 78)

    # ── Phase / lambda_pin smoke test at requested epochs ──────────────────────
    print(f"\n{'Epoch':>6}  {'Phase':>5}  {'val/ep':>6}  {'λpin':>7}  Active Buckets")
    print("-" * 78)
    for ep in [1, 30, 31, 60, 61, 100, 101, 150]:
        ph = get_curriculum_phase(ep)
        lp = interp_lambda_pin(ph, ep)
        boundary = "  ◄ BOUNDARY" if is_phase_boundary(ep) else ""
        buckets = "FULL (unbucketed)" if ph.get("use_full_dataset") else ", ".join(ph["active_buckets"])
        print(f"  {ep:4d}    Ph{ph['phase_idx']}    "
              f"{ph['val_every']:6d}  {lp:7.4f}  {buckets}{boundary}")

    print(f"\nPhase boundaries : {PHASE_START_EPOCHS}")
    print(f"Stream B unfreeze: epoch {STREAM_B_UNFREEZE_EPOCH}")

    # ── Optional live file-count check against a real dataset root ──────────────
    # Usage:  python curriculum_dataset.py  <BASE_DIR>
    #   where BASE_DIR contains Kl/tensors/train and SG/tensors/train
    if len(sys.argv) > 1:
        base = sys.argv[1]
        base_train_dirs = [
            os.path.join(base, "Kl", "tensors", "train"),
            os.path.join(base, "SG", "tensors", "train"),
        ]
        full = count_full_training_set(base_train_dirs)
        print("\n" + "=" * 78)
        print(f"Full unbucketed training set (sampler num_samples): {full} tiles")
        print("=" * 78)

        # Print the pool size for each stage/phase. Stage 1 is the full unbucketed
        # pool (plain shuffle); Phase 1+ are the merged bucket pools.
        for idx in range(len(CURRICULUM_PHASES)):
            ep = PHASE_START_EPOCHS[idx]
            ph = get_curriculum_phase(ep)
            if ph.get("use_full_dataset"):
                print(f"\n--- Stage {idx + 1} (epoch {ep}) FULL unbucketed pool ---")
                for d in base_train_dirs:
                    n = sum(1 for root, _, files in os.walk(d)
                            for f in files if f.endswith(".npy"))
                    print(f"    {os.path.basename(os.path.dirname(os.path.dirname(d)))}: {n} tiles  ({d})")
                print(f"    → full pool = {full} tiles, plain shuffle (no sampler)")
            else:
                print(f"\n--- Phase {idx} (epoch {ep}) merged bucket pool ---")
                ds = CurriculumDataset(base_train_dirs, ph["active_buckets"], ph["bucket_weights"])
                print(f"    → dataset len (active pool) = {len(ds)}, "
                      f"sampler num_samples = {full}")
    else:
        print("\n(Pass a dataset BASE_DIR as argv[1] to also print live file counts:")
        print("   python curriculum_dataset.py D:/Projects/dem-lidar/Dataset )")
