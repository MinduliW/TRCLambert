"""Two-stage self-supervised TRC training for Lambert+J2.

Stage 1: Learn u0 (first head) to match Lambert dv1.
         - Uses stage1_only=True to SKIP coast propagation and refinement loop.
         - Only optimises state_encoder, init_decoder, h_proj, l_proj.
         - ~10x faster per epoch than full forward pass.

Stage 2: Learn iterative refinement to reduce terminal J2 position error.
         - Full forward pass with coast + refinement.
         - Stage 1 weights are FROZEN — only refinement head trains.
         - This prevents the Lambert head from being corrupted.

This keeps the role split explicit:
- init_decoder learns Lambert-like initialization (Stage 1, frozen in Stage 2)
- iterative updates learn J2 residual correction (Stage 2)
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
for _p in (THIS_DIR, PARENT_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from lambert_trc_model import LambertDataset, LambertTRC
from trc import NetConfig, count_params


# ── Config ──────────────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    quick: bool = False
    do_train: bool = True
    resume: bool = False
    data_dir: Path = THIS_DIR / "data"
    ckpt_dir: Path = THIS_DIR / "checkpoints"

    dv_max: float = 5.0
    lr_stage1: float = 1e-4
    lr_stage2: float = 1e-4

    pos_scale_m: float = 300000.0   # 300 km — matches typical J2 error scale
    correction_scale: float = 0.05
    max_correction: float = 0.5
    use_tof_log: bool = True
    state_repr: str = "cartesian"
    tof_max_seconds: float = None

    # Stage-1 (u0-only objective)
    lambda_u0_s1: float = 1.0
    lambda_pos_s1: float = 0.0
    lambda_ps_s1: float = 0.0

    # Stage-2 (refinement objective — no u0 loss needed since S1 is frozen)
    lambda_u0_s2: float = 0.0
    lambda_pos_s2: float = 1.0
    lambda_ps_s2: float = 0.1

    anchor_to_lambert: bool = False

    # Adaptive K for Stage 2
    K_min: int = 3      # minimum iterations before early-stop check
    K_max: int = 3       # maximum iterations
    tol_km: float = 0.1 # early-stop when ALL samples in batch < tol_km

    @property
    def epochs_stage1(self) -> int:
        return 10 if self.quick else 1000

    @property
    def epochs_stage2(self) -> int:
        return 15 if self.quick else 300

    @property
    def batch_size(self) -> int:
        return 32 if self.quick else 64

    @property
    def coast_max_step_s(self) -> float:
        return 60.0 if self.quick else 45.0

    @property
    def net_cfg(self) -> NetConfig:
        if self.quick:
            return NetConfig(d_z=128, d_h=256, n_heads=4, n_blocks=2, K=3, n_inner=6)
        return NetConfig(d_z=256, d_h=512, n_heads=8, n_blocks=3, K=3, n_inner=6)

    @property
    def train_path(self) -> Path:
        return self.data_dir / "lambert_train.npz"

    @property
    def test_path(self) -> Path:
        return self.data_dir / "lambert_test.npz"

    @property
    def ckpt_stage1(self) -> Path:
        suffix = "" if self.state_repr == "cartesian" else f"_{self.state_repr}"
        return self.ckpt_dir / f"trc_lambert_u0_stage1_best{suffix}.pt"

    @property
    def ckpt_stage2(self) -> Path:
        suffix = "" if self.state_repr == "cartesian" else f"_{self.state_repr}"
        return self.ckpt_dir / f"trc_lambert_u0_curriculum_best{suffix}.pt"


# ── Loss ────────────────────────────────────────────────────────────────────

class StageLoss(nn.Module):
    def __init__(self, lambda_u0: float, lambda_pos: float, lambda_ps: float):
        super().__init__()
        self.lambda_u0 = lambda_u0
        self.lambda_pos = lambda_pos
        self.lambda_ps = lambda_ps

    def forward(self, output, batch, model):
        pos_errors = output["pos_errors"]
        dv_iters = output["dv_iterations"]
        u0 = dv_iters[0]
        dv_final = output["dv_final"]

        # u0 loss (may be 0 in Stage 2 if frozen)
        # if self.lambda_u0 > 0:
        #     L_u0 = F.mse_loss(u0, batch["dv1"]) / (model.dv_scale ** 2)
        #     # u0 loss — Huber to prevent outlier domination
        if self.lambda_u0 > 0:
            if model.state_repr == "spherical":
                u0_repr = output["dv_repr_iterations"][0]
                dv1_target = model.encode_control(batch["dv1"])
                L_u0 = F.huber_loss(u0_repr, dv1_target, delta=0.2)
            else:
                L_u0 = F.huber_loss(
                    u0 / model.dv_scale,
                    batch["dv1"] / model.dv_scale,
                    delta=3.0,
                )
        else:
            L_u0 = torch.tensor(0.0, device=u0.device)

        # Position loss
        if self.lambda_pos > 0 and len(pos_errors) > 0:
            pos_err_km = pos_errors[-1]
            pos_norm = getattr(model, "pos_scale", torch.tensor(100.0, device=pos_err_km.device))
            L_pos = (pos_err_km / pos_norm).pow(2).mean()
        else:
            L_pos = torch.tensor(0.0, device=u0.device)

        # Process supervision
        if self.lambda_ps > 0 and len(pos_errors) >= 2:
            err0 = pos_errors[0].detach().clamp(min=1e-3)
            normed = [e / err0 for e in pos_errors]
            improvements = [normed[k] - normed[k + 1] for k in range(len(normed) - 1)]
            L_proc = -torch.stack(improvements).mean()
        else:
            L_proc = torch.tensor(0.0, device=u0.device)

        loss = self.lambda_u0 * L_u0 + self.lambda_pos * L_pos + self.lambda_ps * L_proc

        with torch.no_grad():
            imp_metric = 0.0
            if len(pos_errors) >= 2:
                e0 = pos_errors[0].clamp(min=1e-3)
                for k in range(len(pos_errors) - 1):
                    imp_metric += ((pos_errors[k] - pos_errors[k + 1]) / e0).mean().item()
                imp_metric /= (len(pos_errors) - 1)

        return loss, {
            "loss": loss.item(),
            "L_u0": L_u0.item(),
            "L_pos": L_pos.item(),
            "L_proc": L_proc.item(),
            "imp_metric": imp_metric,
            "u0_res_norm": torch.norm(u0 - batch["dv1"], dim=-1).mean().item() * 1000.0,
            "dv_res_norm": torch.norm(dv_final - batch["dv1"], dim=-1).mean().item() * 1000.0,
            "pos_err_K": pos_errors[-1].mean().item() if len(pos_errors) > 0 else 0.0,
        }


# ── CLI ─────────────────────────────────────────────────────────────────────
_DEFAULTS = RunConfig()

def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="Two-stage self-supervised TRC training.")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no_train", action="store_true")
    p.add_argument("--resume", action="store_true")
    stage_group = p.add_mutually_exclusive_group()
    stage_group.add_argument("--stage1_only", action="store_true")
    stage_group.add_argument("--stage2_only", action="store_true")
    p.add_argument("--data_dir", type=Path, default=_DEFAULTS.data_dir)
    p.add_argument("--ckpt_dir", type=Path, default=_DEFAULTS.ckpt_dir)
    p.add_argument("--dv_max", type=float, default=_DEFAULTS.dv_max)
    p.add_argument("--lr_stage1", type=float, default=_DEFAULTS.lr_stage1)
    p.add_argument("--lr_stage2", type=float, default=_DEFAULTS.lr_stage2)
    p.add_argument("--pos_scale_m", type=float, default=_DEFAULTS.pos_scale_m)
    p.add_argument("--correction_scale", type=float, default=_DEFAULTS.correction_scale)
    p.add_argument("--max_correction", type=float, default=_DEFAULTS.max_correction)
    p.add_argument("--K_min", type=int, default=_DEFAULTS.K_min)
    p.add_argument("--K_max", type=int, default=_DEFAULTS.K_max)
    p.add_argument("--tol_km", type=float, default=_DEFAULTS.tol_km)
    p.add_argument("--tof_max_seconds", type=float, default=_DEFAULTS.tof_max_seconds,
                   help="Filter dataset to samples with TOF <= this value (seconds).")
    p.add_argument("--state_repr", choices=["cartesian", "spherical"], default=_DEFAULTS.state_repr)
    p.add_argument("--use_tof_log", dest="use_tof_log", action="store_true")
    p.add_argument("--no_tof_log", dest="use_tof_log", action="store_false",
                   help="Disable log1p TOF scaling; use linear TOF normalization.")
    p.set_defaults(use_tof_log=_DEFAULTS.use_tof_log)
    a = p.parse_args()

    cfg = RunConfig(
        quick=a.quick,
        do_train=not a.no_train,
        resume=a.resume,
        data_dir=a.data_dir,
        ckpt_dir=a.ckpt_dir,
        dv_max=a.dv_max,
        lr_stage1=a.lr_stage1,
        lr_stage2=a.lr_stage2,
        pos_scale_m=a.pos_scale_m,
        correction_scale=a.correction_scale,
        max_correction=a.max_correction,
        K_min=a.K_min,
        K_max=a.K_max,
        tol_km=a.tol_km,
        use_tof_log=a.use_tof_log,
        state_repr=a.state_repr,
        tof_max_seconds=a.tof_max_seconds,
    )
    cfg._stage1_only = a.stage1_only
    cfg._stage2_only = a.stage2_only
    return cfg


def get_dataset_scales(dataset: LambertDataset, use_tof_log: bool = True):
    """Return normalization stats for encoder inputs plus dv scale."""
    r_min = torch.cat([dataset.r0, dataset.r_target], dim=0).amin(dim=0)
    r_max = torch.cat([dataset.r0, dataset.r_target], dim=0).amax(dim=0)
    v_min = dataset.v0.amin(dim=0)
    v_max = dataset.v0.amax(dim=0)
    r_mag_all = torch.cat([torch.norm(dataset.r0, dim=-1), torch.norm(dataset.r_target, dim=-1)], dim=0)
    v_mag_all = torch.norm(dataset.v0, dim=-1)
    dv_mag_all = torch.norm(dataset.dv1, dim=-1)
    tof_min = dataset.tof.min().item()
    tof_max = dataset.tof.max().item()
    dv_scale = torch.norm(dataset.dv1, dim=-1).median().item()
    tof_log_min = torch.log1p(dataset.tof).min().item() if use_tof_log else None
    tof_log_max = torch.log1p(dataset.tof).max().item() if use_tof_log else None
    nrev_min = float(dataset.nrev.min().item()) if len(dataset.nrev) > 0 else 0.0
    nrev_max = float(dataset.nrev.max().item()) if len(dataset.nrev) > 0 else 1.0
    r_mag_min = float(r_mag_all.min().item())
    r_mag_max = float(r_mag_all.max().item())
    v_mag_min = float(v_mag_all.min().item())
    v_mag_max = float(v_mag_all.max().item())
    dv_mag_min = float(dv_mag_all.min().item())
    dv_mag_max = float(dv_mag_all.max().item())
    return (
        r_min, r_max, v_min, v_max, tof_min, tof_max,
        tof_log_min, tof_log_max, dv_scale, nrev_min, nrev_max,
        r_mag_min, r_mag_max, v_mag_min, v_mag_max, dv_mag_min, dv_mag_max,
    )


def print_dataset_scales(r_min, r_max, v_min, v_max, tof_min, tof_max, dv_scale,
                         nrev_min, nrev_max, r_mag_min, r_mag_max, v_mag_min, v_mag_max,
                         dv_mag_min, dv_mag_max,
                         use_tof_log=False, tof_log_min=None, tof_log_max=None):
    """Human-friendly scale printout used by beginners."""
    r_min = [float(x) for x in r_min]
    r_max = [float(x) for x in r_max]
    v_min = [float(x) for x in v_min]
    v_max = [float(x) for x in v_max]
    print(f"\n--- Data scales ---")
    print(f"  r_min = [{r_min[0]:.1f}, {r_min[1]:.1f}, {r_min[2]:.1f}] km")
    print(f"  r_max = [{r_max[0]:.1f}, {r_max[1]:.1f}, {r_max[2]:.1f}] km")
    print(f"  v_min = [{v_min[0]:.4f}, {v_min[1]:.4f}, {v_min[2]:.4f}] km/s")
    print(f"  v_max = [{v_max[0]:.4f}, {v_max[1]:.4f}, {v_max[2]:.4f}] km/s")
    print(f"  |r| range = [{r_mag_min:.1f}, {r_mag_max:.1f}] km")
    print(f"  |v| range = [{v_mag_min:.4f}, {v_mag_max:.4f}] km/s")
    print(f"  |dv| range = [{dv_mag_min:.4f}, {dv_mag_max:.4f}] km/s")
    print(f"  tof range = [{tof_min:.1f}, {tof_max:.1f}] s")
    if use_tof_log and tof_log_min is not None and tof_log_max is not None:
        print(f"  log1p(tof) range = [{tof_log_min:.3f}, {tof_log_max:.3f}]")
    print(f"  dv_scale = {dv_scale:.4f} km/s  ({dv_scale*1000:.1f} m/s)")
    print(f"  nrev range = [{nrev_min:.1f}, {nrev_max:.1f}]")


def build_scales_metadata(r_min, r_max, v_min, v_max, tof_min, tof_max,
                          dv_scale, tof_log_min, tof_log_max, nrev_min, nrev_max,
                          r_mag_min, r_mag_max, v_mag_min, v_mag_max, dv_mag_min, dv_mag_max, cfg):
    return {
        "r_min": r_min.tolist(),
        "r_max": r_max.tolist(),
        "v_min": v_min.tolist(),
        "v_max": v_max.tolist(),
        "r_mag_min": r_mag_min,
        "r_mag_max": r_mag_max,
        "v_mag_min": v_mag_min,
        "v_mag_max": v_mag_max,
        "dv_mag_min": dv_mag_min,
        "dv_mag_max": dv_mag_max,
        "tof_min": tof_min,
        "tof_max": tof_max,
        "tof_log_min": tof_log_min,
        "tof_log_max": tof_log_max,
        "use_tof_log": cfg.use_tof_log,
        "dv": dv_scale,
        "nrev_min": nrev_min,
        "nrev_max": nrev_max,
        "corr": cfg.correction_scale,
        "pos_m": cfg.pos_scale_m,
    }


# ── Training utilities ──────────────────────────────────────────────────────

def get_stage1_params(model):
    """Return only the parameters needed for Stage 1 (Lambert head)."""
    stage1_modules = [
        model.state_encoder,
        model.init_decoder,
        model.h_proj,
        model.l_proj,
    ]
    params = []
    for mod in stage1_modules:
        params.extend(mod.parameters())
    params.append(model.H_init)
    params.append(model.L_init)
    return params


def freeze_stage1(model):
    """Freeze all Stage 1 parameters so they are not updated during Stage 2."""
    frozen_count = 0
    for p in get_stage1_params(model):
        p.requires_grad = False
        frozen_count += p.numel()
    return frozen_count


def get_trainable_params(model):
    """Return only parameters with requires_grad=True."""
    return [p for p in model.parameters() if p.requires_grad]


def filter_finite_batch(b):
    """Drop rows that contain NaN/Inf in core tensors."""
    finite = torch.isfinite(b["r0"]).all(-1) & torch.isfinite(b["v0"]).all(-1)
    finite &= torch.isfinite(b["r_target"]).all(-1) & torch.isfinite(b["v_target"]).all(-1)
    finite &= torch.isfinite(b["tof"]).squeeze(-1)
    if "dv1" in b and torch.is_tensor(b["dv1"]):
        finite &= torch.isfinite(b["dv1"]).all(-1)
    if "dv1_corrected" in b and torch.is_tensor(b["dv1_corrected"]):
        finite &= torch.isfinite(b["dv1_corrected"]).all(-1)

    if finite.all():
        return b

    out = {}
    B = b["r0"].shape[0]
    for k, v in b.items():
        if torch.is_tensor(v) and v.shape[:1] == b["r0"].shape[:1]:
            out[k] = v[finite]
        elif torch.is_tensor(v) and v.ndim == 0 and int(B) == 1:
            out[k] = v
        else:
            out[k] = v
    return out


def evaluate(model, loader, criterion, device, anchor_to_lambert=False, stage1_only=False,
             K_min=None, K_max=None):
    """Evaluate — always runs at full K_max, no early stopping."""
    model.eval()
    losses, posk, imp, u0res, dvres = [], [], [], [], []
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            b = filter_finite_batch(b)
            if b["r0"].numel() == 0:
                continue
            dv_seed = b["dv1"] if anchor_to_lambert else None
            out = model(b["r0"], b["v0"], b["r_target"], tof=b["tof"],
                        dv_lambert=dv_seed, stage1_only=stage1_only,
                        nrev=b.get("nrev"), ncase=b.get("ncase"), prograde=b.get("prograde"),
                        K_min=K_max, K_max=K_max)
            _, m = criterion(out, b, model)
            if not np.isfinite(m["loss"]):
                continue
            losses.append(m["loss"])
            posk.append(m["pos_err_K"])
            imp.append(m["imp_metric"])
            u0res.append(m["u0_res_norm"])
            dvres.append(m["dv_res_norm"])
    if len(losses) == 0:
        return {
            "loss": float("inf"),
            "pos_err_K": float("nan"),
            "imp": float("nan"),
            "u0_res_norm": float("nan"),
            "dv_res_norm": float("nan"),
        }
    return {
        "loss": float(np.mean(losses)),
        "pos_err_K": float(np.mean(posk)),
        "imp": float(np.mean(imp)),
        "u0_res_norm": float(np.mean(u0res)),
        "dv_res_norm": float(np.mean(dvres)),
    }


def _load_state_dict_compat(model, source_sd):
    """Load checkpoint with graceful handling of old/new architecture mismatch."""
    if source_sd is None:
        return

    model_sd = model.state_dict()
    adapted = {}
    skipped = []
    for k, v in source_sd.items():
        if k not in model_sd:
            skipped.append(k)
            continue

        target = model_sd[k]
        if v.shape == target.shape:
            adapted[k] = v
            continue

        if k == "state_encoder.0.weight" and v.ndim == 2 and target.ndim == 2:
            # Old model used [r0(3), v0(3), r_target(3), v_target(3), tof(1)] = 13
            # New model uses [r0(3), v0(3), r_target(3), tof(1)] = 10.
            if v.shape[0] == target.shape[0] and v.shape[1] == 13 and target.shape[1] == 10:
                new_w = torch.zeros_like(target)
                new_w[:, 0:3] = v[:, 0:3]
                new_w[:, 3:6] = v[:, 3:6]
                new_w[:, 6:9] = v[:, 6:9]
                new_w[:, 9:10] = v[:, 12:13]
                adapted[k] = new_w
                continue

        skipped.append(f"{k}: src={tuple(v.shape)} target={tuple(target.shape)}")

    missing, unexpected = model.load_state_dict(adapted, strict=False)
    if skipped:
        print(f"  [ckpt compat] skipped {len(skipped)} keys (missing/different shape).")
        for s in skipped[:6]:
            print(f"    - {s}")
        if len(skipped) > 6:
            print(f"    ... and {len(skipped) - 6} more")
    if missing:
        print(f"  [ckpt compat] missing {len(missing)} keys in checkpoint (set by init/defaults).")
        for k in list(missing)[:6]:
            print(f"    - {k}")
    if unexpected:
        print(f"  [ckpt compat] unexpected {len(unexpected)} keys in checkpoint.")
        for k in list(unexpected)[:6]:
            print(f"    - {k}")


def save_ckpt(model, optimizer, scheduler, epoch, best_val, history, path, cfg, scales, stage_name):
    torch.save(
        {
            "epoch": epoch,
            "best_val": best_val,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "net_cfg": vars(cfg.net_cfg),
            "scales": scales,
            "cfg": vars(cfg),
            "stage": stage_name,
        },
        path,
    )


def train_stage(stage_name, model, train_loader, test_loader, device, cfg, criterion,
                optimizer, scheduler, epochs, ckpt_path, scales,
                anchor_to_lambert=False, stage1_only=False, resume=False,
                K_min=None, K_max=None, tol_km=None):
    history = {"train_loss": [], "val_loss": [], "val_pos": [], "val_u0res_m": [], "val_dvres_m": []}
    start_epoch, best_val = 1, float("inf")

    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        _load_state_dict_compat(model, ckpt.get("model_state_dict", {}))
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        history = ckpt.get("history", history)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", float("inf")))
        if not np.isfinite(best_val):
            best_val = float("inf")
        print(f"Resuming {stage_name} from epoch {start_epoch} (best={best_val:.4f})")

    for ep in range(start_epoch, start_epoch + epochs):
        t0 = time.time()
        model.train()
        ep_losses = []
        n_skipped = 0
        k_used_sum = 0
        n_batches = 0
        for b in train_loader:
            b = {k: v.to(device) for k, v in b.items()}
            b = filter_finite_batch(b)
            if b["r0"].numel() == 0:
                n_skipped += 1
                continue
            optimizer.zero_grad()
            dv_seed = b["dv1"] if anchor_to_lambert else None
            out = model(b["r0"], b["v0"], b["r_target"], tof=b["tof"],
                        dv_lambert=dv_seed, stage1_only=stage1_only,
                        nrev=b.get("nrev"), ncase=b.get("ncase"), prograde=b.get("prograde"),
                        K_min=K_min, K_max=K_max, tol_km=tol_km)
            loss, _ = criterion(out, b, model)
            if not torch.isfinite(loss):
                n_skipped += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_losses.append(loss.item())
            k_used_sum += out.get("K_used", K_max or 0)
            n_batches += 1

        scheduler.step()
        if len(ep_losses) == 0:
            print(f"{stage_name} [ep {ep:03d}] all train batches were non-finite after filtering; stopping stage.")
            break

        val = evaluate(model, test_loader, criterion, device,
                       anchor_to_lambert=anchor_to_lambert, stage1_only=stage1_only,
                       K_min=K_max, K_max=K_max)
        train_loss = float(np.mean(ep_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_pos"].append(val["pos_err_K"])
        history["val_u0res_m"].append(val["u0_res_norm"])
        history["val_dvres_m"].append(val["dv_res_norm"])

        dt = time.time() - t0
        avg_k = k_used_sum / max(n_batches, 1)
        print(
            f"[{stage_name} {ep:03d}] train={train_loss:.4f} val={val['loss']:.4f} "
            f"posK={val['pos_err_K']:.3f}km u0_res={val['u0_res_norm']:.2f}m/s "
            f"dv_res={val['dv_res_norm']:.2f}m/s imp={val['imp']:.3f} "
            f"K_avg={avg_k:.1f} skip={n_skipped} ({dt:.1f}s)"
        )

        if val["loss"] < best_val:
            best_val = val["loss"]
            save_ckpt(
                model, optimizer, scheduler, ep, best_val, history,
                ckpt_path, cfg, scales, stage_name,
            )
            print(f"  -> saved best: {ckpt_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    cfg = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg.ckpt_dir.mkdir(exist_ok=True)
    if not cfg.train_path.exists() or not cfg.test_path.exists():
        raise FileNotFoundError("Missing lambert_train.npz/lambert_test.npz in data_dir.")

    print(f"Device: {device}")
    print(f"TRAIN={cfg.train_path}")
    print(f"TEST ={cfg.test_path}")
    print(f"CKPT1={cfg.ckpt_stage1}")
    print(f"CKPT2={cfg.ckpt_stage2}")
    print("MODE = two-stage curriculum (u0 pretrain -> refinement)")
    print(f"anchor_to_lambert={cfg.anchor_to_lambert}")

    train_ds = LambertDataset(cfg.train_path, tof_max_seconds=cfg.tof_max_seconds)
    test_ds = LambertDataset(cfg.test_path, tof_max_seconds=cfg.tof_max_seconds)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    r_min, r_max, v_min, v_max, tof_min, tof_max, tof_log_min, tof_log_max, dv_scale, nrev_min, nrev_max, r_mag_min, r_mag_max, v_mag_min, v_mag_max, dv_mag_min, dv_mag_max = get_dataset_scales(
        train_ds, use_tof_log=cfg.use_tof_log
    )
    print_dataset_scales(
        r_min, r_max, v_min, v_max, tof_min, tof_max, dv_scale, nrev_min, nrev_max,
        r_mag_min, r_mag_max, v_mag_min, v_mag_max, dv_mag_min, dv_mag_max,
        use_tof_log=cfg.use_tof_log,
        tof_log_min=tof_log_min,
        tof_log_max=tof_log_max,
    )
    dv1_norms = torch.norm(train_ds.dv1, dim=-1)
    print(f"  dv1 norms: min={dv1_norms.min():.4f} max={dv1_norms.max():.4f} "
          f"mean={dv1_norms.mean():.4f} km/s")
    print(f"  dv_max clamp = {cfg.dv_max} km/s")
    n_clipped = (dv1_norms > cfg.dv_max).sum().item()
    print(f"  samples exceeding dv_max: {n_clipped}/{len(train_ds)} "
          f"({100*n_clipped/len(train_ds):.1f}%)")
    print(f"  pos_scale = {cfg.pos_scale_m/1000:.1f} km")
    print()

    model = LambertTRC(
        cfg.net_cfg,
        max_coast_step_s=cfg.coast_max_step_s,
        dv_max=cfg.dv_max,
        correction_scale=cfg.correction_scale,
        max_correction=cfg.max_correction,
        state_repr=cfg.state_repr,
    ).to(device)
    model.set_normalization(
        r_min, r_max, v_min, v_max, tof_min, tof_max, dv_scale,
        pos_scale_km=cfg.pos_scale_m / 1000.0,
        use_tof_log=cfg.use_tof_log,
        tof_log_min=tof_log_min if cfg.use_tof_log else None,
        tof_log_max=tof_log_max if cfg.use_tof_log else None,
        nrev_min=nrev_min,
        nrev_max=nrev_max,
        r_mag_min=r_mag_min,
        r_mag_max=r_mag_max,
        v_mag_min=v_mag_min,
        v_mag_max=v_mag_max,
        dv_mag_min=dv_mag_min,
        dv_mag_max=dv_mag_max,
    )

    total_params = count_params(model)
    s1_params = sum(p.numel() for p in get_stage1_params(model))
    print(f"Total params: {total_params:,}")
    print(f"Stage1 params (encoder+init_decoder): {s1_params:,} ({100*s1_params/total_params:.1f}%)")
    print(f"Stage2 params (refinement head): {total_params - s1_params:,} ({100*(total_params-s1_params)/total_params:.1f}%)")
    print(f"Stage1 epochs={cfg.epochs_stage1}, Stage2 epochs={cfg.epochs_stage2}")

    scales = build_scales_metadata(
        r_min, r_max, v_min, v_max, tof_min, tof_max,
        dv_scale, tof_log_min, tof_log_max, nrev_min, nrev_max,
        r_mag_min, r_mag_max, v_mag_min, v_mag_max, dv_mag_min, dv_mag_max, cfg
    )

    if not cfg.do_train:
        if cfg.ckpt_stage2.exists():
            ckpt = torch.load(cfg.ckpt_stage2, map_location=device, weights_only=False)
            _load_state_dict_compat(model, ckpt.get("model_state_dict", {}))
        elif cfg.ckpt_stage1.exists():
            ckpt = torch.load(cfg.ckpt_stage1, map_location=device, weights_only=False)
            _load_state_dict_compat(model, ckpt.get("model_state_dict", {}))
        else:
            raise FileNotFoundError("No stage checkpoint found for --no_train.")
    else:
        run_stage1 = not getattr(cfg, "_stage2_only", False)
        run_stage2 = not getattr(cfg, "_stage1_only", False)

        # ── Stage 1: learn u0 ≈ Lambert dv1 ────────────────────────────────
        if run_stage1:
            print("\n=== Stage 1: u0 pretraining (Lambert imitation) ===")
            print("    stage1_only=True  → NO coast propagation, NO refinement loop")
            print(f"    Optimising only encoder + init_decoder ({s1_params:,} params)")

            crit1 = StageLoss(
                lambda_u0=cfg.lambda_u0_s1,
                lambda_pos=cfg.lambda_pos_s1,
                lambda_ps=cfg.lambda_ps_s1,
            )
            opt1 = torch.optim.AdamW(get_stage1_params(model), lr=cfg.lr_stage1)
            sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt1, T_max=cfg.epochs_stage1, eta_min=1e-6
            )
            train_stage(
                "S1", model, train_loader, test_loader, device, cfg,
                crit1, opt1, sch1, cfg.epochs_stage1, cfg.ckpt_stage1, scales,
                anchor_to_lambert=False, stage1_only=True, resume=cfg.resume,
            )

        # ── Stage 2: learn refinement on top of frozen u0 ──────────────────
        if run_stage2:
            print("\n=== Stage 2: refinement training (terminal J2 error) ===")
            print("    stage1_only=False → full forward pass with coast + refinement")

            # Load Stage 1 weights
            if cfg.ckpt_stage1.exists():
                ckpt1 = torch.load(cfg.ckpt_stage1, map_location=device, weights_only=False)
                _load_state_dict_compat(model, ckpt1.get("model_state_dict", {}))
                print(f"    Loaded stage1 weights: {cfg.ckpt_stage1}")
            else:
                print("    WARNING: No stage1 checkpoint found — using random init for Lambert head")

            # Freeze Stage 1 parameters
            frozen_count = freeze_stage1(model)
            trainable_params = get_trainable_params(model)
            trainable_count = sum(p.numel() for p in trainable_params)
            print(f"    Frozen (Stage 1): {frozen_count:,} params")
            print(f"    Trainable (Stage 2): {trainable_count:,} params")

            crit2 = StageLoss(
                lambda_u0=cfg.lambda_u0_s2,
                lambda_pos=cfg.lambda_pos_s2,
                lambda_ps=cfg.lambda_ps_s2,
            )
            # Only optimise trainable (unfrozen) parameters
            opt2 = torch.optim.AdamW(trainable_params, lr=cfg.lr_stage2)
            sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt2, T_max=cfg.epochs_stage2, eta_min=1e-6
            )
            train_stage(
                "S2", model, train_loader, test_loader, device, cfg,
                crit2, opt2, sch2, cfg.epochs_stage2, cfg.ckpt_stage2, scales,
                anchor_to_lambert=False, stage1_only=False, resume=cfg.resume,
                K_min=cfg.K_min, K_max=cfg.K_max, tol_km=cfg.tol_km,
            )

    # ── Final evaluation ────────────────────────────────────────────────────
    final_crit = StageLoss(
        lambda_u0=0.0,
        lambda_pos=cfg.lambda_pos_s2,
        lambda_ps=cfg.lambda_ps_s2,
    )
    final_val = evaluate(model, test_loader, final_crit, device, anchor_to_lambert=False, stage1_only=False)
    print("\nFinal evaluation (test):")
    print(
        f"  loss={final_val['loss']:.4f}, pos_err_K={final_val['pos_err_K']:.3f} km, "
        f"  u0_res={final_val['u0_res_norm']:.2f} m/s, dv_res={final_val['dv_res_norm']:.2f} m/s, "
        f"  imp={final_val['imp']:.3f}"
    )


if __name__ == "__main__":
    main()
