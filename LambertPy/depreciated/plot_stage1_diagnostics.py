"""Diagnostic plots for Stage 1 Lambert approximation quality.

Generates:
1. Training history (loss + u0 residual over epochs)
2. Predicted vs target dv1 (per component scatter)
3. Error distribution histogram
4. Error vs dv magnitude (does it struggle with large/small transfers?)
5. Error vs TOF (does it struggle with short/long transfers?)
"""

import argparse
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
for _p in (THIS_DIR, PARENT_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from lambert_trc_model import LambertDataset, LambertTRC
from trc import NetConfig

device = torch.device("cpu")
PLOTS_DIR = THIS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Stage 1 diagnostics plots.")
    parser.add_argument(
        "--state_repr",
        choices=["cartesian", "spherical"],
        default="cartesian",
        help="Checkpoint family to load when --checkpoint is not provided.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional explicit checkpoint path.",
    )
    parser.add_argument(
        "--output_tag",
        type=str,
        default=None,
        help="Suffix for output filenames. Defaults to the loaded state representation.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving them.",
    )
    return parser.parse_args()


def resolve_checkpoint(args) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    suffix = "" if args.state_repr == "cartesian" else "_spherical"
    return THIS_DIR / "checkpoints" / f"trc_lambert_u0_stage1_best{suffix}.pt"


args = parse_args()

# ── Load model ──────────────────────────────────────────────────────────────
ckpt_path = resolve_checkpoint(args)
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

net_cfg = NetConfig(**ckpt["net_cfg"])
cfg_saved = ckpt.get("cfg", {})
state_repr = cfg_saved.get("state_repr", "cartesian")
output_tag = args.output_tag or state_repr
model = LambertTRC(net_cfg, max_coast_step_s=45.0, dv_max=50.0, correction_scale=0.05, state_repr=state_repr).to(device)

# ── Load both datasets ──────────────────────────────────────────────────────
train_ds = LambertDataset(THIS_DIR / "data" / "lambert_train.npz")
test_ds = LambertDataset(THIS_DIR / "data" / "lambert_test.npz")

scales = ckpt.get("scales", {})
r_min = scales.get("r_min", torch.cat([train_ds.r0, train_ds.r_target], dim=0).amin(dim=0).tolist())
r_max = scales.get("r_max", torch.cat([train_ds.r0, train_ds.r_target], dim=0).amax(dim=0).tolist())
v_min = scales.get("v_min", train_ds.v0.amin(dim=0).tolist())
v_max = scales.get("v_max", train_ds.v0.amax(dim=0).tolist())
tof_min = scales.get("tof_min", train_ds.tof.min().item())
tof_max = scales.get("tof_max", train_ds.tof.max().item())
dv_scale = scales.get("dv", torch.norm(train_ds.dv1, dim=-1).median().item())
nrev_min = scales.get("nrev_min", float(train_ds.nrev.min().item()) if len(train_ds.nrev) > 0 else 0.0)
nrev_max = scales.get("nrev_max", float(train_ds.nrev.max().item()) if len(train_ds.nrev) > 0 else 1.0)
r_mag_min = scales.get("r_mag_min", float(torch.cat([torch.norm(train_ds.r0, dim=-1), torch.norm(train_ds.r_target, dim=-1)], dim=0).min().item()))
r_mag_max = scales.get("r_mag_max", float(torch.cat([torch.norm(train_ds.r0, dim=-1), torch.norm(train_ds.r_target, dim=-1)], dim=0).max().item()))
v_mag_min = scales.get("v_mag_min", float(torch.norm(train_ds.v0, dim=-1).min().item()))
v_mag_max = scales.get("v_mag_max", float(torch.norm(train_ds.v0, dim=-1).max().item()))
dv_mag_min = scales.get("dv_mag_min", float(torch.norm(train_ds.dv1, dim=-1).min().item()))
dv_mag_max = scales.get("dv_mag_max", float(torch.norm(train_ds.dv1, dim=-1).max().item()))
tof_log_min = scales.get("tof_log_min")
tof_log_max = scales.get("tof_log_max")
use_tof_log = scales.get("use_tof_log", False)
model.set_normalization(
    r_min, r_max, v_min, v_max, tof_min, tof_max, dv_scale,
    pos_scale_km=1.0,
    use_tof_log=use_tof_log,
    tof_log_min=tof_log_min if use_tof_log else None,
    tof_log_max=tof_log_max if use_tof_log else None,
    nrev_min=nrev_min,
    nrev_max=nrev_max,
    r_mag_min=r_mag_min,
    r_mag_max=r_mag_max,
    v_mag_min=v_mag_min,
    v_mag_max=v_mag_max,
    dv_mag_min=dv_mag_min,
    dv_mag_max=dv_mag_max,
)
missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
if missing:
    print(f"  load_state_dict: ignoring {len(missing)} missing keys from older checkpoint")
if unexpected:
    print(f"  load_state_dict: ignoring {len(unexpected)} unexpected keys from checkpoint")
model.eval()

history = ckpt.get("history", {})


def collect_predictions(dataset, label):
    """Run model on full dataset, return targets and predictions."""
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    all_pred, all_target, all_tof = [], [], []

    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            out = model(
                b["r0"], b["v0"], b["r_target"], b["v_target"], b["tof"],
                nrev=b.get("nrev"), ncase=b.get("ncase"), prograde=b.get("prograde"),
                stage1_only=True,
            )
            u0 = out["dv_iterations"][0]
            all_pred.append(u0.cpu().numpy())
            all_target.append(b["dv1"].cpu().numpy())
            all_tof.append(b["tof"].cpu().numpy())

    pred = np.concatenate(all_pred, axis=0)
    target = np.concatenate(all_target, axis=0)
    tof = np.concatenate(all_tof, axis=0).squeeze()

    err = pred - target
    err_norm_ms = np.linalg.norm(err, axis=1) * 1000  # m/s
    target_norm = np.linalg.norm(target, axis=1)

    print(f"\n{label} set ({len(target)} samples):")
    print(f"  u0 residual: mean={err_norm_ms.mean():.1f} m/s, "
          f"median={np.median(err_norm_ms):.1f} m/s, "
          f"p95={np.percentile(err_norm_ms, 95):.1f} m/s, "
          f"max={err_norm_ms.max():.1f} m/s")

    return pred, target, tof, err, err_norm_ms, target_norm


train_pred, train_tgt, train_tof, train_err, train_err_ms, train_tgt_norm = collect_predictions(train_ds, "Train")
test_pred, test_tgt, test_tof, test_err, test_err_ms, test_tgt_norm = collect_predictions(test_ds, "Test")

# ── Figure 1: Training history ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

if "train_loss" in history and len(history["train_loss"]) > 0:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    axes[0].semilogy(epochs, history["train_loss"], label="Train loss", color="steelblue")
    if "val_loss" in history:
        axes[0].semilogy(epochs, history["val_loss"], label="Val loss", color="coral", ls="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (normalized MSE)")
    axes[0].set_title("Stage 1: Loss History")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if "val_u0res_m" in history:
        axes[1].plot(epochs, history["val_u0res_m"], label="Val u0 residual", color="coral")
    if "train_loss" in history:
        # Approximate train residual from loss: res ≈ sqrt(loss * dv_scale²) * 1000
        train_res_approx = np.sqrt(np.array(history["train_loss"]) * dv_scale**2) * 1000
        axes[1].plot(epochs, train_res_approx, label="Train (approx)", color="steelblue", ls="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Δv residual (m/s)")
    axes[1].set_title("Stage 1: Δv Residual vs Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
else:
    axes[0].text(0.5, 0.5, "No training history in checkpoint", ha="center", va="center")
    axes[1].text(0.5, 0.5, "No training history in checkpoint", ha="center", va="center")

fig.tight_layout()
history_path = PLOTS_DIR / f"s1_training_history_{output_tag}.png"
fig.savefig(history_path, dpi=150, bbox_inches="tight")
print(f"Saved: {history_path.name}")

# ── Figure 2: Predicted vs Target scatter (per component) ───────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
comp_names = ["Δv_x", "Δv_y", "Δv_z"]

for i, (ax, name) in enumerate(zip(axes, comp_names)):
    ax.scatter(test_tgt[:, i], test_pred[:, i], s=3, alpha=0.3, color="steelblue", label="Test")
    lims = [min(test_tgt[:, i].min(), test_pred[:, i].min()),
            max(test_tgt[:, i].max(), test_pred[:, i].max())]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="Perfect")
    ax.set_xlabel(f"Target {name} (km/s)")
    ax.set_ylabel(f"Predicted {name} (km/s)")
    ax.set_title(f"{name}: Predicted vs Target")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Per-component stats
    comp_err = (test_pred[:, i] - test_tgt[:, i]) * 1000
    ax.text(0.05, 0.92, f"RMS={np.sqrt(np.mean(comp_err**2)):.1f} m/s",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

fig.suptitle("Stage 1: Lambert Δv Approximation (Test Set)", fontsize=13)
fig.tight_layout()
scatter_path = PLOTS_DIR / f"s1_scatter_components_{output_tag}.png"
fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
print(f"Saved: {scatter_path.name}")

# ── Figure 3: Error distribution ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, data, label, color in [(axes[0], train_err_ms, "Train", "steelblue"),
                                 (axes[1], test_err_ms, "Test", "coral")]:
    ax.hist(data, bins=80, color=color, alpha=0.7, edgecolor="white", lw=0.3)
    ax.axvline(np.mean(data), color="black", ls="--", lw=1.2, label=f"Mean={np.mean(data):.1f} m/s")
    ax.axvline(np.median(data), color="green", ls=":", lw=1.2, label=f"Median={np.median(data):.1f} m/s")
    ax.axvline(np.percentile(data, 95), color="red", ls="-.", lw=1.2,
               label=f"P95={np.percentile(data, 95):.1f} m/s")
    ax.set_xlabel("‖Δv error‖ (m/s)")
    ax.set_ylabel("Count")
    ax.set_title(f"{label} Set: Error Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
dist_path = PLOTS_DIR / f"s1_error_distribution_{output_tag}.png"
fig.savefig(dist_path, dpi=150, bbox_inches="tight")
print(f"Saved: {dist_path.name}")

# ── Figure 4: Error vs Δv magnitude ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, err_ms, tgt_norm, label, color in [
    (axes[0], train_err_ms, train_tgt_norm, "Train", "steelblue"),
    (axes[1], test_err_ms, test_tgt_norm, "Test", "coral"),
]:
    ax.scatter(tgt_norm * 1000, err_ms, s=3, alpha=0.3, color=color)
    ax.set_xlabel("Target ‖Δv₁‖ (m/s)")
    ax.set_ylabel("Prediction error (m/s)")
    ax.set_title(f"{label}: Error vs Transfer Magnitude")
    ax.grid(True, alpha=0.3)

    # Add relative error on right axis
    ax2 = ax.twinx()
    rel_err = err_ms / (tgt_norm * 1000 + 1e-6) * 100
    ax2.scatter(tgt_norm * 1000, rel_err, s=1, alpha=0.1, color="gray")
    ax2.set_ylabel("Relative error (%)", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.set_ylim(0, min(100, np.percentile(rel_err, 99)))

fig.tight_layout()
dv_mag_path = PLOTS_DIR / f"s1_error_vs_dv_mag_{output_tag}.png"
fig.savefig(dv_mag_path, dpi=150, bbox_inches="tight")
print(f"Saved: {dv_mag_path.name}")

# ── Figure 5: Error vs TOF ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, err_ms, tof, label, color in [
    (axes[0], train_err_ms, train_tof, "Train", "steelblue"),
    (axes[1], test_err_ms, test_tof, "Test", "coral"),
]:
    tof_min = tof / 60.0
    ax.scatter(tof_min, err_ms, s=3, alpha=0.3, color=color)
    ax.set_xlabel("Time of Flight (min)")
    ax.set_ylabel("Prediction error (m/s)")
    ax.set_title(f"{label}: Error vs TOF")
    ax.grid(True, alpha=0.3)

fig.tight_layout()
tof_path = PLOTS_DIR / f"s1_error_vs_tof_{output_tag}.png"
fig.savefig(tof_path, dpi=150, bbox_inches="tight")
print(f"Saved: {tof_path.name}")

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STAGE 1 SUMMARY")
print("="*60)
print(f"  Checkpoint: {ckpt_path.name}")
print(f"  State repr: {state_repr}")
print(f"  Train: mean={train_err_ms.mean():.1f} m/s, median={np.median(train_err_ms):.1f} m/s, "
      f"p95={np.percentile(train_err_ms, 95):.1f} m/s")
print(f"  Test:  mean={test_err_ms.mean():.1f} m/s, median={np.median(test_err_ms):.1f} m/s, "
      f"p95={np.percentile(test_err_ms, 95):.1f} m/s")
print(f"  Epoch trained: {ckpt.get('epoch', '?')}")
print(f"  Best val loss: {ckpt.get('best_val', '?')}")

if args.show:
    plt.show()
