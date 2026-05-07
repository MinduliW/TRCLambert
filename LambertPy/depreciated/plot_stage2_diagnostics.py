"""Diagnostic plots for Stage 2 refinement quality.

Generates:
1. Training history (loss, pos_err_K, improvement metric over epochs)
2. Per-iteration position error reduction (does each iteration help?)
3. Position error distribution (train vs test)
4. Position error vs TOF and vs dv magnitude
5. Correction analysis (how much is the refinement head changing dv?)
6. Comparison: Lambert-only vs TRC-refined position error
7. Adaptive K analysis (how many iterations does each sample need?)
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
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
    parser = argparse.ArgumentParser(description="Generate Stage 2 diagnostics plots.")
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
        "--eval_k_max",
        type=int,
        default=10,
        help="Maximum refinement iterations to run during diagnostics.",
    )
    parser.add_argument(
        "--eval_tol_km",
        type=float,
        default=1.0,
        help="Convergence threshold used for adaptive-K analysis.",
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
    return THIS_DIR / "checkpoints" / f"trc_lambert_u0_curriculum_best{suffix}.pt"


args = parse_args()

# ── Load model ──────────────────────────────────────────────────────────────
ckpt_path = resolve_checkpoint(args)
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

net_cfg = NetConfig(**ckpt["net_cfg"])
scales = ckpt.get("scales", {})
correction_scale = scales.get("corr", 0.005)
pos_scale_km = scales.get("pos_m", 300000.0) / 1000.0
cfg_saved = ckpt.get("cfg", {})
state_repr = cfg_saved.get("state_repr", "cartesian")
output_tag = args.output_tag or state_repr

model = LambertTRC(
    net_cfg, max_coast_step_s=45.0, dv_max=5.0, correction_scale=correction_scale, state_repr=state_repr
).to(device)

# ── Load both datasets ──────────────────────────────────────────────────────
train_ds = LambertDataset(THIS_DIR / "data" / "lambert_train.npz")
test_ds = LambertDataset(THIS_DIR / "data" / "lambert_test.npz")

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
if use_tof_log and (tof_log_min is None or tof_log_max is None):
    tof_log_min = torch.log1p(train_ds.tof).min().item()
    tof_log_max = torch.log1p(train_ds.tof).max().item()
model.set_normalization(
    r_min, r_max, v_min, v_max, tof_min, tof_max, dv_scale,
    pos_scale_km=pos_scale_km,
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

# ── Config for adaptive-K eval ──────────────────────────────────────────────
EVAL_K_MAX = args.eval_k_max
EVAL_TOL_KM = args.eval_tol_km


def collect_full_predictions(dataset, label, K_max=EVAL_K_MAX):
    """Run full forward pass at K_max, return per-iteration data."""
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_u0, all_dv_final, all_dv_target = [], [], []
    all_tof = []
    all_pos_errors = None
    all_dv_iters = None

    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            out = model(
                b["r0"], b["v0"], b["r_target"], b["v_target"], b["tof"],
                nrev=b.get("nrev"), ncase=b.get("ncase"), prograde=b.get("prograde"),
                stage1_only=False, K_min=K_max, K_max=K_max,
            )

            all_u0.append(out["dv_iterations"][0].cpu().numpy())
            all_dv_final.append(out["dv_final"].cpu().numpy())
            all_dv_target.append(b["dv1"].cpu().numpy())
            all_tof.append(b["tof"].cpu().numpy())

            batch_pos = [pe.cpu().numpy() for pe in out["pos_errors"]]
            if all_pos_errors is None:
                all_pos_errors = [[] for _ in range(len(batch_pos))]
                all_dv_iters = [[] for _ in range(len(out["dv_iterations"]))]
            for k, pe in enumerate(batch_pos):
                all_pos_errors[k].append(pe)
            for k, dv in enumerate(out["dv_iterations"]):
                all_dv_iters[k].append(dv.cpu().numpy())

    u0 = np.concatenate(all_u0, axis=0)
    dv_final = np.concatenate(all_dv_final, axis=0)
    dv_target = np.concatenate(all_dv_target, axis=0)
    tof = np.concatenate(all_tof, axis=0).squeeze()

    pos_errors = [np.concatenate(pe_list, axis=0) for pe_list in all_pos_errors]
    dv_iters = [np.concatenate(dv_list, axis=0) for dv_list in all_dv_iters]

    N = len(dv_target)
    K = len(pos_errors) - 1

    print(f"\n{label} set ({N} samples, K_max={K_max}, {K} iteration checkpoints):")
    for k, pe in enumerate(pos_errors):
        tag = f"iter {k}" if k < K else "final"
        print(f"  pos_err [{tag}]: mean={pe.mean():.2f} km, "
              f"median={np.median(pe):.2f} km, p95={np.percentile(pe, 95):.2f} km")

    dv_correction = dv_final - u0
    corr_norm_ms = np.linalg.norm(dv_correction, axis=1) * 1000
    print(f"  Total dv correction: mean={corr_norm_ms.mean():.1f} m/s, "
          f"max={corr_norm_ms.max():.1f} m/s")

    # Compute per-sample K_needed at various thresholds
    k_needed = {}
    for tol in [0.5, 1.0, 2.0, 5.0, 10.0]:
        needed = np.full(N, K_max, dtype=int)
        for k_idx in range(len(pos_errors)):
            converged = pos_errors[k_idx] < tol
            needed = np.where((needed == K_max) & converged, k_idx, needed)
        k_needed[tol] = needed
        pct_converged = 100.0 * np.mean(needed < K_max)
        avg_k = needed.mean()
        print(f"  tol={tol:.1f} km: K_avg={avg_k:.1f}, converged={pct_converged:.1f}%")

    return {
        "u0": u0, "dv_final": dv_final, "dv_target": dv_target,
        "tof": tof, "pos_errors": pos_errors, "dv_iters": dv_iters,
        "dv_correction": dv_correction, "corr_norm_ms": corr_norm_ms,
        "k_needed": k_needed,
    }


print("Running full inference on both datasets...")
train_data = collect_full_predictions(train_ds, "Train")
test_data = collect_full_predictions(test_ds, "Test")


# ── Figure 1: Training history ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

if "train_loss" in history and len(history["train_loss"]) > 0:
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    axes[0].semilogy(epochs, history["train_loss"], label="Train", color="steelblue")
    if "val_loss" in history:
        axes[0].semilogy(epochs, history["val_loss"], label="Val", color="coral", ls="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Stage 2: Loss History")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if "val_pos" in history:
        axes[1].plot(epochs, history["val_pos"], label="Val pos_err_K", color="coral")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Position Error (km)")
    axes[1].set_title("Stage 2: Terminal Position Error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if "val_u0res_m" in history:
        ax_u0 = axes[2]
        ax_u0.plot(epochs, history["val_u0res_m"], label="u0 residual (frozen)", color="gray", ls=":")
        if "val_dvres_m" in history:
            ax_u0.plot(epochs, history["val_dvres_m"], label="dv_final residual", color="steelblue")
        ax_u0.set_xlabel("Epoch")
        ax_u0.set_ylabel("Δv residual (m/s)")
        ax_u0.set_title("Stage 2: Δv Residuals")
        ax_u0.legend()
        ax_u0.grid(True, alpha=0.3)
else:
    for ax in axes:
        ax.text(0.5, 0.5, "No training history", ha="center", va="center")

fig.tight_layout()
training_history_path = PLOTS_DIR / f"s2_training_history_{output_tag}.png"
fig.savefig(training_history_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {training_history_path.name}")


# ── Figure 2: Per-iteration position error (box plot) ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, data, label in [(axes[0], train_data, "Train"), (axes[1], test_data, "Test")]:
    pe = data["pos_errors"]
    K = len(pe)
    positions = list(range(K))
    box_data = [pe_k for pe_k in pe]

    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black", linewidth=1.5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, K))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    means = [np.mean(pe_k) for pe_k in pe]
    ax.plot(positions, means, "ro-", markersize=6, label="Mean", zorder=5)

    labels = [f"Iter {k}" for k in range(K - 1)] + ["Final"]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Position Error (km)")
    ax.set_title(f"{label}: Position Error per Iteration (K_max={EVAL_K_MAX})")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
per_iter_path = PLOTS_DIR / f"s2_per_iteration_error_{output_tag}.png"
fig.savefig(per_iter_path, dpi=150, bbox_inches="tight")
print(f"Saved: {per_iter_path.name}")


# ── Figure 3: Position error distribution (Lambert-only vs TRC-refined) ────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, data, label in [(axes[0], train_data, "Train"), (axes[1], test_data, "Test")]:
    pe_init = data["pos_errors"][0]
    pe_final = data["pos_errors"][-1]

    bins = np.linspace(0, max(np.percentile(pe_init, 99), np.percentile(pe_final, 99)), 80)

    ax.hist(pe_init, bins=bins, alpha=0.5, color="coral", label=f"Lambert only (mean={pe_init.mean():.1f} km)", edgecolor="white", lw=0.3)
    ax.hist(pe_final, bins=bins, alpha=0.7, color="steelblue", label=f"TRC refined (mean={pe_final.mean():.1f} km)", edgecolor="white", lw=0.3)

    ax.axvline(pe_init.mean(), color="coral", ls="--", lw=1.2)
    ax.axvline(pe_final.mean(), color="steelblue", ls="--", lw=1.2)

    ax.set_xlabel("Terminal Position Error (km)")
    ax.set_ylabel("Count")
    ax.set_title(f"{label}: Lambert vs TRC Refined")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
lambert_vs_refined_path = PLOTS_DIR / f"s2_lambert_vs_refined_{output_tag}.png"
fig.savefig(lambert_vs_refined_path, dpi=150, bbox_inches="tight")
print(f"Saved: {lambert_vs_refined_path.name}")


# ── Figure 4: Position error vs TOF and vs dv magnitude ────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

for col, data, label in [(0, train_data, "Train"), (1, test_data, "Test")]:
    pe_final = data["pos_errors"][-1]
    pe_init = data["pos_errors"][0]
    tof_min = data["tof"] / 60.0
    dv_mag = np.linalg.norm(data["dv_target"], axis=1) * 1000

    ax = axes[0, col]
    ax.scatter(tof_min, pe_init, s=3, alpha=0.2, color="coral", label="Lambert")
    ax.scatter(tof_min, pe_final, s=3, alpha=0.3, color="steelblue", label="TRC refined")
    ax.set_xlabel("Time of Flight (min)")
    ax.set_ylabel("Position Error (km)")
    ax.set_title(f"{label}: Error vs TOF")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, col]
    ax.scatter(dv_mag, pe_init, s=3, alpha=0.2, color="coral", label="Lambert")
    ax.scatter(dv_mag, pe_final, s=3, alpha=0.3, color="steelblue", label="TRC refined")
    ax.set_xlabel("Target ‖Δv₁‖ (m/s)")
    ax.set_ylabel("Position Error (km)")
    ax.set_title(f"{label}: Error vs Transfer Magnitude")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
tof_dv_path = PLOTS_DIR / f"s2_error_vs_tof_dv_{output_tag}.png"
fig.savefig(tof_dv_path, dpi=150, bbox_inches="tight")
print(f"Saved: {tof_dv_path.name}")


# ── Figure 5: Correction analysis ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax_idx, (data, label, color) in enumerate(
    [(train_data, "Train", "steelblue"), (test_data, "Test", "coral")]
):
    axes[0].hist(data["corr_norm_ms"], bins=60, alpha=0.6, color=color,
                 label=f"{label} (mean={data['corr_norm_ms'].mean():.1f} m/s)",
                 edgecolor="white", lw=0.3)
axes[0].set_xlabel("Total ‖Δv correction‖ (m/s)")
axes[0].set_ylabel("Count")
axes[0].set_title("Refinement Correction Magnitude")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

ax = axes[1]
ax.scatter(test_data["pos_errors"][0], test_data["corr_norm_ms"],
           s=5, alpha=0.3, color="steelblue")
ax.set_xlabel("Lambert Terminal Error (km)")
ax.set_ylabel("Correction ‖Δv‖ (m/s)")
ax.set_title("Test: Correction vs Lambert Error")
ax.grid(True, alpha=0.3)

ax = axes[2]
dv_iters = test_data["dv_iters"]
for k in range(1, len(dv_iters)):
    iter_corr = np.linalg.norm(dv_iters[k] - dv_iters[k - 1], axis=1) * 1000
    ax.boxplot([iter_corr], positions=[k], widths=0.5, patch_artist=True,
               showfliers=False,
               boxprops=dict(facecolor=plt.cm.viridis(k / len(dv_iters)), alpha=0.7),
               medianprops=dict(color="black"))
ax.set_xlabel("Iteration")
ax.set_ylabel("‖Δv step‖ (m/s)")
ax.set_title("Test: Correction per Iteration")
ax.set_xticks(range(1, len(dv_iters)))
ax.set_xticklabels([f"Iter {k}" for k in range(1, len(dv_iters))])
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
correction_path = PLOTS_DIR / f"s2_correction_analysis_{output_tag}.png"
fig.savefig(correction_path, dpi=150, bbox_inches="tight")
print(f"Saved: {correction_path.name}")


# ── Figure 6: Improvement factor scatter ────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(7, 6))

pe_init = test_data["pos_errors"][0]
pe_final = test_data["pos_errors"][-1]
improvement_factor = pe_init / np.clip(pe_final, 1e-3, None)

ax.scatter(pe_init, pe_final, s=5, alpha=0.3, c=improvement_factor,
           cmap="RdYlGn", vmin=1, vmax=10)
ax.plot([0, pe_init.max()], [0, pe_init.max()], "k--", lw=1, alpha=0.5, label="No improvement")
ax.set_xlabel("Lambert Terminal Error (km)")
ax.set_ylabel("Final Position Error (km)")
ax.set_title("Test: Lambert vs Final Position Error")
ax.legend()
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(ax.collections[0], ax=ax, label="Improvement Factor")

fig.tight_layout()
improvement_path = PLOTS_DIR / f"s2_improvement_scatter_{output_tag}.png"
fig.savefig(improvement_path, dpi=150, bbox_inches="tight")
print(f"Saved: {improvement_path.name}")


# ── Figure 7: Adaptive K analysis ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 7a: K_needed histogram at tol=1.0 km
ax = axes[0]
k_needed_1km = test_data["k_needed"][EVAL_TOL_KM]
max_k = EVAL_K_MAX
bins = np.arange(-0.5, max_k + 1.5)
ax.hist(k_needed_1km, bins=bins, color="steelblue", edgecolor="white", lw=0.5, alpha=0.8)
ax.axvline(k_needed_1km.mean(), color="red", ls="--", lw=1.5,
           label=f"Mean K={k_needed_1km.mean():.1f}")
pct_conv = 100.0 * np.mean(k_needed_1km < max_k)
ax.set_xlabel("Iterations needed")
ax.set_ylabel("Count")
ax.set_title(f"Test: K needed for tol={EVAL_TOL_KM} km ({pct_conv:.0f}% converged)")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

# 7b: Mean error vs K for different tolerances — convergence curves
ax = axes[1]
pe_list = test_data["pos_errors"]
k_range = np.arange(len(pe_list))
means = [pe.mean() for pe in pe_list]
medians = [np.median(pe) for pe in pe_list]
p95s = [np.percentile(pe, 95) for pe in pe_list]
ax.semilogy(k_range, means, "o-", color="steelblue", label="Mean", markersize=5)
ax.semilogy(k_range, medians, "s-", color="coral", label="Median", markersize=5)
ax.semilogy(k_range, p95s, "^-", color="gray", label="P95", markersize=5)
for tol in [1.0, 2.0, 5.0]:
    ax.axhline(tol, color="green", ls=":", alpha=0.4)
    ax.text(len(pe_list) - 0.5, tol * 1.1, f"{tol} km", fontsize=7, color="green")
ax.set_xlabel("Iteration")
ax.set_ylabel("Position Error (km)")
ax.set_title("Test: Error vs Iteration Count")
ax.legend()
ax.grid(True, alpha=0.3)

# 7c: K_needed vs Lambert terminal error (do harder problems need more K?)
ax = axes[2]
lambert_err = test_data["pos_errors"][0]
k_needed_1km = test_data["k_needed"][EVAL_TOL_KM]
# Jitter K slightly for visibility
k_jittered = k_needed_1km + np.random.uniform(-0.2, 0.2, size=len(k_needed_1km))
converged = k_needed_1km < max_k
ax.scatter(lambert_err[converged], k_jittered[converged],
           s=5, alpha=0.3, color="steelblue", label="Converged")
ax.scatter(lambert_err[~converged], k_jittered[~converged],
           s=5, alpha=0.3, color="red", label=f"Not converged (K={max_k})")
ax.set_xlabel("Lambert Terminal Error (km)")
ax.set_ylabel(f"K needed (tol={EVAL_TOL_KM} km)")
ax.set_title("Test: Harder transfers need more iterations")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.tight_layout()
adaptive_k_path = PLOTS_DIR / f"s2_adaptive_k_analysis_{output_tag}.png"
fig.savefig(adaptive_k_path, dpi=150, bbox_inches="tight")
print(f"Saved: {adaptive_k_path.name}")


# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STAGE 2 SUMMARY")
print("=" * 65)
print(f"  Checkpoint: {ckpt_path.name}")
print(f"  State repr: {state_repr}")
for label, data in [("Train", train_data), ("Test", test_data)]:
    pe0 = data["pos_errors"][0]
    peK = data["pos_errors"][-1]
    corr = data["corr_norm_ms"]
    print(f"  {label}:")
    print(f"    Lambert pos err:  mean={pe0.mean():.1f} km, median={np.median(pe0):.1f} km, p95={np.percentile(pe0, 95):.1f} km")
    print(f"    Refined pos err:  mean={peK.mean():.1f} km, median={np.median(peK):.1f} km, p95={np.percentile(peK, 95):.1f} km")
    print(f"    Improvement:      {(1 - peK.mean()/pe0.mean())*100:.1f}% mean reduction")
    print(f"    Correction size:  mean={corr.mean():.1f} m/s, max={corr.max():.1f} m/s")

    # Adaptive K stats
    k1 = data["k_needed"][EVAL_TOL_KM]
    pct = 100.0 * np.mean(k1 < EVAL_K_MAX)
    print(f"    K needed (tol={EVAL_TOL_KM}km): avg={k1.mean():.1f}, converged={pct:.0f}%")

print(f"\n  Epoch: {ckpt.get('epoch', '?')}")
print(f"  Best val loss: {ckpt.get('best_val', '?'):.4f}")
print(f"  correction_scale: {correction_scale}")
print(f"  Eval K_max: {EVAL_K_MAX}, tol: {EVAL_TOL_KM} km")

if args.show:
    plt.show()
