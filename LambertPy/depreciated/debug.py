import sys
import torch
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
for _p in (THIS_DIR, PARENT_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)


from lambert_trc_model import LambertDataset, LambertTRC
from trc import NetConfig
from torch.utils.data import DataLoader

device = torch.device("cpu")

# Load dataf
test_ds = LambertDataset(THIS_DIR / "data" / "lambert_test.npz")
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

# Load trained Stage 1 model
ckpt_path = THIS_DIR / "checkpoints" / "trc_lambert_u0_stage1_best.pt"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

net_cfg = NetConfig(**ckpt["net_cfg"])
model = LambertTRC(net_cfg, n_coast_steps=200, dv_max=3.0, correction_scale=0.005).to(device)

r_scale, v_scale, tof_scale, dv_scale, _ = test_ds.get_scales()
model.set_normalization(r_scale, v_scale, tof_scale, dv_scale, pos_scale_km=1.0)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Run one batch
b = next(iter(test_loader))
b = {k: v.to(device) for k, v in b.items()}

with torch.no_grad():
    out = model(b["r0"], b["v0"], b["r_target"], b["v_target"], b["tof"], stage1_only=True)

u0 = out["dv_iterations"][0]

# Also check the raw network output BEFORE scaling
r0_n = b["r0"] / model.pos_scale
v0_n = b["v0"] / model.v_scale
rt_n = b["r_target"] / model.pos_scale
vt_n = b["v_target"] / model.v_scale
tof_n = b["tof"] / model.tof_scale
enc_input = torch.cat([r0_n, v0_n, rt_n, vt_n, tof_n], dim=-1)
z0 = model.state_encoder(enc_input)
raw_output = model.init_decoder(z0)
raw_output = raw_output.detach()

print("=== Raw init_decoder output (BEFORE * dv_scale) ===")
print(f"  mean:  {raw_output.mean(dim=0).numpy()}")
print(f"  std:   {raw_output.std(dim=0).numpy()}")
print(f"  range: [{raw_output.min().item():.4f}, {raw_output.max().item():.4f}]")

print(f"\n=== Scaled u0 output (AFTER * dv_scale={model.dv_scale.item():.4f}) ===")
print(f"  mean:  {u0.mean(dim=0).numpy()}")
print(f"  std:   {u0.std(dim=0).numpy()}")
print(f"  norms: min={torch.norm(u0,dim=-1).min():.4f} max={torch.norm(u0,dim=-1).max():.4f} mean={torch.norm(u0,dim=-1).mean():.4f}")

print(f"\n=== Target dv1 ===")
print(f"  mean:  {b['dv1'].mean(dim=0).numpy()}")
print(f"  std:   {b['dv1'].std(dim=0).numpy()}")
print(f"  norms: min={torch.norm(b['dv1'],dim=-1).min():.4f} max={torch.norm(b['dv1'],dim=-1).max():.4f} mean={torch.norm(b['dv1'],dim=-1).mean():.4f}")

print(f"\n=== Per-component error ===")
err = u0 - b["dv1"]
print(f"  mean error:  {err.mean(dim=0).numpy()} km/s")
print(f"  std error:   {err.std(dim=0).numpy()} km/s")

# Add this to your debug.py, after the existing code:

# Check: what does the normalized encoder input look like?
print(f"\n=== Encoder input normalization ===")
print(f"  pos_scale (used for r normalization): {model.pos_scale.item()}")
print(f"  v_scale: {model.v_scale.item()}")
print(f"  tof_scale: {model.tof_scale.item()}")

enc_input = torch.cat([r0_n, v0_n, rt_n, vt_n, tof_n], dim=-1)
print(f"\n  enc_input shape: {enc_input.shape}")
for i, name in enumerate(['r0_x','r0_y','r0_z','v0_x','v0_y','v0_z',
                           'rt_x','rt_y','rt_z','vt_x','vt_y','vt_z','tof']):
    col = enc_input[:, i]
    print(f"  {name:5s}: mean={col.mean():+8.4f}  std={col.std():.4f}  range=[{col.min():.3f}, {col.max():.3f}]")
    
    
"""Diagnostic plots for Stage 2 refinement quality.

Generates:
1. Training history (loss, pos_err_K, improvement metric over epochs)
2. Per-iteration position error reduction (does each iteration help?)
3. Position error distribution (train vs test)
4. Position error vs TOF and vs dv magnitude
5. Correction analysis (how much is the refinement head changing dv?)
6. Comparison: Lambert-only vs TRC-refined position error
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from lambert_trc_model import LambertDataset, LambertTRC
from trc import NetConfig

device = torch.device("cpu")

# ── Load model ──────────────────────────────────────────────────────────────
ckpt_path = THIS_DIR / "checkpoints" / "trc_lambert_u0_curriculum_best.pt"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

net_cfg = NetConfig(**ckpt["net_cfg"])
scales = ckpt.get("scales", {})
correction_scale = scales.get("corr", 0.005)
pos_scale_km = scales.get("pos_m", 300000.0) / 1000.0

model = LambertTRC(
    net_cfg, n_coast_steps=200, dv_max=3.0, correction_scale=correction_scale
).to(device)

# ── Load both datasets ──────────────────────────────────────────────────────
train_ds = LambertDataset(THIS_DIR / "data" / "lambert_train.npz")
test_ds = LambertDataset(THIS_DIR / "data" / "lambert_test.npz")

r_scale, v_scale, tof_scale, dv_scale, _ = train_ds.get_scales()
model.set_normalization(r_scale, v_scale, tof_scale, dv_scale, pos_scale_km=pos_scale_km)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

history = ckpt.get("history", {})
plots_dir = THIS_DIR / "plots"
plots_dir.mkdir(exist_ok=True)


def collect_full_predictions(dataset, label):
    """Run full forward pass, return per-iteration data."""
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_u0, all_dv_final, all_dv_target = [], [], []
    all_tof = []
    # Per-iteration position errors: list of K+1 lists
    all_pos_errors = None
    all_dv_iters = None

    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            out = model(b["r0"], b["v0"], b["r_target"], b["v_target"], b["tof"],
                        stage1_only=False)

            all_u0.append(out["dv_iterations"][0].cpu().numpy())
            all_dv_final.append(out["dv_final"].cpu().numpy())
            all_dv_target.append(b["dv1"].cpu().numpy())
            all_tof.append(b["tof"].cpu().numpy())

            # Collect per-iteration pos errors
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
    K = len(pos_errors) - 1  # K iterations + 1 final

    print(f"\n{label} set ({N} samples, K={K} iterations):")
    for k, pe in enumerate(pos_errors):
        tag = f"iter {k}" if k < K else "final"
        print(f"  pos_err [{tag}]: mean={pe.mean():.2f} km, "
              f"median={np.median(pe):.2f} km, p95={np.percentile(pe, 95):.2f} km")

    dv_correction = dv_final - u0
    corr_norm_ms = np.linalg.norm(dv_correction, axis=1) * 1000
    print(f"  Total dv correction: mean={corr_norm_ms.mean():.1f} m/s, "
          f"max={corr_norm_ms.max():.1f} m/s")

    return {
        "u0": u0, "dv_final": dv_final, "dv_target": dv_target,
        "tof": tof, "pos_errors": pos_errors, "dv_iters": dv_iters,
        "dv_correction": dv_correction, "corr_norm_ms": corr_norm_ms,
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
fig.savefig(plots_dir / "s2_training_history.png", dpi=150, bbox_inches="tight")
print("\nSaved: s2_training_history.png")


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

    # Overlay means
    means = [np.mean(pe_k) for pe_k in pe]
    ax.plot(positions, means, "ro-", markersize=6, label="Mean", zorder=5)

    labels = [f"Iter {k}" for k in range(K - 1)] + ["Final"]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Position Error (km)")
    ax.set_title(f"{label}: Position Error per Iteration")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig(plots_dir / "s2_per_iteration_error.png", dpi=150, bbox_inches="tight")
print("Saved: s2_per_iteration_error.png")


# ── Figure 3: Position error distribution (Lambert-only vs TRC-refined) ────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, data, label in [(axes[0], train_data, "Train"), (axes[1], test_data, "Test")]:
    pe_init = data["pos_errors"][0]   # after u0 (Lambert), before refinement
    pe_final = data["pos_errors"][-1]  # after K refinement iterations

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
fig.savefig(plots_dir / "s2_lambert_vs_refined.png", dpi=150, bbox_inches="tight")
print("Saved: s2_lambert_vs_refined.png")


# ── Figure 4: Position error vs TOF and vs dv magnitude ────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

for col, data, label in [(0, train_data, "Train"), (1, test_data, "Test")]:
    pe_final = data["pos_errors"][-1]
    pe_init = data["pos_errors"][0]
    tof_min = data["tof"] / 60.0
    dv_mag = np.linalg.norm(data["dv_target"], axis=1) * 1000  # m/s

    # Error vs TOF
    ax = axes[0, col]
    ax.scatter(tof_min, pe_init, s=3, alpha=0.2, color="coral", label="Lambert")
    ax.scatter(tof_min, pe_final, s=3, alpha=0.3, color="steelblue", label="TRC refined")
    ax.set_xlabel("Time of Flight (min)")
    ax.set_ylabel("Position Error (km)")
    ax.set_title(f"{label}: Error vs TOF")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Error vs dv magnitude
    ax = axes[1, col]
    ax.scatter(dv_mag, pe_init, s=3, alpha=0.2, color="coral", label="Lambert")
    ax.scatter(dv_mag, pe_final, s=3, alpha=0.3, color="steelblue", label="TRC refined")
    ax.set_xlabel("Target ‖Δv₁‖ (m/s)")
    ax.set_ylabel("Position Error (km)")
    ax.set_title(f"{label}: Error vs Transfer Magnitude")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(plots_dir / "s2_error_vs_tof_dv.png", dpi=150, bbox_inches="tight")
print("Saved: s2_error_vs_tof_dv.png")


# ── Figure 5: Correction analysis ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 5a: Correction magnitude distribution
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

# 5b: Correction magnitude vs initial position error
ax = axes[1]
ax.scatter(test_data["pos_errors"][0], test_data["corr_norm_ms"],
           s=5, alpha=0.3, color="steelblue")
ax.set_xlabel("Initial Position Error (km)")
ax.set_ylabel("Correction ‖Δv‖ (m/s)")
ax.set_title("Test: Correction vs Initial Error")
ax.grid(True, alpha=0.3)

# 5c: Per-iteration correction magnitude
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
fig.savefig(plots_dir / "s2_correction_analysis.png", dpi=150, bbox_inches="tight")
print("Saved: s2_correction_analysis.png")


# ── Figure 6: Improvement factor scatter ────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(7, 6))

pe_init = test_data["pos_errors"][0]
pe_final = test_data["pos_errors"][-1]
improvement_factor = pe_init / np.clip(pe_final, 1e-3, None)

ax.scatter(pe_init, pe_final, s=5, alpha=0.3, c=improvement_factor,
           cmap="RdYlGn", vmin=1, vmax=10)
ax.plot([0, pe_init.max()], [0, pe_init.max()], "k--", lw=1, alpha=0.5, label="No improvement")
ax.set_xlabel("Initial Position Error (km)")
ax.set_ylabel("Final Position Error (km)")
ax.set_title("Test: Initial vs Final Position Error")
ax.legend()
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(ax.collections[0], ax=ax, label="Improvement Factor")

fig.tight_layout()
fig.savefig(plots_dir / "s2_improvement_scatter.png", dpi=150, bbox_inches="tight")
print("Saved: s2_improvement_scatter.png")


# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STAGE 2 SUMMARY")
print("=" * 65)
for label, data in [("Train", train_data), ("Test", test_data)]:
    pe0 = data["pos_errors"][0]
    peK = data["pos_errors"][-1]
    corr = data["corr_norm_ms"]
    print(f"  {label}:")
    print(f"    Lambert pos err:  mean={pe0.mean():.1f} km, median={np.median(pe0):.1f} km, p95={np.percentile(pe0, 95):.1f} km")
    print(f"    Refined pos err:  mean={peK.mean():.1f} km, median={np.median(peK):.1f} km, p95={np.percentile(peK, 95):.1f} km")
    print(f"    Improvement:      {(1 - peK.mean()/pe0.mean())*100:.1f}% mean reduction")
    print(f"    Correction size:  mean={corr.mean():.1f} m/s, max={corr.max():.1f} m/s")

print(f"\n  Epoch: {ckpt.get('epoch', '?')}")
print(f"  Best val loss: {ckpt.get('best_val', '?'):.2f}")
print(f"  correction_scale: {correction_scale}")

plt.show()