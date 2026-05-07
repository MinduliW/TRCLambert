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


from lambert_trc_model import LambertDataset, LambertTRC, CoastSimulator
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

test_ds = LambertDataset(THIS_DIR / "data" / "lambert_test.npz")
r_scale, v_scale, tof_scale, dv_scale, _ = test_ds.get_scales()
model.set_normalization(r_scale, v_scale, tof_scale, dv_scale, pos_scale_km=pos_scale_km)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

plots_dir = THIS_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

K_MAX = 20  # test up to this many iterations


def forward_variable_k(model, r0, v0, r_target, v_target, tof, K_run):
    """Run TRC with a specified number of iterations (ignoring model.net.K)."""
    B = r0.shape[0]
    n = model.net.n_inner

    r0_n, v0_n, rt_n, vt_n, tof_n = model._normalize_input(
        r0, v0, r_target, v_target, tof
    )
    enc_input = torch.cat([r0_n, v0_n, rt_n, vt_n, tof_n], dim=-1)
    z0 = model.state_encoder(enc_input)

    z_H = model.H_init.expand(B, -1) + model.h_proj(z0)
    z_L = model.L_init.expand(B, -1) + model.l_proj(z0)

    dv = model.init_decoder(z0) * model.dv_scale
    dv = model._clip_dv(dv)

    pos_errors = []

    # Record Lambert-only error (propagate u0 before any refinement)
    r_final, _ = model.coast(r0, v0 + dv, tof)
    pos_errors.append(torch.norm(r_final - r_target, dim=-1).detach().cpu().numpy())

    for k in range(K_run):
        r_final, _ = model.coast(r0, v0 + dv, tof)
        pos_err = r_final - r_target
        error = pos_err / model.pos_scale

        z_err = model.error_encoder(error)
        z_ctrl = model.ctrl_embed(dv / model.dv_scale)
        for _ in range(n):
            z_L = model.reason(z_L, z_H, z0, z_err, z_ctrl)
        z_H = model.reason(z_H, z_L)

        delta_dv = model.res_decoder(torch.cat([z_H, dv / model.dv_scale], dim=-1))
        dv = model._clip_dv(dv + delta_dv * model.correction_scale)

        # Record error AFTER this iteration's correction
        r_final, _ = model.coast(r0, v0 + dv, tof)
        pos_errors.append(torch.norm(r_final - r_target, dim=-1).detach().cpu().numpy())

    return pos_errors  # list of K_run+1 arrays, each (B,)


# ── Run inference at K_MAX iterations ───────────────────────────────────────
print(f"Running inference with K={K_MAX} iterations on test set...")
loader = DataLoader(test_ds, batch_size=64, shuffle=False)

all_pos_errors = [[] for _ in range(K_MAX + 1)]

with torch.no_grad():
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        pe_list = forward_variable_k(
            model, b["r0"], b["v0"], b["r_target"], b["v_target"], b["tof"], K_MAX
        )
        for k, pe in enumerate(pe_list):
            all_pos_errors[k].append(pe)

# Concatenate: pos_errors_by_k[k] has shape (N,)
pos_errors_by_k = [np.concatenate(pe_list, axis=0) for pe_list in all_pos_errors]
N = len(pos_errors_by_k[0])

print(f"\nTest set ({N} samples), error at each iteration:")
for k, pe in enumerate(pos_errors_by_k):
    tag = "Lambert" if k == 0 else f"K={k}"
    print(f"  {tag:>8s}: mean={pe.mean():8.2f} km, median={np.median(pe):7.2f} km, "
          f"p95={np.percentile(pe, 95):7.2f} km, max={pe.max():8.2f} km")


# ── Figure 1: Mean/Median/P95 error vs iteration ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ks = np.arange(K_MAX + 1)
means = [pe.mean() for pe in pos_errors_by_k]
medians = [np.median(pe) for pe in pos_errors_by_k]
p95s = [np.percentile(pe, 95) for pe in pos_errors_by_k]
p5s = [np.percentile(pe, 5) for pe in pos_errors_by_k]

# Left: linear scale
ax = axes[0]
ax.plot(ks, means, "o-", color="steelblue", label="Mean", markersize=6)
ax.plot(ks, medians, "s-", color="coral", label="Median", markersize=5)
ax.fill_between(ks, p5s, p95s, alpha=0.15, color="steelblue", label="5th–95th %ile")
ax.axvline(3, color="gray", ls=":", lw=1, alpha=0.7, label="Training K=3")
ax.set_xlabel("Refinement Iteration")
ax.set_ylabel("Terminal Position Error (km)")
ax.set_title("Error vs Iteration Count (linear)")
ax.set_xticks(ks)
ax.set_xticklabels(["Lam"] + [str(k) for k in range(1, K_MAX + 1)])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: log scale
ax = axes[1]
ax.semilogy(ks, means, "o-", color="steelblue", label="Mean", markersize=6)
ax.semilogy(ks, medians, "s-", color="coral", label="Median", markersize=5)
ax.fill_between(ks, p5s, p95s, alpha=0.15, color="steelblue", label="5th–95th %ile")
ax.axvline(3, color="gray", ls=":", lw=1, alpha=0.7, label="Training K=3")
ax.set_xlabel("Refinement Iteration")
ax.set_ylabel("Terminal Position Error (km)")
ax.set_title("Error vs Iteration Count (log)")
ax.set_xticks(ks)
ax.set_xticklabels(["Lam"] + [str(k) for k in range(1, K_MAX + 1)])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(plots_dir / "s2_error_vs_K.png", dpi=150, bbox_inches="tight")
print("\nSaved: s2_error_vs_K.png")


# ── Figure 2: Box plot at each K ───────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(12, 5.5))

bp = ax.boxplot(pos_errors_by_k, positions=ks, widths=0.6, patch_artist=True,
                showfliers=False, medianprops=dict(color="black", linewidth=1.5))

colors = plt.cm.viridis(np.linspace(0.1, 0.9, K_MAX + 1))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.axvline(3, color="red", ls=":", lw=1.5, alpha=0.7, label="Training K=3")
ax.set_xlabel("Refinement Iteration")
ax.set_ylabel("Terminal Position Error (km)")
ax.set_title("Error Distribution at Each Iteration")
ax.set_xticks(ks)
ax.set_xticklabels(["Lambert"] + [f"K={k}" for k in range(1, K_MAX + 1)], fontsize=8)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig(plots_dir / "s2_boxplot_vs_K.png", dpi=150, bbox_inches="tight")
print("Saved: s2_boxplot_vs_K.png")


# ── Figure 3: Individual sample convergence curves ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Pick a spread of samples: some easy, some hard
pe_at_lambert = pos_errors_by_k[0]
sorted_idx = np.argsort(pe_at_lambert)
n_curves = 20

# Easy samples (low Lambert error)
easy_idx = sorted_idx[:n_curves]
# Hard samples (high Lambert error)
hard_idx = sorted_idx[-n_curves:]

for ax, idx, title in [(axes[0], easy_idx, "Easy Transfers (low Lambert error)"),
                         (axes[1], hard_idx, "Hard Transfers (high Lambert error)")]:
    for i in idx:
        curve = [pos_errors_by_k[k][i] for k in range(K_MAX + 1)]
        ax.semilogy(ks, curve, alpha=0.4, lw=1)

    # Mean of selected
    mean_curve = [np.mean([pos_errors_by_k[k][i] for i in idx]) for k in range(K_MAX + 1)]
    ax.semilogy(ks, mean_curve, "k-", lw=2.5, label="Mean")
    ax.axvline(3, color="red", ls=":", lw=1.5, alpha=0.7, label="Training K=3")
    ax.set_xlabel("Refinement Iteration")
    ax.set_ylabel("Terminal Position Error (km)")
    ax.set_title(title)
    ax.set_xticks(ks)
    ax.set_xticklabels(["Lam"] + [str(k) for k in range(1, K_MAX + 1)])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(plots_dir / "s2_convergence_curves.png", dpi=150, bbox_inches="tight")
print("Saved: s2_convergence_curves.png")


# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("VARIABLE-K EVALUATION SUMMARY")
print("=" * 65)
print(f"  Trained at K=3, evaluated up to K={K_MAX}")
print(f"  Lambert:  mean={pos_errors_by_k[0].mean():.1f} km")
for k in [1, 2, 3, 5, 7, 10]:
    if k <= K_MAX:
        pe = pos_errors_by_k[k]
        print(f"  K={k:2d}:     mean={pe.mean():.2f} km, median={np.median(pe):.2f} km")

# Check: does error increase after K=3?
pe3 = pos_errors_by_k[3].mean()
pe_max = pos_errors_by_k[K_MAX].mean()
if pe_max > pe3 * 1.1:
    print(f"\n  WARNING: Error increased from K=3 ({pe3:.2f} km) to K={K_MAX} ({pe_max:.2f} km)")
    print(f"  The refinement operator may diverge beyond training K. Consider retraining with higher K.")
elif pe_max < pe3 * 0.9:
    print(f"\n  Extra iterations help: K=3 ({pe3:.2f} km) → K={K_MAX} ({pe_max:.2f} km)")
    print(f"  The refinement operator generalises beyond training K!")
else:
    print(f"\n  Converged: K=3 ({pe3:.2f} km) ≈ K={K_MAX} ({pe_max:.2f} km)")
    print(f"  Extra iterations neither help nor hurt. K=3 is sufficient.")

plt.show()
