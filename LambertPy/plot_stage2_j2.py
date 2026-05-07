"""Stage 2 diagnostic plots for the J2-perturbed TRC model.

Generates:
1. Per-iteration position error reduction (box plot across K iterations)
2. Lambert-only vs TRC-corrected position error (scatter + histogram)

Usage:
    python plot_stage2_j2.py
    python plot_stage2_j2.py --checkpoint checkpoints/trc_j2_best.pt --val val_struct_small.mat
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from torch.utils.data import DataLoader

# ── Publication style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'serif',
    'font.size':          11,
    'axes.titlesize':     12,
    'axes.labelsize':     11,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    10,
    'axes.linewidth':     0.8,
    'grid.linewidth':     0.5,
    'grid.alpha':         0.35,
    'lines.linewidth':    1.5,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent))

from lambert_trc_j2 import (  # noqa: F401
    LambertTRCJ2, MatLambertDataset, TrainConfig,
    J2Propagator, BODY_PARAMS, filter_propagator_consistency,
)
from trc import NetConfig  # noqa: F401

PLOTS_DIR = THIS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


_DATASET_VAL = {
    'single_rev': 'datasets/val_struct.mat',
    'multi_rev':  'datasets/val_leo_fwd.mat',
    'jovian':     'datasets/val_struct_jupiter.mat',
    'jovian_0rev': 'datasets/val_struct_jupiter_0rev.npz',
}


def parse_args():
    p = argparse.ArgumentParser()
    # Experiment selectors (auto-fill checkpoint, val, tag)
    p.add_argument("--variant", type=str, default=None,
                   choices=['pos_only', 'vel_supervised', 'learned_lambert'])
    p.add_argument("--dataset", type=str, default=None,
                   choices=['single_rev', 'multi_rev', 'jovian', 'jovian_0rev'])
    # Manual overrides
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--val", type=str, default=None)
    p.add_argument("--val_key", type=str, default="val_info")
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    # Auto-fill from variant/dataset if not manually specified
    parts = [p for p in [args.variant, args.dataset] if p]
    run_name = 'trc_' + '_'.join(parts) if parts else 'trc_j2'
    if args.checkpoint is None:
        args.checkpoint = Path(f"checkpoints/{run_name}_best.pt")
    if args.val is None:
        args.val = _DATASET_VAL.get(args.dataset, 'val_struct.mat')
    if args.tag is None:
        args.tag = run_name

    return args


@torch.no_grad()
def collect(model, loader, K, device):
    """Run inference and collect per-iteration position errors."""
    all_pos_errors = None   # list of K arrays (one per iteration)
    all_lambert_err = []    # position error from v1_lambert with NO correction

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch['r1'], batch['r2'], batch['tof'], batch['prograde'],
                    nrev=batch.get('nrev'), ncase=batch.get('ncase'),
                    K=K, init_v1=batch['v1_lambert'])

        # per-iteration position errors (km)
        pos_errs = [pe.cpu().numpy() for pe in out['pos_errors']]  # list of (B,)
        if all_pos_errors is None:
            all_pos_errors = [[] for _ in range(len(pos_errs))]
        for k, pe in enumerate(pos_errs):
            all_pos_errors[k].append(pe)

        # Lambert baseline: pos_errors[0] is the error before any correction
        all_lambert_err.append(out['pos_errors'][0].cpu().numpy())

    pos_errors = [np.concatenate(pe) for pe in all_pos_errors]  # list of K arrays
    lambert_err = np.concatenate(all_lambert_err)                # (N,)

    print(f"\n  Samples: {len(lambert_err)}")
    print(f"  Lambert baseline   : mean={lambert_err.mean():.1f} km  "
          f"median={np.median(lambert_err):.1f} km")
    for k, pe in enumerate(pos_errors):
        print(f"  After iteration {k+1}  : mean={pe.mean():.1f} km  "
              f"median={np.median(pe):.1f} km  "
              f"<10km={100*(pe<10).mean():.1f}%  "
              f"<1km={100*(pe<1).mean():.1f}%")

    return pos_errors, lambert_err


def plot_results(pos_errors, lambert_err, tag):
    """Three-panel figure: box plot | scatter | histogram."""
    K       = len(pos_errors)
    trc_err = pos_errors[-1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # ── Panel 1: Box plot of per-iteration error ──────────────────────────────
    ax = axes[0]
    data   = [lambert_err] + pos_errors
    labels = ["Lambert"] + [f"$k={k+1}$" for k in range(K)]
    base_color = "#C0392B"
    trc_colors = plt.cm.Blues(np.linspace(0.45, 0.85, K))
    colors = [base_color] + list(trc_colors)

    bp = ax.boxplot(data, patch_artist=True, notch=False, widths=0.5,
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(linewidth=0.8, linestyle='--'),
                    capprops=dict(linewidth=0.8),
                    flierprops=dict(marker='.', markersize=2, color='gray', alpha=0.4),
                    whis=[5, 95])
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_linewidth(0.8)

    for i, d in enumerate(data):
        med = np.median(d)
        ax.text(i + 1, med * 1.15, f"{med:.1f}", ha='center', va='bottom',
                fontsize=8)

    ax.set_xticklabels(labels)
    ax.set_ylabel("Position error (km)")
    ax.set_title("Iterative Refinement")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.grid(axis='y')
    ax.tick_params(axis='x', length=0)

    # ── Panel 2: Scatter Lambert vs TRC ──────────────────────────────────────
    ax = axes[1]
    improvement = lambert_err / np.maximum(trc_err, 1e-3)
    sc = ax.scatter(lambert_err, trc_err,
                    c=improvement, cmap='plasma', s=6, alpha=0.55,
                    norm=matplotlib.colors.LogNorm(vmin=0.1, vmax=100),
                    rasterized=True)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Improvement factor", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    lim = max(lambert_err.max(), trc_err.max()) * 1.05
    ax.set_xlabel("Lambert position error (km)")
    ax.set_ylabel("TRC position error (km)")
    ax.set_title(f"Lambert vs. TRC ($K={K}$)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid()

    # ── Panel 3: Histogram ────────────────────────────────────────────────────
    ax = axes[2]
    all_vals = np.concatenate([lambert_err, trc_err])
    bins = np.linspace(0, np.percentile(all_vals, 99), 50)
    ax.hist(lambert_err, bins=bins, alpha=0.70, color='#C0392B',
            edgecolor='white', linewidth=0.3,
            label=f"Lambert  (median $=$ {np.median(lambert_err):.1f} km)")
    ax.hist(trc_err, bins=bins, alpha=0.70, color='#2980B9',
            edgecolor='white', linewidth=0.3,
            label=f"TRC $K={K}$  (median $=$ {np.median(trc_err):.1f} km)")
    ax.set_xlabel("Position error (km)")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution")
    ax.legend(frameon=True, framealpha=0.9)
    ax.grid()

    out = PLOTS_DIR / f"s2_results_{tag}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    args = parse_args()
    device = torch.device('cpu')

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg  = ckpt['cfg']
    norm = ckpt['norm']
    from lambert_trc_j2 import BODY_PARAMS
    mu, re, _j2 = BODY_PARAMS.get(getattr(cfg, 'body', 'earth'), BODY_PARAMS['earth'])

    net = NetConfig(d_z=cfg.d_z, d_h=cfg.d_h,
                    n_heads=cfg.n_heads, n_blocks=cfg.n_blocks,
                    K=args.K, n_inner=cfg.n_inner)
    head1_mode = getattr(cfg, 'head1_mode', None) or (
        'oracle' if getattr(cfg, 'oracle_init', False) else 'direct'
    )
    model = LambertTRCJ2(net, max_step_s=cfg.max_step_s,
                         v_max=cfg.v_max, pos_scale_km=cfg.pos_scale_km,
                         mu=mu, re=re, j2=_j2,
                         head1_mode=head1_mode).to(device)
    ckpt_input_dim = ckpt['model']['state_encoder.0.weight'].shape[1]
    if ckpt_input_dim != model.input_dim:
        print(f"  Legacy checkpoint: input_dim={ckpt_input_dim} (current={model.input_dim}), rebuilding encoder")
        model.input_dim = ckpt_input_dim
        model.state_encoder[0] = torch.nn.Linear(ckpt_input_dim, cfg.d_h).to(device)
    model.set_normalisation(**{k: v for k, v in norm.items() if k != 'dv_scale'},
                            pos_scale_km=cfg.pos_scale_km)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded: {args.checkpoint}  (val pos_err={ckpt.get('pos_err_km', '?'):.2f} km)")

    # ── Load val dataset ──────────────────────────────────────────────────────
    ds = MatLambertDataset(args.val, args.val_key)

    # Apply the same consistency filter training used. Without this, samples
    # where Python's propagator disagrees with the MATLAB-generated r2 label
    # (chaotic-divergent high-nrev orbits) inflate the error metrics — those
    # samples have unlearnable labels regardless of the model.
    if getattr(cfg, 'body', 'earth') == 'jupiter':
        _filter_step = cfg.train_max_step_s if getattr(cfg, 'train_max_step_s', 0.0) > 0 else cfg.max_step_s
        _mu, _re, _j2c = BODY_PARAMS['jupiter']
        _prop = J2Propagator(max_step_s=_filter_step, mu=_mu, re=_re, j2=_j2c)
        print(f"\n=== Filtering inconsistent Jupiter samples (step={_filter_step:.0f}s) ===")
        ds = filter_propagator_consistency(ds, _prop, threshold_km=100.0)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Collect results ───────────────────────────────────────────────────────
    pos_errors, lambert_err = collect(model, loader, args.K, device)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_results(pos_errors, lambert_err, args.tag)


if __name__ == '__main__':
    main()
