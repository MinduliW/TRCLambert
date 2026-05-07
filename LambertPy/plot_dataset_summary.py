"""Dataset summary plots for J2 Lambert datasets.

Generates a 6-panel figure matching the single-rev summary style:
  TOF | Departure Altitude | Arrival Altitude
  J2 Correction Magnitude | Revolution Count | Orbit Direction

Usage:
    python plot_dataset_summary.py --dataset single_rev
    python plot_dataset_summary.py --dataset multi_rev
    python plot_dataset_summary.py --train datasets/train_struct_10h.npz --val datasets/val_struct_10h.npz --tag multi_rev
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PLOTS_DIR = THIS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

BODY_RADII = {'earth': 6378.137, 'jupiter': 71492.0}   # km

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
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

_DATASET_FILES = {
    'single_rev': ('datasets/train_struct.npz',       'datasets/val_struct.npz'),
    'multi_rev':  ('datasets/train_leo_fwd.npz',       'datasets/val_leo_fwd.npz'),
    'jovian':     ('datasets/train_struct_jupiter.npz', 'datasets/val_struct_jupiter.npz'),
    'jovian_0rev': ('datasets/train_struct_jupiter_0rev.npz', 'datasets/val_struct_jupiter_0rev.npz'),
}


def load_npz(path):
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def plot_summary(train_path, val_path, tag, body='earth'):
    train = load_npz(train_path)
    val   = load_npz(val_path)
    R_body = BODY_RADII[body]
    alt_label = 'Radius (RJ)' if body == 'jupiter' else 'Altitude (km)'
    dv_max_display = 100.0 if body == 'jupiter' else 80.0

    Nt = len(train['r1'])
    Nv = len(val['r1'])
    title = f"{tag.replace('_', '-').title()} Dataset Summary  (train N={Nt:,} | val N={Nv:,})"

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    TRAIN_COLOR = '#5B9BD5'
    VAL_COLOR   = '#F4A04A'
    ALPHA = 0.75

    def hist_both(ax, t_data, v_data, xlabel, title, bins=40, pct=99):
        all_data = np.concatenate([t_data, v_data])
        lo = np.percentile(all_data, 0)
        hi = np.percentile(all_data, pct)
        bins_ = np.linspace(lo, hi, bins + 1)
        ax.hist(t_data[t_data <= hi], bins=bins_, color=TRAIN_COLOR, alpha=ALPHA,
                label=f'train (N={len(t_data):,})', edgecolor='white', linewidth=0.3)
        ax.hist(v_data[v_data <= hi], bins=bins_, color=VAL_COLOR, alpha=ALPHA,
                label=f'val (N={len(v_data):,})', edgecolor='white', linewidth=0.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Samples')
        ax.set_title(title)
        ax.legend(frameon=True, framealpha=0.9)
        ax.grid(axis='y')

    # ── Panel 1: Time of Flight ───────────────────────────────────────────────
    t_tof = train['tof'] / 60.0   # s → min
    v_tof = val['tof']   / 60.0
    hist_both(axes[0, 0], t_tof, v_tof, 'TOF (min)', 'Time of Flight')

    # ── Panel 2/3: Departure / Arrival radial position ────────────────────────
    # For Jupiter: |r| in units of RJ. For Earth: altitude in km.
    if body == 'jupiter':
        t_alt1 = np.linalg.norm(train['r1'], axis=1) / R_body
        v_alt1 = np.linalg.norm(val['r1'],   axis=1) / R_body
        t_alt2 = np.linalg.norm(train['r2'], axis=1) / R_body
        v_alt2 = np.linalg.norm(val['r2'],   axis=1) / R_body
        title1, title2 = 'Departure |r1| (RJ)', 'Arrival |r2| (RJ)'
    else:
        t_alt1 = np.linalg.norm(train['r1'], axis=1) - R_body
        v_alt1 = np.linalg.norm(val['r1'],   axis=1) - R_body
        t_alt2 = np.linalg.norm(train['r2'], axis=1) - R_body
        v_alt2 = np.linalg.norm(val['r2'],   axis=1) - R_body
        title1, title2 = 'Departure Altitude', 'Arrival Altitude'
    hist_both(axes[0, 1], t_alt1, v_alt1, alt_label, title1)
    hist_both(axes[0, 2], t_alt2, v_alt2, alt_label, title2)

    # ── Panel 4: J2 Correction Magnitude ─────────────────────────────────────
    t_dv = np.linalg.norm(train['v1_j2'] - train['v1'], axis=1) * 1000  # km/s → m/s
    v_dv = np.linalg.norm(val['v1_j2']   - val['v1'],   axis=1) * 1000
    dv_max = dv_max_display
    bins_dv = np.linspace(0, dv_max, 41)
    ax = axes[1, 0]
    ax.hist(t_dv[t_dv <= dv_max], bins=bins_dv, color=TRAIN_COLOR, alpha=ALPHA,
            label=f'train (N={len(t_dv):,})', edgecolor='white', linewidth=0.3)
    ax.hist(v_dv[v_dv <= dv_max], bins=bins_dv, color=VAL_COLOR, alpha=ALPHA,
            label=f'val (N={len(v_dv):,})', edgecolor='white', linewidth=0.3)
    ax.set_xlim(0, dv_max)
    ax.set_xlabel(r'$|\Delta v_{J2}|$ (m/s)')
    ax.set_ylabel('Samples')
    pct_under = 100 * (t_dv <= dv_max).mean()
    ax.set_title(f'J2 Correction Magnitude ({pct_under:.1f}% ≤ {dv_max:.0f} m/s)')
    ax.legend(frameon=True, framealpha=0.9)
    ax.grid(axis='y')

    # ── Panel 5: Revolution Count ─────────────────────────────────────────────
    ax = axes[1, 1]
    t_nrev = train['nrev'].flatten().astype(int)
    v_nrev = val['nrev'].flatten().astype(int)
    nrev_vals = np.arange(0, max(t_nrev.max(), v_nrev.max()) + 2)
    width = 0.4
    t_counts = np.array([(t_nrev == n).sum() for n in nrev_vals])
    v_counts = np.array([(v_nrev == n).sum() for n in nrev_vals])
    ax.bar(nrev_vals - width/2, t_counts, width=width, color=TRAIN_COLOR, alpha=ALPHA,
           label=f'train (N={len(t_nrev):,})', edgecolor='white', linewidth=0.3)
    ax.bar(nrev_vals + width/2, v_counts, width=width, color=VAL_COLOR, alpha=ALPHA,
           label=f'val (N={len(v_nrev):,})', edgecolor='white', linewidth=0.3)
    ax.set_xlabel('nrev')
    ax.set_ylabel('Samples')
    ax.set_title('Revolution Count')
    ax.set_xticks(nrev_vals)
    ax.legend(frameon=True, framealpha=0.9)
    ax.grid(axis='y')

    # ── Panel 6: Orbit Direction ──────────────────────────────────────────────
    ax = axes[1, 2]
    t_pro = train['prograde'].flatten()
    v_pro = val['prograde'].flatten()
    labels = ['Prograde', 'Retrograde']
    t_counts = np.array([(t_pro == 1).sum(), (t_pro == 0).sum()])
    v_counts = np.array([(v_pro == 1).sum(), (v_pro == 0).sum()])
    x = np.arange(2)
    ax.bar(x - width/2, t_counts, width=width, color=TRAIN_COLOR, alpha=ALPHA,
           label=f'train (N={len(t_pro):,})', edgecolor='white', linewidth=0.3)
    ax.bar(x + width/2, v_counts, width=width, color=VAL_COLOR, alpha=ALPHA,
           label=f'val (N={len(v_pro):,})', edgecolor='white', linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Samples')
    ax.set_title('Orbit Direction')
    ax.legend(frameon=True, framealpha=0.9)
    ax.grid(axis='y')

    fig.tight_layout()
    out = PLOTS_DIR / f"dataset_summary_{tag}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=list(_DATASET_FILES), default=None,
                   help='Auto-select train/val paths by dataset name')
    p.add_argument('--train', type=str, default=None)
    p.add_argument('--val',   type=str, default=None)
    p.add_argument('--tag',   type=str, default=None)
    p.add_argument('--body',  choices=list(BODY_RADII), default=None,
                   help='Central body (auto-detected from tag if omitted).')
    args = p.parse_args()

    if args.dataset:
        train_path, val_path = _DATASET_FILES[args.dataset]
        tag = args.dataset
    elif args.train and args.val:
        train_path = args.train
        val_path   = args.val
        tag = args.tag or Path(args.train).stem
    else:
        p.error('Provide --dataset or both --train and --val')

    body = args.body or ('jupiter' if ('jovian' in tag or 'jupiter' in tag) else 'earth')
    plot_summary(train_path, val_path, tag, body=body)


if __name__ == '__main__':
    main()
