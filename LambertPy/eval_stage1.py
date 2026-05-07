import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from lambert_trc_j2 import (
    LambertTRCJ2, MatLambertDataset, NetConfig, TrainConfig
)

CKPT   = 'checkpoints/trc_j2_stage1_best.pt'
MAT    = 'val_struct.mat'
KEY    = 'val_info'
BATCH  = 512


@torch.no_grad()
def eval_stage1(ckpt_path: str, mat_path: str, struct_key: str):
    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: TrainConfig = ckpt['cfg']
    norm = ckpt['norm']

    print(f"Checkpoint: {ckpt_path}  (epoch {ckpt['epoch']}, "
          f"saved val err {ckpt['v1_err_ms']:.1f} m/s)")

    net = NetConfig(d_z=cfg.d_z, d_h=cfg.d_h,
                    n_heads=cfg.n_heads, n_blocks=cfg.n_blocks,
                    K=cfg.K, n_inner=cfg.n_inner)
    model = LambertTRCJ2(net,
                         max_step_s=cfg.max_step_s,
                         v_max=cfg.v_max,
                         pos_scale_km=cfg.pos_scale_km).to(device)
    model.set_normalisation(**norm, pos_scale_km=cfg.pos_scale_km)
    model.load_state_dict(ckpt['model'])
    model.eval()

    ds     = MatLambertDataset(mat_path, struct_key)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)

    v1_errs   = []   # ||v1_pred - v1_lambert|| [m/s]
    v1_preds  = []
    v1_trues  = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch['r1'], batch['r2'], batch['tof'],
                    batch['prograde'], stage1_only=True)
        err = (torch.norm(out['v1_pred'] - batch['v1_lambert'], dim=-1) * 1000).cpu()
        v1_errs.append(err)
        v1_preds.append(out['v1_pred'].cpu())
        v1_trues.append(batch['v1_lambert'].cpu())

    v1_errs  = torch.cat(v1_errs)
    v1_preds = torch.cat(v1_preds)
    v1_trues = torch.cat(v1_trues)

    # ── Summary stats ─────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  N samples   : {len(v1_errs)}")
    print(f"  Mean  err   : {v1_errs.mean():.1f} m/s")
    print(f"  Median err  : {v1_errs.median():.1f} m/s")
    print(f"  Std   err   : {v1_errs.std():.1f} m/s")
    print(f"  90th pct    : {v1_errs.quantile(0.90):.1f} m/s")
    print(f"  99th pct    : {v1_errs.quantile(0.99):.1f} m/s")
    print(f"  Max   err   : {v1_errs.max():.1f} m/s")
    print(f"  < 50  m/s   : {(v1_errs < 50).float().mean()*100:.1f}%")
    print(f"  < 100 m/s   : {(v1_errs < 100).float().mean()*100:.1f}%")
    print(f"  < 200 m/s   : {(v1_errs < 200).float().mean()*100:.1f}%")
    print(f"{'='*50}\n")

    # ── Per-component error ────────────────────────────────────────────────────
    comp_err = (v1_preds - v1_trues).abs() * 1000   # (N, 3) m/s
    labels = ['v_x', 'v_y', 'v_z']
    print("Per-component MAE [m/s]:")
    for i, lbl in enumerate(labels):
        print(f"  {lbl}: {comp_err[:, i].mean():.1f} m/s")

    # ── Plots ──────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family':       'serif',
        'font.size':         11,
        'axes.titlesize':    12,
        'axes.labelsize':    11,
        'xtick.labelsize':   9,
        'ytick.labelsize':   9,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'savefig.dpi':       300,
        'savefig.bbox':      'tight',
    })
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    # Boxplot of ||Δv|| overall + per component
    ax = axes[0]
    box_data = [v1_errs.numpy()] + [comp_err[:, i].numpy() for i in range(3)]
    bp = ax.boxplot(box_data,
                    labels=[r'$\parallel \mathbf{e}^{(0)}\parallel$', r'$v_x$', r'$v_y$', r'$v_z$'],
                    showfliers=False, patch_artist=True,
                    flierprops=dict(marker='.', markersize=2,
                                    alpha=0.3, linestyle='none',
                                    markeredgecolor='none',
                                    markerfacecolor='grey'))
    colors = ['steelblue', 'green', 'green', 'green']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Annotate each median with an arrow + label
    for i, (data, med_line) in enumerate(zip(box_data, bp['medians']), start=1):
        med_val = np.median(data)
        x = med_line.get_xdata().mean()
        y_top = ax.get_ylim()[1]
        ax.annotate(f'{med_val:.1f}',
                    xy=(x, med_val),
                    xytext=(x + 0.3, med_val + (y_top - med_val) * 0.25),
                    fontsize=8,
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

    ax.set_ylabel(r'Error [m/s]')
    ax.set_title(r'Initial Velocity Error by Component')
    # Error histogram
    ax = axes[1]
    ax.hist(v1_errs.numpy(), bins=80, edgecolor='none')
    ax.set_xlim(0, v1_errs.quantile(0.99).item())
    ax.set_xlabel(r'$\left\|\mathbf{e}^{(0)}\right\|$ [m/s]')
    ax.set_ylabel('Count')
    ax.set_title(r'Distribution of Initial Velocity Error')


    # Predicted vs true speed
    ax = axes[2]
    pred_speed = torch.norm(v1_preds, dim=-1).numpy()
    true_speed = torch.norm(v1_trues, dim=-1).numpy()
    ax.scatter(true_speed, pred_speed, s=1, alpha=0.3)
    lim = [min(true_speed.min(), pred_speed.min()),
           max(true_speed.max(), pred_speed.max())]
    ax.plot(lim, lim, 'r--', linewidth=1)
    ax.set_xlabel(r'$|v_{1,\mathrm{Lambert}}|$ [km/s]')
    ax.set_ylabel(r'$|{v}_1|$ [km/s]')
    ax.set_title(r'Predicted vs. True Departure Speed')


    # Error as % of true speed
    ax = axes[3]
    true_speed_ms = torch.norm(v1_trues, dim=-1).numpy() * 1000  # m/s
    rel_err = v1_errs.numpy() / true_speed_ms * 100              # %
    ax.hist(rel_err, bins=80, edgecolor='none')
    ax.set_xlim(0, np.percentile(rel_err, 99))
    ax.set_xlabel(r'$\left\|\mathbf{e}^{(0)}\right\| \,/\, |v_{1,\mathrm{Lambert}}|$ [%]')
    ax.set_ylabel('Count')
    ax.set_title(r'Relative Initial Velocity Error')


    for ax in axes:
        ax.minorticks_on()
        ax.grid(True, which='major', linewidth=0.6, color='#cccccc')
        ax.grid(True, which='minor', linewidth=0.3, color='#e8e8e8')
        ax.set_axisbelow(True)

    plt.tight_layout()
    out_path = ckpt_path.replace('.pt', '_eval.png')
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")
    plt.show()


if __name__ == '__main__':
    eval_stage1(CKPT, MAT, KEY)
