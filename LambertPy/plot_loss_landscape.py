"""Diagnostic: probe pos-loss landscape around the labeled v1.

Hypothesis under test: vel supervision helps Jovian and hurts LEO because the
pos-loss inverse problem is well-conditioned at LEO but stiff/ill-conditioned
at Jupiter (long TOF amplifies ∂r/∂v).

Produces two figures:

1. pos_loss_slices.png — 1D slice of ||prop(r1, v1 + α·n̂, tof) − r2|| vs α
   along radial / in-track / cross-track directions, for nrev ∈ {0,3,6}.
   LEO and Jovian overlaid. If LEO shows a smooth shallow bowl and Jovian
   shows a much steeper (possibly rugged) valley, hypothesis confirmed.

2. jacobian_stats.png — histograms of ||∂r/∂v||₂ and condition number
   κ(J) = σ_max/σ_min across samples, for each dataset. Separation in
   ||J|| = magnitude of amplification; separation in κ = ill-conditioning.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lambert_trc_j2 import J2Propagator, BODY_PARAMS

THIS_DIR  = Path(__file__).resolve().parent
PLOTS_DIR = THIS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})


def rtn_basis(r, v):
    """Radial / in-track / cross-track unit vectors for state (r, v)."""
    r_hat = r / np.linalg.norm(r)
    h     = np.cross(r, v)
    h_hat = h / np.linalg.norm(h)
    t_hat = np.cross(h_hat, r_hat)
    return r_hat, t_hat, h_hat


def pick_samples(d, nrevs):
    """Return indices of first sample in d at each nrev in `nrevs`."""
    nrev = d['nrev'].astype(int).flatten()
    out = []
    for n in nrevs:
        idx = np.where(nrev == n)[0]
        out.append(int(idx[0]) if len(idx) else None)
    return out


def loss_slice(prop, r1, v1, r2, tof, n_hat, alphas_ms):
    """||propagate(r1, v1 + α·n̂, tof) − r2|| for α in alphas_ms (m/s)."""
    alphas_kms = alphas_ms * 1e-3                                   # (M,)
    v1_batch = v1[None, :] + alphas_kms[:, None] * n_hat[None, :]   # (M, 3)
    M = v1_batch.shape[0]
    r1_b  = torch.from_numpy(np.broadcast_to(r1,  (M, 3)).copy())
    v1_b  = torch.from_numpy(v1_batch)
    tof_b = torch.full((M, 1), float(tof), dtype=torch.float64)
    with torch.no_grad():
        rf, _ = prop(r1_b, v1_b, tof_b)
    rf  = rf.cpu().numpy()
    err = np.linalg.norm(rf - r2[None, :], axis=1)
    return err


def finite_diff_jacobian(prop, r1_arr, v1_arr, tof_arr, eps_ms=1.0):
    """Per-sample 3×3 Jacobian ∂r_f/∂v_0 via central differences (eps in m/s)."""
    N = r1_arr.shape[0]
    eps_kms = eps_ms * 1e-3
    J = np.empty((N, 3, 3))
    r1_t  = torch.from_numpy(r1_arr)
    v1_t  = torch.from_numpy(v1_arr)
    tof_t = torch.from_numpy(tof_arr).unsqueeze(-1)
    for i in range(3):
        dv = torch.zeros_like(v1_t)
        dv[:, i] = eps_kms
        with torch.no_grad():
            rf_p, _ = prop(r1_t, v1_t + dv, tof_t)
            rf_m, _ = prop(r1_t, v1_t - dv, tof_t)
        J[:, :, i] = ((rf_p - rf_m) / (2 * eps_kms)).cpu().numpy()
    return J


def make_slice_figure(samples):
    """samples: dict {label: {nrev: {'alphas': (M,), 'r': (M,), 't': (M,), 'h': (M,)}}}"""
    nrevs = sorted({n for s in samples.values() for n in s})
    fig, axes = plt.subplots(len(samples), len(nrevs),
                              figsize=(4.2 * len(nrevs), 3.6 * len(samples)),
                              sharex=True)
    if len(samples) == 1:
        axes = axes[None, :]
    if len(nrevs) == 1:
        axes = axes[:, None]
    for i, (label, per_nrev) in enumerate(samples.items()):
        for j, n in enumerate(nrevs):
            ax = axes[i, j]
            if n not in per_nrev:
                ax.set_visible(False); continue
            d = per_nrev[n]
            ax.plot(d['alphas'], d['r'], label='radial',       color='#d62728', lw=1.3)
            ax.plot(d['alphas'], d['t'], label='in-track',     color='#1f77b4', lw=1.3)
            ax.plot(d['alphas'], d['h'], label='cross-track',  color='#2ca02c', lw=1.3)
            ax.set_yscale('log')
            ax.set_title(f'{label}  ·  nrev={n}')
            ax.grid(True, which='both', alpha=0.3)
            if i == len(samples) - 1:
                ax.set_xlabel(r'$\alpha$ (m/s)')
            if j == 0:
                ax.set_ylabel(r'$||r_f(\alpha)-r_2||$ (km)')
            if i == 0 and j == 0:
                ax.legend(frameon=True, framealpha=0.9)
    fig.suptitle('Pos-loss 1D slice around labeled $v_1$  ·  perturbation in RTN basis',
                 fontsize=12, y=1.0)
    fig.tight_layout()
    out = PLOTS_DIR / 'pos_loss_slices.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'Saved: {out}')


def make_jacobian_figure(stats):
    """stats: dict {label: {'Jnorm': (N,), 'cond': (N,)}}"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {'LEO': '#5B9BD5', 'Jovian': '#F4A04A'}

    ax = axes[0]
    for label, s in stats.items():
        ax.hist(np.log10(s['Jnorm']), bins=50, alpha=0.65,
                label=f'{label} (med={np.median(s["Jnorm"]):.1e})',
                color=colors.get(label, None), edgecolor='white', linewidth=0.3)
    ax.set_xlabel(r'$\log_{10}\,||\partial r_f / \partial v_0||_2$  (km per km/s)')
    ax.set_ylabel('samples')
    ax.set_title('Jacobian spectral norm (amplification)')
    ax.legend(frameon=True, framealpha=0.9); ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    for label, s in stats.items():
        ax.hist(np.log10(s['cond']), bins=50, alpha=0.65,
                label=f'{label} (med κ={np.median(s["cond"]):.1e})',
                color=colors.get(label, None), edgecolor='white', linewidth=0.3)
    ax.set_xlabel(r'$\log_{10}\,\kappa(J) = \log_{10}(\sigma_{\max}/\sigma_{\min})$')
    ax.set_ylabel('samples')
    ax.set_title('Jacobian condition number (ill-conditioning)')
    ax.legend(frameon=True, framealpha=0.9); ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    out = PLOTS_DIR / 'jacobian_stats.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'Saved: {out}')


def run_dataset(label, npz_path, body, max_step, nrevs, alpha_max_ms,
                n_alpha, n_jac_samples):
    d = dict(np.load(npz_path))
    mu, re, j2 = BODY_PARAMS[body]
    prop = J2Propagator(max_step_s=max_step, mu=mu, re=re, j2=j2)

    # 1D slices
    idxs = pick_samples(d, nrevs)
    alphas = np.linspace(-alpha_max_ms, alpha_max_ms, n_alpha)
    slices = {}
    for n, i in zip(nrevs, idxs):
        if i is None: continue
        r1, v1, r2, tof = d['r1'][i], d['v1_j2'][i], d['r2'][i], float(d['tof'][i])
        r_hat, t_hat, h_hat = rtn_basis(r1, v1)
        slices[n] = {
            'alphas': alphas,
            'r': loss_slice(prop, r1, v1, r2, tof, r_hat, alphas),
            't': loss_slice(prop, r1, v1, r2, tof, t_hat, alphas),
            'h': loss_slice(prop, r1, v1, r2, tof, h_hat, alphas),
        }
        print(f'  [{label}] nrev={n}: slice range '
              f'{min(slices[n]["r"].min(), slices[n]["t"].min(), slices[n]["h"].min()):.2e} '
              f'to {max(slices[n]["r"].max(), slices[n]["t"].max(), slices[n]["h"].max()):.2e} km')

    # Jacobian stats
    N = min(n_jac_samples, len(d['r1']))
    rng = np.random.default_rng(0)
    sel = rng.choice(len(d['r1']), size=N, replace=False)
    J = finite_diff_jacobian(prop, d['r1'][sel], d['v1_j2'][sel], d['tof'][sel])
    svs = np.linalg.svd(J, compute_uv=False)   # (N, 3)
    Jnorm = svs[:, 0]
    cond  = svs[:, 0] / np.clip(svs[:, -1], 1e-30, None)
    print(f'  [{label}] ||J||: med={np.median(Jnorm):.3e}  '
          f'κ(J): med={np.median(cond):.3e}')

    return slices, {'Jnorm': Jnorm, 'cond': cond}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--leo-npz',    default='datasets/train_struct_10h.npz')
    ap.add_argument('--jovian-npz', default='datasets/train_struct_jupiter.npz')
    ap.add_argument('--nrevs',      nargs='+', type=int, default=[0, 3, 6])
    ap.add_argument('--alpha-max-ms', type=float, default=5.0,
                    help='± perturbation range [m/s]')
    ap.add_argument('--n-alpha',    type=int, default=101)
    ap.add_argument('--n-jac',      type=int, default=500,
                    help='samples per dataset for Jacobian stats')
    ap.add_argument('--leo-step',   type=float, default=30.0)
    ap.add_argument('--jovian-step', type=float, default=60.0)
    args = ap.parse_args()

    print('LEO ...')
    leo_slices, leo_jac = run_dataset('LEO', args.leo_npz, 'earth',
                                      args.leo_step, args.nrevs,
                                      args.alpha_max_ms, args.n_alpha, args.n_jac)
    print('Jovian ...')
    jov_slices, jov_jac = run_dataset('Jovian', args.jovian_npz, 'jupiter',
                                      args.jovian_step, args.nrevs,
                                      args.alpha_max_ms, args.n_alpha, args.n_jac)

    make_slice_figure({'LEO': leo_slices, 'Jovian': jov_slices})
    make_jacobian_figure({'LEO': leo_jac, 'Jovian': jov_jac})


if __name__ == '__main__':
    main()
