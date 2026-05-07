"""Visualize outlier Lambert transfer orbits vs typical ones.

Shows why Lambert fails for some cases: the transfer orbit dives close
to Jupiter where J2 dominates, causing huge deviation under J2 propagation.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

THIS_DIR = Path(__file__).resolve().parent
PLOTS_DIR = THIS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

RJ = 71492.0
MU = 126686534.9
J2 = 0.01475

plt.rcParams.update({
    'font.family':        'serif',
    'font.size':          11,
    'axes.titlesize':     12,
    'axes.labelsize':     11,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    9,
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


def j2_eom(t, state, mu, re, j2):
    r = state[:3]
    v = state[3:]
    rmag = np.linalg.norm(r)
    x, y, z = r
    r2 = rmag * rmag
    r5 = r2 * r2 * rmag
    a_grav = -mu / rmag**3 * r
    factor = 1.5 * j2 * mu * re**2 / r5
    common = 5 * z*z / r2
    a_j2 = factor * np.array([x*(common - 1), y*(common - 1), z*(common - 3)])
    return np.concatenate([v, a_grav + a_j2])


def kepler_eom(t, state, mu):
    r = state[:3]; v = state[3:]
    a = -mu / np.linalg.norm(r)**3 * r
    return np.concatenate([v, a])


def propagate(r0, v0, tof, dynamics='j2', n=400):
    t_eval = np.linspace(0, tof, n)
    if dynamics == 'j2':
        sol = solve_ivp(j2_eom, [0, tof], np.concatenate([r0, v0]),
                        args=(MU, RJ, J2), t_eval=t_eval,
                        rtol=1e-10, atol=1e-10, method='DOP853')
    else:
        sol = solve_ivp(kepler_eom, [0, tof], np.concatenate([r0, v0]),
                        args=(MU,), t_eval=t_eval,
                        rtol=1e-12, atol=1e-12, method='DOP853')
    return sol.y[:3].T  # (n, 3)


def main():
    d = np.load(THIS_DIR / 'datasets/train_struct_jupiter_0rev.npz')
    err = d['j2_pos_err']

    # Pick 4 outliers (top errors) and 4 typical (median errors)
    idx_sort = np.argsort(err)
    N = len(err)
    outlier_idx = idx_sort[-4:][::-1]           # top 4
    typical_idx = idx_sort[N//2 - 2: N//2 + 2]  # 4 middle ones

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    ax_xy   = fig.add_subplot(gs[0, 0])
    ax_rt   = fig.add_subplot(gs[0, 1])
    ax_j2   = fig.add_subplot(gs[0, 2])
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_scat = fig.add_subplot(gs[1, 1])
    ax_tbl  = fig.add_subplot(gs[1, 2]); ax_tbl.axis('off')

    # Draw Jupiter
    theta = np.linspace(0, 2*np.pi, 200)
    for ax in [ax_xy]:
        ax.fill(np.cos(theta), np.sin(theta), color='#D2B48C',
                alpha=0.85, zorder=10, label='Jupiter')
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, ec='sienna', lw=0.6, zorder=11))

    # Plot transfer orbits
    colors_out = plt.cm.Reds(np.linspace(0.55, 0.9, 4))
    colors_typ = plt.cm.Blues(np.linspace(0.45, 0.8, 4))

    def plot_case(idx, color, style, label_prefix):
        r0    = d['r1'][idx];  r2_t = d['r2'][idx]
        v_lam = d['v1'][idx];  v_tr = d['v1_j2'][idx]
        tof   = d['tof'][idx]

        # Propagate Lambert velocity under J2 (what Lambert thinks, plus J2)
        traj_lam_j2 = propagate(r0, v_lam, tof, 'j2')
        # Propagate Lambert under Keplerian (pure Lambert, what Lambert guarantees)
        traj_lam_k  = propagate(r0, v_lam, tof, 'kepler')

        # xy plane, normalized to RJ
        ax_xy.plot(traj_lam_k[:, 0]/RJ, traj_lam_k[:, 1]/RJ,
                   color=color, ls=style, lw=1.2, alpha=0.85)

        # distance from Jupiter vs time
        tvec = np.linspace(0, tof/3600, len(traj_lam_k))
        rmag_k  = np.linalg.norm(traj_lam_k,  axis=1)/RJ
        ax_rt.plot(tvec, rmag_k, color=color, ls=style, lw=1.2, alpha=0.85)

        # J2/gravity ratio along the trajectory
        j2_ratio = 1.5 * J2 * (RJ / np.linalg.norm(traj_lam_k, axis=1))**2
        ax_j2.plot(tvec, j2_ratio*100, color=color, ls=style, lw=1.2, alpha=0.85)

        return np.linalg.norm(traj_lam_j2[-1] - r2_t)  # final J2-prop err

    for i, idx in enumerate(outlier_idx):
        plot_case(idx, colors_out[i], '-', 'outlier')
    for i, idx in enumerate(typical_idx):
        plot_case(idx, colors_typ[i], '-', 'typical')

    ax_xy.set_xlabel('x (RJ)')
    ax_xy.set_ylabel('y (RJ)')
    ax_xy.set_title('Lambert Transfer Orbits (xy plane)')
    ax_xy.set_aspect('equal')
    ax_xy.grid()
    ax_xy.set_xlim(-35, 35); ax_xy.set_ylim(-35, 35)

    # Fake legend
    ax_xy.plot([], [], color=colors_out[1], lw=2, label='Outlier (err > 1500 km)')
    ax_xy.plot([], [], color=colors_typ[1], lw=2, label='Typical (err ≈ median)')
    ax_xy.legend(loc='upper left', fontsize=8)

    ax_rt.set_xlabel('Time (hr)')
    ax_rt.set_ylabel('|r| (RJ)')
    ax_rt.set_title('Distance from Jupiter vs. Time')
    ax_rt.axhline(5, color='k', ls=':', lw=0.6, alpha=0.5)
    ax_rt.grid()

    ax_j2.set_xlabel('Time (hr)')
    ax_j2.set_ylabel('|a_J2| / |a_grav| (%)')
    ax_j2.set_title('J2 Perturbation Strength Along Trajectory')
    ax_j2.set_yscale('log')
    ax_j2.grid(which='both')

    # Histogram of transfer perijove
    r1_mag = np.linalg.norm(d['r1'], axis=1)
    v1_mag = np.linalg.norm(d['v1'], axis=1)
    en = 0.5*v1_mag**2 - MU/r1_mag
    a_x = -MU/(2*en)
    h = np.cross(d['r1'], d['v1']); hmag = np.linalg.norm(h, axis=1)
    ecc = np.sqrt(np.clip(1 - (hmag**2/MU)/a_x, 0, None))
    rp_xfr = a_x*(1-ecc)/RJ

    bins = np.linspace(0, 35, 50)
    ax_hist.hist(rp_xfr[err < 1500],  bins=bins, color='#5B9BD5', alpha=0.7,
                 label=f'err < 1500 km (N={(err<1500).sum():,})',
                 edgecolor='white', linewidth=0.3)
    ax_hist.hist(rp_xfr[err >= 1500], bins=bins, color='#C0392B', alpha=0.7,
                 label=f'err ≥ 1500 km (N={(err>=1500).sum():,})',
                 edgecolor='white', linewidth=0.3)
    ax_hist.axvline(5, color='k', ls=':', lw=0.6, label='Jupiter surface (5 RJ min)')
    ax_hist.set_xlabel('Transfer orbit perijove (RJ)')
    ax_hist.set_ylabel('Samples')
    ax_hist.set_title('Distribution of Transfer Perijove')
    ax_hist.legend(fontsize=8)
    ax_hist.grid(axis='y')

    # Scatter: transfer perijove vs error
    sc = ax_scat.scatter(rp_xfr, err, c=np.log10(err+1),
                         cmap='plasma', s=4, alpha=0.5, rasterized=True)
    ax_scat.set_xlabel('Transfer orbit perijove (RJ)')
    ax_scat.set_ylabel('Lambert baseline error (km)')
    ax_scat.set_title('Error vs Transfer Perijove')
    ax_scat.set_yscale('log')
    ax_scat.axhline(1500, color='k', ls=':', lw=0.8)
    ax_scat.grid()

    # Stat table
    low = err < 100; mid = (err >= 100) & (err < 1500); high = err >= 1500
    stats = (
        f"0-rev Jupiter: Why does Lambert fail?\n\n"
        f"                 <100 km  100-1500  >1500 km\n"
        f"  r1 altitude   {r1_mag[low].mean()/RJ:6.1f}   "
        f"{r1_mag[mid].mean()/RJ:7.1f}   {r1_mag[high].mean()/RJ:7.1f} RJ\n"
        f"  Transfer rp   {rp_xfr[low].mean():6.1f}   "
        f"{rp_xfr[mid].mean():7.1f}   {rp_xfr[high].mean():7.1f} RJ\n"
        f"  Eccentricity  {ecc[low].mean():6.2f}   "
        f"{ecc[mid].mean():7.2f}   {ecc[high].mean():7.2f}\n"
        f"  N samples     {low.sum():6d}   {mid.sum():7d}   {high.sum():7d}\n\n"
        f"Outliers dive close to Jupiter where J2 is\n"
        f"strongest. A tiny velocity error there gets\n"
        f"amplified dramatically over the transfer."
    )
    ax_tbl.text(0.0, 0.9, stats, transform=ax_tbl.transAxes,
                fontfamily='monospace', fontsize=9, va='top')

    fig.suptitle("Outlier Analysis: Lambert Transfer Geometry vs. J2 Position Error",
                 fontsize=13, fontweight='bold', y=1.00)

    out = PLOTS_DIR / 'outlier_orbits.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
