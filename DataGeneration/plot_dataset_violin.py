"""
plot_dataset_violin.py — violin-plot version of plot_dataset_bars.

Two side-by-side panels per dataset:
    Left:  Dv0 = v1_true - v1_lambert  components (x, y, z)   [km/s]
    Right: Drf = r2 - rfd              components (x, y, z)   [km]

For each component we draw a violin showing the full per-sample distribution,
with the median, mean, and extrema overlaid. rfd (J2 propagation of
r1 with v1_lambert) is computed once and cached to <stem>_rfd.npy so
subsequent runs are fast.

Run:
    python plot_dataset_violin.py                            # all data/*_train.npz
    python plot_dataset_violin.py path1.npz [...]
    python plot_dataset_violin.py --n 20000 data/jovian_train.npz   # subsample
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

from constants import (
    MU_EARTH, R_EARTH, J2_EARTH,
    MU_JUPITER, R_JUPITER, J2_JUPITER,
    STEPS_PER_PERIOD,
)
from lambert_dataset import propagate_j2_rk4, load_records


PLOTS_DIR = "plots"


def body_constants(body_name):
    if str(body_name) == "earth":
        return MU_EARTH, R_EARTH, J2_EARTH
    if str(body_name) == "jupiter":
        return MU_JUPITER, R_JUPITER, J2_JUPITER
    raise ValueError(f"unknown body: {body_name}")


def lambert_terminal_under_j2(r1, v1_lambert, tof, mu, re, j2):
    energy = 0.5 * float(np.dot(v1_lambert, v1_lambert)) - mu / float(np.linalg.norm(r1))
    if energy < 0.0:
        a = -mu / (2.0 * energy)
        period = 2.0 * np.pi * np.sqrt(a ** 3 / mu)
    else:
        period = float(tof)
    rfd, _, _, _ = propagate_j2_rk4(
        r1, v1_lambert, float(tof), mu, re, j2,
        steps_per_period=STEPS_PER_PERIOD, period=period,
    )
    return rfd


def get_or_compute_rfd(path, data, mu, re, j2):
    """Cache rfd on disk so subsequent plots/analyses skip the slow propagation."""
    cache = f"{os.path.splitext(path)[0]}_rfd.npy"
    n = data["r1"].shape[0]
    if os.path.exists(cache):
        rfd = np.load(cache)
        if rfd.shape == data["r1"].shape:
            print(f"  loaded cached rfd from {cache}", flush=True)
            return rfd
        print(f"  cached rfd has wrong shape; recomputing", flush=True)

    print(f"  computing rfd for {n} samples (cache miss)", flush=True)
    rfd = np.empty_like(data["r1"])
    for i in range(n):
        rfd[i] = lambert_terminal_under_j2(
            data["r1"][i], data["v1_lambert"][i], data["tof"][i], mu, re, j2,
        )
        if (i + 1) % 5000 == 0:
            print(f"    re-propagated {i + 1}/{n}", flush=True)
    np.save(cache, rfd)
    print(f"  cached rfd -> {cache}", flush=True)
    return rfd


def violin_panel(ax, vec, xlabels, ylabel):
    parts = ax.violinplot(
        [vec[:, j] for j in range(vec.shape[1])],
        positions=np.arange(vec.shape[1]),
        showmeans=False,
        showmedians=True,
        showextrema=True,
        widths=0.7,
    )
    for body in parts["bodies"]:
        body.set_facecolor("steelblue")
        body.set_edgecolor("navy")
        body.set_alpha(0.55)
    for line_key in ("cmedians", "cmaxes", "cmins", "cbars"):
        if line_key in parts:
            parts[line_key].set_color("navy")
            parts[line_key].set_linewidth(1.2)

    # Mean as a red dot for direct comparison with the errorbar plots.
    means = vec.mean(axis=0)
    ax.scatter(np.arange(vec.shape[1]), means,
               color="crimson", s=40, zorder=5, label="mean")

    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xticks(np.arange(vec.shape[1]))
    ax.set_xticklabels(xlabels, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.5, vec.shape[1] - 0.5)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)


def plot_dataset(path, n_subsample=0):
    data = load_records(path)
    case = str(data["case"])
    body = str(data["body"])
    mu, re, j2 = body_constants(body)
    n = data["r1"].shape[0]

    print(f"plotting {path}   case={case}   body={body}   N={n}", flush=True)

    rfd = get_or_compute_rfd(path, data, mu, re, j2)
    drf = data["r2"] - rfd
    dv0 = data["v1_true"] - data["v1_lambert"]

    # Optional subsample (only affects the plot — the full rfd is still cached).
    if n_subsample and n_subsample < n:
        rng = np.random.default_rng(seed=0)
        idx = rng.choice(n, size=n_subsample, replace=False)
        dv0 = dv0[idx]
        drf = drf[idx]
        n_plot = n_subsample
        print(f"  subsampled to {n_subsample}/{n} for plotting", flush=True)
    else:
        n_plot = n

    fig, (ax_v, ax_r) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Lambert errors — {case} ({body}, N={n_plot})", fontsize=13)

    violin_panel(
        ax_v, dv0,
        xlabels=[r"$\Delta v_{0dx}$", r"$\Delta v_{0dy}$", r"$\Delta v_{0dz}$"],
        ylabel="Error of initial velocity  $\\Delta v_0$  (km/s)",
    )
    violin_panel(
        ax_r, drf,
        xlabels=[r"$\Delta r_{fdx}$", r"$\Delta r_{fdy}$", r"$\Delta r_{fdz}$"],
        ylabel="Error of terminal position  $\\Delta r_f$  (km)",
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    os.makedirs(PLOTS_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(PLOTS_DIR, f"{base}_violins.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  -> {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*",
                    help=".npz files to plot (default: all data/*_train.npz)")
    ap.add_argument("--n", type=int, default=0,
                    help="random subsample size for the plot (0 = use all)")
    args = ap.parse_args()

    paths = args.paths or sorted(glob.glob("data/*_train.npz"))
    if not paths:
        sys.exit("no data/*_train.npz files found; pass paths explicitly")

    for p in paths:
        plot_dataset(p, n_subsample=args.n)
