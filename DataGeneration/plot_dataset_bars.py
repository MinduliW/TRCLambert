"""
plot_dataset_bars.py — Yang-style error-bar plots (Figs. 7 & 8) for each
generated dataset.

For each case we draw two side-by-side subplots:
    Left:  initial velocity error Dv0 = v1_true - v1_lambert.
           Components (x, y, z) shown as red dots at the mean with blue
           +/- 1 sigma whiskers, connected by a thin blue line.
    Right: terminal position error Drf = r2 - rfd, same style. rfd is the
           result of propagating (r1, v1_lambert) under J2 for tof.

Run:
    python plot_dataset_bars.py                   # all data/*_train.npz
    python plot_dataset_bars.py path1.npz [...]   # explicit files
"""

import os
import sys
import glob
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
    """Propagate (r1, v1_lambert) under two-body + J2 dynamics for `tof`."""
    energy = 0.5 * float(np.dot(v1_lambert, v1_lambert)) - mu / float(np.linalg.norm(r1))
    if energy < 0.0:
        a = -mu / (2.0 * energy)
        period = 2.0 * np.pi * np.sqrt(a ** 3 / mu)
    else:
        period = float(tof)        # fall back for hyperbolic Lambert outputs
    rfd, _vfd, _, _ = propagate_j2_rk4(
        r1, v1_lambert, float(tof), mu, re, j2,
        steps_per_period=STEPS_PER_PERIOD, period=period,
    )
    return rfd


def compute_drf(data, mu, re, j2, progress_every=1000):
    n = data["r1"].shape[0]
    rfd = np.empty_like(data["r1"])
    for i in range(n):
        rfd[i] = lambert_terminal_under_j2(
            data["r1"][i], data["v1_lambert"][i], data["tof"][i], mu, re, j2,
        )
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    re-propagated {i + 1}/{n}", flush=True)
    return data["r2"] - rfd


def errorbar_panel(ax, vec, xlabels, ylabel):
    """Yang-style: red dot at mean, blue +/- 1 sigma whiskers, blue connector."""
    means = vec.mean(axis=0)
    stds  = vec.std(axis=0)
    x = np.arange(len(xlabels))
    ax.errorbar(
        x, means, yerr=stds,
        fmt="o-",
        color="royalblue",       # connecting line
        mfc="crimson", mec="crimson", ms=8,
        ecolor="royalblue", elinewidth=2.0, capsize=5, capthick=2.0,
    )
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.5, len(xlabels) - 0.5)
    ax.grid(True, alpha=0.3)


def plot_dataset(path):
    data = load_records(path)
    case = str(data["case"])
    body = str(data["body"])
    mu, re, j2 = body_constants(body)

    n = data["r1"].shape[0]
    print(f"plotting {path}   case={case}   body={body}   N={n}")

    dv0 = data["v1_true"]   - data["v1_lambert"]
    drf = compute_drf(data, mu, re, j2)

    fig, (ax_v, ax_r) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Lambert errors — {case} ({body}, N={n})", fontsize=13)

    errorbar_panel(
        ax_v, dv0,
        xlabels=[r"$\Delta v_{0dx}$", r"$\Delta v_{0dy}$", r"$\Delta v_{0dz}$"],
        ylabel="Error of initial velocity  $\\Delta v_0$  (km/s)",
    )
    errorbar_panel(
        ax_r, drf,
        xlabels=[r"$\Delta r_{fdx}$", r"$\Delta r_{fdy}$", r"$\Delta r_{fdz}$"],
        ylabel="Error of terminal position  $\\Delta r_f$  (km)",
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    os.makedirs(PLOTS_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(PLOTS_DIR, f"{base}_errorbars.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        paths = sorted(glob.glob("data/*_train.npz"))
        if not paths:
            print("no data/*_train.npz files found; pass paths explicitly")
            sys.exit(1)
    for p in paths:
        plot_dataset(p)
