"""
plot_dataset.py — plot the initial-velocity error and terminal-position error
distributions for each generated dataset (Yang et al. 2022, Figs. 7-8).

For every sample we plot:
    Dv0 = v1_true - v1_lambert
            -- the correction the Keplerian Lambert solver "got wrong"
    Drf = r2 - rfd
            -- the J2 terminal-position error of the Keplerian Lambert solution,
               where rfd comes from propagating (r1, v1_lambert) under J2.

Each figure is a 2 x 3 grid (rows: Dv0, Drf; columns: x, y, z components).
Histograms are overlaid with the per-component mean and +/- 1 sigma.

Run:
    python plot_dataset.py                    # all data/*_train.npz
    python plot_dataset.py path1.npz [...]    # explicit files
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


def compute_drf(data, mu, re, j2):
    n = data["r1"].shape[0]
    rfd = np.empty_like(data["r1"])
    for i in range(n):
        rfd[i] = lambert_terminal_under_j2(
            data["r1"][i], data["v1_lambert"][i], data["tof"][i], mu, re, j2,
        )
    return data["r2"] - rfd


def plot_component(ax, values, label, unit):
    """Histogram of one component with mean and +/- 1 sigma overlaid."""
    mean = float(values.mean())
    std  = float(values.std())
    ax.hist(values, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(mean,         color="crimson", linestyle="-",  linewidth=1.4,
               label=f"mean = {mean:.3g}")
    ax.axvline(mean + std,   color="crimson", linestyle="--", linewidth=1.0,
               label=f"std  = {std:.3g}")
    ax.axvline(mean - std,   color="crimson", linestyle="--", linewidth=1.0)
    ax.axvline(0.0,          color="black",   linestyle=":",  linewidth=0.8)
    ax.set_xlabel(f"{label} [{unit}]")
    ax.set_ylabel("count")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)


def plot_dataset(path):
    data = load_records(path)
    case = str(data["case"])
    body = str(data["body"])
    mu, re, j2 = body_constants(body)

    n = data["r1"].shape[0]
    print(f"plotting {path}   case={case}   body={body}   N={n}")

    dv0 = data["v1_true"] - data["v1_lambert"]
    drf = compute_drf(data, mu, re, j2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f"Lambert errors — {case} ({body}, N={n})",
        fontsize=13,
    )

    for j, comp in enumerate(("x", "y", "z")):
        plot_component(axes[0, j], dv0[:, j],
                       rf"$\Delta v_{{0,{comp}}}$", "km/s")
    for j, comp in enumerate(("x", "y", "z")):
        plot_component(axes[1, j], drf[:, j],
                       rf"$\Delta r_{{f,{comp}}}$", "km")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    os.makedirs(PLOTS_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(PLOTS_DIR, f"{base}_errors.png")
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
