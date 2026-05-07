"""
analyze_dataset.py — Yang-et-al-2022-style sample statistics for the .npz
datasets produced by lambert_dataset.py.

Two output modes:

  (default)  full text dump: per-component mean / std / magdiff for every
             Table-2 variable (r0, rf, v0, vd, Drf, Dv0, oe0, oef, oed, tof)
             in both Cartesian and spherical reps where applicable.

  --latex    focused LaTeX table with five rows only:
             tof, v_Lambert, v_J2, Dv0, Drf
             written to ./tables/<case>_stats.tex (and echoed to stdout).

Statistics (Yang Eq. 9):
    mean    : (1/n) sum X_j
    std     : sqrt((1/n) sum (X_j - mean)^2)              (population, ddof=0)
    magdiff : log10( max|X| / min(|X|>0) )                  (Yang's "rho")

Run:
    python analyze_dataset.py                          # text, all train sets
    python analyze_dataset.py path1.npz [...]
    python analyze_dataset.py --latex                  # LaTeX, all train sets
    python analyze_dataset.py --latex path1.npz
"""

import os
import sys
import glob
import argparse
import numpy as np

from constants import (
    MU_EARTH, R_EARTH, J2_EARTH,
    MU_JUPITER, R_JUPITER, J2_JUPITER,
    STEPS_PER_PERIOD,
)
from lambert_dataset import (
    cartesian_to_kepler,
    propagate_j2_rk4,
    load_records,
)


# ---------------------------------------------------------------------------
# Statistics (Yang Eq. 9)
# ---------------------------------------------------------------------------
def stats_per_component(arr):
    """arr shape (N,) or (N, k). Returns (mean, std, magdiff) arrays."""
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)              # population std (ddof = 0)
    abs_arr = np.abs(arr)

    magdiff = np.zeros(arr.shape[1])
    for j in range(arr.shape[1]):
        col = abs_arr[:, j]
        nonzero = col[col > 0.0]
        if nonzero.size:
            magdiff[j] = np.log10(nonzero.max() / nonzero.min())

    if mean.size == 1:
        return float(mean[0]), float(std[0]), float(magdiff[0])
    return mean, std, magdiff


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------
def cart_to_sph_batch(vec):
    """(N, 3) -> (N, 3) of (magnitude, azimuth in [-pi, pi], elevation in [-pi/2, pi/2])."""
    r  = np.linalg.norm(vec, axis=-1)
    az = np.arctan2(vec[:, 1], vec[:, 0])
    safe_r = np.where(r > 0.0, r, 1.0)
    el = np.arcsin(np.clip(vec[:, 2] / safe_r, -1.0, 1.0))
    return np.stack([r, az, el], axis=-1)


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
        period = float(tof)             # fall back for hyperbolic Lambert outputs

    rfd, _vfd, _nrev, _rmin = propagate_j2_rk4(
        r1, v1_lambert, float(tof), mu, re, j2,
        steps_per_period=STEPS_PER_PERIOD, period=period,
    )
    return rfd


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def fmt_vec(v, width=4):
    """Format a scalar or 1-D array as Yang-style '[a; b; c]' with `width` sig figs."""
    if np.ndim(v) == 0:
        return f"{float(v):.{width}g}"
    return "[" + "; ".join(f"{float(x):.{width}g}" for x in np.asarray(v)) + "]"


def print_row(name, arr):
    m, s, d = stats_per_component(arr)
    print(f"  {name:<10s}  mean = {fmt_vec(m):<48s}  "
          f"std = {fmt_vec(s):<48s}  magdiff = {fmt_vec(d)}")


# ---------------------------------------------------------------------------
# Per-file analysis
# ---------------------------------------------------------------------------
def analyze(path):
    data = load_records(path)
    case = str(data["case"])
    body = str(data["body"])
    mu, re, j2 = body_constants(body)

    n = data["r1"].shape[0]
    print(f"\n=== {path}   case={case}   body={body}   N={n} ===")

    # ---- derived quantities --------------------------------------------------
    # Re-propagate the Keplerian Lambert solution under J2 to get rfd, then Drf.
    rfd = np.empty_like(data["r1"])
    for i in range(n):
        rfd[i] = lambert_terminal_under_j2(
            data["r1"][i], data["v1_lambert"][i], data["tof"][i], mu, re, j2,
        )

    delta_rf = data["r2"] - rfd
    delta_v0 = data["v1_true"] - data["v1_lambert"]

    # Orbital elements at start, end, and for the Keplerian solution.
    oe0 = np.empty((n, 6))
    oef = np.empty((n, 6))
    oed = np.empty((n, 6))
    for i in range(n):
        e0 = cartesian_to_kepler(data["r1"][i], data["v1_true"][i],   mu)
        ef = cartesian_to_kepler(data["r2"][i], data["v2_true"][i],   mu)
        ed = cartesian_to_kepler(data["r1"][i], data["v1_lambert"][i], mu)
        for j, key in enumerate(("a", "e", "inc", "raan", "argp", "M0")):
            oe0[i, j] = e0[key]
            oef[i, j] = ef[key]
            oed[i, j] = ed[key]

    # Spherical reps for the 3-vectors.
    r0_sph  = cart_to_sph_batch(data["r1"])
    rf_sph  = cart_to_sph_batch(data["r2"])
    v0_sph  = cart_to_sph_batch(data["v1_true"])
    vd_sph  = cart_to_sph_batch(data["v1_lambert"])
    drf_sph = cart_to_sph_batch(delta_rf)
    dv0_sph = cart_to_sph_batch(delta_v0)

    # ---- table ---------------------------------------------------------------
    # Order matches Yang Table 2.
    print_row("r0_Cart",  data["r1"])
    print_row("r0_Sph",   r0_sph)
    print_row("rf_Cart",  data["r2"])
    print_row("rf_Sph",   rf_sph)
    print_row("v0_Cart",  data["v1_true"])
    print_row("v0_Sph",   v0_sph)
    print_row("oe0",      oe0)
    print_row("oef",      oef)
    print_row("vd_Cart",  data["v1_lambert"])
    print_row("vd_Sph",   vd_sph)
    print_row("Drf_Cart", delta_rf)
    print_row("Drf_Sph",  drf_sph)
    print_row("oed",      oed)
    print_row("tof",      data["tof"])
    print_row("Dv0_Cart", delta_v0)
    print_row("Dv0_Sph",  dv0_sph)

    print("  units: positions in km, velocities in km/s, angles in rad, tof in s.")


# ---------------------------------------------------------------------------
# LaTeX output (focused 5-row table)
# ---------------------------------------------------------------------------
TABLES_DIR = "tables"


def fmt_cell(v, sig=4):
    """Format a scalar or 3-vector as a math-mode LaTeX cell."""
    if np.ndim(v) == 0:
        return f"${float(v):.{sig}g}$"
    parts = ";\\ ".join(f"{float(x):.{sig}g}" for x in np.asarray(v).ravel())
    return f"$[{parts}]$"


def latex_table(case, body, n, rows):
    lines = [
        r"\begin{table}[hbt!]",
        r"\centering",
        rf"\caption{{Sample statistics for case \texttt{{{case}}} "
        rf"({body}, $N = {n}$). $\rho = \log_{{10}}\!"
        rf"\left( \dfrac{{\max(|X|)}}{{\min(|X|>0)}} \right)$ "
        rf"as in Yang et al.~2022, Eq.~9.}}",
        rf"\label{{tab:stats_{case}}}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Quantity & Mean & Std & $\rho$ \\",
        r"\midrule",
    ]
    for label, mean, std, rho in rows:
        lines.append(
            f"{label} & {fmt_cell(mean)} & {fmt_cell(std)} & {fmt_cell(rho)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def analyze_latex(path):
    data = load_records(path)
    case = str(data["case"])
    body = str(data["body"])
    mu, re, j2 = body_constants(body)
    n = data["r1"].shape[0]

    # Use the input filename's stem (e.g., "leo_single_val") so train/val
    # tables don't clobber each other.
    tag = os.path.splitext(os.path.basename(path))[0]

    print(f"\n=== {path}   case={case}   body={body}   N={n} ===", flush=True)

    # Re-propagate (r1, v1_lambert) under J2 to get rfd, then Drf.
    rfd = np.empty_like(data["r1"])
    for i in range(n):
        rfd[i] = lambert_terminal_under_j2(
            data["r1"][i], data["v1_lambert"][i], data["tof"][i], mu, re, j2,
        )
        if (i + 1) % 1000 == 0:
            print(f"  re-propagated {i+1}/{n}", flush=True)

    drf = data["r2"] - rfd
    dv0 = data["v1_true"] - data["v1_lambert"]

    rows = []
    for label, arr in (
        (r"$t_\mathrm{of}$ (s)",                  data["tof"]),
        (r"$\mathbf{v}_\mathrm{Lambert}$ (km/s)", data["v1_lambert"]),
        (r"$\mathbf{v}_{J_2}$ (km/s)",            data["v1_true"]),
        (r"$\Delta \mathbf{v}_0$ (km/s)",         dv0),
        (r"$\Delta \mathbf{r}_f$ (km)",           drf),
    ):
        m, s, r_ = stats_per_component(arr)
        rows.append((label, m, s, r_))

    tex = latex_table(tag, body, n, rows)
    os.makedirs(TABLES_DIR, exist_ok=True)
    out_path = os.path.join(TABLES_DIR, f"{tag}_stats.tex")
    with open(out_path, "w") as f:
        f.write(tex)
    print(f"  saved -> {out_path}", flush=True)
    print(tex, flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yang-style dataset stats.")
    parser.add_argument("paths", nargs="*",
                        help=".npz files (default: data/*_train.npz)")
    parser.add_argument("--latex", action="store_true",
                        help="emit a focused LaTeX table instead of the text dump")
    args = parser.parse_args()

    paths = args.paths or sorted(glob.glob("data/*_train.npz"))
    if not paths:
        print("no data/*_train.npz files found; pass paths explicitly")
        sys.exit(1)

    for p in paths:
        if args.latex:
            analyze_latex(p)
        else:
            analyze(p)
