"""
to_lambertpy.py — convert .npz datasets from this repo's schema to the
LambertPy schema (matching /Users/minduli/Lambert_TRC/LambertPy/datasets/).

Key remap (this repo  ->  LambertPy):
    n_rev       ->  nrev
    v1_lambert  ->  v1
    v1_true     ->  v1_j2
    v2_true     ->  v2

Computed:
    dv_j2 = v1_j2 - v1

Forward-sampled placeholders (LambertPy convention from train_leo_fwd.npz):
    j2_converged = 1.0   (every forward sample is "converged" by construction)
    j2_iters     = 0.0   (no shooter was run)
    j2_pos_err   = 1e-3  (LambertPy's default tolerance value)
    ncase        = 0.0   (single-case files; user can re-stamp if combining)

Dropped (not in LambertPy schema):
    branch, body, case

Run:
    python to_lambertpy.py                     # all data/*.npz (excl. *_lambertpy*)
    python to_lambertpy.py path1.npz [...]     # explicit files
Outputs: <input_stem>_lambertpy.npz alongside each input.
"""

import os
import sys
import glob
import numpy as np


LAMBERTPY_TOL = 1.0e-3   # the j2_pos_err placeholder used in train_leo_fwd.npz


def convert(path):
    data = np.load(path, allow_pickle=False)
    n = data["r1"].shape[0]

    out = {
        "r1":           data["r1"].astype(np.float64),
        "r2":           data["r2"].astype(np.float64),
        "tof":          data["tof"].astype(np.float64),
        "nrev":         data["n_rev"].astype(np.float64),
        "prograde":     data["prograde"].astype(np.float64),
        "v1":           data["v1_lambert"].astype(np.float64),
        "v1_j2":        data["v1_true"].astype(np.float64),
        "v2":           data["v2_true"].astype(np.float64),
        "dv_j2":        (data["v1_true"] - data["v1_lambert"]).astype(np.float64),
        "j2_converged": np.ones(n, dtype=np.float64),
        "j2_iters":     np.zeros(n, dtype=np.float64),
        "j2_pos_err":   np.full(n, LAMBERTPY_TOL, dtype=np.float64),
        "ncase":        np.zeros(n, dtype=np.float64),
    }

    base, _ = os.path.splitext(path)
    out_path = f"{base}_lambertpy.npz"
    np.savez_compressed(out_path, **out)
    print(f"  {path}  ->  {out_path}  (N={n})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        paths = sorted(p for p in glob.glob("data/*.npz") if "_lambertpy" not in p)
    if not paths:
        print("no .npz files found; pass paths explicitly")
        sys.exit(1)
    for p in paths:
        convert(p)
