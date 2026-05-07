"""
generate_datasets.py — produce train + validation .npz files for each case.

For each case in CASES this script generates two datasets:
    <case>_train.npz   (N_TRAIN samples, seeded with SEED_TRAIN + case offset)
    <case>_val.npz     (N_VAL   samples, seeded with SEED_VAL   + case offset)

The two splits use independent RNG streams so the validation set never
contains a draw that appeared in training.

Edit the constants below to scale up, then:
    python generate_datasets.py
"""

import os
import time
import numpy as np

from lambert_dataset import generate_dataset, save_records
from to_lambertpy import convert as to_lambertpy


# ---- Configuration ----------------------------------------------------------
N_TRAIN = 200e3
N_VAL   = 20e3

SEED_TRAIN = 42
SEED_VAL   = 1337

OUTPUT_DIR      = "data"
CASES           = ("leo_single","leo_multi", "jovian")
EMIT_LAMBERTPY  = True       # also write <case>_<split>_lambertpy.npz


def run_split(case, n_samples, seed, split_name):
    """Generate one split, save it, and return summary stats."""
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    records, attempts, rejects = generate_dataset(case, n_samples, rng=rng)
    elapsed = time.perf_counter() - t0

    out_path = os.path.join(OUTPUT_DIR, f"{case}_{split_name}.npz")
    save_records(records, out_path)

    diffs = np.array([
        np.linalg.norm(r["v1_lambert"] - r["v1_true"]) for r in records
    ])
    print(f"  {split_name:>5s}: {len(records):>5d} samples  "
          f"({rejects} rejected, {elapsed:5.1f}s)  "
          f"median |v_err| = {np.median(diffs):.3e} km/s  "
          f"-> {out_path}")

    if EMIT_LAMBERTPY:
        to_lambertpy(out_path)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Writing datasets to ./{OUTPUT_DIR}/  "
          f"(N_TRAIN={N_TRAIN}, N_VAL={N_VAL})\n")

    for i, case in enumerate(CASES):
        print(f"=== {case} ===")
        run_split(case, N_TRAIN, SEED_TRAIN + i, "train")
        run_split(case, N_VAL,   SEED_VAL   + i, "val")
        print()

    print("Done.")
