"""
eval_with_lambertpy.py — run a LambertPy checkpoint on one of our generated
LambertPy-format datasets.

How it works:
  - LambertPy's `_load_mat_struct` looks for an .npz cache next to the
    requested .mat path. If the cache exists and the .mat is missing (or
    older), it loads from the cache and ignores the .mat path entirely.
  - We copy our `<case>_<split>_lambertpy.npz` into
    `LambertPy/datasets/<stem>.npz`, then call `evaluate()` with a matching
    .mat stem (which need not exist).

Default: leo_single_val on the trc_learned_lambert_single_rev_best.pt
checkpoint, K = [1, 2, 3, 5].

Run:
    python eval_with_lambertpy.py
    python eval_with_lambertpy.py --case leo_single --split val \
        --ckpt trc_vel_supervised_single_rev_best
"""

import os
import sys
import shutil
import argparse


LAMBERTPY_ROOT = "/Users/minduli/Lambert_TRC/LambertPy"
LAMBERTPY_DATASETS = os.path.join(LAMBERTPY_ROOT, "datasets")
LAMBERTPY_CKPTS    = os.path.join(LAMBERTPY_ROOT, "checkpoints")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case",  default="leo_single")
    ap.add_argument("--split", default="val", choices=("train", "val"))
    ap.add_argument("--ckpt",  default="trc_learned_lambert_single_rev_best",
                    help="checkpoint stem (no .pt extension) under LambertPy/checkpoints")
    ap.add_argument("--K", nargs="+", type=int, default=[1, 2, 3, 5])
    args = ap.parse_args()

    src = os.path.abspath(
        f"data/{args.case}_{args.split}_lambertpy.npz"
    )
    if not os.path.isfile(src):
        sys.exit(f"missing source dataset: {src}\nrun generate_datasets.py first.")

    # Stage our .npz into LambertPy/datasets/ as a cache that pretends to be
    # a .mat file's sibling cache.
    stem  = f"{args.case}_{args.split}_lambertpy"
    cache = os.path.join(LAMBERTPY_DATASETS, f"{stem}.npz")
    fake_mat = os.path.join(LAMBERTPY_DATASETS, f"{stem}.mat")
    shutil.copy2(src, cache)
    print(f"staged {src}  ->  {cache}")

    ckpt_path = os.path.join(LAMBERTPY_CKPTS, f"{args.ckpt}.pt")
    if not os.path.isfile(ckpt_path):
        sys.exit(f"missing checkpoint: {ckpt_path}")

    # Make LambertPy importable. `lambert_trc_j2` does `from trc import ...`,
    # which resolves via this script's directory (DataGeneration/trc.py).
    sys.path.insert(0, LAMBERTPY_ROOT)
    os.chdir(LAMBERTPY_ROOT)              # in case the code uses relative paths

    import lambert_trc_j2
    from lambert_trc_j2 import evaluate

    # Checkpoints were pickled with __main__.TrainConfig (because the original
    # run had lambert_trc_j2 as the main module). Re-expose every public class
    # in this run's __main__ so torch.load can resolve them.
    import __main__
    for name in dir(lambert_trc_j2):
        obj = getattr(lambert_trc_j2, name)
        if isinstance(obj, type) and not name.startswith("_"):
            setattr(__main__, name, obj)

    evaluate(
        ckpt_path=ckpt_path,
        mat_path=fake_mat,                 # only the stem matters; .mat need not exist
        struct_key="val_info",             # ignored when cache hits
        K_list=list(args.K),
    )


if __name__ == "__main__":
    main()
