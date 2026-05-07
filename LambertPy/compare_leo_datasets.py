"""Compare LEO datasets: shooting vs forward-propagation."""
import h5py
import numpy as np


def load_v73_struct(path, struct_key, nmax=5000):
    """Load fields from a v7.3 .mat struct-array. Handles both (1,N) and (N,1)
    outer orientations, and both (3,1) and (1,3) inner vector orientations."""
    with h5py.File(path, 'r') as f:
        grp = f[struct_key]
        field_names = list(grp.keys())
        outer_shape = grp[field_names[0]].shape
        N = max(outer_shape)
        print(f"  {path}")
        print(f"    outer shape: {outer_shape}  -> N = {N}  (loading first {min(N,nmax)})")
        Nload = min(N, nmax)

        row_major = outer_shape[0] == 1

        data = {fn: [] for fn in field_names}
        for i in range(Nload):
            for fn in field_names:
                ds = grp[fn]
                if row_major:
                    ref = ds[0, i]
                else:
                    ref = ds[i, 0]
                val = np.array(f[ref]).flatten()
                data[fn].append(val)
        out = {}
        for fn in field_names:
            stacked = np.stack(data[fn], axis=0)
            if stacked.shape[1] == 1:
                stacked = stacked.flatten()
            out[fn] = stacked
        return out, N


def summarize(name, d):
    print(f"\n=========  {name}  =========")
    r1 = d['r1']; r2 = d['r2']
    tof = d['tof']
    v1 = d['v1']
    v1_j2 = d['v1_j2']
    dv = d.get('dv_j2', v1_j2 - v1)

    r1m = np.linalg.norm(r1, axis=1)
    r2m = np.linalg.norm(r2, axis=1)
    v1m = np.linalg.norm(v1, axis=1)
    dvm = np.linalg.norm(dv, axis=1) * 1000

    print(f"  N (loaded)     : {len(r1)}")
    print(f"  |r1| [km]      : mean={r1m.mean():.1f}  min={r1m.min():.1f}  max={r1m.max():.1f}")
    alt1 = r1m - 6378.137
    print(f"  alt [km]       : min={alt1.min():.1f}  median={np.median(alt1):.1f}  max={alt1.max():.1f}")
    print(f"  tof [min]      : min={tof.min()/60:.1f}  median={np.median(tof)/60:.1f}  max={tof.max()/60:.1f}")
    print(f"  |v1| [km/s]    : mean={v1m.mean():.3f}  min={v1m.min():.3f}  max={v1m.max():.3f}")
    print(f"  |dv_j2| [m/s]  : median={np.median(dvm):.3f}  p90={np.quantile(dvm,0.9):.2f}  p99={np.quantile(dvm,0.99):.2f}  max={dvm.max():.2f}")
    print(f"  |dv_j2| quantiles (m/s):")
    for q in [0.10, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]:
        print(f"     q={q:.2f} : {np.quantile(dvm, q):12.4f}")
    print(f"  Fraction |dv_j2| < 0.01 m/s: {(dvm < 0.01).mean()*100:.1f}%")
    print(f"  Fraction |dv_j2| < 0.1  m/s: {(dvm < 0.1).mean()*100:.1f}%")
    print(f"  Fraction |dv_j2| < 1    m/s: {(dvm < 1).mean()*100:.1f}%")
    print(f"  Fraction |dv_j2| < 10   m/s: {(dvm < 10).mean()*100:.1f}%")
    print(f"  Fraction |dv_j2| > 100  m/s: {(dvm > 100).mean()*100:.1f}%")

    if 'nrev' in d:
        nrev = d['nrev'].astype(int)
        uniq, counts = np.unique(nrev, return_counts=True)
        print(f"  nrev distribution:")
        for u, c in zip(uniq, counts):
            print(f"     nrev={u:2d}: {c:5d} ({100*c/len(nrev):5.1f}%)")

    if 'prograde' in d:
        prog = d['prograde'].astype(bool)
        print(f"  prograde       : {prog.mean()*100:.1f}% prograde")

    T_est = 2*np.pi*np.sqrt(r1m**3/398600.4418)
    tof_over_T = tof / T_est
    print(f"  tof / T_est    : min={tof_over_T.min():.3f}  median={np.median(tof_over_T):.3f}  max={tof_over_T.max():.3f}")

    print(f"  Correlations of |dv_j2|:")
    print(f"    with tof     : {np.corrcoef(dvm, tof)[0,1]:+.3f}")
    print(f"    with nrev    : {np.corrcoef(dvm, d['nrev'])[0,1]:+.3f}")
    print(f"    with alt     : {np.corrcoef(dvm, alt1)[0,1]:+.3f}")


print("Loading 10h.mat (shooting-generated)...")
d_10h, _ = load_v73_struct('datasets/train_struct_10h.mat', 'train_info')
summarize("SHOOTING LEO (10h)", d_10h)

print("\nLoading train_leo_fwd.mat (forward-prop)...")
d_fwd, _ = load_v73_struct('datasets/train_leo_fwd.mat', 'train_info')
summarize("FORWARD-PROP LEO", d_fwd)
