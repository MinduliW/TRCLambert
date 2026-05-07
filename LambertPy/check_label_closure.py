"""Label-closure check: propagate (r1, v1_j2) under J2 and compare to stored r2.

Uses the exact same J2Propagator / RK4 used during TRC training — so this
measures the consistency between the stored labels and what the network
would see when forward-propagated.
"""

import argparse
import numpy as np
import torch

from lambert_trc_j2 import J2Propagator, BODY_PARAMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', type=str, default='datasets/train_struct_10h.npz')
    ap.add_argument('--body', type=str, default='earth', choices=list(BODY_PARAMS))
    ap.add_argument('--max-step', type=float, default=30.0,
                    help='RK4 max step [s]. 30 for Earth, 3600 for Jupiter.')
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--max-samples', type=int, default=0, help='0 = all')
    args = ap.parse_args()

    d = np.load(args.npz)
    N = len(d['r1']) if args.max_samples == 0 else min(args.max_samples, len(d['r1']))
    print(f'Dataset: {args.npz}  N={N}  (max_step={args.max_step}s, body={args.body})')

    mu, re, j2 = BODY_PARAMS[args.body]
    prop = J2Propagator(max_step_s=args.max_step, mu=mu, re=re, j2=j2)
    device = torch.device('cpu')

    r1 = torch.from_numpy(d['r1'][:N]).to(device)
    r2 = torch.from_numpy(d['r2'][:N]).to(device)
    v1 = torch.from_numpy(d['v1_j2'][:N]).to(device)
    tof = torch.from_numpy(d['tof'][:N]).to(device).unsqueeze(-1)

    errs = []
    for i in range(0, N, args.batch_size):
        sl = slice(i, min(i + args.batch_size, N))
        rf, _ = prop(r1[sl], v1[sl], tof[sl])
        err = torch.norm(rf - r2[sl], dim=-1).cpu().numpy()
        errs.append(err)
    errs = np.concatenate(errs)

    print(f'\nClosure error ||RK4(r1, v1_j2, tof) − r2_stored||  [km]:')
    for q in [0.5, 0.9, 0.95, 0.99, 1.0]:
        print(f'  q={q:4.2f} : {np.quantile(errs, q):12.4f}')
    print(f'  mean    : {errs.mean():12.4f}')
    print(f'  >1 km   : {(errs > 1).sum()}  ({100*(errs>1).mean():.2f}%)')
    print(f'  >10 km  : {(errs > 10).sum()}  ({100*(errs>10).mean():.2f}%)')
    print(f'  >100 km : {(errs > 100).sum()}  ({100*(errs>100).mean():.2f}%)')

    # Correlation with |Δv| (to see if high-Δv samples are also high-closure)
    dv = np.linalg.norm(d['v1_j2'][:N] - d['v1'][:N], axis=1) * 1000
    if errs.std() > 0 and dv.std() > 0:
        corr = np.corrcoef(errs, dv)[0, 1]
        print(f'\nCorrelation closure_err ↔ |Δv|: {corr:+.3f}')

    # By-nrev breakdown
    if 'nrev' in d.files:
        nrev = d['nrev'][:N].astype(int)
        print(f'\nClosure error by nrev:')
        print(f'  {"nrev":>4} {"N":>6} {"median":>10} {"p99":>10} {"max":>10}')
        for n in sorted(np.unique(nrev)):
            m = nrev == n
            e = errs[m]
            print(f'  {n:>4d} {m.sum():>6d} {np.median(e):>10.3f} '
                  f'{np.quantile(e,0.99):>10.3f} {e.max():>10.3f}')


if __name__ == '__main__':
    main()
