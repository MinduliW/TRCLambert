"""Evaluate PPO Lambert policy against Lambert-only baseline on J2 error."""

import argparse

import numpy as np
import torch

from dynamics import propagate_j2
from rl_policy import TRCRecurrentActorCritic
from trc_core import NetConfig


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PPO Lambert policy")
    p.add_argument("--dataset_path", type=str, default="data/lambert_test.npz")
    p.add_argument("--ckpt", type=str, default="LambertPy_RL/ppo_lambert.pt")
    p.add_argument("--n_prop_steps", type=int, default=300)
    p.add_argument("--max_samples", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    d = np.load(args.dataset_path)
    n = min(args.max_samples, len(d["r0"]))
    r0 = d["r0"][:n].astype(np.float64)
    v0 = d["v0"][:n].astype(np.float64)
    rt = d["r_target"][:n].astype(np.float64)
    vt = d["v_target"][:n].astype(np.float64)
    tof = d["tof"][:n].astype(np.float64).reshape(-1)
    dv1 = d["dv1"][:n].astype(np.float64)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    net = NetConfig(**ckpt["net"], dropout=0.0)
    model = TRCRecurrentActorCritic(ckpt["obs_dim"], ckpt["act_dim"], net=net)
    model.load_state_dict(ckpt["model"])
    model.eval()

    r_scale = float(np.mean(np.linalg.norm(r0, axis=1)))
    v_scale = float(np.mean(np.linalg.norm(v0, axis=1)))
    tof_scale = float(np.mean(tof))
    dv_scale = float(np.mean(np.linalg.norm(dv1, axis=1)))
    pos_scale = max(r_scale * 0.05, 50.0)

    dv = dv1.copy()
    obs0 = np.concatenate(
        [
            r0 / r_scale,
            v0 / v_scale,
            rt / r_scale,
            vt / v_scale,
            (tof / tof_scale)[:, None],
            dv / dv_scale,
            np.zeros((n, 3), dtype=np.float64),
        ],
        axis=1,
    ).astype(np.float32)
    with torch.no_grad():
        z_h, z_l = model.init_memory(torch.from_numpy(obs0))

    for _ in range(int(ckpt["horizon"])):
        err = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            rf, _, _ = propagate_j2(r0[i], v0[i] + dv[i], float(tof[i]), n_steps=args.n_prop_steps)
            err[i] = rf - rt[i]
        obs = np.concatenate(
            [
                r0 / r_scale,
                v0 / v_scale,
                rt / r_scale,
                vt / v_scale,
                (tof / tof_scale)[:, None],
                dv / dv_scale,
                err / pos_scale,
            ],
            axis=1,
        ).astype(np.float32)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs)
            dv_curr_t = obs_t[:, 13:16]
            mu, _, _, z_h, z_l = model.forward_step(obs_t, z_h, z_l, dv_curr_t)
        delta = np.tanh(mu.numpy()) * ckpt["action_scale_kms"]
        dv = dv + delta

    lambert_err = np.zeros(n, dtype=np.float64)
    rl_err = np.zeros(n, dtype=np.float64)
    for i in range(n):
        rl_rf, _, _ = propagate_j2(r0[i], v0[i] + dv[i], float(tof[i]), n_steps=args.n_prop_steps)
        l_rf, _, _ = propagate_j2(r0[i], v0[i] + dv1[i], float(tof[i]), n_steps=args.n_prop_steps)
        rl_err[i] = np.linalg.norm(rl_rf - rt[i])
        lambert_err[i] = np.linalg.norm(l_rf - rt[i])

    dv_delta_ms = np.linalg.norm(dv - dv1, axis=1) * 1000.0
    print(f"Dataset: {args.dataset_path} (N={n})")
    print(f"Lambert J2 err: mean={lambert_err.mean():.3f} km")
    print(f"RL      J2 err: mean={rl_err.mean():.3f} km")
    print(f"Mean correction |Δdv|: {dv_delta_ms.mean():.2f} m/s")


if __name__ == "__main__":
    main()
