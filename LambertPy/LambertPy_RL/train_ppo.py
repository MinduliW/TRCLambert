"""Train PPO agent for Lambert J2 correction without supervision labels."""

import argparse
from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Normal

from rl_env import LambertBatchEnv, RLEnvConfig
from rl_policy import TRCRecurrentActorCritic
from trc_core import NetConfig


@dataclass
class PPOConfig:
    dataset_path: str = "data/lambert_train.npz"
    save_path: str = "LambertPy_RL/ppo_lambert.pt"
    seed: int = 42
    epochs: int = 200
    steps_per_epoch: int = 16
    batch_size: int = 32
    horizon: int = 3
    n_prop_steps: int = 300
    action_scale_kms: float = 0.02
    lambda_action: float = 0.05
    d_z: int = 128
    d_h: int = 256
    n_heads: int = 4
    n_blocks: int = 2
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    train_iters: int = 10
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0


def compute_gae(rew, val, done, gamma=0.99, lam=0.95):
    t, b = rew.shape
    adv = np.zeros_like(rew, dtype=np.float32)
    ret = np.zeros_like(rew, dtype=np.float32)
    gae = np.zeros(b, dtype=np.float32)
    next_v = np.zeros(b, dtype=np.float32)
    for k in reversed(range(t)):
        non_terminal = 1.0 - done[k].astype(np.float32)
        delta = rew[k] + gamma * next_v * non_terminal - val[k]
        gae = delta + gamma * lam * non_terminal * gae
        adv[k] = gae
        ret[k] = adv[k] + val[k]
        next_v = val[k]
    return adv, ret


def parse_args():
    p = argparse.ArgumentParser(description="PPO for Lambert correction (no supervision)")
    p.add_argument("--dataset_path", type=str, default="data/lambert_train.npz")
    p.add_argument("--save_path", type=str, default="LambertPy_RL/ppo_lambert.pt")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--steps_per_epoch", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--n_prop_steps", type=int, default=300)
    p.add_argument("--action_scale_kms", type=float, default=0.02)
    p.add_argument("--lambda_action", type=float, default=0.05)
    p.add_argument("--d_z", type=int, default=128)
    p.add_argument("--d_h", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_blocks", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = PPOConfig(
        dataset_path=args.dataset_path,
        save_path=args.save_path,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        horizon=args.horizon,
        n_prop_steps=args.n_prop_steps,
        action_scale_kms=args.action_scale_kms,
        lambda_action=args.lambda_action,
        d_z=args.d_z,
        d_h=args.d_h,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks,
        lr=args.lr,
        seed=args.seed,
    )

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cpu")

    env = LambertBatchEnv(
        RLEnvConfig(
            dataset_path=cfg.dataset_path,
            batch_size=cfg.batch_size,
            horizon=cfg.horizon,
            n_prop_steps=cfg.n_prop_steps,
            action_scale_kms=cfg.action_scale_kms,
            lambda_action=cfg.lambda_action,
        ),
        seed=cfg.seed,
    )
    net = NetConfig(
        d_z=cfg.d_z,
        d_h=cfg.d_h,
        n_heads=cfg.n_heads,
        n_blocks=cfg.n_blocks,
        dropout=0.0,
    )
    model = TRCRecurrentActorCritic(env.obs_dim, env.act_dim, net=net).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print(f"Dataset={cfg.dataset_path} | obs={env.obs_dim}, act={env.act_dim}")
    print(f"PPO epochs={cfg.epochs}, steps_per_epoch={cfg.steps_per_epoch}, horizon={cfg.horizon}")

    for ep in range(1, cfg.epochs + 1):
        obs_buf, z_h_buf, z_l_buf = [], [], []
        act_buf, logp_buf = [], []
        rew_buf, val_buf, done_buf = [], [], []
        final_err = []
        final_delta = []

        obs = env.reset()
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().to(device)
            z_h, z_l = model.init_memory(obs_t)
        for _ in range(cfg.steps_per_epoch):
            for _ in range(cfg.horizon):
                obs_t = torch.from_numpy(obs).float().to(device)
                dv_curr = obs_t[:, 13:16]
                mu, std, v, z_h_next, z_l_next = model.forward_step(obs_t, z_h, z_l, dv_curr)
                dist = Normal(mu, std)
                act = dist.sample()
                logp = dist.log_prob(act).sum(-1)
                next_obs, rew, done, info = env.step(act.detach().cpu().numpy())

                obs_buf.append(obs.copy())
                z_h_buf.append(z_h.detach().cpu().numpy())
                z_l_buf.append(z_l.detach().cpu().numpy())
                act_buf.append(act.detach().cpu().numpy())
                logp_buf.append(logp.detach().cpu().numpy())
                rew_buf.append(rew.copy())
                val_buf.append(v.detach().cpu().numpy())
                done_buf.append(np.full(cfg.batch_size, done, dtype=bool))
                obs = next_obs
                z_h = z_h_next.detach()
                z_l = z_l_next.detach()

            final_err.append(float(np.mean(info["pos_err_km"])))
            final_delta.append(float(np.mean(info["dv_delta_ms"])))
            obs = env.reset()
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).float().to(device)
                z_h, z_l = model.init_memory(obs_t)

        obs_arr = np.asarray(obs_buf, dtype=np.float32)
        z_h_arr = np.asarray(z_h_buf, dtype=np.float32)
        z_l_arr = np.asarray(z_l_buf, dtype=np.float32)
        act_arr = np.asarray(act_buf, dtype=np.float32)
        logp_arr = np.asarray(logp_buf, dtype=np.float32)
        rew_arr = np.asarray(rew_buf, dtype=np.float32)
        val_arr = np.asarray(val_buf, dtype=np.float32)
        done_arr = np.asarray(done_buf, dtype=bool)

        adv, ret = compute_gae(rew_arr, val_arr, done_arr, gamma=cfg.gamma, lam=cfg.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = obs_arr.shape[0] * obs_arr.shape[1]
        obs_t = torch.from_numpy(obs_arr.reshape(n, env.obs_dim)).float().to(device)
        z_h_t = torch.from_numpy(z_h_arr.reshape(n, cfg.d_z)).float().to(device)
        z_l_t = torch.from_numpy(z_l_arr.reshape(n, cfg.d_z)).float().to(device)
        act_t = torch.from_numpy(act_arr.reshape(n, env.act_dim)).float().to(device)
        old_logp_t = torch.from_numpy(logp_arr.reshape(n)).float().to(device)
        adv_t = torch.from_numpy(adv.reshape(n)).float().to(device)
        ret_t = torch.from_numpy(ret.reshape(n)).float().to(device)

        for _ in range(cfg.train_iters):
            dv_curr_t = obs_t[:, 13:16]
            mu, std, v, _, _ = model.forward_step(obs_t, z_h_t, z_l_t, dv_curr_t)
            dist = Normal(mu, std)
            logp = dist.log_prob(act_t).sum(-1)
            ratio = torch.exp(logp - old_logp_t)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (v - ret_t).pow(2).mean()
            entropy = dist.entropy().sum(-1).mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()

        if ep % 10 == 0 or ep == 1:
            print(
                f"[{ep:4d}/{cfg.epochs}] "
                f"err={np.mean(final_err):7.3f} km  "
                f"|Δdv|={np.mean(final_delta):6.2f} m/s  "
                f"loss={float(loss.item()):.4f}"
            )

    torch.save(
        {
            "model": model.state_dict(),
            "obs_dim": env.obs_dim,
            "act_dim": env.act_dim,
            "net": {
                "d_z": cfg.d_z,
                "d_h": cfg.d_h,
                "n_heads": cfg.n_heads,
                "n_blocks": cfg.n_blocks,
            },
            "action_scale_kms": cfg.action_scale_kms,
            "horizon": cfg.horizon,
        },
        cfg.save_path,
    )
    print(f"Saved policy to {cfg.save_path}")


if __name__ == "__main__":
    main()
