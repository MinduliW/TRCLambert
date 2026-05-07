"""Batch RL environment for J2 Lambert correction."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dynamics import propagate_j2


@dataclass
class RLEnvConfig:
    dataset_path: str = "data/lambert_train.npz"
    batch_size: int = 32
    horizon: int = 3
    n_prop_steps: int = 300
    action_scale_kms: float = 0.02
    lambda_action: float = 0.05


class LambertBatchEnv:
    """Finite-horizon batch environment over fixed transfer samples."""

    def __init__(self, cfg: RLEnvConfig, seed: int = 42):
        self.cfg = cfg
        self.rng = np.random.RandomState(seed)
        p = Path(cfg.dataset_path)
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset: {p}")
        d = np.load(p)
        self.r0 = d["r0"].astype(np.float64)
        self.v0 = d["v0"].astype(np.float64)
        self.rt = d["r_target"].astype(np.float64)
        self.vt = d["v_target"].astype(np.float64)
        self.tof = d["tof"].astype(np.float64).reshape(-1)
        self.dv_lambert = d["dv1"].astype(np.float64)
        self.n = len(self.r0)
        if self.n == 0:
            raise ValueError("Dataset has zero samples")

        self.r_scale = float(np.mean(np.linalg.norm(self.r0, axis=1)))
        self.v_scale = float(np.mean(np.linalg.norm(self.v0, axis=1)))
        self.tof_scale = float(np.mean(self.tof))
        self.dv_scale = float(np.mean(np.linalg.norm(self.dv_lambert, axis=1)))
        self.pos_scale = max(self.r_scale * 0.05, 50.0)

        self._idx = None
        self._step = 0
        self._dv = None
        self._err = None
        self._err_norm = None

    @property
    def obs_dim(self) -> int:
        return 19

    @property
    def act_dim(self) -> int:
        return 3

    def _propagate_error(self, dv):
        b = dv.shape[0]
        err = np.zeros((b, 3), dtype=np.float64)
        for i in range(b):
            j = self._idx[i]
            rf, _, _ = propagate_j2(
                self.r0[j],
                self.v0[j] + dv[i],
                float(self.tof[j]),
                n_steps=self.cfg.n_prop_steps,
            )
            err[i] = rf - self.rt[j]
        err_norm = np.linalg.norm(err, axis=1)
        return err, err_norm

    def _build_obs(self):
        j = self._idx
        return np.concatenate(
            [
                self.r0[j] / self.r_scale,
                self.v0[j] / self.v_scale,
                self.rt[j] / self.r_scale,
                self.vt[j] / self.v_scale,
                (self.tof[j] / self.tof_scale)[:, None],
                self._dv / self.dv_scale,
                self._err / self.pos_scale,
            ],
            axis=1,
        ).astype(np.float32)

    def reset(self):
        self._idx = self.rng.randint(0, self.n, size=self.cfg.batch_size)
        self._step = 0
        self._dv = self.dv_lambert[self._idx].copy()
        self._err, self._err_norm = self._propagate_error(self._dv)
        return self._build_obs()

    def step(self, action):
        """Action is unconstrained; we squash with tanh and scale."""
        delta = np.tanh(action) * self.cfg.action_scale_kms
        prev_norm = self._err_norm.copy()
        self._dv = self._dv + delta
        self._err, self._err_norm = self._propagate_error(self._dv)
        self._step += 1

        improve = (prev_norm - self._err_norm) / self.pos_scale
        action_pen = self.cfg.lambda_action * np.sum((delta / self.cfg.action_scale_kms) ** 2, axis=1)
        reward = improve - action_pen

        done = self._step >= self.cfg.horizon
        if done:
            reward = reward - (self._err_norm / self.pos_scale)

        info = {
            "pos_err_km": self._err_norm.copy(),
            "dv_delta_ms": np.linalg.norm(self._dv - self.dv_lambert[self._idx], axis=1) * 1000.0,
        }
        return self._build_obs(), reward.astype(np.float32), done, info
