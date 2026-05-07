# LambertPy_RL

Standalone non-supervised RL baseline for Lambert correction under J2 dynamics.

## Included

- `rl_env.py`: batch environment over `.npz` transfer samples
- `trc_core.py`: TRC-style building blocks (`NetConfig`, `ReasoningModule`, MLP)
- `rl_policy.py`: recurrent TRC actor-critic with latent memories (`z_H`, `z_L`)
- `train_ppo.py`: PPO training loop (no shooting labels required)
- `evaluate_rl.py`: compare RL policy against Lambert-only on J2 arrival error
- Copied physics utils: `constants.py`, `dynamics.py`, `lambert_solver.py`, `orbital_utils.py`

## Train

From repo root:

```bash
python3 LambertPy_RL/train_ppo.py --dataset_path data/lambert_train.npz
```

Quick run:

```bash
python3 LambertPy_RL/train_ppo.py --dataset_path data_quick/lambert_train.npz --epochs 20 --steps_per_epoch 8
```

## Evaluate

```bash
python3 LambertPy_RL/evaluate_rl.py --dataset_path data/lambert_test.npz --ckpt LambertPy_RL/ppo_lambert.pt
```

## Notes

- This is intentionally separate from supervised TRC code.
- Policy is TRC-like recurrent reasoning, not a plain MLP baseline.
- Reward is based on reduction in terminal-position error plus action penalty.
- Start with smaller `n_prop_steps` for speed; increase for better fidelity.
