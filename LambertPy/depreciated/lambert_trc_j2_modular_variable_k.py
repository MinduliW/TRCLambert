"""Lambert TRC runner with variable-K support (fixed/random training, adaptive inference).

Uses:
- `variable_k.patch_model` to enable variable iteration depth
- `VariableKLoss` and `train_variable_k` for training
- `inference_adaptive` for adaptive-K evaluation
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
for _p in (THIS_DIR, PARENT_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from lambert_trc_model import LambertDataset, LambertTRC
from trc import NetConfig, count_params
from variable_k import VariableKLoss, inference_adaptive, patch_model, train_variable_k


@dataclass
class RunConfig:
    quick: bool = True
    do_train: bool = True
    resume: bool = False
    data_dir: Path = THIS_DIR / "data"
    ckpt_dir: Path = THIS_DIR / "checkpoints"
    lr: float = 1e-4
    dv_max: float = 3.0
    pos_scale_m: float = 1000.0
    correction_scale: float = 0.005
    lambda_u0: float = 5.0
    lambda_pos: float = 1.0
    lambda_ps: float = 0.05
    K_mode: str = "adaptive"      # fixed|random
    K_min: int = 2
    K_max: int = 8
    K_tol_km: float = 0.1
    anchor_to_lambert: bool = True
    eval_batch_size: int = 256

    @property
    def epochs(self) -> int:
        return 15 if self.quick else 60

    @property
    def batch_size(self) -> int:
        return 32 if self.quick else 64

    @property
    def n_coast(self) -> int:
        return 100 if self.quick else 200

    @property
    def net_cfg(self) -> NetConfig:
        if self.quick:
            return NetConfig(d_z=128, d_h=256, n_heads=4, n_blocks=2, K=3, n_inner=6)
        return NetConfig(d_z=256, d_h=512, n_heads=8, n_blocks=3, K=3, n_inner=6)

    @property
    def train_path(self) -> Path:
        return self.data_dir / "lambert_train.npz"

    @property
    def test_path(self) -> Path:
        return self.data_dir / "lambert_test.npz"

    @property
    def ckpt_path(self) -> Path:
        return self.ckpt_dir / "trc_lambert_variable_k_best.pt"


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="Variable-K Lambert TRC runner.")
    p.add_argument("--quick", action="store_true", default=True)
    p.add_argument("--full", action="store_true")
    p.add_argument("--no_train", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--data_dir", type=Path, default=THIS_DIR / "data")
    p.add_argument("--ckpt_dir", type=Path, default=THIS_DIR / "checkpoints")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dv_max", type=float, default=3.0)
    p.add_argument("--pos_scale_m", type=float, default=1000.0)
    p.add_argument("--correction_scale", type=float, default=0.005)
    p.add_argument("--lambda_u0", type=float, default=5.0)
    p.add_argument("--lambda_pos", type=float, default=1.0)
    p.add_argument("--lambda_ps", type=float, default=0.05)
    p.add_argument("--K_mode", type=str, default="random", choices=["fixed", "random", "adaptive"])
    p.add_argument("--K_min", type=int, default=2)
    p.add_argument("--K_max", type=int, default=8)
    p.add_argument("--K_tol_km", type=float, default=0.1)
    p.add_argument("--no_anchor_to_lambert", action="store_true")
    p.add_argument("--eval_batch_size", type=int, default=256)
    a = p.parse_args()

    return RunConfig(
        quick=a.quick and not a.full,
        do_train=not a.no_train,
        resume=a.resume,
        data_dir=a.data_dir,
        ckpt_dir=a.ckpt_dir,
        lr=a.lr,
        dv_max=a.dv_max,
        pos_scale_m=a.pos_scale_m,
        correction_scale=a.correction_scale,
        lambda_u0=a.lambda_u0,
        lambda_pos=a.lambda_pos,
        lambda_ps=a.lambda_ps,
        K_mode=a.K_mode,
        K_min=a.K_min,
        K_max=a.K_max,
        K_tol_km=a.K_tol_km,
        anchor_to_lambert=not a.no_anchor_to_lambert,
        eval_batch_size=a.eval_batch_size,
    )


def build_model(cfg: RunConfig, train_ds: LambertDataset, device: torch.device):
    r_scale, v_scale, tof_scale, dv_scale, _ = train_ds.get_scales()
    model = LambertTRC(
        cfg.net_cfg,
        n_coast_steps=cfg.n_coast,
        dv_max=cfg.dv_max,
        correction_scale=cfg.correction_scale,
    ).to(device)
    model.set_normalization(r_scale, v_scale, tof_scale, dv_scale, pos_scale_km=cfg.pos_scale_m / 1000.0)
    patch_model(model)
    return model


@torch.no_grad()
def evaluate_adaptive_dataset(model, ds, device, cfg: RunConfig, split_name: str):
    model.eval()
    loader = DataLoader(ds, batch_size=cfg.eval_batch_size, shuffle=False)

    pe_all = []
    k_all = []
    dv_res_all = []
    for i, b in enumerate(loader, start=1):
        b = {k: v.to(device) for k, v in b.items()}
        dv_seed = b["dv1"] if cfg.anchor_to_lambert else None
        out = inference_adaptive(
            model,
            b["r0"],
            b["v0"],
            b["r_target"],
            b["v_target"],
            b["tof"],
            dv_lambert=dv_seed,
            K_min=cfg.K_min,
            K_max=cfg.K_max,
            tol_km=cfg.K_tol_km,
        )
        pe = out["pos_errors"][-1].detach().cpu().numpy()
        dv_res = torch.norm(out["dv_final"] - b["dv1"], dim=-1).detach().cpu().numpy() * 1000.0
        k_sample = out["K_per_sample"].detach().cpu().numpy()

        pe_all.append(pe)
        dv_res_all.append(dv_res)
        k_all.append(k_sample)
        print(f"  [{split_name}] batch {i}/{len(loader)}")

    pe_all = np.concatenate(pe_all, axis=0)
    dv_res_all = np.concatenate(dv_res_all, axis=0)
    k_all = np.concatenate(k_all, axis=0)
    print(
        f"{split_name}: pos_err={pe_all.mean():.3f} ± {pe_all.std():.3f} km, "
        f"|Δdv|={dv_res_all.mean():.2f} ± {dv_res_all.std():.2f} m/s, "
        f"K_used={k_all.mean():.2f} ± {k_all.std():.2f}"
    )


def main():
    cfg = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg.ckpt_dir.mkdir(exist_ok=True)
    if not cfg.train_path.exists() or not cfg.test_path.exists():
        raise FileNotFoundError("Missing dataset files in data_dir.")

    print(f"Device: {device}")
    print(f"TRAIN={cfg.train_path}")
    print(f"TEST ={cfg.test_path}")
    print(f"CKPT ={cfg.ckpt_path}")
    print(
        f"mode={cfg.K_mode}, K_range=[{cfg.K_min}, {cfg.K_max}], "
        f"tol={cfg.K_tol_km} km, anchor_to_lambert={cfg.anchor_to_lambert}"
    )

    train_ds = LambertDataset(cfg.train_path)
    test_ds = LambertDataset(cfg.test_path)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    model = build_model(cfg, train_ds, device)
    print(f"Params: {count_params(model):,}")
    criterion = VariableKLoss(
        lambda_u0=cfg.lambda_u0,
        lambda_pos=cfg.lambda_pos,
        lambda_ps=cfg.lambda_ps,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-5)

    if cfg.resume and cfg.ckpt_path.exists():
        ckpt = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Loaded checkpoint epoch={ckpt.get('epoch', 'NA')} best_val={ckpt.get('best_val', float('nan')):.4f}")

    if cfg.do_train:
        train_variable_k(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=cfg.epochs,
            K_min=cfg.K_min,
            K_max=cfg.K_max,
            K_mode=cfg.K_mode,
            anchor_to_lambert=cfg.anchor_to_lambert,
            scheduler=scheduler,
            ckpt_path=cfg.ckpt_path,
            print_every=1,
        )

    if cfg.ckpt_path.exists():
        ckpt = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    print("\nAdaptive-K evaluation:")
    evaluate_adaptive_dataset(model, train_ds, device, cfg, split_name="train")
    evaluate_adaptive_dataset(model, test_ds, device, cfg, split_name="test")
    print("\nDone.")


if __name__ == "__main__":
    main()
