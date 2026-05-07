"""Two-stage self-supervised TRC training for Lambert+J2.

Stage 1: Learn u0 (first head) to match Lambert dv1.
Stage 2: Learn iterative refinement to reduce terminal J2 position error.

This keeps the role split explicit:
- init_decoder learns Lambert-like initialization
- iterative updates learn J2 residual correction
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
for _p in (THIS_DIR, PARENT_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from lambert_trc_model import LambertDataset, LambertTRC
from trc import NetConfig, count_params


@dataclass
class RunConfig:
    quick: bool = False
    do_train: bool = True
    resume: bool = False
    data_dir: Path = THIS_DIR / "data"
    ckpt_dir: Path = THIS_DIR / "checkpoints"

    dv_max: float = 3.0
    lr_stage1: float = 1e-4
    lr_stage2: float = 1e-4

    pos_scale_m: float = 1000.0
    correction_scale: float = 0.005

    # Stage-1 (u0-only objective)
    lambda_u0_s1: float = 5.0
    lambda_pos_s1: float = 0.0
    lambda_ps_s1: float = 0.0

    # Stage-2 (refinement objective)
    lambda_u0_s2: float = 0.2
    lambda_pos_s2: float = 1.0
    lambda_ps_s2: float = 0.05

    # Keep False so model must use learned u0 instead of injected Lambert seed.
    anchor_to_lambert: bool = False

    @property
    def epochs_stage1(self) -> int:
        return 10 if self.quick else 80

    @property
    def epochs_stage2(self) -> int:
        return 15 if self.quick else 220

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
    def ckpt_stage1(self) -> Path:
        return self.ckpt_dir / "trc_lambert_u0_stage1_best.pt"

    @property
    def ckpt_stage2(self) -> Path:
        return self.ckpt_dir / "trc_lambert_u0_curriculum_best.pt"


class StageLoss(nn.Module):
    def __init__(self, lambda_u0: float, lambda_pos: float, lambda_ps: float):
        super().__init__()
        self.lambda_u0 = lambda_u0
        self.lambda_pos = lambda_pos
        self.lambda_ps = lambda_ps

    def forward(self, output, batch, model):
        pos_errors = output["pos_errors"]
        dv_iters = output["dv_iterations"]
        u0 = dv_iters[0]
        dv_final = output["dv_final"]

        L_u0 = F.mse_loss(u0, batch["dv1"]) / (model.dv_scale ** 2)

        pos_err_km = pos_errors[-1]
        pos_norm = getattr(model, "pos_scale", torch.tensor(100.0, device=pos_err_km.device))
        L_pos = (pos_err_km / pos_norm).pow(2).mean()

        if len(pos_errors) >= 2:
            err0 = pos_errors[0].detach().clamp(min=1e-3)
            normed = [e / err0 for e in pos_errors]
            improvements = [normed[k] - normed[k + 1] for k in range(len(normed) - 1)]
            L_proc = -torch.stack(improvements).mean()
        else:
            L_proc = torch.tensor(0.0, device=pos_err_km.device)

        loss = self.lambda_u0 * L_u0 + self.lambda_pos * L_pos + self.lambda_ps * L_proc

        with torch.no_grad():
            imp_metric = 0.0
            if len(pos_errors) >= 2:
                e0 = pos_errors[0].clamp(min=1e-3)
                for k in range(len(pos_errors) - 1):
                    imp_metric += ((pos_errors[k] - pos_errors[k + 1]) / e0).mean().item()
                imp_metric /= (len(pos_errors) - 1)

        return loss, {
            "loss": loss.item(),
            "L_u0": L_u0.item(),
            "L_pos": L_pos.item(),
            "L_proc": L_proc.item(),
            "imp_metric": imp_metric,
            "u0_res_norm": torch.norm(u0 - batch["dv1"], dim=-1).mean().item() * 1000.0,
            "dv_res_norm": torch.norm(dv_final - batch["dv1"], dim=-1).mean().item() * 1000.0,
            "pos_err_K": pos_errors[-1].mean().item(),
        }


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="Two-stage self-supervised TRC training.")
    p.add_argument("--quick", action="store_true", default=False)
    p.add_argument("--no_train", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stage1_only", action="store_true")
    p.add_argument("--stage2_only", action="store_true")
    p.add_argument("--data_dir", type=Path, default=THIS_DIR / "data")
    p.add_argument("--ckpt_dir", type=Path, default=THIS_DIR / "checkpoints")
    p.add_argument("--dv_max", type=float, default=3.0)
    p.add_argument("--lr_stage1", type=float, default=1e-4)
    p.add_argument("--lr_stage2", type=float, default=1e-4)
    p.add_argument("--pos_scale_m", type=float, default=1000.0)
    p.add_argument("--correction_scale", type=float, default=0.005)
    a = p.parse_args()

    cfg = RunConfig(
        quick=a.quick,
        do_train=not a.no_train,
        resume=a.resume,
        data_dir=a.data_dir,
        ckpt_dir=a.ckpt_dir,
        dv_max=a.dv_max,
        lr_stage1=a.lr_stage1,
        lr_stage2=a.lr_stage2,
        pos_scale_m=a.pos_scale_m,
        correction_scale=a.correction_scale,
    )
    cfg._stage1_only = bool(a.stage1_only)
    cfg._stage2_only = bool(a.stage2_only)
    return cfg


def evaluate(model, loader, criterion, device, anchor_to_lambert=False):
    model.eval()
    losses, posk, imp, u0res, dvres = [], [], [], [], []
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            dv_seed = b["dv1"] if anchor_to_lambert else None
            out = model(b["r0"], b["v0"], b["r_target"], b["v_target"], b["tof"], dv_lambert=dv_seed)
            _, m = criterion(out, b, model)
            losses.append(m["loss"])
            posk.append(m["pos_err_K"])
            imp.append(m["imp_metric"])
            u0res.append(m["u0_res_norm"])
            dvres.append(m["dv_res_norm"])
    return {
        "loss": float(np.mean(losses)),
        "pos_err_K": float(np.mean(posk)),
        "imp": float(np.mean(imp)),
        "u0_res_norm": float(np.mean(u0res)),
        "dv_res_norm": float(np.mean(dvres)),
    }


def save_ckpt(model, optimizer, scheduler, epoch, best_val, history, path, cfg, scales, stage_name):
    torch.save(
        {
            "epoch": epoch,
            "best_val": best_val,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "net_cfg": vars(cfg.net_cfg),
            "scales": scales,
            "cfg": vars(cfg),
            "stage": stage_name,
        },
        path,
    )


def train_stage(stage_name, model, train_loader, test_loader, device, cfg, criterion, optimizer, scheduler,
                epochs, ckpt_path, scales, anchor_to_lambert=False, resume=False):
    history = {"train_loss": [], "val_loss": [], "val_pos": [], "val_u0res_m": [], "val_dvres_m": []}
    start_epoch, best_val = 1, float("inf")

    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        history = ckpt.get("history", history)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", float("inf")))
        print(f"Resuming {stage_name} from epoch {start_epoch} (best={best_val:.4f})")

    for ep in range(start_epoch, start_epoch + epochs):
        t0 = time.time()
        model.train()
        ep_losses = []
        n_skipped = 0
        for b in train_loader:
            b = {k: v.to(device) for k, v in b.items()}
            optimizer.zero_grad()
            dv_seed = b["dv1"] if anchor_to_lambert else None
            out = model(b["r0"], b["v0"], b["r_target"], b["v_target"], b["tof"], dv_lambert=dv_seed)
            loss, _ = criterion(out, b, model)
            if not torch.isfinite(loss):
                n_skipped += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_losses.append(loss.item())

        scheduler.step()
        if len(ep_losses) == 0:
            raise RuntimeError(f"{stage_name}: all batches non-finite.")

        val = evaluate(model, test_loader, criterion, device, anchor_to_lambert=anchor_to_lambert)
        train_loss = float(np.mean(ep_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_pos"].append(val["pos_err_K"])
        history["val_u0res_m"].append(val["u0_res_norm"])
        history["val_dvres_m"].append(val["dv_res_norm"])

        dt = time.time() - t0
        print(
            f"[{stage_name} {ep:03d}] train={train_loss:.4f} val={val['loss']:.4f} "
            f"posK={val['pos_err_K']:.3f}km u0_res={val['u0_res_norm']:.2f}m/s "
            f"dv_res={val['dv_res_norm']:.2f}m/s imp={val['imp']:.3f} "
            f"skip={n_skipped} ({dt:.1f}s)"
        )

        if val["loss"] < best_val:
            best_val = val["loss"]
            save_ckpt(
                model,
                optimizer,
                scheduler,
                ep,
                best_val,
                history,
                ckpt_path,
                cfg,
                scales,
                stage_name,
            )
            print(f"  -> saved best: {ckpt_path}")



def main():
    cfg = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg.ckpt_dir.mkdir(exist_ok=True)
    if not cfg.train_path.exists() or not cfg.test_path.exists():
        raise FileNotFoundError("Missing lambert_train.npz/lambert_test.npz in data_dir.")

    print(f"Device: {device}")
    print(f"TRAIN={cfg.train_path}")
    print(f"TEST ={cfg.test_path}")
    print(f"CKPT1={cfg.ckpt_stage1}")
    print(f"CKPT2={cfg.ckpt_stage2}")
    print("MODE = two-stage curriculum (u0 pretrain -> refinement)")
    print(f"anchor_to_lambert={cfg.anchor_to_lambert}")

    train_ds = LambertDataset(cfg.train_path)
    test_ds = LambertDataset(cfg.test_path)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    r_scale, v_scale, tof_scale, dv_scale, _ = train_ds.get_scales()
    model = LambertTRC(
        cfg.net_cfg,
        n_coast_steps=cfg.n_coast,
        dv_max=cfg.dv_max,
        correction_scale=cfg.correction_scale,
    ).to(device)
    model.set_normalization(r_scale, v_scale, tof_scale, dv_scale, pos_scale_km=cfg.pos_scale_m / 1000.0)

    print(f"Params: {count_params(model):,}")
    print(f"Stage1 epochs={cfg.epochs_stage1}, Stage2 epochs={cfg.epochs_stage2}")

    scales = {
        "r": r_scale,
        "v": v_scale,
        "tof": tof_scale,
        "dv": dv_scale,
        "corr": cfg.correction_scale,
        "pos_m": cfg.pos_scale_m,
    }

    if not cfg.do_train:
        if cfg.ckpt_stage2.exists():
            ckpt = torch.load(cfg.ckpt_stage2, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
        elif cfg.ckpt_stage1.exists():
            ckpt = torch.load(cfg.ckpt_stage1, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            raise FileNotFoundError("No stage checkpoint found for --no_train.")
    else:
        run_stage1 = not getattr(cfg, "_stage2_only", False)
        run_stage2 = not getattr(cfg, "_stage1_only", False)

        # Stage 1: learn u0 ~= Lambert dv1
        if run_stage1:
            print("\n=== Stage 1: u0 pretraining (Lambert imitation) ===")
            crit1 = StageLoss(
                lambda_u0=cfg.lambda_u0_s1,
                lambda_pos=cfg.lambda_pos_s1,
                lambda_ps=cfg.lambda_ps_s1,
            )
            opt1 = torch.optim.AdamW(model.parameters(), lr=cfg.lr_stage1)
            sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=cfg.epochs_stage1, eta_min=1e-5)
            train_stage(
                "S1",
                model,
                train_loader,
                test_loader,
                device,
                cfg,
                crit1,
                opt1,
                sch1,
                cfg.epochs_stage1,
                cfg.ckpt_stage1,
                scales,
                anchor_to_lambert=False,
                resume=cfg.resume,
            )

        # Stage 2: learn refinement on top of learned u0
        if run_stage2:
            print("\n=== Stage 2: refinement training (terminal J2 error) ===")
            if (not run_stage1) and cfg.ckpt_stage1.exists():
                ckpt1 = torch.load(cfg.ckpt_stage1, map_location=device, weights_only=False)
                model.load_state_dict(ckpt1["model_state_dict"])
                print(f"Loaded stage1 init: {cfg.ckpt_stage1}")

            crit2 = StageLoss(
                lambda_u0=cfg.lambda_u0_s2,
                lambda_pos=cfg.lambda_pos_s2,
                lambda_ps=cfg.lambda_ps_s2,
            )
            opt2 = torch.optim.AdamW(model.parameters(), lr=cfg.lr_stage2)
            sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=cfg.epochs_stage2, eta_min=1e-5)
            train_stage(
                "S2",
                model,
                train_loader,
                test_loader,
                device,
                cfg,
                crit2,
                opt2,
                sch2,
                cfg.epochs_stage2,
                cfg.ckpt_stage2,
                scales,
                anchor_to_lambert=False,
                resume=cfg.resume,
            )

    final_crit = StageLoss(
        lambda_u0=cfg.lambda_u0_s2,
        lambda_pos=cfg.lambda_pos_s2,
        lambda_ps=cfg.lambda_ps_s2,
    )
    final_val = evaluate(model, test_loader, final_crit, device, anchor_to_lambert=False)
    print("\nFinal evaluation (test):")
    print(
        f"loss={final_val['loss']:.4f}, pos_err_K={final_val['pos_err_K']:.3f} km, "
        f"u0_res={final_val['u0_res_norm']:.2f} m/s, dv_res={final_val['dv_res_norm']:.2f} m/s, "
        f"imp={final_val['imp']:.3f}"
    )


if __name__ == "__main__":
    main()
