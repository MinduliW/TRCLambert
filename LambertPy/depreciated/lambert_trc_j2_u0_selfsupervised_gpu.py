#!/usr/bin/env python3
"""GPU-friendly runner for Lambert TRC u0-self-supervised training/evaluation.

Built as a robust CLI alternative to lambert_trc_j2_u0_selfsupervised.py.
"""

import argparse
import contextlib
import sys
import time
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
from lambert_trc_j2_u0_selfsupervised import (
    U0SelfSupervisedJ2Loss,
    plot_training_curves,
    collect_eval_samples,
    plot_eval_summary,
    plot_arrival_error_comparison,
)
from trc import NetConfig, count_params


@dataclass
class RunCfg:
    quick: bool = False
    data_dir: Path = THIS_DIR / "data"
    ckpt_dir: Path = THIS_DIR / "checkpoints"
    lr: float = 5e-5
    dv_max: float = 3.0
    pos_scale_m: float = 1000.0
    correction_scale: float = 0.01
    lambda_u0: float = 0.1
    lambda_pos: float = 10.0
    lambda_ps: float = 0.05
    anchor_to_lambert: bool = True
    device: str = "auto"
    num_workers: int = 8
    amp: bool = False
    amp_dtype: str = "bf16"
    compile_model: bool = False
    epochs: int | None = None
    batch_size: int | None = None
    no_train: bool = False
    resume: bool = False
    resume_path: Path | None = None
    no_warmup: bool = False
    log_every: int = 50
    plots: bool = True
    d_z: int | None = None
    d_h: int | None = None
    model_size: str = "full"  # small|medium|full

    @property
    def train_path(self) -> Path:
        return self.data_dir / "lambert_train.npz"

    @property
    def test_path(self) -> Path:
        return self.data_dir / "lambert_test.npz"

    @property
    def ckpt_path(self) -> Path:
        return self.ckpt_dir / "trc_lambert_u0_selfsup_best.pt"

    def resolved_epochs(self) -> int:
        if self.epochs is not None:
            return int(self.epochs)
        return 15 if self.quick else 200

    def resolved_batch_size(self) -> int:
        if self.batch_size is not None:
            return int(self.batch_size)
        return 32 if self.quick else 64

    def resolved_n_coast(self) -> int:
        return 100 if self.quick else 200

    def resolved_net_cfg(self) -> NetConfig:
        presets = {
            "small": NetConfig(d_z=64, d_h=128, n_heads=4, n_blocks=1, K=3, n_inner=4),
            "medium": NetConfig(d_z=128, d_h=256, n_heads=4, n_blocks=2, K=3, n_inner=6),
            "full": NetConfig(d_z=256, d_h=512, n_heads=8, n_blocks=3, K=3, n_inner=6),
        }
        size_key = self.model_size.lower()
        if size_key not in presets:
            raise ValueError(f"Invalid model_size={self.model_size!r}; use small|medium|full.")
        base = presets[size_key]
        if self.d_z is not None:
            base.d_z = int(self.d_z)
        if self.d_h is not None:
            base.d_h = int(self.d_h)
        return base


def _pick_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is unavailable.")
        return torch.device("cuda")
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("Requested --device mps but MPS is unavailable.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_device_batch(batch, device, non_blocking=False):
    return {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}


def _unwrap_model(model):
    return getattr(model, "_orig_mod", model)


def _normalize_state_dict_keys(state_dict):
    if not any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return state_dict
    return {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()
    }


def evaluate(model, loader, criterion, device, anchor_to_lambert=True, amp_enabled=False, amp_dtype=torch.bfloat16):
    model.eval()
    losses, posk, imp, u0res, dvres = [], [], [], [], []
    if device.type == "cuda":
        amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled)
    else:
        amp_ctx = contextlib.nullcontext()

    with torch.no_grad():
        for b in loader:
            b = _to_device_batch(b, device, non_blocking=(device.type == "cuda"))
            with amp_ctx:
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


def save_ckpt(model, optimizer, scheduler, epoch, best_val, history, path, net_cfg: NetConfig, cfg: RunCfg, scales):
    model_to_save = _unwrap_model(model)
    torch.save(
        {
            "epoch": epoch,
            "best_val": best_val,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "net_cfg": vars(net_cfg),
            "scales": scales,
            "cfg": vars(cfg),
        },
        path,
    )


def run_warmup_batch(model, loader, criterion, optimizer, device, use_cuda_amp, amp_dtype, anchor_to_lambert):
    model.train()
    t0 = time.time()
    b = next(iter(loader))
    b = _to_device_batch(b, device, non_blocking=(device.type == "cuda"))
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda_amp)
    else:
        amp_ctx = contextlib.nullcontext()

    with amp_ctx:
        dv_seed = b["dv1"] if anchor_to_lambert else None
        out = model(b["r0"], b["v0"], b["r_target"], b["v_target"], b["tof"], dv_lambert=dv_seed)
        loss, _ = criterion(out, b, model)

    # Keep warmup isolated from GradScaler state.
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.time() - t0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GPU-friendly Lambert TRC u0-self-supervised runner")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--data-dir", type=Path, default=THIS_DIR / "data")
    p.add_argument("--ckpt-dir", type=Path, default=THIS_DIR / "checkpoints")
    p.add_argument("--epochs", type=int, default=None, help="Target absolute epoch count.")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--dv-max", type=float, default=3.0)
    p.add_argument("--pos-scale-m", type=float, default=1000.0)
    p.add_argument("--correction-scale", type=float, default=0.01)
    p.add_argument("--lambda-u0", type=float, default=0.1)
    p.add_argument("--lambda-pos", type=float, default=10.0)
    p.add_argument("--lambda-ps", type=float, default=0.05)
    p.add_argument("--no-anchor-to-lambert", action="store_true")
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--num-workers", type=int, default=8)

    p.add_argument("--amp", action="store_true", help="Enable AMP on CUDA (FP32 is default).")
    p.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--compile-model", action="store_true")

    p.add_argument("--no-train", action="store_true", help="Evaluation-only mode.")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint.")
    p.add_argument("--resume-path", type=Path, default=None)

    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--model-size", choices=["small", "medium", "full"], default="full",
                   help="Model preset for network width/depth.")
    p.add_argument("--d-z", type=int, default=None, help="Override latent dimension d_z.")
    p.add_argument("--d-h", type=int, default=None, help="Override hidden dimension d_h.")
    return p


def args_to_cfg(args: argparse.Namespace) -> RunCfg:
    return RunCfg(
        quick=args.quick,
        data_dir=args.data_dir,
        ckpt_dir=args.ckpt_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dv_max=args.dv_max,
        pos_scale_m=args.pos_scale_m,
        correction_scale=args.correction_scale,
        lambda_u0=args.lambda_u0,
        lambda_pos=args.lambda_pos,
        lambda_ps=args.lambda_ps,
        anchor_to_lambert=(not args.no_anchor_to_lambert),
        device=args.device,
        num_workers=args.num_workers,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        compile_model=args.compile_model,
        no_train=args.no_train,
        resume=args.resume,
        resume_path=args.resume_path,
        no_warmup=args.no_warmup,
        log_every=args.log_every,
        plots=(not args.no_plots),
        model_size=args.model_size,
        d_z=args.d_z,
        d_h=args.d_h,
    )


def main():
    args = build_arg_parser().parse_args()
    cfg = args_to_cfg(args)

    torch.manual_seed(0)
    np.random.seed(0)

    device = _pick_device(cfg.device)
    use_cuda_amp = cfg.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    use_grad_scaler = use_cuda_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    cfg.ckpt_dir.mkdir(exist_ok=True)
    if not cfg.train_path.exists() or not cfg.test_path.exists():
        raise FileNotFoundError("Missing lambert_train.npz/lambert_test.npz in data_dir.")

    print(f"Device: {device}")
    print(f"TRAIN={cfg.train_path}")
    print(f"TEST ={cfg.test_path}")
    print(f"CKPT ={cfg.ckpt_path}")
    print("MODE = u0-self-supervised (no dv1_corrected required)")
    print(
        f"amp={use_cuda_amp} (dtype={cfg.amp_dtype}), compile_model={cfg.compile_model}, "
        f"num_workers={cfg.num_workers}, resume={cfg.resume}, no_train={cfg.no_train}"
    )

    train_ds = LambertDataset(cfg.train_path)
    test_ds = LambertDataset(cfg.test_path)

    use_workers = max(0, int(cfg.num_workers))
    pin_mem = device.type == "cuda"
    batch_size = cfg.resolved_batch_size()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=use_workers,
        pin_memory=pin_mem,
        persistent_workers=(use_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=use_workers,
        pin_memory=pin_mem,
        persistent_workers=(use_workers > 0),
    )

    r_scale, v_scale, tof_scale, dv_scale, _ = train_ds.get_scales()

    ckpt = None
    resume_path = cfg.resume_path if cfg.resume_path is not None else cfg.ckpt_path
    model_net_cfg = cfg.resolved_net_cfg()
    if (cfg.resume or cfg.no_train) and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if "net_cfg" in ckpt:
            model_net_cfg = NetConfig(**ckpt["net_cfg"])
        if "scales" in ckpt:
            sc = ckpt["scales"]
            r_scale = float(sc.get("r", r_scale))
            v_scale = float(sc.get("v", v_scale))
            tof_scale = float(sc.get("tof", tof_scale))
            dv_scale = float(sc.get("dv", dv_scale))

    model = LambertTRC(
        model_net_cfg,
        n_coast_steps=cfg.resolved_n_coast(),
        dv_max=cfg.dv_max,
        correction_scale=cfg.correction_scale,
    ).to(device)
    model.set_normalization(r_scale, v_scale, tof_scale, dv_scale, pos_scale_km=cfg.pos_scale_m / 1000.0)

    criterion = U0SelfSupervisedJ2Loss(
        lambda_u0=cfg.lambda_u0,
        lambda_pos=cfg.lambda_pos,
        lambda_ps=cfg.lambda_ps,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    target_epochs = cfg.resolved_epochs()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, target_epochs), eta_min=1e-5)

    print(f"Params: {count_params(model):,}")
    print(
        f"Net: d_z={model_net_cfg.d_z}, d_h={model_net_cfg.d_h}, "
        f"heads={model_net_cfg.n_heads}, blocks={model_net_cfg.n_blocks}, "
        f"K={model_net_cfg.K}, n_inner={model_net_cfg.n_inner}"
    )
    print(
        f"Loss weights: lambda_u0={cfg.lambda_u0}, lambda_pos={cfg.lambda_pos}, lambda_ps={cfg.lambda_ps}"
    )
    print(f"correction_scale={cfg.correction_scale:.4f} km/s")
    print(f"anchor_to_lambert={cfg.anchor_to_lambert}")
    if use_cuda_amp:
        print(f"AMP dtype={cfg.amp_dtype}")
    else:
        print("Precision mode: FP32")

    history = {"train_loss": [], "val_loss": [], "val_pos": [], "val_u0res_m": [], "val_dvres_m": []}
    start_epoch, best_val = 1, float("inf")

    if ckpt is not None and (cfg.resume or cfg.no_train):
        model.load_state_dict(_normalize_state_dict_keys(ckpt["model_state_dict"]))
        if cfg.resume and "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                scheduler.load_state_dict(ckpt.get("scheduler_state_dict", scheduler.state_dict()))
            except Exception as exc:
                print(f"Warning: failed to load optimizer/scheduler state ({exc}); continuing with fresh optimizer state.")
        history = ckpt.get("history", history)
        best_val = float(ckpt.get("best_val", best_val))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Loaded checkpoint: epoch={start_epoch - 1} best_val={best_val:.4f}")
    elif cfg.no_train:
        raise FileNotFoundError(f"Missing checkpoint for --no-train: {resume_path}")

    if cfg.compile_model:
        if hasattr(torch, "compile"):
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile enabled (reduce-overhead)")
        else:
            print("WARNING: torch.compile not available; continuing without compile.")

    num_train_batches = len(train_loader)
    if (not cfg.no_train) and (start_epoch <= target_epochs) and (not cfg.no_warmup):
        print("Running warmup batch (compile/startup)...")
        warmup_dt = run_warmup_batch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_cuda_amp,
            amp_dtype,
            cfg.anchor_to_lambert,
        )
        print(f"Warmup done in {warmup_dt:.1f}s")

    if cfg.no_train:
        print("Training disabled: evaluation-only mode.")
    elif start_epoch > target_epochs:
        print(f"No training needed: checkpoint at epoch {start_epoch - 1}, target epochs={target_epochs}.")

    if not cfg.no_train:
        for ep in range(start_epoch, target_epochs + 1):
            t0 = time.time()
            model.train()
            ep_losses = []
            n_skipped = 0

            for step, b in enumerate(train_loader, start=1):
                b = _to_device_batch(b, device, non_blocking=pin_mem)
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda_amp)
                else:
                    amp_ctx = contextlib.nullcontext()
                with amp_ctx:
                    dv_seed = b["dv1"] if cfg.anchor_to_lambert else None
                    out = model(b["r0"], b["v0"], b["r_target"], b["v_target"], b["tof"], dv_lambert=dv_seed)
                    loss, m = criterion(out, b, model)

                if not torch.isfinite(loss):
                    n_skipped += 1
                    continue

                if use_grad_scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                ep_losses.append(loss.item())

                if cfg.log_every > 0 and (step == 1 or step % cfg.log_every == 0 or step == num_train_batches):
                    print(
                        f"  epoch {ep:3d} step {step:4d}/{num_train_batches} "
                        f"loss={m['loss']:.4f} posK={m['pos_err_K']:.3f}km imp={m['imp_metric']:.3f}"
                    )

            scheduler.step()
            if len(ep_losses) == 0:
                raise RuntimeError("All training batches were non-finite; reduce correction_scale or keep anchoring.")

            val = evaluate(
                model,
                test_loader,
                criterion,
                device,
                anchor_to_lambert=cfg.anchor_to_lambert,
                amp_enabled=use_cuda_amp,
                amp_dtype=amp_dtype,
            )
            train_loss = float(np.mean(ep_losses))
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val["loss"])
            history["val_pos"].append(val["pos_err_K"])
            history["val_u0res_m"].append(val["u0_res_norm"])
            history["val_dvres_m"].append(val["dv_res_norm"])

            dt = time.time() - t0
            print(
                f"[{ep:03d}/{target_epochs}] train={train_loss:.4f} val={val['loss']:.4f} "
                f"posK={val['pos_err_K']:.3f}km u0_res={val['u0_res_norm']:.2f}m/s "
                f"dv_res={val['dv_res_norm']:.2f}m/s imp={val['imp']:.3f} skip={n_skipped} ({dt:.1f}s)"
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
                    cfg.ckpt_path,
                    model_net_cfg,
                    cfg,
                    {
                        "r": r_scale,
                        "v": v_scale,
                        "tof": tof_scale,
                        "dv": dv_scale,
                        "corr": cfg.correction_scale,
                        "pos_m": cfg.pos_scale_m,
                    },
                )
                print(f"  -> saved best: {cfg.ckpt_path}")

    # Evaluate best checkpoint if available; otherwise current model.
    if cfg.ckpt_path.exists():
        best_ck = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(_normalize_state_dict_keys(best_ck["model_state_dict"]))
        print(
            f"Loaded best checkpoint for eval: epoch={best_ck.get('epoch', 'NA')} "
            f"best_val={best_ck.get('best_val', float('nan')):.4f}"
        )

    final_val = evaluate(
        model,
        test_loader,
        criterion,
        device,
        anchor_to_lambert=cfg.anchor_to_lambert,
        amp_enabled=use_cuda_amp,
        amp_dtype=amp_dtype,
    )
    print("\nFinal evaluation (test):")
    print(
        f"loss={final_val['loss']:.4f}, pos_err_K={final_val['pos_err_K']:.3f} km, "
        f"u0_res={final_val['u0_res_norm']:.2f} m/s, dv_res={final_val['dv_res_norm']:.2f} m/s, "
        f"imp={final_val['imp']:.3f}"
    )

    if cfg.plots:
        plot_training_curves(history, cfg.ckpt_dir / "lambert_u0_selfsup_training_curves.png")
        train_eval = collect_eval_samples(
            model,
            train_loader,
            device,
            anchor_to_lambert=cfg.anchor_to_lambert,
            amp_enabled=use_cuda_amp,
            amp_dtype=amp_dtype,
        )
        test_eval = collect_eval_samples(
            model,
            test_loader,
            device,
            anchor_to_lambert=cfg.anchor_to_lambert,
            amp_enabled=use_cuda_amp,
            amp_dtype=amp_dtype,
        )
        plot_eval_summary(test_eval, cfg.ckpt_dir / "lambert_u0_selfsup_eval_summary_test.png", split_name="test")
        plot_arrival_error_comparison(
            train_eval,
            test_eval,
            cfg.ckpt_dir / "lambert_u0_selfsup_vs_lambert_arrival_error.png",
        )


if __name__ == "__main__":
    main()
