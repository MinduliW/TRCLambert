# """Train/evaluate LambertTRC without J2-corrected dV labels.

# Loss design:
# - Make initial control u0 match Lambert dv1.
# - Let iterative refinement reduce terminal J2 position error.
# """

# import sys
# import time
# from dataclasses import dataclass
# from pathlib import Path
# from contextlib import nullcontext

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader

# THIS_DIR = Path(__file__).resolve().parent
# PARENT_DIR = THIS_DIR.parent
# for _p in (THIS_DIR, PARENT_DIR):
#     _ps = str(_p)
#     if _ps not in sys.path:
#         sys.path.insert(0, _ps)

# from lambert_trc_model import LambertDataset, LambertTRC
# from trc import NetConfig, count_params
# import lambert_plotters as plotters



# @dataclass
# class RunConfig:
#     quick: bool = False
#     do_train: bool = True
#     resume: bool = False
#     data_dir: Path = THIS_DIR / "data"
#     ckpt_dir: Path = THIS_DIR / "checkpoints"
#     dv_max: float = 5.0
#     lr: float = 5e-4
#     pos_scale_m: float = 1000.0
#     correction_scale: float = 0.01  # km/s, used for iterative updates inside TRC
#     lambda_u0: float = 0.1
#     lambda_pos: float = 1.0
#     lambda_ps: float = 0.05
#     anchor_to_lambert: bool = True
#     skip_u0_learning: bool = True
#     device: str = "auto"
#     num_workers: int = 8
#     amp: bool = True
#     amp_dtype: str = "bf16"  # bf16|fp16
#     compile_model: bool = False

#     @property
#     def epochs(self) -> int:
#         return 15 if self.quick else 300

#     @property
#     def batch_size(self) -> int:
#         return 32 if self.quick else 64

#     @property
#     def n_coast(self) -> int:
#         return 100 if self.quick else 200

#     @property
#     def net_cfg(self) -> NetConfig:
#         if self.quick:
#             return NetConfig(d_z=128, d_h=256, n_heads=4, n_blocks=2, K=4, n_inner=6)
#         return NetConfig(d_z=256, d_h=512, n_heads=8, n_blocks=3, K=4, n_inner=6)

#     @property
#     def train_path(self) -> Path:
#         return self.data_dir / "lambert_train.npz"

#     @property
#     def test_path(self) -> Path:
#         return self.data_dir / "lambert_test.npz"

#     @property
#     def ckpt_path(self) -> Path:
#         return self.ckpt_dir / "trc_lambert_u0_selfsup_best.pt"


# class U0SelfSupervisedJ2Loss(nn.Module):
#     """No-label TRC loss: u0 Lambert match + iterative J2 terminal improvement."""

#     def __init__(self, lambda_u0=1.0, lambda_pos=1.0, lambda_ps=0.05, use_u0_loss=True):
#         super().__init__()
#         self.lambda_u0 = lambda_u0
#         self.lambda_pos = lambda_pos
#         self.lambda_ps = lambda_ps
#         self.use_u0_loss = use_u0_loss

#     def forward(self, output, batch, model):
#         pos_errors = output["pos_errors"]
#         dv_iters = output["dv_iterations"]
#         u0 = dv_iters[0]
#         dv_final = output["dv_final"]

#         # 1) Initial policy should reproduce Lambert burn.
#         if self.use_u0_loss:
#             L_u0 = F.mse_loss(u0, batch["dv1"]) / (model.dv_scale ** 2)
#         else:
#             L_u0 = torch.tensor(0.0, device=u0.device)

#         # 2) Final terminal J2 position error should be small.
#         pos_err_km = pos_errors[-1]
#         pos_norm = getattr(model, "pos_scale", torch.tensor(100.0, device=pos_err_km.device))
#         L_pos = (pos_err_km / pos_norm).pow(2).mean()

#         # 3) Encourage monotonic refinement across TRC iterations.
#         if len(pos_errors) >= 2:
#             err0 = pos_errors[0].detach().clamp(min=1e-3)
#             normed = [e / err0 for e in pos_errors]
#             improvements = [normed[k] - normed[k + 1] for k in range(len(normed) - 1)]
#             L_proc = -torch.stack(improvements).mean()
#         else:
#             L_proc = torch.tensor(0.0, device=pos_err_km.device)

#         loss = self.lambda_u0 * L_u0 + self.lambda_pos * L_pos + self.lambda_ps * L_proc

#         with torch.no_grad():
#             imp_metric = 0.0
#             if len(pos_errors) >= 2:
#                 e0 = pos_errors[0].clamp(min=1e-3)
#                 for k in range(len(pos_errors) - 1):
#                     imp_metric += ((pos_errors[k] - pos_errors[k + 1]) / e0).mean().item()
#                 imp_metric /= (len(pos_errors) - 1)

#         return loss, {
#             "loss": loss.item(),
#             "L_u0": L_u0.item(),
#             "L_pos": L_pos.item(),
#             "L_proc": L_proc.item(),
#             "imp_metric": imp_metric,
#             "u0_res_norm": torch.norm(u0 - batch["dv1"], dim=-1).mean().item() * 1000.0,
#             "dv_res_norm": torch.norm(dv_final - batch["dv1"], dim=-1).mean().item() * 1000.0,
#             "pos_err_0": pos_errors[0].mean().item(),
#             "pos_err_K": pos_errors[-1].mean().item(),
#         }


# def build_run_config() -> RunConfig:
#     """
#     Lightweight CLI overrides without argparse.

#     Main workflow is editing RunConfig defaults above.
#     Supported flags:
#       --quick / --full
#       --no_train / --resume
#       --no_anchor_to_lambert
#       --learn_u0
#       --device=auto|cuda|mps|cpu
#       --data_dir=... / --ckpt_dir=...
#       --lr=... / --dv_max=... / --pos_scale_m=...
#       --correction_scale=... / --lambda_u0=... / --lambda_pos=... / --lambda_ps=...
#       --num_workers=...
#       --no_amp / --amp_dtype=bf16|fp16
#       --compile_model
#     """
#     cfg = RunConfig()
#     args = sys.argv[1:]
#     aset = set(args)

#     if "--quick" in aset:
#         cfg.quick = True
#     if "--full" in aset:
#         cfg.quick = False
#     if "--no_train" in aset:
#         cfg.do_train = False
#     if "--resume" in aset:
#         cfg.resume = True
#     if "--no_anchor_to_lambert" in aset:
#         cfg.anchor_to_lambert = False
#     if "--learn_u0" in aset:
#         cfg.skip_u0_learning = False
#     if "--no_amp" in aset:
#         cfg.amp = False
#     if "--compile_model" in aset:
#         cfg.compile_model = True

#     for a in args:
#         if a.startswith("--device="):
#             cfg.device = a.split("=", 1)[1]
#         elif a.startswith("--data_dir="):
#             cfg.data_dir = Path(a.split("=", 1)[1])
#         elif a.startswith("--ckpt_dir="):
#             cfg.ckpt_dir = Path(a.split("=", 1)[1])
#         elif a.startswith("--lr="):
#             cfg.lr = float(a.split("=", 1)[1])
#         elif a.startswith("--dv_max="):
#             cfg.dv_max = float(a.split("=", 1)[1])
#         elif a.startswith("--pos_scale_m="):
#             cfg.pos_scale_m = float(a.split("=", 1)[1])
#         elif a.startswith("--correction_scale="):
#             cfg.correction_scale = float(a.split("=", 1)[1])
#         elif a.startswith("--lambda_u0="):
#             cfg.lambda_u0 = float(a.split("=", 1)[1])
#         elif a.startswith("--lambda_pos="):
#             cfg.lambda_pos = float(a.split("=", 1)[1])
#         elif a.startswith("--lambda_ps="):
#             cfg.lambda_ps = float(a.split("=", 1)[1])
#         elif a.startswith("--num_workers="):
#             cfg.num_workers = int(a.split("=", 1)[1])
#         elif a.startswith("--amp_dtype="):
#             cfg.amp_dtype = a.split("=", 1)[1].lower()

#     if cfg.device not in ("auto", "cuda", "mps", "cpu"):
#         raise ValueError(f"Invalid device: {cfg.device}")
#     if cfg.amp_dtype not in ("bf16", "fp16"):
#         raise ValueError(f"Invalid amp_dtype: {cfg.amp_dtype} (use bf16 or fp16)")
#     return cfg


# def _pick_device(requested: str) -> torch.device:
#     if requested == "cuda":
#         if not torch.cuda.is_available():
#             raise RuntimeError("Requested --device cuda but CUDA is unavailable.")
#         return torch.device("cuda")
#     if requested == "mps":
#         if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
#             raise RuntimeError("Requested --device mps but MPS is unavailable.")
#         return torch.device("mps")
#     if requested == "cpu":
#         return torch.device("cpu")
#     if torch.cuda.is_available():
#         return torch.device("cuda")
#     if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#         return torch.device("mps")
#     return torch.device("cpu")


# def _to_device_batch(batch, device, non_blocking=False):
#     return {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}


# def evaluate(model, loader, criterion, device, anchor_to_lambert=True, amp_enabled=False, amp_dtype=torch.bfloat16):
#     model.eval()
#     losses, posk, imp, u0res, dvres = [], [], [], [], []
#     if device.type == "cuda":
#         amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled)
#     elif device.type == "mps":
#         amp_ctx = torch.autocast(device_type="cpu", enabled=False)
#     else:
#         amp_ctx = nullcontext()
#     with torch.no_grad():
#         for b in loader:
#             b = _to_device_batch(b, device, non_blocking=(device.type == "cuda"))
#             with amp_ctx:
#                 dv_seed = b["dv1"] if anchor_to_lambert else None
#                 out = model(b["r0"], b["v0"], b["r_target"], b["v_target"], b["tof"], dv_lambert=dv_seed)
#                 _, m = criterion(out, b, model)
#             losses.append(m["loss"])
#             posk.append(m["pos_err_K"])
#             imp.append(m["imp_metric"])
#             u0res.append(m["u0_res_norm"])
#             dvres.append(m["dv_res_norm"])
#     return {
#         "loss": float(np.mean(losses)),
#         "pos_err_K": float(np.mean(posk)),
#         "imp": float(np.mean(imp)),
#         "u0_res_norm": float(np.mean(u0res)),
#         "dv_res_norm": float(np.mean(dvres)),
#     }


# def save_ckpt(model, optimizer, scheduler, epoch, best_val, history, path, cfg, scales):
#     torch.save(
#         {
#             "epoch": epoch,
#             "best_val": best_val,
#             "model_state_dict": model.state_dict(),
#             "optimizer_state_dict": optimizer.state_dict(),
#             "scheduler_state_dict": scheduler.state_dict(),
#             "history": history,
#             "net_cfg": vars(cfg.net_cfg),
#             "scales": scales,
#             "cfg": vars(cfg),
#         },
#         path,
#     )


# @torch.no_grad()
# def collect_eval_samples(model, loader, device, anchor_to_lambert=True, amp_enabled=False, amp_dtype=torch.bfloat16):
#     pos_err_k = []
#     lambert_j2_err = []
#     dv_res_m = []
#     u0_res_m = []
#     if device.type == "cuda":
#         amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled)
#     else:
#         amp_ctx = nullcontext()
#     for b in loader:
#         b = _to_device_batch(b, device, non_blocking=(device.type == "cuda"))
#         with amp_ctx:
#             dv_seed = b["dv1"] if anchor_to_lambert else None
#             out = model(b["r0"], b["v0"], b["r_target"], b["v_target"], b["tof"], dv_lambert=dv_seed)
#             r_expert_roll, _ = model.coast(b["r0"], b["v0"] + b["dv1"], b["tof"])
#         pos_err_k.append(out["pos_errors"][-1].detach().cpu().numpy())
#         lambert_j2_err.append(torch.norm(r_expert_roll - b["r_target"], dim=-1).detach().cpu().numpy())
#         u0_res_m.append(torch.norm(out["dv_iterations"][0] - b["dv1"], dim=-1).detach().cpu().numpy() * 1000.0)
#         dv_res_m.append(torch.norm(out["dv_final"] - b["dv1"], dim=-1).detach().cpu().numpy() * 1000.0)
#     return {
#         "pos_err_k": np.concatenate(pos_err_k, axis=0),
#         "lambert_j2_err": np.concatenate(lambert_j2_err, axis=0),
#         "u0_res_m": np.concatenate(u0_res_m, axis=0),
#         "dv_res_m": np.concatenate(dv_res_m, axis=0),
#     }


# def main():
#     cfg = build_run_config()
#     device = _pick_device(cfg.device)
#     use_cuda_amp = cfg.amp and device.type == "cuda"
#     amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
#     use_grad_scaler = use_cuda_amp and amp_dtype == torch.float16
#     scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
#     if device.type == "cuda":
#         torch.backends.cudnn.benchmark = True
#         torch.set_float32_matmul_precision("high")

#     cfg.ckpt_dir.mkdir(exist_ok=True)
#     if not cfg.train_path.exists() or not cfg.test_path.exists():
#         raise FileNotFoundError("Missing lambert_train.npz/lambert_test.npz in data_dir.")

#     print(f"Device: {device}")
#     print(f"TRAIN={cfg.train_path}")
#     print(f"TEST ={cfg.test_path}")
#     print(f"CKPT ={cfg.ckpt_path}")
#     print(f"MODE = u0-self-supervised (no dv1_corrected required)")
#     print(
#         f"amp={use_cuda_amp} (dtype={cfg.amp_dtype}), "
#         f"compile_model={cfg.compile_model}, num_workers={cfg.num_workers}"
#     )

#     train_ds = LambertDataset(cfg.train_path)
#     test_ds = LambertDataset(cfg.test_path)
#     use_workers = max(0, int(cfg.num_workers))
#     pin_mem = device.type == "cuda"
#     train_loader = DataLoader(
#         train_ds,
#         batch_size=cfg.batch_size,
#         shuffle=True,
#         num_workers=use_workers,
#         pin_memory=pin_mem,
#         persistent_workers=(use_workers > 0),
#     )
#     test_loader = DataLoader(
#         test_ds,
#         batch_size=cfg.batch_size,
#         shuffle=False,
#         num_workers=use_workers,
#         pin_memory=pin_mem,
#         persistent_workers=(use_workers > 0),
#     )

#     # Default normalization scales come from dataset; checkpoint scales can override.
#     r_scale, v_scale, tof_scale, dv_scale, _ = train_ds.get_scales()
#     model_net_cfg = cfg.net_cfg
#     ckpt = None
#     if (cfg.resume or not cfg.do_train) and cfg.ckpt_path.exists():
#         ckpt = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
#         if "net_cfg" in ckpt:
#             model_net_cfg = NetConfig(**ckpt["net_cfg"])
#         if "scales" in ckpt:
#             sc = ckpt["scales"]
#             r_scale = float(sc.get("r", r_scale))
#             v_scale = float(sc.get("v", v_scale))
#             tof_scale = float(sc.get("tof", tof_scale))
#             dv_scale = float(sc.get("dv", dv_scale))

#     model = LambertTRC(
#         model_net_cfg,
#         n_coast_steps=cfg.n_coast,
#         dv_max=cfg.dv_max,
#         correction_scale=cfg.correction_scale,
#     ).to(device)
#     model.set_normalization(r_scale, v_scale, tof_scale, dv_scale, pos_scale_km=cfg.pos_scale_m / 1000.0)

#     # Optional mode: keep u0 exactly at Lambert seed and learn only iterative refinements.
#     if cfg.anchor_to_lambert and cfg.skip_u0_learning:
#         for p in model.init_decoder.parameters():
#             p.requires_grad = False
#         with torch.no_grad():
#             for p in model.init_decoder.parameters():
#                 p.zero_()

#     criterion = U0SelfSupervisedJ2Loss(
#         lambda_u0=cfg.lambda_u0,
#         lambda_pos=cfg.lambda_pos,
#         lambda_ps=cfg.lambda_ps,
#         use_u0_loss=not cfg.skip_u0_learning,
#     )
#     optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-5)

#     print(f"Params: {count_params(model):,}")
#     print(
#         f"Loss weights: lambda_u0={cfg.lambda_u0}, lambda_pos={cfg.lambda_pos}, lambda_ps={cfg.lambda_ps}"
#     )
#     print(f"correction_scale={cfg.correction_scale:.4f} km/s")
#     print(f"anchor_to_lambert={cfg.anchor_to_lambert}")
#     print(f"skip_u0_learning={cfg.skip_u0_learning}")

#     history = {
#         "train_loss": [], "val_loss": [],
#         "train_pos": [], "val_pos": [],
#         "train_dvres_m": [], "val_dvres_m": [],
#         "train_imp": [], "val_imp": [],
#         "val_u0res_m": [],
#     }
#     start_epoch, best_val = 1, float("inf")

#     if cfg.resume and ckpt is not None:
#         model.load_state_dict(ckpt["model_state_dict"])
#         optimizer.load_state_dict(ckpt["optimizer_state_dict"])
#         scheduler.load_state_dict(ckpt["scheduler_state_dict"])
#         history = ckpt.get("history", history)
#         start_epoch = int(ckpt.get("epoch", 0)) + 1
#         best_val = float(ckpt.get("best_val", float("inf")))
#         print(f"Resuming from epoch {start_epoch} (best_val={best_val:.4f})")
#     elif (not cfg.do_train) and ckpt is not None:
#         model.load_state_dict(ckpt["model_state_dict"])
#         print(
#             f"Loaded checkpoint for eval: epoch={ckpt.get('epoch', 'NA')} "
#             f"best_val={ckpt.get('best_val', float('nan')):.4f}"
#         )
#     elif not cfg.do_train:
#         raise FileNotFoundError(f"Missing checkpoint: {cfg.ckpt_path}")

#     # Compile after loading weights so we don't compile twice / before state restoration.
#     if cfg.compile_model:
#         if hasattr(torch, "compile"):
#             model = torch.compile(model)
#             print("torch.compile enabled")
#         else:
#             print("WARNING: torch.compile not available in this PyTorch build; continuing without compile.")

#     if cfg.do_train:
#         for ep in range(start_epoch, start_epoch + cfg.epochs):
#             t0 = time.time()
#             model.train()
#             ep_losses = []
#             ep_posk = []
#             ep_dvr = []
#             ep_imp = []
#             n_skipped = 0
#             for b in train_loader:
#                 b = _to_device_batch(b, device, non_blocking=pin_mem)
#                 optimizer.zero_grad(set_to_none=True)
#                 if device.type == "cuda":
#                     amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda_amp)
#                 else:
#                     amp_ctx = nullcontext()
#                 with amp_ctx:
#                     dv_seed = b["dv1"] if cfg.anchor_to_lambert else None
#                     out = model(b["r0"], b["v0"], b["r_target"], b["v_target"], b["tof"], dv_lambert=dv_seed)
#                     loss, m = criterion(out, b, model)
#                 if not torch.isfinite(loss):
#                     n_skipped += 1
#                     continue
#                 if use_grad_scaler:
#                     scaler.scale(loss).backward()
#                     scaler.unscale_(optimizer)
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                     scaler.step(optimizer)
#                     scaler.update()
#                 else:
#                     loss.backward()
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                     optimizer.step()
#                 ep_losses.append(loss.item())
#                 ep_posk.append(m["pos_err_K"])
#                 ep_dvr.append(m["dv_res_norm"])
#                 ep_imp.append(m["imp_metric"])

#             scheduler.step()
#             if len(ep_losses) == 0:
#                 raise RuntimeError("All training batches were non-finite; reduce correction_scale or keep anchoring.")
#             val = evaluate(
#                 model,
#                 test_loader,
#                 criterion,
#                 device,
#                 anchor_to_lambert=cfg.anchor_to_lambert,
#                 amp_enabled=use_cuda_amp,
#                 amp_dtype=amp_dtype,
#             )
#             train_loss = float(np.mean(ep_losses))
#             history["train_loss"].append(train_loss)
#             history["val_loss"].append(val["loss"])
#             history["train_pos"].append(float(np.mean(ep_posk)))
#             history["val_pos"].append(val["pos_err_K"])
#             history["train_dvres_m"].append(float(np.mean(ep_dvr)))
#             history["val_u0res_m"].append(val["u0_res_norm"])
#             history["val_dvres_m"].append(val["dv_res_norm"])
#             history["train_imp"].append(float(np.mean(ep_imp)))
#             history["val_imp"].append(val["imp"])

#             dt = time.time() - t0
#             marker = ""

#             if val["loss"] < best_val:
#                 best_val = val["loss"]
#                 save_ckpt(
#                     model,
#                     optimizer,
#                     scheduler,
#                     ep,
#                     best_val,
#                     history,
#                     cfg.ckpt_path,
#                     cfg,
#                     {
#                         "r": r_scale,
#                         "v": v_scale,
#                         "tof": tof_scale,
#                         "dv": dv_scale,
#                         "corr": cfg.correction_scale,
#                         "pos_m": cfg.pos_scale_m,
#                     },
#                 )
#                 print(f"  -> saved best: {cfg.ckpt_path}")
#                 marker = " *"
#             print(
#                 f"Ep {ep}/{cfg.epochs} (abs {ep})  "
#                 f"loss={train_loss:.4f}  "
#                 f"pos={history['train_pos'][-1]*1000.0:.1f}m  "
#                 f"dV={history['train_dvres_m'][-1]:.1f}m/s  "
#                 f"imp={history['train_imp'][-1]:.3f}  |  "
#                 f"val_pos={val['pos_err_K']*1000.0:.1f}m  "
#                 f"val_dV={val['dv_res_norm']:.1f}m/s  "
#                 f"(skip={n_skipped}, {dt:.1f}s){marker}"
#             )
#     final_val = evaluate(
#         model,
#         test_loader,
#         criterion,
#         device,
#         anchor_to_lambert=cfg.anchor_to_lambert,
#         amp_enabled=use_cuda_amp,
#         amp_dtype=amp_dtype,
#     )
#     print("\nFinal evaluation (test):")
#     print(
#         f"loss={final_val['loss']:.4f}, pos_err_K={final_val['pos_err_K']:.3f} km, "
#         f"u0_res={final_val['u0_res_norm']:.2f} m/s, dv_res={final_val['dv_res_norm']:.2f} m/s, "
#         f"imp={final_val['imp']:.3f}"
#     )

#     # Plots
#     plotters.plot_u0_training_curves(history, cfg.ckpt_dir / "lambert_u0_selfsup_training_curves.png")
#     train_eval = collect_eval_samples(
#         model,
#         train_loader,
#         device,
#         anchor_to_lambert=cfg.anchor_to_lambert,
#         amp_enabled=use_cuda_amp,
#         amp_dtype=amp_dtype,
#     )
#     test_eval = collect_eval_samples(
#         model,
#         test_loader,
#         device,
#         anchor_to_lambert=cfg.anchor_to_lambert,
#         amp_enabled=use_cuda_amp,
#         amp_dtype=amp_dtype,
#     )
#     plotters.plot_u0_eval_summary(
#         test_eval,
#         cfg.ckpt_dir / "lambert_u0_selfsup_eval_summary_test.png",
#         split_name="test",
#     )
#     plotters.plot_u0_arrival_error_comparison(
#         train_eval,
#         test_eval,
#         cfg.ckpt_dir / "lambert_u0_selfsup_vs_lambert_arrival_error.png",
#     )


# if __name__ == "__main__":
#     main()
"""Train/evaluate LambertTRC without J2-corrected dV labels.

Loss design:
- Make initial control u0 match Lambert dv1.
- Let iterative refinement reduce terminal J2 position error.
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
    quick: bool = True
    do_train: bool = True
    resume: bool = False
    data_dir: Path = THIS_DIR / "data"
    ckpt_dir: Path = THIS_DIR / "checkpoints"
    dv_max: float = 3.0
    lr: float = 1e-4
    pos_scale_m: float = 1000.0
    correction_scale: float = 0.005  # km/s, used for iterative updates inside TRC
    lambda_u0: float = 5.0
    lambda_pos: float = 1.0
    lambda_ps: float = 0.05
    anchor_to_lambert: bool = False

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
        return self.ckpt_dir / "trc_lambert_u0_selfsup_best.pt"


class U0SelfSupervisedJ2Loss(nn.Module):
    """No-label TRC loss: u0 Lambert match + iterative J2 terminal improvement."""

    def __init__(self, lambda_u0=1.0, lambda_pos=1.0, lambda_ps=0.05):
        super().__init__()
        self.lambda_u0 = lambda_u0
        self.lambda_pos = lambda_pos
        self.lambda_ps = lambda_ps

    def forward(self, output, batch, model):
        pos_errors = output["pos_errors"]
        dv_iters = output["dv_iterations"]
        u0 = dv_iters[0]
        dv_final = output["dv_final"]

        # 1) Initial policy should reproduce Lambert burn.
        L_u0 = F.mse_loss(u0, batch["dv1"]) / (model.dv_scale ** 2)

        # 2) Final terminal J2 position error should be small.
        pos_err_km = pos_errors[-1]
        pos_norm = getattr(model, "pos_scale", torch.tensor(100.0, device=pos_err_km.device))
        L_pos = (pos_err_km / pos_norm).pow(2).mean()

        # 3) Encourage monotonic refinement across TRC iterations.
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
            "pos_err_0": pos_errors[0].mean().item(),
            "pos_err_K": pos_errors[-1].mean().item(),
        }


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="TRC training without J2-corrected dV labels.")
    p.add_argument("--quick", action="store_true", default=True)
    p.add_argument("--full", action="store_true")
    p.add_argument("--no_train", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--data_dir", type=Path, default=THIS_DIR / "data")
    p.add_argument("--ckpt_dir", type=Path, default=THIS_DIR / "checkpoints")
    p.add_argument("--dv_max", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--pos_scale_m", type=float, default=1000.0)
    p.add_argument("--correction_scale", type=float, default=0.005)
    p.add_argument("--lambda_u0", type=float, default=5.0)
    p.add_argument("--lambda_pos", type=float, default=1.0)
    p.add_argument("--lambda_ps", type=float, default=0.05)
    p.add_argument("--anchor_to_lambert", action="store_true",
                   help="Seed rollout with Lambert dv1.")
    p.add_argument("--no_anchor_to_lambert", action="store_true",
                   help="Disable Lambert-anchored rollout (less stable).")
    a = p.parse_args()

    return RunConfig(
        quick=a.quick and not a.full,
        do_train=not a.no_train,
        resume=a.resume,
        data_dir=a.data_dir,
        ckpt_dir=a.ckpt_dir,
        dv_max=a.dv_max,
        lr=a.lr,
        pos_scale_m=a.pos_scale_m,
        correction_scale=a.correction_scale,
        lambda_u0=a.lambda_u0,
        lambda_pos=a.lambda_pos,
        lambda_ps=a.lambda_ps,
        anchor_to_lambert=(a.anchor_to_lambert and not a.no_anchor_to_lambert),
    )


def evaluate(model, loader, criterion, device, anchor_to_lambert=True):
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


def save_ckpt(model, optimizer, scheduler, epoch, best_val, history, path, cfg, scales):
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
        },
        path,
    )


def main():
    cfg = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg.ckpt_dir.mkdir(exist_ok=True)
    if not cfg.train_path.exists() or not cfg.test_path.exists():
        raise FileNotFoundError("Missing lambert_train.npz/lambert_test.npz in data_dir.")

    print(f"Device: {device}")
    print(f"TRAIN={cfg.train_path}")
    print(f"TEST ={cfg.test_path}")
    print(f"CKPT ={cfg.ckpt_path}")
    print(f"MODE = u0-self-supervised (no dv1_corrected required)")

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

    criterion = U0SelfSupervisedJ2Loss(
        lambda_u0=cfg.lambda_u0,
        lambda_pos=cfg.lambda_pos,
        lambda_ps=cfg.lambda_ps,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-5)

    print(f"Params: {count_params(model):,}")
    print(
        f"Loss weights: lambda_u0={cfg.lambda_u0}, lambda_pos={cfg.lambda_pos}, lambda_ps={cfg.lambda_ps}"
    )
    print(f"correction_scale={cfg.correction_scale:.4f} km/s")
    print(f"anchor_to_lambert={cfg.anchor_to_lambert}")

    history = {"train_loss": [], "val_loss": [], "val_pos": [], "val_u0res_m": [], "val_dvres_m": []}
    start_epoch, best_val = 1, float("inf")

    if cfg.resume and cfg.ckpt_path.exists():
        ckpt = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        history = ckpt.get("history", history)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", float("inf")))
        print(f"Resuming from epoch {start_epoch} (best_val={best_val:.4f})")

    if cfg.do_train:
        for ep in range(start_epoch, start_epoch + cfg.epochs):
            t0 = time.time()
            model.train()
            ep_losses = []
            n_skipped = 0
            for b in train_loader:
                b = {k: v.to(device) for k, v in b.items()}
                optimizer.zero_grad()
                dv_seed = b["dv1"] if cfg.anchor_to_lambert else None
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
                raise RuntimeError("All training batches were non-finite; reduce correction_scale or keep anchoring.")
            val = evaluate(model, test_loader, criterion, device, anchor_to_lambert=cfg.anchor_to_lambert)
            train_loss = float(np.mean(ep_losses))
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val["loss"])
            history["val_pos"].append(val["pos_err_K"])
            history["val_u0res_m"].append(val["u0_res_norm"])
            history["val_dvres_m"].append(val["dv_res_norm"])

            dt = time.time() - t0
            print(
                f"[{ep:03d}] train={train_loss:.4f} val={val['loss']:.4f} "
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
                    cfg.ckpt_path,
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
    else:
        if not cfg.ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {cfg.ckpt_path}")
        ckpt = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    final_val = evaluate(model, test_loader, criterion, device, anchor_to_lambert=cfg.anchor_to_lambert)
    print("\nFinal evaluation (test):")
    print(
        f"loss={final_val['loss']:.4f}, pos_err_K={final_val['pos_err_K']:.3f} km, "
        f"u0_res={final_val['u0_res_norm']:.2f} m/s, dv_res={final_val['dv_res_norm']:.2f} m/s, "
        f"imp={final_val['imp']:.3f}"
    )


if __name__ == "__main__":
    main()
