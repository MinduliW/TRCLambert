"""Diagnostic plots for Stage 2 full-velocity learning.

This script evaluates the trained Stage 2 model as a full dv predictor and
produces diagnostics centered on:
1. training history
2. u0 vs dv_final residuals
3. dv_final vs target dv component scatter
4. dv residual distributions
5. dv residual vs TOF and transfer magnitude
6. terminal position error reduction across iterations
7. per-iteration dv update size
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from trc import NetConfig

device = torch.device("cpu")
PLOTS_DIR = THIS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Stage 2 full-velocity diagnostics.")
    parser.add_argument(
        "--state_repr",
        choices=["cartesian", "spherical"],
        default="cartesian",
        help="Checkpoint family to load when --checkpoint is not provided.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional explicit checkpoint path.",
    )
    parser.add_argument(
        "--output_tag",
        type=str,
        default=None,
        help="Suffix for output filenames. Defaults to '<state_repr>_full_velocity'.",
    )
    parser.add_argument(
        "--eval_k_max",
        type=int,
        default=10,
        help="Maximum refinement iterations to run during diagnostics.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving them.",
    )
    return parser.parse_args()


def resolve_checkpoint(args) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    suffix = "" if args.state_repr == "cartesian" else "_spherical"
    return THIS_DIR / "checkpoints" / f"trc_lambert_u0_curriculum_best{suffix}.pt"


def apply_normalization_from_checkpoint(model: LambertTRC, ckpt: dict, train_ds: LambertDataset) -> dict:
    scales = ckpt.get("scales", {})
    r_min = scales.get("r_min", torch.cat([train_ds.r0, train_ds.r_target], dim=0).amin(dim=0).tolist())
    r_max = scales.get("r_max", torch.cat([train_ds.r0, train_ds.r_target], dim=0).amax(dim=0).tolist())
    v_min = scales.get("v_min", train_ds.v0.amin(dim=0).tolist())
    v_max = scales.get("v_max", train_ds.v0.amax(dim=0).tolist())
    tof_min = scales.get("tof_min", train_ds.tof.min().item())
    tof_max = scales.get("tof_max", train_ds.tof.max().item())
    dv_scale = scales.get("dv", torch.norm(train_ds.dv1, dim=-1).median().item())
    nrev_min = scales.get("nrev_min", float(train_ds.nrev.min().item()) if len(train_ds.nrev) > 0 else 0.0)
    nrev_max = scales.get("nrev_max", float(train_ds.nrev.max().item()) if len(train_ds.nrev) > 0 else 1.0)
    r_mag_min = scales.get(
        "r_mag_min",
        float(torch.cat([torch.norm(train_ds.r0, dim=-1), torch.norm(train_ds.r_target, dim=-1)], dim=0).min().item()),
    )
    r_mag_max = scales.get(
        "r_mag_max",
        float(torch.cat([torch.norm(train_ds.r0, dim=-1), torch.norm(train_ds.r_target, dim=-1)], dim=0).max().item()),
    )
    v_mag_min = scales.get("v_mag_min", float(torch.norm(train_ds.v0, dim=-1).min().item()))
    v_mag_max = scales.get("v_mag_max", float(torch.norm(train_ds.v0, dim=-1).max().item()))
    dv_mag_min = scales.get("dv_mag_min", float(torch.norm(train_ds.dv1, dim=-1).min().item()))
    dv_mag_max = scales.get("dv_mag_max", float(torch.norm(train_ds.dv1, dim=-1).max().item()))
    use_tof_log = bool(scales.get("use_tof_log", False))
    tof_log_min = scales.get("tof_log_min")
    tof_log_max = scales.get("tof_log_max")
    if use_tof_log and (tof_log_min is None or tof_log_max is None):
        tof_log_min = torch.log1p(train_ds.tof).min().item()
        tof_log_max = torch.log1p(train_ds.tof).max().item()
    pos_scale_km = scales.get("pos_m", 300000.0) / 1000.0

    model.set_normalization(
        r_min,
        r_max,
        v_min,
        v_max,
        tof_min,
        tof_max,
        dv_scale,
        pos_scale_km=pos_scale_km,
        use_tof_log=use_tof_log,
        tof_log_min=tof_log_min if use_tof_log else None,
        tof_log_max=tof_log_max if use_tof_log else None,
        nrev_min=nrev_min,
        nrev_max=nrev_max,
        r_mag_min=r_mag_min,
        r_mag_max=r_mag_max,
        v_mag_min=v_mag_min,
        v_mag_max=v_mag_max,
        dv_mag_min=dv_mag_min,
        dv_mag_max=dv_mag_max,
    )
    return scales


def load_state_dict_compat(model: LambertTRC, source_sd: dict):
    model_sd = model.state_dict()
    adapted = {}
    skipped = []
    for key, value in source_sd.items():
        if key not in model_sd:
            skipped.append(key)
            continue

        target = model_sd[key]
        if value.shape == target.shape:
            adapted[key] = value
            continue

        if key == "state_encoder.0.weight" and value.ndim == 2 and target.ndim == 2:
            if value.shape[0] == target.shape[0] and value.shape[1] == 10 and target.shape[1] == 13:
                new_weight = torch.zeros_like(target)
                new_weight[:, 0:3] = value[:, 0:3]
                new_weight[:, 3:6] = value[:, 3:6]
                new_weight[:, 6:9] = value[:, 6:9]
                new_weight[:, 9:10] = value[:, 9:10]
                adapted[key] = new_weight
                continue
            if value.shape[0] == target.shape[0] and value.shape[1] == 13 and target.shape[1] == 10:
                new_weight = torch.zeros_like(target)
                new_weight[:, 0:3] = value[:, 0:3]
                new_weight[:, 3:6] = value[:, 3:6]
                new_weight[:, 6:9] = value[:, 6:9]
                new_weight[:, 9:10] = value[:, 12:13]
                adapted[key] = new_weight
                continue

        skipped.append(f"{key}: src={tuple(value.shape)} target={tuple(target.shape)}")

    missing, unexpected = model.load_state_dict(adapted, strict=False)
    if skipped:
        print(f"  [ckpt compat] skipped {len(skipped)} keys")
        for item in skipped[:6]:
            print(f"    - {item}")
        if len(skipped) > 6:
            print(f"    ... and {len(skipped) - 6} more")
    if missing:
        print(f"  [ckpt compat] missing {len(missing)} keys")
    if unexpected:
        print(f"  [ckpt compat] unexpected {len(unexpected)} keys")


def collect_predictions(model: LambertTRC, dataset: LambertDataset, label: str, batch_size: int, eval_k_max: int) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_target = []
    all_tof = []
    all_u0 = []
    all_dv_final = []
    all_pos_errors = None
    all_dv_iters = None

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                batch["r0"],
                batch["v0"],
                batch["r_target"],
                batch["v_target"],
                batch["tof"],
                nrev=batch.get("nrev"),
                ncase=batch.get("ncase"),
                prograde=batch.get("prograde"),
                stage1_only=False,
                K_min=eval_k_max,
                K_max=eval_k_max,
            )
            all_target.append(batch["dv1"].cpu().numpy())
            all_tof.append(batch["tof"].cpu().numpy())
            all_u0.append(out["dv_iterations"][0].cpu().numpy())
            all_dv_final.append(out["dv_final"].cpu().numpy())

            batch_pos = [pe.cpu().numpy() for pe in out["pos_errors"]]
            if all_pos_errors is None:
                all_pos_errors = [[] for _ in range(len(batch_pos))]
                all_dv_iters = [[] for _ in range(len(out["dv_iterations"]))]
            for idx, pe in enumerate(batch_pos):
                all_pos_errors[idx].append(pe)
            for idx, dv in enumerate(out["dv_iterations"]):
                all_dv_iters[idx].append(dv.cpu().numpy())

    dv_target = np.concatenate(all_target, axis=0)
    tof = np.concatenate(all_tof, axis=0).squeeze()
    u0 = np.concatenate(all_u0, axis=0)
    dv_final = np.concatenate(all_dv_final, axis=0)
    pos_errors = [np.concatenate(chunks, axis=0) for chunks in all_pos_errors]
    dv_iters = [np.concatenate(chunks, axis=0) for chunks in all_dv_iters]

    u0_err = u0 - dv_target
    dv_err = dv_final - dv_target
    u0_err_ms = np.linalg.norm(u0_err, axis=1) * 1000.0
    dv_err_ms = np.linalg.norm(dv_err, axis=1) * 1000.0
    target_mag_ms = np.linalg.norm(dv_target, axis=1) * 1000.0
    refinement_delta_ms = np.linalg.norm(dv_final - u0, axis=1) * 1000.0
    step_sizes_ms = [
        np.linalg.norm(dv_iters[k] - dv_iters[k - 1], axis=1) * 1000.0
        for k in range(1, len(dv_iters))
    ]

    print(f"\n{label} set ({len(dv_target)} samples):")
    print(
        f"  u0 residual: mean={u0_err_ms.mean():.1f} m/s, "
        f"median={np.median(u0_err_ms):.1f} m/s, "
        f"p95={np.percentile(u0_err_ms, 95):.1f} m/s"
    )
    print(
        f"  dv_final residual: mean={dv_err_ms.mean():.1f} m/s, "
        f"median={np.median(dv_err_ms):.1f} m/s, "
        f"p95={np.percentile(dv_err_ms, 95):.1f} m/s"
    )
    print(
        f"  terminal pos err: mean={pos_errors[-1].mean():.2f} km, "
        f"median={np.median(pos_errors[-1]):.2f} km, "
        f"p95={np.percentile(pos_errors[-1], 95):.2f} km"
    )

    return {
        "dv_target": dv_target,
        "tof": tof,
        "u0": u0,
        "dv_final": dv_final,
        "u0_err_ms": u0_err_ms,
        "dv_err_ms": dv_err_ms,
        "target_mag_ms": target_mag_ms,
        "refinement_delta_ms": refinement_delta_ms,
        "pos_errors": pos_errors,
        "dv_iters": dv_iters,
        "step_sizes_ms": step_sizes_ms,
    }


def save_training_history(history: dict, output_tag: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    if "train_loss" in history and len(history["train_loss"]) > 0:
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        axes[0].semilogy(epochs, history["train_loss"], label="Train", color="steelblue")
        if "val_loss" in history:
            axes[0].semilogy(epochs, history["val_loss"], label="Val", color="coral", ls="--")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Stage 2 Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        if "val_u0res_m" in history:
            axes[1].plot(epochs, history["val_u0res_m"], color="gray", ls=":", label="u0")
        if "val_dvres_m" in history:
            axes[1].plot(epochs, history["val_dvres_m"], color="steelblue", label="dv_final")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("dv residual (m/s)")
        axes[1].set_title("Velocity Residuals")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        if "val_pos" in history:
            axes[2].plot(epochs, history["val_pos"], color="coral", label="val pos_err_K")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Position Error (km)")
        axes[2].set_title("Terminal Position Error")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "No training history", ha="center", va="center")

    fig.tight_layout()
    path = PLOTS_DIR / f"s2_full_velocity_training_history_{output_tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


def save_dv_component_scatter(test_data: dict, output_tag: str):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    comp_names = ["dv_x", "dv_y", "dv_z"]
    dv_target = test_data["dv_target"]
    dv_final = test_data["dv_final"]

    for idx, ax in enumerate(axes):
        ax.scatter(dv_target[:, idx], dv_final[:, idx], s=3, alpha=0.25, color="steelblue")
        lo = min(dv_target[:, idx].min(), dv_final[:, idx].min())
        hi = max(dv_target[:, idx].max(), dv_final[:, idx].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
        rms_ms = np.sqrt(np.mean(((dv_final[:, idx] - dv_target[:, idx]) * 1000.0) ** 2))
        ax.set_xlabel(f"Target {comp_names[idx]} (km/s)")
        ax.set_ylabel(f"Predicted {comp_names[idx]} (km/s)")
        ax.set_title(f"{comp_names[idx]}: dv_final vs target")
        ax.grid(True, alpha=0.3)
        ax.text(
            0.04,
            0.95,
            f"RMS={rms_ms:.1f} m/s",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
        )

    fig.tight_layout()
    path = PLOTS_DIR / f"s2_full_velocity_scatter_components_{output_tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


def save_residual_distribution(train_data: dict, test_data: dict, output_tag: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    rows = [
        (axes[0], train_data, "Train"),
        (axes[1], test_data, "Test"),
    ]
    for ax, data, label in rows:
        bins = np.linspace(0.0, max(np.percentile(data["u0_err_ms"], 99.5), np.percentile(data["dv_err_ms"], 99.5)), 80)
        ax.hist(data["u0_err_ms"], bins=bins, alpha=0.55, color="gray", label=f"u0 mean={data['u0_err_ms'].mean():.1f}")
        ax.hist(data["dv_err_ms"], bins=bins, alpha=0.7, color="steelblue", label=f"dv_final mean={data['dv_err_ms'].mean():.1f}")
        ax.set_xlabel("Velocity error norm (m/s)")
        ax.set_ylabel("Count")
        ax.set_title(f"{label}: u0 vs dv_final residual")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = PLOTS_DIR / f"s2_full_velocity_error_distribution_{output_tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


def save_error_vs_problem_axes(train_data: dict, test_data: dict, output_tag: str):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    rows = [
        (0, train_data, "Train"),
        (1, test_data, "Test"),
    ]
    for row_idx, data, label in rows:
        tof_min = data["tof"] / 60.0
        ax = axes[row_idx, 0]
        ax.scatter(tof_min, data["u0_err_ms"], s=3, alpha=0.2, color="gray", label="u0")
        ax.scatter(tof_min, data["dv_err_ms"], s=3, alpha=0.3, color="steelblue", label="dv_final")
        ax.set_xlabel("Time of Flight (min)")
        ax.set_ylabel("Velocity error norm (m/s)")
        ax.set_title(f"{label}: residual vs TOF")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[row_idx, 1]
        ax.scatter(data["target_mag_ms"], data["u0_err_ms"], s=3, alpha=0.2, color="gray", label="u0")
        ax.scatter(data["target_mag_ms"], data["dv_err_ms"], s=3, alpha=0.3, color="steelblue", label="dv_final")
        ax.set_xlabel("Target |dv| (m/s)")
        ax.set_ylabel("Velocity error norm (m/s)")
        ax.set_title(f"{label}: residual vs transfer magnitude")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = PLOTS_DIR / f"s2_full_velocity_error_vs_axes_{output_tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


def save_position_iteration_plot(train_data: dict, test_data: dict, output_tag: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, data, label in [(axes[0], train_data, "Train"), (axes[1], test_data, "Test")]:
        pe = data["pos_errors"]
        positions = np.arange(len(pe))
        means = [x.mean() for x in pe]
        medians = [np.median(x) for x in pe]
        p95s = [np.percentile(x, 95) for x in pe]
        ax.semilogy(positions, means, "o-", color="steelblue", label="mean")
        ax.semilogy(positions, medians, "s-", color="coral", label="median")
        ax.semilogy(positions, p95s, "^-", color="gray", label="p95")
        ax.set_xlabel("Iteration checkpoint")
        ax.set_ylabel("Terminal position error (km)")
        ax.set_title(f"{label}: position error vs iteration")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = PLOTS_DIR / f"s2_full_velocity_position_iterations_{output_tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


def save_position_distribution(train_data: dict, test_data: dict, output_tag: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, data, label in [(axes[0], train_data, "Train"), (axes[1], test_data, "Test")]:
        pe0 = data["pos_errors"][0]
        pef = data["pos_errors"][-1]
        xmax = max(np.percentile(pe0, 99), np.percentile(pef, 99))
        bins = np.linspace(0.0, xmax, 80)
        ax.hist(pe0, bins=bins, alpha=0.55, color="coral", label=f"u0 mean={pe0.mean():.1f} km")
        ax.hist(pef, bins=bins, alpha=0.7, color="steelblue", label=f"dv_final mean={pef.mean():.1f} km")
        ax.set_xlabel("Terminal position error (km)")
        ax.set_ylabel("Count")
        ax.set_title(f"{label}: terminal position error")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = PLOTS_DIR / f"s2_full_velocity_position_distribution_{output_tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


def save_update_analysis(test_data: dict, output_tag: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].hist(test_data["refinement_delta_ms"], bins=70, color="steelblue", alpha=0.75, edgecolor="white", lw=0.3)
    axes[0].set_xlabel("|dv_final - u0| (m/s)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Test: net Stage 2 update size")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(test_data["u0_err_ms"], test_data["refinement_delta_ms"], s=4, alpha=0.25, color="steelblue")
    axes[1].set_xlabel("u0 residual (m/s)")
    axes[1].set_ylabel("Net Stage 2 update (m/s)")
    axes[1].set_title("Test: update size vs u0 error")
    axes[1].grid(True, alpha=0.3)

    if test_data["step_sizes_ms"]:
        bp = axes[2].boxplot(
            test_data["step_sizes_ms"],
            positions=np.arange(1, len(test_data["step_sizes_ms"]) + 1),
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black"),
        )
        colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(test_data["step_sizes_ms"])))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("|dv_k - dv_(k-1)| (m/s)")
    axes[2].set_title("Test: per-iteration update size")
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = PLOTS_DIR / f"s2_full_velocity_update_analysis_{output_tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


def print_summary(ckpt: dict, ckpt_path: Path, state_repr: str, train_data: dict, test_data: dict, eval_k_max: int):
    print("\n" + "=" * 68)
    print("STAGE 2 FULL-VELOCITY SUMMARY")
    print("=" * 68)
    print(f"  Checkpoint: {ckpt_path.name}")
    print(f"  State repr: {state_repr}")
    for label, data in [("Train", train_data), ("Test", test_data)]:
        pe0 = data["pos_errors"][0]
        pef = data["pos_errors"][-1]
        print(f"  {label}:")
        print(
            f"    u0 residual:       mean={data['u0_err_ms'].mean():.1f} m/s, "
            f"median={np.median(data['u0_err_ms']):.1f} m/s, p95={np.percentile(data['u0_err_ms'], 95):.1f} m/s"
        )
        print(
            f"    dv_final residual: mean={data['dv_err_ms'].mean():.1f} m/s, "
            f"median={np.median(data['dv_err_ms']):.1f} m/s, p95={np.percentile(data['dv_err_ms'], 95):.1f} m/s"
        )
        print(
            f"    terminal pos err:  mean={pef.mean():.2f} km, "
            f"median={np.median(pef):.2f} km, p95={np.percentile(pef, 95):.2f} km"
        )
        print(
            f"    pos improvement:   {(1.0 - pef.mean() / max(pe0.mean(), 1e-6)) * 100.0:.1f}% mean reduction"
        )
        print(
            f"    net update size:   mean={data['refinement_delta_ms'].mean():.1f} m/s, "
            f"max={data['refinement_delta_ms'].max():.1f} m/s"
        )
    best_val = ckpt.get("best_val", "?")
    print(f"\n  Epoch: {ckpt.get('epoch', '?')}")
    print(f"  Best val loss: {best_val}")
    print(f"  Eval K_max: {eval_k_max}")


def main():
    args = parse_args()
    ckpt_path = resolve_checkpoint(args)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    net_cfg = NetConfig(**ckpt["net_cfg"])
    cfg_saved = ckpt.get("cfg", {})
    state_repr = cfg_saved.get("state_repr", args.state_repr)
    output_tag = args.output_tag or f"{state_repr}_full_velocity"
    correction_scale = ckpt.get("scales", {}).get("corr", 0.05)

    model = LambertTRC(
        net_cfg,
        max_coast_step_s=45.0,
        dv_max=5.0,
        correction_scale=correction_scale,
        state_repr=state_repr,
    ).to(device)

    train_ds = LambertDataset(THIS_DIR / "data" / "lambert_train.npz")
    test_ds = LambertDataset(THIS_DIR / "data" / "lambert_test.npz")
    apply_normalization_from_checkpoint(model, ckpt, train_ds)
    load_state_dict_compat(model, ckpt["model_state_dict"])
    model.eval()

    history = ckpt.get("history", {})
    train_data = collect_predictions(model, train_ds, "Train", args.batch_size, args.eval_k_max)
    test_data = collect_predictions(model, test_ds, "Test", args.batch_size, args.eval_k_max)

    save_training_history(history, output_tag)
    save_dv_component_scatter(test_data, output_tag)
    save_residual_distribution(train_data, test_data, output_tag)
    save_error_vs_problem_axes(train_data, test_data, output_tag)
    save_position_iteration_plot(train_data, test_data, output_tag)
    save_position_distribution(train_data, test_data, output_tag)
    save_update_analysis(test_data, output_tag)
    print_summary(ckpt, ckpt_path, state_repr, train_data, test_data, args.eval_k_max)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
