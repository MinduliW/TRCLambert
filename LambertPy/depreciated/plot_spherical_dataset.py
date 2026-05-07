"""Plot Lambert dataset inputs/outputs in spherical coordinates.

This script visualizes the same spherical conventions used by the model:
    mag = ||vec||
    lon = atan2(y, x)
    lat = atan2(z, sqrt(x^2 + y^2))

It saves overview figures for the encoder inputs and dv target.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent

ENCODER_FEATURE_NAMES_CARTESIAN = (
    "r0_x", "r0_y", "r0_z",
    "v0_x", "v0_y", "v0_z",
    "rt_x", "rt_y", "rt_z",
    "tof",
    "nrev", "ncase", "prograde",
)

ENCODER_FEATURE_NAMES_SPHERICAL = (
    "r0_mag", "r0_alpha", "r0_beta",
    "v0_mag", "v0_alpha", "v0_beta",
    "rt_mag", "rt_alpha", "rt_beta",
    "tof",
    "nrev", "ncase", "prograde",
)


def vector_to_spherical(vec: np.ndarray) -> dict[str, np.ndarray]:
    x = vec[:, 0]
    y = vec[:, 1]
    z = vec[:, 2]
    mag = np.linalg.norm(vec, axis=1)
    xy = np.linalg.norm(vec[:, :2], axis=1)
    alpha = np.arctan2(y, x)
    beta = np.arctan2(z, np.maximum(xy, 1e-12))
    return {
        "mag": mag,
        "alpha_rad": alpha,
        "beta_rad": beta,
        "alpha_deg": np.degrees(alpha),
        "beta_deg": np.degrees(beta),
    }


def sample_arrays(data: dict[str, np.ndarray], limit: int | None) -> dict[str, np.ndarray]:
    if limit is None or limit <= 0:
        return data
    n = len(next(iter(data.values())))
    if limit >= n:
        return data
    idx = np.linspace(0, n - 1, limit, dtype=int)
    return {k: v[idx] for k, v in data.items()}


def load_dataset(path: Path, limit: int | None = None) -> dict[str, np.ndarray]:
    with np.load(path) as raw:
        data = {k: raw[k] for k in raw.files}
    return sample_arrays(data, limit)


def minmax_normalize(value: np.ndarray, vmin: np.ndarray | float, vmax: np.ndarray | float) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    vmin = np.asarray(vmin, dtype=np.float64)
    vmax = np.asarray(vmax, dtype=np.float64)
    half_span = 0.5 * (vmax - vmin)
    half_span = np.where(np.abs(half_span) < 1e-6, 1.0, half_span)
    center = 0.5 * (vmax + vmin)
    return (value - center) / half_span


def normalize_alpha(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float64) / np.pi


def normalize_beta(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float64) / (0.5 * np.pi)


def compute_scales(data: dict[str, np.ndarray], use_tof_log: bool) -> dict[str, np.ndarray | float | bool]:
    r_all = np.concatenate([np.asarray(data["r0"]), np.asarray(data["r_target"])], axis=0)
    v_all = np.asarray(data["v0"])
    dv_all = np.asarray(data["dv1"])
    tof = np.asarray(data["tof"]).reshape(-1, 1)
    nrev = np.asarray(data.get("nrev", np.zeros((len(tof),), dtype=np.float32))).reshape(-1, 1)
    return {
        "r_min": r_all.min(axis=0),
        "r_max": r_all.max(axis=0),
        "v_min": v_all.min(axis=0),
        "v_max": v_all.max(axis=0),
        "r_mag_min": float(np.linalg.norm(r_all, axis=1).min()),
        "r_mag_max": float(np.linalg.norm(r_all, axis=1).max()),
        "v_mag_min": float(np.linalg.norm(v_all, axis=1).min()),
        "v_mag_max": float(np.linalg.norm(v_all, axis=1).max()),
        "dv_mag_min": float(np.linalg.norm(dv_all, axis=1).min()),
        "dv_mag_max": float(np.linalg.norm(dv_all, axis=1).max()),
        "tof_min": float(tof.min()),
        "tof_max": float(tof.max()),
        "tof_log_min": float(np.log1p(tof).min()) if use_tof_log else None,
        "tof_log_max": float(np.log1p(tof).max()) if use_tof_log else None,
        "nrev_min": float(nrev.min()) if len(nrev) > 0 else 0.0,
        "nrev_max": float(nrev.max()) if len(nrev) > 0 else 1.0,
        "use_tof_log": bool(use_tof_log),
    }


def encode_scaled_inputs(
    data: dict[str, np.ndarray],
    scales: dict[str, np.ndarray | float | bool],
    state_repr: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    r0 = np.asarray(data["r0"])
    v0 = np.asarray(data["v0"])
    r_target = np.asarray(data["r_target"])
    tof = np.asarray(data["tof"]).reshape(-1, 1)
    n = len(r0)
    nrev = np.asarray(data.get("nrev", np.zeros(n, dtype=np.float32))).reshape(-1, 1)
    ncase = np.asarray(data.get("ncase", np.zeros(n, dtype=np.float32))).reshape(-1, 1)
    prograde = np.asarray(data.get("prograde", np.zeros(n, dtype=np.float32))).reshape(-1, 1)

    if scales["use_tof_log"]:
        tof_n = minmax_normalize(np.log1p(tof), scales["tof_log_min"], scales["tof_log_max"])
    else:
        tof_n = minmax_normalize(tof, scales["tof_min"], scales["tof_max"])
    nrev_n = minmax_normalize(nrev, scales["nrev_min"], scales["nrev_max"])
    ncase_n = (ncase - 0.5) / 0.5
    prograde_n = (prograde - 0.5) / 0.5

    if state_repr == "cartesian":
        encoded = np.concatenate(
            [
                minmax_normalize(r0, scales["r_min"], scales["r_max"]),
                minmax_normalize(v0, scales["v_min"], scales["v_max"]),
                minmax_normalize(r_target, scales["r_min"], scales["r_max"]),
                tof_n,
                nrev_n,
                ncase_n,
                prograde_n,
            ],
            axis=1,
        )
        return encoded, ENCODER_FEATURE_NAMES_CARTESIAN

    r0_s = vector_to_spherical(r0)
    v0_s = vector_to_spherical(v0)
    rt_s = vector_to_spherical(r_target)
    encoded = np.concatenate(
        [
            np.concatenate(
                [
                    minmax_normalize(r0_s["mag"][:, None], scales["r_mag_min"], scales["r_mag_max"]),
                    normalize_alpha(r0_s["alpha_rad"])[:, None],
                    normalize_beta(r0_s["beta_rad"])[:, None],
                ],
                axis=1,
            ),
            np.concatenate(
                [
                    minmax_normalize(v0_s["mag"][:, None], scales["v_mag_min"], scales["v_mag_max"]),
                    normalize_alpha(v0_s["alpha_rad"])[:, None],
                    normalize_beta(v0_s["beta_rad"])[:, None],
                ],
                axis=1,
            ),
            np.concatenate(
                [
                    minmax_normalize(rt_s["mag"][:, None], scales["r_mag_min"], scales["r_mag_max"]),
                    normalize_alpha(rt_s["alpha_rad"])[:, None],
                    normalize_beta(rt_s["beta_rad"])[:, None],
                ],
                axis=1,
            ),
            tof_n,
            nrev_n,
            ncase_n,
            prograde_n,
        ],
        axis=1,
    )
    return encoded, ENCODER_FEATURE_NAMES_SPHERICAL


def plot_vector_distributions(out_path: Path, series: list[tuple[str, dict[str, np.ndarray], str]]) -> None:
    fig, axes = plt.subplots(len(series), 3, figsize=(14, 3.6 * len(series)))
    if len(series) == 1:
        axes = np.expand_dims(axes, axis=0)
    cols = [
        ("mag", "Magnitude", None),
        ("alpha_deg", "Alpha (deg)", (-180.0, 180.0)),
        ("beta_deg", "Beta (deg)", (-90.0, 90.0)),
    ]
    for row, (label, sph, units) in enumerate(series):
        for col, (key, title, limits) in enumerate(cols):
            ax = axes[row, col]
            values = sph[key]
            ax.hist(values, bins=80, color="steelblue", alpha=0.8, edgecolor="white", lw=0.3)
            ax.axvline(np.mean(values), color="black", ls="--", lw=1.0)
            ax.axvline(np.median(values), color="darkgreen", ls=":", lw=1.0)
            if limits is not None:
                ax.set_xlim(*limits)
            ax.set_title(f"{label}: {title}")
            xlabel = title if units is None else f"{title} [{units}]"
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.25)
            stats = (
                f"mean={np.mean(values):.3f}\n"
                f"std={np.std(values):.3f}\n"
                f"p05={np.percentile(values, 5):.3f}\n"
                f"p95={np.percentile(values, 95):.3f}"
            )
            ax.text(
                0.98,
                0.95,
                stats,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
            )
    fig.suptitle("Spherical Vector Distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_vector_features(out_path: Path, series: list[tuple[str, dict[str, np.ndarray]]]) -> None:
    fig, axes = plt.subplots(len(series), 2, figsize=(12, 3.8 * len(series)))
    if len(series) == 1:
        axes = np.expand_dims(axes, axis=0)
    cols = [
        ("alpha_rad", "alpha / pi", normalize_alpha),
        ("beta_rad", "beta / (pi/2)", normalize_beta),
    ]
    for row, (label, sph) in enumerate(series):
        for col, (key, title, norm_fn) in enumerate(cols):
            ax = axes[row, col]
            values = norm_fn(sph[key])
            ax.hist(values, bins=80, color="coral", alpha=0.8, edgecolor="white", lw=0.3)
            ax.set_xlim(-1.05, 1.05)
            ax.set_title(f"{label}: {title}")
            ax.set_xlabel("scaled value")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.25)
    fig.suptitle("Scaled Raw Angular Features", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_scalar_inputs(out_path: Path, data: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    tof_min = np.asarray(data["tof"]).reshape(-1) / 60.0
    axes[0, 0].hist(tof_min, bins=80, color="slateblue", alpha=0.8, edgecolor="white", lw=0.3)
    axes[0, 0].set_title("TOF")
    axes[0, 0].set_xlabel("Minutes")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(True, alpha=0.25)

    nrev = np.asarray(data.get("nrev", np.zeros_like(tof_min))).reshape(-1)
    bins = np.arange(nrev.min() - 0.5, nrev.max() + 1.5, 1.0)
    axes[0, 1].hist(nrev, bins=bins, color="teal", alpha=0.85, edgecolor="white", lw=0.3)
    axes[0, 1].set_title("nrev")
    axes[0, 1].set_xlabel("Revolutions")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].grid(True, alpha=0.25)

    ncase = np.asarray(data.get("ncase", np.zeros_like(tof_min))).reshape(-1)
    case_counts = np.bincount(ncase.astype(int), minlength=int(max(ncase.max(initial=0), 1)) + 1)
    axes[1, 0].bar(np.arange(len(case_counts)), case_counts, color="goldenrod", alpha=0.85)
    axes[1, 0].set_title("ncase")
    axes[1, 0].set_xlabel("Case")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].grid(True, alpha=0.25, axis="y")

    prograde = np.asarray(data.get("prograde", np.zeros_like(tof_min))).reshape(-1)
    prog_counts = np.bincount(prograde.astype(int), minlength=2)
    axes[1, 1].bar([0, 1], prog_counts[:2], color=["indianred", "seagreen"], alpha=0.85)
    axes[1, 1].set_title("prograde")
    axes[1, 1].set_xlabel("0 = retrograde, 1 = prograde")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].grid(True, alpha=0.25, axis="y")

    fig.suptitle("Scalar and Branch Inputs", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_relationships(
    out_path: Path,
    data: dict[str, np.ndarray],
    r0_sph: dict[str, np.ndarray],
    v0_sph: dict[str, np.ndarray],
    rt_sph: dict[str, np.ndarray],
    dv1_sph: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    tof_min = np.asarray(data["tof"]).reshape(-1) / 60.0
    nrev = np.asarray(data.get("nrev", np.zeros_like(tof_min))).reshape(-1)

    sc = axes[0, 0].scatter(tof_min, dv1_sph["mag"], c=nrev, s=6, alpha=0.45, cmap="viridis")
    plt.colorbar(sc, ax=axes[0, 0], label="nrev")
    axes[0, 0].set_title("dv magnitude vs TOF")
    axes[0, 0].set_xlabel("TOF [min]")
    axes[0, 0].set_ylabel("|dv1| [km/s]")
    axes[0, 0].grid(True, alpha=0.25)

    sc = axes[0, 1].scatter(r0_sph["alpha_deg"], dv1_sph["alpha_deg"], c=tof_min, s=6, alpha=0.45, cmap="plasma")
    plt.colorbar(sc, ax=axes[0, 1], label="TOF [min]")
    axes[0, 1].set_title("dv alpha vs r0 alpha")
    axes[0, 1].set_xlabel("r0 alpha [deg]")
    axes[0, 1].set_ylabel("dv1 alpha [deg]")
    axes[0, 1].grid(True, alpha=0.25)

    sc = axes[0, 2].scatter(v0_sph["beta_deg"], dv1_sph["beta_deg"], c=nrev, s=6, alpha=0.45, cmap="cividis")
    plt.colorbar(sc, ax=axes[0, 2], label="nrev")
    axes[0, 2].set_title("dv beta vs v0 beta")
    axes[0, 2].set_xlabel("v0 beta [deg]")
    axes[0, 2].set_ylabel("dv1 beta [deg]")
    axes[0, 2].grid(True, alpha=0.25)

    axes[1, 0].scatter(r0_sph["mag"], rt_sph["mag"], s=6, alpha=0.35, color="steelblue")
    axes[1, 0].set_title("r_target magnitude vs r0 magnitude")
    axes[1, 0].set_xlabel("|r0| [km]")
    axes[1, 0].set_ylabel("|r_target| [km]")
    axes[1, 0].grid(True, alpha=0.25)

    delta_alpha = ((rt_sph["alpha_deg"] - r0_sph["alpha_deg"] + 180.0) % 360.0) - 180.0
    axes[1, 1].scatter(delta_alpha, dv1_sph["mag"], s=6, alpha=0.35, color="darkorange")
    axes[1, 1].set_title("dv magnitude vs target alpha change")
    axes[1, 1].set_xlabel("alpha(r_target) - alpha(r0) [deg]")
    axes[1, 1].set_ylabel("|dv1| [km/s]")
    axes[1, 1].grid(True, alpha=0.25)

    total_dv = np.asarray(data.get("total_dv", 2.0 * dv1_sph["mag"])).reshape(-1)
    axes[1, 2].scatter(dv1_sph["mag"], total_dv, s=6, alpha=0.35, color="mediumseagreen")
    axes[1, 2].set_title("total dv vs first-burn magnitude")
    axes[1, 2].set_xlabel("|dv1| [km/s]")
    axes[1, 2].set_ylabel("total dv [km/s]")
    axes[1, 2].grid(True, alpha=0.25)

    fig.suptitle("Dataset Relationships in Spherical Coordinates", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_scaled_inputs(out_path: Path, encoded: np.ndarray, feature_names: tuple[str, ...], state_repr: str) -> None:
    n_features = len(feature_names)
    ncols = 4
    nrows = int(np.ceil(n_features / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.3 * nrows))
    axes = np.atleast_2d(axes)
    flat_axes = axes.ravel()

    for i, (ax, name) in enumerate(zip(flat_axes, feature_names)):
        values = encoded[:, i]
        ax.hist(values, bins=80, color="mediumpurple", alpha=0.82, edgecolor="white", lw=0.3)
        ax.axvline(np.mean(values), color="black", ls="--", lw=1.0)
        ax.axvline(np.median(values), color="darkgreen", ls=":", lw=1.0)
        ax.set_xlim(-1.05, 1.05)
        ax.set_title(name)
        ax.set_xlabel("scaled value")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.25)
        stats = (
            f"min={np.min(values):.3f}\n"
            f"max={np.max(values):.3f}\n"
            f"mean={np.mean(values):.3f}\n"
            f"std={np.std(values):.3f}"
        )
        ax.text(
            0.98,
            0.95,
            stats,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
        )

    for ax in flat_axes[n_features:]:
        ax.axis("off")

    fig.suptitle(f"Scaled Encoder Inputs ({state_repr})", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Lambert dataset in spherical coordinates.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=THIS_DIR / "data" / "lambert_train.npz",
        help="Path to dataset .npz file",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=THIS_DIR / "plots",
        help="Directory to save figures",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional evenly-spaced sample cap for faster plotting",
    )
    parser.add_argument(
        "--scale_dataset",
        type=Path,
        default=None,
        help="Dataset used to compute min/max scaling stats. Defaults to --dataset.",
    )
    parser.add_argument(
        "--state_repr",
        choices=["cartesian", "spherical"],
        default="spherical",
        help="Encoder representation to visualize after scaling.",
    )
    parser.add_argument("--use_tof_log", dest="use_tof_log", action="store_true")
    parser.add_argument("--no_tof_log", dest="use_tof_log", action="store_false")
    parser.set_defaults(use_tof_log=True)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.dataset, limit=args.limit if args.limit > 0 else None)
    scale_dataset = args.scale_dataset if args.scale_dataset is not None else args.dataset
    scale_data = load_dataset(scale_dataset, limit=None)
    r0_sph = vector_to_spherical(np.asarray(data["r0"]))
    v0_sph = vector_to_spherical(np.asarray(data["v0"]))
    rt_sph = vector_to_spherical(np.asarray(data["r_target"]))
    dv1_sph = vector_to_spherical(np.asarray(data["dv1"]))
    scales = compute_scales(scale_data, use_tof_log=args.use_tof_log)
    encoded, feature_names = encode_scaled_inputs(data, scales, state_repr=args.state_repr)

    vector_series = [
        ("r0", r0_sph, "km"),
        ("v0", v0_sph, "km/s"),
        ("r_target", rt_sph, "km"),
        ("dv1", dv1_sph, "km/s"),
    ]
    feature_series = [
        ("r0", r0_sph),
        ("v0", v0_sph),
        ("r_target", rt_sph),
        ("dv1", dv1_sph),
    ]

    stem = args.dataset.stem
    plot_vector_distributions(out_dir / f"{stem}_spherical_vectors.png", vector_series)
    plot_vector_features(out_dir / f"{stem}_spherical_features.png", feature_series)
    plot_scalar_inputs(out_dir / f"{stem}_spherical_scalars.png", data)
    plot_relationships(out_dir / f"{stem}_spherical_relationships.png", data, r0_sph, v0_sph, rt_sph, dv1_sph)
    plot_scaled_inputs(out_dir / f"{stem}_scaled_inputs_{args.state_repr}.png", encoded, feature_names, args.state_repr)

    print(f"Loaded {len(data['r0'])} samples from {args.dataset}")
    print(f"Saved: {out_dir / f'{stem}_spherical_vectors.png'}")
    print(f"Saved: {out_dir / f'{stem}_spherical_features.png'}")
    print(f"Saved: {out_dir / f'{stem}_spherical_scalars.png'}")
    print(f"Saved: {out_dir / f'{stem}_spherical_relationships.png'}")
    print(f"Saved: {out_dir / f'{stem}_scaled_inputs_{args.state_repr}.png'}")


if __name__ == "__main__":
    main()
