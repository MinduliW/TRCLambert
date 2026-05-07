"""Plot full 6-panel Lambert data summary using all train+test trajectories."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from constants import MU_EARTH
from dynamics import propagate_twobody


def _load_npz(path: Path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def _collect_trajectories(d: dict, n_steps: int, label: str):
    n = len(d["r0"])
    xyz_all = []
    print(f"[{label}] propagating {n} trajectories")
    for i in range(n):
        r0 = d["r0"][i]
        v0 = d["v0"][i]
        dv1 = d["dv1"][i]
        tof = float(d["tof"][i])
        _, _, traj = propagate_twobody(r0, v0 + dv1, tof, n_steps=n_steps)
        xyz_all.append(traj[:, :3])
        if (i + 1) % max(1, n // 10) == 0:
            print(f"  [{label}] {i + 1}/{n}")
    return xyz_all


def _semi_major_axis(r, v, mu=MU_EARTH):
    r_norm = np.linalg.norm(r, axis=1)
    v2 = np.sum(v * v, axis=1)
    inv_a = 2.0 / np.maximum(r_norm, 1e-12) - v2 / mu
    a = np.full_like(inv_a, np.nan, dtype=np.float64)
    mask = np.abs(inv_a) > 1e-12
    a[mask] = 1.0 / inv_a[mask]
    return a


def _inclination_deg(r, v):
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h, axis=1)
    cos_i = np.clip(h[:, 2] / np.maximum(h_norm, 1e-12), -1.0, 1.0)
    return np.degrees(np.arccos(cos_i))


def _eccentricity(r, v, mu=MU_EARTH):
    r_norm = np.linalg.norm(r, axis=1, keepdims=True)
    v_norm = np.linalg.norm(v, axis=1, keepdims=True)
    rv = np.sum(r * v, axis=1, keepdims=True)
    e_vec = ((v_norm ** 2 - mu / np.maximum(r_norm, 1e-12)) * r - rv * v) / mu
    return np.linalg.norm(e_vec, axis=1)


def plot_all_lambert_summary(
    train_path: Path,
    test_path: Path,
    out_path: Path,
    n_steps: int = 250,
) -> None:
    """
    Plot 6-panel summary with all Lambert transfer trajectories.
    """
    train = _load_npz(train_path)
    test = _load_npz(test_path)
    train_xyz = _collect_trajectories(train, n_steps=n_steps, label="train")
    test_xyz = _collect_trajectories(test, n_steps=n_steps, label="test")

    fig = plt.figure(figsize=(18, 13))
    ax1 = fig.add_subplot(3, 3, 1, projection="3d")

    for traj in train_xyz:
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="tab:blue", alpha=0.03, lw=0.35)
        ax1.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color="limegreen", s=2, alpha=0.25)
        ax1.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color="crimson", s=2, alpha=0.25)
    for traj in test_xyz:
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="tab:orange", alpha=0.08, lw=0.5)
        ax1.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color="limegreen", s=2, alpha=0.30)
        ax1.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color="crimson", s=2, alpha=0.30)

    xyz = np.concatenate(train_xyz + test_xyz, axis=0)
    lo = np.percentile(xyz, 3, axis=0)
    hi = np.percentile(xyz, 97, axis=0)
    ctr = 0.5 * (lo + hi)
    rad = max(float(0.5 * np.max(hi - lo)), 1.0)
    ax1.set_xlim(ctr[0] - rad, ctr[0] + rad)
    ax1.set_ylim(ctr[1] - rad, ctr[1] + rad)
    ax1.set_zlim(ctr[2] - rad, ctr[2] + rad)
    if hasattr(ax1, "set_box_aspect"):
        ax1.set_box_aspect((1, 1, 1))
    ax1.view_init(elev=25, azim=45)
    ax1.set(
        xlabel="X (km)",
        ylabel="Y (km)",
        zlabel="Z (km)",
        title=f"All Transfer Orbits (train={len(train_xyz)}, test={len(test_xyz)})",
    )
    ax1.plot([], [], [], color="tab:blue", lw=1.5, label="Train")
    ax1.plot([], [], [], color="tab:orange", lw=1.5, label="Test")
    ax1.scatter([], [], [], color="limegreen", s=20, label="Start")
    ax1.scatter([], [], [], color="crimson", s=20, label="End")
    ax1.legend(loc="upper left")

    ax2 = fig.add_subplot(3, 3, 2)
    ax2.hist(train["dv1_mag"], bins=40, alpha=0.5, label="|dV1| train")
    ax2.hist(train["dv2_mag"], bins=40, alpha=0.5, label="|dV2| train")
    ax2.hist(test["dv1_mag"], bins=40, histtype="step", lw=1.5, label="|dV1| test")
    ax2.hist(test["dv2_mag"], bins=40, histtype="step", lw=1.5, label="|dV2| test")
    ax2.set(xlabel="dV (km/s)", ylabel="Count", title="dV Distribution")
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(3, 3, 3)
    ax3.hist(train["tof"] / 60.0, bins=40, alpha=0.5, label="Train")
    ax3.hist(test["tof"] / 60.0, bins=40, alpha=0.5, label="Test")
    ax3.set(xlabel="TOF (min)", ylabel="Count", title="Transfer Time")
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(3, 3, 4)
    if "transfer_angle" in train and "transfer_angle" in test:
        ax4.hist(np.degrees(train["transfer_angle"]), bins=40, alpha=0.5, label="Train")
        ax4.hist(np.degrees(test["transfer_angle"]), bins=40, alpha=0.5, label="Test")
    ax4.set(xlabel="Transfer angle (deg)", ylabel="Count", title="Transfer Geometry")
    ax4.legend(fontsize=8)

    ax5 = fig.add_subplot(3, 3, 5)
    sc = ax5.scatter(
        train["tof"] / 60.0,
        train["total_dv"],
        c=np.degrees(train["transfer_angle"]) if "transfer_angle" in train else train["dv1_mag"],
        cmap="viridis",
        s=5,
        alpha=0.5,
    )
    plt.colorbar(sc, ax=ax5, label="theta (deg)" if "transfer_angle" in train else "|dV1|")
    ax5.set(xlabel="TOF (min)", ylabel="Total dV (km/s)", title="dV vs TOF (train)")

    ax6 = fig.add_subplot(3, 3, 6)
    if "pos_err_j2" in train:
        j2e = np.maximum(train["pos_err_j2"], 1e-12)
        sc2 = ax6.scatter(
            train["tof"] / 60.0,
            j2e,
            c=train["total_dv"],
            cmap="viridis",
            s=5,
            alpha=0.5,
        )
        plt.colorbar(sc2, ax=ax6, label="Total dV (km/s)")
        ax6.set_yscale("log")
    ax6.set(xlabel="TOF (min)", ylabel="J2 pos error (km)", title="J2 Mismatch (train)")

    # Additional orbital-change diagnostics.
    a0_tr = _semi_major_axis(train["r0"], train["v0"])
    a1_tr = _semi_major_axis(train["r_target"], train["v_target"])
    a0_te = _semi_major_axis(test["r0"], test["v0"])
    a1_te = _semi_major_axis(test["r_target"], test["v_target"])
    da_tr = a1_tr - a0_tr
    da_te = a1_te - a0_te

    i0_tr = _inclination_deg(train["r0"], train["v0"])
    i1_tr = _inclination_deg(train["r_target"], train["v_target"])
    i0_te = _inclination_deg(test["r0"], test["v0"])
    i1_te = _inclination_deg(test["r_target"], test["v_target"])
    di_tr = i1_tr - i0_tr
    di_te = i1_te - i0_te

    e0_tr = _eccentricity(train["r0"], train["v0"])
    e1_tr = _eccentricity(train["r_target"], train["v_target"])
    e0_te = _eccentricity(test["r0"], test["v0"])
    e1_te = _eccentricity(test["r_target"], test["v_target"])
    de_tr = e1_tr - e0_tr
    de_te = e1_te - e0_te

    ax7 = fig.add_subplot(3, 3, 7)
    ax7.hist(da_tr[np.isfinite(da_tr)], bins=50, alpha=0.5, label="Train")
    ax7.hist(da_te[np.isfinite(da_te)], bins=50, alpha=0.5, label="Test")
    ax7.set(xlabel="Δa (km)", ylabel="Count", title="SMA Change")
    ax7.legend(fontsize=8)

    ax8 = fig.add_subplot(3, 3, 8)
    ax8.hist(di_tr, bins=50, alpha=0.5, label="Train")
    ax8.hist(di_te, bins=50, alpha=0.5, label="Test")
    ax8.set(xlabel="Δi (deg)", ylabel="Count", title="Inclination Change")
    ax8.legend(fontsize=8)

    ax9 = fig.add_subplot(3, 3, 9)
    ax9.hist(de_tr, bins=50, alpha=0.5, label="Train")
    ax9.hist(de_te, bins=50, alpha=0.5, label="Test")
    ax9.set(xlabel="Δe", ylabel="Count", title="Eccentricity Change")
    ax9.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot 6-panel Lambert data summary using all train+test trajectories."
    )
    parser.add_argument("--train_path", type=Path, default=Path("data/lambert_train.npz"))
    parser.add_argument("--test_path", type=Path, default=Path("data/lambert_test.npz"))
    parser.add_argument("--out_path", type=Path, default=Path("all_lambert_summary.png"))
    parser.add_argument("--n_steps", type=int, default=250, help="RK4 propagation steps per trajectory")
    args = parser.parse_args()

    if not args.train_path.exists():
        raise FileNotFoundError(f"Missing train dataset: {args.train_path}")
    if not args.test_path.exists():
        raise FileNotFoundError(f"Missing test dataset: {args.test_path}")

    plot_all_lambert_summary(
        train_path=args.train_path,
        test_path=args.test_path,
        out_path=args.out_path,
        n_steps=args.n_steps,
    )


if __name__ == "__main__":
    main()
