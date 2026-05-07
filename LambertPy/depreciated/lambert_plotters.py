"""Shared plotting utilities for Lambert TRC scripts."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dynamics import propagate_j2, propagate_twobody


def plot_data_overview(train_path: Path, out_dir: Path) -> None:
    d = np.load(train_path)
    n_show = len(d['r0'])
    h = np.cross(d['r0'], d['v0'])
    h_norm = np.linalg.norm(h, axis=1)
    inc_deg = np.degrees(np.arccos(np.clip(h[:, 2] / np.maximum(h_norm, 1e-12), -1.0, 1.0)))

    fig = plt.figure(figsize=(16, 10))
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    xyz_all = []
    cmap_inc = plt.cm.plasma
    norm_inc = plt.Normalize(vmin=0.0, vmax=90.0)
    for i in range(n_show):
        r0 = d['r0'][i]
        v0 = d['v0'][i]
        dv1 = d['dv1'][i]
        tof = float(d['tof'][i])
        _, _, traj = propagate_twobody(r0, v0 + dv1, tof, n_steps=500)
        xyz_all.append(traj[:, :3])
        c = cmap_inc(norm_inc(inc_deg[i]))
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.10, lw=0.5, color=c)
        ax1.scatter(*r0, c='green', s=4, zorder=5)
        ax1.scatter(*d['r_target'][i], c='red', s=4, zorder=5)
    if xyz_all:
        xyz = np.concatenate(xyz_all, axis=0)
        lo = np.percentile(xyz, 5, axis=0)
        hi = np.percentile(xyz, 95, axis=0)
        ctr = 0.5 * (lo + hi)
        rad = 0.5 * np.max(hi - lo)
        rad = max(float(rad), 1.0)
        ax1.set_xlim(ctr[0] - rad, ctr[0] + rad)
        ax1.set_ylim(ctr[1] - rad, ctr[1] + rad)
        ax1.set_zlim(ctr[2] - rad, ctr[2] + rad)
        if hasattr(ax1, 'set_box_aspect'):
            ax1.set_box_aspect((1, 1, 1))
    ax1.view_init(elev=25, azim=45)
    cbar_inc = plt.colorbar(plt.cm.ScalarMappable(norm=norm_inc, cmap=cmap_inc), ax=ax1, pad=0.08)
    cbar_inc.set_label('Inclination (deg)')
    ax1.set(xlabel='X', ylabel='Y', zlabel='Z', title='Transfer orbits')

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.hist(d['dv1_mag'], bins=30, alpha=0.6, label='|dV1|')
    ax2.hist(d['dv2_mag'], bins=30, alpha=0.6, label='|dV2|')
    ax2.set(xlabel='dV (km/s)', ylabel='Count', title='dV Distribution')
    ax2.legend()

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.hist(d['tof'] / 60, bins=30, alpha=0.7)
    ax3.set(xlabel='TOF (min)', ylabel='Count', title='Transfer Time')

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.hist(inc_deg, bins=30, alpha=0.75, color='teal')
    ax4.set(xlabel='Inclination (deg)', ylabel='Count', title='Inclination Distribution')

    ax5 = fig.add_subplot(2, 3, 5)
    sc = ax5.scatter(d['tof'] / 60, d['total_dv'], c=d['dv1_mag'], cmap='viridis', s=5, alpha=0.6)
    plt.colorbar(sc, ax=ax5, label='|dV1|')
    ax5.set(xlabel='TOF (min)', ylabel='Total dV', title='dV vs TOF')

    ax6 = fig.add_subplot(2, 3, 6)
    j2e = np.maximum(d['pos_err_j2'], 1e-12)
    sc2 = ax6.scatter(d['tof'] / 60, j2e, c=d['total_dv'], cmap='viridis', s=5, alpha=0.6)
    plt.colorbar(sc2, ax=ax6, label='Total dV')
    ax6.set_yscale('log')
    ax6.set(xlabel='TOF (min)', ylabel='J2 pos err (km)', title='J2 Mismatch (dataset diagnostic)')

    plt.tight_layout()
    plt.savefig(out_dir / 'lambert_data_overview.png', dpi=150)
    plt.close(fig)
    print(f'Inclination range: {inc_deg.min():.2f} to {inc_deg.max():.2f} deg '
          f'(mean {inc_deg.mean():.2f} deg)')
    print('Data visualization saved.')


def plot_learning_curves(history: dict, out_dir: Path) -> None:
    n_ep = len(history['train_loss'])
    ep = np.arange(1, n_ep + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.semilogy(ep, history['train_loss'], 'b-', alpha=0.7, label='Train Total Loss')
    ax1.semilogy(ep, history['val_loss'], 'b--', alpha=0.7, label='Val Total Loss')
    ax1.set(xlabel='Epoch', ylabel='Total Loss', title='Training Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    train_pos_m = np.array(history['train_pos_err']) * 1000.0
    val_pos_m = np.array(history['val_pos_err']) * 1000.0
    ax2.plot(ep, train_pos_m, 'r-', alpha=0.7, label='Train pos err')
    ax2.plot(ep, val_pos_m, 'r--', alpha=0.7, label='Val pos err')
    ax2b = ax2.twinx()
    ax2b.plot(ep, history['train_imp'], 'g-', alpha=0.7, label='Train Imp')
    ax2b.plot(ep, history['val_imp'], 'g--', alpha=0.7, label='Val Imp')
    ax2.set(xlabel='Epoch', ylabel='Position Error (m)', title='Position Error and Improvement')
    ax2b.set_ylabel('Improvement Metric', color='green')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'lambert_training_curves.png', dpi=150)
    plt.close(fig)


def plot_arrival_error_comparison(train_eval: dict, val_eval: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(train_eval['expert_roll_pos_err'], train_eval['pos_errs'][-1], s=10, alpha=0.35, label='Train')
    ax.scatter(val_eval['expert_roll_pos_err'], val_eval['pos_errs'][-1], s=14, alpha=0.7, label='Validation')
    ax.set(
        xlabel='Lambert (J2) arrival error [km]',
        ylabel='TRC (J2) arrival error [km]',
        title='Arrival Error: Lambert vs TRC (J2), All Samples',
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'lambert_vs_trc_arrival_error.png', dpi=150)
    plt.close(fig)


def plot_dv_change_all(train_eval: dict, val_eval: dict, out_dir: Path) -> None:
    train_delta = train_eval['dv_iters'][-1] - train_eval['all_dv1']
    val_delta = val_eval['dv_iters'][-1] - val_eval['all_dv1']
    train_mag = np.linalg.norm(train_delta, axis=-1) * 1000.0
    val_mag = np.linalg.norm(val_delta, axis=-1) * 1000.0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.hist(train_mag, bins=50, alpha=0.45, label='Train')
    ax1.hist(val_mag, bins=50, alpha=0.65, label='Validation')
    ax1.set(xlabel='|Δv_TRC - Δv_Lambert| [m/s]', ylabel='Count', title='Residual Burn Magnitude')
    ax1.legend()
    ax1.grid(alpha=0.3)
    for c, label in enumerate(['dVx', 'dVy', 'dVz']):
        ax2.hist(train_delta[:, c] * 1000.0, bins=50, histtype='step', lw=1.5, alpha=0.8, label=f'Train {label}')
        ax2.hist(val_delta[:, c] * 1000.0, bins=50, histtype='step', lw=1.5, ls='--', alpha=0.9, label=f'Val {label}')
    ax2.set(xlabel='Residual component [m/s]', ylabel='Count', title='Residual Components Distribution')
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'lambert_trc_dv_change_all.png', dpi=150)
    plt.close(fig)


def plot_trajectory_results(eval_data: dict, out_dir: Path) -> None:
    all_r0 = eval_data['all_r0']
    all_v0 = eval_data['all_v0']
    all_rt = eval_data['all_rt']
    all_tof = eval_data['all_tof']
    all_dv1 = eval_data['all_dv1']
    dv_iters = eval_data['dv_iters']
    pos_errs = eval_data['pos_errs']
    n_show = min(15, len(all_dv1))
    fig = plt.figure(figsize=(16, 10))
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    for i in range(n_show):
        r0_np = all_r0[i].cpu().numpy()
        v0_np = all_v0[i].cpu().numpy()
        dv_np = dv_iters[-1][i]
        tof_np = float(all_tof[i].cpu())
        _, _, traj = propagate_j2(r0_np, v0_np + dv_np, tof_np, n_steps=300)
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.5, lw=0.7)
        ax1.scatter(*r0_np, c='green', s=10, zorder=5)
        ax1.scatter(*all_rt[i].cpu().numpy(), c='red', s=10, zorder=5, marker='x')
    ax1.set(xlabel='X (km)', ylabel='Y (km)', zlabel='Z (km)', title='TRC Transfer Trajectories')
    ax2 = fig.add_subplot(2, 2, 2)
    trc_pe = pos_errs[-1]
    ax2.scatter(np.arange(len(trc_pe)), trc_pe, s=10, alpha=0.7, label='TRC')
    ax2.axhline(np.mean(trc_pe), c='red', ls='--', label=f'Mean: {np.mean(trc_pe):.1f} km')
    ax2.set(xlabel='Sample', ylabel='Position error (km)', title='Arrival Position Error')
    ax2.legend()
    expert_dv_mag = np.linalg.norm(all_dv1, axis=-1)
    pred_dv_mag = np.linalg.norm(dv_iters[-1], axis=-1)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(expert_dv_mag, pred_dv_mag, s=15, alpha=0.6)
    lim = max(expert_dv_mag.max(), pred_dv_mag.max()) * 1.1
    ax3.plot([0, lim], [0, lim], 'r--', alpha=0.5)
    ax3.set(xlabel='Expert |dV1| (km/s)', ylabel='TRC |dV1| (km/s)', title='dV Magnitude: Expert vs TRC')
    ax3.set_aspect('equal')
    ax4 = fig.add_subplot(2, 2, 4)
    for c, label in enumerate(['dVx', 'dVy', 'dVz']):
        ax4.scatter(all_dv1[:, c], dv_iters[-1][:, c], s=8, alpha=0.5, label=label)
    lim = max(abs(all_dv1).max(), abs(dv_iters[-1]).max()) * 1.1
    ax4.plot([-lim, lim], [-lim, lim], 'r--', alpha=0.5)
    ax4.set(xlabel='Expert dV (km/s)', ylabel='TRC dV (km/s)', title='dV Components: Expert vs TRC')
    ax4.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'lambert_trajectory_results.png', dpi=150)
    plt.close(fig)


def plot_refinement_analysis(eval_data: dict, out_dir: Path) -> None:
    pos_errs = eval_data['pos_errs']
    colors = ['#7b2d8e', '#2b8f8f', '#4ca64c', '#d4b830']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for k in range(len(pos_errs)):
        color = colors[k] if k < len(colors) else f'C{k}'
        ax1.hist(pos_errs[k], bins=30, alpha=0.4, color=color, label=f'Iter {k}')
    ax1.set(xlabel='Position Error (km)', ylabel='Count', title='(a) Position Error Distribution')
    ax1.legend()
    mean_pe = [pe.mean() for pe in pos_errs]
    std_pe = [pe.std() for pe in pos_errs]
    colors_bar = colors[:len(mean_pe)] + [f'C{i}' for i in range(len(mean_pe) - len(colors))]
    ax2.bar(range(len(mean_pe)), mean_pe, yerr=std_pe, color=colors_bar[:len(mean_pe)],
            alpha=0.8, capsize=5, edgecolor='k', lw=0.5)
    ax2.set(xlabel='Iteration', ylabel='Mean Position Error (km)', title='(b) Error Reduction')
    if mean_pe[0] > 0:
        red = (1 - mean_pe[-1] / mean_pe[0]) * 100
        ax2.annotate(f'{red:.0f}% reduction', xy=(len(mean_pe) - 1, mean_pe[-1]),
                     xytext=(len(mean_pe) - 1.5, mean_pe[0] * 0.6),
                     arrowprops=dict(arrowstyle='->', color='red'), color='red', fontweight='bold')
    fig.suptitle('Iterative Refinement Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / 'lambert_refinement.png', dpi=150)
    plt.close(fig)


def plot_latent_space(eval_data: dict, out_dir: Path) -> None:
    z_H = eval_data['z_H']
    pos_errs = eval_data['pos_errs']
    all_z = np.concatenate(z_H, axis=0)
    mean_z = all_z.mean(axis=0)
    centered = all_z - mean_z
    _, svals, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    var_exp = svals[:2] ** 2 / (svals ** 2).sum()
    z_2d = [(z - mean_z) @ basis for z in z_H]
    K = len(z_2d) - 1
    colors = ['#7b2d8e', '#2b8f8f', '#4ca64c', '#d4b830']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for k in range(K + 1):
        color = colors[k] if k < len(colors) else f'C{k}'
        ax1.scatter(z_2d[k][:, 0], z_2d[k][:, 1], c=color, s=20, alpha=0.5, label=f'Iter {k}')
        cx, cy = z_2d[k].mean(0)
        ax1.plot(cx, cy, 'x', color=color, ms=10, mew=2)
    ax1.set(xlabel=f'PC1 ({var_exp[0]*100:.1f}%)', ylabel=f'PC2 ({var_exp[1]*100:.1f}%)',
            title='Latent Space (colored by iteration)')
    ax1.legend()
    final_pe = pos_errs[-1]
    norm = plt.Normalize(final_pe.min(), final_pe.max())
    cmap_cost = plt.cm.RdYlGn_r
    for i in range(min(len(final_pe), 200)):
        px = [z_2d[k][i, 0] for k in range(K + 1)]
        py = [z_2d[k][i, 1] for k in range(K + 1)]
        ax2.plot(px, py, '-', color=cmap_cost(norm(final_pe[i])), alpha=0.4, lw=0.8)
        ax2.plot(px[0], py[0], 'o', color=cmap_cost(norm(final_pe[i])), ms=3)
        ax2.plot(px[-1], py[-1], 's', color=cmap_cost(norm(final_pe[i])), ms=4)
    sm = plt.cm.ScalarMappable(cmap=cmap_cost, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax2, label='Final Pos Error (km)')
    ax2.set(xlabel=f'PC1 ({var_exp[0]*100:.1f}%)', ylabel=f'PC2 ({var_exp[1]*100:.1f}%)',
            title='Refinement Paths (colored by error)')
    fig.suptitle('Latent Space Evolution', fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / 'lambert_latent_space.png', dpi=150)
    plt.close(fig)


def plot_single_sample(eval_data: dict, out_dir: Path, idx: int = 0) -> None:
    all_r0 = eval_data['all_r0']
    all_v0 = eval_data['all_v0']
    all_rt = eval_data['all_rt']
    all_tof = eval_data['all_tof']
    all_dv1 = eval_data['all_dv1']
    dv_iters = eval_data['dv_iters']
    r0_1 = all_r0[idx:idx + 1].cpu().numpy()[0]
    v0_1 = all_v0[idx:idx + 1].cpu().numpy()[0]
    rt_1 = all_rt[idx].cpu().numpy()
    tof_1 = float(all_tof[idx].cpu())
    dv_expert = all_dv1[idx]
    dv_trc = dv_iters[-1][idx]
    _, _, traj_expert_tb = propagate_twobody(r0_1, v0_1 + dv_expert, tof_1, n_steps=500)
    _, _, traj_expert_j2 = propagate_j2(r0_1, v0_1 + dv_expert, tof_1, n_steps=500)
    _, _, traj_trc = propagate_j2(r0_1, v0_1 + dv_trc, tof_1, n_steps=500)
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.plot(traj_expert_tb[:, 0], traj_expert_tb[:, 1], traj_expert_tb[:, 2], 'b-', lw=1.5, label='Lambert (2body)')
    ax1.plot(traj_expert_j2[:, 0], traj_expert_j2[:, 1], traj_expert_j2[:, 2], 'b--', lw=1, alpha=0.5, label='Lambert (J2)')
    ax1.plot(traj_trc[:, 0], traj_trc[:, 1], traj_trc[:, 2], 'r--', lw=1.5, label='TRC')
    ax1.scatter(*traj_expert_j2[-1, :3], c='blue', s=40, marker='o', zorder=6, label='Lambert J2 final')
    ax1.scatter(*traj_trc[-1, :3], c='red', s=40, marker='o', zorder=6, label='TRC final')
    ax1.plot([traj_trc[-1, 0], rt_1[0]], [traj_trc[-1, 1], rt_1[1]], [traj_trc[-1, 2], rt_1[2]],
             color='red', lw=1.2, alpha=0.8, label='TRC miss')
    ax1.scatter(*r0_1, c='green', s=50, zorder=5, label='Start')
    ax1.scatter(*rt_1, c='red', s=50, zorder=5, marker='x', label='Target')
    xyz = np.vstack([traj_expert_tb[:, :3], traj_expert_j2[:, :3], traj_trc[:, :3], r0_1.reshape(1, 3), rt_1.reshape(1, 3)])
    mins = xyz.min(axis=0); maxs = xyz.max(axis=0)
    centers = 0.5 * (mins + maxs); radius = 0.5 * (maxs - mins).max()
    ax1.set_xlim(centers[0] - radius, centers[0] + radius)
    ax1.set_ylim(centers[1] - radius, centers[1] + radius)
    ax1.set_zlim(centers[2] - radius, centers[2] + radius)
    if hasattr(ax1, 'set_box_aspect'):
        ax1.set_box_aspect((1, 1, 1))
    ax1.legend(fontsize=8)
    ax1.set_title(f'Sample {idx}: 3D Trajectory')
    ax2 = fig.add_subplot(1, 3, 2)
    time_ax = np.linspace(0, tof_1, len(traj_expert_tb))
    ax2.plot(time_ax, np.linalg.norm(traj_expert_tb[:, :3] - rt_1, axis=1), 'b-', label='Lambert (2body)')
    ax2.plot(time_ax, np.linalg.norm(traj_expert_j2[:, :3] - rt_1, axis=1), 'b--', alpha=0.5, label='Lambert (J2)')
    ax2.plot(time_ax, np.linalg.norm(traj_trc[:, :3] - rt_1, axis=1), 'r--', label='TRC')
    ax2.set(xlabel='Time (s)', ylabel='Distance to target (km)', title='Range to Target')
    ax2.legend(); ax2.grid(alpha=0.3)
    ax3 = fig.add_subplot(1, 3, 3)
    labels = ['dVx', 'dVy', 'dVz']; x_pos = np.arange(3)
    ax3.bar(x_pos - 0.2, dv_expert, 0.35, label='Expert', color='blue', alpha=0.7)
    ax3.bar(x_pos + 0.2, dv_trc, 0.35, label='TRC', color='red', alpha=0.7)
    ax3.set_xticks(x_pos); ax3.set_xticklabels(labels)
    ax3.set(ylabel='dV (km/s)', title='dV Components')
    ax3.legend(); ax3.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'lambert_single_sample.png', dpi=150)
    plt.close(fig)
    print(f'Expert dV: [{dv_expert[0]:+.4f}, {dv_expert[1]:+.4f}, {dv_expert[2]:+.4f}] km/s  |dV|={np.linalg.norm(dv_expert):.4f}')
    print(f'TRC    dV: [{dv_trc[0]:+.4f}, {dv_trc[1]:+.4f}, {dv_trc[2]:+.4f}] km/s  |dV|={np.linalg.norm(dv_trc):.4f}')
    print(f'Lambert (2body) arrival error: {np.linalg.norm(traj_expert_tb[-1, :3] - rt_1):.2f} km')
    print(f'Lambert (J2)    arrival error: {np.linalg.norm(traj_expert_j2[-1, :3] - rt_1):.2f} km')
    print(f'TRC arrival error:             {np.linalg.norm(traj_trc[-1, :3] - rt_1):.2f} km')


def plot_u0_training_curves(history: dict, out_path: Path):
    if len(history.get("train_loss", [])) == 0:
        return
    ep = np.arange(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.semilogy(ep, history["train_loss"], "b-", label="train loss")
    ax1.semilogy(ep, history["val_loss"], "b--", label="val loss")
    ax1.set(xlabel="Epoch", ylabel="Loss", title="U0-SelfSup Training Curves")
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax2.plot(ep, history["train_pos"], "r-", alpha=0.75, label="train posK")
    ax2.plot(ep, history["val_pos"], "r--", alpha=0.85, label="val posK")
    ax2b = ax2.twinx()
    ax2b.plot(ep, history["train_imp"], "g-", alpha=0.75, label="train imp")
    ax2b.plot(ep, history["val_imp"], "g--", alpha=0.85, label="val imp")
    ax2.set(xlabel="Epoch", ylabel="posK (km)", title="Position Error and Improvement")
    ax2b.set_ylabel("Improvement", color="green")
    ax2.grid(alpha=0.3)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_u0_eval_summary(eval_data: dict, out_path: Path, split_name: str = "test"):
    pe = eval_data["pos_err_k"]
    u0r = eval_data["u0_res_m"]
    dvr = eval_data["dv_res_m"]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))
    axs[0].hist(pe, bins=40, alpha=0.75, color="tab:red")
    axs[0].set(xlabel="Position Error K (km)", ylabel="Count", title=f"{split_name}: posK")
    axs[0].grid(alpha=0.3)
    axs[1].hist(dvr, bins=40, alpha=0.75, color="tab:green", label="dv_res")
    axs[1].hist(u0r, bins=40, histtype="step", lw=1.5, color="tab:blue", label="u0_res")
    axs[1].set(xlabel="Residual wrt Lambert (m/s)", ylabel="Count", title=f"{split_name}: Residuals")
    axs[1].legend()
    axs[1].grid(alpha=0.3)
    sc = axs[2].scatter(dvr, pe, s=8, alpha=0.5, c=u0r, cmap="viridis")
    plt.colorbar(sc, ax=axs[2], label="u0_res (m/s)")
    axs[2].set(xlabel="dv_res (m/s)", ylabel="posK (km)", title=f"{split_name}: posK vs dv_res")
    axs[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_u0_arrival_error_comparison(train_eval: dict, test_eval: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(train_eval["lambert_j2_err"], train_eval["pos_err_k"], s=10, alpha=0.30, label="Train")
    ax.scatter(test_eval["lambert_j2_err"], test_eval["pos_err_k"], s=14, alpha=0.70, label="Test")
    ax.set(
        xlabel="Lambert (J2) arrival error [km]",
        ylabel="TRC (J2) arrival error [km]",
        title="Arrival Error: Lambert vs TRC (J2), All Samples",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
