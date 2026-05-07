"""Train and evaluate LambertTRC on pre-generated Lambert datasets.

Expected inputs:
- data/lambert_train.npz
- data/lambert_test.npz

Generate datasets first via:
    python lambert_data.py
"""

import argparse
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

from lambert_trc_model import (
    LambertDataset,
    LambertJ2Loss,
    LambertResidualJ2Loss,
    LambertTRC,
)
from trc import NetConfig, count_params
import lambert_plotters as plotters


@dataclass
class RunConfig:
    quick: bool = False
    do_train: bool = True
    resume: bool = False
    lambert_only: bool = False
    data_dir: Path = THIS_DIR / 'data'
    ckpt_dir: Path = THIS_DIR / 'checkpoints'
    dv_max: float = 3.0
    lr: float = 5e-4
    pos_scale_m: float = 1000.0
    lambda_ps: float = 0.05
    lambda_sup: float = 0.5
    lambda_dv: float = 0.01
    eval_batch_size: int = 256
    eval_max_samples: int = 0

    @property
    def epochs(self) -> int:
        return 15 if self.quick else 300

    @property
    def batch_size(self) -> int:
        return 32 if self.quick else 64

    @property
    def n_coast(self) -> int:
        return 100 if self.quick else 200

    @property
    def net_cfg(self) -> NetConfig:
        if self.quick:
            return NetConfig(d_z=128, d_h=256, n_heads=4, n_blocks=2, K=4, n_inner=6)
        return NetConfig(d_z=256, d_h=512, n_heads=8, n_blocks=3, K=4, n_inner=6)

    @property
    def train_path(self) -> Path:
        return self.data_dir / 'lambert_train.npz'

    @property
    def test_path(self) -> Path:
        return self.data_dir / 'lambert_test.npz'

    @property
    def ckpt_path(self) -> Path:
        if self.lambert_only:
            return self.ckpt_dir / 'trc_lambert_residual_best.pt'
        return self.ckpt_dir / 'trc_lambert_best.pt'


def parse_args() -> RunConfig:
    base = RunConfig()
    parser = argparse.ArgumentParser(description='Train/evaluate LambertTRC from prebuilt datasets.')
    parser.add_argument('--quick', action='store_true', default=base.quick,
                        help='Use quick config (default: off)')
    parser.add_argument('--no_train', action='store_true',
                        help='Skip training and only run evaluation/plots from saved checkpoint')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from best saved checkpoint')
    parser.add_argument('--lambert_only', action='store_true',
                        help='Ignore shooting-corrected labels and train residual from Lambert dV + J2 terminal error')
    parser.add_argument('--data_dir', type=Path, default=base.data_dir)
    parser.add_argument('--ckpt_dir', type=Path, default=base.ckpt_dir)
    parser.add_argument('--dv_max', type=float, default=base.dv_max)
    parser.add_argument('--lr', type=float, default=base.lr)
    parser.add_argument('--pos_scale_m', type=float, default=base.pos_scale_m,
                        help='Position normalization scale in meters')
    parser.add_argument('--lambda_ps', type=float, default=base.lambda_ps)
    parser.add_argument('--lambda_sup', type=float, default=base.lambda_sup)
    parser.add_argument('--lambda_dv', type=float, default=base.lambda_dv,
                        help='Residual magnitude regularization weight for --lambert_only mode')
    parser.add_argument('--eval_batch_size', type=int, default=base.eval_batch_size,
                        help='Batch size for full-dataset evaluation/plot generation')
    parser.add_argument('--eval_max_samples', type=int, default=base.eval_max_samples,
                        help='If > 0, evaluate only the first N samples (debug speed-up)')
    args = parser.parse_args()

    return RunConfig(
        quick=args.quick,
        do_train=not args.no_train,
        resume=args.resume,
        lambert_only=args.lambert_only,
        data_dir=args.data_dir,
        ckpt_dir=args.ckpt_dir,
        dv_max=args.dv_max,
        lr=args.lr,
        pos_scale_m=args.pos_scale_m,
        lambda_ps=args.lambda_ps,
        lambda_sup=args.lambda_sup,
        lambda_dv=args.lambda_dv,
        eval_batch_size=args.eval_batch_size,
        eval_max_samples=args.eval_max_samples,
    )


def ensure_environment(cfg: RunConfig, device: torch.device) -> None:
    cfg.data_dir.mkdir(exist_ok=True)
    cfg.ckpt_dir.mkdir(exist_ok=True)

    print(f'Device: {device}')
    print(f'Datasets: train={cfg.train_path}, test={cfg.test_path}')
    print(f'EPOCHS={cfg.epochs}, BATCH_SIZE={cfg.batch_size}')
    print(f'LR={cfg.lr}, POS_SCALE={cfg.pos_scale_m:.1f} m, '
          f'LAMBDA_PS={cfg.lambda_ps}, LAMBDA_SUP={cfg.lambda_sup}, LAMBDA_DV={cfg.lambda_dv}')
    print(f'MODE={"lambert_only_residual" if cfg.lambert_only else "shooting_supervised"}')
    print(f'TRAINING={cfg.do_train}')
    print(f'RESUME={cfg.resume}')
    print(f'CHECKPOINT={cfg.ckpt_path}')

    for p in (cfg.train_path, cfg.test_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing dataset file: {p}\n"
                "Generate it first, for example:\n"
                "  python lambert_data.py"
            )

    d = np.load(cfg.train_path)
    h = np.cross(d['r0'], d['v0'])
    h_norm = np.linalg.norm(h, axis=1)
    inc_deg = np.degrees(np.arccos(np.clip(h[:, 2] / np.maximum(h_norm, 1e-12), -1.0, 1.0)))
    print(f'Dataset inclination (train): min={inc_deg.min():.2f}°, '
          f'mean={inc_deg.mean():.2f}°, max={inc_deg.max():.2f}°')


def print_dataset_keys(train_path: Path) -> None:
    d = np.load(train_path)
    print('\nDataset keys:')
    for k in sorted(d.files):
        print(f'  {k:25s} {str(d[k].shape):15s} {d[k].dtype}')


def build_train_objects(cfg: RunConfig, device: torch.device):
    train_ds = LambertDataset(cfg.train_path)
    test_ds = LambertDataset(cfg.test_path)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    r_scale, v_scale, tof_scale, dv_scale, corr_scale = train_ds.get_scales()
    print('Normalization scales:')
    print(f'  r={r_scale:.0f} km, v={v_scale:.2f} km/s, tof={tof_scale:.0f} s')
    print(f'  dv={dv_scale:.4f} km/s ({dv_scale*1000:.1f} m/s)')
    print(f'  pos_scale={cfg.pos_scale_m:.1f} m')
    print(f'  correction={corr_scale:.4f} km/s ({corr_scale*1000:.1f} m/s)')
    print(f'Train: {len(train_ds)}, Test: {len(test_ds)}')

    model = LambertTRC(
        cfg.net_cfg,
        n_coast_steps=cfg.n_coast,
        dv_max=cfg.dv_max,
        correction_scale=corr_scale,
    ).to(device)
    model.set_normalization(r_scale, v_scale, tof_scale, dv_scale, pos_scale_km=cfg.pos_scale_m / 1000.0)

    if cfg.lambert_only:
        criterion = LambertResidualJ2Loss(lambda_pos=1.0, lambda_ps=cfg.lambda_ps, lambda_dv=cfg.lambda_dv)
    else:
        criterion = LambertJ2Loss(lambda_pos=1.0, lambda_sup=cfg.lambda_sup, lambda_ps=cfg.lambda_ps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-5)

    print(f'\nParameters: {count_params(model):,}')
    print(f'K={cfg.net_cfg.K} outer iters, n={cfg.net_cfg.n_inner} inner cycles, '
          f'L={cfg.net_cfg.n_blocks} blocks, coast_steps={cfg.n_coast}')

    scales = {
        'r': r_scale,
        'v': v_scale,
        'tof': tof_scale,
        'dv': dv_scale,
        'corr': corr_scale,
        'pos_m': cfg.pos_scale_m,
    }
    return train_ds, test_ds, train_loader, test_loader, model, criterion, optimizer, scheduler, scales


@torch.no_grad()
def compare_direct_vs_forward(model, data_loader, device):
    model.eval()
    b = next(iter(data_loader))
    b = {k: v.to(device) for k, v in b.items()}

    r_direct, _ = model.coast(b['r0'], b['v0'] + b['dv1'], b['tof'])
    out = model(b['r0'], b['v0'], b['r_target'], b['v_target'], b['tof'], dv_lambert=b['dv1'])

    dv_diff = torch.norm(out['dv_final'] - b['dv1'], dim=-1)
    r_diff = torch.norm(out['r_final'] - r_direct, dim=-1)
    pos_err_direct = torch.norm(r_direct - b['r_target'], dim=-1)
    pos_err_forward = torch.norm(out['r_final'] - b['r_target'], dim=-1)

    print('mean |dv_final - dv1| [km/s]:', dv_diff.mean().item())
    print('max  |dv_final - dv1| [km/s]:', dv_diff.max().item())
    print('mean |r_forward - r_direct| [km]:', r_diff.mean().item())
    print('max  |r_forward - r_direct| [km]:', r_diff.max().item())
    print('mean direct pos err [km]:', pos_err_direct.mean().item())
    print('mean forward pos err [km]:', pos_err_forward.mean().item())


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses, imps, pos_errs, dv_res = [], [], [], []
    for b in loader:
        b = to_device(b, device)
        out = model(b['r0'], b['v0'], b['r_target'], b['v_target'], b['tof'], dv_lambert=b['dv1'])
        _, m = criterion(out, b, model)
        losses.append(m['loss'])
        imps.append(m['imp_metric'])
        pos_errs.append(m['pos_err_K'])
        dv_res.append(m['dv_res_norm'])
    return float(np.mean(losses)), float(np.mean(imps)), float(np.mean(pos_errs)), float(np.mean(dv_res))


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, history, path, net_cfg, scales):
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'history': history,
            'net_cfg': vars(net_cfg),
            'scales': scales,
        },
        path,
    )


def train_model(model, criterion, optimizer, scheduler, train_loader, test_loader,
                cfg: RunConfig, device: torch.device, ckpt_path: Path, scales: dict,
                history=None, start_epoch=1, best_val=float('inf')):
    if history is None:
        history = {
            'train_loss': [], 'val_loss': [],
            'train_imp': [], 'val_imp': [],
            'train_pos_err': [], 'val_pos_err': [],
            'train_dv_res': [], 'val_dv_res': [],
        }

    print(f'\n{"="*70}')
    print(f'Training LambertTRC for {cfg.epochs} epochs '
          f'(starting at epoch {start_epoch})')
    print('  Target: ~7 m/s correction, ~0 km position error')
    print('  Baseline: Lambert under J2 -> ~64 km position error')
    print(f'{"="*70}')

    for epoch in range(start_epoch, start_epoch + cfg.epochs):
        model.train()
        ep_loss, ep_imp, ep_pos, ep_dvr = [], [], [], []
        t0 = time.time()

        for b in train_loader:
            b = to_device(b, device)
            optimizer.zero_grad()
            out = model(b['r0'], b['v0'], b['r_target'], b['v_target'], b['tof'], dv_lambert=b['dv1'])
            loss, m = criterion(out, b, model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss.append(m['loss'])
            ep_imp.append(m['imp_metric'])
            ep_pos.append(m['pos_err_K'])
            ep_dvr.append(m['dv_res_norm'])

        scheduler.step()

        val_loss, val_imp, val_pos, val_dvr = evaluate(model, test_loader, criterion, device)
        train_loss = float(np.mean(ep_loss))
        train_imp = float(np.mean(ep_imp))
        train_pos = float(np.mean(ep_pos))
        train_dvr = float(np.mean(ep_dvr))

        for k, v in [
            ('train_loss', train_loss), ('val_loss', val_loss),
            ('train_imp', train_imp), ('val_imp', val_imp),
            ('train_pos_err', train_pos), ('val_pos_err', val_pos),
            ('train_dv_res', train_dvr), ('val_dv_res', val_dvr),
        ]:
            history[k].append(v)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, history, ckpt_path, cfg.net_cfg, scales)
            marker = ' *'
        else:
            marker = ''

        dt = time.time() - t0
        run_ep = epoch - start_epoch + 1
        print(f'Ep {run_ep:3d}/{cfg.epochs} (abs {epoch:3d})  loss={train_loss:.4f}  '
              f'pos={train_pos*1000.0:.1f}m  dV={train_dvr:.1f}m/s  imp={train_imp:.3f}  '
              f'|  val_pos={val_pos*1000.0:.1f}m  val_dV={val_dvr:.1f}m/s  '
              f'({dt:.1f}s){marker}')

    print(f'\nBest val loss: {best_val:.4f} - saved to {ckpt_path}')
    print(f'Target correction: ~{scales["corr"]*1000:.0f} m/s = {scales["corr"]:.4f} km/s')
    return history


def load_best_model(ckpt_path: Path, device: torch.device, n_coast: int, dv_max: float):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f'Loaded checkpoint from epoch {ckpt["epoch"]} (val_loss={ckpt["val_loss"]:.4f})')

    net_ck = NetConfig(**ckpt['net_cfg'])
    corr_scale = ckpt['scales'].get('corr', 0.01)
    model_eval = LambertTRC(net_ck, n_coast_steps=n_coast, dv_max=dv_max, correction_scale=corr_scale).to(device)
    sc = ckpt['scales']
    pos_scale_m = sc.get('pos_m')
    pos_scale_km = None if pos_scale_m is None else pos_scale_m / 1000.0
    model_eval.set_normalization(sc['r'], sc['v'], sc['tof'], sc['dv'], pos_scale_km=pos_scale_km)
    model_eval.load_state_dict(ckpt['model_state_dict'])
    model_eval.eval()
    print(f'Model: {count_params(model_eval):,} params')
    return model_eval


@torch.no_grad()
def run_full_dataset_eval(model_eval, ds, device, batch_size=256, max_samples=0, label='dataset'):
    n_total = len(ds.r0)
    n = n_total if max_samples <= 0 else min(n_total, int(max_samples))
    if n <= 0:
        raise ValueError('No samples selected for evaluation')

    all_r0_cpu = ds.r0[:n].clone()
    all_v0_cpu = ds.v0[:n].clone()
    all_rt_cpu = ds.r_target[:n].clone()
    all_tof_cpu = ds.tof[:n].clone()
    all_dv1 = ds.dv1[:n].cpu().numpy()

    expert_roll_pos_err_parts = []
    expert_tb_pos_err_parts = []
    dv_parts = None
    pe_parts = None
    zh_parts = None

    print(f'\nEvaluating {label}: {n} samples (batch_size={batch_size})')
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b_r0 = ds.r0[start:end].to(device)
        b_v0 = ds.v0[start:end].to(device)
        b_rt = ds.r_target[start:end].to(device)
        b_vt = ds.v_target[start:end].to(device)
        b_tof = ds.tof[start:end].to(device)
        b_dv = ds.dv1[start:end].to(device)

        out = model_eval(b_r0, b_v0, b_rt, b_vt, b_tof, dv_lambert=b_dv)

        r_expert_roll, _ = model_eval.coast(b_r0, b_v0 + b_dv, b_tof)
        expert_roll_pos_err_parts.append(torch.norm(r_expert_roll - b_rt, dim=-1).cpu().numpy())
        r_expert_tb, _ = model_eval.coast_twobody(b_r0, b_v0 + b_dv, b_tof)
        expert_tb_pos_err_parts.append(torch.norm(r_expert_tb - b_rt, dim=-1).cpu().numpy())

        if dv_parts is None:
            dv_parts = [[] for _ in range(len(out['dv_iterations']))]
            pe_parts = [[] for _ in range(len(out['pos_errors']))]
            zh_parts = [[] for _ in range(len(out['z_H_history']))]

        for k, dv in enumerate(out['dv_iterations']):
            dv_parts[k].append(dv.cpu().numpy())
        for k, pe in enumerate(out['pos_errors']):
            pe_parts[k].append(pe.cpu().numpy())
        for k, z in enumerate(out['z_H_history']):
            zh_parts[k].append(z.cpu().numpy())

        print(f'  {label}: {end}/{n}')

    dv_iters = [np.concatenate(parts, axis=0) for parts in dv_parts]
    pos_errs = [np.concatenate(parts, axis=0) for parts in pe_parts]
    z_H = [np.concatenate(parts, axis=0) for parts in zh_parts]

    return {
        'all_r0': all_r0_cpu,
        'all_v0': all_v0_cpu,
        'all_rt': all_rt_cpu,
        'all_tof': all_tof_cpu,
        'all_dv1': all_dv1,
        'expert_roll_pos_err': np.concatenate(expert_roll_pos_err_parts, axis=0),
        'expert_tb_pos_err': np.concatenate(expert_tb_pos_err_parts, axis=0),
        'dv_iters': dv_iters,
        'pos_errs': pos_errs,
        'z_H': z_H,
    }


def print_test_summary(eval_data: dict) -> None:
    all_dv1 = eval_data['all_dv1']
    dv_iters = eval_data['dv_iters']
    pos_errs = eval_data['pos_errs']
    expert_tb_pos_err = eval_data['expert_tb_pos_err']
    expert_roll_pos_err = eval_data['expert_roll_pos_err']

    K = len(dv_iters) - 1
    print(f'\nTest samples: {len(all_dv1)}')
    print('\n--- Expert Lambert dV under J2 rollout vs pure two-body ---')
    print(f'  Two-body pos error:  {expert_tb_pos_err.mean():.4f} +- {expert_tb_pos_err.std():.4f} km  (should be ~0)')
    print(f'  J2 rollout pos err:  {expert_roll_pos_err.mean():.1f} +- {expert_roll_pos_err.std():.1f} km  (the gap TRC must close)')
    print('\n--- TRC refinement iterations (propagated with J2 dynamics) ---')
    print(f'{"Iter":<6} {"Pos Err (km)":>15} {"dV mag (km/s)":>15}')
    print('-' * 40)
    for k in range(K + 1):
        pe = pos_errs[k] if k < len(pos_errs) else pos_errs[-1]
        dv_mag = np.linalg.norm(dv_iters[k], axis=-1)
        print(f'{k:<6} {pe.mean():>12.1f} +- {pe.std():>5.1f} {dv_mag.mean():>10.3f} +- {dv_mag.std():.3f}')

    expert_dv_mag = np.linalg.norm(all_dv1, axis=-1)
    pred_dv_mag = np.linalg.norm(dv_iters[-1], axis=-1)
    print(f'\nExpert |dV1|: {expert_dv_mag.mean():.3f} +- {expert_dv_mag.std():.3f} km/s')
    print(f'TRC    |dV1|: {pred_dv_mag.mean():.3f} +- {pred_dv_mag.std():.3f} km/s')
    if len(pos_errs) >= 2:
        reduction = (1 - pos_errs[-1].mean() / pos_errs[0].mean()) * 100
        print(f'Position error reduction: {reduction:.1f}%')


def main() -> None:
    cfg = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ensure_environment(cfg, device)
    print_dataset_keys(cfg.train_path)
    plotters.plot_data_overview(cfg.train_path, cfg.data_dir)

    if cfg.do_train:
        (train_ds, test_ds, train_loader, test_loader, model, criterion,
         optimizer, scheduler, scales) = build_train_objects(cfg, device)

        compare_direct_vs_forward(model, test_loader, device)
        history = None
        start_epoch = 1
        best_val = float('inf')
        if cfg.resume:
            if not cfg.ckpt_path.exists():
                raise FileNotFoundError(
                    f'Missing checkpoint for resume: {cfg.ckpt_path}\n'
                    'Run training first to create it.'
                )
            ckpt = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            history = ckpt.get('history')
            best_val = float(ckpt.get('val_loss', float('inf')))
            start_epoch = int(ckpt.get('epoch', 0)) + 1
            scales = ckpt.get('scales', scales)
            print(f'\nResuming from {cfg.ckpt_path} at epoch {start_epoch} '
                  f'(best val loss={best_val:.4f})')

        history = train_model(
            model,
            criterion,
            optimizer,
            scheduler,
            train_loader,
            test_loader,
            cfg,
            device,
            cfg.ckpt_path,
            scales,
            history=history,
            start_epoch=start_epoch,
            best_val=best_val,
        )

        plotters.plot_learning_curves(history, cfg.ckpt_dir)
    else:
        print('\nTraining disabled (`--no_train`): loading checkpoint for evaluation only.')
        if not cfg.ckpt_path.exists():
            raise FileNotFoundError(
                f'Missing checkpoint: {cfg.ckpt_path}\n'
                'Train first (without `--no_train`) to create it.'
            )
        train_ds = LambertDataset(cfg.train_path)
        test_ds = LambertDataset(cfg.test_path)

    model_eval = load_best_model(cfg.ckpt_path, device, cfg.n_coast, cfg.dv_max)
    train_eval = run_full_dataset_eval(
        model_eval,
        train_ds,
        device,
        batch_size=cfg.eval_batch_size,
        max_samples=cfg.eval_max_samples,
        label='train',
    )
    val_eval = run_full_dataset_eval(
        model_eval,
        test_ds,
        device,
        batch_size=cfg.eval_batch_size,
        max_samples=cfg.eval_max_samples,
        label='val',
    )
    print_test_summary(val_eval)
    plotters.plot_arrival_error_comparison(train_eval, val_eval, cfg.ckpt_dir)
    plotters.plot_dv_change_all(train_eval, val_eval, cfg.ckpt_dir)
    plotters.plot_trajectory_results(val_eval, cfg.ckpt_dir)
    plotters.plot_refinement_analysis(val_eval, cfg.ckpt_dir)
    plotters.plot_latent_space(val_eval, cfg.ckpt_dir)
    plotters.plot_single_sample(val_eval, cfg.ckpt_dir, idx=0)

    print('\nDone! All figures saved to checkpoints/')


if __name__ == '__main__':
    main()
