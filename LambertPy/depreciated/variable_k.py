"""
Variable-K TRC: Adaptive iteration depth for training and inference.
====================================================================

Three modes:
    1. fixed    - standard fixed K iterations (default, backward-compatible)
    2. random   - per-batch random K in [K_min, K_max] during training
    3. adaptive - stop when position error < tol (inference only)

Usage:
    # Replace LambertTRC.forward with variable-K support:
    model = LambertTRC(net_cfg, ...)

    # Fixed K (default, same as before):
    out = model(r0, v0, r_target, v_target, tof)

    # Random K during training:
    out = model(r0, v0, r_target, v_target, tof, K_mode='random', K_min=2, K_max=8)

    # Adaptive K at inference:
    with torch.no_grad():
        out = model(r0, v0, r_target, v_target, tof, K_mode='adaptive',
                    K_min=2, K_max=10, tol_km=0.1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def forward_variable_k(self, r0, v0, r_target, v_target, tof, dv_lambert=None,
                        K_mode='fixed', K_min=None, K_max=None, tol_km=0.1):
    B = r0.shape[0]
    n = self.net.n_inner

    if K_max is None:
        K_max = self.net.K
    if K_min is None:
        K_min = K_max if K_mode == 'fixed' else 1

    if K_mode == 'fixed':
        K = K_max
    elif K_mode == 'random':
        K = torch.randint(K_min, K_max + 1, (1,)).item()
    elif K_mode == 'adaptive':
        K = K_max
    else:
        raise ValueError(f"Unknown K_mode: {K_mode}")

    # Encode problem
    r0_n, v0_n, rt_n, vt_n, tof_n = self._normalize_input(
        r0, v0, r_target, v_target, tof
    )
    enc_input = torch.cat([r0_n, v0_n, rt_n, vt_n, tof_n], dim=-1)
    z0 = self.state_encoder(enc_input)

    z_H = self.H_init.expand(B, -1) + self.h_proj(z0)
    z_L = self.L_init.expand(B, -1) + self.l_proj(z0)

    if dv_lambert is None:
        dv = self.init_decoder(z0) * self.dv_scale
    else:
        delta0 = self.init_decoder(z0) * self.correction_scale
        dv = dv_lambert + delta0
    dv = self._clip_dv(dv)

    dv_iters = [dv]
    z_H_hist = [z_H.detach()]
    pos_errors = []

    # Per-sample tracking for adaptive mode
    converged = torch.zeros(B, dtype=torch.bool, device=r0.device)
    K_per_sample = torch.full((B,), K, dtype=torch.long, device=r0.device)

    for k in range(K):
        r_final, _ = self.coast(r0, v0 + dv, tof)
        pos_err = r_final - r_target
        error = pos_err / self.pos_scale
        pos_err_norm = torch.norm(pos_err, dim=-1)
        pos_errors.append(pos_err_norm)

        # Per-sample early stopping (k is 0-indexed, so compare against k+1)
        if K_mode == 'adaptive' and (k + 1) >= K_min:
            newly_converged = (~converged) & (pos_err_norm < tol_km)
            K_per_sample[newly_converged] = k + 1
            converged = converged | newly_converged
            if converged.all():
                break

        # Refinement — only update unconverged samples in adaptive mode
        z_err = self.error_encoder(error)
        z_ctrl = self.ctrl_embed(dv / self.dv_scale)
        for _ in range(n):
            z_L = self.reason(z_L, z_H, z0, z_err, z_ctrl)
        z_H = self.reason(z_H, z_L)

        delta_dv = self.res_decoder(torch.cat([z_H, dv / self.dv_scale], dim=-1))
        new_dv = self._clip_dv(dv + delta_dv * self.correction_scale)

        if K_mode == 'adaptive':
            # Freeze converged samples — keep their dv, don't apply correction
            mask = converged.unsqueeze(-1)  # (B, 1)
            dv = torch.where(mask, dv, new_dv)
        else:
            dv = new_dv

        dv_iters.append(dv)
        z_H_hist.append(z_H.detach())

    r_final, v_final = self.coast(r0, v0 + dv, tof)
    pos_errors.append(torch.norm(r_final - r_target, dim=-1))

    return {
        'dv_final': dv,
        'dv_iterations': dv_iters,
        'pos_errors': pos_errors,
        'z_H_history': z_H_hist,
        'r_final': r_final,
        'v_final': v_final,
        'dv_lambert': dv_lambert,
        'K_per_sample': K_per_sample,
        'K_used': k + 1,  # max iterations actually executed
    }

class VariableKLoss(nn.Module):
    """Loss function that handles variable iteration counts.

    Works with any K — the improvement reward adapts to however many
    iterations were actually run. Compatible with fixed, random, or
    adaptive K modes.

    Loss = lambda_u0 * L_u0 + lambda_pos * L_pos + lambda_ps * L_proc

    where:
        L_u0:   initial control should approximate Lambert
        L_pos:  final position error should be small
        L_proc: position error should decrease at each iteration (Eq. 16 style)
    """

    def __init__(self, lambda_u0=5.0, lambda_pos=1.0, lambda_ps=0.05):
        super().__init__()
        self.lambda_u0 = lambda_u0
        self.lambda_pos = lambda_pos
        self.lambda_ps = lambda_ps

    def forward(self, output, batch, model):
        pos_errors = output['pos_errors']
        dv_iters = output['dv_iterations']
        u0 = dv_iters[0]
        K_actual = len(pos_errors) - 1  # pos_errors has K+1 entries

        # 1) Initial policy matches Lambert
        L_u0 = F.mse_loss(u0, batch['dv1']) / (model.dv_scale ** 2)

        # 2) Final terminal position error
        pos_err_km = pos_errors[-1]
        pos_norm = model.pos_scale
        L_pos = (pos_err_km / pos_norm).pow(2).mean()

        # 3) Process supervision: error should decrease at each iteration
        #    Adapts to however many iterations were run
        if K_actual >= 2:
            err0 = pos_errors[0].detach().clamp(min=1e-3)
            normed = [e / err0 for e in pos_errors]
            # Use all adjacent transitions across the executed rollout.
            improvements = [normed[k] - normed[k + 1] for k in range(len(normed) - 1)]
            L_proc = -torch.stack(improvements).mean()
        else:
            L_proc = torch.tensor(0.0, device=pos_err_km.device)

        loss = self.lambda_u0 * L_u0 + self.lambda_pos * L_pos + self.lambda_ps * L_proc

        # Metrics (Eq. 16 style)
        with torch.no_grad():
            if K_actual >= 2:
                e0 = pos_errors[0].clamp(min=1e-3)
                imp_metric = 0.0
                for k in range(len(pos_errors) - 1):
                    imp_metric += ((pos_errors[k] - pos_errors[k + 1]) / e0).mean().item()
                imp_metric /= (len(pos_errors) - 1)
            else:
                imp_metric = 0.0

        return loss, {
            'loss': loss.item(),
            'L_u0': L_u0.item(),
            'L_pos': L_pos.item(),
            'L_proc': L_proc.item(),
            'imp_metric': imp_metric,
            'pos_err_0': pos_errors[0].mean().item(),
            'pos_err_K': pos_errors[-1].mean().item(),
            'K_used': K_actual,
            'u0_res_norm': torch.norm(u0 - batch['dv1'], dim=-1).mean().item() * 1000.0,
            'dv_res_norm': torch.norm(output['dv_final'] - batch['dv1'], dim=-1).mean().item() * 1000.0,
        }


# ── Monkey-patch helper ─────────────────────────────────────────────────────

def patch_model(model):
    """Replace model.forward with variable-K version.

    Usage:
        model = LambertTRC(net_cfg, ...)
        patch_model(model)
        # Now model supports K_mode='fixed'|'random'|'adaptive'
    """
    import types
    model.forward = types.MethodType(forward_variable_k, model)
    return model


# ── Training loop with variable K ───────────────────────────────────────────

def train_variable_k(
    model, criterion, optimizer, train_loader, test_loader, device,
    epochs=60, K_min=2, K_max=8, K_mode='random',
    anchor_to_lambert=True, scheduler=None,
    ckpt_path=None, print_every=1,
):
    """Training loop with variable K support.

    Args:
        model:       LambertTRC (already patched with patch_model)
        criterion:   VariableKLoss or compatible
        optimizer:   torch optimizer
        train_loader, test_loader: data loaders
        device:      torch device
        epochs:      number of training epochs
        K_min, K_max: iteration range for random mode
        K_mode:      'fixed' | 'random' for training
        anchor_to_lambert: if True, seed with Lambert dV
        scheduler:   optional LR scheduler
        ckpt_path:   if provided, save best checkpoint here
        print_every: print frequency

    Returns:
        history dict
    """
    history = {
        'train_loss': [], 'val_loss': [],
        'train_ctrl_loss': [], 'val_ctrl_loss': [],
        'train_imp': [], 'val_imp': [],
        'train_pos_0': [], 'val_pos_0': [],
        'train_pos_K': [], 'val_pos_K': [],
        'train_K_used': [], 'val_K_used': [],
    }
    best_val = float('inf')

    for ep in range(1, epochs + 1):
        # ── Train ──
        model.train()
        ep_loss, ep_ctrl, ep_imp, ep_pos0, ep_posK, ep_K = [], [], [], [], [], []

        for b in train_loader:
            b = {k: v.to(device) for k, v in b.items()}
            optimizer.zero_grad()

            dv_seed = b['dv1'] if anchor_to_lambert else None
            out = model(
                b['r0'], b['v0'], b['r_target'], b['v_target'], b['tof'],
                dv_lambert=dv_seed, K_mode=K_mode, K_min=K_min, K_max=K_max,
            )
            loss, m = criterion(out, b, model)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss.append(m['loss'])
            ep_ctrl.append(m['L_u0'])
            ep_imp.append(m['imp_metric'])
            ep_pos0.append(m['pos_err_0'])
            ep_posK.append(m['pos_err_K'])
            ep_K.append(m['K_used'])

        if scheduler is not None:
            scheduler.step()

        # ── Eval (always use K_max for fair comparison) ──
        model.eval()
        vl, vc, vi, vp0, vpK, vK = [], [], [], [], [], []
        with torch.no_grad():
            for b in test_loader:
                b = {k: v.to(device) for k, v in b.items()}
                dv_seed = b['dv1'] if anchor_to_lambert else None
                out = model(
                    b['r0'], b['v0'], b['r_target'], b['v_target'], b['tof'],
                    dv_lambert=dv_seed, K_mode='fixed', K_max=K_max,
                )
                _, m = criterion(out, b, model)
                vl.append(m['loss']); vc.append(m['L_u0'])
                vi.append(m['imp_metric'])
                vp0.append(m['pos_err_0']); vpK.append(m['pos_err_K'])
                vK.append(m['K_used'])

        # ── Record ──
        tl = np.mean(ep_loss) if ep_loss else float('nan')
        tc = np.mean(ep_ctrl) if ep_ctrl else float('nan')
        ti = np.mean(ep_imp) if ep_imp else float('nan')
        tp0 = np.mean(ep_pos0) if ep_pos0 else float('nan')
        tpK = np.mean(ep_posK) if ep_posK else float('nan')
        tK = np.mean(ep_K) if ep_K else 0

        val_loss = np.mean(vl)
        val_ctrl = np.mean(vc)
        val_imp = np.mean(vi)
        val_p0 = np.mean(vp0)
        val_pK = np.mean(vpK)
        val_Ku = np.mean(vK)

        for key, val in [
            ('train_loss', tl), ('val_loss', val_loss),
            ('train_ctrl_loss', tc), ('val_ctrl_loss', val_ctrl),
            ('train_imp', ti), ('val_imp', val_imp),
            ('train_pos_0', tp0), ('val_pos_0', val_p0),
            ('train_pos_K', tpK), ('val_pos_K', val_pK),
            ('train_K_used', tK), ('val_K_used', val_Ku),
        ]:
            history[key].append(val)

        # ── Checkpoint ──
        marker = ''
        if val_loss < best_val:
            best_val = val_loss
            if ckpt_path is not None:
                torch.save({
                    'epoch': ep, 'best_val': best_val,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history,
                    'K_min': K_min, 'K_max': K_max, 'K_mode': K_mode,
                }, ckpt_path)
            marker = ' ★'

        if ep % print_every == 0:
            print(
                f'[{ep:03d}] loss={tl:.4f} posK={tpK:.3f}km K={tK:.1f} '
                f'| val={val_loss:.4f} posK={val_pK:.3f}km imp={val_imp:.3f}{marker}'
            )

    return history


# ── Inference with adaptive stopping ────────────────────────────────────────

@torch.no_grad()
def inference_adaptive(model, r0, v0, r_target, v_target, tof, dv_lambert=None,
                       K_min=2, K_max=10, tol_km=0.1):
    """Run inference with adaptive early stopping.

    Returns the same output dict as forward, but stops as soon as
    all samples in the batch achieve position error < tol_km.

    Prints convergence info.
    """
    model.eval()
    out = model(
        r0, v0, r_target, v_target, tof,
        dv_lambert=dv_lambert,
        K_mode='adaptive', K_min=K_min, K_max=K_max, tol_km=tol_km,
    )

    K_used = out['K_used']
    final_err = out['pos_errors'][-1]
    converged = (final_err.max().item() < tol_km)

    print(f'Adaptive inference: K_used={K_used}/{K_max}, '
          f'pos_err={final_err.mean():.4f} ± {final_err.std():.4f} km, '
          f'max={final_err.max():.4f} km, '
          f'converged={converged} (tol={tol_km} km)')

    return out
