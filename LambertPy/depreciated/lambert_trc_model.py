"""Lambert TRC model components.

Contains model, loss, and dataset utilities for Lambert TRC.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from constants import MU_EARTH, R_EARTH
from trc import NetConfig, ReasoningModule, make_mlp

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

CONTROL_FEATURE_NAMES_CARTESIAN = (
    "dv_x", "dv_y", "dv_z",
)

CONTROL_FEATURE_NAMES_SPHERICAL = (
    "dv_mag", "dv_alpha", "dv_beta",
)


def j2_dynamics_torch(state, mu=MU_EARTH, r_earth=R_EARTH, j2_coeff=1.08263e-3):
    """Two-body + J2 EOM for torch tensors. state = (B, 6): [r, v]."""
    r = state[:, :3]
    v = state[:, 3:6]
    x, y, z = r[:, 0:1], r[:, 1:2], r[:, 2:3]

    r_mag = torch.norm(r, dim=-1, keepdim=True).clamp(min=1e-6)

    # Central gravity: -mu * r / |r|^3
    a_grav = -mu * r / r_mag**3

    # J2 perturbation
    r2 = r_mag**2
    r5 = r_mag**5
    z2 = z**2
    factor = 1.5 * j2_coeff * mu * (r_earth**2) / r5
    common = 5.0 * z2 / r2

    a_j2 = factor * torch.cat([
        x * (common - 1.0),
        y * (common - 1.0),
        z * (common - 3.0),
    ], dim=-1)

    return torch.cat([v, a_grav + a_j2], dim=-1)


def twobody_dynamics_torch(state, mu=MU_EARTH):
    """Pure two-body EOM (used only for reference/comparison)."""
    r = state[:, :3]
    v = state[:, 3:6]
    r_mag = torch.norm(r, dim=-1, keepdim=True).clamp(min=1e-6)
    a = -mu * r / r_mag**3
    return torch.cat([v, a], dim=-1)


class CoastSimulator(nn.Module):
    """Propagate orbit dynamics for a fixed time using RK4."""

    def __init__(self, n_steps=None, max_step_seconds=45.0, mu=MU_EARTH, dynamics='j2'):
        super().__init__()
        self.n_steps = n_steps
        self.max_step_seconds = float(max_step_seconds)
        self.mu = mu
        self.dynamics = dynamics

    def _rhs(self, state):
        if self.dynamics == 'j2':
            return j2_dynamics_torch(state, self.mu)
        return twobody_dynamics_torch(state, self.mu)

    def forward(self, r0, v0_plus_dv, tof):
        """Propagate from (r0, v0+dV) for time tof."""
        if self.n_steps is not None:
            n_steps = max(int(self.n_steps), 1)
        else:
            tof_max = max(float(torch.max(tof).item()), 0.0)
            n_steps = max(int(np.ceil(tof_max / self.max_step_seconds)), 1)
        dt = tof / n_steps
        state = torch.cat([r0, v0_plus_dv], dim=-1)

        for _ in range(n_steps):
            k1 = self._rhs(state)
            k2 = self._rhs(state + 0.5 * dt * k1)
            k3 = self._rhs(state + 0.5 * dt * k2)
            k4 = self._rhs(state + dt * k3)
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return state[:, :3], state[:, 3:6]


class LambertTRC(nn.Module):
    """Tiny Recursive Control for Lambert orbit transfers."""

    def __init__(self, net: NetConfig, n_coast_steps=None, max_coast_step_s=45.0, mu=MU_EARTH,
                 dv_max=50.0, correction_scale=0.01, max_correction=0.5,
                 pos_scale_km=100.0, state_repr='cartesian'):
        super().__init__()
        self.net = net
        self.dv_max = dv_max
        self.correction_scale = correction_scale
        self.max_correction = float(max_correction)
        self.state_repr = state_repr
        d_z, d_h = net.d_z, net.d_h
        if self.state_repr == 'cartesian':
            encoder_features = ENCODER_FEATURE_NAMES_CARTESIAN
            control_features = CONTROL_FEATURE_NAMES_CARTESIAN
        elif self.state_repr == 'spherical':
            encoder_features = ENCODER_FEATURE_NAMES_SPHERICAL
            control_features = CONTROL_FEATURE_NAMES_SPHERICAL
        else:
            raise ValueError(f"Unsupported state_repr={state_repr!r}")
        self.encoder_feature_names = encoder_features
        self.control_feature_names = control_features
        self.control_dim = len(self.control_feature_names)

        self.error_encoder = make_mlp(3, d_h, d_z)
        self.ctrl_embed = nn.Linear(self.control_dim, d_z)
        self.reason = ReasoningModule(d_z, d_h, net.n_heads, net.n_blocks, net.dropout)
        self.res_decoder = make_mlp(d_z + self.control_dim, d_h, self.control_dim)
        nn.init.zeros_(self.res_decoder[-1].weight)
        nn.init.zeros_(self.res_decoder[-1].bias)
        
        self.state_encoder = nn.Sequential(
            nn.Linear(len(self.encoder_feature_names), d_h),
            nn.LayerNorm(d_h),
            nn.GELU(),
            nn.Linear(d_h, d_h),
            nn.LayerNorm(d_h),
            nn.GELU(),
            nn.Linear(d_h, d_z),
        )

        self.init_decoder = nn.Sequential(
            nn.Linear(d_z, d_h),
            nn.LayerNorm(d_h),
            nn.GELU(),
            nn.Linear(d_h, d_h),
            nn.LayerNorm(d_h),
            nn.GELU(),
            nn.Linear(d_h, d_h // 2),
            nn.LayerNorm(d_h // 2),
            nn.GELU(),
            nn.Linear(d_h // 2, self.control_dim),
        )


        self.H_init = nn.Parameter(torch.randn(d_z) * 0.02)
        self.L_init = nn.Parameter(torch.randn(d_z) * 0.02)
        self.h_proj = nn.Linear(d_z, d_z)
        self.l_proj = nn.Linear(d_z, d_z)

        self.coast = CoastSimulator(
            n_steps=n_coast_steps,
            max_step_seconds=max_coast_step_s,
            mu=mu,
            dynamics='j2',
        )
        self.coast_twobody = CoastSimulator(
            n_steps=n_coast_steps,
            max_step_seconds=max_coast_step_s,
            mu=mu,
            dynamics='twobody',
        )

        self.register_buffer('r_min', torch.full((3,), -1.0))
        self.register_buffer('r_max', torch.full((3,), 1.0))
        self.register_buffer('v_min', torch.full((3,), -1.0))
        self.register_buffer('v_max', torch.full((3,), 1.0))
        self.register_buffer('r_mag_min', torch.tensor(0.0))
        self.register_buffer('r_mag_max', torch.tensor(1.0))
        self.register_buffer('v_mag_min', torch.tensor(0.0))
        self.register_buffer('v_mag_max', torch.tensor(1.0))
        self.register_buffer('tof_min', torch.tensor(0.0))
        self.register_buffer('tof_max', torch.tensor(1.0))
        self.register_buffer('tof_log_min', torch.tensor(0.0))
        self.register_buffer('tof_log_max', torch.tensor(1.0))
        self.register_buffer('dv_scale', torch.tensor(1.0))
        self.register_buffer('dv_mag_min', torch.tensor(0.0))
        self.register_buffer('dv_mag_max', torch.tensor(1.0))
        self.register_buffer('use_tof_log', torch.tensor(False))
        self.register_buffer('nrev_min', torch.tensor(0.0))
        self.register_buffer('nrev_max', torch.tensor(1.0))
        # Fixed position normalization scale (km) for input/feedback conditioning.
        self.register_buffer('pos_scale', torch.tensor(float(pos_scale_km)))

    def set_normalization(self, r_min, r_max, v_min, v_max, tof_min, tof_max, dv_scale,
                          pos_scale_km=None, use_tof_log=False, tof_log_min=None,
                          tof_log_max=None, nrev_min=0.0, nrev_max=1.0,
                          r_mag_min=None, r_mag_max=None, v_mag_min=None, v_mag_max=None,
                          dv_mag_min=None, dv_mag_max=None):
        self.r_min.copy_(torch.as_tensor(r_min, dtype=self.r_min.dtype, device=self.r_min.device))
        self.r_max.copy_(torch.as_tensor(r_max, dtype=self.r_max.dtype, device=self.r_max.device))
        self.v_min.copy_(torch.as_tensor(v_min, dtype=self.v_min.dtype, device=self.v_min.device))
        self.v_max.copy_(torch.as_tensor(v_max, dtype=self.v_max.dtype, device=self.v_max.device))
        if r_mag_min is not None:
            self.r_mag_min.fill_(float(r_mag_min))
        if r_mag_max is not None:
            self.r_mag_max.fill_(float(r_mag_max))
        if v_mag_min is not None:
            self.v_mag_min.fill_(float(v_mag_min))
        if v_mag_max is not None:
            self.v_mag_max.fill_(float(v_mag_max))
        self.tof_min.fill_(float(tof_min))
        self.tof_max.fill_(float(tof_max))
        self.use_tof_log.fill_(1 if use_tof_log else 0)
        self.dv_scale.fill_(dv_scale)
        if dv_mag_min is not None:
            self.dv_mag_min.fill_(float(dv_mag_min))
        if dv_mag_max is not None:
            self.dv_mag_max.fill_(float(dv_mag_max))
        self.nrev_min.fill_(float(nrev_min))
        self.nrev_max.fill_(max(float(nrev_max), float(nrev_min) + 1e-6))
        if use_tof_log:
            if tof_log_min is None or tof_log_max is None:
                raise ValueError("tof_log_min and tof_log_max must be provided when use_tof_log=True")
            self.tof_log_min.fill_(float(tof_log_min))
            self.tof_log_max.fill_(float(tof_log_max))
        if pos_scale_km is not None:
            self.pos_scale.fill_(float(pos_scale_km))

    def _branch_feature(self, value, ref_tensor, *, scale=1.0, offset=0.0):
        if value is None:
            return torch.zeros((ref_tensor.shape[0], 1), device=ref_tensor.device, dtype=ref_tensor.dtype)
        if value.ndim == 0:
            value = value.reshape(1, 1).expand(ref_tensor.shape[0], 1)
        elif value.ndim == 1:
            value = value.unsqueeze(-1)
        return (value.to(dtype=ref_tensor.dtype) + offset) / scale

    def _minmax_normalize(self, value, vmin, vmax):
        vmin = torch.as_tensor(vmin, dtype=value.dtype, device=value.device)
        vmax = torch.as_tensor(vmax, dtype=value.dtype, device=value.device)
        half_span = 0.5 * (vmax - vmin)
        half_span = torch.where(half_span.abs() < 1e-6, torch.ones_like(half_span), half_span)
        center = 0.5 * (vmax + vmin)
        return (value - center) / half_span

    def _normalize_alpha(self, alpha):
        return alpha / np.pi

    def _normalize_beta(self, beta):
        return beta / (0.5 * np.pi)

    def _denormalize_alpha(self, alpha_n):
        return alpha_n * np.pi

    def _denormalize_beta(self, beta_n):
        return beta_n * (0.5 * np.pi)

    def _vector_to_spherical(self, vec, mag_min, mag_max):
        x = vec[:, 0:1]
        y = vec[:, 1:2]
        z = vec[:, 2:3]
        rho = torch.norm(vec, dim=-1, keepdim=True)
        xy = torch.norm(vec[:, :2], dim=-1, keepdim=True).clamp(min=1e-6)
        alpha = torch.atan2(y, x)
        beta = torch.atan2(z, xy)
        rho_n = self._minmax_normalize(rho, mag_min, mag_max)
        return torch.cat([rho_n, self._normalize_alpha(alpha), self._normalize_beta(beta)], dim=-1)

    def _minmax_denormalize(self, value, vmin, vmax):
        vmin = torch.as_tensor(vmin, dtype=value.dtype, device=value.device)
        vmax = torch.as_tensor(vmax, dtype=value.dtype, device=value.device)
        return 0.5 * (value + 1.0) * (vmax - vmin) + vmin

    def _spherical_to_vector(self, sph, mag_min, mag_max):
        rho = self._minmax_denormalize(sph[:, 0:1], mag_min, mag_max)
        alpha = self._denormalize_alpha(sph[:, 1:2])
        beta = self._denormalize_beta(sph[:, 2:3])
        cos_beta = torch.cos(beta)
        return torch.cat([
            rho * cos_beta * torch.cos(alpha),
            rho * cos_beta * torch.sin(alpha),
            rho * torch.sin(beta),
        ], dim=-1)

    def _normalize_input(self, r0, v0, r_target, v_target, tof, nrev=None, ncase=None, prograde=None):
        if self.use_tof_log.item():
            tof_n = self._minmax_normalize(torch.log1p(tof), self.tof_log_min, self.tof_log_max)
        else:
            tof_n = self._minmax_normalize(tof, self.tof_min, self.tof_max)
        nrev_n = self._minmax_normalize(self._branch_feature(nrev, r0), self.nrev_min, self.nrev_max)
        ncase_n = self._branch_feature(ncase, r0, offset=-0.5, scale=0.5)
        prograde_n = self._branch_feature(prograde, r0, offset=-0.5, scale=0.5)
        if self.state_repr == 'cartesian':
            return (
                self._minmax_normalize(r0, self.r_min, self.r_max),
                self._minmax_normalize(v0, self.v_min, self.v_max),
                self._minmax_normalize(r_target, self.r_min, self.r_max),
                tof_n,
                nrev_n,
                ncase_n,
                prograde_n,
            )
        return (
            self._vector_to_spherical(r0, self.r_mag_min, self.r_mag_max),
            self._vector_to_spherical(v0, self.v_mag_min, self.v_mag_max),
            self._vector_to_spherical(r_target, self.r_mag_min, self.r_mag_max),
            tof_n,
            nrev_n,
            ncase_n,
            prograde_n,
        )

    def _encode_problem_state(self, r0, v0, r_target, v_target, tof, nrev=None, ncase=None, prograde=None):
        """Build the encoded state vector from normalized orbital features."""
        r0_n, v0_n, rt_n, tof_n, nrev_n, ncase_n, prograde_n = self._normalize_input(
            r0, v0, r_target, v_target, tof, nrev=nrev, ncase=ncase, prograde=prograde
        )
        return torch.cat([r0_n, v0_n, rt_n, tof_n, nrev_n, ncase_n, prograde_n], dim=-1)

    def encode_control(self, dv):
        if self.state_repr == 'spherical':
            return self._vector_to_spherical(dv, self.dv_mag_min, self.dv_mag_max)
        return dv / self.dv_scale

    def decode_control(self, control_repr):
        if self.state_repr == 'spherical':
            return self._spherical_to_vector(control_repr, self.dv_mag_min, self.dv_mag_max)
        return control_repr * self.dv_scale

    def _initial_dv(self, z0, dv_lambert):
        """Predict first-burn dv in physical units (km/s)."""
        if dv_lambert is None:
            if self.state_repr == 'spherical':
                ctrl0 = torch.tanh(self.init_decoder(z0))
                return ctrl0, self.decode_control(ctrl0)
            ctrl0 = self.init_decoder(z0)
            return ctrl0, self.decode_control(ctrl0)
        if self.state_repr == 'spherical':
            ctrl_seed = self.encode_control(dv_lambert)
            delta0 = torch.tanh(self.init_decoder(z0)) * self.correction_scale
            ctrl0 = torch.tanh(ctrl_seed + delta0)
            return ctrl0, self.decode_control(ctrl0)
        delta0 = self.init_decoder(z0) * self.correction_scale
        dv = dv_lambert + delta0
        return dv / self.dv_scale, dv

    def _clip_dv(self, dv):
        return torch.clamp(dv, -self.dv_max, self.dv_max)

    def _clip_correction(self, dv, dv_anchor):
        if self.max_correction <= 0.0:
            return dv
        return torch.clamp(dv, dv_anchor - self.max_correction, dv_anchor + self.max_correction)

    def forward(self, r0, v0, r_target, v_target=None, tof=None, dv_lambert=None,
                nrev=None, ncase=None, prograde=None, stage1_only=False,
                K_min=None, K_max=None, tol_km=None):
        B = r0.shape[0]
        n = self.net.n_inner

        # Default to net.K if not specified
        if K_max is None:
            K_max = self.net.K
        if K_min is None:
            K_min = K_max
        if tol_km is None:
            tol_km = 0.1  # 100 m default

        if tof is None:
            raise ValueError("tof is required")
        enc_input = self._encode_problem_state(
            r0, v0, r_target, v_target, tof, nrev=nrev, ncase=ncase, prograde=prograde
        )
        z0 = self.state_encoder(enc_input)

        z_H = self.H_init.expand(B, -1) + self.h_proj(z0)
        z_L = self.L_init.expand(B, -1) + self.l_proj(z0)

        dv_repr, dv = self._initial_dv(z0, dv_lambert)
        dv = self._clip_dv(dv)
        dv_anchor = dv.detach()

        # ── Stage-1 fast path: only train state_encoder + init_decoder ──
        if stage1_only:
            # No coast propagation, no refinement loop.
            # Return a dummy pos_error so the loss interface stays compatible.
            dummy_pos_err = torch.zeros(B, device=r0.device)
            return {
                'dv_final': dv,
                'dv_iterations': [dv],
                'dv_final_repr': dv_repr,
                'dv_repr_iterations': [dv_repr],
                'pos_errors': [dummy_pos_err],
                'z_H_history': [z_H.detach()],
                'r_final': torch.zeros_like(r0),
                'v_final': torch.zeros_like(v0),
                'dv_lambert': dv_lambert,
            }

        dv_iters = [dv]
        dv_repr_iters = [dv_repr]
        z_H_hist = [z_H.detach()]
        pos_errors = []

        K_used = K_max
        for k in range(K_max):
            r_final, _ = self.coast(r0, v0 + dv, tof)
            pos_err = r_final - r_target
            error = pos_err / self.pos_scale
            pos_err_norm = torch.norm(pos_err, dim=-1)
            pos_errors.append(pos_err_norm)

            # Early stop if past K_min and all samples converged
            if k >= K_min and pos_err_norm.max().item() < tol_km:
                K_used = k
                break
            

            z_err = self.error_encoder(error)
            z_ctrl = self.ctrl_embed(dv_repr)
            for _ in range(n):
                z_L = self.reason(z_L, z_H, z0, z_err, z_ctrl)
            z_H = self.reason(z_H, z_L)

            if self.state_repr == 'spherical':
                delta_repr = torch.tanh(self.res_decoder(torch.cat([z_H, dv_repr], dim=-1))) * self.correction_scale
                dv_repr = torch.tanh(dv_repr + delta_repr)
                dv = self.decode_control(dv_repr)
            else:
                dv_unit = dv / max(float(self.dv_max), 1e-6)
                delta_dv = self.res_decoder(torch.cat([z_H, dv_unit], dim=-1))
                dv = self._clip_dv(dv + delta_dv * self.correction_scale)
                dv_repr = dv / self.dv_scale
            dv = self._clip_correction(dv, dv_anchor)
            if self.state_repr != 'spherical':
                dv_repr = dv / self.dv_scale

            dv_iters.append(dv)
            dv_repr_iters.append(dv_repr)
            z_H_hist.append(z_H.detach())

        r_final, v_final = self.coast(r0, v0 + dv, tof)
        pos_errors.append(torch.norm(r_final - r_target, dim=-1))

        return {
            'dv_final': dv,
            'dv_iterations': dv_iters,
            'dv_final_repr': dv_repr,
            'dv_repr_iterations': dv_repr_iters,
            'pos_errors': pos_errors,
            'z_H_history': z_H_hist,
            'r_final': r_final,
            'v_final': v_final,
            'dv_lambert': dv_lambert,
            'K_used': K_used,
        }


class LambertJ2Loss(nn.Module):
    """Combined loss for Lambert+J2 correction."""

    def __init__(self, lambda_pos=1.0, lambda_sup=0.5, lambda_ps=0.05):
        super().__init__()
        self.lambda_pos = lambda_pos
        self.lambda_sup = lambda_sup
        self.lambda_ps = lambda_ps

    def forward(self, output, batch, model):
        pos_errors = output['pos_errors']
        dv_hat = output['dv_final']

        pos_err_km = pos_errors[-1]
        pos_norm = getattr(model, 'pos_scale', torch.tensor(100.0, device=pos_err_km.device))
        L_pos = (pos_err_km / pos_norm).pow(2).mean()

        if 'dv1_corrected' in batch:
            dv_target = batch['dv1_corrected']
            L_sup = F.mse_loss(dv_hat, dv_target) / (model.dv_scale ** 2)
        else:
            L_sup = torch.tensor(0.0, device=dv_hat.device)

        if len(pos_errors) >= 2:
            err0 = pos_errors[0].detach().clamp(min=1e-3)
            normed = [e / err0 for e in pos_errors]
            improvements = [normed[k] - normed[k + 1] for k in range(len(normed) - 1)]
            L_proc = -torch.stack(improvements).mean()
        else:
            L_proc = torch.tensor(0.0, device=dv_hat.device)

        loss = self.lambda_pos * L_pos + self.lambda_sup * L_sup + self.lambda_ps * L_proc

        with torch.no_grad():
            imp_metric = 0.0
            if len(pos_errors) >= 2:
                e0 = pos_errors[0].clamp(min=1e-3)
                for k in range(len(pos_errors) - 1):
                    imp_metric += ((pos_errors[k] - pos_errors[k + 1]) / e0).mean().item()
                imp_metric /= (len(pos_errors) - 1)

        return loss, {
            'loss': loss.item(),
            'L_pos': L_pos.item(),
            'L_sup': L_sup.item(),
            'L_proc': L_proc.item(),
            'imp_metric': imp_metric,
            'pos_err_0': pos_errors[0].mean().item(),
            'pos_err_K': pos_errors[-1].mean().item(),
            'dv_mag': torch.norm(dv_hat, dim=-1).mean().item(),
            'dv_res_norm': torch.norm(dv_hat - batch['dv1'], dim=-1).mean().item() * 1000,
        }


class LambertResidualJ2Loss(nn.Module):
    """Lambert-only residual loss (no shooting-corrected supervision)."""

    def __init__(self, lambda_pos=1.0, lambda_ps=0.05, lambda_dv=0.01):
        super().__init__()
        self.lambda_pos = lambda_pos
        self.lambda_ps = lambda_ps
        self.lambda_dv = lambda_dv

    def forward(self, output, batch, model):
        pos_errors = output['pos_errors']
        dv_hat = output['dv_final']

        pos_err_km = pos_errors[-1]
        pos_norm = getattr(model, 'pos_scale', torch.tensor(100.0, device=pos_err_km.device))
        L_pos = (pos_err_km / pos_norm).pow(2).mean()

        if len(pos_errors) >= 2:
            err0 = pos_errors[0].detach().clamp(min=1e-3)
            normed = [e / err0 for e in pos_errors]
            improvements = [normed[k] - normed[k + 1] for k in range(len(normed) - 1)]
            L_proc = -torch.stack(improvements).mean()
        else:
            L_proc = torch.tensor(0.0, device=dv_hat.device)

        # Keep residual corrections small unless they improve terminal accuracy.
        dv_res = dv_hat - batch['dv1']
        L_dv = (dv_res / model.dv_scale).pow(2).mean()

        loss = self.lambda_pos * L_pos + self.lambda_ps * L_proc + self.lambda_dv * L_dv

        with torch.no_grad():
            imp_metric = 0.0
            if len(pos_errors) >= 2:
                e0 = pos_errors[0].clamp(min=1e-3)
                for k in range(len(pos_errors) - 1):
                    imp_metric += ((pos_errors[k] - pos_errors[k + 1]) / e0).mean().item()
                imp_metric /= (len(pos_errors) - 1)

        return loss, {
            'loss': loss.item(),
            'L_pos': L_pos.item(),
            'L_dv': L_dv.item(),
            'L_proc': L_proc.item(),
            'imp_metric': imp_metric,
            'pos_err_0': pos_errors[0].mean().item(),
            'pos_err_K': pos_errors[-1].mean().item(),
            'dv_mag': torch.norm(dv_hat, dim=-1).mean().item(),
            'dv_res_norm': torch.norm(dv_res, dim=-1).mean().item() * 1000,
        }


class LambertDataset(Dataset):
    def __init__(self, path, tof_max_seconds=None):
        d = np.load(path)
        tof_np = np.asarray(d['tof']).astype(np.float32)
        mask = np.ones(len(tof_np), dtype=bool)
        if tof_max_seconds is not None:
            mask &= tof_np <= float(tof_max_seconds)
            kept = int(mask.sum())
            total = int(mask.size)
            print(f"  TOF filter: <= {float(tof_max_seconds):.1f} s  ({kept}/{total} samples kept)")
        if not mask.any():
            raise ValueError(f"No samples match TOF filter <= {tof_max_seconds} s in {path}")

        self.r0 = torch.from_numpy(d['r0'][mask]).float()
        self.v0 = torch.from_numpy(d['v0'][mask]).float()
        self.r_target = torch.from_numpy(d['r_target'][mask]).float()
        self.v_target = torch.from_numpy(d['v_target'][mask]).float()
        self.tof = torch.from_numpy(tof_np[mask]).float().unsqueeze(-1)
        self.dv1 = torch.from_numpy(np.asarray(d['dv1'])[mask]).float()
        self.dv2 = torch.from_numpy(np.asarray(d['dv2'])[mask]).float()
        self.total_dv = torch.from_numpy(np.asarray(d['total_dv'])[mask]).float()
        self.nrev = torch.from_numpy(np.asarray(d['nrev'])[mask]).float() if 'nrev' in d else torch.zeros(len(self.r0))
        self.ncase = torch.from_numpy(np.asarray(d['ncase'])[mask]).float() if 'ncase' in d else torch.zeros(len(self.r0))
        self.prograde = torch.from_numpy(np.asarray(d['prograde'])[mask]).float() if 'prograde' in d else torch.zeros(len(self.r0))

        if 'dv1_corrected' in d:
            self.dv1_corrected = torch.from_numpy(np.asarray(d['dv1_corrected'])[mask]).float()
            self.dv_correction = torch.from_numpy(np.asarray(d['dv_correction'])[mask]).float()
            corr_mag = torch.norm(self.dv_correction, dim=-1) * 1000
            print(f'  Shooting corrections loaded: |Δv|={corr_mag.mean():.1f} ± {corr_mag.std():.1f} m/s')
        else:
            self.dv1_corrected = self.dv1.clone()
            self.dv_correction = torch.zeros_like(self.dv1)
            print('  No shooting corrections — using Lambert dV as target')

        # ── Print data statistics ──
        dv1_n = torch.norm(self.dv1, dim=-1)
        pcts = torch.quantile(dv1_n, torch.tensor([0.25, 0.5, 0.75, 0.95, 0.99]))
        print(f'  Loaded {len(self)} samples from {path}')
        print(f'    |dV1|: {dv1_n.mean():.4f} ± {dv1_n.std():.4f} km/s  '
              f'[{dv1_n.min():.4f}, {dv1_n.max():.4f}]')
        print(f'    |dV1| percentiles: p25={pcts[0]:.4f} p50={pcts[1]:.4f} '
              f'p75={pcts[2]:.4f} p95={pcts[3]:.4f} p99={pcts[4]:.4f}')
        print(f'    total_dV: {self.total_dv.mean():.4f} ± {self.total_dv.std():.4f} km/s  '
              f'[{self.total_dv.min():.4f}, {self.total_dv.max():.4f}]')
        print(f'    TOF: {self.tof.mean():.0f} ± {self.tof.std():.0f} s  '
              f'[{self.tof.min():.0f}, {self.tof.max():.0f}]')

    def __len__(self):
        return len(self.r0)

    def __getitem__(self, i):
        return {
            'r0': self.r0[i], 'v0': self.v0[i],
            'r_target': self.r_target[i], 'v_target': self.v_target[i],
            'tof': self.tof[i], 'dv1': self.dv1[i],
            'dv1_corrected': self.dv1_corrected[i],
            'nrev': self.nrev[i], 'ncase': self.ncase[i], 'prograde': self.prograde[i],
        }

    def get_scales(self):
        r_scale = torch.norm(self.r0, dim=-1).median().item()
        v_scale = torch.norm(self.v0, dim=-1).median().item()
        tof_scale = self.tof.median().item()
        dv_scale = torch.norm(self.dv1, dim=-1).median().item()
        corr_scale = torch.norm(self.dv_correction, dim=-1).median().item()
        return r_scale, v_scale, tof_scale, dv_scale, corr_scale
