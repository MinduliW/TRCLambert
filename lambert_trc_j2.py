"""
lambert_trc_j2.py
=================

Inputs:  r1 (3), r2 (3), Δt (1)  →  7 features
Outputs: v1 (3)  — J2-corrected departure velocity

Head 1 (Initial Guess):
    z0       = MLPstate([r1; r2; Δt])          Eq. (2)
    v1^(0)   = MLPinitial(z0)                  Eq. (3)
    Loss:      ||v1,Lambert - v1^(0)||          Eq. (4)

Head 2 (Iterative):
    For k = 1..K:
        Propagate [r1; v1^(k-1)] under J2 for Δt → rf^(k-1)
        e^(k-1) = rf^(k-1) - r2                Eq. (5)
        zerr    = MLPerror(e^(k-1))             Eq. (6)
        zctrl   = Linear(v1^(k-1))
        z_H, z_L initialised per Eq. (7)
        z_ctx   = [z0; zerr; zctrl]             Eq. (8) — concatenation
        z_L^(i+1) = Lθ(z_L^(i), z_H, z_ctx)   Eq. (9) — n tactical cycles
        z_H^(k+1) = Lθ(z_H^(k), z_L^(n))      Eq. (10)
        Δv^(k)  = MLPresidual([z_H; v1^(k-1)]) Eq. (11)
        v1^(k)  = clip(v1^(k-1) + Δv:(^(k), …)  Eq. (12)
    Loss:  Eq. (13-15)

Data: train_struct.mat / val_struct.mat (MATLAB v7.3 / HDF5)
  Fields (all per-sample, stored as object references):
    r1, r2         — positions [km]
    tof            — time of flight [s]
    v1             — Lambert departure velocity [km/s]  = v1,Lambert
    v1_j2          — J2-corrected departure velocity [km/s] = v1,true
    v2             — Lambert arrival velocity [km/s]
    dv_j2          — correction vector v1_j2 - v1 [km/s]
    nrev, ncase, prograde — Lambert branch parameters
    j2_converged, j2_iters, j2_pos_err — shooting diagnostics
"""


import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')  # avoid OpenMP crash on macOS

import sys
import time
import shutil
import datetime
from pathlib import Path
from dataclasses import dataclass

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Add parent dir so we can import trc.py ──────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from trc import NetConfig, ReasoningModule, make_mlp

# ── Physical constants ────────────────────────────────────────────────────────
MU_EARTH   = 398600.4418   # km³/s²
R_EARTH    = 6378.137      # km
J2_EARTH   = 1.08263e-3    # Earth oblateness (dimensionless)
J2_COEFF   = J2_EARTH      # alias kept for any legacy references

MU_JUPITER = 126686534.0   # km³/s²  — must match MATLAB generator (1.26686534e8)
R_JUPITER  = 71492.0       # km  (equatorial)
J2_JUPITER = 0.014736      # Jupiter oblateness — must match MATLAB (1.4736e-2)

# Body parameter lookup: name → (mu [km³/s²], re [km], j2)
BODY_PARAMS = {
    'earth':   (MU_EARTH,   R_EARTH,   J2_EARTH),
    'jupiter': (MU_JUPITER, R_JUPITER, J2_JUPITER),
}

def _load_npz_arrays(cache_path: Path) -> dict:
    """Load all arrays from a cached .npz file."""
    with np.load(cache_path) as d:
        return {k: d[k] for k in d.files}


def _load_cached_npz(cache_path: Path) -> dict | None:
    """Load a cached .npz if all stored arrays are float64."""
    try:
        arrays = _load_npz_arrays(cache_path)
    except Exception as exc:
        print(f"  Cached data at {cache_path} could not be read ({exc}); rebuilding …")
        return None

    bad_fields = [
        fn for fn, arr in arrays.items()
        if arr.dtype != np.float64
    ]
    if bad_fields:
        fields_str = ", ".join(sorted(bad_fields))
        print(
            f"  Cached data at {cache_path} uses low-precision dtypes for "
            f"{fields_str}; rebuilding from .mat …"
        )
        return None

    return arrays


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _load_mat_struct(mat_path: str, struct_key: str) -> dict:
    """Load a MATLAB struct array stored under struct_key (v7.3 or older).

    Returns a dict mapping field name → numpy array of shape (N, dim).
    Vector fields (3-dim) give shape (N, 3); scalar fields give shape (N,).
    Only samples where j2_converged == 1 are retained.
    """
    # .npz is a fast numpy cache — much quicker to load than .mat on repeat runs
    cache_path = Path(mat_path).with_suffix('.npz')
    if cache_path.exists():
        mat_mtime = Path(mat_path).stat().st_mtime if Path(mat_path).exists() else 0
        npz_mtime = cache_path.stat().st_mtime
        if mat_mtime > npz_mtime:
            print(f"  .mat is newer than cache — rebuilding {cache_path}")
        else:
            arrays = _load_cached_npz(cache_path)
            if arrays is not None:
                print(f"  Loading cached data from {cache_path}")
                return arrays

    # Detect Git LFS pointer files (not the real data) and give a clear message.
    with open(mat_path, 'rb') as _f:
        _header = _f.read(100)
    if _header.startswith(b'version https://git-lfs'):
        if cache_path.exists():
            print(
                f"  {mat_path} is a Git LFS pointer; "
                f"falling back to cached data in {cache_path}"
            )
            return _load_npz_arrays(cache_path)
        raise RuntimeError(
            f"{mat_path} is a Git LFS pointer, not the actual file.\n"
            f"Run:  git lfs pull --include=\"{Path(mat_path).name}\""
        )

    print(f"  Parsing {mat_path} (one-time, will cache to {cache_path}) …")
    t0 = time.time()

    # MATLAB saves .mat files in two different formats; try HDF5 first
    try:
        with h5py.File(mat_path, 'r') as f:
            f[struct_key]  # probe
        use_hdf5 = True
    except OSError:
        use_hdf5 = False

    if use_hdf5:
        with h5py.File(mat_path, 'r') as f:
            grp = f[struct_key]
            field_names = list(grp.keys())
            # MATLAB struct array may be saved as (1, N) or (N, 1) depending on
            # how it was constructed — take the longer dim as the sample count.
            shape0 = grp[field_names[0]].shape
            n_total = max(shape0)
            row_major = shape0[0] == n_total

            # Build a list for each field — we'll append one sample at a time then stack
            raw = {fn: [] for fn in field_names}
            for i in range(n_total):
                for fn in field_names:
                    ref = grp[fn][i, 0] if row_major else grp[fn][0, i]
                    val = np.array(f[ref]).flatten()
                    raw[fn].append(val)
                if (i + 1) % 10000 == 0:
                    print(f"    parsed {i+1}/{n_total} …")
    else:
        import scipy.io
        mat = scipy.io.loadmat(mat_path, squeeze_me=False)
        struct = mat[struct_key]
        field_names = struct.dtype.names
        n_total = struct.shape[1]

        raw = {fn: [] for fn in field_names}
        for i in range(n_total):
            for fn in field_names:
                val = np.array(struct[0, i][fn]).flatten()
                raw[fn].append(val)
            if (i + 1) % 10000 == 0:
                print(f"    parsed {i+1}/{n_total} …")

    print(f"  Parsed {n_total} samples in {time.time()-t0:.1f}s")

    # Stack into arrays in float64 so MATLAB labels survive the cache round-trip
    # without any precision loss.
    arrays = {}
    for fn in field_names:
        stacked = np.stack(raw[fn], axis=0)
        arrays[fn] = stacked.astype(np.float64)

    # Only keep samples where the J2 shooting algorithm converged — discard bad data
    conv = arrays['j2_converged'].flatten().astype(bool)
    kept = int(conv.sum())
    print(f"  Converged: {kept}/{n_total} samples retained")
    arrays = {fn: v[conv] for fn, v in arrays.items()}

    # These fields are scalars per sample — squeeze from shape (N,1) to (N,)
    for fn in ('tof', 'nrev', 'ncase', 'prograde',
               'j2_converged', 'j2_iters', 'j2_pos_err'):
        if fn in arrays:
            arrays[fn] = arrays[fn].flatten()

    # Save to disk so next run skips all the above parsing
    np.savez_compressed(cache_path, **arrays)
    print(f"  Cached to {cache_path}")
    return arrays


# PyTorch Dataset: wraps our data so the DataLoader can feed it to the network
# in random shuffled batches during training.
class MatLambertDataset(Dataset):
    """Dataset loaded from train_struct.mat / val_struct.mat.

    Provides:
        r1, r2     — (3,) positions [km]
        tof        — (1,) time of flight [s]
        v1_lambert — (3,) Lambert departure velocity [km/s]
        v1_true    — (3,) J2-corrected departure velocity [km/s]
        v2         — (3,) Lambert arrival velocity [km/s]
        nrev, ncase, prograde — scalar branch params
    """

    def __init__(self, mat_path: str, struct_key: str, dv_pct_cutoff: float = None):
        arrays = _load_mat_struct(mat_path, struct_key)

        self.r1         = torch.from_numpy(arrays['r1'])                  # (N, 3) float64
        self.r2         = torch.from_numpy(arrays['r2'])                  # (N, 3) float64
        self.tof        = torch.from_numpy(arrays['tof'])                 # (N,)    float64
        self.v1_lambert = torch.from_numpy(arrays['v1'])                  # (N, 3) float64
        self.v1_true    = torch.from_numpy(arrays['v1_j2'])               # (N, 3) float64
        self.v2         = torch.from_numpy(arrays['v2'])                  # (N, 3) float64
        self.nrev       = torch.from_numpy(arrays['nrev'])                # (N,)    float64
        self.ncase      = torch.from_numpy(arrays['ncase'])               # (N,)    float64
        self.prograde   = torch.from_numpy(arrays['prograde'])            # (N,)    float64

        if dv_pct_cutoff is not None:
            dv_m = torch.norm(self.v1_true - self.v1_lambert, dim=-1) * 1000
            thr = torch.quantile(dv_m, dv_pct_cutoff / 100.0)
            keep = dv_m < thr
            N0 = len(self.r1)
            for attr in ('r1','r2','tof','v1_lambert','v1_true','v2','nrev','ncase','prograde'):
                setattr(self, attr, getattr(self, attr)[keep])
            print(f"  {mat_path}: filtered |Δv|<p{dv_pct_cutoff:g} ({thr:.1f} m/s) "
                  f"→ kept {keep.sum()}/{N0}")

        N = len(self.r1)
        v1_mag  = torch.norm(self.v1_lambert, dim=-1)
        dv_corr = torch.norm(self.v1_true - self.v1_lambert, dim=-1) * 1000  # m/s
        r1_mag  = torch.norm(self.r1, dim=-1)
        print(f"  {mat_path}: {N} samples")
        print(f"    |r1| : {r1_mag.mean():.0f} ± {r1_mag.std():.0f} km")
        print(f"    |v1| : {v1_mag.mean():.3f} ± {v1_mag.std():.3f} km/s")
        print(f"    |Δv| : {dv_corr.mean():.1f} ± {dv_corr.std():.1f} m/s  "
              f"(J2 correction)")
        print(f"    tof  : {self.tof.mean():.0f} ± {self.tof.std():.0f} s")

    # PyTorch calls this to know how many samples there are
    def __len__(self):
        return len(self.r1)

    # PyTorch calls this with an index to get one sample — returns a dict of tensors.
    # Physical fields are stored as float64 in the cache for data integrity, but the
    # NN and MPS device only support float32, so we cast here.
    def __getitem__(self, idx):
        return {
            'r1':         self.r1[idx].float(),
            'r2':         self.r2[idx].float(),
            'tof':        self.tof[idx].float().unsqueeze(-1),
            'v1_lambert': self.v1_lambert[idx].float(),
            'v1_true':    self.v1_true[idx].float(),
            'v2':         self.v2[idx].float(),
            'nrev':       self.nrev[idx].float(),
            'ncase':      self.ncase[idx].float(),
            'prograde':   self.prograde[idx].float(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. J2 PROPAGATOR (differentiable, batched)
# ═══════════════════════════════════════════════════════════════════════════════

# This computes the equations of motion (accelerations) for a satellite
# under Earth gravity + J2 oblateness perturbation.
# Called repeatedly inside the RK4 integrator below.
def _j2_rhs(state: torch.Tensor,
            mu: float = MU_EARTH,
            re: float = R_EARTH,
            j2: float = J2_COEFF) -> torch.Tensor:
    """J2 equations of motion.  state: (B, 6) — [rx, ry, rz, vx, vy, vz]."""
    r = state[:, :3]
    v = state[:, 3:6]
    x, y, z = r[:, 0:1], r[:, 1:2], r[:, 2:3]

    r_mag = torch.norm(r, dim=-1, keepdim=True).clamp(min=1e-6)
    r2    = r_mag ** 2
    r5    = r_mag ** 5
    z2    = z ** 2

    # Standard two-body gravitational acceleration: a = -μ * r / |r|³
    a_grav  = -mu * r / r_mag ** 3
    # J2 perturbation: extra acceleration due to Earth's equatorial bulge
    factor  = 1.5 * j2 * mu * re ** 2 / r5
    common  = 5.0 * z2 / r2
    a_j2    = factor * torch.cat([x * (common - 1.0),
                                   y * (common - 1.0),
                                   z * (common - 3.0)], dim=-1)
    return torch.cat([v, a_grav + a_j2], dim=-1)


# Numerically integrates the J2 equations of motion using RK4 (Runge-Kutta 4th order).
# "Differentiable" means PyTorch can compute gradients through it — needed for training.
# "Batched" means it processes B transfers simultaneously.
class J2Propagator(nn.Module):
    """Differentiable batch RK4 propagator under J2 dynamics."""

    def __init__(self, max_step_s: float = 45.0,
                 mu: float = MU_EARTH, re: float = R_EARTH, j2: float = J2_EARTH):
        super().__init__()
        self.max_step_s = max_step_s
        self.mu = mu
        self.re = re
        self.j2 = j2

    def forward(self, r0: torch.Tensor, v0: torch.Tensor,
                tof: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Propagate (r0, v0) for tof seconds.

        Args:
            r0:  (B, 3) departure position [km]
            v0:  (B, 3) departure velocity [km/s]
            tof: (B, 1) time of flight [s]

        Returns:
            rf: (B, 3) arrival position [km]
            vf: (B, 3) arrival velocity [km/s]
        """
        # Use float64 on CPU/CUDA (full precision, ~15 m error).
        # MPS (Apple Silicon) does not support float64, so stay in float32 there
        # (~100 m error, acceptable during training — gradients still flow correctly).
        use_f64 = r0.device.type != 'mps'
        cast = lambda t: t.double() if use_f64 else t.float()

        r0  = cast(r0)
        v0  = cast(v0)
        tof = cast(tof)

        # Per-sample n_steps matching MATLAB shooting_j2_correct.m exactly:
        # n_steps_i = max(50, ceil(tof_i / max_dt))
        per_n  = torch.clamp(torch.ceil(tof / self.max_step_s), min=50).long()  # (B, 1)
        n_max  = int(per_n.max().item())
        dt     = tof / cast(per_n)               # per-sample dt (B, 1)
        state  = torch.cat([r0, v0], dim=-1)     # (B, 6)

        # RK4: iterate for the longest sample; zero out the update for samples
        # that have already reached their individual n_steps.
        rhs = lambda s: _j2_rhs(s, mu=self.mu, re=self.re, j2=self.j2)
        for k in range(n_max):
            dt_eff = dt * cast(k < per_n)        # (B, 1) — zero for finished samples
            k1 = rhs(state)
            k2 = rhs(state + 0.5 * dt_eff * k1)
            k3 = rhs(state + 0.5 * dt_eff * k2)
            k4 = rhs(state + dt_eff * k3)
            state = state + (dt_eff / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return state[:, :3].float(), state[:, 3:6].float()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TRC MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class LambertTRCJ2(nn.Module):
    """TRC for J2-perturbed Lambert problem.

    Inputs:  r1 (3), r2 (3), tof (1), prograde (1)
    Outputs: v1 (3) — J2-corrected departure velocity [km/s]
    """

    def __init__(self,
                 net: NetConfig,
                 max_step_s: float = 45.0,
                 v_max: float = 15.0,
                 pos_scale_km: float = 200.0,
                 mu: float = MU_EARTH,
                 re: float = R_EARTH,
                 j2: float = J2_EARTH,
                 head1_mode: str = 'direct'):
        """
        Args:
            net:          Architecture config (d_z, d_h, n_blocks, K, n_inner)
            max_step_s:   Max RK4 step size for J2 propagation [s]
            v_max:        Velocity clipping bound [km/s]
            pos_scale_km: Position error normalisation for error encoder [km]
            mu:           Gravitational parameter [km³/s²]
            re:           Body equatorial radius [km]
            j2:           J2 oblateness coefficient
            head1_mode:   How Head 1's output is used:
                          'direct'   — Head 1 outputs absolute v1 (legacy, learned_lambert).
                                       init_v1 is ignored.
                          'oracle'   — Head 1 ignored; v1 := init_v1 (legacy pos_only / vel_supervised).
                          'residual' — Encoder input grows to 14 dims by appending v_lambert
                                       (passed via init_v1). Head 1 outputs absolute v1_j2 directly.
                                       Stage 1 supervises this output against v1_j2.
        """
        # nn.Module is the base class for all PyTorch neural networks.
        # super().__init__() runs its setup — always required.
        super().__init__()
        self.net    = net
        self.v_max  = v_max
        if head1_mode not in ('direct', 'oracle', 'residual'):
            raise ValueError(f"head1_mode must be one of direct/oracle/residual, got {head1_mode!r}")
        self.head1_mode = head1_mode
        d_z, d_h    = net.d_z, net.d_h

        # ── Head 1: state encoder + initial decoder ──────────────────────────
        # MLPstate: Linear → LayerNorm → GELU → Linear  (paper Eq. 2)
        # Input dim depends on whether v_lambert is fed as an encoder feature:
        #   8  — legacy without nrev/ncase/arc (only via load-time rebuild)
        #   11 — r1(3)+r2(3)+tof(1)+prograde(1)+nrev(1)+ncase(1)+arc(1)
        #   14 — above + v_lambert(3)   (used when head1_mode='residual')
        self.input_dim = 14 if head1_mode == 'residual' else 11
        self.state_encoder = nn.Sequential(
            nn.Linear(self.input_dim, d_h),
            nn.LayerNorm(d_h),
            nn.GELU(),
            nn.Linear(d_h, d_z),
        )
        # MLPinitial: same architecture, maps z0 → v1^(0)  (paper Eq. 3)
        self.init_decoder = nn.Sequential(
            nn.Linear(d_z, d_h),
            nn.LayerNorm(d_h),
            nn.GELU(),
            nn.Linear(d_h, 3),
        )

        # H_init and L_init are learned starting points for the "strategic" and "tactical"
        # latent vectors in Head 2. W_H and W_L shift them based on the problem (z0).
        # ── Latent initialisation (paper Eq. 7) ──────────────────────────────
        self.H_init = nn.Parameter(torch.randn(d_z) * 0.02)
        self.L_init = nn.Parameter(torch.randn(d_z) * 0.02)
        self.W_H    = nn.Linear(d_z, d_z, bias=False)
        self.W_L    = nn.Linear(d_z, d_z, bias=False)

        # ── Head 2: error encoder, control embedder, reasoning, residual dec ─
        # MLPerror: 2-layer MLP with same architecture  (paper Eq. 6)
        self.error_encoder = make_mlp(3, d_h, d_z)

        # Linear projection for control  (paper: z_ctrl = Linear(flatten(v1)))
        self.ctrl_embed = nn.Linear(3, d_z)

        # Shared reasoning module L_θ  (paper Eqs. 9-10)
        self.reason = ReasoningModule(d_z, d_h, net.n_heads, net.n_blocks,
                                      net.dropout)

        # The residual decoder outputs a velocity correction Δv at each iteration.
        # Initialised to zero so the first correction is zero — stable starting point.
        # MLPresidual: maps [z_H; v1^(k-1)] → Δv^(k)  (paper Eq. 11)
        self.res_decoder = make_mlp(d_z + 3, d_h, 3)
        nn.init.normal_(self.res_decoder[-1].weight, std=1e-4)
        nn.init.zeros_(self.res_decoder[-1].bias)

        # ── J2 propagator ────────────────────────────────────────────────────
        self.propagator = J2Propagator(max_step_s=max_step_s, mu=mu, re=re, j2=j2)

        # Buffers are constants stored inside the model (not trained).
        # They hold the normalisation statistics computed from the training data.
        # register_buffer ensures they move to GPU with the model if needed.
        # ── Normalisation buffers (set via set_normalisation) ────────────────
        self.register_buffer('r_center',  torch.zeros(3))
        self.register_buffer('r_scale',   torch.ones(3))
        self.register_buffer('tof_mean',  torch.tensor(1800.0))
        self.register_buffer('tof_std',   torch.tensor(900.0))
        self.register_buffer('v_mean',    torch.zeros(3))
        self.register_buffer('v_std',     torch.ones(3))
        self.register_buffer('pos_scale', torch.tensor(float(pos_scale_km)))

    # ── Normalisation helpers ────────────────────────────────────────────────

    # Called once after building the model to store training-data statistics.
    # These are used to scale inputs/outputs to roughly [-1, 1] range,
    # which helps the network train faster and more stably.
    def set_normalisation(self, r_center, r_scale,
                          tof_mean, tof_std,
                          v_mean, v_std,
                          pos_scale_km: float = None):
        """Set normalisation statistics from training data."""
        def _t(x): return torch.as_tensor(x, dtype=torch.float32)
        self.r_center.copy_(_t(r_center))
        self.r_scale.copy_(_t(r_scale))
        self.tof_mean.fill_(float(tof_mean))
        self.tof_std.fill_(float(tof_std))
        self.v_mean.copy_(_t(v_mean))
        self.v_std.copy_(_t(v_std))
        if pos_scale_km is not None:
            self.pos_scale.fill_(float(pos_scale_km))

    # z-score: subtract mean, divide by std → ~O(1) values
    def _norm_r(self, r):
        return (r - self.r_center) / self.r_scale.clamp(min=1e-6)

    def _norm_tof(self, tof):
        return (tof - self.tof_mean) / self.tof_std.clamp(min=1e-6)

    def _norm_v(self, v):
        return (v - self.v_mean) / self.v_std.clamp(min=1e-6)

    # reverse the z-score to get back to km/s
    def _denorm_v(self, v_n):
        return v_n * self.v_std + self.v_mean

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, r1: torch.Tensor, r2: torch.Tensor, tof: torch.Tensor,
                prograde: torch.Tensor,
                nrev: torch.Tensor = None, ncase: torch.Tensor = None,
                arc: torch.Tensor = None,
                K: int = None, stage1_only: bool = False,
                init_v1: torch.Tensor = None):
        """
        Args:
            r1:          (B, 3) departure position [km]
            r2:          (B, 3) target position [km]
            tof:         (B, 1) time of flight [s]
            prograde:    (B, 1) prograde flag (1=prograde, 0=retrograde)
            K:           number of refinement iterations (default: net.K)
            stage1_only: if True run only Head 1 (no J2 propagation)

        Returns dict with:
            v1_pred      (B, 3) final predicted departure velocity [km/s]
            v1_iters     list of (B, 3) tensors for each iteration (incl. k=0)
            pos_errors   list of (B,) position error norms [km] at each k
            rf_iters     list of (B, 3) arrival positions at each k
            vf_iters     list of (B, 3) arrival velocities at each k
        """
        if K is None:
            K = self.net.K
        model_dtype = self.r_center.dtype
        r1 = r1.to(dtype=model_dtype)
        r2 = r2.to(dtype=model_dtype)
        tof = tof.to(dtype=model_dtype)
        prograde = prograde.to(dtype=model_dtype)
        # B = batch size — number of transfers processed simultaneously
        B = r1.shape[0]
        n = self.net.n_inner

        # ── Build normalised input [r1_n; r2_n; tof_n; prograde; nrev; ncase; arc] ──
        r1_n   = self._norm_r(r1)
        r2_n   = self._norm_r(r2)
        tof_n  = self._norm_tof(tof)                        # (B, 1)
        p      = prograde.unsqueeze(-1) if prograde.dim() == 1 else prograde  # (B, 1)
        # nrev, ncase, arc: scalar branch identifiers — zero if not provided (single-rev)
        def _scalar_feat(x):
            if x is None:
                return torch.zeros(B, 1, dtype=model_dtype, device=r1.device)
            x = x.to(dtype=model_dtype)
            return x.unsqueeze(-1) if x.dim() == 1 else x
        nrev_f  = _scalar_feat(nrev)
        ncase_f = _scalar_feat(ncase)
        arc_f   = _scalar_feat(arc)
        # Concatenate all inputs. Shape depends on input_dim:
        #   8  — legacy without nrev/ncase/arc
        #   11 — geometry + branch indicators (current default)
        #   14 — above + v_lambert (head1_mode='residual')
        if self.input_dim == 8:
            x_in = torch.cat([r1_n, r2_n, tof_n, p], dim=-1)
        elif self.input_dim == 14:
            if init_v1 is None:
                raise RuntimeError(
                    "head1_mode='residual' (input_dim=14) requires init_v1 (v_lambert) "
                    "to be passed as the encoder feature."
                )
            v_lambert_n = self._norm_v(init_v1.to(dtype=model_dtype))
            x_in = torch.cat([r1_n, r2_n, tof_n, p, nrev_f, ncase_f, arc_f, v_lambert_n], dim=-1)
        else:  # input_dim == 11
            x_in = torch.cat([r1_n, r2_n, tof_n, p, nrev_f, ncase_f, arc_f], dim=-1)

        # ── Head 1 ───────────────────────────────────────────────────────────
        z0     = self.state_encoder(x_in)        # (B, d_z)
        v1_n   = self.init_decoder(z0)           # (B, 3) normalised
        v1     = self._denorm_v(v1_n)            # (B, 3) km/s — absolute v1

        # Oracle override: replace Head 1's output entirely with provided velocity.
        # In 'residual' mode init_v1 is the encoder feature, NOT an override —
        # we always use Head 1's actual output there.
        if self.head1_mode == 'oracle' and init_v1 is not None:
            v1 = init_v1.to(dtype=model_dtype)

        # If we only want the Head 1 guess (Stage 1 training), return now
        if stage1_only:
            return {
                'v1_pred':   v1,
                'v1_iters':  [v1],
                'pos_errors': [],
                'rf_iters':  [],
                'vf_iters':  [],
            }

        # ── Head 2 initialisation  (paper Eq. 7) ─────────────────────────────
        # Initialise the two latent "memory" vectors for this problem:
        # z_H = "strategic" memory (updated once per outer iteration)
        # z_L = "tactical" memory (updated n_inner times per iteration)
        # Both start from a learned base shifted by the problem encoding z0.
        z_H = self.H_init.unsqueeze(0).expand(B, -1) + self.W_H(z0)   # (B, d_z)
        z_L = self.L_init.unsqueeze(0).expand(B, -1) + self.W_L(z0)   # (B, d_z)

        v1_iters   = [v1]
        pos_errors = []
        rf_iters   = []
        vf_iters   = []

        # K outer iterations — each one propagates, checks error, and corrects v1
        for _ in range(K):
            # Fly the satellite forward under J2 using current v1 guess
            rf, vf = self.propagator(r1, v1, tof)
            # Position error: how far we missed the target (km)
            e_pos  = rf - r2                           # (B, 3)  position error [km]
            e_norm = torch.norm(e_pos, dim=-1)         # (B,)
            pos_errors.append(e_norm)
            rf_iters.append(rf)
            vf_iters.append(vf)

            # Error + control encodings  (paper Eqs. 6, 8)
            # Scale error to ~O(1) before feeding to the network
            e_scaled = e_pos / self.pos_scale          # normalise error
            z_err    = self.error_encoder(e_scaled)    # (B, d_z)
            z_ctrl   = self.ctrl_embed(v1)             # (B, d_z)

            # Tactical cycles: z_L = Lθ(z_L, z_H, z0, z_err, z_ctrl)  Eq. (9)
            # (5 tokens: z_L first so it is the "updated" output token)
            # n_inner "tactical" reasoning steps — refine z_L using all context
            for _ in range(n):
                z_L = self.reason(z_L, z_H, z0, z_err, z_ctrl)

            # Strategic update: z_H = Lθ(z_H, z_L)  Eq. (10)
            # One "strategic" update — z_H absorbs what z_L learned
            z_H = self.reason(z_H, z_L)

            # Residual correction  (paper Eqs. 11-12)
            # Predict a small velocity correction from the updated memory
            dv    = self.res_decoder(torch.cat([z_H, v1], dim=-1))  # (B, 3)
            # Apply correction and clip to physically reasonable range
            v1    = (v1 + dv).clamp(-self.v_max, self.v_max)
            v1_iters.append(v1)

        # Propagate the final v1_pred so pos_errors[-1] aligns with v1_pred.
        # Previously pos_errors[-1] was from v1_iters[K-1] (one step behind),
        # causing L_pos and L_vK to supervise different velocities.
        rf, vf = self.propagator(r1, v1, tof)
        pos_errors.append(torch.norm(rf - r2, dim=-1))
        rf_iters.append(rf)
        vf_iters.append(vf)

        return {
            'v1_pred':   v1,
            'v1_iters':  v1_iters,
            'pos_errors': pos_errors,
            'rf_iters':  rf_iters,
            'vf_iters':  vf_iters,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LOSS FUNCTION  (paper Eq. 13-15)
# ═══════════════════════════════════════════════════════════════════════════════

class LambertTRCJ2Loss(nn.Module):
    """Full training loss from paper Eq. (13).

    L = λv [0.3 ||v1^(0) - v1,Lambert||² + 0.7 ||v1^(K) - v1,true||²]
        + λpos ||e^(K)||²
        - λps (1/(K-1)) Σ_{k=1}^{K-1} (J̃^(k-1) - J̃^(k))

    J^(k) = (1/2) e^(k)⊤ Qf e^(k),  Eq. (14)
    e^(k) = [e_r^(k); e_v^(k)],  Eq. (15)
    """

    def __init__(self,
                 lambda_v:   float = 1.0,
                 lambda_pos: float = 1.0,
                 lambda_ps:  float = 0.1,
                 lambda_v0:  float = 0.0,        # joint training: supervise Head 1's output
                 head1_target: str = 'lambert',  # target for L_v0: 'lambert' or 'j2'
                 q_pos:      float = 1.0,
                 pos_scale_km: float = 200.0,
                 v_scale_kms:  float = 1.0,
                 log_pos_loss: bool = False,
                 pos_scale_per_sample: bool = False):
        super().__init__()
        self.lambda_v     = lambda_v
        self.lambda_pos   = lambda_pos
        self.lambda_ps    = lambda_ps
        self.lambda_v0    = lambda_v0
        self.head1_target = head1_target
        self.q_pos        = q_pos
        self.pos_scale    = pos_scale_km
        self.v_scale      = v_scale_kms
        self.log_pos_loss = log_pos_loss
        self.pos_scale_per_sample = pos_scale_per_sample

    def forward(self, out: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        v1_iters   = out['v1_iters']
        pos_errors = out['pos_errors']    # list of (B,) km, length K+1 (K iters + final propagation)
        vf_iters   = out['vf_iters']      # list of (B,3) km/s, length K+1
        rf_iters   = out['rf_iters']      # list of (B,3) km, length K+1

        target_dtype = v1_iters[0].dtype
        v1_lambert = batch['v1_lambert'].to(dtype=target_dtype)  # (B, 3) km/s
        v1_true    = batch['v1_true'].to(dtype=target_dtype)     # (B, 3) km/s
        r2         = batch['r2'].to(dtype=target_dtype)          # (B, 3) km

        # ── Velocity supervision terms ────────────────────────────────────────
        # L_v0: Head 1's first-guess (joint training only — λ_v0 = 0 by default).
        #       Target depends on what Head 1 is meant to learn:
        #       'lambert' for direct mode (learned_lambert), 'j2' for residual mode.
        # L_vK: final TRC output vs J2-corrected ground truth.
        # Both normalised by dv_scale² so they sit on the same order as L_pos.
        head1_target_v = v1_true if self.head1_target == 'j2' else v1_lambert
        L_v0 = F.mse_loss(v1_iters[0], head1_target_v) / (self.v_scale ** 2)
        L_vK = F.mse_loss(v1_iters[-1], v1_true) / (self.v_scale ** 2)
        L_v  = L_vK

        # ── Terminal position error ───────────────────────────────────────────
        # Penalise how far the final propagated position misses r2.
        # log1p loss handles the wide dynamic range of Jupiter-scale problems
        # (errors span 40M km → ~100 km over training); MSE explodes early on.
        if len(pos_errors) > 0:
            if self.pos_scale_per_sample:
                scale = torch.norm(batch['r1'].to(dtype=target_dtype), dim=-1)  # (B,) km
            else:
                scale = self.pos_scale
            if self.log_pos_loss:
                L_pos = torch.log1p(pos_errors[-1] / scale).mean()
            else:
                L_pos = (pos_errors[-1] / scale).pow(2).mean()
        else:
            L_pos = torch.tensor(0.0, device=v1_lambert.device)

        # ── Process supervision (Eq. 13-15) ──────────────────────────────────
        L_ps = torch.tensor(0.0, device=v1_lambert.device)
        # K_ref = number of refinement iterations (exclude the final propagation).
        # pos_errors has K_ref+1 elements; process supervision covers only the K_ref
        # refinement steps, giving K_ref-1 difference terms as per the paper (Eq. 13-15).
        K_ref = len(pos_errors) - 1
        if K_ref >= 2:
            # Process supervision: reward the network for reducing the cost J at every iteration,
            # not just the last one. J^(k) measures combined position error at step k.
            # Build J^(k) for k=0..K_ref-1 (the K_ref refinement propagations only).
            J_vals = []
            for k in range(K_ref):
                e_r = rf_iters[k] - r2                    # (B, 3) position error at arrival
                J_k = 0.5 * self.q_pos * (e_r ** 2).sum(-1)
                J_vals.append(J_k)

            # Normalise by the initial cost so the scale doesn't matter.
            # Clamp J0 from below using a meaningful scale (1 km² ≈ 1 km error)
            # to prevent near-zero initial errors from inflating J_normed unboundedly.
            J0 = J_vals[0].detach().clamp(min=1.0)
            J_normed = [torch.clamp(J / J0, max=10.0) for J in J_vals]   # Ĵ^(k) = J^(k) / J^(0)

            # - (1/(K_ref-1)) Σ_{k=1}^{K_ref-1} (J̃^(k-1) - J̃^(k))  encourages monotonic decrease
            ps_terms = []
            for k in range(1, K_ref):
                ps_terms.append(J_normed[k - 1] - J_normed[k])
            # Negative because we want J to decrease — maximise (J^(k-1) - J^(k))
            L_ps = -torch.stack(ps_terms).mean()

        # ── Total loss ───────────────────────────────────────────────────────
        loss = (self.lambda_v0  * L_v0
                + self.lambda_v   * L_v
                + self.lambda_pos * L_pos
                + self.lambda_ps  * L_ps)

        with torch.no_grad():
            dv_corr_m = (torch.norm(v1_iters[-1] - v1_lambert, dim=-1).mean()
                         * 1000).item()
            pos_err_0 = pos_errors[0].mean().item() if pos_errors else 0.0
            pos_err_K = pos_errors[-1].mean().item() if pos_errors else 0.0

        return loss, {
            'loss':      loss.item(),
            'L_v':       L_v.item(),
            'L_v0':      L_v0.item(),
            'L_vK':      L_vK.item(),
            'L_pos':     L_pos.item(),
            'L_ps':      L_ps.item(),
            'pos_err_0': pos_err_0,
            'pos_err_K': pos_err_K,
            'dv_corr_m': dv_corr_m,
        }


class LambertTRCJ2Stage1Loss(nn.Module):
    """Stage-1 loss: train Head 1's velocity output.

    target='lambert' supervises against v1,Lambert  (paper Eq. 4).
    target='j2'      supervises against v1,J2-corrected — pair this with
                     head1_mode='residual' so the network learns the J2
                     correction Δv = v1_j2 - v1_lambert.
    """

    def __init__(self, target: str = 'lambert'):
        super().__init__()
        if target not in ('lambert', 'j2'):
            raise ValueError(f"target must be 'lambert' or 'j2', got {target!r}")
        self.target = target

    def forward(self, out: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        v1_pred = out['v1_pred']
        target_key = 'v1_lambert' if self.target == 'lambert' else 'v1_true'
        v1_target = batch[target_key].to(dtype=v1_pred.dtype)
        loss = F.mse_loss(v1_pred, v1_target)
        return loss, {
            'loss':    loss.item(),
            'v1_err_ms': (torch.norm(v1_pred - v1_target, dim=-1).mean() * 1000).item(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    # ── Experiment selector ───────────────────────────────────────────────────
    # variant:  'pos_only'        – Lambert given, position loss only (λ_v=0)
    #           'vel_supervised'  – Lambert given, position + velocity loss (λ_v>0)
    #           'learned_lambert' – Head 1 learns Lambert, then Head 2 corrects
    # dataset:  'single_rev'  → train/val_struct.mat
    #           'multi_rev'   → train/val_struct_10h.mat
    #           'jovian'      → train/val_struct_jovian.mat
    # Setting both auto-fills run_name, train_mat, val_mat, skip_stage1, lambda_v.
    # Leave as '' to configure manually.
    variant:        str   = ''
    dataset:        str   = ''

    # Stage 1
    epochs_s1:      int   = 1000
    lr_s1:          float = 1e-3
    resume_stage1_ckpt: str | None = None
    resume_stage1_additional_epochs: int = 0
    stop_after_stage1: bool = False
    # Stage 2
    skip_stage1:    bool  = False      # skip Stage 1 entirely, load best Stage 1 ckpt and go straight to Stage 2
    resume_stage2_ckpt: str | None = None  # resume Stage 2 from this checkpoint
    resume_stage2_additional_epochs: int = 0
    reset_opt:      bool  = False      # on resume, load only model weights (fresh optimizer + scheduler)
    epochs_s2:      int   = 300
    lr_s2:          float = 1e-4
    # Shared
    batch_size:     int   = 512
    grad_clip:      float = 1.0
    K:              int   = 3
    max_batches:    int   = 0         # 0 = use full dataset; >0 caps batches per epoch
    max_train_samples: int = 0        # 0 = use full dataset; >0 subsamples training set
    # Loss weights
    lambda_v:       float = 1.0
    lambda_pos:     float = 1.0
    lambda_ps:      float = 0.1
    lambda_v0:      float = 0.0   # joint training: weight on Head 1 supervision (L_v0).
                                  # 0 = behave as before (separate Stage 1).
                                  # >0 = supervise v1^(0) inline against stage1_target.
    # Architecture
    d_z:            int   = 256
    d_h:            int   = 512
    n_blocks:       int   = 3
    n_inner:        int   = 6
    n_heads:        int   = 8
    # Misc
    v_max:          float = 15.0
    pos_scale_km:      float = 200.0   # input feature scaling for the error encoder [km]
    loss_pos_scale_km: float = 100.0   # loss scaling for L_pos — smaller = stronger position signal
    log_pos_loss:      bool  = False   # use log1p loss for L_pos (better for wide dynamic range)
    pos_scale_per_sample: bool = False # divide pos error by per-sample ||r1|| instead of a scalar
    body:           str   = 'earth'  # gravitational body: 'earth' or 'jupiter'
    max_step_s:     float = 30.0
    train_max_step_s: float = 0.0    # coarser RK4 step for training propagator (0 = same as max_step_s)
                                     # Reduces n_max in the Python loop — critical for jovian speed.
                                     # Val eval always uses max_step_s.
    oracle_init:    bool  = True       # True = pass v1_lambert to Head 2; False = use Head 1 output
    head1_mode:     str   = 'direct'   # 'direct' | 'oracle' | 'residual'  (see LambertTRCJ2 docstring)
    stage1_target:  str   = 'lambert'  # Stage 1 supervises Head 1 against 'lambert' or 'j2'
    ckpt_dir:       str   = 'checkpoints'
    run_name:       str   = 'trc_j2'
    train_mat:      str   = 'data/leo_single_train_lambertpy.npz'
    val_mat:        str   = 'data/leo_single_val_lambertpy.npz'
    dv_pct_cutoff:  float = None   # drop samples with |Δv| ≥ p<cutoff> (e.g. 90 → keep bottom 90%)

    # The *_lambertpy.npz files in data/ already use the field schema this loader
    # expects (r1/r2/tof/v1/v1_j2/v2/nrev/ncase/prograde + j2_* diagnostics, all float64).
    # The non-lambertpy siblings use a different schema (v1_lambert/v1_true/n_rev/branch)
    # and would need a remap; pick those up later if needed.
    _DATASET_FILES = {
        'single_rev': ('data/leo_single_train_lambertpy.npz', 'data/leo_single_val_lambertpy.npz'),
        'multi_rev':  ('data/leo_multi_train_lambertpy.npz',  'data/leo_multi_val_lambertpy.npz'),
        'jovian':     ('data/jovian_train_lambertpy.npz',     'data/jovian_val_lambertpy.npz'),
    }
    # (pos_scale_km, loss_pos_scale_km, max_step_s, log_pos_loss, body, v_max, batch_size, train_max_step_s)
    # jovian: orbits at ~1.5M km radius, J2 errors up to ~126k km, TOF up to 192 days
    # v_max for jovian: Lambert velocities reach 23.4 km/s — must exceed this
    # jovian train_max_step_s=14400 (4 hr): min orbit at 5 RJ is ~33 hr → ~8 steps/orbit.
    # Keeps n_max ≤ ceil(17.5M/14400)=1215 instead of 4862 — 4x fewer loop iterations.
    # The data-gen propagator (max_step_s=3600) is only used for validation metrics;
    # training uses the coarser step for speed.
    _DATASET_SCALES = {
        'single_rev': (200.0,   100.0,   30.0,   False, 'earth',   15.0, 512,    0.0),
        'multi_rev':  (200.0,   100.0,   45.0,   False, 'earth',   15.0, 512,    0.0),
        'jovian':     (50000.0, 10000.0, 3600.0, True,  'jupiter', 30.0, 512, 14400.0),
    }
    # Outer refinement iterations K per dataset (paper § "Validation Results"):
    # K=3 for single-revolution LEO, K=4 for multi-revolution LEO and Jovian.
    _DATASET_K = {
        'single_rev': 3,
        'multi_rev':  4,
        'jovian':     4,
    }
    # Variants A/B/C from the paper:
    #   A 'learned_lambert'  — Head 1 supervised against v1,Lambert (Stage 1, 500 ep, 1e-3→1e-6),
    #                          then Stage 2 refines with full loss (300 ep, 1e-5).  λ_v=1.0
    #   B 'vel_supervised'   — Head 2 only, oracle v1,Lambert init, position+velocity loss.  λ_v=1.0
    #   C 'pos_only'         — Head 2 only, oracle v1,Lambert init, position-only loss.       λ_v=0.0
    _VARIANT_DEFAULTS = {
        # (skip_stage1, oracle_init, lambda_v, lr_s2, epochs_s1, head1_mode, stage1_target)
        'pos_only':                (True,  True,  0.0,   1e-4,  300, 'oracle',   'lambert'),  # Variant C
        'vel_supervised':          (True,  True,  1.0,   1e-4,  300, 'oracle',   'lambert'),  # Variant B
        'learned_lambert':         (False, False, 1.0,   1e-5,  500, 'direct',   'lambert'),  # Variant A
        # Residual-Head-1 variants: Head 1 learns Δv = v1_j2 - v1_lambert (Stage 1),
        # Head 2 then refines further. v1_lambert is always passed in. (Not in the paper.)
        'pos_only_residual':       (False, False, 0.0,   1e-4, 1000, 'residual', 'j2'),
        'vel_supervised_residual': (False, False, 0.001, 1e-4, 1000, 'residual', 'j2'),
    }

    def __post_init__(self):
        if not self.variant and not self.dataset:
            return
        if self.dataset:
            if self.dataset not in self._DATASET_FILES:
                raise ValueError(f"Unknown dataset '{self.dataset}'. "
                                 f"Choose from: {list(self._DATASET_FILES)}")
            self.train_mat, self.val_mat = self._DATASET_FILES[self.dataset]
            ps, lps, ms, lpl, body, v_max, bs, tms = self._DATASET_SCALES[self.dataset]
            self.pos_scale_km      = ps
            self.loss_pos_scale_km = lps
            self.max_step_s        = ms
            self.log_pos_loss      = lpl
            self.body              = body
            self.v_max             = v_max
            self.batch_size        = bs
            self.train_max_step_s  = tms
            self.K                 = self._DATASET_K[self.dataset]
        if self.variant:
            if self.variant not in self._VARIANT_DEFAULTS:
                raise ValueError(f"Unknown variant '{self.variant}'. "
                                 f"Choose from: {list(self._VARIANT_DEFAULTS)}")
            (self.skip_stage1, self.oracle_init, self.lambda_v, self.lr_s2,
             self.epochs_s1, self.head1_mode, self.stage1_target) = self._VARIANT_DEFAULTS[self.variant]
        parts = [p for p in [self.variant, self.dataset] if p]
        self.run_name = 'trc_' + '_'.join(parts)


def compute_normalisation(dataset):
    """Compute normalisation statistics from a dataset (or Subset)."""
    # Unwrap Subset → access underlying tensors via indices
    if isinstance(dataset, torch.utils.data.Subset):
        base = dataset.dataset
        idx  = dataset.indices
        r1         = base.r1[idx]
        r2         = base.r2[idx]
        tof        = base.tof[idx]
        v1_lambert = base.v1_lambert[idx]
        v1_true    = base.v1_true[idx]
    else:
        r1         = dataset.r1
        r2         = dataset.r2
        tof        = dataset.tof
        v1_lambert = dataset.v1_lambert
        v1_true    = dataset.v1_true

    # Pool r1 and r2 together — both are positions, use same scale
    all_r = torch.cat([r1, r2], dim=0)   # (2N, 3)
    r_center = all_r.mean(dim=0)
    r_scale  = all_r.std(dim=0).clamp(min=1e-3)

    tof_mean = tof.mean().item()
    tof_std  = tof.std().item()

    # Per-component mean and std of Lambert velocities from training data
    v_mean = v1_lambert.mean(dim=0)
    v_std  = v1_lambert.std(dim=0).clamp(min=1e-3)

    # Mean magnitude of the J2 correction Δv = v1_true - v1_lambert.
    # Used to normalise L_vK in Stage 2 so it stays in the same order as L_pos.
    dv = v1_true - v1_lambert          # (N, 3) km/s
    dv_scale = torch.norm(dv, dim=-1).mean().clamp(min=1e-6).item()  # scalar km/s

    return {
        'r_center':  r_center.numpy(),
        'r_scale':   r_scale.numpy(),
        'tof_mean':  tof_mean,
        'tof_std':   max(tof_std, 1.0),
        'v_mean':    v_mean.numpy(),
        'v_std':     v_std.numpy(),
        'dv_scale':  dv_scale,   # mean J2 correction magnitude [km/s]
    }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Runs one full pass through all training batches.
# For each batch: forward pass → compute loss → backpropagate → update weights.
def train_one_epoch(model, loader, optimizer, criterion, device,
                    stage: int, K: int, grad_clip: float = 1.0,
                    max_batches: int = 0, oracle_init: bool = True,
                    head1_mode: str = 'direct'):
    model.train()
    stats_sum = {}
    n_batches = 0

    # Residual Head 1 always needs v1_lambert (used as the base of v1 = v1_lambert + Δv).
    # In direct/oracle modes, oracle_init still controls Stage-2 behaviour as before.
    needs_lambert_stage1 = head1_mode == 'residual'
    needs_lambert_stage2 = head1_mode == 'residual' or oracle_init

    for batch in loader:
        if max_batches > 0 and n_batches >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        r1       = batch['r1']
        r2       = batch['r2']
        tof      = batch['tof']
        prograde = batch['prograde']

        # Clear gradients from the previous batch — always do this before backward()
        optimizer.zero_grad()
        nrev  = batch.get('nrev')
        ncase = batch.get('ncase')
        if stage == 1:
            init_v1 = batch['v1_lambert'] if needs_lambert_stage1 else None
            out = model(r1, r2, tof, prograde, nrev=nrev, ncase=ncase,
                        stage1_only=True, init_v1=init_v1)
        else:
            init_v1 = batch['v1_lambert'] if needs_lambert_stage2 else None
            out = model(r1, r2, tof, prograde, nrev=nrev, ncase=ncase,
                        K=K, init_v1=init_v1)

        loss, metrics = criterion(out, batch)
        if not torch.isfinite(loss):
            print(f"  [warn] non-finite loss={loss.item():.4g}, skipping batch")
            optimizer.zero_grad()
            continue
        # Backpropagation: compute how much each weight contributed to the loss
        loss.backward()
        if stage == 2:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # Update weights using the computed gradients
        optimizer.step()

        # Accumulate metrics so we can average over all batches at the end
        for k, v in metrics.items():
            stats_sum[k] = stats_sum.get(k, 0.0) + v
        n_batches += 1

    if n_batches == 0:
        return {k: float('nan') for k in stats_sum}
    return {k: v / n_batches for k, v in stats_sum.items()}


# Same as train_one_epoch but no weight updates.
# @torch.no_grad() disables gradient tracking — faster and uses less memory.
@torch.no_grad()
def eval_epoch(model, loader, criterion, device, stage: int, K: int,
               oracle_init: bool = True, head1_mode: str = 'direct'):
    model.eval()
    stats_sum = {}
    n_batches = 0

    needs_lambert_stage1 = head1_mode == 'residual'
    needs_lambert_stage2 = head1_mode == 'residual' or oracle_init

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        r1       = batch['r1']
        r2       = batch['r2']
        tof      = batch['tof']
        prograde = batch['prograde']

        nrev  = batch.get('nrev')
        ncase = batch.get('ncase')
        if stage == 1:
            init_v1 = batch['v1_lambert'] if needs_lambert_stage1 else None
            out = model(r1, r2, tof, prograde, nrev=nrev, ncase=ncase,
                        stage1_only=True, init_v1=init_v1)
        else:
            init_v1 = batch['v1_lambert'] if needs_lambert_stage2 else None
            out = model(r1, r2, tof, prograde, nrev=nrev, ncase=ncase,
                        K=K, init_v1=init_v1)

        _, metrics = criterion(out, batch)
        for k, v in metrics.items():
            stats_sum[k] = stats_sum.get(k, 0.0) + v
        n_batches += 1

    if n_batches == 0:
        return {k: float('nan') for k in stats_sum}
    return {k: v / n_batches for k, v in stats_sum.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN TRAINING SCRIPT
# ═══════════════════════════════════════════════════════════════════════════════

def train(cfg: TrainConfig = None):
    if cfg is None:
        cfg = TrainConfig()

    # Pick the best available hardware:
    # CUDA = NVIDIA GPU, MPS = Apple Silicon GPU, cpu = fallback
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Load datasets ─────────────────────────────────────────────────────────
    print("\n=== Loading datasets ===")
    train_ds = MatLambertDataset(cfg.train_mat, 'train_info', dv_pct_cutoff=cfg.dv_pct_cutoff)
    val_ds   = MatLambertDataset(cfg.val_mat,   'val_info',   dv_pct_cutoff=cfg.dv_pct_cutoff)

    if cfg.max_train_samples > 0:
        base = train_ds.dataset if isinstance(train_ds, torch.utils.data.Subset) else train_ds
        idx = torch.randperm(len(train_ds))[:cfg.max_train_samples]
        train_ds = torch.utils.data.Subset(base, idx.tolist())
        print(f"  Subsampled training set to {cfg.max_train_samples} samples")

    # Sort training set by TOF so each batch's n_max is driven by its own samples,
    # not the global maximum.  Without sorting, every batch runs n_max = ceil(tof_max/dt)
    # even if its samples have short TOF — wasting 10–100x compute for Jupiter.
    # shuffle=False is intentional: the LR schedule provides implicit randomisation,
    # and the TOF ordering creates a natural curriculum (short → long transfers).
    def _sorted_train_loader(ds, batch_size):
        base = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
        base_idx = list(ds.indices) if isinstance(ds, torch.utils.data.Subset) else list(range(len(base)))
        tof_vals = base.tof[base_idx]
        order    = torch.argsort(tof_vals).tolist()
        sorted_idx = [base_idx[i] for i in order]
        return DataLoader(torch.utils.data.Subset(base, sorted_idx),
                          batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=False)

    train_loader = _sorted_train_loader(train_ds, cfg.batch_size)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    # Effective step sizes: training uses coarser propagator for speed; val uses accurate one.
    _train_step = cfg.train_max_step_s if cfg.train_max_step_s > 0 else cfg.max_step_s
    _val_step   = cfg.max_step_s
    if _train_step != _val_step:
        n_max_train = int(np.ceil(17.5e6 / _train_step))   # worst-case (Jupiter 10T)
        n_max_val   = int(np.ceil(17.5e6 / _val_step))
        print(f"\n  Propagator: train_max_step={_train_step:.0f}s (n_max≤{n_max_train})  "
              f"val_max_step={_val_step:.0f}s (n_max≤{n_max_val})  "
              f"→ ~{n_max_val//n_max_train}x fewer loop iters per train batch")

    # ── Normalisation ─────────────────────────────────────────────────────────
    print("\n=== Computing normalisation statistics ===")
    # Compute mean/std from the TRAINING set only — never use val data for this.
    # These statistics are also saved in the checkpoint so eval uses the same scale.
    norm_stats = compute_normalisation(train_ds)
    print(f"  r_center : {norm_stats['r_center']}")
    print(f"  r_scale  : {norm_stats['r_scale']}")
    print(f"  tof_mean : {norm_stats['tof_mean']:.1f} s")
    print(f"  v_scale  : {norm_stats['v_std'][0]:.3f} km/s")
    print(f"  dv_scale : {norm_stats['dv_scale']*1000:.2f} m/s  (mean J2 correction)")

    # ── Build model ───────────────────────────────────────────────────────────
    print("\n=== Building model ===")
    net = NetConfig(d_z=cfg.d_z, d_h=cfg.d_h,
                    n_heads=cfg.n_heads, n_blocks=cfg.n_blocks,
                    K=cfg.K, n_inner=cfg.n_inner)
    mu, re, j2 = BODY_PARAMS[cfg.body]
    print(f"  Body: {cfg.body}  (mu={mu:.4g} km³/s², re={re:.1f} km, j2={j2:.5g})")
    model = LambertTRCJ2(net,
                         max_step_s=cfg.max_step_s,
                         v_max=cfg.v_max,
                         pos_scale_km=cfg.pos_scale_km,
                         mu=mu, re=re, j2=j2,
                         head1_mode=cfg.head1_mode).to(device)
    model.set_normalisation(**{k: v for k, v in norm_stats.items() if k != 'dv_scale'},
                            pos_scale_km=cfg.pos_scale_km)
    print(f"  Parameters: {count_parameters(model):,}  head1_mode={cfg.head1_mode}  "
          f"input_dim={model.input_dim}")

    s1_ckpt = Path(cfg.resume_stage1_ckpt) if cfg.resume_stage1_ckpt else (
        ckpt_dir / f'{cfg.run_name}_stage1_best.pt'
    )

    if not cfg.skip_stage1:
        # ── Stage 1: train Head 1 (state encoder + init decoder) ─────────────
        print("\n" + "="*60)
        print("=== STAGE 1: Initial Guess Head (paper Eq. 4) ===")
        print("="*60)

        crit_s1 = LambertTRCJ2Stage1Loss(target=cfg.stage1_target)
        print(f"  Stage 1 supervises Head 1 against v1_{cfg.stage1_target}")
        # Only optimise Head 1 parameters (state_encoder + init_decoder).
        # Head 2 parameters exist but receive no gradient updates in Stage 1.
        opt_s1  = torch.optim.AdamW(
            list(model.state_encoder.parameters()) +
            list(model.init_decoder.parameters()),
            lr=cfg.lr_s1, weight_decay=1e-5)
        stage1_start_epoch = 0
        stage1_end_epoch = cfg.epochs_s1
        best_v1_err = float('inf')

        if cfg.resume_stage1_ckpt is not None:
            resume_ckpt = torch.load(cfg.resume_stage1_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(resume_ckpt['model'])
            if 'opt_s1' in resume_ckpt:
                opt_s1.load_state_dict(resume_ckpt['opt_s1'])
            norm_stats = resume_ckpt.get('norm', norm_stats)
            best_v1_err = float(resume_ckpt.get('v1_err_ms', float('inf')))
            stage1_start_epoch = int(resume_ckpt.get('epoch', 0))
            stage1_end_epoch = stage1_start_epoch + cfg.resume_stage1_additional_epochs
            print(
                f"  Resuming Stage 1 from {cfg.resume_stage1_ckpt} "
                f"(epoch {stage1_start_epoch}, best val err {best_v1_err:.1f} m/s)"
            )

        # Cosine annealing: learning rate starts at lr_s1 and smoothly decays to eta_min.
        # This lets the network take big steps early and fine-tune carefully at the end.
        # Paper § Training Methodology: Variant A Stage 1 anneals lr 1e-3 → 1e-6.
        sched_s1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_s1, T_max=max(stage1_end_epoch, 1), eta_min=1e-6)
        if stage1_start_epoch > 0:
            for _ in range(stage1_start_epoch):
                sched_s1.step()

        if stage1_end_epoch <= stage1_start_epoch:
            print("  Stage 1 resume requested with no additional epochs; skipping Stage 1 updates.")

        for epoch in range(stage1_start_epoch + 1, stage1_end_epoch + 1):
            tr = train_one_epoch(model, train_loader, opt_s1, crit_s1,
                                 device, stage=1, K=cfg.K,
                                 grad_clip=cfg.grad_clip,
                                 max_batches=cfg.max_batches,
                                 head1_mode=cfg.head1_mode)
            val = eval_epoch(model, val_loader, crit_s1,
                             device, stage=1, K=cfg.K,
                             head1_mode=cfg.head1_mode)
            sched_s1.step()

            if True:
                print(f"  [S1 {epoch:4d}/{stage1_end_epoch}] "
                      f"train_loss={tr['loss']:.4f}  "
                      f"val_v1_err={val['v1_err_ms']:.1f} m/s  "
                      f"lr={sched_s1.get_last_lr()[0]:.2e}")

            # Save checkpoint whenever we get a new best validation error.
            # This ensures we keep the best model even if training degrades later.
            if val['v1_err_ms'] < best_v1_err:
                best_v1_err = val['v1_err_ms']
                torch.save({'model': model.state_dict(),
                            'norm':  norm_stats,
                            'cfg':   cfg,
                            'opt_s1': opt_s1.state_dict(),
                            'sched_s1': sched_s1.state_dict(),
                            'epoch': epoch,
                            'v1_err_ms': best_v1_err},
                           s1_ckpt)

        print(f"\n  Stage 1 best val v1 error: {best_v1_err:.1f} m/s")
        print(f"  Checkpoint: {s1_ckpt}")

        # Always load the best Stage 1 checkpoint before Stage 2
        # (in-memory state may be the final epoch, which can be worse than best)
        ckpt = torch.load(s1_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        print(f"  Loaded best Stage 1 weights (epoch {ckpt.get('epoch','?')}, "
              f"val v1 err {ckpt.get('v1_err_ms', '?'):.1f} m/s)")

        if cfg.stop_after_stage1:
            return model

    # ── Single training phase ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("=== TRAINING: Full TRC (paper Eq. 13) ===")
    print("="*60)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")

    crit_s2 = LambertTRCJ2Loss(
        lambda_v=cfg.lambda_v,
        lambda_pos=cfg.lambda_pos,
        lambda_ps=cfg.lambda_ps,
        lambda_v0=cfg.lambda_v0,
        head1_target=cfg.stage1_target,
        pos_scale_km=cfg.loss_pos_scale_km,
        v_scale_kms=norm_stats['dv_scale'],
        log_pos_loss=cfg.log_pos_loss,
        pos_scale_per_sample=cfg.pos_scale_per_sample,
    )
    if cfg.lambda_v0 > 0:
        print(f"  Joint training: λ_v0={cfg.lambda_v0:g}  Head 1 supervised against v1_{cfg.stage1_target}")
    opt_s2   = torch.optim.AdamW(trainable, lr=cfg.lr_s2, weight_decay=0.0)
    sched_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_s2, T_max=cfg.epochs_s2, eta_min=1e-7)

    best_pos_err  = float('inf')
    start_epoch   = 1
    s2_ckpt       = ckpt_dir / f'{cfg.run_name}_best.pt'
    s2_last_ckpt  = ckpt_dir / f'{cfg.run_name}_last.pt'

    # Snapshot any existing best.pt before Stage 2 can touch it. Guards against
    # training bugs, bad flags, or operator error clobbering a converged run.
    if s2_ckpt.exists():
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup = ckpt_dir / f'{cfg.run_name}_best.BACKUP_{ts}.pt'
        shutil.copy2(s2_ckpt, backup)
        try:
            prev = torch.load(s2_ckpt, map_location='cpu', weights_only=False)
            prev_err = prev.get('best_pos_err', float('inf'))
            print(f"\n  Backed up existing best.pt (prev best {prev_err:.2f} km) → {backup}")
        except Exception:
            print(f"\n  Backed up existing best.pt → {backup}")

    # Resume from last checkpoint if requested
    resume_path = cfg.resume_stage2_ckpt or (
        s2_last_ckpt if cfg.resume_stage2_ckpt == '' else None)
    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        norm_stats = ckpt.get('norm', norm_stats)
        if 'dv_scale' not in norm_stats:
            norm_stats['dv_scale'] = compute_normalisation(train_ds)['dv_scale']
        # Always carry best_pos_err from the ckpt so a warm-start cannot
        # overwrite a better minimum with a worse one.
        best_pos_err = ckpt.get('best_pos_err', float('inf'))
        if cfg.reset_opt:
            print(f"\n  Warm-start from {resume_path}  "
                  f"(model weights only; fresh optimizer + scheduler, "
                  f"prev best pos_err {best_pos_err:.2f} km — will only overwrite best.pt on improvement)")
        else:
            if 'opt_s2' in ckpt:
                opt_s2.load_state_dict(ckpt['opt_s2'])
            if 'sched_s2' in ckpt:
                sched_s2.load_state_dict(ckpt['sched_s2'])
            start_epoch  = ckpt.get('epoch', 0) + 1
            print(f"\n  Resumed from {resume_path}  "
                  f"(epoch {start_epoch-1}, best pos_err {best_pos_err:.2f} km)")
    elif cfg.resume_stage2_ckpt:
        raise FileNotFoundError(f"Resume checkpoint not found: {cfg.resume_stage2_ckpt}")
    else:
        print("\n  Starting Stage 2 (no resume checkpoint)")

    for epoch in range(start_epoch, cfg.epochs_s2 + 1):
        model.propagator.max_step_s = _train_step   # coarse step for speed
        tr = train_one_epoch(model, train_loader, opt_s2, crit_s2,
                             device, stage=2, K=cfg.K,
                             grad_clip=cfg.grad_clip,
                             max_batches=cfg.max_batches,
                             oracle_init=cfg.oracle_init,
                             head1_mode=cfg.head1_mode)
        model.propagator.max_step_s = _val_step     # accurate step for metrics
        val = eval_epoch(model, val_loader, crit_s2,
                         device, stage=2, K=cfg.K,
                         oracle_init=cfg.oracle_init,
                         head1_mode=cfg.head1_mode)
        sched_s2.step()

        print(f"  [S2 {epoch:4d}/{cfg.epochs_s2}] "
              f"loss={tr['loss']:.4f}  "
              f"pos_err={val['pos_err_K']:.1f} km  "
              f"Δv={val['dv_corr_m']:.1f} m/s  "
              f"L_v={tr['L_v']:.4f}  L_pos={tr['L_pos']:.4f}  "
              f"L_ps={tr['L_ps']:.4f}  "
              f"lr={sched_s2.get_last_lr()[0]:.2e}")

        # Save latest state every epoch for reliable resume
        torch.save({'model':        model.state_dict(),
                    'norm':         norm_stats,
                    'cfg':          cfg,
                    'epoch':        epoch,
                    'best_pos_err': best_pos_err,
                    'opt_s2':       opt_s2.state_dict(),
                    'sched_s2':     sched_s2.state_dict()},
                   s2_last_ckpt)

        if val['pos_err_K'] < best_pos_err:
            best_pos_err = val['pos_err_K']
            torch.save({'model':        model.state_dict(),
                        'norm':         norm_stats,
                        'cfg':          cfg,
                        'epoch':        epoch,
                        'pos_err_km':   best_pos_err,
                        'best_pos_err': best_pos_err,
                        'opt_s2':       opt_s2.state_dict(),
                        'sched_s2':     sched_s2.state_dict()},
                       s2_ckpt)

    print(f"\n  Stage 2 best val pos error: {best_pos_err:.2f} km")
    print(f"  Checkpoint: {s2_ckpt}")

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 7. EVALUATION UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(ckpt_path: str,
             mat_path:  str = 'val_struct.mat',
             struct_key: str = 'val_info',
             K_list: list = None):
    """Evaluate a saved checkpoint on a dataset for multiple K values."""
    if K_list is None:
        K_list = [1, 2, 3, 5]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg    = ckpt['cfg']
    norm   = ckpt['norm']

    net = NetConfig(d_z=cfg.d_z, d_h=cfg.d_h,
                    n_heads=cfg.n_heads, n_blocks=cfg.n_blocks,
                    K=max(K_list), n_inner=cfg.n_inner)
    _mu, _re, _j2 = BODY_PARAMS.get(getattr(cfg, 'body', 'earth'), BODY_PARAMS['earth'])
    _head1_mode = getattr(cfg, 'head1_mode', None) or (
        'oracle' if getattr(cfg, 'oracle_init', False) else 'direct'
    )
    model = LambertTRCJ2(net,
                         max_step_s=cfg.max_step_s,
                         v_max=cfg.v_max,
                         pos_scale_km=cfg.pos_scale_km,
                         mu=_mu, re=_re, j2=_j2,
                         head1_mode=_head1_mode).to(device)
    model.set_normalisation(**{k: v for k, v in norm.items() if k != 'dv_scale'},
                            pos_scale_km=cfg.pos_scale_km)
    model.load_state_dict(ckpt['model'])
    model.eval()

    ds     = MatLambertDataset(mat_path, struct_key)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    print(f"\nEvaluation: {ckpt_path}")
    print(f"{'K':>4}  {'mean pos err':>14}  {'med pos err':>13}  "
          f"{'<10km':>7}  {'<1km':>6}  {'mean Δv (m/s)':>14}")
    print("-" * 72)

    for K in K_list:
        pos_errs = []
        dv_corrs = []

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(batch['r1'], batch['r2'], batch['tof'],
                          batch['prograde'],
                          nrev=batch.get('nrev'), ncase=batch.get('ncase'),
                          K=K, init_v1=batch['v1_lambert'])
            pos_err = out['pos_errors'][-1].cpu()    # (B,)
            dv_corr = (torch.norm(out['v1_pred'] - batch['v1_lambert'],
                                  dim=-1) * 1000).cpu()  # m/s
            pos_errs.append(pos_err)
            dv_corrs.append(dv_corr)

        pos_errs = torch.cat(pos_errs)
        dv_corrs = torch.cat(dv_corrs)

        print(f"  K={K}  {pos_errs.mean():>10.2f} km  "
              f"{pos_errs.median():>10.2f} km  "
              f"{(pos_errs < 10).float().mean()*100:>5.1f}%  "
              f"{(pos_errs < 1).float().mean()*100:>5.1f}%  "
              f"{dv_corrs.mean():>10.1f} m/s")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('variant', nargs='?', default='pos_only',
                    choices=['pos_only', 'vel_supervised', 'learned_lambert',
                             'pos_only_residual', 'vel_supervised_residual'])
    ap.add_argument('dataset', nargs='?', default='single_rev',
                    choices=['single_rev', 'multi_rev', 'jovian'])
    # Paper § Training Methodology: Stage 2 = 300 epochs across all variants.
    ap.add_argument('--epochs', type=int, default=300)
    # K defaults to the per-dataset value from _DATASET_K (3 for single_rev,
    # 4 for multi_rev/jovian, per paper).  Pass --K to override.
    ap.add_argument('--K',      type=int, default=None)
    ap.add_argument('--lr',     type=float, default=1e-4)
    ap.add_argument('--resume', action='store_true',
                    help='Resume from checkpoints/<run_name>_last.pt')
    ap.add_argument('--resume-from', type=str, default=None,
                    help='Resume from a specific checkpoint path')
    ap.add_argument('--reset-opt', action='store_true',
                    help='On resume, load only model weights (fresh optimizer + scheduler). '
                         'Use when the prior run hit the cosine LR floor.')
    ap.add_argument('--skip-stage1', action='store_true',
                    help='Skip Stage 1 and go straight to Stage 2 (useful when resuming)')
    ap.add_argument('--stop-after-stage1', action='store_true',
                    help='Train Stage 1 only, then exit. Useful for sanity-checking Head 1 alone, '
                         'especially with the *_residual variants where Head 1 learns the J2 correction.')
    ap.add_argument('--joint', action='store_true',
                    help='Skip the separate Stage-1 phase and train Head 1 + Head 2 jointly. '
                         'Adds an L_v0 term (weight --lambda-v0, default 1.0) supervising Head 1 '
                         'against stage1_target. Mostly useful with *_residual variants.')
    ap.add_argument('--lambda-v0', type=float, default=None,
                    help='Weight for the L_v0 (Head 1 supervision) loss term. Implies joint training '
                         'when > 0. Default: 0 (off), or 1.0 if --joint is set.')
    ap.add_argument('--max-step', type=float, default=None,
                    help='Max RK4 step size for J2 propagation [s]. Default: 30 s. '
                         'Use larger values (e.g. 3600) for long-TOF datasets like jovian.')
    ap.add_argument('--pos-scale', type=float, default=None,
                    help='Position scale for error encoder input [km]. '
                         'Default: 200 km (Earth). Use ~50000 for jovian.')
    ap.add_argument('--loss-pos-scale', type=float, default=None,
                    help='Position scale for L_pos loss [km]. '
                         'Default: 100 km (Earth). Use ~10000 for jovian.')
    ap.add_argument('--log-pos-loss', action='store_true',
                    help='Use log1p loss for L_pos instead of MSE. '
                         'Strongly recommended for jovian (errors span 40M→100 km).')
    ap.add_argument('--pos-scale-per-sample', action='store_true',
                    help='Divide pos error by per-sample ||r1|| (non-dim) instead of a fixed scale.')
    ap.add_argument('--train-max-step', type=float, default=None,
                    help='Coarser RK4 step for training propagator [s]. '
                         'Reduces the Python loop count (n_max) during training for speed. '
                         'Eval/filter always use --max-step. Default: auto (14400 for jovian).')
    ap.add_argument('--max-batches', type=int, default=0,
                    help='Cap batches per epoch (0 = full dataset). Useful for quick trials.')
    ap.add_argument('--dv-pct-cutoff', type=float, default=None,
                    help='Drop training/val samples with |Δv| ≥ p<cutoff> (e.g. 90 keeps bottom 90%%).')
    args = ap.parse_args()

    # Paper uses λ_pos = 1.0, λ_ps = 0.1, grad_clip = 1.0 across all variants —
    # those are already the TrainConfig defaults; don't override here.
    cfg = TrainConfig(
        variant=args.variant,
        dataset=args.dataset,
        epochs_s2=args.epochs,
    )
    # K: dataset post-init has set the paper value; only override if user passed --K.
    if args.K is not None:
        cfg.K = args.K
    if args.max_step is not None:
        cfg.max_step_s = args.max_step
    if args.train_max_step is not None:
        cfg.train_max_step_s = args.train_max_step
    if args.pos_scale is not None:
        cfg.pos_scale_km = args.pos_scale
    if args.loss_pos_scale is not None:
        cfg.loss_pos_scale_km = args.loss_pos_scale
    if args.log_pos_loss:
        cfg.log_pos_loss = True
    if args.pos_scale_per_sample:
        cfg.pos_scale_per_sample = True
    if args.max_batches > 0:
        cfg.max_batches = args.max_batches
    if args.dv_pct_cutoff is not None:
        cfg.dv_pct_cutoff = args.dv_pct_cutoff
    # lr_s2 default comes from variant; CLI --lr overrides it
    if args.lr != 1e-4:
        cfg.lr_s2 = args.lr
    if args.skip_stage1:
        cfg.skip_stage1 = True
    if args.stop_after_stage1:
        cfg.stop_after_stage1 = True
    if args.joint:
        cfg.skip_stage1 = True
        if args.lambda_v0 is None:
            cfg.lambda_v0 = 1.0
    if args.lambda_v0 is not None:
        cfg.lambda_v0 = args.lambda_v0
    if args.resume_from:
        cfg.resume_stage2_ckpt = args.resume_from
    elif args.resume:
        cfg.resume_stage2_ckpt = f'checkpoints/{cfg.run_name}_last.pt'
    if args.reset_opt:
        cfg.reset_opt = True
    print(f"\n=== Experiment: variant={cfg.variant}  dataset={cfg.dataset} ===")
    print(f"    run_name={cfg.run_name}  train={cfg.train_mat}  val={cfg.val_mat}")
    print(f"    skip_stage1={cfg.skip_stage1}  lambda_v={cfg.lambda_v}\n")
    train(cfg)
