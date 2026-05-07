"""
Lambert Training Data Generator for TRC
======================================
Generate supervised training data for LEO orbit transfers.

Each sample: (r0, v0, r_target, v_target, tof) -> (dV1, dV2)

Lambert gives the two-body optimal impulses. Mid-course corrections
are zero under Keplerian dynamics — TRC's refinement loop learns to
add them when deployed with J2+drag.


NEW (this version):
- Full lambertMR port including multi-revolution (Nrev > 0)
- Natural transfer filtering: bounded total dV, transfer-orbit perigee
  check, sub-period TOF, optional same-orbit sampling mode
- Computes J2 propagation mismatch for each accepted sample
- Stores J2 endpoint error metrics in dataset

Usage:
    python lambert_data.py
    python lambert_data.py --n_train 10000 --n_test 1000 --out_dir data
    python lambert_data.py --n_train 100 --n_test 20 --lambert_debug
"""

import time
from dataclasses import dataclass
from shooting_utils import shooting_correct
import numpy as np

from axisym_initial_guess import AxisymGuessConfig, sample_axisym_position_pair
from constants import MU_EARTH, R_EARTH, J2
from dynamics import propagate_j2, propagate_twobody
from lambert_solver import solve_lambert
from orbital_utils import oe_to_eci, orbital_period


# ── Editable Run Defaults (set these in-file if you prefer no CLI flags) ────
DEFAULT_N_TRAIN = 10000
DEFAULT_N_TEST = 1000
DEFAULT_OUT_DIR = 'data'
DEFAULT_SEED = 42
DEFAULT_QUICK = False
DEFAULT_OVERWRITE = False

DEFAULT_NO_VERIFY = False
DEFAULT_VERIFY_MAX_STEP = 45.0
DEFAULT_J2_MAX_STEP = 45.0
DEFAULT_LAMBERT_DEBUG = False
DEFAULT_MAX_ATTEMPT_FACTOR = 50

DEFAULT_DV_MAX = 5.0
DEFAULT_NO_DV_LIMIT = True
DEFAULT_INC_MIN_DEG = 0.0
DEFAULT_INC_MAX_DEG = 180.0
DEFAULT_TOF_MIN = 0.1
DEFAULT_TOF_MAX = 5.0
DEFAULT_ENABLE_MULTI_REV = True
DEFAULT_REQUIRE_MULTI_REV = False
DEFAULT_MULTI_REV_PROB = 1.0
DEFAULT_NREV_MIN = 0
DEFAULT_NREV_MAX = 5
DEFAULT_REQUIRE_MULTI_REV_CHEAPER = False
DEFAULT_BALANCE_SINGLE_MULTI = False

DEFAULT_AXISYM_INIT_GUESS = True
DEFAULT_AXISYM_RANDOM_YAW = False

DEFAULT_ADD_SHOOTING_LABELS = True
DEFAULT_SHOOTING_TOL = 1e-4
DEFAULT_SHOOTING_MAX_ITER = 50
DEFAULT_SHOOTING_PROP_MAX_STEP = 45.0
DEFAULT_SHOOTING_FD_STEP = 1e-6
DEFAULT_SHOOTING_DAMPING = 1.0


# ── Orbit Sampling ──────────────────────────────────────────────────────────

def generate_and_correct(n, seed, path, label):
    """Generate Lambert data, then apply shooting correction for ground truth."""
    if not FORCE_REGEN and path.exists():
        print(f'Using existing: {path}')
        return

    print(f'Generating {label} data ({n} samples)...')
    raw = gen_lambert_dataset(n, seed=seed, cfg=cfg, max_attempt_factor=200, verbose=True)

    # Shooting correction — get the true J2-optimal dV
    print(f'Applying shooting correction...')
    N = len(raw['r0'])
    dv1_corrected = np.zeros_like(raw['dv1'])
    dv_correction = np.zeros_like(raw['dv1'])
    converged = np.zeros(N, dtype=bool)

    t0 = time.time()
    for i in range(N):
        res = shooting_correct(
            raw['r0'][i], raw['v0'][i], raw['dv1'][i],
            raw['r_target'][i], float(raw['tof'][i]),
            max_step=45.0, tol=1e-3, max_iter=30,
        )
        dv1_corrected[i] = res['dv_corrected']
        dv_correction[i] = res['dv_correction']
        converged[i] = res['converged']

        if (i + 1) % max(1, N // 5) == 0:
            corr_ms = np.linalg.norm(res['dv_correction']) * 1000
            print(f'  [{i+1:4d}/{N}] |Δv|={corr_ms:.1f} m/s  '
                  f'err: {res["pos_err_lambert"]:.1f}→{res["pos_err_final"]:.4f} km')

    elapsed = time.time() - t0
    n_conv = converged.sum()
    print(f'  Converged: {n_conv}/{N} ({n_conv/N*100:.0f}%) in {elapsed:.0f}s')

    # Add to dataset
    raw['dv1_corrected'] = dv1_corrected.astype(np.float32)
    raw['dv_correction'] = dv_correction.astype(np.float32)
    raw['dv_correction_mag'] = np.linalg.norm(dv_correction, axis=-1).astype(np.float32)
    raw['shooting_converged'] = converged

    np.savez(path, **raw)
    print(f'Saved to {path}')

    # Summary
    conv = converged
    if conv.any():
        corr_m = np.linalg.norm(dv_correction[conv], axis=-1) * 1000
        print(f'  Correction magnitude: {corr_m.mean():.1f} ± {corr_m.std():.1f} m/s')


@dataclass
class OrbitConfig:
    """Ranges for random orbit sampling."""
    alt_min: float = 300.0      # km above Earth surface
    alt_max: float = 600.0      # km
    ecc_min: float = 0.0
    ecc_max: float = 0.05       # near-circular LEO
    inc_min: float = 0.0        # rad
    inc_max: float = np.pi  # up to 180 deg
    tof_periods_min: float = 0.5
    tof_periods_max: float = 1.5  # sub-period → natural transfers
    dv_max: float = 10.0          # km/s total dV cap for natural transfers
    nearby: bool = True
    delta_alt: float = 50.0      # km (tight for natural transfers)
    delta_inc: float = 0.05      # rad (~3 deg)
    delta_ecc: float = 0.01
    # Transfer orbit safety
    min_perigee_alt: float = 100.0  # km above Earth — reject suborbital transfers
    # Transfer angle bounds (avoid near-0 and near-360 deg)
    min_transfer_angle: float = 0.0
    max_transfer_angle: float = 2 * np.pi
    # Optional axisymmetry-aware geometry sampler for (r0, r_target)
    axisym_init_guess: bool = False
    axisym_random_yaw: bool = False
    # Optional multi-revolution Lambert sampling
    enable_multi_rev: bool = False
    require_multi_rev: bool = False
    multi_rev_prob: float = 0.0
    nrev_min: int = 1
    nrev_max: int = 1
    require_multi_rev_cheaper: bool = False
    balance_single_multi: bool = False


def sample_orbit(rng, cfg=None):
    """Sample random orbital elements and convert to ECI state."""
    if cfg is None:
        cfg = OrbitConfig()

    alt = rng.uniform(cfg.alt_min, cfg.alt_max)
    a = R_EARTH + alt
    e = rng.uniform(cfg.ecc_min, cfg.ecc_max)
    inc = rng.uniform(cfg.inc_min, cfg.inc_max)
    raan = 0.0
    aop = rng.uniform(0, 2 * np.pi)
    nu = rng.uniform(0, 2 * np.pi)

    rp = a * (1 - e)
    if rp < R_EARTH + 150:
        e = max(0.0, 1.0 - (R_EARTH + 150) / a)

    r, v = oe_to_eci(a, e, inc, raan, aop, nu)
    oe = {'a': a, 'e': e, 'i': inc, 'raan': raan, 'aop': aop, 'nu': nu}
    return r, v, oe


def sample_nearby_orbit(rng, oe_ref, cfg):
    """Sample an arrival orbit near a reference departure orbit."""
    a_ref = oe_ref['a']
    alt_ref = a_ref - R_EARTH
    new_alt = np.clip(alt_ref + rng.uniform(-cfg.delta_alt, cfg.delta_alt),
                      cfg.alt_min, cfg.alt_max)
    a = R_EARTH + new_alt

    e = np.clip(oe_ref['e'] + rng.uniform(-cfg.delta_ecc, cfg.delta_ecc),
                cfg.ecc_min, cfg.ecc_max)
    inc = np.clip(oe_ref['i'] + rng.uniform(-cfg.delta_inc, cfg.delta_inc),
                  cfg.inc_min, cfg.inc_max)

    raan = 0.0
    aop = rng.uniform(0, 2 * np.pi)
    nu = rng.uniform(0, 2 * np.pi)

    rp = a * (1 - e)
    if rp < R_EARTH + 150:
        e = max(0.0, 1.0 - (R_EARTH + 150) / a)

    r, v = oe_to_eci(a, e, inc, raan, aop, nu)
    oe = {'a': a, 'e': e, 'i': inc, 'raan': raan, 'aop': aop, 'nu': nu}
    return r, v, oe


def _safe_unit(vec):
    n = np.linalg.norm(vec)
    if n < 1e-12:
        return None
    return vec / n


def _coplanar_circular_velocity(r_vec, h_hat):
    """Circular in-plane velocity at r_vec with orbit normal h_hat."""
    r_hat = _safe_unit(r_vec)
    if r_hat is None:
        return None
    t_hat = _safe_unit(np.cross(h_hat, r_hat))
    if t_hat is None:
        return None
    v_mag = np.sqrt(MU_EARTH / np.linalg.norm(r_vec))
    return v_mag * t_hat


def sample_axisym_transfer_state(rng, cfg):
    """Sample (r0, v0, r_target, v_target) using axisymmetry-aware geometry."""
    axis_cfg = AxisymGuessConfig(
        alt_min_km=cfg.alt_min,
        alt_max_km=cfg.alt_max,
        inc_min_rad=cfg.inc_min,
        inc_max_rad=cfg.inc_max,
        transfer_angle_min_rad=cfg.min_transfer_angle,
        transfer_angle_max_rad=cfg.max_transfer_angle,
        apply_random_yaw=cfg.axisym_random_yaw,
    )
    sample = sample_axisym_position_pair(rng, axis_cfg)

    r0 = sample['r1'].astype(np.float64)
    r_target = sample['r2'].astype(np.float64)

    h_hat = _safe_unit(np.cross(r0, r_target))
    if h_hat is None:
        return None
    # Canonicalize to prograde orientation so sampled inclinations stay in [0, 90] deg.
    if h_hat[2] < 0.0:
        h_hat = -h_hat

    v0 = _coplanar_circular_velocity(r0, h_hat)
    v_target = _coplanar_circular_velocity(r_target, h_hat)
    if v0 is None or v_target is None:
        return None

    oe_dep = {
        'a': float(np.linalg.norm(r0)),
        'e': 0.0,
        'i': float(sample['i']),
        'raan': 0.0,
        'aop': 0.0,
        'nu': float(sample['phi_1']),
    }
    oe_arr = {
        'a': float(np.linalg.norm(r_target)),
        'e': 0.0,
        'i': float(sample['i']),
        'raan': 0.0,
        'aop': 0.0,
        'nu': float(sample['phi_2']),
    }
    return r0, v0, r_target, v_target, oe_dep, oe_arr


# ── Dataset Generation ──────────────────────────────────────────────────────

def check_transfer_orbit(r0, v1_lambert, cfg):
    """Check that the transfer orbit doesn't dip below minimum altitude.

    Returns True if the transfer is safe, False otherwise.
    """
    r0_mag = np.linalg.norm(r0)
    v_mag2 = np.dot(v1_lambert, v1_lambert)
    energy = 0.5 * v_mag2 - MU_EARTH / r0_mag

    # Hyperbolic or parabolic — skip (shouldn't happen for LEO transfers)
    if energy >= 0:
        return False

    a_transfer = -MU_EARTH / (2.0 * energy)
    h_vec = np.cross(r0, v1_lambert)
    h_mag2 = np.dot(h_vec, h_vec)
    p_transfer = h_mag2 / MU_EARTH
    e_transfer = np.sqrt(max(1.0 - p_transfer / a_transfer, 0.0))
    rp_transfer = a_transfer * (1.0 - e_transfer)

    return rp_transfer >= R_EARTH + cfg.min_perigee_alt


def compute_transfer_angle(r0, r_target, h_hat=None):
    """Compute transfer angle between two position vectors in [0, 2*pi)."""
    cos_th = np.dot(r0, r_target) / (np.linalg.norm(r0) * np.linalg.norm(r_target))
    cos_th = np.clip(cos_th, -1.0, 1.0)
    cr = np.cross(r0, r_target)
    if h_hat is None:
        sin_th = np.linalg.norm(cr) / (np.linalg.norm(r0) * np.linalg.norm(r_target))
    else:
        sin_th = np.dot(h_hat, cr) / (np.linalg.norm(r0) * np.linalg.norm(r_target))
    angle = np.arctan2(sin_th, cos_th)
    if angle < 0:
        angle += 2.0 * np.pi
    return angle


def generate_sample(
    rng, cfg=None,
    verify=True, verify_tol=1.0, verify_max_step=45.0,
    j2_max_step=45.0, lambert_debug=False, force_mode=None
):
    """Generate one Lambert training sample with natural-transfer filtering."""
    if cfg is None:
        cfg = OrbitConfig()

    if cfg.axisym_init_guess:
        sampled = sample_axisym_transfer_state(rng, cfg)
        if sampled is None:
            return None
        r0, v0_orbit, r_target, v_target_orbit, oe_dep, oe_arr = sampled
    else:
        # Sample departure orbit
        r0, v0_orbit, oe_dep = sample_orbit(rng, cfg)

        # Sample arrival orbit
        if cfg.nearby:
            r_target, v_target_orbit, oe_arr = sample_nearby_orbit(rng, oe_dep, cfg)
        else:
            r_target, v_target_orbit, oe_arr = sample_orbit(rng, cfg)

    # Check transfer angle — reject near-degenerate geometry
    h_hat = _safe_unit(np.cross(r0, v0_orbit))
    theta = compute_transfer_angle(r0, r_target, h_hat=h_hat)
    if theta < cfg.min_transfer_angle or theta > cfg.max_transfer_angle:
        return None

    # Transfer time: fraction of departure orbit period
    period = orbital_period(oe_dep['a'])
    tof = rng.uniform(cfg.tof_periods_min * period, cfg.tof_periods_max * period)

    # Select Nrev directly from TOF/period using the requested rule.
    max_feasible_nrev = int(np.floor(max(tof / period - 1e-9, 0.0)))
    min_requested_nrev = max(0, int(cfg.nrev_min))
    max_requested_nrev = max(min_requested_nrev, int(cfg.nrev_max))

    if force_mode == "single":
        target_nrev = 0
    elif force_mode == "multi":
        target_nrev = max(1, int(np.round(tof / period)))
    elif not cfg.enable_multi_rev:
        target_nrev = 0
    else:
        target_nrev = int(np.round(tof / period))

    target_nrev = max(target_nrev, min_requested_nrev)
    target_nrev = min(target_nrev, max_requested_nrev, max_feasible_nrev)
    if cfg.require_multi_rev and target_nrev == 0:
        return []

    # Solve Lambert and keep all successful branch combinations for selected Nrev.
    candidates = []
    last_err = None

    def _collect_solutions(nrev):
        nonlocal last_err
        ncases = (0,) if nrev == 0 else (0, 1)
        for prograde in (True, False):
            for ncase in ncases:
                try:
                    v1_lambert, v2_lambert = solve_lambert(
                        r0, r_target, tof, prograde=prograde, Nrev=nrev, Ncase=ncase
                    )
                    candidates.append((v1_lambert, v2_lambert, nrev, ncase, prograde))
                except RuntimeError as e:
                    last_err = e

    _collect_solutions(target_nrev)

    if not candidates:
        if lambert_debug and last_err is not None:
            print(f"[Lambert fail] Nrev={target_nrev} err={last_err}")
        return []

    # Precompute best available single-rev total dV for optional comparison filter.
    best_single_total = None
    if cfg.require_multi_rev_cheaper:
        for prograde in (True, False):
            try:
                v1_0, v2_0 = solve_lambert(r0, r_target, tof, prograde=prograde, Nrev=0, Ncase=0)
                tdv0 = np.linalg.norm(v1_0 - v0_orbit) + np.linalg.norm(v_target_orbit - v2_0)
                best_single_total = tdv0 if best_single_total is None else min(best_single_total, tdv0)
            except RuntimeError:
                continue

    accepted = []
    for v1_lambert, v2_lambert, nrev_used, ncase_used, prograde_used in candidates:
        dv1 = v1_lambert - v0_orbit
        dv2 = v_target_orbit - v2_lambert
        total_dv = np.linalg.norm(dv1) + np.linalg.norm(dv2)

        if cfg.require_multi_rev_cheaper and nrev_used > 0 and best_single_total is not None:
            if not (total_dv + 1e-9 < best_single_total):
                continue

        # Filter on TOTAL dV (natural transfer criterion)
        if total_dv > cfg.dv_max:
            continue

        # Check transfer orbit perigee
        if not check_transfer_orbit(r0, v1_lambert, cfg):
            continue

        # Two-body propagation error (diagnostic)
        if verify:
            r_tb, v_tb, _ = propagate_twobody(r0, v0_orbit + dv1, tof, max_step=verify_max_step)
            pos_err_tb = np.linalg.norm(r_tb - r_target)
            vel_err_tb = np.linalg.norm(v_tb - v2_lambert)
        else:
            pos_err_tb = np.nan
            vel_err_tb = np.nan

        # J2 propagation error
        r_j2, v_j2, _ = propagate_j2(r0, v0_orbit + dv1, tof, max_step=j2_max_step)
        pos_err_j2 = np.linalg.norm(r_j2 - r_target)
        vel_err_j2_vs_target = np.linalg.norm(v_j2 - v_target_orbit)
        vel_err_j2_vs_lambert = np.linalg.norm(v_j2 - v2_lambert)

        if verify and np.isfinite(pos_err_tb):
            pos_err_j2_vs_tb = np.linalg.norm(r_j2 - r_tb)
            vel_err_j2_vs_tb = np.linalg.norm(v_j2 - v_tb)
        else:
            pos_err_j2_vs_tb = np.nan
            vel_err_j2_vs_tb = np.nan

        accepted.append({
            'r0': r0.astype(np.float32),
            'v0': v0_orbit.astype(np.float32),
            'r_target': r_target.astype(np.float32),
            'v_target': v_target_orbit.astype(np.float32),
            'tof': np.float32(tof),
            'dv1': dv1.astype(np.float32),
            'dv2': dv2.astype(np.float32),
            'dv1_mag': np.float32(np.linalg.norm(dv1)),
            'dv2_mag': np.float32(np.linalg.norm(dv2)),
            'total_dv': np.float32(total_dv),
            'period_dep': np.float32(period),
            'transfer_angle': np.float32(theta),
            'nrev': np.int32(nrev_used),
            'ncase': np.int32(ncase_used),
            'prograde': np.int32(1 if prograde_used else 0),
            'pos_err_tb': np.float32(pos_err_tb) if np.isfinite(pos_err_tb) else np.float32(np.nan),
            'vel_err_tb': np.float32(vel_err_tb) if np.isfinite(vel_err_tb) else np.float32(np.nan),
            'pos_err_j2': np.float32(pos_err_j2),
            'vel_err_j2_vs_target': np.float32(vel_err_j2_vs_target),
            'vel_err_j2_vs_lambert': np.float32(vel_err_j2_vs_lambert),
            'pos_err_j2_vs_tb': np.float32(pos_err_j2_vs_tb) if np.isfinite(pos_err_j2_vs_tb) else np.float32(np.nan),
            'vel_err_j2_vs_tb': np.float32(vel_err_j2_vs_tb) if np.isfinite(vel_err_j2_vs_tb) else np.float32(np.nan),
            'pos_err': np.float32(pos_err_j2),
            'vel_err': np.float32(vel_err_j2_vs_lambert),
        })

    # Keep only the globally best (minimum total dV) candidate among all
    # successful variations (pro/retro and small-a/large-a branches).
    if not accepted:
        return []
    best = min(accepted, key=lambda s: float(s['total_dv']))
    return [best]


def generate_dataset(
    n, seed=42, cfg=None, verbose=True, verify=True, verify_tol=1.0,
    verify_max_step=45.0, j2_max_step=45.0, lambert_debug=False, max_attempt_factor=50,
):
    """Generate n Lambert training samples."""
    if n <= 0:
        raise ValueError("n must be positive")

    rng = np.random.RandomState(seed)
    if cfg is None:
        cfg = OrbitConfig()

    samples = []
    attempts = 0
    max_attempts = n * max_attempt_factor
    target_single = n // 2
    target_multi = n - target_single
    accepted_single = 0
    accepted_multi = 0

    while len(samples) < n and attempts < max_attempts:
        attempts += 1
        force_mode = None
        if cfg.balance_single_multi:
            if accepted_single >= target_single:
                force_mode = "multi"
            elif accepted_multi >= target_multi:
                force_mode = "single"
            else:
                force_mode = "single" if accepted_single <= accepted_multi else "multi"
        sample_set = generate_sample(
            rng, cfg, verify=verify, verify_tol=verify_tol,
            verify_max_step=verify_max_step, j2_max_step=j2_max_step,
            lambert_debug=lambert_debug, force_mode=force_mode,
        )
        if not sample_set:
            continue
        for sample in sample_set:
            if len(samples) >= n:
                break
            if cfg.balance_single_multi:
                is_multi = int(sample['nrev']) > 0
                if is_multi and accepted_multi >= target_multi:
                    continue
                if (not is_multi) and accepted_single >= target_single:
                    continue
            samples.append(sample)
            if int(sample['nrev']) > 0:
                accepted_multi += 1
            else:
                accepted_single += 1
            if verbose and len(samples) % max(1, n // 10) == 0:
                s = samples[-1]
                print(
                    f"  [{len(samples):6d}/{n}] "
                    f"|dV1|={s['dv1_mag']:.3f} km/s  "
                    f"|dV2|={s['dv2_mag']:.3f} km/s  "
                    f"total={s['total_dv']:.3f} km/s  "
                    f"tof={s['tof']/60:.1f} min  "
                    f"θ={s['transfer_angle']*180/np.pi:.0f}°  "
                    f"Nrev={int(s['nrev'])}  "
                    f"J2_err={s['pos_err_j2']:.3f} km  "
                    f"({attempts} att)"
                )

    if len(samples) < n:
        print(f"WARNING: only generated {len(samples)}/{n} samples after {max_attempts} attempts")
        if cfg.balance_single_multi:
            print(
                f"         accepted single={accepted_single}/{target_single}, "
                f"multi={accepted_multi}/{target_multi}"
            )

    success_rate = len(samples) / attempts * 100 if attempts > 0 else 0.0
    if verbose:
        print(f"Success rate: {success_rate:.1f}% ({len(samples)}/{attempts})")

    if len(samples) == 0:
        raise RuntimeError(
            "No valid samples generated. Try: larger --max_attempt_factor, "
            "higher --dv_max, or easier sampling ranges."
        )

    keys = list(samples[0].keys())
    dataset = {k: np.array([s[k] for s in samples]) for k in keys}

    if verbose:
        print(f"\nDataset summary ({len(samples)} samples):")
        print(f"  |dV1|:     {dataset['dv1_mag'].mean():.3f} ± {dataset['dv1_mag'].std():.3f} km/s  "
              f"[{dataset['dv1_mag'].min():.3f}, {dataset['dv1_mag'].max():.3f}]")
        print(f"  |dV2|:     {dataset['dv2_mag'].mean():.3f} ± {dataset['dv2_mag'].std():.3f} km/s  "
              f"[{dataset['dv2_mag'].min():.3f}, {dataset['dv2_mag'].max():.3f}]")
        print(f"  Total dV:  {dataset['total_dv'].mean():.3f} ± {dataset['total_dv'].std():.3f} km/s  "
              f"[{dataset['total_dv'].min():.3f}, {dataset['total_dv'].max():.3f}]")
        print(f"  TOF:       {dataset['tof'].mean()/60:.1f} ± {dataset['tof'].std()/60:.1f} min  "
              f"[{dataset['tof'].min()/60:.1f}, {dataset['tof'].max()/60:.1f}]")
        print(f"  θ:         {np.degrees(dataset['transfer_angle']).mean():.0f} ± "
              f"{np.degrees(dataset['transfer_angle']).std():.0f} deg  "
              f"[{np.degrees(dataset['transfer_angle']).min():.0f}, "
              f"{np.degrees(dataset['transfer_angle']).max():.0f}]")
        if 'nrev' in dataset:
            nrev_vals, nrev_cnt = np.unique(dataset['nrev'], return_counts=True)
            nrev_info = ", ".join([f"Nrev={int(v)}:{int(c)}" for v, c in zip(nrev_vals, nrev_cnt)])
            print(f"  Lambert branches: {nrev_info}")
        print(f"  J2 pos err: {np.nanmean(dataset['pos_err_j2']):.3f} ± "
              f"{np.nanstd(dataset['pos_err_j2']):.3f} km  "
              f"(max {np.nanmax(dataset['pos_err_j2']):.3f})")
        if np.any(np.isfinite(dataset['pos_err_tb'])):
            print(f"  TB pos err: {np.nanmean(dataset['pos_err_tb']):.6f} ± "
                  f"{np.nanstd(dataset['pos_err_tb']):.6f} km")

    return dataset


def augment_dataset_with_shooting_labels(
    dataset,
    max_prop_step=45.0,
    tol=1e-4,
    max_iter=50,
    step=1e-6,
    damping=1.0,
    verbose=True,
):
    """Add J2 shooting-corrected departure-burn labels to an existing dataset.

    Adds:
        dv1_lambert, dv2_lambert
        dv1_corrected, dv_correction, dv_correction_mag
        shooting_converged, shooting_n_iterations
        shooting_pos_err_lambert, shooting_pos_err_final
    """
    N = len(dataset['r0'])

    # Explicit aliases for clarity (keep old keys too)
    dataset['dv1_lambert'] = dataset['dv1'].astype(np.float32).copy()
    dataset['dv2_lambert'] = dataset['dv2'].astype(np.float32).copy()

    dv1_corrected = np.zeros_like(dataset['dv1'], dtype=np.float32)
    dv_correction = np.zeros_like(dataset['dv1'], dtype=np.float32)
    dv_correction_mag = np.zeros(N, dtype=np.float32)

    shooting_converged = np.zeros(N, dtype=bool)
    shooting_n_iterations = np.zeros(N, dtype=np.int32)
    shooting_pos_err_lambert = np.zeros(N, dtype=np.float32)
    shooting_pos_err_final = np.zeros(N, dtype=np.float32)

    t0 = time.time()
    n_conv = 0

    for i in range(N):
        res = shooting_correct(
            dataset['r0'][i].astype(float),
            dataset['v0'][i].astype(float),
            dataset['dv1'][i].astype(float),
            dataset['r_target'][i].astype(float),
            float(dataset['tof'][i]),
            max_step=max_prop_step,
            max_iter=max_iter,
            tol=tol,
            step=step,
            damping=damping,
        )

        dv1_corrected[i] = res['dv_corrected'].astype(np.float32)
        dv_correction[i] = res['dv_correction'].astype(np.float32)
        dv_correction_mag[i] = np.float32(np.linalg.norm(res['dv_correction']))

        shooting_converged[i] = bool(res['converged'])
        shooting_n_iterations[i] = int(res['n_iterations'])
        shooting_pos_err_lambert[i] = np.float32(res['pos_err_lambert'])
        shooting_pos_err_final[i] = np.float32(res['pos_err_final'])

        if res['converged']:
            n_conv += 1

        if verbose and (i + 1) % max(1, N // 10) == 0:
            corr_ms = np.linalg.norm(res['dv_correction']) * 1000.0
            elapsed = time.time() - t0
            print(
                f"  [shoot {i+1:6d}/{N}] conv={n_conv}/{i+1}  "
                f"|Δv|={corr_ms:.2f} m/s  "
                f"err {res['pos_err_lambert']:.2f}->{res['pos_err_final']:.5f} km  "
                f"iters={res['n_iterations']}  ({elapsed:.0f}s)"
            )

    dataset['dv1_corrected'] = dv1_corrected
    dataset['dv_correction'] = dv_correction
    dataset['dv_correction_mag'] = dv_correction_mag

    dataset['shooting_converged'] = shooting_converged
    dataset['shooting_n_iterations'] = shooting_n_iterations
    dataset['shooting_pos_err_lambert'] = shooting_pos_err_lambert
    dataset['shooting_pos_err_final'] = shooting_pos_err_final

    # Optional convenience key for training code that expects a generic "target"
    # (leave commented if you prefer explicit selection in the trainer)
    # dataset['dv1_target'] = dataset['dv1_corrected']

    if verbose:
        conv = shooting_converged
        print("\nShooting label summary:")
        print(f"  Converged: {conv.sum()}/{N} ({conv.mean()*100:.1f}%)")
        print(f"  Lambert J2 miss: {shooting_pos_err_lambert.mean():.3f} ± {shooting_pos_err_lambert.std():.3f} km")
        if conv.any():
            print(f"  Final miss (conv): {shooting_pos_err_final[conv].mean():.6f} ± {shooting_pos_err_final[conv].std():.6f} km")
            corr_m = dv_correction_mag[conv] * 1000.0
            print(f"  |Δv| correction (conv): {corr_m.mean():.2f} ± {corr_m.std():.2f} m/s")

    return dataset


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description='Generate Lambert training data (full lambertMR port + natural transfer filtering)')
    parser.add_argument('--quick', action='store_true', default=DEFAULT_QUICK,
                        help='Quick preset for exploratory runs; writes to data_quick/ by default')
    parser.add_argument('--n_train', type=int, default=DEFAULT_N_TRAIN)
    parser.add_argument('--n_test', type=int, default=DEFAULT_N_TEST)
    parser.add_argument('--out_dir', type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--no_verify', action='store_true', default=DEFAULT_NO_VERIFY)
    parser.add_argument('--verify_max_step', type=float, default=DEFAULT_VERIFY_MAX_STEP,
                        help='Maximum RK4 step size in seconds for two-body verification propagation')
    parser.add_argument('--j2_max_step', type=float, default=DEFAULT_J2_MAX_STEP,
                        help='Maximum RK4 step size in seconds for J2 propagation')
    parser.add_argument('--lambert_debug', action='store_true', default=DEFAULT_LAMBERT_DEBUG)
    parser.add_argument('--max_attempt_factor', type=int, default=DEFAULT_MAX_ATTEMPT_FACTOR)
    parser.add_argument('--dv_max', type=float, default=DEFAULT_DV_MAX,
                        help='Max total delta-v (km/s) for natural transfer filtering')
    parser.add_argument('--no_dv_limit', action='store_true', default=DEFAULT_NO_DV_LIMIT,
                        help='Disable total-dV filtering (equivalent to dv_max=inf)')
    parser.add_argument('--inc_min_deg', type=float, default=DEFAULT_INC_MIN_DEG,
                        help='Minimum inclination in degrees')
    parser.add_argument('--inc_max_deg', type=float, default=DEFAULT_INC_MAX_DEG,
                        help='Maximum inclination in degrees')
    parser.add_argument('--tof_min', type=float, default=DEFAULT_TOF_MIN,
                        help='Min TOF as fraction of departure orbit period')
    parser.add_argument('--tof_max', type=float, default=DEFAULT_TOF_MAX,
                        help='Max TOF as fraction of departure orbit period')
    parser.add_argument('--enable_multi_rev', action='store_true', default=DEFAULT_ENABLE_MULTI_REV,
                        help='Allow sampling Lambert multi-revolution branches (Nrev > 0)')
    parser.add_argument('--disable_multi_rev', dest='enable_multi_rev', action='store_false',
                        help='Disable Lambert multi-revolution sampling')
    parser.add_argument('--require_multi_rev', action='store_true', default=DEFAULT_REQUIRE_MULTI_REV,
                        help='Reject samples unless a multi-revolution Lambert solution is used')
    parser.add_argument('--allow_single_rev', dest='require_multi_rev', action='store_false',
                        help='Allow accepted samples to include Nrev=0 solutions')
    parser.add_argument('--multi_rev_prob', type=float, default=DEFAULT_MULTI_REV_PROB,
                        help='Probability of requesting Nrev>0 on each sample when feasible')
    parser.add_argument('--nrev_min', type=int, default=DEFAULT_NREV_MIN,
                        help='Minimum requested Nrev when multi-rev is sampled')
    parser.add_argument('--nrev_max', type=int, default=DEFAULT_NREV_MAX,
                        help='Maximum requested Nrev when multi-rev is sampled')
    parser.add_argument('--require_multi_rev_cheaper', action='store_true', default=DEFAULT_REQUIRE_MULTI_REV_CHEAPER,
                        help='For Nrev>0 samples, keep only if total dV beats best available Nrev=0 solution')
    parser.add_argument('--no_require_multi_rev_cheaper', dest='require_multi_rev_cheaper', action='store_false',
                        help='Disable strict multi-rev-vs-single dV comparison filter')
    parser.add_argument('--balance_single_multi', action='store_true', default=DEFAULT_BALANCE_SINGLE_MULTI,
                        help='Enforce 50/50 accepted split between Nrev=0 and Nrev>0')
    parser.add_argument('--no_balance_single_multi', dest='balance_single_multi', action='store_false',
                        help='Disable forced 50/50 split between single and multi-rev samples')
    parser.add_argument('--axisym_init_guess', action='store_true', default=DEFAULT_AXISYM_INIT_GUESS,
                        help='Use axisymmetry-aware geometry sampler (RAAN canonicalization) for r0/r_target')
    parser.add_argument('--axisym_random_yaw', action='store_true', default=DEFAULT_AXISYM_RANDOM_YAW,
                        help='With --axisym_init_guess, apply random global z-rotation to sampled geometry')
    parser.add_argument('--overwrite', action='store_true', default=DEFAULT_OVERWRITE,
                        help='Allow overwriting existing output .npz files')
    parser.add_argument('--add_shooting_labels', dest='add_shooting_labels', action='store_true',
                        help='Augment dataset with J2 shooting-corrected dv1 labels (default: on)')
    parser.add_argument('--no_shooting_labels', dest='add_shooting_labels', action='store_false',
                        help='Disable shooting-corrected label augmentation')
    parser.set_defaults(add_shooting_labels=DEFAULT_ADD_SHOOTING_LABELS)
    parser.add_argument('--shooting_tol', type=float, default=DEFAULT_SHOOTING_TOL,
                        help='Shooting corrector position tolerance (km)')
    parser.add_argument('--shooting_max_iter', type=int, default=DEFAULT_SHOOTING_MAX_ITER,
                        help='Max Newton iterations for shooting corrector')
    parser.add_argument('--shooting_prop_max_step', type=float, default=DEFAULT_SHOOTING_PROP_MAX_STEP,
                        help='Maximum RK4 step size in seconds inside the shooting corrector')
    parser.add_argument('--shooting_fd_step', type=float, default=DEFAULT_SHOOTING_FD_STEP,
                        help='Finite-difference step for shooting Jacobian (km/s)')
    parser.add_argument('--shooting_damping', type=float, default=DEFAULT_SHOOTING_DAMPING,
                        help='Newton damping factor for shooting corrector')
    
    args = parser.parse_args()

    if args.quick:
        # Quick mode is for inspection/debugging; keep it separate from full datasets.
        if args.n_train == 10000:
            args.n_train = 100
        if args.n_test == 1000:
            args.n_test = 20
        if args.out_dir == 'data':
            args.out_dir = 'data_quick'

    inc_min = np.deg2rad(args.inc_min_deg)
    inc_max = np.deg2rad(args.inc_max_deg)
    if inc_min < 0.0 or inc_max > np.pi or inc_min > inc_max:
        raise ValueError(
            f"Invalid inclination range: [{args.inc_min_deg}, {args.inc_max_deg}] deg. "
            "Expected 0 <= min <= max <= 180."
        )

    dv_cap = float('inf') if args.no_dv_limit else args.dv_max

    cfg = OrbitConfig(
        dv_max=dv_cap,
        inc_min=inc_min,
        inc_max=inc_max,
        tof_periods_min=args.tof_min,
        tof_periods_max=args.tof_max,
        axisym_init_guess=args.axisym_init_guess,
        axisym_random_yaw=args.axisym_random_yaw,
        enable_multi_rev=args.enable_multi_rev,
        require_multi_rev=args.require_multi_rev,
        multi_rev_prob=float(np.clip(args.multi_rev_prob, 0.0, 1.0)),
        nrev_min=max(0, int(args.nrev_min)),
        nrev_max=max(0, int(args.nrev_max)),
        require_multi_rev_cheaper=args.require_multi_rev_cheaper,
        balance_single_multi=args.balance_single_multi,
    )

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    print("=== Generating training data ===")
    print(
        f"Config: dv_max={'inf' if not np.isfinite(cfg.dv_max) else f'{cfg.dv_max:.3f}'} km/s, "
        f"i=[{args.inc_min_deg:.1f}, {args.inc_max_deg:.1f}] deg, "
        f"TOF=[{cfg.tof_periods_min:.1f}, {cfg.tof_periods_max:.1f}] periods"
    )
    print(
        f"Lambert Nrev: enable={cfg.enable_multi_rev}, require={cfg.require_multi_rev}, "
        f"rule=round(TOF/period), range=[{cfg.nrev_min}, {cfg.nrev_max}]"
    )
    print(f"Multi-rev cheaper-than-single filter: {cfg.require_multi_rev_cheaper}")
    print(f"Balanced single/multi split: {cfg.balance_single_multi}")
    if cfg.axisym_init_guess:
        print(f"Sampler: axisymmetry-aware (RAAN=0 canonical), random_yaw={cfg.axisym_random_yaw}")
    else:
        print("Sampler: orbital-elements random sampler")
    train = generate_dataset(
        args.n_train, seed=args.seed, cfg=cfg,
        verify=not args.no_verify, verify_max_step=args.verify_max_step,
        j2_max_step=args.j2_max_step, lambert_debug=args.lambert_debug,
        max_attempt_factor=args.max_attempt_factor,
    )

    if args.add_shooting_labels:
        print("\n=== Adding shooting-corrected labels to training set ===")
        train = augment_dataset_with_shooting_labels(
            train,
            max_prop_step=args.shooting_prop_max_step,
            tol=args.shooting_tol,
            max_iter=args.shooting_max_iter,
            step=args.shooting_fd_step,
            damping=args.shooting_damping,
            verbose=True,
        )

    train_path = out / 'lambert_train.npz'
    if train_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{train_path} already exists. Use --overwrite, or pick a different --out_dir."
        )
    np.savez(train_path, **train)
    print(f"Saved to {train_path}\n")
    
    print("=== Generating test data ===")
    test = generate_dataset(
        args.n_test, seed=args.seed + 1000, cfg=cfg,
        verify=not args.no_verify, verify_max_step=args.verify_max_step,
        j2_max_step=args.j2_max_step, lambert_debug=args.lambert_debug,
        max_attempt_factor=args.max_attempt_factor,
    )

    if args.add_shooting_labels:
        print("\n=== Adding shooting-corrected labels to test set ===")
        test = augment_dataset_with_shooting_labels(
            test,
            max_prop_step=args.shooting_prop_max_step,
            tol=args.shooting_tol,
            max_iter=args.shooting_max_iter,
            step=args.shooting_fd_step,
            damping=args.shooting_damping,
            verbose=True,
        )

    test_path = out / 'lambert_test.npz'
    if test_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{test_path} already exists. Use --overwrite, or pick a different --out_dir."
        )
    np.savez(test_path, **test)
    print(f"Saved to {test_path}\n")

    # ── Visualization ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(16, 10))

        # 3D trajectories
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        n_show = min(10, len(train['r0']))
        for i in range(n_show):
            _, _, traj = propagate_twobody(
                train['r0'][i], train['v0'][i] + train['dv1'][i],
                float(train['tof'][i]), max_step=45.0)
            ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.6, lw=0.8)
            ax1.scatter(*train['r0'][i], c='green', s=15, zorder=5)
            ax1.scatter(*train['r_target'][i], c='red', s=15, zorder=5)
        ax1.set(xlabel='X', ylabel='Y', zlabel='Z', title='Transfer orbits (2-body)')

        # dV distribution
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.hist(train['dv1_mag'], bins=40, alpha=0.6, label='|dV₁|')
        ax2.hist(train['dv2_mag'], bins=40, alpha=0.6, label='|dV₂|')
        if np.isfinite(cfg.dv_max):
            ax2.axvline(cfg.dv_max, c='red', ls='--', label=f'cap={cfg.dv_max}')
        ax2.set(xlabel='dV (km/s)', ylabel='Count', title='dV Distribution')
        ax2.legend(fontsize=8)

        # TOF distribution
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.hist(train['tof'] / 60.0, bins=40, alpha=0.7)
        ax3.set(xlabel='TOF (min)', ylabel='Count', title='Transfer Time')

        # Transfer angle distribution
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.hist(np.degrees(train['transfer_angle']), bins=40, alpha=0.7, color='purple')
        ax4.set(xlabel='Transfer angle (deg)', ylabel='Count', title='Transfer Geometry')

        # Total dV vs TOF
        ax5 = fig.add_subplot(2, 3, 5)
        sc = ax5.scatter(train['tof'] / 60, train['total_dv'],
                         c=np.degrees(train['transfer_angle']),
                         cmap='viridis', s=5, alpha=0.5)
        plt.colorbar(sc, ax=ax5, label='θ (deg)')
        ax5.set(xlabel='TOF (min)', ylabel='Total dV (km/s)', title='dV vs TOF')

        # J2 error vs TOF
        ax6 = fig.add_subplot(2, 3, 6)
        j2e = np.maximum(train['pos_err_j2'], 1e-12)
        sc2 = ax6.scatter(train['tof'] / 60, j2e, c=train['total_dv'],
                          cmap='viridis', s=5, alpha=0.5)
        plt.colorbar(sc2, ax=ax6, label='Total dV (km/s)')
        ax6.set_yscale('log')
        ax6.set(xlabel='TOF (min)', ylabel='J2 pos error (km)', title='J2 Mismatch')

        plt.tight_layout()
        plt.savefig(out / 'lambert_data_summary.png', dpi=150)
        print(f"Plot saved to {out / 'lambert_data_summary.png'}")

        # Nrev-separated dV-vs-TOF visualization
        if 'nrev' in train:
            nrev_vals = np.unique(train['nrev'].astype(int))
            n_pan = max(1, len(nrev_vals))
            n_cols = min(3, n_pan)
            n_rows = int(np.ceil(n_pan / n_cols))
            fig2, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows), squeeze=False)
            tof_min = np.min(train['tof'] / 60.0)
            tof_max = np.max(train['tof'] / 60.0)
            dv_min = np.min(train['total_dv'])
            dv_max = np.max(train['total_dv'])

            for i, nrev in enumerate(nrev_vals):
                r = i // n_cols
                c = i % n_cols
                ax = axes[r][c]
                m = train['nrev'].astype(int) == int(nrev)
                scn = ax.scatter(
                    train['tof'][m] / 60.0,
                    train['total_dv'][m],
                    c=np.degrees(train['transfer_angle'][m]),
                    cmap='viridis',
                    s=7,
                    alpha=0.55,
                )
                ax.set_title(f'Nrev = {int(nrev)}  (N={int(m.sum())})')
                ax.set_xlabel('TOF (min)')
                ax.set_ylabel('Total dV (km/s)')
                ax.set_xlim(tof_min, tof_max)
                ax.set_ylim(dv_min, dv_max)
                ax.grid(alpha=0.25)
                cbar = fig2.colorbar(scn, ax=ax)
                cbar.set_label('θ (deg)')

            for j in range(len(nrev_vals), n_rows * n_cols):
                r = j // n_cols
                c = j % n_cols
                axes[r][c].axis('off')

            fig2.tight_layout()
            plt.savefig(out / 'lambert_data_summary_by_nrev.png', dpi=150)
            print(f"Plot saved to {out / 'lambert_data_summary_by_nrev.png'}")

    except ImportError:
        print("matplotlib not available, skipping plots")
