"""
lambert_dataset.py
Generate Lambert-correction training records using forward sampling under
two-body + J2 dynamics (Yang et al. 2022).

PIPELINE (per sample)
---------------------
  1. Draw orbital elements (a, e, i, raan, argp, M0) uniformly from a
     case-specific range.
  2. Convert to the Cartesian state (r1, v1_true).
  3. Draw a time of flight tof uniformly from (0, k * T_orbital).
  4. Propagate (r1, v1_true) for tof using a fixed-step RK4 under
     two-body + J2 dynamics. Track:
         - min |r|  (for collision check)
         - total angular sweep of r  (for revolution count n_rev)
     The result is (r2, v2_true).
  5. Reject if any part of the trajectory dipped below the body radius.
  6. Solve the Keplerian Lambert problem on (r1, r2, tof) with M=n_rev:
         - prograde flag = sign of (r1 x v1_true)_z
         - branch (low / high path) = the one that minimises
           || v1_Lambert - v1_true ||  (only meaningful for M >= 1).
  7. Store the record:
         (r1, r2, tof, n_rev, prograde, branch, v1_Lambert, v1_true).

CASES (Table 1 of the paper)
----------------------------
  leo_single : Earth LEO,  tof in (0,  1) periods
  leo_multi  : Earth LEO,  tof in (0, 10) periods
  jovian     : Jupiter,    tof in (0, 10) periods

Run as a script to demo a small batch of each case.
Install:  pip install lamberthub numpy
"""

import numpy as np
from lamberthub import izzo2015

from constants import (
    MU_EARTH, R_EARTH, J2_EARTH,
    MU_JUPITER, R_JUPITER, J2_JUPITER,
    STEPS_PER_PERIOD,
    COLLINEAR_REJECT_COS,
)


# ===========================================================================
# 2.  ORBITAL ELEMENTS  ->  CARTESIAN STATE
# ===========================================================================
def solve_kepler(mean_anomaly, eccentricity, tol=1e-12, max_iter=50):
    """Solve Kepler's equation  M = E - e sin E  for E (Newton iteration)."""
    M = mean_anomaly % (2.0 * np.pi)
    E = M if eccentricity < 0.8 else np.pi   # standard initial guess
    for _ in range(max_iter):
        f  = E - eccentricity * np.sin(E) - M
        fp = 1.0 - eccentricity * np.cos(E)
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            break
    return E


def kepler_to_cartesian(a, e, inc, raan, argp, M0, mu):
    """Classical orbital elements -> inertial position and velocity."""
    # 1) Solve Kepler's equation, then convert eccentric -> true anomaly.
    E  = solve_kepler(M0, e)
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(E / 2.0),
        np.sqrt(1.0 - e) * np.cos(E / 2.0),
    )

    # 2) Position and velocity in the perifocal (orbit) frame.
    p     = a * (1.0 - e * e)                 # semi-latus rectum
    r_mag = p / (1.0 + e * np.cos(nu))
    r_pf  = np.array([r_mag * np.cos(nu), r_mag * np.sin(nu), 0.0])
    v_pf  = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    # 3) Rotate from perifocal to inertial frame: R3(-Omega) * R1(-i) * R3(-omega).
    cO, sO = np.cos(raan), np.sin(raan)
    co, so = np.cos(argp), np.sin(argp)
    ci, si = np.cos(inc),  np.sin(inc)
    R = np.array([
        [cO * co - sO * so * ci, -cO * so - sO * co * ci,  sO * si],
        [sO * co + cO * so * ci, -sO * so + cO * co * ci, -cO * si],
        [             so * si,                co * si,        ci],
    ])

    return R @ r_pf, R @ v_pf


# ---------------------------------------------------------------------------
# Inverse: (r, v) -> classical orbital elements. Used only to sanity-check
# the forward converter via a round trip.
# ---------------------------------------------------------------------------
def cartesian_to_kepler(r, v, mu):
    """Convert (r, v) to (a, e, inc, raan, argp, M0). Assumes elliptic orbit."""
    r_vec = np.asarray(r, dtype=float)
    v_vec = np.asarray(v, dtype=float)
    r_mag = np.linalg.norm(r_vec)
    v_mag = np.linalg.norm(v_vec)

    h_vec = np.cross(r_vec, v_vec)
    h_mag = np.linalg.norm(h_vec)

    n_vec = np.cross(np.array([0.0, 0.0, 1.0]), h_vec)
    n_mag = np.linalg.norm(n_vec)

    e_vec = (np.cross(v_vec, h_vec) / mu) - (r_vec / r_mag)
    e = np.linalg.norm(e_vec)

    energy = 0.5 * v_mag * v_mag - mu / r_mag
    a = -mu / (2.0 * energy)

    inc = np.arccos(np.clip(h_vec[2] / h_mag, -1.0, 1.0))

    if n_mag > 1e-12:
        raan = np.arccos(np.clip(n_vec[0] / n_mag, -1.0, 1.0))
        if n_vec[1] < 0.0:
            raan = 2.0 * np.pi - raan
    else:
        raan = 0.0

    if e > 1e-12 and n_mag > 1e-12:
        argp = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n_mag * e), -1.0, 1.0))
        if e_vec[2] < 0.0:
            argp = 2.0 * np.pi - argp
    else:
        argp = 0.0

    if e > 1e-12:
        nu = np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r_mag), -1.0, 1.0))
        if np.dot(r_vec, v_vec) < 0.0:
            nu = 2.0 * np.pi - nu
    else:
        nu = 0.0

    E  = 2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(nu / 2.0),
                          np.sqrt(1.0 + e) * np.cos(nu / 2.0))
    M0 = (E - e * np.sin(E)) % (2.0 * np.pi)

    return dict(a=a, e=e, inc=inc, raan=raan, argp=argp, M0=M0)


def _angle_diff(a, b):
    """Smallest absolute difference between two angles, in [0, pi]."""
    d = (a - b) % (2.0 * np.pi)
    return min(d, 2.0 * np.pi - d)


def verify_kepler_to_cartesian(rng=None, n_tests=2000):
    """Round-trip random elements through (r, v) and back; print worst error."""
    if rng is None:
        rng = np.random.default_rng(seed=0)

    worst = dict(a=0.0, e=0.0, inc=0.0, raan=0.0, argp=0.0, M0=0.0)
    for _ in range(n_tests):
        a    = rng.uniform(7_000.0, 50_000.0)
        e    = rng.uniform(0.0, 0.7)
        inc  = rng.uniform(0.01, np.pi - 0.01)        # avoid the equatorial pole
        raan = rng.uniform(0.01, 2.0 * np.pi - 0.01)
        argp = rng.uniform(0.01, 2.0 * np.pi - 0.01)
        M0   = rng.uniform(0.01, 2.0 * np.pi - 0.01)

        r, v = kepler_to_cartesian(a, e, inc, raan, argp, M0, MU_EARTH)
        back = cartesian_to_kepler(r, v, MU_EARTH)

        worst["a"]    = max(worst["a"],    abs(back["a"] - a) / a)
        worst["e"]    = max(worst["e"],    abs(back["e"] - e))
        worst["inc"]  = max(worst["inc"],  _angle_diff(back["inc"],  inc))
        worst["raan"] = max(worst["raan"], _angle_diff(back["raan"], raan))
        worst["argp"] = max(worst["argp"], _angle_diff(back["argp"], argp))
        worst["M0"]   = max(worst["M0"],   _angle_diff(back["M0"],   M0))
    return worst


def verify_curtis_example():
    """Curtis 'Orbital Mechanics' Example 4.3: known COE -> known (r, v).

    Textbook values (rounded): r = [-6045, -3490, 2500] km,
                                v = [-3.457, 6.618, 2.533] km/s
    """
    a    = 8788.0
    e    = 0.1712
    inc  = np.deg2rad(153.249)
    raan = np.deg2rad(255.279)
    argp = np.deg2rad(20.068)
    nu   = np.deg2rad(28.446)

    # True anomaly -> mean anomaly via the eccentric anomaly.
    E  = 2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(nu / 2.0),
                          np.sqrt(1.0 + e) * np.cos(nu / 2.0))
    M0 = E - e * np.sin(E)

    r, v = kepler_to_cartesian(a, e, inc, raan, argp, M0, MU_EARTH)
    r_expected = np.array([-6045.0, -3490.0,  2500.0])
    v_expected = np.array([-3.457,   6.618,   2.533])
    return r, v, r_expected, v_expected


# ===========================================================================
# 3.  J2 RHS  +  FIXED-STEP RK4 PROPAGATOR
# ===========================================================================
def j2_rhs(state, mu, re, j2):
    """Right-hand side dy/dt for two-body + J2 dynamics. state = [r, v]."""
    x, y, z = state[0], state[1], state[2]
    r_mag = np.sqrt(x * x + y * y + z * z)
    r2    = r_mag * r_mag
    r5    = r_mag ** 5

    # Two-body acceleration: a_grav = -mu * r / |r|^3.
    a_grav = -mu * state[:3] / r_mag ** 3

    # J2 acceleration (oblateness perturbation).
    factor = 1.5 * j2 * mu * re * re / r5
    common = 5.0 * z * z / r2
    a_j2 = factor * np.array([
        x * (common - 1.0),
        y * (common - 1.0),
        z * (common - 3.0),
    ])

    return np.concatenate([state[3:], a_grav + a_j2])


def propagate_j2_rk4(r0, v0, tof, mu, re, j2, steps_per_period, period):
    """Propagate (r0, v0) for time `tof` under two-body + J2 with fixed-step RK4.

    The step size is chosen so that one orbital period contains
    `steps_per_period` steps:  dt = period / steps_per_period.

    While integrating we also track:
        - min_radius : smallest |r| along the trajectory  (collision check)
        - n_rev      : number of complete revolutions     (Lambert M)

    n_rev is the floor of (total swept angle) / (2*pi), where the swept
    angle is accumulated by summing the unsigned angle between consecutive
    position vectors. This is exactly what the Keplerian Lambert solver
    expects as `M`.

    Returns (r_final, v_final, n_rev, min_radius).
    """
    dt      = period / steps_per_period
    n_steps = max(1, int(np.ceil(tof / dt)))
    dt      = tof / n_steps                  # adjust so the last step lands on tof

    state      = np.concatenate([r0, v0])
    min_radius = np.linalg.norm(r0)
    swept      = 0.0
    r_prev     = state[:3].copy()

    for _ in range(n_steps):
        k1 = j2_rhs(state,                       mu, re, j2)
        k2 = j2_rhs(state + 0.5 * dt * k1,        mu, re, j2)
        k3 = j2_rhs(state + 0.5 * dt * k2,        mu, re, j2)
        k4 = j2_rhs(state + dt * k3,              mu, re, j2)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        r_curr = state[:3]
        r_mag  = np.linalg.norm(r_curr)
        if r_mag < min_radius:
            min_radius = r_mag

        # Unsigned angle between r_prev and r_curr; clamp to avoid acos NaN.
        cos_d  = np.dot(r_prev, r_curr) / (np.linalg.norm(r_prev) * r_mag)
        cos_d  = max(-1.0, min(1.0, cos_d))
        swept += np.arccos(cos_d)

        r_prev = r_curr.copy()

    n_rev = int(swept / (2.0 * np.pi))
    return state[:3], state[3:], n_rev, min_radius


# ===========================================================================
# 4.  LAMBERT BRANCH SELECTION
# ===========================================================================
def select_lambert_branch(r1, r2, tof, v1_true, mu, n_rev):
    """Run the Keplerian Lambert solver on (r1, r2, tof) with M = n_rev.

    Returns (v1_lambert, prograde, branch) where:
      - prograde is True iff (r1 x v1_true)_z > 0
      - branch is True for low_path, False for high_path
        (irrelevant for M=0; we report True by convention)

    For M >= 1 we try both low/high path and keep the one closest to
    v1_true. If both Lambert calls fail (geometry below the M-rev minimum),
    returns (None, prograde, None).
    """
    h = np.cross(r1, v1_true)
    prograde = bool(h[2] > 0.0)

    candidates = []
    branch_options = (True,) if n_rev == 0 else (True, False)
    for low_path in branch_options:
        try:
            v1, _v2 = izzo2015(
                mu=mu, r1=r1, r2=r2, tof=tof,
                M=n_rev, prograde=prograde, low_path=low_path,
            )
            candidates.append((v1, low_path))
        except Exception:
            continue

    if not candidates:
        return None, prograde, None

    v1_lambert, branch = min(
        candidates,
        key=lambda c: np.linalg.norm(c[0] - v1_true),
    )
    return v1_lambert, prograde, branch


# ===========================================================================
# 5.  CASE-SPECIFIC SAMPLING
# ===========================================================================
CASES = {
    "leo_single": dict(
        body="earth",
        rp_min=R_EARTH + 300.0,  rp_max=R_EARTH + 2000.0,
        ra_max=R_EARTH + 2000.0,
        i_min=0.0,               i_max=np.pi,
        period_factor_max=1.0,
    ),
    "leo_multi": dict(
        body="earth",
        rp_min=R_EARTH + 300.0,  rp_max=R_EARTH + 2000.0,
        ra_max=R_EARTH + 2000.0,
        i_min=0.0,               i_max=np.pi,
        period_factor_max=10.0,
    ),
    "jovian": dict(
        body="jupiter",
        rp_min=5.0  * R_JUPITER, rp_max=30.0 * R_JUPITER,
        ra_max=30.0 * R_JUPITER,
        i_min=0.0,               i_max=1.0,    # radians, per the table
        period_factor_max=10.0,
    ),
}


def _body_constants(body):
    if body == "earth":
        return MU_EARTH, R_EARTH, J2_EARTH
    if body == "jupiter":
        return MU_JUPITER, R_JUPITER, J2_JUPITER
    raise ValueError(f"unknown body: {body}")


def sample_orbital_elements(case_name, rng):
    """Draw one set of orbital elements and a time-of-flight for the case."""
    cfg = CASES[case_name]
    rp  = rng.uniform(cfg["rp_min"], cfg["rp_max"])
    ra  = rng.uniform(rp,            cfg["ra_max"])
    inc = rng.uniform(cfg["i_min"],  cfg["i_max"])
    raan = rng.uniform(0.0, 2.0 * np.pi)
    argp = rng.uniform(0.0, 2.0 * np.pi)
    M0   = rng.uniform(0.0, 2.0 * np.pi)

    a = 0.5 * (rp + ra)
    e = (ra - rp) / (ra + rp)

    mu, _re, _j2 = _body_constants(cfg["body"])
    period = 2.0 * np.pi * np.sqrt(a ** 3 / mu)
    tof    = rng.uniform(0.0, cfg["period_factor_max"]) * period

    return dict(a=a, e=e, inc=inc, raan=raan, argp=argp, M0=M0,
                tof=tof, period=period, body=cfg["body"])


# ===========================================================================
# 6.  END-TO-END SAMPLE GENERATION
# ===========================================================================
def generate_one_sample(case_name, rng, steps_per_period=STEPS_PER_PERIOD):
    """Try to produce one valid record. Returns the dict, or None if rejected."""
    elem = sample_orbital_elements(case_name, rng)
    mu, re, j2 = _body_constants(elem["body"])

    r1, v1_true = kepler_to_cartesian(
        elem["a"], elem["e"], elem["inc"],
        elem["raan"], elem["argp"], elem["M0"], mu,
    )

    # Sanity: starting position must be above the body surface.
    if np.linalg.norm(r1) < re or elem["tof"] <= 0.0:
        return None

    r2, v2_true, n_rev, min_radius = propagate_j2_rk4(
        r1, v1_true, elem["tof"], mu, re, j2,
        steps_per_period=steps_per_period, period=elem["period"],
    )

    # Reject samples where the trajectory clipped the central body.
    if min_radius < re:
        return None

    # Reject near-collinear (r1, r2) configurations: Lambert's orbit plane is
    # geometrically under-determined within ~5 degrees of 0 deg / 180 deg
    # transfer angle, and the solver returns essentially random velocities.
    cos_xfer = float(np.dot(r1, r2)) / (np.linalg.norm(r1) * np.linalg.norm(r2))
    if abs(cos_xfer) > COLLINEAR_REJECT_COS:
        return None

    v1_lambert, prograde, branch = select_lambert_branch(
        r1, r2, elem["tof"], v1_true, mu, n_rev,
    )
    if v1_lambert is None:
        return None

    return dict(
        case=case_name, body=elem["body"],
        r1=r1, r2=r2, tof=elem["tof"],
        n_rev=n_rev, prograde=prograde, branch=branch,
        v1_lambert=v1_lambert, v1_true=v1_true, v2_true=v2_true,
    )


def generate_dataset(case_name, n_samples, rng=None,
                     max_attempts=None, steps_per_period=STEPS_PER_PERIOD):
    """Generate `n_samples` valid records for `case_name`."""
    if rng is None:
        rng = np.random.default_rng()
    if max_attempts is None:
        max_attempts = 10 * n_samples

    records, attempts, rejects = [], 0, 0
    while len(records) < n_samples and attempts < max_attempts:
        attempts += 1
        rec = generate_one_sample(case_name, rng, steps_per_period)
        if rec is None:
            rejects += 1
            continue
        records.append(rec)
    return records, attempts, rejects


# ===========================================================================
# 7.  SAVE / LOAD  (.npz)
# ===========================================================================
def save_records(records, path):
    """Stack a list of record dicts into arrays and save as a .npz file."""
    if not records:
        raise ValueError("no records to save")

    fields_3d = ("r1", "r2", "v1_lambert", "v1_true", "v2_true")
    arrays = {f: np.stack([r[f] for r in records]).astype(np.float64)
              for f in fields_3d}
    arrays["tof"]      = np.array([r["tof"]      for r in records], dtype=np.float64)
    arrays["n_rev"]    = np.array([r["n_rev"]    for r in records], dtype=np.int64)
    arrays["prograde"] = np.array([r["prograde"] for r in records], dtype=bool)
    arrays["branch"]   = np.array([r["branch"]   for r in records], dtype=bool)

    np.savez_compressed(
        path,
        case=records[0]["case"],
        body=records[0]["body"],
        **arrays,
    )


def load_records(path):
    """Load a .npz file written by save_records into a dict of arrays."""
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


# ===========================================================================
# 8.  DEMO MAIN
# ===========================================================================
if __name__ == "__main__":
    # ---- 1.  Verify the Kepler -> Cartesian converter ----
    print("=== converter verification ===")
    r, v, r_exp, v_exp = verify_curtis_example()
    print(f"  Curtis Example 4.3  r = {r}  (expected ~{r_exp})")
    print(f"  Curtis Example 4.3  v = {v}  (expected ~{v_exp})")
    print(f"  |r_err| = {np.linalg.norm(r - r_exp):.3e} km, "
          f"|v_err| = {np.linalg.norm(v - v_exp):.3e} km/s "
          f"(textbook values are rounded to 4 sig figs)")

    worst = verify_kepler_to_cartesian(n_tests=2000)
    print(f"  round-trip on 2000 random orbits, worst errors:")
    print(f"    rel a = {worst['a']:.2e},  abs e = {worst['e']:.2e}")
    print(f"    inc   = {worst['inc']:.2e} rad,  raan = {worst['raan']:.2e} rad")
    print(f"    argp  = {worst['argp']:.2e} rad,  M0   = {worst['M0']:.2e} rad")

    # ---- 2.  Generate demo datasets and save to .npz ----
    rng = np.random.default_rng(seed=42)
    for case in ("leo_single", "leo_multi", "jovian"):
        print(f"\n=== {case} ===")
        records, attempts, rejects = generate_dataset(case, n_samples=20, rng=rng)
        print(f"  generated {len(records)} samples in {attempts} attempts "
              f"({rejects} rejected)")

        diffs = np.array([
            np.linalg.norm(r["v1_lambert"] - r["v1_true"]) for r in records
        ])
        n_revs = np.array([r["n_rev"] for r in records])
        prograde_frac = float(np.mean([r["prograde"] for r in records]))
        print(f"  ||v1_Lambert - v1_true||  km/s : "
              f"min={diffs.min():.3e}, median={np.median(diffs):.3e}, "
              f"max={diffs.max():.3e}")
        print(f"  n_rev distribution            : "
              f"min={n_revs.min()}, max={n_revs.max()}, "
              f"mean={n_revs.mean():.2f}")
        print(f"  prograde fraction             : {prograde_frac:.2f}")

        out_path = f"{case}.npz"
        save_records(records, out_path)
        print(f"  saved -> {out_path}")
