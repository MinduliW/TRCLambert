"""
Lambert problem tester (beginner-friendly version).

WHAT IS LAMBERT'S PROBLEM?
--------------------------
Given two position vectors r1 and r2 (where the spacecraft is now and where
we want it to be) and a time of flight (tof), find the velocity vectors v1
and v2 at those two points such that an unpowered (Keplerian) orbit connects
them in exactly tof seconds.

We use the Izzo 2015 algorithm from the `lamberthub` package to solve it.

CANONICAL UNITS (Re = 1, mu = 1)
--------------------------------
Instead of working in kilometers and seconds, we rescale so Earth's radius
is 1 distance unit (DU) and Earth's gravitational parameter mu is 1. This
keeps the numbers near 1, which is numerically nicer. The conversions are:

    1 DU = R_earth          = 6378.137 km
    1 TU = sqrt(DU^3 / mu)  ~ 806.811 s   (time unit)
    1 VU = DU / TU          ~ 7.9054 km/s (velocity unit)

To install the dependencies:  pip install lamberthub numpy scipy
"""

import numpy as np
from lamberthub import izzo2015
from scipy.integrate import solve_ivp

from constants import MU_EARTH, R_EARTH


# ---------------------------------------------------------------------------
# Derived canonical-unit conversions (1 DU = R_EARTH, mu = 1).
# ---------------------------------------------------------------------------
TU_SECONDS  = np.sqrt(R_EARTH**3 / MU_EARTH)   # one time unit, in seconds
VU_KM_PER_S = R_EARTH / TU_SECONDS              # one velocity unit, km/s


# ---------------------------------------------------------------------------
# Two-body propagator: given a starting position and velocity, integrate the
# orbit forward in time. We use this only to *check* the Lambert solver's
# answer by re-propagating (r1, v1) and seeing whether we land at r2.
# ---------------------------------------------------------------------------
def propagate_two_body(r0, v0, tof, mu=1.0):
    """Integrate the two-body problem from (r0, v0) for time `tof`.

    Returns the final (position, velocity) pair.
    """

    # The state vector y has 6 numbers: [rx, ry, rz, vx, vy, vz].
    # `equations_of_motion` returns dy/dt: how each of those numbers changes
    # with time. The position derivatives are just the velocities, and the
    # velocity derivatives come from Newton's law of gravity: a = -mu * r / |r|^3.
    def equations_of_motion(_t, y):
        position = y[:3]
        velocity = y[3:]
        distance = np.linalg.norm(position)
        acceleration = -mu * position / distance**3
        return np.concatenate([velocity, acceleration])

    initial_state = np.concatenate([r0, v0])
    solution = solve_ivp(
        equations_of_motion,
        t_span=[0.0, tof],
        y0=initial_state,
        method="DOP853",     # high-accuracy Runge-Kutta
        rtol=1e-12,
        atol=1e-12,
    )

    final_state = solution.y[:, -1]      # last column = state at t = tof
    final_position = final_state[:3]
    final_velocity = final_state[3:]
    return final_position, final_velocity


# ---------------------------------------------------------------------------
# Fixed-step RK4 propagator (NumPy version of the J2 propagator pattern in
# LambertPy/lambert_trc_j2.py, but with two-body dynamics only).
#
# Why also have this one?
#   - SciPy's solve_ivp uses adaptive step sizes — great for accuracy, but the
#     step count varies per problem.
#   - A *fixed-step* RK4 is what the training pipeline (J2Propagator) uses,
#     because it has to be deterministic and differentiable under PyTorch.
#   - Re-implementing the same step-count rule here is a useful sanity check.
#
# Step-count rule (same as J2Propagator):
#       n_steps = max(min_steps, ceil(tof / max_step))
#       dt      = tof / n_steps
# ---------------------------------------------------------------------------
def propagate_two_body_rk4(
    r0, v0, tof, mu=1.0, max_step=30.0 / TU_SECONDS, min_steps=50
):
    """Fixed-step Runge-Kutta 4 propagator for the two-body problem.

    Args:
        r0, v0   : length-3 arrays — initial position and velocity (canonical).
        tof      : time of flight (canonical TU).
        mu       : gravitational parameter (1.0 in canonical units).
        max_step : largest allowed time step (canonical TU). Default is 30 s
                   converted to TU (~0.0372 TU) — finer than the J2
                   propagator's 45 s for tighter agreement.
        min_steps: floor on the number of steps, so short transfers still
                   get enough resolution.

    Returns:
        (final_position, final_velocity), each a length-3 numpy array.
    """

    # Right-hand side: state = [rx, ry, rz, vx, vy, vz], returns ds/dt.
    # Position derivative is just the velocity; velocity derivative is
    # gravitational acceleration: a = -mu * r / |r|^3.
    def rhs(state):
        position = state[:3]
        velocity = state[3:]
        distance = np.linalg.norm(position)
        acceleration = -mu * position / distance**3
        return np.concatenate([velocity, acceleration])

    # Pick the number of integration steps and the per-step dt.
    n_steps = max(min_steps, int(np.ceil(tof / max_step)))
    dt = tof / n_steps

    # Standard 4-stage Runge-Kutta loop.
    state = np.concatenate([r0, v0])
    for _ in range(n_steps):
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    final_position = state[:3]
    final_velocity = state[3:]
    return final_position, final_velocity


# ---------------------------------------------------------------------------
# Solve one Lambert problem and print a tidy report. All inputs/outputs are
# in canonical units (mu = 1).
# ---------------------------------------------------------------------------
def solve_and_report(
    r1,
    r2,
    tof,
    revolutions=0,
    prograde=True,
    low_path=True,
    label="",
):
    """Run Izzo's Lambert solver and verify the result by propagation."""

    # Make sure inputs are numpy float arrays (the solver expects this).
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)

    # --- Step 1: ask the solver for v1 and v2. ---
    v1, v2 = izzo2015(
        mu=1.0,
        r1=r1,
        r2=r2,
        tof=tof,
        M=revolutions,        # 0 = direct transfer; 1+ = multi-revolution
        prograde=prograde,    # True = same direction as Earth's spin
        low_path=low_path,    # picks between the two M>=1 solutions
    )

    # --- Step 2: independently check the answer. ---
    # If (r1, v1) really is on an orbit that reaches r2 in time `tof`, then
    # propagating (r1, v1) forward by `tof` should land at r2 with velocity v2.
    # We do the check *twice*: once with SciPy's adaptive DOP853 (very accurate
    # reference) and once with our fixed-step RK4 (matches the training-time
    # propagator). Both should land within numerical noise of r2.
    r2_dop853, v2_dop853 = propagate_two_body(r1, v1, tof)
    r2_rk4,    v2_rk4    = propagate_two_body_rk4(r1, v1, tof)

    err_r_dop853 = np.linalg.norm(r2_dop853 - r2)
    err_v_dop853 = np.linalg.norm(v2_dop853 - v2)
    err_r_rk4    = np.linalg.norm(r2_rk4    - r2)
    err_v_rk4    = np.linalg.norm(v2_rk4    - v2)

    # --- Step 3: print the results. ---
    print(f"--- {label} ---")
    print(f"  r1  = {r1} DU")
    print(f"  r2  = {r2} DU")
    print(
        f"  tof = {tof:.4f} TU,  revs = {revolutions},  "
        f"prograde = {prograde},  low_path = {low_path}"
    )
    print(f"  v1  = {v1} VU   (speed = {np.linalg.norm(v1):.6f})")
    print(f"  v2  = {v2} VU   (speed = {np.linalg.norm(v2):.6f})")
    print(
        f"  DOP853 check: |r_err| = {err_r_dop853:.2e}, "
        f"|v_err| = {err_v_dop853:.2e}"
    )
    print(
        f"  RK4    check: |r_err| = {err_r_rk4:.2e}, "
        f"|v_err| = {err_v_rk4:.2e}"
    )
    print()
    return v1, v2


# ---------------------------------------------------------------------------
# Main script: run a couple of test cases.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"1 TU = {TU_SECONDS:.6f} s,  1 VU = {VU_KM_PER_S:.6f} km/s\n")

    # =====================================================================
    # Test 1: Curtis "Orbital Mechanics for Engineering Students", Ex. 5.2.
    # The textbook gives the expected v1 and v2 in km/s; we convert to
    # canonical units, solve, then convert back to compare.
    # =====================================================================
    r1_km = np.array([5000.0, 10000.0, 2100.0])
    r2_km = np.array([-14600.0, 2500.0, 7000.0])
    tof_seconds = 3600.0

    # Convert km -> DU and s -> TU.
    r1_canonical = r1_km / R_EARTH
    r2_canonical = r2_km / R_EARTH
    tof_canonical = tof_seconds / TU_SECONDS

    v1_pro, v2_pro = solve_and_report(
        r1=r1_canonical,
        r2=r2_canonical,
        tof=tof_canonical,
        prograde=True,
        label="Curtis Example 5.2 (prograde)",
    )

    # Convert the velocity solution back to km/s and compare to the textbook.
    print("  (km/s)  v1 =", v1_pro * VU_KM_PER_S)
    print("  (km/s)  v2 =", v2_pro * VU_KM_PER_S)
    print("  textbook  v1 ~ [-5.9925,  1.9254,  3.2456]")
    print("  textbook  v2 ~ [-3.3125, -4.1966, -0.3853]\n")

    solve_and_report(
        r1=r1_canonical,
        r2=r2_canonical,
        tof=tof_canonical,
        prograde=False,
        label="Curtis Example 5.2 (retrograde)",
    )

    # =====================================================================
    # Test 2: Multi-revolution case.
    # We pick two points on a circle of radius 1.3 DU, 120 degrees apart.
    # A circular orbit at that radius has period T = 2*pi*r^(3/2). For a
    # multi-rev (M=1) solution to exist, the time of flight must be longer
    # than that period. Here we use 1.5 * T.
    # The solver returns *two* M=1 solutions; we get them with low_path
    # =True and =False.
    # =====================================================================
    radius = 1.3
    angle_radians = np.deg2rad(120.0)

    r1_multirev = np.array([radius, 0.0, 0.0])
    r2_multirev = radius * np.array(
        [np.cos(angle_radians), np.sin(angle_radians), 0.0]
    )

    circular_period = 2 * np.pi * radius**1.5
    tof_multirev = 1.5 * circular_period

    solve_and_report(
        r1=r1_multirev,
        r2=r2_multirev,
        tof=tof_multirev,
        revolutions=1,
        low_path=True,
        label="Multi-revolution (M=1, low path)",
    )
    solve_and_report(
        r1=r1_multirev,
        r2=r2_multirev,
        tof=tof_multirev,
        revolutions=1,
        low_path=False,
        label="Multi-revolution (M=1, high path)",
    )
