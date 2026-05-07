"""
Shooting-correction utilities for J2-corrected Lambert transfers.
"""

import numpy as np

from dynamics import propagate_j2


def shooting_correct(
    r0, v0, dv_lambert, r_target, tof,
    n_prop_steps=None,
    max_step=45.0,
    max_iter=50,
    tol=1e-4,        # km position tolerance
    step=1e-6,       # km/s finite-difference step
    damping=1.0,     # Newton step damping
):
    """Single-shooting correction of Lambert dV under J2 dynamics."""
    dv = dv_lambert.copy().astype(float)
    history = []

    # Lambert error under J2 (baseline)
    r_j2, _, _ = propagate_j2(r0, v0 + dv_lambert, tof, n_steps=n_prop_steps, max_step=max_step)
    pos_err_lambert = np.linalg.norm(r_j2 - r_target)

    for iteration in range(max_iter):
        # Forward propagation
        r_final, _, _ = propagate_j2(r0, v0 + dv, tof, n_steps=n_prop_steps, max_step=max_step)
        miss = r_final - r_target
        err = np.linalg.norm(miss)
        history.append((dv.copy(), err))

        if err < tol:
            return {
                'converged': True,
                'dv_corrected': dv.copy(),
                'dv_correction': dv - dv_lambert,
                'pos_err_final': err,
                'pos_err_lambert': pos_err_lambert,
                'n_iterations': iteration + 1,
                'history': history,
            }

        # Finite-difference Jacobian: ∂r_final/∂dV (3×3)
        J = np.zeros((3, 3))
        for j in range(3):
            dv_plus = dv.copy();  dv_plus[j] += step
            dv_minus = dv.copy(); dv_minus[j] -= step

            r_plus, _, _ = propagate_j2(r0, v0 + dv_plus, tof, n_steps=n_prop_steps, max_step=max_step)
            r_minus, _, _ = propagate_j2(r0, v0 + dv_minus, tof, n_steps=n_prop_steps, max_step=max_step)

            J[:, j] = (r_plus - r_minus) / (2.0 * step)

        # Newton update
        try:
            correction = np.linalg.solve(J, miss)
        except np.linalg.LinAlgError:
            correction = np.linalg.lstsq(J, miss, rcond=None)[0]

        dv = dv - damping * correction

    # Didn't converge
    r_final, _, _ = propagate_j2(r0, v0 + dv, tof, n_steps=n_prop_steps, max_step=max_step)
    err = np.linalg.norm(r_final - r_target)

    return {
        'converged': False,
        'dv_corrected': dv.copy(),
        'dv_correction': dv - dv_lambert,
        'pos_err_final': err,
        'pos_err_lambert': pos_err_lambert,
        'n_iterations': max_iter,
        'history': history,
    }
