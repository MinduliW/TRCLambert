# does RK4 integration for two-body and J2 dynamics

import numpy as np

from constants import MU_EARTH, R_EARTH, J2


DEFAULT_MAX_STEP_S = 45.0


def resolve_step_count(tof, n_steps=None, max_step=DEFAULT_MAX_STEP_S):
    """Return an RK4 step count with optional max-step control."""
    if n_steps is not None:
        return max(int(n_steps), 1)
    tof = max(float(tof), 0.0)
    max_step = float(max_step)
    if max_step <= 0.0:
        raise ValueError("max_step must be positive")
    return max(int(np.ceil(tof / max_step)), 1)


def twobody_rhs(state, mu=MU_EARTH):
    """Two-body equations of motion."""
    r = state[:3]
    v = state[3:6]
    r_mag = np.linalg.norm(r)
    a = -mu * r / r_mag**3
    return np.concatenate([v, a])


def propagate_twobody(r0, v0, tof, n_steps=None, max_step=DEFAULT_MAX_STEP_S, mu=MU_EARTH):
    """Propagate two-body dynamics using RK4."""
    n_steps = resolve_step_count(tof, n_steps=n_steps, max_step=max_step)
    dt = tof / n_steps
    state = np.concatenate([r0, v0]).astype(float)
    trajectory = [state.copy()]

    for _ in range(n_steps):
        k1 = twobody_rhs(state, mu)
        k2 = twobody_rhs(state + 0.5 * dt * k1, mu)
        k3 = twobody_rhs(state + 0.5 * dt * k2, mu)
        k4 = twobody_rhs(state + dt * k3, mu)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        trajectory.append(state.copy())

    trajectory = np.array(trajectory)
    return state[:3], state[3:6], trajectory


def j2_rhs(state, mu=MU_EARTH, r_earth=R_EARTH, j2=J2):
    """Two-body + J2 equations of motion in ECI."""
    r = state[:3]
    v = state[3:6]

    x, y, z = r
    r2 = float(np.dot(r, r))
    rmag = np.sqrt(r2)

    a_grav = -mu * r / (rmag**3)

    z2 = z * z
    r5 = rmag**5
    factor = 1.5 * j2 * mu * (r_earth**2) / r5
    common = 5.0 * z2 / r2
    a_j2 = factor * np.array([
        x * (common - 1.0),
        y * (common - 1.0),
        z * (common - 3.0),
    ])

    return np.concatenate([v, a_grav + a_j2])


def propagate_j2(r0, v0, tof, n_steps=None, max_step=DEFAULT_MAX_STEP_S,
                 mu=MU_EARTH, r_earth=R_EARTH, j2=J2):
    """Propagate two-body + J2 dynamics using RK4."""
    n_steps = resolve_step_count(tof, n_steps=n_steps, max_step=max_step)
    dt = tof / n_steps
    state = np.concatenate([r0, v0]).astype(float)
    trajectory = [state.copy()]

    for _ in range(n_steps):
        k1 = j2_rhs(state, mu=mu, r_earth=r_earth, j2=j2)
        k2 = j2_rhs(state + 0.5 * dt * k1, mu=mu, r_earth=r_earth, j2=j2)
        k3 = j2_rhs(state + 0.5 * dt * k2, mu=mu, r_earth=r_earth, j2=j2)
        k4 = j2_rhs(state + dt * k3, mu=mu, r_earth=r_earth, j2=j2)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        trajectory.append(state.copy())

    trajectory = np.array(trajectory)
    return state[:3], state[3:6], trajectory
