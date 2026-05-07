"""
Axisymmetry-aware transfer-geometry sampler.

This module builds initial position-pair guesses using the fact that a global
rotation about the inertial z-axis (RAAN shift) is a symmetry for axisymmetric
Earth models. We therefore canonicalize with RAAN = 0 and sample geometry in
that reduced space.

Sampling flow:
1) Set RAAN = 0 and sample inclination i to define the transfer plane.
2) On that plane, sample two in-plane angles phi_1, phi_2.
3) Sample one radius for each endpoint and construct r1, r2.
4) (Optional) Apply a shared random z-rotation to recover full ECI variety.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from constants import R_EARTH


@dataclass
class AxisymGuessConfig:
    """Configuration for axisymmetry-aware position sampling."""

    alt_min_km: float = 300.0
    alt_max_km: float = 600.0
    inc_min_rad: float = 0.0
    inc_max_rad: float = np.pi
    transfer_angle_min_rad: float = np.pi / 6
    transfer_angle_max_rad: float = 5 * np.pi / 3
    apply_random_yaw: bool = False


def _rotation_z(yaw_rad: float) -> np.ndarray:
    c = np.cos(yaw_rad)
    s = np.sin(yaw_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _plane_basis_raan_zero(inc_rad: float) -> np.ndarray:
    """
    Return 3x2 matrix [p, q] for the canonical plane with RAAN = 0.

    p is aligned with +x (line of nodes), and q is +y rotated by +inc about +x.
    Any in-plane direction is cos(phi)*p + sin(phi)*q.
    """
    p = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    q = np.array([0.0, np.cos(inc_rad), np.sin(inc_rad)], dtype=np.float64)
    return np.column_stack((p, q))


def _sample_radius_km(rng: np.random.RandomState, cfg: AxisymGuessConfig) -> float:
    alt_km = rng.uniform(cfg.alt_min_km, cfg.alt_max_km)
    return float(R_EARTH + alt_km)


def sample_axisym_position_pair(
    rng: np.random.RandomState, cfg: Optional[AxisymGuessConfig] = None
) -> Dict[str, np.ndarray]:
    """
    Sample one axisymmetry-aware position-pair guess.

    Returns:
        Dictionary with:
        - r1, r2: endpoint position vectors in ECI (km), shape (3,)
        - i, raan, phi_1, phi_2, transfer_angle, r1_mag, r2_mag, yaw: scalars
    """
    if cfg is None:
        cfg = AxisymGuessConfig()

    inc = float(rng.uniform(cfg.inc_min_rad, cfg.inc_max_rad))
    basis = _plane_basis_raan_zero(inc)

    phi_1 = float(rng.uniform(0.0, 2.0 * np.pi))
    transfer_angle = float(
        rng.uniform(cfg.transfer_angle_min_rad, cfg.transfer_angle_max_rad)
    )
    direction = 1.0 if rng.rand() < 0.5 else -1.0
    phi_2 = float((phi_1 + direction * transfer_angle) % (2.0 * np.pi))

    r1_mag = _sample_radius_km(rng, cfg)
    r2_mag = _sample_radius_km(rng, cfg)

    u1 = np.cos(phi_1) * basis[:, 0] + np.sin(phi_1) * basis[:, 1]
    u2 = np.cos(phi_2) * basis[:, 0] + np.sin(phi_2) * basis[:, 1]
    r1 = r1_mag * u1
    r2 = r2_mag * u2

    yaw = 0.0
    if cfg.apply_random_yaw:
        yaw = float(rng.uniform(0.0, 2.0 * np.pi))
        rz = _rotation_z(yaw)
        r1 = rz @ r1
        r2 = rz @ r2

    return {
        "r1": r1.astype(np.float64),
        "r2": r2.astype(np.float64),
        "i": np.float64(inc),
        "raan": np.float64(0.0),
        "phi_1": np.float64(phi_1),
        "phi_2": np.float64(phi_2),
        "transfer_angle": np.float64(transfer_angle),
        "r1_mag": np.float64(r1_mag),
        "r2_mag": np.float64(r2_mag),
        "yaw": np.float64(yaw),
    }


def sample_axisym_batch(
    n: int, seed: int = 42, cfg: Optional[AxisymGuessConfig] = None
) -> Dict[str, np.ndarray]:
    """Sample a batch of axisymmetry-aware endpoint guesses."""
    if n <= 0:
        raise ValueError("n must be > 0")
    if cfg is None:
        cfg = AxisymGuessConfig()

    rng = np.random.RandomState(seed)
    out = {
        "r1": np.zeros((n, 3), dtype=np.float64),
        "r2": np.zeros((n, 3), dtype=np.float64),
        "i": np.zeros(n, dtype=np.float64),
        "raan": np.zeros(n, dtype=np.float64),
        "phi_1": np.zeros(n, dtype=np.float64),
        "phi_2": np.zeros(n, dtype=np.float64),
        "transfer_angle": np.zeros(n, dtype=np.float64),
        "r1_mag": np.zeros(n, dtype=np.float64),
        "r2_mag": np.zeros(n, dtype=np.float64),
        "yaw": np.zeros(n, dtype=np.float64),
    }
    for k in range(n):
        s = sample_axisym_position_pair(rng, cfg)
        for key in out:
            out[key][k] = s[key]
    return out
