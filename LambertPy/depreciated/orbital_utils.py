import numpy as np
from constants import MU_EARTH


def oe_to_eci(a, e, i, raan, aop, nu):
    """Classical orbital elements -> ECI position and velocity.

    Args:
        a:    semi-major axis (km)
        e:    eccentricity
        i:    inclination (rad)
        raan: right ascension of ascending node (rad)
        aop:  argument of periapsis (rad)
        nu:   true anomaly (rad)

    Returns:
        r: (3,) position in ECI (km)
        v: (3,) velocity in ECI (km/s)
    """
    p = a * (1 - e**2)
    r_mag = p / (1 + e * np.cos(nu))

    # Perifocal frame
    r_pqw = r_mag * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pqw = np.sqrt(MU_EARTH / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    # Rotation matrix: perifocal -> ECI
    cos_raan, sin_raan = np.cos(raan), np.sin(raan)
    cos_aop, sin_aop = np.cos(aop), np.sin(aop)
    cos_i, sin_i = np.cos(i), np.sin(i)

    R = np.array([
        [cos_raan * cos_aop - sin_raan * sin_aop * cos_i,
         -cos_raan * sin_aop - sin_raan * cos_aop * cos_i,
         sin_raan * sin_i],
        [sin_raan * cos_aop + cos_raan * sin_aop * cos_i,
         -sin_raan * sin_aop + cos_raan * cos_aop * cos_i,
         -cos_raan * sin_i],
        [sin_aop * sin_i,
         cos_aop * sin_i,
         cos_i]
    ])

    return R @ r_pqw, R @ v_pqw


def orbital_period(a, mu=MU_EARTH):
    """Orbital period in seconds."""
    return 2 * np.pi * np.sqrt(a**3 / mu)