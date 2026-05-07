"""
constants.py — single source of truth for physical constants and integrator
settings used across the data-generation pipeline.

Edit values here rather than redefining them inline in other files.
"""

# ---------------------------------------------------------------------------
# Earth
# ---------------------------------------------------------------------------
MU_EARTH = 398600.4418        # gravitational parameter, km^3 / s^2
R_EARTH  = 6378.137           # equatorial radius, km
J2_EARTH = 1.082626683e-3     # second zonal harmonic, dimensionless


# ---------------------------------------------------------------------------
# Jupiter
# ---------------------------------------------------------------------------
MU_JUPITER = 1.26686534e8     # gravitational parameter, km^3 / s^2
R_JUPITER  = 71492.0          # equatorial radius, km
J2_JUPITER = 1.4736e-2        # second zonal harmonic, dimensionless


# ---------------------------------------------------------------------------
# Fixed-step RK4 integrator
# ---------------------------------------------------------------------------
# Number of RK4 substeps per orbital period. Per-sample step size is
# dt = orbital_period / STEPS_PER_PERIOD, so the resolution scales naturally
# with the orbit (Earth LEO -> ~25-30 s, Jupiter -> hours).
STEPS_PER_PERIOD = 200


# ---------------------------------------------------------------------------
# Lambert geometric-degeneracy guard
# ---------------------------------------------------------------------------
# The Keplerian Lambert problem becomes geometrically degenerate when r1 and
# r2 are nearly collinear (transfer angle near 0° or 180°): the orbit plane
# is under-determined and the solver returns essentially random velocities.
# Reject samples whose |cos(angle between r1, r2)| is above this threshold.
# 0.9962 corresponds to ~5° from 0° or 180°.
COLLINEAR_REJECT_COS = 0.9962
