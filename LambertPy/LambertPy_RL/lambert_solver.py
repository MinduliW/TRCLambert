import numpy as np
from constants import MU_EARTH


def _qck(angle):
    """Reduce angle to [0, 2*pi). Equivalent to qck() in lambertMR.m."""
    twopi = 2.0 * np.pi
    diff = twopi * (int(angle / twopi) + min(0, np.sign(angle)))
    return angle - diff


def _h_E(E, y, m, Nrev):
    """Equation h(E) for multi-revolution Lambert (Shen & Tsiotras).

    h(E) = (Nrev*pi + E - sin(E)) / tan(E/2)^3 - 4/m * (y^3 - y^2)

    Returns (h, dh/dE).
    """
    tanE2 = np.tan(E / 2.0)
    if abs(tanE2) < 1e-30:
        tanE2 = 1e-30 * np.sign(tanE2) if tanE2 != 0 else 1e-30

    h = (Nrev * np.pi + E - np.sin(E)) / tanE2**3 - 4.0 / m * (y**3 - y**2)

    # derivative dh/dE
    cosE2 = np.cos(E / 2.0)
    if abs(cosE2) < 1e-30:
        cosE2 = 1e-30
    secE2_sq = 1.0 / cosE2**2
    dh = (
        (1.0 - np.cos(E)) / tanE2**3
        - 1.5 * (Nrev * np.pi + E - np.sin(E)) * secE2_sq / tanE2**4
    )

    return h, dh


def lambertMR(RI, RF, TOF, MU, orbitType=0, Nrev=0, Ncase=0, optionsLMR=0):
    """Full Python port of lambertMR.m (Colombo / D'Souza / Battin / Shen-Tsiotras).

    Solves Lambert's problem for zero-revolution and multi-revolution cases,
    with direct or retrograde transfer option.

    Args:
        RI:        (3,) initial position vector [km]
        RF:        (3,) final position vector [km]
        TOF:       time of flight [s]
        MU:        gravitational parameter [km^3/s^2]
        orbitType: 0 = direct (prograde), 1 = retrograde
        Nrev:      number of revolutions (0 for zero-rev)
        Ncase:     for Nrev>0: 0 = small-a solution, 1 = large-a solution
        optionsLMR: 0 = no display, 1 = convergence warnings, 2 = full warnings

    Returns:
        A, P, E, ERROR, VI, VF, TPAR, THETA
        ERROR codes: 0=ok, 1=no converge, -1=180deg, 2=360deg, 3=Nrev>Nmax, 4=max iter
    """
    nitermax = 2000
    TOL = 1e-14
    TWOPI = 2.0 * np.pi

    RI = np.asarray(RI, dtype=float).ravel()
    RF = np.asarray(RF, dtype=float).ravel()

    # RESET
    A = 0.0
    P = 0.0
    E = 0.0
    VI = np.zeros(3)
    VF = np.zeros(3)

    # Magnitudes and products
    RIM2 = float(np.dot(RI, RI))
    RIM = np.sqrt(RIM2)
    RFM2 = float(np.dot(RF, RF))
    RFM = np.sqrt(RFM2)
    CTH = float(np.dot(RI, RF)) / (RIM * RFM)
    CTH = np.clip(CTH, -1.0, 1.0)
    CR = np.cross(RI, RF)
    STH = float(np.linalg.norm(CR)) / (RIM * RFM)

    # Choose angle for angular momentum sign
    if orbitType == 0:  # direct
        if CR[2] < 0:
            STH = -STH
    elif orbitType == 1:  # retrograde
        if CR[2] > 0:
            STH = -STH
    else:
        raise ValueError(f"{orbitType} is not an allowed orbitType")

    THETA = _qck(np.arctan2(STH, CTH))

    # 360 / 0 degree singularity
    if THETA == TWOPI or THETA == 0.0:
        return 0., 0., 0., 2, np.zeros(3), np.zeros(3), 0., 0.

    B1 = np.sign(STH)
    if STH == 0:
        B1 = 1.0

    # Chord and semi-perimeter
    C_chord = np.sqrt(max(RIM2 + RFM2 - 2.0 * RIM * RFM * CTH, 0.0))
    S = (RIM + RFM + C_chord) / 2.0
    BETA = 2.0 * np.arcsin(np.sqrt(max((S - C_chord) / S, 0.0)))
    PMIN = TWOPI * np.sqrt(S**3 / (8.0 * MU))
    TMIN = PMIN * (np.pi - BETA + np.sin(BETA)) / TWOPI
    LAMBDA = B1 * np.sqrt(max((S - C_chord) / S, 0.0))

    if 4.0 * TOF * LAMBDA == 0.0:
        return 0., 0., 0., -1, np.zeros(3), np.zeros(3), 0., 0.

    # Compute L
    if THETA * 180.0 / np.pi <= 5.0:
        W = np.arctan((RFM / RIM)**0.25) - np.pi / 4.0
        R1 = np.sin(THETA / 4.0)**2
        S1 = np.tan(2.0 * W)**2
        L = (R1 + S1) / (R1 + S1 + np.cos(THETA / 2.0))
    else:
        L = ((1.0 - LAMBDA) / (1.0 + LAMBDA))**2

    M_param = 8.0 * MU * TOF**2 / (S**3 * (1.0 + LAMBDA)**6)
    TPAR = (np.sqrt(2.0 / MU) / 3.0) * (S**1.5 - B1 * (S - C_chord)**1.5)
    L1 = (1.0 - L) / 2.0

    CHECKFEAS = 0
    N1 = 0
    N = 0

    # ── helper: Gauticci continued-fraction + Y iteration (shared code) ──
    def _gauticci_cf(X_val):
        """Compute C1 via Gauticci continued fraction for given X."""
        sqrt1pX = np.sqrt(max(1.0 + X_val, 0.0))
        ETA = X_val / (sqrt1pX + 1.0)**2

        DELTA_ = 1.0
        U_ = 1.0
        SIGMA_ = 1.0
        M1_ = 0
        while abs(U_) > TOL and M1_ <= nitermax:
            M1_ += 1
            GAMMA_ = (M1_ + 3.0)**2 / (4.0 * (M1_ + 3.0)**2 - 1.0)
            DELTA_ = 1.0 / (1.0 + GAMMA_ * ETA * DELTA_)
            U_ = U_ * (DELTA_ - 1.0)
            SIGMA_ = SIGMA_ + U_
        C1_ = 8.0 * (sqrt1pX + 1.0) / (
            3.0 + 1.0 / (5.0 + ETA + (9.0 * ETA / 7.0) * SIGMA_)
        )
        return C1_, M1_

    def _ku_cf(U_val):
        """Compute K(u) continued fraction and return KU, N1_count."""
        DELTA_ = 1.0
        U0_ = 1.0
        SIGMA_ = 1.0
        N1_ = 0
        while N1_ < nitermax and abs(U0_) >= TOL:
            if N1_ == 0:
                GAMMA_ = 4.0 / 27.0
                DELTA_ = 1.0 / (1.0 - GAMMA_ * U_val * DELTA_)
                U0_ = U0_ * (DELTA_ - 1.0)
                SIGMA_ = SIGMA_ + U0_
            else:
                for I8 in (1, 2):
                    if I8 == 1:
                        GAMMA_ = (
                            2.0 * (3.0 * N1_ + 1.0) * (6.0 * N1_ - 1.0)
                            / (9.0 * (4.0 * N1_ - 1.0) * (4.0 * N1_ + 1.0))
                        )
                    else:
                        GAMMA_ = (
                            2.0 * (3.0 * N1_ + 2.0) * (6.0 * N1_ + 1.0)
                            / (9.0 * (4.0 * N1_ + 1.0) * (4.0 * N1_ + 3.0))
                        )
                    DELTA_ = 1.0 / (1.0 - GAMMA_ * U_val * DELTA_)
                    U0_ = U0_ * (DELTA_ - 1.0)
                    SIGMA_ = SIGMA_ + U0_
            N1_ += 1
        KU_ = (SIGMA_ / 3.0)**2
        return KU_, N1_

    # ====================================================================
    # ZERO-REVOLUTION (Nrev == 0)
    # ====================================================================
    if Nrev == 0:
        Y = 1.0
        N = 0
        N1 = 0
        ERROR = 0

        if (TOF - TPAR) <= 1e-3:
            X0 = 0.0
        else:
            X0 = L

        X = -1.0e8

        while abs(X0 - X) >= abs(X) * TOL + TOL and N <= nitermax:
            N += 1
            X = X0
            CHECKFEAS = 1

            C1, _m1 = _gauticci_cf(X)

            # H1, H2
            if N == 1:
                DENOM = (1.0 + 2.0 * X + L) * (3.0 * C1 + X * C1 + 4.0 * X)
                H1 = (L + X)**2 * (C1 + 1.0 + 3.0 * X) / DENOM
                H2 = M_param * (C1 + X - L) / DENOM
            else:
                QR = np.sqrt(max(L1**2 + M_param / Y**2, 0.0))
                XPLL = QR - L1
                LP2XP1 = 2.0 * QR
                DENOM = LP2XP1 * (3.0 * C1 + X * C1 + 4.0 * X)
                H1 = XPLL**2 * (C1 + 1.0 + 3.0 * X) / DENOM
                H2 = M_param * (C1 + X - L) / DENOM

            B_val = 27.0 * H2 / (4.0 * (1.0 + H1)**3)
            sqrtBp1 = np.sqrt(max(B_val + 1.0, 0.0))
            U_val = -B_val / (2.0 * (sqrtBp1 + 1.0))

            KU, N1 = _ku_cf(U_val)

            denomY = 1.0 - 2.0 * U_val * KU
            if abs(denomY) < 1e-30:
                return 0., 0., 0., 1, np.zeros(3), np.zeros(3), TPAR, THETA

            Y = ((1.0 + H1) / 3.0) * (2.0 + sqrtBp1 / denomY)

            if abs(Y) < 1e-30:
                return 0., 0., 0., 1, np.zeros(3), np.zeros(3), TPAR, THETA

            X0 = np.sqrt(max(((1.0 - L) / 2.0)**2 + M_param / Y**2, 0.0)) - (1.0 + L) / 2.0

    # ====================================================================
    # MULTI-REVOLUTION (Nrev > 0)
    # ====================================================================
    elif Nrev > 0 and 4.0 * TOF * LAMBDA != 0.0:

        checkNconvRSS = 1
        checkNconvOSS = 1
        N3 = 1

        while N3 < 3:
            # ── Original Successive Substitution (converges to small-a) ──
            if Ncase == 0 or checkNconvRSS == 0:
                Y = 1.0
                N = 0
                N1 = 0
                ERROR = 0

                if checkNconvOSS == 0:
                    X0 = 2.0 * X0
                    checkNconvOSS = 1
                elif checkNconvRSS == 0:
                    pass  # X0 taken from RSS
                else:
                    X0 = L

                X = -1.0e8

                while abs(X0 - X) >= abs(X) * TOL + TOL and N <= nitermax:
                    N += 1
                    X = X0
                    CHECKFEAS = 1
                    C1, _m1 = _gauticci_cf(X)

                    if N == 1:
                        DENOM = (1.0 + 2.0 * X + L) * (3.0 * C1 + X * C1 + 4.0 * X)
                        H1 = (L + X)**2 * (C1 + 1.0 + 3.0 * X) / DENOM
                        H2 = M_param * (C1 + X - L) / DENOM
                    else:
                        QR = np.sqrt(max(L1**2 + M_param / Y**2, 0.0))
                        XPLL = QR - L1
                        LP2XP1 = 2.0 * QR
                        DENOM = LP2XP1 * (3.0 * C1 + X * C1 + 4.0 * X)
                        H1 = XPLL**2 * (C1 + 1.0 + 3.0 * X) / DENOM
                        H2 = M_param * (C1 + X - L) / DENOM

                    # Multi-rev correction
                    if abs(X) > 1e-30:
                        H3 = M_param * Nrev * np.pi / (4.0 * X * np.sqrt(max(X, 0.0)))
                    else:
                        H3 = 0.0
                    H2 = H3 + H2

                    B_val = 27.0 * H2 / (4.0 * (1.0 + H1)**3)
                    sqrtBp1 = np.sqrt(max(B_val + 1.0, 0.0))
                    U_val = -B_val / (2.0 * (sqrtBp1 + 1.0))

                    KU, N1 = _ku_cf(U_val)
                    denomY = 1.0 - 2.0 * U_val * KU
                    if abs(denomY) < 1e-30:
                        denomY = 1e-30

                    Y = ((1.0 + H1) / 3.0) * (2.0 + sqrtBp1 / denomY)

                    if Y > np.sqrt(M_param / L):
                        checkNconvOSS = 0
                        break

                    X0 = np.sqrt(max(((1.0 - L) / 2.0)**2 + M_param / Y**2, 0.0)) - (1.0 + L) / 2.0

                if N >= nitermax:
                    checkNconvOSS = 0

            # ── Reverse Successive Substitution (converges to large-a) ──
            if (Ncase == 1 or checkNconvOSS == 0) and not (checkNconvRSS == 0 and checkNconvOSS == 0):
                N = 0
                N1 = 0
                ERROR = 0

                if checkNconvRSS == 0:
                    X0 = X0 / 2.0
                    checkNconvRSS = 1
                elif checkNconvOSS == 0:
                    pass  # X0 from OSS
                else:
                    X0 = L

                X = -1.0e8

                while abs(X0 - X) >= abs(X) * TOL + TOL and N <= nitermax:
                    N += 1
                    X = X0
                    CHECKFEAS = 1

                    # Y from reverse substitution: y1 = sqrt(M / ((L+X)*(1+X)))
                    denom_y1 = (L + X) * (1.0 + X)
                    if denom_y1 <= 0:
                        checkNconvRSS = 0
                        break
                    Y = np.sqrt(M_param / denom_y1)

                    if Y < 1.0:
                        checkNconvRSS = 0
                        break

                    # Newton-Raphson for E_rss
                    Erss = 2.0 * np.arctan(np.sqrt(max(X, 0.0)))
                    h_val, _ = _h_E(Erss, Y, M_param, Nrev)
                    while h_val < 0:
                        Erss = Erss / 2.0
                        h_val, _ = _h_E(Erss, Y, M_param, Nrev)

                    Nnew = 1
                    Erss_old = -1.0e8
                    while abs(Erss - Erss_old) >= abs(Erss) * TOL + TOL and Nnew < nitermax:
                        Nnew += 1
                        h_val, dh_val = _h_E(Erss, Y, M_param, Nrev)
                        Erss_old = Erss
                        if abs(dh_val) > 1e-30:
                            Erss = Erss - h_val / dh_val

                    X0 = np.tan(Erss / 2.0)**2

            if checkNconvOSS == 1 and checkNconvRSS == 1:
                break

            if checkNconvRSS == 0 and checkNconvOSS == 0:
                return 0., 0., 0., 3, np.zeros(3), np.zeros(3), 0., 0.

            N3 += 1

        if N3 == 3:
            return 0., 0., 0., 3, np.zeros(3), np.zeros(3), 0., 0.

    # ====================================================================
    # Compute velocity vectors
    # ====================================================================
    if CHECKFEAS == 0:
        return 0., 0., 0., 1, np.zeros(3), np.zeros(3), 0., 0.

    if N1 >= nitermax or N >= nitermax:
        return 0., 0., 0., 4, np.zeros(3), np.zeros(3), 0., 0.

    # Guard singular cases
    if abs(X0) < 1e-30 or abs(Y) < 1e-30 or abs(LAMBDA) < 1e-30:
        return 0., 0., 0., 1, np.zeros(3), np.zeros(3), TPAR, THETA

    CONST = M_param * S * (1.0 + LAMBDA)**2
    if abs(CONST) < 1e-30:
        return 0., 0., 0., 1, np.zeros(3), np.zeros(3), TPAR, THETA

    A = CONST / (8.0 * X0 * Y**2)

    R11 = (1.0 + LAMBDA)**2 / (4.0 * TOF * LAMBDA)
    S11 = Y * (1.0 + X0)
    T11 = CONST / S11

    VI = -R11 * (S11 * (RI - RF) - T11 * RI / RIM)
    VF = -R11 * (S11 * (RI - RF) + T11 * RF / RFM)

    P = (
        2.0 * RIM * RFM * Y**2 * (1.0 + X0)**2 * np.sin(THETA / 2.0)**2
    ) / CONST

    e2 = 1.0 - P / A
    if e2 < 0 and abs(e2) < 1e-12:
        e2 = 0.0
    E = np.sqrt(abs(e2))

    return A, P, E, ERROR, VI, VF, TPAR, THETA


def solve_lambert(r1_vec, r2_vec, tof, mu=MU_EARTH, prograde=True, Nrev=0, Ncase=0):
    """Convenience wrapper around lambertMR."""
    orbitType = 0 if prograde else 1
    A, P, E, ERROR, VI, VF, TPAR, THETA = lambertMR(
        r1_vec, r2_vec, tof, mu, orbitType=orbitType, Nrev=Nrev, Ncase=Ncase
    )
    if ERROR != 0:
        raise RuntimeError(f"lambertMR failed with ERROR={ERROR}")
    if not (np.all(np.isfinite(VI)) and np.all(np.isfinite(VF))):
        raise RuntimeError("lambertMR returned non-finite velocities")
    return VI, VF