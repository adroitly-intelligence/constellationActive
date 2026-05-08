"""Analytical frozen-orbit theory based on J2+J3 secular perturbations.

Supports three orbit types, applied as sequential constraints
(D'Amico 2004, TerraSAR-X reference orbit design):

  "near_equatorial"  — inclination is a free input; SMA solved from repeat condition.
  "sun_synchronous"  — inclination is derived from the sun-sync condition given SMA;
                       SMA and inclination solved jointly.
  "custom"           — both SMA and inclination supplied externally; only frozen
                       eccentricity is computed analytically.

References
----------
A34934 : "Designing a Reference Trajectory for Frozen Repeat Near-Equatorial
           Low Earth Orbits", Journal of Spacecraft and Rockets, 2020.
DAmico04: "TerraSAR-X Reference Orbit Design", D'Amico & Montenbruck, 2004.

All angles in radians, distances in metres, time in seconds.
"""

import numpy as np

# EGM96 / WGS84 constants
J2          =  1.08262668e-3
J3          = -2.5415e-6
RE          =  6_378_137.0          # m
MU          =  3.986004418e14       # m^3 s^-2
OMEGA_EARTH =  7.2921150e-5         # rad/s  (Earth sidereal rotation rate)
T_SIDEREAL  =  86_164.0905          # s      (one sidereal day)
OMEGA_SUN   =  2.0 * np.pi / (365.25 * 86400.0)   # rad/s  (solar precession rate)


# ── Basic orbital mechanics ───────────────────────────────────────────────────

def two_body_potential(r):
    """Two-body (Keplerian) gravitational potential [m^2 s^-2].

    U_0 = mu / r

    This is the leading term of the geopotential expansion (D'Amico 2004, Eq. 1):

        U = (mu / r) * [1 - sum_{n=2}^{N} J_n * (R_E/r)^n * P_n(sin phi)]

    where the two-body part U_0 = mu/r drives the Keplerian reference orbit and
    the remaining zonal harmonics (J2, J3, ...) are treated as perturbations.

    Parameters
    ----------
    r : float  Geocentric distance [m].

    Returns
    -------
    float  U_0 [m^2 s^-2] (positive geodetic convention).
    """
    return MU / r


def mean_motion(a):
    """Keplerian mean motion [rad/s]."""
    return np.sqrt(MU / a**3)


def orbital_period(a):
    """Keplerian orbital period [s]."""
    return 2.0 * np.pi / mean_motion(a)


def nodal_regression_rate(a, e, i):
    """J2 secular RAAN rate dΩ/dt [rad/s]. Negative for prograde orbits."""
    n = mean_motion(a)
    return -(3.0 / 2.0) * n * J2 * (RE / a)**2 * np.cos(i) / (1 - e**2)**2


def perigee_precession_rate(a, e, i):
    """J2 secular argument-of-perigee rate dω/dt [rad/s]."""
    n = mean_motion(a)
    return (3.0 / 4.0) * n * J2 * (RE / a)**2 * (5 * np.cos(i)**2 - 1) / (1 - e**2)**2


def n_eff_factor(a, e, i):
    """Combined J2 secular correction γ such that n_eff = n · (1 + γ).

    Merges the secular mean anomaly rate (Ṁ) and argument-of-perigee rate (ω̇)
    into the single effective angular velocity of the argument of latitude
    (D'Amico 2004, governing repeat condition):

        n_eff = n · [1 + (3 J2 Rₑ²)/(4 p²) · (√(1−e²)·(3cos²i−1) + (5cos²i−1))]

    where p = a(1−e²) and the two bracket terms are:
      √(1−e²)·(3cos²i−1)  ← secular Ṁ correction  (Brouwer / Vallado Eq. 9-42)
      (5cos²i−1)           ← secular ω̇ correction  (= 4−5sin²i)

    Note: some sources write the Ṁ term with (3sin²i−1) which equals (2−3cos²i)
    and gives the wrong sign for retrograde/sun-sync inclinations.  The correct
    Brouwer form is (3cos²i−1), which is negative for i ∈ (54.7°, 125.3°).
    """
    eta2  = 1.0 - e**2                             # (1−e²)
    p2    = (a * eta2)**2                           # p²
    c2i   = np.cos(i)**2
    return (3.0 * J2 * RE**2 / (4.0 * p2)) * (np.sqrt(eta2) * (3.0*c2i - 1.0) + (5.0*c2i - 1.0))


# ── Repeat ground-track SMA ───────────────────────────────────────────────────

def repeat_ground_track_sma(k_orbits, q_days, i_rad, e=0.001, tol=1.0, max_iter=100):
    """SMA [m] for a k:q repeat ground-track orbit at fixed inclination.

    Solves the J2-corrected repeat condition (D'Amico 2004):

        n_eff = (m/k) · (ωE − Ω̇)

    where n_eff = n·(1+γ) combines the secular Ṁ and ω̇ rates (see n_eff_factor),
    and Ω̇ is the J2 nodal regression rate.  Rearranging for n then a:

        n = (m/k)·(ωE − Ω̇) / (1 + γ)
        a = (μ / n²)^(1/3)

    Parameters
    ----------
    k_orbits : int    Number of satellite revolutions per repeat cycle.
    q_days   : float  Repeat-cycle duration in sidereal days.
    i_rad    : float  Inclination [rad].
    e        : float  Eccentricity (for J2 corrections).
    """
    n0 = k_orbits * OMEGA_EARTH / q_days
    a  = (MU / n0**2) ** (1.0 / 3.0)

    for _ in range(max_iter):
        dOmega = nodal_regression_rate(a, e, i_rad)
        gamma  = n_eff_factor(a, e, i_rad)
        n_req  = (k_orbits * (OMEGA_EARTH - dOmega) / q_days) / (1.0 + gamma)
        a_new  = (MU / n_req**2) ** (1.0 / 3.0)
        if abs(a_new - a) < tol:
            return a_new
        a = a_new

    return a


# ── Sun-synchronous constraints ───────────────────────────────────────────────

def sun_sync_inclination(a, e=0.0):
    """Inclination [rad] for a sun-synchronous orbit at semi-major axis a [m].

    Derived by setting dΩ/dt = +OMEGA_SUN and solving for i (J2 secular model):

        cos(i) = -2 * OMEGA_SUN * a^3.5 * (1-e^2)^2
                 / (3 * sqrt(MU) * J2 * RE^2)

    Returns a retrograde inclination (> 90 deg) for typical LEO altitudes.

    Raises ValueError if the altitude is outside the sun-sync achievable range.
    """
    cos_i = (-2.0 * OMEGA_SUN * a**3.5 * (1.0 - e**2)**2
             / (3.0 * np.sqrt(MU) * J2 * RE**2))
    if abs(cos_i) > 1.0:
        raise ValueError(
            f"Sun-sync not achievable at a = {a/1e3:.1f} km  "
            f"(|cos i| = {abs(cos_i):.3f} > 1)"
        )
    return np.arccos(cos_i)


def repeat_sma_sun_sync(k_orbits, q_days, e=0.001, tol=1.0, max_iter=50):
    """Jointly solve for (a [m], i [rad]) satisfying repeat AND sun-sync.

    The two conditions are coupled:
      - Repeat fixes n given i  (via repeat_ground_track_sma)
      - Sun-sync fixes i given a  (via sun_sync_inclination)

    Alternates between the two until both converge.

    Parameters
    ----------
    k_orbits : int    Revolutions per repeat cycle.
    q_days   : float  Repeat cycle in sidereal days.
    e        : float  Eccentricity (for J2 corrections).

    Returns
    -------
    a : float  Semi-major axis [m]
    i : float  Inclination [rad]
    """
    i = np.radians(97.4)   # typical sun-sync seed

    for _ in range(max_iter):
        a = repeat_ground_track_sma(k_orbits, q_days, i, e=e, tol=tol / 10.0)
        i_new = sun_sync_inclination(a, e=e)
        if abs(i_new - i) < 1e-9:
            break
        i = i_new

    return a, i


# ── Frozen eccentricity ───────────────────────────────────────────────────────

def frozen_eccentricity(a, i_rad):
    """Analytical frozen eccentricity from J2+J3 secular balance (ω = 90 deg).

    Uses the Coffey-Deprit formula:
        e_f = -(J3 / 2J2) * (RE / a) * sin(i)

    Valid for ω = 90 deg (stable frozen point for J3 < 0).
    For ω = 270 deg the sign is reversed.

    Note: for sun-synchronous orbits (i ~ 97 deg) sin(i) ~ 0.99, giving a
    slightly larger frozen eccentricity than for near-equatorial orbits.
    """
    return -(J3 / (2.0 * J2)) * (RE / a) * np.sin(i_rad)


def eccentricity_vector(e, omega_rad):
    """Convert (e, omega) to Cartesian eccentricity vector (ex, ey)."""
    return e * np.cos(omega_rad), e * np.sin(omega_rad)


# ── Top-level design entry point ──────────────────────────────────────────────

def analytical_frozen_orbit(orbit_type, k_orbits, q_days,
                             i_deg=None, a_m=None, e_init=0.001,
                             aop_deg=90.0):
    """Compute analytical (a, i, e, omega) for the requested orbit type.

    Parameters
    ----------
    orbit_type : str   "near_equatorial" | "sun_synchronous" | "custom"
    k_orbits   : int   Revolutions per repeat cycle.
    q_days     : float Repeat cycle in sidereal days.
    i_deg      : float Required for "near_equatorial" and "custom" [deg].
    a_m        : float Required for "custom" [m].
    e_init     : float Eccentricity seed for J2 corrections.
    aop_deg    : float Argument of perigee [deg]. 90 or 270 for frozen condition.

    Returns
    -------
    dict with keys: a, i_rad, i_deg, e, aop_rad, orbit_type
    """
    if orbit_type == "near_equatorial":
        if i_deg is None:
            raise ValueError("near_equatorial requires i_deg")
        i_rad = np.radians(i_deg)
        a     = repeat_ground_track_sma(k_orbits, q_days, i_rad, e=e_init)

    elif orbit_type == "sun_synchronous":
        a, i_rad = repeat_sma_sun_sync(k_orbits, q_days, e=e_init)
        i_deg    = np.degrees(i_rad)

    elif orbit_type == "custom":
        if i_deg is None or a_m is None:
            raise ValueError("custom requires both i_deg and a_m")
        i_rad = np.radians(i_deg)
        a     = float(a_m)

    else:
        raise ValueError(f"Unknown orbit_type '{orbit_type}'. "
                         "Choose: near_equatorial | sun_synchronous | custom")

    e       = frozen_eccentricity(a, i_rad)
    aop_rad = np.radians(aop_deg)

    return dict(a=a, i_rad=i_rad, i_deg=float(i_deg),
                e=e, aop_rad=aop_rad, orbit_type=orbit_type)


# ── Console summary ───────────────────────────────────────────────────────────

def print_analytical_estimates(k, q, orbit_type, a, i_deg, e, omega_deg=90.0):
    """Pretty-print the analytical frozen-orbit estimates.

    Shows SMA for two gravity models:
      two-body : Keplerian seed — no perturbations.
      J2       : secular RAAN/perigee/mean-anomaly corrections applied.

    J3 does not have its own SMA correction; it determines the frozen
    eccentricity e_f, which feeds back into the J2 repeat condition only
    through the (1-e²) terms — a difference of a few metres, not shown
    separately.  The passed-in `a` is the J2 solution at the frozen e_f.
    """
    i_rad  = np.radians(i_deg)
    alt_km = (a - RE) / 1e3
    T_orb  = orbital_period(a) / 60.0
    dOmega = np.degrees(nodal_regression_rate(a, e, i_rad)) * 86400
    domega = np.degrees(perigee_precession_rate(a, e, i_rad)) * 86400

    # SMA comparison: two-body seed vs J2-corrected design value
    if k and q:
        n0      = k * OMEGA_EARTH / q
        a_2body = (MU / n0**2) ** (1.0 / 3.0)
        sma_lines = (
            f"  SMA (two-body)     : {a_2body/1e3:.3f} km\n"
            f"  SMA (J2)           : {a/1e3:.3f} km  "
            f"(+{(a - a_2body):.0f} m from J2)"
        )
    else:
        sma_lines = f"  SMA                : {a/1e3:.3f} km"

    # For sun-sync: show how close RAAN rate is to target
    raan_note = ""
    if orbit_type == "sun_synchronous":
        target = np.degrees(OMEGA_SUN) * 86400
        raan_note = f"  (target {target:.4f} deg/day)"

    print("=" * 60)
    print(f"  Analytical Frozen-Orbit Estimates  [{orbit_type}]")
    print("=" * 60)
    print(f"  Repeat cycle       : {k} orbits / {q} sidereal days")
    print(f"  Inclination        : {i_deg:.4f} deg")
    print(sma_lines)
    print(f"  Altitude           : {alt_km:.1f} km")
    print(f"  Eccentricity       : {e:.6e}")
    print(f"  Arg. of perigee    : {omega_deg:.1f} deg")
    print(f"  Orbital period     : {T_orb:.3f} min")
    print(f"  RAAN regression    : {dOmega:.4f} deg/day{raan_note}")
    print(f"  Perigee precession : {domega:.4f} deg/day")
    print("=" * 60)
