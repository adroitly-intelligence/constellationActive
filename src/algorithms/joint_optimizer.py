"""Joint Rosengren-style iteration for SMA (repeat closure) + eccentricity (frozen).

Per iteration, two sequential corrections are applied:

  A. SMA  — propagate one repeat cycle; measure ascending-node closure (geocentric
             longitude of the last AN minus the first); apply analytical Newton step:
               Δa = −closure_rad / J_a
             where J_a = (k·T_nodal/a)·(−2·Ω̇_J2 − 1.5·ω⊕).
             Derived from d(closure)/da accounting for both dΩ̇_J2/da and
             the SMA dependence of the nodal period T_nodal.

  B. Ecc  — propagate n_cycles repeat cycles; compute rotating-frame time-average
             (ex_avg, ey_avg); apply Rosengren residual correction:
               Δ(ex, ey) = (ex_target − ex_avg, ey_target − ey_avg)

Reference: Section IV of A34934.
"""

import numpy as np

from org.orekit.orbits import KeplerianOrbit

from src.core.frozen_orbit    import nodal_regression_rate, OMEGA_EARTH
from src.models.j3_propagator import build_orbit, build_propagator, propagate
from src.utils.analysis       import ascending_node_longitudes
from src.algorithms.rosengren import eccentricity_timeseries_rotating


def joint_frozen_repeat(
    a_init, ex_init, ey_init,
    ex_target, ey_target,
    i_rad, raan_rad, m0_rad,
    epoch, t_cycle, dt, dt_coarse,
    gcrf, itrf, earth, mu,
    omega_dot, n_cycles,
    k_orbits,
    n_iter        = 20,
    tol_closure_deg = 0.01,
    tol_ecc       = 1e-9,
    verbose       = True,
):
    """Joint SMA + eccentricity Rosengren correction.

    Parameters
    ----------
    a_init, ex_init, ey_init : float  Initial osculating values (analytical seed).
    ex_target, ey_target     : float  Analytical frozen eccentricity target.
    i_rad, raan_rad, m0_rad  : float  Fixed elements [rad].
    epoch                    : AbsoluteDate
    t_cycle                  : float  Repeat cycle duration [s].
    dt                       : float  Fine step for eccentricity propagation [s].
    dt_coarse                : float  Coarse step for ascending-node detection [s].
    gcrf, itrf               : Frame
    earth                    : OneAxisEllipsoid
    mu                       : float  Gravitational parameter [m³/s²].
    omega_dot                : float  J2 perigee-precession rate [rad/s] (for
                                      rotating-frame detrending in eccentricity step).
    n_cycles                 : int    Repeat cycles for eccentricity averaging.
    k_orbits                 : int    Number of orbits per repeat cycle.
    n_iter                   : int    Maximum iterations.
    tol_closure_deg          : float  Closure tolerance [deg].
    tol_ecc                  : float  Eccentricity residual tolerance [—].
    verbose                  : bool

    Returns
    -------
    a_opt, ex_opt, ey_opt : float
    history               : list of dict  [{a, ex, ey, closure_deg, ecc_residual, da}]
    mean_history          : list of (ex_avg, ey_avg)   rotating-frame means per iter
    """
    a   = float(a_init)
    exc = float(ex_init)
    eyc = float(ey_init)

    history      = []
    mean_history = []

    for k in range(n_iter):
        e_use   = float(np.hypot(exc, eyc))
        aop_use = float(np.arctan2(eyc, exc))

        if verbose:
            print(f"\nJoint iter {k + 1}/{n_iter}")

        # ── A: SMA correction — ascending-node closure ────────────────────────
        orb_a  = build_orbit(a, e_use, i_rad, raan_rad, aop_use, m0_rad, epoch, gcrf, mu)
        prop_a = build_propagator(orb_a, itrf, gravity_degree=3)
        _, _, lat_1c, lon_1c, _ = propagate(
            prop_a, epoch, t_cycle, dt_coarse, gcrf, itrf, earth,
        )

        an_lons, _ = ascending_node_longitudes(lat_1c, lon_1c)

        if len(an_lons) >= 2:
            closure_deg = float((an_lons[-1] - an_lons[0] + 180.0) % 360.0 - 180.0)
        else:
            closure_deg = 0.0
            if verbose:
                print("  [SMA]  WARNING: fewer than 2 AN crossings found — skipping correction")

        # Analytical Newton step. Closure = (Ω̇_J2 − ω⊕)·k·T_nodal.
        # d(closure)/da = k·T_nodal/a · (−2·Ω̇_J2 − 1.5·ω⊕)
        # from d(Ω̇_J2)/da ∝ a^(−7/2) and dT_nodal/da ∝ a^(3/2).
        nodal_rate = nodal_regression_rate(a, e_use, i_rad)   # rad/s  (<0 prograde)
        T_nodal    = t_cycle / k_orbits                        # s
        J_a        = (k_orbits * T_nodal / a) * (-2.0 * nodal_rate - 1.5 * OMEGA_EARTH)
        da         = -np.radians(closure_deg) / J_a
        a  += da

        if verbose:
            print(f"  [SMA]  closure = {closure_deg:+.5f} deg   Δa = {da:+.5f} m"
                  f"   a = {a:.4f} m")

        # ── B: Eccentricity correction — Rosengren rotating-frame average ────
        orb_e  = build_orbit(a, e_use, i_rad, raan_rad, aop_use, m0_rad, epoch, gcrf, mu)
        prop_e = build_propagator(orb_e, itrf, gravity_degree=3)
        _, ex_rot, ey_rot = eccentricity_timeseries_rotating(
            prop_e, epoch, n_cycles * t_cycle, dt, omega_dot,
        )

        ex_avg = float(np.mean(ex_rot))
        ey_avg = float(np.mean(ey_rot))
        mean_history.append((ex_avg, ey_avg))

        d_ex = float(ex_target - ex_avg)
        d_ey = float(ey_target - ey_avg)
        exc += d_ex
        eyc += d_ey

        ecc_residual = float(np.hypot(d_ex, d_ey))

        if verbose:
            print(f"  [Ecc]  mean = ({ex_avg:.4e}, {ey_avg:.4e})"
                  f"   residual = {ecc_residual:.3e}"
                  f"   (ex, ey) → ({exc:.4e}, {eyc:.4e})")

        history.append({
            'a':            a,
            'ex':           exc,
            'ey':           eyc,
            'closure_deg':  closure_deg,
            'ecc_residual': ecc_residual,
            'da':           da,
        })

        if abs(closure_deg) < tol_closure_deg and ecc_residual < tol_ecc:
            if verbose:
                print(f"  Converged after {k + 1} iterations.")
            break

    return a, exc, eyc, history, mean_history
