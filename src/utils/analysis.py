"""Orbit analysis utilities: RTN error computation and eccentricity phase-space.

No propagation or Orekit calls live here — inputs are plain numpy arrays.
"""

import numpy as np


# ── Ground-track repeat verification ─────────────────────────────────────────

def ascending_node_longitudes(lat_arr, lon_arr):
    """Find the geodetic longitude at each ascending node crossing.

    Ascending nodes are detected where geodetic latitude crosses zero from
    south to north (lat[i] < 0, lat[i+1] >= 0).  Linear interpolation gives
    sub-step longitude accuracy.

    Parameters
    ----------
    lat_arr : (N,)  Geodetic latitude  [deg]
    lon_arr : (N,)  Geodetic longitude [deg]

    Returns
    -------
    an_lons : (M,)  Longitude [deg] at each ascending node, in crossing order.
    an_idx  : (M,)  Integer index of the sample just before each crossing.
    """
    an_lons = []
    an_idx  = []
    for i in range(len(lat_arr) - 1):
        if lat_arr[i] < 0.0 and lat_arr[i + 1] >= 0.0:
            # linear interpolation fraction
            frac = -lat_arr[i] / (lat_arr[i + 1] - lat_arr[i])
            lon  = lat_arr[i] + frac * (lon_arr[i + 1] - lon_arr[i])
            # handle ±180° wrap when interpolating longitude
            dlon = lon_arr[i + 1] - lon_arr[i]
            if dlon > 180:
                dlon -= 360
            elif dlon < -180:
                dlon += 360
            lon = (lon_arr[i] + frac * dlon + 180) % 360 - 180
            an_lons.append(lon)
            an_idx.append(i)
    return np.array(an_lons), np.array(an_idx)


def check_ground_track_repeat(t_arr, lat_arr, lon_arr, k_orbits, tol_deg=0.5):
    """Verify that the ground track closes after k_orbits ascending node crossings.

    The repeat condition requires that the longitude at ascending node k equals
    the longitude at ascending node 0 (modulo 360°).  The per-orbit longitude
    drift should be approximately constant across all k orbits.

    Parameters
    ----------
    t_arr    : (N,)   Time array [s]
    lat_arr  : (N,)   Geodetic latitude  [deg]
    lon_arr  : (N,)   Geodetic longitude [deg]
    k_orbits : int    Expected number of orbits per repeat cycle.
    tol_deg  : float  Closure tolerance [deg].

    Returns
    -------
    dict with keys:
      'an_lons'        : (M,)    Ascending-node longitudes [deg]
      'drift_per_orbit': (M-1,)  Per-orbit longitude drift [deg]
      'closure_deg'    : float   lon[k] - lon[0] (should be ~0 deg)
      'passes'         : bool    True if |closure| < tol_deg
      'n_nodes'        : int     Number of ascending nodes found
    """
    an_lons, _ = ascending_node_longitudes(lat_arr, lon_arr)
    n = len(an_lons)

    if n < 2:
        return {'an_lons': an_lons, 'drift_per_orbit': np.array([]),
                'closure_deg': np.nan, 'passes': False, 'n_nodes': n}

    # Unwrap longitudes to remove ±180° jumps before computing drift
    an_lons_unwrap = np.unwrap(np.radians(an_lons)) * 180 / np.pi
    drift = np.diff(an_lons_unwrap)

    # Closure: difference between first and last AN longitude (wrapped to ±180)
    closure = (an_lons[-1] - an_lons[0] + 180) % 360 - 180

    print("── Ground-Track Repeat Check ─────────────────────────────")
    print(f"  Ascending nodes found : {n}  (expected {k_orbits + 1})")
    print(f"  Per-orbit drift       : {np.mean(drift):.4f} ± {np.std(drift):.4f} deg")
    print(f"  Closure residual      : {closure:.4f} deg  (tol = ±{tol_deg} deg)")
    if abs(closure) < tol_deg:
        print(f"  RESULT  : PASS ✓  ground track repeats within tolerance")
    else:
        print(f"  RESULT  : FAIL ✗  closure exceeds tolerance")
    print("──────────────────────────────────────────────────────────")

    return {
        'an_lons':         an_lons,
        'drift_per_orbit': drift,
        'closure_deg':     closure,
        'passes':          abs(closure) < tol_deg,
        'n_nodes':         n,
    }


# ── Eccentricity phase-space ──────────────────────────────────────────────────

def ecc_phase_space_center(ex_arr, ey_arr):
    """Return the mean (ex_c, ey_c) of the eccentricity phase-space cloud."""
    return float(np.mean(ex_arr)), float(np.mean(ey_arr))


def ecc_phase_space_radius(ex_arr, ey_arr):
    """RMS radius of the phase-space orbit around its centre."""
    ex_c, ey_c = ecc_phase_space_center(ex_arr, ey_arr)
    return float(np.sqrt(np.mean((ex_arr - ex_c)**2 + (ey_arr - ey_c)**2)))


def check_frozen_eccentricity(ex_arr, ey_arr, ex_target, ey_target, tol_frac=0.05):
    """Verify that the time-mean eccentricity vector matches the frozen target.

    Parameters
    ----------
    ex_arr, ey_arr : (N,)  Osculating eccentricity vector time series.
    ex_target      : float Target mean ex = e_f * cos(aop_f).
    ey_target      : float Target mean ey = e_f * sin(aop_f).
    tol_frac       : float Tolerance as a fraction of |e_target|.

    Returns
    -------
    dict with keys: ex_mean, ey_mean, residual, r_rms, passes
    """
    ex_mean, ey_mean = ecc_phase_space_center(ex_arr, ey_arr)
    r_rms            = ecc_phase_space_radius(ex_arr, ey_arr)
    e_target_mag     = np.hypot(ex_target, ey_target)
    residual         = np.hypot(ex_mean - ex_target, ey_mean - ey_target)
    tol              = tol_frac * e_target_mag
    passes           = residual < tol

    print("── Frozen Eccentricity Check ────────────────────────────")
    print(f"  Target  (ex, ey) : ({ex_target:.4e},  {ey_target:.4e})")
    print(f"  Mean    (ex, ey) : ({ex_mean:.4e},  {ey_mean:.4e})")
    print(f"  Residual         : {residual:.2e}  (tol = {tol:.2e} = {tol_frac*100:.0f}% of e_f)")
    print(f"  Phase-space RMS  : {r_rms:.2e}  (e_f = {e_target_mag:.4e})")
    if passes:
        print(f"  RESULT  : PASS ✓  frozen condition satisfied within tolerance")
    else:
        print(f"  RESULT  : FAIL ✗  mean eccentricity offset exceeds tolerance")
    print("──────────────────────────────────────────────────────────")

    return {
        'ex_mean': ex_mean, 'ey_mean': ey_mean,
        'residual': residual, 'r_rms': r_rms, 'passes': passes,
    }


# ── RTN (Radial-Transverse-Normal) frame errors ───────────────────────────────

def rtn_basis(r_ref, v_ref):
    """Return (e_R, e_T, e_N) unit vectors for the RTN frame.

    e_R : radial      (along position vector)
    e_N : normal      (perpendicular to orbital plane)
    e_T : transverse  (completes right-handed system)
    """
    e_R = r_ref / np.linalg.norm(r_ref)
    h   = np.cross(r_ref, v_ref)
    e_N = h / np.linalg.norm(h)
    e_T = np.cross(e_N, e_R)
    return e_R, e_T, e_N


def spatial_errors_rtn(r_ref, v_ref, r_actual):
    """Project position difference onto the RTN frame.

    Parameters
    ----------
    r_ref    : (3,) reference position [m]
    v_ref    : (3,) reference velocity [m/s]
    r_actual : (3,) actual position    [m]

    Returns
    -------
    eR, eT, eN : float  Radial, transverse, normal errors [m]
    """
    e_R, e_T, e_N = rtn_basis(np.asarray(r_ref), np.asarray(v_ref))
    delta = np.asarray(r_actual) - np.asarray(r_ref)
    return float(delta @ e_R), float(delta @ e_T), float(delta @ e_N)


def compute_rtn_timeseries(xyz_ref, vxyz_ref, xyz_actual):
    """Compute RTN error time series for two trajectory arrays.

    Parameters
    ----------
    xyz_ref    : (N, 3)  reference positions  [m]
    vxyz_ref   : (N, 3)  reference velocities [m/s]
    xyz_actual : (N, 3)  actual positions     [m]

    Returns
    -------
    eR, eT, eN : (N,) arrays of RTN errors [m]
    """
    n   = len(xyz_ref)
    eR  = np.empty(n)
    eT  = np.empty(n)
    eN  = np.empty(n)
    for i in range(n):
        eR[i], eT[i], eN[i] = spatial_errors_rtn(
            xyz_ref[i], vxyz_ref[i], xyz_actual[i]
        )
    return eR, eT, eN


# ── Summary statistics ────────────────────────────────────────────────────────

def rtn_summary(eR, eT, eN):
    """Print mean and STD of RTN errors."""
    for name, arr in [("Radial", eR), ("Transverse", eT), ("Normal", eN)]:
        print(f"  {name:12s}: mean = {np.mean(arr):+.2f} m,  "
              f"STD = {np.std(arr):.2f} m,  "
              f"peak = {np.max(np.abs(arr)):.2f} m")
