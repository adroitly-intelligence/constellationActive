#!/usr/bin/env python3
"""
design_frozen_repeat.py  -  Frozen Repeat Orbit designer (Phase 1 + 2).

Supports three orbit types (set in config.yaml  neqfro.orbit_type):

  near_equatorial  — inclination is a fixed input; SMA solved from repeat.
  sun_synchronous  — SMA and inclination solved jointly from repeat + sun-sync
                     conditions; inclination_deg in config is ignored.
                     (D'Amico 2004, TerraSAR-X approach)
  custom           — both SMA (sma_m) and inclination_deg supplied directly.

Workflow
--------
1.  Analytical estimates via J2+J3 secular theory.
1b. J2+J3 zonal propagator ground-track repeat check.
1c. 2-D ground-track map.
1d. Altitude comparison: frozen (mean-init) vs circular.
2.  Rosengren eccentricity-vector iteration — corrects mean → osculating init.
2b. Altitude comparison: osculating-corrected vs mean-init.
2c. Eccentricity phase-space plot with Rosengren iteration history.

References
----------
A34934  : Frozen Repeat Near-Equatorial Low Earth Orbits, JSR 2020.
DAmico04: TerraSAR-X Reference Orbit Design, D'Amico & Montenbruck 2004.
"""

import pathlib
import sys

import numpy as np
import yaml

# ── 1. Orekit init MUST come before any Java-class imports ────────────────────
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from src.core.setup import init_orekit

ROOT = pathlib.Path(__file__).parent
cfg  = yaml.safe_load((ROOT / "config.yaml").read_text())
nfg  = cfg["neqfro"]

init_orekit(str(ROOT / cfg["paths"]["orekit_data"]))

# ── 2. Java-backed imports ────────────────────────────────────────────────────
from org.orekit.time   import AbsoluteDate
from org.orekit.orbits import KeplerianOrbit

from src.core.frames       import get_frames, MU
from src.core.frozen_orbit import (
    analytical_frozen_orbit, print_analytical_estimates,
    orbital_period, perigee_precession_rate,
)
from src.models.j3_propagator import build_orbit, build_propagator, propagate
from src.utils.analysis       import check_ground_track_repeat, check_frozen_eccentricity
from src.algorithms.rosengren import find_frozen_eccentricity
from src.utils.plotting       import (plot_ground_track, plot_altitude_frozen,
                                      plot_ecc_phase_space)


# ── 3. Reference frames ───────────────────────────────────────────────────────
utc, gcrf, itrf, earth = get_frames()

# ── 4. Read config ────────────────────────────────────────────────────────────
orbit_type = nfg.get("orbit_type", "near_equatorial")
raan_rad   = np.radians(nfg["raan_deg"])
aop_rad    = np.radians(nfg["aop_deg"])
m0_rad     = np.radians(nfg["mean_anomaly_deg"])
k          = int(nfg["repeat_k"])
q          = float(nfg["repeat_q_days"])
dt         = float(nfg["step_size_s"])
plot_dt    = min(dt, 30.0)               # finer step for dense ground-track
t_cycle    = q * 86_164.0905
out_dir    = ROOT / nfg["output_dir"]

y, mo, d = (int(x) for x in nfg["epoch"][:10].split("-"))
epoch = AbsoluteDate(y, mo, d, 0, 0, 0.0, utc)


# ── 5. Analytical estimates ───────────────────────────────────────────────────
print(f"\nStep 1: Analytical estimates  [orbit_type = {orbit_type}]")

params = analytical_frozen_orbit(
    orbit_type  = orbit_type,
    k_orbits    = k,
    q_days      = q,
    i_deg       = nfg.get("inclination_deg"),
    a_m         = nfg.get("sma_m"),
    e_init      = 0.001,
    aop_deg     = nfg["aop_deg"],
)

a0    = params["a"]
i_rad = params["i_rad"]
e0    = params["e"]

print_analytical_estimates(k, q, orbit_type, a0, params["i_deg"], e0,
                            omega_deg=nfg["aop_deg"])

# Cover at least 2 eccentricity beat periods so the circular orbit completes
# full oscillation cycles, making the frozen vs non-frozen contrast clear.
_T_beat   = 2 * np.pi / abs(perigee_precession_rate(a0, e0, i_rad))  # [s]
n_alt_cyc = max(int(nfg.get("n_alt_cycles", 5)),
                int(np.ceil(2.0 * _T_beat / t_cycle)))


# ── 5b. J2+J3 zonal propagator ground-track repeat check ─────────────────────
print("\nStep 1b: J2+J3 zonal propagator ground-track repeat check")

# Ground-track: 2 cycles, fine time step for visual density
_j3_orb_gt  = build_orbit(a0, e0, i_rad, raan_rad, aop_rad, m0_rad, epoch, gcrf, MU)
_j3_prop_gt = build_propagator(_j3_orb_gt, itrf, gravity_degree=3)
t_gt, _, lat_gt, lon_gt, _ = propagate(
    _j3_prop_gt, epoch, 2 * t_cycle, plot_dt, gcrf, itrf, earth,
)
mask = t_gt <= t_cycle
check_ground_track_repeat(t_gt[mask], lat_gt[mask], lon_gt[mask], k)

# Altitude: frozen orbit over n_alt_cyc cycles to reveal long-term stability
_j3_orb_alt  = build_orbit(a0, e0, i_rad, raan_rad, aop_rad, m0_rad, epoch, gcrf, MU)
_j3_prop_alt = build_propagator(_j3_orb_alt, itrf, gravity_degree=3)
t_alt, _, _, _, alt_frozen = propagate(
    _j3_prop_alt, epoch, n_alt_cyc * t_cycle, dt, gcrf, itrf, earth,
)

# Circular reference (e=0): J3 freely pumps eccentricity; altitude envelope
# drifts over the beat period (~360/ω̇ days), showing why frozen design matters.
_circ_orbit = build_orbit(a0, 0.0, i_rad, raan_rad, aop_rad, m0_rad, epoch, gcrf, MU)
_circ_prop  = build_propagator(_circ_orbit, itrf, gravity_degree=3)
_, _, _, _, alt_circular = propagate(
    _circ_prop, epoch, n_alt_cyc * t_cycle, dt, gcrf, itrf, earth,
)


# ── 5c. 2-D ground-track map (2 cycles, fine step) ───────────────────────────
print("\nStep 1c: Ground-track map")

plot_ground_track(
    t_gt, lat_gt, lon_gt, k,
    t_cycle=t_cycle,
    out_file=out_dir / "ground_track.png",
)

# ── 5d. Altitude comparison: frozen vs circular (n_alt_cyc cycles) ────────────
print("\nStep 1d: Altitude comparison (frozen vs circular)")

plot_altitude_frozen(
    t_alt, alt_frozen, alt_nonfrozen=alt_circular,
    label_frozen    = "Analytically frozen (mean-init)",
    label_nonfrozen = "Circular (e = 0)",
    k_orbits=k, t_cycle=t_cycle,
    orbital_period_s=orbital_period(a0),
    out_file=out_dir / "altitude_frozen.png",
)


# ── Step 1e: Eccentricity phase-space — mean-init (pre-Rosengren) ─────────────
print("\nStep 1e: Eccentricity phase-space (mean-init, pre-Rosengren)")

_j3_orb_pre  = build_orbit(a0, e0, i_rad, raan_rad, aop_rad, m0_rad, epoch, gcrf, MU)
_j3_prop_pre = build_propagator(_j3_orb_pre, itrf, gravity_degree=3)

_t_pre = np.arange(0.0, n_alt_cyc * t_cycle + dt, dt)
ex_pre = np.empty(len(_t_pre))
ey_pre = np.empty(len(_t_pre))

for _n, _t in enumerate(_t_pre):
    _state       = _j3_prop_pre.propagate(epoch.shiftedBy(float(_t)))
    _kep         = KeplerianOrbit(_state.getOrbit())
    ex_pre[_n]   = float(_kep.getE()) * np.cos(float(_kep.getPerigeeArgument()))
    ey_pre[_n]   = float(_kep.getE()) * np.sin(float(_kep.getPerigeeArgument()))

_orb_nums_pre = _t_pre / orbital_period(a0)

plot_ecc_phase_space(
    ex_pre, ey_pre,
    ecc_target   = (e0 * np.cos(aop_rad), e0 * np.sin(aop_rad)),
    orbit_numbers = _orb_nums_pre,
    title        = "Phase Space of Eccentricity Vector (mean-init)",
    out_file     = out_dir / "ecc_phase_space_pre_rosengren.png",
)



# ── Step 2: Rosengren eccentricity-vector correction ─────────────────────────
print("\nStep 2: Rosengren eccentricity-vector iteration")

def _rosen_factory(ex_c, ey_c):
    e_use   = np.hypot(ex_c, ey_c)
    aop_use = np.arctan2(ey_c, ex_c)
    orb = build_orbit(a0, e_use, i_rad, raan_rad, aop_use, m0_rad, epoch, gcrf, MU)
    return build_propagator(orb, itrf, gravity_degree=3)

# Perigee precession rate — needed to detrend the secular ω̇ rotation so
# the rotating-frame average converges to (0, e_f) for any n_cycles.
_omega_dot   = perigee_precession_rate(a0, e0, i_rad)
_n_rosen_cyc = int(nfg.get("rosengren_n_cycles", 5))
print(f"  Averaging window : {_n_rosen_cyc} repeat cycles "
      f"(rotating-frame average; beat-period constraint removed)")

ex_f, ey_f, rosen_history, mean_history, (ex_avg_f, ey_avg_f) = find_frozen_eccentricity(
    propagator_factory = _rosen_factory,
    epoch              = epoch,
    t_cycle            = t_cycle,
    dt                 = dt,
    gcrf               = gcrf,
    ex_target          = e0 * np.cos(aop_rad),
    ey_target          = e0 * np.sin(aop_rad),
    omega_dot          = _omega_dot,
    n_cycles           = _n_rosen_cyc,
    n_iter             = int(nfg.get("rosengren_n_iter", 50)),
    verbose            = True,
)

e_osc   = np.hypot(ex_f, ey_f)
aop_osc = np.arctan2(ey_f, ex_f)

# e_osc >> e_f is expected: J2 short-period correction at u₀ = ω₀+M₀ dominates
# the osculating eccentricity; its TIME-MEAN equals e_f by design.
print(f"\n  Osculating correction:")
print(f"    e   : {e0:.6e}  ->  {e_osc:.6e}  (J2 SP correction dominates at u0={np.degrees(aop_rad+m0_rad):.1f} deg)")
print(f"    aop : {np.degrees(aop_rad):.3f} deg  ->  {np.degrees(aop_osc):.3f} deg")


# ── Step 2b: Altitude — mean-init vs osculating-corrected ─────────────────────
print("\nStep 2b: Altitude comparison — mean-init vs osculating-corrected")

_j3_orb_corr  = build_orbit(a0, e_osc, i_rad, raan_rad, aop_osc, m0_rad, epoch, gcrf, MU)
_j3_prop_corr = build_propagator(_j3_orb_corr, itrf, gravity_degree=3)
_, _, _, _, alt_corrected = propagate(
    _j3_prop_corr, epoch, n_alt_cyc * t_cycle, dt, gcrf, itrf, earth,
)

plot_altitude_frozen(
    t_alt, alt_corrected, alt_nonfrozen=alt_frozen,
    label_frozen    = "Frozen (osculating-corrected)",
    label_nonfrozen = "Frozen (mean-init, Phase 1)",
    k_orbits=k, t_cycle=t_cycle,
    orbital_period_s=orbital_period(a0),
    out_file=out_dir / "altitude_corrected.png",
)


# ── Step 2c: Eccentricity phase-space ─────────────────────────────────────────
print("\nStep 2c: Eccentricity phase-space")

_j3_orb_ps  = build_orbit(a0, e_osc, i_rad, raan_rad, aop_osc, m0_rad, epoch, gcrf, MU)
_j3_prop_ps = build_propagator(_j3_orb_ps, itrf, gravity_degree=3)

_t_ps = np.arange(0.0, n_alt_cyc * t_cycle + dt, dt)
ex_ps = np.empty(len(_t_ps))
ey_ps = np.empty(len(_t_ps))

for _n, _t in enumerate(_t_ps):
    _state    = _j3_prop_ps.propagate(epoch.shiftedBy(float(_t)))
    _kep      = KeplerianOrbit(_state.getOrbit())
    _e        = float(_kep.getE())
    _aop      = float(_kep.getPerigeeArgument())
    ex_ps[_n] = _e * np.cos(_aop)
    ey_ps[_n] = _e * np.sin(_aop)

_ex_target  = e0 * np.cos(aop_rad)
_ey_target  = e0 * np.sin(aop_rad)
_ecc_result = check_frozen_eccentricity(
    ex_avg_f, ey_avg_f,       # rotating-frame mean from Rosengren
    _ex_target, _ey_target,
)

plot_ecc_phase_space(
    ex_ps, ey_ps,
    history       = rosen_history,
    mean_history  = mean_history,
    ecc_target    = (e0 * np.cos(aop_rad), e0 * np.sin(aop_rad)),
    orbit_numbers = _t_ps / orbital_period(a0),
    title         = "Phase Space of Eccentricity Vector (osculating-corrected)",
    out_file      = out_dir / "ecc_phase_space.png",
)


# ── Final design summary ───────────────────────────────────────────────────────
print("\n══ Final Design Summary ══════════════════════════════════════")

print(f"\n  Semi-major axis (analytical)  : {a0:.3f} m")

print(f"\n  Eccentricity")
print(f"    Mean frozen  e_f    : {e0:.6e}  at  aop = {np.degrees(aop_rad):.3f} deg")
print(f"    Osculating   e_osc  : {e_osc:.6e}  at  aop = {np.degrees(aop_osc):.3f} deg")
print(f"    Frozen check        : {'PASS ✓' if _ecc_result['passes'] else 'FAIL ✗'}  "
      f"(residual = {_ecc_result['residual']:.2e})")

print("\n══════════════════════════════════════════════════════════════")
