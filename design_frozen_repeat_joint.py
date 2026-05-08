#!/usr/bin/env python3
"""design_frozen_repeat_joint.py — Joint SMA + eccentricity optimisation.

Extends the frozen-repeat design with a joint Rosengren-style iteration that
simultaneously corrects:

  1. Mean SMA  — via ascending-node closure of the repeat ground track
  2. Mean eccentricity vector — via Rosengren rotating-frame averaging

Per iteration the two corrections are applied sequentially (SMA first, then
eccentricity), matching the approach described in A34934 §IV.

Outputs
-------
  joint_convergence.png        — Closure error + eccentricity residual vs iteration
  ecc_phase_space_joint.png    — Eccentricity phase space (osculating-corrected, joint)
  ground_track_joint.png       — Ground track with joint-corrected SMA
  altitude_joint.png           — Altitude: joint-corrected vs analytical mean-init
"""

import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from src.core.setup import init_orekit

ROOT = pathlib.Path(__file__).parent
cfg  = yaml.safe_load((ROOT / "config.yaml").read_text())
nfg  = cfg["neqfro"]

init_orekit(str(ROOT / cfg["paths"]["orekit_data"]))

# ── Java-backed imports ───────────────────────────────────────────────────────
from org.orekit.time   import AbsoluteDate
from org.orekit.orbits import KeplerianOrbit

from src.core.frames       import get_frames, MU
from src.core.frozen_orbit import (
    analytical_frozen_orbit, print_analytical_estimates,
    orbital_period, perigee_precession_rate,
)
from src.models.j3_propagator  import build_orbit, build_propagator, propagate
from src.utils.analysis        import check_ground_track_repeat, check_frozen_eccentricity
from src.algorithms.joint_optimizer import joint_frozen_repeat
from src.utils.plotting        import (plot_ground_track, plot_altitude_frozen,
                                       plot_ecc_phase_space)

# ── Reference frames ──────────────────────────────────────────────────────────
utc, gcrf, itrf, earth = get_frames()

# ── Config ────────────────────────────────────────────────────────────────────
orbit_type = nfg.get("orbit_type", "near_equatorial")
raan_rad   = np.radians(nfg["raan_deg"])
aop_rad    = np.radians(nfg["aop_deg"])
m0_rad     = np.radians(nfg["mean_anomaly_deg"])
k          = int(nfg["repeat_k"])
q          = float(nfg["repeat_q_days"])
dt         = float(nfg["step_size_s"])
plot_dt    = min(dt, 30.0)
dt_coarse  = min(dt, 60.0)          # coarse step for ascending-node detection
t_cycle    = q * 86_164.0905
out_dir    = ROOT / nfg["output_dir"]

y, mo, d = (int(x) for x in nfg["epoch"][:10].split("-"))
epoch = AbsoluteDate(y, mo, d, 0, 0, 0.0, utc)

# ── Step 1: Analytical estimates ──────────────────────────────────────────────
print(f"\nStep 1: Analytical estimates  [orbit_type = {orbit_type}]")

params = analytical_frozen_orbit(
    orbit_type = orbit_type,
    k_orbits   = k,
    q_days     = q,
    i_deg      = nfg.get("inclination_deg"),
    a_m        = nfg.get("sma_m"),
    e_init     = 0.001,
    aop_deg    = nfg["aop_deg"],
)

a0    = params["a"]
i_rad = params["i_rad"]
e0    = params["e"]

print_analytical_estimates(k, q, orbit_type, a0, params["i_deg"], e0,
                           omega_deg=nfg["aop_deg"])

_T_beat   = 2 * np.pi / abs(perigee_precession_rate(a0, e0, i_rad))
n_alt_cyc = max(int(nfg.get("n_alt_cycles", 5)),
                int(np.ceil(2.0 * _T_beat / t_cycle)))

# ── Step 1b: Baseline ground-track closure check ──────────────────────────────
print("\nStep 1b: Baseline ground-track closure (analytical SMA)")

_j3_orb_base  = build_orbit(a0, e0, i_rad, raan_rad, aop_rad, m0_rad, epoch, gcrf, MU)
_j3_prop_base = build_propagator(_j3_orb_base, itrf, gravity_degree=3)
t_gt0, _, lat_gt0, lon_gt0, _ = propagate(
    _j3_prop_base, epoch, 2 * t_cycle, plot_dt, gcrf, itrf, earth,
)
mask0 = t_gt0 <= t_cycle
check_ground_track_repeat(t_gt0[mask0], lat_gt0[mask0], lon_gt0[mask0], k)

# ── Step 2: Joint Rosengren iteration ─────────────────────────────────────────
print("\nStep 2: Joint SMA + eccentricity iteration")

ex_target  = e0 * np.cos(aop_rad)
ey_target  = e0 * np.sin(aop_rad)
_omega_dot = perigee_precession_rate(a0, e0, i_rad)
_n_cycles  = int(nfg.get("rosengren_n_cycles", 5))
_n_iter    = int(nfg.get("joint_n_iter", 20))
_tol_cl    = float(nfg.get("joint_tol_closure_deg", 0.01))
_tol_ecc   = float(nfg.get("joint_tol_ecc", 1e-9))

a_opt, ex_opt, ey_opt, history, mean_history = joint_frozen_repeat(
    a_init    = a0,
    ex_init   = ex_target,
    ey_init   = ey_target,
    ex_target = ex_target,
    ey_target = ey_target,
    i_rad     = i_rad,
    raan_rad  = raan_rad,
    m0_rad    = m0_rad,
    epoch     = epoch,
    t_cycle   = t_cycle,
    dt        = dt,
    dt_coarse = dt_coarse,
    gcrf      = gcrf,
    itrf      = itrf,
    earth     = earth,
    mu        = MU,
    omega_dot = _omega_dot,
    n_cycles  = _n_cycles,
    k_orbits  = k,
    n_iter    = _n_iter,
    tol_closure_deg = _tol_cl,
    tol_ecc         = _tol_ecc,
    verbose   = True,
)

e_opt   = float(np.hypot(ex_opt, ey_opt))
aop_opt = float(np.arctan2(ey_opt, ex_opt))

print(f"\n  Converged SMA        : {a0:.3f} m  →  {a_opt:.3f} m  (Δa = {a_opt - a0:+.4f} m)")
print(f"  Mean frozen e_f      : {e0:.6e}  at aop = {np.degrees(aop_rad):.3f} deg")
print(f"  Osculating e_opt     : {e_opt:.6e}  at aop = {np.degrees(aop_opt):.3f} deg")

# ── Step 2a: Convergence plot ─────────────────────────────────────────────────
print("\nStep 2a: Joint convergence plot")

iters     = list(range(1, len(history) + 1))
closures  = [abs(h['closure_deg'])  for h in history]
ecc_resid = [h['ecc_residual']      for h in history]
sma_corr  = [h['da']                for h in history]

fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)

axes[0].plot(iters, [h['a'] - a0 for h in history], 'o-', color='steelblue')
axes[0].axhline(0, color='gray', lw=0.8, ls='--')
axes[0].set_ylabel('SMA correction Δa [m]')
axes[0].set_title('Joint Optimisation Convergence')
axes[0].grid(True, alpha=0.3)

axes[1].semilogy(iters, [max(c, 1e-10) for c in closures], 's-', color='tomato')
axes[1].axhline(_tol_cl, color='gray', lw=0.8, ls='--',
                label=f'Closure tol ({_tol_cl} deg)')
axes[1].set_ylabel('|Closure error| [deg]')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3, which='both')

axes[2].semilogy(iters, [max(r, 1e-15) for r in ecc_resid], '^-', color='seagreen')
axes[2].axhline(_tol_ecc, color='gray', lw=0.8, ls='--',
                label=f'Ecc tol ({_tol_ecc:.0e})')
axes[2].set_xlabel('Iteration')
axes[2].set_ylabel('Ecc residual |Δ(ex,ey)|')
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3, which='both')

plt.tight_layout()
_conv_path = out_dir / "joint_convergence.png"
_conv_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(_conv_path, dpi=150, bbox_inches='tight')
print(f"Figure saved -> {_conv_path}")
plt.show()

# ── Step 2b: Ground-track check + plot with optimised SMA ─────────────────────
print("\nStep 2b: Ground-track check (joint-corrected SMA)")

_orb_gt_opt  = build_orbit(a_opt, e_opt, i_rad, raan_rad, aop_opt, m0_rad, epoch, gcrf, MU)
_prop_gt_opt = build_propagator(_orb_gt_opt, itrf, gravity_degree=3)
t_gt_opt, _, lat_gt_opt, lon_gt_opt, _ = propagate(
    _prop_gt_opt, epoch, 2 * t_cycle, plot_dt, gcrf, itrf, earth,
)
mask_opt = t_gt_opt <= t_cycle
check_ground_track_repeat(t_gt_opt[mask_opt], lat_gt_opt[mask_opt], lon_gt_opt[mask_opt], k)

plot_ground_track(
    t_gt_opt, lat_gt_opt, lon_gt_opt, k,
    t_cycle  = t_cycle,
    out_file = out_dir / "ground_track_joint.png",
)

# ── Step 2c: Altitude comparison ──────────────────────────────────────────────
print("\nStep 2c: Altitude comparison (joint-corrected vs analytical mean-init)")

_orb_alt_opt  = build_orbit(a_opt, e_opt, i_rad, raan_rad, aop_opt, m0_rad, epoch, gcrf, MU)
_prop_alt_opt = build_propagator(_orb_alt_opt, itrf, gravity_degree=3)
t_alt, _, _, _, alt_opt = propagate(
    _prop_alt_opt, epoch, n_alt_cyc * t_cycle, dt, gcrf, itrf, earth,
)

_orb_alt_base  = build_orbit(a0, e0, i_rad, raan_rad, aop_rad, m0_rad, epoch, gcrf, MU)
_prop_alt_base = build_propagator(_orb_alt_base, itrf, gravity_degree=3)
_, _, _, _, alt_base = propagate(
    _prop_alt_base, epoch, n_alt_cyc * t_cycle, dt, gcrf, itrf, earth,
)

plot_altitude_frozen(
    t_alt, alt_opt, alt_nonfrozen=alt_base,
    label_frozen    = "Joint-corrected (SMA + ecc)",
    label_nonfrozen = "Analytical mean-init",
    k_orbits        = k,
    t_cycle         = t_cycle,
    orbital_period_s = orbital_period(a_opt),
    out_file        = out_dir / "altitude_joint.png",
)

# ── Step 2d: Eccentricity phase-space ─────────────────────────────────────────
print("\nStep 2d: Eccentricity phase-space (joint-corrected)")

_orb_ps  = build_orbit(a_opt, e_opt, i_rad, raan_rad, aop_opt, m0_rad, epoch, gcrf, MU)
_prop_ps = build_propagator(_orb_ps, itrf, gravity_degree=3)

_t_ps = np.arange(0.0, n_alt_cyc * t_cycle + dt, dt)
ex_ps = np.empty(len(_t_ps))
ey_ps = np.empty(len(_t_ps))

for _n, _t in enumerate(_t_ps):
    _state    = _prop_ps.propagate(epoch.shiftedBy(float(_t)))
    _kep      = KeplerianOrbit(_state.getOrbit())
    _e        = float(_kep.getE())
    _aop      = float(_kep.getPerigeeArgument())
    ex_ps[_n] = _e * np.cos(_aop)
    ey_ps[_n] = _e * np.sin(_aop)

# Use iteration history for overlay
_osc_history  = [(h['ex'], h['ey'])    for h in history]
_mean_history = mean_history

_ecc_check = check_frozen_eccentricity(
    float(mean_history[-1][0]), float(mean_history[-1][1]),
    ex_target, ey_target,
)

plot_ecc_phase_space(
    ex_ps, ey_ps,
    history       = _osc_history,
    mean_history  = _mean_history,
    ecc_target    = (ex_target, ey_target),
    orbit_numbers = _t_ps / orbital_period(a_opt),
    title         = "Phase Space of Eccentricity Vector (joint-corrected)",
    out_file      = out_dir / "ecc_phase_space_joint.png",
)

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n══ Joint Design Summary ══════════════════════════════════════")
print(f"\n  Semi-major axis")
print(f"    Analytical          : {a0:.3f} m")
print(f"    Joint-corrected     : {a_opt:.3f} m  (Δa = {a_opt - a0:+.4f} m)")
print(f"    Ground-track closure: {history[-1]['closure_deg']:+.5f} deg  "
      f"({'PASS ✓' if abs(history[-1]['closure_deg']) < _tol_cl else 'FAIL ✗'})")
print(f"\n  Eccentricity")
print(f"    Mean frozen  e_f    : {e0:.6e}  at aop = {np.degrees(aop_rad):.3f} deg")
print(f"    Osculating   e_opt  : {e_opt:.6e}  at aop = {np.degrees(aop_opt):.3f} deg")
print(f"    Frozen check        : {'PASS ✓' if _ecc_check['passes'] else 'FAIL ✗'}  "
      f"(residual = {_ecc_check['residual']:.2e})")
print("\n══════════════════════════════════════════════════════════════")
