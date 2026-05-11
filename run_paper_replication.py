"""
Paper Replication: Table 2.1 — Near-Equatorial Frozen Repeat Orbit
Low et al. (2020), A34934, Section 2.

Workflow:
  1. Analytical seed (J2+J3 theory)
  2. Broyden SMA optimisation (paper Eqs. 4-7)
  3. Rosengren eccentricity optimisation (paper Eq. 9)
  4. Table 2.1 validation
  5. Eccentricity phase-space plots (paper Fig. 2.2A/B)
"""

import pathlib, sys, time, io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

# Write all output to a UTF-8 log file so conda run's cp1252 pipe never sees unicode
_log_path = ROOT / 'outputs' / 'paper_replication' / 'replication_run.log'
_log_path.parent.mkdir(parents=True, exist_ok=True)
_log = open(_log_path, 'w', encoding='utf-8')

class _Tee:
    """Write to multiple streams; ASCII-sanitize writes to the original stdout/stderr
    so conda run's cp1252 pipe never sees unicode box-drawing characters."""
    def __init__(self, raw_stream, log_stream):
        self.raw = raw_stream
        self.log = log_stream
    def write(self, s):
        self.log.write(s)
        self.raw.write(s.encode('ascii', errors='replace').decode('ascii'))
    def flush(self):
        self.log.flush()
        self.raw.flush()
    @property
    def encoding(self): return 'utf-8'

sys.stdout = _Tee(sys.__stdout__, _log)
sys.stderr = _Tee(sys.__stderr__, _log)

# ── Orekit must be first ───────────────────────────────────────────────────────
print("Initialising Orekit ...", flush=True)
from src.core.setup import init_orekit
init_orekit(str(ROOT / 'data' / 'orekit-data-main.zip'))
print("Orekit ready.\n", flush=True)

# ── Post-JVM imports ───────────────────────────────────────────────────────────
from org.orekit.time import AbsoluteDate

from src.core.frames       import get_frames, MU
from src.core.frozen_orbit import (
    repeat_ground_track_sma, frozen_eccentricity,
    perigee_precession_rate, nodal_regression_rate,
    OMEGA_EARTH, T_SIDEREAL, RE,
)
from src.models.j3_propagator  import build_orbit, build_propagator, propagate
from src.algorithms.rosengren  import find_frozen_eccentricity, eccentricity_timeseries
from src.utils.analysis        import ascending_node_longitudes

utc, gcrf, itrf, earth = get_frames()
print("Frames loaded.\n", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1 — PAPER PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
K_ORBITS  = 59       # revolutions per repeat cycle
Q_DAYS    = 4.0      # sidereal days per repeat cycle  (≈ 3.99 solar days)
I_DEG     = 10.0
AOP_DEG   = 90.0
RAAN_DEG  = 0.0
M0_DEG    = 0.0

i_rad    = np.radians(I_DEG)
aop_rad  = np.radians(AOP_DEG)
raan_rad = np.radians(RAAN_DEG)
m0_rad   = np.radians(M0_DEG)

GRAV_DEG   = 70
GRAV_ORD   = 70
THIRD_BODY = True

BROYDEN_MAX_ITER  = 15
BROYDEN_DELTA_A   = 100.0   # m
BROYDEN_TOL_DEG   = 1e-3
ROSEN_N_CYCLES    = 5     # rotating-frame averaging; >~10 cycles breaks due to omega_dot error accumulation
ROSEN_N_ITER      = 50
DT_COARSE         = 60.0    # s
DT_FINE           = 30.0    # s

epoch   = AbsoluteDate(2021, 1, 1, 0, 0, 0.0, utc)
T_CYCLE = Q_DAYS * T_SIDEREAL

# Table 2.1 targets
TARGET_OSC_A_KM  = 6934.63
TARGET_OSC_E     = 1.431e-3
TARGET_MEAN_A_KM = 6934.91
TARGET_MEAN_E    = 1.781e-8

print("=" * 60, flush=True)
print("  PAPER PARAMETERS")
print("=" * 60, flush=True)
print(f"  Repeat cycle  : {K_ORBITS}:{Q_DAYS:.0f}  ({K_ORBITS} orbits / {Q_DAYS:.0f} sid. days)")
print(f"  Cycle duration: {T_CYCLE/3600:.3f} hr  = {T_CYCLE/86400:.3f} solar days")
print(f"  Inclination   : {I_DEG} deg")
print(f"  Propagator    : EGM96 {GRAV_DEG}×{GRAV_ORD} + Sun + Moon")
print("=" * 60, "\n", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# 2 — ANALYTICAL SEED
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 60)
print("  STEP 1: Analytical seed")
print("─" * 60, flush=True)

a_seed     = repeat_ground_track_sma(K_ORBITS, Q_DAYS, i_rad, e=0.001)
e_f        = frozen_eccentricity(a_seed, i_rad)
ex_seed    = e_f * np.cos(aop_rad)   # = 0
ey_seed    = e_f * np.sin(aop_rad)   # = e_f
omega_dot  = perigee_precession_rate(a_seed, e_f, i_rad)
raan_dot   = nodal_regression_rate(a_seed, e_f, i_rad)
T_beat     = 2 * np.pi / abs(omega_dot)

print(f"  SMA   (J2)    : {a_seed/1e3:.4f} km   target mean: {TARGET_MEAN_A_KM:.2f} km"
      f"  Δ = {a_seed/1e3 - TARGET_MEAN_A_KM:+.3f} km")
print(f"  e_f           : {e_f:.4e}   target mean: {TARGET_MEAN_E:.3e}")
print(f"  (ex, ey) seed : ({ex_seed:.3e}, {ey_seed:.3e})")
print(f"  dΩ/dt         : {np.degrees(raan_dot)*86400:.4f} deg/day")
print(f"  dω/dt         : {np.degrees(omega_dot)*86400:.4f} deg/day")
print(f"  Beat period   : {T_beat/86400:.2f} days\n", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3 — BROYDEN SMA OPTIMISATION  (paper Eqs. 4–7)
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 60)
print("  STEP 2: Broyden SMA optimisation  (paper Eqs. 4-7)")
print(f"  Each propagation: {T_CYCLE/3600:.2f} hr at {DT_COARSE}s steps, EGM96 70×70 + 3rd body")
print("─" * 60, flush=True)

_prop_count = [0]

def measure_closure(a, ex_c, ey_c):
    """ΔΘ(a): ascending-node closure [deg] after one repeat cycle (paper Eq. 3)."""
    t0 = time.time()
    e_c   = float(np.hypot(ex_c, ey_c))
    aop_c = float(np.arctan2(ey_c, ex_c))
    orb   = build_orbit(a, e_c, i_rad, raan_rad, aop_c, m0_rad, epoch, gcrf, MU)
    prop  = build_propagator(orb, itrf,
                             gravity_degree=GRAV_DEG, gravity_order=GRAV_ORD,
                             third_body=THIRD_BODY)
    _, _, lat, lon, _ = propagate(prop, epoch, T_CYCLE, DT_COARSE, gcrf, itrf, earth)
    an_lons, _ = ascending_node_longitudes(lat, lon)
    closure = float((an_lons[-1] - an_lons[0] + 180.0) % 360.0 - 180.0) if len(an_lons) >= 2 else 0.0
    _prop_count[0] += 1
    print(f"    prop #{_prop_count[0]:2d}: a={a/1e3:.4f} km  closure={closure:+.5f} deg"
          f"  ({time.time()-t0:.1f}s)", flush=True)
    return closure


def broyden_sma(a_init, ex_c, ey_c):
    """Scalar Broyden root-finder for SMA ground-track closure (paper Eqs. 4-7)."""
    a  = float(a_init)
    print(f"\n  Computing f(a_seed) ...", flush=True)
    f0 = measure_closure(a, ex_c, ey_c)
    print(f"  Computing f(a_seed + δa) for initial Jacobian ...", flush=True)
    fp = measure_closure(a + BROYDEN_DELTA_A, ex_c, ey_c)
    J  = (fp - f0) / BROYDEN_DELTA_A   # [deg/m]
    print(f"\n  Initial Jacobian J = {J:.4e} deg/m\n", flush=True)

    history = [{'iter': 0, 'a_km': a/1e3, 'closure_deg': f0}]

    for k in range(BROYDEN_MAX_ITER):
        da    = -f0 / J
        a_new = a + da
        print(f"\n  Broyden iter {k+1}: Δa = {da:+.3f} m  →  a = {a_new/1e3:.4f} km", flush=True)
        f_new = measure_closure(a_new, ex_c, ey_c)
        history.append({'iter': k+1, 'a_km': a_new/1e3, 'closure_deg': f_new, 'da_m': da})

        if abs(f_new) < BROYDEN_TOL_DEG:
            print(f"\n  [OK] Broyden converged after {k+1} iterations."
                  f"  Final closure = {f_new:+.6f} deg", flush=True)
            return a_new, history

        # Broyden rank-1 update (Eq. 5)
        J = J + (f_new - f0 - J * (a_new - a)) / (a_new - a) ** 2
        a, f0 = a_new, f_new

    print(f"\n  WARNING: Broyden did not converge in {BROYDEN_MAX_ITER} iterations.", flush=True)
    return a_new, history


t_broyden_start = time.time()
a_opt, broyden_history = broyden_sma(a_seed, ex_seed, ey_seed)
t_broyden = time.time() - t_broyden_start

print(f"\n  Broyden wall time : {t_broyden/60:.1f} min")
print(f"  Optimised SMA     : {a_opt/1e3:.4f} km")
print(f"  Paper osc a*      : {TARGET_OSC_A_KM:.2f} km")
print(f"  Δa                : {a_opt/1e3 - TARGET_OSC_A_KM:+.3f} km\n", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# 4 — ROSENGREN ECCENTRICITY OPTIMISATION  (paper Eq. 9, 1-D formulation)
#
# Paper optimises SCALAR e only, with ω fixed at 90° throughout (ex = 0).
# IC is always (0, ey_c). Only ey is corrected each iteration.
# Averaging over 30 repeat cycles (paper Eq. 9: "N samples through 30 ground
# track repeats at 30s sampling interval").
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 60)
print("  STEP 3: Rosengren eccentricity optimisation  (paper Eq. 9, 1-D)")
print(f"  omega fixed at 90 deg (ex = 0), only ey corrected each iteration")
print(f"  Each iteration: {ROSEN_N_CYCLES} cycles = "
      f"{ROSEN_N_CYCLES * T_CYCLE / 86400:.1f} days  at {DT_FINE}s steps")
print("─" * 60, flush=True)

from src.algorithms.rosengren import eccentricity_timeseries_rotating

# Seed: analytical frozen eccentricity at omega = 90 deg
ex_c = 0.0          # omega = 90 deg -> ex = e*cos(90) = 0, fixed throughout
ey_c = float(e_f)   # omega = 90 deg -> ey = e*sin(90) = e_f

rosen_history_1d = []
t_rosen_start = time.time()

for k in range(ROSEN_N_ITER):
    t_iter = time.time()
    # Build propagator with current (ex_c=0, ey_c) — omega always = 90 deg
    orb_r = build_orbit(a_opt, ey_c, i_rad, raan_rad, aop_rad, m0_rad, epoch, gcrf, MU)
    prop_r = build_propagator(orb_r, itrf,
                              gravity_degree=GRAV_DEG, gravity_order=GRAV_ORD,
                              third_body=THIRD_BODY)

    _, ex_rot, ey_rot = eccentricity_timeseries_rotating(
        prop_r, epoch, ROSEN_N_CYCLES * T_CYCLE, DT_FINE, omega_dot,
    )

    # Rotating-frame mean — only correct ey; ex stays 0
    ey_avg = float(np.mean(ey_rot))
    d_ey   = float(ey_seed - ey_avg)    # ey_seed = e_f (target)
    ey_c  += d_ey
    residual = abs(d_ey)

    rosen_history_1d.append({'iter': k+1, 'ey_c': ey_c, 'ey_avg': ey_avg,
                              'residual': residual})
    print(f"  Rosengren iter {k+1:2d}: ey_c={ey_c:.6e}  ey_avg={ey_avg:.6e}"
          f"  |d_ey|={residual:.2e}  ({time.time()-t_iter:.0f}s)", flush=True)

    if residual < 1e-9:
        print(f"  [OK] Converged after {k+1} iterations.", flush=True)
        break

t_rosen = time.time() - t_rosen_start

# Final osculating elements: omega = 90 deg, e = ey_c
ex_opt  = 0.0
ey_opt  = ey_c
e_opt   = float(ey_c)
aop_opt = 90.0

print(f"\n  Rosengren wall time : {t_rosen/60:.1f} min")
print(f"  Optimised e         : {e_opt:.4e}")
print(f"  Optimised omega     : {aop_opt:.3f} deg")
print(f"  Paper osc e*        : {TARGET_OSC_E:.3e}")
print(f"  Delta e             : {e_opt - TARGET_OSC_E:+.3e}\n", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# 5 — TABLE 2.1 VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
print("═" * 62)
print("  TABLE 2.1 REPLICATION — Near-Equatorial Frozen Repeat Orbit")
print("  i = 10°,  k:q = 59:4,  EGM96 70×70 + Sun + Moon")
print("═" * 62)
print(f"  {'':20s}  {'Paper':>12s}  {'This run':>12s}  {'Δ':>10s}")
print("─" * 62)
print(f"  {'a* mean [km]':20s}  {TARGET_MEAN_A_KM:>12.2f}  {a_seed/1e3:>12.4f}  "
      f"{a_seed/1e3 - TARGET_MEAN_A_KM:>+10.3f}")
print(f"  {'e* mean':20s}  {TARGET_MEAN_E:>12.3e}  {e_f:>12.3e}  "
      f"{e_f - TARGET_MEAN_E:>+10.3e}")
print(f"  {'a* osc [km]':20s}  {TARGET_OSC_A_KM:>12.2f}  {a_opt/1e3:>12.4f}  "
      f"{a_opt/1e3 - TARGET_OSC_A_KM:>+10.3f}")
print(f"  {'e* osc':20s}  {TARGET_OSC_E:>12.3e}  {e_opt:>12.3e}  "
      f"{e_opt - TARGET_OSC_E:>+10.3e}")
print(f"  {'ω* osc [deg]':20s}  {90.000:>12.3f}  {aop_opt:>12.3f}  "
      f"{aop_opt - 90.0:>+10.4f}")
print(f"  {'i [deg]':20s}  {10.000:>12.3f}  {I_DEG:>12.3f}  {0.0:>+10.4f}")
print("═" * 62, flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# 6 — PHASE-SPACE PLOTS  (paper Fig. 2.2A / 2.2B)
# ══════════════════════════════════════════════════════════════════════════════
print("\n─" * 60)
print("  STEP 4: Eccentricity phase-space propagation (30 repeat cycles)")
print("─" * 60, flush=True)

N_PLOT   = 30
T_PLOT   = N_PLOT * T_CYCLE
DT_PLOT  = 30.0

print("  Propagating before-optimisation (analytical seed) ...", flush=True)
t0 = time.time()
orb_b  = build_orbit(a_seed, e_f, i_rad, raan_rad, aop_rad, m0_rad, epoch, gcrf, MU)
prop_b = build_propagator(orb_b, itrf,
                           gravity_degree=GRAV_DEG, gravity_order=GRAV_ORD,
                           third_body=THIRD_BODY)
_, ex_b, ey_b = eccentricity_timeseries(prop_b, epoch, T_PLOT, DT_PLOT)
print(f"  Done ({time.time()-t0:.0f}s)  n_pts={len(ex_b)}", flush=True)

print("  Propagating after-optimisation (converged osculating) ...", flush=True)
t0 = time.time()
orb_a  = build_orbit(a_opt, e_opt, i_rad, raan_rad,
                      float(np.arctan2(ey_opt, ex_opt)), m0_rad, epoch, gcrf, MU)
prop_a = build_propagator(orb_a, itrf,
                           gravity_degree=GRAV_DEG, gravity_order=GRAV_ORD,
                           third_body=THIRD_BODY)
_, ex_a, ey_a = eccentricity_timeseries(prop_a, epoch, T_PLOT, DT_PLOT)
print(f"  Done ({time.time()-t0:.0f}s)  n_pts={len(ex_a)}", flush=True)

# ── Mean long eccentricity: smooth over one orbital period ────────────────────
# Paper plots "mean long eccentricity" (paper p.6, line 72) — short-period
# oscillations are removed by averaging over one orbital period, leaving only
# the long-period variation that the Rosengren algorithm actually controls.
T_orb_s  = 2 * np.pi / np.sqrt(MU / a_opt**3)   # ~96 min
WIN_PTS  = max(1, int(round(T_orb_s / DT_PLOT))) # ~193 pts at 30s
kernel   = np.ones(WIN_PTS) / WIN_PTS

def mean_long_ecc(ex, ey):
    """Box-average over one orbital period to strip short-period oscillations."""
    ex_m = np.convolve(ex, kernel, mode='valid')
    ey_m = np.convolve(ey, kernel, mode='valid')
    return ex_m, ey_m

ex_b_m, ey_b_m = mean_long_ecc(ex_b, ey_b)
ex_a_m, ey_a_m = mean_long_ecc(ex_a, ey_a)

print(f"\n  Smoothing window : {WIN_PTS} pts = {T_orb_s/60:.1f} min (one orbital period)", flush=True)

# Scatter = sqrt(var(ex) + var(ey)) = ring radius for a circular trace.
# NOT std(distance from centroid) which is ~0 for a perfect circle.
r_b = float(np.sqrt(np.var(ex_b_m) + np.var(ey_b_m)))
r_a = float(np.sqrt(np.var(ex_a_m) + np.var(ey_a_m)))
print(f"  Phase-space scatter before (mean long): {r_b:.3e}  (ring radius)")
print(f"  Phase-space scatter after  (mean long): {r_a:.3e}")
print(f"  Reduction factor                      : {r_b/r_a:.1f}x  (paper reports > 5x)", flush=True)

OUT = ROOT / 'outputs' / 'paper_replication'
OUT.mkdir(parents=True, exist_ok=True)

# Colour by time (repeat cycle index) to match paper's colorbar
n_b = len(ex_b_m);  c_b = np.linspace(0, N_PLOT, n_b)
n_a = len(ex_a_m);  c_a = np.linspace(0, N_PLOT, n_a)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, ex_m, ey_m, c, title in [
    (axes[0], ex_b_m, ey_b_m, c_b,
     f'Before optimisation (Fig. 2.2A)\nanalytical seed  e_f={e_f:.2e}'),
    (axes[1], ex_a_m, ey_a_m, c_a,
     f'After optimisation (Fig. 2.2B)\nRosengren 1-D  e*={e_opt:.2e}'),
]:
    sc = ax.scatter(ex_m, ey_m, c=c, cmap='viridis', s=0.3, alpha=0.6)
    ax.plot(ex_seed, ey_seed, 'r*', ms=10, zorder=5, label=f'Frozen target (0, e_f)')
    ax.set_xlabel('$e_x = e\\cos\\omega$  (mean long)')
    ax.set_ylabel('$e_y = e\\sin\\omega$  (mean long)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label='Repeat cycle', pad=0.01)

fig.suptitle(
    f'Mean Long Eccentricity Phase Space  i=10 deg  k=59:q=4  EGM96 70x70 + Sun/Moon\n'
    f'{N_PLOT} repeat cycles ({N_PLOT * T_CYCLE / 86400:.1f} days)'
    f'  |  SP smoothed over 1 orbital period ({T_orb_s/60:.0f} min)',
    fontsize=10,
)
plt.tight_layout()
out_path = OUT / 'ecc_phase_space_paper_replication.png'
plt.savefig(out_path, dpi=130, bbox_inches='tight')
print(f"\n  Phase-space plot saved → {out_path}", flush=True)

# Broyden convergence plot
fig2, ax2 = plt.subplots(figsize=(6, 3.5))
iters    = [h['iter'] for h in broyden_history]
closures = [abs(h['closure_deg']) for h in broyden_history]
ax2.semilogy(iters, closures, 'o-', color='tomato')
ax2.axhline(BROYDEN_TOL_DEG, ls='--', color='gray', lw=1,
            label=f'Tolerance {BROYDEN_TOL_DEG:.0e}°')
ax2.set_xlabel('Broyden iteration')
ax2.set_ylabel('|Closure error| [deg]')
ax2.set_title('SMA Broyden convergence (paper Fig. 2.7 equivalent)')
ax2.legend()
plt.tight_layout()
out2 = OUT / 'broyden_convergence_paper_replication.png'
plt.savefig(out2, dpi=130, bbox_inches='tight')
print(f"  Broyden plot saved   → {out2}", flush=True)

print("\nDone.", flush=True)
