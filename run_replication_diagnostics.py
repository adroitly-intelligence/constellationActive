"""
Diagnostic tests to determine the cause of the 8% e* residual in paper replication.

Three hypotheses:
  H1 -- Orekit vs STK numerical integration differences (irreducible)
  H2 -- Wrong omega_dot used for rotating-frame detrending (J2-only != true EGM96)
  H3 -- Wrong target e_f (Coffey-Deprit J2+J3 != true EGM96 frozen eccentricity)

Tests (run in order B -> C -> A, since A derives Δomega_dot from C's rolling mean):
  B  -- Full Broyden + 1-D Rosengren with J2+J3 only            (isolates H3)
  C  -- Propagate converged orbit 30 cycles; track rotating-frame mean drift
         and estimate true EGM96 frozen point                     (ground-truth)
  A  -- Extract numerical Δomega_dot from rolling mean of Test C  (isolates H2)
         Re-run Rosengren with corrected omega_dot to quantify H2 impact
"""

import pathlib, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

OUT = ROOT / 'outputs' / 'paper_replication'
OUT.mkdir(parents=True, exist_ok=True)

_log_path = OUT / 'diagnostics_run.log'
_log = open(_log_path, 'w', encoding='utf-8')

class _Tee:
    def __init__(self, raw, log):
        self.raw, self.log = raw, log
    def write(self, s):
        self.log.write(s)
        self.raw.write(s.encode('ascii', errors='replace').decode('ascii'))
    def flush(self): self.log.flush(); self.raw.flush()
    @property
    def encoding(self): return 'utf-8'

sys.stdout = _Tee(sys.__stdout__, _log)
sys.stderr = _Tee(sys.__stderr__, _log)

print("Initialising Orekit ...", flush=True)
from src.core.setup import init_orekit
init_orekit(str(ROOT / 'data' / 'orekit-data-main.zip'))
print("Orekit ready.\n", flush=True)

from org.orekit.time import AbsoluteDate
from src.core.frames       import get_frames, MU
from src.core.frozen_orbit import (
    repeat_ground_track_sma, frozen_eccentricity,
    perigee_precession_rate, nodal_regression_rate,
    OMEGA_EARTH, T_SIDEREAL,
)
from src.models.j3_propagator  import build_orbit, build_propagator, propagate
from src.algorithms.rosengren  import (
    eccentricity_timeseries, eccentricity_timeseries_rotating,
)
from src.utils.analysis import ascending_node_longitudes

utc, gcrf, itrf, earth = get_frames()

# ── Known values from paper replication run ───────────────────────────────────
K_ORBITS = 59;  Q_DAYS = 4.0;  I_DEG = 10.0
i_rad    = np.radians(I_DEG)
aop_rad  = np.radians(90.0)
raan_rad = np.radians(0.0)
m0_rad   = np.radians(0.0)
epoch    = AbsoluteDate(2021, 1, 1, 0, 0, 0.0, utc)
T_CYCLE  = Q_DAYS * T_SIDEREAL

a_seed    = repeat_ground_track_sma(K_ORBITS, Q_DAYS, i_rad, e=0.001)
e_f       = frozen_eccentricity(a_seed, i_rad)
ey_seed   = float(e_f)

omega_dot_J2_seed = perigee_precession_rate(a_seed, e_f, i_rad)   # rad/s  (J2 at seed SMA)

A_OPT  = 6934611.95      # m   (converged from EGM96 run)
E_OPT  = 1.541425e-3     # e*  (converged from EGM96 run)
omega_dot_J2_aopt = perigee_precession_rate(A_OPT, E_OPT, i_rad)  # J2 at optimised SMA

PAPER_E = 1.431e-3
DT_FINE   = 30.0
DT_COARSE = 60.0
N_CYCLES  = 5
T_ORB     = 2 * np.pi / np.sqrt(MU / A_OPT**3)   # approximate orbital period [s]
WIN_PTS   = max(1, int(round(T_ORB / DT_FINE)))    # pts per orbital period for smoothing

print(f"Seed SMA           : {a_seed/1e3:.4f} km")
print(f"Coffey-Deprit e_f  : {e_f:.4e}")
print(f"omega_dot J2 (seed): {np.degrees(omega_dot_J2_seed)*86400:.4f} deg/day")
print(f"omega_dot J2 (aopt): {np.degrees(omega_dot_J2_aopt)*86400:.4f} deg/day")
print(f"Converged a* (EGM96): {A_OPT/1e3:.4f} km")
print(f"Converged e* (EGM96): {E_OPT:.4e}")
print(f"Paper e*            : {PAPER_E:.4e}")
print(f"Orbital period      : {T_ORB/60:.2f} min  -> smoothing window = {WIN_PTS} pts\n", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# TEST B -- J2+J3 only pipeline  (isolates H3: model-order effect on e*)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  TEST B: Full Broyden + 1-D Rosengren with J2+J3 only")
print("  Question: does removing higher harmonics bring e* closer to paper?")
print("=" * 65, flush=True)

def measure_closure_j3(a):
    orb  = build_orbit(a, e_f, i_rad, raan_rad, aop_rad, m0_rad, epoch, gcrf, MU)
    prop = build_propagator(orb, itrf, gravity_degree=3, gravity_order=0, third_body=False)
    _, _, lat, lon, _ = propagate(prop, epoch, T_CYCLE, DT_COARSE, gcrf, itrf, earth)
    an_lons, _ = ascending_node_longitudes(lat, lon)
    if len(an_lons) < 2: return 0.0
    return float((an_lons[-1] - an_lons[0] + 180.0) % 360.0 - 180.0)

print("  Broyden SMA (J2+J3) ...")
a_j3 = float(a_seed)
f0   = measure_closure_j3(a_j3)
fp   = measure_closure_j3(a_j3 + 100.0)
J    = (fp - f0) / 100.0
print(f"    seed: a={a_j3/1e3:.4f} km  closure={f0:+.5f} deg  J={J:.4e}", flush=True)

for k in range(15):
    da    = -f0 / J
    a_new = a_j3 + da
    f_new = measure_closure_j3(a_new)
    print(f"    iter {k+1}: a={a_new/1e3:.4f} km  closure={f_new:+.5f} deg  da={da:+.2f} m", flush=True)
    if abs(f_new) < 1e-3:
        print(f"    Converged.", flush=True)
        a_j3 = a_new; break
    J = J + (f_new - f0 - J*(a_new - a_j3)) / (a_new - a_j3)**2
    a_j3, f0 = a_new, f_new

omega_dot_j3 = perigee_precession_rate(a_j3, e_f, i_rad)
print(f"\n  Optimised SMA (J2+J3) : {a_j3/1e3:.4f} km")

print(f"\n  Rosengren 1-D (J2+J3) ...")
ex_c = 0.0;  ey_c = float(e_f)
for k in range(50):
    orb_r  = build_orbit(a_j3, ey_c, i_rad, raan_rad, aop_rad, m0_rad, epoch, gcrf, MU)
    prop_r = build_propagator(orb_r, itrf, gravity_degree=3, gravity_order=0, third_body=False)
    _, ex_rot, ey_rot = eccentricity_timeseries_rotating(
        prop_r, epoch, N_CYCLES * T_CYCLE, DT_FINE, omega_dot_j3,
    )
    ey_avg = float(np.mean(ey_rot))
    d_ey   = float(ey_seed - ey_avg)
    ey_c  += d_ey
    print(f"    iter {k+1:2d}: ey_c={ey_c:.6e}  ey_avg={ey_avg:.6e}  |d|={abs(d_ey):.2e}", flush=True)
    if abs(d_ey) < 1e-9:
        print(f"    Converged after {k+1} iterations.", flush=True)
        break

e_j3 = float(ey_c)
print(f"\n  e* (J2+J3)   : {e_j3:.4e}")
print(f"  e* (EGM96)   : {E_OPT:.4e}")
print(f"  Paper e*     : {PAPER_E:.4e}")
print(f"  J2+J3 delta from paper  : {e_j3 - PAPER_E:+.3e}  ({(e_j3/PAPER_E - 1)*100:+.1f}%)")
print(f"  EGM96 delta from paper  : {E_OPT - PAPER_E:+.3e}  ({(E_OPT/PAPER_E - 1)*100:+.1f}%)")

if e_j3 < E_OPT:
    print(f"\n  RESULT: Higher harmonics (EGM96 vs J2+J3) ADD {E_OPT - e_j3:.3e} to e*")
    print(f"          This accounts for {(E_OPT - e_j3) / (E_OPT - PAPER_E) * 100:.1f}% of the EGM96 vs paper gap")
    print(f"          Remainder ({(e_j3 - PAPER_E) / (E_OPT - PAPER_E) * 100:.1f}%) exists even in J2+J3 model")
else:
    print(f"\n  RESULT: Higher harmonics reduce e* by {e_j3 - E_OPT:.3e}")
    print(f"          J2+J3 model overestimates e* by {e_j3 - PAPER_E:.3e}")
print(flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# TEST C -- 30-cycle frozen orbit quality + true EGM96 frozen point  (H3)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  TEST C: Frozen orbit quality + true EGM96 frozen point")
print("  Question: is our converged orbit actually frozen? Where is the")
print("  true EGM96 frozen point vs our Coffey-Deprit analytical target?")
print("=" * 65, flush=True)

N_LONG = 30
T_LONG = N_LONG * T_CYCLE

print(f"  Propagating EGM96 converged orbit {N_LONG} cycles ({T_LONG/86400:.1f} days) ...")
t0 = time.time()
orb_c  = build_orbit(A_OPT, E_OPT, i_rad, raan_rad, aop_rad, m0_rad, epoch, gcrf, MU)
prop_c = build_propagator(orb_c, itrf, gravity_degree=70, gravity_order=70, third_body=True)
t_long, ex_long, ey_long = eccentricity_timeseries(prop_c, epoch, T_LONG, DT_FINE)
print(f"  Done ({time.time()-t0:.0f}s)  {len(t_long)} points", flush=True)

# Rotating-frame coordinates (detrended at J2 analytical omega_dot)
cos_rot = np.cos(-omega_dot_J2_seed * t_long)
sin_rot = np.sin(-omega_dot_J2_seed * t_long)
xi_long  =  ex_long * cos_rot - ey_long * sin_rot
eta_long =  ex_long * sin_rot + ey_long * cos_rot

# Rolling mean over one repeat cycle window (WIN_CYCLE points)
WIN_CYCLE = max(1, int(round(T_CYCLE / DT_FINE)))
stride    = WIN_CYCLE // 4
n_pts     = len(t_long)

roll_t   = []
roll_xi  = []
roll_eta = []
for i in range(0, n_pts - WIN_CYCLE, stride):
    roll_t.append(t_long[i + WIN_CYCLE // 2])
    roll_xi.append(np.mean(xi_long[i:i + WIN_CYCLE]))
    roll_eta.append(np.mean(eta_long[i:i + WIN_CYCLE]))

roll_t    = np.array(roll_t)
roll_xi   = np.array(roll_xi)
roll_eta  = np.array(roll_eta)

# Full-window mean = estimate of true EGM96 frozen point in rotating frame
xi_true  = float(np.mean(roll_xi))
eta_true = float(np.mean(roll_eta))
e_true   = float(np.hypot(xi_true, eta_true))

# Drift = std of rolling mean (how much the mean moves over time)
xi_drift  = float(np.std(roll_xi))
eta_drift = float(np.std(roll_eta))

print(f"\n  Analytical frozen target  : (xi, eta) = (0, {e_f:.4e})")
print(f"  True EGM96 rotating mean  : (xi, eta) = ({xi_true:+.4e}, {eta_true:+.4e})")
print(f"  True frozen |e| magnitude : {e_true:.4e}")
print(f"  Coffey-Deprit e_f         : {e_f:.4e}")
print(f"  True - Coffey-Deprit (eta): {eta_true - e_f:+.4e}  ({(eta_true/e_f - 1)*100:+.1f}%)")
print(f"\n  Rolling-mean drift (std):")
print(f"    xi  std : {xi_drift:.3e}")
print(f"    eta std : {eta_drift:.3e}")
print(f"  Drift/e*  : {eta_drift/E_OPT*100:.2f}%")

frozen_quality = "WELL-FROZEN" if eta_drift / E_OPT < 0.05 else "DRIFTING -- NOT well-frozen"
print(f"  Orbit is  : {frozen_quality}")

# H3 verdict: does the offset of the true frozen point from e_f explain the residual?
target_offset = abs(eta_true - e_f)
total_residual = abs(E_OPT - PAPER_E)
h3_pct = target_offset / total_residual * 100 if total_residual > 0 else 0
print(f"\n  H3: true frozen eta = {eta_true:.4e}, Coffey-Deprit target = {e_f:.4e}")
print(f"      offset = {target_offset:.3e} vs residual = {total_residual:.3e}")
print(f"      H3 explains approximately {h3_pct:.1f}% of the EGM96-vs-paper residual")
print(f"  VERDICT: H3 is {'significant' if h3_pct > 20 else 'minor'}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# TEST A -- Δomega_dot from rolling-mean angle drift  (isolates H2)
# The rotating-frame rolling mean slowly rotates at Δomega_dot if the true
# EGM96 omega_dot differs from the analytical J2 value used for detrending.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  TEST A: Numerical delta_omega_dot from rotating-frame drift")
print("  Method: fit slope of arctan2(eta_roll, xi_roll) vs time")
print("=" * 65, flush=True)

# Angle of rolling mean in the rotating frame
theta_roll = np.unwrap(np.arctan2(roll_eta, roll_xi))  # rad
# Fit linear slope = Δomega_dot [rad/s]
coeff_A = np.polyfit(roll_t, theta_roll, 1)
delta_omega_dot = coeff_A[0]   # rad/s  (EGM96 omega_dot - analytical J2 omega_dot)
omega_dot_EGM96 = omega_dot_J2_seed + delta_omega_dot

print(f"  omega_dot analytical J2 (seed SMA): {np.degrees(omega_dot_J2_seed)*86400:+.6f} deg/day")
print(f"  delta_omega_dot (EGM96 - J2)      : {np.degrees(delta_omega_dot)*86400:+.6f} deg/day")
print(f"  omega_dot EGM96 estimate          : {np.degrees(omega_dot_EGM96)*86400:+.6f} deg/day")

# Phase error over N_CYCLES * T_CYCLE
phase_err_deg = np.degrees(delta_omega_dot) * (N_CYCLES * T_CYCLE / 86400)
bias_ey = E_OPT * abs(np.sin(np.radians(phase_err_deg)))
print(f"  Phase error over {N_CYCLES} cycles ({N_CYCLES*T_CYCLE/86400:.1f} days): {phase_err_deg:.4f} deg")
print(f"  Implied ey_avg bias                : {bias_ey:.3e}")
print(f"  Residual to explain                : {total_residual:.3e}")
h2_pct = bias_ey / total_residual * 100 if total_residual > 0 else 0
print(f"  H2 explains approximately          : {h2_pct:.1f}% of the residual")
print(f"  VERDICT: H2 is {'significant' if h2_pct > 20 else 'minor'}", flush=True)

# Re-run Rosengren with corrected omega_dot only if the correction is meaningful
if abs(delta_omega_dot) > 1e-8 and abs(delta_omega_dot) < 1e-4:   # sanity guard
    print(f"\n  Re-running 1-D Rosengren with EGM96 omega_dot = {np.degrees(omega_dot_EGM96)*86400:.4f} deg/day ...")
    ex_c = 0.0;  ey_c = float(e_f)
    for k in range(50):
        ey_c_safe = max(ey_c, 1e-8)   # guard against negative eccentricity
        orb_r  = build_orbit(A_OPT, ey_c_safe, i_rad, raan_rad, aop_rad, m0_rad, epoch, gcrf, MU)
        prop_r = build_propagator(orb_r, itrf, gravity_degree=70, gravity_order=70, third_body=True)
        _, ex_rot, ey_rot = eccentricity_timeseries_rotating(
            prop_r, epoch, N_CYCLES * T_CYCLE, DT_FINE, omega_dot_EGM96,
        )
        ey_avg = float(np.mean(ey_rot))
        d_ey   = float(ey_seed - ey_avg)
        ey_c   = ey_c_safe + d_ey
        print(f"    iter {k+1:2d}: ey_c={ey_c:.6e}  ey_avg={ey_avg:.6e}  |d|={abs(d_ey):.2e}", flush=True)
        if abs(d_ey) < 1e-9:
            print(f"    Converged after {k+1} iterations.", flush=True)
            break
    e_corrected = float(ey_c)
    print(f"\n  e* with corrected omega_dot : {e_corrected:.4e}")
    print(f"  e* with J2 omega_dot        : {E_OPT:.4e}")
    print(f"  Change from H2 correction   : {e_corrected - E_OPT:+.3e}")
else:
    e_corrected = E_OPT
    print(f"  delta_omega_dot too small or suspicious — skipping re-run")
print(flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Plot B: e* by model
ax = axes[0]
labels = ['J2+J3\n(Test B)', 'EGM96 70x70\n(replication)', 'Paper\n(STK EGM96)']
values = [e_j3, E_OPT, PAPER_E]
colors = ['steelblue', 'seagreen', 'tomato']
bars   = ax.bar(labels, [v*1e3 for v in values], color=colors, alpha=0.8, edgecolor='k', lw=0.8)
ax.set_ylabel('e* osculating [x 10^-3]')
ax.set_title('Test B: e* vs gravity model')
ax.axhline(PAPER_E*1e3, color='tomato', ls='--', lw=1, alpha=0.6)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val*1e3:.3f}', ha='center', va='bottom', fontsize=8)

# Plot C: rolling rotating-frame mean drift
ax = axes[1]
ax.plot(roll_t/86400, np.array(roll_xi)*1e4, 'o-', ms=2.5, lw=0.8, color='steelblue', label='xi_rot mean')
ax.plot(roll_t/86400, np.array(roll_eta)*1e4, 's-', ms=2.5, lw=0.8, color='seagreen', label='eta_rot mean')
ax.axhline(0,        color='blue',  ls='--', lw=0.8, alpha=0.5, label='xi target (0)')
ax.axhline(e_f*1e4,  color='green', ls='--', lw=0.8, alpha=0.5, label=f'eta target e_f={e_f:.2e}')
ax.axhline(eta_true*1e4, color='red', ls=':', lw=1.2, label=f'true EGM96 mean ({eta_true:.2e})')
ax.set_xlabel('Time [days]')
ax.set_ylabel('Rotating-frame mean [x 10^-4]')
ax.set_title('Test C: Mean ecc drift in rotating frame')
ax.legend(fontsize=7)

# Plot A: rolling-mean angle in rotating frame (should be linear if Δomega_dot is uniform)
ax = axes[2]
theta_deg = np.degrees(theta_roll)
ax.plot(roll_t/86400, theta_deg - theta_deg[0], 'o-', ms=2.5, lw=0.8, color='darkorange', label='angle(rolling mean)')
fit_deg = np.degrees(coeff_A[0]) * (roll_t - roll_t[0])
ax.plot(roll_t/86400, fit_deg, 'r--', lw=1.2,
        label=f'Fit: d_omega_dot={np.degrees(delta_omega_dot)*86400:+.4f} deg/day')
ax.set_xlabel('Time [days]')
ax.set_ylabel('Angle drift of rotating-frame mean [deg]')
ax.set_title('Test A: Δomega_dot from rolling-mean angle')
ax.legend(fontsize=7)

fig.suptitle('Replication Diagnostics: H1/H2/H3 isolation for e* residual', fontsize=11)
plt.tight_layout()
out_p = OUT / 'diagnostics.png'
plt.savefig(out_p, dpi=130, bbox_inches='tight')
print(f"  Diagnostic plot saved -> {out_p}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  DIAGNOSTIC SUMMARY")
print("=" * 65)
print(f"  Total residual (EGM96 - paper): {E_OPT - PAPER_E:+.3e}  ({(E_OPT/PAPER_E - 1)*100:+.1f}%)")
print()
print(f"  H2 (wrong omega_dot)    : {h2_pct:.1f}%  (delta_omega_dot = {np.degrees(delta_omega_dot)*86400:+.5f} deg/day)")
print(f"  H3 (wrong e_f target)   : {h3_pct:.1f}%  (true eta - e_f = {eta_true - e_f:+.3e})")
print(f"  H1 (Orekit/STK diff)    : {max(0, 100 - h2_pct - h3_pct):.1f}%  (remainder after H2+H3)")
print()
print(f"  J2+J3 e*  = {e_j3:.4e}  (delta from paper: {e_j3-PAPER_E:+.3e})")
print(f"  EGM96 e*  = {E_OPT:.4e}  (delta from paper: {E_OPT-PAPER_E:+.3e})")
print(f"  Paper e*  = {PAPER_E:.4e}")
print("=" * 65)
print("\nDone.", flush=True)
