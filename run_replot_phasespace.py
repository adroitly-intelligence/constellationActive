"""Re-generate the phase-space plot using mean long eccentricity (SP-smoothed).
Re-uses converged values from the completed replication run — no propagation needed
for Broyden/Rosengren, only the two 30-cycle phase-space propagations.
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

_log_path = OUT / 'replot_run.log'
_log = open(_log_path, 'w', encoding='utf-8')

class _Tee:
    def __init__(self, raw, log): self.raw, self.log = raw, log
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
print("Orekit ready.", flush=True)

from org.orekit.time import AbsoluteDate
from src.core.frames       import get_frames, MU
from src.core.frozen_orbit import repeat_ground_track_sma, frozen_eccentricity, T_SIDEREAL
from src.models.j3_propagator  import build_orbit, build_propagator
from src.algorithms.rosengren  import eccentricity_timeseries

utc, gcrf, itrf, earth = get_frames()

# Converged values from the completed replication run
K_ORBITS = 59;  Q_DAYS = 4.0;  I_DEG = 10.0
i_rad = np.radians(I_DEG);  aop_rad = np.radians(90.0)
raan_rad = np.radians(0.0); m0_rad = np.radians(0.0)
epoch    = AbsoluteDate(2021, 1, 1, 0, 0, 0.0, utc)
T_CYCLE  = Q_DAYS * T_SIDEREAL

a_seed  = repeat_ground_track_sma(K_ORBITS, Q_DAYS, i_rad, e=0.001)
e_f     = frozen_eccentricity(a_seed, i_rad)
ex_seed = 0.0;  ey_seed = float(e_f)

A_OPT  = 6934611.95
E_OPT  = 1.541425e-3

GRAV_DEG = 70;  GRAV_ORD = 70;  THIRD_BODY = True
N_PLOT   = 30
T_PLOT   = N_PLOT * T_CYCLE
DT_PLOT  = 30.0

T_orb_s = 2 * np.pi / np.sqrt(MU / A_OPT**3)
WIN_PTS = max(1, int(round(T_orb_s / DT_PLOT)))
kernel  = np.ones(WIN_PTS) / WIN_PTS
print(f"Smoothing window : {WIN_PTS} pts = {T_orb_s/60:.1f} min (one orbital period)", flush=True)

def mean_long_ecc(ex, ey):
    return np.convolve(ex, kernel, mode='valid'), np.convolve(ey, kernel, mode='valid')

# ── Propagate before-optimisation ────────────────────────────────────────────
print(f"\nPropagating before-optimisation ({N_PLOT} cycles) ...", flush=True)
t0 = time.time()
orb_b  = build_orbit(a_seed, e_f, i_rad, raan_rad, aop_rad, m0_rad, epoch, gcrf, MU)
prop_b = build_propagator(orb_b, itrf, gravity_degree=GRAV_DEG,
                           gravity_order=GRAV_ORD, third_body=THIRD_BODY)
_, ex_b, ey_b = eccentricity_timeseries(prop_b, epoch, T_PLOT, DT_PLOT)
print(f"  done ({time.time()-t0:.0f}s)  {len(ex_b)} pts", flush=True)

# ── Propagate after-optimisation ─────────────────────────────────────────────
print(f"Propagating after-optimisation ({N_PLOT} cycles) ...", flush=True)
t0 = time.time()
orb_a  = build_orbit(A_OPT, E_OPT, i_rad, raan_rad, aop_rad, m0_rad, epoch, gcrf, MU)
prop_a = build_propagator(orb_a, itrf, gravity_degree=GRAV_DEG,
                           gravity_order=GRAV_ORD, third_body=THIRD_BODY)
_, ex_a, ey_a = eccentricity_timeseries(prop_a, epoch, T_PLOT, DT_PLOT)
print(f"  done ({time.time()-t0:.0f}s)  {len(ex_a)} pts", flush=True)

# ── Mean long eccentricity ────────────────────────────────────────────────────
ex_b_m, ey_b_m = mean_long_ecc(ex_b, ey_b)
ex_a_m, ey_a_m = mean_long_ecc(ex_a, ey_a)

# Scatter = sqrt(var(ex) + var(ey)) = ring radius for a circular trace.
# NOT std(distance from centroid) which is ~0 for a perfect circle.
r_b = float(np.sqrt(np.var(ex_b_m) + np.var(ey_b_m)))
r_a = float(np.sqrt(np.var(ex_a_m) + np.var(ey_a_m)))
print(f"\nPhase-space scatter before (mean long): {r_b:.3e}  (ring radius ~= sqrt(var_x+var_y))")
print(f"Phase-space scatter after  (mean long): {r_a:.3e}")
print(f"Reduction factor                      : {r_b/r_a:.1f}x  (paper reports > 5x)", flush=True)

# ── Plot ──────────────────────────────────────────────────────────────────────
n_b = len(ex_b_m);  c_b = np.linspace(0, N_PLOT, n_b)
n_a = len(ex_a_m);  c_a = np.linspace(0, N_PLOT, n_a)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, ex_m, ey_m, c, title in [
    (axes[0], ex_b_m, ey_b_m, c_b,
     f'Before optimisation (Fig. 2.2A)\nanalytical seed  e_f={e_f:.2e}'),
    (axes[1], ex_a_m, ey_a_m, c_a,
     f'After optimisation (Fig. 2.2B)\nRosengren 1-D  e*={E_OPT:.2e}'),
]:
    sc = ax.scatter(ex_m, ey_m, c=c, cmap='viridis', s=0.3, alpha=0.6)
    ax.plot(ex_seed, ey_seed, 'r*', ms=10, zorder=5, label='Frozen target (0, e_f)')
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
print(f"\nPlot saved -> {out_path}", flush=True)
print("Done.", flush=True)
