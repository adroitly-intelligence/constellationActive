# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the code

```bash
# Main entry point — joint SMA + eccentricity frozen-repeat orbit design
python design_frozen_repeat_joint.py

# Simpler frozen-repeat design (eccentricity only, no joint SMA correction)
python design_frozen_repeat.py
```

All hyperparameters (orbit type, repeat cycle k/q, inclination, propagator settings, tolerances) and paths are controlled through `config.yaml`. Edit that file rather than touching script arguments.

## Critical Orekit constraint

**`init_orekit()` must be called before any `org.orekit.*` or `org.hipparchus.*` import.** The JVM must be live before Java-backed classes are loaded. All entry-point scripts call `init_orekit(cfg["paths"]["orekit_data"])` at the top before any downstream `src.*` imports that touch Orekit. Violating this ordering causes a `jnius` JVM crash. This also means `src/models/j3_propagator.py` and `src/algorithms/rosengren.py` must only be imported after `init_orekit()` runs.

## Architecture

### Data flow

```
config.yaml
    │
    ▼
analytical_frozen_orbit()       ← src/core/frozen_orbit.py
    │  (Coffey-Deprit J2+J3 theory, repeat-condition SMA)
    ▼
joint_frozen_repeat()           ← src/algorithms/joint_optimizer.py
    │  Loop:
    │    A. SMA correction  — ascending-node closure via Newton step
    │    B. Ecc correction  — Rosengren rotating-frame average
    │
    │  Each step propagates via build_propagator() / propagate()
    │                         ← src/models/j3_propagator.py (Orekit NumericalPropagator)
    ▼
outputs/   (PNG figures + convergence data)
```

### Key modules

- **`src/core/frozen_orbit.py`** — All analytical orbital mechanics theory: Coffey-Deprit frozen eccentricity formula, J2-corrected repeat-ground-track SMA solver, sun-synchronous inclination, `analytical_frozen_orbit()` entry point. No Orekit dependency — pure numpy.

- **`src/core/frames.py`** — GCRF/ITRF frame construction and Earth ellipsoid via Orekit. `get_frames()` returns `(utc, gcrf, itrf, earth)`.

- **`src/models/j3_propagator.py`** — Orekit `NumericalPropagator` with Holmes-Featherstone zonal gravity (degree/order configurable; default J2+J3). `build_orbit()` → `build_propagator()` → `propagate()` pipeline returns numpy time-series arrays.

- **`src/algorithms/rosengren.py`** — Eccentricity-vector iteration in the perigee-rotating frame. The rotating frame (detrended at J2 ω̇ rate) is essential: averaging in the inertial frame biases toward (0,0) over timescales longer than the beat period.

- **`src/algorithms/joint_optimizer.py`** — Joint SMA + eccentricity correction loop. SMA uses analytical Newton step from ascending-node closure error; eccentricity uses Rosengren rotating-frame averaging. Both corrections applied sequentially each iteration.

- **`src/utils/plotting.py`** — All figure generation (ground track, altitude, eccentricity phase space). No computational side effects; call only after all propagations are done.

- **`src/utils/analysis.py`** — Metrics: ascending-node longitude detection, ground-track repeat check, frozen eccentricity check.

### Orbit types (`config.yaml → neqfro.orbit_type`)

| Value | SMA source | Inclination source |
|---|---|---|
| `near_equatorial` | Solved from k:q repeat condition | Fixed input (`inclination_deg`) |
| `sun_synchronous` | Jointly solved with inclination | Derived from sun-sync condition |
| `custom` | Direct input (`sma_m`) | Fixed input (`inclination_deg`) |

### Frozen orbit conventions

- Frozen condition: `aop = 90°` (or 270°) with eccentricity from Coffey-Deprit: `e_f = -(J3/2J2)(RE/a)sin(i)`
- The rotating-frame eccentricity vector `(ξ_rot, η_rot)` detrends at the secular J2 ω̇ rate; the frozen fixed point is stationary at `(0, e_f)` in this frame.
- At near-equatorial inclinations `e_f ≪ δe_sp` (J2 short-period oscillation), so raw osculating averages are noise-dominated — the rotating frame is mandatory.

### Planned extension (convex.md)

`convex.md` documents a Successive Convex Optimisation (SCO) replacement for the Broyden-Rosengren approach, minimising the TV-norm of the mean-filtered eccentricity vector. The LP sub-problem uses `scipy.optimize.linprog` (HiGHS backend). This is planned work, not yet implemented in `src/`.

## Guidelines

- Use `config.yaml` for all hyperparameters and paths.
- Plotting functions must be isolated in `src/utils/plotting.py`.
- `src/core/` and `src/utils/analysis.py` must have no side effects (no I/O, no propagation calls).
- Orekit-dependent imports belong in `src/models/` and `src/algorithms/`; `src/core/frozen_orbit.py` stays pure-Python.
