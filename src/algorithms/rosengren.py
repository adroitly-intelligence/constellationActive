"""Rosengren eccentricity-vector iteration.

Finds the osculating initial (ex_c, ey_c) such that the time-average of the
eccentricity vector in the PERIGEE-ROTATING frame equals the analytical frozen
target (0, e_f).

Why the rotating frame matters
-------------------------------
The secular J2 perigee precession (rate omega_dot) causes the mean-element
eccentricity vector to rotate in the inertial (ex, ey) frame with period
T_beat = 2π / omega_dot.  Averaging (ex, ey) in the inertial frame over any
window longer than ~half a beat period gives a mean biased toward (0, 0), not
(0, e_f).  In the rotating frame the frozen fixed point is stationary — the
average converges to (0, e_f) for any window of a few repeat cycles.

Correction rule per iteration (in rotating frame, applied to inertial init):
    (ex_c, ey_c) += (ex_target_rot - ex_rot_avg, ey_target_rot - ey_rot_avg)

At t = 0 the rotating frame coincides with the inertial frame, so the
correction to the inertial initial condition equals the rotating-frame error.

Reference: Section IV-B of A34934.
"""

import numpy as np


def eccentricity_timeseries(propagator, epoch, t_end, dt):
    """Propagate and return (t, ex, ey) in the inertial frame.

    Used for phase-space visualisation (inertial trajectory shows the circle).
    """
    from jpype import JImplements, JOverride
    from org.orekit.propagation.sampling import OrekitFixedStepHandler
    from org.orekit.orbits import KeplerianOrbit

    t_list  = []
    ex_list = []
    ey_list = []
    t0_box  = [None]

    @JImplements(OrekitFixedStepHandler)
    class _Handler:
        @JOverride
        def init(self, s0, _t, _step):
            t0_box[0] = s0.getDate()

        @JOverride
        def handleStep(self, state):
            kep   = KeplerianOrbit(state.getOrbit())
            e     = float(kep.getE())
            omega = float(kep.getPerigeeArgument())
            t_list.append(float(state.getDate().durationFrom(t0_box[0])))
            ex_list.append(e * np.cos(omega))
            ey_list.append(e * np.sin(omega))

        @JOverride
        def finish(self, _):
            pass

    propagator.getMultiplexer().add(float(dt), _Handler())
    propagator.propagate(epoch.shiftedBy(float(t_end)))

    return (np.array(t_list),
            np.array(ex_list),
            np.array(ey_list))


def eccentricity_timeseries_rotating(propagator, epoch, t_end, dt, omega_dot):
    """Propagate and return (t, ex_rot, ey_rot) in the perigee-rotating frame.

    In the frame rotating at omega_dot (secular J2 perigee precession):
        ex_rot(t) = e(t) * cos(omega_osc(t) - omega_dot * t)
        ey_rot(t) = e(t) * sin(omega_osc(t) - omega_dot * t)

    The frozen orbit's eccentricity vector is stationary at (0, e_f) in this
    frame, so its time-average equals (0, e_f) for any averaging window.
    """
    from jpype import JImplements, JOverride
    from org.orekit.propagation.sampling import OrekitFixedStepHandler
    from org.orekit.orbits import KeplerianOrbit

    _odot   = float(omega_dot)
    t_list  = []
    ex_list = []
    ey_list = []
    t0_box  = [None]

    @JImplements(OrekitFixedStepHandler)
    class _Handler:
        @JOverride
        def init(self, s0, _t, _step):
            t0_box[0] = s0.getDate()

        @JOverride
        def handleStep(self, state):
            t     = float(state.getDate().durationFrom(t0_box[0]))
            kep   = KeplerianOrbit(state.getOrbit())
            e     = float(kep.getE())
            omega = float(kep.getPerigeeArgument()) - _odot * t
            t_list.append(t)
            ex_list.append(e * np.cos(omega))
            ey_list.append(e * np.sin(omega))

        @JOverride
        def finish(self, _):
            pass

    propagator.getMultiplexer().add(float(dt), _Handler())
    propagator.propagate(epoch.shiftedBy(float(t_end)))

    return (np.array(t_list),
            np.array(ex_list),
            np.array(ey_list))


def find_frozen_eccentricity(
    propagator_factory, epoch, t_cycle, dt, gcrf,
    ex_target, ey_target,
    omega_dot,
    n_cycles=5, n_iter=50, tol=1e-9, verbose=True,
):
    """Return osculating (ex_c, ey_c) whose rotating-frame mean = (ex_target, ey_target).

    Parameters
    ----------
    propagator_factory : callable  f(ex_c, ey_c) -> NumericalPropagator
    epoch              : AbsoluteDate
    t_cycle            : float         One repeat-cycle duration [s].
    dt                 : float         Sampling interval [s].
    gcrf               : Frame         Kept for API compatibility.
    ex_target          : float         Target ex in rotating frame = e_f * cos(aop_f).
    ey_target          : float         Target ey in rotating frame = e_f * sin(aop_f).
    omega_dot          : float         Secular J2 perigee precession rate [rad/s].
    n_cycles           : int           Repeat cycles per iteration.  3–5 is sufficient;
                                       the rotating frame removes the beat-period constraint.
    n_iter             : int           Maximum iterations.
    tol                : float         Convergence on |(d_ex, d_ey)|.
    verbose            : bool

    Returns
    -------
    ex_c, ey_c : float  Converged inertial-frame initial osculating eccentricity vector.
    history    : list   [(ex_c, ey_c)] after each iteration (for phase-space plot).
    """
    t_total  = n_cycles * t_cycle
    ex_c, ey_c = ex_target, ey_target   # seed at the analytical mean target
    history      = []
    mean_history = []

    for k in range(n_iter):
        if verbose:
            print(f"\nRosengren iteration {k+1}/{n_iter} ...")
        prop = propagator_factory(ex_c, ey_c)
        _, ex_rot, ey_rot = eccentricity_timeseries_rotating(
            prop, epoch, t_total, dt, omega_dot,
        )

        ex_avg = float(np.mean(ex_rot))
        ey_avg = float(np.mean(ey_rot))
        mean_history.append((ex_avg, ey_avg))

        # Correct toward target (rotating-frame correction = inertial correction at t=0)
        d_ex = ex_target - ex_avg
        d_ey = ey_target - ey_avg
        ex_c += d_ex
        ey_c += d_ey
        history.append((ex_c, ey_c))

        if verbose:
            e_mag = np.hypot(ex_c, ey_c)
            omega = np.degrees(np.arctan2(ey_c, ex_c))
            print(f"  correction |d| = {np.hypot(d_ex, d_ey):.2e}  |  "
                  f"e = {e_mag:.6e},  omega = {omega:.2f} deg,  "
                  f"(ex, ey) = ({ex_c:.4e}, {ey_c:.4e})")

        if np.hypot(d_ex, d_ey) < tol:
            if verbose:
                print(f"  Converged after {k+1} iterations.")
            break

    return ex_c, ey_c, history, mean_history, (ex_avg, ey_avg)
