"""J3 numerical propagator (zonal harmonics up to degree 3, order 0).

Must be imported *after* init_orekit() has been called so the JVM is live.
"""

import numpy as np

from orekit import JArray_double

from org.hipparchus.ode.nonstiff          import DormandPrince853Integrator
from org.orekit.forces.gravity            import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity.potential  import GravityFieldFactory
from org.orekit.orbits                    import KeplerianOrbit, OrbitType, PositionAngleType
from org.orekit.propagation               import SpacecraftState
from org.orekit.propagation.numerical     import NumericalPropagator


def build_orbit(a, e, inc, raan, aop, m0, epoch, gcrf, mu):
    """Return a KeplerianOrbit from classical elements (all SI / radians)."""
    return KeplerianOrbit(
        float(a), float(e), float(inc), float(aop), float(raan), float(m0),
        PositionAngleType.MEAN,
        gcrf, epoch, mu,
    )


def build_propagator(orbit, itrf, pos_tol=1.0, min_step=0.001, max_step=300.0,
                     gravity_degree=3):
    """Build a zonal-only NumericalPropagator with a Dormand-Prince 8(5,3) integrator.

    Parameters
    ----------
    gravity_degree : int  Maximum zonal degree (order=0).  2 → J2 only, 3 → J2+J3.
    """
    gravity_provider = GravityFieldFactory.getNormalizedProvider(gravity_degree, 0)
    gravity_model    = HolmesFeatherstoneAttractionModel(itrf, gravity_provider)

    tols = NumericalPropagator.tolerances(pos_tol, orbit, OrbitType.CARTESIAN)
    integrator = DormandPrince853Integrator(
        min_step, max_step,
        JArray_double.cast_(tols[0]),
        JArray_double.cast_(tols[1]),
    )

    propagator = NumericalPropagator(integrator)
    propagator.setOrbitType(OrbitType.CARTESIAN)
    propagator.addForceModel(gravity_model)
    propagator.setInitialState(SpacecraftState(orbit))
    return propagator


def propagate(propagator, epoch, t_end, dt, gcrf, itrf, earth):
    """Step the propagator and return time-series arrays.

    Returns
    -------
    t_arr    : (N,)  elapsed seconds from epoch
    xyz_eci  : (N,3) ECI position in metres
    lat_arr  : (N,)  geodetic latitude  [deg]
    lon_arr  : (N,)  geodetic longitude [deg]
    alt_arr  : (N,)  geodetic altitude  [km]
    """
    t_arr   = np.arange(0.0, t_end + dt, dt)
    n_pts   = len(t_arr)
    xyz_eci = np.empty((n_pts, 3))
    lat_arr = np.empty(n_pts)
    lon_arr = np.empty(n_pts)
    alt_arr = np.empty(n_pts)

    for n, t in enumerate(t_arr):
        state = propagator.propagate(epoch.shiftedBy(float(t)))

        p_eci = state.getPVCoordinates(gcrf).getPosition()
        p_ecf = state.getPVCoordinates(itrf).getPosition()

        xyz_eci[n] = [p_eci.getX(), p_eci.getY(), p_eci.getZ()]

        gp = earth.transform(p_ecf, itrf, state.getDate())
        lat_arr[n] = np.degrees(gp.getLatitude())
        lon_arr[n] = np.degrees(gp.getLongitude())
        alt_arr[n] = gp.getAltitude() / 1e3

    return t_arr, xyz_eci, lat_arr, lon_arr, alt_arr
