"""Reference frames, time scales, and physical constants.

Must be imported *after* init_orekit() has been called so the JVM is live.
"""

from org.orekit.bodies  import OneAxisEllipsoid
from org.orekit.frames  import FramesFactory
from org.orekit.time    import TimeScalesFactory
from org.orekit.utils   import Constants, IERSConventions

MU = Constants.WGS84_EARTH_MU
RE = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
F  = Constants.WGS84_EARTH_FLATTENING
J2 = 1.08262668e-3


def get_frames():
    """Return (UTC, GCRF, ITRF, EARTH) ready for propagation."""
    utc   = TimeScalesFactory.getUTC()
    gcrf  = FramesFactory.getGCRF()
    itrf  = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    earth = OneAxisEllipsoid(RE, F, itrf)
    return utc, gcrf, itrf, earth
