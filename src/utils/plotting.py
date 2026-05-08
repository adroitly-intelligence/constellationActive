"""Orbit visualisation utilities.  No propagation logic lives here.

Contains:
  plot_orbit_results        — 4-panel overview (existing)
  plot_ecc_phase_space      — eccentricity (ex, ey) phase-space evolution
  plot_spatial_errors       — RTN error time series
"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers 3-D projection


def plot_orbit_results(
    t_arr, xyz_eci, lat_arr, lon_arr, alt_arr,
    a, e, inc_rad, raan_rad, n_orbits, re_km, out_file,
):
    """Four-panel orbit figure: 3-D ECI, ground track, altitude, lat/lon vs time."""
    xyz_km = xyz_eci / 1e3

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f"J3 Orbit Propagation  -  "
        f"a = {a/1e3:.0f} km,  e = {e},  i = {np.degrees(inc_rad):.1f} deg,  "
        f"RAAN = {np.degrees(raan_rad):.1f} deg  ({n_orbits} orbits)",
        fontsize=11,
    )

    # Panel 1 — 3-D orbit in ECI
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.plot(xyz_km[:, 0], xyz_km[:, 1], xyz_km[:, 2],
             lw=0.8, color="steelblue", label="Trajectory")
    ax1.scatter(*xyz_km[0],  color="green", s=40, zorder=5, label="Start")
    ax1.scatter(*xyz_km[-1], color="red",   s=40, zorder=5, label="End")
    u_g, v_g = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
    ax1.plot_surface(
        re_km * np.cos(u_g) * np.sin(v_g),
        re_km * np.sin(u_g) * np.sin(v_g),
        re_km * np.cos(v_g),
        alpha=0.12, color="cyan", linewidth=0,
    )
    ax1.set_xlabel("X [km]")
    ax1.set_ylabel("Y [km]")
    ax1.set_zlabel("Z [km]")
    ax1.set_title("3-D Orbit (GCRF / ECI)")
    ax1.legend(fontsize=8, loc="upper left")

    # Panel 2 — Ground track
    t_h = t_arr / 3600.0
    ax2 = fig.add_subplot(2, 2, 2)
    sc = ax2.scatter(lon_arr, lat_arr, c=t_h, cmap="plasma", s=2, zorder=3)
    ax2.scatter(lon_arr[0], lat_arr[0],
                color="green", s=60, zorder=5, label="Start", marker="^")
    ax2.axhline(0, color="gray", lw=0.5, ls="--")
    fig.colorbar(sc, ax=ax2, label="Time [h]", pad=0.01)
    ax2.set_xlim(-180, 180)
    ax2.set_ylim(-90, 90)
    ax2.set_xlabel("Longitude [deg]")
    ax2.set_ylabel("Latitude [deg]")
    ax2.set_title("Ground Track (ITRF)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3 — Altitude vs time
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(t_h, alt_arr, color="darkorange", lw=0.9)
    ax3.axhline(alt_arr.mean(), color="gray", ls="--", lw=0.8,
                label=f"Mean = {alt_arr.mean():.2f} km")
    ax3.set_xlabel("Time [h]")
    ax3.set_ylabel("Altitude [km]")
    ax3.set_title("Geodetic Altitude vs Time")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4 — Sub-satellite latitude & longitude vs time
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(t_h, lat_arr, label="Latitude",  lw=0.9, color="steelblue")
    ax4.plot(t_h, lon_arr, label="Longitude", lw=0.9, color="tomato", alpha=0.7)
    ax4.set_xlabel("Time [h]")
    ax4.set_ylabel("Angle [deg]")
    ax4.set_title("Sub-satellite Latitude & Longitude vs Time")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = pathlib.Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved -> {out_path}")
    plt.show()


def plot_ground_track(t_arr, lat_arr, lon_arr, k_orbits, t_cycle=None, out_file=None):
    """Standalone 2-D ground-track map on an equirectangular grid.

    If t_cycle is given each repeat cycle is drawn in a distinct colour so
    overlapping tracks directly confirm the ground-track repeat condition.
    Otherwise points are coloured by orbit number.

    Parameters
    ----------
    t_arr    : (N,)   Time [s]
    lat_arr  : (N,)   Geodetic latitude  [deg]
    lon_arr  : (N,)   Geodetic longitude [deg]
    k_orbits : int    Number of orbits in one repeat cycle
    t_cycle  : float  Duration of one repeat cycle [s].  Optional.
    out_file : path   Optional save path.
    """
    # Detect ascending nodes (lat crosses zero going north)
    an_lons, an_idx = [], []
    for i in range(len(lat_arr) - 1):
        if lat_arr[i] < 0.0 and lat_arr[i + 1] >= 0.0:
            frac = -lat_arr[i] / (lat_arr[i + 1] - lat_arr[i])
            dlon = lon_arr[i + 1] - lon_arr[i]
            if dlon > 180:   dlon -= 360
            elif dlon < -180: dlon += 360
            an_lons.append((lon_arr[i] + frac * dlon + 180) % 360 - 180)
            an_idx.append(i)
    an_lons = np.array(an_lons)
    an_idx  = np.array(an_idx, dtype=int)

    # ── Figure and axes ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-90, 91, 30))
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.grid(True, alpha=0.35, linewidth=0.5, zorder=1)
    ax.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.5, zorder=1)
    _kw = {}

    # ── Ground-track scatter ──────────────────────────────────────────────────
    cycle_colours = ["steelblue", "tomato", "seagreen", "darkorange", "purple"]
    if t_cycle is not None:
        cycle_idx = (t_arr / t_cycle).astype(int)
        n_cycles  = int(cycle_idx.max()) + 1
        for c in range(n_cycles):
            mask = cycle_idx == c
            ax.scatter(lon_arr[mask], lat_arr[mask],
                       color=cycle_colours[c % len(cycle_colours)],
                       s=1.0, alpha=0.8, label=f"Cycle {c + 1}",
                       zorder=3, **_kw)
    else:
        n_cycles  = 1
        orbit_num = np.zeros(len(lat_arr), dtype=int)
        for idx in an_idx:
            orbit_num[idx + 1:] += 1
        sc = ax.scatter(lon_arr, lat_arr, c=orbit_num, cmap="tab20",
                        s=1.0, alpha=0.85, vmin=0, vmax=k_orbits,
                        zorder=3, **_kw)
        fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.85, label="Orbit number")

    # ── Ascending nodes (first cycle only) ────────────────────────────────────
    if len(an_lons):
        nodes = an_lons[:k_orbits + 1] if t_cycle is not None else an_lons
        ax.scatter(nodes, np.zeros(len(nodes)),
                   color="black", s=30, zorder=5, marker="^",
                   label=f"Ascending nodes — cycle 1 ({len(nodes)})",
                   **_kw)

    ax.scatter(lon_arr[0], lat_arr[0], color="lime", s=30, zorder=6,
               marker="*", edgecolors="black", linewidths=0.5,
               label="Start", **_kw)

    n_cycles_label = f"{n_cycles} cycles" if n_cycles > 1 else "1 cycle"
    ax.set_title(
        f"Ground Track — {k_orbits}-orbit repeat  ({n_cycles_label})",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="lower right", markerscale=3,
              framealpha=0.8)

    plt.tight_layout()
    if out_file:
        out_path = pathlib.Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved -> {out_path}")
    plt.show()


def _orbit_envelope(t_arr, alt_arr, orbital_period_s):
    """Compute per-orbit min/max altitude envelope.

    Bins the altitude time series into non-overlapping windows of one orbital
    period and returns the centre time, per-orbit minimum, and per-orbit maximum.
    """
    dt_s   = float(t_arr[1] - t_arr[0])
    n_win  = max(int(round(orbital_period_s / dt_s)), 2)
    t_env, alt_lo, alt_hi = [], [], []
    for i in range(0, len(alt_arr) - n_win + 1, n_win):
        chunk = alt_arr[i : i + n_win]
        t_env.append(t_arr[i + n_win // 2] / 3600.0)
        alt_lo.append(chunk.min())
        alt_hi.append(chunk.max())
    return np.array(t_env), np.array(alt_lo), np.array(alt_hi)


def plot_altitude_frozen(t_arr, alt_frozen, alt_nonfrozen=None,
                         label_frozen="Frozen orbit",
                         label_nonfrozen="Circular reference (e = 0)",
                         k_orbits=None, t_cycle=None,
                         orbital_period_s=None,
                         out_file=None):
    """Two-panel altitude figure comparing a frozen orbit against a reference.

    Top panel — absolute per-orbit altitude envelope (perigee / apogee lines).
    Bottom panel — per-orbit altitude amplitude deviation from the initial value,
        defined as  Δamp(t) = (apogee(t) − perigee(t))/2 − (apogee(0) − perigee(0))/2.
        Both series start at zero; the frozen orbit stays near zero while the
        circular reference oscillates as J3 pumps the eccentricity with beat
        period T_beat = 2π / |ω̇|.  This panel directly answers whether the
        frozen design keeps the eccentricity amplitude stable.

    Parameters
    ----------
    t_arr            : (N,)   Time [s]
    alt_frozen       : (N,)   Altitude — frozen orbit [km]
    alt_nonfrozen    : (N,)   Altitude — reference orbit [km].  Optional.
    k_orbits         : int    Orbits per repeat cycle (axis label).
    t_cycle          : float  Repeat cycle duration [s] (draws cycle boundaries).
    orbital_period_s : float  Keplerian orbital period [s] for envelope binning.
    out_file         : path   Optional save path.
    """
    def _cycle_vlines(ax_, t_arr_, t_cycle_):
        if t_cycle_ is not None:
            n = int(t_arr_[-1] / t_cycle_ + 0.5)
            for c in range(1, n):
                ax_.axvline(c * t_cycle_ / 3600, color="gray",
                            lw=0.7, ls="--", alpha=0.45,
                            label="Cycle boundary" if c == 1 else None)

    n_label = f"{int(t_arr[-1] / t_cycle + 0.5)} × " if t_cycle else ""
    k_label = f"{k_orbits}-orbit repeat" if k_orbits else ""

    if orbital_period_s is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        t_f, f_lo, f_hi = _orbit_envelope(t_arr, alt_frozen, orbital_period_s)
        ax1.fill_between(t_f, f_lo, f_hi, alpha=0.20, color="steelblue")
        ax1.plot(t_f, f_lo, color="steelblue", lw=1.3,
                 label=f"{label_frozen}  (perigee)")
        ax1.plot(t_f, f_hi, color="steelblue", lw=1.3, ls="--",
                 label=f"{label_frozen}  (apogee)")

        amp_f  = (f_hi - f_lo) / 2.0
        ax2.plot(t_f, amp_f - amp_f[0], color="steelblue", lw=1.4,
                 label=label_frozen, zorder=3)

        if alt_nonfrozen is not None:
            t_c, c_lo, c_hi = _orbit_envelope(t_arr, alt_nonfrozen, orbital_period_s)
            ax1.fill_between(t_c, c_lo, c_hi, alpha=0.12, color="tomato")
            ax1.plot(t_c, c_lo, color="tomato", lw=1.3,
                     label=f"{label_nonfrozen}  (perigee)")
            ax1.plot(t_c, c_hi, color="tomato", lw=1.3, ls="--",
                     label=f"{label_nonfrozen}  (apogee)")
            amp_c = (c_hi - c_lo) / 2.0
            ax2.plot(t_c, amp_c - amp_c[0], color="tomato", lw=1.4,
                     label=label_nonfrozen, zorder=2)

        ax2.axhline(0, color="gray", lw=0.9, ls="--", alpha=0.7)
        _cycle_vlines(ax1, t_arr, t_cycle)
        _cycle_vlines(ax2, t_arr, t_cycle)

        ax1.set_ylabel("Altitude [km]")
        ax1.set_title(f"Per-orbit Altitude Envelope  ({n_label}{k_label})")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("Time [h]")
        ax2.set_ylabel("Δ Altitude amplitude [km]")
        ax2.set_title("Per-orbit amplitude change from t = 0  "
                      "(frozen ≈ 0,  circular grows with eccentricity beat)")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

    else:
        fig, ax = plt.subplots(figsize=(12, 5))
        t_h = t_arr / 3600.0
        if alt_nonfrozen is not None:
            ax.plot(t_h, alt_nonfrozen, lw=0.7, color="tomato",
                    alpha=0.75, label=label_nonfrozen, zorder=2)
        ax.plot(t_h, alt_frozen, lw=0.8, color="steelblue",
                label=label_frozen, zorder=3)
        _cycle_vlines(ax, t_arr, t_cycle)
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("Altitude [km]")
        ax.set_title(f"Altitude vs Time  ({n_label}{k_label})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if out_file:
        out_path = pathlib.Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved -> {out_path}")
    plt.show()


def plot_ecc_phase_space(ex_arr, ey_arr, history=None, mean_history=None,
                         ecc_target=None, orbit_numbers=None, title=None, out_file=None):
    """Plot eccentricity phase-space (ex vs ey).

    Parameters
    ----------
    ex_arr, ey_arr : (N,)            Eccentricity vector components over the propagation.
    history        : list of (ex, ey)  Osculating initial (ex, ey) per Rosengren iteration.
    mean_history   : list of (ex, ey)  Rotating-frame mean (ex, ey) per iteration —
                                       shows convergence toward the frozen target.
    ecc_target     : (ex_t, ey_t)    Analytical frozen target — drawn as a cross.
    orbit_numbers  : (N,) array-like  If provided, scatter is coloured by orbit number
                                      with a viridis colorbar (matches A34934 style).
    title          : str             Override the default figure title.
    out_file       : str or Path     If given, saves the figure.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    if orbit_numbers is not None:
        sc = ax.scatter(ex_arr, ey_arr, c=orbit_numbers, cmap="viridis",
                        s=1, alpha=0.6, zorder=3)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Orbit number")
    else:
        ax.plot(ex_arr, ey_arr, lw=0.5, color="steelblue", alpha=0.7, label="Phase-space orbit")
        ax.scatter(ex_arr[0], ey_arr[0], color="green", s=50, zorder=5, label="Start")

    if ecc_target is not None:
        ax.scatter(*ecc_target, color="gold", s=120, zorder=8,
                   marker="+", linewidths=2,
                   label=f"Analytical target ({ecc_target[0]:.2e}, {ecc_target[1]:.2e})")

    if history:
        hx = [p[0] for p in history]
        hy = [p[1] for p in history]
        ax.scatter(hx, hy, c=range(len(history)), cmap="Reds", s=40,
                   zorder=6, label="Osculating init (Rosengren)")
        ax.scatter(hx[-1], hy[-1], color="red", s=80, zorder=7,
                   marker="*", label=f"Converged osc. ({hx[-1]:.3e}, {hy[-1]:.3e})")

    if mean_history:
        mx = [p[0] for p in mean_history]
        my = [p[1] for p in mean_history]
        ax.scatter(mx, my, c=range(len(mean_history)), cmap="Blues", s=40,
                   zorder=6, label="Mean e-vector (rotating frame)")
        ax.scatter(mx[-1], my[-1], color="blue", s=80, zorder=7,
                   marker="*", label=f"Converged mean ({mx[-1]:.3e}, {my[-1]:.3e})")

    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("EX = e x cosine ( Arg Periapsis )")
    ax.set_ylabel("EY = e x sine ( Arg Periapsis )")
    ax.set_title(title or "Phase Space of Eccentricity Vector")
    if orbit_numbers is None:
        ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if out_file:
        out_path = pathlib.Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved -> {out_path}")
    plt.show()


def plot_sma_convergence(history, a_analytical, tol_deg=0.01, out_file=None):
    """Two-panel SMA optimisation convergence plot.

    Top panel   — SMA [m] vs evaluation index; horizontal reference at analytical value.
    Bottom panel — |closure error| [deg] vs evaluation index (log scale);
                   horizontal line at tolerance.

    Parameters
    ----------
    history      : list of dict  [{a, closure_deg, da}] from find_repeat_sma.
    a_analytical : float         Analytical SMA [m] (reference line).
    tol_deg      : float         Convergence tolerance drawn on bottom panel.
    out_file     : path          Optional save path.
    """
    iters    = list(range(len(history)))
    smas     = [h['a']           for h in history]
    closures = [abs(h['closure_deg']) for h in history]

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    ax1.plot(iters, [s - a_analytical for s in smas], 'o-', color='steelblue')
    ax1.axhline(0, color='gray', lw=0.8, ls='--', label=f'Analytical ({a_analytical:.1f} m)')
    ax1.set_ylabel('SMA correction  Δa [m]')
    ax1.set_title('SMA Optimisation Convergence')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(iters, closures, 's-', color='tomato')
    ax2.axhline(tol_deg, color='gray', lw=0.8, ls='--',
                label=f'Tolerance ({tol_deg} deg)')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('|Closure error|  [deg]')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    if out_file:
        out_path = pathlib.Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved -> {out_path}")
    plt.show()


def plot_an_drift(an_lons_before, an_lons_after, out_file=None):
    """Ascending-node longitude deviation — before vs after SMA optimisation.

    Both series are plotted as deviation from their first node longitude so
    the Y-axis shows cumulative drift over the repeat cycle.  A perfect
    repeat orbit ends at zero drift.

    Parameters
    ----------
    an_lons_before : (M,)  Ascending-node longitudes [deg] before optimisation.
    an_lons_after  : (M,)  Ascending-node longitudes [deg] after optimisation.
    out_file       : path  Optional save path.
    """
    def _unwrap_dev(lons):
        unwrapped = np.unwrap(np.radians(lons)) * 180.0 / np.pi
        return unwrapped - unwrapped[0]

    dev_before = _unwrap_dev(an_lons_before)
    dev_after  = _unwrap_dev(an_lons_after)

    orbs_b = np.arange(len(dev_before))
    orbs_a = np.arange(len(dev_after))

    _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(orbs_b, dev_before, 'o-', ms=4, color='tomato',
            label=f'Before opt  (closure = {dev_before[-1]:+.4f}°)')
    ax.plot(orbs_a, dev_after,  's-', ms=4, color='steelblue',
            label=f'After opt   (closure = {dev_after[-1]:+.4f}°)')
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel('Orbit number')
    ax.set_ylabel('Ascending-node longitude deviation  [deg]')
    ax.set_title('Ground-Track Closure: Before vs After SMA Optimisation')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if out_file:
        out_path = pathlib.Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved -> {out_path}")
    plt.show()


def plot_spatial_errors(t_arr, eR, eT, eN, out_file=None):
    """Plot RTN spatial error time series.

    Parameters
    ----------
    t_arr      : (N,)  Time array [s].
    eR, eT, eN : (N,)  Radial, transverse, normal errors [m].
    out_file   : str or Path  If given, saves the figure.
    """
    t_h = t_arr / 3600.0

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    labels  = ["Radial",    "Transverse", "Normal"]
    colors  = ["steelblue", "darkorange", "seagreen"]
    arrays  = [eR, eT, eN]

    for ax, label, color, arr in zip(axes, labels, colors, arrays):
        ax.plot(t_h, arr, lw=0.7, color=color)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_ylabel(f"{label} error [m]")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{label}  (mean={np.mean(arr):+.1f} m, STD={np.std(arr):.1f} m)")

    axes[-1].set_xlabel("Time [h]")
    fig.suptitle("Spatial Errors (RTN Frame)", fontsize=11)

    plt.tight_layout()
    if out_file:
        out_path = pathlib.Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved -> {out_path}")
    plt.show()
