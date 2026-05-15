"""
Part 3 and Part 4 demos: velocity, dispersion, and dip inversion.

* :func:`demo_intercept_time` -- Part 3: Coppens & Mari (1995)
  intercept-time velocity inversion.
* :func:`demo_dipole`         -- Part 3: dipole flexural
  dispersion modelling and STC processing (Kimball 1998).
* :func:`demo_dip`            -- Part 4: dip and azimuth
  estimation from a 3D azimuthal arrival synthesis.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from fwap._common import US_PER_FT, logger
from fwap.coherence import STCResult
from fwap.dip import estimate_dip, synthesize_azimuthal_arrival
from fwap.dispersion import (
    dispersive_stc,
    narrow_band_stc,
    phase_slowness_from_f_k,
    phase_slowness_matrix_pencil,
    shear_slowness_from_dispersion,
)
from fwap.plotting import save_figure as _savefig
from fwap.plotting import wiggle_plot as _wiggle
from fwap.synthetic import (
    ArrayGeometry,
    Mode,
    dipole_flexural_dispersion,
    synthesize_gather,
)
from fwap.tomography import (
    assemble_observations_from_picks,
    solve_intercept_time,
)


def demo_intercept_time(figdir: str = "figures", show: bool = False) -> None:
    import matplotlib.pyplot as plt

    logger.info("=== Demo: Intercept-time inversion (Coppens & Mari, 1995) ===")
    rng = np.random.default_rng(1)
    n_depth = 60
    dz = 0.1524
    n_rec = 8
    tr_offset = 3.0
    dr = 0.1524
    s_bg = 1.0 / 4500.0
    z = np.arange(n_depth) * dz
    offsets = tr_offset + np.arange(n_rec) * dr
    zc = z.mean()
    delay = 2.0e-5 * np.exp(-0.5 * ((z - zc) / (4 * dz)) ** 2)  # 20 us peak
    tt = np.zeros((n_depth, n_rec))
    for j, zs in enumerate(z):
        for k, x in enumerate(offsets):
            z_rec = zs + x
            idx = int(np.clip(round((z_rec - z[0]) / dz), 0, n_depth - 1))
            tt[j, k] = s_bg * x + delay[j] + delay[idx]
    tt += rng.normal(scale=3.0e-6, size=tt.shape)

    packed = assemble_observations_from_picks(z, offsets, tt)
    travel_times, off_vec, src_idx, rec_idx, n_d, depth_axis = packed

    # Midpoint method: offset assigned to a single midpoint cell.
    r_mp = solve_intercept_time(
        travel_times,
        off_vec,
        src_idx,
        rec_idx,
        n_d,
        depth_axis=depth_axis,
        mean_delay_zero=True,
        smooth_s=5.0e3,
        smooth_src=1.0e3,
        smooth_rec=1.0e3,
        delay_l2=1.0e2,
        method="midpoint",
    )

    # Segmented method: offset split across every traversed cell.
    src_depth = depth_axis[src_idx]
    rec_depth = depth_axis[rec_idx]
    r_seg = solve_intercept_time(
        travel_times,
        off_vec,
        src_depth,
        rec_depth,
        n_d,
        depth_axis=depth_axis,
        mean_delay_zero=True,
        smooth_s=5.0e3,
        smooth_src=1.0e3,
        smooth_rec=1.0e3,
        delay_l2=1.0e2,
        method="segmented",
    )

    logger.info(
        "  midpoint  RMS %.2f us   mean s %.2f us/ft (truth %.2f)",
        r_mp.rms_residual * 1e6,
        np.mean(r_mp.slowness) / US_PER_FT,
        s_bg / US_PER_FT,
    )
    logger.info(
        "  segmented RMS %.2f us   mean s %.2f us/ft",
        r_seg.rms_residual * 1e6,
        np.mean(r_seg.slowness) / US_PER_FT,
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)

    ax = axes[0]
    ax.plot(
        r_mp.slowness / US_PER_FT,
        depth_axis,
        "b-",
        alpha=0.5,
        label="Inverted (midpoint)",
    )
    ax.fill_betweenx(
        depth_axis,
        (r_mp.slowness - r_mp.sigma_slowness) / US_PER_FT,
        (r_mp.slowness + r_mp.sigma_slowness) / US_PER_FT,
        alpha=0.15,
        color="b",
    )
    ax.plot(r_seg.slowness / US_PER_FT, depth_axis, "g-", label="Inverted (segmented)")
    ax.fill_betweenx(
        depth_axis,
        (r_seg.slowness - r_seg.sigma_slowness) / US_PER_FT,
        (r_seg.slowness + r_seg.sigma_slowness) / US_PER_FT,
        alpha=0.15,
        color="g",
    )
    ax.plot(np.full(n_d, s_bg) / US_PER_FT, depth_axis, "k--", label="Truth")
    ax.invert_yaxis()
    ax.set_xlabel("Slowness (us/ft)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Virgin formation slowness\n(shaded = +/- sigma)")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(
        r_mp.delay_src * 1e6, depth_axis, "b-", alpha=0.5, label="Inverted (midpoint)"
    )
    ax.plot(r_seg.delay_src * 1e6, depth_axis, "g-", label="Inverted (segmented)")
    ax.plot(delay * 1e6, depth_axis, "k--", label="Truth")
    ax.invert_yaxis()
    ax.set_xlabel("Source delay (us)")
    ax.set_title("Source-side delay")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[2]
    ax.plot(
        r_mp.delay_rec * 1e6, depth_axis, "b-", alpha=0.5, label="Inverted (midpoint)"
    )
    ax.plot(r_seg.delay_rec * 1e6, depth_axis, "g-", label="Inverted (segmented)")
    ax.plot(delay * 1e6, depth_axis, "k--", label="Truth")
    ax.invert_yaxis()
    ax.set_xlabel("Receiver delay (us)")
    ax.set_title("Receiver-side delay")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.suptitle(
        "Intercept-time inversion -- Coppens & Mari (1995)\n"
        "Segmented tomography design, posterior sigma, L2 prior on delays",
        fontsize=11,
    )
    plt.tight_layout()
    _savefig(fig, figdir, "demo_intercept_time.png", show=show)


def demo_dipole(figdir: str = "figures", show: bool = False) -> None:
    import matplotlib.pyplot as plt

    logger.info("=== Demo: Dipole flexural dispersion ===")
    Vs = 2500.0
    geom = ArrayGeometry(n_rec=8, tr_offset=3.0, dr=0.1524, dt=2.0e-5, n_samples=2048)
    disp = dipole_flexural_dispersion(vs=Vs, a_borehole=0.1)
    mode = Mode(
        name="Flex", slowness=1.0 / Vs, f0=4000.0, amplitude=1.0, dispersion=disp
    )
    data = synthesize_gather(geom, [mode], noise=0.03, seed=7)

    # Narrow-band STC at low frequency: approaches the shear slowness.
    res_lowf = narrow_band_stc(
        data,
        dt=geom.dt,
        offsets=geom.offsets,
        f_lo=500.0,
        f_hi=1500.0,
        slowness_range=(50e-6, 800e-6),
        n_slowness=151,
        window_length=1.5e-3,
        time_step=4,
    )
    # Wide-band STC: high-f dispersion biases the estimate away from Vs.
    res_wide = narrow_band_stc(
        data,
        dt=geom.dt,
        offsets=geom.offsets,
        f_lo=500.0,
        f_hi=10_000.0,
        slowness_range=(50e-6, 800e-6),
        n_slowness=151,
        window_length=1.5e-3,
        time_step=4,
    )

    # Dispersion-corrected STC: unbiased shear slowness across the band.
    def disp_family(s_shear: float) -> Callable[[np.ndarray], np.ndarray]:
        return dipole_flexural_dispersion(vs=1.0 / s_shear, a_borehole=0.1)

    res_disp = dispersive_stc(
        data,
        dt=geom.dt,
        offsets=geom.offsets,
        dispersion_family=disp_family,
        shear_slowness_range=(200e-6, 600e-6),
        n_slowness=81,
        f_range=(500.0, 4000.0),
        window_length=1.5e-3,
        time_step=4,
    )

    def peak_slow(r: STCResult) -> float:
        rho = np.nan_to_num(r.coherence)
        i, _ = np.unravel_index(np.argmax(rho), rho.shape)
        return float(r.slowness[i])

    s_true = 1.0 / Vs
    s_lowf = peak_slow(res_lowf)
    s_wide = peak_slow(res_wide)
    s_disp = peak_slow(res_disp)

    # Two dispersion estimators
    curve_fu = phase_slowness_from_f_k(
        data,
        dt=geom.dt,
        offsets=geom.offsets,
        f_range=(500.0, 8000.0),
        method="frequency_unwrap",
    )
    curve_mp = phase_slowness_matrix_pencil(
        data, dt=geom.dt, offsets=geom.offsets, f_range=(500.0, 8000.0)
    )
    s_avg = shear_slowness_from_dispersion(
        curve_fu, f_lo=1500.0, f_hi=2500.0, quality_threshold=0.3
    )

    logger.info(
        "  True Vs                = %.0f m/s (%.2f us/ft)", Vs, s_true / US_PER_FT
    )
    logger.info(
        "  Narrow-band STC Vs     = %.0f (%.2f us/ft)", 1.0 / s_lowf, s_lowf / US_PER_FT
    )
    logger.info(
        "  Wide-band STC Vs       = %.0f (biased low by high-f dispersion)",
        1.0 / s_wide,
    )
    logger.info("  Dispersive STC Vs      = %.0f (Kimball 1998)", 1.0 / s_disp)
    logger.info("  Low-f asymptote Vs     = %.0f", 1.0 / s_avg)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    _wiggle(
        axes[0, 0],
        data,
        geom.t,
        xmax=geom.t[-1] * 0.5,
        title="Dispersive flexural gather",
    )

    ax = axes[0, 1]
    f_grid = np.linspace(50, 8000, 200)
    ax.plot(f_grid, disp(f_grid) / US_PER_FT, "k--", label="True s(f)")
    ax.plot(
        curve_fu.freq,
        curve_fu.slowness / US_PER_FT,
        "b.-",
        alpha=0.6,
        ms=3,
        label="freq_unwrap",
    )
    ax.plot(
        curve_mp.freq,
        curve_mp.slowness / US_PER_FT,
        "r.",
        alpha=0.5,
        ms=3,
        label="matrix_pencil",
    )
    ax.axhline(
        1.0 / Vs / US_PER_FT,
        color="g",
        ls=":",
        label=f"True shear ({1.0 / Vs / US_PER_FT:.1f} us/ft)",
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase slowness (us/ft)")
    ax.set_ylim(0.8 / Vs / US_PER_FT, 1.5 / Vs / US_PER_FT)
    ax.set_title("Flexural dispersion")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    rho_disp = np.nan_to_num(res_disp.coherence)
    pcm = ax.pcolormesh(
        res_disp.time * 1e3,
        res_disp.slowness / US_PER_FT,
        rho_disp,
        shading="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    plt.colorbar(pcm, ax=ax, label="Coherence")
    ax.axhline(1.0 / Vs / US_PER_FT, color="r", ls="--", label="True shear")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("SHEAR slowness (us/ft)")
    ax.set_title(
        "Dispersive STC (Kimball, 1998)\n-- shear slowness, not phase slowness"
    )
    ax.legend()

    ax = axes[1, 1]
    ax.plot(curve_fu.freq, curve_fu.quality, "b.-", label="freq_unwrap")
    ax.plot(curve_mp.freq, curve_mp.quality, "r.-", label="matrix_pencil")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Fit quality")
    ax.set_ylim(0, 1.05)
    ax.set_title("Phase-fit quality")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(
        "Dipole flexural processing -- Kimball (1998); Ekstroem (1995)\n"
        "Dispersive STC, matrix pencil, frequency-domain unwrap",
        fontsize=11,
    )
    plt.tight_layout()
    _savefig(fig, figdir, "demo_dipole.png", show=show)


def demo_dip(figdir: str = "figures", show: bool = False) -> None:
    import matplotlib.pyplot as plt

    logger.info("=== Demo: Dip / azimuth estimation ===")
    true_dip = np.deg2rad(35.0)
    true_az = np.deg2rad(60.0)
    data, dt, ax_off, az, a, slow = synthesize_azimuthal_arrival(
        n_rec=8,
        n_samples=1024,
        dt=2.0e-5,
        tool_radius=0.08,
        slowness=1.0 / 4000.0,
        dip=true_dip,
        azimuth=true_az,
        f0=8000.0,
        noise=0.02,
        seed=3,
    )
    # Coarse grid + refinement
    result = estimate_dip(
        data,
        dt=dt,
        axial_offsets=ax_off,
        azimuths=az,
        tool_radius=a,
        slowness=slow,
        dip_range=(0.0, np.deg2rad(60.0)),
        n_dip=31,
        n_az=72,
        refine=True,
    )
    logger.info(
        "  True   dip=%6.2f  az=%6.2f", np.rad2deg(true_dip), np.rad2deg(true_az)
    )
    logger.info(
        "  Recov. dip=%6.2f  az=%6.2f  coh=%.3f  refined=%s",
        np.rad2deg(result.dip),
        np.rad2deg(result.azimuth),
        result.coherence,
        result.refined,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    g = 1.5 / (np.max(np.abs(data)) + 1e-12)
    t_ms = np.arange(data.shape[1]) * dt * 1e3
    ax = axes[0]
    for i, tr in enumerate(data):
        ax.plot(t_ms, tr * g + i, "k", lw=0.7)
    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(-0.7, data.shape[0] - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Receiver (by azimuth)")
    ax.set_title(
        "Azimuthal array -- arrival time is a cosine\n"
        "of receiver azimuth for a dipping bed"
    )

    ax = axes[1]
    pcm = ax.pcolormesh(
        np.rad2deg(result.azimuth_axis),
        np.rad2deg(result.dip_axis),
        result.surface,
        shading="auto",
        cmap="viridis",
    )
    plt.colorbar(pcm, ax=ax, label="Coherence")
    ax.plot(
        np.rad2deg(result.azimuth),
        np.rad2deg(result.dip),
        "ro",
        ms=12,
        mfc="none",
        mew=2,
        label=f"Recovered ({'refined' if result.refined else 'grid'})",
    )
    ax.plot(
        np.rad2deg(true_az), np.rad2deg(true_dip), "w+", ms=14, mew=2, label="Truth"
    )
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Dip (deg)")
    ax.set_title("Coherence over (dip, azimuth)")
    ax.legend()
    plt.suptitle(
        "Dip / azimuth from azimuthal acoustic array\n"
        "Mari, Coppens, Gavin & Wicquart (1994), Part 4",
        fontsize=11,
    )
    plt.tight_layout()
    _savefig(fig, figdir, "demo_dip.png", show=show)
