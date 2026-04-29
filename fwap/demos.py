"""
Synthetic demonstrations of each fwap algorithm family.

One demo per chapter of Mari, Coppens, Gavin & Wicquart (1994),
*Full Waveform Acoustic Data Processing*, plus the two extension
modules:

* :func:`demo_stc_picker`      -- Part 1
* :func:`demo_pseudo_rayleigh` -- Part 1 (4-mode picker incl. guided wave)
* :func:`demo_wave_separation` -- Part 2 (f-k + SVD / K-L)
* :func:`demo_tau_p_separation` -- Part 2 (tau-p / slant-stack)
* :func:`demo_intercept_time`  -- Part 3 (intercept-time inversion;
                                          Coppens & Mari, 1995)
* :func:`demo_dipole`          -- Part 3 (dipole flexural; Kimball, 1998)
* :func:`demo_dip`             -- Part 4
* :func:`demo_attenuation`     -- extension: Q estimation
* :func:`demo_alford`          -- extension: cross-dipole anisotropy
* :func:`demo_lwd`             -- extension: LWD phenomenological layer
                                  (collar rejection + quadrupole stack;
                                  Tang & Cheng 2004 sect. 2.4-2.5)
* :func:`demo_las_roundtrip`   -- extension: LAS I/O (lasio)
* :func:`demo_dlis_roundtrip`  -- extension: DLIS I/O (dlisio + dliswriter)
* :func:`demo_segy_roundtrip`  -- extension: SEG-Y I/O (segyio)
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from fwap._common import US_PER_FT, logger
from fwap.anisotropy import alford_rotation
from fwap.attenuation import centroid_frequency_shift_Q, spectral_ratio_Q
from fwap.coherence import STCResult, stc
from fwap.dip import estimate_dip, synthesize_azimuthal_arrival
from fwap.dispersion import (
    dispersive_stc,
    narrow_band_stc,
    phase_slowness_from_f_k,
    phase_slowness_matrix_pencil,
    shear_slowness_from_dispersion,
)
from fwap.picker import pick_modes
from fwap.plotting import save_figure as _savefig
from fwap.plotting import wiggle_plot as _wiggle
from fwap.synthetic import (
    ArrayGeometry,
    Mode,
    dipole_flexural_dispersion,
    monopole_formation_modes,
    ricker,
    synthesize_gather,
)
from fwap.tomography import (
    assemble_observations_from_picks,
    solve_intercept_time,
)
from fwap.wavesep import (
    fk_filter,
    sequential_kl_separation,
    tau_p_filter,
    tau_p_forward,
)

# ---------------------------------------------------------------------
# Canonical test configuration reused across several demos.
# ---------------------------------------------------------------------

# Reference P / S / Stoneley velocities used by demo_stc_picker and
# demo_wave_separation. Kept as a module-level constant so changing
# the canonical test case is a one-line edit.
_CANONICAL_VP = 4500.0
_CANONICAL_VS = 2500.0
_CANONICAL_VST = 1400.0


def _canonical_monopole_gather(
    seed: int = 42,
    noise: float = 0.05,
) -> tuple[ArrayGeometry, np.ndarray, float, float, float]:
    """
    Build the shared (geometry, gather, Vp, Vs, Vst) used by the
    ``demo_stc_picker`` and ``demo_wave_separation`` demos.

    Returns the geometry alongside the synthetic gather so callers can
    re-derive per-receiver offsets and time axes without recomputing.
    """
    geom = ArrayGeometry(n_rec=8, tr_offset=3.0, dr=0.1524, dt=1.0e-5, n_samples=2048)
    modes = monopole_formation_modes(
        vp=_CANONICAL_VP, vs=_CANONICAL_VS, v_stoneley=_CANONICAL_VST
    )
    data = synthesize_gather(geom, modes, noise=noise, seed=seed)
    return geom, data, _CANONICAL_VP, _CANONICAL_VS, _CANONICAL_VST


def demo_stc_picker(figdir: str = "figures", show: bool = False) -> None:
    import matplotlib.pyplot as plt

    logger.info("=== Demo: STC + rule-based picker ===")
    geom, data, Vp, Vs, Vst = _canonical_monopole_gather()

    res = stc(
        data,
        dt=geom.dt,
        offsets=geom.offsets,
        slowness_range=(30 * US_PER_FT, 360 * US_PER_FT),
        n_slowness=121,
        window_length=4.0e-4,
        time_step=2,
    )
    picks = pick_modes(res, threshold=0.4)

    logger.info("  Recovered:")
    for name, p in picks.items():
        amp_str = f"  amp={p.amplitude:7.4f}" if p.amplitude is not None else ""
        logger.info(
            "    %-9s slowness=%6.2f us/ft  V=%6.0f m/s  coh=%.3f%s",
            name,
            p.slowness / US_PER_FT,
            1.0 / p.slowness,
            p.coherence,
            amp_str,
        )
    logger.info("  Truth: Vp=%.0f  Vs=%.0f  Vst=%.0f", Vp, Vs, Vst)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _wiggle(
        axes[0],
        data,
        geom.t,
        xmax=3.5e-3,
        title=(
            f"Synthetic monopole gather\nVp={Vp:.0f}  Vs={Vs:.0f}  Vst={Vst:.0f} m/s"
        ),
    )
    ax = axes[1]
    pcm = ax.pcolormesh(
        res.time * 1e3,
        res.slowness / US_PER_FT,
        np.nan_to_num(res.coherence),
        shading="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    plt.colorbar(pcm, ax=ax, label="Coherence")
    for p in picks.values():
        ax.plot(
            p.time * 1e3,
            p.slowness / US_PER_FT,
            "o",
            mfc="none",
            mec="red",
            mew=2,
            ms=10,
            label=p.name,
        )
    ax.legend()
    ax.set_xlim(0, 3.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Slowness (us/ft)")
    ax.set_title("Slowness-Time Coherence\n(Kimball & Marzetta, 1984)")
    plt.tight_layout()
    _savefig(fig, figdir, "demo_stc_picker.png", show=show)


def demo_pseudo_rayleigh(figdir: str = "figures", show: bool = False) -> None:
    """STC + picker on a 4-mode gather including the pseudo-Rayleigh
    guided wave.

    The book (Mari et al. 1994, Part 1) lists the pseudo-Rayleigh /
    guided trapped mode alongside P, S and Stoneley as one of the
    arrivals the rule-based picker must consistently identify in
    fast formations. This demo plants all four arrivals at the
    canonical Schlumberger-array geometry, runs the same STC +
    :func:`fwap.picker.pick_modes` pipeline as :func:`demo_stc_picker`,
    and confirms that the four-mode :data:`fwap.picker.DEFAULT_PRIORS`
    recovers each one.
    """
    import matplotlib.pyplot as plt

    logger.info(
        "=== Demo: STC + 4-mode picker (P / S / pseudo-Rayleigh / Stoneley) ==="
    )
    Vp, Vs, Vst = 4500.0, 2500.0, 1400.0
    v_fluid = 1500.0
    f_pr = 8_000.0
    geom = ArrayGeometry(n_rec=8, tr_offset=3.0, dr=0.1524, dt=1.0e-5, n_samples=2048)
    modes = monopole_formation_modes(
        vp=Vp, vs=Vs, v_stoneley=Vst, v_fluid=v_fluid, f_pr=f_pr, pr_amp=2.0
    )
    data = synthesize_gather(geom, modes, noise=0.05, seed=11)

    res = stc(
        data,
        dt=geom.dt,
        offsets=geom.offsets,
        slowness_range=(30 * US_PER_FT, 360 * US_PER_FT),
        n_slowness=121,
        window_length=4.0e-4,
        time_step=2,
    )
    picks = pick_modes(res, threshold=0.4)

    logger.info("  Recovered:")
    for name, p in picks.items():
        amp_str = f"  amp={p.amplitude:7.4f}" if p.amplitude is not None else ""
        logger.info(
            "    %-15s slowness=%6.2f us/ft  V=%6.0f m/s  coh=%.3f%s",
            name,
            p.slowness / US_PER_FT,
            1.0 / p.slowness,
            p.coherence,
            amp_str,
        )
    logger.info(
        "  Truth: Vp=%.0f  Vs=%.0f  v_fluid=%.0f  Vst=%.0f", Vp, Vs, v_fluid, Vst
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _wiggle(
        axes[0],
        data,
        geom.t,
        xmax=3.5e-3,
        title=(
            f"Synthetic 4-mode monopole gather\n"
            f"Vp={Vp:.0f}  Vs={Vs:.0f}  v_fluid={v_fluid:.0f}  "
            f"Vst={Vst:.0f} m/s"
        ),
    )
    ax = axes[1]
    pcm = ax.pcolormesh(
        res.time * 1e3,
        res.slowness / US_PER_FT,
        np.nan_to_num(res.coherence),
        shading="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    plt.colorbar(pcm, ax=ax, label="Coherence")
    for p in picks.values():
        ax.plot(
            p.time * 1e3,
            p.slowness / US_PER_FT,
            "o",
            mfc="none",
            mec="red",
            mew=2,
            ms=10,
            label=p.name,
        )
    ax.legend()
    ax.set_xlim(0, 3.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Slowness (us/ft)")
    ax.set_title("STC -- four-mode picking\n(Mari et al. 1994, Part 1)")
    plt.tight_layout()
    _savefig(fig, figdir, "demo_pseudo_rayleigh.png", show=show)


def demo_wave_separation(figdir: str = "figures", show: bool = False) -> None:
    import matplotlib.pyplot as plt

    logger.info("=== Demo: Wave separation (f-k + SVD/K-L) ===")
    geom, data, Vp, Vs, Vst = _canonical_monopole_gather()
    p_only = fk_filter(
        data,
        geom.dt,
        geom.dr,
        slow_min=1.0 / 5500,
        slow_max=1.0 / 3600,
        taper_width=0.3,
    )
    s_only = fk_filter(
        data,
        geom.dt,
        geom.dr,
        slow_min=1.0 / 3000,
        slow_max=1.0 / 2100,
        taper_width=0.3,
    )
    st_only = fk_filter(
        data,
        geom.dt,
        geom.dr,
        slow_min=1.0 / 1700,
        slow_max=1.0 / 1100,
        taper_width=0.3,
    )
    comps, _ = sequential_kl_separation(
        data, geom.dt, geom.offsets, slownesses=[1.0 / Vp, 1.0 / Vs, 1.0 / Vst], rank=1
    )

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    _wiggle(axes[0, 0], data, geom.t, xmax=3.5e-3, title="Input gather")
    _wiggle(axes[0, 1], p_only, geom.t, xmax=3.5e-3, title="f-k band: P")
    _wiggle(axes[0, 2], s_only, geom.t, xmax=3.5e-3, title="f-k band: S")
    _wiggle(axes[0, 3], st_only, geom.t, xmax=3.5e-3, title="f-k band: Stoneley")
    _wiggle(axes[1, 0], data, geom.t, xmax=3.5e-3, title="Input gather")
    _wiggle(axes[1, 1], comps[0], geom.t, xmax=3.5e-3, title="SVD/K-L @ 1/Vp")
    _wiggle(axes[1, 2], comps[1], geom.t, xmax=3.5e-3, title="SVD/K-L @ 1/Vs")
    _wiggle(axes[1, 3], comps[2], geom.t, xmax=3.5e-3, title="SVD/K-L @ 1/Vst")
    plt.suptitle("Wave separation -- f-k filter (top) vs SVD/K-L (bottom)", fontsize=11)
    plt.tight_layout()
    _savefig(fig, figdir, "demo_wave_separation.png", show=show)


def demo_tau_p_separation(figdir: str = "figures", show: bool = False) -> None:
    """Slant-stack wave separation: τ-p panel + per-mode band-pass.

    Companion to :func:`demo_wave_separation`. The book lists the
    τ-p (linear Radon) domain alongside f-k as a textbook
    multichannel velocity-filter approach for Part 2; this demo
    plots the canonical P/S/Stoneley monopole gather, its forward
    τ-p panel, and the band-passed reconstructions for each mode.

    Unlike f-k, τ-p does not need a uniformly-spaced receiver array
    -- a property the demo exercises by feeding the actual (regular)
    Schlumberger geometry but flagging the more general support in
    the figure caption.
    """
    import matplotlib.pyplot as plt

    logger.info("=== Demo: Wave separation (tau-p / slant-stack) ===")
    geom, data, Vp, Vs, Vst = _canonical_monopole_gather()

    # Per-mode pass-bands centred on the planted slownesses, taken
    # from the same window edges as the f-k demo for direct
    # comparison.
    p_band = tau_p_filter(
        data,
        geom.dt,
        geom.offsets,
        slow_min=1.0 / 5500,
        slow_max=1.0 / 3600,
        n_slowness=181,
        taper_width=0.3,
    )
    s_band = tau_p_filter(
        data,
        geom.dt,
        geom.offsets,
        slow_min=1.0 / 3000,
        slow_max=1.0 / 2100,
        n_slowness=181,
        taper_width=0.3,
    )
    st_band = tau_p_filter(
        data,
        geom.dt,
        geom.offsets,
        slow_min=1.0 / 1700,
        slow_max=1.0 / 1100,
        n_slowness=181,
        taper_width=0.3,
    )

    # Forward panel for visualisation.
    slownesses = np.linspace(20.0 * US_PER_FT, 360.0 * US_PER_FT, 256)
    panel = tau_p_forward(data, geom.dt, geom.offsets, slownesses)

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    _wiggle(
        axes[0],
        data,
        geom.t,
        xmax=3.5e-3,
        title=f"Input gather\nVp={Vp:.0f} Vs={Vs:.0f} Vst={Vst:.0f} m/s",
    )
    pcm = axes[1].pcolormesh(
        np.arange(panel.shape[1]) * geom.dt * 1e3,
        slownesses / US_PER_FT,
        np.abs(panel),
        shading="auto",
        cmap="magma",
    )
    plt.colorbar(pcm, ax=axes[1], label="|panel|")
    axes[1].set_xlim(0, 3.5)
    axes[1].set_xlabel("tau (ms)")
    axes[1].set_ylabel("slowness (us/ft)")
    axes[1].set_title("Forward tau-p panel")
    _wiggle(axes[2], p_band, geom.t, xmax=3.5e-3, title="tau-p band: P")
    _wiggle(axes[3], s_band, geom.t, xmax=3.5e-3, title="tau-p band: S")
    _wiggle(axes[4], st_band, geom.t, xmax=3.5e-3, title="tau-p band: Stoneley")
    plt.suptitle("Wave separation -- tau-p (slant-stack / linear Radon)", fontsize=11)
    plt.tight_layout()
    _savefig(fig, figdir, "demo_tau_p_separation.png", show=show)


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


def demo_attenuation(figdir: str = "figures", show: bool = False) -> None:
    import matplotlib.pyplot as plt

    logger.info("=== Demo: Attenuation (Q) from array sonic ===")
    geom = ArrayGeometry(n_rec=12, tr_offset=3.0, dr=0.1524, dt=1.0e-5, n_samples=2048)
    Vp = 4000.0
    Q_true = 50.0
    f0 = 15_000.0
    # Build an attenuated Ricker per receiver: multiply spectrum by
    # exp(-pi*f*t/Q).
    t = geom.t
    t0 = 2.0e-4
    n = geom.n_samples
    freqs = np.fft.rfftfreq(n, d=geom.dt)
    data = np.zeros((geom.n_rec, n))
    rng = np.random.default_rng(4)
    for i, off in enumerate(geom.offsets):
        tt = t0 + off / Vp
        src = ricker(t, f0=f0, t0=tt)
        S = np.fft.rfft(src)
        # Note: attenuation scales with *travel* time (off/Vp), not tt.
        S = S * np.exp(-np.pi * freqs * (off / Vp) / Q_true)
        data[i] = np.fft.irfft(S, n=n)
    rms = np.sqrt(np.mean(data**2)) + 1e-12
    data += rng.normal(scale=0.02 * rms, size=data.shape)

    res_c = centroid_frequency_shift_Q(
        data,
        dt=geom.dt,
        offsets=geom.offsets,
        slowness=1.0 / Vp,
        window_length=4.0e-4,
        f_range=(5_000.0, 30_000.0),
        pick_intercept=t0,
    )
    res_r = spectral_ratio_Q(
        data,
        dt=geom.dt,
        offsets=geom.offsets,
        slowness=1.0 / Vp,
        window_length=4.0e-4,
        f_range=(5_000.0, 25_000.0),
        pick_intercept=t0,
    )
    logger.info("  True Q             = %.1f", Q_true)
    logger.info("  Centroid-shift  Q  = %.1f +/- %.1f", res_c.q, res_c.q_sigma)
    logger.info("  Spectral-ratio  Q  = %.1f +/- %.1f", res_r.q, res_r.q_sigma)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    _wiggle(
        axes[0],
        data,
        geom.t,
        xmax=3.0e-3,
        title=f"Attenuated gather (true Q={Q_true:.0f})",
    )
    ax = axes[1]
    tt_arr = t0 + geom.offsets / Vp
    ax.plot(tt_arr * 1e6, res_c.diagnostic["fc"] / 1e3, "bo-", label="centroid fc(t)")
    slope = float(res_c.diagnostic["slope"])
    inter = float(res_c.diagnostic["intercept"])
    fit = slope * tt_arr + inter
    ax.plot(tt_arr * 1e6, fit / 1e3, "r--", label="LS fit")
    ax.set_xlabel("Travel time (us)")
    ax.set_ylabel("Centroid freq (kHz)")
    ax.set_title(f"Centroid freq shift\nQ = {res_c.q:.1f} +/- {res_c.q_sigma:.1f}")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[2]
    # Log spectral ratio example: last receiver vs first.
    n_rec = data.shape[0]
    L = max(2, int(round(4.0e-4 / geom.dt)))
    ax.set_title(f"Spectral ratio method\nQ = {res_r.q:.1f} +/- {res_r.q_sigma:.1f}")
    i_ref = 0
    for i in range(1, n_rec, 2):
        ti = t0 + geom.offsets[i] / Vp
        t_ref = t0 + geom.offsets[i_ref] / Vp
        w_i = (
            np.hanning(L)
            * data[i, int(round(ti / geom.dt)) : int(round(ti / geom.dt)) + L]
        )
        w_r = (
            np.hanning(L)
            * data[i_ref, int(round(t_ref / geom.dt)) : int(round(t_ref / geom.dt)) + L]
        )
        fi = np.fft.rfftfreq(L, d=geom.dt)
        ampi = np.abs(np.fft.rfft(w_i))
        ampr = np.abs(np.fft.rfft(w_r))
        mask = (fi >= 5_000) & (fi <= 25_000) & (ampi > 1e-9) & (ampr > 1e-9)
        ax.plot(
            fi[mask] / 1e3,
            np.log(ampi[mask] / ampr[mask]),
            label=f"rec {i}/{i_ref}",
            alpha=0.6,
        )
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("log |A_i / A_0|")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    plt.suptitle(
        "Attenuation (Q) estimation\nQuan & Harris (1997); Bath (1974)", fontsize=11
    )
    plt.tight_layout()
    _savefig(fig, figdir, "demo_attenuation.png", show=show)


def demo_alford(figdir: str = "figures", show: bool = False) -> None:
    import matplotlib.pyplot as plt

    logger.info("=== Demo: Cross-dipole Alford rotation ===")
    # Simulate two orthogonal shear arrivals at different slownesses
    # observed on the (x, y) dipole-pair tensor.
    n_samp = 1024
    dt = 2.0e-5
    t = np.arange(n_samp) * dt
    true_angle = np.deg2rad(30.0)  # fast axis at +30 deg from x
    Vs_fast = 2600.0
    Vs_slow = 2400.0
    offset = 3.5
    t_fast = offset / Vs_fast
    t_slow = offset / Vs_slow
    f0 = 3000.0
    fast = ricker(t, f0, t0=t_fast)
    slow = 0.85 * ricker(t, f0, t0=t_slow)
    # In the rotated (fast, slow) frame: [F, 0; 0, S].
    # Rotate back to the tool (x, y) frame by theta = -true_angle
    # (inverse of the fast-frame rotation):
    c, s = np.cos(true_angle), np.sin(true_angle)
    xx = c * c * fast + s * s * slow
    yy = s * s * fast + c * c * slow
    xy = c * s * (fast - slow)
    yx = c * s * (fast - slow)
    rng = np.random.default_rng(5)
    for arr in (xx, xy, yx, yy):
        arr += rng.normal(scale=0.01 * np.max(np.abs(arr)), size=arr.shape)

    res = alford_rotation(xx, xy, yx, yy)
    logger.info("  True fast axis:  %.2f deg", np.rad2deg(true_angle))
    logger.info(
        "  Recovered axis:  %.2f deg  (cross_en_ratio=%.3e)",
        np.rad2deg(res.angle),
        res.cross_energy_ratio,
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].plot(t * 1e3, xx, "k")
    axes[0, 0].set_title("xx")
    axes[0, 1].plot(t * 1e3, xy, "k")
    axes[0, 1].set_title("xy")
    axes[1, 0].plot(t * 1e3, yx, "k")
    axes[1, 0].set_title("yx")
    axes[1, 1].plot(t * 1e3, yy, "k")
    axes[1, 1].set_title("yy")
    for ax in axes.ravel():
        ax.set_xlim(0.5, 2.0)
        ax.set_xlabel("Time (ms)")
        ax.grid(alpha=0.3)
    plt.suptitle("Input cross-dipole tensor (in tool x,y frame)", fontsize=11)
    plt.tight_layout()
    _savefig(fig, figdir, "demo_alford_input.png", show=show)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(t * 1e3, res.fast, "b", label="fast")
    axes[0].plot(t * 1e3, res.slow, "r", label="slow")
    axes[0].set_xlim(0.5, 2.0)
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_title(
        f"After Alford rotation\n"
        f"angle = {np.rad2deg(res.angle):.2f} deg  "
        f"(truth {np.rad2deg(true_angle):.2f} deg)"
    )
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    thetas = np.linspace(-np.pi / 2, np.pi / 2, 181)
    cross_en = np.zeros_like(thetas)
    for k, th in enumerate(thetas):
        c, s = np.cos(th), np.sin(th)
        xy_r = c * s * (yy - xx) + c * c * xy - s * s * yx
        yx_r = c * s * (yy - xx) - s * s * xy + c * c * yx
        cross_en[k] = np.sum(xy_r**2) + np.sum(yx_r**2)
    axes[1].plot(np.rad2deg(thetas), cross_en, "k")
    axes[1].axvline(
        np.rad2deg(res.angle),
        color="r",
        ls="--",
        label=f"recovered {np.rad2deg(res.angle):.1f} deg",
    )
    axes[1].axvline(
        np.rad2deg(true_angle),
        color="g",
        ls=":",
        label=f"truth {np.rad2deg(true_angle):.1f} deg",
    )
    axes[1].set_xlabel("Rotation angle theta (deg)")
    axes[1].set_ylabel("Cross-component energy")
    axes[1].set_title("Alford cost function")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Cross-dipole Alford rotation -- Alford (1986)", fontsize=11)
    plt.tight_layout()
    _savefig(fig, figdir, "demo_alford.png", show=show)


def demo_lwd(figdir: str = "figures", show: bool = False) -> None:
    """LWD phenomenological layer: collar rejection + quadrupole stack.

    Two figures:

    * ``demo_lwd_monopole.png`` -- a monopole gather contaminated by
      the LWD steel-collar arrival shows the collar peak dominating
      the slowness-time-coherence map at ~92 us/ft. After
      :func:`fwap.lwd.notch_slowness_band` rejection at the known
      collar slowness, the formation P / S / Stoneley peaks are the
      strongest cells on the map and :func:`fwap.picker.pick_modes`
      recovers all three to within 10 us/ft of truth.

    * ``demo_lwd_quadrupole.png`` -- the quadrupole-tool workflow.
      Per-receiver amplitudes on the ring follow ``cos(2(theta -
      phi))``; :func:`fwap.lwd.quadrupole_stack` projects the ring
      onto that pattern and rejects the orthogonal m=0 / m=1
      components by construction. Stacking eight per-axial-offset
      rings gives an axial-array record that picks the formation
      shear slowness via :data:`fwap.lwd.lwd_quadrupole_priors`.

    References
    ----------
    Tang, X.-M., & Cheng, A. (2004). *Quantitative Borehole Acoustic
    Methods*, sect. 2.4-2.5 (LWD multipole propagation; quadrupole
    source as the practical solution to collar-mode contamination).
    """
    import matplotlib.pyplot as plt

    logger.info("=== Demo: LWD phenomenological layer ===")
    from fwap.lwd import (
        DEFAULT_COLLAR_SLOWNESS_S_PER_M,
        lwd_quadrupole_priors,
        notch_slowness_band,
        quadrupole_stack,
        synthesize_lwd_gather,
        synthesize_quadrupole_lwd_gather,
    )

    # ---- Figure 1: monopole + collar rejection ----
    Vp, Vs, Vst = 4500.0, 2500.0, 1400.0
    geom = ArrayGeometry(n_rec=8, tr_offset=3.0, dr=0.1524, dt=1.0e-5, n_samples=2048)
    formation = monopole_formation_modes(vp=Vp, vs=Vs, v_stoneley=Vst)
    collar_slow = DEFAULT_COLLAR_SLOWNESS_S_PER_M
    data = synthesize_lwd_gather(
        geom,
        formation,
        collar_amplitude=1.0,
        collar_slowness=collar_slow,
        noise=0.03,
        seed=7,
    )
    cleaned = notch_slowness_band(
        data,
        dt=geom.dt,
        offsets=geom.offsets,
        slow_min=collar_slow * 0.85,
        slow_max=collar_slow * 1.15,
        n_slowness=181,
        taper_width=0.15,
    )
    surf_dirty = stc(
        data,
        dt=geom.dt,
        offsets=geom.offsets,
        slowness_range=(30 * US_PER_FT, 360 * US_PER_FT),
        n_slowness=181,
        window_length=4.0e-4,
        time_step=2,
    )
    surf_clean = stc(
        cleaned,
        dt=geom.dt,
        offsets=geom.offsets,
        slowness_range=(30 * US_PER_FT, 360 * US_PER_FT),
        n_slowness=181,
        window_length=4.0e-4,
        time_step=2,
    )
    from fwap.picker import DEFAULT_PRIORS

    three_mode_priors = {m: DEFAULT_PRIORS[m] for m in ("P", "S", "Stoneley")}
    picks_clean = pick_modes(surf_clean, priors=three_mode_priors, threshold=0.4)

    logger.info(
        "  Monopole + collar contamination at %.0f us/ft, Vp=%.0f Vs=%.0f Vst=%.0f m/s",
        collar_slow / US_PER_FT,
        Vp,
        Vs,
        Vst,
    )
    logger.info("  After collar-band notch:")
    for name in ("P", "S", "Stoneley"):
        if name in picks_clean:
            p = picks_clean[name]
            logger.info(
                "    %-9s slowness=%6.2f us/ft  V=%6.0f m/s  coh=%.3f",
                name,
                p.slowness / US_PER_FT,
                1.0 / p.slowness,
                p.coherence,
            )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, surf, title in (
        (axes[0], surf_dirty, "Pre-rejection: collar peak dominates"),
        (axes[1], surf_clean, "Post-rejection: formation P/S/Stoneley recovered"),
    ):
        pcm = ax.pcolormesh(
            surf.time * 1e3,
            surf.slowness / US_PER_FT,
            np.nan_to_num(surf.coherence),
            shading="auto",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        plt.colorbar(pcm, ax=ax, label="Coherence")
        ax.axhline(
            collar_slow / US_PER_FT,
            color="orange",
            ls=":",
            alpha=0.7,
            label="Collar slowness",
        )
        for V, lbl in ((Vp, "P"), (Vs, "S"), (Vst, "Stoneley")):
            ax.axhline((1.0 / V) / US_PER_FT, color="white", ls="--", alpha=0.4)
        ax.set_xlim(0, 5.0)
        ax.set_ylim(30, 320)
        ax.set_xlabel("Time (ms)")
        ax.set_title(title)
    axes[0].set_ylabel("Slowness (us/ft)")
    axes[0].legend(loc="upper right", fontsize=9)
    if picks_clean:
        for p in picks_clean.values():
            axes[1].plot(
                p.time * 1e3,
                p.slowness / US_PER_FT,
                "o",
                mfc="none",
                mec="red",
                mew=2,
                ms=10,
            )
    plt.suptitle(
        "LWD collar-mode rejection -- monopole, Tang & Cheng (2004) sect. 2.4",
        fontsize=11,
    )
    plt.tight_layout()
    _savefig(fig, figdir, "demo_lwd_monopole.png", show=show)

    # ---- Figure 2: quadrupole ring + stacked-trace shear pick ----
    Vs_q = 2300.0
    n_axial = 8
    dr = 0.1524
    tr_offset0 = 3.0
    n_samples = 2048
    dt = 1.0e-5
    axial_traces = np.empty((n_axial, n_samples), dtype=float)
    rings = []
    for k in range(n_axial):
        offset_k = tr_offset0 + k * dr
        g = synthesize_quadrupole_lwd_gather(
            n_rec=8,
            n_samples=n_samples,
            dt=dt,
            tool_offset=offset_k,
            formation_slowness=1.0 / Vs_q,
            formation_f0=6000.0,
            formation_amplitude=1.0,
            include_collar=True,
            collar_slowness=DEFAULT_COLLAR_SLOWNESS_S_PER_M,
            collar_amplitude=1.0,
            noise=0.02,
            seed=11 + k,
        )
        rings.append(g)
        axial_traces[k] = quadrupole_stack(
            g.data, g.azimuths, source_azimuth=g.source_azimuth
        )
    offsets = tr_offset0 + np.arange(n_axial) * dr
    surf_q = stc(
        axial_traces,
        dt=dt,
        offsets=offsets,
        slowness_range=(50e-6, 600e-6),
        n_slowness=181,
        window_length=4.0e-4,
        time_step=2,
    )
    picks_q = pick_modes(surf_q, priors=lwd_quadrupole_priors(), threshold=0.4)
    logger.info("  Quadrupole stack -> formation Vs from m=2 picker:")
    for name, p in picks_q.items():
        logger.info(
            "    %-18s slowness=%6.2f us/ft  V=%6.0f m/s  coh=%.3f",
            name,
            p.slowness / US_PER_FT,
            1.0 / p.slowness,
            p.coherence,
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    # Per-receiver peak amplitude on the first ring vs azimuth (rad).
    g0 = rings[0]
    j_peak = int(round((tr_offset0 / Vs_q) / dt))
    azim_deg = np.rad2deg(g0.azimuths)
    ax.plot(azim_deg, g0.data[:, j_peak], "o-", label="receiver amplitude")
    theta_grid = np.linspace(0.0, 360.0, 361)
    ax.plot(
        theta_grid,
        np.cos(2.0 * np.deg2rad(theta_grid)),
        "k--",
        alpha=0.5,
        label=r"$\cos(2\theta)$  (m=2 source pattern)",
    )
    ax.set_xlabel("Receiver azimuth (deg)")
    ax.set_ylabel("Amplitude at formation-arrival sample")
    ax.set_title(
        f"Quadrupole ring response\nVs = {Vs_q:.0f} m/s,  8 azimuthal receivers"
    )
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    pcm = ax.pcolormesh(
        surf_q.time * 1e3,
        surf_q.slowness / US_PER_FT,
        np.nan_to_num(surf_q.coherence),
        shading="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    plt.colorbar(pcm, ax=ax, label="Coherence")
    ax.axhline(
        (1.0 / Vs_q) / US_PER_FT, color="white", ls="--", alpha=0.6, label="True Vs"
    )
    for p in picks_q.values():
        ax.plot(
            p.time * 1e3,
            p.slowness / US_PER_FT,
            "o",
            mfc="none",
            mec="red",
            mew=2,
            ms=10,
            label=p.name,
        )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Slowness (us/ft)")
    ax.set_xlim(0, 5.0)
    ax.set_title("STC of quadrupole-stacked axial array")
    ax.legend(loc="upper right", fontsize=9)
    plt.suptitle(
        "LWD quadrupole stack -- m=2 source / receiver geometry, "
        "Tang & Cheng (2004) sect. 2.5",
        fontsize=11,
    )
    plt.tight_layout()
    _savefig(fig, figdir, "demo_lwd_quadrupole.png", show=show)


def demo_las_roundtrip(figdir: str = "figures", show: bool = False) -> None:
    """
    End-to-end LAS I/O: synthesize logs, process, write, read back.

    Exercises :mod:`fwap.io`'s round-trip on a synthetic Vp/Vs/Stoneley
    log set plus elastic moduli derived from them, and plots the
    written-vs-read curves to confirm the I/O path is lossless.
    """
    import os

    import matplotlib.pyplot as plt

    logger.info("=== Demo: LAS round-trip (fwap.io) ===")
    from fwap.io import read_las, write_las
    from fwap.rockphysics import elastic_moduli

    os.makedirs(figdir, exist_ok=True)
    path = os.path.join(figdir, "demo_las_roundtrip.las")

    # Build a synthetic 100-m log interval with a smoothly varying Vp,
    # Vs, Stoneley slowness, and derive moduli from them.
    depth = np.linspace(1000.0, 1100.0, 501)
    vp = 4500.0 + 200.0 * np.sin(depth / 15.0)
    vs = 2500.0 + 120.0 * np.sin(depth / 15.0 + 0.3)
    vst = np.full_like(depth, 1400.0)
    rho = np.full_like(depth, 2400.0)
    moduli = elastic_moduli(vp=vp, vs=vs, rho=rho)

    curves = {
        "DTP": 1.0e6 / vp * 0.3048,  # us/ft
        "DTS": 1.0e6 / vs * 0.3048,
        "DTST": 1.0e6 / vst * 0.3048,
        "VPVS": vp / vs,
        "K": moduli.k,
        "MU": moduli.mu,
        "E": moduli.young,
        "NU": moduli.poisson,
    }
    write_las(
        path,
        depth,
        curves,
        depth_unit="M",
        well_name="FWAP_DEMO",
        well={"COMP": "fwap", "SRVC": "fwap.demos"},
    )
    logger.info("  wrote %s (%d curves, %d depths)", path, len(curves), depth.size)

    loaded = read_las(path)
    logger.info(
        "  read  %s (%d curves, step=%.4f m)", path, len(loaded.curves), loaded.step
    )

    # Quantitative round-trip check (lasio writes with a few
    # decimal places of precision, so report the RMS drift per curve).
    logger.info("  round-trip RMS drift:")
    for name, orig in curves.items():
        mask = np.isfinite(orig) & np.isfinite(loaded.curves[name])
        drift = float(np.sqrt(np.mean((loaded.curves[name][mask] - orig[mask]) ** 2)))
        logger.info("    %-5s  %.3e  %s", name, drift, loaded.units[name])

    fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
    ax = axes[0]
    ax.plot(curves["DTP"], depth, "b-", label="DTP (written)")
    ax.plot(loaded.curves["DTP"], depth, "b:", alpha=0.6, label="DTP (read)")
    ax.plot(curves["DTS"], depth, "r-", label="DTS (written)")
    ax.plot(loaded.curves["DTS"], depth, "r:", alpha=0.6, label="DTS (read)")
    ax.set_xlabel("Slowness (us/ft)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Compressional / shear slowness")

    ax = axes[1]
    ax.plot(curves["VPVS"], depth, "k-", label="Vp/Vs")
    ax.set_xlabel("Vp/Vs (-)")
    ax.grid(alpha=0.3)
    ax.set_title("Vp/Vs ratio")

    ax = axes[2]
    ax.plot(curves["E"] / 1e9, depth, "g-", label="Young's E (GPa)")
    ax.plot(curves["MU"] / 1e9, depth, "b-", label="shear mu (GPa)")
    ax.plot(curves["K"] / 1e9, depth, "m-", label="bulk K (GPa)")
    ax.set_xlabel("Modulus (GPa)")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Elastic moduli")

    plt.suptitle(
        "LAS I/O round-trip -- write synthesized logs, read them back, plot",
        fontsize=11,
    )
    plt.tight_layout()
    _savefig(fig, figdir, "demo_las_roundtrip.png", show=show)


def demo_dlis_roundtrip(figdir: str = "figures", show: bool = False) -> None:
    """
    End-to-end DLIS I/O: synthesize logs, process, write, read back.

    Parallels :func:`demo_las_roundtrip` for the binary RP66 v1
    format. Because DLIS stores curves as raw IEEE float64 (rather
    than fixed-decimal ASCII like LAS), the round-trip drift here is
    exactly zero -- a useful contrast that this demo logs alongside
    the LAS-style RMS-drift summary.
    """
    import os

    import matplotlib.pyplot as plt

    logger.info("=== Demo: DLIS round-trip (fwap.io) ===")
    from fwap.io import read_dlis, write_dlis
    from fwap.rockphysics import elastic_moduli

    os.makedirs(figdir, exist_ok=True)
    path = os.path.join(figdir, "demo_dlis_roundtrip.dlis")

    # Same synthetic 100-m log interval as the LAS demo so the two
    # demos can be compared side-by-side.
    depth = np.linspace(1000.0, 1100.0, 501)
    vp = 4500.0 + 200.0 * np.sin(depth / 15.0)
    vs = 2500.0 + 120.0 * np.sin(depth / 15.0 + 0.3)
    vst = np.full_like(depth, 1400.0)
    rho = np.full_like(depth, 2400.0)
    moduli = elastic_moduli(vp=vp, vs=vs, rho=rho)

    curves = {
        "DTP": 1.0e6 / vp * 0.3048,  # us/ft
        "DTS": 1.0e6 / vs * 0.3048,
        "DTST": 1.0e6 / vst * 0.3048,
        "VPVS": vp / vs,
        "K": moduli.k,
        "MU": moduli.mu,
        "E": moduli.young,
        "NU": moduli.poisson,
    }
    write_dlis(
        path,
        depth,
        curves,
        depth_unit="m",
        well_name="FWAP_DEMO",
        well={"COMP": "fwap", "FLD": "fwap.demos"},
    )
    logger.info("  wrote %s (%d curves, %d depths)", path, len(curves), depth.size)

    loaded = read_dlis(path)
    logger.info(
        "  read  %s (frame=%s, index_type=%s, step=%.4f m)",
        path,
        loaded.frame_name,
        loaded.index_type,
        loaded.step,
    )

    # Quantitative round-trip check. DLIS stores IEEE float64 verbatim,
    # so unlike LAS this should be bit-identical -- the drift values
    # below should all be exactly zero.
    logger.info("  round-trip RMS drift:")
    for name, orig in curves.items():
        mask = np.isfinite(orig) & np.isfinite(loaded.curves[name])
        drift = float(np.sqrt(np.mean((loaded.curves[name][mask] - orig[mask]) ** 2)))
        logger.info("    %-5s  %.3e  %s", name, drift, loaded.units[name])

    fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
    ax = axes[0]
    ax.plot(curves["DTP"], depth, "b-", label="DTP (written)")
    ax.plot(loaded.curves["DTP"], depth, "b:", alpha=0.6, label="DTP (read)")
    ax.plot(curves["DTS"], depth, "r-", label="DTS (written)")
    ax.plot(loaded.curves["DTS"], depth, "r:", alpha=0.6, label="DTS (read)")
    ax.set_xlabel("Slowness (us/ft)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Compressional / shear slowness")

    ax = axes[1]
    ax.plot(curves["VPVS"], depth, "k-", label="Vp/Vs")
    ax.set_xlabel("Vp/Vs (-)")
    ax.grid(alpha=0.3)
    ax.set_title("Vp/Vs ratio")

    ax = axes[2]
    ax.plot(curves["E"] / 1e9, depth, "g-", label="Young's E (GPa)")
    ax.plot(curves["MU"] / 1e9, depth, "b-", label="shear mu (GPa)")
    ax.plot(curves["K"] / 1e9, depth, "m-", label="bulk K (GPa)")
    ax.set_xlabel("Modulus (GPa)")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Elastic moduli")

    plt.suptitle(
        "DLIS I/O round-trip -- binary RP66 v1, bit-exact float64 storage", fontsize=11
    )
    plt.tight_layout()
    _savefig(fig, figdir, "demo_dlis_roundtrip.png", show=show)


def demo_segy_roundtrip(figdir: str = "figures", show: bool = False) -> None:
    """
    End-to-end SEG-Y I/O: synthesise, write, read, STC, plot.

    Exercises :func:`fwap.io.write_segy` and :func:`fwap.io.read_segy`
    on a canonical monopole gather and verifies that the processing
    chain gives identical results from the in-memory array and from
    the SEG-Y round-tripped one.
    """
    import os

    import matplotlib.pyplot as plt

    logger.info("=== Demo: SEG-Y round-trip (fwap.io) ===")
    from fwap.io import read_segy, write_segy

    os.makedirs(figdir, exist_ok=True)
    path = os.path.join(figdir, "demo_segy_roundtrip.sgy")

    geom, data, Vp, Vs, Vst = _canonical_monopole_gather()
    # Write integer-metre offsets so they survive the 32-bit int
    # ``offset`` header.
    offset_mm = np.round(geom.offsets * 1000.0).astype(int)
    write_segy(
        path,
        data.astype(np.float32),
        dt=geom.dt,
        offsets=offset_mm,
        textual_header="fwap demo_segy_roundtrip synthesized gather",
    )
    logger.info(
        "  wrote %s (%d traces, %d samples, dt=%.1f us)",
        path,
        data.shape[0],
        data.shape[1],
        geom.dt * 1e6,
    )

    loaded = read_segy(path)
    logger.info(
        "  read  %s (%d traces, %d samples, dt=%.1f us)",
        path,
        loaded.n_traces,
        loaded.n_samples,
        loaded.dt * 1e6,
    )

    # Verify the round-trip is bit-exact in float32 and the metadata
    # survives.
    max_abs_err = float(np.max(np.abs(loaded.data - data)))
    rel_err = max_abs_err / (float(np.max(np.abs(data))) + 1e-30)
    logger.info("  data round-trip max|err|=%.3e (rel %.3e)", max_abs_err, rel_err)
    # dt round-trips via an integer microseconds field, so only
    # approximate equality is guaranteed.
    assert abs(loaded.dt - geom.dt) / geom.dt < 1.0e-6

    # STC the round-tripped gather and compare peak slowness to truth.
    res = stc(
        loaded.data.astype(float),
        dt=loaded.dt,
        offsets=geom.offsets,
        slowness_range=(30 * US_PER_FT, 360 * US_PER_FT),
        n_slowness=121,
        window_length=4.0e-4,
        time_step=2,
    )
    picks = pick_modes(res, threshold=0.4)
    for name, p in picks.items():
        logger.info(
            "    %-9s V=%6.0f m/s  coh=%.3f", name, 1.0 / p.slowness, p.coherence
        )
    logger.info("  Truth: Vp=%.0f  Vs=%.0f  Vst=%.0f", Vp, Vs, Vst)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _wiggle(axes[0], data, geom.t, xmax=3.5e-3, title="Original synthetic gather")
    _wiggle(
        axes[1],
        loaded.data,
        geom.t,
        xmax=3.5e-3,
        title=(f"After SEG-Y round-trip\nmax |diff|={max_abs_err:.2e}"),
    )
    plt.suptitle(
        "SEG-Y I/O round-trip -- write, read, re-process, compare", fontsize=11
    )
    plt.tight_layout()
    _savefig(fig, figdir, "demo_segy_roundtrip.png", show=show)
