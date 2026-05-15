"""
Extension demos: attenuation, cross-dipole anisotropy, LWD.

* :func:`demo_attenuation` -- Q estimation via the centroid-
  frequency-shift and spectral-ratio methods.
* :func:`demo_alford`      -- cross-dipole Alford rotation
  for shear-wave azimuthal anisotropy.
* :func:`demo_lwd`         -- LWD phenomenological layer:
  collar rejection + quadrupole stack (Tang & Cheng 2004
  sect. 2.4-2.5).
"""

from __future__ import annotations

import numpy as np

from fwap._common import US_PER_FT, logger
from fwap.anisotropy import alford_rotation
from fwap.attenuation import centroid_frequency_shift_Q, spectral_ratio_Q
from fwap.coherence import stc
from fwap.picker import pick_modes
from fwap.plotting import save_figure as _savefig
from fwap.plotting import wiggle_plot as _wiggle
from fwap.synthetic import (
    ArrayGeometry,
    monopole_formation_modes,
    ricker,
)


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
        for V, _lbl in ((Vp, "P"), (Vs, "S"), (Vst, "Stoneley")):
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
    axial_traces: np.ndarray = np.empty((n_axial, n_samples), dtype=float)
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
