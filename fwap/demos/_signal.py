"""
Part 1 and Part 2 demos: STC picking and wavefield separation.

Covers the four book-chapter demos that run on the canonical
monopole gather:

* :func:`demo_stc_picker`      -- Part 1: STC + rule-based
  mode picker on a monopole gather.
* :func:`demo_pseudo_rayleigh` -- Part 1: STC picker on a
  4-mode gather (P, S, Stoneley, guided pseudo-Rayleigh).
* :func:`demo_wave_separation` -- Part 2: f-k filtering plus
  sequential K-L (SVD) separation.
* :func:`demo_tau_p_separation` -- Part 2: tau-p / slant-stack
  forward and inverse with band-pass reconstruction.
"""

from __future__ import annotations

import numpy as np

from fwap._common import US_PER_FT, logger
from fwap.coherence import stc
from fwap.demos._common import _canonical_monopole_gather
from fwap.picker import pick_modes
from fwap.plotting import save_figure as _savefig
from fwap.plotting import wiggle_plot as _wiggle
from fwap.synthetic import (
    ArrayGeometry,
    monopole_formation_modes,
    synthesize_gather,
)
from fwap.wavesep import (
    fk_filter,
    sequential_kl_separation,
    tau_p_filter,
    tau_p_forward,
)


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
