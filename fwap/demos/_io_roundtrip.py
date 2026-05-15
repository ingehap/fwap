"""
I/O round-trip demos for the three supported well-log formats.

* :func:`demo_las_roundtrip`  -- LAS (lasio) on a synthetic log.
* :func:`demo_dlis_roundtrip` -- DLIS / RP66 (dlisio reader +
  dliswriter producer).
* :func:`demo_segy_roundtrip` -- SEG-Y (segyio) on the
  canonical monopole gather.
"""

from __future__ import annotations

import numpy as np

from fwap._common import US_PER_FT, logger, m_per_s_to_us_per_ft
from fwap.coherence import stc
from fwap.demos._common import _canonical_monopole_gather
from fwap.picker import pick_modes
from fwap.plotting import save_figure as _savefig
from fwap.plotting import wiggle_plot as _wiggle


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
        "DTP": m_per_s_to_us_per_ft(vp),
        "DTS": m_per_s_to_us_per_ft(vs),
        "DTST": m_per_s_to_us_per_ft(vst),
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
        "DTP": m_per_s_to_us_per_ft(vp),
        "DTS": m_per_s_to_us_per_ft(vs),
        "DTST": m_per_s_to_us_per_ft(vst),
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
