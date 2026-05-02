"""
Command-line entry point: run demos, or process a real SEG-Y file.

Usage shapes
------------

Run every demo and save figures::

    fwap                         # == "fwap all"
    fwap all

Run a single demo::

    fwap stc                     # Part 1: STC + picker
    fwap wavesep                 # Part 2: wave separation
    ... (see ``fwap --help`` for the full list)

Process a single SEG-Y gather: reads the file, runs STC, picks P / S /
Stoneley, prints the recovered slownesses to stdout::

    fwap process gather.sgy
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Callable

import numpy as np

from fwap import demos as _demos
from fwap._common import US_PER_FT, logger

_DEMOS: dict[str, Callable[..., None]] = {
    "stc": _demos.demo_stc_picker,
    "pseudorayleigh": _demos.demo_pseudo_rayleigh,
    "wavesep": _demos.demo_wave_separation,
    "taup": _demos.demo_tau_p_separation,
    "intercept": _demos.demo_intercept_time,
    "dipole": _demos.demo_dipole,
    "dip": _demos.demo_dip,
    "attenuation": _demos.demo_attenuation,
    "alford": _demos.demo_alford,
    "lwd": _demos.demo_lwd,
    "las": _demos.demo_las_roundtrip,
    "dlis": _demos.demo_dlis_roundtrip,
    "segy": _demos.demo_segy_roundtrip,
}


def _pick_one_gather(
    path: str, offset_header: str, offset_scale: float, threshold: float
):
    """Read a single SEG-Y file and run STC + pick_modes.

    Returns ``(picks, gather)`` where ``picks`` is the
    ``dict[str, ModePick]`` from :func:`pick_modes` (possibly empty)
    and ``gather`` is the underlying :class:`fwap.io.SegyGather`.
    Raises ``ValueError`` when the SEG-Y has no usable offset column.
    """
    from fwap.coherence import stc
    from fwap.io import read_segy
    from fwap.picker import pick_modes

    g = read_segy(path, offset_header=offset_header)
    if g.offsets is None:
        raise ValueError(
            f"{path}: offsets column is all zero; cannot process "
            f"without source-to-receiver offsets. Try another "
            f"--offset-header (e.g. SourceX)."
        )
    offsets_m = g.offsets / offset_scale
    surface = stc(
        g.data,
        dt=g.dt,
        offsets=offsets_m,
        slowness_range=(30.0 * US_PER_FT, 360.0 * US_PER_FT),
        n_slowness=121,
        window_length=4.0e-4,
        time_step=2,
    )
    picks = pick_modes(surface, threshold=threshold)
    return picks, g


def _print_picks_row(picks: dict) -> None:
    """Print one stdout row per mode for the single-gather shape."""
    for mode in ("P", "S", "Stoneley"):
        p = picks.get(mode)
        if p is None:
            continue
        print(
            f"{mode:9s}  {p.slowness / US_PER_FT:18.2f}"
            f"  {1.0 / p.slowness:9.0f}  {p.coherence:9.3f}"
        )


def _cmd_process(argv: list[str]) -> int:
    """Run STC + pick_modes on one or more SEG-Y gathers.

    Single-file mode prints the picks to stdout. Multi-file mode (one
    gather per depth) optionally writes the Vp/Vs/Stoneley-slowness
    log to LAS.

    Examples
    --------
    Quick-look the modes on a single gather (offsets stored in mm)::

        fwap process gather.sgy --offset-scale 1000

    Process a depth sweep (one SEG-Y per depth, sorted lexicographically),
    print a summary table to stdout::

        fwap process depth_*.sgy --depth-start 1000 --depth-step 0.1524

    Same sweep, write a LAS file with DTP/DTS/DTST plus Vp/Vs and
    elastic moduli for downstream petrophysics::

        fwap process depth_*.sgy --depth-start 1000 --depth-step 0.1524 \
            --rho 2400 --output sonic.las

    Read offsets from a non-standard trace header (e.g. ``SourceX``)::

        fwap process gather.sgy --offset-header SourceX --offset-scale 1.0

    Lower the coherence threshold to recover noisy Stoneley picks::

        fwap process gather.sgy --threshold 0.25
    """
    ap = argparse.ArgumentParser(
        prog="fwap process",
        description=(
            "Pick P / S / Stoneley slownesses on one or more "
            "SEG-Y gathers. Multi-file inputs are taken as "
            "one gather per depth; the depth axis is "
            "constructed from --depth-start and --depth-step."
        ),
    )
    ap.add_argument(
        "inputs",
        nargs="+",
        help="One or more SEG-Y files. Multi-file mode "
        "treats each file as a separate depth.",
    )
    ap.add_argument(
        "--offset-header",
        default="offset",
        help="TraceField to read offsets from (default: 'offset').",
    )
    ap.add_argument(
        "--offset-scale",
        type=float,
        default=1.0,
        help=(
            "Divide the raw offset-header value by this "
            "factor to get metres (default: 1.0). "
            "Set to 1000 when offsets are stored in "
            "millimetres."
        ),
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Coherence threshold for peak picking (default: 0.4).",
    )
    ap.add_argument(
        "--depth-start",
        type=float,
        default=0.0,
        help="Depth (m) of the first gather in multi-file mode (default: 0.0).",
    )
    ap.add_argument(
        "--depth-step",
        type=float,
        default=0.1524,
        help="Depth step (m) between successive gathers "
        "(default: 0.1524, i.e. 6 inches).",
    )
    ap.add_argument(
        "--rho",
        type=float,
        default=2400.0,
        help="Formation density (kg/m^3) used to derive "
        "elastic moduli in the LAS output (default: "
        "2400).",
    )
    ap.add_argument(
        "--output",
        help="Path to a LAS output file. When given, "
        "writes DTP / DTS / DTST / VPVS / K / MU / E "
        "/ NU curves over the gather-derived depth "
        "axis.",
    )
    ap.add_argument(
        "--quiet", action="store_true", help="Log only warnings and errors."
    )
    args = ap.parse_args(argv)

    # Physical-bounds validation. Argparse already coerces to float;
    # what's left is to reject values that don't make sense and tell
    # the user concretely which flag they need to fix.
    errors: list[str] = []
    if args.offset_scale <= 0.0:
        errors.append(f"--offset-scale must be > 0 (got {args.offset_scale})")
    if not 0.0 <= args.threshold <= 1.0:
        errors.append(f"--threshold must be in [0, 1] (got {args.threshold})")
    if args.depth_step <= 0.0:
        errors.append(f"--depth-step must be > 0 (got {args.depth_step})")
    if args.rho <= 0.0:
        errors.append(f"--rho must be > 0 (got {args.rho})")
    if errors:
        for msg in errors:
            print(msg, file=sys.stderr)
        return 2

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
    )

    # Single-file mode: preserve the pre-existing stdout behaviour.
    if len(args.inputs) == 1 and args.output is None:
        path = args.inputs[0]
        try:
            picks, g = _pick_one_gather(
                path, args.offset_header, args.offset_scale, args.threshold
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

        logger.info(
            "%s: %d traces, %d samples, dt=%.2e s", path, g.n_traces, g.n_samples, g.dt
        )

        if not picks:
            print(
                f"{path}: no P / S / Stoneley picks above threshold {args.threshold}",
                file=sys.stderr,
            )
            return 1

        print("mode       slowness_us_per_ft  V_m_per_s  coherence")
        _print_picks_row(picks)
        return 0

    # Multi-file mode OR single-file with --output: collect per-depth
    # picks, optionally emit LAS.
    inputs = sorted(args.inputs)
    n = len(inputs)
    depths = args.depth_start + np.arange(n) * args.depth_step
    vp = np.full(n, np.nan)
    vs = np.full(n, np.nan)
    vst = np.full(n, np.nan)
    cohp = np.full(n, np.nan)
    cohs = np.full(n, np.nan)
    cohst = np.full(n, np.nan)

    picks_any = False
    for i, path in enumerate(inputs):
        try:
            picks, _ = _pick_one_gather(
                path, args.offset_header, args.offset_scale, args.threshold
            )
        except ValueError as exc:
            logger.warning("%s: %s", path, exc)
            continue
        if "P" in picks:
            vp[i] = 1.0 / picks["P"].slowness
            cohp[i] = picks["P"].coherence
        if "S" in picks:
            vs[i] = 1.0 / picks["S"].slowness
            cohs[i] = picks["S"].coherence
        if "Stoneley" in picks:
            vst[i] = 1.0 / picks["Stoneley"].slowness
            cohst[i] = picks["Stoneley"].coherence
        if picks:
            picks_any = True
        logger.info(
            "  depth %7.2f m: %s",
            float(depths[i]),
            ", ".join(
                f"{n}={1.0 / picks[n].slowness:.0f} m/s"
                for n in ("P", "S", "Stoneley")
                if n in picks
            )
            or "-- no picks --",
        )

    if not picks_any:
        print(
            "No picks above threshold on any input; nothing to write.", file=sys.stderr
        )
        return 1

    if args.output is None:
        # Multi-file mode without --output prints a summary table.
        print("depth_m  Vp_m_per_s  Vs_m_per_s  Vst_m_per_s")
        for i in range(n):
            print(f"{depths[i]:7.2f}  {vp[i]:10.0f}  {vs[i]:10.0f}  {vst[i]:11.0f}")
        return 0

    # LAS output: include rock-physics curves for the non-NaN rows.
    from fwap.io import write_las
    from fwap.rockphysics import elastic_moduli

    curves = {
        "DTP": 1.0e6 / vp * 0.3048,
        "DTS": 1.0e6 / vs * 0.3048,
        "DTST": 1.0e6 / vst * 0.3048,
        "VPVS": vp / vs,
        "COHP": cohp,
        "COHS": cohs,
        "COHST": cohst,
    }
    for name in ("K", "MU", "E", "NU"):
        curves[name] = np.full(n, np.nan)
    mask = np.isfinite(vp) & np.isfinite(vs)
    if mask.any():
        m = elastic_moduli(vp=vp[mask], vs=vs[mask], rho=np.full(mask.sum(), args.rho))
        curves["K"][mask] = m.k
        curves["MU"][mask] = m.mu
        curves["E"][mask] = m.young
        curves["NU"][mask] = m.poisson

    write_las(
        args.output,
        depths,
        curves,
        well_name="FWAP_PROCESS",
        well={"SRVC": "fwap.cli process"},
    )
    logger.info("Wrote %s (%d depths, %d curves)", args.output, n, len(curves))
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Defaults to the demo runner; if the first argument is ``process``
    the rest is parsed by :func:`_cmd_process` instead.
    """
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] == "process":
        return _cmd_process(argv[1:])

    ap = argparse.ArgumentParser(
        prog="fwap",
        description=(
            "Full-Waveform Acoustic Processing. "
            "Runs synthetic demonstrations of each algorithm "
            "and writes diagnostic figures, or (with "
            "``fwap process FILE``) picks modes on a real "
            "SEG-Y gather."
        ),
    )
    ap.add_argument(
        "demo",
        nargs="?",
        default="all",
        choices=["all", "process"] + list(_DEMOS.keys()),
        help="Which demo to run (default: all). Use "
        "``process`` followed by a SEG-Y path for "
        "real-data picking; see ``fwap process --help``.",
    )
    ap.add_argument(
        "--list-demos",
        action="store_true",
        help="Print the available demo names (one per line) and exit.",
    )
    ap.add_argument(
        "--figdir",
        default="figures",
        help="Output directory for figures (default: figures/).",
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="Open each figure in a window as well as saving.",
    )
    ap.add_argument(
        "--quiet", action="store_true", help="Log only warnings and errors."
    )
    args = ap.parse_args(argv)

    if args.list_demos:
        for name in _DEMOS:
            print(name)
        return 0

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
    )

    targets = list(_DEMOS.keys()) if args.demo == "all" else [args.demo]
    for name in targets:
        _DEMOS[name](figdir=args.figdir, show=args.show)
    logger.info("\nDone. Figures in: %s", os.path.abspath(args.figdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
