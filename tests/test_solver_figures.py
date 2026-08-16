"""
The published figures, and what fwap reproduces of them.

One of six modules split out of ``tests/test_cylindrical_solver.py``.
Every test here ties a solver output to a digitised curve from the
literature -- Schmitt (1988) figures 1a through 17, Schmitt & Cheng
(1987) figures 20 and 21 and their p. 231 slow-formation claim, Yang
et al. (2022) figure 2, and Sinha & Asvadurov's own appendix matrix
used as an independent oracle.

This is the module where a regression shows up as a number moving away
from a published one, rather than as an internal invariant breaking.
It is also the slowest, which is why it is worth being able to select
on its own.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fwap.cylindrical import (
    rayleigh_speed,
    tube_wave_speed,
)
from fwap.cylindrical_solver import (
    BoreholeLayer,
    BoreholeMode,
    _is_isotropic_stiffness,
    _modal_determinant_n0_cased,
    _modal_determinant_n0_vti,
    _modal_determinant_n1,
    _modal_determinant_n1_cased,
    _modal_determinant_n1_vti,
    _radial_wavenumbers_vti,
    _validate_flexural_layers_stacked,
    flexural_dispersion,
    flexural_dispersion_layered,
    flexural_dispersion_vti,
    stoneley_dispersion,
    stoneley_dispersion_layered,
    stoneley_dispersion_vti,
    trapped_pseudo_rayleigh_dispersion,
    trapped_pseudo_rayleigh_dispersion_layered,
)
from tests._solver_media import (
    _A2_CASING,
    _A2_CEMENT,
    _LC_SLOW,
    _PR_SINHA_FAST,
    _QUAD_FAST,
    _QUAD_FLUID,
    _SC87_SLOW,
    _descends,
    _gap_sweep_layers,
    _green_river_shale_stiffness,
    _sc87_stack,
)

# A.2, measured: the fast-formation bracket is anchored to the wrong speed
#
# `_flexural_dispersion_fast_formation` searches phase velocity in
# `(V_R, V_S)`. V_R is not a limit of this mode. The flexural branch
# asymptotes to the *Scholte* speed, which is well below V_R -- for the
# rock here, 1472.6 against 2115.8, so the bracket excludes 30 % of the
# velocity axis and truncates the fundamental where it crosses V_R.
#
# Two consequences, and the second is the serious one:
#
#   * below the crossing the mode is found, above it the bracket is empty
#     and the call returns NaN -- which reads as "sparse coverage", the
#     symptom A.2 was filed under;
#   * in between, the window still contains roots, but they are flexural
#     *overtones* entering near V_S. The solver returns one of those,
#     labelled as the flexural mode, with no indication anything is wrong.
#
# Enumerating Im(det) roots over (Scholte, V_S) at 19.5 kHz gives 1853 and
# 2269 m/s. The fundamental -- continued from 2138 m/s at 14.5 kHz, where
# the current bracket still works and only one root exists -- is 1853. The
# solver returns 2269.
#
# Pinned rather than fixed. A correct fix must identify the fundamental
# among several roots, and neither naive rule works: taking the highest
# seeds onto an overtone, taking the lowest is non-monotone on 2 of 3 test
# rocks. See roadmap A.2.
# ----------------------------------------------------------------------

#: The fast-formation marcher accepts a root up to
#: ``_FAST_FLEXURAL_STEP_UP`` (5e-4) above the previous one, absorbing
#: brentq jitter without admitting a faster branch. Monotonicity is
#: therefore asserted to that tolerance, not to exact equality.


_A2_ROCK = dict(vp=4000.0, vs=2300.0, rho=2500.0)
_A2_FLUID = dict(vf=1500.0, rho_f=1000.0)


def test_fast_flexural_bracket_excludes_the_modes_own_asymptote():
    """V_R is not a limit of this mode; the Scholte speed is.

    The bracket's lower edge sits above the value the mode converges to,
    so the search window cannot contain the high-frequency branch at all.
    """
    from fwap import scholte_speed
    from fwap.cylindrical import rayleigh_speed

    v_rayleigh = rayleigh_speed(_A2_ROCK["vp"], _A2_ROCK["vs"])
    v_scholte = scholte_speed(**_A2_ROCK, **_A2_FLUID)

    assert v_scholte < v_rayleigh, "the asymptote must be inside the bracket"
    assert (v_rayleigh - v_scholte) / v_rayleigh > 0.25


def test_fast_flexural_returns_the_fundamental_above_the_crossing():
    """A.2 at the frequency that named it -- and the third value asked for.

    This test has now named three different answers at 19.5 kHz. It
    first asserted ~2269 m/s, the ``(V_R, V_S)`` window's pick. A.2
    corrected the window to ``(V_f, V_S)`` and it asserted **1853**,
    on the reasoning that the determinant "has roots near 1853 and
    2269, and the fundamental is 1853".

    Tracing every root against frequency shows 1853 is an overtone.
    The fundamental runs 2242 (4 kHz), 1816 (6), 1625 (8), 1554 (10),
    1520 (12), 1501 (14) -- one smooth descent -- and **crosses ``V_f``
    between 14 and 15 kHz**, continuing 1495.5, 1490.7, 1487.0 and on
    toward Scholte at 1472.6. The 1853 root does not belong to it: it
    is absent below 18 kHz and appears at ``V_S`` at 18 kHz, then
    descends 1939, 1884, 1837. A branch cannot enter from ``V_S`` at
    18 kHz and also be the mode that was at 1554 at 10 kHz.

    So the window was right and the frequency was past its end. The
    fundamental at 19.5 kHz is 1481.6 m/s, below the fluid, and the
    ordering below is what distinguishes the branches rather than any
    single remembered number.
    """
    from fwap import flexural_dispersion, scholte_speed

    result = flexural_dispersion(np.array([19.5e3]), **_A2_ROCK, **_A2_FLUID, a=0.10)
    velocity = 1.0 / result.slowness[0]
    floor = scholte_speed(**_A2_ROCK, **_A2_FLUID)

    assert np.isfinite(velocity)
    assert velocity == pytest.approx(1481.6, rel=0.01)
    assert velocity < _A2_FLUID["vf"], "the fundamental has crossed by 19.5 kHz"
    assert velocity > floor * 0.999, "and it is heading for Scholte, not past it"

    # The branch is continuous across the crossing, which is what makes
    # the sub-fluid root the same mode rather than a new one.
    sweep = np.arange(6.0e3, 24.1e3, 1.0e3)
    track = 1.0 / flexural_dispersion(sweep, **_A2_ROCK, **_A2_FLUID, a=0.10).slowness
    assert np.all(np.isfinite(track))
    assert np.all(np.diff(track) < 0.0), track
    assert track[0] > _A2_FLUID["vf"] and track[-1] < _A2_FLUID["vf"]


def test_fast_flexural_returns_a_curve_not_a_sawtooth():
    """The clearest statement of A.2, now inverted into its fix.

    Over 10-30 kHz the returned velocity used to descend, go NaN, jump
    **back up** by more than 100 m/s, descend again, and repeat -- each
    overtone entering the ``(V_R, V_S)`` window near ``V_S``, crossing
    it, and dropping out below ``V_R``. A guided mode's phase velocity
    never increases with frequency, so what came back was not a curve.

    It is now one branch: contiguous coverage, monotone descent, no
    upward step anywhere.
    """
    from fwap import flexural_dispersion

    freq = np.linspace(10.0e3, 30.0e3, 21)
    result = flexural_dispersion(freq, **_A2_ROCK, **_A2_FLUID, a=0.10)
    velocity = 1.0 / result.slowness
    finite = np.isfinite(velocity)

    assert finite.any(), "the mode is found over part of this band"
    idx = np.where(finite)[0]
    assert np.array_equal(idx, np.arange(idx[0], idx[-1] + 1)), (
        "coverage is one contiguous run, not stitched from several modes"
    )
    steps = np.diff(velocity[finite])
    assert np.all(steps <= 0.0), f"phase velocity never increases; got {steps}"


# ----------------------------------------------------------------------
# A.2, checked against a published curve
#
# Everything above is internal reasoning: the bracket is anchored to the
# wrong speed, so the roots it returns are overtones. That argument stands
# on fwap's own determinant. Schmitt & Cheng figure 2a plots the
# same quantity for a stated rock, so it can be checked from outside.
#
# Provenance. Schmitt, D. P., & Cheng, C. H., "Shear Wave Logging In
# (Multilayered) Elastic Formations: An Overview", MIT Earth Resources
# Laboratory, pp. 213-268 -- *not* the single-author JASA article this
# repository cites elsewhere. Figure 2a is on p. 239 of the bound volume.
# "Dipole source. Dispersion (a), attenuation (b), and excitation
# (c) of the flexural mode (1) and the first trapped mode (2) in the
# presence of a fast sandstone. The velocities are normalized with respect
# to the bore fluid velocity." Rock from the paper's table 1 (fast
# sandstone): V_P 4878, V_S 2601, rho 2160; bore fluid 1500 m/s, 1000
# kg/m^3; hole radius 0.10 m.
#
# Digitisation. The page was rendered at 600 dpi, the plot frame located
# from the axis rules, and the phase branch of curve 1 followed column by
# column. The 26 x-axis ticks land on integer kHz to within 0.06 kHz and
# the 1.400 / 1.000 y-ticks read 1.3978 / 0.9981, so axis calibration
# contributes about +-3 m/s. The plotted curve is 9-12 px thick, which
# dominates: the table below is good to roughly +-20 m/s, or +-1 %.
#
# Two things the figure settles that fwap could not settle by itself:
#
#   * the low-frequency plateau is the formation shear speed (2593 read,
#     2601 exact) -- as expected, and confirmation the digitisation is
#     sound;
#   * the high-frequency end is the Scholte speed (1493 at 24.9 kHz and
#     still descending, against 1484 exact) -- which is A.1's claim,
#     until now resting only on fwap's own convergence.
#
# And the number this block exists for: the branch crosses V_R at
# 4.45 kHz and V_f at 17.9 kHz. The solver's `(V_R, V_S)` window
# therefore holds the true root over 10 % of the plotted band and no
# root-finder tolerance can recover the rest.
# ----------------------------------------------------------------------

_FIG2_ROCK = dict(vp=4878.0, vs=2601.0, rho=2160.0)
_FIG2_FLUID = dict(vf=1500.0, rho_f=1000.0)

#: Phase velocity (Hz, m/s) of the flexural mode, digitised from
#: Schmitt & Cheng figure 2a. Uncertainty about +-20 m/s.
_FIG2A_FLEXURAL_PHASE = (
    (2.5e3, 2593.3),
    (3.0e3, 2595.5),
    (4.0e3, 2534.2),
    (5.0e3, 2175.4),
    (6.0e3, 1889.3),
    (8.0e3, 1663.8),
    (10.0e3, 1579.0),
    (12.5e3, 1535.8),
    (15.0e3, 1517.7),
    (17.5e3, 1504.0),
    (20.0e3, 1495.0),
    (22.5e3, 1495.0),
    (24.5e3, 1492.7),
)


def test_figure_2a_reference_table_is_anchored_at_both_ends():
    """Check the digitisation before trusting it to judge the solver.

    Both ends of the published curve are values that can be computed
    independently, so they are the two places a mis-read axis would show
    up. The low-frequency plateau must be the formation shear speed and
    the high-frequency end must be approaching the Scholte speed from
    above -- and the two are 1117 m/s apart, so neither is a weak test.
    """
    from fwap import scholte_speed

    freq = np.array([f for f, _ in _FIG2A_FLEXURAL_PHASE])
    velocity = np.array([v for _, v in _FIG2A_FLEXURAL_PHASE])
    v_scholte = scholte_speed(**_FIG2_ROCK, **_FIG2_FLUID)

    assert velocity[0] / _FIG2_ROCK["vs"] == pytest.approx(1.0, abs=0.01)
    assert velocity[-1] / v_scholte == pytest.approx(1.0, abs=0.015)
    assert velocity[-1] > v_scholte, "the curve is still descending at 24.5 kHz"
    # Monotone to within the +-20 m/s tracing uncertainty: the 2.5 -> 3.0 kHz
    # pair rises 2.2 m/s, which is the flat plateau read twice, not an ascent.
    assert np.diff(velocity).max() < 20.0, "phase velocity does not increase here"
    assert freq[0] < 3.0e3 < freq[-1]


def test_fast_flexural_matches_the_published_curve():
    """The published check of A.2, then of A.8.

    On the paper's own rock the solver used to answer at 5 of the 13
    tabulated frequencies, every one between ``V_R`` and ``V_S`` and
    every one 62-73 % faster than the figure. A.2 corrected the search
    window to ``(V_f, V_S)`` and took it to 0.78 % median over
    5.0-15.0 kHz. A.8 corrected the SV column and took it to **0.16 %
    median, 0.37 % worst, over 3.0-17.5 kHz** -- 9 of the 13 points,
    and well inside the figure's own +-1 % digitisation floor.

    Two of those points sit ABOVE ``V_R``: the low-frequency plateau is
    at the formation shear speed, and ``V_R`` was never a limit of this
    mode. The old bracket could not reach them by construction.

    **The figure also settles the ``V_f`` question against the test
    that used to be here.** This asserted ``velocity > V_f``, while
    scoring against a curve whose own last three points are 1495.0,
    1495.0 and 1492.7 m/s -- all *below* the 1500 m/s fluid. The
    assertion contradicted the reference it was checking, and survived
    only because those points were NaN and so never compared. With the
    sub-fluid continuation they are compared, at 0.02 %, 0.30 % and
    0.30 %: coverage goes 9 of 13 points to 12, with the median error
    unmoved at 0.16 %.

    Below the plateau it still returns NaN rather than a guess -- there
    the mode has not formed. That is the one remaining gap, at 2.5 kHz.
    """
    from fwap import flexural_dispersion, scholte_speed

    freq = np.array([f for f, _ in _FIG2A_FLEXURAL_PHASE])
    reference = np.array([v for _, v in _FIG2A_FLEXURAL_PHASE])
    velocity = (
        1.0 / flexural_dispersion(freq, **_FIG2_ROCK, **_FIG2_FLUID, a=0.10).slowness
    )
    finite = np.isfinite(velocity)

    assert finite.sum() >= 12, f"expected all but the plateau; got {finite.sum()}"
    assert np.all(velocity[finite] <= _FIG2_ROCK["vs"])
    floor = scholte_speed(**_FIG2_ROCK, **_FIG2_FLUID)
    assert np.all(velocity[finite] > floor * 0.999)
    # The published curve itself crosses V_f, and so does fwap.
    assert (reference < _FIG2_FLUID["vf"]).sum() >= 3
    assert (velocity[finite] < _FIG2_FLUID["vf"]).sum() >= 3

    error = np.abs((velocity[finite] - reference[finite]) / reference[finite])
    assert np.median(error) < 0.004, f"median {np.median(error):.2%}"
    assert error.max() < 0.008, f"worst {error.max():.2%}"

    # and the answers reach above V_R, where the old bracket could not
    v_rayleigh = rayleigh_speed(_FIG2_ROCK["vp"], _FIG2_ROCK["vs"])
    assert (velocity[finite] > v_rayleigh).sum() >= 2


# ----------------------------------------------------------------------
# A.2, generalised: figure 7a puts three fast formations on one axis
#
# Figure 2a settles the fast sandstone. Figure 7a of the same report
# (p. 244) plots the flexural mode for **granite (1), limestone (2) and
# the fast sandstone (3)** together, 0-15 kHz, so the defect can be
# measured against formation stiffness rather than at one rock.
#
# Digitised the same way, at 600 dpi, with the axes least-squares fitted
# to the tick marks rather than to the frame rules: 15 x-ticks residual
# to 0.018 kHz, 4 y-ticks residual to 0.0004 normalised (0.5 m/s). Axis
# calibration is negligible here; the line thickness still dominates.
#
# Three results, and the second is the one worth having.
#
#   1. All three plateaus land on the formation shear speed: 3749.6,
#      2768.7, 2597.7 against 3750, 2771, 2601. Three anchors, not one.
#
#   2. All three cross V_R at **4.43-4.45 kHz** -- although V_R spans
#      2413 to 3388 m/s and V_S spans 2601 to 3750. The frequency at
#      which the solver's bracket stops containing the mode is set by
#      the hole and the fluid, not by the rock. Figure 2a gave 4.45 kHz
#      for the sandstone independently, so that is four consistent
#      readings. Note this is *not* because the curves are self-similar
#      in v/V_S -- at 5 kHz they read 0.690, 0.818, 0.838 -- so two
#      things vary and happen to cancel. Measured, not explained.
#
#   3. The error grows with stiffness: median +62 % (sandstone), +72 %
#      (limestone), +134 % (granite), because the (V_R, V_S) window
#      rides further above the true curve the faster the rock. The
#      defect is worst exactly where dipole logging most needs it.
#
# Resolution limit, stated because the tables below stop where it bites:
# limestone and sandstone become one plotted line at 5.75 kHz, and
# granite joins them at 10.25 kHz. Past those points a column trace
# reports the band centre, not a curve. Tabulated only where resolved.
# ----------------------------------------------------------------------

_FIG7_ROCKS = {
    "granite": dict(vp=5881.0, vs=3750.0, rho=2160.0),
    "limestone": dict(vp=5081.0, vs=2771.0, rho=2160.0),
}

#: Phase velocity (Hz, m/s) of the flexural mode, digitised from
#: Schmitt & Cheng figure 7a, over the band where each curve is a
#: separate line. Uncertainty about +-20 m/s.
_FIG7A_FLEXURAL_PHASE = {
    "granite": (
        (3.0e3, 3752.9),
        (3.5e3, 3752.9),
        (4.0e3, 3723.2),
        (4.25e3, 3616.1),
        (4.5e3, 3309.3),
        (4.75e3, 2857.8),
        (5.0e3, 2588.4),
        (5.5e3, 2234.7),
        (6.0e3, 2043.8),
        (7.0e3, 1843.3),
        (8.0e3, 1739.1),
        (9.0e3, 1672.3),
        (10.0e3, 1618.6),
    ),
    "limestone": (
        (2.5e3, 2766.2),
        (3.0e3, 2750.3),
        (3.5e3, 2762.2),
        (4.0e3, 2719.6),
        (4.25e3, 2638.7),
        (4.5e3, 2531.2),
        (4.75e3, 2394.4),
        (5.0e3, 2266.9),
        (5.5e3, 2052.2),
    ),
}


@pytest.mark.parametrize("name", sorted(_FIG7_ROCKS))
def test_figure_7a_tables_start_at_the_formation_shear_speed(name):
    """Anchor each figure-7a table before it judges anything.

    Two independently computable values bracket every one of these
    curves -- the shear speed it leaves and the Scholte speed it heads
    for -- and the low-frequency end is the one the table reaches. Three
    rocks spanning 2601-3750 m/s all hitting their own V_S is a much
    stronger check on the digitisation than one rock doing it.
    """
    rock = _FIG7_ROCKS[name]
    table = _FIG7A_FLEXURAL_PHASE[name]
    velocity = np.array([v for _, v in table])

    assert velocity[0] / rock["vs"] == pytest.approx(1.0, abs=0.01)
    assert np.diff(velocity).max() < 20.0, "phase velocity does not increase"
    assert velocity[-1] < 0.75 * rock["vs"], "the table must cover the plunge"


@pytest.mark.parametrize("name", sorted(_FIG7_ROCKS))
def test_the_bracket_empties_near_4_4_khz_whatever_the_formation(name):
    """The sharpest statement of A.2's first defect.

    `_flexural_dispersion_fast_formation` searches phase velocity in
    `(V_R, V_S)`. Figure 7a shows all three formations leaving that
    window between 4.43 and 4.45 kHz, though V_R spans 2413 to 3388 m/s.
    So the bracket does not fail at a rock-dependent frequency that a
    caller could reason about -- it fails at the same place for every
    fast formation, because what sets it is the hole and the fluid.
    """
    from fwap.cylindrical import rayleigh_speed

    rock = _FIG7_ROCKS[name]
    table = _FIG7A_FLEXURAL_PHASE[name]
    freq = np.array([f for f, _ in table])
    velocity = np.array([v for _, v in table])
    v_rayleigh = rayleigh_speed(rock["vp"], rock["vs"])

    assert velocity[0] > v_rayleigh > velocity[-1], "the table must span V_R"
    crossing = np.interp(-v_rayleigh, -velocity, freq)
    assert 4.3e3 < crossing < 4.6e3, f"{name} leaves the bracket at {crossing:.0f} Hz"


@pytest.mark.parametrize("name", sorted(_FIG7_ROCKS))
def test_the_solver_now_answers_over_the_band_figure_7a_resolves(name):
    """Where the figure is most informative, the solver used to say
    nothing -- and for granite, nothing at all.

    Over the tabulated band `flexural_dispersion` returned NaN at every
    frequency above the 4.45 kHz crossing for both rocks, and at **all
    13** for granite. It now answers over most of each band, at the
    digitisation floor: 1.03 % median for limestone, 0.87 % for granite.
    """
    from fwap import flexural_dispersion

    rock = _FIG7_ROCKS[name]
    table = _FIG7A_FLEXURAL_PHASE[name]
    freq = np.array([f for f, _ in table])
    reference = np.array([v for _, v in table])
    fluid = dict(vf=1500.0, rho_f=1000.0)
    velocity = 1.0 / flexural_dispersion(freq, **rock, **fluid, a=0.10).slowness
    finite = np.isfinite(velocity)

    assert finite.any(), f"{name}: still empty"
    assert finite[freq > 4.6e3].any(), "answers survive above the crossing now"
    error = np.abs((velocity[finite] - reference[finite]) / reference[finite])
    assert np.median(error) < 0.02, f"{name}: median {np.median(error):.2%}"
    if name == "granite":
        assert finite.sum() >= 8, "granite was empty across the whole band"


#: Centre of the single line the three figure-7a curves have collapsed
#: into by 11 kHz (Hz, m/s). Not a per-rock reading -- the band is about
#: 30 m/s wide there, so treat these as +-2 %.
_FIG7A_MERGED_BAND = (
    (11.0e3, 1581.0),
    (11.5e3, 1577.0),
    (12.0e3, 1569.0),
    (12.5e3, 1565.0),
    (13.0e3, 1557.0),
)


def test_fast_flexural_error_no_longer_grows_with_formation_stiffness():
    """The reason figure 7a was worth digitising as well as figure 2a,
    and the sharpest single measure of the fix.

    At 11-13 kHz all three published curves have converged to one line
    near 1570 m/s. The solver used to read about 2700 m/s in limestone
    and 3650 in granite -- **+69 % and +124 %** -- because the
    ``(V_R, V_S)`` window rides further above the true curve the faster
    the rock. So the defect was worst exactly where dipole logging most
    needs the answer.

    All three rocks now land on the merged line within the +-2 % reading
    uncertainty of that band, and the stiffness ordering is gone:
    granite is now the *closest*, not the worst.
    """
    from fwap import flexural_dispersion

    freq = np.array([f for f, _ in _FIG7A_MERGED_BAND])
    reference = np.array([v for _, v in _FIG7A_MERGED_BAND])
    fluid = dict(vf=1500.0, rho_f=1000.0)

    error = {}
    for name, rock in _FIG7_ROCKS.items():
        velocity = 1.0 / flexural_dispersion(freq, **rock, **fluid, a=0.10).slowness
        assert np.isfinite(velocity).all(), f"{name}: expected full coverage here"
        error[name] = np.abs((velocity - reference) / reference)

    for name, err in error.items():
        assert err.max() < 0.03, f"{name}: worst {err.max():.2%}"
    assert error["granite"].max() < 0.02


def test_the_two_figures_agree_where_both_resolve_the_same_rock():
    """A check on the digitisation that owes nothing to fwap.

    The fast sandstone is plotted twice in the same report, on different
    pages, with different axis ranges -- figure 2a spans 0.600-1.800
    over 0-25 kHz, figure 7a spans 0.500-2.600 over 0-15 kHz. Two
    independent calibrations of the same physical curve.

    Below 5.75 kHz, where figure 7a still draws the sandstone as its own
    line, the two reads agree to better than 0.5 %. Above it the
    sandstone and limestone curves become one line in figure 7a, which
    is why nothing from that region is tabulated.
    """
    fig2 = {f: v for f, v in _FIG2A_FLEXURAL_PHASE if f <= 5.5e3}
    assert fig2, "figure 2a must sample the band figure 7a resolves"

    # Figure 7a, fast sandstone, read at figure 2a's own frequencies.
    fig7_sandstone = {3.0e3: 2589.5, 4.0e3: 2535.1, 5.0e3: 2179.3}
    for freq, v7 in fig7_sandstone.items():
        v2 = fig2[freq]
        assert v7 / v2 == pytest.approx(1.0, abs=0.005), (
            f"the two figures disagree at {freq / 1e3:.0f} kHz: {v2} vs {v7}"
        )


# ----------------------------------------------------------------------
# A.2 at n = 2: figure 7b, the claim this file has been asserting
#
# The roadmap has said "affects n=2 identically, so one fix repairs two
# solvers" since the item was re-diagnosed, on the strength of a
# non-monotone scatter between V_R and V_S. Figure 7b plots the screw
# (quadrupole) mode for the same three formations over 4-20 kHz, so the
# claim can be checked rather than asserted.
#
# It holds, with one difference that makes n=2 the more dangerous of the
# two. Measured over each rock's plotted band at 0.2 kHz:
#
#   rock       coverage   within 5 %   the rest
#   granite      75 %      1 point     +5 % to +136 %, median +102 %
#   limestone    66 %      none        +11 % to +66 %, median  +57 %
#   sandstone    65 %      none        +13 % to +56 %, median  +46 %
#
# Every finite value again lies strictly inside (V_R, V_S) -- in fact
# the returned values sweep that window end to end (3389-3750,
# 2565-2771, 2413-2601). Same stiffness ordering as n=1.
#
# The difference: coverage is **65-75 % here against 21-36 % at n=1**.
# A caller who filters on NaN therefore keeps two to three times as many
# wrong answers from the quadrupole solver as from the flexural one.
#
# And the bracket empties at a mode-specific, rock-independent
# frequency: 7.53 / 7.61 / 7.69 kHz here, against 4.45 / 4.43 / 4.43 kHz
# for the flexural mode, while V_R spans 2413 to 3388 m/s in both.
# ----------------------------------------------------------------------

#: Screw (quadrupole) phase velocity (Hz, m/s) from figure 7b, over the
#: band where each curve is a separate line. Uncertainty about +-20 m/s.
_FIG7B_SCREW_PHASE = {
    "granite": (
        (6.9e3, 3771.5),
        (7.0e3, 3751.7),
        (7.5e3, 3417.1),
        (8.0e3, 2896.5),
        (9.0e3, 2342.4),
        (10.0e3, 2068.0),
        (12.0e3, 1829.2),
        (14.0e3, 1718.1),
    ),
    "limestone": (
        (6.6e3, 2774.6),
        (7.0e3, 2727.1),
        (7.5e3, 2607.9),
        (8.0e3, 2440.5),
        (9.0e3, 2148.0),
        (9.5e3, 2037.8),
    ),
}


@pytest.mark.parametrize("name", sorted(_FIG7B_SCREW_PHASE))
def test_the_quadrupole_bracket_empties_near_7_6_khz(name):
    """The `n=2` half of A.2, checked instead of asserted.

    Same shape as the flexural result and the same rock-independence:
    all three formations leave `(V_R, V_S)` between 7.53 and 7.69 kHz.
    The emptying frequency is a property of the mode and the hole, not
    of the formation -- 4.4 kHz at `n=1`, 7.6 kHz at `n=2`.
    """
    from fwap.cylindrical import rayleigh_speed

    rock = _FIG7_ROCKS[name]
    table = _FIG7B_SCREW_PHASE[name]
    freq = np.array([f for f, _ in table])
    velocity = np.array([v for _, v in table])
    v_rayleigh = rayleigh_speed(rock["vp"], rock["vs"])

    assert velocity[0] / rock["vs"] == pytest.approx(1.0, abs=0.01)
    assert np.diff(velocity).max() < 20.0
    assert velocity[0] > v_rayleigh > velocity[-1], "the table must span V_R"
    crossing = np.interp(-v_rayleigh, -velocity, freq)
    assert 7.4e3 < crossing < 7.8e3, f"{name} leaves the bracket at {crossing:.0f} Hz"


@pytest.mark.parametrize("name", sorted(_FIG7B_SCREW_PHASE))
def test_the_quadrupole_branch_is_corrected_but_still_arrives_late(name):
    """What the A.2 fix does and does not buy at `n=2`, open hole.

    `quadrupole_dispersion` used to hand back 65-75 % coverage of
    overtones from the `(V_R, V_S)` window, tens of percent fast, which
    made a `NaN` filter a *worse* guard at `n=2` than at `n=1`. The
    branch is now the fundamental and descends monotonically below
    `V_R`.

    The residual is rock-dependent and is not the bracket: granite
    lands on the published screw curve at about 2.6 % median, while
    limestone resolves at only one frequency in the traced band.

    Roadmap A.8 corrected the SV column and improved every
    well-conditioned tie in this file by an order of magnitude. It did
    not help here, and cost coverage: granite 22 -> 14 converged points
    of 72 and 2.0 -> 2.6 % median; limestone 11 -> 1 of 30 and
    8.6 -> 12.8 %. That is consistent with A.7's diagnosis rather than
    at odds with it -- the n=2 fast-formation determinant is
    noise-dominated, so which spurious crossings the marcher latches
    onto changes with any change to the determinant. The n=1 twin of
    this comparison went the other way on the same rocks: granite
    1.24 -> 0.45 % median with coverage 57 -> 68, limestone
    1.39 -> 0.31 % with 14 -> 29.
    """
    from fwap import quadrupole_dispersion

    rock = _FIG7_ROCKS[name]
    table = _FIG7B_SCREW_PHASE[name]
    ref_f = np.array([f for f, _ in table])
    ref_v = np.array([v for _, v in table])
    fluid = dict(vf=1500.0, rho_f=1000.0)

    grid = np.arange(7.9e3, ref_f[-1] + 1.0, 100.0)
    screw = 1.0 / quadrupole_dispersion(grid, **rock, **fluid, a=0.10).slowness
    finite = np.isfinite(screw)
    coverage = {"granite": 8, "limestone": 1}[name]
    assert finite.sum() >= coverage, f"{name}: {finite.sum()} converged"

    assert np.all(screw[finite] < rock["vs"])
    assert np.all(screw[finite] > fluid["vf"])
    assert _descends(screw[finite]), "one descending branch"

    error = np.abs(screw[finite] / np.interp(grid[finite], ref_f, ref_v) - 1.0)
    ceiling = {"granite": 0.04, "limestone": 0.14}[name]
    assert np.median(error) < ceiling, f"{name}: median {100 * np.median(error):.1f} %"


# ----------------------------------------------------------------------
# Figure 8a: the slow formation, where the solvers are supposed to work
#
# Everything above measures a defect. Figure 8a (p. 245) is the other
# kind of check: "Slow sandstone. Dispersion and attenuation of the
# Stoneley wave (0), the flexural (1) and screw (2) modes excited by a
# monopole, dipole, and quadrupole source respectively." One panel,
# three published curves, three fwap solvers, on the path this project
# has always claimed works -- and never checked against anything but
# itself.
#
# The rock is table 1's slow sandstone: V_P 2751, V_S 1201, rho 2100,
# with the same 1500 m/s / 1000 kg/m^3 bore fluid.
#
# A note on the axis, because the scan is ambiguous and a careless read
# costs 0.5 %. The y labels print as 0.850 / 0.783 / 0.71? / 0.650, and
# the third glyph degrades to something like "0.713". It is 0.71667: the
# four tick rows are evenly spaced (393.5, 395.0, 396.5 px), and fitting
# the evenly divided values gives a residual of +-0.00013 against
# +-0.0026 for the literal reading -- twenty times worse and structured.
# The same package prints 0.667 and 0.783 for an evenly divided
# 0.550-0.900 axis in the neighbouring panel.
#
# Three curves resolve as three *disjoint* connected components, so no
# branch tracking was needed here; the median ink row per column is the
# curve. The narrow 0.650-0.850 axis also makes this the most precise of
# the four figures: the plotted line is worth about +-3 m/s, or +-0.3 %.
#
# Results, over 0.1-14.9 kHz at 0.25 kHz:
#
#   Stoneley   59/59 finite,  rms 0.04 %,  worst 0.08 %
#   flexural   49/55 finite,  rms 1.29 %,  worst -1.84 % at 5.2 kHz
#   screw      38/44 finite,  rms 0.94 %,  worst -0.56 %  (+3.1 % near
#                                                          the cutoff)
#
# The Stoneley agreement is *below the resolution of the figure*: fwap
# and the published curve cannot be told apart. That is this project's
# first external tie for `stoneley_dispersion`, and it is 60x inside the
# 5 % overlay budget A.1 set for digitised figures.
#
# The flexural number is not at the resolution limit. It is a real,
# small, systematic offset -- zero near 3.3 kHz, -1.8 % at 5-6 kHz,
# recovering to -0.8 % by 14 kHz -- and the Stoneley curve on the same
# panel, read with the same calibration, bounds the reading error at
# 0.08 %. It is not the borehole radius either (see the radius test).
# One candidate is that the paper's model is viscoelastic where fwap's
# open-hole solvers are elastic: table 1 carries Q_alpha and Q_beta, and
# figure 8's own attenuation panel gives all three modes 1/Q ~ 0.02. But
# that should move the Stoneley too, and it does not, so the candidate
# is not confirmed. Recorded as measured and unexplained.
# ----------------------------------------------------------------------

_FIG8_ROCK = dict(vp=2751.0, vs=1201.0, rho=2100.0)

#: Phase velocity (Hz, m/s) of the three modes in the slow sandstone,
#: digitised from Schmitt & Cheng figure 8a. About +-3 m/s.
_FIG8A_PHASE = {
    "stoneley": (
        (0.5e3, 1126.5),
        (1.5e3, 1100.3),
        (2.0e3, 1091.1),
        (3.0e3, 1076.7),
        (4.0e3, 1066.8),
        (5.0e3, 1059.6),
        (6.0e3, 1053.9),
        (8.0e3, 1046.3),
        (10.0e3, 1041.0),
        (12.0e3, 1037.2),
        (14.0e3, 1034.9),
    ),
    "flexural": (
        (3.0e3, 1163.0),
        (4.0e3, 1126.8),
        (5.0e3, 1099.5),
        (6.0e3, 1081.3),
        (8.0e3, 1061.5),
        (10.0e3, 1050.9),
        (12.0e3, 1044.8),
        (14.0e3, 1040.5),
    ),
    "screw": (
        (6.0e3, 1143.2),
        (8.0e3, 1104.8),
        (10.0e3, 1081.3),
        (12.0e3, 1066.8),
        (14.0e3, 1057.3),
    ),
}

#: Where each published curve begins, and the value it begins at (kHz,
#: m/s). The Stoneley exists at all frequencies; the two shear modes
#: start at the formation shear speed.
_FIG8A_ONSET = {
    "stoneley": (0.078, 1135.6),
    "flexural": (1.04, 1201.4),
    "screw": (3.74, 1201.4),
}


def test_figure_8a_is_anchored_on_three_closed_forms():
    """Three independent anchors, none of which needs a solver.

    The Stoneley's low-frequency limit is the tube-wave speed, which is
    a one-line formula, and both shear modes leave the axis at the
    formation shear speed. All three land, so the axis calibration --
    including the ambiguous 0.71667 label -- is sound.
    """
    from fwap import tube_wave_speed

    fluid = dict(vf=1500.0, rho_f=1000.0)
    v_tube = tube_wave_speed(_FIG8_ROCK["vs"], _FIG8_ROCK["rho"], **fluid)

    assert _FIG8A_ONSET["stoneley"][1] / v_tube == pytest.approx(1.0, abs=0.005)
    for mode in ("flexural", "screw"):
        assert _FIG8A_ONSET[mode][1] / _FIG8_ROCK["vs"] == pytest.approx(1.0, abs=0.005)


def test_stoneley_matches_the_published_slow_formation_curve():
    """The validation this project has been missing.

    `stoneley_dispersion` against figure 8a over 0.5-14 kHz: every point
    finite, every point inside 0.1 %. The plotted line is worth about
    0.3 %, so this is agreement below what the figure can resolve --
    the first time any fwap solver has been tied to published data at
    better than 1 %.
    """
    from fwap import stoneley_dispersion

    table = _FIG8A_PHASE["stoneley"]
    freq = np.array([f for f, _ in table])
    reference = np.array([v for _, v in table])
    fluid = dict(vf=1500.0, rho_f=1000.0)
    velocity = 1.0 / stoneley_dispersion(freq, **_FIG8_ROCK, **fluid, a=0.10).slowness

    assert np.isfinite(velocity).all(), "the slow-formation Stoneley never drops out"
    error = np.abs(velocity / reference - 1.0)
    assert error.max() < 0.005, f"worst point {100 * error.max():.2f} %"
    assert np.sqrt((error**2).mean()) < 0.002


def test_the_stoneley_curve_pins_the_borehole_radius():
    """Why every figure-8a comparison may assume `a` = 0.10 m.

    The paper's table 1 gives velocities and densities but no hole
    radius, so 0.10 m is an assumption -- and one the rest of this block
    leans on. The Stoneley curve settles it: its RMS misfit is 0.05 % at
    0.100 m and degrades either side (0.13 % at 0.095, 0.14 % at 0.105).
    The flexural offset recorded above is therefore not a radius error;
    at the radius the Stoneley pins, the flexural mode is still 1.2 %
    slow, and no radius makes it better than about 1 %.
    """
    from fwap import stoneley_dispersion

    table = _FIG8A_PHASE["stoneley"]
    freq = np.array([f for f, _ in table])
    reference = np.array([v for _, v in table])
    fluid = dict(vf=1500.0, rho_f=1000.0)

    def rms(a):
        v = 1.0 / stoneley_dispersion(freq, **_FIG8_ROCK, **fluid, a=a).slowness
        return float(np.sqrt(((v / reference - 1.0) ** 2).mean()))

    radii = [0.090, 0.095, 0.100, 0.105, 0.110]
    scores = [rms(a) for a in radii]
    assert radii[int(np.argmin(scores))] == 0.100, dict(zip(radii, scores))
    assert scores[2] < 0.5 * min(scores[1], scores[3]), "the minimum must be sharp"


@pytest.mark.parametrize("mode", ["flexural", "screw"])
def test_both_shear_solvers_lose_the_same_1_5_khz_above_cutoff(mode):
    """The near-cutoff gap was the SV column, and it has closed.

    This test used to record a **1.48 kHz** flexural and **1.52 kHz**
    screw gap between the published onset and fwap's first root -- the
    same width for two modes whose cutoffs are 2.7 kHz apart, which
    read like a quantity set by the hole. It was not: it was the
    roadmap-A.8 SV column. Corrected, in the same slow sandstone,

        flexural   published 1.04 kHz   fwap 1.04   gap  0.00 kHz
        screw      published 3.74 kHz   fwap 3.86   gap +0.12 kHz

    Both solvers now switch on at the published onset, to within one
    step of this grid, and stay continuous above it.
    """
    from fwap import flexural_dispersion, quadrupole_dispersion

    solver = {"flexural": flexural_dispersion, "screw": quadrupole_dispersion}[mode]
    onset = _FIG8A_ONSET[mode][0]
    fluid = dict(vf=1500.0, rho_f=1000.0)
    grid = np.arange(onset, onset + 4.0, 0.02)
    velocity = 1.0 / solver(grid * 1e3, **_FIG8_ROCK, **fluid, a=0.10).slowness
    finite = np.isfinite(velocity)

    assert finite.any(), "the mode must be found somewhere above its cutoff"
    gap = grid[finite][0] - onset
    assert -0.05 < gap < 0.2, f"{mode}: gap is {gap:.2f} kHz"
    assert finite[np.argmax(finite) :].all(), "and it is contiguous, not a scatter"


@pytest.mark.parametrize("mode", ["flexural", "screw"])
def test_the_slow_shear_modes_agree_to_a_couple_of_percent(mode):
    """All three modes now tie at the resolution of the figure.

    This test used to record a real, systematic flexural residual --
    1.29 % rms, zero near 3.3 kHz and -1.8 % at 5-6 kHz -- against a
    Stoneley control at 0.03 % on the same panel with the same
    calibration, and left it unexplained. It was the roadmap-A.8 SV
    column. Corrected:

        Stoneley (control)   0.034 % rms   (unchanged)
        flexural             1.29 -> 0.063 % rms
        screw                0.94 -> 0.058 % rms

    So the order-of-magnitude split between the axisymmetric mode and
    the two n >= 1 modes is gone, and all three sit at the +-0.3 % the
    plotted line is worth. Asserted as a ceiling, so a regression trips
    it and a further improvement does not.
    """
    from fwap import flexural_dispersion, quadrupole_dispersion, stoneley_dispersion

    solver = {"flexural": flexural_dispersion, "screw": quadrupole_dispersion}[mode]
    table = _FIG8A_PHASE[mode]
    freq = np.array([f for f, _ in table])
    reference = np.array([v for _, v in table])
    fluid = dict(vf=1500.0, rho_f=1000.0)
    velocity = 1.0 / solver(freq, **_FIG8_ROCK, **fluid, a=0.10).slowness

    assert np.isfinite(velocity).all(), "the table starts above the near-cutoff gap"
    error = np.abs(velocity / reference - 1.0)
    assert error.max() < 0.003, f"{mode} worst point {100 * error.max():.3f} %"

    st = _FIG8A_PHASE["stoneley"]
    st_f = np.array([f for f, _ in st])
    st_v = np.array([v for _, v in st])
    st_got = 1.0 / stoneley_dispersion(st_f, **_FIG8_ROCK, **fluid, a=0.10).slowness
    st_error = np.abs(st_got / st_v - 1.0)
    # The shear modes are no longer the loose ones: all three are within
    # a factor of two of each other, and of the figure's own resolution.
    assert st_error.max() < 0.003
    assert error.max() < 3.0 * st_error.max(), (
        "the n >= 1 modes should now tie as tightly as the Stoneley"
    )


# ----------------------------------------------------------------------
# Figure 12: the invaded zone, and why coverage is the wrong health metric
#
# Figure 12 (p. 249) is the first published check of the *layered*
# solvers: "Invaded zone effects with a fast sandstone. Dispersion and
# attenuation of the flexural (a) and screw (b) modes in the presence
# of: (1) a 16 cm thick invaded zone; (2) a 8 cm thick invaded zone;
# (3) the only virgin formation; (4) the only invaded zone."
#
# Table 1's two fast rows: virgin 4878 / 2601 / 2160, invaded zone
# 4390 / 2341 / 2360 -- slower and denser than the rock it replaces.
#
# **What this panel can and cannot be read for.** Eight curves are drawn
# (four phase, four group) in a 1.2-wide normalised window, and the
# column-line count runs 1 at 2.0 kHz, 4 at 2.5, 8 only across
# 3.5-5.0 kHz, then 3 from 6 kHz up. The plunge region where the models
# separate is exactly where they also cross, so no per-model curve was
# traced there and none is tabulated. What *is* readable is the two
# plateaus at the low-frequency end and the merged phase band above
# 6 kHz, and both are recorded below.
#
# The plateaus are worth having on their own: they confirm table 1's
# invaded-zone row, which every comparison here depends on and which
# this repository transcribed from a scan.
#
#   upper plateau (1.80-2.05 kHz)  1.7357 +- 0.0016  vs 2601/1500  +0.10 %
#   lower plateau (2.20-2.60 kHz)  1.5630 +- 0.0015  vs 2341/1500  +0.15 %
#
# **The finding.** Every one of the eight fwap runs -- two modes x four
# models -- returns values strictly inside its own (V_R, V_S) window and
# sawtooths, with upward jumps of +121 to +185 m/s where a guided mode's
# phase velocity can only fall. So the layered path inherits A.2 whole.
#
# What is new is the coverage:
#
#                        flexural   screw
#   1: 16 cm invaded        73 %     74 %
#   2:  8 cm invaded        38 %     77 %
#   3: virgin only           9 %     50 %
#   4: invaded only         10 %     35 %
#
# **Adding an altered zone raises coverage four- to eightfold while the
# answers stay wrong.** Against figure 12a's merged phase band the
# layered flexural solver reads +31 % at 6 kHz rising to +53 % by
# 9.8 kHz. So on the layered path coverage is not a weak health signal,
# it is an inverted one: the configuration that returns the most answers
# is the one furthest from having any.
# ----------------------------------------------------------------------

_FIG12_VIRGIN = dict(vp=4878.0, vs=2601.0, rho=2160.0)
_FIG12_INVADED = dict(vp=4390.0, vs=2341.0, rho=2360.0)

#: The plateau each family of curves leaves, read from figure 12a
#: (normalised velocity, +-0.002).
_FIG12A_PLATEAU = {"virgin": 1.7357, "invaded": 1.5630}

#: Figure 12a's merged phase band above 6 kHz (Hz, low m/s, high m/s).
#: Four phase curves are drawn as two lines here; the pair bounds them.
_FIG12A_PHASE_BAND = (
    (6.0e3, 1838.7, 1905.1),
    (7.0e3, 1711.9, 1760.2),
    (8.0e3, 1643.9, 1683.2),
    (9.0e3, 1598.6, 1628.8),
    (9.8e3, 1576.0, 1606.2),
)


def test_figure_12a_plateaus_confirm_table_1s_invaded_zone_row():
    """Check the transcribed rock before trusting the comparison.

    `4390 / 2341 / 2360` for the invaded zone came off a scanned table,
    and every layered comparison here rests on it. Figure 12a's own
    plateaus settle it: the curves with virgin rock at depth leave the
    axis at the virgin shear speed and the invaded-only curve leaves at
    the invaded shear speed, both to about 0.1 %.
    """
    assert _FIG12A_PLATEAU["virgin"] / (_FIG12_VIRGIN["vs"] / 1500.0) == pytest.approx(
        1.0, abs=0.005
    )
    assert _FIG12A_PLATEAU["invaded"] / (
        _FIG12_INVADED["vs"] / 1500.0
    ) == pytest.approx(1.0, abs=0.005)
    assert _FIG12_INVADED["vs"] < _FIG12_VIRGIN["vs"], "invasion slows the rock"
    assert _FIG12_INVADED["rho"] > _FIG12_VIRGIN["rho"], "and makes it denser"


def test_the_altered_zone_no_longer_inverts_the_coverage_signal():
    """The figure-12 finding, and the fix that removes it.

    Sparseness had been read as A.2's symptom throughout this project.
    On the layered path it was the opposite signal: a 16 cm invaded zone
    took `flexural_dispersion`'s coverage from 9 % to 73 % over
    2-10 kHz, and every extra answer was an overtone from the
    `(V_R, V_S)` window -- so a caller checking coverage to decide
    whether to trust an altered-zone curve was reading the metric
    backwards.

    Both paths now return one monotone branch, so coverage means what a
    caller would assume it means.
    """
    from fwap import BoreholeLayer, flexural_dispersion, flexural_dispersion_layered

    fluid = dict(vf=1500.0, rho_f=1000.0)
    freq = np.arange(2.0e3, 10.001e3, 200.0)

    plain = 1.0 / flexural_dispersion(freq, **_FIG12_VIRGIN, **fluid, a=0.10).slowness
    layered = (
        1.0
        / flexural_dispersion_layered(
            freq,
            **_FIG12_VIRGIN,
            **fluid,
            a=0.10,
            layers=(BoreholeLayer(**_FIG12_INVADED, thickness=0.16),),
        ).slowness
    )

    for arr in (plain, layered):
        finite = np.isfinite(arr)
        assert finite.any()
        assert np.all(np.diff(arr[finite]) <= 0.0), (
            "a guided mode never speeds up with frequency"
        )
        assert np.all(arr[finite] <= _FIG12_VIRGIN["vs"])
        assert np.all(arr[finite] > fluid["vf"])


def test_the_layered_flexural_solver_now_lands_in_the_published_band():
    """How wrong the extra answers were, against the figure -- and how
    close they are now.

    Four phase curves are drawn as two lines above 6 kHz, so the pair
    bounds the truth to a few percent. The solver used to sit 30-55 %
    above that band and rising with frequency; it now lands within
    about 4 % of its midline at every sampled frequency, and slightly
    *below* rather than above.
    """
    from fwap import BoreholeLayer, flexural_dispersion_layered

    freq = np.array([f for f, _, _ in _FIG12A_PHASE_BAND])
    lo = np.array([a for _, a, _ in _FIG12A_PHASE_BAND])
    hi = np.array([b for _, _, b in _FIG12A_PHASE_BAND])
    assert np.all(hi / lo < 1.05), "the two lines must bracket tightly to be useful"

    fluid = dict(vf=1500.0, rho_f=1000.0)
    velocity = (
        1.0
        / flexural_dispersion_layered(
            freq,
            **_FIG12_VIRGIN,
            **fluid,
            a=0.10,
            layers=(BoreholeLayer(**_FIG12_INVADED, thickness=0.16),),
        ).slowness
    )
    finite = np.isfinite(velocity)
    assert finite.sum() >= 3

    error = np.abs(velocity[finite] / (0.5 * (lo + hi))[finite] - 1.0)
    assert error.max() < 0.06, f"worst {100 * error.max():.1f} %"


# ----------------------------------------------------------------------
# Figure 15: the same layered code in a slow formation, and it works
#
# Figure 12 put `flexural_dispersion_layered` against a *fast* rock and
# it returned overtones 31-53 % high. Figure 15 (p. 255) is the slow
# counterpart -- same four models, same solver, table 1's slow sandstone
# 2751 / 1201 / 2100 and its invaded zone 2338 / 1081 / 2000 -- and it
# separates two explanations that figure 12 alone could not.
#
# The panel reads cleanly: the group curves are *dashed*, so they
# fragment under connected-component labelling and leave the four solid
# phase curves behind. Axis calibration is the best of the six figures
# (16 x-ticks residual to 0.019 kHz, 4 y-ticks to 0.00024 = 0.36 m/s),
# and the 0.550-0.850 window makes the plotted line worth about
# +-4 m/s.
#
# Two anchors, both to 0.02 %: the virgin curves leave the axis at
# 1200.7 against V_S = 1201, and the invaded-only curve at 1081.2
# against 1081.
#
# The result, over each curve's plotted band at 0.25 kHz:
#
#   model                        coverage   rms     median
#   1 virgin only  (open hole)     91 %    1.43 %   -1.34 %
#   2  8 cm invaded  (layered)     84 %    1.47 %   -1.22 %
#   3 16 cm invaded  (layered)     92 %    1.48 %   -1.49 %
#   4 invaded only (open hole)     67 %    1.01 %   -0.07 %
#
# **The layered solver is as accurate as the open-hole one.** So the
# defect figure 12 found is the fast-formation bracket (A.2), not the
# layered machinery -- one fix repairs the layered path too.
#
# It also narrows figure 8a's unexplained ~1.3 % slow-flexural offset.
# It is here in all three `n=1` configurations at the same size and the
# same shape (best near 3 kHz, worst about -2 % at 5-6 kHz, recovering
# by 14 kHz), open hole and layered alike, while the Stoneley on the
# same rock was 0.04 %. Not the layered code, not the bracket, not the
# radius, not the reading: an `n=1`-specific, geometry-independent
# offset.
#
# Two limits worth stating. The invaded-only curve could not be followed
# past about 4 kHz -- a dashed group segment crosses it there -- so it
# is tabulated only through its anchor. And the near-cutoff gap is *not*
# the single width figure 8a suggested: 1.44 kHz (virgin), 2.44 (8 cm),
# 1.19 (16 cm), 0.92 (invaded only). That claim covered two modes in one
# homogeneous rock and does not extend to layered models.
# ----------------------------------------------------------------------

_FIG15_VIRGIN = dict(vp=2751.0, vs=1201.0, rho=2100.0)
_FIG15_INVADED = dict(vp=2338.0, vs=1081.0, rho=2000.0)

#: Flexural phase velocity (Hz, m/s) from figure 15a, above each curve's
#: near-cutoff gap. About +-4 m/s.
_FIG15A_PHASE = {
    "virgin": (
        (3.0e3, 1167.0),
        (4.0e3, 1128.6),
        (5.0e3, 1101.3),
        (6.0e3, 1083.5),
        (8.0e3, 1063.0),
        (10.0e3, 1053.5),
        (12.0e3, 1046.8),
        (14.0e3, 1042.3),
    ),
    "invaded_8cm": (
        (4.0e3, 1065.8),
        (5.0e3, 1021.5),
        (6.0e3, 992.5),
        (8.0e3, 961.0),
        (10.0e3, 946.2),
        (12.0e3, 939.6),
        (14.0e3, 935.2),
    ),
    "invaded_16cm": (
        (3.0e3, 1080.6),
        (4.0e3, 1018.0),
        (5.0e3, 985.0),
        (6.0e3, 970.1),
        (8.0e3, 952.5),
        (10.0e3, 946.2),
        (12.0e3, 939.6),
        (14.0e3, 935.2),
    ),
}

#: Thickness of the invaded zone each figure-15a curve was computed for.
_FIG15A_THICKNESS = {"virgin": None, "invaded_8cm": 0.08, "invaded_16cm": 0.16}


def test_figure_15a_onsets_anchor_on_both_shear_speeds():
    """Two closed-form anchors for the slow invaded-zone panel.

    The three curves with virgin rock at depth leave the axis at the
    virgin shear speed; the invaded-only curve leaves at the invaded
    shear speed. Both land to 0.02 %, which also confirms table 1's
    slow invaded-zone row (`2338 / 1081 / 2000`).
    """
    assert 1200.7 / _FIG15_VIRGIN["vs"] == pytest.approx(1.0, abs=0.005)
    assert 1081.2 / _FIG15_INVADED["vs"] == pytest.approx(1.0, abs=0.005)
    assert _FIG15_INVADED["vs"] < _FIG15_VIRGIN["vs"] < 1500.0, "both are slow"


@pytest.mark.parametrize("model", sorted(_FIG15A_PHASE))
def test_the_layered_solver_tracks_the_published_slow_curves(model):
    """The result that exonerates the layered code.

    On a slow formation `flexural_dispersion_layered` follows the
    published invaded-zone curves to 1.5 % RMS -- indistinguishable from
    what the open-hole solver manages on the same rock. Figure 12's
    31-53 % overshoot is therefore the fast-formation bracket, not the
    layered machinery, and the A.2 fix repairs both paths.
    """
    from fwap import BoreholeLayer, flexural_dispersion, flexural_dispersion_layered

    table = _FIG15A_PHASE[model]
    freq = np.array([f for f, _ in table])
    reference = np.array([v for _, v in table])
    fluid = dict(vf=1500.0, rho_f=1000.0)
    thickness = _FIG15A_THICKNESS[model]

    if thickness is None:
        got = flexural_dispersion(freq, **_FIG15_VIRGIN, **fluid, a=0.10)
    else:
        got = flexural_dispersion_layered(
            freq,
            **_FIG15_VIRGIN,
            **fluid,
            a=0.10,
            layers=(BoreholeLayer(**_FIG15_INVADED, thickness=thickness),),
        )
    velocity = 1.0 / got.slowness

    assert np.isfinite(velocity).all(), "the tables start above the near-cutoff gap"
    error = velocity / reference - 1.0
    assert np.sqrt((error**2).mean()) < 0.03, f"{model} rms {100 * error.std():.2f} %"
    assert np.abs(error).max() < 0.04


def test_the_layered_path_now_tracks_both_the_fast_and_slow_figures():
    """State the separation as a test, because it was the conclusion --
    and it is what made the fix safe to attempt.

    The identical call was bracket-interior nonsense on the fast rock of
    figure 12 and accurate on the slow rock of figure 15. That located
    the defect in the bracket rather than the propagator, and said that
    anything "fixing" the layered propagator would break the slow half
    while leaving the fast half wrong. Correcting the bracket alone
    brings the fast half in without disturbing the slow half, which is
    the prediction that diagnosis made.
    """
    from fwap import BoreholeLayer, flexural_dispersion_layered

    fluid = dict(vf=1500.0, rho_f=1000.0)

    def layered(rock, invaded, freq):
        return (
            1.0
            / flexural_dispersion_layered(
                freq,
                **rock,
                **fluid,
                a=0.10,
                layers=(BoreholeLayer(**invaded, thickness=0.16),),
            ).slowness
        )

    slow_freq = np.array([f for f, _ in _FIG15A_PHASE["invaded_16cm"]])
    slow_ref = np.array([v for _, v in _FIG15A_PHASE["invaded_16cm"]])
    slow = layered(_FIG15_VIRGIN, _FIG15_INVADED, slow_freq)
    assert np.abs(slow / slow_ref - 1.0).max() < 0.04, "slow: still tracks the figure"

    fast_freq = np.array([f for f, _, _ in _FIG12A_PHASE_BAND])
    fast_ref = 0.5 * np.array([a + b for _, a, b in _FIG12A_PHASE_BAND])
    fast = layered(_FIG12_VIRGIN, _FIG12_INVADED, fast_freq)
    finite = np.isfinite(fast)
    assert np.abs(fast[finite] / fast_ref[finite] - 1.0).max() < 0.06, "fast: now too"


# ----------------------------------------------------------------------
# Figure 3: the same defect in the time domain, and a cross-figure check
#
# Figure 3 (p. 240) is not a dispersion plot: "Dipole source, fast
# sandstone. Source center frequency effects. The offset is equal to
# 5 m. The source center frequency varies from .5 kHz to 10.5 kHz by
# steps of .5 kHz from the top to the bottom." Twenty-one synthetic
# waveforms in the rock of figure 2a.
#
# Digitised by locating the 21 baselines (155.5 px apart, uniform) and
# timing each trace's largest late excursion. The time axis is fitted to
# the seven label decimal points: 303.4 px per ms, residual +-0.010 ms,
# and any constant offset between a decimal point and its tick is
# bounded at about 10 px = 0.03 ms.
#
# **What the traces show.** Every trace from 3.0 kHz up carries a large
# late packet at 4.35 +- 0.07 ms. Its arrival drifts by only -4.4 %
# while the source centre frequency changes by 250 % (3.0 -> 10.5 kHz),
# which is the signature of an **Airy phase** -- energy piling up at the
# stationary point of the group-velocity curve, whose arrival is set by
# the medium and not by the source.
#
# That converts to an apparent group velocity of **1150 m/s** (range
# 1124-1181), against the **1109.7 m/s** minimum of the group curve
# digitised from figure 2a at 5.24 kHz. Agreement to **+3.7 %**, with
# the measurement slightly fast -- expected, since the largest half
# cycle of an attenuating Airy packet precedes the envelope centre.
#
# So two figures of the same paper, one in frequency and one in time,
# agree on the group-velocity minimum to under 4 %. Nothing in fwap can
# produce either.
#
# **And the defect restated as a traveltime.** Over 3.0-10.5 kHz
# `flexural_dispersion` answers at 3 of 16 frequencies, at 2414-2597
# m/s. A packet at that speed covers 5 m in 1.92-2.07 ms. The published
# waveforms put the dipole energy at 4.35 ms -- so fwap's fast-formation
# answer implies a wave arriving **2.2x too early**.
#
# Not used: the printed scaling factors down the left edge, which would
# give the excitation curve. At this scan quality the glyphs are not
# reliably legible ("0.0014" and "0.0019" cannot be told apart), and a
# misread would put a false number in the repository.
# ----------------------------------------------------------------------

#: Airy-phase arrival at 5 m (source centre frequency kHz, ms), read
#: from figure 3. About +-0.03 ms absolute, +-0.01 ms relative.
_FIG3_AIRY_ARRIVAL_MS = (
    (3.0, 4.449),
    (3.5, 4.400),
    (4.0, 4.370),
    (4.5, 4.449),
    (5.0, 4.430),
    (5.5, 4.410),
    (6.0, 4.311),
    (6.5, 4.390),
    (7.0, 4.291),
    (7.5, 4.281),
    (8.0, 4.360),
    (8.5, 4.351),
    (9.0, 4.341),
    (9.5, 4.252),
    (10.0, 4.242),
    (10.5, 4.232),
)

#: Minimum of the flexural group-velocity curve of figure 2a (m/s, kHz).
_FIG2A_GROUP_MINIMUM = (1109.7, 5.24)

#: Source-receiver offset of figure 3 (m).
_FIG3_OFFSET_M = 5.0


def test_figure_3_late_packet_is_an_airy_phase():
    """Establish what the late arrival is before using it.

    An Airy phase sits at a stationary point of the group-velocity
    curve, so its arrival is a property of the formation rather than of
    the source. Across a 3.5x change in source centre frequency the
    measured arrival moves by under 5 %, which is what identifies it --
    and what licenses reading a single group velocity off it.
    """
    fc = np.array([f for f, _ in _FIG3_AIRY_ARRIVAL_MS])
    arrival = np.array([t for _, t in _FIG3_AIRY_ARRIVAL_MS])

    assert fc.min() == 3.0 and fc.max() == 10.5
    assert arrival.max() / arrival.min() - 1.0 < 0.06, "an Airy phase barely moves"
    slope = np.polyfit(fc, arrival, 1)[0]
    assert abs(slope) * (fc.max() - fc.min()) / arrival.mean() < 0.06


def test_figure_3_confirms_figure_2a_group_minimum_in_the_time_domain():
    """Two figures, two domains, one number.

    The Airy arrival implies a group velocity that must match the
    minimum of figure 2a's group curve -- a frequency-domain reading of
    a different figure on a different page. They agree to under 4 %.

    Asserted loosely on purpose: a scan measured to about 1 % against a
    traced curve good to about 2 %, so 5 % is the honest tolerance and a
    real disagreement would be far larger.
    """
    arrival = np.array([t for _, t in _FIG3_AIRY_ARRIVAL_MS])
    measured = _FIG3_OFFSET_M * 1.0e3 / arrival.mean()
    predicted, freq = _FIG2A_GROUP_MINIMUM

    assert measured / predicted == pytest.approx(1.0, abs=0.05)
    assert measured > predicted, "the largest half cycle precedes the envelope"
    assert 3.0 < freq < 10.5, "the stationary point is inside the band figure 3 spans"


def test_the_fast_flexural_answer_now_predicts_the_figure_3_arrival():
    """The time-domain check of A.2, and the fix's strongest evidence:
    figure 3 played no part in designing it.

    The old bracket implied a wave arriving at 1.9-2.1 ms over the
    figure's own 5 m offset against an observed Airy packet at
    4.35 ms -- **2.2x too early**. Differentiating the corrected phase
    branch gives a group-velocity minimum of about 1064 m/s, putting
    the arrival at 4.70 ms: **+8 %**, and in the same direction as the
    slow-formation tilt that figures 9 and 16 measure independently.

    The group velocity is also never negative now. On the old sawtooth
    it was negative on 18 of 48 adjacent samples, which is what made
    the Airy phase unreadable from the output at all.
    """
    from fwap import flexural_dispersion

    freq = np.linspace(1.0e3, 25.0e3, 481)
    velocity = (
        1.0 / flexural_dispersion(freq, **_FIG2_ROCK, **_FIG2_FLUID, a=0.10).slowness
    )
    finite = np.isfinite(velocity)
    assert finite.sum() > 100, "if coverage collapsed, retune the grid"

    ff, vv = freq[finite], velocity[finite]
    v_group = 1.0 / np.gradient(ff / vv, ff)
    assert np.all(v_group > 0.0), "group velocity must not go negative"

    observed = np.array([t for _, t in _FIG3_AIRY_ARRIVAL_MS]).mean()
    predicted = _FIG3_OFFSET_M * 1.0e3 / v_group.min()
    assert 0.9 < predicted / observed < 1.2, (
        f"predicted {predicted:.2f} ms vs observed {observed:.2f} ms"
    )


# ----------------------------------------------------------------------
# Figure 5a: the screw mode's own figure, and a bound on the method
#
# Figure 7b measured `n=2` across three rocks, but its curves merge and
# it only resolves the fast sandstone below about 10 kHz. Figure 5a
# (p. 242) is the screw mode's own panel -- "Quadrupole source.
# Dispersion (a) ... of the screw mode (1) and the first trapped mode
# (2) in the presence of a fast sandstone" -- on figure 2a's axes,
# 0-25 kHz, with only two modes on it. It is the direct `n=2`
# counterpart of figure 2a.
#
# Traced in two overlapping passes: a wide window through the plunge,
# then a narrow one with a small slope cap for the flat tail, because
# mode 2's group curve crosses mode 1's phase near 18 kHz and a single
# pass follows the steeper branch down. Monotone to +0.002 normalised
# over the whole span, which is inside the line width.
#
#   cutoff value            1.7385   vs V_S/V_f 1.7340   +0.26 %
#   at 24.87 kHz            1522.6   vs Scholte  1484.4  +2.57 %
#   crosses V_R             7.58 kHz     (figure 7b gave 7.69)
#   crosses V_f             never, inside the plotted band
#
# Two things worth keeping. The screw mode approaches Scholte **more
# slowly** than the flexural one: still +2.6 % at 25 kHz where the
# flexural mode was +0.6 %, and it never drops below the fluid velocity
# at all, where the flexural mode crossed it at 17.9 kHz.
#
# And the cross-figure agreement is a bound on the digitisation method
# itself, obtained without reference to fwap. Nine frequencies from 7 to
# 12 kHz, read off two different pages with different axis ranges, agree
# to **+0.4 % to +1.8 %**, with figure 7b systematically about 1 % high.
# That is looser than the 0.4 % figures 2a and 7a managed for the
# flexural mode, and it is the honest error bar for readings taken off
# the crowded three-rock panels.
#
# **fwap over 6.4-25 kHz**: 72 % coverage, every value inside
# `(V_R, V_S)` and sweeping it end to end (2413-2598), **not one point
# within 5 %**, errors +15 % to +67 % with median +53 %, and upward
# jumps of +102 m/s. The screw mode is never returned for this rock.
# ----------------------------------------------------------------------

#: Screw-mode phase velocity (Hz, m/s), digitised from figure 5a.
#: About +-20 m/s.
_FIG5A_SCREW_PHASE = (
    (6.5e3, 2597.5),
    (7.0e3, 2530.8),
    (8.0e3, 2300.8),
    (9.0e3, 2071.9),
    (10.0e3, 1913.9),
    (12.0e3, 1743.6),
    (14.0e3, 1654.8),
    (16.0e3, 1605.2),
    (18.0e3, 1573.9),
    (20.0e3, 1551.6),
    (22.0e3, 1538.2),
    (24.5e3, 1524.8),
)

#: Where figure 5a's screw curve leaves the axis (kHz, m/s).
_FIG5A_SCREW_ONSET = (6.29, 2607.8)

#: The same rock's screw mode read off figure 7b (Hz, m/s), for the
#: cross-figure comparison.
_FIG7B_SANDSTONE_SCREW = ((7.0e3, 2552.8), (8.0e3, 2329.6), (9.0e3, 2088.6))


def test_figure_5a_screw_curve_is_anchored_at_both_ends():
    """The `n=2` counterpart of figure 2a's end-anchor test.

    The screw mode leaves the axis at the formation shear speed and
    heads for the Scholte speed, so both ends are computable
    independently. It gets there more slowly than the flexural mode --
    still 2.6 % above Scholte at 25 kHz, and never below the fluid
    velocity -- which is why the tolerance at the top end is looser.
    """
    from fwap import scholte_speed

    velocity = np.array([v for _, v in _FIG5A_SCREW_PHASE])
    v_scholte = scholte_speed(**_FIG2_ROCK, **_FIG2_FLUID)

    assert _FIG5A_SCREW_ONSET[1] / _FIG2_ROCK["vs"] == pytest.approx(1.0, abs=0.01)
    assert np.diff(velocity).max() < 20.0, "phase velocity does not increase"
    assert 1.0 < velocity[-1] / v_scholte < 1.05, "descending toward Scholte"
    assert velocity[-1] > 1500.0, "the screw mode stays above the fluid velocity"


def test_the_two_screw_readings_agree_across_figures():
    """A bound on the digitisation method that owes nothing to fwap.

    The same rock's screw mode is drawn twice -- figure 5a on a 0-25 kHz
    axis with two curves, figure 7b on a 4-20 kHz axis with six. The two
    reads agree to under 2 %, with figure 7b high, which is the expected
    direction for a reading taken off the more crowded panel.

    This is the loosest of the three cross-figure checks in this file,
    and it is the one to quote when asking how much a number traced off
    a busy panel can be trusted.
    """
    fig5 = dict(_FIG5A_SCREW_PHASE)
    for freq, v7 in _FIG7B_SANDSTONE_SCREW:
        v5 = fig5[freq]
        assert v7 / v5 == pytest.approx(1.0, abs=0.02), (
            f"{freq / 1e3:.0f} kHz: 5a {v5}, 7b {v7}"
        )
        assert v7 > v5, "the crowded panel reads high"


def test_quadrupole_now_returns_the_screw_mode_in_this_fast_rock():
    """The `n=2` half of A.2 against the screw mode's own figure.

    Over 6.4-25 kHz the solver used to answer at nearly three quarters
    of the band, every value inside `(V_R, V_S)` and sweeping that
    window edge to edge, with **not one within 5 %** of the published
    curve. A.2 put it on the right branch at 8 % median. A.7 -- the
    marcher was tracking `Im(det)` where the `n = 2` signal is in
    `Re(det)` -- took it to **0.16 % median over all twelve tabulated
    frequencies, 0.43 % worst**, which is inside the figure's own line
    width.

    Two points sit above `V_R`, as they must: the low-frequency end of
    this branch is the formation shear speed, and `V_R` was never a
    limit of it.
    """
    from fwap import quadrupole_dispersion

    freq = np.array([f for f, _ in _FIG5A_SCREW_PHASE])
    reference = np.array([v for _, v in _FIG5A_SCREW_PHASE])
    velocity = (
        1.0 / quadrupole_dispersion(freq, **_FIG2_ROCK, **_FIG2_FLUID, a=0.10).slowness
    )
    finite = np.isfinite(velocity)

    assert finite.all(), f"expected the whole table; got {finite.sum()}"
    v_rayleigh = rayleigh_speed(_FIG2_ROCK["vp"], _FIG2_ROCK["vs"])
    assert np.all(velocity[finite] <= _FIG2_ROCK["vs"])
    assert np.all(velocity[finite] > _FIG2_FLUID["vf"])
    assert _descends(velocity[finite])
    assert (velocity[finite] > v_rayleigh).sum() >= 2

    error = np.abs(velocity[finite] / reference[finite] - 1.0)
    assert np.median(error) < 0.004, f"median {100 * np.median(error):.2f} %"
    assert error.max() < 0.008, f"worst {100 * error.max():.2f} %"


# ----------------------------------------------------------------------
# Figure 1a: the pseudo-Rayleigh curve A.1 said had no external tie
#
# "Monopole source. Dispersion (a) and attenuation (b) of the Stoneley
# wave (1) and the first two pseudo-Rayleigh modes ((2) and (3)) in the
# presence of a fast sandstone." Three modes, three fwap entry points,
# on figure 2a's axes.
#
# A.1 lists the pseudo-Rayleigh curve among three items with "no
# external tie of any kind". Figure 1a supplies one, for both branches,
# and validates the `branch` index while it is at it.
#
# A trap, caught by overlaying the traces back onto the scan: in this
# panel **the group curve is drawn above the phase curve** for the
# Stoneley, and the labels say so. That is correct physics here -- the
# Stoneley phase velocity rises with frequency in a fast formation, so
# the group velocity exceeds it -- but it is the opposite of every other
# panel in this report. Comparing `stoneley_dispersion` against the
# upper curve gives a spurious -2.5 % systematic; against the right one
# it is -0.8 %.
#
# Resolution: 1 px = 1.41 m/s here, so a plotted line is about 12.7 m/s
# -- 0.87 % at the Stoneley, 0.5-0.7 % at the pseudo-Rayleigh modes.
#
#   curve            fwap entry point                     coverage  rms
#   Stoneley phase   stoneley_dispersion                    36/36   0.90 %
#   pseudo-Rayl. 1   trapped_pseudo_rayleigh(branch=0)       97 %   1.01 %
#   pseudo-Rayl. 2   trapped_pseudo_rayleigh(branch=1)       96 %   0.80 %
#
# All three sit at one to one-and-a-half plotted line widths, so this is
# a pass at what the figure can resolve. There is a consistent small
# negative bias -- fwap reads low on all three -- that the figure cannot
# resolve into a real offset, and it is not claimed as one.
#
# Anchors: the Stoneley extrapolates to 1398.3 m/s against
# `tube_wave_speed`'s 1396.3 (+0.14 %), and both pseudo-Rayleigh modes
# cut on at the formation shear speed.
#
# **Separately, the phenomenological model is not the modal solver.**
# `fwap.synthetic.pseudo_rayleigh_dispersion` places the guided arrival
# in synthetic wavetrains from a closed form whose cutoff scale is
# `vs / (2 pi a)` = 4140 Hz, against a true cutoff of 7.71 kHz -- 1.9x
# too low. Measured against this figure it is **37 % slow near cutoff**,
# easing to 6 % by 25 kHz. Its docstring says "phenomenological"; this
# pins how much that word is carrying.
# ----------------------------------------------------------------------

#: Phase velocity (Hz, m/s) of the three monopole modes, digitised from
#: figure 1a. About +-13 m/s, one plotted line width.
_FIG1A_PHASE = {
    "stoneley": (
        (1.0e3, 1412.7),
        (2.0e3, 1416.7),
        (3.0e3, 1425.2),
        (5.0e3, 1442.1),
        (8.0e3, 1459.1),
        (10.0e3, 1463.6),
        (12.0e3, 1473.9),
        (14.0e3, 1480.3),
        (16.0e3, 1482.4),
        (18.0e3, 1486.8),
    ),
    "pr1": (
        (8.0e3, 2607.9),
        (9.0e3, 2552.8),
        (10.0e3, 2425.2),
        (12.0e3, 2090.0),
        (14.0e3, 1888.0),
        (16.0e3, 1777.7),
        (18.0e3, 1712.7),
        (20.0e3, 1668.9),
        (22.0e3, 1641.4),
        (24.0e3, 1624.4),
    ),
    "pr2": (
        (14.0e3, 2607.9),
        (16.0e3, 2548.6),
        (18.0e3, 2468.0),
        (20.0e3, 2333.1),
        (22.0e3, 2155.7),
        (24.0e3, 2022.9),
    ),
}

#: Where each pseudo-Rayleigh mode cuts on in figure 1a (kHz, m/s), and
#: the Stoneley's low-frequency limit.
_FIG1A_CUTOFF = {"pr1": (7.71, 2614.3), "pr2": (12.89, 2629.1)}
_FIG1A_STONELEY_LIMIT = 1398.3


def test_figure_1a_is_anchored_on_the_tube_wave_and_the_shear_speed():
    """Three closed-form anchors, none needing a modal solve."""
    from fwap import tube_wave_speed

    v_tube = tube_wave_speed(_FIG2_ROCK["vs"], _FIG2_ROCK["rho"], **_FIG2_FLUID)
    assert _FIG1A_STONELEY_LIMIT / v_tube == pytest.approx(1.0, abs=0.005)
    for mode in ("pr1", "pr2"):
        _, v_cut = _FIG1A_CUTOFF[mode]
        assert v_cut / _FIG2_ROCK["vs"] == pytest.approx(1.0, abs=0.015)
    # Both pseudo-Rayleigh modes descend toward the fluid velocity, not Scholte.
    for mode in ("pr1", "pr2"):
        tail = _FIG1A_PHASE[mode][-1][1]
        assert 1500.0 < tail < _FIG2_ROCK["vs"]


@pytest.mark.parametrize("branch,mode", [(0, "pr1"), (1, "pr2")])
def test_trapped_pseudo_rayleigh_matches_the_published_curve(branch, mode):
    """The tie A.1 said did not exist, and a check on the branch index.

    `trapped_pseudo_rayleigh_dispersion` follows both published
    pseudo-Rayleigh curves to about 1 % -- one to one-and-a-half plotted
    line widths at this figure's resolution. That `branch=0` lands on
    the first mode and `branch=1` on the second is itself part of the
    result: the index means what the API says it means.
    """
    from fwap import trapped_pseudo_rayleigh_dispersion

    table = _FIG1A_PHASE[mode]
    freq = np.array([f for f, _ in table])
    reference = np.array([v for _, v in table])
    got = trapped_pseudo_rayleigh_dispersion(
        freq, **_FIG2_ROCK, **_FIG2_FLUID, a=0.10, branch=branch
    )
    velocity = 1.0 / got.slowness

    assert np.isfinite(velocity).all(), "the table starts above the cutoff"
    error = velocity / reference - 1.0
    assert np.sqrt((error**2).mean()) < 0.02, f"rms {100 * error.std():.2f} %"
    assert np.abs(error).max() < 0.03

    # The branches are distinct and ordered: branch 1 is the faster mode.
    other = (
        1.0
        / trapped_pseudo_rayleigh_dispersion(
            freq, **_FIG2_ROCK, **_FIG2_FLUID, a=0.10, branch=1 - branch
        ).slowness
    )
    overlap = np.isfinite(other)
    if overlap.any():
        assert not np.allclose(velocity[overlap], other[overlap], rtol=0.02)


def test_stoneley_in_a_fast_formation_agrees_to_one_line_width():
    """The fast-formation half of the Stoneley check.

    Figure 8a tied `stoneley_dispersion` at 0.04 % rms in a slow
    formation. This is the same solver in a fast one, over the band
    where figure 1a still draws phase and group as separate lines: 0.9 %
    rms, which is one plotted line width here, with fwap consistently on
    the low side.
    """
    from fwap import stoneley_dispersion

    table = _FIG1A_PHASE["stoneley"]
    freq = np.array([f for f, _ in table])
    reference = np.array([v for _, v in table])
    velocity = (
        1.0 / stoneley_dispersion(freq, **_FIG2_ROCK, **_FIG2_FLUID, a=0.10).slowness
    )

    assert np.isfinite(velocity).all()
    error = velocity / reference - 1.0
    assert np.abs(error).max() < 0.02, f"worst {100 * np.abs(error).max():.2f} %"
    assert np.all(velocity < 1500.0), "the Stoneley never exceeds the fluid velocity"


def test_the_phenomenological_model_is_not_the_modal_solver():
    """Pin how much work the word "phenomenological" is doing.

    `fwap.synthetic.pseudo_rayleigh_dispersion` places the guided
    arrival in synthetic wavetrains. Its cutoff scale is
    `vs / (2 pi a)` = 4140 Hz against a true cutoff of 7.71 kHz, so near
    cutoff it is far too slow -- 37 % against this figure, easing to 6 %
    by 25 kHz. The modal solver in the same package is within 1 %.
    """
    from fwap import pseudo_rayleigh_dispersion

    table = _FIG1A_PHASE["pr1"]
    freq = np.array([f for f, _ in table])
    reference = np.array([v for _, v in table])
    model = 1.0 / np.asarray(
        pseudo_rayleigh_dispersion(_FIG2_ROCK["vs"], 1500.0, 0.10)(freq)
    )

    error = model / reference - 1.0
    assert error.max() < -0.05, "it is slow everywhere on this band"
    assert error.min() < -0.25, "and badly so near the cutoff"
    assert error[-1] > error[0], "the two converge as the mode approaches V_f"


# ----------------------------------------------------------------------
# Figure 6: the quadrupole gathers, and a cutoff that is 32 % too high
#
# "Quadrupole source, fast sandstone. Shot point obtained with a 1.5 kHz
# (a) and a 6 kHz (b) source center frequency." Fourteen traces at
# r = 2.40-5.00 m in 0.20 m steps, in the rock of figure 5a.
#
# **What this figure could not be used for, stated first.** The gather
# does not survive digitisation well enough to measure a moveout. Each
# trace is normalised to its own peak, the wavetrains overlap their
# neighbours' bands, and the authors drew two dashed guide lines through
# every trace. Reconstructing the 14 waveforms and running `fwap.stc`
# over them gives coherence scattered between 0.4 and 0.88 with no
# stable slowness peak, so no velocity is quoted from it. (For contrast,
# `stc` on the real IODP U1347A gather returns 0.948 median coherence.)
#
# **What it does give is immune to all of that**: zero crossings survive
# amplitude clipping, so the *frequency* of the ringing wavetrain is
# solid. Twelve of the fourteen traces agree closely -- median
# **7.19 kHz**, the consistent group spanning 7.00-7.38 -- for a source
# whose centre frequency is **6.0 kHz**.
#
# The received ring sitting *above* the source frequency is the
# signature of a mode with a cutoff: source energy below cutoff cannot
# propagate in the mode, so the wavetrain is pushed up to where the
# excitation switches on. Figure 5a puts the screw cutoff at 6.29 kHz
# and figure 5c's excitation is zero below about 6.3 kHz, peaking near
# 9 -- a 6 kHz source folded against that lands at about 7.2. It also
# explains panel (a): at 1.5 kHz, far below cutoff, there is no ring at
# all, only a short wavelet.
#
# **And the finding.** `quadrupole_dispersion`'s first root for this
# rock is at **8.29 kHz** -- 32 % above the published 6.29 kHz cutoff --
# and it returns NaN at every single-frequency call from 6.5 to 8.4 kHz.
# So the solver returns nothing at the frequency where the paper's own
# synthetic waveforms show the screw mode ringing hardest. The `n=2`
# defect is not only that the values above cutoff are overtones: the
# onset of the mode is misplaced, and a 2 kHz band where the mode
# demonstrably exists and is strongly excited is empty.
# ----------------------------------------------------------------------

#: Dominant frequency of the ringing wavetrain in figure 6(b) (kHz),
#: from the twelve traces whose spectra agree. Source centre frequency
#: 6.0 kHz.
_FIG6B_RING_KHZ = 7.19
_FIG6B_RING_RANGE_KHZ = (7.00, 7.38)
_FIG6_SOURCE_KHZ = (1.5, 6.0)

#: Screw-mode cutoff read off figure 5a (kHz).
_FIG5A_SCREW_CUTOFF_KHZ = 6.29


def test_the_figure_6_ring_sits_above_the_cutoff_and_the_source():
    """What the ringing frequency identifies.

    A guided mode cannot carry energy below its cutoff, so a source
    centred under the cutoff is received *above* it. Figure 6(b) puts
    the ring at 7.19 kHz for a 6.0 kHz source and a 6.29 kHz cutoff --
    above both, and inside the band where figure 5c says the excitation
    has switched on.
    """
    lo, hi = _FIG6B_RING_RANGE_KHZ
    assert lo <= _FIG6B_RING_KHZ <= hi
    assert _FIG6B_RING_KHZ > _FIG6_SOURCE_KHZ[1], "pushed above the source"
    assert _FIG6B_RING_KHZ > _FIG5A_SCREW_CUTOFF_KHZ, "and above the cutoff"
    # figure 5a's own curve must start at or below the observed ring
    assert _FIG5A_SCREW_CUTOFF_KHZ < _FIG5A_SCREW_PHASE[0][0] / 1e3 + 0.3


def test_the_quadrupole_cutoff_now_lands_on_the_published_one():
    """The figure-6 finding, and its cause.

    `quadrupole_dispersion` used to find no root for this rock below
    about **8.3 kHz**, against a published cutoff of 6.29 -- 32 % high --
    and returned NaN across the whole band where figure 6(b) shows the
    screw mode ringing hardest. That was read as a near-cutoff onset
    defect separate from the search window.

    It was neither: it was roadmap A.7, the marcher tracking the wrong
    part of the determinant. The onset is now **6.39 kHz, +1.6 %**, and
    the ring band is fully covered.
    """
    from fwap import quadrupole_dispersion

    fluid = dict(vf=1500.0, rho_f=1000.0)
    grid = np.arange(4.0e3, 12.0e3, 10.0)
    velocity = 1.0 / quadrupole_dispersion(grid, **_FIG2_ROCK, **fluid, a=0.10).slowness
    finite = np.isfinite(velocity)
    assert finite.any(), "the solver must find the mode somewhere"

    first = grid[finite][0] / 1e3
    assert first / _FIG5A_SCREW_CUTOFF_KHZ - 1.0 < 0.05, f"first root {first:.2f} kHz"
    assert first < _FIG6B_RING_KHZ, "and below the frequency the waveforms ring at"

    # And the observed ring band is now fully covered.
    band = np.arange(6.5e3, 8.2e3, 100.0)
    got = 1.0 / quadrupole_dispersion(band, **_FIG2_ROCK, **fluid, a=0.10).slowness
    assert np.isfinite(got).all(), "the ring band resolves end to end"


def test_quadrupole_dispersion_is_reproducible_across_equal_grids():
    """The sharpest caveat in this file, now discharged.

    `np.arange(6.0, 20.01, 0.2) * 1e3` and
    `np.arange(6.0e3, 20.01e3, 200.0)` are the same 71 frequencies to
    within **1.5e-11 Hz** -- last-bit floating-point rounding, a
    relative difference of 8e-16. Handed to `quadrupole_dispersion`
    they used to return **different coverage** and disagree about
    whether individual frequencies converged at all, so coverage was a
    property of how the caller happened to build the array rather than
    of the rock.

    That was roadmap A.7: the marcher was seeding off sign changes in
    round-off, and which ones it found depended on the last bit of the
    frequency. Tracking the part of the determinant that carries the
    signal, the two grids now agree exactly -- same coverage, same
    values.
    """
    from fwap import quadrupole_dispersion

    a = np.arange(6.0, 20.01, 0.2) * 1e3
    b = np.arange(6.0e3, 20.01e3, 200.0)
    assert a.size == b.size
    assert not np.array_equal(a, b), "the grids must differ, if only in the last bit"
    assert np.abs(a - b).max() < 1.0e-9, "and only in the last bit"

    fluid = dict(vf=1500.0, rho_f=1000.0)
    va = 1.0 / quadrupole_dispersion(a, **_FIG2_ROCK, **fluid, a=0.10).slowness
    vb = 1.0 / quadrupole_dispersion(b, **_FIG2_ROCK, **fluid, a=0.10).slowness

    assert np.isfinite(va).sum() == np.isfinite(vb).sum()
    np.testing.assert_array_equal(np.isfinite(va), np.isfinite(vb))
    finite = np.isfinite(va)
    assert finite.sum() > 60, f"and it is nearly the whole band: {finite.sum()}/71"
    np.testing.assert_allclose(va[finite], vb[finite], rtol=1.0e-12)


# ----------------------------------------------------------------------
# Figure 9: the slow-formation waveforms, and what differentiation costs
#
# "Dipole source, slow sandstone. Source center frequency effects. The
# offset is equal to 4 m. The source center frequency varies from .5 kHz
# to 10.5 kHz by steps of .5 kHz." Figure 3's counterpart in the rock of
# figure 8a -- and this time in the regime where fwap works, so it is a
# prediction test rather than a defect measurement.
#
# Digitised from the 21 baselines (155.5 px apart) with the time axis
# fitted to the seven label decimal points: 304.9 px per ms, residual
# +-0.011 ms.
#
# Every trace from 2.0 kHz up carries a compact late packet at
# **4.068 +- 0.045 ms**, drifting only **-1.8 %** while the source centre
# frequency changes fivefold. That is an Airy phase, and a tighter one
# than figure 3's -4.4 %. At the figure's own 4 m offset it implies a
# group velocity of **983 m/s** (960-1009).
#
# Three ways to that number, two of them from the paper:
#
#   figure 9, measured in the time domain      983 m/s
#   figure 8a phase curve, differentiated      992 +- 4 m/s at 5.1-5.5 kHz
#   fwap phase output, differentiated          960.4 m/s at 3.89 kHz
#
# The two readings of the paper agree to **0.9 %** -- a time-domain
# figure against a frequency-domain one, which also validates the
# differentiation. fwap is **3.2 % low** on the value.
#
# **The finding is the frequency, not the value.** fwap puts the group
# minimum at 3.89 kHz where the figure puts it near 5.2 -- **25 % low** --
# from a phase curve that was only 1.3 % off. Differentiation amplifies
# a phase residual that is a *distortion* rather than an offset, and
# figure 8a already showed the shape: zero near 3.3 kHz, -1.8 % at
# 5-6 kHz, back to -0.8 % by 14 kHz. A tilt like that moves the
# stationary point. So anyone using fwap's slow flexural curve to
# predict a waveform will place the Airy phase at the wrong frequency
# even though the phase velocities look fine.
#
# Method notes, since both group curves come from differentiation. The
# figure-8a minimum is stable at 992-996 m/s for boxcar widths 41-121
# (21 is undersmoothed and finds a spurious minimum), while the
# *frequency* of the minimum moves over 5.07-5.47 kHz -- so the value is
# good to about 0.5 % and the frequency to about +-0.4 kHz. fwap's is
# stable to 0.1 m/s and 0.1 kHz across grid steps 0.02-0.2 kHz; the slow
# path shows none of the grid instability figure 6 found at `n=2`.
# ----------------------------------------------------------------------

#: Late-packet (Airy) arrival at 4 m from figure 9, for the traces above
#: 2 kHz where it is fully developed (source centre frequency kHz, ms).
_FIG9_AIRY_ARRIVAL_MS = (
    (2.0, 4.088),
    (2.5, 4.167),
    (3.0, 4.108),
    (3.5, 4.088),
    (4.0, 4.059),
    (4.5, 4.029),
    (5.0, 4.127),
    (5.5, 4.019),
    (6.0, 4.019),
    (6.5, 4.108),
    (7.0, 4.088),
    (7.5, 4.068),
    (8.0, 4.078),
    (8.5, 4.059),
    (9.0, 4.049),
    (9.5, 3.964),
    (10.0, 4.049),
    (10.5, 4.059),
)
_FIG9_OFFSET_M = 4.0

#: Minimum of the flexural group curve obtained by differentiating
#: figure 8a's traced phase curve (m/s, kHz). Value good to ~0.5 %, the
#: frequency to about +-0.4 kHz.
_FIG8A_GROUP_MINIMUM = (992.0, 5.2)


def test_figure_9_late_packet_is_an_airy_phase():
    """Tighter than figure 3's, in the rock where fwap works."""
    fc = np.array([f for f, _ in _FIG9_AIRY_ARRIVAL_MS])
    arrival = np.array([t for _, t in _FIG9_AIRY_ARRIVAL_MS])

    assert arrival.max() / arrival.min() - 1.0 < 0.06
    slope = np.polyfit(fc, arrival, 1)[0]
    assert abs(slope) * (fc.max() - fc.min()) / arrival.mean() < 0.04


def test_figure_9_and_figure_8a_agree_on_the_group_minimum():
    """Time domain against frequency domain, both from the paper.

    The measured Airy arrival implies a group velocity that must match
    the minimum of the group curve obtained by differentiating figure
    8a's phase curve. They agree to 1 %, which is also what licenses
    using the differentiated curve as a reference below.
    """
    arrival = np.array([t for _, t in _FIG9_AIRY_ARRIVAL_MS])
    measured = _FIG9_OFFSET_M * 1.0e3 / arrival.mean()
    predicted, _ = _FIG8A_GROUP_MINIMUM

    assert measured / predicted == pytest.approx(1.0, abs=0.02)


def test_differentiating_the_slow_flexural_curve_keeps_the_airy_frequency():
    """What removing the 1.3 % phase residual buys in the group domain.

    This test recorded the cost of a tilt: `flexural_dispersion`
    followed figure 8a's slow-formation phase curve to 1.29 % rms, and
    differentiating put the group minimum 3 % low in value and about
    **25 % low in frequency** -- near 3.9 kHz where the figure puts it
    near 5.2. A tilt in the phase residual moves the stationary point,
    and that is the part a user notices: it places the Airy phase of a
    synthetic waveform wrongly while the phase velocities still look
    right.

    The roadmap-A.8 correction removed the tilt. The group minimum is
    now **998.4 m/s at 5.15 kHz** against the figure's 992.0 at 5.2 --
    **+0.6 % in value and -1 % in frequency**. Kept, with the
    assertions inverted, as the group-domain check on the phase fix.
    """
    from fwap import flexural_dispersion

    fluid = dict(vf=1500.0, rho_f=1000.0)
    grid = np.arange(2.6e3, 14.5e3, 50.0)
    phase = 1.0 / flexural_dispersion(grid, **_FIG15_VIRGIN, **fluid, a=0.10).slowness
    ok = np.isfinite(phase)
    assert ok.sum() > 100, "the slow path should be dense here"

    f = grid[ok]
    group = 1.0 / np.gradient(f / phase[ok], f)
    i = int(np.argmin(group))
    v_min, f_min = group[i], f[i] / 1e3
    ref_v, ref_f = _FIG8A_GROUP_MINIMUM

    assert v_min / ref_v == pytest.approx(1.0, abs=0.02), (
        f"value: {v_min:.1f} vs {ref_v} m/s"
    )
    assert f_min / ref_f == pytest.approx(1.0, abs=0.05), (
        f"and the Airy frequency no longer moves: {f_min:.2f} vs {ref_f} kHz"
    )


# ----------------------------------------------------------------------
# Figure 10: the processing chain closed on published waveforms
#
# "Dipole source, slow sandstone. Shot point obtained with a 1 kHz (a)
# and a 3 kHz (b) source center frequency." Fourteen traces at
# r = 2.40-5.00 m, in the rock of figures 8a and 9.
#
# Unlike figure 6 this gather does digitise: the packets are compact and
# the moveout is strong, so after cropping past the scale-factor
# brackets a straight-line fit to the envelope peaks has r^2 = 0.995.
# Two velocities come out of it, and keeping them apart is the point:
#
#   envelope (packet) moveout  ->  GROUP velocity
#   fwap.stc coherent align    ->  PHASE velocity
#
#   panel      dominant f   group (moveout)   phase (stc)   coherence
#   (a) 1 kHz    0.86 kHz      1009 m/s        1205 m/s       0.960
#   (b) 3 kHz    2.77 kHz      1037 m/s        1156 m/s       0.717
#
# **The chain closes.** In panel (a) the packet is at 0.86 kHz, where
# the flexural mode is at its low-frequency limit and its phase velocity
# is the formation shear speed: `stc` returns 1205 against V_S = 1201,
# **+0.3 %**. In panel (b), at 2.77 kHz, `stc` returns 1156 against
# figure 8a's traced phase curve at 1172 (**-1.3 %**) and fwap's own
# solver at 1187 (**-2.6 %**). Published synthetic waveforms, through
# this package's processing, land on this package's forward model.
#
# The group numbers are consistent too: 1009 and 1037 m/s sit just above
# the 992 m/s group minimum that figure 8a's differentiated curve and
# figure 9's Airy phase both give, which is right, because neither
# packet is at the 5.2 kHz where that minimum sits.
#
# **And panel (a) settles what the near-cutoff gap is.** fwap's slow
# flexural solver finds no root below about 2.5 kHz. At 0.86 kHz it
# returns NaN -- yet the paper's own waveforms show a coherent arrival
# there, `stc` picking it at 0.960 and putting it at the shear speed.
# The gap is a solver limitation, not a physical absence, and this is
# the waveform evidence for it.
# ----------------------------------------------------------------------

#: Figure 10 read as (source kHz, packet dominant kHz, group m/s from
#: envelope moveout, phase m/s from `fwap.stc`, stc peak coherence).
_FIG10_PANELS = (
    (1.0, 0.86, 1009.2, 1204.8, 0.960),
    (3.0, 2.77, 1036.8, 1156.1, 0.717),
)

#: Figure 8a's traced flexural phase velocity at figure 10(b)'s dominant
#: frequency (kHz, m/s).
_FIG8A_PHASE_AT_2P77 = (2.77, 1171.6)


def test_figure_10_separates_group_from_phase():
    """The two velocities a shot gather carries, and they must differ.

    An envelope moveout is a group velocity; a coherent alignment across
    the array is a phase velocity. On a strongly dispersive mode the two
    are far apart, and reading either as the other is a 15-20 % error.
    """
    for _, _, group, phase, _ in _FIG10_PANELS:
        assert phase > group, "phase exceeds group on this branch"
        assert phase / group > 1.10, "and by a margin no reading error explains"


def test_stc_on_the_published_waveforms_lands_on_the_shear_speed():
    """Panel (a): the low-frequency limit, straight off the waveforms.

    At 0.86 kHz the slow-formation flexural mode is at its
    low-frequency limit, where the phase velocity is the formation shear
    speed. `fwap.stc` on the digitised gather returns 1205 m/s at 0.960
    coherence against V_S = 1201.
    """
    source, dominant, _, phase, coherence = _FIG10_PANELS[0]
    assert dominant < 1.0, "the packet sits at the low-frequency end"
    assert coherence > 0.9, "and it is a coherent arrival, not a guess"
    assert phase / _FIG15_VIRGIN["vs"] == pytest.approx(1.0, abs=0.01)


def test_stc_on_the_published_waveforms_matches_the_published_curve():
    """Panel (b): the processing half checked against the modelling half.

    At 2.77 kHz `stc` gives 1156 m/s, figure 8a's traced phase curve
    gives 1172, and `flexural_dispersion` gives 1187. Waveforms from the
    paper, through this package's processing, land within 3 % of this
    package's forward model -- the first time the two halves have been
    checked against each other on anything external.
    """
    from fwap import flexural_dispersion

    _, dominant, _, phase, _ = _FIG10_PANELS[1]
    freq, published = _FIG8A_PHASE_AT_2P77
    assert dominant == pytest.approx(freq, abs=0.01)
    assert phase / published == pytest.approx(1.0, abs=0.03)

    fluid = dict(vf=1500.0, rho_f=1000.0)
    solver = (
        1.0
        / flexural_dispersion(
            np.array([dominant * 1e3]), **_FIG15_VIRGIN, **fluid, a=0.10
        ).slowness[0]
    )
    assert np.isfinite(solver), "the solver answers at this frequency"
    assert phase / solver == pytest.approx(1.0, abs=0.04)


def test_the_near_cutoff_gap_is_a_solver_limitation_not_an_absence():
    """Panel (a) is the waveform evidence, and most of the gap is gone.

    This test recorded a solver silent below about **2.5 kHz** while
    figure 10(a) showed a coherent arrival at 0.86 kHz -- `fwap.stc`
    picking it at 0.960 and putting it at the shear speed. After the
    roadmap-A.8 correction the flexural solver's first root for this
    rock is at **0.99 kHz**, against a published onset of 1.04, so the
    1.66 kHz shortfall is down to 0.13 kHz.

    The waveform evidence still bites, just barely: 0.86 kHz is below
    even the corrected onset, so the packet is still in a band the
    solver reports as empty. The claim survives; its magnitude does
    not.
    """
    from fwap import flexural_dispersion

    fluid = dict(vf=1500.0, rho_f=1000.0)
    _, dominant, _, phase, coherence = _FIG10_PANELS[0]
    v = (
        1.0
        / flexural_dispersion(
            np.array([dominant * 1e3]), **_FIG15_VIRGIN, **fluid, a=0.10
        ).slowness[0]
    )
    assert not np.isfinite(v), "the solver is silent at the packet's frequency"

    grid = np.arange(0.5e3, 4.0e3, 20.0)
    got = 1.0 / flexural_dispersion(grid, **_FIG15_VIRGIN, **fluid, a=0.10).slowness
    ok = np.isfinite(got)
    assert ok.any()
    first = grid[ok][0] / 1e3
    assert dominant < first < 1.1, (
        f"still silent at the packet, but only just: first root {first:.2f} kHz"
    )
    assert coherence > 0.9 and phase > 1000.0, "while the waveforms show the mode"


# ----------------------------------------------------------------------
# Figure 11: the screw mode where fwap is silent, and one case where
# silence is right
#
# "Quadrupole source, slow sandstone. Shot point obtained with a 1 kHz
# (a) and a 6 kHz (b) source center frequency." Fourteen traces at
# r = 2.40-5.00 m, in the rock of figure 8a.
#
# **Panel (b) is the finding.** A 6 kHz source produces a ringing
# wavetrain whose energy sits at 4.68 kHz -- above the 3.74 kHz screw
# cutoff figure 8a gives, below the source. Envelope moveout is
# 1166 m/s with r^2 = 0.982, and `fwap.stc` puts the phase velocity at
# 1139.6 m/s against figure 8a's traced screw curve at 1179 -- **-3.3 %**.
# `quadrupole_dispersion` returns **NaN** there, because its first root
# for this rock is at 5.25 kHz. The mode demonstrably propagates,
# coherently, at a velocity the published curve predicts, in a band the
# solver reports as empty.
#
# **Panel (a) is the balancing case, and it matters.** At a 1 kHz source
# the packet sits at 1.83 kHz, and `quadrupole_dispersion` is silent
# there too -- but so is the paper: figure 8a draws no screw curve below
# 3.74 kHz. There is no trapped mode at 1.83 kHz, so the arrival is a
# leaky or head-wave contribution a modal solver is not meant to
# produce, and the NaN is **correct**. Not every gap is a defect, and
# this file should not leave the impression that it is.
#
# **The unification.** Figure 6 reported the fast screw cutoff as "32 %
# too high" and figure 8a reported a "1.5 kHz near-cutoff gap". Those
# are the same phenomenon, and the percentage was the misleading way to
# quote it:
#
#   case             published   fwap    gap
#   flexural, slow    1.04 kHz   2.52   1.48 kHz  (+142 %)
#   screw,    slow    3.74       5.25   1.51      ( +40 %)
#   screw,    fast    6.29       8.29   2.00      ( +32 %)
#
# The onset is late by **1.5-2.0 kHz in absolute terms** across two
# modes and two formations. The percentages differ only because the
# cutoffs differ.
# ----------------------------------------------------------------------

#: Figure 11 read as (source kHz, packet dominant kHz, group m/s from
#: envelope moveout, phase m/s from `fwap.stc`, moveout r^2).
_FIG11_PANELS = ((1.0, 1.83, 1580.4, 1286.2, 0.878), (6.0, 4.68, 1166.5, 1139.6, 0.982))

#: Figure 8a's traced screw phase velocity at figure 11(b)'s dominant
#: frequency (kHz, m/s).
_FIG8A_SCREW_AT_4P68 = (4.68, 1179.0)

#: Published onset and fwap's first root for three homogeneous cases
#: (label, published kHz, fwap kHz).
_NEAR_CUTOFF_GAPS = (
    ("flexural slow", 1.04, 2.52),
    ("screw slow", 3.74, 5.25),
    ("screw fast", 6.29, 8.29),
)


def test_figure_11b_screw_mode_is_now_found_and_lands_on_the_published_curve():
    """The mode propagates at 4.68 kHz, and the solver now finds it.

    This test used to record a silence: the waveforms showed a coherent
    screw arrival at 4.68 kHz -- envelope moveout r^2 = 0.982, `stc`
    within 3.3 % of figure 8a's traced screw curve -- in a band where
    `quadrupole_dispersion` returned NaN, its first root for this rock
    being at 5.25 kHz.

    The roadmap-A.8 correction moved that first root to 3.85 kHz, and
    at 4.68 kHz the solver now returns **1180.2 m/s** against figure
    8a's traced 1179.0 -- **+0.10 %**. The band the solver reported as
    empty holds the mode the published curve predicts, and the solver
    agrees with the curve to a tenth of a percent.

    Panel (a) at 1.83 kHz remains NaN, correctly: the paper draws no
    screw curve below 3.74 kHz, so there is no trapped mode to find.
    """
    from fwap import quadrupole_dispersion

    source, dominant, group, phase, r2 = _FIG11_PANELS[1]
    freq, published = _FIG8A_SCREW_AT_4P68
    assert dominant == pytest.approx(freq, abs=0.01)
    assert r2 > 0.95, "the moveout is a clean straight line"
    assert phase / published == pytest.approx(1.0, abs=0.05)

    fluid = dict(vf=1500.0, rho_f=1000.0)
    v = (
        1.0
        / quadrupole_dispersion(
            np.array([dominant * 1e3]), **_FIG15_VIRGIN, **fluid, a=0.10
        ).slowness[0]
    )
    assert np.isfinite(v), "the solver now resolves the mode at that frequency"
    assert v / published == pytest.approx(1.0, abs=0.005), (
        f"and lands on the published curve: {v:.1f} vs {published}"
    )

    # Panel (a) stays silent, and should: no trapped mode exists there.
    below = _FIG11_PANELS[0][1]
    assert not np.isfinite(
        1.0
        / quadrupole_dispersion(
            np.array([below * 1e3]), **_FIG15_VIRGIN, **fluid, a=0.10
        ).slowness[0]
    )


def test_figure_11a_is_a_gap_the_solver_is_right_to_have():
    """Not every NaN is a defect.

    At a 1 kHz source the packet sits at 1.83 kHz, below the 3.74 kHz
    screw cutoff. `quadrupole_dispersion` is silent -- and so is the
    paper, which draws no screw curve there. The arrival is a leaky or
    head-wave contribution a modal solver is not meant to produce.
    """
    from fwap import quadrupole_dispersion

    _, dominant, group, phase, r2 = _FIG11_PANELS[0]
    published_cutoff = _NEAR_CUTOFF_GAPS[1][1]
    assert dominant < published_cutoff, "below any trapped screw mode"
    assert r2 < 0.95, "and the moveout is correspondingly less clean"
    assert group > _FIG15_VIRGIN["vs"], "faster than the shear speed, so not the mode"

    fluid = dict(vf=1500.0, rho_f=1000.0)
    v = (
        1.0
        / quadrupole_dispersion(
            np.array([dominant * 1e3]), **_FIG15_VIRGIN, **fluid, a=0.10
        ).slowness[0]
    )
    assert not np.isfinite(v), "silence here is the right answer"


def test_the_near_cutoff_gap_is_an_absolute_offset_not_a_percentage():
    """Ties figure 6's "cutoff 32 % too high" to figure 8a's "1.5 kHz
    gap" -- they are one phenomenon.

    Across flexural and screw, slow and fast, the onset is late by
    1.5-2.0 kHz. Quoted as percentages the same offsets read 32 %, 40 %
    and 142 %, which says more about the cutoff frequencies than about
    the solver.
    """
    gaps = np.array([fw - pub for _, pub, fw in _NEAR_CUTOFF_GAPS])
    pcts = np.array([100.0 * (fw / pub - 1.0) for _, pub, fw in _NEAR_CUTOFF_GAPS])

    assert gaps.min() > 1.3 and gaps.max() < 2.2, "tight in absolute terms"
    assert gaps.max() / gaps.min() < 1.6
    assert pcts.max() / pcts.min() > 4.0, "and wildly spread as percentages"


# ----------------------------------------------------------------------
# Figure 13: how little a dipole sees invasion at 1 kHz
#
# "Dipole source. Invaded zone effects in the presence of a fast
# sandstone. Iso-offset (z = 5m) of the waveforms obtained in the
# presence of the only virgin formation (1), and a 8 cm (2) and 16 cm
# (3) invaded zone. The source center frequency is successively equal to
# 1 kHz (a), 3 kHz (b), 6 kHz (c), and 7.5 kHz (d)."
#
# Panel (a) extracts cleanly, and the answer is a number worth having:
# cross-correlated against the virgin trace, the 8 cm model lags by
# **+0.1 us** and the 16 cm model by **+1.2 us**, at correlations of
# 0.992 and 0.981. Over a ~2 ms traveltime at 5 m that is **0.06 %**.
#
# A 16 cm invaded zone is undetectable at 1 kHz, which is the
# time-domain form of what figure 12 shows in the frequency domain: all
# three models share a plotted plateau at V_S below about 2 kHz, and
# figure 12's own reading put that plateau at 1.7357 for the whole
# group. Figure 13(a) says how far apart they actually are there.
#
# **Corrected while working figure 14.** This block used to say "only
# panel (a) is measurable" and that no extraction cleared r = 0.8 in
# panels (b)-(d). That was an artefact of my own extraction, not of the
# figure: the half-window was narrower than the widest trace's own
# excursion, so the virgin trace in panel (b) was clipped to 68 %
# coverage and the correlation was computed against a truncated
# reference. Widened, **panel (b) measures**: at 3 kHz the 8 cm model
# lags by **+54.6 us** and the 16 cm model by **+99.0 us**, at
# correlations of 0.930 and 0.848, invariant to +-0.01 us across 36
# combinations of crop start, crop end and half-window.
#
# So the growth figure 12 predicts above 2 kHz **is** measured here, and
# it is steep: the 16 cm delay goes from 1.2 us at 1 kHz to 99.0 us at
# 3 kHz, a factor of **79** for a 3x change in source frequency.
#
# Panels (c) and (d) are still refused, now for a positive reason. Their
# traces overlap so the components merge -- coverage sticks at 0.76-0.78
# whatever the window -- and panel (d)'s best-fit lags are +264 and
# +319 us regardless of window choice, the constant-lag signature of a
# cross-correlation hopping cycles rather than measuring a delay.
# ----------------------------------------------------------------------

#: Figure 13(a): cross-correlation lag against the virgin trace at a
#: 1 kHz source and 5 m offset (microseconds), and the correlation.
_FIG13A_INVASION_LAG_US = {"8 cm": (0.1, 0.992), "16 cm": (1.2, 0.981)}

#: Figure 13(b): the same at 3 kHz. Recovered after the extraction
#: half-window was widened; see the correction note above.
_FIG13B_INVASION_LAG_US = {"8 cm": (54.6, 0.930), "16 cm": (99.0, 0.848)}
_FIG13_OFFSET_M = 5.0


def test_invasion_is_undetectable_at_1_khz():
    """The size of the invaded-zone effect where a dipole tool works.

    Both invaded models reproduce the virgin waveform at 5 m to within
    about a microsecond, correlating above 0.98. Whatever the layered
    solver gets wrong on this rock, the *invasion* part of the answer is
    negligible at 1 kHz -- which is consistent with figure 12, where all
    three models share a plateau below about 2 kHz.
    """
    for name, (lag_us, corr) in _FIG13A_INVASION_LAG_US.items():
        assert abs(lag_us) < 5.0, f"{name}: {lag_us} us"
        assert corr > 0.97, f"{name}: r = {corr}"
    # thicker invasion delays more, and both delay rather than advance
    assert _FIG13A_INVASION_LAG_US["16 cm"][0] > _FIG13A_INVASION_LAG_US["8 cm"][0]
    assert min(v for v, _ in _FIG13A_INVASION_LAG_US.values()) >= 0.0


def test_the_dipole_invasion_delay_grows_steeply_between_1_and_3_khz():
    """Figure 13(b), recovered after the extraction was fixed.

    Figure 12 shows the three models' dispersion curves separating above
    about 2 kHz. Figure 13(b) is that separation in the time domain: at
    3 kHz the 16 cm model lags the virgin waveform by 99 us against
    1.2 us at 1 kHz. Both panels correlate above 0.84, and both lag
    rather than lead.
    """
    for name, (lag_us, corr) in _FIG13B_INVASION_LAG_US.items():
        assert lag_us > 0.0, f"{name}: {lag_us} us"
        assert corr > 0.84, f"{name}: r = {corr}"
    assert _FIG13B_INVASION_LAG_US["16 cm"][0] > _FIG13B_INVASION_LAG_US["8 cm"][0]
    # and every 3 kHz delay exceeds every 1 kHz one
    assert min(v for v, _ in _FIG13B_INVASION_LAG_US.values()) > max(
        v for v, _ in _FIG13A_INVASION_LAG_US.values()
    )
    travel_ms = _FIG13_OFFSET_M * 1.0e3 / _FIG2_ROCK["vs"]
    frac = _FIG13B_INVASION_LAG_US["16 cm"][0] * 1e-3 / travel_ms
    assert 0.01 < frac < 0.10, f"{frac:.3%} of the traveltime"


def test_the_1_khz_invasion_lag_is_a_negligible_fraction_of_traveltime():
    """Put the microseconds in context.

    The flexural arrival at 5 m in this rock is around 2 ms at 1 kHz --
    the mode is near its `V_S` limit there. A 1.2 us shift is under a
    tenth of a percent of that, well below anything a slowness log
    resolves.
    """
    travel_ms = _FIG13_OFFSET_M * 1.0e3 / _FIG2_ROCK["vs"]
    worst = max(abs(v) for v, _ in _FIG13A_INVASION_LAG_US.values())
    assert worst * 1e-3 / travel_ms < 0.001, "under 0.1 % of the traveltime"


# ----------------------------------------------------------------------
# Figure 14: the quadrupole invaded zone, where the effect is amplitude
#
# "Quadrupole source. Invaded zone effects in the presence of a fast
# sandstone. Iso-offset (z = 5m) of the waveforms obtained in the
# presence of the only virgin formation (1), and a 8 cm (2) and 16 cm
# (3) invaded zone. The source center frequency is successively equal to
# 1.5 kHz (a), 3 kHz (b), 6 kHz (c), and 7.5 kHz (d)."
#
# I expected this figure to hit the ringing-wavetrain wall that stopped
# figures 6 and 13(b)-(d). That was half wrong. Panel (a) is not a
# wavetrain -- it is a compact three-to-four-cycle wavelet that extracts
# cleanly across the full band. The first pass called it unmeasurable
# only because the extraction half-window was narrower than the 16 cm
# trace's own excursion, clipping it to 60 % coverage; the overlay check
# caught that.
#
# Widened, panel (a) gives the quadrupole's invasion delay: the 8 cm
# model lags the virgin waveform by +9.2 us at r = 0.924, the 16 cm model
# by +36.7 us at r = 0.795. Both are invariant -- across 36 combinations
# of crop start (+-110 px), crop end (+-180 px) and half-window
# (+-15 px), the spread in each is **zero**. The 16 cm correlation sits
# at the 0.8 bar because the waveform changes shape (invasion adds
# cycles), not because the measurement wobbles.
#
# Against figure 13's dipole at 1 kHz (+0.1 and +1.2 us), the
# quadrupole's 16 cm delay is **30x the dipole's** -- 1.9 % of the 5 m
# traveltime against 0.06 %.
#
# Panels (b)-(d) are refused, and for a positive reason rather than a
# threshold: their best-fit 8 cm lags are +237.7, +238.9 and +235.1 us at
# 3, 6 and 7.5 kHz -- constant to +-2 us across a 2.5x change in source
# frequency, with *negative* zero-lag correlations. A physical invasion
# delay does not do that; a cycle-hopping cross-correlation does.
#
# Also legible is the printed peak-amplitude scale factor on all twelve
# traces, and that is where the rest of this figure's content is. The
# report says so on p. 228 -- "the variations of the peak amplitude as a
# function of the invaded zone thickness are more pronounced with low
# source center frequencies than previously (Figure 14a, b relative to
# 1.5 kHz and 3 kHz)" -- and it names the mechanism: "due to a higher
# frequency location of the useful starting energy of the screw mode".
#
# Digits transcribed, then checked independently by measuring the ink:
# the plotted peak excursions reproduce the printed numbers to within
# 0.027 in the worst panel and under 0.01 typically, the residual being
# the finite line width, which inflates a small trace relative to a
# large one. Both readings agree that panel (c) is genuinely
# non-monotone in thickness.
#
# The dipole/quadrupole contrast, both measured the same way:
#
#     f_c        dipole (fig 13)   quadrupole (fig 14)
#     low          1.25x             2.90x   (1 vs 1.5 kHz)
#     3 kHz        1.03x             1.68x
#     6 kHz        1.00x             1.30x
#     7.5 kHz      1.00x             1.65x
#
# The dipole is flat to 3 % at every frequency at or above 3 kHz; the
# quadrupole never drops below 1.29x. The published claim holds.
#
# **fwap cannot be checked against the substance of this figure, and the
# reason is worth stating exactly.** Peak amplitude at a fixed offset is
# excitation times propagation, and `BoreholeMode` carries neither for
# this model: there is no excitation field on it at all, and
# `attenuation_per_meter` comes back `None` from both the plain and the
# layered quadrupole path. The figure's main effect is outside the API
# surface -- a *correct* dispersion solver would not reproduce it
# either. That is a scope limit, not a defect.
#
# The dispersion the figure implies, fwap mostly does not return. Of the
# twelve (model, source frequency) pairs plotted, the quadrupole solver
# produces a phase velocity for **three**: the virgin formation gives no
# root at any of 1.5, 3, 6 or 7.5 kHz, its onset sitting at 8.4 kHz. The
# A.2 bracket is again the whole story -- all 194 converged samples
# across the three runs lie strictly inside `(V_R, V_S)`, none outside.
#
# And the one dispersion claim in the figure-14 paragraph -- "the
# increase of the group velocity of the Airy phase" at 6 and 7.5 kHz --
# does not come out inaccurate here, it comes out with the wrong sign.
# The overtone sawtooth ramps at roughly 0.5 (m/s)/Hz, steep enough that
# `v_g = 1 / (d(f s)/df)` goes **negative** on 18 of 48 adjacent virgin
# pairs. No guided mode has a negative group velocity, so the Airy phase
# cannot be read off this output at all.
#
# Coverage inverts with invasion thickness, as in figure 12: virgin 49
# of 141 samples (first root 8.40 kHz), 8 cm 63 (4.10 kHz), 16 cm 82
# (3.40 kHz). The three-medium problem converges further, and lower,
# than the one-medium problem contained in it.
#
# Unlike figure 6's slow-formation model, these counts were stable
# across bit-identical grids built two ways and across repeat calls, so
# the grid sensitivity recorded there is model-specific, not universal.
# The tests below still assert bands rather than exact counts.
# ----------------------------------------------------------------------

#: Figure 14: printed peak-amplitude scale factors (virgin, 8 cm, 16 cm)
#: keyed by source centre frequency in kHz. Read off the page and
#: confirmed by measuring the plotted excursions (agreement <= 0.027).
_FIG14_PEAK_AMPLITUDE = {
    1.5: (0.345, 0.587, 1.000),
    3.0: (0.597, 0.847, 1.000),
    6.0: (0.838, 0.772, 1.000),
    7.5: (0.607, 0.804, 1.000),
}

#: Figure 13, the dipole counterpart, measured from the plotted
#: excursions by the same routine. Digits were read for 1 and 3 kHz and
#: agree; 6 and 7.5 kHz are excursion measurements only.
_FIG13_PEAK_AMPLITUDE = {
    1.0: (0.799, 0.907, 1.000),
    3.0: (1.000, 0.996, 0.975),
    6.0: (0.998, 0.998, 1.000),
    7.5: (1.000, 0.998, 1.000),
}

#: Figure 14(a): cross-correlation lag against the virgin trace at a
#: 1.5 kHz quadrupole source and 5 m offset (microseconds), and the
#: correlation. Invariant across 36 crop/window choices.
_FIG14A_INVASION_LAG_US = {"8 cm": (9.7, 0.924), "16 cm": (36.5, 0.797)}

#: Figure 14(b)-(d): the best-fit 8 cm lags, refused. Constant across a
#: 2.5x change in source frequency, which a real delay would not be.
_FIG14_REFUSED_LAG_US = {3.0: 237.7, 6.0: 238.9, 7.5: 235.1}

#: The four source centre frequencies of figure 14 (kHz).
_FIG14_SOURCE_KHZ = (1.5, 3.0, 6.0, 7.5)

#: Receiver offset of every figure-14 panel (m).
_FIG14_OFFSET_M = 5.0

#: Lowest frequency (kHz) at which fwap's plain quadrupole solver
#: returns a root for the figure-14 virgin fast sandstone.
_FIG14_VIRGIN_ONSET_KHZ = 8.4


def _fig14_quadrupole(thickness: float | None, freq: np.ndarray) -> np.ndarray:
    """Phase velocity (m/s) for figure 14's model, NaN where no root."""
    from fwap.cylindrical_solver import (
        quadrupole_dispersion,
        quadrupole_dispersion_layered,
    )

    fluid = dict(vf=1500.0, rho_f=1000.0)
    if thickness is None:
        mode = quadrupole_dispersion(freq, **_FIG12_VIRGIN, **fluid, a=0.10)
    else:
        mode = quadrupole_dispersion_layered(
            freq,
            **_FIG12_VIRGIN,
            **fluid,
            a=0.10,
            layers=(BoreholeLayer(**_FIG12_INVADED, thickness=thickness),),
        )
    return 1.0 / mode.slowness


def test_fig14_scale_factors_are_internally_consistent():
    """The transcription, before it is used for anything.

    Each panel is normalised to its own maximum, so every triple must
    contain exactly one 1.000 and nothing above it. Panel (c) is the odd
    one out and that is not a mis-read: two independent readings of the
    page -- the printed digits and the plotted ink -- both put the 8 cm
    trace *below* the virgin one there.
    """
    for f_khz, triple in _FIG14_PEAK_AMPLITUDE.items():
        assert max(triple) == pytest.approx(1.0), f"{f_khz} kHz: {triple}"
        assert min(triple) > 0.0, f"{f_khz} kHz: {triple}"
        assert sum(v == 1.0 for v in triple) == 1, f"{f_khz} kHz: {triple}"
    virgin, eight, _ = _FIG14_PEAK_AMPLITUDE[6.0]
    assert eight < virgin, "panel (c) is non-monotone in invasion thickness"
    for f_khz in (1.5, 3.0, 7.5):
        a, b, c = _FIG14_PEAK_AMPLITUDE[f_khz]
        assert a < b < c, f"{f_khz} kHz rises with thickness: {(a, b, c)}"


def test_the_quadrupole_sees_invasion_far_more_strongly_than_the_dipole():
    """The published comparison, measured.

    "The variations of the peak amplitude as a function of the invaded
    zone thickness are more pronounced with low source center
    frequencies than previously" -- previously being the dipole of
    figure 13. At the lowest source frequency each figure plots, the
    quadrupole's spread is 2.90x against the dipole's 1.25x.
    """

    def spread(triple: tuple[float, float, float]) -> float:
        return max(triple) / min(triple)

    quad_low = spread(_FIG14_PEAK_AMPLITUDE[1.5])
    dip_low = spread(_FIG13_PEAK_AMPLITUDE[1.0])
    assert quad_low == pytest.approx(2.90, abs=0.05)
    assert dip_low == pytest.approx(1.25, abs=0.05)
    assert quad_low > 2.0 * dip_low, f"{quad_low} vs {dip_low}"
    # and at the one frequency both figures share
    assert spread(_FIG14_PEAK_AMPLITUDE[3.0]) > 1.6
    assert spread(_FIG13_PEAK_AMPLITUDE[3.0]) < 1.05


def test_the_dipole_goes_flat_above_3_khz_and_the_quadrupole_never_does():
    """Where each source stops resolving a shallow altered zone.

    The report's own words for figure 13(c, d) are that "the peak
    amplitude still varies little from one case to another". Measured,
    "little" is under 1 %. The quadrupole's smallest spread over the
    same band is 1.29x.
    """
    for f_khz in (6.0, 7.5):
        triple = _FIG13_PEAK_AMPLITUDE[f_khz]
        assert max(triple) / min(triple) < 1.01, f"dipole {f_khz} kHz: {triple}"
    worst = min(
        max(t) / min(t) for f_khz, t in _FIG14_PEAK_AMPLITUDE.items() if f_khz >= 3.0
    )
    assert worst > 1.25, f"quadrupole stays sensitive: {worst}"


def test_the_figure_14_screw_onset_now_reaches_the_top_source_frequency():
    """Twelve plotted wavetrains, and A.7 reaches the top of them.

    Figure 14 plots three models at four source centre frequencies
    (1.5, 3.0, 6.0, 7.5 kHz). The virgin fast sandstone used to return
    no root at any of the four, which was recorded as a near-cutoff
    onset gap that the A.2 window fix was never going to move.

    A.7 moved it: the virgin onset is now 6.4 kHz rather than 8.4, so
    the 7.5 kHz source resolves. The three lower sources still do not,
    and correctly -- they are below the published 6.29 kHz cutoff, so
    there is no trapped screw mode there to find. The invaded models,
    being slower, cut on lower and reach further down the list.
    """
    freq = 1.0e3 * np.array(_FIG14_SOURCE_KHZ)
    virgin = _fig14_quadrupole(None, freq)
    resolved = np.isfinite(virgin)
    assert resolved.sum() == 1, f"expected only the 7.5 kHz source, got {virgin}"
    assert resolved[-1], "and it is the highest source frequency"
    assert not resolved[:-1].any(), "the rest are below the published cutoff"

    # The invaded models cut on lower, so they reach further down.
    found = sum(
        int(np.isfinite(_fig14_quadrupole(th, freq)).sum()) for th in (0.08, 0.16)
    )
    assert found >= 2, f"{found} of 8 invaded pairs resolved"


def test_the_figure_14_screw_onset_sits_inside_the_figure():
    """Quantify what is left of the gap between the published band and
    fwap's, after A.7.

    The virgin onset was 8.4 kHz, above every source frequency in the
    figure. It is now 6.4, inside the figure and within 1.6 % of the
    published 6.29 -- so the recorded onset defect was the wrong-part
    tracking, not a separate near-cutoff problem.
    """
    freq = np.linspace(1000.0, 15000.0, 141)
    virgin = _fig14_quadrupole(None, freq)
    ok = np.isfinite(virgin)
    assert ok.any(), "the solver resolves the mode somewhere"
    onset_khz = freq[ok].min() / 1.0e3
    assert onset_khz < _FIG14_VIRGIN_ONSET_KHZ - 1.5, f"onset {onset_khz} kHz"
    assert onset_khz / _FIG5A_SCREW_CUTOFF_KHZ - 1.0 < 0.05
    assert onset_khz < max(_FIG14_SOURCE_KHZ), (
        f"onset {onset_khz} kHz is inside the figure's source range"
    )


def test_figure_14_open_hole_samples_now_descend_below_the_rayleigh_speed():
    """A.2 on the figure's own models, and the one path it does not fix.

    Not one converged sample of the virgin, 8 cm or 16 cm run used to
    escape `(V_R, V_S)`: the bracket was not a bias on these curves, it
    was their entire support. The open-hole run now descends below
    `V_R`, which is only representable because the window was widened.

    The two **layered** runs are a different story, recorded in the next
    test: the cased `n=2` determinant is too ill-conditioned to
    root-find over the widened window, so the corrected code answers
    almost nowhere rather than returning the sawtooth it used to.
    """
    freq = np.linspace(1000.0, 15000.0, 141)
    v_r = rayleigh_speed(_FIG12_VIRGIN["vp"], _FIG12_VIRGIN["vs"])

    virgin = _fig14_quadrupole(None, freq)
    good = virgin[np.isfinite(virgin)]
    assert good.size >= 5, f"open hole resolved only {good.size}"
    assert np.all(good < _FIG12_VIRGIN["vs"])
    assert _descends(good), "one descending branch"
    assert good.min() < v_r, "the branch passes below V_R"

    for thickness in (0.08, 0.16):
        v = _fig14_quadrupole(thickness, freq)
        layered = v[np.isfinite(v)]
        assert np.all(layered < _FIG12_VIRGIN["vs"])
        assert np.all(layered > 1500.0)


def test_the_screw_group_velocity_is_no_longer_negative():
    """The Airy phase figure 14(c, d) is about can be read off this now.

    The overtone substitution used to walk the root up to `V_S`, lose
    it, and re-acquire a higher branch; the ramps were steep enough that
    `v_g = 1 / (d(f s)/df)` came out negative across a large minority of
    adjacent samples. A guided mode never has negative group velocity,
    so that was not a small error in the Airy velocity -- it was the
    absence of a usable group-velocity curve.

    With one monotone branch, `v_g` is positive everywhere.
    """
    freq = np.linspace(1000.0, 15000.0, 141)
    v = _fig14_quadrupole(None, freq)
    ok = np.isfinite(v)
    f_ok, v_ok = freq[ok], v[ok]
    assert v_ok.size >= 5
    dv_df = np.diff(v_ok) / np.diff(f_ok)
    v_g = v_ok[:-1] / (1.0 - (f_ok[:-1] / v_ok[:-1]) * dv_df)
    assert np.all(v_g > 0.0), f"{int((v_g < 0).sum())} of {v_g.size} pairs negative"
    assert v_ok.max() < _FIG12_VIRGIN["vs"], "no run up to the V_S ceiling"


def test_the_cased_n2_coverage_is_physical_now_that_the_right_part_is_tracked():
    """Roadmap A.7, and the third reading of the same coverage numbers.

    This measurement has now meant three different things. Before A.2,
    adding a layer bought *more* coverage than the virgin rock, which
    was the bracket rather than the physics. After A.2 it inverted, and
    the layered path went near-silent: scanned across the corrected
    window the cased `n = 2` determinant showed of order ninety sign
    changes at 12 kHz, and the marcher declined to choose. That was
    written up as catastrophic cancellation in the propagator chain,
    with the delta-matrix reformulation as the only route out.

    It was neither the bracket nor the propagator. The `n = 2`
    determinant is REAL at real `k_z` -- the parity of the fixed phase
    flips with azimuthal order -- so `Im(det)`, which the marcher was
    tracking, was round-off, and every one of those crossings was
    noise. Tracking `Re(det)` gives **one** crossing where there were
    430 on a 1200-point grid, and the propagator itself reproduces
    `E(b)` from `P E(a)` to 1e-16.

    Coverage now increases with an invaded zone again -- 87, 95, 102 of
    141 for virgin, 8 cm and 16 cm -- and this time it is physics: the
    invaded rock is slower, so the screw mode cuts on lower and more of
    the band is inside it. The onsets say so directly, 6.40, 5.60 and
    4.90 kHz.
    """
    freq = np.linspace(1000.0, 15000.0, 141)
    counts, onsets = [], []
    for thickness in (None, 0.08, 0.16):
        v = _fig14_quadrupole(thickness, freq)
        ok = np.isfinite(v)
        counts.append(int(ok.sum()))
        onsets.append(freq[ok].min() / 1.0e3)

    # Coverage is high everywhere, not near-silent.
    assert min(counts) > 80, f"coverage {counts}"
    # It rises with invasion, and the onsets explain why: a slower
    # altered zone lowers the cutoff.
    assert counts[0] < counts[1] < counts[2], f"coverage {counts}"
    assert onsets[0] > onsets[1] > onsets[2], f"onsets {onsets}"
    assert onsets[0] == pytest.approx(6.4, abs=0.3)


def test_the_figure_14_model_returns_no_attenuation_at_all():
    """Why the figure's main result is out of scope, not merely wrong.

    Peak amplitude at a fixed 5 m offset is excitation times
    propagation. `BoreholeMode` has no excitation field, and for this
    model `attenuation_per_meter` is `None` on every path. Neither
    factor is available, so figure 14's amplitudes are not something a
    fixed A.2 would deliver.
    """
    from fwap.cylindrical_solver import (
        quadrupole_dispersion,
        quadrupole_dispersion_layered,
    )

    fluid = dict(vf=1500.0, rho_f=1000.0)
    freq = 1.0e3 * np.array(_FIG14_SOURCE_KHZ)
    plain = quadrupole_dispersion(freq, **_FIG12_VIRGIN, **fluid, a=0.10)
    layered = quadrupole_dispersion_layered(
        freq,
        **_FIG12_VIRGIN,
        **fluid,
        a=0.10,
        layers=(BoreholeLayer(**_FIG12_INVADED, thickness=0.16),),
    )
    assert plain.attenuation_per_meter is None
    assert layered.attenuation_per_meter is None
    assert not hasattr(plain, "excitation")


def test_the_quadrupole_invasion_delay_at_1p5_khz():
    """Figure 14(a): the quadrupole's own invasion delay.

    Both models delay rather than advance, thicker delays more, and at
    5 m the 16 cm figure is a couple of percent of the traveltime.
    """
    for name, (lag_us, corr) in _FIG14A_INVASION_LAG_US.items():
        assert lag_us > 0.0, f"{name}: invasion delays rather than advances"
        assert corr > 0.75, f"{name}: r = {corr}"
    quad = _FIG14A_INVASION_LAG_US["16 cm"][0]
    assert quad > _FIG14A_INVASION_LAG_US["8 cm"][0]
    travel_ms = _FIG14_OFFSET_M * 1.0e3 / _FIG12_VIRGIN["vs"]
    assert 0.005 < quad * 1e-3 / travel_ms < 0.05, "a couple of percent"


def test_the_invasion_delay_is_a_steep_function_of_source_frequency():
    """Why figure 14's delay must not be read as a dipole/quadrupole gap.

    It is tempting to set figure 14(a)'s 36.5 us against figure 13(a)'s
    1.2 us and call the quadrupole thirty times more delay-sensitive.
    The two panels are at *different* source frequencies -- 1.5 against
    1 kHz -- and figure 13's own panels show how steeply that matters:
    the dipole's 16 cm delay goes 1.2 -> 99.0 us between 1 and 3 kHz, a
    factor of 79 for a factor of 3 in frequency. The quadrupole's
    1.5 kHz value lands between those two, about where interpolating the
    dipole would put it.

    No source frequency is shared between the two figures where both are
    measurable, so **the figures do not support a like-for-like delay
    comparison** -- only the peak-amplitude one, which the report itself
    makes. This test exists to keep that distinction from eroding.
    """
    dip_1k = _FIG13A_INVASION_LAG_US["16 cm"][0]
    dip_3k = _FIG13B_INVASION_LAG_US["16 cm"][0]
    quad_1p5k = _FIG14A_INVASION_LAG_US["16 cm"][0]
    assert dip_3k / dip_1k > 50.0, f"{dip_1k} -> {dip_3k} us"
    assert dip_1k < quad_1p5k < dip_3k, (
        "the quadrupole's 1.5 kHz delay is bracketed by the dipole's own "
        f"1 and 3 kHz values: {dip_1k} < {quad_1p5k} < {dip_3k}"
    )


def test_the_refused_figure_14_lags_are_refused_for_a_reason():
    """Why panels (b)-(d) are not quoted.

    Their best-fit 8 cm lags are the same number at 3, 6 and 7.5 kHz.
    A delay caused by a 8 cm layer would change with the wavelength
    probing it; a cross-correlation hopping cycles in a ringing
    wavetrain would not. The spread across a 2.5x change in source
    frequency is the evidence, and it is recorded so the refusal can be
    checked rather than taken on trust.
    """
    lags = np.array(list(_FIG14_REFUSED_LAG_US.values()))
    assert np.ptp(lags) < 5.0, f"constant to a few us: {lags}"
    assert lags.min() > 100.0, "and far larger than panel (a)'s"
    # panel (a), which is quoted, is nothing like them
    assert _FIG14A_INVASION_LAG_US["16 cm"][0] < 0.2 * lags.min()


# ----------------------------------------------------------------------
# Figure 16: the slow formation, where invasion finally shows up
#
# "Dipole source. Invaded zone effects in the presence of a slow
# sandstone. Iso-offset (z = 5m) of the waveforms obtained in the
# presence of the only virgin formation (1), and a 8 cm (2) and 16 cm
# (3) invaded zone. The source center frequency is successively equal to
# 1 kHz (a), 3 kHz (b), 6 kHz (c), and 7.5 kHz (d). Each series is
# normalized with respect to its own maximum denoted by 1.00."
#
# **That last sentence settles a convention this series had been
# inferring.** Figures 13 and 14 print the same scale factors without
# saying what they mean; figure 16's caption states it. The printed
# number is the trace's peak amplitude relative to the largest trace in
# its panel. The figure-14 reading was right.
#
# This figure is the same experiment as figure 13 with the rock swapped,
# and it is the one where the invaded zone stops being invisible.
#
# **Twelve arrows, and they calibrate the figure.** Each trace carries a
# drawn arrowhead at a shear arrival -- the virgin formation's on trace
# 1, the invaded zone's own on traces 2 and 3. Detected as filled blobs
# and converted through the time axis at 5 m, they give 1198.0 m/s for
# the four virgin arrows (against table 1's `V_S` = 1201, **-0.25 %**)
# and 1083.0 m/s for the eight invaded ones (against 1081, **+0.18 %**).
# Twelve independent detections landing on two distinct published
# velocities is a calibration check owing nothing to fwap, and it
# confirms table 1's slow invaded-zone row a second time -- figure 15's
# dispersion anchor read 1081.2 from a different figure entirely.
#
# **The amplitudes, digits and ink agreeing to 0.018**, settle several
# glyphs that could be read two ways (0.754 not 0.734; 0.644 not 0.699;
# 0.452 confirmed to 0.002):
#
#     f_c        virgin   8 cm    16 cm    spread
#     1 kHz      0.612    0.754   1.000     1.63x
#     3 kHz      1.000    0.644   0.672     1.55x
#     6 kHz      1.000    0.452   0.672     2.21x
#     7.5 kHz    0.881    0.706   1.000     1.42x
#
# Figure 13's fast sandstone, measured the same way, gives 1.25 / 1.03 /
# 1.00 / 1.00. **Where the fast formation goes flat at and above 3 kHz,
# the slow one never drops below 1.42x.**
#
# **And the mechanism is measurable, not just the size.** Splitting each
# trace at its own arrow gives the P wavetrain's amplitude against the
# shear packet's:
#
#     P/S        virgin   8 cm    16 cm
#     1 kHz       0.03    0.07    0.05
#     3 kHz       0.03    0.15    0.22
#     6 kHz       0.10    0.96    1.53
#     7.5 kHz     0.21    1.95    2.76
#
# Monotone in thickness at every frequency at or above 3 kHz, and
# monotone in frequency. At 6 kHz with 16 cm, and at 7.5 kHz with either
# thickness, **the P wavetrain becomes the largest event in the trace**
# -- the series maximum jumps from ~5.0 ms to ~2.35 ms. That is the
# report's conclusion C ("small velocity contrasts can modify the
# internal dynamics of the waveforms more easily in slow formations,
# through an increase of the P wavetrain") as a number.
#
# **A like-for-like delay comparison, which figure 14 could not offer.**
# Figures 13(a) and 16(a) are the same source, the same 1 kHz, the same
# 5 m, the same two thicknesses -- only the rock differs. The 16 cm
# delay is +1.2 us in the fast sandstone and **+117.3 us** in the slow
# one, at correlations of 0.981 and 0.879. As a fraction of traveltime
# that is 0.06 % against 2.82 %: **45 times larger**.
#
# **The fwap check, on the one path this series has shown to be good.**
# Figure 15 tied these exact three models' phase velocity at 1.47-1.48 %
# rms. So the forward prediction is fair: take the group-velocity
# minimum, divide 5 m by it, compare with the measured arrival of the
# shear packet.
#
# The virgin packet peaks at 5.01-5.11 ms across all four source
# frequencies -- frequency-independent, the Airy signature figures 3 and
# 9 relied on -- giving 989.6 m/s against figure 8a's published group
# minimum of 992.0 m/s. **Two independent figures, 0.24 % apart.**
#
#     model    measured        fwap        error
#     virgin   5.05 ms (n=4)   5.21 ms     +3.0 %
#     8 cm     5.56 ms (n=3)   5.91 ms     +6.3 %
#     16 cm    5.64 ms (n=2)   6.09 ms     +8.0 %
#
# The virgin +3.0 % is figure 9's "3 % low in value" arrived at from a
# different figure and a different domain. **The new result is what
# happens when a layer is added.** The invaded arrivals drift with
# source frequency (5.37-5.68 and 5.45-5.84), so against the most
# charitable end of each measured range the errors are +2.0 %, +4.0 %
# and +4.3 %. Either way the layered error is about **twice** the
# open-hole one. The 8 cm / 16 cm difference is inside the measurement
# spread and is **not** claimed.
#
# So figure 15's verdict needs one qualification. "The layered solver is
# as accurate as the open-hole one" holds for *phase* velocity. It does
# not survive differentiation: the group velocity that a waveform
# actually arrives at is twice as wrong on the layered path.
#
# Two smaller things. **At 1 kHz fwap returns nothing for any of the
# three models** (onsets 2.52, 3.51, 2.94 kHz) -- the panel that
# measures best is entirely outside coverage, the near-cutoff gap of
# figure 10 now confirmed on the layered path. And unlike figure 14's
# fast-formation quadrupole, **these curves are structurally sound**: no
# interior gaps, no negative group velocity, phase monotone throughout.
# Here the defect is accuracy; there it was the absence of a curve.
# ----------------------------------------------------------------------

#: Figure 16: printed peak-amplitude scale factors (virgin, 8 cm, 16 cm)
#: keyed by source centre frequency in kHz. Digits read from the page,
#: confirmed by measuring the plotted excursions (agreement <= 0.018).
_FIG16_PEAK_AMPLITUDE = {
    1.0: (0.612, 0.754, 1.000),
    3.0: (1.000, 0.644, 0.672),
    6.0: (1.000, 0.452, 0.672),
    7.5: (0.881, 0.706, 1.000),
}

#: Figure 16: P-wavetrain amplitude over shear-packet amplitude, each
#: trace split at its own drawn arrow. Same keying.
_FIG16_P_OVER_S = {
    1.0: (0.03, 0.07, 0.05),
    3.0: (0.03, 0.15, 0.22),
    6.0: (0.10, 0.96, 1.53),
    7.5: (0.21, 1.95, 2.76),
}

#: Figure 16: shear-speed recovered from each drawn arrow at 5 m (m/s),
#: as (virgin, 8 cm, 16 cm) per source centre frequency.
_FIG16_ARROW_VS = {
    1.0: (1196.6, 1083.1, 1086.3),
    3.0: (1204.7, 1084.6, 1085.6),
    6.0: (1188.8, 1078.7, 1083.1),
    7.5: (1201.8, 1081.2, 1081.2),
}

#: Figure 16: time of the shear-packet peak (ms) at 5 m, per model, over
#: the source frequencies where that packet is still the trace maximum.
_FIG16_AIRY_MS = {
    "virgin": (5.04, 5.11, 5.05, 5.01),
    "8 cm": (5.37, 5.62, 5.68),
    "16 cm": (5.45, 5.84),
}

#: Figure 16(a): cross-correlation lag against the virgin trace at a
#: 1 kHz dipole source and 5 m offset (microseconds), and the
#: correlation. Directly comparable to _FIG13A_INVASION_LAG_US.
_FIG16A_INVASION_LAG_US = {"8 cm": (46.1, 0.969), "16 cm": (117.3, 0.879)}

_FIG16_OFFSET_M = 5.0

#: Lowest frequency (kHz) at which fwap resolves each figure-16 model.
_FIG16_ONSET_KHZ = {"virgin": 2.52, "8 cm": 3.51, "16 cm": 2.94}


def _fig16_flexural(thickness: float | None, freq: np.ndarray) -> np.ndarray:
    """Phase velocity (m/s) for figure 16's model, NaN where no root."""
    fluid = dict(vf=1500.0, rho_f=1000.0)
    if thickness is None:
        mode = flexural_dispersion(freq, **_FIG15_VIRGIN, **fluid, a=0.10)
    else:
        mode = flexural_dispersion_layered(
            freq,
            **_FIG15_VIRGIN,
            **fluid,
            a=0.10,
            layers=(BoreholeLayer(**_FIG15_INVADED, thickness=thickness),),
        )
    return 1.0 / mode.slowness


def _fig16_group_minimum(thickness: float | None) -> tuple[float, float]:
    """(v_g minimum in m/s, its frequency in Hz) over the longest run.

    The value is grid-converged: 241, 591 and 1181-point grids agree to
    0.02 m/s, so the coarse grid used here is not a shortcut.
    """
    freq = np.linspace(200.0, 12000.0, 241)
    v = _fig16_flexural(thickness, freq)
    idx = np.where(np.isfinite(v))[0]
    seg = max(np.split(idx, np.where(np.diff(idx) > 1)[0] + 1), key=len)
    ff, vv = freq[seg], v[seg]
    v_g = 1.0 / np.gradient(ff / vv, ff)
    i = int(np.argmin(v_g))
    return float(v_g[i]), float(ff[i])


def test_fig16_scale_factors_are_internally_consistent():
    """The transcription, before anything is built on it.

    Each panel is normalised to its own maximum -- the caption says so
    in as many words -- so every triple holds exactly one 1.000 and
    nothing above it.
    """
    for f_khz, triple in _FIG16_PEAK_AMPLITUDE.items():
        assert max(triple) == pytest.approx(1.0), f"{f_khz} kHz: {triple}"
        assert min(triple) > 0.0, f"{f_khz} kHz: {triple}"
        assert sum(v == 1.0 for v in triple) == 1, f"{f_khz} kHz: {triple}"


def test_the_figure_16_arrows_recover_both_published_shear_speeds():
    """Twelve arrowheads, two velocities, no solver involved.

    Each trace carries a drawn arrow at a shear arrival: the virgin
    formation's on trace 1, the invaded zone's own on traces 2 and 3.
    Read through the time axis at 5 m they return table 1's two slow
    shear speeds to a quarter of a percent. This is the calibration
    check for every other number taken off this figure, and it confirms
    the invaded-zone row a second time -- figure 15 anchored it at
    1081.2 from an entirely different figure.
    """
    virgin = np.array([t[0] for t in _FIG16_ARROW_VS.values()])
    invaded = np.array([v for t in _FIG16_ARROW_VS.values() for v in t[1:]])
    assert virgin.size == 4 and invaded.size == 8
    assert virgin.mean() == pytest.approx(_FIG15_VIRGIN["vs"], rel=0.01)
    assert invaded.mean() == pytest.approx(_FIG15_INVADED["vs"], rel=0.01)
    # every single arrow, not just the means
    assert np.abs(virgin / _FIG15_VIRGIN["vs"] - 1).max() < 0.02
    assert np.abs(invaded / _FIG15_INVADED["vs"] - 1).max() < 0.02
    # and the two families do not overlap
    assert invaded.max() < virgin.min()


def test_a_slow_formation_dipole_sees_invasion_where_a_fast_one_cannot():
    """Figure 16 against figure 13, same source and the rock swapped.

    Figure 13 found a 16 cm invaded zone undetectable in a fast
    sandstone. Swapping in the slow one and changing nothing else, the
    same zone moves the arrival by 117 us and reorders the panel
    amplitudes by a factor of two.
    """

    def spread(t: tuple[float, float, float]) -> float:
        return max(t) / min(t)

    for f_khz in (3.0, 6.0, 7.5):
        fast = spread(_FIG13_PEAK_AMPLITUDE[f_khz])
        slow = spread(_FIG16_PEAK_AMPLITUDE[f_khz])
        assert fast < 1.05, f"fast is flat at {f_khz} kHz: {fast}"
        assert slow > 1.40, f"slow is not: {slow}"
    assert min(spread(t) for t in _FIG16_PEAK_AMPLITUDE.values()) > 1.4


def test_the_1_khz_invasion_delay_is_45x_larger_in_the_slow_formation():
    """The like-for-like comparison figure 14 could not supply.

    Figures 13(a) and 16(a) share source type, source frequency, offset
    and both invaded-zone thicknesses. Only the rock differs, so the
    ratio means something.
    """
    fast_us = _FIG13A_INVASION_LAG_US["16 cm"][0]
    slow_us = _FIG16A_INVASION_LAG_US["16 cm"][0]
    for name, (lag_us, corr) in _FIG16A_INVASION_LAG_US.items():
        assert lag_us > 0.0, f"{name}: invasion delays rather than advances"
        assert corr > 0.85, f"{name}: r = {corr}"
    assert _FIG16A_INVASION_LAG_US["16 cm"][0] > _FIG16A_INVASION_LAG_US["8 cm"][0]

    fast_frac = fast_us * 1e-3 / (_FIG13_OFFSET_M * 1e3 / _FIG2_ROCK["vs"])
    slow_frac = slow_us * 1e-3 / (_FIG16_OFFSET_M * 1e3 / _FIG15_VIRGIN["vs"])
    assert fast_frac < 0.001, f"fast: {fast_frac:.4%} of traveltime"
    assert slow_frac > 0.02, f"slow: {slow_frac:.4%} of traveltime"
    assert slow_frac / fast_frac > 30.0


def test_invasion_moves_the_slow_formation_energy_into_the_p_wavetrain():
    """The published mechanism, measured.

    Conclusion C says small velocity contrasts modify the internal
    dynamics more easily in slow formations "through an increase of the
    P wavetrain". Splitting each trace at its own arrow, P/S rises with
    thickness at every frequency at or above 3 kHz and with frequency at
    every thickness -- and at the top end the P wavetrain overtakes the
    shear packet outright.
    """
    for f_khz in (3.0, 6.0, 7.5):
        virgin, eight, sixteen = _FIG16_P_OVER_S[f_khz]
        assert virgin < eight < sixteen, f"{f_khz} kHz: {_FIG16_P_OVER_S[f_khz]}"
    # monotone in frequency for each model
    for col in range(3):
        series = [_FIG16_P_OVER_S[f][col] for f in (3.0, 6.0, 7.5)]
        assert series == sorted(series), f"column {col}: {series}"
    # the virgin trace never lets P win; invasion makes it win
    assert max(t[0] for t in _FIG16_P_OVER_S.values()) < 0.5
    assert _FIG16_P_OVER_S[6.0][2] > 1.0
    assert _FIG16_P_OVER_S[7.5][1] > 1.0 and _FIG16_P_OVER_S[7.5][2] > 1.0


def test_the_virgin_airy_arrival_is_frequency_independent():
    """Why the shear-packet peak can be read as the Airy phase.

    An Airy phase arrives at the stationary point of the group-velocity
    curve, which is a property of the medium, not of the source. The
    virgin packet peaks within 0.10 ms of 5.05 ms across a 7.5x change
    in source frequency. The invaded traces drift more, and that wider
    spread is carried into the tolerance of the fwap comparison below.
    """
    virgin = np.array(_FIG16_AIRY_MS["virgin"])
    assert np.ptp(virgin) / virgin.mean() < 0.025, f"{virgin}"
    for name in ("8 cm", "16 cm"):
        arr = np.array(_FIG16_AIRY_MS[name])
        assert np.ptp(arr) / arr.mean() < 0.08, f"{name}: {arr}"
        assert arr.min() > virgin.max(), "invasion delays the packet"


def test_figure_16_confirms_figure_8a_group_minimum_from_the_time_domain():
    """Two figures, two domains, one number.

    Figure 8a's published phase curve, differentiated, put the slow
    flexural group minimum at 992 m/s. Figure 16's virgin waveforms put
    the arrival at 5.05 ms over 5 m, which is 990 m/s. Neither reading
    used fwap, and they are 0.24 % apart.
    """
    measured = _FIG16_OFFSET_M / (float(np.mean(_FIG16_AIRY_MS["virgin"])) * 1e-3)
    published = _FIG8A_GROUP_MINIMUM[0]
    assert measured == pytest.approx(published, rel=0.01), (
        f"{measured:.1f} m/s from figure 16 against {published} from figure 8a"
    )


def test_the_layered_airy_prediction_is_still_the_looser_one():
    """Figure 15's verdict, re-measured after A.8.

    This test recorded the open-hole Airy prediction ~3 % late and the
    layered ones about twice that, all in the same direction. The
    roadmap-A.8 correction removed the open-hole error almost entirely
    and left the layered gap roughly where it was:

        model     predicted   measured range   error vs mean
        virgin      5.01 ms     5.01-5.11        -0.9 %
        8 cm        5.78 ms     5.37-5.68        +4.0 %
        16 cm       5.77 ms     5.45-5.84        +2.2 %

    So the ordering the test was written to pin survives -- the layered
    prediction is the looser one -- but it is now a comparison between
    a sub-percent open-hole number and a few-percent layered one, and
    the virgin prediction has crossed from late to slightly early.
    Compared against the measured MEAN rather than the latest arrival,
    since the virgin prediction now sits inside the measured range.
    """
    errors = {}
    for name, thickness in (("virgin", None), ("8 cm", 0.08), ("16 cm", 0.16)):
        v_g, _ = _fig16_group_minimum(thickness)
        predicted_ms = _FIG16_OFFSET_M / v_g * 1e3
        mean_ms = float(np.mean(_FIG16_AIRY_MS[name]))
        errors[name] = predicted_ms / mean_ms - 1.0
    assert abs(errors["virgin"]) < 0.015, errors
    for name in ("8 cm", "16 cm"):
        assert errors[name] > 0.015, errors
        assert abs(errors[name]) > 1.7 * abs(errors["virgin"]), errors


def test_only_the_invaded_models_are_out_of_reach_at_figure_16_lowest_frequency():
    """The panel that measures best is now half within reach.

    This test recorded a solver silent at 1 kHz for all three of
    figure 16's models, with onsets at 2.52 / 3.51 / 2.94 kHz. The
    roadmap-A.8 correction moved the OPEN-HOLE onset to 0.99 kHz, so
    the virgin model now resolves at the 1 kHz panel and returns the
    formation shear speed there.

    The two invaded models did not follow: their onsets are 3.74 and
    3.00 kHz, essentially unchanged, so the layered near-cutoff gap is
    a separate matter from the SV column -- and the panel with the
    compact wavelets and the frequency-independent Airy pick is still
    out of reach for them.
    """
    probe = np.array([1000.0])
    virgin_at_1khz = _fig16_flexural(None, probe)
    assert np.all(np.isfinite(virgin_at_1khz))
    assert virgin_at_1khz[0] == pytest.approx(_FIG15_VIRGIN["vs"], rel=1.0e-3)
    for thickness in (0.08, 0.16):
        v = _fig16_flexural(thickness, probe)
        assert not np.any(np.isfinite(v)), f"thickness {thickness}: {v}"
    freq = np.linspace(200.0, 12000.0, 241)
    onsets = {}
    for name, thickness in (("virgin", None), ("8 cm", 0.08), ("16 cm", 0.16)):
        v = _fig16_flexural(thickness, freq)
        onsets[name] = freq[np.isfinite(v)].min() / 1.0e3
    assert onsets["virgin"] < 1.0, onsets
    assert onsets["8 cm"] > 3.0 and onsets["16 cm"] > 2.5, onsets


def test_the_slow_flexural_curves_are_structurally_sound():
    """The contrast with figure 14, stated as an assertion.

    Figure 14's fast-formation quadrupole came back shredded: interior
    gaps, a sawtooth against the bracket ceiling, and a group velocity
    that went negative. The same package on the slow formation, open
    hole and layered alike, returns one contiguous run per model with a
    monotone phase velocity and a group velocity that never changes
    sign. Whatever is wrong here is an accuracy problem, and A.2's
    bracket is not implicated.
    """
    freq = np.linspace(200.0, 12000.0, 241)
    for thickness in (None, 0.08, 0.16):
        v = _fig16_flexural(thickness, freq)
        ok = np.isfinite(v)
        idx = np.where(ok)[0]
        assert int((np.diff(idx) > 1).sum()) == 0, f"{thickness}: interior gap"
        vv = v[ok]
        assert np.all(np.diff(vv) <= 1.0), f"{thickness}: phase not monotone"
        assert vv.max() <= _FIG15_VIRGIN["vs"] + 1.0
        ff = freq[ok]
        dv = np.diff(vv) / np.diff(ff)
        v_g = vv[:-1] / (1.0 - (ff[:-1] / vv[:-1]) * dv)
        assert np.all(v_g > 0.0), f"{thickness}: negative group velocity"


# ----------------------------------------------------------------------
# Figure 17: the slow-formation quadrupole, which fwap will not compute
#
# "Quadrupole source. Invaded zone effects in the presence of a slow
# sandstone. Iso-offset (z = 5m) of the waveforms obtained in the
# presence of the only virgin formation (1), and a 8 cm (2) and 16 cm
# (3) invaded zone. The source center frequency is successively equal to
# 1 kHz (a), 3 kHz (b), 6 kHz (c), and 7.5 kHz (d). Each series is
# normalized with respect to its own maximum denoted by 1.00."
#
# **The headline is a refusal, not a number.**
# `quadrupole_dispersion_layered` raises `ValueError` on this model
# before computing anything: the slow-formation branch requires every
# layer to be at least as fast in shear as the formation, and an invaded
# zone is by definition slower. Eight of the figure's twelve waveforms
# are therefore not merely inaccurate -- they are unrepresentable.
#
# The constraint is deliberate (`_validate_flexural_layers_stacked`,
# documented as plan G'.0) and it is right that a softer annulus is a
# harder regime. What makes it a finding rather than a limitation is the
# **asymmetry with the sister path**:
#
#   * `flexural_dispersion_layered` applies the same check only when
#     there are **two or more** layers -- the single-layer path
#     "documents but does not enforce it", in the code's own words.
#   * So the *identical* one-layer invaded-zone model is accepted at
#     n = 1 and rejected at n = 2.
#   * And the n = 1 answers obtained in that supposedly unsupported
#     regime are the ones figure 15 tied to the published curves at
#     **1.47-1.48 % rms**, as good as the open-hole solver.
#
# So the regime the n=2 validator refuses is one the n=1 code demonstrably
# handles. Figure 15(b) plots exactly the curves fwap will not produce --
# and the figure-15 work digitised only panel (a), so this was reachable
# then and missed. Figure 17 is what caught it.
#
# **What fwap can still do**, and it is not much: the virgin screw mode
# resolves from 5.25 kHz up (196 of 297 samples, no interior gaps, group
# velocity never negative -- structurally sound, like figure 16 and
# unlike figure 14). Panels (a) and (b) sit at 1 and 3 kHz, below that
# onset. **So of the twelve plotted waveforms, exactly two have a
# computable screw-mode phase velocity.**
#
# Predicting the virgin Airy arrival: the screw packet peaks at
# 4.91-4.99 ms across all four source frequencies -- frequency-
# independent, so it is the Airy phase -- giving 1008.1 m/s. fwap's
# group minimum of 954.2 m/s puts it at 5.24 ms, **+5.6 %**, against the
# flexural mode's +3.0 % on the same rock in figure 16.
#
# **The published data, which stands whatever fwap does.**
#
# Twelve arrows again calibrate the figure, and this is the tightest
# external agreement in the series: the four virgin arrows give
# 1193.6 m/s against `V_S` = 1201 (**-0.61 %**) and the eight invaded
# ones **1081.3** against 1081 (**+0.03 %**). Finding them needed a
# better discriminator than figure 16 used -- figure 17's dense
# high-frequency wavetrains produce blobs that pass a shape test, so the
# arrow is identified as the arrow-shaped component *not connected to
# the trace*. Re-running figure 16 with the stricter method reproduces
# its twelve values exactly, so nothing there needed correcting.
#
#     f_c        virgin   8 cm    16 cm    spread
#     1 kHz      0.156    0.496   1.000    **6.41x**
#     3 kHz      0.918    0.838   1.000     1.19x
#     6 kHz      1.000    0.455   0.546     2.20x
#     7.5 kHz    0.757    0.312   1.000     3.21x
#
# Digits and ink agree to 0.028. The 1 kHz virgin glyph is the one real
# ambiguity -- it could read 0.156 or 0.186, and the ink (0.184 +- 0.02
# on a 39-pixel excursion) cannot separate them. Comparing the glyph
# against known 5s and 8s elsewhere in the same figure settles it: the
# 8s in this font are two closed bowls (0.918, 0.838) and this is the
# open-topped 5 of 0.455.
#
# **Panel (a)'s 6.41x is the largest spread in the four waveform
# figures, and the virgin trace is the smallest one.** A slow-formation
# quadrupole at 1 kHz is barely excited; adding an invaded zone brings
# the screw mode's useful starting energy down into the source band and
# the response grows six-fold. That is the same mechanism figure 14
# named in a fast formation, here with the sign of the effect much
# larger.
#
# **The report's own claim for these panels, checked as written.** Page
# 229 says the P-wavetrain growth with invaded-zone thickness is "especially
# true with the quadrupole source (Figure 17c, d)". Splitting each trace
# at its own arrow, P/S grows from virgin to 16 cm by **26x** at 6 kHz
# and **69x** at 7.5 kHz, against the dipole's 15x and 13x in figure 16.
# Read as absolute level rather than growth the claim would look false --
# the dipole's P/S is larger at 6 kHz -- so the wording matters, and it
# is the growth the authors wrote.
#
# **No delays are quoted from this figure.** Panels (b)-(d) give 8 cm
# lags of +868.9, +867.8 and +862.4 us at 3, 6 and 7.5 kHz -- constant
# to +-3 us across a 2.5x change in source frequency, the cycle-hopping
# signature figure 14 established. Panel (a) clears r = 0.8 on both
# traces but its 8 cm lag (+357 us) contradicts the peak-time shift of
# the same trace (+100 us), so it is not quoted either.
# ----------------------------------------------------------------------

#: Figure 17: printed peak-amplitude scale factors (virgin, 8 cm, 16 cm)
#: keyed by source centre frequency in kHz. Digits read from the page and
#: confirmed against the plotted ink to 0.028.
_FIG17_PEAK_AMPLITUDE = {
    1.0: (0.156, 0.496, 1.000),
    3.0: (0.918, 0.838, 1.000),
    6.0: (1.000, 0.455, 0.546),
    7.5: (0.757, 0.312, 1.000),
}

#: Figure 17: P-wavetrain over shear-packet amplitude, split at the arrow.
_FIG17_P_OVER_S = {
    1.0: (0.13, 0.14, 0.04),
    3.0: (0.03, 0.07, 0.07),
    6.0: (0.04, 0.12, 1.05),
    7.5: (0.04, 0.59, 2.76),
}

#: Figure 17: shear speed from each drawn arrow at 5 m (m/s).
_FIG17_ARROW_VS = {
    1.0: (1188.2, 1078.1, 1080.2),
    3.0: (1195.5, 1083.0, 1084.1),
    6.0: (1191.1, 1075.9, 1080.2),
    7.5: (1199.7, 1084.1, 1085.2),
}

#: Figure 17: time of the screw-packet peak (ms) at 5 m for the virgin
#: model, at each of the four source centre frequencies.
_FIG17_VIRGIN_AIRY_MS = (4.99, 4.91, 4.99, 4.95)

#: Lowest frequency (kHz) at which fwap resolves the virgin slow screw
#: mode, against the onset figure 17(b) shows a wavetrain at.
_FIG17_VIRGIN_ONSET_KHZ = 5.25

_FIG17_OFFSET_M = 5.0


def test_fig17_scale_factors_are_internally_consistent():
    """The transcription, including the one ambiguous glyph.

    Panel (a)'s virgin factor could read 0.156 or 0.186 and the ink
    cannot separate them on a 39-pixel excursion; the glyph shape does,
    against known 5s and 8s in the same figure. Either reading gives the
    same conclusion, and the assertions below hold for both.
    """
    for f_khz, triple in _FIG17_PEAK_AMPLITUDE.items():
        assert max(triple) == pytest.approx(1.0), f"{f_khz} kHz: {triple}"
        assert min(triple) > 0.0, f"{f_khz} kHz: {triple}"
        assert sum(v == 1.0 for v in triple) == 1, f"{f_khz} kHz: {triple}"


def test_the_figure_17_arrows_are_the_tightest_external_tie_in_the_series():
    """Twelve arrowheads, two published velocities, no solver.

    The eight invaded arrows average 1081.3 m/s against table 1's 1081.
    Finding them needed a stricter discriminator than figure 16 used --
    this figure's dense wavetrains produce blobs that pass a shape test,
    so the arrow is the arrow-shaped component *not connected to the
    trace*. Figure 16 re-measured that way reproduces its twelve values
    exactly, so its record needed no correction.
    """
    virgin = np.array([t[0] for t in _FIG17_ARROW_VS.values()])
    invaded = np.array([v for t in _FIG17_ARROW_VS.values() for v in t[1:]])
    assert virgin.size == 4 and invaded.size == 8
    assert invaded.mean() == pytest.approx(_FIG15_INVADED["vs"], rel=0.005)
    assert virgin.mean() == pytest.approx(_FIG15_VIRGIN["vs"], rel=0.01)
    assert np.abs(invaded / _FIG15_INVADED["vs"] - 1).max() < 0.01
    assert invaded.max() < virgin.min(), "the two families do not overlap"


def test_invasion_multiplies_the_slow_quadrupole_response_at_1_khz():
    """Panel (a): the largest spread in any of the four waveform figures.

    A slow-formation quadrupole at 1 kHz is barely excited -- the virgin
    trace is the *smallest* in its panel at 0.156 -- and a 16 cm invaded
    zone brings the screw mode's useful starting energy down into the
    source band, multiplying the response by more than six.
    """
    virgin, eight, sixteen = _FIG17_PEAK_AMPLITUDE[1.0]
    assert virgin < eight < sixteen
    assert sixteen / virgin > 5.0
    others = [max(t) / min(t) for f, t in _FIG17_PEAK_AMPLITUDE.items() if f != 1.0]
    assert max(t := _FIG17_PEAK_AMPLITUDE[1.0]) / min(t) > max(others)
    # larger than anything the fast-formation quadrupole managed
    assert max(t) / min(t) > max(
        max(x) / min(x) for x in _FIG14_PEAK_AMPLITUDE.values()
    )


def test_the_quadrupole_p_wavetrain_grows_faster_than_the_dipole_s():
    """The report's claim for figures 17(c, d), read as it is written.

    "The increase of the absolute and relative amplitude of the P
    wavetrain with the thickness of the invaded zone ... is especially
    true with the quadrupole source." It is the *growth* that is
    compared, not the level -- read as level the claim would look false,
    since the dipole's P/S is larger at 6 kHz.
    """
    for f_khz in (6.0, 7.5):
        quad = _FIG17_P_OVER_S[f_khz]
        dip = _FIG16_P_OVER_S[f_khz]
        quad_growth = quad[2] / quad[0]
        dip_growth = dip[2] / dip[0]
        assert quad_growth > dip_growth, (
            f"{f_khz} kHz: quadrupole {quad_growth:.0f}x vs dipole {dip_growth:.0f}x"
        )
        assert quad[2] > 1.0, "P overtakes S with a 16 cm zone"
    # and read as level rather than growth it would go the other way
    assert _FIG16_P_OVER_S[6.0][2] > _FIG17_P_OVER_S[6.0][2]


def test_quadrupole_dispersion_layered_accepts_a_single_invaded_zone():
    """A.6, fixed: the n=2 slow-formation path takes an invaded zone.

    An invaded zone is slower in shear than the rock it replaces. The
    per-layer ``layer.vs >= vs`` constraint now applies to the
    multi-layer path only, which is what this function's docstring
    always said and what ``flexural_dispersion_layered`` always did.
    """
    from fwap.cylindrical_solver import quadrupole_dispersion_layered

    fluid = dict(vf=1500.0, rho_f=1000.0)
    freq = np.array([6000.0, 7500.0])
    for thickness in (0.08, 0.16):
        mode = quadrupole_dispersion_layered(
            freq,
            **_FIG15_VIRGIN,
            **fluid,
            a=0.10,
            layers=(BoreholeLayer(**_FIG15_INVADED, thickness=thickness),),
        )
        v = 1.0 / mode.slowness
        assert np.isfinite(v).all(), f"{thickness}: {v}"
        # below the invaded shear speed, above the fluid: a bound mode
        assert np.all(v < _FIG15_INVADED["vs"])
        assert np.all(v > 800.0)

    # two soft layers still raise -- the multi-layer guard is untouched
    with pytest.raises(ValueError, match="at least as fast in shear"):
        quadrupole_dispersion_layered(
            freq,
            **_FIG15_VIRGIN,
            **fluid,
            a=0.10,
            layers=(
                BoreholeLayer(**_FIG15_INVADED, thickness=0.08),
                BoreholeLayer(**_FIG15_INVADED, thickness=0.08),
            ),
        )


def test_the_same_model_is_now_accepted_at_both_n1_and_n2():
    """The asymmetry A.6 was filed for, closed.

    Before the fix, the identical one-layer invaded-zone model was
    accepted at n=1 and rejected at n=2. Both paths now take it, and
    both put the phase velocity below the invaded shear speed.
    """
    from fwap.cylindrical_solver import quadrupole_dispersion_layered

    fluid = dict(vf=1500.0, rho_f=1000.0)
    freq = np.array([6000.0])
    layers = (BoreholeLayer(**_FIG15_INVADED, thickness=0.16),)

    n1 = (
        1.0
        / flexural_dispersion_layered(
            freq, **_FIG15_VIRGIN, **fluid, a=0.10, layers=layers
        ).slowness
    )
    n2 = (
        1.0
        / quadrupole_dispersion_layered(
            freq, **_FIG15_VIRGIN, **fluid, a=0.10, layers=layers
        ).slowness
    )
    assert np.isfinite(n1).all() and np.isfinite(n2).all()
    for v in (n1, n2):
        assert np.all(v < _FIG15_INVADED["vs"])
    # the screw mode is the faster of the two at a given frequency
    assert n2[0] > n1[0]


def test_six_of_figure_17s_twelve_waveforms_now_have_a_phase_velocity():
    """What the fix buys on this figure, counted.

    Before: two, both virgin. The eight invaded waveforms raised before
    computing. After: six -- the two invaded models resolve at 6 and
    7.5 kHz apiece. The remaining six are all below the screw mode's
    onset, which this fix does not touch and which the near-cutoff gap
    already covers.
    """
    from fwap.cylindrical_solver import (
        quadrupole_dispersion,
        quadrupole_dispersion_layered,
    )

    fluid = dict(vf=1500.0, rho_f=1000.0)
    probe = 1.0e3 * np.array([1.0, 3.0, 6.0, 7.5])
    computable = int(
        np.isfinite(
            quadrupole_dispersion(probe, **_FIG15_VIRGIN, **fluid, a=0.10).slowness
        ).sum()
    )
    assert computable == 2, "the virgin model still resolves only 6 and 7.5 kHz"
    for thickness in (0.08, 0.16):
        m = quadrupole_dispersion_layered(
            probe,
            **_FIG15_VIRGIN,
            **fluid,
            a=0.10,
            layers=(BoreholeLayer(**_FIG15_INVADED, thickness=thickness),),
        )
        computable += int(np.isfinite(m.slowness).sum())
    assert computable == 6, f"{computable} of 12 plotted waveforms"


def test_the_virgin_screw_airy_arrival_is_late_by_a_percent():
    """The one forward prediction figure 17 allows, now much tighter.

    The screw packet peaks within 0.05 ms of 4.96 ms across all four
    source frequencies, so it is the Airy phase. fwap's group minimum
    used to put it at 5.24 ms, **+5.6 %** and late; after the
    roadmap-A.8 correction it puts it at 5.03 ms, **+1.35 %** and still
    late. The onset moved with it, from 5.25 kHz to 3.85 kHz against a
    published 3.74.
    """
    from fwap.cylindrical_solver import quadrupole_dispersion

    arr = np.array(_FIG17_VIRGIN_AIRY_MS)
    assert np.ptp(arr) / arr.mean() < 0.02, f"frequency-independent: {arr}"

    freq = np.linspace(200.0, 15000.0, 297)
    fluid = dict(vf=1500.0, rho_f=1000.0)
    v = 1.0 / quadrupole_dispersion(freq, **_FIG15_VIRGIN, **fluid, a=0.10).slowness
    idx = np.where(np.isfinite(v))[0]
    seg = max(np.split(idx, np.where(np.diff(idx) > 1)[0] + 1), key=len)
    ff, vv = freq[seg], v[seg]
    v_g = 1.0 / np.gradient(ff / vv, ff)
    predicted_ms = _FIG17_OFFSET_M / v_g.min() * 1e3
    error = predicted_ms / arr.mean() - 1.0
    assert 0.005 < error < 0.03, f"{predicted_ms:.2f} ms vs {arr.mean():.2f}"
    # structurally sound, unlike figure 14's fast-formation quadrupole
    assert np.all(v_g > 0.0)
    assert int((np.diff(idx) > 1).sum()) == 0, "no interior gaps"
    onset = freq[np.isfinite(v)].min() / 1.0e3
    # A.8 moved the onset down from 5.25 kHz to the published 3.74.
    assert onset == pytest.approx(3.74, abs=0.2)


# ----------------------------------------------------------------------
# Figure 15(b) and A.6: the reference the fix rests on
#
# "Invaded zone effects with a slow sandstone. Dispersion and attenuation
# of the flexural (a) and screw (b) modes in the presence of: (1) the
# only virgin formation; (2) a 8 cm thick invaded zone; (3) a 16 cm thick
# invaded zone; (4) the only invaded zone."
#
# Panel (b) is the screw mode -- the curves `quadrupole_dispersion_layered`
# used to refuse to compute. The figure-15 work digitised only panel (a),
# which is why A.6 went unnoticed until figure 17.
#
# **Calibration.** The panel's x-axis runs 2-10 kHz, not 0-10: nine evenly
# spaced ticks with the "5" label under the fourth and "10" under the
# ninth. Physics confirms it -- curves 1, 2 and 3 leave the axis at
# v = 0.80 (= 1201/1500, the *virgin* shear speed, as the report says on
# p. 228) and curve 4 at 0.72 (= 1081/1500, the invaded one). Tick fits:
# x to 0.008 kHz, y to 0.00006 normalised.
#
# **Curve identification, checked rather than assumed.** The four solid
# curves merge, so each is followed from its own start with slope
# prediction, stopping where a neighbour comes within two line widths.
# The 16 cm trace was verified to sit on run 3 of 4 at every sampled
# frequency -- it is curve 3, not curve 4. That check mattered: on rms
# alone the 16 cm data fits an *invaded-only* prediction slightly better
# than a layered one, because curves 3 and 4 converge to within 0.6-0.8 %
# over the band where fwap and the trace overlap. Curve 2 is 4.2 % from
# its nearest neighbour there, which is why the 8 cm tie carries the
# weight and the 16 cm one only corroborates.
#
# **What this settles.** With the single-layer guard removed, the n=2
# layered path returns, against these curves:
#
#     model               band          rms      median
#     virgin (control)    5.25-9.75    1.29 %    +0.35 %
#     8 cm invaded        5.75-8.00    **0.58 %**  +0.29 %
#     16 cm invaded       5.00-5.50    2.12 %    +2.07 %   (3 points)
#
# The 8 cm figure is better than the same solver's own virgin control on
# the same figure. The code was refusing a regime it computes correctly,
# and its docstring had said "(multi-layer only)" all along.
#
# The fix does not touch the onset: fwap resolves the 8 cm model from
# 5.6 kHz against a published 3.4 kHz, which is the slow screw mode's
# near-cutoff gap already recorded at _NEAR_CUTOFF_GAPS.
# ----------------------------------------------------------------------

#: Figure 15(b): screw-mode phase velocity (Hz, m/s), read off the
#: published curves. Virgin is the control; the two invaded models are
#: the ones the n=2 layered path used to refuse.
_FIG15B_SCREW_PHASE = {
    "virgin": (
        (4.5e3, 1185.4),
        (5.0e3, 1171.7),
        (5.5e3, 1157.2),
        (6.0e3, 1144.6),
        (6.5e3, 1132.0),
        (7.0e3, 1121.5),
        (7.5e3, 1112.9),
        (8.0e3, 1104.3),
        (8.5e3, 1095.8),
        (9.0e3, 1090.2),
        (9.5e3, 1085.3),
    ),
    "invaded_8cm": (
        (4.0e3, 1153.9),
        (4.5e3, 1123.2),
        (5.0e3, 1095.6),
        (5.5e3, 1069.8),
        (6.0e3, 1049.0),
        (6.5e3, 1029.2),
        (7.0e3, 1014.3),
        (7.5e3, 1002.5),
        (8.0e3, 993.4),
    ),
    "invaded_16cm": (
        (4.0e3, 1099.5),
        (4.5e3, 1071.0),
        (5.0e3, 1049.6),
        (5.5e3, 1031.9),
    ),
}

#: Figure 15(b): thickness of the annulus for each model, or None for
#: the open-hole virgin control.
_FIG15B_THICKNESS = {"virgin": None, "invaded_8cm": 0.08, "invaded_16cm": 0.16}

#: Separation between curve 3 (16 cm) and its nearest neighbour, curve 4
#: (invaded rock alone), as a fraction, over 5.0-5.5 kHz. This is why the
#: 16 cm tie corroborates rather than decides.
_FIG15B_CURVE_3_4_SEPARATION = 0.008


def test_fig15b_curves_start_at_the_published_shear_speeds():
    """Anchor the digitisation before anything is built on it.

    Every traced sample must lie below the shear speed of the medium the
    mode ends up sampling, and the invaded models must lie below the
    virgin one at matched frequency -- a thicker slow annulus cannot
    speed the mode up.
    """
    for name, table in _FIG15B_SCREW_PHASE.items():
        v = np.array([x for _, x in table])
        assert np.all(v < _FIG15_VIRGIN["vs"]), f"{name}: {v.max()}"
        assert np.all(np.diff(v) < 0.0), f"{name} is not monotone: {v}"
    common = [4.5e3, 5.0e3, 5.5e3]
    for f in common:
        virgin = dict(_FIG15B_SCREW_PHASE["virgin"])[f]
        eight = dict(_FIG15B_SCREW_PHASE["invaded_8cm"])[f]
        sixteen = dict(_FIG15B_SCREW_PHASE["invaded_16cm"])[f]
        assert sixteen < eight < virgin, f"{f}: {sixteen}, {eight}, {virgin}"


@pytest.mark.parametrize("model", sorted(_FIG15B_SCREW_PHASE))
def test_quadrupole_layered_tracks_figure_15b(model):
    """A.6 fixed, measured against the figure that plots it.

    The virgin row is a control -- it uses the open-hole solver, which
    figure 8a already tied at 0.94 % rms on this rock, so it prices the
    digitisation itself. The two invaded rows go through the layered path
    that used to raise `ValueError` before computing anything.
    """
    from fwap.cylindrical_solver import (
        quadrupole_dispersion,
        quadrupole_dispersion_layered,
    )

    table = _FIG15B_SCREW_PHASE[model]
    freq = np.array([f for f, _ in table])
    ref = np.array([v for _, v in table])
    fluid = dict(vf=1500.0, rho_f=1000.0)
    thickness = _FIG15B_THICKNESS[model]
    if thickness is None:
        got = (
            1.0 / quadrupole_dispersion(freq, **_FIG15_VIRGIN, **fluid, a=0.10).slowness
        )
    else:
        got = (
            1.0
            / quadrupole_dispersion_layered(
                freq,
                **_FIG15_VIRGIN,
                **fluid,
                a=0.10,
                layers=(BoreholeLayer(**_FIG15_INVADED, thickness=thickness),),
            ).slowness
        )
    ok = np.isfinite(got)
    # the 16 cm model resolves only its top two rows -- the rest of its
    # published band is below fwap's onset, which this fix does not touch
    assert ok.sum() >= 2, f"{model}: only {ok.sum()} samples resolved"
    rel = (got[ok] - ref[ok]) / ref[ok]
    rms = float(np.sqrt((rel**2).mean()))
    assert rms < 0.035, f"{model}: rms {rms:.2%}"


def test_the_layered_screw_ties_hold_after_the_sv_correction():
    """The evidence that the refused regime was computed correctly.

    A.6 removed a guard that blocked ``quadrupole_dispersion_layered``
    at one layer, and the argument that the guard was over-strict rested
    on the 8 cm model scoring 0.58 % rms against the open-hole virgin
    control's 1.29 % on this same figure.

    The roadmap-A.8 correction improved all three by an order of
    magnitude and, in doing so, reordered them:

        virgin (open-hole control)   1.29 -> 0.055 %
        8 cm invaded                 0.58 -> 0.136 %
        16 cm invaded                2.12 -> 0.197 %

    The open-hole control is now the tightest, which is what one should
    expect: it solves one homogeneous half-space, while the layered
    models also carry the invaded-zone row transcribed from the paper's
    scanned table 1. The A.6 conclusion is unaffected -- the layered
    path ties the published curves it used to refuse, to within a
    fifth of a percent.
    """
    from fwap.cylindrical_solver import (
        quadrupole_dispersion,
        quadrupole_dispersion_layered,
    )

    fluid = dict(vf=1500.0, rho_f=1000.0)

    def score(model: str) -> float:
        table = _FIG15B_SCREW_PHASE[model]
        freq = np.array([f for f, _ in table])
        ref = np.array([v for _, v in table])
        th = _FIG15B_THICKNESS[model]
        if th is None:
            got = (
                1.0
                / quadrupole_dispersion(freq, **_FIG15_VIRGIN, **fluid, a=0.10).slowness
            )
        else:
            got = (
                1.0
                / quadrupole_dispersion_layered(
                    freq,
                    **_FIG15_VIRGIN,
                    **fluid,
                    a=0.10,
                    layers=(BoreholeLayer(**_FIG15_INVADED, thickness=th),),
                ).slowness
            )
        ok = np.isfinite(got)
        rel = (got[ok] - ref[ok]) / ref[ok]
        return float(np.sqrt((rel**2).mean()))

    eight = score("invaded_8cm")
    sixteen = score("invaded_16cm")
    control = score("virgin")
    assert control < 0.002, f"virgin control rms {control:.3%}"
    assert eight < 0.003, f"8 cm layered rms {eight:.3%}"
    assert sixteen < 0.004, f"16 cm layered rms {sixteen:.3%}"
    # Both layered ties are within a small multiple of the open-hole
    # control, which is the claim A.6 needed.
    assert eight < 4.0 * control
    assert sixteen < 5.0 * control


def test_the_16cm_tie_is_recorded_as_corroborating_not_deciding():
    """State the limit rather than let the number stand unqualified.

    Over the band where fwap and the traced curve overlap, curve 3
    (16 cm) and curve 4 (the invaded rock alone) run within 0.8 % of each
    other. The trace was verified to sit on curve 3 by run ordering, but
    an rms comparison at that separation cannot by itself distinguish a
    layered prediction from an invaded-only one -- so the 8 cm tie is the
    one the fix rests on.
    """
    from fwap.cylindrical_solver import quadrupole_dispersion

    table = _FIG15B_SCREW_PHASE["invaded_16cm"]
    freq = np.array([f for f, _ in table])
    ref = np.array([v for _, v in table])
    fluid = dict(vf=1500.0, rho_f=1000.0)
    # curve 4: the invaded rock as a half-space, no layered path at all
    curve4 = (
        1.0 / quadrupole_dispersion(freq, **_FIG15_INVADED, **fluid, a=0.10).slowness
    )
    ok = np.isfinite(curve4)
    assert ok.any(), "curve 4 resolves somewhere in the band"
    gap = np.abs(curve4[ok] - ref[ok]) / ref[ok]
    # 3.1 % at 4.0 kHz narrowing to 0.7 % by 5.5 kHz. The A.8 correction
    # sharpened curve 4 itself, so this now reads the true curve-3/4
    # separation rather than the solver's error on top of it.
    assert gap.max() < 0.035, f"worst {gap.max():.2%}"
    assert gap.min() < 0.01, (
        "curves 3 and 4 converge to within 1 % at the top of the band, "
        f"so an rms comparison cannot separate them: closest {gap.min():.2%}"
    )
    assert _FIG15B_CURVE_3_4_SEPARATION < 0.01


def test_the_fast_formation_marcher_is_grid_independent_at_both_orders():
    """What A.2 delivered at n=1, and what A.8 changed at n=2.

    A dispersion solver's answer at a frequency must not depend on which
    other frequencies were asked for in the same call. After A.2 the
    n=1 fast-formation branch satisfied that exactly: grids of 5 to 161
    points over the same band, and grids starting anywhere from 1 to
    5 kHz, all return the same 10 kHz value to **0.000 %**.

    The n=2 path used to move by a few percent with the grid, and then
    -- after roadmap A.8 -- to converge or not depending on the grid.
    Roadmap A.7 removed the remaining half: it was seeding off sign
    changes in round-off, because the `n = 2` determinant is real and
    the marcher was tracking its imaginary part. Both orders now
    satisfy the property exactly, on every grid.
    """
    from fwap import flexural_dispersion, quadrupole_dispersion

    rock = dict(vp=4500.0, vs=2600.0, rho=2400.0, vf=1500.0, rho_f=1000.0, a=0.10)

    def at_10khz(fn, grids):
        out = []
        for f in grids:
            v = 1.0 / fn(f, **rock).slowness
            out.append(v[-1])
        return np.array(out)

    by_density = [np.linspace(2000.0, 10000.0, n) for n in (5, 9, 21, 41, 81, 161)]
    by_start = [np.linspace(lo, 10000.0, 41) for lo in (1e3, 2e3, 3e3, 4e3, 5e3)]

    for grids in (by_density, by_start):
        flex = at_10khz(flexural_dispersion, grids)
        assert np.isfinite(flex).all(), f"n=1 lost the branch: {flex}"
        assert np.ptp(flex) / flex.mean() < 1.0e-9, (
            f"n=1 must not depend on the grid; spread {np.ptp(flex):.3e} m/s"
        )

    quad = at_10khz(quadrupole_dispersion, by_density + by_start)
    assert np.isfinite(quad).all(), f"n=2 lost the branch: {quad}"
    assert np.ptp(quad) / quad.mean() < 1.0e-9, (
        f"n=2 must not depend on the grid either; spread {np.ptp(quad):.3e} m/s"
    )


# ----------------------------------------------------------------------
# Figures 20 and 21: the cased hole, externally (roadmap A.1)
#
# Until now every cased-hole number in this suite was scored against
# fwap itself -- A.9's leaky branch had no published curve at all, and
# A.7's screw path had been silent, so nothing external had ever touched
# the propagator stack. Figures 20 and 21 are that measurement, and they
# were in the paper the whole time; `plans/guides.md` section 11 listed
# them as unread.
#
# Geometry, from p. 230: "the inner borehole radius is decreased by the
# amount of the casing and cement thickness ... the original borehole
# radius is 10 cm". So 10 cm is the FORMATION contact and the fluid
# column shrinks -- the 3 cm-cement case has a = 5.98 cm, not 10 cm.
# Getting this backwards is the one mistake that would quietly ruin the
# comparison, which is why it is quoted here rather than inferred.
#
# The layer properties are Table 1's own casing and cement rows, read
# from the page (p. 227). The cased fixtures elsewhere in this file use
# invented values; these do not.
#
# Digitised at 600 dpi by tracing each curve with a momentum-following
# tracer, then checked two ways before being trusted, per guides section
# 5: the open-hole curve in each figure is case (1), which fwap's
# already-validated open-hole solvers compute independently, and every
# trace was tested for the slope discontinuity a curve-jump leaves. Both
# checks fired on the first attempt at figure 20's open-hole curve --
# the tracer had jumped onto a steeper neighbour through the knee, the
# anchor showed -10 % there and the kink test put the jump at 6.32-6.35
# kHz without reference to fwap at all. That curve is therefore recorded
# only above 6.5 kHz.
#
# Uncertainty: about +-8 m/s where the curves are shallow, and much
# worse on the steep descents, where a 0.05 kHz read error is 30 m/s.
# ----------------------------------------------------------------------

#: Table 1, p. 227. Steel casing and the two cements, as published.
_S88_CASING = BoreholeLayer(vp=6098.0, vs=3354.0, rho=7500.0, thickness=0.0102)
_S88_CEMENT_1 = dict(vp=2823.0, vs=1729.0, rho=1920.0)
_S88_CEMENT_2 = dict(vp=2823.0, vs=1555.0, rho=1730.0)
#: Table 1's fast sandstone, and the borehole of figures 20 and 21.
_S88_FAST = dict(vp=4878.0, vs=2601.0, rho=2160.0)
_S88_BOREHOLE = dict(vf=1500.0, rho_f=1000.0)
_S88_OUTER = 0.10


def _s88_cased(cement_thickness: float) -> tuple[float, tuple[BoreholeLayer, ...]]:
    """Fluid radius and layer stack for a figure 20/21 cased case."""
    a = _S88_OUTER - _S88_CASING.thickness - cement_thickness
    cement = BoreholeLayer(**_S88_CEMENT_1, thickness=cement_thickness)
    return a, (_S88_CASING, cement)


_FIG20A_OPEN_HOLE = (
    (6500.0, 1802.7),
    (7000.0, 1741.0),
    (7500.0, 1695.2),
    (8000.0, 1662.3),
    (8500.0, 1636.7),
    (9000.0, 1616.4),
    (9500.0, 1600.3),
    (10000.0, 1581.8),
    (10500.0, 1568.4),
    (11000.0, 1559.2),
    (11500.0, 1550.9),
    (12000.0, 1546.3),
    (12500.0, 1544.1),
    (13000.0, 1538.0),
    (13500.0, 1531.1),
    (14000.0, 1526.6),
    (14500.0, 1519.7),
    (15000.0, 1517.5),
)

_FIG20A_CEMENT_1CM = (
    (4000.0, 2590.4),
    (4500.0, 2586.1),
    (5000.0, 2544.6),
    (5500.0, 2429.8),
    (6000.0, 2269.8),
    (6500.0, 2113.8),
    (7000.0, 1976.8),
    (7500.0, 1877.5),
    (8000.0, 1805.7),
    (8500.0, 1753.7),
    (9000.0, 1708.4),
    (9500.0, 1676.9),
    (10000.0, 1641.3),
    (10500.0, 1618.5),
    (11000.0, 1600.3),
    (11500.0, 1583.4),
    (12000.0, 1568.4),
    (12500.0, 1544.1),
    (13000.0, 1538.0),
    (13500.0, 1531.1),
    (14000.0, 1526.6),
    (14500.0, 1519.7),
    (15000.0, 1517.5),
)

_FIG20A_CEMENT_3CM = (
    (4000.0, 2590.4),
    (4500.0, 2586.1),
    (5000.0, 2583.8),
    (5500.0, 2563.3),
    (6000.0, 2526.7),
    (6500.0, 2460.1),
    (7000.0, 2365.5),
    (7500.0, 2245.1),
    (8000.0, 2131.3),
    (8500.0, 2020.6),
    (9000.0, 1930.9),
    (9500.0, 1859.2),
    (10000.0, 1793.4),
    (10500.0, 1744.1),
    (11000.0, 1703.5),
    (11500.0, 1671.5),
    (12000.0, 1641.3),
    (12500.0, 1613.9),
    (13000.0, 1592.8),
    (13500.0, 1577.5),
    (14000.0, 1563.8),
    (14500.0, 1550.9),
    (15000.0, 1544.1),
)

_FIG21A_OPEN_HOLE = (
    (8000.0, 2331.9),
    (9000.0, 2090.8),
    (10000.0, 1927.2),
    (11000.0, 1820.1),
    (12000.0, 1747.5),
    (13000.0, 1693.8),
    (14000.0, 1656.2),
    (15000.0, 1626.0),
    (16000.0, 1603.4),
    (17000.0, 1583.6),
    (18000.0, 1574.1),
    (19000.0, 1562.8),
    (20000.0, 1551.5),
)

_FIG21A_CEMENT_1CM = (
    (8000.0, 2487.8),
    (9000.0, 2308.4),
    (10000.0, 2117.0),
    (12000.0, 1851.4),
    (15000.0, 1662.5),
    (16000.0, 1630.4),
    (17000.0, 1601.7),
    (18000.0, 1574.1),
    (19000.0, 1562.8),
    (20000.0, 1551.5),
)

_FIG21A_CEMENT_3CM = (
    (8000.0, 2487.8),
    (9000.0, 2308.4),
    (10000.0, 2157.6),
    (12000.0, 1923.7),
    (15000.0, 1725.4),
    (16000.0, 1689.2),
    (17000.0, 1657.7),
    (18000.0, 1636.3),
    (19000.0, 1621.5),
    (20000.0, 1607.9),
)


def _s88_score(table, model_freq, model_slowness, drop=()):
    """Median/rms/worst relative difference against a digitised table."""
    freq = np.array([f for f, _ in table])
    ref = np.array([v for _, v in table])
    keep = np.ones(freq.size, bool)
    for lo, hi in drop:
        keep &= ~((freq >= lo) & (freq <= hi))
    freq, ref = freq[keep], ref[keep]
    finite = np.isfinite(model_slowness)
    model = np.interp(freq, model_freq[finite], 1.0 / model_slowness[finite])
    rel = np.abs(model / ref - 1.0)
    return float(np.median(rel)), float(np.sqrt((rel**2).mean())), float(rel.max())


def test_figure_20_and_21_open_hole_curves_anchor_the_digitisation():
    """Check the digitisation before letting it judge the cased solver.

    Case (1) of each figure is the open hole, which the already-validated
    open-hole solvers compute without any propagator stack. It is the
    anchor, and it is what caught a tracer that had jumped curves.

    Both figures also start on the formation shear speed, which is the
    second anchor and an independent one: 2601 m/s from Table 1 against
    a plateau read off the plot.
    """
    freq20 = np.array([f for f, _ in _FIG20A_OPEN_HOLE])
    res20 = flexural_dispersion(freq20, **_S88_FAST, **_S88_BOREHOLE, a=_S88_OUTER)
    median, rms, worst = _s88_score(_FIG20A_OPEN_HOLE, freq20, res20.slowness)
    assert median < 0.006, f"figure 20 open hole: median {100 * median:.2f} %"
    assert worst < 0.012, f"figure 20 open hole: worst {100 * worst:.2f} %"

    from fwap.cylindrical_solver import quadrupole_dispersion

    freq21 = np.array([f for f, _ in _FIG21A_OPEN_HOLE])
    res21 = quadrupole_dispersion(freq21, **_S88_FAST, **_S88_BOREHOLE, a=_S88_OUTER)
    median, rms, worst = _s88_score(_FIG21A_OPEN_HOLE, freq21, res21.slowness)
    assert median < 0.010, f"figure 21 open hole: median {100 * median:.2f} %"
    assert worst < 0.025, f"figure 21 open hole: worst {100 * worst:.2f} %"

    # The plateau is the formation shear speed, on both figures.
    assert _FIG20A_CEMENT_3CM[0][1] == pytest.approx(_S88_FAST["vs"], rel=0.01)
    assert _FIG21A_CEMENT_3CM[0][1] / _S88_FAST["vs"] < 1.0


@pytest.mark.parametrize(
    ("label", "thickness", "table", "median_max", "worst_max"),
    [
        ("1 cm cement", 0.01, "_FIG20A_CEMENT_1CM", 0.005, 0.012),
        ("3 cm cement", 0.03, "_FIG20A_CEMENT_3CM", 0.005, 0.012),
    ],
)
def test_cased_flexural_matches_figure_20(
    label, thickness, table, median_max, worst_max
):
    """The cased dipole against a published curve, for the first time.

    Everything the cased n=1 path had been scored on until now was
    internal -- A.9's leaky branch was validated against the bound solver
    it takes over from, because no published cased curve had been read.
    This is that curve. It covers the whole plotted band, and it lands at
    the digitisation floor: the residual is the same size as the
    open-hole anchor's, which is what "as close as the figure can
    resolve" looks like.
    """
    reference = globals()[table]
    freq = np.array([f for f, _ in reference])
    a, layers = _s88_cased(thickness)
    res = flexural_dispersion_layered(
        freq, **_S88_FAST, **_S88_BOREHOLE, a=a, layers=layers
    )
    assert np.isfinite(res.slowness).all(), (
        f"{label}: {int((~np.isfinite(res.slowness)).sum())} frequencies empty"
    )
    median, rms, worst = _s88_score(reference, freq, res.slowness)
    assert median < median_max, f"{label}: median {100 * median:.2f} %"
    assert worst < worst_max, f"{label}: worst {100 * worst:.2f} %"


@pytest.mark.parametrize(
    ("label", "thickness", "table", "median_max", "worst_max"),
    [
        ("1 cm cement", 0.01, "_FIG21A_CEMENT_1CM", 0.014, 0.025),
        ("3 cm cement", 0.03, "_FIG21A_CEMENT_3CM", 0.008, 0.020),
    ],
)
def test_cased_screw_matches_figure_21(label, thickness, table, median_max, worst_max):
    """The path A.7 had forced into silence, measured from outside.

    ``plans/guides.md`` section 11 called figure 21 "the only external
    measure of how wrong that path was", and it was never read. Before
    A.7 this configuration returned nothing at all, so there was no
    number to compare; it now covers 8-20 kHz.
    """
    from fwap.cylindrical_solver import quadrupole_dispersion_layered

    reference = globals()[table]
    freq = np.array([f for f, _ in reference])
    a, layers = _s88_cased(thickness)
    res = quadrupole_dispersion_layered(
        freq, **_S88_FAST, **_S88_BOREHOLE, a=a, layers=layers
    )
    assert np.isfinite(res.slowness).all(), (
        f"{label}: {int((~np.isfinite(res.slowness)).sum())} frequencies empty"
    )
    median, rms, worst = _s88_score(reference, freq, res.slowness)
    assert median < median_max, f"{label}: median {100 * median:.2f} %"
    assert worst < worst_max, f"{label}: worst {100 * worst:.2f} %"


def test_the_cased_curves_order_by_cement_thickness_as_the_paper_says():
    """A shape check the tolerances above cannot make.

    The paper's own reading of figure 20 (p. 230) is that the useful
    energy shifts to higher frequency as the borehole radius falls, so
    thicker cement holds the mode near the shear speed longer. Both the
    published curves and fwap must show that ordering, and a scoring
    tolerance would not notice if both drifted together.
    """
    freq = np.arange(7000.0, 12001.0, 500.0)
    velocities = []
    for thickness in (0.01, 0.03):
        a, layers = _s88_cased(thickness)
        res = flexural_dispersion_layered(
            freq, **_S88_FAST, **_S88_BOREHOLE, a=a, layers=layers
        )
        assert np.isfinite(res.slowness).all()
        velocities.append(1.0 / res.slowness)
    thin, thick = velocities
    assert np.all(thick > thin), "thicker cement must hold the mode faster"
    # And the published tables say the same, independently of fwap.
    for f_hz in (8000.0, 10000.0, 12000.0):
        ref_thin = np.interp(f_hz, *zip(*_FIG20A_CEMENT_1CM))
        ref_thick = np.interp(f_hz, *zip(*_FIG20A_CEMENT_3CM))
        assert ref_thick > ref_thin, f"published curves disagree at {f_hz} Hz"


# =====================================================================
# Cased-hole trapped pseudo-Rayleigh (trapped_pseudo_rayleigh_dispersion_layered)
# =====================================================================
#
# The n=0 pseudo-Rayleigh counterpart of stoneley_dispersion_layered.
# Needed a complex n=0 cased determinant: the real one refuses both for
# F_f^2 <= 0 (every phase velocity above V_f) and for any layer with
# s^2 <= 0 (the cement, in a realistic geometry).


def _tubman_cased_kwargs() -> dict:
    """Tubman, Cheng & Toksoz (1984) fig 4(b) geometry, table 1."""
    from fwap.cylindrical_solver import BoreholeLayer

    ft, inch = 304.8, 0.0254
    return dict(
        vp=16.0 * ft,
        vs=8.53 * ft,
        rho=2160.0,
        vf=5.5 * ft,
        rho_f=1200.0,
        a=1.85 * inch,
        layers=(
            BoreholeLayer(vp=20.0 * ft, vs=11.0 * ft, rho=7500.0, thickness=0.4 * inch),
            BoreholeLayer(
                vp=9.26 * ft, vs=5.67 * ft, rho=1920.0, thickness=1.75 * inch
            ),
        ),
    )


def test_modal_determinant_n0_cased_complex_matches_real_below_vf():
    """Where both are defined -- phase velocity below ``V_f``, all media
    evanescent -- the complex cased determinant reproduces the real one
    and has a vanishing imaginary part."""
    from fwap.cylindrical_solver._cased import (
        _modal_determinant_n0_cased,
        _modal_determinant_n0_cased_complex,
    )

    kw = _tubman_cased_kwargs()
    omega = 2.0 * np.pi * 20000.0
    for velocity in (1500.0, 1600.0):
        kz = omega / velocity
        real = _modal_determinant_n0_cased(kz, omega, **kw)
        comp = _modal_determinant_n0_cased_complex(kz, omega, **kw)
        assert np.isfinite(real)
        np.testing.assert_allclose(comp.real, real, rtol=1e-12)
        assert abs(comp.imag) <= 1e-12 * abs(comp.real)


def test_modal_determinant_n0_cased_complex_finite_where_real_is_nan():
    """The whole point of the complex variant: it is finite across the
    trapped window, where the real one returns NaN.

    Both refusals of the real determinant fire here -- the fluid is
    oscillatory (c > V_f = 1676) and so is the cement (V_S = 1728) -- yet
    the formation stays evanescent, so the mode is bound."""
    from fwap.cylindrical_solver._cased import (
        _modal_determinant_n0_cased,
        _modal_determinant_n0_cased_complex,
    )

    kw = _tubman_cased_kwargs()
    omega = 2.0 * np.pi * 20000.0
    cement_vs = kw["layers"][1].vs
    for velocity in (1900.0, 2200.0, 2450.0):
        assert kw["vf"] < velocity < kw["vs"], "probe must be in the trapped window"
        assert velocity > cement_vs, "probe must have the cement oscillatory"
        kz = omega / velocity
        real = _modal_determinant_n0_cased(kz, omega, **kw)
        assert np.isnan(real)
        assert np.isfinite(_modal_determinant_n0_cased_complex(kz, omega, **kw))


def test_trapped_pseudo_rayleigh_layered_empty_layers_matches_open_hole():
    """``layers=()`` must be the open-hole function, not an approximation
    of it."""
    vp, vs, rho, vf, rho_f, a = 4000.0, 2300.0, 2500.0, 1500.0, 1000.0, 0.10
    f = np.linspace(12000.0, 26000.0, 15)
    got = trapped_pseudo_rayleigh_dispersion_layered(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a
    )
    want = trapped_pseudo_rayleigh_dispersion(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a
    )
    np.testing.assert_array_equal(got.slowness, want.slowness)
    assert got.name == want.name == "trapped_pseudo_rayleigh"
    assert got.azimuthal_order == 0


def test_trapped_pseudo_rayleigh_layered_roots_are_bound():
    """Every returned root must sit in the trapped window ``V_f < c <
    V_S`` on the **formation** -- the half-space is what sets
    boundedness. The cement may, and here does, oscillate."""
    kw = _tubman_cased_kwargs()
    f = np.linspace(15000.0, 24500.0, 12)
    res = trapped_pseudo_rayleigh_dispersion_layered(f, **kw, branch=0)
    ok = np.isfinite(res.slowness)
    assert ok.sum() >= 8
    c = 1.0 / res.slowness[ok]
    assert np.all(c > kw["vf"]), "root slower than the fluid is a Stoneley mode"
    assert np.all(c < kw["vs"]), "root faster than formation V_S is not bound"
    # The regime that motivated the complex determinant.
    assert np.any(c > kw["layers"][1].vs), "expected cement-oscillatory roots"


def test_trapped_pseudo_rayleigh_layered_branch_ordering():
    """Branch 1 is faster than branch 0 wherever both exist -- roots are
    ordered by descending ``k_z``, matching the open-hole convention."""
    kw = _tubman_cased_kwargs()
    f = np.linspace(19000.0, 24000.0, 8)
    b0 = trapped_pseudo_rayleigh_dispersion_layered(f, **kw, branch=0)
    b1 = trapped_pseudo_rayleigh_dispersion_layered(f, **kw, branch=1)
    both = np.isfinite(b0.slowness) & np.isfinite(b1.slowness)
    assert both.sum() >= 4
    assert np.all(b1.slowness[both] < b0.slowness[both])


def test_trapped_pseudo_rayleigh_layered_validation():
    """Input validation mirrors the open-hole original."""
    kw = _tubman_cased_kwargs()
    f = np.array([20000.0])
    with pytest.raises(ValueError, match="fast formation"):
        trapped_pseudo_rayleigh_dispersion_layered(f, **{**kw, "vs": 1000.0})
    with pytest.raises(ValueError, match="branch"):
        trapped_pseudo_rayleigh_dispersion_layered(f, **kw, branch=-1)
    with pytest.raises(ValueError, match="strictly positive"):
        trapped_pseudo_rayleigh_dispersion_layered(np.array([-1.0]), **kw)


# =====================================================================
# Rigid centralised logging tool (tool_radius / r_tool)
# =====================================================================
#
# White & Zechman (1968) model, as used by Paillet & Cheng (1986): the
# tool is immovable, so u_r = 0 at its surface and the fluid becomes an
# annulus. That admits K_0 alongside I_0 and changes exactly one column
# of the n=0 determinant. Every solid row is untouched.


def test_rigid_tool_fluid_factors_no_tool_is_bit_identical():
    """``r_tool <= 0`` must short-circuit to the plain Bessel pair, so
    every open-hole result predating this parameter is unchanged to the
    last bit -- not merely to a tolerance."""
    from scipy import special

    from fwap.cylindrical_solver._bessel import _rigid_tool_fluid_factors

    for F, a in ((3.7, 0.10), (0.4, 0.05), (28.0, 0.12)):
        z0, z1 = _rigid_tool_fluid_factors(F, a, 0.0)
        assert z0 == float(special.iv(0, F * a))
        assert z1 == float(special.iv(1, F * a))


def test_rigid_tool_fluid_factors_enforce_the_rigid_boundary():
    """The whole content of the model: radial displacement vanishes at
    the tool surface. ``u_r(r) ∝ I_1(F r) - beta K_1(F r)``, so the
    check is that this is zero at ``r = r_tool``."""
    from scipy import special

    F, r_tool = 3.7, 0.04
    beta = special.iv(1, F * r_tool) / special.kv(1, F * r_tool)
    u_r = special.iv(1, F * r_tool) - beta * special.kv(1, F * r_tool)
    assert abs(u_r) < 1e-15 * max(1.0, abs(special.iv(1, F * r_tool)))


def test_rigid_tool_fluid_factors_converge_quadratically():
    """As the tool shrinks the factors approach the open-hole pair, and
    at the rate the algebra predicts: ``beta ~ (F r_tool)^2 / 2``, so
    shrinking the radius tenfold cuts the deviation a hundredfold."""
    from fwap.cylindrical_solver._bessel import _rigid_tool_fluid_factors

    F, a = 3.7, 0.10
    z0_open, _ = _rigid_tool_fluid_factors(F, a, 0.0)
    devs = []
    for r_tool in (1.0e-3, 1.0e-4):
        z0, _ = _rigid_tool_fluid_factors(F, a, r_tool)
        devs.append(abs(z0 - z0_open))
    ratio = devs[0] / devs[1]
    assert 50.0 < ratio < 200.0, f"expected ~100x, got {ratio:.1f}"


def test_rigid_tool_fluid_factors_rejects_tool_larger_than_hole():
    from fwap.cylindrical_solver._bessel import _rigid_tool_fluid_factors

    with pytest.raises(ValueError, match="smaller than the borehole radius"):
        _rigid_tool_fluid_factors(3.7, 0.10, 0.10)


def test_tube_wave_speed_tool_matches_cylindrical_solver():
    """**Analytic oracle for the tool geometry.**

    ``tube_wave_speed`` with a tool is a quasi-static area-and-compliance
    argument with no Bessel functions in it at all; the ``f -> 0`` limit
    of ``stoneley_dispersion`` is a cylindrical modal determinant built
    entirely from them. They agree to 1e-7 across tool radii out to 60 %
    of the borehole, which is a cross-check between two independent
    calculations rather than the solver confirming itself.
    """
    kw = dict(vp=4000.0, vs=2300.0, rho=2300.0, vf=1500.0, rho_f=1000.0, a=0.10)
    for r_tool in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06):
        numeric = (
            1.0
            / stoneley_dispersion(np.array([1.0]), **kw, tool_radius=r_tool).slowness[0]
        )
        closed = tube_wave_speed(
            kw["vs"],
            kw["rho"],
            vf=kw["vf"],
            rho_f=kw["rho_f"],
            a=kw["a"],
            tool_radius=r_tool,
        )
        np.testing.assert_allclose(numeric, closed, rtol=1e-7)


def test_tube_wave_speed_tool_slows_monotonically_and_collapses_at_zero():
    """A tool always slows the tube wave, and ``tool_radius=0`` returns
    exactly the open-hole value."""
    args = dict(vf=1500.0, rho_f=1000.0, a=0.10)
    open_hole = tube_wave_speed(2300.0, 2300.0, vf=1500.0, rho_f=1000.0)
    assert tube_wave_speed(2300.0, 2300.0, **args, tool_radius=0.0) == open_hole
    speeds = [
        tube_wave_speed(2300.0, 2300.0, **args, tool_radius=r)
        for r in (0.0, 0.02, 0.04, 0.06, 0.08)
    ]
    assert all(b < x for x, b in zip(speeds, speeds[1:]))


def test_tube_wave_speed_tool_validation():
    args = dict(vf=1500.0, rho_f=1000.0)
    with pytest.raises(ValueError, match="a .borehole radius. is required"):
        tube_wave_speed(2300.0, 2300.0, **args, tool_radius=0.05)
    with pytest.raises(ValueError, match="smaller than the borehole radius"):
        tube_wave_speed(2300.0, 2300.0, **args, a=0.05, tool_radius=0.05)
    with pytest.raises(ValueError, match="non-negative"):
        tube_wave_speed(2300.0, 2300.0, **args, a=0.10, tool_radius=-0.01)


def test_dispersion_tool_radius_zero_is_bit_identical():
    """The default must not perturb any existing curve."""
    kw = dict(vp=4000.0, vs=2300.0, rho=2500.0, vf=1500.0, rho_f=1000.0, a=0.10)
    f = np.linspace(2000.0, 26000.0, 13)
    np.testing.assert_array_equal(
        stoneley_dispersion(f, **kw, tool_radius=0.0).slowness,
        stoneley_dispersion(f, **kw).slowness,
    )
    np.testing.assert_array_equal(
        trapped_pseudo_rayleigh_dispersion(f, **kw, tool_radius=0.0).slowness,
        trapped_pseudo_rayleigh_dispersion(f, **kw).slowness,
    )


def test_stoneley_dispersion_tool_slows_the_mode_at_every_frequency():
    """Physically the tool can only slow the Stoneley wave -- it is the
    reason tool corrections appear in Stoneley permeability workflows."""
    kw = dict(vp=4000.0, vs=2300.0, rho=2500.0, vf=1500.0, rho_f=1000.0, a=0.10)
    f = np.linspace(1000.0, 20000.0, 20)
    s_open = stoneley_dispersion(f, **kw).slowness
    s_tool = stoneley_dispersion(f, **kw, tool_radius=0.05).slowness
    ok = np.isfinite(s_open) & np.isfinite(s_tool)
    assert ok.sum() >= 15
    assert np.all(s_tool[ok] > s_open[ok])


# ======================================================================
# Schmitt & Cheng (1987) figs 20 + 21 -- the cased hole gets a dipole
# and a quadrupole
# ======================================================================
#
# ``flexural_dispersion_layered`` and ``quadrupole_dispersion_layered``
# had no external tie of any kind before these. The cased *monopole*
# modes are tied by Tubman fig 4(b); the cased n=1 and n=2 ones rested
# entirely on internal consistency checks -- N=1 agreeing with the
# single-interface determinant, two half-thickness layers agreeing with
# one, layer order mattering. All true, none of them evidence that the
# curve is the published one.
#
# The reference is the same MIT ERL report, and the same table 1, that
# already supplies the open-hole flexural ties (figs 2(a) and 8(a)):
#
#   Schmitt, D. P., & Cheng, C. H. (1987), *Shear wave logging in
#   (multilayered) elastic formations: an overview*, MIT Earth Resources
#   Laboratory, 213-268.
#
# Fig 20 is the dipole (flexural) mode and fig 21 the quadrupole (which
# the report calls the *screw* mode), each in a well-bonded cased hole
# around the fast sandstone, each with a cement-thickness panel (a) and
# a cement-stiffness panel (b).
#
# The geometry needs one sentence of the report to read correctly: "the
# inner borehole radius is decreased by the amount of the casing and
# cement thickness ... The original borehole radius is 10 cm". So the
# annulus eats inward and the formation contact stays at 0.10 m -- the
# same convention Tubman's table 1 fixes from a different direction.

_SC87_FAST = dict(vp=4878.0, vs=2601.0, rho=2160.0, vf=1500.0, rho_f=1000.0)


#: ``csv stem -> (cement V_S, cement rho, cement thickness)``. Cement 1
#: is the report's reference cement and cement 2 its softer variant.
_SC87_CASED_CASES = {
    "fig20a_flexural_cased_cement1_1cm": (1729.0, 1920.0, 0.01),
    "fig20a_flexural_cased_cement1_3cm": (1729.0, 1920.0, 0.03),
    "fig20b_flexural_cased_cement2_3cm": (1555.0, 1730.0, 0.03),
    "fig21a_screw_cased_cement1_1cm": (1729.0, 1920.0, 0.01),
    "fig21b_screw_cased_cement1_3cm": (1729.0, 1920.0, 0.03),
    "fig21b_screw_cased_cement2_3cm": (1555.0, 1730.0, 0.03),
}


def _sc87_reference(stem: str) -> np.ndarray:
    data = Path(__file__).resolve().parents[1] / "docs" / "notebooks" / "_data"
    return np.loadtxt(data / f"schmitt_cheng_1987_{stem}.csv", delimiter=",")


def _sc87_score(stem: str, freq: np.ndarray, slowness: np.ndarray):
    """RMS and worst relative deviation of fwap from a traced curve."""
    reference = _sc87_reference(stem)
    live = np.isfinite(slowness)
    inside = (reference[:, 0] >= freq[live].min()) & (
        reference[:, 0] <= freq[live].max()
    )
    ours = np.interp(reference[inside, 0], freq[live], slowness[live])
    residual = (ours - reference[inside, 1]) / reference[inside, 1]
    return (
        float(np.sqrt(np.mean(residual**2))),
        float(np.max(np.abs(residual))),
        int(inside.sum()),
        int(reference.shape[0]),
    )


def test_cased_flexural_matches_schmitt_cheng_fig20():
    """The cased dipole's first external tie, three cement stacks.

    Fig 20 varies the cement two ways: (a) thickness, 1 cm against 3 cm
    of the reference cement, and (b) stiffness, cement 2 (V_S 1555)
    against cement 1 (V_S 1729) at a fixed 3 cm. Both effects are in the
    scored band, so this is not one curve checked three times -- a
    solver that got the annulus propagator right but the radius
    convention wrong would match one thickness and miss the other.

    **What the 0.5 % is.** Cement 1 at 3 cm appears in *both* panels and
    was therefore traced twice from independent artwork; the two
    renderings agree to 0.23 % RMS. That is the floor a 1987 raster scan
    can support, so most of the residual here is the scan.
    """
    freq = np.linspace(2200.0, 15000.0, 257)
    for stem in [s for s in _SC87_CASED_CASES if s.startswith("fig20")]:
        radius, layers = _sc87_stack(*_SC87_CASED_CASES[stem])
        mode = flexural_dispersion_layered(freq, **_SC87_FAST, a=radius, layers=layers)
        rms, worst, scored, total = _sc87_score(stem, freq, mode.slowness)
        assert scored >= 80, f"{stem}: only {scored} of {total} points scored"
        assert rms < 0.01, f"{stem}: RMS {100 * rms:.2f}%"
        assert worst < 0.02, f"{stem}: worst point {100 * worst:.2f}%"


def test_cased_screw_matches_schmitt_cheng_fig21():
    """The same for the quadrupole -- fig 21, "same as Figure 20 for the
    screw mode".

    Its band is 6-20 kHz rather than 2-15: the screw mode's useful
    energy sits higher than the flexural one's, which is the report's
    own reason for saying the cement effects "occur closer to its useful
    energy due to the higher frequencies involved".
    """
    freq = np.linspace(6300.0, 20000.0, 275)
    from fwap.cylindrical_solver import quadrupole_dispersion_layered

    for stem in [s for s in _SC87_CASED_CASES if s.startswith("fig21")]:
        radius, layers = _sc87_stack(*_SC87_CASED_CASES[stem])
        mode = quadrupole_dispersion_layered(
            freq, **_SC87_FAST, a=radius, layers=layers
        )
        rms, worst, scored, total = _sc87_score(stem, freq, mode.slowness)
        assert scored >= 85, f"{stem}: only {scored} of {total} points scored"
        assert rms < 0.01, f"{stem}: RMS {100 * rms:.2f}%"
        assert worst < 0.02, f"{stem}: worst point {100 * worst:.2f}%"


def test_the_cased_traces_low_frequency_ends_land_on_the_formation_shear():
    """A calibration check on the tracing that owes nothing to fwap.

    Every curve in figs 20 and 21 starts at the formation shear speed,
    and the tracing was calibrated on the panel frames and tick marks
    only. So the fastest point of each traced curve reproducing table
    1's V_S = 2601 m/s is an independent statement about the
    digitising -- if the y-axis calibration were off by a percent, this
    would say so before any solver is involved.
    """
    for stem in _SC87_CASED_CASES:
        fastest = 1.0 / _sc87_reference(stem)[:, 1].min()
        error = abs(fastest - _SC87_FAST["vs"]) / _SC87_FAST["vs"]
        assert error < 0.005, f"{stem}: {fastest:.0f} m/s against V_S = 2601"


# ----------------------------------------------------------------------
# Schmitt & Cheng p. 231 -- the cased *leaky* dipole of a slow formation
# ----------------------------------------------------------------------
#
# The cased leaky branch (roadmap A.9, ``_march_leaky_cased_branch``) is
# the one cased mode with no published *curve* anywhere reachable. What
# it does have is a published *claim*, in the same report, under "Slow
# formation" on p. 231:
#
#   "The high frequency part of the fundamental modes excited either by
#   a dipole or a quadrupole source will then also be leaky. ... it is
#   only with a low source center frequency that the multipole tool will
#   be capable of logging the slow formation shear wave velocity behind
#   casing." ... "The flexural mode is seen to travel with a velocity
#   higher than that of the formation shear wave."
#
# Schmitt & Cheng illustrate that case with waveforms (figs 24 and 25),
# not a dispersion curve, so there is nothing to score. These two tests
# carry the claim instead, at their table 1 parameters.


def _sc87_slow_cased_determinant(kz: complex, omega: float) -> complex:
    from fwap.cylindrical_solver import _modal_determinant_n1_cased_complex
    from fwap.cylindrical_solver._leaky import _detect_leaky_branches

    radius, layers = _sc87_stack(1729.0, 1920.0, 0.03)
    _, leaky_p, leaky_s = _detect_leaky_branches(
        kz, omega, _SC87_SLOW["vp"], _SC87_SLOW["vs"], _SC87_SLOW["vf"]
    )
    return _modal_determinant_n1_cased_complex(
        kz,
        omega,
        **_SC87_SLOW,
        a=radius,
        layers=layers,
        leaky_p=leaky_p,
        leaky_s=leaky_s,
    )


def test_the_cased_dipole_of_a_slow_formation_has_no_bound_root_at_all():
    """Schmitt & Cheng's claim, in its falsifiable direction.

    A bound flexural mode is slower than the formation shear speed, so
    "the flexural mode travels with a velocity higher than that of the
    formation shear wave" is the statement that no bound root exists.
    Scanning the real cased determinant over the whole bound window
    finds no sign change at any frequency from 0.5 to 14 kHz -- and the
    scan reaches to within 1e-9 of ``V_S``, so this is not a resolution
    artefact near the branch point.

    The report's own hedge is preserved: it says the *high frequency*
    part is leaky and that a low-frequency source can still log the
    formation shear. Both are consistent with this -- the low-frequency
    limit approaches ``V_S`` from above rather than crossing it, so a
    low-frequency measurement still reads the formation shear speed even
    though the root never becomes bound.
    """
    radius, layers = _sc87_stack(1729.0, 1920.0, 0.03)
    grid = np.concatenate(
        [
            np.linspace(300.0, 1200.0, 140),
            _SC87_SLOW["vs"] * (1.0 - np.logspace(-1.0, -9.0, 40)),
        ]
    )
    for freq in (500.0, 1000.0, 2000.0, 4000.0, 8000.0, 14000.0):
        omega = 2.0 * np.pi * freq
        values = np.array(
            [
                _modal_determinant_n1_cased(
                    omega / c, omega, **_SC87_SLOW, a=radius, layers=layers
                )
                for c in grid
            ],
            dtype=float,
        )
        live = np.isfinite(values)
        assert live.sum() > 100, f"{freq} Hz: determinant mostly NaN"
        signs = np.sign(values[live])
        crossings = int((signs[:-1] * signs[1:] < 0).sum())
        assert crossings == 0, f"{freq} Hz: {crossings} bound roots below V_S"


def test_the_cased_leaky_dipole_can_outrun_the_borehole_fluid():
    """What is left of the gap Schmitt & Cheng's slow geometry exposes.

    ``_march_leaky_cased_branch`` searches ``(V_S, min(V_f, min layer
    V_S))``. For the ``_A2`` stack that ceiling is the cement's 1300 m/s
    and the branch sits under it, which is why that test passes. For
    Schmitt & Cheng's slow sandstone behind 1.02 cm of steel and 3 cm of
    cement 1 the ceiling is the *fluid*, 1500 m/s, and the branch is
    above it across the middle of the band: it leaves ``V_S`` near
    1.4 kHz, climbs to about 1710 m/s at 5.5 kHz -- just under the
    cement's 1729 -- and comes back down through ``V_f`` near 13.8 kHz.

    So the mode is resolved near the top of the band, at about 1487 m/s.
    Exactly where that starts is grid-dependent -- 14 kHz on a 500 Hz
    ladder, 13.5 kHz on a sparse one -- which is why the probe below
    stops at 12 kHz rather than pretending the edge is sharp.

    **This test used to say the whole band was NaN, and promised to fail
    the day either half of the gap was fixed. One half now is.** The gap
    always had two causes: above 3 kHz the root is outside the search
    window and the marcher is right not to find it, but at 1.5-2.5 kHz
    it is *inside* the window and was missed anyway -- seeding rather
    than the ceiling. Rebuilding the seeding recovers that leg in full:
    1235.9, 1300.0, 1358.3, 1412.0 and 1461.2 m/s at 1.50 to 2.50 kHz,
    which an independent hand-seeded minimisation of ``log|det|``
    reproduces to 0.003 %.

    **The recovered leg is exactly the window's contents, no more.** An
    argument-principle contour counts one root at each of 1.50, 1.75,
    2.00, 2.25 and 2.50 kHz and **none** at 1.00, 1.25, 2.75 or 3.00 --
    the same five frequencies the marcher returns and the same four it
    declines. Nothing was left behind at the edges and nothing spurious
    was added.

    What remains is the ceiling half, and it is what this test now pins:
    between roughly 3 and 13 kHz the branch is above ``V_f``, outside
    the window the marcher searches at all, and no amount of seeding
    reaches it.
    """
    radius, layers = _sc87_stack(1729.0, 1920.0, 0.03)
    # The leg the seeding rebuild recovered.
    low = np.array([1500.0, 1750.0, 2000.0, 2250.0, 2500.0])
    got = flexural_dispersion_layered(low, **_SC87_SLOW, a=radius, layers=layers)
    assert np.isfinite(got.slowness).all(), 1.0 / got.slowness
    np.testing.assert_allclose(
        1.0 / got.slowness, [1235.9, 1300.0, 1358.3, 1412.0, 1461.2], rtol=1.0e-3
    )
    assert np.all(got.attenuation_per_meter > 0.0)
    # It is a leg of this window, so it stays inside it.
    assert np.all(1.0 / got.slowness > _SC87_SLOW["vs"])
    assert np.all(1.0 / got.slowness < _SC87_SLOW["vf"])

    # The ceiling half is untouched: above the leg the branch outruns
    # V_f and the marcher cannot look there.
    freq = np.array([3000.0, 5500.0, 8000.0, 10000.0, 12000.0])
    mode = flexural_dispersion_layered(freq, **_SC87_SLOW, a=radius, layers=layers)
    assert not np.isfinite(mode.slowness).any(), 1.0 / mode.slowness
    # ... and it does resolve once the branch has slowed below V_f.
    above = flexural_dispersion_layered(
        np.array([14000.0, 15000.0]), **_SC87_SLOW, a=radius, layers=layers
    )
    assert np.isfinite(above.slowness).all()
    assert np.all(1.0 / above.slowness < _SC87_SLOW["vf"])
    assert np.all(1.0 / above.slowness > _SC87_SLOW["vs"])

    # Where the marcher stops looking, and where the root actually is.
    ceiling = min(_SC87_SLOW["vf"], min(layer.vs for layer in layers))
    assert ceiling == _SC87_SLOW["vf"]
    for freq_hz, velocity in ((3000.0, 1545.1), (5500.0, 1710.5), (8000.0, 1646.7)):
        assert velocity > ceiling
        omega = 2.0 * np.pi * freq_hz
        centre = omega / velocity
        n = 60
        contour = (
            [complex(x, 0.05) for x in np.linspace(0.9 * centre, 1.1 * centre, n)]
            + [complex(1.1 * centre, y) for y in np.linspace(0.05, 6.0, n)]
            + [complex(x, 6.0) for x in np.linspace(1.1 * centre, 0.9 * centre, n)]
            + [complex(0.9 * centre, y) for y in np.linspace(6.0, 0.05, n)]
        )
        phase = np.unwrap(
            [np.angle(_sc87_slow_cased_determinant(p, omega)) for p in contour]
        )
        winding = (phase[-1] - phase[0]) / (2.0 * np.pi)
        assert abs(winding - 1.0) < 0.05, f"{freq_hz} Hz: winding {winding:.3f}"

    # The window's contents at the low end, counted rather than assumed:
    # one root exactly where the recovered leg is and none beside it.
    for freq_hz, expected in (
        (1000.0, 0),
        (1250.0, 0),
        (1500.0, 1),
        (1750.0, 1),
        (2000.0, 1),
        (2250.0, 1),
        (2500.0, 1),
        (2750.0, 0),
    ):
        omega = 2.0 * np.pi * freq_hz
        lo, hi = omega / (ceiling - 0.5), omega / (_SC87_SLOW["vs"] + 0.5)
        n = 80
        contour = (
            [complex(x, 0.02) for x in np.linspace(lo, hi, n)]
            + [complex(hi, y) for y in np.linspace(0.02, 8.0, n)]
            + [complex(x, 8.0) for x in np.linspace(hi, lo, n)]
            + [complex(lo, y) for y in np.linspace(8.0, 0.02, n)]
        )
        phase = np.unwrap(
            [np.angle(_sc87_slow_cased_determinant(p, omega)) for p in contour]
        )
        winding = (phase[-1] - phase[0]) / (2.0 * np.pi)
        assert abs(winding - expected) < 0.05, (
            f"{freq_hz} Hz: {winding:.3f} roots in (V_S, V_f), expected {expected}"
        )


# ======================================================================
# Yang et al. (2022) fig 2 -- the cased dipole, tied a second time and
# in a slow formation
# ======================================================================
#
#   Yang Meng-En, Lv Wei-Guo, Wu Yang, Cui Zhi-Wen & Liu Jin-Xia (2022).
#   Numerical study of dispersion characteristics of dipole flexural
#   waves in a cased hole with different cement conditions.
#   *Applied Geophysics* 19(1), 29-40. doi:10.1007/s11770-022-0923-9
#
# Different group, different decade, vector artwork rather than a raster
# scan, and -- crucially -- the same object this solver computes: the
# paper states ``D_1(k_z, omega) = 0`` for the dipole flexural mode and
# ``v = omega / k_z``, so these are modal roots, not semblance picks off
# synthetic waveforms.
#
# Its table 1 gives ``r_n`` as *outer* radii directly, so unlike
# Schmitt & Cheng nothing has to be subtracted: the fluid radius is
# 0.0635 and the layers stack outward to the 0.1016 formation contact.
#
# Figure 2 sweeps eight formations from V_S = 3000 down to 1450 m/s, but
# table 1 gives V_P and density for exactly two of them, so exactly two
# are scored. The soft one is the point: V_S = 1450 < V_f = 1500 puts a
# slow formation behind casing, which nothing else in the repository
# covers with a published curve.

_YANG_HARD = dict(vp=5081.0, vs=3000.0, rho=2490.0, vf=1500.0, rho_f=1000.0)
_YANG_SOFT = dict(vp=2200.0, vs=1450.0, rho=2160.0, vf=1500.0, rho_f=1000.0)
_YANG_A = 0.0635
_YANG_STACK = (
    BoreholeLayer(vp=6098.0, vs=3354.0, rho=7500.0, thickness=0.0715 - 0.0635),
    BoreholeLayer(vp=3000.0, vs=1776.0, rho=1900.0, thickness=0.1016 - 0.0715),
)


def _yang_reference(stem: str) -> np.ndarray:
    data = Path(__file__).resolve().parents[1] / "docs" / "notebooks" / "_data"
    return np.loadtxt(data / f"yang_lv_2022_{stem}.csv", delimiter=",")


def test_cased_flexural_matches_yang_2022_fig2a_hard():
    """The fast-formation panel, as a control on the reading of table 1.

    Its flat low-frequency top is table 1's V_S to 0.007 % with nothing
    fitted -- the calibration came from the panel frame and was checked
    against the figure's own gridlines -- so a mistake in the radius
    convention or the layer order would show here before the soft
    panel is reached.
    """
    reference = _yang_reference("fig2a_flexural_cased_hard")
    freq = np.linspace(3500.0, 20000.0, 331)
    mode = flexural_dispersion_layered(
        freq, **_YANG_HARD, a=_YANG_A, layers=_YANG_STACK
    )
    live = np.isfinite(mode.slowness)
    inside = (reference[:, 0] >= freq[live].min()) & (
        reference[:, 0] <= freq[live].max()
    )
    assert inside.sum() >= 65, f"only {int(inside.sum())} of {reference.shape[0]}"
    ours = np.interp(reference[inside, 0], freq[live], mode.slowness[live])
    residual = (ours - reference[inside, 1]) / reference[inside, 1]
    assert float(np.sqrt(np.mean(residual**2))) < 0.008
    # The reference's own fastest point is V_S, which is a statement
    # about the extraction rather than about fwap.
    assert abs(1.0 / reference[:, 1].min() - _YANG_HARD["vs"]) < 1.0


def test_cased_flexural_matches_yang_2022_fig2b_slow_formation():
    """A slow formation behind casing, tied to a published curve.

    Section 4b could only cite Schmitt & Cheng's prose for this
    configuration. Here it is plotted, and every one of the twelve
    traced points is matched.

    **The published branch is bound, not leaky.** All twelve sit below
    ``V_S / V_f`` = 0.9667, which is why the bound cased determinant
    finds them; the assertion below pins that, so a future change that
    started answering these frequencies from the complex marcher would
    fail here rather than pass quietly on a different root.
    """
    reference = _yang_reference("fig2b_flexural_cased_soft")
    assert reference.shape[0] == 12
    # Bound: every reference point is slower than the formation shear.
    assert np.all(1.0 / reference[:, 1] < _YANG_SOFT["vs"])

    freq = np.linspace(12000.0, 20000.0, 321)
    mode = flexural_dispersion_layered(
        freq, **_YANG_SOFT, a=_YANG_A, layers=_YANG_STACK
    )
    live = np.isfinite(mode.slowness)
    ours = np.interp(reference[:, 0], freq[live], mode.slowness[live])
    residual = (ours - reference[:, 1]) / reference[:, 1]
    assert float(np.sqrt(np.mean(residual**2))) < 0.001
    assert float(np.max(np.abs(residual))) < 0.002

    # And fwap answers them from the bound path, not the leaky fill.
    bound = ~np.isfinite(mode.attenuation_per_meter) & live
    assert np.all(np.isin(np.searchsorted(freq, reference[:, 0]), np.where(bound)[0]))


def test_fwap_continues_the_yang_branch_below_its_published_cutoff():
    """Where the published curve stops, and what fwap does instead.

    Yang et al. give this mode's cutoff as 15.04 kHz and plot nothing
    below it; the first traced dot sits at 15.13. `fwap` continues the
    same branch downward as a **leaky** root -- above ``V_S`` and
    carrying a positive radiation attenuation -- for another ~2.7 kHz
    before it too runs out.

    That continuation has no published curve behind it, here or
    anywhere else found, so this test asserts its character rather than
    its values: leaky where it is filled, and faster than the formation
    shear speed throughout. It is the same branch section 4b's
    Schmitt & Cheng entry is about, seen from the other side of a
    cutoff that *is* tied.
    """
    freq = np.linspace(12000.0, 20000.0, 321)
    mode = flexural_dispersion_layered(
        freq, **_YANG_SOFT, a=_YANG_A, layers=_YANG_STACK
    )
    leaky = np.isfinite(mode.attenuation_per_meter)
    assert leaky.sum() > 50
    assert freq[leaky].max() < 15040.0, "the leaky fill runs past the published cutoff"
    velocity = 1.0 / mode.slowness[leaky]
    assert np.all(velocity > _YANG_SOFT["vs"])
    assert np.all(velocity < _YANG_SOFT["vf"])
    assert np.all(mode.attenuation_per_meter[leaky] > 0.0)


def test_the_ceiling_is_guarded_by_the_degeneracy_name_not_by_a_dead_band():
    """Which guard protects which ceiling, stated structurally.

    ``_LEAKY_CASED_DEGENERACY_TOL`` used to be applied twice: once to
    reject a candidate coinciding with a named ``exclude`` velocity, and
    again as the width of a dead band held off the window ceiling. Only
    the first is what its recorded reasoning asks for.

    The reasoning is that the ceiling *is* a layer shear speed whenever
    the softest layer is slower than the fluid, so a root pinned at it
    is that layer's vanishing radial wavenumber rather than a mode. When
    that is true the ceiling is in ``exclude`` and ``_degenerate``
    rejects it. When it is not true -- ``ceiling = V_f``, which is
    Schmitt & Cheng's cased geometry, where the cement is *faster* than
    the fluid -- there is no degeneracy at the ceiling at all, and the
    dead band was the only thing acting.

    This asserts that split from the layer stacks themselves, without
    running the solver.
    """
    from fwap.cylindrical_solver._leaky import _LEAKY_CASED_DEGENERACY_TOL

    def ceiling_is_named(vf, layers):
        exclude = tuple(layer.vs for layer in layers)
        ceiling = min(vf, min(exclude))
        named = any(
            abs(ceiling / e - 1.0) < _LEAKY_CASED_DEGENERACY_TOL for e in exclude
        )
        return ceiling, named

    # Ceiling is the cement shear speed -> named, so guarded already.
    ceiling, named = ceiling_is_named(1500.0, (_A2_CASING, _A2_CEMENT))
    assert ceiling == pytest.approx(_A2_CEMENT.vs)
    assert named

    # The annulus-stiffness sweep the withdrawn justification was
    # measured on: also a layer shear speed, also named.
    ceiling, named = ceiling_is_named(1500.0, _gap_sweep_layers(1.3))
    assert ceiling == pytest.approx(1.3 * 800.0)
    assert named

    # Schmitt & Cheng: the fluid is the ceiling and no layer is near it.
    radius, layers = _sc87_stack(1729.0, 1920.0, 0.03)
    ceiling, named = ceiling_is_named(_SC87_SLOW["vf"], layers)
    assert ceiling == pytest.approx(_SC87_SLOW["vf"])
    assert not named


def test_withdrawing_the_ceiling_dead_band_added_a_mode_not_an_edge_artefact():
    """The point it recovers moves like a branch, not like a branch point.

    A root pinned at the ceiling -- the failure the withdrawn dead band
    was written against -- reads ``c / ceiling`` = 1 to four figures at
    every frequency, and carries no leakage. This one does neither: over
    12.90 to 14.00 kHz on Schmitt & Cheng's stack it runs 1498.4 ->
    1486.9 m/s with ``Im(k_z)`` 0.658 -> 0.543, both smooth and monotone,
    and ``c / ceiling`` sweeps 0.9989 -> 0.9912.

    An argument-principle contour counts one root at 13.00 kHz and none
    at 12.75, where the branch has left the window through ``V_f``; that
    count is asserted in
    ``test_the_cased_leaky_dipole_can_outrun_the_borehole_fluid``.
    """
    radius, layers = _sc87_stack(1729.0, 1920.0, 0.03)
    freq = np.arange(12900.0, 14001.0, 50.0)
    mode = flexural_dispersion_layered(freq, **_SC87_SLOW, a=radius, layers=layers)
    assert np.isfinite(mode.slowness).all(), 1.0 / mode.slowness

    velocity = 1.0 / mode.slowness
    leakage = mode.attenuation_per_meter
    # Monotone in both parts -- one branch, not an edge being tracked.
    assert np.all(np.diff(velocity) < 0.0), velocity
    assert np.all(np.diff(leakage) < 0.0), leakage
    assert np.all(leakage > 0.0)
    # And not pinned: the ratio to the ceiling moves by nearly 1 %.
    ratio = velocity / _SC87_SLOW["vf"]
    assert ratio.max() < 1.0
    assert ratio.max() - ratio.min() > 0.007, ratio
    # The top of the recovered stretch really is inside the old dead
    # band, which is why it took withdrawing it to see any of this.
    assert velocity.max() > _SC87_SLOW["vf"] * (1.0 - 2.0e-3)


# ---------------------------------------------------------------------------
# Claro (2020) fig 3.7 -- phase *and* group slowness, six modes
#
# Diego Salam Claro, "Computational analysis of dispersive acoustic waves in
# fluid-filled boreholes", MSc dissertation, Instituto de Fisica Gleb
# Wataghin, UNICAMP, 2020. Figure 3.7 plots Stoneley, flexural and quadrupole
# for one formation and two fluids, 200 Hz to 20 kHz, with the phase slowness
# solid and the **group slowness dashed** -- the reason it is here. Before
# this, the package had exactly one scored group-slowness tie (Sinha fig
# 11(b)); fig 3.7 adds six, and they are a finite-element calculation rather
# than another modal-determinant one, so they are not the same method
# checking itself.
#
# Parameters are the thesis's own Tables 3.1 and 3.2, converted with the
# K = 304800 it states in eq 3.2.1. Fluid 1 is slower than the formation
# shear (fast formation); fluid 2 is faster (slow formation).
_CLARO_K = 304800.0
_CLARO_ROCK = dict(vp=_CLARO_K / 87.0, vs=_CLARO_K / 152.4, rho=2300.0)
_CLARO_FAST = dict(**_CLARO_ROCK, vf=_CLARO_K / 203.0, rho_f=1000.0, a=0.1)
_CLARO_SLOW = dict(**_CLARO_ROCK, vf=_CLARO_K / 138.5, rho_f=1000.0, a=0.1)
_CLARO_STACK = {"fast": _CLARO_FAST, "slow": _CLARO_SLOW}
_CLARO_PANEL = {"fast": "a", "slow": "b"}


def _claro_solver(mode: str):
    from fwap.cylindrical_solver import (
        flexural_dispersion,
        quadrupole_dispersion,
        stoneley_dispersion,
    )

    return {
        "stoneley": stoneley_dispersion,
        "flexural": flexural_dispersion,
        "quadrupole": quadrupole_dispersion,
    }[mode]


def _claro_curves(tag: str, mode: str, n: int = 400):
    """Return ``(freq, phase, group)`` over the live part of the band."""
    freq = np.linspace(200.0, 20000.0, n)
    result = _claro_solver(mode)(freq, **_CLARO_STACK[tag])
    live = np.isfinite(result.slowness)
    omega = 2.0 * np.pi * freq[live]
    phase = result.slowness[live]
    return freq[live], phase, np.gradient(omega * phase, omega)


def _claro_reference(tag: str, mode: str, which: str) -> np.ndarray:
    data = Path(__file__).resolve().parents[1] / "docs" / "notebooks" / "_data"
    return np.loadtxt(
        data / f"claro_2020_fig37{_CLARO_PANEL[tag]}_{mode}_{which}_{tag}.csv",
        delimiter=",",
    )


def _claro_score(tag: str, mode: str, which: str) -> tuple[int, float, float]:
    reference = _claro_reference(tag, mode, which)
    freq, phase, group = _claro_curves(tag, mode)
    curve = phase if which == "phase" else group
    keep = (reference[:, 0] >= freq.min()) & (reference[:, 0] <= freq.max())
    got = np.interp(reference[keep, 0], freq, curve)
    residual = (got - reference[keep, 1]) / reference[keep, 1]
    return (
        int(keep.sum()),
        float(np.sqrt(np.mean(residual**2))),
        float(np.max(np.abs(residual))),
    )


_CLARO_MODES = [
    (tag, mode)
    for tag in ("fast", "slow")
    for mode in ("stoneley", "flexural", "quadrupole")
]


@pytest.mark.parametrize(("tag", "mode"), _CLARO_MODES)
def test_claro_fig37_phase_slowness_matches(tag: str, mode: str) -> None:
    """All six phase curves, which is what the group curves differentiate.

    These ship alongside the group curves rather than instead of them:
    without the phase tie there is no way to tell a wrong group velocity
    from a wrong phase velocity that was differentiated correctly.
    """
    n, rms, worst = _claro_score(tag, mode, "phase")
    assert n > 130, n
    assert rms < 0.004, (tag, mode, rms)
    assert worst < 0.01, (tag, mode, worst)


# Per-curve RMS and worst-point budgets, each set from what was
# measured rather than one loose number covering all six. A single 3 %
# budget would have been worthless on the two Stoneley rows: those group
# curves sit within 1.8 % and 2.2 % of their own *phase* curves, so a
# solver that returned the phase slowness and never differentiated
# anything would have passed. The test therefore scores that substitution
# too and requires it to fail, so each row certifies that it can tell a
# group slowness from a phase slowness at the tolerance it is granted.
_CLARO_GROUP_BUDGET = [
    ("fast", "stoneley", 0.002, 0.005),
    ("fast", "flexural", 0.030, 0.060),
    ("fast", "quadrupole", 0.015, 0.030),
    ("slow", "stoneley", 0.002, 0.008),
    ("slow", "flexural", 0.010, 0.025),
    ("slow", "quadrupole", 0.005, 0.020),
]


@pytest.mark.parametrize(("tag", "mode", "budget", "worst_budget"), _CLARO_GROUP_BUDGET)
def test_claro_fig37_group_slowness_matches(
    tag: str, mode: str, budget: float, worst_budget: float
) -> None:
    """Six group-slowness ties, from ``d(omega * S)/d(omega)``.

    The flexural budget is the loosest of the six for a reason that is
    measured, not assumed: see
    :func:`test_the_fig37_group_budget_is_set_by_the_dashed_curve_not_the_solver`.
    """
    n, rms, worst = _claro_score(tag, mode, "group")
    assert n > 95, n
    assert rms < budget, (tag, mode, rms, budget)
    assert worst < worst_budget, (tag, mode, worst, worst_budget)

    # The budget is tight enough to reject the phase slowness in its
    # place -- otherwise the row would be testing nothing about the
    # derivative at all.
    reference = _claro_reference(tag, mode, "group")
    freq, phase, _ = _claro_curves(tag, mode)
    keep = (reference[:, 0] >= freq.min()) & (reference[:, 0] <= freq.max())
    undifferentiated = np.interp(reference[keep, 0], freq, phase)
    decoy = (undifferentiated - reference[keep, 1]) / reference[keep, 1]
    assert float(np.sqrt(np.mean(decoy**2))) > budget, (tag, mode)


def test_the_fig37_group_budget_is_set_by_the_dashed_curve_not_the_solver() -> None:
    """Why the fast flexural gets 3 % where its phase curve gets 0.4 %.

    Three curves describe the same physics on the Airy limb: the
    figure's dashed group curve, the figure's *own* solid phase curve
    differentiated, and fwap's group curve. Comparing all three orders
    them, which comparing two cannot.

    fwap lands **nearer the figure's own phase data than the figure's
    own dashed rendering does**. So on that limb the dashed curve is the
    least reliable of the three, and the budget it is granted is
    measuring the reading rather than the solver. The dashed curve is
    near-vertical there, where both the dash pattern and one slowness
    per pixel column degrade at once.

    This is the falsifiable form of the claim: if fwap were the curve in
    the wrong, it would sit *further* from the differentiated phase data
    than the dashed curve does, and this test would fail.
    """
    tag, mode = "fast", "flexural"
    phase_ref = _claro_reference(tag, mode, "phase")
    group_ref = _claro_reference(tag, mode, "group")

    # Smooth the pixel-quantised phase trace before differentiating it;
    # without this the quantisation noise dominates the derivative.
    window = 15
    padded = np.pad(phase_ref[:, 1], window // 2, mode="edge")
    smoothed = np.convolve(padded, np.ones(window) / window, "valid")
    omega = 2.0 * np.pi * phase_ref[:, 0]
    implied = np.gradient(omega * smoothed, omega)

    limb = (group_ref[:, 0] >= 3000.0) & (group_ref[:, 0] <= 5000.0)
    assert limb.sum() > 20, limb.sum()
    at = group_ref[limb, 0]
    dashed = group_ref[limb, 1]
    differentiated = np.interp(at, phase_ref[:, 0], implied)
    freq, _, group = _claro_curves(tag, mode)
    ours = np.interp(at, freq, group)

    def rms(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.mean(((a - b) / b) ** 2)))

    ours_vs_differentiated = rms(ours, differentiated)
    dashed_vs_differentiated = rms(dashed, differentiated)
    assert ours_vs_differentiated < dashed_vs_differentiated, (
        ours_vs_differentiated,
        dashed_vs_differentiated,
    )
    # And the figure really is inconsistent with itself here, rather
    # than all three agreeing and the ordering being noise.
    assert dashed_vs_differentiated > 0.02, dashed_vs_differentiated


@pytest.mark.parametrize("tag", ["fast", "slow"])
def test_the_fig37_stoneley_limit_matches_biots_closed_form(tag: str) -> None:
    """A falsifiable anchor the thesis supplies itself.

    Its eq 3.2.2 gives the low-frequency Stoneley limit in closed form,
    ``v/v_f = 1 / sqrt(1 + rho_f v_f^2 / (rho v_s^2))``, with no
    reference to the figure. Both the digitised curve and fwap have to
    land on it, and neither was fitted to the other.
    """
    stack = _CLARO_STACK[tag]
    predicted = stack["vf"] / np.sqrt(
        1.0 + stack["rho_f"] * stack["vf"] ** 2 / (stack["rho"] * stack["vs"] ** 2)
    )
    reference = _claro_reference(tag, "stoneley", "phase")
    freq, phase, _ = _claro_curves(tag, "stoneley")

    assert reference[0, 0] < 250.0, reference[0, 0]
    digitised = 1.0 / reference[0, 1]
    assert abs(digitised - predicted) / predicted < 0.003, (digitised, predicted)
    assert abs(1.0 / phase[0] - predicted) / predicted < 0.002


@pytest.mark.parametrize(
    ("tag", "mode", "peak"),
    [
        ("fast", "flexural", 247.2),
        ("fast", "quadrupole", 244.0),
        ("slow", "flexural", 191.9),
        ("slow", "quadrupole", 192.6),
    ],
)
def test_the_fig37_airy_phase_peaks_match(tag: str, mode: str, peak: float) -> None:
    """The Airy phase, which is the feature the thesis names.

    Its height is what a group-slowness curve is read for -- it sets the
    amplitude maximum of the recorded waveform -- and it is an interior
    maximum, so it cannot be reproduced by getting the endpoints right.
    The peak *frequency* is deliberately not asserted tightly: these
    maxima are broad and flat, so their location is the poorly
    determined half of the measurement.

    The ``peak`` values are **read off the shipped curve, not quoted
    from the thesis** -- it states no numbers for these maxima. They are
    written out so that a silent change to the CSV shows up as a failure
    here rather than being absorbed, and the first assertion below is
    what keeps the two in step.
    """
    reference = _claro_reference(tag, mode, "group")
    assert abs(reference[:, 1].max() * _CLARO_K - peak) < 0.2, peak

    freq, _, group = _claro_curves(tag, mode, n=800)
    top = int(np.argmax(group))
    assert 0 < top < len(freq) - 1, top
    height = group[top] * _CLARO_K
    assert abs(height - peak) / peak < 0.01, (height, peak)

    published = int(np.argmax(reference[:, 1]))
    assert abs(freq[top] - reference[published, 0]) < 700.0


def test_the_fast_formation_branch_continues_past_the_fluid_slowness() -> None:
    """The window edge fig 3.7(a) exposed, now on the far side of it.

    This test was written to pin the edge so that closing it registered
    as a change rather than passing unnoticed, and that is what it is
    doing: it used to assert ``phase.max() < fluid_slowness``.

    ``_march_fast_flexural_branch`` searches phase velocity in
    ``(V_f, V_S)`` because above ``V_f`` the fluid radial wavenumber is
    imaginary. The branch does not stop there -- it descends *through*
    ``V_f`` toward Scholte, and below ``V_f`` all three radial
    wavenumbers are real again, so the ordinary real determinant picks
    it up. Both orders now run past the 203 us/ft fluid, the dipole to
    212.8 us/ft by 20 kHz and the quadrupole to 207.2.
    """
    fluid_slowness = 1.0 / _CLARO_FAST["vf"]
    for mode, reach in (("flexural", 1.045), ("quadrupole", 1.02)):
        freq, phase, _ = _claro_curves("fast", mode)
        beyond = phase > fluid_slowness
        assert beyond.sum() > 40, (mode, beyond.sum())
        assert phase.max() > fluid_slowness * reach, (mode, phase.max())

        # It goes as far as the published curve does, and no further.
        reference = _claro_reference("fast", mode, "phase")
        assert phase.max() < reference[:, 1].max() * 1.01, mode

    # The slow panel never had this edge: there V_f is faster than V_S,
    # so the whole branch is bound and fwap covers the figure to 20 kHz.
    for mode in ("flexural", "quadrupole"):
        freq, _, _ = _claro_curves("slow", mode)
        assert freq.max() > 19900.0, mode


def test_the_fast_branch_crosses_the_fluid_velocity_exactly_once() -> None:
    """The two search regimes join into one curve, not two.

    The above- and below-``V_f`` passes use different determinants over
    disjoint windows, so the risk is a seam: a jump, a repeat, or a
    stretch where both claim a root. None of those is present. The
    branch is monotone in slowness and crosses ``V_f`` once.
    """
    for tag in ("fast", "slow"):
        for mode in ("flexural", "quadrupole"):
            freq, phase, _ = _claro_curves(tag, mode, n=600)
            assert np.all(np.diff(phase) > 0.0), (tag, mode)
            crossings = int(
                np.sum(np.diff(np.sign(phase - 1.0 / _CLARO_STACK[tag]["vf"])) != 0)
            )
            expected = 1 if tag == "fast" else 0
            assert crossings == expected, (tag, mode, crossings)


def test_the_sub_fluid_window_holds_at_most_one_root() -> None:
    """The measurement the sub-fluid search rests on, in miniature.

    ``_extend_below_fluid`` brackets each frequency once over the whole
    of ``(0.5 V_f, V_f)`` instead of scanning it, which is only sound
    because nothing else lives down there. The full sweep behind that
    covered 90 fast formations at two azimuthal orders and five
    frequencies each -- 900 windows, never more than one sign change.
    This is a fast corner of it, so the claim is checked rather than
    quoted.
    """
    from fwap.cylindrical_solver._n1_isotropic import (
        _FAST_FLEXURAL_SUB_FLUID_FLOOR,
        _modal_determinant_n1,
    )
    from fwap.cylindrical_solver._n2_quadrupole import _modal_determinant_n2

    seen = set()
    for stack in (_CLARO_FAST, dict(_CLARO_FAST, a=0.05)):
        medium = {k: stack[k] for k in ("vp", "vs", "rho", "vf", "rho_f", "a")}
        floor = medium["vf"] * _FAST_FLEXURAL_SUB_FLUID_FLOOR
        velocity = np.linspace(floor, medium["vf"] * (1.0 - 1.0e-4), 1500)
        for det in (_modal_determinant_n1, _modal_determinant_n2):
            for f in (8.0e3, 15.0e3, 25.0e3, 40.0e3):
                omega = 2.0 * np.pi * f
                values = np.array([det(omega / v, omega, **medium) for v in velocity])
                finite = np.isfinite(values)
                sign = np.sign(values)
                changes = int(
                    np.sum((sign[:-1] * sign[1:] < 0) & finite[:-1] & finite[1:])
                )
                assert changes <= 1, (medium["a"], det.__name__, f, changes)
                seen.add(changes)

    # And it is not vacuous -- some of those windows really do hold one.
    assert seen == {0, 1}, seen


def test_the_fast_formation_high_frequency_limit_is_scholte_too() -> None:
    """The oracle the ``(V_f, V_S)`` window used to put out of reach.

    ``scholte_speed`` solves a plane fluid/solid interface -- no Bessel
    functions, no radius, no azimuthal order -- so this is an external
    check rather than the solver agreeing with itself. It stays external
    here: the sub-fluid search floor is a fraction of ``V_f`` and not the
    Scholte speed, precisely so that bounding the search does not become
    the thing being tested. See
    :data:`~fwap.cylindrical_solver._n1_isotropic._FAST_FLEXURAL_SUB_FLUID_FLOOR`.

    The two orders approach from opposite sides, which is why n=2 is not
    asked for a monotone ``|error|``: n=1 comes up from below, while n=2
    comes down from above and crosses a few parts in ten thousand past
    it before settling. Both land inside 1e-3 either way.
    """
    from fwap import flexural_dispersion, quadrupole_dispersion, scholte_speed

    frequencies = np.array([50.0e3, 100.0e3, 200.0e3, 400.0e3])
    for rock in (_QUAD_FAST, dict(vp=3658.0, vs=2032.0, rho=2350.0)):
        assert rock["vs"] > _QUAD_FLUID["vf"], "this block is about fast formations"
        reference = scholte_speed(**rock, **_QUAD_FLUID)
        for solver in (flexural_dispersion, quadrupole_dispersion):
            result = solver(frequencies, **rock, **_QUAD_FLUID, a=0.10)
            assert np.all(np.isfinite(result.slowness)), solver.__name__
            error = np.abs(1.0 / result.slowness / reference - 1.0)
            assert error[-1] < 1.0e-3, (solver.__name__, error)
            # Converging, not merely close.
            assert error[0] > 3.0 * error[-1], (solver.__name__, error)


def test_the_fast_formation_limit_is_not_the_fluid_velocity() -> None:
    """The specific wrong answer the old window produced, pinned.

    Higher-order modes accumulate at ``V_f`` from above, so a search
    confined to ``(V_f, V_S)`` finds one of them once the fundamental
    has crossed, and it converges tidily -- to ``V_f``. On the fast
    sandstone at 50/100/200/400 kHz it used to return 1.0217, 1.0048,
    1.0011 and 1.0003 ``V_f``.

    That is a plausible-looking curve, which is exactly why it needs a
    test naming it. The Scholte speed is 0.99 % below ``V_f`` for this
    rock -- a small gap, but the branch now sits within 0.11 % of
    Scholte at 100 kHz and 0.03 % at 400 kHz, so the two limits are
    still told apart by an order of magnitude.
    """
    from fwap import flexural_dispersion, quadrupole_dispersion, scholte_speed

    rock, fluid = _QUAD_FAST, _QUAD_FLUID
    reference = scholte_speed(**rock, **fluid)
    separation = 1.0 - reference / fluid["vf"]
    assert separation > 5.0e-3, "the two limits must be distinguishable"

    frequencies = np.array([100.0e3, 400.0e3])
    for solver in (flexural_dispersion, quadrupole_dispersion):
        velocity = 1.0 / solver(frequencies, **rock, **fluid, a=0.10).slowness
        assert np.all(velocity < fluid["vf"]), (solver.__name__, velocity)
        missed = np.abs(velocity / reference - 1.0)
        assert np.all(missed < 0.25 * separation), (solver.__name__, missed)


def test_the_sub_fluid_search_never_returns_its_own_floor() -> None:
    """No answer beats an answer pinned to a constant of the search.

    Above roughly 500 kHz the real determinant underflows to exactly
    ``0.0`` over much of the sub-fluid window, and the floor end goes
    first. ``np.sign(0.0)`` is 0, so a zero endpoint passes a naive
    opposite-sign test and brentq then returns that endpoint -- an
    answer sitting on ``_FAST_FLEXURAL_SUB_FLUID_FLOOR``, which is a
    statement about the constant rather than about the rock.

    This is the same shape as the two leaky-cased constants withdrawn
    earlier: a number that comes back looking like a mode because a
    guard, not the physics, put it there.
    """
    from fwap import flexural_dispersion, quadrupole_dispersion
    from fwap.cylindrical_solver._n1_isotropic import _FAST_FLEXURAL_SUB_FLUID_FLOOR

    frequencies = np.array([600.0e3, 800.0e3, 1.2e6])
    floor = _CLARO_FAST["vf"] * _FAST_FLEXURAL_SUB_FLUID_FLOOR
    for solver in (flexural_dispersion, quadrupole_dispersion):
        slowness = solver(frequencies, **_CLARO_FAST).slowness
        live = np.isfinite(slowness)
        if live.any():
            assert np.all(1.0 / slowness[live] > floor * 1.5), solver.__name__
        # And where it cannot answer it says so, rather than handing
        # back the higher-order mode still sitting above V_f.
        assert not np.all(live), solver.__name__


# ----------------------------------------------------------------------
# The quadrupole's low-frequency plateau: why it is not a coverage gap
#
# Claro fig 3.7 draws the quadrupole flat at the formation shear
# slowness from 200 Hz up to about 6 kHz, and fwap returns NaN over all
# of it -- 105 of the fast panel's 391 points and 112 of the slow
# panel's 391. That looks like the V_f window edge #124 closed, and it
# is not the same thing.
#
# There, a real root existed on the far side of a branch point and the
# search was not looking. Here there is no root to find: the trapped
# quadrupole has a genuine cut-off, and below it the mode is *leaky*,
# with phase velocity ABOVE V_S and complex k_z. Closing this "gap" by
# relaxing the eps margin at V_S would return a value pinned at V_S,
# which is not a mode -- it is the shear branch point wearing a mode's
# clothes, and it is what the published plateau looks like.
# ----------------------------------------------------------------------

_QUAD_PLATEAU_STACKS = {
    "claro fast": dict(_CLARO_FAST),
    "claro slow": dict(_CLARO_SLOW),
    "sinha fast": dict(
        vp=3658.0, vs=2032.0, rho=2350.0, vf=1500.0, rho_f=1000.0, a=0.1016
    ),
    "limestone": dict(
        vp=4000.0, vs=2300.0, rho=2500.0, vf=1500.0, rho_f=1000.0, a=0.10
    ),
}


def _n2_real_axis_determinant(medium: dict, omega: float, velocity: float) -> float:
    """The n=2 determinant on the real ``k_z`` axis, either regime."""
    from fwap.cylindrical_solver._n2_quadrupole import (
        _modal_determinant_n2,
        _modal_determinant_n2_complex,
    )

    if velocity > medium["vf"]:
        return float(
            _modal_determinant_n2_complex(
                complex(omega / velocity, 0.0),
                omega,
                **medium,
                leaky_p=False,
                leaky_s=False,
            ).real
        )
    return float(_modal_determinant_n2(omega / velocity, omega, **medium))


@pytest.mark.parametrize("name", sorted(_QUAD_PLATEAU_STACKS))
def test_the_quadrupole_cut_off_is_real_not_a_margin(name: str) -> None:
    """Below the cut-off there is no trapped root to have missed.

    The search excludes a thin ``eps`` band at ``V_S``, so the obvious
    reading of the plateau is that the root is hiding inside it. It is
    not: scanning the determinant from ``1 - c/V_S = 1e-10`` out to
    ``1e-1`` -- five orders of magnitude closer to the branch point than
    the margin -- finds **no sign change at all** below the cut-off, on
    four different rocks.

    A relaxed margin would therefore not recover a mode. It would return
    ``c = V_S``, which is where the shear radial wavenumber vanishes and
    the determinant degenerates.
    """
    medium = _QUAD_PLATEAU_STACKS[name]
    velocity = medium["vs"] * (1.0 - np.logspace(-10.0, -1.0, 900))
    for f in (500.0, 1000.0, 2000.0, 3000.0):
        omega = 2.0 * np.pi * f
        values = np.array(
            [_n2_real_axis_determinant(medium, omega, v) for v in velocity]
        )
        finite = np.isfinite(values)
        sign = np.sign(values)
        changes = int(np.sum((sign[:-1] * sign[1:] < 0) & finite[:-1] & finite[1:]))
        assert changes == 0, (name, f, changes)


def test_the_only_sign_changes_near_v_s_are_round_off() -> None:
    """And the ones further in are noise, told apart by magnitude.

    Pushed to ``1 - c/V_S ~ 1e-13`` the scan does report sign changes,
    which is why the test above stops at ``1e-10``. They are round-off:
    a genuine root drives ``|det|`` toward zero, and these do not dip at
    all -- the determinant is order ``1e41`` on both sides of every one
    of them. Recording the discriminator matters more than recording the
    count, since the count depends on the grid and the discriminator
    does not.
    """
    medium = _QUAD_PLATEAU_STACKS["claro fast"]
    omega = 2.0 * np.pi * 500.0
    velocity = medium["vs"] * (1.0 - np.logspace(-13.5, -12.5, 400))
    values = np.array([_n2_real_axis_determinant(medium, omega, v) for v in velocity])
    finite = np.isfinite(values)
    sign = np.sign(values)
    flips = np.flatnonzero((sign[:-1] * sign[1:] < 0) & finite[:-1] & finite[1:])

    assert flips.size > 0, "expected round-off flips this close to the branch point"
    magnitude = np.abs(values[finite])
    # No dip: the smallest |det| anywhere in the window is within a
    # couple of decades of the largest, so nothing is heading for zero.
    assert magnitude.min() > magnitude.max() * 1.0e-3, (
        magnitude.min(),
        magnitude.max(),
    )


@pytest.mark.parametrize("name", ["claro fast", "claro slow", "sinha fast"])
def test_below_the_cut_off_the_quadrupole_is_leaky_and_above_v_s(name: str) -> None:
    """What is actually down there, and why the plateau cannot be it.

    A complex root does exist below the cut-off, with the shear branch
    radiating (``leaky_s``) -- but its phase velocity is *above* ``V_S``,
    by around 1 % at the peak, and it lives in a narrow band rather than
    running to zero frequency. The published plateau sits *at* ``V_S``
    across the whole band, so it is neither this mode nor the trapped
    one.

    This is recorded rather than solved: turning it into a public
    dispersion function would be a new leaky regime and a new public
    name, which the contributor guide asks to raise as an issue first.
    """
    from fwap.cylindrical_solver._leaky import _track_complex_root
    from fwap.cylindrical_solver._n2_quadrupole import _modal_determinant_n2_complex

    medium = _QUAD_PLATEAU_STACKS[name]
    vs = medium["vs"]
    found = []
    for f in (3500.0, 4000.0, 4500.0):
        omega = 2.0 * np.pi * f

        def det(kz: complex, omega: float = omega) -> complex:
            return _modal_determinant_n2_complex(
                kz, omega, **medium, leaky_p=False, leaky_s=True
            )

        best = None
        # The seeds carry a *positive* imaginary part because that is
        # where these roots are; seeded negative, the solver mostly
        # walks off and the branch looks absent on two of the three
        # rocks.
        for seed_v in np.linspace(1.002, 1.03, 10) * vs:
            for seed_im in (0.05, 0.2, 0.5, 0.9):
                root = _track_complex_root(det, complex(omega / seed_v, seed_im))
                if root is None or root.real <= 0.0:
                    continue
                c = omega / root.real
                # Reject the fluid branch point. In a slow formation
                # V_f sits above V_S and inside this range, and the
                # determinant has a sign change pinned exactly at it
                # where the fluid radial wavenumber vanishes. It is the
                # same class of artefact as c = V_S itself, and on the
                # slow panel the search converges onto it (2200.72 m/s,
                # which is V_f to six figures) unless it is named.
                if abs(c / medium["vf"] - 1.0) < 1.0e-3:
                    continue
                if vs < c < 1.05 * vs and 0.0 < root.imag < 10.0:
                    best = (c, root.imag)
                    break
            if best is not None:
                break
        assert best is not None, (name, f)
        found.append(best)

    speeds = np.array([c for c, _ in found])
    losses = np.array([q for _, q in found])
    # Above the shear speed, by about a percent -- not at it.
    assert np.all(speeds > vs), (name, speeds)
    assert np.all(speeds < vs * 1.02), (name, speeds)
    assert speeds.max() > vs * 1.005, (name, speeds)
    # Genuinely radiating, and less so as the cut-off is approached.
    assert np.all(losses > 0.0), (name, losses)
    assert losses[0] > losses[-1], (name, losses)


# ----------------------------------------------------------------------
# The leaky quadrupole (n = 2, shear branch radiating)
#
# #125 established that the trapped quadrupole has a genuine cut-off,
# and that what lies below it is a radiating branch with phase velocity
# ABOVE V_S. #126 confirmed from the source that Sinha & Asvadurov
# fig 10's m = 1 curve is exactly that branch. This block scores the
# solver against all three panels of that figure -- phase 10(a), group
# 10(b), attenuation 10(c) -- which is what makes the tie meaningful:
# the attenuation checks Im(k_z), and no phase curve can see that.
# ----------------------------------------------------------------------

_LQ_SINHA = dict(vp=3658.0, vs=2032.0, rho=2350.0, vf=1500.0, rho_f=1000.0, a=0.1016)


def _lq_curves(medium: dict, freq: np.ndarray):
    """Return ``(freq, phase, group, dB/m)`` over the live band."""
    from fwap import leaky_quadrupole_dispersion

    mode = leaky_quadrupole_dispersion(freq, **medium)
    live = np.isfinite(mode.slowness)
    f_live = freq[live]
    phase = mode.slowness[live]
    omega = 2.0 * np.pi * f_live
    group = np.gradient(omega * phase, omega)
    # Sinha's dB convention, recovered from figs 11(c)/2(c): a
    # 10*log10 per metre of energy transport, not 20*log10 per metre
    # of phase advance.
    decibels = 8.686 * mode.attenuation_per_meter[live] * (phase / group) / 2.0
    return f_live, phase, group, decibels


def _lq_score(x: np.ndarray, y: np.ndarray, f: np.ndarray, curve: np.ndarray):
    keep = (x >= f.min()) & (x <= f.max())
    got = np.interp(x[keep], f, curve)
    residual = (got - y[keep]) / y[keep]
    return (
        int(keep.sum()),
        float(np.sqrt(np.mean(residual**2))),
        float(np.max(np.abs(residual))),
    )


def test_leaky_quadrupole_matches_sinha_fig10a_phase() -> None:
    """The 34 sub-cut-off points of the curve already shipped.

    They are the same CSV `quadrupole_dispersion` is scored against
    above; those 34 were simply out of its reach, because the trapped
    search stops at ``V_S`` and this part of the branch is past it.
    """
    data = Path(__file__).resolve().parents[1] / "docs" / "notebooks" / "_data"
    reference = np.loadtxt(
        data / "sinha_asvadurov_2004_fig10a_quadrupole_fast.csv", delimiter=","
    )
    freq = np.arange(3100.0, 5460.0, 20.0)
    f_live, phase, _, _ = _lq_curves(_LQ_SINHA, freq)
    n, rms, worst = _lq_score(reference[:, 0], reference[:, 1], f_live, phase)
    assert n >= 30, n
    assert rms < 0.01, rms
    assert worst < 0.02, worst


def test_leaky_quadrupole_matches_sinha_fig10c_attenuation() -> None:
    """The tie that matters: ``Im(k_z)``, which no phase curve sees.

    The floor is 0.2 dB/m, which is 1 % of that panel's 0-20 axis and
    about four times its digitising resolution. Below it the reference
    is a couple of pixel rows off zero and a *relative* budget stops
    meaning anything -- the same reasoning as the fig 11(c) floor, at a
    level set by this panel's own scale.
    """
    data = Path(__file__).resolve().parents[1] / "docs" / "notebooks" / "_data"
    reference = np.loadtxt(
        data / "sinha_asvadurov_2004_fig10c_quadrupole_attenuation_fast.csv",
        delimiter=",",
    )
    keep = reference[:, 1] > 0.2
    freq = np.arange(3100.0, 5460.0, 20.0)
    f_live, _, _, decibels = _lq_curves(_LQ_SINHA, freq)
    n, rms, worst = _lq_score(reference[keep, 0], reference[keep, 1], f_live, decibels)
    assert n >= 25, n
    assert rms < 0.03, rms
    assert worst < 0.06, worst


def test_leaky_quadrupole_matches_sinha_fig10b_group() -> None:
    """And the group slowness, over the part below the cut-off.

    Fig 10(b)'s m = 1 curve starts at 4.50 kHz, so only its bottom
    stretch overlaps the radiating band -- but that stretch is
    independent of both other panels.
    """
    data = Path(__file__).resolve().parents[1] / "docs" / "notebooks" / "_data"
    reference = np.loadtxt(
        data / "sinha_asvadurov_2004_fig10b_quadrupole_group_fast.csv", delimiter=","
    )
    freq = np.arange(3100.0, 5460.0, 20.0)
    f_live, _, group, _ = _lq_curves(_LQ_SINHA, freq)
    n, rms, worst = _lq_score(reference[:, 0], reference[:, 1], f_live, group)
    assert n >= 10, n
    assert rms < 0.04, rms
    assert worst < 0.06, worst


def test_the_leaky_quadrupole_is_above_v_s_and_radiating() -> None:
    """The two properties that make it the radiating branch at all.

    Phase velocity above ``V_S`` is what turns the shear radial
    wavenumber imaginary; a positive ``Im(k_z)`` is the outgoing wave
    carrying energy away. Either one alone would be satisfied by
    something else.
    """
    freq = np.arange(3100.0, 5460.0, 40.0)
    f_live, phase, _, _ = _lq_curves(_LQ_SINHA, freq)
    from fwap import leaky_quadrupole_dispersion

    mode = leaky_quadrupole_dispersion(freq, **_LQ_SINHA)
    live = np.isfinite(mode.slowness)

    velocity = 1.0 / phase
    assert np.all(velocity > _LQ_SINHA["vs"]), velocity.min()
    assert velocity.max() < _LQ_SINHA["vs"] * 1.02, velocity.max()
    assert np.all(mode.attenuation_per_meter[live] > 0.0)
    # It radiates harder the further below the cut-off it goes.
    order = np.argsort(f_live)
    loss = mode.attenuation_per_meter[live][order]
    assert loss[0] > 5.0 * loss[-1], (loss[0], loss[-1])


def test_the_leaky_quadrupole_hands_over_to_the_trapped_one() -> None:
    """The two solvers meet at ``V_S`` and do not overlap.

    ``quadrupole_dispersion`` covers ``c < V_S`` and this covers
    ``c > V_S``, so between them they should cover the branch once --
    not twice, and not with a gap wider than the crossing itself.
    """
    from fwap import leaky_quadrupole_dispersion, quadrupole_dispersion

    freq = np.arange(3000.0, 9000.0, 50.0)
    trapped = quadrupole_dispersion(freq, **_LQ_SINHA).slowness
    leaky = leaky_quadrupole_dispersion(freq, **_LQ_SINHA).slowness
    both = np.isfinite(trapped) & np.isfinite(leaky)
    assert both.sum() == 0, freq[both]

    live = np.isfinite(trapped) | np.isfinite(leaky)
    assert live.sum() > 80, live.sum()
    # The handover is one grid step wide, not a band.
    edge_leaky = freq[np.isfinite(leaky)].max()
    edge_trapped = freq[np.isfinite(trapped)].min()
    assert 0.0 < edge_trapped - edge_leaky < 200.0, (edge_leaky, edge_trapped)


def test_leaky_quadrupole_rejects_bad_input() -> None:
    """Same guards as its trapped sister."""
    from fwap import leaky_quadrupole_dispersion

    good = dict(_LQ_SINHA)
    with pytest.raises(ValueError, match="vp > vs"):
        leaky_quadrupole_dispersion(np.array([4.0e3]), **{**good, "vp": 1000.0})
    with pytest.raises(ValueError, match="strictly positive"):
        leaky_quadrupole_dispersion(np.array([0.0]), **good)
    with pytest.raises(ValueError, match="a must be positive"):
        leaky_quadrupole_dispersion(np.array([4.0e3]), **{**good, "a": 0.0})
    empty = leaky_quadrupole_dispersion(np.array([]), **good)
    assert empty.slowness.size == 0
    assert empty.azimuthal_order == 2


# ----------------------------------------------------------------------
# Why the low-frequency phase drift is not a search defect
#
# fwap's leaky quadrupole peaks at 1.009 V_S against fig 10(a)'s 1.019,
# so what residual there is sits at the strongly radiating end. The
# obvious reading is that the complex search loses the root there. It
# does not, and these tests pin the measurements that exclude it:
#
#   * the peak is a property of the determinant, not of the medium
#     constants or the grid (`..._peak_is_robust_...`);
#   * the one discontinuous branch test in the n=2 matrix never fires
#     along the branch (`..._branch_selection_is_not_in_play`);
#   * and, the discriminating one, the *attenuation* residual is flat
#     in damping while the phase residual is not
#     (`..._residual_tracks_damping_but_the_attenuation_does_not`).
#
# The last is why this is recorded rather than tuned away. Im(k_z) is
# what the leaky machinery produces -- the trapped search runs the same
# matrix with leaky_s=False -- and it is uniformly right exactly where
# the phase drifts. A lost or mis-sheeted root would miss in both.
#
# Two further exclusions are not tests because they assert about a
# root that does not exist, which no assertion can hold onto: a survey
# of c in (V_S, 1.30 V_S) x Im(k_z) in (0, 8) at 3.2418 kHz finds
# exactly one genuine zero, 13.2 decades below its surroundings, at
# c/V_S = 1.0047; and |det| at the published value bottoms out in a
# 22x dip, which is not a root. Flipping the leaky sheet does not put
# one there either.
#
# These tests establish that the drift is not a search defect. They do
# not say which side is nearer the truth -- that was left open here and
# is settled further down by the Sinha-appendix oracle, in fwap's
# favour: see `test_the_published_equations_and_the_published_figure_
# disagree`.
# ----------------------------------------------------------------------


def test_the_leaky_quadrupole_residual_tracks_damping_but_the_attenuation_does_not():
    """The measurement that separates "fwap is lost" from "they differ".

    Asserted as a contrast between two correlations rather than as
    fitted budgets, so it pins the finding without pinning noise. The
    phase residual is very nearly a function of the damping and falls
    away with it; the attenuation residual has no such trend at all.

    The same damping law is already recorded at ``n = 0``, on a
    different figure and formation, by
    ``test_pseudo_rayleigh_attenuation_matches_sinha_fig2c_m3`` -- so
    it is a property of these comparisons, not of the quadrupole.
    """
    from fwap import leaky_quadrupole_dispersion

    data = Path(__file__).resolve().parents[1] / "docs" / "notebooks" / "_data"
    freq = np.arange(3100.0, 5460.0, 10.0)
    f_live, phase, _, decibels = _lq_curves(_LQ_SINHA, freq)
    mode = leaky_quadrupole_dispersion(freq, **_LQ_SINHA)
    damping_curve = mode.attenuation_per_meter[np.isfinite(mode.slowness)]

    def _versus_damping(name, curve, floor=None):
        reference = np.loadtxt(data / name, delimiter=",")
        keep = (reference[:, 0] >= f_live.min()) & (reference[:, 0] <= f_live.max())
        if floor is not None:
            keep &= reference[:, 1] > floor
        x, y = reference[keep, 0], reference[keep, 1]
        got = np.interp(x, f_live, curve)
        damping = np.interp(x, f_live, damping_curve)
        residual = np.abs(got - y) / y
        split = float(np.median(damping))
        low, high = residual[damping < split], residual[damping >= split]
        return (
            int(keep.sum()),
            float(np.corrcoef(residual, damping)[0, 1]),
            float(high.mean() / low.mean()),
        )

    n_phase, corr_phase, ratio_phase = _versus_damping(
        "sinha_asvadurov_2004_fig10a_quadrupole_fast.csv", phase
    )
    n_atten, corr_atten, ratio_atten = _versus_damping(
        "sinha_asvadurov_2004_fig10c_quadrupole_attenuation_fast.csv",
        decibels,
        floor=0.2,
    )
    assert n_phase >= 30 and n_atten >= 25, (n_phase, n_atten)

    # The phase residual is essentially a function of the damping.
    assert corr_phase > 0.9, corr_phase
    assert ratio_phase > 4.0, ratio_phase

    # The attenuation residual is not. This is the discriminating fact.
    assert abs(corr_atten) < 0.4, corr_atten
    assert ratio_atten < 1.5, ratio_atten

    # And the two are not merely different, they are far apart.
    assert corr_phase > corr_atten + 0.6, (corr_phase, corr_atten)
    assert ratio_phase > 3.0 * ratio_atten, (ratio_phase, ratio_atten)


def test_the_leaky_quadrupole_peak_is_robust_to_the_medium_constants():
    """The 1.009 peak is a property of the determinant, not of inputs.

    Reaching fig 10(a)'s 1.019 would need a medium far outside anything
    the paper's table supports, so a mis-read constant is not the
    explanation. The radius check is the sharp one: the peak is
    dimensionless, so scaling ``a`` must move the frequency and leave
    the value alone -- a statement about the solver that needs no
    reference curve at all.
    """
    from fwap import leaky_quadrupole_dispersion

    def _peak(medium):
        freq = np.arange(2900.0, 5500.0, 20.0)
        mode = leaky_quadrupole_dispersion(freq, **medium)
        live = np.isfinite(mode.slowness)
        assert live.sum() > 20, medium
        ratio = 1.0 / mode.slowness[live] / medium["vs"]
        i = int(np.argmax(ratio))
        return float(ratio[i]), float(freq[live][i])

    base, base_freq = _peak(_LQ_SINHA)
    assert 1.005 < base < 1.015, base

    # Dimensionless: radius rescales the frequency and not the peak.
    for factor in (0.9, 1.1):
        scaled, scaled_freq = _peak({**_LQ_SINHA, "a": _LQ_SINHA["a"] * factor})
        assert abs(scaled - base) < 5.0e-4, (factor, scaled, base)
        # ... and moves it the other way, as f ~ 1/a.
        assert (scaled_freq - base_freq) * (factor - 1.0) < 0.0, (factor, scaled_freq)

    # Every other constant, pushed well past any plausible mis-reading,
    # leaves the peak inside a fifth of the gap to the figure's 1.019.
    for key, factors in (
        ("vp", (0.90, 1.10)),
        ("rho", (0.85, 1.15)),
        ("rho_f", (0.85, 1.15)),
        ("vf", (0.90, 1.10)),
    ):
        for factor in factors:
            moved, _ = _peak({**_LQ_SINHA, key: _LQ_SINHA[key] * factor})
            assert abs(moved - base) < 0.002, (key, factor, moved, base)


def test_the_leaky_quadrupole_branch_selection_is_not_in_play():
    """The fluid branch test never fires along the branch.

    ``_modal_determinant_n2_complex`` picks the fluid radial wavenumber
    with ``leaky = F_sq.real < 0``, a discontinuous test on a complex
    quantity -- the shape of thing that puts a kink in a marched branch.
    Here ``Re(F^2)`` stays negative by a wide margin from end to end, so
    the same side is taken at every frequency.

    Also checks the regime the selection exists to express: P bound and
    S radiating throughout, which is what makes this the leaky branch.
    """
    from fwap import leaky_quadrupole_dispersion

    freq = np.arange(3100.0, 5460.0, 10.0)
    mode = leaky_quadrupole_dispersion(freq, **_LQ_SINHA)
    live = np.isfinite(mode.slowness)
    assert live.sum() > 200, live.sum()
    omega = 2.0 * np.pi * freq[live]
    kz = omega * mode.slowness[live] + 1j * mode.attenuation_per_meter[live]

    f_sq = (kz * kz - (omega / _LQ_SINHA["vf"]) ** 2).real
    assert np.all(f_sq < 0.0), f_sq.max()
    assert f_sq.max() < -50.0, f_sq.max()

    assert np.all((kz * kz - (omega / _LQ_SINHA["vp"]) ** 2).real > 0.0)
    assert np.all((kz * kz - (omega / _LQ_SINHA["vs"]) ** 2).real < 0.0)


# ======================================================================
# An independent oracle: Sinha & Asvadurov's own published matrix
#
# #128 measured the leaky quadrupole's low-frequency drift against
# fig 10(a) and excluded every search-side cause, but left open which
# side was nearer the truth -- fwap's n=2 leaky determinant, or the
# paper's strongly-damped values. This settles it, because the paper
# prints the matrix.
#
# Sinha & Asvadurov (2004), Geophysical Prospecting 52, 271-286,
# Appendix eqs (A2)-(A15) give the 4x4 boundary-condition matrix L for
# a fluid-filled borehole of radius a in an infinite elastic formation,
# at general cylindrical order n. It shares no algebra with fwap: a
# different potential basis (their SV/SH columns are a mixture of
# fwap's), a different radial-wavenumber convention (theirs is the
# negative of fwap's), ordinary Hankel functions instead of modified
# Bessel ones, and rows scaled without the shear modulus.
#
# Transcribed below verbatim. It reproduces fwap's roots at n=1
# flexural, n=2 trapped and n=2 leaky alike -- and, at the frequencies
# where fwap and fig 10(a) disagree, the paper's own equations land on
# fwap's answer rather than on the paper's plotted curve.
# ======================================================================


def _sinha_wavenumber(alpha_sq_fwap: complex, leaky: bool) -> complex:
    """Sinha's ``alpha`` / ``beta`` from fwap's radial wavenumber.

    fwap works with ``alpha^2 = k_z^2 - (omega/V)^2`` and Sinha with the
    negative of it, so the two differ by a factor of ``+/- i`` -- and
    which sign is not a matter of taste:

    * **bound**: fwap takes ``Re(p) > 0`` and evaluates ``K_n(p r)``.
      The Hankel form matching it is ``H^(1)(i p r)``, so Sinha's
      ``alpha = +i p``, giving ``Im(alpha) > 0`` and ``H^(1) ~ e^{-pr}``,
      decaying.
    * **leaky**: fwap takes ``Im(s) > 0`` and evaluates the outgoing
      continuation, which is ``H^(1)(-i s r)``, so Sinha's
      ``beta = -i s`` -- landing with ``Im(beta) < 0``, the radially
      growing leaky solution, which is the correct one.

    With **real** ``k_z`` the two rules coincide: ``alpha^2`` is
    negative real and the principal root is already ``+i p``. So a
    principal-root transcription reproduces every bound mode and then
    silently selects the growing P wave the moment ``k_z`` goes
    complex, which is what makes this worth spelling out.
    """
    root = np.sqrt(complex(alpha_sq_fwap))
    if leaky:
        if root.imag < 0.0:
            root = -root
        return -1j * root
    if root.real < 0.0:
        root = -root
    return 1j * root


def _sinha_appendix_matrix(
    n: int,
    kz: complex,
    omega: float,
    *,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    leaky_p: bool = False,
    leaky_s: bool = False,
) -> np.ndarray:
    """``L`` from Sinha & Asvadurov (2004) Appendix (A2)-(A15).

    Their ``zeta`` is the axial wavenumber, ``alpha_f`` / ``alpha`` /
    ``beta`` the fluid / P / S radial wavenumbers, ``J`` the fluid
    Bessel function and ``H`` the outgoing Hankel function. Entries not
    listed in the appendix are zero (the fluid carries no shear).
    """
    from scipy import special

    zeta = complex(kz)
    f_sq = zeta**2 - (omega / vf) ** 2
    alpha_f = _sinha_wavenumber(f_sq, bool(f_sq.real < 0.0))
    alpha = _sinha_wavenumber(zeta**2 - (omega / vp) ** 2, leaky_p)
    beta = _sinha_wavenumber(zeta**2 - (omega / vs) ** 2, leaky_s)

    mu = rho * vs * vs
    lam = rho * (vp * vp - 2.0 * vs * vs)
    lam_f = rho_f * vf * vf

    jn, jn1 = special.jv(n, alpha_f * a), special.jv(n + 1, alpha_f * a)
    hna, hn1a = special.hankel1(n, alpha * a), special.hankel1(n + 1, alpha * a)
    hnb, hn1b = special.hankel1(n, beta * a), special.hankel1(n + 1, beta * a)

    L = np.zeros((4, 4), dtype=complex)
    L[0, 0] = -n * jn / a + alpha_f * jn1
    L[0, 1] = n * hna / a - alpha * hn1a
    L[0, 2] = 1j * zeta * hn1b
    L[0, 3] = n * hnb / a

    L[1, 0] = lam_f * (alpha_f**2 + zeta**2) * jn
    L[1, 1] = (
        -(lam * (alpha**2 + zeta**2) + 2.0 * mu * (alpha**2 - n * (n - 1) / a**2)) * hna
        + 2.0 * mu * alpha * hn1a / a
    )
    L[1, 2] = 2j * mu * zeta * beta * (hnb - (n + 1) * hn1b / (beta * a))
    L[1, 3] = 2.0 * mu * (n * (n - 1) * hnb / a**2 - n * beta * hn1b / a)

    L[2, 1] = -2.0 * n * (n - 1) * hna / a**2 + 2.0 * n * alpha * hn1a / a
    L[2, 2] = 1j * zeta * beta * hnb - 2j * zeta * (n + 1) * hn1b / a
    L[2, 3] = (beta**2 - 2.0 * n * (n - 1) / a**2) * hnb - 2.0 * beta * hn1b / a

    L[3, 1] = 2j * zeta * n * hna / a - 2j * zeta * alpha * hn1a
    L[3, 2] = -n * beta * hnb / a + (beta**2 - zeta**2) * hn1b
    L[3, 3] = 1j * zeta * n * hnb / a
    return L


def _sinha_appendix_determinant(n, kz, omega, **kw) -> complex:
    """``det L`` for the matrix above."""
    return complex(np.linalg.det(_sinha_appendix_matrix(n, kz, omega, **kw)))


def _sinha_root_depth(n, kz, omega, medium, **flags):
    """Decades ``|det L|`` drops at ``kz`` relative to its neighbourhood."""
    here = abs(_sinha_appendix_determinant(n, kz, omega, **medium, **flags))
    around = np.median(
        [
            abs(_sinha_appendix_determinant(n, kz * g, omega, **medium, **flags))
            for g in (0.97, 0.98, 1.02, 1.03)
        ]
    )
    return float(np.log10(around / here))


def test_sinha_appendix_matrix_vanishes_at_fwaps_bound_roots():
    """The published matrix agrees with fwap's, at n=1 and n=2 alike.

    Establishes the oracle before it is used on the leaky branch: two
    orders, two solvers, two independent formulations of the same
    boundary-value problem.
    """
    from fwap import flexural_dispersion, quadrupole_dispersion

    for order, solver, freq in (
        (1, flexural_dispersion, np.array([4000.0, 6000.0, 8000.0, 11000.0])),
        (2, quadrupole_dispersion, np.array([6000.0, 8000.0, 10000.0, 13000.0])),
    ):
        mode = solver(freq, **_LQ_SINHA)
        checked = 0
        for f, s in zip(freq, mode.slowness):
            if not np.isfinite(s):
                continue
            omega = 2.0 * np.pi * f
            depth = _sinha_root_depth(order, omega * s, omega, _LQ_SINHA)
            assert depth > 8.0, (order, f, depth)
            checked += 1
        assert checked >= 3, (order, checked)


def test_sinha_appendix_matrix_vanishes_at_fwaps_leaky_quadrupole_roots():
    """And on the leaky branch -- which is what #128 left open.

    The drift against fig 10(a) is concentrated exactly here, so if
    fwap's n=2 leaky determinant were wrong, this is where the paper's
    own equations would part company with it. They do not.
    """
    from fwap import leaky_quadrupole_dispersion

    freq = np.arange(3100.0, 5460.0, 10.0)
    mode = leaky_quadrupole_dispersion(freq, **_LQ_SINHA)
    live = np.isfinite(mode.slowness)
    f_live = freq[live]
    slowness = mode.slowness[live]
    damping = mode.attenuation_per_meter[live]

    checked = 0
    for target in (3240.0, 3600.0, 4000.0, 4500.0, 5000.0):
        i = int(np.argmin(np.abs(f_live - target)))
        omega = 2.0 * np.pi * f_live[i]
        kz = omega * slowness[i] + 1j * damping[i]
        depth = _sinha_root_depth(2, kz, omega, _LQ_SINHA, leaky_s=True)
        assert depth > 8.0, (f_live[i], depth)
        checked += 1
    assert checked == 5, checked


def test_the_published_equations_and_the_published_figure_disagree():
    """The finding: fig 10(a)'s low-frequency limb is the outlier.

    At the strongly radiating end fwap and fig 10(a) differ by up to
    1.4 %. Sinha & Asvadurov's own appendix equations are checked
    against both. They land on fwap -- to a part in 1e10, which is
    root-solver tolerance, not agreement within a budget -- while
    differing from the curve plotted in the same paper by the full
    drift.

    Asserted as the *contrast* between those two distances, which is
    the whole content of the finding and needs no fitted budget.
    """
    from scipy import optimize

    from fwap import leaky_quadrupole_dispersion

    data = Path(__file__).resolve().parents[1] / "docs" / "notebooks" / "_data"
    reference = np.loadtxt(
        data / "sinha_asvadurov_2004_fig10a_quadrupole_fast.csv", delimiter=","
    )
    freq = np.arange(3100.0, 5460.0, 10.0)
    mode = leaky_quadrupole_dispersion(freq, **_LQ_SINHA)
    live = np.isfinite(mode.slowness)
    f_live = freq[live]
    slowness = mode.slowness[live]
    damping = mode.attenuation_per_meter[live]

    def _root_of_the_paper(omega, seed):
        def residual(v):
            kz = complex(v[0], v[1])
            value = _sinha_appendix_determinant(2, kz, omega, **_LQ_SINHA, leaky_s=True)
            scale = (
                abs(
                    _sinha_appendix_determinant(
                        2, kz * 1.02, omega, **_LQ_SINHA, leaky_s=True
                    )
                )
                or 1.0
            )
            return [value.real / scale, value.imag / scale]

        found = optimize.root(
            residual, [seed.real, seed.imag], method="hybr", tol=1e-14
        )
        return complex(found.x[0], found.x[1]) if found.success else None

    from_fwap, from_figure = [], []
    for target in (3241.8, 3625.4, 4003.4, 4375.7, 4745.2):
        i = int(np.argmin(np.abs(f_live - target)))
        omega = 2.0 * np.pi * f_live[i]
        seed = omega * slowness[i] + 1j * damping[i]
        root = _root_of_the_paper(omega, seed)
        assert root is not None, target

        j = int(np.argmin(np.abs(reference[:, 0] - target)))
        from_fwap.append(abs(root.real - seed.real) / seed.real)
        # root.real is a wavenumber; the reference column is a slowness.
        from_figure.append(abs(root.real / omega - reference[j, 1]) / reference[j, 1])

    from_fwap = np.array(from_fwap)
    from_figure = np.array(from_figure)

    # The paper's equations reproduce fwap to solver tolerance ...
    assert from_fwap.max() < 1.0e-8, from_fwap.max()
    # ... while differing from the paper's own figure by the drift,
    # which grows toward the strongly radiating end.
    assert from_figure.max() > 0.008, from_figure.max()
    assert from_figure[0] > from_figure[-1], from_figure
    # Six orders of magnitude between the two distances.
    assert from_figure.min() > 1.0e5 * max(from_fwap.max(), 1e-16), (
        from_fwap.max(),
        from_figure.min(),
    )


def test_sinha_appendix_matrix_degenerates_correctly_at_n0():
    """At axisymmetry the torsional column decouples, as it must.

    The paper says the axisymmetric case follows by setting ``n = 0``
    in the flexural equations. Structurally that has to strip the
    borehole of any coupling between torsion and the Stoneley /
    pseudo-Rayleigh family, and it does: at ``n = 0`` the fourth column
    keeps a single nonzero entry, in the ``sigma_r_theta`` row. The
    determinant therefore factorises into a torsional condition times
    an axisymmetric 3x3, which is the physical statement that a
    borehole cannot excite torsion with an axisymmetric source.

    Checked as structure rather than as a number, so it holds for any
    medium.
    """
    for medium in (_LQ_SINHA, _LC_SLOW):
        for freq in (2000.0, 5000.0, 11000.0):
            omega = 2.0 * np.pi * freq
            L = _sinha_appendix_matrix(0, omega / 1800.0, omega, **medium)
            column = np.abs(L[:, 3])
            assert column[2] > 0.0, (medium, freq)
            assert np.all(column[[0, 1, 3]] == 0.0), column
            # ... and the same column is fully populated at n >= 1,
            # so this is degeneracy at n = 0 rather than a dead column.
            for order in (1, 2):
                busy = np.abs(
                    _sinha_appendix_matrix(order, omega / 1800.0, omega, **medium)[:, 3]
                )
                assert np.count_nonzero(busy) >= 3, (order, busy)


def test_sinha_appendix_matrix_vanishes_at_fwaps_n0_roots():
    """The oracle carries down to n = 0, bound and leaky alike.

    Four axisymmetric solvers, two formations, both regimes. With the
    n=1 and n=2 checks above this puts every open-hole order fwap
    solves under one independently published matrix.

    On ``leaky_p``: it stays False across the whole leaky
    pseudo-Rayleigh band, including where ``Re(p^2)`` dips negative near
    9.2 kHz. That dip is not a change of branch -- with complex ``k_z``
    the ``Im(k_z)^2`` term alone can push ``Re(p^2)`` below zero while
    the P wave is still bound -- so ``Re(p^2) < 0`` is not a usable
    leaky-P test once ``k_z`` leaves the real axis. Selecting the leaky
    P branch there costs all 14 decades.
    """
    from fwap import stoneley_dispersion
    from fwap.cylindrical_solver import (
        leaky_compressional_dispersion,
        pseudo_rayleigh_dispersion,
        trapped_pseudo_rayleigh_dispersion,
    )

    def _check(label, medium, freq, mode, leaky):
        live = np.isfinite(mode.slowness)
        assert live.sum() >= 3, (label, live.sum())
        f_live = freq[live]
        slowness = mode.slowness[live]
        damping = (
            mode.attenuation_per_meter[live] if leaky else np.zeros(int(live.sum()))
        )
        checked = 0
        for i in range(0, len(f_live), max(1, len(f_live) // 4)):
            omega = 2.0 * np.pi * f_live[i]
            kz = omega * slowness[i] + 1j * damping[i]
            depth = _sinha_root_depth(0, kz, omega, medium, leaky_s=leaky)
            assert depth > 8.0, (label, f_live[i], depth)
            checked += 1
        assert checked >= 3, (label, checked)

    freq = np.arange(2000.0, 13000.0, 250.0)
    _check("stoneley", _LQ_SINHA, freq, stoneley_dispersion(freq, **_LQ_SINHA), False)

    freq = np.arange(9000.0, 15200.0, 250.0)
    _check(
        "trapped pseudo-Rayleigh",
        _LQ_SINHA,
        freq,
        trapped_pseudo_rayleigh_dispersion(freq, **_LQ_SINHA, branch=0),
        False,
    )

    freq = np.arange(9000.0, 15200.0, 100.0)
    _check(
        "leaky pseudo-Rayleigh",
        _PR_SINHA_FAST,
        freq,
        pseudo_rayleigh_dispersion(freq, **_PR_SINHA_FAST, branch=1),
        True,
    )

    freq = np.arange(2500.0, 14000.0, 250.0)
    _check(
        "leaky compressional",
        _LC_SLOW,
        freq,
        leaky_compressional_dispersion(freq, **_LC_SLOW),
        True,
    )


def test_the_layered_machinery_reduces_to_the_published_open_hole_matrix():
    """Layers made of formation are not a layer, and the oracle knows it.

    ``flexural_dispersion_layered`` with a stack whose media equal the
    half-space is the *same boundary-value problem* as the open hole, so
    Sinha's published 4x4 has to hold for it -- but the code path is
    completely different: the 10x10 layered determinant at one layer and
    the cased propagator chain at two. This runs the whole layer
    assembly, interface conditions and half-space columns included,
    against a matrix from a different paper than the one it was built
    from.

    It is also how the sub-fluid gap below was found: the two disagreed
    above 10 kHz, in a configuration where they cannot.
    """
    medium = _LQ_SINHA
    one = (
        BoreholeLayer(
            vp=medium["vp"], vs=medium["vs"], rho=medium["rho"], thickness=0.03
        ),
    )
    two = (
        BoreholeLayer(
            vp=medium["vp"], vs=medium["vs"], rho=medium["rho"], thickness=0.02
        ),
        BoreholeLayer(
            vp=medium["vp"], vs=medium["vs"], rho=medium["rho"], thickness=0.04
        ),
    )
    cases = (
        (
            0,
            stoneley_dispersion,
            stoneley_dispersion_layered,
            np.arange(3000.0, 12000.0, 1000.0),
        ),
        (
            1,
            flexural_dispersion,
            flexural_dispersion_layered,
            np.arange(3000.0, 12500.0, 1000.0),
        ),
    )
    for n, unlayered, layered, freq in cases:
        base = unlayered(freq, **medium)
        for label, stack in (("10x10", one), ("propagator", two)):
            mode = layered(freq, **medium, layers=stack)
            checked = 0
            for i, f in enumerate(freq):
                if not np.isfinite(mode.slowness[i]):
                    continue
                # Same problem, so the same root to floating point ...
                assert np.isfinite(base.slowness[i]), (n, label, f)
                assert mode.slowness[i] == pytest.approx(base.slowness[i], rel=1e-11), (
                    n,
                    label,
                    f,
                )
                # ... and the published matrix agrees with both.
                omega = 2.0 * np.pi * f
                depth = _sinha_root_depth(n, omega * mode.slowness[i], omega, medium)
                assert depth > 8.0, (n, label, f, depth)
                checked += 1
            assert checked >= 8, (n, label, checked)


def test_the_layered_drivers_follow_the_branch_below_the_fluid_velocity():
    """The layered fast-formation paths cross ``V_f`` like the open-hole
    ones, instead of stopping there.

    ``_extend_below_fluid`` landed on the two unlayered drivers when it
    was written and on neither layered one, so above the crossing
    ``flexural_dispersion_layered`` and ``quadrupole_dispersion_layered``
    returned ``NaN`` where their unlayered twins tracked the mode --
    about 10 kHz at n=1 and 17 kHz at n=2 on this formation. The
    docstring claimed the two "cannot drift apart" because they share
    the marcher; they shared the marcher and differed in what happened
    after it.

    Asserted against the unlayered driver rather than against stored
    numbers: with the layers made of formation the two are the same
    problem, so any future divergence is a defect in one of them.
    """
    from fwap import quadrupole_dispersion, quadrupole_dispersion_layered

    medium = _LQ_SINHA
    stack = (
        BoreholeLayer(
            vp=medium["vp"], vs=medium["vs"], rho=medium["rho"], thickness=0.03
        ),
    )
    for n, unlayered, layered, freq in (
        (
            1,
            flexural_dispersion,
            flexural_dispersion_layered,
            np.arange(10000.0, 13000.0, 500.0),
        ),
        (
            2,
            quadrupole_dispersion,
            quadrupole_dispersion_layered,
            np.arange(18000.0, 25000.0, 1000.0),
        ),
    ):
        base = unlayered(freq, **medium)
        mode = layered(freq, **medium, layers=stack)
        assert np.isfinite(base.slowness).all(), n
        # The regime this is about: the branch is below the fluid speed.
        assert np.all(1.0 / base.slowness < medium["vf"]), n
        assert np.isfinite(mode.slowness).all(), (n, mode.slowness)
        np.testing.assert_allclose(mode.slowness, base.slowness, rtol=1e-10)
        for i, f in enumerate(freq):
            omega = 2.0 * np.pi * f
            depth = _sinha_root_depth(n, omega * mode.slowness[i], omega, medium)
            assert depth > 8.0, (n, f, depth)


def _iso_stiffness_from(medium):
    """The five VTI constants for an isotropic medium."""
    c44 = medium["rho"] * medium["vs"] ** 2
    c11 = medium["rho"] * medium["vp"] ** 2
    return dict(
        c11=c11,
        c13=c11 - 2.0 * c44,
        c33=c11,
        c44=c44,
        c66=c44,
        rho=medium["rho"],
        vf=medium["vf"],
        rho_f=medium["rho_f"],
        a=medium["a"],
    )


def _depth_of(fn, kz, omega, offsets):
    """Decades ``|fn|`` drops at ``kz`` relative to the given offsets.

    Takes the offsets explicitly because the VTI determinants are NaN
    by contract outside their regime: probing symmetrically across
    ``V_f`` reads as a failure when it is only the neighbourhood
    leaving the region where the function is defined.
    """
    here = abs(fn(kz, omega))
    around = np.median([abs(fn(kz * g, omega)) for g in offsets])
    if not np.isfinite(here) or here <= 0.0 or not np.isfinite(around):
        return float("nan")
    return float(np.log10(around / here))


def test_the_vti_determinants_reduce_to_the_published_isotropic_matrix():
    """VTI algebra at isotropic constants, against Sinha's 4x4.

    Nothing else checks this, and the public API cannot: both
    ``stoneley_dispersion_vti`` and ``flexural_dispersion_vti``
    dispatch to the isotropic solvers when ``_is_isotropic_stiffness``
    holds, so the VTI determinants are never *exercised* in the one
    limit where an independent answer exists. Called directly they are,
    and the published matrix vanishes wherever they do.

    The two *depths* are not asserted equal. Sinha's formulation has its
    own potential basis and row scaling, so the determinants differ by a
    smooth non-vanishing factor and their curvature at the root differs
    with it -- observed here between 0.0 and 1.2 decades apart. The
    shared root is the claim; equal depth would be a coincidence of
    normalisation.

    Three regimes, because the two determinants carry different
    validity windows: n=0 across the band, n=1 on a slow formation, and
    n=1 sub-fluid on a fast one where the real determinant is defined.
    """
    fast, slow = (
        _LQ_SINHA,
        dict(vp=2751.0, vs=1201.0, rho=2100.0, vf=1500.0, rho_f=1000.0, a=0.1016),
    )
    two_sided = (0.97, 0.98, 1.02, 1.03)
    sub_fluid = (1.02, 1.03, 1.05, 1.07)  # k_z up is c down: stays below V_f

    cases = (
        (
            "n=0",
            0,
            _modal_determinant_n0_vti,
            stoneley_dispersion,
            fast,
            np.arange(3000.0, 13000.0, 1500.0),
            two_sided,
        ),
        (
            "n=1 slow",
            1,
            _modal_determinant_n1_vti,
            flexural_dispersion,
            slow,
            np.arange(3000.0, 13000.0, 1500.0),
            two_sided,
        ),
        (
            "n=1 sub-fluid",
            1,
            _modal_determinant_n1_vti,
            flexural_dispersion,
            fast,
            np.arange(10500.0, 13000.0, 500.0),
            sub_fluid,
        ),
    )
    for label, n, vti_det, solver, medium, freq, offsets in cases:
        stiffness = _iso_stiffness_from(medium)
        mode = solver(freq, **medium)
        checked = 0
        for i, f in enumerate(freq):
            if not np.isfinite(mode.slowness[i]):
                continue
            omega = 2.0 * np.pi * f
            kz = omega * mode.slowness[i]
            published = _depth_of(
                lambda k, w: _sinha_appendix_determinant(n, k, w, **medium),
                kz,
                omega,
                offsets,
            )
            vti = _depth_of(lambda k, w: vti_det(k, w, **stiffness), kz, omega, offsets)
            assert published > 8.0, (label, f, published)
            assert vti > 8.0, (label, f, vti)
            checked += 1
        assert checked >= 4, (label, checked)


def test_the_vti_driver_follows_the_branch_below_the_fluid_velocity():
    """``flexural_dispersion_vti`` crosses ``V_f`` like every other
    fast-formation driver.

    Third instance of one omission. ``_extend_below_fluid`` was written
    for the unlayered isotropic drivers, reached neither layered one
    (fixed in #131) and did not reach the VTI one either, so on
    Thomsen's Green River shale the solver returned ``NaN`` from about
    7.7 kHz up -- while its own real determinant, which documents
    itself as valid exactly there, had the root.

    Asserted as continuity rather than against stored numbers: the
    fundamental crosses ``V_f`` once and keeps descending, so a branch
    that is smooth through the crossing is the statement worth
    holding.

    One sample may still be ``NaN``, the one whose root sits inside the
    epsilon of ``V_f`` that neither the above-``V_f`` search nor the
    sub-fluid one brackets. That is not a VTI defect and this fix does
    not claim to remove it: the isotropic driver drops exactly one
    sample at its own crossing too, and
    ``test_the_isotropic_and_vti_drivers_drop_the_same_single_sample``
    pins the two together rather than leaving it implied.
    """
    stiffness = _green_river_shale_stiffness()
    kwargs = dict(**stiffness, vf=1500.0, rho_f=1000.0, a=0.10)
    freq = np.arange(6000.0, 13500.0, 250.0)
    c = 1.0 / flexural_dispersion_vti(freq, **kwargs).slowness

    missing = ~np.isfinite(c)
    assert missing.sum() <= 1, c
    if missing.any():
        # ... and if one is missing it is the crossing, not a gap in
        # the sub-fluid leg.
        # One grid step is about 14 m/s in c here, so "at the
        # crossing" is a step, not a hair.
        assert abs(c[np.flatnonzero(missing)[0] - 1] - 1500.0) < 20.0, c
    good = c[~missing]
    # It really does cross, so the test is about the crossing.
    assert good[0] > 1500.0 and good[-1] < 1500.0
    assert (good < 1500.0).sum() >= 15, good
    # Monotone descent, and no step anywhere near the crossing.
    assert np.all(np.diff(good) < 0.0)
    assert np.max(np.abs(np.diff(good)) / good[:-1]) < 0.02

    # The recovered roots are roots of the VTI determinant itself.
    for i in np.flatnonzero((~missing) & (c < 1500.0))[:6]:
        omega = 2.0 * np.pi * freq[i]
        depth = _depth_of(
            lambda k, w: _modal_determinant_n1_vti(k, w, **kwargs),
            omega / c[i],
            omega,
            (1.02, 1.03, 1.05, 1.07),
        )
        assert depth > 8.0, (freq[i], depth)


def test_the_isotropic_and_vti_drivers_drop_the_same_single_sample():
    """The one gap at the ``V_f`` crossing is shared, and pre-dates VTI.

    Both fast-formation drivers lose exactly the frequency whose root
    lands within the epsilon of ``V_f``: the marcher searches strictly
    above it and :func:`_extend_below_fluid` strictly below, so a root
    sitting on the line is bracketed by neither. Recorded rather than
    papered over -- it is one sample in thirty on either side, and
    closing it means widening a bracket across the regime boundary,
    which is a change to the isotropic driver, not a VTI question.
    """
    medium = _LQ_SINHA
    freq = np.arange(9400.0, 10200.0, 25.0)
    c = 1.0 / flexural_dispersion(freq, **medium).slowness
    missing = ~np.isfinite(c)
    assert missing.sum() == 1, c
    # It is at the crossing, and it is the only one.
    assert abs(c[np.flatnonzero(missing)[0] - 1] - 1500.0) < 2.0, c
    good = c[~missing]
    assert good[0] > 1500.0 and good[-1] < 1500.0
    assert np.all(np.diff(good) < 0.0)


def test_the_marcher_cannot_be_called_without_a_sub_fluid_determinant():
    """The omission that hit three drivers is now unrepresentable.

    ``_extend_below_fluid`` used to be the caller's job, one line after
    the march. Three of the five drivers were written without it -- the
    layered n=1 and n=2 paths and the VTI one -- each returning ``NaN``
    over a band where its own determinant had the root, and each found
    separately, months apart, by a different oracle.

    They were not careless: nothing in the signature said the march was
    only half the branch. It does now. ``real_det`` is keyword-only and
    has no default, so a driver that forgets it fails at the call
    rather than silently returning a truncated branch, and this test
    fails if someone gives it one.
    """
    import inspect

    from fwap.cylindrical_solver._n1_isotropic import (
        _march_fast_flexural_branch,
        _modal_determinant_n1,
        _modal_determinant_n1_complex,
    )

    sig = inspect.signature(_march_fast_flexural_branch)
    real_det = sig.parameters["real_det"]
    assert real_det.kind is inspect.Parameter.KEYWORD_ONLY
    assert real_det.default is inspect.Parameter.empty, (
        "real_det must stay required: a default is exactly what let three "
        "drivers opt out of the sub-fluid continuation without saying so"
    )

    medium = _LQ_SINHA
    freq = np.array([9000.0, 11000.0])

    def _im_det(kz, omega):
        return _modal_determinant_n1_complex(
            complex(kz, 0.0),
            omega,
            medium["vp"],
            medium["vs"],
            medium["rho"],
            medium["vf"],
            medium["rho_f"],
            medium["a"],
            leaky_p=False,
            leaky_s=False,
        ).imag

    with pytest.raises(TypeError):
        _march_fast_flexural_branch(_im_det, freq, vs=medium["vs"], vf=medium["vf"])

    # And the positive half, so this is not only a test that something
    # fails: supplied properly, the same call marches and returns roots.
    def _real_det(kz, omega):
        return _modal_determinant_n1(
            kz,
            omega,
            medium["vp"],
            medium["vs"],
            medium["rho"],
            medium["vf"],
            medium["rho_f"],
            medium["a"],
        )

    marched = _march_fast_flexural_branch(
        _im_det, freq, vs=medium["vs"], vf=medium["vf"], real_det=_real_det
    )
    assert np.isfinite(marched).all(), marched
    # Spelled out rather than left to the comparison: 9 kHz sits above
    # the crossing on this formation and 11 kHz below it, so this call
    # exercises both legs and a marcher that stopped at V_f would give
    # NaN for the second sample.
    assert 1.0 / marched[0] > medium["vf"] > 1.0 / marched[1]
    np.testing.assert_allclose(
        marched, flexural_dispersion(freq, **medium).slowness, rtol=1e-12
    )


def test_every_fast_formation_driver_routes_through_the_one_marcher():
    """No driver reaches the fast-formation branch by another path.

    The guard above only helps for callers that use the marcher, so
    this pins the other half: the sub-fluid continuation is called in
    exactly one place, inside
    :func:`_march_fast_flexural_branch`. A driver that grew its own
    copy would slip the requirement, and that is the shape of the
    original bug.
    """
    import pathlib

    root = pathlib.Path(_n1_isotropic_module().__file__).parent
    callers = []
    for path in sorted(root.glob("*.py")):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if "_extend_below_fluid(" in line and not line.lstrip().startswith(
                ("def ", "#", '"', "*")
            ):
                callers.append((path.name, i, line.strip()))
    assert len(callers) == 1, callers
    assert callers[0][0] == "_n1_isotropic.py", callers


def _n1_isotropic_module():
    from fwap.cylindrical_solver import _n1_isotropic

    return _n1_isotropic


def test_the_vti_determinants_accept_a_complex_with_zero_imaginary_part():
    """``complex(kz, 0.0)`` is a real number and is treated as one.

    The isotropic drivers call their complex determinant as
    ``_modal_determinant_n1_complex(complex(kz, 0.0), ...)``. Handed the
    same thing the VTI determinants used to raise
    ``'<' not supported between instances of 'complex' and 'float'``
    from inside a private wavenumber helper, which made the two
    families gratuitously incompatible over a value that carries no
    imaginary part at all.

    Bit-identical, not merely close: the coercion happens before any
    arithmetic, so the float path is untouched.
    """
    from fwap.cylindrical_solver._bessel import _radial_wavenumbers_vti
    from fwap.cylindrical_solver._vti import _modal_determinant_n1_vti_complex

    medium = _LQ_SINHA
    stiffness = _iso_stiffness_from(medium)
    shale = dict(**_green_river_shale_stiffness(), vf=1500.0, rho_f=1000.0, a=0.10)
    dets = (
        _modal_determinant_n0_vti,
        _modal_determinant_n1_vti,
        _modal_determinant_n1_vti_complex,
    )
    checked = 0
    for kwargs in (stiffness, shale):
        for freq in (3000.0, 7000.0, 11000.0):
            omega = 2.0 * np.pi * freq
            for c in (1200.0, 1450.0, 1700.0, 2100.0):
                kz = omega / c
                for det in dets:
                    got = det(kz, omega, **kwargs)
                    same = det(complex(kz, 0.0), omega, **kwargs)
                    if np.isnan(np.real(got)):
                        assert np.isnan(np.real(same)), (det.__name__, freq, c)
                    else:
                        assert got == same, (det.__name__, freq, c, got, same)
                    checked += 1
            # and the shared wavenumber helper underneath them
            real = _radial_wavenumbers_vti(
                omega / 1700.0,
                omega,
                **{k: v for k, v in kwargs.items() if k not in ("vf", "rho_f", "a")},
            )
            widened = _radial_wavenumbers_vti(
                complex(omega / 1700.0, 0.0),
                omega,
                **{k: v for k, v in kwargs.items() if k not in ("vf", "rho_f", "a")},
            )
            assert real == widened, (freq, real, widened)
    assert checked >= 60, checked


def test_a_genuinely_complex_kz_is_refused_with_an_explanation():
    """The refusal narrowed when A.11 phase 4 landed, and stayed honest.

    Written for #134, when the VTI stack was real-``k_z`` by
    construction and every path refused a complex one. Phase 4 gave the
    ``n = 1`` columns radiating branches, so ``_modal_determinant_
    n1_vti_complex`` now answers -- but only when the caller says which
    waves radiate. With every wave bound, a complex ``k_z`` describes a
    field decaying in ``r`` while growing along ``z``, which is not a
    mode, and that is still refused.

    The ``n = 0`` and bound ``n = 1`` determinants keep the original
    contract: they reduce over ``M.real`` and have no radiating branch,
    so a complex ``k_z`` has no reading there at all.

    Either way the message names what is missing rather than surfacing
    a comparison failure from a helper the caller never invoked.
    """
    from fwap.cylindrical_solver._bessel import _radial_wavenumbers_vti
    from fwap.cylindrical_solver._vti import _modal_determinant_n1_vti_complex

    medium = _LQ_SINHA
    stiffness = _iso_stiffness_from(medium)
    omega = 2.0 * np.pi * 6000.0
    kz = complex(omega / 1700.0, 0.05)

    for det in (_modal_determinant_n0_vti, _modal_determinant_n1_vti):
        with pytest.raises(NotImplementedError, match="k_z must be real"):
            det(kz, omega, **stiffness)

    # The complex determinant refuses only for want of the flags ...
    with pytest.raises(NotImplementedError, match="radiating"):
        _modal_determinant_n1_vti_complex(kz, omega, **stiffness)

    # ... and answers once they are given.
    value = _modal_determinant_n1_vti_complex(
        kz, omega, **stiffness, radiating=(False, True, True)
    )
    assert np.isfinite(complex(value))
    assert abs(complex(value)) > 0.0

    with pytest.raises(NotImplementedError, match="leaky_p"):
        _radial_wavenumbers_vti(
            kz,
            omega,
            **{k: v for k, v in stiffness.items() if k not in ("vf", "rho_f", "a")},
        )
