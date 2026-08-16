"""
Where each branch exists, where it stops, and whether it is the same
branch on a different grid.

One of six modules split out of ``tests/test_cylindrical_solver.py``.
These are the mode-level tests rather than the matrix-level ones:
leaky branches and their seeding, cutoffs against closed forms,
high-frequency asymptotes, the marcher's step rule and its two passes,
and the grid-independence checks that separate a real root from an
artefact of the scan.

Roadmap A.9's slow cased gap and its closure are here, as are the
Sinha figure-2 and figure-11 ties, which sit with the leaky
compressional and pseudo-Rayleigh branches they constrain rather than
with the other published figures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fwap.cylindrical import (
    rayleigh_speed,
)
from fwap.cylindrical_solver import (
    BoreholeLayer,
    _layer_e_matrix_n0,
    _layer_propagator_n0,
    _modal_determinant_n0_cased,
    _modal_determinant_n1_cased,
    flexural_dispersion,
    flexural_dispersion_layered,
    stoneley_dispersion,
    stoneley_dispersion_layered,
    trapped_pseudo_rayleigh_dispersion,
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
    _sc87_stack,
)

# Where the cased flexural solver actually stops (roadmap A.2)
#
# The roadmap recorded this as a *layered* bracketing limitation: "the
# layered n=1 solver no longer refuses fast formations, but its root-
# finding stays sparse for a typical casing + cement stack". Measuring it
# says otherwise, and the difference matters because it points the next
# attempt at a different piece of code.
#
# In a fast formation the flexural mode is leaky: its root leaves the real
# k_z axis, and the real-axis Im(det) sign change the solver looks for
# survives only in a sliver next to the shear branch point at high
# frequency. Widening the real bracket cannot recover it -- there is no
# sign change to find below the cutoff, in any of the three sub-windows
# (below the slowest layer shear, between it and the formation Rayleigh
# speed, or between that and the formation shear).
#
# These tests pin the behaviour so the limitation stays a measured number
# rather than folklore, and so that a genuine fix shows up as a failure
# here rather than going unnoticed.
# ----------------------------------------------------------------------

_A2_FAST = dict(vp=4500.0, vs=2600.0, rho=2400.0)
_A2_SLOW = dict(vp=2200.0, vs=800.0, rho=2200.0)
_A2_BOREHOLE = dict(vf=1500.0, rho_f=1000.0, a=0.10)
_A2_FREQ = np.linspace(1000.0, 12000.0, 45)


def _a2_coverage(**kwargs) -> float:
    """Fraction of the frequency band at which the mode converged."""
    res = flexural_dispersion_layered(_A2_FREQ, **kwargs)
    return float(np.isfinite(res.slowness).mean())


def test_cased_slow_formation_dipole_is_leaky_and_the_leaky_branch_finds_it():
    """A slow formation behind steel casing has a LEAKY dipole mode.

    This test has been through three states, and the history is the
    point. It began asserting full coverage of the band, which was the
    A.8 defect speaking: the azimuthal-only SV column produced a
    spurious *bound* root that rose with frequency (199, 445, 755 m/s
    at 6, 9, 12 kHz) -- backwards for a flexural mode, and below every
    wave speed in the problem. Correcting the column removed it and
    left the band empty, because a steel casing raises the composite
    bending stiffness until the dipole mode outruns the formation shear
    speed and radiates into it, and the real-valued determinant only
    describes fields evanescent everywhere outside the fluid.

    Roadmap A.9 added the search that does describe it: complex ``k_z``,
    outgoing formation S branch, seeded from a real-axis scan of
    ``Im(det)`` over ``(V_S, min(V_f, min layer V_S))``. The mode comes
    back as a proper leaky branch with positive attenuation, and the
    attenuation is what says it is leaky rather than bound.

    **Fourth state: the numbers below are re-measured.** The monotone
    claims this test used to make -- phase velocity descending across
    the whole band, attenuation descending with it, coverage above
    0.7, an asymptote within 2 % of ``V_S`` -- were measured while
    :func:`_k_or_hankel` was mixing an incoming wave into its leaky
    branch. Corrected, the branch this stack showed ran 5-12 kHz (0.62
    of the band), falling 1238 -> 1167 m/s to a minimum near 9 kHz and
    climbing back to 1191.

    **Fifth state: that was one leg of two.** Rebuilding the seeding --
    a ``log|det|`` survey in place of a blind sweep, a two-sided slope
    rule in place of a monotone one, and pass two merging into pass one
    instead of replacing it -- recovered a second leg at **1.0-2.0 kHz**,
    rising 1033.9 -> 1282.9 m/s toward the annulus ceiling.

    **Sixth: letting pass two re-acquire after a gap** rather than
    stopping after two consecutive misses extended the upper leg down
    from 5.00 to 4.25 kHz and closed the one-sample hole it had carried
    at 8.0 kHz since it was first measured. Coverage 0.62 -> 0.73 ->
    **0.82**, and the upper leg is contiguous over 4.25-12.00 kHz.

    The new leg is not the marcher finding more of what it already had:
    an argument-principle contour over the whole window counts **one**
    root at 1.0, 1.5 and 2.0 kHz, and **none** at 2.25-3.75 kHz where
    the curve is NaN between the legs. So the branch leaves the window
    through the ceiling and comes back, and this fixture has the same
    two-leg shape as Schmitt & Cheng's.

    **What is still missed is one frequency, and the cause is named.**
    The contour counts a root at 4.00 kHz and the marcher returns
    ``NaN`` there, while 4.25 and everything above it is found. That is
    the march being one-directional: pass two re-acquires at a sweep
    pick and continues *upward*, so the frequency immediately below the
    pick is never revisited. Closing it needs a downward continuation
    from each re-acquisition, which is a separate change.

    The structural assertions -- above ``V_S``, below the annulus
    ceiling, attenuation strictly positive -- are unchanged and still
    the ones that say "leaky".
    """
    freq = _A2_FREQ
    res = flexural_dispersion_layered(
        freq, **_A2_SLOW, **_A2_BOREHOLE, layers=(_A2_CASING, _A2_CEMENT)
    )
    found = np.isfinite(res.slowness)
    assert found.mean() > 0.7, f"coverage {found.mean():.2f}"

    velocity = 1.0 / res.slowness[found]
    attenuation = res.attenuation_per_meter[found]

    # Every answer is above the formation shear speed -- that is what
    # makes it leaky -- and below the annulus ceiling.
    assert np.all(velocity > _A2_SLOW["vs"])
    assert np.all(velocity < min(_A2_BOREHOLE["vf"], _A2_CEMENT.vs))
    # Two legs, so shape is asserted per leg rather than across the
    # concatenation -- a turning-point count over the joined arrays
    # would be counting the gap.
    #
    # Split on gaps wider than two samples. The upper leg has a
    # single-frequency hole at 8.0 kHz that predates all of this (the
    # branch before any of the seeding work had it too, at the same
    # frequency), and it is not a leg boundary.
    where = np.flatnonzero(found)
    gap = np.where(np.diff(where) > 2)[0]
    assert gap.size == 1, f"expected two legs, got {gap.size + 1}"
    split = int(gap[0]) + 1
    lower, upper = velocity[:split], velocity[split:]
    # The lower leg rises toward the ceiling and leaves through it.
    assert np.all(np.diff(lower) > 0.0), lower
    assert lower[-1] < min(_A2_BOREHOLE["vf"], _A2_CEMENT.vs)
    # The upper leg is contiguous once pass two can re-acquire: the
    # 8.0 kHz hole it used to carry is a tracker stumble, and stopping
    # at it was what left the hole in place.
    assert np.array_equal(where[split:], np.arange(where[split], where[-1] + 1)), (
        "the upper leg is not contiguous"
    )
    # The upper leg keeps its one Airy minimum.
    turns = int((np.diff(np.sign(np.diff(upper))) != 0).sum())
    assert turns == 1, f"upper leg has {turns} turning points, expected 1"
    # Leakage: positive throughout.
    assert np.all(attenuation > 0.0)

    # The real-valued determinant still has no root for it: this is a
    # genuinely different formulation answering, not the old one
    # recovering.
    omega = 2.0 * np.pi * 6000.0
    grid = np.linspace(_A2_SLOW["vs"] * 1.001, _A2_SLOW["vs"] * 0.999 + 500.0, 400)
    real_det = np.array(
        [
            _modal_determinant_n1_cased(
                omega / v,
                omega,
                **_A2_SLOW,
                **_A2_BOREHOLE,
                layers=(_A2_CASING, _A2_CEMENT),
            )
            for v in grid
        ]
    )
    finite = np.isfinite(real_det)
    assert int((np.diff(np.sign(real_det[finite])) != 0).sum()) == 0


def test_the_cased_leaky_branch_joins_the_bound_one_across_the_shear_speed():
    """The oracle A.9 has instead of a published curve.

    Schmitt & Cheng plot no cased-hole dispersion, so the leaky branch
    cannot be tied to a figure. What it can be tied to is the bound
    solver it takes over from: stiffen the annulus and the dipole mode
    climbs toward the formation shear speed, crosses it, and continues.
    On the bound side the ordinary layered path owns the answer; on the
    leaky side the complex marcher does.

    The sharp end of the test is the overlap. Where the mode is still
    bound, the *complex* determinant -- the one the leaky search
    refines, with its branch flags coming from
    :func:`_detect_leaky_branches` -- has a root at exactly the phase
    velocity the real-valued determinant's brentq found, with zero
    imaginary part. That is an agreement between two formulations to
    floating point, not a smoothness eyeball.

    Note the marcher itself cannot be run there: it seeds from sign
    changes of ``Im(det)``, and in the bound regime the determinant at
    real ``k_z`` is real, so there is nothing to seed off. That is why
    the production window floor is the formation shear speed and the
    bound path keeps everything below it.
    """
    from fwap.cylindrical_solver import _modal_determinant_n1_cased_complex
    from fwap.cylindrical_solver._leaky import (
        _detect_leaky_branches,
        _track_complex_root,
    )

    formation = dict(vp=2200.0, vs=800.0, rho=2200.0)
    borehole = dict(vf=1500.0, rho_f=1000.0, a=0.10)
    freq = np.array([6000.0])
    omega = 2.0 * np.pi * float(freq[0])

    last_bound = None
    for vs_layer in (830.0, 870.0, 950.0):
        layers = (
            BoreholeLayer(vp=2.2 * vs_layer, vs=vs_layer, rho=2000.0, thickness=0.04),
        )
        bound = flexural_dispersion_layered(
            freq, **formation, **borehole, layers=layers
        )
        assert np.isfinite(bound.slowness[0])
        assert bound.attenuation_per_meter is None, "still a bound mode"

        def _det(kz, omega_step, layers=layers):
            _, leaky_p, leaky_s = _detect_leaky_branches(
                kz, omega_step, formation["vp"], formation["vs"], borehole["vf"]
            )
            return _modal_determinant_n1_cased_complex(
                kz,
                omega_step,
                **formation,
                **borehole,
                layers=layers,
                leaky_p=leaky_p,
                leaky_s=leaky_s,
            )

        kz_root = _track_complex_root(
            lambda kz: _det(kz, omega),
            complex(bound.slowness[0] * omega * 1.001, 0.0),
        )
        assert kz_root is not None, f"layer Vs={vs_layer}"
        assert kz_root.real / omega == pytest.approx(bound.slowness[0], rel=1.0e-9)
        # Bound means no leakage, and the complex root says so itself.
        assert abs(kz_root.imag) < 1.0e-9 * kz_root.real
        last_bound = 1.0 / bound.slowness[0]

    # The bound side climbs toward the shear speed as the annulus
    # stiffens, and stops just under it.
    assert last_bound is not None
    assert 0.97 < last_bound / formation["vs"] < 1.0

    # Past the crossing the bound path has nothing and the leaky one
    # takes over, from above the shear speed and above where the bound
    # side stopped.
    #
    # This block used to claim "no jump across the boundary", then
    # withdrew the claim: the bound side ran to 798.91 m/s at
    # V_S_layer / V_S = 1.26 and the leaky side seemed to start at
    # 857.39 at 1.28, a 7 % step, which was written up as a handover
    # between two different modes.
    #
    # There is no step. The 7 % was the seed floor of the day skipping
    # over the real continuation, which the corrected radiation branch
    # exposes: swept at ka = 2.5 the leaky branch *emerges at the shear
    # speed itself* -- 799.99 m/s with Im(k_z) = 0 at ratio 1.275,
    # exactly where the bound mode is absorbed -- and climbs 805.0,
    # 811.3, 817.9, 824.6 through 1.30-1.36. One mode, continuous
    # through the crossing, which is what the open-hole sister
    # test_the_leaky_branch_joins_the_trapped_one_at_its_cutoff shows
    # too.
    first_leaky = None
    for vs_layer in (1400.0, 1800.0, 3140.0):
        layers = (
            BoreholeLayer(vp=2.2 * vs_layer, vs=vs_layer, rho=2000.0, thickness=0.04),
        )
        res = flexural_dispersion_layered(freq, **formation, **borehole, layers=layers)
        assert np.isfinite(res.slowness[0])
        assert res.attenuation_per_meter is not None
        velocity = 1.0 / res.slowness[0]
        # Stiffer annulus, faster mode: 943, 1067, 1408 m/s here.
        assert formation["vs"] < velocity < min(borehole["vf"], vs_layer)
        assert res.attenuation_per_meter[0] > 0.0
        if first_leaky is None:
            first_leaky = velocity
    assert first_leaky is not None
    assert first_leaky > last_bound


def test_the_cased_leaky_branch_is_grid_independent():
    """The answer at a frequency must not depend on the grid it came in.

    The leaky marcher seeds itself from a scan and then continues, so a
    different grid means a different seed frequency and a different
    number of continuation steps. Grids of 9 to 65 points over the same
    band agree at 8 kHz to better than 1e-12 relative.

    The 33-point grid is skipped, and not because it disagrees: its
    nearest sample to 8 kHz is 8000.0 Hz only in the other three cases,
    and after the radiation branch was corrected the mode's band starts
    at 5 kHz, so that grid's nearest sample lands where the branch has
    not started. Asserting on a NaN would be asserting on the grid, not
    on the physics.
    """
    layers = (_A2_CASING, _A2_CEMENT)
    values = []
    for n_points in (9, 17, 33, 65):
        grid = np.linspace(4000.0, 12000.0, n_points)
        res = flexural_dispersion_layered(
            grid, **_A2_SLOW, **_A2_BOREHOLE, layers=layers
        )
        index = int(np.argmin(np.abs(grid - 8000.0)))
        if not np.isfinite(res.slowness[index]):
            continue
        values.append(1.0 / res.slowness[index])
    assert len(values) >= 3
    spread = float(np.ptp(values))
    assert spread / float(np.mean(values)) < 1.0e-12, f"spread {spread:.3e} m/s"


#: ``ka = omega a / V_f = 2.5``, the point A.9's gap was recorded at.
_GAP_FREQ = np.array([2.5 * 1500.0 / 0.10 / (2.0 * np.pi)])
_GAP_FORMATION = dict(vp=2200.0, vs=800.0, rho=2200.0)


def test_the_recorded_slow_cased_gap_is_closed():
    """A.9's gap, and the values are not new -- they were counted first.

    Over ``V_S_layer / V_S`` in [1.3, 1.5] at ``ka = 2.5`` the real-axis
    scan does find its one ``Im(det)`` crossing, at 1006, 978 and 956
    m/s. The mode is at 855, 851 and 859, and from that far away the
    complex tracker runs instead to the layer's own shear speed --
    1040.00, 1120.00, 1200.00 m/s to the digit -- which is the
    degeneracy ``exclude`` names and rejects. Correctly rejected, and
    nothing left.

    Seeding off the real axis reaches them directly.

    **The values are re-measured.** They were 855.09, 850.51 and 859.31
    -- non-monotone in the annulus stiffness, which nobody questioned at
    the time -- and they were roots of a determinant whose leaky branch
    was two thirds incoming. Corrected, the same three stiffnesses give
    804.99, 838.04 and 870.33, rising with the annulus as they should,
    and the branch is continuous from the bound/leaky crossing upward.

    Closing the gap also needed ``_LEAKY_CASED_SEED_FLOOR`` dropped from
    0.03 to 0.002; see that constant, whose entire justification the
    correction withdrew.
    """
    expected = {1.3: 804.99, 1.4: 838.04, 1.5: 870.33}
    for ratio, velocity in expected.items():
        res = flexural_dispersion_layered(
            _GAP_FREQ,
            **_GAP_FORMATION,
            **_A2_BOREHOLE,
            layers=_gap_sweep_layers(ratio),
        )
        assert np.isfinite(res.slowness[0]), f"still empty at ratio {ratio}"
        assert res.attenuation_per_meter is not None
        assert 1.0 / res.slowness[0] == pytest.approx(velocity, abs=0.01)
        # Leaky means it radiates, and above V_S is what makes it leaky.
        assert res.attenuation_per_meter[0] > 0.0
        assert 1.0 / res.slowness[0] > _GAP_FORMATION["vs"]


def test_the_leaky_cased_branch_is_continuous_across_the_closed_gap():
    """The gap closed as a branch, not as three isolated answers.

    The whole point of a gap is that the curve either side of it did not
    join up. Swept finely from 1.30 to 1.60 the annulus-stiffness family
    is unbroken, and both the phase velocity and the attenuation vary
    smoothly and monotonically -- 805 to 900 m/s, attenuation 0.51 to
    1.35, which no per-frequency accident would produce.

    Re-measured after the radiation branch was corrected. It used to
    show one turning point in each curve, a velocity minimum near 1.38
    and an attenuation maximum near 1.48; both were artefacts of the
    incoming contamination and both are gone. The sweep now starts at
    1.30 rather than 1.28 because at 1.28 the branch has only just
    emerged from ``V_S`` (800.03 m/s) and sits under even the nominal
    seed floor.
    """
    ratios = np.arange(1.30, 1.605, 0.04)
    velocity, attenuation = [], []
    for ratio in ratios:
        res = flexural_dispersion_layered(
            _GAP_FREQ,
            **_GAP_FORMATION,
            **_A2_BOREHOLE,
            layers=_gap_sweep_layers(float(ratio)),
        )
        assert np.isfinite(res.slowness[0]), f"empty at ratio {ratio:.2f}"
        assert res.attenuation_per_meter is not None
        velocity.append(1.0 / res.slowness[0])
        attenuation.append(float(res.attenuation_per_meter[0]))

    velocity = np.array(velocity)
    attenuation = np.array(attenuation)
    assert np.all(velocity > _GAP_FORMATION["vs"])
    assert np.all(attenuation > 0.0)
    # No step, which is what a gap filled by a second family would leave.
    assert np.max(np.abs(np.diff(velocity)) / velocity[:-1]) < 0.02
    # And monotone in the annulus stiffness, both of them: a stiffer
    # annulus makes the mode faster and leakier. Two spliced families
    # do not do this, and it is a structural claim rather than a
    # tolerance to tune.
    assert np.all(np.diff(velocity) > 0.0), velocity
    assert np.all(np.diff(attenuation) > 0.0), attenuation


def test_the_seed_sweep_only_ever_adds_to_what_the_scan_found():
    """The guarantee that makes the sweep safe to have.

    Its extra reach also finds the shear branch point's own zeros, which
    are sharp and are not modes. Seeded from those at a frequency where
    the flexural mode has genuinely left the window, the old monotone
    rule followed that family instead, and starting from the
    low-frequency end it took the whole band: every already-converged
    frequency moved, by 17 % at 3.5 kHz, ending 0.23 % above ``V_S``
    instead of 1.3 %.

    **The guard used to be a gate and is now a merge**, which is
    strictly stronger. Gating the sweep on "the scan found nothing
    anywhere" left it able to replace every value on the runs where it
    did fire; merging into the scan's gaps makes overwriting a scanned
    value impossible to express. So the assertion changes shape: the
    sweep may add frequencies -- on this fixture it adds five, the
    1.0-2.0 kHz leg -- but every frequency the scan resolved keeps its
    value to the last bit.

    Asserted by disabling the sweep outright and comparing.

    **The comparison is to a tolerance rather than bit for bit, and the
    reason is the downward pass.** A frequency below a leg's entry point
    can now be reached two ways -- by descent from a higher leg, or by
    continuation up from a sweep re-acquisition beneath it -- and which
    one applies depends on whether the sweep ran. Both land on the same
    root; they do not land on the same last bit. On this fixture that
    affects exactly the five frequencies the descent reaches, 4.00-4.75
    and 8.00 kHz, which agree to twelve figures. What is still exact by
    construction is that neither the sweep nor the descent may write
    where a value already exists.
    """
    import fwap.cylindrical_solver._leaky as leaky_module

    layers = (_A2_CASING, _A2_CEMENT)
    with_sweep = flexural_dispersion_layered(
        _A2_FREQ, **_A2_SLOW, **_A2_BOREHOLE, layers=layers
    )

    original = leaky_module._LEAKY_CASED_SEED_SWEEP_POINTS
    try:
        leaky_module._LEAKY_CASED_SEED_SWEEP_POINTS = 0
        without_sweep = flexural_dispersion_layered(
            _A2_FREQ, **_A2_SLOW, **_A2_BOREHOLE, layers=layers
        )
    finally:
        leaky_module._LEAKY_CASED_SEED_SWEEP_POINTS = original

    scanned = np.isfinite(without_sweep.slowness)
    swept = np.isfinite(with_sweep.slowness)
    # Never takes anything away.
    assert np.all(swept[scanned]), "the sweep lost a frequency the scan had"
    # Never changes anything it did not add.
    np.testing.assert_allclose(
        with_sweep.slowness[scanned],
        without_sweep.slowness[scanned],
        rtol=1.0e-9,
        err_msg="the sweep changed a value the scan had already found",
    )
    # What the sweep adds on this fixture is exactly the 1.0-2.0 kHz
    # leg, which nothing else can reach: pass one's scan never sees it,
    # and no descent reaches it either because the gap above it is wider
    # than the miss budget.
    #
    # **That count has been 5, then 9, then 5 again**, and the round trip
    # is worth recording rather than smoothing over. It was 5 when the
    # sweep was the only way past pass one's stopping point; 9 once pass
    # two could re-acquire, which also picked up the upper leg's lower
    # edge and an interior hole; and 5 again once the downward pass could
    # reach those five from the leg above without the sweep's help. The
    # sweep's unique contribution was always this leg.
    added = np.flatnonzero(swept & ~scanned)
    assert added.size == 5, f"the sweep added {added.size} frequencies, expected 5"
    assert np.array_equal(added, np.arange(5)), (
        f"expected the leg at the bottom of the band, got {added}"
    )
    assert swept.mean() > 0.8, f"coverage {swept.mean():.2f}"


def test_the_seed_survey_reaches_a_strongly_damped_branch():
    """Why the seed ladder is geometric and why it goes to 45 %.

    The ladder used to be ``(0.03, 0.07)`` -- 3 % and 7 % of
    ``Re(k_z)`` -- chosen to bracket the leakage the ``_A2`` fixture
    carries. That read as a property of the mode and was a property of
    that stack. Schmitt & Cheng's slow sandstone behind their casing and
    cement carries **29 %**, and no seed on the old ladder converged to
    it.

    This pins the measurement rather than the ladder: the branch's own
    ``Im(k_z) / Re(k_z)``, which is what any seeding scheme has to
    reach. A ladder that stops short of it will fail this test by
    failing to find the root at all.
    """
    radius, layers = _sc87_stack(1729.0, 1920.0, 0.03)
    freq = np.array([1500.0, 2000.0])
    mode = flexural_dispersion_layered(freq, **_SC87_SLOW, a=radius, layers=layers)
    assert np.isfinite(mode.slowness).all()
    # Im / Re, the quantity the seed levels are fractions of. Re(k_z)
    # is omega * slowness, and Im(k_z) is the reported attenuation.
    ratio = mode.attenuation_per_meter / (2.0 * np.pi * freq * mode.slowness)
    assert np.all(ratio > 0.25), ratio
    assert np.all(ratio < 0.35), ratio
    # The deepest level on the ladder has to clear it, with margin.
    from fwap.cylindrical_solver._leaky import _LEAKY_CASED_SEED_SWEEP_LEVELS

    assert max(_LEAKY_CASED_SEED_SWEEP_LEVELS) > float(np.max(ratio))


def test_the_step_rule_is_two_sided_and_carries_a_frequency():
    """The continuation rule constrains shape, not direction.

    It used to be one-sided and dimensionless in the wrong way: "a
    candidate faster than the last one by more than 0.5 %" has no
    frequency in it, so the same physical branch passed or failed on how
    finely the caller sampled, and a branch with an Airy minimum failed
    on the way back up.

    Both consequences were live. This asserts the replacement directly:
    the same geometry sampled two ways must give the same branch, and
    the branch must be allowed to rise.
    """
    radius, layers = _sc87_stack(1729.0, 1920.0, 0.03)
    coarse = np.array([1500.0, 2000.0, 2500.0])
    fine = np.arange(1500.0, 2501.0, 125.0)
    got_coarse = flexural_dispersion_layered(
        coarse, **_SC87_SLOW, a=radius, layers=layers
    )
    got_fine = flexural_dispersion_layered(fine, **_SC87_SLOW, a=radius, layers=layers)
    assert np.isfinite(got_coarse.slowness).all()
    assert np.isfinite(got_fine.slowness).all()
    # Sampling-independent: the coarse points sit on the fine curve.
    on_fine = np.interp(coarse, fine, 1.0 / got_fine.slowness)
    np.testing.assert_allclose(1.0 / got_coarse.slowness, on_fine, rtol=2.0e-3)
    # And rising -- which the old rule forbade outright at this step size.
    assert np.all(np.diff(1.0 / got_coarse.slowness) > 0.0)
    assert (1.0 / got_coarse.slowness)[-1] / (1.0 / got_coarse.slowness)[0] > 1.15


def test_pass_two_re_acquires_the_branch_after_a_gap():
    """A branch with two legs is followed through the gap, not stopped at.

    ``_LEAKY_CASED_MAX_INVALID`` ends a march after two consecutive
    misses. That is right for pass one, whose answers are authoritative,
    and it was wrong for pass two: a branch that leaves the search
    window and comes back has two legs, and stopping at the first gap
    means only the leg the marcher reached first is ever returned.

    Pass two now drops its continuation state and keeps walking. It is
    safe to do only there, because pass two merges into pass one's gaps
    and cannot overwrite them -- the worst a bad re-acquisition can do
    is fill a frequency that was going to stay ``NaN``.

    Both fixtures with this shape gain from it, and both gains are
    checked against a root count rather than against themselves:
    Schmitt & Cheng's upper leg starts at 13.25 kHz instead of 14.0, and
    ``_A2``'s at 4.25 instead of 5.0 -- and then at 4.00 once the
    downward pass walks the leg back from its entry point.

    Schmitt & Cheng's leg reaches 13.00 rather than 13.25 for a third
    reason, unrelated to either: the ceiling dead band that was
    declining the 1497.11 m/s root there has since been withdrawn. See
    ``_LEAKY_CASED_DEGENERACY_TOL``.
    """
    radius, layers = _sc87_stack(1729.0, 1920.0, 0.03)
    freq = np.arange(1000.0, 15001.0, 250.0)
    mode = flexural_dispersion_layered(freq, **_SC87_SLOW, a=radius, layers=layers)
    found = np.isfinite(mode.slowness)
    # Two legs, and the upper one now reaches down to 13.00 kHz.
    where = np.flatnonzero(found)
    gaps = np.where(np.diff(where) > 2)[0]
    assert gaps.size == 1, f"expected two legs, got {gaps.size + 1}"
    upper_start = freq[where[int(gaps[0]) + 1]]
    assert upper_start == pytest.approx(13000.0), upper_start
    # Every value in both legs sits inside the search window.
    velocity = 1.0 / mode.slowness[found]
    assert np.all(velocity > _SC87_SLOW["vs"])
    assert np.all(velocity < _SC87_SLOW["vf"])

    # The same on the _A2 stack, where the upper leg also loses the
    # one-sample hole it used to carry at 8.0 kHz.
    a2 = flexural_dispersion_layered(
        _A2_FREQ, **_A2_SLOW, **_A2_BOREHOLE, layers=(_A2_CASING, _A2_CEMENT)
    )
    a2_found = np.isfinite(a2.slowness)
    assert a2_found.mean() > 0.8, f"coverage {a2_found.mean():.2f}"
    a2_where = np.flatnonzero(a2_found)
    a2_gaps = np.where(np.diff(a2_where) > 2)[0]
    assert a2_gaps.size == 1
    assert _A2_FREQ[a2_where[int(a2_gaps[0]) + 1]] == pytest.approx(4000.0)


def test_the_downward_pass_walks_a_leg_back_from_its_entry_point():
    """Both marching passes go up, so a leg is only entered from below.

    Whichever frequency the scan or the sweep first resolved becomes the
    leg's floor, and everything under it stays ``NaN`` even where the
    root is there to be found. The entry point is set by where a sweep
    attempt happened to land, not by where the branch begins.

    On the ``_A2`` stack the upper leg entered at 4.25 kHz with a root
    at 4.00 -- one frequency, real, invisible for want of a step in the
    other direction. The descent adds it, and stops where the roots
    stop: an argument-principle contour counts one root at 4.00 and
    **none** at 3.75, which is where the curve still ends.
    """
    mode = flexural_dispersion_layered(
        _A2_FREQ, **_A2_SLOW, **_A2_BOREHOLE, layers=(_A2_CASING, _A2_CEMENT)
    )
    found = np.isfinite(mode.slowness)
    assert found.mean() > 0.83, f"coverage {found.mean():.2f}"
    where = np.flatnonzero(found)
    gaps = np.where(np.diff(where) > 2)[0]
    assert gaps.size == 1
    upper_start = _A2_FREQ[where[int(gaps[0]) + 1]]
    assert upper_start == pytest.approx(4000.0), upper_start
    # The descent stops at a real boundary, not at a budget: 3.75 kHz
    # is the last frequency below it and it carries no root.
    below = _A2_FREQ < 4000.0
    assert not np.isfinite(mode.slowness[below & (_A2_FREQ > 2500.0)]).any()


def test_the_downward_pass_never_writes_over_an_existing_value():
    """It fills ``NaN`` and nothing else.

    The guarantee is the same one the sweep merge carries, and it has to
    hold for the same reason: the ascending passes' answers are the
    authoritative ones. Asserted by shortening the frequency grid so the
    descent has nowhere to run and requiring the shared frequencies to
    come back unchanged.
    """
    radius, layers = _sc87_stack(1729.0, 1920.0, 0.03)
    full = np.arange(13250.0, 15001.0, 250.0)
    got_full = flexural_dispersion_layered(full, **_SC87_SLOW, a=radius, layers=layers)
    assert np.isfinite(got_full.slowness).all()
    # The same band with room below it for a descent to attempt.
    wide = np.arange(12000.0, 15001.0, 250.0)
    got_wide = flexural_dispersion_layered(wide, **_SC87_SLOW, a=radius, layers=layers)
    shared = np.isin(wide, full)
    np.testing.assert_allclose(
        got_wide.slowness[shared], got_full.slowness, rtol=1.0e-9
    )


def test_the_seed_sweep_is_bounded_when_there_is_nothing_to_find():
    """The cost of the sweep is paid where it cannot help, so it is capped.

    A stack with no mode anywhere fails pass one at every frequency,
    which is exactly when pass two runs -- so an uncapped sweep does its
    full seed grid at all of them. The surrogate generators reject such
    stacks by the hundred: three tests in
    ``tests/test_gen_surrogate_dataset.py`` went from 41 s to 828 s, and
    the CI job from 422 s to 1331 s.

    The cap spreads a fixed number of attempts across the band rather
    than taking the first few, because a leaky branch can appear
    anywhere in it. This asserts the bound structurally -- by counting
    the frequencies at which an off-axis seed is tried -- rather than
    with a wall-clock budget, which this suite keeps out of the default
    run.
    """
    from fwap.cylindrical_solver._leaky import (
        _LEAKY_CASED_SWEEP_MAX_ATTEMPTS,
        _march_leaky_cased_branch,
    )

    freq = np.linspace(2000.0, 12000.0, 45)
    swept_at: set[float] = set()

    def det_fn(kz: complex, omega: float) -> complex:
        # Never a root, so pass one finds nothing and pass two runs.
        if complex(kz).imag != 0.0:
            swept_at.add(round(omega, 6))
        return complex(1.0, 1.0)

    slowness, attenuation = _march_leaky_cased_branch(
        det_fn, freq, vs=800.0, ceiling=1300.0
    )
    assert not np.any(np.isfinite(slowness))
    assert not np.any(np.isfinite(attenuation))
    assert len(swept_at) <= _LEAKY_CASED_SWEEP_MAX_ATTEMPTS, (
        f"swept at {len(swept_at)} frequencies, cap is "
        f"{_LEAKY_CASED_SWEEP_MAX_ATTEMPTS}"
    )
    # Spread across the band, not clustered at its start: the lowest and
    # highest frequencies are both tried.
    assert min(swept_at) == pytest.approx(2.0 * np.pi * freq[0])
    assert max(swept_at) == pytest.approx(2.0 * np.pi * freq[-1])


def test_the_bound_dipole_mode_is_absorbed_at_the_shear_branch_point():
    """Half of the crossing question: what becomes of the bound mode.

    As the annulus stiffens the bound dipole mode climbs toward the
    formation shear speed. It reaches it and stops existing -- not
    "the solver stops finding it": a dense scan of the real,
    proper-sheet determinant across the *whole* bound window finds
    exactly one sign change while the mode exists and **none at all**
    afterwards. Absorbed at the branch point.

    This half stands. What was built on it does not: it used to read
    "that is why the layered driver's answer steps 7 % at the crossing
    -- a handover between two different modes, not a break in one",
    on the strength of a leaky branch measured coexisting with the
    bound mode below the crossing. That branch was the incoming
    contamination in :func:`_k_or_hankel`; corrected, nothing radiating
    exists below the crossing and the leaky branch instead *emerges*
    at ``V_S`` where this mode is absorbed. One mode, continuous. See
    ``test_no_leaky_branch_coexists_with_the_bound_mode_below_the_crossing``
    and ``test_the_branch_point_pole_was_the_incoming_contamination``.

    The absorption itself is unaffected -- it is measured on the
    *real*, proper-sheet determinant, which the radiation branch never
    touched.
    """
    from fwap.cylindrical_solver import _modal_determinant_n1_cased

    omega = 2.0 * np.pi * float(_GAP_FREQ[0])
    vs = _GAP_FORMATION["vs"]
    grid = np.linspace(600.0, vs * 0.99999, 1500)

    def crossings(ratio: float) -> int:
        layers = _gap_sweep_layers(ratio, vs)
        values = np.array(
            [
                _modal_determinant_n1_cased(
                    omega / c, omega, **_GAP_FORMATION, **_A2_BOREHOLE, layers=layers
                )
                for c in grid
            ]
        )
        finite = np.isfinite(values)
        return int((np.diff(np.sign(values[finite])) != 0).sum())

    for ratio in (1.26, 1.27, 1.275):
        assert crossings(ratio) == 1, f"the bound mode should exist at {ratio}"
    for ratio in (1.28, 1.30, 1.35, 1.40):
        assert crossings(ratio) == 0, (
            f"a bound root survives at {ratio}; it should have been absorbed"
        )


def test_no_leaky_branch_coexists_with_the_bound_mode_below_the_crossing():
    """The bound mode is alone below the crossing, which is what makes
    the leaky branch its continuation rather than a second family.

    **This test asserted the opposite until it was re-measured.** It
    claimed that at ``V_S_layer / V_S = 1.20`` three objects coexist at
    one stiffness -- the bound mode at 786.67 m/s, a leaky branch
    already radiating at 868.30, and a branch-point pole at 810.14
    between them -- and that the 7 % step the driver shows across the
    crossing is therefore a handover between two modes rather than a
    break in one.

    None of that survives correcting :func:`_k_or_hankel`'s radiation
    branch. Swept above ``V_S`` at this stiffness the *only* root is
    the layer's own shear speed, 960.0 m/s to the digit with zero
    imaginary part -- the degeneracy the production search already
    names and rejects. The leaky branch does not exist here at all; it
    emerges at ``V_S`` near ratio 1.28 and climbs, as
    ``test_the_branch_point_pole_was_the_incoming_contamination``
    shows. One mode, continuous through the crossing.

    It is worth recording *how* the old assertion survived the
    correction locally: seeded at 868 m/s the tracker lands on that
    960.0 m/s degeneracy, whose imaginary part is zero to rounding, and
    ``root.imag > 0.0`` then turns on the sign of a 1e-12 rounding
    error. It came out ``+5e-12`` on one machine and ``-6e-13`` in CI.
    A test that passes on the sign of a rounding error is not passing.
    """
    from fwap.cylindrical_solver import _modal_determinant_n1_cased_complex
    from fwap.cylindrical_solver._leaky import (
        _detect_leaky_branches,
        _track_complex_root,
    )

    omega = 2.0 * np.pi * float(_GAP_FREQ[0])
    vs = _GAP_FORMATION["vs"]
    ratio = 1.20
    layers = _gap_sweep_layers(ratio, vs)

    bound = flexural_dispersion_layered(
        _GAP_FREQ, **_GAP_FORMATION, **_A2_BOREHOLE, layers=layers
    )
    assert np.isfinite(bound.slowness[0])
    bound_velocity = 1.0 / bound.slowness[0]
    assert bound_velocity < vs, "below the crossing the driver reports a bound mode"
    assert bound.attenuation_per_meter is None or (
        float(bound.attenuation_per_meter[0]) == 0.0
    )

    def determinant(kz: complex) -> complex:
        _, leaky_p, leaky_s = _detect_leaky_branches(
            kz, omega, _GAP_FORMATION["vp"], vs, _A2_BOREHOLE["vf"]
        )
        return _modal_determinant_n1_cased_complex(
            kz,
            omega,
            **_GAP_FORMATION,
            **_A2_BOREHOLE,
            layers=layers,
            leaky_p=leaky_p,
            leaky_s=leaky_s,
        )

    # Sweep the whole window above V_S rather than probing one guess:
    # the claim is an absence, and a single seed cannot establish one.
    layer_velocity = ratio * vs
    found = []
    for k_real in np.linspace(omega / 1500.0 * 1.001, omega / (vs * 1.0006), 90):
        for level in (0.002, 0.01, 0.03, 0.08, 0.2):
            root = _track_complex_root(determinant, complex(k_real, level * k_real))
            if root is None or root.imag <= 1.0e-9:
                continue
            if not (omega / 1500.0 < root.real < omega / (vs * 1.0006)):
                continue
            found.append(omega / root.real)

    # Everything the sweep reaches is the layer's own shear speed, and
    # nothing else -- so there is no radiating branch here to coexist.
    assert all(v == pytest.approx(layer_velocity, rel=1.0e-6) for v in found), sorted(
        set(round(v, 2) for v in found)
    )

    # And that degeneracy is not a mode: its attenuation is zero to
    # rounding, which is exactly why seeding at it and asserting
    # Im(k_z) > 0 was a coin flip.
    degenerate = _track_complex_root(
        determinant, complex(omega / 868.0, 0.05 * omega / 868.0)
    )
    assert degenerate is not None
    assert omega / degenerate.real == pytest.approx(layer_velocity, rel=1.0e-6)
    assert abs(degenerate.imag) < 1.0e-9 * degenerate.real


def test_the_branch_point_pole_was_the_incoming_contamination():
    """Two tests used to live here. Their subject no longer exists.

    They characterised a "branch-point pole" hugging ``V_S`` in the
    cased n=1 determinant: sharp to 1e-13, carrying winding number +1,
    nearly static in annulus stiffness away from the crossing, and --
    swept finely through it -- dipping **below** ``V_S`` to 798.86 m/s
    at ``V_S_layer / V_S`` = 1.295 with positive attenuation, which is
    a mode propagating faster than the shear speed it is supposedly
    bound by. ``_LEAKY_CASED_SEED_FLOOR`` was built to keep the seed
    sweep off it, at the cost of A.9's recorded gap.

    **It was the incoming wave** :func:`_k_or_hankel` mixed into its
    leaky branch; see that function's docstring. Corrected, the same
    sweep on the same stiffness ladder shows no excursion below ``V_S``
    at all. What is there instead is the leaky continuation of the
    bound dipole mode: it emerges *at* the shear speed with zero
    attenuation, exactly where the bound branch is absorbed, and rises
    monotonically with the annulus from there.

    This test replaces both of them and asserts the replacement, so the
    withdrawal is a thing that runs rather than a note.
    """
    from fwap.cylindrical_solver import _modal_determinant_n1_cased_complex
    from fwap.cylindrical_solver._leaky import _track_complex_root

    omega = 2.0 * np.pi * float(_GAP_FREQ[0])
    vs = _GAP_FORMATION["vs"]

    velocity, attenuation, ratios = [], [], []
    root: complex | None = None
    for ratio in np.arange(1.275, 1.3451, 0.005):
        layers = _gap_sweep_layers(float(ratio), vs)

        def determinant(kz: complex, layers=layers) -> complex:
            return _modal_determinant_n1_cased_complex(
                kz,
                omega,
                **_GAP_FORMATION,
                **_A2_BOREHOLE,
                layers=layers,
                leaky_p=False,
                leaky_s=True,
            )

        seed = (
            root if root is not None else complex(omega / 803.0, 0.03 * omega / 803.0)
        )
        root = _track_complex_root(determinant, seed)
        if root is None:
            continue
        assert abs(determinant(root)) < 1.0e-6 * abs(determinant(root * 1.002))
        ratios.append(float(ratio))
        velocity.append(omega / root.real)
        attenuation.append(root.imag)

    velocity = np.array(velocity)
    attenuation = np.array(attenuation)
    assert velocity.size >= 12

    # It emerges at the shear speed rather than from under it. The
    # softest annulus that still carries the branch reads 798.02 m/s,
    # 0.25 % under V_S -- the branch point is singular and the last
    # step before it is not a clean measurement -- and every step after
    # that is above V_S. The old artefact was a 12-point excursion with
    # a turning point in it, not a single ragged first sample.
    assert velocity[0] > vs * 0.995, velocity[0]
    assert np.all(velocity[1:] > vs), velocity.min()
    assert int((np.diff(np.sign(np.diff(velocity))) != 0).sum()) == 0, velocity

    # And from there it is a mode, not a pinned artefact: monotone in
    # the annulus stiffness, in both velocity and leakage.
    assert np.all(np.diff(velocity) > 0.0), velocity
    assert np.all(np.diff(attenuation) > 0.0), attenuation
    # Anything but static -- the old pole moved under 1 % across a far
    # wider stiffness span than this one.
    assert velocity[-1] / velocity[0] - 1.0 > 0.02


def test_the_leaky_cased_search_never_evaluates_the_incoming_branch():
    """The search must stay on the outgoing sheet for its whole run.

    The returned roots always had ``Im(k_z) > 0``, so the answers were
    on the right sheet; the search getting there was not. 14 % of the
    dipole run's leaky Bessel evaluations, and 3 % of the screw's, sat
    at ``Im(alpha) < 0`` -- an *incoming* wave, the opposite of the
    radiation condition the leaky branch exists to impose -- because the
    principal square root flips below the real ``k_z`` axis. Roadmap
    A.10.
    """
    import fwap.cylindrical_solver._bessel as bessel_module
    import fwap.cylindrical_solver._cased as cased_module
    import fwap.cylindrical_solver._leaky as leaky_module
    import fwap.cylindrical_solver._n1_isotropic as n1_module
    import fwap.cylindrical_solver._n2_quadrupole as n2_module
    from fwap.cylindrical_solver import quadrupole_dispersion_layered

    seen: list[complex] = []
    original = bessel_module._k_or_hankel
    modules = (
        bessel_module,
        cased_module,
        leaky_module,
        n1_module,
        n2_module,
    )

    def spy(n, alpha, r, *, leaky):
        if leaky:
            seen.append(complex(alpha))
        return original(n, alpha, r, leaky=leaky)

    patched = [m for m in modules if hasattr(m, "_k_or_hankel")]
    try:
        for module in patched:
            module._k_or_hankel = spy
        layers = (_A2_CASING, _A2_CEMENT)
        for driver in (flexural_dispersion_layered, quadrupole_dispersion_layered):
            seen.clear()
            driver(_A2_FREQ, **_A2_SLOW, **_A2_BOREHOLE, layers=layers)
            assert seen, "the leaky branch was never exercised"
            incoming = [alpha for alpha in seen if alpha.imag < 0.0]
            assert not incoming, (
                f"{driver.__name__}: {len(incoming)} of {len(seen)} leaky "
                f"evaluations were on the incoming branch"
            )
    finally:
        for module in patched:
            module._k_or_hankel = original


def test_one_root_sits_in_the_slow_cased_gap_a9_could_not_seed():
    """What continuity across the axis buys, on A.9's own open question.

    A.9 recorded a gap: around ``V_S_layer / V_S`` in [1.3, 1.5] at
    ``ka = 2.5`` the real-axis scan finds no ``Im(det)`` crossing to
    seed from, and the roadmap noted that "a pole off the real axis
    cannot be seen that way, so an argument-principle search would be
    needed to say whether one is there".

    The argument principle needs a single-valued analytic function on
    and inside the contour, which is exactly what the branch rule
    supplies -- before it, a contour dipping below the real axis crossed
    the determinant's discontinuity and its winding number meant
    nothing. It now answers the question: one root, in the gap and on
    either side of it. Locating it is A.9 driver work and is not done
    here; this test pins the count.
    """
    from fwap.cylindrical_solver import _modal_determinant_n1_cased_complex
    from fwap.cylindrical_solver._leaky import _detect_leaky_branches

    formation = dict(vp=2200.0, vs=800.0, rho=2200.0)
    borehole = dict(vf=1500.0, rho_f=1000.0, a=0.10)
    omega = 2.5 * borehole["vf"] / borehole["a"]  # ka = 2.5

    # A box holding the leaky branch, with every formation branch point
    # outside it: k_P = 17.0, k_f = 25.0, k_S = 46.875 rad/m.
    #
    # It was (40, 46, -1, 6) and has been re-drawn. Correcting the
    # radiation branch of ``_k_or_hankel`` moved the root into the
    # *upper* half plane -- it had been below the real axis, which is
    # growth along the borehole -- and, at the soft end of the sweep,
    # up against the shear branch point: 46.6 rad/m at ratio 1.3,
    # outside the old box's 46.0 ceiling and inside the new one's
    # 46.8. The floor moves off the axis for the same reason.
    re_lo, re_hi, im_lo, im_hi = 40.0, 46.8, 0.02, 6.0
    steps = np.linspace(0.0, 1.0, 400, endpoint=False)
    contour = np.concatenate(
        [
            (re_lo + (re_hi - re_lo) * steps) + 1j * im_lo,
            re_hi + 1j * (im_lo + (im_hi - im_lo) * steps),
            (re_hi - (re_hi - re_lo) * steps) + 1j * im_hi,
            re_lo + 1j * (im_hi - (im_hi - im_lo) * steps),
        ]
    )

    for ratio in (1.3, 1.4, 1.5, 1.6):
        vs_layer = ratio * formation["vs"]
        layers = (
            BoreholeLayer(vp=2.2 * vs_layer, vs=vs_layer, rho=2000.0, thickness=0.04),
        )

        def determinant(kz: complex, layers=layers) -> complex:
            _, leaky_p, leaky_s = _detect_leaky_branches(
                kz, omega, formation["vp"], formation["vs"], borehole["vf"]
            )
            return _modal_determinant_n1_cased_complex(
                kz,
                omega,
                **formation,
                **borehole,
                layers=layers,
                leaky_p=leaky_p,
                leaky_s=leaky_s,
            )

        values = np.array([determinant(complex(z)) for z in contour])
        assert np.all(np.isfinite(values)), f"ratio {ratio}: contour hit a pole"
        phase = np.unwrap(np.angle(np.append(values, values[0])))
        winding = (phase[-1] - phase[0]) / (2.0 * np.pi)
        assert winding == pytest.approx(1.0, abs=1.0e-3), (
            f"Vs_layer/Vs = {ratio}: winding number {winding:.4f}, expected one root"
        )


def test_cased_flexural_fast_formation_covers_a_contiguous_middle_band():
    """After the A.2 fix the coverage is a contiguous band, not a tail.

    This test used to assert the *defect*: converged points only at the
    top of the band, because the ``(V_R, V_S)`` window kept whatever
    trapped mode happened to be inside it. With the window corrected to
    ``(V_f, V_S)`` and the fundamental selected, coverage is the band
    between the two crossings -- the mode is leaky below the ``V_R``
    crossing and has passed ``V_f`` above it, and both ends return NaN
    rather than a wrong branch.
    """
    res = flexural_dispersion_layered(
        _A2_FREQ, **_A2_FAST, **_A2_BOREHOLE, layers=(_A2_CASING, _A2_CEMENT)
    )
    finite = np.isfinite(res.slowness)
    assert finite.any()
    idx = np.where(finite)[0]
    assert np.array_equal(idx, np.arange(idx[0], idx[-1] + 1)), (
        "coverage must be one contiguous band, not scattered"
    )
    velocity = 1.0 / res.slowness[finite]
    assert np.all(np.diff(velocity) <= 0.0), "and monotonically descending"


def test_the_open_hole_and_cased_paths_agree_after_the_fix():
    """The open hole used to be just as sparse, which is what relocated
    A.2 from the layer stack to the fast-formation bracket.

    Both paths now share one marcher, so the point is stronger than
    before: they cover the same band and neither is the sparse tail the
    defect produced.
    """
    cased = flexural_dispersion_layered(
        _A2_FREQ, **_A2_FAST, **_A2_BOREHOLE, layers=(_A2_CASING, _A2_CEMENT)
    )
    open_hole = flexural_dispersion(_A2_FREQ, **_A2_FAST, **_A2_BOREHOLE)

    for res in (cased, open_hole):
        finite = np.isfinite(res.slowness)
        assert finite.any()
        idx = np.where(finite)[0]
        assert np.array_equal(idx, np.arange(idx[0], idx[-1] + 1))
        velocity = 1.0 / res.slowness[finite]
        assert np.all(np.diff(velocity) <= 0.0)


def test_converged_fast_formation_points_sit_below_the_shear_velocity():
    """The branch that is found is formation-controlled, bounded by V_S.

    It runs down from the shear branch point rather than tracking the
    cement, which is the other reason to stop calling this a cased-hole
    bracketing problem.
    """
    res = flexural_dispersion_layered(
        _A2_FREQ, **_A2_FAST, **_A2_BOREHOLE, layers=(_A2_CASING, _A2_CEMENT)
    )
    finite = np.isfinite(res.slowness)
    velocity = 1.0 / res.slowness[finite]
    v_rayleigh = rayleigh_speed(_A2_FAST["vp"], _A2_FAST["vs"])
    assert np.all(velocity < _A2_FAST["vs"])
    # It used to be asserted that the branch stays above V_R. It does not,
    # and that assumption is what A.2 turned out to be: the mode descends
    # through V_R toward Scholte. The real floor is V_f.
    assert np.all(velocity > _A2_BOREHOLE["vf"])
    assert velocity.min() < v_rayleigh, (
        "the branch must be seen to pass below V_R, or the fix is inactive"
    )
    # and it is dispersive downward with frequency, not a flat artefact
    assert velocity[-1] < velocity[0]


def test_complex_marcher_reproduces_the_real_cased_flexural_branch():
    """The complex root tracker and the n=1 cased determinant compose.

    Above the cutoff the cased flexural root is real, so marching it in the
    complex plane must return the same curve the real-axis solver finds,
    with an imaginary part at the level of floating-point noise. That is a
    prerequisite for any future leaky-mode work on this determinant --- if
    the marcher could not reproduce the part of the branch that is already
    known, its results below the cutoff would not be trustworthy either.

    Each frequency is seeded from the known root at that frequency rather
    than by continuation from the previous one. That is deliberate: with
    1 kHz steps, continuation seeding hops to a different branch (a root
    below the formation Rayleigh speed), which is one of the reasons the
    leaky extension needs the validated marcher's regime checks rather
    than the bare tracker.

    It is deliberately *not* a claim that the branch continues below the
    cutoff; see the roadmap for why that remains open.
    """
    from fwap.cylindrical_solver._cased import _modal_determinant_n1_cased_complex
    from fwap.cylindrical_solver._leaky import (
        _detect_leaky_branches,
        _track_complex_root,
    )

    def det(kz: complex, omega: float) -> complex:
        _, leaky_p, leaky_s = _detect_leaky_branches(
            kz, omega, _A2_FAST["vp"], _A2_FAST["vs"], _A2_BOREHOLE["vf"]
        )
        return _modal_determinant_n1_cased_complex(
            kz,
            omega,
            **_A2_FAST,
            **_A2_BOREHOLE,
            layers=(_A2_CASING, _A2_CEMENT),
            leaky_p=leaky_p,
            leaky_s=leaky_s,
        )

    frequencies = np.array([12000.0, 11000.0, 10000.0, 9000.0])
    reference = flexural_dispersion_layered(
        frequencies, **_A2_FAST, **_A2_BOREHOLE, layers=(_A2_CASING, _A2_CEMENT)
    )
    assert np.all(np.isfinite(reference.slowness))

    for frequency, expected in zip(frequencies, reference.slowness, strict=True):
        omega = 2.0 * np.pi * float(frequency)
        seed = complex(expected * omega, 0.0)
        root = _track_complex_root(lambda kz, w=omega: det(kz, w), seed)
        assert root is not None
        assert abs(root.imag) < 1.0e-9 * abs(root.real)
        assert root.real / omega == pytest.approx(expected, rel=1e-6)


# ----------------------------------------------------------------------
# The pseudo-Rayleigh geometric cutoff against its rigid-pipe closed form
#
# `_J1_FIRST_ZERO` has been in the module since the leaky work landed, with
# a docstring offering the rigid-pipe estimate
#
#     f_c ~ j_{1,1} V_f V_S / (2 pi a sqrt(V_S^2 - V_f^2))
#
# to "callers that want to guard against requesting frequencies below the
# cutoff". Nothing checked it against the solver. Doing so splits into a
# part that holds cleanly and a part that does not, and both are pinned
# here because the second one contradicts that docstring advice.
# ----------------------------------------------------------------------

_PR_ROCK = dict(vp=4500.0, vs=2600.0, rho=2400.0)
_PR_FLUID = dict(vf=1500.0, rho_f=1000.0)
# One fixed grid for every case. Tying the grid to the estimate would make
# any grid-determined stopping point produce a constant ratio for free,
# since the estimate scales as 1/a and so would the grid.
_PR_GRID = np.linspace(500.0, 30000.0, 1500)


def _rigid_pipe_cutoff(vs: float, vf: float, a: float) -> float:
    from fwap.cylindrical_solver._leaky import _J1_FIRST_ZERO

    return _J1_FIRST_ZERO * vf * vs / (2.0 * np.pi * a * np.sqrt(vs**2 - vf**2))


def _lowest_converged(a: float) -> float:
    from fwap import pseudo_rayleigh_modal_dispersion

    result = pseudo_rayleigh_modal_dispersion(_PR_GRID, **_PR_ROCK, **_PR_FLUID, a=a)
    finite = np.isfinite(result.slowness)
    assert finite.any()
    return float(_PR_GRID[finite].min())


# =====================================================================
# Slow-formation leaky compressional (n = 0).
# =====================================================================
#
# Sinha & Asvadurov (2004) Table 1 slow formation (B) -- the geometry
# the solver is scored on in the validation notebook, at 0.03 % RMS
# over 107 of 107 points of fig 11(a) curve m=3.


# Paillet & Cheng (1986) Table 1 shale B, which carries a 5 cm tool.
_LC_SHALE_B = dict(vp=2000.0, vs=1000.0, rho=2300.0, vf=1500.0, rho_f=1380.0, a=0.125)


def test_leaky_compressional_sits_in_its_window_and_decays():
    """The two properties that define the mode, neither of which needs
    a digitised figure.

    Between ``1/V_P`` and ``1/V_f`` the fluid is oscillatory, the
    formation P wave is evanescent and the formation S wave radiates.
    The last of those is what makes ``Im(k_z)`` non-zero, and its
    *sign* is the assertion: the mode must decay along +z. The
    radiation branch of :func:`_k_or_hankel` returned the incoming
    wave until it was corrected against Sinha & Asvadurov fig 11(a),
    and under it this root came out with ``Im(k_z) < 0`` -- a mode
    growing along the borehole.
    """
    from fwap.cylindrical_solver import leaky_compressional_dispersion

    freq = np.linspace(2500.0, 15000.0, 40)
    mode = leaky_compressional_dispersion(freq, **_LC_SLOW)

    assert mode.name == "leaky_compressional"
    assert mode.azimuthal_order == 0
    ok = np.isfinite(mode.slowness)
    assert ok.sum() > 35

    assert np.all(mode.slowness[ok] > 1.0 / _LC_SLOW["vp"])
    assert np.all(mode.slowness[ok] < 1.0 / _LC_SLOW["vf"])
    assert np.all(mode.attenuation_per_meter[ok] > 0.0)

    # Slower with frequency, toward the fluid slowness -- the paper's
    # own description of the high-frequency asymptote.
    assert np.all(np.diff(mode.slowness[ok]) > 0.0)


def test_leaky_compressional_roots_are_roots():
    """Each returned value solves the determinant it claims to.

    Checked against the determinant's own magnitude nearby, because the
    absolute value spans many orders across the window and an absolute
    threshold would mean nothing.
    """
    from fwap.cylindrical_solver import (
        _modal_determinant_n0_complex,
        leaky_compressional_dispersion,
    )

    freq = np.linspace(4000.0, 12000.0, 6)
    mode = leaky_compressional_dispersion(freq, **_LC_SLOW)
    for f, s, att in zip(freq, mode.slowness, mode.attenuation_per_meter):
        assert np.isfinite(s), f
        omega = 2.0 * np.pi * f
        kz = complex(s * omega, att)

        def det(z, omega=omega):
            return _modal_determinant_n0_complex(
                z, omega, **_LC_SLOW, leaky_p=False, leaky_s=True
            )

        radius = 0.001 * abs(kz)
        ring = float(
            np.median(
                [
                    abs(det(kz + radius * np.exp(1j * t)))
                    for t in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
                ]
            )
        )
        assert abs(det(kz)) < 1.0e-9 * ring, f


def test_leaky_compressional_has_no_low_frequency_cutoff():
    """The fundamental approaches ``1/V_P`` asymptotically, not at a
    cut-off, and the difference is worth pinning.

    The function's docstring asserted a cut-off near 2.2 kHz on this
    formation, taken from where Sinha's fig 11(a) starts drawing the
    curve and from the paper's "cuts in around 3 kHz". Both are
    statements about detectability. Measured, the branch runs
    continuously below them: the slowness closes on ``1/V_P`` like a
    limit, with the attenuation vanishing alongside it, and never
    terminates.

    Below ~2 kHz that limit is what it looks like -- a wave at the
    formation compressional speed radiating essentially nothing, which
    no receiver would separate from the P head wave. It is returned
    anyway, because trimming it would mean inventing a threshold.
    """
    from fwap.cylindrical_solver import leaky_compressional_dispersion

    freq = np.linspace(1300.0, 6000.0, 60)
    mode = leaky_compressional_dispersion(freq, **_LC_SLOW)
    ok = np.isfinite(mode.slowness)

    # No gap: one unbroken run down to the bottom of the grid.
    assert ok.all(), np.flatnonzero(~ok)

    excess = mode.slowness / (1.0 / _LC_SLOW["vp"]) - 1.0
    assert np.all(excess > 0.0), excess.min()
    assert np.all(np.diff(excess) > 0.0), "the approach is not monotone"
    # Six orders of magnitude of approach across the band, and the
    # attenuation tracks it down.
    assert excess[0] < 1.0e-8, excess[0]
    assert excess[-1] > 1.0e-2, excess[-1]
    assert mode.attenuation_per_meter[0] < 1.0e-7
    assert np.all(np.diff(mode.attenuation_per_meter) > 0.0)


def test_leaky_compressional_grid_independence():
    """The curve is a property of the medium, not of the request."""
    from fwap.cylindrical_solver import leaky_compressional_dispersion

    coarse_grid = np.arange(3000.0, 14000.0, 250.0)
    fine_grid = np.arange(3000.0, 14000.0, 125.0)
    coarse = leaky_compressional_dispersion(coarse_grid, **_LC_SLOW)
    fine = leaky_compressional_dispersion(fine_grid, **_LC_SLOW)
    shared = np.searchsorted(fine_grid, coarse_grid)

    ok = np.isfinite(coarse.slowness) & np.isfinite(fine.slowness[shared])
    assert ok.sum() > 40
    assert coarse.slowness[ok] == pytest.approx(fine.slowness[shared][ok], rel=1.0e-9)


def test_leaky_compressional_tool_radius_moves_the_answer():
    """The rigid tool is load-bearing on Paillet & Cheng's shale B.

    Not a smoke test: the same fundamental scores 1.81 % against fig
    12(a) with the 5 cm tool and 10.66 % without it, so the two curves
    have to be well separated. Asserted as a lower bound on the shift
    rather than against a stored curve.
    """
    from fwap.cylindrical_solver import leaky_compressional_dispersion

    freq = np.linspace(8000.0, 24000.0, 30)
    with_tool = leaky_compressional_dispersion(freq, **_LC_SHALE_B, tool_radius=0.05)
    open_hole = leaky_compressional_dispersion(freq, **_LC_SHALE_B)

    both = np.isfinite(with_tool.slowness) & np.isfinite(open_hole.slowness)
    assert both.sum() > 20
    shift = np.abs(open_hole.slowness[both] / with_tool.slowness[both] - 1.0)
    assert shift.max() > 0.03, shift.max()


def test_leaky_compressional_branches_are_distinct():
    """Higher radial orders exist and are addressable.

    Shale B carries three roots at 24.8 kHz, and the middle one is a
    cut-off mode two orders more attenuated than its neighbours -- so
    the paper's "first mode" is ``branch=2``. That ordering trap is
    documented on the function and pinned here.
    """
    from fwap.cylindrical_solver import leaky_compressional_dispersion

    freq = np.array([24800.0])
    slownesses, attenuations = [], []
    for branch in (0, 1, 2):
        mode = leaky_compressional_dispersion(
            freq, **_LC_SHALE_B, branch=branch, tool_radius=0.05
        )
        assert np.isfinite(mode.slowness[0]), branch
        slownesses.append(float(mode.slowness[0]) * 1e6)
        attenuations.append(float(mode.attenuation_per_meter[0]))

    # Ordered by descending Re(k_z), i.e. descending slowness.
    assert slownesses == sorted(slownesses, reverse=True), slownesses
    # ...and branch 1 is the heavily damped one sitting between the two
    # propagating branches in slowness, which is the trap.
    assert attenuations[1] > 10.0 * attenuations[0], attenuations
    assert attenuations[1] > 5.0 * attenuations[2], attenuations


def test_the_sinha_attenuation_convention_reproduces_its_own_group_slowness():
    """The relation behind the fig 11(c) score, checked the way it was found.

    Sinha & Asvadurov state no dB convention -- all six of their
    attenuation panels are labelled only "Attenuation (dB/m)" -- so the
    one used to score fig 11(c),

        Sinha dB/m = 8.686 * Im(k_z) * (V_g / V_p) / 2

    was recovered rather than read. This test re-runs that recovery in
    the direction that makes it falsifiable: invert the *measured* ratio
    between fwap's naive ``8.686 Im(k_z)`` and the digitised fig 11(c)
    into an implied group slowness, and require it to reproduce the
    group slowness fig 11(b) plots independently.

    The two panels share nothing but the mode. Fig 11(b) is calibrated
    on its own gridlines (500-3000 us/m over six lines) and fig 11(c) on
    its own (0-15 dB/m over four), and neither played any part in
    producing the other. If the convention were wrong -- or if panel
    (c)'s y-axis had been mis-calibrated by any factor -- the implied
    group slowness would be off by that factor and would miss.

    This doubles as the calibration check panel (c) cannot have on its
    own: attenuation panels carry no dashed reference lines.
    """
    from fwap.cylindrical_solver import leaky_compressional_dispersion

    data = Path(__file__).resolve().parents[1] / "docs" / "notebooks" / "_data"
    attenuation = np.loadtxt(
        data / "sinha_asvadurov_2004_fig11c_leaky_compressional_attenuation_slow.csv",
        delimiter=",",
    )
    group = np.loadtxt(
        data / "sinha_asvadurov_2004_fig11b_leaky_compressional_group_slow.csv",
        delimiter=",",
    )

    freq = np.arange(2200.0, 15001.0, 25.0)
    mode = leaky_compressional_dispersion(freq, **_LC_SLOW)
    live = np.isfinite(mode.slowness)
    f_live = freq[live]
    s_live = mode.slowness[live]
    naive = 8.686 * mode.attenuation_per_meter[live]

    # Only where the reference attenuation is big enough for a ratio to
    # mean anything, and inside fig 11(b)'s own band.
    band = (
        (attenuation[:, 1] > 0.05)
        & (attenuation[:, 0] >= max(group[:, 0].min(), f_live.min()))
        & (attenuation[:, 0] <= min(group[:, 0].max(), f_live.max()))
    )
    probe = attenuation[band]
    assert probe.shape[0] > 15

    ours = np.interp(probe[:, 0], f_live, naive)
    phase = np.interp(probe[:, 0], f_live, s_live)
    # theirs = ours * (V_g / V_p) / 2  =>  V_g / V_p = 2 * theirs / ours
    implied_vg_over_vp = 2.0 * probe[:, 1] / ours
    implied_group_slowness = phase / implied_vg_over_vp

    measured = np.interp(probe[:, 0], group[:, 0], group[:, 1])
    residual = (implied_group_slowness - measured) / measured
    rms = float(np.sqrt(np.mean(residual**2)))
    assert rms < 0.01, f"implied group slowness is {100 * rms:.2f}% RMS off"

    # And the naive reading really is the ~2.2x it was first found to
    # be -- so the factor is recorded, not silently absorbed.
    ratio = ours / probe[:, 1]
    assert 2.0 < float(np.median(ratio)) < 2.4, float(np.median(ratio))


def test_leaky_compressional_attenuation_matches_sinha_fig11c():
    """The imaginary part, scored -- the only such tie in the package.

    Every other external tie in ``docs/notebooks/`` looks at a phase
    slowness. This one uses fwap's own group velocity to convert
    ``Im(k_z)`` into Sinha's quantity, so fig 11(b) is not an input and
    the two scores stay independent.
    """
    from fwap.cylindrical_solver import leaky_compressional_dispersion

    data = Path(__file__).resolve().parents[1] / "docs" / "notebooks" / "_data"
    reference = np.loadtxt(
        data / "sinha_asvadurov_2004_fig11c_leaky_compressional_attenuation_slow.csv",
        delimiter=",",
    )

    freq = np.arange(2200.0, 15001.0, 25.0)
    mode = leaky_compressional_dispersion(freq, **_LC_SLOW)
    live = np.isfinite(mode.slowness)
    f_live = freq[live]
    s_live = mode.slowness[live]
    omega = 2.0 * np.pi * f_live
    group_slowness = np.gradient(omega * s_live, omega)
    predicted = (
        8.686 * mode.attenuation_per_meter[live] * (s_live / group_slowness) / 2.0
    )

    # The floor is not a tolerance: below it the reference is a few
    # thousandths of a dB/m and a relative budget cannot mean anything.
    keep = (
        (reference[:, 1] > 0.05)
        & (reference[:, 0] >= f_live.min())
        & (reference[:, 0] <= f_live.max())
    )
    assert keep.sum() > 80
    got = np.interp(reference[keep, 0], f_live, predicted)
    residual = (got - reference[keep, 1]) / reference[keep, 1]
    assert float(np.sqrt(np.mean(residual**2))) < 0.01
    assert float(np.max(np.abs(residual))) < 0.05


def test_leaky_compressional_group_slowness_matches_sinha_fig11b():
    """The group slowness scored on its own, straight from the curve."""
    from fwap.cylindrical_solver import leaky_compressional_dispersion

    data = Path(__file__).resolve().parents[1] / "docs" / "notebooks" / "_data"
    reference = np.loadtxt(
        data / "sinha_asvadurov_2004_fig11b_leaky_compressional_group_slow.csv",
        delimiter=",",
    )

    freq = np.arange(2200.0, 15001.0, 25.0)
    mode = leaky_compressional_dispersion(freq, **_LC_SLOW)
    live = np.isfinite(mode.slowness)
    omega = 2.0 * np.pi * freq[live]
    group_slowness = np.gradient(omega * mode.slowness[live], omega)

    got = np.interp(reference[:, 0], freq[live], group_slowness)
    residual = (got - reference[:, 1]) / reference[:, 1]
    assert float(np.sqrt(np.mean(residual**2))) < 0.01

    # A group slowness slower than the phase slowness everywhere the
    # mode is normally dispersive -- true of the curve, not just of the
    # match, and it is what makes Sinha's factor bigger than 1/2.
    phase = np.interp(reference[:, 0], freq[live], mode.slowness[live])
    assert np.all(got > phase)


def test_leaky_compressional_rejects_a_fast_formation():
    """A fast formation has no such window; say which function to use."""
    from fwap.cylindrical_solver import leaky_compressional_dispersion

    fast = dict(vp=4000.0, vs=2300.0, rho=2500.0, vf=1500.0, rho_f=1000.0, a=0.10)
    with pytest.raises(ValueError, match="pseudo_rayleigh_dispersion"):
        leaky_compressional_dispersion(np.array([8000.0]), **fast)

    with pytest.raises(ValueError, match="tool_radius must be smaller"):
        leaky_compressional_dispersion(
            np.array([8000.0]), **_LC_SHALE_B, tool_radius=0.2
        )
    with pytest.raises(ValueError, match="branch must be non-negative"):
        leaky_compressional_dispersion(np.array([8000.0]), **_LC_SLOW, branch=-1)


# Sinha & Asvadurov (2004) Table 1 fast formation (A) -- the geometry
# pseudo_rayleigh_dispersion is scored on, fig 2 curve m=3.


def _sinha_fig2_reference(name):
    data = Path(__file__).resolve().parents[1] / "docs" / "notebooks" / "_data"
    return np.loadtxt(data / name, delimiter=",")


def test_pseudo_rayleigh_matches_sinha_fig2a_m3():
    """The external tie this function went its whole life without.

    Sinha calls m=3 of fig 2(a) a *leaky compressional* mode, and in a
    fast formation that window -- ``1/V_P < s < 1/V_S`` -- is the one
    this function tracks. It is ``branch=1``: the formation's trapped
    branches cut off at 7.45 and 15.6 kHz, and m=3 leaves ``1/V_S`` at
    the top of the plotted band to reach ``1/V_P`` at 8.95 kHz, so it is
    branch 1's continuation rather than branch 0's.

    An earlier attempt at this comparison returned 11.3 % and was
    rejected as the wrong mode. It was the wrong branch index, on a
    contaminated radiation branch, with grid-dependent seeding. All
    three are fixed and the same comparison now lands at 1.06 %.
    """
    from fwap.cylindrical_solver import pseudo_rayleigh_dispersion

    reference = _sinha_fig2_reference(
        "sinha_asvadurov_2004_fig2a_leaky_compressional_fast.csv"
    )
    freq = np.arange(8000.0, 15300.0, 10.0)
    mode = pseudo_rayleigh_dispersion(freq, **_PR_SINHA_FAST, branch=1)
    live = np.isfinite(mode.slowness)
    assert live.sum() > 400

    inside = (reference[:, 0] >= freq[live].min()) & (
        reference[:, 0] <= freq[live].max()
    )
    assert inside.sum() > 150
    got = np.interp(reference[inside, 0], freq[live], mode.slowness[live])
    residual = (got - reference[inside, 1]) / reference[inside, 1]
    assert float(np.sqrt(np.mean(residual**2))) < 0.02

    # The window itself, which no digitised curve is needed to check.
    assert np.all(mode.slowness[live] > 1.0 / _PR_SINHA_FAST["vp"])
    assert np.all(mode.slowness[live] < 1.0 / _PR_SINHA_FAST["vs"])
    assert np.all(mode.attenuation_per_meter[live] > 0.0)


def test_pseudo_rayleigh_attenuation_matches_sinha_fig2c_m3():
    """The dB convention transfers to a different formation and mode.

    It was recovered on fig 11 -- slow formation, ``n = 0, m = 3`` --
    by predicting a group slowness and checking it against fig 11(b).
    Here the same relation is applied to a *fast* formation, a
    different figure and a different branch index, with nothing
    re-derived.

    The transfer is the evidence, and it is stronger than a repeated
    constant would be. The correction factor is ``2 V_p / V_g``, so it
    is a property of the mode's dispersion rather than a units
    constant -- and it does differ: the naive ``8.686 Im(k_z)`` reading
    overshoots by a median **4.15x** on this strongly dispersive fast
    branch against **2.2x** on the slow-formation curve it was derived
    from. A convention that were merely a coincidental factor of two
    would fail here; this one lands inside budget.
    """
    from fwap.cylindrical_solver import pseudo_rayleigh_dispersion

    reference = _sinha_fig2_reference(
        "sinha_asvadurov_2004_fig2c_leaky_compressional_attenuation_fast.csv"
    )
    freq = np.arange(8000.0, 15300.0, 10.0)
    mode = pseudo_rayleigh_dispersion(freq, **_PR_SINHA_FAST, branch=1)
    live = np.isfinite(mode.slowness)
    f_live = freq[live]
    s_live = mode.slowness[live]
    omega = 2.0 * np.pi * f_live
    group = np.gradient(omega * s_live, omega)
    predicted = 8.686 * mode.attenuation_per_meter[live] * (s_live / group) / 2.0

    inside = (reference[:, 0] >= f_live.min()) & (reference[:, 0] <= f_live.max())
    got = np.interp(reference[inside, 0], f_live, predicted)
    residual = (got - reference[inside, 1]) / reference[inside, 1]
    assert float(np.sqrt(np.mean(residual**2))) < 0.05

    # ...and the naive reading is far off, by a factor this formation
    # sets rather than a universal one, so the convention is doing real
    # work rather than being decorative.
    naive = np.interp(
        reference[inside, 0], f_live, 8.686 * mode.attenuation_per_meter[live]
    )
    ratio = float(np.median(naive / reference[inside, 1]))
    assert 3.8 < ratio < 4.6, ratio
    # The factor is 2 V_p / V_g, so it must track the group velocity
    # rather than being a constant -- check that directly.
    group_ratio = np.interp(reference[inside, 0], f_live, 2.0 * group / s_live)
    assert (
        float(np.median(np.abs(naive / reference[inside, 1] / group_ratio - 1.0)))
        < 0.05
    )


def test_the_pseudo_rayleigh_low_frequency_end_is_a_window_edge_not_a_cutoff():
    """What stops the curve at its fast end, measured rather than assumed.

    On Sinha's fast formation this routine returns nothing below
    9.17 kHz, and it is tempting to read that as the mode's cut-on --
    an earlier version of this module's write-up did exactly that, and
    reported a "2.5 % cut-on offset" against the figure. It is not a
    cut-on. It is the ``slowness > 1/V_P`` floor in the validator, and
    the root passes straight through it: continued by hand it reaches
    9.02 kHz at 263.5 us/m, faster than ``V_P``, still converging.

    The marcher stops there for a good reason -- below the floor the
    formation P wave ought to radiate as well, which is a different
    determinant -- but "the mode ends here" is not that reason.

    Nor is the floor near a branch point, which is the other tempting
    assumption. The root is strongly damped there, so ``p`` stays far
    from the ``p = 0`` compressional branch point.
    """
    from fwap.cylindrical_solver import pseudo_rayleigh_dispersion
    from fwap.cylindrical_solver._bessel import _radial_wavenumber
    from fwap.cylindrical_solver._leaky import (
        _modal_determinant_n0_complex,
        _track_complex_root,
    )

    freq = np.arange(9000.0, 15300.0, 5.0)
    mode = pseudo_rayleigh_dispersion(freq, **_PR_SINHA_FAST, branch=1)
    live = np.isfinite(mode.slowness)
    edge = float(freq[live].min())
    assert 9.1e3 < edge < 9.25e3, edge
    # It stops *at* the floor, not short of it.
    assert mode.slowness[live][0] == pytest.approx(
        1.0 / _PR_SINHA_FAST["vp"], rel=5.0e-4
    )

    # Continue the same root below the floor by hand: it is still there.
    kz = complex(
        2.0 * np.pi * edge * mode.slowness[live][0], mode.attenuation_per_meter[live][0]
    )
    previous = edge
    for f in np.arange(edge - 20.0, 9.00e3, -20.0):
        omega = 2.0 * np.pi * f

        def det(z, omega=omega):
            return _modal_determinant_n0_complex(
                z, omega, **_PR_SINHA_FAST, leaky_p=False, leaky_s=True
            )

        kz = _track_complex_root(det, kz * (f / previous))
        assert kz is not None, f
        previous = f

    omega = 2.0 * np.pi * previous
    slowness_below = kz.real / omega
    assert slowness_below < 1.0 / _PR_SINHA_FAST["vp"], "root did not cross the C line"
    assert slowness_below * 1e6 == pytest.approx(263.5, abs=1.0)
    assert kz.imag > 0.0

    # And at the crossing the root is nowhere near the compressional
    # branch point, because it is strongly damped: p = 6.9 + 7.7i.
    omega_edge = 2.0 * np.pi * edge
    kz_edge = complex(
        omega_edge * mode.slowness[live][0], mode.attenuation_per_meter[live][0]
    )
    assert kz_edge.imag > 3.0
    p_edge = _radial_wavenumber(
        kz_edge * kz_edge - (omega_edge / _PR_SINHA_FAST["vp"]) ** 2, leaky=False
    )
    assert abs(p_edge) > 5.0, p_edge


def test_agreement_with_sinha_degrades_with_the_mode_damping():
    """Why the fast curve scores 1.06 % and the slow one 0.03 %.

    Two earlier readings of that gap blamed the C line -- first "a 2.5 %
    cut-on offset", then "fwap's curve is 2.6x steeper there". Both were
    replaced. The residual is not a feature of the C line: it tracks
    ``Im(k_z)``, and the damping simply happens to be largest at this
    branch's low-frequency end.

    Same solver, same paper, same branch machinery, two decades of
    damping apart:

    * fig 11(a), slow formation, ``Im(k_z)`` <= 0.19 rad/m -> 0.025 %
    * fig 2(a), fast formation, ``Im(k_z)`` up to 3.6 rad/m -> 0.79 %

    and within the fast curve the residual rises with ``Im(k_z)`` too.
    That is coherent: a strongly damped mode is a pole far from the real
    axis, where which pole you get depends on branch-cut placement and
    on how the radiation condition is imposed, so two implementations
    can legitimately differ. Near the real axis the answer is
    essentially unique.

    Asserted as an ordering rather than as fitted coefficients, so it
    pins the finding without pinning noise.
    """
    from fwap.cylindrical_solver import (
        leaky_compressional_dispersion,
        pseudo_rayleigh_dispersion,
    )

    fast_ref = _sinha_fig2_reference(
        "sinha_asvadurov_2004_fig2a_leaky_compressional_fast.csv"
    )
    freq = np.arange(9100.0, 15300.0, 5.0)
    fast = pseudo_rayleigh_dispersion(freq, **_PR_SINHA_FAST, branch=1)
    live = np.isfinite(fast.slowness)
    inside = (fast_ref[:, 0] >= freq[live].min()) & (fast_ref[:, 0] <= freq[live].max())
    got = np.interp(fast_ref[inside, 0], freq[live], fast.slowness[live])
    damping = np.interp(
        fast_ref[inside, 0], freq[live], fast.attenuation_per_meter[live]
    )
    residual = np.abs(got - fast_ref[inside, 1]) / fast_ref[inside, 1]

    # Within the fast curve: the more damped half misses by more.
    split = float(np.median(damping))
    low, high = residual[damping < split], residual[damping >= split]
    assert low.size > 40 and high.size > 40
    assert high.mean() > 2.0 * low.mean(), (low.mean(), high.mean())

    # And the weakly damped slow-formation mode does far better still.
    slow_ref = _sinha_fig2_reference(
        "sinha_asvadurov_2004_fig11a_leaky_compressional_slow.csv"
    )
    slow_freq = np.arange(2200.0, 15100.0, 5.0)
    slow = leaky_compressional_dispersion(slow_freq, **_LC_SLOW)
    slow_live = np.isfinite(slow.slowness)
    slow_got = np.interp(slow_ref[:, 0], slow_freq[slow_live], slow.slowness[slow_live])
    slow_damping = np.interp(
        slow_ref[:, 0], slow_freq[slow_live], slow.attenuation_per_meter[slow_live]
    )
    slow_residual = np.abs(slow_got - slow_ref[:, 1]) / slow_ref[:, 1]

    assert slow_damping.max() < 0.5, slow_damping.max()
    assert damping.max() > 3.0, damping.max()
    assert slow_residual.mean() < 0.1 * residual.mean(), (
        slow_residual.mean(),
        residual.mean(),
    )


def test_the_sinha_fig2_m3_branch_is_the_only_root_in_its_window():
    """No ambiguity about *which* root is being scored.

    The argument principle counts one root in ``(1/V_P, 1/V_S)`` at
    every frequency across the band, so the identification rests on a
    count rather than on the overlay agreeing.
    """
    from fwap.cylindrical_solver._leaky import _modal_determinant_n0_complex

    for probe in (9.5e3, 11.0e3, 13.0e3, 14.5e3):
        omega = 2.0 * np.pi * probe

        def det(kz, omega=omega):
            return _modal_determinant_n0_complex(
                kz, omega, **_PR_SINHA_FAST, leaky_p=False, leaky_s=True
            )

        lo = omega / _PR_SINHA_FAST["vp"] * 1.0005
        hi = omega / _PR_SINHA_FAST["vs"] * 0.9995
        steps = np.linspace(0.0, 1.0, 2000, endpoint=False)
        contour = np.concatenate(
            [
                lo + (hi - lo) * steps + 1j * 1.0e-4,
                hi + 1j * (1.0e-4 + 20.0 * steps),
                hi - (hi - lo) * steps + 1j * 20.0,
                lo + 1j * (20.0 - 20.0 * steps),
            ]
        )
        values = np.array([det(complex(z)) for z in contour])
        assert np.all(np.isfinite(values)), probe
        phase = np.unwrap(np.angle(np.append(values, values[0])))
        winding = (phase[-1] - phase[0]) / (2.0 * np.pi)
        assert winding == pytest.approx(1.0, abs=1.0e-3), (probe, winding)


def test_the_leaky_branch_joins_the_trapped_one_at_its_cutoff():
    """Below a trapped branch's cut-off the same mode continues as a
    leaky root, leaving ``1/V_S`` with ``Im(k_z) -> 0^+``.

    This is the physics oracle the n=0 leaky branch never had, and the
    one that would have caught the radiation-branch sign error on the
    *fast* formation without any digitised figure. Continuity at the
    cut-off is not a convention: a trapped mode at ``c = V_S`` radiates
    no shear, so its leaky continuation must start with zero
    attenuation and gain it smoothly as frequency drops.

    With the incoming-contaminated branch the continuation was not
    there at all -- the root just below the cut-off came out with
    ``Im(k_z) < 0``, growing along the borehole.

    Driven through the determinant rather than through
    :func:`pseudo_rayleigh_dispersion`, because that routine seeds only
    at the top of the caller's grid and so does not reach the
    fundamental's continuation; see its docstring.
    """
    from fwap.cylindrical_solver import trapped_pseudo_rayleigh_dispersion
    from fwap.cylindrical_solver._leaky import (
        _modal_determinant_n0_complex,
        _track_complex_root,
    )

    medium = dict(vp=4000.0, vs=2300.0, rho=2500.0, vf=1500.0, rho_f=1000.0, a=0.10)

    # Locate the fundamental trapped branch's cut-off, and confirm it
    # sits at the shear slowness as it must.
    freq = np.arange(6.0e3, 12.0e3, 25.0)
    trapped = trapped_pseudo_rayleigh_dispersion(freq, **medium, branch=0)
    live = np.isfinite(trapped.slowness)
    assert live.any()
    f_cutoff = float(freq[live].min())
    assert trapped.slowness[live][0] == pytest.approx(1.0 / medium["vs"], rel=1.0e-3)

    def det(kz: complex, omega: float) -> complex:
        return _modal_determinant_n0_complex(
            kz, omega, **medium, leaky_p=False, leaky_s=True
        )

    # March the leaky root downward from just under the cut-off.
    offsets = np.array([50.0, 200.0, 500.0, 1000.0, 2000.0])
    kz, previous = None, 0.0
    attenuation, slowness = [], []
    for f in f_cutoff - offsets:
        omega = 2.0 * np.pi * f
        seed = (
            kz * (f / previous)
            if kz is not None
            else complex(omega / medium["vs"] * 0.999, 0.02)
        )
        root = _track_complex_root(lambda z, w=omega: det(z, w), seed)
        assert root is not None, f
        kz, previous = root, f
        attenuation.append(root.imag)
        slowness.append(root.real / omega)

    attenuation = np.array(attenuation)
    slowness = np.array(slowness)

    # Decaying along +z, not growing. This is the assertion the old
    # branch failed: it returned Im(k_z) < 0 throughout.
    assert np.all(attenuation > 0.0), attenuation

    # Zero attenuation at the cut-off, monotonically gained below it.
    assert attenuation[0] < 0.01
    assert np.all(np.diff(attenuation) > 0.0), attenuation

    # And it leaves from the shear slowness, faster than V_S but never
    # faster than V_P.
    assert slowness[0] == pytest.approx(1.0 / medium["vs"], rel=2.0e-3)
    assert np.all(slowness < 1.0 / medium["vs"])
    assert np.all(slowness > 1.0 / medium["vp"])


def test_pseudo_rayleigh_cutoff_scales_inversely_with_borehole_radius():
    """The geometric 1/a scaling of the closed form is reproduced exactly.

    This is the part of the rigid-pipe comparison that holds. Measured on a
    *fixed* frequency grid across a 3.3x range of borehole radius, the ratio
    of solver cutoff to closed form is constant to about 1 part in 300 --
    so the radius enters the solver's cutoff exactly as the closed form
    says it should.

    It would catch a radius/diameter confusion, which is the classic way to
    get this wrong and is invisible to any single-radius check.
    """
    radii = [0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
    ratios = [
        _lowest_converged(a) / _rigid_pipe_cutoff(2600.0, 1500.0, a) for a in radii
    ]
    assert max(ratios) - min(ratios) < 0.01 * np.mean(ratios)


def test_the_rigid_pipe_estimate_is_not_usable_as_a_cutoff_guard():
    """It overestimates by ~2.8x, so guarding with it discards a valid band.

    The docstring of `pseudo_rayleigh_dispersion` offers the rigid-pipe
    formula to callers wanting to avoid requesting frequencies below the
    cutoff. Taken literally that is bad advice: the solver converges well
    below the estimate, because a compliant elastic wall admits the mode at
    lower frequency than a rigid pipe does.

    Pinned as a bound rather than a constant, since the factor is not
    universal -- it varies strongly with formation velocity (see the
    module-level note and plans/roadmap.md A.1).
    """
    estimate = _rigid_pipe_cutoff(2600.0, 1500.0, 0.10)
    measured = _lowest_converged(0.10)
    assert measured < 0.5 * estimate
    # and the discarded band is wide enough to matter
    assert estimate - measured > 5000.0


# ----------------------------------------------------------------------
# Quadrupole high-frequency asymptote, and where it stops being usable
#
# At short wavelength the borehole wall looks flat to every azimuthal
# order, so the n=2 quadrupole must approach the same plane-interface
# Scholte speed the n=0 Stoneley does. `scholte_speed` computes that from
# a different equation, so this is an external check on the n=2 solver.
#
# It holds in slow formations. In fast ones the mode is leaky and the
# real-axis search returns a non-monotone scatter between the Rayleigh and
# shear speeds -- the same failure as the n=1 flexural case (roadmap A.2),
# now known to affect n=2 as well.
# ----------------------------------------------------------------------

_QUAD_SLOW = dict(vp=2200.0, vs=800.0, rho=2200.0)
_QUAD_MID = dict(vp=3000.0, vs=1400.0, rho=2300.0)


@pytest.mark.parametrize("rock", [_QUAD_SLOW, _QUAD_MID], ids=["slow", "mid"])
def test_quadrupole_converges_to_the_plane_scholte_speed(rock):
    """n=2 approaches the same plane-interface limit as n=0.

    Checked against `scholte_speed`, which solves a plane fluid/solid
    interface problem -- a different equation from the n=2 modal
    determinant -- so agreement is a cross-check rather than the solver
    confirming itself.
    """
    from fwap import quadrupole_dispersion, scholte_speed

    reference = scholte_speed(**rock, **_QUAD_FLUID)
    frequencies = np.array([50.0e3, 100.0e3, 200.0e3, 400.0e3])
    result = quadrupole_dispersion(frequencies, **rock, **_QUAD_FLUID, a=0.10)
    assert np.all(np.isfinite(result.slowness))

    error = np.abs(1.0 / result.slowness / reference - 1.0)
    assert np.all(np.diff(error) < 0.0)  # converging, not merely close
    assert error[-1] < 1.0e-3


def test_quadrupole_and_stoneley_share_the_high_frequency_limit():
    """Both azimuthal orders must land on the same flat-wall answer."""
    from fwap import quadrupole_dispersion, stoneley_dispersion

    high = np.array([400.0e3])
    quad = quadrupole_dispersion(high, **_QUAD_SLOW, **_QUAD_FLUID, a=0.10)
    stoneley = stoneley_dispersion(high, **_QUAD_SLOW, **_QUAD_FLUID, a=0.10)
    assert 1.0 / quad.slowness[0] == pytest.approx(1.0 / stoneley.slowness[0], rel=1e-4)


def test_fast_formation_quadrupole_is_now_a_usable_curve():
    """In a fast formation the returned values are a guided mode again.

    They used to be finite -- the hazard, since a caller filtering on
    NaN keeps them -- but non-monotone, scattered between the Rayleigh
    and shear speeds as successive overtones crossed the old window.
    With the window corrected to ``(V_f, V_S)`` and the fundamental
    selected, phase velocity descends monotonically, and every value
    sits below ``V_R`` where the old bracket could not reach.

    The lower bound has since moved again: with the sub-fluid
    continuation the curve descends past ``V_f`` toward Scholte, so
    ``velocity > V_f`` is no longer true of it and the floor asserted
    here is the Scholte speed.
    """
    from fwap import quadrupole_dispersion, scholte_speed

    frequencies = np.array([10.0e3, 15.0e3, 20.0e3, 30.0e3, 60.0e3, 100.0e3])
    result = quadrupole_dispersion(frequencies, **_QUAD_FAST, **_QUAD_FLUID, a=0.10)
    finite = np.isfinite(result.slowness)
    assert finite.sum() >= 4

    velocity = 1.0 / result.slowness[finite]
    assert _descends(velocity), f"not monotone: {velocity}"

    v_rayleigh = rayleigh_speed(_QUAD_FAST["vp"], _QUAD_FAST["vs"])
    floor = scholte_speed(**_QUAD_FAST, **_QUAD_FLUID)
    assert np.all(velocity < v_rayleigh)
    assert np.all(velocity > floor * 0.999)


def test_slow_formation_quadrupole_is_a_usable_curve():
    """The control: in the regime it is meant for, the curve is monotone."""
    from fwap import quadrupole_dispersion

    frequencies = np.linspace(5.0e3, 60.0e3, 12)
    result = quadrupole_dispersion(frequencies, **_QUAD_SLOW, **_QUAD_FLUID, a=0.10)
    finite = np.isfinite(result.slowness)
    assert finite.sum() >= 10
    velocity = 1.0 / result.slowness[finite]
    assert np.all(np.diff(velocity) <= 1.0e-9)


# ----------------------------------------------------------------------
# The same high-frequency asymptote, for n=1 (roadmap A.1)
#
# "The wall looks flat to every azimuthal order" is the argument the n=2
# block above rests on, and it was never applied to n=1 -- the mode the
# package sells, since dipole shear is its headline product. Applying it
# closes the loosest external tie in the solver: the flexural high-f check
# was anchored to `rayleigh_speed`, which the mode does *not* approach.
#
# Measured, slow formation, a = 0.10 m: flexural velocity over the plane
# Scholte speed runs 1.0166 -> 1.00025 across 10-400 kHz, monotone. Over
# the Rayleigh speed it settles at 0.908 -- a fixed 9 % offset, not a
# limit. That is why the old check needed rel=0.10 and used 8.3 % of it:
# the tolerance was absorbing the wrong reference, not solver error.
#
# Slow formations only, deliberately -- see
# `test_the_fast_formation_high_frequency_limit_is_scholte_too` for the
# fast case, which needed the sub-fluid continuation before it could be
# asked at all. Until that existed, the search was confined to
# `(V_f, V_S)`, the fundamental had left through `V_f` long before
# 50 kHz, and what came back was a higher-order mode accumulating at
# `V_f` -- converging tidily to the wrong limit.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("rock", [_QUAD_SLOW, _QUAD_MID], ids=["slow", "mid"])
def test_flexural_converges_to_the_plane_scholte_speed(rock):
    """n=1 approaches the same plane-interface limit as n=0 and n=2.

    `scholte_speed` solves a plane fluid/solid interface problem -- no
    Bessel functions, no borehole radius, no azimuthal order -- so this is
    an external check rather than the solver agreeing with itself. It is
    also the tightest tie the flexural mode has: convergence, not
    proximity, and to 1e-3 rather than the 10 % the Rayleigh comparison
    could manage.
    """
    from fwap import flexural_dispersion, scholte_speed

    reference = scholte_speed(**rock, **_QUAD_FLUID)
    frequencies = np.array([50.0e3, 100.0e3, 200.0e3, 400.0e3])
    result = flexural_dispersion(frequencies, **rock, **_QUAD_FLUID, a=0.10)
    assert np.all(np.isfinite(result.slowness))

    error = np.abs(1.0 / result.slowness / reference - 1.0)
    assert np.all(np.diff(error) < 0.0)  # converging, not merely close
    assert error[-1] < 1.0e-3


def test_flexural_does_not_converge_to_the_rayleigh_speed():
    """The negative half, and the reason the old anchor was wrong.

    A vacuum-loaded Rayleigh wave is the wrong limit for a fluid-filled
    borehole: fluid loading slows the surface wave to the Scholte speed,
    which here is 9 % below Rayleigh. So the ratio against Rayleigh must
    *stop* at a constant offset instead of approaching one, and pinning
    that keeps anyone from re-anchoring to it.
    """
    from fwap import flexural_dispersion, rayleigh_speed

    v_rayleigh = rayleigh_speed(_QUAD_SLOW["vp"], _QUAD_SLOW["vs"])
    frequencies = np.array([100.0e3, 400.0e3])
    result = flexural_dispersion(frequencies, **_QUAD_SLOW, **_QUAD_FLUID, a=0.10)

    ratio = 1.0 / result.slowness / v_rayleigh
    assert np.all(ratio < 0.95), "fluid loading must hold it well below Rayleigh"
    # Flat: the residual change over two octaves is far smaller than the gap.
    assert abs(ratio[-1] - ratio[0]) < 0.01 * (1.0 - ratio[-1])


def test_all_three_azimuthal_orders_share_the_high_frequency_limit():
    """n=0, n=1 and n=2 must land on the same flat-wall answer.

    Three different modal determinants, one plane-interface limit. A
    branch error in any single order shows up here as a disagreement,
    which no per-mode check can see.
    """
    from fwap import flexural_dispersion, quadrupole_dispersion, stoneley_dispersion

    high = np.array([400.0e3])
    kwargs = dict(**_QUAD_SLOW, **_QUAD_FLUID, a=0.10)
    stoneley = 1.0 / stoneley_dispersion(high, **kwargs).slowness[0]
    flexural = 1.0 / flexural_dispersion(high, **kwargs).slowness[0]
    quadrupole = 1.0 / quadrupole_dispersion(high, **kwargs).slowness[0]

    assert flexural == pytest.approx(stoneley, rel=1e-4)
    assert quadrupole == pytest.approx(stoneley, rel=1e-4)


# ---------------------------------------------------------------------------
# Branch selection for the pseudo-Rayleigh marcher.
#
# These replace five tests that pinned the *old* seeding heuristic's defects:
# the tracked branch depended on the grid's top frequency, and two silent
# all-NaN failures. The seed is now enumerated, so the same assertions run
# the other way round -- they check the answer no longer moves. The measured
# numbers are carried over from the defect tests unchanged, which is what
# makes them evidence that behaviour changed in the intended direction.
# ---------------------------------------------------------------------------

_LEAKY_FAST = {"vp": 4000.0, "vs": 2300.0, "rho": 2500.0}
_LEAKY_BRINE = {"vf": 1500.0, "rho_f": 1000.0}


def _pr_at(probe, top, *, branch=0, a=0.10, n=400):
    """Run the solver on a 2 kHz-to-``top`` grid and read off ``probe``."""
    from fwap import pseudo_rayleigh_modal_dispersion

    grid = np.sort(np.unique(np.concatenate([np.linspace(2.0e3, top, n), [probe]])))
    mode = pseudo_rayleigh_modal_dispersion(
        grid, **_LEAKY_FAST, **_LEAKY_BRINE, a=a, branch=branch
    )
    j = int(np.argmin(np.abs(grid - probe)))
    return 1.0 / mode.slowness[j], mode.attenuation_per_meter[j]


# Each branch's leaky segment lives just *below* that branch's trapped
# cut-off, so the probe frequency has to be chosen per branch. On
# 4000/2300/2500, water, a = 0.10 m the first four cut-offs are 7578,
# 13974, 23398 and 33179 Hz; at a = 0.07 m they are 10827, 19963, 33428
# and 47398 Hz. Everything below probes inside those bands.
_PR_CUTOFF_10 = (7578.43, 13974.30, 23398.44, 33178.71)


def test_pseudo_rayleigh_does_not_depend_on_the_grid_top_frequency():
    """The mode returned is a function of the medium, not of the request.

    This is the regression that motivated seeding from the branch's own
    trapped cut-off. With the original heuristic seed the answer at
    30 kHz switched from 2486 m/s to 2952 m/s somewhere between a 55 kHz
    and a 60 kHz grid top -- silently, and to a different but equally
    genuine root. Enumerating at the grid top appeared to fix that, but
    only because the contaminated radiation branch littered the window
    with roots; once corrected, the grid top decided *which trapped
    branch's* continuation came back under the label ``branch=0``.

    Seeding at the cut-off makes the coupling structurally impossible
    rather than merely small: the ladder starts at a frequency the
    medium sets, which the grid top cannot reach. What is left is
    rounding -- the caller's frequencies are members of the ladder, so
    a different request changes the step pattern and the marched path
    differs in the last bits.
    """
    reference = _pr_at(7.0e3, 40.0e3)
    assert np.isfinite(reference[0])
    for top in (8.0e3, 32.0e3, 55.0e3, 60.0e3, 80.0e3, 100.0e3):
        c, att = _pr_at(7.0e3, top)
        assert c == pytest.approx(reference[0], rel=1.0e-12), top
        assert att == pytest.approx(reference[1], rel=1.0e-11), top


def test_pseudo_rayleigh_branches_are_distinct_and_both_are_genuine_roots():
    """``branch`` selects radial order, and every order is a real root.

    Each branch is probed inside its own band -- they do not overlap in
    frequency, which is itself the point: a leaky branch exists only
    below its trapped cut-off, so asking for ``branch=0`` at 30 kHz
    correctly returns NaN rather than some other branch's root.
    """
    from fwap.cylindrical_solver._leaky import _modal_determinant_n0_complex

    def is_root(kz: complex, probe: float) -> bool:
        omega = 2.0 * np.pi * probe

        def det(z):
            return _modal_determinant_n0_complex(
                z,
                omega,
                **_LEAKY_FAST,
                **_LEAKY_BRINE,
                a=0.10,
                leaky_p=False,
                leaky_s=True,
            )

        radius = 0.01 * abs(kz)
        ring = float(
            np.median(
                [
                    abs(det(kz + radius * np.exp(1j * t)))
                    for t in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
                ]
            )
        )
        return abs(det(kz)) < 1.0e-9 * ring

    speeds = []
    for branch, probe in ((0, 7.0e3), (1, 13.0e3), (2, 22.0e3)):
        c, att = _pr_at(probe, 40.0e3, branch=branch)
        assert np.isfinite(c), (branch, probe)
        assert att > 0.0
        omega = 2.0 * np.pi * probe
        assert is_root(complex(omega / c, att), probe), branch
        speeds.append(c)

    # Higher radial orders radiate harder at the same distance below
    # their own cut-off, so the attenuation ordering is monotone.
    attenuations = [
        _pr_at(p, 40.0e3, branch=b)[1]
        for b, p in ((0, 7.0e3), (1, 13.0e3), (2, 22.0e3))
    ]
    assert attenuations == sorted(attenuations), attenuations

    # A branch asked for outside its own band is NaN, not another root.
    assert not np.isfinite(_pr_at(30.0e3, 40.0e3, branch=0)[0])


def test_pseudo_rayleigh_recovers_the_band_on_a_coarse_grid():
    """A coarse grid gives a coarse answer, not silence.

    A 0.07 m borehole over 4-30 kHz used to return *nothing at all* at
    60 samples while recovering the band at 80 -- a total, silent
    failure of the heuristic seed's first step. Neither grid is silent
    now, and their sample counts stand in the ratio of the grids
    because the marcher runs on its own ladder and the caller's grid
    only decides where it is sampled.
    """
    from fwap import pseudo_rayleigh_modal_dispersion

    medium = dict(**_LEAKY_FAST, **_LEAKY_BRINE, a=0.07)
    coarse = pseudo_rayleigh_modal_dispersion(np.linspace(4.0e3, 30.0e3, 60), **medium)
    fine = pseudo_rayleigh_modal_dispersion(np.linspace(4.0e3, 30.0e3, 120), **medium)

    n_coarse = int(np.isfinite(coarse.slowness).sum())
    n_fine = int(np.isfinite(fine.slowness).sum())
    assert n_coarse > 0 and n_fine > 0
    assert n_fine == pytest.approx(2 * n_coarse, abs=2)


def test_pseudo_rayleigh_sub_window_agrees_with_the_full_band():
    """Asking about part of a band gives the same answer as asking about all of it.

    Requesting only a slice used to return nothing, while a wide grid
    converged across that whole interval. Now the narrow request not
    only converges but reproduces the wide one's values to rounding --
    the ladder the marcher walks starts from the same cut-off in both
    cases, and differs only in which of the caller's frequencies were
    spliced into it.
    """
    from fwap import pseudo_rayleigh_modal_dispersion

    medium = dict(**_LEAKY_FAST, **_LEAKY_BRINE, a=0.10)
    band = np.arange(5.5e3, 7.5e3 + 1.0, 25.0)

    narrow = pseudo_rayleigh_modal_dispersion(band, **medium)
    assert np.isfinite(narrow.slowness).sum() > 0.5 * band.size

    wide_grid = np.arange(2.0e3, 40.0e3 + 1.0, 25.0)
    wide = pseudo_rayleigh_modal_dispersion(wide_grid, **medium)
    shared = np.searchsorted(wide_grid, band)
    live = np.isfinite(narrow.slowness)
    assert np.array_equal(live, np.isfinite(wide.slowness[shared]))
    assert narrow.slowness[live] == pytest.approx(
        wide.slowness[shared][live], rel=1.0e-12
    )
    assert narrow.attenuation_per_meter[live] == pytest.approx(
        wide.attenuation_per_meter[shared][live], rel=1.0e-10
    )


def test_pseudo_rayleigh_rejects_a_branch_that_does_not_exist():
    """A radial order with no cut-off at all is silence, not an error.

    The distinction moved when seeding moved. It used to be that a
    branch index beyond what the grid top exposed raised, because the
    index was into a list enumerated at that grid top. The index is now
    into the trapped branch sequence, which is a property of the medium,
    so "branch 40 of this borehole" is not a malformed request -- it is
    a request for a mode whose cut-off the search cannot reach, and the
    honest answer is an all-NaN curve. Only a *negative* index is
    malformed.
    """
    from fwap import pseudo_rayleigh_modal_dispersion

    medium = dict(**_LEAKY_FAST, **_LEAKY_BRINE, a=0.10)
    unreachable = pseudo_rayleigh_modal_dispersion(
        np.linspace(2.0e3, 15.0e3, 50), **medium, branch=40
    )
    assert not np.any(np.isfinite(unreachable.slowness))

    with pytest.raises(ValueError, match="branch must be non-negative"):
        pseudo_rayleigh_modal_dispersion(
            np.linspace(2.0e3, 40.0e3, 50), **medium, branch=-1
        )


def test_pseudo_rayleigh_returns_all_nan_below_the_lowest_cutoff():
    """No branch exists at the top of the band, so there is nothing to march.

    This is the one case where an all-NaN curve is the right answer rather
    than a search failure: at 100-200 Hz the wavelength is two orders above
    the borehole radius and no leaky mode has reached its cutoff. The
    enumeration finds nothing and the solver says so, instead of raising --
    "this mode does not propagate here" is a physical statement, and it is
    distinguished from an out-of-range ``branch``, which does raise.
    """
    from fwap import pseudo_rayleigh_modal_dispersion
    from fwap.cylindrical_solver._leaky import _enumerate_leaky_roots_n0

    medium = dict(**_LEAKY_FAST, **_LEAKY_BRINE, a=0.10)
    assert _enumerate_leaky_roots_n0(2.0 * np.pi * 200.0, **medium) == []

    freq = np.linspace(100.0, 200.0, 5)
    mode = pseudo_rayleigh_modal_dispersion(freq, **medium)
    assert mode.slowness.shape == freq.shape
    assert not np.any(np.isfinite(mode.slowness))
    assert not np.any(np.isfinite(mode.attenuation_per_meter))


def test_leaky_root_enumeration_count_is_insensitive_to_scan_density():
    """The seed scan must not be the thing that decides how many modes exist.

    If the recovered count moved with the scan resolution, "how many
    branches are there" would be an artefact of the search rather than a
    property of the medium -- and ``branch=1`` would mean different things
    at different densities. Checked over a 3.3x span of seed density.
    """
    from fwap.cylindrical_solver._leaky import _enumerate_leaky_roots_n0

    medium = dict(**_LEAKY_FAST, **_LEAKY_BRINE, a=0.10)
    for freq in (15.0e3, 30.0e3, 60.0e3):
        omega = 2.0 * np.pi * freq
        counts = {
            len(_enumerate_leaky_roots_n0(omega, **medium, n_re=n_re, n_im=n_im))
            for n_re, n_im in ((24, 5), (40, 8), (80, 16))
        }
        assert len(counts) == 1, (freq, counts)

    # ...and the count grows with frequency, as radial orders pass cutoff.
    counts = [
        len(_enumerate_leaky_roots_n0(2.0 * np.pi * f, **medium))
        for f in (15.0e3, 30.0e3, 60.0e3)
    ]
    assert counts == sorted(counts) and counts[0] < counts[-1], counts


def test_pseudo_rayleigh_is_independent_of_grid_density_where_it_converges():
    """The flip side: once it converges, refinement changes nothing.

    Halving the step reproduces the attenuation to near machine
    precision, because the marcher walks a ladder anchored at the
    branch's cut-off and the caller's grid only decides where it is
    sampled -- and, in the last bits, which extra rungs it has.
    """
    from fwap import pseudo_rayleigh_modal_dispersion

    medium = dict(**_LEAKY_FAST, **_LEAKY_BRINE, a=0.10)
    coarse_grid = np.arange(4.0e3, 8.0e3 + 1.0, 50.0)
    fine_grid = np.arange(4.0e3, 8.0e3 + 1.0, 25.0)

    coarse = pseudo_rayleigh_modal_dispersion(coarse_grid, **medium)
    fine = pseudo_rayleigh_modal_dispersion(fine_grid, **medium)
    shared = np.searchsorted(fine_grid, coarse_grid)

    ok = np.isfinite(coarse.slowness) & np.isfinite(fine.slowness[shared])
    assert ok.sum() > 40
    assert coarse.attenuation_per_meter[ok] == pytest.approx(
        fine.attenuation_per_meter[shared][ok], rel=1.0e-9
    )


# ---------------------------------------------------------------------------
# Energy balance for the leaky modes: a candidate oracle that does NOT work.
#
# `plans/learning.md` listed this as the most promising remaining candidate,
# on the reasoning that radiated power over axial power must reproduce
# Im(k_z) with no free geometry in it -- and so might explain the ~0.6 offset
# that `leaky_radiation_attenuation` leaves unexplained. It does neither, and
# these tests record why in a form that runs, so the next attempt does not
# re-derive it and mistake the result for a confirmation.
#
# The derivation is sound and the agreement is perfect. That is the problem:
# it is perfect for k_z values that are not roots either.
# ---------------------------------------------------------------------------

_EB_MEDIUM = {
    "vp": 4000.0,
    "vs": 2300.0,
    "rho": 2500.0,
    "vf": 1500.0,
    "rho_f": 1000.0,
    "a": 0.10,
}


def _fluid_energy_balance_im_kz(kz, omega, *, vf, a):
    """``Im(k_z)`` from radiated power over twice the axial power.

    Closes the energy balance over the fluid column only. The wall
    boundary conditions (``sigma_rz = 0`` and ``sigma_rr = -P``) put both
    the radiated flux at ``r = a`` and the axial flux in terms of the same
    fluid amplitude, which cancels.
    """
    from scipy import special

    f_radial = np.sqrt(kz**2 - (omega / vf) ** 2)
    i0a = complex(special.iv(0, f_radial * a))
    i1a = complex(special.iv(1, f_radial * a))
    radiated = -a * np.imag(i0a * np.conj(f_radial * i1a))

    r = np.linspace(0.0, a, 2001)
    axial = np.trapezoid(np.abs(special.iv(0, f_radial * r)) ** 2 * r, r)
    return radiated / (2.0 * kz.real * axial)


def test_fluid_energy_balance_reproduces_the_leaky_attenuation():
    """At genuine roots the balance returns Im(k_z) exactly.

    Establishes that the derivation is right, which is what makes the next
    test's result meaningful rather than a sign of a broken formula.
    """
    from fwap import pseudo_rayleigh_modal_dispersion

    # 5.0-7.5 kHz: just under the fundamental's 7578 Hz trapped
    # cut-off, which is where this branch lives once the radiation
    # branch of ``_k_or_hankel`` is the outgoing one.
    freq = np.linspace(5.0e3, 7.5e3, 12)
    mode = pseudo_rayleigh_modal_dispersion(freq, **_EB_MEDIUM)
    ok = np.isfinite(mode.slowness)
    assert ok.sum() > 8

    for f, s, att in zip(freq[ok], mode.slowness[ok], mode.attenuation_per_meter[ok]):
        omega = 2.0 * np.pi * f
        kz = complex(s * omega, att)
        predicted = _fluid_energy_balance_im_kz(
            kz, omega, vf=_EB_MEDIUM["vf"], a=_EB_MEDIUM["a"]
        )
        assert predicted == pytest.approx(att, rel=1.0e-6), f


def test_fluid_energy_balance_is_an_identity_and_so_validates_nothing():
    """...and it returns Im(k_z) for k_z values that are not roots at all.

    That is the whole finding. Closing the balance inside the fluid gives
    the divergence theorem applied to a source-free Helmholtz solution,
    which is true of *any* field of the form ``A I0(F r) exp(i k_z z)``
    with ``F^2 = k_z^2 - (omega/V_f)^2``. Nothing about the formation, and
    hence nothing about the eigenvalue condition, enters it.

    A check that cannot fail is not a check. This is the "limit that
    cannot discriminate" failure mode in `plans/learning.md`, caught by
    the rule that document states: ask what the check would do to a wrong
    answer.
    """
    omega = 2.0 * np.pi * 20.0e3
    rng = np.random.default_rng(0)

    kz_lo = omega / _EB_MEDIUM["vp"]
    kz_hi = omega / _EB_MEDIUM["vs"]
    for _ in range(8):
        kz = complex(rng.uniform(kz_lo, kz_hi), rng.uniform(0.2, 8.0))
        predicted = _fluid_energy_balance_im_kz(
            kz, omega, vf=_EB_MEDIUM["vf"], a=_EB_MEDIUM["a"]
        )
        # Arbitrary k_z, no root anywhere near it, and it still comes back.
        assert predicted == pytest.approx(kz.imag, rel=1.0e-5), kz


def test_full_energy_balance_has_no_finite_denominator():
    """Extending the balance into the formation does not rescue it.

    The obvious repair is to include the formation's axial flux, which
    would bring the outgoing-wave condition into the balance. It cannot be
    done: the leaky-S field *grows* with radius -- the standard leaky-mode
    divergence -- so the axial power integral has no finite value to
    divide by.

    Checked with the solver's own radial evaluator rather than a
    hand-rolled Hankel call, because the two Hankel kinds differ here by
    growth versus decay and picking the wrong one reverses the conclusion.
    """
    from fwap import pseudo_rayleigh_modal_dispersion
    from fwap.cylindrical_solver._bessel import _k_or_hankel

    freq = np.linspace(5.0e3, 7.5e3, 12)
    mode = pseudo_rayleigh_modal_dispersion(freq, **_EB_MEDIUM)
    j = int(np.argmin(np.abs(freq - 6.5e3)))
    omega = 2.0 * np.pi * freq[j]
    kz = complex(mode.slowness[j] * omega, mode.attenuation_per_meter[j])

    s_radial = np.sqrt(kz**2 - (omega / _EB_MEDIUM["vs"]) ** 2)
    radii = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
    magnitude = np.array(
        [abs(_k_or_hankel(0, s_radial, float(r), leaky=True)[0]) for r in radii]
    )
    # Growth is asserted from 1 m outward, not from 0.1 m. Near the
    # borehole the leaky-S field is oscillatory rather than monotone --
    # it reads 2.65, 2.23, 2.99 at 0.1, 0.5, 1.0 m -- and the divergence
    # this test is about is a far-field statement. The 0.1 m point was
    # inside that oscillation before the radiation branch was
    # corrected too; the old strict-monotone assertion passed on the
    # contaminated branch by luck.
    assert np.all(np.diff(magnitude) > 0.0), magnitude
    assert magnitude[-1] > 1.0e20 * magnitude[0], magnitude

    # The bound P wave decays, as it must -- so the growth above is the
    # leaky branch specifically, not every field in the formation.
    p_radial = np.sqrt(kz**2 - (omega / _EB_MEDIUM["vp"]) ** 2)
    p_magnitude = np.array(
        [abs(_k_or_hankel(0, p_radial, float(r), leaky=False)[0]) for r in radii]
    )
    assert np.all(np.diff(p_magnitude) < 0.0), p_magnitude


# ---------------------------------------------------------------------------
# Layer-subdivision invariance: an oracle for the layered propagator stack.
#
# Subdividing a homogeneous annulus into several adjacent layers with the same
# properties changes the *description* and not the medium, so the dispersion
# must be bit-stable. It exercises exactly the machinery the single-layer
# tests cannot reach -- interface matching and propagator composition across
# more than one boundary -- and it is independent of the physics of any one
# layer, because it compares the solver against itself under a relabelling
# that no correct implementation may notice.
#
# Note what this does NOT test: an error common to every interface cancels
# out. It is a consistency oracle, not an absolute one.
# ---------------------------------------------------------------------------

_SUB_SLOW = {"vp": 2600.0, "vs": 1300.0, "rho": 2300.0}
_SUB_FAST = {"vp": 4000.0, "vs": 2300.0, "rho": 2500.0}
_SUB_FLUID = {"vf": 1500.0, "rho_f": 1000.0}
_SUB_MUD = {"vp": 3000.0, "vs": 1600.0, "rho": 2100.0}
_SUB_STEEL = {"vp": 5860.0, "vs": 3140.0, "rho": 7800.0}
_SUB_CEMENT = {"vp": 2800.0, "vs": 1600.0, "rho": 1920.0}


def _subdivide(layers, index, fractions):
    """Replace ``layers[index]`` by adjacent copies summing to its thickness."""
    from fwap import BoreholeLayer

    target = layers[index]
    assert abs(sum(fractions) - 1.0) < 1.0e-12
    pieces = tuple(
        BoreholeLayer(
            vp=target.vp,
            vs=target.vs,
            rho=target.rho,
            thickness=target.thickness * fraction,
        )
        for fraction in fractions
    )
    return layers[:index] + pieces + layers[index + 1 :]


def test_subdividing_a_homogeneous_layer_leaves_the_dispersion_unchanged():
    """Relabelling one annulus as several must change nothing at all.

    Covered for all three azimuthal orders, for an open-hole mudcake and a
    cased steel-plus-cement stack, and splitting the inner as well as the
    outer layer of that stack -- so the invariance is checked where the
    propagator has to compose across a large impedance contrast, not only
    in the easy case.
    """
    from fwap import (
        BoreholeLayer,
        flexural_dispersion_layered,
        quadrupole_dispersion_layered,
        stoneley_dispersion_layered,
    )

    mud = (BoreholeLayer(**_SUB_MUD, thickness=0.04),)
    cased = (
        BoreholeLayer(**_SUB_STEEL, thickness=0.01),
        BoreholeLayer(**_SUB_CEMENT, thickness=0.03),
    )
    freq = np.linspace(2.0e3, 14.0e3, 13)

    cases = [
        (stoneley_dispersion_layered, _SUB_SLOW, mud, 0),
        (flexural_dispersion_layered, _SUB_SLOW, mud, 0),
        (quadrupole_dispersion_layered, _SUB_SLOW, mud, 0),
        (stoneley_dispersion_layered, _SUB_FAST, mud, 0),
        (stoneley_dispersion_layered, _SUB_SLOW, cased, 0),
        (stoneley_dispersion_layered, _SUB_SLOW, cased, 1),
    ]
    for solver, formation, layers, index in cases:
        medium = dict(**formation, **_SUB_FLUID, a=0.10)
        base = solver(freq, **medium, layers=layers).slowness
        for fractions in ((0.5, 0.5), (0.3, 0.7), (0.25, 0.35, 0.40)):
            split = solver(
                freq, **medium, layers=_subdivide(layers, index, fractions)
            ).slowness
            ok = np.isfinite(base) & np.isfinite(split)
            assert ok.sum() >= 4, (solver.__name__, layers, fractions)
            assert np.allclose(base[ok], split[ok], rtol=1.0e-12, atol=0.0), (
                solver.__name__,
                index,
                fractions,
                np.max(np.abs(split[ok] / base[ok] - 1.0)),
            )


def test_layer_subdivision_invariance_is_not_vacuous():
    """A subdivision that does not preserve the medium must be detected.

    The invariance above is only worth asserting if a wrong stack fails it.
    A thickness error of one part in ten thousand moves the slowness by
    ~3e-6 relative -- nine orders above the 1e-15 floor the correct split
    achieves -- so the check discriminates with enormous margin.
    """
    from fwap import BoreholeLayer, stoneley_dispersion_layered

    medium = dict(**_SUB_SLOW, **_SUB_FLUID, a=0.10)
    freq = np.linspace(2.0e3, 14.0e3, 13)
    layers = (BoreholeLayer(**_SUB_MUD, thickness=0.04),)
    base = stoneley_dispersion_layered(freq, **medium, layers=layers).slowness

    exact = stoneley_dispersion_layered(
        freq, **medium, layers=_subdivide(layers, 0, (0.5, 0.5))
    ).slowness
    assert np.max(np.abs(exact / base - 1.0)) < 1.0e-12

    for relative_error, floor in ((1.0e-4, 1.0e-6), (1.0e-3, 1.0e-5)):
        wrong = (
            BoreholeLayer(**_SUB_MUD, thickness=0.02),
            BoreholeLayer(**_SUB_MUD, thickness=0.02 * (1.0 + relative_error)),
        )
        got = stoneley_dispersion_layered(freq, **medium, layers=wrong).slowness
        assert np.max(np.abs(got / base - 1.0)) > floor, relative_error


def test_swapping_layer_order_is_not_an_invariance():
    """Recorded because a planning note claimed it was.

    `plans/learning.md` listed "swapping layer order should leave the
    dispersion invariant" as a candidate oracle. It is false for a
    cylindrical stack: the layers sit at *different radii*, so exchanging
    them moves material from one radius to another and changes the medium.
    Measured here at about 1 % -- far too large to be numerical, and in the
    direction physics requires.
    """
    from fwap import BoreholeLayer, stoneley_dispersion_layered

    medium = dict(**_SUB_FAST, **_SUB_FLUID, a=0.10)
    freq = np.linspace(2.0e3, 15.0e3, 8)
    inner = BoreholeLayer(**_SUB_MUD, thickness=0.02)
    outer = BoreholeLayer(vp=3500.0, vs=1900.0, rho=2300.0, thickness=0.03)

    forward = stoneley_dispersion_layered(
        freq, **medium, layers=(inner, outer)
    ).slowness
    reversed_ = stoneley_dispersion_layered(
        freq, **medium, layers=(outer, inner)
    ).slowness

    ok = np.isfinite(forward) & np.isfinite(reversed_)
    assert ok.sum() >= 6
    assert np.max(np.abs(reversed_[ok] / forward[ok] - 1.0)) > 1.0e-3


# ---------------------------------------------------------------------------
# Where the transparency invariance stops holding.
#
# Appending a layer whose properties equal the formation is a no-op on the
# medium, so it must be a no-op on the dispersion. It is -- until the radial
# dynamic range across the added layer gets large, at which point the root
# search returns finite, plausible, wrong values. These tests pin that the
# boundary exists and, importantly, establish *which* side is right using an
# oracle outside the layered solver entirely.
#
# They pin a limitation, so a future fix should make the second one fail; it
# should then be rewritten as a guarantee rather than worked around.
#
# Nothing about the size or location of the error is asserted. Both proved
# platform-dependent, and two earlier versions of these tests failed in CI by
# pinning first where the spurious root lands and then how far off it is.
#
# The existing single-layer transparency tests use a 0.005 m layer over
# 0.5-8 kHz, which is far inside the safe region -- which is why the
# limitation went unnoticed rather than being a regression.
# ---------------------------------------------------------------------------


def test_transparent_layer_is_a_no_op_while_the_dynamic_range_is_moderate():
    """Thin formation-equal layers are transparent, as they must be."""
    from fwap import BoreholeLayer, stoneley_dispersion_layered

    medium = dict(**_SUB_SLOW, **_SUB_FLUID, a=0.10)
    freq = np.linspace(2.0e3, 14.0e3, 13)
    stack = (BoreholeLayer(**_SUB_MUD, thickness=0.02),)
    base = stoneley_dispersion_layered(freq, **medium, layers=stack).slowness

    for thickness in (0.01, 0.05, 0.10):
        padded = stack + (BoreholeLayer(**_SUB_SLOW, thickness=thickness),)
        got = stoneley_dispersion_layered(freq, **medium, layers=padded).slowness
        ok = np.isfinite(base) & np.isfinite(got)
        assert ok.sum() >= 10, thickness
        assert np.max(np.abs(got[ok] / base[ok] - 1.0)) < 1.0e-10, thickness


def test_transparent_layer_stops_being_a_no_op_when_it_is_thick():
    """...and stops being one well before it returns NaN.

    A formation-equal layer is physically nothing at all, so the padded and
    plain stacks must agree to the same ~1e-15 that layer subdivision
    achieves. For thin layers they do -- the previous test asserts 1e-10.
    Somewhere above 0.1 m at 100 kHz they stop agreeing, silently: both
    calls return finite slownesses that look like dispersion curves.

    Which one is wrong is settled from outside the layered solver, with
    ``scholte_speed``: at 100 kHz the wavelength in the 2 cm mudcake is
    ~1.6 cm, so the mode rides the *innermost* layer and must approach that
    layer's Scholte speed. The plain stack does, to 0.05 %.

    **Nothing about the size or location of the error is asserted, because
    neither is a property of the physics.** Both move with thickness and
    with platform: the padded answer has been seen at 289 m/s and at
    1095 m/s for the same stack, disagreeing with the plain answer by 7 % on
    one machine and by a factor of four on another, and two earlier versions
    of this test failed in CI by pinning first the location and then the
    magnitude. What is stable, and all that is claimed here, is that
    transparency is lost somewhere in this range -- which is the finding.
    """
    from fwap import BoreholeLayer, scholte_speed, stoneley_dispersion_layered

    medium = dict(**_SUB_SLOW, **_SUB_FLUID, a=0.10)
    stack = (BoreholeLayer(**_SUB_MUD, thickness=0.02),)
    freq = np.array([100.0e3])

    plain = stoneley_dispersion_layered(freq, **medium, layers=stack).slowness[0]
    assert np.isfinite(plain)
    mud_scholte = scholte_speed(**_SUB_MUD, **_SUB_FLUID)
    assert abs((1.0 / plain) / mud_scholte - 1.0) < 1.0e-3

    # Transparency means agreement to ~1e-15; 1e-3 is twelve orders looser
    # than that and seven looser than the thin-layer test's tolerance, so
    # exceeding it cannot be round-off on any platform.
    def is_transparent(thickness):
        padded = stack + (BoreholeLayer(**_SUB_SLOW, thickness=thickness),)
        got = stoneley_dispersion_layered(freq, **medium, layers=padded).slowness[0]
        if not np.isfinite(got):
            return False  # NaN is a clean failure, but still not transparent
        return abs(got / plain - 1.0) < 1.0e-3

    thicknesses = (0.12, 0.15, 0.18, 0.20, 0.25)
    assert not all(is_transparent(t) for t in thicknesses), thicknesses


def test_genuine_thick_layers_still_converge_to_the_right_limit():
    """The breakdown is specific to a *redundant* layer, not to thick ones.

    A real altered zone with genuine contrast keeps converging to the
    innermost layer's Scholte speed at every thickness tried, and its error
    barely moves with thickness. That matters for how the limitation should
    be read: it is a defect in a construction used to *verify* the solver,
    not a defect in the configurations the solver exists to model.
    """
    from fwap import BoreholeLayer, scholte_speed, stoneley_dispersion_layered

    medium = dict(**_SUB_SLOW, **_SUB_FLUID, a=0.10)
    inner_scholte = scholte_speed(**_SUB_MUD, **_SUB_FLUID)
    freq = np.array([50.0e3, 100.0e3, 200.0e3])

    for thickness in (0.02, 0.10, 0.25):
        layers = (BoreholeLayer(**_SUB_MUD, thickness=thickness),)
        slowness = stoneley_dispersion_layered(freq, **medium, layers=layers).slowness
        assert np.all(np.isfinite(slowness)), thickness
        error = np.abs((1.0 / slowness) / inner_scholte - 1.0)
        assert np.all(error < 5.0e-3), (thickness, error)
        # ...and it tightens with frequency, as a short-wavelength limit must.
        assert np.all(np.diff(error) < 0.0), (thickness, error)


# ---------------------------------------------------------------------------
# The n=1 / n=2 cutoffs are NOT rigid-pipe fluid-column cutoffs.
#
# `plans/learning.md` proposed checking them against the rigid-pipe closed form
# with the appropriate Bessel zeros, as PR #61 did for n=0. The premise does not
# survive measurement, and the reason is structural rather than numerical:
#
#   * The n=0 mode that check applies to is the *pseudo-Rayleigh* mode, which
#     is a fluid-column resonance -- a higher-order mode of the borehole fluid,
#     perturbed by the wall.
#   * `flexural_dispersion` and `quadrupole_dispersion` return the
#     *fundamental* modes at their azimuthal orders. Those are interface modes,
#     not fluid-column ones, so there is no rigid-pipe resonance for them to be
#     compared against. The solver exposes no n=1 or n=2 counterpart of
#     pseudo-Rayleigh.
#
# What the measurement shows instead is recorded here, because a scaling law is
# falsifiable even when a closed form is not available.
# ---------------------------------------------------------------------------

_CUT_GRID = np.linspace(100.0, 40.0e3, 2000)


def _lowest_converged_frequency(solver, **medium):
    mode = solver(_CUT_GRID, **medium)
    ok = np.isfinite(mode.slowness)
    return float(_CUT_GRID[ok].min()) if ok.any() else float("nan")


def _cutoff_solvers():
    from fwap import flexural_dispersion, quadrupole_dispersion

    return (("flexural", flexural_dispersion), ("quadrupole", quadrupole_dispersion))


def test_n1_n2_cutoffs_scale_inversely_with_borehole_radius():
    """A geometric cutoff exists, and it goes as 1/a.

    This much the rigid-pipe picture would also predict, so on its own it does
    not discriminate; it is asserted because it establishes that there *is* a
    clean geometric cutoff to reason about.
    """
    for name, solver in _cutoff_solvers():
        products = []
        for a in (0.06, 0.10, 0.20):
            medium = dict(vp=2200.0, vs=1000.0, rho=2200.0, vf=1500.0, rho_f=1000.0)
            cutoff = _lowest_converged_frequency(solver, **medium, a=a)
            assert np.isfinite(cutoff), (name, a)
            products.append(cutoff * a)
        spread = max(products) / min(products) - 1.0
        assert spread < 0.05, (name, products)


def test_n1_n2_cutoffs_are_shear_controlled_not_fluid_controlled():
    """The discriminating measurement, and the one that kills the candidate.

    A rigid-pipe fluid-column cutoff is set by the *fluid*: on the n=0 form
    it carries a full power of ``V_f`` and none of ``V_S`` in the rigid limit.
    The n=1 and n=2 cutoffs behave the other way round. Measured as log-log
    sensitivities over a factor ~2 in ``V_S`` and ~1.6 in ``V_f``, the
    exponents are about 0.87 on ``V_S`` and 0.10 on ``V_f``.

    Bounds are set well clear of the measured values rather than tight to
    them, since the point is the qualitative ordering -- shear-controlled, not
    fluid-controlled -- and not the exponent itself.
    """
    for name, solver in _cutoff_solvers():
        slow_shear = _lowest_converged_frequency(
            solver,
            vp=2.2 * 700.0,
            vs=700.0,
            rho=2200.0,
            vf=1500.0,
            rho_f=1000.0,
            a=0.10,
        )
        fast_shear = _lowest_converged_frequency(
            solver,
            vp=2.2 * 1450.0,
            vs=1450.0,
            rho=2200.0,
            vf=1500.0,
            rho_f=1000.0,
            a=0.10,
        )
        slow_fluid = _lowest_converged_frequency(
            solver, vp=2200.0, vs=1000.0, rho=2200.0, vf=1200.0, rho_f=1000.0, a=0.10
        )
        fast_fluid = _lowest_converged_frequency(
            solver, vp=2200.0, vs=1000.0, rho=2200.0, vf=1900.0, rho_f=1000.0, a=0.10
        )
        assert all(
            np.isfinite(x) for x in (slow_shear, fast_shear, slow_fluid, fast_fluid)
        ), name

        exponent_vs = np.log(fast_shear / slow_shear) / np.log(1450.0 / 700.0)
        exponent_vf = np.log(fast_fluid / slow_fluid) / np.log(1900.0 / 1200.0)

        assert exponent_vs > 0.6, (name, exponent_vs)
        assert exponent_vf < 0.4, (name, exponent_vf)
        assert exponent_vs > 3.0 * exponent_vf, (name, exponent_vs, exponent_vf)


def test_n1_n2_cutoffs_do_not_match_the_rigid_pipe_closed_form():
    """Stated as a test so the candidate cannot quietly come back.

    Evaluated where the rigid-pipe form is even defined -- a fast formation,
    ``V_S > V_f`` -- the closed form and the solver disagree, and by different
    factors for the two orders, so no single constant reconciles them. In that
    regime both solvers are separately known to be defective (roadmap A.2),
    which is the second reason the comparison cannot be made to work.
    """
    from scipy.special import jnp_zeros

    medium = dict(vp=4000.0, vs=2300.0, rho=2500.0, vf=1500.0, rho_f=1000.0, a=0.10)
    rigid_pipe = {
        order: jnp_zeros(order, 1)[0]
        * medium["vf"]
        * medium["vs"]
        / (2.0 * np.pi * medium["a"] * np.sqrt(medium["vs"] ** 2 - medium["vf"] ** 2))
        for order in (1, 2)
    }

    ratios = {}
    for order, (name, solver) in zip((1, 2), _cutoff_solvers()):
        cutoff = _lowest_converged_frequency(solver, **medium)
        assert np.isfinite(cutoff), name
        ratios[order] = cutoff / rigid_pipe[order]

    # No single constant reconciles the two orders, so this is not the
    # n=0 situation of a fixed offset that could be documented and used.
    assert abs(ratios[1] / ratios[2] - 1.0) > 0.15, ratios


# ---------------------------------------------------------------------------
# Modal biorthogonality: the first oracle here that needs two solutions at once.
#
# Every earlier check evaluates something on a single mode, which is why the
# energy balance turned out vacuous -- a law evaluated on one solution in a
# region where that solution already satisfies the governing equations comes
# back exact and means nothing. Auld's waveguide reciprocity relation does not
# have that escape: it couples two *different* eigenvectors, so it cannot be
# satisfied by construction from either.
#
# The test set is real. In a fast formation the n=0 bound spectrum holds the
# Stoneley mode (c < V_f) *and* the trapped pseudo-Rayleigh modes
# (V_f < c < V_S) -- four of them at 30 kHz, all bound, all azimuthal order 0.
# Note that `stoneley_dispersion` returns only the first: its bracket stops at
# omega/V_f, so the trapped modes are found here directly from the determinant.
# ---------------------------------------------------------------------------

_BIO_MEDIUM = {
    "vp": 4000.0,
    "vs": 2300.0,
    "rho": 2500.0,
    "vf": 1500.0,
    "rho_f": 1000.0,
    "a": 0.10,
}


def _bound_n0_roots(omega, medium):
    """Every bound n=0 root at one frequency: Stoneley plus trapped modes."""
    from scipy.optimize import brentq

    from fwap.cylindrical_solver._leaky import _modal_determinant_n0_complex

    def det(kz):
        return _modal_determinant_n0_complex(
            kz, omega, **medium, leaky_p=False, leaky_s=False
        )

    roots = []
    windows = (
        (omega / medium["vs"] * 1.0001, omega / medium["vf"] * 0.9999),
        (omega / medium["vf"] * 1.0001, omega / medium["vf"] * 3.0),
    )
    for lo, hi in windows:
        grid = np.linspace(lo, hi, 3000)
        values = np.array([det(k) for k in grid])
        use_real = np.nanmax(np.abs(values.real)) >= np.nanmax(np.abs(values.imag))
        scalar = (lambda k: det(k).real) if use_real else (lambda k: det(k).imag)
        flat = values.real if use_real else values.imag
        for i in range(grid.size - 1):
            if not (np.isfinite(flat[i]) and np.isfinite(flat[i + 1])):
                continue
            if np.sign(flat[i]) != np.sign(flat[i + 1]):
                roots.append(
                    brentq(scalar, grid[i], grid[i + 1], xtol=1e-14, rtol=8.9e-16)
                )
    return sorted(roots)


def _mode_shape(kz, omega, medium):
    """Null vector of the 3x3 bound matrix, plus the radial wavenumbers."""
    from scipy import special

    mu = medium["rho"] * medium["vs"] ** 2
    a = medium["a"]
    f_r = np.sqrt(complex(kz * kz - (omega / medium["vf"]) ** 2))
    p = np.sqrt(complex(kz * kz - (omega / medium["vp"]) ** 2))
    s = np.sqrt(complex(kz * kz - (omega / medium["vs"]) ** 2))
    i0, i1 = complex(special.iv(0, f_r * a)), complex(special.iv(1, f_r * a))
    k0p, k1p = complex(special.kv(0, p * a)), complex(special.kv(1, p * a))
    k0s, k1s = complex(special.kv(0, s * a)), complex(special.kv(1, s * a))
    two_kz2 = 2.0 * kz * kz - (omega / medium["vs"]) ** 2
    matrix = np.array(
        [
            [f_r * i1 / (medium["rho_f"] * omega**2), p * k1p, kz * k1s],
            [
                -i0,
                -mu * (two_kz2 * k0p + 2.0 * p * k1p / a),
                -2.0 * kz * mu * (s * k0s + k1s / a),
            ],
            [0.0, 2.0 * kz * p * mu * k1p, mu * two_kz2 * k1s],
        ],
        dtype=complex,
    )
    _, _, vh = np.linalg.svd(matrix)
    vec = vh[-1].conj()
    # The matrix writes continuity as u_r(fluid) + u_r(solid) = 0, so the solid
    # amplitudes carry the opposite sign to the field expressions below.
    return (vec[0], -vec[1], -vec[2]), f_r, p, s


def _mode_fields(r, kz, omega, medium, shape):
    """u_r, u_z, sigma_rz, sigma_zz across the whole cross-section."""
    from scipy import special

    (amp_f, amp_p, amp_s), f_r, p, s = shape
    mu = medium["rho"] * medium["vs"] ** 2
    lam = medium["rho"] * (medium["vp"] ** 2 - 2.0 * medium["vs"] ** 2)
    two_kz2 = 2.0 * kz * kz - (omega / medium["vs"]) ** 2

    r = np.atleast_1d(r).astype(float)
    u_r = np.zeros_like(r, dtype=complex)
    u_z = np.zeros_like(r, dtype=complex)
    s_rz = np.zeros_like(r, dtype=complex)
    s_zz = np.zeros_like(r, dtype=complex)

    inside = r < medium["a"]
    if inside.any():
        ri = r[inside]
        i0, i1 = special.iv(0, f_r * ri), special.iv(1, f_r * ri)
        u_r[inside] = amp_f * f_r * i1 / (medium["rho_f"] * omega**2)
        u_z[inside] = 1j * kz * amp_f * i0 / (medium["rho_f"] * omega**2)
        s_zz[inside] = -amp_f * i0  # sigma_rz vanishes in an inviscid fluid
    if (~inside).any():
        ro = r[~inside]
        k0p, k1p = special.kv(0, p * ro), special.kv(1, p * ro)
        k0s, k1s = special.kv(0, s * ro), special.kv(1, s * ro)
        u_r[~inside] = amp_p * p * k1p + amp_s * kz * k1s
        u_z[~inside] = -1j * (kz * amp_p * k0p + amp_s * s * k0s)
        s_rz[~inside] = 1j * mu * (2.0 * kz * p * amp_p * k1p + amp_s * two_kz2 * k1s)
        s_zz[~inside] = lam * (
            omega**2 / medium["vp"] ** 2
        ) * amp_p * k0p + 2.0 * mu * (kz * (kz * amp_p * k0p + amp_s * s * k0s))
    return u_r, u_z, s_rz, s_zz


def _cross_integral(kz_m, kz_n, omega, medium, shapes, span=15.0):
    """INT (conj(u^m) . T^n . zhat) r dr, by fixed-node Gauss-Legendre.

    Fixed nodes rather than an adaptive rule on purpose: the integrand
    underflows in the evanescent tail, and adaptive quadrature spends its
    error budget out there and returns noise ~1e-4, which is large enough to
    look like a failed orthogonality relation.
    """
    a = medium["a"]
    decay = [
        np.sqrt(max(k * k - (omega / medium["vs"]) ** 2, 1e-12)) for k in (kz_m, kz_n)
    ]
    outer = a + span / min(decay)
    total = 0.0 + 0.0j
    for lo, hi, count in ((0.0, a, 300), (a, outer, 600)):
        x, weights = np.polynomial.legendre.leggauss(count)
        r = 0.5 * (hi - lo) * (x + 1.0) + lo
        u_r, u_z, _, _ = _mode_fields(r, kz_m, omega, medium, shapes[kz_m])
        _, _, s_rz, s_zz = _mode_fields(r, kz_n, omega, medium, shapes[kz_n])
        total += (
            0.5
            * (hi - lo)
            * np.sum(weights * (np.conj(u_r) * s_rz + np.conj(u_z) * s_zz) * r)
        )
    return total


def test_bound_n0_modes_satisfy_the_boundary_conditions_they_were_built_from():
    """Sanity gate on the eigenfunctions before they are used for anything.

    Written first because a sign slip in the solid amplitudes was caught here
    -- ``|du_r| / |u_r|`` came back as exactly 2.0, which is what equal and
    opposite gives -- and would otherwise have shown up as a failed
    orthogonality relation, i.e. as a fake finding about the solver.
    """
    omega = 2.0 * np.pi * 30.0e3
    roots = _bound_n0_roots(omega, _BIO_MEDIUM)
    assert len(roots) >= 3, roots

    step = 1.0e-9
    for kz in roots:
        shape = _mode_shape(kz, omega, _BIO_MEDIUM)
        u_in = _mode_fields(_BIO_MEDIUM["a"] - step, kz, omega, _BIO_MEDIUM, shape)[0]
        u_out = _mode_fields(_BIO_MEDIUM["a"] + step, kz, omega, _BIO_MEDIUM, shape)[0]
        scale = max(abs(u_in[0]), abs(u_out[0]))
        assert abs(u_in[0] - u_out[0]) / scale < 1.0e-6, kz


def test_bound_n0_modes_are_biorthogonal():
    """Auld's relation across every pair of coexisting bound modes.

    ``S_mn - conj(S_nm)`` must vanish for ``m != n``. Measured over the four
    bound modes at 30 kHz it does, to ~1e-13 relative, while the diagonal
    stays O(1) -- so this is orthogonality rather than everything being small.
    """
    omega = 2.0 * np.pi * 30.0e3
    roots = _bound_n0_roots(omega, _BIO_MEDIUM)
    shapes = {kz: _mode_shape(kz, omega, _BIO_MEDIUM) for kz in roots}
    n = len(roots)

    gram = np.array(
        [
            [
                _cross_integral(roots[i], roots[j], omega, _BIO_MEDIUM, shapes)
                for j in range(n)
            ]
            for i in range(n)
        ]
    )
    for i in range(n):
        for j in range(n):
            scale = np.sqrt(abs(gram[i, i] * gram[j, j]))
            assert scale > 0.0
            residual = abs(gram[i, j] - np.conj(gram[j, i])) / scale
            if i == j:
                continue
            assert residual < 1.0e-9, (i, j, residual)


def test_biorthogonality_check_rejects_the_wrong_bilinear_form():
    """The relation is specific, not a statement that everything is small.

    Using one term of the pairing instead of Auld's difference -- a natural
    wrong guess, and the one tried first -- leaves off-diagonals around 1e-2,
    ten orders above the tolerance the correct form meets. That gap is what
    makes the test above evidence rather than an accident of normalisation.
    """
    omega = 2.0 * np.pi * 30.0e3
    roots = _bound_n0_roots(omega, _BIO_MEDIUM)
    shapes = {kz: _mode_shape(kz, omega, _BIO_MEDIUM) for kz in roots}

    worst = 0.0
    for i in range(len(roots)):
        for j in range(len(roots)):
            if i == j:
                continue
            s_ij = _cross_integral(roots[i], roots[j], omega, _BIO_MEDIUM, shapes)
            s_ii = _cross_integral(roots[i], roots[i], omega, _BIO_MEDIUM, shapes)
            s_jj = _cross_integral(roots[j], roots[j], omega, _BIO_MEDIUM, shapes)
            worst = max(worst, abs(s_ij) / np.sqrt(abs(s_ii * s_jj)))
    assert worst > 1.0e-3, worst


# ---------------------------------------------------------------------------
# Trapped pseudo-Rayleigh modes, now public.
#
# These were found while building the biorthogonality check and were reachable
# only by scanning the determinant directly: `stoneley_dispersion` brackets
# from omega/min(V_S, V_f) upward and so returns just the Stoneley mode, and
# `pseudo_rayleigh_dispersion` covers the leaky half above V_S.
# ---------------------------------------------------------------------------

_TRAP_MEDIUM = {
    "vp": 4000.0,
    "vs": 2300.0,
    "rho": 2500.0,
    "vf": 1500.0,
    "rho_f": 1000.0,
    "a": 0.10,
}


def test_trapped_pseudo_rayleigh_lies_strictly_inside_the_trapped_window():
    """Bound between the fluid and shear velocities, and lossless.

    The defining property: both formation waves evanescent, fluid field
    oscillatory. Outside ``V_f < c < V_S`` the mode is either the Stoneley
    wave or a leaky one, and neither belongs here.
    """
    from fwap import trapped_pseudo_rayleigh_dispersion

    freq = np.linspace(8.0e3, 60.0e3, 20)
    mode = trapped_pseudo_rayleigh_dispersion(freq, **_TRAP_MEDIUM)
    assert mode.name == "trapped_pseudo_rayleigh"
    assert mode.azimuthal_order == 0
    assert mode.attenuation_per_meter is None  # bound modes do not radiate

    ok = np.isfinite(mode.slowness)
    assert ok.sum() > 15
    speeds = 1.0 / mode.slowness[ok]
    assert np.all(speeds > _TRAP_MEDIUM["vf"])
    assert np.all(speeds < _TRAP_MEDIUM["vs"])


def test_trapped_pseudo_rayleigh_branches_appear_in_order_and_descend():
    """Higher orders switch on at higher frequency; each then slows toward V_f.

    Each branch starts at its own cutoff near ``V_S`` and decreases toward
    the fluid velocity, so at any one frequency the fundamental is the
    *slowest* of the trapped modes -- which is the ordering convention the
    ``branch`` argument uses, shared with the leaky sister function.
    """
    from fwap import trapped_pseudo_rayleigh_dispersion

    freq = np.linspace(5.0e3, 60.0e3, 40)
    curves = [
        trapped_pseudo_rayleigh_dispersion(freq, **_TRAP_MEDIUM, branch=b).slowness
        for b in range(3)
    ]

    cutoffs = []
    for slowness in curves:
        ok = np.isfinite(slowness)
        assert ok.any()
        cutoffs.append(freq[ok].min())
        speeds = 1.0 / slowness[ok]
        # monotonically slowing with frequency, allowing for grid coarseness
        assert speeds[0] > speeds[-1]
    assert cutoffs[0] < cutoffs[1] < cutoffs[2], cutoffs

    # ...and at a frequency where all three exist, branch 0 is the slowest.
    high = np.array([60.0e3])
    speeds = [
        1.0
        / trapped_pseudo_rayleigh_dispersion(high, **_TRAP_MEDIUM, branch=b).slowness[0]
        for b in range(3)
    ]
    assert speeds[0] < speeds[1] < speeds[2], speeds


def test_trapped_pseudo_rayleigh_is_independent_of_the_frequency_grid():
    """No marching, so no grid coupling -- unlike the leaky sister function.

    Each frequency is solved on its own, which is why this function needs
    none of the seed-enumeration machinery `pseudo_rayleigh_dispersion`
    required. Worth pinning, because that grid independence is the whole
    reason the simpler algorithm is safe here.
    """
    from fwap import trapped_pseudo_rayleigh_dispersion

    probe = 33.0e3
    reference = None
    for base in (
        np.array([]),
        np.linspace(5.0e3, 40.0e3, 37),
        np.linspace(20.0e3, 90.0e3, 200),
        np.array([90.0e3, 12.0e3]),  # unordered once the probe is inserted
    ):
        # The probe must be an exact member of every grid: comparing the
        # *nearest* sample would compare different frequencies, which is a
        # different quantity rather than a grid effect.
        grid = np.concatenate([base, [probe]])
        mode = trapped_pseudo_rayleigh_dispersion(grid, **_TRAP_MEDIUM)
        j = int(np.flatnonzero(grid == probe)[0])
        value = mode.slowness[j]
        assert np.isfinite(value)
        if reference is None:
            reference = value
        else:
            assert value == pytest.approx(reference, rel=1.0e-12)


def test_trapped_pseudo_rayleigh_is_biorthogonal_to_the_stoneley_mode():
    """Checked against physics rather than against itself.

    The trapped modes and the Stoneley mode coexist at the same frequency
    and the same azimuthal order, so Auld's relation must hold across them.
    That ties the new function to the biorthogonality oracle above: a
    spurious root would not be orthogonal to the Stoneley mode.
    """
    from fwap import stoneley_dispersion, trapped_pseudo_rayleigh_dispersion

    omega = 2.0 * np.pi * 30.0e3
    freq = np.array([30.0e3])

    trapped = [
        trapped_pseudo_rayleigh_dispersion(freq, **_TRAP_MEDIUM, branch=b).slowness[0]
        for b in range(3)
    ]
    stoneley = stoneley_dispersion(freq, **_TRAP_MEDIUM).slowness[0]
    assert all(np.isfinite(s) for s in trapped) and np.isfinite(stoneley)

    wavenumbers = [s * omega for s in trapped] + [stoneley * omega]
    shapes = {kz: _mode_shape(kz, omega, _BIO_MEDIUM) for kz in wavenumbers}
    n = len(wavenumbers)
    gram = np.array(
        [
            [
                _cross_integral(
                    wavenumbers[i], wavenumbers[j], omega, _BIO_MEDIUM, shapes
                )
                for j in range(n)
            ]
            for i in range(n)
        ]
    )
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            scale = np.sqrt(abs(gram[i, i] * gram[j, j]))
            assert abs(gram[i, j] - np.conj(gram[j, i])) / scale < 1.0e-9, (i, j)


def test_trapped_pseudo_rayleigh_rejects_slow_formations_and_bad_input():
    """A slow formation has no trapped window at all."""
    from fwap import trapped_pseudo_rayleigh_dispersion

    freq = np.array([10.0e3])
    with pytest.raises(ValueError, match="fast formation"):
        trapped_pseudo_rayleigh_dispersion(
            freq, vp=2600.0, vs=1300.0, rho=2300.0, vf=1500.0, rho_f=1000.0, a=0.10
        )
    with pytest.raises(ValueError, match="branch must be non-negative"):
        trapped_pseudo_rayleigh_dispersion(freq, **_TRAP_MEDIUM, branch=-1)
    with pytest.raises(ValueError, match="require vp > vs"):
        trapped_pseudo_rayleigh_dispersion(
            freq, vp=2000.0, vs=2300.0, rho=2500.0, vf=1500.0, rho_f=1000.0, a=0.10
        )
    with pytest.raises(ValueError, match="freq must be strictly positive"):
        trapped_pseudo_rayleigh_dispersion(np.array([0.0]), **_TRAP_MEDIUM)

    for bad in ({"vs": -1.0}, {"rho_f": 0.0}, {"a": -0.1}):
        with pytest.raises(ValueError, match="must (all )?be positive"):
            trapped_pseudo_rayleigh_dispersion(freq, **{**_TRAP_MEDIUM, **bad})


def test_trapped_root_scan_resolution_does_not_decide_how_many_modes_exist():
    """The scan density must not set the branch count, or `branch` drifts."""
    from fwap.cylindrical_solver._leaky import (
        _modal_determinant_n0_complex,
        _scan_bound_roots,
    )

    omega = 2.0 * np.pi * 50.0e3

    def det(kz):
        return _modal_determinant_n0_complex(
            kz, omega, **_TRAP_MEDIUM, leaky_p=False, leaky_s=False
        ).real

    lo = omega / _TRAP_MEDIUM["vs"] * (1.0 + 1.0e-9)
    hi = omega / _TRAP_MEDIUM["vf"] * (1.0 - 1.0e-9)
    counts = {
        len(_scan_bound_roots(det, lo, hi, samples=n)) for n in (500, 1000, 2000, 4000)
    }
    assert len(counts) == 1, counts


# ---------------------------------------------------------------------------
# Compliant layers in the cased stack: NaN rather than nonsense.
#
# Found while starting the free-pipe / debonded item (roadmap G.2). Modelling
# debonding as a very compliant *elastic* layer does not work, and the way it
# failed was the dangerous way: the propagator's dynamic range ran past double
# precision, the 7x7 determinant became meaningless, and the bracket search
# found sign changes in the garbage and returned them as roots -- finite
# slownesses corresponding to phase velocities of 3-12 m/s, against a fluid
# velocity of 1500.
#
# The guard rejects a propagator product that cannot be formed in double
# precision and returns NaN instead. The bonded regime is untouched.
# ---------------------------------------------------------------------------

_DEBOND_FORMATION = {"vp": 4000.0, "vs": 2300.0, "rho": 2500.0}
_DEBOND_FLUID = {"vf": 1500.0, "rho_f": 1000.0}


def _debond_stack(layer_vs, thickness_m):
    from fwap import BoreholeLayer

    return (
        BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01),
        BoreholeLayer(
            vp=max(1600.0, 2.0 * layer_vs),
            vs=layer_vs,
            rho=1000.0,
            thickness=thickness_m,
        ),
        BoreholeLayer(vp=2800.0, vs=1600.0, rho=1920.0, thickness=0.03),
    )


def test_compliant_cased_layer_returns_nan_not_a_spurious_root():
    """A phase velocity of a few m/s is not a mode, and must not be reported.

    Checked across thicknesses and compliances because the failure was not
    monotone in either: some configurations produced a spurious root with no
    warning at all, which is why a warning filter would not have caught it.
    """
    from fwap import stoneley_dispersion_layered

    freq = np.linspace(2.0e3, 14.0e3, 13)
    for thickness in (1.0e-3, 0.2e-3):
        for layer_vs in (600.0, 300.0, 100.0, 30.0):
            mode = stoneley_dispersion_layered(
                freq,
                **_DEBOND_FORMATION,
                **_DEBOND_FLUID,
                a=0.10,
                layers=_debond_stack(layer_vs, thickness),
            )
            finite = mode.slowness[np.isfinite(mode.slowness)]
            # Either nothing at all, or nothing absurd. Never a 3 m/s "mode".
            if finite.size:
                assert np.all(1.0 / finite > 0.5 * _DEBOND_FLUID["vf"]), (
                    thickness,
                    layer_vs,
                    1.0 / finite,
                )


def test_compliant_cased_layer_does_not_warn():
    """...and it gets there without emitting numerical warnings.

    The overflow was raised by the matmul itself, so testing the *result* for
    finiteness cleaned up the determinant but left the warning. The guard
    checks the product's magnitude before forming it.
    """
    import warnings

    from fwap import stoneley_dispersion_layered

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for thickness in (1.0e-3, 0.2e-3):
            for layer_vs in (300.0, 100.0, 30.0):
                stoneley_dispersion_layered(
                    np.linspace(2.0e3, 14.0e3, 13),
                    **_DEBOND_FORMATION,
                    **_DEBOND_FLUID,
                    a=0.10,
                    layers=_debond_stack(layer_vs, thickness),
                )


def test_bonded_cased_stoneley_is_unaffected_by_the_guard():
    """The guard must not cost anything in the regime that works.

    Cement stiffer than the fluid converges across the whole band and the
    values are pinned, so a guard that clipped valid configurations would
    show up here rather than as a silent loss of coverage.
    """
    from fwap import BoreholeLayer, stoneley_dispersion_layered

    freq = np.linspace(2.0e3, 14.0e3, 13)
    expected = {2200.0: 1440.30, 1800.0: 1420.19, 1600.0: 1404.12, 1500.0: 1393.53}
    for cement_vs, speed_at_8k in expected.items():
        layers = (
            BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01),
            BoreholeLayer(
                vp=max(2.0 * cement_vs, 2800.0),
                vs=cement_vs,
                rho=1920.0,
                thickness=0.03,
            ),
        )
        mode = stoneley_dispersion_layered(
            freq, **_DEBOND_FORMATION, **_DEBOND_FLUID, a=0.10, layers=layers
        )
        assert np.all(np.isfinite(mode.slowness)), cement_vs
        assert 1.0 / mode.slowness[6] == pytest.approx(speed_at_8k, rel=1.0e-4)


def test_cased_stoneley_stops_being_bound_once_cement_is_softer_than_the_fluid():
    """The documented restriction on the cased dataset's prior, measured.

    `CasingCementPriors` keeps `cement_vs >= vf` on the stated grounds that
    the cased Stoneley stops being bound below it. That holds: full
    convergence at and above the fluid velocity, partial just below, none
    further down. It is the reason the shipped cased dataset spans graded
    cement quality rather than reaching debonding.
    """
    from fwap import BoreholeLayer, stoneley_dispersion_layered

    freq = np.linspace(2.0e3, 14.0e3, 13)

    def converged(cement_vs):
        layers = (
            BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01),
            BoreholeLayer(
                vp=max(2.0 * cement_vs, 2800.0),
                vs=cement_vs,
                rho=1920.0,
                thickness=0.03,
            ),
        )
        mode = stoneley_dispersion_layered(
            freq, **_DEBOND_FORMATION, **_DEBOND_FLUID, a=0.10, layers=layers
        )
        return int(np.isfinite(mode.slowness).sum())

    assert converged(1600.0) == freq.size
    assert converged(1500.0) == freq.size
    assert converged(1200.0) == 0
    assert converged(800.0) == 0


# ---------------------------------------------------------------------------
# Fluid-annulus propagator (n=0): the first piece of the microannulus model.
#
# Starting roadmap G.2 established that debonding modelled as a fluid
# microannulus is not blocked by the leaky-mode derivation the roadmap assumed
# -- a fluid contributes no shear-velocity floor to the bound-mode bracket --
# but that it does need a propagator element the module did not have. This is
# that element, verified in isolation before anything is wired to it.
#
# It is deliberately not reachable from the public API yet. The layer stack
# cannot express a fluid (`BoreholeLayer` requires vs > 0) and the global
# assembly changes shape when one is present, since a fluid annulus carries two
# amplitudes rather than four and imposes sigma_rz = 0 at both faces. Shipping a
# public layer type no solver accepts would be worse than shipping nothing.
# ---------------------------------------------------------------------------

_FA_OMEGA = 2.0 * np.pi * 8.0e3
_FA_FLUID = {"vf": 1500.0, "rho": 1000.0}


def test_fluid_annulus_e_matrix_determinant_matches_the_wronskian():
    """``det E_f(r) = -1 / (rho omega^2 r)``, with no dependence on ``F``.

    The Bessel Wronskian ``I0(x) K1(x) + I1(x) K0(x) = 1/x`` collapses the
    determinant to a closed form that does not involve the radial
    wavenumber at all. That makes it a check from outside this module: it
    holds for every ``k_z``, frequency and fluid, so a sign slip or a
    swapped Bessel order breaks it immediately.
    """
    from fwap.cylindrical_solver._cased import _fluid_layer_e_matrix_n0

    for phase_velocity in (900.0, 1200.0, 1450.0):
        kz = _FA_OMEGA / phase_velocity
        for r in (0.09, 0.11, 0.15, 0.30):
            e_matrix = _fluid_layer_e_matrix_n0(kz, _FA_OMEGA, **_FA_FLUID, r=r)
            expected = -1.0 / (_FA_FLUID["rho"] * _FA_OMEGA**2 * r)
            assert np.linalg.det(e_matrix) == pytest.approx(expected, rel=1.0e-10)


def test_fluid_annulus_e_matrix_reproduces_the_momentum_relation():
    """The physics, checked against a numerical derivative rather than algebra.

    ``u_r = (1 / rho omega^2) dp/dr`` and ``sigma_rr = -p``. Building the
    pressure from the amplitudes and differentiating it numerically tests
    the matrix against the equation it is supposed to encode, independently
    of how it was derived -- which is the check that would have caught the
    sign convention mistakes made earlier in this work.
    """
    from scipy import special

    from fwap.cylindrical_solver._cased import _fluid_layer_e_matrix_n0

    kz = _FA_OMEGA / 1200.0
    f_radial = np.sqrt(kz**2 - (_FA_OMEGA / _FA_FLUID["vf"]) ** 2)
    amplitudes = np.array([0.37, -1.9])

    def pressure(r):
        return amplitudes[0] * special.iv(0, f_radial * r) + amplitudes[1] * special.kv(
            0, f_radial * r
        )

    for r in (0.10, 0.13):
        state = _fluid_layer_e_matrix_n0(kz, _FA_OMEGA, **_FA_FLUID, r=r) @ amplitudes
        step = 1.0e-7
        dp_dr = (pressure(r + step) - pressure(r - step)) / (2.0 * step)
        assert state[0] == pytest.approx(
            dp_dr / (_FA_FLUID["rho"] * _FA_OMEGA**2), rel=1.0e-6
        )
        assert state[1] == pytest.approx(-pressure(r), rel=1.0e-12)


def test_fluid_annulus_propagator_determinant_is_the_radius_ratio():
    """``det P_f = r_inner / r_outer``, independent of everything else.

    Follows from the determinant identity above, and is the sharpest
    available statement about the propagator: no frequency, velocity,
    density or ``k_z`` appears in it.
    """
    from fwap.cylindrical_solver._cased import _fluid_layer_propagator_n0

    for phase_velocity in (900.0, 1200.0, 1450.0):
        kz = _FA_OMEGA / phase_velocity
        # Microannulus geometries (microns to a few mm) plus a deliberately
        # generous annulus. All sit inside the accuracy range characterised by
        # the next test.
        for r_inner, r_outer in (
            (0.11, 0.11002),
            (0.11, 0.1101),
            (0.10, 0.11),
            (0.10, 0.14),
        ):
            propagator = _fluid_layer_propagator_n0(
                kz, _FA_OMEGA, **_FA_FLUID, r_inner=r_inner, r_outer=r_outer
            )
            assert np.linalg.det(propagator) == pytest.approx(
                r_inner / r_outer, rel=1.0e-10
            )


def test_fluid_annulus_propagator_accuracy_is_set_by_the_bessel_span():
    """Where the element stops being usable, measured rather than assumed.

    The propagator carries ``I`` and ``K`` Bessel functions across the
    annulus, so its dynamic range grows with ``F * (r_outer - r_inner)`` and
    eventually exceeds double precision. Using the determinant identity as
    the error measure, accuracy degrades smoothly: machine precision up to a
    span of about 2, ~1e-11 by 7, and no significant digits at all by 20.

    This matters for how the element should be used, not for the case it was
    built for: a microannulus is microns to millimetres thick, which puts
    ``F * dr`` below 0.1 and nowhere near the limit. Recorded because the
    same exponential-range failure produced spurious roots elsewhere in this
    module, and because an element with an undocumented validity range is
    how that happens again.
    """
    from fwap.cylindrical_solver._cased import _fluid_layer_propagator_n0

    kz = _FA_OMEGA / 900.0
    f_radial = np.sqrt(kz**2 - (_FA_OMEGA / _FA_FLUID["vf"]) ** 2)

    def relative_error(r_inner, r_outer):
        propagator = _fluid_layer_propagator_n0(
            kz, _FA_OMEGA, **_FA_FLUID, r_inner=r_inner, r_outer=r_outer
        )
        return abs(np.linalg.det(propagator) / (r_inner / r_outer) - 1.0)

    # Comfortably inside the range: exact.
    assert f_radial * (0.14 - 0.10) < 3.0
    assert relative_error(0.10, 0.14) < 1.0e-12

    # Far outside it: the identity fails outright, which is the point.
    assert f_radial * (0.50 - 0.05) > 15.0
    assert relative_error(0.05, 0.50) > 1.0e-3


def test_fluid_annulus_propagator_composes_and_degenerates():
    """Zero thickness gives the identity; subdivision composes.

    The same invariance that the elastic layer stack satisfies, and for
    the same reason -- relabelling one annulus as two changes the
    description and not the medium. Checked here on the element itself, so
    that when the assembly is built any failure is attributable to the
    assembly.
    """
    from fwap.cylindrical_solver._cased import _fluid_layer_propagator_n0

    kz = _FA_OMEGA / 1150.0

    identity = _fluid_layer_propagator_n0(
        kz, _FA_OMEGA, **_FA_FLUID, r_inner=0.12, r_outer=0.12
    )
    assert np.allclose(identity, np.eye(2), atol=1.0e-12)

    whole = _fluid_layer_propagator_n0(
        kz, _FA_OMEGA, **_FA_FLUID, r_inner=0.10, r_outer=0.16
    )
    first = _fluid_layer_propagator_n0(
        kz, _FA_OMEGA, **_FA_FLUID, r_inner=0.10, r_outer=0.13
    )
    second = _fluid_layer_propagator_n0(
        kz, _FA_OMEGA, **_FA_FLUID, r_inner=0.13, r_outer=0.16
    )
    assert np.allclose(whole, second @ first, rtol=1.0e-9)


def test_fluid_annulus_rejects_a_propagating_radial_wavenumber():
    """The bound formulation needs ``kz > omega / vf`` in the annulus.

    Below it the radial wavenumber turns imaginary, the fluid field
    oscillates instead of decaying, and the real-valued state matrix here
    no longer describes it. Refused rather than silently returning the
    real part -- the microannulus case this element exists for is squarely
    in the bound regime, and a caller who has left it should be told.
    """
    from fwap.cylindrical_solver._cased import (
        _fluid_layer_e_matrix_n0,
        _fluid_layer_propagator_n0,
    )

    kz_oscillatory = _FA_OMEGA / (2.0 * _FA_FLUID["vf"])
    with pytest.raises(ValueError, match="not real"):
        _fluid_layer_e_matrix_n0(kz_oscillatory, _FA_OMEGA, **_FA_FLUID, r=0.12)

    kz = _FA_OMEGA / 1200.0
    with pytest.raises(ValueError, match="must be positive"):
        _fluid_layer_e_matrix_n0(kz, _FA_OMEGA, vf=-1.0, rho=1000.0, r=0.12)
    with pytest.raises(ValueError, match="r must be positive"):
        _fluid_layer_e_matrix_n0(kz, _FA_OMEGA, **_FA_FLUID, r=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        _fluid_layer_propagator_n0(kz, _FA_OMEGA, **_FA_FLUID, r_inner=0.0, r_outer=0.1)
    with pytest.raises(ValueError, match="require r_outer >= r_inner"):
        _fluid_layer_propagator_n0(kz, _FA_OMEGA, **_FA_FLUID, r_inner=0.2, r_outer=0.1)


# ---------------------------------------------------------------------------
# Microannulus global assembly (n=0): the 11x11 determinant for
# `fluid | casing | microannulus | cement | formation`.
#
# The fluid element above is verified in isolation; this section verifies the
# assembly that uses it. There is no reduction to the existing solver available
# here -- the `annulus_thickness -> 0` limit is a frictionless slip interface,
# not the bonded stack -- so the checks are:
#
# * the Krauklis (fluid-filled-crack) wave, an analytic result derived from
#   lubrication flow and quasi-static wall compliance, with no Bessel functions
#   and no cylindrical geometry in it at all. This is the strong one: it fixes
#   the absolute phase velocity of the slow root, not just its scaling, so a
#   wrong row or a swapped condition would show up as an O(1) prefactor error.
# * an independently assembled 13x13 form that keeps the annulus amplitudes as
#   explicit unknowns. Different size, different column layout -- an index slip
#   in one does not reproduce in the other.
# * subdivision invariance of the elastic blocks.
#
# The assembly is deliberately not reachable from the public API yet: choosing
# which root family a dispersion curve should follow is a separate decision,
# and this section shows there are two.
# ---------------------------------------------------------------------------

_MA_FORMATION = {"vp": 4000.0, "vs": 2300.0, "rho": 2500.0}
_MA_FLUID = {"vf": 1500.0, "rho_f": 1000.0}
_MA_ANNULUS = {"annulus_vf": 1500.0, "annulus_rho": 1000.0}
_MA_A = 0.10
_MA_CASING = BoreholeLayer(vp=5900.0, vs=3200.0, rho=7800.0, thickness=0.01)
_MA_CEMENT = BoreholeLayer(vp=2800.0, vs=1600.0, rho=1900.0, thickness=0.03)
# Blocks thick compared with the gap mode's decay length ~ 1 / k_z, so the
# half-space idealisation the Krauklis formula assumes actually applies. The
# confinement test below measures what happens when they are not.
_MA_THICK_CASING = BoreholeLayer(vp=5900.0, vs=3200.0, rho=7800.0, thickness=0.05)
_MA_THICK_CEMENT = BoreholeLayer(vp=2800.0, vs=1600.0, rho=1900.0, thickness=0.20)


def _ma_det(
    phase_velocity,
    freq,
    *,
    thickness=1.0e-4,
    inner=(_MA_CASING,),
    outer=(_MA_CEMENT,),
    **overrides,
):
    """The 11x11 microannulus determinant as a function of phase velocity."""
    from fwap.cylindrical_solver._cased import _modal_determinant_n0_microannulus

    omega = 2.0 * np.pi * freq
    kwargs = {
        **_MA_FORMATION,
        **_MA_FLUID,
        **_MA_ANNULUS,
        "a": _MA_A,
        "inner_layers": inner,
        "outer_layers": outer,
        "annulus_thickness": thickness,
    }
    kwargs.update(overrides)
    return _modal_determinant_n0_microannulus(omega / phase_velocity, omega, **kwargs)


def _ma_roots(det_fn, lo, hi, samples=800, log_grid=False):
    """Every sign change of ``det_fn`` on ``[lo, hi]``, refined by brentq.

    800 samples rather than the few thousand a first pass used: the root set
    is unchanged from 200 samples upward and under moved endpoints, measured
    below in ``test_microannulus_carries_two_root_families``.
    """
    from scipy import optimize

    grid = np.geomspace(lo, hi, samples) if log_grid else np.linspace(lo, hi, samples)
    values = np.array([det_fn(c) for c in grid])
    finite = np.isfinite(values)
    found = []
    for i in range(grid.size - 1):
        if not (finite[i] and finite[i + 1]):
            continue
        if np.sign(values[i]) != np.sign(values[i + 1]):
            found.append(
                optimize.brentq(det_fn, grid[i], grid[i + 1], xtol=1e-13, rtol=1e-15)
            )
    return found


def _krauklis_velocity(freq, thickness, inner, outer, rho_annulus):
    r"""Phase velocity of the crack (Krauklis) wave in a thin fluid gap.

    Derived outside this module and with none of its machinery. Pressure in a
    gap thin compared with the wavelength is uniform across it, so the fluid
    accelerates along the gap under its own pressure gradient,
    ``u_z = i k p / (rho_f omega^2)``, and volume conservation ties the flow to
    the opening ``Delta`` of the two walls:

        d(h u_z)/dz + Delta = 0   =>   Delta = h k^2 p / (rho_f omega^2).

    Each wall is a half-space loaded by a normal traction ``p e^{ikz}``, whose
    quasi-static surface displacement is ``p (1 - nu) / (mu k)``. Summing the
    two compliances, ``C = sum (1 - nu) / mu``, and eliminating ``p``:

        k^3 = C rho_f omega^2 / h,   c = omega / k = (omega h / (C rho_f))^{1/3}.

    Bessel-free, cylinder-free, and independent of the borehole fluid and the
    formation. Valid when the gap is thin compared with both the wavelength and
    the wall thicknesses.

    Parameters
    ----------
    freq : float
        Frequency (Hz).
    thickness : float
        Gap thickness (m).
    inner, outer : BoreholeLayer
        The two walls; only their elastic moduli are used.
    rho_annulus : float
        Gap fluid density (kg/m^3).

    Returns
    -------
    float
        Phase velocity (m/s).
    """
    compliance = 0.0
    for wall in (inner, outer):
        mu = wall.rho * wall.vs**2
        poisson = (wall.vp**2 - 2.0 * wall.vs**2) / (2.0 * (wall.vp**2 - wall.vs**2))
        compliance += (1.0 - poisson) / mu
    omega = 2.0 * np.pi * freq
    return float((omega * thickness / (compliance * rho_annulus)) ** (1.0 / 3.0))


def _ma_gap_root(freq, thickness, inner=None, outer=None, rho_annulus=1000.0):
    """The slow gap-mode root, searched on a log grid well below the fluid."""
    inner = inner or _MA_THICK_CASING
    outer = outer or _MA_THICK_CEMENT

    def det(c):
        return _ma_det(
            c,
            freq,
            thickness=thickness,
            inner=(inner,),
            outer=(outer,),
            annulus_rho=rho_annulus,
        )

    found = _ma_roots(det, 5.0, 1400.0, log_grid=True)
    assert found, "no gap-mode root found"
    return found[0]


def test_microannulus_slow_root_is_the_krauklis_crack_wave():
    """The slow root converges to the analytic crack-wave speed as the gap thins.

    This is the check that the assembly is right rather than merely
    self-consistent. ``_krauklis_velocity`` shares no code, no special
    functions and no geometry with the solver, and it predicts an absolute
    velocity: if the shear-traction-free rows at the two gap faces, or the
    ``(u_r, sigma_rr)`` pair carried across the gap, were wrong, the prefactor
    would be off by an O(1) factor rather than by the O(k h) thin-gap error.

    Measured ratios at 8 kHz, thick walls: 1.0002 at a 1 um gap, 0.998 at
    10 um, 0.983 at 100 um, 0.915 at 1 mm -- converging to the analytic value
    exactly where the derivation says it should, and departing as ``k h``
    stops being small.
    """
    for thickness, tol in ((1.0e-6, 2.0e-3), (1.0e-5, 1.0e-2), (1.0e-4, 3.0e-2)):
        measured = _ma_gap_root(8.0e3, thickness)
        predicted = _krauklis_velocity(
            8.0e3, thickness, _MA_THICK_CASING, _MA_THICK_CEMENT, 1000.0
        )
        assert measured == pytest.approx(predicted, rel=tol)

    # The approximation degrades monotonically in k h, and does so in the
    # direction the derivation implies (fluid inertia neglected across the gap
    # makes the analytic wave too fast), so the ordering is a claim too.
    ratios = [
        _ma_gap_root(8.0e3, h)
        / _krauklis_velocity(8.0e3, h, _MA_THICK_CASING, _MA_THICK_CEMENT, 1000.0)
        for h in (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3)
    ]
    assert ratios == sorted(ratios, reverse=True)
    assert ratios[-1] < 0.95


def test_microannulus_gap_mode_follows_the_crack_wave_parameter_scaling():
    """``c ~ (omega h / (C rho_f))^{1/3}`` in every variable separately.

    A prefactor match at one operating point could be a coincidence of the
    chosen numbers. Varying frequency, wall stiffness and gap-fluid density
    -- each of which enters the analytic formula differently -- makes that
    much harder: the cube-root scaling in ``omega`` and ``h``, the linear
    scaling in wall compliance, and the inverse scaling in gap-fluid density
    are four independent predictions.
    """
    soft = BoreholeLayer(vp=2000.0, vs=1100.0, rho=1800.0, thickness=0.20)
    stiff = BoreholeLayer(vp=4500.0, vs=2600.0, rho=2400.0, thickness=0.20)
    cases = [
        (500.0, 1.0e-5, _MA_THICK_CEMENT, 1000.0),
        (2.0e3, 1.0e-5, _MA_THICK_CEMENT, 1000.0),
        (2.0e4, 1.0e-5, _MA_THICK_CEMENT, 1000.0),
        (8.0e3, 1.0e-5, soft, 1000.0),
        (8.0e3, 1.0e-5, stiff, 1000.0),
        (8.0e3, 1.0e-5, _MA_THICK_CEMENT, 700.0),
        (8.0e3, 1.0e-5, _MA_THICK_CEMENT, 1500.0),
    ]
    for freq, thickness, outer, rho_annulus in cases:
        measured = _ma_gap_root(freq, thickness, outer=outer, rho_annulus=rho_annulus)
        predicted = _krauklis_velocity(
            freq, thickness, _MA_THICK_CASING, outer, rho_annulus
        )
        assert measured == pytest.approx(predicted, rel=2.0e-2)


def test_microannulus_gap_mode_needs_walls_thicker_than_its_decay_length():
    """Where the analytic oracle stops applying, and why -- measured, not assumed.

    The crack wave is confined to within ``~1 / k_z`` of the gap, so the
    half-space compliance the formula uses is only right when the walls are
    thicker than that. At 8 kHz and a 100 um gap the mode's ``1 / k_z`` is
    about 6 mm: a 2 mm casing gives 0.64 of the analytic speed, a 5 mm casing
    0.87, and the ratio saturates near 0.98 once the casing exceeds ~1 cm.
    Recorded so the oracle's domain is a measurement rather than a hope.
    """
    ratios = {}
    for casing_thickness in (0.002, 0.005, 0.01, 0.05):
        casing = BoreholeLayer(
            vp=5900.0, vs=3200.0, rho=7800.0, thickness=casing_thickness
        )
        measured = _ma_gap_root(8.0e3, 1.0e-4, inner=casing)
        ratios[casing_thickness] = measured / _krauklis_velocity(
            8.0e3, 1.0e-4, casing, _MA_THICK_CEMENT, 1000.0
        )
    assert ratios[0.002] < 0.7
    assert ratios[0.005] < ratios[0.01] < ratios[0.05]
    assert ratios[0.05] == pytest.approx(0.98, abs=0.02)


def _ma_det_13(phase_velocity, freq, thickness, inner, outer):
    """Independent 13x13 assembly keeping the annulus amplitudes explicit.

    Same physics, different bookkeeping: instead of folding the two gap
    amplitudes out through the fluid propagator, they stay as unknowns and
    ``(u_r, sigma_rr)`` is matched at both gap faces. Thirteen unknowns
    (1 borehole + 4 + 2 + 4 + 2) against thirteen equations (3 + 3 + 3 + 4).
    """
    from scipy import special

    from fwap.cylindrical_solver._cased import _fluid_layer_e_matrix_n0

    omega = 2.0 * np.pi * freq
    kz = omega / phase_velocity

    def block(layers, r_start):
        e_inner = _layer_e_matrix_n0(
            kz=kz,
            omega=omega,
            vp=layers[0].vp,
            vs=layers[0].vs,
            rho=layers[0].rho,
            r=r_start,
        )
        transfer = np.eye(4)
        r_lo = r_start
        for layer in layers:
            transfer = (
                _layer_propagator_n0(
                    kz=kz,
                    omega=omega,
                    vp=layer.vp,
                    vs=layer.vs,
                    rho=layer.rho,
                    r_inner=r_lo,
                    r_outer=r_lo + layer.thickness,
                )
                @ transfer
            )
            r_lo += layer.thickness
        return e_inner, transfer, r_lo

    e_in_a, transfer_in, r_b = block(inner, _MA_A)
    r_c = r_b + thickness
    e_out_c, transfer_out, r_d = block(outer, r_c)
    e_form = _layer_e_matrix_n0(kz=kz, omega=omega, **_MA_FORMATION, r=r_d)
    fluid_kwargs = {"vf": _MA_ANNULUS["annulus_vf"], "rho": _MA_ANNULUS["annulus_rho"]}
    e_gap_b = _fluid_layer_e_matrix_n0(kz, omega, **fluid_kwargs, r=r_b)
    e_gap_c = _fluid_layer_e_matrix_n0(kz, omega, **fluid_kwargs, r=r_c)
    state_b = transfer_in @ e_in_a
    state_d = transfer_out @ e_out_c

    f_core = np.sqrt(kz * kz - (omega / _MA_FLUID["vf"]) ** 2)
    i0 = float(special.iv(0, f_core * _MA_A))
    i1 = float(special.iv(1, f_core * _MA_A))

    # Columns: [A | inner (4) | gap (2) | outer (4) | formation (2)]
    m = np.zeros((13, 13))
    m[0, 0] = f_core * i1 / (_MA_FLUID["rho_f"] * omega**2)
    m[0, 1:5] = -e_in_a[0, :]
    m[1, 0] = -i0
    m[1, 1:5] = -e_in_a[2, :]
    m[2, 1:5] = e_in_a[3, :]
    m[3, 1:5] = state_b[0, :]
    m[3, 5:7] = -e_gap_b[0, :]
    m[4, 1:5] = state_b[2, :]
    m[4, 5:7] = -e_gap_b[1, :]
    m[5, 1:5] = state_b[3, :]
    m[6, 5:7] = e_gap_c[0, :]
    m[6, 7:11] = -e_out_c[0, :]
    m[7, 5:7] = e_gap_c[1, :]
    m[7, 7:11] = -e_out_c[2, :]
    m[8, 7:11] = e_out_c[3, :]
    for state in range(4):
        m[9 + state, 7:11] = state_d[state, :]
        m[9 + state, 11] = -e_form[state, 1]
        m[9 + state, 12] = -e_form[state, 3]
    if not np.all(np.isfinite(m)):
        return float("nan")
    return float(np.linalg.det(m))


def test_microannulus_matches_an_independent_explicit_amplitude_assembly():
    """The 11x11 and a 13x13 that never folds the gap amplitudes out agree.

    The two matrices differ in size, column layout and row content -- the
    11x11 carries ``(u_r, sigma_rr)`` across the gap with the fluid
    propagator and matches at ``r = c`` only, the 13x13 matches at both gap
    faces -- so a transposed index or a dropped sign in one does not
    reproduce in the other. Both call ``_fluid_layer_e_matrix_n0``, so this
    checks the assembly and not the fluid element; the Wronskian test above
    covers that separately.
    """
    for freq in (2.0e3, 8.0e3, 1.2e4):
        folded = _ma_roots(lambda c: _ma_det(c, freq), 200.0, 1499.0)
        explicit = _ma_roots(
            lambda c: _ma_det_13(c, freq, 1.0e-4, (_MA_CASING,), (_MA_CEMENT,)),
            200.0,
            1499.0,
        )
        assert len(folded) == len(explicit)
        for lhs, rhs in zip(folded, explicit):
            assert lhs == pytest.approx(rhs, rel=1.0e-9)


def test_microannulus_roots_are_invariant_under_layer_subdivision():
    """Splitting either elastic block into sub-layers leaves the roots fixed.

    The established invariance oracle for propagator stacks, applied to the
    two-block form: subdivision changes the number of propagator factors and
    the radii they are evaluated at, but not the physics, so any residual is
    the assembly's own bookkeeping error.
    """

    def split(layer, pieces):
        piece = layer.thickness / pieces
        return tuple(
            BoreholeLayer(vp=layer.vp, vs=layer.vs, rho=layer.rho, thickness=piece)
            for _ in range(pieces)
        )

    freq = 8.0e3
    whole = _ma_roots(lambda c: _ma_det(c, freq), 200.0, 1499.0)
    assert len(whole) == 2
    for inner_pieces, outer_pieces in ((3, 1), (1, 4), (2, 5)):
        subdivided = _ma_roots(
            lambda c: _ma_det(
                c,
                freq,
                inner=split(_MA_CASING, inner_pieces),
                outer=split(_MA_CEMENT, outer_pieces),
            ),
            200.0,
            1499.0,
        )
        assert len(subdivided) == len(whole)
        for lhs, rhs in zip(whole, subdivided):
            assert lhs == pytest.approx(rhs, rel=1.0e-9)


def test_microannulus_thin_gap_limit_is_a_slip_interface_not_the_bonded_stack():
    """``h -> 0`` converges, and not to the bonded root. That is the point.

    A vanishing gap still forbids shear traction on both faces and still lets
    ``u_z`` slip, so the limit is frictionless contact, not welded contact.
    There is therefore no reduction of this assembly to the all-elastic 7x7 to
    validate against -- which is why the checks above had to come from
    outside. Measured at 8 kHz: the Stoneley-like root converges as O(h) to
    1383.446 m/s while the bonded stack gives 1400.038 m/s, a 16.6 m/s
    (1.2 %) offset that does not shrink as the gap closes.
    """
    from fwap.cylindrical_solver._cased import _modal_determinant_n0_cased

    freq = 8.0e3
    omega = 2.0 * np.pi * freq

    def bonded(c):
        return _modal_determinant_n0_cased(
            omega / c,
            omega,
            **_MA_FORMATION,
            **_MA_FLUID,
            a=_MA_A,
            layers=(_MA_CASING, _MA_CEMENT),
        )

    bonded_root = _ma_roots(bonded, 1000.0, 1499.0)
    assert len(bonded_root) == 1

    thicknesses = [1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8]
    stoneley = [
        _ma_roots(lambda c: _ma_det(c, freq, thickness=h), 1000.0, 1499.0)[0]
        for h in thicknesses
    ]
    # Cauchy convergence: each tenfold thinning moves the root ~10x less.
    steps = np.abs(np.diff(stoneley))
    assert np.all(steps[1:] < steps[:-1] / 5.0)
    # ... but not towards the bonded root.
    offsets = [bonded_root[0] - c for c in stoneley]
    assert all(o > 15.0 for o in offsets)
    assert offsets[-1] == pytest.approx(offsets[0], rel=1.0e-3)


def test_microannulus_carries_two_root_families():
    """Two roots, not one: a Stoneley-like mode and the slow gap mode.

    Recorded because it is a trap for the root finder that would come next.
    The n=0 branch-selection defect fixed earlier in this module came from
    exactly this -- a bracket that assumed a single root, silently returning
    whichever one the grid happened to straddle. The two families are far
    apart and move in opposite directions with gap thickness: thinning the gap
    slows the crack wave towards zero while leaving the Stoneley-like root
    essentially fixed.
    """
    freq = 8.0e3
    for thickness in (1.0e-3, 1.0e-4, 1.0e-5):
        found = _ma_roots(
            lambda c: _ma_det(c, freq, thickness=thickness),
            5.0,
            1499.0,
            log_grid=True,
        )
        assert len(found) == 2
        gap_mode, stoneley = found
        assert gap_mode < 0.5 * stoneley
        assert 1300.0 < stoneley < 1400.0

    thin = _ma_roots(
        lambda c: _ma_det(c, freq, thickness=1.0e-5), 5.0, 1499.0, log_grid=True
    )
    thick = _ma_roots(
        lambda c: _ma_det(c, freq, thickness=1.0e-3), 5.0, 1499.0, log_grid=True
    )
    assert thin[0] < thick[0] / 3.0
    assert thin[1] == pytest.approx(thick[1], rel=1.0e-3)

    # "Exactly two" is only a claim about the assembly if it survives a change
    # of grid. The n=0 branch-selection defect looked like solver noise on one
    # grid and was only exposed by moving the endpoints, so both knobs are
    # turned here: sample count and window.
    reference = _ma_roots(
        lambda c: _ma_det(c, freq), 5.0, 1499.0, samples=6000, log_grid=True
    )
    for samples, lo, hi in (
        (200, 5.0, 1499.0),
        (800, 3.1, 1499.9),
        (400, 11.0, 1490.0),
    ):
        found = _ma_roots(
            lambda c: _ma_det(c, freq), lo, hi, samples=samples, log_grid=True
        )
        assert len(found) == len(reference)
        for lhs, rhs in zip(found, reference):
            assert lhs == pytest.approx(rhs, rel=1.0e-9)


def test_microannulus_returns_nan_rather_than_warning_or_raising():
    """The determinant contract: NaN everywhere it cannot be formed.

    A root scan evaluates this at arbitrary trial velocities, so an escaping
    ``RuntimeWarning`` or ``LinAlgError`` is a defect, not a diagnostic. Both
    happened while this assembly was being built: unscaled ``I_n`` overflows
    at low phase velocity, and the fluid propagator's ``solve`` hits an
    exactly singular matrix once the Bessel dynamic range collapses. Three
    regimes are covered -- above the bound floor (including the gap fluid's
    own floor, which the borehole fluid's does not imply), past the Bessel
    argument cap, and past the span where the fluid propagator is meaningful.
    """
    import warnings

    from fwap.cylindrical_solver._cased import _BESSEL_ARG_MAX

    # Above the bound floor: the borehole fluid and the gap fluid each set one.
    assert np.isnan(_ma_det(1600.0, 8.0e3))
    assert np.isnan(_ma_det(1450.0, 8.0e3, annulus_vf=1400.0))
    assert np.isfinite(_ma_det(1450.0, 8.0e3, annulus_vf=1500.0))

    # Past the Bessel argument cap: kz * r_outermost > log(sqrt(DBL_MAX)).
    r_outermost = _MA_A + _MA_CASING.thickness + 1.0e-4 + _MA_CEMENT.thickness
    c_cap = 2.0 * np.pi * 8.0e3 * r_outermost / _BESSEL_ARG_MAX
    assert np.isnan(_ma_det(c_cap * 0.9, 8.0e3))
    assert np.isfinite(_ma_det(c_cap * 1.5, 8.0e3))

    # Past the fluid propagator's usable Bessel span, where its exact
    # determinant identity fails long before its entries become non-finite.
    assert np.isnan(_ma_det(300.0, 8.0e3, thickness=0.5))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for phase_velocity in np.geomspace(0.05, 1600.0, 400):
            for thickness in (1.0e-6, 1.0e-4, 1.0e-2, 0.3):
                _ma_det(phase_velocity, 8.0e3, thickness=thickness)


def test_microannulus_rejects_configurations_it_cannot_represent():
    """Both blocks must exist, and the gap must be a real fluid gap.

    A zero-thickness inner block is refused rather than accommodated: it makes
    the two ``sigma_rz = 0`` rows at ``r = a`` and ``r = b`` the same row, so
    the determinant vanishes identically and every trial ``k_z`` looks like a
    root.
    """
    from fwap.cylindrical_solver._cased import _modal_determinant_n0_microannulus

    base = {
        **_MA_FORMATION,
        **_MA_FLUID,
        **_MA_ANNULUS,
        "a": _MA_A,
        "inner_layers": (_MA_CASING,),
        "outer_layers": (_MA_CEMENT,),
        "annulus_thickness": 1.0e-4,
    }
    omega = 2.0 * np.pi * 8.0e3
    kz = omega / 1400.0
    assert np.isfinite(_modal_determinant_n0_microannulus(kz, omega, **base))

    for bad in (
        {"inner_layers": ()},
        {"outer_layers": ()},
    ):
        with pytest.raises(ValueError, match="non-empty"):
            _modal_determinant_n0_microannulus(kz, omega, **{**base, **bad})
    with pytest.raises(ValueError, match="annulus_thickness must be positive"):
        _modal_determinant_n0_microannulus(
            kz, omega, **{**base, "annulus_thickness": 0.0}
        )
    for bad in ({"annulus_vf": 0.0}, {"annulus_rho": -1.0}):
        with pytest.raises(ValueError, match="must be positive"):
            _modal_determinant_n0_microannulus(kz, omega, **{**base, **bad})


# ---------------------------------------------------------------------------
# Public microannulus API (n=0), and the elastic-propagator identity found
# while building it.
#
# The determinant tested above carries two root families, so the public entry
# point's job is as much selection as evaluation. The rule used is structural:
# the Stoneley-like mode is the fastest bound n=0 mode, so the first sign
# change above the bound floor is it, whatever else the stack supports.
#
# The propagator identity is a separate find. `_layer_propagator_n0` has
# shipped for a long time with no check on its *value* beyond composition and
# round-trip; it turns out to have a closed-form determinant, in the same way
# the fluid element does, and that pins it against arithmetic rather than
# against itself.
# ---------------------------------------------------------------------------


def test_elastic_layer_propagator_determinant_is_the_radius_ratio_squared():
    r"""``det P = (r_inner / r_outer)^2``, with nothing else in it.

    The 4x4 elastic counterpart of the fluid annulus's Wronskian identity.
    Each of the two ``(I, K)`` Bessel pairs in ``E`` contributes one factor
    of ``1/r`` to ``det E``, so the material- and ``k_z``-dependent parts
    cancel in the ratio and the propagator's determinant is pure geometry:
    no frequency, no velocity, no density, no ``k_z``.

    That makes it a check from outside the module, and it is a sharp one --
    a swapped Bessel order or a sign slip in any of the sixteen entries
    breaks it. It had not been checked: the existing propagator tests pin
    the group law ``P(c|a) = P(c|b) P(b|a)`` and the round trip
    ``P(b|a) P(a|b) = I``, both of which a systematically wrong ``E``
    would still satisfy.
    """
    cases = [
        (5900.0, 3200.0, 7800.0),
        (2800.0, 1600.0, 1900.0),
        (4000.0, 2300.0, 2500.0),
        (2000.0, 1100.0, 1800.0),
    ]
    checked = 0
    for vp, vs, rho in cases:
        for freq in (1.0e3, 8.0e3, 4.0e4):
            omega = 2.0 * np.pi * freq
            for phase_velocity in (1400.0, 900.0, 600.0):
                if phase_velocity >= vs:
                    # Outside the bound regime for this layer: the radial
                    # wavenumber is imaginary and E is NaN by contract.
                    continue
                kz = omega / phase_velocity
                for r_inner, r_outer in ((0.10, 0.11), (0.11, 0.14), (0.09, 0.12)):
                    # Only where the propagator is representable at all -- the
                    # next test measures that boundary rather than assuming it.
                    span = np.sqrt(kz * kz - (omega / vs) ** 2) * (r_outer - r_inner)
                    if span > 2.0:
                        continue
                    propagator = _layer_propagator_n0(
                        kz=kz,
                        omega=omega,
                        vp=vp,
                        vs=vs,
                        rho=rho,
                        r_inner=r_inner,
                        r_outer=r_outer,
                    )
                    assert np.linalg.det(propagator) == pytest.approx(
                        (r_inner / r_outer) ** 2, rel=1.0e-9
                    )
                    checked += 1
    # Guard against the span filter quietly emptying the test.
    assert checked > 30


def test_elastic_layer_propagator_identity_bounds_where_it_is_representable():
    """Its accuracy tracks the Bessel span, exactly as the fluid element's does.

    Same exponential-range failure, same measure: the error in the identity
    against the dimensionless span ``s * dr`` with
    ``s = sqrt(kz^2 - (omega/vs)^2)``. Machine precision below a span of
    about 2, ~1e-9 by 5, and no significant digits by 20.

    Recorded because it is a real limit on the *propagator*, and because
    the next test shows it is emphatically not a limit on the *roots* --
    a distinction that would be easy to get backwards.
    """
    layer = BoreholeLayer(vp=2800.0, vs=1600.0, rho=1900.0, thickness=0.03)
    omega = 2.0 * np.pi * 8.0e3
    r_inner = 0.11

    def identity_error(phase_velocity):
        kz = omega / phase_velocity
        propagator = _layer_propagator_n0(
            kz=kz,
            omega=omega,
            vp=layer.vp,
            vs=layer.vs,
            rho=layer.rho,
            r_inner=r_inner,
            r_outer=r_inner + layer.thickness,
        )
        expected = (r_inner / (r_inner + layer.thickness)) ** 2
        with np.errstate(all="ignore"):
            return abs(float(np.linalg.det(propagator)) / expected - 1.0)

    def span(phase_velocity):
        kz = omega / phase_velocity
        return np.sqrt(kz * kz - (omega / layer.vs) ** 2) * layer.thickness

    assert span(1400.0) < 1.0
    assert identity_error(1400.0) < 1.0e-13
    assert 2.0 < span(600.0) < 3.0
    assert identity_error(600.0) < 1.0e-11
    assert 4.0 < span(300.0) < 6.0
    assert identity_error(300.0) < 1.0e-8
    assert span(100.0) > 12.0
    assert identity_error(100.0) > 1.0e-2


def test_microannulus_crack_root_survives_the_propagator_precision_loss():
    """The identity above fails by 1e232 where the crack root is exact to 1e-9.

    This is the measurement that stopped the identity being used as a gate on
    root quality, which was the obvious next move and would have been wrong.

    The crack wave is confined to within ``~1 / k_z`` of the gap -- 1.35 mm at
    a 1 um gap and 8 kHz. Once the outer block is much thicker than that, its
    far field cannot influence the root, and the catastrophic error lives
    entirely in the growing branch that the root condition never sees. So the
    root is fixed to 1.5e-9 across a tenfold range of cement thickness over
    which the propagator's own determinant identity degrades from 1e0 to
    1e232.

    The lesson generalises past this module: a conditioning measure on an
    intermediate quantity bounds *that quantity*, and says nothing on its own
    about a root computed from it.
    """
    from fwap.cylindrical_solver._cased import _modal_determinant_n0_microannulus

    freq, thickness = 8.0e3, 1.0e-6
    omega = 2.0 * np.pi * freq
    casing = BoreholeLayer(vp=5900.0, vs=3200.0, rho=7800.0, thickness=0.05)

    def crack_root(cement_thickness):
        cement = BoreholeLayer(
            vp=2800.0, vs=1600.0, rho=1900.0, thickness=cement_thickness
        )

        def det(c):
            return _modal_determinant_n0_microannulus(
                omega / c,
                omega,
                **_MA_FORMATION,
                **_MA_FLUID,
                a=_MA_A,
                inner_layers=(casing,),
                annulus_vf=1500.0,
                annulus_rho=1000.0,
                annulus_thickness=thickness,
                outer_layers=(cement,),
            )

        found = _ma_roots(det, 20.0, 1400.0, samples=1200, log_grid=True)
        return found[0]

    def identity_error(cement_thickness, root):
        cement = BoreholeLayer(
            vp=2800.0, vs=1600.0, rho=1900.0, thickness=cement_thickness
        )
        propagator = _layer_propagator_n0(
            kz=omega / root,
            omega=omega,
            vp=cement.vp,
            vs=cement.vs,
            rho=cement.rho,
            r_inner=0.15,
            r_outer=0.15 + cement_thickness,
        )
        expected = (0.15 / (0.15 + cement_thickness)) ** 2
        with np.errstate(all="ignore"):
            return abs(float(np.linalg.det(propagator)) / expected - 1.0)

    thicknesses = (0.02, 0.05, 0.10, 0.20)
    roots = [crack_root(t) for t in thicknesses]
    errors = [identity_error(t, r) for t, r in zip(thicknesses, roots)]

    # The root does not move ...
    for root in roots:
        assert root == pytest.approx(roots[-1], rel=1.0e-8)
    # ... while the propagator it is computed through stops meaning anything.
    assert errors[0] > 1.0
    assert errors[-1] > 1.0e100
    # Deliberately no ordering assertion on the errors in between. Once the
    # identity is destroyed its value is arbitrary: a first pass asserted the
    # sequence was monotonic, which held locally and failed on CI, where the
    # 0.05 m case returned 1.0 against 1e38 here. Asserting the ordering of
    # garbage is the same mistake as asserting where a spurious root lands.


def _ma_public(freq, thickness=1.0e-4, inner=None, outer=None, **overrides):
    """The public microannulus API, on the section's standard stack."""
    from fwap.cylindrical_solver import FluidAnnulus, stoneley_dispersion_microannulus

    kwargs = {
        **_MA_FORMATION,
        **_MA_FLUID,
        "a": _MA_A,
        "inner_layers": inner or (_MA_CASING,),
        "outer_layers": outer or (_MA_CEMENT,),
        "annulus": FluidAnnulus(vf=1500.0, rho=1000.0, thickness=thickness),
    }
    kwargs.update(overrides)
    return stoneley_dispersion_microannulus(np.asarray(freq, dtype=float), **kwargs)


def test_microannulus_public_api_reproduces_the_determinant_root():
    """The API returns the fastest bound root of the determinant it wraps.

    Checked against an independent scan rather than against a stored number,
    so it pins the selection rule and not just today's arithmetic.
    """
    freqs = np.array([2.0e3, 5.0e3, 8.0e3, 1.2e4])
    mode = _ma_public(freqs)
    assert mode.name == "Stoneley"
    assert mode.azimuthal_order == 0
    assert mode.attenuation_per_meter is None
    assert mode.freq.shape == freqs.shape

    for i, freq in enumerate(freqs):
        scanned = _ma_roots(lambda c: _ma_det(c, freq), 200.0, 1499.0)
        assert 1.0 / mode.slowness[i] == pytest.approx(max(scanned), rel=1.0e-9)


def test_microannulus_public_api_returns_the_stoneley_family_not_the_crack_wave():
    """The structural claim the selection rule rests on.

    The Stoneley-like root sits just below the borehole-fluid velocity at
    every gap thickness, while the crack wave moves over an order of
    magnitude with it. If the rule ever picked the wrong family the returned
    velocity would collapse towards the crack wave as the gap thins; instead
    it is fixed to within 0.06 % over three decades of thickness.
    """
    freq = np.array([8.0e3])
    velocities = {}
    for thickness in (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6):
        velocities[thickness] = float(1.0 / _ma_public(freq, thickness).slowness[0])

    for thickness, velocity in velocities.items():
        assert 1300.0 < velocity < 1500.0, thickness
        # Far above every crack-wave velocity measured for these gaps.
        assert velocity > 2.0 * _krauklis_velocity(
            8.0e3, thickness, _MA_CASING, _MA_CEMENT, 1000.0
        )
    spread = max(velocities.values()) / min(velocities.values()) - 1.0
    assert spread < 1.0e-3


def test_microannulus_public_api_is_grid_and_resolution_independent():
    """No frequency marching, so the answer cannot depend on the grid.

    Both knobs are turned: the frequency grid a caller passes, and the scan
    resolution. The first is the property whose absence caused the ``n=0``
    branch-selection defect; the second is what makes the default ``samples``
    a detail rather than a tuning parameter.
    """
    probe = 8.0e3
    # The probe has to sit *exactly* on every grid, or this compares different
    # frequencies and passes for the wrong reason.
    dense = np.sort(np.append(np.linspace(1.0e3, 2.0e4, 41), probe))
    sparse = np.array([5.0e3, probe, 1.5e4])
    assert probe in dense and probe in sparse

    from_dense = float(_ma_public(dense).slowness[list(dense).index(probe)])
    from_sparse = float(_ma_public(sparse).slowness[1])
    alone = float(_ma_public(np.array([probe])).slowness[0])
    assert from_dense == pytest.approx(alone, rel=1.0e-12)
    assert from_sparse == pytest.approx(alone, rel=1.0e-12)

    for samples in (120, 250, 400, 900):
        got = float(_ma_public(np.array([probe]), samples=samples).slowness[0])
        assert got == pytest.approx(alone, rel=1.0e-9)


def test_microannulus_public_api_thin_gap_does_not_reach_the_bonded_stack():
    """Through the public API, the slip-interface limit is still 1.2 % off.

    The same claim the determinant-level test makes, repeated here because
    this is the surface a caller sees and the temptation to read a thin gap
    as a bonded stack lives at this level, not inside the assembly.
    """
    from fwap.cylindrical_solver import stoneley_dispersion_layered

    freq = np.array([8.0e3])
    bonded = float(
        1.0
        / stoneley_dispersion_layered(
            freq,
            **_MA_FORMATION,
            **_MA_FLUID,
            a=_MA_A,
            layers=(_MA_CASING, _MA_CEMENT),
        ).slowness[0]
    )
    debonded = [
        float(1.0 / _ma_public(freq, thickness).slowness[0])
        for thickness in (1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8)
    ]
    steps = np.abs(np.diff(debonded))
    assert np.all(steps[1:] < steps[:-1] / 5.0)
    offsets = [bonded - c for c in debonded]
    assert all(o > 15.0 for o in offsets)
    assert offsets[-1] == pytest.approx(offsets[0], rel=1.0e-3)


def test_microannulus_public_api_validates_its_inputs():
    """Including the degenerate stacks that would silently produce nonsense."""
    from fwap.cylindrical_solver import FluidAnnulus, stoneley_dispersion_microannulus

    good = {
        **_MA_FORMATION,
        **_MA_FLUID,
        "a": _MA_A,
        "inner_layers": (_MA_CASING,),
        "outer_layers": (_MA_CEMENT,),
        "annulus": FluidAnnulus(vf=1500.0, rho=1000.0, thickness=1.0e-4),
    }
    freq = np.array([8.0e3])
    assert np.isfinite(stoneley_dispersion_microannulus(freq, **good).slowness[0])

    for bad, match in (
        ({"inner_layers": ()}, "non-empty"),
        ({"outer_layers": ()}, "non-empty"),
        ({"vs": -1.0}, "must all be positive"),
        ({"vp": 1000.0}, "require vp > vs"),
        ({"vf": 0.0}, "vf and rho_f must be positive"),
        ({"a": 0.0}, "a must be positive"),
        ({"samples": 1}, "samples must be at least 2"),
        ({"annulus": FluidAnnulus(vf=0.0, rho=1000.0, thickness=1e-4)}, "positive"),
        ({"annulus": FluidAnnulus(vf=1500.0, rho=1000.0, thickness=0.0)}, "positive"),
        ({"annulus": "not an annulus"}, "must be a FluidAnnulus"),
    ):
        with pytest.raises(ValueError, match=match):
            stoneley_dispersion_microannulus(freq, **{**good, **bad})

    with pytest.raises(ValueError, match="freq must be strictly positive"):
        stoneley_dispersion_microannulus(np.array([0.0]), **good)


def test_fluid_annulus_is_a_distinct_type_from_a_soft_layer():
    """A gap is not a limiting case of an elastic layer, and the API says so.

    `BoreholeLayer` cannot express a fluid -- it requires ``vs > 0`` -- and
    the two are not interchangeable even in the limit, because a compliant
    solid drags the bound-mode bracket floor down with its shear velocity
    while a fluid gap's floor is its acoustic velocity. That is the whole
    reason this configuration is reachable, so the type separation is load
    bearing rather than cosmetic.
    """
    from fwap.cylindrical_solver import FluidAnnulus, _validate_fluid_annulus

    annulus = FluidAnnulus(vf=1500.0, rho=1000.0, thickness=1.0e-4)
    _validate_fluid_annulus(annulus)
    assert (annulus.vf, annulus.rho, annulus.thickness) == (1500.0, 1000.0, 1.0e-4)
    assert annulus == FluidAnnulus(vf=1500.0, rho=1000.0, thickness=1.0e-4)
    with pytest.raises(AttributeError):
        annulus.thickness = 2.0e-4  # type: ignore[misc]

    with pytest.raises(ValueError, match="must be a FluidAnnulus"):
        _validate_fluid_annulus(
            BoreholeLayer(vp=1500.0, vs=1.0, rho=1000.0, thickness=1.0e-4)  # type: ignore[arg-type]
        )

    # The floor a fluid gap sets is its acoustic velocity, not a shear one.
    from fwap.cylindrical_solver import _microannulus_kz_window

    omega = 2.0 * np.pi * 8.0e3
    kz_lo, _ = _microannulus_kz_window(
        omega,
        vs=2300.0,
        rho=2500.0,
        vf=1500.0,
        rho_f=1000.0,
        inner_layers=(_MA_CASING,),
        annulus=annulus,
        outer_layers=(_MA_CEMENT,),
    )
    assert kz_lo == pytest.approx(omega / 1500.0, rel=1.0e-6)
    slow_gap = FluidAnnulus(vf=1200.0, rho=1000.0, thickness=1.0e-4)
    kz_lo_slow, _ = _microannulus_kz_window(
        omega,
        vs=2300.0,
        rho=2500.0,
        vf=1500.0,
        rho_f=1000.0,
        inner_layers=(_MA_CASING,),
        annulus=slow_gap,
        outer_layers=(_MA_CEMENT,),
    )
    assert kz_lo_slow == pytest.approx(omega / 1200.0, rel=1.0e-6)


def test_microannulus_public_api_degrades_gracefully_on_a_thick_gap():
    """A gap far outside the microannulus regime still cannot produce junk.

    At 0.30 m the gap is three hundred times a real debonding microannulus and
    the fluid propagator's Bessel span leaves its usable range, so its runtime
    determinant gate refuses more than half the scan window -- 208 of 400 grid
    points at 8 kHz. The scan skips those rather than reading a sign change
    across a NaN boundary, and the surviving root is still the Stoneley-like
    one.

    That it drifts *upward* towards the open-hole value (1416 m/s here) as the
    gap thickens is the physically right direction: a thick enough fluid
    annulus decouples the formation, and the stack starts to look like an open
    hole of the wider radius.
    """
    from fwap.cylindrical_solver import FluidAnnulus, stoneley_dispersion_microannulus

    freq = np.array([8.0e3])
    velocities = []
    for thickness in (1.0e-4, 0.05, 0.15, 0.30):
        mode = stoneley_dispersion_microannulus(
            freq,
            **_MA_FORMATION,
            **_MA_FLUID,
            a=_MA_A,
            inner_layers=(_MA_CASING,),
            annulus=FluidAnnulus(vf=1500.0, rho=1000.0, thickness=thickness),
            outer_layers=(_MA_CEMENT,),
        )
        assert np.isfinite(mode.slowness[0])
        velocities.append(float(1.0 / mode.slowness[0]))

    assert velocities == sorted(velocities)
    assert velocities[0] < 1390.0
    assert 1410.0 < velocities[-1] < 1420.0

    # Push it further and the window is refused outright rather than yielding
    # a number: at 20 kHz the same 0.30 m gap leaves no representable stretch
    # containing a root, so the scan skips every NaN and reports NaN. That is
    # the required direction of failure -- this module has twice shipped sign
    # changes read across unrepresentable regions as roots.
    refused = stoneley_dispersion_microannulus(
        np.array([2.0e4]),
        **_MA_FORMATION,
        **_MA_FLUID,
        a=_MA_A,
        inner_layers=(_MA_CASING,),
        annulus=FluidAnnulus(vf=1500.0, rho=1000.0, thickness=0.30),
        outer_layers=(_MA_CEMENT,),
    )
    assert np.isnan(refused.slowness[0])


# ---------------------------------------------------------------------------
# Public crack-wave API (n=0): the second root family of the microannulus
# determinant, and the spurious-root filter it needs.
#
# The Stoneley entry point above stops at the first sign change above the bound
# floor and so never reaches the low phase velocities where the elastic
# propagators lose precision. This one scans down to them deliberately, which
# is why it is the function that needs the filter.
# ---------------------------------------------------------------------------

# The one configuration in 270 sampled that produced spurious roots: a
# duplicated pair at 3.9499 m/s alongside the genuine 1434.78 and 72.0911.
_CW_SPURIOUS = {
    "vp": 3000.0 * 1.74,
    "vs": 3000.0,
    "rho": 2500.0,
    "vf": 1500.0,
    "rho_f": 1000.0,
    "a": 0.10,
    "inner_layers": (BoreholeLayer(vp=5900.0, vs=3200.0, rho=7800.0, thickness=0.01),),
    "outer_layers": (
        BoreholeLayer(vp=2000.0 * 1.75, vs=2000.0, rho=1900.0, thickness=0.03),
    ),
}


def _cw_public(freq, thickness=1.0e-4, inner=None, outer=None, **overrides):
    """The public crack-wave API on the thick-walled standard stack."""
    from fwap.cylindrical_solver import FluidAnnulus, crack_wave_dispersion

    kwargs = {
        **_MA_FORMATION,
        **_MA_FLUID,
        "a": _MA_A,
        "inner_layers": inner or (_MA_THICK_CASING,),
        "outer_layers": outer or (_MA_THICK_CEMENT,),
        "annulus": FluidAnnulus(vf=1500.0, rho=1000.0, thickness=thickness),
    }
    kwargs.update(overrides)
    return crack_wave_dispersion(np.asarray(freq, dtype=float), **kwargs)


def test_crack_wave_api_reproduces_the_analytic_crack_wave_speed():
    """The headline check, and it is an absolute one.

    ``_krauklis_velocity`` shares no code, no special functions and no geometry
    with the solver, and the API's scan window is not derived from it -- the
    window runs from the determinant's representability limit to the bound
    floor -- so this stays an independent check rather than a self-confirming
    one.

    Ratios at 8 kHz on thick walls: 1.0002 at a 1 um gap, 0.998 at 10 um,
    0.983 at 100 um, 0.915 at 1 mm. Converging where the thin-gap derivation
    says it should and departing as ``k h`` grows.
    """
    for thickness, tol in ((1.0e-6, 2.0e-3), (1.0e-5, 1.0e-2), (1.0e-4, 3.0e-2)):
        measured = float(1.0 / _cw_public(np.array([8.0e3]), thickness).slowness[0])
        predicted = _krauklis_velocity(
            8.0e3, thickness, _MA_THICK_CASING, _MA_THICK_CEMENT, 1000.0
        )
        assert measured == pytest.approx(predicted, rel=tol)


def test_crack_wave_api_follows_the_cube_root_scaling():
    """``c ~ (f h)^{1/3}`` in both variables, through the public surface.

    The exponent is what identifies the mode, and it is measured rather than
    assumed: a least-squares slope of ``log c`` against ``log h`` and against
    ``log f``, both of which should be 1/3.
    """
    thicknesses = np.array([1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3])
    by_thickness = np.array(
        [float(1.0 / _cw_public(np.array([8.0e3]), h).slowness[0]) for h in thicknesses]
    )
    slope_h = np.polyfit(np.log(thicknesses), np.log(by_thickness), 1)[0]
    assert slope_h == pytest.approx(1.0 / 3.0, abs=0.02)

    freqs = np.array([1.0e3, 2.0e3, 5.0e3, 1.0e4, 2.0e4])
    by_freq = np.array(1.0 / _cw_public(freqs, 1.0e-5).slowness)
    assert np.all(np.isfinite(by_freq))
    slope_f = np.polyfit(np.log(freqs), np.log(by_freq), 1)[0]
    assert slope_f == pytest.approx(1.0 / 3.0, abs=0.02)


def test_crack_wave_api_rejects_the_spurious_roots():
    """The configuration that motivated the filter, asserted portably.

    A raw scan of the same determinant over the same window finds the two
    genuine modes and, on some machines, a duplicated pair near 4 m/s that is
    not a mode -- sign changes read across propagators that have lost all
    precision. The public API returns the genuine crack wave either way.

    **The artefact's presence is deliberately not asserted.** A first version
    of this test required the raw scan to find four roots; that held on the
    development machine and failed on CI, which finds only the two genuine
    ones. Lost-precision results have no stable value *or existence* across
    platforms -- the same lesson the padded-layer measurements in
    ``plans/log_output.md`` are marked machine-specific for. It argues for the
    filter rather than against it: a caller cannot rely on the artefact being
    absent on their machine either.
    """
    from fwap.cylindrical_solver import FluidAnnulus, crack_wave_dispersion
    from fwap.cylindrical_solver._cased import (
        _BESSEL_ARG_MAX,
        _modal_determinant_n0_microannulus,
    )

    freq, thickness = 1.0e3, 1.0e-5
    omega = 2.0 * np.pi * freq

    def det(c):
        return _modal_determinant_n0_microannulus(
            omega / c,
            omega,
            vp=_CW_SPURIOUS["vp"],
            vs=_CW_SPURIOUS["vs"],
            rho=_CW_SPURIOUS["rho"],
            vf=_CW_SPURIOUS["vf"],
            rho_f=_CW_SPURIOUS["rho_f"],
            a=_CW_SPURIOUS["a"],
            inner_layers=_CW_SPURIOUS["inner_layers"],
            annulus_vf=1500.0,
            annulus_rho=1000.0,
            annulus_thickness=thickness,
            outer_layers=_CW_SPURIOUS["outer_layers"],
        )

    r_outermost = 0.10 + 0.01 + thickness + 0.03
    c_lo = omega * r_outermost / _BESSEL_ARG_MAX * 1.001
    # _ma_roots returns ascending phase velocity. The two genuine modes are
    # always there; anything down at the propagators' precision floor may or
    # may not be, depending on the machine.
    raw = _ma_roots(det, c_lo, 1500.0 * (1.0 - 1.0e-9), samples=800, log_grid=True)
    genuine = [r for r in raw if r > 20.0]
    assert len(genuine) == 2
    assert genuine[0] == pytest.approx(72.0911, rel=1.0e-4)
    assert genuine[1] == pytest.approx(1434.78, rel=1.0e-4)

    mode = crack_wave_dispersion(
        np.array([freq]),
        annulus=FluidAnnulus(vf=1500.0, rho=1000.0, thickness=thickness),
        **_CW_SPURIOUS,
    )
    # The second-fastest genuine root, whatever the raw scan turned up below it.
    assert float(1.0 / mode.slowness[0]) == pytest.approx(genuine[0], rel=1.0e-9)


def test_crack_wave_api_never_returns_a_lost_precision_root_across_a_sweep():
    """No sub-20 m/s answer anywhere in the parameter sweep that found the one.

    A structural claim rather than a value: the crack wave for a gap this size
    is tens to hundreds of m/s, so anything down at the propagators' precision
    floor is the failure mode, not a mode.
    """
    from fwap.cylindrical_solver import FluidAnnulus, crack_wave_dispersion

    casing = _CW_SPURIOUS["inner_layers"]
    checked = 0
    for freq in (1.0e3, 8.0e3, 2.0e4):
        for thickness in (1.0e-6, 1.0e-4, 1.0e-3):
            for formation_vs in (1700.0, 3000.0):
                mode = crack_wave_dispersion(
                    np.array([freq]),
                    vp=formation_vs * 1.74,
                    vs=formation_vs,
                    rho=2500.0,
                    vf=1500.0,
                    rho_f=1000.0,
                    a=0.10,
                    inner_layers=casing,
                    annulus=FluidAnnulus(vf=1500.0, rho=1000.0, thickness=thickness),
                    outer_layers=_CW_SPURIOUS["outer_layers"],
                )
                value = float(mode.slowness[0])
                if np.isfinite(value):
                    assert 1.0 / value > 20.0
                    checked += 1
    assert checked > 12


def test_crack_wave_api_is_grid_and_resolution_independent():
    """Same two knobs as the Stoneley API, and the filter survives both."""
    probe = 8.0e3
    dense = np.sort(np.append(np.linspace(1.0e3, 2.0e4, 6), probe))
    alone = float(_cw_public(np.array([probe]), 1.0e-5).slowness[0])
    from_dense = float(_cw_public(dense, 1.0e-5).slowness[list(dense).index(probe)])
    assert from_dense == pytest.approx(alone, rel=1.0e-12)

    for samples in (60, 150, 400):
        got = float(_cw_public(np.array([probe]), 1.0e-5, samples=samples).slowness[0])
        assert got == pytest.approx(alone, rel=1.0e-8)


def test_crack_wave_and_stoneley_apis_return_different_families():
    """Two functions, two modes, moving in opposite directions with the gap.

    This is the pair the single-root bracket would have confused. Thinning the
    gap drives the crack wave towards zero while leaving the Stoneley-like root
    essentially fixed, so the ratio between them is not a constant offset.
    """
    freq = np.array([8.0e3])
    for thickness in (1.0e-3, 1.0e-4, 1.0e-5):
        crack = float(1.0 / _cw_public(freq, thickness).slowness[0])
        stoneley = float(
            1.0
            / _ma_public(
                freq,
                thickness,
                inner=(_MA_THICK_CASING,),
                outer=(_MA_THICK_CEMENT,),
            ).slowness[0]
        )
        assert crack < 0.5 * stoneley
        assert 1300.0 < stoneley < 1500.0

    thin = float(1.0 / _cw_public(freq, 1.0e-6).slowness[0])
    thick = float(1.0 / _cw_public(freq, 1.0e-3).slowness[0])
    assert thick > 8.0 * thin


def test_crack_wave_api_validates_its_inputs():
    """Same contract as the Stoneley entry point."""
    from fwap.cylindrical_solver import FluidAnnulus, crack_wave_dispersion

    good = {
        **_MA_FORMATION,
        **_MA_FLUID,
        "a": _MA_A,
        "inner_layers": (_MA_THICK_CASING,),
        "outer_layers": (_MA_THICK_CEMENT,),
        "annulus": FluidAnnulus(vf=1500.0, rho=1000.0, thickness=1.0e-4),
    }
    freq = np.array([8.0e3])
    assert np.isfinite(crack_wave_dispersion(freq, **good).slowness[0])

    for bad, match in (
        ({"inner_layers": ()}, "non-empty"),
        ({"outer_layers": ()}, "non-empty"),
        ({"rho": 0.0}, "must all be positive"),
        ({"vp": 1000.0}, "require vp > vs"),
        ({"rho_f": -1.0}, "vf and rho_f must be positive"),
        ({"a": -0.1}, "a must be positive"),
        ({"samples": 0}, "samples must be at least 2"),
        ({"annulus": FluidAnnulus(vf=1500.0, rho=-1.0, thickness=1e-4)}, "positive"),
    ):
        with pytest.raises(ValueError, match=match):
            crack_wave_dispersion(freq, **{**good, **bad})

    with pytest.raises(ValueError, match="freq must be strictly positive"):
        crack_wave_dispersion(np.array([-1.0]), **good)


def test_crack_wave_api_reports_nan_where_no_second_root_survives():
    """A gap far outside the model yields NaN rather than a number.

    At 20 kHz a 0.30 m gap leaves no representable stretch of the window
    containing a root at all, so neither scan finds a second family and the
    result is NaN. Required direction of failure for a solver whose spurious
    roots have twice been finite and plausible-looking.
    """
    from fwap.cylindrical_solver import FluidAnnulus, crack_wave_dispersion

    mode = crack_wave_dispersion(
        np.array([2.0e4]),
        **_MA_FORMATION,
        **_MA_FLUID,
        a=_MA_A,
        inner_layers=(_MA_CASING,),
        annulus=FluidAnnulus(vf=1500.0, rho=1000.0, thickness=0.30),
        outer_layers=(_MA_CEMENT,),
    )
    assert np.isnan(mode.slowness[0])

    # There is an upper frequency limit too, and it is measured rather than
    # derived -- see
    # ``test_the_crack_wave_ceiling_is_set_by_the_propagator_product``.
    assert np.isnan(_cw_public(np.array([2.0e5]), 1.0e-4).slowness[0])


def test_the_crack_wave_ceiling_is_set_by_the_propagator_product():
    """Where the crack-wave band actually stops, and what stops it.

    This block used to assert arithmetic on a constant --
    ``_BESSEL_ARG_MAX * V_f / (2 pi r)``, about 242 kHz -- and then check
    that 1.5x that frequency returns NaN. Both halves passed and neither
    measured the solver: the real ceiling is **84 kHz**, so everything
    above it is NaN and the check could not fail.

    The ceiling is not the Bessel-argument bound either. Raising that
    constant fourfold moves it from 84 kHz to 84 kHz. What binds is the
    *product*: the determinant goes non-finite over the bottom of the scan
    window while its inputs are all still fine, and that floor climbs with
    frequency faster than the crack root does. At 84 kHz the root sits
    0.3 % above the floor; two kilohertz later it is underneath it.

    Roadmap A.5's residue is the reformulation that would lift this, and
    the numbers here are what it should be aimed at.
    """
    assert np.isfinite(_cw_public(np.array([8.4e4]), 1.0e-4).slowness[0])
    assert np.isnan(_cw_public(np.array([8.6e4]), 1.0e-4).slowness[0])


def test_raising_the_bessel_bound_does_not_raise_the_crack_wave_ceiling():
    """The experiment that says which constraint binds.

    ``_BESSEL_ARG_MAX`` caps the argument of the unscaled ``I_n``, and it
    is what the scan window's floor is computed from -- so it looks like
    the thing holding the ceiling down. It is not. Lifting it fourfold
    lets the window reach lower phase velocities and buys nothing,
    because the determinant is already non-finite there for a different
    reason: the propagator product overflows.

    Patching takes two names. The determinant reads the module global in
    ``_cased``; the driver re-imports it from the package namespace on
    every call. Patching only the first moves the guard but not the
    window, which measures nothing -- and did, on the first attempt here.
    """
    import fwap.cylindrical_solver as cylindrical_solver
    import fwap.cylindrical_solver._cased as cased

    original = cased._BESSEL_ARG_MAX

    def ceiling() -> float:
        """Highest 2 kHz step still returning a root, searched upward."""
        last = 0.0
        for f_khz in np.arange(78.0, 120.0, 2.0):
            if np.isfinite(_cw_public(np.array([f_khz * 1e3]), 1.0e-4).slowness[0]):
                last = float(f_khz)
            elif last and f_khz > last + 6.0:
                break
        return last

    baseline = ceiling()
    assert 80.0 <= baseline <= 90.0, f"ceiling moved to {baseline} kHz"

    try:
        raised = original * 4.0
        cased._BESSEL_ARG_MAX = raised
        cylindrical_solver._BESSEL_ARG_MAX = raised
        with np.errstate(invalid="ignore", over="ignore"):
            widened = ceiling()
    finally:
        cased._BESSEL_ARG_MAX = original
        cylindrical_solver._BESSEL_ARG_MAX = original

    assert abs(widened - baseline) <= 4.0, (
        f"the ceiling moved from {baseline} to {widened} kHz when the Bessel "
        f"bound was raised 4x; it is supposed to be held down by the "
        f"propagator product instead"
    )


def test_microannulus_stable_root_filter_drops_grid_dependent_roots():
    """The filter's contract, tested directly rather than through the physics.

    The configuration that motivated it produces its artefact on some machines
    and not others, so the physical test above cannot assert that the filter
    ever fires. This one can: a synthetic determinant with two sign changes,
    one spanning a wide interval that any grid resolves and one confined to a
    needle placed on a point of the first scan grid and between points of the
    second. The wide root must survive and the needle must not.
    """
    from fwap.cylindrical_solver import _microannulus_stable_roots

    c_lo, c_hi, samples = 5.0, 1000.0, 120
    first = np.geomspace(c_lo, c_hi, samples)
    second = np.geomspace(c_lo * 1.013, c_hi, int(samples * 1.37) + 1)

    # A point of the first grid, low enough that the second grid's spacing
    # there is far wider than the needle we hang on it.
    needle_centre = float(first[3])
    gaps = np.abs(second - needle_centre)
    assert gaps.min() > 1.0e-3, "second grid must not sample the needle"
    half_width = 1.0e-4

    def det(c: float) -> float:
        if abs(c - needle_centre) < half_width:
            return -1.0
        return c - 400.0

    roots = _microannulus_stable_roots(det, c_lo, c_hi, samples)
    assert len(roots) == 1
    assert roots[0] == pytest.approx(400.0, rel=1.0e-9)

    # And the needle really is visible to the first grid on its own, so the
    # test is about the filter and not about the needle being unreachable.
    solo = [c for c in first if abs(c - needle_centre) < half_width]
    assert solo


# ----------------------------------------------------------------------
