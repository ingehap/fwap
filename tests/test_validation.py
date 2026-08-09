"""Tests for scoring dispersion curves against digitised reference figures.

The point of this module is to be *suspicious of its input*, so most of what is
asserted here is refusal: a reference in the wrong units, a velocity column
mistaken for a slowness one, an unsorted trace, a curve that barely overlaps the
model. Each of those produces a file that looks fine, and the worst outcome is
not a crash but a comparison that quietly succeeds against the wrong thing.

The scoring itself is checked against hand-computable cases, so a change in the
metric shows up as a failing arithmetic assertion rather than as a slightly
different-looking number.
"""

from __future__ import annotations

import numpy as np
import pytest

from fwap import (
    OverlayScore,
    ReferenceDataError,
    format_overlay_score,
    load_reference_curve,
    score_against_reference,
)


def write_csv(tmp_path, name, freq, slowness):
    p = tmp_path / name
    p.write_text(
        "\n".join(f"{f},{s}" for f, s in zip(freq, slowness, strict=True)) + "\n"
    )
    return p


@pytest.fixture
def good_csv(tmp_path):
    freq = np.linspace(2000.0, 12000.0, 11)
    slowness = np.linspace(1.0 / 1500.0, 1.0 / 1700.0, 11)
    return write_csv(tmp_path, "ref.csv", freq, slowness)


# ------------------------------------------------------------------
# load_reference_curve -- the refusals
# ------------------------------------------------------------------


def test_loads_a_well_formed_reference(good_csv):
    curve = load_reference_curve(good_csv)
    assert curve.n_points == 11
    assert curve.source == "ref.csv"
    assert curve.freq_range == (2000.0, 12000.0)
    assert np.all(np.diff(curve.freq) > 0.0)


def test_source_label_can_be_overridden(good_csv):
    curve = load_reference_curve(good_csv, source="Paillet & Cheng 1991 fig 4.5")
    assert curve.source == "Paillet & Cheng 1991 fig 4.5"


def test_digitiser_click_order_is_sorted_not_rejected(tmp_path):
    """Digitisers emit click order; an unsorted trace is normal, not an error.

    It must be sorted on load, because silently interpolating against an
    unsorted x-axis is the kind of bug that produces a plausible wrong number.
    """
    freq = np.array([8000.0, 2000.0, 5000.0, 11000.0])
    slowness = np.array([6.2e-4, 6.6e-4, 6.4e-4, 6.0e-4])
    curve = load_reference_curve(write_csv(tmp_path, "shuffled.csv", freq, slowness))
    np.testing.assert_array_equal(curve.freq, np.sort(freq))
    np.testing.assert_array_equal(curve.slowness, [6.6e-4, 6.4e-4, 6.2e-4, 6.0e-4])


def test_slowness_in_us_per_ft_is_refused_with_a_diagnosis(tmp_path):
    """The dangerous case: a figure axis read in the units it was printed in."""
    freq = np.linspace(2000.0, 10000.0, 6)
    us_per_ft = np.linspace(203.0, 179.0, 6)  # ~1/1500..1/1700 s/m in us/ft
    with pytest.raises(ReferenceDataError, match="us/ft or us/m"):
        load_reference_curve(write_csv(tmp_path, "usft.csv", freq, us_per_ft))


def test_velocity_column_is_refused_as_a_velocity(tmp_path):
    """A velocity axis digitised as slowness must be named, not just rejected."""
    freq = np.linspace(2000.0, 10000.0, 6)
    velocity = np.linspace(1500.0, 1700.0, 6)
    with pytest.raises(ReferenceDataError, match="velocity"):
        load_reference_curve(write_csv(tmp_path, "vel.csv", freq, velocity))


def test_frequency_in_kilohertz_is_refused(tmp_path):
    freq_khz = np.linspace(2.0, 12.0, 6)
    slowness = np.linspace(6.6e-4, 5.9e-4, 6)
    with pytest.raises(ReferenceDataError, match="kHz"):
        load_reference_curve(write_csv(tmp_path, "khz.csv", freq_khz, slowness))


def test_units_are_never_silently_converted(tmp_path):
    """A rescaled reference could make a wrong solver pass, so refuse instead.

    This is the whole reason the loader diagnoses rather than fixes.
    """
    freq = np.linspace(2000.0, 10000.0, 6)
    us_per_ft = np.linspace(203.0, 179.0, 6)
    path = write_csv(tmp_path, "usft.csv", freq, us_per_ft)
    with pytest.raises(ReferenceDataError):
        load_reference_curve(path)
    # the file on disk is untouched, so nothing was "helpfully" rewritten
    assert "203" in path.read_text()


def test_wildly_out_of_band_slowness_still_gets_a_message(tmp_path):
    """Not every mistake is a recognisable unit swap; refuse those too."""
    freq = np.linspace(2000.0, 10000.0, 4)
    nonsense = np.full(4, 1.0e8)
    with pytest.raises(ReferenceDataError, match="plausible slowness band"):
        load_reference_curve(write_csv(tmp_path, "odd.csv", freq, nonsense))


def test_duplicate_frequencies_are_refused(tmp_path):
    freq = np.array([2000.0, 5000.0, 5000.0, 9000.0])
    slowness = np.array([6.6e-4, 6.4e-4, 6.3e-4, 6.0e-4])
    with pytest.raises(ReferenceDataError, match="duplicate"):
        load_reference_curve(write_csv(tmp_path, "dup.csv", freq, slowness))


def test_too_few_points_is_refused(tmp_path):
    with pytest.raises(ReferenceDataError, match="at least 3"):
        load_reference_curve(
            write_csv(tmp_path, "tiny.csv", [2000.0, 4000.0], [6.6e-4, 6.4e-4])
        )


def test_empty_and_missing_files(tmp_path):
    (tmp_path / "empty.csv").write_text("")
    with pytest.raises(ReferenceDataError, match="empty"):
        load_reference_curve(tmp_path / "empty.csv")
    with pytest.raises(FileNotFoundError, match="no digitised reference"):
        load_reference_curve(tmp_path / "absent.csv")


def test_non_finite_and_non_positive_values_are_refused(tmp_path):
    freq = np.linspace(2000.0, 8000.0, 4)
    with pytest.raises(ReferenceDataError, match="non-finite"):
        load_reference_curve(
            write_csv(tmp_path, "nan.csv", freq, [6.6e-4, np.nan, 6.2e-4, 6.0e-4])
        )
    with pytest.raises(ReferenceDataError, match="non-positive"):
        load_reference_curve(
            write_csv(tmp_path, "neg.csv", freq, [6.6e-4, -1e-4, 6.2e-4, 6.0e-4])
        )


def test_wrong_column_count_is_refused(tmp_path):
    p = tmp_path / "three.csv"
    p.write_text("2000,6.6e-4,1\n5000,6.4e-4,2\n9000,6.0e-4,3\n")
    with pytest.raises(ReferenceDataError, match="exactly 2 columns"):
        load_reference_curve(p)


# ------------------------------------------------------------------
# score_against_reference
# ------------------------------------------------------------------


def test_an_exact_match_scores_zero(good_csv):
    curve = load_reference_curve(good_csv)
    score = score_against_reference(curve.freq, curve.slowness, curve)
    assert score.rms_relative == pytest.approx(0.0)
    assert score.max_relative == pytest.approx(0.0)
    assert score.passed
    assert score.coverage == pytest.approx(1.0)


def test_a_uniform_offset_scores_that_offset(good_csv):
    """A model 3% slow everywhere must score 3% RMS -- checkable by hand."""
    curve = load_reference_curve(good_csv)
    score = score_against_reference(curve.freq, curve.slowness * 1.03, curve)
    assert score.rms_relative == pytest.approx(0.03, rel=1e-9)
    assert score.max_relative == pytest.approx(0.03, rel=1e-9)


def test_the_budget_decides_pass_and_fail(good_csv):
    curve = load_reference_curve(good_csv)
    model = curve.slowness * 1.07
    assert not score_against_reference(curve.freq, model, curve).passed
    assert score_against_reference(curve.freq, model, curve, budget=0.10).passed


def test_worst_point_is_reported_next_to_the_rms(good_csv):
    """A single bad end must be visible, since RMS alone can absorb it."""
    curve = load_reference_curve(good_csv)
    model = curve.slowness.copy()
    model[-1] *= 1.30
    score = score_against_reference(curve.freq, model, curve, budget=0.10)
    assert score.max_relative == pytest.approx(0.30, rel=1e-6)
    assert score.rms_relative < 0.10  # RMS alone would call this fine
    assert score.passed  # ... and it does, which is why max is reported


def test_absent_mode_frequencies_are_excluded_not_treated_as_data(good_csv):
    curve = load_reference_curve(good_csv)
    model = curve.slowness.copy()
    model[:4] = np.nan  # mode does not exist at the low end
    score = score_against_reference(curve.freq, model, curve)
    assert score.n_compared == 7
    assert score.coverage == pytest.approx(7 / 11)
    assert score.rms_relative == pytest.approx(0.0)


def test_coverage_exposes_a_score_computed_over_almost_nothing(good_csv):
    """Three points of an eleven-point figure is not the check it looks like."""
    curve = load_reference_curve(good_csv)
    narrow_f = curve.freq[-3:]
    score = score_against_reference(narrow_f, curve.slowness[-3:], curve)
    assert score.n_compared == 3
    assert score.coverage < 0.3
    assert score.passed  # passes, and coverage is how a reader sees the catch


def test_non_overlapping_curves_cannot_be_scored(good_csv):
    """An empty comparison must raise, never return a passing score."""
    curve = load_reference_curve(good_csv)
    with pytest.raises(ReferenceDataError, match="do not overlap"):
        score_against_reference(np.linspace(50e3, 80e3, 20), np.full(20, 6.0e-4), curve)


def test_scoring_rejects_malformed_model_input(good_csv):
    curve = load_reference_curve(good_csv)
    with pytest.raises(ValueError, match="1-D"):
        score_against_reference(curve.freq[:, None], curve.slowness[:, None], curve)
    with pytest.raises(ValueError, match="must match"):
        score_against_reference(curve.freq[:-1], curve.slowness, curve)
    with pytest.raises(ValueError, match="budget must be positive"):
        score_against_reference(curve.freq, curve.slowness, curve, budget=0.0)
    with pytest.raises(ReferenceDataError, match="fewer than 2 finite"):
        score_against_reference(curve.freq, np.full(curve.n_points, np.nan), curve)


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------


def test_the_verdict_is_always_stated(good_csv):
    curve = load_reference_curve(good_csv)
    passing = score_against_reference(curve.freq, curve.slowness, curve)
    failing = score_against_reference(curve.freq, curve.slowness * 1.2, curve)
    assert format_overlay_score(passing).startswith("PASS")
    assert format_overlay_score(failing).startswith("FAIL")
    assert "ref.csv" in format_overlay_score(passing)
    assert "coverage" in format_overlay_score(passing)


def test_score_is_a_plain_value_object():
    score = OverlayScore(
        source="fig",
        rms_relative=0.02,
        max_relative=0.04,
        n_compared=8,
        n_reference=10,
        overlap_hz=(2000.0, 9000.0),
        budget=0.05,
    )
    assert score.passed
    assert score.coverage == pytest.approx(0.8)


# ------------------------------------------------------------------
# End to end against the real solver
# ------------------------------------------------------------------


def test_a_solver_curve_scores_against_a_reference_traced_from_itself(tmp_path):
    """The wiring works on a real dispersion curve, not just on arrays.

    The 'reference' here is sampled from fwap's own solver, so this asserts
    nothing about physics -- it asserts that a curve, a CSV round-trip and the
    scorer compose. The physics check is the digitised figure, which needs data
    this repository does not yet ship.
    """
    from fwap import stoneley_dispersion

    kwargs = dict(vp=4880.0, vs=2820.0, rho=2700.0, vf=1500.0, rho_f=1000.0, a=0.10)
    dense = np.linspace(2000.0, 12000.0, 101)
    model = stoneley_dispersion(dense, **kwargs)

    sparse = np.linspace(3000.0, 11000.0, 9)
    traced = stoneley_dispersion(sparse, **kwargs)
    path = write_csv(tmp_path, "traced.csv", sparse, traced.slowness)

    score = score_against_reference(
        dense, model.slowness, load_reference_curve(path), budget=0.01
    )
    assert score.passed
    assert score.n_compared == 9
    assert score.rms_relative < 1e-3
