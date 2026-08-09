"""
Tests for the ``scripts/gen_surrogate_dataset.py`` surrogate-data
generator. The script is not an installed package, so it is loaded by
path via :mod:`importlib`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "gen_surrogate_dataset.py"
)
_spec = importlib.util.spec_from_file_location("gen_surrogate_dataset", _SCRIPT)
assert _spec is not None and _spec.loader is not None
gen = importlib.util.module_from_spec(_spec)
# Register before exec so dataclasses can resolve the module by name when
# ``from __future__ import annotations`` defers annotation evaluation.
sys.modules[_spec.name] = gen
_spec.loader.exec_module(gen)


# ------------------------------------------------------------------
# FormationPriors.sample
# ------------------------------------------------------------------


def test_sample_respects_ranges_and_vp_gt_vs():
    """Every draw is in-range and satisfies the solver's vp > vs bound."""
    priors = gen.FormationPriors()
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = priors.sample(rng)
        assert set(p) == set(gen.PARAM_NAMES)
        assert priors.vs_min <= p["vs"] <= priors.vs_max
        assert priors.a_min <= p["a"] <= priors.a_max
        assert priors.rho_min <= p["rho"] <= priors.rho_max
        assert p["vp"] > p["vs"]  # required by stoneley/flexural_dispersion
        assert p["vf"] == priors.vf and p["rho_f"] == priors.rho_f


def test_sample_is_reproducible():
    """Same seed -> identical draw sequence."""
    priors = gen.FormationPriors()
    a = [priors.sample(np.random.default_rng(7)) for _ in range(1)]
    b = [priors.sample(np.random.default_rng(7)) for _ in range(1)]
    assert a == b


# ------------------------------------------------------------------
# default_freq_grid
# ------------------------------------------------------------------


def test_default_freq_grid_shape_and_bounds():
    f = gen.default_freq_grid(n_freq=64, f_min=1500.0, f_max=9000.0)
    assert f.shape == (64,)
    assert f[0] == pytest.approx(1500.0)
    assert f[-1] == pytest.approx(9000.0)
    assert np.all(np.diff(f) > 0)


# ------------------------------------------------------------------
# dispersion_callable
# ------------------------------------------------------------------


def test_dispersion_callable_interpolates_and_edge_clamps():
    """Callable reproduces finite samples and clamps outside support."""
    from fwap.cylindrical_solver import BoreholeMode

    freq = np.linspace(1000.0, 5000.0, 5)
    slow = np.array([np.nan, 6.0e-4, 5.0e-4, 4.0e-4, np.nan])
    mode = BoreholeMode("test", 0, freq, slow)
    s_of_f = gen.dispersion_callable(mode, min_finite=3)
    assert s_of_f is not None
    # Reproduces an interior node exactly.
    assert s_of_f(np.array([3000.0]))[0] == pytest.approx(5.0e-4)
    # Edge-clamps below/above the finite support.
    assert s_of_f(np.array([0.0]))[0] == pytest.approx(6.0e-4)
    assert s_of_f(np.array([9000.0]))[0] == pytest.approx(4.0e-4)


def test_dispersion_callable_returns_none_when_too_sparse():
    from fwap.cylindrical_solver import BoreholeMode

    freq = np.linspace(1000.0, 5000.0, 5)
    slow = np.array([np.nan, np.nan, 5.0e-4, np.nan, np.nan])
    mode = BoreholeMode("test", 0, freq, slow)
    assert gen.dispersion_callable(mode, min_finite=8) is None


# ------------------------------------------------------------------
# generate_sample / generate_dataset
# ------------------------------------------------------------------


def _small_grid() -> np.ndarray:
    # Coarser grid keeps the forward solves fast in the test suite.
    return gen.default_freq_grid(n_freq=48)


def test_generate_sample_shapes_and_labels():
    rng = np.random.default_rng(0)
    from fwap import ArrayGeometry

    geom = ArrayGeometry(n_rec=8, n_samples=512)
    freq = _small_grid()
    sample = gen.generate_sample(
        rng, geom, freq, priors=gen.FormationPriors(), min_finite=6
    )
    assert sample is not None
    n_modes = len(gen.DEFAULT_MODES)
    assert sample.slowness.shape == (n_modes, freq.size)
    assert sample.gather.shape == (geom.n_rec, geom.n_samples)
    assert sample.mode_in_gather.shape == (n_modes,)
    assert sample.param_vector().shape == (len(gen.PARAM_NAMES),)
    # At least one mode was injected (else the sample would be rejected).
    assert sample.mode_in_gather.any()
    # A mode flagged as injected must have some finite slowness.
    for i in range(n_modes):
        if sample.mode_in_gather[i]:
            assert np.isfinite(sample.slowness[i]).any()
    # The gather is finite and non-trivial.
    assert np.all(np.isfinite(sample.gather))
    assert np.std(sample.gather) > 0.0


def test_generate_dataset_is_reproducible_and_sized():
    ds_a = gen.generate_dataset(4, seed=1, freq=_small_grid())
    ds_b = gen.generate_dataset(4, seed=1, freq=_small_grid())
    assert len(ds_a) == 4
    for sa, sb in zip(ds_a, ds_b):
        assert sa.params == sb.params
        np.testing.assert_array_equal(sa.gather, sb.gather)
        np.testing.assert_array_equal(
            np.nan_to_num(sa.slowness), np.nan_to_num(sb.slowness)
        )


def test_generate_dataset_rejects_negative_n():
    with pytest.raises(ValueError):
        gen.generate_dataset(-1)


def test_generate_dataset_raises_when_priors_never_accept():
    # Degenerate prior: vs pinned absurdly high so vp > vs still holds but
    # both modes fail/are too sparse is unlikely; instead force starvation
    # with a tiny max_attempts and an impossible acceptance via min_finite
    # larger than the grid.
    with pytest.raises(RuntimeError):
        gen.generate_dataset(
            5, seed=0, freq=_small_grid(), min_finite=10_000, max_attempts=6
        )


# ------------------------------------------------------------------
# stack_dataset / save_npz
# ------------------------------------------------------------------


def test_stack_dataset_shapes():
    freq = _small_grid()
    samples = gen.generate_dataset(3, seed=2, freq=freq)
    stacked = gen.stack_dataset(samples)
    n = len(samples)
    n_modes = len(gen.DEFAULT_MODES)
    assert stacked["params"].shape == (n, len(gen.PARAM_NAMES))
    assert stacked["slowness"].shape == (n, n_modes, freq.size)
    assert stacked["gather"].shape[0] == n
    assert stacked["mode_in_gather"].shape == (n, n_modes)
    assert stacked["freq"].shape == (freq.size,)
    assert tuple(stacked["param_names"]) == gen.PARAM_NAMES


def test_stack_dataset_empty_raises():
    with pytest.raises(ValueError):
        gen.stack_dataset([])


def test_save_npz_round_trip(tmp_path):
    freq = _small_grid()
    samples = gen.generate_dataset(3, seed=3, freq=freq)
    out = tmp_path / "ds.npz"
    gen.save_npz(str(out), samples)
    assert out.exists()
    with np.load(out, allow_pickle=False) as data:
        assert data["params"].shape == (3, len(gen.PARAM_NAMES))
        assert data["gather"].shape[0] == 3
        np.testing.assert_array_equal(data["freq"], freq)


def test_main_writes_file(tmp_path, capsys):
    out = tmp_path / "cli.npz"
    rc = gen.main(["--n", "2", "--seed", "0", "--out", str(out), "--n-freq", "48"])
    assert rc == 0
    assert out.exists()
    captured = capsys.readouterr()
    assert "wrote 2 samples" in captured.out


# ------------------------------------------------------------------
# Optional n=2 quadrupole mode
# ------------------------------------------------------------------


def test_default_modes_stay_lean():
    # QUADRUPOLE_MODE is opt-in; the default dataset is unchanged (2 modes).
    assert [m.name for m in gen.DEFAULT_MODES] == ["Stoneley", "flexural"]
    assert gen.QUADRUPOLE_MODE.name == "quadrupole"


def test_three_mode_dataset_generates_and_stacks(tmp_path):
    freq = gen.default_freq_grid(n_freq=64)
    modes = (*gen.DEFAULT_MODES, gen.QUADRUPOLE_MODE)
    samples = gen.generate_dataset(6, seed=0, freq=freq, modes=modes)
    stacked = gen.stack_dataset(samples)
    # Mode-agnostic: M grows to 3, arrays follow, mode_names carries the label.
    assert tuple(stacked["mode_names"]) == ("Stoneley", "flexural", "quadrupole")
    n = len(samples)
    assert stacked["slowness"].shape == (n, 3, freq.size)
    assert stacked["mode_in_gather"].shape == (n, 3)
    # Round-trips through save/load.
    out = tmp_path / "q.npz"
    gen.save_npz(str(out), samples)
    with np.load(out, allow_pickle=False) as data:
        assert tuple(data["mode_names"]) == ("Stoneley", "flexural", "quadrupole")


# ------------------------------------------------------------------
# Leaky-mode attenuation channel (schema v3)
# ------------------------------------------------------------------


def test_pseudo_rayleigh_mode_is_optin():
    assert gen.PSEUDO_RAYLEIGH_MODE.name == "pseudo_rayleigh"
    assert "pseudo_rayleigh" not in [m.name for m in gen.DEFAULT_MODES]


def test_attenuation_is_all_nan_for_bound_default_modes():
    freq = gen.default_freq_grid(n_freq=48)
    stacked = gen.stack_dataset(gen.generate_dataset(6, seed=0, freq=freq))
    assert stacked["attenuation"].shape == stacked["slowness"].shape
    # Stoneley + flexural are bound -> no attenuation anywhere.
    assert not np.isfinite(stacked["attenuation"]).any()


def test_leaky_mode_populates_attenuation():
    freq = gen.default_freq_grid(n_freq=64)
    priors = gen.FormationPriors(vs_min=1700.0, vs_max=3200.0)  # fast-only
    modes = (*gen.DEFAULT_MODES, gen.PSEUDO_RAYLEIGH_MODE)
    stacked = gen.stack_dataset(
        gen.generate_dataset(12, seed=0, freq=freq, priors=priors, modes=modes)
    )
    names = list(stacked["mode_names"])
    pr = names.index("pseudo_rayleigh")
    att_pr = stacked["attenuation"][:, pr, :]
    assert np.isfinite(att_pr).any()  # the leaky mode carries attenuation
    assert np.nanmax(att_pr) > 0.0
    # a bound mode's attenuation row stays NaN
    assert not np.isfinite(stacked["attenuation"][:, names.index("Stoneley"), :]).any()


# ------------------------------------------------------------------
# Cased-hole dataset (schema v4)
# ------------------------------------------------------------------


def test_open_hole_default_has_empty_layers_and_nan_bond():
    # The default (open-hole) dataset is schema v4 but carries no annulus.
    freq = _small_grid()
    stacked = gen.stack_dataset(gen.generate_dataset(3, seed=0, freq=freq))
    assert stacked["layer_params"].shape == (3, 0, len(gen.LAYER_PARAM_NAMES))
    assert tuple(stacked["layer_names"]) == ()
    assert not np.isfinite(stacked["bond_index"]).any()


def test_casing_cement_priors_sample_ranges_and_bond():
    priors = gen.CasingCementPriors()
    rng = np.random.default_rng(0)
    for _ in range(50):
        (casing, cement), bond = priors.sample(rng)
        assert priors.casing_vp[0] <= casing.vp <= priors.casing_vp[1]
        assert priors.cement_vs[0] <= cement.vs <= priors.cement_vs[1]
        assert cement.vs >= priors.vf  # stays in the bound Stoneley regime
        assert 0.0 <= bond <= 1.0
        assert casing.thickness > 0.0 and cement.thickness > 0.0
    assert priors.layer_names == ("casing", "cement")


def test_casing_cement_priors_reproducible():
    priors = gen.CasingCementPriors()
    a = priors.sample(np.random.default_rng(3))
    b = priors.sample(np.random.default_rng(3))
    assert a[1] == b[1]  # same bond index
    assert a[0][0].vp == b[0][0].vp and a[0][1].vs == b[0][1].vs


def test_generate_cased_dataset_shapes_and_finite_stoneley():
    freq = gen.default_freq_grid(n_freq=48)
    samples = gen.generate_cased_dataset(6, seed=0, freq=freq)
    assert len(samples) == 6
    stacked = gen.stack_dataset(samples)
    # Single bound Stoneley mode, two-layer annulus.
    assert tuple(stacked["mode_names"]) == ("Stoneley",)
    assert stacked["layer_params"].shape == (6, 2, len(gen.LAYER_PARAM_NAMES))
    assert tuple(stacked["layer_names"]) == ("casing", "cement")
    # The Stoneley curve is bound (finite) across the band in this regime.
    assert np.isfinite(stacked["slowness"]).mean() > 0.9
    # Every cased sample carries a finite bond index and was injected.
    assert np.all(np.isfinite(stacked["bond_index"]))
    assert stacked["mode_in_gather"].all()


def test_generate_cased_dataset_reproducible():
    freq = gen.default_freq_grid(n_freq=48)
    a = gen.generate_cased_dataset(4, seed=1, freq=freq)
    b = gen.generate_cased_dataset(4, seed=1, freq=freq)
    for sa, sb in zip(a, b):
        assert sa.params == sb.params
        assert sa.bond_index == sb.bond_index
        np.testing.assert_array_equal(sa.layer_params, sb.layer_params)
        np.testing.assert_array_equal(sa.gather, sb.gather)


def test_cased_dataset_round_trips_through_npz(tmp_path):
    freq = gen.default_freq_grid(n_freq=48)
    samples = gen.generate_cased_dataset(4, seed=2, freq=freq)
    out = tmp_path / "cased.npz"
    gen.save_npz(str(out), samples)
    with np.load(out, allow_pickle=False) as data:
        assert data["layer_params"].shape == (4, 2, 4)
        assert tuple(data["layer_names"]) == ("casing", "cement")
        assert np.all(np.isfinite(data["bond_index"]))


def test_main_cased_flag_writes_cased_file(tmp_path, capsys):
    out = tmp_path / "cli_cased.npz"
    rc = gen.main(
        ["--n", "2", "--seed", "0", "--out", str(out), "--n-freq", "48", "--cased"]
    )
    assert rc == 0
    with np.load(out, allow_pickle=False) as data:
        assert tuple(data["mode_names"]) == ("Stoneley",)
        assert tuple(data["layer_names"]) == ("casing", "cement")
        assert data["layer_params"].shape == (2, 2, 4)
        assert np.all(np.isfinite(data["bond_index"]))


# ----------------------------------------------------------------------
# Two-mode cased dataset, restricted to the window where both modes bind
# ----------------------------------------------------------------------


def test_slow_two_mode_cased_dataset_carries_both_modes_fully_bound():
    """Both cased modes finite across the whole band, and both injected.

    This is the whole point of the restricted prior: the default cased
    dataset is single-mode because flexural is sparse in fast formations,
    and this configuration is the measured window where that stops being
    true without costing the Stoneley.
    """
    freq = gen.default_freq_grid(n_freq=48)
    samples = gen.generate_slow_two_mode_cased_dataset(6, seed=0, freq=freq)
    assert len(samples) == 6

    stacked = gen.stack_dataset(samples)
    assert tuple(stacked["mode_names"]) == ("Stoneley", "flexural")
    assert stacked["slowness"].shape == (6, 2, freq.size)
    # every sample, every mode, every frequency
    assert np.all(np.isfinite(stacked["slowness"]))
    assert stacked["mode_in_gather"].all()
    assert np.all(np.isfinite(stacked["bond_index"]))
    # the annulus is still there -- this is a cased dataset, not an open one
    assert tuple(stacked["layer_names"]) == ("casing", "cement")


def test_two_mode_window_is_slow_and_disjoint_from_the_default_cased_prior():
    """The restriction is a documented property, so it is pinned as one.

    A dataset from this prior must not be pooled with the default cased
    one; asserting the disjointness keeps a later widening of either prior
    from quietly making that pooling look legitimate.
    """
    window = gen.SLOW_TWO_MODE_PRIORS
    fluid = window.vf
    assert window.vs_max < fluid  # slow formations, by construction
    assert window.vs_min >= 1420.0  # the measured both-modes-bound floor

    default_cased = gen.FormationPriors(vs_min=1700.0, vs_max=3000.0)
    assert window.vs_max < default_cased.vs_min  # disjoint, not a subset


def test_the_two_mode_window_has_a_reason_for_its_lower_edge():
    """Below the window the Stoneley stops being bound; flexural does not.

    The window's floor looks arbitrary until you see that the two modes
    fail in opposite directions. This checks the direction of that trade
    rather than a precise coverage number, which would be pinning the
    solver's numerics.
    """
    from fwap import flexural_dispersion_layered, stoneley_dispersion_layered

    freq = gen.default_freq_grid(n_freq=48)
    layers, _bond = gen.CasingCementPriors().sample(np.random.default_rng(0))
    borehole = dict(vf=1500.0, rho_f=1000.0, a=0.10, layers=layers)

    inside = dict(vp=1450.0 * 1.8, vs=1450.0, rho=2300.0, **borehole)
    below = dict(vp=1300.0 * 1.8, vs=1300.0, rho=2300.0, **borehole)

    # flexural is healthy on both sides -- it is not what sets the floor
    for formation in (inside, below):
        flexural = flexural_dispersion_layered(freq, **formation)
        assert np.isfinite(flexural.slowness).all()

    # the Stoneley is what degrades going down
    inside_st = np.isfinite(stoneley_dispersion_layered(freq, **inside).slowness)
    below_st = np.isfinite(stoneley_dispersion_layered(freq, **below).slowness)
    assert inside_st.all()
    assert below_st.mean() < inside_st.mean()


def test_slow_two_mode_cased_dataset_reproducible():
    freq = gen.default_freq_grid(n_freq=48)
    a = gen.generate_slow_two_mode_cased_dataset(3, seed=5, freq=freq)
    b = gen.generate_slow_two_mode_cased_dataset(3, seed=5, freq=freq)
    for sa, sb in zip(a, b):
        assert sa.params == sb.params
        assert sa.bond_index == sb.bond_index
        np.testing.assert_array_equal(sa.gather, sb.gather)


# ---------------------------------------------------------------------
# Debonded (fluid-microannulus) cased sampling -- roadmap G.2.
#
# The measurements behind the design live in `MicroannulusPriors`; these
# pin the parts a change could break silently. The dataset itself is not
# generated here: a debonded sample costs ~14 s against ~0.5 s bonded,
# so end-to-end generation is a batch job rather than a unit test.
# ---------------------------------------------------------------------


def test_microannulus_priors_draw_ranges_and_layers():
    priors = gen.MicroannulusPriors()
    rng = np.random.default_rng(0)
    lo, hi = priors.gap_thickness
    for _ in range(50):
        draw = priors.draw(rng)
        assert draw.layer_names == ("casing", "microannulus", "cement")
        assert draw.layer_params.shape == (3, len(gen.LAYER_PARAM_NAMES))
        assert 0.0 <= draw.bond_index <= 1.0

        casing, gap, cement = draw.layer_params
        assert priors.casing_vp[0] <= casing[0] <= priors.casing_vp[1]
        assert priors.cement_vs[0] <= cement[1] <= priors.cement_vs[1]
        # The gap is a fluid: no shear velocity, and a thickness in range.
        assert gap[1] == 0.0
        assert lo <= gap[3] <= hi
        assert gap[0] == priors.gap_vf


def test_microannulus_draw_builds_the_microannulus_solver_signature():
    """A debonded prior calls a different solver signature than a bonded one.

    That is the whole reason `AnnulusDraw` exists: `generate_sample` hands
    `solver_kwargs` straight through, so the two priors are interchangeable
    without it knowing which kind of stack it drew.
    """
    draw = gen.MicroannulusPriors().draw(np.random.default_rng(1))
    assert set(draw.solver_kwargs) == {"inner_layers", "annulus", "outer_layers"}
    assert isinstance(draw.solver_kwargs["annulus"], gen.FluidAnnulus)
    assert len(draw.solver_kwargs["inner_layers"]) == 1  # casing
    assert len(draw.solver_kwargs["outer_layers"]) == 1  # cement

    bonded = gen.CasingCementPriors().draw(np.random.default_rng(1))
    assert set(bonded.solver_kwargs) == {"layers"}
    assert len(bonded.solver_kwargs["layers"]) == 2


def test_microannulus_gap_is_sampled_log_uniformly():
    """Uniform-in-log, because the crack wave goes as the cube root of gap.

    A linear-uniform draw over 10 um-1 mm would put 90 % of the samples in
    the top decade and leave the observable badly covered at the tight end,
    which is the end that matters for detecting an incipient microannulus.
    """
    priors = gen.MicroannulusPriors()
    rng = np.random.default_rng(4)
    gaps = np.array([priors.draw(rng).layer_params[1, 3] for _ in range(4000)])
    lo, hi = priors.gap_thickness

    # Each decade of a two-decade range gets about half the samples.
    lower_decade = np.mean(gaps < np.sqrt(lo * hi))
    assert 0.45 < lower_decade < 0.55
    # A linear-uniform draw would fail this badly (it would sit near 0.09).
    assert np.median(gaps) == pytest.approx(np.sqrt(lo * hi), rel=0.1)


def test_microannulus_bond_index_falls_as_the_gap_opens():
    """`bond_index` keeps its meaning across both cased priors: 1 is best.

    It is driven by cement stiffness when bonded and by gap width when
    debonded, which is exactly why the two datasets must not be pooled --
    same column, different question.
    """
    priors = gen.MicroannulusPriors()
    rng = np.random.default_rng(5)
    draws = [priors.draw(rng) for _ in range(200)]
    gaps = np.array([d.layer_params[1, 3] for d in draws])
    bonds = np.array([d.bond_index for d in draws])
    assert np.corrcoef(np.log(gaps), bonds)[0, 1] < -0.99


def test_debonded_modes_record_the_crack_wave_without_injecting_it():
    """The crack wave is a label, not an arrival, and the spec says so.

    At 63-620 m/s it reaches the 3 m near offset between 4.8 ms and 47.6 ms,
    against a 5.12 ms record -- so injecting it would be planting something
    the tool could not have recorded.
    """
    stoneley, crack = gen.DEBONDED_MODES
    assert stoneley.name == "Stoneley"
    assert stoneley.solver is gen.stoneley_dispersion_microannulus
    assert stoneley.inject is True

    assert crack.name == "crack_wave"
    assert crack.solver is gen.crack_wave_dispersion
    assert crack.inject is False


def test_mode_spec_inject_defaults_true_for_every_shipped_set():
    """`inject=False` is opt-in; no existing dataset changes behaviour."""
    for modes in (gen.DEFAULT_MODES, gen.CASED_MODES, gen.CASED_TWO_MODES):
        assert all(spec.inject for spec in modes)


def test_generate_sample_records_a_non_injected_mode_curve(monkeypatch):
    """A non-injected mode keeps its dispersion row and stays out of the gather.

    Driven through `generate_sample` with a stub solver rather than the real
    microannulus one, which costs seconds per call.
    """
    freq = gen.default_freq_grid(16)

    def fake_solver(f, **_kw):
        from fwap.cylindrical_solver import BoreholeMode

        return BoreholeMode(
            name="stub",
            azimuthal_order=0,
            freq=f,
            slowness=np.full(f.shape, 1.0 / 1400.0),
            attenuation_per_meter=None,
        )

    modes = (
        gen.ModeSpec("kept", fake_solver, f0=3000.0, amplitude=1.0),
        gen.ModeSpec("curve_only", fake_solver, f0=3000.0, amplitude=1.0, inject=False),
    )
    sample = gen.generate_sample(
        np.random.default_rng(0),
        gen.ArrayGeometry(),
        freq,
        priors=gen.FormationPriors(),
        modes=modes,
        noise_max=0.0,
        min_finite=4,
    )
    assert sample is not None
    assert sample.mode_names == ("kept", "curve_only")
    # Both curves are recorded ...
    assert np.isfinite(sample.slowness[0]).all()
    assert np.isfinite(sample.slowness[1]).all()
    # ... but only the injectable one reaches the gather.
    assert sample.mode_in_gather.tolist() == [True, False]


def test_cli_debonded_flag_selects_the_debonded_generator(monkeypatch, tmp_path):
    """The wiring, without paying ~14 s a sample to check it.

    Also pins the coarser default grid: `--debonded` drops to 32 points
    unless the caller asks for something else, because the microannulus
    solvers cost ~100x the bonded ones per frequency.
    """
    seen = {}

    def fake_generate(n, **kwargs):
        seen["n"] = n
        seen["freq"] = kwargs["freq"]
        return gen.generate_dataset(1, seed=0, freq=gen.default_freq_grid(8))

    monkeypatch.setattr(gen, "generate_debonded_dataset", fake_generate)
    out = tmp_path / "d.npz"
    assert gen.main(["--debonded", "--n", "3", "--out", str(out)]) == 0
    assert seen["n"] == 3
    assert seen["freq"].size == 32
    assert out.exists()

    seen.clear()
    assert (
        gen.main(["--debonded", "--n", "1", "--n-freq", "16", "--out", str(out)]) == 0
    )
    assert seen["freq"].size == 16, "an explicit --n-freq still wins"
