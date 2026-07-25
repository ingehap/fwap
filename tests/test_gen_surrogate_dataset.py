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
