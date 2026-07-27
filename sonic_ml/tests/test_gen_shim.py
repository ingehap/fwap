"""Tests for sonic_ml.gen_shim."""

from __future__ import annotations

import numpy as np
import pytest

from sonic_ml import gen_shim
from sonic_ml.loader import load_npz


def test_generator_exposes_expected_surface():
    g = gen_shim.generator()
    for name in (
        "FormationPriors",
        "ModeSpec",
        "DEFAULT_MODES",
        "default_freq_grid",
        "generate_dataset",
        "stack_dataset",
        "save_npz",
        "PARAM_NAMES",
        "SCHEMA_VERSION",
    ):
        assert hasattr(g, name), name
    assert g.PARAM_NAMES == ("vp", "vs", "rho", "vf", "rho_f", "a")


def test_generator_is_cached():
    assert gen_shim.generator() is gen_shim.generator()


def test_generate_forwards_kwargs(freq):
    samples = gen_shim.generate(5, seed=0, freq=freq)
    assert len(samples) == 5


def test_build_npz_writes_loadable_file(tmp_path, freq):
    p = tmp_path / "out.npz"
    gen_shim.build_npz(str(p), n=6, seed=1, freq=freq)
    b = load_npz(str(p))
    assert b.n_samples == 6
    assert b.n_freq == freq.size


def test_custom_priors_are_honoured(freq):
    g = gen_shim.generator()
    priors = g.FormationPriors(vs_min=1300.0, vs_max=1400.0)
    samples = gen_shim.generate(8, seed=2, freq=freq, priors=priors)
    vs = np.array([s.params["vs"] for s in samples])
    assert np.all((vs >= 1300.0) & (vs <= 1400.0))


def test_env_override_missing_file(monkeypatch):
    monkeypatch.setenv("SONIC_ML_GENERATOR", "/no/such/generator.py")
    with pytest.raises(FileNotFoundError):
        gen_shim._default_generator_path()
