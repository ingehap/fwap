"""Tests for sonic_ml.oracles -- vendored closed forms vs the fwap solver."""

from __future__ import annotations

import numpy as np
import pytest
from fwap.cylindrical import rayleigh_speed

from sonic_ml import oracles


def test_stoneley_lf_matches_solver_at_10hz():
    # Mirrors tests/test_cylindrical_solver.py: the modal Stoneley slowness
    # converges to White's closed form at ~10 Hz.
    from fwap import stoneley_dispersion

    vp, vs, rho = 4000.0, 2300.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    res = stoneley_dispersion(
        np.array([10.0]), vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a
    )
    s_truth = oracles.stoneley_lf_slowness(vs, rho, vf, rho_f)
    assert res.slowness[0] == pytest.approx(s_truth, rel=1.0e-4)


def test_stoneley_lf_formula():
    s = oracles.stoneley_lf_slowness(vs=2300.0, rho=2400.0, vf=1500.0, rho_f=1000.0)
    mu = 2400.0 * 2300.0**2
    assert s == pytest.approx(np.sqrt(1.0 / 1500.0**2 + 1000.0 / mu))
    # Stoneley is always slower than the bare fluid.
    assert s > 1.0 / 1500.0


def test_flexural_lf_is_inverse_vs():
    assert oracles.flexural_lf_slowness(2500.0) == pytest.approx(1.0 / 2500.0)
    with pytest.raises(ValueError):
        oracles.flexural_lf_slowness(0.0)


def test_flexural_hf_is_rayleigh_and_faster_than_shear():
    vp, vs = 4000.0, 2300.0
    s_hf = oracles.flexural_hf_slowness(vp, vs)
    assert s_hf == pytest.approx(1.0 / rayleigh_speed(vp, vs))
    # Rayleigh speed < shear speed -> high-f slowness > low-f (1/vs).
    assert s_hf > oracles.flexural_lf_slowness(vs)


def test_solver_flexural_asymptotes_bracket_oracles():
    # The modal flexural mode sits between the LF (1/vs) and HF (1/V_R) limits.
    from fwap import flexural_dispersion

    vp, vs, rho = 4000.0, 2300.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    freq = np.linspace(1000.0, 12000.0, 128)
    res = flexural_dispersion(freq, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a)
    s = res.slowness[np.isfinite(res.slowness)]
    lf = oracles.flexural_lf_slowness(vs)
    hf = oracles.flexural_hf_slowness(vp, vs)
    # Finite modal slownesses lie within (just outside numerically) the band.
    assert s.min() >= lf - 5e-6
    assert s.max() <= hf + 5e-6
