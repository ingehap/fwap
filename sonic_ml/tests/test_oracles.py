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
    """The flexural mode starts at ``1/vs`` and descends **past** ``1/V_R``.

    This test used to assert that the mode stays between ``1/vs`` and
    ``1/V_R``. It does not: ``V_R`` is not the high-frequency limit of
    the flexural mode, the fluid-loaded (Scholte) speed is, and for this
    rock Scholte is 1470 m/s against a Rayleigh speed of 2116 -- 30 %
    apart, not the "few percent" the oracle's docstring used to claim.

    The old assertion passed only because ``flexural_dispersion`` had the
    same wrong bound built into its search window (fwap roadmap A.2), so
    it could not return anything below ``V_R`` to violate it. With that
    fixed the mode descends to 1508 m/s here, and 86 of its 93 resolved
    samples are below ``V_R``.

    ``flexural_hf_slowness`` is unchanged -- it returns the Rayleigh
    slowness, which is what it says it does. What was wrong was using it
    as a bound the mode has to respect.
    """
    from fwap import flexural_dispersion

    vp, vs, rho = 4000.0, 2300.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    freq = np.linspace(1000.0, 12000.0, 128)
    res = flexural_dispersion(freq, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a)
    s = res.slowness[np.isfinite(res.slowness)]
    assert s.size > 10

    lf = oracles.flexural_lf_slowness(vs)
    hf = oracles.flexural_hf_slowness(vp, vs)

    # Low-frequency end: the formation shear speed, as before.
    assert s.min() >= lf - 5e-6
    # High-frequency end: bounded by the fluid speed, which is where this
    # solver's bound-mode regime ends, not by the Rayleigh slowness.
    assert s.max() <= 1.0 / vf + 5e-6
    # And the branch is seen to pass below V_R, which is the correction.
    assert s.max() > hf, "the mode must descend past the Rayleigh slowness"
    assert np.all(np.diff(1.0 / s) <= 1.0e-3 * (1.0 / s[:-1])), (
        "phase velocity descends monotonically"
    )
