"""
Plan F: the layered (invaded-zone) determinants, row by row.

One of six modules split out of ``tests/test_cylindrical_solver.py``.
The n=0 seven-row and n=1 ten-row assemblies are tested one row at a
time before the determinants that use them, which is why this module
is mostly row builders: substeps F.1.b.2 through F.2.e.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fwap.cylindrical_solver import (
    BoreholeLayer,
    BoreholeMode,
    _layered_n0_bessel_pack,
    _layered_n0_radial_wavenumbers,
    _layered_n0_row1_at_a,
    _layered_n0_row2_at_a,
    _layered_n0_row3_at_a,
    _layered_n0_row4_at_b,
    _layered_n0_row5_at_b,
    _layered_n0_row6_at_b,
    _layered_n0_row7_at_b,
    _layered_n1_row1_at_a,
    _layered_n1_row2_at_a,
    _layered_n1_row3_at_a,
    _layered_n1_row4_at_a,
    _layered_n1_row5_at_b,
    _layered_n1_row6_at_b,
    _layered_n1_row7_at_b,
    _layered_n1_row8_at_b,
    _layered_n1_row9_at_b,
    _layered_n1_row10_at_b,
    _modal_determinant_n0,
    _modal_determinant_n0_layered,
    _modal_determinant_n1,
    _modal_determinant_n1_layered,
    flexural_dispersion,
    flexural_dispersion_layered,
    stoneley_dispersion,
    stoneley_dispersion_layered,
)
from tests._solver_media import (
    _stoneley_lf_truth,
)

# =====================================================================
# Plan item F (foundation): layered Stoneley dispersion API
# =====================================================================
#
# Foundation tests for the layered n=0 public API. The 7x7 layered
# modal determinant itself is the next step of plan item F; here we
# only exercise:
#
#   * the ``BoreholeLayer`` dataclass + validator,
#   * the empty-layers degenerate dispatch (must be bit-equivalent
#     to ``stoneley_dispersion``), and
#   * the ``NotImplementedError`` sentinel for non-empty layers.


def test_borehole_layer_dataclass_construction():
    layer = BoreholeLayer(vp=4000.0, vs=2200.0, rho=2300.0, thickness=0.005)
    assert layer.vp == 4000.0
    assert layer.vs == 2200.0
    assert layer.rho == 2300.0
    assert layer.thickness == 0.005


def test_stoneley_dispersion_layered_empty_layers_bit_matches_unlayered():
    """Degenerate single-interface case: ``layers=()`` must produce a
    slowness curve bit-identical to :func:`stoneley_dispersion`. This
    is the floating-point oracle that will continue to anchor the
    layered solver once the 7x7 modal determinant lands."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = np.linspace(500.0, 8000.0, 16)
    res_unlayered = stoneley_dispersion(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    res_layered = stoneley_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(),
    )
    np.testing.assert_array_equal(res_layered.slowness, res_unlayered.slowness)
    np.testing.assert_array_equal(res_layered.freq, res_unlayered.freq)
    assert res_layered.name == res_unlayered.name == "Stoneley"
    assert res_layered.azimuthal_order == 0


def test_stoneley_dispersion_layered_empty_layers_returns_borehole_mode():
    f = np.linspace(500.0, 5000.0, 5)
    res = stoneley_dispersion_layered(
        f,
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    assert isinstance(res, BoreholeMode)
    assert res.name == "Stoneley"
    assert res.azimuthal_order == 0


def test_stoneley_dispersion_layered_rejects_bad_layer_object():
    f = np.array([1000.0])
    with pytest.raises(ValueError, match="BoreholeLayer"):
        stoneley_dispersion_layered(
            f,
            vp=4500.0,
            vs=2500.0,
            rho=2400.0,
            vf=1500.0,
            rho_f=1000.0,
            a=0.1,
            layers=("not a layer",),
        )


@pytest.mark.parametrize(
    "kwargs, msg",
    [
        ({"vp": 0.0, "vs": 1.0, "rho": 1.0, "thickness": 1.0}, "positive"),
        ({"vp": 1.0, "vs": -1.0, "rho": 1.0, "thickness": 1.0}, "positive"),
        ({"vp": 1.0, "vs": 1.0, "rho": 0.0, "thickness": 1.0}, "positive"),
        ({"vp": 1.0, "vs": 2.0, "rho": 1.0, "thickness": 1.0}, "vp > vs"),
        ({"vp": 4.0, "vs": 2.0, "rho": 1.0, "thickness": 0.0}, "thickness"),
        ({"vp": 4.0, "vs": 2.0, "rho": 1.0, "thickness": -0.1}, "thickness"),
    ],
)
def test_stoneley_dispersion_layered_rejects_malformed_layer_params(kwargs, msg):
    f = np.array([1000.0])
    layer = BoreholeLayer(**kwargs)
    with pytest.raises(ValueError, match=msg):
        stoneley_dispersion_layered(
            f,
            vp=4500.0,
            vs=2500.0,
            rho=2400.0,
            vf=1500.0,
            rho_f=1000.0,
            a=0.1,
            layers=(layer,),
        )


def test_stoneley_dispersion_layered_accepts_list_for_layers():
    """``layers`` should accept any iterable that ``tuple(...)``
    consumes; the empty list must dispatch to the unlayered solver
    just like ``()``."""
    f = np.linspace(500.0, 5000.0, 4)
    res_tuple = stoneley_dispersion_layered(
        f,
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
        layers=(),
    )
    res_list = stoneley_dispersion_layered(
        f,
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
        layers=[],
    )
    np.testing.assert_array_equal(res_tuple.slowness, res_list.slowness)


# =====================================================================
# Plan item F.1.b.1 -- radial-wavenumber + Bessel-pack helpers
# =====================================================================


def _layered_typical_params():
    """Bound-regime fast-formation parameters with a slower-than-
    formation mudcake. Used as the default fixture for F.1.b
    helper / row tests."""
    return dict(
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
        layer=BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.005),
    )


def test_layered_radial_wavenumbers_bound_regime_returns_real_positive():
    """Above the bound-regime floor ``omega / min(V_S, V_S_m, V_f)``
    every wavenumber is real positive."""
    p = _layered_typical_params()
    omega = 2.0 * np.pi * 5000.0
    floor = omega / min(p["vs"], p["layer"].vs, p["vf"])
    kz = floor * 1.5
    F_f, p_m, s_m, pp, ss = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    for v in (F_f, p_m, s_m, pp, ss):
        assert np.isfinite(v)
        assert v > 0.0


def test_layered_radial_wavenumbers_below_bound_floor_returns_nan():
    """Below the bound-regime floor the slowest-wave radial
    wavenumber goes imaginary; numpy.sqrt of a negative real
    returns NaN. The helper passes NaN through (brentq-safe;
    no raise)."""
    p = _layered_typical_params()
    omega = 2.0 * np.pi * 5000.0
    # Pick kz strictly below ``omega / max(...)`` so every wavenumber
    # argument is negative (kz^2 - (omega/V)^2 < 0 needs
    # kz < omega/V for *every* wave speed V, which means
    # kz < omega/max(V)).
    fastest = max(p["vf"], p["vs"], p["layer"].vs, p["layer"].vp, p["vp"])
    kz = omega / fastest * 0.5
    with np.errstate(invalid="ignore"):
        F_f, p_m, s_m, pp, ss = _layered_n0_radial_wavenumbers(
            kz,
            omega,
            vp=p["vp"],
            vs=p["vs"],
            vf=p["vf"],
            layer=p["layer"],
        )
    for v in (F_f, p_m, s_m, pp, ss):
        assert np.isnan(v)


def test_layered_radial_wavenumbers_satisfy_definition():
    """Each wavenumber squared equals ``kz^2 - (omega / V)^2`` per
    substep F.1.a.1."""
    p = _layered_typical_params()
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(p["vs"], p["layer"].vs, p["vf"]) * 1.5
    F_f, p_m, s_m, pp, ss = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    assert F_f**2 == pytest.approx(kz**2 - (omega / p["vf"]) ** 2)
    assert p_m**2 == pytest.approx(kz**2 - (omega / p["layer"].vp) ** 2)
    assert s_m**2 == pytest.approx(kz**2 - (omega / p["layer"].vs) ** 2)
    assert pp**2 == pytest.approx(kz**2 - (omega / p["vp"]) ** 2)
    assert ss**2 == pytest.approx(kz**2 - (omega / p["vs"]) ** 2)


def test_layered_radial_wavenumbers_layer_equals_formation_collapses():
    """When the annulus material matches the formation, ``p_m == p``
    and ``s_m == s`` to floating-point precision -- the algebraic
    cornerstone of the substep F.1.a.6 self-check."""
    vp, vs = 4500.0, 2500.0
    p = dict(
        vp=vp,
        vs=vs,
        rho=2400.0,
        vf=1500.0,
        layer=BoreholeLayer(vp=vp, vs=vs, rho=2400.0, thickness=0.01),
    )
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, p["vf"]) * 1.5
    F_f, p_m, s_m, pp, ss = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    assert p_m == pp
    assert s_m == ss


def test_layered_bessel_pack_has_22_keys():
    """Substep F.1.b.1 plan: the pack covers 2 (fluid r=a) + 8
    (annulus P, both interfaces) + 8 (annulus S, both interfaces)
    + 4 (formation r=b) = 22 Bessel values."""
    p = _layered_typical_params()
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(p["vs"], p["layer"].vs, p["vf"]) * 1.5
    a = p["a"]
    b = a + p["layer"].thickness
    F_f, p_m, s_m, pp, ss = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    pack = _layered_n0_bessel_pack(F_f, p_m, s_m, pp, ss, a, b)
    assert len(pack) == 22
    expected_keys = {
        "I0_Ff_a",
        "I1_Ff_a",
        "I0_pm_a",
        "I1_pm_a",
        "K0_pm_a",
        "K1_pm_a",
        "I0_sm_a",
        "I1_sm_a",
        "K0_sm_a",
        "K1_sm_a",
        "I0_pm_b",
        "I1_pm_b",
        "K0_pm_b",
        "K1_pm_b",
        "I0_sm_b",
        "I1_sm_b",
        "K0_sm_b",
        "K1_sm_b",
        "K0_p_b",
        "K1_p_b",
        "K0_s_b",
        "K1_s_b",
    }
    assert set(pack.keys()) == expected_keys


def test_layered_bessel_pack_matches_scipy_directly():
    """Each entry in the pack must equal the corresponding direct
    scipy.special call to floating-point precision; this is the
    primary unit oracle for the helper."""
    from scipy import special

    p = _layered_typical_params()
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(p["vs"], p["layer"].vs, p["vf"]) * 1.5
    a = p["a"]
    b = a + p["layer"].thickness
    F_f, p_m, s_m, pp, ss = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    pack = _layered_n0_bessel_pack(F_f, p_m, s_m, pp, ss, a, b)

    # Fluid at r = a.
    assert pack["I0_Ff_a"] == float(special.iv(0, F_f * a))
    assert pack["I1_Ff_a"] == float(special.iv(1, F_f * a))

    # Annulus P/S at both interfaces; formation P/S at r = b.
    cases = [
        ("pm", p_m, ("a", "b")),
        ("sm", s_m, ("a", "b")),
    ]
    for wave, alpha, radii in cases:
        for r_label, r in zip(radii, (a, b)):
            x = alpha * r
            assert pack[f"I0_{wave}_{r_label}"] == float(special.iv(0, x))
            assert pack[f"I1_{wave}_{r_label}"] == float(special.iv(1, x))
            assert pack[f"K0_{wave}_{r_label}"] == float(special.kv(0, x))
            assert pack[f"K1_{wave}_{r_label}"] == float(special.kv(1, x))

    assert pack["K0_p_b"] == float(special.kv(0, pp * b))
    assert pack["K1_p_b"] == float(special.kv(1, pp * b))
    assert pack["K0_s_b"] == float(special.kv(0, ss * b))
    assert pack["K1_s_b"] == float(special.kv(1, ss * b))


def test_layered_bessel_pack_layer_equals_formation_p_columns_match():
    """Substep F.1.a.6 self-check at the Bessel level: when the
    annulus material matches the formation, the K-flavour pack
    entries at ``r = b`` for the annulus P (``K0_pm_b``) match the
    formation P (``K0_p_b``); same for S."""
    vp, vs = 4500.0, 2500.0
    layer = BoreholeLayer(vp=vp, vs=vs, rho=2400.0, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, 1500.0) * 1.5
    a = 0.1
    b = a + layer.thickness
    F_f, p_m, s_m, pp, ss = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=vp,
        vs=vs,
        vf=1500.0,
        layer=layer,
    )
    pack = _layered_n0_bessel_pack(F_f, p_m, s_m, pp, ss, a, b)
    assert pack["K0_pm_b"] == pack["K0_p_b"]
    assert pack["K1_pm_b"] == pack["K1_p_b"]
    assert pack["K0_sm_b"] == pack["K0_s_b"]
    assert pack["K1_sm_b"] == pack["K1_s_b"]


def test_layered_bessel_pack_propagates_nan_inputs():
    """Out-of-regime radial wavenumbers (NaN) propagate to NaN
    pack entries; the helper does not raise."""
    nan = float("nan")
    pack = _layered_n0_bessel_pack(nan, 10.0, 10.0, 10.0, 10.0, 0.1, 0.105)
    assert np.isnan(pack["I0_Ff_a"])
    assert np.isnan(pack["I1_Ff_a"])
    # Non-NaN inputs still produce finite Bessel values.
    assert np.isfinite(pack["K0_pm_a"])


# =====================================================================
# Plan item F.1.b.2.a -- row 1 of the n=0 layered determinant (r = a)
# =====================================================================


def _row1_test_setup():
    """Bound-regime kz / omega above every wavenumber floor for the
    typical fast-formation + soft-mudcake fixture."""
    p = _layered_typical_params()
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(p["vs"], p["layer"].vs, p["vf"]) * 1.5
    return p, omega, kz


def test_layered_row1_at_a_layer_equals_formation_per_element():
    """Substep F.1.a.6 self-check at the row level: with annulus
    properties identical to the formation, row 1 of the layered
    matrix has its (A, B_K, C_K) entries equal to ``M11, M12, M13``
    of :func:`_modal_determinant_n0` to floating-point precision."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n0_row1_at_a(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )

    # Reconstruct the corresponding entries of the unlayered matrix
    # without invoking the determinant routine: M_11, M_12, M_13 of
    # _modal_determinant_n0 (lifted from the docstring / source).
    F = float(np.sqrt(kz * kz - (omega / vf) ** 2))
    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    M11 = F * float(sp.iv(1, F * a)) / (rho_f * omega**2)
    M12 = p * float(sp.kv(1, p * a))
    M13 = kz * float(sp.kv(1, s * a))

    # Layer=formation collapses ``p_m -> p``, ``s_m -> s``. The
    # K-flavour columns (B_K, C_K) at indices 2, 4 then equal the
    # M_12 / M_13 entries of the unlayered matrix.
    assert row[0].real == pytest.approx(M11)
    assert row[2].real == pytest.approx(M12)
    assert row[4].real == pytest.approx(M13)
    assert abs(row[0].imag) < 1.0e-14
    assert abs(row[2].imag) < 1.0e-14
    assert abs(row[4].imag) < 1.0e-14


def test_layered_row1_at_a_formation_columns_are_zero():
    """Sparsity: at ``r = a`` the formation columns (indices 5, 6)
    are zero -- the formation half-space ``r > b`` doesn't touch
    the fluid-annulus interface."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row1_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    assert row[5] == 0.0
    assert row[6] == 0.0


def test_layered_row1_at_a_is_real_in_bound_regime():
    """Substep F.1.a.5 phase rescale: post-rescale row entries are
    purely real in the bound regime. Any non-zero imaginary part
    flags a sign error in the C-flavour rescaling."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row1_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_row1_at_a_i_k_sign_flip():
    """Substep F.1.a.2 sign convention: the I-flavour annulus
    columns (B_I, C_I) carry the opposite sign of the K-flavour
    counterparts (B_K, C_K) on the B amplitudes, the *same* sign on
    the C amplitudes. Specifically:

        row[1] / row[2] == -I_1(p_m a) / K_1(p_m a)    (B_I vs B_K)
        row[3] / row[4] == +I_1(s_m a) / K_1(s_m a)    (C_I vs C_K)
    """
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    from scipy import special as sp

    row = _layered_n0_row1_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )

    expected_ratio_B = -float(sp.iv(1, p_m * p["a"])) / float(sp.kv(1, p_m * p["a"]))
    expected_ratio_C = +float(sp.iv(1, s_m * p["a"])) / float(sp.kv(1, s_m * p["a"]))
    assert row[1].real / row[2].real == pytest.approx(expected_ratio_B)
    assert row[3].real / row[4].real == pytest.approx(expected_ratio_C)


# =====================================================================
# Plan item F.1.b.2.b -- row 2 of the n=0 layered determinant (r = a)
# =====================================================================


def test_layered_row2_at_a_layer_equals_formation_per_element():
    """At layer=formation, row 2's (A, B_K, C_K) entries match
    M21, M22, M23 of :func:`_modal_determinant_n0` to floating-
    point precision -- the primary correctness oracle for the row's
    Lame-reduction bookkeeping."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n0_row2_at_a(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )

    F = float(np.sqrt(kz * kz - (omega / vf) ** 2))
    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    M21 = -float(sp.iv(0, F * a))
    M22 = -mu * (
        two_kz2_minus_kS2 * float(sp.kv(0, p * a))
        + 2.0 * p * float(sp.kv(1, p * a)) / a
    )
    M23 = -2.0 * kz * mu * (s * float(sp.kv(0, s * a)) + float(sp.kv(1, s * a)) / a)

    assert row[0].real == pytest.approx(M21)
    assert row[2].real == pytest.approx(M22)
    assert row[4].real == pytest.approx(M23)


def test_layered_row2_at_a_formation_columns_are_zero():
    """Sparsity: at ``r = a`` the formation columns (5, 6) are
    zero."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row2_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    assert row[5] == 0.0
    assert row[6] == 0.0


def test_layered_row2_at_a_is_real_in_bound_regime():
    """Substep F.1.a.5 phase rescale: post-rescale row 2 entries
    are purely real in the bound regime."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row2_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_row2_at_a_i_flavour_columns_match_derivation():
    """The I-flavour annulus columns (B_I, C_I) have no single-
    interface analog. Cross-check them against the closed-form
    expressions read directly off the substep-F.1.a.3 derivation:

        row[1] (B_I) =
            -mu_m [(2 k_z^2 - k_Sm^2) I_0(p_m a) - 2 p_m I_1(p_m a) / a]
        row[3] (C_I) =
            +2 mu_m k_z [s_m I_0(s_m a) - I_1(s_m a) / a]
    """
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    from scipy import special as sp

    mu_m = p["layer"].rho * p["layer"].vs ** 2
    kSm2 = (omega / p["layer"].vs) ** 2
    two_kz2_minus_kSm2 = 2.0 * kz * kz - kSm2
    a = p["a"]

    expected_BI = -mu_m * (
        two_kz2_minus_kSm2 * float(sp.iv(0, p_m * a))
        - 2.0 * p_m * float(sp.iv(1, p_m * a)) / a
    )
    expected_CI = (
        +2.0
        * mu_m
        * kz
        * (s_m * float(sp.iv(0, s_m * a)) - float(sp.iv(1, s_m * a)) / a)
    )

    row = _layered_n0_row2_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    assert row[1].real == pytest.approx(expected_BI)
    assert row[3].real == pytest.approx(expected_CI)


# =====================================================================
# Plan item F.1.b.2.c -- row 3 of the n=0 layered determinant (r = a)
# =====================================================================


def test_layered_row3_at_a_layer_equals_formation_per_element():
    """At layer=formation, row 3's (A, B_K, C_K) entries match
    M31 (= 0), M32, M33 of :func:`_modal_determinant_n0` to
    floating-point precision."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n0_row3_at_a(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )

    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    M32 = 2.0 * kz * p * mu * float(sp.kv(1, p * a))
    M33 = mu * two_kz2_minus_kS2 * float(sp.kv(1, s * a))

    assert row[0] == 0.0  # M31 = 0 (fluid no shear)
    assert row[2].real == pytest.approx(M32)
    assert row[4].real == pytest.approx(M33)


def test_layered_row3_at_a_fluid_column_is_zero():
    """Row 3 column 0 (the A / fluid-pressure amplitude) is
    identically zero -- the fluid carries no shear stress so it
    contributes nothing to the ``sigma_rz = 0`` BC."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row3_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    assert row[0] == 0.0


def test_layered_row3_at_a_formation_columns_are_zero():
    """Sparsity: at ``r = a`` the formation columns (5, 6) are
    zero."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row3_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    assert row[5] == 0.0
    assert row[6] == 0.0


def test_layered_row3_at_a_is_real_in_bound_regime():
    """Substep F.1.a.5: the full ``row * i`` plus column-by-(-i)
    rescale lands row 3 in real form."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row3_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_row3_at_a_i_k_sign_flip():
    """Same I-K sign structure as row 1 (different physics, same
    pattern):

        row[1] / row[2] == -I_1(p_m a) / K_1(p_m a)    (B_I vs B_K)
        row[3] / row[4] == +I_1(s_m a) / K_1(s_m a)    (C_I vs C_K)
    """
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    from scipy import special as sp

    row = _layered_n0_row3_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )

    expected_ratio_B = -float(sp.iv(1, p_m * p["a"])) / float(sp.kv(1, p_m * p["a"]))
    expected_ratio_C = +float(sp.iv(1, s_m * p["a"])) / float(sp.kv(1, s_m * p["a"]))
    assert row[1].real / row[2].real == pytest.approx(expected_ratio_B)
    assert row[3].real / row[4].real == pytest.approx(expected_ratio_C)


# =====================================================================
# Plan item F.1.b.3.a -- row 4 of the n=0 layered determinant (r = b)
# =====================================================================
#
# First of the four interface-continuity rows at the second
# interface ``r = b``. Unlike rows 1-3, no single-interface analog
# exists, so the primary correctness oracle is the substep-F.1.a.6
# K-flavour cancellation identity at layer=formation.


def test_layered_row4_at_b_layer_equals_formation_K_flavour_cancels():
    """Substep F.1.a.6 self-check at the row level: at layer=
    formation the K-flavour annulus and formation columns of row 4
    cancel pair-wise. Specifically:

        row4[2] (B_K) + row4[5] (B) == 0
        row4[4] (C_K) + row4[6] (C) == 0

    Physically: when the annulus material matches the formation,
    the second interface is fictitious, and the outgoing-wave
    K-flavour contributions from both sides represent the same
    field, so continuity is trivially satisfied. This is the
    central correctness invariant for rows 4-7."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n0_row4_at_b(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )
    assert row[2].real + row[5].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[4].real + row[6].real == pytest.approx(0.0, abs=1.0e-14)


def test_layered_row4_at_b_fluid_column_is_zero():
    """The fluid lives at ``r < a``; it does not reach ``r = b``.
    Column 0 (A) is identically zero in row 4."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row4_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    assert row[0] == 0.0


def test_layered_row4_at_b_is_real_in_bound_regime():
    """Substep F.1.a.5: post-rescale row 4 is real in the bound
    regime (no row scaling; column-by-(-i) on C_I, C_K, C kills
    the explicit ``i`` factors)."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row4_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_row4_at_b_matches_closed_form_per_column():
    """Cross-check every non-zero entry against the substep-F.1.a.2
    closed form, evaluated at ``r = b``. No single-interface analog
    to compare against, so this is the per-element transcription
    check."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, p_form, s_form = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n0_row4_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=a,
        layer=p["layer"],
    )

    assert row[1].real == pytest.approx(+p_m * float(sp.iv(1, p_m * b)))
    assert row[2].real == pytest.approx(-p_m * float(sp.kv(1, p_m * b)))
    assert row[3].real == pytest.approx(-kz * float(sp.iv(1, s_m * b)))
    assert row[4].real == pytest.approx(-kz * float(sp.kv(1, s_m * b)))
    assert row[5].real == pytest.approx(+p_form * float(sp.kv(1, p_form * b)))
    assert row[6].real == pytest.approx(+kz * float(sp.kv(1, s_form * b)))


def test_layered_row4_at_b_annulus_K_sign_opposite_to_row1_at_a():
    """Sign-flow consistency between the two interfaces. In row 1
    (``u_r^{(f)} - u_r^{(m)} = 0`` at r=a) the annulus B_K
    coefficient is ``+p_m K_1(p_m a)``. In row 4
    (``u_r^{(m)} - u_r^{(s)} = 0`` at r=b) the same physical
    quantity is ``-p_m K_1(p_m b)`` -- opposite sign because the
    annulus appears with opposite sign in the two BCs."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, _, _, _ = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    from scipy import special as sp

    row4 = _layered_n0_row4_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    b = p["a"] + p["layer"].thickness
    assert row4[2].real == pytest.approx(-p_m * float(sp.kv(1, p_m * b)))


# =====================================================================
# Plan item F.1.b.3.b -- row 5 of the n=0 layered determinant (r = b)
# =====================================================================
#
# Row 5 is the u_z continuity BC at the second interface. Genuinely
# new at the layered case: no single-interface analog because the
# fluid-solid interface at r = a replaces u_z continuity with
# sigma_rz = 0. Imaginary-power pattern is the *opposite* of row 4
# (B-imag / C-real pre-rescale, like rows 3 and 7); the post-
# rescale row * i scaling is what makes the row real.


def test_layered_row5_at_b_layer_equals_formation_K_flavour_cancels():
    """Substep F.1.a.6 self-check: at layer=formation the K-flavour
    annulus + formation columns of row 5 cancel pair-wise.

        row5[2] (B_K) + row5[5] (B) == 0
        row5[4] (C_K) + row5[6] (C) == 0
    """
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n0_row5_at_b(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )
    assert row[2].real + row[5].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[4].real + row[6].real == pytest.approx(0.0, abs=1.0e-14)


def test_layered_row5_at_b_fluid_column_is_zero():
    """Fluid lives at ``r < a``; column 0 (A) is identically zero."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row5_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    assert row[0] == 0.0


def test_layered_row5_at_b_is_real_in_bound_regime():
    """Substep F.1.a.5 phase rescale: the row * i scaling on row 5
    (z-derivative-bearing) plus column-by-(-i) on C_I, C_K, C
    leaves the post-rescale row real-valued in the bound regime.
    Forgetting the row * i is the most direct transcription error
    F.1.a.5 calls out -- this test catches it."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row5_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_row5_at_b_matches_closed_form_per_column():
    """Per-column transcription check against substep F.1.a.2 at
    r = b, with the row * i / col * -i rescaling applied. Notable
    feature: row 5 uses degree-0 Bessel functions (I_0 / K_0),
    distinguishing it from rows 1, 4, 6 (degree-1)."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, p_form, s_form = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n0_row5_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=a,
        layer=p["layer"],
    )

    assert row[1].real == pytest.approx(-kz * float(sp.iv(0, p_m * b)))
    assert row[2].real == pytest.approx(-kz * float(sp.kv(0, p_m * b)))
    assert row[3].real == pytest.approx(+s_m * float(sp.iv(0, s_m * b)))
    assert row[4].real == pytest.approx(-s_m * float(sp.kv(0, s_m * b)))
    assert row[5].real == pytest.approx(+kz * float(sp.kv(0, p_form * b)))
    assert row[6].real == pytest.approx(+s_form * float(sp.kv(0, s_form * b)))


def test_layered_row5_at_b_uses_degree0_not_degree1_bessel():
    """Structural check distinguishing u_z (row 5) from u_r (row 4):
    at the same kz, omega, layer, the B_K coefficient in row 5 is
    proportional to ``K_0(p_m b)`` while row 4's B_K coefficient is
    proportional to ``K_1(p_m b)``. The Bessel-index difference
    flows from the ``u_z = i k_z phi`` term (no derivative) vs
    ``u_r = d_r phi`` (one derivative; bumps the Bessel index)."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, _, _, _ = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    from scipy import special as sp

    row4 = _layered_n0_row4_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    row5 = _layered_n0_row5_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    b = p["a"] + p["layer"].thickness

    # row 4: B_K column = -p_m K_1(p_m b)
    # row 5: B_K column = -k_z K_0(p_m b)
    # Their ratio should be (p_m K_1) / (k_z K_0).
    assert row4[2].real / row5[2].real == pytest.approx(
        (p_m * float(sp.kv(1, p_m * b))) / (kz * float(sp.kv(0, p_m * b)))
    )


# =====================================================================
# Plan item F.1.b.3.c -- row 6 of the n=0 layered determinant (r = b)
# =====================================================================


def test_layered_row6_at_b_layer_equals_formation_K_flavour_cancels():
    """Substep F.1.a.6 self-check: at layer=formation the K-flavour
    annulus and formation columns of row 6 cancel pair-wise.

        row6[2] (B_K) + row6[5] (B) == 0
        row6[4] (C_K) + row6[6] (C) == 0
    """
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n0_row6_at_b(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )
    assert row[2].real + row[5].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[4].real + row[6].real == pytest.approx(0.0, abs=1.0e-14)


def test_layered_row6_at_b_fluid_column_is_zero():
    """Fluid lives at ``r < a``; column 0 (A) is identically zero."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row6_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    assert row[0] == 0.0


def test_layered_row6_at_b_is_real_in_bound_regime():
    """Substep F.1.a.5: post-rescale row 6 is real in the bound
    regime. Same imaginary-power pattern as rows 1, 4 (B-real,
    C-imag pre-rescale) so no row scaling needed; only the
    column-by-(-i) on C_I, C_K, C is applied."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row6_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_row6_at_b_matches_closed_form_per_column():
    """Per-column transcription check against substep F.1.a.3 at
    r = b. Row 6 carries the Lame combination ``(2 k_z^2 - k_Sm^2)``
    on each B / C column, identical in structure to the row-2 form
    but evaluated at r = b with non-zero formation columns."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, p_form, s_form = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n0_row6_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=a,
        layer=p["layer"],
    )

    mu_m = p["layer"].rho * p["layer"].vs ** 2
    kSm2 = (omega / p["layer"].vs) ** 2
    two_kz2_minus_kSm2 = 2.0 * kz * kz - kSm2
    mu = p["rho"] * p["vs"] ** 2
    kS2 = (omega / p["vs"]) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    expected_BI = mu_m * (
        two_kz2_minus_kSm2 * float(sp.iv(0, p_m * b))
        - 2.0 * p_m * float(sp.iv(1, p_m * b)) / b
    )
    expected_BK = mu_m * (
        two_kz2_minus_kSm2 * float(sp.kv(0, p_m * b))
        + 2.0 * p_m * float(sp.kv(1, p_m * b)) / b
    )
    expected_CI = (
        -2.0
        * kz
        * mu_m
        * (s_m * float(sp.iv(0, s_m * b)) - float(sp.iv(1, s_m * b)) / b)
    )
    expected_CK = (
        +2.0
        * kz
        * mu_m
        * (s_m * float(sp.kv(0, s_m * b)) + float(sp.kv(1, s_m * b)) / b)
    )
    expected_B = -mu * (
        two_kz2_minus_kS2 * float(sp.kv(0, p_form * b))
        + 2.0 * p_form * float(sp.kv(1, p_form * b)) / b
    )
    expected_C = (
        -2.0
        * kz
        * mu
        * (s_form * float(sp.kv(0, s_form * b)) + float(sp.kv(1, s_form * b)) / b)
    )

    assert row[1].real == pytest.approx(expected_BI)
    assert row[2].real == pytest.approx(expected_BK)
    assert row[3].real == pytest.approx(expected_CI)
    assert row[4].real == pytest.approx(expected_CK)
    assert row[5].real == pytest.approx(expected_B)
    assert row[6].real == pytest.approx(expected_C)


def test_layered_row6_at_b_layer_equals_formation_annulus_K_matches_negated_row2():
    """At layer=formation, row 6's annulus K-flavour entries (B_K,
    C_K) -- which carry the unnegated stress form -- equal the
    *negation* of row 2's M22, M23-equivalents evaluated at r = b
    (row 2 uses the negated ``-(sigma_rr + P)`` convention; row 6
    uses unnegated continuity). This pins down the convention
    choice and confirms the two row builders are using the same
    underlying stress formula."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5
    b = a + layer.thickness

    row6 = _layered_n0_row6_at_b(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )

    # Compute an "M22-like at r = b" entry using the row 2 formula.
    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    M22_at_b = -mu * (
        two_kz2_minus_kS2 * float(sp.kv(0, p * b))
        + 2.0 * p * float(sp.kv(1, p * b)) / b
    )
    M23_at_b = (
        -2.0 * kz * mu * (s * float(sp.kv(0, s * b)) + float(sp.kv(1, s * b)) / b)
    )

    # row6[2] is unnegated stress; M22_at_b is negated. They differ
    # by sign, so row6[2] = -M22_at_b at layer=formation.
    assert row6[2].real == pytest.approx(-M22_at_b)
    assert row6[4].real == pytest.approx(-M23_at_b)


# =====================================================================
# Plan item F.1.b.3.d -- row 7 of the n=0 layered determinant (r = b)
# =====================================================================
#
# Final row of the 7x7 layered determinant; closes F.1.b.3 and
# unblocks F.1.b.4 assembly. Same z-derivative-bearing pattern as
# rows 3 and 5; structurally analogous to row 3 at r=a but with
# non-zero formation columns.


def test_layered_row7_at_b_layer_equals_formation_K_flavour_cancels():
    """Substep F.1.a.6 self-check: at layer=formation the K-flavour
    annulus and formation columns of row 7 cancel pair-wise.

        row7[2] (B_K) + row7[5] (B) == 0
        row7[4] (C_K) + row7[6] (C) == 0
    """
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n0_row7_at_b(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )
    assert row[2].real + row[5].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[4].real + row[6].real == pytest.approx(0.0, abs=1.0e-14)


def test_layered_row7_at_b_fluid_column_is_zero():
    """Row 7 column 0 (A) is identically zero. The fluid carries
    no shear AND lives at r < a, so it contributes nothing to the
    sigma_rz continuity at r=b."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row7_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    assert row[0] == 0.0


def test_layered_row7_at_b_is_real_in_bound_regime():
    """Substep F.1.a.5: post-rescale row 7 is real in the bound
    regime. Row 7 is z-derivative-bearing (B-imag / C-real pre-
    rescale, like rows 3 and 5), so it requires the row * i scaling
    plus column-by-(-i) on C_I, C_K, C. Forgetting the row * i is
    the same easy-to-miss F.1.a.5 error as in row 5; this test is
    the safety net."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n0_row7_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_row7_at_b_matches_closed_form_per_column():
    """Per-column transcription check against substep F.1.a.3 at
    r = b. Row 7 carries the Lame combination ``(2 k_z^2 - k_Sm^2)``
    on the C-flavour columns and the ``2 k_z mu_m p_m`` factor on
    the B-flavour columns (single-Bessel-term entries throughout,
    like row 4)."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, p_form, s_form = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n0_row7_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=a,
        layer=p["layer"],
    )

    mu_m = p["layer"].rho * p["layer"].vs ** 2
    kSm2 = (omega / p["layer"].vs) ** 2
    two_kz2_minus_kSm2 = 2.0 * kz * kz - kSm2
    mu = p["rho"] * p["vs"] ** 2
    kS2 = (omega / p["vs"]) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    assert row[1].real == pytest.approx(
        -2.0 * kz * mu_m * p_m * float(sp.iv(1, p_m * b))
    )
    assert row[2].real == pytest.approx(
        +2.0 * kz * mu_m * p_m * float(sp.kv(1, p_m * b))
    )
    assert row[3].real == pytest.approx(
        +mu_m * two_kz2_minus_kSm2 * float(sp.iv(1, s_m * b))
    )
    assert row[4].real == pytest.approx(
        +mu_m * two_kz2_minus_kSm2 * float(sp.kv(1, s_m * b))
    )
    assert row[5].real == pytest.approx(
        -2.0 * kz * mu * p_form * float(sp.kv(1, p_form * b))
    )
    assert row[6].real == pytest.approx(
        -mu * two_kz2_minus_kS2 * float(sp.kv(1, s_form * b))
    )


def test_layered_row7_at_b_layer_equals_formation_annulus_K_matches_row3_at_b():
    """At layer=formation, row 7's annulus K-flavour entries (B_K,
    C_K) match the ``M32, M33`` form of :func:`_modal_determinant_n0`
    (the n=0 single-interface row 3) evaluated at ``r = b`` instead
    of ``r = a``. Confirms row 7 and row 3 share the same
    underlying ``sigma_rz`` formula -- they differ only in which
    interface they live at."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5
    b = a + layer.thickness

    row7 = _layered_n0_row7_at_b(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )

    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    M32_at_b = 2.0 * kz * p * mu * float(sp.kv(1, p * b))
    M33_at_b = mu * two_kz2_minus_kS2 * float(sp.kv(1, s * b))

    assert row7[2].real == pytest.approx(M32_at_b)
    assert row7[4].real == pytest.approx(M33_at_b)


# =====================================================================
# Plan item F.1.b.4 -- assembly + dispatch
# =====================================================================
#
# Closes the F.1.b chain. Tests fall into two groups:
#
#   * ``_modal_determinant_n0_layered``: real-valued in bound regime;
#     evaluates without raising; behaves correctly at the layer=
#     formation degenerate point.
#   * ``stoneley_dispersion_layered`` end-to-end: layer=formation
#     reproduces ``stoneley_dispersion`` slowness curve to
#     ``rtol=1e-8``; thickness->0 limit ditto; dispatched correctly.


def test_modal_determinant_n0_layered_is_real_in_bound_regime():
    """Substep F.1.a.5 phase rescale: each row builder applies the
    rescale internally, so the assembled 7x7 is real-valued in
    the bound regime."""
    p, omega, kz = _row1_test_setup()
    det = _modal_determinant_n0_layered(
        kz,
        omega,
        p["vp"],
        p["vs"],
        p["rho"],
        p["vf"],
        p["rho_f"],
        p["a"],
        layer=p["layer"],
    )
    assert np.isfinite(det)
    assert isinstance(det, float)


def test_modal_determinant_n0_layered_layer_equals_formation_root_matches_unlayered():
    """The substep-F.1.a.6 self-check at the determinant level: at
    layer=formation, the layered determinant has the same
    Stoneley root as :func:`_modal_determinant_n0`. The two
    determinants are not numerically equal (the 7x7 has a
    different overall scale than the 3x3), but they share the
    same root in ``k_z``.

    Verify by: (a) computing the Stoneley root from
    ``stoneley_dispersion`` (the 3x3); (b) evaluating the layered
    7x7 at that root; (c) checking ``|det_layered|`` is small
    relative to its order of magnitude away from the root."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0

    bound = stoneley_dispersion(
        np.array([5000.0]),
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    kz_root = float(bound.slowness[0]) * omega

    det_at_root = _modal_determinant_n0_layered(
        kz_root,
        omega,
        vp,
        vs,
        rho,
        vf,
        rho_f,
        a,
        layer=layer,
    )
    det_off_root = _modal_determinant_n0_layered(
        kz_root * 1.05,
        omega,
        vp,
        vs,
        rho,
        vf,
        rho_f,
        a,
        layer=layer,
    )
    # Not strictly zero (different matrix size + numerical noise),
    # but several orders of magnitude smaller than away from root.
    assert abs(det_at_root) < abs(det_off_root) * 1.0e-3


def test_stoneley_dispersion_layered_layer_equals_formation_matches_unlayered():
    """End-to-end integration test: with a layer whose properties
    match the formation, the layered solver produces the same
    Stoneley dispersion curve as the unlayered solver to
    ``rtol=1e-8``. This is the floating-point oracle for the
    entire F.1.b chain. Any algebra error accumulated across the
    seven row builders surfaces here."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    f = np.linspace(500.0, 8000.0, 16)

    res_unlayered = stoneley_dispersion(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    res_layered = stoneley_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer,),
    )
    np.testing.assert_allclose(
        res_layered.slowness,
        res_unlayered.slowness,
        rtol=1.0e-8,
        equal_nan=True,
    )


def test_stoneley_dispersion_layered_thickness_zero_limit():
    """As ``layer.thickness -> 0`` (with arbitrary layer material),
    the layered solver continuously approaches the unlayered
    answer. Algebraic identity: in the limit ``b -> a``, the rows
    at r=b approach the rows at r=a, the second interface degenerates,
    and the converged k_z must approach the single-interface root."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = 5000.0

    res_unlayered = stoneley_dispersion(
        np.array([f]),
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )

    # Even a "different" layer with vanishing thickness should
    # converge to the unlayered Stoneley slowness.
    layer_thin = BoreholeLayer(
        vp=3500.0,
        vs=1800.0,
        rho=2100.0,
        thickness=1.0e-9,
    )
    res_thin = stoneley_dispersion_layered(
        np.array([f]),
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer_thin,),
    )
    assert res_thin.slowness[0] == pytest.approx(
        res_unlayered.slowness[0],
        rel=1.0e-4,
    )


def test_stoneley_dispersion_layered_non_trivial_layer_runs():
    """End-to-end smoke: a soft mudcake layer different from the
    formation produces a finite slowness curve in the bound
    regime. No analytic oracle is asserted here (that's the
    Schmitt 1988 fig 6 validation in F.1.d); the test just
    confirms the dispatch + matrix + brentq + bracket all wire up
    without raising."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.005)
    f = np.linspace(1000.0, 8000.0, 8)

    res = stoneley_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer,),
    )
    assert res.name == "Stoneley"
    assert res.azimuthal_order == 0
    assert res.slowness.shape == f.shape
    # All slownesses finite in this bound-regime fast-formation case.
    assert np.all(np.isfinite(res.slowness))
    # All slownesses above the slowest-shear floor.
    assert np.all(res.slowness > 1.0 / max(vs, layer.vs, vf))


def test_stoneley_dispersion_layered_softer_mudcake_slows_down():
    """Sanity check: a mudcake softer than the formation
    (lower V_S) increases the Stoneley slowness compared to
    the unlayered formation -- the qualitative effect documented
    in Schmitt 1988 fig 6 and the F.1.d validation target."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = np.array([3000.0])

    res_unlayered = stoneley_dispersion(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    soft_layer = BoreholeLayer(
        vp=3500.0,
        vs=1800.0,
        rho=2100.0,
        thickness=0.01,
    )
    res_layered = stoneley_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(soft_layer,),
    )
    assert res_layered.slowness[0] > res_unlayered.slowness[0]


# =====================================================================
# Plan item F.1.d -- validation tightening on top of F.1.b.4
# =====================================================================
#
# Hardening tests for the assembled layered Stoneley solver. Each
# tests an asymptotic / self-consistency property that the
# layer=formation regression alone doesn't pin down.


def test_stoneley_dispersion_layered_thickness_dominant_limit():
    """As the layer thickness grows much larger than the field's
    radial extent at r = a (set roughly by ``1 / p_m``), the
    second interface becomes irrelevant and the Stoneley wave
    propagates as if the *layer* material were the formation
    half-space.

    Concretely: layered_dispersion(formation=X, layer=Y,
    thickness=large) -> stoneley_dispersion(formation=Y) as
    thickness * p_m -> infty. We test at a frequency high enough
    that p_m * thickness >> 1 with a 0.5 m thick layer.
    """
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer_vp, layer_vs, layer_rho = 3500.0, 1800.0, 2100.0
    f = np.array([5000.0])
    # 0.5 m -- far thicker than any physical mudcake, just to
    # stress-test the limit. p_m * thickness is well above 1
    # at f = 5 kHz, so the field at r = b is exponentially small.
    layer = BoreholeLayer(
        vp=layer_vp,
        vs=layer_vs,
        rho=layer_rho,
        thickness=0.5,
    )

    res_layered = stoneley_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer,),
    )
    # Limit value: unlayered Stoneley with the LAYER properties as
    # the formation half-space.
    res_limit = stoneley_dispersion(
        f,
        vp=layer_vp,
        vs=layer_vs,
        rho=layer_rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    # Tolerance is loose because the limit is asymptotic; tighter
    # match would need an even thicker layer (numerically delicate
    # because K_n decays exponentially).
    assert res_layered.slowness[0] == pytest.approx(
        res_limit.slowness[0],
        rel=1.0e-3,
    )


def test_modal_determinant_n0_layered_vanishes_at_converged_root():
    """Self-consistency: at the converged ``k_z`` returned by
    :func:`stoneley_dispersion_layered`, the layered determinant
    is small compared to its off-root value. Tighter check than
    the layer=formation det-at-root test (which only verifies the
    F.1.a.6 self-check); this works for any non-trivial layer."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.005)
    f = 5000.0
    omega = 2.0 * np.pi * f

    res = stoneley_dispersion_layered(
        np.array([f]),
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer,),
    )
    kz_root = float(res.slowness[0]) * omega

    det_at = _modal_determinant_n0_layered(
        kz_root,
        omega,
        vp,
        vs,
        rho,
        vf,
        rho_f,
        a,
        layer=layer,
    )
    det_off = _modal_determinant_n0_layered(
        kz_root * 1.05,
        omega,
        vp,
        vs,
        rho,
        vf,
        rho_f,
        a,
        layer=layer,
    )
    # brentq returns a converged root, so |det_at| should be at
    # least ~6 orders of magnitude smaller than |det_off|.
    assert abs(det_at) < abs(det_off) * 1.0e-6


def test_stoneley_dispersion_layered_multiple_frequencies_bound_regime():
    """Smoke test across a wide frequency band to confirm the
    bracket + brentq combination stays well-behaved over a range
    spanning ~3 decades. The Stoneley slowness *decreases*
    monotonically with frequency in a fast-formation borehole
    (the wave speeds up toward a fluid-loaded Rayleigh /
    Scholte-like asymptote at high f); same dispersion direction
    as the unlayered case from
    ``test_stoneley_dispersion_speeds_up_with_frequency``."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.005)
    f = np.geomspace(100.0, 20000.0, 25)

    res = stoneley_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer,),
    )
    assert np.all(np.isfinite(res.slowness))
    diffs = np.diff(res.slowness)
    # Slowness decreases with frequency in a fast formation
    # (phase velocity increases). Monotonic decrease across the
    # full 100 Hz - 20 kHz band; allow a tiny tolerance for
    # near-asymptote flatness.
    assert np.all(diffs < 1.0e-9)


def test_stoneley_dispersion_layered_low_f_layer_shifts_off_formation_white():
    """At very low frequency the layer is NOT invisible: even a
    5 mm mudcake at 10 Hz (wavelength ~2 km) shifts the Stoneley
    slowness off the unlayered White (1983) formation-only
    closed-form. Reason: the layer sits at the borehole wall
    where the radial field amplitude is highest, so the effective
    near-wall shear modulus is the layer's, not the formation's.

    Verify the layered slowness lies *between* the two unlayered
    White-formula values (formation and layer-as-formation),
    closer to the formation value because the formation half-space
    still provides the bulk of the back-field-decay support at
    low f."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    layer_vp, layer_vs, layer_rho = 3500.0, 1800.0, 2100.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(
        vp=layer_vp,
        vs=layer_vs,
        rho=layer_rho,
        thickness=0.005,
    )
    f = np.array([10.0])

    res = stoneley_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer,),
    )
    s_formation = _stoneley_lf_truth(vs, rho, vf, rho_f)
    s_layer = _stoneley_lf_truth(layer_vs, layer_rho, vf, rho_f)
    # layer is softer than formation -> s_layer > s_formation.
    # The layered slowness must lie between the two.
    assert s_formation < res.slowness[0] < s_layer


# =====================================================================
# Plan item F.2.0 -- public-API foundation for layered flexural
# =====================================================================
#
# Sister of the F.1 foundation tests. The 10x10 layered modal
# determinant is scheduled in plan item F.2; here we only exercise
# the public-API surface (validation, empty-layers dispatch,
# NotImplementedError sentinel for non-empty).


def test_flexural_dispersion_layered_empty_layers_bit_matches_unlayered():
    """Degenerate single-interface case: ``layers=()`` must produce
    a slowness curve bit-identical to :func:`flexural_dispersion`.
    Floating-point oracle that will continue to anchor the layered
    flexural solver once the 10x10 modal determinant lands in F.2.d.
    """
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = np.linspace(2000.0, 8000.0, 12)
    res_unlayered = flexural_dispersion(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    res_layered = flexural_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(),
    )
    np.testing.assert_array_equal(res_layered.slowness, res_unlayered.slowness)
    np.testing.assert_array_equal(res_layered.freq, res_unlayered.freq)
    assert res_layered.name == res_unlayered.name == "flexural"
    assert res_layered.azimuthal_order == 1


def test_flexural_dispersion_layered_empty_layers_returns_borehole_mode():
    f = np.linspace(2000.0, 5000.0, 5)
    res = flexural_dispersion_layered(
        f,
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    assert isinstance(res, BoreholeMode)
    assert res.name == "flexural"
    assert res.azimuthal_order == 1


def test_flexural_dispersion_layered_fast_formation_dispatches_to_complex_path():
    """Fast-formation layered flexural (``V_S > V_f`` with a
    non-empty layer) dispatches to the complex-determinant path
    via ``_modal_determinant_n1_cased_complex`` and
    ``_flexural_dispersion_fast_formation_layered``. The earlier
    ``NotImplementedError`` from F.2.d is gone -- fast-formation
    layered flexural is now a supported regime."""
    f = np.array([2000.0, 4000.0])
    # Fast formation: vs (2500) > vf (1500).
    layer = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.01)
    res = flexural_dispersion_layered(
        f,
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
        layers=(layer,),
    )
    assert isinstance(res, BoreholeMode)
    assert res.name == "flexural"
    assert res.azimuthal_order == 1
    # Bound mode: real-valued slowness (or NaN outside the
    # geometric cutoff). Either way, no exception.
    assert res.slowness.dtype == np.float64
    assert res.attenuation_per_meter is None


def test_flexural_dispersion_layered_rejects_bad_layer_object():
    f = np.array([2000.0])
    with pytest.raises(ValueError, match="BoreholeLayer"):
        flexural_dispersion_layered(
            f,
            vp=4500.0,
            vs=2500.0,
            rho=2400.0,
            vf=1500.0,
            rho_f=1000.0,
            a=0.1,
            layers=("not a layer",),
        )


@pytest.mark.parametrize(
    "kwargs, msg",
    [
        ({"vp": 0.0, "vs": 1.0, "rho": 1.0, "thickness": 1.0}, "positive"),
        ({"vp": 1.0, "vs": -1.0, "rho": 1.0, "thickness": 1.0}, "positive"),
        ({"vp": 1.0, "vs": 1.0, "rho": 0.0, "thickness": 1.0}, "positive"),
        ({"vp": 1.0, "vs": 2.0, "rho": 1.0, "thickness": 1.0}, "vp > vs"),
        ({"vp": 4.0, "vs": 2.0, "rho": 1.0, "thickness": 0.0}, "thickness"),
        ({"vp": 4.0, "vs": 2.0, "rho": 1.0, "thickness": -0.1}, "thickness"),
    ],
)
def test_flexural_dispersion_layered_rejects_malformed_layer_params(kwargs, msg):
    f = np.array([2000.0])
    layer = BoreholeLayer(**kwargs)
    with pytest.raises(ValueError, match=msg):
        flexural_dispersion_layered(
            f,
            vp=4500.0,
            vs=2500.0,
            rho=2400.0,
            vf=1500.0,
            rho_f=1000.0,
            a=0.1,
            layers=(layer,),
        )


def test_flexural_dispersion_layered_accepts_list_for_layers():
    """``layers`` should accept any iterable that ``tuple(...)``
    consumes; empty list dispatches to the unlayered solver."""
    f = np.linspace(2000.0, 5000.0, 4)
    res_tuple = flexural_dispersion_layered(
        f,
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
        layers=(),
    )
    res_list = flexural_dispersion_layered(
        f,
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
        layers=[],
    )
    np.testing.assert_array_equal(res_tuple.slowness, res_list.slowness)


# =====================================================================
# Plan item F.2.c.1 -- row 3 of the n=1 layered determinant (r = a)
# =====================================================================
#
# First sin-sector row of the F.2 chain. Encodes
# ``sigma_rtheta^{(m)}(a) = 0``. The 10x10 layered determinant is
# dense (per the F.2.a.6 erratum) so this row builder returns
# shape-(10,) covering all amplitude columns. Primary correctness
# oracle: per-element layer=formation match against M31, M32, M33,
# M34 of :func:`_modal_determinant_n1` (the existing n=1
# single-interface form).


def test_layered_n1_row3_at_a_layer_equals_formation_per_element():
    """Substep F.2.a.7 (a) self-check at the row level: at
    layer=formation the K-flavour annulus columns of row 3 match
    M31, M32, M33, M34 of :func:`_modal_determinant_n1` to
    floating-point precision.

    Specifically:
        row[0] (A)   = M31 = 0     (fluid no shear)
        row[2] (B_K) = M32         (K-flavor B coefficient at r=a)
        row[4] (C_K) = M33         (K-flavor C coefficient at r=a)
        row[8] (D_K) = M34         (K-flavor D coefficient at r=a)
    """
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n1_row3_at_a(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )

    # Reconstruct the M31-M34 entries directly from the n=1
    # single-interface formula (see _modal_determinant_n1
    # docstring lines for row 3).
    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs

    M31 = 0.0
    M32 = (
        2.0
        * mu
        * (p * float(sp.kv(0, p * a)) / a + 2.0 * float(sp.kv(1, p * a)) / (a * a))
    )
    M33 = (
        2.0
        * kz
        * mu
        * (s * float(sp.kv(0, s * a)) / a + 2.0 * float(sp.kv(1, s * a)) / (a * a))
    )
    M34 = -mu * (
        s * s * float(sp.kv(1, s * a))
        + 2.0 * s * float(sp.kv(0, s * a)) / a
        + 4.0 * float(sp.kv(1, s * a)) / (a * a)
    )

    assert row[0].real == pytest.approx(M31)
    assert row[2].real == pytest.approx(M32)
    assert row[4].real == pytest.approx(M33)
    assert row[8].real == pytest.approx(M34)


def test_layered_n1_row3_at_a_sparsity():
    """Sparsity per the corrected F.2.a.4: A column is zero (fluid
    no shear); formation columns (5 = B, 6 = C, 9 = D) are zero
    because the formation half-space lives at ``r > b`` and doesn't
    reach r = a."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row3_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    assert row[0] == 0.0  # A
    assert row[5] == 0.0  # formation B
    assert row[6] == 0.0  # formation C
    assert row[9] == 0.0  # formation D
    # The remaining six columns (1, 2, 3, 4, 7, 8) are generically
    # non-zero in the bound regime.
    for i in (1, 2, 3, 4, 7, 8):
        assert row[i] != 0.0


def test_layered_n1_row3_at_a_is_real_in_bound_regime():
    """Substep F.2.a.5: row 3 has the no-row-rescale pattern;
    only the column-by-(-i) on C_I, C_K is applied. Post-rescale
    row is real-valued in the bound regime."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row3_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_n1_row3_at_a_matches_closed_form_per_column():
    """Per-column transcription check against substeps F.2.a.2 /
    F.2.a.3 closed forms (with the F.1.a.2 sign-flip pattern
    applied to the I-flavour annulus terms)."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    a = p["a"]
    from scipy import special as sp

    row = _layered_n1_row3_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=a,
        layer=p["layer"],
    )

    mu_m = p["layer"].rho * p["layer"].vs ** 2

    expected_BI = (
        2.0
        * mu_m
        * (
            -p_m * float(sp.iv(0, p_m * a)) / a
            + 2.0 * float(sp.iv(1, p_m * a)) / (a * a)
        )
    )
    expected_BK = (
        2.0
        * mu_m
        * (
            +p_m * float(sp.kv(0, p_m * a)) / a
            + 2.0 * float(sp.kv(1, p_m * a)) / (a * a)
        )
    )
    expected_CI = (
        2.0
        * kz
        * mu_m
        * (
            -s_m * float(sp.iv(0, s_m * a)) / a
            + 2.0 * float(sp.iv(1, s_m * a)) / (a * a)
        )
    )
    expected_CK = (
        2.0
        * kz
        * mu_m
        * (
            +s_m * float(sp.kv(0, s_m * a)) / a
            + 2.0 * float(sp.kv(1, s_m * a)) / (a * a)
        )
    )
    expected_DI = -mu_m * (
        s_m * s_m * float(sp.iv(1, s_m * a))
        - 2.0 * s_m * float(sp.iv(0, s_m * a)) / a
        + 4.0 * float(sp.iv(1, s_m * a)) / (a * a)
    )
    expected_DK = -mu_m * (
        s_m * s_m * float(sp.kv(1, s_m * a))
        + 2.0 * s_m * float(sp.kv(0, s_m * a)) / a
        + 4.0 * float(sp.kv(1, s_m * a)) / (a * a)
    )

    assert row[1].real == pytest.approx(expected_BI)
    assert row[2].real == pytest.approx(expected_BK)
    assert row[3].real == pytest.approx(expected_CI)
    assert row[4].real == pytest.approx(expected_CK)
    assert row[7].real == pytest.approx(expected_DI)
    assert row[8].real == pytest.approx(expected_DK)


def test_layered_n1_row3_at_a_C_column_i_k_sign_flip():
    """The C-amplitude entries carry the derivative-induced
    ``X_0/a`` term after the roadmap-A.8 correction, so the I-K
    ratio is not the bare ``+I_1/K_1``. Both flavours are the same
    functional of ``(sigma, X_0, X_1)`` -- sigma = +1 for I, -1 for
    K -- which is the F.1.a.2 pattern this test pins."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    from scipy import special as sp

    row = _layered_n1_row3_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )

    a = p["a"]
    mu_m = p["layer"].rho * p["layer"].vs ** 2

    def c_entry(sig, B0, B1):
        """``-2 k_z mu_m (d_r X_1 / a - X_1 / a^2)`` at r = a."""
        return -2.0 * kz * mu_m * ((sig * s_m * B0 - B1 / a) / a - B1 / (a * a))

    assert row[3].real == pytest.approx(
        c_entry(+1.0, float(sp.iv(0, s_m * a)), float(sp.iv(1, s_m * a)))
    )
    assert row[4].real == pytest.approx(
        c_entry(-1.0, float(sp.kv(0, s_m * a)), float(sp.kv(1, s_m * a)))
    )
    bare_ratio = +float(sp.iv(1, s_m * a)) / float(sp.kv(1, s_m * a))
    assert row[3].real / row[4].real != pytest.approx(bare_ratio, rel=1.0e-3)


# =====================================================================
# Plan item F.2.c.2 -- row 6 of the n=1 layered determinant (r = b)
# =====================================================================
#
# Genuinely new BC type at the layered case: u_theta continuity at
# r=b has no single-interface analog (the fluid-solid interface at
# r=a replaces it with sigma_rtheta = 0). C does NOT appear in
# u_theta per substep F.2.a.2; row 6 has six non-zero entries
# (B and D amplitudes only) and three explicit zero entries beyond
# the standard A=0 sparsity.


def test_layered_n1_row6_at_b_layer_equals_formation_K_flavour_cancels():
    """Substep F.2.a.7 (a) self-check: at layer=formation the
    K-flavour annulus and formation columns cancel pair-wise.

        row6[2] (B_K) + row6[5] (B) == 0
        row6[4] (C_K) + row6[6] (C) == 0
        row6[8] (D_K) + row6[9] (D) == 0

    The C pair is new with the roadmap-A.8 correction: u_theta does
    couple to the SV amplitude at n >= 1.
    """
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n1_row6_at_b(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )
    assert row[2].real + row[5].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[4].real + row[6].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[8].real + row[9].real == pytest.approx(0.0, abs=1.0e-14)


def test_layered_n1_row6_at_b_C_columns_are_nonzero():
    """Roadmap A.8: u_theta DOES have a C contribution at n >= 1.

    The Hansen SV field ``u = curl curl(chi z)`` carries
    ``u_theta = i k_z (n/r) chi``, which vanishes only at n = 0.
    Substep F.2.a.2 asserted the opposite -- that C never appears
    in u_theta -- which was an artefact of the azimuthal-only
    vector potential the SV columns used to encode. That ansatz has
    no u_theta at all, so the entries looked structurally absent
    rather than merely small.

    Columns 3 (C_I), 4 (C_K) and 6 (formation C) are therefore
    ``-/+ k_z X_1(s b) / b`` and non-zero; the row's only genuine
    zero is the A column (the fluid lives at r < a)."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row6_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    from scipy import special as sp

    _, _, s_m, _, s_form = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    b = p["a"] + p["layer"].thickness

    assert row[3].real == pytest.approx(-kz * float(sp.iv(1, s_m * b)) / b)
    assert row[4].real == pytest.approx(-kz * float(sp.kv(1, s_m * b)) / b)
    assert row[6].real == pytest.approx(+kz * float(sp.kv(1, s_form * b)) / b)
    assert row[0] == 0.0  # A (fluid r<a)
    # Every other column is generically non-zero in the bound regime.
    for i in (1, 2, 3, 4, 5, 6, 7, 8, 9):
        assert row[i] != 0.0


def test_layered_n1_row6_at_b_is_real_in_bound_regime():
    """Substep F.2.a.5: row 6 is NOT z-derivative-bearing (no
    row * i scaling). C columns are zero so column-by-(-i) is
    irrelevant. Pre- and post-rescale are both real-valued in the
    bound regime."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row6_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_n1_row6_at_b_matches_closed_form_per_column():
    """Per-column transcription check against substep F.2.a.2's
    u_theta closed forms. The B and D coefficients carry the
    F.1.a.2 sign-flip pattern: ``s I_0`` flips, ``K_1/r``-style
    direct terms keep sign."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, p_form, s_form = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n1_row6_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=a,
        layer=p["layer"],
    )

    expected_BI = -float(sp.iv(1, p_m * b)) / b
    expected_BK = -float(sp.kv(1, p_m * b)) / b
    expected_B = +float(sp.kv(1, p_form * b)) / b
    expected_DI = -s_m * float(sp.iv(0, s_m * b)) + float(sp.iv(1, s_m * b)) / b
    expected_DK = +s_m * float(sp.kv(0, s_m * b)) + float(sp.kv(1, s_m * b)) / b
    expected_D = -s_form * float(sp.kv(0, s_form * b)) - float(sp.kv(1, s_form * b)) / b

    assert row[1].real == pytest.approx(expected_BI)
    assert row[2].real == pytest.approx(expected_BK)
    assert row[5].real == pytest.approx(expected_B)
    assert row[7].real == pytest.approx(expected_DI)
    assert row[8].real == pytest.approx(expected_DK)
    assert row[9].real == pytest.approx(expected_D)


def test_layered_n1_row6_at_b_B_column_i_k_sign_flip():
    """The B-amplitude entries are single-Bessel-term:
    row6[B_I] = -I_1/b, row6[B_K] = -K_1/b. Their ratio is
    ``+I_1(p_m b) / K_1(p_m b)`` -- KEEP-sign per the F.1.a.2
    pattern (direct ``K_1/r`` term, no derivative-induced
    Bessel-index shift)."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, _, _, _ = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    from scipy import special as sp

    row = _layered_n1_row6_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    b = p["a"] + p["layer"].thickness

    expected_ratio = +float(sp.iv(1, p_m * b)) / float(sp.kv(1, p_m * b))
    assert row[1].real / row[2].real == pytest.approx(expected_ratio)


# =====================================================================
# Plan item F.2.c.3 -- row 9 of the n=1 layered determinant (r = b)
# =====================================================================
#
# Closes substep F.2.c. Same algebraic structure as row 3 at r=a
# but at the second interface with non-zero formation columns.
# The cross-row-3 identity at layer=formation is the structural
# safety net.


def test_layered_n1_row9_at_b_layer_equals_formation_K_flavour_cancels():
    """Substep F.2.a.7 (a) self-check: at layer=formation all THREE
    K-flavour annulus / formation column pairs cancel:

        row9[2] (B_K) + row9[5] (B) == 0
        row9[4] (C_K) + row9[6] (C) == 0
        row9[8] (D_K) + row9[9] (D) == 0
    """
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n1_row9_at_b(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )
    assert row[2].real + row[5].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[4].real + row[6].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[8].real + row[9].real == pytest.approx(0.0, abs=1.0e-14)


def test_layered_n1_row9_at_b_sparsity():
    """Sparsity: A column zero (fluid r<a, no shear); all other
    nine columns generically non-zero."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row9_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    assert row[0] == 0.0
    for i in range(1, 10):
        assert row[i] != 0.0


def test_layered_n1_row9_at_b_is_real_in_bound_regime():
    """Substep F.2.a.5: row 9 has the no-row-rescale pattern;
    only column-by-(-i) on C cols. Post-rescale row is real-valued
    in the bound regime."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row9_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_n1_row9_at_b_matches_closed_form_per_column():
    """Per-column transcription check against substeps F.2.a.2 /
    F.2.a.3 closed forms for all nine non-zero entries (annulus
    B/C/D + formation B/C/D)."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, p_form, s_form = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n1_row9_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=a,
        layer=p["layer"],
    )

    mu_m = p["layer"].rho * p["layer"].vs ** 2
    mu = p["rho"] * p["vs"] ** 2

    expected_BI = (
        2.0
        * mu_m
        * (
            -p_m * float(sp.iv(0, p_m * b)) / b
            + 2.0 * float(sp.iv(1, p_m * b)) / (b * b)
        )
    )
    expected_BK = (
        2.0
        * mu_m
        * (
            +p_m * float(sp.kv(0, p_m * b)) / b
            + 2.0 * float(sp.kv(1, p_m * b)) / (b * b)
        )
    )
    expected_CI = (
        2.0
        * kz
        * mu_m
        * (
            -s_m * float(sp.iv(0, s_m * b)) / b
            + 2.0 * float(sp.iv(1, s_m * b)) / (b * b)
        )
    )
    expected_CK = (
        2.0
        * kz
        * mu_m
        * (
            +s_m * float(sp.kv(0, s_m * b)) / b
            + 2.0 * float(sp.kv(1, s_m * b)) / (b * b)
        )
    )
    expected_B = (
        -2.0
        * mu
        * (
            +p_form * float(sp.kv(0, p_form * b)) / b
            + 2.0 * float(sp.kv(1, p_form * b)) / (b * b)
        )
    )
    expected_C = (
        -2.0
        * kz
        * mu
        * (
            s_form * float(sp.kv(0, s_form * b)) / b
            + 2.0 * float(sp.kv(1, s_form * b)) / (b * b)
        )
    )
    expected_DI = -mu_m * (
        s_m * s_m * float(sp.iv(1, s_m * b))
        - 2.0 * s_m * float(sp.iv(0, s_m * b)) / b
        + 4.0 * float(sp.iv(1, s_m * b)) / (b * b)
    )
    expected_DK = -mu_m * (
        s_m * s_m * float(sp.kv(1, s_m * b))
        + 2.0 * s_m * float(sp.kv(0, s_m * b)) / b
        + 4.0 * float(sp.kv(1, s_m * b)) / (b * b)
    )
    expected_D = +mu * (
        s_form * s_form * float(sp.kv(1, s_form * b))
        + 2.0 * s_form * float(sp.kv(0, s_form * b)) / b
        + 4.0 * float(sp.kv(1, s_form * b)) / (b * b)
    )

    assert row[1].real == pytest.approx(expected_BI)
    assert row[2].real == pytest.approx(expected_BK)
    assert row[3].real == pytest.approx(expected_CI)
    assert row[4].real == pytest.approx(expected_CK)
    assert row[5].real == pytest.approx(expected_B)
    assert row[6].real == pytest.approx(expected_C)
    assert row[7].real == pytest.approx(expected_DI)
    assert row[8].real == pytest.approx(expected_DK)
    assert row[9].real == pytest.approx(expected_D)


def test_layered_n1_row9_at_b_annulus_K_matches_row3_M32_M33_M34_at_b():
    """Cross-row identity: at layer=formation, row 9's annulus
    K-flavour entries match row 3's M32, M33, M34-equivalent forms
    evaluated at r = b (same underlying sigma_rtheta formula at
    both interfaces; only the evaluation radius differs)."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5
    b = a + layer.thickness

    row9 = _layered_n1_row9_at_b(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )

    # Row 3's M32, M33, M34 form evaluated at r=b instead of r=a.
    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs

    M32_at_b = (
        2.0
        * mu
        * (p * float(sp.kv(0, p * b)) / b + 2.0 * float(sp.kv(1, p * b)) / (b * b))
    )
    M33_at_b = (
        2.0
        * kz
        * mu
        * (s * float(sp.kv(0, s * b)) / b + 2.0 * float(sp.kv(1, s * b)) / (b * b))
    )
    M34_at_b = -mu * (
        s * s * float(sp.kv(1, s * b))
        + 2.0 * s * float(sp.kv(0, s * b)) / b
        + 4.0 * float(sp.kv(1, s * b)) / (b * b)
    )

    assert row9[2].real == pytest.approx(M32_at_b)
    assert row9[4].real == pytest.approx(M33_at_b)
    assert row9[8].real == pytest.approx(M34_at_b)


# =====================================================================
# Plan item F.2.b.1 -- row 1 of the n=1 layered determinant (r = a)
# =====================================================================
#
# First cos-sector row of the F.2 chain. The genuinely new content
# vs F.1.b.2.a (n=0 row 1) is the D-amplitude column: at n=1 the
# SH amplitude D appears in cos-sector u_r via the
# ``(1/r) d_theta psi_z`` cross-coupling. Primary oracle: per-
# element layer=formation match against M11-M14 of
# :func:`_modal_determinant_n1`.


def test_layered_n1_row1_at_a_layer_equals_formation_per_element():
    """At layer=formation, row 1's K-flavour annulus columns and
    the A column match M11, M12, M13, M14 of
    :func:`_modal_determinant_n1` to floating-point precision.

    Specifically:
        row[0] (A)   = M11   (fluid pressure coefficient)
        row[2] (B_K) = M12   (P-amplitude at r=a)
        row[4] (C_K) = M13   (SV-amplitude at r=a, post-rescale)
        row[8] (D_K) = M14   (SH-amplitude at r=a -- new at n=1)
    """
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n1_row1_at_a(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )

    F = float(np.sqrt(kz * kz - (omega / vf) ** 2))
    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    M11 = (F * float(sp.iv(0, F * a)) - float(sp.iv(1, F * a)) / a) / (rho_f * omega**2)
    M12 = p * float(sp.kv(0, p * a)) + float(sp.kv(1, p * a)) / a
    M13 = kz * (s * float(sp.kv(0, s * a)) + float(sp.kv(1, s * a)) / a)
    M14 = -float(sp.kv(1, s * a)) / a

    assert row[0].real == pytest.approx(M11)
    assert row[2].real == pytest.approx(M12)
    assert row[4].real == pytest.approx(M13)
    assert row[8].real == pytest.approx(M14)


def test_layered_n1_row1_at_a_formation_columns_are_zero():
    """Sparsity: at r=a the formation columns (5 = B, 6 = C, 9 = D)
    are zero -- the formation half-space lives at r > b and doesn't
    touch the fluid-annulus interface."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row1_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    assert row[5] == 0.0
    assert row[6] == 0.0
    assert row[9] == 0.0
    # All other columns generically non-zero in the bound regime.
    for i in (0, 1, 2, 3, 4, 7, 8):
        assert row[i] != 0.0


def test_layered_n1_row1_at_a_is_real_in_bound_regime():
    """Substep F.2.a.5: row 1 has the no-row-rescale pattern;
    only the column-by-(-i) on C_I, C_K is applied. Post-rescale
    row is real-valued in the bound regime."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row1_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_n1_row1_at_a_matches_closed_form_per_column():
    """Per-column transcription check against substep F.2.a.2's
    u_r decomposition (with the F.1.a.2 sign-flip pattern applied
    to the I-flavour annulus terms)."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    a = p["a"]
    from scipy import special as sp

    row = _layered_n1_row1_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=a,
        layer=p["layer"],
    )

    expected_A = (F_f * float(sp.iv(0, F_f * a)) - float(sp.iv(1, F_f * a)) / a) / (
        p["rho_f"] * omega**2
    )
    expected_BI = -p_m * float(sp.iv(0, p_m * a)) + float(sp.iv(1, p_m * a)) / a
    expected_BK = +p_m * float(sp.kv(0, p_m * a)) + float(sp.kv(1, p_m * a)) / a
    expected_CI = +kz * (float(sp.iv(1, s_m * a)) / a - s_m * float(sp.iv(0, s_m * a)))
    expected_CK = +kz * (s_m * float(sp.kv(0, s_m * a)) + float(sp.kv(1, s_m * a)) / a)
    expected_DI = -float(sp.iv(1, s_m * a)) / a
    expected_DK = -float(sp.kv(1, s_m * a)) / a

    assert row[0].real == pytest.approx(expected_A)
    assert row[1].real == pytest.approx(expected_BI)
    assert row[2].real == pytest.approx(expected_BK)
    assert row[3].real == pytest.approx(expected_CI)
    assert row[4].real == pytest.approx(expected_CK)
    assert row[7].real == pytest.approx(expected_DI)
    assert row[8].real == pytest.approx(expected_DK)


def test_layered_n1_row1_at_a_C_and_D_column_i_k_sign_flips():
    """The D entry is single-Bessel-term, so its I-K ratio
    collapses to a clean ``+I_1(s_m a) / K_1(s_m a)`` -- KEEP-sign
    per F.1.a.2.

    The C entry does NOT: after the roadmap-A.8 correction it is
    ``d_r`` of the SV scalar, so it carries a derivative-induced
    ``X_0`` term that flips sign between the I and K flavours while
    the direct ``X_1/a`` term keeps sign. Both flavours are the
    same functional of ``(sigma, X_0, X_1)`` with sigma = +1 for I
    and -1 for K, which is what this test pins."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    from scipy import special as sp

    row = _layered_n1_row1_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    a = p["a"]

    def c_entry(sig, B0, B1):
        """``-k_z d_r X_1(s_m r)`` at r = a, one functional for both
        flavours; ``d_r X_1 = sigma s_m X_0 - X_1/r``."""
        return -kz * (sig * s_m * B0 - B1 / a)

    assert row[3].real == pytest.approx(
        c_entry(+1.0, float(sp.iv(0, s_m * a)), float(sp.iv(1, s_m * a)))
    )
    assert row[4].real == pytest.approx(
        c_entry(-1.0, float(sp.kv(0, s_m * a)), float(sp.kv(1, s_m * a)))
    )
    # The derivative term really is present, so the C ratio is NOT
    # the bare I_1/K_1 that the single-Bessel D column gives.
    bare_ratio = +float(sp.iv(1, s_m * a)) / float(sp.kv(1, s_m * a))
    assert row[3].real / row[4].real != pytest.approx(bare_ratio, rel=1.0e-3)
    # D ratio: I_1(s_m a) / K_1(s_m a) (single Bessel term).
    assert row[7].real / row[8].real == pytest.approx(bare_ratio)


# =====================================================================
# Plan item F.2.b.2 -- row 2 of the n=1 layered determinant (r = a)
# =====================================================================
#
# Lame-reduction row at the first interface; algebraically heaviest
# of the cos-sector r=a rows. Multi-term entries on every B / C / D
# column, so the I-K sign-flip pattern is verified through the
# closed-form per-column transcription test rather than via a
# clean ratio (single-Bessel-term ratios don't apply).


def test_layered_n1_row2_at_a_layer_equals_formation_per_element():
    """At layer=formation row 2's K-flavour annulus columns and the
    A column match M21, M22, M23, M24 of
    :func:`_modal_determinant_n1` to floating-point precision."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n1_row2_at_a(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )

    F = float(np.sqrt(kz * kz - (omega / vf) ** 2))
    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    M21 = -float(sp.iv(1, F * a))
    M22 = -mu * (
        two_kz2_minus_kS2 * float(sp.kv(1, p * a))
        + 2.0 * p * float(sp.kv(0, p * a)) / a
        + 4.0 * float(sp.kv(1, p * a)) / (a * a)
    )
    M23 = (
        -2.0
        * kz
        * mu
        * (
            s * s * float(sp.kv(1, s * a))
            + s * float(sp.kv(0, s * a)) / a
            + 2.0 * float(sp.kv(1, s * a)) / (a * a)
        )
    )
    M24 = (
        +2.0
        * mu
        * (s * float(sp.kv(0, s * a)) / a + 2.0 * float(sp.kv(1, s * a)) / (a * a))
    )

    assert row[0].real == pytest.approx(M21)
    assert row[2].real == pytest.approx(M22)
    assert row[4].real == pytest.approx(M23)
    assert row[8].real == pytest.approx(M24)


def test_layered_n1_row2_at_a_formation_columns_are_zero():
    """Sparsity: at r=a the formation columns (5, 6, 9) are zero;
    remaining seven columns (0, 1, 2, 3, 4, 7, 8) generically non-
    zero."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row2_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    assert row[5] == 0.0
    assert row[6] == 0.0
    assert row[9] == 0.0
    for i in (0, 1, 2, 3, 4, 7, 8):
        assert row[i] != 0.0


def test_layered_n1_row2_at_a_is_real_in_bound_regime():
    """Substep F.2.a.5: row 2 has the no-row-rescale pattern;
    column-by-(-i) on C_I, C_K only. Post-rescale row is real."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row2_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_n1_row2_at_a_matches_closed_form_per_column():
    """Per-column transcription check against substep F.2.a.3's
    sigma_rr decomposition. The I-flavour entries (B_I, C_I, D_I)
    encode the F.1.a.2 sign-flip pattern: derivative-induced
    ``p_m X_0/a`` and ``s_m X_0/a`` terms flip sign vs the
    K-flavour twins; direct ``X_1`` and ``X_1/r^n`` terms keep
    sign."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    a = p["a"]
    from scipy import special as sp

    row = _layered_n1_row2_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=a,
        layer=p["layer"],
    )

    mu_m = p["layer"].rho * p["layer"].vs ** 2
    kSm2 = (omega / p["layer"].vs) ** 2
    two_kz2_minus_kSm2 = 2.0 * kz * kz - kSm2

    expected_A = -float(sp.iv(1, F_f * a))
    expected_BI = -mu_m * (
        two_kz2_minus_kSm2 * float(sp.iv(1, p_m * a))
        - 2.0 * p_m * float(sp.iv(0, p_m * a)) / a
        + 4.0 * float(sp.iv(1, p_m * a)) / (a * a)
    )
    expected_BK = -mu_m * (
        two_kz2_minus_kSm2 * float(sp.kv(1, p_m * a))
        + 2.0 * p_m * float(sp.kv(0, p_m * a)) / a
        + 4.0 * float(sp.kv(1, p_m * a)) / (a * a)
    )
    expected_CI = (
        -2.0
        * kz
        * mu_m
        * (
            s_m * s_m * float(sp.iv(1, s_m * a))
            - s_m * float(sp.iv(0, s_m * a)) / a
            + 2.0 * float(sp.iv(1, s_m * a)) / (a * a)
        )
    )
    expected_CK = (
        -2.0
        * kz
        * mu_m
        * (
            s_m * s_m * float(sp.kv(1, s_m * a))
            + s_m * float(sp.kv(0, s_m * a)) / a
            + 2.0 * float(sp.kv(1, s_m * a)) / (a * a)
        )
    )
    expected_DI = (
        +2.0
        * mu_m
        * (
            -s_m * float(sp.iv(0, s_m * a)) / a
            + 2.0 * float(sp.iv(1, s_m * a)) / (a * a)
        )
    )
    expected_DK = (
        +2.0
        * mu_m
        * (
            +s_m * float(sp.kv(0, s_m * a)) / a
            + 2.0 * float(sp.kv(1, s_m * a)) / (a * a)
        )
    )

    assert row[0].real == pytest.approx(expected_A)
    assert row[1].real == pytest.approx(expected_BI)
    assert row[2].real == pytest.approx(expected_BK)
    assert row[3].real == pytest.approx(expected_CI)
    assert row[4].real == pytest.approx(expected_CK)
    assert row[7].real == pytest.approx(expected_DI)
    assert row[8].real == pytest.approx(expected_DK)


# =====================================================================
# Plan item F.2.b.3 -- row 4 of the n=1 layered determinant (r = a)
# =====================================================================
#
# First z-derivative-bearing cos row of the F.2 chain. Per substep
# F.2.a.5: row * i AND col-by-(-i) on C cols. Both rescales must be
# correctly applied for the post-rescale row to be real.


def test_layered_n1_row4_at_a_layer_equals_formation_per_element():
    """At layer=formation row 4's K-flavour annulus columns and the
    A column match M41-M44 of :func:`_modal_determinant_n1`."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n1_row4_at_a(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )

    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    M41 = 0.0
    M42 = +2.0 * kz * mu * (p * float(sp.kv(0, p * a)) + float(sp.kv(1, p * a)) / a)
    M43 = (
        +mu
        * two_kz2_minus_kS2
        * (s * float(sp.kv(0, s * a)) + float(sp.kv(1, s * a)) / a)
    )
    M44 = -kz * mu * float(sp.kv(1, s * a)) / a

    assert row[0].real == pytest.approx(M41)
    assert row[2].real == pytest.approx(M42)
    assert row[4].real == pytest.approx(M43)
    assert row[8].real == pytest.approx(M44)


def test_layered_n1_row4_at_a_sparsity():
    """Sparsity: A column zero (fluid no shear); formation columns
    (5, 6, 9) zero (don't reach r=a); remaining six columns
    generically non-zero."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row4_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    assert row[0] == 0.0
    assert row[5] == 0.0
    assert row[6] == 0.0
    assert row[9] == 0.0
    for i in (1, 2, 3, 4, 7, 8):
        assert row[i] != 0.0


def test_layered_n1_row4_at_a_is_real_in_bound_regime():
    """Substep F.2.a.5: row 4 has the FULL z-bearing rescale (row
    * i AND col-by-(-i) on C cols). Both must be correctly applied
    for the post-rescale row to be real. Forgetting the row * i is
    the most direct transcription error per F.2.a.5 commentary;
    this test catches it."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row4_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_n1_row4_at_a_matches_closed_form_per_column():
    """Per-column transcription check against substep F.2.a.3's
    sigma_rz decomposition. The B columns have multi-term
    (``+p_m X_0 +/- X_1/r``); C and D columns are single-Bessel-term."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    a = p["a"]
    from scipy import special as sp

    row = _layered_n1_row4_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=a,
        layer=p["layer"],
    )

    mu_m = p["layer"].rho * p["layer"].vs ** 2
    kSm2 = (omega / p["layer"].vs) ** 2
    two_kz2_minus_kSm2 = 2.0 * kz * kz - kSm2

    expected_BI = (
        -2.0
        * kz
        * mu_m
        * (p_m * float(sp.iv(0, p_m * a)) - float(sp.iv(1, p_m * a)) / a)
    )
    expected_BK = (
        +2.0
        * kz
        * mu_m
        * (p_m * float(sp.kv(0, p_m * a)) + float(sp.kv(1, p_m * a)) / a)
    )
    expected_CI = (
        +mu_m
        * two_kz2_minus_kSm2
        * (float(sp.iv(1, s_m * a)) / a - s_m * float(sp.iv(0, s_m * a)))
    )
    expected_CK = (
        +mu_m
        * two_kz2_minus_kSm2
        * (s_m * float(sp.kv(0, s_m * a)) + float(sp.kv(1, s_m * a)) / a)
    )
    expected_DI = -kz * mu_m * float(sp.iv(1, s_m * a)) / a
    expected_DK = -kz * mu_m * float(sp.kv(1, s_m * a)) / a

    assert row[1].real == pytest.approx(expected_BI)
    assert row[2].real == pytest.approx(expected_BK)
    assert row[3].real == pytest.approx(expected_CI)
    assert row[4].real == pytest.approx(expected_CK)
    assert row[7].real == pytest.approx(expected_DI)
    assert row[8].real == pytest.approx(expected_DK)


def test_layered_n1_row4_at_a_C_and_D_column_i_k_sign_flips():
    """The D entry is a single-Bessel-term direct ``X_1`` form, so
    its I-K ratio collapses to a clean ``+I_1(s_m a) / K_1(s_m a)``
    -- KEEP-sign per F.1.a.2.

    The C entry does not: after the roadmap-A.8 correction it
    carries ``d_r X_1(s_m r)``, whose ``X_0`` term flips sign
    between the flavours. Both are the same functional of
    ``(sigma, X_0, X_1)`` with sigma = +1 for I and -1 for K."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    from scipy import special as sp

    row = _layered_n1_row4_at_a(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )

    a = p["a"]
    two_kz2_minus_kSm2 = 2.0 * kz * kz - (omega / p["layer"].vs) ** 2
    mu_m = p["layer"].rho * p["layer"].vs ** 2

    def c_entry(sig, B0, B1):
        """``-mu_m (2 k_z^2 - k_Sm^2) d_r X_1(s_m r)`` at r = a."""
        return -mu_m * two_kz2_minus_kSm2 * (sig * s_m * B0 - B1 / a)

    assert row[3].real == pytest.approx(
        c_entry(+1.0, float(sp.iv(0, s_m * a)), float(sp.iv(1, s_m * a)))
    )
    assert row[4].real == pytest.approx(
        c_entry(-1.0, float(sp.kv(0, s_m * a)), float(sp.kv(1, s_m * a)))
    )
    expected_ratio = +float(sp.iv(1, s_m * p["a"])) / float(sp.kv(1, s_m * p["a"]))
    # C carries a derivative term after the A.8 correction, so its
    # I-K ratio is no longer the bare single-Bessel one; D still is.
    assert row[3].real / row[4].real != pytest.approx(expected_ratio, rel=1.0e-3)
    assert row[7].real / row[8].real == pytest.approx(expected_ratio)


# =====================================================================
# Plan item F.2.b.4 -- row 5 of the n=1 layered determinant (r = b)
# =====================================================================
#
# Mirror of row 1 evaluated at r=b with non-zero formation columns.
# No single-interface analog; primary oracle is K-flavour
# cancellation at layer=formation.


def test_layered_n1_row5_at_b_layer_equals_formation_K_flavour_cancels():
    """Substep F.2.a.7 (a) self-check: at layer=formation all THREE
    K-flavour annulus / formation column pairs cancel:

        row5[2] (B_K) + row5[5] (B) == 0
        row5[4] (C_K) + row5[6] (C) == 0
        row5[8] (D_K) + row5[9] (D) == 0
    """
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n1_row5_at_b(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )
    assert row[2].real + row[5].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[4].real + row[6].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[8].real + row[9].real == pytest.approx(0.0, abs=1.0e-14)


def test_layered_n1_row5_at_b_fluid_column_is_zero():
    """Sparsity: A column zero (fluid r<a doesn't reach r=b);
    remaining nine columns generically non-zero."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row5_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    assert row[0] == 0.0
    for i in range(1, 10):
        assert row[i] != 0.0


def test_layered_n1_row5_at_b_is_real_in_bound_regime():
    """Substep F.2.a.5: row 5 has the no-row-rescale pattern;
    column-by-(-i) on C_I, C_K, C only. Post-rescale row is real."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row5_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_n1_row5_at_b_matches_closed_form_per_column():
    """Per-column transcription check against substep F.2.a.2's
    u_r decomposition at r=b."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, p_form, s_form = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n1_row5_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=a,
        layer=p["layer"],
    )

    expected_BI = +p_m * float(sp.iv(0, p_m * b)) - float(sp.iv(1, p_m * b)) / b
    expected_BK = -p_m * float(sp.kv(0, p_m * b)) - float(sp.kv(1, p_m * b)) / b
    expected_CI = +kz * (s_m * float(sp.iv(0, s_m * b)) - float(sp.iv(1, s_m * b)) / b)
    expected_CK = -kz * (s_m * float(sp.kv(0, s_m * b)) + float(sp.kv(1, s_m * b)) / b)
    expected_B = +p_form * float(sp.kv(0, p_form * b)) + float(sp.kv(1, p_form * b)) / b
    expected_C = +kz * (
        s_form * float(sp.kv(0, s_form * b)) + float(sp.kv(1, s_form * b)) / b
    )
    expected_DI = +float(sp.iv(1, s_m * b)) / b
    expected_DK = +float(sp.kv(1, s_m * b)) / b
    expected_D = -float(sp.kv(1, s_form * b)) / b

    assert row[1].real == pytest.approx(expected_BI)
    assert row[2].real == pytest.approx(expected_BK)
    assert row[3].real == pytest.approx(expected_CI)
    assert row[4].real == pytest.approx(expected_CK)
    assert row[5].real == pytest.approx(expected_B)
    assert row[6].real == pytest.approx(expected_C)
    assert row[7].real == pytest.approx(expected_DI)
    assert row[8].real == pytest.approx(expected_DK)
    assert row[9].real == pytest.approx(expected_D)


def test_layered_n1_row5_at_b_annulus_K_sign_opposite_to_row1_at_a():
    """Sign-flow consistency vs row 1: the BC subtraction direction
    flips between row 1 (``u_r^{(f)} - u_r^{(m)} = 0``, annulus
    appears with - sign) and row 5 (``u_r^{(m)} - u_r^{(s)} = 0``,
    annulus appears with + sign). Consequently row 5's annulus
    K-flavour B_K is the negation of row 1's B_K (modulo the
    radius shift a -> b)."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, _, _, _ = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    from scipy import special as sp

    row5 = _layered_n1_row5_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    b = p["a"] + p["layer"].thickness
    # Row 5 B_K should be -p_m K_0(p_m b) - K_1(p_m b)/b = -(row 1 B_K form at r=b).
    assert row5[2].real == pytest.approx(
        -(p_m * float(sp.kv(0, p_m * b)) + float(sp.kv(1, p_m * b)) / b)
    )


# =====================================================================
# Plan item F.2.b.5 -- row 7 of the n=1 layered determinant (r = b)
# =====================================================================
#
# Z-derivative-bearing cos row at the second interface. Distinctive
# sparsity: D columns (7, 8, 9) are identically zero because u_z
# does not couple to psi_z under the curl decomposition (curl_z =
# (1/r) d_r(r psi_theta), no psi_z term).


def test_layered_n1_row7_at_b_K_flavour_cancels_at_layer_equals_formation():
    """Substep F.2.a.7 (a) self-check: K-flavour cancellation pairs
    at layer=formation. The D pair is trivial since both D_K and D
    are zero (u_z has no psi_z contribution)."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n1_row7_at_b(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )
    assert row[2].real + row[5].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[4].real + row[6].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[8].real + row[9].real == 0.0  # both 0 -> exact


def test_layered_n1_row7_at_b_D_columns_are_identically_zero():
    """Distinctive sparsity of row 7 in F.2.b: u_z does not couple
    to psi_z under the curl decomposition (curl_z = (1/r)
    d_r(r psi_theta), no psi_z term). D columns (7, 8, 9) are
    identically zero -- the structural feature that distinguishes
    row 7 from rows 5, 8, 10."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row7_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    assert row[7] == 0.0  # D_I
    assert row[8] == 0.0  # D_K
    assert row[9] == 0.0  # formation D
    assert row[0] == 0.0  # A (fluid r<a)
    # Six remaining columns (1, 2, 3, 4, 5, 6) generically non-zero.
    for i in (1, 2, 3, 4, 5, 6):
        assert row[i] != 0.0


def test_layered_n1_row7_at_b_is_real_in_bound_regime():
    """Substep F.2.a.5: row 7 is z-derivative-bearing -- gets the
    FULL rescale (row * i + col-by-(-i) on C cols). Both must be
    correctly applied."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row7_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_n1_row7_at_b_matches_closed_form_per_column():
    """Per-column transcription check against substep F.2.a.2's
    u_z decomposition at r=b for the six non-zero entries."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, p_form, s_form = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n1_row7_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=a,
        layer=p["layer"],
    )

    expected_BI = -kz * float(sp.iv(1, p_m * b))
    expected_BK = -kz * float(sp.kv(1, p_m * b))
    expected_CI = -s_m * s_m * float(sp.iv(1, s_m * b))
    expected_CK = -s_m * s_m * float(sp.kv(1, s_m * b))
    expected_B = +kz * float(sp.kv(1, p_form * b))
    expected_C = +s_form * s_form * float(sp.kv(1, s_form * b))

    assert row[1].real == pytest.approx(expected_BI)
    assert row[2].real == pytest.approx(expected_BK)
    assert row[3].real == pytest.approx(expected_CI)
    assert row[4].real == pytest.approx(expected_CK)
    assert row[5].real == pytest.approx(expected_B)
    assert row[6].real == pytest.approx(expected_C)


def test_layered_n1_row7_at_b_C_column_I_K_ratio_has_sign_flip():
    """Both the B and C columns of row 7 (axial-displacement
    continuity at r = b) are single-Bessel-term direct ``X_1``
    forms, so both I-K ratios are ``+I_1/K_1`` -- KEEP-sign per
    F.1.a.2, no derivative-induced ``X_0`` term in either.

    For C this is a consequence of the roadmap-A.8 correction: the
    Hansen SV field has ``u_z = -s^2 chi``, proportional to the
    scalar itself rather than to its radial derivative. The old
    azimuthal-only ansatz put ``+/- s X_0`` here instead, which is
    why this ratio used to be sign-flipped."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    from scipy import special as sp

    row = _layered_n1_row7_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    b = p["a"] + p["layer"].thickness

    # B ratio: +I_1/K_1 (KEEP sign).
    expected_B_ratio = +float(sp.iv(1, p_m * b)) / float(sp.kv(1, p_m * b))
    assert row[1].real / row[2].real == pytest.approx(expected_B_ratio)
    # C ratio: +I_1/K_1 (KEEP sign; u_z carries no d_r term).
    expected_C_ratio = +float(sp.iv(1, s_m * b)) / float(sp.kv(1, s_m * b))
    assert row[3].real / row[4].real == pytest.approx(expected_C_ratio)


# =====================================================================
# Plan item F.2.b.6 -- row 8 of the n=1 layered determinant (r = b)
# =====================================================================
#
# Lame-reduction row at the second interface; uses the unnegated
# continuity convention. Algebraically heaviest of the r=b cos
# rows.


def test_layered_n1_row8_at_b_K_flavour_cancels_at_layer_equals_formation():
    """Substep F.2.a.7 (a) self-check: all THREE K-flavour pairs
    cancel at layer=formation.

        row8[2] (B_K) + row8[5] (B) == 0
        row8[4] (C_K) + row8[6] (C) == 0
        row8[8] (D_K) + row8[9] (D) == 0
    """
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n1_row8_at_b(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )
    assert row[2].real + row[5].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[4].real + row[6].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[8].real + row[9].real == pytest.approx(0.0, abs=1.0e-14)


def test_layered_n1_row8_at_b_fluid_column_is_zero():
    """Sparsity: A column zero (fluid r<a doesn't reach r=b);
    remaining nine columns generically non-zero."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row8_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    assert row[0] == 0.0
    for i in range(1, 10):
        assert row[i] != 0.0


def test_layered_n1_row8_at_b_is_real_in_bound_regime():
    """Substep F.2.a.5: row 8 has the no-row-rescale pattern;
    only column-by-(-i) on C cols. Post-rescale row is real."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row8_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_n1_row8_at_b_matches_closed_form_per_column():
    """Per-column transcription check against substep F.2.a.3's
    sigma_rr decomposition at r=b for all nine non-zero entries."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, p_form, s_form = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n1_row8_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=a,
        layer=p["layer"],
    )

    mu_m = p["layer"].rho * p["layer"].vs ** 2
    kSm2 = (omega / p["layer"].vs) ** 2
    two_kz2_minus_kSm2 = 2.0 * kz * kz - kSm2
    mu = p["rho"] * p["vs"] ** 2
    kS2 = (omega / p["vs"]) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    expected_BI = +mu_m * (
        two_kz2_minus_kSm2 * float(sp.iv(1, p_m * b))
        - 2.0 * p_m * float(sp.iv(0, p_m * b)) / b
        + 4.0 * float(sp.iv(1, p_m * b)) / (b * b)
    )
    expected_BK = +mu_m * (
        two_kz2_minus_kSm2 * float(sp.kv(1, p_m * b))
        + 2.0 * p_m * float(sp.kv(0, p_m * b)) / b
        + 4.0 * float(sp.kv(1, p_m * b)) / (b * b)
    )
    expected_CI = (
        +2.0
        * kz
        * mu_m
        * (
            s_m * s_m * float(sp.iv(1, s_m * b))
            - s_m * float(sp.iv(0, s_m * b)) / b
            + 2.0 * float(sp.iv(1, s_m * b)) / (b * b)
        )
    )
    expected_CK = (
        +2.0
        * kz
        * mu_m
        * (
            s_m * s_m * float(sp.kv(1, s_m * b))
            + s_m * float(sp.kv(0, s_m * b)) / b
            + 2.0 * float(sp.kv(1, s_m * b)) / (b * b)
        )
    )
    expected_B = -mu * (
        two_kz2_minus_kS2 * float(sp.kv(1, p_form * b))
        + 2.0 * p_form * float(sp.kv(0, p_form * b)) / b
        + 4.0 * float(sp.kv(1, p_form * b)) / (b * b)
    )
    expected_C = (
        -2.0
        * kz
        * mu
        * (
            s_form * s_form * float(sp.kv(1, s_form * b))
            + s_form * float(sp.kv(0, s_form * b)) / b
            + 2.0 * float(sp.kv(1, s_form * b)) / (b * b)
        )
    )
    expected_DI = (
        +2.0
        * mu_m
        * (
            s_m * float(sp.iv(0, s_m * b)) / b
            - 2.0 * float(sp.iv(1, s_m * b)) / (b * b)
        )
    )
    expected_DK = (
        -2.0
        * mu_m
        * (
            s_m * float(sp.kv(0, s_m * b)) / b
            + 2.0 * float(sp.kv(1, s_m * b)) / (b * b)
        )
    )
    expected_D = (
        +2.0
        * mu
        * (
            s_form * float(sp.kv(0, s_form * b)) / b
            + 2.0 * float(sp.kv(1, s_form * b)) / (b * b)
        )
    )

    assert row[1].real == pytest.approx(expected_BI)
    assert row[2].real == pytest.approx(expected_BK)
    assert row[3].real == pytest.approx(expected_CI)
    assert row[4].real == pytest.approx(expected_CK)
    assert row[5].real == pytest.approx(expected_B)
    assert row[6].real == pytest.approx(expected_C)
    assert row[7].real == pytest.approx(expected_DI)
    assert row[8].real == pytest.approx(expected_DK)
    assert row[9].real == pytest.approx(expected_D)


def test_layered_n1_row8_at_b_annulus_K_matches_negated_row2_M22_M23_M24_at_b():
    """Convention cross-check: at layer=formation, row 8's annulus
    K-flavour entries (B_K, C_K, D_K) equal the NEGATION of
    row 2's M22, M23, M24-equivalent forms evaluated at r=b
    (row 2 uses negated ``-(sigma_rr + P)`` convention; row 8 uses
    unnegated continuity). Pins down the convention difference."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5
    b = a + layer.thickness

    row8 = _layered_n1_row8_at_b(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )

    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    # Row 2's M22, M23, M24 form evaluated at r=b instead of r=a.
    M22_at_b = -mu * (
        two_kz2_minus_kS2 * float(sp.kv(1, p * b))
        + 2.0 * p * float(sp.kv(0, p * b)) / b
        + 4.0 * float(sp.kv(1, p * b)) / (b * b)
    )
    M23_at_b = (
        -2.0
        * kz
        * mu
        * (
            s * s * float(sp.kv(1, s * b))
            + s * float(sp.kv(0, s * b)) / b
            + 2.0 * float(sp.kv(1, s * b)) / (b * b)
        )
    )
    M24_at_b = (
        +2.0
        * mu
        * (s * float(sp.kv(0, s * b)) / b + 2.0 * float(sp.kv(1, s * b)) / (b * b))
    )

    # Row 8 unnegated; row 2 negated. row8 = -M2j_at_b.
    assert row8[2].real == pytest.approx(-M22_at_b)
    assert row8[4].real == pytest.approx(-M23_at_b)
    assert row8[8].real == pytest.approx(-M24_at_b)


# =====================================================================
# Plan item F.2.b.7 -- row 10 of the n=1 layered determinant (r = b)
# =====================================================================
#
# Final row of the 10x10 layered determinant; closes substep F.2.b.
# Z-derivative-bearing cos row at the second interface; analogous
# to row 4 at r=a with non-zero formation cols.


def test_layered_n1_row10_at_b_K_flavour_cancels_at_layer_equals_formation():
    """All three K-flavour pairs cancel at layer=formation."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _layered_n1_row10_at_b(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )
    assert row[2].real + row[5].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[4].real + row[6].real == pytest.approx(0.0, abs=1.0e-14)
    assert row[8].real + row[9].real == pytest.approx(0.0, abs=1.0e-14)


def test_layered_n1_row10_at_b_fluid_column_is_zero():
    """Sparsity: A column zero (fluid no shear AND fluid r<a);
    remaining nine columns generically non-zero."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row10_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    assert row[0] == 0.0
    for i in range(1, 10):
        assert row[i] != 0.0


def test_layered_n1_row10_at_b_is_real_in_bound_regime():
    """Substep F.2.a.5: row 10 is z-derivative-bearing -- gets the
    FULL rescale (row * i + col-by-(-i) on C cols)."""
    p, omega, kz = _row1_test_setup()
    row = _layered_n1_row10_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layer=p["layer"],
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_layered_n1_row10_at_b_matches_closed_form_per_column():
    """Per-column transcription check against substep F.2.a.3's
    sigma_rz decomposition at r=b for all nine non-zero entries."""
    p, omega, kz = _row1_test_setup()
    F_f, p_m, s_m, p_form, s_form = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        vf=p["vf"],
        layer=p["layer"],
    )
    a = p["a"]
    b = a + p["layer"].thickness
    from scipy import special as sp

    row = _layered_n1_row10_at_b(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=a,
        layer=p["layer"],
    )

    mu_m = p["layer"].rho * p["layer"].vs ** 2
    kSm2 = (omega / p["layer"].vs) ** 2
    two_kz2_minus_kSm2 = 2.0 * kz * kz - kSm2
    mu = p["rho"] * p["vs"] ** 2
    kS2 = (omega / p["vs"]) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    expected_BI = (
        -2.0
        * kz
        * mu_m
        * (p_m * float(sp.iv(0, p_m * b)) - float(sp.iv(1, p_m * b)) / b)
    )
    expected_BK = (
        +2.0
        * kz
        * mu_m
        * (p_m * float(sp.kv(0, p_m * b)) + float(sp.kv(1, p_m * b)) / b)
    )
    expected_CI = (
        +mu_m
        * two_kz2_minus_kSm2
        * (float(sp.iv(1, s_m * b)) / b - s_m * float(sp.iv(0, s_m * b)))
    )
    expected_CK = (
        +mu_m
        * two_kz2_minus_kSm2
        * (s_m * float(sp.kv(0, s_m * b)) + float(sp.kv(1, s_m * b)) / b)
    )
    expected_B = (
        -2.0
        * kz
        * mu
        * (p_form * float(sp.kv(0, p_form * b)) + float(sp.kv(1, p_form * b)) / b)
    )
    expected_C = (
        -mu
        * two_kz2_minus_kS2
        * (s_form * float(sp.kv(0, s_form * b)) + float(sp.kv(1, s_form * b)) / b)
    )
    expected_DI = -kz * mu_m * float(sp.iv(1, s_m * b)) / b
    expected_DK = -kz * mu_m * float(sp.kv(1, s_m * b)) / b
    expected_D = +kz * mu * float(sp.kv(1, s_form * b)) / b

    assert row[1].real == pytest.approx(expected_BI)
    assert row[2].real == pytest.approx(expected_BK)
    assert row[3].real == pytest.approx(expected_CI)
    assert row[4].real == pytest.approx(expected_CK)
    assert row[5].real == pytest.approx(expected_B)
    assert row[6].real == pytest.approx(expected_C)
    assert row[7].real == pytest.approx(expected_DI)
    assert row[8].real == pytest.approx(expected_DK)
    assert row[9].real == pytest.approx(expected_D)


def test_layered_n1_row10_at_b_annulus_K_matches_row4_M42_M43_M44_at_b():
    """Cross-row identity: at layer=formation, row 10's annulus
    K-flavour entries (B_K, C_K, D_K) match row 4's M42, M43, M44-
    equivalent forms evaluated at r=b instead of r=a (same
    underlying sigma_rz formula at both interfaces)."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5
    b = a + layer.thickness

    row10 = _layered_n1_row10_at_b(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layer=layer,
    )

    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    M42_at_b = (
        +2.0 * kz * mu * (p * float(sp.kv(0, p * b)) + float(sp.kv(1, p * b)) / b)
    )
    M43_at_b = (
        +mu
        * two_kz2_minus_kS2
        * (s * float(sp.kv(0, s * b)) + float(sp.kv(1, s * b)) / b)
    )
    M44_at_b = -kz * mu * float(sp.kv(1, s * b)) / b

    assert row10[2].real == pytest.approx(M42_at_b)
    assert row10[4].real == pytest.approx(M43_at_b)
    assert row10[8].real == pytest.approx(M44_at_b)


# =====================================================================
# Plan item F.2.d -- assembly + dispatch
# =====================================================================
#
# Closes the F.2.b/c chain. Tests fall into two groups:
#
#   * ``_modal_determinant_n1_layered``: real-valued in bound regime;
#     evaluates without raising; behaves correctly at the layer=
#     formation degenerate point.
#   * ``flexural_dispersion_layered`` end-to-end: layer=formation
#     reproduces ``flexural_dispersion`` slowness curve to
#     ``rtol=1e-8``; thickness->0 limit ditto; dispatched correctly.


def _layered_n1_slow_formation_params():
    """Slow-formation fixture (vs < vf) for end-to-end layered
    flexural tests. The layer must satisfy ``layer.vs >= vs``
    (a ``harder'' layer) for the wave to stay in the bound regime
    in the annulus: flexural slowness in slow formations is very
    close to ``1/vs``, and a softer layer (``layer.vs < vs``)
    would put ``s_m^2 < 0`` in the annulus -- the leaky regime
    handled by future fast-formation-layered work."""
    return dict(
        vp=3000.0,
        vs=1200.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
        layer=BoreholeLayer(vp=3200.0, vs=1300.0, rho=2350.0, thickness=0.005),
    )


def test_modal_determinant_n1_layered_is_real_in_bound_regime():
    """Substep F.2.a.5 phase rescale: each row builder applies the
    rescale internally, so the assembled 10x10 is real-valued in
    the bound regime."""
    p = _layered_n1_slow_formation_params()
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(p["vs"], p["layer"].vs, p["vf"]) * 1.05
    det = _modal_determinant_n1_layered(
        kz,
        omega,
        p["vp"],
        p["vs"],
        p["rho"],
        p["vf"],
        p["rho_f"],
        p["a"],
        layer=p["layer"],
    )
    assert np.isfinite(det)
    assert isinstance(det, float)


def test_modal_determinant_n1_layered_layer_equals_formation_root_matches_unlayered():
    """The substep-F.2.a.7 (a) self-check at the determinant level:
    at layer=formation, the layered determinant has the same
    flexural root as :func:`_modal_determinant_n1`. The two
    determinants are not numerically equal (the 10x10 has a
    different overall scale than the 4x4), but they share the
    same root in ``k_z``."""
    vp, vs, rho = 3000.0, 1200.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0

    bound = flexural_dispersion(
        np.array([5000.0]),
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    kz_root = float(bound.slowness[0]) * omega

    det_at_root = _modal_determinant_n1_layered(
        kz_root,
        omega,
        vp,
        vs,
        rho,
        vf,
        rho_f,
        a,
        layer=layer,
    )
    det_off_root = _modal_determinant_n1_layered(
        kz_root * 1.05,
        omega,
        vp,
        vs,
        rho,
        vf,
        rho_f,
        a,
        layer=layer,
    )
    # Determinant at root much smaller than off-root.
    assert abs(det_at_root) < abs(det_off_root) * 1.0e-3


def test_flexural_dispersion_layered_layer_equals_formation_matches_unlayered():
    """End-to-end integration test: with a layer whose properties
    match the formation, the layered solver produces the same
    flexural dispersion curve as the unlayered solver to
    ``rtol=1e-8``. Floating-point oracle for the entire F.2 chain.
    Any algebra error accumulated across the ten row builders
    surfaces here."""
    vp, vs, rho = 3000.0, 1200.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    f = np.linspace(2000.0, 8000.0, 12)

    res_unlayered = flexural_dispersion(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    res_layered = flexural_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer,),
    )
    np.testing.assert_allclose(
        res_layered.slowness,
        res_unlayered.slowness,
        rtol=1.0e-8,
        equal_nan=True,
    )


def test_flexural_dispersion_layered_thickness_zero_limit():
    """As ``layer.thickness -> 0`` (with arbitrary layer material),
    the layered solver continuously approaches the unlayered
    answer. Algebraic identity: in the limit ``b -> a``, the rows
    at r=b approach the rows at r=a, the second interface
    degenerates, and the converged k_z must approach the single-
    interface root."""
    vp, vs, rho = 3000.0, 1200.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = 5000.0

    res_unlayered = flexural_dispersion(
        np.array([f]),
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )

    # Even a "different" layer with vanishing thickness should
    # converge to the unlayered flexural slowness. Use a harder
    # layer (layer.vs > vs) so the bound regime holds in the
    # annulus per the F.2.d slow-formation gate.
    layer_thin = BoreholeLayer(
        vp=3200.0,
        vs=1300.0,
        rho=2350.0,
        thickness=1.0e-9,
    )
    res_thin = flexural_dispersion_layered(
        np.array([f]),
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer_thin,),
    )
    assert res_thin.slowness[0] == pytest.approx(
        res_unlayered.slowness[0],
        rel=1.0e-4,
    )


def test_flexural_dispersion_layered_non_trivial_layer_runs():
    """Smoke test: a soft mudcake layer different from the
    formation produces a finite slowness curve in the slow-
    formation bound regime. No analytic oracle here (Schmitt 1988
    fig 6 is the F.2.e validation target); the test confirms that
    the dispatch + 10x10 + brentq + bracket all wire up without
    raising."""
    p = _layered_n1_slow_formation_params()
    f = np.linspace(2000.0, 8000.0, 8)

    res = flexural_dispersion_layered(
        f,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(p["layer"],),
    )
    assert res.name == "flexural"
    assert res.azimuthal_order == 1
    assert res.slowness.shape == f.shape
    # Most slownesses finite in the bound regime; cutoff effects may
    # leave a few low-frequency NaNs, so check that at least the
    # high-f half is fully populated.
    n_finite = int(np.sum(np.isfinite(res.slowness)))
    assert n_finite >= len(f) // 2


# =====================================================================
# Plan item F.2.e -- validation hardening on top of F.2.d
# =====================================================================
#
# Hardening tests for the assembled layered flexural solver. Each
# tests an asymptotic / self-consistency property that the
# layer=formation regression alone doesn't pin down.
#
# Note: the F.1.d "thickness -> infty" test does NOT translate
# cleanly to F.2 because the layer's natural flexural mode has
# phase velocity in (V_R_layer, V_S_layer), and the F.2.d
# "harder layer" requirement (V_S_layer >= V_S_formation) means
# this band lies AT OR ABOVE V_S_formation -- outside the bound
# regime captured by the formation half-space. The layered
# flexural slowness in the thickness -> infty limit thus exits
# the bound regime; a faithful test would need fast-formation
# layered handling (future work).


def test_modal_determinant_n1_layered_vanishes_at_converged_root():
    """Self-consistency: at the converged ``k_z`` returned by
    :func:`flexural_dispersion_layered` (any non-trivial layer), the
    layered determinant is several orders of magnitude smaller than
    its value off-root. Sharper than the layer=formation det-at-root
    check from F.2.d; works for any harder layer in the slow-
    formation bound regime."""
    p = _layered_n1_slow_formation_params()
    f = 5000.0
    omega = 2.0 * np.pi * f

    res = flexural_dispersion_layered(
        np.array([f]),
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(p["layer"],),
    )
    kz_root = float(res.slowness[0]) * omega

    det_at = _modal_determinant_n1_layered(
        kz_root,
        omega,
        p["vp"],
        p["vs"],
        p["rho"],
        p["vf"],
        p["rho_f"],
        p["a"],
        layer=p["layer"],
    )
    det_off = _modal_determinant_n1_layered(
        kz_root * 1.01,
        omega,
        p["vp"],
        p["vs"],
        p["rho"],
        p["vf"],
        p["rho_f"],
        p["a"],
        layer=p["layer"],
    )
    # brentq-converged root: |det_at| >= 6 orders of magnitude
    # smaller than |det_off| at 1% off the root.
    assert abs(det_at) < abs(det_off) * 1.0e-6


def test_flexural_dispersion_layered_multiple_frequencies_bound_regime():
    """Smoke test across the slow-formation bound band. The flexural
    slowness in a slow formation INCREASES with frequency: low-f
    cutoff is at slowness ~1/V_S (formation), high-f asymptote is
    at slowness ~1/V_R > 1/V_S (Rayleigh / Scholte limit). Confirm
    monotonicity holds across a wide band with a non-trivial
    layer."""
    p = _layered_n1_slow_formation_params()
    # Skip the very low-f cutoff region; pick frequencies safely
    # above the geometric cutoff f ~ V_S / (2 pi a) ~ 1900 Hz
    # (which the layer can shift slightly upward).
    f = np.geomspace(3000.0, 15000.0, 12)

    res = flexural_dispersion_layered(
        f,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(p["layer"],),
    )
    assert np.all(np.isfinite(res.slowness))
    # Slowness increases monotonically with frequency in slow-
    # formation flexural (1/V_S at cutoff -> 1/V_R at high f, with
    # V_R < V_S so 1/V_R > 1/V_S). Tiny negative tolerance for
    # asymptotic-flatness rounding noise.
    diffs = np.diff(res.slowness)
    assert np.all(diffs > -1.0e-9)


def test_flexural_dispersion_layered_harder_layer_speeds_up_flexural():
    """Headline physics validation: a layer with ``V_S_layer >
    V_S_formation`` (harder near-borehole zone) speeds up the
    flexural wave -- the layered slowness is BELOW the unlayered
    slowness at the same frequency. Direct test of the qualitative
    expectation behind altered-zone interpretation: stiffer
    near-wall material shifts flexural slowness toward the layer's
    Rayleigh-like speed (faster than the formation's).

    Quantitative: the smoke test above showed ~1-1.3% speedup at
    a few kHz; this test confirms the inequality holds at every
    frequency in a typical band."""
    vp, vs, rho = 3000.0, 1200.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = np.linspace(3000.0, 8000.0, 10)

    res_unlayered = flexural_dispersion(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    hard_layer = BoreholeLayer(
        vp=3500.0,
        vs=1500.0,
        rho=2400.0,
        thickness=0.01,
    )
    res_layered = flexural_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(hard_layer,),
    )
    # Both fully populated in this band.
    assert np.all(np.isfinite(res_unlayered.slowness))
    assert np.all(np.isfinite(res_layered.slowness))
    # Harder layer => faster flexural => smaller slowness.
    assert np.all(res_layered.slowness < res_unlayered.slowness)
    # Speedup should be within physically reasonable range
    # (0.1% to 5% for a 1 cm layer with modest contrast).
    speedup_frac = 1.0 - res_layered.slowness / res_unlayered.slowness
    assert np.all(speedup_frac > 0.001)
    assert np.all(speedup_frac < 0.05)
