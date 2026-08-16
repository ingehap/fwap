"""
Plans G, G' and G'': the cased-hole stack at n=0, n=1 and n=2.

One of six modules split out of ``tests/test_cylindrical_solver.py``.
The state-vector matrix ``E(r)``, the per-layer propagator, the stacked
10x10 determinant and the public entry points, for each azimuthal
order in turn, plus the fast-formation follow-ups that came after
G''.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fwap.cylindrical_solver import (
    BoreholeLayer,
    BoreholeMode,
    _layer_e_matrix_n0,
    _layer_e_matrix_n1,
    _layer_e_matrix_n2,
    _layer_propagator_n0,
    _layer_propagator_n1,
    _layer_propagator_n2,
    _layered_n0_row1_at_a,
    _layered_n0_row2_at_a,
    _layered_n0_row3_at_a,
    _layered_n0_row5_at_b,
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
    _modal_determinant_n0_cased,
    _modal_determinant_n0_layered,
    _modal_determinant_n1_cased,
    _modal_determinant_n1_layered,
    _validate_borehole_layers_stacked,
    _validate_flexural_layers_stacked,
    flexural_dispersion,
    flexural_dispersion_layered,
    stoneley_dispersion,
    stoneley_dispersion_layered,
)
from tests._solver_media import (
    SLOW_A,
    SLOW_RHO,
    SLOW_RHO_F,
    SLOW_VF,
    SLOW_VP,
    SLOW_VS,
)

# =====================================================================
# Plan item G.0 -- public-API foundation for cased-hole multi-layer
# =====================================================================
#
# G.0 widens the multi-layer dispatch in stoneley_dispersion_layered
# / flexural_dispersion_layered with: (a) a sharper NotImplementedError
# message that points at the G.c / G.d / G' follow-ups (verified by
# the existing multilayer_raises_not_implemented tests above, with
# their match strings updated), and (b) a new helper
# _validate_borehole_layers_stacked that wraps F's per-layer
# validation with the borehole-radius check. The propagator-matrix
# path itself lands in G.b / G.c / G.d.


def test_validate_borehole_layers_stacked_accepts_typical_two_layer_stack():
    """A casing + cement geometry passes the stacked validator
    without raising. Same ``BoreholeLayer`` validation rules as
    F's per-layer validator, plus ``a > 0``."""
    casing = BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01)
    cement = BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.05)
    # Should not raise.
    _validate_borehole_layers_stacked((casing, cement), a=0.1)


def test_validate_borehole_layers_stacked_accepts_empty_stack():
    """Empty stack ``()`` is the degenerate "no extra layers" case
    and validates trivially as long as ``a > 0``."""
    _validate_borehole_layers_stacked((), a=0.1)


def test_validate_borehole_layers_stacked_rejects_zero_thickness_in_multi_stack():
    """A zero-thickness layer in a multi-layer stack is rejected
    by the per-layer validation (delegated to
    ``_validate_borehole_layers``). The error message identifies
    the offending index."""
    bad = BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.0)
    casing = BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01)
    with pytest.raises(ValueError, match=r"layers\[1\].*thickness must be positive"):
        _validate_borehole_layers_stacked((casing, bad), a=0.1)


def test_validate_borehole_layers_stacked_rejects_non_positive_a():
    """Non-positive borehole radius is rejected with a clear
    error. Catches it earlier than the public-API dispatch, which
    is useful when G.c starts using the helper."""
    casing = BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01)
    cement = BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.05)
    with pytest.raises(ValueError, match="a must be positive"):
        _validate_borehole_layers_stacked((casing, cement), a=0.0)
    with pytest.raises(ValueError, match="a must be positive"):
        _validate_borehole_layers_stacked((casing, cement), a=-0.1)


def test_stoneley_dispersion_layered_zero_and_one_layer_paths_unchanged():
    """G.0 must not perturb the two existing collapse paths
    (``len(layers) == 0`` and ``len(layers) == 1``). Regression:
    each produces the same slowness curve at a representative
    frequency."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = np.array([5000.0])

    # Empty-layer path: dispatches to stoneley_dispersion.
    res_empty = stoneley_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(),
    )
    res_unlayered = stoneley_dispersion(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    np.testing.assert_array_equal(res_empty.slowness, res_unlayered.slowness)

    # Single-layer path: F.1 hand-coded determinant. Just smoke;
    # bit-equivalent regressions are exercised elsewhere.
    layer = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.005)
    res_one = stoneley_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer,),
    )
    assert np.isfinite(res_one.slowness[0])
    assert isinstance(res_one, BoreholeMode)
    assert res_one.azimuthal_order == 0


# =====================================================================
# Plan item G.b.1 -- mode-amplitude-to-state-vector matrix E(r)
# =====================================================================
#
# Per-element oracle: at r=a, the layer-amplitude columns of
# F.1.b row 1 / 2 / 3 (with explicit sign factors per the BC's
# subtraction convention) match rows 0, 2, 3 of E(a). At r=b,
# F.1.b row 5 layer cols match row 1 (u_z) of E(b). Together
# these cover all four rows of E.


def _typical_g_b1_layer_params():
    """Representative non-isotropic-collapse layer + (kz, omega)
    fixture for G.b.1 / G.b.2 tests. Sits in the slow-formation
    bound regime so the propagator-matrix path is well-defined."""
    return dict(
        vp=3500.0,
        vs=1800.0,
        rho=2100.0,
        kz=2.0 * np.pi * 5000.0 / 1500.0,  # bound: kz > omega/V_S
        omega=2.0 * np.pi * 5000.0,
    )


def test_layer_e_matrix_n0_row0_matches_F1_row1_at_a_layer_cols():
    """Row 0 of E(a) (u_r) matches the layer-amplitude columns
    (1..5) of ``_layered_n0_row1_at_a`` with a sign flip: F.1's
    BC1 is ``u_r^(f) - u_r^(m) = 0``, so the layer side is
    negated in the row builder. This is the cleanest per-element
    oracle for the u_r row of E(r)."""
    p = _typical_g_b1_layer_params()
    layer = BoreholeLayer(vp=p["vp"], vs=p["vs"], rho=p["rho"], thickness=0.005)
    a = 0.1
    # E(a) for the layer.
    E = _layer_e_matrix_n0(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=a,
    )
    # F.1.b row 1: signature uses (vp, vs, rho) for the formation
    # half-space; the layer is passed via ``layer``. Row 1 doesn't
    # touch the formation parameters except for signature uniformity.
    row1 = _layered_n0_row1_at_a(
        kz=p["kz"],
        omega=p["omega"],
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
        layer=layer,
    )
    # Layer cols 1..5 of row 1 = -E[0, :] (negation from f - m).
    np.testing.assert_allclose(
        row1[1:5].real,
        -E[0, :],
        rtol=1.0e-12,
    )


def test_layer_e_matrix_n0_row2_matches_F1_row2_at_a_layer_cols():
    """Row 2 of E(a) (sigma_rr) matches the layer cols of
    ``_layered_n0_row2_at_a`` with a sign flip: BC2 is
    ``-(sigma_rr^(m) + P^(f)) = 0``, layer side negated."""
    p = _typical_g_b1_layer_params()
    layer = BoreholeLayer(vp=p["vp"], vs=p["vs"], rho=p["rho"], thickness=0.005)
    a = 0.1
    E = _layer_e_matrix_n0(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=a,
    )
    row2 = _layered_n0_row2_at_a(
        kz=p["kz"],
        omega=p["omega"],
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
        layer=layer,
    )
    np.testing.assert_allclose(
        row2[1:5].real,
        -E[2, :],
        rtol=1.0e-12,
    )


def test_layer_e_matrix_n0_row3_matches_F1_row3_at_a_layer_cols():
    """Row 3 of E(a) (sigma_rz) matches the layer cols of
    ``_layered_n0_row3_at_a`` with NO sign flip: BC3 is
    ``sigma_rz^(m) = 0`` (no subtraction with the fluid)."""
    p = _typical_g_b1_layer_params()
    layer = BoreholeLayer(vp=p["vp"], vs=p["vs"], rho=p["rho"], thickness=0.005)
    a = 0.1
    E = _layer_e_matrix_n0(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=a,
    )
    row3 = _layered_n0_row3_at_a(
        kz=p["kz"],
        omega=p["omega"],
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
        layer=layer,
    )
    np.testing.assert_allclose(
        row3[1:5].real,
        E[3, :],
        rtol=1.0e-12,
    )


def test_layer_e_matrix_n0_row1_uz_matches_F1_row5_at_b_layer_cols():
    """Row 1 of E(b) (u_z) matches the layer cols of
    ``_layered_n0_row5_at_b`` with NO sign flip: BC5 is
    ``u_z^(m)(b) - u_z^(s)(b) = 0`` and the layer cols carry the
    layer's direct contribution (the formation cols carry the
    subtracted contribution). Validates the u_z row of E,
    which has no analog at r=a (the fluid doesn't impose u_z
    continuity at the borehole wall)."""
    p = _typical_g_b1_layer_params()
    layer = BoreholeLayer(vp=p["vp"], vs=p["vs"], rho=p["rho"], thickness=0.005)
    a = 0.1
    b = a + layer.thickness
    E = _layer_e_matrix_n0(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=b,
    )
    row5 = _layered_n0_row5_at_b(
        kz=p["kz"],
        omega=p["omega"],
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
        layer=layer,
    )
    np.testing.assert_allclose(
        row5[1:5].real,
        E[1, :],
        rtol=1.0e-12,
    )


def test_layer_e_matrix_n0_returns_nan_below_bound_floor():
    """Below the layer's bound floor (``kz < omega / V_S``), at
    least one of ``p^2``, ``s^2`` becomes negative -- the Bessel
    arguments would be imaginary. The helper returns NaN-filled
    so downstream propagator / determinant evaluations propagate
    NaN cleanly (brentq-safe convention, mirrors
    ``_modal_determinant_n0`` and friends)."""
    omega = 2.0 * np.pi * 5000.0
    vp, vs, rho = 3500.0, 1800.0, 2100.0
    # kz well below omega/V_S.
    kz = omega / vs * 0.5
    with np.errstate(invalid="ignore"):
        E = _layer_e_matrix_n0(
            kz=kz,
            omega=omega,
            vp=vp,
            vs=vs,
            rho=rho,
            r=0.1,
        )
    assert np.all(np.isnan(E))


def test_layer_e_matrix_n0_determinant_nonzero_in_bound_regime():
    """The G.b.2 propagator path requires inverting E(r). Confirm
    that ``det(E(r))`` is well above floating-point noise for a
    representative bound-regime ``(kz, omega, layer)``. The
    quantitative budget is loose -- the absolute scale of
    ``det(E)`` depends on the Bessel-pack magnitudes, which can
    be very large or very small; we just want to rule out the
    near-singular case that would defeat the inverse."""
    p = _typical_g_b1_layer_params()
    a = 0.1
    E = _layer_e_matrix_n0(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=a,
    )
    det = float(np.linalg.det(E))
    # Just a finite, non-zero determinant. The propagator
    # round-trip oracle in G.b.2 will catch any conditioning
    # issue more sharply.
    assert np.isfinite(det)
    assert abs(det) > 0.0


# =====================================================================
# Plan item G.b.2 -- per-layer propagator P(r_outer | r_inner)
# =====================================================================
#
# Group-law oracles for ``_layer_propagator_n0`` plus an end-to-end
# state-vector continuity check. Each oracle is independent of the
# F.1.b transcription used in G.b.1, so this layer adds genuinely
# new constraints on top of the per-element match.


def test_layer_propagator_n0_identity_when_r_inner_equals_r_outer():
    """Identity oracle: ``r_inner == r_outer`` -> propagator is
    ``eye(4)`` to floating-point precision. Catches sign / shape
    errors in the solve."""
    p = _typical_g_b1_layer_params()
    P = _layer_propagator_n0(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=0.105,
        r_outer=0.105,
    )
    np.testing.assert_array_equal(P, np.eye(4))


def test_layer_propagator_n0_round_trip_preserves_state_vector():
    """Round-trip oracle: applying ``P(a|b) @ P(b|a)`` to a
    physical state vector ``v`` returns ``v`` to floating-point
    precision. Equivalent to ``P(a|b) P(b|a) = I`` in exact
    arithmetic; phrasing as a state-vector identity avoids the
    spurious ~1e-6 off-diagonals from the disparate-magnitude
    rows (displacement ~ O(1) vs stress ~ O(mu) ~ O(1e10)) that
    would defeat ``assert_allclose(M, eye, atol=1e-10)`` directly
    at the matrix level."""
    p = _typical_g_b1_layer_params()
    a = 0.1
    b = a + 0.005
    P_b_from_a = _layer_propagator_n0(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=a,
        r_outer=b,
    )
    P_a_from_b = _layer_propagator_n0(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=b,
        r_outer=a,
    )
    # Physical state vector (displacement ~ O(1), stress ~ O(mu)).
    mu = p["rho"] * p["vs"] ** 2
    v = np.array([1.0, 2.0, 3.0 * mu, 4.0 * mu])
    v_round = P_a_from_b @ (P_b_from_a @ v)
    np.testing.assert_allclose(v_round, v, rtol=1.0e-10)
    # Other direction.
    v_round_other = P_b_from_a @ (P_a_from_b @ v)
    np.testing.assert_allclose(v_round_other, v, rtol=1.0e-10)


def test_layer_propagator_n0_composition_law():
    """Composition oracle: ``P(r3|r1) ~ P(r3|r2) @ P(r2|r1)`` for
    any intermediate ``r2 in (r1, r3)``. The propagator-group law
    in the radial coordinate. Independent of the F.1.b oracle
    in G.b.1."""
    p = _typical_g_b1_layer_params()
    r1, r2, r3 = 0.1, 0.105, 0.115
    P_3_from_1 = _layer_propagator_n0(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=r1,
        r_outer=r3,
    )
    P_2_from_1 = _layer_propagator_n0(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=r1,
        r_outer=r2,
    )
    P_3_from_2 = _layer_propagator_n0(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=r2,
        r_outer=r3,
    )
    np.testing.assert_allclose(P_3_from_1, P_3_from_2 @ P_2_from_1, atol=1.0e-10)


def test_layer_propagator_n0_state_vector_continuity():
    """End-to-end state-vector check: pick an arbitrary amplitude
    vector ``c``; compute ``v(r1) = E(r1) c`` and apply
    ``P(r2|r1)`` to get ``v(r2)``; verify the result matches
    ``E(r2) c`` directly. Strongest single-test oracle for the
    G.b.1 + G.b.2 chain combined."""
    p = _typical_g_b1_layer_params()
    r1, r2 = 0.1, 0.115
    E_r1 = _layer_e_matrix_n0(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=r1,
    )
    E_r2 = _layer_e_matrix_n0(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=r2,
    )
    P = _layer_propagator_n0(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=r1,
        r_outer=r2,
    )
    # Arbitrary amplitude vector.
    c = np.array([1.3, -0.7, 2.1, 0.4])
    v_r1 = E_r1 @ c
    v_r2_via_P = P @ v_r1
    v_r2_direct = E_r2 @ c
    np.testing.assert_allclose(v_r2_via_P, v_r2_direct, rtol=1.0e-10)


def test_layer_propagator_n0_returns_nan_below_bound_floor():
    """Below the layer's bound floor, ``E(r)`` is NaN-filled; the
    propagator inherits the NaN. Confirms brentq-safe propagation
    so the G.c assembly's bound-regime gate is reliable."""
    omega = 2.0 * np.pi * 5000.0
    vp, vs, rho = 3500.0, 1800.0, 2100.0
    kz = omega / vs * 0.5  # well below bound floor
    with np.errstate(invalid="ignore"):
        P = _layer_propagator_n0(
            kz=kz,
            omega=omega,
            vp=vp,
            vs=vs,
            rho=rho,
            r_inner=0.1,
            r_outer=0.105,
        )
    assert np.all(np.isnan(P))


# =====================================================================
# Plan item G.c -- stacked modal determinant
# =====================================================================
#
# Tests anchor on the N=1 collapse to F.1 (``_modal_determinant_n0_layered``)
# as the floating-point oracle, plus a few oracles that exercise
# the N >= 2 propagator chain (order-matters; two-identical-layers
# equivalent to one double-thickness layer via the group law).


def _typical_g_c_params():
    """Slow-formation cased-hole fixture for G.c tests. Keeps the
    Stoneley root in the bound regime across a representative
    band; layers are typical casing / cement / mudcake values."""
    return dict(
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )


def test_modal_determinant_n0_cased_N1_matches_F1_off_root():
    """N=1 floating-point oracle: at any (kz, omega) in the bound
    regime away from the Stoneley root, G.c's determinant matches
    F.1's ``_modal_determinant_n0_layered`` to relative precision
    ``rtol=1e-10`` (no extra scale factor; the propagator chain
    at N=1 reduces P_1 @ E_1(a) -> E_1(b), exactly the F.1 form).

    Strongest pinning of the G.c assembly against the existing
    F.1 row-builder transcription that has shipped through F.1.b.4."""
    p = _typical_g_c_params()
    layer = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    # Pick kz away from the Stoneley root (1.05x off the unlayered
    # bound floor; well into the bound regime).
    kz = omega / p["vf"] * 1.05
    det_F1 = _modal_determinant_n0_layered(
        kz,
        omega,
        p["vp"],
        p["vs"],
        p["rho"],
        p["vf"],
        p["rho_f"],
        p["a"],
        layer=layer,
    )
    det_Gc = _modal_determinant_n0_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(layer,),
    )
    assert det_Gc == pytest.approx(det_F1, rel=1.0e-10)


def test_modal_determinant_n0_cased_N1_vanishes_at_F1_brentq_root():
    """N=1 brentq-root oracle: at the Stoneley root recovered by
    ``stoneley_dispersion_layered(layers=(layer,))``, G.c's
    determinant is many orders of magnitude smaller than its
    value 1% off the root. Confirms the brentq pipeline G.d will
    drive against G.c will find the same root as F.1."""
    p = _typical_g_c_params()
    layer = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0

    res = stoneley_dispersion_layered(
        np.array([5000.0]),
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(layer,),
    )
    kz_root = float(res.slowness[0]) * omega
    det_at = _modal_determinant_n0_cased(
        kz_root,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(layer,),
    )
    det_off = _modal_determinant_n0_cased(
        kz_root * 1.01,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(layer,),
    )
    assert abs(det_at) < abs(det_off) * 1.0e-10


def test_modal_determinant_n0_cased_returns_nan_below_bound_floor():
    """``kz < omega / V_f`` -> ``F_f^2 < 0`` -> NaN; or ``kz`` below
    the slowest layer / formation V_S -> propagator chain returns
    NaN. Either way the assembly propagates NaN cleanly so brentq
    can reject the bracket."""
    p = _typical_g_c_params()
    layer = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    # kz well below the fluid floor.
    kz = omega / p["vf"] * 0.5
    with np.errstate(invalid="ignore"):
        det = _modal_determinant_n0_cased(
            kz,
            omega,
            vp=p["vp"],
            vs=p["vs"],
            rho=p["rho"],
            vf=p["vf"],
            rho_f=p["rho_f"],
            a=p["a"],
            layers=(layer,),
        )
    assert np.isnan(det)


def test_modal_determinant_n0_cased_two_identical_layers_equals_one_double_thickness():
    """Group-law oracle for the propagator chain: two contiguous
    identical layers (L, L) of thickness ``h`` each compose to a
    single layer of thickness ``2h`` via ``P_2 @ P_1 = P(r3 | r1)``.

    Direct test that G.c.7 propagator-chain composition is wired
    correctly. Independent of F.1: would catch any error in the
    inside-out layer-radii arithmetic or the chain accumulator."""
    p = _typical_g_c_params()
    omega = 2.0 * np.pi * 5000.0
    kz = omega / p["vf"] * 1.05  # bound regime
    # Single layer of thickness 0.01.
    L_double = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.01)
    # Two layers of thickness 0.005 each, same params.
    L_half = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.005)
    det_single = _modal_determinant_n0_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(L_double,),
    )
    det_split = _modal_determinant_n0_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(L_half, L_half),
    )
    # The two assemblies use different innermost-layer E_1(a) factors
    # (both layers have the same params, but the half-layer's
    # E_1(a) has different bessel-pack values than the double-layer's
    # E_1(a)). The brentq root in kz is the same; the absolute
    # determinant magnitudes can differ. Verify the root match by
    # checking that |det_single| / |det_split| is the same as the
    # ratio of innermost-layer det(E_1(a))s (the layer-1-amplitude
    # scale factor that distinguishes them).
    #
    # Simplest oracle: both should change sign across the same kz
    # window, captured by the same-sign / same-magnitude-order test.
    # Tight ratio: at this off-root kz, both should be the SAME up
    # to an overall sign because L_double's E_1(a) is identical to
    # L_half's E_1(a) (same vp/vs/rho/r=a). The propagator chain
    # composes to the same total transformation across thickness 0.01.
    assert det_single == pytest.approx(det_split, rel=1.0e-10)


def test_modal_determinant_n0_cased_order_matters_at_N2():
    """Physical sanity: with two distinct layers ``(L_a, L_b)``,
    swapping the order to ``(L_b, L_a)`` produces a different
    determinant -- the inside-out layer ordering is a physical
    parameter (a casing inside a cement looks different from a
    cement inside a casing).

    Independent of F.1; would catch any error where the
    propagator chain ignored layer ordering or composed in the
    wrong direction."""
    p = _typical_g_c_params()
    omega = 2.0 * np.pi * 5000.0
    L_a = BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01)  # casing
    L_b = BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.01)  # cement
    # kz safely above the slowest-shear bound floor (cement V_S = 1300).
    kz = omega / min(L_a.vs, L_b.vs, p["vs"], p["vf"]) * 1.05
    det_ab = _modal_determinant_n0_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(L_a, L_b),
    )
    det_ba = _modal_determinant_n0_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(L_b, L_a),
    )
    assert np.isfinite(det_ab) and np.isfinite(det_ba)
    # Layer permutation is non-trivial (well above floating-point noise).
    rel_diff = abs(det_ab - det_ba) / max(abs(det_ab), abs(det_ba))
    assert rel_diff > 0.01


def test_modal_determinant_n0_cased_N2_runs_smoke():
    """Smoke test for the N=2 (cased-hole) path: a typical
    casing + cement geometry produces a finite real determinant
    at a representative bound-regime ``kz``."""
    p = _typical_g_c_params()
    omega = 2.0 * np.pi * 5000.0
    casing = BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01)
    cement = BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.05)
    # kz safely above the slowest-shear bound floor.
    kz = omega / min(casing.vs, cement.vs, p["vs"], p["vf"]) * 1.05
    det = _modal_determinant_n0_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(casing, cement),
    )
    assert np.isfinite(det)
    assert isinstance(det, float)


# =====================================================================
# Plan item G.d -- public-API hook for cased-hole Stoneley
# =====================================================================
#
# Replaces the G.0 ``len(layers) > 1 -> NotImplementedError`` raise
# with a brentq loop on ``_modal_determinant_n0_cased``. Tests
# anchor on a multi-layer regression (matching the F.1 single-layer
# brentq root when an N=2 stack collapses to the F.1 case),
# layer-permutation distinctness, and a three-layer smoke.


def _typical_cased_geometry():
    """Realistic casing + cement geometry used as the fixture for
    G.d cased-hole tests. V_S_cement = 1800 m/s (high-strength
    cement) keeps the cement above V_f = 1500 so the bound floor
    stays at the fluid value across the dipole-sonic band."""
    return dict(
        casing=BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01),
        cement=BoreholeLayer(vp=2300.0, vs=1800.0, rho=1900.0, thickness=0.05),
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )


def test_stoneley_dispersion_layered_N1_path_unchanged_after_G_d():
    """Regression: the N=1 dispatch (F.1 hand-coded path) remains
    functional after the G.d multi-layer wiring. Picks a typical
    single-layer setup and verifies the slowness curve is finite
    and well-formed."""
    p = _typical_g_c_params()
    layer = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.005)
    f = np.linspace(2000.0, 8000.0, 6)
    res = stoneley_dispersion_layered(
        f,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(layer,),
    )
    assert np.all(np.isfinite(res.slowness))
    assert res.name == "Stoneley"
    assert res.azimuthal_order == 0


def test_stoneley_dispersion_layered_N2_runs_smoke():
    """G.d two-layer regression: a typical casing + cement
    geometry produces a finite, smoothly-dispersive Stoneley
    slowness curve across the dipole-sonic band."""
    g = _typical_cased_geometry()
    f = np.linspace(1000.0, 12000.0, 8)
    res = stoneley_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], g["cement"]),
    )
    assert np.all(np.isfinite(res.slowness))
    assert res.name == "Stoneley"
    assert res.azimuthal_order == 0
    np.testing.assert_array_equal(res.freq, f)
    # Smoothness fence: relative step-to-step change capped at 5%.
    rel_steps = np.abs(np.diff(res.slowness)) / res.slowness[:-1]
    assert np.all(rel_steps < 0.05)


def test_stoneley_dispersion_layered_N2_returns_borehole_mode():
    """``BoreholeMode`` return-type contract on the multi-layer
    dispatch (G.d)."""
    g = _typical_cased_geometry()
    f = np.linspace(2000.0, 6000.0, 4)
    res = stoneley_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], g["cement"]),
    )
    assert isinstance(res, BoreholeMode)
    assert res.attenuation_per_meter is None  # bound mode


def test_stoneley_dispersion_layered_N2_layer_permutation_changes_slowness():
    """Casing-inside-cement and cement-inside-casing produce
    distinct slowness curves -- the inside-out layer ordering is
    a physical parameter, not a labelling convention."""
    g = _typical_cased_geometry()
    f = np.array([5000.0])
    res_cs = stoneley_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], g["cement"]),
    )
    res_sc = stoneley_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["cement"], g["casing"]),
    )
    assert np.all(np.isfinite(res_cs.slowness))
    assert np.all(np.isfinite(res_sc.slowness))
    rel_diff = abs(res_cs.slowness[0] - res_sc.slowness[0]) / res_cs.slowness[0]
    assert rel_diff > 0.001


def test_stoneley_dispersion_layered_N2_collapse_to_N1_via_thin_outer_layer():
    """Two-layer-collapse oracle: with a vanishingly-thin outer
    layer that has the formation's parameters, the G.d two-layer
    path should match the F.1 single-layer slowness to high
    precision. Confirms the propagator chain and BC bookkeeping
    handle the trivial outer-layer limit cleanly."""
    p = _typical_g_c_params()
    f = np.array([5000.0])
    layer1 = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.005)
    # Outer "layer" has formation properties + tiny thickness;
    # equivalent to no outer layer at all.
    layer2_trivial = BoreholeLayer(
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        thickness=1.0e-5,
    )
    res_one_layer = stoneley_dispersion_layered(
        f,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(layer1,),
    )
    res_two_layer = stoneley_dispersion_layered(
        f,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(layer1, layer2_trivial),
    )
    assert res_two_layer.slowness[0] == pytest.approx(
        res_one_layer.slowness[0],
        rel=1.0e-4,
    )


def test_stoneley_dispersion_layered_three_layer_runs_smoke():
    """Smoke test for N=3 (casing + cement + mudcake): produces
    a finite slowness curve. No analytic oracle; just confirms
    the propagator chain extends past N=2 cleanly."""
    g = _typical_cased_geometry()
    mudcake = BoreholeLayer(vp=2000.0, vs=1600.0, rho=1700.0, thickness=0.003)
    f = np.linspace(2000.0, 8000.0, 5)
    res = stoneley_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], g["cement"], mudcake),
    )
    assert np.all(np.isfinite(res.slowness))


# =====================================================================
# Plan item G.e -- validation hardening for the cased-hole solver
# =====================================================================
#
# Mirror of F.2.e / H.e for the propagator-matrix Stoneley path.
# An external cased-Stoneley reference is still wanted; this block
# named "Tang & Cheng 2004 fig 7.1", which does not exist -- that
# book has six chapters. The four tests below are the cheap
# self-consistency / collapse oracles that the propagator chain
# can satisfy without external reference data.


def test_modal_determinant_n0_cased_vanishes_at_converged_root_multi_freq():
    """Self-consistency: at every brentq-converged ``k_z`` from
    ``stoneley_dispersion_layered`` (cased-hole, two-layer), the
    propagator-matrix determinant is many orders of magnitude
    smaller than its value at ``k_z * 1.005``. Multi-frequency
    sharper than the H.c.2 / G.d single-frequency oracle: catches
    regressions where the brentq pipeline converges to something
    other than the true root for some frequencies."""
    g = _typical_cased_geometry()
    f = np.geomspace(1500.0, 10000.0, 6)
    res = stoneley_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], g["cement"]),
    )
    assert np.all(np.isfinite(res.slowness))
    for i, fi in enumerate(f):
        omega = 2.0 * np.pi * float(fi)
        kz_root = float(res.slowness[i]) * omega
        det_at = _modal_determinant_n0_cased(
            kz_root,
            omega,
            vp=g["vp"],
            vs=g["vs"],
            rho=g["rho"],
            vf=g["vf"],
            rho_f=g["rho_f"],
            a=g["a"],
            layers=(g["casing"], g["cement"]),
        )
        det_off = _modal_determinant_n0_cased(
            kz_root * 1.005,
            omega,
            vp=g["vp"],
            vs=g["vs"],
            rho=g["rho"],
            vf=g["vf"],
            rho_f=g["rho_f"],
            a=g["a"],
            layers=(g["casing"], g["cement"]),
        )
        assert abs(det_at) < abs(det_off) * 1.0e-6, (
            f"f={fi:.1f}: |det_at|={abs(det_at):.3e} not << "
            f"|det_off|={abs(det_off):.3e}"
        )


def test_stoneley_dispersion_layered_thin_inner_layer_collapses_to_outer_only():
    """Two-layer-collapse oracle: with a vanishingly-thin INNER
    layer that has the outer layer's parameters, the G.d
    two-layer slowness should match the F.1 single-layer
    answer with just the outer layer (effectively ignoring
    the trivial inner annulus). Mirror of the G.d
    ``thin_outer_layer`` test, this time exercising the
    propagator chain's first link."""
    p = _typical_g_c_params()
    f = np.array([5000.0])
    outer = BoreholeLayer(vp=3500.0, vs=1800.0, rho=2100.0, thickness=0.01)
    inner_trivial = BoreholeLayer(
        vp=outer.vp,
        vs=outer.vs,
        rho=outer.rho,
        thickness=1.0e-5,
    )
    res_one_layer = stoneley_dispersion_layered(
        f,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(outer,),
    )
    res_two_layer = stoneley_dispersion_layered(
        f,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(inner_trivial, outer),
    )
    assert res_two_layer.slowness[0] == pytest.approx(
        res_one_layer.slowness[0],
        rel=1.0e-4,
    )


def test_stoneley_dispersion_layered_two_formation_layers_collapse_to_unlayered():
    """Master-plan G validation bullet 1: with both annular
    layers carrying formation properties, the multi-layer
    Stoneley slowness should match the unlayered
    ``stoneley_dispersion`` answer to high precision. The
    layers are physically vacuous (just slabs of formation
    pretending to be a casing + cement).

    Strongest pinning of the G.d brentq pipeline against the
    pre-G unlayered baseline: would catch any bias introduced
    by the propagator-chain assembly that survives at small
    layer-formation contrast."""
    p = _typical_g_c_params()
    f = np.linspace(2000.0, 8000.0, 5)
    # Two layers with formation properties; thicknesses are
    # arbitrary since the layer = formation collapse is exact.
    formation_l1 = BoreholeLayer(
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        thickness=0.01,
    )
    formation_l2 = BoreholeLayer(
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        thickness=0.05,
    )
    res_unlayered = stoneley_dispersion(
        f,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
    )
    res_cased = stoneley_dispersion_layered(
        f,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(formation_l1, formation_l2),
    )
    # Tolerance sized to brentq's xtol (1e-10) propagated into
    # slowness via division by omega ~ 1e4: relative ~1e-6.
    np.testing.assert_allclose(
        res_cased.slowness,
        res_unlayered.slowness,
        rtol=1.0e-6,
    )


def test_stoneley_dispersion_layered_stiffer_cement_speeds_up_stoneley():
    """Cement-bond physics: a stiffer cement (higher V_S)
    couples the Stoneley wave more strongly to the formation,
    pulling the slowness DOWN (faster wave) at the same casing
    geometry. Soft cement (lower V_S, closer to free-pipe)
    leaves the Stoneley closer to its fluid-coupled limit
    (slower wave; larger slowness).

    Direct test of the qualitative cement-bond logging
    signature without committing to digitised reference data
    (no external cased-Stoneley curve is available; the
    "Tang & Cheng 2004 fig 7.1" this named does not exist).
    Quantitative: the
    speedup is on the order of a few percent for typical
    cement-stiffness contrasts."""
    g = _typical_cased_geometry()
    f = np.array([5000.0])
    casing = g["casing"]
    cement_stiff = BoreholeLayer(
        vp=2500.0,
        vs=2000.0,
        rho=2000.0,
        thickness=g["cement"].thickness,
    )
    cement_soft = BoreholeLayer(
        vp=1900.0,
        vs=1600.0,
        rho=1700.0,
        thickness=g["cement"].thickness,
    )
    res_stiff = stoneley_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(casing, cement_stiff),
    )
    res_soft = stoneley_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(casing, cement_soft),
    )
    assert np.all(np.isfinite(res_stiff.slowness))
    assert np.all(np.isfinite(res_soft.slowness))
    # Stiffer cement -> smaller slowness (faster Stoneley).
    assert res_stiff.slowness[0] < res_soft.slowness[0]
    # Quantitative: at least 0.1% speedup.
    rel_speedup = (res_soft.slowness[0] - res_stiff.slowness[0]) / res_soft.slowness[0]
    assert rel_speedup > 0.001


# =====================================================================
# Plan item G'.0 -- public-API foundation for cased-hole flexural
# =====================================================================
#
# G'.0 sharpens the multi-layer NIE in flexural_dispersion_layered
# (verified by the test_flexural_dispersion_layered_two_layer_NIE_points_at_G_prime_c_d
# and ..._multilayer_raises_not_implemented tests above, with their
# match strings updated), and adds the per-layer slow-formation
# validator _validate_flexural_layers_stacked. The propagator-
# matrix path itself lands in G'.b / G'.c / G'.d.


def test_validate_flexural_layers_stacked_accepts_typical_harder_layer_stack():
    """A typical multi-layer stack with all layers harder than
    the formation (``layer.vs >= vs``) passes validation. Mirrors
    the slow-formation regime constraint that the F.2 single-layer
    path documents but does not enforce programmatically."""
    casing = BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01)
    cement = BoreholeLayer(vp=2900.0, vs=2700.0, rho=1900.0, thickness=0.05)
    # Formation V_S = 2500; both layers harder.
    _validate_flexural_layers_stacked(
        (casing, cement),
        a=0.1,
        vs=2500.0,
    )


def test_validate_flexural_layers_stacked_rejects_softer_layer():
    """A stack with one layer slower than the formation in shear
    is rejected with a clear error identifying the offending
    index. Catches a soft-cement-bond-like configuration that
    would otherwise drive the propagator path into the unbound
    regime."""
    casing = BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01)
    cement_soft = BoreholeLayer(
        vp=2300.0,
        vs=1300.0,
        rho=1900.0,
        thickness=0.05,
    )
    # Formation V_S = 2500; cement V_S = 1300 < 2500 -> reject.
    with pytest.raises(ValueError, match=r"layers\[1\].*layer\.vs"):
        _validate_flexural_layers_stacked(
            (casing, cement_soft),
            a=0.1,
            vs=2500.0,
        )


def test_validate_flexural_layers_stacked_rejects_softer_inner_layer():
    """Inner layer (index 0) softer than the formation is also
    rejected, with the index identified in the error."""
    casing_soft = BoreholeLayer(
        vp=2300.0,
        vs=1300.0,
        rho=1900.0,
        thickness=0.01,
    )
    cement = BoreholeLayer(vp=2900.0, vs=2700.0, rho=1900.0, thickness=0.05)
    with pytest.raises(ValueError, match=r"layers\[0\].*layer\.vs"):
        _validate_flexural_layers_stacked(
            (casing_soft, cement),
            a=0.1,
            vs=2500.0,
        )


def test_validate_flexural_layers_stacked_inherits_geometry_checks():
    """The flexural validator chains to
    ``_validate_borehole_layers_stacked`` for geometry checks
    (zero-thickness, non-positive ``a``, etc.). Pin one
    geometry rejection here so the chain is explicit."""
    casing = BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01)
    cement = BoreholeLayer(vp=2900.0, vs=2700.0, rho=1900.0, thickness=0.05)
    with pytest.raises(ValueError, match="a must be positive"):
        _validate_flexural_layers_stacked(
            (casing, cement),
            a=0.0,
            vs=2500.0,
        )


def test_flexural_dispersion_layered_zero_and_one_layer_paths_unchanged_after_G_prime_0():
    """Regression: the two existing F.2 dispatch paths
    (``len(layers) == 0`` and ``len(layers) == 1``) remain
    functional after the G'.0 NIE message edit. Smoke only;
    bit-equivalent regressions are covered elsewhere. Uses the
    slow-formation fixture (V_S < V_f) since the F.2 single-
    layer path requires it."""
    f = np.array([5000.0])

    # Empty-layer path: dispatches to flexural_dispersion.
    res_empty = flexural_dispersion_layered(
        f,
        vp=SLOW_VP,
        vs=SLOW_VS,
        rho=SLOW_RHO,
        vf=SLOW_VF,
        rho_f=SLOW_RHO_F,
        a=SLOW_A,
        layers=(),
    )
    res_unlayered = flexural_dispersion(
        f,
        vp=SLOW_VP,
        vs=SLOW_VS,
        rho=SLOW_RHO,
        vf=SLOW_VF,
        rho_f=SLOW_RHO_F,
        a=SLOW_A,
    )
    np.testing.assert_array_equal(res_empty.slowness, res_unlayered.slowness)
    assert res_empty.azimuthal_order == 1

    # Single-layer F.2 hand-coded path. Layer harder than the
    # slow formation (vs > SLOW_VS = 800).
    layer = BoreholeLayer(vp=2500.0, vs=1100.0, rho=2100.0, thickness=0.005)
    res_one = flexural_dispersion_layered(
        f,
        vp=SLOW_VP,
        vs=SLOW_VS,
        rho=SLOW_RHO,
        vf=SLOW_VF,
        rho_f=SLOW_RHO_F,
        a=SLOW_A,
        layers=(layer,),
    )
    assert isinstance(res_one, BoreholeMode)
    assert res_one.azimuthal_order == 1


# =====================================================================
# Plan item G'.b.1 -- 6x6 mode-amplitude-to-state-vector matrix at n=1
# =====================================================================
#
# Per-element oracle: every row of E(r) post-rescale matches an
# F.2.b/c row builder's layer-amplitude columns to ``rtol=1e-12``,
# with explicit BC sign factors (rows at r=a from BC1-4 with
# specific subtraction conventions; rows at r=b from BC5-10 all
# m-s positive).


def _typical_g_prime_b1_layer_params():
    """Slow-formation harder-than-formation layer fixture for
    G'.b.1 / G'.b.2 tests. ``layer.vs > formation.vs`` so the
    flexural mode stays bound in the annulus."""
    return dict(
        vp=2500.0,
        vs=1100.0,
        rho=2100.0,
        kz=2.0 * np.pi * 5000.0 / 800.0,  # bound: kz > omega/V_S
        omega=2.0 * np.pi * 5000.0,
    )


def test_layer_e_matrix_n1_at_a_rows_match_F2_row1_to_row4_layer_cols():
    """Per-element oracle at r=a: rows 0 (u_r), 3 (sigma_rr),
    4 (sigma_rz), 5 (sigma_rtheta) of E(a) match the layer
    cols (cols 1-4 for B/C and 7-8 for D in F.2's column packing
    ``[A | B_I, B_K, C_I, C_K | B, C | D_I, D_K | D]``) of the
    corresponding F.2 row builders, with BC sign factors:

    * Row 1 (BC1: u_r^f - u_r^m = 0): layer cols negated.
      ``-_layered_n1_row1_at_a[1:5, 7:9] == E[0, :]`` permuted.
    * Row 2 (BC2: -(sigma_rr^m + P^f) = 0): layer cols negated.
    * Row 3 (BC3: sigma_rtheta^m = 0): layer cols positive.
    * Row 4 (BC4: sigma_rz^m = 0): layer cols positive.

    F.2 col packing maps to E col packing as
    ``[B_I, B_K, C_I, C_K, D_I, D_K] = F2[1, 2, 3, 4, 7, 8]``."""
    p = _typical_g_prime_b1_layer_params()
    layer = BoreholeLayer(vp=p["vp"], vs=p["vs"], rho=p["rho"], thickness=0.005)
    a = 0.1
    E = _layer_e_matrix_n1(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=a,
    )

    # Helper: extract layer cols (1, 2, 3, 4, 7, 8) from F.2 row.
    def _layer_cols(row):
        return np.array([row[1], row[2], row[3], row[4], row[7], row[8]]).real

    # Row 0 (u_r) at a vs F.2 row 1 (BC1 negation).
    row1 = _layered_n1_row1_at_a(
        kz=p["kz"],
        omega=p["omega"],
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
        layer=layer,
    )
    np.testing.assert_allclose(_layer_cols(row1), -E[0, :], rtol=1.0e-12)

    # Row 3 (sigma_rr) at a vs F.2 row 2 (BC2 negation).
    row2 = _layered_n1_row2_at_a(
        kz=p["kz"],
        omega=p["omega"],
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
        layer=layer,
    )
    np.testing.assert_allclose(_layer_cols(row2), -E[3, :], rtol=1.0e-12)

    # Row 5 (sigma_rtheta) at a vs F.2 row 3 (BC3 no negation).
    row3 = _layered_n1_row3_at_a(
        kz=p["kz"],
        omega=p["omega"],
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
        layer=layer,
    )
    np.testing.assert_allclose(_layer_cols(row3), E[5, :], rtol=1.0e-12)

    # Row 4 (sigma_rz) at a vs F.2 row 4 (BC4 no negation).
    row4 = _layered_n1_row4_at_a(
        kz=p["kz"],
        omega=p["omega"],
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
        layer=layer,
    )
    np.testing.assert_allclose(_layer_cols(row4), E[4, :], rtol=1.0e-12)


def test_layer_e_matrix_n1_at_b_rows_match_F2_row5_to_row10_layer_cols():
    """Per-element oracle at r=b: all six rows of E(b) match the
    layer cols of F.2 rows 5-10 (BC5-10 are all m-s continuity
    with positive layer-side sign). Exhaustive coverage of every
    row of E."""
    p = _typical_g_prime_b1_layer_params()
    layer = BoreholeLayer(vp=p["vp"], vs=p["vs"], rho=p["rho"], thickness=0.005)
    a = 0.1
    b = a + layer.thickness
    E = _layer_e_matrix_n1(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=b,
    )

    def _layer_cols(row):
        return np.array([row[1], row[2], row[3], row[4], row[7], row[8]]).real

    # Row 0 (u_r) at b vs F.2 row 5.
    row5 = _layered_n1_row5_at_b(
        kz=p["kz"],
        omega=p["omega"],
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
        layer=layer,
    )
    np.testing.assert_allclose(_layer_cols(row5), E[0, :], rtol=1.0e-12)

    # Row 2 (u_theta) at b vs F.2 row 6.
    row6 = _layered_n1_row6_at_b(
        kz=p["kz"],
        omega=p["omega"],
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
        layer=layer,
    )
    np.testing.assert_allclose(_layer_cols(row6), E[2, :], rtol=1.0e-12)

    # Row 1 (u_z) at b vs F.2 row 7.
    row7 = _layered_n1_row7_at_b(
        kz=p["kz"],
        omega=p["omega"],
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
        layer=layer,
    )
    np.testing.assert_allclose(_layer_cols(row7), E[1, :], rtol=1.0e-12)

    # Row 3 (sigma_rr) at b vs F.2 row 8.
    row8 = _layered_n1_row8_at_b(
        kz=p["kz"],
        omega=p["omega"],
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
        layer=layer,
    )
    np.testing.assert_allclose(_layer_cols(row8), E[3, :], rtol=1.0e-12)

    # Row 5 (sigma_rtheta) at b vs F.2 row 9.
    row9 = _layered_n1_row9_at_b(
        kz=p["kz"],
        omega=p["omega"],
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
        layer=layer,
    )
    np.testing.assert_allclose(_layer_cols(row9), E[5, :], rtol=1.0e-12)

    # Row 4 (sigma_rz) at b vs F.2 row 10.
    row10 = _layered_n1_row10_at_b(
        kz=p["kz"],
        omega=p["omega"],
        vp=4500.0,
        vs=2500.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
        layer=layer,
    )
    np.testing.assert_allclose(_layer_cols(row10), E[4, :], rtol=1.0e-12)


def test_layer_e_matrix_n1_sparsity_pattern():
    """Pin the known-zero entries of E(r) at n=1, and the ones that
    are NOT zero after the roadmap-A.8 correction.

    * Row 1 (``u_z``) cols 4, 5 (``D_I``, ``D_K``) ARE zero: the SH
      potential ``psi_z`` does not contribute to ``u_z``.
    * Row 2 (``u_theta``) cols 2, 3 (``C_I``, ``C_K``) are NOT
      zero. The Hansen SV field ``curl curl(chi z)`` carries
      ``u_theta = i k_z (n/r) chi``, which vanishes only at n = 0.
      This test used to assert them zero, which held only for the
      azimuthal-only vector potential ``psi_theta e_theta`` the SV
      columns encoded before A.8 -- an ansatz that is not a
      solution of the elastodynamic equations for n >= 1.

    Only one (state-row, amplitude-col) pair decouples at n=1, so
    E(r) is denser than the F.2.a.6 erratum recorded."""
    p = _typical_g_prime_b1_layer_params()
    E = _layer_e_matrix_n1(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=0.1,
    )
    # u_z row, D cols: genuinely zero.
    assert E[1, 4] == 0.0
    assert E[1, 5] == 0.0
    # u_theta row, C cols: non-zero at n >= 1 (roadmap A.8).
    assert E[2, 2] != 0.0
    assert E[2, 3] != 0.0


def test_layer_e_matrix_n1_real_in_bound_regime():
    """All 36 entries are finite real in the bound regime
    post-rescale."""
    p = _typical_g_prime_b1_layer_params()
    E = _layer_e_matrix_n1(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=0.1,
    )
    assert np.all(np.isfinite(E))
    assert E.dtype == np.float64


def test_layer_e_matrix_n1_returns_nan_below_bound_floor():
    """Below the layer's bound floor (``kz < omega / V_S``), at
    least one of ``p^2``, ``s^2`` is negative -- the helper
    returns NaN-filled so downstream propagator and determinant
    evaluations propagate NaN cleanly."""
    omega = 2.0 * np.pi * 5000.0
    vp, vs, rho = 2500.0, 1100.0, 2100.0
    kz = omega / vs * 0.5  # well below bound floor
    with np.errstate(invalid="ignore"):
        E = _layer_e_matrix_n1(
            kz=kz,
            omega=omega,
            vp=vp,
            vs=vs,
            rho=rho,
            r=0.1,
        )
    assert np.all(np.isnan(E))


def test_layer_e_matrix_n1_determinant_nonzero_in_bound_regime():
    """The G'.b.2 propagator path requires inverting E(r). Just
    a finiteness + non-zero check; the round-trip oracle in
    G'.b.2 catches any sharper conditioning issue."""
    p = _typical_g_prime_b1_layer_params()
    E = _layer_e_matrix_n1(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=0.1,
    )
    det = float(np.linalg.det(E))
    assert np.isfinite(det)
    assert abs(det) > 0.0


# =====================================================================
# Plan item G'.b.2 -- per-layer propagator P(r_outer | r_inner) at n=1
# =====================================================================
#
# Group-law oracles for ``_layer_propagator_n1`` (6x6 sister of
# ``_layer_propagator_n0``). Round-trip uses the state-vector
# form to avoid the ``cond(E) ~ mu`` issue called out in G.b.2.


def test_layer_propagator_n1_identity_when_r_inner_equals_r_outer():
    """Identity oracle: ``r_inner == r_outer`` -> propagator is
    ``eye(6)`` to floating-point precision."""
    p = _typical_g_prime_b1_layer_params()
    P = _layer_propagator_n1(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=0.105,
        r_outer=0.105,
    )
    np.testing.assert_array_equal(P, np.eye(6))


def test_layer_propagator_n1_round_trip_preserves_state_vector():
    """Round-trip oracle via state-vector identity: applying
    ``P(a|b) @ P(b|a)`` to a physical state vector ``v`` returns
    ``v`` to ``rtol=1e-10``. State-vector phrasing avoids the
    spurious ~1e-6 off-diagonals from disparate-magnitude rows
    (displacement ~ O(1) vs stress ~ O(mu) ~ O(1e10)) per the
    G.b.2 lesson."""
    p = _typical_g_prime_b1_layer_params()
    a = 0.1
    b = a + 0.005
    P_b_from_a = _layer_propagator_n1(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=a,
        r_outer=b,
    )
    P_a_from_b = _layer_propagator_n1(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=b,
        r_outer=a,
    )
    mu = p["rho"] * p["vs"] ** 2
    # Physical state vector (3 displacement components ~ O(1),
    # 3 stress components ~ O(mu)).
    v = np.array([1.0, 2.0, 1.5, 3.0 * mu, 4.0 * mu, 2.5 * mu])
    v_round = P_a_from_b @ (P_b_from_a @ v)
    np.testing.assert_allclose(v_round, v, rtol=1.0e-10)
    v_round_other = P_b_from_a @ (P_a_from_b @ v)
    np.testing.assert_allclose(v_round_other, v, rtol=1.0e-10)


def test_layer_propagator_n1_composition_law():
    """Composition oracle: ``P(r3|r1) ~ P(r3|r2) @ P(r2|r1)`` for
    any intermediate ``r2 in (r1, r3)``. The propagator-group law
    in the radial coordinate."""
    p = _typical_g_prime_b1_layer_params()
    r1, r2, r3 = 0.1, 0.105, 0.115
    P_3_from_1 = _layer_propagator_n1(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=r1,
        r_outer=r3,
    )
    P_2_from_1 = _layer_propagator_n1(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=r1,
        r_outer=r2,
    )
    P_3_from_2 = _layer_propagator_n1(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=r2,
        r_outer=r3,
    )
    np.testing.assert_allclose(P_3_from_1, P_3_from_2 @ P_2_from_1, atol=1.0e-10)


def test_layer_propagator_n1_state_vector_continuity():
    """End-to-end state-vector check: pick an arbitrary amplitude
    vector ``c``; compute ``v(r1) = E(r1) c`` and apply
    ``P(r2|r1)`` to get ``v(r2)``; verify the result matches
    ``E(r2) c`` directly. Strongest single-test oracle for the
    G'.b.1 + G'.b.2 chain combined."""
    p = _typical_g_prime_b1_layer_params()
    r1, r2 = 0.1, 0.115
    E_r1 = _layer_e_matrix_n1(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=r1,
    )
    E_r2 = _layer_e_matrix_n1(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=r2,
    )
    P = _layer_propagator_n1(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=r1,
        r_outer=r2,
    )
    # Arbitrary amplitude vector (6 components for n=1).
    c = np.array([1.3, -0.7, 2.1, 0.4, -1.1, 0.6])
    v_r1 = E_r1 @ c
    v_r2_via_P = P @ v_r1
    v_r2_direct = E_r2 @ c
    np.testing.assert_allclose(v_r2_via_P, v_r2_direct, rtol=1.0e-10)


def test_layer_propagator_n1_returns_nan_below_bound_floor():
    """Below the layer's bound floor, ``E(r)`` is NaN-filled; the
    propagator inherits the NaN."""
    omega = 2.0 * np.pi * 5000.0
    vp, vs, rho = 2500.0, 1100.0, 2100.0
    kz = omega / vs * 0.5
    with np.errstate(invalid="ignore"):
        P = _layer_propagator_n1(
            kz=kz,
            omega=omega,
            vp=vp,
            vs=vs,
            rho=rho,
            r_inner=0.1,
            r_outer=0.105,
        )
    assert np.all(np.isnan(P))


# =====================================================================
# Plan item G'.c -- stacked modal determinant at n=1 (10x10)
# =====================================================================
#
# Tests anchor on the N=1 collapse to F.2 (numerical equality at
# rtol=1e-10) as the floating-point oracle, plus a few oracles
# that exercise the N >= 2 propagator chain.


def _typical_g_prime_c_params():
    """Slow-formation cased-hole fixture for G'.c tests. Keeps
    the flexural root in the bound regime; layers harder than
    formation V_S = 800."""
    return dict(
        vp=2200.0,
        vs=800.0,
        rho=2200.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )


def test_modal_determinant_n1_cased_N1_matches_F2_off_root():
    """N=1 floating-point oracle: at any (kz, omega) in the bound
    regime away from the flexural root, G'.c's determinant matches
    F.2's ``_modal_determinant_n1_layered`` to ``rtol=1e-10`` (no
    extra scale factor; same column packing). The propagator chain
    at N=1 reduces ``P_1 @ E_1(a)`` to ``E_1(b)``, exactly the F.2
    form.

    Strongest pinning of the G'.c assembly against the F.2 row-
    builder transcription that has shipped through F.2.d."""
    p = _typical_g_prime_c_params()
    layer = BoreholeLayer(vp=2500.0, vs=1100.0, rho=2100.0, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    # kz safely above the formation V_S bound floor (slowest in the stack).
    kz = omega / p["vs"] * 1.05
    det_F2 = _modal_determinant_n1_layered(
        kz,
        omega,
        p["vp"],
        p["vs"],
        p["rho"],
        p["vf"],
        p["rho_f"],
        p["a"],
        layer=layer,
    )
    det_Gp_c = _modal_determinant_n1_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(layer,),
    )
    assert det_Gp_c == pytest.approx(det_F2, rel=1.0e-10)


def test_modal_determinant_n1_cased_N1_vanishes_at_F2_brentq_root():
    """N=1 brentq-root oracle: at the flexural root recovered by
    ``flexural_dispersion_layered(layers=(layer,))``, G'.c's
    determinant is many orders of magnitude smaller than its
    value 0.5% off the root."""
    p = _typical_g_prime_c_params()
    layer = BoreholeLayer(vp=2500.0, vs=1100.0, rho=2100.0, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0

    res = flexural_dispersion_layered(
        np.array([5000.0]),
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(layer,),
    )
    kz_root = float(res.slowness[0]) * omega
    det_at = _modal_determinant_n1_cased(
        kz_root,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(layer,),
    )
    det_off = _modal_determinant_n1_cased(
        kz_root * 1.005,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(layer,),
    )
    assert abs(det_at) < abs(det_off) * 1.0e-6


def test_modal_determinant_n1_cased_returns_nan_below_bound_floor():
    """``kz`` below the slowest-shear bound floor -> at least one
    radial wavenumber goes imaginary -> NaN. brentq-safe
    propagation."""
    p = _typical_g_prime_c_params()
    layer = BoreholeLayer(vp=2500.0, vs=1100.0, rho=2100.0, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / p["vs"] * 0.5  # well below bound floor
    with np.errstate(invalid="ignore"):
        det = _modal_determinant_n1_cased(
            kz,
            omega,
            vp=p["vp"],
            vs=p["vs"],
            rho=p["rho"],
            vf=p["vf"],
            rho_f=p["rho_f"],
            a=p["a"],
            layers=(layer,),
        )
    assert np.isnan(det)


def test_modal_determinant_n1_cased_two_identical_layers_equals_one_double_thickness():
    """Group-law oracle: two contiguous identical layers (L, L)
    of thickness ``h`` each compose to a single layer of
    thickness ``2h``. Direct test of the propagator chain."""
    p = _typical_g_prime_c_params()
    omega = 2.0 * np.pi * 5000.0
    kz = omega / p["vs"] * 1.05
    L_double = BoreholeLayer(vp=2500.0, vs=1100.0, rho=2100.0, thickness=0.01)
    L_half = BoreholeLayer(vp=2500.0, vs=1100.0, rho=2100.0, thickness=0.005)
    det_single = _modal_determinant_n1_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(L_double,),
    )
    det_split = _modal_determinant_n1_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(L_half, L_half),
    )
    # Same E_1(a), identical propagator chain composes through the
    # same total thickness 2h.
    assert det_single == pytest.approx(det_split, rel=1.0e-10)


def test_modal_determinant_n1_cased_order_matters_at_N2():
    """Physical sanity: with two distinct layers ``(L_a, L_b)``,
    swapping the order to ``(L_b, L_a)`` produces a different
    determinant -- inside-out layer ordering is a physical
    parameter."""
    p = _typical_g_prime_c_params()
    omega = 2.0 * np.pi * 5000.0
    L_a = BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01)  # casing
    L_b = BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.01)  # cement
    # kz safely above the slowest-shear bound floor. Slowest in
    # this stack: cement V_S = 1300 > formation V_S = 800.
    kz = omega / min(L_a.vs, L_b.vs, p["vs"], p["vf"]) * 1.05
    det_ab = _modal_determinant_n1_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(L_a, L_b),
    )
    det_ba = _modal_determinant_n1_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(L_b, L_a),
    )
    assert np.isfinite(det_ab) and np.isfinite(det_ba)
    rel_diff = abs(det_ab - det_ba) / max(abs(det_ab), abs(det_ba))
    assert rel_diff > 0.01


def test_modal_determinant_n1_cased_N2_runs_smoke():
    """Smoke for the cased-hole flexural N=2 path: typical casing
    + cement geometry produces a finite real determinant at a
    representative bound-regime ``kz``."""
    p = _typical_g_prime_c_params()
    omega = 2.0 * np.pi * 5000.0
    casing = BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01)
    cement = BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.05)
    kz = omega / min(casing.vs, cement.vs, p["vs"], p["vf"]) * 1.05
    det = _modal_determinant_n1_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(casing, cement),
    )
    assert np.isfinite(det)
    assert isinstance(det, float)


# =====================================================================
# Plan item G'.d -- public-API hook for cased-hole flexural
# =====================================================================
#
# Replaces the G'.0 ``len(layers) > 1 -> NotImplementedError`` raise
# with a brentq loop on ``_modal_determinant_n1_cased`` (G'.c).
# Tests anchor on multi-layer regression / smoothness, plus the
# slow-formation per-layer constraint enforced by
# ``_validate_flexural_layers_stacked``.


def _typical_g_prime_d_cased_geometry():
    """Cased-hole fixture for G'.d tests. Casing + cement layers
    both faster in shear than the formation; the per-layer
    constraint (validated by ``_validate_flexural_layers_stacked``)
    holds.

    The formation is FAST (V_S = 2600 m/s). It used to be slow
    (V_S = 800 m/s), which the roadmap-A.8 correction put out of
    reach: with a steel casing the dipole mode sits just above a
    slow formation's shear speed (measured 830-880 m/s across
    3-9 kHz for the old parameters), so it is leaky in the
    formation and the real-valued bound-regime determinant has no
    root for it. Before A.8 the defective SV column produced a
    spurious bound root there, rising with frequency -- backwards
    for a flexural mode -- which is what these tests were pinning.
    See ``test_cased_slow_formation_dipole_is_leaky_not_bound``.

    The fast-formation stack routes through the complex leaky
    marcher instead and gives a proper cased dipole curve,
    2600 -> 1506 m/s over 2-6.5 kHz."""
    return dict(
        casing=BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01),
        cement=BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.05),
        vp=4500.0,
        vs=2600.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )


def _bound_invaded_geometry():
    """Slow-formation fixture with an annulus soft enough that the
    n >= 1 modes stay inside the bound regime.

    This is the invaded-zone regime -- the one Schmitt & Cheng
    figure 15 validates fwap against -- rather than the cased-hole
    regime. After the roadmap-A.8 correction the real-valued
    layered determinant only sees modes slower than every shear
    speed in the stack, which a stiff annulus pushes past; a
    lightly-altered annulus keeps them bound. Flexural coverage is
    45/45 and screw 36/45 over 1-12 kHz for one, two and three
    layers."""
    return dict(
        inner=BoreholeLayer(vp=1980.0, vs=900.0, rho=2300.0, thickness=0.01),
        outer=BoreholeLayer(vp=1804.0, vs=820.0, rho=2000.0, thickness=0.04),
        vp=2200.0,
        vs=800.0,
        rho=2200.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )


def test_flexural_dispersion_layered_N1_path_unchanged_after_G_prime_d():
    """Regression: the existing N=1 dispatch (F.2 hand-coded path)
    remains functional after the G'.d multi-layer wiring. Smoke
    test on a slow-formation single-layer setup."""
    layer = BoreholeLayer(vp=2500.0, vs=1100.0, rho=2100.0, thickness=0.005)
    f = np.linspace(2000.0, 6000.0, 4)
    res = flexural_dispersion_layered(
        f,
        vp=SLOW_VP,
        vs=SLOW_VS,
        rho=SLOW_RHO,
        vf=SLOW_VF,
        rho_f=SLOW_RHO_F,
        a=SLOW_A,
        layers=(layer,),
    )
    assert res.name == "flexural"
    assert res.azimuthal_order == 1
    finite = np.isfinite(res.slowness)
    assert finite.any()


def test_flexural_dispersion_layered_N2_runs_smoke():
    """G'.d two-layer regression: typical casing + cement
    geometry produces a finite, smoothly-dispersive flexural
    slowness curve across the band where the cased dipole mode is
    resolvable (2-6.5 kHz; above that it has passed V_f)."""
    g = _typical_g_prime_d_cased_geometry()
    f = np.linspace(2000.0, 6500.0, 8)
    res = flexural_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], g["cement"]),
    )
    finite = np.isfinite(res.slowness)
    assert finite.sum() >= len(f) - 1  # allow at most one cutoff miss
    assert res.name == "flexural"
    assert res.azimuthal_order == 1
    np.testing.assert_array_equal(res.freq, f)
    # Smoothness fence on contiguous finite values: a coarse sanity
    # check, not a tight oracle.
    s_finite = res.slowness[finite]
    rel_steps = np.abs(np.diff(s_finite)) / s_finite[:-1]
    assert np.all(rel_steps < 0.50)


def test_flexural_dispersion_layered_N2_returns_borehole_mode():
    """``BoreholeMode`` return-type contract on the multi-layer
    flexural dispatch (G'.d). ``attenuation_per_meter is None``
    confirms the slow-formation bound mode."""
    g = _typical_g_prime_d_cased_geometry()
    f = np.linspace(3000.0, 6000.0, 3)
    res = flexural_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], g["cement"]),
    )
    assert isinstance(res, BoreholeMode)
    assert res.attenuation_per_meter is None


def test_flexural_dispersion_layered_N2_layer_permutation_changes_slowness():
    """Casing-inside-cement and cement-inside-casing produce
    distinct flexural slowness curves -- inside-out layer
    ordering is a physical parameter."""
    g = _typical_g_prime_d_cased_geometry()
    f = np.array([5000.0])
    res_cs = flexural_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], g["cement"]),
    )
    res_sc = flexural_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["cement"], g["casing"]),
    )
    assert np.all(np.isfinite(res_cs.slowness))
    assert np.all(np.isfinite(res_sc.slowness))
    rel_diff = abs(res_cs.slowness[0] - res_sc.slowness[0]) / res_cs.slowness[0]
    assert rel_diff > 0.001


def test_flexural_dispersion_layered_N2_collapse_to_N1_via_thin_outer_layer():
    """Two-layer-collapse oracle: with a vanishingly-thin OUTER
    layer that has the formation's parameters, the G'.d two-
    layer slowness should match the F.2 single-layer answer with
    just the inner layer."""
    layer1 = BoreholeLayer(vp=2500.0, vs=1100.0, rho=2100.0, thickness=0.005)
    # Outer "layer" has formation properties + tiny thickness;
    # equivalent to no outer layer at all. layer.vs == vs satisfies
    # the slow-formation per-layer constraint (>=).
    layer2_trivial = BoreholeLayer(
        vp=SLOW_VP,
        vs=SLOW_VS,
        rho=SLOW_RHO,
        thickness=1.0e-5,
    )
    f = np.array([5000.0])
    res_one = flexural_dispersion_layered(
        f,
        vp=SLOW_VP,
        vs=SLOW_VS,
        rho=SLOW_RHO,
        vf=SLOW_VF,
        rho_f=SLOW_RHO_F,
        a=SLOW_A,
        layers=(layer1,),
    )
    res_two = flexural_dispersion_layered(
        f,
        vp=SLOW_VP,
        vs=SLOW_VS,
        rho=SLOW_RHO,
        vf=SLOW_VF,
        rho_f=SLOW_RHO_F,
        a=SLOW_A,
        layers=(layer1, layer2_trivial),
    )
    assert res_two.slowness[0] == pytest.approx(
        res_one.slowness[0],
        rel=1.0e-4,
    )


def test_flexural_dispersion_layered_three_layer_runs_smoke():
    """Smoke test for N=3 (casing + cement + mudcake): the
    propagator chain extends past N=2 cleanly."""
    g = _typical_g_prime_d_cased_geometry()
    mudcake = BoreholeLayer(vp=2000.0, vs=1100.0, rho=1700.0, thickness=0.003)
    f = np.linspace(4000.0, 6500.0, 4)
    res = flexural_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], g["cement"], mudcake),
    )
    finite = np.isfinite(res.slowness)
    assert finite.any()


def test_flexural_dispersion_layered_N2_rejects_softer_layer():
    """The G'.d brentq path enforces ``layer.vs >= vs`` per layer
    via ``_validate_flexural_layers_stacked``. A softer layer
    triggers ``ValueError`` rather than NaN slownesses or a
    silent unbound-mode failure.

    The constraint is a slow-formation one, so this uses the
    bound-regime (invaded-zone) fixture rather than the cased
    fixture, whose formation is fast."""
    g = _bound_invaded_geometry()
    soft_cement = BoreholeLayer(
        vp=1500.0,
        vs=600.0,
        rho=1700.0,
        thickness=0.05,
    )  # vs = 600 < formation vs = 800 -> reject
    with pytest.raises(ValueError, match=r"layer\.vs"):
        flexural_dispersion_layered(
            np.array([5000.0]),
            vp=g["vp"],
            vs=g["vs"],
            rho=g["rho"],
            vf=g["vf"],
            rho_f=g["rho_f"],
            a=g["a"],
            layers=(g["inner"], soft_cement),
        )


# =====================================================================
# Plan item G'.e -- validation hardening for the cased-hole flexural
# =====================================================================
#
# Mirror of G.e for the propagator-matrix flexural path.


def test_modal_determinant_n1_cased_vanishes_at_converged_root_multi_freq():
    """Self-consistency at every brentq-converged ``k_z`` from
    ``flexural_dispersion_layered`` (cased-hole, two-layer): the
    propagator-matrix determinant is at least 6 orders of
    magnitude smaller than its value at ``kz * 1.005``. Multi-
    frequency sharper than G'.d's single-frequency oracles.

    Uses the bound-regime (invaded-zone) fixture: this is the
    real-valued determinant, so it only has roots where the mode
    is slower than every shear speed in the stack."""
    g = _bound_invaded_geometry()
    f = np.linspace(4000.0, 12000.0, 6)
    res = flexural_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["inner"], g["outer"]),
    )
    finite = np.isfinite(res.slowness)
    assert finite.any()
    for i, fi in enumerate(f):
        if not finite[i]:
            continue
        omega = 2.0 * np.pi * float(fi)
        kz_root = float(res.slowness[i]) * omega
        det_at = _modal_determinant_n1_cased(
            kz_root,
            omega,
            vp=g["vp"],
            vs=g["vs"],
            rho=g["rho"],
            vf=g["vf"],
            rho_f=g["rho_f"],
            a=g["a"],
            layers=(g["inner"], g["outer"]),
        )
        det_off = _modal_determinant_n1_cased(
            kz_root * 1.005,
            omega,
            vp=g["vp"],
            vs=g["vs"],
            rho=g["rho"],
            vf=g["vf"],
            rho_f=g["rho_f"],
            a=g["a"],
            layers=(g["inner"], g["outer"]),
        )
        assert abs(det_at) < abs(det_off) * 1.0e-6, (
            f"f={fi:.1f}: |det_at|={abs(det_at):.3e} not << "
            f"|det_off|={abs(det_off):.3e}"
        )


def test_flexural_dispersion_layered_thin_inner_layer_collapses_to_outer_only():
    """Two-layer-collapse oracle: with a vanishingly-thin INNER
    layer that has the outer layer's parameters, the G'.d two-
    layer slowness should match the F.2 single-layer answer with
    just the outer layer.

    Uses an outer layer just stiffer than the formation, which
    keeps the mode inside the bound regime the real-valued layered
    determinant can see. (It used to use ``vs = 2700 m/s``; after
    the roadmap-A.8 correction an annulus that stiff pushes the
    dipole mode past the formation shear speed, where it is leaky
    -- see ``_bound_invaded_geometry``.)"""
    vp_form, vs_form, rho_form = 2200.0, 800.0, 2200.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    outer = BoreholeLayer(vp=1804.0, vs=820.0, rho=2400.0, thickness=0.05)
    inner_trivial = BoreholeLayer(
        vp=outer.vp,
        vs=outer.vs,
        rho=outer.rho,
        thickness=1.0e-5,
    )
    f = np.array([5000.0])
    res_one = flexural_dispersion_layered(
        f,
        vp=vp_form,
        vs=vs_form,
        rho=rho_form,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(outer,),
    )
    res_two = flexural_dispersion_layered(
        f,
        vp=vp_form,
        vs=vs_form,
        rho=rho_form,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(inner_trivial, outer),
    )
    assert np.isfinite(res_one.slowness[0])
    assert np.isfinite(res_two.slowness[0])
    assert res_two.slowness[0] == pytest.approx(
        res_one.slowness[0],
        rel=1.0e-3,
    )


def test_flexural_dispersion_layered_two_formation_layers_collapse_to_unlayered():
    """Master-plan G' validation bullet: with both annular
    layers carrying formation properties (``layer.vs == vs``),
    the multi-layer flexural slowness matches the unlayered
    ``flexural_dispersion`` answer.

    Strongest pinning of the G'.d brentq pipeline against the
    pre-G' unlayered baseline."""
    vp_form, vs_form, rho_form = 2200.0, 800.0, 2200.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = np.linspace(3000.0, 8000.0, 5)
    L1 = BoreholeLayer(
        vp=vp_form,
        vs=vs_form,
        rho=rho_form,
        thickness=0.01,
    )
    L2 = BoreholeLayer(
        vp=vp_form,
        vs=vs_form,
        rho=rho_form,
        thickness=0.05,
    )
    res_unlayered = flexural_dispersion(
        f,
        vp=vp_form,
        vs=vs_form,
        rho=rho_form,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    res_cased = flexural_dispersion_layered(
        f,
        vp=vp_form,
        vs=vs_form,
        rho=rho_form,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(L1, L2),
    )
    np.testing.assert_allclose(
        res_cased.slowness,
        res_unlayered.slowness,
        rtol=1.0e-6,
        equal_nan=True,
    )


def test_flexural_dispersion_layered_cement_stiffness_sensitivity():
    """Cement-bond physics for the dipole sonic: stiffer vs
    softer cement produce distinct flexural slowness curves.
    Direct test of the qualitative cement-stiffness sensitivity
    that the cased-hole propagator captures and the unlayered
    ``flexural_dispersion`` does not.

    Direction (from the typical cased-hole fixture at 5 kHz):
    stiffer cement -> SMALLER slowness (faster wave). Same
    direction as the Stoneley cement-bond signature (commit
    9df7a78), and monotone across V_S = 1100 .. 2600 m/s of
    cement. Pinned here as a regression target with a loose
    quantitative tolerance."""
    g = _typical_g_prime_d_cased_geometry()
    f = np.array([5000.0])
    casing = g["casing"]
    cement_stiff = BoreholeLayer(
        vp=2500.0,
        vs=2000.0,
        rho=2000.0,
        thickness=g["cement"].thickness,
    )
    cement_soft = BoreholeLayer(
        vp=1900.0,
        vs=1100.0,
        rho=1700.0,
        thickness=g["cement"].thickness,
    )
    res_stiff = flexural_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(casing, cement_stiff),
    )
    res_soft = flexural_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(casing, cement_soft),
    )
    assert np.isfinite(res_stiff.slowness[0])
    assert np.isfinite(res_soft.slowness[0])
    # Distinct curves: at least 1% relative difference.
    rel_diff = abs(res_stiff.slowness[0] - res_soft.slowness[0]) / res_soft.slowness[0]
    assert rel_diff > 0.01
    # Empirical direction: stiffer cement -> smaller slowness.
    assert res_stiff.slowness[0] < res_soft.slowness[0]


# =====================================================================
# Plan item G''.0 -- public-API foundation for cased-hole quadrupole
# =====================================================================
#
# Introduces ``quadrupole_dispersion_layered`` from scratch: with
# ``layers=()`` it dispatches to ``quadrupole_dispersion``; with
# any non-empty layer stack it raises NotImplementedError pointing
# at G''.c / G''.d (the propagator-matrix scaffolding that ships
# in subsequent sub-units). Validation rules cover the slow-
# formation per-layer constraint and a fast-formation NIE for
# the deferred complex-determinant follow-up.


def test_quadrupole_dispersion_layered_layers_empty_dispatches_to_unlayered():
    """``layers=()`` -> bit-equivalent to
    ``quadrupole_dispersion``. Floating-point regression oracle
    for plan G''."""
    from fwap.cylindrical_solver import (
        quadrupole_dispersion,
        quadrupole_dispersion_layered,
    )

    f = np.linspace(8000.0, 25000.0, 5)
    # LWD-quadrupole-relevant slow formation.
    vp, vs, rho = 2200.0, 800.0, 2200.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    res_unl = quadrupole_dispersion(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    res_lyr = quadrupole_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(),
    )
    np.testing.assert_array_equal(res_lyr.slowness, res_unl.slowness)
    np.testing.assert_array_equal(res_lyr.freq, res_unl.freq)
    assert res_lyr.name == "quadrupole"
    assert res_lyr.azimuthal_order == 2


def test_quadrupole_dispersion_layered_returns_borehole_mode_for_unlayered():
    """``BoreholeMode`` return-type contract on the unlayered
    dispatch. Type and azimuthal_order are pinned for every
    public path; the actual slowness numerics are covered by
    the existing ``quadrupole_dispersion`` tests."""
    from fwap.cylindrical_solver import quadrupole_dispersion_layered

    f = np.linspace(8000.0, 15000.0, 3)
    res = quadrupole_dispersion_layered(
        f,
        vp=2200.0,
        vs=800.0,
        rho=2200.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    assert isinstance(res, BoreholeMode)
    assert res.name == "quadrupole"
    assert res.azimuthal_order == 2
    np.testing.assert_array_equal(res.freq, f)


def test_quadrupole_dispersion_layered_rejects_softer_layer_multi_only():
    """The per-layer slow-formation constraint ``layer.vs >= vs``
    applies to the **multi-layer** path only, via the G'.0
    ``_validate_flexural_layers_stacked`` helper -- the same rule, and
    the same layer-count condition, as ``flexural_dispersion_layered``.

    This test used to assert that a *single* softer layer was rejected.
    That was the behaviour of the code but not of its docstring, which
    has always said "(multi-layer only)", and it made every invaded zone
    unrepresentable at n=2 while the identical model was accepted at
    n=1. Corrected against Schmitt & Cheng figure 15(b); see the A.6
    block later in this file."""
    from fwap.cylindrical_solver import quadrupole_dispersion_layered

    soft = BoreholeLayer(vp=1500.0, vs=600.0, rho=1700.0, thickness=0.05)
    base = dict(vp=2200.0, vs=800.0, rho=2200.0, vf=1500.0, rho_f=1000.0, a=0.1)

    # one soft layer: accepted (an invaded zone is exactly this case)
    res = quadrupole_dispersion_layered(np.array([10000.0]), **base, layers=(soft,))
    assert res.slowness.shape == (1,)

    # two: the multi-layer guard still fires, with the offending index
    with pytest.raises(ValueError, match=r"layer\.vs"):
        quadrupole_dispersion_layered(
            np.array([10000.0]),
            **base,
            layers=(
                soft,
                BoreholeLayer(vp=1500.0, vs=650.0, rho=1700.0, thickness=0.05),
            ),
        )


def test_quadrupole_dispersion_layered_fast_formation_with_layer_dispatches_to_complex_path():
    """Fast-formation cased-hole quadrupole (``V_S > V_f`` with
    a non-empty layer) dispatches to the complex-determinant
    path via ``_modal_determinant_n2_cased_complex`` and
    ``_quadrupole_dispersion_fast_formation_layered``. The
    earlier ``NotImplementedError`` raise from G''.0 is gone --
    fast-formation cased quadrupole is now a supported regime."""
    from fwap.cylindrical_solver import (
        BoreholeMode,
        quadrupole_dispersion_layered,
    )

    layer = BoreholeLayer(vp=4000.0, vs=3300.0, rho=2500.0, thickness=0.005)
    # Fast formation: V_S = 3100 > V_f = 1500.
    res = quadrupole_dispersion_layered(
        np.array([10000.0]),
        vp=5500.0,
        vs=3100.0,
        rho=2500.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
        layers=(layer,),
    )
    assert isinstance(res, BoreholeMode)
    assert res.name == "quadrupole"
    assert res.azimuthal_order == 2
    # Bound-mode result: real-valued slowness (or NaN if outside
    # the geometric cutoff window). Either way no exception.
    assert res.slowness.dtype == np.float64


def test_quadrupole_dispersion_layered_rejects_invalid_inputs():
    """Standard input validation: positivity, ``vp > vs``, and
    strictly-positive frequencies."""
    from fwap.cylindrical_solver import quadrupole_dispersion_layered

    base = dict(
        vp=2200.0,
        vs=800.0,
        rho=2200.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    layer = BoreholeLayer(vp=2500.0, vs=1100.0, rho=2100.0, thickness=0.005)
    f = np.array([10000.0])
    # Non-positive vp.
    with pytest.raises(ValueError, match="vp, vs, rho must all be positive"):
        quadrupole_dispersion_layered(
            f,
            **{**base, "vp": 0.0},
            layers=(layer,),
        )
    # vp <= vs.
    with pytest.raises(ValueError, match=r"vp > vs"):
        quadrupole_dispersion_layered(
            f,
            **{**base, "vp": 700.0},
            layers=(layer,),
        )
    # Non-positive freq.
    with pytest.raises(ValueError, match="freq must be strictly positive"):
        quadrupole_dispersion_layered(
            np.array([0.0]),
            **base,
            layers=(layer,),
        )


# =====================================================================
# Plan item G''.b.1 -- 6x6 mode-amplitude-to-state-vector matrix at n=2
# =====================================================================
#
# Without an F.3-equivalent oracle, the per-element pinning is
# weaker than G.b.1 / G'.b.1. The structural oracles (sparsity,
# finiteness, NaN propagation) plus a layer=formation cross-
# check vs the K-flavour entries transcribed from the existing
# ``_modal_determinant_n2`` docstring catch transcription errors
# at the entry level; the determinant-level checks in G''.c /
# G''.d / G''.e anchor correctness via root-match identities.


def _typical_g_pp_b1_layer_params():
    """Slow-formation harder-than-formation layer fixture for
    G''.b.1 / G''.b.2 tests."""
    return dict(
        vp=2500.0,
        vs=1100.0,
        rho=2100.0,
        kz=2.0 * np.pi * 5000.0 / 800.0,
        omega=2.0 * np.pi * 5000.0,
    )


def test_layer_e_matrix_n2_real_in_bound_regime():
    """All 36 entries are finite real in the bound regime
    post-rescale."""
    p = _typical_g_pp_b1_layer_params()
    E = _layer_e_matrix_n2(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=0.1,
    )
    assert np.all(np.isfinite(E))
    assert E.dtype == np.float64


def test_layer_e_matrix_n2_returns_nan_below_bound_floor():
    """Below the bound floor (``kz < omega / V_S``), at least one
    of ``p^2``, ``s^2`` is negative -- the helper returns NaN-
    filled."""
    omega = 2.0 * np.pi * 5000.0
    vp, vs, rho = 2500.0, 1100.0, 2100.0
    kz = omega / vs * 0.5
    with np.errstate(invalid="ignore"):
        E = _layer_e_matrix_n2(
            kz=kz,
            omega=omega,
            vp=vp,
            vs=vs,
            rho=rho,
            r=0.1,
        )
    assert np.all(np.isnan(E))


def test_layer_e_matrix_n2_determinant_nonzero_in_bound_regime():
    """Precondition for the G''.b.2 propagator inverse."""
    p = _typical_g_pp_b1_layer_params()
    E = _layer_e_matrix_n2(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=0.1,
    )
    det = float(np.linalg.det(E))
    assert np.isfinite(det)
    assert abs(det) > 0.0


def test_layer_e_matrix_n2_sparsity_pattern():
    """Pin the known-zero entries of E_n2(r), and the ones that are
    NOT zero after the roadmap-A.8 correction.

    * Row 1 (``u_z``) cols 4, 5 (``D_I``, ``D_K``) ARE zero: the SH
      potential ``psi_z`` doesn't contribute to ``u_z``.
    * Row 2 (``u_theta``) cols 2, 3 (``C_I``, ``C_K``) are NOT
      zero: the Hansen SV field carries ``u_theta = i k_z (n/r)
      chi``, non-zero for every n >= 1. See the n=1 twin of this
      test for the full note."""
    p = _typical_g_pp_b1_layer_params()
    E = _layer_e_matrix_n2(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=0.1,
    )
    # u_z row, D cols: genuinely zero.
    assert E[1, 4] == 0.0
    assert E[1, 5] == 0.0
    # u_theta row, C cols: non-zero at n >= 1 (roadmap A.8).
    assert E[2, 2] != 0.0
    assert E[2, 3] != 0.0


def test_layer_e_matrix_n2_K_flavour_matches_modal_determinant_n2_docstring():
    """Cross-check vs the K-flavour entries transcribed from
    ``_modal_determinant_n2``'s docstring at r=a:

    * Row 1 (BC1: f - m, layer cols negated) -> u_r row of E,
      K-flavour cols (B_K, C_K, D_K).
    * Row 2 (BC2: -m, layer cols negated) -> sigma_rr row.
    * Row 3 (BC3: m=0, layer cols positive) -> sigma_rtheta row.
    * Row 4 (BC4: m=0, layer cols positive) -> sigma_rz row.

    The unlayered formula from the docstring at radius ``r=a``
    becomes (for the layer's E entries) the formation parameters
    evaluated at ``r=a`` (since the formation half-space's K-only
    contribution to the modal matrix uses the same Bessel ansatz
    as the layer's K-flavour part at this radius)."""
    from scipy import special as _special

    vp, vs, rho = 2200.0, 800.0, 2200.0
    omega = 2.0 * np.pi * 5000.0
    kz = omega / vs * 1.05  # bound regime
    a = 0.1

    E = _layer_e_matrix_n2(
        kz=kz,
        omega=omega,
        vp=vp,
        vs=vs,
        rho=rho,
        r=a,
    )

    # Local Bessel evaluations matching the docstring's notation.
    p_ = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s_ = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    K1pa = float(_special.kv(1, p_ * a))
    K2pa = float(_special.kv(2, p_ * a))
    K1sa = float(_special.kv(1, s_ * a))
    K2sa = float(_special.kv(2, s_ * a))
    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    # Row 1 (BC1) layer cols are -E[u_r row, K-flavour cols]:
    # B_K: +p K_1 + 2 K_2/a    (docstring)
    # C_K: +kz [s K_1 + 2 K_2/a]   (docstring, post-A.8)
    # D_K: -2 K_2/a            (docstring)
    assert -E[0, 1] == pytest.approx(+p_ * K1pa + 2.0 * K2pa / a, rel=1e-12)
    assert -E[0, 3] == pytest.approx(+kz * (s_ * K1sa + 2.0 * K2sa / a), rel=1e-12)
    assert -E[0, 5] == pytest.approx(-2.0 * K2sa / a, rel=1e-12)

    # Row 2 (BC2) layer cols negated:
    # B_K: -mu * [(2 kz^2 - kS^2) K_2 + 2 p K_1/a + 12 K_2/a^2]
    # C_K: -2 mu kz * [s^2 K_2 + s K_1/a + 6 K_2/a^2]
    # D_K: +4 mu * [s K_1/a + 3 K_2/a^2]
    assert -E[3, 1] == pytest.approx(
        -mu * (two_kz2_minus_kS2 * K2pa + 2.0 * p_ * K1pa / a + 12.0 * K2pa / (a * a)),
        rel=1e-12,
    )
    assert -E[3, 3] == pytest.approx(
        -2.0 * mu * kz * (s_ * s_ * K2sa + s_ * K1sa / a + 6.0 * K2sa / (a * a)),
        rel=1e-12,
    )
    assert -E[3, 5] == pytest.approx(
        +4.0 * mu * (s_ * K1sa / a + 3.0 * K2sa / (a * a)),
        rel=1e-12,
    )

    # Row 3 (BC3) layer cols positive (no negation):
    # B_K: +4 mu * [p K_1/a + 3 K_2/a^2]
    # C_K: +4 mu kz * [s K_1/a + 3 K_2/a^2]
    # D_K: -mu * [(s^2 + 12/a^2) K_2 + 2 s K_1/a]
    assert E[5, 1] == pytest.approx(
        +4.0 * mu * (p_ * K1pa / a + 3.0 * K2pa / (a * a)),
        rel=1e-12,
    )
    assert E[5, 3] == pytest.approx(
        +4.0 * mu * kz * (s_ * K1sa / a + 3.0 * K2sa / (a * a)),
        rel=1e-12,
    )
    assert E[5, 5] == pytest.approx(
        -mu * ((s_ * s_ + 12.0 / (a * a)) * K2sa + 2.0 * s_ * K1sa / a),
        rel=1e-12,
    )

    # Row 4 (BC4) layer cols positive:
    # B_K: +2 mu kz * [p K_1 + 2 K_2/a]
    # C_K: +mu (2 kz^2 - kS^2) * [s K_1 + 2 K_2/a]
    # D_K: -2 mu kz K_2/a
    assert E[4, 1] == pytest.approx(
        +2.0 * mu * kz * (p_ * K1pa + 2.0 * K2pa / a),
        rel=1e-12,
    )
    assert E[4, 3] == pytest.approx(
        +mu * two_kz2_minus_kS2 * (s_ * K1sa + 2.0 * K2sa / a),
        rel=1e-12,
    )
    assert E[4, 5] == pytest.approx(-2.0 * mu * kz * K2sa / a, rel=1e-12)


def test_layer_e_matrix_n2_n2_factors_appear():
    """Pin the explicit n=2 factors: ``12 = 2 n (n+1)`` in the
    F_2 / r^2 term of the P column of sigma_rr, and ``6 = n (n+1)``
    in the same term of its SV column. Catches a transcription
    error where an n=1 formula was reused at n=2.

    The SV anchor moved with the roadmap-A.8 correction: the old
    SV column of sigma_rz carried a spurious ``3 = n^2 - 1`` term
    that the Hansen form does not produce."""
    from scipy import special as _special

    p = _typical_g_pp_b1_layer_params()
    a = 0.1
    E = _layer_e_matrix_n2(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=a,
    )
    # sigma_rr row (3) B_K col (1): expect (2 kz^2 - kS^2) K_2
    # + 2 p K_1/a + 12 K_2/a^2 (with leading +mu).
    p_ = float(np.sqrt(p["kz"] * p["kz"] - (p["omega"] / p["vp"]) ** 2))
    K1pa = float(_special.kv(1, p_ * a))
    K2pa = float(_special.kv(2, p_ * a))
    mu = p["rho"] * p["vs"] ** 2
    kS2 = (p["omega"] / p["vs"]) ** 2
    expected = mu * (
        (2.0 * p["kz"] ** 2 - kS2) * K2pa + 2.0 * p_ * K1pa / a + 12.0 * K2pa / (a * a)
    )
    assert E[3, 1] == pytest.approx(expected, rel=1e-12)

    # sigma_rr row (3) C_K col (3): expect +2 mu kz * [s^2 K_2
    # + s K_1/a + 6 K_2/a^2] (E carries the un-negated sigma_rr,
    # BC2 negates it); the 6 is n (n+1) at n = 2.
    s_ = float(np.sqrt(p["kz"] * p["kz"] - (p["omega"] / p["vs"]) ** 2))
    K1sa = float(_special.kv(1, s_ * a))
    K2sa = float(_special.kv(2, s_ * a))
    expected_sv = (
        2.0 * mu * p["kz"] * (s_ * s_ * K2sa + s_ * K1sa / a + 6.0 * K2sa / (a * a))
    )
    assert E[3, 3] == pytest.approx(expected_sv, rel=1e-12)


# =====================================================================
# Plan item G''.b.2 -- per-layer propagator P(r_outer | r_inner) at n=2
# =====================================================================
#
# Group-law oracles for ``_layer_propagator_n2`` (6x6 sister of
# ``_layer_propagator_n1``). Round-trip uses the state-vector
# form to avoid the ``cond(E) ~ mu`` issue called out in
# G.b.2 / G'.b.2.


def test_layer_propagator_n2_identity_when_r_inner_equals_r_outer():
    """Identity oracle: ``r_inner == r_outer`` -> propagator is
    ``eye(6)`` to floating-point precision."""
    p = _typical_g_pp_b1_layer_params()
    P = _layer_propagator_n2(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=0.105,
        r_outer=0.105,
    )
    np.testing.assert_array_equal(P, np.eye(6))


def test_layer_propagator_n2_round_trip_preserves_state_vector():
    """Round-trip oracle via state-vector identity at n=2:
    applying ``P(a|b) @ P(b|a)`` to a physical state vector
    ``v`` returns ``v`` to ``rtol=1e-10``. State-vector phrasing
    avoids the spurious ~1e-6 off-diagonals from disparate-
    magnitude rows (displacement ~ O(1) vs stress ~ O(mu) ~
    O(1e10)) per the G.b.2 / G'.b.2 lesson."""
    p = _typical_g_pp_b1_layer_params()
    a = 0.1
    b = a + 0.005
    P_b_from_a = _layer_propagator_n2(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=a,
        r_outer=b,
    )
    P_a_from_b = _layer_propagator_n2(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=b,
        r_outer=a,
    )
    mu = p["rho"] * p["vs"] ** 2
    v = np.array([1.0, 2.0, 1.5, 3.0 * mu, 4.0 * mu, 2.5 * mu])
    v_round = P_a_from_b @ (P_b_from_a @ v)
    np.testing.assert_allclose(v_round, v, rtol=1.0e-10)
    v_round_other = P_b_from_a @ (P_a_from_b @ v)
    np.testing.assert_allclose(v_round_other, v, rtol=1.0e-10)


def test_layer_propagator_n2_composition_law():
    """Composition oracle: ``P(r3|r1) ~ P(r3|r2) @ P(r2|r1)`` for
    any intermediate ``r2``. The propagator-group law in radius
    at n=2."""
    p = _typical_g_pp_b1_layer_params()
    r1, r2, r3 = 0.1, 0.105, 0.115
    P_3_from_1 = _layer_propagator_n2(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=r1,
        r_outer=r3,
    )
    P_2_from_1 = _layer_propagator_n2(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=r1,
        r_outer=r2,
    )
    P_3_from_2 = _layer_propagator_n2(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=r2,
        r_outer=r3,
    )
    np.testing.assert_allclose(P_3_from_1, P_3_from_2 @ P_2_from_1, atol=1.0e-10)


def test_layer_propagator_n2_state_vector_continuity():
    """End-to-end state-vector check: pick an arbitrary 6-amp
    vector ``c``; verify ``P(r2|r1) @ E_n2(r1) c == E_n2(r2) c``
    to ``rtol=1e-10``. Strongest single-test oracle for the
    G''.b.1 + G''.b.2 chain combined."""
    p = _typical_g_pp_b1_layer_params()
    r1, r2 = 0.1, 0.115
    E_r1 = _layer_e_matrix_n2(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=r1,
    )
    E_r2 = _layer_e_matrix_n2(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r=r2,
    )
    P = _layer_propagator_n2(
        kz=p["kz"],
        omega=p["omega"],
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        r_inner=r1,
        r_outer=r2,
    )
    c = np.array([1.3, -0.7, 2.1, 0.4, -1.1, 0.6])
    v_r1 = E_r1 @ c
    v_r2_via_P = P @ v_r1
    v_r2_direct = E_r2 @ c
    np.testing.assert_allclose(v_r2_via_P, v_r2_direct, rtol=1.0e-10)


def test_layer_propagator_n2_returns_nan_below_bound_floor():
    """Below the bound floor, ``E_n2(r)`` is NaN-filled; the
    propagator inherits the NaN."""
    omega = 2.0 * np.pi * 5000.0
    vp, vs, rho = 2500.0, 1100.0, 2100.0
    kz = omega / vs * 0.5
    with np.errstate(invalid="ignore"):
        P = _layer_propagator_n2(
            kz=kz,
            omega=omega,
            vp=vp,
            vs=vs,
            rho=rho,
            r_inner=0.1,
            r_outer=0.105,
        )
    assert np.all(np.isnan(P))


# =====================================================================
# Plan item G''.c -- 10x10 stacked modal determinant at n=2
# =====================================================================
#
# Tests for ``_modal_determinant_n2_cased``. With no F.3-equivalent
# hand-coded n=2 single-layer oracle, the strongest root-level
# oracle is the layer = formation collapse identity: at layer params
# equal to formation params, the brentq root in ``k_z`` of the 10x10
# cased determinant matches the unlayered 4x4
# ``_modal_determinant_n2`` root (per the substep G''.a.6 algebra:
# ``det(M_10) = det(E_form(b)) * det(M_4)``). The "N=0 dispatch to
# unlayered" check is covered by the existing G''.0 test
# ``test_quadrupole_dispersion_layered_layers_empty_dispatches_to_unlayered``
# (the dispatch lives in ``quadrupole_dispersion_layered``, not in
# ``_modal_determinant_n2_cased``), so this block ships 5 tests.


def _typical_g_pp_c_params():
    """Slow-formation cased-hole fixture for G''.c tests. Keeps
    the unlayered quadrupole root in the bound regime (V_S = 800
    < V_f = 1500); layers must satisfy the slow-formation
    constraint ``layer.vs >= formation.vs``."""
    return dict(
        vp=2200.0,
        vs=800.0,
        rho=2200.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )


def test_modal_determinant_n2_cased_N1_layer_eq_formation_vanishes_at_unlayered_root():
    """G''.a.6 collapse identity: at ``layer.{vp,vs,rho} ==
    formation.{vp,vs,rho}``, the cased determinant has a brentq
    root in ``k_z`` exactly where the unlayered
    ``_modal_determinant_n2`` does. Verified at the
    ``quadrupole_dispersion``-converged root: ``|det_at|`` is
    many orders of magnitude smaller than ``|det_off|`` 0.5%
    away.

    Strongest root-level oracle in the absence of an F.3
    per-element form."""
    from fwap.cylindrical_solver import (
        _modal_determinant_n2_cased,
        quadrupole_dispersion,
    )

    p = _typical_g_pp_c_params()
    # Layer = formation. Thickness arbitrary; G''.a.6 holds for any
    # ``b > a``.
    layer = BoreholeLayer(
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        thickness=0.01,
    )
    omega = 2.0 * np.pi * 5000.0
    res = quadrupole_dispersion(
        np.array([5000.0]),
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
    )
    assert np.isfinite(res.slowness[0]), (
        "fixture must put the unlayered quadrupole root in the bound "
        "regime so the collapse identity is exercisable"
    )
    kz_root = float(res.slowness[0]) * omega
    det_at = _modal_determinant_n2_cased(
        kz_root,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(layer,),
    )
    det_off = _modal_determinant_n2_cased(
        kz_root * 1.005,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(layer,),
    )
    # Guard against the unphysical case where both are tiny.
    assert abs(det_off) > 0.0
    assert abs(det_at) < abs(det_off) * 1.0e-6


def test_modal_determinant_n2_cased_returns_nan_below_bound_floor():
    """``kz`` below the slowest-shear bound floor -> at least one
    radial wavenumber goes imaginary -> NaN. brentq-safe
    propagation."""
    from fwap.cylindrical_solver import _modal_determinant_n2_cased

    p = _typical_g_pp_c_params()
    layer = BoreholeLayer(vp=2500.0, vs=1100.0, rho=2100.0, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / p["vs"] * 0.5  # well below bound floor
    with np.errstate(invalid="ignore"):
        det = _modal_determinant_n2_cased(
            kz,
            omega,
            vp=p["vp"],
            vs=p["vs"],
            rho=p["rho"],
            vf=p["vf"],
            rho_f=p["rho_f"],
            a=p["a"],
            layers=(layer,),
        )
    assert np.isnan(det)


def test_modal_determinant_n2_cased_two_identical_layers_equals_one_double_thickness():
    """Group-law oracle: two contiguous identical layers (L, L)
    of thickness ``h`` each compose to a single layer of
    thickness ``2h``. Direct test of the ``P_total = P_N ... P_1``
    propagator chain at n=2."""
    from fwap.cylindrical_solver import _modal_determinant_n2_cased

    p = _typical_g_pp_c_params()
    omega = 2.0 * np.pi * 5000.0
    kz = omega / p["vs"] * 1.05
    L_double = BoreholeLayer(vp=2500.0, vs=1100.0, rho=2100.0, thickness=0.01)
    L_half = BoreholeLayer(vp=2500.0, vs=1100.0, rho=2100.0, thickness=0.005)
    det_single = _modal_determinant_n2_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(L_double,),
    )
    det_split = _modal_determinant_n2_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(L_half, L_half),
    )
    # Same E_n2(a), identical propagator chain composes through the
    # same total thickness 2h.
    assert det_single == pytest.approx(det_split, rel=1.0e-10)


def test_modal_determinant_n2_cased_order_matters_at_N2():
    """Physical sanity: with two distinct layers ``(L_a, L_b)``,
    swapping the inside-out order to ``(L_b, L_a)`` produces a
    different determinant -- layer ordering is a physical
    parameter, not a labelling convention."""
    from fwap.cylindrical_solver import _modal_determinant_n2_cased

    p = _typical_g_pp_c_params()
    omega = 2.0 * np.pi * 5000.0
    L_a = BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01)  # casing
    L_b = BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.01)  # cement
    # kz safely above the slowest-shear bound floor.
    kz = omega / min(L_a.vs, L_b.vs, p["vs"], p["vf"]) * 1.05
    det_ab = _modal_determinant_n2_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(L_a, L_b),
    )
    det_ba = _modal_determinant_n2_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(L_b, L_a),
    )
    assert np.isfinite(det_ab) and np.isfinite(det_ba)
    rel_diff = abs(det_ab - det_ba) / max(abs(det_ab), abs(det_ba))
    assert rel_diff > 0.01


def test_modal_determinant_n2_cased_N2_casing_plus_cement_smoke():
    """Smoke for the cased-hole quadrupole N=2 path: typical
    casing + cement geometry produces a finite real determinant
    at a representative bound-regime ``kz``. Mirrors the
    G'.c smoke test for the LWD-band cement-bond geometry."""
    from fwap.cylindrical_solver import _modal_determinant_n2_cased

    p = _typical_g_pp_c_params()
    omega = 2.0 * np.pi * 5000.0
    casing = BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01)
    cement = BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.05)
    kz = omega / min(casing.vs, cement.vs, p["vs"], p["vf"]) * 1.05
    det = _modal_determinant_n2_cased(
        kz,
        omega,
        vp=p["vp"],
        vs=p["vs"],
        rho=p["rho"],
        vf=p["vf"],
        rho_f=p["rho_f"],
        a=p["a"],
        layers=(casing, cement),
    )
    assert np.isfinite(det)
    assert isinstance(det, float)


# =====================================================================
# Plan item G''.d -- public-API hook for cased-hole quadrupole
# =====================================================================
#
# Replaces the G''.0 ``len(layers) >= 1 -> NotImplementedError`` raise
# with a brentq loop on ``_modal_determinant_n2_cased`` (G''.c). The
# cased-hole quadrupole bound-mode regime in slow formation is much
# narrower than the n=1 flexural sister: at strong layer perturbations
# (real casing + cement; layer Vs much larger than formation Vs) the
# mode is mostly cut off across the LWD band. The smoke tests therefore
# pick formation params close to the slow-formation ceiling
# (Vs = 1200 m/s, vf = 1500 m/s) so casing + cement still satisfies the
# layer.vs >= formation.vs constraint *and* the geometric quadrupole
# bound regime stays open in a usable frequency window (~15-18 kHz).


def _typical_g_pp_d_cased_geometry():
    """Slow-formation cased-hole fixture for G''.d tests. Formation
    Vs = 1200 m/s sits just below ``vf = 1500`` so the slow-
    formation regime (``vs < vf``) holds; casing + cement layers
    remain faster in shear than the formation, satisfying
    ``_validate_flexural_layers_stacked``. The bound-mode quadrupole
    window for this geometry is ~15-18 kHz (narrower than the
    flexural sister's 3-12 kHz)."""
    return dict(
        casing=BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01),
        cement=BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.05),
        vp=2400.0,
        vs=1200.0,
        rho=2200.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )


def test_quadrupole_dispersion_layered_N1_layer_eq_formation_matches_unlayered():
    """G''.a.6 root-match oracle wired through the public API:
    when ``layer.{vp,vs,rho} == formation.{vp,vs,rho}`` the
    multi-layer brentq path returns the unlayered
    ``quadrupole_dispersion`` slowness bit-identically (modulo
    brentq's xtol=1e-10). Strongest end-to-end pinning of the
    G''.d wiring."""
    from fwap.cylindrical_solver import (
        BoreholeMode,
        quadrupole_dispersion,
        quadrupole_dispersion_layered,
    )

    vp, vs, rho = 2200.0, 800.0, 2200.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.01)
    f = np.linspace(4000.0, 12000.0, 9)
    res_unl = quadrupole_dispersion(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    res_lyr = quadrupole_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer,),
    )
    assert isinstance(res_lyr, BoreholeMode)
    finite = np.isfinite(res_unl.slowness) & np.isfinite(res_lyr.slowness)
    assert finite.any(), "fixture must put the bound mode in band"
    np.testing.assert_allclose(
        res_lyr.slowness[finite],
        res_unl.slowness[finite],
        rtol=1.0e-9,
    )


def test_quadrupole_dispersion_layered_N2_runs_smoke():
    """G''.d two-layer regression: a two-layer invaded-zone stack
    produces finite quadrupole slownesses across 6-12 kHz.

    Retargeted for roadmap A.8: this used to run on the cased
    fixture (steel casing + cement). With the SV column corrected
    the n >= 1 modes of a stiff annulus sit above the formation
    shear speed and are leaky, so the real-valued layered
    determinant -- the only n=2 layered path fwap has -- has no
    root there. See
    ``test_cased_slow_formation_dipole_is_leaky_not_bound``. The
    invaded-zone stack keeps them bound and exercises the same
    behaviour.
    """
    from fwap.cylindrical_solver import quadrupole_dispersion_layered

    g = _bound_invaded_geometry()
    f = np.linspace(6000.0, 12000.0, 4)
    res = quadrupole_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["inner"], g["outer"]),
    )
    assert res.name == "quadrupole"
    assert res.azimuthal_order == 2
    np.testing.assert_array_equal(res.freq, f)
    finite = np.isfinite(res.slowness)
    assert finite.sum() >= 3, (
        f"expected at least 3 of 4 frequencies to land in the cased "
        f"quadrupole bound regime; got {finite.sum()}"
    )
    # Bound-regime sanity: slowness > formation 1/V_S; phase
    # velocity below formation V_S (low-f asymptote) but above
    # formation Rayleigh (high-f asymptote, ~0.92 V_S).
    sl = res.slowness[finite]
    assert np.all(sl > 1.0 / g["vs"] * 0.99)
    assert np.all(sl < 1.0 / (g["vs"] * 0.85))


def test_quadrupole_dispersion_layered_N2_returns_borehole_mode():
    """``BoreholeMode`` return-type contract on the multi-layer
    quadrupole dispatch (G''.d).

    The cased fixture's screw mode is faster than its slow formation's
    shear speed, so roadmap A.9's leaky branch answers here and the
    result carries a real ``attenuation_per_meter``. This test used to
    assert that field was ``None``; that held only while the cased
    slow-formation path returned nothing at all. The bound-mode
    ``None`` contract is checked on the invaded-zone fixture below,
    where the mode really is bound.
    """
    from fwap.cylindrical_solver import quadrupole_dispersion_layered

    g = _typical_g_pp_d_cased_geometry()
    f = np.linspace(15000.0, 18000.0, 4)
    res = quadrupole_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], g["cement"]),
    )
    from fwap.cylindrical_solver import BoreholeMode

    assert isinstance(res, BoreholeMode)
    # Leaky branch: finite, positive attenuation wherever it answered.
    assert res.attenuation_per_meter is not None
    found = np.isfinite(res.slowness)
    assert found.any()
    assert np.all(res.attenuation_per_meter[found] > 0.0)

    # The bound path still reports no attenuation at all.
    bound = _bound_invaded_geometry()
    res_bound = quadrupole_dispersion_layered(
        np.linspace(6000.0, 12000.0, 4),
        vp=bound["vp"],
        vs=bound["vs"],
        rho=bound["rho"],
        vf=bound["vf"],
        rho_f=bound["rho_f"],
        a=bound["a"],
        layers=(bound["inner"], bound["outer"]),
    )
    assert np.isfinite(res_bound.slowness).all()
    assert res_bound.attenuation_per_meter is None


def test_quadrupole_dispersion_layered_N2_layer_permutation_changes_slowness():
    """Casing-inside-cement and cement-inside-casing produce
    distinct quadrupole slowness curves -- inside-out layer
    ordering is a physical parameter, not a labelling convention."""
    from fwap.cylindrical_solver import quadrupole_dispersion_layered

    g = _bound_invaded_geometry()
    f = np.array([8000.0, 10000.0, 12000.0])
    res_cs = quadrupole_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["inner"], g["outer"]),
    )
    res_sc = quadrupole_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["outer"], g["inner"]),
    )
    # Both should produce at least some finite slownesses for the
    # comparison to be meaningful.
    common = np.isfinite(res_cs.slowness) & np.isfinite(res_sc.slowness)
    assert common.any(), "fixture must yield bound modes in both orderings"
    rel_diff = np.abs(res_cs.slowness[common] - res_sc.slowness[common]) / np.abs(
        res_cs.slowness[common]
    )
    assert np.any(rel_diff > 1.0e-3)


def test_quadrupole_dispersion_layered_N2_thin_outer_layer_eq_formation_collapses_to_N1():
    """G''.a.6 propagator-chain corollary: stacking a thin outer
    layer with formation parameters on top of an inner layer
    leaves the cased-hole quadrupole brentq root unchanged
    (within brentq xtol). The trivial outer layer's per-layer
    propagator collapses to ``E_form(b_outer) E_form(b_inner)^{-1}``
    and feeds the same effective state vector at the formation
    interface as the N=1 path would."""
    from fwap.cylindrical_solver import quadrupole_dispersion_layered

    vp, vs, rho = 2200.0, 800.0, 2200.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    inner = BoreholeLayer(vp=2300.0, vs=900.0, rho=2200.0, thickness=0.005)
    trivial_outer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.003)
    f = np.linspace(4000.0, 12000.0, 6)
    res_n1 = quadrupole_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(inner,),
    )
    res_n2 = quadrupole_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(inner, trivial_outer),
    )
    common = np.isfinite(res_n1.slowness) & np.isfinite(res_n2.slowness)
    assert common.any(), "fixture must produce some bound modes"
    np.testing.assert_allclose(
        res_n2.slowness[common],
        res_n1.slowness[common],
        rtol=1.0e-7,
    )


def test_quadrupole_dispersion_layered_N3_runs_smoke():
    """G''.d N=3 smoke (casing + cement + mudcake): the brentq +
    propagator-chain path runs to completion at three layers and
    returns a finite slowness in the LWD-relevant band where the
    cased quadrupole is bound. Mudcake placed *inside* the invaded zone
    requires the radial-outward layer order ``(mudcake, inner,
    outer)``."""
    from fwap.cylindrical_solver import quadrupole_dispersion_layered

    g = _bound_invaded_geometry()
    mudcake = BoreholeLayer(vp=1980.0, vs=900.0, rho=1700.0, thickness=0.002)
    f = np.array([8000.0, 10000.0, 12000.0])
    res = quadrupole_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(mudcake, g["inner"], g["outer"]),
    )
    assert res.name == "quadrupole"
    assert res.azimuthal_order == 2
    finite = np.isfinite(res.slowness)
    assert finite.any(), "expected at least one frequency in bound regime"


# =====================================================================
# Plan item G''.e -- n=2 hardening
# =====================================================================
#
# Hardening tests for the G''.d brentq path: multi-frequency
# det-at-root self-consistency, thin-inner-formation-layer collapse to
# the N=1 outer-only case, two-formation-layers collapse to the
# unlayered ``quadrupole_dispersion`` (master-plan G'' validation
# bullet), and the LWD-quadrupole cement-bond physics direction
# (stiffer cement -> faster phase velocity / smaller slowness; the
# *opposite* of the original plan-doc impedance-coupling argument,
# now corrected).


def test_quadrupole_dispersion_layered_det_vanishes_at_brentq_roots_multi_freq():
    """G''.e self-consistency: at every brentq-converged ``k_z`` in
    the LWD band, ``|_modal_determinant_n2_cased(k_z_root)|`` is
    many orders of magnitude smaller than the same determinant
    evaluated 0.5 % off the root. Catches a brentq that returned
    a non-zero (non-converged) ``k_z`` -- e.g., from a bracket that
    touched but did not cross zero.

    Sister of the multi-frequency det-at-root oracle in G.e / G'.e."""
    from fwap.cylindrical_solver import (
        _modal_determinant_n2_cased,
        quadrupole_dispersion_layered,
    )

    g = _bound_invaded_geometry()
    cement_thin = g["outer"]
    f = np.linspace(6000.0, 14000.0, 5)
    res = quadrupole_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["inner"], cement_thin),
    )
    finite_idx = np.where(np.isfinite(res.slowness))[0]
    assert finite_idx.size >= 4, (
        f"need >=4 bound-mode frequencies for the multi-freq oracle; "
        f"got {finite_idx.size}"
    )
    for i in finite_idx:
        omega = 2.0 * np.pi * float(res.freq[i])
        kz_root = float(res.slowness[i]) * omega
        det_at = _modal_determinant_n2_cased(
            kz_root,
            omega,
            vp=g["vp"],
            vs=g["vs"],
            rho=g["rho"],
            vf=g["vf"],
            rho_f=g["rho_f"],
            a=g["a"],
            layers=(g["inner"], cement_thin),
        )
        det_off = _modal_determinant_n2_cased(
            kz_root * 1.005,
            omega,
            vp=g["vp"],
            vs=g["vs"],
            rho=g["rho"],
            vf=g["vf"],
            rho_f=g["rho_f"],
            a=g["a"],
            layers=(g["inner"], cement_thin),
        )
        assert abs(det_off) > 0.0
        assert abs(det_at) < abs(det_off) * 1.0e-6, (
            f"f={res.freq[i] / 1000:.1f} kHz: |det_at|={abs(det_at):.2e}  "
            f"|det_off|={abs(det_off):.2e}  ratio={abs(det_at) / abs(det_off):.2e}"
        )


def test_quadrupole_dispersion_layered_two_formation_layers_collapse_to_unlayered():
    """G''.e master-plan validation: with both layers carrying
    formation parameters, the multi-layer brentq path returns the
    unlayered ``quadrupole_dispersion`` slowness to ``rtol=1e-6``
    across the full bound-regime band. Sister of the G''.d N=1
    layer = formation oracle, extended to N=2 telescoping
    (G''.a.6 propagator chain ``P_total = E_form(b) E_form(a)^-1``
    independent of ``N`` when all layers equal formation)."""
    from fwap.cylindrical_solver import (
        quadrupole_dispersion,
        quadrupole_dispersion_layered,
    )

    vp, vs, rho = 2200.0, 800.0, 2200.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer1 = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.005)
    layer2 = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.008)
    f = np.linspace(4000.0, 12000.0, 9)
    res_unl = quadrupole_dispersion(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    res_n2 = quadrupole_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer1, layer2),
    )
    finite = np.isfinite(res_unl.slowness) & np.isfinite(res_n2.slowness)
    assert finite.sum() >= 6, f"need >=6 bound-mode frequencies; got {finite.sum()}"
    np.testing.assert_allclose(
        res_n2.slowness[finite],
        res_unl.slowness[finite],
        rtol=1.0e-6,
    )


def test_quadrupole_dispersion_layered_thin_inner_formation_layer_approaches_outer_only():
    """G''.e thin-inner-layer collapse: with a thin inner layer
    carrying formation parameters, the (thin_inner, outer) N=2
    brentq root converges to the (outer,) N=1 root as
    ``thickness_inner -> 0``. Continuity / vanishing-perturbation
    test on the propagator chain.

    The inner-layer-at-formation-params propagator
    ``P_inner = E_form(a + h) E_form(a)^{-1}`` is a near-identity
    perturbation as ``h -> 0``; the outer layer's contribution at
    its inner radius (now at ``a + h`` instead of ``a``) is the
    only residual difference, which vanishes linearly in ``h``."""
    from fwap.cylindrical_solver import quadrupole_dispersion_layered

    g = _bound_invaded_geometry()
    cement_thin = g["outer"]
    inner_form = BoreholeLayer(
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        thickness=1.0e-4,
    )
    f = np.array([8000.0, 10000.0])
    res_n1 = quadrupole_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["inner"], cement_thin),
    )
    res_n2 = quadrupole_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(inner_form, g["inner"], cement_thin),
    )
    finite = np.isfinite(res_n1.slowness) & np.isfinite(res_n2.slowness)
    assert finite.any(), "fixture must yield bound modes in both"
    # rtol scales with thickness_inner / borehole_radius ~ 1e-3.
    np.testing.assert_allclose(
        res_n2.slowness[finite],
        res_n1.slowness[finite],
        rtol=1.0e-3,
    )


def test_quadrupole_dispersion_layered_cement_bond_stiffer_cement_makes_mode_faster():
    """G''.e LWD-quadrupole cement-bond physics: holding casing
    and formation fixed, sweeping ``vs_cement`` over the valid
    range (``vs_cement >= vs_form``) shows the guided-mode phase
    velocity rising monotonically with cement stiffness -- i.e.,
    slowness *decreases* as cement gets stiffer.

    This is the *opposite* of the original plan-doc prediction
    (which argued by impedance-coupling intuition that stiffer
    cement should pull slowness toward the formation-shear
    asymptote). The numerical evidence over annulus Vs in
    [810, 900] m/s with the bound-regime invaded fixture
    (``vs_form = 800``) consistently shows the simpler "stiffer
    annulus transmits the wave faster" reading: stiffer annulus
    -> smaller slowness.

    Retargeted for roadmap A.8: this used to run on the cased
    fixture (steel casing + cement). With the SV column corrected
    the n >= 1 modes of a stiff annulus sit above the formation
    shear speed and are leaky, so the real-valued layered
    determinant -- the only n=2 layered path fwap has -- has no
    root there. See
    ``test_cased_slow_formation_dipole_is_leaky_not_bound``. The
    invaded-zone stack keeps them bound and exercises the same
    behaviour.
    """
    from fwap.cylindrical_solver import quadrupole_dispersion_layered

    g = _bound_invaded_geometry()
    cement_soft = BoreholeLayer(
        vp=1782.0,
        vs=810.0,
        rho=2000.0,
        thickness=0.04,
    )
    cement_stiff = BoreholeLayer(
        vp=1980.0,
        vs=900.0,
        rho=2000.0,
        thickness=0.04,
    )
    f = np.array([8000.0, 10000.0, 12000.0])
    res_soft = quadrupole_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["inner"], cement_soft),
    )
    res_stiff = quadrupole_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["inner"], cement_stiff),
    )
    common = np.isfinite(res_soft.slowness) & np.isfinite(res_stiff.slowness)
    assert common.sum() >= 2, (
        f"need >=2 frequencies with both cement variants in the bound "
        f"regime; got {common.sum()}"
    )
    # Stiffer cement should give SMALLER slowness (faster phase
    # velocity) at every comparable frequency.
    soft_sl = res_soft.slowness[common]
    stiff_sl = res_stiff.slowness[common]
    assert np.all(stiff_sl < soft_sl), (
        f"expected stiffer cement to give smaller slowness;\n"
        f"  soft (Vs=810) slownesses: {soft_sl}\n"
        f"  stiff(Vs=900) slownesses: {stiff_sl}"
    )
    # Sanity: the shift should be a few percent (not tiny noise,
    # not >50% which would indicate a different-mode ambiguity).
    rel_shift = (soft_sl - stiff_sl) / soft_sl
    assert np.all(rel_shift > 0.005)
    assert np.all(rel_shift < 0.30)


# =====================================================================
# Fast-formation cased-hole quadrupole (deferred follow-up to G'')
# =====================================================================
#
# Tests for the complex-determinant cased-hole quadrupole dispatch in
# ``quadrupole_dispersion_layered`` when ``V_S > V_f``. The path
# uses ``_layer_e_matrix_n2_complex`` /
# ``_layer_propagator_n2_complex`` /
# ``_modal_determinant_n2_cased_complex`` to evaluate the modal
# determinant at complex ``k_z``, and brentq's ``Im(det)`` along the
# real-``k_z`` axis (mirroring the unlayered fast-formation auto-
# dispatch in ``quadrupole_dispersion``).
#
# Per-layer slow-formation constraint does NOT apply in this regime
# (a cement layer softer than a fast carbonate formation is
# physically permissible); only the formation half-space is
# required to be in the fast regime ``V_S > V_f``.


def _typical_fast_formation_cased_geometry():
    """Fast-formation cased-hole fixture for the n=2 complex-
    determinant tests. Formation V_S = 2600 > V_f = 1500 puts the
    bound regime in slowness ``(1/V_S, 1/V_R) = (~3.85e-4, ~4.18e-4)``;
    layers carry typical casing + cement properties (no slow-
    formation per-layer constraint at fast formation)."""
    return dict(
        vp=4500.0,
        vs=2600.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
        casing=BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01),
        cement=BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.05),
    )


def test_modal_determinant_n2_cased_complex_real_kz_slow_formation_matches_real_cased():
    """Slow-formation regression: at real ``k_z`` and slow
    formation, the complex cased determinant matches the real
    one to floating-point precision (its imaginary part is
    zero). Anchors the complex-extension correctness against the
    G''.c real-valued path."""
    from fwap.cylindrical_solver import (
        _modal_determinant_n2_cased,
        _modal_determinant_n2_cased_complex,
    )

    vp, vs, rho = 2200.0, 800.0, 2200.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.01)
    omega = 2.0 * np.pi * 5000.0
    kz = 1.05 * omega / vs
    det_re = _modal_determinant_n2_cased(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer,),
    )
    det_cx = _modal_determinant_n2_cased_complex(
        complex(kz, 0.0),
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer,),
    )
    assert det_cx.real == pytest.approx(det_re, rel=1.0e-12)
    # ``Im(det)`` is exactly zero when the matrix is real (slow
    # formation, real ``k_z``, all layers / formation in the bound
    # regime).
    assert det_cx.imag == 0.0


def test_modal_determinant_n2_cased_complex_layer_eq_formation_im_det_sign_matches_unlayered():
    """G''.a.6 collapse identity in the fast-formation regime:
    when ``layer == formation``, the cased determinant is the
    unlayered determinant times a real-valued ``det(E_form(b))``
    factor. The sign-change locations of ``Im(det_cas)`` along
    real ``k_z`` therefore coincide with the sign-change
    locations of ``Im(det_unl)`` -- i.e., the modal roots match
    even though the brentq-picked specific value can differ
    when there are multiple sign changes in the bracket."""
    from fwap.cylindrical_solver import (
        _modal_determinant_n2_cased_complex,
        _modal_determinant_n2_complex,
    )

    g = _typical_fast_formation_cased_geometry()
    layer = BoreholeLayer(vp=g["vp"], vs=g["vs"], rho=g["rho"], thickness=0.01)
    omega = 2.0 * np.pi * 14000.0
    vR = 0.92 * g["vs"]
    slows = np.linspace(1.0 / g["vs"] * 1.0001, 1.0 / vR * 0.9999, 40)
    sgn_unl = []
    sgn_cas = []
    for slow in slows:
        kz = slow * omega
        d_unl = _modal_determinant_n2_complex(
            complex(kz, 0.0),
            omega,
            g["vp"],
            g["vs"],
            g["rho"],
            g["vf"],
            g["rho_f"],
            g["a"],
        ).imag
        d_cas = _modal_determinant_n2_cased_complex(
            complex(kz, 0.0),
            omega,
            vp=g["vp"],
            vs=g["vs"],
            rho=g["rho"],
            vf=g["vf"],
            rho_f=g["rho_f"],
            a=g["a"],
            layers=(layer,),
        ).imag
        sgn_unl.append(np.sign(d_unl))
        sgn_cas.append(np.sign(d_cas))
    sgn_unl = np.array(sgn_unl)
    sgn_cas = np.array(sgn_cas)
    # Sign-change indices must match exactly. The sign relationship
    # between the two streams is uniformly opposite (the real
    # ``det(E_form(b))`` scale factor is negative for this fixture)
    # but a uniform sign flip preserves the zero-crossing locations.
    changes_unl = np.where(np.diff(sgn_unl) != 0)[0]
    changes_cas = np.where(np.diff(sgn_cas) != 0)[0]
    np.testing.assert_array_equal(changes_unl, changes_cas)
    # And both streams must have at least one sign change in the
    # bound regime -- otherwise the test fixture would not exercise
    # the modal-root topology.
    assert changes_unl.size >= 1


def test_quadrupole_dispersion_layered_fast_formation_layer_eq_formation_matches_open_hole():
    """The oracle A.7 said could not be written, written for real.

    With ``layer = formation`` -- a 1 cm annulus identical to the
    half-space -- the cased dispatch is physically the open hole, so it
    must return the open-hole answer. It used to return nothing: the
    marcher, tracking ``Im(det)``, found 10 sign changes at 10 kHz and
    33 at 14 kHz and declined to choose, and that was recorded as
    catastrophic cancellation in the propagator chain.

    It was not the propagator. The ``n = 2`` determinant is real, not
    imaginary, so ``Im(det)`` was round-off and every one of those
    crossings was noise; the propagator reproduces ``E(b)`` from
    ``P E(a)`` to 1e-16. Tracking the part that carries the signal, the
    cased path now reproduces the open-hole branch to **1e-13**.
    """
    from fwap.cylindrical_solver import (
        BoreholeMode,
        quadrupole_dispersion,
        quadrupole_dispersion_layered,
    )

    g = _typical_fast_formation_cased_geometry()
    layer = BoreholeLayer(vp=g["vp"], vs=g["vs"], rho=g["rho"], thickness=0.01)
    f = np.linspace(10000.0, 16000.0, 4)
    res = quadrupole_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(layer,),
    )
    assert isinstance(res, BoreholeMode)
    assert res.attenuation_per_meter is None  # bound mode
    assert np.isfinite(res.slowness).all()

    open_hole = quadrupole_dispersion(
        f, vp=g["vp"], vs=g["vs"], rho=g["rho"], vf=g["vf"], rho_f=g["rho_f"], a=g["a"]
    )
    finite = np.isfinite(open_hole.slowness)
    assert finite.all(), "the open-hole path resolves the branch"
    np.testing.assert_allclose(res.slowness, open_hole.slowness, rtol=1.0e-11)

    velocity = 1.0 / open_hole.slowness[finite]
    assert np.all(velocity < g["vs"])
    assert np.all(velocity > g["vf"])


def test_quadrupole_dispersion_layered_fast_formation_N2_runs_smoke():
    """Multi-layer fast-formation smoke (casing + cement on a
    fast carbonate formation): the brentq-on-Im(det) path runs
    to completion at N=2 and returns the same return-type
    contract as the slow-formation path."""
    from fwap.cylindrical_solver import quadrupole_dispersion_layered

    g = _typical_fast_formation_cased_geometry()
    f = np.linspace(8000.0, 18000.0, 6)
    res = quadrupole_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], g["cement"]),
    )
    assert res.name == "quadrupole"
    assert res.azimuthal_order == 2
    assert res.attenuation_per_meter is None
    np.testing.assert_array_equal(res.freq, f)
    # Don't pin the slowness values themselves -- the cased fast-
    # formation bound mode for casing+cement is in a narrow
    # window and the brentq-on-Im(det) bracket can land on
    # spurious sign changes outside the physical regime when
    # the mode is cut off (same behaviour as the slow-formation
    # sister; documented in PR #71). Just verify the path
    # executes without exceptions.
    assert res.slowness.shape == f.shape


def test_quadrupole_dispersion_layered_fast_formation_does_not_break_slow_formation_path():
    """Regression: adding the fast-formation dispatch did not
    perturb the slow-formation cased-hole quadrupole brentq path
    that landed in G''.d. With slow formation params (``V_S = 800
    < V_f = 1500``), the dispatch goes through the real-valued
    G''.c path and the result matches the unlayered
    ``quadrupole_dispersion`` slowness when ``layer = formation``."""
    from fwap.cylindrical_solver import (
        quadrupole_dispersion,
        quadrupole_dispersion_layered,
    )

    vp, vs, rho = 2200.0, 800.0, 2200.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.01)
    f = np.linspace(4000.0, 12000.0, 9)
    res_unl = quadrupole_dispersion(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    res_lyr = quadrupole_dispersion_layered(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer,),
    )
    finite = np.isfinite(res_unl.slowness) & np.isfinite(res_lyr.slowness)
    assert finite.any()
    np.testing.assert_allclose(
        res_lyr.slowness[finite],
        res_unl.slowness[finite],
        rtol=1.0e-9,
    )


def test_quadrupole_dispersion_layered_fast_formation_does_not_apply_slow_formation_constraint():
    """In fast formation, the per-layer ``layer.vs >= formation.vs``
    constraint enforced by ``_validate_flexural_layers_stacked`` for
    the slow-formation regime does NOT apply: a cement layer with
    ``vs < formation.vs`` is physically permissible (e.g., soft
    cement behind a fast-carbonate formation). The fast-formation
    dispatch must accept such a configuration without raising."""
    from fwap.cylindrical_solver import quadrupole_dispersion_layered

    # Fast formation V_S = 2600 > V_f = 1500.
    # Cement V_S = 1300 < V_S = 2600 (would fail slow-formation
    # constraint, but allowed in fast formation).
    soft_layer = BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.02)
    res = quadrupole_dispersion_layered(
        np.array([10000.0, 14000.0]),
        vp=4500.0,
        vs=2600.0,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
        layers=(soft_layer,),
    )
    # No exception; structural contract holds.
    assert res.name == "quadrupole"
    assert res.azimuthal_order == 2


# =====================================================================
# Fast-formation cased-hole flexural (n=1 sister of the G'' follow-up)
# =====================================================================
#
# Tests for the complex-determinant dispatch in
# ``flexural_dispersion_layered`` when ``V_S > V_f``. The path uses
# ``_layer_e_matrix_n1_complex`` / ``_layer_propagator_n1_complex`` /
# ``_modal_determinant_n1_cased_complex`` to evaluate the modal
# determinant at complex ``k_z``, and brentq's ``Im(det)`` along the
# real-``k_z`` axis (mirroring the unlayered fast-formation flexural
# auto-dispatch in ``flexural_dispersion``).
#
# Unlike the n=2 sister, the n=1 layer=formation collapse is clean
# enough that brentq lands on the *same* root as the unlayered
# solver, so the oracle here can pin slowness values directly rather
# than only sign-change locations.


def _typical_fast_formation_cased_n1_geometry():
    """Fast-formation cased-hole fixture for the n=1 complex-
    determinant tests. Formation V_S = 2820 > V_f = 1500 (the
    Paillet & Cheng limestone), so the bound regime is slowness in
    ``(1/V_S, 1/V_R) ~= (3.55e-4, 3.85e-4)``. Layers carry typical
    casing + cement properties; no slow-formation per-layer
    constraint applies at fast formation."""
    return dict(
        vp=4880.0,
        vs=2820.0,
        rho=2700.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
        casing=BoreholeLayer(vp=5860.0, vs=3140.0, rho=7800.0, thickness=0.01),
        cement=BoreholeLayer(vp=2300.0, vs=1300.0, rho=1900.0, thickness=0.05),
    )


def test_modal_determinant_n1_cased_complex_real_kz_slow_formation_matches_real():
    """Slow-formation regression: at real ``k_z`` and slow
    formation the complex cased determinant matches the real
    G'.c one to floating-point precision, with an exactly-zero
    imaginary part. Anchors the complex-extension correctness."""
    from fwap.cylindrical_solver import (
        _modal_determinant_n1_cased,
        _modal_determinant_n1_cased_complex,
    )

    vp, vs, rho = 2200.0, 800.0, 2200.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=2500.0, vs=1100.0, rho=2100.0, thickness=0.005)
    omega = 2.0 * np.pi * 5000.0
    kz = 1.05 * omega / vs
    det_re = _modal_determinant_n1_cased(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer,),
    )
    det_cx = _modal_determinant_n1_cased_complex(
        complex(kz, 0.0),
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=(layer,),
    )
    assert det_cx.real == pytest.approx(det_re, rel=1.0e-12)
    assert det_cx.imag == 0.0


def test_layer_e_matrix_n1_complex_matches_real_at_real_kz():
    """Entry-level oracle for the complex E-matrix: every one of
    the 36 entries matches the real ``_layer_e_matrix_n1`` in the
    bound regime, with zero imaginary part."""
    from fwap.cylindrical_solver import (
        _layer_e_matrix_n1,
        _layer_e_matrix_n1_complex,
    )

    omega = 2.0 * np.pi * 5000.0
    vp, vs, rho, r = 2500.0, 1100.0, 2100.0, 0.1
    kz = 1.05 * omega / vs
    E_re = _layer_e_matrix_n1(kz, omega, vp=vp, vs=vs, rho=rho, r=r)
    E_cx = _layer_e_matrix_n1_complex(
        complex(kz, 0.0), omega, vp=vp, vs=vs, rho=rho, r=r
    )
    assert np.all(E_cx.imag == 0.0)
    np.testing.assert_allclose(E_cx.real, E_re, rtol=1.0e-12)


def test_flexural_dispersion_layered_fast_formation_layer_eq_formation_matches_unlayered():
    """Collapse identity through the public API: with
    ``layer = formation`` in the fast-formation regime, the cased
    dispatch reproduces the unlayered ``flexural_dispersion``
    slowness to ``rtol=1e-9``, and the NaN pattern (frequencies
    outside the geometric cutoff) matches exactly.

    Strongest end-to-end oracle for the n=1 fast-formation
    wiring."""
    g = _typical_fast_formation_cased_n1_geometry()
    layer = BoreholeLayer(vp=g["vp"], vs=g["vs"], rho=g["rho"], thickness=0.01)
    f = np.linspace(3000.0, 15000.0, 7)
    res_unl = flexural_dispersion(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
    )
    res_lyr = flexural_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(layer,),
    )
    # NaN pattern must agree exactly -- the layer=formation stack
    # cannot open or close the geometric cutoff.
    np.testing.assert_array_equal(
        np.isfinite(res_unl.slowness), np.isfinite(res_lyr.slowness)
    )
    finite = np.isfinite(res_unl.slowness)
    assert finite.any(), "fixture must put the bound mode in band"
    np.testing.assert_allclose(
        res_lyr.slowness[finite], res_unl.slowness[finite], rtol=1.0e-9
    )


def test_flexural_dispersion_layered_fast_formation_N2_runs_smoke():
    """Multi-layer fast-formation smoke (casing + cement behind a
    fast limestone): the brentq-on-Im(det) path runs to completion
    at N=2 and lands inside the ``(V_f, V_S)`` bound window where
    it converges. The window was written ``(V_R, V_S)`` before A.2 was
    fixed; ``V_R`` is not a bound of this mode."""
    g = _typical_fast_formation_cased_n1_geometry()
    f = np.linspace(4000.0, 16000.0, 7)
    res = flexural_dispersion_layered(
        f,
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["casing"], g["cement"]),
    )
    assert res.name == "flexural"
    assert res.azimuthal_order == 1
    assert res.attenuation_per_meter is None
    np.testing.assert_array_equal(res.freq, f)
    finite = np.isfinite(res.slowness)
    assert finite.any(), "expected at least one bound-regime root"
    # Bound regime: phase velocity between V_f and V_S. The upper
    # slowness bound used to be 1/(0.90 V_S), i.e. "not far below the
    # Rayleigh speed" -- which is the A.2 assumption, not a property of
    # the mode. The branch descends toward Scholte and passes V_R.
    sl = res.slowness[finite]
    assert np.all(sl > 1.0 / g["vs"] * 0.99)
    assert np.all(sl < 1.0 / g["vf"])


def test_flexural_dispersion_layered_fast_formation_does_not_break_slow_formation():
    """Regression: adding the fast-formation dispatch did not
    perturb the slow-formation layered path. With slow-formation
    params (``V_S = 800 < V_f = 1500``) and ``layer = formation``,
    the result still matches the unlayered ``flexural_dispersion``
    to ``rtol=1e-9``."""
    vp, vs, rho = 2200.0, 800.0, 2200.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    layer = BoreholeLayer(vp=vp, vs=vs, rho=rho, thickness=0.01)
    f = np.linspace(2000.0, 8000.0, 5)
    res_unl = flexural_dispersion(f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a)
    res_lyr = flexural_dispersion_layered(
        f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a, layers=(layer,)
    )
    finite = np.isfinite(res_unl.slowness) & np.isfinite(res_lyr.slowness)
    assert finite.any()
    np.testing.assert_allclose(
        res_lyr.slowness[finite], res_unl.slowness[finite], rtol=1.0e-9
    )


def test_flexural_dispersion_layered_fast_formation_allows_layer_softer_than_formation():
    """In fast formation the per-layer ``layer.vs >= formation.vs``
    constraint enforced for the slow-formation regime does NOT
    apply: cement softer in shear than a fast carbonate is
    physically permissible. The dispatch must accept it without
    raising ``ValueError``."""
    g = _typical_fast_formation_cased_n1_geometry()
    # Cement V_S = 1300 < formation V_S = 2820: would fail the
    # slow-formation stacked constraint, but is legal here.
    res = flexural_dispersion_layered(
        np.array([8000.0, 12000.0]),
        vp=g["vp"],
        vs=g["vs"],
        rho=g["rho"],
        vf=g["vf"],
        rho_f=g["rho_f"],
        a=g["a"],
        layers=(g["cement"],),
    )
    assert res.name == "flexural"
    assert res.azimuthal_order == 1


# ----------------------------------------------------------------------
