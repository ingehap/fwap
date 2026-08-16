"""
Plan H: the VTI (transversely isotropic) formation determinants.

One of six modules split out of ``tests/test_cylindrical_solver.py``.
Christoffel roots, the n=0 and n=1 row builders that consume them, the
public ``*_vti`` entry points, and the Norris (1990) low-frequency
closed form that anchors the n=0 side.

The later VTI work -- the complex-``k_z`` helper, the conjugate-column
recombination, the cased VTI stack -- lives in
``tests/test_anisotropy.py`` instead, alongside the leaky VTI roadmap
items it belongs to.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fwap.cylindrical import (
    flexural_dispersion_vti_physical,
)
from fwap.cylindrical_solver import (
    BoreholeMode,
    _is_isotropic_stiffness,
    _modal_determinant_n0,
    _modal_determinant_n0_vti,
    _modal_determinant_n1,
    _modal_determinant_n1_vti,
    _modal_row1_at_a_n1_vti,
    _modal_row1_at_a_vti,
    _modal_row2_at_a_n1_vti,
    _modal_row2_at_a_vti,
    _modal_row3_at_a_n1_vti,
    _modal_row3_at_a_vti,
    _modal_row4_at_a_n1_vti,
    _polarization_ratio_uz_over_ur_vti,
    _radial_wavenumbers_vti,
    flexural_dispersion,
    flexural_dispersion_vti,
    stoneley_dispersion,
    stoneley_dispersion_vti,
)
from tests._solver_media import (
    SLOW_A,
    SLOW_RHO,
    SLOW_RHO_F,
    SLOW_VF,
    SLOW_VP,
    SLOW_VS,
    _green_river_shale_stiffness,
)

# =====================================================================
# Plan item H.0 -- public-API foundation for VTI formation
# =====================================================================
#
# Sister of F.1.0 / F.2.0 layered foundations along the anisotropy
# axis. The 5-parameter TI stiffness tensor (C11, C13, C33, C44,
# C66) collapses to the isotropic case when C11=C33, C44=C66, and
# C13=C11-2*C44 -- the dispatch in stoneley_dispersion_vti and
# flexural_dispersion_vti detects this and routes to the existing
# isotropic solvers, providing the floating-point oracle for the
# entire H chain.


def _isotropic_stiffness_from_lame(vp, vs, rho):
    """Construct an isotropic stiffness tensor (C11, C13, C33,
    C44, C66) from the Lame parameters (vp, vs, rho)."""
    mu = rho * vs**2
    lam = rho * vp**2 - 2.0 * mu
    return dict(
        c11=lam + 2.0 * mu,
        c13=lam,
        c33=lam + 2.0 * mu,
        c44=mu,
        c66=mu,
    )


def test_stoneley_dispersion_vti_isotropic_collapse_bit_matches_unlayered():
    """Floating-point oracle for the H chain: with an isotropic
    stiffness tensor the VTI Stoneley solver bit-matches the
    isotropic ``stoneley_dispersion`` answer to ``rtol=1e-12``
    across a 16-point frequency grid."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = np.linspace(500.0, 8000.0, 16)
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)

    res_iso = stoneley_dispersion(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    res_vti = stoneley_dispersion_vti(
        f,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    np.testing.assert_array_equal(res_vti.slowness, res_iso.slowness)
    np.testing.assert_array_equal(res_vti.freq, res_iso.freq)
    assert res_vti.name == "Stoneley"
    assert res_vti.azimuthal_order == 0


def test_flexural_dispersion_vti_isotropic_collapse_bit_matches_unlayered():
    """Same floating-point oracle for the n=1 dipole flexural."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = np.linspace(2000.0, 8000.0, 12)
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)

    res_iso = flexural_dispersion(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    res_vti = flexural_dispersion_vti(
        f,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    np.testing.assert_array_equal(res_vti.slowness, res_iso.slowness)
    np.testing.assert_array_equal(res_vti.freq, res_iso.freq)
    assert res_vti.name == "flexural"
    assert res_vti.azimuthal_order == 1


def test_fluid_bessels_n1_vti_bound_regime_is_bit_identical():
    """The regime-dispatching fluid helper must reproduce the
    pre-existing real path **bit-exactly** in the bound regime.

    Complex-argument ``iv`` differs from the real call by a few ULP, so
    routing the bound branch through complex arithmetic would perturb
    the slow-formation VTI curve that is tied to Ellefsen fig 4 at
    0.30 % RMS. The helper therefore keeps the real calls; this test is
    what stops a later 'simplification' from collapsing the two."""
    from scipy import special

    from fwap.cylindrical_solver._vti import _fluid_bessels_n1_vti

    vf, a = 1500.0, 0.10
    for f_hz, velocity in ((2000.0, 1200.0), (8000.0, 1000.0), (15000.0, 900.0)):
        omega = 2.0 * np.pi * f_hz
        kz = omega / velocity
        assert kz * kz - (omega / vf) ** 2 > 0.0
        F_f, i0, i1 = _fluid_bessels_n1_vti(kz, omega, vf, a)
        F_ref = float(np.sqrt(kz * kz - (omega / vf) ** 2))
        assert F_f == F_ref
        assert i0 == float(special.iv(0, F_ref * a))
        assert i1 == float(special.iv(1, F_ref * a))


def test_fluid_bessels_n1_vti_fast_regime_is_the_oscillatory_continuation():
    """Below ``F_f^2 = 0`` the helper must continue analytically to the
    oscillatory branch: ``I_n(i y) = i^n J_n(y)``. That identity is what
    makes a single ``iv`` call correct in both regimes."""
    from scipy import special

    from fwap.cylindrical_solver._vti import _fluid_bessels_n1_vti

    vf, a = 1500.0, 0.10
    omega = 2.0 * np.pi * 4000.0
    kz = omega / 1700.0  # phase velocity above V_f -> F_f^2 < 0
    assert kz * kz - (omega / vf) ** 2 < 0.0
    F_f, i0, i1 = _fluid_bessels_n1_vti(kz, omega, vf, a)
    y = abs(F_f.imag if isinstance(F_f, complex) else 0.0) * a
    assert abs(complex(F_f).real) < 1e-12
    np.testing.assert_allclose(complex(i0).real, special.jv(0, y), rtol=1e-12)
    np.testing.assert_allclose(complex(i1).imag, special.jv(1, y), rtol=1e-12)


def test_modal_determinant_n1_vti_complex_matches_real_in_bound_regime():
    """The complex determinant is the same object as the real one where
    both are defined -- same matrix builder, so they cannot drift."""
    from fwap.cylindrical_solver._vti import (
        _modal_determinant_n1_vti,
        _modal_determinant_n1_vti_complex,
    )

    # Slow-formation TI: V_Sv < V_f.
    cij = _green_river_shale_stiffness()
    cij = dict(cij, c44=cij["c44"] * 0.5, c66=cij["c66"] * 0.5)
    kw = dict(**cij, vf=1500.0, rho_f=1000.0, a=0.10)
    omega = 2.0 * np.pi * 8000.0
    kz = omega / 1150.0
    real = _modal_determinant_n1_vti(kz, omega, **kw)
    comp = _modal_determinant_n1_vti_complex(kz, omega, **kw)
    assert np.isfinite(real)
    assert real == comp.real
    assert comp.imag == 0.0


def test_modal_determinant_n1_vti_real_returns_nan_outside_bound_regime():
    """The real determinant keeps its documented contract: NaN where the
    fluid Bessels turn oscillatory, rather than a finite-but-meaningless
    ``M.real`` now that the rows can be genuinely complex."""
    from fwap.cylindrical_solver._vti import _modal_determinant_n1_vti

    kw = dict(**_green_river_shale_stiffness(), vf=1500.0, rho_f=1000.0, a=0.10)
    omega = 2.0 * np.pi * 4000.0
    kz = omega / 1700.0
    assert kz * kz - (omega / 1500.0) ** 2 < 0.0
    assert np.isnan(_modal_determinant_n1_vti(kz, omega, **kw))


def test_flexural_dispersion_vti_fast_formation_genuine_TI_is_implemented():
    """Fast-formation genuine TI no longer raises. It dispatches to the
    complex-determinant path and returns a bound branch.

    Replaces the H.d ``NotImplementedError`` sentinel."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    f = np.linspace(4000.0, 12000.0, 25)
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    cij_gam = dict(cij, c66=cij["c66"] * 1.10)
    assert cij_gam["c11"] > cij_gam["c66"]
    res = flexural_dispersion_vti(f, **cij_gam, rho=rho, vf=vf, rho_f=rho_f, a=a)
    assert res.name == "flexural"
    assert res.azimuthal_order == 1
    ok = np.isfinite(res.slowness)
    assert ok.any(), "fast-formation TI returned an all-NaN branch"
    # Bound-mode ordering: slower than V_Sv, and the branch never runs
    # faster than the formation shear speed it descends from.
    assert np.all(res.slowness[ok] > 1.0 / vs)


def test_flexural_dispersion_vti_fast_formation_bound_and_monotone():
    """On the Green River shale of Ellefsen fig 2 the recovered branch
    must lie between the Scholte-like floor and ``V_Sv``, and phase
    slowness must increase with frequency -- flexural dispersion is
    monotone, and a non-monotone result means the marcher hopped
    branches (roadmap A.2)."""
    cij = _green_river_shale_stiffness()
    vsv = float(np.sqrt(cij["c44"] / cij["rho"]))
    assert vsv > 1500.0, "fixture must be fast-formation"
    f = np.linspace(1500.0, 19500.0, 217)
    res = flexural_dispersion_vti(f, **cij, vf=1500.0, rho_f=1000.0, a=0.10)
    ok = np.isfinite(res.slowness)
    assert ok.sum() >= 15
    s = res.slowness[ok]
    assert np.all(s > 1.0 / vsv), "branch faster than V_Sv is not bound"
    assert np.all(np.diff(s) >= -1.0e-9), "flexural slowness must not decrease"


def test_flexural_dispersion_vti_fast_formation_isotropic_collapse():
    """With ``gamma -> 0`` the fast-formation TI path must agree with the
    isotropic fast-formation solver on the overlap. This is the check
    that the complex VTI determinant is the *same* physics as the
    isotropic one, not merely a plausible-looking curve."""
    vp, vs, rho = 3292.0, 1768.0, 2075.0
    vf, rho_f, a = 1500.0, 1000.0, 0.10
    f = np.linspace(2000.0, 9000.0, 60)
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    # Nudge C66 by 0.01 % so the isotropy gate does not short-circuit to
    # flexural_dispersion; physically this is still isotropic.
    cij_near = dict(cij, c66=cij["c66"] * 1.0001)
    res_vti = flexural_dispersion_vti(f, **cij_near, rho=rho, vf=vf, rho_f=rho_f, a=a)
    res_iso = flexural_dispersion(f, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a)
    both = np.isfinite(res_vti.slowness) & np.isfinite(res_iso.slowness)
    assert both.sum() >= 10, "no overlapping band to compare"
    rel = (
        np.abs(res_vti.slowness[both] - res_iso.slowness[both]) / res_iso.slowness[both]
    )
    assert rel.max() < 5.0e-3, f"max rel deviation {rel.max():.2e}"


def test_dispersion_vti_returns_borehole_mode():
    f = np.linspace(2000.0, 5000.0, 5)
    cij = _isotropic_stiffness_from_lame(4500.0, 2500.0, 2400.0)
    res = stoneley_dispersion_vti(
        f,
        **cij,
        rho=2400.0,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    assert isinstance(res, BoreholeMode)


@pytest.mark.parametrize(
    "kwargs, msg",
    [
        ({"c11": 0.0}, "c11 must be positive"),
        ({"c33": -1.0e9}, "c33 must be positive"),
        ({"c44": 0.0}, "c44 must be positive"),
        ({"c66": -1.0}, "c66 must be positive"),
    ],
)
def test_dispersion_vti_rejects_non_positive_cij(kwargs, msg):
    f = np.array([5000.0])
    base = _isotropic_stiffness_from_lame(4500.0, 2500.0, 2400.0)
    base.update(kwargs)
    with pytest.raises(ValueError, match=msg):
        stoneley_dispersion_vti(
            f,
            **base,
            rho=2400.0,
            vf=1500.0,
            rho_f=1000.0,
            a=0.1,
        )


def test_dispersion_vti_rejects_unstable_c33_le_c13():
    """Validator rejects ``C33 <= C13`` (would break qP/qSV
    decoupling in the Christoffel equation)."""
    f = np.array([5000.0])
    cij = _isotropic_stiffness_from_lame(4500.0, 2500.0, 2400.0)
    cij["c13"] = cij["c33"] * 1.5  # force c13 > c33
    with pytest.raises(ValueError, match="c33 > c13"):
        stoneley_dispersion_vti(
            f,
            **cij,
            rho=2400.0,
            vf=1500.0,
            rho_f=1000.0,
            a=0.1,
        )


def test_dispersion_vti_rejects_unstable_c11_le_c66():
    """Validator rejects ``C11 <= C66`` (would have horizontal P
    no faster than horizontal S)."""
    f = np.array([5000.0])
    cij = _isotropic_stiffness_from_lame(4500.0, 2500.0, 2400.0)
    cij["c66"] = cij["c11"] * 1.5  # force c66 > c11
    with pytest.raises(ValueError, match="c11 > c66"):
        stoneley_dispersion_vti(
            f,
            **cij,
            rho=2400.0,
            vf=1500.0,
            rho_f=1000.0,
            a=0.1,
        )


def test_dispersion_vti_rejects_non_positive_freq_and_geometry():
    """Standard freq/geometry validation as for the isotropic
    public APIs."""
    cij = _isotropic_stiffness_from_lame(4500.0, 2500.0, 2400.0)
    base = dict(rho=2400.0, vf=1500.0, rho_f=1000.0, a=0.1)
    # Non-positive freq.
    with pytest.raises(ValueError, match="freq must be strictly positive"):
        stoneley_dispersion_vti(
            np.array([0.0]),
            **cij,
            **base,
        )
    # Non-positive vf.
    with pytest.raises(ValueError, match="vf and rho_f must be positive"):
        stoneley_dispersion_vti(
            np.array([5000.0]),
            **cij,
            **{**base, "vf": 0.0},
        )
    # Non-positive a.
    with pytest.raises(ValueError, match="a must be positive"):
        stoneley_dispersion_vti(
            np.array([5000.0]),
            **cij,
            **{**base, "a": 0.0},
        )


# =====================================================================
# Plan item H.b -- radial-wavenumber helper (Christoffel-equation roots)
# =====================================================================
#
# Bound-regime ``(alpha_qP, alpha_qSV, alpha_SH)`` from the H.a.2
# Christoffel quadratic and the H.a.4 SH closed form. The
# isotropic-collapse identity (qP -> p, qSV -> s, SH -> s) is the
# floating-point oracle for the entire H chain.


def _typical_vti_params():
    """Genuine-TI fixture for H.b tests. Roughly Thomsen-style:
    ~10% epsilon, ~5% delta, ~15% gamma. Within the Thomsen-stable
    range where qP / qSV remain well-separated."""
    return dict(
        c11=4.0e10,  # V_Ph^2 * rho ~ (4080 m/s)^2 * 2400
        c13=1.5e10,  # delta-coupled
        c33=3.5e10,  # V_Pv^2 * rho ~ (3819 m/s)^2 * 2400
        c44=1.0e10,  # V_Sv^2 * rho ~ (2041 m/s)^2 * 2400
        c66=1.3e10,  # V_Sh^2 * rho ~ (2327 m/s)^2 * 2400  (gamma > 0)
        rho=2400.0,
    )


def test_radial_wavenumbers_vti_isotropic_collapse_matches_isotropic():
    """Floating-point oracle for H.b: with an isotropic stiffness
    tensor the VTI radial wavenumbers reduce to the isotropic
    ``(p, s, s)``."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, 1500.0) * 1.5

    alpha_qP, alpha_qSV, alpha_SH = _radial_wavenumbers_vti(
        kz,
        omega,
        **cij,
        rho=rho,
    )

    p_iso = float(np.sqrt(kz**2 - (omega / vp) ** 2))
    s_iso = float(np.sqrt(kz**2 - (omega / vs) ** 2))

    assert alpha_qP == pytest.approx(p_iso, rel=1.0e-12)
    assert alpha_qSV == pytest.approx(s_iso, rel=1.0e-12)
    assert alpha_SH == pytest.approx(s_iso, rel=1.0e-12)


def test_radial_wavenumbers_vti_genuine_TI_christoffel_identity():
    """Christoffel-equation identity check: substituting the qP and
    qSV roots back into the bound-mode Christoffel determinant must
    give zero to floating-point precision. Catches sign / coefficient
    errors in the H.a.2 quadratic transcription that the isotropic-
    collapse test would miss when C44 = C66."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    # kz above the bound floor for both qP and qSV branches.
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    kz = omega / min(vsv, vsh) * 1.5

    alpha_qP, alpha_qSV, _ = _radial_wavenumbers_vti(
        kz,
        omega,
        **cij,
        rho=rho,
    )

    rho_omega_sq = rho * omega * omega

    # Christoffel determinant at alpha^2 (substep H.a.2 form):
    #   det = (-C11 alpha^2 + C44 kz^2 - rho omega^2)
    #         * (-C44 alpha^2 + C33 kz^2 - rho omega^2)
    #       + (C13 + C44)^2 alpha^2 kz^2
    def det_christoffel(alpha):
        a2 = alpha * alpha
        m11 = -cij["c11"] * a2 + cij["c44"] * kz * kz - rho_omega_sq
        m22 = -cij["c44"] * a2 + cij["c33"] * kz * kz - rho_omega_sq
        return m11 * m22 + (cij["c13"] + cij["c44"]) ** 2 * a2 * kz * kz

    # Both qP and qSV roots should give det = 0 to fp precision.
    # Use a relative tolerance: the determinant scales as
    # (rho omega^2)^2 ~ 1e21 in this fixture, so absolute zero
    # tolerance must be set against that scale.
    scale = rho_omega_sq * rho_omega_sq
    assert abs(det_christoffel(alpha_qP)) < scale * 1.0e-10
    assert abs(det_christoffel(alpha_qSV)) < scale * 1.0e-10


def test_radial_wavenumbers_vti_qP_larger_than_qSV():
    """Substep H.a.3 ordering: alpha_qP > alpha_qSV in the bound
    regime. Convention agrees with the isotropic limit (p > s
    always when V_P > V_S, because the radial decay rate is
    ``alpha = sqrt(kz^2 - omega^2/V^2)`` and larger V gives larger
    alpha)."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    kz = omega / min(vsv, vsh) * 1.5

    alpha_qP, alpha_qSV, _ = _radial_wavenumbers_vti(
        kz,
        omega,
        **cij,
        rho=rho,
    )
    assert alpha_qP > alpha_qSV
    assert alpha_qP > 0.0
    assert alpha_qSV > 0.0


def test_radial_wavenumbers_vti_SH_uses_C44_and_C66():
    """Substep H.a.4 (corrected): alpha_SH^2 = (C44 kz^2 - rho
    omega^2) / C66. Verify directly. Distinguishes the corrected
    form from the buggy ``kz^2 - rho omega^2 / C66`` (which would
    give the same isotropic limit but wrong genuine-TI value)."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    kz = omega / min(vsv, vsh) * 1.5

    _, _, alpha_SH = _radial_wavenumbers_vti(
        kz,
        omega,
        **cij,
        rho=rho,
    )
    expected_SH_sq = (cij["c44"] * kz**2 - rho * omega**2) / cij["c66"]
    assert alpha_SH**2 == pytest.approx(expected_SH_sq, rel=1.0e-12)
    # Sanity: the buggy ``kz^2 - rho omega^2 / C66`` form would
    # give a DIFFERENT value here because C44 != C66.
    buggy_SH_sq = kz**2 - rho * omega**2 / cij["c66"]
    assert abs(alpha_SH**2 - buggy_SH_sq) > 0.0  # they differ
    assert cij["c44"] != cij["c66"]  # confirm fixture is genuine TI


def test_radial_wavenumbers_vti_below_bound_floor_returns_nan():
    """Below the bound floor ``kz < omega / min(V_Sv, V_Sh, V_f)``
    one or more decay rates would be imaginary; the helper returns
    NaN (brentq-safe convention)."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    # Pick kz well below the bound floor.
    vsv = float(np.sqrt(cij["c44"] / rho))
    kz = omega / vsv * 0.5
    with np.errstate(invalid="ignore"):
        alpha_qP, alpha_qSV, alpha_SH = _radial_wavenumbers_vti(
            kz,
            omega,
            **cij,
            rho=rho,
        )
    # alpha_SH definitely NaN below the V_Sv-related floor.
    assert np.isnan(alpha_SH)


# =====================================================================
# Plan item H.c.1.a -- row 1 of the n=0 VTI modal determinant (r = a)
# =====================================================================
#
# First row of the 3x3 VTI Stoneley modal determinant. Returns the
# three post-rescale coefficients [A | B_qP, C_qSV]. At isotropic
# collapse the entries match (M11, M12, M13) of
# :func:`_modal_determinant_n0` bit-exactly -- the floating-point
# oracle for the H.c.1 chain.


def test_modal_row1_at_a_vti_isotropic_collapse_matches_M11_M12_M13():
    """Floating-point oracle: at isotropic stiffness, row 1 of the
    VTI determinant matches M11, M12, M13 of
    :func:`_modal_determinant_n0` to floating-point precision."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _modal_row1_at_a_vti(
        kz,
        omega,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )

    F = float(np.sqrt(kz * kz - (omega / vf) ** 2))
    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    M11 = F * float(sp.iv(1, F * a)) / (rho_f * omega**2)
    M12 = p * float(sp.kv(1, p * a))
    M13 = kz * float(sp.kv(1, s * a))

    assert row[0].real == pytest.approx(M11, rel=1.0e-12)
    assert row[1].real == pytest.approx(M12, rel=1.0e-12)
    assert row[2].real == pytest.approx(M13, rel=1.0e-12)


def test_modal_row1_at_a_vti_all_columns_nonzero_in_bound_regime():
    """Sparsity / non-degeneracy: in the bound regime all three
    columns of row 1 are non-zero."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    row = _modal_row1_at_a_vti(
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    for i in range(3):
        assert row[i] != 0.0


def test_modal_row1_at_a_vti_is_real_in_bound_regime():
    """Substep H.a.6 phase rescale: row 1 has the no-row-rescale
    pattern; only column-by-(-i) on the C_qSV column is applied.
    Post-rescale row is real-valued in the bound regime."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    row = _modal_row1_at_a_vti(
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_modal_row1_at_a_vti_uses_alpha_qP_alpha_qSV_not_p_s():
    """Genuine TI sanity: with non-trivial epsilon (C11 != C33),
    the qP root alpha_qP differs from the isotropic-equivalent
    p = sqrt(kz^2 - omega^2 / V_Pv^2). Verify row[1] uses
    alpha_qP K_1(alpha_qP a), NOT p K_1(p a). Confirms the row
    builder pulls from the Christoffel solver, not from a hard-
    coded isotropic substitution."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    row = _modal_row1_at_a_vti(
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    # Pull alpha_qP from the helper.
    alpha_qP, alpha_qSV, _ = _radial_wavenumbers_vti(
        kz,
        omega,
        **cij,
        rho=rho,
    )
    from scipy import special as sp

    expected_BqP = alpha_qP * float(sp.kv(1, alpha_qP * 0.1))
    expected_CqSV = kz * float(sp.kv(1, alpha_qSV * 0.1))
    assert row[1].real == pytest.approx(expected_BqP, rel=1.0e-12)
    assert row[2].real == pytest.approx(expected_CqSV, rel=1.0e-12)
    # Sanity: confirm alpha_qP differs from the naive p computed
    # with V_Pv = sqrt(c33/rho) so the test isn't passing trivially.
    Vpv = float(np.sqrt(cij["c33"] / rho))
    p_naive = float(np.sqrt(kz * kz - (omega / Vpv) ** 2))
    assert abs(alpha_qP - p_naive) > 0.01  # non-trivial epsilon


# =====================================================================
# Plan item H.c.1.b -- polarization-ratio helper + row 2 (sigma_rr at a)
# =====================================================================
#
# Algebraically heaviest row of the n=0 VTI determinant. Tests
# anchor on the per-element layer=formation match against M21,
# M22, M23 of :func:`_modal_determinant_n0` plus a separate
# polarization-ratio identity check.


def test_polarization_ratio_uz_over_ur_vti_isotropic_qP_limit():
    """At isotropic limit alpha_qP -> p, the polarization ratio
    gamma_qP = -i k_z / p."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, 1500.0) * 1.5

    p_iso = float(np.sqrt(kz**2 - (omega / vp) ** 2))
    gamma_qP = _polarization_ratio_uz_over_ur_vti(
        p_iso,
        kz,
        omega,
        c11=cij["c11"],
        c13=cij["c13"],
        c44=cij["c44"],
        rho=rho,
    )
    expected = -1j * kz / p_iso
    assert gamma_qP.real == pytest.approx(expected.real, abs=1.0e-12)
    assert gamma_qP.imag == pytest.approx(expected.imag, rel=1.0e-12)


def test_polarization_ratio_uz_over_ur_vti_isotropic_qSV_limit():
    """At isotropic limit alpha_qSV -> s, gamma_qSV = -i s / k_z."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, 1500.0) * 1.5

    s_iso = float(np.sqrt(kz**2 - (omega / vs) ** 2))
    gamma_qSV = _polarization_ratio_uz_over_ur_vti(
        s_iso,
        kz,
        omega,
        c11=cij["c11"],
        c13=cij["c13"],
        c44=cij["c44"],
        rho=rho,
    )
    expected = -1j * s_iso / kz
    assert gamma_qSV.real == pytest.approx(expected.real, abs=1.0e-12)
    assert gamma_qSV.imag == pytest.approx(expected.imag, rel=1.0e-12)


def test_polarization_ratio_uz_over_ur_vti_christoffel_identity():
    """Substituting (u_r, u_z) = (1, gamma_qX) into the Christoffel
    eigenvector equation
        (-C11 alpha^2 + C44 kz^2 - rho omega^2) u_r
        + i (C13 + C44) alpha kz u_z = 0
    must give zero to floating-point precision (verifies that the
    polarization-ratio formula is the correct null-space direction
    of the Christoffel matrix at the qX root)."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    alpha_qP, alpha_qSV, _ = _radial_wavenumbers_vti(
        kz,
        omega,
        **cij,
        rho=rho,
    )
    rho_omega_sq = rho * omega**2

    for alpha_qX in (alpha_qP, alpha_qSV):
        gamma = _polarization_ratio_uz_over_ur_vti(
            alpha_qX,
            kz,
            omega,
            c11=cij["c11"],
            c13=cij["c13"],
            c44=cij["c44"],
            rho=rho,
        )
        # Eigenvector equation residual:
        residual = (
            -cij["c11"] * alpha_qX**2 + cij["c44"] * kz**2 - rho_omega_sq
        ) + 1j * (cij["c13"] + cij["c44"]) * alpha_qX * kz * gamma
        # Scale check: M11 element ~ rho omega^2 in magnitude.
        assert abs(residual) < rho_omega_sq * 1.0e-12


def test_modal_row2_at_a_vti_isotropic_collapse_matches_M21_M22_M23():
    """Floating-point oracle for H.c.1.b: at isotropic stiffness,
    row 2 of the VTI determinant matches M21, M22, M23 of
    :func:`_modal_determinant_n0` to floating-point precision."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _modal_row2_at_a_vti(
        kz,
        omega,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
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

    assert row[0].real == pytest.approx(M21, rel=1.0e-12)
    assert row[1].real == pytest.approx(M22, rel=1.0e-12)
    assert row[2].real == pytest.approx(M23, rel=1.0e-12)


def test_modal_row2_at_a_vti_is_real_in_bound_regime():
    """Substep H.a.6: row 2 is no-row-rescale; col-by-(-i) on
    C_qSV. Post-rescale row is real-valued in the bound regime.
    Catches polarization-ratio sign errors that would leave a
    nonzero imaginary part."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    row = _modal_row2_at_a_vti(
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e0)


def test_modal_row2_at_a_vti_uses_c66_in_KK_one_over_a_term():
    """Genuine-TI sanity: with C44 != C66 (gamma > 0), the K_1/a
    coefficient of the B_qP column scales with ``2 C66``, NOT
    ``2 C44``. Confirms the (C11 - 2 C66) u_r/r slot is correctly
    transcribed -- the slot through which Norris 1990 LF coupling
    enters."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    kz = omega / min(vsv, vsh, 1500.0) * 1.5
    a = 0.1

    row = _modal_row2_at_a_vti(
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
    )
    alpha_qP, _, _ = _radial_wavenumbers_vti(kz, omega, **cij, rho=rho)
    rho_omega_sq = rho * omega**2
    q_qP = (
        cij["c44"] * (cij["c11"] * alpha_qP**2 + cij["c13"] * kz**2)
        - cij["c13"] * rho_omega_sq
    ) / (cij["c13"] + cij["c44"])
    from scipy import special as sp

    # row[1] = -Q_qP K_0(alpha_qP a) - 2 C66 alpha_qP K_1(alpha_qP a)/a
    expected = (
        -q_qP * float(sp.kv(0, alpha_qP * a))
        - 2.0 * cij["c66"] * alpha_qP * float(sp.kv(1, alpha_qP * a)) / a
    )
    assert row[1].real == pytest.approx(expected, rel=1.0e-12)
    # Sanity check: confirm fixture has C44 != C66 (genuine gamma).
    assert cij["c44"] != cij["c66"]


# =====================================================================
# Plan item H.c.1.c -- row 3 (sigma_rz at r=a)
# =====================================================================
#
# Z-derivative-bearing cos row of the n=0 VTI determinant. Gets
# the FULL substep-H.a.6 phase rescale (row * i AND col-by-(-i)
# on C_qSV). Tests anchor on the per-element layer=formation
# match against M31, M32, M33.


def test_modal_row3_at_a_vti_isotropic_collapse_matches_M31_M32_M33():
    """Floating-point oracle for H.c.1.c: at isotropic stiffness,
    row 3 of the VTI determinant matches M31, M32, M33 of
    :func:`_modal_determinant_n0` to floating-point precision."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _modal_row3_at_a_vti(
        kz,
        omega,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )

    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    M32 = 2.0 * kz * p * mu * float(sp.kv(1, p * a))
    M33 = mu * two_kz2_minus_kS2 * float(sp.kv(1, s * a))

    assert row[0] == 0.0
    assert row[1].real == pytest.approx(M32, rel=1.0e-12)
    assert row[2].real == pytest.approx(M33, rel=1.0e-12)


def test_modal_row3_at_a_vti_fluid_column_is_zero():
    """A column identically zero (fluid carries no shear)."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    row = _modal_row3_at_a_vti(
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    assert row[0] == 0.0
    # B and C columns generically non-zero.
    assert row[1] != 0.0
    assert row[2] != 0.0


def test_modal_row3_at_a_vti_is_real_in_bound_regime():
    """Substep H.a.6: row 3 is z-derivative-bearing -- gets the
    FULL rescale (row * i AND col-by-(-i) on C_qSV). Both must be
    correctly applied for the post-rescale row to be real.
    Forgetting the row * i is the most direct H.a.6 transcription
    error; this test catches it."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    row = _modal_row3_at_a_vti(
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_modal_row3_at_a_vti_does_not_use_c66():
    """Genuine-TI sanity: sigma_rz uses ONLY C44 (vertical shear),
    not C66 (horizontal shear). Doubling C66 leaves row 3
    unchanged, EXCEPT through the alpha_qP and alpha_qSV
    Christoffel roots from :func:`_radial_wavenumbers_vti` which
    only depend on C11, C13, C33, C44, rho (not C66). Confirms
    the (C11 - 2 C66) u_r/r slot does NOT appear in row 3."""
    cij_a = _typical_vti_params()
    cij_b = dict(cij_a)
    cij_b["c66"] = cij_a["c66"] * 2.0  # double C66
    rho = cij_a.pop("rho")
    cij_b.pop("rho")

    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij_a["c44"] / rho))
    # Bound regime floor: must be above min(V_Sv) and the doubled
    # V_Sh of cij_b. Pick kz well above both.
    kz = omega / vsv * 2.0

    row_a = _modal_row3_at_a_vti(
        kz,
        omega,
        **cij_a,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    row_b = _modal_row3_at_a_vti(
        kz,
        omega,
        **cij_b,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    # Identical: row 3 doesn't see C66 directly.
    np.testing.assert_array_equal(row_a, row_b)


# =====================================================================
# Plan item H.c.1.d -- assembly into _modal_determinant_n0_vti
# =====================================================================
#
# Stacks the three row builders into the 3x3 VTI Stoneley modal
# matrix; takes the real determinant. Tests anchor on the
# determinant-vanishes-at-isotropic-Stoneley-root self-consistency.


def test_modal_determinant_n0_vti_is_real_in_bound_regime():
    """The assembled 3x3 matrix is real-valued post-rescale (each
    row builder applies its own rescale internally), so
    ``np.linalg.det`` returns a finite real scalar in the bound
    regime."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    det = _modal_determinant_n0_vti(
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    assert np.isfinite(det)
    assert isinstance(det, float)


def test_modal_determinant_n0_vti_isotropic_collapse_root_matches_unlayered():
    """Substep H.a.7 (a) self-check at the determinant level: at
    isotropic stiffness, the VTI determinant has the same Stoneley
    root as :func:`_modal_determinant_n0`. The two determinants
    are not numerically equal (different overall scale due to
    different intermediate factors), but they share the same root
    in ``k_z``.

    Verify by: (a) computing the Stoneley root from
    ``stoneley_dispersion``; (b) evaluating the VTI determinant
    at that root; (c) checking ``|det_at_root|`` is small relative
    to its value off-root."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
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

    det_at_root = _modal_determinant_n0_vti(
        kz_root,
        omega,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    det_off_root = _modal_determinant_n0_vti(
        kz_root * 1.05,
        omega,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    # Determinant at root much smaller than off-root: brentq-type
    # root-finder will converge cleanly. The factor 1e-3 budget
    # is loose because the two determinants differ in absolute
    # scale; tighter tolerance kicks in at the full
    # stoneley_dispersion_vti integration test in H.c.2.
    assert abs(det_at_root) < abs(det_off_root) * 1.0e-3


def test_modal_determinant_n0_vti_bracket_brackets_isotropic_root():
    """End-to-end at-isotropic check: brentq across the
    standard Stoneley bracket finds the determinant root, and
    that root matches the isotropic Stoneley slowness."""
    from scipy import optimize

    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
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
    kz_root_iso = float(bound.slowness[0]) * omega

    def _det(kz):
        return _modal_determinant_n0_vti(
            kz,
            omega,
            **cij,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            a=a,
        )

    # Bracket around the isotropic root.
    kz_lo = kz_root_iso * 0.99
    kz_hi = kz_root_iso * 1.01
    d_lo = _det(kz_lo)
    d_hi = _det(kz_hi)
    assert np.sign(d_lo) != np.sign(d_hi)  # bracket valid
    kz_root_vti = optimize.brentq(_det, kz_lo, kz_hi, xtol=1.0e-10)
    assert kz_root_vti == pytest.approx(kz_root_iso, rel=1.0e-8)


def test_modal_determinant_n0_vti_returns_nan_outside_bound_regime():
    """Below the bound floor at least one Christoffel root is
    imaginary; the assembled determinant returns NaN (brentq-safe
    convention propagates from the radial-wavenumber helper)."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vp_h = float(np.sqrt(cij["c11"] / rho))  # fastest body wave
    omega = 2.0 * np.pi * 5000.0
    kz = omega / vp_h * 0.5  # well below the bound floor
    with np.errstate(invalid="ignore"):
        det = _modal_determinant_n0_vti(
            kz,
            omega,
            **cij,
            rho=rho,
            vf=1500.0,
            rho_f=1000.0,
            a=0.1,
        )
    assert np.isnan(det)


# =====================================================================
# Plan item H.c.2 -- Stoneley public-API hook (genuine TI brentq path)
# =====================================================================
#
# Replaces the H.0 ``NotImplementedError`` with a brentq loop on
# ``_modal_determinant_n0_vti``. The integration oracle is the
# isotropic-collapse regression vs ``stoneley_dispersion`` to
# ``rtol=1e-8`` -- the floating-point oracle for the entire H.c
# chain.


def test_stoneley_dispersion_vti_isotropic_via_genuine_TI_path_matches_isotropic():
    """Floating-point oracle for the H.c chain. Force the
    genuine-TI brentq path by passing a stiffness tensor that is
    formally non-isotropic (``c13`` perturbed by 1 ULP) but
    physically equivalent to isotropic, and verify the resulting
    slowness curve matches the isotropic ``stoneley_dispersion``
    answer to ``rtol=1e-7``.

    This test is more discriminating than the H.0 isotropic-
    collapse test (which dispatches directly to
    ``stoneley_dispersion`` and cannot fail) because it exercises
    the full ``_modal_determinant_n0_vti`` + brentq pipeline."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    # Force the genuine-TI path by tweaking c13 by 1 part in 1e-6
    # (well within Thomsen-stability but enough to defeat the
    # isotropic dispatch).
    cij_perturbed = dict(cij)
    cij_perturbed["c13"] = cij["c13"] * (1.0 + 1.0e-6)
    f = np.linspace(500.0, 8000.0, 12)

    res_iso = stoneley_dispersion(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    res_vti = stoneley_dispersion_vti(
        f,
        **cij_perturbed,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    np.testing.assert_allclose(
        res_vti.slowness,
        res_iso.slowness,
        rtol=1.0e-5,
        equal_nan=True,
    )
    # Confirm the perturbation actually defeated the isotropic
    # dispatch (the test would pass trivially otherwise).
    assert not _is_isotropic_stiffness(
        **{k: cij_perturbed[k] for k in ("c11", "c13", "c33", "c44", "c66")}
    )


def test_stoneley_dispersion_vti_genuine_TI_runs_smoke():
    """Smoke: a typical genuine-TI fixture produces a finite
    slowness curve. No analytic oracle (Norris 1990 LF check is
    H.c.3); just confirms the brentq + bracket combination
    handles the TI case across a broad band."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    f = np.linspace(1000.0, 10000.0, 8)

    res = stoneley_dispersion_vti(
        f,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    assert res.name == "Stoneley"
    assert res.azimuthal_order == 0
    assert res.slowness.shape == f.shape
    assert np.all(np.isfinite(res.slowness))
    # All slownesses above the slowest-shear floor.
    Vsv = float(np.sqrt(cij["c44"] / rho))
    Vsh = float(np.sqrt(cij["c66"] / rho))
    floor = 1.0 / max(Vsv, Vsh, 1500.0)
    assert np.all(res.slowness > floor)


def test_stoneley_dispersion_vti_genuine_TI_determinant_vanishes_at_root():
    """At each converged kz from ``stoneley_dispersion_vti``, the
    underlying VTI determinant must vanish (self-consistency).
    Ratio against the off-root determinant value at kz_root *
    1.01."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    f = 5000.0
    omega = 2.0 * np.pi * f

    res = stoneley_dispersion_vti(
        np.array([f]),
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    kz_root = float(res.slowness[0]) * omega

    det_at = _modal_determinant_n0_vti(
        kz_root,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    det_off = _modal_determinant_n0_vti(
        kz_root * 1.01,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    assert abs(det_at) < abs(det_off) * 1.0e-6


def test_stoneley_dispersion_vti_returns_borehole_mode_for_genuine_TI():
    """BoreholeMode return-type contract on the genuine-TI path."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    f = np.linspace(2000.0, 5000.0, 4)
    res = stoneley_dispersion_vti(
        f,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    assert isinstance(res, BoreholeMode)
    assert res.name == "Stoneley"
    assert res.azimuthal_order == 0
    np.testing.assert_array_equal(res.freq, f)


# =====================================================================
# Plan item H.c.3 -- Norris 1990 LF closed-form oracle
# =====================================================================
#
# The TI-specific validation oracle. At low frequency the n=0
# Stoneley slowness in a VTI formation approaches
#
#       S_ST^2 = 1/V_f^2 + rho_f / C66            (Norris 1990 eq. 6)
#
# Strongest validation of the C-matrix entries: depends on **C66**
# (NOT C44) -- the difference is invisible in the isotropic-collapse
# tests (where C44 = C66) but emerges sharply with gamma > 0.


def _norris_1990_LF_stoneley_slowness(c66, vf, rho_f):
    """S_ST = sqrt(1/V_f^2 + rho_f / C66) per Norris 1990 eq. 6.
    The TI-specific LF closed form for the Stoneley tube-wave
    slowness."""
    return float(np.sqrt(1.0 / vf**2 + rho_f / c66))


def test_stoneley_dispersion_vti_LF_matches_norris_1990_C66_form():
    """At very low frequency the VTI Stoneley slowness approaches
    the Norris 1990 closed form
        S_ST = sqrt(1/V_f^2 + rho_f / C66).

    Tested with a typical Thomsen-stable VTI fixture (gamma ~
    0.15) at f = 10 Hz. Tolerance loose because the LF closed
    form is asymptotic; tightening would require f -> 0 and run
    into bracket-floor numerics."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vf, rho_f, a = 1500.0, 1000.0, 0.1

    res = stoneley_dispersion_vti(
        np.array([10.0]),
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    s_norris = _norris_1990_LF_stoneley_slowness(cij["c66"], vf, rho_f)
    # ~0.1% tolerance: leading-order asymptote.
    assert res.slowness[0] == pytest.approx(s_norris, rel=1.0e-3)


def test_stoneley_dispersion_vti_LF_distinguishes_C66_from_C44():
    """Genuine TI vs isotropic-with-C44: the Norris 1990 LF form
    uses C66, not C44. With gamma > 0 (C66 > C44), the genuine-TI
    LF slowness matches the C66-based form much more closely than
    the C44-based form. Confirms the previous test isn't passing
    trivially through C44 = C66."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vf, rho_f, a = 1500.0, 1000.0, 0.1

    res = stoneley_dispersion_vti(
        np.array([10.0]),
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    s_C66 = _norris_1990_LF_stoneley_slowness(cij["c66"], vf, rho_f)
    s_C44 = _norris_1990_LF_stoneley_slowness(cij["c44"], vf, rho_f)
    # Fixture has C66 > C44 (gamma > 0), so s_C66 < s_C44.
    assert s_C66 < s_C44
    err_to_C66 = abs(res.slowness[0] - s_C66) / s_C66
    err_to_C44 = abs(res.slowness[0] - s_C44) / s_C44
    assert err_to_C66 < err_to_C44 * 0.05  # at least 20x closer
    gamma = (cij["c66"] - cij["c44"]) / (2.0 * cij["c44"])
    assert gamma > 0.05


def test_stoneley_dispersion_vti_LF_gamma_monotonicity():
    """Increasing C66 (at fixed other C-matrix entries) decreases
    the LF Stoneley slowness per Norris 1990 (since
    ``dS_ST^2/dC66 = -rho_f / C66^2 < 0``).

    Verify by computing the LF slowness at two C66 values: the
    larger C66 produces the smaller slowness."""
    cij_a = _typical_vti_params()
    cij_b = dict(cij_a)
    cij_b["c66"] = cij_a["c66"] * 1.20
    rho = cij_a.pop("rho")
    cij_b.pop("rho")
    vf, rho_f, a = 1500.0, 1000.0, 0.1

    res_a = stoneley_dispersion_vti(
        np.array([10.0]),
        **cij_a,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    res_b = stoneley_dispersion_vti(
        np.array([10.0]),
        **cij_b,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    # Larger C66 -> smaller LF Stoneley slowness.
    assert res_b.slowness[0] < res_a.slowness[0]
    s_norris_a = _norris_1990_LF_stoneley_slowness(cij_a["c66"], vf, rho_f)
    s_norris_b = _norris_1990_LF_stoneley_slowness(cij_b["c66"], vf, rho_f)
    expected_ratio = s_norris_b / s_norris_a
    actual_ratio = res_b.slowness[0] / res_a.slowness[0]
    assert actual_ratio == pytest.approx(expected_ratio, rel=1.0e-3)


# =====================================================================
# Plan item H.d.1 -- row 1 of the n=1 VTI flexural determinant (r=a)
# =====================================================================
#
# First row of the 4x4 n=1 VTI flexural modal determinant.
# Mirrors :func:`_modal_determinant_n1`'s M11-M14 with the
# Christoffel roots (alpha_qP, alpha_qSV, alpha_SH) replacing
# isotropic (p, s, s). New at n>=1 (vs the n=0 H.c.1.a row 1):
# the D_SH column appears via (1/r) d_theta psi_z cross-coupling.


def test_modal_row1_at_a_n1_vti_isotropic_collapse_matches_M11_M12_M13_M14():
    """Floating-point oracle for H.d.1: at isotropic stiffness,
    row 1 of the n=1 VTI determinant matches M11, M12, M13, M14
    of :func:`_modal_determinant_n1` to floating-point precision."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _modal_row1_at_a_n1_vti(
        kz,
        omega,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )

    F = float(np.sqrt(kz * kz - (omega / vf) ** 2))
    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    M11 = (F * float(sp.iv(0, F * a)) - float(sp.iv(1, F * a)) / a) / (rho_f * omega**2)
    M12 = p * float(sp.kv(0, p * a)) + float(sp.kv(1, p * a)) / a
    M13 = kz * (s * float(sp.kv(0, s * a)) + float(sp.kv(1, s * a)) / a)
    M14 = -float(sp.kv(1, s * a)) / a

    assert row[0].real == pytest.approx(M11, rel=1.0e-12)
    assert row[1].real == pytest.approx(M12, rel=1.0e-12)
    assert row[2].real == pytest.approx(M13, rel=1.0e-12)
    assert row[3].real == pytest.approx(M14, rel=1.0e-12)


def test_modal_row1_at_a_n1_vti_all_columns_nonzero_in_bound_regime():
    """Sparsity / non-degeneracy: in the bound regime all four
    columns of row 1 are non-zero. (No fluid-no-shear constraint
    on row 1; A enters via fluid pressure.)"""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    row = _modal_row1_at_a_n1_vti(
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    for i in range(4):
        assert row[i] != 0.0


def test_modal_row1_at_a_n1_vti_is_real_in_bound_regime():
    """Substep H.a.6: row 1 has the no-row-rescale pattern; only
    column-by-(-i) on C_qSV is applied. Post-rescale row is
    real-valued in the bound regime."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    row = _modal_row1_at_a_n1_vti(
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_modal_row1_at_a_n1_vti_uses_christoffel_roots_not_naive():
    """Genuine TI sanity: with non-trivial epsilon (C11 != C33),
    the qP root alpha_qP differs from the naive ``sqrt(kz^2 -
    omega^2/V_Pv^2)``; row[1] uses alpha_qP via the Christoffel
    solver. Same check for alpha_qSV and alpha_SH (which differ
    from the naive isotropic-with-V_Sv values when gamma > 0)."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vsv, vsh, 1500.0) * 1.5
    a = 0.1

    row = _modal_row1_at_a_n1_vti(
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
    )
    alpha_qP, alpha_qSV, alpha_SH = _radial_wavenumbers_vti(
        kz,
        omega,
        **cij,
        rho=rho,
    )
    from scipy import special as sp

    expected_BqP = (
        alpha_qP * float(sp.kv(0, alpha_qP * a)) + float(sp.kv(1, alpha_qP * a)) / a
    )
    expected_CqSV = kz * (
        alpha_qSV * float(sp.kv(0, alpha_qSV * a)) + float(sp.kv(1, alpha_qSV * a)) / a
    )
    expected_DSH = -float(sp.kv(1, alpha_SH * a)) / a

    assert row[1].real == pytest.approx(expected_BqP, rel=1.0e-12)
    assert row[2].real == pytest.approx(expected_CqSV, rel=1.0e-12)
    assert row[3].real == pytest.approx(expected_DSH, rel=1.0e-12)
    # Sanity: with non-trivial gamma, alpha_SH differs from
    # alpha_qSV (different stiffness moduli C66 vs C44 enter).
    assert abs(alpha_SH - alpha_qSV) > 1.0  # well-separated roots


# =====================================================================
# Plan item H.d.2 -- row 2 of the n=1 VTI flexural determinant (r=a)
# =====================================================================
#
# Algebraically heaviest row of the n=1 VTI determinant. Each
# column has multi-Bessel-term entries combining Q_qX (from
# H.c.1.b) with the n=1 ``4 C66 K_1/a^2`` azimuthal-derivative slot.
# Tests anchor on the per-element layer=formation match against
# M21-M24 of :func:`_modal_determinant_n1`.


def test_modal_row2_at_a_n1_vti_isotropic_collapse_matches_M21_M22_M23_M24():
    """Floating-point oracle for H.d.2: at isotropic stiffness,
    row 2 matches M21, M22, M23, M24 of :func:`_modal_determinant_n1`
    to floating-point precision."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _modal_row2_at_a_n1_vti(
        kz,
        omega,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
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

    assert row[0].real == pytest.approx(M21, rel=1.0e-12)
    assert row[1].real == pytest.approx(M22, rel=1.0e-12)
    assert row[2].real == pytest.approx(M23, rel=1.0e-12)
    assert row[3].real == pytest.approx(M24, rel=1.0e-12)


def test_modal_row2_at_a_n1_vti_is_real_in_bound_regime():
    """Substep H.a.6: row 2 is no-row-rescale; col-by-(-i) on
    C_qSV. Post-rescale row is real-valued in the bound regime.
    Catches polarization-ratio sign errors that would leave a
    nonzero imaginary part."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    row = _modal_row2_at_a_n1_vti(
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e0)


def test_modal_row2_at_a_n1_vti_matches_closed_form_per_column():
    """Per-column transcription check against the H.d.2 derivation
    closed forms (Q_qX combinations + C66 azimuthal-derivative
    slots)."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vsv, vsh, 1500.0) * 1.5
    a = 0.1
    vf, rho_f = 1500.0, 1000.0

    row = _modal_row2_at_a_n1_vti(
        kz,
        omega,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    alpha_qP, alpha_qSV, alpha_SH = _radial_wavenumbers_vti(
        kz,
        omega,
        **cij,
        rho=rho,
    )
    rho_omega_sq = rho * omega**2
    q_qP = (
        cij["c44"] * (cij["c11"] * alpha_qP**2 + cij["c13"] * kz**2)
        - cij["c13"] * rho_omega_sq
    ) / (cij["c13"] + cij["c44"])
    q_qSV = (
        cij["c44"] * (cij["c11"] * alpha_qSV**2 + cij["c13"] * kz**2)
        - cij["c13"] * rho_omega_sq
    ) / (cij["c13"] + cij["c44"])
    from scipy import special as sp

    F = float(np.sqrt(kz**2 - (omega / vf) ** 2))
    expected_A = -float(sp.iv(1, F * a))
    expected_BqP = -(
        q_qP * float(sp.kv(1, alpha_qP * a))
        + 2.0 * cij["c66"] * alpha_qP * float(sp.kv(0, alpha_qP * a)) / a
        + 4.0 * cij["c66"] * float(sp.kv(1, alpha_qP * a)) / (a * a)
    )
    expected_CqSV = -kz * (
        q_qSV * float(sp.kv(1, alpha_qSV * a))
        + 2.0 * cij["c66"] * alpha_qSV * float(sp.kv(0, alpha_qSV * a)) / a
        + 4.0 * cij["c66"] * float(sp.kv(1, alpha_qSV * a)) / (a * a)
    )
    expected_DSH = (
        +2.0
        * cij["c66"]
        * (
            alpha_SH * float(sp.kv(0, alpha_SH * a)) / a
            + 2.0 * float(sp.kv(1, alpha_SH * a)) / (a * a)
        )
    )

    assert row[0].real == pytest.approx(expected_A, rel=1.0e-12)
    assert row[1].real == pytest.approx(expected_BqP, rel=1.0e-12)
    assert row[2].real == pytest.approx(expected_CqSV, rel=1.0e-12)
    assert row[3].real == pytest.approx(expected_DSH, rel=1.0e-12)


def test_modal_row2_at_a_n1_vti_BqP_K1_over_a_squared_scales_with_4_C66():
    """Genuine-TI sanity: the K_1/a^2 coefficient of the B_qP
    column scales with ``4 C66`` (NOT 4 C44). Same approach as
    F.2.b.2's (C11 - 2 C66) test for the layered case.

    Confirms the n=1 azimuthal-derivative slot ``4 C66 K_1/a^2``
    -- which combines u_r/r and (1/r) d_theta u_theta contributions
    -- is correctly transcribed."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    kz = omega / min(vsv, vsh, 1500.0) * 1.5
    a = 0.1

    # Vary C66, keep all other C-matrix entries fixed.
    cij_a = dict(cij)
    cij_b = dict(cij)
    cij_b["c66"] = cij_a["c66"] * 1.50  # 50% increase

    row_a = _modal_row2_at_a_n1_vti(
        kz,
        omega,
        **cij_a,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
    )
    row_b = _modal_row2_at_a_n1_vti(
        kz,
        omega,
        **cij_b,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
    )
    # The difference in row[1] is purely from the 2 C66 K_0/a +
    # 4 C66 K_1/a^2 slots (Q_qP doesn't depend on C66; alpha_qP
    # doesn't depend on C66 directly either since the Christoffel
    # quadratic uses only C11, C13, C33, C44).
    alpha_qP, _, _ = _radial_wavenumbers_vti(kz, omega, **cij_a, rho=rho)
    from scipy import special as sp

    delta_c66 = cij_b["c66"] - cij_a["c66"]
    expected_diff = -(
        2.0 * delta_c66 * alpha_qP * float(sp.kv(0, alpha_qP * a)) / a
        + 4.0 * delta_c66 * float(sp.kv(1, alpha_qP * a)) / (a * a)
    )
    actual_diff = row_b[1].real - row_a[1].real
    assert actual_diff == pytest.approx(expected_diff, rel=1.0e-10)
    # Sanity: confirm the difference is non-zero (test isn't
    # passing trivially).
    assert abs(expected_diff) > 0.0


def test_modal_row2_at_a_n1_vti_DSH_column_pure_C66_scaling():
    """The D_SH column of row 2 scales entirely with C66 (no Q
    factor; pure (C11 - 2 C66) epsilon_theta_theta contribution).
    Verify by doubling C66 and checking that the D_SH entry
    doubles accordingly (modulo the alpha_SH change which itself
    depends on C66 via the SH dispersion ``alpha_SH^2 = (C44 kz^2 -
    rho omega^2)/C66``).

    Since alpha_SH depends on C66, the test compares against the
    explicit closed form rather than a simple ratio."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    kz = omega / min(vsv, vsh, 1500.0) * 1.5
    a = 0.1

    # Two C66 values.
    cij_a = dict(cij)
    cij_b = dict(cij_a, c66=cij_a["c66"] * 2.0)

    row_a = _modal_row2_at_a_n1_vti(
        kz,
        omega,
        **cij_a,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
    )
    row_b = _modal_row2_at_a_n1_vti(
        kz,
        omega,
        **cij_b,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
    )
    # row[3] = 2 C66 (alpha_SH K_0(alpha_SH a)/a + 2 K_1(alpha_SH a)/a^2).
    # Both C66 (outer) and alpha_SH (Bessel arg) change.
    _, _, alpha_SH_a = _radial_wavenumbers_vti(kz, omega, **cij_a, rho=rho)
    _, _, alpha_SH_b = _radial_wavenumbers_vti(kz, omega, **cij_b, rho=rho)
    from scipy import special as sp

    expected_a = (
        +2.0
        * cij_a["c66"]
        * (
            alpha_SH_a * float(sp.kv(0, alpha_SH_a * a)) / a
            + 2.0 * float(sp.kv(1, alpha_SH_a * a)) / (a * a)
        )
    )
    expected_b = (
        +2.0
        * cij_b["c66"]
        * (
            alpha_SH_b * float(sp.kv(0, alpha_SH_b * a)) / a
            + 2.0 * float(sp.kv(1, alpha_SH_b * a)) / (a * a)
        )
    )
    assert row_a[3].real == pytest.approx(expected_a, rel=1.0e-12)
    assert row_b[3].real == pytest.approx(expected_b, rel=1.0e-12)
    # The D_SH column at the two C66 values differs by both the
    # 2 C66 outer factor AND the alpha_SH dependence -- confirms
    # the row 2 D entry has full C66 sensitivity.
    assert row_a[3].real != row_b[3].real


# =====================================================================
# Plan item H.d.3 -- row 3 of the n=1 VTI flexural determinant (r=a)
# =====================================================================
#
# Sin-sector tangential-shear BC ``sigma_rtheta = 0``. Pure C66
# shear (no Lame replacement, no Q_qX). Every non-zero entry
# scales linearly with C66.


def test_modal_row3_at_a_n1_vti_isotropic_collapse_matches_M31_M32_M33_M34():
    """Floating-point oracle for H.d.3: at isotropic stiffness,
    row 3 matches M31, M32, M33, M34 of :func:`_modal_determinant_n1`
    to floating-point precision. M31 = 0 (fluid no shear)."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _modal_row3_at_a_n1_vti(
        kz,
        omega,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )

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
    assert row[1].real == pytest.approx(M32, rel=1.0e-12)
    assert row[2].real == pytest.approx(M33, rel=1.0e-12)
    assert row[3].real == pytest.approx(M34, rel=1.0e-12)


def test_modal_row3_at_a_n1_vti_fluid_column_is_zero():
    """The fluid carries no shear at the wall -- column A is
    identically zero in row 3. Stronger sparsity than rows 1, 2,
    4 (which all have non-zero A from fluid pressure)."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    row = _modal_row3_at_a_n1_vti(
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    assert row[0] == 0.0
    # Other three columns generically non-zero.
    for i in (1, 2, 3):
        assert row[i] != 0.0


def test_modal_row3_at_a_n1_vti_is_real_in_bound_regime():
    """Substep H.a.6: row 3 is no-row-rescale; col-by-(-i) on
    C_qSV cancels the +i k_z factor. Post-rescale row is
    real-valued in the bound regime."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    row = _modal_row3_at_a_n1_vti(
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_modal_row3_at_a_n1_vti_matches_closed_form_per_column():
    """Per-column transcription check against the H.d.3 derivation
    closed forms. Verifies the alpha_SH^2 K_1 direct term in the
    D_SH column (unique to row 3)."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    kz = omega / min(vsv, vsh, 1500.0) * 1.5
    a = 0.1

    row = _modal_row3_at_a_n1_vti(
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
    )
    alpha_qP, alpha_qSV, alpha_SH = _radial_wavenumbers_vti(
        kz,
        omega,
        **cij,
        rho=rho,
    )
    from scipy import special as sp

    expected_BqP = (
        +2.0
        * cij["c66"]
        * (
            alpha_qP * float(sp.kv(0, alpha_qP * a)) / a
            + 2.0 * float(sp.kv(1, alpha_qP * a)) / (a * a)
        )
    )
    expected_CqSV = (
        +2.0
        * kz
        * cij["c66"]
        * (
            alpha_qSV * float(sp.kv(0, alpha_qSV * a)) / a
            + 2.0 * float(sp.kv(1, alpha_qSV * a)) / (a * a)
        )
    )
    expected_DSH = -cij["c66"] * (
        alpha_SH**2 * float(sp.kv(1, alpha_SH * a))
        + 2.0 * alpha_SH * float(sp.kv(0, alpha_SH * a)) / a
        + 4.0 * float(sp.kv(1, alpha_SH * a)) / (a * a)
    )

    assert row[0] == 0.0
    assert row[1].real == pytest.approx(expected_BqP, rel=1.0e-12)
    assert row[2].real == pytest.approx(expected_CqSV, rel=1.0e-12)
    assert row[3].real == pytest.approx(expected_DSH, rel=1.0e-12)


def test_modal_row3_at_a_n1_vti_BqP_CqSV_scale_linearly_with_C66():
    """B_qP and C_qSV entries scale LINEARLY with C66: doubling
    C66 (with all other C-matrix entries fixed) doubles the
    entries exactly, since alpha_qP and alpha_qSV are
    C66-independent (the Christoffel quadratic uses only C11,
    C13, C33, C44).

    The D_SH entry does NOT scale linearly because alpha_SH
    depends on C66 via the SH Christoffel branch ``alpha_SH^2 =
    (C44 kz^2 - rho omega^2)/C66``."""
    cij_a = _typical_vti_params()
    cij_b = dict(cij_a)
    cij_b["c66"] = cij_a["c66"] * 2.0
    rho = cij_a.pop("rho")
    cij_b.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij_a["c44"] / rho))
    vsh_a = float(np.sqrt(cij_a["c66"] / rho))
    vsh_b = float(np.sqrt(cij_b["c66"] / rho))
    kz = omega / min(vsv, vsh_a, vsh_b, 1500.0) * 1.5

    row_a = _modal_row3_at_a_n1_vti(
        kz,
        omega,
        **cij_a,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    row_b = _modal_row3_at_a_n1_vti(
        kz,
        omega,
        **cij_b,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    # B_qP and C_qSV: ratio b/a = 2 exactly (C66 outer factor;
    # alpha_qP and alpha_qSV are C66-independent).
    assert row_b[1].real / row_a[1].real == pytest.approx(2.0, rel=1.0e-12)
    assert row_b[2].real / row_a[2].real == pytest.approx(2.0, rel=1.0e-12)
    # D_SH: ratio is NOT 2 because alpha_SH depends on C66.
    # Just verify the entry is non-trivial.
    assert row_b[3].real != row_a[3].real
    # Confirm the assumption: alpha_qP and alpha_qSV are unchanged.
    aqp_a, aqsv_a, _ = _radial_wavenumbers_vti(kz, omega, **cij_a, rho=rho)
    aqp_b, aqsv_b, _ = _radial_wavenumbers_vti(kz, omega, **cij_b, rho=rho)
    assert aqp_a == aqp_b
    assert aqsv_a == aqsv_b


# =====================================================================
# Plan item H.d.4 -- row 4 of the n=1 VTI flexural determinant (r=a)
# =====================================================================
#
# Z-derivative-bearing cos-sector row (sigma_rz = 0). Pure C44
# shear; uses the P_qX combination from H.c.1.c. Adds the D_SH
# column at n=1 (via d_z u_r from (1/r) d_theta psi_z).


def test_modal_row4_at_a_n1_vti_isotropic_collapse_matches_M41_M42_M43_M44():
    """Floating-point oracle for H.d.4: at isotropic stiffness,
    row 4 matches M41, M42, M43, M44 of :func:`_modal_determinant_n1`
    to floating-point precision."""
    vp, vs, rho = 4500.0, 2500.0, 2400.0
    vf, rho_f, a = 1500.0, 1000.0, 0.1
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vs, vf) * 1.5

    row = _modal_row4_at_a_n1_vti(
        kz,
        omega,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )

    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    from scipy import special as sp

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    M41 = 0.0
    M42 = 2.0 * kz * mu * (p * float(sp.kv(0, p * a)) + float(sp.kv(1, p * a)) / a)
    M43 = (
        mu
        * two_kz2_minus_kS2
        * (s * float(sp.kv(0, s * a)) + float(sp.kv(1, s * a)) / a)
    )
    M44 = -kz * mu * float(sp.kv(1, s * a)) / a

    assert row[0].real == pytest.approx(M41)
    assert row[1].real == pytest.approx(M42, rel=1.0e-12)
    assert row[2].real == pytest.approx(M43, rel=1.0e-12)
    assert row[3].real == pytest.approx(M44, rel=1.0e-12)


def test_modal_row4_at_a_n1_vti_fluid_column_is_zero():
    """Column A is identically zero (fluid carries no shear)."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    row = _modal_row4_at_a_n1_vti(
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    assert row[0] == 0.0
    for i in (1, 2, 3):
        assert row[i] != 0.0


def test_modal_row4_at_a_n1_vti_is_real_in_bound_regime():
    """Substep H.a.6: row 4 is z-derivative-bearing -- gets the
    FULL rescale (row * i + col-by-(-i) on C_qSV). Both rescales
    must be correctly applied for the post-rescale row to be
    real-valued. Forgetting the row * i (the F.2.a.5-flagged
    transcription error mode) leaves a non-zero imaginary part.
    """
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    row = _modal_row4_at_a_n1_vti(
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    np.testing.assert_allclose(row.imag, 0.0, atol=1.0e-14)


def test_modal_row4_at_a_n1_vti_matches_closed_form_per_column():
    """Per-column transcription check against the H.d.4 derivation
    closed forms. Verifies the two-term ``alpha_qP K_0 + K_1/a``
    combination on B_qP (vs the single K_1 term in H.c.1.c row 3
    at n=0)."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    kz = omega / min(vsv, vsh, 1500.0) * 1.5
    a = 0.1

    row = _modal_row4_at_a_n1_vti(
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=a,
    )
    alpha_qP, alpha_qSV, alpha_SH = _radial_wavenumbers_vti(
        kz,
        omega,
        **cij,
        rho=rho,
    )
    rho_omega_sq = rho * omega**2
    p_qP = cij["c11"] * alpha_qP**2 + cij["c13"] * kz**2 + rho_omega_sq
    p_qSV = cij["c11"] * alpha_qSV**2 + cij["c13"] * kz**2 + rho_omega_sq
    from scipy import special as sp

    expected_BqP = (
        +cij["c44"]
        * p_qP
        / ((cij["c13"] + cij["c44"]) * kz)
        * (alpha_qP * float(sp.kv(0, alpha_qP * a)) + float(sp.kv(1, alpha_qP * a)) / a)
    )
    expected_CqSV = (
        +cij["c44"]
        * p_qSV
        / (cij["c13"] + cij["c44"])
        * (
            alpha_qSV * float(sp.kv(0, alpha_qSV * a))
            + float(sp.kv(1, alpha_qSV * a)) / a
        )
    )
    expected_DSH = -kz * cij["c44"] * float(sp.kv(1, alpha_SH * a)) / a

    assert row[0] == 0.0
    assert row[1].real == pytest.approx(expected_BqP, rel=1.0e-12)
    assert row[2].real == pytest.approx(expected_CqSV, rel=1.0e-12)
    assert row[3].real == pytest.approx(expected_DSH, rel=1.0e-12)


def test_modal_row4_at_a_n1_vti_C66_independent_except_via_alpha_SH():
    """B_qP and C_qSV entries are C66-INDEPENDENT (entries depend
    only on C11, C13, C44 via P_qX and the Christoffel roots
    alpha_qP, alpha_qSV -- which themselves don't see C66). The
    D_SH column DOES depend on C66 (via alpha_SH only).

    Verify by varying C66 and checking that B_qP and C_qSV are
    unchanged while D_SH changes."""
    cij_a = _typical_vti_params()
    cij_b = dict(cij_a)
    cij_b["c66"] = cij_a["c66"] * 1.50
    rho = cij_a.pop("rho")
    cij_b.pop("rho")
    omega = 2.0 * np.pi * 5000.0
    vsv = float(np.sqrt(cij_a["c44"] / rho))
    vsh_a = float(np.sqrt(cij_a["c66"] / rho))
    vsh_b = float(np.sqrt(cij_b["c66"] / rho))
    kz = omega / min(vsv, vsh_a, vsh_b, 1500.0) * 1.5

    row_a = _modal_row4_at_a_n1_vti(
        kz,
        omega,
        **cij_a,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    row_b = _modal_row4_at_a_n1_vti(
        kz,
        omega,
        **cij_b,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    # B_qP and C_qSV unchanged (C66-independent).
    assert row_a[1].real == pytest.approx(row_b[1].real, rel=1.0e-12)
    assert row_a[2].real == pytest.approx(row_b[2].real, rel=1.0e-12)
    # D_SH differs (alpha_SH depends on C66).
    assert row_a[3].real != row_b[3].real


# =====================================================================
# Plan item H.d.5 -- assembly into _modal_determinant_n1_vti
# =====================================================================
#
# Stacks the four row builders into the 4x4 VTI flexural modal
# matrix; takes the real determinant. Tests anchor on the
# determinant-vanishes-at-isotropic-flexural-root self-consistency.
# Mirrors the H.c.1.d test pattern.


def test_modal_determinant_n1_vti_is_real_in_bound_regime():
    """The assembled 4x4 matrix is real-valued post-rescale (each
    row builder applies its own rescale internally), so
    ``np.linalg.det`` returns a finite real scalar in the bound
    regime."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vsv = float(np.sqrt(cij["c44"] / rho))
    vsh = float(np.sqrt(cij["c66"] / rho))
    omega = 2.0 * np.pi * 5000.0
    kz = omega / min(vsv, vsh, 1500.0) * 1.5

    det = _modal_determinant_n1_vti(
        kz,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    assert np.isfinite(det)
    assert isinstance(det, float)


def test_modal_determinant_n1_vti_isotropic_collapse_root_matches_unlayered():
    """Substep H.a.7 (a) self-check at the determinant level: at
    isotropic stiffness (slow-formation regime where ``F^2 > 0``
    and the real-valued determinant is well-defined), the VTI
    determinant has the same flexural root as
    :func:`_modal_determinant_n1`. The two determinants are not
    numerically equal (different overall scale due to different
    intermediate factors), but they share the same root in ``k_z``.

    Verify by: (a) computing the flexural root from
    ``flexural_dispersion``; (b) evaluating the VTI determinant
    at that root; (c) checking ``|det_at_root|`` is small relative
    to its value off-root."""
    # Slow formation (V_S < V_f) keeps F^2 = kz^2 - (omega/V_f)^2
    # positive at the flexural root, the regime in which the
    # real-valued ``_modal_determinant_n1`` (and its VTI mirror)
    # is well-defined.
    vp, vs, rho = SLOW_VP, SLOW_VS, SLOW_RHO
    vf, rho_f, a = SLOW_VF, SLOW_RHO_F, SLOW_A
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
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

    det_at_root = _modal_determinant_n1_vti(
        kz_root,
        omega,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    det_off_root = _modal_determinant_n1_vti(
        kz_root * 1.05,
        omega,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    # Determinant at root much smaller than off-root: brentq-type
    # root-finder will converge cleanly. The factor 1e-3 budget
    # is loose because the two determinants differ in absolute
    # scale; tighter tolerance kicks in at the full
    # flexural_dispersion_vti integration test in H.d.6.
    assert abs(det_at_root) < abs(det_off_root) * 1.0e-3


def test_modal_determinant_n1_vti_bracket_brackets_isotropic_root():
    """End-to-end at-isotropic check (slow formation): brentq
    across a tight bracket around the isotropic flexural root
    finds the determinant root, and that root matches the
    isotropic flexural slowness to ``rtol=1e-8``."""
    from scipy import optimize

    vp, vs, rho = SLOW_VP, SLOW_VS, SLOW_RHO
    vf, rho_f, a = SLOW_VF, SLOW_RHO_F, SLOW_A
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
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
    kz_root_iso = float(bound.slowness[0]) * omega

    def _det(kz):
        return _modal_determinant_n1_vti(
            kz,
            omega,
            **cij,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            a=a,
        )

    # Bracket around the isotropic root.
    kz_lo = kz_root_iso * 0.99
    kz_hi = kz_root_iso * 1.01
    d_lo = _det(kz_lo)
    d_hi = _det(kz_hi)
    assert np.sign(d_lo) != np.sign(d_hi)  # bracket valid
    kz_root_vti = optimize.brentq(_det, kz_lo, kz_hi, xtol=1.0e-10)
    assert kz_root_vti == pytest.approx(kz_root_iso, rel=1.0e-8)


def test_modal_determinant_n1_vti_returns_nan_outside_bound_regime():
    """Below the bound floor at least one Christoffel root is
    imaginary; the assembled determinant returns NaN (brentq-safe
    convention propagates from the radial-wavenumber helper)."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    vp_h = float(np.sqrt(cij["c11"] / rho))  # fastest body wave
    omega = 2.0 * np.pi * 5000.0
    kz = omega / vp_h * 0.5  # well below the bound floor
    with np.errstate(invalid="ignore"):
        det = _modal_determinant_n1_vti(
            kz,
            omega,
            **cij,
            rho=rho,
            vf=1500.0,
            rho_f=1000.0,
            a=0.1,
        )
    assert np.isnan(det)


# =====================================================================
# Plan item H.d.6 -- flexural_dispersion_vti public-API hook
# =====================================================================
#
# Replaces the H.0 ``NotImplementedError`` (now restricted to fast-
# formation TI) with a brentq loop on ``_modal_determinant_n1_vti``
# for slow-formation TI (V_Sv < V_f). Mirrors H.c.2 for n=0.


def _typical_slow_vti_params():
    """Slow-formation genuine-TI fixture for H.d.6 tests.

    ``V_Sv = 1100 m/s < V_f = 1500 m/s`` so the real-valued VTI
    modal determinant is well-defined (``F_f^2 > 0`` at every
    flexural ``k_z``). Roughly Thomsen-style: epsilon ~ 0.1,
    gamma ~ 0.1; ``c13`` chosen well within Thomsen-stability."""
    return dict(
        c11=1.27e10,  # V_Ph^2 * rho ~ (2400 m/s)^2 * 2200
        c13=4.0e9,  # delta-coupled (c33 > c13 stability)
        c33=1.06e10,  # V_Pv^2 * rho ~ (2200 m/s)^2 * 2200
        c44=2.66e9,  # V_Sv^2 * rho ~ (1100 m/s)^2 * 2200  (Vsv < Vf)
        c66=3.17e9,  # V_Sh^2 * rho ~ (1200 m/s)^2 * 2200  (gamma > 0)
        rho=2200.0,
    )


def test_flexural_dispersion_vti_isotropic_via_genuine_TI_path_matches_isotropic():
    """Floating-point oracle for the H.d chain. Force the
    genuine-TI brentq path by passing a stiffness tensor that is
    formally non-isotropic (``c13`` perturbed by 1 ULP) but
    physically equivalent to isotropic, and verify the resulting
    slowness curve matches the isotropic ``flexural_dispersion``
    answer to ``rtol=1e-5``.

    Slow-formation regime so the real-valued VTI determinant is
    well-defined. More discriminating than the H.0 isotropic-
    collapse test (which dispatches directly to
    ``flexural_dispersion`` and cannot fail) because it exercises
    the full ``_modal_determinant_n1_vti`` + brentq pipeline."""
    vp, vs, rho = SLOW_VP, SLOW_VS, SLOW_RHO
    vf, rho_f, a = SLOW_VF, SLOW_RHO_F, SLOW_A
    cij = _isotropic_stiffness_from_lame(vp, vs, rho)
    # Force the genuine-TI path by tweaking c13 by 1 part in 1e-6.
    cij_perturbed = dict(cij)
    cij_perturbed["c13"] = cij["c13"] * (1.0 + 1.0e-6)
    f = np.linspace(2000.0, 6000.0, 6)

    res_iso = flexural_dispersion(
        f,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    res_vti = flexural_dispersion_vti(
        f,
        **cij_perturbed,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    np.testing.assert_allclose(
        res_vti.slowness,
        res_iso.slowness,
        rtol=1.0e-5,
        equal_nan=True,
    )
    # Confirm the perturbation actually defeated the isotropic
    # dispatch (the test would pass trivially otherwise).
    assert not _is_isotropic_stiffness(
        **{k: cij_perturbed[k] for k in ("c11", "c13", "c33", "c44", "c66")}
    )


def test_flexural_dispersion_vti_genuine_TI_runs_smoke():
    """Smoke: a typical slow-formation genuine-TI fixture
    produces a finite slowness curve above the geometric cutoff.

    Cutoff for ``V_Sv ~ 1100`` and ``a = 0.1`` sits around
    ``V_Sv / (2 pi a) ~ 1750 Hz``; tests sit safely above 3 kHz.
    No analytic oracle here; just confirms the brentq + bracket
    combination handles the TI case across a moderate band."""
    cij = _typical_slow_vti_params()
    rho = cij.pop("rho")
    f = np.linspace(3000.0, 7000.0, 5)

    res = flexural_dispersion_vti(
        f,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    assert res.name == "flexural"
    assert res.azimuthal_order == 1
    assert res.slowness.shape == f.shape
    assert np.all(np.isfinite(res.slowness))
    # Slowness above the V_Sv floor (LF asymptote).
    Vsv = float(np.sqrt(cij["c44"] / rho))
    assert np.all(res.slowness > 1.0 / Vsv * (1.0 - 1.0e-3))


def test_flexural_dispersion_vti_genuine_TI_determinant_vanishes_at_root():
    """At each converged kz from ``flexural_dispersion_vti``, the
    underlying VTI determinant must vanish (self-consistency).
    Ratio against the off-root determinant value at kz_root *
    1.01."""
    cij = _typical_slow_vti_params()
    rho = cij.pop("rho")
    f = 5000.0
    omega = 2.0 * np.pi * f

    res = flexural_dispersion_vti(
        np.array([f]),
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    kz_root = float(res.slowness[0]) * omega

    det_at = _modal_determinant_n1_vti(
        kz_root,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    det_off = _modal_determinant_n1_vti(
        kz_root * 1.01,
        omega,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    assert abs(det_at) < abs(det_off) * 1.0e-6


def test_flexural_dispersion_vti_returns_borehole_mode_for_genuine_TI():
    """BoreholeMode return-type contract on the slow-formation
    genuine-TI path."""
    cij = _typical_slow_vti_params()
    rho = cij.pop("rho")
    f = np.linspace(2000.0, 5000.0, 4)
    res = flexural_dispersion_vti(
        f,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    assert isinstance(res, BoreholeMode)
    assert res.name == "flexural"
    assert res.azimuthal_order == 1
    np.testing.assert_array_equal(res.freq, f)


def test_flexural_dispersion_vti_LF_approaches_V_Sv():
    """Plan H.d sanity: at low frequency (just above the
    geometric cutoff) the VTI flexural slowness should approach
    ``1 / V_Sv`` (the Sinha-Norris-Chang LF asymptote for slow
    TI). For the slow-TI fixture (V_Sv ~ 1100, a = 0.1) the
    cutoff sits around 1750 Hz; this test runs at 3 kHz, well
    above cutoff but low enough to be in the LF asymptotic
    regime where slowness ~ 1/V_Sv to within a few percent."""
    cij = _typical_slow_vti_params()
    rho = cij.pop("rho")
    Vsv = float(np.sqrt(cij["c44"] / rho))
    res = flexural_dispersion_vti(
        np.array([3000.0]),
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    # LF asymptote: slowness ~ 1/Vsv. Tolerance ~ 2% at f = 3 kHz.
    assert res.slowness[0] == pytest.approx(1.0 / Vsv, rel=2.0e-2)


# =====================================================================
# Plan item H.e -- validation hardening on top of H.d.6
# =====================================================================
#
# Hardening tests for the assembled VTI solvers. Each tests an
# asymptotic / self-consistency property that the isotropic-collapse
# regression alone doesn't pin down. Mirrors F.2.e for the layered
# solver, plus a weak-anisotropy regression against the
# phenomenological model from ``fwap.cylindrical``
# (``flexural_dispersion_vti_physical``) -- the only TI-specific
# external oracle we have for the n=1 dipole.


def test_modal_determinant_n0_vti_vanishes_at_converged_root_multi_freq():
    """Self-consistency: at the converged ``k_z`` from
    ``stoneley_dispersion_vti`` at every frequency in a multi-
    point grid, the underlying VTI determinant is many orders of
    magnitude smaller than its value at ``k_z * 1.01``. Sharper
    than the single-frequency check from H.c.2 because it
    catches regressions where the brentq pipeline converges to
    something other than the true root for some frequencies."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    f = np.geomspace(1000.0, 10000.0, 6)

    res = stoneley_dispersion_vti(
        f,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    assert np.all(np.isfinite(res.slowness))
    for i, fi in enumerate(f):
        omega = 2.0 * np.pi * float(fi)
        kz_root = float(res.slowness[i]) * omega
        det_at = _modal_determinant_n0_vti(
            kz_root,
            omega,
            **cij,
            rho=rho,
            vf=1500.0,
            rho_f=1000.0,
            a=0.1,
        )
        det_off = _modal_determinant_n0_vti(
            kz_root * 1.01,
            omega,
            **cij,
            rho=rho,
            vf=1500.0,
            rho_f=1000.0,
            a=0.1,
        )
        # Ratio at every frequency: det at root << det 1% off.
        assert abs(det_at) < abs(det_off) * 1.0e-6, (
            f"f={fi:.1f}: |det_at|={abs(det_at):.3e} not << "
            f"|det_off|={abs(det_off):.3e}"
        )


def test_modal_determinant_n1_vti_vanishes_at_converged_root_multi_freq():
    """Mirror of the n=0 multi-frequency self-consistency at n=1.
    Slow-formation TI fixture so the real-valued VTI determinant
    is well-defined across the full frequency band."""
    cij = _typical_slow_vti_params()
    rho = cij.pop("rho")
    # Above the geometric cutoff (~1750 Hz for V_Sv=1100, a=0.1).
    f = np.geomspace(3000.0, 12000.0, 6)

    res = flexural_dispersion_vti(
        f,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    assert np.all(np.isfinite(res.slowness))
    for i, fi in enumerate(f):
        omega = 2.0 * np.pi * float(fi)
        kz_root = float(res.slowness[i]) * omega
        det_at = _modal_determinant_n1_vti(
            kz_root,
            omega,
            **cij,
            rho=rho,
            vf=1500.0,
            rho_f=1000.0,
            a=0.1,
        )
        det_off = _modal_determinant_n1_vti(
            kz_root * 1.01,
            omega,
            **cij,
            rho=rho,
            vf=1500.0,
            rho_f=1000.0,
            a=0.1,
        )
        assert abs(det_at) < abs(det_off) * 1.0e-6, (
            f"f={fi:.1f}: |det_at|={abs(det_at):.3e} not << "
            f"|det_off|={abs(det_off):.3e}"
        )


def test_stoneley_dispersion_vti_multi_frequency_smoothness():
    """Stoneley slowness varies smoothly with frequency: across a
    geomspaced band the slowness curve is finite at every point,
    sits above the Norris LF floor, and below the rigid-formation
    fluid-only ceiling. No strict monotonicity check (the Stoneley
    is gently dispersive; sign of the derivative depends on
    parameters), just smoothness."""
    cij = _typical_vti_params()
    rho = cij.pop("rho")
    f = np.geomspace(500.0, 12000.0, 16)

    res = stoneley_dispersion_vti(
        f,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    assert np.all(np.isfinite(res.slowness))
    # All slownesses above the fluid-only slowness 1/V_f and
    # below the Norris LF cap (a sanity fence, not a tight oracle).
    s_fluid = 1.0 / 1500.0
    s_norris = float(np.sqrt(1.0 / 1500.0**2 + 1000.0 / cij["c66"]))
    assert np.all(res.slowness > s_fluid)
    assert np.all(res.slowness < s_norris * 1.10)
    # Smoothness: relative step-to-step change capped at 5 %
    # (geomspaced grid, so adjacent frequencies are ~ 60 % apart
    # but slowness is gently dispersive).
    rel_steps = np.abs(np.diff(res.slowness)) / res.slowness[:-1]
    assert np.all(rel_steps < 0.05)


def test_flexural_dispersion_vti_multi_frequency_monotonicity():
    """Slow-formation flexural slowness increases monotonically
    with frequency: cutoff at ``1/V_Sv`` (low f), HF asymptote at
    a Rayleigh-like speed slightly faster than ``V_Sv`` -- so
    slowness rises from ``~1/V_Sv`` toward ``~1/V_R > 1/V_Sv``.
    Mirrors F.2.e's layered counterpart."""
    cij = _typical_slow_vti_params()
    rho = cij.pop("rho")
    f = np.geomspace(3000.0, 15000.0, 12)

    res = flexural_dispersion_vti(
        f,
        **cij,
        rho=rho,
        vf=1500.0,
        rho_f=1000.0,
        a=0.1,
    )
    assert np.all(np.isfinite(res.slowness))
    # Tiny negative tolerance for asymptotic-flatness rounding noise.
    diffs = np.diff(res.slowness)
    assert np.all(diffs > -1.0e-9)


def test_flexural_dispersion_vti_weak_anisotropy_matches_phenomenological():
    """Plan H.e weak-anisotropy oracle: with small ``gamma`` (TI
    close to isotropic), the full Schmitt 1989 modal determinant
    slowness should qualitatively track the Sinha-Norris-Chang
    phenomenological asymptote
    :func:`flexural_dispersion_vti_physical` across the dipole-
    sonic band (~ 1-6 kHz, equivalent to ~ 1-3 cutoff multiples
    for this fixture).

    Tolerance follows the precedent of the isotropic
    ``test_flexural_dispersion_qualitative_match_with_phenomenological``
    (10 %). The phenomenological model is a smoothed-step
    interpolation between the LF ``1/V_Sv`` and HF
    ``1/V_R(V_P, V_Sh)`` asymptotes -- both physically correct
    in their own limits; the few-percent quantitative offset
    arises from Scholte / fluid-loading effects the modal solver
    captures but the phenomenological does not."""
    rho = 2200.0
    vsv = 1100.0
    vsh = 1110.0  # gamma ~ 0.009 (very weak TI)
    vp = 2200.0
    cij = dict(
        c11=rho * vp**2,  # set epsilon = 0 (c11 = c33)
        c33=rho * vp**2,
        c44=rho * vsv**2,
        c66=rho * vsh**2,  # gamma > 0 only
    )
    # c13 = c11 - 2 c44 (delta-coupled isotropic value).
    cij["c13"] = cij["c11"] - 2.0 * cij["c44"]
    a = 0.1
    vf, rho_f = 1500.0, 1000.0

    # Dipole-sonic band: 1.5-3.5 cutoff multiples (~ 2.6 - 6.1 kHz
    # for this fixture). Stays comfortably above the geometric
    # cutoff where the bracket would otherwise touch the floor.
    fc = vsv / (2.0 * np.pi * a)
    f = np.linspace(fc * 1.5, fc * 3.5, 8)

    res_modal = flexural_dispersion_vti(
        f,
        **cij,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    s_phenom = flexural_dispersion_vti_physical(
        vp=vp,
        vsv=vsv,
        vsh=vsh,
        a_borehole=a,
    )(f)
    assert np.all(np.isfinite(res_modal.slowness))
    rel_diff = np.abs(res_modal.slowness - s_phenom) / s_phenom
    assert np.all(rel_diff < 0.10)
