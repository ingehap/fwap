"""VTI (transversely isotropic) formation modal determinants and dispersion.

Extracted from ``fwap.cylindrical_solver.__init__`` as part of the
Phase 1 package split. The original module-level docstring with the
full physics derivation lives in the package ``__init__``; refer
there for the field ansatz, sign conventions, and references.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, special

from fwap._common import logger
from fwap.cylindrical_solver._bessel import (
    _as_real_kz,
    _k_or_hankel,
    _radial_wavenumbers_vti_complex,
)
from fwap.cylindrical_solver._dataclasses import BoreholeMode
from fwap.cylindrical_solver._n0_isotropic import (
    stoneley_dispersion,
)
from fwap.cylindrical_solver._n1_isotropic import (
    _march_fast_flexural_branch,
    _real_root_function,
    flexural_dispersion,
)

# =====================================================================
# Plan item H.0 -- public-API foundation for VTI formation
# =====================================================================
#
# Sister of the F.1.0 / F.2.0 layered foundations, but along the
# anisotropy axis instead of the radial-layer axis. Lands the public
# API surface and the **isotropic-collapse oracle** ahead of the
# Schmitt 1989 modal-determinant work scheduled in plan items H.b /
# H.c / H.d (``docs/plans/cylindrical_biot_H.md``).
#
# Parameterisation: the full 5-parameter TI stiffness tensor
# ``(C11, C13, C33, C44, C66)`` plus density ``rho``. Thomsen
# parameters are derivable but not exposed at the API level (gamma
# = (C66 - C44) / (2 C44), epsilon = (C11 - C33) / (2 C33), delta
# encoded in C13).
#
# Isotropic-collapse condition:
#
#       C11 == C33   AND   C44 == C66   AND   C13 == C11 - 2 C44
#
# When this holds, the C-tensor degenerates to the isotropic Lame
# parameters ``(lambda = C13, mu = C44)`` and the VTI dispersion
# matches the isotropic dispersion bit-exactly. The dispatch in
# the public APIs detects this case and routes to the existing
# isotropic solvers; this is the floating-point oracle for the
# entire H chain.


def _validate_vti_stiffness(
    c11: float,
    c13: float,
    c33: float,
    c44: float,
    c66: float,
    rho: float,
) -> None:
    """
    Validate a VTI stiffness tensor + density.

    Raises ``ValueError`` if any coefficient is non-positive or
    if the tensor violates basic Thomsen-stability conditions
    (C33 > C13 for qP/qSV decoupling; C11 > C66 to keep
    horizontal P faster than horizontal S).
    """
    if rho <= 0:
        raise ValueError("rho must be positive")
    for name, value in (
        ("c11", c11),
        ("c13", c13),
        ("c33", c33),
        ("c44", c44),
        ("c66", c66),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive (got {value!r})")
    if c33 <= c13:
        raise ValueError(
            f"require c33 > c13 (got c33={c33!r}, c13={c13!r}); "
            "ensures qP/qSV decoupling in the Christoffel equation."
        )
    if c11 <= c66:
        raise ValueError(
            f"require c11 > c66 (got c11={c11!r}, c66={c66!r}); "
            "ensures horizontal P faster than horizontal S."
        )


def _is_isotropic_stiffness(
    c11: float,
    c13: float,
    c33: float,
    c44: float,
    c66: float,
    *,
    rtol: float = 1.0e-12,
) -> bool:
    """
    Detect the degenerate isotropic case for a VTI stiffness tensor.

    Returns ``True`` when ``C11 == C33``, ``C44 == C66``, and
    ``C13 == C11 - 2 * C44`` to within ``rtol``. The three
    conditions together are necessary and sufficient for the
    five-parameter TI tensor to reduce to the isotropic two-
    parameter Lame form ``(lambda = C13, mu = C44)``.

    The relative tolerance accommodates floating-point round-off
    in cases where the user constructs an isotropic tensor via
    ``c13 = c11 - 2 * c44`` from ``(c11, c44)`` directly: the
    arithmetic is exact in IEEE 754 for representable inputs, but
    user-facing scripts may pass values that have been multiplied
    or divided through and pick up a few ULPs of error.
    """
    scale = max(abs(c11), abs(c33), abs(c44), abs(c66), abs(c13), 1.0)
    if abs(c11 - c33) > rtol * scale:
        return False
    if abs(c44 - c66) > rtol * scale:
        return False
    if abs(c13 - (c11 - 2.0 * c44)) > rtol * scale:
        return False
    return True


def stoneley_dispersion_vti(
    freq: np.ndarray,
    *,
    c11: float,
    c13: float,
    c33: float,
    c44: float,
    c66: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
) -> BoreholeMode:
    r"""
    Stoneley-wave (n=0) phase slowness vs frequency for a borehole
    in a VTI formation (transversely isotropic, vertical symmetry
    axis).

    With an isotropic stiffness tensor (``C11 = C33``, ``C44 = C66``,
    ``C13 = C11 - 2 * C44``) this is bit-equivalent to
    :func:`stoneley_dispersion` with ``vp = sqrt(C33 / rho)``,
    ``vs = sqrt(C44 / rho)``. With a genuinely-anisotropic tensor,
    runs the Schmitt 1989 VTI modal determinant
    :func:`_modal_determinant_n0_vti` through brentq with the
    Norris-1990-driven bracket :func:`_stoneley_kz_bracket_vti`.

    Parameters
    ----------
    freq : ndarray
        Frequency grid (Hz). Must be strictly positive.
    c11, c13, c33, c44, c66 : float
        VTI stiffness tensor entries (Pa). All must be positive
        and satisfy ``C33 > C13`` and ``C11 > C66``.
    rho : float
        Formation density (kg/m^3). Must be positive.
    vf, rho_f : float
        Borehole-fluid velocity (m/s) and density (kg/m^3).
    a : float
        Borehole radius (m).

    Returns
    -------
    BoreholeMode
        ``name = "Stoneley"``, ``azimuthal_order = 0``.

    Raises
    ------
    ValueError
        If any input is non-positive, the stiffness tensor
        violates Thomsen-stability, or ``freq`` contains a
        non-positive entry.
    """
    _validate_vti_stiffness(c11, c13, c33, c44, c66, rho)
    if vf <= 0 or rho_f <= 0:
        raise ValueError("vf and rho_f must be positive")
    if a <= 0:
        raise ValueError("a must be positive")
    f_arr = np.asarray(freq, dtype=float)
    if np.any(f_arr <= 0):
        raise ValueError("freq must be strictly positive")
    if _is_isotropic_stiffness(c11, c13, c33, c44, c66):
        vp = float(np.sqrt(c33 / rho))
        vs = float(np.sqrt(c44 / rho))
        return stoneley_dispersion(
            f_arr,
            vp=vp,
            vs=vs,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            a=a,
        )
    # Genuine TI: brentq loop on _modal_determinant_n0_vti per H.c.2.
    slowness = np.full_like(f_arr, np.nan, dtype=float)
    for i, f in enumerate(f_arr):
        omega = 2.0 * np.pi * float(f)

        def _det(kz_, omega=omega):
            return _modal_determinant_n0_vti(
                kz_,
                omega,
                c11=c11,
                c13=c13,
                c33=c33,
                c44=c44,
                c66=c66,
                rho=rho,
                vf=vf,
                rho_f=rho_f,
                a=a,
            )

        kz_lo, kz_hi = _stoneley_kz_bracket_vti(
            omega,
            c11=c11,
            c13=c13,
            c33=c33,
            c44=c44,
            c66=c66,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            a=a,
        )
        try:
            d_lo = _det(kz_lo)
            d_hi = _det(kz_hi)
            n_expand = 0
            while (
                np.isfinite(d_lo)
                and np.isfinite(d_hi)
                and np.sign(d_lo) == np.sign(d_hi)
                and n_expand < 8
            ):
                kz_hi *= 1.5
                d_hi = _det(kz_hi)
                n_expand += 1
            if (not np.isfinite(d_lo)) or (not np.isfinite(d_hi)):
                logger.debug(
                    "stoneley_dispersion_vti: det evaluation NaN "
                    "at f=%.1f Hz (likely outside bound regime)",
                    f,
                )
                continue
            if np.sign(d_lo) == np.sign(d_hi):
                logger.debug(
                    "stoneley_dispersion_vti: failed to bracket at f=%.1f Hz",
                    f,
                )
                continue
            kz_root = optimize.brentq(_det, kz_lo, kz_hi, xtol=1.0e-10)
            slowness[i] = kz_root / omega
        except (ValueError, RuntimeError) as exc:
            logger.debug(
                "stoneley_dispersion_vti: brentq failed at f=%.1f Hz: %s",
                f,
                exc,
            )

    return BoreholeMode(
        name="Stoneley",
        azimuthal_order=0,
        freq=f_arr,
        slowness=slowness,
    )


def _stoneley_kz_bracket_vti(
    omega: float,
    *,
    c11: float,
    c13: float,
    c33: float,
    c44: float,
    c66: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
) -> tuple[float, float]:
    """
    Bracket the n=0 VTI Stoneley root in (k_z_lo, k_z_hi).

    Lower bound: just above the bound-regime floor
    ``omega / min(V_Sv, V_Sh, V_f)``. Upper bound from the
    Norris 1990 LF closed-form ``S_ST^2 = 1/V_f^2 + rho_f / C66``;
    the actual converged kz is bracketed to within a factor of 1.5.

    Note: uses C66 (NOT C44) in the LF estimate -- the TI-specific
    coupling per Norris 1990. In isotropic this matches the
    standard isotropic bracket since C66 = C44 = mu.
    """
    s_st_lf = float(np.sqrt(1.0 / vf**2 + rho_f / c66))
    kz_lf_est = omega * s_st_lf
    Vsv = float(np.sqrt(c44 / rho))
    Vsh = float(np.sqrt(c66 / rho))
    slowest = min(Vsv, Vsh, vf)
    kz_lo = omega / slowest * (1.0 + 1.0e-6)
    kz_hi = max(kz_lf_est * 1.5, kz_lo * 2.0)
    return kz_lo, kz_hi


def _flexural_kz_bracket_vti(
    omega: float,
    *,
    c11: float,
    c13: float,
    c33: float,
    c44: float,
    c66: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
) -> tuple[float, float]:
    """
    Bracket the n=1 VTI flexural root in (k_z_lo, k_z_hi) for
    the slow-formation regime.

    The Sinha-Norris-Chang phenomenological model places the
    flexural phase velocity in a band around ``V_Sv`` at low f,
    descending to a TI-Rayleigh-like speed at high f. Slowness
    sits between roughly ``1/V_Sv`` (LF asymptote) and
    ``~1.1/V_Sv`` (HF Rayleigh-like).

    Lower bound: just above ``omega / V_Sv`` -- the SH-decoupled
    bound floor (``alpha_SH^2 = (C44 k_z^2 - rho omega^2) / C66``
    requires ``k_z > omega / V_Sv``).

    Upper bound: ``1.5 omega / V_Sv``, generous enough to cover
    the HF Rayleigh-like asymptote. Caller may expand outward if
    no sign change is found.

    Notes
    -----
    Restricted to the slow-formation TI regime (``V_Sv < V_f``);
    the caller is expected to dispatch fast-formation TI elsewhere
    since the real-valued VTI determinant requires
    ``k_z > omega / V_f`` (``F_f^2 > 0``), which is automatic only
    when ``V_Sv < V_f``.
    """
    Vsv = float(np.sqrt(c44 / rho))
    kz_lo = omega / Vsv * (1.0 + 1.0e-6)
    kz_hi = omega / Vsv * 1.5
    if kz_hi <= kz_lo:
        kz_hi = kz_lo * 2.0
    return kz_lo, kz_hi


def flexural_dispersion_vti(
    freq: np.ndarray,
    *,
    c11: float,
    c13: float,
    c33: float,
    c44: float,
    c66: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
) -> BoreholeMode:
    r"""
    Flexural-wave (n=1) phase slowness vs frequency for a borehole
    in a VTI formation.

    Sister of :func:`stoneley_dispersion_vti` at azimuthal order 1.
    With an isotropic stiffness tensor this is bit-equivalent to
    :func:`flexural_dispersion`. With a genuinely-anisotropic
    tensor in the slow-formation regime (``V_Sv = sqrt(C44/rho)
    < V_f``), runs the Schmitt 1989 VTI modal determinant
    :func:`_modal_determinant_n1_vti` through brentq with the
    Sinha-Norris-Chang-driven bracket
    :func:`_flexural_kz_bracket_vti`.

    Fast-formation TI (``V_Sv > V_f``) dispatches to
    :func:`_flexural_dispersion_fast_formation_vti`, which runs the
    complex VTI determinant
    :func:`_modal_determinant_n1_vti_complex` through the same
    signal-part measurement and branch marcher the isotropic
    fast-formation path uses. There ``F_f^2 < 0`` over the part of
    the band where the phase velocity exceeds ``V_f``, so the fluid
    Bessels are oscillatory; the formation qP / qSV / SH
    wavenumbers stay real and ``k_z`` stays real, so the root is a
    genuine bound mode rather than a leaky one.

    Parameters and validation rules: identical to
    :func:`stoneley_dispersion_vti`.

    Returns
    -------
    BoreholeMode
        ``name = "flexural"``, ``azimuthal_order = 1``.

    Raises
    ------
    ValueError
        Same conditions as :func:`stoneley_dispersion_vti`.
    """
    _validate_vti_stiffness(c11, c13, c33, c44, c66, rho)
    if vf <= 0 or rho_f <= 0:
        raise ValueError("vf and rho_f must be positive")
    if a <= 0:
        raise ValueError("a must be positive")
    f_arr = np.asarray(freq, dtype=float)
    if np.any(f_arr <= 0):
        raise ValueError("freq must be strictly positive")
    if _is_isotropic_stiffness(c11, c13, c33, c44, c66):
        vp = float(np.sqrt(c33 / rho))
        vs = float(np.sqrt(c44 / rho))
        return flexural_dispersion(
            f_arr,
            vp=vp,
            vs=vs,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            a=a,
        )
    Vsv = float(np.sqrt(c44 / rho))
    if Vsv > vf:
        return _flexural_dispersion_fast_formation_vti(
            f_arr,
            c11=c11,
            c13=c13,
            c33=c33,
            c44=c44,
            c66=c66,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            a=a,
        )
    # Genuine TI, slow formation: brentq loop on
    # _modal_determinant_n1_vti per H.d.6.
    slowness = np.full_like(f_arr, np.nan, dtype=float)
    for i, f in enumerate(f_arr):
        omega = 2.0 * np.pi * float(f)

        def _det(kz_, omega=omega):
            return _modal_determinant_n1_vti(
                kz_,
                omega,
                c11=c11,
                c13=c13,
                c33=c33,
                c44=c44,
                c66=c66,
                rho=rho,
                vf=vf,
                rho_f=rho_f,
                a=a,
            )

        kz_lo, kz_hi = _flexural_kz_bracket_vti(
            omega,
            c11=c11,
            c13=c13,
            c33=c33,
            c44=c44,
            c66=c66,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            a=a,
        )
        try:
            d_lo = _det(kz_lo)
            d_hi = _det(kz_hi)
            n_expand = 0
            while (
                np.isfinite(d_lo)
                and np.isfinite(d_hi)
                and np.sign(d_lo) == np.sign(d_hi)
                and n_expand < 8
            ):
                kz_hi *= 1.5
                d_hi = _det(kz_hi)
                n_expand += 1
            if (not np.isfinite(d_lo)) or (not np.isfinite(d_hi)):
                logger.debug(
                    "flexural_dispersion_vti: det evaluation NaN "
                    "at f=%.1f Hz (likely outside bound regime)",
                    f,
                )
                continue
            if np.sign(d_lo) == np.sign(d_hi):
                logger.debug(
                    "flexural_dispersion_vti: failed to bracket at f=%.1f Hz",
                    f,
                )
                continue
            kz_root = optimize.brentq(_det, kz_lo, kz_hi, xtol=1.0e-10)
            slowness[i] = kz_root / omega
        except (ValueError, RuntimeError) as exc:
            logger.debug(
                "flexural_dispersion_vti: brentq failed at f=%.1f Hz: %s",
                f,
                exc,
            )

    return BoreholeMode(
        name="flexural",
        azimuthal_order=1,
        freq=f_arr,
        slowness=slowness,
    )


def _flexural_dispersion_fast_formation_vti(
    freq: np.ndarray,
    *,
    c11: float,
    c13: float,
    c33: float,
    c44: float,
    c66: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
) -> BoreholeMode:
    r"""
    Fast-formation (``V_Sv > V_f``) VTI flexural dispersion.

    Anisotropy sister of
    :func:`_flexural_dispersion_fast_formation`. The structure is
    deliberately identical -- complex determinant, measured signal
    part, branch marcher -- because the only thing anisotropy
    changes is the determinant itself. Reusing the isotropic
    marcher rather than writing a second one means the two paths
    cannot disagree about *which* root is the fundamental, which is
    the part that was historically got wrong (roadmap A.2).

    The shear velocity that bounds the search is ``V_Sv =
    sqrt(C44/rho)``, the vertically-polarised shear speed. That is
    the right one for a dipole flexural mode in a vertical borehole:
    the mode is built on the qSV branch, and ``C44`` is the
    stiffness that branch samples at ``k_z``-dominated propagation.
    ``V_Sh = sqrt(C66/rho)`` governs the SH-decoupled floor instead,
    and for the Green River shale of Ellefsen fig 2 the two differ
    by 18 % -- large enough that using the wrong one loses the
    branch rather than merely shifting it.

    Parameters
    ----------
    freq : ndarray
        Frequency grid (Hz), strictly positive.
    c11, c13, c33, c44, c66 : float
        VTI stiffness tensor entries (Pa).
    rho : float
        Formation density (kg / m^3).
    vf, rho_f : float
        Borehole-fluid velocity (m / s) and density (kg / m^3).
    a : float
        Borehole radius (m).

    Returns
    -------
    BoreholeMode
        ``name = "flexural"``, ``azimuthal_order = 1``. ``NaN`` at
        frequencies where the branch is not a real-``k_z`` root --
        the same two unreachable bands the isotropic path documents,
        below the Rayleigh-like crossing and above the ``V_f``
        crossing.
    """
    f_arr = np.asarray(freq, dtype=float)
    slowness = np.full(f_arr.size, np.nan, dtype=float)
    if f_arr.size == 0:
        return BoreholeMode(
            name="flexural",
            azimuthal_order=1,
            freq=f_arr,
            slowness=slowness,
        )
    Vsv = float(np.sqrt(c44 / rho))

    def _det(kz: float, _omega: float) -> complex:
        return _modal_determinant_n1_vti_complex(
            float(kz),
            _omega,
            c11=c11,
            c13=c13,
            c33=c33,
            c44=c44,
            c66=c66,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            a=a,
        )

    # The branch descends through V_f here exactly as it does in the
    # isotropic and layered drivers, and below it the fluid Bessels are
    # non-oscillatory again, so the *real* VTI determinant applies --
    # which is precisely the regime it documents itself as valid in.
    # Without this the VTI driver stopped at the crossing: on a
    # 3800/3658/2032/2100 shale it returned NaN from 11 kHz up while its
    # own determinant had a root at 1489 m/s descending smoothly from
    # the 1504 m/s it had just reported at 10 kHz.
    def _real_det(kz: float, _omega: float) -> float:
        return _modal_determinant_n1_vti(
            kz,
            _omega,
            c11=c11,
            c13=c13,
            c33=c33,
            c44=c44,
            c66=c66,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            a=a,
        )

    root_fn = _real_root_function(_det, f_arr, vs=Vsv, vf=vf)
    slowness = _march_fast_flexural_branch(
        root_fn, f_arr, vs=Vsv, vf=vf, real_det=_real_det
    )

    return BoreholeMode(
        name="flexural",
        azimuthal_order=1,
        freq=f_arr,
        slowness=slowness,
    )


# =====================================================================
# Substep H.a -- math scaffolding for the VTI modal determinant
# =====================================================================
#
# Sister of F.2.a along the anisotropy axis. Inherits all conventions
# from the module docstring and from the existing isotropic n=0 / n=1
# derivations (substeps in :func:`_modal_determinant_n0` and
# :func:`_modal_determinant_n1`); extends with TI-specific machinery:
# the Christoffel-equation roots for radial wavenumbers, the
# C-matrix replacement of the Lame reduction, and the qP / qSV root-
# selection convention.
#
# The substep blocks below land the maths only; the radial-wavenumber
# helper (substep H.b), the modal-determinant builders (H.c, H.d),
# and the public-API wiring (already in H.0) follow in dependent
# commits per ``docs/plans/cylindrical_biot_H.md``.

# =====================================================================
# Substep H.a.1 -- sign conventions, TI stiffness tensor, regime gate
# =====================================================================
#
# Conventions inherited from the module docstring:
#
#   * Time dependence ``e^{-i omega t}``.
#   * Axial dependence ``e^{i k_z z}``.
#   * Azimuthal dependence ``e^{i n theta}`` (n = 0 axisymmetric,
#     n = 1 dipole).
#   * Bound regime: every radial wavenumber is real positive.
#
# TI stiffness tensor (5 independent entries + density):
#
#       C11 = rho V_Ph^2     (horizontal P)
#       C13                  (Thomsen-delta-coupled)
#       C33 = rho V_Pv^2     (vertical P)
#       C44 = rho V_Sv^2     (vertical S)
#       C66 = rho V_Sh^2     (horizontal S)
#
# Vertical symmetry axis aligned with ``z`` (the borehole axis); the
# borehole-wall normal lies in the horizontal ``(r, theta)`` plane,
# so the modal determinant decouples by azimuthal order n exactly as
# in the isotropic case.
#
# Thomsen parameters (derivable from the C-matrix; not exposed at
# the public-API level):
#
#       gamma   = (C66 - C44) / (2 C44)        (SH anisotropy)
#       epsilon = (C11 - C33) / (2 C33)        (P anisotropy in (r,z))
#       delta   encoded in C13 via
#       C13 = sqrt((C33 - C44)(C33 - C44 + 2 delta C33)) - C44
#
# Bound-regime gate for the layered-mode dispersion:
#
#       k_z > omega / min(V_Sv, V_Sh, V_f)
#
# Both shear speeds appear because the qSV branch's natural radial
# wavenumber is set by ``V_Sv`` while the SH branch (relevant at
# n >= 1) is set by ``V_Sh``. In typical VTI shales V_Sh > V_Sv (gamma
# > 0) so V_Sv is the binding constraint, but the gate is written
# symmetrically.
#
# Field representation (bound regime):
#
#   Fluid (r < a):
#       P^{(f)}        = A * I_n(F_f r)              (n=0 or n=1)
#
#   Formation (r > a) -- four scalar potentials:
#       qP scalar:     phi^{(qP)}   = B * K_n(alpha_qP r)
#       qSV theta:     psi^{(qSV)}  = C * K_n(alpha_qSV r)
#       SH z:          psi_z^{(SH)} = D * K_n(alpha_SH r)
#                                     (n >= 1 only)
#       (azimuthal factors cos(n theta) or sin(n theta) per substep
#       1.1 of the n=1 single-interface block)
#
# Five amplitudes total at n=1 (A, B, C, D + the additional
# polarization-vector-mediated coupling between qP and qSV); three
# at n=0 (A, B, C; SH decoupled).
#
# References:
#   * Schmitt, D. P. (1989). Acoustic multipole logging in
#     transversely isotropic poroelastic formations. *J. Acoust.
#     Soc. Am.* 86(6), 2397-2421, doi:10.1121/1.398448. Sect. II for
#     the four-potential decomposition.
#     The DOI is corroborated but not confirmed: it agrees with the
#     volume, issue and pages recorded here, and every route that could
#     settle it against the publisher -- doi.org, Crossref, OpenAlex,
#     Semantic Scholar, the AIP page -- is blocked by this project's
#     network policy. Worth one look from a session that can reach
#     them, because the sibling citation was corrected across nine
#     files and still had its page range wrong (plans/guides.md 9).
#   * Tsvankin, I. (2001). *Seismic Signatures and Analysis of
#     Reflection Data in Anisotropic Media.* Pergamon, ch. 1.

# =====================================================================
# Substep H.a.2 -- Christoffel equation: quadratic in alpha^2
# =====================================================================
#
# Restrict to the cos / sin sector at azimuthal order n. The qP and
# qSV polarizations live in the (r, z) plane; SH is the theta
# polarization. For n = 0 axisymmetric, only qP and qSV participate
# (SH decoupled by axisymmetry). For n = 1, all three couple via
# d_theta operations -- but the radial-wavenumber equation is the
# same Christoffel-determinant condition in all cases.
#
# Convention: ``alpha`` is the radial DECAY RATE used by modified
# Bessel ``K_n(alpha r)`` for bound-mode (decaying-outward) field
# representation, matching the isotropic ``p`` and ``s`` definitions
# above. In propagating-plane-wave form, the horizontal wavenumber
# is ``k_h = i alpha`` (purely imaginary).
#
# The Christoffel matrix in the (r, z) plane for a bound-mode plane
# wave with horizontal-wavenumber substitution ``k_h -> i alpha``
# (alpha real positive in the bound regime) and axial wavenumber
# ``k_z``, evaluated in a VTI half-space:
#
#       | -C11 alpha^2 + C44 k_z^2 - rho omega^2    i (C13+C44) alpha k_z |
#       |                                                                  |
#       | i (C13+C44) alpha k_z                     -C44 alpha^2 + C33 k_z^2 - rho omega^2 |
#
# The qP / qSV dispersion follows from the secular equation
# ``det(Christoffel) = 0``:
#
#       (-C11 alpha^2 + C44 k_z^2 - rho omega^2) *
#       (-C44 alpha^2 + C33 k_z^2 - rho omega^2)
#       + (C13 + C44)^2 alpha^2 k_z^2 = 0.
#
# Note the SIGN on the off-diagonal contribution: ``(i alpha k_z)^2
# = - alpha^2 k_z^2``, so the off-diagonal term enters with a
# *positive* sign in the bound-mode convention (vs the negative sign
# under the propagating-k_h convention).
#
# Expand and collect on alpha^2:
#
#       A_eff alpha^4 + B_eff(k_z, omega) alpha^2 + C_eff(k_z, omega) = 0
#
# with
#
#       A_eff = C11 * C44
#       B_eff = (C11 + C44) rho omega^2
#               - [(C11 C33 + C44^2) - (C13 + C44)^2] k_z^2
#       C_eff = C44 C33 k_z^4 - (C44 + C33) rho omega^2 k_z^2
#               + (rho omega^2)^2
#
# The two roots alpha_qP^2 and alpha_qSV^2 are the squared radial
# decay rates for the quasi-P and quasi-SV branches. In the bound
# regime both roots are real positive.
#
# Discriminant and explicit roots:
#
#       Delta = B_eff^2 - 4 A_eff C_eff
#       alpha_{qP, qSV}^2 = (-B_eff +/- sqrt(Delta)) / (2 A_eff)
#
# Both A_eff > 0 and (under Thomsen-stable inputs) Delta >= 0.
# Vieta's formulas: alpha_qP^2 + alpha_qSV^2 = -B_eff / A_eff and
# alpha_qP^2 * alpha_qSV^2 = C_eff / A_eff. In the bound regime both
# (-B_eff / A_eff) and (C_eff / A_eff) are positive (by direct
# calculation under the H.a.1 gate), so both roots are real positive.
#
# Isotropic-collapse identity (substep H.a.7 self-check):
# with C11 = C33 = lambda + 2 mu, C44 = mu, C13 = lambda,
#
#       (C13 + C44)^2 = (lambda + mu)^2
#       C11 C33 + C44^2 = (lambda + 2 mu)^2 + mu^2
#       (C11 C33 + C44^2) - (C13 + C44)^2 = 2 mu (lambda + 2 mu)
#
#       A_eff = mu (lambda + 2 mu)
#       B_eff = (lambda + 3 mu) rho omega^2 - 2 mu (lambda + 2 mu) k_z^2
#       C_eff = mu (lambda + 2 mu) k_z^4
#               - (lambda + 3 mu) rho omega^2 k_z^2 + (rho omega^2)^2
#
# After the algebra (verifiable via Vieta's formulas) the two roots
# reduce to
#
#       alpha_qP^2  -> p^2 = k_z^2 - omega^2 / V_P^2
#       alpha_qSV^2 -> s^2 = k_z^2 - omega^2 / V_S^2
#
# matching the isotropic radial-wavenumber definitions exactly.
# Sum check: -B_eff / A_eff = 2 k_z^2 - omega^2 / V_P^2 - omega^2 /
# V_S^2 = p^2 + s^2. Product check: C_eff / A_eff = (k_z^2 -
# omega^2 / V_P^2)(k_z^2 - omega^2 / V_S^2) = p^2 s^2. Both Vieta
# identities hold to floating-point. This is the oracle for H.b's
# unit tests.
#
# References: Schmitt 1989 eqs. 17-18; Ellefsen, Cheng, Toksoz 1991
# eq. 7; Tsvankin 2001 sect. 1.4.

# =====================================================================
# Substep H.a.3 -- qP / qSV root-selection convention
# =====================================================================
#
# The Christoffel quadratic gives two roots; their physical labels
# (qP vs qSV) need an unambiguous selection rule. Convention:
#
#   * **qP** is the LARGER root in the bound regime (larger radial
#     decay constant -- faster body wave is MORE localized at the
#     borehole wall in a bound mode). At isotropic limit
#     alpha_qP^2 -> p^2 = k_z^2 - omega^2 / V_P^2 (> s^2 since
#     V_P > V_S means omega^2 / V_P^2 < omega^2 / V_S^2).
#
#   * **qSV** is the SMALLER root. At isotropic limit
#     alpha_qSV^2 -> s^2 = k_z^2 - omega^2 / V_S^2 (< p^2).
#
# Reasoning for the LARGER -> qP convention: at fixed (k_z, omega)
# in the bound regime, the radial decay rate ``alpha`` for a wave
# of speed V satisfies ``alpha^2 = k_z^2 - omega^2 / V^2``. Larger
# V means smaller ``omega^2 / V^2`` means LARGER alpha^2. So the
# faster wave (qP, V_P > V_S in any stable formation) has the
# larger decay rate.
#
# Implementation: take the two roots from the quadratic, pick
# alpha_qP^2 = min(root_1, root_2) and alpha_qSV^2 = max. The
# convention agrees with the isotropic limit by inspection (since
# p^2 < s^2 always in the bound regime when V_P > V_S).
#
# Edge case: if the two roots are equal (alpha_qP = alpha_qSV), the
# Christoffel matrix has a degenerate eigenspace and the modal
# matrix becomes singular. This corresponds to a "shear-bispherical"
# point in the dispersion locus -- not relevant in physically
# stable bound-mode regimes, but the H.b helper raises a clear
# error if encountered.
#
# Polarization vectors (Christoffel eigenvectors corresponding to
# the two roots) determine the displacement-amplitude coupling for
# the modal-matrix entries; pinned in substep H.a.4 below.

# =====================================================================
# Substep H.a.4 -- per-region displacements (qP, qSV, SH polarizations)
# =====================================================================
#
# For each Christoffel root alpha_qX (X in {qP, qSV}), the
# corresponding polarization vector ``(u_r, u_z)`` is the null
# eigenvector of the (now-singular) Christoffel matrix:
#
#       (C11 alpha_qX^2 + C44 k_z^2 - rho omega^2) u_r
#       + (C13 + C44) alpha_qX k_z u_z = 0
#
# Solving for the polarization ratio (with the convention u_r = 1):
#
#       u_z / u_r = - [C11 alpha_qX^2 + C44 k_z^2 - rho omega^2]
#                   / [(C13 + C44) alpha_qX k_z]                   (qX-spec)
#
# In the isotropic limit this reduces to
#
#       (u_r, u_z)_qP  -> (p, i k_z)         (longitudinal: aligned
#                                              with wavefront normal)
#       (u_r, u_z)_qSV -> (-i k_z, s)        (transverse: perpendicular
#                                              to wavefront normal)
#
# matching the isotropic potential-decomposition derivation in the
# n=0 / n=1 substep blocks above.
#
# Per-region displacements (cos sector unless noted; sin for the SH
# contribution at n=1):
#
#   Fluid u_r^{(f)}, u_z^{(f)}: identical to the isotropic case
#   (fluid is isotropic regardless of formation anisotropy).
#
#   Formation u_r^{(s)}, u_z^{(s)} from qP and qSV:
#       u_r^{(qP)}  = -B_qP * (alpha_qP K_n'(alpha_qP r))     (cos)
#       u_z^{(qP)}  = +i k_z B_qP * (u_z/u_r)_qP * K_n        (cos)
#       u_r^{(qSV)} = -i k_z C_qSV K_{n+1}(alpha_qSV r)       (cos)
#       u_z^{(qSV)} = -C_qSV alpha_qSV K_n(alpha_qSV r)       (cos)
#
# (with the polarization ratios pinned in H.a.3 -- the explicit form
# matches the isotropic n=0 / n=1 blocks at the isotropic limit).
#
# At n=1, the SH amplitude D appears via:
#       u_theta^{(SH)} contributes through the standard
#       ``(curl psi_z)_r = (1/r) d_theta psi_z`` mechanism.
#
#       SH dispersion in VTI (decoupled from qP / qSV at the
#       symmetry axis): the equation of motion is
#           rho omega^2 u_theta = C44 d_z^2 u_theta
#                                 + C66 nabla_h^2 u_theta,
#       which under the bound-mode substitution k_h -> i alpha_SH
#       (decay-rate convention; matches the isotropic ``s`` form)
#       yields
#
#           alpha_SH^2 = (C44 k_z^2 - rho omega^2) / C66.
#
#       Both C44 (vertical-shear coupling) and C66 (horizontal-
#       shear coupling) appear -- the decay rate depends on the
#       full SH stiffness pair, not on C66 alone. An earlier draft
#       of this substep wrote ``alpha_SH^2 = k_z^2 - rho omega^2 /
#       C66``, which collapses correctly at the isotropic limit
#       (C44 = C66 = mu) but is wrong in genuine TI. The H.b unit
#       tests catch this via the Christoffel-equation-identity
#       check substituted back at the qP / qSV roots.
#
#       Isotropic-collapse identity: with C44 = C66 = mu,
#           alpha_SH^2 -> (mu k_z^2 - rho omega^2) / mu
#                      = k_z^2 - omega^2 / V_S^2 = s^2.

# =====================================================================
# Substep H.a.5 -- per-region stresses + C-matrix Lame replacement
# =====================================================================
#
# Hooke's law in the VTI half-space (in the (r, z) plane, n = 0 / 1):
#
#       sigma_rr = C11 eps_rr + C13 eps_zz + (theta-derivative terms)
#       sigma_rz = 2 C44 eps_rz
#       sigma_rtheta = 2 C66 eps_rtheta             (n >= 1; SH-driven)
#       sigma_zz = C13 eps_rr + C33 eps_zz + ...
#
# vs the isotropic forms which use lambda, mu (single shear modulus
# everywhere); the VTI form has C13 instead of lambda in sigma_rr,
# C33 in sigma_zz, C44 in sigma_rz, C66 in sigma_rtheta.
#
# The Lame reduction
#
#       -lambda k_P^2 + 2 mu p^2 = mu (2 k_z^2 - k_S^2)
#
# from the isotropic n=0 / n=1 blocks does NOT carry over directly.
# Instead, for the qP-amplitude contribution to sigma_rr at the
# borehole wall, the analogous combination is (Schmitt 1989 eq. 21):
#
#       [C11 alpha_qP^2 + C13 i k_z (u_z/u_r)_qP - C13 rho omega^2 / C33]
#       * (Bessel terms)
#
# i.e., the C-matrix combination depends on the qP polarization
# ratio, which itself depends on alpha_qP from the Christoffel
# equation. In the isotropic limit this collapses to the Lame
# reduction; in the genuine TI case it doesn't simplify further.
#
# Same structural change for the qSV column. The sigma_rtheta entries
# at n >= 1 use C66 directly (no Lame combination needed):
#
#       sigma_rtheta from D_SH = C66 * (D_SH polarization terms)
#
# This is the slot through which gamma (= (C66 - C44) / (2 C44))
# enters the dispersion via the D-amplitude. In the layered
# extension (plan G), this slot is also where the Norris 1990
# closed-form Stoneley LF limit ``S_ST^2 = 1/V_f^2 + rho_f / C66``
# originates: at low frequency the n=0 Stoneley wave samples C66
# (not C44) of the formation through the wall coupling.
#
# Detailed entries deferred to the per-row builders in H.c (n=0)
# and H.d (n=1); the substep block here pins the symbolic form
# and the isotropic-collapse identity (every C-matrix combination
# reduces to the corresponding Lame combination).

# =====================================================================
# Substep H.a.6 -- modal-determinant row layout + phase rescaling
# =====================================================================
#
# **n = 0 axisymmetric (Stoneley)**: 3x3 modal determinant. Same
# row layout as :func:`_modal_determinant_n0`:
#
#       Row 1   u_r^{(f)} - u_r^{(s)} = 0     (cos sector)
#       Row 2   sigma_rr^{(s)} + P^{(f)} = 0  (cos)
#       Row 3   sigma_rz^{(s)} = 0            (cos; rescaled by i)
#
# Columns: ``[A | B_qP, C_qSV]``. SH is decoupled by axisymmetry
# at n = 0 (no D column).
#
# Phase rescaling: row 3 multiplied by ``i``; column C_qSV
# multiplied by ``-i``. Net factor i * (-i) = 1; same convention as
# isotropic n=0.
#
# **n = 1 dipole (flexural)**: 4x4 modal determinant. Same row
# layout as :func:`_modal_determinant_n1`:
#
#       Row 1   u_r^{(f)} - u_r^{(s)} = 0       (cos)
#       Row 2   -(sigma_rr^{(s)} + P^{(f)}) = 0  (cos)
#       Row 3   sigma_rtheta^{(s)} = 0           (sin)
#       Row 4   sigma_rz^{(s)} = 0               (cos; rescaled by i)
#
# Columns: ``[A | B_qP, C_qSV, D_SH]``. SH appears via the
# ``(1/r) d_theta psi_z`` mechanism from F.2.a / the n=1 substep
# blocks (this cross-couples sin-sector D into cos-sector u_r BC).
#
# Phase rescaling: same row-4-by-i / column-C_qSV-by-(-i) as the
# isotropic n=1 form. Net factor +1.
#
# At n = 0 the layered VTI extension would be a 7x7 (mirroring
# F.1's layered Stoneley); at n = 1 it would be 10x10 (mirroring
# F.2). Both are out of scope for plan H -- the layered + VTI
# combination is a follow-up beyond plan items F, G, H.

# =====================================================================
# Substep H.a.7 -- isotropic-collapse self-check protocol
# =====================================================================
#
# Three structural identities the VTI determinant must satisfy by
# construction, tested at successive levels of the H chain:
#
# (a) **Isotropic-collapse at the radial-wavenumber level.**
#     With C11 = C33 = lambda + 2 mu, C44 = C66 = mu, C13 = lambda,
#     the two Christoffel-equation roots reduce to
#       alpha_qP^2  = k_z^2 - omega^2 / V_P^2
#       alpha_qSV^2 = k_z^2 - omega^2 / V_S^2
#     to floating-point precision. Tested at the H.b unit-test
#     level via the radial-wavenumber helper.
#
# (b) **Isotropic-collapse at the modal-matrix level.**
#     Every entry M_ij^vti reduces to the corresponding isotropic
#     M_ij entry to floating-point precision under the same
#     C-matrix substitution. Tested at the H.c (n=0) and H.d (n=1)
#     per-row-builder level. Direct mirror of the F.1 / F.2
#     layer=formation regression -- the floating-point oracle for
#     the matrix algebra.
#
# (c) **Norris 1990 closed-form Stoneley LF limit.**
#     At low frequency the n=0 Stoneley slowness in a VTI formation
#     approaches
#       S_ST^2 = 1 / V_f^2 + rho_f / C66
#     (Norris 1990 eq. 6). The TI-specific oracle: depends on C66,
#     not C44. Tested at the H.c integration level. Strong
#     validation of the SH-decoupling claim in substep H.a.4 (the
#     C66 dependence enters only through the sigma_rtheta slot at
#     n >= 1, but at n = 0 the wall-displacement matching couples
#     C66 into the Stoneley wave through the radial-shear stress
#     condition).
#
# Reference for (c): Norris, A. N. (1990). The speed of a tube
# wave. *J. Acoust. Soc. Am.* 87(1), 414-417. Eq. 6 derives the
# closed form for the TI tube-wave speed at long wavelength.
#
# References for the broader VTI cylindrical-mode framework:
#   * Schmitt 1989 (the canonical paper for VTI multipole logging).
#   * Ellefsen, Cheng & Toksoz (1991). Effects of anisotropy on
#     the shear-wave logging in a TI formation. *J. Acoust. Soc.
#     Am.* 89(5), 2197-2210. Weak-anisotropy expansions.
#   * Sinha, Norris, Chang (1994). Borehole flexural modes in
#     anisotropic formations. *Geophysics* 59(7), 1037-1052.


# =====================================================================
# Substep H.c.1.a -- row 1 of the n=0 VTI modal determinant (r = a)
# =====================================================================
#
# BC1: ``u_r^{(f)}(a) - u_r^{(s)}(a) = 0`` (cos-sector continuity of
# radial displacement at the borehole wall). First row of the n=0
# VTI Stoneley modal determinant; same row layout as
# :func:`_modal_determinant_n0` but with the Christoffel roots
# ``alpha_qP``, ``alpha_qSV`` replacing the isotropic ``p``, ``s``.
#
# Per substep H.a.4, the u_r contributions in the formation come
# from the radial r-component of each Christoffel eigenvector:
#
#       u_r^{(qP)}(r)  = -B_qP * alpha_qP * K_1(alpha_qP r)
#       u_r^{(qSV)}(r) = -i k_z * C_qSV * K_1(alpha_qSV r)
#
# The qP r-component normalises as ``-alpha_qP`` (carries the factor
# from the radial derivative of the qP scalar potential); the qSV
# r-component normalises as ``-i k_z`` (carries the axial-derivative
# factor from the vector-potential-style ansatz). Both are
# convention-matched to the isotropic n=0 forms, so the polarization
# ratio ``gamma_qX = u_z/u_r`` does NOT enter row 1 (only u_r matters
# here).
#
# Subtracting (fluid - solid):
#
#       Row 1 (pre-rescale) = [
#           +F_f I_1(F_f a) / (rho_f omega^2),     # A column
#           +alpha_qP K_1(alpha_qP a),              # B_qP column
#           +i k_z K_1(alpha_qSV a),                # C_qSV column (pre-rescale)
#       ]
#
# Phase rescale (substep H.a.6 = same as isotropic n=0): row 1 is
# NOT z-derivative-bearing (no row * i). The C-column is rescaled
# by ``-i``; post-rescale C entry: ``(-i)(+i k_z K_1) = +k_z K_1``.
#
# Post-rescale row 1:
#
#       Row 1 = [
#           +F_f I_1(F_f a) / (rho_f omega^2),     # A
#           +alpha_qP K_1(alpha_qP a),              # B_qP
#           +k_z K_1(alpha_qSV a),                  # C_qSV
#       ]
#
# Isotropic-collapse identity: with the isotropic stiffness tensor,
# alpha_qP -> p and alpha_qSV -> s (substep H.b unit oracle), so
# row 1 reduces bit-exactly to (M11, M12, M13) of
# :func:`_modal_determinant_n0` -- the floating-point oracle for
# this row builder.


def _modal_row1_at_a_vti(
    kz: float,
    omega: float,
    *,
    c11: float,
    c13: float,
    c33: float,
    c44: float,
    c66: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    radiating: tuple[bool, bool, bool] = (False, False, False),
) -> np.ndarray:
    r"""
    Row 1 of the n=0 VTI Stoneley modal determinant evaluated at
    the borehole wall ``r = a``.

    Encodes the radial-displacement continuity BC
    ``u_r^{(f)}(a) - u_r^{(s)}(a) = 0`` (cos sector). Returns the
    three post-rescale coefficients in the column order
    ``[A | B_qP, C_qSV]``.

    Parameters
    ----------
    kz : float
        Trial axial wavenumber (rad / m).
    omega : float
        Angular frequency (rad / s).
    c11, c13, c33, c44, c66 : float
        VTI stiffness tensor entries (Pa).
    rho : float
        Formation density (kg / m^3). Carried for signature
        uniformity; not used by row 1 (no stress / no Lame term).
    vf : float
        Fluid velocity (m/s).
    rho_f : float
        Fluid density (kg/m^3).
    a : float
        Borehole radius (m).

    Returns
    -------
    ndarray, shape (3,) complex
        Coefficients of (A, B_qP, C_qSV) in row 1. Real-valued in
        the bound regime.

    See Also
    --------
    _modal_determinant_n0 : Isotropic n=0 form. At isotropic-
        collapse C-matrix, ``row[0] = M11``, ``row[1] = M12``,
        ``row[2] = M13`` bit-exactly.
    _radial_wavenumbers_vti : Provides ``alpha_qP`` and
        ``alpha_qSV`` (the Christoffel roots).
    """
    F_f = float(np.sqrt(kz * kz - (omega / vf) ** 2))
    alpha_qP, alpha_qSV, _ = _radial_wavenumbers_vti_complex(
        kz,
        omega,
        c11=c11,
        c13=c13,
        c33=c33,
        c44=c44,
        c66=c66,
        rho=rho,
        radiating=radiating,
    )

    I1_Ff_a = float(special.iv(1, F_f * a))
    _, K1_qP_a = _k_or_hankel(0, alpha_qP, a, leaky=radiating[0])
    _, K1_qSV_a = _k_or_hankel(0, alpha_qSV, a, leaky=radiating[1])

    row: np.ndarray = np.zeros(3, dtype=complex)
    # A column: fluid u_r contribution (matches M11 at any C-matrix).
    row[0] = F_f * I1_Ff_a / (rho_f * omega**2)
    # B_qP column (matches M12 at isotropic limit alpha_qP -> p).
    row[1] = +alpha_qP * K1_qP_a
    # C_qSV column post-rescale (col-by-(-i) cancels the +i factor).
    row[2] = +kz * K1_qSV_a
    return row


# =====================================================================
# Substep H.c.1.b -- polarization-ratio helper + row 2 (sigma_rr at r=a)
# =====================================================================
#
# BC2: ``-(sigma_rr^{(s)}(a) + P^{(f)}(a)) = 0`` (cos-sector
# normal-stress balance at the borehole wall; row negated for
# visual parallel with the n=0 isotropic form). Algebraically the
# heaviest row of the n=0 VTI determinant: combines the C-matrix
# Lame-replacement at the K_0 coefficient with the polarization-
# ratio-mediated u_z contribution.
#
# **VTI sigma_rr formula at n=0** (axisymmetric):
#
#       sigma_rr = C11 d_r u_r + (C11 - 2 C66) (u_r / r) + C13 d_z u_z
#
# The middle term ``(C11 - 2 C66) (u_r / r)`` is the slot through
# which **C66 enters the n=0 modal determinant** -- via the
# ``epsilon_theta_theta = u_r / r`` strain in the cylindrical
# Hooke law. (At isotropic C11 - 2 C66 = lambda; the term reduces
# to ``lambda u_r / r`` and combines with the divergence into the
# standard ``lambda div(u) + 2 mu d_r u_r`` form.)
#
# This is what gives the Norris 1990 LF closed-form
# ``S_ST^2 = 1/V_f^2 + rho_f / C66`` its C66 dependence at n=0:
# the matrix entries carry C66 through the u_r/r slot, and the
# LF expansion of the Stoneley root extracts that dependence.
#
# **Polarization ratio** for each Christoffel root:
#
#   gamma_qX = u_z / u_r |_qX
#            = -i (C11 alpha_qX^2 - C44 k_z^2 + rho omega^2)
#               / [(C13 + C44) alpha_qX k_z]
#
# Independent of r-component normalization choice. At isotropic
# limit: gamma_qP -> -i k_z / p, gamma_qSV -> -i s / k_z.
#
# **Stress factor Q_qX** (the K_0 coefficient combination):
#
#   Q_qX = (C44 (C11 alpha_qX^2 + C13 k_z^2) - C13 rho omega^2)
#          / (C13 + C44)
#
# At isotropic limit: Q_qP -> mu (2 k_z^2 - k_S^2) (the standard
# Lame combination); Q_qSV -> 2 mu s^2.
#
# **Row 2 entries**:
#
#   A column: M21 = -I_0(F_f a)  (same as isotropic)
#
#   B_qP column (post-rescale; no row scaling, no C-col scaling):
#       row[1] = -Q_qP * K_0(alpha_qP a) - 2 C66 alpha_qP K_1(alpha_qP a) / a
#
#   C_qSV column (post-rescale via col-by-(-i)):
#       row[2] = -k_z Q_qSV / alpha_qSV * K_0(alpha_qSV a)
#                - 2 k_z C66 K_1(alpha_qSV a) / a
#
# Isotropic-collapse identity: row[1] -> M22 = -mu (2 k_z^2 -
# k_S^2) K_0(p a) - 2 mu p K_1(p a) / a; row[2] -> M23 =
# -2 k_z mu (s K_0(s a) + K_1(s a) / a). Verified algebraically
# above; the tests in H.c.1.b confirm to floating-point precision.


def _polarization_ratio_uz_over_ur_vti(
    alpha_qX: float,
    kz: float,
    omega: float,
    *,
    c11: float,
    c13: float,
    c44: float,
    rho: float,
) -> complex:
    r"""
    Christoffel-eigenvector polarization ratio
    ``gamma_qX = u_z / u_r`` at the qX Christoffel root.

    Independent of any normalization choice for the (u_r, u_z)
    eigenvector; multiply by the chosen u_r normalization to
    recover the absolute u_z component.

    Formula:
        gamma_qX = -i (C11 alpha_qX^2 - C44 k_z^2 + rho omega^2)
                   / [(C13 + C44) alpha_qX k_z]

    Isotropic limit (verified in H.c.1.b tests):
        gamma_qP  -> -i k_z / p
        gamma_qSV -> -i s / k_z

    Parameters
    ----------
    alpha_qX : float
        Christoffel root for the qX branch (qP or qSV).
    kz, omega : float
        Axial wavenumber and angular frequency.
    c11, c13, c44 : float
        VTI stiffness coefficients (Pa).
    rho : float
        Formation density (kg/m^3).

    Returns
    -------
    complex
        The polarization ratio. Imaginary in general; the actual
        u_z normalization in the row builders combines this with
        the chosen u_r normalization (e.g., -alpha_qP for qP,
        -i k_z for qSV).
    """
    common = c11 * alpha_qX * alpha_qX - c44 * kz * kz + rho * omega * omega
    return -1j * common / ((c13 + c44) * alpha_qX * kz)


def _modal_row2_at_a_vti(
    kz: float,
    omega: float,
    *,
    c11: float,
    c13: float,
    c33: float,
    c44: float,
    c66: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    radiating: tuple[bool, bool, bool] = (False, False, False),
) -> np.ndarray:
    r"""
    Row 2 of the n=0 VTI Stoneley modal determinant at ``r = a``.

    Encodes the negated normal-stress balance BC
    ``-(sigma_rr^{(s)}(a) + P^{(f)}(a)) = 0`` in the cos sector.
    Returns the three post-rescale coefficients in column order
    ``[A | B_qP, C_qSV]``.

    Parameters
    ----------
    kz, omega : float
        Axial wavenumber and angular frequency.
    c11, c13, c33, c44, c66 : float
        VTI stiffness tensor entries (Pa). All used here:
        ``c11, c13, c44`` set Q_qX and the polarization ratio;
        ``c66`` enters via the ``(C11 - 2 C66) u_r/r`` slot in
        sigma_rr (the Norris 1990 C66-coupling slot at n=0).
        ``c33`` enters indirectly via the Christoffel roots
        from :func:`_radial_wavenumbers_vti`.
    rho : float
        Formation density (kg/m^3).
    vf, rho_f : float
        Fluid velocity and density.
    a : float
        Borehole radius (m).

    Returns
    -------
    ndarray, shape (3,) complex
        Coefficients of (A, B_qP, C_qSV) in row 2. Real-valued
        in the bound regime.

    See Also
    --------
    _modal_determinant_n0 : Isotropic n=0 form. At isotropic-
        collapse C-matrix, ``row[0] = M21``, ``row[1] = M22``,
        ``row[2] = M23`` bit-exactly.
    _polarization_ratio_uz_over_ur_vti : The Christoffel
        polarization ratio (used internally; surfaced for the
        H.c.1.b tests).
    """
    F_f = float(np.sqrt(kz * kz - (omega / vf) ** 2))
    alpha_qP, alpha_qSV, _ = _radial_wavenumbers_vti_complex(
        kz,
        omega,
        c11=c11,
        c13=c13,
        c33=c33,
        c44=c44,
        c66=c66,
        rho=rho,
        radiating=radiating,
    )

    I0_Ff_a = float(special.iv(0, F_f * a))
    K0_qP_a, K1_qP_a = _k_or_hankel(0, alpha_qP, a, leaky=radiating[0])
    K0_qSV_a, K1_qSV_a = _k_or_hankel(0, alpha_qSV, a, leaky=radiating[1])

    rho_omega_sq = rho * omega * omega
    # Stress factor Q_qX = (C44 (C11 alpha_qX^2 + C13 kz^2) - C13 rho omega^2)
    #                       / (C13 + C44)
    # Reduces to mu (2 kz^2 - kS^2) at isotropic limit for qP, and
    # 2 mu s^2 for qSV.
    q_qP = (c44 * (c11 * alpha_qP * alpha_qP + c13 * kz * kz) - c13 * rho_omega_sq) / (
        c13 + c44
    )
    q_qSV = (
        c44 * (c11 * alpha_qSV * alpha_qSV + c13 * kz * kz) - c13 * rho_omega_sq
    ) / (c13 + c44)

    row: np.ndarray = np.zeros(3, dtype=complex)
    # A column: -P^{(f)}(a) = -A I_0(F_f a) (same as isotropic).
    row[0] = -I0_Ff_a
    # B_qP column: -Q_qP K_0 - 2 C66 alpha_qP K_1/a.
    # In isotropic Q_qP = mu (2 kz^2 - kS^2) and 2 C66 = 2 mu, so
    # row[1] -> -mu ((2 kz^2 - kS^2) K_0(p a) + 2 p K_1(p a)/a) = M22.
    row[1] = -q_qP * K0_qP_a - 2.0 * c66 * alpha_qP * K1_qP_a / a
    # C_qSV column post-rescale: -kz (Q_qSV / alpha_qSV) K_0 - 2 kz C66 K_1/a.
    # In isotropic Q_qSV / alpha_qSV = 2 mu s^2 / s = 2 mu s, so
    # row[2] -> -kz (2 mu s K_0 + 2 mu K_1/a) = -2 mu kz (s K_0 + K_1/a) = M23.
    row[2] = -kz * (q_qSV / alpha_qSV) * K0_qSV_a - 2.0 * kz * c66 * K1_qSV_a / a
    return row


# =====================================================================
# Substep H.c.1.c -- row 3 (sigma_rz at r=a)
# =====================================================================
#
# BC3: ``sigma_rz^{(s)}(a) = 0`` (cos-sector axial-shear vanishing
# at the borehole wall; fluid carries no shear so column A is
# identically zero). Z-derivative-bearing row; gets the FULL
# substep-H.a.6 phase rescale (row * i AND col-by-(-i) on C_qSV).
#
# **VTI sigma_rz formula** (same form as isotropic with mu -> C44):
#
#       sigma_rz = C44 (d_z u_r + d_r u_z)
#
# C44 (vertical shear modulus) is the only stiffness coefficient
# that appears explicitly in the constitutive relation; C66 does
# NOT enter row 3. The Christoffel roots and the polarization
# ratio (which depend on the full C-matrix) do enter through u_r
# and u_z, however.
#
# **Stress factor P_qX** for sigma_rz (single combination,
# distinct from the Q_qX of substep H.c.1.b):
#
#   P_qX = C11 alpha_qX^2 + C13 k_z^2 + rho omega^2
#
# In isotropic limit (verified algebraically):
#   P_qP  -> 2 (lambda + mu) k_z^2     (using p^2 = k_z^2 -
#                                        rho omega^2 / (lambda + 2 mu))
#   P_qSV -> (lambda + mu) (2 k_z^2 - k_S^2)
#
# **Row 3 entries (post-rescale)**:
#
#   A column: 0  (fluid no shear)
#
#   B_qP column:
#       row[1] = C44 alpha_qP P_qP / [(C13 + C44) k_z] * K_1(alpha_qP a)
#       Isotropic limit -> 2 mu p k_z K_1(p a) = M32.
#
#   C_qSV column:
#       row[2] = C44 P_qSV / (C13 + C44) * K_1(alpha_qSV a)
#       Isotropic limit -> mu (2 k_z^2 - k_S^2) K_1(s a) = M33.
#
# The phase rescale (row * i + col-by-(-i) on C) is what flips
# the pre-rescale ``-i`` factor on B_qP and the ``+real`` factor
# on C_qSV into the post-rescale real entries above. Forgetting
# either rescale is the most direct H.a.6 transcription error;
# the row.imag == 0 test catches it.


def _modal_row3_at_a_vti(
    kz: float,
    omega: float,
    *,
    c11: float,
    c13: float,
    c33: float,
    c44: float,
    c66: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    radiating: tuple[bool, bool, bool] = (False, False, False),
) -> np.ndarray:
    r"""
    Row 3 of the n=0 VTI Stoneley modal determinant at ``r = a``.

    Encodes BC3 ``sigma_rz^{(s)}(a) = 0`` (cos sector). The
    fluid-no-shear identity makes column A identically zero.
    Returns the three post-rescale coefficients in column order
    ``[A | B_qP, C_qSV]``.

    Parameters
    ----------
    kz, omega : float
        Axial wavenumber and angular frequency.
    c11, c13, c33, c44, c66 : float
        VTI stiffness tensor entries (Pa). ``c66`` is carried for
        signature uniformity but does NOT enter row 3 (sigma_rz
        depends only on C44).
    rho : float
        Formation density (kg/m^3).
    vf, rho_f : float
        Fluid velocity / density. Carried for signature uniformity
        but not used by row 3 (fluid carries no shear).
    a : float
        Borehole radius (m).

    Returns
    -------
    ndarray, shape (3,) complex
        Coefficients of (A, B_qP, C_qSV) in row 3. Real-valued in
        the bound regime.

    See Also
    --------
    _modal_determinant_n0 : Isotropic n=0 form. At isotropic
        collapse, ``row[0] = M31 = 0``, ``row[1] = M32``,
        ``row[2] = M33`` bit-exactly.
    """
    del vf, rho_f  # fluid carries no shear; not used by row 3
    alpha_qP, alpha_qSV, _ = _radial_wavenumbers_vti_complex(
        kz,
        omega,
        c11=c11,
        c13=c13,
        c33=c33,
        c44=c44,
        c66=c66,
        rho=rho,
        radiating=radiating,
    )
    _, K1_qP_a = _k_or_hankel(0, alpha_qP, a, leaky=radiating[0])
    _, K1_qSV_a = _k_or_hankel(0, alpha_qSV, a, leaky=radiating[1])

    rho_omega_sq = rho * omega * omega
    # Stress factor P_qX = C11 alpha_qX^2 + C13 kz^2 + rho omega^2.
    # Distinct from the Q_qX of row 2; appears here through the
    # sigma_rz combination C44 (d_z u_r + d_r u_z).
    p_qP = c11 * alpha_qP * alpha_qP + c13 * kz * kz + rho_omega_sq
    p_qSV = c11 * alpha_qSV * alpha_qSV + c13 * kz * kz + rho_omega_sq

    row: np.ndarray = np.zeros(3, dtype=complex)
    # A column: fluid carries no shear; sigma_rz from A is zero.
    row[0] = 0.0
    # B_qP column (post-rescale via row * i):
    #   pre-rescale: -i C44 alpha_qP P_qP / [(C13 + C44) k_z] K_1
    #   post: +C44 alpha_qP P_qP / [(C13 + C44) k_z] K_1
    # Isotropic limit P_qP -> 2 (lambda + mu) k_z^2, so
    #   row[1] -> 2 mu p k_z K_1(p a) = M32.
    row[1] = c44 * alpha_qP * p_qP / ((c13 + c44) * kz) * K1_qP_a
    # C_qSV column (post-rescale via row * i AND col-by-(-i),
    # net factor +1):
    #   pre-rescale: +C44 P_qSV / (C13 + C44) K_1 (already real)
    #   post-rescale: same value (i * (-i) = 1).
    # Isotropic limit P_qSV -> (lambda + mu)(2 k_z^2 - k_S^2), so
    #   row[2] -> mu (2 k_z^2 - k_S^2) K_1(s a) = M33.
    row[2] = c44 * p_qSV / (c13 + c44) * K1_qSV_a
    return row


# =====================================================================
# Substep H.c.1.d -- assembly into _modal_determinant_n0_vti
# =====================================================================
#
# Stacks the three row builders (H.c.1.a, H.c.1.b, H.c.1.c) into
# the 3x3 VTI Stoneley modal matrix and returns the determinant
# as a real scalar.
#
# Each row builder applies the substep-H.a.6 phase rescale
# internally (row * i for the z-derivative-bearing row 3, col-by-
# (-i) on the C_qSV column), so the assembled matrix is real-valued
# in the bound regime and ``np.linalg.det`` returns the real
# determinant directly.
#
# Reduces to :func:`_modal_determinant_n0` (up to an irrelevant
# overall scale factor) when the C-matrix is isotropic; the
# determinant *root* coincides at floating-point precision. The
# determinant *value* differs by a constant pre-factor of no
# physical relevance -- the H.c.2 integration test
# (``stoneley_dispersion_vti`` end-to-end) is the floating-point
# oracle that closes this assembly.


def _modal_determinant_n0_vti(
    kz: complex,
    omega: float,
    *,
    c11: float,
    c13: float,
    c33: float,
    c44: float,
    c66: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
) -> float:
    r"""
    3x3 axisymmetric modal determinant for the n=0 (Stoneley)
    VTI dispersion.

    Stacks the three row builders from H.c.1.a / H.c.1.b /
    H.c.1.c into the 3x3 VTI Stoneley modal matrix, applies the
    substep-H.a.6 phase rescale (already absorbed into each
    row), and returns the determinant as a real scalar.

    Reduces to :func:`_modal_determinant_n0` (sharing the same
    root in ``k_z``, with an irrelevant overall scale) when the
    stiffness tensor is isotropic. The H.c.2 integration test in
    :func:`stoneley_dispersion_vti` is the floating-point oracle
    that anchors this assembly.

    Parameters
    ----------
    kz : float
        Trial axial wavenumber (rad / m). Must lie in the bound
        regime ``kz > omega / min(V_Sv, V_Sh, V_f)``.
    omega : float
        Angular frequency (rad / s).
    c11, c13, c33, c44, c66 : float
        VTI stiffness tensor entries (Pa).
    rho : float
        Formation density (kg/m^3).
    vf, rho_f : float
        Borehole-fluid velocity and density.
    a : float
        Borehole radius (m).

    Returns
    -------
    float
        ``det(M)`` of the 3x3 VTI Stoneley modal matrix, real-
        valued in the bound regime. NaN outside the bound regime.
    """
    kz = _as_real_kz(kz, caller="_modal_determinant_n0_vti")
    rows = [
        _modal_row1_at_a_vti(
            kz,
            omega,
            c11=c11,
            c13=c13,
            c33=c33,
            c44=c44,
            c66=c66,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            a=a,
        ),
        _modal_row2_at_a_vti(
            kz,
            omega,
            c11=c11,
            c13=c13,
            c33=c33,
            c44=c44,
            c66=c66,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            a=a,
        ),
        _modal_row3_at_a_vti(
            kz,
            omega,
            c11=c11,
            c13=c13,
            c33=c33,
            c44=c44,
            c66=c66,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            a=a,
        ),
    ]
    M = np.vstack(rows)
    # Each row is real-valued post-rescale; imaginary parts are
    # zero to floating-point precision in the bound regime. Take
    # the real part to discard sub-machine-epsilon imaginary noise.
    return float(np.linalg.det(M.real))


# =====================================================================
# Substep H.d.1 -- row 1 of the n=1 VTI flexural modal determinant (r=a)
# =====================================================================
#
# BC1: ``u_r^{(f)}(a) - u_r^{(s)}(a) = 0`` (cos-sector continuity of
# radial displacement at the borehole wall, dipole order). First
# row of the 4x4 n=1 VTI flexural modal determinant; same row
# layout as :func:`_modal_determinant_n1`, with the Christoffel
# roots ``alpha_qP``, ``alpha_qSV``, ``alpha_SH`` from
# :func:`_radial_wavenumbers_vti` replacing the isotropic ``p``,
# ``s``, ``s``.
#
# **Bessel-index pattern at n=1**: the natural index of each
# scalar / vector potential shifts by 1 from n=0:
#
#       Potential                n=0           n=1
#       ----------------------   ----------    ----------
#       qP scalar phi            K_0(alpha_qP r)  K_1(alpha_qP r)
#       qSV vector-theta psi     K_1(alpha_qSV r) K_1(alpha_qSV r)
#       SH vector-z psi_z        n/a              K_1(alpha_SH r)
#
# At n=1 the d_r of the qP scalar gives a TWO-TERM result
# ``K_1' = -K_0 - K_1/(alpha_qP r)``, so the qP u_r contribution
# carries the ``+alpha_qP K_0 + K_1/r`` combination at r=a (vs the
# single ``-alpha_qP K_1`` term at n=0).
#
# **Cross-sector coupling**: the SH amplitude D enters the cos-
# sector u_r BC via ``(1/r) d_theta(psi_z)`` (sin -> cos
# conversion via d_theta). New at n>=1; absent at n=0 by
# axisymmetry.
#
# Polarization ratio does NOT enter row 1: the contributions are
# all from u_r alone, and the r-component normalizations of qP,
# qSV, SH match the isotropic conventions directly.
#
# **Row 1 entries (post-rescale)**:
#
#       row1 = [
#           +(F_f I_0(F_f a) - I_1(F_f a)/a) / (rho_f omega^2),  # A -> M11
#           +alpha_qP K_0(alpha_qP a) + K_1(alpha_qP a) / a,     # B_qP -> M12
#           +k_z K_1(alpha_qSV a),                                # C_qSV -> M13
#           -K_1(alpha_SH a) / a,                                 # D_SH -> M14
#       ]
#
# Phase rescale: row 1 is NOT z-derivative-bearing (no row * i).
# Column C_qSV by ``-i`` cancels the ``+i k_z`` factor in the
# pre-rescale C entry. Columns A and D are not rescaled.


def _vti_polarisation_ratio(
    alpha: complex,
    kz: complex,
    omega: float,
    *,
    c13: float,
    c33: float,
    c44: float,
    rho: float,
) -> complex:
    r"""
    The ``gamma`` that fixes a qP / qSV column's axial displacement.

    For fields going as ``e^{i(n theta + k_z z)}`` the coupled qP / qSV
    pair admits the potential form

        ``u_r = d(phi)/dr``,  ``u_theta = (i n / r) phi``,
        ``u_z = i k_z gamma phi``,     ``phi = K_n(alpha r)``,

    and substituting it into the axial equation of motion leaves a
    single condition, whose solution is

        ``gamma = -alpha^2 (C13 + C44)
                  / (rho omega^2 + C44 alpha^2 - C33 k_z^2)``.

    The isotropic limit is the check that this is the right object.
    With ``C11 = C33 = lambda + 2 mu``, ``C44 = mu``, ``C13 = lambda``
    it returns exactly ``1`` at the P root -- recovering ``u = grad
    phi`` -- and exactly ``alpha^2 / k_z^2`` at the S root, which is
    the Hansen form ``u = curl curl(chi z)`` the isotropic assembly
    already uses. Both are reproduced to floating point.

    Parameters
    ----------
    alpha : complex
        Radial wavenumber of the column (rad / m), from
        :func:`_radial_wavenumbers_vti_complex`.
    kz : complex
        Axial wavenumber (rad / m).
    omega : float
        Angular frequency (rad / s).
    c13, c33, c44 : float
        VTI stiffnesses (Pa).
    rho : float
        Formation density (kg / m^3).

    Returns
    -------
    complex
        ``gamma``, dimensionless.
    """
    numerator = -(complex(alpha) ** 2) * (c13 + c44)
    denominator = (
        rho * omega * omega + c44 * complex(alpha) ** 2 - c33 * complex(kz) ** 2
    )
    return complex(numerator / denominator)


def _formation_displacements_n1_vti(
    kz: complex,
    omega: float,
    *,
    c11: float,
    c13: float,
    c33: float,
    c44: float,
    c66: float,
    rho: float,
    r: float,
    radiating: tuple[bool, bool, bool] = (False, False, False),
) -> np.ndarray:
    r"""
    The three formation displacement columns at radius ``r``, ``n = 1``.

    Returns ``u_r``, ``u_theta`` and ``u_z`` for the qP, qSV and SH
    columns. The open-hole assembly never needed the last two: its
    boundary conditions at ``r = a`` are ``u_r`` continuity with the
    fluid plus three tractions. A **layered** stack needs continuity of
    all six quantities across ``r = b``, which is what this supplies
    (roadmap A.12).

    qP and qSV come from the potential form in
    :func:`_vti_polarisation_ratio`; SH is decoupled and carries
    ``u = curl(psi z)``, so ``u_z`` is identically zero for it.

    **Normalisation matches the existing rows**, so these columns
    compose with them rather than needing a rescale: the qP column
    carries ``-1``, the qSV column ``-k_z`` and the SH column ``+i``,
    which is what ``_modal_row1_at_a_n1_vti`` already uses. Applied per
    column, so all three components of one column share one factor and
    the physics is untouched.

    Parameters
    ----------
    kz : complex
        Axial wavenumber (rad / m).
    omega : float
        Angular frequency (rad / s).
    c11, c13, c33, c44, c66 : float
        VTI stiffnesses (Pa).
    rho : float
        Formation density (kg / m^3).
    r : float
        Radius at which to evaluate (m) -- ``a`` for an open hole, the
        outermost layer radius ``b`` for a cased stack.
    radiating : tuple of bool
        Per-wave ``(qP, qSV, SH)`` outgoing-branch selection, as for
        :func:`_radial_wavenumbers_vti_complex`.

    Returns
    -------
    ndarray
        ``(3, 3)`` complex array. Rows are ``(u_r, u_theta, u_z)``;
        columns are ``(qP, qSV, SH)``.

    Notes
    -----
    Verified two ways. The full field satisfies the VTI equations of
    motion in cylindrical coordinates to ~1e-9 relative under
    fourth-order finite differences, at ``n = 1`` and ``n = 2`` across
    three Thomsen media; and its ``u_r`` row reproduces the already
    validated ``_modal_row1_at_a_n1_vti`` formation entries exactly.
    """
    n = 1
    alpha_qP, alpha_qSV, alpha_SH = _radial_wavenumbers_vti_complex(
        kz,
        omega,
        c11=c11,
        c13=c13,
        c33=c33,
        c44=c44,
        c66=c66,
        rho=rho,
        radiating=radiating,
    )

    def _pair(alpha: complex, leaky: bool) -> tuple[complex, complex]:
        """``(K_n(alpha r), dK_n/dr)`` on the column's own branch."""
        k_n, k_n1 = _k_or_hankel(n, alpha, r, leaky=leaky)
        return k_n, alpha * (-k_n1 + (n / (alpha * r)) * k_n)

    columns = []
    for alpha, leaky, scale in (
        (alpha_qP, radiating[0], -1.0 + 0.0j),
        (alpha_qSV, radiating[1], -complex(kz)),
    ):
        phi, dphi = _pair(alpha, leaky)
        gamma = _vti_polarisation_ratio(
            alpha, kz, omega, c13=c13, c33=c33, c44=c44, rho=rho
        )
        columns.append(
            scale * np.array([dphi, (1j * n / r) * phi, 1j * complex(kz) * gamma * phi])
        )

    psi, dpsi = _pair(alpha_SH, radiating[2])
    columns.append(1j * np.array([(1j * n / r) * psi, -dpsi, 0.0 + 0.0j]))

    return np.column_stack(columns)


def _fluid_bessels_n1_vti(
    kz: complex,
    omega: float,
    vf: float,
    a: float,
) -> tuple[complex, complex, complex]:
    r"""
    Fluid radial wavenumber and its Bessel values, in either regime.

    Returns ``(F_f, I_0(F_f a), I_1(F_f a))`` where
    ``F_f = sqrt(k_z^2 - (omega / V_f)^2)``.

    **Slow-formation / bound regime** (``F_f^2 > 0``): the fluid field
    decays in ``r`` and every quantity is real. This branch performs
    exactly the same three calls the row builders used before this
    helper existed, in the same order and on the same real arguments,
    so its results are **bit-identical** to the pre-existing path. That
    matters: the slow-formation VTI flexural curve is tied to Ellefsen
    fig 4 at 0.30 % RMS, and routing it through complex arithmetic
    instead would perturb it at the 1e-15 level for no benefit.

    **Fast-formation regime** (``F_f^2 < 0``): ``F_f`` is purely
    imaginary, the fluid pressure is oscillatory rather than decaying,
    and ``I_n(F_f a)`` continues analytically to ``i^n J_n(|F_f| a)``.
    :func:`scipy.special.iv` performs that continuation directly on a
    complex argument, so no separate ``J_n`` branch is needed.

    At exactly ``F_f^2 == 0`` the bound branch is taken; ``I_n(0)`` is
    finite (1 and 0), so this is continuous.

    **Complex ``k_z``** (A.11 phase 4) needs no third branch, and in
    particular no outgoing form. The fluid occupies ``0 <= r <= a``,
    so its boundary condition is regularity at the origin rather than
    radiation at infinity, and ``I_n`` -- entire for integer ``n`` --
    supplies that for any complex argument. What the leaky case adds
    is only that ``F_f^2`` is now complex rather than real of either
    sign.

    The sign of ``F_f`` is immaterial to the modes. The fluid column
    enters the ``n = 1`` matrix in two rows, ``(F_f I_0 - I_1 / a)``
    and ``-I_1``, and both are **odd** in ``F_f`` (``I_0`` is even,
    ``I_1`` odd), so flipping the branch negates the whole column and
    multiplies the determinant by ``-1`` without moving a root. The
    principal root is nonetheless the right one to take: over the
    leaky search region ``Im(k_z) >= 0`` with ``c > V_f`` the argument
    ``F_f^2`` sits in the closed upper half plane, where the principal
    square root is continuous, and its value on the negative real axis
    (``Im(k_z) = 0``) is the limit from above. So the determinant does
    not pick up a spurious sign flip anywhere the search goes.

    Parameters
    ----------
    kz : complex
        Trial axial wavenumber (rad / m). Real or complex.
    omega : float
        Angular frequency (rad / s).
    vf : float
        Borehole-fluid velocity (m / s).
    a : float
        Borehole radius (m).

    Returns
    -------
    tuple of complex
        ``(F_f, I_0(F_f a), I_1(F_f a))``. Real-valued (zero imaginary
        part) in the bound regime; genuinely complex otherwise.
    """
    arg2 = complex(kz) * complex(kz) - (omega / vf) ** 2
    if arg2.imag == 0.0 and arg2.real >= 0.0:
        F_f = float(np.sqrt(arg2.real))
        return (
            F_f,
            float(special.iv(0, F_f * a)),
            float(special.iv(1, F_f * a)),
        )
    F_f_c = np.sqrt(arg2)
    return (
        complex(F_f_c),
        complex(special.iv(0, F_f_c * a)),
        complex(special.iv(1, F_f_c * a)),
    )


def _modal_row1_at_a_n1_vti(
    kz: complex,
    omega: float,
    *,
    c11: float,
    c13: float,
    c33: float,
    c44: float,
    c66: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    radiating: tuple[bool, bool, bool] = (False, False, False),
) -> np.ndarray:
    r"""
    Row 1 of the n=1 VTI flexural modal determinant evaluated at
    the borehole wall ``r = a``.

    Encodes BC1 ``u_r^{(f)}(a) - u_r^{(s)}(a) = 0`` (cos sector,
    dipole order). Returns the four post-rescale coefficients in
    column order ``[A | B_qP, C_qSV, D_SH]``.

    Parameters
    ----------
    kz, omega : float
        Axial wavenumber and angular frequency.
    c11, c13, c33, c44, c66 : float
        VTI stiffness tensor entries (Pa).
    rho : float
        Formation density (kg/m^3). Carried for signature
        uniformity; not used by row 1 (no stress / no Lame term).
    vf, rho_f : float
        Fluid velocity and density.
    a : float
        Borehole radius (m).

    Returns
    -------
    ndarray, shape (4,) complex
        Coefficients of (A, B_qP, C_qSV, D_SH) in row 1. Real-
        valued in the bound regime.

    See Also
    --------
    _modal_determinant_n1 : Isotropic n=1 form. At isotropic-
        collapse, ``row[0] = M11``, ``row[1] = M12``,
        ``row[2] = M13``, ``row[3] = M14`` bit-exactly.
    _modal_row1_at_a_vti : The n=0 (Stoneley) counterpart. The
        n=1 form adds the D_SH column (cross-coupled via
        ``(1/r) d_theta psi_z``) and shifts the Bessel index from
        K_0 to K_1 for the qP scalar potential.
    """
    F_f, I0_Ff_a, I1_Ff_a = _fluid_bessels_n1_vti(kz, omega, vf, a)
    alpha_qP, alpha_qSV, alpha_SH = _radial_wavenumbers_vti_complex(
        kz,
        omega,
        c11=c11,
        c13=c13,
        c33=c33,
        c44=c44,
        c66=c66,
        rho=rho,
        radiating=radiating,
    )

    K0_qP_a, K1_qP_a = _k_or_hankel(0, alpha_qP, a, leaky=radiating[0])
    K0_qSV_a, K1_qSV_a = _k_or_hankel(0, alpha_qSV, a, leaky=radiating[1])
    _, K1_SH_a = _k_or_hankel(0, alpha_SH, a, leaky=radiating[2])

    row: np.ndarray = np.zeros(4, dtype=complex)
    # A column: fluid u_r contribution (matches M11 at any C-matrix).
    row[0] = (F_f * I0_Ff_a - I1_Ff_a / a) / (rho_f * omega**2)
    # B_qP column: qP scalar's two-term radial derivative
    # ``alpha_qP K_0 + K_1/a`` (matches M12 at alpha_qP -> p).
    row[1] = +alpha_qP * K0_qP_a + K1_qP_a / a
    # C_qSV column post-rescale (col-by-(-i) cancels the +i factor;
    # matches M13 at alpha_qSV -> s). Same two-term radial
    # derivative as B_qP -- roadmap A.8. Both Christoffel branches
    # carry the SAME field structure ``u_h = grad_h Phi``,
    # ``u_z = -gamma alpha Phi``, so u_r is ``d_r Phi`` for qSV
    # exactly as it is for qP. At n=0 that derivative happens to be
    # proportional to K_1, which is why the single-K_1 form was
    # correct there and is not correct here.
    row[2] = +kz * (alpha_qSV * K0_qSV_a + K1_qSV_a / a)
    # D_SH column: SH cross-coupling via (1/r) d_theta psi_z;
    # KEEP-sign on the K_1/r term per F.1.a.2 (matches M14 at
    # alpha_SH -> s).
    row[3] = -K1_SH_a / a
    return row


# =====================================================================
# Substep H.d.2 -- row 2 of the n=1 VTI flexural determinant (r=a)
# =====================================================================
#
# BC2: ``-(sigma_rr^{(s)}(a) + P^{(f)}(a)) = 0`` (cos sector,
# dipole order). Algebraically heaviest row of the n=1 VTI
# determinant: each column has multi-Bessel-term entries combining
# the C-matrix Lame replacement (Q_qX from H.c.1.b) with the n=1
# azimuthal-derivative-induced 4/r^2 factor.
#
# **VTI sigma_rr formula at n=1**:
#
#       sigma_rr = C11 epsilon_rr + (C11 - 2 C66) epsilon_theta_theta
#                  + C13 epsilon_zz
#       epsilon_theta_theta = u_r/r + (1/r) d_theta u_theta
#
# At n=1 the ``(1/r) d_theta u_theta`` term is non-zero (vs n=0
# axisymmetric) and feeds an EXTRA K_1/r^2 contribution that
# combines with the u_r/r term to give the ``4 K_1/a^2`` coefficient
# in the B_qP column. Same algebra as the existing isotropic
# substep 1.3.c ``2 K_1/r^2`` (from u_r/r) + ``K_1/r^2`` (from
# (1/r) d_theta u_theta) sum.
#
# **Q_qX reuse from H.c.1.b** (independent of azimuthal order n):
#
#       Q_qX = (C44 (C11 alpha_qX^2 + C13 k_z^2) - C13 rho omega^2)
#              / (C13 + C44)
#
# Reduces to mu (2 k_z^2 - k_S^2) for qP at isotropic limit; to
# 2 mu s^2 for qSV.
#
# **C66 enters at TWO slots in row 2** (distinct from H.c.1.b
# where C66 entered only one slot at n=0):
#
#   1. ``2 C66 alpha_qX K_0/a`` from the (C11 - 2 C66) coefficient
#      on the d_r u_r and u_r/r contributions in B_qP and C_qSV.
#   2. ``4 C66 K_1/a^2`` (B_qP) and ``2 C66 K_1/a`` (C_qSV) and
#      the entire 2 C66 prefactor on D_SH from the
#      (C11 - 2 C66) coefficient on the (1/r) d_theta u_theta
#      contribution.
#
# **D_SH column scales entirely with C66** -- pure
# (C11 - 2 C66) epsilon_theta_theta contribution; no Q_qX factor.
# Distinct mechanism from B_qP / C_qSV.
#
# **Row 2 entries (post-rescale)**:
#
#       row[0] = -I_1(F_f a)                                  # A -> M21
#
#       row[1] = -(  Q_qP K_1(alpha_qP a)
#                   + 2 C66 alpha_qP K_0(alpha_qP a) / a
#                   + 4 C66 K_1(alpha_qP a) / a^2 )            # B_qP -> M22
#
#       row[2] = -k_z (  Q_qSV K_1(alpha_qSV a)
#                       + 2 C66 alpha_qSV K_0(alpha_qSV a) / a
#                       + 4 C66 K_1(alpha_qSV a) / a^2 )       # C_qSV -> M23
#
#       row[3] = +2 C66 (  alpha_SH K_0(alpha_SH a) / a
#                         + 2 K_1(alpha_SH a) / a^2 )          # D_SH -> M24
#
# Phase rescale: row 2 is NOT z-derivative-bearing (no row * i).
# Column C_qSV by ``-i`` cancels the ``-i k_z`` factor in the
# pre-rescale C entry. Columns A, B_qP, D_SH are not rescaled
# (entries already real pre-rescale).


def _modal_row2_at_a_n1_vti(
    kz: complex,
    omega: float,
    *,
    c11: float,
    c13: float,
    c33: float,
    c44: float,
    c66: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    radiating: tuple[bool, bool, bool] = (False, False, False),
) -> np.ndarray:
    r"""
    Row 2 of the n=1 VTI flexural modal determinant evaluated at
    the borehole wall ``r = a``.

    Encodes the negated normal-stress balance BC
    ``-(sigma_rr^{(s)}(a) + P^{(f)}(a)) = 0`` (cos sector, dipole
    order). Returns the four post-rescale coefficients in column
    order ``[A | B_qP, C_qSV, D_SH]``.

    Parameters
    ----------
    kz, omega : float
        Axial wavenumber and angular frequency.
    c11, c13, c33, c44, c66 : float
        VTI stiffness tensor entries (Pa). All used:
        ``c11, c13, c44`` set Q_qX and the polarization ratio;
        ``c66`` enters via two distinct slots (the
        ``(C11 - 2 C66)`` epsilon_theta_theta coefficient at the
        ``4 K_1/a^2`` and ``2 K_0/a`` terms, plus the full
        D_SH column).
        ``c33`` enters indirectly via the Christoffel roots.
    rho : float
        Formation density (kg/m^3).
    vf, rho_f : float
        Fluid velocity and density.
    a : float
        Borehole radius (m).

    Returns
    -------
    ndarray, shape (4,) complex
        Coefficients of (A, B_qP, C_qSV, D_SH) in row 2. Real-
        valued in the bound regime.

    See Also
    --------
    _modal_determinant_n1 : Isotropic n=1 form. At isotropic-
        collapse, ``row[0] = M21``, ``row[1] = M22``, ``row[2] =
        M23``, ``row[3] = M24`` bit-exactly.
    _modal_row2_at_a_vti : The n=0 (Stoneley) counterpart from
        H.c.1.b. Q_qX formula is the same; n=1 adds the D_SH
        column and the extra K_1/r^2 azimuthal-derivative term.
    """
    _F_f, _I0_Ff_a, I1_Ff_a = _fluid_bessels_n1_vti(kz, omega, vf, a)
    alpha_qP, alpha_qSV, alpha_SH = _radial_wavenumbers_vti_complex(
        kz,
        omega,
        c11=c11,
        c13=c13,
        c33=c33,
        c44=c44,
        c66=c66,
        rho=rho,
        radiating=radiating,
    )

    K0_qP_a, K1_qP_a = _k_or_hankel(0, alpha_qP, a, leaky=radiating[0])
    K0_qSV_a, K1_qSV_a = _k_or_hankel(0, alpha_qSV, a, leaky=radiating[1])
    K0_SH_a, K1_SH_a = _k_or_hankel(0, alpha_SH, a, leaky=radiating[2])

    rho_omega_sq = rho * omega * omega
    # Q_qX = (C44 (C11 alpha_qX^2 + C13 kz^2) - C13 rho omega^2)
    #          / (C13 + C44)
    # Reuses the same formula from H.c.1.b (n=0); independent of
    # azimuthal order.
    q_qP = (c44 * (c11 * alpha_qP * alpha_qP + c13 * kz * kz) - c13 * rho_omega_sq) / (
        c13 + c44
    )
    q_qSV = (
        c44 * (c11 * alpha_qSV * alpha_qSV + c13 * kz * kz) - c13 * rho_omega_sq
    ) / (c13 + c44)

    row: np.ndarray = np.zeros(4, dtype=complex)
    # A column: matches M21 at any C-matrix.
    row[0] = -I1_Ff_a
    # B_qP column: three-term entry with Q_qP, 2 C66 azimuthal-deriv
    # slot at K_0/a, and 4 C66 azimuthal-deriv slot at K_1/a^2.
    row[1] = -(
        q_qP * K1_qP_a
        + 2.0 * c66 * alpha_qP * K0_qP_a / a
        + 4.0 * c66 * K1_qP_a / (a * a)
    )
    # C_qSV column post-rescale: same three-term shape as B_qP with
    # alpha_qP -> alpha_qSV and Q_qP -> Q_qSV -- roadmap A.8. The
    # old two-term ``Q_qSV/alpha_qSV K_0 + 2 C66 K_1/a`` form is the
    # n=0 entry with the Bessel index shifted; it does not follow
    # from d_r Phi at n=1.
    row[2] = -kz * (
        q_qSV * K1_qSV_a
        + 2.0 * c66 * alpha_qSV * K0_qSV_a / a
        + 4.0 * c66 * K1_qSV_a / (a * a)
    )
    # D_SH column: pure 2 C66 prefactor times (alpha_SH K_0/a +
    # 2 K_1/a^2). Distinct mechanism from B_qP / C_qSV (no Q
    # factor); the entire column scales with C66.
    row[3] = +2.0 * c66 * (alpha_SH * K0_SH_a / a + 2.0 * K1_SH_a / (a * a))
    return row


# =====================================================================
# Substep H.d.3 -- row 3 of the n=1 VTI flexural determinant (r=a)
# =====================================================================
#
# BC3: ``sigma_rtheta^{(s)}(a) = 0`` (sin sector, dipole order).
# Tangential-shear vanishing at the borehole wall. Unique sector
# pattern: sin-sector BC (vs cos for rows 1, 2, 4); fluid carries
# no shear so column A = 0.
#
# **VTI sigma_rtheta formula**:
#
#       sigma_rtheta = 2 C66 epsilon_rtheta
#                    = C66 ((1/r) d_theta u_r + d_r u_theta - u_theta/r)
#
# Pure C66 shear stress -- no Lame replacement, no Q_qX, no
# polarization-ratio factor (the polarization ratio determines
# u_z, but row 3 doesn't see u_z). This makes row 3 the simplest
# C-matrix-coupled row in H.d.
#
# **C66 enters every non-zero entry directly**:
#
#   B_qP: scales with 2 C66
#   C_qSV: scales with C66 (post-rescale ``k_z C66 K_1/a``)
#   D_SH: scales with -C66; outer prefactor times the SH shear
#         combination ``alpha_SH^2 K_1 + 2 alpha_SH K_0/a + 4 K_1/a^2``
#
# The ``alpha_SH^2 K_1`` direct term in D_SH is unique to row 3
# (no other row in the n=1 matrix has this term). Comes from
# ``d_r u_theta`` involving ``d_r [alpha_SH K_0(alpha_SH r)] =
# -alpha_SH^2 K_1``.
#
# Cross-sector mechanism: B and C contribute via cos -> sin
# d_theta conversion in (1/r) d_theta u_r; D contributes
# directly (psi_z is naturally sin sector).
#
# **Row 3 entries (post-rescale)**:
#
#       row[0] = 0                                          # A (no shear)
#       row[1] = +2 C66 (alpha_qP K_0(alpha_qP a)/a
#                        + 2 K_1(alpha_qP a)/a^2)            # B_qP -> M32
#       row[2] = +2 k_z C66 (alpha_qSV K_0(alpha_qSV a)/a
#                            + 2 K_1(alpha_qSV a)/a^2)         # C_qSV -> M33
#       row[3] = -C66 (alpha_SH^2 K_1(alpha_SH a)
#                      + 2 alpha_SH K_0(alpha_SH a)/a
#                      + 4 K_1(alpha_SH a)/a^2)              # D_SH -> M34
#
# Phase rescale: row 3 is NOT z-derivative-bearing (no row * i).
# Column C_qSV by ``-i`` cancels the ``+i k_z`` factor in the
# pre-rescale C entry.


def _modal_row3_at_a_n1_vti(
    kz: complex,
    omega: float,
    *,
    c11: float,
    c13: float,
    c33: float,
    c44: float,
    c66: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    radiating: tuple[bool, bool, bool] = (False, False, False),
) -> np.ndarray:
    r"""
    Row 3 of the n=1 VTI flexural modal determinant evaluated at
    the borehole wall ``r = a``.

    Encodes the tangential-shear-vanishing BC
    ``sigma_rtheta^{(s)}(a) = 0`` (sin sector, dipole order).
    Returns the four post-rescale coefficients in column order
    ``[A | B_qP, C_qSV, D_SH]``. A column is identically zero
    because the fluid carries no shear; the other three columns
    all scale linearly with C66 (pure shear stress, no Lame
    replacement).

    Parameters
    ----------
    kz, omega : float
        Axial wavenumber and angular frequency.
    c11, c13, c33, c44, c66 : float
        VTI stiffness tensor entries (Pa). Only ``c66`` enters
        the matrix entries directly (as the outer factor on every
        non-zero column); ``c11, c13, c33, c44`` enter indirectly
        via the Christoffel roots.
    rho : float
        Formation density (kg/m^3).
    vf, rho_f : float
        Fluid velocity and density. Not used (fluid carries no
        shear); carried for signature uniformity.
    a : float
        Borehole radius (m).

    Returns
    -------
    ndarray, shape (4,) complex
        Coefficients of (A, B_qP, C_qSV, D_SH) in row 3. Real-
        valued in the bound regime.

    See Also
    --------
    _modal_determinant_n1 : Isotropic n=1 form. At isotropic-
        collapse, ``row[0] = M31 = 0``, ``row[1] = M32``,
        ``row[2] = M33``, ``row[3] = M34`` bit-exactly.
    """
    del vf, rho_f  # not used by row 3 (fluid no shear)
    alpha_qP, alpha_qSV, alpha_SH = _radial_wavenumbers_vti_complex(
        kz,
        omega,
        c11=c11,
        c13=c13,
        c33=c33,
        c44=c44,
        c66=c66,
        rho=rho,
        radiating=radiating,
    )

    K0_qP_a, K1_qP_a = _k_or_hankel(0, alpha_qP, a, leaky=radiating[0])
    K0_qSV_a, K1_qSV_a = _k_or_hankel(0, alpha_qSV, a, leaky=radiating[1])
    K0_SH_a, K1_SH_a = _k_or_hankel(0, alpha_SH, a, leaky=radiating[2])

    row: np.ndarray = np.zeros(4, dtype=complex)
    # A column: fluid carries no shear at the wall.
    row[0] = 0.0
    # B_qP column: 2 C66 (alpha_qP K_0/a + 2 K_1/a^2). At isotropic
    # alpha_qP -> p, C66 -> mu, gives M32.
    row[1] = +2.0 * c66 * (alpha_qP * K0_qP_a / a + 2.0 * K1_qP_a / (a * a))
    # C_qSV column post-rescale: same shape as B_qP with alpha_qP ->
    # alpha_qSV, times the k_z column normalisation -- roadmap A.8.
    # Matches M33 at alpha_qSV -> s, C66 -> mu.
    row[2] = +2.0 * kz * c66 * (alpha_qSV * K0_qSV_a / a + 2.0 * K1_qSV_a / (a * a))
    # D_SH column: -C66 (alpha_SH^2 K_1 + 2 alpha_SH K_0/a +
    # 4 K_1/a^2). The alpha_SH^2 K_1 direct term is unique to
    # row 3 -- comes from d_r [alpha_SH K_0(alpha_SH r)] =
    # -alpha_SH^2 K_1. Matches M34 at alpha_SH -> s.
    row[3] = -c66 * (
        alpha_SH * alpha_SH * K1_SH_a
        + 2.0 * alpha_SH * K0_SH_a / a
        + 4.0 * K1_SH_a / (a * a)
    )
    return row


# =====================================================================
# Substep H.d.4 -- row 4 of the n=1 VTI flexural determinant (r=a)
# =====================================================================
#
# BC4: ``sigma_rz^{(s)}(a) = 0`` (cos sector, dipole order).
# Z-derivative-bearing row; gets the FULL substep-H.a.6 rescale
# (row * i + col-by-(-i) on C_qSV).
#
# **VTI sigma_rz formula**:
#
#       sigma_rz = 2 C44 epsilon_rz = C44 (d_z u_r + d_r u_z)
#
# Pure C44 shear -- no Lame replacement (C44 is the constitutive
# shear modulus). Same form as H.c.1.c's n=0 row 3, plus the new
# D_SH column at n=1.
#
# **P_qX combination reused from H.c.1.c**:
#
#       P_qX = C11 alpha_qX^2 + C13 k_z^2 + rho omega^2
#
# Reduces to ``2 (lambda + mu) k_z^2`` for qP at isotropic limit;
# to ``(lambda + mu)(2 k_z^2 - k_S^2)`` for qSV.
#
# The P_qX combination naturally appears via the algebraic identity
# ``i k_z + z_qX_polarization = i P_qX / [(C13+C44) k_z]`` (qP) or
# ``k_z^2 - z_qSV alpha_qSV = P_qSV / (C13+C44)`` (qSV) -- the
# (i k_z + z_X) sum collapses to a clean P_qX expression at the
# wall.
#
# **D_SH at row 4**: D enters via ``d_z u_r`` from
# ``(1/r) d_theta psi_z`` in u_r. ``d_r u_z`` from D is zero
# (curl_z drops psi_z, so u_z has no D contribution). The
# d_z u_r contribution gives ``+i k_z C44 D K_1(alpha_SH a)/a``
# pre-rescale; row * i flips to ``-k_z C44 K_1/a`` post-rescale,
# matching M44.
#
# **Row 4 entries (post-rescale)**:
#
#       row[0] = 0                                          # A (no shear)
#       row[1] = +C44 P_qP / [(C13 + C44) k_z]
#                * (alpha_qP K_0(alpha_qP a)
#                   + K_1(alpha_qP a) / a)                  # B_qP -> M42
#       row[2] = +C44 P_qSV / (C13 + C44)
#                * (alpha_qSV K_0(alpha_qSV a)
#                   + K_1(alpha_qSV a) / a)                  # C_qSV -> M43
#       row[3] = -k_z C44 K_1(alpha_SH a) / a               # D_SH -> M44
#
# Note: the B_qP entry has the TWO-TERM ``alpha_qP K_0 + K_1/a``
# combination at n=1 (vs the single ``K_1`` term at n=0 in
# H.c.1.c row 3). Same Bessel-derivative pattern as row 1 (H.d.1)
# reflecting the K_1-indexed qP scalar potential at n=1.


def _modal_row4_at_a_n1_vti(
    kz: complex,
    omega: float,
    *,
    c11: float,
    c13: float,
    c33: float,
    c44: float,
    c66: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    radiating: tuple[bool, bool, bool] = (False, False, False),
) -> np.ndarray:
    r"""
    Row 4 of the n=1 VTI flexural modal determinant evaluated at
    the borehole wall ``r = a``.

    Encodes the axial-shear-vanishing BC ``sigma_rz^{(s)}(a) = 0``
    (cos sector, dipole order). Z-derivative-bearing; gets the
    FULL substep-H.a.6 rescale (``row * i`` + ``col-by-(-i)`` on
    C_qSV). Returns the four post-rescale coefficients in column
    order ``[A | B_qP, C_qSV, D_SH]``.

    Parameters
    ----------
    kz, omega : float
        Axial wavenumber and angular frequency.
    c11, c13, c33, c44, c66 : float
        VTI stiffness tensor entries (Pa). All used:
        ``c11, c13, c44`` set P_qX combinations on B_qP and C_qSV
        columns; ``c44`` is the outer constitutive shear modulus
        (the only C-matrix term in the D_SH column);
        ``c66`` doesn't appear directly in the entries but
        affects ``alpha_SH`` via the SH Christoffel branch.
    rho : float
        Formation density (kg/m^3).
    vf, rho_f : float
        Fluid velocity / density. Not used (fluid carries no
        shear); carried for signature uniformity.
    a : float
        Borehole radius (m).

    Returns
    -------
    ndarray, shape (4,) complex
        Coefficients of (A, B_qP, C_qSV, D_SH) in row 4. Real-
        valued in the bound regime.

    See Also
    --------
    _modal_determinant_n1 : Isotropic n=1 form. At isotropic-
        collapse, ``row[0] = M41 = 0``, ``row[1] = M42``,
        ``row[2] = M43``, ``row[3] = M44`` bit-exactly.
    _modal_row3_at_a_vti : The n=0 sigma_rz counterpart. The n=1
        form adds D_SH (cross-coupled via ``d_z u_r`` from
        ``(1/r) d_theta psi_z``) and uses the two-term
        ``alpha_qP K_0 + K_1/a`` combination on B_qP (vs single
        K_1 at n=0).
    """
    del vf, rho_f  # not used by row 4 (fluid no shear)
    alpha_qP, alpha_qSV, alpha_SH = _radial_wavenumbers_vti_complex(
        kz,
        omega,
        c11=c11,
        c13=c13,
        c33=c33,
        c44=c44,
        c66=c66,
        rho=rho,
        radiating=radiating,
    )

    K0_qP_a, K1_qP_a = _k_or_hankel(0, alpha_qP, a, leaky=radiating[0])
    K0_qSV_a, K1_qSV_a = _k_or_hankel(0, alpha_qSV, a, leaky=radiating[1])
    _, K1_SH_a = _k_or_hankel(0, alpha_SH, a, leaky=radiating[2])

    rho_omega_sq = rho * omega * omega
    p_qP = c11 * alpha_qP * alpha_qP + c13 * kz * kz + rho_omega_sq
    p_qSV = c11 * alpha_qSV * alpha_qSV + c13 * kz * kz + rho_omega_sq

    row: np.ndarray = np.zeros(4, dtype=complex)
    # A column: fluid carries no shear at the wall.
    row[0] = 0.0
    # B_qP column: C44 P_qP / [(C13+C44) kz] times the two-term
    # alpha_qP K_0 + K_1/a combination. At isotropic alpha_qP -> p,
    # P_qP -> 2(lambda+mu) kz^2, gives 2 mu kz (p K_0 + K_1/a) = M42.
    row[1] = +c44 * p_qP / ((c13 + c44) * kz) * (alpha_qP * K0_qP_a + K1_qP_a / a)
    # C_qSV column post-rescale: C44 P_qSV / (C13 + C44) times the
    # same two-term alpha_qSV K_0 + K_1/a combination -- roadmap A.8;
    # sigma_rz is C44 (d_z u_r + d_r u_z) and both terms carry
    # d_r Phi, not Phi. At isotropic P_qSV / (C13+C44) ->
    # (2 kz^2 - kS^2), gives mu (2 kz^2 - kS^2)(s K_0 + K_1/a) = M43.
    row[2] = +c44 * p_qSV / (c13 + c44) * (alpha_qSV * K0_qSV_a + K1_qSV_a / a)
    # D_SH column post-rescale: -kz C44 K_1(alpha_SH a)/a.
    # Matches M44 at alpha_SH -> s, C44 -> mu.
    row[3] = -kz * c44 * K1_SH_a / a
    return row


# =====================================================================
# Substep H.d.5 -- assembly into _modal_determinant_n1_vti
# =====================================================================
#
# Stacks the four row builders (H.d.1 / H.d.2 / H.d.3 / H.d.4) into
# the 4x4 VTI flexural modal matrix and returns the determinant
# as a real scalar.
#
# Each row builder applies the substep-H.a.6 phase rescale
# internally: row 4 (the only z-derivative-bearing row in the n=1
# 4x4) by ``i``; column C_qSV by ``-i``. The assembled matrix is
# real-valued in the bound regime; ``np.linalg.det`` on the
# .real part returns the real determinant directly.
#
# Reduces to :func:`_modal_determinant_n1` (sharing the same
# flexural root in k_z, with an irrelevant overall scale factor)
# when the stiffness tensor is isotropic. The H.d.6 follow-up
# (``flexural_dispersion_vti`` end-to-end brentq) is the
# floating-point oracle that anchors this assembly.


def _modal_determinant_n1_vti(
    kz: complex,
    omega: float,
    *,
    c11: float,
    c13: float,
    c33: float,
    c44: float,
    c66: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
) -> float:
    r"""
    4x4 dipole modal determinant for the n=1 (flexural) VTI
    dispersion.

    Stacks the four row builders from H.d.1 / H.d.2 / H.d.3 /
    H.d.4 into the 4x4 VTI flexural modal matrix, applies the
    substep-H.a.6 phase rescale (already absorbed into each
    row), and returns the determinant as a real scalar.

    Reduces to :func:`_modal_determinant_n1` (sharing the same
    flexural root in ``k_z``, with an irrelevant overall scale
    factor) when the stiffness tensor is isotropic. The H.d.6
    integration test in :func:`flexural_dispersion_vti` is the
    floating-point oracle that anchors this assembly.

    Parameters
    ----------
    kz : float
        Trial axial wavenumber (rad / m). Must lie in the bound
        regime ``kz > omega / min(V_Sv, V_Sh, V_f)``.
    omega : float
        Angular frequency (rad / s).
    c11, c13, c33, c44, c66 : float
        VTI stiffness tensor entries (Pa).
    rho : float
        Formation density (kg/m^3).
    vf, rho_f : float
        Borehole-fluid velocity and density.
    a : float
        Borehole radius (m).

    Returns
    -------
    float
        ``det(M)`` of the 4x4 VTI flexural modal matrix, real-
        valued in the bound regime. NaN outside the bound regime.
    """
    kz = _as_real_kz(kz, caller="_modal_determinant_n1_vti")
    if kz * kz - (omega / vf) ** 2 < 0.0:
        # Fast-formation regime: the fluid Bessels are oscillatory and
        # the rows are genuinely complex, so ``M.real`` would be a
        # finite but meaningless number rather than the mode function.
        # The documented contract is NaN outside the bound regime;
        # :func:`_modal_determinant_n1_vti_complex` is the entry point
        # that handles this regime properly.
        return float("nan")
    M = _modal_matrix_n1_vti(
        kz,
        omega,
        c11=c11,
        c13=c13,
        c33=c33,
        c44=c44,
        c66=c66,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    # Each row is real-valued post-rescale; imaginary parts are
    # zero to floating-point precision in the bound regime.
    return float(np.linalg.det(M.real))


def _recombine_conjugate_columns(
    M: np.ndarray,
    kz: complex,
    omega: float,
    *,
    c11: float,
    c13: float,
    c33: float,
    c44: float,
    c66: float,
    rho: float,
    radiating: tuple[bool, bool, bool] = (False, False, False),
) -> np.ndarray:
    r"""
    Put the qP / qSV columns on a real basis when their radial
    wavenumbers are a conjugate pair.

    The two builders share one functional form ``f`` and differ only
    by a real factor: ``col_qP = f(alpha_qP)`` and
    ``col_qSV = k_z f(alpha_qSV)`` (verified to the digit -- the
    residual ``||col_qSV - k_z conj(col_qP)||`` vanishes where the
    roots are conjugate). The pair is replaced by

    ``(f(alpha_qP) + f(alpha_qSV)) / 2``  and
    ``(f(alpha_qP) - f(alpha_qSV)) / (alpha_qP - alpha_qSV)``

    which span the same space, so every root survives, and which are
    **real in both regimes**: real roots give real ``f``, and a
    conjugate pair gives ``Re f`` and ``Im f / Im(alpha)``.

    Two failures this avoids, both silent. Taking ``det(M.real)``
    across genuinely complex columns drops the imaginary halves of two
    independent solutions and returns a determinant whose sign changes
    are spurious -- on Mesaverde shale(5) it moves the 3 kHz root from
    2070.94 to 1500.33 m/s and scatters the band between 750 and 1470.
    And a plain conjugate split, correct on its own side, leaves the
    determinant vanishing at the branch point itself, where the two
    columns merge: the root finder then locks onto that degeneracy and
    reports the cutoff velocity (1760.58 m/s) at every frequency whose
    true root lies above it. The divided difference removes both,
    because it tends to ``df/dalpha`` at the merge instead of to
    zero -- the secular solution a repeated root actually calls for.

    Parameters
    ----------
    M : ndarray
        ``(4, 4)`` assembled modal matrix; columns 1 and 2 are qP and
        qSV.
    kz, omega, c11, c13, c33, c44, c66, rho
        As for the row builders, used to recover the root pair.

    Returns
    -------
    ndarray
        ``M`` unchanged where the roots are real, or with columns 1
        and 2 replaced by a real-spanning pair where they are
        conjugate.
    """
    alpha_qP, alpha_qSV, _ = _radial_wavenumbers_vti_complex(
        kz,
        omega,
        c11=c11,
        c13=c13,
        c33=c33,
        c44=c44,
        c66=c66,
        rho=rho,
        radiating=radiating,
    )
    split = alpha_qP - alpha_qSV
    if split == 0.0 or kz == 0.0:
        return M
    out = M.copy()
    g_qP = M[:, 1]
    g_qSV = M[:, 2] / kz
    out[:, 1] = 0.5 * (g_qP + g_qSV)
    out[:, 2] = (g_qP - g_qSV) / split
    return out


def _modal_matrix_n1_vti(
    kz: complex,
    omega: float,
    *,
    c11: float,
    c13: float,
    c33: float,
    c44: float,
    c66: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    radiating: tuple[bool, bool, bool] = (False, False, False),
) -> np.ndarray:
    """
    Assemble the 4x4 n=1 VTI flexural modal matrix.

    Shared by :func:`_modal_determinant_n1_vti` (bound regime, takes
    the real part) and :func:`_modal_determinant_n1_vti_complex`
    (either regime, keeps the full complex determinant), so the two
    cannot drift apart.

    Returns
    -------
    ndarray
        ``(4, 4)`` complex array.
    """
    if complex(kz).imag == 0.0:
        # A complex with zero imaginary part is a real number, and #134
        # guarantees it takes the real path bit-for-bit. Leaving it
        # complex would route the rows through complex arithmetic and
        # shift the determinant by an ulp or two.
        kz = float(complex(kz).real)
    elif not any(radiating):
        raise NotImplementedError(
            "_modal_matrix_n1_vti: a genuinely complex k_z needs "
            "radiating=... naming which of (qP, qSV, SH) leave on the "
            "outgoing branch. With every wave bound the columns all "
            "decay while the field grows along z, which is not a mode."
        )
    kw = dict(
        c11=c11,
        c13=c13,
        c33=c33,
        c44=c44,
        c66=c66,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
    )
    M = np.vstack(
        [
            _modal_row1_at_a_n1_vti(kz, omega, **kw, radiating=radiating),
            _modal_row2_at_a_n1_vti(kz, omega, **kw, radiating=radiating),
            _modal_row3_at_a_n1_vti(kz, omega, **kw, radiating=radiating),
            _modal_row4_at_a_n1_vti(kz, omega, **kw, radiating=radiating),
        ]
    )
    return _recombine_conjugate_columns(
        M,
        kz,
        omega,
        radiating=radiating,
        c11=c11,
        c13=c13,
        c33=c33,
        c44=c44,
        c66=c66,
        rho=rho,
    )


def _modal_determinant_n1_vti_complex(
    kz: complex,
    omega: float,
    *,
    radiating: tuple[bool, bool, bool] = (False, False, False),
    c11: float,
    c13: float,
    c33: float,
    c44: float,
    c66: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
) -> complex:
    r"""
    Complex 4x4 n=1 VTI flexural modal determinant, valid in both
    fluid regimes.

    Sister of :func:`_modal_determinant_n1_complex` along the
    anisotropy axis, and the fast-formation counterpart of
    :func:`_modal_determinant_n1_vti`.

    The distinction that matters is **which wavenumbers go complex**.
    For a bound flexural mode the phase velocity stays below ``V_Sv``,
    so the formation qP, qSV and SH radial wavenumbers are real in
    both regimes and the formation fields decay in ``r``. Only the
    fluid wavenumber ``F_f = sqrt(k_z^2 - (omega/V_f)^2)`` turns
    imaginary, and only where the phase velocity exceeds ``V_f`` --
    which in a fast formation is the low-frequency part of the band.
    So this is not a leaky-mode solver: ``k_z`` stays real, the root
    is a genuine bound mode, and the determinant is complex only
    because the fluid pressure oscillates rather than decays.

    Evaluated at real ``k_z`` the result is a real quantity times a
    fixed phase, exactly as in the isotropic case, so one of
    ``Re`` and ``Im`` is the mode function and the other is
    round-off. Which one is **measured** by
    :func:`_real_root_function` rather than assumed -- see roadmap
    A.7, where assuming it cost the n=2 path its correctness.

    Parameters
    ----------
    kz : complex
        Trial axial wavenumber (rad / m). Real-valued: a ``float``, or
        a ``complex`` whose imaginary part is zero. See
        :func:`_as_real_kz` for why a genuinely complex one is refused
        rather than continued onto some branch.
    omega : float
        Angular frequency (rad / s).
    c11, c13, c33, c44, c66 : float
        VTI stiffness tensor entries (Pa).
    rho : float
        Formation density (kg / m^3).
    vf, rho_f : float
        Borehole-fluid velocity (m / s) and density (kg / m^3).
    a : float
        Borehole radius (m).

    Returns
    -------
    complex
        ``det(M)``. In the bound regime the imaginary part is zero to
        floating-point precision and the real part equals
        :func:`_modal_determinant_n1_vti`.
    """
    M = _modal_matrix_n1_vti(
        kz,
        omega,
        c11=c11,
        c13=c13,
        c33=c33,
        c44=c44,
        c66=c66,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        radiating=radiating,
    )
    return complex(np.linalg.det(M))
