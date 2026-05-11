"""
Bessel-function and radial-wavenumber helpers for the cylindrical-
borehole modal-determinant solver.

Phase 1 of the cylindrical_solver refactoring plan extracted these
real-/complex-argument modified-Bessel + Hankel wrappers
(``_i0_i1``, ``_k0_k1``, ``_k_or_hankel``) and the per-geometry
radial-wavenumber + Bessel-pack helpers
(``_layered_n0_radial_wavenumbers``, ``_layered_n0_bessel_pack``,
``_radial_wavenumbers_vti``) from the 14 kLoC monolith into this
submodule. The names remain re-exported from
``fwap.cylindrical_solver`` so neither the public API nor the
private symbols imported by ``tests/test_cylindrical_solver.py``
move.
"""

from __future__ import annotations

import numpy as np
from scipy import special

from fwap.cylindrical_solver._dataclasses import BoreholeLayer

# ---------------------------------------------------------------------
# Bessel-function helpers (real arguments, positive)
# ---------------------------------------------------------------------
#
# For bound modes (k_z > omega / V_alpha for every wave speed
# V_alpha = V_f, V_P, V_S), the radial wavenumbers F, p, s are real
# and positive, so we use the standard real-argument modified Bessel
# routines from scipy.special.


def _i0_i1(x: float) -> tuple[float, float]:
    """Return (I_0(x), I_1(x)). I_0'(x) = I_1(x)."""
    return float(special.iv(0, x)), float(special.iv(1, x))


def _k0_k1(x: float) -> tuple[float, float]:
    """Return (K_0(x), K_1(x)). K_0'(x) = -K_1(x);
    K_1'(x) = -K_0(x) - K_1(x)/x."""
    return float(special.kv(0, x)), float(special.kv(1, x))


def _k_or_hankel(
    n: int, alpha: complex, r: float, *, leaky: bool
) -> tuple[complex, complex]:
    """
    Return ``(K_n(alpha r), K_{n+1}(alpha r))`` -- bound branch -- or
    the leaky-equivalent Hankel-via-analytic-continuation values.

    Bound branch (``leaky=False``): the standard modified Bessel
    function K of the second kind, evaluated at the (possibly
    complex) argument ``alpha r``.

    Leaky branch (``leaky=True``): for outgoing-radiation BCs with
    ``e^{-i omega t}`` time convention, replace ``K_n(alpha r)``
    with ``(pi / 2) * i^{n+1} * H_n^{(2)}(i alpha r)``. The
    ``i^{n+1}`` constant phase factor is absorbed into the unknown
    amplitudes of the modal determinant, but we keep it here so
    that the BOUND limit (alpha real and positive) of the Hankel
    formula matches the corresponding K_n value -- a structural
    consistency check that the regression test exercises.

    Returns the same ``(K_n, K_{n+1})`` tuple shape regardless of
    branch, so the matrix-building code is identical in both
    regimes.
    """
    z = alpha * r
    if leaky:
        # K_n(z) = (pi/2) i^{n+1} H_n^{(2)}(i z) by analytic
        # continuation. Use ``ix = 1j * z`` as the Hankel argument.
        ix = 1j * z
        h_n = special.hankel2(n, ix)
        h_np1 = special.hankel2(n + 1, ix)
        phase_n = (np.pi / 2.0) * (1j ** (n + 1))
        phase_np1 = (np.pi / 2.0) * (1j ** (n + 2))
        return complex(phase_n * h_n), complex(phase_np1 * h_np1)
    return complex(special.kv(n, z)), complex(special.kv(n + 1, z))


# =====================================================================
# Substep F.1.b.1 -- radial-wavenumber + Bessel-pack helpers
# =====================================================================


def _layered_n0_radial_wavenumbers(
    kz: float,
    omega: float,
    *,
    vp: float,
    vs: float,
    vf: float,
    layer: BoreholeLayer,
) -> tuple[float, float, float, float, float]:
    r"""
    Bound-regime radial wavenumbers for the n=0 layered problem.

    Computes the five radial wavenumbers in the field ansatz pinned
    by substep F.1.a.1:

        F_f = sqrt(k_z^2 - (omega / V_f)^2)         (fluid)
        p_m = sqrt(k_z^2 - (omega / V_P_m)^2)       (annulus P)
        s_m = sqrt(k_z^2 - (omega / V_S_m)^2)       (annulus S)
        p   = sqrt(k_z^2 - (omega / V_P)^2)         (formation P)
        s   = sqrt(k_z^2 - (omega / V_S)^2)         (formation S)

    No regime gating is applied here. If any radial-wavenumber
    argument is negative (``k_z`` below the bound-regime floor
    ``omega / min(V_S, V_S_m, V_f)``), the corresponding output is
    ``NaN`` via :func:`numpy.sqrt` of a negative real input. The
    public dispatch (:func:`stoneley_dispersion_layered`) is
    responsible for choosing brackets that keep brentq inside the
    bound regime; this helper is brentq-safe by passing through
    out-of-regime inputs as NaN rather than raising.

    Parameters
    ----------
    kz : float
        Trial axial wavenumber (rad / m).
    omega : float
        Angular frequency (rad / s).
    vp, vs, vf : float
        Formation P-wave, formation S-wave, and borehole-fluid
        velocities (m / s).
    layer : BoreholeLayer
        The annular layer between fluid and formation. Only
        ``layer.vp`` and ``layer.vs`` are used here.

    Returns
    -------
    tuple of five floats
        ``(F_f, p_m, s_m, p, s)`` in the order pinned by
        substep F.1.a.1.

    See Also
    --------
    _layered_n0_bessel_pack : The Bessel-evaluation downstream of
        this helper.
    """
    F_f = float(np.sqrt(kz * kz - (omega / vf) ** 2))
    p_m = float(np.sqrt(kz * kz - (omega / layer.vp) ** 2))
    s_m = float(np.sqrt(kz * kz - (omega / layer.vs) ** 2))
    p = float(np.sqrt(kz * kz - (omega / vp) ** 2))
    s = float(np.sqrt(kz * kz - (omega / vs) ** 2))
    return F_f, p_m, s_m, p, s


def _layered_n0_bessel_pack(
    F_f: float,
    p_m: float,
    s_m: float,
    p: float,
    s: float,
    a: float,
    b: float,
) -> dict[str, float]:
    """
    Evaluate the 22 Bessel values needed by the n=0 layered modal
    determinant.

    The dict has 22 entries with the naming pattern
    ``"<X><n>_<wavenumber>_<radius>"``:

        ``X``         in ``{"I", "K"}``      (modified-Bessel kind)
        ``n``         in ``{"0", "1"}``      (order)
        ``wavenumber`` in ``{"Ff", "pm", "sm", "p", "s"}``
                       (matches substep F.1.a.1)
        ``radius``    in ``{"a", "b"}``       (interface radius)

    Coverage:

    * Fluid at ``r = a``: ``I_0(F_f a)``, ``I_1(F_f a)`` -- the
      fluid is regular at the borehole axis ``r = 0``, no K-flavour
      branch. Two values.
    * Annulus P at ``r = a`` and ``r = b``: full I + K pairs at
      both interfaces (eight values).
    * Annulus S at ``r = a`` and ``r = b``: full I + K pairs at
      both interfaces (eight values).
    * Formation P, S at ``r = b``: K-flavour only (decaying at
      infinity), four values.

    Total: 2 + 8 + 8 + 4 = 22.

    Parameters
    ----------
    F_f, p_m, s_m, p, s : float
        Output of :func:`_layered_n0_radial_wavenumbers`.
    a, b : float
        Interface radii (m). ``a`` is the fluid-annulus interface
        (= borehole wall); ``b = a + layer.thickness`` is the
        annulus-formation interface.

    Returns
    -------
    dict of {str: float}
        See key naming above.

    Notes
    -----
    Out-of-regime inputs (NaN from :func:`_layered_n0_radial_wavenumbers`)
    propagate to NaN values in the output dict. Callers that need
    finiteness must check explicitly.
    """
    pack: dict[str, float] = {}

    # Fluid at r = a (no K-flavour because the regular-at-axis
    # solution is I_n(F_f r); K_n(F_f r) is singular at r = 0).
    pack["I0_Ff_a"] = float(special.iv(0, F_f * a))
    pack["I1_Ff_a"] = float(special.iv(1, F_f * a))

    # Annulus P-wave at r = a and r = b (full I + K pairs).
    for radius_label, radius in (("a", a), ("b", b)):
        x = p_m * radius
        pack[f"I0_pm_{radius_label}"] = float(special.iv(0, x))
        pack[f"I1_pm_{radius_label}"] = float(special.iv(1, x))
        pack[f"K0_pm_{radius_label}"] = float(special.kv(0, x))
        pack[f"K1_pm_{radius_label}"] = float(special.kv(1, x))

    # Annulus S-wave at r = a and r = b (full I + K pairs).
    for radius_label, radius in (("a", a), ("b", b)):
        x = s_m * radius
        pack[f"I0_sm_{radius_label}"] = float(special.iv(0, x))
        pack[f"I1_sm_{radius_label}"] = float(special.iv(1, x))
        pack[f"K0_sm_{radius_label}"] = float(special.kv(0, x))
        pack[f"K1_sm_{radius_label}"] = float(special.kv(1, x))

    # Formation P and S at r = b (K-flavour only; decaying at
    # infinity rules out the I-flavour branch).
    pack["K0_p_b"] = float(special.kv(0, p * b))
    pack["K1_p_b"] = float(special.kv(1, p * b))
    pack["K0_s_b"] = float(special.kv(0, s * b))
    pack["K1_s_b"] = float(special.kv(1, s * b))

    return pack


# =====================================================================
# Substep H.b -- radial-wavenumber helper (Christoffel-equation roots)
# =====================================================================
#
# Computes the three radial decay rates ``(alpha_qP, alpha_qSV,
# alpha_SH)`` for a VTI half-space at a given ``(kz, omega)``. The
# qP / qSV pair comes from the Christoffel quadratic of substep
# H.a.2; the SH branch is decoupled and follows the simple form of
# substep H.a.4.
#
# Convention: ``alpha`` is the real positive decay rate appearing in
# the bound-mode field representation ``K_n(alpha r)``. The helper
# returns ``alpha`` (not ``alpha^2``) for direct substitution into
# the modal-matrix entries in H.c / H.d.
#
# The qP / qSV ordering follows substep H.a.3: ``alpha_qP`` is the
# smaller root (faster wave / smaller decay), ``alpha_qSV`` is the
# larger root. This convention agrees with the isotropic limit
# ``alpha_qP -> p < s -> alpha_qSV`` by inspection.
#
# Out-of-regime behaviour: NaN is returned for any of the three
# decay rates whose square would be negative. Matches the F.1 /
# F.2 helpers' brentq-safe convention.


def _radial_wavenumbers_vti(
    kz: float,
    omega: float,
    *,
    c11: float,
    c13: float,
    c33: float,
    c44: float,
    c66: float,
    rho: float,
) -> tuple[float, float, float]:
    r"""
    Bound-regime radial decay rates ``(alpha_qP, alpha_qSV,
    alpha_SH)`` for the VTI Christoffel equation at a given
    ``(kz, omega)``.

    Solves the H.a.2 quadratic
    ``A_eff alpha^4 + B_eff alpha^2 + C_eff = 0`` for the qP / qSV
    pair (with ``alpha_qP`` the smaller root and ``alpha_qSV`` the
    larger), and the H.a.4 closed form
    ``alpha_SH^2 = (C44 k_z^2 - rho omega^2) / C66`` for the SH
    branch.

    Parameters
    ----------
    kz : float
        Trial axial wavenumber (rad / m).
    omega : float
        Angular frequency (rad / s).
    c11, c13, c33, c44, c66 : float
        VTI stiffness tensor entries (Pa).
    rho : float
        Formation density (kg / m^3).

    Returns
    -------
    tuple of three floats
        ``(alpha_qP, alpha_qSV, alpha_SH)``. Each is real positive
        in the bound regime ``k_z > omega / min(V_Sv, V_Sh)``;
        NaN otherwise. Sorted so ``alpha_qP <= alpha_qSV`` per
        substep H.a.3 convention.

    See Also
    --------
    _layered_n0_radial_wavenumbers : The isotropic-layered
        counterpart for plan F. The VTI helper here is the
        isotropic-collapse generalisation: with the isotropic
        C-matrix substitution, ``alpha_qP -> p`` and
        ``alpha_qSV -> s`` to floating-point precision.
    """
    # Christoffel quadratic in alpha^2 (substep H.a.2 corrected
    # form for the decay-rate convention):
    rho_omega_sq = rho * omega * omega
    a_eff = c11 * c44
    b_eff = (c11 + c44) * rho_omega_sq - (
        c11 * c33 + c44 * c44 - (c13 + c44) ** 2
    ) * kz * kz
    c_eff = (
        c44 * c33 * kz**4
        - (c44 + c33) * rho_omega_sq * kz * kz
        + rho_omega_sq * rho_omega_sq
    )
    # Quadratic in alpha^2: A_eff x^2 + B_eff x + C_eff = 0 where
    # x = alpha^2. Roots via the standard formula.
    disc = b_eff * b_eff - 4.0 * a_eff * c_eff
    if disc < 0.0:
        # Complex roots -- not physical in the bound regime.
        return (float("nan"), float("nan"), float("nan"))
    sqrt_disc = float(np.sqrt(disc))
    x_minus = (-b_eff - sqrt_disc) / (2.0 * a_eff)
    x_plus = (-b_eff + sqrt_disc) / (2.0 * a_eff)
    # Substep H.a.3 ordering convention: ``alpha_qP`` is the LARGER
    # decay rate (faster wave -> more localized at the wall;
    # isotropic limit alpha_qP -> p > s -> alpha_qSV when V_P > V_S).
    alpha_qP_sq = max(x_minus, x_plus)
    alpha_qSV_sq = min(x_minus, x_plus)

    # SH decay rate (substep H.a.4 corrected form): decoupled from
    # qP / qSV.
    alpha_SH_sq = (c44 * kz * kz - rho_omega_sq) / c66

    # NaN-on-out-of-regime per the brentq-safe convention.
    alpha_qP = float(np.sqrt(alpha_qP_sq)) if alpha_qP_sq >= 0.0 else float("nan")
    alpha_qSV = float(np.sqrt(alpha_qSV_sq)) if alpha_qSV_sq >= 0.0 else float("nan")
    alpha_SH = float(np.sqrt(alpha_SH_sq)) if alpha_SH_sq >= 0.0 else float("nan")
    return alpha_qP, alpha_qSV, alpha_SH
