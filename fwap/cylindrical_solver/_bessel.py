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


def _radial_wavenumber(alpha_squared: complex, *, leaky: bool) -> complex:
    r"""
    Select the physically admissible square root of ``alpha^2``.

    ``alpha^2 = k_z^2 - (omega / V)^2`` has two square roots, and which
    one is admissible depends on the branch the caller will evaluate:

    * **Bound** (``leaky=False``, field ``~ K_n(alpha r) ~ e^{-alpha r}``):
      the field must *decay* outward, so ``Re(alpha) > 0``.
    * **Leaky** (``leaky=True``, field ``~ K_n(alpha r e^{i pi}) ~
      e^{+alpha r}`` -- see :func:`_k_or_hankel`): the field must
      *radiate* outward, so with the ``e^{-i omega t}`` time convention
      the unwrapped phase must grow with ``r``, i.e. ``Im(alpha) > 0``.

    These are different conditions, and ``numpy.sqrt`` implements only
    the first: its principal branch returns ``Re(alpha) >= 0``, with
    ``sign(Im(alpha)) = sign(Im(alpha^2))``. Since
    ``Im(alpha^2) = 2 Re(k_z) Im(k_z)``, the principal root is outgoing
    exactly when ``Im(k_z) >= 0`` and *incoming* below the real ``k_z``
    axis -- which is where a complex root search seeded on that axis
    spends part of its time.

    Parameters
    ----------
    alpha_squared : complex
        ``k_z^2 - (omega / V)^2`` in rad^2 / m^2.
    leaky : bool
        Whether the caller will evaluate this wavenumber on the leaky
        (outgoing Hankel) branch. Use the flag from
        :func:`_detect_leaky_branches` so the selection rule and the
        Bessel evaluation agree inside its marginal tolerance band.

    Returns
    -------
    complex
        The admissible root, in rad / m.

    Notes
    -----
    Applying this makes the complex determinant a *single* analytic
    function of ``k_z`` across the real axis. Without it the determinant
    jumps there: by an overall ``-1`` when only the oscillatory fluid
    branch flips (harmless -- an overall factor does not move roots),
    and by a ``k_z``-dependent factor when a formation branch flips,
    which is a different function with different roots -- roadmap A.10.

    Layer blocks need no such rule. They retain both ``I_n`` and
    ``K_n``, and flipping the sign of a layer wavenumber maps that pair
    onto a linear combination of itself; the change of basis cancels in
    the propagator ``E(r_out) E(r_in)^{-1}``.
    """
    alpha = np.sqrt(complex(alpha_squared))
    if leaky:
        if alpha.imag < 0.0:
            alpha = -alpha
    elif alpha.real < 0.0:
        alpha = -alpha
    return complex(alpha)


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
    with ``(pi / 2) * i^{n+1} * H_n^{(2)}(i alpha r)``. This is
    ``K_n`` continued one sheet counter-clockwise -- expanding the
    Hankel function gives, for integer ``n``,

        (pi/2) i^{n+1} H_n^{(2)}(i z) = (-1)^{n+1} K_n(z e^{i pi})
                                      = -K_n(z) + i pi (-1)^n I_n(z)

    (verified to 1e-17) -- so it is a single solution of the modified
    Bessel equation, and ``K_n(z e^{i pi}) ~ e^{+z}`` rather than
    ``e^{-z}``. At a genuinely leaky ``alpha`` that is the outgoing,
    radially growing field the radiation condition asks for: the
    unwrapped phase runs at ``+Im(alpha)`` rad/m, against ``-Im(alpha)``
    for plain ``K_n(alpha r)``.

    Returns the same ``(K_n, K_{n+1})`` tuple shape regardless of
    branch, so the matrix-building code is identical in both
    regimes.

    Notes
    -----
    The two returned orders are consecutive orders of *one* solution,
    which is what callers rely on: they form radial derivatives with
    ``d/dx K_n(x) = -K_{n+1}(x) + (n/x) K_n(x)``, and that identity
    holds for this pair to ~1e-10 at bound, radiating and complex
    ``alpha`` alike (see
    ``test_k_or_hankel_leaky_pair_are_consecutive_orders_of_one_solution``).
    Anything that changes the two slots by different factors -- for
    instance "fixing" the branch by negating ``alpha``, which leaves the
    caller's ``alpha`` inconsistent with the values it gets back --
    breaks that identity and corrupts every column built from the pair.

    This docstring used to claim that the bound limit of the leaky
    formula matches ``K_n``. It does not: at real positive ``alpha``
    the two differ by factors of 2 to 3e3, because ``K_n(z e^{i pi})``
    is the other sheet. The property that matters is the order
    consistency above, and it was untested until the claim was checked.
    """
    z = alpha * r
    if leaky:
        # Evaluated through the right-hand side of
        #
        #     (pi/2) i^{n+1} H_n^{(2)}(i z) = -K_n(z) + i pi (-1)^n I_n(z)
        #
        # rather than through ``hankel2`` directly. The two agree to
        # 1e-16 while ``arg(i z) < pi``, which is every argument the
        # solvers used to reach; but ``hankel2``'s branch cut sits at
        # ``arg(i z) = pi``, and the outgoing root selected by
        # :func:`_radial_wavenumber` crosses it below the real ``k_z``
        # axis, where ``hankel2`` then returns the wrong sheet. ``kv``
        # and ``iv`` have their cut on the negative real axis instead,
        # which ``z`` does not reach in the leaky regime, so this form
        # continues smoothly across -- roadmap A.10.
        sign = 1.0 if n % 2 == 0 else -1.0
        val_n = -special.kv(n, z) + 1j * np.pi * sign * special.iv(n, z)
        val_np1 = -special.kv(n + 1, z) - 1j * np.pi * sign * special.iv(n + 1, z)
        return complex(val_n), complex(val_np1)
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


def _rigid_tool_fluid_factors(
    F: complex,
    a: float,
    r_tool: float,
) -> tuple[complex, complex]:
    r"""
    Fluid pressure and radial-displacement factors at ``r = a`` for a
    borehole containing a rigid centralised logging tool.

    Without a tool the fluid column reaches the axis and regularity
    there leaves one solution, ``P = A I_0(F r)``, so the modal
    determinant's fluid column is built from ``I_0(F a)`` and
    ``I_1(F a)``. A tool makes the fluid an **annulus**
    ``r_tool < r < a``; the axis is no longer in the domain, ``K_0``
    is admissible, and the general solution is

    .. math::

        P = A \left[ I_0(F r) + \beta K_0(F r) \right].

    A rigid tool is immovable, so ``u_r(r_tool) = 0``. With
    ``u_r \propto \mathrm{d}P/\mathrm{d}r
    = A F [ I_1(F r) - \beta K_1(F r) ]`` that fixes

    .. math::

        \beta = I_1(F\,r_\mathrm{tool}) / K_1(F\,r_\mathrm{tool}),

    and the two factors this helper returns are ``P(a)/A`` and
    ``(u_r(a) \rho_f \omega^2)/(A F)``. Substituting them for
    ``I_0(F a)`` and ``I_1(F a)`` is the **entire** change a rigid
    tool makes to the n=0 modal determinant -- every solid row is
    untouched, because the tool changes the fluid's radial basis and
    nothing else.

    This is the White & Zechman (1968) centralised-tool model, the
    one Paillet & Cheng (1986) use: their table 1 gives a tool
    *radius* and no tool elastic properties, and their pressure is
    "measured at the surface of the logging tool (r = R)".

    **The no-tool limit is exact, not approximate.** As
    ``F r_tool -> 0``, ``I_1 -> F r_tool / 2`` and ``K_1 -> 1 / (F
    r_tool)``, so ``beta -> (F r_tool)^2 / 2 -> 0``. Rather than rely
    on that numerically, ``r_tool <= 0`` short-circuits to the plain
    ``I_0``/``I_1`` pair, which keeps every existing open-hole result
    bit-identical.

    Parameters
    ----------
    F : complex
        Fluid radial wavenumber (1 / m). Real in the bound regime,
        imaginary where the fluid field oscillates.
    a : float
        Borehole (fluid-formation contact) radius, m.
    r_tool : float
        Rigid tool radius, m. ``<= 0`` means no tool. Must be less
        than ``a``.

    Returns
    -------
    tuple of complex
        ``(Z0, Z1)``, replacing ``I_0(F a)`` and ``I_1(F a)``.

    Raises
    ------
    ValueError
        If ``r_tool >= a``.
    """
    if r_tool <= 0.0:
        return complex(special.iv(0, F * a)), complex(special.iv(1, F * a))
    if r_tool >= a:
        raise ValueError(
            f"tool radius must be smaller than the borehole radius; "
            f"got r_tool={r_tool}, a={a}"
        )
    Fr = F * r_tool
    beta = complex(special.iv(1, Fr)) / complex(special.kv(1, Fr))
    Fa = F * a
    z0 = complex(special.iv(0, Fa)) + beta * complex(special.kv(0, Fa))
    z1 = complex(special.iv(1, Fa)) - beta * complex(special.kv(1, Fa))
    return z0, z1
