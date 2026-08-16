"""n=0 layered (mudcake / altered-zone) Stoneley solver.

Extracted from ``fwap.cylindrical_solver.__init__`` as part of the
Phase 1 package split. The original module-level docstring with the
full physics derivation lives in the package ``__init__``; refer
there for the field ansatz, sign conventions, and references.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, special

from fwap._common import logger
from fwap.cylindrical_solver._bessel import _layered_n0_radial_wavenumbers
from fwap.cylindrical_solver._dataclasses import (
    BoreholeLayer,
    BoreholeMode,
    FluidAnnulus,
    _validate_borehole_layers,
    _validate_fluid_annulus,
)
from fwap.cylindrical_solver._n0_isotropic import (
    stoneley_dispersion,
)

# ---------------------------------------------------------------------
# Plan item F -- single-extra-layer (mudcake / altered zone) extension
# ---------------------------------------------------------------------
#
# Scope of the foundation landed here:
#
#   * :class:`BoreholeLayer` dataclass + :func:`_validate_borehole_layers`
#     pin the parameter shape used by the layered dispersion APIs.
#   * :func:`stoneley_dispersion_layered` is the public n=0 entry
#     point. With ``layers=()`` it is bit-equivalent to
#     :func:`stoneley_dispersion` (degenerate single-interface
#     case); with non-empty layers it raises ``NotImplementedError``
#     to flag that the 7x7 layered modal determinant is the next
#     step of plan item F.
#
# What this lets downstream work assume:
#
#   * The public-API surface (parameter names, validation rules,
#     return type) is fixed. The follow-up implementing
#     ``_modal_determinant_n0_layered`` only needs to change the
#     dispatch branch below.
#   * The layer=formation regression test in
#     ``tests/test_solver_layered.py`` is already passing with
#     ``layers=()`` and stays as the floating-point oracle for the
#     non-trivial case.

# =====================================================================
# Substep F.1.a -- math scaffolding for the n=0 layered determinant
# =====================================================================
#
# Conventions and notation used throughout substeps F.1.a.1 through
# F.1.a.6 below; the n=0 single-interface derivation in
# :func:`_modal_determinant_n0` (and the n=1 substep blocks above) is
# the structural prior. The new ingredient is one annular elastic
# layer sandwiched between the borehole fluid and the formation
# half-space:
#
#       fluid (r < a)  |  annulus (a < r < b)  |  formation (r > b)
#
# with ``b = a + h`` and ``h = layer.thickness``. The annulus has its
# own ``(V_P_m, V_S_m, rho_m)``; primes are not used (they would
# clash with derivative notation), instead the suffix ``_m`` ("mudcake")
# tags annulus quantities and the unsuffixed names continue to denote
# the formation half-space.
#
# The substep blocks below land the maths only; the matrix
# transcription (substep F.1.b) and the public-API hook (substep F.1.c)
# follow in dependent commits per ``docs/plans/cylindrical_biot_F.md``.

# =====================================================================
# Substep F.1.a.1 -- sign conventions, regime gate, field ansatz
# =====================================================================
#
# Conventions inherited verbatim from the module docstring:
#
#   * Time dependence ``e^{-i omega t}``.
#   * Axial dependence ``e^{i k_z z}``.
#   * Bound regime: every radial wavenumber is real positive.
#
# Five radial wavenumbers (one per body-wave / region pair):
#
#       F_f = sqrt(k_z^2 - (omega / V_f)^2)         > 0    (fluid)
#       p_m = sqrt(k_z^2 - (omega / V_P_m)^2)       > 0    (annulus P)
#       s_m = sqrt(k_z^2 - (omega / V_S_m)^2)       > 0    (annulus S)
#       p   = sqrt(k_z^2 - (omega / V_P)^2)         > 0    (formation P)
#       s   = sqrt(k_z^2 - (omega / V_S)^2)         > 0    (formation S)
#
# Bound-regime gate: all five are real positive iff
#
#       k_z > omega / min(V_S_m, V_S, V_f).
#
# (V_P_m and V_P are always faster than the corresponding shear, so
# they don't tighten the bound.) The implementation in F.1.b raises
# ``ValueError`` when the gate fails -- the I_n + K_n basis no longer
# spans the annulus solution space and a J_n + Y_n sister
# (oscillatory-radial annulus) is required, which is out of scope for
# F.1 and re-enters in plan item G.
#
# Field representation (four scalar potentials, one per
# region+wave). Only the regular-at-origin / decaying-at-infinity
# pieces survive in the bounded regions; the annulus carries both:
#
#   Fluid (r < a):
#       P^{(f)}        = A * I_0(F_f r)
#
#   Annulus (a < r < b):
#       phi^{(m)}      = B_I * I_0(p_m r) + B_K * K_0(p_m r)
#       psi_theta^{(m)} = C_I * I_1(s_m r) + C_K * K_1(s_m r)
#
#   Formation (r > b):
#       phi^{(s)}      = B   * K_0(p r)
#       psi_theta^{(s)} = C   * K_1(s r)
#
# Seven amplitudes in column order:  [ A | B_I, B_K, C_I, C_K | B, C ]
#
# The column partition keeps the three BCs at r = a (substep F.1.a.4
# rows 1-3) supported on columns 1-5 and the four BCs at r = b (rows
# 4-7) supported on columns 2-7 -- a sparse block-bidiagonal layout
# that pays off when the matrix is assembled in F.1.b and that
# generalises to multi-layer in plan item G.

# =====================================================================
# Substep F.1.a.2 -- displacements per region
# =====================================================================
#
# Bessel-derivative identities used below.
#
# Modified-Bessel derivatives (regular and singular branches; sign
# difference is the only structural change going from K_n to I_n):
#
#       I_0'(x) = +I_1(x)             K_0'(x) = -K_1(x)
#       I_1'(x) = I_0(x) - I_1(x)/x   K_1'(x) = -K_0(x) - K_1(x)/x
#
# "(1/r) d_r [r J_1(alpha r)]" identity (J_1 = either I_1 or K_1):
#
#       (1/r) d_r [r I_1(alpha r)] = +alpha * I_0(alpha r)
#       (1/r) d_r [r K_1(alpha r)] = -alpha * K_0(alpha r)
#
# (Derivation for the I-flavor:
#       d_r [r I_1(alpha r)] = I_1(alpha r) + r alpha I_1'(alpha r)
#                            = I_1(alpha r) + r alpha [I_0(alpha r)
#                                                       - I_1(alpha r)/(alpha r)]
#                            = r alpha I_0(alpha r).)
#
# Sign convention: the I-flavor identities are the "positive twin" of
# the K-flavor; every formula below that has a +alpha I-term has a
# -alpha K-term in the same slot.
#
# Fluid displacement (Euler equation, same as the n=0 single-
# interface case):
#
#       u_r^{(f)} = (A F_f / (rho_f omega^2)) * I_1(F_f r)
#       u_z^{(f)} = (i k_z A / (rho_f omega^2)) * I_0(F_f r)
#
# Annulus displacement (Helmholtz decomposition u = grad phi +
# curl psi with psi_r = 0 and psi_z = 0 at axisymmetric n=0):
#
#   u_r^{(m)} = d_r phi^{(m)} - d_z psi_theta^{(m)}
#
#       d_r phi^{(m)}     = B_I p_m I_1(p_m r) - B_K p_m K_1(p_m r)
#       d_z psi_theta^{(m)} = i k_z [C_I I_1(s_m r) + C_K K_1(s_m r)]
#
#       ==> u_r^{(m)} =  B_I p_m I_1(p_m r)
#                       - B_K p_m K_1(p_m r)
#                       - i k_z C_I I_1(s_m r)
#                       - i k_z C_K K_1(s_m r)
#
#   u_z^{(m)} = d_z phi^{(m)} + (1/r) d_r [r psi_theta^{(m)}]
#
#       d_z phi^{(m)}                = i k_z [B_I I_0(p_m r)
#                                              + B_K K_0(p_m r)]
#       (1/r) d_r [r psi_theta^{(m)}] = + s_m C_I I_0(s_m r)
#                                       - s_m C_K K_0(s_m r)
#
#       ==> u_z^{(m)} =  i k_z B_I I_0(p_m r)
#                       + i k_z B_K K_0(p_m r)
#                       +     C_I s_m I_0(s_m r)
#                       -     C_K s_m K_0(s_m r)
#
# Formation displacement (only the K-amplitudes survive, identical to
# the single-interface n=0 form):
#
#       u_r^{(s)} = -B p K_1(p r) - i k_z C K_1(s r)
#       u_z^{(s)} =  i k_z B K_0(p r) - C s K_0(s r)
#
# Note: the annulus expressions reduce to the formation expressions
# when the I-amplitudes vanish (B_I = C_I = 0) and the K-amplitudes
# are renamed (B_K -> -B, C_K -> -C). This sign-flip on the rename is
# fixed by the sign of d_r [B_I I_0] = +B_I p_m I_1 vs the formation
# convention u_r = -B p K_1 (the formation amplitude B absorbs the
# minus from K_0' = -K_1; the annulus K-amplitude B_K is defined so
# d_r [B_K K_0] = -B_K p_m K_1 carries its own sign explicitly).
# Substep F.1.a.6's degenerate-limit cross-check pins this down.

# =====================================================================
# Substep F.1.a.3 -- stresses per region
# =====================================================================
#
# Hooke's law in cylindrical coordinates, isotropic solid:
#
#       sigma_rr = lambda * div(u) + 2 mu * d_r u_r
#       sigma_rz = mu * (d_z u_r + d_r u_z)
#
# (sigma_r_theta vanishes identically for n=0 axisymmetric fields and
# is not a BC; that boundary condition re-appears at n=1.)
#
# Lamé reduction in each region. With the gauge u = grad phi + curl
# psi and psi_r = 0:
#
#       div(u) = laplacian(phi)       (curl is divergence-free).
#
# Both I_n and K_n solve the same modified-Bessel equation
# ``(d_r^2 + (1/r) d_r) f - alpha^2 f = 0``, so for any
# scalar phi(r,z) = f(r) e^{i k_z z} with f built from
# ``I_0(alpha r)`` or ``K_0(alpha r)``:
#
#       laplacian(phi) = (alpha^2 - k_z^2) phi
#                      = -(omega / V_P_region)^2 phi
#                      = - k_P_region^2 phi.
#
# The same algebraic identity used in
# :func:`_modal_determinant_n0` therefore carries through region by
# region:
#
#       - lambda_R k_P_R^2 + 2 mu_R p_R^2 = mu_R (2 k_z^2 - k_S_R^2)
#
# with ``R`` ranging over {annulus, formation} and the per-region
# constants
#
#       lambda_m = rho_m (V_P_m^2 - 2 V_S_m^2),  mu_m = rho_m V_S_m^2
#       lambda   = rho   (V_P^2   - 2 V_S^2  ),  mu   = rho   V_S^2
#       k_S_m    = omega / V_S_m,   k_S = omega / V_S
#
# Stress expressions at a generic radius r (annulus form; formation
# form drops the I-amplitudes):
#
#   sigma_rr^{(m)} =  mu_m (2 k_z^2 - k_S_m^2)
#                          [B_I I_0(p_m r) + B_K K_0(p_m r)]
#                   + 2 mu_m d_r u_r^{(m)}
#
#   d_r u_r^{(m)}  =   B_I p_m^2 I_0(p_m r)
#                            - B_I p_m I_1(p_m r) / r
#                    + B_K p_m^2 K_0(p_m r)
#                            + B_K p_m K_1(p_m r) / r
#                    - i k_z C_I [s_m I_0(s_m r) - I_1(s_m r) / r]
#                    - i k_z C_K [-s_m K_0(s_m r) - K_1(s_m r) / r]
#
# (using I_1' = I_0 - I_1 / x at x = alpha r, etc.)
#
# After collecting on r = a or r = b the stress combinations reduce
# to the same pattern as the n=0 single-interface form, with two
# extra columns per body-wave (the I_n flavour). The explicit
# r = a / r = b row entries are deferred to substep F.1.a.4 below;
# carrying the symbolic forms here keeps the substep blocks
# transcription-checkable.
#
# Axial-shear stress (single Lamé constant ``mu``):
#
#   sigma_rz^{(m)} = mu_m (d_z u_r^{(m)} + d_r u_z^{(m)})
#
#       d_z u_r^{(m)} = i k_z * [annulus u_r^{(m)} expression]
#       d_r u_z^{(m)} = i k_z [B_I p_m I_1(p_m r) - B_K p_m K_1(p_m r)]
#                       + C_I s_m^2 I_1(s_m r)
#                       + C_K s_m * (-K_0' on s_m r form):
#                          d_r [s_m C_K * (-K_0(s_m r))] handled
#                          via d_r K_0(s_m r) = -s_m K_1(s_m r)
#                       i.e. d_r [-C_K s_m K_0(s_m r)] = +C_K s_m^2 K_1(s_m r)
#
# After the row-3-by-i / C-columns-by-(-i) phase rescale (substep
# F.1.a.5) every entry in the resulting matrix is real, exactly as
# in the single-interface n=0 case.
#
# Formation-side stresses are obtained from the annulus forms by
# setting ``B_I = C_I = 0``, swapping ``rho_m -> rho``,
# ``V_P_m -> V_P``, ``V_S_m -> V_S``, and renaming ``B_K -> -B`` ,
# ``C_K -> -C``. This is the same algebraic step that makes
# :func:`_modal_determinant_n0` line up; see substep F.1.a.6 for the
# explicit cross-check.

# =====================================================================
# Substep F.1.a.4 -- 7x7 boundary-condition row layout
# =====================================================================
#
# Rows are the seven boundary conditions; columns are the seven
# amplitudes in the order [A | B_I, B_K, C_I, C_K | B, C].
#
#   Row 1   u_r^{(f)} - u_r^{(m)} = 0           at r = a
#   Row 2   sigma_rr^{(m)} + P^{(f)} = 0        at r = a
#   Row 3   sigma_rz^{(m)} = 0                  at r = a
#   Row 4   u_r^{(m)} - u_r^{(s)} = 0           at r = b
#   Row 5   u_z^{(m)} - u_z^{(s)} = 0           at r = b
#   Row 6   sigma_rr^{(m)} - sigma_rr^{(s)} = 0 at r = b
#   Row 7   sigma_rz^{(m)} - sigma_rz^{(s)} = 0 at r = b
#
# Sparsity pattern (X = generically non-zero, . = identically zero):
#
#               A   B_I B_K C_I C_K |  B   C
#       Row 1 [ X    X   X   X   X  |  .   . ]   <- r = a, fluid+annulus
#       Row 2 [ X    X   X   X   X  |  .   . ]
#       Row 3 [ .    X   X   X   X  |  .   . ]   <- fluid carries no shear
#       Row 4 [ .    X   X   X   X  |  X   X ]   <- r = b, annulus+formation
#       Row 5 [ .    X   X   X   X  |  X   X ]
#       Row 6 [ .    X   X   X   X  |  X   X ]
#       Row 7 [ .    X   X   X   X  |  X   X ]
#
# Block-bidiagonal: A only enters rows 1-2; B and C only enter rows
# 4-7. This is the n=2 generalisation of the 3x3 single-interface
# case (which had A | B, C as the column partition and rows 1-3 all
# at r = a). The extra rows 4-7 are the second interface; the extra
# columns B_I, C_I are the I-flavour annulus amplitudes that absorb
# the displacement jumps continuity demands.
#
# Per-row entries (rows 1-3 reuse the single-interface n=0
# expressions with K -> {I, K} columns; rows 4-7 reuse the same
# expressions for the annulus side and the formation-side sign-
# matched versions for the formation).
#
# The explicit entry list lives next to the matrix construction in
# F.1.b; the cross-check is "feed the layer=formation degenerate
# values and sum the I_K + K columns: the result must match the
# 3-column single-interface form to floating-point precision".

# =====================================================================
# Substep F.1.a.5 -- phase rescaling for a real-valued matrix
# =====================================================================
#
# Pre-rescale phase pattern of each row (zero entries excluded;
# entries are either purely real ``R`` or purely imaginary ``i*R``):
#
#       Row | A  | B_I B_K B | C_I C_K C   <- column groupings
#       ----+----+-----------+-----------
#        1  | R  |   R       |   i*R       (u_r at r = a)
#        2  | R  |   R       |   i*R       (sigma_rr at r = a)
#        3  | 0  |   i*R     |   R         (sigma_rz at r = a)
#        4  | 0  |   R       |   i*R       (u_r at r = b)
#        5  | 0  |   i*R     |   R         (u_z at r = b)        <-- new at layered
#        6  | 0  |   R       |   i*R       (sigma_rr at r = b)
#        7  | 0  |   i*R     |   R         (sigma_rz at r = b)
#
# The ``B-imag, C-real`` pattern is shared by every row whose physical
# meaning is "z-derivative-bearing": sigma_rz (rows 3, 7) and u_z
# (row 5). The complementary ``B-real, C-imag`` pattern lives on
# every "no-z-derivative" row: u_r and sigma_rr (rows 1, 2, 4, 6).
#
# A real matrix is recovered by:
#
#   * Multiplying rows 3, 5, 7 by ``i``  (flips ``i*R -> R`` on B
#     columns and ``R -> i*R`` on C columns of those rows).
#   * Multiplying columns C_I, C_K, C by ``-i``  (flips ``i*R -> R``
#     on rows 1, 2, 4, 6 of the C columns, and ``i*R -> R`` on rows
#     3, 5, 7 of the C columns -- net factor on those crossings is
#     ``i * (-i) = 1``).
#
# Net effect on the determinant:
#
#   det(M_rescaled) = (i^3) * ((-i)^3) * det(M)
#                   = (-i) * (i) * det(M)
#                   = +1 * det(M).
#
# So the rescaling is determinant-preserving (the row factor of ``-i``
# and the column factor of ``+i`` cancel) -- the same property that
# the n=0 single-interface form relies on. The modal-determinant
# convention used throughout this module is to return the rescaled
# (real) determinant directly; substep F.1.a.6's degenerate-limit
# check is what verifies the rescale was wired up correctly.
#
# Note vs the n=0 single-interface case: the single-interface form
# rescales one row (``sigma_rz`` at the only interface) and one
# column (the only ``C``). The layered form rescales three of each
# because two extra rows of the new pattern (row 5 ``u_z`` continuity
# at r = b, plus the second ``sigma_rz`` at r = b) joined the matrix,
# along with two extra C columns (``C_I``, ``C_K``). Mismatching
# this count -- e.g., forgetting to rescale row 5 -- produces a
# matrix with a non-vanishing imaginary part in the bound regime,
# the most direct way to catch a transcription error in F.1.b.

# =====================================================================
# Substep F.1.a.6 -- self-check protocol (degenerate-limit collapse)
# =====================================================================
#
# Two algebraic identities the layered determinant must satisfy by
# construction; both are landed as tests in F.1.d:
#
# (a) Layer = formation. Setting ``(V_P_m, V_S_m, rho_m) =
#     (V_P, V_S, rho)`` makes the annulus material identical to the
#     formation. The annulus then has only the bound (decaying-
#     outward) physical solution; the I-amplitudes (B_I, C_I) must
#     therefore vanish at the root for any thickness. The
#     determinant *root* must coincide with the
#     :func:`_modal_determinant_n0` root to floating-point precision
#     (the determinant *value* differs by a constant pre-factor of
#     no physical relevance).
#
# (b) Thickness -> 0. The rows at r = b approach the rows at r = a
#     in the limit b -> a. In that limit the seven BCs reduce to
#     three independent ones (the four interface-continuity rows
#     collapse onto each other) and the converged k_z must approach
#     the single-interface value continuously.
#
# The two checks together pin both the matrix entries (via (a)) and
# the matrix algebra around the second interface (via (b)). The F.1.b
# transcription is implementation-correct iff both pass.
#
# References for the layered derivation: Cheng & Toksoz (1981) Sect.
# IV, Schmitt (1988) Appendix A, and Tang & Cheng (2004), which
# generalises the same I_n + K_n annulus to N stacked layers and
# feeds plan item G. The section number is dropped deliberately:
# this cited "Sect. 7.1", and that book has six chapters. Its
# multilayered cased-hole treatment is in ch. 2 by the repository's
# own summary (docs/ideas/Tang2004.md), which is not the same as
# having checked the book, so no replacement number is asserted.


# =====================================================================
# Substep F.1.b.2.a -- row 1 of the n=0 layered determinant (r = a)
# =====================================================================
#
# BC1: ``u_r^{(f)}(a) - u_r^{(m)}(a) = 0`` (cos-sector continuity of
# radial displacement at the fluid-annulus interface). Coefficients
# are read off the substep-F.1.a.2 displacement formulae:
#
#   u_r^{(f)} = (A F_f / (rho_f omega^2)) I_1(F_f r)                (cos)
#   u_r^{(m)} =  B_I p_m I_1(p_m r)
#               - B_K p_m K_1(p_m r)
#               - i k_z C_I I_1(s_m r)
#               - i k_z C_K K_1(s_m r)                              (cos)
#
# Substituting r = a and collecting on each amplitude:
#
#       Row 1 (pre-rescale) =
#           [  F_f I_1(F_f a) / (rho_f omega^2),
#             -p_m I_1(p_m a),
#             +p_m K_1(p_m a),
#             +i k_z I_1(s_m a),
#             +i k_z K_1(s_m a),
#              0,                  (formation B; r > b, untouched at a)
#              0  ]                (formation C; r > b, untouched at a)
#
# Phase rescale (substep F.1.a.5) for *this* row: row 1 is one of
# the no-z-derivative rows (B-real, C-imag pre-rescale), so it is
# NOT scaled by ``i`` itself. The C-flavour columns (C_I, C_K, C)
# are scaled by ``-i``, which kills the explicit ``i k_z`` factors
# above. The post-rescale row is therefore real in the bound
# regime; that real form is what
# :func:`_modal_determinant_n0` compares against in the layer=
# formation degenerate limit.


def _layered_n0_row1_at_a(
    kz: float,
    omega: float,
    *,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    layer: BoreholeLayer,
) -> np.ndarray:
    r"""
    Row 1 of the n=0 layered modal determinant evaluated at the
    fluid-annulus interface ``r = a``.

    The row encodes the radial-displacement continuity BC
    ``u_r^{(f)}(a) - u_r^{(m)}(a) = 0`` (cos-sector). Returns the
    seven coefficients in the column order pinned by substep
    F.1.a.4: ``[A | B_I, B_K, C_I, C_K | B, C]``.

    Post-rescale (substep F.1.a.5) form: column-C-by-``-i`` has
    been applied so the ``i k_z`` factors on the C_I / C_K columns
    are cancelled and the row is real-valued in the bound regime.
    Row 1 itself receives no row scaling (it is a no-``z``-derivative
    row).

    Parameters
    ----------
    kz : float
        Trial axial wavenumber (rad / m).
    omega : float
        Angular frequency (rad / s).
    vp, vs, rho : float
        Formation half-space P / S velocity (m/s) and density
        (kg/m^3). Not used by row 1; carried for signature
        uniformity with rows 2-3 of F.1.b.2.
    vf, rho_f : float
        Fluid velocity (m/s) and density (kg/m^3).
    a : float
        Fluid-annulus interface radius (= borehole wall, m).
    layer : BoreholeLayer
        The annular layer; only ``layer.vp``, ``layer.vs`` are used
        by this row.

    Returns
    -------
    ndarray, shape (7,) complex
        Coefficients of (A, B_I, B_K, C_I, C_K, B, C) in row 1.
        Real-valued in the bound regime; complex dtype for
        downstream uniform handling.

    See Also
    --------
    _modal_determinant_n0 : The single-interface n=0 form. At
        layer=formation, ``row[0]``, ``row[2]``, ``row[4]`` of this
        function equal ``M11``, ``M12``, ``M13`` of the single-
        interface matrix bit-exactly.
    """
    del rho  # not used by row 1 (no Lame term); kept for signature uniformity
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=vp,
        vs=vs,
        vf=vf,
        layer=layer,
    )
    I1_Ff_a = float(special.iv(1, F_f * a))
    I1_pm_a = float(special.iv(1, p_m * a))
    K1_pm_a = float(special.kv(1, p_m * a))
    I1_sm_a = float(special.iv(1, s_m * a))
    K1_sm_a = float(special.kv(1, s_m * a))

    row: np.ndarray = np.zeros(7, dtype=complex)
    row[0] = F_f * I1_Ff_a / (rho_f * omega**2)  # A column
    row[1] = -p_m * I1_pm_a  # B_I column
    row[2] = +p_m * K1_pm_a  # B_K column
    # Pre-rescale C columns carry +i k_z; column-by-(-i) makes them real.
    row[3] = +kz * I1_sm_a  # C_I column
    row[4] = +kz * K1_sm_a  # C_K column
    # Formation columns vanish at r = a (fields evaluated outside
    # their region of support).
    row[5] = 0.0
    row[6] = 0.0
    return row


# =====================================================================
# Substep F.1.b.2.b -- row 2 of the n=0 layered determinant (r = a)
# =====================================================================
#
# BC2: ``-(sigma_rr^{(m)}(a) + P^{(f)}(a)) = 0`` (cos-sector normal-
# stress balance, row negated for visual parallel with the n=0
# single-interface form). Coefficients follow the substep-F.1.a.3
# stress derivation. With the Lame reduction
#
#       -lambda_m k_Pm^2 + 2 mu_m p_m^2 = mu_m (2 k_z^2 - k_Sm^2),
#
# and the I_n / K_n derivative identities from F.1.a.2, the
# pre-rescale row is:
#
#       Row 2 (pre-rescale) =
#           [ -I_0(F_f a),
#             -mu_m [(2 k_z^2 - k_Sm^2) I_0(p_m a)
#                    - 2 p_m I_1(p_m a) / a],
#             -mu_m [(2 k_z^2 - k_Sm^2) K_0(p_m a)
#                    + 2 p_m K_1(p_m a) / a],
#             +2 i mu_m k_z [s_m I_0(s_m a) - I_1(s_m a) / a],
#             -2 i mu_m k_z [s_m K_0(s_m a) + K_1(s_m a) / a],
#              0,
#              0 ]
#
# Sign-flip pattern between the I- and K-flavour columns (the
# "d_r-induced" twist): the I_0 / K_0 coefficient carries the same
# sign in both columns, while the I_1 / K_1 coefficient flips sign
# (because ``I_0' = +I_1`` vs ``K_0' = -K_1``; same effect on the
# tangential SV term ``s I_0 - I_1/a`` vs ``s K_0 + K_1/a``).
#
# Phase rescale (substep F.1.a.5): row 2 is no-``z``-derivative, no
# row scaling. Column-by-(-i) on the C_I, C_K columns kills the
# explicit ``i`` factors; the post-rescale row is real-valued in
# the bound regime.


def _layered_n0_row2_at_a(
    kz: float,
    omega: float,
    *,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    layer: BoreholeLayer,
) -> np.ndarray:
    r"""
    Row 2 of the n=0 layered modal determinant evaluated at the
    fluid-annulus interface ``r = a``.

    The row encodes the negated normal-stress balance BC
    ``-(sigma_rr^{(m)}(a) + P^{(f)}(a)) = 0`` in the cos-sector.
    Returns the seven post-rescale coefficients in the column
    order pinned by substep F.1.a.4: ``[A | B_I, B_K, C_I, C_K |
    B, C]``.

    Parameters
    ----------
    kz : float
        Trial axial wavenumber (rad / m).
    omega : float
        Angular frequency (rad / s).
    vp, vs, rho : float
        Formation half-space P / S velocity and density. Carried
        for signature uniformity; row 2 only consumes them
        indirectly via the radial-wavenumber helper.
    vf, rho_f : float
        Fluid velocity (m/s) and density (kg/m^3).
    a : float
        Fluid-annulus interface radius (m).
    layer : BoreholeLayer
        The annular layer. ``layer.rho`` and ``layer.vs`` set
        ``mu_m``; ``layer.vp``, ``layer.vs`` set ``k_Sm``,
        ``p_m``, ``s_m``.

    Returns
    -------
    ndarray, shape (7,) complex
        Coefficients of (A, B_I, B_K, C_I, C_K, B, C) in row 2.
        Real-valued in the bound regime.

    See Also
    --------
    _modal_determinant_n0 : The single-interface n=0 form. At
        layer=formation, ``row[0]``, ``row[2]``, ``row[4]`` of this
        function equal ``M21``, ``M22``, ``M23`` of the single-
        interface matrix bit-exactly.
    """
    del rho  # not used by row 2; kept for signature uniformity
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=vp,
        vs=vs,
        vf=vf,
        layer=layer,
    )
    I0_Ff_a = float(special.iv(0, F_f * a))
    I0_pm_a = float(special.iv(0, p_m * a))
    I1_pm_a = float(special.iv(1, p_m * a))
    K0_pm_a = float(special.kv(0, p_m * a))
    K1_pm_a = float(special.kv(1, p_m * a))
    I0_sm_a = float(special.iv(0, s_m * a))
    I1_sm_a = float(special.iv(1, s_m * a))
    K0_sm_a = float(special.kv(0, s_m * a))
    K1_sm_a = float(special.kv(1, s_m * a))

    mu_m = layer.rho * layer.vs * layer.vs
    kSm2 = (omega / layer.vs) ** 2
    two_kz2_minus_kSm2 = 2.0 * kz * kz - kSm2

    row: np.ndarray = np.zeros(7, dtype=complex)
    # A column: fluid pressure -P^{(f)}(a) = -A I_0(F_f a).
    row[0] = -I0_Ff_a
    # B_I column: annulus P, regular branch. Sign on the I_1 term
    # is opposite the K_1 sign (d_r-induced twist).
    row[1] = -mu_m * (two_kz2_minus_kSm2 * I0_pm_a - 2.0 * p_m * I1_pm_a / a)
    # B_K column: annulus P, singular branch. Mirrors the
    # single-interface M22 with ``p -> p_m``, ``mu -> mu_m``,
    # ``kS^2 -> kSm^2``.
    row[2] = -mu_m * (two_kz2_minus_kSm2 * K0_pm_a + 2.0 * p_m * K1_pm_a / a)
    # C_I column (post-rescale; col-by-(-i) cancels the +i).
    row[3] = +2.0 * kz * mu_m * (s_m * I0_sm_a - I1_sm_a / a)
    # C_K column (post-rescale).
    row[4] = -2.0 * kz * mu_m * (s_m * K0_sm_a + K1_sm_a / a)
    # Formation columns vanish at r = a.
    row[5] = 0.0
    row[6] = 0.0
    return row


# =====================================================================
# Substep F.1.b.2.c -- row 3 of the n=0 layered determinant (r = a)
# =====================================================================
#
# BC3: ``sigma_rz^{(m)}(a) = 0`` (cos-sector axial-shear vanishing
# at the fluid-annulus interface; fluid carries no shear, so the
# fluid side is identically zero -- column A is therefore identically
# zero in this row). Coefficients follow from
# ``sigma_rz = mu (d_z u_r + d_r u_z)`` evaluated at the annulus side
# with the substep-F.1.a.2 displacement formulae:
#
#       d_z u_r^{(m)}(r) + d_r u_z^{(m)}(r)
#         = +2 i k_z B_I p_m I_1(p_m r)
#           - 2 i k_z B_K p_m K_1(p_m r)
#           + (k_z^2 + s_m^2) C_I I_1(s_m r)
#           + (k_z^2 + s_m^2) C_K K_1(s_m r)
#
# with ``k_z^2 + s_m^2 = 2 k_z^2 - k_Sm^2`` (using
# ``s_m^2 = k_z^2 - k_Sm^2``). Multiplying by ``mu_m`` gives the
# pre-rescale row:
#
#       Row 3 (pre-rescale) =
#           [  0,
#             +2 i k_z mu_m p_m I_1(p_m a),
#             -2 i k_z mu_m p_m K_1(p_m a),
#             +mu_m (2 k_z^2 - k_Sm^2) I_1(s_m a),
#             +mu_m (2 k_z^2 - k_Sm^2) K_1(s_m a),
#              0,
#              0 ]
#
# Imaginary-power pattern: B-columns are ``i*R``, C-columns are
# ``R`` -- the *opposite* of rows 1, 2 (the "z-derivative-bearing"
# pattern from substep F.1.a.5). This is what makes row 3 (and
# rows 5, 7) require an extra ``row * i`` rescale.
#
# Phase rescale (substep F.1.a.5):
#
#   * Row 3 is multiplied by ``i``: B-columns become
#     ``i * (i*R) = -R`` (real, opposite sign). C-columns become
#     ``i * R = i*R`` (imaginary, but cancelled by column rescale).
#   * Columns C_I, C_K are then multiplied by ``-i``: the post-
#     row-rescale ``i*R`` becomes ``(-i)(i*R) = R`` (real).
#
# Post-rescale row:
#
#       row3 = [  0,
#                -2 k_z mu_m p_m I_1(p_m a),
#                +2 k_z mu_m p_m K_1(p_m a),
#                +mu_m (2 k_z^2 - k_Sm^2) I_1(s_m a),
#                +mu_m (2 k_z^2 - k_Sm^2) K_1(s_m a),
#                 0,
#                 0 ]
#
# At layer=formation this matches ``M31, M32, M33`` from
# :func:`_modal_determinant_n0` directly.


def _layered_n0_row3_at_a(
    kz: float,
    omega: float,
    *,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    layer: BoreholeLayer,
) -> np.ndarray:
    r"""
    Row 3 of the n=0 layered modal determinant evaluated at the
    fluid-annulus interface ``r = a``.

    Encodes the axial-shear-vanishing BC ``sigma_rz^{(m)}(a) = 0``
    in the cos sector. Returns the seven post-rescale coefficients
    in the column order pinned by substep F.1.a.4: ``[A | B_I,
    B_K, C_I, C_K | B, C]``.

    Row 3 is one of the substep-F.1.a.5 ``z``-derivative-bearing
    rows: pre-rescale, the B-columns are pure imaginary and the
    C-columns are real. The full ``row * i`` rescale plus the
    column-by-(-i) on C columns lands the row in real form (in
    the bound regime). Column A is identically zero because the
    fluid carries no shear stress.

    Parameters
    ----------
    kz : float
        Trial axial wavenumber (rad / m).
    omega : float
        Angular frequency (rad / s).
    vp, vs, rho : float
        Formation half-space P / S velocity and density. Carried
        for signature uniformity; not used by row 3.
    vf, rho_f : float
        Fluid velocity and density. Not used by row 3 (fluid
        carries no shear); carried for signature uniformity.
    a : float
        Fluid-annulus interface radius (m).
    layer : BoreholeLayer
        The annular layer; ``layer.rho``, ``layer.vs`` set
        ``mu_m``; ``layer.vp``, ``layer.vs`` set ``p_m``,
        ``s_m``, ``k_Sm``.

    Returns
    -------
    ndarray, shape (7,) complex
        Coefficients of (A, B_I, B_K, C_I, C_K, B, C) in row 3.
        Real-valued in the bound regime.

    See Also
    --------
    _modal_determinant_n0 : The single-interface n=0 form. At
        layer=formation, ``row[0] = M31 = 0``, ``row[2] = M32``,
        ``row[4] = M33`` bit-exactly.
    """
    del rho, rho_f  # not used by row 3; kept for signature uniformity
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=vp,
        vs=vs,
        vf=vf,
        layer=layer,
    )
    del F_f  # row 3 doesn't touch the fluid (no-shear) column
    I1_pm_a = float(special.iv(1, p_m * a))
    K1_pm_a = float(special.kv(1, p_m * a))
    I1_sm_a = float(special.iv(1, s_m * a))
    K1_sm_a = float(special.kv(1, s_m * a))

    mu_m = layer.rho * layer.vs * layer.vs
    kSm2 = (omega / layer.vs) ** 2
    two_kz2_minus_kSm2 = 2.0 * kz * kz - kSm2

    row: np.ndarray = np.zeros(7, dtype=complex)
    # A column: fluid carries no shear; identically zero.
    row[0] = 0.0
    # B_I column (post-rescale; row-by-i flips the +i*R to -R).
    row[1] = -2.0 * kz * mu_m * p_m * I1_pm_a
    # B_K column (post-rescale; row-by-i flips -i*R to +R).
    row[2] = +2.0 * kz * mu_m * p_m * K1_pm_a
    # C_I column (post-rescale; row*i AND col*-i, net factor i*-i=1
    # leaves the original real R unchanged).
    row[3] = +mu_m * two_kz2_minus_kSm2 * I1_sm_a
    # C_K column (post-rescale; same as C_I).
    row[4] = +mu_m * two_kz2_minus_kSm2 * K1_sm_a
    # Formation columns vanish at r = a.
    row[5] = 0.0
    row[6] = 0.0
    return row


# =====================================================================
# Substep F.1.b.3.a -- row 4 of the n=0 layered determinant (r = b)
# =====================================================================
#
# BC4: ``u_r^{(m)}(b) - u_r^{(s)}(b) = 0`` (cos-sector continuity of
# radial displacement at the annulus-formation interface, r = b =
# a + layer.thickness). First of the four interface-continuity rows
# at the second interface; no single-interface analog exists for
# rows 4-7.
#
# Coefficients from substep F.1.a.2 (same displacement formulae as
# row 1, evaluated at r = b instead of r = a):
#
#   u_r^{(m)}(b) = +B_I p_m I_1(p_m b) - B_K p_m K_1(p_m b)
#                  - i k_z C_I I_1(s_m b) - i k_z C_K K_1(s_m b)
#   u_r^{(s)}(b) = -B p K_1(p b) - i k_z C K_1(s b)
#
# Subtracting (annulus - formation):
#
#       Row 4 (pre-rescale) =
#           [  0,                              (A; fluid r<a, untouched)
#             +p_m I_1(p_m b),                 (B_I)
#             -p_m K_1(p_m b),                 (B_K)
#             -i k_z I_1(s_m b),               (C_I)
#             -i k_z K_1(s_m b),               (C_K)
#             +p K_1(p b),                     (B; subtracted, sign flip)
#             +i k_z K_1(s b) ]                (C; subtracted, sign flip)
#
# Sign-flip pattern between annulus and formation K-flavour columns:
# the annulus B_K coefficient is ``-p_m K_1(p_m b)`` while the
# formation B coefficient is ``+p K_1(p b)``. At layer=formation
# (``p_m -> p``) these add to ZERO -- the substep-F.1.a.6 self-check
# at the row level. Same for C_K vs C: ``-i k_z K_1(s_m b)`` vs
# ``+i k_z K_1(s b)`` cancel at layer=formation.
#
# This cancellation is the central correctness invariant for rows
# 4-7: when the annulus material matches the formation, the second
# interface becomes fictitious and the K-flavour outgoing-wave
# columns from both sides must cancel. The I-flavour columns
# (B_I, C_I) have no formation counterpart and are not constrained
# by this identity (the eigenvector at the Stoneley root has zero
# I-amplitudes when layer=formation, but each individual entry is
# generically non-zero).
#
# Phase rescale (substep F.1.a.5): row 4 is a no-z-derivative row
# (B-real, C-imag pre-rescale), so no row scaling. Column-by-(-i)
# on C_I, C_K, C kills the explicit ``i k_z`` factors.


def _layered_n0_row4_at_b(
    kz: float,
    omega: float,
    *,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    layer: BoreholeLayer,
) -> np.ndarray:
    r"""
    Row 4 of the n=0 layered modal determinant evaluated at the
    annulus-formation interface ``r = b = a + layer.thickness``.

    Encodes the radial-displacement continuity BC
    ``u_r^{(m)}(b) - u_r^{(s)}(b) = 0`` in the cos sector. Returns
    the seven post-rescale coefficients in the column order pinned
    by substep F.1.a.4: ``[A | B_I, B_K, C_I, C_K | B, C]``.

    Parameters
    ----------
    kz : float
        Trial axial wavenumber (rad / m).
    omega : float
        Angular frequency (rad / s).
    vp, vs : float
        Formation half-space P / S velocities (m/s). Set the
        formation radial wavenumbers ``p, s`` used by columns 5, 6.
    rho : float
        Formation density (kg/m^3). Carried for signature
        uniformity; row 4 doesn't use density (no stress terms).
    vf, rho_f : float
        Fluid velocity / density. Not used (fluid doesn't reach
        r = b); carried for signature uniformity.
    a : float
        Fluid-annulus interface radius (m); ``b = a + layer.thickness``.
    layer : BoreholeLayer
        The annular layer; sets ``b``, ``p_m``, ``s_m``.

    Returns
    -------
    ndarray, shape (7,) complex
        Coefficients of (A, B_I, B_K, C_I, C_K, B, C) in row 4.
        Real-valued in the bound regime.

    See Also
    --------
    _layered_n0_row1_at_a : The same physical BC (u_r continuity)
        at the first interface ``r = a``. The annulus-side entries
        in row 4 carry the *opposite* sign vs row 1 because the
        annulus appears with opposite sign in the two BCs
        (``u_r^{(f)} - u_r^{(m)}`` at r=a vs
        ``u_r^{(m)} - u_r^{(s)}`` at r=b).
    """
    del rho, rho_f  # not used by row 4; kept for signature uniformity
    F_f, p_m, s_m, p, s = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=vp,
        vs=vs,
        vf=vf,
        layer=layer,
    )
    del F_f  # row 4 doesn't touch the fluid column
    b = a + layer.thickness
    I1_pm_b = float(special.iv(1, p_m * b))
    K1_pm_b = float(special.kv(1, p_m * b))
    I1_sm_b = float(special.iv(1, s_m * b))
    K1_sm_b = float(special.kv(1, s_m * b))
    K1_p_b = float(special.kv(1, p * b))
    K1_s_b = float(special.kv(1, s * b))

    row: np.ndarray = np.zeros(7, dtype=complex)
    # A column: fluid is at r < a; doesn't reach r = b.
    row[0] = 0.0
    # B_I column (annulus P, regular branch).
    row[1] = +p_m * I1_pm_b
    # B_K column (annulus P, singular branch).
    row[2] = -p_m * K1_pm_b
    # C_I column (post-rescale; col-by-(-i) cancels the -i factor,
    # leaving -k_z * I_1(s_m b)).
    row[3] = -kz * I1_sm_b
    # C_K column (post-rescale).
    row[4] = -kz * K1_sm_b
    # B column (formation P; carries opposite sign vs B_K because
    # u_r^{(s)} appears subtracted in the BC).
    row[5] = +p * K1_p_b
    # C column (post-rescale; sign-flipped vs C_K for the same
    # reason).
    row[6] = +kz * K1_s_b
    return row


# =====================================================================
# Substep F.1.b.3.b -- row 5 of the n=0 layered determinant (r = b)
# =====================================================================
#
# BC5: ``u_z^{(m)}(b) - u_z^{(s)}(b) = 0`` (cos-sector continuity of
# axial displacement at the annulus-formation interface). This BC
# is GENUINELY NEW at the layered case -- the single-interface
# fluid-solid BC at r = a replaces u_z continuity with
# ``sigma_rz = 0`` (an inviscid fluid imposes no axial-shear
# constraint on the formation, so the formation wall is free
# axially).
#
# Coefficients from substep F.1.a.2 displacement formulae:
#
#   u_z^{(m)}(r) = +i k_z B_I I_0(p_m r) + i k_z B_K K_0(p_m r)
#                  + C_I s_m I_0(s_m r) - C_K s_m K_0(s_m r)
#   u_z^{(s)}(r) = +i k_z B K_0(p r) - C s K_0(s r)
#
# Subtracting (annulus - formation):
#
#       Row 5 (pre-rescale) =
#           [  0,                              (A; fluid r<a)
#             +i k_z I_0(p_m b),               (B_I)
#             +i k_z K_0(p_m b),               (B_K)
#             +s_m I_0(s_m b),                 (C_I)
#             -s_m K_0(s_m b),                 (C_K)
#             -i k_z K_0(p b),                 (B; subtracted, sign flip)
#             +s K_0(s b) ]                    (C; subtracted, sign flip)
#
# Imaginary-power pattern: B-columns are ``i*R``, C-columns are
# ``R`` -- the *opposite* of rows 1, 4 and the *same* as the
# sigma_rz rows 3, 7. This is the substep-F.1.a.5 "z-derivative-
# bearing" pattern, and the source of the row 5 / row 7 row-by-i
# rescale. Forgetting that row 5 needs scaling -- treating it as
# a u_r-style row -- is the most direct transcription error to
# make at the layered case (specifically called out in F.1.a.5).
#
# Phase rescale (substep F.1.a.5):
#
#   * Row 5 is multiplied by ``i``: B-columns become real
#     (post-rescale ``-k_z * I_0`` etc.), C-columns become
#     imaginary (``+i s_m * I_0`` etc.).
#   * Columns C_I, C_K, C are then multiplied by ``-i``: the C
#     entries become real (``+s_m I_0`` etc.).
#
# Note: row 5 uses degree-0 Bessel functions (``I_0``, ``K_0``)
# on every annulus / formation column, contrasting with row 4
# (and rows 1, 6) which use degree-1 (``I_1``, ``K_1``). This
# reflects the ``u_z = (i k_z phi) + (1/r) d_r(r psi_theta)``
# decomposition: the P term carries ``B I_0`` (no Bessel-index
# shift) and the SV term carries ``C s I_0`` after the
# ``(1/r) d_r [r I_1] = +s I_0`` reduction.


def _layered_n0_row5_at_b(
    kz: float,
    omega: float,
    *,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    layer: BoreholeLayer,
) -> np.ndarray:
    r"""
    Row 5 of the n=0 layered modal determinant evaluated at the
    annulus-formation interface ``r = b = a + layer.thickness``.

    Encodes the axial-displacement continuity BC
    ``u_z^{(m)}(b) - u_z^{(s)}(b) = 0`` in the cos sector. This BC
    is new at the layered case (no single-interface analog).
    Returns the seven post-rescale coefficients in the column
    order pinned by substep F.1.a.4.

    Parameters
    ----------
    kz : float
        Trial axial wavenumber (rad / m).
    omega : float
        Angular frequency (rad / s).
    vp, vs : float
        Formation half-space P / S velocities (m/s); set the
        formation radial wavenumbers ``p, s`` used by columns 5, 6.
    rho : float
        Formation density. Carried for signature uniformity;
        not used by row 5 (no stress terms).
    vf, rho_f : float
        Fluid velocity / density. Not used by row 5 (fluid
        doesn't reach r = b); carried for signature uniformity.
    a : float
        Fluid-annulus interface radius (m); ``b = a + layer.thickness``.
    layer : BoreholeLayer
        The annular layer.

    Returns
    -------
    ndarray, shape (7,) complex
        Coefficients of (A, B_I, B_K, C_I, C_K, B, C) in row 5.
        Real-valued in the bound regime.

    See Also
    --------
    _layered_n0_row3_at_a : Shares the substep-F.1.a.5 imaginary-
        power pattern (B-imag, C-real pre-rescale). Row 5 is the
        u_z continuity counterpart at the second interface.
    """
    del rho, rho_f  # not used by row 5; kept for signature uniformity
    F_f, p_m, s_m, p, s = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=vp,
        vs=vs,
        vf=vf,
        layer=layer,
    )
    del F_f  # row 5 doesn't touch the fluid column
    b = a + layer.thickness
    I0_pm_b = float(special.iv(0, p_m * b))
    K0_pm_b = float(special.kv(0, p_m * b))
    I0_sm_b = float(special.iv(0, s_m * b))
    K0_sm_b = float(special.kv(0, s_m * b))
    K0_p_b = float(special.kv(0, p * b))
    K0_s_b = float(special.kv(0, s * b))

    row: np.ndarray = np.zeros(7, dtype=complex)
    # A column: fluid is at r < a; doesn't reach r = b.
    row[0] = 0.0
    # B_I column (post-rescale; row*i flips +i*R to -R).
    row[1] = -kz * I0_pm_b
    # B_K column (post-rescale; row*i flips +i*R to -R).
    row[2] = -kz * K0_pm_b
    # C_I column (post-rescale; row*i AND col*-i, net factor 1).
    row[3] = +s_m * I0_sm_b
    # C_K column (post-rescale; same as C_I).
    row[4] = -s_m * K0_sm_b
    # B column (formation; row*i flips -i*R to +R).
    row[5] = +kz * K0_p_b
    # C column (post-rescale; row*i AND col*-i, net factor 1).
    row[6] = +s * K0_s_b
    return row


# =====================================================================
# Substep F.1.b.3.c -- row 6 of the n=0 layered determinant (r = b)
# =====================================================================
#
# BC6: ``sigma_rr^{(m)}(b) - sigma_rr^{(s)}(b) = 0`` (cos-sector
# normal-stress continuity at the annulus-formation interface).
# Lame-reduction row at the second interface; structurally similar
# to row 2 at r=a, but with non-zero formation columns. Convention
# differs from row 2 in one nuance: row 2 used the negated form
# ``-(sigma_rr^{(s)} + P^{(f)}) = 0`` for visual parallel with the
# n=0 single-interface; row 6 uses the unnegated continuity form
# directly. The choice is internal (it flips an overall sign on
# the row, which preserves the determinant root).
#
# Coefficients via the F.1.a.3 derivation. The annulus side is
# row-2-like with the same Lame reduction
# ``-lambda_m k_Pm^2 + 2 mu_m p_m^2 = mu_m (2 k_z^2 - k_Sm^2)``
# but evaluated at r = b. The formation side reuses the n=0
# single-interface ``sigma_rr^{(s)}`` with the Lame reduction
# carried over (formation params).
#
# Pre-rescale row:
#
#       Row 6 =
#           [  0,                                  (A; fluid r<a)
#             +mu_m (2 k_z^2 - k_Sm^2) I_0(p_m b)
#                  - 2 mu_m p_m I_1(p_m b) / b,    (B_I)
#             +mu_m (2 k_z^2 - k_Sm^2) K_0(p_m b)
#                  + 2 mu_m p_m K_1(p_m b) / b,    (B_K)
#             -2 i mu_m k_z [s_m I_0(s_m b)
#                            - I_1(s_m b) / b],    (C_I)
#             +2 i mu_m k_z [s_m K_0(s_m b)
#                            + K_1(s_m b) / b],    (C_K)
#             -mu (2 k_z^2 - k_S^2) K_0(p b)
#                  - 2 mu p K_1(p b) / b,          (B; subtracted)
#             -2 i mu k_z [s K_0(s b)
#                          + K_1(s b) / b] ]       (C; subtracted)
#
# Imaginary-power pattern: B-real, C-imag pre-rescale (the same
# substep-F.1.a.5 pattern as rows 1, 4 -- no row scaling). The
# column-by-(-i) on C_I, C_K, C kills the explicit ``i`` factors.
#
# Substep-F.1.a.6 K-flavour cancellation at layer=formation:
# ``mu_m -> mu``, ``p_m -> p``, ``s_m -> s``, ``k_Sm -> k_S``
# makes the annulus B_K coefficient and the formation B
# coefficient negatives of each other (B_K is +Lame-form, B is
# -Lame-form), and similarly for C_K vs C. Both pairs cancel.


def _layered_n0_row6_at_b(
    kz: float,
    omega: float,
    *,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    layer: BoreholeLayer,
) -> np.ndarray:
    r"""
    Row 6 of the n=0 layered modal determinant evaluated at the
    annulus-formation interface ``r = b = a + layer.thickness``.

    Encodes the normal-stress continuity BC
    ``sigma_rr^{(m)}(b) - sigma_rr^{(s)}(b) = 0`` in the cos
    sector. Returns the seven post-rescale coefficients in the
    column order pinned by substep F.1.a.4.

    Parameters
    ----------
    kz : float
        Trial axial wavenumber (rad / m).
    omega : float
        Angular frequency (rad / s).
    vp, vs, rho : float
        Formation half-space P / S velocities and density. All
        used (mu = rho * vs^2; k_S = omega / vs; p, s set the
        Bessel arguments on columns 5, 6).
    vf, rho_f : float
        Fluid velocity / density. Not used (fluid doesn't reach
        r = b); carried for signature uniformity.
    a : float
        Fluid-annulus interface radius (m); ``b = a + layer.thickness``.
    layer : BoreholeLayer
        The annular layer; sets ``mu_m``, ``k_Sm``, ``p_m``, ``s_m``.

    Returns
    -------
    ndarray, shape (7,) complex
        Coefficients of (A, B_I, B_K, C_I, C_K, B, C) in row 6.
        Real-valued in the bound regime.

    See Also
    --------
    _layered_n0_row2_at_a : Row 2 (sigma_rr balance at r=a). Same
        Lame-reduction structure; row 2 uses the ``-(sigma_rr + P)
        = 0`` negated convention while row 6 uses the unnegated
        continuity form directly.
    """
    del rho_f  # not used by row 6; kept for signature uniformity
    F_f, p_m, s_m, p, s = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=vp,
        vs=vs,
        vf=vf,
        layer=layer,
    )
    del F_f  # row 6 doesn't touch the fluid column
    b = a + layer.thickness
    I0_pm_b = float(special.iv(0, p_m * b))
    I1_pm_b = float(special.iv(1, p_m * b))
    K0_pm_b = float(special.kv(0, p_m * b))
    K1_pm_b = float(special.kv(1, p_m * b))
    I0_sm_b = float(special.iv(0, s_m * b))
    I1_sm_b = float(special.iv(1, s_m * b))
    K0_sm_b = float(special.kv(0, s_m * b))
    K1_sm_b = float(special.kv(1, s_m * b))
    K0_p_b = float(special.kv(0, p * b))
    K1_p_b = float(special.kv(1, p * b))
    K0_s_b = float(special.kv(0, s * b))
    K1_s_b = float(special.kv(1, s * b))

    mu_m = layer.rho * layer.vs * layer.vs
    kSm2 = (omega / layer.vs) ** 2
    two_kz2_minus_kSm2 = 2.0 * kz * kz - kSm2

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    row: np.ndarray = np.zeros(7, dtype=complex)
    # A column: fluid is at r < a; doesn't reach r = b.
    row[0] = 0.0
    # B_I column (annulus P, regular branch).
    row[1] = +mu_m * (two_kz2_minus_kSm2 * I0_pm_b - 2.0 * p_m * I1_pm_b / b)
    # B_K column (annulus P, singular branch).
    row[2] = +mu_m * (two_kz2_minus_kSm2 * K0_pm_b + 2.0 * p_m * K1_pm_b / b)
    # C_I column (post-rescale; col-by-(-i) cancels the -i factor).
    row[3] = -2.0 * kz * mu_m * (s_m * I0_sm_b - I1_sm_b / b)
    # C_K column (post-rescale).
    row[4] = +2.0 * kz * mu_m * (s_m * K0_sm_b + K1_sm_b / b)
    # B column (formation; subtracted, hence opposite sign of B_K
    # at layer=formation; the K-flavour cancellation works
    # automatically).
    row[5] = -mu * (two_kz2_minus_kS2 * K0_p_b + 2.0 * p * K1_p_b / b)
    # C column (post-rescale; subtracted).
    row[6] = -2.0 * kz * mu * (s * K0_s_b + K1_s_b / b)
    return row


# =====================================================================
# Substep F.1.b.3.d -- row 7 of the n=0 layered determinant (r = b)
# =====================================================================
#
# BC7: ``sigma_rz^{(m)}(b) - sigma_rz^{(s)}(b) = 0`` (cos-sector
# axial-shear continuity at the annulus-formation interface).
# Final row of the 7x7 layered determinant; closes substep F.1.b.3
# and unlocks F.1.b.4 assembly.
#
# Coefficients via the F.1.a.3 derivation. The annulus side mirrors
# row 3 at r=a but evaluated at r=b; the formation side reuses the
# n=0 single-interface ``sigma_rz^{(s)}`` form, derived from
# ``mu (d_z u_r^{(s)} + d_r u_z^{(s)})`` with the same ``k_z^2 +
# s^2 = 2 k_z^2 - k_S^2`` collapse used in row 3:
#
#       sigma_rz^{(s)}(r) = mu * [ -2 i k_z B p K_1(p r)
#                                  + (2 k_z^2 - k_S^2) C K_1(s r) ]
#
# Subtracting (annulus - formation):
#
#       Row 7 (pre-rescale) =
#           [  0,                                          (A)
#             +2 i k_z mu_m p_m I_1(p_m b),                (B_I)
#             -2 i k_z mu_m p_m K_1(p_m b),                (B_K)
#             +mu_m (2 k_z^2 - k_Sm^2) I_1(s_m b),         (C_I)
#             +mu_m (2 k_z^2 - k_Sm^2) K_1(s_m b),         (C_K)
#             +2 i k_z mu p K_1(p b),                      (B; subtracted)
#             -mu (2 k_z^2 - k_S^2) K_1(s b) ]             (C; subtracted)
#
# Imaginary-power pattern: B-imag, C-real (the substep-F.1.a.5
# ``z``-derivative-bearing pattern, like rows 3, 5). Phase rescale:
# row 7 multiplied by ``i``; columns C_I, C_K, C multiplied by
# ``-i``. Forgetting the row * i is the same easy-to-miss error
# F.1.a.5 calls out for row 5; the bound-regime imaginary-part
# test catches it.
#
# Substep-F.1.a.6 K-flavour cancellation at layer=formation:
# ``mu_m -> mu, p_m -> p, s_m -> s, k_Sm -> k_S``. The annulus
# B_K coefficient ``+2 k_z mu p K_1(p b)`` (post-rescale) and the
# formation B coefficient ``-2 k_z mu p K_1(p b)`` (post-rescale)
# cancel. Same for C_K vs C with the ``mu (2 k_z^2 - k_S^2) K_1``
# pair.
#
# Layer=formation analog of row 3 at r=a: ``row7[2]`` and
# ``row7[4]`` post-rescale match the n=0 ``M32`` and ``M33``
# entries of :func:`_modal_determinant_n0` evaluated at r=b
# (same Lame-form coefficients, different Bessel argument).


def _layered_n0_row7_at_b(
    kz: float,
    omega: float,
    *,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    layer: BoreholeLayer,
) -> np.ndarray:
    r"""
    Row 7 of the n=0 layered modal determinant evaluated at the
    annulus-formation interface ``r = b = a + layer.thickness``.

    Encodes the axial-shear continuity BC
    ``sigma_rz^{(m)}(b) - sigma_rz^{(s)}(b) = 0`` in the cos
    sector. Final row of the 7x7 layered determinant.

    Parameters
    ----------
    kz : float
        Trial axial wavenumber (rad / m).
    omega : float
        Angular frequency (rad / s).
    vp, vs, rho : float
        Formation half-space P / S velocities and density. All
        used (mu = rho * vs^2; k_S = omega / vs; p, s set the
        Bessel arguments on columns 5, 6).
    vf, rho_f : float
        Fluid velocity / density. Not used (fluid doesn't reach
        r = b and carries no shear); carried for signature
        uniformity.
    a : float
        Fluid-annulus interface radius (m); ``b = a + layer.thickness``.
    layer : BoreholeLayer
        The annular layer; sets ``mu_m``, ``k_Sm``, ``p_m``, ``s_m``.

    Returns
    -------
    ndarray, shape (7,) complex
        Coefficients of (A, B_I, B_K, C_I, C_K, B, C) in row 7.
        Real-valued in the bound regime.

    See Also
    --------
    _layered_n0_row3_at_a : Row 3 (sigma_rz = 0 at r=a). Same
        ``z``-derivative-bearing imaginary-power pattern; row 7 is
        the continuity counterpart at the second interface.
    _layered_n0_row5_at_b : Row 5 (u_z continuity at r=b). Shares
        the ``B-imag, C-real`` pre-rescale pattern.
    """
    del rho_f  # not used by row 7; kept for signature uniformity
    F_f, p_m, s_m, p, s = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=vp,
        vs=vs,
        vf=vf,
        layer=layer,
    )
    del F_f  # row 7 doesn't touch the fluid column
    b = a + layer.thickness
    I1_pm_b = float(special.iv(1, p_m * b))
    K1_pm_b = float(special.kv(1, p_m * b))
    I1_sm_b = float(special.iv(1, s_m * b))
    K1_sm_b = float(special.kv(1, s_m * b))
    K1_p_b = float(special.kv(1, p * b))
    K1_s_b = float(special.kv(1, s * b))

    mu_m = layer.rho * layer.vs * layer.vs
    kSm2 = (omega / layer.vs) ** 2
    two_kz2_minus_kSm2 = 2.0 * kz * kz - kSm2

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    row: np.ndarray = np.zeros(7, dtype=complex)
    # A column: fluid carries no shear and doesn't reach r = b.
    row[0] = 0.0
    # B_I column (post-rescale; row*i flips +i*R to -R).
    row[1] = -2.0 * kz * mu_m * p_m * I1_pm_b
    # B_K column (post-rescale; row*i flips -i*R to +R).
    row[2] = +2.0 * kz * mu_m * p_m * K1_pm_b
    # C_I column (post-rescale; row*i AND col*-i, net factor 1).
    row[3] = +mu_m * two_kz2_minus_kSm2 * I1_sm_b
    # C_K column (post-rescale; same as C_I).
    row[4] = +mu_m * two_kz2_minus_kSm2 * K1_sm_b
    # B column (formation; subtracted, opposite sign of B_K at
    # layer=formation; K-flavour cancellation automatic).
    row[5] = -2.0 * kz * mu * p * K1_p_b
    # C column (post-rescale; subtracted, opposite sign of C_K at
    # layer=formation).
    row[6] = -mu * two_kz2_minus_kS2 * K1_s_b
    return row


# =====================================================================
# Substep F.1.b.4 -- assembly + public-API dispatch
# =====================================================================
#
# Stack the seven row builders (:func:`_layered_n0_row1_at_a` through
# :func:`_layered_n0_row7_at_b`) into the 7x7 modal matrix and return
# the determinant. Each row builder applies the substep-F.1.a.5
# row / column phase rescale internally, so the assembled matrix is
# real-valued in the bound regime; ``np.linalg.det`` returns the
# real determinant directly. The public-API dispatch in
# :func:`stoneley_dispersion_layered` replaces the previous
# ``NotImplementedError`` with a brentq loop driven by the bound-
# regime bracket extended to ``min(V_S, V_S_m, V_f)``.


def _modal_determinant_n0_layered(
    kz: float,
    omega: float,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    *,
    layer: BoreholeLayer,
) -> float:
    r"""
    7x7 axisymmetric modal determinant for a borehole with one
    annular layer between fluid and formation.

    Stacks the seven row builders from substeps F.1.b.2 and F.1.b.3
    into the 7x7 layered matrix, applies the substep-F.1.a.5 phase
    rescale (already absorbed into each row), and returns the
    determinant as a real scalar.

    Reduces to :func:`_modal_determinant_n0` (up to an irrelevant
    overall scale factor) when ``(layer.vp, layer.vs, layer.rho) =
    (vp, vs, rho)``: the layer=formation regression test in
    :func:`stoneley_dispersion_layered` is the floating-point
    oracle for this routine.

    Parameters
    ----------
    kz : float
        Trial axial wavenumber (rad / m). Must lie in the bound
        regime ``kz > omega / min(V_S, V_S_m, V_f)``; outside the
        regime the determinant is NaN (one or more radial wavenumbers
        go imaginary).
    omega, vp, vs, rho, vf, rho_f, a : float
        Formation, fluid, and borehole geometry parameters.
    layer : BoreholeLayer
        Annular layer between fluid and formation.

    Returns
    -------
    float
        ``det(M)`` of the 7x7 layered modal matrix, real-valued in
        the bound regime. NaN outside the bound regime.
    """
    rows = [
        _layered_n0_row1_at_a(
            kz,
            omega,
            vp=vp,
            vs=vs,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            a=a,
            layer=layer,
        ),
        _layered_n0_row2_at_a(
            kz,
            omega,
            vp=vp,
            vs=vs,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            a=a,
            layer=layer,
        ),
        _layered_n0_row3_at_a(
            kz,
            omega,
            vp=vp,
            vs=vs,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            a=a,
            layer=layer,
        ),
        _layered_n0_row4_at_b(
            kz,
            omega,
            vp=vp,
            vs=vs,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            a=a,
            layer=layer,
        ),
        _layered_n0_row5_at_b(
            kz,
            omega,
            vp=vp,
            vs=vs,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            a=a,
            layer=layer,
        ),
        _layered_n0_row6_at_b(
            kz,
            omega,
            vp=vp,
            vs=vs,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            a=a,
            layer=layer,
        ),
        _layered_n0_row7_at_b(
            kz,
            omega,
            vp=vp,
            vs=vs,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            a=a,
            layer=layer,
        ),
    ]
    M = np.vstack(rows)
    # Each row is real-valued post-rescale; the imaginary parts are
    # zero to floating-point precision in the bound regime. Take the
    # real part to discard sub-machine-epsilon imaginary noise.
    return float(np.linalg.det(M.real))


def _stoneley_kz_bracket_layered(
    omega: float,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    layer: BoreholeLayer,
) -> tuple[float, float]:
    """
    Bracket the layered n=0 Stoneley root in (k_z_lo, k_z_hi).

    Lower bound: just above the bound-regime floor
    ``omega / min(V_S, V_S_m, V_f)`` so all five radial wavenumbers
    are real positive. Upper bound: scaled from the closed-form
    Stoneley low-f estimate ``S_ST = sqrt(1/V_f^2 + rho_f / mu)``
    using the *formation* shear modulus (the layer's contribution
    to the low-f limit is a perturbation around the formation-only
    value; the bracket need not be tight, just enclose the root).
    """
    mu_formation = rho * vs * vs
    s_st_lf = np.sqrt(1.0 / vf**2 + rho_f / mu_formation)
    kz_lf_est = omega * s_st_lf
    # Bound-regime floor: the slowest body-wave kz floor across all
    # waves in the stack.
    slowest = min(vs, layer.vs, vf)
    kz_lo = omega / slowest * (1.0 + 1.0e-6)
    kz_hi = max(kz_lf_est * 1.5, kz_lo * 2.0)
    return kz_lo, kz_hi


def _stoneley_kz_bracket_cased(
    omega: float,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    layers: tuple[BoreholeLayer, ...],
) -> tuple[float, float]:
    """
    Bracket the cased-hole n=0 Stoneley root in (k_z_lo, k_z_hi)
    for an arbitrary-N layer stack.

    Generalisation of :func:`_stoneley_kz_bracket_layered` to
    N >= 1 layers. Lower bound is the slowest-body-wave floor
    across the entire stack: fluid, every layer, formation
    half-space. Upper bound is the same formation-mu-based LF
    estimate, with a 1.5x cushion -- the layers' contribution
    to the low-f limit is a perturbation that the brentq
    expansion-loop in :func:`stoneley_dispersion_layered`
    handles if needed.
    """
    mu_formation = rho * vs * vs
    s_st_lf = np.sqrt(1.0 / vf**2 + rho_f / mu_formation)
    kz_lf_est = omega * s_st_lf
    slowest = min(vs, vf, *(L.vs for L in layers))
    kz_lo = omega / slowest * (1.0 + 1.0e-6)
    kz_hi = max(kz_lf_est * 1.5, kz_lo * 2.0)
    return kz_lo, kz_hi


def stoneley_dispersion_layered(
    freq: np.ndarray,
    *,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    layers: tuple[BoreholeLayer, ...] = (),
) -> BoreholeMode:
    r"""
    Stoneley-wave phase slowness vs frequency for a borehole with
    optional mudcake / altered-zone annular layers between the
    fluid and the formation half-space.

    With ``layers=()`` (no extra layers) this is bit-equivalent to
    :func:`stoneley_dispersion`. With a single
    :class:`BoreholeLayer` it dispatches to the 7x7 layered modal
    determinant :func:`_modal_determinant_n0_layered` and brentq's
    its lowest root across the frequency grid -- the public-API
    payload of plan item F.

    Parameters
    ----------
    freq : ndarray
        Frequency grid (Hz). Must be strictly positive.
    vp, vs, rho : float
        Formation half-space P-wave velocity (m/s), S-wave
        velocity (m/s), and bulk density (kg/m^3). Must satisfy
        ``vp > vs > 0`` and ``rho > 0``.
    vf, rho_f : float
        Borehole-fluid velocity (m/s) and density (kg/m^3).
    a : float
        Borehole (fluid-side) radius (m).
    layers : tuple of BoreholeLayer, default ()
        Annular elastic layers between the fluid (``r < a``) and
        the formation half-space, ordered radially outward
        (``layers[0]`` adjacent to fluid; ``layers[-1]`` adjacent
        to formation). ``()`` dispatches to
        :func:`stoneley_dispersion`. A single-element tuple
        dispatches to the 7x7 layered solver
        (:func:`_modal_determinant_n0_layered`). Multi-layer
        stacks (``N >= 2``) dispatch to the Thomson-Haskell-style
        propagator-matrix path
        (:func:`_modal_determinant_n0_cased`) shipped in plan
        item G.

    Returns
    -------
    BoreholeMode
        ``name = "Stoneley"``, ``azimuthal_order = 0``. ``slowness``
        is ``NaN`` at any frequency where the bracket failed (rare
        in the bound regime).

    Raises
    ------
    ValueError
        If any input is non-positive, ``vp <= vs``, ``freq``
        contains a non-positive entry, or any layer is malformed.

    Notes
    -----
    **Subdividing a layer is exactly invariant.** Describing one
    homogeneous annulus as several adjacent layers with the same
    properties reproduces the dispersion to ~1e-15 relative, for
    every azimuthal order and for open-hole and cased stacks alike.
    That is a useful property to lean on -- it means a stack may be
    refined for convenience without perturbing the answer -- and
    ``tests/test_solver_branches.py`` pins it.

    **A redundant layer is not always transparent, though.** Appending
    a layer whose properties equal the formation is physically a no-op,
    and the solver treats it as one only while the radial dynamic range
    across that layer stays moderate. Beyond that -- somewhere above
    0.1 m at 100 kHz for a 2 cm mudcake -- the root search returns
    finite, plausible, wrong values. Neither the size of the error nor
    which spurious root comes back is stable; both move with thickness
    and across platforms, which is the signature of lost precision
    rather than of a different physical branch. The effect grows with
    the product of the radial decay constant and the layer thickness, so
    it is reached either by thick layers or by high frequencies.

    This matters for *verification* rather than for use: padding a stack
    with a redundant layer is a technique for checking the solver
    against its unlayered counterpart, not a configuration anyone
    models. Genuine layers with real contrast keep converging to the
    correct short-wavelength limit at every thickness tested, and fail
    to ``NaN`` rather than to a wrong number. Prefer thin redundant
    layers and moderate frequencies when using that technique, and
    treat a disagreement as the check expiring rather than as the
    solver being wrong until an independent oracle says otherwise.
    """
    layers_tuple = tuple(layers)
    _validate_borehole_layers(layers_tuple)
    if not layers_tuple:
        return stoneley_dispersion(
            freq,
            vp=vp,
            vs=vs,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            a=a,
        )
    if vp <= 0 or vs <= 0 or rho <= 0:
        raise ValueError("vp, vs, rho must all be positive")
    if vf <= 0 or rho_f <= 0:
        raise ValueError("vf and rho_f must be positive")
    if a <= 0:
        raise ValueError("a must be positive")
    if vp <= vs:
        raise ValueError("require vp > vs")
    f_arr = np.asarray(freq, dtype=float)
    if np.any(f_arr <= 0):
        raise ValueError("freq must be strictly positive")

    from fwap.cylindrical_solver import _modal_determinant_n0_cased

    n_layers = len(layers_tuple)
    slowness = np.full_like(f_arr, np.nan, dtype=float)
    for i, f in enumerate(f_arr):
        omega = 2.0 * np.pi * float(f)

        if n_layers == 1:
            layer = layers_tuple[0]

            def _det(kz_, omega=omega, layer=layer):
                return _modal_determinant_n0_layered(
                    kz_,
                    omega,
                    vp,
                    vs,
                    rho,
                    vf,
                    rho_f,
                    a,
                    layer=layer,
                )

            kz_lo, kz_hi = _stoneley_kz_bracket_layered(
                omega,
                vp,
                vs,
                rho,
                vf,
                rho_f,
                a,
                layer,
            )
        else:

            def _det(kz_, omega=omega, layers_tuple=layers_tuple):
                return _modal_determinant_n0_cased(
                    kz_,
                    omega,
                    vp=vp,
                    vs=vs,
                    rho=rho,
                    vf=vf,
                    rho_f=rho_f,
                    a=a,
                    layers=layers_tuple,
                )

            kz_lo, kz_hi = _stoneley_kz_bracket_cased(
                omega,
                vp,
                vs,
                rho,
                vf,
                rho_f,
                a,
                layers_tuple,
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
                    "stoneley_dispersion_layered: det evaluation NaN "
                    "at f=%.1f Hz (likely outside bound regime)",
                    f,
                )
                continue
            if np.sign(d_lo) == np.sign(d_hi):
                logger.debug(
                    "stoneley_dispersion_layered: failed to bracket at f=%.1f Hz",
                    f,
                )
                continue
            kz_root = optimize.brentq(_det, kz_lo, kz_hi, xtol=1.0e-10)
            slowness[i] = kz_root / omega
        except (ValueError, RuntimeError) as exc:
            logger.debug(
                "stoneley_dispersion_layered: brentq failed at f=%.1f Hz: %s",
                f,
                exc,
            )

    return BoreholeMode(
        name="Stoneley",
        azimuthal_order=0,
        freq=f_arr,
        slowness=slowness,
    )


# =====================================================================
# Plan item G.2.d -- public dispersion API for the fluid microannulus
# =====================================================================
#
# Wraps ``_modal_determinant_n0_microannulus`` (the 11x11 assembly)
# in the first public entry point for the debonded regime.
#
# The determinant carries *two* families of bound root, so the
# selection rule is the substance of this item rather than an
# afterthought -- a bracket that assumes a single root is exactly the
# n=0 defect that shipped once already. The rule used here is
# structural rather than tuned: the Stoneley-like mode is the *fastest*
# bound n=0 mode, sitting just below the borehole-fluid velocity, so
# scanning upward in ``k_z`` from the bound floor and taking the first
# sign change identifies it whatever the other family is doing.
#
# The second family -- the crack (Krauklis) wave guided by the gap
# itself -- is deliberately not returned here. See the function's Notes
# for the measurement that says why it needs its own entry point rather
# than a ``branch`` argument on this one.


def _microannulus_kz_window(
    omega: float,
    *,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    inner_layers: tuple[BoreholeLayer, ...],
    annulus: FluidAnnulus,
    outer_layers: tuple[BoreholeLayer, ...],
) -> tuple[float, float]:
    """
    Scan window ``(kz_lo, kz_hi)`` for the microannulus Stoneley root.

    Lower bound is the bound-regime floor: every radial wavenumber in the
    stack must be real, so ``k_z`` exceeds ``omega`` over the slowest body
    wave anywhere in it. The gap fluid contributes its *acoustic* velocity
    to that minimum, which is the whole reason this configuration is
    reachable while a compliant-solid stand-in is not.

    Upper bound is twice the tube-wave low-frequency estimate, matching
    :func:`_stoneley_kz_bracket_cased`. It is a scan limit rather than a
    bracket: the root is located by the first sign change above the floor,
    so the bound only has to be generous, and a slip in it costs a NaN
    rather than the wrong branch.
    """
    mu_formation = rho * vs * vs
    kz_lf_est = omega * float(np.sqrt(1.0 / vf**2 + rho_f / mu_formation))
    slowest = min(vs, vf, annulus.vf, *(L.vs for L in inner_layers + outer_layers))
    kz_lo = omega / slowest * (1.0 + 1.0e-9)
    kz_hi = max(kz_lf_est * 2.0, kz_lo * 2.0)
    return kz_lo, kz_hi


def stoneley_dispersion_microannulus(
    freq: np.ndarray,
    *,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    inner_layers: tuple[BoreholeLayer, ...],
    annulus: FluidAnnulus,
    outer_layers: tuple[BoreholeLayer, ...],
    samples: int = 400,
) -> BoreholeMode:
    r"""
    Stoneley dispersion for a cased stack split by a fluid microannulus.

    The debonded-regime counterpart of :func:`stoneley_dispersion_layered`:
    a stack of the form ``borehole fluid | casing | microannulus | cement |
    formation``, which is the standard model of casing debonding in
    cement-bond logging.

    Parameters
    ----------
    freq : ndarray, shape (n_f,)
        Frequency grid (Hz). Must be strictly positive.
    vp, vs, rho : float
        Formation half-space P-wave velocity (m/s), S-wave velocity (m/s)
        and bulk density (kg/m^3). Require ``vp > vs > 0`` and ``rho > 0``.
    vf, rho_f : float
        Borehole-fluid velocity (m/s) and density (kg/m^3).
    a : float
        Borehole (fluid-side) radius (m).
    inner_layers, outer_layers : tuple of BoreholeLayer
        Elastic layers inside and outside the gap, each ordered radially
        outward -- typically ``(casing,)`` and ``(cement,)``. Both must be
        non-empty. A zero-thickness inner block would make the two
        shear-traction-free conditions at ``r = a`` and at the gap's inner
        face the same equation and the determinant identically zero, so
        the degenerate case is refused rather than accommodated.
    annulus : FluidAnnulus
        The fluid gap between the two blocks.
    samples : int, default 400
        Scan resolution for the sign-change search. The window is narrow
        (the Stoneley root sits just below the fluid velocity), so this
        sits far above the density at which the root moves; see the
        resolution-independence test in
        ``tests/test_solver_branches.py``.

    Returns
    -------
    BoreholeMode
        ``name = "Stoneley"``, ``azimuthal_order = 0``,
        ``attenuation_per_meter = None`` -- the mode is bound, so its
        attenuation is zero by construction. ``slowness`` is ``NaN`` at
        any frequency where no bound root was found.

    Raises
    ------
    ValueError
        On non-positive velocities, densities, radius or frequencies; on
        ``vp <= vs``; on an empty ``inner_layers`` or ``outer_layers``; or
        on an invalid ``annulus``.

    Notes
    -----
    **Two root families, and why only one is returned.** The underlying
    determinant has a second bound root: the crack (Krauklis) wave guided
    by the gap itself, measured at 68-620 m/s over four decades of gap
    thickness. It is not exposed through a ``branch`` argument on this
    function, and the reason is a measurement rather than a preference.
    Over 270 sampled configurations the bound window held exactly two
    roots in 269 of them; the exception produced a *duplicated* pair near
    4 m/s, which is the signature of the lost-precision spurious roots
    this module has shipped before. Separating the genuine crack wave
    from those needs a filter this function does not have, so the crack
    wave gets its own entry point once it does. Selecting the Stoneley
    family is safe without one, because it is the fastest bound mode and
    is found by the first sign change above the floor.

    **Grid independence.** Each frequency is solved on its own, by
    scanning for a sign change and polishing with
    :func:`scipy.optimize.brentq`; there is no frequency marching, so the
    result does not depend on the frequency grid. That is the property
    whose absence caused the ``n=0`` branch-selection defect.

    **The gap does not close smoothly onto the bonded stack.** Letting
    ``annulus.thickness -> 0`` converges to a frictionless *slip*
    interface, not to :func:`stoneley_dispersion_layered` on the same
    layers: shear traction stays zero on both faces and ``u_z`` stays
    free however thin the gap. Measured at 8 kHz on a 1 cm casing and 3 cm
    cement, this function converges as ``O(h)`` to 1383.45 m/s against
    1400.04 m/s bonded -- a 1.2 % offset that does not close. Do not read
    a thin gap as a validation of the bonded solver.

    **Validity.** The underlying assembly is validated against the
    Krauklis crack-wave closed form, which fixes an absolute velocity
    with no Bessel functions and no cylindrical geometry in it, to 0.02 %
    at a 1 um gap. See :func:`_modal_determinant_n0_microannulus`.

    See Also
    --------
    stoneley_dispersion_layered : The bonded counterpart.
    FluidAnnulus : The gap description this takes.

    Examples
    --------
    >>> import numpy as np
    >>> from fwap import BoreholeLayer, FluidAnnulus
    >>> from fwap import stoneley_dispersion_microannulus
    >>> casing = BoreholeLayer(vp=5900.0, vs=3200.0, rho=7800.0, thickness=0.01)
    >>> cement = BoreholeLayer(vp=2800.0, vs=1600.0, rho=1900.0, thickness=0.03)
    >>> mode = stoneley_dispersion_microannulus(
    ...     np.array([8.0e3]),
    ...     vp=4000.0, vs=2300.0, rho=2500.0, vf=1500.0, rho_f=1000.0, a=0.10,
    ...     inner_layers=(casing,),
    ...     annulus=FluidAnnulus(vf=1500.0, rho=1000.0, thickness=1.0e-4),
    ...     outer_layers=(cement,),
    ... )
    >>> bool(1370.0 < 1.0 / mode.slowness[0] < 1395.0)
    True
    """
    inner = tuple(inner_layers)
    outer = tuple(outer_layers)
    _validate_borehole_layers(inner)
    _validate_borehole_layers(outer)
    _validate_fluid_annulus(annulus)
    if not inner or not outer:
        raise ValueError("inner_layers and outer_layers must both be non-empty")
    if vp <= 0 or vs <= 0 or rho <= 0:
        raise ValueError("vp, vs, rho must all be positive")
    if vf <= 0 or rho_f <= 0:
        raise ValueError("vf and rho_f must be positive")
    if a <= 0:
        raise ValueError("a must be positive")
    if vp <= vs:
        raise ValueError("require vp > vs")
    if samples < 2:
        raise ValueError("samples must be at least 2")

    f_arr = np.asarray(freq, dtype=float)
    if np.any(f_arr <= 0):
        raise ValueError("freq must be strictly positive")

    from fwap.cylindrical_solver import _modal_determinant_n0_microannulus

    slowness = np.full(f_arr.shape, np.nan, dtype=float)
    for i, f in enumerate(f_arr.ravel()):
        omega = 2.0 * np.pi * float(f)

        def _det(kz_: float, omega: float = omega) -> float:
            return _modal_determinant_n0_microannulus(
                kz_,
                omega,
                vp=vp,
                vs=vs,
                rho=rho,
                vf=vf,
                rho_f=rho_f,
                a=a,
                inner_layers=inner,
                annulus_vf=annulus.vf,
                annulus_rho=annulus.rho,
                annulus_thickness=annulus.thickness,
                outer_layers=outer,
            )

        kz_lo, kz_hi = _microannulus_kz_window(
            omega,
            vs=vs,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            inner_layers=inner,
            annulus=annulus,
            outer_layers=outer,
        )
        # Walk upward from the floor and stop at the first sign change: the
        # Stoneley-like mode is the fastest bound root, so the first one
        # found is the right one regardless of what lies further up.
        grid = np.linspace(kz_lo, kz_hi, samples)
        values = np.array([_det(float(k)) for k in grid])
        for j in range(grid.size - 1):
            if not (np.isfinite(values[j]) and np.isfinite(values[j + 1])):
                continue
            if np.sign(values[j]) != np.sign(values[j + 1]):
                try:
                    root = optimize.brentq(
                        _det, grid[j], grid[j + 1], xtol=1.0e-14, rtol=8.9e-16
                    )
                except (ValueError, RuntimeError) as exc:  # pragma: no cover
                    logger.debug(
                        "stoneley_dispersion_microannulus: brentq failed "
                        "at f=%.1f Hz: %s",
                        f,
                        exc,
                    )
                    break
                slowness.ravel()[i] = root / omega
                break

    return BoreholeMode(
        name="Stoneley",
        azimuthal_order=0,
        freq=f_arr,
        slowness=slowness,
        attenuation_per_meter=None,
    )


def _microannulus_stable_roots(
    det_of_velocity,
    c_lo: float,
    c_hi: float,
    samples: int,
) -> list[float]:
    """
    Phase-velocity roots that survive a change of scan grid, fastest first.

    The spurious-root filter the crack-wave API needs. Scanning this
    determinant down to the low velocities the gap mode occupies eventually
    reaches phase velocities where the elastic propagators have lost all
    precision, and sign changes there are read as roots -- the defect this
    module has shipped twice. They are not stable under a change of grid,
    and the genuine roots are, so running two scans and keeping the
    intersection separates them.

    Measured on the one configuration in 270 sampled that produced them: a
    duplicated pair near 4 m/s appeared in one grid of six, while the
    Stoneley and crack roots appeared in all six and agreed to 1e-9.

    The two grids differ in resolution and in their lower endpoint, which is
    what breaks any coincidence between them. The upper endpoint is left
    alone deliberately -- it sits just below the bound-regime limit, and
    moving it inward risks stepping over a Stoneley root that lies close to
    it, which would silently renumber everything below.

    Parameters
    ----------
    det_of_velocity : callable
        Real-valued modal determinant as a function of phase velocity (m/s).
    c_lo, c_hi : float
        Scan window (m/s), searched geometrically because the two families
        are decades apart.
    samples : int
        Resolution of the first grid; the second uses ``1.37 x`` as many.

    Returns
    -------
    list of float
        Phase velocities (m/s) present in both scans, largest first.
    """

    def scan(grid: np.ndarray) -> list[float]:
        values = np.array([det_of_velocity(float(c)) for c in grid])
        out: list[float] = []
        for i in range(grid.size - 1):
            if not (np.isfinite(values[i]) and np.isfinite(values[i + 1])):
                continue
            if np.sign(values[i]) != np.sign(values[i + 1]):
                try:
                    out.append(
                        float(
                            optimize.brentq(
                                det_of_velocity,
                                grid[i],
                                grid[i + 1],
                                xtol=1.0e-13,
                                rtol=8.9e-16,
                            )
                        )
                    )
                except (ValueError, RuntimeError):  # pragma: no cover
                    continue
        return out

    first = scan(np.geomspace(c_lo, c_hi, samples))
    second = scan(np.geomspace(c_lo * 1.013, c_hi, int(samples * 1.37) + 1))
    stable = [
        root
        for root in first
        if any(abs(root / other - 1.0) < 1.0e-6 for other in second)
    ]
    return sorted(stable, reverse=True)


def crack_wave_dispersion(
    freq: np.ndarray,
    *,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    inner_layers: tuple[BoreholeLayer, ...],
    annulus: FluidAnnulus,
    outer_layers: tuple[BoreholeLayer, ...],
    samples: int = 250,
) -> BoreholeMode:
    r"""
    Crack-wave (Krauklis) dispersion for a cased stack with a fluid gap.

    The second bound root of the microannulus determinant, and the more
    sensitive of the two debonding indicators. Where the Stoneley-like mode
    shifts about 1 % on debonding and then barely moves with gap thickness,
    the crack wave is *guided by the gap* and scales as ``(f h)^{1/3}``: at
    8 kHz it runs from 68 m/s at a 1 um gap to 620 m/s at 1 mm. Because that
    closed form is invertible, a measured crack-wave velocity gives a gap
    thickness directly.

    Parameters
    ----------
    freq : ndarray, shape (n_f,)
        Frequency grid (Hz). Must be strictly positive.
    vp, vs, rho : float
        Formation half-space velocities (m/s) and density (kg/m^3).
    vf, rho_f : float
        Borehole-fluid velocity (m/s) and density (kg/m^3).
    a : float
        Borehole (fluid-side) radius (m).
    inner_layers, outer_layers : tuple of BoreholeLayer
        Elastic layers inside and outside the gap, ordered radially outward.
        Both must be non-empty.
    annulus : FluidAnnulus
        The fluid gap the wave is guided by.
    samples : int, default 250
        Resolution of the first of the two scan grids. The window spans
        decades, so it is searched geometrically. The result is unchanged
        from 60 samples upward (to 1e-12), and the spurious-root filter
        below holds at every resolution tried, so this is a cost setting
        rather than a tuning parameter.

    Returns
    -------
    BoreholeMode
        ``name = "crack_wave"``, ``azimuthal_order = 0``,
        ``attenuation_per_meter = None``. ``slowness`` is ``NaN`` wherever no
        second bound root survives the filter described below.

    Raises
    ------
    ValueError
        Same conditions as :func:`stoneley_dispersion_microannulus`.

    Notes
    -----
    **Selection.** The crack wave is the second-fastest bound n=0 root: the
    Stoneley-like mode sits just below the borehole-fluid velocity and this
    one is the next below it. Selecting by order rather than by proximity to
    an expected value keeps the closed form available as an independent
    check instead of building it into the search.

    **Spurious roots, and why this function needs a filter where
    :func:`stoneley_dispersion_microannulus` does not.** That function stops
    at the first sign change above the bound floor and never reaches the low
    phase velocities where the elastic propagators lose precision. This one
    scans down to them, and sign changes there get read as roots -- a defect
    this module has shipped twice. Over 270 sampled configurations one
    produced a duplicated pair near 4 m/s. They are not stable under a
    change of grid, so the scan is run twice at different resolutions and
    lower endpoints and only roots common to both are kept; in that case the
    spurious pair appeared in one grid of six while the genuine roots
    appeared in all six.

    The obvious alternative filter does **not** work and was measured
    before being discarded: the elastic propagator's determinant identity
    ``det P = (r_inner/r_outer)^2`` is violated by 1e232 at operating points
    where the crack root is nonetheless correct to 1e-9, because the mode is
    confined within ``~1/k_z`` of the gap and the error lives in a growing
    branch the root condition never sees.

    **Validity.** The solver reproduces the analytic crack-wave speed

    .. math::

        c = \left(\frac{\omega h}{C \rho_f}\right)^{1/3},
        \qquad C = \sum_{\text{walls}} \frac{1 - \nu}{\mu}

    to 0.02 % at a 1 um gap, departing as ``k h`` stops being small. That
    formula assumes half-space walls, so it holds only while both blocks are
    thicker than the mode's decay length ``~1/k_z``; a 2 mm casing against a
    6 mm decay length gives 0.64 of it.

    See Also
    --------
    stoneley_dispersion_microannulus : The other root family of the same
        determinant.

    Examples
    --------
    >>> import numpy as np
    >>> from fwap import BoreholeLayer, FluidAnnulus, crack_wave_dispersion
    >>> casing = BoreholeLayer(vp=5900.0, vs=3200.0, rho=7800.0, thickness=0.05)
    >>> cement = BoreholeLayer(vp=2800.0, vs=1600.0, rho=1900.0, thickness=0.20)
    >>> mode = crack_wave_dispersion(
    ...     np.array([8.0e3]),
    ...     vp=4000.0, vs=2300.0, rho=2500.0, vf=1500.0, rho_f=1000.0, a=0.10,
    ...     inner_layers=(casing,),
    ...     annulus=FluidAnnulus(vf=1500.0, rho=1000.0, thickness=1.0e-4),
    ...     outer_layers=(cement,),
    ... )
    >>> bool(290.0 < 1.0 / mode.slowness[0] < 320.0)
    True
    """
    inner = tuple(inner_layers)
    outer = tuple(outer_layers)
    _validate_borehole_layers(inner)
    _validate_borehole_layers(outer)
    _validate_fluid_annulus(annulus)
    if not inner or not outer:
        raise ValueError("inner_layers and outer_layers must both be non-empty")
    if vp <= 0 or vs <= 0 or rho <= 0:
        raise ValueError("vp, vs, rho must all be positive")
    if vf <= 0 or rho_f <= 0:
        raise ValueError("vf and rho_f must be positive")
    if a <= 0:
        raise ValueError("a must be positive")
    if vp <= vs:
        raise ValueError("require vp > vs")
    if samples < 2:
        raise ValueError("samples must be at least 2")

    f_arr = np.asarray(freq, dtype=float)
    if np.any(f_arr <= 0):
        raise ValueError("freq must be strictly positive")

    from fwap.cylindrical_solver import (
        _BESSEL_ARG_MAX,
        _modal_determinant_n0_microannulus,
    )

    r_outermost = (
        a
        + sum(L.thickness for L in inner)
        + annulus.thickness
        + sum(L.thickness for L in outer)
    )
    slowest = min(vs, vf, annulus.vf, *(L.vs for L in inner + outer))

    slowness = np.full(f_arr.shape, np.nan, dtype=float)
    for i, f in enumerate(f_arr.ravel()):
        omega = 2.0 * np.pi * float(f)

        def _det(phase_velocity: float, omega: float = omega) -> float:
            return _modal_determinant_n0_microannulus(
                omega / phase_velocity,
                omega,
                vp=vp,
                vs=vs,
                rho=rho,
                vf=vf,
                rho_f=rho_f,
                a=a,
                inner_layers=inner,
                annulus_vf=annulus.vf,
                annulus_rho=annulus.rho,
                annulus_thickness=annulus.thickness,
                outer_layers=outer,
            )

        # Bottom of the window is where the determinant stops being
        # representable at all, not a velocity chosen to suit the answer: below
        # ``omega * r_outermost / _BESSEL_ARG_MAX`` the assembly returns NaN.
        c_lo = omega * r_outermost / _BESSEL_ARG_MAX * 1.001
        c_hi = slowest * (1.0 - 1.0e-9)
        if c_lo >= c_hi:
            continue
        roots = _microannulus_stable_roots(_det, c_lo, c_hi, samples)
        if len(roots) >= 2:
            slowness.ravel()[i] = 1.0 / roots[1]

    return BoreholeMode(
        name="crack_wave",
        azimuthal_order=0,
        freq=f_arr,
        slowness=slowness,
        attenuation_per_meter=None,
    )
