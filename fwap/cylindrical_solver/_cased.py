"""Cased-hole multi-layer propagator-matrix modal determinants.

Provides the per-layer mode-amplitude-to-state-vector matrices,
the per-layer propagators, and the stacked modal determinants for
azimuthal orders n=0 (Stoneley), n=1 (flexural), and n=2 (quadrupole).
The leaky / complex-``k_z`` n=2 path is included.

Extracted from ``fwap.cylindrical_solver.__init__`` as part of the
Phase 1 package split. The original module-level docstring with the
full physics derivation lives in the package ``__init__``; refer
there for the field ansatz, sign conventions, and references.
"""

from __future__ import annotations

import numpy as np
from scipy import special

from fwap.cylindrical_solver._bessel import _i0_i1, _k0_k1, _k_or_hankel
from fwap.cylindrical_solver._dataclasses import BoreholeLayer

# =====================================================================
# Substep G.a -- math scaffolding for the cased-hole multi-layer
#                propagator-matrix determinant (n=0)
# =====================================================================
#
# Generalises the F.1 single-extra-layer 7x7 hand-coded form to N
# stacked annular layers via a Thomson-Haskell-style propagator
# matrix in cylindrical geometry (Schmitt-Bouchon 1985; Tang &
# Cheng 2004 ch. 7). Comments-only block; the helpers themselves
# land in plan items G.b (per-layer propagator), G.c (assembly),
# and G.d (public-API hook). See ``docs/plans/cylindrical_biot_G.md``
# for the full sub-plan.
#
# Sign / Bessel / phase conventions are the same as F.1.a (which
# this scaffolding extends), unchanged from the isotropic solver
# upstream:
# * Time dependence ``e^{-i omega t}``, axial ``e^{i k_z z}``,
#   azimuthal ``e^{i n theta}``.
# * Bound regime ``k_z > omega / V_min`` so all radial wavenumbers
#   ``F = sqrt(k_z^2 - (omega/V_f)^2)``,
#   ``p = sqrt(k_z^2 - (omega/V_P)^2)``,
#   ``s = sqrt(k_z^2 - (omega/V_S)^2)`` are real and positive.
# * Per-region scalar / vector potentials: in each elastic layer
#   ``phi = B_I I_0(p r) + B_K K_0(p r)`` (P scalar potential)
#   and ``psi_theta = C_I I_1(s r) + C_K K_1(s r)`` (SV vector
#   potential, theta-component for axisymmetric n=0). Two
#   amplitudes per wave family because the layer is bounded on
#   both sides; the formation half-space keeps only the
#   ``K``-flavour (decay outward) and the fluid keeps only the
#   ``I``-flavour (regular at the axis) -- same as F.1.a.
#
# The N=0 limit reduces to the unlayered ``_modal_determinant_n0``
# ; the N=1 limit reduces to ``_modal_determinant_n0_layered``
# (within an overall scale factor; same brentq root in k_z). Both
# limits anchor the floating-point regression oracles in G.c.
#
# -----
# Substep G.a.1 -- State vector at the axisymmetric order n=0
# -----
#
# Within any uniform elastic layer (or the formation half-space),
# the radial-dependent part of the displacement / stress field at
# n=0 reduces to a 4-component state vector ``v(r)``:
#
#       v(r) = ( u_r(r),  u_z(r),  sigma_rr(r),  sigma_rz(r) )^T
#
# Four components (not six) because the n=0 axisymmetric problem
# has no theta-dependent quantities: ``u_theta = 0`` identically
# from the axisymmetric ansatz, ``sigma_r_theta = 0`` from
# ``epsilon_r_theta = 0``, and the remaining stresses
# ``sigma_theta_theta`` and ``sigma_zz`` are determined by
# ``u_r, u_z`` via Hooke. The four components above are exactly
# the four whose continuity is required at every radial interface.
#
# At the layer/layer (or fluid/layer, or layer/formation) boundary
# ``r = b_j``, the n=0 boundary conditions are
# 1. ``u_r`` continuous,
# 2. ``u_z`` continuous (slip-free across solid/solid interface),
# 3. ``sigma_rr`` continuous,
# 4. ``sigma_rz`` continuous,
# i.e. ``v(b_j^-) = v(b_j^+)`` -- a single 4-vector identity.
#
# The fluid/solid interface at ``r = a`` is the only place where
# the four components do NOT match one-for-one: in the fluid
# ``u_z`` is unconstrained (slip allowed), so only three of the
# four solid-side components match the fluid-side counterparts.
# That mismatch is what reduces the final determinant below the
# 4N+? row count one would naively expect from N layers. See
# G.a.4 for the bookkeeping.
#
# -----
# Substep G.a.2 -- Mode-amplitude to state-vector matrix E(r)
# -----
#
# Within a single uniform layer (or the formation half-space),
# ``v(r)`` is a linear combination of the four wave amplitudes
# ``c = (B_I, B_K, C_I, C_K)^T`` collecting the I/K-flavours of
# the P scalar potential and the SV vector potential:
#
#       v(r) = E(r) c
#
# The 4x4 matrix ``E(r)`` is obtained by transcribing the
# substep-F.1.a.2 (displacement) and F.1.a.3 (stress) field forms
# at radius r, with each column corresponding to one wave
# amplitude:
#
#                | (B_I-col)  (B_K-col)  (C_I-col)  (C_K-col) |
#       E(r) =   |     u_r entries from each amplitude         |
#                |     u_z entries                              |
#                |    sigma_rr entries                          |
#                |    sigma_rz entries                          |
#
# Using the standard modified-Bessel recurrences
# ``I_0'(x) = I_1(x)``, ``K_0'(x) = -K_1(x)``,
# ``(1/r) (d/dr)[r I_1(s r)] = s I_0(s r)``, etc., and the F.1.a
# Lame reduction ``-lambda k_P^2 + 2 mu p^2 = mu (2 k_z^2 - k_S^2)``,
# the four rows are (pre-rescale; phase rescaling follows the
# F.1.a.5 row * i, col * -i convention so the assembled
# determinant is real):
#
# Row 1 (u_r):
#       [ +p I_1(p r),  -p K_1(p r),  -i k_z I_1(s r),  -i k_z K_1(s r) ]
#
# Row 2 (u_z):
#       [ +i k_z I_0(p r),  +i k_z K_0(p r),  +s I_0(s r),  -s K_0(s r) ]
#
# Row 3 (sigma_rr):
#       [ -lambda k_P^2 I_0(p r) + 2 mu p^2 I_0(p r) - 2 mu p I_1(p r)/r,
#         -lambda k_P^2 K_0(p r) + 2 mu p^2 K_0(p r) + 2 mu p K_1(p r)/r,
#         -2 i mu k_z s I_0(s r) + 2 i mu k_z I_1(s r)/r,
#         +2 i mu k_z s K_0(s r) + 2 i mu k_z K_1(s r)/r ]
#
# Row 4 (sigma_rz):
#       [ +2 i mu k_z p I_1(p r),
#         -2 i mu k_z p K_1(p r),
#         +mu (2 k_z^2 - k_S^2) I_1(s r),
#         +mu (2 k_z^2 - k_S^2) K_1(s r) ]
#
# (Erratum: an earlier draft of this block had ``-mu (2 k_z^2 -
# k_S^2)`` for the C_I / C_K columns. The correct sign is ``+``,
# matching ``_layered_n0_row3_at_a`` post-rescale and the
# direct ``sigma_rz = mu (d_z u_r + d_r u_z)`` derivation:
# ``d_z u_r contribution from psi_theta = +k_z^2 psi_theta``
# and ``d_r u_z contribution from psi_theta = +s^2 psi_theta``,
# summing to ``+(k_z^2 + s^2) psi_theta = +(2 k_z^2 - k_S^2) psi_theta``
# after the ``s^2 = k_z^2 - k_S^2`` reduction.)
#
# (The signs above pre-rescale; absorb the row * i / col * -i
# rescale at the assembly step in G.c so each row is real-valued
# in the bound regime. See substep G.a.5.)
#
# Note: row 3's first two entries reduce, via the Lame identity,
# to ``+/- mu (2 k_z^2 - k_S^2) I_0/K_0(p r) +/- 2 mu p K_1(p r)/r``
# -- the same combination that appears in F.1's row 5 (sigma_rr
# at r = b). Direct transcription from F.1.b with the I-flavour
# kept alongside the K-flavour.
#
# -----
# Substep G.a.3 -- Layer propagator P_j(r_outer | r_inner)
# -----
#
# Within layer j with mode amplitudes c_j (constant across the
# annulus), the state vector at the inner and outer radii are
#       v_j(r_inner_j) = E_j(r_inner_j) c_j
#       v_j(r_outer_j) = E_j(r_outer_j) c_j
# so the 4-vector at the outer radius is related to the 4-vector
# at the inner radius by the propagator
#
#       P_j(r_outer_j | r_inner_j) = E_j(r_outer_j) E_j(r_inner_j)^{-1}
#
# i.e. ``v_j(r_outer_j) = P_j v_j(r_inner_j)``.
#
# Continuity of the 4-vector across each layer/layer interface
# (substep G.a.1) lets the per-layer propagators be composed
# directly:
#
#       v(r_N_outer) = P_N P_{N-1} ... P_2 P_1  v(r_0_inner)
#
# where ``r_0_inner = a`` (borehole wall, fluid side) and
# ``r_N_outer = b = a + sum(thickness_j)`` (formation half-space
# inner boundary). The composed propagator ``P_total`` is the
# only 4x4 quantity the assembly step in G.c needs to evaluate;
# the per-layer propagators do not need to be retained
# separately.
#
# Note: in cylindrical geometry P_j is NOT a matrix exponential
# of a constant-coefficient matrix (unlike Thomson-Haskell for
# plane-layered media). The radial ODE for ``v(r)`` has Bessel
# coefficients depending on r, so the propagator is genuinely
# the product ``E(r_outer) E(r_inner)^{-1}`` of two 4x4 Bessel
# matrices. The "matrix exponential" intuition transfers
# qualitatively (composition law, identity at r_outer = r_inner)
# but the explicit form differs.
#
# -----
# Substep G.a.4 -- Boundary conditions and modal determinant
# -----
#
# Three sets of BCs assemble the dispersion relation:
#
# (i) Fluid side at r = a. The fluid holds 1 amplitude
# ``A`` from ``P^(f) = A I_0(F r)``, giving ``u_r^(f)(a) = (F I_1(F a))/(rho_f omega^2)``,
# ``sigma_rr^(f)(a) = -A I_0(F a)``, and ``sigma_rz^(f)(a) = 0``
# (inviscid). The three solid-side BCs at r = a are
#       BC1: u_r^(f)(a) - u_r^(s)(a) = 0
#       BC2: sigma_rr^(s)(a) + A I_0(F a) = 0
#       BC3: sigma_rz^(s)(a) = 0
# Encoded as a 3x4 row block ``B_fluid_at_a`` extracting the
# three components ``(u_r, sigma_rr - <fluid contribution>,
# sigma_rz)`` of the layer-1 state vector ``v_1(a)``, plus a 3x1
# column for the fluid amplitude ``A``.
#
# (ii) Formation half-space at r = b. Only the K-flavour
# survives (decay outward): two amplitudes ``B_form_K``,
# ``C_form_K``. Continuity of all four components of ``v`` at
# r = b sets up four equations involving ``v_N(b)``, ``B_form_K``,
# and ``C_form_K``:
#       v_N(b) = E_form(b) (B_form_K, C_form_K, 0, 0)^T
#                                          [I-flavours dropped]
# Encoded as a 4x4 row block ``B_outer_at_b`` extracting all four
# components against the half-space form.
#
# (iii) Layer/layer continuity at r = b_j for j=1..N-1: handled
# implicitly by the propagator chain P_total = P_N ... P_1 (G.a.3).
#
# Assembled modal determinant. Stack the BCs into a square
# system. The unknowns are
#       (A, B_form_K, C_form_K)^T              (three amplitudes)
# and the equations are the three fluid-side BCs at r = a (using
# ``v_1(a) = E_1(a) c_1``) and the four formation-side BCs at
# r = b (using ``v_N(b) = P_total E_1(a) c_1``), composed so the
# layer-internal amplitudes ``c_j`` are eliminated via the
# propagator chain. Concretely:
#
#       M = [[ B_fluid_at_a,                 col_A ],
#            [ B_outer_at_b @ P_total,       col_form ]]
#
# where ``col_A`` is the 3-vector encoding the fluid amplitude's
# contribution to BC1-3 and ``col_form`` is the 4x2 block
# encoding the half-space K-flavour amplitudes' contribution to
# the four r=b continuity rows, evaluated against E_form(b).
# Net dimensions: ``M`` is (3 + 4) x (1 + 2 + 4) = 7 x 7 (where
# the +4 columns are the four layer-1-amplitude unknowns
# ``c_1``). Wait -- that's the un-reduced count.
#
# Reduction. The four ``c_1`` columns are eliminated by absorbing
# E_1(a) into the rows: pre-multiply the bottom block by
# ``B_outer_at_b @ P_total @ E_1(a)``. The resulting 7x7 system
# has unknowns ``(A, B_form_K, C_form_K, c_1_cols)`` and seven
# rows (three at r=a, four at r=b).
#
# **Key reduction at single-layer collapse (N=1)**: P_total reduces
# to ``E_1(b) E_1(a)^{-1}``, so ``B_outer_at_b @ P_total @ E_1(a) =
# B_outer_at_b @ E_1(b)``, i.e. the assembly drops to the same
# 7x7 form as F.1's hand-coded ``_modal_determinant_n0_layered``.
# This is the floating-point oracle for G.c (substep G.a.6).
#
# **Multi-layer (N >= 2)**: P_total = P_N ... P_1 has the same 4x4
# shape; the assembly stays 7x7. The propagator chain is the
# only N-dependent piece. Plan item G.c handles the explicit
# row/column packing.
#
# -----
# Substep G.a.5 -- Numerical conditioning (kz * thickness)
# -----
#
# The propagator P_j = E_j(r_outer) E_j(r_inner)^{-1} requires
# inverting E_j at the inner boundary. Within a layer of
# thickness ``h`` and radial wavenumber ``kappa`` (whichever of
# F, p, s is active), the modified-Bessel matrix entries scale
# as
#       I_0(kappa r), I_1(kappa r) ~ exp(+kappa r)/sqrt(2 pi kappa r)
#       K_0(kappa r), K_1(kappa r) ~ sqrt(pi/(2 kappa r)) exp(-kappa r)
# so the four columns of E(r) span a range of ~ exp(+/- kappa r)
# at large argument. Across a layer of thickness h, the
# **conditioning ratio** of E(r_inner) is roughly
# ``cond(E) ~ exp(2 kappa_max h)`` with kappa_max the largest of
# (F, p, s) in the layer.
#
# **Typical cased-hole geometries.**
# * Borehole radius a ~ 0.1 m, casing thickness ~ 0.01 m, cement
#   thickness ~ 0.05 m.
# * Sonic band 1-15 kHz: omega/V_S ~ 2-30 / m for typical V_S.
# * kz at the Stoneley root sits between 1/V_f and 1/V_S in
#   slowness, i.e. omega * (1/1500 to 1/3000) ~ 4 to 60 / m.
# * kappa h ~ 60 / m * 0.05 m = 3 (top of band, thick cement).
# * Conditioning ratio ~ exp(6) ~ 400 -- well within double-
#   precision tolerance.
#
# **Heavily-anisotropic / very-thick / very-high-frequency cases.**
# When ``kappa h >> 30`` the I-flavour columns overflow and the
# K-flavour underflow; ``inv(E_inner)`` becomes useless. Standard
# remedies (deferred to a follow-up plan):
# * **Knopoff delta-matrix** (Knopoff 1964): composes 6x6 minor
#   determinants instead of full propagators; avoids the
#   inversion step.
# * **Kennett reflection / transmission**: tracks per-layer
#   reflection coefficients instead of propagators; numerically
#   stable for arbitrarily thick stacks.
# Out of scope for G's first pass; the N >= 30 conditioning
# warning hook is placed in ``_validate_borehole_layers_stacked``
# (G.0) for later G.c expansion.
#
# -----
# Substep G.a.6 -- Single-layer collapse identity (G.b -> F.1)
# -----
#
# When ``len(layers) == 1`` the propagator path's modal
# determinant must agree with F.1's hand-coded
# ``_modal_determinant_n0_layered`` to a non-trivial overall
# scale factor (different intermediate factorisations: F.1's
# 7x7 hand-coded form folds the propagator implicitly through
# the per-row builders, while G.c assembles the propagator
# explicitly).
#
# Concretely:
#       det(M_propagator)  =  scale * det(M_F1)
# with ``scale`` independent of k_z, omega, layer / formation
# parameters (it cancels powers of ``det(E_1(a))`` between
# the two assemblies). The brentq root in k_z is the same.
# Pin ``scale`` empirically at the test: at a representative
# ``(k_z, omega)``, compute the ratio ``det(M_propagator) /
# det(M_F1)`` once and verify it stays constant across a small
# k_z grid.
#
# Same scaling identity holds at ``len(layers) == 0`` (collapse
# to ``_modal_determinant_n0`` via the formation-half-space-only
# assembly).
#
# -----
# Substep G.a.7 -- Self-check protocol
# -----
#
# Each implementation step lands with one or more of the
# following oracles (test names mirror the H.a.7 / F.1.a.6
# protocols):
#
# (a) Identity propagator: r_outer = r_inner -> P = eye(4) to
#     floating-point precision. Tests E(r)^{-1} E(r) round-trip.
# (b) Composition: P(r3 | r1) = P(r3 | r2) @ P(r2 | r1) for any
#     intermediate r2. Tests the propagator group law.
# (c) State-vector continuity: pick c, verify that
#     v(r2) = P(r2 | r1) v(r1) matches E(r2) c directly.
# (d) Zero-layer collapse (G.c): the propagator-matrix
#     determinant matches ``_modal_determinant_n0`` (within an
#     overall k_z-independent scale).
# (e) Single-layer collapse (G.c): matches
#     ``_modal_determinant_n0_layered`` (same overall-scale
#     identity).
# (f) Order-matters at N=2 (G.c / G.d): swapping (L_a, L_b) ->
#     (L_b, L_a) gives a different slowness curve.
# (g) Cement-bond regression (G.d): well-bonded vs free-pipe
#     synthetic produces distinct Stoneley dispersion curves
#     (Tang & Cheng 2004 fig 7.1).


# =====================================================================
# Plan item G.b.1 -- mode-amplitude-to-state-vector matrix E(r)
# =====================================================================
#
# 4x4 matrix mapping a uniform layer's wave amplitudes
# ``c = (B_I, B_K, C_I, C_K)`` to the four-component state vector
# ``v(r) = (u_r, u_z, sigma_rr, sigma_rz)`` at radius r. Direct
# transcription of substep G.a.2 with the substep-F.1.a.5 phase
# rescale (row * i for u_z and sigma_rz; col * -i for the C
# columns) absorbed so every entry is real-valued in the bound
# regime.
#
# The 16 entries below are the post-rescale forms; per-element
# oracles in G.b.1 tests verify them against the layer-amplitude
# columns of the existing F.1.b row builders.


def _layer_e_matrix_n0(
    kz: float,
    omega: float,
    *,
    vp: float,
    vs: float,
    rho: float,
    r: float,
) -> np.ndarray:
    r"""
    4x4 mode-amplitude-to-state-vector matrix ``E(r)`` for a
    uniform isotropic-elastic layer at azimuthal order n=0.

    Maps the four wave amplitudes
    ``c = (B_I, B_K, C_I, C_K)`` to the state vector
    ``v(r) = (u_r, u_z, sigma_rr, sigma_rz)`` via ``v(r) = E(r) c``,
    with the substep-F.1.a.5 phase rescale (row * i for
    z-derivative-bearing rows; col * -i for the SV columns)
    absorbed so every entry is real-valued in the bound regime.

    The four columns correspond to the I- and K-flavours of the
    P scalar potential ``phi = B_I I_0(p r) + B_K K_0(p r)`` and
    the SV vector potential ``psi_theta = C_I I_1(s r) + C_K
    K_1(s r)``; the four rows correspond to the four state-vector
    components whose continuity is enforced at every radial
    interface (substep G.a.1).

    Parameters
    ----------
    kz, omega : float
        Trial axial wavenumber (rad / m) and angular frequency
        (rad / s).
    vp, vs, rho : float
        Layer P / S velocity (m/s) and density (kg/m^3). Must
        satisfy ``vp > vs > 0`` and ``rho > 0``.
    r : float
        Radius at which to evaluate ``E(r)`` (m). Must be
        positive.

    Returns
    -------
    ndarray, shape (4, 4)
        ``E(r)`` post-rescale. Real-valued in the bound regime;
        ``NaN``-filled outside the bound regime (``kz`` below the
        layer's bound floor ``omega / V_S``).

    Notes
    -----
    Per-element oracles vs F.1.b in :mod:`tests.test_cylindrical_solver`:

    * Row 0 (``u_r``): cols match
      ``-_layered_n0_row1_at_a[1:5]`` (negation from the BC's
      ``f - m`` subtraction).
    * Row 1 (``u_z``): cols match the layer side of
      ``_layered_n0_row5_at_b`` evaluated at ``r=b`` (positive
      sign; row 5 BC is ``m - s``).
    * Row 2 (``sigma_rr``): cols match
      ``-_layered_n0_row2_at_a[1:5]`` (negation; row 2 BC is
      ``-(sigma_rr_m + P_f) = 0``).
    * Row 3 (``sigma_rz``): cols match
      ``_layered_n0_row3_at_a[1:5]`` (positive sign; row 3 BC is
      ``sigma_rz_m = 0``, no subtraction).

    See Also
    --------
    _layer_propagator_n0 : G.b.2 follow-up using
        ``E(r_outer) E(r_inner)^{-1}`` to map the state vector
        across a layer.
    """
    p2 = kz * kz - (omega / vp) ** 2
    s2 = kz * kz - (omega / vs) ** 2
    if p2 <= 0.0 or s2 <= 0.0 or r <= 0.0:
        return np.full((4, 4), np.nan)
    p = float(np.sqrt(p2))
    s = float(np.sqrt(s2))
    pr = p * r
    sr = s * r
    I0_p, I1_p = _i0_i1(pr)
    K0_p, K1_p = _k0_k1(pr)
    I0_s, I1_s = _i0_i1(sr)
    K0_s, K1_s = _k0_k1(sr)
    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    E = np.zeros((4, 4))
    # Row 0: u_r = d_r phi - i k_z psi_theta. Post-rescale (col-by-(-i)
    # on the C columns kills the explicit -i k_z).
    E[0, 0] = +p * I1_p
    E[0, 1] = -p * K1_p
    E[0, 2] = -kz * I1_s
    E[0, 3] = -kz * K1_s
    # Row 1: u_z = i k_z phi + (1/r) d_r(r psi_theta).
    # Post-rescale: row*i flips +i*R to -R on B cols; row*i AND col*-i
    # leaves +R unchanged on C cols.
    E[1, 0] = -kz * I0_p
    E[1, 1] = -kz * K0_p
    E[1, 2] = +s * I0_s
    E[1, 3] = -s * K0_s
    # Row 2: sigma_rr = lambda div u + 2 mu d_r u_r. Post-rescale: no
    # row scaling (sigma_rr is no-z-derivative). col*-i on C cols.
    E[2, 0] = +mu * (two_kz2_minus_kS2 * I0_p - 2.0 * p * I1_p / r)
    E[2, 1] = +mu * (two_kz2_minus_kS2 * K0_p + 2.0 * p * K1_p / r)
    E[2, 2] = -2.0 * mu * kz * (s * I0_s - I1_s / r)
    E[2, 3] = +2.0 * mu * kz * (s * K0_s + K1_s / r)
    # Row 3: sigma_rz = mu (d_z u_r + d_r u_z). Post-rescale: row*i
    # flips +/- i*R on B cols, leaves +R on C cols (row*i AND col*-i).
    E[3, 0] = -2.0 * mu * kz * p * I1_p
    E[3, 1] = +2.0 * mu * kz * p * K1_p
    E[3, 2] = +mu * two_kz2_minus_kS2 * I1_s
    E[3, 3] = +mu * two_kz2_minus_kS2 * K1_s
    return E


# =====================================================================
# Plan item G.b.2 -- per-layer propagator P(r_outer | r_inner)
# =====================================================================
#
# Wraps two evaluations of ``_layer_e_matrix_n0`` into the layer
# propagator P_j = E(r_outer) E(r_inner)^{-1} (substep G.a.3).
# Uses ``np.linalg.solve`` rather than explicit ``inv`` for
# better conditioning (and to avoid the standard "compose,
# don't invert" footgun on near-singular E).
#
# The propagator maps the full state vector
# ``v(r) = (u_r, u_z, sigma_rr, sigma_rz)`` from r_inner to
# r_outer within a uniform layer:
#
#       v(r_outer) = P(r_outer | r_inner) v(r_inner)
#
# Composition / round-trip oracles in the G.b.2 tests pin the
# group-law identities P(r3|r1) = P(r3|r2) P(r2|r1) and
# P(r2|r1) P(r1|r2) = I to floating-point precision.


def _fluid_layer_e_matrix_n0(
    kz: float,
    omega: float,
    *,
    vf: float,
    rho: float,
    r: float,
) -> np.ndarray:
    r"""
    2x2 amplitude-to-state matrix ``E_f(r)`` for a fluid annulus, n=0.

    The fluid counterpart of :func:`_layer_e_matrix_n0`, and the first
    piece of the microannulus model for the debonded regime (roadmap
    G.2). A fluid annulus differs from an elastic one in three ways
    that together change the shape of the problem rather than just its
    numbers: it carries **two** wave amplitudes rather than four (there
    is no shear potential), its shear traction vanishes identically,
    and the axial displacement is free to slip at its boundaries. So
    the propagated state is the reduced pair

    .. math::

        w(r) = \big(u_r(r),\; \sigma_{rr}(r)\big)

    rather than the elastic four-vector
    ``(u_r, u_z, sigma_rr, sigma_rz)``.

    With pressure ``p = A_I I_0(F r) + A_K K_0(F r)`` and
    ``F = sqrt(k_z^2 - omega^2 / V_f^2)``, the fluid momentum equation
    ``u = grad p / (rho omega^2)`` gives

    .. math::

        u_r = \frac{F}{\rho \omega^2}
              \big(A_I I_1(F r) - A_K K_1(F r)\big),
        \qquad
        \sigma_{rr} = -p .

    Parameters
    ----------
    kz, omega : float
        Trial axial wavenumber (rad/m) and angular frequency (rad/s).
    vf : float
        Fluid acoustic velocity of the annulus (m/s); must be positive.
        This is the annulus fluid, which need not equal the borehole
        fluid.
    rho : float
        Fluid density of the annulus (kg/m^3); must be positive.
    r : float
        Radius at which to evaluate (m); must be positive.

    Returns
    -------
    ndarray
        Real ``(2, 2)`` array mapping ``(A_I, A_K)`` to
        ``(u_r, sigma_rr)`` at ``r``.

    Notes
    -----
    The determinant has a closed form and no dependence on ``F`` at
    all. The Bessel Wronskian ``I_0(x) K_1(x) + I_1(x) K_0(x) = 1/x``
    collapses it to

    .. math::

        \det E_f(r) = -\frac{1}{\rho \omega^2 r},

    which is the identity the tests use to check this matrix without
    reference to any other implementation.
    """
    if vf <= 0.0 or rho <= 0.0:
        raise ValueError("vf and rho must be positive")
    if r <= 0.0:
        raise ValueError("r must be positive")

    f_radial = np.sqrt(complex(kz * kz - (omega / vf) ** 2))
    if abs(f_radial.imag) > 1.0e-12 * max(abs(f_radial.real), 1.0):
        raise ValueError(
            "fluid annulus radial wavenumber is not real; the bound-mode "
            "formulation requires kz > omega / vf for the annulus fluid"
        )
    f_real = float(f_radial.real)

    i0, i1 = _i0_i1(f_real * r)
    k0, k1 = _k0_k1(f_real * r)
    scale = f_real / (rho * omega**2)

    e_matrix = np.zeros((2, 2))
    e_matrix[0, 0] = scale * i1
    e_matrix[0, 1] = -scale * k1
    e_matrix[1, 0] = -i0
    e_matrix[1, 1] = -k0
    return e_matrix


def _fluid_layer_propagator_n0(
    kz: float,
    omega: float,
    *,
    vf: float,
    rho: float,
    r_inner: float,
    r_outer: float,
) -> np.ndarray:
    r"""
    2x2 propagator ``w(r_outer) = P_f w(r_inner)`` across a fluid annulus.

    ``P_f = E_f(r_outer) E_f(r_inner)^{-1}`` with ``E_f`` from
    :func:`_fluid_layer_e_matrix_n0`, mirroring
    :func:`_layer_propagator_n0` for the elastic case.

    Parameters
    ----------
    kz, omega : float
        Trial axial wavenumber (rad/m) and angular frequency (rad/s).
    vf, rho : float
        Annulus fluid velocity (m/s) and density (kg/m^3).
    r_inner, r_outer : float
        Inner and outer radii of the annulus (m). Both positive;
        ``r_outer >= r_inner``.

    Returns
    -------
    ndarray
        Real ``(2, 2)`` propagator for the state ``(u_r, sigma_rr)``.

    Notes
    -----
    Its determinant is fixed by the Wronskian identity quoted in
    :func:`_fluid_layer_e_matrix_n0`:

    .. math::

        \det P_f = \frac{\det E_f(r_{\text{outer}})}
                          {\det E_f(r_{\text{inner}})}
                  = \frac{r_{\text{inner}}}{r_{\text{outer}}},

    independent of frequency, velocity, density and ``k_z``. That makes
    it a sharp check on the implementation: a sign slip or a swapped
    Bessel order breaks it immediately, and it holds for reasons
    outside this module.

    **Validity range.** The ``I`` and ``K`` Bessel functions grow and
    decay across the annulus, so the dynamic range of ``E_f`` scales
    with ``F * (r_outer - r_inner)`` and eventually exceeds double
    precision. Using the determinant identity as the error measure:
    machine precision up to a span of about 2, ~1e-11 by 7, and no
    significant digits by 20. This is not a constraint for the
    microannulus case the element exists for -- a debonding gap is
    microns to millimetres, putting the span below 0.1 -- but a thick
    fluid annulus at high ``k_z`` is outside what this formulation can
    represent, and the same exponential-range failure has produced
    spurious roots elsewhere in this module.
    """
    if r_inner <= 0.0 or r_outer <= 0.0:
        raise ValueError("r_inner and r_outer must be positive")
    if r_outer < r_inner:
        raise ValueError("require r_outer >= r_inner")

    e_inner = _fluid_layer_e_matrix_n0(kz, omega, vf=vf, rho=rho, r=r_inner)
    e_outer = _fluid_layer_e_matrix_n0(kz, omega, vf=vf, rho=rho, r=r_outer)
    # P = E_out @ inv(E_in), solved rather than inverted.
    return np.linalg.solve(e_inner.T, e_outer.T).T


def _layer_propagator_n0(
    kz: float,
    omega: float,
    *,
    vp: float,
    vs: float,
    rho: float,
    r_inner: float,
    r_outer: float,
) -> np.ndarray:
    r"""
    4x4 layer propagator ``P(r_outer | r_inner)`` mapping the
    state vector ``v(r) = (u_r, u_z, sigma_rr, sigma_rz)`` from
    ``r_inner`` to ``r_outer`` within a uniform isotropic-elastic
    layer at azimuthal order n=0.

    Implements substep G.a.3:

    .. math::
        P_j(r_{\text{outer}} \mid r_{\text{inner}}) =
        E(r_{\text{outer}}) \, E(r_{\text{inner}})^{-1}

    with ``E`` from :func:`_layer_e_matrix_n0`. The inversion is
    performed via :func:`numpy.linalg.solve` (with the transpose
    trick so the right-multiplication by the inverse becomes a
    left-multiplication-style solve).

    Parameters
    ----------
    kz, omega : float
        Trial axial wavenumber (rad / m) and angular frequency
        (rad / s).
    vp, vs, rho : float
        Layer P / S velocity (m/s) and density (kg/m^3). Must
        satisfy ``vp > vs > 0`` and ``rho > 0``.
    r_inner, r_outer : float
        Inner and outer radii of the layer (m). May be in
        either order; the round-trip identity
        ``P(b|a) P(a|b) = I`` is verified in the tests.
        ``r_inner == r_outer`` returns ``eye(4)``.

    Returns
    -------
    ndarray, shape (4, 4)
        The layer propagator. Real-valued in the bound regime;
        ``NaN``-filled outside the bound regime (propagated from
        :func:`_layer_e_matrix_n0`).

    See Also
    --------
    _layer_e_matrix_n0 : G.b.1 helper that this function calls
        twice (at ``r_inner`` and ``r_outer``).
    """
    if r_inner == r_outer:
        return np.eye(4)
    E_inner = _layer_e_matrix_n0(
        kz=kz,
        omega=omega,
        vp=vp,
        vs=vs,
        rho=rho,
        r=r_inner,
    )
    E_outer = _layer_e_matrix_n0(
        kz=kz,
        omega=omega,
        vp=vp,
        vs=vs,
        rho=rho,
        r=r_outer,
    )
    if not (np.all(np.isfinite(E_inner)) and np.all(np.isfinite(E_outer))):
        return np.full((4, 4), np.nan)
    # Compose ``E_outer @ inv(E_inner)`` via ``solve``:
    #   X = E_outer @ inv(E_inner)
    #   X @ E_inner = E_outer
    #   E_inner.T @ X.T = E_outer.T
    # so X.T = solve(E_inner.T, E_outer.T), and X is the transpose.
    return np.linalg.solve(E_inner.T, E_outer.T).T


# =====================================================================
# Plan item G.c -- stacked modal determinant
# =====================================================================
#
# Assembles the 7x7 cased-hole modal determinant for an N-layer
# stack via the propagator chain. Mirrors F.1's hand-coded
# ``_modal_determinant_n0_layered`` (N=1) but folds the layer-
# internal amplitudes through ``P_total = P_N ... P_2 P_1`` so
# the final 7x7 form has the same column structure
# ``[A | c_1 (4 layer-1 amps) | B_form_K, C_form_K]`` regardless
# of N.
#
# Algorithm:
#   1. Compute layer radii r_0 = a, r_j = r_{j-1} + thickness_j.
#   2. Build per-layer propagators P_j via G.b.2 and compose
#      P_total = P_N @ ... @ P_2 @ P_1.
#   3. Build E_1(a) for the innermost layer.
#   4. Compose v_at_b = (P_total @ E_1(a)): the 4x4 map from c_1
#      to the state vector at the outermost interface r = b.
#   5. Build E_form(b) for the formation half-space; pick its
#      K-flavour columns (1, 3) for the 4x2 formation block.
#   6. Assemble the 7x7 matrix following the F.1 layout:
#      - Row 0: BC1 (u_r continuity at r=a, fluid - layer).
#      - Row 1: BC2 (-(sigma_rr + P_f) = 0 at r=a).
#      - Row 2: BC3 (sigma_rz = 0 at r=a).
#      - Rows 3-6: BC4-7 (u_r, u_z, sigma_rr, sigma_rz continuity
#        at r=b, layer - formation).
#   7. Take det(M.real).
#
# Reduces to ``_modal_determinant_n0_layered`` at N=1 (sharing
# the same brentq root in k_z, with a layer-1-amplitude scale
# factor that's absorbed at the determinant level). The N=1
# brentq-root match is the floating-point oracle for G.c.


def _modal_determinant_n0_cased(
    kz: float,
    omega: float,
    *,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    layers: tuple[BoreholeLayer, ...],
) -> float:
    r"""
    7x7 axisymmetric modal determinant for a borehole with N
    annular layers between fluid and formation.

    Generalises :func:`_modal_determinant_n0_layered` (N=1,
    hand-coded F.1) to arbitrary ``N >= 1`` via the Thomson-
    Haskell-style propagator chain
    ``P_total = P_N ... P_2 P_1`` (substep G.a.3). The 7x7 final
    form has column structure
    ``[A | c_1 (4 innermost-layer amplitudes) | B_form_K, C_form_K]``
    regardless of N; the propagator chain folds the layer-
    internal amplitudes ``c_2, ..., c_N`` out via radial
    continuity at every layer/layer interface (substep G.a.4).

    Reduces to :func:`_modal_determinant_n0_layered` at N=1
    (same brentq root in ``k_z``, up to an overall layer-1-
    amplitude scale factor that cancels at the root). The
    floating-point oracle for this routine is the multi-
    frequency ``stoneley_dispersion_layered(layers=(layer,))``
    regression in :func:`stoneley_dispersion_cased` (G.d).

    Parameters
    ----------
    kz, omega, vp, vs, rho, vf, rho_f, a : float
        Same as :func:`_modal_determinant_n0_layered`. ``vp``,
        ``vs``, ``rho`` are the formation half-space (outermost
        side); ``vf``, ``rho_f`` are the borehole fluid.
    layers : tuple of BoreholeLayer
        Annular layer stack ordered inside-out: ``layers[0]``
        adjacent to fluid, ``layers[-1]`` adjacent to formation.
        Must have at least one layer (``len(layers) >= 1``); the
        unlayered case dispatches to :func:`_modal_determinant_n0`
        in :func:`stoneley_dispersion_cased`.

    Returns
    -------
    float
        ``det(M)`` of the 7x7 cased-hole modal matrix, real-
        valued in the bound regime. ``NaN`` outside the bound
        regime (any layer or the formation has imaginary radial
        wavenumber, or the propagator chain fails the bound-
        regime gate).
    """
    F_f_sq = kz * kz - (omega / vf) ** 2
    if F_f_sq <= 0.0:
        return float("nan")
    F_f = float(np.sqrt(F_f_sq))
    # Layer radii: r_0 = a, r_j = r_{j-1} + thickness_j.
    radii = [a]
    for L in layers:
        radii.append(radii[-1] + L.thickness)
    b = radii[-1]
    # Innermost-layer E_1(a).
    L1 = layers[0]
    E_1_a = _layer_e_matrix_n0(
        kz=kz,
        omega=omega,
        vp=L1.vp,
        vs=L1.vs,
        rho=L1.rho,
        r=a,
    )
    if not np.all(np.isfinite(E_1_a)):
        return float("nan")
    # Per-layer propagator chain P_total = P_N ... P_2 P_1.
    P_total = np.eye(4)
    for j, L in enumerate(layers):
        P_j = _layer_propagator_n0(
            kz=kz,
            omega=omega,
            vp=L.vp,
            vs=L.vs,
            rho=L.rho,
            r_inner=radii[j],
            r_outer=radii[j + 1],
        )
        if not np.all(np.isfinite(P_j)):
            return float("nan")
        P_total = P_j @ P_total
    # Formation E_form(b); only K-flavour columns (1, 3) contribute
    # because the half-space requires decay outward.
    E_form_b = _layer_e_matrix_n0(
        kz=kz,
        omega=omega,
        vp=vp,
        vs=vs,
        rho=rho,
        r=b,
    )
    if not np.all(np.isfinite(E_form_b)):
        return float("nan")
    # State vector at b from c_1: v_N(b) = P_total @ E_1(a) c_1.
    #
    # The propagator carries growing and decaying exponentials across each
    # layer, and a very compliant layer (V_S far below the fluid velocity,
    # e.g. a debonding proxy) drives its dynamic range past double
    # precision. Overflow here does not raise -- it produces inf, the 7x7
    # determinant below becomes meaningless, and the bracket search then
    # finds sign changes in the garbage and reports them as roots. Bail out
    # instead, so the caller gets NaN.
    if not (np.all(np.isfinite(P_total)) and np.all(np.isfinite(E_1_a))):
        return float("nan")
    # Check the product cannot overflow *before* forming it: the matmul
    # itself emits the RuntimeWarning, so a post-hoc isfinite test cleans up
    # the determinant but not the warning.
    scale = float(np.max(np.abs(P_total))) * float(np.max(np.abs(E_1_a)))
    if not np.isfinite(scale) or scale > np.sqrt(np.finfo(float).max):
        return float("nan")
    v_at_b = P_total @ E_1_a
    if not np.all(np.isfinite(v_at_b)):
        return float("nan")
    # Fluid Bessel pack at r = a.
    I0_Ff_a = float(special.iv(0, F_f * a))
    I1_Ff_a = float(special.iv(1, F_f * a))
    # Assemble the 7x7 modal matrix. Column order:
    # [A | B_I^1, B_K^1, C_I^1, C_K^1 | B_form_K, C_form_K]
    M = np.zeros((7, 7))
    # Row 0: BC1 u_r^(f)(a) - u_r^(layer)(a) = 0.
    M[0, 0] = F_f * I1_Ff_a / (rho_f * omega**2)
    M[0, 1:5] = -E_1_a[0, :]
    # Formation cols vanish at r=a.
    # Row 1: BC2 -(sigma_rr^(layer)(a) + P^(f)(a)) = 0.
    M[1, 0] = -I0_Ff_a
    M[1, 1:5] = -E_1_a[2, :]
    # Row 2: BC3 sigma_rz^(layer)(a) = 0 (no fluid contribution).
    M[2, 1:5] = +E_1_a[3, :]
    # Rows 3-6: BC4-7 continuity at r=b, (layer - formation) for each
    # of (u_r, u_z, sigma_rr, sigma_rz).
    for i_state, i_M in zip(range(4), range(3, 7)):
        M[i_M, 1:5] = +v_at_b[i_state, :]
        # Formation K-flavour cols: B_K -> col 1, C_K -> col 3 of
        # E_form. Subtracted in BC, so negative sign.
        M[i_M, 5] = -E_form_b[i_state, 1]
        M[i_M, 6] = -E_form_b[i_state, 3]
    return float(np.linalg.det(M))


# Largest modified-Bessel argument the unscaled ``I_n`` representation can
# carry with headroom for the stress prefactors: ``I_0(x) ~ e^x``, and the
# stacked assemblies below already demand that products stay under
# ``sqrt(DBL_MAX)``, so the same rule in the exponent gives
# ``x <= log(sqrt(DBL_MAX)) ~ 354.9``.
_BESSEL_ARG_MAX = float(np.log(np.sqrt(np.finfo(float).max)))


# =====================================================================
# Plan item G.2.c -- global assembly for the fluid microannulus
# =====================================================================
#
# Assembles the 11x11 n=0 modal determinant for a stack of the form
#
#     borehole fluid | inner elastic block | fluid annulus
#                    | outer elastic block | formation
#
# i.e. ``fluid | casing | microannulus | cement | formation``, the
# standard model of debonding in cement-bond logging (roadmap G.2).
#
# Why the assembly is not the 7x7 with one more layer. A fluid
# annulus carries two wave amplitudes rather than four, its shear
# traction vanishes identically, and axial displacement slips across
# it. So the elastic four-vector cannot be propagated through it, and
# the stack splits into two independent elastic blocks joined by a
# two-component state ``(u_r, sigma_rr)``.
#
# Counting, with the annulus amplitudes folded out through the fluid
# propagator exactly as the layer-internal amplitudes are folded out
# in the all-elastic 7x7:
#
#   unknowns   A (borehole fluid, regular at the axis)          1
#              inner block innermost amplitudes c               4
#              outer block innermost amplitudes c'              4
#              formation K-flavour amplitudes                   2
#                                                             ---
#                                                              11
#
#   equations  r = a  fluid | solid: u_r, sigma_rr, sigma_rz    3
#              r = b  solid | annulus: sigma_rz = 0             1
#              r = c  annulus | solid: u_r, sigma_rr, sigma_rz  3
#              r = d  solid | formation: full four-vector       4
#                                                             ---
#                                                              11
#
# The ``(u_r, sigma_rr)`` pair is not matched at ``r = b``: it is
# carried across the annulus by ``_fluid_layer_propagator_n0`` and
# matched at ``r = c``. Extra layers inside either block add four
# unknowns and four continuity equations apiece and are absorbed by
# the elastic propagator, leaving the size unchanged.
#
# Bracket floor. The roadmap argument for why this configuration is
# reachable while a soft elastic layer is not needs one correction:
# the annulus fluid does contribute a floor, at its *acoustic*
# velocity. What matters is that this is ~1500 m/s rather than the
# near-zero shear velocity of a compliant-solid debonding proxy, so
# it does not drag ``min(V_S, V_f, ...)`` below the fluid velocity and
# collapse the bound window.


def _modal_determinant_n0_microannulus(
    kz: float,
    omega: float,
    *,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    inner_layers: tuple[BoreholeLayer, ...],
    annulus_vf: float,
    annulus_rho: float,
    annulus_thickness: float,
    outer_layers: tuple[BoreholeLayer, ...],
) -> float:
    r"""
    11x11 axisymmetric modal determinant for a stack split by a fluid
    microannulus.

    The debonded-regime counterpart of
    :func:`_modal_determinant_n0_cased`: two elastic blocks joined by a
    thin fluid annulus, as in ``borehole fluid | casing | microannulus
    | cement | formation``. A fluid annulus is the standard model of
    debonding in cement-bond logging, and it is *not* reachable by
    softening a cement layer -- an elastic layer, however compliant,
    drags the bound-regime bracket floor below the fluid velocity,
    while a fluid annulus contributes a floor at its own acoustic
    velocity. See the block comment above and roadmap G.2.

    Column structure ``[A | c (4) | c' (4) | B_form_K, C_form_K]``,
    where ``c`` and ``c'`` are the innermost-layer amplitudes of the
    inner and outer elastic blocks; the annulus amplitudes are folded
    out through :func:`_fluid_layer_propagator_n0`, mirroring the way
    layer-internal amplitudes are folded out in the 7x7.

    Parameters
    ----------
    kz, omega : float
        Trial axial wavenumber (rad/m) and angular frequency (rad/s).
    vp, vs, rho : float
        Formation half-space P / S velocity (m/s) and density
        (kg/m^3).
    vf, rho_f : float
        Borehole-fluid velocity (m/s) and density (kg/m^3).
    a : float
        Borehole radius (m).
    inner_layers, outer_layers : tuple of BoreholeLayer
        Elastic layers inside and outside the annulus, each ordered
        radially outward. Both must be non-empty. A zero-thickness
        inner block makes the two ``sigma_rz = 0`` rows at ``r = a``
        and ``r = b`` identical and the matrix singular for every
        ``k_z``, so the degenerate case is not merely unsupported but
        meaningless.
    annulus_vf, annulus_rho, annulus_thickness : float
        Annulus fluid velocity (m/s), density (kg/m^3) and radial
        thickness (m). A debonding gap is microns to millimetres.

    Returns
    -------
    float
        ``det M`` at the trial ``k_z``. ``NaN`` outside the bound
        regime -- which now includes ``k_z <= omega / annulus_vf`` --
        and where the propagator chain cannot be formed in double
        precision (same magnitude guard as the all-elastic assembly).

    Raises
    ------
    ValueError
        If either block is empty, or ``annulus_thickness``,
        ``annulus_vf`` or ``annulus_rho`` is not positive.

    Notes
    -----
    **Two root families.** This determinant has more than one bound
    root, and any bracketing added on top of it has to choose. Besides
    the Stoneley-like mode just below the fluid velocity there is a
    slow mode guided by the gap itself, 68 m/s at a 1 um gap rising to
    620 m/s at 1 mm (8 kHz). The two move in opposite directions as the
    gap closes. The ``n=0`` branch-selection defect fixed earlier in
    this module was exactly this failure -- a bracket that assumed one
    root and returned whichever the grid straddled.

    Checks used in :mod:`tests.test_cylindrical_solver`, and what each
    can and cannot catch:

    * **The Krauklis crack wave.** The slow root is the wave guided by
      a thin fluid layer between elastic walls, whose speed has a
      closed form obtained from lubrication flow plus the quasi-static
      half-space response:

      .. math::

          c = \left(\frac{\omega h}{C \rho_f}\right)^{1/3},
          \qquad C = \sum_{\text{walls}} \frac{1 - \nu}{\mu}.

      No Bessel functions, no cylindrical geometry and no shared code,
      and it fixes an *absolute* velocity rather than a scaling -- so a
      wrong row or a swapped interface condition shows up as an O(1)
      prefactor error. Reproduced to 0.02 % at a 1 um gap, departing
      as ``k h`` grows. This is the check that this assembly is right
      rather than merely self-consistent.
    * **Independent assembly.** A 13x13 form that keeps the two
      annulus amplitudes as explicit unknowns, assembled separately in
      the tests, locates the same roots. Different size, different
      column layout, different rows -- so an index or sign slip in one
      does not reproduce in the other. It shares
      :func:`_fluid_layer_e_matrix_n0` with this routine, so it does
      not independently confirm the fluid element itself; the
      Wronskian identity ``det E_f = -1/(rho omega^2 r)`` does that.
    * **Subdivision invariance.** Splitting any elastic layer of
      either block into sub-layers leaves the roots fixed to rounding.
    * There is **no reduction to the existing solver**, which is why
      the checks above have to come from outside. The
      ``annulus_thickness -> 0`` limit is a frictionless *slip*
      interface, not the bonded stack: ``u_z`` stays free and
      ``sigma_rz`` stays zero on both faces however thin the gap. The
      limit is measured and its offset from the bonded root reported
      (16.6 m/s, 1.2 %, at 8 kHz), not asserted to vanish.
    """
    if not inner_layers or not outer_layers:
        raise ValueError("inner_layers and outer_layers must both be non-empty")
    if annulus_thickness <= 0.0:
        raise ValueError("annulus_thickness must be positive")
    if annulus_vf <= 0.0 or annulus_rho <= 0.0:
        raise ValueError("annulus_vf and annulus_rho must be positive")

    f_core_sq = kz * kz - (omega / vf) ** 2
    f_ann_sq = kz * kz - (omega / annulus_vf) ** 2
    if f_core_sq <= 0.0 or f_ann_sq <= 0.0:
        return float("nan")
    f_core = float(np.sqrt(f_core_sq))

    # Bound every Bessel argument *before* any E matrix is built. The helpers
    # evaluate unscaled ``I_n``, which overflows at large argument -- and the
    # overflow happens inside the helper, so a post-hoc isfinite test cleans up
    # the determinant but not the RuntimeWarning. Every radial wavenumber in
    # the stack is below ``kz`` and every radius is below ``r_outermost``, so
    # ``kz * r_outermost`` bounds all of them at once. The cap is the same
    # square-root-of-double-max headroom rule the product guard below uses,
    # expressed in the exponent: ``I_0(x) ~ e^x``. It only bites at phase
    # velocities far below any physical mode (tens of m/s at logging
    # frequencies), where the determinant is not representable anyway.
    r_outermost = a + sum(L.thickness for L in inner_layers) + annulus_thickness
    r_outermost += sum(L.thickness for L in outer_layers)
    if kz * r_outermost > _BESSEL_ARG_MAX:
        return float("nan")

    def _block(
        layers: tuple[BoreholeLayer, ...], r_start: float
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """``(E at the block's inner face, propagator chain, outer radius)``."""
        e_inner = _layer_e_matrix_n0(
            kz=kz,
            omega=omega,
            vp=layers[0].vp,
            vs=layers[0].vs,
            rho=layers[0].rho,
            r=r_start,
        )
        transfer = np.eye(4)
        r_lo = r_start
        for layer in layers:
            r_hi = r_lo + layer.thickness
            transfer = (
                _layer_propagator_n0(
                    kz=kz,
                    omega=omega,
                    vp=layer.vp,
                    vs=layer.vs,
                    rho=layer.rho,
                    r_inner=r_lo,
                    r_outer=r_hi,
                )
                @ transfer
            )
            r_lo = r_hi
        return e_inner, transfer, r_lo

    e_in_at_a, transfer_in, r_b = _block(inner_layers, a)
    r_c = r_b + annulus_thickness
    e_out_at_c, transfer_out, r_d = _block(outer_layers, r_c)
    e_form_at_d = _layer_e_matrix_n0(kz=kz, omega=omega, vp=vp, vs=vs, rho=rho, r=r_d)
    p_annulus = _fluid_layer_propagator_n0(
        kz,
        omega,
        vf=annulus_vf,
        rho=annulus_rho,
        r_inner=r_b,
        r_outer=r_c,
    )

    parts = (e_in_at_a, transfer_in, e_out_at_c, transfer_out, e_form_at_d, p_annulus)
    if not all(bool(np.all(np.isfinite(x))) for x in parts):
        return float("nan")
    # The fluid propagator has an exact determinant, ``r_inner / r_outer`` from
    # the Bessel Wronskian, so its own validity can be checked at runtime
    # instead of guessed at: past the span where the I / K dynamic range
    # exhausts double precision the identity fails first and the propagator
    # entries become meaningless second. Enforcing it turns the documented
    # validity range of _fluid_layer_propagator_n0 into a hard gate -- this
    # module has twice produced spurious roots from sign changes in
    # unrepresentable propagators, and a microannulus span is ~1e-3, so a
    # configuration that trips this is outside the model, not near its edge.
    wronskian = float(np.linalg.det(p_annulus)) * r_c / r_b
    if not np.isfinite(wronskian) or abs(wronskian - 1.0) > 1.0e-6:
        return float("nan")
    # Same pre-multiplication magnitude guard as the all-elastic assembly: a
    # compliant layer can drive the propagator's dynamic range past double
    # precision, and the matmul itself warns, so the product has to be
    # rejected before it is formed rather than after.
    limit = float(np.sqrt(np.finfo(float).max))
    for transfer, e_inner in ((transfer_in, e_in_at_a), (transfer_out, e_out_at_c)):
        scale = float(np.max(np.abs(transfer))) * float(np.max(np.abs(e_inner)))
        if not np.isfinite(scale) or scale > limit:
            return float("nan")
    v_in_at_b = transfer_in @ e_in_at_a
    v_out_at_d = transfer_out @ e_out_at_c

    # ``(u_r, sigma_rr)`` carried across the annulus, still expressed in the
    # inner block's amplitudes ``c``. Rows 0 and 2 of the elastic state vector
    # are exactly the pair the fluid propagates.
    pair_at_c = p_annulus @ v_in_at_b[(0, 2), :]

    i0_a = float(special.iv(0, f_core * a))
    i1_a = float(special.iv(1, f_core * a))

    # Columns: [A | c (4) | c' (4) | B_form_K, C_form_K]
    m = np.zeros((11, 11))
    # r = a: borehole fluid against the inner block, same three rows as the
    # 7x7 (u_r continuity, -(sigma_rr + P_f) = 0, sigma_rz = 0).
    m[0, 0] = f_core * i1_a / (rho_f * omega**2)
    m[0, 1:5] = -e_in_at_a[0, :]
    m[1, 0] = -i0_a
    m[1, 1:5] = -e_in_at_a[2, :]
    m[2, 1:5] = e_in_at_a[3, :]
    # r = b: the annulus carries no shear traction, so the inner block's outer
    # face is traction-free in shear. Its (u_r, sigma_rr) are not matched here.
    m[3, 1:5] = v_in_at_b[3, :]
    # r = c: the propagated pair against the outer block, plus the matching
    # shear-traction-free condition on the outer block's inner face.
    m[4, 1:5] = pair_at_c[0, :]
    m[4, 5:9] = -e_out_at_c[0, :]
    m[5, 1:5] = pair_at_c[1, :]
    m[5, 5:9] = -e_out_at_c[2, :]
    m[6, 5:9] = e_out_at_c[3, :]
    # r = d: outer block against the formation half-space, K-flavour columns
    # only (decay outward).
    for state in range(4):
        m[7 + state, 5:9] = v_out_at_d[state, :]
        m[7 + state, 9] = -e_form_at_d[state, 1]
        m[7 + state, 10] = -e_form_at_d[state, 3]

    # Single backstop rather than one check per intermediate product: the
    # guards above bound every factor that feeds ``m``, so this was not reached
    # by any of 129,000 randomised stacks that passed them. Kept anyway because
    # the alternative failure -- ``det`` of a non-finite matrix, warning and
    # then reporting sign changes in the result as roots -- is the exact defect
    # this module shipped twice.
    if not np.all(np.isfinite(m)):  # pragma: no cover - guarded upstream
        return float("nan")
    return float(np.linalg.det(m))


# =====================================================================
# Substep G'.a -- math scaffolding for the cased-hole multi-layer
#                 propagator-matrix determinant (n=1 flexural)
# =====================================================================
#
# Sister of substep G.a at azimuthal order n=1. Generalises F.2's
# single-extra-layer 10x10 hand-coded form to N stacked annular
# layers via a 6x6 propagator matrix in cylindrical geometry
# (the n=1 analogue of G.a's 4x4 propagator). Comments-only block;
# the helpers themselves land in plan items G'.b (per-layer
# propagator), G'.c (assembly), and G'.d (public-API hook). See
# ``docs/plans/cylindrical_biot_G_prime.md`` for the full sub-plan.
#
# Sign / Bessel / phase conventions are the same as F.2.a (which
# this scaffolding extends), unchanged from the isotropic
# single-layer flexural solver upstream.
#
# Per-region scalar / vector potentials at azimuthal order n=1:
# in each elastic layer (or formation half-space)
#   phi          = [B_I I_1(p r) + B_K K_1(p r)]      cos(theta)
#   psi_theta    = [C_I I_1(s r) + C_K K_1(s r)]      cos(theta)
#   psi_z        = [D_I I_1(s r) + D_K K_1(s r)]      sin(theta)
# with three wave families (P scalar, SV vector theta-comp, SH
# vector z-comp) and two flavours each (I, K). Six amplitudes per
# layer, vs four at n=0 -- the SH polarisation ``psi_z`` is
# genuinely coupled at n=1 via the d_theta operations and cannot
# be dropped (this is the F.2.a.6 erratum: cos / sin sectors at
# n=1 are NOT block-diagonal). The fluid keeps the I-flavour
# only (regular at axis) and the formation half-space keeps the
# K-flavour only (decay outward), unchanged from F.2.
#
# The N=0 limit reduces to the unlayered ``_modal_determinant_n1``;
# the N=1 limit reduces to ``_modal_determinant_n1_layered`` with
# no overall scale factor (P_1 @ E_1(a) = E_1(b) at N=1; same
# brentq root). Both limits anchor the floating-point regression
# oracles in G'.c.
#
# -----
# Substep G'.a.1 -- State vector at the dipole order n=1
# -----
#
# Within any uniform elastic layer (or the formation half-space),
# the radial-dependent part of the displacement / stress field at
# n=1 reduces to a 6-component state vector ``v(r)``:
#
#       v(r) = ( u_r(r),  u_z(r),  u_theta(r),
#                sigma_rr(r),  sigma_rz(r),  sigma_r_theta(r) )^T
#
# Six components (vs four at n=0) because at n>=1 the d_theta
# operations make ``u_theta`` and ``sigma_r_theta`` non-trivial.
# The cos / sin azimuthal factors have been stripped per substep
# F.2.a.4 (cos for u_r, u_z, sigma_rr, sigma_rz; sin for u_theta,
# sigma_r_theta), so the state vector entries are functions of r
# only.
#
# At the layer/layer (or annulus/formation) boundary r = b_j, the
# n=1 boundary conditions are full state-vector continuity:
# ``v(b_j^-) = v(b_j^+)``, six identities per interface.
#
# At the fluid-solid interface r = a, only FOUR of the six state-
# vector components are constrained:
#   BC1  ``u_r^(f) = u_r^(m)``               (cos sector)
#   BC2  ``-(sigma_rr^(m) + P^(f)) = 0``     (cos sector)
#   BC3  ``sigma_r_theta^(m) = 0``           (sin sector)
#   BC4  ``sigma_rz^(m) = 0``                (cos sector)
# The fluid imposes no constraint on ``u_z`` (inviscid: no axial
# drag) or ``u_theta`` (inviscid: no azimuthal drag). Two of the
# six state-vector rows at r = a are unused; the corresponding
# fluid amplitude ``A`` only enters BC1 and BC2.
#
# -----
# Substep G'.a.2 -- Mode-amplitude to state-vector matrix E(r) (6x6)
# -----
#
# Within a single uniform layer (or the formation half-space),
# ``v(r)`` is a linear combination of the six wave amplitudes
# ``c = (B_I, B_K, C_I, C_K, D_I, D_K)^T``:
#
#       v(r) = E(r) c
#
# The 6x6 matrix ``E(r)`` is obtained by transcribing substep
# F.2.a.2 (displacements) and F.2.a.3 (stresses) at radius r,
# with each column corresponding to one wave amplitude:
#
#                | (B_I-col)  (B_K-col)  (C_I-col)  (C_K-col)  (D_I-col)  (D_K-col) |
#       E(r) =   |    u_r entries from each amplitude                          |
#                |    u_z entries (column D_I, D_K vanish for u_z)             |
#                |    u_theta entries (column C_I, C_K vanish for u_theta)     |
#                |    sigma_rr entries (D_I, D_K contribute via 2 mu * d_theta)|
#                |    sigma_rz entries (D_I, D_K vanish for sigma_rz)          |
#                |    sigma_r_theta entries                                    |
#
# Sparsity pattern (zero entries pinned by G'.b.1 tests):
#   * D_I, D_K cols of u_z and sigma_rz: zero (SH has no z-coupling
#     at n=1 cos sector).
#   * C_I, C_K cols of u_theta and sigma_r_theta: zero in the cos-
#     sector convention, but the F.2.a.6 erratum is that the d_theta
#     cross-coupling makes some entries non-zero in the assembled
#     determinant. CAVEAT: the column conventions here use the
#     state-vector form (separate cos / sin sectors stripped), so
#     the C-cols of u_theta / sigma_r_theta may be non-zero --
#     pin during G'.b.1 implementation against F.2.b/c row builders.
#
# Phase rescale (F.2.a.5): row-by-i for z-derivative-bearing rows
# (u_z and sigma_rz); col-by-(-i) for the SV columns (C_I, C_K) and
# possibly the SH columns (D_I, D_K) -- pin during the per-element
# oracle test. The post-rescale E(r) is real-valued in the bound
# regime.
#
# Reference for the per-row entries: F.2's existing row builders
# at r=a (rows 1, 2, 3, 4) and r=b (rows 5, 6, 7, 8, 9, 10) cover
# all six state-vector components (six rows total at r=b). The
# G'.b.1 per-element oracle reads each row off F.2's row builders
# with explicit BC sign factors:
#   * Rows 1, 2, 5, 6, 7, 8 are cos-sector (u_r, sigma_rr, u_r,
#     u_theta, u_z, sigma_rr).
#   * Rows 3, 4, 9, 10 are sin/cos sigma rows (sigma_r_theta,
#     sigma_rz, sigma_r_theta, sigma_rz).
# The state vector rows (u_r, u_z, u_theta, sigma_rr, sigma_rz,
# sigma_r_theta) map to F.2 row builders as listed above.
#
# -----
# Substep G'.a.3 -- Layer propagator P_j(r_outer | r_inner)  (6x6)
# -----
#
# Within layer j with mode amplitudes c_j (constant across the
# annulus), the 6-vector at the inner and outer radii are
#       v_j(r_inner_j) = E_j(r_inner_j) c_j
#       v_j(r_outer_j) = E_j(r_outer_j) c_j
# so the 6-vector at the outer radius is related to the 6-vector
# at the inner radius by the propagator
#
#       P_j(r_outer_j | r_inner_j) = E_j(r_outer_j) E_j(r_inner_j)^{-1}
#
# i.e. ``v_j(r_outer_j) = P_j v_j(r_inner_j)``.
#
# Continuity of the 6-vector across each layer/layer interface
# (substep G'.a.1) lets the per-layer propagators be composed
# directly:
#
#       v(r_N_outer) = P_N P_{N-1} ... P_2 P_1  v(r_0_inner)
#
# Same algebraic structure as G.a.3 with the dimension change
# 4 -> 6.
#
# -----
# Substep G'.a.4 -- Boundary conditions and 10x10 modal determinant
# -----
#
# Three sets of BCs assemble the dispersion relation:
#
# (i) Fluid side at r = a (4 BCs).  The fluid holds 1 amplitude
# ``A`` from the regular-at-axis form. Four of the six state-
# vector rows at r = a are picked: u_r, sigma_rr, sigma_r_theta,
# sigma_rz. The u_z and u_theta rows at r = a are unused.
#
# (ii) Formation half-space at r = b (6 BCs).  Three K-flavour
# amplitudes (B_form_K, C_form_K, D_form_K). All six state-vector
# rows at r = b are picked (full continuity).
#
# (iii) Layer/layer continuity at r = b_j for j=1..N-1: handled
# implicitly by the propagator chain P_total = P_N ... P_1.
#
# Assembled 10x10 modal determinant. Stack the BCs into a square
# system. Unknowns are
#       (A, B_form_K, C_form_K, D_form_K)^T            (4 amps)
#   plus the 6 innermost-layer amplitudes (eliminated via
#   ``v_1(a) = E_1(a) c_1``):
# Effective unknowns: ``(A, c_1 [6], B_form_K, C_form_K, D_form_K)``
# = 1 + 6 + 3 = 10. Effective equations: 4 BCs at r=a + 6 BCs at
# r=b = 10. Square 10x10 system.
#
# Concretely:
#
#       M = [[ B_fluid_at_a (4 BC rows; layer cols use E_1(a)
#              [4 of 6 state rows];
#              fluid col carries A's contribution to BC1 / BC2;
#              formation cols are zero at r=a),
#            ],
#            [ B_outer_at_b (6 BC rows; layer cols use
#              v_at_b = P_total @ E_1(a) [all 6 state rows];
#              formation cols use E_form(b) K-flavour cols, negated)
#            ]]
#
# Net 10 x 10. At N=1, P_1 = E_1(b) E_1(a)^{-1}, so
# P_total @ E_1(a) = E_1(b), and the assembly drops to F.2's
# hand-coded ``_modal_determinant_n1_layered`` form. This is the
# floating-point oracle for G'.c (substep G'.a.6).
#
# -----
# Substep G'.a.5 -- Numerical conditioning (kz * thickness)
# -----
#
# Same disparate-magnitude story as G.a.5. The 6x6 E(r) has
# displacement rows (u_r, u_z, u_theta) of magnitude ~ O(I_n) and
# stress rows (sigma_rr, sigma_rz, sigma_r_theta) of magnitude
# ~ O(mu * I_n) ~ 1e10 * O(1). The condition number of E is
# dominated by this disparity, giving cond(E) ~ mu ~ 1e10.
#
# Implication for the round-trip / composition oracles in G'.b.2:
# raw matrix-equality tests P @ P^{-1} ~ eye(6) hit the same
# spurious off-diagonal residuals (~1e-6) as G.b.2. Mitigation:
# state-vector form, ``P(a|b) P(b|a) v ~ v`` for a physical state
# vector ``v`` (displacements ~ O(1), stresses ~ O(mu)). The
# exact same lesson called out in the G.b.2 commit
# (123e634); G'.b.2 inherits the pattern.
#
# Layer-thickness / frequency conditioning: same as G.a.5.
# Typical cased-hole geometries (h ~ 5 cm, sonic band 1-15 kHz)
# give cond(E) ~ 400 from the Bessel-magnitude ratio, still well
# within double precision. Knopoff / Kennett delta-matrix
# alternatives deferred (out of scope).
#
# -----
# Substep G'.a.6 -- Single-layer collapse identity (G'.b -> F.2)
# -----
#
# When ``len(layers) == 1`` the propagator path's modal determinant
# matches F.2's hand-coded ``_modal_determinant_n1_layered`` to
# floating-point precision (no overall scale factor; same brentq
# root in k_z). Direct mirror of G.a.6: at N=1, P_1 = E_1(b)
# E_1(a)^{-1} so P_1 @ E_1(a) = E_1(b), the F.2 form.
#
# Same scaling identity holds at ``len(layers) == 0`` (collapse
# to ``_modal_determinant_n1`` via the formation-half-space-only
# assembly, with the borehole-wall interface at r=a now between
# fluid and formation directly).
#
# -----
# Substep G'.a.7 -- Self-check protocol
# -----
#
# Each implementation step lands with one or more of the
# following oracles (test names mirror G.a.7 / F.1.a.6 / H.a.7):
#
# (a) Identity propagator: r_outer = r_inner -> P = eye(6) to
#     floating-point precision. Tests E(r)^{-1} E(r) round-trip.
# (b) Composition: P(r3 | r1) = P(r3 | r2) @ P(r2 | r1) for any
#     intermediate r2. Tests the propagator group law.
# (c) State-vector continuity: pick c, verify that
#     v(r2) = P(r2 | r1) v(r1) matches E(r2) c directly.
# (d) Zero-layer collapse (G'.c): the propagator-matrix
#     determinant matches ``_modal_determinant_n1`` (within an
#     overall k_z-independent scale).
# (e) Single-layer collapse (G'.c): matches
#     ``_modal_determinant_n1_layered`` (same overall-scale
#     identity).
# (f) Order-matters at N=2 (G'.c / G'.d): swapping (L_a, L_b) ->
#     (L_b, L_a) gives a different slowness curve.
# (g) Cement-bond physics for flexural (G'.e): stiffer cement
#     gives a flexural slowness closer to the formation V_S; soft
#     cement gives slowness closer to the casing-extensional speed.


# =====================================================================
# Plan item G'.b.1 -- mode-amplitude-to-state-vector matrix E(r) at n=1
# =====================================================================
#
# 6x6 matrix mapping a uniform layer's wave amplitudes
# ``c = (B_I, B_K, C_I, C_K, D_I, D_K)`` to the six-component
# state vector ``v(r) = (u_r, u_z, u_theta, sigma_rr, sigma_rz,
# sigma_r_theta)`` at azimuthal order n=1. Direct transcription
# of substep G'.a.2 with the substep-F.2.a.5 phase rescale
# absorbed.
#
# The 36 entries below are read off F.2's existing 10 row
# builders (rows 1-4 at r=a + rows 5-10 at r=b) with explicit BC
# sign factors per the F.2 substeps. Each entry has a per-element
# oracle in the G'.b.1 tests.


def _layer_e_matrix_n1(
    kz: float,
    omega: float,
    *,
    vp: float,
    vs: float,
    rho: float,
    r: float,
) -> np.ndarray:
    r"""
    6x6 mode-amplitude-to-state-vector matrix ``E(r)`` for a
    uniform isotropic-elastic layer at azimuthal order n=1
    (dipole flexural).

    Maps the six wave amplitudes
    ``c = (B_I, B_K, C_I, C_K, D_I, D_K)`` to the state vector
    ``v(r) = (u_r, u_z, u_theta, sigma_rr, sigma_rz,
    sigma_r_theta)`` via ``v(r) = E(r) c``, with the substep-
    F.2.a.5 phase rescale (row * i for u_z and sigma_rz; col *
    -i for the SV columns C_I, C_K) absorbed so every entry is
    real-valued in the bound regime.

    Six wave families: I/K flavours of the P scalar potential
    ``phi = B_I I_1(p r) + B_K K_1(p r)``, the SV vector
    potential theta-component ``psi_theta = C_I I_1(s r) + C_K
    K_1(s r)``, and the SH vector potential z-component
    ``psi_z = D_I I_1(s r) + D_K K_1(s r)``. Six state-vector
    components corresponding to the six BC rows whose continuity
    is enforced at every layer/layer interface (substep G'.a.1).

    Sparsity pattern (pinned by G'.b.1 tests):

    * Row 1 (``u_z``) cols 4, 5 (``D_I``, ``D_K``) are zero --
      SH potential ``psi_z`` has no axial-displacement
      contribution at n=1 (the curl-z component of psi_z theta-
      hat does not enter ``u_z = (curl psi)_z + d_z phi``).
    * Row 2 (``u_theta``) cols 2, 3 (``C_I``, ``C_K``) are zero
      -- SV potential ``psi_theta`` does not contribute to
      ``u_theta`` (only to ``u_r`` and ``u_z``).
    * Row 4 (``sigma_rz``) cols 4, 5 (``D_I``, ``D_K``) are
      non-zero (D contributes via ``-k_z mu D K_1 / r`` cross-
      coupling) -- this is the F.2.a.6 erratum: cos / sin
      sectors are NOT block-diagonal at n=1.

    Parameters
    ----------
    kz, omega : float
        Trial axial wavenumber (rad / m) and angular frequency
        (rad / s).
    vp, vs, rho : float
        Layer P / S velocity (m/s) and density (kg/m^3). Must
        satisfy ``vp > vs > 0`` and ``rho > 0``.
    r : float
        Radius at which to evaluate ``E(r)`` (m). Must be
        positive.

    Returns
    -------
    ndarray, shape (6, 6)
        ``E(r)`` post-rescale. Real-valued in the bound regime;
        ``NaN``-filled outside the bound regime (``kz`` below the
        layer's bound floor ``omega / V_S``).

    See Also
    --------
    _layer_e_matrix_n0 : The n=0 (Stoneley) sister, 4x4.
    _layer_propagator_n1 : G'.b.2 follow-up using
        ``E(r_outer) E(r_inner)^{-1}`` to map the state vector
        across a layer at n=1.
    """
    p2 = kz * kz - (omega / vp) ** 2
    s2 = kz * kz - (omega / vs) ** 2
    if p2 <= 0.0 or s2 <= 0.0 or r <= 0.0:
        return np.full((6, 6), np.nan)
    p = float(np.sqrt(p2))
    s = float(np.sqrt(s2))
    pr = p * r
    sr = s * r
    I0_p, I1_p = _i0_i1(pr)
    K0_p, K1_p = _k0_k1(pr)
    I0_s, I1_s = _i0_i1(sr)
    K0_s, K1_s = _k0_k1(sr)
    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    E = np.zeros((6, 6))
    # Row 0: u_r = d_r phi + (1/r) d_theta psi_z - d_z psi_theta.
    # Post-rescale: col*(-i) on C cols cancels the -i k_z factor.
    E[0, 0] = +p * I0_p - I1_p / r  # B_I
    E[0, 1] = -p * K0_p - K1_p / r  # B_K
    E[0, 2] = -kz * I1_s  # C_I
    E[0, 3] = -kz * K1_s  # C_K
    E[0, 4] = +I1_s / r  # D_I
    E[0, 5] = +K1_s / r  # D_K
    # Row 1: u_z = i k_z phi + (1/r) d_r(r psi_theta).
    # Post-rescale: row*i on B cols flips +i k_z to -k_z; col*(-i)
    # on C cols leaves +s I_0 / -s K_0.
    E[1, 0] = -kz * I1_p  # B_I
    E[1, 1] = -kz * K1_p  # B_K
    E[1, 2] = +s * I0_s  # C_I
    E[1, 3] = -s * K0_s  # C_K
    E[1, 4] = 0.0  # D_I (SH no u_z)
    E[1, 5] = 0.0  # D_K
    # Row 2: u_theta = (1/r) d_theta phi - d_r psi_z. Sin sector;
    # no row scaling (no z-derivative). C cols zero (SV no u_theta).
    E[2, 0] = -I1_p / r  # B_I
    E[2, 1] = -K1_p / r  # B_K
    E[2, 2] = 0.0  # C_I (SV no u_theta)
    E[2, 3] = 0.0  # C_K
    E[2, 4] = -s * I0_s + I1_s / r  # D_I (sign flip on s I_0)
    E[2, 5] = +s * K0_s + K1_s / r  # D_K
    # Row 3: sigma_rr = lambda div u + 2 mu d_r u_r. No row scaling.
    # col*(-i) on C cols.
    E[3, 0] = +mu * (
        two_kz2_minus_kS2 * I1_p - 2.0 * p * I0_p / r + 4.0 * I1_p / (r * r)
    )  # B_I
    E[3, 1] = +mu * (
        two_kz2_minus_kS2 * K1_p + 2.0 * p * K0_p / r + 4.0 * K1_p / (r * r)
    )  # B_K
    E[3, 2] = -2.0 * kz * mu * (s * I0_s - I1_s / r)  # C_I
    E[3, 3] = +2.0 * kz * mu * (s * K0_s + K1_s / r)  # C_K
    E[3, 4] = +2.0 * mu * (s * I0_s / r - 2.0 * I1_s / (r * r))  # D_I
    E[3, 5] = -2.0 * mu * (s * K0_s / r + 2.0 * K1_s / (r * r))  # D_K
    # Row 4: sigma_rz = mu (d_z u_r + d_r u_z). Post-rescale: row*i
    # on B cols (z-derivative-bearing); col*(-i) on C cols.
    E[4, 0] = -2.0 * kz * mu * (p * I0_p - I1_p / r)  # B_I
    E[4, 1] = +2.0 * kz * mu * (p * K0_p + K1_p / r)  # B_K
    E[4, 2] = +mu * two_kz2_minus_kS2 * I1_s  # C_I
    E[4, 3] = +mu * two_kz2_minus_kS2 * K1_s  # C_K
    # D cols: SH cross-couples to sigma_rz via -k_z mu D K_1/r at n=1
    # (F.2.a.6 erratum: cos/sin sectors are NOT block-diagonal).
    E[4, 4] = -kz * mu * I1_s / r  # D_I
    E[4, 5] = -kz * mu * K1_s / r  # D_K
    # Row 5: sigma_r_theta = mu [d_r(u_theta) + (1/r) d_theta u_r
    #                            - u_theta/r]. Sin sector; no row
    # scaling.
    E[5, 0] = +2.0 * mu * (-p * I0_p / r + 2.0 * I1_p / (r * r))  # B_I
    E[5, 1] = +2.0 * mu * (+p * K0_p / r + 2.0 * K1_p / (r * r))  # B_K
    E[5, 2] = +kz * mu * I1_s / r  # C_I
    E[5, 3] = +kz * mu * K1_s / r  # C_K
    E[5, 4] = -mu * (s * s * I1_s - 2.0 * s * I0_s / r + 4.0 * I1_s / (r * r))  # D_I
    E[5, 5] = -mu * (s * s * K1_s + 2.0 * s * K0_s / r + 4.0 * K1_s / (r * r))  # D_K
    return E


# =====================================================================
# Plan item G'.b.2 -- per-layer propagator P(r_outer | r_inner) at n=1
# =====================================================================
#
# Sister of G.b.2 at azimuthal order n=1 with the dimension change
# 4 -> 6. Wraps two evaluations of ``_layer_e_matrix_n1`` into the
# layer propagator P_j = E(r_outer) E(r_inner)^{-1} (substep
# G'.a.3). Uses ``np.linalg.solve`` rather than explicit ``inv``
# for better conditioning.
#
# Round-trip / composition oracles in the G'.b.2 tests use the
# state-vector form (``P(a|b) P(b|a) v ~ v``) to avoid the
# spurious ~1e-6 off-diagonal residuals of raw matrix equality
# tests at ``cond(E_n1) ~ mu ~ 1e10`` -- the same lesson
# documented in the G.b.2 commit (123e634).


def _layer_propagator_n1(
    kz: float,
    omega: float,
    *,
    vp: float,
    vs: float,
    rho: float,
    r_inner: float,
    r_outer: float,
) -> np.ndarray:
    r"""
    6x6 layer propagator ``P(r_outer | r_inner)`` mapping the
    n=1 state vector ``v(r) = (u_r, u_z, u_theta, sigma_rr,
    sigma_rz, sigma_r_theta)`` from ``r_inner`` to ``r_outer``
    within a uniform isotropic-elastic layer.

    Implements substep G'.a.3:

    .. math::
        P_j(r_{\text{outer}} \mid r_{\text{inner}}) =
        E(r_{\text{outer}}) \, E(r_{\text{inner}})^{-1}

    with ``E`` from :func:`_layer_e_matrix_n1`.

    Parameters
    ----------
    kz, omega : float
        Trial axial wavenumber (rad / m) and angular frequency
        (rad / s).
    vp, vs, rho : float
        Layer P / S velocity (m/s) and density (kg/m^3). Must
        satisfy ``vp > vs > 0`` and ``rho > 0``.
    r_inner, r_outer : float
        Inner and outer radii of the layer (m). May be in
        either order; the round-trip identity
        ``P(b|a) P(a|b) v ~ v`` is verified in the tests.
        ``r_inner == r_outer`` returns ``eye(6)``.

    Returns
    -------
    ndarray, shape (6, 6)
        The layer propagator. Real-valued in the bound regime;
        ``NaN``-filled outside the bound regime (propagated from
        :func:`_layer_e_matrix_n1`).

    See Also
    --------
    _layer_e_matrix_n1 : G'.b.1 helper that this function calls
        twice (at ``r_inner`` and ``r_outer``).
    _layer_propagator_n0 : The n=0 (Stoneley) sister, 4x4.
    """
    if r_inner == r_outer:
        return np.eye(6)
    E_inner = _layer_e_matrix_n1(
        kz=kz,
        omega=omega,
        vp=vp,
        vs=vs,
        rho=rho,
        r=r_inner,
    )
    E_outer = _layer_e_matrix_n1(
        kz=kz,
        omega=omega,
        vp=vp,
        vs=vs,
        rho=rho,
        r=r_outer,
    )
    if not (np.all(np.isfinite(E_inner)) and np.all(np.isfinite(E_outer))):
        return np.full((6, 6), np.nan)
    return np.linalg.solve(E_inner.T, E_outer.T).T


# =====================================================================
# Plan item G'.c -- stacked modal determinant at n=1 (10x10)
# =====================================================================
#
# Sister of G.c at azimuthal order n=1. Stacks the 6x6 propagators
# (G'.b.2) into the same 10x10 column packing as F.2's hand-coded
# ``_modal_determinant_n1_layered`` so the N=1 collapse gives a
# clean numerical-equality match (not just a shared brentq root).
#
# Column packing (matching F.2.a.4 exactly):
#   [A | B_I, B_K, C_I, C_K | B_form, C_form | D_I, D_K | D_form]
#
# The layer's six amplitudes c_1 = (B_I, B_K, C_I, C_K, D_I, D_K)
# are split across cols 1-4 (P/SV) and cols 7-8 (SH); the
# formation's three K-flavour amplitudes are in cols 5-6 (P/SV)
# and col 9 (SH).
#
# State-row to BC-row mapping at r=a (4 BCs, fluid-solid):
#   BC1 (u_r continuity, layer negated)        -> state row 0 (u_r)
#   BC2 (-(sigma_rr + P^f), layer negated)     -> state row 3 (sigma_rr)
#   BC3 (sigma_rtheta = 0, layer positive)     -> state row 5 (sigma_rtheta)
#   BC4 (sigma_rz = 0, layer positive)         -> state row 4 (sigma_rz)
#
# State-row to BC-row mapping at r=b (6 BCs, layer-formation):
#   BC5  (u_r continuity)                      -> state row 0
#   BC6  (u_theta continuity)                  -> state row 2
#   BC7  (u_z continuity)                      -> state row 1
#   BC8  (sigma_rr continuity)                 -> state row 3
#   BC9  (sigma_rtheta continuity)             -> state row 5
#   BC10 (sigma_rz continuity)                 -> state row 4
#
# All r=b rows carry m - s convention (layer cols positive,
# formation cols negative).


def _modal_determinant_n1_cased(
    kz: float,
    omega: float,
    *,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    layers: tuple[BoreholeLayer, ...],
) -> float:
    r"""
    10x10 dipole-flexural modal determinant for a borehole with N
    annular layers between fluid and formation, slow-formation
    regime.

    Generalises :func:`_modal_determinant_n1_layered` (N=1, hand-
    coded F.2) to arbitrary ``N >= 1`` via the Thomson-Haskell-
    style 6x6 propagator chain ``P_total = P_N ... P_2 P_1``
    (substep G'.a.3). The 10x10 final form has the same column
    structure as F.2.a.4 regardless of N; the propagator chain
    folds the layer-internal amplitudes ``c_2, ..., c_N`` out via
    radial continuity at every layer/layer interface.

    Reduces to :func:`_modal_determinant_n1_layered` at N=1 to
    floating-point precision (no extra scale factor; same brentq
    root in ``k_z``). The floating-point oracle for this routine
    is the multi-frequency
    ``flexural_dispersion_layered(layers=(layer,))`` regression
    in :func:`flexural_dispersion_layered` (G'.d).

    Parameters
    ----------
    kz, omega, vp, vs, rho, vf, rho_f, a : float
        Same as :func:`_modal_determinant_n1_layered`. ``vp``,
        ``vs``, ``rho`` are the formation half-space (outermost
        side); ``vf``, ``rho_f`` are the borehole fluid.
    layers : tuple of BoreholeLayer
        Annular layer stack ordered inside-out: ``layers[0]``
        adjacent to fluid, ``layers[-1]`` adjacent to formation.
        Must have at least one layer (``len(layers) >= 1``); the
        unlayered case dispatches to :func:`_modal_determinant_n1`
        in :func:`flexural_dispersion_layered`. Slow-formation
        constraint ``layer.vs >= vs`` is enforced upstream by
        :func:`_validate_flexural_layers_stacked`.

    Returns
    -------
    float
        ``det(M)`` of the 10x10 cased-hole flexural modal matrix,
        real-valued in the bound regime. ``NaN`` outside the bound
        regime.
    """
    F_f_sq = kz * kz - (omega / vf) ** 2
    if F_f_sq <= 0.0:
        return float("nan")
    F_f = float(np.sqrt(F_f_sq))
    radii = [a]
    for L in layers:
        radii.append(radii[-1] + L.thickness)
    b = radii[-1]
    L1 = layers[0]
    E_1_a = _layer_e_matrix_n1(
        kz=kz,
        omega=omega,
        vp=L1.vp,
        vs=L1.vs,
        rho=L1.rho,
        r=a,
    )
    if not np.all(np.isfinite(E_1_a)):
        return float("nan")
    P_total = np.eye(6)
    for j, L in enumerate(layers):
        P_j = _layer_propagator_n1(
            kz=kz,
            omega=omega,
            vp=L.vp,
            vs=L.vs,
            rho=L.rho,
            r_inner=radii[j],
            r_outer=radii[j + 1],
        )
        if not np.all(np.isfinite(P_j)):
            return float("nan")
        P_total = P_j @ P_total
    E_form_b = _layer_e_matrix_n1(
        kz=kz,
        omega=omega,
        vp=vp,
        vs=vs,
        rho=rho,
        r=b,
    )
    if not np.all(np.isfinite(E_form_b)):
        return float("nan")
    v_at_b = P_total @ E_1_a
    I0_Ff_a = float(special.iv(0, F_f * a))
    I1_Ff_a = float(special.iv(1, F_f * a))

    # Build M[10, 10] in the F.2 column packing.
    # Layer-amplitude blocks: cols 1-4 (P/SV) and cols 7-8 (SH)
    # carry the layer's contribution; formation cols 5-6 and 9
    # carry the formation half-space K-flavour contribution.
    M = np.zeros((10, 10))

    # ---- Rows 0-3 at r=a (4 BCs, fluid-solid interface). ----
    # The layer cols use E_1(a); formation cols are zero at r=a.

    # Helper: copy E_1(a)'s state row to layer cols of M with sign.
    def _layer_at_a(state_row: int, sign: float) -> np.ndarray:
        """Returns 6 layer-amplitude entries in the order
        (B_I, B_K, C_I, C_K, D_I, D_K) for the given state row."""
        return sign * E_1_a[state_row, :]

    # Row 0: BC1, u_r^(f)(a) - u_r^(m)(a) = 0.  Layer negated.
    M[0, 0] = (F_f * I0_Ff_a - I1_Ff_a / a) / (rho_f * omega**2)
    layer0 = _layer_at_a(0, -1.0)  # u_r row, negated
    M[0, 1:5] = layer0[0:4]  # B_I, B_K, C_I, C_K
    M[0, 7:9] = layer0[4:6]  # D_I, D_K
    # Row 1: BC2, -(sigma_rr^(m)(a) + P^(f)(a)) = 0.  Layer negated;
    # fluid pressure is -A I_0(F_f a) per F.2 row 2.
    M[1, 0] = -I1_Ff_a
    layer1 = _layer_at_a(3, -1.0)  # sigma_rr row, negated
    M[1, 1:5] = layer1[0:4]
    M[1, 7:9] = layer1[4:6]
    # Row 2: BC3, sigma_rtheta^(m)(a) = 0.  Layer positive (no
    # fluid contribution; fluid is inviscid).
    layer2 = _layer_at_a(5, +1.0)  # sigma_rtheta row
    M[2, 1:5] = layer2[0:4]
    M[2, 7:9] = layer2[4:6]
    # Row 3: BC4, sigma_rz^(m)(a) = 0.  Layer positive.
    layer3 = _layer_at_a(4, +1.0)  # sigma_rz row
    M[3, 1:5] = layer3[0:4]
    M[3, 7:9] = layer3[4:6]

    # ---- Rows 4-9 at r=b (6 BCs, layer-formation continuity). ----
    # Layer cols use v_at_b (= P_total @ E_1(a)); formation cols
    # use E_form(b)'s K-flavour cols (1, 3, 5 in our state-vector
    # E indexing) negated.

    def _layer_at_b(state_row: int) -> np.ndarray:
        """6 layer-amplitude entries (positive, m - s convention)
        from v_at_b for the given state row."""
        return v_at_b[state_row, :]

    def _formation_at_b(state_row: int) -> tuple[float, float, float]:
        """3 formation K-flavour entries (B_form_K = E[:,1],
        C_form_K = E[:,3], D_form_K = E[:,5]); negated for the
        m - s subtraction convention. The destination matrix ``M`` is
        real-dtype and the values are real-valued by physics, so the
        ``.real`` extraction is safe and silences ``ComplexWarning``."""
        return (
            -float(E_form_b[state_row, 1].real),  # B_form
            -float(E_form_b[state_row, 3].real),  # C_form
            -float(E_form_b[state_row, 5].real),  # D_form
        )

    # State-row mapping at r=b: BC5..10 -> state rows 0, 2, 1, 3, 5, 4.
    bc_to_state_b = [0, 2, 1, 3, 5, 4]
    for i_bc, state_row in enumerate(bc_to_state_b):
        i_M = 4 + i_bc
        layer = _layer_at_b(state_row)
        M[i_M, 1:5] = layer[0:4]
        M[i_M, 7:9] = layer[4:6]
        b_form, c_form, d_form = _formation_at_b(state_row)
        M[i_M, 5] = b_form
        M[i_M, 6] = c_form
        M[i_M, 9] = d_form

    return float(np.linalg.det(M))


# =====================================================================
# Substep G''.a -- math scaffolding for the cased-hole multi-layer
#                  propagator-matrix determinant (n=2 quadrupole)
# =====================================================================
#
# Sister of substeps G.a (n=0) and G'.a (n=1) at azimuthal order
# n=2. Generalises the existing unlayered ``_modal_determinant_n2``
# (4x4 single-interface form) to N stacked annular layers via a
# Thomson-Haskell-style 6x6 propagator matrix in cylindrical
# geometry. Comments-only block; the helpers themselves land in
# plan items G''.b (per-layer propagator), G''.c (assembly), and
# G''.d (public-API hook). See
# ``docs/plans/cylindrical_biot_G_pp.md`` for the full sub-plan.
#
# Sign / Bessel / phase conventions are the same as the existing
# ``_modal_determinant_n2`` substep block (Tang & Cheng 2004
# sect. 2.5; Kurkjian-Chang 1986). Per-region scalar / vector
# potentials at azimuthal order n=2:
#   phi          = [B_I I_2(p r) + B_K K_2(p r)]      cos(2 theta)
#   psi_theta    = [C_I I_2(s r) + C_K K_2(s r)]      cos(2 theta)
#   psi_z        = [D_I I_2(s r) + D_K K_2(s r)]      sin(2 theta)
# Six amplitudes per layer, same as n=1; the structural difference
# from G'.a is the Bessel-function index shift
# ``(I_{n-1}, I_n) -> (I_1, I_2)`` and the explicit n=2 azimuthal
# factors that come out of ``d_theta cos(n theta) = -n sin(n
# theta)``.
#
# The N=0 limit reduces to the unlayered ``_modal_determinant_n2``
# (4x4) via dispatch in ``quadrupole_dispersion_layered``. The
# N=1 limit reduces to the same 4x4 form via ``layer = formation``
# parameter equality, anchored as the floating-point oracle for
# G''.c (root-level match, since no F.3-equivalent hand-coded
# 10x10 single-layer form exists).
#
# -----
# Substep G''.a.1 -- State vector at the quadrupole order n=2
# -----
#
# Within any uniform elastic layer (or the formation half-space),
# the radial-dependent part of the displacement / stress field at
# n=2 reduces to the same 6-component state vector as n=1:
#
#       v(r) = ( u_r(r),  u_z(r),  u_theta(r),
#                sigma_rr(r),  sigma_rz(r),  sigma_r_theta(r) )^T
#
# Six components (same as n>=1) because the d_theta operations at
# any n>=1 make ``u_theta`` and ``sigma_r_theta`` non-trivial.
# The cos / sin azimuthal factors (cos(2 theta) for u_r, u_z,
# sigma_rr, sigma_rz; sin(2 theta) for u_theta, sigma_r_theta)
# have been stripped per the existing n=2 substep convention.
#
# Boundary conditions:
#   * At fluid-solid interface r = a (4 BCs): u_r continuity,
#     -(sigma_rr + P^f) = 0, sigma_r_theta = 0, sigma_rz = 0.
#     The inviscid fluid imposes no constraint on u_z or
#     u_theta -- two of the six state-vector rows at r=a are
#     unused.
#   * At each layer/layer or layer/formation interface r = b_j
#     (6 BCs): full state-vector continuity ``v(b_j^-) = v(b_j^+)``.
#
# Same BC structure as n=1 (G'.a.1); the only change is the
# Bessel index in each entry of the state-vector / amplitude
# expressions.
#
# -----
# Substep G''.a.2 -- 6x6 mode-amplitude-to-state-vector matrix E_n2(r)
# -----
#
# Within a single uniform layer (or the formation half-space),
# ``v(r) = E_n2(r) c`` with
# ``c = (B_I, B_K, C_I, C_K, D_I, D_K)^T``. The 6x6 matrix
# E_n2(r) is obtained by transcribing the field-from-potential
# expansions at n=2 with the substep-F.2.a.5 phase rescale
# (row * i for u_z and sigma_rz; col * -i for SV columns C_I /
# C_K) absorbed.
#
# **Bessel-derivative identities at n=2** (from the existing
# ``_modal_determinant_n2`` substep block):
#
#   d_r I_2(p r) = p I_1(p r) - 2 I_2(p r) / r
#   d_r K_2(p r) = -p K_1(p r) - 2 K_2(p r) / r
#   (1/r) d_r [r I_2(s r)] = s I_1(s r) - I_2(s r) / r
#   (1/r) d_r [r K_2(s r)] = -s K_1(s r) - K_2(s r) / r
#
# The Bessel-index shift ``(I_1, I_2)`` (vs ``(I_0, I_1)`` at
# n=1) propagates through every entry; the ``-(n-1) F_n / r``
# correction term ``-2 F_2 / r`` (vs ``-F_1 / r`` at n=1)
# appears in every Bessel derivative.
#
# **Sparsity pattern at n=2** (same as n=1; pinned by G''.b.1
# tests):
#   * Row 1 (``u_z``) cols 4, 5 (``D_I``, ``D_K``): 0.
#     SH potential ``psi_z`` doesn't contribute to ``u_z`` at
#     any n>=1.
#   * Row 2 (``u_theta``) cols 2, 3 (``C_I``, ``C_K``): 0.
#     SV potential ``psi_theta`` doesn't contribute to
#     ``u_theta`` at any n>=1.
#
# **Explicit n=2 factors** (transcribed from
# ``_modal_determinant_n2``):
#   * ``2 n (n + 1) = 12`` in the ``B_K K_2 / r^2`` term of
#     sigma_rr (and in the I-flavour analog).
#   * ``n^2 - 1 = 3`` in the SV column of sigma_rz (multiplying
#     the ``K_2(s r) / r^2`` term).
#   * ``n = 2`` factor from ``d_theta sin(n theta) = +n cos(n
#     theta)`` and ``d_theta cos(n theta) = -n sin(n theta)``.
#
# Reference for the per-row entries: the existing
# ``_modal_determinant_n2`` formulas (transcribed in the
# function's docstring) cover the K-flavour columns at the
# single-interface r=a; the I-flavour columns are obtained via
# the substep-F.1.a.2 sign-flip pattern (carried over from F.2
# and the n=1 case in G'.a.2). G''.b.1 implementation
# transcribes both flavours into the 6x6 matrix.
#
# -----
# Substep G''.a.3 -- Layer propagator P_j(r_outer | r_inner) (6x6)
# -----
#
# Identical to substep G'.a.3 with the dimension change baked in
# via E_n2(r) (now using I_1/I_2 and K_1/K_2 Bessels instead of
# I_0/I_1 and K_0/K_1). Composition law:
#
#       P_j(r_outer_j | r_inner_j) = E_n2_j(r_outer_j) E_n2_j(r_inner_j)^{-1}
#
# Continuity of the 6-vector across each layer/layer interface
# (G''.a.1) lets the per-layer propagators be composed:
#
#       v(r_N_outer) = P_N P_{N-1} ... P_2 P_1  v(r_0_inner)
#
# Same algebraic structure as G'.a.3.
#
# -----
# Substep G''.a.4 -- Boundary conditions and 10x10 modal determinant
# -----
#
# Same BC bookkeeping as G'.a.4 (4 BCs at r=a + 6 BCs at r=b
# = 10 BC rows; 1 fluid + 6 innermost-layer + 3 formation =
# 10 unknowns). Final 10x10 form has the same column packing as
# the F.2 / G' convention:
#
#   [A | B_I, B_K, C_I, C_K | B_form, C_form | D_I, D_K | D_form]
#
# State-row to BC-row mapping at r=a (4 BCs):
#   BC1 (u_r continuity, layer negated)        -> state row 0 (u_r)
#   BC2 (-(sigma_rr + P^f), layer negated)     -> state row 3 (sigma_rr)
#   BC3 (sigma_r_theta = 0, layer positive)    -> state row 5 (sigma_r_theta)
#   BC4 (sigma_rz = 0, layer positive)         -> state row 4 (sigma_rz)
#
# State-row to BC-row mapping at r=b (6 BCs, m - s convention):
#   BC5  (u_r continuity)                      -> state row 0
#   BC6  (u_theta continuity)                  -> state row 2
#   BC7  (u_z continuity)                      -> state row 1
#   BC8  (sigma_rr continuity)                 -> state row 3
#   BC9  (sigma_r_theta continuity)            -> state row 5
#   BC10 (sigma_rz continuity)                 -> state row 4
#
# Identical to G'.a.4; the only change is that v(b) = P_total @
# E_n2(a) c_1 carries the n=2 Bessel functions.
#
# **Layer = formation collapse identity at N=1 (G''.a.6 oracle)**:
# at layer params equal to formation params, the propagator-
# chain G''.c assembly and the unlayered ``_modal_determinant_n2``
# share the same brentq root in k_z. The determinant magnitudes
# differ (different overall scale factors; the cased path is
# 10x10 vs the unlayered 4x4) but the root is invariant. This
# is the floating-point oracle for G''.c in the absence of an
# F.3-equivalent hand-coded 10x10 single-layer form.
#
# -----
# Substep G''.a.5 -- Numerical conditioning (kz * thickness, n=2)
# -----
#
# Same disparate-magnitude story as G.a.5 / G'.a.5: cond(E_n2) ~
# mu ~ 1e10. The state-vector form for round-trip oracles in
# G''.b.2 mitigates the raw-matrix-equality footgun (G.b.2 /
# G'.b.2 lesson; see commit 123e634).
#
# **n=2-specific concern**: ``I_2(p r)`` is small near ``p r =
# 0`` (proportional to ``(p r)^2 / 8``), making the I-flavour
# columns of E_n2(r) disproportionately small near small radii.
# Conditioning of ``E_n2(r_inner)`` may degrade slightly compared
# to n=0 / n=1 at low frequencies. Likely still within double
# precision for typical cased-hole geometries (kz * thickness ~
# O(1)); the G''.b.2 round-trip oracle catches the practical
# impact.
#
# Layer-thickness / frequency conditioning: same as G.a.5.
# Knopoff / Kennett delta-matrix alternatives deferred (out of
# scope; only needed for ``kz * thickness > 30``).
#
# -----
# Substep G''.a.6 -- Layer = formation collapse identity (G''.c oracle)
# -----
#
# When ``len(layers) == 1`` AND ``layer.{vp, vs, rho} ==
# formation.{vp, vs, rho}``, the propagator-chain modal
# determinant and the unlayered ``_modal_determinant_n2`` share
# the same brentq root in k_z. The determinants themselves
# differ (10x10 propagator-chain form vs 4x4 unlayered form,
# different overall scale factors), so the oracle is at the
# **root level**: the cased determinant evaluated at the
# unlayered-recovered root is many orders of magnitude smaller
# than its value at ``kz * 1.005``.
#
# Same ``layer = formation`` identity holds for any N >= 1 with
# all layers carrying formation properties (master-plan G''
# validation bullet, exercised in G''.e).
#
# Difference from G.a.6 / G'.a.6: those plans had
# ``rtol=1e-10`` numerical-equality oracles via F.1 / F.2's
# hand-coded layered determinants. G'' has no F.3-equivalent,
# so the determinant-level numerical-equality oracle is
# replaced by the brentq-root-match oracle.
#
# -----
# Substep G''.a.7 -- Self-check protocol
# -----
#
# Each implementation step lands with one or more of the
# following oracles (test names mirror G.a.7 / G'.a.7):
#
# (a) Identity propagator: r_outer = r_inner -> P = eye(6) to
#     floating-point precision.
# (b) Composition: P(r3 | r1) = P(r3 | r2) @ P(r2 | r1) for any
#     intermediate r2.
# (c) State-vector continuity: pick c, verify v(r2) = P(r2 | r1)
#     v(r1) matches E_n2(r2) c directly.
# (d) Zero-layer collapse (G''.c): ``layers=()`` dispatches to
#     ``_modal_determinant_n2`` (4x4 unlayered form).
# (e) Layer = formation N=1 collapse (G''.c): cased determinant
#     vanishes at the unlayered brentq root.
# (f) Order-matters at N=2 (G''.c / G''.d).
# (g) LWD-quadrupole cement-bond physics (G''.e): stiffer cement
#     gives a slowness shift in a measurable direction relative
#     to softer cement at the same casing geometry.


# =====================================================================
# Plan item G''.b.1 -- mode-amplitude-to-state-vector matrix E_n2(r)
# =====================================================================
#
# 6x6 matrix mapping a uniform layer's wave amplitudes
# ``c = (B_I, B_K, C_I, C_K, D_I, D_K)`` to the six-component
# state vector ``v(r) = (u_r, u_z, u_theta, sigma_rr, sigma_rz,
# sigma_r_theta)`` at azimuthal order n=2 (quadrupole). Direct
# transcription of substep G''.a.2 with the substep-F.2.a.5
# phase rescale absorbed.
#
# Each entry is derived from the n=2 ansatz in G''.a (potentials
# ``phi = B I_2(p r) cos(2 theta)``, ``psi_theta = C I_2(s r)
# cos(2 theta)``, ``psi_z = D I_2(s r) sin(2 theta)``) plus the
# Bessel-derivative identities at n=2:
#
#   d_r I_2(p r) = p I_1(p r) - 2 I_2(p r) / r
#   d_r K_2(p r) = -p K_1(p r) - 2 K_2(p r) / r
#   (1/r) d_r [r I_2(s r)] = s I_1(s r) - I_2(s r) / r
#   (1/r) d_r [r K_2(s r)] = -s K_1(s r) - K_2(s r) / r
#
# K-flavour entries cross-check against the unlayered
# ``_modal_determinant_n2`` (which carries B / C / D = K_K /
# C_K / D_K only); the I-flavour entries follow the F.2.a.2
# sign-flip pattern (BesseI-index-up coefficient flips,
# Bessel-index-up + correction-term coefficient keeps).


def _layer_e_matrix_n2(
    kz: float,
    omega: float,
    *,
    vp: float,
    vs: float,
    rho: float,
    r: float,
) -> np.ndarray:
    r"""
    6x6 mode-amplitude-to-state-vector matrix ``E_n2(r)`` for a
    uniform isotropic-elastic layer at azimuthal order n=2
    (quadrupole).

    Sister of :func:`_layer_e_matrix_n0` (4x4 at n=0) and
    :func:`_layer_e_matrix_n1` (6x6 at n=1) at azimuthal order
    2. Maps the six wave amplitudes
    ``c = (B_I, B_K, C_I, C_K, D_I, D_K)`` to the state vector
    ``v(r) = (u_r, u_z, u_theta, sigma_rr, sigma_rz,
    sigma_r_theta)`` via ``v(r) = E_n2(r) c`` at n=2, with the
    substep-F.2.a.5 phase rescale (row * i for u_z and sigma_rz;
    col * -i for SV columns C_I, C_K) absorbed.

    Bessel functions: ``(I_1, I_2)`` and ``(K_1, K_2)`` (vs
    ``(I_0, I_1)`` at n=1). Explicit n=2 azimuthal factors
    (``2 n (n+1) = 12``, ``n^2 - 1 = 3``) appear in the stress
    rows (sigma_rr, sigma_rz, sigma_r_theta).

    Sparsity pattern (pinned by G''.b.1 tests; identical to n=1):

    * Row 1 (``u_z``) cols 4, 5 (``D_I``, ``D_K``): 0. SH
      potential ``psi_z`` doesn't contribute to ``u_z``.
    * Row 2 (``u_theta``) cols 2, 3 (``C_I``, ``C_K``): 0. SV
      potential ``psi_theta`` doesn't contribute to ``u_theta``.

    Parameters
    ----------
    kz, omega : float
        Trial axial wavenumber (rad / m) and angular frequency
        (rad / s).
    vp, vs, rho : float
        Layer P / S velocity (m/s) and density (kg/m^3). Must
        satisfy ``vp > vs > 0`` and ``rho > 0``.
    r : float
        Radius at which to evaluate ``E_n2(r)`` (m). Must be
        positive.

    Returns
    -------
    ndarray, shape (6, 6)
        ``E_n2(r)`` post-rescale. Real-valued in the bound
        regime; ``NaN``-filled outside the bound regime.

    See Also
    --------
    _layer_e_matrix_n1 : The n=1 (flexural) sister.
    _layer_propagator_n2 : G''.b.2 follow-up using
        ``E_n2(r_outer) E_n2(r_inner)^{-1}`` to map the state
        vector across a layer at n=2.
    """
    p2 = kz * kz - (omega / vp) ** 2
    s2 = kz * kz - (omega / vs) ** 2
    if p2 <= 0.0 or s2 <= 0.0 or r <= 0.0:
        return np.full((6, 6), np.nan)
    p = float(np.sqrt(p2))
    s = float(np.sqrt(s2))
    pr = p * r
    sr = s * r
    I1_p = float(special.iv(1, pr))
    I2_p = float(special.iv(2, pr))
    K1_p = float(special.kv(1, pr))
    K2_p = float(special.kv(2, pr))
    I1_s = float(special.iv(1, sr))
    I2_s = float(special.iv(2, sr))
    K1_s = float(special.kv(1, sr))
    K2_s = float(special.kv(2, sr))
    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    E = np.zeros((6, 6))
    # Row 0: u_r = d_r phi + (n/r) psi_z - i k_z psi_theta. At n=2,
    # the d_theta-induced term is +(2/r) psi_z. col*(-i) on C cols.
    E[0, 0] = +p * I1_p - 2.0 * I2_p / r  # B_I
    E[0, 1] = -p * K1_p - 2.0 * K2_p / r  # B_K
    E[0, 2] = -kz * I2_s  # C_I
    E[0, 3] = -kz * K2_s  # C_K
    E[0, 4] = +2.0 * I2_s / r  # D_I
    E[0, 5] = +2.0 * K2_s / r  # D_K
    # Row 1: u_z = i k_z phi + (1/r) d_r [r psi_theta]. row*i on
    # all entries; col*(-i) on C cols (net factor 1).
    E[1, 0] = -kz * I2_p  # B_I
    E[1, 1] = -kz * K2_p  # B_K
    E[1, 2] = +s * I1_s - I2_s / r  # C_I
    E[1, 3] = -s * K1_s - K2_s / r  # C_K
    E[1, 4] = 0.0  # D_I (SH no u_z)
    E[1, 5] = 0.0  # D_K
    # Row 2: u_theta = (n/r) phi cos -> -(n/r) F_n sin sector;
    # contributions also from -d_r psi_z. C cols are zero (SV no
    # u_theta).
    E[2, 0] = -2.0 * I2_p / r  # B_I
    E[2, 1] = -2.0 * K2_p / r  # B_K
    E[2, 2] = 0.0  # C_I
    E[2, 3] = 0.0  # C_K
    E[2, 4] = -s * I1_s + 2.0 * I2_s / r  # D_I
    E[2, 5] = +s * K1_s + 2.0 * K2_s / r  # D_K
    # Row 3: sigma_rr = lambda div u + 2 mu d_r u_r. Lame
    # reduction; n^2 (n+1) factor 12 in the F_n / r^2 term.
    E[3, 0] = +mu * (
        two_kz2_minus_kS2 * I2_p - 2.0 * p * I1_p / r + 12.0 * I2_p / (r * r)
    )  # B_I
    E[3, 1] = +mu * (
        two_kz2_minus_kS2 * K2_p + 2.0 * p * K1_p / r + 12.0 * K2_p / (r * r)
    )  # B_K
    E[3, 2] = -2.0 * kz * mu * (s * I1_s - 2.0 * I2_s / r)  # C_I
    E[3, 3] = +2.0 * kz * mu * (s * K1_s + 2.0 * K2_s / r)  # C_K
    E[3, 4] = +4.0 * mu * (s * I1_s / r - 3.0 * I2_s / (r * r))  # D_I
    E[3, 5] = -4.0 * mu * (s * K1_s / r + 3.0 * K2_s / (r * r))  # D_K
    # Row 4: sigma_rz = mu (d_z u_r + d_r u_z). row*i on all
    # entries; col*(-i) on C cols. n^2-1 = 3 factor in the C cols.
    E[4, 0] = -2.0 * kz * mu * (p * I1_p - 2.0 * I2_p / r)  # B_I
    E[4, 1] = +2.0 * kz * mu * (p * K1_p + 2.0 * K2_p / r)  # B_K
    E[4, 2] = +mu * (two_kz2_minus_kS2 + 3.0 / (r * r)) * I2_s  # C_I
    E[4, 3] = +mu * (two_kz2_minus_kS2 + 3.0 / (r * r)) * K2_s  # C_K
    E[4, 4] = -2.0 * kz * mu * I2_s / r  # D_I
    E[4, 5] = -2.0 * kz * mu * K2_s / r  # D_K
    # Row 5: sigma_r_theta = mu [d_r u_theta + (1/r) d_theta u_r
    #                            - u_theta/r]. Sin sector; no row
    # scaling. n^2 (n+1) = 12 factor in the F_n / r^2 term.
    E[5, 0] = +4.0 * mu * (-p * I1_p / r + 3.0 * I2_p / (r * r))  # B_I
    E[5, 1] = +4.0 * mu * (+p * K1_p / r + 3.0 * K2_p / (r * r))  # B_K
    E[5, 2] = +2.0 * kz * mu * I2_s / r  # C_I
    E[5, 3] = +2.0 * kz * mu * K2_s / r  # C_K
    E[5, 4] = -mu * ((s * s + 12.0 / (r * r)) * I2_s - 2.0 * s * I1_s / r)  # D_I
    E[5, 5] = -mu * ((s * s + 12.0 / (r * r)) * K2_s + 2.0 * s * K1_s / r)  # D_K
    return E


# =====================================================================
# Plan item G''.b.2 -- per-layer propagator P(r_outer | r_inner) at n=2
# =====================================================================
#
# Sister of G.b.2 / G'.b.2 at azimuthal order n=2. Wraps two
# evaluations of ``_layer_e_matrix_n2`` into the layer
# propagator P_j = E_n2(r_outer) E_n2(r_inner)^{-1} (substep
# G''.a.3). Uses ``np.linalg.solve`` rather than explicit
# ``inv`` for better conditioning.
#
# Round-trip / composition oracles in the G''.b.2 tests use the
# state-vector form to avoid the spurious ~1e-6 off-diagonal
# residuals of raw matrix-equality tests at ``cond(E_n2) ~ mu
# ~ 1e10`` -- the same lesson documented in the G.b.2 commit
# (123e634), inherited from G'.b.2.


def _layer_propagator_n2(
    kz: float,
    omega: float,
    *,
    vp: float,
    vs: float,
    rho: float,
    r_inner: float,
    r_outer: float,
) -> np.ndarray:
    r"""
    6x6 layer propagator ``P(r_outer | r_inner)`` mapping the
    n=2 state vector ``v(r) = (u_r, u_z, u_theta, sigma_rr,
    sigma_rz, sigma_r_theta)`` from ``r_inner`` to ``r_outer``
    within a uniform isotropic-elastic layer.

    Implements substep G''.a.3:

    .. math::
        P_j(r_{\text{outer}} \mid r_{\text{inner}}) =
        E_{n=2}(r_{\text{outer}}) \, E_{n=2}(r_{\text{inner}})^{-1}

    with ``E_{n=2}`` from :func:`_layer_e_matrix_n2`.

    Parameters
    ----------
    kz, omega : float
        Trial axial wavenumber (rad / m) and angular frequency
        (rad / s).
    vp, vs, rho : float
        Layer P / S velocity (m/s) and density (kg/m^3).
    r_inner, r_outer : float
        Inner and outer radii of the layer (m). May be in either
        order; the round-trip identity ``P(b|a) P(a|b) v ~ v`` is
        verified in the tests. ``r_inner == r_outer`` returns
        ``eye(6)``.

    Returns
    -------
    ndarray, shape (6, 6)
        The layer propagator. Real-valued in the bound regime;
        ``NaN``-filled outside the bound regime.

    See Also
    --------
    _layer_propagator_n1 : The n=1 (flexural) sister, 6x6.
    _layer_e_matrix_n2 : G''.b.1 helper that this function calls
        twice.
    """
    if r_inner == r_outer:
        return np.eye(6)
    E_inner = _layer_e_matrix_n2(
        kz=kz,
        omega=omega,
        vp=vp,
        vs=vs,
        rho=rho,
        r=r_inner,
    )
    E_outer = _layer_e_matrix_n2(
        kz=kz,
        omega=omega,
        vp=vp,
        vs=vs,
        rho=rho,
        r=r_outer,
    )
    if not (np.all(np.isfinite(E_inner)) and np.all(np.isfinite(E_outer))):
        return np.full((6, 6), np.nan)
    return np.linalg.solve(E_inner.T, E_outer.T).T


# =====================================================================
# Plan item G''.c -- 10x10 stacked modal determinant at n=2
# =====================================================================
#
# Sister of ``_modal_determinant_n0_cased`` (n=0, 7x7) and
# ``_modal_determinant_n1_cased`` (n=1, 10x10). The n=2 case has
# the same 10x10 final-form structure as n=1 with the
# Bessel-function index shift ``(I_0, I_1) -> (I_1, I_2)`` in the
# fluid contribution and the n=2 ``E_n2`` block from G''.b.1 in
# place of ``E_n1``.
#
# BC layout transcribed from substep G''.a.4 (see source-side
# comment block earlier in this file). Column packing matches
# the F.2 / G' convention:
#
#   [A | B_I, B_K, C_I, C_K | B_form, C_form | D_I, D_K | D_form]
#
# Layer = formation collapse identity (substep G''.a.6) is the
# main floating-point oracle: at layer params equal to formation
# params, the brentq root in k_z of this 10x10 determinant equals
# the brentq root of the unlayered 4x4 ``_modal_determinant_n2``
# (the magnitudes differ by a ``det(E_form(b))`` factor; the
# *roots* are invariant). No F.3-equivalent hand-coded form
# exists for n=2, so per-element pinning is via the G''.b.1
# E_n2 transcription rather than a determinant-level
# numerical-equality oracle.


def _modal_determinant_n2_cased(
    kz: float,
    omega: float,
    *,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    layers: tuple[BoreholeLayer, ...],
) -> float:
    r"""
    10x10 quadrupole modal determinant for a borehole with N
    annular layers between fluid and formation, slow-formation
    regime.

    Generalises the unlayered :func:`_modal_determinant_n2` (4x4)
    to arbitrary ``N >= 1`` via the Thomson-Haskell-style 6x6
    propagator chain ``P_total = P_N ... P_2 P_1`` (substep
    G''.a.3). Mirrors :func:`_modal_determinant_n1_cased`
    structurally; the only differences are the Bessel-function
    index shift in the fluid term (``I_1, I_2`` instead of
    ``I_0, I_1``) and the n=2 ``_layer_e_matrix_n2`` /
    ``_layer_propagator_n2`` helpers (G''.b.1, G''.b.2).

    At ``layer == formation`` and ``N == 1`` the brentq root in
    ``k_z`` matches the unlayered :func:`_modal_determinant_n2`
    root (per the G''.a.6 collapse identity:
    ``det(M_10) = det(E_form(b)) * det(M_4)``); the determinant
    magnitudes differ by the ``det(E_form(b))`` scale factor,
    but the root is invariant.

    Parameters
    ----------
    kz, omega, vp, vs, rho, vf, rho_f, a : float
        Same as :func:`_modal_determinant_n2`. ``vp``, ``vs``,
        ``rho`` are the formation half-space (outermost side);
        ``vf``, ``rho_f`` are the borehole fluid.
    layers : tuple of BoreholeLayer
        Annular layer stack ordered inside-out: ``layers[0]``
        adjacent to fluid, ``layers[-1]`` adjacent to formation.
        Must have at least one layer (``len(layers) >= 1``); the
        unlayered case dispatches to :func:`_modal_determinant_n2`
        in :func:`quadrupole_dispersion_layered`. Slow-formation
        constraint ``layer.vs >= vs`` is enforced upstream by
        :func:`_validate_flexural_layers_stacked` (the n=1 / n=2
        constraint is identical).

    Returns
    -------
    float
        ``det(M)`` of the 10x10 cased-hole quadrupole modal
        matrix, real-valued in the bound regime. ``NaN`` outside
        the bound regime.

    See Also
    --------
    _modal_determinant_n2 : Unlayered (4x4) reference.
    _modal_determinant_n1_cased : The n=1 (flexural) sister.
    _layer_e_matrix_n2 : G''.b.1 helper used at r=a and r=b.
    _layer_propagator_n2 : G''.b.2 helper used to compose the
        per-layer propagators.
    """
    F_f_sq = kz * kz - (omega / vf) ** 2
    if F_f_sq <= 0.0:
        return float("nan")
    F_f = float(np.sqrt(F_f_sq))
    radii = [a]
    for L in layers:
        radii.append(radii[-1] + L.thickness)
    b = radii[-1]
    L1 = layers[0]
    E_1_a = _layer_e_matrix_n2(
        kz=kz,
        omega=omega,
        vp=L1.vp,
        vs=L1.vs,
        rho=L1.rho,
        r=a,
    )
    if not np.all(np.isfinite(E_1_a)):
        return float("nan")
    P_total = np.eye(6)
    for j, L in enumerate(layers):
        P_j = _layer_propagator_n2(
            kz=kz,
            omega=omega,
            vp=L.vp,
            vs=L.vs,
            rho=L.rho,
            r_inner=radii[j],
            r_outer=radii[j + 1],
        )
        if not np.all(np.isfinite(P_j)):
            return float("nan")
        P_total = P_j @ P_total
    E_form_b = _layer_e_matrix_n2(
        kz=kz,
        omega=omega,
        vp=vp,
        vs=vs,
        rho=rho,
        r=b,
    )
    if not np.all(np.isfinite(E_form_b)):
        return float("nan")
    v_at_b = P_total @ E_1_a
    # Bessel-index shift relative to the n=1 sister: the fluid
    # pressure ansatz P = A I_n(F r) cos(n theta) at n=2 uses
    # I_1 / I_2 in place of n=1's I_0 / I_1.
    I1_Ff_a = float(special.iv(1, F_f * a))
    I2_Ff_a = float(special.iv(2, F_f * a))

    # Build M[10, 10] in the F.2 / G' column packing.
    # Layer-amplitude blocks: cols 1-4 (P/SV) and cols 7-8 (SH)
    # carry the layer's contribution; formation cols 5-6 and 9
    # carry the formation half-space K-flavour contribution.
    M = np.zeros((10, 10))

    # ---- Rows 0-3 at r=a (4 BCs, fluid-solid interface). ----
    # The layer cols use E_n2(a); formation cols are zero at r=a.

    def _layer_at_a(state_row: int, sign: float) -> np.ndarray:
        """Returns 6 layer-amplitude entries in the order
        (B_I, B_K, C_I, C_K, D_I, D_K) for the given state row."""
        return sign * E_1_a[state_row, :]

    # Row 0: BC1, u_r^(f)(a) - u_r^(m)(a) = 0. Layer negated.
    # Fluid u_r coefficient at n=2 is (F I_1(Fa) - 2 I_2(Fa)/a) /
    # (rho_f omega^2) (n=2 instance of the general-n derivative
    # I_n'(x) = I_{n-1}(x) - (n/x) I_n(x)).
    M[0, 0] = (F_f * I1_Ff_a - 2.0 * I2_Ff_a / a) / (rho_f * omega**2)
    layer0 = _layer_at_a(0, -1.0)  # u_r row, negated
    M[0, 1:5] = layer0[0:4]  # B_I, B_K, C_I, C_K
    M[0, 7:9] = layer0[4:6]  # D_I, D_K
    # Row 1: BC2, -(sigma_rr^(m)(a) + P^(f)(a)) = 0. Layer negated;
    # fluid pressure -P contributes -A I_n(Fa) at n=2 -> -I_2(Fa).
    M[1, 0] = -I2_Ff_a
    layer1 = _layer_at_a(3, -1.0)  # sigma_rr row, negated
    M[1, 1:5] = layer1[0:4]
    M[1, 7:9] = layer1[4:6]
    # Row 2: BC3, sigma_rtheta^(m)(a) = 0. Layer positive (no
    # fluid contribution; fluid is inviscid).
    layer2 = _layer_at_a(5, +1.0)  # sigma_rtheta row
    M[2, 1:5] = layer2[0:4]
    M[2, 7:9] = layer2[4:6]
    # Row 3: BC4, sigma_rz^(m)(a) = 0. Layer positive.
    layer3 = _layer_at_a(4, +1.0)  # sigma_rz row
    M[3, 1:5] = layer3[0:4]
    M[3, 7:9] = layer3[4:6]

    # ---- Rows 4-9 at r=b (6 BCs, layer-formation continuity). ----
    # Layer cols use v_at_b (= P_total @ E_n2(a)); formation cols
    # use E_form(b)'s K-flavour cols (1, 3, 5 in our state-vector
    # E indexing) negated.

    def _layer_at_b(state_row: int) -> np.ndarray:
        """6 layer-amplitude entries (positive, m - s convention)
        from v_at_b for the given state row."""
        return v_at_b[state_row, :]

    def _formation_at_b(state_row: int) -> tuple[float, float, float]:
        """3 formation K-flavour entries (B_form_K = E[:,1],
        C_form_K = E[:,3], D_form_K = E[:,5]); negated for the
        m - s subtraction convention. The destination matrix ``M`` is
        real-dtype and the values are real-valued by physics, so the
        ``.real`` extraction is safe and silences ``ComplexWarning``."""
        return (
            -float(E_form_b[state_row, 1].real),  # B_form
            -float(E_form_b[state_row, 3].real),  # C_form
            -float(E_form_b[state_row, 5].real),  # D_form
        )

    # State-row mapping at r=b: BC5..10 -> state rows 0, 2, 1, 3, 5, 4.
    bc_to_state_b = [0, 2, 1, 3, 5, 4]
    for i_bc, state_row in enumerate(bc_to_state_b):
        i_M = 4 + i_bc
        layer = _layer_at_b(state_row)
        M[i_M, 1:5] = layer[0:4]
        M[i_M, 7:9] = layer[4:6]
        b_form, c_form, d_form = _formation_at_b(state_row)
        M[i_M, 5] = b_form
        M[i_M, 6] = c_form
        M[i_M, 9] = d_form

    return float(np.linalg.det(M))


# =====================================================================
# Fast-formation cased-hole flexural -- complex-``k_z`` helpers (n=1)
# =====================================================================
#
# Sisters of the slow-formation-only ``_layer_e_matrix_n1``,
# ``_layer_propagator_n1``, and ``_modal_determinant_n1_cased``. Same
# construction as the n=2 complex block further down, with the
# Bessel-index shift ``(I_1, I_2) -> (I_0, I_1)`` and the n=1
# azimuthal factors.
#
# In the fast-formation regime (formation ``V_S > V_f``) the guided
# flexural mode has phase velocity in ``(V_R, V_S)``, both above
# ``V_f``. The fluid radial wavenumber ``F = sqrt(k_z^2 -
# omega^2/V_f^2)`` therefore goes purely imaginary while the
# formation P / S radial wavenumbers stay real, so the determinant
# picks up a uniform ``i^k`` phase from the fluid whose imaginary
# part crosses zero at the modal root. The driver
# ``_flexural_dispersion_fast_formation_layered`` brentq's
# ``Im(det)`` along the real-``k_z`` axis; the formation half-space
# uses ``_k_or_hankel`` so a future genuinely-leaky extension can
# flip to the Hankel branch without touching the assembly.


def _layer_e_matrix_n1_complex(
    kz: complex,
    omega: float,
    *,
    vp: float,
    vs: float,
    rho: float,
    r: float,
) -> np.ndarray:
    """
    Complex-``k_z`` 6x6 mode-amplitude-to-state-vector matrix
    ``E_n1(r)`` at azimuthal order n=1.

    Mirrors :func:`_layer_e_matrix_n1` entry-for-entry; the only
    differences are (1) ``kz`` is complex, so ``p`` and ``s`` are
    complex too, and (2) there is no real-``p^2 > 0`` /
    ``s^2 > 0`` bound-regime check -- the layer is computed via
    complex Bessel functions throughout. ``scipy.special.iv`` and
    ``scipy.special.kv`` accept complex arguments transparently;
    in the bound-regime real-``k_z`` limit the entries reduce to
    real values matching the real :func:`_layer_e_matrix_n1`.

    Parameters
    ----------
    kz : complex
        Trial axial wavenumber (rad / m). May be complex.
    omega, vp, vs, rho, r : float
        Angular frequency (rad / s), layer P / S velocity (m/s),
        density (kg/m^3), and evaluation radius (m). Same as
        :func:`_layer_e_matrix_n1`.

    Returns
    -------
    ndarray, shape (6, 6), complex dtype
        ``E_n1(r)`` post-rescale, in the amplitude order
        ``(B_I, B_K, C_I, C_K, D_I, D_K)`` and the state-row order
        ``(u_r, u_z, u_theta, sigma_rr, sigma_rz, sigma_r_theta)``.

    See Also
    --------
    _layer_e_matrix_n1 : Real-``k_z`` slow-formation counterpart.
    _layer_e_matrix_n2_complex : The n=2 sister.
    """
    kz_c = complex(kz)
    p = np.sqrt(kz_c * kz_c - (omega / vp) ** 2)
    s = np.sqrt(kz_c * kz_c - (omega / vs) ** 2)
    pr = p * r
    sr = s * r
    I0_p = complex(special.iv(0, pr))
    I1_p = complex(special.iv(1, pr))
    K0_p = complex(special.kv(0, pr))
    K1_p = complex(special.kv(1, pr))
    I0_s = complex(special.iv(0, sr))
    I1_s = complex(special.iv(1, sr))
    K0_s = complex(special.kv(0, sr))
    K1_s = complex(special.kv(1, sr))
    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz_c * kz_c - kS2

    E: np.ndarray = np.zeros((6, 6), dtype=complex)
    # Row 0: u_r
    E[0, 0] = +p * I0_p - I1_p / r
    E[0, 1] = -p * K0_p - K1_p / r
    E[0, 2] = -kz_c * I1_s
    E[0, 3] = -kz_c * K1_s
    E[0, 4] = +I1_s / r
    E[0, 5] = +K1_s / r
    # Row 1: u_z
    E[1, 0] = -kz_c * I1_p
    E[1, 1] = -kz_c * K1_p
    E[1, 2] = +s * I0_s
    E[1, 3] = -s * K0_s
    E[1, 4] = 0.0
    E[1, 5] = 0.0
    # Row 2: u_theta
    E[2, 0] = -I1_p / r
    E[2, 1] = -K1_p / r
    E[2, 2] = 0.0
    E[2, 3] = 0.0
    E[2, 4] = -s * I0_s + I1_s / r
    E[2, 5] = +s * K0_s + K1_s / r
    # Row 3: sigma_rr
    E[3, 0] = +mu * (
        two_kz2_minus_kS2 * I1_p - 2.0 * p * I0_p / r + 4.0 * I1_p / (r * r)
    )
    E[3, 1] = +mu * (
        two_kz2_minus_kS2 * K1_p + 2.0 * p * K0_p / r + 4.0 * K1_p / (r * r)
    )
    E[3, 2] = -2.0 * kz_c * mu * (s * I0_s - I1_s / r)
    E[3, 3] = +2.0 * kz_c * mu * (s * K0_s + K1_s / r)
    E[3, 4] = +2.0 * mu * (s * I0_s / r - 2.0 * I1_s / (r * r))
    E[3, 5] = -2.0 * mu * (s * K0_s / r + 2.0 * K1_s / (r * r))
    # Row 4: sigma_rz. D cols cross-couple at n=1 (F.2.a.6 erratum:
    # the cos / sin sectors are NOT block-diagonal).
    E[4, 0] = -2.0 * kz_c * mu * (p * I0_p - I1_p / r)
    E[4, 1] = +2.0 * kz_c * mu * (p * K0_p + K1_p / r)
    E[4, 2] = +mu * two_kz2_minus_kS2 * I1_s
    E[4, 3] = +mu * two_kz2_minus_kS2 * K1_s
    E[4, 4] = -kz_c * mu * I1_s / r
    E[4, 5] = -kz_c * mu * K1_s / r
    # Row 5: sigma_r_theta
    E[5, 0] = +2.0 * mu * (-p * I0_p / r + 2.0 * I1_p / (r * r))
    E[5, 1] = +2.0 * mu * (+p * K0_p / r + 2.0 * K1_p / (r * r))
    E[5, 2] = +kz_c * mu * I1_s / r
    E[5, 3] = +kz_c * mu * K1_s / r
    E[5, 4] = -mu * (s * s * I1_s - 2.0 * s * I0_s / r + 4.0 * I1_s / (r * r))
    E[5, 5] = -mu * (s * s * K1_s + 2.0 * s * K0_s / r + 4.0 * K1_s / (r * r))
    return E


def _layer_propagator_n1_complex(
    kz: complex,
    omega: float,
    *,
    vp: float,
    vs: float,
    rho: float,
    r_inner: float,
    r_outer: float,
) -> np.ndarray:
    """
    Complex-``k_z`` 6x6 per-layer propagator at azimuthal order n=1.

    Identical algebra to :func:`_layer_propagator_n1` but with
    ``_layer_e_matrix_n1_complex`` in place of the real version.

    Parameters
    ----------
    kz : complex
        Trial axial wavenumber (rad / m). May be complex.
    omega, vp, vs, rho : float
        Angular frequency (rad / s), layer P / S velocity (m/s),
        and density (kg/m^3).
    r_inner, r_outer : float
        Inner and outer radius of the layer (m).

    Returns
    -------
    ndarray, shape (6, 6), complex dtype
        ``E_n1(r_outer) E_n1(r_inner)^{-1}``.
    """
    if r_inner == r_outer:
        return np.eye(6, dtype=complex)
    E_inner = _layer_e_matrix_n1_complex(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        r=r_inner,
    )
    E_outer = _layer_e_matrix_n1_complex(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        r=r_outer,
    )
    return np.linalg.solve(E_inner.T, E_outer.T).T


def _modal_determinant_n1_cased_complex(
    kz: complex,
    omega: float,
    *,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    layers: tuple[BoreholeLayer, ...],
    leaky_p: bool = False,
    leaky_s: bool = False,
) -> complex:
    r"""
    Complex-``k_z`` 10x10 cased-hole flexural modal determinant.

    Sister of :func:`_modal_determinant_n1_cased` lifted to
    complex ``k_z`` with optional leaky-wave branches for the
    formation half-space (selected via :func:`_k_or_hankel`).
    The annular-layer columns use the complex per-layer E and
    propagator helpers (:func:`_layer_e_matrix_n1_complex`,
    :func:`_layer_propagator_n1_complex`); both Bessel flavours
    are valid in a finite-thickness layer at finite radius
    regardless of regime.

    In the fast-formation bound regime (``V_S > V_f`` with real
    ``k_z`` in ``(omega/V_S, omega/V_R)``), the fluid radial
    wavenumber is purely imaginary while the formation P / S
    radial wavenumbers stay real; the fluid contribution
    introduces a uniform ``i^k`` phase to the determinant whose
    imaginary part vanishes at the modal root. The fast-formation
    cased driver
    (:func:`~fwap.cylindrical_solver._n1_layered._flexural_dispersion_fast_formation_layered`)
    brentq's on ``Im(det)`` along the real-``k_z`` axis with
    ``leaky_p = leaky_s = False``.

    Parameters
    ----------
    kz : complex
        Axial wavenumber (rad / m). May be complex.
    omega, vp, vs, rho, vf, rho_f, a : float
        Angular frequency (rad / s); formation half-space P / S
        velocity (m/s) and density (kg/m^3); borehole-fluid
        velocity (m/s) and density (kg/m^3); borehole radius (m).
        Same as :func:`_modal_determinant_n1_cased`.
    layers : tuple of BoreholeLayer
        Annular layer stack ordered inside-out. Must have
        ``len(layers) >= 1``.
    leaky_p, leaky_s : bool, default False
        Select the leaky branch (Hankel-via-analytic-continuation)
        for the formation P / S Bessel pair. Only the formation
        half-space at ``r = b`` is affected; the layer columns
        always carry both I- and K-flavour.

    Returns
    -------
    complex
        ``det M(k_z)`` evaluated with the chosen branches; ``NaN``
        if any layer or formation block is non-finite.

    See Also
    --------
    _modal_determinant_n1_cased : Real-``k_z`` slow-formation
        counterpart.
    _modal_determinant_n2_cased_complex : The n=2 sister.
    """
    kz_c = complex(kz)
    F = np.sqrt(kz_c * kz_c - (omega / vf) ** 2)
    radii = [a]
    for L in layers:
        radii.append(radii[-1] + L.thickness)
    b = radii[-1]
    L1 = layers[0]
    E_1_a = _layer_e_matrix_n1_complex(
        kz_c,
        omega,
        vp=L1.vp,
        vs=L1.vs,
        rho=L1.rho,
        r=a,
    )
    if not np.all(np.isfinite(E_1_a)):
        return complex("nan")
    P_total = np.eye(6, dtype=complex)
    for j, L in enumerate(layers):
        P_j = _layer_propagator_n1_complex(
            kz_c,
            omega,
            vp=L.vp,
            vs=L.vs,
            rho=L.rho,
            r_inner=radii[j],
            r_outer=radii[j + 1],
        )
        if not np.all(np.isfinite(P_j)):
            return complex("nan")
        P_total = P_j @ P_total
    # Formation half-space: only K-flavour cols (radiation
    # condition); use _k_or_hankel to support the leaky branch.
    # At n=1 the pair needed is (K_0, K_1), i.e. ``_k_or_hankel(0, ...)``.
    p_form = np.sqrt(kz_c * kz_c - (omega / vp) ** 2)
    s_form = np.sqrt(kz_c * kz_c - (omega / vs) ** 2)
    K0pb, K1pb = _k_or_hankel(0, p_form, b, leaky=leaky_p)
    K0sb, K1sb = _k_or_hankel(0, s_form, b, leaky=leaky_s)
    mu_form = rho * vs * vs
    kS2_form = (omega / vs) ** 2
    A_form = 2.0 * kz_c * kz_c - kS2_form
    # Formation K-only state vector at r=b (rows 0..5: u_r, u_z,
    # u_theta, sigma_rr, sigma_rz, sigma_r_theta; cols B_form,
    # C_form, D_form). Mirrors the K-flavour cols (1, 3, 5) of
    # _layer_e_matrix_n1 evaluated at ``r=b`` with formation params.
    E_form_b: np.ndarray = np.zeros((6, 3), dtype=complex)
    # Col 0: B_form (P scalar, K-flavour)
    E_form_b[0, 0] = -p_form * K0pb - K1pb / b
    E_form_b[1, 0] = -kz_c * K1pb
    E_form_b[2, 0] = -K1pb / b
    E_form_b[3, 0] = +mu_form * (
        A_form * K1pb + 2.0 * p_form * K0pb / b + 4.0 * K1pb / (b * b)
    )
    E_form_b[4, 0] = +2.0 * kz_c * mu_form * (p_form * K0pb + K1pb / b)
    E_form_b[5, 0] = +2.0 * mu_form * (p_form * K0pb / b + 2.0 * K1pb / (b * b))
    # Col 1: C_form (SV vector, K-flavour); col*(-i) absorbed
    E_form_b[0, 1] = -kz_c * K1sb
    E_form_b[1, 1] = -s_form * K0sb
    E_form_b[2, 1] = 0.0
    E_form_b[3, 1] = +2.0 * kz_c * mu_form * (s_form * K0sb + K1sb / b)
    E_form_b[4, 1] = +mu_form * A_form * K1sb
    E_form_b[5, 1] = +kz_c * mu_form * K1sb / b
    # Col 2: D_form (SH vector, K-flavour)
    E_form_b[0, 2] = +K1sb / b
    E_form_b[1, 2] = 0.0
    E_form_b[2, 2] = +s_form * K0sb + K1sb / b
    E_form_b[3, 2] = -2.0 * mu_form * (s_form * K0sb / b + 2.0 * K1sb / (b * b))
    E_form_b[4, 2] = -kz_c * mu_form * K1sb / b
    E_form_b[5, 2] = -mu_form * (
        s_form * s_form * K1sb + 2.0 * s_form * K0sb / b + 4.0 * K1sb / (b * b)
    )
    if not np.all(np.isfinite(E_form_b)):
        return complex("nan")
    v_at_b = P_total @ E_1_a
    # At n=1 the fluid pressure ansatz P = A I_n(F r) cos(n theta)
    # uses I_0 / I_1 (vs I_1 / I_2 at n=2).
    I0_Ff_a = complex(special.iv(0, F * a))
    I1_Ff_a = complex(special.iv(1, F * a))

    M: np.ndarray = np.zeros((10, 10), dtype=complex)

    # Rows 0-3 at r=a, fluid-layer interface.
    def _layer_at_a_c(state_row: int, sign: float) -> np.ndarray:
        return sign * E_1_a[state_row, :]

    # Row 0: BC1, u_r continuity. Layer side negated.
    M[0, 0] = (F * I0_Ff_a - I1_Ff_a / a) / (rho_f * omega**2)
    layer0 = _layer_at_a_c(0, -1.0)
    M[0, 1:5] = layer0[0:4]
    M[0, 7:9] = layer0[4:6]
    # Row 1: BC2, sigma_rr balance. Layer side negated; the fluid
    # pressure carries ``-A I_n(F a)``, i.e. ``-I_1`` at n=1.
    M[1, 0] = -I1_Ff_a
    layer1 = _layer_at_a_c(3, -1.0)
    M[1, 1:5] = layer1[0:4]
    M[1, 7:9] = layer1[4:6]
    # Row 2: BC3, sigma_r_theta = 0.
    layer2 = _layer_at_a_c(5, +1.0)
    M[2, 1:5] = layer2[0:4]
    M[2, 7:9] = layer2[4:6]
    # Row 3: BC4, sigma_rz = 0.
    layer3 = _layer_at_a_c(4, +1.0)
    M[3, 1:5] = layer3[0:4]
    M[3, 7:9] = layer3[4:6]

    # Rows 4-9 at r=b, layer-formation continuity (m - s).
    def _layer_at_b_c(state_row: int) -> np.ndarray:
        return v_at_b[state_row, :]

    bc_to_state_b = [0, 2, 1, 3, 5, 4]
    for i_bc, state_row in enumerate(bc_to_state_b):
        i_M = 4 + i_bc
        layer = _layer_at_b_c(state_row)
        M[i_M, 1:5] = layer[0:4]
        M[i_M, 7:9] = layer[4:6]
        M[i_M, 5] = -E_form_b[state_row, 0]
        M[i_M, 6] = -E_form_b[state_row, 1]
        M[i_M, 9] = -E_form_b[state_row, 2]

    return complex(np.linalg.det(M))


# =====================================================================
# Fast-formation cased-hole quadrupole -- complex-``k_z`` helpers
# =====================================================================
#
# Sisters of the slow-formation-only ``_layer_e_matrix_n2``,
# ``_layer_propagator_n2``, and ``_modal_determinant_n2_cased`` from
# G''.b / G''.c. Drop the bound-regime check (``p^2 > 0`` and
# ``s^2 > 0``) and accept arbitrary complex ``k_z`` so that the
# fast-formation regime (phase velocity in (V_R, V_S), both above
# V_f -- so the fluid radial wavenumber goes purely imaginary) can be
# handled by the same matrix-assembly machinery. The formation
# half-space at ``r = b`` uses ``_k_or_hankel`` to select between the
# bound K-Bessel branch and the leaky Hankel-via-analytic-continuation
# branch (default ``leaky_p = leaky_s = False`` matches the unlayered
# fast-formation regime, where the formation P / S waves stay bound
# and only the fluid is oscillatory).
#
# In the fully-bound regime (real ``k_z`` in (omega/V_S, omega/V_R)
# with V_S > V_f), the determinant is real-valued up to a uniform
# ``i^k`` phase from the fluid; the n=2 fast-formation driver
# brentq's on ``Im(det)`` along the real-``k_z`` axis (see
# ``_quadrupole_dispersion_fast_formation_layered``).


def _layer_e_matrix_n2_complex(
    kz: complex,
    omega: float,
    *,
    vp: float,
    vs: float,
    rho: float,
    r: float,
) -> np.ndarray:
    """
    Complex-``k_z`` 6x6 mode-amplitude-to-state-vector matrix
    ``E_n2(r)`` at azimuthal order n=2.

    Mirrors :func:`_layer_e_matrix_n2` entry-for-entry; the only
    differences are (1) ``kz`` is complex, so ``p`` and ``s`` are
    complex too, and (2) there is no real-``p^2 > 0`` /
    ``s^2 > 0`` check -- the layer is computed via complex
    Bessel functions throughout. ``scipy.special.iv`` and
    ``scipy.special.kv`` accept complex arguments transparently;
    in the bound-regime real-``k_z`` limit the entries reduce to
    real values matching the real :func:`_layer_e_matrix_n2`.

    Parameters
    ----------
    kz : complex
        Trial axial wavenumber (rad / m). May be complex.
    omega, vp, vs, rho, r : float
        Same as :func:`_layer_e_matrix_n2`.

    Returns
    -------
    ndarray, shape (6, 6), complex dtype.
    """
    kz_c = complex(kz)
    p = np.sqrt(kz_c * kz_c - (omega / vp) ** 2)
    s = np.sqrt(kz_c * kz_c - (omega / vs) ** 2)
    pr = p * r
    sr = s * r
    I1_p = complex(special.iv(1, pr))
    I2_p = complex(special.iv(2, pr))
    K1_p = complex(special.kv(1, pr))
    K2_p = complex(special.kv(2, pr))
    I1_s = complex(special.iv(1, sr))
    I2_s = complex(special.iv(2, sr))
    K1_s = complex(special.kv(1, sr))
    K2_s = complex(special.kv(2, sr))
    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz_c * kz_c - kS2

    E: np.ndarray = np.zeros((6, 6), dtype=complex)
    # Row 0: u_r
    E[0, 0] = +p * I1_p - 2.0 * I2_p / r
    E[0, 1] = -p * K1_p - 2.0 * K2_p / r
    E[0, 2] = -kz_c * I2_s
    E[0, 3] = -kz_c * K2_s
    E[0, 4] = +2.0 * I2_s / r
    E[0, 5] = +2.0 * K2_s / r
    # Row 1: u_z
    E[1, 0] = -kz_c * I2_p
    E[1, 1] = -kz_c * K2_p
    E[1, 2] = +s * I1_s - I2_s / r
    E[1, 3] = -s * K1_s - K2_s / r
    E[1, 4] = 0.0
    E[1, 5] = 0.0
    # Row 2: u_theta
    E[2, 0] = -2.0 * I2_p / r
    E[2, 1] = -2.0 * K2_p / r
    E[2, 2] = 0.0
    E[2, 3] = 0.0
    E[2, 4] = -s * I1_s + 2.0 * I2_s / r
    E[2, 5] = +s * K1_s + 2.0 * K2_s / r
    # Row 3: sigma_rr
    E[3, 0] = +mu * (
        two_kz2_minus_kS2 * I2_p - 2.0 * p * I1_p / r + 12.0 * I2_p / (r * r)
    )
    E[3, 1] = +mu * (
        two_kz2_minus_kS2 * K2_p + 2.0 * p * K1_p / r + 12.0 * K2_p / (r * r)
    )
    E[3, 2] = -2.0 * kz_c * mu * (s * I1_s - 2.0 * I2_s / r)
    E[3, 3] = +2.0 * kz_c * mu * (s * K1_s + 2.0 * K2_s / r)
    E[3, 4] = +4.0 * mu * (s * I1_s / r - 3.0 * I2_s / (r * r))
    E[3, 5] = -4.0 * mu * (s * K1_s / r + 3.0 * K2_s / (r * r))
    # Row 4: sigma_rz
    E[4, 0] = -2.0 * kz_c * mu * (p * I1_p - 2.0 * I2_p / r)
    E[4, 1] = +2.0 * kz_c * mu * (p * K1_p + 2.0 * K2_p / r)
    E[4, 2] = +mu * (two_kz2_minus_kS2 + 3.0 / (r * r)) * I2_s
    E[4, 3] = +mu * (two_kz2_minus_kS2 + 3.0 / (r * r)) * K2_s
    E[4, 4] = -2.0 * kz_c * mu * I2_s / r
    E[4, 5] = -2.0 * kz_c * mu * K2_s / r
    # Row 5: sigma_r_theta
    E[5, 0] = +4.0 * mu * (-p * I1_p / r + 3.0 * I2_p / (r * r))
    E[5, 1] = +4.0 * mu * (+p * K1_p / r + 3.0 * K2_p / (r * r))
    E[5, 2] = +2.0 * kz_c * mu * I2_s / r
    E[5, 3] = +2.0 * kz_c * mu * K2_s / r
    E[5, 4] = -mu * ((s * s + 12.0 / (r * r)) * I2_s - 2.0 * s * I1_s / r)
    E[5, 5] = -mu * ((s * s + 12.0 / (r * r)) * K2_s + 2.0 * s * K1_s / r)
    return E


def _layer_propagator_n2_complex(
    kz: complex,
    omega: float,
    *,
    vp: float,
    vs: float,
    rho: float,
    r_inner: float,
    r_outer: float,
) -> np.ndarray:
    """
    Complex-``k_z`` 6x6 per-layer propagator at azimuthal order n=2.

    Identical algebra to :func:`_layer_propagator_n2` but with
    ``_layer_e_matrix_n2_complex`` in place of the real version.
    """
    if r_inner == r_outer:
        return np.eye(6, dtype=complex)
    E_inner = _layer_e_matrix_n2_complex(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        r=r_inner,
    )
    E_outer = _layer_e_matrix_n2_complex(
        kz,
        omega,
        vp=vp,
        vs=vs,
        rho=rho,
        r=r_outer,
    )
    return np.linalg.solve(E_inner.T, E_outer.T).T


def _modal_determinant_n2_cased_complex(
    kz: complex,
    omega: float,
    *,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    layers: tuple[BoreholeLayer, ...],
    leaky_p: bool = False,
    leaky_s: bool = False,
) -> complex:
    r"""
    Complex-``k_z`` 10x10 cased-hole quadrupole modal determinant.

    Sister of :func:`_modal_determinant_n2_cased` lifted to
    complex ``k_z`` with optional leaky-wave branches for the
    formation half-space (selected via :func:`_k_or_hankel`).
    The annular-layer cols use the complex per-layer E and
    propagator helpers (:func:`_layer_e_matrix_n2_complex`,
    :func:`_layer_propagator_n2_complex`); both flavours of
    Bessel function are valid in a finite-thickness layer at
    finite radius regardless of regime.

    In the fast-formation bound regime (``V_S > V_f`` with
    real ``k_z`` in ``(omega/V_S, omega/V_R)``), the fluid
    radial wavenumber ``F = sqrt(k_z^2 - omega^2/V_f^2)`` is
    purely imaginary while the formation P / S radial
    wavenumbers stay real; the fluid contribution introduces a
    uniform ``i^k`` phase to the determinant whose imaginary
    part vanishes at the modal root. The fast-formation cased
    driver (:func:`_quadrupole_dispersion_fast_formation_layered`)
    brentq's on ``Im(det)`` along the real-``k_z`` axis with
    ``leaky_p = leaky_s = False``.

    Parameters
    ----------
    kz : complex
        Axial wavenumber. May be complex.
    omega, vp, vs, rho, vf, rho_f, a : float
        Same as :func:`_modal_determinant_n2_cased`.
    layers : tuple of BoreholeLayer
        Annular layer stack ordered inside-out. Must have
        ``len(layers) >= 1``.
    leaky_p, leaky_s : bool, default False
        Select the leaky branch (Hankel-via-analytic-
        continuation) for the formation P / S K-Bessel pair.
        Only the formation half-space at ``r = b`` is affected;
        the layer columns always use both I- and K-flavour
        regardless.

    Returns
    -------
    complex
        ``det M(k_z)`` evaluated with the chosen branches.

    See Also
    --------
    _modal_determinant_n2_cased : Real-``k_z`` slow-formation
        counterpart (G''.c).
    _modal_determinant_n2_complex : Unlayered (4x4) complex-
        ``k_z`` n=2 determinant.
    _quadrupole_dispersion_fast_formation_layered : Driver that
        brentq's ``Im(det)`` for the fast-formation cased mode.
    """
    kz_c = complex(kz)
    F = np.sqrt(kz_c * kz_c - (omega / vf) ** 2)
    radii = [a]
    for L in layers:
        radii.append(radii[-1] + L.thickness)
    b = radii[-1]
    L1 = layers[0]
    E_1_a = _layer_e_matrix_n2_complex(
        kz_c,
        omega,
        vp=L1.vp,
        vs=L1.vs,
        rho=L1.rho,
        r=a,
    )
    if not np.all(np.isfinite(E_1_a)):
        return complex("nan")
    P_total = np.eye(6, dtype=complex)
    for j, L in enumerate(layers):
        P_j = _layer_propagator_n2_complex(
            kz_c,
            omega,
            vp=L.vp,
            vs=L.vs,
            rho=L.rho,
            r_inner=radii[j],
            r_outer=radii[j + 1],
        )
        if not np.all(np.isfinite(P_j)):
            return complex("nan")
        P_total = P_j @ P_total
    # Formation half-space: only K-flavour cols (radiation
    # condition); use _k_or_hankel to support the leaky branch.
    p_form = np.sqrt(kz_c * kz_c - (omega / vp) ** 2)
    s_form = np.sqrt(kz_c * kz_c - (omega / vs) ** 2)
    K1pb, K2pb = _k_or_hankel(1, p_form, b, leaky=leaky_p)
    K1sb, K2sb = _k_or_hankel(1, s_form, b, leaky=leaky_s)
    mu_form = rho * vs * vs
    kS2_form = (omega / vs) ** 2
    A_form = 2.0 * kz_c * kz_c - kS2_form
    # Build the formation K-only state vector at r=b for the 6x6
    # block (rows 0..5: u_r, u_z, u_theta, sigma_rr, sigma_rz,
    # sigma_r_theta; cols B_form, C_form, D_form). Mirrors the
    # K-flavour cols of _layer_e_matrix_n2 with the layer's
    # ``r=b``, ``vp_form, vs_form, rho_form``.
    E_form_b: np.ndarray = np.zeros((6, 3), dtype=complex)
    # Col 0: B_form (P scalar, K-flavour)
    E_form_b[0, 0] = -p_form * K1pb - 2.0 * K2pb / b
    E_form_b[1, 0] = -kz_c * K2pb
    E_form_b[2, 0] = -2.0 * K2pb / b
    E_form_b[3, 0] = +mu_form * (
        A_form * K2pb + 2.0 * p_form * K1pb / b + 12.0 * K2pb / (b * b)
    )
    E_form_b[4, 0] = +2.0 * kz_c * mu_form * (p_form * K1pb + 2.0 * K2pb / b)
    E_form_b[5, 0] = +4.0 * mu_form * (p_form * K1pb / b + 3.0 * K2pb / (b * b))
    # Col 1: C_form (SV vector, K-flavour); col*(-i) absorbed
    E_form_b[0, 1] = -kz_c * K2sb
    E_form_b[1, 1] = -s_form * K1sb - K2sb / b
    E_form_b[2, 1] = 0.0
    E_form_b[3, 1] = +2.0 * kz_c * mu_form * (s_form * K1sb + 2.0 * K2sb / b)
    E_form_b[4, 1] = +mu_form * (A_form + 3.0 / (b * b)) * K2sb
    E_form_b[5, 1] = +2.0 * kz_c * mu_form * K2sb / b
    # Col 2: D_form (SH vector, K-flavour)
    E_form_b[0, 2] = +2.0 * K2sb / b
    E_form_b[1, 2] = 0.0
    E_form_b[2, 2] = +s_form * K1sb + 2.0 * K2sb / b
    E_form_b[3, 2] = -4.0 * mu_form * (s_form * K1sb / b + 3.0 * K2sb / (b * b))
    E_form_b[4, 2] = -2.0 * kz_c * mu_form * K2sb / b
    E_form_b[5, 2] = -mu_form * (
        (s_form * s_form + 12.0 / (b * b)) * K2sb + 2.0 * s_form * K1sb / b
    )
    if not np.all(np.isfinite(E_form_b)):
        return complex("nan")
    v_at_b = P_total @ E_1_a
    # Bessel-index shift: at n=2 the fluid pressure ansatz
    # P = A I_n(F r) cos(n theta) uses I_1 / I_2.
    I1_Ff_a = complex(special.iv(1, F * a))
    I2_Ff_a = complex(special.iv(2, F * a))

    M: np.ndarray = np.zeros((10, 10), dtype=complex)

    # Rows 0-3 at r=a, fluid-layer interface.
    def _layer_at_a_c(state_row: int, sign: float) -> np.ndarray:
        return sign * E_1_a[state_row, :]

    # Row 0: BC1, u_r continuity. Layer side negated.
    M[0, 0] = (F * I1_Ff_a - 2.0 * I2_Ff_a / a) / (rho_f * omega**2)
    layer0 = _layer_at_a_c(0, -1.0)
    M[0, 1:5] = layer0[0:4]
    M[0, 7:9] = layer0[4:6]
    # Row 1: BC2, sigma_rr balance. Layer side negated.
    M[1, 0] = -I2_Ff_a
    layer1 = _layer_at_a_c(3, -1.0)
    M[1, 1:5] = layer1[0:4]
    M[1, 7:9] = layer1[4:6]
    # Row 2: BC3, sigma_r_theta = 0.
    layer2 = _layer_at_a_c(5, +1.0)
    M[2, 1:5] = layer2[0:4]
    M[2, 7:9] = layer2[4:6]
    # Row 3: BC4, sigma_rz = 0.
    layer3 = _layer_at_a_c(4, +1.0)
    M[3, 1:5] = layer3[0:4]
    M[3, 7:9] = layer3[4:6]

    # Rows 4-9 at r=b, layer-formation continuity (m - s).
    def _layer_at_b_c(state_row: int) -> np.ndarray:
        return v_at_b[state_row, :]

    bc_to_state_b = [0, 2, 1, 3, 5, 4]
    for i_bc, state_row in enumerate(bc_to_state_b):
        i_M = 4 + i_bc
        layer = _layer_at_b_c(state_row)
        M[i_M, 1:5] = layer[0:4]
        M[i_M, 7:9] = layer[4:6]
        M[i_M, 5] = -E_form_b[state_row, 0]
        M[i_M, 6] = -E_form_b[state_row, 1]
        M[i_M, 9] = -E_form_b[state_row, 2]

    return complex(np.linalg.det(M))
