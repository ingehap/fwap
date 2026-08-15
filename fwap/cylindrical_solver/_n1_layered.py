"""n=1 layered (mudcake / altered-zone) flexural solver.

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
    _validate_borehole_layers,
    _validate_flexural_layers_stacked,
)
from fwap.cylindrical_solver._n1_isotropic import (
    flexural_dispersion,
)

# =====================================================================
# Plan item F.2.0 -- public-API foundation for layered flexural
# =====================================================================
#
# Sister of :func:`stoneley_dispersion_layered` at azimuthal order 1.
# Lands the public-API surface and the layer=formation regression
# oracle ahead of the 10x10 layered modal-determinant work scheduled
# in plan item F.2 (``docs/plans/cylindrical_biot_F_2.md``).
#
# Scope of this foundation:
#
#   * :func:`flexural_dispersion_layered` is the public n=1 entry
#     point. With ``layers=()`` it dispatches bit-equivalently to
#     :func:`flexural_dispersion`; with non-empty layers it raises
#     ``NotImplementedError`` referencing the F.2 plan.
#   * Reuses :class:`BoreholeLayer` and
#     :func:`_validate_borehole_layers` from the F.1 foundation
#     (PR #43); no new data structures required.
#
# What this lets downstream F.2 work assume:
#
#   * The public-API surface (parameter names, validation rules,
#     return type) is fixed. The follow-up implementing
#     ``_modal_determinant_n1_layered`` and the per-row builders
#     (substeps F.2.b and F.2.c) only needs to swap the dispatch
#     branch below.
#   * The layer=formation regression test in
#     ``tests/test_cylindrical_solver.py`` is already passing with
#     ``layers=()`` and stays as the floating-point oracle for the
#     non-trivial case.


# =====================================================================
# Substep F.2.d -- assembly + public-API dispatch (n=1 layered)
# =====================================================================
#
# Stack the ten row builders (F.2.b.1-7 cos-sector + F.2.c.1-3 sin-
# sector) into the 10x10 modal matrix and return the determinant.
# Each row builder applies the substep-F.2.a.5 row / column phase
# rescale internally, so the assembled matrix is real-valued in the
# bound regime; ``np.linalg.det`` returns the real determinant
# directly.
#
# The public-API dispatch in :func:`flexural_dispersion_layered`
# replaces the previous ``NotImplementedError`` with a brentq loop
# driven by the bound-regime bracket extended to ``min(V_S, V_S_m,
# V_f)``. That path covers the slow-formation regime
# (``V_S < V_f``). Fast formation (``V_S > V_f``) is handled by
# :func:`_flexural_dispersion_fast_formation_layered`, which
# brentq's ``Im(det)`` of the complex-``k_z`` cased determinant
# along the real axis -- the n=1 sister of the fast-formation
# cased-hole quadrupole path.


def _modal_determinant_n1_layered(
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
    10x10 dipole modal determinant for a borehole with one annular
    layer between fluid and formation.

    Stacks the ten row builders from substeps F.2.b and F.2.c into
    the 10x10 layered matrix, applies the substep-F.2.a.5 phase
    rescale (already absorbed into each row), and returns the
    determinant as a real scalar.

    Reduces to :func:`_modal_determinant_n1` (up to an irrelevant
    overall scale factor) when ``(layer.vp, layer.vs, layer.rho) =
    (vp, vs, rho)``: the layer=formation regression test in
    :func:`flexural_dispersion_layered` is the floating-point
    oracle for this routine.

    Parameters
    ----------
    kz : float
        Trial axial wavenumber (rad / m). Must lie in the bound
        regime ``kz > omega / min(V_S, V_S_m, V_f)``; outside the
        regime the determinant is NaN.
    omega, vp, vs, rho, vf, rho_f, a : float
        Formation, fluid, and borehole geometry parameters.
    layer : BoreholeLayer
        Annular layer between fluid and formation.

    Returns
    -------
    float
        ``det(M)`` of the 10x10 layered modal matrix, real-valued
        in the bound regime. NaN outside the bound regime.
    """
    rows = [
        _layered_n1_row1_at_a(
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
        _layered_n1_row2_at_a(
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
        _layered_n1_row3_at_a(
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
        _layered_n1_row4_at_a(
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
        _layered_n1_row5_at_b(
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
        _layered_n1_row6_at_b(
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
        _layered_n1_row7_at_b(
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
        _layered_n1_row8_at_b(
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
        _layered_n1_row9_at_b(
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
        _layered_n1_row10_at_b(
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


def _flexural_kz_bracket_layered(
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
    Bracket the layered n=1 flexural root in (k_z_lo, k_z_hi).

    Lower bound: just above the bound-regime floor
    ``omega / min(V_S, V_S_m, V_f)`` so all five radial wavenumbers
    are real positive. Upper bound: 10 % above the formation
    Rayleigh-speed slowness (the high-f flexural asymptote in the
    unlayered slow-formation case; the layered perturbation is
    typically bounded so this is a generous outer bracket).
    """
    from fwap.cylindrical import rayleigh_speed

    vR = rayleigh_speed(vp, vs)
    slowest = min(vs, layer.vs, vf)
    kz_lo = omega / slowest * (1.0 + 1.0e-6)
    kz_hi = omega / vR * 1.10
    return kz_lo, kz_hi


def _flexural_kz_bracket_cased(
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
    Bracket the cased-hole n=1 flexural root in (k_z_lo, k_z_hi)
    for an arbitrary-N layer stack.

    Generalisation of :func:`_flexural_kz_bracket_layered` to
    ``N >= 1`` layers. Lower bound is the slowest-body-wave floor
    across the entire stack: fluid, every layer, formation
    half-space. Upper bound is the same 10 % above formation
    Rayleigh-speed slowness; the brentq expansion-loop in
    :func:`flexural_dispersion_layered` handles bracket
    extension if the multi-layer perturbation lifts the actual
    root above the cushion.
    """
    from fwap.cylindrical import rayleigh_speed

    vR = rayleigh_speed(vp, vs)
    slowest = min(vs, vf, *(L.vs for L in layers))
    kz_lo = omega / slowest * (1.0 + 1.0e-6)
    kz_hi = omega / vR * 1.10
    return kz_lo, kz_hi


def _flexural_dispersion_fast_formation_layered(
    freq: np.ndarray,
    *,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    layers: tuple[BoreholeLayer, ...],
) -> BoreholeMode:
    r"""
    Fast-formation (``V_S > V_f``) cased-hole flexural dispersion
    via brentq on ``Im(det)``.

    n=1 sister of
    :func:`~fwap.cylindrical_solver._n2_quadrupole._quadrupole_dispersion_fast_formation_layered`,
    lifted from the unlayered fast-formation flexural driver to the
    multi-layer cased determinant
    :func:`~fwap.cylindrical_solver._cased._modal_determinant_n1_cased_complex`.

    The bound regime is ``k_z`` real in ``(omega/V_S, omega/V_f)``;
    in that window the formation P / S radial wavenumbers stay real
    while the fluid radial wavenumber goes purely imaginary, so the
    cased determinant evaluated at real ``k_z`` has an imaginary
    part that crosses zero at the modal root. Both the layer and
    formation columns use ``leaky_p = leaky_s = False`` -- the mode
    is bound everywhere outside the borehole fluid.

    Branch tracking is shared with the unlayered case via
    :func:`~fwap.cylindrical_solver._n1_isotropic._march_fast_flexural_branch`:
    the window is ``(V_f, V_S)`` and the fundamental is the slowest
    root that is no faster than the previous one. See that function and
    :func:`~fwap.cylindrical_solver._n1_isotropic._flexural_dispersion_fast_formation`
    for why ``V_R`` is not a bound of this mode (roadmap A.2).

    Sharing the marcher is **not** on its own enough to keep this path
    and the unlayered one together, and this docstring used to say it
    was. The branch descends through ``V_f``; what happens at that
    crossing is decided after the march, by
    :func:`~fwap.cylindrical_solver._n1_isotropic._extend_below_fluid`,
    which for a long time was called by the unlayered driver and not by
    this one. The two then disagreed above about 10 kHz -- ``NaN`` here
    against a tracked mode there -- in configurations where they solve
    the same problem. Both call it now; the check that they still agree
    is
    ``test_the_layered_drivers_follow_the_branch_below_the_fluid_velocity``.

    Parameters
    ----------
    freq : ndarray, shape (n_freq,)
        Frequency grid (Hz). Strictly positive.
    vp, vs, rho : float
        Formation half-space P / S velocity (m/s) and density
        (kg/m^3). Fast formation, i.e. ``vs > vf``.
    vf, rho_f : float
        Borehole-fluid velocity (m/s) and density (kg/m^3).
    a : float
        Borehole (fluid-side) radius (m).
    layers : tuple of BoreholeLayer
        Annular layer stack ordered inside-out. Non-empty.

    Returns
    -------
    BoreholeMode
        ``name = "flexural"``, ``azimuthal_order = 1``,
        ``slowness`` in s/m with ``NaN`` where no root was
        bracketed. The converged ``k_z`` is real to floating-point
        precision, so ``attenuation_per_meter`` is ``None``.
    """
    from fwap.cylindrical_solver import _modal_determinant_n1_cased_complex

    f_arr = np.asarray(freq, dtype=float)
    n_f = f_arr.size
    slowness = np.full(n_f, np.nan, dtype=float)
    if n_f == 0:
        return BoreholeMode(
            name="flexural",
            azimuthal_order=1,
            freq=f_arr,
            slowness=slowness,
        )

    def _det(kz: float, _omega: float) -> complex:
        return _modal_determinant_n1_cased_complex(
            complex(kz, 0.0),
            _omega,
            vp=vp,
            vs=vs,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            a=a,
            layers=layers,
            leaky_p=False,
            leaky_s=False,
        )

    from fwap.cylindrical_solver._cased import _modal_determinant_n1_cased
    from fwap.cylindrical_solver._n1_isotropic import (
        _FAST_FLEXURAL_MAX_CASED_ROOTS,
        _march_fast_flexural_branch,
        _real_root_function,
    )

    # Roadmap A.7: which part of the determinant carries the signal is
    # measured, not assumed. It is Im at n=1 and Re at n=2, and the
    # n=2 path tracked the wrong one for as long as it existed.

    # The branch descends through V_f exactly as it does without a
    # layer stack, and below it all three radial wavenumbers are real
    # again, so the real cased determinant applies. Without this the
    # layered driver stops at V_f and returns NaN where the unlayered
    # one keeps going -- the two are the same problem when the layers
    # are made of formation, and they disagreed above about 10 kHz.
    def _real_det(kz: float, _omega: float) -> float:
        return _modal_determinant_n1_cased(
            kz,
            _omega,
            vp=vp,
            vs=vs,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            a=a,
            layers=layers,
        )

    root_fn = _real_root_function(_det, f_arr, vs=vs, vf=vf)
    slowness = _march_fast_flexural_branch(
        root_fn,
        f_arr,
        vs=vs,
        vf=vf,
        real_det=_real_det,
        exclude=tuple(layer.vs for layer in layers),
        max_roots=_FAST_FLEXURAL_MAX_CASED_ROOTS,
    )

    return BoreholeMode(
        name="flexural",
        azimuthal_order=1,
        freq=f_arr,
        slowness=slowness,
    )


def _slow_cased_velocity_floor(
    vs: float,
    vp: float,
    rho: float,
    vf: float,
    rho_f: float,
    layers: tuple[BoreholeLayer, ...],
) -> float:
    """
    Physical floor on the phase velocity of a bound layered n >= 1 mode.

    The Scholte speed of the borehole fluid against the softest solid in
    the stack. No interface mode of this geometry runs slower than that,
    so a converged root below it is the bracket-expansion loop having
    walked out of the mode and into the determinant's far tail rather
    than a physical answer.

    Falls back to a small fraction of the fluid velocity if the plane
    Scholte solve does not converge, so the guard can never itself
    reject a good root.
    """
    from fwap.cylindrical import scholte_speed

    softest_vs = min([vs] + [layer.vs for layer in layers])
    if softest_vs == vs:
        soft_vp, soft_rho = vp, rho
    else:
        soft = min(layers, key=lambda layer: layer.vs)
        soft_vp, soft_rho = soft.vp, soft.rho
    try:
        floor = float(scholte_speed(soft_vp, softest_vs, soft_rho, vf, rho_f))
    except (ValueError, RuntimeError):
        return 0.05 * vf
    if not np.isfinite(floor) or floor <= 0.0:
        return 0.05 * vf
    # A little below the plane-interface value: curvature moves the
    # borehole mode off it, always upward at finite frequency, but the
    # guard should not sit exactly on the physics it is protecting.
    return 0.9 * floor


def flexural_dispersion_layered(
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
    Flexural-wave (n=1) phase slowness vs frequency for a borehole
    with optional mudcake / altered-zone annular layers between the
    fluid and the formation half-space.

    With ``layers=()`` (no extra layers) this is bit-equivalent to
    :func:`flexural_dispersion`. With a single
    :class:`BoreholeLayer` it dispatches to the 10x10 layered modal
    determinant :func:`_modal_determinant_n1_layered` and brentq's
    its lowest root across the frequency grid -- the public-API
    payload of plan item F.2.

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
        the formation half-space, ordered radially outward.
        ``()`` dispatches to :func:`flexural_dispersion`. A
        single-element tuple dispatches to the 10x10 layered
        solver; ``N >= 2`` uses the cased-hole propagator chain
        (plan item G'). In the fast-formation regime
        (``V_S > V_f``) any non-empty stack routes to the
        complex-determinant path
        :func:`_flexural_dispersion_fast_formation_layered`.

    Returns
    -------
    BoreholeMode
        ``name = "flexural"``, ``azimuthal_order = 1``, with
        ``slowness[i]`` the phase slowness at each frequency.
        ``NaN`` at any frequency where the bracket failed
        (typically below the geometric cutoff).

    Notes
    -----
    **Fast formations return a sparse curve, and the layer stack is not
    why.** For ``V_S > V_f`` the flexural mode is leaky: its root leaves
    the real ``k_z`` axis, and the real-axis sign change this routine
    searches for survives only beside the shear branch point at high
    frequency. Expect roughly a third of a 1-12 kHz band to converge,
    all of it at the top. The identical formation in an *open* hole is
    just as sparse, so the cause is the fast-formation regime rather
    than the layering; recovering the rest needs complex-plane root
    tracking, not a wider real bracket. Slow formations
    (``V_S < V_f``) converge across the whole band.

    Raises
    ------
    ValueError
        If any input is non-positive, ``vp <= vs``, ``freq``
        contains a non-positive entry, any layer is malformed, or
        any layer fails the slow-formation constraint
        ``layer.vs >= vs`` (multi-layer, slow-formation only --
        the constraint does not apply when ``V_S > V_f``).

    Notes
    -----
    Slow formation (``V_S < V_f``) uses the real-valued
    determinant with brentq on ``det`` itself; fast formation
    (``V_S > V_f``) uses the complex-``k_z`` determinant with
    brentq on ``Im(det)`` along the real axis, because the fluid
    radial wavenumber turns imaginary once the phase velocity
    exceeds ``V_f``.

    **External validation.** Schmitt & Cheng (1987) fig 20 plots this
    mode for a fast sandstone behind a well-bonded steel casing, varying
    the cement thickness (1 cm against 3 cm) in panel (a) and the cement
    shear velocity (their cement 2 against cement 1) in panel (b). Three
    traced curves ship in ``docs/notebooks/_data/`` and score 0.39 %,
    0.52 % and 0.55 % RMS; the two independent renderings of the shared
    3 cm case agree with each other to 0.23 %, which is the floor a 1987
    raster scan supports. Note the radius convention that figure fixes:
    the annulus eats *inward* from a 10 cm original hole, so ``a`` is
    ``0.10 - t_casing - t_cement`` and the formation contact stays at
    0.10 m.

    **A second tie, and a slow formation.** Yang et al. (2022) fig 2
    plots the same mode from vector artwork, and its table 1 gives
    ``r_n`` as *outer* radii directly, so nothing has to be subtracted.
    Two of its eight formations have a stated ``V_P`` and density and
    both ship: the hard one (``V_S`` = 3000) at 0.38 % over 71 of 105
    points, and the soft one (``V_S`` = 1450, *below* the borehole
    fluid's 1500) at **0.017 % over 12 of 12** -- the tightest
    cased-hole tie in the package. That soft curve is bound throughout,
    every point below ``V_S / V_f``; below its published 15.04 kHz
    cutoff this function continues the branch as a leaky root, and there
    is no published curve there to score. See
    :func:`_fill_slow_cased_leaky_n1`.

    References
    ----------
    * Schmitt, D. P., & Cheng, C. H. (1987). Shear wave logging in
      (multilayered) elastic formations: an overview. *MIT Earth
      Resources Laboratory*, 213-268.
    * Yang, M.-E., Lv, W.-G., Wu, Y., Cui, Z.-W., & Liu, J.-X. (2022).
      Numerical study of dispersion characteristics of dipole flexural
      waves in a cased hole with different cement conditions. *Applied
      Geophysics* 19(1), 29-40. doi:10.1007/s11770-022-0923-9
    """
    layers_tuple = tuple(layers)
    _validate_borehole_layers(layers_tuple)
    if not layers_tuple:
        return flexural_dispersion(
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

    if vs > vf:
        # Fast-formation cased-hole flexural: phase velocity is in
        # (V_R, V_S) > V_f, so the fluid radial wavenumber goes
        # purely imaginary while the formation P / S branches stay
        # bound. Brentq on Im(det) along the real-k_z axis via the
        # complex-determinant cased helper. The slow-formation
        # per-layer constraint does NOT apply here: a layer softer
        # in shear than a fast formation (e.g. cement behind a fast
        # carbonate) is physically permissible, and the complex
        # Bessel functions handle the mixed-regime layer kinematics
        # transparently.
        return _flexural_dispersion_fast_formation_layered(
            f_arr,
            vp=vp,
            vs=vs,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            a=a,
            layers=layers_tuple,
        )

    n_layers = len(layers_tuple)
    if n_layers >= 2:
        # Multi-layer path (G'.d) requires the per-layer slow-formation
        # constraint layer.vs >= vs. Single-layer F.2 path documents
        # but does not enforce it (left as the user's responsibility).
        _validate_flexural_layers_stacked(layers_tuple, a, vs)

    from fwap.cylindrical_solver import _modal_determinant_n1_cased

    slowness = np.full_like(f_arr, np.nan, dtype=float)
    velocity_floor = _slow_cased_velocity_floor(vs, vp, rho, vf, rho_f, layers_tuple)
    for i, f in enumerate(f_arr):
        omega = 2.0 * np.pi * float(f)

        # Every layer count goes through the cased determinant. A
        # single layer used to be routed through the standalone
        # ``_modal_determinant_n1_layered``, a second implementation of
        # the same boundary-value problem. Both carried the SV ansatz
        # corrected under roadmap A.8, and keeping two copies of that
        # correction in step is exactly the drift the
        # ``layer == formation`` invariants exist to catch.
        # ``_modal_determinant_n1_layered`` is retained as a directly
        # pinned algebraic reference; it is no longer on this path.
        def _det(kz_, omega=omega, layers_tuple=layers_tuple):
            return _modal_determinant_n1_cased(
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

        kz_lo, kz_hi = _flexural_kz_bracket_cased(
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
                    "flexural_dispersion_layered: det evaluation NaN "
                    "at f=%.1f Hz (likely outside bound regime)",
                    f,
                )
                continue
            if np.sign(d_lo) == np.sign(d_hi):
                logger.debug(
                    "flexural_dispersion_layered: failed to bracket "
                    "at f=%.1f Hz (likely below cutoff)",
                    f,
                )
                continue
            kz_root = optimize.brentq(_det, kz_lo, kz_hi, xtol=1.0e-10)
            if omega / kz_root < velocity_floor:
                # The expansion loop walked past the mode into the
                # determinant's far tail; no interface mode of this
                # geometry is slower than the Scholte speed.
                logger.debug(
                    "flexural_dispersion_layered: rejected a %.1f m/s root at "
                    "f=%.1f Hz, below the %.1f m/s Scholte floor",
                    omega / kz_root,
                    f,
                    velocity_floor,
                )
                continue
            slowness[i] = kz_root / omega
        except (ValueError, RuntimeError) as exc:
            logger.debug(
                "flexural_dispersion_layered: brentq failed at f=%.1f Hz: %s",
                f,
                exc,
            )

    attenuation = _fill_slow_cased_leaky_n1(
        slowness,
        f_arr,
        vp=vp,
        vs=vs,
        rho=rho,
        vf=vf,
        rho_f=rho_f,
        a=a,
        layers=layers_tuple,
    )

    return BoreholeMode(
        name="flexural",
        azimuthal_order=1,
        freq=f_arr,
        slowness=slowness,
        attenuation_per_meter=attenuation,
    )


def _fill_slow_cased_leaky_n1(
    slowness: np.ndarray,
    f_arr: np.ndarray,
    *,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    layers: tuple[BoreholeLayer, ...],
) -> np.ndarray | None:
    r"""
    Fill the frequencies where the bound n=1 layered search found
    nothing with the leaky cased branch, in place.

    Roadmap A.9. A stiff annulus raises the composite bending stiffness
    until the dipole mode outruns a slow formation's shear speed; the
    mode then radiates into the formation and the real-valued
    determinant, which only describes fields evanescent everywhere
    outside the fluid, has no root for it. This runs the complex-``k_z``
    marcher over ``(V_S, min(V_f, min layer V_S))`` for exactly those
    frequencies.

    Parameters
    ----------
    slowness : ndarray, shape (n_f,)
        Bound-path result, modified in place at the frequencies the
        leaky branch resolves.
    f_arr : ndarray, shape (n_f,)
        Frequency grid (Hz).
    vp, vs, rho, vf, rho_f, a : float
        As in :func:`flexural_dispersion_layered`.
    layers : tuple of BoreholeLayer
        Annular stack, inside-out.

    Returns
    -------
    ndarray or None
        Attenuation (1/m) aligned with ``f_arr``, ``NaN`` at the
        frequencies the bound path already covered (a bound mode has
        no attenuation) and at those neither path resolved. ``None``
        when the leaky branch contributed nothing, so a purely bound
        curve still reports ``attenuation_per_meter is None``.

    See Also
    --------
    ~fwap.cylindrical_solver._leaky._march_leaky_cased_branch :
        The shared marcher, and the n=2 sister's entry point.

    Notes
    -----
    **The published statement behind this branch**, and the only
    external evidence it has, is Schmitt & Cheng (1987) p. 231: behind
    casing in a slow formation "the high frequency part of the
    fundamental modes excited either by a dipole or a quadrupole source
    will then also be leaky", travelling "with a velocity higher than
    that of the formation shear wave". They illustrate it with
    *waveforms* (their figs 24 and 25), not with a dispersion curve, so
    unlike the bound cased branch -- tied by their figs 20 and 21 at
    0.4-0.6 % -- this one cannot be scored against a figure. It is
    covered by tests instead; see
    ``test_the_cased_dipole_of_a_slow_formation_has_no_bound_root_at_all``.

    Yang et al. (2022) fig 2(b) is the nearest thing to a curve for it,
    and it stops just short: their slow formation's cased dipole is
    plotted only down to its 15.04 kHz cutoff, where the branch is still
    bound. :func:`flexural_dispersion_layered` matches that bound part
    at 0.017 % and then continues *this* branch below it, over
    12.10-14.75 kHz, with nothing published to compare against.

    **The ceiling is a real limit, with a measured example.** The window
    stops at ``min(V_f, min layer V_S)``, and for Schmitt & Cheng's own
    slow sandstone (2751 / 1201 / 2100) behind 1.02 cm of steel and 3 cm
    of their cement 1 the binding term is the *fluid*: the branch leaves
    ``V_S`` near 1.4 kHz, climbs to about 1710 m/s at 5.5 kHz -- just
    under the cement's 1729 -- and comes back down through ``V_f`` near
    13.8 kHz.

    **That gap had two causes and one of them is now fixed.** Above
    3 kHz the root is outside the window and the marcher is right not to
    find it. At 1.5-2.5 kHz it is *inside* the window -- a winding count
    over the whole ``(V_S, V_f)`` box returns 1 -- and it used to be
    missed anyway, which was seeding rather than the ceiling. Rebuilding
    the seeding recovers that leg in full (1235.9, 1300.0, 1358.3,
    1412.0, 1461.2 m/s at 1.50-2.50 kHz), and an argument-principle
    contour confirms it is exactly the window's contents: one root at
    each of those five frequencies and none at 1.00, 1.25, 2.75 or 3.00.

    Letting pass two **re-acquire** after a gap, rather than stopping
    two misses into one, then extended the upper leg down from 14.00 to
    13.25 kHz -- and on the ``_A2`` stack from 5.00 to 4.25 kHz, closing
    a one-sample hole at 8.0 kHz that fixture had carried since it was
    first measured. A **downward pass** then walks each leg back from
    the frequency it was entered at, which both marching passes reach
    from below; on ``_A2`` that recovers 4.00 kHz and the leg ends where
    the roots do, with a contour counting none at 3.75.

    **Schmitt & Cheng's 13.00 kHz was a third cause, and it is closed
    too.** An earlier note here blamed the one-directional march, which
    is what ``_A2``'s 4.00 kHz was; measured, the descent reached 13.00
    and found a root at 1497.11 m/s that ``_valid`` then declined for
    sitting inside a 0.2 % dead band held off the window ceiling. That
    dead band has since been withdrawn -- it was a second, unasked-for
    use of ``_LEAKY_CASED_DEGENERACY_TOL``, redundant wherever the
    ceiling is a layer shear speed and active only where it is ``V_f``,
    which is this geometry. See that constant for the three measurements
    behind the withdrawal. 13.00 kHz now comes back.

    What is left besides that is the ceiling. Between roughly 3 and 13 kHz the branch
    is above ``V_f``, outside the window this searches at all, and no
    amount of seeding reaches it -- a contour still counts one root
    there at 3.0, 5.5 and 8.0 kHz. Raising the ceiling is the remaining
    half, and it would need the fluid field handled as oscillatory
    rather than evanescent.

    References
    ----------
    * Schmitt, D. P., & Cheng, C. H. (1987). Shear wave logging in
      (multilayered) elastic formations: an overview. *MIT Earth
      Resources Laboratory*, 213-268.
    """
    missing = ~np.isfinite(slowness)
    if not layers or not missing.any():
        return None
    ceiling = min(vf, min(layer.vs for layer in layers))
    if not ceiling > vs:
        return None

    from fwap.cylindrical_solver import _modal_determinant_n1_cased_complex
    from fwap.cylindrical_solver._leaky import (
        _detect_leaky_branches,
        _march_leaky_cased_branch,
    )

    def _det(kz: complex, omega: float) -> complex:
        _, leaky_p, leaky_s = _detect_leaky_branches(kz, omega, vp, vs, vf)
        return _modal_determinant_n1_cased_complex(
            kz,
            omega,
            vp=vp,
            vs=vs,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            a=a,
            layers=layers,
            leaky_p=leaky_p,
            leaky_s=leaky_s,
        )

    leaky_slowness, leaky_atten = _march_leaky_cased_branch(
        _det,
        f_arr,
        vs=vs,
        ceiling=ceiling,
        exclude=tuple(layer.vs for layer in layers),
    )
    fill = missing & np.isfinite(leaky_slowness)
    if not fill.any():
        return None
    slowness[fill] = leaky_slowness[fill]
    attenuation = np.full(f_arr.size, np.nan, dtype=float)
    attenuation[fill] = leaky_atten[fill]
    return attenuation


# =====================================================================
# Substep F.2.a -- math scaffolding for the n=1 layered determinant
# =====================================================================
#
# Sister of F.1.a at azimuthal order 1. Inherits all conventions and
# derivations from the existing n=1 single-interface derivation
# (substeps 1.1 through 1.6.e at lines ~401-2356) plus the F.1.a
# substep blocks; extends with the I-flavour annulus terms and the
# cos / sin sector decomposition.
#
# Key structural addition vs F.1.a: three new field components
# (the SH potential ``psi_z`` and the sin-azimuthal-sector
# u_theta / sigma_rtheta BCs) extend the matrix from 7x7 to
# 10x10. The cos and sin azimuthal sectors do NOT decouple --
# d_theta operations in u_r, sigma_rtheta, sigma_rz cross-couple
# every amplitude family into BCs of either azimuthal sector
# (see substeps F.2.a.4 and F.2.a.6 for the dense-matrix
# structure and the erratum that withdrew an earlier mistaken
# block-diagonal claim).

# =====================================================================
# Substep F.2.a.1 -- sign conventions, regime gate, field ansatz
# =====================================================================
#
# Conventions inherited from substep F.1.a.1 (n=0 layered) and the
# existing n=1 single-interface block (substep 1.1 at line ~410):
#
#   * Time dependence ``e^{-i omega t}``.
#   * Axial dependence ``e^{i k_z z}``.
#   * Bound regime: every radial wavenumber is real positive.
#
# Same five radial wavenumbers as F.1.a.1:
#
#       F_f = sqrt(k_z^2 - (omega / V_f)^2)         > 0    (fluid)
#       p_m = sqrt(k_z^2 - (omega / V_P_m)^2)       > 0    (annulus P)
#       s_m = sqrt(k_z^2 - (omega / V_S_m)^2)       > 0    (annulus S)
#       p   = sqrt(k_z^2 - (omega / V_P)^2)         > 0    (formation P)
#       s   = sqrt(k_z^2 - (omega / V_S)^2)         > 0    (formation S)
#
# Bound-regime gate: ``k_z > omega / min(V_S, V_S_m, V_f)``.
#
# New at n=1: cos / sin azimuthal sector partition. The four
# scalar potentials (P, SV-theta, SH-z) and the fluid pressure
# distribute as:
#
#   Cos sector:  P^{(f)}, phi^{(m,s)}, psi_theta^{(m,s)}
#   Sin sector:  psi_z^{(m,s)}
#
# Rationale (substep 1.1 of the existing n=1 block, full derivation
# there): the curl identity ``(curl psi)_r = (1/r) d_theta(psi_z)
# - d_z(psi_theta)`` lands on cos(theta) iff ``psi_z`` carries
# sin(theta) and ``psi_theta`` carries cos(theta); the same
# constraint independently fixes u_z and the sectors close
# without cross-talk.
#
# Field representation, bound regime, layered case (10 amplitudes
# total in column order):
#
#   Cos sector (7 amplitudes):
#       Fluid:        P^{(f)} = A I_1(F_f r) cos(theta)
#       Annulus P:    phi^{(m)} = (B_I I_1(p_m r) + B_K K_1(p_m r))
#                                 * cos(theta)
#       Annulus SV:   psi_theta^{(m)} = (C_I I_1(s_m r) + C_K K_1(s_m r))
#                                       * cos(theta)
#       Formation P:  phi^{(s)} = B K_1(p r) cos(theta)
#       Formation SV: psi_theta^{(s)} = C K_1(s r) cos(theta)
#
#   Sin sector (3 amplitudes):
#       Annulus SH:   psi_z^{(m)} = (D_I I_1(s_m r) + D_K K_1(s_m r))
#                                   * sin(theta)
#       Formation SH: psi_z^{(s)} = D K_1(s r) sin(theta)
#
# Column order for the 10x10 matrix:
#
#       [ A | B_I, B_K, C_I, C_K | B, C | D_I, D_K | D ]
#         |---------- annulus + fluid ----------||formation|
#
# The bars group amplitudes by region (fluid + annulus 7 cols,
# formation 3 cols). The amplitudes' natural azimuthal-factor
# labels (cos for A, B, C; sin for D) describe the underlying
# potential structure but DO NOT translate to a block-diagonal
# matrix partition: every BC row at n=1 generically couples to
# all amplitude families through the d_theta-induced cross-terms
# (substep F.2.a.4 sparsity diagram + F.2.a.6 erratum).

# =====================================================================
# Substep F.2.a.2 -- displacements per region (cos and sin sectors)
# =====================================================================
#
# The SV amplitude C is the Hansen potential -- roadmap A.8. Its
# displacement field is
#
#       u^{(C)} = curl curl (chi z),   chi = C X_1(s r) e^{i theta}
#
# which has THREE non-zero components,
#
#       u_r     = i k_z d_r chi
#       u_theta = i k_z (n / r) chi          (n = 1 here)
#       u_z     = -s^2 chi
#
# An azimuthal-only vector potential ``psi_theta e_theta`` -- which
# is what the C columns encoded before the A.8 correction, and what
# gives ``u_r ~ X_1`` with no u_theta at all -- is not a solution of
# the elastodynamic equations for n >= 1: the cylindrical vector
# Laplacian couples the radial and azimuthal components through a
# term proportional to n that such an ansatz has no term to cancel.
# It happens to be a solution at n = 0, which is why the n=0 layered
# block (F.1) is unaffected.
#
# Bessel-derivative identities reused from F.1.a.2 (sign-flipped
# I-twin of the K-flavour identities the existing n=1 block uses):
#
#       I_0'(x) = +I_1(x)             K_0'(x) = -K_1(x)
#       I_1'(x) = I_0(x) - I_1(x)/x   K_1'(x) = -K_0(x) - K_1(x)/x
#       (1/r) d_r [r I_1(alpha r)] = +alpha * I_0(alpha r)
#       (1/r) d_r [r K_1(alpha r)] = -alpha * K_0(alpha r)
#
# K-flavour displacements (annulus and formation share the same
# functional form; substep 1.2 of the existing n=1 block has the
# full per-component derivation. The annulus result is obtained
# by ``B -> B_K, C -> C_K, D -> D_K, p -> p_m, s -> s_m``):
#
#   Cos sector (u_r and u_z; D contributes to u_r via the curl
#   identity):
#
#     u_r^{(s,K)}     = -B  p   K_0(p r)
#                       - B  K_1(p r) / r
#                       + D  K_1(s r) / r
#                       - i k_z C  [s K_0(s r)
#                                    + K_1(s r) / r]   (cos)
#
#     u_z^{(s,K)}     = i k_z B  K_1(p r)
#                       - C  s^2 K_1(s r)          (cos)
#
#   Sin sector (u_theta; B, C and D all contribute):
#
#     u_theta^{(s,K)} = -B  K_1(p r) / r
#                       - k_z C  K_1(s r) / r
#                       + D  s   K_0(s r)
#                       + D  K_1(s r) / r          (sin)
#
# I-flavour displacements (annulus only; sign-flipped on the
# Bessel-derivative-induced terms per the F.1.a.2 pattern):
#
#   Cos sector:
#
#     u_r^{(m,I)}     = +B_I p_m I_0(p_m r)
#                       - B_I I_1(p_m r) / r
#                       + D_I I_1(s_m r) / r
#                       + i k_z C_I [s_m I_0(s_m r)
#                                     - I_1(s_m r) / r]  (cos)
#
#     u_z^{(m,I)}     = i k_z B_I I_1(p_m r)
#                       - C_I s_m^2 I_1(s_m r)     (cos)
#
#   Sin sector:
#
#     u_theta^{(m,I)} = -B_I I_1(p_m r) / r
#                       - k_z C_I I_1(s_m r) / r
#                       + D_I s_m I_0(s_m r)
#                       + D_I I_1(s_m r) / r       (sin)
#
# Sign-flip pattern between K and I flavours (carried over from
# F.1.a.2 mutatis mutandis):
#
#   * The "p I_0" / "p K_0" coefficient in u_r flips sign:
#     +B_I p_m I_0(p_m r) vs -B_K p_m K_0(p_m r).
#   * The "I_1 / r" / "K_1 / r" coefficient KEEPS sign:
#     -B_I I_1(p_m r) / r vs -B_K K_1(p_m r) / r.
#   * The "s I_0" / "s K_0" coefficient in u_r and u_theta flips
#     sign: +C_I s_m I_0 vs -C_K s_m K_0; +D_I s_m I_0 vs
#     -D_K s_m K_0.
#
# Same pattern as F.1.a.2: the d_r-induced derivatives twin (one
# Bessel index up vs down) get sign-flipped; the no-derivative
# direct ``F / r`` terms keep sign.
#
# Fluid displacements (cos and sin sectors, regular at axis):
#
#     u_r^{(f)}     = (A / (rho_f omega^2)) [F_f I_0(F_f r)
#                                              - I_1(F_f r) / r] cos(theta)
#     u_theta^{(f)} = -(A / (rho_f omega^2 r)) I_1(F_f r) sin(theta)
#     u_z^{(f)}     = (i k_z A / (rho_f omega^2)) I_1(F_f r) cos(theta)
#
# (Direct from substep 1.2 of the existing n=1 block; the layered
# fluid is identical to the single-interface fluid since the
# fluid lives at ``r < a`` regardless of what's beyond.)

# =====================================================================
# Substep F.2.a.3 -- stresses per region (cos and sin sectors)
# =====================================================================
#
# Hooke's law reused from F.1.a.3 / substep 1.3.a of the existing
# n=1 block. The Lame reduction
#
#       -lambda_R k_PR^2 + 2 mu_R p_R^2 = mu_R (2 k_z^2 - k_SR^2)
#
# carries through region by region (R in {annulus_m, formation_s}).
#
# Three stress quantities appear in the n=1 BCs (vs two at n=0):
#
#   sigma_rr        (cos sector; appears in BC2, BC8)
#   sigma_rtheta    (sin sector; appears in BC3, BC9 -- NEW at n>=1)
#   sigma_rz        (cos sector; appears in BC4, BC10)
#
# K-flavour stresses (annulus / formation, with the formation form
# obtained by ``X_K -> -X`` rename and parameter substitution; the
# full r=a forms with the row-4-by-i / column-C-by-(-i) rescale
# absorbed are in the docstring of :func:`_modal_determinant_n1`
# as the M21-M44 entries). At general radius r:
#
#   sigma_rr (cos sector) -- coefficients of (B, C, D) for the K-flavour:
#
#       B:  mu (2 k_z^2 - k_S^2) K_1(p r)
#           + 2 mu p K_0(p r) / r        (... Lame-reduced)
#           + 4 mu K_1(p r) / r^2
#
#       C:  -2 i k_z mu [s^2 K_1(s r) + s K_0(s r) / r
#                         + 2 K_1(s r) / r^2]
#
#       D:  +2 mu [s K_0(s r) / r + 2 K_1(s r) / r^2]
#
#   sigma_rtheta (sin sector) -- coefficients:
#
#       B:  -2 mu [p K_0(p r) / r + 2 K_1(p r) / r^2]
#
#       C:  -2 mu k_z [s K_0(s r) / r + 2 K_1(s r) / r^2]
#
#       D:  mu [s^2 K_1(s r) + 2 s K_0(s r) / r + 4 K_1(s r) / r^2]
#
#   sigma_rz (cos sector) -- coefficients:
#
#       B:  -2 i k_z mu [p K_0(p r) + K_1(p r) / r]
#
#       C:  i mu (2 k_z^2 - k_S^2) [s K_0(s r)
#                                     + K_1(s r) / r]  ... (the i
#           comes from the substep-1.5 row-4-by-i convention; in the
#           pre-rescale form this is ``mu (2 k_z^2 - k_S^2) d_r K_1``
#           multiplied by i.)
#
#       D:  -i k_z mu K_1(s r) / r
#
# I-flavour stresses (annulus only): same algebraic form as the
# K-flavour with the substep-F.1.a.2 sign-flip pattern applied to
# every "Bessel-index-up" term. The B_I sigma_rr coefficient, for
# example, is:
#
#       B_I:  mu_m (2 k_z^2 - k_Sm^2) I_1(p_m r)
#             - 2 mu_m p_m I_0(p_m r) / r        (sign flip on I_0/r vs +K_0/r)
#             + 4 mu_m I_1(p_m r) / r^2          (KEEP sign on I_1/r^2)
#
# The full per-row entries are not tabulated here; they will be
# read off and transcribed in the per-row builders (F.2.b.1.b for
# row 2, F.2.b.2.c for row 8, F.2.c.1 for row 3, etc.). The
# substep-F.1.a.6 self-check at layer=formation collapses the
# annulus K-flavour columns to the formation columns (with sign
# flip) and is the primary correctness oracle.

# =====================================================================
# Substep F.2.a.4 -- 10x10 boundary-condition row layout
# =====================================================================
#
# Rows are the ten boundary conditions; columns are the ten
# amplitudes in the column order
# ``[A | B_I, B_K, C_I, C_K | B, C | D_I, D_K | D]``.
#
#   Row 1   u_r^{(f)} - u_r^{(m)} = 0           at r = a   (cos)
#   Row 2   sigma_rr^{(m)} + P^{(f)} = 0        at r = a   (cos)
#   Row 3   sigma_rtheta^{(m)} = 0              at r = a   (sin)
#   Row 4   sigma_rz^{(m)} = 0                  at r = a   (cos)
#   Row 5   u_r^{(m)} - u_r^{(s)} = 0           at r = b   (cos)
#   Row 6   u_theta^{(m)} - u_theta^{(s)} = 0   at r = b   (sin)
#   Row 7   u_z^{(m)} - u_z^{(s)} = 0           at r = b   (cos)
#   Row 8   sigma_rr^{(m)} - sigma_rr^{(s)} = 0 at r = b   (cos)
#   Row 9   sigma_rtheta^{(m)}
#               - sigma_rtheta^{(s)} = 0        at r = b   (sin)
#   Row 10  sigma_rz^{(m)}
#               - sigma_rz^{(s)} = 0            at r = b   (cos)
#
# Sparsity pattern (X = generically non-zero, . = identically
# zero by interface localization or BC physics):
#
#               A  B_I B_K C_I C_K  B   C  | D_I D_K  D
#       Row  1[ X   X   X   X   X   .   .  |  X   X   . ]  cos, r=a
#       Row  2[ X   X   X   X   X   .   .  |  X   X   . ]  cos, r=a
#       Row  3[ .   X   X   X   X   .   .  |  X   X   . ]  sin, r=a
#       Row  4[ .   X   X   X   X   .   .  |  X   X   . ]  cos, r=a
#       Row  5[ .   X   X   X   X   X   X  |  X   X   X ]  cos, r=b
#       Row  6[ .   X   X   .   .   X   .  |  X   X   X ]  sin, r=b
#       Row  7[ .   X   X   X   X   X   X  |  .   .   . ]  cos, r=b
#       Row  8[ .   X   X   X   X   X   X  |  X   X   X ]  cos, r=b
#       Row  9[ .   X   X   X   X   X   X  |  X   X   X ]  sin, r=b
#       Row 10[ .   X   X   X   X   X   X  |  X   X   X ]  cos, r=b
#
# **Erratum vs the original substep block** (replaced 2024-12; see
# also the now-obsolete F.2.a.6 below): an earlier draft of this
# diagram claimed cos and sin sectors decouple into a 7x7-cos-block-
# plus-3x3-sin-block direct sum. That was wrong. At n=1, the d_theta
# operations couple every amplitude family into BCs of EITHER
# azimuthal sector:
#
#   * The (1/r) d_theta(psi_z) term in u_r couples the SH amplitudes
#     (D_I, D_K, D -- ``sin``-factor potentials) into the cos-sector
#     u_r and sigma_rr BCs (rows 1, 2, 5, 8). Concretely visible in
#     the existing single-interface ``_modal_determinant_n1`` as
#     ``M14 = -K_1(s a) / a`` (D column non-zero in cos-sector u_r
#     row).
#
#   * The (1/r) d_theta u_r term in sigma_rtheta couples the cos-
#     sector P / SV amplitudes (B, C and their I-flavour twins)
#     into the sin-sector sigma_rtheta BCs (rows 3, 9). Visible as
#     ``M32, M33 != 0`` (B and C columns non-zero in sin-sector
#     sigma_rtheta row at r=a).
#
# Result: the 10x10 layered determinant is a fully-populated dense
# matrix, with sparsity only from interface localization (column A
# only in rows touching r=a; columns B, C, D only in rows touching
# r=b) and BC-specific zeros (column A is zero in row 3 because
# fluid carries no shear; the C-flavour columns are zero in rows 6
# and the formation B / C / D columns are zero in row 7's BC
# physics for u_z continuity per substep F.2.a.2's u_theta and u_z
# decompositions).
#
# Two additional zero patterns visible in the diagram:
#
#   * Column 0 (A) is zero in rows 3, 4, 5-10: the fluid lives at
#     ``r < a`` and carries no shear (so it is also absent from
#     the r=a sigma_rtheta and sigma_rz rows even though they touch
#     r=a).
#   * Columns 5, 6, 9 (formation) are zero in rows 1-4: the
#     formation half-space lives at ``r > b`` and contributes
#     nothing at r=a.
#
# (These two are the "pure interface localization" zeros; the row 6
# / row 7 partial zeros are BC-physics zeros that the per-row
# builders in F.2.b / F.2.c will surface explicitly in their
# closed-form transcriptions.)

# =====================================================================
# Substep F.2.a.5 -- phase rescaling for a real-valued matrix
# =====================================================================
#
# Same per-row imaginary-power pattern analysis as F.1.a.5; the
# additional sin sector follows the same logic.
#
# Pre-rescale phase pattern of each row, with all 10 columns
# laid out (A | B_I B_K B | C_I C_K C | D_I D_K D); ``R`` denotes
# a generically real entry, ``i*R`` a generically imaginary one,
# ``.`` an identically zero entry per the F.2.a.4 sparsity.
#
#   Cos rows (``z``-bearing where noted):
#
#     Row 1  (u_r at r=a)        : A R | B R     | C i*R   | D R
#     Row 2  (sigma_rr at r=a)   : A R | B R     | C i*R   | D R
#     Row 4  (sigma_rz at r=a)   : A 0 | B i*R   | C R     | D i*R   <- z-bearing
#     Row 5  (u_r at r=b)        : A 0 | B R     | C i*R   | D R
#     Row 7  (u_z at r=b)        : A 0 | B i*R   | C R     | D 0     <- z-bearing
#     Row 8  (sigma_rr at r=b)   : A 0 | B R     | C i*R   | D R
#     Row 10 (sigma_rz at r=b)   : A 0 | B i*R   | C R     | D i*R   <- z-bearing
#
#   Sin rows:
#
#     Row 3  (sigma_rtheta at a) : A 0 | B R     | C i*R   | D R
#     Row 6  (u_theta at r=b)    : A 0 | B R     | C i*R   | D R
#     Row 9  (sigma_rtheta at b) : A 0 | B R     | C i*R   | D R
#
# (The C-columns enter the sin-sector rows -- sigma_rtheta and
# u_theta alike -- with the ``i k_z`` factor on the C amplitude
# preserved by the d_theta. The D-columns
# enter the cos-sector u_r / sigma_rr rows via the ``(1/r)
# d_theta(psi_z)`` mechanism, with no ``i k_z`` factor since
# psi_z carries no ``d_z``. Row 7 has ``D 0`` because u_z does
# not couple to psi_z under the curl: ``(curl psi)_z = (1/r)
# d_r(r psi_theta)``, no psi_z contribution.)
#
# A real matrix is recovered by:
#
#   * Multiplying the three ``z``-bearing rows (4, 7, 10) by ``i``
#     -- B columns become real, D columns become real, C columns
#     pick up an extra ``i`` factor.
#   * Multiplying the C columns (3, 4, 6) by ``-i`` -- the C
#     entries on rows 1, 2, 5, 8 (i.e., ``i*R``) become real;
#     the C entries on rows 4, 7, 10 (which were ``R`` pre-rescale
#     and became ``i*R`` after row*i) become real (net factor
#     ``i * (-i) = 1``); the C entries on sin rows 3, 9 (``i*R``)
#     also become real.
#
# Net effect on det(M_10):
#
#       det(M_10_rescaled) = (i^3) * ((-i)^3) * det(M_10)
#                          = (-i) * (i) * det(M_10)
#                          = +1 * det(M_10)
#
# So the rescaling is determinant-preserving, the same property
# F.1.a.5 relied on. Row 6 (u_theta) does couple to the C amplitude
# at n >= 1 -- see the roadmap-A.8 note in substep F.2.a.2 -- with
# an ``i k_z (n/r) chi`` entry that the col-by-(-i) makes real like
# every other C entry.

# =====================================================================
# Substep F.2.a.6 -- assembly structure (block-diagonal claim
#                    WITHDRAWN; layered determinant is dense)
# =====================================================================
#
# **Erratum (replaces the original F.2.a.6 substep block, which
# claimed ``det(M_10) = det(M_7^cos) * det(M_3^sin)``)**.
#
# The original block-diagonal claim was wrong. The cos / sin
# azimuthal sectors at n=1 do NOT decouple into independent
# sub-blocks. The d_theta operations in u_r, sigma_rtheta, and
# sigma_rz cross-couple amplitude families across BC azimuthal
# sectors:
#
#   * (1/r) d_theta(psi_z) in u_r (cos BC): the SH amplitudes
#     (D_I, D_K, D -- ``sin``-factor potentials) appear in the
#     cos-sector u_r rows (rows 1, 5) with non-zero coefficients.
#     Concretely visible in :func:`_modal_determinant_n1` as
#     ``M14 = -K_1(s a) / a`` (D column non-zero in cos row 1).
#
#   * (1/r) d_theta u_r in sigma_rtheta (sin BC): the cos-sector
#     P / SV amplitudes (B, C and their I-flavour twins) appear
#     in the sin-sector sigma_rtheta rows (rows 3, 9). Visible as
#     ``M32, M33 != 0`` in the existing single-interface form.
#
# So the layered 10x10 modal determinant is a fully-populated
# dense matrix (modulo interface-localization sparsity per F.2.a.4)
# and admits no cos/sin block factorisation. The F.2.d assembly
# computes ``np.linalg.det`` directly on the assembled 10x10.
#
# Implications for the F.2.b and F.2.c row builders:
#
#   * Each row builder (cos or sin sector) returns a shape-(10,)
#     complex array covering all 10 amplitude columns -- not
#     shape-(7,) or shape-(3,) as the original (incorrect) plan
#     called for. The interface- and BC-localization zeros from
#     F.2.a.4 are surfaced as explicit ``0.0`` entries in the
#     row builder, with each zero documented next to its physical
#     reason.
#
#   * The "cross-row K-flavour cancellation at layer=formation"
#     oracle from F.1.b.3 generalises straightforwardly: at
#     layer=formation each of the seven shared K columns
#     (B_K + B, C_K + C, D_K + D) cancels in every r=b row
#     (rows 5-10) where the corresponding column pair is non-
#     zero. The full K-cancellation invariant is row-by-row
#     and a strong correctness test even without the (now-
#     withdrawn) block factorisation.
#
# Why the block-diagonal claim was tempting: at azimuthal order n,
# Kurkjian & Chang (1986) decompose the field into a single n-mode
# system (``decoupling`` from neighbouring n's, hence eq. 4-9's
# (n+1)x(n+1) modal matrix). The n=1 single-interface form is
# 4x4. The "cos / sin sector" labels in substep 1.1 of the
# pre-existing module describe which AZIMUTHAL FACTOR each BC
# carries, NOT a partition of amplitudes into independent
# sub-systems. After stripping the azimuthal factor, all four
# n=1 amplitudes generically appear in every BC row -- the
# decoupling is between azimuthal orders n, not within n=1.
#
# This substep block is preserved (rather than deleted) so the
# git history of this file documents the misconception and the
# fix; readers tracing the F.2 derivation will not silently
# re-invent the bad claim.

# =====================================================================
# Substep F.2.a.7 -- self-check protocol (degenerate-limit collapses)
# =====================================================================
#
# Three structural identities the layered n=1 determinant must
# satisfy by construction; landed as tests in F.2.e:
#
# (a) Layer = formation. With the annulus material identical to the
#     formation, the layered determinant must vanish at the same
#     k_z as :func:`_modal_determinant_n1` at every frequency.
#     Tested via end-to-end ``flexural_dispersion_layered`` vs
#     ``flexural_dispersion`` with ``rtol=1e-8``. Same as F.1's
#     primary oracle.
#
# (b) Thickness -> 0. The rows at r=b approach the rows at r=a in
#     the limit ``b -> a``; the converged k_z must approach the
#     single-interface value continuously.
#
# (c) Real-valued post-rescale assembly. At every (k_z, omega) in
#     the bound regime, the assembled 10x10 matrix (with the
#     substep-F.2.a.5 row / column rescaling applied) must have
#     identically zero imaginary part to floating-point precision.
#     Verified at every per-row test in F.2.b / F.2.c via the
#     ``row.imag == 0`` invariant; verified at the assembly level
#     in F.2.d via the determinant's imaginary part. (The original
#     "block-diagonal evaluation" framing in this slot was
#     withdrawn together with substep F.2.a.6's block-diagonal
#     claim; see that block for the erratum.)
#
# References for the layered n=1 derivation: same as the n=1
# single-interface set (Schmitt 1988, Kurkjian & Chang 1986,
# Paillet & Cheng 1991 ch. 4) plus Tang & Cheng 2004 for the
# layered generalisation that feeds plan item G. Section number
# dropped: this said "sect. 7.1" and the book has six chapters.


# =====================================================================
# Substep F.2.c.1 -- row 3 of the n=1 layered determinant (r = a)
# =====================================================================
#
# BC3: ``sigma_rtheta^{(m)}(a) = 0`` (sin-sector tangential-shear
# vanishing at the fluid-annulus interface; fluid carries no shear).
# First sin-sector row of the F.2 chain. Shape-(10,) array per the
# corrected substep F.2.a.6 (the layered 10x10 is dense; rows are
# NOT block-decomposed by azimuthal sector).
#
# Coefficients derived from the substep F.2.a.2 displacement
# decompositions and substep F.2.a.3 stress formulae. Each amplitude
# contributes through both ``(1/r) d_theta u_r`` and ``d_r u_theta -
# u_theta / r`` channels of the sigma_rtheta = mu (eps_rtheta + ...)
# combination; the full per-amplitude derivation lives in the n=1
# single-interface substep block (lines ~736+) for the K-flavour and
# is mirrored here with the F.1.a.2 sign-flip pattern for the
# I-flavour.
#
# Pre-rescale coefficients, by amplitude:
#
#   A:    0                                       (fluid no shear)
#   B_I:  +2 mu_m (-p_m I_0(p_m a)/a + 2 I_1(p_m a)/a^2)
#         (sign flip on the d_r-induced ``p_m I_0/a`` term vs B_K)
#   B_K:  +2 mu_m (+p_m K_0(p_m a)/a + 2 K_1(p_m a)/a^2)
#         (matches M32 of :func:`_modal_determinant_n1` at
#         layer=formation with ``p_m -> p, mu_m -> mu``)
#   C_I:  +i k_z mu_m I_1(s_m a) / a
#   C_K:  +i k_z mu_m K_1(s_m a) / a
#         (matches M33's pre-rescale form)
#   B:    0  (formation, r > b -- doesn't reach r = a)
#   C:    0  (formation, r > b)
#   D_I:  -mu_m (s_m^2 I_1(s_m a) - 2 s_m I_0(s_m a)/a + 4 I_1(s_m a)/a^2)
#         (sign flip on the d_r-induced ``s_m I_0/a`` term vs D_K)
#   D_K:  -mu_m (s_m^2 K_1(s_m a) + 2 s_m K_0(s_m a)/a + 4 K_1(s_m a)/a^2)
#         (matches M34 at layer=formation)
#   D:    0  (formation, r > b)
#
# Imaginary-power pattern: B columns real, C columns imaginary
# (i*R; the i k_z factor enters via the ``(1/r) d_theta u_r``
# channel applied to the ``-i k_z C K_1(s r)`` C-amplitude
# contribution to u_r), D columns real. Matches the substep-F.2.a.5
# row-3 pattern entry ``A 0 | B R | C i*R | D R``.
#
# Phase rescale: row 3 is NOT z-derivative-bearing in the substep-
# F.2.a.5 sense (no row * i scaling). Only the column-by-(-i) on
# C_I, C_K, C is applied; that lands the C entries in real form
# post-rescale.


def _layered_n1_row3_at_a(
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
    Row 3 of the n=1 layered modal determinant evaluated at the
    fluid-annulus interface ``r = a``.

    Encodes the tangential-shear vanishing BC
    ``sigma_rtheta^{(m)}(a) = 0`` in the sin sector. Returns the
    ten post-rescale coefficients in the column order pinned by
    substep F.2.a.4: ``[A | B_I, B_K, C_I, C_K | B, C |
    D_I, D_K | D]``.

    Parameters
    ----------
    kz : float
        Trial axial wavenumber (rad / m).
    omega : float
        Angular frequency (rad / s).
    vp, vs : float
        Formation half-space P / S velocities (m/s). Carried for
        signature uniformity; not used by row 3 (formation
        columns are zero at r=a).
    rho : float
        Formation density (kg/m^3). Same as above; not used.
    vf, rho_f : float
        Fluid velocity / density. Not used (fluid carries no shear);
        carried for signature uniformity.
    a : float
        Fluid-annulus interface radius (m).
    layer : BoreholeLayer
        The annular layer; sets ``mu_m``, ``p_m``, ``s_m``.

    Returns
    -------
    ndarray, shape (10,) complex
        Coefficients of (A, B_I, B_K, C_I, C_K, B, C, D_I, D_K, D)
        in row 3. Real-valued in the bound regime.

    See Also
    --------
    _modal_determinant_n1 : The n=1 single-interface form. At
        layer=formation, ``row[2]`` (annulus B_K) matches ``M32``
        bit-exactly; ``row[4]`` (C_K) matches ``M33``; ``row[8]``
        (D_K) matches ``M34``; ``row[0]`` matches ``M31 = 0``.
    """
    del vp, vs, rho, rho_f  # not used by row 3 (formation cols zero)
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=layer.vp,
        vs=layer.vs,
        vf=vf,
        layer=layer,
    )
    del F_f  # row 3 doesn't touch the fluid column

    I0_pm_a = float(special.iv(0, p_m * a))
    I1_pm_a = float(special.iv(1, p_m * a))
    K0_pm_a = float(special.kv(0, p_m * a))
    K1_pm_a = float(special.kv(1, p_m * a))
    I0_sm_a = float(special.iv(0, s_m * a))
    I1_sm_a = float(special.iv(1, s_m * a))
    K0_sm_a = float(special.kv(0, s_m * a))
    K1_sm_a = float(special.kv(1, s_m * a))

    mu_m = layer.rho * layer.vs * layer.vs

    row: np.ndarray = np.zeros(10, dtype=complex)
    # A column: fluid carries no shear.
    row[0] = 0.0
    # B_I column (sign-flipped d_r-induced term).
    row[1] = 2.0 * mu_m * (-p_m * I0_pm_a / a + 2.0 * I1_pm_a / (a * a))
    # B_K column (matches M32 at layer=formation).
    row[2] = 2.0 * mu_m * (+p_m * K0_pm_a / a + 2.0 * K1_pm_a / (a * a))
    # SV columns (Hansen form) -- roadmap A.8: u = curl curl(chi z)
    # has radial, azimuthal AND axial components; the azimuthal-only
    # vector potential the old columns encoded is not a solution of
    # the elastodynamic equations for n >= 1.
    # C_I column (post-rescale; col-by-(-i) cancels the +i factor).
    row[3] = 2.0 * kz * mu_m * (-s_m * I0_sm_a / a + 2.0 * I1_sm_a / (a * a))
    # C_K column (matches M33 at layer=formation).
    row[4] = 2.0 * kz * mu_m * (+s_m * K0_sm_a / a + 2.0 * K1_sm_a / (a * a))
    # Formation columns (B, C, D) vanish at r = a.
    row[5] = 0.0
    row[6] = 0.0
    # D_I column (sign-flipped d_r-induced ``s_m I_0/a`` term).
    row[7] = -mu_m * (
        s_m * s_m * I1_sm_a - 2.0 * s_m * I0_sm_a / a + 4.0 * I1_sm_a / (a * a)
    )
    # D_K column (matches M34 at layer=formation).
    row[8] = -mu_m * (
        s_m * s_m * K1_sm_a + 2.0 * s_m * K0_sm_a / a + 4.0 * K1_sm_a / (a * a)
    )
    # D column (formation): zero at r = a.
    row[9] = 0.0
    return row


# =====================================================================
# Substep F.2.c.2 -- row 6 of the n=1 layered determinant (r = b)
# =====================================================================
#
# BC6: ``u_theta^{(m)}(b) - u_theta^{(s)}(b) = 0`` (sin-sector
# tangential-displacement continuity at the annulus-formation
# interface). Genuinely new BC type at the layered case (no
# single-interface analog: the fluid-solid interface at r=a
# replaces u_theta continuity with sigma_rtheta = 0 -- inviscid
# fluid imposes no tangential-shear constraint on the formation).
#
# Coefficients from substep F.2.a.2's u_theta formulae:
#
#   u_theta^{(m,K)} = -B_K K_1(p_m r)/r - i k_z C_K K_1(s_m r)/r
#                                        + D_K [s_m K_0(s_m r) + K_1(s_m r)/r]
#   u_theta^{(m,I)} = -B_I I_1(p_m r)/r - i k_z C_I I_1(s_m r)/r
#                                        + D_I [-s_m I_0(s_m r) + I_1(s_m r)/r]
#                                          (sign flip on +s I_0)
#   u_theta^{(s)}   = -B   K_1(p r)/r   - i k_z C   K_1(s r)/r
#                                        + D   [s   K_0(s r) + K_1(s r)/r]
#
# C DOES appear in u_theta at n >= 1 -- roadmap A.8. The Hansen SV
# field ``curl curl(chi z)`` carries ``u_theta = i k_z (n/r) chi``,
# which vanishes only at n = 0. Substep F.2.a.2 originally asserted
# the opposite, because the C columns then encoded an azimuthal-only
# vector potential ``psi_theta e_theta``, which has no u_theta at
# all -- and is not a solution of the elastodynamic equations for
# n >= 1. Columns 3, 4 and 6 are therefore non-zero in row 6.
#
# Subtracting (annulus - formation):
#
#       Row 6 (pre-rescale; the B and D entries are real, the C
#       entries carry the ``i k_z`` of u_theta) =
#
#           [  0,                                  (A; fluid r<a)
#             -I_1(p_m b) / b,                     (B_I)
#             -K_1(p_m b) / b,                     (B_K)
#             -i k_z I_1(s_m b) / b,               (C_I)
#             -i k_z K_1(s_m b) / b,               (C_K)
#             +K_1(p b) / b,                       (B; subtracted)
#             +i k_z K_1(s b) / b,                 (C; subtracted)
#             -s_m I_0(s_m b) + I_1(s_m b) / b,    (D_I; sign flip on s_m I_0)
#             +s_m K_0(s_m b) + K_1(s_m b) / b,    (D_K)
#             -s K_0(s b) - K_1(s b) / b ]        (D; subtracted)
#
# Imaginary-power pattern: matches the F.2.a.5 row-6 entry
# ``A 0 | B R | C i*R | D R``. Phase rescale: row 6 is NOT
# z-derivative-bearing (no row * i); the column-by-(-i) on the C
# columns makes their ``i k_z`` entries real, so the row is real
# post-rescale.
#
# Substep-F.2.a.7 (a) K-flavour cancellation at layer=formation:
# B_K + B = 0, C_K + C = 0 and D_K + D = 0.


def _layered_n1_row6_at_b(
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
    Row 6 of the n=1 layered modal determinant evaluated at the
    annulus-formation interface ``r = b = a + layer.thickness``.

    Encodes the tangential-displacement continuity BC
    ``u_theta^{(m)}(b) - u_theta^{(s)}(b) = 0`` in the sin sector.
    Genuinely new BC type: the single-interface n=1 form has no
    u_theta continuity row (the fluid-solid interface replaces it
    with sigma_rtheta = 0). Returns the ten post-rescale
    coefficients in the column order pinned by substep F.2.a.4.

    Parameters
    ----------
    kz : float
        Trial axial wavenumber (rad / m).
    omega : float
        Angular frequency (rad / s).
    vp, vs : float
        Formation half-space P / S velocities (m/s); set the
        formation radial wavenumbers ``p, s`` used by columns 5, 9.
    rho : float
        Formation density. Carried for signature uniformity;
        not used by row 6 (no stress terms; ``mu`` doesn't appear).
    vf, rho_f : float
        Fluid velocity / density. Not used (fluid doesn't reach
        r=b); carried for signature uniformity.
    a : float
        Fluid-annulus interface radius (m); ``b = a + layer.thickness``.
    layer : BoreholeLayer
        The annular layer.

    Returns
    -------
    ndarray, shape (10,) complex
        Coefficients of (A, B_I, B_K, C_I, C_K, B, C, D_I, D_K, D)
        in row 6. Real-valued in the bound regime.

    See Also
    --------
    _layered_n1_row3_at_a : The other sin-sector row at r=a; same
        no-row-rescale pattern. Row 6 differs in (a) being a u_theta
        BC (not sigma_rtheta), (b) having a zero C column entirely,
        and (c) having non-zero formation columns (B, D at r=b).
    """
    del rho, rho_f  # not used by row 6; kept for signature uniformity
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

    I0_sm_b = float(special.iv(0, s_m * b))
    I1_pm_b = float(special.iv(1, p_m * b))
    I1_sm_b = float(special.iv(1, s_m * b))
    K0_sm_b = float(special.kv(0, s_m * b))
    K1_pm_b = float(special.kv(1, p_m * b))
    K1_sm_b = float(special.kv(1, s_m * b))
    K0_s_b = float(special.kv(0, s * b))
    K1_p_b = float(special.kv(1, p * b))
    K1_s_b = float(special.kv(1, s * b))

    row: np.ndarray = np.zeros(10, dtype=complex)
    # A column: fluid r<a; doesn't reach r=b.
    row[0] = 0.0
    # B_I column (annulus P, regular branch).
    row[1] = -I1_pm_b / b
    # B_K column (annulus P, singular branch).
    row[2] = -K1_pm_b / b
    # C columns -- roadmap A.8. They are NOT zero: the Hansen SV
    # field ``curl curl(chi z)`` carries u_theta = i k_z (n/r) chi,
    # which vanishes only at n = 0. The old azimuthal-only vector
    # potential had no u_theta at all, which is what made these
    # entries look structurally absent.
    row[3] = -kz * I1_sm_b / b
    row[4] = -kz * K1_sm_b / b
    # B column (formation P; sign-flipped vs B_K because subtracted).
    row[5] = +K1_p_b / b
    # C column (formation; subtracted -- see the annulus C columns).
    row[6] = +kz * K1_s_b / b
    # D_I column (annulus SH; sign flip on the d_r-induced ``s_m I_0`` term).
    row[7] = -s_m * I0_sm_b + I1_sm_b / b
    # D_K column (annulus SH).
    row[8] = +s_m * K0_sm_b + K1_sm_b / b
    # D column (formation SH; subtracted).
    row[9] = -s * K0_s_b - K1_s_b / b
    return row


# =====================================================================
# Substep F.2.c.3 -- row 9 of the n=1 layered determinant (r = b)
# =====================================================================
#
# BC9: ``sigma_rtheta^{(m)}(b) - sigma_rtheta^{(s)}(b) = 0``
# (sin-sector tangential-stress continuity at the annulus-formation
# interface). Closes substep F.2.c. Same algebraic structure as
# row 3 at r=a -- annulus side reuses the row-3 form evaluated at
# r=b; formation side is the single-interface n=1 sigma_rtheta
# formula, subtracted.
#
# Coefficients (post-rescale; B real, C real after col-by-(-i),
# D real -- same imaginary-power pattern as row 3):
#
#       Row 9 = [
#           0,                                                  # A
#          +2 mu_m (-p_m I_0(p_m b)/b + 2 I_1(p_m b)/b^2),     # B_I
#          +2 mu_m (+p_m K_0(p_m b)/b + 2 K_1(p_m b)/b^2),     # B_K
#          +k_z mu_m I_1(s_m b) / b,                            # C_I
#          +k_z mu_m K_1(s_m b) / b,                            # C_K
#          -2 mu (p K_0(p b)/b + 2 K_1(p b)/b^2),               # B (subtracted)
#          -k_z mu K_1(s b) / b,                                # C (subtracted)
#          -mu_m (s_m^2 I_1 - 2 s_m I_0/b + 4 I_1/b^2),         # D_I
#          -mu_m (s_m^2 K_1 + 2 s_m K_0/b + 4 K_1/b^2),         # D_K
#          +mu (s^2 K_1 + 2 s K_0/b + 4 K_1/b^2),               # D (subtracted)
#       ]
#
# Substep-F.2.a.7 (a) K-flavour cancellation at layer=formation:
# all three K/non-K pairs cancel (B_K + B = 0, C_K + C = 0,
# D_K + D = 0) -- the strongest correctness oracle for r=b
# sin-sector rows since no single-interface analog exists for
# rows 5-10.
#
# Cross-row identity: at layer=formation, row 9's annulus
# K-flavour entries match row 3's M32, M33, M34 forms evaluated
# at r=b instead of r=a (same underlying sigma_rtheta formula at
# both interfaces; the only difference is the evaluation radius
# and the subtracted-formation contribution).


def _layered_n1_row9_at_b(
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
    Row 9 of the n=1 layered modal determinant evaluated at the
    annulus-formation interface ``r = b = a + layer.thickness``.

    Encodes the tangential-stress continuity BC
    ``sigma_rtheta^{(m)}(b) - sigma_rtheta^{(s)}(b) = 0`` in the
    sin sector. Closes the F.2.c sin-sector chain. Returns the
    ten post-rescale coefficients.

    Parameters
    ----------
    kz : float
        Trial axial wavenumber (rad / m).
    omega : float
        Angular frequency (rad / s).
    vp, vs, rho : float
        Formation half-space P / S velocities and density. All
        used (mu = rho * vs^2 sets the formation B / C / D
        coefficients on columns 5, 6, 9).
    vf, rho_f : float
        Fluid velocity / density. Not used (fluid doesn't reach
        r=b); carried for signature uniformity.
    a : float
        Fluid-annulus interface radius (m); ``b = a + layer.thickness``.
    layer : BoreholeLayer
        The annular layer; sets ``mu_m, p_m, s_m``.

    Returns
    -------
    ndarray, shape (10,) complex
        Coefficients of (A, B_I, B_K, C_I, C_K, B, C, D_I, D_K, D)
        in row 9. Real-valued in the bound regime.

    See Also
    --------
    _layered_n1_row3_at_a : Same physical BC at the first
        interface r=a. The annulus-side entries are identical
        in form (just evaluated at r=b vs r=a); the formation
        columns are non-zero in row 9 (vs zero in row 3).
    """
    del rho_f  # not used by row 9; kept for signature uniformity
    F_f, p_m, s_m, p, s = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=vp,
        vs=vs,
        vf=vf,
        layer=layer,
    )
    del F_f  # row 9 doesn't touch the fluid column
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
    mu = rho * vs * vs

    row: np.ndarray = np.zeros(10, dtype=complex)
    # A column: fluid carries no shear; doesn't reach r=b.
    row[0] = 0.0
    # B_I (sign-flipped d_r-induced ``p_m I_0/b`` term).
    row[1] = 2.0 * mu_m * (-p_m * I0_pm_b / b + 2.0 * I1_pm_b / (b * b))
    # B_K (matches row 3 B_K form at r=b instead of r=a).
    row[2] = 2.0 * mu_m * (+p_m * K0_pm_b / b + 2.0 * K1_pm_b / (b * b))
    # SV columns (Hansen form) -- roadmap A.8: u = curl curl(chi z)
    # has radial, azimuthal AND axial components; the azimuthal-only
    # vector potential the old columns encoded is not a solution of
    # the elastodynamic equations for n >= 1.
    # C_I, C_K (post-rescale; col-by-(-i) cancels +i factor).
    row[3] = 2.0 * kz * mu_m * (-s_m * I0_sm_b / b + 2.0 * I1_sm_b / (b * b))
    row[4] = 2.0 * kz * mu_m * (+s_m * K0_sm_b / b + 2.0 * K1_sm_b / (b * b))
    # B column (formation; subtracted, sign-flipped vs B_K at layer=formation).
    row[5] = -2.0 * mu * (+p * K0_p_b / b + 2.0 * K1_p_b / (b * b))
    # C column (formation; subtracted, post-rescale).
    row[6] = -2.0 * kz * mu * (s * K0_s_b / b + 2.0 * K1_s_b / (b * b))
    # D_I (sign-flipped d_r-induced ``s_m I_0/b`` term).
    row[7] = -mu_m * (
        s_m * s_m * I1_sm_b - 2.0 * s_m * I0_sm_b / b + 4.0 * I1_sm_b / (b * b)
    )
    # D_K (matches row 3 D_K form at r=b).
    row[8] = -mu_m * (
        s_m * s_m * K1_sm_b + 2.0 * s_m * K0_sm_b / b + 4.0 * K1_sm_b / (b * b)
    )
    # D column (formation; subtracted, sign-flipped vs D_K at layer=formation).
    row[9] = +mu * (s * s * K1_s_b + 2.0 * s * K0_s_b / b + 4.0 * K1_s_b / (b * b))
    return row


# =====================================================================
# Substep F.2.b.1 -- row 1 of the n=1 layered determinant (r = a)
# =====================================================================
#
# BC1: ``u_r^{(f)}(a) - u_r^{(m)}(a) = 0`` (cos-sector continuity
# of radial displacement at the fluid-annulus interface). First
# cos-sector row of the F.2 chain. Returns shape-(10,) per the
# F.2.a.6 dense-matrix convention.
#
# Coefficients from substep F.2.a.2:
#
#   u_r^{(f)} = (A / (rho_f omega^2)) [F_f I_0(F_f r) - I_1(F_f r)/r] cos(theta)
#
#   u_r^{(m)} =  B_I [p_m I_0(p_m r) - I_1(p_m r)/r]     (d_r phi^{(m,I)})
#              + B_K [-p_m K_0(p_m r) - K_1(p_m r)/r]    (d_r phi^{(m,K)})
#              + D_I I_1(s_m r) / r                       ((1/r) d_theta psi_z^{(m,I)})
#              + D_K K_1(s_m r) / r                       ((1/r) d_theta psi_z^{(m,K)})
#              - i k_z C_I I_1(s_m r)                     (-d_z psi_theta^{(m,I)})
#              - i k_z C_K K_1(s_m r)                     (-d_z psi_theta^{(m,K)})
#                                                          (cos sector)
#
# The D contribution is GENUINELY NEW vs the F.1 n=0 layered case
# (substep F.1.b.2.a row 1 had no D column because at n=0 the
# (1/r) d_theta psi_z term is killed by the axisymmetric ansatz).
# At n=1 the SH amplitude D appears directly in cos-sector u_r
# rows -- this is one of the cross-sector couplings that breaks
# the (now-withdrawn) F.2.a.6 block-diagonal claim.
#
# Subtracting (fluid - annulus) and stripping cos(theta):
#
#       Row 1 (pre-rescale) =
#           [  +(F_f I_0(F_f a) - I_1(F_f a)/a) / (rho_f omega^2),  # A
#              -p_m I_0(p_m a) + I_1(p_m a)/a,                       # B_I
#              +p_m K_0(p_m a) + K_1(p_m a)/a,                       # B_K
#              +i k_z I_1(s_m a),                                     # C_I
#              +i k_z K_1(s_m a),                                     # C_K
#               0,                                                    # B
#               0,                                                    # C
#              -I_1(s_m a) / a,                                       # D_I
#              -K_1(s_m a) / a,                                       # D_K
#               0 ]                                                   # D
#
# Imaginary-power pattern: A R | B R | C i*R | D R -- matches
# substep F.2.a.5's row-1 entry. Phase rescale: row 1 is NOT
# z-derivative-bearing; only column-by-(-i) on C_I, C_K is
# applied. Post-rescale C entries are real.


def _layered_n1_row1_at_a(
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
    Row 1 of the n=1 layered modal determinant evaluated at the
    fluid-annulus interface ``r = a``.

    Encodes the radial-displacement continuity BC
    ``u_r^{(f)}(a) - u_r^{(m)}(a) = 0`` in the cos sector. Returns
    the ten post-rescale coefficients.

    Parameters
    ----------
    kz : float
        Trial axial wavenumber (rad / m).
    omega : float
        Angular frequency (rad / s).
    vp, vs : float
        Formation half-space P / S velocities (m/s). Carried for
        signature uniformity; not used by row 1 (formation columns
        zero at r=a).
    rho : float
        Formation density. Same as above; not used.
    vf : float
        Borehole-fluid velocity (m/s).
    rho_f : float
        Borehole-fluid density (kg/m^3).
    a : float
        Fluid-annulus interface radius (m).
    layer : BoreholeLayer
        The annular layer; sets ``p_m``, ``s_m``.

    Returns
    -------
    ndarray, shape (10,) complex
        Coefficients of (A, B_I, B_K, C_I, C_K, B, C, D_I, D_K, D)
        in row 1. Real-valued in the bound regime.

    See Also
    --------
    _modal_determinant_n1 : The n=1 single-interface form. At
        layer=formation, ``row[0] = M11``, ``row[2] = M12``,
        ``row[4] = M13``, ``row[8] = M14`` bit-exactly.
    _layered_n0_row1_at_a : The F.1 n=0 layered counterpart. The
        n=1 form adds D-amplitude contributions (cols 7-9) via
        the ``(1/r) d_theta psi_z`` cross-coupling, which is
        absent at n=0.
    """
    del vp, vs, rho  # not used by row 1 (formation cols zero)
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=layer.vp,
        vs=layer.vs,
        vf=vf,
        layer=layer,
    )

    I0_Ff_a = float(special.iv(0, F_f * a))
    I1_Ff_a = float(special.iv(1, F_f * a))
    I0_pm_a = float(special.iv(0, p_m * a))
    I1_pm_a = float(special.iv(1, p_m * a))
    K0_pm_a = float(special.kv(0, p_m * a))
    K1_pm_a = float(special.kv(1, p_m * a))
    I0_sm_a = float(special.iv(0, s_m * a))
    I1_sm_a = float(special.iv(1, s_m * a))
    K0_sm_a = float(special.kv(0, s_m * a))
    K1_sm_a = float(special.kv(1, s_m * a))

    row: np.ndarray = np.zeros(10, dtype=complex)
    # A column (matches M11 at layer=formation).
    row[0] = (F_f * I0_Ff_a - I1_Ff_a / a) / (rho_f * omega**2)
    # B_I column (sign-flipped d_r-induced ``p_m I_0`` term).
    row[1] = -p_m * I0_pm_a + I1_pm_a / a
    # B_K column (matches M12 at layer=formation).
    row[2] = +p_m * K0_pm_a + K1_pm_a / a
    # SV columns (Hansen form) -- roadmap A.8: u = curl curl(chi z)
    # has radial, azimuthal AND axial components; the azimuthal-only
    # vector potential the old columns encoded is not a solution of
    # the elastodynamic equations for n >= 1.
    # C_I column (post-rescale; col-by-(-i) cancels +i factor).
    row[3] = +kz * (I1_sm_a / a - s_m * I0_sm_a)
    # C_K column (matches M13 at layer=formation; post-rescale).
    row[4] = +kz * (s_m * K0_sm_a + K1_sm_a / a)
    # Formation columns (B, C) vanish at r = a.
    row[5] = 0.0
    row[6] = 0.0
    # D_I column (SH cross-coupling via (1/r) d_theta psi_z;
    # KEEP-sign on the I_1/r term per F.1.a.2).
    row[7] = -I1_sm_a / a
    # D_K column (matches M14 at layer=formation).
    row[8] = -K1_sm_a / a
    # D column (formation; zero at r = a).
    row[9] = 0.0
    return row


# =====================================================================
# Substep F.2.b.2 -- row 2 of the n=1 layered determinant (r = a)
# =====================================================================
#
# BC2: ``-(sigma_rr^{(m)}(a) + P^{(f)}(a)) = 0`` (cos-sector
# normal-stress balance at the fluid-annulus interface; row negated
# for visual parallel with the n=0 / n=1 single-interface forms).
# Lame-reduction row -- the algebraically heaviest of the cos-
# sector rows at r=a.
#
# Coefficients via the F.2.a.3 stress derivation. The annulus K-
# flavour entries mirror the M21-M24 form of
# :func:`_modal_determinant_n1`; the I-flavour twins follow the
# F.1.a.2 sign-flip rule:
#
#   * "(2 k_z^2 - k_Sm^2) X_1" terms KEEP sign across I/K (direct
#     terms; X_1 keeps natural index).
#   * "2 p_m X_0/a" terms FLIP sign (derivative-induced from
#     X_1' = -X_0 - X_1/(p_m r) for K, +X_0 - X_1/(p_m r) for I).
#   * "4 X_1/a^2" terms KEEP sign (direct).
#   * "s_m X_0/a" derivative-induced terms FLIP sign in D entries.
#   * "X_1/a", "X_1/a^2" direct terms KEEP sign.
#
# Plus the genuinely new D-amplitude column at n=1 (D contributes
# to u_r via ``(1/r) d_theta psi_z``, hence to sigma_rr via
# ``2 mu d_r u_r``):
#
#       Coefficient of D_K in -(sigma_rr + P) at r=a
#           = +2 mu_m [s_m K_0(s_m a) / a + 2 K_1(s_m a) / a^2]
#
# (matches M24 at layer=formation).
#
# Phase rescale: row 2 is NOT z-derivative-bearing; column-by-(-i)
# on C_I, C_K only. Post-rescale C entries flip from i*R to R.


def _layered_n1_row2_at_a(
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
    Row 2 of the n=1 layered modal determinant evaluated at the
    fluid-annulus interface ``r = a``.

    Encodes the negated normal-stress balance BC
    ``-(sigma_rr^{(m)}(a) + P^{(f)}(a)) = 0`` in the cos sector.
    Returns the ten post-rescale coefficients.

    Parameters
    ----------
    kz : float
        Trial axial wavenumber (rad / m).
    omega : float
        Angular frequency (rad / s).
    vp, vs : float
        Formation half-space P / S velocities. Carried for
        signature uniformity; not used by row 2.
    rho : float
        Formation density. Same as above; not used.
    vf, rho_f : float
        Fluid velocity / density.
    a : float
        Fluid-annulus interface radius (m).
    layer : BoreholeLayer
        The annular layer; sets ``mu_m, k_Sm, p_m, s_m``.

    Returns
    -------
    ndarray, shape (10,) complex
        Coefficients of (A, B_I, B_K, C_I, C_K, B, C, D_I, D_K, D)
        in row 2. Real-valued in the bound regime.

    See Also
    --------
    _modal_determinant_n1 : The n=1 single-interface form. At
        layer=formation, ``row[0] = M21``, ``row[2] = M22``,
        ``row[4] = M23``, ``row[8] = M24`` bit-exactly.
    _layered_n0_row2_at_a : The F.1 n=0 layered counterpart. The
        n=1 form adds D_I, D_K columns via the
        ``(1/r) d_theta psi_z`` cross-coupling absent at n=0,
        and switches Bessel index 0 -> 1 in the Lame combination
        (M22 here uses K_1 vs K_0 for the n=0 analog).
    """
    del vp, vs, rho  # not used by row 2 (formation cols zero)
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=layer.vp,
        vs=layer.vs,
        vf=vf,
        layer=layer,
    )

    I0_pm_a = float(special.iv(0, p_m * a))
    I1_pm_a = float(special.iv(1, p_m * a))
    K0_pm_a = float(special.kv(0, p_m * a))
    K1_pm_a = float(special.kv(1, p_m * a))
    I0_sm_a = float(special.iv(0, s_m * a))
    I1_sm_a = float(special.iv(1, s_m * a))
    K0_sm_a = float(special.kv(0, s_m * a))
    K1_sm_a = float(special.kv(1, s_m * a))
    I1_Ff_a = float(special.iv(1, F_f * a))

    mu_m = layer.rho * layer.vs * layer.vs
    kSm2 = (omega / layer.vs) ** 2
    two_kz2_minus_kSm2 = 2.0 * kz * kz - kSm2

    row: np.ndarray = np.zeros(10, dtype=complex)
    # A column: -P^{(f)}(a) coefficient (matches M21 at layer=formation).
    row[0] = -I1_Ff_a
    # B_I column (sign-flipped d_r-induced ``2 p_m I_0/a`` term).
    row[1] = -mu_m * (
        two_kz2_minus_kSm2 * I1_pm_a - 2.0 * p_m * I0_pm_a / a + 4.0 * I1_pm_a / (a * a)
    )
    # B_K column (matches M22 at layer=formation).
    row[2] = -mu_m * (
        two_kz2_minus_kSm2 * K1_pm_a + 2.0 * p_m * K0_pm_a / a + 4.0 * K1_pm_a / (a * a)
    )
    # SV columns (Hansen form) -- roadmap A.8: u = curl curl(chi z)
    # has radial, azimuthal AND axial components; the azimuthal-only
    # vector potential the old columns encoded is not a solution of
    # the elastodynamic equations for n >= 1.
    # C_I column (post-rescale; col-by-(-i) cancels +i factor).
    row[3] = (
        -2.0
        * kz
        * mu_m
        * (s_m * s_m * I1_sm_a - s_m * I0_sm_a / a + 2.0 * I1_sm_a / (a * a))
    )
    # C_K column (matches M23 at layer=formation).
    row[4] = (
        -2.0
        * kz
        * mu_m
        * (s_m * s_m * K1_sm_a + s_m * K0_sm_a / a + 2.0 * K1_sm_a / (a * a))
    )
    # Formation columns (B, C) vanish at r = a.
    row[5] = 0.0
    row[6] = 0.0
    # D_I column (sign-flipped d_r-induced ``s_m I_0/a`` term).
    row[7] = +2.0 * mu_m * (-s_m * I0_sm_a / a + 2.0 * I1_sm_a / (a * a))
    # D_K column (matches M24 at layer=formation).
    row[8] = +2.0 * mu_m * (+s_m * K0_sm_a / a + 2.0 * K1_sm_a / (a * a))
    # D column (formation; zero at r = a).
    row[9] = 0.0
    return row


# =====================================================================
# Substep F.2.b.3 -- row 4 of the n=1 layered determinant (r = a)
# =====================================================================
#
# BC4: ``sigma_rz^{(m)}(a) = 0`` (cos-sector axial-shear vanishing
# at the fluid-annulus interface; fluid carries no shear, so column
# A is identically zero). First z-derivative-bearing cos row of
# the F.2 chain. Per substep F.2.a.5: row * i scaling AND col-by-
# (-i) on C cols.
#
# Pre-rescale imaginary-power pattern (F.2.a.5):
#       Row 4: A 0 | B i*R | C R | D i*R   <- z-bearing
#
# Coefficients via sigma_rz = mu (d_z u_r + d_r u_z) and the
# F.2.a.2 displacement decompositions:
#
#   * B contribution: sigma_rz from B_X = -2 i k_z mu_m B_X
#     (p_m X_0 + X_1/r) for K-flavour (combining d_z u_r and d_r u_z
#     which both contribute -i k_z B p_m K_1' = -i k_z B (p_m K_0 +
#     K_1/r) terms). I-flavour: sigma_rz from B_I = +2 i k_z mu_m B_I
#     (p_m I_0 - I_1/r) (sign flip in the bracket from I_1' = +I_0
#     - I_1/(p_m r)).
#
#   * C contribution: sigma_rz from C_X = +mu_m (k_z^2 + s_m^2) C_X
#     X_1(s_m r) = +mu_m (2 k_z^2 - k_Sm^2) C_X X_1(s_m r). NO
#     ``i k_z`` factor (the i k_z's from d_z u_r and d_r u_z combine
#     with the i k_z on u_r-from-C and the s s' on u_z-from-C; net
#     real). Same outer sign for I and K (direct X_1 term).
#
#   * D contribution: sigma_rz from D_X = +i k_z mu_m D_X X_1(s_m r)/r
#     (from d_z u_r only; d_r u_z from D is zero because u_z carries
#     no D contribution). I-flavour: same sign (direct X_1/r term).
#
# After row * i + col-by-(-i) on C cols:
#
#   * B columns: pre i*R becomes R after row * i.
#   * C columns: pre R becomes i*R after row * i, then R after col-i.
#   * D columns: pre i*R becomes R after row * i.
#
# Post-rescale row 4:
#
#       Row 4 = [
#           0,                                          # A
#           -2 k_z mu_m (p_m I_0(p_m a) - I_1(p_m a)/a),# B_I
#           +2 k_z mu_m (p_m K_0(p_m a) + K_1(p_m a)/a),# B_K -> M42
#           +mu_m (2 k_z^2 - k_Sm^2) I_1(s_m a),        # C_I
#           +mu_m (2 k_z^2 - k_Sm^2) K_1(s_m a),        # C_K -> M43
#            0, 0,                                      # B, C (formation)
#           -k_z mu_m I_1(s_m a) / a,                   # D_I
#           -k_z mu_m K_1(s_m a) / a,                   # D_K -> M44
#            0,                                         # D (formation)
#       ]
#
# I-K sign-flip pattern: B columns have OPPOSITE outer sign and a
# sign-flipped bracket internal structure (-p_m I_0 vs +p_m K_0;
# +I_1/a vs +K_1/a). C columns are direct X_1 -- KEEP sign
# (single-Bessel-term, +I_1 / +K_1, ratio +I_1/K_1). D columns are
# direct X_1/r -- KEEP sign (-I_1/a vs -K_1/a, ratio +I_1/K_1).


def _layered_n1_row4_at_a(
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
    Row 4 of the n=1 layered modal determinant evaluated at the
    fluid-annulus interface ``r = a``.

    Encodes the axial-shear vanishing BC ``sigma_rz^{(m)}(a) = 0``
    in the cos sector. First z-derivative-bearing cos row of the
    F.2 chain; gets row * i AND col-by-(-i) on C cols per substep
    F.2.a.5.

    Parameters
    ----------
    kz : float
        Trial axial wavenumber (rad / m).
    omega : float
        Angular frequency (rad / s).
    vp, vs : float
        Formation half-space P / S velocities. Carried for
        signature uniformity; not used by row 4.
    rho : float
        Formation density. Same as above; not used.
    vf, rho_f : float
        Fluid velocity / density. Not used (fluid carries no
        shear); carried for signature uniformity.
    a : float
        Fluid-annulus interface radius (m).
    layer : BoreholeLayer
        The annular layer; sets ``mu_m, k_Sm, p_m, s_m``.

    Returns
    -------
    ndarray, shape (10,) complex
        Coefficients of (A, B_I, B_K, C_I, C_K, B, C, D_I, D_K, D)
        in row 4. Real-valued in the bound regime.

    See Also
    --------
    _modal_determinant_n1 : The n=1 single-interface form. At
        layer=formation, ``row[0] = M41 = 0``, ``row[2] = M42``,
        ``row[4] = M43``, ``row[8] = M44`` bit-exactly.
    _layered_n0_row3_at_a : The F.1 n=0 layered counterpart for the
        sigma_rz BC. The n=1 form adds the D-amplitude D_I, D_K
        columns (cols 7, 8) via the ``(1/r) d_theta psi_z`` u_r
        cross-coupling absent at n=0, and switches Bessel index
        0 -> 1 in the multi-term entries.
    """
    del vp, vs, rho, rho_f  # not used by row 4
    F_f, p_m, s_m, _, _ = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=layer.vp,
        vs=layer.vs,
        vf=vf,
        layer=layer,
    )
    del F_f  # row 4 doesn't touch the fluid column

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

    row: np.ndarray = np.zeros(10, dtype=complex)
    # A column: fluid carries no shear.
    row[0] = 0.0
    # B_I column (post-rescale; sign-flipped bracket from I_1' = +I_0 - I_1/(p_m r)).
    row[1] = -2.0 * kz * mu_m * (p_m * I0_pm_a - I1_pm_a / a)
    # B_K column (matches M42 at layer=formation).
    row[2] = +2.0 * kz * mu_m * (p_m * K0_pm_a + K1_pm_a / a)
    # SV columns (Hansen form) -- roadmap A.8: u = curl curl(chi z)
    # has radial, azimuthal AND axial components; the azimuthal-only
    # vector potential the old columns encoded is not a solution of
    # the elastodynamic equations for n >= 1.
    # C_I column (post-rescale; row * i AND col-by-(-i), net factor 1).
    row[3] = +mu_m * two_kz2_minus_kSm2 * (I1_sm_a / a - s_m * I0_sm_a)
    # C_K column (matches M43 at layer=formation).
    row[4] = +mu_m * two_kz2_minus_kSm2 * (s_m * K0_sm_a + K1_sm_a / a)
    # Formation columns (B, C) vanish at r = a.
    row[5] = 0.0
    row[6] = 0.0
    # D_I column (post-rescale; row * i flips +i*R to -R).
    row[7] = -kz * mu_m * I1_sm_a / a
    # D_K column (matches M44 at layer=formation).
    row[8] = -kz * mu_m * K1_sm_a / a
    # D column (formation; zero at r = a).
    row[9] = 0.0
    return row


# =====================================================================
# Substep F.2.b.4 -- row 5 of the n=1 layered determinant (r = b)
# =====================================================================
#
# BC5: ``u_r^{(m)}(b) - u_r^{(s)}(b) = 0`` (cos-sector continuity
# of radial displacement at the annulus-formation interface).
# Mirror of row 1 evaluated at r=b, with formation columns now
# non-zero. No single-interface analog at r=b -- primary oracle
# is the F.2.a.7 (a) K-flavour cancellation at layer=formation.
#
# Coefficients via substep F.2.a.2 displacement decompositions
# (annulus side mirrors row 1 at r=b; formation side is the n=1
# single-interface u_r at r=b, subtracted):
#
#       Row 5 (post-rescale) =
#           [  0,                                          # A (fluid r<a)
#             +p_m I_0(p_m b) - I_1(p_m b)/b,             # B_I
#             -p_m K_0(p_m b) - K_1(p_m b)/b,             # B_K
#             -k_z I_1(s_m b),                             # C_I
#             -k_z K_1(s_m b),                             # C_K
#             +p K_0(p b) + K_1(p b)/b,                    # B (subtracted)
#             +k_z K_1(s b),                               # C (subtracted)
#             +I_1(s_m b) / b,                             # D_I
#             +K_1(s_m b) / b,                             # D_K
#             -K_1(s b) / b ]                             # D (subtracted)
#
# K-flavour cancellation pairs at layer=formation:
#   row[2] (B_K) + row[5] (B) = (-p K_0 - K_1/b) + (+p K_0 + K_1/b) = 0
#   row[4] (C_K) + row[6] (C) = (-k_z K_1) + (+k_z K_1) = 0
#   row[8] (D_K) + row[9] (D) = (+K_1/b) + (-K_1/b) = 0
#
# Sign-flow consistency vs row 1: the BC subtraction direction
# flips between row 1 (``u_r^{(f)} - u_r^{(m)} = 0``) and row 5
# (``u_r^{(m)} - u_r^{(s)} = 0``). The annulus contributes with
# OPPOSITE sign in the two rows; consequently row 5's K-flavour
# B_K coefficient (annulus) is the negation of row 1's B_K
# coefficient (annulus, evaluated at r=a vs r=b).


def _layered_n1_row5_at_b(
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
    Row 5 of the n=1 layered modal determinant evaluated at the
    annulus-formation interface ``r = b = a + layer.thickness``.

    Encodes the radial-displacement continuity BC
    ``u_r^{(m)}(b) - u_r^{(s)}(b) = 0`` in the cos sector. Returns
    the ten post-rescale coefficients.

    Parameters
    ----------
    kz : float
        Trial axial wavenumber (rad / m).
    omega : float
        Angular frequency (rad / s).
    vp, vs : float
        Formation half-space P / S velocities (m/s); set the
        formation radial wavenumbers ``p, s`` used by columns 5, 6, 9.
    rho : float
        Formation density (kg/m^3). Carried for signature
        uniformity; not used by row 5 (no stress terms).
    vf, rho_f : float
        Fluid velocity / density. Not used (fluid r<a doesn't
        reach r=b); carried for signature uniformity.
    a : float
        Fluid-annulus interface radius (m); ``b = a + layer.thickness``.
    layer : BoreholeLayer
        The annular layer; sets ``p_m, s_m``.

    Returns
    -------
    ndarray, shape (10,) complex
        Coefficients of (A, B_I, B_K, C_I, C_K, B, C, D_I, D_K, D)
        in row 5. Real-valued in the bound regime.

    See Also
    --------
    _layered_n1_row1_at_a : Same physical BC at r=a. Row 5's
        annulus K-flavour entries have OPPOSITE sign vs row 1's
        (BC subtraction flip; annulus appears with - sign in
        row 1 and + sign in row 5).
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

    row: np.ndarray = np.zeros(10, dtype=complex)
    # A column: fluid r<a; doesn't reach r=b.
    row[0] = 0.0
    # B_I column (annulus; sign flip on p_m I_0 vs B_K's p_m K_0).
    row[1] = +p_m * I0_pm_b - I1_pm_b / b
    # B_K column (annulus, sign flipped vs row 1 due to BC subtraction).
    row[2] = -p_m * K0_pm_b - K1_pm_b / b
    # SV columns (Hansen form) -- roadmap A.8: u = curl curl(chi z)
    # has radial, azimuthal AND axial components; the azimuthal-only
    # vector potential the old columns encoded is not a solution of
    # the elastodynamic equations for n >= 1.
    # C_I column (post-rescale; col-by-(-i) cancels -i factor).
    row[3] = +kz * (s_m * I0_sm_b - I1_sm_b / b)
    # C_K column (post-rescale).
    row[4] = -kz * (s_m * K0_sm_b + K1_sm_b / b)
    # B column (formation; subtracted, sign-flipped vs B_K at layer=formation).
    row[5] = +p * K0_p_b + K1_p_b / b
    # C column (formation; subtracted, post-rescale).
    row[6] = +kz * (s * K0_s_b + K1_s_b / b)
    # D_I column (annulus SH cross-coupling).
    row[7] = +I1_sm_b / b
    # D_K column (annulus SH).
    row[8] = +K1_sm_b / b
    # D column (formation SH; subtracted).
    row[9] = -K1_s_b / b
    return row


# =====================================================================
# Substep F.2.b.5 -- row 7 of the n=1 layered determinant (r = b)
# =====================================================================
#
# BC7: ``u_z^{(m)}(b) - u_z^{(s)}(b) = 0`` (cos-sector continuity
# of axial displacement at the annulus-formation interface).
# Z-derivative-bearing cos row; gets row * i + col-by-(-i) on C
# cols per substep F.2.a.5. Distinctive sparsity at n=1: D
# columns are identically zero because u_z does not couple to
# psi_z under the curl decomposition (curl_z = (1/r) d_r(r
# psi_theta), no psi_z contribution).
#
# Coefficients via substep F.2.a.2 u_z formulae:
#
#   u_z^{(m)}(r) = i k_z (B_I I_1(p_m r) + B_K K_1(p_m r)) cos(theta)
#                  + C_I s_m I_0(s_m r) cos(theta)
#                  - C_K s_m K_0(s_m r) cos(theta)
#                  (from d_z phi^{(m)} + (1/r) d_r [r psi_theta^{(m)}];
#                   D contributes nothing because curl_z drops psi_z)
#
#   u_z^{(s)}(r) = i k_z B K_1(p r) cos(theta) - C s K_0(s r) cos(theta)
#
# Pre-rescale (A 0 | B i*R | C R | D 0 per F.2.a.5):
#
#       Row 7 = [
#           0,                                  # A (fluid r<a)
#          +i k_z I_1(p_m b),                   # B_I
#          +i k_z K_1(p_m b),                   # B_K
#          +s_m I_0(s_m b),                     # C_I
#          -s_m K_0(s_m b),                     # C_K
#          -i k_z K_1(p b),                     # B (subtracted)
#          +s K_0(s b),                         # C (subtracted)
#           0, 0, 0,                            # D's (u_z has no D)
#       ]
#
# Phase rescale (full z-bearing): row * i flips B i*R -> R and
# C R -> i*R (then col-by-(-i) on C cols completes the rescale to
# C R). Post-rescale row 7:
#
#       Row 7 = [
#           0,
#           -k_z I_1(p_m b),                    # B_I
#           -k_z K_1(p_m b),                    # B_K
#           +s_m I_0(s_m b),                    # C_I
#           -s_m K_0(s_m b),                    # C_K
#           +k_z K_1(p b),                      # B (subtracted)
#           +s K_0(s b),                        # C (subtracted)
#           0, 0, 0,                            # D's
#       ]
#
# K-flavour cancellation pairs at layer=formation:
#   row[2] (B_K) + row[5] (B) = -k_z K_1(p_m b) + k_z K_1(p b) = 0
#   row[4] (C_K) + row[6] (C) = -s_m K_0(s_m b) + s K_0(s b) = 0
#   row[8] (D_K) + row[9] (D) = 0 + 0 = 0 (trivially)
#
# Cross-row Bessel-index distinction: row 5 (u_r) uses degree-1
# Bessel functions; row 7 (u_z) uses degree-1 on B and degree-0
# on C/D. Same n=0 / n=1 distinction as F.1.b.3.b row 5 carries.


def _layered_n1_row7_at_b(
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
    Row 7 of the n=1 layered modal determinant evaluated at the
    annulus-formation interface ``r = b = a + layer.thickness``.

    Encodes the axial-displacement continuity BC
    ``u_z^{(m)}(b) - u_z^{(s)}(b) = 0`` in the cos sector. Z-
    derivative-bearing; gets the FULL F.2.a.5 rescale (row * i +
    col-by-(-i) on C cols).

    Distinctive feature: D columns (7, 8, 9) are identically zero
    because u_z does not couple to psi_z under the curl
    decomposition.

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
        not used (no stress terms).
    vf, rho_f : float
        Fluid velocity / density. Not used (fluid r<a doesn't reach r=b);
        carried for signature uniformity.
    a : float
        Fluid-annulus interface radius (m); ``b = a + layer.thickness``.
    layer : BoreholeLayer
        The annular layer; sets ``p_m, s_m``.

    Returns
    -------
    ndarray, shape (10,) complex
        Coefficients of (A, B_I, B_K, C_I, C_K, B, C, D_I, D_K, D)
        in row 7. Real-valued in the bound regime. D columns
        identically zero.
    """
    del rho, rho_f  # not used by row 7; kept for signature uniformity
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
    I1_sm_b = float(special.iv(1, s_m * b))
    K1_pm_b = float(special.kv(1, p_m * b))
    K1_sm_b = float(special.kv(1, s_m * b))
    K1_p_b = float(special.kv(1, p * b))
    K1_s_b = float(special.kv(1, s * b))

    row: np.ndarray = np.zeros(10, dtype=complex)
    # A column: fluid r<a; doesn't reach r=b.
    row[0] = 0.0
    # B_I column (post-rescale; row * i flips +i k_z I_1 to -k_z I_1).
    row[1] = -kz * I1_pm_b
    # B_K column (post-rescale).
    row[2] = -kz * K1_pm_b
    # SV columns (Hansen form) -- roadmap A.8: u = curl curl(chi z)
    # has radial, azimuthal AND axial components; the azimuthal-only
    # vector potential the old columns encoded is not a solution of
    # the elastodynamic equations for n >= 1.
    # C_I column (post-rescale; row * i AND col-by-(-i), net factor 1).
    row[3] = -s_m * s_m * I1_sm_b
    # C_K column (post-rescale).
    row[4] = -s_m * s_m * K1_sm_b
    # B column (formation; subtracted, sign-flipped vs B_K via row * i on -i k_z K_1).
    row[5] = +kz * K1_p_b
    # C column (formation; subtracted, post-rescale).
    row[6] = +s * s * K1_s_b
    # D columns: u_z has no D contribution (curl_z drops psi_z).
    row[7] = 0.0
    row[8] = 0.0
    row[9] = 0.0
    return row


# =====================================================================
# Substep F.2.b.6 -- row 8 of the n=1 layered determinant (r = b)
# =====================================================================
#
# BC8: ``sigma_rr^{(m)}(b) - sigma_rr^{(s)}(b) = 0`` (cos-sector
# normal-stress continuity at the annulus-formation interface).
# Lame-reduction row at the second interface; uses the unnegated
# continuity convention (matching F.1.b.3.c's row 6 choice for
# the n=0 layered analog -- vs row 2's negated ``-(sigma_rr + P)``
# convention which was for visual parallel with the single-
# interface form).
#
# The convention choice is internal: it flips the overall sign of
# the row, which preserves the determinant root. The K-flavour
# cancellation at layer=formation is verifiable in either
# convention.
#
# Coefficients (post-rescale; A 0 | B R | C i*R -> R | D R per
# F.2.a.5; no row scaling, only col-by-(-i) on C cols):
#
#       Row 8 = [
#           0,                                                  # A (fluid r<a)
#          +mu_m [(2 kz^2 - kSm^2) I_1(p_m b)
#                 - 2 p_m I_0(p_m b)/b
#                 + 4 I_1(p_m b)/b^2],                          # B_I
#          +mu_m [(2 kz^2 - kSm^2) K_1(p_m b)
#                 + 2 p_m K_0(p_m b)/b
#                 + 4 K_1(p_m b)/b^2],                          # B_K
#          -2 k_z mu_m (s_m I_0(s_m b) - I_1(s_m b)/b),         # C_I
#          +2 k_z mu_m (s_m K_0(s_m b) + K_1(s_m b)/b),         # C_K
#          -mu [(2 kz^2 - kS^2) K_1(p b)
#               + 2 p K_0(p b)/b
#               + 4 K_1(p b)/b^2],                              # B (subtracted)
#          -2 k_z mu (s K_0(s b) + K_1(s b)/b),                  # C (subtracted)
#          +2 mu_m (s_m I_0(s_m b)/b - 2 I_1(s_m b)/b^2),        # D_I
#          -2 mu_m (s_m K_0(s_m b)/b + 2 K_1(s_m b)/b^2),        # D_K
#          +2 mu (s K_0(s b)/b + 2 K_1(s b)/b^2),                # D (subtracted)
#       ]
#
# Sign-flip pattern (F.1.a.2): in the I-flavour entries,
# derivative-induced ``p_m X_0/b``, ``s_m X_0`` and ``s_m X_0/b``
# terms flip sign vs K-flavour twins; direct ``X_1``, ``X_1/b``,
# ``X_1/b^2`` terms keep sign.
#
# K-flavour cancellation at layer=formation: all three pairs
# (B_K + B, C_K + C, D_K + D) cancel pair-wise. The convention
# flip vs row 2 means row 8's annulus K-flavour entries equal
# the NEGATION of row 2's M22-M24 form evaluated at r=b.


def _layered_n1_row8_at_b(
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
    Row 8 of the n=1 layered modal determinant evaluated at the
    annulus-formation interface ``r = b = a + layer.thickness``.

    Encodes the normal-stress continuity BC
    ``sigma_rr^{(m)}(b) - sigma_rr^{(s)}(b) = 0`` in the cos
    sector. Returns the ten post-rescale coefficients.

    Parameters
    ----------
    kz : float
        Trial axial wavenumber (rad / m).
    omega : float
        Angular frequency (rad / s).
    vp, vs, rho : float
        Formation half-space P / S velocities and density. All
        used (mu = rho * vs^2 sets formation B/C/D coefficients).
    vf, rho_f : float
        Fluid velocity / density. Not used (fluid r<a doesn't reach
        r=b); carried for signature uniformity.
    a : float
        Fluid-annulus interface radius (m); ``b = a + layer.thickness``.
    layer : BoreholeLayer
        The annular layer; sets ``mu_m, k_Sm, p_m, s_m``.

    Returns
    -------
    ndarray, shape (10,) complex
        Coefficients of (A, B_I, B_K, C_I, C_K, B, C, D_I, D_K, D)
        in row 8. Real-valued in the bound regime.

    See Also
    --------
    _layered_n1_row2_at_a : Same physical BC at r=a but using the
        negated ``-(sigma_rr + P)`` convention. Row 8's annulus
        K-flavour entries equal the NEGATION of row 2's M22-M24
        form evaluated at r=b (convention difference).
    _layered_n0_row6_at_b : The F.1 n=0 layered counterpart for
        sigma_rr continuity. Same unnegated convention.
    """
    del rho_f  # not used by row 8; kept for signature uniformity
    F_f, p_m, s_m, p, s = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=vp,
        vs=vs,
        vf=vf,
        layer=layer,
    )
    del F_f  # row 8 doesn't touch the fluid column
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

    row: np.ndarray = np.zeros(10, dtype=complex)
    # A column: fluid r<a doesn't reach r=b.
    row[0] = 0.0
    # B_I (sign-flipped d_r-induced ``2 p_m I_0/b`` term).
    row[1] = +mu_m * (
        two_kz2_minus_kSm2 * I1_pm_b - 2.0 * p_m * I0_pm_b / b + 4.0 * I1_pm_b / (b * b)
    )
    # B_K (annulus, positive sigma_rr form).
    row[2] = +mu_m * (
        two_kz2_minus_kSm2 * K1_pm_b + 2.0 * p_m * K0_pm_b / b + 4.0 * K1_pm_b / (b * b)
    )
    # SV columns (Hansen form) -- roadmap A.8: u = curl curl(chi z)
    # has radial, azimuthal AND axial components; the azimuthal-only
    # vector potential the old columns encoded is not a solution of
    # the elastodynamic equations for n >= 1.
    # C_I (post-rescale; col-by-(-i) cancels -i factor).
    row[3] = (
        +2.0
        * kz
        * mu_m
        * (s_m * s_m * I1_sm_b - s_m * I0_sm_b / b + 2.0 * I1_sm_b / (b * b))
    )
    # C_K (post-rescale).
    row[4] = (
        +2.0
        * kz
        * mu_m
        * (s_m * s_m * K1_sm_b + s_m * K0_sm_b / b + 2.0 * K1_sm_b / (b * b))
    )
    # B column (formation, subtracted; cancels B_K at layer=formation).
    row[5] = -mu * (
        two_kz2_minus_kS2 * K1_p_b + 2.0 * p * K0_p_b / b + 4.0 * K1_p_b / (b * b)
    )
    # C column (formation, subtracted; post-rescale).
    row[6] = -2.0 * kz * mu * (s * s * K1_s_b + s * K0_s_b / b + 2.0 * K1_s_b / (b * b))
    # D_I (sign-flipped d_r-induced ``s_m I_0/b`` term).
    row[7] = +2.0 * mu_m * (s_m * I0_sm_b / b - 2.0 * I1_sm_b / (b * b))
    # D_K (annulus, negative sigma_rr form because d_r [K_1/r] gives
    # -s K_0/r - 2 K_1/r^2 with both negative; outer 2 mu_m gives
    # leading minus).
    row[8] = -2.0 * mu_m * (s_m * K0_sm_b / b + 2.0 * K1_sm_b / (b * b))
    # D column (formation, subtracted; cancels D_K at layer=formation).
    row[9] = +2.0 * mu * (s * K0_s_b / b + 2.0 * K1_s_b / (b * b))
    return row


# =====================================================================
# Substep F.2.b.7 -- row 10 of the n=1 layered determinant (r = b)
# =====================================================================
#
# BC10: ``sigma_rz^{(m)}(b) - sigma_rz^{(s)}(b) = 0`` (cos-sector
# axial-shear continuity at the annulus-formation interface).
# Final row of the 10x10 layered determinant. Z-derivative-bearing;
# gets the FULL F.2.a.5 rescale (row * i + col-by-(-i) on C cols).
#
# Coefficients via the row-4 derivation evaluated at r=b plus the
# subtracted formation contributions:
#
#   * B contribution (annulus + formation): same Lame-derived
#     ``+/- 2 k_z mu (p X_0 + X_1/r)`` form as row 4.
#   * C contribution: same ``+/- mu (2 k_z^2 - k_S^2) X_1`` form
#     (single-Bessel-term direct ``X_1`` -- KEEP-sign pattern).
#   * D contribution: same ``-k_z mu X_1/r`` form.
#
# Post-rescale row 10:
#
#       Row 10 = [
#           0,                                              # A (no shear)
#           -2 k_z mu_m (p_m I_0(p_m b) - I_1(p_m b)/b),   # B_I
#           +2 k_z mu_m (p_m K_0(p_m b) + K_1(p_m b)/b),   # B_K
#           +mu_m (2 k_z^2 - k_Sm^2) I_1(s_m b),            # C_I
#           +mu_m (2 k_z^2 - k_Sm^2) K_1(s_m b),            # C_K
#           -2 k_z mu (p K_0(p b) + K_1(p b)/b),            # B (subtracted)
#           -mu (2 k_z^2 - k_S^2) K_1(s b),                 # C (subtracted)
#           -k_z mu_m I_1(s_m b) / b,                       # D_I
#           -k_z mu_m K_1(s_m b) / b,                       # D_K
#           +k_z mu K_1(s b) / b,                           # D (subtracted)
#       ]
#
# K-flavour cancellation pairs at layer=formation: all three
# (B_K + B, C_K + C, D_K + D) cancel pair-wise.
#
# Cross-row identity: row 10's annulus K-flavour entries (B_K,
# C_K, D_K) match row 4's M42, M43, M44-equivalent forms
# evaluated at r=b instead of r=a (same underlying sigma_rz
# formula at both interfaces).


def _layered_n1_row10_at_b(
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
    Row 10 of the n=1 layered modal determinant evaluated at the
    annulus-formation interface ``r = b = a + layer.thickness``.

    Encodes the axial-shear continuity BC
    ``sigma_rz^{(m)}(b) - sigma_rz^{(s)}(b) = 0`` in the cos
    sector. Final row of the 10x10 layered determinant; closes
    substep F.2.b.

    Parameters
    ----------
    kz : float
        Trial axial wavenumber (rad / m).
    omega : float
        Angular frequency (rad / s).
    vp, vs, rho : float
        Formation half-space P / S velocities and density. All
        used (mu = rho * vs^2 sets formation B/C/D coefficients).
    vf, rho_f : float
        Fluid velocity / density. Not used (fluid r<a, no shear);
        carried for signature uniformity.
    a : float
        Fluid-annulus interface radius (m); ``b = a + layer.thickness``.
    layer : BoreholeLayer
        The annular layer; sets ``mu_m, k_Sm, p_m, s_m``.

    Returns
    -------
    ndarray, shape (10,) complex
        Coefficients of (A, B_I, B_K, C_I, C_K, B, C, D_I, D_K, D)
        in row 10. Real-valued in the bound regime.

    See Also
    --------
    _layered_n1_row4_at_a : Same physical BC at r=a. Row 10's
        annulus K-flavour entries match row 4's M42, M43, M44-
        equivalent forms evaluated at r=b.
    """
    del rho_f  # not used by row 10; kept for signature uniformity
    F_f, p_m, s_m, p, s = _layered_n0_radial_wavenumbers(
        kz,
        omega,
        vp=vp,
        vs=vs,
        vf=vf,
        layer=layer,
    )
    del F_f  # row 10 doesn't touch the fluid column
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

    row: np.ndarray = np.zeros(10, dtype=complex)
    # A column: fluid carries no shear and doesn't reach r=b.
    row[0] = 0.0
    # B_I column (post-rescale; sign-flipped bracket from F.1.a.2).
    row[1] = -2.0 * kz * mu_m * (p_m * I0_pm_b - I1_pm_b / b)
    # B_K column (annulus, mirrors row 4 at r=b).
    row[2] = +2.0 * kz * mu_m * (p_m * K0_pm_b + K1_pm_b / b)
    # SV columns (Hansen form) -- roadmap A.8: u = curl curl(chi z)
    # has radial, azimuthal AND axial components; the azimuthal-only
    # vector potential the old columns encoded is not a solution of
    # the elastodynamic equations for n >= 1.
    # C_I column (post-rescale; row * i AND col-by-(-i), net factor 1).
    row[3] = +mu_m * two_kz2_minus_kSm2 * (I1_sm_b / b - s_m * I0_sm_b)
    # C_K column (annulus).
    row[4] = +mu_m * two_kz2_minus_kSm2 * (s_m * K0_sm_b + K1_sm_b / b)
    # B column (formation, subtracted; cancels B_K at layer=formation).
    row[5] = -2.0 * kz * mu * (p * K0_p_b + K1_p_b / b)
    # C column (formation, subtracted; post-rescale).
    row[6] = -mu * two_kz2_minus_kS2 * (s * K0_s_b + K1_s_b / b)
    # D_I column (annulus SH; post-rescale).
    row[7] = -kz * mu_m * I1_sm_b / b
    # D_K column (annulus).
    row[8] = -kz * mu_m * K1_sm_b / b
    # D column (formation, subtracted; sign-flipped vs D_K at layer=formation).
    row[9] = +kz * mu * K1_s_b / b
    return row
