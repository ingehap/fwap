"""Leaky-mode extension: complex-``k_z`` marcher and pseudo-Rayleigh.

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
    _k_or_hankel,
    _radial_wavenumber,
)
from fwap.cylindrical_solver._dataclasses import BoreholeMode, BranchSegment

# =====================================================================
# Leaky-mode extension (Roadmap A continuation, phases L1 + L2)
# =====================================================================
#
# The bound-mode solvers above (Stoneley n=0 + flexural n=1) require
# real ``k_z > omega / V_alpha`` for every wave speed ``V_alpha``, so
# all radial wavenumbers ``F, p, s`` are real and positive and the
# K-Bessel functions decay outward. That covers the Stoneley mode
# universally and the flexural mode in slow formations.
#
# Three borehole modes need a *complex* ``k_z`` and outgoing
# (Hankel-function) boundary conditions:
#
#   * **Pseudo-Rayleigh (n=0 leaky)**: fast-formation guided mode at
#     slowness between ``1/V_P`` and ``1/V_S``. ``s^2 = k_z^2 -
#     k_S^2 < 0`` so the formation S wave radiates outward; ``F`` and
#     ``p`` stay bound. Has a low-frequency cutoff at ``f =
#     V_S / (2 pi a)`` (geometric).
#   * **Fast-formation flexural (n=1 leaky)**: dipole flexural in
#     formations with ``V_S > V_f``. Phase velocity sits between
#     ``V_R`` and ``V_S``, both above ``V_f``, so the fluid radial
#     wavenumber ``F^2 < 0`` and the wave radiates into the borehole
#     fluid. The ``flexural_dispersion`` function above returns NaN
#     for these depths.
#   * **Quadrupole (n=2)**: the m=2 azimuthal mode used by LWD tools
#     to bypass steel-collar contamination (Tang & Cheng 2004 sect.
#     2.5). Bound in slow formations, leaky in fast formations.
#
# Phases L1 + L2 below build the mathematical scaffolding (sign
# conventions, Hankel-function ansatz, branch-cut handling) and
# generalise the n=0 modal determinant to accept complex ``k_z`` and
# return a complex value. Phase L3 (the complex-``k_z`` root finder)
# and phases L4-L6 (the three public-API leaky-mode functions) are
# planned follow-ups; see ``plans/roadmap.md`` section A for the full
# sequencing.

# ---------------------------------------------------------------------
# L1.1 -- Sign conventions for complex ``k_z`` and complex radial
# wavenumbers.
# ---------------------------------------------------------------------
#
# The bound-mode conventions (top-of-module docstring) carry over
# verbatim:
#
#   * Time dependence ``e^{-i omega t}``.
#   * Axial dependence ``e^{i k_z z}``.
#
# What's new at the leaky regime:
#
#   * ``k_z`` is in general complex: ``k_z = k_z' + i k_z''`` with
#     ``k_z' > 0`` (forward-propagating) and ``k_z'' >= 0`` (energy
#     decays in the +z direction). For perfectly bound modes,
#     ``k_z'' = 0``.
#
#   * The radial wavenumbers
#
#         F^2 = k_z^2 - omega^2 / V_f^2
#         p^2 = k_z^2 - omega^2 / V_P^2
#         s^2 = k_z^2 - omega^2 / V_S^2
#
#     are complex too. For each of the three body waves
#     (alpha = f, P, S):
#
#       - **Bound**: ``Re(alpha^2) > 0`` and ``Im(alpha^2)`` small.
#         The wave decays in radius via ``K_n(alpha r)``.
#       - **Leaky**: ``Re(alpha^2) < 0``. The wave propagates
#         outward as a radiating cylindrical wave, expressed via
#         ``H_n^{(2)}(i alpha r)``.
#
#   * The square root of a complex ``alpha^2`` follows the principal
#     branch convention with one sign flip on the leaky side: pick
#     the root with ``Im(alpha) > 0`` so that
#     ``H_n^{(2)}(i alpha r)`` decays as ``Im(alpha r) > 0`` -- the
#     standard "outgoing-wave at infinity" condition for an
#     ``e^{-i omega t}`` time convention. (For ``e^{+i omega t}``
#     the convention is ``H_n^{(1)}`` instead; we use ``H_n^{(2)}``
#     to match the existing time convention in the bound-mode
#     module docstring.)
#
# Per-mode regime table:
#
#     Mode                    F-branch    p-branch    s-branch
#     ---------------------------------------------------------
#     Stoneley (n=0)          bound       bound       bound
#     Pseudo-Rayleigh (n=0)   bound       bound       leaky
#     Flexural slow (n=1)     bound       bound       bound
#     Flexural fast (n=1)     leaky       bound       bound
#     Quadrupole slow (n=2)   bound       bound       bound
#     Quadrupole fast (n=2)   leaky       bound       bound
#
# Note that ``p`` (formation P-wave radial wavenumber) stays bound
# for every mode of practical interest; the ``F`` (fluid) and ``s``
# (S-wave) branches are the ones that flip between bound and leaky.

# ---------------------------------------------------------------------
# L1.2 -- Hankel-function ansatz for the radiating components.
# ---------------------------------------------------------------------
#
# In the leaky regime, the regular-at-infinity ``K_n(alpha r)``
# Bessel function is replaced by the outgoing Hankel function
# ``H_n^{(2)}(i alpha r)``. The two are related by the analytic
# continuation
#
#     K_n(z) = (pi / 2) * i^{n+1} * H_n^{(2)}(i z),
#
# i.e. they differ only by a constant ``i^{n+1}`` phase factor at
# fixed ``n``. For the modal-determinant calculation this phase is
# absorbed into the unknown amplitude (one of A, B, C, D), so the
# matrix structure is the same in both regimes -- only the Bessel
# evaluation routine changes per branch.
#
# Per-field ansatz for the four scalar potentials (n=0 case shown;
# n=1 and n=2 extend with cos/sin azimuthal factors per substep
# 1.1):
#
#     Fluid pressure:        P    = A * I_1(F r) cos(n theta)
#     Solid P potential:     phi  = B * J_n^{p}(p r) cos(n theta)
#     Solid SV potential:    psi  = C * J_n^{s}(s r) sin/cos(...)
#     Solid SH potential:    psi  = D * J_n^{s}(s r) sin/cos(...)
#
# where the ``J_n^{alpha}`` symbol is shorthand for "K_n if alpha is
# bound, H_n^{(2)} of (i alpha r) (with the constant phase factor
# from L1.1) if alpha is leaky". The fluid pressure always uses
# ``I_1`` (regular at the borehole axis r=0, regardless of whether
# F is bound or leaky); the F-branch leaky behaviour shows up only
# in how F enters the BC equations (complex F is fine, no Hankel
# substitution needed because the I-Bessel is what's used).
#
# scipy support: ``scipy.special.iv``, ``kv``, and ``hankel2`` all
# accept complex arguments. The bound-mode solver above already uses
# ``iv`` and ``kv`` with real inputs; switching to complex inputs is
# transparent.

# ---------------------------------------------------------------------
# L1.3 -- Branch cuts and outgoing-wave selection.
# ---------------------------------------------------------------------
#
# For each radial wavenumber ``alpha = sqrt(k_z^2 - omega^2 / V^2)``,
# the principal-branch ``numpy.sqrt`` returns the value with
# ``Re(alpha) >= 0``. That is the *decay* condition, and it is the
# right one in the bound regime (``alpha`` real and positive).
#
# The outgoing-wave condition with ``e^{-i omega t}`` time dependence
# is a different one: ``Im(alpha) > 0``. This block used to claim the
# principal branch satisfies it too, and it does not in general. The
# principal root carries
#
#     sign(Im(alpha)) = sign(Im(alpha^2)) = sign(2 Re(k_z) Im(k_z)),
#
# so it is outgoing exactly while ``Im(k_z) >= 0`` and *incoming*
# below the real ``k_z`` axis -- which is half of the plane a complex
# root search seeded on that axis moves through. ``_radial_wavenumber``
# applies the correct rule per branch; see its docstring, and roadmap
# A.10 for the measurements. Document the convention explicitly
# because the other common textbook choice (``Re(alpha) < 0``) flips
# the sign and uses ``H_n^{(1)}``.
#
# Detection rule for the regime classifier (L2 below):
#
#   * Bound:  ``Re(alpha^2) > tolerance``  --> use ``K_n(alpha r)``.
#   * Leaky:  ``Re(alpha^2) < -tolerance`` --> use
#                                  ``H_n^{(2)}(i alpha r)``.
#   * Marginal:  ``|Re(alpha^2)| < tolerance`` --> the mode is at
#                                  its cutoff frequency; the
#                                  numerical solution is
#                                  ill-conditioned. Caller's job
#                                  to skip / interpolate.
#
# The marginal-region tolerance can be tightened in L3 once the
# complex root finder is in place.

# ---------------------------------------------------------------------
# L2 -- Complex-aware n=0 modal determinant.
# ---------------------------------------------------------------------


def _detect_leaky_branches(
    kz: complex,
    omega: float,
    vp: float,
    vs: float,
    vf: float,
    tolerance: float = 1.0e-9,
) -> tuple[bool, bool, bool]:
    """
    Classify the (F, p, s) branches at a given (kz, omega) as
    bound or leaky.

    Returns a tuple ``(leaky_F, leaky_p, leaky_s)`` of booleans.
    ``True`` means the corresponding wave is leaky (radiates
    outward); ``False`` means bound (decays outward).

    Classification uses the sign of ``Re(alpha^2)`` for each wave
    speed; values within ``tolerance`` of zero are treated as
    bound by convention (the numerical solution is ill-
    conditioned at the cutoff, but the K-Bessel evaluation is
    well-defined there while the H-Bessel limit is not).
    """
    kz_c = complex(kz)
    F2 = kz_c * kz_c - (omega / vf) ** 2
    p2 = kz_c * kz_c - (omega / vp) ** 2
    s2 = kz_c * kz_c - (omega / vs) ** 2
    leaky_F = float(F2.real) < -tolerance
    leaky_p = float(p2.real) < -tolerance
    leaky_s = float(s2.real) < -tolerance
    return leaky_F, leaky_p, leaky_s


def _modal_determinant_n0_complex(
    kz: complex,
    omega: float,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    *,
    leaky_p: bool = False,
    leaky_s: bool = False,
) -> complex:
    """
    Complex-``k_z`` n=0 modal determinant with optional leaky-wave
    branches.

    Mirrors the matrix structure of the real-valued
    :func:`_modal_determinant_n0` (see its docstring for the full
    Kirchhoff derivation): three boundary conditions at ``r = a``
    (continuity of u_r, sigma_rr balance, sigma_rz = 0), three
    unknown amplitudes (A in the fluid, B and C in the solid),
    and the same row/column phase rescaling that makes the matrix
    real in the fully-bound regime.

    What's new:

    * Inputs ``kz`` is complex. The radial wavenumbers F, p, s are
      complex too.
    * ``leaky_p`` and ``leaky_s`` flags select the K-Bessel (bound)
      vs Hankel (leaky) evaluator for the formation P and S waves.
      The fluid I-Bessel always uses ``iv`` (regular at the
      borehole axis); ``F`` complex is handled transparently.
    * Returns a complex scalar. In the fully-bound regime
      (real ``kz``, both ``leaky_*`` flags False) the imaginary
      part is zero to floating-point precision and the real part
      matches the real-only :func:`_modal_determinant_n0` exactly
      -- a regression invariant tested in
      ``tests/test_cylindrical_solver.py``.

    Parameters
    ----------
    kz : complex
        Axial wavenumber. May be complex.
    omega, vp, vs, rho, vf, rho_f, a : float
        Same as :func:`_modal_determinant_n0`.
    leaky_p, leaky_s : bool, default False
        Select the leaky branch (Hankel evaluator) for the
        formation P and S waves. Use :func:`_detect_leaky_branches`
        to set these from ``(kz, omega)`` for typical regime-
        detection workflows.

    Returns
    -------
    complex
        ``det M(kz, omega)`` evaluated with the chosen branches.

    See Also
    --------
    _modal_determinant_n0 : The real-valued bound-only counterpart.
        The two functions agree exactly when ``kz`` is real and
        both ``leaky_*`` flags are False.
    _detect_leaky_branches : Helper to classify ``(F, p, s)`` as
        bound or leaky from ``(kz, omega)``.
    """
    kz_c = complex(kz)
    # The fluid always uses I-Bessel, so the sign of F is immaterial to
    # the physics -- but not to continuity: the oscillatory branch flips
    # across the real k_z axis and takes the determinant's sign with it.
    F_sq = kz_c * kz_c - (omega / vf) ** 2
    F = _radial_wavenumber(F_sq, leaky=bool(F_sq.real < 0.0))
    p = _radial_wavenumber(kz_c * kz_c - (omega / vp) ** 2, leaky=leaky_p)
    s = _radial_wavenumber(kz_c * kz_c - (omega / vs) ** 2, leaky=leaky_s)
    Fa = F * a

    # Fluid: I-Bessel always (regular at r=0). scipy.special.iv
    # supports complex arguments transparently.
    I0Fa = complex(special.iv(0, Fa))
    I1Fa = complex(special.iv(1, Fa))

    # Formation P (K or Hankel via analytic continuation).
    K0pa, K1pa = _k_or_hankel(0, p, a, leaky=leaky_p)

    # Formation S (K or Hankel).
    K0sa, K1sa = _k_or_hankel(0, s, a, leaky=leaky_s)

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz_c * kz_c - kS2

    # Same matrix layout as _modal_determinant_n0; entries are now
    # complex but the structure is identical.

    # Row 1 (continuity of u_r at r = a):
    M11 = F * I1Fa / (rho_f * omega**2)
    M12 = p * K1pa
    M13 = kz_c * K1sa

    # Row 2 (sigma_rr^{(s)} = -P^{(f)}):
    M21 = -I0Fa
    M22 = -mu * (two_kz2_minus_kS2 * K0pa + 2.0 * p * K1pa / a)
    M23 = -2.0 * kz_c * mu * (s * K0sa + K1sa / a)

    # Row 3 (sigma_rz^{(s)} = 0; rescaled by i so that entries are
    # real in the fully-bound regime):
    M31 = 0.0 + 0j
    M32 = 2.0 * kz_c * p * mu * K1pa
    M33 = mu * two_kz2_minus_kS2 * K1sa

    M = np.array([[M11, M12, M13], [M21, M22, M23], [M31, M32, M33]], dtype=complex)
    return complex(np.linalg.det(M))


# ---------------------------------------------------------------------
# L3 -- Complex-``k_z`` root finder + frequency-marching tracker.
# ---------------------------------------------------------------------
#
# The bound-mode solvers above use ``scipy.optimize.brentq`` on a
# real-valued determinant: at each frequency, bracket the root
# along the real ``k_z`` axis and bisect. That doesn't extend to
# complex ``k_z`` because there's no 1D bracketing in 2D.
#
# For the leaky regime, ``det M(k_z, omega)`` is a complex-valued
# function of complex ``k_z``. A root is a point where both
# ``Re(det)`` and ``Im(det)`` vanish simultaneously -- a 2D root-
# finding problem. We solve it with ``scipy.optimize.root(method=
# 'hybr')`` on the (Re, Im) split and chain successive frequencies
# via a continuation marcher that uses each frequency's root as
# the initial guess for the next.
#
# Algorithm summary:
#
#   * Single-frequency: :func:`_track_complex_root` wraps
#     ``scipy.optimize.root`` and returns the converged complex
#     ``k_z`` (or None on convergence failure).
#
#   * Frequency-marching: :func:`_march_complex_dispersion` walks
#     a frequency grid, seeding the next step's initial guess
#     from the previous step's converged root. Returns an array
#     of complex ``k_z`` values, NaN where convergence failed.
#
# This module covers the ROOT-FINDING mechanics only. The
# leaky-mode public APIs (pseudo-Rayleigh, fast-formation
# flexural, quadrupole) build on top of these helpers in phases
# L4-L6.


def _track_complex_root(
    det_fn,
    kz_start: complex,
    *,
    xtol: float = 1.0e-12,
) -> complex | None:
    r"""
    Find a complex root of ``det_fn`` near ``kz_start``.

    Splits the complex determinant ``det_fn(kz)`` into its real
    and imaginary parts and feeds them to
    :func:`scipy.optimize.root` (Powell's hybrid method, ``hybr``)
    as a 2-equation, 2-unknown nonlinear system.

    Parameters
    ----------
    det_fn : callable
        Function ``det_fn(kz: complex) -> complex``.
    kz_start : complex
        Initial guess for the root.
    xtol : float, default 1e-12
        Parameter-space convergence tolerance passed to
        :func:`scipy.optimize.root`.

    Returns
    -------
    complex or None
        Converged complex ``k_z`` if successful; ``None`` if the
        root finder failed (e.g. no root within the convergence
        radius, det_fn raised on an iterate, etc.).

    Notes
    -----
    The hybrid method works well for analytic complex det
    functions when the initial guess is within the local-quadratic
    convergence radius of the root. For dispersion-curve work the
    typical use is via :func:`_march_complex_dispersion`, which
    seeds each step from the previous step's root -- the local-
    quadratic radius is then never the limiting factor.

    The function is private because it's designed for the
    leaky-mode public APIs in phases L4-L6, not as a general-
    purpose user tool. Callers wanting a general complex-root
    finder should use :func:`scipy.optimize.root` directly.
    """

    def _residual(x):
        kz = complex(x[0], x[1])
        try:
            d = det_fn(kz)
        except (ValueError, OverflowError, ZeroDivisionError):
            # Return a large penalty residual so the solver
            # backs off; raising would abort the iteration.
            return [1.0e300, 1.0e300]
        return [d.real, d.imag]

    try:
        result = optimize.root(
            _residual,
            x0=[float(kz_start.real), float(kz_start.imag)],
            method="hybr",
            options={"xtol": xtol},
        )
    except (ValueError, RuntimeError):
        return None

    if not result.success:
        return None
    return complex(result.x[0], result.x[1])


def _march_complex_dispersion(
    det_fn,
    freq_grid: np.ndarray,
    kz_start: complex,
    *,
    xtol: float = 1.0e-12,
) -> np.ndarray:
    r"""
    Walk a complex root through a frequency grid via continuation.

    For each frequency ``f`` in ``freq_grid`` (in ascending or
    descending order, the marcher just consumes the grid as
    given), call :func:`_track_complex_root` seeded by the
    previous frequency's converged ``k_z``. The first step uses
    ``kz_start`` as the seed.

    Parameters
    ----------
    det_fn : callable
        Function ``det_fn(kz: complex, omega: float) -> complex``.
        The marcher binds ``omega`` per step and passes a
        single-argument closure to :func:`_track_complex_root`.
    freq_grid : ndarray, shape (n_f,)
        Frequency grid in Hz. Order matters: the marcher walks
        the grid sequentially, so a descending grid (high to low
        frequency) is appropriate for modes that are easier to
        bracket near a high-frequency asymptote (e.g., pseudo-
        Rayleigh near ``1/V_S``).
    kz_start : complex
        Initial guess for the root at the FIRST frequency in
        ``freq_grid``.
    xtol : float, default 1e-12
        Per-step convergence tolerance.

    Returns
    -------
    ndarray, shape (n_f,) complex
        Complex ``k_z`` at each frequency. NaN+NaNj where the
        per-step root finder failed; once a step fails the
        remaining steps stay NaN (the marcher cannot recover
        without a fresh seed).

    Notes
    -----
    The continuation strategy is what makes 2D root-finding
    tractable for dispersion problems: the per-step problem only
    needs to handle a *small* perturbation in ``k_z``, so
    ``scipy.optimize.root`` always converges quickly when the
    underlying physical mode is continuous. Cutoff frequencies
    where the mode disappears appear naturally as convergence
    failures, leaving NaN values that signal "mode not present
    here" to downstream callers.

    Branch tracking across leaky-vs-bound transitions is the
    caller's responsibility: ``det_fn`` should internally re-
    classify the regime via :func:`_detect_leaky_branches`
    each time it's called, OR the caller should split the
    frequency grid at the cutoff and call
    :func:`_march_complex_dispersion` separately on each side.
    """
    f_arr = np.asarray(freq_grid, dtype=float)
    n = f_arr.size
    kz_curve = np.full(n, np.nan + 1j * np.nan, dtype=complex)
    kz_prev = complex(kz_start)
    f_prev: float | None = None
    for i in range(n):
        f = float(f_arr[i])
        omega = 2.0 * np.pi * f
        # Scale-invariant continuation in SLOWNESS: dispersion
        # slowness varies slowly across frequency, while ``k_z``
        # scales linearly with frequency. Seed the next step
        # with ``k_z_prev * (f / f_prev)`` so the seed is on the
        # constant-slowness extrapolation of the previous step --
        # close to the actual root for any smooth dispersion law.
        if f_prev is None:
            kz_seed = kz_prev
        else:
            kz_seed = kz_prev * (f / f_prev)
        det_at_omega = lambda kz, _omega=omega: (  # noqa: E731
            det_fn(kz, _omega)
        )
        kz_root = _track_complex_root(det_at_omega, kz_seed, xtol=xtol)
        if kz_root is None:
            # Mode disappeared at this frequency. Leave the rest
            # of the curve as NaN; the marcher cannot continue
            # without a fresh seed.
            break
        kz_curve[i] = kz_root
        kz_prev = kz_root
        f_prev = f
    return kz_curve


# ---------------------------------------------------------------------
# Cutoff handling + branch tracker (plan item C in
# docs/plans/cylindrical_biot.md). The naive marcher above stops at
# the first convergence failure; the validated marcher below
# distinguishes "the converged root is in a different physical
# regime" (regime exit) from "the root finder failed altogether"
# (convergence failure), and tolerates a small budget of consecutive
# bad steps before giving up. Together with :class:`BranchSegment`
# and :func:`segments_from_kz_curve` this lets a public dispersion
# API recover from one-off branch hops and report the contiguous
# stretches where the mode was physically present.
# ---------------------------------------------------------------------


def _classify_marcher_step(
    kz_root: complex | None,
    omega: float,
    validator,
) -> str:
    """
    Classify a single marcher step as ``"ok"``, ``"regime_exit"``,
    or ``"convergence_failure"``.

    Parameters
    ----------
    kz_root : complex or None
        Output of :func:`_track_complex_root`; ``None`` means
        the underlying root finder did not converge.
    omega : float
        Angular frequency of the step (passed through to
        ``validator``).
    validator : callable or None
        ``(kz: complex, omega: float) -> bool``. ``True`` means the
        converged ``kz`` lies in the regime the caller wants to
        track. ``None`` disables regime checking (every converged
        root is accepted).

    Returns
    -------
    str
        One of:

        * ``"ok"`` -- ``kz_root`` is a converged complex value and
          (if a validator was given) it accepted the root.
        * ``"regime_exit"`` -- ``kz_root`` converged but the
          validator rejected it. Typical causes are crossing a
          cutoff into a regime that needs different leaky flags,
          or the root finder hopping to a neighbouring mode.
        * ``"convergence_failure"`` -- ``kz_root`` is ``None``.

    Notes
    -----
    The classifier is intentionally narrower than what the original
    plan called out: a "branch flipped" verdict (re-detect leaky
    flags via :func:`_detect_leaky_branches` and retry) would
    require the marcher to rebuild ``det_fn`` mid-march, which is
    structurally heavier than the validator-callback design here.
    For modes whose flag pattern is fixed across the whole band of
    interest (Stoneley, pseudo-Rayleigh, slow-formation flexural)
    the validator-callback version covers the same ground; for
    modes that flip flags at a cutoff (fast-formation flexural,
    plan item B) the marcher can be re-driven from the cutoff with
    fresh flags and a fresh seed -- a public-API responsibility,
    not a marcher one.
    """
    if kz_root is None:
        return "convergence_failure"
    if validator is None:
        return "ok"
    try:
        ok = bool(validator(kz_root, omega))
    except (ValueError, ArithmeticError):
        return "regime_exit"
    return "ok" if ok else "regime_exit"


def segments_from_kz_curve(
    freq_grid: np.ndarray,
    kz_curve: np.ndarray,
) -> list[BranchSegment]:
    """
    Split a marcher output into contiguous :class:`BranchSegment`s.

    A "segment" is a maximal run of samples for which both
    ``Re(kz_curve[i])`` and ``Im(kz_curve[i])`` are finite. Pure-
    NaN samples (the marcher's ``"this step was rejected"``
    sentinel) split segments.

    Parameters
    ----------
    freq_grid : ndarray, shape (n_f,)
        Frequencies in the original input order.
    kz_curve : ndarray, shape (n_f,) complex
        Complex axial wavenumbers, one per frequency. NaN+NaNj at
        rejected / failed steps.

    Returns
    -------
    list of BranchSegment
        Empty list if no finite samples exist. Otherwise one entry
        per maximal run of finite samples, preserving input order.

    Examples
    --------
    >>> import numpy as np
    >>> f = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> nan = np.nan + 1j * np.nan
    >>> kz = np.array([1.0+0j, 2.0+0j, nan, 4.0+0j, 5.0+0j])
    >>> segs = segments_from_kz_curve(f, kz)
    >>> len(segs)
    2
    >>> segs[0].start_idx, segs[0].end_idx
    (0, 1)
    >>> segs[1].start_idx, segs[1].end_idx
    (3, 4)
    """
    f_arr = np.asarray(freq_grid)
    kz_arr = np.asarray(kz_curve, dtype=complex)
    if f_arr.size != kz_arr.size:
        raise ValueError(
            f"freq_grid and kz_curve must have the same length; "
            f"got {f_arr.size} and {kz_arr.size}"
        )
    finite = np.isfinite(kz_arr.real) & np.isfinite(kz_arr.imag)
    segments: list[BranchSegment] = []
    n = f_arr.size
    i = 0
    while i < n:
        if not finite[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and finite[j + 1]:
            j += 1
        segments.append(
            BranchSegment(
                start_idx=int(i),
                end_idx=int(j),
                freq=f_arr[i : j + 1].copy(),
                kz=kz_arr[i : j + 1].copy(),
            )
        )
        i = j + 1
    return segments


def _march_complex_dispersion_validated(
    det_fn,
    freq_grid: np.ndarray,
    kz_start: complex,
    *,
    validator=None,
    max_consecutive_invalid: int = 3,
    xtol: float = 1.0e-12,
) -> np.ndarray:
    r"""
    :func:`_march_complex_dispersion` plus a per-step validator and a
    consecutive-invalid budget for tolerating one-off bad steps.

    Each step's converged ``kz`` is classified by
    :func:`_classify_marcher_step`. ``"ok"`` steps are recorded and
    used as the seed for the next step's continuation; ``"regime_exit"``
    and ``"convergence_failure"`` steps are recorded as NaN+NaNj
    and counted against ``max_consecutive_invalid``. As long as the
    invalid count stays below the budget, marching continues with
    the seed pinned to the last good step. Once the budget is
    exhausted, the marcher stops and the rest of the curve stays
    NaN.

    Parameters
    ----------
    det_fn : callable
        ``det_fn(kz: complex, omega: float) -> complex``.
    freq_grid : ndarray, shape (n_f,)
        Frequency grid (Hz) walked in the order given.
    kz_start : complex
        Initial seed for the first frequency.
    validator : callable or None, optional
        ``(kz: complex, omega: float) -> bool``. Returns ``True``
        when the converged root sits in the regime the caller
        wants to track. ``None`` (default) accepts every converged
        root.
    max_consecutive_invalid : int, default 3
        Number of consecutive non-``"ok"`` steps the marcher will
        skip past before stopping. Setting this to ``0`` recovers
        the strict-stop semantics of
        :func:`_march_complex_dispersion`.
    xtol : float, default 1e-12
        Per-step ``scipy.optimize.root`` parameter-space tolerance.

    Returns
    -------
    ndarray, shape (n_f,) complex
        Complex ``k_z`` at each frequency, NaN+NaNj at every
        rejected step (validator failure, root-finder failure, or
        post-budget tail).
    """
    f_arr = np.asarray(freq_grid, dtype=float)
    n = f_arr.size
    kz_curve = np.full(n, np.nan + 1j * np.nan, dtype=complex)
    if n == 0:
        return kz_curve
    kz_prev = complex(kz_start)
    omega_prev: float | None = None
    consecutive_invalid = 0
    for i in range(n):
        omega = 2.0 * np.pi * float(f_arr[i])
        if omega_prev is None:
            kz_seed = kz_prev
        else:
            kz_seed = kz_prev * (omega / omega_prev)
        det_at_omega = lambda kz, _omega=omega: (  # noqa: E731
            det_fn(kz, _omega)
        )
        kz_root = _track_complex_root(det_at_omega, kz_seed, xtol=xtol)
        verdict = _classify_marcher_step(kz_root, omega, validator)
        if verdict == "ok":
            kz_curve[i] = kz_root
            kz_prev = kz_root  # type: ignore[assignment]
            omega_prev = omega
            consecutive_invalid = 0
            continue
        # Rejected step: leave NaN, do not update kz_prev / omega_prev
        # so the next step still extrapolates from the last good one.
        consecutive_invalid += 1
        logger.debug(
            "_march_complex_dispersion_validated: step %d/%d at f=%.1f "
            "Hz rejected (%s, consecutive=%d/%d)",
            i,
            n,
            omega / (2.0 * np.pi),
            verdict,
            consecutive_invalid,
            max_consecutive_invalid,
        )
        if consecutive_invalid > max_consecutive_invalid:
            break
    return kz_curve


# ---------------------------------------------------------------------
# A.9 -- slow-formation cased-hole leaky branch (n >= 1)
# ---------------------------------------------------------------------
#
# Roadmap A.9. Behind a stiff annulus -- steel casing, or any layer
# much faster in shear than the rock -- the n >= 1 borehole modes of a
# SLOW formation are faster than that formation's shear speed. The
# formation S branch is then radiating rather than evanescent, the mode
# leaks energy into the rock, and the real-valued bound-regime
# determinant has no root for it at all: it is not that the solver
# misses the mode, it is that the mode is not in the space that
# determinant describes.
#
# The window is bounded below by the formation shear speed (below it
# the mode is bound and the ordinary layered path owns it) and above by
# ``min(V_f, min layer V_S)``: past ``V_f`` the fluid column turns
# oscillatory, which is the fast-formation regime with its own marcher,
# and past a layer's shear speed that layer's radial wavenumber turns
# over and the propagator conditioning degrades.
#
# Structurally this is the pseudo-Rayleigh problem one azimuthal order
# up and with a layer stack in the way, so it reuses that machinery
# wholesale: ``_detect_leaky_branches`` for the flags,
# ``_track_complex_root`` for the refinement, and
# ``_march_complex_dispersion_validated`` for the continuation. What is
# new is the seeding. The pseudo-Rayleigh driver enumerates its roots
# from a closed-form cutoff estimate; there is no such estimate here,
# so the seed comes from a real-``k_z`` scan of ``Im(det)``, taking the
# SLOWEST crossing as the fundamental and marching up in frequency --
# the same selection rule roadmap A.2 established for the
# fast-formation branch, for the same reason.
#
# Two artefacts have to be excluded by name, both already known from
# A.2: the window edges, where a radial wavenumber vanishes, and each
# layer's own shear speed, where the layer's radial wavenumber vanishes
# and the determinant has a sign change that is not a mode.


#: Real-axis scan resolution when seeding the leaky cased branch. The
#: window is a few hundred m/s wide and carries up to about six
#: crossings at the top of the band, so this is ~2 m/s resolution.
_LEAKY_CASED_SCAN_POINTS = 192

#: Fractional margin held off both ends of the scan window.
_LEAKY_CASED_EDGE_EPS = 5.0e-4

#: Relative tolerance for calling a candidate degenerate with one of
#: the ``exclude`` velocities, and the width of the dead band held off
#: the window ceiling.
#:
#: Larger than ``_FAST_FLEXURAL_DEGENERACY_TOL`` (1e-5) on purpose, and
#: the reason is a dimensionless one. The ceiling of this window IS a
#: layer shear speed whenever the softest layer is slower than the
#: fluid, so the ``exclude`` velocity and the scan edge coincide. With a
#: tolerance tighter than the edge margin, a root pinned at
#: ``ceiling (1 - eps)`` -- the layer's vanishing radial wavenumber, not
#: a mode -- slips between the two and is accepted. Measured on an
#: annulus-stiffness sweep it captured the whole answer over
#: ``1.31 <= V_S_layer / V_S <= 1.50``, returning ``c / V_S = ceiling /
#: V_S`` to four figures. The tolerance must therefore exceed
#: ``_LEAKY_CASED_EDGE_EPS``.
_LEAKY_CASED_DEGENERACY_TOL = 2.0e-3

#: A seed is only accepted if the determinant at the refined root is
#: this much smaller than 0.2 % away from it. Guards against the
#: complex tracker settling somewhere that is not a root.
_LEAKY_CASED_SHARPNESS = 1.0e-6

#: Consecutive rejected steps the marcher walks past before giving up.
_LEAKY_CASED_MAX_INVALID = 2

#: Slack on the monotone-descent rule, as a fraction of the previous
#: step's phase velocity. The branch descends, so a candidate faster
#: than the last one by more than this is a different branch.
_LEAKY_CASED_STEP_UP = 5.0e-3


def _march_leaky_cased_branch(
    det_fn,
    freq: np.ndarray,
    *,
    vs: float,
    ceiling: float,
    exclude: tuple[float, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Follow the fundamental leaky cased-hole branch of a slow formation.

    Shared by the ``n = 1`` and ``n = 2`` layered drivers so the two
    cannot drift apart, in the same way
    :func:`~fwap.cylindrical_solver._n1_isotropic._march_fast_flexural_branch`
    is shared by their bound siblings.

    Parameters
    ----------
    det_fn : callable
        ``det_fn(kz: complex, omega: float) -> complex``. The caller is
        responsible for setting the formation's leaky flags, normally
        from :func:`_detect_leaky_branches`, so the same callable is
        valid on both sides of the shear-speed crossing.
    freq : ndarray, shape (n,)
        Frequencies in Hz, any order.
    vs : float
        Formation shear velocity (m/s); the window floor.
    ceiling : float
        Window ceiling (m/s), normally ``min(V_f, min layer V_S)``.
    exclude : tuple of float, optional
        Phase velocities at which the determinant is degenerate rather
        than modal -- each layer's shear speed. See
        :func:`~fwap.cylindrical_solver._n1_isotropic._march_fast_flexural_branch`
        for why these have to be named.

    Returns
    -------
    (ndarray, ndarray)
        Phase slowness (s/m) from ``Re(k_z)`` and attenuation (1/m)
        from ``Im(k_z)``, both aligned with ``freq`` and ``NaN`` where
        the branch was not found.

    Notes
    -----
    The returned ``k_z`` is genuinely complex: this mode radiates into
    the formation, so unlike the fast-formation bound branch it has a
    real attenuation rather than a numerically-zero one. ``Im(k_z)``
    runs about 6 % of ``Re(k_z)`` where the branch first appears and
    falls below 0.5 % at the top of the dipole band, so the leakage is
    weak but not negligible.
    """
    f_arr = np.asarray(freq, dtype=float)
    slowness = np.full(f_arr.size, np.nan, dtype=float)
    attenuation = np.full(f_arr.size, np.nan, dtype=float)
    if f_arr.size == 0 or not ceiling > vs:
        return slowness, attenuation

    lo = vs * (1.0 + _LEAKY_CASED_EDGE_EPS)
    hi = ceiling * (1.0 - _LEAKY_CASED_EDGE_EPS)

    def _degenerate(v: float) -> bool:
        return any(
            abs(v / e - 1.0) < _LEAKY_CASED_DEGENERACY_TOL for e in exclude if e > 0.0
        )

    def _valid(kz: complex, omega: float, ceiling_v: float) -> bool:
        if not (np.isfinite(kz.real) and np.isfinite(kz.imag)):
            return False
        if kz.real <= 0.0 or kz.imag < 0.0:
            # Im(k_z) < 0 is a wave growing along the borehole.
            return False
        v = omega / kz.real
        # The dead band at the top keeps the window ceiling -- which is
        # itself a branch point -- out of the accepted set.
        if not (lo * 0.98 < v < hi * (1.0 - _LEAKY_CASED_DEGENERACY_TOL)):
            return False
        if v > ceiling_v:
            return False
        if _degenerate(v):
            return False
        try:
            at = abs(det_fn(kz, omega))
            off = abs(det_fn(kz * 1.002, omega))
        except (ValueError, ArithmeticError, OverflowError):
            return False
        if not (np.isfinite(at) and np.isfinite(off)) or off == 0.0:
            return False
        return at < _LEAKY_CASED_SHARPNESS * off

    def _refine(kz_guess: complex, omega: float, ceiling_v: float) -> complex | None:
        root = _track_complex_root(lambda k: det_fn(k, omega), kz_guess)
        if root is None or not _valid(root, omega, ceiling_v):
            return None
        return root

    def _scan(omega: float, ceiling_v: float) -> complex | None:
        """Refine the slowest real-axis ``Im(det)`` crossing in the window.

        Walks from the slow end (largest ``k_z``) so the first crossing
        accepted is the fundamental, which is the same selection rule
        the fast-formation marcher uses and for the same reason.
        """
        grid = np.linspace(omega / hi, omega / lo, _LEAKY_CASED_SCAN_POINTS)
        try:
            vals = [det_fn(complex(k, 0.0), omega).imag for k in grid]
        except (ValueError, ArithmeticError, OverflowError):
            return None
        for j in range(grid.size - 1, 0, -1):
            a_val, b_val = vals[j], vals[j - 1]
            if not (np.isfinite(a_val) and np.isfinite(b_val)):
                continue
            if a_val == 0.0 or np.sign(a_val) == np.sign(b_val):
                continue
            mid = complex(0.5 * (grid[j] + grid[j - 1]), 0.0)
            root = _refine(mid, omega, ceiling_v)
            if root is not None:
                return root
        return None

    kz_prev: complex | None = None
    omega_prev: float | None = None
    misses = 0
    for i in np.argsort(f_arr):
        omega = 2.0 * np.pi * float(f_arr[i])
        ceiling_v = (
            np.inf
            if (kz_prev is None or omega_prev is None)
            else (omega_prev / kz_prev.real) * (1.0 + _LEAKY_CASED_STEP_UP)
        )
        root: complex | None = None
        if kz_prev is not None and omega_prev is not None:
            root = _refine(kz_prev * (omega / omega_prev), omega, ceiling_v)
        if root is None:
            root = _scan(omega, ceiling_v)

        if root is None:
            logger.debug(
                "leaky cased marcher: no root at f=%.1f Hz in (%.1f, %.1f) m/s",
                omega / (2.0 * np.pi),
                lo,
                hi,
            )
            if kz_prev is not None:
                misses += 1
                if misses > _LEAKY_CASED_MAX_INVALID:
                    break
            continue

        misses = 0
        slowness[i] = root.real / omega
        attenuation[i] = root.imag
        kz_prev = root
        omega_prev = omega

    return slowness, attenuation


# ---------------------------------------------------------------------
# L4 -- Public n=0 leaky API: pseudo-Rayleigh dispersion.
# ---------------------------------------------------------------------
#
# First product on top of the L1-L3 scaffolding above. The pseudo-
# Rayleigh wave is the n=0 leaky mode of a fluid-filled borehole in a
# fast formation (V_S > V_f). Its phase velocity sits between V_S and
# V_P; the formation S wave radiates into the formation (s-branch
# leaky) while the fluid I-Bessel and the formation P K-Bessel remain
# bound. See Paillet & Cheng (1991) sect. 4.4 and fig 4.5.
#
# The mode appears above a low-frequency cutoff where it merges with
# the body S head wave (slowness = 1/V_S, k_z = omega / V_S). A
# closed-form approximation for the first-mode cutoff is
#
#     f_c ~ j_{1,1} V_f V_S / (2 pi a sqrt(V_S^2 - V_f^2))
#
# (rigid-pipe limit V_S -> infty recovers the Pochhammer-Chree first
# cutoff f_c = j_{1,1} V_f / (2 pi a)). The implementation uses this
# as a sanity bracket for the marcher's frequency grid rather than as
# a hard cutoff -- the actual cutoff comes out of the root finder
# losing convergence, which is the reliable test.


# First positive zero of the Bessel function J_1. Used in the
# rigid-pipe-limit cutoff approximation for n=0 leaky modes.
_J1_FIRST_ZERO = 3.831705970207512


def _enumerate_leaky_roots_n0(
    omega: float,
    *,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    n_re: int = 40,
    n_im: int = 8,
) -> list[complex]:
    r"""
    Find every n=0 leaky-S root at one frequency, ordered by radial order.

    Sweeps a grid of seeds across the leaky-S window and polishes each
    with :func:`_track_complex_root`, then keeps the converged roots
    that are genuinely roots -- verified by comparing ``|det|`` at the
    root against its median on a small circle around it, which rejects
    the flat regions the hybrid solver sometimes reports as converged.

    Ordering is by **descending** ``Re(k_z)``, which is ascending radial
    order: the fundamental has the smallest radial wavenumber in the
    fluid, hence the largest axial wavenumber and the slowest phase
    velocity. So index 0 is the fundamental, index 1 the first
    overtone, and so on.

    Parameters
    ----------
    omega : float
        Angular frequency (rad/s).
    vp, vs, rho, vf, rho_f, a : float
        Media and geometry, as in :func:`pseudo_rayleigh_dispersion`.
    n_re, n_im : int
        Seed-grid resolution across ``Re(k_z)`` and ``Im(k_z)``. The
        defaults are chosen with margin: the recovered root *count* is
        unchanged from ``24 x 5`` up to ``80 x 16`` across borehole
        radii 0.07-0.15 m, formations with ``V_S`` 1700-2800 m/s and
        frequencies 15-60 kHz, so the defaults sit well above the
        density where anything is missed.

    Returns
    -------
    list of complex
        Converged ``k_z`` values with ``Im(k_z) > 0`` and slowness
        strictly inside ``(1/V_P, 1/V_S)``, ordered fundamental first.
        Empty if the frequency is below the lowest branch's cutoff.
    """

    def det(kz: complex) -> complex:
        return _modal_determinant_n0_complex(
            kz, omega, vp, vs, rho, vf, rho_f, a, leaky_p=False, leaky_s=True
        )

    kz_lo = omega / vp
    kz_hi = omega / vs
    roots: list[complex] = []
    for re_kz in np.linspace(kz_lo * 1.001, kz_hi * 0.999, n_re):
        for im_kz in np.geomspace(0.02, 60.0, n_im):
            polished = _track_complex_root(det, complex(float(re_kz), float(im_kz)))
            if polished is None or polished.imag <= 0.0:
                continue
            if not kz_lo < polished.real < kz_hi:
                continue
            # A converged point is only a root if the determinant dips
            # sharply there relative to its own neighbourhood; the
            # absolute magnitude is meaningless because the determinant
            # spans many orders of magnitude across the window.
            radius = 0.01 * abs(polished)
            try:
                ring = float(
                    np.median(
                        [
                            abs(det(polished + radius * np.exp(1j * t)))
                            for t in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
                        ]
                    )
                )
            except (ValueError, OverflowError, ZeroDivisionError):
                continue
            if not np.isfinite(ring) or abs(det(polished)) > 1.0e-9 * ring:
                continue
            if any(abs(polished - seen) < 1.0e-6 * abs(polished) for seen in roots):
                continue
            roots.append(polished)

    return sorted(roots, key=lambda z: -z.real)


def _scan_bound_roots(
    det_fn, kz_lo: float, kz_hi: float, samples: int = 2000
) -> list[float]:
    """
    Real roots of ``det_fn`` in ``(kz_lo, kz_hi)``, ordered by descending k_z.

    Straight sign-change scan plus :func:`scipy.optimize.brentq`. Descending
    order means ascending radial order, matching
    :func:`_enumerate_leaky_roots_n0`.

    Parameters
    ----------
    det_fn : callable
        Real-valued determinant of one real argument.
    kz_lo, kz_hi : float
        Open interval to scan (rad/m).
    samples : int, default 2000
        Scan resolution. The trapped roots are simple and well separated,
        so this sits far above the density where any are missed; see the
        resolution-independence test in ``tests/test_cylindrical_solver.py``.

    Returns
    -------
    list of float
        Converged ``k_z`` values (rad/m), largest first. Empty if the
        window holds no root.
    """
    grid = np.linspace(kz_lo, kz_hi, samples)
    values = np.array([det_fn(float(k)) for k in grid])
    roots: list[float] = []
    for i in range(grid.size - 1):
        if not (np.isfinite(values[i]) and np.isfinite(values[i + 1])):
            continue
        if np.sign(values[i]) != np.sign(values[i + 1]):
            roots.append(
                float(
                    optimize.brentq(
                        det_fn, grid[i], grid[i + 1], xtol=1.0e-14, rtol=8.9e-16
                    )
                )
            )
    return sorted(roots, reverse=True)


def trapped_pseudo_rayleigh_dispersion(
    freq: np.ndarray,
    *,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    branch: int = 0,
) -> BoreholeMode:
    r"""
    Trapped (fully bound) pseudo-Rayleigh dispersion from the n=0 determinant.

    The pseudo-Rayleigh family splits in two by phase velocity, and this
    function covers the half that does not radiate. For
    :math:`V_f < c < V_S` the formation P and S waves are both evanescent
    while the fluid field oscillates radially, so the mode is a genuine
    trapped resonance of the borehole fluid column with a real ``k_z`` and
    no attenuation. Above :math:`V_S` the shear wave propagates, the mode
    leaks, and :func:`pseudo_rayleigh_dispersion` takes over with a complex
    ``k_z``.

    These modes exist in fast formations only (``V_S > V_f``), each above
    its own cutoff, and several coexist at a given frequency: at 30 kHz in
    a 0.10 m hole through a ``4000/2300/2500`` formation there are three,
    alongside the Stoneley mode. They are what
    :func:`stoneley_dispersion` cannot return -- its bracket starts at
    ``omega / min(V_S, V_f)`` and so covers only ``c < V_f``.

    Parameters
    ----------
    freq : ndarray
        Frequency grid (Hz), shape ``(n_freq,)``. Must be strictly
        positive. Frequencies are solved independently, so the grid need
        not be ordered or evenly spaced.
    vp, vs, rho : float
        Formation P-wave velocity (m/s), S-wave velocity (m/s) and bulk
        density (kg/m^3). Require ``vp > vs > 0`` and ``rho > 0``.
    vf, rho_f : float
        Borehole-fluid velocity (m/s) and density (kg/m^3). Require
        ``vs > vf > 0``: a slow formation has no trapped window.
    a : float
        Borehole radius (m); must be positive.
    branch : int, default 0
        Radial order, ``0`` being the fundamental. Roots are ordered by
        descending ``k_z``, i.e. ascending radial order -- the same
        convention :func:`pseudo_rayleigh_dispersion` uses, and the
        fundamental is the slowest of the trapped modes at any frequency.

    Returns
    -------
    BoreholeMode
        ``name = "trapped_pseudo_rayleigh"``, ``azimuthal_order = 0``,
        ``freq`` echoed, and ``slowness`` of shape ``(n_freq,)`` in s/m.
        ``attenuation_per_meter`` is ``None``: these modes are bound and
        lossless, which is the substantive difference from the leaky
        sister function. ``slowness`` is ``NaN`` at frequencies below the
        requested branch's cutoff, where that mode does not exist.

    Raises
    ------
    ValueError
        If any input is non-positive, ``vp <= vs``, ``vs <= vf`` (no
        trapped window), ``branch`` is negative, or ``freq`` contains a
        non-positive entry.

    Notes
    -----
    Each frequency is solved independently by scanning the trapped window
    ``omega/V_S < k_z < omega/V_f`` for sign changes of the modal
    determinant and polishing with :func:`scipy.optimize.brentq`. There is
    no frequency marching and therefore none of the branch-continuity
    machinery :func:`pseudo_rayleigh_dispersion` needs: the roots are real
    and simple here, and ordering them at each frequency is enough to keep
    a branch identified. A consequence worth relying on is that the result
    does not depend on the frequency grid at all.

    In this window the determinant is real to rounding -- the fluid radial
    wavenumber is imaginary, which turns the modified Bessel functions into
    ordinary ones -- so its imaginary part is discarded rather than being
    carried through a complex root finder.

    See Also
    --------
    pseudo_rayleigh_dispersion : The leaky half of the same family,
        ``c > V_S``, with a complex ``k_z``.
    stoneley_dispersion : The other bound n=0 mode, ``c < V_f``.

    References
    ----------
    * Paillet, F. L., & Cheng, C. H. (1991). *Acoustic Waves in
      Boreholes.* CRC Press, sect. 4.4 and fig 4.5.
    * Cheng, C. H., & Toksoz, M. N. (1981). Elastic wave propagation in a
      fluid-filled borehole and synthetic acoustic logs. *Geophysics*
      46(7), 1042-1053.
    """
    if vp <= 0 or vs <= 0 or rho <= 0:
        raise ValueError("vp, vs, rho must all be positive")
    if vf <= 0 or rho_f <= 0:
        raise ValueError("vf and rho_f must be positive")
    if a <= 0:
        raise ValueError("a must be positive")
    if vp <= vs:
        raise ValueError("require vp > vs")
    if vs <= vf:
        raise ValueError(
            f"trapped pseudo-Rayleigh modes require a fast formation "
            f"(vs > vf); got vs={vs}, vf={vf}"
        )
    if branch < 0:
        raise ValueError(f"branch must be non-negative, got {branch}")

    f_arr = np.asarray(freq, dtype=float)
    if np.any(f_arr <= 0):
        raise ValueError("freq must be strictly positive")

    slowness = np.full(f_arr.shape, np.nan, dtype=float)
    for i, f in enumerate(f_arr.ravel()):
        omega = 2.0 * np.pi * float(f)

        def det(kz: float, omega: float = omega) -> float:
            return _modal_determinant_n0_complex(
                kz, omega, vp, vs, rho, vf, rho_f, a, leaky_p=False, leaky_s=False
            ).real

        roots = _scan_bound_roots(
            det, omega / vs * (1.0 + 1.0e-9), omega / vf * (1.0 - 1.0e-9)
        )
        if branch < len(roots):
            slowness.ravel()[i] = roots[branch] / omega

    return BoreholeMode(
        name="trapped_pseudo_rayleigh",
        azimuthal_order=0,
        freq=f_arr,
        slowness=slowness,
        attenuation_per_meter=None,
    )


def pseudo_rayleigh_dispersion(
    freq: np.ndarray,
    *,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
    branch: int = 0,
) -> BoreholeMode:
    r"""
    Pseudo-Rayleigh leaky-mode dispersion from the n=0 modal
    determinant.

    Tracks the n=0 leaky root with the formation S wave radiating
    outward (``s``-branch leaky) while the fluid pressure and the
    formation P wave stay bound. The mode exists in fast formations
    only (``V_S > V_f``) above a low-frequency cutoff where its phase
    velocity merges with the body S head wave (``slowness -> 1 / V_S``).

    Parameters
    ----------
    freq : ndarray
        Frequency grid (Hz). Must be strictly positive. The marcher
        walks the grid from high to low frequency internally; the
        return arrays are indexed in input order.
    vp, vs, rho : float
        Formation P-wave velocity (m/s), S-wave velocity (m/s), and
        bulk density (kg/m^3). Must satisfy ``vp > vs > 0`` and
        ``rho > 0``.
    vf, rho_f : float
        Borehole-fluid velocity (m/s) and density (kg/m^3). Must
        satisfy ``vs > vf`` (fast formation).
    a : float
        Borehole radius (m).
    branch : int, default 0
        Radial order to track, ``0`` being the fundamental (the
        slowest of the leaky modes, and the one that survives to the
        lowest frequency). Branches are enumerated at the highest
        requested frequency and ordered by descending ``Re(k_z)``.
        Higher orders appear only above their own cutoffs, so a
        ``branch`` that does not exist at the top of the requested
        band raises rather than returning an empty curve.

    Returns
    -------
    BoreholeMode
        ``name = "pseudo_rayleigh"``, ``azimuthal_order = 0``.
        ``slowness[i] = Re(k_z(omega[i])) / omega[i]`` (s/m), and
        ``attenuation_per_meter[i] = Im(k_z(omega[i]))`` (1/m, the
        spatial decay rate of the mode in the +z direction). ``NaN``
        at frequencies below the geometric cutoff, where the root
        finder fails to converge, or where the converged root falls
        outside the leaky-S regime ``1/V_P < slowness < 1/V_S`` with
        ``Im(k_z) > 0``.

    Raises
    ------
    ValueError
        If any input is non-positive, ``vp <= vs``, ``vs <= vf``
        (slow formation -- mode does not exist), or ``freq``
        contains a non-positive entry.

    Notes
    -----
    Implementation strategy: walk the frequency grid from high to
    low frequency, seeded at the highest input frequency by the
    analytic high-frequency asymptote (slowness slightly below
    ``1/V_S`` with a small positive imaginary part, pushing the
    s-branch into the leaky regime). At each step the converged
    ``k_z`` from the previous frequency is rescaled by the
    constant-slowness extrapolation ``k_z * (omega / omega_prev)``
    and fed to :func:`scipy.optimize.root` as the seed for the next
    step. The marcher stops as soon as

    1. ``scipy.optimize.root`` fails to converge,
    2. the converged ``k_z`` has ``Im(k_z) <= 0`` (mode left the
       leaky regime, either by physical merger with the bulk S wave
       at the cutoff, or by numerical drift to a non-physical
       root), or
    3. the converged slowness ``Re(k_z) / omega`` falls outside the
       open interval ``(1/V_P, 1/V_S)`` (mode hopped to a different
       physical regime).

    All three stopping conditions leave the remaining low-frequency
    samples as NaN. The implementation does not currently attempt
    branch-stitching across the cutoff; that is plan item C
    (`docs/plans/cylindrical_biot.md`).

    Branch selection
    ----------------
    The leaky-S window holds several roots at once -- one per radial
    order of the fluid column -- and their number grows with frequency
    as successive orders pass their cutoffs. They are enumerated at the
    highest requested frequency by :func:`_enumerate_leaky_roots_n0`
    and selected by ``branch``, so the returned curve is a function of
    the medium and the requested order alone.

    **This replaces a heuristic seed that made the answer depend on the
    caller's grid.** The previous implementation guessed a point near
    ``1/V_S`` and followed whichever root the hybrid solver fell into,
    with three measured consequences: a 0.10 m hole in a
    ``V_P/V_S/rho = 4000/2300/2500`` formation reported ``c = 2486 m/s``
    at 30 kHz for a grid ending at 40 kHz and ``c = 2952 m/s`` for one
    ending at 80 kHz (both genuine roots, but a silent switch between
    them); a 0.07 m hole returned *all-NaN* on a 60-sample 4-30 kHz
    grid while recovering the band at 80 samples; and a request for
    24-32 kHz alone returned nothing where a 2-40 kHz grid converged
    throughout. All three are fixed -- ``branch=0`` now returns the
    same curve for every grid top from 32 kHz to 100 kHz, and the two
    silent failures recover their full bands.

    What remains: the march can still terminate early at a genuine
    cutoff, which is the intended behaviour, and the resulting NaNs
    mark where the mode ceases to exist rather than where the search
    gave up. An all-NaN result now means the requested branch has no
    root at the top of the band -- a physical statement, not a search
    failure.

    Accuracy of the attenuation
    ---------------------------
    ``attenuation_per_meter`` has been checked against
    :func:`fwap.leaky_radiation_attenuation`, an independent ray
    estimate built from the plane-wave fluid/solid reflection
    coefficient with no modal determinant in it. The two agree in size
    and in their radius scaling, with a stable systematic offset near
    0.6 and a superimposed transverse resonance; see that function's
    docstring for the measured numbers and what the comparison does and
    does not establish.

    The geometric cutoff frequency is approximately

    .. math::
        f_c \approx \frac{j_{1,1} V_f V_S}
                         {2 \pi a \sqrt{V_S^2 - V_f^2}}

    where ``j_{1,1} \approx 3.832`` is the first positive zero of
    :math:`J_1`, exposed as :data:`_J1_FIRST_ZERO`.

    **Do not use that estimate as a guard on the requested frequency
    band.** It is a rigid-pipe limit, and a compliant elastic wall
    admits the mode well below it: at ``V_S = 2600``, ``V_f = 1500``,
    ``a = 0.10`` the formula gives 11.2 kHz while this routine
    converges down to about 4.1 kHz, so guarding with it would throw
    away a valid band nearly 3 kHz wide. What the estimate *does*
    capture is the geometry --- the measured cutoff scales as
    :math:`1/a` exactly as the formula says, to about 1 part in 300
    over a 3.3x range of radius, which is what
    ``tests/test_cylindrical_solver.py`` pins.

    The offset is not a universal constant that could simply be
    folded in: it varies strongly with formation velocity, and for
    some parameter combinations the marcher's termination frequency
    is not a stable quantity at all (a 1 % change in ``vp`` moved it
    by 20 % in one measured case). Treat the returned ``NaN``
    boundary as this implementation's convergence limit rather than
    as a physical cutoff.

    See Also
    --------
    stoneley_dispersion : The fully-bound n=0 sister.
    flexural_dispersion : The bound n=1 sister (slow formations).
    fwap.synthetic.pseudo_rayleigh_dispersion : Phenomenological
        callable-factory model used as the synthetic-gather
        dispersion law; the present function is the modal-
        determinant counterpart.

    References
    ----------
    * Paillet, F. L., & Cheng, C. H. (1991). *Acoustic Waves in
      Boreholes.* CRC Press, sect. 4.4 and fig 4.5.
    * Schmitt, D. P. (1988). Shear wave logging in elastic
      formations. *J. Acoust. Soc. Am.* 84(6), 2215-2229.
    * Tang, X.-M., & Cheng, A. (2004). *Quantitative Borehole
      Acoustic Methods.* Elsevier, sect. 3.2.
    """
    if vp <= 0 or vs <= 0 or rho <= 0:
        raise ValueError("vp, vs, rho must all be positive")
    if vf <= 0 or rho_f <= 0:
        raise ValueError("vf and rho_f must be positive")
    if a <= 0:
        raise ValueError("a must be positive")
    if vp <= vs:
        raise ValueError("require vp > vs")
    if vs <= vf:
        raise ValueError(
            f"pseudo-Rayleigh requires a fast formation (vs > vf); got vs={vs}, vf={vf}"
        )
    f_arr = np.asarray(freq, dtype=float)
    if np.any(f_arr <= 0):
        raise ValueError("freq must be strictly positive")

    n_f = f_arr.size
    slowness = np.full(n_f, np.nan, dtype=float)
    attenuation = np.full(n_f, np.nan, dtype=float)

    if n_f == 0:
        return BoreholeMode(
            name="pseudo_rayleigh",
            azimuthal_order=0,
            freq=f_arr,
            slowness=slowness,
            attenuation_per_meter=attenuation,
        )

    # Sort frequencies descending. The marcher seeds from the
    # high-f asymptote and walks toward the cutoff.
    order_desc = np.argsort(-f_arr)
    f_desc = f_arr[order_desc]

    # High-frequency seed: enumerate every leaky root at the top of
    # the grid and take the requested radial order, rather than
    # guessing a point near 1/V_S and marching whichever root the
    # hybrid solver happens to fall into. The old heuristic seed
    # (slowness ~ 0.95 / V_S) made the answer depend on where the
    # caller's grid stopped -- a 2-40 kHz request and a 2-80 kHz
    # request returned different modes at the same frequency, both
    # of them genuine roots. Enumerating removes that coupling: the
    # branch is now selected by index, and the same index gives the
    # same curve for any grid that reaches it.
    omega_max = 2.0 * np.pi * float(f_desc[0])
    seeds = _enumerate_leaky_roots_n0(
        omega_max, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a
    )
    if not seeds:
        # No leaky root exists at the top of the requested band, so
        # there is nothing to march. An all-NaN curve is the honest
        # answer; it means "no mode at these frequencies", which is
        # a physical statement here rather than a search failure.
        return BoreholeMode(
            name="pseudo_rayleigh",
            azimuthal_order=0,
            freq=f_arr,
            slowness=slowness,
            attenuation_per_meter=attenuation,
        )
    if branch < 0:
        raise ValueError(f"branch must be non-negative, got {branch}")
    if branch >= len(seeds):
        raise ValueError(
            f"branch={branch} was requested but only {len(seeds)} leaky "
            f"branch(es) exist at {f_desc[0]:.6g} Hz, the top of the "
            "requested band. Higher radial orders appear only above "
            "their own cutoffs, so extend the frequency grid upward to "
            "reach them."
        )
    kz_seed = seeds[branch]

    # Valid leaky-S regime in slowness terms: open interval
    # (1/V_P, 1/V_S), with a small upper-side numerical slack so
    # a converged kz exactly at omega/V_S (boundary case) is still
    # accepted. The validated marcher (plan item C) uses this
    # callable per step; a step whose converged root falls outside
    # the regime is rejected as "regime_exit", left as NaN, and
    # the marcher continues from the last good step within the
    # consecutive-invalid budget.
    slowness_lo = 1.0 / vp
    slowness_hi = 1.0 / vs
    slowness_slack = 1.0e-6 * slowness_hi

    def _validator(kz: complex, omega_step: float) -> bool:
        if kz.imag <= 0.0:
            return False
        s = kz.real / omega_step
        return slowness_lo < s < slowness_hi + slowness_slack

    def _det(kz: complex, omega_step: float) -> complex:
        return _modal_determinant_n0_complex(
            kz,
            omega_step,
            vp,
            vs,
            rho,
            vf,
            rho_f,
            a,
            leaky_p=False,
            leaky_s=True,
        )

    kz_curve_desc = _march_complex_dispersion_validated(
        _det,
        f_desc,
        kz_seed,
        validator=_validator,
        max_consecutive_invalid=3,
    )

    omega_desc = 2.0 * np.pi * f_desc
    with np.errstate(invalid="ignore"):
        slowness_desc = kz_curve_desc.real / omega_desc
    attenuation_desc = kz_curve_desc.imag
    finite_desc = np.isfinite(kz_curve_desc.real) & np.isfinite(kz_curve_desc.imag)
    slowness_desc = np.where(finite_desc, slowness_desc, np.nan)
    attenuation_desc = np.where(finite_desc, attenuation_desc, np.nan)

    slowness[order_desc] = slowness_desc
    attenuation[order_desc] = attenuation_desc

    return BoreholeMode(
        name="pseudo_rayleigh",
        azimuthal_order=0,
        freq=f_arr,
        slowness=slowness,
        attenuation_per_meter=attenuation,
    )
