"""n=2 quadrupole mode: modal determinant and dispersion solver.

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
from fwap.cylindrical_solver._dataclasses import (
    BoreholeLayer,
    BoreholeMode,
    _validate_borehole_layers,
    _validate_flexural_layers_stacked,
)

# =====================================================================
# n = 2 quadrupole modal determinant (plan item D)
# =====================================================================
#
# General-n extension of the n = 0 / n = 1 derivations. The
# Helmholtz-decomposition machinery, gauge choice, and BC structure
# are identical to n = 1 (substep blocks 1.1 - 1.6 above); the only
# thing that changes is which (n-1, n) Bessel pair appears in each
# entry and which factors of ``n`` come out of the
# ``d_theta cos(n theta) = -n sin(n theta)`` step.
#
# Generalisation rules used to build the entries below (verified by
# specialising to n = 1 and matching the existing
# :func:`_modal_determinant_n1` line by line):
#
# * Wherever the n = 1 form has ``I_0 / I_1``, the general form has
#   ``I_{n-1} / I_n``; same for ``K_0 / K_1 -> K_{n-1} / K_n``.
# * Each azimuthal-derivative factor of 1 in the n = 1 form
#   generalises to ``n`` (e.g., the ``- K_1(sa)/a`` in M14 becomes
#   ``- n K_n(sa)/a`` at general n).
# * Each ``2 K_1(pa)/a^2`` "1/r^2 correction" in M22 / M32 / M34
#   generalises via the ``K_{n-2}(pa) -> K_n(pa) - 2(n-1) K_{n-1}(pa)/(pa)``
#   recurrence to a clean ``2 n(n+1) K_n(.)/a^2`` form (the
#   ``K_{n-2}`` cancels against an offsetting term in the
#   ``2(n-1)`` recurrence coefficient and leaves only ``K_n / K_{n-1}``
#   evaluations).
# * The sigma_rz C-coefficient picks up an ``(n^2 - 1)/a^2``
#   correction at general n (zero at n = 1, finite at n >= 2)
#   from the new ``(1 - n) / r * K_n(sr)`` term in u_z that vanishes
#   for the dipole case but contributes for the quadrupole.
#
# Specialised to n = 2: K_{n-1} = K_1, K_n = K_2, I_{n-1} = I_1,
# I_n = I_2; the ``n(n+1) = 6`` and ``n^2 - 1 = 3`` factors that
# appear repeatedly below come out of those rules.
#
# The whole module-docstring sign convention (time dependence
# ``e^{-i omega t}``, ``e^{i k_z z}``, ``e^{i n theta}``) and the
# row-4-by-i / column-C-by-(-i) phase rescaling that makes the
# bound-regime matrix purely real are unchanged from n = 1; the
# entries below are already in the real form.


def _modal_determinant_n2(
    kz: float,
    omega: float,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
) -> float:
    r"""
    4x4 quadrupole (n = 2) modal determinant in the bound-mode
    regime.

    Same boundary-condition layout, gauge choice, and row-4 / column-C
    phase rescaling as :func:`_modal_determinant_n1`; the entries
    differ only in the Bessel-function index pair ``(K_{n-1}, K_n) =
    (K_1, K_2)`` and the explicit ``n = 2`` factors that come out of
    the ``d_theta cos(n theta) = -n sin(n theta)`` step.

    Field representation (bound regime, ``cos(2 theta)`` and
    ``sin(2 theta)`` sectors):

    * Fluid pressure:  :math:`P = A I_2(F r) \cos(2 \theta)`,
      :math:`F = \sqrt{k_z^2 - \omega^2 / V_f^2}`.
    * Formation P scalar potential:
      :math:`\phi = B K_2(p r) \cos(2 \theta)`,
      :math:`p = \sqrt{k_z^2 - \omega^2 / V_P^2}`.
    * Formation SV vector-potential, theta component:
      :math:`\psi_\theta = C K_2(s r) \cos(2 \theta)`,
      :math:`s = \sqrt{k_z^2 - \omega^2 / V_S^2}`.
    * Formation SH vector-potential, z component:
      :math:`\psi_z = D K_2(s r) \sin(2 \theta)`.

    Matrix entries (post-rescaling, all real when ``kz, F, p, s``
    are all real positive):

    Row 1 (BC1, ``u_r^{(f)} - u_r^{(s)} = 0``, cos(2 theta) sector):
        ``[ (F I_1(Fa) - 2 I_2(Fa) / a) / (rho_f omega^2),
            p K_1(pa) + 2 K_2(pa) / a,
            kz K_2(sa),
            -2 K_2(sa) / a ]``

    Row 2 (BC2, ``-(sigma_rr^{(s)} + P) = 0``, cos(2 theta) sector,
    row negated for visual parallel with the n = 0 / n = 1 forms):
        ``[ -I_2(Fa),
            -mu * [(2 kz^2 - kS^2) K_2(pa) + 2 p K_1(pa)/a + 12 K_2(pa)/a^2],
            -2 mu kz * [s K_1(sa) + 2 K_2(sa)/a],
            +4 mu * [s K_1(sa)/a + 3 K_2(sa)/a^2] ]``

    Row 3 (BC3, ``sigma_r_theta^{(s)} = 0``, sin(2 theta) sector):
        ``[ 0,
            +4 mu * [p K_1(pa)/a + 3 K_2(pa)/a^2],
            +2 mu kz K_2(sa)/a,
            -mu * [(s^2 + 12/a^2) K_2(sa) + 2 s K_1(sa)/a] ]``

    Row 4 (BC4, ``sigma_rz^{(s)} = 0``, cos(2 theta) sector,
    after row-4-by-i / column-C-by-(-i) rescale):
        ``[ 0,
            +2 mu kz * [p K_1(pa) + 2 K_2(pa)/a],
            +mu * [(2 kz^2 - kS^2) + 3/a^2] K_2(sa),
            -2 mu kz K_2(sa)/a ]``

    Where ``Fa = F a, pa = p a, sa = s a, mu = rho V_S^2,
    kS = omega / V_S``. The ``12 = 2 n(n+1)`` and ``3 = n^2 - 1``
    factors with ``n = 2`` are the only structural differences
    from the n = 1 form in :func:`_modal_determinant_n1`; the
    Bessel-index shift ``(K_0, K_1) -> (K_1, K_2)`` accounts for
    everything else.

    See Also
    --------
    _modal_determinant_n0 : The n = 0 axisymmetric (Stoneley)
        counterpart (3x3).
    _modal_determinant_n1 : The n = 1 dipole counterpart.

    References
    ----------
    * Tang, X.-M., & Cheng, A. (2004). *Quantitative Borehole
      Acoustic Methods.* Elsevier, sect. 2.5 (LWD quadrupole
      modal determinant).
    * Kurkjian, A. L., & Chang, S.-K. (1986). Acoustic multipole
      sources in fluid-filled boreholes. *Geophysics* 51(1),
      148-163 (general-n derivation, equations 8 and 9).
    """
    F = np.sqrt(kz * kz - (omega / vf) ** 2)
    p = np.sqrt(kz * kz - (omega / vp) ** 2)
    s = np.sqrt(kz * kz - (omega / vs) ** 2)
    Fa, pa, sa = F * a, p * a, s * a

    I1Fa = float(special.iv(1, Fa))
    I2Fa = float(special.iv(2, Fa))
    K1pa = float(special.kv(1, pa))
    K2pa = float(special.kv(2, pa))
    K1sa = float(special.kv(1, sa))
    K2sa = float(special.kv(2, sa))

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    # Row 1: u_r^{(f)} - u_r^{(s)} = 0 at r = a (cos(2 theta) sector).
    M11 = (F * I1Fa - 2.0 * I2Fa / a) / (rho_f * omega**2)
    M12 = p * K1pa + 2.0 * K2pa / a
    # SV column (Hansen form) -- roadmap A.8. The azimuthal-only
    # vector-potential ansatz is not a solution of the elastodynamic
    # equations for n >= 1; the cylindrical vector Laplacian couples the
    # radial and azimuthal components through a term proportional to n
    # that such an ansatz has no term to cancel. This is Schmitt &
    # Cheng's appendix column (pp. 235-236) with A = s K_{n-1} +
    # n K_n / a, rewritten in the (K_1, K_2) pair via the K_{n+1}
    # recurrence, with n = 2.
    M13 = kz * (s * K1sa + 2.0 * K2sa / a)
    M14 = -2.0 * K2sa / a

    # Row 2: -(sigma_rr^{(s)} + P) = 0 at r = a (cos(2 theta) sector;
    # row negated for visual parallel with the n = 0 / n = 1 forms).
    M21 = -I2Fa
    M22 = -mu * (two_kz2_minus_kS2 * K2pa + 2.0 * p * K1pa / a + 12.0 * K2pa / (a * a))
    M23 = -2.0 * kz * mu * (s * s * K2sa + s * K1sa / a + 6.0 * K2sa / (a * a))
    M24 = 4.0 * mu * (s * K1sa / a + 3.0 * K2sa / (a * a))

    # Row 3: sigma_r_theta^{(s)} = 0 at r = a (sin(2 theta) sector;
    # fluid carries no shear, so M31 = 0).
    M31 = 0.0
    M32 = 4.0 * mu * (p * K1pa / a + 3.0 * K2pa / (a * a))
    M33 = 4.0 * kz * mu * (s * K1sa / a + 3.0 * K2sa / (a * a))
    M34 = -mu * ((s * s + 12.0 / (a * a)) * K2sa + 2.0 * s * K1sa / a)

    # Row 4: sigma_rz^{(s)} = 0 at r = a (cos(2 theta) sector; M41 = 0
    # for the same fluid-no-shear reason). Entries below are the
    # post-rescaling form: row 4 multiplied by i and column C
    # (= column 3 here) by -i, leaving a real matrix.
    M41 = 0.0
    M42 = 2.0 * kz * mu * (p * K1pa + 2.0 * K2pa / a)
    M43 = two_kz2_minus_kS2 * mu * (s * K1sa + 2.0 * K2sa / a)
    M44 = -2.0 * kz * mu * K2sa / a

    M = np.array(
        [
            [M11, M12, M13, M14],
            [M21, M22, M23, M24],
            [M31, M32, M33, M34],
            [M41, M42, M43, M44],
        ],
        dtype=float,
    )
    return float(np.linalg.det(M))


def _modal_determinant_n2_complex(
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
    Complex-``k_z`` n=2 quadrupole modal determinant with optional
    leaky-wave branches.

    Mirrors the matrix structure of the real-valued
    :func:`_modal_determinant_n2` (see its docstring for the full
    set of entries): four boundary conditions at ``r = a`` in the
    cos(2 theta) / sin(2 theta) sectors, four unknown amplitudes
    (A in the fluid, B / C / D in the solid for the P / SV / SH
    potentials), and the same row-4-by-i / column-C-by-(-i) phase
    rescaling that makes the matrix real in the fully-bound regime.

    What's new (relative to the real version):

    * ``kz`` is complex. The radial wavenumbers F, p, s are
      complex too.
    * ``leaky_p`` / ``leaky_s`` flags select the K-Bessel (bound)
      vs Hankel (leaky) evaluator for the formation P and S
      waves; the fluid I-Bessel always uses ``iv`` (regular at
      the borehole axis), with ``F`` complex handled
      transparently by scipy.
    * Returns a complex scalar. In the fully-bound regime
      (real ``kz``, both ``leaky_*`` flags False) the imaginary
      part is zero to floating-point precision and the real
      part matches the real-only :func:`_modal_determinant_n2`
      exactly -- the regression invariant.

    Parameters
    ----------
    kz : complex
        Axial wavenumber. May be complex.
    omega, vp, vs, rho, vf, rho_f, a : float
        Same as :func:`_modal_determinant_n2`.
    leaky_p, leaky_s : bool, default False
        Select the leaky branch (Hankel evaluator) for the
        formation P and S waves. Use
        :func:`_detect_leaky_branches` to set these from
        ``(kz, omega)`` for typical regime-detection workflows.

    Returns
    -------
    complex
        ``det M(kz, omega)`` evaluated with the chosen branches.

    See Also
    --------
    _modal_determinant_n2 : The real-valued bound-only counterpart.
    _modal_determinant_n1_complex : The n=1 sister.
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
    # supports complex arguments transparently; for ``F^2 < 0``
    # (fast-formation quadrupole regime, F purely imaginary) iv
    # returns the J-Bessel-equivalent oscillatory pattern with the
    # appropriate i^n phase, and the row/column rescaling carries
    # that phase consistently.
    I1Fa = complex(special.iv(1, Fa))
    I2Fa = complex(special.iv(2, Fa))

    # Formation P and S K-Bessel pairs (or Hankel via analytic
    # continuation in the leaky regime). For n=2 we need
    # K_{n-1} = K_1 and K_n = K_2, which is exactly what
    # ``_k_or_hankel(1, ...)`` returns.
    K1pa, K2pa = _k_or_hankel(1, p, a, leaky=leaky_p)
    K1sa, K2sa = _k_or_hankel(1, s, a, leaky=leaky_s)

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz_c * kz_c - kS2

    # Same matrix layout as _modal_determinant_n2; entries are now
    # complex but the structure is identical.

    # Row 1: u_r^{(f)} - u_r^{(s)} = 0 at r = a (cos(2 theta) sector).
    M11 = (F * I1Fa - 2.0 * I2Fa / a) / (rho_f * omega**2)
    M12 = p * K1pa + 2.0 * K2pa / a
    # SV column (Hansen form) -- roadmap A.8. The azimuthal-only
    # vector-potential ansatz is not a solution of the elastodynamic
    # equations for n >= 1; the cylindrical vector Laplacian couples the
    # radial and azimuthal components through a term proportional to n
    # that such an ansatz has no term to cancel. This is Schmitt &
    # Cheng's appendix column (pp. 235-236) with A = s K_{n-1} +
    # n K_n / a, rewritten in the (K_1, K_2) pair via the K_{n+1}
    # recurrence, with n = 2.
    M13 = kz_c * (s * K1sa + 2.0 * K2sa / a)
    M14 = -2.0 * K2sa / a

    # Row 2: -(sigma_rr^{(s)} + P) = 0 at r = a (cos(2 theta) sector;
    # row negated for visual parallel with the n=0 / n=1 forms).
    M21 = -I2Fa
    M22 = -mu * (two_kz2_minus_kS2 * K2pa + 2.0 * p * K1pa / a + 12.0 * K2pa / (a * a))
    M23 = -2.0 * kz_c * mu * (s * s * K2sa + s * K1sa / a + 6.0 * K2sa / (a * a))
    M24 = 4.0 * mu * (s * K1sa / a + 3.0 * K2sa / (a * a))

    # Row 3: sigma_r_theta^{(s)} = 0 at r = a (sin(2 theta) sector;
    # fluid carries no shear, M31 = 0).
    M31 = 0.0 + 0j
    M32 = 4.0 * mu * (p * K1pa / a + 3.0 * K2pa / (a * a))
    M33 = 4.0 * kz_c * mu * (s * K1sa / a + 3.0 * K2sa / (a * a))
    M34 = -mu * ((s * s + 12.0 / (a * a)) * K2sa + 2.0 * s * K1sa / a)

    # Row 4: sigma_rz^{(s)} = 0 at r = a (cos(2 theta) sector;
    # M41 = 0 same fluid-no-shear reason). Same row-4-by-i /
    # column-C-by-(-i) rescale as the real version.
    M41 = 0.0 + 0j
    M42 = 2.0 * kz_c * mu * (p * K1pa + 2.0 * K2pa / a)
    M43 = two_kz2_minus_kS2 * mu * (s * K1sa + 2.0 * K2sa / a)
    M44 = -2.0 * kz_c * mu * K2sa / a

    M = np.array(
        [
            [M11, M12, M13, M14],
            [M21, M22, M23, M24],
            [M31, M32, M33, M34],
            [M41, M42, M43, M44],
        ],
        dtype=complex,
    )
    return complex(np.linalg.det(M))


def _quadrupole_kz_bracket(
    omega: float,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
) -> tuple[float, float]:
    """
    Bracket the n=2 quadrupole bound root in (k_z_lo, k_z_hi).

    Same shape as :func:`_flexural_kz_bracket`: the slow-formation
    bound mode has phase velocity between ``V_R`` (high-f) and
    ``V_S`` (low-f cutoff), so the slowness is in
    ``(1/V_S, ~1.1/V_R)``. The brentq caller can expand the upper
    bound outward if no sign change is found in this initial
    range -- mirrors the n = 1 bracket-expansion loop.
    """
    from fwap.cylindrical import rayleigh_speed

    vR = rayleigh_speed(vp, vs)
    kz_lo = omega / vs * (1.0 + 1.0e-6)
    kz_hi = omega / vR * 1.10
    return kz_lo, kz_hi


def _quadrupole_dispersion_fast_formation(
    freq: np.ndarray,
    *,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
) -> BoreholeMode:
    r"""
    Fast-formation (``V_S > V_f``) quadrupole dispersion (plan
    item E).

    Direct n=2 analogue of :func:`_flexural_dispersion_fast_formation`:
    in the slowness window ``(1/V_S, 1/V_R)`` (phase velocity in
    ``(V_R, V_S)``, both above ``V_f``) the fluid radial wavenumber
    becomes purely imaginary (``F^2 < 0``) while the formation
    P / S branches stay bound. The complex modal determinant
    :func:`_modal_determinant_n2_complex` evaluated at real
    ``k_z`` is overwhelmingly imaginary in this regime, so the
    root condition reduces to ``Im(det) = 0`` and brentq along
    the real axis is the natural tool. Continuation across
    frequency uses the previous root as a narrow-bracket seed,
    falling back to the wide ``(omega/V_S, omega/V_R)`` bracket
    when the narrow bracket has no sign change.

    The converged ``k_z`` is real to floating-point precision
    (mode is bound; ``F^2 < 0`` only adds an overall ``i^k`` phase
    to the determinant), so the returned
    ``BoreholeMode.attenuation_per_meter`` is ``None``.
    """

    f_arr = np.asarray(freq, dtype=float)
    n_f = f_arr.size
    slowness = np.full(n_f, np.nan, dtype=float)
    if n_f == 0:
        return BoreholeMode(
            name="quadrupole",
            azimuthal_order=2,
            freq=f_arr,
            slowness=slowness,
        )

    def _det(kz: float, _omega: float) -> complex:
        return _modal_determinant_n2_complex(
            complex(kz, 0.0),
            _omega,
            vp,
            vs,
            rho,
            vf,
            rho_f,
            a,
            leaky_p=False,
            leaky_s=False,
        )

    from fwap.cylindrical_solver._n1_isotropic import (
        _march_fast_flexural_branch,
        _real_root_function,
    )

    # Roadmap A.7: at n=2 the signal is in Re(det), not Im(det).
    # Measured rather than assumed -- see _real_root_function.
    def _real_det(kz: float, _omega: float) -> float:
        return _modal_determinant_n2(kz, _omega, vp, vs, rho, vf, rho_f, a)

    root_fn = _real_root_function(_det, f_arr, vs=vs, vf=vf)
    slowness = _march_fast_flexural_branch(
        root_fn, f_arr, vs=vs, vf=vf, real_det=_real_det
    )

    return BoreholeMode(
        name="quadrupole",
        azimuthal_order=2,
        freq=f_arr,
        slowness=slowness,
    )


def quadrupole_dispersion(
    freq: np.ndarray,
    *,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
) -> BoreholeMode:
    r"""
    Quadrupole-wave (n = 2) phase slowness vs frequency from the
    isotropic-elastic modal determinant.

    Auto-dispatches on the formation type:

    * **Slow formation** (``V_S < V_f``): bound mode in the
      slowness window ``(1/V_S, ~1.1/V_R)``. All radial
      wavenumbers ``F``, ``p``, ``s`` are real positive and the
      real-valued :func:`_modal_determinant_n2` is brentq'd
      directly.
    * **Fast formation** (``V_S > V_f``): bound mode in the
      slowness window ``(1/V_S, 1/V_R)``, now with phase velocity
      *above* ``V_f``. The fluid radial wavenumber becomes purely
      imaginary (``F^2 < 0``); the formation P / S branches stay
      real positive. Dispatched to
      :func:`_quadrupole_dispersion_fast_formation`, which
      brentq's the imaginary part of
      :func:`_modal_determinant_n2_complex` along the real
      ``k_z`` axis. Direct n=2 sister of the n=1 path
      :func:`_flexural_dispersion_fast_formation`.

    In both regimes the converged ``k_z`` is real and the
    ``BoreholeMode.attenuation_per_meter`` field is ``None``.

    Parameters
    ----------
    freq : ndarray
        Frequency grid (Hz). Must be strictly positive.
    vp, vs, rho : float
        Formation P-wave velocity (m/s), S-wave velocity (m/s),
        and bulk density (kg/m^3). Must satisfy ``vp > vs > 0``
        and ``rho > 0``.
    vf, rho_f : float
        Borehole-fluid velocity (m/s) and density (kg/m^3).
    a : float
        Borehole radius (m).

    Returns
    -------
    BoreholeMode
        ``name = "quadrupole"``, ``azimuthal_order = 2``, with
        ``freq`` echoed and ``slowness[i] = k_z(omega[i]) /
        omega[i]``. ``NaN`` at any frequency where the bracket
        fails -- typically below the geometric cutoff
        ``f ~ V_S / (2 pi a)``. The bound- and leaky-regime
        branches share this asymptotic layout; the only practical
        difference is which determinant evaluator is used
        internally (:func:`_modal_determinant_n2` for slow,
        :func:`_modal_determinant_n2_complex` for fast).

    Raises
    ------
    ValueError
        If any input is non-positive, ``vp <= vs``, or ``freq``
        contains a non-positive entry.

    Notes
    -----
    Long-wavelength asymptote (``omega a / V_S -> 0``): the lowest
    bound n = 2 root sits just above ``k_z = omega / V_S``, so
    ``slowness -> 1 / V_S`` at the geometric cutoff. The upper-
    frequency asymptote is the Scholte / fluid-loaded Rayleigh
    speed (slightly above ``1 / V_R``), same as for n = 1.

    See Also
    --------
    fwap.lwd.lwd_quadrupole_priors : phenomenological LWD-
        quadrupole prior factory the present function supersedes
        (the prior is still useful as a Viterbi seed when only
        rough V_S is known and the full formation properties are
        not).

    References
    ----------
    * Tang, X.-M., & Cheng, A. (2004). *Quantitative Borehole
      Acoustic Methods.* Elsevier, sect. 2.5 (LWD quadrupole).
      *The "fig 3.7" formerly cited here is a waveform-matching
      figure, not a quadrupole dispersion curve, and has been
      dropped; sect. 2.5 is unverified.*
    * Kurkjian, A. L., & Chang, S.-K. (1986). Acoustic multipole
      sources in fluid-filled boreholes. *Geophysics* 51(1),
      148-163.
    """
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
        # Fast formation: F^2 < 0, dispatch to complex-determinant
        # path with brentq on Im(det) along the real-kz axis.
        return _quadrupole_dispersion_fast_formation(
            f_arr,
            vp=vp,
            vs=vs,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            a=a,
        )

    slowness = np.full_like(f_arr, np.nan, dtype=float)

    for i, f in enumerate(f_arr):
        omega = 2.0 * np.pi * float(f)

        def _det(kz, omega=omega):
            return _modal_determinant_n2(kz, omega, vp, vs, rho, vf, rho_f, a)

        kz_lo, kz_hi = _quadrupole_kz_bracket(
            omega,
            vp,
            vs,
            rho,
            vf,
            rho_f,
            a,
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
                    "quadrupole_dispersion: bound-regime det evaluation "
                    "failed at f=%.1f Hz",
                    f,
                )
                continue
            if np.sign(d_lo) == np.sign(d_hi):
                logger.debug(
                    "quadrupole_dispersion: failed to bracket at "
                    "f=%.1f Hz (likely below cutoff)",
                    f,
                )
                continue
            kz_root = optimize.brentq(_det, kz_lo, kz_hi, xtol=1.0e-10)
            slowness[i] = kz_root / omega
        except (ValueError, RuntimeError) as exc:
            logger.debug(
                "quadrupole_dispersion: brentq failed at f=%.1f Hz: %s",
                f,
                exc,
            )

    return BoreholeMode(
        name="quadrupole",
        azimuthal_order=2,
        freq=f_arr,
        slowness=slowness,
    )


def _quadrupole_dispersion_fast_formation_layered(
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
    Fast-formation (``V_S > V_f``) cased-hole quadrupole
    dispersion via brentq on ``Im(det)``.

    Mirrors :func:`_quadrupole_dispersion_fast_formation` (the
    unlayered fast-formation driver) lifted to the multi-layer
    cased determinant :func:`_modal_determinant_n2_cased_complex`.
    The bound regime is ``k_z`` real in ``(omega/V_S, omega/V_R)``;
    in that window the formation P / S radial wavenumbers stay
    real while the fluid radial wavenumber goes purely imaginary,
    and the cased determinant evaluated at real ``k_z`` has an
    imaginary part that crosses zero at the modal root. Both the
    layer and formation columns use ``leaky_p = leaky_s = False``
    -- the mode is bound everywhere outside the borehole fluid.

    Continuation strategy mirrors the unlayered case: walk
    high-to-low frequency, narrow bracket centred on the previous
    step's slowness first, fall back to the wide
    ``(omega/V_S, omega/V_R)`` bracket if the narrow one fails.
    The converged ``k_z`` is real to floating-point precision; the
    returned ``BoreholeMode.attenuation_per_meter`` is therefore
    ``None``.
    """
    from fwap.cylindrical_solver import _modal_determinant_n2_cased_complex

    f_arr = np.asarray(freq, dtype=float)
    n_f = f_arr.size
    slowness = np.full(n_f, np.nan, dtype=float)
    if n_f == 0:
        return BoreholeMode(
            name="quadrupole",
            azimuthal_order=2,
            freq=f_arr,
            slowness=slowness,
        )

    def _det(kz: float, _omega: float) -> complex:
        return _modal_determinant_n2_cased_complex(
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

    from fwap.cylindrical_solver._cased import _modal_determinant_n2_cased
    from fwap.cylindrical_solver._n1_isotropic import (
        _FAST_FLEXURAL_MAX_CASED_ROOTS,
        _march_fast_flexural_branch,
        _real_root_function,
    )

    # Roadmap A.7: the n=2 signal is in Re(det). This was the whole of
    # A.7 -- the path was tracking round-off, and the propagator chain,
    # which the roadmap blamed, is accurate to 1e-16.
    def _real_det(kz: float, _omega: float) -> float:
        return _modal_determinant_n2_cased(
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
        name="quadrupole",
        azimuthal_order=2,
        freq=f_arr,
        slowness=slowness,
    )


def _quadrupole_kz_bracket_cased(
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
    Bracket the cased-hole n=2 quadrupole root in (k_z_lo, k_z_hi)
    for an arbitrary-N layer stack.

    Mirrors :func:`_flexural_kz_bracket_cased`: lower bound is the
    slowest-body-wave floor across the entire stack (fluid, every
    layer, formation half-space); upper bound is 10 % above the
    formation Rayleigh-speed slowness. The brentq expansion-loop
    in :func:`quadrupole_dispersion_layered` extends the bracket
    if the multi-layer perturbation lifts the actual root above
    the cushion.
    """
    from fwap.cylindrical import rayleigh_speed

    vR = rayleigh_speed(vp, vs)
    slowest = min(vs, vf, *(L.vs for L in layers))
    kz_lo = omega / slowest * (1.0 + 1.0e-6)
    kz_hi = omega / vR * 1.10
    return kz_lo, kz_hi


def quadrupole_dispersion_layered(
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
    Quadrupole-wave (n = 2) phase slowness vs frequency for a
    borehole with optional cased-hole annular layers between the
    fluid and the formation half-space.

    Sister of :func:`stoneley_dispersion_layered` (n=0) and
    :func:`flexural_dispersion_layered` (n=1) at azimuthal order
    2. With ``layers=()`` (no extra layers) this is bit-equivalent
    to :func:`quadrupole_dispersion`. With one or more
    :class:`BoreholeLayer` instances, the multi-layer cased-hole
    propagator-matrix path dispatches to the n=2 stacked modal
    determinant (:func:`_modal_determinant_n2_cased` for the
    slow-formation regime, G''.c; or
    :func:`_modal_determinant_n2_cased_complex` for the
    fast-formation regime where the fluid radial wavenumber
    goes purely imaginary while the formation P / S branches
    stay bound) and brentq's its lowest root across the
    frequency grid (G''.d for slow formation;
    :func:`_quadrupole_dispersion_fast_formation_layered` for
    fast formation).

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
        :func:`quadrupole_dispersion`. Multi-layer dispatch is
        plan items G''.c (stacked modal determinant) and G''.d
        (public-API hook); see
        ``docs/plans/cylindrical_biot_G_pp.md``.

    Returns
    -------
    BoreholeMode
        ``name = "quadrupole"``, ``azimuthal_order = 2``.

    Raises
    ------
    ValueError
        If any input is non-positive, ``vp <= vs``, ``freq``
        contains a non-positive entry, any layer is malformed,
        or any layer fails the slow-formation constraint
        ``layer.vs >= vs`` (multi-layer only).

    Notes
    -----
    The ``layer.vs >= vs`` constraint applies to the **multi-layer**
    path only, exactly as in :func:`flexural_dispersion_layered`.
    A single layer softer in shear than a slow formation -- an
    invaded zone, which is slower than the rock it replaces -- is
    accepted.

    Until this was corrected the check ran for every layer count,
    which made the whole invaded-zone family unrepresentable at
    ``n = 2`` while the identical model was accepted at ``n = 1``.
    The single-layer allowance is not a guess: Schmitt & Cheng
    figure 15(b) plots the screw mode for this exact configuration,
    and against the digitised curves this path returns **0.58 % rms**
    for an 8 cm invaded zone -- better than the same solver's
    1.29 % on the virgin rock of the same figure. See
    ``tests/test_solver_figures.py`` (figure 15b / A.6).

    The correction does not touch the mode's onset, which remains
    late by the near-cutoff margin recorded for the slow screw mode
    (fwap 5.6 kHz against a published 3.4 kHz for the 8 cm model).

    **External validation of the cased-hole path.** Schmitt & Cheng
    (1987) fig 21 -- "same as Figure 20 for the screw mode" -- plots this
    mode for a fast sandstone behind a well-bonded steel casing, over
    the same three cement stacks as the dipole. All three traced curves
    ship in ``docs/notebooks/_data/`` and score 0.86 %, 0.18 % and
    0.26 % RMS. The band is 6-20 kHz rather than the dipole's 2-15: the
    screw mode's useful energy sits higher, which is the report's own
    reason for saying the cement effects "occur closer to its useful
    energy due to the higher frequencies involved". As in
    :func:`flexural_dispersion_layered`, ``a`` is
    ``0.10 - t_casing - t_cement`` -- the annulus eats inward from the
    original 10 cm hole.
    """
    layers_tuple = tuple(layers)
    _validate_borehole_layers(layers_tuple)
    if not layers_tuple:
        return quadrupole_dispersion(
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
        # Fast-formation cased-hole quadrupole: phase velocity is
        # in (V_R, V_S) > V_f, so the fluid radial wavenumber goes
        # purely imaginary while the formation P / S branches
        # stay bound. Brentq on Im(det) along the real-k_z axis
        # via the complex-determinant cased helper. Per-layer
        # slow-formation constraint does NOT apply: a layer
        # softer than the formation in the fast-formation regime
        # is physically permissible (e.g., cement softer than a
        # fast carbonate formation), and the complex Bessel
        # functions in _layer_e_matrix_n2_complex handle the
        # mixed-regime layer kinematics transparently.
        return _quadrupole_dispersion_fast_formation_layered(
            f_arr,
            vp=vp,
            vs=vs,
            rho=rho,
            vf=vf,
            rho_f=rho_f,
            a=a,
            layers=layers_tuple,
        )
    if len(layers_tuple) >= 2:
        # Multi-layer path requires the per-layer slow-formation
        # constraint ``layer.vs >= vs``; reuse the n=1 helper, since
        # the constraint is the same at n=2 (Sinha-Norris-Chang-style
        # soft-formation bound-mode regime). The single-layer path
        # deliberately does *not* enforce it, mirroring
        # ``flexural_dispersion_layered``: see the note in the
        # docstring for the measurement that settled this.
        _validate_flexural_layers_stacked(layers_tuple, a, vs)

    from fwap.cylindrical_solver import _modal_determinant_n2_cased
    from fwap.cylindrical_solver._n1_layered import _slow_cased_velocity_floor

    # Multi-layer cased-hole brentq loop on top of
    # ``_modal_determinant_n2_cased`` (G''.c). Mirrors the
    # ``flexural_dispersion_layered`` n>=1 branch.
    slowness = np.full_like(f_arr, np.nan, dtype=float)
    velocity_floor = _slow_cased_velocity_floor(vs, vp, rho, vf, rho_f, layers_tuple)
    for i, f in enumerate(f_arr):
        omega = 2.0 * np.pi * float(f)

        def _det(kz_, omega=omega, layers_tuple=layers_tuple):
            return _modal_determinant_n2_cased(
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

        kz_lo, kz_hi = _quadrupole_kz_bracket_cased(
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
                    "quadrupole_dispersion_layered: det evaluation "
                    "NaN at f=%.1f Hz (likely outside bound regime)",
                    f,
                )
                continue
            if np.sign(d_lo) == np.sign(d_hi):
                logger.debug(
                    "quadrupole_dispersion_layered: failed to bracket "
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
                    "quadrupole_dispersion_layered: rejected a %.1f m/s root at "
                    "f=%.1f Hz, below the %.1f m/s Scholte floor",
                    omega / kz_root,
                    f,
                    velocity_floor,
                )
                continue
            slowness[i] = kz_root / omega
        except (ValueError, RuntimeError) as exc:
            logger.debug(
                "quadrupole_dispersion_layered: brentq failed at f=%.1f Hz: %s",
                f,
                exc,
            )

    attenuation = _fill_slow_cased_leaky_n2(
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
        name="quadrupole",
        azimuthal_order=2,
        freq=f_arr,
        slowness=slowness,
        attenuation_per_meter=attenuation,
    )


def _fill_slow_cased_leaky_n2(
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
    n=2 sister of
    :func:`~fwap.cylindrical_solver._n1_layered._fill_slow_cased_leaky_n1`.

    Roadmap A.9. Fills the frequencies where the bound n=2 layered
    search found nothing with the leaky cased branch, in place. See the
    n=1 twin for the physics and
    :func:`~fwap.cylindrical_solver._leaky._march_leaky_cased_branch`
    for the shared marcher.

    This is a different window from roadmap A.7's. A.7 is about the
    FAST-formation cased ``n = 2`` determinant, which is
    noise-dominated -- about 90 sign changes at 12 kHz where the physics
    supports a handful, and 430 on a fine grid even with the layer set
    equal to the formation. The slow-formation leaky window scanned here
    carries one to six crossings over the whole dipole band, which is a
    mode spectrum rather than cancellation noise, so A.7 does not block
    this path.

    Parameters
    ----------
    slowness : ndarray, shape (n_f,)
        Bound-path result, modified in place.
    f_arr : ndarray, shape (n_f,)
        Frequency grid (Hz).
    vp, vs, rho, vf, rho_f, a : float
        As in :func:`quadrupole_dispersion_layered`.
    layers : tuple of BoreholeLayer
        Annular stack, inside-out.

    Returns
    -------
    ndarray or None
        Attenuation (1/m), or ``None`` when the leaky branch
        contributed nothing.
    """
    missing = ~np.isfinite(slowness)
    if not layers or not missing.any():
        return None
    ceiling = min(vf, min(layer.vs for layer in layers))
    if not ceiling > vs:
        return None

    from fwap.cylindrical_solver._cased import _modal_determinant_n2_cased_complex
    from fwap.cylindrical_solver._leaky import (
        _detect_leaky_branches,
        _march_leaky_cased_branch,
    )

    def _det(kz: complex, omega: float) -> complex:
        _, leaky_p, leaky_s = _detect_leaky_branches(kz, omega, vp, vs, vf)
        return _modal_determinant_n2_cased_complex(
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
