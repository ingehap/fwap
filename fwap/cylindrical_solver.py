"""
Cylindrical-borehole modal-determinant solver (Schmitt 1988).

Implements the isotropic-elastic n=0 (axisymmetric) modal
determinant whose lowest-:math:`k_z` zero is the Stoneley wave.
Replaces the rational-interpolation phenomenology in
:mod:`fwap.cylindrical` for the Stoneley wave with a real
boundary-value-problem dispersion law derived from continuity and
stress conditions at the borehole wall.

Scope (this module)
-------------------
* Isotropic-elastic formation, single-layer (no mudcake / altered
  zone). The VTI extension lives on the roadmap as a follow-up
  (Schmitt 1989 elastic part); the numerical scaffolding here will
  be reused.
* **Bound-mode regime only**: ``k_z > omega / V_S`` (and therefore
  also ``> omega / V_P`` and ``> omega / V_f``). This covers the
  Stoneley wave throughout its band on a typical sonic record.
  The leaky-mode pseudo-Rayleigh and high-frequency leaky-flexural
  regimes need outgoing-wave (Hankel-function) boundary conditions
  and are out of scope.
* **n=0 monopole only.** The n=1 dipole flexural mode follows the
  same approach but with a 4x4 modal matrix derived from a three-
  scalar Helmholtz decomposition (P, SV, SH potentials with
  cos/sin azimuthal symmetries). It is a follow-up commit; see the
  roadmap.

Sign conventions
----------------
* Time dependence ``e^{-i omega t}``.
* Axial dependence ``e^{i k_z z}``; bound modes have ``k_z > 0``.
* Azimuthal dependence ``e^{i n theta}``.
* Radial decay constants are defined positive in the bound regime:

    F = sqrt(k_z^2 - omega^2 / V_f^2)   > 0    (fluid evanescent)
    p = sqrt(k_z^2 - omega^2 / V_P^2)   > 0    (formation P decay)
    s = sqrt(k_z^2 - omega^2 / V_S^2)   > 0    (formation S decay)

* Fluid pressure ``P = A I_0(F r)`` (the regular-at-origin
  modified Bessel; equivalent to ``J_0(alpha r)`` with
  ``alpha = i F``).
* Formation: scalar P potential ``phi = B K_0(p r)`` and SV
  vector-potential theta-component ``psi_theta = C K_1(s r)``
  (Helmholtz decomposition for axisymmetric fields; see
  Aki & Richards 2002 sect. 7.2).

Validation strategy
-------------------
Validates the modal matrix against:

* The closed-form Stoneley low-frequency formula
  ``S_ST^2 = 1/V_f^2 + rho_f / mu`` (White 1983 sect. 5.5).
* The dipole flexural mode's two asymptotic limits already
  encoded in :func:`fwap.cylindrical.flexural_dispersion_physical`:
  ``s_low = 1/V_S`` and ``s_high = 1/V_R`` from
  :func:`fwap.cylindrical.rayleigh_speed`.

If the modal matrix entries below were transcribed wrong, those
checks fail loudly.

References
----------
* Schmitt, D. P. (1988). Shear-wave logging in elastic formations.
  *Journal of the Acoustical Society of America* 84(6), 2230-2244.
* Cheng, C. H., & Toksoz, M. N. (1981). Elastic wave propagation
  in a fluid-filled borehole and synthetic acoustic logs.
  *Geophysics* 46(7), 1042-1053.
* Paillet, F. L., & Cheng, C. H. (1991). *Acoustic Waves in
  Boreholes.* CRC Press, Ch. 2-3.
* White, J. E. (1983). *Underground Sound: Application of Seismic
  Waves.* Elsevier, sect. 5.5 (Stoneley low-f closed form).
* Aki, K., & Richards, P. G. (2002). *Quantitative Seismology*,
  2nd ed., sect. 7.2 (Helmholtz scalar/vector potential
  decomposition in cylindrical coordinates).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special

from fwap._common import logger


@dataclass
class BoreholeMode:
    """
    Per-frequency phase-slowness curve of a single guided mode.

    Attributes
    ----------
    name : str
        Mode label (``"Stoneley"`` for the n=0 first root,
        ``"flexural"`` for the n=1 first root).
    azimuthal_order : int
        Cylindrical mode index (``0`` for monopole, ``1`` for
        dipole).
    freq : ndarray, shape (n_f,)
        Frequencies (Hz).
    slowness : ndarray, shape (n_f,)
        Phase slowness (s/m): ``slowness[i] = k_z(omega[i]) /
        omega[i]``. ``NaN`` at frequencies where the root
        bracketing failed (typically the leaky regime above the
        bound-mode band, which is out of scope here).
    """
    name: str
    azimuthal_order: int
    freq: np.ndarray
    slowness: np.ndarray


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


# ---------------------------------------------------------------------
# n = 0 axisymmetric modal determinant (Stoneley wave)
# ---------------------------------------------------------------------


def _modal_determinant_n0(
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
    3x3 axisymmetric modal determinant in the bound-mode regime.

    Three boundary conditions at ``r = a``:

    1. continuity of radial displacement
       :math:`u_r^{(f)}(a) = u_r^{(s)}(a)`,
    2. normal-stress balance
       :math:`\sigma_{rr}^{(s)}(a) = -P^{(f)}(a)`,
    3. tangential-stress vanishing on the formation side
       :math:`\sigma_{rz}^{(s)}(a) = 0`.

    A guided mode at fixed ``omega`` is a value of ``k_z`` that
    makes ``det(M) = 0``. Returns a real scalar; multiplied by an
    overall ``i^k`` factor that does not depend on ``k_z`` and is
    therefore harmless for root-finding.

    Field representation (bound regime):

    * Fluid pressure:  :math:`P = A \, I_0(F r)`,
      :math:`F = \sqrt{k_z^2 - \omega^2 / V_f^2}`
    * Formation P potential:  :math:`\phi = B \, K_0(p r)`,
      :math:`p = \sqrt{k_z^2 - \omega^2 / V_P^2}`
    * Formation SV potential (theta component):
      :math:`\psi_\theta = C \, K_1(s r)`,
      :math:`s = \sqrt{k_z^2 - \omega^2 / V_S^2}`

    Derivation summary (full derivation in module docstring):

    * Fluid radial displacement
      ``u_r^{(f)}(r) = (A F / (rho_f omega^2)) I_1(F r)``
      from the Euler equation with pressure
      :math:`P = A I_0(F r)`.
    * Formation displacement components
      ``u_r^{(s)} = -B p K_1(p r) - i k_z C K_1(s r)``
      and ``u_z^{(s)} = i k_z B K_0(p r) - C s K_0(s r)``
      from the Helmholtz decomposition, using
      ``K_0' = -K_1`` and the cylindrical-coordinate
      identity ``(1/r) d(r K_1(s r))/dr = -s K_0(s r)``.
    * Stress combinations involve the standard reduction
      ``-lambda omega^2/V_P^2 + 2 mu p^2 = mu (2 k_z^2 - k_S^2)``
      with :math:`k_S = \omega / V_S`.

    After multiplying row 3 by ``i`` and column 3 by ``-i`` (which
    leaves the determinant unchanged because the two factors of
    ``i`` and ``-i`` multiply to 1), every entry is purely real:

    Row 1 (continuity of u_r): coefficients of (A, B, C):
        ``[ F I_1(Fa) / (rho_f omega^2),
            p K_1(pa),
            k_z K_1(sa) ]``

    Row 2 (sigma_rr + P = 0): coefficients of (A, B, C):
        ``[ -I_0(Fa),
            -mu [(2 k_z^2 - k_S^2) K_0(pa) + 2 p K_1(pa)/a],
            -2 k_z mu [s K_0(sa) + K_1(sa)/a] ]``

    Row 3 (sigma_rz = 0; rescaled by i): coefficients of (A, B, C):
        ``[ 0,
            2 k_z p mu K_1(pa),
            mu (2 k_z^2 - k_S^2) K_1(sa) ]``

    Where ``Fa = F a``, ``pa = p a``, ``sa = s a``, ``mu = rho V_S^2``,
    ``k_S = omega / V_S``.
    """
    F = np.sqrt(kz * kz - (omega / vf) ** 2)
    p = np.sqrt(kz * kz - (omega / vp) ** 2)
    s = np.sqrt(kz * kz - (omega / vs) ** 2)
    Fa, pa, sa = F * a, p * a, s * a

    I0Fa, I1Fa = _i0_i1(Fa)
    K0pa, K1pa = _k0_k1(pa)
    K0sa, K1sa = _k0_k1(sa)

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz * kz - kS2

    # Row 1: continuity of u_r at r = a.
    M11 = F * I1Fa / (rho_f * omega ** 2)
    M12 = p * K1pa
    M13 = kz * K1sa

    # Row 2: sigma_rr^{(s)} = -P^{(f)} at r = a.
    M21 = -I0Fa
    M22 = -mu * (two_kz2_minus_kS2 * K0pa + 2.0 * p * K1pa / a)
    M23 = -2.0 * kz * mu * (s * K0sa + K1sa / a)

    # Row 3: sigma_rz^{(s)} = 0 at r = a (rescaled by i so all
    # entries are real).
    M31 = 0.0
    M32 = 2.0 * kz * p * mu * K1pa
    M33 = mu * two_kz2_minus_kS2 * K1sa

    M = np.array([[M11, M12, M13],
                  [M21, M22, M23],
                  [M31, M32, M33]], dtype=float)
    return float(np.linalg.det(M))


# ---------------------------------------------------------------------
# Stoneley dispersion: track the lowest n=0 root across frequency
# ---------------------------------------------------------------------


def _stoneley_kz_bracket(
    omega: float, vp: float, vs: float, rho: float,
    vf: float, rho_f: float, a: float,
) -> tuple[float, float]:
    """
    Bracket the n=0 Stoneley root in (k_z_lo, k_z_hi).

    The Stoneley wave is the slowest of the bound modes; its k_z
    is strictly larger than the body-wave k_z's, so a generous
    bracket starts just above ``omega / min(V_S, V_f)`` and runs
    out to a few times the closed-form Stoneley k_z estimate.
    """
    # Closed-form low-f estimate of k_z (White 1983):
    # S_ST^2 = 1/V_f^2 + rho_f / mu, k_z ~ omega * S_ST.
    mu = rho * vs * vs
    s_st_lf = np.sqrt(1.0 / vf ** 2 + rho_f / mu)
    kz_lf_est = omega * s_st_lf
    # Lower bound: just above the slowest body-wave k_z so all radial
    # decay constants are real.
    kz_lo = omega / min(vs, vf) * (1.0 + 1.0e-6)
    # Upper bound: well above the closed-form estimate to catch the
    # slight rightward drift at higher frequency.
    kz_hi = max(kz_lf_est * 1.5, kz_lo * 2.0)
    return kz_lo, kz_hi


def stoneley_dispersion(
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
    Stoneley-wave phase slowness vs frequency from the n=0
    isotropic-elastic modal determinant.

    Tracks the slowest-:math:`k_z` zero of the n=0 modal
    determinant across the supplied frequency grid. At each
    frequency the bound regime is :math:`k_z > \omega/V_S`; a
    bracketing search seeded by the closed-form Stoneley low-f
    estimate refines the root via
    :func:`scipy.optimize.brentq`.

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
        ``name = "Stoneley"``, ``azimuthal_order = 0``, with
        ``freq`` echoed and ``slowness[i]`` the phase slowness
        ``k_z(omega[i]) / omega[i]``. ``NaN`` at any frequency
        where the bracket failed (rare in the bound regime).

    Raises
    ------
    ValueError
        If any input is non-positive, ``vp <= vs``, or ``freq``
        contains a non-positive entry.
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

    slowness = np.full_like(f_arr, np.nan, dtype=float)
    for i, f in enumerate(f_arr):
        omega = 2.0 * np.pi * float(f)

        def _det(kz):
            return _modal_determinant_n0(
                kz, omega, vp, vs, rho, vf, rho_f, a)

        kz_lo, kz_hi = _stoneley_kz_bracket(
            omega, vp, vs, rho, vf, rho_f, a)
        try:
            d_lo = _det(kz_lo)
            d_hi = _det(kz_hi)
            # If the bracket doesn't straddle a sign change, expand
            # outward in steps of 1.5x. The Stoneley root sits below
            # the closed-form Lf estimate at high frequency, but
            # above the body-wave k_z floor at low frequency.
            n_expand = 0
            while np.sign(d_lo) == np.sign(d_hi) and n_expand < 8:
                kz_hi *= 1.5
                d_hi = _det(kz_hi)
                n_expand += 1
            if np.sign(d_lo) == np.sign(d_hi):
                logger.debug(
                    "stoneley_dispersion: failed to bracket at f=%.1f Hz",
                    f,
                )
                continue
            kz_root = optimize.brentq(_det, kz_lo, kz_hi, xtol=1.0e-10)
            slowness[i] = kz_root / omega
        except (ValueError, RuntimeError) as exc:
            logger.debug(
                "stoneley_dispersion: brentq failed at f=%.1f Hz: %s",
                f, exc,
            )

    return BoreholeMode(
        name="Stoneley",
        azimuthal_order=0,
        freq=f_arr,
        slowness=slowness,
    )


# =====================================================================
# n = 1 dipole flexural mode -- conventions and field ansatz
# =====================================================================
#
# The block below pins the conventions for the n=1 dipole flexural
# solver. Substeps that follow (matrix construction, root-finder,
# public API) all reference the symbol names and sign choices fixed
# here; later commits MUST NOT silently re-letter the unknowns or
# flip the azimuthal split, because a transcription bug introduced by
# such a rename would only surface as a wrong number from
# ``flexural_dispersion`` later.
#
# Inheritance from n = 0
# ----------------------
# The sign conventions in the module docstring carry over verbatim:
#
# * Time dependence  ``e^{-i omega t}``
# * Axial dependence ``e^{i k_z z}`` with bound modes ``k_z > 0``
# * Bound regime     ``k_z > omega / V_S``  (so ``F, p, s`` real, > 0)
#
#   F = sqrt(k_z^2 - omega^2 / V_f^2)        (fluid evanescent)
#   p = sqrt(k_z^2 - omega^2 / V_P^2)        (formation P decay)
#   s = sqrt(k_z^2 - omega^2 / V_S^2)        (formation S decay)
#
# The scalar Helmholtz decomposition (P potential ``phi``, vector
# potential ``psi``) is identical; only the azimuthal index
# ``e^{i n theta}`` flips from ``n = 0`` to ``n = 1``.
#
# Azimuthal split (new at n = 1)
# ------------------------------
# At n = 1 the boundary conditions decouple into two independent
# Fourier sectors. We pick the cos(theta) branch so the field
# components on each sector are:
#
#     cos(theta) sector :  u_r , u_z , sigma_rr , sigma_rz
#     sin(theta) sector :  u_theta , sigma_r_theta
#
# The four boundary conditions at ``r = a`` distribute as:
#
#     row 1   u_r continuity              (cos theta)
#     row 2   sigma_rr balance            (cos theta)
#     row 3   sigma_r_theta = 0           (sin theta)
#     row 4   sigma_rz = 0                (cos theta)
#
# After stripping the azimuthal factor, the four BCs close on the
# four amplitudes (A, B, C, D) defined below. Mixing of cos / sin
# sectors in any single row indicates a sign error in the ansatz
# and must be caught before transcription (substep 1.4 cross-check).
#
# Field representation, bound regime
# ----------------------------------
# Fluid pressure (regular at ``r = 0``):
#
#     P  =  A * I_1(F r) * cos(theta)
#
# Formation P scalar potential (decaying at ``r -> infty``):
#
#     phi  =  B * K_1(p r) * cos(theta)
#
# Formation SV vector-potential, theta component:
#
#     psi_theta  =  C * K_1(s r) * cos(theta)
#
# Formation SH vector-potential, z component:
#
#     psi_z  =  D * K_1(s r) * sin(theta)
#
# All four amplitudes ``A, B, C, D`` are taken complex in general;
# in the bound regime, after the row/column phase rescaling fixed
# in substep 1.5, the resulting 4x4 modal matrix has purely real
# entries.
#
# Sector-closure check (the constraint that pins the cos / sin
# choices above): u_r and u_z must live on the cos(theta) sector
# to match the fluid pressure ``P proportional to cos(theta)`` and
# the resulting fluid u_r and u_z. In the curl identity
# ``(curl psi)_r = (1/r) d_theta(psi_z) - d_z(psi_theta)`` the term
# ``(1/r) d_theta(psi_z)`` lands on cos(theta) only if ``psi_z``
# carries sin(theta); the term ``d_z(psi_theta)`` lands on
# cos(theta) only if ``psi_theta`` carries cos(theta). The same
# constraint independently fixes u_z via
# ``(curl psi)_z = (1/r) d_r(r psi_theta)`` (cos(theta) iff
# ``psi_theta`` ~ cos(theta)). The earlier draft of this block had
# the two factors swapped; substep 1.2's full per-component
# derivation is what surfaced the inconsistency, exactly the kind
# of self-check substep 1.6 is supposed to catch in code form.
#
# Why two solid-side vector-potential components (not one)
# --------------------------------------------------------
# At n = 0 axisymmetric the SH polarization decouples and only
# ``psi_theta`` is needed (see ``_modal_determinant_n0`` above).
# At n >= 1 the tangential-shear (sigma_r_theta = 0) and axial-
# shear (sigma_rz = 0) boundary conditions couple SV and SH through
# their azimuthal derivatives. Both potential components must be
# retained: ``psi_theta`` carries the SV part of the wall response
# on the cos(theta) sector, while ``psi_z`` carries the SH part on
# the sin(theta) sector. The two sectors couple through the four
# boundary conditions enumerated above.
#
# Gauge fixing
# ------------
# We adopt ``div psi = 0`` (the standard Helmholtz gauge) and set
# ``psi_r = 0``. The remaining two solid-side vector-potential
# components ``psi_theta`` and ``psi_z`` together with the P scalar
# potential ``phi`` give three solid-side amplitudes; with the
# fluid pressure amplitude that is four unknowns total, matching
# the four boundary conditions above.
#
# Kurkjian & Chang (1986) eq. 4 use an equivalent gauge written in
# terms of ``(chi, Gamma)`` potentials; the algebra in substeps
# 1.2-1.4 follows Paillet & Cheng (1991) ch. 4 with ``psi_r = 0``
# already enforced.
#
# References
# ----------
# * Kurkjian, A. L., & Chang, S.-K. (1986). Acoustic multipole
#   sources in fluid-filled boreholes. *Geophysics* 51(1), 148-163.
#   Eqs. 4-9 give the azimuthal-order field decomposition; the
#   most explicit derivation of the n=1 4x4 system in print.
# * Paillet, F. L., & Cheng, C. H. (1991). *Acoustic Waves in
#   Boreholes*, ch. 4. CRC Press. The four-potential cylindrical
#   decomposition with explicit n=1 forms; the boundary-condition
#   table this block implements.
# * Schmitt, D. P. (1988). Shear-wave logging in elastic
#   formations. *J. Acoust. Soc. Am.* 84(6), 2230-2244 (the
#   isotropic n=1 dispersion curves the validation tests target).
# * Aki, K., & Richards, P. G. (2002). *Quantitative Seismology*,
#   2nd ed., sect. 7.2. Vector-potential Helmholtz decomposition
#   and gauge choice in cylindrical coordinates.
#
# =====================================================================
# Substep 1.2 -- displacements from the four potentials
# =====================================================================
#
# Goal: write u_r, u_theta, u_z in each region as linear combinations
# of (A, B, C, D) with all azimuthal factors made explicit. The
# results below feed substep 1.3 (stress components) directly.
#
# Cylindrical-coordinate identities
# ---------------------------------
# Scalar gradient:
#
#     (grad phi)_r     = d_r phi
#     (grad phi)_theta = (1/r) d_theta phi
#     (grad phi)_z     = d_z phi
#
# Vector curl (with our gauge psi_r = 0, so the psi_r-bearing terms
# are dropped):
#
#     (curl psi)_r     = (1/r) d_theta(psi_z) - d_z(psi_theta)
#     (curl psi)_theta = -d_r(psi_z)
#     (curl psi)_z     = (1/r) d_r(r psi_theta)
#
# Bessel recurrences used below (derive from
# ``I_n'(x) = I_{n-1}(x) - n I_n(x)/x`` and
# ``K_n'(x) = -K_{n-1}(x) - n K_n(x)/x``):
#
#     I_1'(x) = I_0(x) - I_1(x) / x
#     K_1'(x) = -K_0(x) - K_1(x) / x
#
# And the "(1/r) d_r [r K_1(s r)] = -s K_0(s r)" identity, which
# follows from ``d/dx [x K_1(x)] = -x K_0(x)``:
#
#     d_r [r K_1(s r)] = K_1(s r) + r s K_1'(s r)
#                      = K_1(s r) + r s [-K_0(s r) - K_1(s r)/(s r)]
#                      = -r s K_0(s r)
#     ==>  (1/r) d_r [r K_1(s r)] = -s K_0(s r).
#
# z-derivative: ``e^{i k_z z}`` makes ``d_z -> i k_z``.
#
# Fluid region (r < a)
# --------------------
# The fluid is inviscid; the only scalar is the pressure. Linearised
# Euler with time dependence ``e^{-i omega t}``:
#
#     -rho_f omega^2 u^{(f)} = -grad P     ==>   u^{(f)} = grad P / (rho_f omega^2).
#
# With ``P = A I_1(F r) cos(theta) e^{i k_z z}``:
#
#     u_r^{(f)}     = (A / (rho_f omega^2)) F I_1'(F r) * cos(theta)
#                   = (A / (rho_f omega^2)) [F I_0(F r) - I_1(F r) / r] * cos(theta)
#     u_theta^{(f)} = -(A / (rho_f omega^2 r)) I_1(F r) * sin(theta)
#     u_z^{(f)}     = (i k_z A / (rho_f omega^2)) I_1(F r) * cos(theta)
#
# Sector check: u_r and u_z carry cos(theta), u_theta carries sin(theta).
# Matches the cos / sin partition pinned in 1.1.
#
# Solid region (r > a)
# --------------------
# Substitute the gauge-fixed Helmholtz decomposition
# ``u^{(s)} = grad phi + curl psi`` with the 1.1 ansatz
# (``phi = B K_1(p r) cos(theta)``, ``psi_theta = C K_1(s r) cos(theta)``,
# ``psi_z = D K_1(s r) sin(theta)``).
#
# Per-component derivation:
#
# u_r^{(s)} = d_r phi + (1/r) d_theta(psi_z) - d_z(psi_theta)
#
#     d_r phi              = B p K_1'(p r) cos(theta)
#                          = -B [p K_0(p r) + K_1(p r) / r] cos(theta)
#     (1/r) d_theta(psi_z) = (1/r) D K_1(s r) cos(theta)
#                          = (D / r) K_1(s r) cos(theta)
#     d_z(psi_theta)       = i k_z C K_1(s r) cos(theta)
#
#     ==> u_r^{(s)} = [ -B p K_0(p r)
#                       - B K_1(p r) / r
#                       + D K_1(s r) / r
#                       - i k_z C K_1(s r) ] * cos(theta)
#
# u_theta^{(s)} = (1/r) d_theta phi + (curl psi)_theta
#               = (1/r) d_theta phi - d_r(psi_z)
#
#     (1/r) d_theta phi = -(B / r) K_1(p r) sin(theta)
#     d_r(psi_z)        = D s K_1'(s r) sin(theta)
#                       = -D [s K_0(s r) + K_1(s r) / r] sin(theta)
#
#     ==> u_theta^{(s)} = [ -B K_1(p r) / r
#                           + D s K_0(s r)
#                           + D K_1(s r) / r ] * sin(theta)
#
# u_z^{(s)} = d_z phi + (curl psi)_z
#           = d_z phi + (1/r) d_r(r psi_theta)
#
#     d_z phi               = i k_z B K_1(p r) cos(theta)
#     (1/r) d_r(r psi_theta) = (C cos(theta) / r) * d_r[r K_1(s r)]
#                            = (C cos(theta) / r) * [-r s K_0(s r)]
#                            = -C s K_0(s r) cos(theta)
#
#     ==> u_z^{(s)} = [ i k_z B K_1(p r)
#                       - C s K_0(s r) ] * cos(theta)
#
# Sector check: u_r and u_z on cos(theta), u_theta on sin(theta).
# Matches the partition pinned in 1.1; ``A, B, D`` appear with cos
# in u_r and ``B, C`` with cos in u_z, while ``B, D`` carry sin in
# u_theta. No sector cross-talk -- the gauge + ansatz close as
# advertised.
#
# Wall (r = a) summary -- direct input to substep 1.3
# ---------------------------------------------------
# Drop the azimuthal factors and the implicit ``e^{i k_z z}``;
# write ``Fa = F a, pa = p a, sa = s a`` for compactness.
#
# Fluid (cos(theta) sector except u_theta which is sin(theta)):
#
#     u_r^{(f)}(a) / cos = (A / (rho_f omega^2)) [F I_0(Fa) - I_1(Fa) / a]
#     u_theta^{(f)}(a) / sin = -(A / (rho_f omega^2 a)) I_1(Fa)
#     u_z^{(f)}(a) / cos = (i k_z A / (rho_f omega^2)) I_1(Fa)
#
# Solid:
#
#     u_r^{(s)}(a) / cos = -B p K_0(pa)
#                          - B K_1(pa) / a
#                          + D K_1(sa) / a
#                          - i k_z C K_1(sa)
#
#     u_theta^{(s)}(a) / sin = -B K_1(pa) / a
#                              + D s K_0(sa)
#                              + D K_1(sa) / a
#
#     u_z^{(s)}(a) / cos = i k_z B K_1(pa) - C s K_0(sa)
#
# The factor ``i k_z`` on the C and "i k_z B" terms anticipates the
# substep-1.5 phase rescaling: we will multiply the C column by ``-i``
# and the row carrying these factors by ``i`` so the combined effect
# leaves ``det M_1`` unchanged but kills every imaginary entry.

# =====================================================================
# Substep 1.3.a -- Hooke's law, strains, and the Lame-reduction trick
# =====================================================================
#
# Goal: pin the strain-displacement, Hooke, and divergence identities
# that 1.3.c-e use to turn the displacement forms in 1.2 into the
# four stress quantities the boundary conditions need at r = a:
#
#     sigma_rr^{(f)}(a)       (= -P(a); fluid carries no shear)
#     sigma_rr^{(s)}(a)       (cos(theta) sector; BC2)
#     sigma_r_theta^{(s)}(a)  (sin(theta) sector; BC3)
#     sigma_rz^{(s)}(a)       (cos(theta) sector; BC4)
#
# Strain-displacement relations, only the three components used at r = a:
#
#     eps_rr        = d_r u_r
#     eps_r_theta   = (1/2) [ (1/r) d_theta u_r + d_r u_theta - u_theta / r ]
#     eps_rz        = (1/2) [ d_z u_r + d_r u_z ]
#
# Hooke's law (isotropic solid, Lame parameters lambda, mu):
#
#     sigma_rr        = lambda * div(u) + 2 mu * eps_rr
#     sigma_r_theta   = 2 mu * eps_r_theta
#     sigma_rz        = 2 mu * eps_rz
#
# Note that lambda enters only sigma_rr; the two shear stresses are
# pure mu, which is what makes substeps 1.3.d and 1.3.e short.
#
# Divergence in cylindrical coordinates:
#
#     div(u) = (1/r) d_r(r u_r) + (1/r) d_theta(u_theta) + d_z(u_z)
#
# Helmholtz reduction. With the gauge ``u = grad phi + curl psi`` (and
# ``psi_r = 0`` from 1.1):
#
#     div(u) = div(grad phi) + div(curl psi)
#            = laplacian(phi) + 0
#            = laplacian(phi).
#
# In the formation, phi satisfies the P-wave equation
# ``laplacian(phi) + (omega / V_P)^2 phi = 0``, so
#
#     laplacian(phi) = - (omega / V_P)^2 phi
#                    = - k_P^2 phi.
#
# This is what kills the ``lambda``-bearing term in sigma_rr at the
# wall: substituting laplacian(phi) = -k_P^2 phi and using the
# definition p^2 = k_z^2 - k_P^2, the lambda-vs-mu combination
# collapses via the algebraic identity
#
#     - lambda * k_P^2 + 2 mu * p^2 = mu * (2 k_z^2 - k_S^2),
#
# i.e. the same Lame reduction used in the n=0 derivation
# (``_modal_determinant_n0`` docstring; see the
# ``mu * (2 k_z^2 - k_S^2)`` line). Verification: from
# ``rho V_P^2 = lambda + 2 mu`` and ``rho V_S^2 = mu``,
#
#     LHS = - lambda * (rho omega^2 / (lambda + 2 mu)) + 2 mu * p^2
#         = - lambda * (rho omega^2 / (lambda + 2 mu))
#           + 2 mu * (k_z^2 - omega^2 / V_P^2)
#         = 2 mu * k_z^2
#           - rho omega^2 * (lambda + 2 mu) / (lambda + 2 mu)
#         = 2 mu * k_z^2 - rho omega^2
#         = mu * (2 k_z^2 - k_S^2)              (using mu k_S^2 = rho omega^2).
#
# This identity is the only place lambda enters the n=1 derivation;
# substeps 1.3.d (sigma_r_theta) and 1.3.e (sigma_rz) use only mu.
#

# =====================================================================
# Substep 1.3.b -- Bessel second-derivative identities
# =====================================================================
#
# Goal: pin compact forms for ``d_r^2 K_1(p r)`` and ``d_r^2 K_1(s r)``,
# the only second derivatives needed in 1.3.c-e. Forms below are
# written in K_0 and K_1 only, matching the existing ``_k0_k1``
# helper -- no K_2 helper is needed in code.
#
# Modified-Bessel ODE for K_1 (n=1):
#
#     x^2 K_1''(x) + x K_1'(x) - (x^2 + 1) K_1(x) = 0
#
# Combined with the recurrence ``K_1'(x) = -K_0(x) - K_1(x) / x``
# (from substep 1.2), this gives
#
#     K_1''(x) = K_1(x) + K_0(x) / x + 2 K_1(x) / x^2.       (*)
#
# Derivation:
#     K_1''(x) = [(x^2 + 1) K_1(x) - x K_1'(x)] / x^2
#              = K_1(x) + K_1(x) / x^2 - K_1'(x) / x
#              = K_1(x) + K_1(x) / x^2
#                + K_0(x) / x + K_1(x) / x^2
#              = K_1(x) + K_0(x) / x + 2 K_1(x) / x^2.
#
# Equivalent compressed form via the recurrence
# ``K_2(x) = K_0(x) + 2 K_1(x) / x``:
#
#     K_1''(x) = K_1(x) + K_2(x) / x.
#
# We use form (*) downstream because the existing ``_k0_k1`` helper
# returns exactly K_0 and K_1; adopting a K_2 helper would add code
# for no algebraic gain.
#
# Per-coordinate forms used in 1.3.c-e (chain rule with x = p r or
# x = s r introduces a factor of p or s on each derivative):
#
#     d_r^2 [K_1(p r)] = p^2 * K_1''(p r)
#                      = p^2 K_1(p r)
#                        + p K_0(p r) / r
#                        + 2 K_1(p r) / r^2
#
#     d_r^2 [K_1(s r)] = s^2 * K_1''(s r)
#                      = s^2 K_1(s r)
#                        + s K_0(s r) / r
#                        + 2 K_1(s r) / r^2
#
# Sanity check at high frequency: for ``p a >> 1`` and ``s a >> 1``,
# both K_0 and K_1 decay as ``e^{-x} / sqrt(x)``, so the leading term
# in each form is ``p^2 K_1(p r)`` (resp. ``s^2 K_1(s r)``). The
# subleading ``1/r`` and ``1/r^2`` corrections are what distinguish
# the cylindrical algebra from the planar half-space Rayleigh-equation
# limit that substep 1.6 will check.
#
# Fluid side: no second derivatives needed. The only fluid input to
# the four BCs is ``u_r^{(f)}(a)`` (from 1.2) and the wall pressure
# ``-P(a) = -A I_1(F a) cos(theta)`` (from the 1.1 ansatz directly).
# No I-Bessel second derivatives appear; the I-side recurrence
# ``I_1'(x) = I_0(x) - I_1(x) / x`` from 1.2 is sufficient.

# =====================================================================
# Substep 1.3.c -- solid sigma_rr with Lame reduction
# =====================================================================
#
# Goal: write sigma_rr^{(s)}(a) on the cos(theta) sector as a
# coefficient sum on (B, C, D). The fluid contribution at the wall
# is the trivial ``sigma_rr^{(f)}(a) = -P(a) = -A I_1(F a) cos(theta)``.
#
# Inputs from 1.2 (cos(theta) factor stripped; r kept symbolic):
#
#     u_r^{(s)} / cos(theta) = - B p K_0(p r)
#                              - B K_1(p r) / r
#                              + D K_1(s r) / r
#                              - i k_z C K_1(s r)
#
# Decomposition (from 1.3.a):
#
#     sigma_rr^{(s)} = lambda * laplacian(phi) + 2 mu * d_r u_r^{(s)}
#                    = - lambda k_P^2 * (B K_1(p r) cos(theta))
#                      + 2 mu * d_r u_r^{(s)}
#
# Per-amplitude differentiation of u_r^{(s)} (each line uses the
# Bessel recurrences from 1.2):
#
# B contributions to d_r u_r^{(s)}:
#
#     d_r [- B p K_0(p r)] = - B p^2 K_0'(p r) = + B p^2 K_1(p r)
#         (uses K_0'(x) = -K_1(x))
#
#     d_r [- B K_1(p r) / r]
#         = - B [ p K_1'(p r) / r - K_1(p r) / r^2 ]
#         = - B [ p (- K_0(p r) - K_1(p r) / (p r)) / r - K_1(p r) / r^2 ]
#         = + B [ p K_0(p r) / r + 2 K_1(p r) / r^2 ]
#
#     B-sum:   B [ p^2 K_1(p r) + p K_0(p r) / r + 2 K_1(p r) / r^2 ]
#
# Equivalently (1.3.b sanity check): the B contribution to u_r^{(s)}
# is ``B d_r K_1(p r)`` (since -p K_0 - K_1/r = d_r K_1(p r)), so
# ``d_r [B-part] = B d_r^2 K_1(p r)``. Substep 1.3.b gives
# d_r^2 K_1(p r) = p^2 K_1(p r) + p K_0(p r) / r + 2 K_1(p r) / r^2.
# Both routes match. [End sanity check.]
#
# C contributions to d_r u_r^{(s)}:
#
#     d_r [- i k_z C K_1(s r)]
#         = - i k_z C s K_1'(s r)
#         = - i k_z C s (- K_0(s r) - K_1(s r) / (s r))
#         = + i k_z C [ s K_0(s r) + K_1(s r) / r ]
#
# D contributions to d_r u_r^{(s)}:
#
#     d_r [+ D K_1(s r) / r]
#         = D [ s K_1'(s r) / r - K_1(s r) / r^2 ]
#         = D [ s (- K_0(s r) - K_1(s r) / (s r)) / r - K_1(s r) / r^2 ]
#         = - D [ s K_0(s r) / r + 2 K_1(s r) / r^2 ]
#
# Sum, multiplied by 2 mu:
#
#     2 mu * d_r u_r^{(s)} / cos(theta) = 2 mu * {
#         B [ p^2 K_1(p r) + p K_0(p r) / r + 2 K_1(p r) / r^2 ]
#         + i k_z C [ s K_0(s r) + K_1(s r) / r ]
#         - D [ s K_0(s r) / r + 2 K_1(s r) / r^2 ]
#     }
#
# Add the lambda piece (B-only, on K_1(p r)):
#
#     - lambda k_P^2 * B K_1(p r)
#
# This combines with ``2 mu p^2 B K_1(p r)`` from the strain piece via
# the substep-1.3.a Lame reduction
#
#     - lambda k_P^2 + 2 mu p^2 = mu (2 k_z^2 - k_S^2),
#
# leaving the B-coefficient of K_1(p r) as ``mu (2 k_z^2 - k_S^2)`` --
# the same form that appears at row 2 of ``_modal_determinant_n0``.
# All other B / C / D terms have no lambda dependence and pass
# through 2 mu untouched.
#
# Result (cos(theta) factor reinstated, r = a):
#
#     sigma_rr^{(s)}(a) / cos(theta) = mu * {
#         B [
#             (2 k_z^2 - k_S^2) K_1(p a)
#             + 2 p K_0(p a) / a
#             + 4 K_1(p a) / a^2
#         ]
#         + 2 i k_z C [
#             s K_0(s a) + K_1(s a) / a
#         ]
#         - 2 D [
#             s K_0(s a) / a + 2 K_1(s a) / a^2
#         ]
#     }
#
# Fluid side at the wall:
#
#     sigma_rr^{(f)}(a) / cos(theta) = - P(a) / cos(theta) = - A I_1(F a)
#
# Sector check
# ------------
# Every term on the right is a real coefficient (modulo the i k_z
# factor on the C entry, which is killed by the substep-1.5 phase
# rescaling) times a Bessel evaluation, all multiplied by cos(theta).
# No sin(theta) appears, confirming sigma_rr lives entirely on the
# cos(theta) sector. This is row 2 of the upcoming 4x4 in 1.4.
#
# Comparison with n = 0
# ---------------------
# The B-coefficient at n = 0 is
# ``mu * [(2 k_z^2 - k_S^2) K_0(p a) + 2 p K_1(p a) / a]``
# (see ``_modal_determinant_n0`` row 2). At n = 1 the corresponding
# expression has K_0 / K_1 swapped (because phi ~ K_1(p r) at n = 1
# instead of K_0(p r) at n = 0) and gains an extra ``+ 4 K_1(p a) / a^2``
# term -- the genuine 1/r^2 correction that 1.3.b's K_1''
# identity introduces. The C-coefficient form
# ``2 i k_z mu [s K_0(s a) + K_1(s a) / a]`` is structurally
# identical at both orders, since the SV potential ``psi_theta`` is
# proportional to K_1(s r) at both n = 0 and n = 1. The D entry is
# new at n = 1; it carries the SH potential ``psi_z`` that the n = 0
# case does not have (substep 1.1 "Why two solid-side
# vector-potential components" paragraph).

# =====================================================================
# Substep 1.3.d -- solid sigma_r_theta (sin(theta) sector)
# =====================================================================
#
# Goal: write sigma_r_theta^{(s)}(a) on the sin(theta) sector as a
# coefficient sum on (B, C, D). Fluid contribution: identically zero
# (an inviscid fluid carries no shear stress); the A entry of this
# row of the upcoming 4x4 will be 0.
#
# Decomposition (from 1.3.a, no lambda, pure mu):
#
#     sigma_r_theta^{(s)} = mu * [
#         (1/r) d_theta u_r + d_r u_theta - u_theta / r
#     ]
#
# Inputs from 1.2 (azimuthal factors stripped; r kept symbolic):
#
#     u_r^{(s)} / cos(theta) = - B p K_0(p r)
#                              - B K_1(p r) / r
#                              + D K_1(s r) / r
#                              - i k_z C K_1(s r)
#
#     u_theta^{(s)} / sin(theta) = - B K_1(p r) / r
#                                  + D s K_0(s r)
#                                  + D K_1(s r) / r
#
# The cos(theta) -> sin(theta) sector flip happens at
# ``(1/r) d_theta u_r``: since u_r ~ cos(theta), one gets
# ``d_theta u_r / sin(theta) = - [u_r / cos(theta)]``. The other two
# pieces (d_r u_theta, u_theta / r) are already on sin(theta).
#
# Piece (i):  (1/r) d_theta u_r / sin(theta)
# -----------------------------------------
#
#     = - (1/r) [ - B p K_0(p r)
#                 - B K_1(p r) / r
#                 + D K_1(s r) / r
#                 - i k_z C K_1(s r) ]
#
#     = + B p K_0(p r) / r
#       + B K_1(p r) / r^2
#       - D K_1(s r) / r^2
#       + i k_z C K_1(s r) / r
#
# Piece (ii):  d_r u_theta / sin(theta)
# -------------------------------------
# Per-amplitude differentiation:
#
#     d_r [- B K_1(p r) / r] = + B [ p K_0(p r) / r + 2 K_1(p r) / r^2 ]
#         (same calculation as the B / d_r u_r line in 1.3.c, applied
#          to the K_1(p r)/r combination)
#
#     d_r [+ D s K_0(s r)] = + D s^2 K_0'(s r)
#                          = - D s^2 K_1(s r)
#         (uses K_0'(x) = -K_1(x))
#
#     d_r [+ D K_1(s r) / r] = - D [ s K_0(s r) / r + 2 K_1(s r) / r^2 ]
#         (same calculation as the D entry in 1.3.c)
#
#     Sum:  + B [ p K_0(p r) / r + 2 K_1(p r) / r^2 ]
#           - D [ s^2 K_1(s r) + s K_0(s r) / r + 2 K_1(s r) / r^2 ]
#
# Piece (iii):  - u_theta / r / sin(theta)
# ----------------------------------------
#
#     = - (1/r) [ - B K_1(p r) / r
#                 + D s K_0(s r)
#                 + D K_1(s r) / r ]
#
#     = + B K_1(p r) / r^2
#       - D s K_0(s r) / r
#       - D K_1(s r) / r^2
#
# Sum, collected by amplitude
# ---------------------------
# Multiply the (i) + (ii) + (iii) sum by mu.
#
# B contributions (each from one or more pieces -- counts in brackets):
#
#     p K_0(p r) / r:  (i) + (ii)        =  2
#     K_1(p r) / r^2:  (i) + (ii) + (iii) =  1 + 2 + 1 = 4
#
#     ==> B-sum = 2 * [ p K_0(p r) / r + 2 K_1(p r) / r^2 ]
#
# C contributions:
#
#     K_1(s r) / r:    (i)               =  1 (with i k_z prefactor)
#
#     ==> C-sum = i k_z * K_1(s r) / r
#
# D contributions (note opposite signs between piece (ii) and the
# combined (i) + (iii) bookkeeping):
#
#     s^2 K_1(s r):     - 1 from (ii)
#     s K_0(s r) / r:   - 1 from (ii) - 1 from (iii)  =  - 2
#     K_1(s r) / r^2:   - 1 from (i) - 2 from (ii) - 1 from (iii)  =  - 4
#
#     ==> D-sum = - [ s^2 K_1(s r) + 2 s K_0(s r) / r + 4 K_1(s r) / r^2 ]
#
# Result (sin(theta) factor reinstated, r = a)
# --------------------------------------------
#
#     sigma_r_theta^{(s)}(a) / sin(theta) = mu * {
#         2 B [
#             p K_0(p a) / a + 2 K_1(p a) / a^2
#         ]
#         + i k_z C [
#             K_1(s a) / a
#         ]
#         - D [
#             s^2 K_1(s a)
#             + 2 s K_0(s a) / a
#             + 4 K_1(s a) / a^2
#         ]
#     }
#
# Fluid side at the wall:
#
#     sigma_r_theta^{(f)}(a) = 0     (inviscid fluid, no shear stress)
#
# Sector check
# ------------
# All right-hand terms carry sin(theta) (the piece-(i) cos(theta)
# was flipped to sin(theta) by ``d_theta``); no cos(theta) survives.
# This will be row 3 of the upcoming 4x4 in 1.4, with A = 0 in that
# row from the fluid-side identity above.
#
# The C entry is the smallest of the three (just K_1(s a)/a, no
# K_0 mixing) because C only appears in u_r and not in u_theta; the
# (1/r) d_theta u_r piece is the only one that touches C. The B
# entry has identical bracket structure to its sigma_rr counterpart
# in 1.3.c (``p K_0(p a)/a + 2 K_1(p a)/a^2``) but with prefactor
# 2 instead of the (2 k_z^2 - k_S^2) + ``...`` form -- because the
# Lame reduction is absent here (no lambda). The D entry is new
# at n = 1 and carries the s^2 K_1 piece from differentiating
# the ``D s K_0(s r)`` term, which has no analogue in 1.3.c.

# =====================================================================
# Substep 1.3.e -- solid sigma_rz (cos(theta) sector)
# =====================================================================
#
# Goal: write sigma_rz^{(s)}(a) on the cos(theta) sector as a
# coefficient sum on (B, C, D). Fluid contribution: identically
# zero (inviscid fluid carries no shear); A entry of this row of
# the upcoming 4x4 will be 0.
#
# Decomposition (from 1.3.a, no lambda, pure mu):
#
#     sigma_rz^{(s)} = mu * [ d_z u_r + d_r u_z ]
#
# Inputs from 1.2 (cos(theta) factors stripped; r kept symbolic):
#
#     u_r^{(s)} / cos(theta) = - B p K_0(p r)
#                              - B K_1(p r) / r
#                              + D K_1(s r) / r
#                              - i k_z C K_1(s r)
#
#     u_z^{(s)} / cos(theta) = + i k_z B K_1(p r)
#                              - C s K_0(s r)
#
# Both pieces sit naturally on cos(theta): d_z is i k_z * (.) so it
# preserves the azimuthal factor, and d_r touches r only. No sector
# flip is needed.
#
# Piece (i):  d_z u_r / cos(theta)
# --------------------------------
# Pull down i k_z onto every term of u_r:
#
#     = i k_z * [ - B p K_0(p r) - B K_1(p r) / r
#                 + D K_1(s r) / r - i k_z C K_1(s r) ]
#
# The C term picks up an extra ``i`` factor; combining
# (i k_z) * (- i k_z) = + k_z^2:
#
#     = - i k_z B p K_0(p r)
#       - i k_z B K_1(p r) / r
#       + i k_z D K_1(s r) / r
#       + k_z^2 C K_1(s r)
#
# Piece (ii):  d_r u_z / cos(theta)
# ---------------------------------
# Per-amplitude differentiation:
#
#     d_r [+ i k_z B K_1(p r)] = + i k_z B p K_1'(p r)
#         = i k_z B p * [ - K_0(p r) - K_1(p r) / (p r) ]
#         = - i k_z B p K_0(p r) - i k_z B K_1(p r) / r
#
#     d_r [- C s K_0(s r)] = - C s * s K_0'(s r)
#                          = - C s^2 * (- K_1(s r))
#                          = + C s^2 K_1(s r)
#         (uses K_0'(x) = -K_1(x))
#
#     Sum:  - i k_z B p K_0(p r) - i k_z B K_1(p r) / r
#           + C s^2 K_1(s r)
#
# Sum, collected by amplitude
# ---------------------------
# Multiply piece (i) + piece (ii) by mu.
#
# B contributions (each piece contributes one ``- i k_z`` copy on
# the same two Bessel evaluations):
#
#     - i k_z p K_0(p r):  (i) + (ii)  =  - 2 i k_z
#     - i k_z K_1(p r)/r:  (i) + (ii)  =  - 2 i k_z
#
#     ==> B-sum = - 2 i k_z * [ p K_0(p r) + K_1(p r) / r ]
#
# C contributions (one piece each):
#
#     k_z^2  K_1(s r):    + 1 from (i)
#     s^2    K_1(s r):    + 1 from (ii)
#
#     Sum: (k_z^2 + s^2) C K_1(s r). Using the bound-regime
#     definition s^2 = k_z^2 - k_S^2, this equals
#     (2 k_z^2 - k_S^2) C K_1(s r). The same combination
#     ``2 k_z^2 - k_S^2`` shows up in 1.3.c's sigma_rr B-coefficient,
#     but via a completely different route -- there it came from the
#     Lame reduction ``- lambda k_P^2 + 2 mu p^2``; here it comes
#     from k_z^2 + s^2 directly. The agreement is a structural
#     consistency check on both derivations.
#
#     ==> C-sum = (2 k_z^2 - k_S^2) * K_1(s r)
#
# D contributions (only piece (i)):
#
#     i k_z K_1(s r) / r:  + 1 from (i)
#
#     ==> D-sum = i k_z * K_1(s r) / r
#
# Result (cos(theta) factor reinstated, r = a)
# --------------------------------------------
#
#     sigma_rz^{(s)}(a) / cos(theta) = mu * {
#         - 2 i k_z B [
#             p K_0(p a) + K_1(p a) / a
#         ]
#         + (2 k_z^2 - k_S^2) C K_1(s a)
#         + i k_z D K_1(s a) / a
#     }
#
# Fluid side at the wall:
#
#     sigma_rz^{(f)}(a) = 0     (inviscid fluid, no shear stress)
#
# Sector check
# ------------
# Every term carries cos(theta) (no sin(theta) appears anywhere in
# either piece). This will be row 4 of the 4x4 in 1.4, with A = 0
# in that row from the fluid-side identity above. Both B and D
# entries carry an i k_z factor that the substep-1.5 phase rescaling
# will kill; the C entry is already real (no i factor).
#
# Comparison with n = 0 and 1.3.c
# -------------------------------
# The n = 0 row 3 has B-coefficient ``2 k_z p mu K_1(p a)``
# (post-rescaling; see ``_modal_determinant_n0`` row 3) and
# C-coefficient ``mu * (2 k_z^2 - k_S^2) K_1(s a)``. The C entry
# is structurally identical at n = 1 (still ``mu * (2 k_z^2 - k_S^2)
# K_1(s a)``) because the SV potential ``psi_theta`` has the same
# K_1(s r) shape at both orders. The B entry gains a second term
# at n = 1: the n = 0 form had only ``K_1(p a)`` from
# ``u_r ~ - B p K_1(p r)``, while at n = 1 the K_1-vs-K_0
# splitting from ``u_r ~ - B p K_0(p r) - B K_1(p r)/r`` produces
# the two-term bracket ``[p K_0(p a) + K_1(p a)/a]``. The D entry
# is new at n = 1 (no SH potential at n = 0).
#
# Note: the (2 k_z^2 - k_S^2) combination in the C-coefficient of
# sigma_rz is the *same number* as the (2 k_z^2 - k_S^2) in the
# B-coefficient of sigma_rr (1.3.c), but they arrive there by
# different routes -- the sigma_rr one is from the Lame reduction
# applied to a B K_1(p a) term, the sigma_rz one is from
# ``k_z^2 + s^2`` applied to a C K_1(s a) term. The shared identity
# is one of the cross-checks substep 1.6 will exercise.

# =====================================================================
# Substep 1.3.f -- wall summary at r = a + sector closure check
# =====================================================================
#
# Goal: consolidate the per-amplitude coefficients from 1.2 (u_r at
# the wall) and 1.3.c-e (the three solid stress components) into one
# table that 1.4 can strip the azimuthal factors from and 1.7 can
# transcribe row-by-row. Document all imaginary entries explicitly
# so the substep-1.5 phase rescaling has a complete inventory.
#
# Boundary conditions (4) -> rows of the upcoming 4x4
# ---------------------------------------------------
# Each BC is enforced at r = a; each lives on exactly one azimuthal
# sector. Sector tags are tracked so 1.4's strip-the-azimuthal-factor
# step is mechanical:
#
#     Row 1 (BC1):  u_r^{(f)}(a) = u_r^{(s)}(a)         (cos sector)
#     Row 2 (BC2):  sigma_rr^{(s)}(a) = - P(a)          (cos sector)
#     Row 3 (BC3):  sigma_r_theta^{(s)}(a) = 0          (sin sector)
#     Row 4 (BC4):  sigma_rz^{(s)}(a) = 0               (cos sector)
#
# The fluid carries no shear, so the fluid contributions to BC3 and
# BC4 are identically zero, which gives those two rows ``A = 0``
# entries automatically.
#
# Pre-rescaling 4x4 coefficient table M_pre[BC, amplitude]
# --------------------------------------------------------
# All entries below are written as the coefficient of the listed
# amplitude in the LHS of the BC, with the BC normalised so each row
# evaluates to zero at the modal root. Common shorthand:
#
#     Fa = F * a,   pa = p * a,   sa = s * a
#     I0 = I_0(Fa), I1 = I_1(Fa)
#     K0p = K_0(pa), K1p = K_1(pa)
#     K0s = K_0(sa), K1s = K_1(sa)
#     kz2_kS2 = 2 * k_z^2 - k_S^2
#
# Row 1 (BC1, cos sector, u_r^{(f)} - u_r^{(s)} = 0):
#
#     M_pre[1, A] = (F * I0 - I1 / a) / (rho_f * omega^2)
#     M_pre[1, B] = p * K0p + K1p / a
#     M_pre[1, C] = i * k_z * K1s
#     M_pre[1, D] = - K1s / a
#
# Row 2 (BC2, cos sector, sigma_rr^{(s)} + P = 0):
#
#     M_pre[2, A] = I1
#     M_pre[2, B] = mu * [ kz2_kS2 * K1p
#                          + 2 * p * K0p / a
#                          + 4 * K1p / a^2 ]
#     M_pre[2, C] = 2 * i * k_z * mu * [ s * K0s + K1s / a ]
#     M_pre[2, D] = - 2 * mu * [ s * K0s / a + 2 * K1s / a^2 ]
#
# Row 3 (BC3, sin sector, sigma_r_theta^{(s)} = 0):
#
#     M_pre[3, A] = 0
#     M_pre[3, B] = 2 * mu * [ p * K0p / a + 2 * K1p / a^2 ]
#     M_pre[3, C] = i * k_z * mu * K1s / a
#     M_pre[3, D] = - mu * [ s^2 * K1s
#                            + 2 * s * K0s / a
#                            + 4 * K1s / a^2 ]
#
# Row 4 (BC4, cos sector, sigma_rz^{(s)} = 0):
#
#     M_pre[4, A] = 0
#     M_pre[4, B] = - 2 * i * k_z * mu * [ p * K0p + K1p / a ]
#     M_pre[4, C] = mu * kz2_kS2 * K1s
#     M_pre[4, D] = i * k_z * mu * K1s / a
#
# Sector closure check across all four rows
# -----------------------------------------
# Rows 1, 2, 4: cos(theta) sector. Strip cos(theta) from each side
# of the BC; the four amplitudes (A, B, C, D) survive.
# Row 3: sin(theta) sector. Strip sin(theta); the same four
# amplitudes survive but A is identically zero from the fluid-side
# fact ``sigma_r_theta^{(f)} = 0``.
#
# No row mixes sectors. No amplitude appears with an unmatched
# azimuthal factor. The 4x4 system on (A, B, C, D) closes exactly,
# matching the substep-1.1 prediction.
#
# Imaginary-entry inventory (for substep 1.5)
# -------------------------------------------
# Five entries carry an explicit ``i`` factor (each from an i k_z
# z-derivative of u_r or u_z, never from the Bessel-recurrence
# algebra):
#
#     M_pre[1, C]   = + i * k_z * (...)
#     M_pre[2, C]   = + 2 i * k_z * mu * (...)
#     M_pre[3, C]   = + i * k_z * mu * K1s / a
#     M_pre[4, B]   = - 2 i * k_z * mu * (...)
#     M_pre[4, D]   = + i * k_z * mu * K1s / a
#
# All other 11 entries are real. The rescaling target in 1.5 is to
# absorb the ``i`` factors via row/column multiplications whose
# product is 1 (so ``det M_1`` is invariant); the per-entry pattern
# above is what 1.5 will engineer against. A spoiler for 1.5:
# multiplying *row 4* by i and *column C* by (-i) (overall factor
# i * (-i) = 1) leaves det unchanged and -- by inspection of the
# pattern above -- produces a fully real matrix. Verification of
# that, plus a comparison with the n = 0 ``row 3 by i, column 3 by
# (-i)`` rescaling, is 1.5's job.
#
# Hand-off to substep 1.4
# -----------------------
# 1.4 takes M_pre as written and:
#
#   1. Strips ``cos(theta)`` and ``sin(theta)`` from each row's BC
#      (mechanical; the table above already shows what survives).
#   2. Confirms that the four BCs are independent (rank check before
#      the determinant search starts).
#   3. Documents the n=0-style row-2 sign convention (multiplied by
#      -1 vs the natural form) to keep the n=0 and n=1 code visually
#      consistent.
#
# The matrix entries do not change between 1.4 and 1.7; only the
# substep-1.5 phase rescaling and the 1.5/1.7 cosmetic sign
# adjustments touch them.

# =====================================================================
# Substep 1.4 -- BC application and azimuthal-factor strip
# =====================================================================
#
# Goal: turn the four wall equations from 1.3.f into a 4x4 linear
# system on (A, B, C, D) with no theta dependence, confirm the
# system is generically non-degenerate, and document the row-2 sign
# convention that 1.7 will adopt to match the n=0 code visually.
#
# Azimuthal-factor strip (Fourier orthogonality)
# ----------------------------------------------
# Each of the four BCs is enforced at every point on the borehole
# wall ``r = a``, i.e. for all ``theta in [0, 2 pi)``. Each LHS in
# 1.3.f has the form
#
#     [ coefficient bracket on (A, B, C, D) ] * cos(theta)         (BC1, BC2, BC4)
#     [ coefficient bracket on (A, B, C, D) ] * sin(theta)         (BC3)
#
# Since ``{1, cos(theta), sin(theta)}`` are pairwise orthogonal under
# the L^2(0, 2 pi) inner product, requiring the LHS to vanish for
# every theta forces each bracket to vanish independently. The four
# scalar equations are exactly the rows of the M_pre table from
# 1.3.f; "stripping cos / sin" is the explicit name for taking each
# bracket as a self-contained linear equation in (A, B, C, D).
#
# After the strip, the modal equation is
#
#     M_pre(omega, k_z) @ (A, B, C, D)^T = 0,
#
# with M_pre real-valued except on the five entries flagged in
# 1.3.f's imaginary-entry inventory. The dispersion curve k_z(omega)
# is the locus where ``det M_pre = 0`` (equivalently, where the
# system has a non-trivial null vector).
#
# Block structure
# ---------------
# M_pre has a clean 2x4 / 2x3 block split because the fluid does not
# carry shear stress:
#
#                     |  A     B     C     D  |
#               BC1   |  *     *     *     *  |   <-- cos sector
#               BC2   |  *     *     *     *  |   <-- cos sector
#               BC3   |  0     *     *     *  |   <-- sin sector
#               BC4   |  0     *     *     *  |   <-- cos sector
#
# Rows 3 and 4 live entirely in the (B, C, D) sub-block; only rows
# 1 and 2 couple the fluid amplitude A to the solid amplitudes.
# Expanding ``det M_pre`` along the A column gives the structurally
# meaningful identity
#
#     det M_pre = M_pre[1, A] * det(rows={2,3,4}, cols={B,C,D})
#               - M_pre[2, A] * det(rows={1,3,4}, cols={B,C,D}),
#
# i.e. the dispersion equation is a difference of two 3x3 minors
# weighted by the two non-zero fluid-side entries. The signs and
# magnitudes of those minors govern the low-f and high-f limits
# 1.6 will check.
#
# Rank / generic non-degeneracy check
# -----------------------------------
# For the dispersion equation ``det M_pre = 0`` to define a curve
# (rather than to be satisfied identically), M_pre must have full
# rank 4 generically. Two structural observations confirm this:
#
# 1. **No identically-zero column.** Every column has at least one
#    entry that is a positive multiple of an evaluated Bessel
#    function (and Bessels have isolated zeros at most). The A
#    column has a non-zero entry on rows 1 and 2 (the I_0 / I_1
#    forms), so it is not the zero vector. Columns B, C, D each
#    have non-zero entries on all four rows, so likewise.
#
# 2. **No identically-zero row.** Each of the four rows has at
#    least one non-vanishing entry on the (B, C, D) sub-columns.
#
# Together these rule out the two trivial ways the determinant
# could vanish identically. Linear dependence among rows or among
# columns at specific (omega, k_z) is exactly what defines the
# dispersion curve; the substep-1.6 limit checks confirm that
# those special points reduce to the expected analytical forms
# (low-f -> ``k_z = omega / V_S``; high-f -> Scholte / Rayleigh).
#
# Row-2 sign convention (matches the n=0 code)
# --------------------------------------------
# The natural form of BC2 is ``sigma_rr^{(s)}(a) + P(a) = 0`` with
# coefficients as written in 1.3.f. The existing n=0 code
# (``_modal_determinant_n0`` row 2 in the docstring) multiplies the
# row by -1 to write it as ``-sigma_rr^{(s)} - P = 0``, which puts
# the negative sign on every entry. This is purely cosmetic --
# multiplying a row of M by -1 multiplies ``det M`` by -1, leaving
# the locus ``det M = 0`` unchanged.
#
# Substep 1.7's transcription should adopt the same convention so
# the n=0 and n=1 code blocks are visually parallel. The signs
# come out as:
#
#     M[2, A] = - I1
#     M[2, B] = - mu * [ kz2_kS2 * K1p + 2 p K0p / a + 4 K1p / a^2 ]
#     M[2, C] = - 2 i k_z mu * [ s K0s + K1s / a ]
#     M[2, D] = + 2 mu * [ s K0s / a + 2 K1s / a^2 ]
#
# (each entry has its 1.3.f form negated).
#
# Hand-off to substep 1.5
# -----------------------
# 1.4 leaves M_pre as a 4x4 real matrix with five complex entries
# explicitly flagged. Substep 1.5 will absorb the ``i`` factors via
# a row/column rescaling whose product is 1, leaving ``det M_pre``
# numerically invariant but ensuring every entry of the rescaled
# matrix is real. The spoiler from 1.3.f -- "row 4 by i, column C
# by (-i)" -- is the rescaling that 1.5 will verify and apply.

# =====================================================================
# Substep 1.5 -- phase rescaling to a fully real M_1
# =====================================================================
#
# Goal: produce a real-valued 4x4 ``M_1(omega, k_z)`` such that
# ``det M_1 = det M_pre``, so a real-valued root finder operating on
# ``det M_1`` recovers exactly the dispersion curve of M_pre. The
# rescaling required is small (one row, one column) and the proof
# of det invariance is two lines.
#
# The five imaginary entries (recap from 1.3.f, with the row-2 sign
# convention from 1.4 already applied):
#
#     M_pre[1, C] = + i k_z K1s
#     M_pre[2, C] = - 2 i k_z mu [s K0s + K1s / a]
#     M_pre[3, C] = + i k_z mu K1s / a
#     M_pre[4, B] = - 2 i k_z mu [p K0p + K1p / a]
#     M_pre[4, D] = + i k_z mu K1s / a
#
# Each is a real coefficient times an explicit ``i`` factor; nothing
# else in M_pre carries an imaginary part.
#
# Rescaling
# ---------
# Apply the two operations in either order (they commute on the
# entries that aren't simultaneously in row 4 *and* column C):
#
#     Step 1.  Multiply *row 4* by ``i``.
#     Step 2.  Multiply *column C* by ``-i``.
#
# Determinant scaling factor:
#
#     row scale * column scale = i * (-i) = - i^2 = +1.
#
# So ``det M_rescaled = det M_pre``; the locus
# ``det M_rescaled = 0`` is identical to the dispersion curve.
#
# Per-entry verification
# ----------------------
# 16 entries; group them by which rescalings they receive.
#
# A. **Untouched** (rows 1-3, columns A, B, D; nine entries):
#    no factor applied. Each was real in M_pre and stays real.
#    Specifically:
#
#         [1, A], [1, B], [1, D]
#         [2, A], [2, B], [2, D]
#         [3, A], [3, B], [3, D]
#
# B. **Column C only** (rows 1-3, column C; three entries):
#    factor (-i) applied. Each was ``+ i * (real)`` in M_pre, so
#    the product is ``+ i * (real) * (-i) = + (real)`` -- real.
#
#         [1, C] = i k_z K1s              -->  k_z K1s
#         [2, C] = - 2 i k_z mu [...]     -->  - 2 k_z mu [s K0s + K1s/a]
#         [3, C] = i k_z mu K1s / a       -->  k_z mu K1s / a
#
# C. **Row 4 only** (row 4, columns A, B, D; three entries):
#    factor (i) applied. The A entry is zero (unchanged). The B
#    and D entries were imaginary in M_pre and become real:
#
#         [4, A] = 0                      -->  0  (zero is zero)
#         [4, B] = - 2 i k_z mu [...]     -->  + 2 k_z mu [p K0p + K1p/a]
#         [4, D] = + i k_z mu K1s / a     -->  - k_z mu K1s / a
#
#    (verifications: ``i * (-2 i) = -2 i^2 = +2``;
#                    ``i * (+ i) = i^2 = -1``)
#
# D. **Both row 4 and column C** (the [4, C] corner; one entry):
#    factor i * (-i) = 1 applied. Originally ``mu kz2_kS2 K1s``
#    (real in M_pre), stays exactly that. The corner entry is
#    explicitly the only one whose two rescaling factors cancel,
#    which is *also* the reason the rescaling works at all -- if
#    [4, C] had been imaginary in M_pre, no row-4 + col-C rescaling
#    could clear it without unbalancing det.
#
#         [4, C] = mu kz2_kS2 K1s         -->  mu kz2_kS2 K1s
#
# Total count: 9 (A) + 3 (B) + 3 (C) + 1 (D) = 16 entries; covers
# every cell of M_1.
#
# Final form of M_1 (all entries real)
# ------------------------------------
# With the 1.4 row-2 negation and the 1.5 row-4 / column-C
# rescaling both applied:
#
#     M_1[1, A] = (F I0 - I1 / a) / (rho_f omega^2)
#     M_1[1, B] = p K0p + K1p / a
#     M_1[1, C] = k_z K1s
#     M_1[1, D] = - K1s / a
#
#     M_1[2, A] = - I1
#     M_1[2, B] = - mu * [ kz2_kS2 K1p + 2 p K0p / a + 4 K1p / a^2 ]
#     M_1[2, C] = - 2 k_z mu * [ s K0s + K1s / a ]
#     M_1[2, D] = + 2 mu * [ s K0s / a + 2 K1s / a^2 ]
#
#     M_1[3, A] = 0
#     M_1[3, B] = 2 mu * [ p K0p / a + 2 K1p / a^2 ]
#     M_1[3, C] = k_z mu K1s / a
#     M_1[3, D] = - mu * [ s^2 K1s + 2 s K0s / a + 4 K1s / a^2 ]
#
#     M_1[4, A] = 0
#     M_1[4, B] = + 2 k_z mu * [ p K0p + K1p / a ]
#     M_1[4, C] = mu * kz2_kS2 K1s
#     M_1[4, D] = - k_z mu K1s / a
#
# This is the form 1.7 transcribes; the entries above are exactly
# what ``_modal_determinant_n1`` will assemble.
#
# Comparison with the n = 0 row-3 / column-3 rescaling
# ----------------------------------------------------
# The n = 0 code multiplies "row 3 by i, column 3 by -i" (see
# ``_modal_determinant_n0`` docstring). The row index moves
# 3 -> 4 at n = 1 only because the dipole problem has the extra
# sigma_r_theta = 0 BC inserted above sigma_rz = 0; the column
# index 3 -> "C" is unchanged (always the SV potential amplitude).
# Both rescalings are the same operation -- "rescale the sigma_rz
# row by i, the C column by -i" -- expressed against different
# row-numbering conventions. Net rescaling factor i * (-i) = +1
# in both cases. The structural parallel is what lets 1.7's n=1
# transcription mirror the n=0 implementation almost
# line-for-line.
#
# Hand-off to substep 1.6
# -----------------------
# M_1 is now a real 4x4 polynomial-in-Bessels. Substep 1.6 will
# substitute the low-frequency expansion ``omega a / V_S << 1`` and
# the high-frequency expansion ``omega a / V_S >> 1`` into M_1 and
# verify the leading roots reduce to ``k_z = omega / V_S`` and
# ``k_z = omega / V_R`` respectively. Those are the only checks
# possible without numerical evaluation; the published-curve match
# (Paillet & Cheng 1991 fig. 4.5) waits for substep 1.7 + 1.8 +
# Step 4's validation tests.

# =====================================================================
# Substep 1.6.a -- small-x Bessel asymptotics + low-f entry table
# =====================================================================
#
# Goal: pin the small-argument forms of the four modified Bessel
# functions used in M_1, substitute them into each of the 16
# entries, and tabulate the leading + subleading behavior. Substep
# 1.6.b uses the table to identify the dominant balance that
# defines the flexural-mode low-f asymptote.
#
# Bessel functions in the small-argument limit (x -> 0+)
# ------------------------------------------------------
# Standard expansions (Abramowitz & Stegun 9.6, NIST DLMF 10.30):
#
#     I_0(x) = 1 + x^2 / 4 + O(x^4)
#     I_1(x) = x / 2 + x^3 / 16 + O(x^5)
#
#     K_0(x) = - ln(x / 2) - gamma_E + O(x^2 ln x)
#     K_1(x) = 1 / x + (x / 2) [ ln(x / 2) + gamma_E - 1/2 ]
#                    + O(x^3 ln x)
#
# where gamma_E = 0.5772... is the Euler-Mascheroni constant.
#
# Key observation: the I-Bessels are regular at the origin, but
# K_0 is logarithmically divergent and K_1 is algebraically
# divergent (1/x). The flexural mode at low f sits at
# ``k_z ~ omega / V_S`` so ``s = sqrt(k_z^2 - k_S^2) -> 0`` faster
# than F or p; this is what makes ``K_0(sa)`` and ``K_1(sa)`` the
# most strongly divergent objects in the small-x limit and what
# drives the dominant-balance argument in 1.6.b.
#
# Entry-by-entry leading-order substitution
# -----------------------------------------
# Substitute the small-x forms directly into each of the 16 entries
# of M_1 (from substep 1.5). "Leading" means the most strongly
# divergent term in (Fa, pa, sa); "subleading" lists the next-order
# correction when it's qualitatively different (logarithmic
# corrections from K_0, etc.). All entries below are quoted
# verbatim from M_1; only the Bessel evaluations are replaced.
#
# Row 1 (BC1, u_r continuity)
#
#     [1, A] = (F I0 - I1 / a) / (rho_f omega^2)
#       I0 ~ 1, I1 ~ Fa/2 ==> F I0 - I1 / a ~ F - F / 2 = F / 2
#       LEADING:  F / (2 rho_f omega^2)        (cancellation by 1/2)
#
#     [1, B] = p K0p + K1p / a
#       K0p ~ -ln(pa/2) - gamma_E,  K1p ~ 1 / (pa)
#       LEADING:  1 / (p a^2)                 (from K1p / a)
#       SUBLEADING:  -p ln(pa/2) - p gamma_E  (from p K0p)
#
#     [1, C] = k_z K1s
#       K1s ~ 1 / (sa)
#       LEADING:  k_z / (s a)
#
#     [1, D] = - K1s / a
#       LEADING:  - 1 / (s a^2)
#
# Row 2 (BC2, sigma_rr balance, with the 1.4 row-2 sign convention)
#
#     [2, A] = - I1
#       I1 ~ Fa / 2
#       LEADING:  - F a / 2                   (regular, vanishes as omega -> 0)
#
#     [2, B] = - mu [ kz2_kS2 K1p + 2 p K0p / a + 4 K1p / a^2 ]
#       K1p ~ 1 / (pa), K0p ~ -ln(pa/2) - gamma_E
#       (a) kz2_kS2 K1p     ~ kz2_kS2 / (pa)        ~  kz2_kS2 / (p a)
#       (b) 2 p K0p / a     ~ -2 p ln(pa/2) / a     ~  log-correction
#       (c) 4 K1p / a^2     ~ 4 / (pa) / a^2         ~  4 / (p a^3)
#       LEADING:  - 4 mu / (p a^3)            (from (c))
#       SUBLEADING:  -mu kz2_kS2 / (p a)      (from (a); important when k_z >> omega/V_S)
#
#     [2, C] = - 2 k_z mu [ s K0s + K1s / a ]
#       K0s ~ -ln(sa/2) - gamma_E,  K1s ~ 1 / (sa)
#       LEADING:  - 2 k_z mu / (s a^2)        (from K1s / a)
#       SUBLEADING:  + 2 k_z mu s ln(sa/2)    (from s K0s)
#
#     [2, D] = + 2 mu [ s K0s / a + 2 K1s / a^2 ]
#       LEADING:  + 4 mu / (s a^3)            (from K1s / a^2)
#       SUBLEADING:  - 2 mu s ln(sa/2) / a    (from s K0s / a)
#
# Row 3 (BC3, sigma_r_theta = 0)
#
#     [3, A] = 0
#
#     [3, B] = 2 mu [ p K0p / a + 2 K1p / a^2 ]
#       LEADING:  + 4 mu / (p a^3)            (from K1p / a^2)
#       SUBLEADING:  - 2 mu p ln(pa/2) / a    (from p K0p / a)
#
#     [3, C] = k_z mu K1s / a
#       LEADING:  k_z mu / (s a^2)
#
#     [3, D] = - mu [ s^2 K1s + 2 s K0s / a + 4 K1s / a^2 ]
#       (a) s^2 K1s       ~ s^2 / (sa) = s / a
#       (b) 2 s K0s / a   ~ -2 s ln(sa/2) / a    log-correction
#       (c) 4 K1s / a^2   ~ 4 / (sa) / a^2 = 4 / (s a^3)
#       LEADING:  - 4 mu / (s a^3)            (from (c))
#       SUBLEADING:  - mu s / a               (from (a); regular at sa = 0)
#
# Row 4 (BC4, sigma_rz = 0)
#
#     [4, A] = 0
#
#     [4, B] = + 2 k_z mu [ p K0p + K1p / a ]
#       LEADING:  + 2 k_z mu / (p a^2)        (from K1p / a)
#       SUBLEADING:  - 2 k_z mu p ln(pa/2)    (from p K0p)
#
#     [4, C] = mu kz2_kS2 K1s
#       LEADING:  mu kz2_kS2 / (s a)
#
#     [4, D] = - k_z mu K1s / a
#       LEADING:  - k_z mu / (s a^2)
#
# Cross-check on the divergence pattern
# -------------------------------------
# Among the entries above, the most strongly divergent ones (those
# scaling as ``1 / (s a^3)`` or ``1 / (p a^3)``) are:
#
#     [2, B] ~ -4 mu / (p a^3)      (P-divergent)
#     [2, D] ~ +4 mu / (s a^3)      (S-divergent)
#     [3, B] ~ +4 mu / (p a^3)      (P-divergent)
#     [3, D] ~ -4 mu / (s a^3)      (S-divergent)
#
# Notice the sign flip between rows 2 and 3 on each column, which
# anticipates a determinant cancellation: at the leading divergent
# order, the rows-2-and-3 sub-block contributes
# ``[+4 mu / (p a^3)] [-4 mu / (s a^3)] - [-4 mu / (p a^3)]
# [+4 mu / (s a^3)]`` to the (B, D) 2x2 minor, which is *zero* --
# the leading divergence cancels exactly, leaving the next-order
# terms to govern the dispersion equation.
#
# This is the cleanest hint that the low-f flexural root sits at
# the *subleading* balance, not the leading one. Substep 1.6.b
# turns this observation into a dominant-balance argument that
# locks down ``k_z = omega / V_S`` as the asymptote.
#
# References
# ----------
# * Abramowitz, M., & Stegun, I. A. (1964). *Handbook of
#   Mathematical Functions*. Dover. Sect. 9.6 (small-argument
#   modified Bessel asymptotics).
# * NIST Digital Library of Mathematical Functions, sect. 10.30
#   (online: https://dlmf.nist.gov/10.30).

# =====================================================================
# Substep 1.6.b -- low-f dominant balance: confirm k_z = omega / V_S
# =====================================================================
#
# Goal: take the leading-order cancellation surfaced in 1.6.a and
# argue that the *subleading* balance forces ``k_z -> omega / V_S``
# (equivalently ``s -> 0``) as ``omega a / V_S -> 0``. Scope is
# structural consistency with the published Ellefsen-Cheng-Toksoz
# (1991) result, not a from-scratch perturbation derivation -- the
# latter takes several pages and lives in the cited reference.
#
# Published result (target asymptote)
# -----------------------------------
# Ellefsen, Cheng & Toksoz (1991), sect. III.B, derive the
# long-wavelength flexural-mode limit by perturbation expansion
# of the cylindrical n = 1 modal determinant about ``omega = 0``.
# The leading-order phase slowness is
#
#     s_low = 1 / V_S          (isotropic formation),
#
# i.e. ``k_z(omega) -> omega / V_S`` as ``omega -> 0``. For VTI
# formations the same expansion gives ``s_low = 1 / V_Sv`` (the
# vertical shear slowness; see ``flexural_dispersion_vti_physical``
# in ``fwap.cylindrical`` for the phenomenological VTI version
# anchored on the same limit). The isotropic case is what M_1
# implements and what 1.6.b checks.
#
# Why direct evaluation at ``s = 0`` is singular
# ----------------------------------------------
# Setting ``s = 0`` (equivalently ``sa = 0``) sends ``K_0(sa)``
# and ``K_1(sa)`` to infinity, so ``M_1`` as written has divergent
# entries in column C and the C-derived parts of column D. The
# correct interpretation is "the dispersion locus passes through
# the ``s = 0`` limit as ``omega a / V_S -> 0``", i.e. the *root*
# of ``det M_1 = 0`` approaches the singular point along a
# specific direction in (omega, k_z) space rather than the matrix
# being evaluable there.
#
# Useful regularised limits at small ``sa``
# -----------------------------------------
# Several products that appear in M_1 have finite limits even as
# ``sa -> 0``. Combining the small-x asymptotics from 1.6.a:
#
#     s K_1(sa)   ~ s * 1/(sa) = 1/a               (finite)
#     s^2 K_1(sa) ~ s^2 * 1/(sa) = s/a -> 0        (vanishing)
#     s K_0(sa)   ~ s * (- ln(sa/2)) -> 0          (slower than s)
#     K_1(sa)/a   ~ 1/(sa^2)                       (divergent)
#     K_1(sa)/a^2 ~ 1/(sa^3)                       (more divergent)
#
# These let us trace which entries stay finite vs which carry the
# divergence. In particular, the [2, C] and [3, C] entries
# diverge as 1/(sa^2) and the [2, D], [3, D] entries diverge as
# 1/(sa^3); but the *combinations* that appear in the (B, C, D)
# 3x3 minors of det M_1 (1.4 block-structure observation) admit
# row-and-column factorings that pull the divergence outside,
# leaving a regular (B, C, D) sub-determinant.
#
# Subleading-balance argument
# ---------------------------
# 1.6.a flagged that the leading 1/(s a^3) divergences in
# [2, B], [2, D], [3, B], [3, D] cancel exactly on the (B, D) 2x2
# minor. The next-order contributions to that minor come from the
# subleading terms tabulated in 1.6.a:
#
#     [2, C]_sub = + 2 k_z mu s ln(sa/2)              (from s K_0(sa))
#     [3, D]_sub = - mu s / a                         (from s^2 K_1(sa))
#     [2, D]_sub = - 2 mu s ln(sa/2) / a              (from s K_0(sa) / a)
#     [3, B]_sub = - 2 mu p ln(pa/2) / a              (from p K_0(pa) / a)
#
# Combined with the leading parts that survive on rows 1 and 4,
# the dispersion equation ``det M_1 = 0`` becomes a balance
# between *finite* (s, p, log) terms and *vanishing* (s -> 0)
# terms. The only way to satisfy that balance asymptotically is
# to drive the vanishing-term coefficients to dominate the finite
# coefficients, which forces ``s -> 0``.
#
# Equivalently: in the regularised (B, D) minor, the dispersion
# root corresponds to a zero of a function that has a simple
# analytic factor of ``s`` at leading order in (omega a). The
# zero of that factor is at ``s = 0``, recovering ``k_z =
# omega / V_S``.
#
# This sketch is consistent with the EC&T derivation (which goes
# further to compute the higher-order corrections in (omega a /
# V_S)^2). For a full quantitative match between M_1 and EC&T,
# the test is numerical and belongs to substep 1.8 / Step 4.
#
# Connection to existing fwap code
# --------------------------------
# ``fwap.cylindrical.flexural_dispersion_physical(vp, vs, a)``
# already uses ``s_low = 1 / vs`` as its low-f anchor (see
# ``cylindrical.py`` line 176). The modal solver in this module
# replaces the rational-interpolation transition between
# ``s_low`` and the high-f Rayleigh asymptote with the actual
# determinant root, but the low-f anchor is the same value. A
# test in Step 4 will confirm that ``flexural_dispersion`` (the
# 1.7 + Step 2-3 product) returns ``1 / vs`` to within a few
# percent at f = 200 Hz for typical sonic parameters, replicating
# the agreement that ``flexural_dispersion_physical`` already
# enforces by construction.
#
# Honest scope of this comment block
# ----------------------------------
# What this comment establishes:
#
#   1. The published EC&T result is ``k_z -> omega / V_S`` at low
#      f for the isotropic flexural mode.
#   2. M_1 as written cannot be evaluated at ``s = 0`` directly;
#      the limit must be taken along the dispersion locus.
#   3. The 1.6.a divergence cancellation in the (B, D) minor
#      forces the dispersion root to the subleading balance,
#      which is structurally consistent with ``s -> 0``.
#
# What this comment does *not* establish:
#
#   * A first-principles algebraic derivation of the EC&T result
#     from M_1. That requires several pages of perturbation
#     expansion in (omega a / V_S) and is well-trodden in the
#     reference; reproducing it here adds zero value over a
#     pointer to EC&T sect. III.B.
#   * Quantitative verification. That belongs to the numerical
#     tests in 1.8 + Step 4 (``s(200 Hz) approx 1 / vs`` for a
#     fast formation).
#
# References
# ----------
# * Ellefsen, K. J., Cheng, C. H., & Toksoz, M. N. (1991).
#   Effects of anisotropy upon the resonances of normal modes
#   in a borehole. *J. Acoust. Soc. Am.* 89(6), 2597-2616.
#   Section III.B gives the long-wavelength flexural-mode
#   limit ``s_low = 1 / V_Sv``.
# * Sinha, B. K., Norris, A. N., & Chang, S. K. (1994).
#   Borehole flexural modes in anisotropic formations.
#   *Geophysics* 59(7), 1037-1052. Eq. 14 confirms the same
#   ``s_low = 1 / V_Sv`` low-frequency limit on a different
#   (isotropic-and-VTI) starting point.
#
# Status
# ------
# Substep 1.1 (conventions pinned, sin/cos corrected): done.
# Substep 1.2 (displacements from potentials)        : done.
# Substep 1.3.a (Hooke + strains + Lame reduction)   : done.
# Substep 1.3.b (Bessel second-derivative identities): done.
# Substep 1.3.c (sigma_rr with Lame reduction)       : done.
# Substep 1.3.d (sigma_r_theta on sin sector)        : done.
# Substep 1.3.e (sigma_rz on cos sector)             : done.
# Substep 1.3.f (wall summary + sector check)        : done.
# Substep 1.4 (BCs, azimuthal-factor strip)          : done.
# Substep 1.5 (phase-rescale to real entries)        : done.
# Substep 1.6.a (small-x Bessel + low-f entry table) : done.
# Substep 1.6.b (low-f dominant balance)             : done.
# Substep 1.6.c (large-x Bessel + exponential structure): TODO.
# Substep 1.6.d (high-f Rayleigh-secular reduction)  : TODO.
# Substep 1.6.e (cross-consistency + n=0 + hand-off) : TODO.
# Substep 1.7 (_modal_determinant_n1 in code)        : TODO.
# Substep 1.8 (transcription smoke test)             : TODO.
# Then Step 2 (root-finder) and Step 3 (public ``flexural_dispersion``
# API) per the parent implementation plan.
