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
#
# Status
# ------
# Substep 1.1 (conventions pinned, sin/cos corrected): done.
# Substep 1.2 (displacements from potentials)        : done.
# Substep 1.3 (stresses, Lame reduction)             : TODO.
# Substep 1.4 (BCs, azimuthal-factor strip)          : TODO.
# Substep 1.5 (phase-rescale to real entries)        : TODO.
# Substep 1.6 (analytical-limit cross-check)         : TODO.
# Substep 1.7 (_modal_determinant_n1 in code)        : TODO.
# Substep 1.8 (transcription smoke test)             : TODO.
# Then Step 2 (root-finder) and Step 3 (public ``flexural_dispersion``
# API) per the parent implementation plan.
