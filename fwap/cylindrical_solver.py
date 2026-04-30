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
* **Bound-mode + n=0 leaky regimes**: the bound-mode public APIs
  (:func:`stoneley_dispersion`, :func:`flexural_dispersion`)
  cover the regime ``k_z > omega / V_S`` (all radial wavenumbers
  real). The first leaky-mode public API
  (:func:`pseudo_rayleigh_dispersion`) extends this to the n=0
  leaky regime via outgoing-wave Hankel-function boundary
  conditions and complex-:math:`k_z` Mueller iteration. The
  high-frequency leaky-flexural and quadrupole regimes follow
  the same scaffolding and are scheduled as plan items B and E
  in ``docs/plans/cylindrical_biot.md``.
* **n=0 and n=1 monopole/dipole.** The n=2 quadrupole mode
  follows the same approach but with a 4x4 modal matrix derived
  from a three-scalar Helmholtz decomposition. It is plan item D;
  see ``docs/plans/cylindrical_biot.md``.

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
        ``"flexural"`` for the n=1 first root,
        ``"pseudo_rayleigh"`` for the n=0 leaky branch).
    azimuthal_order : int
        Cylindrical mode index (``0`` for monopole, ``1`` for
        dipole).
    freq : ndarray, shape (n_f,)
        Frequencies (Hz).
    slowness : ndarray, shape (n_f,)
        Phase slowness (s/m): ``slowness[i] = Re(k_z(omega[i])) /
        omega[i]``. ``NaN`` at frequencies where the root finder
        failed (typically below the geometric cutoff for guided
        modes, or in the wrong physical regime for the chosen
        solver).
    attenuation_per_meter : ndarray or None, optional
        Spatial attenuation rate ``Im(k_z(omega[i]))`` in 1/m, for
        leaky modes only. ``None`` (default) for purely-bound
        modes (Stoneley, slow-formation flexural) where the
        attenuation is zero by construction. ``NaN`` at the same
        frequencies where ``slowness`` is NaN.
    """

    name: str
    azimuthal_order: int
    freq: np.ndarray
    slowness: np.ndarray
    attenuation_per_meter: np.ndarray | None = None


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
    M11 = F * I1Fa / (rho_f * omega**2)
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

    M = np.array([[M11, M12, M13], [M21, M22, M23], [M31, M32, M33]], dtype=float)
    return float(np.linalg.det(M))


# ---------------------------------------------------------------------
# Stoneley dispersion: track the lowest n=0 root across frequency
# ---------------------------------------------------------------------


def _stoneley_kz_bracket(
    omega: float,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
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
    s_st_lf = np.sqrt(1.0 / vf**2 + rho_f / mu)
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

        def _det(kz, omega=omega):
            return _modal_determinant_n0(kz, omega, vp, vs, rho, vf, rho_f, a)

        kz_lo, kz_hi = _stoneley_kz_bracket(omega, vp, vs, rho, vf, rho_f, a)
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

# =====================================================================
# Substep 1.6.c -- large-x Bessel asymptotics + exponential structure
# =====================================================================
#
# Goal: pin the large-argument forms of the modified Bessel
# functions, tabulate the dominant exponential factor of each of
# the 16 entries of M_1, and show that the exponentials factor
# globally out of det M_1, leaving a planar-Rayleigh-style
# secular equation for substep 1.6.d to reduce.
#
# Bessel functions in the large-argument limit (x -> infinity)
# ------------------------------------------------------------
# Standard expansions (Abramowitz & Stegun 9.7, NIST DLMF 10.40):
#
#     I_0(x) ~ e^x / sqrt(2 pi x) * [ 1 + 1/(8 x) + O(1/x^2) ]
#     I_1(x) ~ e^x / sqrt(2 pi x) * [ 1 - 3/(8 x) + O(1/x^2) ]
#
#     K_0(x) ~ sqrt(pi / (2 x)) * e^{-x}
#                              * [ 1 - 1/(8 x) + O(1/x^2) ]
#     K_1(x) ~ sqrt(pi / (2 x)) * e^{-x}
#                              * [ 1 + 3/(8 x) + O(1/x^2) ]
#
# Note: in the high-f limit Fa, pa, sa all scale linearly with
# ``omega a`` (since F, p, s -> omega/V_R, omega/V_R^2-correction,
# etc., as ``k_z -> omega / V_R``). All three Bessel arguments are
# therefore large together; no parameter sub-asymptote is needed
# to land in the large-x regime.
#
# Per-entry exponential factor
# ----------------------------
# Substituting the leading-order ``e^{+x}`` (I-Bessels) and
# ``e^{-x}`` (K-Bessels) into each entry of M_1 (from substep 1.5)
# gives a column-only pattern: every entry's exponential factor
# is determined solely by which column it sits in.
#
#     Column   Bessel       Exponential factor      Rows where non-zero
#     A        I_0, I_1     e^{+ Fa}                rows 1, 2 only
#     B        K_0, K_1     e^{- pa}                all 4 rows
#     C        K_0, K_1     e^{- sa}                all 4 rows
#     D        K_0, K_1     e^{- sa}                all 4 rows
#
# Per-entry table (matching the 1.6.a low-f table column-for-column;
# the same shorthand Fa, pa, sa applies):
#
#     [1, A] ~ e^{+ Fa}      [1, B] ~ e^{- pa}
#     [1, C] ~ e^{- sa}      [1, D] ~ e^{- sa}
#
#     [2, A] ~ e^{+ Fa}      [2, B] ~ e^{- pa}
#     [2, C] ~ e^{- sa}      [2, D] ~ e^{- sa}
#
#     [3, A] = 0             [3, B] ~ e^{- pa}
#     [3, C] ~ e^{- sa}      [3, D] ~ e^{- sa}
#
#     [4, A] = 0             [4, B] ~ e^{- pa}
#     [4, C] ~ e^{- sa}      [4, D] ~ e^{- sa}
#
# Column-uniformity argument
# --------------------------
# In the Leibniz expansion of det M_1, each term is a product
# ``sign(sigma) * M[1, sigma(1)] * M[2, sigma(2)] * M[3, sigma(3)]
# * M[4, sigma(4)]`` over a permutation sigma of (A, B, C, D).
# Because the exponential factor of each entry depends only on its
# column, the exponential factor of the product is
#
#     e^{eta_A + eta_B + eta_C + eta_D}
#         = e^{Fa - pa - sa - sa}
#         = e^{Fa - pa - 2 sa},
#
# *the same factor for every non-zero permutation*. The 12 of 24
# permutations that put A on row 3 or row 4 are zero (since
# [3, A] = [4, A] = 0); the remaining 12 all carry the same global
# exponential.
#
# Consequence:
#
#     det M_1(omega, k_z) = e^{Fa - pa - 2 sa} * D_red(omega, k_z)
#
# where D_red is built from the algebraic Bessel prefactors
# ``sqrt(2 pi Fa)``, ``sqrt(2 pi / pa)``, etc. plus the M_1
# entry coefficients (F, p, s, k_z, mu, kz2_kS2, ...). D_red has
# no exponential dependence at leading order in 1 / (omega a).
#
# Since e^{Fa - pa - 2 sa} > 0 for all bound-mode parameters, the
# dispersion equation
#
#     det M_1 = 0    <==>    D_red = 0
#
# in the high-f limit. The structural observation is that D_red is
# precisely the planar half-space modal determinant -- substep
# 1.6.d will show the explicit reduction to the Rayleigh secular
# equation that ``rayleigh_speed`` already implements.
#
# Subleading corrections
# ----------------------
# The ``1 + O(1/x)`` correction factors from the I_n / K_n
# expansions become ``1 + O(1 / (omega a))`` corrections to D_red.
# These are responsible for:
#
#   1. Cylindrical-radius corrections to the planar Rayleigh
#      asymptote -- finite a vs the planar half-space limit.
#   2. The Scholte / fluid-loading offset -- the few-percent
#      reduction below the vacuum-loaded Rayleigh speed that
#      ``rayleigh_speed`` returns. The fluid loading enters via
#      the e^{Fa} I-Bessel column A; the size of the offset
#      depends on rho_f / rho_solid and V_f / V_S.
#
# Both corrections are mentioned in the existing
# ``flexural_dispersion_physical`` docstring (see
# ``cylindrical.py`` around line 145, "fluid-loading correction")
# as a noted limitation of the vacuum-loaded asymptote. The full
# modal solver this module implements does include them; the
# Rayleigh asymptote is just the leading-order term.
#
# Hand-off to substep 1.6.d
# -------------------------
# 1.6.d will:
#
#   1. Substitute the per-entry algebraic prefactors (after
#      stripping e^{Fa}, e^{-pa}, e^{-sa}) into D_red.
#   2. Eliminate the column scaling factors by row / column
#      operations (the planar-limit reduction).
#   3. Match the resulting polynomial in (V_R/V_S)^2 and
#      (V_R/V_P)^2 against the Rayleigh secular equation in
#      ``rayleigh_speed`` (cylindrical.py:48).
#
# References
# ----------
# * Abramowitz, M., & Stegun, I. A. (1964). *Handbook of
#   Mathematical Functions*. Dover. Sect. 9.7 (large-argument
#   modified Bessel asymptotics).
# * NIST Digital Library of Mathematical Functions, sect. 10.40
#   (online: https://dlmf.nist.gov/10.40).

# =====================================================================
# Substep 1.6.d -- high-f reduction to the planar Rayleigh secular eq
# =====================================================================
#
# Goal: take ``D_red`` from 1.6.c (the algebraic factor of det M_1
# after stripping the global exponential ``e^{Fa - pa - 2 sa}``) and
# show it reduces, at leading order in ``1 / (omega a)``, to the
# planar Rayleigh secular equation that ``rayleigh_speed``
# (``cylindrical.py`` line 48) already solves. Scope is structural
# correspondence; the full algebraic reduction is several pages
# of textbook material in Schmitt (1988) and Paillet-Cheng (1991).
#
# Target equation (vacuum-loaded planar Rayleigh)
# -----------------------------------------------
# From ``rayleigh_speed`` and Rayleigh (1885) Proc. London Math.
# Soc. 17, 4-11:
#
#     (2 - xi)^2 = 4 * sqrt( (1 - xi * (V_S / V_P)^2) * (1 - xi) )
#
# with ``xi = (V_R / V_S)^2 in (0, 1)``. The unique non-trivial
# root in that interval is ``V_R``, the Rayleigh speed of a
# vacuum-loaded elastic half-space. ``flexural_dispersion_physical``
# uses this same ``V_R`` as its high-frequency anchor (see
# ``cylindrical.py`` line 178).
#
# Algebraic prefactor structure of D_red
# --------------------------------------
# After substituting the leading-order I_n, K_n forms from 1.6.c
# (note: at strict leading order in ``1 / x``, ``I_0(x) approx
# I_1(x) approx e^x / sqrt(2 pi x)`` and ``K_0(x) approx K_1(x)
# approx sqrt(pi / (2x)) e^{-x}``; the ``1 + O(1 / x)`` corrections
# are subleading), each entry of M_1 factorises as
#
#     [i, j] = a_{ij}(omega, k_z) * Bprefactor_j(x_j) * Eprefactor_j(x_j)
#
# where ``a_{ij}`` is the algebraic coefficient, ``Bprefactor_j`` is
# the column-only Bessel prefactor (``1 / sqrt(2 pi Fa)`` for
# column A, ``sqrt(pi / (2 pa))`` for column B,
# ``sqrt(pi / (2 sa))`` for columns C and D), and ``Eprefactor_j``
# is the column-only exponential (1.6.c). The Bessel prefactors
# themselves factor uniformly out of the determinant, so D_red is
# proportional to ``det A_red`` where ``A_red`` is the 4x4 of
# algebraic coefficients ``a_{ij}``.
#
# Algebraic-coefficient table (leading order)
# -------------------------------------------
# Substituting K_0 ~ K_1 ~ E_K(x) into each M_1 entry and reading
# off the algebraic part:
#
#     A_red[1, A] = (F - 1 / a) / (rho_f omega^2)  -> F / (rho_f omega^2)
#     A_red[1, B] = p + 1 / a                       -> p
#     A_red[1, C] = k_z
#     A_red[1, D] = - 1 / a                         -> small (subleading)
#
#     A_red[2, A] = - 1
#     A_red[2, B] = - mu * [ kz2_kS2 + 2 p / a + 4 / a^2 ]  -> - mu kz2_kS2
#     A_red[2, C] = - 2 k_z mu * (s + 1 / a)              -> - 2 k_z mu s
#     A_red[2, D] = + 2 mu * (s / a + 2 / a^2)            -> small (subleading)
#
#     A_red[3, A] = 0
#     A_red[3, B] = 2 mu * (p / a + 2 / a^2)              -> small (subleading)
#     A_red[3, C] = k_z mu / a                            -> small (subleading)
#     A_red[3, D] = - mu * (s^2 + 2 s / a + 4 / a^2)      -> - mu s^2
#
#     A_red[4, A] = 0
#     A_red[4, B] = + 2 k_z mu * (p + 1 / a)              -> 2 k_z mu p
#     A_red[4, C] = mu * kz2_kS2
#     A_red[4, D] = - k_z mu / a                          -> small (subleading)
#
# At strict leading order in ``1 / (omega a)`` (with k_z scaling as
# omega / V_R, so k_z and F, p, s all linear in omega; ``1 / a``
# fixed), the entries marked "small" vanish. The reduced 4x4 is
# block-structured:
#
#     A_red_lead = | F / (rho_f omega^2)   p              k_z          0       |
#                  | -1                    - mu kz2_kS2  - 2 k_z mu s  0       |
#                  | 0                     0             0             - mu s^2 |
#                  | 0                     2 k_z mu p    mu kz2_kS2    0       |
#
# Row 3 has only the [3, D] entry non-zero (= -mu s^2). Expanding
# the determinant along row 3:
#
#     det A_red_lead = (- mu s^2) * (- 1)^{3+4} * det( minor_3D )
#                    = + mu s^2 * det( minor_3D )
#
# where ``minor_3D`` is the 3x3 obtained by deleting row 3 and
# column D from A_red_lead:
#
#     minor_3D = | F / (rho_f omega^2)   p              k_z          |
#                | -1                    - mu kz2_kS2  - 2 k_z mu s |
#                | 0                     2 k_z mu p    mu kz2_kS2   |
#
# This 3x3 is exactly the n=0 axisymmetric modal determinant
# structure (compare to the row-1 / row-2 / row-3 form in
# ``_modal_determinant_n0`` after the column-A entry adjustments
# for the K_1 -> K_0 ansatz change). Its determinant, after
# eliminating row 1 by Gaussian reduction (multiply row 1 by
# rho_f omega^2 / F and row-add to absorb the [2, A] = -1 entry),
# reduces to a 2x2 in (B, C) columns with entries scaling like
# the planar Rayleigh secular equation.
#
# Identification with the Rayleigh equation
# -----------------------------------------
# The 2x2 block on (B, C) columns of the reduced ``minor_3D`` is
# (after dividing each entry by ``mu`` and gathering ``k_z`` and
# ``s, p`` factors):
#
#     | -kz2_kS2   - 2 k_z s |
#     |  2 k_z p     kz2_kS2 |
#
# whose determinant is
#
#     det = - kz2_kS2^2 + 4 k_z^2 p s.
#
# Setting this to zero (the dispersion equation in the high-f
# leading limit) gives
#
#     (2 k_z^2 - k_S^2)^2 = 4 k_z^2 p s.
#
# Substituting the bound-mode definitions
# ``p^2 = k_z^2 - omega^2 / V_P^2`` and
# ``s^2 = k_z^2 - omega^2 / V_S^2`` and parametrising ``k_z = omega
# / V`` for some test phase velocity ``V``:
#
#     k_S^2 = omega^2 / V_S^2,   k_z^2 = omega^2 / V^2,
#     p^2 = omega^2 (1/V^2 - 1/V_P^2),
#     s^2 = omega^2 (1/V^2 - 1/V_S^2).
#
# Defining ``xi = (V / V_S)^2`` and ``a_PS^2 = (V_S / V_P)^2``,
# substituting and simplifying with the Mathematica-grade algebra
# in Schmitt (1988) eq. 24-26 yields
#
#     (2 - xi)^2 = 4 * sqrt( (1 - xi * a_PS^2) * (1 - xi) ),
#
# *exactly* the secular equation in ``rayleigh_speed`` (with
# ``a^2 = (V_S / V_P)^2 = a_PS^2`` and the same ``xi``). The
# solution ``xi = (V_R / V_S)^2`` recovers the Rayleigh speed,
# i.e. ``k_z -> omega / V_R`` at high f. Done.
#
# Subleading correction: the Scholte / fluid-loading offset
# ---------------------------------------------------------
# The "small" entries dropped above (involving ``1 / a``, ``1 / a^2``)
# do *not* drop the column-A entry [1, A] = F / (rho_f omega^2),
# which carries the fluid loading. Re-including them at first
# subleading order ``O(1 / (omega a))`` adds a fluid-density and
# fluid-velocity dependent correction that pulls ``V_R``
# downward by a few percent to the Scholte interface-wave speed.
# This correction is what makes the full modal solver more
# accurate than ``flexural_dispersion_physical`` -- the
# phenomenological function only knows the Rayleigh anchor and
# uses a smoothed-step transition; the modal solver tracks the
# Scholte offset directly.
#
# Honest scope of this comment block
# ----------------------------------
# What this comment establishes:
#
#   1. D_red factorises into a column-Bessel-prefactor block and
#      an algebraic-coefficient block A_red.
#   2. At strict leading order in ``1 / (omega a)``, A_red has a
#      block structure where row 3 contributes only via the [3, D]
#      entry, reducing det A_red to a 3x3 structurally identical
#      to the n=0 axisymmetric modal matrix.
#   3. That 3x3 reduces by row operations to a 2x2 whose vanishing
#      condition is *exactly* the Rayleigh secular equation
#      ``(2 - xi)^2 = 4 sqrt( (1 - xi a^2) (1 - xi) )`` already
#      coded in ``rayleigh_speed``.
#
# What this comment does *not* establish:
#
#   * Algebraic execution of the row-reduction steps from the 3x3
#     to the 2x2. That reduction is in Schmitt (1988) eqs. 24-26
#     and Paillet & Cheng (1991) sect. 4.2; reproducing it here
#     adds zero value over a pointer.
#   * Quantitative reproduction of any specific dispersion-curve
#     value. That belongs to substep 1.8 + Step 4's published-
#     curve match against Paillet & Cheng 1991 fig. 4.5.
#
# References
# ----------
# * Schmitt, D. P. (1988). Shear-wave logging in elastic
#   formations. *J. Acoust. Soc. Am.* 84(6), 2230-2244. Eqs.
#   24-26 give the high-frequency reduction of the dipole modal
#   determinant to the Rayleigh secular equation.
# * Paillet, F. L., & Cheng, C. H. (1991). *Acoustic Waves in
#   Boreholes*, Sect. 4.2 (high-frequency asymptotic forms of
#   the cylindrical-mode dispersion equation).
# * Rayleigh, Lord (1885). On waves propagating along the plane
#   surface of an elastic solid. *Proc. London Math. Soc.* 17,
#   4-11 (the secular equation itself).

# =====================================================================
# Substep 1.6.e -- cross-consistency, n=0 comparison, hand-off to 1.7
# =====================================================================
#
# Goal: three short structural cross-checks that confirm 1.6.a-d
# are internally consistent, then compile a transcription-ready
# reference table for 1.7.
#
# Cross-check 1: the (2 k_z^2 - k_S^2) combination
# ------------------------------------------------
# This combination appears in M_1 in two distinct entries via two
# distinct routes (already noted in 1.3.c and 1.3.e):
#
#   * ``M_1[2, B] = - mu * [ kz2_kS2 K1(pa) + ... ]``
#     -- arrived via the Lame reduction
#     ``- lambda k_P^2 + 2 mu p^2 = mu (2 k_z^2 - k_S^2)``
#     applied to the K_1(p r) part of u_r.
#
#   * ``M_1[4, C] = mu * kz2_kS2 K1(sa)``
#     -- arrived via the bound-regime identity
#     ``s^2 + k_z^2 = 2 k_z^2 - k_S^2``
#     applied to the K_1(s r) part of u_z.
#
# 1.6.d showed that these two entries are exactly the *diagonal*
# entries of the (B, C) 2x2 sub-block whose vanishing condition,
# in the high-f leading limit, is the Rayleigh secular equation
# ``(2 - xi)^2 = 4 sqrt((1 - xi a^2) (1 - xi))``. The off-diagonal
# entries pair them with ``2 k_z s`` (in [2, C]) and ``2 k_z p``
# (in [4, B]), which are exactly the cross-terms of the Rayleigh
# equation. So the two-route derivation of the same number lands
# in the structurally correct places of the high-f sub-block --
# confirming that 1.3.c and 1.3.e were not just numerically
# right but algebraically right.
#
# Cross-check 2: monotonic dispersion between the two limits
# ----------------------------------------------------------
# 1.6.b: low-f asymptote ``s_low = 1 / V_S``, i.e.
#        ``k_z -> omega / V_S``.
# 1.6.d: high-f asymptote ``s_high = 1 / V_R``, with
#        ``V_R < V_S``, so ``k_z -> omega / V_R > omega / V_S``.
#
# Since V_R < V_S strictly (Rayleigh speed is always strictly
# less than the shear speed; see ``rayleigh_speed`` for the
# Poisson-ratio dependence), the slowness ``s(omega) = k_z / omega``
# *increases* monotonically from ``1 / V_S`` at low f to
# ``1 / V_R`` at high f. The dispersion curve is monotonic in
# this slowness sense, which matches:
#
#   * The qualitative shape coded in
#     ``flexural_dispersion_physical(vp, vs, a)``
#     (``cylindrical.py`` line 109) -- a smoothed-step rise
#     from ``s_low`` to ``s_high``.
#   * The fast-formation dipole-flexural curves in Paillet &
#     Cheng (1991) fig. 4.5 and Tang & Cheng (2004) fig. 3.4.
#
# A non-monotonic dispersion would indicate a sign error somewhere
# in the 1.3.c / 1.3.d / 1.3.e / 1.5 chain. The two endpoints
# being on opposite sides of ``1 / V_S`` is the cleanest single
# cross-check that the matrix is built right.
#
# Cross-check 3: structural reduction to the n=0 modal form
# ---------------------------------------------------------
# At n = 0, the sigma_r_theta = 0 BC is trivially satisfied (no
# theta dependence to differentiate), so row 3 of M_1 disappears.
# The SH potential ``psi_z`` is also absent at n = 0, so column D
# disappears. Removing row 3 and column D from M_1 should give a
# 3x3 structurally identical to ``_modal_determinant_n0``.
#
# 1.6.d's ``minor_3D`` is exactly this 3x3 (rows 1, 2, 4 / columns
# A, B, C). At the algebraic level, ``minor_3D`` matches
# ``_modal_determinant_n0`` row-for-row and column-for-column,
# with two notation differences:
#
#   1. **Bessel order swap**: at n = 0 the P-potential ansatz uses
#      ``phi = B K_0(p r)`` (not K_1); at n = 1 it's K_1(p r). So
#      the K_1(pa) entries in ``minor_3D`` correspond to K_0(pa)
#      entries in ``_modal_determinant_n0`` and vice versa. This
#      is a structural difference, not a transcription mistake.
#   2. **No ``+ 4 K_1(p a) / a^2`` term**: the n = 0 derivation
#      doesn't have this 1/r^2 correction in its B-coefficient
#      (it arises at n = 1 from the K_1''(p r) identity, which
#      doesn't apply when the radial dependence is K_0(p r)).
#      See 1.3.c "Comparison with n = 0".
#
# Modulo those two structural changes, ``minor_3D`` is the n = 0
# modal determinant. The fact that the high-f reduction
# (1.6.d) lands on ``minor_3D``, and that ``minor_3D`` is
# structurally the n=0 form, is reassuring: both n = 0 and n = 1
# share the same Rayleigh asymptote at high f, which is
# physically correct because both modes localise within ~one
# wavelength of the borehole wall and propagate as surface waves
# on the fluid-solid interface.
#
# Hand-off table for substep 1.7 (the actual code transcription)
# --------------------------------------------------------------
# 1.7 will write the function ``_modal_determinant_n1(kz, omega,
# vp, vs, rho, vf, rho_f, a)`` mirroring the n = 0 counterpart.
# The body should follow this template, in order:
#
#     1. Compute radial decay constants:
#            F = sqrt(kz^2 - (omega / vf)^2)
#            p = sqrt(kz^2 - (omega / vp)^2)
#            s = sqrt(kz^2 - (omega / vs)^2)
#
#     2. Compute Bessel arguments:
#            Fa, pa, sa = F * a, p * a, s * a
#
#     3. Evaluate Bessels (use existing helpers):
#            I0, I1   = _i0_i1(Fa)
#            K0p, K1p = _k0_k1(pa)
#            K0s, K1s = _k0_k1(sa)
#
#     4. Compute auxiliary scalars:
#            mu      = rho * vs**2
#            kS2     = (omega / vs)**2
#            kz2_kS2 = 2.0 * kz * kz - kS2
#
#     5. Assemble M_1 (16 real entries; the 1.5 + 1.4 forms):
#
#         Row 1 (BC1, u_r continuity, cos sector):
#             M[0, 0] = (F * I0 - I1 / a) / (rho_f * omega**2)
#             M[0, 1] = p * K0p + K1p / a
#             M[0, 2] = kz * K1s
#             M[0, 3] = - K1s / a
#
#         Row 2 (BC2, sigma_rr balance, cos sector, row negated):
#             M[1, 0] = - I1
#             M[1, 1] = - mu * (kz2_kS2 * K1p
#                               + 2.0 * p * K0p / a
#                               + 4.0 * K1p / a**2)
#             M[1, 2] = - 2.0 * kz * mu * (s * K0s + K1s / a)
#             M[1, 3] = + 2.0 * mu * (s * K0s / a + 2.0 * K1s / a**2)
#
#         Row 3 (BC3, sigma_r_theta = 0, sin sector):
#             M[2, 0] = 0.0
#             M[2, 1] = 2.0 * mu * (p * K0p / a + 2.0 * K1p / a**2)
#             M[2, 2] = kz * mu * K1s / a
#             M[2, 3] = - mu * (s * s * K1s
#                               + 2.0 * s * K0s / a
#                               + 4.0 * K1s / a**2)
#
#         Row 4 (BC4, sigma_rz = 0, cos sector, after 1.5 row+col rescale):
#             M[3, 0] = 0.0
#             M[3, 1] = + 2.0 * kz * mu * (p * K0p + K1p / a)
#             M[3, 2] = mu * kz2_kS2 * K1s
#             M[3, 3] = - kz * mu * K1s / a
#
#     6. Return ``float(np.linalg.det(M))``.
#
# Validation (substep 1.8 + Step 4): bound-mode bracket scan
# starting from the existing ``flexural_dispersion_physical``
# anchor ``s_low = 1 / vs`` at f = 200 Hz; published-curve match
# against Paillet & Cheng 1991 fig. 4.5 in Step 4.
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
# Substep 1.6.c (large-x Bessel + exponential structure): done.
# Substep 1.6.d (high-f Rayleigh-secular reduction)  : done.
# Substep 1.6.e (cross-consistency + n=0 + hand-off) : done.
# Substep 1.7 (_modal_determinant_n1 in code)        : done.
# Substep 1.8 (transcription smoke test)             : done.
# Then Step 2 (root-finder) and Step 3 (public ``flexural_dispersion``
# API) per the parent implementation plan.
#
# Substep-1.8 smoke-test results (slow formation, vp=2200, vs=800,
# rho=2200, vf=1500, rho_f=1000, a=0.1):
#
#   f = 2 kHz   : sign change at k_z corresponding to slowness ~1251 us/m
#                 (low-f asymptote 1/V_S = 1250 us/m matched).
#   f = 5 kHz   : sign change at slowness ~1337 us/m
#                 (high-f asymptote 1/V_R = 1322 us/m + ~1.1% Scholte
#                  fluid-loading offset; matches 1.6.c-d predictions).
#   f = 10 kHz  : sign change at slowness ~1338 us/m (asymptote saturated).
#
# Note on the bound-mode regime: the matrix is built assuming
# k_z > omega / V_alpha for every wave speed V_alpha (V_f, V_P, V_S),
# so all radial wavenumbers F, p, s are real. In a *fast* formation
# (V_S > V_f) the flexural mode has phase velocity between V_R and
# V_S, which is above V_f -- the mode is then *narrowly leaky* into
# the fluid (F^2 < 0) and outside the scope of this solver. The
# leaky-flexural regime is on the roadmap as a follow-up to this
# bound-mode solver.


# ---------------------------------------------------------------------
# n = 1 dipole flexural modal determinant (Schmitt 1988)
# ---------------------------------------------------------------------


def _modal_determinant_n1(
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
    4x4 dipole modal determinant in the bound-mode regime.

    Four boundary conditions at ``r = a``:

    1. continuity of radial displacement
       :math:`u_r^{(f)}(a) = u_r^{(s)}(a)` (cos sector),
    2. normal-stress balance
       :math:`\sigma_{rr}^{(s)}(a) = -P^{(f)}(a)` (cos sector;
       row negated to match the n=0 sign convention),
    3. tangential-shear vanishing on the formation side
       :math:`\sigma_{r\theta}^{(s)}(a) = 0` (sin sector;
       fluid-side identically zero),
    4. axial-shear vanishing on the formation side
       :math:`\sigma_{rz}^{(s)}(a) = 0` (cos sector;
       fluid-side identically zero).

    A guided dipole mode at fixed ``omega`` is a value of ``k_z``
    that makes ``det(M) = 0``. Returns a real scalar; after the
    substep-1.5 phase rescaling (row 4 by ``i``, column C by
    ``-i``, net factor 1) every entry of M is real, so the
    determinant is itself real and root-findable with
    :func:`scipy.optimize.brentq`.

    Field representation (bound regime):

    * Fluid pressure:  :math:`P = A I_1(F r) \cos\theta`,
      :math:`F = \sqrt{k_z^2 - \omega^2 / V_f^2}`
    * Formation P scalar potential:
      :math:`\phi = B K_1(p r) \cos\theta`,
      :math:`p = \sqrt{k_z^2 - \omega^2 / V_P^2}`
    * Formation SV vector-potential, theta component:
      :math:`\psi_\theta = C K_1(s r) \cos\theta`,
      :math:`s = \sqrt{k_z^2 - \omega^2 / V_S^2}`
    * Formation SH vector-potential, z component:
      :math:`\psi_z = D K_1(s r) \sin\theta`

    The full derivation -- field-from-potential expansions, stress
    components with the Lame reduction
    :math:`-\lambda k_P^2 + 2\mu p^2 = \mu (2 k_z^2 - k_S^2)`,
    and the row-4 / column-C phase rescaling that makes M real --
    is in the substep blocks above (1.1 through 1.6.e) ending at
    :func:`stoneley_dispersion`.

    The 16 entries below are the substep-1.5 final form; each row's
    derivation is summarised next to the entry. The ``cos(theta)``
    and ``sin(theta)`` azimuthal factors have been stripped (substep
    1.4); after the row-4 / column-C phase rescaling (substep 1.5)
    every entry is real.

    Row 1 (BC1, ``u_r^{(f)} - u_r^{(s)} = 0``, cos sector):
        ``[ (F I_0(Fa) - I_1(Fa) / a) / (rho_f omega^2),
            p K_0(pa) + K_1(pa) / a,
            k_z K_1(sa),
            -K_1(sa) / a ]``

    Row 2 (BC2, ``-(sigma_rr^{(s)} + P) = 0``, cos sector,
    row negated for n=0 visual parallel):
        ``[ -I_1(Fa),
            -mu * [ (2 k_z^2 - k_S^2) K_1(pa)
                    + 2 p K_0(pa) / a
                    + 4 K_1(pa) / a^2 ],
            -2 k_z mu * [ s K_0(sa) + K_1(sa) / a ],
            +2 mu * [ s K_0(sa) / a + 2 K_1(sa) / a^2 ] ]``

    Row 3 (BC3, ``sigma_r_theta^{(s)} = 0``, sin sector):
        ``[ 0,
            2 mu * [ p K_0(pa) / a + 2 K_1(pa) / a^2 ],
            k_z mu K_1(sa) / a,
            -mu * [ s^2 K_1(sa) + 2 s K_0(sa) / a
                    + 4 K_1(sa) / a^2 ] ]``

    Row 4 (BC4, ``sigma_rz^{(s)} = 0``, cos sector,
    after row-4 by ``i`` and column-C by ``-i`` rescale):
        ``[ 0,
            +2 k_z mu * [ p K_0(pa) + K_1(pa) / a ],
            mu * (2 k_z^2 - k_S^2) K_1(sa),
            -k_z mu K_1(sa) / a ]``

    Where ``Fa = F a``, ``pa = p a``, ``sa = s a``, ``mu = rho V_S^2``,
    ``k_S = omega / V_S``.

    See Also
    --------
    _modal_determinant_n0 : The n=0 axisymmetric (Stoneley)
        counterpart. Structurally similar 3x3 form; the n=1
        problem has an extra row (sigma_r_theta = 0) and an
        extra column (the SH potential D) on top.

    References
    ----------
    * Schmitt, D. P. (1988). Shear-wave logging in elastic
      formations. *J. Acoust. Soc. Am.* 84(6), 2230-2244
      (the n=1 dipole modal determinant).
    * Kurkjian, A. L., & Chang, S.-K. (1986). Acoustic multipole
      sources in fluid-filled boreholes. *Geophysics* 51(1),
      148-163 (most explicit derivation of the 4x4 dipole system).
    * Paillet, F. L., & Cheng, C. H. (1991). *Acoustic Waves in
      Boreholes.* CRC Press, Ch. 4.
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

    # Row 1: u_r^{(f)} - u_r^{(s)} = 0 at r = a (cos sector).
    M11 = (F * I0Fa - I1Fa / a) / (rho_f * omega**2)
    M12 = p * K0pa + K1pa / a
    M13 = kz * K1sa
    M14 = -K1sa / a

    # Row 2: -(sigma_rr^{(s)} + P) = 0 at r = a (cos sector;
    # row negated for visual parallel with the n=0 form).
    M21 = -I1Fa
    M22 = -mu * (two_kz2_minus_kS2 * K1pa + 2.0 * p * K0pa / a + 4.0 * K1pa / (a * a))
    M23 = -2.0 * kz * mu * (s * K0sa + K1sa / a)
    M24 = 2.0 * mu * (s * K0sa / a + 2.0 * K1sa / (a * a))

    # Row 3: sigma_r_theta^{(s)} = 0 at r = a (sin sector;
    # fluid carries no shear, so M31 = 0).
    M31 = 0.0
    M32 = 2.0 * mu * (p * K0pa / a + 2.0 * K1pa / (a * a))
    M33 = kz * mu * K1sa / a
    M34 = -mu * (s * s * K1sa + 2.0 * s * K0sa / a + 4.0 * K1sa / (a * a))

    # Row 4: sigma_rz^{(s)} = 0 at r = a (cos sector; M41 = 0
    # for the same fluid-no-shear reason). Entries below are the
    # substep-1.5 form: row 4 has been multiplied by i and
    # column C (= column 3 here) by -i, leaving a real matrix.
    M41 = 0.0
    M42 = 2.0 * kz * mu * (p * K0pa + K1pa / a)
    M43 = mu * two_kz2_minus_kS2 * K1sa
    M44 = -kz * mu * K1sa / a

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


def _modal_determinant_n1_complex(
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
    Complex-``k_z`` n=1 dipole modal determinant with optional
    leaky-wave branches.

    Mirrors the matrix structure of the real-valued
    :func:`_modal_determinant_n1` (see its docstring for the full
    Kurkjian-Chang derivation): four boundary conditions at
    ``r = a`` (continuity of u_r in the cos sector, sigma_rr +
    P = 0 in the cos sector, sigma_r_theta = 0 in the sin sector,
    sigma_rz = 0 in the cos sector), four unknown amplitudes
    (A in the fluid, B / C / D in the solid for the P / SV / SH
    potentials), and the same row-4-by-i / column-C-by-(-i) phase
    rescaling that makes the matrix real in the fully-bound
    regime.

    What's new:

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
      part matches the real-only :func:`_modal_determinant_n1`
      exactly -- a regression invariant tested in
      ``tests/test_cylindrical_solver.py``.

    Parameters
    ----------
    kz : complex
        Axial wavenumber. May be complex.
    omega, vp, vs, rho, vf, rho_f, a : float
        Same as :func:`_modal_determinant_n1`.
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
    _modal_determinant_n1 : The real-valued bound-only counterpart.
        The two functions agree exactly when ``kz`` is real and
        both ``leaky_*`` flags are False.
    _modal_determinant_n0_complex : The n=0 sister (Stoneley +
        pseudo-Rayleigh).
    """
    kz_c = complex(kz)
    F = np.sqrt(kz_c * kz_c - (omega / vf) ** 2)
    p = np.sqrt(kz_c * kz_c - (omega / vp) ** 2)
    s = np.sqrt(kz_c * kz_c - (omega / vs) ** 2)
    Fa = F * a

    # Fluid: I-Bessel always (regular at r=0). scipy.special.iv
    # supports complex arguments transparently; for ``F^2 < 0``
    # (fast-formation flexural regime, F purely imaginary)
    # ``iv`` returns the J-Bessel-equivalent oscillatory pattern
    # with the appropriate i^n phase, and the row/column
    # rescaling below carries through that phase consistently.
    I0Fa = complex(special.iv(0, Fa))
    I1Fa = complex(special.iv(1, Fa))

    # Formation P and S K-Bessel pairs (or Hankel via analytic
    # continuation in the leaky regime).
    K0pa, K1pa = _k_or_hankel(0, p, a, leaky=leaky_p)
    K0sa, K1sa = _k_or_hankel(0, s, a, leaky=leaky_s)

    mu = rho * vs * vs
    kS2 = (omega / vs) ** 2
    two_kz2_minus_kS2 = 2.0 * kz_c * kz_c - kS2

    # Same matrix layout as _modal_determinant_n1; entries are now
    # complex but the structure is identical.

    # Row 1: u_r^{(f)} - u_r^{(s)} = 0 at r = a (cos sector).
    M11 = (F * I0Fa - I1Fa / a) / (rho_f * omega ** 2)
    M12 = p * K0pa + K1pa / a
    M13 = kz_c * K1sa
    M14 = -K1sa / a

    # Row 2: -(sigma_rr^{(s)} + P) = 0 at r = a (cos sector;
    # row negated for visual parallel with the n=0 form).
    M21 = -I1Fa
    M22 = -mu * (
        two_kz2_minus_kS2 * K1pa + 2.0 * p * K0pa / a + 4.0 * K1pa / (a * a)
    )
    M23 = -2.0 * kz_c * mu * (s * K0sa + K1sa / a)
    M24 = 2.0 * mu * (s * K0sa / a + 2.0 * K1sa / (a * a))

    # Row 3: sigma_r_theta^{(s)} = 0 at r = a (sin sector;
    # fluid carries no shear, so M31 = 0).
    M31 = 0.0 + 0j
    M32 = 2.0 * mu * (p * K0pa / a + 2.0 * K1pa / (a * a))
    M33 = kz_c * mu * K1sa / a
    M34 = -mu * (s * s * K1sa + 2.0 * s * K0sa / a + 4.0 * K1sa / (a * a))

    # Row 4: sigma_rz^{(s)} = 0 at r = a (cos sector). Same
    # row-4-by-i / column-C-by-(-i) rescaling as the real
    # version, applied after the K -> Hankel substitution above.
    M41 = 0.0 + 0j
    M42 = 2.0 * kz_c * mu * (p * K0pa + K1pa / a)
    M43 = mu * two_kz2_minus_kS2 * K1sa
    M44 = -kz_c * mu * K1sa / a

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


# ---------------------------------------------------------------------
# Flexural dispersion: track the lowest n=1 root across frequency
# ---------------------------------------------------------------------


def _flexural_kz_bracket(
    omega: float,
    vp: float,
    vs: float,
    rho: float,
    vf: float,
    rho_f: float,
    a: float,
) -> tuple[float, float]:
    """
    Bracket the n=1 dipole flexural root in (k_z_lo, k_z_hi).

    The flexural mode in a slow formation (V_S < V_f) is bound;
    its phase velocity ranges from V_S at low f down to the
    Rayleigh / Scholte speed at high f. Slowness sits between
    1/V_S and slightly above 1/V_R.

    Lower bound: just above omega / V_S (the bound-regime floor;
    s = sqrt(k_z^2 - k_S^2) must be real and positive).

    Upper bound: 10% above omega / V_R, generous enough to capture
    the few-percent Scholte / fluid-loading offset that puts the
    actual root above the vacuum-loaded Rayleigh slowness. The
    caller may expand if no sign change is found in this range.
    """
    # Vacuum-loaded Rayleigh-speed proxy for the high-f asymptote.
    # ``rayleigh_speed`` validates vp > vs internally.
    from fwap.cylindrical import rayleigh_speed

    vR = rayleigh_speed(vp, vs)
    kz_lo = omega / vs * (1.0 + 1.0e-6)
    kz_hi = omega / vR * 1.10
    return kz_lo, kz_hi


def _flexural_dispersion_fast_formation(
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
    Fast-formation (``V_S > V_f``) flexural dispersion.

    Tracks the lowest n=1 root in the slowness window
    ``(1/V_S, 1/V_R)`` -- equivalently, phase velocity in
    ``(V_R, V_S)``. In this regime ``F^2 = k_z^2 - (omega/V_f)^2 < 0``
    so the fluid radial wavenumber is purely imaginary and the
    fluid pressure is oscillatory in ``r``; the formation P and S
    branches stay bound (``p^2, s^2 > 0``), so the converged root
    has ``Im(k_z) = 0`` to floating-point precision and the mode
    is genuinely bound rather than truly leaky.

    The complex modal determinant
    :func:`_modal_determinant_n1_complex` evaluated at real ``k_z``
    is overwhelmingly imaginary in this regime (the real part is
    suppressed by ~10 orders of magnitude relative to the
    imaginary part), so the root condition reduces to
    ``Im(det) = 0`` and :func:`scipy.optimize.brentq` along the
    real ``k_z`` axis is the natural tool. Bracket: ``k_z`` in
    ``(omega/V_S * (1 + eps), omega/V_R * (1 - eps))`` -- strictly
    inside the leaky-F regime to keep ``F^2 < 0`` and to avoid the
    numerical degeneracy at ``k_z = omega/V_S`` (where ``s = 0``).
    """
    from fwap.cylindrical import rayleigh_speed

    f_arr = np.asarray(freq, dtype=float)
    n_f = f_arr.size
    slowness = np.full(n_f, np.nan, dtype=float)
    if n_f == 0:
        return BoreholeMode(
            name="flexural", azimuthal_order=1,
            freq=f_arr, slowness=slowness,
        )

    vR = rayleigh_speed(vp, vs)
    eps = 1.0e-4

    def _im_det(kz: float, _omega: float) -> float:
        return _modal_determinant_n1_complex(
            complex(kz, 0.0), _omega, vp, vs, rho, vf, rho_f, a,
            leaky_p=False, leaky_s=False,
        ).imag

    def _find_root_in_bracket(
        kz_lo: float, kz_hi: float, omega: float,
    ) -> float | None:
        """brentq on Im(det) within the given bracket; returns
        ``None`` if the bracket has no sign change or evaluation
        fails."""
        try:
            d_lo = _im_det(kz_lo, omega)
            d_hi = _im_det(kz_hi, omega)
            if not (np.isfinite(d_lo) and np.isfinite(d_hi)):
                return None
            if np.sign(d_lo) == np.sign(d_hi):
                return None
            return float(optimize.brentq(
                _im_det, kz_lo, kz_hi, args=(omega,), xtol=1.0e-10,
            ))
        except (ValueError, RuntimeError):
            return None

    # Walk high to low frequency. At each step, try a narrow
    # bracket centred on the previous step's slowness first; if
    # that fails, fall back to the wide ``(1/V_S, 1/V_R)`` bracket.
    # This continuation strategy keeps the marcher on the same
    # physical branch even when the determinant has multiple
    # competing roots in the wide bracket.
    order_desc = np.argsort(-f_arr)
    f_desc = f_arr[order_desc]
    slowness_desc = np.full(f_desc.size, np.nan, dtype=float)
    slowness_prev: float | None = None

    for i, f in enumerate(f_desc):
        omega = 2.0 * np.pi * float(f)
        kz_root: float | None = None

        # Continuation bracket: previous slowness +- 2 % (a wide
        # neighbourhood since fast-flexural slowness is nearly flat
        # vs frequency, ~ V_R asymptote).
        if slowness_prev is not None:
            kz_centre = slowness_prev * omega
            kz_lo = max(kz_centre * 0.98, omega / vs * (1.0 + eps))
            kz_hi = min(kz_centre * 1.02, omega / vR * (1.0 - eps))
            if kz_hi > kz_lo:
                kz_root = _find_root_in_bracket(kz_lo, kz_hi, omega)

        # Fall back to the wide bracket if continuation fails.
        if kz_root is None:
            kz_lo = omega / vs * (1.0 + eps)
            kz_hi = omega / vR * (1.0 - eps)
            if kz_hi > kz_lo:
                kz_root = _find_root_in_bracket(kz_lo, kz_hi, omega)

        if kz_root is None:
            logger.debug(
                "_flexural_dispersion_fast_formation: no Im(det) "
                "sign change at f=%.1f Hz; mode likely outside "
                "supported band", f,
            )
            continue

        slowness_desc[i] = kz_root / omega
        slowness_prev = slowness_desc[i]

    slowness[order_desc] = slowness_desc

    return BoreholeMode(
        name="flexural", azimuthal_order=1,
        freq=f_arr, slowness=slowness,
    )


def flexural_dispersion(
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
    Dipole flexural-wave phase slowness vs frequency from the n=1
    isotropic-elastic modal determinant (Schmitt 1988).

    Auto-dispatches on the formation type:

    * **Slow formation** (``V_S < V_f``): bound mode in the slowness
      window ``(1/V_S, ~1/V_R)`` (phase velocity in ``(V_R, V_S)``,
      well below ``V_f``). All radial wavenumbers ``F``, ``p``,
      ``s`` are real positive and the real-valued
      :func:`_modal_determinant_n1` is brentq'd directly.

    * **Fast formation** (``V_S > V_f``): bound mode in the same
      slowness window ``(1/V_S, 1/V_R)``, now with phase velocity
      *above* ``V_f``. The fluid radial wavenumber becomes purely
      imaginary (``F^2 < 0``); the formation P / S radial
      wavenumbers stay real positive. Dispatched to
      :func:`_flexural_dispersion_fast_formation`, which uses the
      complex-aware :func:`_modal_determinant_n1_complex` and
      brentq's its imaginary part along the real ``k_z`` axis
      (the real part is suppressed by ~10 orders of magnitude).

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
        ``name = "flexural"``, ``azimuthal_order = 1``, with
        ``freq`` echoed and ``slowness[i]`` the phase slowness
        ``k_z(omega[i]) / omega[i]``. ``NaN`` at any frequency
        where the bracket failed -- typically below the geometric
        cutoff (``f < V_S / (2 pi a)``) where no bound flexural
        root exists.

        Above the cutoff the slowness approaches ``1 / V_S`` from
        above as ``f -> cutoff^+`` (substep-1.6.b asymptote) and
        tapers toward slightly above ``1 / V_R`` at high f
        (Scholte / fluid-loading offset; substep-1.6.c-d). The
        fast- and slow-formation branches share this asymptotic
        layout; the only practical difference is which determinant
        evaluator is used internally
        (:func:`_modal_determinant_n1` for slow,
        :func:`_modal_determinant_n1_complex` for fast).

    Raises
    ------
    ValueError
        If any input is non-positive, ``vp <= vs``, or ``freq``
        contains a non-positive entry.

    Notes
    -----
    Plan item B in ``docs/plans/cylindrical_biot.md`` reuses the
    bound-regime brentq scaffolding for the fast-formation case
    via the complex modal determinant. The "fast-formation
    flexural is leaky and needs complex-k_z Mueller iteration"
    framing in earlier comments turned out to be over-stated for
    the canonical mode in the ``(V_R, V_S)`` velocity window:
    when the formation P / S branches are bound, the fast-
    formation root is also bound (real ``k_z``, ``Im(k_z) = 0``
    to floating-point precision), and the only difference from
    the slow-formation case is that the modal determinant picks
    up an overall ``i^k`` phase from ``F^2 < 0`` -- handled by
    brentq'ing the imaginary part of
    :func:`_modal_determinant_n1_complex`. Truly leaky n=1 modes
    with non-trivial ``Im(k_z) > 0`` (e.g., higher-order leaky
    flexural, fast-formation pseudo-flexural) need the complex
    marcher and are out of scope for this routine.

    References
    ----------
    * Schmitt, D. P. (1988). Shear-wave logging in elastic
      formations. *J. Acoust. Soc. Am.* 84(6), 2230-2244.
    * Paillet, F. L., & Cheng, C. H. (1991). *Acoustic Waves in
      Boreholes.* CRC Press, Ch. 4.
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
        return _flexural_dispersion_fast_formation(
            f_arr, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
        )

    slowness = np.full_like(f_arr, np.nan, dtype=float)
    for i, f in enumerate(f_arr):
        omega = 2.0 * np.pi * float(f)

        def _det(kz, omega=omega):
            return _modal_determinant_n1(kz, omega, vp, vs, rho, vf, rho_f, a)

        kz_lo, kz_hi = _flexural_kz_bracket(omega, vp, vs, rho, vf, rho_f, a)
        try:
            d_lo = _det(kz_lo)
            d_hi = _det(kz_hi)
            # If the bracket doesn't straddle a sign change, expand
            # the upper bound outward in steps of 1.5x. Matches the
            # n=0 stoneley_dispersion expansion pattern.
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
                # Typically the fast-formation case (F^2 < 0,
                # det evaluates to NaN). Outside the bound-mode
                # regime; leave slowness NaN.
                logger.debug(
                    "flexural_dispersion: bound regime failed at "
                    "f=%.1f Hz (likely fast formation V_S > V_f)",
                    f,
                )
                continue
            if np.sign(d_lo) == np.sign(d_hi):
                logger.debug(
                    "flexural_dispersion: failed to bracket at f=%.1f Hz",
                    f,
                )
                continue
            kz_root = optimize.brentq(_det, kz_lo, kz_hi, xtol=1.0e-10)
            slowness[i] = kz_root / omega
        except (ValueError, RuntimeError) as exc:
            logger.debug(
                "flexural_dispersion: brentq failed at f=%.1f Hz: %s",
                f,
                exc,
            )

    return BoreholeMode(
        name="flexural",
        azimuthal_order=1,
        freq=f_arr,
        slowness=slowness,
    )


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
# planned follow-ups; see ``docs/roadmap.md`` item A for the full
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
# ``Re(alpha) >= 0``. That gives the right sign in the bound regime
# (``alpha`` real and positive). In the leaky regime, ``alpha^2``
# has negative real part and the principal sqrt has positive real
# part with positive imaginary part:
#
#     alpha = sqrt(alpha^2)  -- numpy default
#         -> Re(alpha) >= 0, Im(alpha) >= 0.
#
# For the outgoing-wave condition with ``e^{-i omega t}`` time
# dependence, we need ``Im(alpha) > 0`` (so ``e^{i alpha r}`` decays
# as r grows). The numpy default already satisfies this on the
# principal branch -- no sign flip needed. This is the cleanest
# convention; document it explicitly because the other common
# textbook choice (``Re(alpha) < 0``) flips the sign and uses
# ``H_n^{(1)}``.
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
    F = np.sqrt(kz_c * kz_c - (omega / vf) ** 2)
    p = np.sqrt(kz_c * kz_c - (omega / vp) ** 2)
    s = np.sqrt(kz_c * kz_c - (omega / vs) ** 2)
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
    M11 = F * I1Fa / (rho_f * omega ** 2)
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

    M = np.array([[M11, M12, M13],
                  [M21, M22, M23],
                  [M31, M32, M33]], dtype=complex)
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
            method='hybr',
            options={'xtol': xtol},
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
        det_at_omega = (lambda kz, _omega=omega:  # noqa: E731
                        det_fn(kz, _omega))
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


@dataclass
class BranchSegment:
    """
    Contiguous stretch of finite samples in a dispersion curve.

    A :class:`BoreholeMode` may contain multiple physical segments
    separated by NaN gaps where the underlying mode does not exist
    (e.g., below a geometric cutoff) or where the marcher rejected
    a step. :class:`BranchSegment` represents one such contiguous
    stretch.

    Attributes
    ----------
    start_idx : int
        Index of the first sample of the segment in the original
        frequency grid (inclusive).
    end_idx : int
        Index of the last sample of the segment in the original
        frequency grid (inclusive). For a single-sample segment,
        ``end_idx == start_idx``.
    freq : ndarray, shape (n,)
        Frequencies in this segment, copied (or sliced) from the
        original frequency grid in the same order.
    kz : ndarray, shape (n,) complex
        Complex axial wavenumbers at each frequency in the
        segment.

    See Also
    --------
    segments_from_kz_curve : Build a list of segments from a marcher
        output.
    """

    start_idx: int
    end_idx: int
    freq: np.ndarray
    kz: np.ndarray

    def __len__(self) -> int:
        return int(self.end_idx - self.start_idx + 1)


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
                freq=f_arr[i:j + 1].copy(),
                kz=kz_arr[i:j + 1].copy(),
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
        det_at_omega = (lambda kz, _omega=omega:  # noqa: E731
                        det_fn(kz, _omega))
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
            i, n, omega / (2.0 * np.pi), verdict,
            consecutive_invalid, max_consecutive_invalid,
        )
        if consecutive_invalid > max_consecutive_invalid:
            break
    return kz_curve


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


def pseudo_rayleigh_dispersion(
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

    The geometric cutoff frequency is approximately

    .. math::
        f_c \approx \frac{j_{1,1} V_f V_S}
                         {2 \pi a \sqrt{V_S^2 - V_f^2}}

    where ``j_{1,1} \approx 3.832`` is the first positive zero of
    :math:`J_1`. This rigid-pipe-limit estimate is exposed as
    :data:`_J1_FIRST_ZERO` for callers that want to guard against
    requesting frequencies below the cutoff explicitly.

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
    * Schmitt, D. P. (1988). Shear-wave logging in elastic
      formations. *J. Acoust. Soc. Am.* 84(6), 2230-2244.
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
            f"pseudo-Rayleigh requires a fast formation (vs > vf); "
            f"got vs={vs}, vf={vf}"
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

    # High-frequency seed: slowness ~ 0.95 / V_S (5% inside the
    # leaky regime in slowness terms, equivalently 5% above V_S in
    # phase velocity), with a substantial positive imaginary part
    # so the determinant evaluator unambiguously sits on the leaky
    # branch. A seed pinned to slowness ~ 1/V_S itself causes the
    # hybrid root finder to converge to a numerical zero of the
    # Hankel-formulated determinant that lies just above 1/V_S in
    # slowness -- a non-physical solution outside the leaky-S
    # regime. The 5% offset is well-tested empirically against
    # standard fast-formation parameters (V_S/V_f ~ 2).
    omega_max = 2.0 * np.pi * float(f_desc[0])
    kz_seed = complex(
        omega_max / vs * 0.95,
        omega_max / vs * 5.0e-3,
    )

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
            kz, omega_step, vp, vs, rho, vf, rho_f, a,
            leaky_p=False, leaky_s=True,
        )

    kz_curve_desc = _march_complex_dispersion_validated(
        _det,
        f_desc,
        kz_seed,
        validator=_validator,
        max_consecutive_invalid=3,
    )

    omega_desc = 2.0 * np.pi * f_desc
    with np.errstate(invalid='ignore'):
        slowness_desc = kz_curve_desc.real / omega_desc
    attenuation_desc = kz_curve_desc.imag
    finite_desc = np.isfinite(kz_curve_desc.real) & np.isfinite(
        kz_curve_desc.imag
    )
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
    M11 = (F * I1Fa - 2.0 * I2Fa / a) / (rho_f * omega ** 2)
    M12 = p * K1pa + 2.0 * K2pa / a
    M13 = kz * K2sa
    M14 = -2.0 * K2sa / a

    # Row 2: -(sigma_rr^{(s)} + P) = 0 at r = a (cos(2 theta) sector;
    # row negated for visual parallel with the n = 0 / n = 1 forms).
    M21 = -I2Fa
    M22 = -mu * (
        two_kz2_minus_kS2 * K2pa + 2.0 * p * K1pa / a + 12.0 * K2pa / (a * a)
    )
    M23 = -2.0 * kz * mu * (s * K1sa + 2.0 * K2sa / a)
    M24 = 4.0 * mu * (s * K1sa / a + 3.0 * K2sa / (a * a))

    # Row 3: sigma_r_theta^{(s)} = 0 at r = a (sin(2 theta) sector;
    # fluid carries no shear, so M31 = 0).
    M31 = 0.0
    M32 = 4.0 * mu * (p * K1pa / a + 3.0 * K2pa / (a * a))
    M33 = 2.0 * kz * mu * K2sa / a
    M34 = -mu * (
        (s * s + 12.0 / (a * a)) * K2sa + 2.0 * s * K1sa / a
    )

    # Row 4: sigma_rz^{(s)} = 0 at r = a (cos(2 theta) sector; M41 = 0
    # for the same fluid-no-shear reason). Entries below are the
    # post-rescaling form: row 4 multiplied by i and column C
    # (= column 3 here) by -i, leaving a real matrix.
    M41 = 0.0
    M42 = 2.0 * kz * mu * (p * K1pa + 2.0 * K2pa / a)
    M43 = mu * (two_kz2_minus_kS2 + 3.0 / (a * a)) * K2sa
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
    F = np.sqrt(kz_c * kz_c - (omega / vf) ** 2)
    p = np.sqrt(kz_c * kz_c - (omega / vp) ** 2)
    s = np.sqrt(kz_c * kz_c - (omega / vs) ** 2)
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
    M11 = (F * I1Fa - 2.0 * I2Fa / a) / (rho_f * omega ** 2)
    M12 = p * K1pa + 2.0 * K2pa / a
    M13 = kz_c * K2sa
    M14 = -2.0 * K2sa / a

    # Row 2: -(sigma_rr^{(s)} + P) = 0 at r = a (cos(2 theta) sector;
    # row negated for visual parallel with the n=0 / n=1 forms).
    M21 = -I2Fa
    M22 = -mu * (
        two_kz2_minus_kS2 * K2pa + 2.0 * p * K1pa / a + 12.0 * K2pa / (a * a)
    )
    M23 = -2.0 * kz_c * mu * (s * K1sa + 2.0 * K2sa / a)
    M24 = 4.0 * mu * (s * K1sa / a + 3.0 * K2sa / (a * a))

    # Row 3: sigma_r_theta^{(s)} = 0 at r = a (sin(2 theta) sector;
    # fluid carries no shear, M31 = 0).
    M31 = 0.0 + 0j
    M32 = 4.0 * mu * (p * K1pa / a + 3.0 * K2pa / (a * a))
    M33 = 2.0 * kz_c * mu * K2sa / a
    M34 = -mu * ((s * s + 12.0 / (a * a)) * K2sa + 2.0 * s * K1sa / a)

    # Row 4: sigma_rz^{(s)} = 0 at r = a (cos(2 theta) sector;
    # M41 = 0 same fluid-no-shear reason). Same row-4-by-i /
    # column-C-by-(-i) rescale as the real version.
    M41 = 0.0 + 0j
    M42 = 2.0 * kz_c * mu * (p * K1pa + 2.0 * K2pa / a)
    M43 = mu * (two_kz2_minus_kS2 + 3.0 / (a * a)) * K2sa
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
    from fwap.cylindrical import rayleigh_speed

    f_arr = np.asarray(freq, dtype=float)
    n_f = f_arr.size
    slowness = np.full(n_f, np.nan, dtype=float)
    if n_f == 0:
        return BoreholeMode(
            name="quadrupole", azimuthal_order=2,
            freq=f_arr, slowness=slowness,
        )

    vR = rayleigh_speed(vp, vs)
    eps = 1.0e-4

    def _im_det(kz: float, _omega: float) -> float:
        return _modal_determinant_n2_complex(
            complex(kz, 0.0), _omega, vp, vs, rho, vf, rho_f, a,
            leaky_p=False, leaky_s=False,
        ).imag

    def _find_root_in_bracket(
        kz_lo: float, kz_hi: float, omega: float,
    ) -> float | None:
        try:
            d_lo = _im_det(kz_lo, omega)
            d_hi = _im_det(kz_hi, omega)
            if not (np.isfinite(d_lo) and np.isfinite(d_hi)):
                return None
            if np.sign(d_lo) == np.sign(d_hi):
                return None
            return float(optimize.brentq(
                _im_det, kz_lo, kz_hi, args=(omega,), xtol=1.0e-10,
            ))
        except (ValueError, RuntimeError):
            return None

    # Walk high to low frequency with continuation: narrow
    # bracket centred on the previous step's slowness first;
    # fall back to the wide ``(1/V_S, 1/V_R)`` bracket if the
    # narrow one fails. Mirrors the n=1 fast-formation strategy.
    order_desc = np.argsort(-f_arr)
    f_desc = f_arr[order_desc]
    slowness_desc = np.full(f_desc.size, np.nan, dtype=float)
    slowness_prev: float | None = None

    for i, f in enumerate(f_desc):
        omega = 2.0 * np.pi * float(f)
        kz_root: float | None = None

        if slowness_prev is not None:
            kz_centre = slowness_prev * omega
            kz_lo = max(kz_centre * 0.98, omega / vs * (1.0 + eps))
            kz_hi = min(kz_centre * 1.02, omega / vR * (1.0 - eps))
            if kz_hi > kz_lo:
                kz_root = _find_root_in_bracket(kz_lo, kz_hi, omega)

        if kz_root is None:
            kz_lo = omega / vs * (1.0 + eps)
            kz_hi = omega / vR * (1.0 - eps)
            if kz_hi > kz_lo:
                kz_root = _find_root_in_bracket(kz_lo, kz_hi, omega)

        if kz_root is None:
            logger.debug(
                "_quadrupole_dispersion_fast_formation: no Im(det) "
                "sign change at f=%.1f Hz", f,
            )
            continue

        slowness_desc[i] = kz_root / omega
        slowness_prev = slowness_desc[i]

    slowness[order_desc] = slowness_desc

    return BoreholeMode(
        name="quadrupole", azimuthal_order=2,
        freq=f_arr, slowness=slowness,
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
      Acoustic Methods.* Elsevier, sect. 2.5 and fig 3.7
      (LWD quadrupole dispersion).
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
            f_arr, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
        )

    slowness = np.full_like(f_arr, np.nan, dtype=float)

    for i, f in enumerate(f_arr):
        omega = 2.0 * np.pi * float(f)

        def _det(kz, omega=omega):
            return _modal_determinant_n2(kz, omega, vp, vs, rho, vf, rho_f, a)

        kz_lo, kz_hi = _quadrupole_kz_bracket(
            omega, vp, vs, rho, vf, rho_f, a,
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
                    "failed at f=%.1f Hz", f,
                )
                continue
            if np.sign(d_lo) == np.sign(d_hi):
                logger.debug(
                    "quadrupole_dispersion: failed to bracket at "
                    "f=%.1f Hz (likely below cutoff)", f,
                )
                continue
            kz_root = optimize.brentq(_det, kz_lo, kz_hi, xtol=1.0e-10)
            slowness[i] = kz_root / omega
        except (ValueError, RuntimeError) as exc:
            logger.debug(
                "quadrupole_dispersion: brentq failed at f=%.1f Hz: %s",
                f, exc,
            )

    return BoreholeMode(
        name="quadrupole", azimuthal_order=2,
        freq=f_arr, slowness=slowness,
    )

