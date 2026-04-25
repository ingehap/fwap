"""
Elastic moduli from sonic velocities.

Closes the loop from the raw-waveform processing in :mod:`fwap.picker`
/ :mod:`fwap.dispersion` (which produce ``Vp`` and ``Vs`` logs) to the
geomechanical curves a petrophysicist wants next to the other log
tracks: bulk modulus, shear modulus, Young's modulus, and Poisson's
ratio.

The formulas used here are the standard isotropic small-strain
relations; all four quantities follow from ``Vp``, ``Vs``, and the
formation bulk density ``rho``.

References
----------
* Mari, J.-L., Coppens, F., Gavin, P., & Wicquart, E. (1994).
  *Full Waveform Acoustic Data Processing*, Part 1 (workflow from
  shear logs to elastic moduli). Editions Technip, Paris.
  ISBN 978-2-7108-0664-6.
* Mavko, G., Mukerji, T., & Dvorkin, J. (2009). *The Rock Physics
  Handbook*, 2nd ed., Section 1.5. Cambridge University Press.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ElasticModuli:
    """
    Isotropic elastic moduli derived from Vp, Vs, and density.

    Attributes
    ----------
    k : ndarray or float
        Bulk modulus (Pa).
    mu : ndarray or float
        Shear modulus (Pa).
    young : ndarray or float
        Young's modulus (Pa).
    poisson : ndarray or float
        Poisson's ratio (dimensionless, in ``(-1, 0.5]``).
    lambda_ : ndarray or float
        First Lame parameter (Pa).
    """
    k: np.ndarray
    mu: np.ndarray
    young: np.ndarray
    poisson: np.ndarray
    lambda_: np.ndarray


def elastic_moduli(vp: np.ndarray,
                   vs: np.ndarray,
                   rho: np.ndarray) -> ElasticModuli:
    """
    Isotropic elastic moduli from compressional and shear velocity.

    For an isotropic medium of bulk density ``rho``, compressional
    velocity ``Vp``, and shear velocity ``Vs``::

        mu      = rho * Vs**2                           # shear modulus
        lambda_ = rho * Vp**2 - 2 * mu                  # first Lame
        K       = lambda_ + (2 / 3) * mu                # bulk modulus
        E       = mu * (3 * lambda_ + 2 * mu) / (lambda_ + mu)
        nu      = lambda_ / (2 * (lambda_ + mu))        # Poisson's ratio

    Inputs may be scalars or arrays; operations broadcast in the usual
    NumPy way.

    Parameters
    ----------
    vp : scalar or ndarray
        Compressional wave velocity (m/s). Must be positive.
    vs : scalar or ndarray
        Shear wave velocity (m/s). Must be positive and satisfy
        ``vs < vp``; otherwise the denominator of Poisson's ratio
        goes non-physical.
    rho : scalar or ndarray
        Bulk density (kg/m^3). Must be positive.

    Returns
    -------
    ElasticModuli
        Per-sample moduli, same shape as the broadcast of the inputs.
        All moduli in Pa; Poisson's ratio is dimensionless.

    Raises
    ------
    ValueError
        If any of ``vp``, ``vs``, ``rho`` is non-positive, or if
        ``vs >= vp`` anywhere (the isotropic rock-physics model
        requires ``Vp > Vs * sqrt(4/3)`` for stable positive
        moduli; we enforce the stricter and more familiar
        ``Vp > Vs``).
    """
    vp = np.asarray(vp, dtype=float)
    vs = np.asarray(vs, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if np.any(vp <= 0) or np.any(vs <= 0) or np.any(rho <= 0):
        raise ValueError("vp, vs, rho must all be positive")
    if np.any(vs >= vp):
        raise ValueError("require vs < vp everywhere")

    mu = rho * vs ** 2
    lam = rho * vp ** 2 - 2.0 * mu
    k = lam + (2.0 / 3.0) * mu
    # Young's modulus: E = mu (3 lam + 2 mu) / (lam + mu). Denominator
    # is positive whenever the inputs are physically valid (vp > vs).
    young = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
    # Poisson's ratio: nu = lam / (2 (lam + mu)). Range is (-1, 0.5].
    poisson = lam / (2.0 * (lam + mu))
    return ElasticModuli(k=k, mu=mu, young=young,
                         poisson=poisson, lambda_=lam)


def vp_vs_ratio(vp: np.ndarray, vs: np.ndarray) -> np.ndarray:
    """
    Vp/Vs ratio, a common lithology / fluid-content indicator.

    Typical values: ~1.5 for gas sands, ~1.7-1.8 for clean water
    sands and oil sands, >1.9 for shales and carbonates.
    """
    vp = np.asarray(vp, dtype=float)
    vs = np.asarray(vs, dtype=float)
    return vp / vs


# ---------------------------------------------------------------------
# Gassmann fluid substitution.
# ---------------------------------------------------------------------


@dataclass
class GassmannResult:
    """
    Fluid-saturated bulk and shear moduli from Gassmann substitution.

    Attributes
    ----------
    k_sat : ndarray or float
        Saturated-rock bulk modulus (Pa).
    mu_sat : ndarray or float
        Saturated-rock shear modulus (Pa). Equal to ``mu_dry`` because
        the Gassmann framework assumes the pore fluid carries no shear
        stress, so the shear modulus is fluid-insensitive.
    """
    k_sat: np.ndarray
    mu_sat: np.ndarray


def gassmann_fluid_substitution(
    k_dry: np.ndarray,
    mu_dry: np.ndarray,
    k_mineral: np.ndarray,
    k_fluid: np.ndarray,
    porosity: np.ndarray,
) -> GassmannResult:
    r"""
    Gassmann (1951) fluid substitution.

    Given the drained ("dry-frame") bulk and shear moduli of a porous
    rock, the bulk modulus of its solid mineral grains, and the bulk
    modulus of a replacement pore fluid at the rock's porosity,
    Gassmann gives the closed-form saturated-rock moduli:

    .. math::

        K_\mathrm{sat} \;=\; K_\mathrm{dry} \;+\;
        \frac{\bigl(1 - K_\mathrm{dry}/K_s\bigr)^2}
             {\phi/K_f + (1-\phi)/K_s - K_\mathrm{dry}/K_s^2}

    .. math::

        \mu_\mathrm{sat} \;=\; \mu_\mathrm{dry}

    The shear-modulus invariance follows from the assumption that the
    pore fluid supports no shear; it is the practical fact that lets
    the Vs log be read as a rock-frame (lithology) indicator largely
    independent of fluid content.

    Gassmann's equations assume (i) a closed, isotropic, homogeneous
    pore space with all pores communicating, (ii) the rock frame and
    pore fluid are in pressure equilibrium at the measurement
    frequency (no wave-induced local fluid flow), and (iii) the fluid
    does not chemically alter the frame. Violations of any of these
    produce the high-frequency / squirt corrections that Biot (1956)
    and later workers added.

    Parameters
    ----------
    k_dry : scalar or ndarray
        Dry-frame (drained) bulk modulus (Pa). Must be strictly
        positive and :math:`\le K_\mathrm{mineral}`.
    mu_dry : scalar or ndarray
        Dry-frame shear modulus (Pa). Returned unchanged as
        ``mu_sat``.
    k_mineral : scalar or ndarray
        Bulk modulus of the solid mineral phase (Pa). For quartz this
        is ~37 GPa, for calcite ~76.8 GPa. Composite mineralogies can
        be pre-averaged with :func:`hill_average`.
    k_fluid : scalar or ndarray
        Bulk modulus of the saturating pore fluid (Pa). Brine is
        ~2.2 GPa, oil ~1.0 GPa, gas 0.01-0.1 GPa. For multi-phase
        fluids, Reuss-average the individual fluid moduli first
        (Wood's law; Mavko et al. 2009 Section 6.1).
    porosity : scalar or ndarray
        Fractional porosity :math:`\phi \in [0, 1]`.

    Returns
    -------
    GassmannResult
        ``k_sat`` (Pa) and ``mu_sat`` (Pa) broadcast to the common
        shape of the inputs.

    Raises
    ------
    ValueError
        If any modulus is non-positive, if ``porosity`` falls outside
        ``[0, 1]``, or if ``k_dry > k_mineral`` anywhere (the dry
        frame cannot be stiffer than the mineral it is made of).

    Notes
    -----
    Limiting cases, verifiable by direct substitution:

    * :math:`\phi = 0` -- no pore space,
      :math:`K_\mathrm{sat} = K_s`.
    * :math:`K_\mathrm{dry} = K_s` -- a solid mineral with no
      compliant pore space, :math:`K_\mathrm{sat} = K_s`.
    * :math:`K_f \to 0` -- a "fluid" as soft as vacuum,
      :math:`K_\mathrm{sat} \to K_\mathrm{dry}`.

    References
    ----------
    * Gassmann, F. (1951). Ueber die Elastizitaet poroeser Medien.
      *Vierteljahrsschrift der Naturforschenden Gesellschaft in
      Zuerich* 96, 1-23.
    * Biot, M. A. (1956). Theory of propagation of elastic waves in a
      fluid-saturated porous solid. *J. Acoust. Soc. Am.* 28(2),
      168-191 (low-frequency consistency with Gassmann).
    * Mavko, G., Mukerji, T., & Dvorkin, J. (2009). *The Rock Physics
      Handbook*, 2nd ed., Section 6.3. Cambridge University Press.
    * Smith, T. M., Sondergeld, C. H., & Rai, C. S. (2003). Gassmann
      fluid substitutions: A tutorial. *Geophysics* 68(2), 430-440.
    """
    k_dry = np.asarray(k_dry, dtype=float)
    mu_dry = np.asarray(mu_dry, dtype=float)
    k_mineral = np.asarray(k_mineral, dtype=float)
    k_fluid = np.asarray(k_fluid, dtype=float)
    porosity = np.asarray(porosity, dtype=float)

    if np.any(k_dry <= 0):
        raise ValueError("k_dry must be strictly positive")
    if np.any(mu_dry <= 0):
        raise ValueError("mu_dry must be strictly positive")
    if np.any(k_mineral <= 0):
        raise ValueError("k_mineral must be strictly positive")
    if np.any(k_fluid <= 0):
        raise ValueError("k_fluid must be strictly positive")
    if np.any(porosity < 0) or np.any(porosity > 1):
        raise ValueError("porosity must lie in [0, 1]")
    if np.any(k_dry > k_mineral):
        raise ValueError(
            "k_dry must be <= k_mineral "
            "(the dry frame cannot be stiffer than its mineral grains)"
        )

    numerator = (1.0 - k_dry / k_mineral) ** 2
    denominator = (
        porosity / k_fluid
        + (1.0 - porosity) / k_mineral
        - k_dry / k_mineral ** 2
    )
    k_sat = k_dry + numerator / denominator
    # Align mu_sat with the broadcast shape of k_sat so downstream code
    # can trust the two arrays are shape-matched.
    mu_sat = np.broadcast_to(mu_dry, k_sat.shape).copy()
    return GassmannResult(k_sat=k_sat, mu_sat=mu_sat)


# ---------------------------------------------------------------------
# Reuss-Voigt-Hill mixing laws for composite media.
# ---------------------------------------------------------------------


def _validate_mixture(moduli: np.ndarray, fractions: np.ndarray) -> None:
    if moduli.shape != fractions.shape:
        raise ValueError(
            f"moduli and fractions must have the same shape; got "
            f"{moduli.shape} and {fractions.shape}"
        )
    if np.any(moduli <= 0):
        raise ValueError("every modulus must be strictly positive")
    if np.any(fractions < 0):
        raise ValueError("every volume fraction must be non-negative")
    total = float(np.sum(fractions))
    if not (0.999 <= total <= 1.001):
        raise ValueError(
            f"fractions must sum to 1.0 (got {total:.4f})"
        )


def reuss_average(moduli: np.ndarray,
                  fractions: np.ndarray) -> float:
    r"""
    Reuss (harmonic, isostress) average of a composite modulus.

    Gives the *lower* (softer) bound on the effective modulus of a
    mixture of N isotropic components at arbitrary volume fractions:

    .. math::

        M_R \;=\; \left( \sum_i \frac{f_i}{M_i} \right)^{-1}

    The Reuss bound is the exact answer when the components are
    arranged as layers perpendicular to the loading direction
    (isostress), and is therefore the worst-case stiffness for any
    isotropic mixing geometry.

    Parameters
    ----------
    moduli : ndarray, shape (N,)
        Per-component modulus of the same type (all bulk, or all
        shear). Units are preserved (Pa in, Pa out).
    fractions : ndarray, shape (N,)
        Volume fractions, must sum to 1.0 to within 1e-3 and all be
        non-negative.

    Returns
    -------
    float
        Effective modulus in the same unit as ``moduli``.

    References
    ----------
    * Reuss, A. (1929). Berechnung der Fliessgrenze von Mischkristallen
      auf Grund der Plastizitaetsbedingung fuer Einkristalle.
      *Zeitschrift fuer Angewandte Mathematik und Mechanik* 9, 49-58.
    * Mavko, Mukerji & Dvorkin (2009), *The Rock Physics Handbook*,
      2nd ed., Section 4.1.
    """
    moduli = np.asarray(moduli, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    _validate_mixture(moduli, fractions)
    return float(1.0 / np.sum(fractions / moduli))


def voigt_average(moduli: np.ndarray,
                  fractions: np.ndarray) -> float:
    r"""
    Voigt (arithmetic, isostrain) average of a composite modulus.

    Gives the *upper* (stiffer) bound on the effective modulus:

    .. math::

        M_V \;=\; \sum_i f_i M_i

    Exact when the components are arranged as layers parallel to the
    loading direction (isostrain); the best-case stiffness for any
    isotropic mixing geometry.

    Parameters
    ----------
    moduli : ndarray, shape (N,)
    fractions : ndarray, shape (N,)
        Volume fractions, must sum to 1.0.

    Returns
    -------
    float

    References
    ----------
    * Voigt, W. (1889). Ueber die Beziehung zwischen den beiden
      Elastizitaetskonstanten isotroper Koerper. *Annalen der Physik
      und Chemie* 38, 573-587.
    * Mavko, Mukerji & Dvorkin (2009), Section 4.1.
    """
    moduli = np.asarray(moduli, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    _validate_mixture(moduli, fractions)
    return float(np.sum(fractions * moduli))


def stoneley_permeability_indicator(
    slowness_observed: np.ndarray,
    slowness_reference: float | np.ndarray,
) -> np.ndarray:
    r"""
    Dimensionless Stoneley-wave permeability indicator.

    Returns the fractional Stoneley-slowness shift relative to a
    tight (low-permeability) reference zone:

    .. math::

        \alpha_\mathrm{ST}(d) \;=\;
        \frac{s_\mathrm{ST,obs}(d)}{s_\mathrm{ST,ref}} - 1

    The Stoneley wave in a fluid-filled borehole loses energy into
    Darcy flow in a permeable formation, which slows its phase
    velocity (increases its slowness). At low frequencies the
    fractional shift scales linearly with formation permeability
    (Tang & Cheng 2004, Section 5.1), so this indicator rank-orders
    zones by permeability.

    Important: this is an **uncalibrated, dimensionless** indicator,
    not an absolute permeability in m^2 or mD. Converting to SI
    permeability needs a calibration constant that depends on the
    tool's source frequency, borehole fluid viscosity and bulk
    modulus, and tube-wave phase velocity in the reference zone --
    quantities that in practice are tuned against a known-permeability
    interval in the well (core data or a mudlog), not computed from
    first principles. The simple closed-form Biot low-frequency
    formula is given in Kostek & Johnson (1992), Section 3, but its
    parameters vary enough with local fluid conditions that
    calibration against a reference zone is the standard workflow.

    Parameters
    ----------
    slowness_observed : ndarray or float
        Per-depth Stoneley-wave slowness (s/m). Typically
        :attr:`fwap.picker.ModePick.slowness` for the ``"Stoneley"``
        mode gathered across depths, or the ``"Stoneley"`` column of
        a LAS log read via :func:`fwap.io.read_las`.
    slowness_reference : float or ndarray
        Tight-reference Stoneley slowness (s/m). Either a single
        value (one tight zone) or a per-depth reference baseline
        (e.g., a low-pass-smoothed version of the observed log).

    Returns
    -------
    ndarray
        Dimensionless indicator, typically in ``[-0.01, 0.5]``.
        Positive values flag permeable intervals; near-zero values
        are zones that match the reference. Negative values indicate
        the reference itself was not tight relative to that depth or
        that SNR issues have corrupted the pick.

    References
    ----------
    * Tang, X.-M., & Cheng, A. (2004). *Quantitative Borehole Acoustic
      Methods.* Elsevier, Section 5.1.
    * Kostek, S., & Johnson, D. L. (1992). The interaction of tube
      waves with borehole fractures, Part I: Numerical models.
      *Geophysics* 57(6), 784-795.
    * Mari, J.-L., Coppens, F., Gavin, P., & Wicquart, E. (1994).
      *Full Waveform Acoustic Data Processing*, Part 1 (Stoneley
      amplitude as permeability / fracture proxy). Editions Technip.
    """
    obs = np.asarray(slowness_observed, dtype=float)
    ref = np.asarray(slowness_reference, dtype=float)
    if np.any(obs <= 0):
        raise ValueError("slowness_observed must be strictly positive")
    if np.any(ref <= 0):
        raise ValueError("slowness_reference must be strictly positive")
    return obs / ref - 1.0


def stoneley_amplitude_fracture_indicator(
    amplitude_observed: np.ndarray,
    amplitude_reference: float | np.ndarray,
) -> np.ndarray:
    r"""
    Dimensionless Stoneley-amplitude fracture / permeability indicator.

    Returns the fractional Stoneley amplitude *deficit* relative to a
    tight (unfractured, low-permeability) reference zone:

    .. math::

        \beta_\mathrm{ST}(d) \;=\;
        1 \;-\; \frac{A_\mathrm{ST,obs}(d)}{A_\mathrm{ST,ref}}.

    The Stoneley wave attenuates as it crosses a fracture or permeable
    interval -- energy radiates into the formation through Darcy flow
    in the rock matrix and through fracture-pumping (oscillatory fluid
    motion in and out of fracture apertures) at fractures intersecting
    the borehole wall (Tang & Cheng 2004, sect. 5.2; Hornby, Johnson,
    Winkler & Plumb 1989, *Geophysics* 54(10), 1274-1288). The
    fractional amplitude deficit therefore flags the same fractures
    and permeable zones as :func:`stoneley_permeability_indicator`,
    but with **complementary noise characteristics**: amplitude
    attenuation responds primarily to the *loss* of acoustic energy
    along the wavetrain, while the slowness-shift indicator responds
    primarily to the dynamic poroelastic delay. Combining the two
    (e.g. as a coincidence flag, or by averaging after rank-
    standardisation) is more robust than either alone.

    Important: like the slowness-shift companion, this is an
    **uncalibrated, dimensionless** indicator -- a depth-by-depth
    rank-ordering of fracture / permeability strength, not an
    absolute permeability or fracture aperture. Conversion to SI
    quantities needs a calibration tied to the tool's source
    frequency, borehole-fluid viscosity / bulk modulus, formation
    Stoneley impedance, and (for fractures) the fracture-aperture
    distribution; in practice it is tuned against a known interval
    (image log, core, mudlog) rather than computed from first
    principles. The closed-form low-frequency expressions are in
    Hornby et al. (1989), eqs. (3)-(7) (single-fracture reflection
    / transmission) and Tang & Cheng (2004), sect. 5.2 (matrix-
    permeability transmission loss).

    Parameters
    ----------
    amplitude_observed : ndarray or float
        Per-depth Stoneley-wave amplitude. Typically
        :attr:`fwap.picker.ModePick.amplitude` for the ``"Stoneley"``
        mode gathered across depths, or the ``"AMPST"`` column of a
        log set written via :func:`fwap.picker.track_to_log_curves`.
        Must be non-negative.
    amplitude_reference : float or ndarray
        Tight-reference Stoneley amplitude. Either a single value
        (a hand-picked tight zone) or a per-depth baseline (e.g. a
        median-filtered or low-pass-smoothed version of the observed
        log). Must be strictly positive.

    Returns
    -------
    ndarray
        Dimensionless indicator, typically in ``[-0.05, 0.9]``.
        Positive values flag permeable / fractured intervals (lower
        observed amplitude than reference); near-zero values are
        zones that match the reference. Negative values indicate
        either an unusual amplification (rare; resonance from
        thin-bed multiples) or that the reference itself was lower
        than this depth.

    Raises
    ------
    ValueError
        If ``amplitude_observed`` is negative anywhere or
        ``amplitude_reference`` is non-positive anywhere.

    See Also
    --------
    stoneley_permeability_indicator :
        The slowness-shift companion. Run both and combine as a
        coincidence flag for robust fracture / permeability picks.

    References
    ----------
    * Hornby, B. E., Johnson, D. L., Winkler, K. W., & Plumb, R. A.
      (1989). Fracture evaluation using reflected Stoneley-wave
      arrivals. *Geophysics* 54(10), 1274-1288.
    * Tang, X.-M., & Cheng, A. (2004). *Quantitative Borehole
      Acoustic Methods.* Elsevier, Section 5.2.
    * Mari, J.-L., Coppens, F., Gavin, P., & Wicquart, E. (1994).
      *Full Waveform Acoustic Data Processing*, Part 1 (Stoneley
      amplitude as permeability / fracture proxy). Editions Technip.
    """
    obs = np.asarray(amplitude_observed, dtype=float)
    ref = np.asarray(amplitude_reference, dtype=float)
    if np.any(obs < 0):
        raise ValueError("amplitude_observed must be non-negative")
    if np.any(ref <= 0):
        raise ValueError("amplitude_reference must be strictly positive")
    return 1.0 - obs / ref


def hill_average(moduli: np.ndarray,
                 fractions: np.ndarray) -> float:
    r"""
    Voigt-Reuss-Hill average: arithmetic mean of the Voigt and Reuss
    bounds.

    .. math::

        M_\mathrm{VRH} \;=\; \tfrac{1}{2}\,(M_V + M_R)

    Commonly used as a pragmatic "best estimate" for the effective
    modulus of an isotropic composite when the component geometry is
    unknown; always lies between the true stiffness bounds, but has
    no deeper physical justification than "halfway between".

    Parameters
    ----------
    moduli : ndarray, shape (N,)
    fractions : ndarray, shape (N,)
        Volume fractions, must sum to 1.0.

    Returns
    -------
    float

    References
    ----------
    * Hill, R. (1952). The elastic behaviour of a crystalline
      aggregate. *Proceedings of the Physical Society of London A*
      65, 349-354.
    * Mavko, Mukerji & Dvorkin (2009), Section 4.1.
    """
    moduli = np.asarray(moduli, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    _validate_mixture(moduli, fractions)
    voigt = float(np.sum(fractions * moduli))
    reuss = float(1.0 / np.sum(fractions / moduli))
    return 0.5 * (voigt + reuss)
