"""
Stoneley-wave estimators: permeability indicators, fracture indicators,
and slow-formation V_S inversion.

A Stoneley tube wave in a fluid-filled borehole is one of the most
information-rich logs available from a sonic tool. Its phase slowness
encodes formation permeability (Darcy flow into / out of the borehole
slows the wave); its amplitude encodes fracture density and aperture
(Stoneley reflection at a fracture removes energy from the transmitted
wave); and at low frequency in a slow formation (``V_S < V_fluid``) the
Stoneley wave is the only sonic mode whose slowness still tracks
``V_S``, providing the primary shear-velocity estimator there.

The functions in this module work directly off the Stoneley slowness
and amplitude logs produced by :mod:`fwap.picker` / :mod:`fwap.dispersion`
and require no further dispersion modelling. They are deliberately kept
together (rather than scattered across rockphysics / anisotropy /
fracture modules) so that the full Stoneley-derived petrophysical
workflow is in one place.

Public functions
----------------
* :func:`stoneley_permeability_indicator` -- dimensionless
  fractional slowness shift vs. a tight reference zone.
* :func:`stoneley_permeability_tang_cheng` -- Tang & Cheng (1996)
  forward-model-based permeability inversion in Darcy units.
* :func:`stoneley_amplitude_fracture_indicator` -- amplitude-drop
  fracture flag from a tight reference.
* :func:`stoneley_reflection_coefficient` -- ``|R|`` from incident
  and reflected pulse amplitudes at a fracture.
* :func:`hornby_fracture_aperture` -- single-fracture aperture from
  ``|R|`` (Hornby et al., 1989).
* :func:`stoneley_fracture_density` -- combined slowness + amplitude
  fracture-density indicator with optional matrix-permeability
  partitioning.
* :func:`vs_from_stoneley_slow_formation` -- primary slow-formation
  ``V_S`` estimator from low-frequency Stoneley slowness.

References
----------
* Hornby, B. E., Johnson, D. L., Winkler, K. W., & Plumb, R. A.
  (1989). Fracture evaluation using reflected Stoneley-wave arrivals.
  *Geophysics* 54(10), 1274-1288.
* Tang, X. M., & Cheng, C. H. (1996). Fast inversion of formation
  permeability from Stoneley wave logs using a simplified Biot-
  Rosenbaum model. *Geophysics* 61(3), 639-645.
* Tang, X. M., & Cheng, C. H. (2004). *Quantitative Borehole Acoustic
  Methods*. Elsevier (Section 5).
* Mari, J.-L., Coppens, F., Gavin, P., & Wicquart, E. (1994).
  *Full Waveform Acoustic Data Processing*. Editions Technip.
"""

from __future__ import annotations

import numpy as np


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


def stoneley_permeability_tang_cheng(
    slowness_observed: np.ndarray,
    slowness_reference: float | np.ndarray,
    *,
    frequency: float,
    fluid_bulk_modulus: float,
    fluid_viscosity: float,
    fluid_density: float,
    porosity: np.ndarray,
    frame_bulk_modulus: np.ndarray,
) -> np.ndarray:
    r"""
    Quantitative Stoneley permeability via Tang-Cheng-Toksoz (1991).

    Closed-form low-frequency inversion of the Stoneley slowness
    shift to absolute formation permeability in m^2. Calibrated
    complement to the dimensionless rank-ordering returned by
    :func:`stoneley_permeability_indicator`.

    Model
    -----
    The simplified Biot-Rosenbaum closed form (Tang, Cheng & Toksoz
    1991, eq. 36; Tang & Cheng 2004, sect. 5.1) at angular frequency
    :math:`\omega` well below the Biot characteristic frequency
    relates the dispersive part of the fractional Stoneley slowness
    shift :math:`\alpha_\mathrm{ST}` to the Biot characteristic
    angular frequency :math:`\omega_c`:

    .. math::

        \alpha_\mathrm{ST}(\omega) \;\approx\;
        \frac{1}{2}\,\frac{K_f}{K_\phi}
        \cdot \frac{i\,\omega/\omega_c}{1 + i\,\omega/\omega_c},

        \qquad
        \omega_c \;=\; \frac{\eta\,\phi}{\kappa\,\rho_f}.

    The dispersion factor is written so that
    :math:`\alpha_\mathrm{ST} \to 0` as :math:`\kappa \to 0` (tight
    formation; no slowdown vs the tight reference) and
    :math:`\alpha_\mathrm{ST} \to K_f/(2 K_\phi)` as
    :math:`\kappa \to \infty` (model upper bound), matching the
    sign convention of :func:`stoneley_permeability_indicator`
    where positive :math:`\alpha_\mathrm{ST}` flags permeable zones.

    Inversion (real-valued :math:`\alpha_\mathrm{ST}`)
    --------------------------------------------------
    For real :math:`\alpha_\mathrm{ST}` (the typical case using the
    slowness shift only, not the imaginary attenuation part), take
    the real part of the dispersion factor. Letting
    :math:`x = \omega / \omega_c` and :math:`A = K_f / (2 K_\phi)`,

    .. math::

        \mathrm{Re}\!\left[\frac{i x}{1 + i x}\right] = \frac{x^2}{1 + x^2},

        \qquad
        \alpha_\mathrm{ST} = A \cdot \frac{x^2}{1 + x^2}.

    Solving for :math:`x^2`:

    .. math::

        x^2 = \frac{\alpha_\mathrm{ST}}{A - \alpha_\mathrm{ST}},
        \qquad
        \omega_c = \frac{\omega}{\sqrt{x^2}},
        \qquad
        \kappa = \frac{\eta\,\phi}{\omega_c\,\rho_f}
              = \frac{\eta\,\phi}{\omega\,\rho_f}
                \sqrt{\frac{\alpha_\mathrm{ST}}{A - \alpha_\mathrm{ST}}}.

    Out-of-model handling:

    * :math:`\alpha_\mathrm{ST} \le 0` (tight formation or
      noise-driven negative): clipped to :math:`\kappa = 0`.
    * :math:`0 < \alpha_\mathrm{ST} < A`: standard inversion.
    * :math:`\alpha_\mathrm{ST} \ge A` (observed shift exceeds the
      model upper bound): NaN. Typically indicates open fractures
      or the reference zone was not truly tight; the
      :func:`hornby_fracture_aperture` model is the natural
      complement when the cause is fractures.

    Parameters
    ----------
    slowness_observed : ndarray, shape (n_depths,)
        Per-depth Stoneley-wave slowness (s/m). Same input as for
        :func:`stoneley_permeability_indicator`.
    slowness_reference : float or ndarray
        Tight-reference Stoneley slowness (s/m). Either a single
        value (one tight zone) or a per-depth reference baseline.
    frequency : float
        Frequency (Hz) at which the slowness was measured. Must
        be well below the Biot characteristic frequency for the
        formations of interest -- typically 1-2 kHz for sonic
        Stoneley logging in moderate-permeability rocks.
    fluid_bulk_modulus : float
        Borehole-fluid bulk modulus :math:`K_f` (Pa). Typical
        values: water 2.2 GPa = 2.2e9 Pa, oil 1.0-1.5 GPa,
        drilling mud 2.0-2.5 GPa.
    fluid_viscosity : float
        Borehole-fluid dynamic viscosity :math:`\eta` (Pa s).
        Typical: water 1e-3 Pa s, light oil 1e-3 to 1e-2 Pa s,
        drilling mud 1e-2 to 1e-1 Pa s.
    fluid_density : float
        Borehole-fluid mass density :math:`\rho_f` (kg/m^3).
    porosity : ndarray, shape (n_depths,)
        Per-depth formation porosity :math:`\phi`, dimensionless,
        in the open interval (0, 1).
    frame_bulk_modulus : ndarray, shape (n_depths,)
        Per-depth dry-frame bulk modulus :math:`K_\phi` of the
        porous formation (Pa). Sets the model upper bound
        :math:`A = K_f / (2 K_\phi)` on :math:`\alpha_\mathrm{ST}`.

    Returns
    -------
    ndarray, shape (n_depths,)
        Permeability :math:`\kappa` in m^2. Multiply by
        ``9.869233e-13`` to convert to darcies (1 darcy
        :math:`\approx` 9.87e-13 m^2; 1 millidarcy
        :math:`\approx` 9.87e-16 m^2). Zero where the slowness
        shift is non-positive (clipped). NaN where the shift
        exceeds the model upper bound.

    Raises
    ------
    ValueError
        If ``frequency``, ``fluid_bulk_modulus``,
        ``fluid_viscosity``, or ``fluid_density`` is non-positive;
        if any slowness, frame modulus, or porosity is out of its
        physical range; or if input array shapes are incompatible.

    See Also
    --------
    stoneley_permeability_indicator : Dimensionless rank-ordering
        without the Biot calibration; the natural input to this
        function via the alpha_ST = (s_obs / s_ref - 1) form.
    hornby_fracture_aperture : Reflected-wave inversion for the
        complementary case where the slowness shift exceeds the
        Biot-Rosenbaum upper bound A and the cause is open
        fractures rather than matrix permeability.
    stoneley_amplitude_fracture_indicator : Energy-loss-based
        permeability indicator with complementary noise
        characteristics.

    Notes
    -----
    The model assumes a uniform, isotropic, simply-connected pore
    space with Darcy-flow exchange between the borehole and the
    formation. It does not model:

    * The imaginary part of :math:`\alpha_\mathrm{ST}` (Stoneley
      attenuation); the real-part inversion uses the slowness
      shift only. The imaginary-part inversion would carry
      independent permeability information when amplitude data
      is reliable -- a follow-up.
    * Mudcake or formation-altered-zone radial layering.
    * Anisotropic permeability (each depth gets a single scalar).
    * Open fractures (use ``hornby_fracture_aperture`` for those).

    Validation against Tang & Cheng (2004) Figure 5.3 (synthetic
    permeable bed of 1-2 darcy bracketed by tight limestone of
    0.01-0.1 mD) is in the test suite as a round-trip check on
    the forward / inverse pair.

    References
    ----------
    * Tang, X.-M., Cheng, A., & Toksoz, M. N. (1991). Dynamic
      permeability and borehole Stoneley waves: A simplified
      Biot-Rosenbaum model. *J. Acoust. Soc. Am.* 90(3),
      1632-1646.
    * Tang, X.-M., & Cheng, A. (2004). *Quantitative Borehole
      Acoustic Methods.* Elsevier, Section 5.1 (the closed-form
      inversion as implemented here, plus the Figure 5.3
      validation example).
    * Kostek, S., & Johnson, D. L. (1992). The interaction of
      tube waves with borehole fractures, Part I: Numerical
      models. *Geophysics* 57(6), 784-795.
    """
    if frequency <= 0:
        raise ValueError("frequency must be positive")
    if fluid_bulk_modulus <= 0:
        raise ValueError("fluid_bulk_modulus must be positive")
    if fluid_viscosity <= 0:
        raise ValueError("fluid_viscosity must be positive")
    if fluid_density <= 0:
        raise ValueError("fluid_density must be positive")

    s_obs = np.asarray(slowness_observed, dtype=float)
    s_ref = np.asarray(slowness_reference, dtype=float)
    phi = np.asarray(porosity, dtype=float)
    K_phi = np.asarray(frame_bulk_modulus, dtype=float)

    if np.any(s_obs <= 0):
        raise ValueError("slowness_observed must be strictly positive")
    if np.any(s_ref <= 0):
        raise ValueError("slowness_reference must be strictly positive")
    if np.any(phi <= 0) or np.any(phi >= 1):
        raise ValueError("porosity must be strictly between 0 and 1")
    if np.any(K_phi <= 0):
        raise ValueError("frame_bulk_modulus must be strictly positive")

    omega = 2.0 * np.pi * frequency
    K_f = fluid_bulk_modulus
    eta = fluid_viscosity
    rho_f = fluid_density

    # Fractional slowness shift (matches stoneley_permeability_indicator).
    alpha_ST = s_obs / s_ref - 1.0

    # Model upper bound on alpha_ST: A = K_f / (2 K_phi). Per-depth.
    A = K_f / (2.0 * K_phi)

    # Broadcast to a common shape; raises ValueError on incompatible
    # input shapes.
    alpha_b, A_b, phi_b = np.broadcast_arrays(alpha_ST, A, phi)

    kappa = np.zeros_like(alpha_b, dtype=float)

    # Out-of-model: alpha_ST >= A. Set NaN before the valid mask so
    # the valid branch can overwrite when bounds happen to coincide.
    out_of_model = alpha_b >= A_b
    kappa[out_of_model] = np.nan

    # Standard inversion: 0 < alpha_ST < A.
    valid = (alpha_b > 0) & (alpha_b < A_b)
    if np.any(valid):
        ratio = alpha_b[valid] / (A_b[valid] - alpha_b[valid])
        kappa[valid] = eta * phi_b[valid] * np.sqrt(ratio) / (omega * rho_f)

    # alpha_ST <= 0 (tight or noise-driven negative): kappa stays 0.

    return kappa


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


# ---------------------------------------------------------------------
# Stoneley reflection-coefficient fracture-aperture inversion
# (Hornby, Johnson, Winkler & Plumb, 1989)
# ---------------------------------------------------------------------


def stoneley_reflection_coefficient(
    amplitude_incident: np.ndarray,
    amplitude_reflected: np.ndarray,
) -> np.ndarray:
    r"""
    Stoneley reflection coefficient :math:`|R|` from incident and
    reflected pulse amplitudes.

    Returns

    .. math::

        |R| \;=\; |A_\mathrm{reflected}| \, / \, |A_\mathrm{incident}|,

    clipped to ``[0, 1]`` -- numerical drift in the up/down-going
    decomposition can push noisy estimates a few percent above 1, but
    a physical reflection coefficient is bounded by unity.

    Prerequisite: the caller has separated the up-going (reflected)
    and down-going (incident) Stoneley arrivals at each fracture
    candidate. That decomposition is non-trivial and depends on the
    array geometry: f-k filtering (:func:`fwap.wavesep.fk_filter`) or
    tau-p filtering (:func:`fwap.wavesep.tau_p_filter`) at the Stoneley
    apparent slowness gives the two propagation directions; their
    peak-to-peak amplitudes near each picked fracture depth are the
    inputs here.

    Parameters
    ----------
    amplitude_incident, amplitude_reflected : ndarray or float
        Incident (down-going) and reflected (up-going) Stoneley pulse
        amplitudes near each fracture candidate. Same shape; the
        function broadcasts. Sign is irrelevant -- the function takes
        absolute values.

    Returns
    -------
    ndarray
        :math:`|R| \in [0, 1]`. ``1`` is full reflection (impossible
        for a real fracture; flag for QC); ``0`` is a non-reflecting
        depth.

    Raises
    ------
    ValueError
        If any incident amplitude is non-positive (the ratio is
        undefined).
    """
    a_i = np.abs(np.asarray(amplitude_incident, dtype=float))
    a_r = np.abs(np.asarray(amplitude_reflected, dtype=float))
    if np.any(a_i <= 0):
        raise ValueError("amplitude_incident must be strictly positive")
    return np.clip(a_r / a_i, 0.0, 1.0)


def hornby_fracture_aperture(
    reflection_coefficient: np.ndarray,
    frequency_hz: float | np.ndarray,
    stoneley_velocity: float | np.ndarray,
    *,
    small_amplitude_approx: bool = False,
) -> np.ndarray:
    r"""
    Fracture aperture from the Hornby et al. (1989) Stoneley
    reflection-coefficient inversion.

    Hornby, Johnson, Winkler & Plumb (1989, *Geophysics* 54(10),
    1274-1288) treat a horizontal open fluid-filled fracture
    intersecting the borehole as a thin compliant layer. In the
    rigid-frame, low-frequency, single-fracture limit their reflection
    coefficient is

    .. math::

        |R(\omega)| \;=\; \frac{\omega L_0}
                              {\sqrt{V_T^2 + \omega^2 L_0^{\,2}}},

    where :math:`L_0` is the fracture aperture (m), :math:`\omega = 2
    \pi f` is the angular frequency, and :math:`V_T` is the Stoneley
    phase velocity in the unfractured borehole. Inverting for
    :math:`L_0`:

    .. math::

        L_0 \;=\; \frac{V_T \, |R|}
                       {\omega \, \sqrt{1 - |R|^2}}
              \;=\; \frac{V_T \, |R|}
                         {2\pi f \, \sqrt{1 - |R|^2}}.

    For small reflection coefficients the radical can be dropped,
    giving the small-amplitude approximation

    .. math::

        L_0 \;\approx\; \frac{V_T \, |R|}{2\pi f},

    which is exact to better than 5% for :math:`|R| \le 0.3`.

    Assumptions
    -----------
    * Single horizontal open fracture intersecting the borehole at a
      right angle. Inclined fractures pick up a ``cos(theta)`` factor
      that is not modelled here.
    * Rigid frame (no matrix permeability). Permeable formations
      attenuate the Stoneley wave through Darcy flow as well as
      through fracture pumping; separating the two needs the full
      Hornby et al. (1989) eqs. (5)-(7) or the simplified
      Biot-Rosenbaum decomposition of Tang, Cheng & Toksoz (1991,
      *J. Acoust. Soc. Am.* 90(3), 1632-1646). Use this routine on
      the reflected-wave channel only -- where the matrix
      contribution is small relative to the fracture pumping.
    * Low frequency (:math:`\omega L_0 / V_T \lesssim 1`). At higher
      frequencies the rigid-frame approximation breaks down and the
      full dispersive form must be used.
    * Stoneley wavelength much larger than fracture aperture
      (:math:`L_0 \ll V_T / f`). For typical sonic frequencies
      (1-5 kHz) this holds up to centimetre-scale apertures.

    Parameters
    ----------
    reflection_coefficient : ndarray or float
        :math:`|R| \in [0, 1)`. Output of
        :func:`stoneley_reflection_coefficient`.
    frequency_hz : float or ndarray
        Stoneley dominant frequency at the reflection (Hz). Either a
        single scalar (one dominant frequency for the whole log) or
        a per-fracture array. Must be strictly positive.
    stoneley_velocity : float or ndarray
        Stoneley phase velocity in the unfractured borehole (m/s)
        near the fracture. The reciprocal of the picked Stoneley
        slowness in a tight reference zone is the standard estimate;
        compute as ``1.0 / s_Stoneley_ref``. Must be strictly
        positive.
    small_amplitude_approx : bool, default False
        If ``True``, use :math:`L_0 = V_T |R| / (2 \pi f)` instead of
        the full Hornby form. Saves the radical evaluation; differs
        by < 5% for :math:`|R| \le 0.3`.

    Returns
    -------
    ndarray
        Fracture aperture :math:`L_0` (m). Same shape as the broadcast
        of the inputs. ``+inf`` at depths where ``|R| = 1`` (the
        inversion saturates); the caller can mask these as
        catastrophic / non-physical.

    Raises
    ------
    ValueError
        If any reflection coefficient is outside ``[0, 1]``, or any
        frequency / velocity is non-positive.

    See Also
    --------
    stoneley_reflection_coefficient :
        Builds :math:`|R|` from incident / reflected pulse amplitudes.
    stoneley_permeability_indicator :
        Slowness-shift indicator for matrix permeability and
        fractures.
    stoneley_amplitude_fracture_indicator :
        Amplitude-deficit indicator (transmission-loss form;
        complementary to this reflection-coefficient inversion).

    References
    ----------
    * Hornby, B. E., Johnson, D. L., Winkler, K. W., & Plumb, R. A.
      (1989). Fracture evaluation using reflected Stoneley-wave
      arrivals. *Geophysics* 54(10), 1274-1288.
    * Tang, X.-M., & Cheng, A. (2004). *Quantitative Borehole
      Acoustic Methods.* Elsevier, Section 4.5 (Stoneley reflection
      and aperture inversion in field practice).
    """
    R = np.asarray(reflection_coefficient, dtype=float)
    f = np.asarray(frequency_hz, dtype=float)
    vt = np.asarray(stoneley_velocity, dtype=float)
    if np.any((R < 0.0) | (R > 1.0)):
        raise ValueError("reflection_coefficient must be in [0, 1]")
    if np.any(f <= 0.0):
        raise ValueError("frequency_hz must be strictly positive")
    if np.any(vt <= 0.0):
        raise ValueError("stoneley_velocity must be strictly positive")

    omega = 2.0 * np.pi * f
    if small_amplitude_approx:
        return vt * R / omega

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = omega * np.sqrt(np.clip(1.0 - R * R, 0.0, None))
        aperture = np.where(denom > 0.0, vt * R / denom, np.inf)
    return aperture


def stoneley_fracture_density(
    slowness_indicator: np.ndarray,
    amplitude_indicator: np.ndarray | None = None,
    matrix_permeability: np.ndarray | None = None,
    fracture_aperture: np.ndarray | None = None,
    *,
    slowness_weight: float = 0.5,
    amplitude_weight: float = 0.5,
    aperture_weight: float = 0.0,
    slowness_scale: float = 0.1,
    aperture_scale: float = 1.0e-3,
) -> np.ndarray:
    r"""
    Unified Stoneley fracture-density log from the four indicator family.

    Combines the dimensionless slowness and amplitude indicators with
    the Tang-Cheng-Toksoz matrix-permeability inversion and the
    Hornby fracture-aperture inversion into a single per-depth
    fracture-intensity score in ``[0, 1]``. Pure combiner: no new
    physics, just a calibrated mixture of the four primitive
    quantities the rest of the module computes.

    The score is heuristic, not a calibrated geomechanical fracture
    density. Callers wanting quantitative fracture-density work
    should use the four primitive quantities directly.

    Definition
    ----------
    Let :math:`\alpha_S` = ``slowness_indicator`` (dimensionless
    fractional Stoneley-slowness shift; output of
    :func:`stoneley_permeability_indicator`),
    :math:`\alpha_A` = ``amplitude_indicator`` (fractional
    amplitude deficit; output of
    :func:`stoneley_amplitude_fracture_indicator`), and let
    ``L_0`` = ``fracture_aperture`` (fracture aperture in metres;
    output of :func:`hornby_fracture_aperture`).

    The fracture-only slowness contribution :math:`\alpha_S^{(F)}`
    depends on whether matrix-permeability information is supplied:

    * No ``matrix_permeability``: :math:`\alpha_S^{(F)} = \alpha_S`
      (full slowness shift counts).
    * With ``matrix_permeability`` (output of
      :func:`stoneley_permeability_tang_cheng`): the shift is
      "explained" by matrix flow at depths where the inversion
      returned a finite kappa, and fracture-driven where it
      returned NaN (out-of-model = simplified Biot-Rosenbaum
      cannot account for the observed shift). So
      :math:`\alpha_S^{(F)} = \alpha_S` where ``matrix_permeability``
      is NaN, ``0`` elsewhere.

    The fracture-density score is then

    .. math::

        \mathrm{FI} \;=\; \mathrm{clip}\!\Big(
            w_s \cdot \frac{\alpha_S^{(F)}}{\sigma_s}
            \;+\; w_a \cdot \alpha_A
            \;+\; w_\kappa \cdot \tanh\!\big(L_0 / \sigma_\kappa\big),
            \;0,\;1
        \Big),

    with weights :math:`w_s, w_a, w_\kappa` (default 0.5, 0.5, 0.0
    -- aperture is opt-in because Hornby data is not always
    available) and scaling parameters
    :math:`\sigma_s` = ``slowness_scale`` (default 0.1; a 10%
    slowness shift saturates the slowness contribution) and
    :math:`\sigma_\kappa` = ``aperture_scale`` (default 1 mm).

    Aperture contribution uses ``tanh`` to saturate gracefully:
    a 1 mm aperture contributes ~0.76 to the un-weighted aperture
    term; 5 mm saturates at ~1.0.

    Parameters
    ----------
    slowness_indicator : ndarray, shape (n_depths,)
        Dimensionless fractional Stoneley-slowness shift, e.g.
        ``stoneley_permeability_indicator(s_obs, s_ref)``.
    amplitude_indicator : ndarray, shape (n_depths,), optional
        Dimensionless amplitude deficit, e.g.
        ``stoneley_amplitude_fracture_indicator(A_obs, A_ref)``.
        If omitted, the amplitude term contributes zero.
    matrix_permeability : ndarray, shape (n_depths,), optional
        Per-depth matrix permeability in m^2 (e.g.
        ``stoneley_permeability_tang_cheng(...)``). Used to
        partition the slowness shift into matrix-explained and
        fracture-driven components. NaN entries flag fracture-
        suspected depths; finite entries flag matrix-explained
        depths. If omitted, the slowness term uses the raw
        :math:`\alpha_S` without partitioning.
    fracture_aperture : ndarray, shape (n_depths,), optional
        Per-depth fracture aperture in metres
        (``hornby_fracture_aperture``). Contributes only if
        ``aperture_weight > 0``.
    slowness_weight, amplitude_weight, aperture_weight : float
        Mixing weights (default 0.5 / 0.5 / 0.0). Negative weights
        are not allowed.
    slowness_scale : float, default 0.1
        Slowness-indicator value at which the slowness term reaches
        the value 1.0 before clipping. The default 0.1 corresponds
        to a 10% slowness shift -- typical for moderately
        permeable / lightly fractured zones.
    aperture_scale : float, default 1e-3
        Aperture (m) at which the ``tanh`` aperture term has the
        value :math:`\tanh(1) \approx 0.76`. Default 1 mm matches
        the typical low-frequency Hornby-inversion sensitivity band.

    Returns
    -------
    ndarray, shape (n_depths,)
        Per-depth fracture-density score in ``[0, 1]``. Zero where
        all indicators are zero. Increasing values flag stronger
        fracture activity.

    Raises
    ------
    ValueError
        If any non-negative weight is negative; if any positive
        scale is non-positive; if any optional array shape
        disagrees with ``slowness_indicator``.

    See Also
    --------
    stoneley_permeability_indicator : The primitive that returns
        :math:`\alpha_S`.
    stoneley_amplitude_fracture_indicator : The primitive that
        returns :math:`\alpha_A`.
    stoneley_permeability_tang_cheng : The matrix-permeability
        inversion this combiner uses to partition the slowness shift.
    hornby_fracture_aperture : The reflection-coefficient inversion
        this combiner consumes via the aperture term.

    Notes
    -----
    The matrix-subtraction logic uses a binary in/out-of-model
    flag rather than an explicit "fracture excess slowness"
    calculation. The TCT inversion saturates the model when
    :math:`\alpha_S \ge K_f / (2 K_\phi)`, so any depth where it
    returns NaN is one where matrix flow alone cannot account for
    the observed shift -- structurally equivalent to flagging that
    depth as fracture-suspected. A continuous "fracture excess"
    formulation would require running the forward model with the
    inverted kappa, which is circular.

    The default weights (0.5 slowness + 0.5 amplitude, no aperture)
    treat slowness and amplitude as equally informative. In
    practice the amplitude indicator is more fracture-specific
    because matrix Darcy flow is largely energy-conservative at
    low frequency, while it absolutely produces a slowness shift.
    Callers may want to bump ``amplitude_weight`` toward 0.7-0.8
    when the amplitude data is reliable, or set
    ``matrix_permeability`` to suppress the matrix-explained
    slowness contribution entirely.

    References
    ----------
    * Tang, X.-M., & Cheng, A. (2004). *Quantitative Borehole
      Acoustic Methods*, Section 5.2 (joint slowness-amplitude
      Stoneley analysis as a permeable-zone discriminator).
    * Hornby, B. E., Johnson, D. L., Winkler, K. W., & Plumb,
      R. A. (1989). Fracture evaluation using reflected
      Stoneley-wave arrivals. *Geophysics* 54(10), 1274-1288.
    """
    if slowness_weight < 0:
        raise ValueError("slowness_weight must be non-negative")
    if amplitude_weight < 0:
        raise ValueError("amplitude_weight must be non-negative")
    if aperture_weight < 0:
        raise ValueError("aperture_weight must be non-negative")
    if slowness_scale <= 0:
        raise ValueError("slowness_scale must be positive")
    if aperture_scale <= 0:
        raise ValueError("aperture_scale must be positive")

    alpha_s = np.asarray(slowness_indicator, dtype=float)
    n = alpha_s.shape

    def _check(name: str, arr: np.ndarray) -> np.ndarray:
        out = np.asarray(arr, dtype=float)
        if out.shape != n:
            raise ValueError(
                f"{name} shape {out.shape} does not match slowness_indicator shape {n}"
            )
        return out

    if matrix_permeability is not None:
        kappa = _check("matrix_permeability", matrix_permeability)
        # Fracture-only slowness: full alpha_s where the matrix
        # model failed (NaN), zero where matrix flow explains the
        # shift (finite kappa).
        alpha_s_frac = np.where(np.isnan(kappa), alpha_s, 0.0)
    else:
        alpha_s_frac = alpha_s

    score = slowness_weight * (alpha_s_frac / slowness_scale)

    if amplitude_indicator is not None:
        alpha_a = _check("amplitude_indicator", amplitude_indicator)
        score = score + amplitude_weight * alpha_a

    if fracture_aperture is not None and aperture_weight > 0:
        L0 = _check("fracture_aperture", fracture_aperture)
        # tanh saturation; treat NaN apertures as no contribution.
        with np.errstate(invalid="ignore"):
            ap_term = np.tanh(np.where(np.isnan(L0), 0.0, L0) / aperture_scale)
        score = score + aperture_weight * ap_term

    return np.clip(score, 0.0, 1.0)


def vs_from_stoneley_slow_formation(
    slowness_stoneley: np.ndarray,
    rho: np.ndarray,
    *,
    rho_fluid: float,
    v_fluid: float,
) -> np.ndarray:
    r"""
    Formation shear velocity :math:`V_S` from low-frequency Stoneley
    slowness.

    The primary sonic-only :math:`V_S` estimator for **slow
    formations** (:math:`V_S < V_\mathrm{fluid}`), where the
    formation has no critically-refracted S head wave to pick on a
    monopole gather (Paillet & Cheng 1991, Ch. 3) and the pseudo-
    Rayleigh / guided trapped mode does not exist
    (:func:`fwap.synthetic.pseudo_rayleigh_dispersion` raises in
    this regime). The Stoneley wave's low-frequency phase velocity
    carries the formation shear-modulus information through the
    classical White (1983) tube-wave formula

    .. math::

        S_\mathrm{ST}^2 \;=\; \frac{1}{V_f^2}
                              \;+\; \frac{\rho_f}{\mu},

    where :math:`\mu = \rho V_S^2` is the formation shear modulus.
    Inverting for :math:`V_S`:

    .. math::

        V_S \;=\; \sqrt{\frac{\rho_f}
                             {\rho \,(S_\mathrm{ST}^2 - 1/V_f^2)}}.

    The formula is identical to the
    :func:`fwap.anisotropy.stoneley_horizontal_shear_modulus`
    expression after dividing by :math:`\rho` and taking the square
    root; the difference is interpretation. For an isotropic
    formation :math:`C_{66} = \mu` and this function returns
    :math:`V_S` directly. For a VTI formation use
    :func:`fwap.anisotropy.stoneley_horizontal_shear_modulus` to get
    :math:`C_{66}` (the horizontal shear modulus, *not* equal to
    :math:`\rho V_{Sv}^2` in general).

    Assumptions
    -----------
    * Low-frequency limit (Stoneley pulse well below the dipole-
      flexural and pseudo-Rayleigh cutoffs). At higher frequencies
      the Stoneley wave is dispersive in slow formations and a full
      cylindrical-mode solver is needed (Paillet & Cheng 1991,
      Ch. 4). The slowness here should therefore be the picked
      Stoneley slowness in a low-pass-filtered or low-frequency
      band.
    * Inviscid borehole fluid; centred tool; circular borehole
      cross-section. Tool-eccentricity and mudcake corrections
      (Tang & Cheng 2004, sect. 5.2) are not applied.
    * Isotropic formation. For VTI shales the inverted modulus is
      :math:`C_{66}`, not :math:`\mu`; the function will then
      systematically under- or over-estimate the *vertical*
      :math:`V_{Sv}` depending on the sign of Thomsen
      :math:`\gamma`. Use
      :func:`fwap.anisotropy.thomsen_gamma_from_logs` when an
      independent dipole shear log is available to detect this case.

    Parameters
    ----------
    slowness_stoneley : ndarray or float
        Per-depth low-frequency Stoneley slowness (s/m). Must be
        strictly greater than ``1 / v_fluid`` everywhere -- the
        Stoneley wave is always slower than the unconfined fluid
        wave because the formation loads it.
    rho : ndarray or float
        Per-depth formation bulk density (kg/m^3) from the bulk-
        density log (typically the ``RHOB`` curve).
    rho_fluid : float
        Borehole-fluid density (kg/m^3). Brine ~ 1000-1100; oil
        ~ 800-900; gas / foam << 1000.
    v_fluid : float
        Borehole-fluid acoustic velocity (m/s). Brine ~ 1500;
        oil ~ 1300; gas << 1000.

    Returns
    -------
    ndarray
        Formation shear velocity :math:`V_S` (m/s), broadcast to the
        common shape of ``slowness_stoneley`` and ``rho``.

    Raises
    ------
    ValueError
        If any input is non-positive, or if any Stoneley slowness is
        at or below the fluid slowness ``1 / v_fluid`` (the
        inversion is undefined there).

    See Also
    --------
    fwap.anisotropy.stoneley_horizontal_shear_modulus :
        Same physics in the VTI shale case; returns :math:`C_{66}`.
    fwap.anisotropy.thomsen_gamma_from_logs :
        When a dipole shear log is also available, combines both to
        flag VTI anisotropy and give the Thomsen :math:`\gamma`
        directly.

    References
    ----------
    * Paillet, F. L., & Cheng, C. H. (1991). *Acoustic Waves in
      Boreholes*, Chapter 3. CRC Press (Stoneley low-f velocity as
      slow-formation Vs estimator).
    * White, J. E. (1983). *Underground Sound: Application of
      Seismic Waves.* Elsevier, Section 5.5 (tube-wave formula).
    * Norris, A. N. (1990). The speed of a tube wave. *J. Acoust.
      Soc. Am.* 87(1), 414-417.
    * Tang, X.-M., & Cheng, A. (2004). *Quantitative Borehole
      Acoustic Methods.* Elsevier, Section 5.2.
    """
    if rho_fluid <= 0.0:
        raise ValueError("rho_fluid must be strictly positive")
    if v_fluid <= 0.0:
        raise ValueError("v_fluid must be strictly positive")
    s_st = np.asarray(slowness_stoneley, dtype=float)
    rho_arr = np.asarray(rho, dtype=float)
    if np.any(s_st <= 0):
        raise ValueError("slowness_stoneley must be strictly positive")
    if np.any(rho_arr <= 0):
        raise ValueError("rho must be strictly positive")
    s_f2 = 1.0 / (v_fluid * v_fluid)
    diff = s_st * s_st - s_f2
    if np.any(diff <= 0.0):
        raise ValueError(
            "slowness_stoneley must exceed 1 / v_fluid everywhere "
            "(Stoneley wave is slower than the unconfined fluid wave); "
            f"got min slowness {float(np.min(s_st)):.3e} s/m, fluid "
            f"slowness {1.0 / v_fluid:.3e} s/m."
        )
    return np.sqrt(rho_fluid / (rho_arr * diff))
