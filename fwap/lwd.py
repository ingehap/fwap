"""
Logging-While-Drilling (LWD) acoustic processing -- phenomenological layer.

LWD acoustic tools record waveforms while a steel drill collar is still
inside the borehole. The collar supports its own propagating modes
(flexural, longitudinal, quadrupole) that contaminate the formation
arrivals -- Tang & Cheng (2004), sect. 2.4-2.5 emphasise this as the
defining processing problem of the LWD era.

This module provides a **phenomenological** layer for that contamination:

* :func:`lwd_collar_mode` returns a pre-configured :class:`Mode`
  representing the collar arrival as a Gabor-windowed wavetrain at a
  fixed apparent slowness. This is *not* a true cylindrical-Biot
  collar-mode forward solver -- it captures the dominant contamination
  signature in slowness-time pickers (a strong, narrow-band arrival at
  the collar slowness) without modelling the collar's full dispersion.

* :func:`synthesize_lwd_gather` is a convenience wrapper around
  :func:`fwap.synthetic.synthesize_gather` that adds the collar
  arrival on top of a list of formation modes.

* :func:`notch_slowness_band` removes a slowness band from a
  multichannel gather using a tau-p forward / mask / inverse round-
  trip. Its main use case is collar-mode rejection: notch the band
  around the known collar slowness and re-run the picker / dispersion
  estimator on the cleaned record.

For a quantitative LWD inversion (e.g. recovering shear slowness from
the LWD quadrupole flexural mode in a slow formation) a true layered-
cylindrical-Biot solver over (collar, mud annulus, formation) is
required. That is research-grade work and outside the scope of fwap
today; the phenomenological layer here is enough to test the
processing chain on collar-contaminated synthetics.

References
----------
* Tang, X.-M., & Cheng, A. (2004). *Quantitative Borehole Acoustic
  Methods.* Elsevier, sect. 2.4-2.5 (LWD modes; quadrupole source as
  the practical solution to collar-mode contamination).
* Aron, J., Chang, S. K., Codazzi, D., Dworak, R., Hsu, K., Lau, T.,
  Minerbo, G., & Yogeswaren, E. (1994). Real-time sonic logging while
  drilling in hard and soft rocks. *SEG Technical Program Expanded
  Abstracts*, 13, 1-4.
* Kinoshita, T., Endo, T., Iwasaki, S., Saito, S., Mori, K., Inoue, T.,
  Yoshikawa, R., & Hiraga, A. (2008). Real-time deep-shear sonic-while-
  drilling in fast and slow formations. *SPWLA 49th Annual Logging
  Symposium*.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from fwap.synthetic import ArrayGeometry, Mode, synthesize_gather
from fwap.wavesep import tau_p_adjoint, tau_p_forward

# Realistic monopole-collar arrival defaults.
#
# The LWD steel-drill-collar longitudinal/flexural modes typically peak
# at 80-130 us/ft (1/3700-1/2300 m/s) once mud loading lowers the bare-
# steel speed; the source-signature center frequency sits in the
# 8-15 kHz band for current-generation tools (Aron et al. 1994;
# Kinoshita et al. 2008). The defaults below land in the middle of
# both ranges.
DEFAULT_COLLAR_SLOWNESS_S_PER_M: float = 1.0 / 3300.0   # ~92 us/ft
DEFAULT_COLLAR_FREQUENCY_HZ: float = 12_000.0
DEFAULT_COLLAR_GABOR_SIGMA_S: float = 1.5e-4   # narrow-band envelope


def lwd_collar_mode(
    *,
    apparent_slowness: float = DEFAULT_COLLAR_SLOWNESS_S_PER_M,
    f0: float = DEFAULT_COLLAR_FREQUENCY_HZ,
    sigma: float = DEFAULT_COLLAR_GABOR_SIGMA_S,
    amplitude: float = 1.5,
    intercept: float = 0.0,
) -> Mode:
    r"""
    Pre-configured :class:`Mode` representing the LWD steel-collar
    arrival.

    Phenomenological model: a non-dispersive Gabor wavetrain at a
    fixed apparent slowness, narrow-band around ``f0``. The collar
    mode is in reality weakly dispersive and modally complex
    (longitudinal + flexural + screw modes overlap on a monopole
    record), but for processing-chain testing the dominant
    contamination signature on a slowness-time-coherence map is a
    single strong narrow-band peak at the collar slowness; that is
    what this :class:`Mode` produces.

    Use the returned :class:`Mode` either alongside formation modes
    in a call to :func:`fwap.synthetic.synthesize_gather`, or via
    the convenience wrapper :func:`synthesize_lwd_gather`.

    Parameters
    ----------
    apparent_slowness : float, default 1/3300 (~92 us/ft)
        Collar apparent slowness (s/m). The LWD steel-collar
        longitudinal/flexural mode is typically observed in the
        80-130 us/ft band; 92 us/ft is the published median across
        recent generations of tools.
    f0 : float, default 12 kHz
        Source-signature centre frequency (Hz). Current-generation
        LWD acoustic tools sit in the 8-15 kHz band.
    sigma : float, default 1.5e-4 s
        Gabor envelope width (s). The narrow envelope captures the
        finite-duration character of the collar wavetrain compared
        to a broadband formation P or S impulse.
    amplitude : float, default 1.5
        Per-trace amplitude multiplier. The default places the
        collar arrival at roughly the same level as the strongest
        formation mode, the contamination regime where rejection
        processing is most useful.
    intercept : float, default 0.0
        Absolute time offset (s) at which the collar wavetrain
        appears at zero offset; carry the same value as the
        formation modes for a coherent gather.

    Returns
    -------
    Mode
        Ready to drop into the ``modes`` list of
        :func:`fwap.synthetic.synthesize_gather` or
        :func:`synthesize_lwd_gather`.
    """
    return Mode(
        name="Collar",
        slowness=apparent_slowness,
        f0=f0,
        amplitude=amplitude,
        intercept=intercept,
        wavelet="gabor",
        sigma=sigma,
    )


def synthesize_lwd_gather(
    geom: ArrayGeometry,
    formation_modes: Sequence[Mode],
    *,
    collar_amplitude: float = 1.5,
    collar_slowness: float = DEFAULT_COLLAR_SLOWNESS_S_PER_M,
    collar_f0: float = DEFAULT_COLLAR_FREQUENCY_HZ,
    collar_intercept: float = 0.0,
    collar_sigma: float = DEFAULT_COLLAR_GABOR_SIGMA_S,
    noise: float = 0.02,
    seed: int | None = None,
) -> np.ndarray:
    r"""
    Build a multichannel LWD gather: formation arrivals + a planted
    steel-collar wavetrain.

    Convenience wrapper around :func:`fwap.synthetic.synthesize_gather`
    that adds an :func:`lwd_collar_mode` to the supplied formation
    modes before synthesis. Lets users exercise the picker / wave-
    separation / dispersion-estimation pipeline against a synthetic
    LWD record without hand-building the collar :class:`Mode`.

    Parameters
    ----------
    geom : ArrayGeometry
        Tool geometry (offsets, sampling) -- see
        :class:`fwap.synthetic.ArrayGeometry`.
    formation_modes : sequence of Mode
        Formation arrivals to synthesise (e.g. the output of
        :func:`fwap.synthetic.monopole_formation_modes`). The collar
        arrival is appended to this list.
    collar_amplitude, collar_slowness, collar_f0, collar_intercept,
    collar_sigma
        Parameters of the planted collar :class:`Mode`. See
        :func:`lwd_collar_mode` for the conventions.
    noise : float, default 0.02
        Per-trace Gaussian noise standard deviation as a fraction of
        the noise-free RMS, identical to
        :func:`fwap.synthetic.synthesize_gather`.
    seed : int or None
        Seed for the noise RNG. Pass ``int`` for reproducible
        synthetics (test gathers) and ``None`` for non-deterministic
        noise (visual demos).

    Returns
    -------
    ndarray, shape (n_rec, n_samples)
        Common-source LWD gather, real-valued, ``float64``.
    """
    collar = lwd_collar_mode(
        apparent_slowness=collar_slowness,
        f0=collar_f0,
        sigma=collar_sigma,
        amplitude=collar_amplitude,
        intercept=collar_intercept,
    )
    return synthesize_gather(
        geom, list(formation_modes) + [collar],
        noise=noise, seed=seed,
    )


def notch_slowness_band(
    data: np.ndarray,
    dt: float,
    offsets: np.ndarray,
    slow_min: float,
    slow_max: float,
    *,
    n_slowness: int = 181,
    slowness_pad_factor: float = 1.5,
    taper_width: float = 0.1,
) -> np.ndarray:
    r"""
    Notch out an apparent-slowness band from a multichannel gather.

    Inverse of :func:`fwap.wavesep.tau_p_filter`: forward-transforms
    the gather to (tau, p), zeroes the slowness band
    ``[slow_min, slow_max]`` (with cosine-tapered edges), and projects
    the residual back to (t, x) via the tau-p adjoint. The result is
    the input gather with energy at apparent slownesses inside the
    notch band attenuated.

    Primary use case: **LWD collar-mode rejection.** When the LWD
    steel-collar arrival sits at a known apparent slowness
    (typically 80-130 us/ft per :data:`DEFAULT_COLLAR_SLOWNESS_S_PER_M`),
    notching that band before running the picker / dispersion
    estimator removes the dominant collar contamination from the
    record while preserving the formation arrivals at slownesses
    outside the band.

    Notes
    -----
    * Implementation: forward-transform to (tau, p), keep **only** the
      notched band via a cosine-tapered band-PASS mask, project that
      back to (t, x) via the tau-p adjoint, and **subtract** from the
      original data. This subtraction route preserves any signal at
      slownesses *outside* the tau-p grid (e.g. Stoneley arrivals at
      slownesses well above the collar band), which a naive band-stop
      mask + adjoint round-trip would destroy because the adjoint
      cannot reconstruct out-of-grid energy.
    * Uses the tau-p adjoint rather than the LSQR inverse; the
      inverse does not commute with masking and would amplify the
      band rather than reconstruct it.
    * Amplitudes inside the band are attenuated by ~50 % (the
      slant-stack point-spread function is non-unitary). Deeper
      rejection would need a multi-pass implementation or a different
      filter family.
    * The notch is **band-stop**, not "ideal brick-wall"; choose
      ``taper_width`` according to how clean a separation is needed.

    Parameters
    ----------
    data : ndarray, shape (n_rec, n_samples)
        Multichannel gather (real-valued).
    dt : float
        Time sampling interval (s).
    offsets : ndarray, shape (n_rec,)
        Source-to-receiver offsets (m). Need not be uniform.
    slow_min, slow_max : float
        Notched apparent-slowness band edges (s/m). Must satisfy
        ``0 < slow_min < slow_max``.
    n_slowness : int, default 181
        Number of slowness samples in the tau-p grid.
    slowness_pad_factor : float, default 1.5
        Extra slowness coverage outside the notch, as a multiple of
        the notch width. Carried over from :func:`tau_p_filter` for
        consistent grid behaviour.
    taper_width : float, default 0.1
        Half-cosine taper width on each notch edge, expressed as a
        fraction of the notch width. ``0.0`` gives a hard
        rectangular notch.

    Returns
    -------
    ndarray, shape (n_rec, n_samples)
        Gather with the notched band attenuated.

    Raises
    ------
    ValueError
        If ``slow_min`` / ``slow_max`` are mis-ordered or non-
        positive.

    See Also
    --------
    fwap.wavesep.tau_p_filter :
        Band-pass companion (keep a slowness band, drop everything
        else).
    """
    if not (slow_max > slow_min > 0):
        raise ValueError("require 0 < slow_min < slow_max")
    pad = slowness_pad_factor * (slow_max - slow_min)
    s_lo = max(slow_min - pad, 1.0e-12)
    s_hi = slow_max + pad
    slownesses = np.linspace(s_lo, s_hi, n_slowness)

    panel = tau_p_forward(data, dt, offsets, slownesses)

    # Cosine-tapered band-PASS mask (1 inside the notch, 0 outside).
    # We reconstruct the in-band component and subtract it from the
    # original data so out-of-grid slownesses pass through unchanged.
    mask = np.zeros(n_slowness, dtype=float)
    in_band = (slownesses >= slow_min) & (slownesses <= slow_max)
    mask[in_band] = 1.0
    if taper_width > 0:
        w = taper_width * (slow_max - slow_min)
        # Tapers ramp from 0 (outside the notch) to 1 (at the notch
        # edge). Lower side: (slow_min - w, slow_min). Upper side:
        # (slow_max, slow_max + w).
        lo = (slownesses >= slow_min - w) & (slownesses < slow_min)
        hi = (slownesses > slow_max) & (slownesses <= slow_max + w)
        mask[lo] = 0.5 * (
            1.0 - np.cos(np.pi * (slownesses[lo] - (slow_min - w)) / w)
        )
        mask[hi] = 0.5 * (
            1.0 + np.cos(np.pi * (slownesses[hi] - slow_max) / w)
        )

    in_band_signal = tau_p_adjoint(
        panel * mask[:, None], dt, offsets, slownesses,
    )
    return data - in_band_signal
