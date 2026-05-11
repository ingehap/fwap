"""
Dip and azimuth estimation from a borehole ring array.

Implements Part 4 of the book ("Dip measurement based on acoustic
data"). A ring array sees a dipping bed as an azimuthally cosine-
varying arrival; maximising stack coherence over (dip, azimuth) after
removing that cosine yields the structural dip and dip direction at
each depth.

References
----------
* Mari, J.-L., Coppens, F., Gavin, P., & Wicquart, E. (1994).
  *Full Waveform Acoustic Data Processing*, Part 4. Editions Technip,
  Paris. ISBN 978-2-7108-0664-6.
* Hsu, K., & Esmersoy, C. (1992). Parametric estimation of phase and
  group slownesses from sonic logging waveforms. *Geophysics* 57(8),
  978-985 (azimuthal coherence framework).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize

from fwap.coherence import semblance


class AzimuthalGather(NamedTuple):
    """
    Output of :func:`synthesize_azimuthal_arrival`.

    A ``NamedTuple`` so that existing tuple-unpacking call sites keep
    working while new code can access fields by name.

    Fields
    ------
    data : ndarray, shape (n_rec, n_samples)
        Synthetic azimuthal gather.
    dt : float
        Sampling interval (s).
    axial_offsets : ndarray, shape (n_rec,)
        All zeros -- the ring-array model places every receiver on a
        common transverse ring.
    azimuths : ndarray, shape (n_rec,)
        Receiver azimuths (rad), uniformly spaced in ``[0, 2*pi)``.
    tool_radius : float
        Ring radius (m).
    slowness : float
        Formation slowness of the arrival (s/m).
    """

    data: np.ndarray
    dt: float
    axial_offsets: np.ndarray
    azimuths: np.ndarray
    tool_radius: float
    slowness: float


@dataclass
class DipResult:
    """
    Output of :func:`estimate_dip`.

    Attributes
    ----------
    dip : float
        Estimated bed dip (rad), ``dip in [0, pi/2]``.
    azimuth : float
        Estimated dip azimuth (rad), folded to ``(-pi, pi]``. Measured
        from the reference direction of the ``azimuths`` array passed
        to :func:`estimate_dip`.
    coherence : float
        Stack coherence at the recovered ``(dip, azimuth)``; in
        ``[0, 1]``, 1 = perfectly aligned after de-tilt.
    surface : ndarray, shape (n_dip, n_az)
        Coherence evaluated on the full coarse grid.
    dip_axis : ndarray, shape (n_dip,)
        Trial dip angles (rad) corresponding to the rows of
        ``surface``.
    azimuth_axis : ndarray, shape (n_az,)
        Trial azimuth angles (rad, in ``[-pi, pi)``) corresponding to
        the columns of ``surface``.
    refined : bool
        ``True`` if the returned ``(dip, azimuth)`` came from the
        Nelder-Mead polish; ``False`` if it is the raw grid maximum.
    """

    dip: float
    azimuth: float
    coherence: float
    surface: np.ndarray
    dip_axis: np.ndarray
    azimuth_axis: np.ndarray
    refined: bool = False


def _make_detilt_evaluator(
    data: np.ndarray,
    dt: float,
    azimuths: np.ndarray,
    tool_radius: float,
    slowness: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    int,
]:
    """
    Pre-compute the pieces of :func:`_coherence_after_detilt` that do
    not depend on ``(dip, az)``.

    Returns ``(spec, freqs, n_samp)`` where ``spec = rfft(data)``.
    Hoisting these out of the grid-search inner loop cuts the
    per-candidate work of :func:`estimate_dip` from one rFFT + one
    iFFT to just one iFFT.
    """
    n_samp = data.shape[1]
    freqs = np.fft.rfftfreq(n_samp, d=dt)
    spec = np.fft.rfft(data, axis=1)
    return spec, freqs, n_samp


def _detilt_coherence_scalar(
    spec: np.ndarray,
    freqs: np.ndarray,
    n_samp: int,
    azimuths: np.ndarray,
    tool_radius: float,
    slowness: float,
    dip: float,
    az: float,
) -> float:
    """
    Stack coherence at a single ``(dip, az)`` using pre-computed
    ``spec`` / ``freqs`` (see :func:`_make_detilt_evaluator`).
    """
    dts = slowness * tool_radius * np.sin(dip) * np.cos(azimuths - az)
    phase = np.exp(1j * 2.0 * np.pi * np.outer(dts, freqs))
    shifted = np.fft.irfft(spec * phase, n=n_samp, axis=1)
    # Delegate the actual stack-coherence ratio to the shared semblance
    # helper so this module and the STC surface cannot disagree on the
    # definition. semblance returns NaN for an all-zero window; the
    # optimiser would reject that branch anyway, but we coerce to 0.0
    # so the coarse-grid surface stays finite.
    rho = semblance(shifted)
    return 0.0 if np.isnan(rho) else float(rho)


def _detilt_coherence_az_row(
    spec: np.ndarray,
    freqs: np.ndarray,
    n_samp: int,
    azimuths: np.ndarray,
    tool_radius: float,
    slowness: float,
    dip: float,
    phis: np.ndarray,
) -> np.ndarray:
    r"""
    Stack coherence at a single ``dip`` for every azimuth in ``phis``.

    Vectorised over azimuth: builds a single ``(n_az, n_rec, n_freq)``
    phase tensor and produces an ``(n_az,)`` coherence vector in one
    broadcast pass instead of looping one azimuth at a time. This is
    the hot inner loop of :func:`estimate_dip`'s grid sweep.
    """
    n_rec = azimuths.size
    # delta_phi[j, i] = azimuths[i] - phis[j]
    delta = azimuths[None, :] - phis[:, None]  # (n_az, n_rec)
    dts = slowness * tool_radius * np.sin(dip) * np.cos(delta)  # (n_az, n_rec)
    phase = np.exp(
        1j * 2.0 * np.pi * dts[:, :, None] * freqs[None, None, :]  # (n_az, n_rec, n_f)
    )
    shifted_spec = phase * spec[None, :, :]  # (n_az, n_rec, n_f)
    shifted = np.fft.irfft(shifted_spec, n=n_samp, axis=-1)  # (n_az, n_rec, n_samp)

    stack = shifted.sum(axis=1)  # (n_az, n_samp)
    num = (stack**2).sum(axis=-1)  # (n_az,)
    den = n_rec * (shifted**2).sum(axis=(1, 2))  # (n_az,)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(den > 0, num / den, 0.0)
    rho = np.where(np.isnan(rho), 0.0, rho)
    return rho


def _coherence_after_detilt(
    data: np.ndarray,
    dt: float,
    azimuths: np.ndarray,
    tool_radius: float,
    slowness: float,
    dip: float,
    az: float,
) -> float:
    """
    Stack coherence of an azimuthal ring array after removing the
    cosine-of-azimuth time shift expected for a dipping bed.

    For a bed with dip ``alpha`` and strike-perpendicular azimuth
    ``phi``, the arrival time at a receiver at azimuth ``phi_i`` and
    tool radius ``a`` carries an extra delay
    ``a * s * sin(alpha) * cos(phi_i - phi)``.

    Legacy scalar entry point retained for third-party callers; the
    grid sweep inside :func:`estimate_dip` uses the hoisted FFT
    helpers :func:`_make_detilt_evaluator` and
    :func:`_detilt_coherence_az_row` directly.
    """
    spec, freqs, n_samp = _make_detilt_evaluator(
        data, dt, azimuths, tool_radius, slowness
    )
    return _detilt_coherence_scalar(
        spec, freqs, n_samp, azimuths, tool_radius, slowness, dip, az
    )


def estimate_dip(
    data: np.ndarray,
    dt: float,
    axial_offsets: np.ndarray,
    azimuths: np.ndarray,
    tool_radius: float,
    slowness: float,
    dip_range: tuple[float, float] = (0.0, np.deg2rad(60.0)),
    n_dip: int = 31,
    n_az: int = 36,
    refine: bool = True,
) -> DipResult:
    """
    Estimate (dip, azimuth) maximising de-tilted array coherence.

    Parameters
    ----------
    data : ndarray, shape (n_rec, n_samples)
        Azimuthal ring-array gather at a single depth.
    dt : float
        Sampling interval (s).
    axial_offsets : ndarray, shape (n_rec,)
        **Ignored.** The ring-array dip model (Part 4 of Mari et al.,
        1994) treats all receivers as lying on a common transverse
        ring; only the azimuthal spread and the tool radius enter the
        cosine-of-azimuth moveout expected for a dipping bed. The
        parameter is retained so that callers feeding the
        ``synthesize_azimuthal_arrival`` tuple do not need to reshape
        their call sites, and so that a future mixed axial/ring array
        can reuse the same signature.
    azimuths : ndarray, shape (n_rec,)
        Receiver azimuth (rad) around the tool, measured from an
        arbitrary reference.
    tool_radius : float
        Radius (m) of the receiver ring.
    slowness : float
        Formation slowness (s/m) of the arrival whose azimuthal
        moveout is being analysed.
    dip_range : (float, float)
        Lower and upper dip angles to scan (rad).
    n_dip, n_az : int
        Resolution of the coarse grid over (dip, azimuth).
    refine : bool
        If ``True``, polish the grid maximum with Nelder-Mead.

    Strategy
    --------
    1. Coarse grid search over ``(alpha, phi)``.
    2. If ``refine=True``, Nelder-Mead polish starting at the grid
       maximum. The coherence surface near the true maximum is smooth
       and unimodal, so a simplex method converges quickly.

    Returns
    -------
    DipResult
    """
    del axial_offsets  # see docstring for rationale
    alphas = np.linspace(*dip_range, n_dip)
    phis = np.linspace(-np.pi, np.pi, n_az, endpoint=False)

    # Hoist the one-off rFFT of the gather out of the (n_dip, n_az)
    # grid sweep so each candidate cell only pays for one iFFT plus
    # the cheap phase multiplication, and vectorise the inner loop
    # over azimuth at fixed dip.
    spec, freqs, n_samp = _make_detilt_evaluator(
        data, dt, azimuths, tool_radius, slowness
    )
    surf = np.zeros((n_dip, n_az))
    for i, a in enumerate(alphas):
        surf[i, :] = _detilt_coherence_az_row(
            spec, freqs, n_samp, azimuths, tool_radius, slowness, a, phis
        )

    idx = np.unravel_index(np.argmax(surf), surf.shape)
    dip0 = float(alphas[idx[0]])
    az0 = float(phis[idx[1]])
    coh0 = float(surf[idx])
    refined = False

    if refine:

        def neg_coh(x: np.ndarray) -> float:
            a, p = x
            if a < 0 or a > np.pi / 2:
                return 1.0
            return -_detilt_coherence_scalar(
                spec, freqs, n_samp, azimuths, tool_radius, slowness, a, p
            )

        res = minimize(
            neg_coh,
            x0=np.array([dip0, az0]),
            method="Nelder-Mead",
            options=dict(xatol=1.0e-4, fatol=1.0e-5),
        )
        if res.success and -res.fun >= coh0:
            dip0, az0 = float(res.x[0]), float(res.x[1])
            # Fold azimuth back to (-pi, pi].
            az0 = (az0 + np.pi) % (2 * np.pi) - np.pi
            coh0 = float(-res.fun)
            refined = True

    return DipResult(
        dip=dip0,
        azimuth=az0,
        coherence=coh0,
        surface=surf,
        dip_axis=alphas,
        azimuth_axis=phis,
        refined=refined,
    )


def synthesize_azimuthal_arrival(
    n_rec: int = 8,
    n_samples: int = 1024,
    dt: float = 2.0e-5,
    tool_radius: float = 0.08,
    slowness: float = 1.0 / 4000.0,
    dip: float = np.deg2rad(30.0),
    azimuth: float = np.deg2rad(45.0),
    f0: float = 8000.0,
    noise: float = 0.02,
    seed: int = 0,
):
    """
    Synthetic azimuthal ring-array arrival with a dipping-bed signature.

    Builds a single-mode gather in which receiver ``i`` sees an arrival
    at ``t0 + slowness * tool_radius * sin(dip) * cos(az_i - azimuth)``
    -- the cosine-of-azimuth moveout that :func:`estimate_dip` inverts
    for.

    Parameters
    ----------
    n_rec : int, default 8
        Number of receivers on the ring.
    n_samples : int, default 1024
        Samples per trace.
    dt : float, default 2.0e-5
        Sampling interval (s).
    tool_radius : float, default 0.08
        Ring radius (m).
    slowness : float, default 1/4000
        Formation slowness of the arrival (s/m).
    dip : float, default deg2rad(30)
        True bed dip (rad).
    azimuth : float, default deg2rad(45)
        True dip azimuth (rad).
    f0 : float, default 8000
        Ricker peak frequency (Hz).
    noise : float, default 0.02
        Gaussian-noise RMS relative to the gather RMS.
    seed : int, default 0
        Seed for the noise RNG.

    Returns
    -------
    AzimuthalGather
        Named 6-tuple with fields ``data``, ``dt``, ``axial_offsets``,
        ``azimuths``, ``tool_radius``, ``slowness``. Tuple-unpacking
        at the call site continues to work:
        ``data, dt, ax_off, az, a, slow = synthesize_azimuthal_arrival()``.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) * dt
    az = np.linspace(0.0, 2.0 * np.pi, n_rec, endpoint=False)
    ax_off = np.zeros(n_rec)
    t0 = 0.4e-3
    data = np.zeros((n_rec, n_samples))
    for i in range(n_rec):
        dt_az = slowness * tool_radius * np.sin(dip) * np.cos(az[i] - azimuth)
        tc = t0 + dt_az
        a = (np.pi * f0 * (t - tc)) ** 2
        data[i] = (1.0 - 2.0 * a) * np.exp(-a)
    rms = np.sqrt(np.mean(data**2)) + 1e-12
    data += rng.normal(scale=noise * rms, size=data.shape)
    return AzimuthalGather(
        data=data,
        dt=dt,
        axial_offsets=ax_off,
        azimuths=az,
        tool_radius=tool_radius,
        slowness=slowness,
    )
