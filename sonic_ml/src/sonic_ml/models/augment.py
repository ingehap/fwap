"""
Training-time waveform augmentation for the inverse net.

All sonic_ml training is on synthetic modal-solver gathers with one fixed noise
level and wavelet, so a net can key on idiosyncrasies that will not survive a
distribution shift (noisier logs, a different source). :class:`GatherAugmentation`
perturbs each training gather on the fly -- an SNR sweep (additive noise to a
random signal-to-noise ratio) and optional amplitude jitter -- to narrow that
sim-to-real gap.

.. warning::

   Augmentation reduces the *synthetic* generalization gap; it does **not**
   make a "beats classical" number a real-world claim. On real data neither the
   modal law nor the classical processing is exact, so any deployment claim
   still requires a real (or independently simulated) holdout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Augmentation(Protocol):
    """
    Structural type for a per-gather augmentation.

    Anything with this ``apply`` can be passed as the ``augment=`` argument of
    the inverse datasets, so alternative perturbations (see
    :class:`CasingRingAugmentation`) drop in without touching the datasets.
    """

    def apply(self, gather: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...


@dataclass(frozen=True)
class GatherAugmentation:
    """
    Stochastic per-gather augmentation.

    Attributes
    ----------
    snr_db_range : (float, float) or None, default (10.0, 40.0)
        If set, additive white Gaussian noise is added to bring each gather to
        a signal-to-noise ratio drawn uniformly (in dB) from this range. A
        wider/lower range trains for noisier conditions. ``None`` disables it.
    amp_jitter : float, default 0.0
        If ``> 0``, the gather is scaled by a random factor drawn uniformly
        from ``[1 - amp_jitter, 1 + amp_jitter]`` (amplitude is not a modal
        observable, so the net should be invariant to it).
    """

    snr_db_range: tuple[float, float] | None = (10.0, 40.0)
    amp_jitter: float = 0.0

    def apply(self, gather: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Return an augmented copy of one gather.

        Parameters
        ----------
        gather : ndarray, shape (R, T)
            One raw multi-receiver gather.
        rng : numpy.random.Generator
            Randomness source (seed it for reproducible augmentation).

        Returns
        -------
        ndarray, shape (R, T), float64
        """
        g = np.asarray(gather, dtype=float).copy()
        if self.amp_jitter > 0.0:
            g *= float(rng.uniform(1.0 - self.amp_jitter, 1.0 + self.amp_jitter))
        if self.snr_db_range is not None:
            snr_db = float(rng.uniform(*self.snr_db_range))
            signal_power = float(np.mean(g**2))
            if signal_power > 0.0:
                noise_power = signal_power / (10.0 ** (snr_db / 10.0))
                g = g + rng.normal(scale=float(np.sqrt(noise_power)), size=g.shape)
        return g


@dataclass(frozen=True)
class CasingRingAugmentation:
    """
    Plant a synthetic steel-casing ring arrival on a cased-hole gather.

    A real cased-hole record contains the casing extensional ("ring") arrival --
    the thing field cement-bond logging actually measures. The M5a synthetics do
    not: they are built from the cased Stoneley dispersion alone, because that is
    what fwap's bound-mode layered solver produces. This augmentation adds a
    phenomenological ring -- a non-dispersive Gabor wavetrain at a fixed casing
    slowness, the same modelling stance as :func:`fwap.lwd.lwd_collar_mode` --
    so a cased model can be stress-tested against contamination it never saw in
    training.

    .. important::

       The ring amplitude is drawn **independently of the bond index**, and that
       is a deliberate scientific choice rather than an oversight. Physically a
       poorly bonded casing rings louder, so coupling amplitude to bond would
       manufacture a CBL-like signal in the data. A model then "recovering" bond
       from ring amplitude would only be recovering a relationship hard-coded
       here, and a CBL-amplitude baseline scored on it would be measuring the
       same invention. Keeping the ring bond-independent makes this a clean
       robustness probe: a nuisance arrival that carries no information about
       the target.

    So this narrows the sim-to-real gap in the same limited sense as
    :class:`GatherAugmentation`, and it is **not** a route to free-pipe
    detection -- that needs a leaky-mode forward model, not a planted wavetrain.

    Attributes
    ----------
    slowness : float, default 1/5300 s/m (~58 us/ft)
        Casing extensional apparent slowness (s/m). Steel plate/extensional
        speeds sit well above formation velocities, so the ring arrives early.
    f0 : float, default 12 kHz
        Ring centre frequency (Hz).
    sigma : float, default 1.5e-4 s
        Gabor envelope width (s) -- the ring is narrow-band and long-lived
        relative to a formation arrival.
    amplitude_range : (float, float), default (0.2, 1.5)
        Ring amplitude as a fraction of the gather's RMS, drawn uniformly.
    dt : float, default 1e-5
        Sampling interval (s) of the gathers being augmented.
    tr_offset, dr : float
        First-receiver offset and receiver spacing (m), used to give the ring
        the correct move-out across the array.
    """

    slowness: float = 1.0 / 5300.0
    f0: float = 12_000.0
    sigma: float = 1.5e-4
    amplitude_range: tuple[float, float] = (0.2, 1.5)
    dt: float = 1.0e-5
    tr_offset: float = 3.0
    dr: float = 0.1524

    def apply(self, gather: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Return a copy of ``gather`` with a casing-ring arrival added.

        Parameters
        ----------
        gather : ndarray, shape (R, T)
            One raw multi-receiver gather.
        rng : numpy.random.Generator
            Randomness source (seed it for reproducible augmentation).

        Returns
        -------
        ndarray, shape (R, T), float64
        """
        g = np.asarray(gather, dtype=float).copy()
        n_rec, n_samples = g.shape
        t = np.arange(n_samples) * self.dt
        offsets = self.tr_offset + self.dr * np.arange(n_rec)
        rms = float(np.sqrt(np.mean(g**2))) + 1.0e-12
        amplitude = rms * float(rng.uniform(*self.amplitude_range))

        arrival = offsets * self.slowness
        envelope = np.exp(-0.5 * ((t[None, :] - arrival[:, None]) / self.sigma) ** 2)
        ring = np.cos(2.0 * np.pi * self.f0 * (t[None, :] - arrival[:, None]))
        return g + amplitude * envelope * ring
