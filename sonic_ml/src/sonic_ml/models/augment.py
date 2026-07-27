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

import numpy as np


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
