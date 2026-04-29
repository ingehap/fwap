"""
Synthetic multi-receiver borehole gathers.

Provides the canonical P / S / Stoneley monopole gather and the
phenomenological dipole flexural dispersion law used by the demos and
unit tests. The default :class:`ArrayGeometry` matches the original
Schlumberger Array Sonic specification (8 receivers, 10 ft source-to-
first-receiver, 6 in spacing) used throughout Mari et al. (1994).

References
----------
* Mari, J.-L., Coppens, F., Gavin, P., & Wicquart, E. (1994).
  *Full Waveform Acoustic Data Processing.* Editions Technip, Paris.
  ISBN 978-2-7108-0664-6.
* Paillet, F. L., & Cheng, C. H. (1991). *Acoustic Waves in Boreholes*,
  Chapter 4. CRC Press (cylindrical-mode dispersion).
* Schmitt, D. P. (1988). Shear-wave logging in elastic formations.
  *Journal of the Acoustical Society of America* 84(6), 2230-2244.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Callable

import numpy as np


def ricker(t: np.ndarray, f0: float, t0: float = 0.0) -> np.ndarray:
    """Zero-phase Ricker wavelet, peak frequency ``f0`` (Hz)."""
    a = (np.pi * f0 * (t - t0)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def gabor(t: np.ndarray, f0: float, t0: float, sigma: float) -> np.ndarray:
    """Gabor (Morlet-like) wavelet useful for narrow-band modes."""
    env = np.exp(-((t - t0) ** 2) / (2.0 * sigma**2))
    return env * np.cos(2.0 * np.pi * f0 * (t - t0))


@dataclass
class ArrayGeometry:
    """
    Axial multi-receiver borehole array geometry.

    The geometry is semantically immutable: changing ``n_rec``,
    ``dr``, etc. on a constructed instance would invalidate the
    cached :attr:`offsets` / :attr:`t` arrays, which is almost never
    what the caller intended. ``offsets`` and ``t`` are therefore
    computed on first access and cached via
    :class:`functools.cached_property` so repeated use inside a
    processing loop incurs no recomputation cost.

    Attributes
    ----------
    n_rec : int
        Number of receivers.
    tr_offset : float
        Transmitter-to-first-receiver offset (m).
    dr : float
        Inter-receiver spacing (m).
    dt : float
        Sampling interval (s).
    n_samples : int
        Samples per trace.
    """

    n_rec: int = 8
    tr_offset: float = 3.0
    dr: float = 0.1524  # 6 inches
    dt: float = 1.0e-5
    n_samples: int = 2048

    def __repr__(self) -> str:
        return (
            f"ArrayGeometry(n_rec={self.n_rec}, "
            f"tr_offset={self.tr_offset:.3f} m, "
            f"dr={self.dr:.4f} m, "
            f"dt={self.dt:.2e} s, "
            f"n_samples={self.n_samples})"
        )

    @cached_property
    def offsets(self) -> np.ndarray:
        """Source-to-receiver offsets (m), shape ``(n_rec,)``."""
        return self.tr_offset + np.arange(self.n_rec) * self.dr

    @cached_property
    def t(self) -> np.ndarray:
        """Time axis (s), shape ``(n_samples,)``."""
        return np.arange(self.n_samples) * self.dt

    @classmethod
    def from_imperial(
        cls,
        n_rec: int = 8,
        tr_offset_ft: float = 10.0,
        dr_in: float = 6.0,
        dt: float = 1.0e-5,
        n_samples: int = 2048,
    ) -> ArrayGeometry:
        """
        Convenience constructor using imperial units (ft, in).

        Matches the original Schlumberger Array Sonic specification
        (8 receivers, 10 ft source-to-first-receiver, 6 in spacing).
        """
        FT = 0.3048
        IN = 0.0254
        return cls(
            n_rec=n_rec,
            tr_offset=tr_offset_ft * FT,
            dr=dr_in * IN,
            dt=dt,
            n_samples=n_samples,
        )

    @classmethod
    def schlumberger_array_sonic(
        cls, dt: float = 1.0e-5, n_samples: int = 2048
    ) -> ArrayGeometry:
        """
        Canonical Schlumberger Array Sonic geometry.

        8 receivers, 10 ft (3.048 m) source-to-first-receiver offset,
        6 in (0.1524 m) inter-receiver spacing -- the reference tool
        geometry throughout Mari et al. (1994). The dataclass default
        constructor already encodes these numbers in metric; this
        factory documents the intent at the call site.
        """
        return cls.from_imperial(
            n_rec=8, tr_offset_ft=10.0, dr_in=6.0, dt=dt, n_samples=n_samples
        )


@dataclass
class Mode:
    """
    A single wave mode (P-head, S-head, Stoneley, flexural, ...).

    If ``dispersion`` is provided (callable ``f -> slowness(f)``), the
    arrival is synthesised in the frequency domain with that dispersion
    law; otherwise it is a non-dispersive wavelet arriving at
    ``intercept + offset * slowness + (src_delay + rec_delay)``.

    Notes
    -----
    The ``dispersion`` callable is called with a NumPy array of
    positive frequencies and must return a same-shape array of phase
    slownesses (s/m). A purely scalar callable will fail at runtime;
    wrap it with ``np.vectorize`` at the call site if needed. The
    built-in :func:`dipole_flexural_dispersion` already satisfies the
    array contract.
    """

    name: str
    slowness: float
    f0: float
    amplitude: float = 1.0
    intercept: float = 0.0
    dispersion: Callable[[np.ndarray], np.ndarray] | None = None
    src_delay: float = 0.0
    rec_delay: float = 0.0
    wavelet: str = "ricker"
    sigma: float = 2.0e-4


def _dispersive_arrival(
    t: np.ndarray,
    offset: float,
    f0: float,
    slowness_of_f: Callable[[np.ndarray], np.ndarray],
    intercept: float = 0.0,
    bandwidth: float = 0.6,
) -> np.ndarray:
    """
    Synthesize one dispersive arrival on a single trace.

    Constructs the trace in the frequency domain as

        X(f) = A(f) * exp(-2 pi i f (intercept + offset * s(f)))

    where ``A(f)`` is a Gaussian envelope centred at ``f0`` with
    fractional bandwidth ``bandwidth`` (standard deviation
    ``sigma_f = bandwidth * f0``), zeroed below ``0.1 * f0`` to mimic
    a source low-frequency cutoff, and ``s(f) = slowness_of_f(f)`` is
    the per-frequency phase slowness supplied by the caller. The
    inverse rFFT gives a real-valued time-domain arrival that carries
    the prescribed dispersion.

    See :func:`dipole_flexural_dispersion` for the phenomenological
    ``s(f)`` used for the dipole flexural mode.
    """
    dt = t[1] - t[0]
    n = t.size
    freqs = np.fft.rfftfreq(n, d=dt)
    sigma_f = bandwidth * f0
    A = np.exp(-((freqs - f0) ** 2) / (2.0 * sigma_f**2))
    A[freqs < 0.1 * f0] = 0.0
    # ``slowness_of_f`` is evaluated on the positive-frequency subset
    # only; the DC bin is pinned to 0 so a dispersion law that diverges
    # at f -> 0 does not have to guard the call itself.
    s_f = np.zeros_like(freqs)
    pos = freqs > 0
    s_f[pos] = slowness_of_f(freqs[pos])
    phase = -2.0 * np.pi * freqs * (intercept + offset * s_f)
    return np.fft.irfft(A * np.exp(1j * phase), n=n)


def synthesize_gather(
    geom: ArrayGeometry,
    modes: Sequence[Mode],
    noise: float = 0.02,
    seed: int | None = None,
) -> np.ndarray:
    """Build a common-source multi-receiver gather ``(n_rec, n_samples)``.

    Parameters
    ----------
    geom : ArrayGeometry
        Tool geometry (offsets, sampling).
    modes : sequence of Mode
        Modes to synthesise; their slownesses, intercepts, and
        wavelets are summed coherently before noise is added.
    noise : float, default 0.02
        Per-trace Gaussian noise standard deviation, expressed as a
        fraction of the noise-free RMS of the synthesised gather.
        Set to 0 for a clean noise-free output.
    seed : int or None, default None
        Seed for the noise generator (see :class:`numpy.random.Generator`).
        Pass an integer (e.g. ``seed=0``) for reproducible noise --
        critical for unit tests and any property-based or regression
        check. Pass ``None`` for fresh non-deterministic noise on
        every call; use that for visual demos where exact byte-for-
        byte reproducibility is not required.

    Returns
    -------
    ndarray, shape ``(n_rec, n_samples)``
        Common-source gather, real-valued, ``float64``.
    """
    rng = np.random.default_rng(seed)
    t = geom.t
    offsets = geom.offsets
    data: np.ndarray = np.zeros((geom.n_rec, geom.n_samples), dtype=float)

    for mode in modes:
        for i, off in enumerate(offsets):
            extra = mode.src_delay + mode.rec_delay
            if mode.dispersion is None:
                t_arr = mode.intercept + off * mode.slowness + extra
                if mode.wavelet == "ricker":
                    tr = ricker(t, mode.f0, t0=t_arr)
                else:
                    tr = gabor(t, mode.f0, t_arr, sigma=mode.sigma)
                data[i] += mode.amplitude * tr
            else:
                tr = _dispersive_arrival(
                    t,
                    off,
                    mode.f0,
                    mode.dispersion,
                    intercept=mode.intercept + extra,
                )
                data[i] += mode.amplitude * tr
    if noise > 0:
        rms = np.sqrt(np.mean(data**2)) + 1e-12
        data += rng.normal(scale=noise * rms, size=data.shape)
    return data


def monopole_formation_modes(
    vp: float = 4000.0,
    vs: float = 2300.0,
    v_stoneley: float = 1400.0,
    f_p: float = 15_000.0,
    f_s: float = 10_000.0,
    f_st: float = 3_000.0,
    p_amp: float = 1.0,
    s_amp: float = 1.5,
    st_amp: float = 2.0,
    *,
    v_fluid: float = 1500.0,
    a_borehole: float = 0.1,
    f_pr: float | None = None,
    pr_amp: float = 1.5,
) -> list[Mode]:
    """Canonical P / S / Stoneley mode list for a monopole sonic tool.

    When ``f_pr`` is given (any positive frequency, in Hz), a fourth
    pseudo-Rayleigh / guided arrival is appended at the band-centre
    phase slowness predicted by :func:`pseudo_rayleigh_dispersion` --
    i.e. as a non-dispersive Ricker so that the test gather is
    pickable by ordinary slowness-time coherence. The mode only
    exists in fast formations (``vs > v_fluid``); the dispersion
    factory raises if that condition is violated.

    Callers who want a *dispersive* pseudo-Rayleigh wavetrain (so
    plain STC sees a smeared peak that requires :func:`narrow_band_stc`
    or :func:`dispersive_stc` to pick cleanly) should construct the
    fourth :class:`Mode` by hand, passing
    ``dispersion=pseudo_rayleigh_dispersion(vs, v_fluid, a_borehole)``
    explicitly.
    """
    modes = [
        Mode("P", slowness=1.0 / vp, f0=f_p, amplitude=p_amp),
        Mode("S", slowness=1.0 / vs, f0=f_s, amplitude=s_amp),
        Mode(
            "Stoneley",
            slowness=1.0 / v_stoneley,
            f0=f_st,
            amplitude=st_amp,
            wavelet="gabor",
            sigma=3.0e-4,
        ),
    ]
    if f_pr is not None:
        s_of_f = pseudo_rayleigh_dispersion(
            vs=vs, v_fluid=v_fluid, a_borehole=a_borehole
        )
        s_at_f_pr = float(s_of_f(np.array([f_pr]))[0])
        modes.append(
            Mode(
                "PseudoRayleigh",
                slowness=s_at_f_pr,
                f0=f_pr,
                amplitude=pr_amp,
            )
        )
    return modes


def dipole_flexural_dispersion(
    vs: float, a_borehole: float = 0.1
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Phenomenological flexural-mode dispersion ``s(f)``.

    Low-frequency limit equals the true shear slowness (``1/vs``), high-
    frequency limit is 25% above. This is *not* a full cylindrical Biot
    model; for production use one should solve the 3x3 modal determinant
    in Hankel/Bessel functions (Paillet & Cheng, 1991, chap. 4; Schmitt,
    1988, *JASA* 84(6), 2230-2244).

    The returned callable accepts a NumPy array of frequencies and
    returns a same-shape array of slownesses.
    """
    s_low = 1.0 / vs
    s_high = 1.25 / vs
    fc = vs / (2.0 * np.pi * a_borehole)

    def s_of_f(f: np.ndarray) -> np.ndarray:
        x = np.asarray(f) / fc
        return s_low + (s_high - s_low) * (x**2) / (1.0 + x**2)

    return s_of_f


def pseudo_rayleigh_dispersion(
    vs: float,
    v_fluid: float = 1500.0,
    a_borehole: float = 0.1,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Phenomenological pseudo-Rayleigh dispersion ``s(f)``.

    Pseudo-Rayleigh is a guided trapped mode in the fluid-filled borehole
    that exists only in *fast* formations (``vs > v_fluid``). At its
    low-frequency cutoff its phase velocity equals the formation shear
    velocity (so phase slowness ``= 1/vs``); at high frequency its phase
    velocity asymptotes to the fluid velocity (phase slowness
    ``= 1/v_fluid``). The cutoff frequency scale is set by the borehole
    radius via ``fc = vs / (2 pi a_borehole)``.

    Parameters
    ----------
    vs : float
        Formation shear velocity (m/s).
    v_fluid : float, default 1500.0
        Borehole-fluid acoustic velocity (m/s).
    a_borehole : float, default 0.1
        Borehole radius (m).

    Returns
    -------
    Callable that accepts a NumPy array of frequencies and returns a
    same-shape array of phase slownesses (s/m).

    Raises
    ------
    ValueError
        If ``vs <= v_fluid`` (slow formation -- pseudo-Rayleigh does
        not exist).

    Notes
    -----
    This phenomenological law has the same rational-Lorentzian shape
    as :func:`dipole_flexural_dispersion` -- a smooth interpolation
    between the cutoff and the high-frequency asymptote -- and is
    *not* a full cylindrical Biot model. For quantitative work see
    Paillet & Cheng (1991) chapter 4 or Schmitt (1988) for the 3x3
    Hankel/Bessel modal determinant.
    """
    if vs <= v_fluid:
        raise ValueError(
            f"pseudo-Rayleigh requires a fast formation (vs > v_fluid); "
            f"got vs={vs} m/s, v_fluid={v_fluid} m/s"
        )
    s_low = 1.0 / vs  # at cutoff
    s_high = 1.0 / v_fluid  # high-f asymptote (toward fluid)
    fc = vs / (2.0 * np.pi * a_borehole)

    def s_of_f(f: np.ndarray) -> np.ndarray:
        x = np.asarray(f) / fc
        return s_low + (s_high - s_low) * (x**2) / (1.0 + x**2)

    return s_of_f
