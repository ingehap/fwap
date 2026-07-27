"""
Surrogate-model training-data generator for borehole-acoustic dispersion.

Wraps the cylindrical-Biot forward modal solver (:mod:`fwap`) as a
labelled-pair factory for machine-learning *surrogate* and *inverse*
models -- the borehole-acoustic analog of the seismic DL-FWI /
neural-operator training loop (InversionNet, OpenFWI, Fourier neural
operators). Each generated sample carries everything both model
families need:

* **Forward-surrogate label** -- the per-mode phase-slowness curve
  ``slowness(f)`` returned by the modal solver. A network that maps
  formation parameters -> this curve learns a fast stand-in for the
  determinant root-finding in :func:`fwap.stoneley_dispersion` /
  :func:`fwap.flexural_dispersion`.
* **Inverse-net input** -- the realistic multi-receiver waveform
  gather produced by :func:`fwap.synthesize_gather` from those same
  dispersion curves. A network that maps this gather -> the formation
  parameters is the direct DL-FWI analog.

The generator is deliberately **NumPy/SciPy-only**: it depends on
nothing beyond the existing ``fwap`` runtime dependencies. Model
training (PyTorch/JAX) is intentionally out of scope for this
repository -- consume the ``.npz`` this script writes from a separate
project or an optional ``[ml]`` extra (see the surrogate-layer issue
in the tracker). This keeps the core package a clean reference
implementation while still shipping a reproducible data pipeline.

Usage
-----
Library::

    from gen_surrogate_dataset import generate_dataset, save_npz
    samples = generate_dataset(1000, seed=0)
    save_npz("surrogate_dataset.npz", samples)

Command line::

    python scripts/gen_surrogate_dataset.py --n 5000 --seed 0 \\
        --out surrogate_dataset.npz

References
----------
* Deng, C., et al. (2022). OpenFWI: Large-scale multi-structural
  benchmark datasets for full waveform inversion. *NeurIPS Datasets
  and Benchmarks.*
* Schmitt, D. P. (1988). Shear-wave logging in elastic formations.
  *J. Acoust. Soc. Am.* 84(6), 2230-2244. (The forward physics this
  script wraps.)
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from fwap import (
    ArrayGeometry,
    Mode,
    flexural_dispersion,
    stoneley_dispersion,
    synthesize_gather,
)
from fwap.cylindrical_solver import BoreholeMode

# Order of the scalar formation parameters in :meth:`SurrogateSample.
# param_vector` and the stacked ``params`` array written by
# :func:`save_npz`. These are exactly the keyword arguments the modal
# solvers accept.
PARAM_NAMES: tuple[str, ...] = ("vp", "vs", "rho", "vf", "rho_f", "a")

# Version of the ``.npz`` on-disk contract (key set, ``PARAM_NAMES``
# order, per-array dtypes and shapes) written by :func:`stack_dataset` /
# :func:`save_npz`. Bump this whenever that layout changes in a way a
# downstream consumer must notice (a reordered/renamed key or column, a
# changed dtype, an added axis). Consumers should read it back and refuse
# a version they do not understand; ``tests/test_npz_schema_contract.py``
# pins the layout so a breaking change fails CI here rather than silently
# mislabelling data downstream.
SCHEMA_VERSION: int = 2

SlownessOfF = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ModeSpec:
    """
    One guided mode to forward-model and (optionally) inject.

    Attributes
    ----------
    name : str
        Mode label, stored alongside the sample (e.g. ``"Stoneley"``).
    solver : Callable
        Forward dispersion solver with the signature
        ``solver(freq, *, vp, vs, rho, vf, rho_f, a) -> BoreholeMode``
        (both :func:`fwap.stoneley_dispersion` and
        :func:`fwap.flexural_dispersion` match).
    f0 : float
        Source centre frequency (Hz) used when synthesising this
        mode's arrival.
    amplitude : float
        Relative amplitude of this mode in the summed gather.
    wavelet : str, default ``"ricker"``
        Source wavelet family passed to :class:`fwap.Mode`.
    sigma : float, default ``2.0e-4``
        Gabor half-width (s); only used when ``wavelet == "gabor"``.
    """

    name: str
    solver: Callable[..., BoreholeMode]
    f0: float
    amplitude: float
    wavelet: str = "ricker"
    sigma: float = 2.0e-4


DEFAULT_MODES: tuple[ModeSpec, ...] = (
    ModeSpec(
        "Stoneley",
        stoneley_dispersion,
        f0=3000.0,
        amplitude=2.0,
        wavelet="gabor",
        sigma=3.0e-4,
    ),
    ModeSpec("flexural", flexural_dispersion, f0=3000.0, amplitude=1.5),
)


@dataclass(frozen=True)
class FormationPriors:
    """
    Uniform sampling ranges for a single isotropic formation.

    All velocities are in m/s, densities in kg/m^3, radii in m. The
    ``vp/vs`` ratio is sampled directly (rather than ``vp``) so every
    draw automatically satisfies the ``vp > vs`` constraint the modal
    solvers enforce. Fluid properties are fixed (typical borehole
    brine) but exposed as fields for easy override.

    Attributes
    ----------
    vs_min, vs_max : float
        Shear-velocity range (m/s). The default straddles the
        fluid velocity so the draw mixes slow (``vs < vf``) and fast
        (``vs > vf``) formations -- the two flexural regimes.
    vpvs_min, vpvs_max : float
        Range of the ``vp/vs`` ratio (dimensionless).
    rho_min, rho_max : float
        Bulk-density range (kg/m^3).
    a_min, a_max : float
        Borehole-radius range (m).
    vf : float
        Borehole-fluid velocity (m/s).
    rho_f : float
        Borehole-fluid density (kg/m^3).
    """

    vs_min: float = 1200.0
    vs_max: float = 3200.0
    vpvs_min: float = 1.6
    vpvs_max: float = 2.0
    rho_min: float = 2100.0
    rho_max: float = 2700.0
    a_min: float = 0.08
    a_max: float = 0.12
    vf: float = 1500.0
    rho_f: float = 1000.0

    def sample(self, rng: np.random.Generator) -> dict[str, float]:
        """
        Draw one formation as solver keyword arguments.

        Parameters
        ----------
        rng : numpy.random.Generator
            Source of randomness (seed it for reproducibility).

        Returns
        -------
        dict of str to float
            Keys ``vp, vs, rho, vf, rho_f, a`` -- ready to splat into
            :func:`fwap.stoneley_dispersion` and friends. Guaranteed
            ``vp > vs`` and all-positive.
        """
        vs = float(rng.uniform(self.vs_min, self.vs_max))
        vp = vs * float(rng.uniform(self.vpvs_min, self.vpvs_max))
        rho = float(rng.uniform(self.rho_min, self.rho_max))
        a = float(rng.uniform(self.a_min, self.a_max))
        return {
            "vp": vp,
            "vs": vs,
            "rho": rho,
            "vf": self.vf,
            "rho_f": self.rho_f,
            "a": a,
        }


@dataclass
class SurrogateSample:
    """
    One labelled training pair.

    Attributes
    ----------
    params : dict of str to float
        Formation parameters (the inverse-net label / surrogate
        input). Keys are :data:`PARAM_NAMES`.
    freq : ndarray, shape (n_f,)
        Frequency grid shared by every mode's slowness curve (Hz).
    slowness : ndarray, shape (n_modes, n_f)
        Per-mode phase slowness (s/m); ``NaN`` at frequencies where a
        mode does not exist (below its geometric cutoff, or in the
        wrong regime for the solver). This is the forward-surrogate
        label.
    gather : ndarray, shape (n_rec, n_samples)
        Synthetic multi-receiver waveform (the inverse-net input),
        summed over whichever modes cleared ``min_finite`` and had
        their dispersion injected.
    mode_names : tuple of str
        Mode labels, aligned with axis 0 of ``slowness`` and
        ``mode_in_gather``.
    mode_in_gather : ndarray, shape (n_modes,) of bool
        ``True`` where the mode was injected into ``gather``. A mode
        may have a slowness curve (partly finite) yet be excluded
        from the waveform because its finite support was too sparse
        to interpolate.
    geom : fwap.ArrayGeometry
        Acquisition geometry the ``gather`` was synthesized with
        (sampling interval, receiver offsets, sample count). Shared by
        every sample of a dataset and persisted once by
        :func:`stack_dataset` so the waveform is self-describing.
    """

    params: dict[str, float]
    freq: np.ndarray
    slowness: np.ndarray
    gather: np.ndarray
    mode_names: tuple[str, ...]
    mode_in_gather: np.ndarray
    geom: ArrayGeometry

    def param_vector(self) -> np.ndarray:
        """Formation parameters as an array in :data:`PARAM_NAMES` order.

        Returns
        -------
        ndarray, shape (len(PARAM_NAMES),)
            ``float64`` vector ``[vp, vs, rho, vf, rho_f, a]``.
        """
        return np.array([self.params[k] for k in PARAM_NAMES], dtype=float)


def default_freq_grid(
    n_freq: int = 128,
    f_min: float = 1000.0,
    f_max: float = 12000.0,
) -> np.ndarray:
    """
    Standard dispersion-curve frequency grid.

    Parameters
    ----------
    n_freq : int, default 128
        Number of grid points.
    f_min, f_max : float
        Band edges (Hz). The default ``1-12 kHz`` covers the monopole
        Stoneley and dipole flexural bands of a typical wireline tool.

    Returns
    -------
    ndarray, shape (n_freq,)
        Linearly spaced frequencies (Hz).
    """
    return np.linspace(f_min, f_max, n_freq)


def dispersion_callable(mode: BoreholeMode, min_finite: int) -> SlownessOfF | None:
    """
    Build an interpolating ``slowness(f)`` from a solved mode.

    The modal solver returns ``NaN`` outside a mode's support. This
    helper restricts to the finite samples and returns a callable that
    linearly interpolates (and edge-clamps) across them -- the array
    contract :class:`fwap.Mode` expects for its ``dispersion`` field.

    Parameters
    ----------
    mode : BoreholeMode
        A solved dispersion curve.
    min_finite : int
        Minimum number of finite samples required. Below this the
        support is too sparse to interpolate meaningfully and the
        function returns ``None`` (the caller then drops the mode
        from the waveform).

    Returns
    -------
    Callable[[ndarray], ndarray] or None
        Array-in / array-out slowness law (s/m), or ``None`` if the
        mode has fewer than ``min_finite`` finite samples.

    Notes
    -----
    Interpolation is constant-extrapolated (:func:`numpy.interp`
    edge-clamp) beyond the finite support, so the synthesised arrival
    is most faithful where the source band overlaps that support.
    """
    finite = np.isfinite(mode.slowness)
    if int(finite.sum()) < min_finite:
        return None
    f_fin = mode.freq[finite]
    s_fin = mode.slowness[finite]

    def s_of_f(f: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(f, dtype=float), f_fin, s_fin)

    return s_of_f


def generate_sample(
    rng: np.random.Generator,
    geom: ArrayGeometry,
    freq: np.ndarray,
    *,
    priors: FormationPriors,
    modes: Sequence[ModeSpec] = DEFAULT_MODES,
    noise_max: float = 0.06,
    min_finite: int = 8,
) -> SurrogateSample | None:
    """
    Generate a single training pair, or ``None`` if the draw is rejected.

    Draws a formation from ``priors``, forward-models every mode in
    ``modes``, records their slowness curves, and synthesises a
    waveform from the modes whose dispersion could be interpolated.

    Parameters
    ----------
    rng : numpy.random.Generator
        Randomness for the formation draw, the noise level, and the
        per-gather noise seed.
    geom : ArrayGeometry
        Tool geometry passed to :func:`fwap.synthesize_gather`.
    freq : ndarray, shape (n_f,)
        Frequency grid for the dispersion curves (Hz).
    priors : FormationPriors
        Sampling ranges for the formation.
    modes : sequence of ModeSpec, default :data:`DEFAULT_MODES`
        Modes to forward-model and try to inject.
    noise_max : float, default 0.06
        Upper bound of the per-gather Gaussian noise fraction; the
        actual level is drawn uniformly in ``[0, noise_max]``.
    min_finite : int, default 8
        Minimum finite slowness samples for a mode to enter the gather.

    Returns
    -------
    SurrogateSample or None
        ``None`` when no mode cleared ``min_finite`` (so the waveform
        would be empty) or the solver rejected the draw; the caller
        resamples.
    """
    params = priors.sample(rng)
    n_modes = len(modes)
    slowness = np.full((n_modes, freq.size), np.nan, dtype=float)
    in_gather = np.zeros(n_modes, dtype=bool)
    disp_modes: list[Mode] = []

    for i, spec in enumerate(modes):
        try:
            bm = spec.solver(freq, **params)
        except ValueError:
            # Physically invalid draw for this solver/regime; leave the
            # curve NaN and skip injection.
            continue
        slowness[i] = bm.slowness
        callable_s = dispersion_callable(bm, min_finite)
        if callable_s is None:
            continue
        disp_modes.append(
            Mode(
                spec.name,
                slowness=float(np.nanmedian(bm.slowness)),  # unused when dispersive
                f0=spec.f0,
                amplitude=spec.amplitude,
                dispersion=callable_s,
                wavelet=spec.wavelet,
                sigma=spec.sigma,
            )
        )
        in_gather[i] = True

    if not disp_modes:
        return None

    noise = float(rng.uniform(0.0, noise_max))
    gather = synthesize_gather(
        geom, disp_modes, noise=noise, seed=int(rng.integers(2**31))
    )
    return SurrogateSample(
        params=params,
        freq=freq,
        slowness=slowness,
        gather=gather,
        mode_names=tuple(spec.name for spec in modes),
        mode_in_gather=in_gather,
        geom=geom,
    )


def generate_dataset(
    n: int,
    *,
    seed: int = 0,
    geom: ArrayGeometry | None = None,
    freq: np.ndarray | None = None,
    priors: FormationPriors | None = None,
    modes: Sequence[ModeSpec] = DEFAULT_MODES,
    noise_max: float = 0.06,
    min_finite: int = 8,
    max_attempts: int | None = None,
) -> list[SurrogateSample]:
    """
    Generate ``n`` accepted training pairs.

    Parameters
    ----------
    n : int
        Number of samples to accept.
    seed : int, default 0
        Seed for the master :class:`numpy.random.Generator`; the whole
        dataset is reproducible from it.
    geom : ArrayGeometry or None
        Tool geometry; defaults to :class:`fwap.ArrayGeometry`.
    freq : ndarray or None
        Frequency grid; defaults to :func:`default_freq_grid`.
    priors : FormationPriors or None
        Sampling ranges; defaults to :class:`FormationPriors`.
    modes : sequence of ModeSpec, default :data:`DEFAULT_MODES`
        Modes to model.
    noise_max : float, default 0.06
        Upper bound on per-gather noise fraction.
    min_finite : int, default 8
        Minimum finite slowness samples for a mode to enter the gather.
    max_attempts : int or None
        Cap on draws (accepted + rejected). ``None`` uses
        ``100 * n + 100``. Guards against a degenerate prior that
        rejects everything.

    Returns
    -------
    list of SurrogateSample
        Exactly ``n`` samples.

    Raises
    ------
    ValueError
        If ``n`` is negative.
    RuntimeError
        If ``max_attempts`` is hit before ``n`` samples are accepted.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if geom is None:
        geom = ArrayGeometry()
    if freq is None:
        freq = default_freq_grid()
    if priors is None:
        priors = FormationPriors()
    if max_attempts is None:
        max_attempts = 100 * n + 100

    rng = np.random.default_rng(seed)
    samples: list[SurrogateSample] = []
    attempts = 0
    while len(samples) < n and attempts < max_attempts:
        attempts += 1
        sample = generate_sample(
            rng,
            geom,
            freq,
            priors=priors,
            modes=modes,
            noise_max=noise_max,
            min_finite=min_finite,
        )
        if sample is not None:
            samples.append(sample)

    if len(samples) < n:
        raise RuntimeError(
            f"accepted only {len(samples)}/{n} samples in {attempts} "
            "attempts; loosen the priors or raise max_attempts"
        )
    return samples


def stack_dataset(samples: Sequence[SurrogateSample]) -> dict[str, np.ndarray]:
    """
    Stack a sample list into rectangular arrays for storage.

    Every sample must share the same frequency grid, mode set, and
    gather shape (guaranteed when produced by :func:`generate_dataset`
    with a single configuration).

    Parameters
    ----------
    samples : sequence of SurrogateSample
        Non-empty sample list.

    Returns
    -------
    dict of str to ndarray
        Keys and shapes (``N`` samples, ``M`` modes):

        * ``params`` -- ``(N, len(PARAM_NAMES))``
        * ``slowness`` -- ``(N, M, n_f)`` (NaN where a mode is absent)
        * ``gather`` -- ``(N, n_rec, n_samples)``
        * ``mode_in_gather`` -- ``(N, M)`` bool
        * ``freq`` -- ``(n_f,)``
        * ``param_names`` -- ``(len(PARAM_NAMES),)`` str
        * ``mode_names`` -- ``(M,)`` str
        * ``schema_version`` -- ``()`` int (0-d), the :data:`SCHEMA_VERSION`
          of this on-disk layout
        * ``dt`` -- ``()`` float (0-d), sampling interval (s)
        * ``tr_offset`` -- ``()`` float (0-d), transmitter-to-first-receiver
          offset (m)
        * ``dr`` -- ``()`` float (0-d), inter-receiver spacing (m)

        The three geometry scalars (added in schema v2) plus the
        ``gather``'s ``(n_rec, n_samples)`` fully reconstruct the
        acquisition :class:`fwap.ArrayGeometry`, making the waveform
        self-describing.

    Raises
    ------
    ValueError
        If ``samples`` is empty.
    """
    if len(samples) == 0:
        raise ValueError("cannot stack an empty sample list")
    geom = samples[0].geom
    return {
        "params": np.stack([s.param_vector() for s in samples]),
        "slowness": np.stack([s.slowness for s in samples]),
        "gather": np.stack([s.gather for s in samples]),
        "mode_in_gather": np.stack([s.mode_in_gather for s in samples]),
        "freq": np.asarray(samples[0].freq, dtype=float),
        "param_names": np.array(PARAM_NAMES),
        "mode_names": np.array(samples[0].mode_names),
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
        "dt": np.asarray(geom.dt, dtype=float),
        "tr_offset": np.asarray(geom.tr_offset, dtype=float),
        "dr": np.asarray(geom.dr, dtype=float),
    }


def save_npz(path: str, samples: Sequence[SurrogateSample]) -> None:
    """
    Write a stacked dataset to a compressed ``.npz``.

    Parameters
    ----------
    path : str
        Output path (``.npz`` appended by NumPy if absent).
    samples : sequence of SurrogateSample
        Dataset to serialise; see :func:`stack_dataset` for the keys.
    """
    np.savez_compressed(path, **stack_dataset(samples))


def _build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a borehole-acoustic surrogate-model training set "
            "by wrapping the fwap forward modal solver."
        )
    )
    parser.add_argument("--n", type=int, default=1000, help="samples to generate")
    parser.add_argument("--seed", type=int, default=0, help="master RNG seed")
    parser.add_argument(
        "--out",
        type=str,
        default="surrogate_dataset.npz",
        help="output .npz path",
    )
    parser.add_argument("--n-freq", type=int, default=128, help="frequency samples")
    parser.add_argument("--f-min", type=float, default=1000.0, help="min freq (Hz)")
    parser.add_argument("--f-max", type=float, default=12000.0, help="max freq (Hz)")
    parser.add_argument(
        "--noise-max",
        type=float,
        default=0.06,
        help="upper bound of per-gather noise fraction",
    )
    parser.add_argument(
        "--min-finite",
        type=int,
        default=8,
        help="min finite slowness samples for a mode to enter the gather",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Command-line entry point.

    Parameters
    ----------
    argv : sequence of str or None
        Argument vector (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit status (``0`` on success).
    """
    args = _build_parser().parse_args(argv)
    freq = default_freq_grid(args.n_freq, args.f_min, args.f_max)
    samples = generate_dataset(
        args.n,
        seed=args.seed,
        freq=freq,
        noise_max=args.noise_max,
        min_finite=args.min_finite,
    )
    save_npz(args.out, samples)

    stacked = stack_dataset(samples)
    print(f"wrote {len(samples)} samples to {args.out}")
    print(f"  params        {stacked['params'].shape}")
    print(f"  slowness      {stacked['slowness'].shape}")
    print(f"  gather        {stacked['gather'].shape}")
    print(f"  modes         {tuple(stacked['mode_names'])}")
    in_gather_rate = stacked["mode_in_gather"].mean(axis=0)
    for name, rate in zip(stacked["mode_names"], in_gather_rate):
        print(f"    {name:>16s} injected in {rate:6.1%} of gathers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
