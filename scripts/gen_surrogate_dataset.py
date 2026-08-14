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
* Schmitt, D. P. (1988). Shear wave logging in elastic formations.
  *J. Acoust. Soc. Am.* 84(6), 2215-2229. (The forward physics this
  script wraps.)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from fwap import (
    ArrayGeometry,
    BoreholeLayer,
    FluidAnnulus,
    Mode,
    crack_wave_dispersion,
    flexural_dispersion,
    flexural_dispersion_layered,
    pseudo_rayleigh_modal_dispersion,
    quadrupole_dispersion,
    stoneley_dispersion,
    stoneley_dispersion_layered,
    stoneley_dispersion_microannulus,
    synthesize_gather,
)
from fwap.cylindrical_solver import BoreholeMode

# Order of the scalar formation parameters in :meth:`SurrogateSample.
# param_vector` and the stacked ``params`` array written by
# :func:`save_npz`. These are exactly the keyword arguments the modal
# solvers accept.
PARAM_NAMES: tuple[str, ...] = ("vp", "vs", "rho", "vf", "rho_f", "a")

# Column order of the per-layer rows in the ``layer_params`` array written for
# cased-hole datasets (schema v4). One row per annular layer between the
# borehole fluid and the formation half-space.
LAYER_PARAM_NAMES: tuple[str, ...] = ("vp", "vs", "rho", "thickness")

# Layer labels for a cased-hole (casing + cement) sample, aligned with axis 1
# of ``layer_params``.
CASED_LAYER_NAMES: tuple[str, ...] = ("casing", "cement")

# Layer labels for a *debonded* cased sample: the fluid microannulus sits
# between casing and cement and is written into ``layer_params`` like any
# other layer, with ``vs = 0`` because a fluid carries no shear. Its
# ``thickness`` column is therefore the debonding gap, and no schema change
# is needed to carry it.
MICROANNULUS_LAYER_NAMES: tuple[str, ...] = ("casing", "microannulus", "cement")

# Version of the ``.npz`` on-disk contract (key set, ``PARAM_NAMES``
# order, per-array dtypes and shapes) written by :func:`stack_dataset` /
# :func:`save_npz`. Bump this whenever that layout changes in a way a
# downstream consumer must notice (a reordered/renamed key or column, a
# changed dtype, an added axis). Consumers should read it back and refuse
# a version they do not understand; ``tests/test_npz_schema_contract.py``
# pins the layout so a breaking change fails CI here rather than silently
# mislabelling data downstream.
SCHEMA_VERSION: int = 4

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
    inject : bool, default True
        Whether to add this mode's arrival to the synthesised gather. Set
        ``False`` for a mode whose dispersion curve is worth recording but
        whose arrival does not physically fall inside the record -- see
        :data:`CRACK_WAVE_MODE`. The curve is still written to
        ``slowness``; only ``mode_in_gather`` and the gather change.
    """

    name: str
    solver: Callable[..., BoreholeMode]
    f0: float
    amplitude: float
    wavelet: str = "ricker"
    sigma: float = 2.0e-4
    inject: bool = True


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

# Opt-in n=2 quadrupole mode -- a signature-compatible drop-in kept out of
# DEFAULT_MODES so the default dataset stays lean (two modes, two solves per
# sample). Pass e.g. ``modes=(*DEFAULT_MODES, QUADRUPOLE_MODE)`` to
# :func:`generate_dataset` for a three-mode dataset; the loader and models are
# mode-count-agnostic (they read ``M`` from ``mode_names``).
#
# **Pair this with a slow-formation prior.** The quadrupole is a clean bound
# mode only for ``vs < vf``. An earlier version of this note said that in fast
# formations it "often falls below ``min_finite``", so that bad draws would be
# filtered out downstream. Measured over the default mixed prior, they usually
# are not: 19 of 31 fast draws cleared ``min_finite``, and 18 of those 19
# returned a *non-monotone* curve scattered between the Rayleigh and shear
# speeds rather than a guided mode. Those would be marked present in
# ``mode_in_gather`` and injected into the gather.
#
# The cause is the same leaky-mode limitation as the n=1 flexural case (see
# plans/roadmap.md A.2): for ``vs > vf`` the root leaves the real axis and the
# real-axis search returns spurious values. Slow-formation draws are unaffected
# (11 of 11 monotone in the same sample).
QUADRUPOLE_MODE: ModeSpec = ModeSpec(
    "quadrupole", quadrupole_dispersion, f0=4000.0, amplitude=1.0
)

# Opt-in leaky pseudo-Rayleigh mode. Unlike the bound modes it also carries a
# non-trivial spatial attenuation (stored in the ``attenuation`` channel). It
# exists only in fast formations (``vs > vf``); the solver raises for slow
# draws, which ``generate_sample`` skips -- so pair it with a fast-only prior
# (e.g. ``FormationPriors(vs_min=1600.0)``) to avoid rejecting most samples.
PSEUDO_RAYLEIGH_MODE: ModeSpec = ModeSpec(
    "pseudo_rayleigh", pseudo_rayleigh_modal_dispersion, f0=6000.0, amplitude=1.0
)

# Opt-in cased-hole Stoneley mode (schema v4). Uses the layered modal solver
# over the annular stack fluid -> casing -> cement -> formation
# (:func:`fwap.stoneley_dispersion_layered`). The Stoneley tube wave is the
# bond-sensitive mode used for cement-bond evaluation, and it is a clean
# *bound* mode only in the well-bonded regime (stiff cement with
# ``V_S_cement >= V_f``) and a fast formation (``V_S > V_f``) -- pair it with a
# :class:`CasingCementPriors` and a fast-formation :class:`FormationPriors`
# (see :func:`generate_cased_dataset`). Cased flexural is intentionally
# excluded from this default set: it is sparse in fast formations (~38 % of a
# 1-12 kHz band). That was once attributed to layered bracketing; it is
# actually leakage, and the identical formation is just as sparse in an open
# hole (plans/roadmap.md A.2). Recovering it needs complex-plane root tracking,
# so the *default* cased dataset stays single-mode. For a two-mode cased
# dataset over the narrow formation window where both modes are bound, see
# :data:`CASED_TWO_MODES` and :func:`generate_slow_two_mode_cased_dataset`.
CASED_STONELEY_MODE: ModeSpec = ModeSpec(
    "Stoneley",
    stoneley_dispersion_layered,
    f0=3000.0,
    amplitude=2.0,
    wavelet="gabor",
    sigma=3.0e-4,
)

# Default cased-hole mode set: the single bound Stoneley mode.
CASED_MODES: tuple[ModeSpec, ...] = (CASED_STONELEY_MODE,)

# Opt-in cased-hole flexural mode, usable only with SLOW_TWO_MODE_PRIORS below.
CASED_FLEXURAL_MODE: ModeSpec = ModeSpec(
    "flexural",
    flexural_dispersion_layered,
    f0=3000.0,
    amplitude=1.5,
)

# Two-mode cased set. Valid only over SLOW_TWO_MODE_PRIORS -- see there.
CASED_TWO_MODES: tuple[ModeSpec, ...] = (CASED_STONELEY_MODE, CASED_FLEXURAL_MODE)

# Debonded (fluid-microannulus) cased modes -- see :class:`MicroannulusPriors`
# for the measurements that make this a two-mode set rather than one.
#
# The Stoneley branch responds to the *presence* of the slip interface and is
# almost blind to how wide it is (0.05 % over a 100x range of gap), so it
# labels a bonded/debonded state. The crack wave carries the width, going as
# ``h**(1/3)`` -- 4.8x over that same range against 0.03 % for the formation.
DEBONDED_STONELEY_MODE: ModeSpec = ModeSpec(
    "Stoneley",
    stoneley_dispersion_microannulus,
    f0=3000.0,
    amplitude=2.0,
    wavelet="gabor",
    sigma=3.0e-4,
)

# The crack wave is recorded but **not injected**, and that is physics rather
# than convenience: at 63-620 m/s over this prior it arrives between 4.8 ms
# and 47.6 ms at the 3 m near offset, against a 5.12 ms record. Only the very
# widest gap would even enter the window. Its dispersion curve is the useful
# product; a planted arrival would be fiction. ``amplitude`` is unused while
# ``inject=False`` and is kept only so the spec reads like its neighbours.
CRACK_WAVE_MODE: ModeSpec = ModeSpec(
    "crack_wave",
    crack_wave_dispersion,
    f0=3000.0,
    amplitude=1.0,
    wavelet="gabor",
    sigma=3.0e-4,
    inject=False,
)

#: Debonded cased-hole mode set: the state branch and the thickness branch.
DEBONDED_MODES: tuple[ModeSpec, ...] = (DEBONDED_STONELEY_MODE, CRACK_WAVE_MODE)


def _layer_rows(layers: Sequence[BoreholeLayer]) -> np.ndarray:
    """Pack layers into the ``(n_layer, 4)`` ``layer_params`` row order."""
    if not layers:
        return np.empty((0, len(LAYER_PARAM_NAMES)), dtype=float)
    return np.array(
        [[ly.vp, ly.vs, ly.rho, ly.thickness] for ly in layers], dtype=float
    )


@dataclass(frozen=True)
class AnnulusDraw:
    """
    One sampled annular stack, in the form :func:`generate_sample` consumes.

    Exists so that priors which build *different* solver call signatures can
    be used interchangeably: a bonded stack passes ``layers=``, while a
    debonded one passes ``inner_layers=`` / ``annulus=`` / ``outer_layers=``.
    The caller does not need to know which.

    Attributes
    ----------
    solver_kwargs : dict
        Extra keyword arguments handed to every mode solver for this sample.
    layer_params : ndarray, shape (n_layer, 4)
        Rows of :data:`LAYER_PARAM_NAMES` for the sampled stack.
    layer_names : tuple of str
        Labels aligned with axis 0 of ``layer_params``.
    bond_index : float
        Normalised cement-quality proxy in ``[0, 1]``; 1 is the best bond
        this prior can draw. What *drives* it differs by prior -- cement
        stiffness when bonded, gap width when debonded -- which is the
        point of keeping the name and the range fixed.
    """

    solver_kwargs: dict[str, object]
    layer_params: np.ndarray
    layer_names: tuple[str, ...]
    bond_index: float


@dataclass(frozen=True)
class CasingCementPriors:
    """
    Uniform sampling ranges for a steel casing + cement annulus.

    Draws the two annular layers between the borehole fluid and the formation
    for a cased-hole sample, plus a scalar *bond index* -- a normalized
    cement-quality proxy in ``[0, 1]`` derived from the sampled cement shear
    velocity. A high bond index is a stiff, well-bonded cement; the index
    decreases as the cement softens toward the fluid velocity.

    Notes
    -----
    The cement shear-velocity range is deliberately kept in the **bound
    Stoneley regime** (``cement_vs`` lower bound ``>= vf``): below the fluid
    velocity the cased Stoneley mode leaks and the modal-determinant solver no
    longer returns a bound curve. The *free-pipe* (fully debonded) waveform
    signature -- the classic CBL casing-ring amplitude -- is a phenomenological
    effect outside this bound-mode dataset and is deferred to a later
    milestone; here the bond index spans graded cement quality within the
    bonded regime.

    Attributes
    ----------
    casing_vp, casing_vs : (float, float)
        Casing P-/S-velocity ranges (m/s). Defaults bracket steel.
    casing_rho : float
        Casing density (kg/m^3), fixed (steel).
    casing_thickness : (float, float)
        Casing radial thickness range (m).
    cement_vp, cement_vs, cement_rho, cement_thickness : (float, float)
        Cement-layer velocity (m/s), density (kg/m^3), and thickness (m)
        ranges. ``cement_vs`` stays ``>= vf`` to keep the Stoneley mode bound.
    vf : float
        Borehole-fluid velocity (m/s); the Stoneley bound floor.
    """

    casing_vp: tuple[float, float] = (5700.0, 6000.0)
    casing_vs: tuple[float, float] = (3050.0, 3230.0)
    casing_rho: float = 7800.0
    casing_thickness: tuple[float, float] = (0.008, 0.012)
    cement_vp: tuple[float, float] = (2000.0, 2600.0)
    cement_vs: tuple[float, float] = (1500.0, 1950.0)
    cement_rho: tuple[float, float] = (1700.0, 2000.0)
    cement_thickness: tuple[float, float] = (0.03, 0.06)
    vf: float = 1500.0

    @property
    def layer_names(self) -> tuple[str, ...]:
        """Layer labels aligned with the sampled layer stack."""
        return CASED_LAYER_NAMES

    def sample(
        self, rng: np.random.Generator
    ) -> tuple[tuple[BoreholeLayer, ...], float]:
        """
        Draw one casing + cement stack and its bond index.

        Parameters
        ----------
        rng : numpy.random.Generator

        Returns
        -------
        layers : tuple of fwap.BoreholeLayer
            ``(casing, cement)``, ordered radially outward from the fluid.
        bond_index : float
            Normalized cement-quality proxy in ``[0, 1]`` (from the sampled
            cement shear velocity).
        """
        casing = BoreholeLayer(
            vp=float(rng.uniform(*self.casing_vp)),
            vs=float(rng.uniform(*self.casing_vs)),
            rho=self.casing_rho,
            thickness=float(rng.uniform(*self.casing_thickness)),
        )
        cement_vs = float(rng.uniform(*self.cement_vs))
        cement = BoreholeLayer(
            vp=float(rng.uniform(*self.cement_vp)),
            vs=cement_vs,
            rho=float(rng.uniform(*self.cement_rho)),
            thickness=float(rng.uniform(*self.cement_thickness)),
        )
        lo, hi = self.cement_vs
        bond_index = (cement_vs - lo) / (hi - lo) if hi > lo else 1.0
        return (casing, cement), float(bond_index)

    def draw(self, rng: np.random.Generator) -> AnnulusDraw:
        """One annulus draw in the form :func:`generate_sample` consumes."""
        layers, bond_index = self.sample(rng)
        return AnnulusDraw(
            solver_kwargs={"layers": layers},
            layer_params=_layer_rows(layers),
            layer_names=self.layer_names,
            bond_index=bond_index,
        )


@dataclass(frozen=True)
class MicroannulusPriors:
    """
    Sampling ranges for a **debonded** casing + cement annulus.

    Draws the same steel casing and cement as :class:`CasingCementPriors`
    with a fluid **microannulus** between them -- the standard model of
    casing-to-cement debonding, and a bound-mode problem needing no
    complex-plane tracking (``plans/roadmap.md`` A.5).

    What the dataset can and cannot be asked
    ----------------------------------------
    Measured on a representative stack over a 1-12 kHz band, holding
    everything but the named quantity fixed:

    ==================================== ============== ==============
    quantity varied                      Stoneley curve crack wave
    ==================================== ============== ==============
    gap 10 -> 1000 um                    0.05 %         **+301 %**
    formation ``vs`` across its prior    1.0-1.5 %      0.03 %
    cement ``vs`` across its prior       0.48 %         1.0-3.3 %
    bonded -> debonded (any gap)         **4.14 %**     n/a
    ==================================== ============== ==============

    Two things follow, and they set the shape of this prior.

    **The cased Stoneley mode is blind to the gap width.** It responds to
    the *slip interface* -- shear traction is zero on both faces of a fluid
    layer however thin it is -- and that response is the same at 10 um as
    at 1 mm. So the Stoneley curve supports a bonded/debonded *state*, at
    roughly 3:1 against the nuisance parameters, and not a thickness
    regression. Training a regressor on it would fit noise.

    **The crack wave carries the thickness, at roughly 100:1.** The
    Krauklis crack-wave velocity goes as ``h**(1/3)``, so a 100x range of
    gap moves it 4.8x while the formation moves it 0.03 %. That is why
    :data:`DEBONDED_MODES` carries both branches and why ``gap_thickness``
    is sampled **log-uniformly**: uniform-in-log is uniform-in-observable
    for a cube-root law.

    A third measured result is a caution rather than a design input: a
    100 um gap cuts the cement-stiffness sensitivity of the Stoneley curve
    from 3.22 % to 0.48 %. The bonded cement-bond inverse keys on exactly
    that sensitivity, so it is not merely untested in this regime -- the
    signal it reads has largely gone.

    Attributes
    ----------
    casing_vp, casing_vs, casing_rho, casing_thickness
        As :class:`CasingCementPriors`.
    gap_thickness : (float, float)
        Microannulus thickness range (m), sampled log-uniformly. The
        default ``1e-5`` to ``1e-3`` spans 10 um to 1 mm, the range
        cement-bond logging treats as a microannulus.
    gap_vf, gap_rho : float
        Velocity (m/s) and density (kg/m^3) of the fluid in the gap;
        defaults match the borehole fluid.
    cement_vp, cement_vs, cement_rho, cement_thickness
        As :class:`CasingCementPriors`. The ``cement_vs`` floor may sit at
        ``vf`` here as it does there -- measured, the microannulus Stoneley
        root stays bound and fully finite at ``cement_vs == vf``.
    vf : float
        Borehole-fluid velocity (m/s).
    """

    casing_vp: tuple[float, float] = (5700.0, 6000.0)
    casing_vs: tuple[float, float] = (3050.0, 3230.0)
    casing_rho: float = 7800.0
    casing_thickness: tuple[float, float] = (0.008, 0.012)
    gap_thickness: tuple[float, float] = (1.0e-5, 1.0e-3)
    gap_vf: float = 1500.0
    gap_rho: float = 1000.0
    cement_vp: tuple[float, float] = (2000.0, 2600.0)
    cement_vs: tuple[float, float] = (1500.0, 1950.0)
    cement_rho: tuple[float, float] = (1700.0, 2000.0)
    cement_thickness: tuple[float, float] = (0.03, 0.06)
    vf: float = 1500.0

    @property
    def layer_names(self) -> tuple[str, ...]:
        """Layer labels aligned with the sampled layer stack."""
        return MICROANNULUS_LAYER_NAMES

    def draw(self, rng: np.random.Generator) -> AnnulusDraw:
        """
        Draw one debonded stack.

        Parameters
        ----------
        rng : numpy.random.Generator

        Returns
        -------
        AnnulusDraw
            ``solver_kwargs`` carries ``inner_layers`` / ``annulus`` /
            ``outer_layers`` for the microannulus solvers;
            ``layer_params`` carries the gap as a ``vs = 0`` row, so the
            sampled thickness is recoverable from the dataset without a
            schema change; ``bond_index`` is 1 at the tightest gap this
            prior can draw and 0 at the widest, on a log scale so that it
            is linear in what the crack wave actually shows.
        """
        casing = BoreholeLayer(
            vp=float(rng.uniform(*self.casing_vp)),
            vs=float(rng.uniform(*self.casing_vs)),
            rho=self.casing_rho,
            thickness=float(rng.uniform(*self.casing_thickness)),
        )
        lo, hi = self.gap_thickness
        gap = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        cement = BoreholeLayer(
            vp=float(rng.uniform(*self.cement_vp)),
            vs=float(rng.uniform(*self.cement_vs)),
            rho=float(rng.uniform(*self.cement_rho)),
            thickness=float(rng.uniform(*self.cement_thickness)),
        )
        annulus = FluidAnnulus(vf=self.gap_vf, rho=self.gap_rho, thickness=gap)
        bond_index = (
            1.0 - (np.log(gap) - np.log(lo)) / (np.log(hi) - np.log(lo))
            if hi > lo
            else 1.0
        )
        # The gap joins ``layer_params`` as an ordinary layer with no shear
        # velocity, between casing and cement.
        layer_params = np.array(
            [
                [casing.vp, casing.vs, casing.rho, casing.thickness],
                [self.gap_vf, 0.0, self.gap_rho, gap],
                [cement.vp, cement.vs, cement.rho, cement.thickness],
            ],
            dtype=float,
        )
        return AnnulusDraw(
            solver_kwargs={
                "inner_layers": (casing,),
                "annulus": annulus,
                "outer_layers": (cement,),
            },
            layer_params=layer_params,
            layer_names=self.layer_names,
            bond_index=float(bond_index),
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


# Formation prior over which *both* cased modes are present.
#
# The two cased modes fail in opposite directions, which is why the default
# cased dataset is single-mode rather than merely under-ambitious:
#
#   * cased flexural is sparse in fast formations (a leaky-mode problem, not a
#     bracketing one -- see plans/roadmap.md A.2), and
#   * cased Stoneley stops being bound as the formation slows away from the
#     fluid velocity.
#
# Measured across the CasingCementPriors annulus at 48 frequencies over
# 1-12 kHz, 25 draws per formation:
#
#   V_S (m/s)   Stoneley bound everywhere   flexural coverage (median)
#     1350               0.00                        0.29
#     1380               0.40                        0.40
#     1400               1.00                        0.46
#     1420               1.00                        0.50
#     1450               1.00                        0.54
#     1495               1.00                        0.60
#
# The lower bound below is the Stoneley's measured floor, and 1420 is also
# where the flexural mode stops vanishing entirely on some annulus draws. The
# upper bound stops just short of the 1500 m/s fluid velocity, above which the
# flexural mode enters the fast regime and goes sparse again.
#
# **The flexural mode is no longer bound across the whole band here.** It used
# to be: this prior was chosen as the window where both modes were finite at
# every frequency. The roadmap-A.8 correction to the SV column removed a
# spurious bound branch behind stiff annuli, and with a steel casing the cased
# dipole mode is genuinely leaky against a slow formation over the lower part
# of the band. What survives is a contiguous UPPER sub-band -- typically half
# the grid, reaching the top of it -- whose lower edge moves with the cement
# draw. :func:`generate_slow_two_mode_cased_dataset` therefore passes
# ``require_all_modes=True``, so an accepted sample still carries both modes
# with at least ``min_finite`` points each; it just does not carry the flexural
# one everywhere.
#
# The window is therefore genuinely narrow -- about 80 m/s -- and it is
# *disjoint* from the default cased prior (1700-3000 m/s), so this is a
# different dataset rather than a subset of the usual one. That is acceptable
# for cement-bond work, where the label is the bond index and formation V_S is
# a nuisance parameter (sweeping cement stiffness moves the cased Stoneley
# ~7 %, formation V_S ~1.5 %), but it would be the wrong dataset for anything
# that needs formation-property variety.
SLOW_TWO_MODE_PRIORS: FormationPriors = FormationPriors(
    vs_min=1420.0,
    vs_max=1495.0,
)


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
    attenuation : ndarray, shape (n_modes, n_f)
        Per-mode spatial attenuation rate (1/m) for leaky modes (e.g.
        pseudo-Rayleigh); ``NaN`` for bound modes (which have no
        attenuation) and at frequencies where the mode is absent. A
        free extra label the modal solver produces alongside slowness.
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
    layer_params : ndarray, shape (n_layers, 4)
        Per-annular-layer ``[vp, vs, rho, thickness]`` (columns in
        :data:`LAYER_PARAM_NAMES` order) for a cased-hole sample; an empty
        ``(0, 4)`` array for an open-hole sample. Schema v4.
    layer_names : tuple of str, length n_layers
        Layer labels aligned with axis 0 of ``layer_params`` (e.g.
        ``("casing", "cement")``); empty for an open-hole sample.
    bond_index : float
        Normalized cement-quality proxy in ``[0, 1]`` for a cased-hole
        sample; ``NaN`` for an open-hole sample (no cement). The
        cement-bond-evaluation inverse target.
    """

    params: dict[str, float]
    freq: np.ndarray
    slowness: np.ndarray
    attenuation: np.ndarray
    gather: np.ndarray
    mode_names: tuple[str, ...]
    mode_in_gather: np.ndarray
    geom: ArrayGeometry
    layer_params: np.ndarray
    layer_names: tuple[str, ...]
    bond_index: float

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


#: Missing samples a curve's support may contain without being treated
#: as two separate segments.
#:
#: A modal curve can drop a sample where the tracker stumbles and pick it
#: straight back up -- the standard cased slow-formation fixture has done
#: that at one frequency for as long as it has existed -- and a hole that
#: narrow should not split a branch. A *wide* gap is different in kind:
#: it is the mode leaving the search window and coming back, and the two
#: sides are separate segments of the curve with no root between them.
_SUPPORT_MAX_GAP = 2


def principal_support(
    slowness: np.ndarray, max_gap: int = _SUPPORT_MAX_GAP
) -> np.ndarray:
    """
    Mask selecting the longest usable segment of a modal curve.

    Parameters
    ----------
    slowness : ndarray, shape (n_freq,)
        Modal phase slowness (s/m), ``NaN`` outside the mode's support.
    max_gap : int, default 2
        Missing samples tolerated inside a segment. Runs separated by
        more than this are separate segments.

    Returns
    -------
    ndarray of bool, shape (n_freq,)
        True on the finite samples of the longest segment, False
        everywhere else. All-False when nothing is finite.

    Notes
    -----
    **This exists because interpolation cannot be trusted across a wide
    gap.** :func:`dispersion_callable` linearly interpolates the finite
    support, so a curve whose support is one orphaned sample plus a
    block far away would have a straight line drawn between them --
    through a band where the solver found no root at all, and where
    there is none to find. That became reachable when the leaky cased
    marcher's seeding was rebuilt: it recovers a genuine second leg of
    the branch at the bottom of the band, which on a coarse grid can be
    a single sample.

    Keeping the longest segment rather than merging them is the
    conservative reading: one continuous stretch of a branch is what an
    interpolant can describe.
    """
    finite = np.isfinite(np.asarray(slowness, dtype=float))
    index = np.flatnonzero(finite)
    mask = np.zeros(finite.shape, dtype=bool)
    if index.size == 0:
        return mask
    missing = np.diff(index) - 1
    breaks = np.flatnonzero(missing > max_gap)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [index.size - 1]))
    widest = int(np.argmax(ends - starts + 1))
    mask[index[starts[widest]] : index[ends[widest]] + 1] = True
    return mask & finite


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
    cased_priors: CasingCementPriors | MicroannulusPriors | None = None,
    noise_max: float = 0.06,
    min_finite: int = 8,
    require_all_modes: bool = False,
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
    cased_priors : CasingCementPriors or MicroannulusPriors or None, default None
        When given, draws a casing + cement annulus and passes it as
        ``layers=`` to each mode solver (a cased-hole sample); the sampled
        layer stack and its bond index are recorded on the sample. ``None``
        produces an open-hole sample (no layers). The ``modes`` must use the
        layered solvers (e.g. :data:`CASED_MODES`) when this is set.
    noise_max : float, default 0.06
        Upper bound of the per-gather Gaussian noise fraction; the
        actual level is drawn uniformly in ``[0, noise_max]``.
    min_finite : int, default 8
        Minimum finite slowness samples for a mode to enter the gather.
    require_all_modes : bool, default False
        Reject the draw unless EVERY mode in ``modes`` cleared
        ``min_finite``. The default accepts a draw as soon as one mode
        does, which is right for mode sets whose members are known to be
        regime-dependent; set it when the dataset's contract is that all
        of them are present.

    Returns
    -------
    SurrogateSample or None
        ``None`` when no mode cleared ``min_finite`` (so the waveform
        would be empty) or the solver rejected the draw; the caller
        resamples.
    """
    params = priors.sample(rng)
    if cased_priors is not None:
        draw = cased_priors.draw(rng)
        solver_kwargs = draw.solver_kwargs
        bond_index = draw.bond_index
        layer_names = draw.layer_names
        layer_params = draw.layer_params
    else:
        solver_kwargs = {}
        bond_index = float("nan")
        layer_names = ()
        layer_params = np.empty((0, len(LAYER_PARAM_NAMES)), dtype=float)

    n_modes = len(modes)
    slowness = np.full((n_modes, freq.size), np.nan, dtype=float)
    attenuation = np.full((n_modes, freq.size), np.nan, dtype=float)
    in_gather = np.zeros(n_modes, dtype=bool)
    disp_modes: list[Mode] = []

    for i, spec in enumerate(modes):
        try:
            # Open-hole solvers take the bare formation kwargs (bit-identical
            # path); annular ones take whatever their prior built -- ``layers``
            # for a bonded stack, ``inner_layers`` / ``annulus`` /
            # ``outer_layers`` for a debonded one.
            bm = spec.solver(freq, **params, **solver_kwargs)
        except ValueError:
            # Physically invalid draw for this solver/regime; leave the
            # curve NaN and skip injection.
            continue
        # Keep one continuous segment of the curve. A mode can leave the
        # solver's search window and come back -- the cased dipole in a
        # slow formation does exactly that -- and the stored row has to
        # agree with the arrival injected from it, which is interpolated.
        keep = principal_support(bm.slowness)
        slowness[i] = np.where(keep, bm.slowness, np.nan)
        # Leaky modes (e.g. pseudo-Rayleigh) also carry a spatial attenuation
        # rate; bound modes leave it None -> the row stays NaN.
        if bm.attenuation_per_meter is not None:
            attenuation[i] = np.where(keep, bm.attenuation_per_meter, np.nan)
        callable_s = dispersion_callable(replace(bm, slowness=slowness[i]), min_finite)
        if callable_s is None or not spec.inject:
            # ``inject=False`` keeps the curve and skips the arrival; see
            # :data:`CRACK_WAVE_MODE`.
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
    if require_all_modes and not in_gather.all():
        # Every mode in the spec must have cleared ``min_finite``. Used by
        # the two-mode cased dataset, whose contract is that both modes are
        # present -- see :func:`generate_slow_two_mode_cased_dataset`.
        return None

    noise = float(rng.uniform(0.0, noise_max))
    gather = synthesize_gather(
        geom, disp_modes, noise=noise, seed=int(rng.integers(2**31))
    )
    return SurrogateSample(
        params=params,
        freq=freq,
        slowness=slowness,
        attenuation=attenuation,
        gather=gather,
        mode_names=tuple(spec.name for spec in modes),
        mode_in_gather=in_gather,
        geom=geom,
        layer_params=layer_params,
        layer_names=layer_names,
        bond_index=bond_index,
    )


def generate_dataset(
    n: int,
    *,
    seed: int = 0,
    geom: ArrayGeometry | None = None,
    freq: np.ndarray | None = None,
    priors: FormationPriors | None = None,
    modes: Sequence[ModeSpec] = DEFAULT_MODES,
    cased_priors: CasingCementPriors | MicroannulusPriors | None = None,
    noise_max: float = 0.06,
    min_finite: int = 8,
    max_attempts: int | None = None,
    require_all_modes: bool = False,
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
    cased_priors : CasingCementPriors or MicroannulusPriors or None, default None
        When given, every sample is a cased-hole draw (casing + cement
        annulus passed as ``layers=`` to the layered mode solvers); ``None``
        produces open-hole samples. Pair with ``modes=``:data:`CASED_MODES`
        and a fast-formation ``priors`` -- see :func:`generate_cased_dataset`.
    noise_max : float, default 0.06
        Upper bound on per-gather noise fraction.
    min_finite : int, default 8
        Minimum finite slowness samples for a mode to enter the gather.
    require_all_modes : bool, default False
        Reject a draw unless every mode in ``modes`` cleared
        ``min_finite``; see :func:`generate_sample`.
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
            cased_priors=cased_priors,
            noise_max=noise_max,
            min_finite=min_finite,
            require_all_modes=require_all_modes,
        )
        if sample is not None:
            samples.append(sample)

    if len(samples) < n:
        raise RuntimeError(
            f"accepted only {len(samples)}/{n} samples in {attempts} "
            "attempts; loosen the priors or raise max_attempts"
        )
    return samples


def generate_cased_dataset(
    n: int,
    *,
    seed: int = 0,
    geom: ArrayGeometry | None = None,
    freq: np.ndarray | None = None,
    priors: FormationPriors | None = None,
    cased_priors: CasingCementPriors | MicroannulusPriors | None = None,
    modes: Sequence[ModeSpec] = CASED_MODES,
    noise_max: float = 0.06,
    min_finite: int = 8,
    max_attempts: int | None = None,
    require_all_modes: bool = False,
) -> list[SurrogateSample]:
    """
    Generate ``n`` cased-hole training pairs (casing + cement + formation).

    Convenience wrapper over :func:`generate_dataset` that pins the cased-hole
    configuration: the bound Stoneley mode (:data:`CASED_MODES`), a
    :class:`CasingCementPriors` annulus, and -- unless overridden -- a
    *fast-formation* :class:`FormationPriors` (``vs`` in ``1700-3000 m/s``, all
    above the fluid velocity), which is the regime where the cased Stoneley mode
    stays bound across the cement-stiffness range.

    Parameters
    ----------
    n : int
        Number of samples to accept.
    seed, geom, freq : as in :func:`generate_dataset`.
    priors : FormationPriors or None
        Formation ranges; defaults to a fast-only prior.
    cased_priors : CasingCementPriors or MicroannulusPriors or None
        Casing/cement ranges; defaults to :class:`CasingCementPriors`.
    modes : sequence of ModeSpec, default :data:`CASED_MODES`
        Cased-hole (layered-solver) modes.
    noise_max, min_finite, max_attempts : as in :func:`generate_dataset`.

    Returns
    -------
    list of SurrogateSample
        Exactly ``n`` cased-hole samples.
    """
    if priors is None:
        priors = FormationPriors(vs_min=1700.0, vs_max=3000.0)
    if cased_priors is None:
        cased_priors = CasingCementPriors()
    return generate_dataset(
        n,
        seed=seed,
        geom=geom,
        freq=freq,
        priors=priors,
        modes=modes,
        cased_priors=cased_priors,
        noise_max=noise_max,
        min_finite=min_finite,
        max_attempts=max_attempts,
        require_all_modes=require_all_modes,
    )


def debonded_freq_grid(n_freq: int = 32) -> np.ndarray:
    """
    Frequency grid for a debonded dataset -- coarser, and deliberately so.

    The microannulus solvers cost about 0.45 s per frequency per sample for
    the two modes together, against 0.004 s for the bonded layered solver:
    at the 128-point :func:`default_freq_grid` a debonded sample takes
    ~54 s, so the CLI's default ``--n 1000`` would run for 15 hours. At 32
    points it is ~14 s a sample, which is a few hours for a useful set.

    That is a real trade and not a free one -- a coarser curve is a coarser
    label. It is affordable here because both debonded branches are smooth:
    the Stoneley curve is nearly flat in gap width by construction, and the
    crack wave follows a cube-root law with no structure to alias.

    Parameters
    ----------
    n_freq : int, default 32
        Number of grid points over the standard 1-12 kHz band.

    Returns
    -------
    ndarray, shape (n_freq,)
    """
    return default_freq_grid(n_freq)


def generate_debonded_dataset(
    n: int,
    *,
    seed: int = 0,
    geom: ArrayGeometry | None = None,
    freq: np.ndarray | None = None,
    priors: FormationPriors | None = None,
    annulus_priors: MicroannulusPriors | None = None,
    modes: Sequence[ModeSpec] = DEBONDED_MODES,
    noise_max: float = 0.06,
    min_finite: int = 8,
    max_attempts: int | None = None,
) -> list[SurrogateSample]:
    """
    Generate ``n`` **debonded** cased-hole pairs (casing + gap + cement).

    Roadmap G.2. Pins the debonded configuration: a
    :class:`MicroannulusPriors` annulus, the two-branch
    :data:`DEBONDED_MODES`, a fast-formation :class:`FormationPriors`, and
    :func:`debonded_freq_grid` unless a grid is given.

    Read :class:`MicroannulusPriors` before using this. In short: the
    Stoneley branch labels a bonded/debonded *state* and is blind to the gap
    width; the crack-wave branch carries the width at roughly 100:1 and is
    recorded without being injected, because at these velocities it arrives
    outside the record. ``bond_index`` is driven by gap width here and by
    cement stiffness in :func:`generate_cased_dataset`, so the two datasets
    share a column name and not a meaning -- **do not pool them**.

    Parameters
    ----------
    n : int
        Number of samples to accept.
    seed, geom, freq : as in :func:`generate_dataset`.
    priors : FormationPriors or None
        Formation ranges; defaults to the same fast-only prior the bonded
        cased dataset uses.
    annulus_priors : MicroannulusPriors or None
        Debonded annulus ranges; defaults to :class:`MicroannulusPriors`.
    modes : sequence of ModeSpec, default :data:`DEBONDED_MODES`
    noise_max, min_finite, max_attempts : as in :func:`generate_dataset`.

    Returns
    -------
    list of SurrogateSample

    Notes
    -----
    This is slow: ~14 s a sample on the default grid, against ~0.5 s for a
    bonded cased sample. Budget accordingly, and see
    :func:`debonded_freq_grid` for why the grid is coarser.
    """
    return generate_dataset(
        n,
        seed=seed,
        geom=geom,
        freq=debonded_freq_grid() if freq is None else freq,
        priors=(
            priors
            if priors is not None
            else FormationPriors(vs_min=1700.0, vs_max=3000.0)
        ),
        cased_priors=(
            annulus_priors if annulus_priors is not None else MicroannulusPriors()
        ),
        modes=modes,
        noise_max=noise_max,
        min_finite=min_finite,
        max_attempts=max_attempts,
    )


def generate_slow_two_mode_cased_dataset(
    n: int,
    *,
    seed: int = 0,
    geom: ArrayGeometry | None = None,
    freq: np.ndarray | None = None,
    cased_priors: CasingCementPriors | MicroannulusPriors | None = None,
    noise_max: float = 0.06,
    min_finite: int = 8,
    max_attempts: int | None = None,
) -> list[SurrogateSample]:
    """
    Generate ``n`` cased-hole pairs carrying **both** the Stoneley and the
    flexural mode.

    The default cased dataset (:func:`generate_cased_dataset`) is single-mode
    because the two cased modes fail in opposite directions: flexural is sparse
    in fast formations, and Stoneley stops being bound as the formation slows
    away from the fluid velocity. :data:`SLOW_TWO_MODE_PRIORS` is the measured
    window where both are present, and this wrapper pins it together with
    :data:`CASED_TWO_MODES` and ``require_all_modes=True``, so every accepted
    sample carries both.

    "Both modes" means both are present and injected, not both bound at every
    frequency: the Stoneley is finite across the grid, the flexural mode over a
    contiguous upper sub-band of typically half of it. Read
    :data:`SLOW_TWO_MODE_PRIORS` for why.

    Read :data:`SLOW_TWO_MODE_PRIORS` before using this. The window is about
    80 m/s wide and **disjoint from the default cased prior**, so a dataset from
    here is not a subset of the usual one and the two must not be pooled. It
    suits cement-bond work, where the label is the bond index and formation
    ``V_S`` is a nuisance parameter; it is the wrong dataset for anything that
    needs formation-property variety.

    Parameters
    ----------
    n : int
        Number of samples to accept.
    seed, geom, freq : as in :func:`generate_dataset`.
    cased_priors : CasingCementPriors or MicroannulusPriors or None
        Casing/cement ranges; defaults to :class:`CasingCementPriors`.
    noise_max, min_finite, max_attempts : as in :func:`generate_dataset`.

    Returns
    -------
    list of SurrogateSample
        Exactly ``n`` cased-hole samples, each with two modes.

    See Also
    --------
    generate_cased_dataset : the single-mode default, over a fast prior.
    """
    return generate_cased_dataset(
        n,
        seed=seed,
        geom=geom,
        freq=freq,
        priors=SLOW_TWO_MODE_PRIORS,
        cased_priors=cased_priors,
        modes=CASED_TWO_MODES,
        noise_max=noise_max,
        min_finite=min_finite,
        max_attempts=max_attempts,
        require_all_modes=True,
    )


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
        * ``attenuation`` -- ``(N, M, n_f)`` leaky-mode attenuation (1/m),
          NaN for bound modes / absent frequencies
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
        * ``layer_params`` -- ``(N, L, 4)`` per-annular-layer
          ``[vp, vs, rho, thickness]`` (schema v4); ``L = 0`` for open-hole
        * ``layer_names`` -- ``(L,)`` str, layer labels (empty for open-hole)
        * ``bond_index`` -- ``(N,)`` float, cement-bond proxy in ``[0, 1]``;
          ``NaN`` for open-hole samples

        The three geometry scalars (added in schema v2) plus the
        ``gather``'s ``(n_rec, n_samples)`` fully reconstruct the
        acquisition :class:`fwap.ArrayGeometry`, making the waveform
        self-describing. The three cased-hole arrays (schema v4) describe
        the casing/cement annulus and the bond-evaluation label.

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
        "attenuation": np.stack([s.attenuation for s in samples]),
        "gather": np.stack([s.gather for s in samples]),
        "mode_in_gather": np.stack([s.mode_in_gather for s in samples]),
        "freq": np.asarray(samples[0].freq, dtype=float),
        "param_names": np.array(PARAM_NAMES),
        "mode_names": np.array(samples[0].mode_names),
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
        "dt": np.asarray(geom.dt, dtype=float),
        "tr_offset": np.asarray(geom.tr_offset, dtype=float),
        "dr": np.asarray(geom.dr, dtype=float),
        "layer_params": np.stack([s.layer_params for s in samples]),
        "layer_names": np.array(samples[0].layer_names, dtype=np.str_),
        "bond_index": np.array([s.bond_index for s in samples], dtype=float),
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
    parser.add_argument(
        "--cased",
        action="store_true",
        help=(
            "generate a cased-hole dataset (casing + cement annulus, bound "
            "Stoneley mode, fast-formation prior) instead of the open-hole default"
        ),
    )
    parser.add_argument(
        "--debonded",
        action="store_true",
        help=(
            "generate a debonded cased-hole dataset (casing + fluid "
            "microannulus + cement, Stoneley and crack-wave branches). "
            "Slow: ~14 s a sample, so --n-freq defaults to 32 here. "
            "Overrides --cased"
        ),
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
    # The three generators share a call shape but not a signature (the
    # debonded one names its annulus prior differently), so the variable is
    # typed by what main() actually uses of it.
    generate: Callable[..., list[SurrogateSample]]
    if args.debonded:
        # The debonded solvers cost ~100x the bonded ones per frequency, so
        # this dataset defaults to a coarser grid unless asked otherwise.
        n_freq = args.n_freq if "--n-freq" in (argv or sys.argv[1:]) else 32
        freq = default_freq_grid(n_freq, args.f_min, args.f_max)
        generate = generate_debonded_dataset
    else:
        freq = default_freq_grid(args.n_freq, args.f_min, args.f_max)
        generate = generate_cased_dataset if args.cased else generate_dataset
    samples = generate(
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
