r"""
Surrogate-in-the-loop inversion: dispersion curves -> formation parameters.

The M2 forward surrogate maps parameters to dispersion curves with a
differentiable network. That differentiability was the point of building it, and
until now nothing used it: this module closes the loop by putting the surrogate
inside a gradient-based optimisation. Given an observed dispersion curve, it
searches parameter space for the formation whose predicted curve matches,
back-propagating through the surrogate to get the gradient.

How this differs from the M3 inverse net
----------------------------------------
The M3 net is *amortised*: it learns gather -> parameters once, then answers in a
single forward pass. This is *iterative*: it optimises per sample, from a
different observable (the dispersion curve rather than the waveform). The two
are complements, not competitors --- iterative inversion pays per sample but
accepts constraints, priors and misfit weighting chosen at inference time, and
it reports a data misfit you can inspect when the answer looks wrong.

The control ships with the method
---------------------------------
:func:`invert_with_solver` runs the same inversion against fwap's **real** modal
solver instead of the surrogate. It is slower by roughly an order of magnitude,
and it is the number that makes the surrogate's result meaningful: "inversion
through a learned forward model works" is only interesting relative to inversion
through the exact one. Both take the same multi-start budget and the same
bounds, because a control given less is not a control.

What the comparison actually showed
-----------------------------------
On the default two-mode open-hole dataset, held-out samples, four active
parameters:

======  ==========  ========  ==========
param   surrogate   solver    no-skill
======  ==========  ========  ==========
vp        182 m/s     236       593
vs         39 m/s      31       351
rho        91 kg/m3     3.4     108
a         0.0038 m     0.0010   0.0096
======  ==========  ========  ==========

The surrogate route is about **ten times faster** per sample (~0.36 s vs ~3.5 s)
and **less accurate on three of the four parameters** --- dramatically so for
density, where the exact solver is better by a factor of about 25. The trade is
speed for accuracy, which is what a surrogate is supposed to buy; it is not a
free win, and an earlier draft of this module claimed otherwise because its
control was given a single start and a smaller budget than the surrogate got.

Why the errors land where they do
---------------------------------
The pattern is predictable, and usefully so. A surrogate cannot resolve a
parameter more finely than its own forward error allows: if the surrogate
mispredicts the curve by :math:`\epsilon`, then any parameter change that moves
the curve by less than :math:`\epsilon` is invisible to the search. Comparing
the surrogate's forward error against each parameter's curve *signature* over
its prior range predicts the inversion error well:

.. math::

    \text{expected error} \;\approx\;
    \frac{\text{forward error}}{\text{signature}} \times \text{range}

At a measured forward error of 1.5 us/ft this predicts 191 m/s for vp
(observed 182), 103 kg/m3 for rho (observed 91) and 0.0035 m for a
(observed 0.0038) --- within about 12% each. Shear velocity is the loose case
(predicted 24 m/s, observed 39) because its sensitivity is strongly non-linear,
so a single linearised signature understates the error at the slow end.

The practical consequence: **density has the weakest curve signature of the
four, so it is the first casualty of surrogate error** --- and improving it needs
a better surrogate, not a better optimiser. Check
:attr:`InversionResult.data_misfit` before trusting any recovered value, and be
most suspicious of a good-looking fit on a low-signature parameter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from fwap import flexural_dispersion, stoneley_dispersion
from scipy.optimize import least_squares

from sonic_ml.determinism import seed_everything
from sonic_ml.loader import DatasetBundle
from sonic_ml.models.forward import TrainedForwardSurrogate

#: Solvers used by :func:`invert_with_solver`, matching the generator's
#: ``DEFAULT_MODES`` order (Stoneley, flexural).
DEFAULT_SOLVERS = (stoneley_dispersion, flexural_dispersion)


@dataclass(frozen=True)
class InversionResult:
    """
    Outcome of an inversion for a batch of observed dispersion curves.

    Attributes
    ----------
    params : ndarray, shape (N, K), float64
        Recovered values of the active parameters, in raw physical units.
    active_names : tuple of str, length K
        Which parameters were solved for. Parameters held constant by the
        dataset (the borehole-fluid properties) are not inverted for; they are
        known, not fitted.
    data_misfit : ndarray, shape (N,), float64
        Final mean-squared misfit between predicted and observed curves, over
        the frequencies where the observation is finite. **Check this before
        trusting a recovered parameter**: a large misfit means the search never
        explained the data, and a small one on a weakly identifiable parameter
        means the data did not constrain it.
    n_starts : int
        Number of independent initialisations; the best-misfit start is kept
        per sample.
    method : str
        ``"surrogate"`` or ``"solver"``.
    """

    params: np.ndarray
    active_names: tuple[str, ...]
    data_misfit: np.ndarray
    n_starts: int
    method: str

    def value(self, name: str) -> np.ndarray:
        """
        Return the recovered column for one parameter.

        Raises
        ------
        KeyError
            If ``name`` was not among the inverted parameters.
        """
        try:
            index = self.active_names.index(name)
        except ValueError as exc:
            raise KeyError(
                f"{name!r} was not inverted for; have {self.active_names}"
            ) from exc
        return self.params[:, index]


def invert_with_surrogate(
    trained: TrainedForwardSurrogate,
    observed: np.ndarray,
    *,
    n_starts: int = 4,
    steps: int = 300,
    lr: float = 0.05,
    init_scale: float = 0.8,
    seed: int = 0,
    device: str = "cpu",
) -> InversionResult:
    """
    Invert observed dispersion curves by optimising through the surrogate.

    Optimises in the surrogate's *standardised* parameter space, so the search
    is naturally scaled and an initialisation of zero is the prior mean. Every
    sample in the batch is optimised simultaneously --- the per-sample losses are
    independent, so one backward pass serves the whole batch, which is most of
    where the speed advantage over the solver route comes from.

    Parameters
    ----------
    trained : TrainedForwardSurrogate
        A trained M2 surrogate; supplies both the network and the normalisers.
    observed : ndarray, shape (N, M, F)
        Observed per-mode slowness (s/m). ``NaN`` marks frequencies where a mode
        is absent; those entries are excluded from the misfit rather than
        imputed.
    n_starts : int, default 4
        Independent random initialisations. The inverse problem is non-convex,
        so a single start can land in a local minimum; the best-misfit start is
        kept per sample.
    steps : int, default 300
        Adam iterations per start.
    lr : float, default 0.05
        Adam learning rate, in standardised parameter units.
    init_scale : float, default 0.8
        Standard deviation of the random initialisation, in standardised units.
        The first start always begins at the prior mean.
    seed : int, default 0
    device : str, default "cpu"

    Returns
    -------
    InversionResult
    """
    observed = np.asarray(observed, dtype=float)
    if observed.ndim != 3:
        raise ValueError(f"observed must be (N, M, F), got shape {observed.shape}")

    seed_everything(seed)
    dev = torch.device(device)
    model = trained.model.to(dev).eval()
    n_samples = observed.shape[0]
    n_active = trained.param_std.n_active

    target = torch.as_tensor(
        np.nan_to_num(trained.slowness_norm.transform(observed), nan=0.0),
        dtype=torch.float32,
    ).to(dev)
    mask = torch.as_tensor(np.isfinite(observed)).to(dev)
    counts = mask.sum(dim=(1, 2)).clamp(min=1)

    def misfit(z: torch.Tensor) -> torch.Tensor:
        predicted, _ = model(z)
        return (((predicted - target) ** 2) * mask).sum(dim=(1, 2)) / counts

    best_z: torch.Tensor | None = None
    best_misfit: torch.Tensor | None = None

    for start in range(max(n_starts, 1)):
        generator = torch.Generator().manual_seed(seed + start)
        if start == 0:
            z0 = torch.zeros(n_samples, n_active)  # the prior mean
        else:
            z0 = torch.randn(n_samples, n_active, generator=generator) * init_scale
        z = z0.to(dev).requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            misfit(z).sum().backward()
            optimizer.step()

        with torch.no_grad():
            final = misfit(z)
        if best_z is None or best_misfit is None:
            best_z, best_misfit = z.detach().clone(), final.clone()
        else:
            improved = final < best_misfit
            best_z[improved] = z.detach()[improved]
            best_misfit[improved] = final[improved]

    assert best_z is not None and best_misfit is not None
    return InversionResult(
        params=trained.param_std.inverse_transform(best_z.cpu().numpy()),
        active_names=trained.param_std.active_names,
        data_misfit=best_misfit.cpu().numpy().astype(float),
        n_starts=max(n_starts, 1),
        method="surrogate",
    )


def invert_with_solver(
    bundle: DatasetBundle,
    indices: np.ndarray,
    *,
    active_names: Sequence[str] | None = None,
    n_starts: int = 4,
    max_nfev: int = 400,
    solvers: Sequence[object] = DEFAULT_SOLVERS,
    seed: int = 0,
) -> InversionResult:
    """
    Reference control: invert the same curves through fwap's real modal solver.

    Least-squares over the active parameters, calling the actual
    cylindrical-Biot solvers each iteration with finite-difference gradients.
    This is what the surrogate route has to be judged against; without it,
    "gradient-based inversion through a learned forward model works" is a claim
    with no scale.

    Both routes are given the same advantages on purpose: identical multi-start
    budget, and bounds taken from the dataset's own parameter ranges. The
    constant borehole-fluid properties are supplied to both as known values ---
    they are measured, not fitted, so using them is not leakage.

    Frequencies where a solve fails contribute **zero** residual rather than a
    large sentinel. Injecting a big fake residual would let the optimiser chase
    solver failures instead of data misfit.

    Parameters
    ----------
    bundle : DatasetBundle
        Supplies the observed curves, the frequency grid, the known constants
        and the parameter bounds.
    indices : ndarray of int
        Samples to invert.
    active_names : sequence of str or None
        Parameters to solve for. ``None`` uses every column that varies in the
        dataset.
    n_starts : int, default 4
        Independent initialisations; the first is the midpoint of the bounds.
    max_nfev : int, default 400
        Residual evaluations per start, passed to
        :func:`scipy.optimize.least_squares`.
    solvers : sequence of callable
        Per-mode dispersion solvers, aligned with the bundle's ``mode_names``.
    seed : int, default 0

    Returns
    -------
    InversionResult
    """
    idx = np.asarray(indices, dtype=int)
    freq = np.asarray(bundle.freq, dtype=float)
    column = {name: i for i, name in enumerate(bundle.param_names)}

    if active_names is None:
        varying = bundle.params.std(axis=0) > 0.0
        active_names = tuple(
            name
            for name, is_varying in zip(bundle.param_names, varying, strict=True)
            if is_varying
        )
    active = tuple(active_names)
    if not active:
        raise ValueError("no active parameters to invert for")

    lower = np.array([bundle.params[:, column[n]].min() for n in active])
    upper = np.array([bundle.params[:, column[n]].max() for n in active])

    def forward(param_vector: np.ndarray) -> np.ndarray:
        kwargs = {n: float(param_vector[column[n]]) for n in bundle.param_names}
        curves = []
        for solver in solvers:
            try:
                curves.append(solver(freq, **kwargs).slowness)  # type: ignore[operator]
            except (ValueError, RuntimeError):
                curves.append(np.full(freq.size, np.nan))
        return np.stack(curves)

    recovered = np.empty((idx.size, len(active)), dtype=float)
    misfits = np.empty(idx.size, dtype=float)

    for row, sample in enumerate(idx):
        observed = bundle.slowness[sample]
        finite = np.isfinite(observed)
        base = bundle.params[sample].copy()  # known constants live here

        def residual(values: np.ndarray, _o=observed, _f=finite, _b=base) -> np.ndarray:
            vector = _b.copy()
            for j, name in enumerate(active):
                vector[column[name]] = values[j]
            predicted = forward(vector)[_f]
            out = np.zeros(int(_f.sum()), dtype=float)
            ok = np.isfinite(predicted)
            out[ok] = (predicted[ok] - _o[_f][ok]) / 1.0e-6  # us/m, an O(1) scale
            return out

        rng = np.random.default_rng(seed + int(sample))
        best_values: np.ndarray | None = None
        best_cost = np.inf
        for start in range(max(n_starts, 1)):
            if start == 0:
                x0 = 0.5 * (lower + upper)
            else:
                x0 = lower + rng.random(len(active)) * (upper - lower)
            try:
                fit = least_squares(
                    residual, x0, bounds=(lower, upper), max_nfev=max_nfev
                )
            except (ValueError, RuntimeError):  # pragma: no cover - solver blowup
                continue
            if fit.cost < best_cost:
                best_values, best_cost = fit.x, float(fit.cost)

        if best_values is None:  # pragma: no cover - every start failed
            recovered[row] = 0.5 * (lower + upper)
            misfits[row] = np.inf
            continue
        recovered[row] = best_values
        n_finite = max(int(finite.sum()), 1)
        misfits[row] = 2.0 * best_cost / n_finite

    return InversionResult(
        params=recovered,
        active_names=active,
        data_misfit=misfits,
        n_starts=max(n_starts, 1),
        method="solver",
    )


def inversion_mae(
    result: InversionResult,
    bundle: DatasetBundle,
    indices: np.ndarray,
    name: str,
) -> float:
    """
    Mean absolute error of one recovered parameter, in its own units.

    Parameters
    ----------
    result : InversionResult
    bundle : DatasetBundle
        Supplies the truth.
    indices : ndarray of int
        The samples that were inverted, in the same order.
    name : str
        Parameter to score.

    Returns
    -------
    float
    """
    idx = np.asarray(indices, dtype=int)
    truth = bundle.param(name)[idx]
    return float(np.mean(np.abs(result.value(name) - truth)))


def no_skill_mae(
    bundle: DatasetBundle,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    name: str,
) -> float:
    """
    MAE of predicting the training mean --- the reference every claim needs.

    A recovered parameter that does not beat this was not recovered; the search
    simply returned something plausible. Reporting inversion accuracy without it
    hides exactly that case.

    Parameters
    ----------
    bundle : DatasetBundle
    train_indices, test_indices : ndarray of int
    name : str

    Returns
    -------
    float
    """
    values = bundle.param(name)
    reference = float(values[np.asarray(train_indices, dtype=int)].mean())
    truth = values[np.asarray(test_indices, dtype=int)]
    return float(np.mean(np.abs(truth - reference)))
