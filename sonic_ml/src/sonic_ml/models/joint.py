r"""
Joint multi-depth inversion: one coupled solve for a whole logged interval.

:mod:`sonic_ml.models.inversion` inverts each depth frame on its own. That
throws away the one thing a *log* has that a single measurement does not:
adjacent frames are not independent draws, because rock does not change
completely between one sample and the next half-foot. This module uses that,
adding a penalty on how much the recovered parameters change from frame to
frame and solving the whole interval as one optimisation.

What it is actually exploiting, and what it is not
--------------------------------------------------
The first draft of this module asserted that coupling averages away the
**surrogate's own forward error**, on the reasoning that the error varies from
one point of parameter space to the next and so is effectively random across
depth. That is wrong, and the noise-free experiment below falsifies it: with
clean curves the best available penalty is *no penalty* on almost every profile
and parameter, and the largest gain anywhere in the sweep is 1.4%.

The reason is worth stating, because it is easy to get backwards. Inside a bed
every frame carries **identical** parameters, so the surrogate is evaluated at
the identical input and makes the **identical** error. That error is perfectly
correlated down the bed, not random, and averaging identical numbers returns the
number. There is nothing there to cancel. What does differ between frames of one
bed is the observation noise --- and that is the whole of the mechanism. Depth
coupling here is noise averaging, and where there is no noise it is pure bias.

So the effect only appears once the curves are noisy, which is also when it is
large: at a realistic 2 us/ft of dispersion-picking scatter it removes 31-45% of
the error across the four parameters. The predicted bed/boundary asymmetry does
show up in that regime --- the gain is concentrated inside beds and contacts are
damaged --- and :func:`bed_vs_boundary_mae` reports the two separately rather
than hiding the trade inside one average.

Why the control is not just "lambda = 0"
----------------------------------------
:func:`invert_joint` **warm-starts from the independent solution** ---
:func:`~sonic_ml.models.inversion.invert_with_surrogate`, at its own best
multi-start setting --- and then refines it under the coupled objective. This
is deliberate. Multi-start on a coupled objective is much weaker than
multi-start on a separable one (a whole profile gets one draw where independent
inversion gets one per frame), so a joint solver that did its own cold multi-start
would look better than independent inversion mostly by being a worse optimiser
in a way that happened to help, or worse by being handicapped. Refining *from*
the independent answer removes that confound: both arms see the identical
starting point, and every difference in the result is attributable to the
penalty. An earlier draft of this experiment did cold whole-profile multi-start
and made the independent control look 3x worse than it is.

The control that decides whether this was worth building
--------------------------------------------------------
There is a much simpler way to use the same prior: inverting each frame on its
own and then running a moving average down the log. If that gets the same
answer, a coupled solve is complexity for nothing --- so
:func:`smooth_independent` ships beside :func:`invert_joint` and is scored on
every result below, both arms tuned to their own best setting.

What was measured
-----------------
Five 60-frame bedded profiles (7-9 beds each) from :func:`synthesize_profile`, a
surrogate trained to 2.5 us/ft forward error, scored against the ``lam=0``
control at the same warm start and the same refinement budget. Mean absolute
error over the five profiles, at 2 us/ft of picking noise. Each method appears
twice: at the setting cross-validation actually chose, and at its **oracle**
setting (the best in the grid, scored against truth --- an upper bound nobody can
reach on a real log):

=====================  =========  =========  ========  =========
2 us/ft noise           vp (m/s)   vs (m/s)   rho       a (m)
=====================  =========  =========  ========  =========
independent               505.7      58.2      161.1     0.00910
joint, CV-selected        419.9      45.4      113.7     0.00746
joint, oracle             351.5      36.0       89.2     0.00608
smoothed, CV-selected     505.7      58.2      161.1     0.00910
smoothed, oracle          382.4      58.2       92.2     0.00739
=====================  =========  =========  ========  =========

Read the oracle rows first: coupling removes 31-45% of the error against
smoothing's 24%, 0%, 43%, 19% --- better on all four, but narrowly on ``rho``
(44.6 against 42.8, a tie in five profiles) and overwhelmingly on ``vs``, where
smoothing's best window is a single frame and averaging the independent log
cannot help at all.

Then read the CV rows, because they are what a user gets. Coupling keeps
17-29%. Smoothing keeps **nothing** --- cross-validation chose a one-frame window
on all five profiles, which is the honest outcome of a method that cannot be
tuned from data: post-hoc averaging moves parameters away from the data-fitting
optimum and so degrades held-out misfit monotonically, whatever it does to
accuracy. Its oracle numbers are real and unreachable.

That asymmetry is the argument for doing the coupling inside the objective
rather than after it. A frame whose data pins it down can resist being dragged
toward its neighbours, which is also why coupling is gentler at bed contacts ---
at oracle settings, smoothing leaves a boundary error of 535 m/s on ``vp``
against coupling's 421.

Noise-free, all of it vanishes. The oracle penalty is ``lam=0`` for ``vp`` and
``rho`` in every profile, and the largest oracle gain anywhere is 1.4% ---
selection noise. This is the falsification described above, and it is why the
module leads with the mechanism rather than the headline number.

Choosing lambda without looking at the truth
--------------------------------------------
:func:`select_lambda` holds out a random subset of the observed *(mode,
frequency)* entries, inverts using only the rest, and scores each candidate on
entries it never fit --- ordinary cross-validation, using no truth anywhere.
This is what produced the CV rows above.

Selecting this way costs about half the gain: 17-29% against the oracle's
31-45%. That is the number to quote as achievable.

In the noise-free regime the selector **fails**, and not harmlessly. It chooses
a non-zero penalty on four of five profiles when the correct answer is zero, and
the resulting profile is **18-29% worse than not coupling at all**. The
mechanism is a standard cross-validation pathology worth naming: withholding
30% of the frequencies leaves each frame less determined during selection than
it will be at inference, so the prior looks more valuable to the selector than
it actually is. Cross-validating on frequencies also only indirectly asks
whether the *depth* prior is real, which is the other half of the problem.

Note which way that cuts. On a clean log the untunable control is the *safer*
method: smoothing's refusal to be tuned costs it 24-43% of available accuracy in
the noisy case and saves it 18-29% in the clean one. Being impossible to talk
into a bad setting is worth something.

The practical consequence is a condition on using this at all: **depth coupling
pays when the dispersion picks are noisy and costs when they are clean, and the
selector cannot reliably tell you which case you are in.** If the picking
scatter is known to be small, pass ``lam=0`` and keep the independent answer.

Two penalties, because squared differences hate bed contacts
------------------------------------------------------------
Everything above uses ``penalty="l2"``, which charges each frame-to-frame
transition its square. That has a consequence worth seeing plainly: given a
fixed amount of total change, squaring makes it **four times cheaper** to spread
it over four gentle frames than to deliver it as one contact. A bed boundary is
therefore the single most expensive feature in the log, and the optimiser spends
its budget flattening exactly the thing a logger wanted to keep. The
bed/boundary split shows the damage as it happens, but showing it is not fixing
it.

``penalty="tv"`` fixes the incentive. It uses the pseudo-Huber norm --- quadratic
for transitions small compared with ``tv_eps``, linear beyond --- which is very
nearly **indifferent** to how a given total change is distributed. It is worth
being precise about the mechanism, because the obvious reading is wrong: TV does
not *prefer* sharp contacts, it simply stops paying to remove them. Small
wiggles inside a bed are still suppressed quadratically; a real jump is charged
once, at its size, and then left alone.

Whether that indifference is worth having is a measurement, and
:func:`contact_precision` is the one that answers it --- can you still find the
bed boundaries in the recovered log? It is quoted against
:func:`no_skill_contact_precision`, the score blind guessing already achieves,
because on a log with many contacts a bare precision figure flatters everything.

On the same five profiles at 2 us/ft, each penalty at its own CV-selected
weight, the diagnosis is confirmed and the fix works:

=============  ==========  ==========  ==============
2 us/ft, CV    vp overall  vp at beds  contact precision
=============  ==========  ==========  ==============
independent       505.7       485.8         0.492
``"l2"``          419.9       499.7         0.832
``"tv"``          388.2       406.4         0.914
no-skill            --          --          0.363
=============  ==========  ==========  ==============

The middle row is the problem this option exists for: at the weight
cross-validation picks, squared-difference coupling improves the log overall
(506 to 420) while making the **bed contacts worse than not coupling at all**
(486 to 500). It buys its average by spending the boundaries. ``"tv"`` improves
both --- 388 overall and 406 at the contacts --- and finds 91% of the boundaries
against 83%.

Where ``"tv"`` stops winning
-----------------------------
That test bed is piecewise-constant, which is the most favourable possible
setting for a penalty designed to preserve sharp contacts, so it cannot settle
the question on its own. Re-running it with ``gradation_frames=4``, which ramps
each contact over four frames and leaves nothing sharp to preserve, the
advantage narrows and partly **inverts**: ``"tv"`` stays slightly ahead on
overall error (321 against 329 on ``vp``), but is now *worse* at the transition
frames (262 against 241) and worse at locating them (0.72 against 0.84). With
no sharp contacts in the ground truth, smoothing through the transition is the
correct prior and tolerating jumps is the wrong one.

Hence the default is still ``"l2"``, and the recommendation is a question about
your rock rather than about the algorithm: **on a bedded log, pass
``penalty="tv"``; on a gradational one, do not.** If you do not know, the
overall-error column says ``"tv"`` is never worse in these measurements, but it
is only clearly better when the contacts are real.

Scope
-----
Depth coupling is a *prior*, and like every prior it buys variance reduction by
paying bias. Where a formation genuinely changes fast --- thin beds, a washed-out
interval, a fracture swarm --- the penalty is the wrong assumption and this method
will smooth through the feature it was supposed to resolve. The right reading of
:func:`bed_vs_boundary_mae` is not "how much did the average improve" but "was
the boundary damage worth the in-bed gain for what I am logging for".
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from sonic_ml.determinism import seed_everything
from sonic_ml.loader import DatasetBundle
from sonic_ml.models.forward import TrainedForwardSurrogate
from sonic_ml.models.inversion import (
    DEFAULT_SOLVERS,
    InversionResult,
    invert_with_surrogate,
)

#: Slowness in s/m -> us/ft.
_US_PER_FT: float = 0.3048e6

#: Default depth increment (m) --- 0.5 ft, the usual sonic log sampling.
DEFAULT_SPACING: float = 0.1524


@dataclass(frozen=True)
class DepthProfile:
    """
    A synthetic logged interval: depth-ordered formations and their curves.

    Unlike a dataset bundle, whose samples are independent draws, the frames
    here are *ordered in depth* and piecewise constant --- a stack of beds. That
    structure is what a depth-coupled inversion is meant to exploit, and
    :attr:`bed_boundaries` is what makes it possible to check whether it
    exploited it honestly (by scoring inside beds and across contacts
    separately).

    Attributes
    ----------
    depths : ndarray, shape (D,), float64
        Measured depth of each frame (m), increasing.
    params : ndarray, shape (D, K), float64
        True formation parameters, columns in :attr:`active_names` order.
    active_names : tuple of str, length K
        Parameter names, matching a trained surrogate's active columns.
    slowness : ndarray, shape (D, M, F), float64
        Observed per-mode phase slowness (s/m), including noise if any was
        added. ``NaN`` where a mode is absent.
    clean_slowness : ndarray, shape (D, M, F), float64
        The same curves before noise --- the noise-free forward model.
    freq : ndarray, shape (F,), float64
        Frequency grid (Hz).
    bed_boundaries : ndarray, shape (B,), int
        Frame indices at which a new bed starts (so frame ``b - 1`` and frame
        ``b`` are different rock). Empty for a single-bed profile.
    noise_us_per_ft : float
        Standard deviation of the additive slowness noise, in us/ft; ``0.0``
        for a noise-free profile.
    """

    depths: np.ndarray
    params: np.ndarray
    active_names: tuple[str, ...]
    slowness: np.ndarray
    clean_slowness: np.ndarray
    freq: np.ndarray
    bed_boundaries: np.ndarray
    noise_us_per_ft: float

    @property
    def n_depths(self) -> int:
        """Number of depth frames."""
        return int(self.params.shape[0])

    @property
    def n_beds(self) -> int:
        """Number of distinct beds."""
        return int(self.bed_boundaries.size) + 1

    def value(self, name: str) -> np.ndarray:
        """
        True depth series for one parameter.

        Raises
        ------
        KeyError
            If ``name`` is not one of :attr:`active_names`.
        """
        try:
            index = self.active_names.index(name)
        except ValueError as exc:
            raise KeyError(
                f"{name!r} is not in this profile; have {self.active_names}"
            ) from exc
        return self.params[:, index]

    def boundary_mask(self, halo: int = 1) -> np.ndarray:
        """
        Frames within ``halo`` samples of a bed contact.

        Parameters
        ----------
        halo : int, default 1
            How many frames on each side of a contact to count as "at the
            boundary". ``1`` marks the last frame of the upper bed and the
            first frame of the lower one.

        Returns
        -------
        ndarray, shape (D,), bool
        """
        mask = np.zeros(self.n_depths, dtype=bool)
        for b in self.bed_boundaries:
            mask[max(0, int(b) - halo) : min(self.n_depths, int(b) + halo)] = True
        return mask


def _gradate(
    full: np.ndarray,
    boundaries: list[int],
    width: int,
    varying: np.ndarray,
) -> np.ndarray:
    """
    Replace sharp contacts with linear ramps ``width`` frames wide.

    Turns a blocky log into a gradational one *without* changing the bed values
    themselves, so the two versions differ only in how the transition is
    reached. That is what makes them a fair pair for asking whether a
    contact-preserving penalty still helps when there are no sharp contacts to
    preserve.
    """
    out = full.copy()
    half = max(width // 2, 1)
    for b in boundaries:
        lo, hi = max(0, b - half), min(full.shape[0] - 1, b + half)
        if hi <= lo:
            continue
        span = hi - lo
        for i in range(lo + 1, hi):
            weight = (i - lo) / span
            out[i, varying] = (1.0 - weight) * full[lo, varying] + weight * full[
                hi, varying
            ]
    return out


def synthesize_profile(
    bundle: DatasetBundle,
    *,
    n_depths: int = 60,
    mean_bed_frames: float = 8.0,
    gradation_frames: int = 0,
    spacing: float = DEFAULT_SPACING,
    noise_us_per_ft: float = 2.0,
    seed: int = 0,
    solvers: Sequence[object] = DEFAULT_SOLVERS,
) -> DepthProfile:
    """
    Build a bedded synthetic log and forward-model it with fwap's real solvers.

    Bed values are drawn from the ranges **observed in** ``bundle``, not from
    an independent prior, so the profile is inside the surrogate's training
    support by construction. That matters: a profile drawn from wider ranges
    would measure extrapolation, and any conclusion about depth coupling would
    be confounded by the surrogate being asked questions it was never trained
    on.

    Where the bundle has both ``vp`` and ``vs``, the ``vp/vs`` *ratio* is
    sampled over its observed range and ``vp`` is formed as ``vs * ratio``.
    Sampling the two velocities independently from their own marginals would
    produce ratios the surrogate never saw --- and could violate the
    ``vp > vs`` constraint the modal solvers enforce.

    Parameters
    ----------
    bundle : DatasetBundle
        Supplies the parameter ranges, the frequency grid, the constant
        (known) fluid properties and the mode count.
    n_depths : int, default 60
        Number of depth frames.
    mean_bed_frames : float, default 8.0
        Mean bed thickness in frames; thicknesses are Poisson-drawn with a
        floor of 2 frames.
    gradation_frames : int, default 0
        Width, in frames, over which a contact ramps linearly instead of
        stepping. ``0`` gives sharp contacts. Non-zero builds the *unfavourable*
        case for a contact-preserving penalty --- a log with no sharp contacts to
        preserve --- which is what makes the ``"tv"`` versus ``"l2"`` comparison
        more than a restatement of how the test data was generated.
        :attr:`DepthProfile.bed_boundaries` still marks the ramp centres, so a
        gradational profile's "boundary" frames are where the log is changing
        fastest rather than where it jumps.
    spacing : float, default 0.1524
        Depth increment (m).
    noise_us_per_ft : float, default 2.0
        Standard deviation of additive Gaussian slowness noise (us/ft),
        representing dispersion-picking uncertainty. ``0.0`` for a noise-free
        profile.
    seed : int, default 0
    solvers : sequence of callable
        Per-mode dispersion solvers aligned with ``bundle.mode_names``.

    Returns
    -------
    DepthProfile

    Notes
    -----
    A frequency at which a solve fails or a mode is unbound is left ``NaN``,
    exactly as the dataset generator leaves it, so the inversion sees the same
    kind of gaps it sees on dataset samples.
    """
    if n_depths < 2:
        raise ValueError(f"n_depths must be at least 2, got {n_depths}")
    if len(solvers) != len(bundle.mode_names):
        raise ValueError(
            f"got {len(solvers)} solvers for {len(bundle.mode_names)} modes"
        )

    rng = np.random.default_rng(seed)
    freq = np.asarray(bundle.freq, dtype=float)
    names = bundle.param_names
    column = {name: i for i, name in enumerate(names)}
    varying = bundle.params.std(axis=0) > 0.0
    active_names = tuple(n for n, v in zip(names, varying, strict=True) if v)

    lo = bundle.params.min(axis=0)
    hi = bundle.params.max(axis=0)
    paired = "vp" in column and "vs" in column
    if paired:
        ratio = bundle.params[:, column["vp"]] / bundle.params[:, column["vs"]]
        ratio_lo, ratio_hi = float(ratio.min()), float(ratio.max())

    rows: list[np.ndarray] = []
    boundaries: list[int] = []
    depth = 0
    while depth < n_depths:
        thickness = max(2, int(rng.poisson(mean_bed_frames)))
        bed = bundle.params[0].copy()  # carries the constant columns
        for i, name in enumerate(names):
            if varying[i] and not (paired and name == "vp"):
                bed[i] = rng.uniform(lo[i], hi[i])
        if paired:
            bed[column["vp"]] = bed[column["vs"]] * rng.uniform(ratio_lo, ratio_hi)
        take = min(thickness, n_depths - depth)
        rows.extend([bed] * take)
        if depth > 0:
            boundaries.append(depth)
        depth += take

    full = np.array(rows, dtype=float)
    if gradation_frames > 0 and boundaries:
        full = _gradate(full, boundaries, gradation_frames, varying)

    clean = np.full((n_depths, len(bundle.mode_names), freq.size), np.nan)
    for d in range(n_depths):
        kwargs = {name: float(full[d, i]) for i, name in enumerate(names)}
        for m, solver in enumerate(solvers):
            try:
                clean[d, m] = solver(freq, **kwargs).slowness  # type: ignore[operator]
            except (ValueError, RuntimeError):  # pragma: no cover - solver refusal
                pass

    observed = clean.copy()
    if noise_us_per_ft:
        observed = observed + rng.normal(0.0, noise_us_per_ft / _US_PER_FT, clean.shape)

    active_index = [column[n] for n in active_names]
    return DepthProfile(
        depths=np.arange(n_depths, dtype=float) * spacing,
        params=full[:, active_index],
        active_names=active_names,
        slowness=observed,
        clean_slowness=clean,
        freq=freq,
        bed_boundaries=np.array(boundaries, dtype=int),
        noise_us_per_ft=float(noise_us_per_ft),
    )


def _misfit_terms(
    trained: TrainedForwardSurrogate,
    observed: np.ndarray,
    fit_mask: np.ndarray | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalised target, boolean fit mask and per-frame finite counts."""
    target = torch.as_tensor(
        np.nan_to_num(trained.slowness_norm.transform(observed), nan=0.0),
        dtype=torch.float32,
    )
    mask = torch.as_tensor(np.isfinite(observed))
    if fit_mask is not None:
        mask = mask & torch.as_tensor(np.asarray(fit_mask, dtype=bool))
    return target, mask, mask.sum(dim=(1, 2)).clamp(min=1)


def _roughness_term(
    penalty: str, tv_eps: float
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Build the depth-roughness term, mean over the ``D-1`` frame transitions.

    ``"l2"`` squares each transition, so *how* a given amount of change is
    distributed matters enormously: one contact of size 4 costs four times what
    the same total change spread over four frames costs. That is a standing
    incentive to erase contacts. ``"tv"`` uses the pseudo-Huber norm, quadratic
    while a transition is small compared with ``tv_eps`` and linear once it is
    large, which makes it very nearly **indifferent** to the distribution --- it
    prices the total amount of change, not its shape. Indifference, not a
    preference for jumps, is what lets a contact survive.
    """
    if penalty == "l2":

        def l2(z: torch.Tensor) -> torch.Tensor:
            return ((z[1:] - z[:-1]) ** 2).sum(dim=1).mean()

        return l2

    if penalty == "tv":

        def tv(z: torch.Tensor) -> torch.Tensor:
            squared = ((z[1:] - z[:-1]) ** 2).sum(dim=1)
            # Algebraically sqrt(s + eps^2) - eps, written to avoid the
            # catastrophic cancellation that form suffers in float32 once
            # eps greatly exceeds the transition size.
            return (squared / (torch.sqrt(squared + tv_eps**2) + tv_eps)).mean()

        return tv

    raise ValueError(f"penalty must be 'l2' or 'tv', got {penalty!r}")


def invert_joint(
    trained: TrainedForwardSurrogate,
    observed: np.ndarray,
    *,
    lam: float,
    penalty: str = "l2",
    tv_eps: float = 0.05,
    steps: int = 400,
    lr: float = 0.02,
    warm_start: np.ndarray | None = None,
    n_starts: int = 6,
    fit_mask: np.ndarray | None = None,
    seed: int = 0,
) -> InversionResult:
    r"""
    Invert a depth-ordered stack of curves as one depth-coupled problem.

    Minimises, over the standardised parameter profile :math:`z_{1..D}`,

    .. math::

        \frac{1}{D}\sum_d \text{misfit}_d(z_d)
        \;+\; \frac{\lambda}{D-1}\sum_d \rho(z_{d+1} - z_d)

    where :math:`\rho` is squared length for ``penalty="l2"`` and the
    pseudo-Huber norm :math:`\sqrt{\lVert\cdot\rVert^2 + \epsilon^2} - \epsilon`
    for ``penalty="tv"``. Both terms are means, so ``lam`` is a ratio of two
    comparable quantities rather than a number whose useful scale depends on how
    many frames you happened to log. Because :math:`z` is standardised, one unit
    of roughness means the same thing for every parameter, and ``lam`` is
    dimensionless.

    The two penalties differ in what they do at a bed contact, and that is the
    whole reason both exist. Squared differences price a jump at its square, so
    a real contact is the most expensive feature in the log and the optimiser
    spends its budget flattening exactly the thing you wanted to keep. The
    pseudo-Huber cost is linear once a jump is large, so the marginal price of a
    contact stops rising and it can survive. See the module docstring for what
    that is measured to be worth.

    The optimisation starts from the independent solution (see the module
    docstring for why that is the honest choice) and refines it. With
    ``lam=0`` the objective is separable and this is simply further independent
    optimisation --- it cannot use depth structure, which is exactly what makes
    it the control.

    Parameters
    ----------
    trained : TrainedForwardSurrogate
        Trained M2 surrogate supplying the differentiable forward model.
    observed : ndarray, shape (D, M, F)
        Depth-ordered observed slowness (s/m); ``NaN`` marks absent modes.
        **Row order is meaningful** --- rows must be sorted by depth, or the
        roughness penalty couples frames that are not neighbours.
    lam : float
        Penalty weight; ``0.0`` disables coupling. Must be non-negative.
    penalty : {"l2", "tv"}, default "l2"
        Roughness cost. ``"l2"`` penalises squared frame-to-frame change;
        ``"tv"`` uses the pseudo-Huber norm, which preserves bed contacts. The
        default is ``"l2"`` for continuity with the measurements already
        published in the module docstring, not because it is the better choice
        on a bedded log --- see there for which to pick.
    tv_eps : float, default 0.05
        Transition size (in standardised units) at which ``"tv"`` crosses from
        quadratic to linear. Smaller approaches true total variation and is
        harder to optimise; larger degenerates toward ``"l2"``. Ignored when
        ``penalty="l2"``. Must be positive.
    steps : int, default 400
        Adam iterations of the coupled refinement.
    lr : float, default 0.02
        Adam learning rate in standardised units. Lower than the independent
        default because the refinement starts from an already-good point.
    warm_start : ndarray, shape (D, K), or None
        Starting parameters in raw units. ``None`` runs
        :func:`~sonic_ml.models.inversion.invert_with_surrogate` to get them.
        Pass an explicit value to reuse one independent solve across a lambda
        sweep instead of repeating it.
    n_starts : int, default 6
        Multi-start budget for the independent warm start (ignored when
        ``warm_start`` is given).
    fit_mask : ndarray, shape (D, M, F), bool, or None
        Restrict the misfit to these entries. Used by :func:`select_lambda` to
        hold data out; ``None`` fits everything finite.
    seed : int, default 0

    Returns
    -------
    InversionResult
        With ``method="joint"``. ``data_misfit`` is per depth frame and is
        computed on the fitted entries only.

    Raises
    ------
    ValueError
        If ``observed`` is not 3-D, if fewer than two frames are given, if
        ``lam`` is negative, if ``tv_eps`` is not positive, or if ``penalty`` is
        not a recognised name.
    """
    observed = np.asarray(observed, dtype=float)
    if observed.ndim != 3:
        raise ValueError(f"observed must be (D, M, F), got shape {observed.shape}")
    if observed.shape[0] < 2:
        raise ValueError("joint inversion needs at least 2 depth frames")
    if lam < 0.0:
        raise ValueError(f"lam must be non-negative, got {lam}")
    if tv_eps <= 0.0:
        raise ValueError(f"tv_eps must be positive, got {tv_eps}")

    seed_everything(seed)
    if warm_start is None:
        warm_start = invert_with_surrogate(
            trained, observed, n_starts=n_starts, steps=400, seed=seed
        ).params

    std = trained.param_std
    mean = std.mean[std.active]
    scale = std.std[std.active]
    z0 = (np.asarray(warm_start, dtype=float) - mean) / scale

    model = trained.model.eval()
    target, mask, counts = _misfit_terms(trained, observed, fit_mask)

    def data_misfit(z: torch.Tensor) -> torch.Tensor:
        predicted, _ = model(z)
        return (((predicted - target) ** 2) * mask).sum(dim=(1, 2)) / counts

    roughness = _roughness_term(penalty, tv_eps)
    z = torch.as_tensor(z0, dtype=torch.float32).clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = data_misfit(z).mean()
        if lam > 0.0:
            loss = loss + lam * roughness(z)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final = data_misfit(z)
    return InversionResult(
        params=std.inverse_transform(z.detach().numpy()),
        active_names=std.active_names,
        data_misfit=final.numpy().astype(float),
        n_starts=n_starts,
        method="joint",
    )


@dataclass(frozen=True)
class LambdaSelection:
    """
    A penalty weight chosen from data alone.

    Attributes
    ----------
    lam : float
        The selected candidate --- the one with the lowest held-out misfit.
    candidates : tuple of float
        Candidates scored, in the order given.
    holdout_misfit : ndarray, shape (len(candidates),), float64
        Mean squared misfit on the held-out entries, in normalised slowness
        units. Comparable across candidates, not across datasets.
    n_holdout : int
        Number of held-out ``(depth, mode, frequency)`` entries.
    """

    lam: float
    candidates: tuple[float, ...]
    holdout_misfit: np.ndarray
    n_holdout: int

    @property
    def selected_coupling(self) -> bool:
        """``True`` if the data preferred some coupling over none."""
        return self.lam > 0.0


def select_lambda(
    trained: TrainedForwardSurrogate,
    observed: np.ndarray,
    candidates: Sequence[float],
    *,
    penalty: str = "l2",
    tv_eps: float = 0.05,
    holdout_frac: float = 0.3,
    steps: int = 400,
    lr: float = 0.02,
    n_starts: int = 6,
    seed: int = 0,
) -> LambdaSelection:
    """
    Choose the penalty weight by cross-validation on held-out frequencies.

    A random subset of the finite ``(depth, mode, frequency)`` entries is
    withheld; each candidate is inverted using only the rest and scored on the
    entries it never saw. **No parameter truth is used anywhere**, which is the
    point --- a lambda tuned against known answers is not a lambda anyone could
    use on a real log, and reporting accuracy at such a lambda overstates the
    method.

    Parameters
    ----------
    trained : TrainedForwardSurrogate
    observed : ndarray, shape (D, M, F)
        Depth-ordered observed slowness (s/m).
    candidates : sequence of float
        Penalty weights to score. Include ``0.0`` so "no coupling at all"
        can win.
    penalty : {"l2", "tv"}, default "l2"
    tv_eps : float, default 0.05
        Roughness cost to select for, passed to :func:`invert_joint`. Selection
        is per-penalty: a weight chosen for one is not valid for the other,
        because the two price a bed contact differently.
    holdout_frac : float, default 0.3
        Fraction of finite entries to withhold. Must be in ``(0, 1)``.
    steps, lr, n_starts : int / float
        Passed through to :func:`invert_joint`.
    seed : int, default 0

    Returns
    -------
    LambdaSelection

    Raises
    ------
    ValueError
        If ``candidates`` is empty, ``holdout_frac`` is out of range, or the
        split leaves no entries on either side.
    """
    observed = np.asarray(observed, dtype=float)
    grid = tuple(float(c) for c in candidates)
    if not grid:
        raise ValueError("need at least one candidate lambda")
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError(f"holdout_frac must be in (0, 1), got {holdout_frac}")

    rng = np.random.default_rng(seed)
    finite = np.isfinite(observed)
    held = finite & (rng.random(observed.shape) < holdout_frac)
    fit = finite & ~held
    if not held.any() or not fit.any():
        raise ValueError("hold-out split left one side empty; adjust holdout_frac")

    # One independent warm start, shared by every candidate: the candidates
    # must differ only in lambda.
    seed_everything(seed)
    warm = invert_with_surrogate(
        trained,
        np.where(fit, observed, np.nan),
        n_starts=n_starts,
        steps=400,
        seed=seed,
    ).params

    model = trained.model.eval()
    target, held_mask, _ = _misfit_terms(trained, observed, held)
    scores = np.empty(len(grid), dtype=float)
    for i, lam in enumerate(grid):
        result = invert_joint(
            trained,
            observed,
            lam=lam,
            penalty=penalty,
            tv_eps=tv_eps,
            steps=steps,
            lr=lr,
            warm_start=warm,
            fit_mask=fit,
            seed=seed,
        )
        z = (result.params - trained.param_std.mean[trained.param_std.active]) / (
            trained.param_std.std[trained.param_std.active]
        )
        with torch.no_grad():
            predicted, _ = model(torch.as_tensor(z, dtype=torch.float32))
            squared = ((predicted - target) ** 2) * held_mask
            scores[i] = float(squared.sum().item() / held_mask.sum().item())

    return LambdaSelection(
        lam=grid[int(np.argmin(scores))],
        candidates=grid,
        holdout_misfit=scores,
        n_holdout=int(held.sum()),
    )


def smooth_independent(result: InversionResult, width: int) -> InversionResult:
    """
    The control that decides whether joint inversion was worth building.

    Runs a centred moving average of ``width`` frames over an already-recovered
    parameter log. This is the boring alternative to a coupled solve: invert
    each frame on its own, then smooth the answer. It uses the same prior ---
    "the log does not change much from frame to frame" --- and costs one pass of
    arithmetic instead of a second optimisation.

    A depth-coupled inversion is only worth its complexity if it beats this.
    Coupling *inside* the objective can in principle do something smoothing
    afterwards cannot, because it lets neighbouring frames trade off against the
    data misfit rather than being averaged regardless of how well each frame was
    determined. Whether that theoretical advantage survives contact with a real
    surrogate is a measurement, not an argument --- which is why this function
    ships next to :func:`invert_joint` rather than being left as an exercise.

    Parameters
    ----------
    result : InversionResult
        A depth-ordered result, typically from
        :func:`~sonic_ml.models.inversion.invert_with_surrogate`.
    width : int
        Window length in frames; must be >= 1. ``1`` returns the input
        unchanged. Windows are truncated at the ends rather than padded, so no
        value outside the logged interval is invented.

    Returns
    -------
    InversionResult
        With ``method="smoothed"``. ``data_misfit`` is copied from the input
        and is **no longer the misfit of the returned parameters** --- smoothing
        happens outside the forward model, so re-evaluating it would require
        the surrogate.

    Raises
    ------
    ValueError
        If ``width`` is less than 1.
    """
    if width < 1:
        raise ValueError(f"width must be at least 1, got {width}")
    params = np.asarray(result.params, dtype=float)
    n_depths = params.shape[0]
    half = width // 2
    out = np.empty_like(params)
    for d in range(n_depths):
        lo = max(0, d - half)
        hi = min(n_depths, d + half + 1)
        out[d] = params[lo:hi].mean(axis=0)
    return InversionResult(
        params=out,
        active_names=result.active_names,
        data_misfit=result.data_misfit.copy(),
        n_starts=result.n_starts,
        method="smoothed",
    )


def contact_precision(
    result: InversionResult,
    profile: DepthProfile,
    *,
    tolerance: int = 1,
) -> float:
    """
    Can you find the bed contacts in the recovered log?

    Mean absolute error says how close the values are; it says nothing about
    whether the *boundaries* survived, which for anyone picking beds is the
    question. This takes the ``B`` largest frame-to-frame changes in the
    recovered profile, where ``B`` is the true number of contacts, and reports
    what fraction of them land within ``tolerance`` frames of a real one.

    Score it against :func:`no_skill_contact_precision`, never on its own: on a
    log with many contacts, jumps placed at random already score well, and a
    bare precision figure hides that.

    Parameters
    ----------
    result : InversionResult
        Recovered profile, in the same depth order as ``profile``.
    profile : DepthProfile
        Supplies the true contacts.
    tolerance : int, default 1
        How many frames off a pick may be and still count. ``0`` demands the
        exact transition.

    Returns
    -------
    float
        In ``[0, 1]``; ``nan`` for a profile with no contacts, which has nothing
        to find.

    Raises
    ------
    ValueError
        If ``tolerance`` is negative.
    """
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    truth = np.asarray(profile.bed_boundaries, dtype=int)
    if truth.size == 0:
        return float("nan")

    scale = np.abs(result.params).mean(axis=0)
    scale[scale == 0.0] = 1.0
    jumps = np.abs(np.diff(result.params, axis=0) / scale).sum(axis=1)
    # transition i sits between frames i and i+1, i.e. it *is* boundary i+1
    picked = np.argsort(jumps)[::-1][: truth.size] + 1
    hits = sum(bool(np.any(np.abs(truth - p) <= tolerance)) for p in picked)
    return hits / truth.size


def no_skill_contact_precision(
    profile: DepthProfile,
    *,
    tolerance: int = 1,
) -> float:
    """
    Precision a contact-picker gets by guessing --- the bar for the number above.

    Computed exactly rather than simulated: it is the fraction of the available
    transitions that lie within ``tolerance`` of a true contact, which is the
    probability that a single blind pick lands on one.

    Parameters
    ----------
    profile : DepthProfile
    tolerance : int, default 1

    Returns
    -------
    float
        In ``[0, 1]``; ``nan`` for a profile with no contacts.
    """
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    truth = np.asarray(profile.bed_boundaries, dtype=int)
    if truth.size == 0:
        return float("nan")
    transitions = np.arange(1, profile.n_depths)
    near = [bool(np.any(np.abs(truth - t) <= tolerance)) for t in transitions]
    return float(np.mean(near))


def profile_mae(
    result: InversionResult,
    profile: DepthProfile,
    name: str,
) -> float:
    """
    Mean absolute error of one recovered parameter over the whole profile.

    Parameters
    ----------
    result : InversionResult
        From :func:`invert_joint` or
        :func:`~sonic_ml.models.inversion.invert_with_surrogate`, in profile
        depth order.
    profile : DepthProfile
        Supplies the truth.
    name : str

    Returns
    -------
    float
    """
    return float(np.mean(np.abs(result.value(name) - profile.value(name))))


def bed_vs_boundary_mae(
    result: InversionResult,
    profile: DepthProfile,
    name: str,
    *,
    halo: int = 1,
) -> tuple[float, float]:
    """
    Split the error into "inside a bed" and "at a bed contact".

    This is the measurement that makes a depth-coupled result interpretable.
    Averaged over a profile, a smoothing prior can look like a clean win while
    quietly destroying every contact in the log --- which, for anyone picking
    bed boundaries, is the part that mattered. Reporting the two numbers
    separately makes that trade visible instead of averaging it away.

    Parameters
    ----------
    result : InversionResult
    profile : DepthProfile
    name : str
    halo : int, default 1
        Frames each side of a contact counted as "at the boundary".

    Returns
    -------
    in_bed_mae, boundary_mae : float
        In the parameter's own units. ``boundary_mae`` is ``nan`` for a
        single-bed profile, which has no contacts to damage.
    """
    error = np.abs(result.value(name) - profile.value(name))
    at_boundary = profile.boundary_mask(halo)
    in_bed = float(np.mean(error[~at_boundary])) if (~at_boundary).any() else np.nan
    boundary = float(np.mean(error[at_boundary])) if at_boundary.any() else np.nan
    return in_bed, boundary
