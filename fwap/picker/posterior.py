"""
Posterior marginals over the joint Viterbi trellis via forward-backward.

Returns, in addition to the MAP path, the per-mode per-depth
posterior probability distribution over candidate slownesses --
useful for confidence-aware downstream processing where a single
best-estimate would discard relevant uncertainty.

Shares the trellis-building primitives
(:func:`fwap.picker.viterbi._build_triple_trellis` and
:func:`fwap.picker.viterbi._joint_transition_matrix`) with the
joint Viterbi picker so the two are guaranteed to agree at the
trellis-emission layer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fwap._common import US_PER_FT
from fwap.coherence import STCResult
from fwap.picker._types import DEFAULT_PRIORS, DepthPicks, ModePick
from fwap.picker.viterbi import (
    _build_triple_trellis,
    _joint_transition_matrix,
)


@dataclass
class PosteriorPick:
    """
    Per-mode, per-depth posterior marginal from
    :func:`viterbi_posterior_marginals`.

    Unlike :class:`ModePick`, which carries a single best-estimate
    slowness and its coherence, a ``PosteriorPick`` describes the
    full posterior probability distribution over candidate picks
    (including the probability that the mode is absent at this
    depth).

    Attributes
    ----------
    slownesses : ndarray, shape (n_candidates,)
        Slownesses of the in-window candidates for this mode at
        this depth.
    times : ndarray, shape (n_candidates,)
        Arrival times of the candidates.
    coherences : ndarray, shape (n_candidates,)
        Coherence values of the candidates (the per-cell emission
        before time / absence bonuses).
    probabilities : ndarray, shape (n_candidates,)
        Posterior probability that this mode is picked at the
        corresponding candidate, summed over all triples containing
        it. ``probabilities.sum() + p_absent == 1.0``.
    p_absent : float
        Posterior probability that the mode is absent at this depth.
    """

    slownesses: np.ndarray
    times: np.ndarray
    coherences: np.ndarray
    probabilities: np.ndarray
    p_absent: float


def _logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Numerically-stable log-sum-exp along ``axis``."""
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return np.array(-np.inf)
    m = np.max(a, axis=axis, keepdims=True)
    m_safe = np.where(np.isneginf(m), 0.0, m)
    lse = np.log(np.sum(np.exp(a - m_safe), axis=axis, keepdims=True)) + m_safe
    if axis is None:
        return np.asarray(lse).squeeze()
    return np.squeeze(lse, axis=axis)


def viterbi_posterior_marginals(
    stc_results: Sequence[STCResult],
    depths: np.ndarray,
    priors: dict[str, dict[str, float]] | None = None,
    threshold: float = 0.4,
    slow_jump_sigma: float = 20.0 * US_PER_FT,
    time_order_slack: float = 0.0,
    time_prior_weight: float = 500.0,
    absence_cost: float = 3.0,
    top_k_per_mode: int | None = None,
    soft_time_order: float | None = None,
    max_triples_per_depth: int = 2000,
) -> tuple[list[DepthPicks], list[dict[str, PosteriorPick]]]:
    r"""
    Joint N-mode forward-backward: MAP picks plus per-mode posterior
    marginals.

    Runs exactly the same trellis as :func:`viterbi_pick_joint`, but
    in addition to the max-sum forward pass it also computes the
    log-sum-exp forward and backward messages. Marginalising the
    posterior over the (depth, tuple) lattice yields, at every
    depth, the probability that each mode is picked at each of its
    candidate slownesses -- plus the probability that the mode is
    absent.

    Defaults to the full :data:`DEFAULT_PRIORS` (4 modes); pass an
    explicit ``priors`` subset to restrict to fewer modes. Same
    auto-fallback variable-candidate-budget machinery as
    :func:`viterbi_pick_joint` keeps the trellis tractable.

    Useful for:

    - **Uncertainty quantification**: the MAP pick's slowness is
      accompanied by a distribution, not just a coherence value.
    - **Ambiguous picks**: if two candidates have similar posterior
      probability, the MAP answer is not the whole story; a caller
      can flag such depths for manual QC.
    - **Absence-probability mask**: ``1 - p_absent`` is a cleaner
      mask than the raw MAP coherence when used to weight
      downstream products.

    Parameters
    ----------
    stc_results, depths, priors, threshold, slow_jump_sigma,
    time_order_slack, time_prior_weight, absence_cost,
    top_k_per_mode, soft_time_order, max_triples_per_depth
        Identical to :func:`viterbi_pick_joint`.

    Returns
    -------
    map_picks : list of DepthPicks
        Same as :func:`viterbi_pick_joint` on this input (the MAP /
        Viterbi path through the trellis).
    posteriors : list of dict[str, PosteriorPick]
        One dict per depth, keyed by mode name. Each entry carries
        the per-candidate posterior probability vector and the
        probability that the mode is absent at that depth.

    References
    ----------
    * Rabiner, L. R. (1989). A tutorial on hidden Markov models and
      selected applications in speech recognition. *Proceedings of
      the IEEE* 77(2), 257-286 (Algorithm 2, forward-backward).
    """
    if priors is None:
        # Default to the full DEFAULT_PRIORS (4 modes including
        # PseudoRayleigh). N-mode generic; the variable-candidate-
        # budget auto-fallback in ``_build_triple_trellis`` keeps
        # the wider trellis tractable.
        priors = dict(DEFAULT_PRIORS)
    if not priors:
        raise ValueError("priors must contain at least one mode; got an empty dict.")
    depths = np.asarray(depths, dtype=float)
    n_depth = depths.size
    if n_depth == 0:
        return [], []
    if len(stc_results) != n_depth:
        raise ValueError("stc_results and depths must have the same length")

    trellis = _build_triple_trellis(
        stc_results=stc_results,
        n_depth=n_depth,
        priors=priors,
        threshold=threshold,
        time_order_slack=time_order_slack,
        soft_time_order=soft_time_order,
        time_prior_weight=time_prior_weight,
        absence_cost=absence_cost,
        top_k_per_mode=top_k_per_mode,
        max_triples_per_depth=max_triples_per_depth,
    )
    mode_names = trellis.mode_names
    triples = trellis.triples
    emissions = trellis.emissions
    slows = trellis.slows
    per_mode_per_depth = trellis.per_mode_per_depth

    # MAP (max-sum) forward pass + backtrack, identical to
    # ``viterbi_pick_joint``.
    scores: list[np.ndarray] = [emissions[0].copy()]
    back_ptrs: list[np.ndarray] = [np.full(scores[0].size, -1, dtype=np.intp)]
    for d in range(1, n_depth):
        total_trans = _joint_transition_matrix(slows[d - 1], slows[d], slow_jump_sigma)
        step = scores[d - 1][:, None] - total_trans + emissions[d][None, :]
        best_prev = np.argmax(step, axis=0)
        best_score = step[best_prev, np.arange(step.shape[1])]
        scores.append(best_score)
        back_ptrs.append(best_prev.astype(np.intp))

    path: np.ndarray = np.empty(n_depth, dtype=np.intp)
    path[-1] = int(np.argmax(scores[-1]))
    for d in range(n_depth - 1, 0, -1):
        path[d - 1] = back_ptrs[d][path[d]]

    map_picks: list[DepthPicks] = []
    for d in range(n_depth):
        dp = DepthPicks(depth=float(depths[d]))
        triple = triples[d][path[d]]
        per_mode_cands = [per_mode_per_depth[name][d] for name in mode_names]
        for i, name in enumerate(mode_names):
            ci = int(triple[i])
            if ci < 0:
                continue
            row = per_mode_cands[i][ci]
            dp.picks[name] = ModePick(
                name=name,
                slowness=float(row[0]),
                time=float(row[1]),
                coherence=float(row[2]),
                amplitude=float(row[3]) if row.size >= 4 else None,
            )
        map_picks.append(dp)

    # Log-sum-exp forward pass (alpha).
    alpha: list[np.ndarray] = [emissions[0].copy()]
    for d in range(1, n_depth):
        total_trans = _joint_transition_matrix(slows[d - 1], slows[d], slow_jump_sigma)
        combined = alpha[d - 1][:, None] - total_trans
        alpha.append(emissions[d] + _logsumexp(combined, axis=0))

    # Log-sum-exp backward pass (beta).
    beta: list[np.ndarray] = [np.zeros(triples[-1].shape[0], dtype=float)]
    for d in range(n_depth - 2, -1, -1):
        total_trans = _joint_transition_matrix(slows[d], slows[d + 1], slow_jump_sigma)
        combined = emissions[d + 1][None, :] + beta[0][None, :] - total_trans
        beta.insert(0, _logsumexp(combined, axis=1))

    # Posterior marginals over triples: gamma[d][j] normalised.
    posteriors: list[dict[str, PosteriorPick]] = []
    for d in range(n_depth):
        log_gamma = alpha[d] + beta[d]
        log_norm = _logsumexp(log_gamma)
        probs_triple = np.exp(log_gamma - log_norm)

        mode_post: dict[str, PosteriorPick] = {}
        tri_matrix = triples[d]  # (n_triples, n_modes)
        for i, name in enumerate(mode_names):
            cands = per_mode_per_depth[name][d]
            n_cand = cands.shape[0]
            if n_cand == 0:
                mode_post[name] = PosteriorPick(
                    slownesses=np.empty(0),
                    times=np.empty(0),
                    coherences=np.empty(0),
                    probabilities=np.empty(0),
                    p_absent=1.0,
                )
                continue
            mode_col = tri_matrix[:, i]
            absent_mask = mode_col == -1
            p_absent = float(probs_triple[absent_mask].sum())
            probs_cand = np.zeros(n_cand, dtype=float)
            for c in range(n_cand):
                probs_cand[c] = float(probs_triple[mode_col == c].sum())
            mode_post[name] = PosteriorPick(
                slownesses=cands[:, 0].copy(),
                times=cands[:, 1].copy(),
                coherences=cands[:, 2].copy(),
                probabilities=probs_cand,
                p_absent=p_absent,
            )
        posteriors.append(mode_post)

    return map_picks, posteriors
