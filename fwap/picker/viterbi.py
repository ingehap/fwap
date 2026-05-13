"""
Joint log-likelihood (Viterbi) pickers across a depth sweep.

Two variants:

* :func:`viterbi_pick` -- per-mode Viterbi (each mode picked
  independently across depths with a Gaussian slowness-jump prior
  and a soft time-order penalty between modes).
* :func:`viterbi_pick_joint` -- N-mode joint Viterbi where the
  state at each depth is the full :math:`(P, S, \\ldots)` triple
  / tuple. Strictly more expressive than the per-mode form but
  pays for it with a wider trellis; the candidate budget is
  managed automatically (see :func:`_auto_fallback_k`).

Both are replacements for the greedy :func:`fwap.picker.greedy.track_modes`
that handle the S-before-P-in-altered-zones case and miss-pick
propagation more robustly. The trellis-building primitives
(:func:`_build_triple_trellis`, :func:`_joint_transition_matrix`)
are shared with :mod:`fwap.picker.posterior` to keep the joint
Viterbi and the posterior-marginal forward-backward implementations
bit-equivalent on the trellis layer.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fwap._common import US_PER_FT, logger
from fwap.coherence import STCResult, find_peaks
from fwap.picker._types import DEFAULT_PRIORS, DepthPicks, ModePick

_NEG_INF = float("-inf")


def viterbi_pick(
    stc_results: Sequence[STCResult],
    depths: np.ndarray,
    priors: dict[str, dict[str, float]] | None = None,
    threshold: float = 0.4,
    slow_jump_sigma: float = 20.0 * US_PER_FT,
    time_order_slack: float = 0.0,
    time_prior_weight: float = 500.0,
    absence_cost: float = 3.0,
) -> list[DepthPicks]:
    r"""
    Joint-log-likelihood (Viterbi) mode picker across a depth sweep.

    Replacement for the greedy :func:`track_modes` that addresses the
    two known failure modes of the per-depth rule-based pipeline:

    * S-head-wave arriving before the formation P in altered zones,
      which the strict time-ordering rule of :func:`pick_modes`
      refuses to pick at all (see the note at
      :func:`pick_modes`).
    * Missed picks inside noisy depth intervals propagating into
      the continuity tolerance of :func:`track_modes` and eventually
      letting a noise peak be reacquired far from the true mode.

    For each mode, Viterbi over depths maximises

    .. math::

       \sum_d \log \rho_d(s, t) \;-\;
       \sum_d \frac{(s_d - s_{d-1})^2}{2 \sigma^2}

    where :math:`\rho_d` is the STC coherence at the candidate cell
    and :math:`\sigma` is ``slow_jump_sigma``. A "mode absent at this
    depth" state is available at cost ``absence_cost`` so long gaps
    do not force spurious picks.

    Modes are processed in their ``priors[*]["order"]`` order (default
    P, S, Stoneley). Time ordering between modes is enforced on each
    depth: candidates for mode *k+1* must arrive no earlier than the
    Viterbi-picked time of mode *k* at the same depth, minus
    ``time_order_slack`` (positive slack permits the S-before-P case
    flagged above).

    Parameters
    ----------
    stc_results : sequence of STCResult
        One STC surface per depth.
    depths : ndarray, shape (n_depth,)
        Tool depth (m) for each STC surface.
    priors : dict, optional
        Per-mode prior windows. Defaults to :data:`DEFAULT_PRIORS`.
    threshold : float, default 0.4
        Coherence floor passed through to
        :func:`fwap.coherence.find_peaks`.
    slow_jump_sigma : float, default 20 us/ft
        Gaussian scale (s/m) of the per-mode depth-to-depth slowness
        jump penalty.
    time_order_slack : float, default 0.0
        Allowed amount (s) that mode *k+1* may arrive before mode *k*
        at the same depth. Zero enforces strict ordering; a small
        positive value permits the S-before-P-in-altered-zones case.
    time_prior_weight : float, default 500.0
        Per-mode preference for earlier arrivals within each depth's
        candidate pool, in log-probability units per second. With the
        default a 1 ms later arrival is penalised by ~0.5
        log-probability units, enough to break ties between nearby-
        coherence peaks (e.g. a P-window candidate at the P slowness
        vs one at the S slowness) but not so much that a strong late
        peak is overridden by a weak early one.
    absence_cost : float, default 3.0
        Cost (in log-probability units, i.e. ~``-log(coherence)``) of
        declaring a mode absent at a given depth. 3.0 is roughly
        equivalent to a minimum-coherence threshold of
        :math:`e^{-3} \approx 0.05`.

    Returns
    -------
    list of DepthPicks
        One :class:`DepthPicks` per depth. A mode absent at a given
        depth simply does not appear in the ``picks`` dict.

    Notes
    -----
    The current implementation runs a separate Viterbi pass per mode,
    with the previous mode's time-path fed into the emission score of
    the next mode (sequential-within-depth, joint-across-depth). A
    fully joint picker over all three modes would be possible but
    would blow up the state space cubically for no meaningful
    accuracy gain on typical sonic data.

    References
    ----------
    * Viterbi, A. (1967). Error bounds for convolutional codes and an
      asymptotically optimum decoding algorithm.
      *IEEE Transactions on Information Theory* 13(2), 260-269.
    * Aron, J., et al. (1994). Real-time sonic logging while drilling
      (flagged S-before-P altered-zone case).
    """
    if priors is None:
        priors = DEFAULT_PRIORS

    depths = np.asarray(depths, dtype=float)
    n_depth = depths.size
    if n_depth == 0:
        return []
    if len(stc_results) != n_depth:
        raise ValueError("stc_results and depths must have the same length")

    # Pre-compute candidate peaks per depth once.
    cand_lists = [find_peaks(r, threshold=threshold) for r in stc_results]

    # Accumulate results as we pick one mode at a time. previous_time[d]
    # is the Viterbi-picked time of the previously processed mode at
    # depth d, or -inf if it was absent (so no constraint on later
    # modes at that depth).
    previous_time: np.ndarray = np.full(n_depth, -np.inf, dtype=float)
    all_picks: list[DepthPicks] = [
        DepthPicks(depth=float(depths[d])) for d in range(n_depth)
    ]

    for name in sorted(priors, key=lambda n: priors[n]["order"]):
        prior = priors[name]

        # Per-depth candidate arrays (post window + time-order filter)
        per_depth: list[np.ndarray] = []
        for d in range(n_depth):
            peaks = cand_lists[d]
            if peaks.size == 0:
                per_depth.append(peaks)
                continue
            t_floor = previous_time[d] - time_order_slack
            mask = (
                (peaks[:, 0] >= prior["slow_min"])
                & (peaks[:, 0] <= prior["slow_max"])
                & (peaks[:, 2] >= prior["coherence_min"])
                & (peaks[:, 1] >= t_floor)
            )
            per_depth.append(peaks[mask])

        # Viterbi forward pass. State at each depth = index into that
        # depth's per_depth[d] array, with an extra "absent" state
        # represented by -1. We store trellis arrays as lists because
        # candidate counts vary per depth.
        scores: list[np.ndarray] = []  # (n_candidates + 1,) per depth
        back_ptrs: list[np.ndarray] = []  # int: index into prev depth

        for d in range(n_depth):
            cands = per_depth[d]
            n_cand = cands.shape[0]
            # State order: [c0, c1, ..., c_{n-1}, absent]
            emission = np.full(n_cand + 1, _NEG_INF, dtype=float)
            if n_cand > 0:
                # Emission = log(coherence) plus a time-earliness
                # bonus, referenced to the earliest candidate at this
                # depth so that the bonus is 0 for the earliest peak
                # and negative for later ones. Breaks ties between
                # candidates with similar coherence inside overlapping
                # prior windows (notably the common case where the S
                # arrival falls inside P's [40, 140] us/ft window).
                t_ref = float(cands[:, 1].min())
                emission[:n_cand] = np.log(
                    np.clip(cands[:, 2], 1.0e-12, None)
                ) - time_prior_weight * (cands[:, 1] - t_ref)
            emission[n_cand] = -absence_cost

            if d == 0:
                scores.append(emission.copy())
                back_ptrs.append(np.full(n_cand + 1, -1, dtype=np.intp))
                continue

            prev_scores = scores[d - 1]
            prev_cands = per_depth[d - 1]
            n_prev = prev_cands.shape[0]

            # Transition: gaussian penalty on slowness jump between
            # candidates; absent<->absent is free; absent<->candidate
            # only adds emission (i.e. transition cost 0).
            trans = np.full((n_prev + 1, n_cand + 1), _NEG_INF, dtype=float)
            if n_prev > 0 and n_cand > 0:
                ds = cands[:, 0][None, :] - prev_cands[:, 0][:, None]
                trans[:n_prev, :n_cand] = -0.5 * (ds / slow_jump_sigma) ** 2
            if n_prev > 0:
                trans[:n_prev, n_cand] = 0.0
            if n_cand > 0:
                trans[n_prev, :n_cand] = 0.0
            trans[n_prev, n_cand] = 0.0

            step = prev_scores[:, None] + trans + emission[None, :]
            best_prev = np.argmax(step, axis=0)
            best_score = step[best_prev, np.arange(n_cand + 1)]
            # Cells where every prev state is -inf stay -inf.
            mask_valid = np.isfinite(best_score)
            best_score = np.where(mask_valid, best_score, _NEG_INF)
            scores.append(best_score)
            back_ptrs.append(best_prev.astype(np.intp))

        # Backtrack.
        path: np.ndarray = np.empty(n_depth, dtype=np.intp)
        path[-1] = int(np.argmax(scores[-1]))
        for d in range(n_depth - 1, 0, -1):
            path[d - 1] = back_ptrs[d][path[d]]

        # Populate picks for this mode and update previous_time.
        new_previous_time = previous_time.copy()
        for d in range(n_depth):
            cands = per_depth[d]
            n_cand = cands.shape[0]
            state = int(path[d])
            if state == n_cand:
                continue  # mode absent at this depth
            row = cands[state]
            pick = ModePick(
                name=name,
                slowness=float(row[0]),
                time=float(row[1]),
                coherence=float(row[2]),
                amplitude=float(row[3]) if row.size >= 4 else None,
            )
            all_picks[d].picks[name] = pick
            new_previous_time[d] = max(new_previous_time[d], pick.time)
        previous_time = new_previous_time

    return all_picks


# ---------------------------------------------------------------------
# Joint N-mode Viterbi (state = (P, S, Stoneley, ...) tuple)
# ---------------------------------------------------------------------


@dataclass
class _TripleTrellis:
    """Per-depth triple enumeration output of :func:`_build_triple_trellis`.

    Attributes
    ----------
    mode_names : list[str]
        Mode names in processing order.
    per_mode_per_depth : dict[str, list[ndarray]]
        Filtered candidate arrays (post prior-window, post top-K).
    triples : list[ndarray (n_triples_d, n_modes) int]
        Per-depth candidate-index triples; -1 denotes absent.
    emissions : list[ndarray (n_triples_d,) float]
        Log-probability emission for each triple (includes any
        soft-time-order penalty already applied).
    slows : list[ndarray (n_triples_d, n_modes) float]
        Slowness of each mode in each triple; NaN for absent modes.
    """

    mode_names: list[str]
    per_mode_per_depth: dict[str, list[np.ndarray]]
    triples: list[np.ndarray]
    emissions: list[np.ndarray]
    slows: list[np.ndarray]


def _auto_fallback_k(n_per_mode: list[int], budget: int) -> int:
    """Find largest K such that prod(min(n_i, K) + 1) <= budget.

    Used by ``_build_triple_trellis`` to tighten per-mode top-K when
    the raw triple count would otherwise exceed
    ``max_triples_per_depth``. Returns the largest non-negative
    integer K fitting the budget; iterates from max(n_per_mode)
    downward, which is O(max_n * n_modes) -- trivial for typical
    sonic gathers.
    """
    if not n_per_mode:
        return 0
    max_n = max(n_per_mode)
    for K in range(max_n, -1, -1):
        prod = 1
        for n in n_per_mode:
            prod *= min(n, K) + 1
            if prod > budget:
                break
        if prod <= budget:
            return K
    return 0


def _build_triple_trellis(
    stc_results: Sequence[STCResult],
    n_depth: int,
    priors: dict[str, dict[str, float]],
    threshold: float,
    time_order_slack: float,
    soft_time_order: float | None,
    time_prior_weight: float,
    absence_cost: float,
    top_k_per_mode: int | None,
    max_triples_per_depth: int,
) -> _TripleTrellis:
    """Shared trellis builder for the two joint-Viterbi inference paths."""
    mode_names = sorted(priors, key=lambda n: priors[n]["order"])
    n_modes = len(mode_names)

    # Step 1: per-mode, per-depth candidate arrays (prior-window +
    # coherence filter, then optional top-K per mode).
    per_mode_per_depth: dict[str, list[np.ndarray]] = {name: [] for name in mode_names}
    for d in range(n_depth):
        peaks = find_peaks(stc_results[d], threshold=threshold)
        for name in mode_names:
            prior = priors[name]
            if peaks.size == 0:
                per_mode_per_depth[name].append(peaks)
                continue
            mask = (
                (peaks[:, 0] >= prior["slow_min"])
                & (peaks[:, 0] <= prior["slow_max"])
                & (peaks[:, 2] >= prior["coherence_min"])
            )
            cands = peaks[mask]
            if top_k_per_mode is not None and cands.shape[0] > top_k_per_mode:
                # Keep the top-K by coherence (descending).
                order = np.argsort(-cands[:, 2])
                cands = cands[order[:top_k_per_mode]]
            per_mode_per_depth[name].append(cands)

    # Step 2: enumerate triples per depth.
    triples: list[np.ndarray] = []
    emissions: list[np.ndarray] = []
    slows: list[np.ndarray] = []

    for d in range(n_depth):
        per_mode_cands = [per_mode_per_depth[name][d] for name in mode_names]
        n_per_mode = [c.shape[0] for c in per_mode_cands]

        # Variable candidate budget: if the raw triple count
        # ``prod(n_i + 1)`` would exceed ``max_triples_per_depth``,
        # tighten the per-mode top-K to fit -- preferring high-
        # coherence candidates within each mode. This replaces the
        # earlier "raise on overflow" behaviour with graceful
        # degradation; pathological peak-heavy STC surfaces no
        # longer kill the whole sweep.
        raw_count = 1
        for n in n_per_mode:
            raw_count *= n + 1
        if raw_count > max_triples_per_depth:
            auto_K = _auto_fallback_k(n_per_mode, max_triples_per_depth)
            for i, name in enumerate(mode_names):
                cands = per_mode_cands[i]
                if cands.shape[0] > auto_K:
                    order = np.argsort(-cands[:, 2])
                    cands = cands[order[:auto_K]]
                    per_mode_cands[i] = cands
                    per_mode_per_depth[name][d] = cands
            new_n_per_mode = [c.shape[0] for c in per_mode_cands]
            logger.debug(
                "trellis: depth %d auto-fallback K=%d (raw=%d, n_per_mode %s -> %s)",
                d,
                auto_K,
                raw_count,
                n_per_mode,
                new_n_per_mode,
            )
            n_per_mode = new_n_per_mode

        t_min_at_d = np.inf
        for cand in per_mode_cands:
            if cand.size > 0:
                t_min_at_d = min(t_min_at_d, float(cand[:, 1].min()))
        if not np.isfinite(t_min_at_d):
            t_min_at_d = 0.0

        rows_triples: list[tuple[int, ...]] = []
        rows_emission: list[float] = []
        rows_slow: list[list[float]] = []

        per_mode_ranges = [range(-1, n_ci) for n_ci in n_per_mode]
        for combo in itertools.product(*per_mode_ranges):
            # Within-depth ordering. Hard if soft_time_order is None,
            # soft (penalised) otherwise.
            last_t = -np.inf
            ordering_violation = 0.0
            ordering_ok = True
            for i, ci in enumerate(combo):
                if ci < 0:
                    continue
                t = float(per_mode_cands[i][ci, 1])
                gap = last_t - time_order_slack - t
                if gap > 1.0e-12:
                    if soft_time_order is None:
                        ordering_ok = False
                        break
                    ordering_violation += gap
                last_t = max(last_t, t)
            if not ordering_ok:
                continue

            em = 0.0
            slow_row: list[float] = []
            for i, ci in enumerate(combo):
                if ci < 0:
                    em -= absence_cost
                    slow_row.append(float("nan"))
                    continue
                cand = per_mode_cands[i][ci]
                em += float(np.log(max(cand[2], 1.0e-12)))
                em -= time_prior_weight * (float(cand[1]) - t_min_at_d)
                slow_row.append(float(cand[0]))
            if soft_time_order is not None and ordering_violation > 0.0:
                em -= soft_time_order * ordering_violation
            rows_triples.append(tuple(combo))
            rows_emission.append(em)
            rows_slow.append(slow_row)

        # Final safety net: the auto-fallback above guarantees
        # ``prod(n_i + 1) <= max_triples_per_depth``, so the time-
        # ordering filter can only reduce the count further. If we
        # somehow still exceed the budget, that's a bug in the
        # auto-fallback math; raise rather than silently passing.
        if len(rows_triples) > max_triples_per_depth:
            raise RuntimeError(
                f"depth {d} produced {len(rows_triples)} candidate "
                f"triples post-auto-fallback, exceeding "
                f"max_triples_per_depth={max_triples_per_depth}. "
                f"This is an internal bug in _auto_fallback_k; "
                f"please report it."
            )

        triples.append(np.asarray(rows_triples, dtype=np.intp).reshape(-1, n_modes))
        emissions.append(np.asarray(rows_emission, dtype=float))
        slows.append(np.asarray(rows_slow, dtype=float).reshape(-1, n_modes))

    return _TripleTrellis(
        mode_names=mode_names,
        per_mode_per_depth=per_mode_per_depth,
        triples=triples,
        emissions=emissions,
        slows=slows,
    )


def _joint_transition_matrix(
    prev_slow: np.ndarray, curr_slow: np.ndarray, slow_jump_sigma: float
) -> np.ndarray:
    """Per-mode Gaussian slowness-jump penalty summed across modes.

    Returns a ``(n_prev, n_curr)`` matrix of the total transition
    cost (a non-negative number to subtract from the score during the
    max/sum pass). Pairs where either endpoint has a mode absent
    (NaN slowness) contribute 0 for that mode -- the absence cost is
    already paid through the emission term.
    """
    n_prev = prev_slow.shape[0]
    n_curr = curr_slow.shape[0]
    n_modes = prev_slow.shape[1]
    total = np.zeros((n_prev, n_curr), dtype=float)
    for m in range(n_modes):
        jump = curr_slow[None, :, m] - prev_slow[:, None, m]
        with np.errstate(invalid="ignore"):
            cost_m = 0.5 * (jump / slow_jump_sigma) ** 2
        total += np.where(np.isnan(cost_m), 0.0, cost_m)
    return total


def viterbi_pick_joint(
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
) -> list[DepthPicks]:
    r"""
    Fully-joint N-mode Viterbi picker.

    State at each depth is an N-tuple of per-mode candidate indices
    (with an "absent" option per mode), subject to the within-depth
    time-ordering constraint along the prior ``order`` field (strict
    by default, soft if ``soft_time_order`` is set). Viterbi DP
    runs over ``(depth, tuple)``; the result is the globally optimal
    per-mode path across the full sweep.

    Defaults to the full :data:`DEFAULT_PRIORS` (4 modes: P, S,
    PseudoRayleigh, Stoneley). The trellis builder is N-mode
    generic; the auto-fallback variable-candidate-budget machinery
    keeps the wider 4-mode trellis tractable on noisy gathers
    (substep "variable candidate budget" in roadmap item C).
    Pass an explicit ``priors`` subset to restrict to fewer modes
    (e.g. just ``("P", "S", "Stoneley")`` to skip pseudo-Rayleigh).

    Differences vs :func:`viterbi_pick`
    -----------------------------------
    :func:`viterbi_pick` runs Viterbi on each mode independently and
    feeds the best path's picked time as a soft constraint into the
    next mode. Joint Viterbi optimises over the triple as a single
    unit, so coupling effects -- e.g. a depth where the best-by-
    coherence P pick would force an impossible S in the next depth,
    or where S-before-P in an altered zone is the jointly optimal
    answer -- are handled exactly instead of through a sequential
    relaxation. On clean data the two produce identical picks; the
    difference appears on noisy or altered-zone intervals.

    Cost
    ----
    Per-depth tuple enumeration is ``prod(n_i + 1)`` before the
    time-ordering filter. Transition cost between depth steps is
    ``n_prev * n_curr`` and is bounded by ``max_triples_per_depth``.
    On a realistic 30-depth, 4-mode sweep with ~15 peaks per mode,
    total runtime is well under one second.

    Complexity
    ----------
    Time is ``O(n_depth * T^2)`` where ``T`` is the per-depth tuple
    count; memory is ``O(n_depth * T)``. ``T`` grows as the *product*
    of per-mode candidate counts before the time-ordering filter,
    so very peaky STC surfaces can blow up the trellis quickly. The
    variable-candidate-budget machinery handles this gracefully:
    when ``prod(n_i + 1) > max_triples_per_depth`` for any depth,
    the per-mode top-K is automatically tightened (preferring high-
    coherence candidates within each mode) so the budget is met.
    Set ``top_k_per_mode`` (typical: 5-10) explicitly to bound
    runtime more aggressively, or raise the coherence ``threshold``
    to thin the candidate pool.

    Parameters
    ----------
    stc_results : sequence of STCResult
    depths : ndarray, shape (n_depth,)
    priors : dict, optional
        Per-mode prior windows. Defaults to :data:`DEFAULT_PRIORS`.
    threshold, slow_jump_sigma, time_order_slack, time_prior_weight,
    absence_cost
        See :func:`viterbi_pick`; same semantics.
    top_k_per_mode : int, optional
        If set, keep only the K most-coherent candidates per mode
        per depth before triple enumeration. Bounds the trellis
        size (and runtime) in the presence of very peaky STC
        surfaces without hitting ``max_triples_per_depth``.
        ``None`` (default) keeps every candidate that passed the
        prior window + coherence-threshold filter.
    soft_time_order : float, optional
        If set to a positive value ``lambda``, the strict
        within-depth ordering constraint (along each prior's
        ``order`` field) is replaced with a soft penalty
        ``lambda * violation_magnitude`` added to the emission.
        Useful in altered zones where S legitimately arrives before
        P and the strict constraint would kill the entire tuple.
        ``None`` (default) keeps the hard constraint.
    max_triples_per_depth : int, default 2000
        Per-depth tuple-count budget. When the raw count
        ``prod(n_i + 1)`` would exceed the budget, the per-mode
        top-K is automatically tightened to fit (preferring high-
        coherence candidates within each mode). The default 2000 is
        comfortable for 3-mode picking and triggers mild auto-
        fallback for 4-mode picking on peaky surfaces; bump to
        ~5000 for 4-mode picking that needs to retain ~5+
        candidates per mode without auto-fallback.

    Returns
    -------
    list of DepthPicks

    References
    ----------
    * Viterbi, A. (1967). Error bounds for convolutional codes and
      an asymptotically optimum decoding algorithm. *IEEE
      Transactions on Information Theory* 13(2), 260-269.
    """
    if priors is None:
        # Default to the full DEFAULT_PRIORS (4 modes including
        # PseudoRayleigh). Joint Viterbi is now N-mode generic; the
        # auto-fallback variable-candidate-budget machinery in
        # ``_build_triple_trellis`` handles the larger trellis width
        # gracefully. Use ``track_modes`` or ``viterbi_pick`` if
        # per-mode independence is preferable to joint optimisation.
        priors = dict(DEFAULT_PRIORS)
    if not priors:
        raise ValueError("priors must contain at least one mode; got an empty dict.")
    depths = np.asarray(depths, dtype=float)
    n_depth = depths.size
    if n_depth == 0:
        return []
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

    # Viterbi forward pass (max-sum).
    scores: list[np.ndarray] = [emissions[0].copy()]
    back_ptrs: list[np.ndarray] = [np.full(scores[0].size, -1, dtype=np.intp)]

    for d in range(1, n_depth):
        total_trans = _joint_transition_matrix(slows[d - 1], slows[d], slow_jump_sigma)
        step = scores[d - 1][:, None] - total_trans + emissions[d][None, :]
        best_prev = np.argmax(step, axis=0)
        best_score = step[best_prev, np.arange(step.shape[1])]
        scores.append(best_score)
        back_ptrs.append(best_prev.astype(np.intp))

    # Backtrack.
    path: np.ndarray = np.empty(n_depth, dtype=np.intp)
    path[-1] = int(np.argmax(scores[-1]))
    for d in range(n_depth - 1, 0, -1):
        path[d - 1] = back_ptrs[d][path[d]]

    # Build the DepthPicks output.
    all_picks: list[DepthPicks] = []
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
        all_picks.append(dp)
    return all_picks
