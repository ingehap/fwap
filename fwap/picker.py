"""
Rule-based P / S / pseudo-Rayleigh / Stoneley picker on STC coherence
surfaces.

Implements the knowledge-based picker described in Part 1 of the book
(an "AI approach for the picking of waves on full-waveform acoustic
data"): expert rules on slowness range, coherence threshold, mode
ordering and depth-to-depth continuity are codified into a deterministic
selection over the candidate peaks returned by
:func:`fwap.coherence.find_peaks`. Two further book-listed expert
rules -- **wavelet shape** and **onset polarity** -- are exposed as
post-pick filters via :func:`filter_picks_by_shape` (single-depth)
and :func:`filter_track_by_shape` (multi-depth track) so callers can
opt in without altering the core slowness / coherence / continuity
selection. Default priors cover the four
monopole arrivals enumerated by Mari et al. -- the P head-wave, the S
head-wave, the pseudo-Rayleigh / guided trapped mode (when present in
fast formations), and the low-frequency Stoneley tube wave.

References
----------
* Mari, J.-L., Coppens, F., Gavin, P., & Wicquart, E. (1994).
  *Full Waveform Acoustic Data Processing*, Part 1. Editions Technip,
  Paris. ISBN 978-2-7108-0664-6.
* Aron, J., Chang, S. K., Codazzi, D., Dworak, R., Hsu, K., Lau, T.,
  Minerbo, G., & Yogeswaren, E. (1994). Real-time sonic logging while
  drilling in hard and soft rocks. *SEG Technical Program Expanded
  Abstracts*, 13, 1-4.
* Serra, O. (1984). *Fundamentals of Well-Log Interpretation: 1. The
  Acquisition of Logging Data.* Elsevier (for the slowness windows used
  in :data:`DEFAULT_PRIORS`).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

SelectionRule = Literal["max_coherence", "scored"]

# Shape-bearing aliases. NumPy's static type system cannot express
# array shapes at this level, so these are documentary type aliases
# only -- they communicate the contract to readers and to mypy
# without claiming runtime shape enforcement.
#
#   PeakArray:  shape (n_candidates, 3) holding rows of
#               [slowness, time, coherence] as returned by
#               :func:`fwap.coherence.find_peaks`.
#   PeakRow:    shape (3,)  -- one row of a PeakArray.
#   SlowAxis:   shape (n_slowness,)   -- the slowness axis of an STC.
#   TimeAxis:   shape (n_time,)       -- the time axis of an STC.
#   STCSurface: shape (n_slowness, n_time) -- the coherence surface.
PeakArray = np.ndarray
PeakRow = np.ndarray
SlowAxis = np.ndarray
TimeAxis = np.ndarray
STCSurface = np.ndarray

from fwap._common import US_PER_FT, _phase_shift, logger
from fwap.coherence import STCResult, find_peaks

# Per-mode prior windows used by :func:`pick_modes`. Slowness bounds
# are stored in **seconds per metre** (the unit used everywhere in the
# package); ``40.0 * US_PER_FT`` is simply the convenient way to write
# "40 microseconds per foot" at declaration time. ``coherence_min`` is
# unitless; ``order`` is the processing order
# (P -> S -> PseudoRayleigh -> Stoneley).
#
# PseudoRayleigh is a guided trapped mode that exists in fast
# formations between the formation shear slowness (its low-frequency
# cutoff) and the fluid slowness (its high-f asymptote). The default
# 130-200 us/ft window sits above the typical S slowness and below
# the Stoneley window so that the time-ordering rule cleanly
# separates the four modes; on a 3-mode-only gather no peak falls
# inside this window and PseudoRayleigh is reported absent.
DEFAULT_PRIORS: dict[str, dict[str, float]] = {
    "P": dict(slow_min=40.0 * US_PER_FT, slow_max=140.0 * US_PER_FT,
              coherence_min=0.5, order=0),
    "S": dict(slow_min=80.0 * US_PER_FT, slow_max=260.0 * US_PER_FT,
              coherence_min=0.4, order=1),
    "PseudoRayleigh": dict(slow_min=130.0 * US_PER_FT,
                           slow_max=200.0 * US_PER_FT,
                           coherence_min=0.4, order=2),
    # Stoneley starts at the borehole-fluid slowness floor (~200 us/ft
    # for a typical mud); below that you are in the pseudo-Rayleigh /
    # guided regime, not Stoneley. Keeping the windows non-overlapping
    # is what allows the four-mode picker to pick Stoneley correctly
    # in a gather that also carries a pseudo-Rayleigh peak.
    "Stoneley": dict(slow_min=200.0 * US_PER_FT, slow_max=360.0 * US_PER_FT,
                     coherence_min=0.4, order=3),
}

# Both :func:`viterbi_pick_joint` and :func:`viterbi_posterior_marginals`
# are now N-mode generic and default to the full :data:`DEFAULT_PRIORS`
# (4 modes: P / S / PseudoRayleigh / Stoneley). The auto-fallback
# variable-candidate-budget in :func:`_build_triple_trellis` keeps the
# wider trellis tractable; pass an explicit subset via the ``priors``
# argument to restrict to fewer modes.

# Mnemonic suffix per canonical mode name, used by
# :func:`track_to_log_curves` to build LAS/DLIS-friendly column names
# (DTP / DTS / DTST / DTPR, COHP / COHS / COHST / COHPR, etc.). Modes
# outside this map fall through to ``mode_name.upper()``.
_MODE_MNEMONIC_SUFFIX: dict[str, str] = {
    "P":              "P",
    "S":              "S",
    "Stoneley":       "ST",
    "PseudoRayleigh": "PR",
}


@dataclass
class ModePick:
    """A single mode pick at one depth.

    ``amplitude`` is the per-trace stack amplitude at the picked
    (slowness, time) cell of the STC surface (see
    :attr:`fwap.coherence.STCResult.amplitude` for the exact
    definition). It is the second leg -- alongside ``coherence`` --
    of the per-mode amplitude/coherence log pair that Mari et al.
    (1994), Part 1 list as the rule-based picker's deliverable.
    ``None`` when the upstream STC variant did not populate
    ``STCResult.amplitude`` (currently only some legacy paths).
    """

    name: str
    slowness: float
    time: float
    coherence: float
    amplitude: float | None = None

    def __repr__(self) -> str:
        amp = (f", amp={self.amplitude:.3g}"
               if self.amplitude is not None else "")
        return (f"ModePick({self.name!r}, "
                f"slowness={self.slowness / US_PER_FT:.2f} us/ft, "
                f"t={self.time * 1e3:.2f} ms, "
                f"coh={self.coherence:.3f}{amp})")


@dataclass
class DepthPicks:
    depth: float
    picks: dict[str, ModePick] = field(default_factory=dict)

    def __repr__(self) -> str:
        if not self.picks:
            return f"DepthPicks(depth={self.depth:.2f} m, picks={{}})"
        body = ", ".join(
            f"{n}={p.slowness / US_PER_FT:.1f}us/ft@{p.coherence:.2f}"
            for n, p in self.picks.items()
        )
        return f"DepthPicks(depth={self.depth:.2f} m, {body})"


_VALID_SELECTION_RULES = frozenset({"max_coherence", "scored"})


def _best_candidate(candidates: PeakArray,
                    prior: dict[str, float],
                    *,
                    t_earliest: float = 0.0,
                    selection_rule: SelectionRule = "scored",
                    time_penalty: float = 0.1,
                    time_scale: float = 1.0e-3,
                    ) -> PeakRow | None:
    """
    Select the best candidate inside a prior window.

    Parameters
    ----------
    candidates : ndarray, shape (n, 3)
        Rows of ``[slowness, time, coherence]``, typically the output
        of :func:`find_peaks`.
    prior : dict
        Must contain ``slow_min``, ``slow_max``, ``coherence_min``
        keys bounding the physically-reasonable region.
    t_earliest : float
        Lower bound on the candidate's arrival time (used by the
        mode-ordering rule in :func:`pick_modes`). Only affects the
        ``'scored'`` rule; the caller is responsible for pre-filtering
        by ``t_earliest`` if a hard cutoff is wanted.
    selection_rule : {'max_coherence', 'scored'}
        How to pick among candidates in the window:
          * ``'max_coherence'``: highest coherence wins. Preferred
            for late, guided modes like Stoneley, where an earlier
            but weaker noise peak inside the prior window should not
            be preferred over a later, stronger peak.
          * ``'scored'``: rank by
            ``coherence - time_penalty * max(0, (t - t_earliest) /
            time_scale)``, giving a soft preference for earlier
            arrivals that does not override clear coherence
            differences. Falls back to the ``'max_coherence'`` result
            when ``time_penalty == 0``.
    time_penalty, time_scale : float
        Only used with ``selection_rule='scored'``.

    Returns
    -------
    ndarray, shape (3,) or None
        The winning row, or ``None`` if the window is empty.
    """
    if selection_rule not in _VALID_SELECTION_RULES:
        raise ValueError(
            f"selection_rule must be one of {sorted(_VALID_SELECTION_RULES)}"
        )
    if candidates.size == 0:
        return None
    mask = ((candidates[:, 0] >= prior["slow_min"]) &
            (candidates[:, 0] <= prior["slow_max"]) &
            (candidates[:, 2] >= prior["coherence_min"]))
    c = candidates[mask]
    if c.size == 0:
        return None

    if selection_rule == "max_coherence":
        idx = int(np.argmax(c[:, 2]))
    else:  # "scored"
        time_excess = np.clip(c[:, 1] - t_earliest, 0.0, None)
        score = c[:, 2] - time_penalty * (time_excess / max(time_scale, 1e-12))
        idx = int(np.argmax(score))
    return c[idx]


def pick_modes(stc_result: STCResult,
               priors: dict[str, dict[str, float]] | None = None,
               threshold: float = 0.4,
               *,
               selection_rule: SelectionRule = "scored",
               time_penalty: float = 0.1,
               time_scale: float | None = None,
               ) -> dict[str, ModePick]:
    """
    Label P / S / PseudoRayleigh / Stoneley modes on an STC surface
    via physical rules.

    The ``selection_rule`` keyword controls how ties within a prior
    window are broken; see :func:`_best_candidate` for the three
    strategies. PseudoRayleigh is silently absent on gathers that
    don't carry a guided-mode arrival -- its prior window is empty
    on a 3-mode (P + S + Stoneley) synthetic.

    Parameters
    ----------
    stc_result : STCResult
        Slowness-time coherence surface from :func:`stc`.
    priors : dict, optional
        Per-mode prior windows. Defaults to :data:`DEFAULT_PRIORS`.
    threshold : float
        Coherence threshold for the peak picker.
    selection_rule : str
        See :func:`_best_candidate`.
    time_penalty : float
        Weight of the time penalty in the ``'scored'`` rule; ignored
        otherwise.
    time_scale : float, optional
        Time normaliser for the ``'scored'`` rule. Defaults to
        ``stc_result.window_length``.

    Notes
    -----
    The mode-ordering rule is still strict (P -> S -> Stoneley, each
    required to be no earlier in time than the previous). In altered
    zones the S head-wave can appear before the formation P re-emerges
    (see Aron et al., 1994, *SEG Expanded Abstracts*); for those cases
    a joint log-likelihood across modes is more robust than the greedy
    rule used here. This is flagged as future work.
    """
    if priors is None:
        priors = DEFAULT_PRIORS
    if time_scale is None:
        time_scale = max(stc_result.window_length, 1e-12)
    peaks = find_peaks(stc_result, threshold=threshold)
    out: dict[str, ModePick] = {}
    t_earliest = 0.0
    for name in sorted(priors, key=lambda n: priors[n]["order"]):
        prior = priors[name]
        valid = peaks[peaks[:, 1] >= t_earliest] if peaks.size else peaks
        winner = _best_candidate(
            valid, prior,
            t_earliest=t_earliest,
            selection_rule=selection_rule,
            time_penalty=time_penalty,
            time_scale=time_scale,
        )
        if winner is None:
            continue
        out[name] = ModePick(
            name=name, slowness=float(winner[0]),
            time=float(winner[1]), coherence=float(winner[2]),
            amplitude=float(winner[3]) if winner.size >= 4 else None,
        )
        t_earliest = max(t_earliest, float(winner[1]))
    return out


def track_modes(stc_results: Sequence[STCResult],
                depths: np.ndarray,
                priors: dict[str, dict[str, float]] | None = None,
                threshold: float = 0.4,
                max_slow_jump: float = 50.0 * US_PER_FT,
                continuity_max_gap: float | None = None,
                continuity_tol_growth: float = 0.5,
                continuity_tol_cap_factor: float = 3.0,
                *,
                selection_rule: SelectionRule = "scored",
                time_penalty: float = 0.1,
                time_scale: float | None = None,
                ) -> list[DepthPicks]:
    """
    Per-depth picking with a depth-aware continuity regulariser.

    The continuity constraint stores both the last successful pick's
    slowness and the depth at which it was picked per mode. The
    effective jump tolerance grows with the depth gap since the last
    pick, so a mode missed in a disturbed zone can be re-acquired at a
    slightly different slowness a few depths later without the tracker
    treating that as a violation. Beyond ``continuity_max_gap`` the
    constraint is dropped entirely, preventing the tracker from
    remaining locked onto a stale slowness across extended data gaps.

    Parameters
    ----------
    stc_results : sequence of STCResult
        One STC surface per depth.
    depths : ndarray, shape (n_depth,)
        Tool depth (m) for each STC surface, same length as
        ``stc_results``.
    priors : dict, optional
        Per-mode prior windows. Defaults to :data:`DEFAULT_PRIORS`.
    threshold : float
        Coherence threshold for the peak picker.
    max_slow_jump : float
        Slowness jump tolerance (s/m) between adjacent depths with
        zero gap growth. Scaled up by ``continuity_tol_growth`` for
        each depth-unit of gap since the last successful pick on the
        same mode. Default: 50 us/ft.
    continuity_max_gap : float, optional
        Depth gap (m) beyond which the continuity constraint is
        dropped entirely for a given mode. Defaults to five times the
        median depth spacing of ``depths`` (or infinity for a single
        depth).
    continuity_tol_growth : float, default 0.5
        Fractional growth of ``max_slow_jump`` per unit depth gap.
        Set to 0 for a gap-independent absolute tolerance.
    continuity_tol_cap_factor : float, default 3.0
        Hard cap on the effective tolerance, expressed as a multiple of
        ``max_slow_jump``. Prevents runaway widening of the continuity
        window when a caller sets a large ``continuity_max_gap``. The
        cap is disabled by passing ``float("inf")``.
    selection_rule, time_penalty, time_scale
        Passed through to :func:`_best_candidate`; see its docs.

    Returns
    -------
    list of DepthPicks
    """

    if priors is None:
        priors = DEFAULT_PRIORS

    depths = np.asarray(depths, dtype=float)
    if continuity_max_gap is None:
        if depths.size >= 2:
            continuity_max_gap = 5.0 * float(
                np.median(np.abs(np.diff(depths)))
            )
        else:
            continuity_max_gap = float("inf")

    cand_lists = [find_peaks(r, threshold=threshold) for r in stc_results]

    if time_scale is None:
        if stc_results:
            time_scale = max(stc_results[0].window_length, 1e-12)
        else:
            time_scale = 1.0e-3

    # Track (slowness, depth) of the last successful pick per mode.
    last: dict[str, tuple[float, float]] = {}
    all_picks: list[DepthPicks] = []

    for depth, peaks in zip(depths, cand_lists):
        dp = DepthPicks(depth=float(depth))
        t_earliest = 0.0
        for name in sorted(priors, key=lambda n: priors[n]["order"]):
            prior = priors[name]
            valid = peaks[peaks[:, 1] >= t_earliest] if peaks.size else peaks

            if name in last and valid.size:
                last_s, last_d = last[name]
                gap = float(abs(depth - last_d))
                if gap <= continuity_max_gap:
                    effective_tol = max_slow_jump * (
                        1.0 + continuity_tol_growth * gap
                    )
                    # Cap runaway widening after many consecutive
                    # missed picks (gap keeps growing until a success
                    # resets last_d). Without this, a noise peak far
                    # from the true mode can be reacquired once the
                    # tolerance exceeds any physical jump.
                    effective_tol = min(
                        effective_tol,
                        continuity_tol_cap_factor * max_slow_jump,
                    )
                    jump = np.abs(valid[:, 0] - last_s)
                    valid_cont = valid[jump <= effective_tol]
                    if valid_cont.size:
                        valid = valid_cont
                    # If nothing survives the jump filter, fall through
                    # to the prior-window-only set rather than failing
                    # outright -- the primary picker's coherence
                    # threshold still guards against noise.
                # gap > continuity_max_gap: drop constraint entirely.

            winner = _best_candidate(
                valid, prior,
                t_earliest=t_earliest,
                selection_rule=selection_rule,
                time_penalty=time_penalty,
                time_scale=time_scale,
            )
            if winner is None:
                # Fallback: retry against the prior-only window,
                # still honouring t_earliest. Dropping t_earliest
                # here would admit a pick earlier than a previously-
                # picked mode, which violates mode ordering.
                valid_fb = (peaks[peaks[:, 1] >= t_earliest]
                            if peaks.size else peaks)
                if valid_fb.size:
                    winner = _best_candidate(
                        valid_fb, prior,
                        t_earliest=t_earliest,
                        selection_rule=selection_rule,
                        time_penalty=time_penalty,
                        time_scale=time_scale,
                    )
                if winner is None:
                    continue

            pick = ModePick(
                name=name, slowness=float(winner[0]),
                time=float(winner[1]), coherence=float(winner[2]),
                amplitude=float(winner[3]) if winner.size >= 4 else None,
            )
            dp.picks[name] = pick
            last[name] = (pick.slowness, float(depth))
            t_earliest = max(t_earliest, pick.time)
        all_picks.append(dp)
    return all_picks


# ---------------------------------------------------------------------
# Viterbi picker (joint log-likelihood across depths)
# ---------------------------------------------------------------------

_NEG_INF = float("-inf")


def viterbi_pick(stc_results: Sequence[STCResult],
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
            mask = ((peaks[:, 0] >= prior["slow_min"]) &
                    (peaks[:, 0] <= prior["slow_max"]) &
                    (peaks[:, 2] >= prior["coherence_min"]) &
                    (peaks[:, 1] >= t_floor))
            per_depth.append(peaks[mask])

        # Viterbi forward pass. State at each depth = index into that
        # depth's per_depth[d] array, with an extra "absent" state
        # represented by -1. We store trellis arrays as lists because
        # candidate counts vary per depth.
        scores: list[np.ndarray] = []            # (n_candidates + 1,) per depth
        back_ptrs: list[np.ndarray] = []         # int: index into prev depth

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
                emission[:n_cand] = (
                    np.log(np.clip(cands[:, 2], 1.0e-12, None))
                    - time_prior_weight * (cands[:, 1] - t_ref)
                )
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
                continue   # mode absent at this depth
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
# Joint 3-mode Viterbi (state = (P, S, Stoneley) triple)
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
            prod *= (min(n, K) + 1)
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
    per_mode_per_depth: dict[str, list[np.ndarray]] = {
        name: [] for name in mode_names
    }
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
        per_mode_cands = [per_mode_per_depth[name][d]
                          for name in mode_names]
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
            raw_count *= (n + 1)
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
                "trellis: depth %d auto-fallback K=%d "
                "(raw=%d, n_per_mode %s -> %s)",
                d, auto_K, raw_count, n_per_mode, new_n_per_mode,
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

        triples.append(np.asarray(rows_triples, dtype=np.intp).reshape(
            -1, n_modes))
        emissions.append(np.asarray(rows_emission, dtype=float))
        slows.append(np.asarray(rows_slow, dtype=float).reshape(
            -1, n_modes))

    return _TripleTrellis(
        mode_names=mode_names,
        per_mode_per_depth=per_mode_per_depth,
        triples=triples,
        emissions=emissions,
        slows=slows,
    )


def _joint_transition_matrix(prev_slow: np.ndarray,
                             curr_slow: np.ndarray,
                             slow_jump_sigma: float) -> np.ndarray:
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
        raise ValueError(
            "priors must contain at least one mode; got an empty dict."
        )
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
    back_ptrs: list[np.ndarray] = [
        np.full(scores[0].size, -1, dtype=np.intp)
    ]

    for d in range(1, n_depth):
        total_trans = _joint_transition_matrix(
            slows[d - 1], slows[d], slow_jump_sigma)
        step = (scores[d - 1][:, None]
                - total_trans
                + emissions[d][None, :])
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
        per_mode_cands = [per_mode_per_depth[name][d]
                          for name in mode_names]
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


# ---------------------------------------------------------------------
# Posterior marginals via forward-backward
# ---------------------------------------------------------------------


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
        raise ValueError(
            "priors must contain at least one mode; got an empty dict."
        )
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
    back_ptrs: list[np.ndarray] = [
        np.full(scores[0].size, -1, dtype=np.intp)
    ]
    for d in range(1, n_depth):
        total_trans = _joint_transition_matrix(
            slows[d - 1], slows[d], slow_jump_sigma)
        step = (scores[d - 1][:, None]
                - total_trans
                + emissions[d][None, :])
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
        per_mode_cands = [per_mode_per_depth[name][d]
                          for name in mode_names]
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
        total_trans = _joint_transition_matrix(
            slows[d - 1], slows[d], slow_jump_sigma)
        combined = alpha[d - 1][:, None] - total_trans
        alpha.append(emissions[d] + _logsumexp(combined, axis=0))

    # Log-sum-exp backward pass (beta).
    beta: list[np.ndarray] = [np.zeros(triples[-1].shape[0], dtype=float)]
    for d in range(n_depth - 2, -1, -1):
        total_trans = _joint_transition_matrix(
            slows[d], slows[d + 1], slow_jump_sigma)
        combined = (emissions[d + 1][None, :]
                    + beta[0][None, :]
                    - total_trans)
        beta.insert(0, _logsumexp(combined, axis=1))

    # Posterior marginals over triples: gamma[d][j] normalised.
    posteriors: list[dict[str, PosteriorPick]] = []
    for d in range(n_depth):
        log_gamma = alpha[d] + beta[d]
        log_norm = _logsumexp(log_gamma)
        probs_triple = np.exp(log_gamma - log_norm)

        mode_post: dict[str, PosteriorPick] = {}
        tri_matrix = triples[d]             # (n_triples, n_modes)
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
                probs_cand[c] = float(
                    probs_triple[mode_col == c].sum()
                )
            mode_post[name] = PosteriorPick(
                slownesses=cands[:, 0].copy(),
                times=cands[:, 1].copy(),
                coherences=cands[:, 2].copy(),
                probabilities=probs_cand,
                p_absent=p_absent,
            )
        posteriors.append(mode_post)

    return map_picks, posteriors


# ---------------------------------------------------------------------
# Wavelet-shape + onset-polarity expert rules (post-pick filters)
# ---------------------------------------------------------------------


def _align_and_stack(data: np.ndarray,
                     dt: float,
                     offsets: np.ndarray,
                     slowness: float,
                     stc_window_start: float,
                     stc_window_length: float,
                     analysis_factor: float = 2.0,
                     ) -> np.ndarray:
    """Frequency-domain align + per-trace average over an analysis window.

    Returns the per-trace mean of the moveout-aligned waveforms over
    a window of width ``analysis_factor * stc_window_length`` centred
    on the **midpoint** of the STC window (i.e. on
    ``stc_window_start + stc_window_length / 2``). Centring -- rather
    than starting -- on the STC window's midpoint matters for the
    polarity / shape gates because ``pick.time`` is a window-*start*
    time; a window started there may capture only a sidelobe of the
    underlying wavelet, whose true centroid sits roughly at the STC
    window's centre. The widened analysis window is also clipped to
    the available data range, so picks at the very edge of the
    record still produce a usable stack.
    """
    n_rec, n_samp = data.shape
    spec = np.fft.rfft(data, axis=1)
    f = np.fft.rfftfreq(n_samp, d=dt)
    rel_off = offsets - offsets[0]
    tau = rel_off * slowness
    shifted = np.fft.irfft(_phase_shift(spec, f, tau), n=n_samp, axis=1)
    L_analysis = max(2, int(round(analysis_factor * stc_window_length / dt)))
    centre_sample = int(round((stc_window_start + stc_window_length / 2.0) / dt))
    j0 = max(0, min(n_samp - L_analysis, centre_sample - L_analysis // 2))
    window = shifted[:, j0:j0 + L_analysis]
    return window.mean(axis=0)


def onset_polarity(stack: np.ndarray) -> int:
    """Sign of the largest-absolute sample in a stacked waveform.

    Returns ``+1`` when the dominant excursion is positive, ``-1``
    when negative, and ``0`` for an all-zero input. The book's "onset
    polarity" expert rule (Mari et al. 1994, Part 1) gates picks
    against an expected first-motion sign convention; the dominant
    excursion of an STC-window-stacked pulse-like wavelet is the
    main lobe at the prior centre frequency, so its sign is the
    natural per-pick polarity readout.

    Caveat -- pulse-like wavelets only
    ----------------------------------
    This heuristic is only meaningful for *pulse-like* wavelets
    where one excursion clearly dominates the others (Ricker P /
    Ricker S, in the canonical 1994 monopole gather). For
    multi-cycle wavetrains -- Gabor / Stoneley -- the dominant
    excursion can be either a peak or a trough depending on the
    sub-sample alignment of the analysis window, so the polarity
    readout is not stable. In practice the polarity gate in
    :data:`DEFAULT_PRIORS` should only be enabled on the impulsive
    modes (P, S); leave Stoneley and PseudoRayleigh at
    ``polarity=0`` (the default = "ignore").
    """
    if stack.size == 0:
        return 0
    j = int(np.argmax(np.abs(stack)))
    val = float(stack[j])
    if val > 0.0:
        return 1
    if val < 0.0:
        return -1
    return 0


def wavelet_shape_score(stack: np.ndarray,
                        dt: float,
                        f0: float,
                        ) -> float:
    """Absolute Pearson correlation of a stacked window vs a Ricker(f0).

    Returns a score in ``[0, 1]``: ``1.0`` for a stacked window
    that is exactly a Ricker at the prior centre frequency
    ``f0`` (modulo amplitude and time shift), ``0.0`` for an
    uncorrelated waveform. The Ricker template is centred on the
    location of ``stack``'s largest-absolute sample so the
    correlation is invariant to sub-window jitter of the picked
    arrival time.

    The score is **polarity-blind** -- it returns the absolute
    correlation -- so it can be combined orthogonally with
    :func:`onset_polarity` for a separate sign check.
    """
    n = stack.size
    if n < 2:
        return 0.0
    j_peak = int(np.argmax(np.abs(stack)))
    t = (np.arange(n) - j_peak) * dt
    a = (np.pi * f0 * t) ** 2
    template = (1.0 - 2.0 * a) * np.exp(-a)

    s = stack - stack.mean()
    tmpl = template - template.mean()
    s_norm = float(np.sqrt(np.sum(s * s)))
    t_norm = float(np.sqrt(np.sum(tmpl * tmpl)))
    if s_norm == 0.0 or t_norm == 0.0:
        return 0.0
    return float(abs(np.sum(s * tmpl)) / (s_norm * t_norm))


def _filter_one_depth(picks: dict[str, ModePick],
                      data: np.ndarray,
                      dt: float,
                      offsets: np.ndarray,
                      priors: dict[str, dict[str, float]],
                      window_length: float,
                      analysis_factor: float,
                      ) -> dict[str, ModePick]:
    """Apply per-mode polarity / shape gates to one depth's picks."""
    out: dict[str, ModePick] = {}
    for name, pick in picks.items():
        prior = priors.get(name, {})
        polarity_expected = int(prior.get("polarity", 0))
        shape_min = float(prior.get("shape_match_min", 0.0))
        if polarity_expected == 0 and shape_min <= 0.0:
            # Neither gate enabled for this mode -- keep the pick.
            out[name] = pick
            continue

        stack = _align_and_stack(
            data, dt, offsets, pick.slowness, pick.time, window_length,
            analysis_factor=analysis_factor)

        if polarity_expected != 0:
            actual = onset_polarity(stack)
            if actual != polarity_expected:
                continue   # polarity mismatch -- drop pick

        if shape_min > 0.0:
            f0 = prior.get("f0")
            if f0 is None:
                raise ValueError(
                    f"prior for {name!r} sets shape_match_min={shape_min} "
                    f"but no `f0` (Hz) -- the wavelet-shape gate needs the "
                    f"per-mode centre frequency"
                )
            score = wavelet_shape_score(stack, dt, float(f0))
            if score < shape_min:
                continue   # shape mismatch -- drop pick

        out[name] = pick
    return out


def filter_picks_by_shape(picks: dict[str, ModePick],
                          data: np.ndarray,
                          dt: float,
                          offsets: np.ndarray,
                          *,
                          priors: dict[str, dict[str, float]] | None = None,
                          window_length: float = 4.0e-4,
                          analysis_factor: float = 2.0,
                          ) -> dict[str, ModePick]:
    """
    Drop picks whose stacked waveform fails the polarity / shape rules.

    Implements two of the book's expert-rule layers (Mari et al.
    1994, Part 1) as a post-pick filter, applied to the dict of
    per-mode picks returned by :func:`pick_modes`. Picks whose mode
    in ``priors`` declares neither rule are passed through unchanged.

    Two opt-in rules per mode (read from each mode's prior dict):

    * ``polarity`` (``+1`` / ``-1`` / ``0``): expected sign of the
      stacked window's largest-absolute sample. ``0`` (the default)
      disables the gate.
    * ``shape_match_min`` (float in ``[0, 1]``): minimum absolute
      Pearson correlation between the stacked window and a Ricker
      template at the prior's ``f0`` (Hz). ``0.0`` disables the
      gate. When enabled, the prior **must** also carry an ``f0``
      key.

    Parameters
    ----------
    picks : dict from str to ModePick
        Per-mode picks at one depth, e.g. the output of
        :func:`pick_modes`.
    data : ndarray, shape (n_rec, n_samples)
        The same gather the picks were derived from. Used to
        re-stack the moveout-aligned window at each pick's
        ``(slowness, time)``.
    dt : float
        Sampling interval (s).
    offsets : ndarray, shape (n_rec,)
        Source-to-receiver offsets (m).
    priors : dict, optional
        Per-mode prior windows, with optional ``polarity``,
        ``shape_match_min`` and ``f0`` keys driving the gates.
        Defaults to :data:`DEFAULT_PRIORS`, which declares neither
        gate so the default-priors call is a no-op pass-through.
    window_length : float, default 4e-4
        STC time-window length (s) used to produce the picks. The
        polarity / shape gates analyse a window centred on the STC
        window's midpoint -- ``pick.time + window_length / 2`` --
        because ``pick.time`` is a window-start time and a window
        started there can capture a sidelobe rather than the main
        wavelet lobe.
    analysis_factor : float, default 2.0
        Width of the analysis window as a multiple of
        ``window_length``. The default ``2.0`` gives the polarity /
        shape gates one STC-window of context on each side of the
        STC window's centre, which is enough to bracket a Ricker's
        main lobe plus its near sidelobes for the typical sonic
        prior frequencies (3-15 kHz).

    Returns
    -------
    dict from str to ModePick
        A new dict containing only the picks that passed every
        enabled gate. The original ``picks`` dict is not mutated.
    """
    if priors is None:
        priors = DEFAULT_PRIORS
    return _filter_one_depth(picks, data, dt, offsets, priors,
                             window_length, analysis_factor)


def filter_track_by_shape(track_picks: Sequence[DepthPicks],
                          datas: Sequence[np.ndarray],
                          dt: float,
                          offsets: np.ndarray,
                          *,
                          priors: dict[str, dict[str, float]] | None = None,
                          window_length: float = 4.0e-4,
                          analysis_factor: float = 2.0,
                          ) -> list[DepthPicks]:
    """Apply :func:`filter_picks_by_shape` per-depth across a track.

    The multi-depth analogue of :func:`filter_picks_by_shape`: the
    same polarity / shape gates run once per depth against the
    matching per-depth gather. ``track_picks`` and ``datas`` must
    have the same length.

    Parameters
    ----------
    track_picks : sequence of DepthPicks
        Output of :func:`track_modes`, :func:`viterbi_pick`, or
        :func:`viterbi_pick_joint`.
    datas : sequence of ndarray, shape (n_rec, n_samples)
        One gather per depth, in the same order as ``track_picks``.
    dt, offsets : as in :func:`filter_picks_by_shape`.
    priors, window_length : as in :func:`filter_picks_by_shape`.

    Returns
    -------
    list of DepthPicks
        Filtered track. ``DepthPicks`` instances are new; the
        ``picks`` dicts are filtered copies of the originals.
    """
    if priors is None:
        priors = DEFAULT_PRIORS
    if len(track_picks) != len(datas):
        raise ValueError(
            f"track_picks and datas must have the same length; got "
            f"{len(track_picks)} and {len(datas)}"
        )
    out: list[DepthPicks] = []
    for dp, data in zip(track_picks, datas):
        filt = _filter_one_depth(
            dp.picks, data, dt, offsets, priors, window_length,
            analysis_factor)
        out.append(DepthPicks(depth=dp.depth, picks=filt))
    return out


# ---------------------------------------------------------------------
# Cross-mode consistency QC
# ---------------------------------------------------------------------


# Canonical sedimentary-rock Vp/Vs band, used as the default gate in
# :func:`quality_control_picks`. Sources: gas-charged sandstones
# bottom out around ~1.4; high-clay shales and saturated carbonates
# top out around ~2.5-2.6 (Castagna et al. 1985; Mavko, Mukerji &
# Dvorkin, *Rock Physics Handbook*, 2nd ed., chap. 7).
_DEFAULT_VP_VS_MIN = 1.4
_DEFAULT_VP_VS_MAX = 2.6

# Physically-reasonable Thomsen gamma band, used as the default gate
# in :func:`quality_control_picks` when a per-depth gamma is supplied.
# The canonical *VTI shale* window is the tighter ``[0.05, 0.30]``
# (Thomsen 1986; Tang & Cheng 2004 sect. 5.4); the defaults here are
# wider so the gate catches obvious mispicks (gamma < -0.05 is
# unusual; gamma > 0.50 almost always means bad picks or a violated
# VTI assumption) without false-positive-ing isotropic carbonates or
# clean sands at gamma ~ 0.
_DEFAULT_GAMMA_MIN = -0.05
_DEFAULT_GAMMA_MAX = 0.50

# Canonical mode time ordering: P first, then S, then the guided
# pseudo-Rayleigh, then Stoneley last. Modes not in this list (or
# absent from the picks dict) are silently skipped.
_CANONICAL_MODE_TIME_ORDER = ("P", "S", "PseudoRayleigh", "Stoneley")


@dataclass(frozen=True)
class PickQualityFlags:
    """
    Per-depth cross-mode consistency QC for a multi-mode pick set.

    Returned by :func:`quality_control_picks` (and the multi-depth
    :func:`quality_control_track`). Covers the *cross-consistency
    between modes* layer of the book's QC philosophy (Mari et al.
    1994, Part 1, closing paragraph) -- the *log continuity* layer
    is enforced inside the Viterbi pickers.

    Attributes
    ----------
    depth : float
        Depth (m) the QC was computed at.
    vp_vs : float or None
        Vp/Vs ratio derived from the P and S picks
        (= ``s_S / s_P``). ``None`` when either pick is missing or
        when ``s_P`` is zero.
    vp_vs_in_band : bool
        True when ``vp_vs`` lies inside the configured
        ``[vp_vs_min, vp_vs_max]`` physical band, *or* when no
        Vp/Vs could be computed (a missing Vp/Vs is not flagged as
        inconsistent -- it just isn't checked).
    time_order_ok : bool
        True when the picked arrival times respect the canonical
        ordering ``t_P <= t_S <= t_PseudoRayleigh <= t_Stoneley``,
        skipping modes that weren't picked. Useful when the
        upstream picker was run with a soft time-order constraint
        (``viterbi_pick(time_order_slack > 0)`` /
        ``viterbi_pick_joint(soft_time_order=...)``) where the
        ordering can be deliberately violated.
    gamma : float or None
        Thomsen shear-anisotropy parameter for this depth, as passed
        in via the ``gamma`` keyword to :func:`quality_control_picks`.
        ``None`` when not supplied (the gate is then skipped).
    gamma_in_band : bool
        True when ``gamma`` lies inside the configured
        ``[gamma_min, gamma_max]`` band, *or* when no ``gamma`` was
        supplied (a missing gamma is not flagged -- it just isn't
        checked).
    flagged : bool
        True when any check failed at this depth.
    reasons : tuple of str
        Human-readable per-check failure descriptions; empty when
        ``flagged`` is False.
    """
    depth: float
    vp_vs: float | None
    vp_vs_in_band: bool
    time_order_ok: bool
    flagged: bool
    reasons: tuple[str, ...]
    gamma: float | None = None
    gamma_in_band: bool = True


def quality_control_picks(picks: dict[str, ModePick] | DepthPicks,
                          *,
                          depth: float | None = None,
                          vp_vs_min: float = _DEFAULT_VP_VS_MIN,
                          vp_vs_max: float = _DEFAULT_VP_VS_MAX,
                          require_time_order: bool = True,
                          gamma: float | None = None,
                          gamma_min: float = _DEFAULT_GAMMA_MIN,
                          gamma_max: float = _DEFAULT_GAMMA_MAX,
                          ) -> PickQualityFlags:
    """
    Cross-mode consistency QC at one depth.

    Three checks (all opt-out / opt-in):

    * **Vp/Vs in physical band.** Computed as ``s_S / s_P`` (which
      equals Vp/Vs since slowness is the reciprocal of velocity).
      Flagged when outside ``[vp_vs_min, vp_vs_max]``. Skipped --
      and reported as ``vp_vs_in_band=True`` -- when either P or S
      is missing.
    * **Canonical time ordering.** Flagged when the picked arrival
      times do not satisfy ``t_P <= t_S <= t_PseudoRayleigh <=
      t_Stoneley`` over the modes that were picked. Disable by
      passing ``require_time_order=False``.
    * **Thomsen gamma in physical band** (opt-in). Flagged when the
      Thomsen shear-anisotropy parameter (computed externally via
      :func:`fwap.anisotropy.thomsen_gamma_from_logs` or
      :func:`fwap.anisotropy.vti_moduli_from_logs` and passed in
      via the ``gamma`` keyword) lies outside
      ``[gamma_min, gamma_max]``. Skipped -- and reported as
      ``gamma_in_band=True`` -- when ``gamma`` is ``None``. The
      default band is wider than the canonical VTI shale window
      (Thomsen 1986; Tang & Cheng 2004 sect. 5.4 give shales at
      ``[0.05, 0.30]``); the wider default catches mispicks
      (negative gamma is unusual; gamma > 0.50 almost always
      indicates bad picks or a violated VTI assumption) without
      false-positive-ing isotropic carbonates or clean sands at
      gamma ~ 0. Tighten to ``gamma_min=0.05, gamma_max=0.30`` if
      you specifically want a "this depth is in a VTI shale" gate.

    The function only **flags** -- it never modifies the picks. The
    caller decides what to do with flagged depths (drop, mark in
    plots, hand to a human for review).

    Parameters
    ----------
    picks : dict from str to ModePick, or DepthPicks
        Per-mode picks at one depth. When a :class:`DepthPicks` is
        passed, its ``depth`` field is used unless overridden by
        the explicit ``depth`` keyword.
    depth : float, optional
        Override depth value. Required when ``picks`` is a plain
        dict; ignored otherwise unless given.
    vp_vs_min, vp_vs_max : float
        Inclusive Vp/Vs band. Defaults span the canonical
        sedimentary-rock range from gas-charged sandstones (~1.4)
        to clay-rich shales / fluid-saturated carbonates (~2.6).
    require_time_order : bool, default True
        Disable to skip the canonical-ordering check entirely.
    gamma : float, optional
        Thomsen shear-anisotropy parameter for this depth. When
        supplied the function checks it against
        ``[gamma_min, gamma_max]``. ``None`` (default) skips the
        gate.
    gamma_min, gamma_max : float
        Inclusive Thomsen-gamma band. Defaults
        ``[-0.05, 0.50]`` flag obvious mispicks without
        false-positive-ing isotropic samples at gamma ~ 0.

    Returns
    -------
    PickQualityFlags
    """
    if isinstance(picks, DepthPicks):
        if depth is None:
            depth = picks.depth
        picks_dict = picks.picks
    else:
        if depth is None:
            depth = float("nan")
        picks_dict = picks

    reasons: list[str] = []

    # Vp/Vs gate
    vp_vs: float | None = None
    vp_vs_in_band = True
    if "P" in picks_dict and "S" in picks_dict:
        s_p = float(picks_dict["P"].slowness)
        s_s = float(picks_dict["S"].slowness)
        if s_p > 0.0:
            vp_vs = s_s / s_p
            if not (vp_vs_min <= vp_vs <= vp_vs_max):
                vp_vs_in_band = False
                reasons.append(
                    f"Vp/Vs={vp_vs:.2f} outside band "
                    f"[{vp_vs_min:.2f}, {vp_vs_max:.2f}]"
                )

    # Canonical time-ordering gate
    time_order_ok = True
    if require_time_order:
        present = [m for m in _CANONICAL_MODE_TIME_ORDER if m in picks_dict]
        times = [float(picks_dict[m].time) for m in present]
        if times != sorted(times):
            time_order_ok = False
            order_str = ", ".join(
                f"t_{m}={picks_dict[m].time * 1.0e3:.2f}ms"
                for m in present
            )
            reasons.append(
                f"canonical time order violated: {order_str}"
            )

    # Thomsen-gamma gate (opt-in)
    gamma_value: float | None = None
    gamma_in_band = True
    if gamma is not None:
        gamma_value = float(gamma)
        if not (gamma_min <= gamma_value <= gamma_max):
            gamma_in_band = False
            reasons.append(
                f"Thomsen gamma={gamma_value:.3f} outside band "
                f"[{gamma_min:.2f}, {gamma_max:.2f}]"
            )

    flagged = (not vp_vs_in_band) or (not time_order_ok) or (not gamma_in_band)
    return PickQualityFlags(
        depth=float(depth),
        vp_vs=vp_vs,
        vp_vs_in_band=vp_vs_in_band,
        time_order_ok=time_order_ok,
        flagged=flagged,
        reasons=tuple(reasons),
        gamma=gamma_value,
        gamma_in_band=gamma_in_band,
    )


def quality_control_track(track: Sequence[DepthPicks],
                          *,
                          vp_vs_min: float = _DEFAULT_VP_VS_MIN,
                          vp_vs_max: float = _DEFAULT_VP_VS_MAX,
                          require_time_order: bool = True,
                          gammas: Sequence[float] | np.ndarray | None = None,
                          gamma_min: float = _DEFAULT_GAMMA_MIN,
                          gamma_max: float = _DEFAULT_GAMMA_MAX,
                          ) -> list[PickQualityFlags]:
    """
    Apply :func:`quality_control_picks` per-depth across a track.

    Parameters
    ----------
    track : sequence of DepthPicks
        Output of :func:`track_modes`, :func:`viterbi_pick`,
        :func:`viterbi_pick_joint`, or any other multi-depth picker.
    vp_vs_min, vp_vs_max, require_time_order : as in
        :func:`quality_control_picks`.
    gammas : sequence of float or ndarray, optional
        Per-depth Thomsen-gamma values (one per entry in ``track``).
        When supplied, each is forwarded to the per-depth
        :func:`quality_control_picks` call as the ``gamma`` keyword,
        enabling the gamma-band gate. ``None`` (default) skips the
        gate everywhere. Pass ``np.nan`` for individual depths
        where gamma is unavailable -- the per-depth call treats NaN
        as "skip the gamma gate at this depth".
    gamma_min, gamma_max : float
        Inclusive Thomsen-gamma band, forwarded to every per-depth
        call. See :func:`quality_control_picks` for the convention.

    Returns
    -------
    list of PickQualityFlags
        One entry per depth, in the order of ``track``.

    Raises
    ------
    ValueError
        If ``gammas`` is supplied with a different length than
        ``track``.
    """
    if gammas is None:
        per_depth_gammas: list[float | None] = [None] * len(track)
    else:
        gammas_arr = np.asarray(gammas, dtype=float)
        if gammas_arr.size != len(track):
            raise ValueError(
                "gammas must have the same length as track; got "
                f"len(gammas)={gammas_arr.size}, len(track)={len(track)}"
            )
        # Treat NaN as "skip the gamma gate at this depth".
        per_depth_gammas = [
            None if not np.isfinite(g) else float(g)
            for g in gammas_arr
        ]
    return [
        quality_control_picks(
            dp,
            vp_vs_min=vp_vs_min, vp_vs_max=vp_vs_max,
            require_time_order=require_time_order,
            gamma=per_depth_gammas[i],
            gamma_min=gamma_min, gamma_max=gamma_max,
        )
        for i, dp in enumerate(track)
    ]


# ---------------------------------------------------------------------
# Track -> log-curve bridge (picker output -> LAS/DLIS writer input)
# ---------------------------------------------------------------------


def track_to_log_curves(
    track: Sequence[DepthPicks],
    *,
    modes: Sequence[str] | None = None,
    include_amplitude: bool = True,
    include_coherence: bool = True,
    include_vp_vs: bool = True,
    include_time: bool = False,
    include_vti: bool = False,
    rho: float | np.ndarray | None = None,
    rho_fluid: float | None = None,
    v_fluid: float | None = None,
    correct_for_p_modulus: bool = True,
    null_value: float = float("nan"),
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Convert a per-depth pick track into LAS/DLIS-ready log curves.

    Bridges the picker output (:func:`track_modes`,
    :func:`viterbi_pick`, :func:`viterbi_pick_joint`) and the I/O
    writers (:func:`fwap.io.write_las`, :func:`fwap.io.write_dlis`)
    by building one fixed-length ``(n_depth,)`` array per
    (mode, attribute) pair, keyed by the standard fwap mnemonics.
    Slownesses are converted to **us/ft** (the borehole-acoustic
    unit used by the LAS/DLIS unit table); coherences and amplitudes
    are kept dimensionless / in their input units.

    The Workflow-1 deliverable per Mari et al. (1994), Part 1 is a
    set of continuous Vp / Vs / Stoneley slowness curves with
    matching coherence (and amplitude) tracks. This function is the
    last mile: it produces the dict that
    :func:`fwap.io.write_las(path, depth, curves)` and
    :func:`fwap.io.write_dlis(path, depth, curves)` consume directly.

    Mnemonic conventions
    --------------------
    Canonical mode -> suffix mapping:
      ``P`` -> ``P``, ``S`` -> ``S``, ``Stoneley`` -> ``ST``,
      ``PseudoRayleigh`` -> ``PR``. Modes outside this set use
      ``mode_name.upper()`` as the suffix.

    Per mode, the columns produced are:

    =========  ===================  ========
    Mnemonic   Quantity              Unit
    =========  ===================  ========
    DT*        slowness              us/ft
    COH*       coherence             (-)
    AMP*       per-cell amplitude    (-)
    TIM*       pick time             s
    =========  ===================  ========

    plus a single ``VPVS`` (= ``s_S / s_P`` = ``Vp / Vs``) when both
    P and S are picked.

    With ``include_vti=True`` (and the required ``rho`` /
    ``rho_fluid`` / ``v_fluid`` inputs) the function additionally
    emits the seven VTI columns:

    =========  ============================================  =====
    Mnemonic   Quantity                                       Unit
    =========  ============================================  =====
    C33        :math:`\\rho V_P^2`                            Pa
    C44        :math:`\\rho V_{Sv}^2`                         Pa
    C66        Stoneley-derived horizontal shear modulus      Pa
    GAMMA      Thomsen :math:`\\gamma = (C_{66}-C_{44})/(2 C_{44})` (-)
    VP         :math:`\\sqrt{C_{33}/\\rho}`                   m/s
    VSV        :math:`\\sqrt{C_{44}/\\rho}`                   m/s
    VSH        :math:`\\sqrt{C_{66}/\\rho}`                   m/s
    =========  ============================================  =====

    Each VTI cell is computed only at depths where the underlying
    pick(s) are present:

    * C33 / VP need ``"P"``,
    * C44 / VSV need ``"S"``,
    * C66 / VSH need ``"Stoneley"`` (with a Stoneley slowness above
      :math:`1/V_f`),
    * GAMMA needs both C44 and C66.

    Cells where the relevant pick is missing receive ``null_value``.
    With ``correct_for_p_modulus=True`` (default) C66 uses the Tang
    & Cheng (2004) §5.4 finite-impedance correction at depths where
    the P pick is *also* present; depths where the P pick is
    missing fall back to the literal White (1983) reading
    transparently. This per-depth fall-back is the right
    operational choice for a track that is dense in S/Stoneley but
    sparse in P -- the resulting C66/GAMMA log uses the best
    physics available cell-by-cell rather than dropping out
    entirely on every missed P pick.

    Parameters
    ----------
    track : sequence of DepthPicks
        Output of :func:`track_modes`, :func:`viterbi_pick`,
        :func:`viterbi_pick_joint`, or any other multi-depth picker.
    modes : sequence of str, optional
        Restrict output to these mode names. Defaults to every mode
        that appears anywhere in ``track`` (preserving first-seen
        order).
    include_amplitude : bool, default True
        Emit ``AMP*`` columns. Skipped per mode if no pick of that
        mode carries an amplitude.
    include_coherence : bool, default True
        Emit ``COH*`` columns.
    include_vp_vs : bool, default True
        Emit a ``VPVS`` column when both ``P`` and ``S`` columns
        exist in the output.
    include_time : bool, default False
        Emit ``TIM*`` columns (pick time in seconds). Off by default
        because pick times are intermediate diagnostics rather than
        published log curves.
    include_vti : bool, default False
        Emit the seven VTI columns (``C33``, ``C44``, ``C66``,
        ``GAMMA``, ``VP``, ``VSV``, ``VSH``). Requires ``rho``,
        ``rho_fluid``, ``v_fluid``; raises if any of those is
        ``None``.
    rho : float or ndarray, optional
        Formation bulk density (kg/m^3). Either a scalar (constant
        density across the track) or a length-``n_depth`` per-depth
        array. Required when ``include_vti=True``; ignored
        otherwise.
    rho_fluid : float, optional
        Borehole-fluid density (kg/m^3). Required when
        ``include_vti=True``; ignored otherwise.
    v_fluid : float, optional
        Borehole-fluid acoustic velocity (m/s). Required when
        ``include_vti=True``; ignored otherwise.
    correct_for_p_modulus : bool, default True
        With ``include_vti=True``, apply the Tang & Cheng (2004)
        §5.4 finite-impedance correction to the Stoneley → C66
        inversion at depths where the P pick is also present.
        Depths without a P pick fall back to the literal White
        (1983) reading regardless of this flag. Pass ``False`` to
        force the legacy White path everywhere.
    null_value : float, default ``NaN``
        Fill value at depths where a mode was not picked. ``NaN`` is
        the LAS / DLIS native null marker; pass a numeric sentinel
        like ``-999.25`` if a downstream consumer requires that.

    Returns
    -------
    depths : ndarray, shape (n_depth,)
        Depth axis pulled from ``DepthPicks.depth``, in the same unit
        the picker was called with (typically metres).
    curves : dict[str, ndarray]
        Mnemonic -> ``(n_depth,)`` array. All arrays are aligned on
        ``depths``. Suitable to pass to :func:`fwap.io.write_las` or
        :func:`fwap.io.write_dlis` directly.

    Examples
    --------
    >>> from fwap import (
    ...     track_modes, track_to_log_curves, write_las,
    ... )
    >>> track = track_modes(stc_results, depths)
    >>> depths, curves = track_to_log_curves(track)
    >>> write_las("output.las", depths, curves)
    """
    # Coerce ``null_value`` to a float up-front so a wrong type (e.g.
    # ``None``) raises ``TypeError`` cleanly rather than slipping
    # through the NaN check and producing object-dtype curves.
    null_value = float(null_value)

    n_depth = len(track)
    if n_depth == 0:
        return np.empty(0, dtype=float), {}

    depths = np.array([float(dp.depth) for dp in track], dtype=float)

    if modes is None:
        seen: list[str] = []
        for dp in track:
            for name in dp.picks:
                if name not in seen:
                    seen.append(name)
        modes = seen

    # Build the per-mode columns with NaN as the internal "missing"
    # marker so VPVS arithmetic propagates correctly even when the
    # caller passes a numeric ``null_value``. NaNs are remapped to
    # the requested null_value at the very end.
    nan = float("nan")
    curves: dict[str, np.ndarray] = {}
    for mode in modes:
        suffix = _MODE_MNEMONIC_SUFFIX.get(mode, mode.upper())
        slow_arr = np.full(n_depth, nan, dtype=float)
        coh_arr  = np.full(n_depth, nan, dtype=float)
        amp_arr  = np.full(n_depth, nan, dtype=float)
        time_arr = np.full(n_depth, nan, dtype=float)
        any_amp = False
        any_pick = False
        for d, dp in enumerate(track):
            pick = dp.picks.get(mode)
            if pick is None:
                continue
            any_pick = True
            slow_arr[d] = float(pick.slowness) / US_PER_FT
            coh_arr[d]  = float(pick.coherence)
            time_arr[d] = float(pick.time)
            if pick.amplitude is not None:
                amp_arr[d] = float(pick.amplitude)
                any_amp = True
        if not any_pick:
            # Mode never appeared in this track -- skip rather than
            # emit an all-null column.
            continue
        curves[f"DT{suffix}"] = slow_arr
        if include_coherence:
            curves[f"COH{suffix}"] = coh_arr
        if include_amplitude and any_amp:
            curves[f"AMP{suffix}"] = amp_arr
        if include_time:
            curves[f"TIM{suffix}"] = time_arr

    if include_vp_vs and "DTP" in curves and "DTS" in curves:
        # s_S / s_P = (1/v_S) / (1/v_P) = v_P / v_S = Vp/Vs.
        # Both columns are us/ft, so the unit cancels. Compute on the
        # NaN-marked internals so a missing P or S at any depth gives
        # NaN here; that NaN is converted to ``null_value`` below.
        with np.errstate(divide="ignore", invalid="ignore"):
            vpvs = curves["DTS"] / curves["DTP"]
        vpvs = np.where(np.isfinite(vpvs), vpvs, nan)
        curves["VPVS"] = vpvs

    if include_vti:
        if rho is None:
            raise ValueError(
                "include_vti=True requires `rho` (formation density "
                "in kg/m^3, scalar or per-depth array)"
            )
        if rho_fluid is None or v_fluid is None:
            raise ValueError(
                "include_vti=True requires `rho_fluid` and `v_fluid` "
                "(borehole-fluid density in kg/m^3 and acoustic "
                "velocity in m/s)"
            )
        if rho_fluid <= 0.0 or v_fluid <= 0.0:
            raise ValueError(
                "rho_fluid and v_fluid must be strictly positive"
            )
        rho_arr = np.asarray(rho, dtype=float)
        if rho_arr.ndim == 0:
            rho_arr = np.full(n_depth, float(rho_arr), dtype=float)
        elif rho_arr.shape != (n_depth,):
            raise ValueError(
                "rho must be a scalar or a length-n_depth array; got "
                f"shape {rho_arr.shape} for n_depth={n_depth}"
            )
        if np.any(rho_arr <= 0):
            raise ValueError("rho must be strictly positive everywhere")

        # Per-depth slownesses in s/m (NaN where the pick is missing).
        s_p_arr  = np.full(n_depth, nan, dtype=float)
        s_s_arr  = np.full(n_depth, nan, dtype=float)
        s_st_arr = np.full(n_depth, nan, dtype=float)
        for d, dp in enumerate(track):
            p = dp.picks.get("P")
            if p is not None:
                s_p_arr[d] = float(p.slowness)
            s = dp.picks.get("S")
            if s is not None:
                s_s_arr[d] = float(s.slowness)
            st = dp.picks.get("Stoneley")
            if st is not None:
                s_st_arr[d] = float(st.slowness)

        with np.errstate(divide="ignore", invalid="ignore"):
            c33 = rho_arr / (s_p_arr * s_p_arr)
            c44 = rho_arr / (s_s_arr * s_s_arr)
            # White (1983) C66 forward inversion at every Stoneley-
            # picked depth.
            s_f2 = 1.0 / (v_fluid * v_fluid)
            diff = s_st_arr * s_st_arr - s_f2
            c66_white = np.where(diff > 0.0,
                                 rho_fluid / diff, np.nan)
            if correct_for_p_modulus:
                # Tang & Cheng (2004) §5.4 correction at depths where
                # the P pick is *also* present (and the resulting
                # correction factor stays positive). Depths without a
                # P pick keep the literal White reading -- documented
                # in the docstring.
                rho_vp2 = rho_arr / (s_p_arr * s_p_arr)
                rho_f_vf2 = rho_fluid * v_fluid * v_fluid
                factor = 1.0 - rho_f_vf2 / rho_vp2
                use_corrected = (np.isfinite(factor)
                                 & (factor > 0.0)
                                 & np.isfinite(c66_white))
                c66 = np.where(use_corrected,
                               c66_white / factor,
                               c66_white)
            else:
                c66 = c66_white

            gamma = (c66 - c44) / (2.0 * c44)
            vp  = np.sqrt(c33 / rho_arr)
            vsv = np.sqrt(c44 / rho_arr)
            vsh = np.sqrt(c66 / rho_arr)

        # Replace any +/- inf or other non-finite values with NaN so
        # the null_value substitution downstream catches them.
        for arr in (c33, c44, c66, gamma, vp, vsv, vsh):
            np.copyto(arr, nan, where=~np.isfinite(arr))

        curves["C33"] = c33
        curves["C44"] = c44
        curves["C66"] = c66
        curves["GAMMA"] = gamma
        curves["VP"]  = vp
        curves["VSV"] = vsv
        curves["VSH"] = vsh

    if not (isinstance(null_value, float) and np.isnan(null_value)):
        # Caller wants a numeric sentinel instead of NaN; remap.
        for name, arr in curves.items():
            curves[name] = np.where(np.isnan(arr), null_value, arr)

    return depths, curves
