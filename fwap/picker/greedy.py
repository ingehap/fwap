"""
Greedy rule-based picker: per-depth :func:`pick_modes` plus the
depth-tracking variant :func:`track_modes`.

Both are the directly-from-the-book "AI" picker: a deterministic
walk over the candidate peaks returned by
:func:`fwap.coherence.find_peaks` constrained by per-mode slowness
windows and a coherence floor. :func:`track_modes` adds a
continuity regulariser across depths; for harder problems use the
Viterbi pickers in :mod:`fwap.picker.viterbi` instead.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from fwap._common import US_PER_FT
from fwap.coherence import STCResult, find_peaks
from fwap.picker._types import (
    DEFAULT_PRIORS,
    DepthPicks,
    ModePick,
    SelectionRule,
)

_VALID_SELECTION_RULES = frozenset({"max_coherence", "scored"})


def _best_candidate(
    candidates: np.ndarray,
    prior: dict[str, float],
    *,
    t_earliest: float = 0.0,
    selection_rule: SelectionRule = "scored",
    time_penalty: float = 0.1,
    time_scale: float = 1.0e-3,
) -> np.ndarray | None:
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
    mask = (
        (candidates[:, 0] >= prior["slow_min"])
        & (candidates[:, 0] <= prior["slow_max"])
        & (candidates[:, 2] >= prior["coherence_min"])
    )
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


def pick_modes(
    stc_result: STCResult,
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
            valid,
            prior,
            t_earliest=t_earliest,
            selection_rule=selection_rule,
            time_penalty=time_penalty,
            time_scale=time_scale,
        )
        if winner is None:
            continue
        out[name] = ModePick(
            name=name,
            slowness=float(winner[0]),
            time=float(winner[1]),
            coherence=float(winner[2]),
            amplitude=float(winner[3]) if winner.size >= 4 else None,
        )
        t_earliest = max(t_earliest, float(winner[1]))
    return out


def track_modes(
    stc_results: Sequence[STCResult],
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

    .. warning::

       **This picker confuses P with a more coherent shear arrival**, and on
       real monopole data that is common rather than exotic. Prefer
       :func:`viterbi_pick_joint` when compressional slowness matters.

       Mode ordering here is enforced on arrival *time*, never on slowness, so
       nothing requires P to be faster than S; and the ``P`` prior window
       (40-140 us/ft) contains the shear arrival of most formations. When shear
       is the more coherent of the two, the ``scored`` rule's ``time_penalty``
       is too small to overcome the coherence difference and both modes select
       the same peak.

       Measured against a Schlumberger DSI log over 400 depths: this function
       reported the shear slowness as compressional at 143 of them, agreeing
       with the vendor's own compressional pick on 62 % of depths.
       :func:`viterbi_pick_joint`, on identical STC surfaces and in the same
       runtime, confused 34 and agreed on 89 %. Shear was unaffected either way
       (96 %). The greedy failure is inherent to greedy selection rather than a
       tuning error: the ``time_penalty`` that would flip those depths has a
       median of 0.18 but a 90th percentile of 0.43, against a default of 0.1,
       and raising it that far would bias every late mode.

       ``tests/test_picker.py`` reproduces both behaviours on a seeded
       synthetic, and :func:`quality_control_picks` flags the resulting picks
       (it checks the same shear-slower-than-compressional invariant this
       violates).

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
            continuity_max_gap = 5.0 * float(np.median(np.abs(np.diff(depths))))
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
                    effective_tol = max_slow_jump * (1.0 + continuity_tol_growth * gap)
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
                valid,
                prior,
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
                valid_fb = peaks[peaks[:, 1] >= t_earliest] if peaks.size else peaks
                if valid_fb.size:
                    winner = _best_candidate(
                        valid_fb,
                        prior,
                        t_earliest=t_earliest,
                        selection_rule=selection_rule,
                        time_penalty=time_penalty,
                        time_scale=time_scale,
                    )
                if winner is None:
                    continue

            pick = ModePick(
                name=name,
                slowness=float(winner[0]),
                time=float(winner[1]),
                coherence=float(winner[2]),
                amplitude=float(winner[3]) if winner.size >= 4 else None,
            )
            dp.picks[name] = pick
            last[name] = (pick.slowness, float(depth))
            t_earliest = max(t_earliest, pick.time)
        all_picks.append(dp)
    return all_picks
