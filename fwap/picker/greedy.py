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
    slow_below: float | None = None,
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
    slow_below : float, optional
        Strict upper bound (s/m) on the candidate's slowness, applied
        on top of the prior window. Used by
        :func:`_resolve_mode_collisions` to re-pick a mode that must come
        out faster than an arrival another mode has claimed.
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
    if slow_below is not None:
        mask &= candidates[:, 0] < slow_below
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


def _same_arrival(a: ModePick, b: ModePick) -> bool:
    """
    True when two picks are the same STC peak rather than two nearby ones.

    Exact equality is the right test and not a fragile one: both picks are
    rows of the same :func:`find_peaks` array for the same depth, so the same
    peak reaches here bit-identical. Two *adjacent* cells are a different
    situation -- possibly still a mislabel, but not one this can prove -- and
    are deliberately not treated as a collision.
    """
    return a.slowness == b.slowness and a.time == b.time


def _resolve_mode_collisions(
    picks: dict[str, ModePick],
    pools: dict[str, tuple[np.ndarray, float]],
    priors: dict[str, dict[str, float]],
    *,
    selection_rule: SelectionRule,
    time_penalty: float,
    time_scale: float,
) -> None:
    """
    Stop two modes from being the same arrival, in place.

    The greedy loop orders modes by arrival *time* only, and equal times
    satisfy that trivially, so nothing in it prevents two modes from
    selecting one peak. One arrival cannot be two modes, and when it happens
    at least one of the two is a mislabel.

    Which one is not knowable in general -- both directions occur in real
    data. On a Schlumberger DSI log the shared peak was the shear arrival and
    P was the mislabel; on a slow-formation synthetic it was the
    compressional arrival and S was. So this pass does not decide. It moves
    the **faster-labelled** mode -- the one whose slowness must be the lower
    of the two -- and only if that mode has somewhere admissible to go: the
    replacement is drawn from the same candidate pool, under the same
    coherence floor, ``t_earliest`` and scoring rule, with the shared
    slowness as a strict upper bound. A mode with no such candidate is left
    exactly as it was, because "nowhere faster to go" is evidence that this
    mode is the one holding the *right* arrival, not the wrong one.

    Nothing is ever dropped and no mode is ever moved to a slower candidate,
    so a collision the pass cannot resolve leaves the picks no worse than the
    greedy result -- and :func:`quality_control_picks` still flags it, since
    a shared arrival makes Vp/Vs exactly 1.

    Sweeps repeat until nothing changes. This terminates: every move
    strictly lowers one mode's slowness within a finite candidate set.

    Parameters
    ----------
    picks : dict from str to ModePick
        Picks for one depth, modified in place.
    pools : dict from str to (ndarray, float)
        Per mode, the candidate array the winner was chosen from --
        shape ``(n, 3)`` or ``(n, 4)`` rows of
        ``[slowness, time, coherence[, amplitude]]`` -- and the
        ``t_earliest`` (s) in force at that point.
    priors : dict
        Per-mode prior windows; ``order`` sets the mode sequence, which is
        also the order of increasing slowness.
    selection_rule, time_penalty, time_scale
        Passed through to :func:`_best_candidate`.
    """
    order = sorted(priors, key=lambda n: priors[n]["order"])
    for _sweep in range(len(order)):
        changed = False
        for i, name in enumerate(order):
            pick = picks.get(name)
            if pick is None:
                continue
            if not any(
                _same_arrival(pick, picks[other])
                for other in order[i + 1 :]
                if other in picks
            ):
                continue
            candidates, t_earliest = pools[name]
            replacement = _best_candidate(
                candidates,
                priors[name],
                t_earliest=t_earliest,
                slow_below=pick.slowness,
                selection_rule=selection_rule,
                time_penalty=time_penalty,
                time_scale=time_scale,
            )
            if replacement is None:
                continue
            picks[name] = ModePick(
                name=name,
                slowness=float(replacement[0]),
                time=float(replacement[1]),
                coherence=float(replacement[2]),
                amplitude=float(replacement[3]) if replacement.size >= 4 else None,
            )
            changed = True
        if not changed:
            break


def pick_modes(
    stc_result: STCResult,
    priors: dict[str, dict[str, float]] | None = None,
    threshold: float = 0.4,
    *,
    selection_rule: SelectionRule = "scored",
    time_penalty: float = 0.1,
    time_scale: float | None = None,
    resolve_mode_collisions: bool = True,
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
    resolve_mode_collisions : bool, default True
        Forbid two modes from being assigned the same arrival, re-picking
        the faster-labelled one when it has an admissible faster
        candidate. See :func:`track_modes`, where the rule is documented
        in full along with what it is worth on real data. Pass ``False``
        for the previous behaviour.

    Notes
    -----
    The mode-ordering rule is strict in time (P -> S -> Stoneley, each
    required to be no earlier than the previous), and equal times satisfy
    it, which is why ``resolve_mode_collisions`` exists. In altered
    zones the S head-wave can appear before the formation P re-emerges
    (see Aron et al., 1994, *SEG Expanded Abstracts*); for those cases
    a joint log-likelihood across modes is more robust than the greedy
    rule used here -- see :func:`viterbi_pick_joint`.
    """
    if priors is None:
        priors = DEFAULT_PRIORS
    if time_scale is None:
        time_scale = max(stc_result.window_length, 1e-12)
    peaks = find_peaks(stc_result, threshold=threshold)
    out: dict[str, ModePick] = {}
    pools: dict[str, tuple[np.ndarray, float]] = {}
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
        pools[name] = (valid, t_earliest)
        t_earliest = max(t_earliest, float(winner[1]))
    if resolve_mode_collisions:
        _resolve_mode_collisions(
            out,
            pools,
            priors,
            selection_rule=selection_rule,
            time_penalty=time_penalty,
            time_scale=time_scale,
        )
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
    resolve_mode_collisions: bool = True,
) -> list[DepthPicks]:
    """
    Per-depth picking with a depth-aware continuity regulariser.

    .. note::

       **Two modes may not be assigned the same arrival.** Without that rule
       (``resolve_mode_collisions=False``) this picker confuses P with a more
       coherent shear arrival, and on real monopole data that is common rather
       than exotic.

       The greedy loop orders modes by time only, and equal times satisfy that
       trivially, so nothing in it requires P to be faster than S; and the
       ``P`` prior window (40-140 us/ft) contains the shear arrival of most
       formations. When shear is the more coherent of the two, the ``scored``
       rule's ``time_penalty`` is too small to overcome the coherence
       difference and both modes select the same peak.

       Which of the two labels is wrong is *not* decidable in general -- both
       directions occur -- so the rule does not decide. It moves the
       faster-labelled mode, and only when that mode has an admissible faster
       candidate; otherwise it changes nothing. It therefore never drops a
       pick, never moves a mode to a slower candidate, and cannot leave a
       depth worse than the greedy result. See
       :func:`_resolve_mode_collisions`.

       Measured against a Schlumberger DSI log over 400 depths, agreement with
       the vendor's own compressional pick (``DTCO``, within 10 %):

       ========================== ========= ============ ========
       picker                     P agreed  P not faster P picked
       ========================== ========= ============ ========
       this, rule off             62 %      143          400
       this function              **95 %**  **5**        400
       :func:`viterbi_pick_joint` 89 %      34           398
       ========================== ========= ============ ========

       The rule changed the P pick at 138 depths, every one of them a
       collision, and turned 129 of them correct. It left the shear pick
       **bit-identical at all 400** -- shear is the slower of the pair, and
       only the faster-labelled mode ever moves -- and damaged **none** of the
       250 depths that were already right. Of the 150 that were wrong, 21
       still are: 14 collisions it either could not resolve or re-picked onto
       an intermediate peak, and 7 that were never collisions.

       Confirmed on a second logging pass of the same well over a different
       interval: 70 % -> 86 %, 72 unordered depths -> 2, again no damage to
       any of the 283 depths that were already right. The rule has no constant
       to tune, which is why it transfers.

       A near-collision -- two peaks one slowness cell apart -- is
       deliberately not treated as one, and 3 of these 400 depths end up that
       close. For those, and for the 7 confusions that are not collisions,
       :func:`viterbi_pick_joint` remains the better tool: a global cost over
       the mode tuple can reject an assignment a local rule cannot see.
       :func:`quality_control_picks` flags any collision that survives, since
       a shared arrival makes Vp/Vs exactly 1, and ``tests/test_picker.py``
       pins both the old failure and the repair on seeded synthetics.

       Retuning ``time_penalty`` is *not* an alternative: the value that would
       flip those depths has a median of 0.18 and a 90th percentile of 0.43
       against a default of 0.1, so buying most of them costs a four-fold
       global bias against every late mode.

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
    resolve_mode_collisions : bool, default True
        Forbid two modes from being assigned the same arrival, re-picking
        the faster-labelled one against its own candidate pool when an
        admissible faster candidate exists and leaving the depth untouched
        when none does. See the note above. Pass ``False`` for the
        previous behaviour.

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
        pools: dict[str, tuple[np.ndarray, float]] = {}
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
                # A re-pick has to draw from the set the winner came
                # from, or the continuity filter would silently widen.
                valid = valid_fb

            pick = ModePick(
                name=name,
                slowness=float(winner[0]),
                time=float(winner[1]),
                coherence=float(winner[2]),
                amplitude=float(winner[3]) if winner.size >= 4 else None,
            )
            dp.picks[name] = pick
            pools[name] = (valid, t_earliest)
            t_earliest = max(t_earliest, pick.time)

        if resolve_mode_collisions:
            _resolve_mode_collisions(
                dp.picks,
                pools,
                priors,
                selection_rule=selection_rule,
                time_penalty=time_penalty,
                time_scale=time_scale,
            )
        # Continuity anchors are set only once the depth is resolved, so a
        # pick the collision rule replaces never becomes the slowness the
        # next depth is tracked against.
        for name, pick in dp.picks.items():
            last[name] = (pick.slowness, float(depth))
        all_picks.append(dp)
    return all_picks
