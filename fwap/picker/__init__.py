"""
Rule-based and Viterbi P / S / pseudo-Rayleigh / Stoneley pickers on
STC coherence surfaces.

Implements the knowledge-based picker described in Part 1 of the
book (an "AI approach for the picking of waves on full-waveform
acoustic data"): expert rules on slowness range, coherence
threshold, mode ordering and depth-to-depth continuity are codified
into a deterministic selection over the candidate peaks returned by
:func:`fwap.coherence.find_peaks`. Two further book-listed expert
rules -- **wavelet shape** and **onset polarity** -- are exposed as
post-pick filters via :func:`filter_picks_by_shape` (single-depth)
and :func:`filter_track_by_shape` (multi-depth track) so callers can
opt in without altering the core slowness / coherence / continuity
selection.

Default priors cover the four monopole arrivals enumerated by Mari
et al. -- the P head-wave, the S head-wave, the pseudo-Rayleigh /
guided trapped mode (when present in fast formations), and the
low-frequency Stoneley tube wave.

Subpackage layout
-----------------
* :mod:`fwap.picker._types`    -- dataclasses (:class:`ModePick`,
                                  :class:`DepthPicks`), the
                                  :data:`SelectionRule` literal, and
                                  :data:`DEFAULT_PRIORS`.
* :mod:`fwap.picker.greedy`    -- :func:`pick_modes`,
                                  :func:`track_modes` (per-depth
                                  rule-based pickers).
* :mod:`fwap.picker.viterbi`   -- :func:`viterbi_pick` (per-mode)
                                  and :func:`viterbi_pick_joint`
                                  (joint N-mode over the depth
                                  trellis).
* :mod:`fwap.picker.posterior` -- :func:`viterbi_posterior_marginals`
                                  (MAP path plus per-candidate
                                  posterior probabilities via
                                  forward-backward).
* :mod:`fwap.picker.shape`     -- :func:`wavelet_shape_score`,
                                  :func:`onset_polarity` and the
                                  pick-filter wrappers
                                  :func:`filter_picks_by_shape` /
                                  :func:`filter_track_by_shape`.
* :mod:`fwap.picker.quality`   -- :class:`PickQualityFlags`,
                                  :func:`quality_control_picks`,
                                  :func:`quality_control_track`,
                                  and the picker-to-log-curve bridge
                                  :func:`track_to_log_curves`.

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
  Acquisition of Logging Data.* Elsevier (for the slowness windows
  used in :data:`DEFAULT_PRIORS`).
"""

from __future__ import annotations

from fwap.picker._types import (
    DEFAULT_PRIORS,
    DepthPicks,
    ModePick,
    SelectionRule,
)
from fwap.picker.greedy import pick_modes, track_modes
from fwap.picker.posterior import PosteriorPick, viterbi_posterior_marginals
from fwap.picker.quality import (
    PickQualityFlags,
    quality_control_picks,
    quality_control_track,
    track_to_log_curves,
)
from fwap.picker.shape import (
    filter_picks_by_shape,
    filter_track_by_shape,
    onset_polarity,
    wavelet_shape_score,
)
from fwap.picker.viterbi import (
    _auto_fallback_k,
    viterbi_pick,
    viterbi_pick_joint,
)

__all__ = [
    "DEFAULT_PRIORS",
    "DepthPicks",
    "ModePick",
    "PickQualityFlags",
    "PosteriorPick",
    "SelectionRule",
    "_auto_fallback_k",
    "filter_picks_by_shape",
    "filter_track_by_shape",
    "onset_polarity",
    "pick_modes",
    "quality_control_picks",
    "quality_control_track",
    "track_modes",
    "track_to_log_curves",
    "viterbi_pick",
    "viterbi_pick_joint",
    "viterbi_posterior_marginals",
    "wavelet_shape_score",
]
