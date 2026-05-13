"""
Public dataclasses, the :data:`SelectionRule` literal, and
:data:`DEFAULT_PRIORS` for the picker submodules.

Kept in a single small module so the other submodules
(:mod:`fwap.picker.greedy`, :mod:`fwap.picker.viterbi`,
:mod:`fwap.picker.posterior`) can all import the shared contract
without circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from fwap._common import US_PER_FT

SelectionRule = Literal["max_coherence", "scored"]


# Per-mode prior windows used by :func:`fwap.picker.greedy.pick_modes`.
# Slowness bounds are stored in **seconds per metre** (the unit used
# everywhere in the package); ``40.0 * US_PER_FT`` is simply the
# convenient way to write "40 microseconds per foot" at declaration
# time. ``coherence_min`` is unitless; ``order`` is the processing order
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
    "P": dict(
        slow_min=40.0 * US_PER_FT,
        slow_max=140.0 * US_PER_FT,
        coherence_min=0.5,
        order=0,
    ),
    "S": dict(
        slow_min=80.0 * US_PER_FT,
        slow_max=260.0 * US_PER_FT,
        coherence_min=0.4,
        order=1,
    ),
    "PseudoRayleigh": dict(
        slow_min=130.0 * US_PER_FT,
        slow_max=200.0 * US_PER_FT,
        coherence_min=0.4,
        order=2,
    ),
    # Stoneley starts at the borehole-fluid slowness floor (~200 us/ft
    # for a typical mud); below that you are in the pseudo-Rayleigh /
    # guided regime, not Stoneley. Keeping the windows non-overlapping
    # is what allows the four-mode picker to pick Stoneley correctly
    # in a gather that also carries a pseudo-Rayleigh peak.
    "Stoneley": dict(
        slow_min=200.0 * US_PER_FT,
        slow_max=360.0 * US_PER_FT,
        coherence_min=0.4,
        order=3,
    ),
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
        amp = f", amp={self.amplitude:.3g}" if self.amplitude is not None else ""
        return (
            f"ModePick({self.name!r}, "
            f"slowness={self.slowness / US_PER_FT:.2f} us/ft, "
            f"t={self.time * 1e3:.2f} ms, "
            f"coh={self.coherence:.3f}{amp})"
        )


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
