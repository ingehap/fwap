"""
Root-identity helpers: counting roots, not just finding them.

The cylindrical solver's characteristic failure is not a missing root
or a noisy one. It is a **real root, sharp to 1e-13, that is the wrong
one** -- the fundamental returned under a higher branch index, a
branch-point degeneracy returned as a mode, two waves swapped so the
determinant is right while its labels are not. No root-quality check
rejects any of those, because the thing genuinely is a root.

What separates them is counting. The argument principle says how many
zeros a contour encloses, independently of which one a search happens
to converge on, so it can answer "is this the root I think it is?"
rather than only "is this a root?".

These helpers live in the test tree rather than in ``fwap`` because
they are an instrument for checking the solver, not part of it.

Notes
-----
Two traps are worth naming, both hit in practice while this was being
written (roadmap A.11 phase 5).

**The instrument must be controlled before its null results mean
anything.** The first version of :func:`winding_number` unwrapped the
phase and *then* summed differences around the closed cycle, which
telescopes to exactly zero for any input whatsoever. It returned ``0``
for a box drawn around a root whose position was already known, and an
entire survey was briefly read as "no mode exists" on that basis. Any
use of this module should assert :func:`winding_number` finds a root it
is known to contain -- :func:`assert_instrument_is_sound` does that.

**A contour must not straddle a branch switch.** Where
``_detect_leaky_branches`` changes a flag the determinant becomes a
different function, so the count across the switch is meaningless: in
practice it comes back *negative*, which is impossible for a zero count
and is the tell. Contours must sit inside one branch regime.
"""

from __future__ import annotations

import numpy as np


def winding_number(
    fn,
    re_lo: float,
    re_hi: float,
    im_lo: float,
    im_hi: float,
    n: int = 240,
) -> float | None:
    """
    Zeros of ``fn`` inside a rectangle, by the argument principle.

    Parameters
    ----------
    fn : callable
        ``complex -> complex``. Must be analytic inside the rectangle;
        in particular it must not cross a branch switch, or the result
        is not a root count.
    re_lo, re_hi : float
        Real bounds of the contour (rad / m for a ``k_z`` plane).
    im_lo, im_hi : float
        Imaginary bounds. Keep ``im_lo`` off zero when the real axis
        carries the seeding behaviour.
    n : int, default 240
        Samples per side. Raise it if the count is not near-integral.

    Returns
    -------
    float or None
        The winding number, which is the enclosed zero count for an
        analytic ``fn``. ``None`` when ``fn`` was not finite and
        non-zero everywhere on the contour, since the count is then
        undefined rather than zero.

    Notes
    -----
    The loop is closed **before** unwrapping. Unwrapping first and then
    summing differences around a closed cycle telescopes to exactly
    zero for any input -- see the module docstring.
    """
    points: list[complex] = []
    for t in np.linspace(0.0, 1.0, n, endpoint=False):
        points.append(complex(re_lo + (re_hi - re_lo) * t, im_lo))
    for t in np.linspace(0.0, 1.0, n, endpoint=False):
        points.append(complex(re_hi, im_lo + (im_hi - im_lo) * t))
    for t in np.linspace(0.0, 1.0, n, endpoint=False):
        points.append(complex(re_hi + (re_lo - re_hi) * t, im_hi))
    for t in np.linspace(0.0, 1.0, n, endpoint=False):
        points.append(complex(re_lo, im_hi + (im_lo - im_hi) * t))

    values = np.array([fn(z) for z in points])
    if not np.all(np.isfinite(values)) or np.any(values == 0):
        return None
    phase = np.unwrap(np.angle(np.concatenate([values, values[:1]])))
    return float((phase[-1] - phase[0]) / (2.0 * np.pi))


def count_roots(fn, re_lo, re_hi, im_lo, im_hi, n: int = 240) -> int | None:
    """
    :func:`winding_number` rounded to an integer, or ``None``.

    Returns ``None`` when the contour is unusable *or* when the winding
    number is not near-integral -- a fractional count means the contour
    is crossing something, not that a fraction of a root is enclosed,
    and rounding it would hide exactly the condition worth seeing.
    """
    value = winding_number(fn, re_lo, re_hi, im_lo, im_hi, n=n)
    if value is None:
        return None
    nearest = round(value)
    if abs(value - nearest) > 0.05:
        return None
    return int(nearest)


def assert_instrument_is_sound(fn, root: complex, *, half_width: float) -> None:
    """
    Check the counter on a root whose position is already known.

    Call this before trusting a null result from :func:`count_roots`.
    A box around ``root`` must enclose exactly one zero, and a box of
    the same size displaced well away from it must enclose none.

    Parameters
    ----------
    fn : callable
        ``complex -> complex``, as for :func:`winding_number`.
    root : complex
        A ``k_z`` at which ``fn`` is known to vanish.
    half_width : float
        Half-width of the test box in ``Re(k_z)``; the box spans the
        same distance in ``Im(k_z)``.
    """
    around = count_roots(
        fn,
        root.real - half_width,
        root.real + half_width,
        root.imag - half_width,
        root.imag + half_width,
    )
    assert around == 1, f"counter found {around} roots around a known root"

    offset = root.real + 6.0 * half_width
    empty = count_roots(
        fn,
        offset - half_width,
        offset + half_width,
        root.imag - half_width,
        root.imag + half_width,
    )
    assert empty == 0, f"counter found {empty} roots in a box expected empty"
