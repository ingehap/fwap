"""
Numerical-characterisation regression for ``fwap.cylindrical_solver``.

This test pins the numerical output of every public dispersion
function on a small canonical fixture grid (slow & fast formation,
with and without one annular layer, n=0/1/2, isotropic and VTI).
The expected outputs live in ``tests/data/cylindrical_solver_golden.npz``
and were captured from the pre-refactor monolithic
``fwap/cylindrical_solver.py``.

The test is the near-bit-exact regression gate for the
cylindrical_solver refactoring sequence laid out in the plan that
introduced this file (Phase 0). Phase 1 splits the 14 kLoC monolith
into a package; Phases 2-4 unify duplicated row-builder, E-matrix /
propagator, and modal-determinant families. None of those phases
may change the values pinned here beyond a few ULPs.

Comparison is ``rtol=1e-12`` with the NaN mask checked exactly. The
slack (vs. bit-identity) absorbs last-ULP round-off differences
between SciPy releases inside the declared dependency range -- the
root-finders (``brentq`` / Mueller) and Bessel routines in
``scipy.special`` are not guaranteed to be bit-stable across minor
versions. A real refactor regression would be orders of magnitude
above this tolerance.

To regenerate the golden file (do this only when the change is
genuinely physics-modifying and intentional, e.g. a bug fix in a
dispersion law)::

    python tests/test_cylindrical_solver_characterisation.py --regenerate

and commit the updated ``cylindrical_solver_golden.npz``.

Regenerations to date
---------------------
* ``n0_pseudo_rayleigh_fast`` (both arrays), when
  :func:`pseudo_rayleigh_dispersion` moved from a heuristic seed to an
  enumerated one. The golden had pinned **all-NaN** over 8-16 kHz --
  it was recording the old seeding's failure to find the mode, not an
  absence of one. The replacement values were checked before being
  committed: each is a root of the modal determinant to ~1e-14
  relative to the determinant's magnitude in its own neighbourhood,
  all five lie inside the leaky-S window, an independent 4-40 kHz grid
  at 50 Hz spacing reproduces them exactly, and the independent ray
  estimate in :func:`fwap.leaky_radiation_attenuation` brackets the
  attenuations. No other array in the file changed.
* ``n0_pseudo_rayleigh_fast`` (both arrays) and
  ``n2_quadrupole_layered_slow``, when :func:`_k_or_hankel`'s leaky
  branch was corrected from a mixture that was two thirds *incoming*
  to the pure outgoing Hankel wave. ``n0_leaky_compressional_slow``
  was added in the same change.

  ``_FREQ_LEAKY`` moved with it, 8-16 kHz down to 5.5-7.2 kHz. That is
  not a tolerance being relaxed: the fast-formation leaky branch lives
  just under its own trapped cut-off, 7899 Hz on ``_FAST``, and above
  that the mode is trapped rather than leaky. Leaving the grid where
  it was would have pinned all-NaN again, which is precisely the
  failure the first entry below records.

  Checked before being committed, following the precedent below. Each
  ``n0_pseudo_rayleigh_fast`` value is a root of the modal determinant
  to **2e-14 - 2e-13** relative to the determinant 0.1 % away, all five
  lie inside the leaky-S window with ``Im(k_z) > 0``, and an
  independent 50 Hz grid reproduces them to 8e-5. Every
  ``n0_leaky_compressional_slow`` value is a root to **2e-14 - 4e-13**,
  a 25 Hz grid reproduces them to **2e-16**, and -- the check the
  others could not have -- all five sit within **0.04 %** of Sinha &
  Asvadurov (2004) fig 11(a). ``n2_quadrupole_layered_slow`` gained
  its 2 kHz sample (993.99 us/m) and no existing value moved.
* ``n1_flexural_fast`` and ``n2_quadrupole_fast``, when the
  fast-formation search window was corrected from ``(V_R, V_S)`` to
  ``(V_f, V_S)`` with fundamental selection (roadmap A.2). The old
  golden pinned the defect: a single finite value at 10 kHz, 2591.9 m/s
  at n=1 -- essentially ``V_S``, i.e. a trapped mode against the ceiling
  -- and 2390.4 at n=2, which is ``V_R`` to four figures, i.e. the
  bracket edge. Both are outside the fundamental's range over this band.

  The n=1 replacements (1873.2 / 1650.4 / 1573.1 m/s at 6 / 8 / 10 kHz)
  were checked before being committed, following the precedent above:
  each is a root of the modal determinant to **1e-11 - 1e-14** relative
  to the determinant 0.1 % away, all lie strictly inside ``(V_f, V_S)``
  and **below** ``V_R`` where the old window could not reach, the
  sequence descends monotonically as a guided mode must, and grids of 5
  to 161 points over the same band reproduce the 10 kHz value to
  **0.000 %**.

  The n=2 replacement (2084.9 m/s at 10 kHz) is on the fundamental and
  below ``V_R``, but is **not** of that quality and the value pinned
  here should be read as characterisation rather than a reference: it
  moves by **1.7 %** across grid densities and **3.2 %** across grid
  start points, and vanishes entirely on some. That instability is the
  n=2 root-finding, not the bracket -- it is the same effect figure 6
  recorded, where two grids differing by last-bit rounding gave 47 and
  42 converged points of 71 -- and correcting A.2 did not remove it.
  ``test_the_fast_formation_marcher_is_grid_independent_at_n1`` states
  the contrast as an assertion.
* Every ``n >= 1`` array -- ``n1_flexural_slow``, ``n1_flexural_fast``,
  ``n1_flexural_layered_slow``, ``n1_flexural_vti``,
  ``n2_quadrupole_slow``, ``n2_quadrupole_fast`` and
  ``n2_quadrupole_layered_slow`` -- when the SV column was corrected
  from an azimuthal-only vector potential to the Hansen form
  ``curl curl(chi z)``, which is Schmitt & Cheng's appendix column
  (roadmap A.8). The ``n = 0`` arrays are untouched: the old ansatz is
  a solution there, which is why ``stoneley_dispersion`` never carried
  the error.

  Each replacement was checked before being committed, following the
  precedent above. Every finite value is a root of its own modal
  determinant to **5e-14 - 2e-9** relative to the determinant 0.1 %
  away; every value lies inside its physical window; every sequence
  descends monotonically as a guided mode must; and an independent
  50 Hz grid over 2-10 kHz reproduces the fast-formation n=1 values
  exactly. Coverage went up in six of the seven arrays -- the corrected
  solvers resolve about a kHz lower than before -- so several entries
  that used to be NaN now carry values.

  ``n2_quadrupole_fast`` was the exception, and roadmap A.7 removed it
  -- see the next entry.
* ``n2_quadrupole_fast`` again, when roadmap A.7 corrected which part
  of the determinant the fast-formation marcher tracks. The ``n = 2``
  determinant at real ``k_z`` is REAL, not imaginary -- the parity of
  its fixed phase flips with azimuthal order -- so the marcher had been
  seeding off sign changes in round-off. The old single value
  (2082.2 m/s at 10 kHz) was one of those.

  The replacement is two values, 2305.2 m/s at 8 kHz and 1916.7 at
  10 kHz, and unlike its predecessors this array is now reference
  quality rather than characterisation: both are sharp zeros of
  ``Re(det)`` (2.6e-11 and 1.8e-11 relative to the determinant 0.1 %
  away), both lie strictly inside ``(V_f, V_S)``, the pair descends,
  and an independent 50 Hz grid over 2-10 kHz reproduces both exactly.
  Coverage went up because the mode's onset moved from 8.3 kHz to
  6.4 against a published 6.29.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np
import pytest

from fwap.cylindrical_solver import (
    BoreholeLayer,
    BoreholeMode,
    flexural_dispersion,
    flexural_dispersion_layered,
    flexural_dispersion_vti,
    leaky_compressional_dispersion,
    pseudo_rayleigh_dispersion,
    quadrupole_dispersion,
    quadrupole_dispersion_layered,
    stoneley_dispersion,
    stoneley_dispersion_layered,
    stoneley_dispersion_vti,
)

GOLDEN_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "cylindrical_solver_golden.npz",
)


# ---------------------------------------------------------------------
# Canonical fixture grid
# ---------------------------------------------------------------------
#
# The grids are deliberately small (5 points each) so the golden
# file stays under a few kB. The slow/fast formation parameters are
# the same numbers used by the existing non-regression tests in
# ``tests/test_cylindrical_solver.py`` so the characterisation
# values are physically meaningful.

# Slow formation: a generic shale (V_S < V_f).
_SLOW = dict(vp=2300.0, vs=1000.0, rho=2200.0, vf=1500.0, rho_f=1000.0, a=0.10)
# Fast formation: a generic carbonate (V_S > V_f).
_FAST = dict(vp=4500.0, vs=2600.0, rho=2400.0, vf=1500.0, rho_f=1000.0, a=0.10)
# Slow formation in the strict sense V_S < V_f < V_P, which the leaky
# compressional mode needs and which ``_SLOW`` above does not satisfy
# (its V_S is 1000 m/s but its V_P, 2300, is fine -- it is the same
# ordering; kept separate so the two fixtures stay independent).
# Sinha & Asvadurov (2004) Table 1 slow formation (B), the geometry the
# solver is scored on.
_SLOW_LEAKY_P = dict(vp=1890.0, vs=508.0, rho=2054.0, vf=1500.0, rho_f=1000.0, a=0.1016)
_FREQ_LEAKY_P = np.linspace(4_000.0, 12_000.0, 5)

_FREQ_BOUND = np.linspace(2_000.0, 10_000.0, 5)
# The n=0 leaky branch of radial order 0 lives just under that branch's
# trapped cut-off -- 7899 Hz on ``_FAST`` -- so this grid sits at
# 5500-7200 Hz. It was 8000-16000 Hz, which is above the cut-off, where
# the mode is trapped and ``pseudo_rayleigh_dispersion`` returns NaN by
# design. See that function's "Branch selection" section.
_FREQ_LEAKY = np.linspace(5_500.0, 7_200.0, 5)

# Single annular layer (mudcake) for the layered fixtures.
# Slow-formation flexural / quadrupole layered paths require the
# per-layer constraint ``layer.vs >= formation.vs`` (see
# ``_validate_flexural_layers_stacked``); we use a stiffer layer.
_MUDCAKE = (BoreholeLayer(vp=2800.0, vs=1200.0, rho=2300.0, thickness=0.005),)

# VTI stiffness tensor (Pa) for a transversely-isotropic shale,
# slow-formation regime (V_Sv = sqrt(C44/rho) ~ 1230 m/s < V_f).
# Numbers are the same canonical TI fixture used by the existing
# VTI tests.
_VTI = dict(
    c11=1.50e10,
    c13=0.45e10,
    c33=1.20e10,
    c44=0.36e10,
    c66=0.50e10,
    rho=2400.0,
    vf=1500.0,
    rho_f=1000.0,
    a=0.10,
)


def _call_iso(
    fn: Callable[..., BoreholeMode],
    params: dict,
    freq: np.ndarray,
    *,
    layers: tuple[BoreholeLayer, ...] | None = None,
) -> BoreholeMode:
    """Invoke an isotropic-formation dispersion function."""
    if layers is None:
        return fn(freq, **params)
    return fn(freq, **params, layers=layers)


def _call_vti(
    fn: Callable[..., BoreholeMode],
    freq: np.ndarray,
) -> BoreholeMode:
    """Invoke a VTI dispersion function."""
    return fn(freq, **_VTI)


# Each entry: ``label -> callable producing a BoreholeMode``. The
# label is the NPZ key prefix; ``label__slowness`` and (when
# present) ``label__atten`` keys hold the arrays.
_CASES: dict[str, Callable[[], BoreholeMode]] = {
    # --- n=0 isotropic -------------------------------------------
    "n0_stoneley_slow": lambda: _call_iso(stoneley_dispersion, _SLOW, _FREQ_BOUND),
    "n0_stoneley_fast": lambda: _call_iso(stoneley_dispersion, _FAST, _FREQ_BOUND),
    "n0_pseudo_rayleigh_fast": lambda: _call_iso(
        pseudo_rayleigh_dispersion, _FAST, _FREQ_LEAKY
    ),
    "n0_leaky_compressional_slow": lambda: _call_iso(
        leaky_compressional_dispersion, _SLOW_LEAKY_P, _FREQ_LEAKY_P
    ),
    # --- n=1 isotropic -------------------------------------------
    "n1_flexural_slow": lambda: _call_iso(flexural_dispersion, _SLOW, _FREQ_BOUND),
    "n1_flexural_fast": lambda: _call_iso(flexural_dispersion, _FAST, _FREQ_BOUND),
    # --- n=2 isotropic -------------------------------------------
    "n2_quadrupole_slow": lambda: _call_iso(quadrupole_dispersion, _SLOW, _FREQ_BOUND),
    "n2_quadrupole_fast": lambda: _call_iso(quadrupole_dispersion, _FAST, _FREQ_BOUND),
    # --- one-annulus layered (slow formation only) ---------------
    "n0_stoneley_layered_slow": lambda: _call_iso(
        stoneley_dispersion_layered, _SLOW, _FREQ_BOUND, layers=_MUDCAKE
    ),
    "n1_flexural_layered_slow": lambda: _call_iso(
        flexural_dispersion_layered, _SLOW, _FREQ_BOUND, layers=_MUDCAKE
    ),
    "n2_quadrupole_layered_slow": lambda: _call_iso(
        quadrupole_dispersion_layered, _SLOW, _FREQ_BOUND, layers=_MUDCAKE
    ),
    # --- VTI -----------------------------------------------------
    "n0_stoneley_vti": lambda: _call_vti(stoneley_dispersion_vti, _FREQ_BOUND),
    "n1_flexural_vti": lambda: _call_vti(flexural_dispersion_vti, _FREQ_BOUND),
}


def _produce() -> dict[str, np.ndarray]:
    """Run every case and collect the per-case slowness / atten arrays."""
    out: dict[str, np.ndarray] = {}
    for label, factory in _CASES.items():
        mode = factory()
        out[f"{label}__slowness"] = np.asarray(mode.slowness, dtype=float)
        if mode.attenuation_per_meter is not None:
            out[f"{label}__atten"] = np.asarray(mode.attenuation_per_meter, dtype=float)
    return out


def _regenerate(path: str = GOLDEN_PATH) -> None:
    """Regenerate the golden NPZ file. Called by ``__main__``."""
    arrays = _produce()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **arrays)


@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    """Load the committed golden arrays."""
    if not os.path.exists(GOLDEN_PATH):
        pytest.fail(
            f"Golden file missing: {GOLDEN_PATH}. Regenerate with "
            "`python tests/test_cylindrical_solver_characterisation.py "
            "--regenerate`."
        )
    with np.load(GOLDEN_PATH) as data:
        return {key: data[key].copy() for key in data.files}


# Refactor regressions show up orders of magnitude above this; the
# slack only absorbs last-ULP differences between SciPy releases.
_GOLDEN_RTOL = 1.0e-12


def _assert_matches_golden(
    label: str, actual: np.ndarray, expected: np.ndarray
) -> None:
    """Compare an array against its golden, preserving the NaN mask exactly."""
    assert actual.shape == expected.shape, (
        f"{label}: shape changed ({actual.shape} vs {expected.shape})"
    )
    nan_actual = np.isnan(actual)
    nan_expected = np.isnan(expected)
    assert np.array_equal(nan_actual, nan_expected), (
        f"{label}: NaN mask changed (cutoff structure of the mode drifted)"
    )
    finite = ~nan_expected
    assert np.allclose(actual[finite], expected[finite], rtol=_GOLDEN_RTOL, atol=0.0), (
        f"{label}: array drifted beyond rtol={_GOLDEN_RTOL:.0e}; "
        "either the refactor changed numerical behaviour (forbidden) "
        "or the golden file needs an intentional regeneration."
    )


@pytest.mark.parametrize("label", list(_CASES))
def test_dispersion_matches_golden(label: str, golden: dict[str, np.ndarray]) -> None:
    """Each public dispersion function reproduces its golden output.

    NaN entries (frequencies where the bracket failed) are part of the
    contract -- they encode the cutoff structure of each mode -- so the
    NaN mask is compared exactly. Finite entries are compared with
    ``rtol=1e-12`` to absorb last-ULP round-off variation between SciPy
    releases inside the declared dependency range.
    """
    expected_slow = golden[f"{label}__slowness"]
    expected_atten = golden.get(f"{label}__atten")

    mode = _CASES[label]()
    actual_slow = np.asarray(mode.slowness, dtype=float)
    _assert_matches_golden(f"{label} slowness", actual_slow, expected_slow)

    if expected_atten is None:
        assert mode.attenuation_per_meter is None, (
            f"{label}: gained an attenuation array that was None in the golden file"
        )
    else:
        assert mode.attenuation_per_meter is not None, (
            f"{label}: lost an attenuation array that was present in the golden file"
        )
        actual_atten = np.asarray(mode.attenuation_per_meter, dtype=float)
        _assert_matches_golden(f"{label} attenuation", actual_atten, expected_atten)


def test_golden_covers_all_public_dispersion_functions() -> None:
    """Guard against the fixture grid silently shrinking.

    If a new public dispersion function is added to the package
    after Phase 0, this list should grow to cover it. The
    explicit tally makes that requirement visible in code review
    rather than buried in a parametric ID list.
    """
    expected_labels = {
        "n0_stoneley_slow",
        "n0_stoneley_fast",
        "n0_pseudo_rayleigh_fast",
        "n0_leaky_compressional_slow",
        "n1_flexural_slow",
        "n1_flexural_fast",
        "n2_quadrupole_slow",
        "n2_quadrupole_fast",
        "n0_stoneley_layered_slow",
        "n1_flexural_layered_slow",
        "n2_quadrupole_layered_slow",
        "n0_stoneley_vti",
        "n1_flexural_vti",
    }
    assert set(_CASES) == expected_labels


if __name__ == "__main__":
    import sys

    if "--regenerate" in sys.argv:
        _regenerate()
        print(f"Wrote {GOLDEN_PATH}")
    else:
        print(
            "Usage: python tests/test_cylindrical_solver_characterisation.py "
            "--regenerate"
        )
