"""
Bit-exact characterisation regression for ``fwap.cylindrical_solver``.

This test pins the numerical output of every public dispersion
function on a small canonical fixture grid (slow & fast formation,
with and without one annular layer, n=0/1/2, isotropic and VTI).
The expected outputs live in ``tests/data/cylindrical_solver_golden.npz``
and were captured from the pre-refactor monolithic
``fwap/cylindrical_solver.py``.

The test is the bit-exactness gate for the cylindrical_solver
refactoring sequence laid out in the plan that introduced this
file (Phase 0). Phase 1 splits the 14 kLoC monolith into a
package; Phases 2-4 unify duplicated row-builder, E-matrix /
propagator, and modal-determinant families. None of those phases
may change the values pinned here.

To regenerate the golden file (do this only when the change is
genuinely physics-modifying and intentional, e.g. a bug fix in a
dispersion law)::

    python tests/test_cylindrical_solver_characterisation.py --regenerate

and commit the updated ``cylindrical_solver_golden.npz``.
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

_FREQ_BOUND = np.linspace(2_000.0, 10_000.0, 5)
_FREQ_LEAKY = np.linspace(8_000.0, 16_000.0, 5)

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


@pytest.mark.parametrize("label", list(_CASES))
def test_dispersion_matches_golden(label: str, golden: dict[str, np.ndarray]) -> None:
    """Each public dispersion function reproduces its golden output exactly.

    Bit-exact comparison via ``np.array_equal`` (NaN-aware), to make
    this the strictest possible regression gate for the upcoming
    refactoring phases. NaN entries (frequencies where the bracket
    failed) are intentionally part of the contract -- they encode
    the cutoff structure of each mode.
    """
    expected_slow = golden[f"{label}__slowness"]
    expected_atten = golden.get(f"{label}__atten")

    mode = _CASES[label]()
    actual_slow = np.asarray(mode.slowness, dtype=float)

    assert actual_slow.shape == expected_slow.shape, (
        f"{label}: slowness shape changed "
        f"({actual_slow.shape} vs {expected_slow.shape})"
    )
    assert np.array_equal(actual_slow, expected_slow, equal_nan=True), (
        f"{label}: slowness array drifted from golden values; "
        "either the refactor changed numerical behaviour (forbidden) "
        "or the golden file needs an intentional regeneration."
    )

    if expected_atten is None:
        assert mode.attenuation_per_meter is None, (
            f"{label}: gained an attenuation array that was None in the golden file"
        )
    else:
        assert mode.attenuation_per_meter is not None, (
            f"{label}: lost an attenuation array that was present in the golden file"
        )
        actual_atten = np.asarray(mode.attenuation_per_meter, dtype=float)
        assert np.array_equal(actual_atten, expected_atten, equal_nan=True), (
            f"{label}: attenuation array drifted from golden values"
        )


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
