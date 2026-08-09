"""
Dataclasses and per-stack validators for the cylindrical-borehole
modal-determinant solver.

Phase 1 of the cylindrical_solver refactoring plan extracted these
public dataclasses (``BoreholeLayer``, ``BoreholeMode``,
``BranchSegment``) and the layer-stack validators
(``_validate_borehole_layers`` and friends) from the 14 kLoC
monolith into this submodule. The names remain re-exported from
``fwap.cylindrical_solver`` so neither the public API nor the
private symbols imported by ``tests/test_cylindrical_solver.py``
move.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoreholeLayer:
    """
    One annular elastic layer between the borehole fluid and the
    formation half-space.

    Used by :func:`stoneley_dispersion_layered` (and the n=1
    counterpart, scheduled in plan item F of
    ``docs/plans/cylindrical_biot.md``) to describe a mudcake or
    altered zone wrapping the borehole. The layer is bounded
    radially by the borehole wall on the inside and by the
    formation half-space (or the next layer, in the multi-layer
    extension scheduled in plan item G) on the outside.

    Attributes
    ----------
    vp : float
        Compressional-wave velocity of the layer (m/s).
    vs : float
        Shear-wave velocity of the layer (m/s). Must satisfy
        ``0 < vs < vp``.
    rho : float
        Bulk density of the layer (kg/m^3). Must be positive.
    thickness : float
        Radial thickness of the layer (m). Must be positive.
    """

    vp: float
    vs: float
    rho: float
    thickness: float


@dataclass(frozen=True)
class FluidAnnulus:
    """
    A fluid-filled gap separating two elastic regions of a cased stack.

    The standard model of casing debonding in cement-bond logging: a
    microannulus between casing and cement. It is deliberately a separate
    type from :class:`BoreholeLayer` rather than a layer with ``vs = 0``,
    because a fluid gap is not a limiting case of an elastic layer in this
    solver. It differs in three ways that change the *shape* of the modal
    problem rather than its numbers -- two wave amplitudes rather than
    four, shear traction identically zero, and axial displacement free to
    slip across it -- so the elastic four-vector cannot be propagated
    through it and the stack splits into two independent blocks.

    Nor is it reachable by softening cement. An elastic layer, however
    compliant, drags the bound-mode bracket floor down with its shear
    velocity and eventually leaves no window containing the Stoneley mode;
    a fluid gap's floor is its *acoustic* velocity, which is ~1500 m/s and
    so changes nothing. Measured, a compliant-solid stand-in fails to
    converge at every thickness tried, down to 0.2 mm.

    Attributes
    ----------
    vf : float
        Acoustic velocity of the gap fluid (m/s). Must be positive. Need
        not equal the borehole fluid's velocity.
    rho : float
        Density of the gap fluid (kg/m^3). Must be positive.
    thickness : float
        Radial thickness of the gap (m). Must be positive. A debonding
        microannulus is microns to millimetres.

    See Also
    --------
    stoneley_dispersion_microannulus : The dispersion API that takes one.
    BoreholeLayer : The elastic counterpart.
    """

    vf: float
    rho: float
    thickness: float


def _validate_fluid_annulus(annulus: FluidAnnulus) -> None:
    """
    Validate a :class:`FluidAnnulus`.

    Raises ``ValueError`` if it is not a ``FluidAnnulus`` or if any of
    ``vf``, ``rho``, ``thickness`` is non-positive.
    """
    if not isinstance(annulus, FluidAnnulus):
        raise ValueError(
            f"annulus must be a FluidAnnulus instance, got {type(annulus).__name__}"
        )
    if annulus.vf <= 0 or annulus.rho <= 0:
        raise ValueError("annulus: vf and rho must be positive")
    if annulus.thickness <= 0:
        raise ValueError("annulus: thickness must be positive")


def _validate_borehole_layers(layers: tuple[BoreholeLayer, ...]) -> None:
    """
    Validate a layer stack used by the layered dispersion APIs.

    Raises ``ValueError`` if any layer has a non-positive parameter
    or violates ``vp > vs``. The empty stack ``()`` is the
    degenerate "no extra layers" case and validates trivially.
    """
    for i, layer in enumerate(layers):
        if not isinstance(layer, BoreholeLayer):
            raise ValueError(
                f"layers[{i}] must be a BoreholeLayer instance, "
                f"got {type(layer).__name__}"
            )
        if layer.vp <= 0 or layer.vs <= 0 or layer.rho <= 0:
            raise ValueError(f"layers[{i}]: vp, vs, rho must all be positive")
        if layer.vp <= layer.vs:
            raise ValueError(f"layers[{i}]: require vp > vs")
        if layer.thickness <= 0:
            raise ValueError(f"layers[{i}]: thickness must be positive")


def _validate_borehole_layers_stacked(
    layers: tuple[BoreholeLayer, ...],
    a: float,
) -> None:
    """
    Validate a stacked borehole-layer geometry for the multi-layer
    (cased-hole) dispersion APIs.

    Wraps :func:`_validate_borehole_layers` with the borehole-radius
    check ``a > 0``. The radii of the assembled stack
    (``r_0 = a; r_j = r_{j-1} + layers[j-1].thickness``) are then
    automatically positive and strictly increasing because each
    ``thickness > 0``.

    The hook also marks the entry point that plan item G.c will
    extend with the optional ``kz * thickness`` ill-conditioning
    warning when the propagator-matrix path lands; G.0 ships only
    the geometric checks.

    Raises ``ValueError`` on the same conditions as
    :func:`_validate_borehole_layers`, plus when ``a <= 0``.
    """
    _validate_borehole_layers(layers)
    if a <= 0:
        raise ValueError("a must be positive")


def _validate_flexural_layers_stacked(
    layers: tuple[BoreholeLayer, ...],
    a: float,
    vs: float,
) -> None:
    """
    Validate a stacked borehole-layer geometry for the flexural
    multi-layer (cased-hole) dispersion API.

    Adds the per-layer slow-formation constraint
    ``layer.vs >= vs`` (every layer must be at least as fast in
    shear as the formation half-space) on top of the geometry
    checks in :func:`_validate_borehole_layers_stacked`. The
    constraint is required for the n=1 dipole flexural mode to
    remain bound in every annulus of the stack -- a softer
    layer (``layer.vs < vs``) would let the SV-polarised
    radiation leak into the annulus and the propagator-chain
    bound-regime gate would fail.

    Mirrors the slow-formation constraint that the existing F.2
    single-layer ``flexural_dispersion_layered`` path
    documents but does not enforce programmatically; G' makes
    it a hard rejection at validation time.

    Raises ``ValueError`` on the same conditions as
    :func:`_validate_borehole_layers_stacked`, plus when any
    ``layers[i].vs < vs``.
    """
    _validate_borehole_layers_stacked(layers, a)
    for i, layer in enumerate(layers):
        if layer.vs < vs:
            raise ValueError(
                f"layers[{i}]: layer.vs ({layer.vs}) < formation vs "
                f"({vs}); the flexural cased-hole path requires "
                "every layer to be at least as fast in shear as the "
                "formation (slow-formation regime; see plan G'.0)."
            )


@dataclass
class BoreholeMode:
    """
    Per-frequency phase-slowness curve of a single guided mode.

    Attributes
    ----------
    name : str
        Mode label (``"Stoneley"`` for the n=0 first root,
        ``"flexural"`` for the n=1 first root,
        ``"pseudo_rayleigh"`` for the n=0 leaky branch).
    azimuthal_order : int
        Cylindrical mode index (``0`` for monopole, ``1`` for
        dipole).
    freq : ndarray, shape (n_f,)
        Frequencies (Hz).
    slowness : ndarray, shape (n_f,)
        Phase slowness (s/m): ``slowness[i] = Re(k_z(omega[i])) /
        omega[i]``. ``NaN`` at frequencies where the root finder
        failed (typically below the geometric cutoff for guided
        modes, or in the wrong physical regime for the chosen
        solver).
    attenuation_per_meter : ndarray or None, optional
        Spatial attenuation rate ``Im(k_z(omega[i]))`` in 1/m, for
        leaky modes only. ``None`` (default) for purely-bound
        modes (Stoneley, slow-formation flexural) where the
        attenuation is zero by construction. ``NaN`` at the same
        frequencies where ``slowness`` is NaN.
    """

    name: str
    azimuthal_order: int
    freq: np.ndarray
    slowness: np.ndarray
    attenuation_per_meter: np.ndarray | None = None


@dataclass
class BranchSegment:
    """
    Contiguous stretch of finite samples in a dispersion curve.

    A :class:`BoreholeMode` may contain multiple physical segments
    separated by NaN gaps where the underlying mode does not exist
    (e.g., below a geometric cutoff) or where the marcher rejected
    a step. :class:`BranchSegment` represents one such contiguous
    stretch.

    Attributes
    ----------
    start_idx : int
        Index of the first sample of the segment in the original
        frequency grid (inclusive).
    end_idx : int
        Index of the last sample of the segment in the original
        frequency grid (inclusive). For a single-sample segment,
        ``end_idx == start_idx``.
    freq : ndarray, shape (n,)
        Frequencies in this segment, copied (or sliced) from the
        original frequency grid in the same order.
    kz : ndarray, shape (n,) complex
        Complex axial wavenumbers at each frequency in the
        segment.

    See Also
    --------
    segments_from_kz_curve : Build a list of segments from a marcher
        output.
    """

    start_idx: int
    end_idx: int
    freq: np.ndarray
    kz: np.ndarray

    def __len__(self) -> int:
        return int(self.end_idx - self.start_idx + 1)
