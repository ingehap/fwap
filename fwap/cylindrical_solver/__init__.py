"""
Cylindrical-borehole modal-determinant solver (Schmitt 1988).

Implements the isotropic-elastic n=0 (axisymmetric) modal
determinant whose lowest-:math:`k_z` zero is the Stoneley wave.
Replaces the rational-interpolation phenomenology in
:mod:`fwap.cylindrical` for the Stoneley wave with a real
boundary-value-problem dispersion law derived from continuity and
stress conditions at the borehole wall.

Scope (this module)
-------------------
* Isotropic-elastic formation, single-interface and single-extra-
  annular-layer geometries; **VTI formation** (transversely
  isotropic, vertical symmetry axis) for the n=0 (Stoneley) and
  n=1 (flexural, slow formation) bound modes via the Schmitt 1989
  modal determinant. Multi-layer (cased-hole / propagator-matrix)
  extension is plan item G; fast-formation TI flexural via the
  complex-determinant variant is a deferred follow-up to plan H.
* **VTI public APIs**: :func:`stoneley_dispersion_vti` (n=0; any
  formation) and :func:`flexural_dispersion_vti` (n=1; slow-
  formation TI, ``V_Sv < V_f``). Five-parameter stiffness tensor
  (C11, C13, C33, C44, C66) plus density. With an isotropic
  tensor both routines bit-match the unlayered isotropic
  counterparts. The Norris 1990 LF closed-form
  ``S_ST^2 = 1/V_f^2 + rho_f / C66`` (TI-specific oracle) anchors
  the n=0 validation; the Sinha-Norris-Chang LF asymptote
  ``slowness -> 1/V_Sv`` and the
  :func:`fwap.cylindrical.flexural_dispersion_vti_physical`
  weak-anisotropy regression anchor the n=1 case. See plan H in
  ``docs/plans/cylindrical_biot.md`` for the full breakdown.
* **Bound-mode + n=0 leaky regimes**: the bound-mode public APIs
  (:func:`stoneley_dispersion`, :func:`flexural_dispersion`)
  cover the regime ``k_z > omega / V_S`` (all radial wavenumbers
  real). The first leaky-mode public API
  (:func:`pseudo_rayleigh_dispersion`) extends this to the n=0
  leaky regime via outgoing-wave Hankel-function boundary
  conditions and complex-:math:`k_z` Mueller iteration. The
  fast-formation flexural and quadrupole regimes are wired into
  the public APIs (auto-dispatched).
* **n=0, n=1, n=2 azimuthal orders**: monopole (Stoneley + pseudo-
  Rayleigh), dipole (flexural), quadrupole. All three share the
  Helmholtz-decomposition machinery; the n=2 case uses a
  three-scalar decomposition with a 4x4 modal matrix.
* **Single-extra-layer (mudcake / altered zone) extension**: the
  layered public APIs :func:`stoneley_dispersion_layered` (n=0,
  7x7 modal determinant) and :func:`flexural_dispersion_layered`
  (n=1, 10x10 modal determinant) handle one annular layer between
  fluid and formation. n=0 layered covers both fast- and slow-
  formation bound regimes; n=1 layered covers slow-formation only
  with the additional ``layer.vs >= vs`` constraint required to
  keep the wave bound in the annulus. Layer parameters are
  collected in :class:`BoreholeLayer`. See plan F in
  ``docs/plans/cylindrical_biot.md`` for the full decomposition.
* **Cased-hole multi-layer extension** (n=0, n=1, n=2; slow-
  formation): for an arbitrary ``N``-layer stack (typically
  ``fluid -> casing -> cement -> formation`` or ``fluid ->
  casing -> cement -> mudcake -> formation``), all of
  :func:`stoneley_dispersion_layered`,
  :func:`flexural_dispersion_layered`, and
  :func:`quadrupole_dispersion_layered` dispatch to Thomson-
  Haskell-style propagator-matrix paths
  (:func:`_modal_determinant_n0_cased` for n=0, 4x4 per-layer
  propagators folded into a 7x7 modal determinant;
  :func:`_modal_determinant_n1_cased` for n=1, 6x6 per-layer
  propagators folded into a 10x10 modal determinant;
  :func:`_modal_determinant_n2_cased` for n=2, 6x6 per-layer
  propagators folded into a 10x10 modal determinant with the
  Bessel-function index shift ``(I_0, I_1) -> (I_1, I_2)`` and
  the explicit ``n = 2`` azimuthal factors). The ``N=0`` paths
  remain bit-equivalent to the unlayered solvers; the n=0
  ``N=1`` path is bit-equivalent to F.1's hand-coded form and
  the n=1 ``N=1`` path is bit-equivalent to F.2. The n=1 and
  n=2 slow-formation paths additionally enforce the
  ``layer.vs >= vs`` per-layer constraint at validation time
  via :func:`_validate_flexural_layers_stacked` (same
  constraint at both azimuthal orders). At n=1 and n=2 with
  fast formation (``V_S > V_f`` and a non-empty layer stack),
  the dispatch instead routes through the complex-``k_z``
  determinants :func:`_modal_determinant_n1_cased_complex` /
  :func:`_modal_determinant_n2_cased_complex` and the drivers
  :func:`_flexural_dispersion_fast_formation_layered` /
  :func:`_quadrupole_dispersion_fast_formation_layered`, which
  mirror the unlayered fast-formation solvers by brentq'ing
  ``Im(det)`` along the real-``k_z`` axis (the fluid radial
  wavenumber turns imaginary once the phase velocity exceeds
  ``V_f``, while the formation P / S branches stay bound). The
  per-layer slow-formation constraint does not apply in that
  regime -- a cement layer softer in shear than the formation
  is physically permissible behind a fast carbonate. See plan G
  in ``docs/plans/cylindrical_biot.md`` and the G / G' / G''
  sub-plan docs for the full breakdown.

Sign conventions
----------------
* Time dependence ``e^{-i omega t}``.
* Axial dependence ``e^{i k_z z}``; bound modes have ``k_z > 0``.
* Azimuthal dependence ``e^{i n theta}``.
* Radial decay constants are defined positive in the bound regime:

    F = sqrt(k_z^2 - omega^2 / V_f^2)   > 0    (fluid evanescent)
    p = sqrt(k_z^2 - omega^2 / V_P^2)   > 0    (formation P decay)
    s = sqrt(k_z^2 - omega^2 / V_S^2)   > 0    (formation S decay)

* Fluid pressure ``P = A I_0(F r)`` (the regular-at-origin
  modified Bessel; equivalent to ``J_0(alpha r)`` with
  ``alpha = i F``).
* Formation: scalar P potential ``phi = B K_0(p r)`` and SV
  vector-potential theta-component ``psi_theta = C K_1(s r)``
  (Helmholtz decomposition for axisymmetric fields; see
  Aki & Richards 2002 sect. 7.2).

Validation strategy
-------------------
Validates the modal matrix against:

* The closed-form Stoneley low-frequency formula
  ``S_ST^2 = 1/V_f^2 + rho_f / mu`` (White 1983 sect. 5.5).
* The dipole flexural mode's two asymptotic limits already
  encoded in :func:`fwap.cylindrical.flexural_dispersion_physical`:
  ``s_low = 1/V_S`` and ``s_high = 1/V_R`` from
  :func:`fwap.cylindrical.rayleigh_speed`.

If the modal matrix entries below were transcribed wrong, those
checks fail loudly.

References
----------
* Schmitt, D. P. (1988). Shear-wave logging in elastic formations.
  *Journal of the Acoustical Society of America* 84(6), 2230-2244.
* Cheng, C. H., & Toksoz, M. N. (1981). Elastic wave propagation
  in a fluid-filled borehole and synthetic acoustic logs.
  *Geophysics* 46(7), 1042-1053.
* Paillet, F. L., & Cheng, C. H. (1991). *Acoustic Waves in
  Boreholes.* CRC Press, Ch. 2-3.
* White, J. E. (1983). *Underground Sound: Application of Seismic
  Waves.* Elsevier, sect. 5.5 (Stoneley low-f closed form).
* Aki, K., & Richards, P. G. (2002). *Quantitative Seismology*,
  2nd ed., sect. 7.2 (Helmholtz scalar/vector potential
  decomposition in cylindrical coordinates).
"""

from __future__ import annotations

# Re-exports for the Bessel-function and radial-wavenumber helpers
# extracted to ``_bessel`` in Phase 1 step 2 of the package split.
# Same explicit ``X as X`` form as for ``_dataclasses`` above.
from fwap.cylindrical_solver._bessel import _i0_i1 as _i0_i1
from fwap.cylindrical_solver._bessel import _k0_k1 as _k0_k1
from fwap.cylindrical_solver._bessel import _k_or_hankel as _k_or_hankel
from fwap.cylindrical_solver._bessel import (
    _layered_n0_bessel_pack as _layered_n0_bessel_pack,
)
from fwap.cylindrical_solver._bessel import (
    _layered_n0_radial_wavenumbers as _layered_n0_radial_wavenumbers,
)
from fwap.cylindrical_solver._bessel import (
    _radial_wavenumbers_vti as _radial_wavenumbers_vti,
)

# Re-exports for the cased-hole multi-layer helpers extracted to
# ``_cased`` in Phase 1 step 9 of the package split.
from fwap.cylindrical_solver._cased import (
    _layer_e_matrix_n0 as _layer_e_matrix_n0,
)
from fwap.cylindrical_solver._cased import (
    _layer_e_matrix_n1 as _layer_e_matrix_n1,
)
from fwap.cylindrical_solver._cased import (
    _layer_e_matrix_n1_complex as _layer_e_matrix_n1_complex,
)
from fwap.cylindrical_solver._cased import (
    _layer_e_matrix_n2 as _layer_e_matrix_n2,
)
from fwap.cylindrical_solver._cased import (
    _layer_e_matrix_n2_complex as _layer_e_matrix_n2_complex,
)
from fwap.cylindrical_solver._cased import (
    _layer_propagator_n0 as _layer_propagator_n0,
)
from fwap.cylindrical_solver._cased import (
    _layer_propagator_n1 as _layer_propagator_n1,
)
from fwap.cylindrical_solver._cased import (
    _layer_propagator_n1_complex as _layer_propagator_n1_complex,
)
from fwap.cylindrical_solver._cased import (
    _layer_propagator_n2 as _layer_propagator_n2,
)
from fwap.cylindrical_solver._cased import (
    _layer_propagator_n2_complex as _layer_propagator_n2_complex,
)
from fwap.cylindrical_solver._cased import (
    _modal_determinant_n0_cased as _modal_determinant_n0_cased,
)
from fwap.cylindrical_solver._cased import (
    _modal_determinant_n0_microannulus as _modal_determinant_n0_microannulus,
)
from fwap.cylindrical_solver._cased import (
    _modal_determinant_n1_cased as _modal_determinant_n1_cased,
)
from fwap.cylindrical_solver._cased import (
    _modal_determinant_n1_cased_complex as _modal_determinant_n1_cased_complex,
)
from fwap.cylindrical_solver._cased import (
    _modal_determinant_n2_cased as _modal_determinant_n2_cased,
)
from fwap.cylindrical_solver._cased import (
    _modal_determinant_n2_cased_complex as _modal_determinant_n2_cased_complex,
)

# Re-exports for the dataclasses + per-stack validators that were
# extracted to a sibling submodule in the Phase 1 package split. The
# explicit ``X as X`` form preserves backwards-compatible imports
# (``from fwap.cylindrical_solver import BoreholeLayer`` and the
# private validators consumed by ``tests/test_cylindrical_solver.py``)
# without ruff F401-flagging the private names.
from fwap.cylindrical_solver._dataclasses import BoreholeLayer as BoreholeLayer
from fwap.cylindrical_solver._dataclasses import BoreholeMode as BoreholeMode
from fwap.cylindrical_solver._dataclasses import BranchSegment as BranchSegment
from fwap.cylindrical_solver._dataclasses import FluidAnnulus as FluidAnnulus
from fwap.cylindrical_solver._dataclasses import (
    _validate_borehole_layers as _validate_borehole_layers,
)
from fwap.cylindrical_solver._dataclasses import (
    _validate_borehole_layers_stacked as _validate_borehole_layers_stacked,
)
from fwap.cylindrical_solver._dataclasses import (
    _validate_flexural_layers_stacked as _validate_flexural_layers_stacked,
)
from fwap.cylindrical_solver._dataclasses import (
    _validate_fluid_annulus as _validate_fluid_annulus,
)

# Re-exports for the leaky-mode helpers extracted to ``_leaky``
# in Phase 1 step 5 of the package split.
from fwap.cylindrical_solver._leaky import (
    _classify_marcher_step as _classify_marcher_step,
)
from fwap.cylindrical_solver._leaky import (
    _detect_leaky_branches as _detect_leaky_branches,
)
from fwap.cylindrical_solver._leaky import (
    _march_complex_dispersion as _march_complex_dispersion,
)
from fwap.cylindrical_solver._leaky import (
    _march_complex_dispersion_validated as _march_complex_dispersion_validated,
)
from fwap.cylindrical_solver._leaky import (
    _modal_determinant_n0_complex as _modal_determinant_n0_complex,
)
from fwap.cylindrical_solver._leaky import (
    _track_complex_root as _track_complex_root,
)
from fwap.cylindrical_solver._leaky import (
    pseudo_rayleigh_dispersion as pseudo_rayleigh_dispersion,
)
from fwap.cylindrical_solver._leaky import (
    segments_from_kz_curve as segments_from_kz_curve,
)
from fwap.cylindrical_solver._leaky import (
    trapped_pseudo_rayleigh_dispersion as trapped_pseudo_rayleigh_dispersion,
)

# Re-exports for the n=0 isotropic (Stoneley) helpers extracted to
# ``_n0_isotropic`` in Phase 1 step 3 of the package split. Same
# explicit ``X as X`` form as for ``_dataclasses`` and ``_bessel``.
from fwap.cylindrical_solver._n0_isotropic import (
    _modal_determinant_n0 as _modal_determinant_n0,
)
from fwap.cylindrical_solver._n0_isotropic import (
    _stoneley_kz_bracket as _stoneley_kz_bracket,
)
from fwap.cylindrical_solver._n0_isotropic import (
    stoneley_dispersion as stoneley_dispersion,
)

# Re-exports for the n=0 layered (Stoneley) helpers extracted to
# ``_n0_layered`` in Phase 1 step 7 of the package split.
from fwap.cylindrical_solver._n0_layered import (
    _layered_n0_row1_at_a as _layered_n0_row1_at_a,
)
from fwap.cylindrical_solver._n0_layered import (
    _layered_n0_row2_at_a as _layered_n0_row2_at_a,
)
from fwap.cylindrical_solver._n0_layered import (
    _layered_n0_row3_at_a as _layered_n0_row3_at_a,
)
from fwap.cylindrical_solver._n0_layered import (
    _layered_n0_row4_at_b as _layered_n0_row4_at_b,
)
from fwap.cylindrical_solver._n0_layered import (
    _layered_n0_row5_at_b as _layered_n0_row5_at_b,
)
from fwap.cylindrical_solver._n0_layered import (
    _layered_n0_row6_at_b as _layered_n0_row6_at_b,
)
from fwap.cylindrical_solver._n0_layered import (
    _layered_n0_row7_at_b as _layered_n0_row7_at_b,
)
from fwap.cylindrical_solver._n0_layered import (
    _microannulus_kz_window as _microannulus_kz_window,
)
from fwap.cylindrical_solver._n0_layered import (
    _modal_determinant_n0_layered as _modal_determinant_n0_layered,
)
from fwap.cylindrical_solver._n0_layered import (
    _stoneley_kz_bracket_cased as _stoneley_kz_bracket_cased,
)
from fwap.cylindrical_solver._n0_layered import (
    _stoneley_kz_bracket_layered as _stoneley_kz_bracket_layered,
)
from fwap.cylindrical_solver._n0_layered import (
    stoneley_dispersion_layered as stoneley_dispersion_layered,
)
from fwap.cylindrical_solver._n0_layered import (
    stoneley_dispersion_microannulus as stoneley_dispersion_microannulus,
)

# Re-exports for the n=1 isotropic flexural helpers extracted to
# ``_n1_isotropic`` in Phase 1 step 4 of the package split.
from fwap.cylindrical_solver._n1_isotropic import (
    _flexural_dispersion_fast_formation as _flexural_dispersion_fast_formation,
)
from fwap.cylindrical_solver._n1_isotropic import (
    _flexural_kz_bracket as _flexural_kz_bracket,
)
from fwap.cylindrical_solver._n1_isotropic import (
    _modal_determinant_n1 as _modal_determinant_n1,
)
from fwap.cylindrical_solver._n1_isotropic import (
    _modal_determinant_n1_complex as _modal_determinant_n1_complex,
)
from fwap.cylindrical_solver._n1_isotropic import (
    flexural_dispersion as flexural_dispersion,
)
from fwap.cylindrical_solver._n1_layered import (
    _flexural_kz_bracket_cased as _flexural_kz_bracket_cased,
)
from fwap.cylindrical_solver._n1_layered import (
    _flexural_kz_bracket_layered as _flexural_kz_bracket_layered,
)

# Re-exports for the n=1 layered (flexural) helpers extracted to
# ``_n1_layered`` in Phase 1 step 8 of the package split.
from fwap.cylindrical_solver._n1_layered import (
    _layered_n1_row1_at_a as _layered_n1_row1_at_a,
)
from fwap.cylindrical_solver._n1_layered import (
    _layered_n1_row2_at_a as _layered_n1_row2_at_a,
)
from fwap.cylindrical_solver._n1_layered import (
    _layered_n1_row3_at_a as _layered_n1_row3_at_a,
)
from fwap.cylindrical_solver._n1_layered import (
    _layered_n1_row4_at_a as _layered_n1_row4_at_a,
)
from fwap.cylindrical_solver._n1_layered import (
    _layered_n1_row5_at_b as _layered_n1_row5_at_b,
)
from fwap.cylindrical_solver._n1_layered import (
    _layered_n1_row6_at_b as _layered_n1_row6_at_b,
)
from fwap.cylindrical_solver._n1_layered import (
    _layered_n1_row7_at_b as _layered_n1_row7_at_b,
)
from fwap.cylindrical_solver._n1_layered import (
    _layered_n1_row8_at_b as _layered_n1_row8_at_b,
)
from fwap.cylindrical_solver._n1_layered import (
    _layered_n1_row9_at_b as _layered_n1_row9_at_b,
)
from fwap.cylindrical_solver._n1_layered import (
    _layered_n1_row10_at_b as _layered_n1_row10_at_b,
)
from fwap.cylindrical_solver._n1_layered import (
    _modal_determinant_n1_layered as _modal_determinant_n1_layered,
)
from fwap.cylindrical_solver._n1_layered import (
    flexural_dispersion_layered as flexural_dispersion_layered,
)

# Re-exports for the n=2 quadrupole helpers extracted to
# ``_n2_quadrupole`` in Phase 1 step 6 of the package split.
from fwap.cylindrical_solver._n2_quadrupole import (
    _modal_determinant_n2 as _modal_determinant_n2,
)
from fwap.cylindrical_solver._n2_quadrupole import (
    _modal_determinant_n2_complex as _modal_determinant_n2_complex,
)
from fwap.cylindrical_solver._n2_quadrupole import (
    _quadrupole_dispersion_fast_formation as _quadrupole_dispersion_fast_formation,
)
from fwap.cylindrical_solver._n2_quadrupole import (
    _quadrupole_dispersion_fast_formation_layered as _quadrupole_dispersion_fast_formation_layered,
)
from fwap.cylindrical_solver._n2_quadrupole import (
    _quadrupole_kz_bracket as _quadrupole_kz_bracket,
)
from fwap.cylindrical_solver._n2_quadrupole import (
    _quadrupole_kz_bracket_cased as _quadrupole_kz_bracket_cased,
)
from fwap.cylindrical_solver._n2_quadrupole import (
    quadrupole_dispersion as quadrupole_dispersion,
)
from fwap.cylindrical_solver._n2_quadrupole import (
    quadrupole_dispersion_layered as quadrupole_dispersion_layered,
)
from fwap.cylindrical_solver._vti import (
    _flexural_kz_bracket_vti as _flexural_kz_bracket_vti,
)
from fwap.cylindrical_solver._vti import (
    _is_isotropic_stiffness as _is_isotropic_stiffness,
)
from fwap.cylindrical_solver._vti import (
    _modal_determinant_n0_vti as _modal_determinant_n0_vti,
)
from fwap.cylindrical_solver._vti import (
    _modal_determinant_n1_vti as _modal_determinant_n1_vti,
)
from fwap.cylindrical_solver._vti import (
    _modal_row1_at_a_n1_vti as _modal_row1_at_a_n1_vti,
)
from fwap.cylindrical_solver._vti import (
    _modal_row1_at_a_vti as _modal_row1_at_a_vti,
)
from fwap.cylindrical_solver._vti import (
    _modal_row2_at_a_n1_vti as _modal_row2_at_a_n1_vti,
)
from fwap.cylindrical_solver._vti import (
    _modal_row2_at_a_vti as _modal_row2_at_a_vti,
)
from fwap.cylindrical_solver._vti import (
    _modal_row3_at_a_n1_vti as _modal_row3_at_a_n1_vti,
)
from fwap.cylindrical_solver._vti import (
    _modal_row3_at_a_vti as _modal_row3_at_a_vti,
)
from fwap.cylindrical_solver._vti import (
    _modal_row4_at_a_n1_vti as _modal_row4_at_a_n1_vti,
)
from fwap.cylindrical_solver._vti import (
    _polarization_ratio_uz_over_ur_vti as _polarization_ratio_uz_over_ur_vti,
)
from fwap.cylindrical_solver._vti import (
    _stoneley_kz_bracket_vti as _stoneley_kz_bracket_vti,
)

# Re-exports for the VTI formation helpers extracted to ``_vti``
# in Phase 1 step 10 of the package split (final extraction).
from fwap.cylindrical_solver._vti import (
    _validate_vti_stiffness as _validate_vti_stiffness,
)
from fwap.cylindrical_solver._vti import (
    flexural_dispersion_vti as flexural_dispersion_vti,
)
from fwap.cylindrical_solver._vti import (
    stoneley_dispersion_vti as stoneley_dispersion_vti,
)
