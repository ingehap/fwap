# Changelog

All notable changes to this project are documented here. The format
loosely follows [Keep a Changelog](https://keepachangelog.com/), and
the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Cross-mode consistency QC for the picker.** Closes the soft
  Workflow-3 gap flagged in the closing paragraph of the docx
  review: the book's QC philosophy is *"log continuity AND
  cross-consistency between modes"*; continuity was already
  enforced inside the Viterbi pickers but no API surfaced the
  cross-consistency layer. New ``quality_control_picks(picks,
  depth=, *, vp_vs_min=1.4, vp_vs_max=2.6, require_time_order=
  True)`` returns a ``PickQualityFlags`` with two checks: the
  Vp/Vs ratio (``s_S / s_P``) is gated against the canonical
  sedimentary-rock band, and the canonical time ordering
  ``t_P <= t_S <= t_PseudoRayleigh <= t_Stoneley`` is verified
  over the modes that were picked. The function only flags --
  callers decide whether to drop, mark, or human-review the
  flagged depths. ``quality_control_track`` is the multi-depth
  analogue. Picks whose Vp or S is missing have ``vp_vs=None``
  and ``vp_vs_in_band=True`` (the gate is skipped, not failed).
- **Stress-direction / flexural-fracture-indicator API on top of the
  cross-dipole Alford rotation.** The book frames the Workflow-3
  dipole-sonic deliverable as "shear anisotropy, mechanical
  properties and fracture indicators from the flexural wave" plus
  "stress-direction estimation"; the numerics already lived inside
  ``alford_rotation`` (the fast-shear angle, the cross-energy
  ratio) but no API surfaced them in those petrophysical terms.
  New ``stress_anisotropy_from_alford(alford, dt)`` returns a
  ``StressAnisotropyEstimate`` carrying ``max_horizontal_stress
  _azimuth`` (= fast-shear angle, with a docstring caveat that the
  conventional stress-direction interpretation depends on whether
  the anisotropy is stress- or fracture-induced),
  ``min_horizontal_stress_azimuth`` (orthogonal, folded into
  ``(-pi/2, pi/2]``), ``splitting_time_delay`` (cross-correlation
  lag of slow vs fast), ``anisotropy_strength`` (relative L2 norm
  in ``[0, 1]``), ``rotation_quality`` (= ``1 - cross_energy_ratio``)
  and a heuristic ``fracture_indicator`` (their product). The
  underlying ``alford`` is kept on the result so callers that need
  the rotated waveforms can still reach them.
- **Wavelet-shape + onset-polarity expert rules in the picker.** The
  book (Mari et al. 1994, Part 1) lists *"expert rules on slowness
  range, **wavelet shape, onset polarity**, coherence across the
  receiver array, depth-to-depth continuity"* as the picker's
  knowledge-based discriminator set; the package only had the
  bolded subset (slowness windows + STC coherence + Viterbi
  continuity). Two opt-in expert rules now run as post-pick gates:
  ``polarity`` (``+1``/``-1``/``0``) checks the sign of the
  stacked window's largest-absolute sample, and ``shape_match_min``
  gates picks against the absolute Pearson correlation between the
  stacked window and a Ricker template at the prior's ``f0``. Both
  are exposed via ``filter_picks_by_shape(picks, data, dt,
  offsets, *, priors)`` and ``filter_track_by_shape(track, datas,
  dt, offsets, *, priors)``; the underlying ``onset_polarity`` and
  ``wavelet_shape_score`` primitives are also public. Default
  priors leave both gates disabled (``polarity=0``,
  ``shape_match_min=0.0``) so existing callers are unaffected.
- **Per-mode amplitude logs in the Workflow-1 picker pipeline.** The
  book (Mari et al. 1994, Part 1) frames the rule-based picker's
  deliverable as *"continuous Vp, Vs and Stoneley slowness curves
  together with per-mode amplitude **and coherence** logs"*; the
  pipeline only carried the coherence half. ``STCResult`` now has an
  ``amplitude`` ``ndarray | None`` field of the same
  ``(n_slowness, n_time)`` shape as ``coherence``, populated by both
  ``coherence.stc`` and ``dispersion.dispersive_stc`` with the RMS
  of the per-trace stack contribution at each cell (a unit-amplitude
  sine on every trace gives ``amplitude = 1/sqrt(2)``).
  ``find_peaks`` returns a 4-column ``[slowness, time, coherence,
  amplitude]`` table when the input STC carries amplitude, and
  ``ModePick`` gained an ``amplitude: float | None`` field that
  ``pick_modes`` / ``track_modes`` / ``viterbi_pick`` /
  ``viterbi_pick_joint`` / ``viterbi_posterior_marginals`` now
  populate from the picked cell. Existing 3-column / amplitude=None
  call sites continue to work unchanged.
- **Altered-zone velocity contrast as a Workflow-2 deliverable.**
  The book (Mari et al. 1994, Part 3) frames the intercept-time
  workflow's altered-zone product as the *(thickness, velocity-
  contrast)* pair, but `fwap.tomography` only had
  `delay_to_altered_zone_thickness` -- which forces the caller to
  supply the altered-zone slowness as an input. The single
  refraction-geometry equation `delay = 2 * h * (s_altered -
  s_virgin)` is one constraint in two unknowns, so the package now
  exposes both directions plus a joint helper:
  `delay_to_altered_zone_velocity_contrast(delay, thickness)` is
  the algebraic dual; `altered_zone_estimate(delay, s_virgin,
  thickness=...)` or `altered_zone_estimate(delay, s_virgin,
  slowness_altered=...)` returns an `AlteredZoneEstimate` dataclass
  carrying thickness, absolute altered slowness, and slowness
  contrast at every depth, with the helper rejecting calls that
  pin both or neither anchor.
- **τ-p (slant-stack / linear Radon) wave separation.** The book
  (Mari et al. 1994, Part 2) lists the τ-p domain alongside f-k as
  a textbook multichannel velocity-filter, but `fwap.wavesep`
  previously offered only `fk_filter` and SVD/K-L. New
  `tau_p_forward`, `tau_p_adjoint`, `tau_p_inverse`, and
  `tau_p_filter` mirror the f-k API: forward stacks a (t, x) gather
  into a (τ, p) panel, the LSQR-style `tau_p_inverse` is a true
  per-frequency pseudoinverse (round-trip identity to ~0.1 % on a
  clean monopole gather), and the convenience `tau_p_filter` does
  forward → cosine-tapered slowness mask → adjoint for band-pass
  separation. Unlike `fk_filter`, τ-p tolerates non-uniform
  receiver spacings. New `demos.demo_tau_p_separation` and CLI
  subcommand `fwap taup` exercise the pipeline on the canonical
  P/S/Stoneley monopole gather.
- **Pseudo-Rayleigh / guided-mode picking.** The book (Mari et al.
  1994, Part 1) lists pseudo-Rayleigh alongside P, S, and Stoneley as
  one of the arrivals the rule-based picker must identify in fast
  formations; it was the only mode in that list missing from the
  package. `fwap.picker.DEFAULT_PRIORS` now carries a
  `"PseudoRayleigh"` entry (130-200 us/ft), and Stoneley's lower
  bound has been tightened from 180 to 200 us/ft so the four
  windows are non-overlapping. New phenomenological dispersion law
  `fwap.synthetic.pseudo_rayleigh_dispersion(vs, v_fluid, a_borehole)`
  matches the formation shear slowness at the low-frequency cutoff
  and asymptotes to the borehole-fluid slowness at high frequency.
  `monopole_formation_modes(...)` gained a `f_pr=` kwarg that
  appends a fourth mode at the band-centre slowness predicted by
  that law. New `demos.demo_pseudo_rayleigh` and CLI subcommand
  `fwap pseudorayleigh` exercise the four-mode pipeline end-to-end.
  `viterbi_pick_joint` and `viterbi_posterior_marginals` remain
  hardcoded to the (P, S, Stoneley) triple (extending the trellis
  to 4 modes squares its width); both now subset the default priors
  for backward compatibility and raise on a 4-mode prior dict.
- **DLIS read / write** — `fwap.io.read_dlis`, `fwap.io.write_dlis`,
  and `DlisCurves` mirror the existing LAS API for the binary RP66 v1
  format. Wraps `dlisio` for reading and `dliswriter` for writing.
  Well metadata is re-keyed to LAS-2.0 mnemonics (`WELL`, `COMP`,
  `FLD`, `PROD`, `UWI`) so the same dict can be passed to either
  writer.

### Changed
- **All log-format libraries are now core dependencies.** `lasio`,
  `dlisio`, `dliswriter`, and `segyio` are folded into the base
  `dependencies` list; the `[io]`, `[dlis]`, and `[segy]` extras
  are gone. The corresponding lazy-import helpers (`_require_lasio`,
  `_require_dlisio`, `_require_dliswriter`, `_require_segyio`) and
  their friendly-error guards have been removed.

## [0.4.0] - 2026-04-22

First formally-versioned release. Promotes the port of the 1994 Mari
et al. algorithms from a prototype into a tested, documented Python
package.

### Added
- **Package infrastructure**
  - Repository renamed `src/` to `fwap/` so the `from fwap.X import Y`
    imports throughout the codebase actually resolve.
  - `pyproject.toml` with runtime dependencies (NumPy, SciPy,
    Matplotlib) and optional `dev` / `docs` / `io` extras.
  - `LICENSE` (MIT).
  - `CITATION.cff` citing both the software and the 1994 book.
  - `README.md` expanded from a title-only placeholder to a
    chapter-to-module map, install + quick-start, companion
    references, and links to tests and docs.
  - `CHANGELOG.md` (this file).
- **Tests** (`tests/`, 83 cases, ~9 s run):
  - One file per algorithm module plus edge cases, sign-convention
    invariants, and demo regression tests that assert on the numerics
    logged by each `demo_*` function.
  - `.github/workflows/ci.yml` runs pytest on Python 3.9 / 3.11 /
    3.12 and smoke-tests `python -m fwap --quiet`.
- **Documentation** (`docs/`):
  - Sphinx skeleton (`conf.py`, `index.rst`, `quickstart.rst`,
    `chapter_map.rst`, `api.rst`, `changelog.rst`) that autogenerates
    an API reference from the docstrings.
  - `.readthedocs.yaml` so the repo can be connected to ReadTheDocs
    without further configuration.
  - CI `docs` job builds the site and uploads the rendered HTML as
    an artifact.
- **New algorithms / modules**:
  - `fwap.rockphysics.elastic_moduli(vp, vs, rho)` -> bulk / shear /
    Young's modulus and Poisson's ratio, closing the loop from
    raw-waveform Vp/Vs to geomechanical curves.
  - `fwap.rockphysics.vp_vs_ratio` (lithology / fluid indicator).
  - `fwap.io.read_las` / `fwap.io.write_las` via the optional
    `lasio` dependency.
- **API extensions**:
  - `fwap.anisotropy.alford_rotation_from_tensor` accepts a packed
    `(2, 2, n_samples)` cross-dipole tensor.
  - `fwap.dip.AzimuthalGather` NamedTuple returned by
    `synthesize_azimuthal_arrival` (tuple unpacking still works).
  - `fwap.synthetic.ArrayGeometry.schlumberger_array_sonic()`
    classmethod documenting the canonical 8/10 ft/6 in reference
    geometry.
  - `fwap.logger` is the shared package logger; every submodule
    imports it from `fwap._common`.
  - `fwap.picker.track_modes` gained `continuity_tol_cap_factor`
    (default 3.0) bounding the effective jump tolerance across long
    runs of missed picks.
- **Performance**: per-frequency inner loops vectorised in
  `dispersive_stc`, `phase_slowness_from_f_k` (both methods),
  `phase_slowness_matrix_pencil`, and `synthesize_gather` (5x-42x
  speedups on the reference benchmarks).
- **Docstrings**: every algorithm module now carries a book reference
  and a chapter map linking back to Mari, Coppens, Gavin & Wicquart
  (1994); every public symbol has Parameters / Returns sections with
  units and array shapes.

### Changed
- `solve_intercept_time` with `mean_delay_zero=True` now emits two
  separate zero-sum constraint rows (one per delay block). The
  previous single joint row left a
  `(d_src, d_rec) -> (d_src + c, d_rec - c)` gauge unresolved.
- `centroid_frequency_shift_Q` / `spectral_ratio_Q`: internal
  variable renamed `t_arr -> t_travel` to match Quan & Harris (1997);
  the docstring now flags the Gaussian-source assumption.
- `shear_slowness_from_dispersion` emits `logging.warning` when it
  has to fall back from the quality-weighted set to the unweighted
  set.
- `_coherence_after_detilt` now calls `fwap.coherence.semblance`
  instead of reimplementing the ratio; the two sites can no longer
  diverge on the semblance definition.
- `dispersion_family` (argument of `dispersive_stc`) and
  `Mode.dispersion` now document an array-in / array-out contract;
  `dipole_flexural_dispersion`'s type hint reflects it.

### Fixed
- Alford rotation docstring formula was off by a factor of 2:
  `tan(4 theta) = (A + B) / (C - D) = 2 A / (C - D)`. The code was
  already correct.
- `fk_filter` sign convention (`S = -k/f`) is now documented in the
  function's docstring rather than only an inline comment.

### Removed
- `fwap.coherence.semblance` no longer takes a `min_energy`
  parameter. The default was effectively dead (it fired only on a
  bit-exact zero sum). Callers needing an energy floor should filter
  windows upstream.
- `fwap.picker.pick_modes` / `track_modes`:
  `selection_rule="earliest"` was carrying a legacy fwap03 picker
  rule with no tests exercising it. Removed; use
  `"max_coherence"` or `"scored"` (the default).
- `fwap.picker.track_modes`: the deprecated `max_slow_jump_per_depth`
  keyword alias was removed. Callers must use `max_slow_jump`; the
  old name now raises `TypeError` with the standard "unexpected
  keyword argument" message.
- `fwap.tomography.solve_intercept_time`: the deprecated
  `smoothing` scalar was removed. Use the explicit per-block
  weights `smooth_s`, `smooth_src`, `smooth_rec` (all default 0.0).
- Internal references to legacy version tags ("fwap01", "fwap02",
  "v1", "v2", "testing_fwap03.py") have been stripped from
  docstrings and user-facing comments. Those were porting notes
  that did not belong in the published API documentation.
