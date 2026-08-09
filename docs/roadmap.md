# fwap roadmap

A living document of open items that would meaningfully extend fwap
beyond the 0.4.0 release, ordered by estimated effort × user-value.

**Where things stand.** Most of what this document was originally
written to track has shipped. The book's four Parts and the extension
layer were complete by the end of the post-0.4.0 cycle; the
cylindrical-Biot solver family (section A) has since closed out
through leaky modes, quadrupole, layered / cased-hole and VTI; and a
machine-learning layer that was not contemplated when this file was
written now sits alongside the package (section G). Sections B, C and
E are closed and kept for reference.

What remains is shorter and sharper than the list below suggests:

| Open item | Why it matters |
|-----------|----------------|
| **F. Real-data fixtures** | Harness shipped; a real *sonic* gather is still missing. The binding constraint on every quantitative claim in the repo, `sonic_ml`'s included. |
| **G. `sonic_ml` follow-ons** | Free-pipe / leaky cased regime (single-frame *and* joint multi-depth surrogate inversion are done). |
| **A.1 Validation figures** | Ties the solver to published literature rather than to itself. |
| **D. Conda-forge recipe** | Packaging only; unblocked once a PyPI release is live. |

A note on how this file is kept honest: items are marked closed only
when the code and its tests are on `main`, and status claims are
checked against the tree rather than against memory. Section A carried
a "leaky modes are still open" note for some time after the leaky
solvers had actually shipped; that is the failure mode this heading
exists to prevent.

## Released (for reference)

### 0.4.0

All of these landed on the 0.4.0 PR and are covered by tests:

- Part 1: STC + rule-based picker, Viterbi picker (per-mode and
  fully-joint 3-mode).
- Part 2: f-k filtering, SVD / Karhunen-Loeve separation.
- Part 3: intercept-time inversion (midpoint + segmented), dipole
  flexural dispersion (phenomenological + Rayleigh-speed physical
  limit), delay-to-altered-zone-thickness conversion.
- Part 4: dip / azimuth estimation.
- Extensions: cross-dipole Alford rotation, Q from centroid-shift and
  spectral-ratio, elastic moduli, Reuss / Voigt / Hill mixing laws,
  Stoneley permeability indicator, LAS / SEG-Y I/O, `fwap process`
  CLI.

### Since 0.4.0 ([Unreleased])

The post-0.4.0 cycle was a sweep of book-completeness gaps flagged
in `ideas/Mari1994.docx`; every algorithm-level item the book names
is now in the package:

- **Part 1 picker**: pseudo-Rayleigh / guided-mode picking is now a
  fourth default mode (`DEFAULT_PRIORS["PseudoRayleigh"]`,
  `pseudo_rayleigh_dispersion`); per-mode amplitude logs are
  exposed alongside coherence (`STCResult.amplitude`,
  `ModePick.amplitude`); the wavelet-shape and onset-polarity
  expert rules are available as post-pick filters
  (`onset_polarity`, `wavelet_shape_score`,
  `filter_picks_by_shape`, `filter_track_by_shape`); cross-mode
  consistency QC flags depths where Vp/Vs is unphysical or the
  canonical time ordering is violated (`PickQualityFlags`,
  `quality_control_picks`, `quality_control_track`).
- **Part 2 wave separation**: tau-p / slant-stack / linear Radon
  joins f-k and SVD/K-L (`tau_p_forward`, `tau_p_inverse`,
  `tau_p_adjoint`, `tau_p_filter`).
- **Part 3 altered zone**: the joint (thickness, velocity-contrast)
  deliverable is now a first-class output
  (`altered_zone_estimate`, `delay_to_altered_zone_velocity_contrast`,
  `AlteredZoneEstimate`); the original
  `delay_to_altered_zone_thickness` stays as the thickness-anchor
  branch.
- **Workflow 3 dipole-sonic**: a petrophysical labelling layer
  re-frames the Alford rotation in stress-direction terms
  (`StressAnisotropyEstimate`, `stress_anisotropy_from_alford`)
  with max-horizontal-stress azimuth, splitting-time delay,
  anisotropy strength, and a heuristic fracture indicator.
- **I/O**: DLIS read / write joins LAS and SEG-Y as a core
  dependency (`read_dlis`, `write_dlis`, `DlisCurves`); all four
  log-format libraries (`lasio`, `dlisio`, `dliswriter`, `segyio`)
  are now in the base `dependencies` list, and the optional
  `[io]` / `[dlis]` / `[segy]` extras are gone.
- **Demos / CLI**: `fwap pseudorayleigh`, `fwap taup`, and
  `fwap dlis` are wired into the demo registry alongside the
  existing chapter demos.

A second sweep (after the docx pair `Paillet1991.docx` and
`Tang2004.docx` were added to `ideas/`) closed the gaps Tang & Cheng
(2004) flag as the post-1994 borehole-acoustic processing literature:

- **Picker → log-curve bridge**: `track_to_log_curves(track) ->
  (depths, curves)` converts a per-depth pick track from
  `track_modes` / `viterbi_pick` / `viterbi_pick_joint` into the
  fixed-length `{mnemonic: ndarray}` dict the LAS / DLIS writers
  consume directly. Slowness is converted to us/ft (the LAS unit
  table convention); missing picks become NaN by default with an
  optional numeric sentinel.
- **Geomechanics layer (`fwap.geomechanics`)**: Rickman 2008
  brittleness / fracability index, Eaton 1969 uniaxial-strain
  closure stress, Lacy 1997 (Chang 2006 form) sandstone UCS,
  Bratli–Risnes 1981 sand-stability flag, density-log overburden
  integration, and a one-call `geomechanics_indices(moduli, ...)`
  bundle returning a `GeomechanicsIndices` dataclass with all four
  indices (closure stress optional, conditional on a supplied
  overburden).
- **Dispersive pseudo-Rayleigh STC**:
  `dispersive_pseudo_rayleigh_stc` is the pseudo-Rayleigh analogue
  of `dispersive_stc`; same back-projection machinery, only the
  per-mode dispersion law differs. Enforces the fast-formation
  existence constraint (`shear_slowness_range[1] < 1 / v_fluid`).
- **Stoneley amplitude fracture indicator**:
  `stoneley_amplitude_fracture_indicator(A_obs, A_ref)` =
  `1 - A_obs / A_ref` — companion to the existing
  `stoneley_permeability_indicator`. Detects the same fractures /
  permeable zones via energy loss rather than via the
  poroelastic-delay slowness shift; the two have complementary
  noise characteristics.
- **Hornby et al. (1989) Stoneley reflection-coefficient fracture-
  aperture inversion**: `stoneley_reflection_coefficient(...)`
  builds `|R|`; `hornby_fracture_aperture(R, frequency_hz,
  V_T, ...)` inverts the low-frequency closed form
  `|R| = ω L₀ / sqrt(V_T² + ω² L₀²)` for the fracture aperture
  `L₀` (m). Quantitative complement to the two slowness- and
  amplitude-based indicators.
- **Thomsen-gamma from combined dipole + Stoneley
  (`fwap.anisotropy`)**: `thomsen_gamma(c44, c66)`,
  `stoneley_horizontal_shear_modulus(s_ST, rho_fluid, v_fluid)`
  (White 1983 / Norris 1990 tube-wave inversion), and a one-call
  `thomsen_gamma_from_logs(s_dipole, s_stoneley, rho, ...)`
  returning a `ThomsenGammaResult` with C44, C66, gamma per depth.
- **Slow-formation Vs from low-frequency Stoneley
  (`fwap.stoneley`)**: `vs_from_stoneley_slow_formation(...)` is
  the primary sonic-only V_S estimator for the case where the
  formation has no S head wave on a monopole gather and
  pseudo-Rayleigh does not exist (V_S < V_fluid; Paillet & Cheng
  1991 Ch. 3).
- **Stress-vs-intrinsic anisotropy classifier
  (`fwap.dispersion`)**: `classify_flexural_anisotropy(curve_a,
  curve_b)` labels a cross-dipole record as `"isotropic"`,
  `"intrinsic"`, `"stress_induced"`, or `"ambiguous"` based on
  whether the slowness difference Δs(f) crosses zero between a
  low-f band and a high-f band — the Sinha & Kostek 1996
  diagnostic that distinguishes far-field rock fabric from
  borehole-wall stress concentration.
- **LWD phenomenological layer (`fwap.lwd`)**: `lwd_collar_mode`,
  `synthesize_lwd_gather`, and `notch_slowness_band` (subtract-
  the-in-band route, preserves out-of-grid signals) deliver the
  monopole-side collar-rejection workflow; `QuadrupoleRingGather`,
  `synthesize_quadrupole_lwd_gather`, `quadrupole_stack` and
  `lwd_quadrupole_priors` deliver the m=2 source / receiver
  geometry that Tang & Cheng 2004 sect. 2.5 frame as the practical
  solution to LWD collar contamination. `fwap lwd` runs the
  worked-example demo. **Not** a layered cylindrical-Biot solver
  (still flagged as Open item A below).

A third sweep was a maintenance pass: behaviour-preserving module
splits and helper consolidations driven by the in-repo
`IMPROVEMENTS.md` and a code-quality review. No public-API changes;
every `from fwap import …` resolves to the same object as before.

- **`fwap.stoneley` split out of `fwap.rockphysics`**: the seven
  Stoneley-wave petrophysical estimators (the four-tool fracture /
  permeability suite plus `vs_from_stoneley_slow_formation`) move
  into a dedicated module. `fwap.rockphysics` shrinks from 1374 to
  428 LoC and now contains only elastic-moduli core +
  Gassmann / Reuss / Voigt / Hill mixing.
- **`fwap.picker` becomes a package**: 2260-LoC monolith → six
  submodules (`_types`, `greedy`, `viterbi`, `posterior`, `shape`,
  `quality`) with the joint-Viterbi trellis primitives shared
  between `viterbi.py` and `posterior.py`. Largest submodule is
  676 LoC.
- **`fwap.geomechanics` becomes a package**: 2197-LoC monolith →
  four submodules (`indices`, `pressures`, `vertical`, `inclined`).
  Sole cross-submodule dependency is `inclined.py` importing
  `MudWeightWindow` from `vertical.py`.
- **Helper consolidations**: `m_per_s_to_us_per_ft(v)` in
  `fwap._common` replaces six inline `1.0e6 / v * 0.3048`
  expressions in `cli.py` / `demos.py`; private `_mohr_coulomb_q`
  (in `fwap.geomechanics.vertical`) labels the
  `(1+sin(phi))/(1-sin(phi))` stress ratio used by both the
  vertical and inclined breakout calculators; private
  `_principal_stresses_at_pw` (in `fwap.geomechanics.inclined`)
  composes `inclined_wellbore_wall_stresses` with
  `_wall_principal_stresses` for a single candidate mud pressure,
  shared by both `inclined_breakout_pressure` and
  `inclined_breakdown_pressure`.
- **Test repairs**: the six `test_dispersion_matches_golden[…]`
  characterisation cases were drifting 1-5 ULPs against goldens
  captured on an older SciPy; the comparison is now `rtol=1e-12`
  with the NaN-mask still checked exactly (a real refactor
  regression would show drift orders of magnitude larger). The
  `test_semblance_scale_invariant` hypothesis strategy no longer
  explores inputs whose squares flush to subnormal float64. Both
  fixes are in the test layer; no implementation behaviour
  changed.
- **`.gitignore`**: the standard Python build / cache artifact
  set, which an editable install + `pytest` run would otherwise
  leave as untracked noise in `git status`.

## Open items

### A. Full cylindrical-Biot dispersion solver

**Status (updated)**: the **bound-mode** halves of the Schmitt /
Paillet–Cheng solver are now both shipped:

- n=0 monopole Stoneley solver: `fwap.stoneley_dispersion` (3×3
  modal determinant in the bound regime; `_modal_determinant_n0`).
- n=1 dipole flexural solver: `fwap.flexural_dispersion` (4×4
  modal determinant in the bound regime; `_modal_determinant_n1`).
  Closed in the [Unreleased] cycle. Slow-formation only
  (`V_S < V_f`); produces slowness ~ `1/V_S` just above the
  geometric cutoff and ~ `1/V_R + Scholte offset` at high f.

The phenomenological models stay shipped
(`fwap.synthetic.dipole_flexural_dispersion`,
`fwap.cylindrical.flexural_dispersion_physical`) for callers that
need a closed-form smoothed-step dispersion curve without solving
the determinant per frequency.

**Status (superseded)**: the paragraphs above describe the state
when only the two bound-mode solvers shipped. The family is now
essentially complete -- plan items A through H in
`plans/cylindrical_biot.md` are all closed, including everything the
"remaining work" note below anticipated:

- **Leaky modes** (the item this section was written to track):
  `pseudo_rayleigh_modal_dispersion` (n=0), leaky flexural (n=1) and
  leaky quadrupole (n=2), with complex-`k_z` root-finding, outgoing-
  wave boundary conditions, and branch tracking across the leaky
  cutoff. Leaky solutions carry a spatial attenuation rate alongside
  the phase slowness.
- **Quadrupole (n=2)**: `quadrupole_dispersion`, bound and leaky.
- **Layered / cased hole**: `stoneley_dispersion_layered`,
  `flexural_dispersion_layered`, `quadrupole_dispersion_layered` over
  a `BoreholeLayer` stack (mudcake, altered zone, casing + cement),
  including fast-formation cased flexural.
- **VTI formations**: `stoneley_dispersion_vti`,
  `flexural_dispersion_vti`.

**What is actually still open here** is narrow:

1. **Validation-figure coverage** (plan item I, marked partial).
   These are the only checks that tie the solver to literature rather
   than to itself, and the item splits cleanly in two.

   *The machinery is done.* `fwap.validation` scores an fwap curve
   against a digitised reference and the notebook asserts a 5 % RMS
   budget per curve, verified to fail on a 12 %-perturbed reference.
   Most of that module is input validation, because hand-tracing a
   printed figure fails in a handful of ways that all produce
   plausible files (µs/ft read as s/m, a velocity axis traced as a
   slowness one, kHz left unconverted); each is refused with a named
   diagnosis, and units are never silently rescaled, since a
   reference adjusted to fit would agree with a wrong solver too.

   *The data is not, and cannot be from here.* No reference CSV is
   shipped, so the notebook currently validates nothing against
   literature — its closing cell says so rather than letting green
   plots imply otherwise. The remaining work is digitising three
   figures (Paillet & Cheng 1991 fig 4.5; Schmitt 1988 fig 4; Tang &
   Cheng 2004 figs 3.7/3.10, 7.1; Schmitt 1989 fig 5), which needs
   the published figures themselves. This repository's sandbox
   permits egress to GitHub only, so obtaining them is a task for a
   human with the books, not a coding session. Once a CSV lands in
   `docs/notebooks/_data/` under the documented name, no code changes:
   the section scores and gates automatically.

   Note the figure numbering: this list previously cited "Tang &
   Cheng 2004 Fig. 3.4", which does not match the notebook's sections
   (figs 3.7 and 3.10 for quadrupole, 7.1 for cased Stoneley). The
   notebook is the accurate list.
2. **Cased flexural bracketing.** The layered n=1 solver no longer
   refuses fast formations, but its root-finding stays sparse for a
   typical casing + cement stack (only a few frequencies converge).
   That sparseness is why `scripts/gen_surrogate_dataset.py` keeps
   the cased dataset single-mode; better bracketing would unlock a
   two-mode cased-hole dataset.

For reference, the original from-scratch problem statement is
preserved below.

Root-find the zeros of the modal determinant ``M_n(ω, k) = 0`` in
complex phase-slowness (axial wavenumber ``k`` for mode order ``n``),
then sample the resulting dispersion curve ``s(f) = k(ω)/ω`` at the
caller's frequencies.

*Radial wavenumbers* (same for every azimuthal order):

```
f_f = sqrt(k² − ω² / V_p_fluid²)       # in the fluid (r < a)
f_p = sqrt(k² − ω² / V_p_solid²)       # P wave in the solid
f_s = sqrt(k² − ω² / V_s_solid²)       # S wave in the solid
```

Each of these is real when the corresponding mode is evanescent at
that radius, imaginary when it's propagating; the branch selection
is what makes the root-finder non-trivial.

*Modal determinants*, from Kurkjian & Chang (1986) Section II and
Paillet & Cheng (1991) Chapter 4:

- **Monopole (n=0)**: a 2×2 determinant involving only fluid and
  solid radial terms. Zeros give the Stoneley mode (c < V_p_f) and
  the pseudo-Rayleigh branch (leaky, c ≈ V_s for high frequency).

- **Dipole (n=1)**: a 3×3 determinant with entries built from
  modified-Bessel combinations ``I_0(f_f a)``, ``I_1(f_f a)``,
  ``K_0(f_p a)``, ``K_1(f_p a)``, ``K_0(f_s a)``, ``K_1(f_s a)``,
  weighted by the Lamé parameters. Zeros give the flexural mode
  that is currently approximated phenomenologically by
  :func:`fwap.synthetic.dipole_flexural_dispersion`.

Boundary conditions at ``r = a`` are the standard elastodynamic set:

1. radial displacement continuity (fluid normal velocity matches
   solid radial velocity);
2. radial stress continuity (fluid pressure matches solid normal
   stress);
3. tangential stress vanishes in the fluid (no shear-wave coupling).

Writing these out in terms of the per-region potentials and
substituting the Bessel-function solutions gives the 3×3 matrix;
the exact layout is in Kurkjian & Chang (1986) equations 8 and 9.

*Implementation strategy*:

1. Start at a frequency where the answer is known (e.g., low
   frequency where ``s → 1 / V_s`` is exact for the flexural mode).
2. March in frequency, using the previous iterate as the initial
   guess for ``scipy.optimize.newton``.
3. Track the sign of ``Im(f_s)`` to stay on the right branch
   (propagating vs evanescent shear); flip branches explicitly at
   the cutoff frequency.
4. For the dipole high-frequency limit, converge to the Scholte
   speed at the fluid-solid interface (slightly below the Rayleigh
   speed used by the current phenomenological code).

Public API target:

```python
fwap.cylindrical.modal_dispersion(
    vp: float,
    vs: float,
    vp_fluid: float,
    rho_solid: float,
    rho_fluid: float,
    a_borehole: float,
    mode: Literal["flexural", "stoneley", "pseudo_rayleigh"] = "flexural",
) -> Callable[[np.ndarray], np.ndarray]
```

Returns the same callable contract as the existing
``dipole_flexural_dispersion``: array of frequencies in, array of
phase slownesses out.

**Scope**: ~500 lines of physics code plus a validation notebook
that reproduces the published dispersion curves (Paillet & Cheng
1991 Figure 4.5; Schmitt 1988 Figure 4; Tang & Cheng 2004 Figure
3.4) to within plotting accuracy. Several days of focused work. The
hardest piece is robust branch selection across the pseudo-Rayleigh
cutoff; start with the Stoneley mode (no cutoff in the band of
interest) before attempting dipole flexural.

**References**:

- Schmitt, D. P. (1988). Shear-wave logging in elastic formations.
  *J. Acoust. Soc. Am.* 84(6), 2230-2244.
- Paillet, F. L., & Cheng, C. H. (1991). *Acoustic Waves in
  Boreholes*, Chapter 4. CRC Press.
- Tang, X.-M., & Cheng, A. (2004). *Quantitative Borehole Acoustic
  Methods*, Chapter 3. Elsevier.
- Kurkjian, A. L., & Chang, S.-K. (1986). Acoustic multipole sources
  in fluid-filled boreholes. *Geophysics* 51(1), 148-163 (most
  explicit derivation of the 3×3 dipole determinant).
- Ellefsen, K. J., Cheng, C. H., & Toksoz, M. N. (1991). Applications
  of perturbation theory to acoustic logging. *J. Geophys. Res.*
  96(B1), 537-549 (starting-guess strategy for the dipole root-finder).

### B. Quantitative Stoneley permeability (Tang–Cheng–Toksöz 1991)

**Status**: closed in the [Unreleased] cycle. fwap now ships four
complementary Stoneley permeability / fracture inversions:

- `stoneley_permeability_indicator` -- dimensionless fractional
  slowness shift vs a tight reference (rank-ordering only).
- `stoneley_amplitude_fracture_indicator` -- fractional amplitude
  deficit (transmission-loss form; complementary noise
  characteristics).
- `hornby_fracture_aperture` -- reflected-wave-coefficient
  inversion for fracture aperture in metres (rigid-frame, single-
  fracture limit).
- **`stoneley_permeability_tang_cheng`** *(new)* -- absolute matrix
  permeability in m^2 from the Tang-Cheng-Toksoz (1991) simplified
  Biot-Rosenbaum closed form. Real-valued inversion of the
  slowness shift; out-of-model cases (`alpha_ST <= 0` clipped to
  `kappa = 0`; `alpha_ST >= K_f / (2 K_phi)` returns NaN with a
  pointer to `hornby_fracture_aperture` for the open-fracture
  case). Validated by round-trip recovery on a Tang & Cheng 2004
  fig 5.3 synthetic (1-2 darcy bed in tight limestone). 11 tests.

The original problem statement is preserved below for reference.

**What to build**:

The closed-form low-frequency inversion of the Stoneley slowness
shift into formation permeability, from Tang–Cheng–Toksöz (1991).

*Starting point*: the observed slowness shift ``α_ST`` (dimensionless
fractional shift vs a tight reference, already computed by
:func:`fwap.stoneley.stoneley_permeability_indicator`) and a set
of Biot / fluid parameters.

*Tang–Cheng–Toksöz (1991) simplified Biot-Rosenbaum*: at angular
frequency ``ω`` well below the Biot characteristic frequency,

```
α_ST(ω) ≈ 1 / 2 · (K_f / K_φ) · (1 / (1 − i ω / ω_c))
```

where

- ``K_f``  : fluid bulk modulus (Pa)
- ``K_φ``  : frame bulk modulus of the porous formation (Pa)
- ``ω_c``  : Biot characteristic angular frequency,
             ``ω_c = η φ / (κ ρ_f)`` for dynamic fluid viscosity
             ``η``, porosity ``φ``, permeability ``κ``, fluid
             density ``ρ_f``.

Solving for ``κ`` given an observed ``α_ST(ω)`` gives the per-depth
permeability. Real and imaginary parts of ``α_ST`` carry
independent permeability information — the real part
(amplitude-based) is preferred when SNR allows.

*Implementation*:

1. Given observed ``α_ST`` (``1 - i 0`` approximation for real
   slowness shifts), invert the closed form for ``ω_c``.
2. Recover ``κ = η φ / (ω_c ρ_f)``.
3. Clip to non-negative values (noise-driven negatives are bounded
   by zero).

Public API target:

```python
fwap.stoneley.stoneley_permeability_tang_cheng(
    slowness_observed: np.ndarray,
    slowness_reference: np.ndarray | float,
    *,
    frequency: float,
    fluid_bulk_modulus: float,           # K_f in Pa
    fluid_viscosity: float,              # η in Pa·s
    fluid_density: float,                # ρ_f in kg/m³
    porosity: np.ndarray,                # φ, unitless
    frame_bulk_modulus: np.ndarray,      # K_φ in Pa
) -> np.ndarray
```

Returns permeability in m² (multiply by ``9.869e-13`` to convert
to darcies).

*Validation target*: reproduce Figure 5.3 of Tang & Cheng (2004) —
a synthetic permeable bed (1-2 darcy) sandwiched between tight
limestone (0.01-0.1 millidarcy) recovered from the Stoneley slowness
shift alone.

**Scope**: ~60 lines of code plus one validation test against the
Tang & Cheng (2004) Figure 5.3 numbers. One focused day with the
reference open.

**References**:

- Tang, X.-M., Cheng, A., & Toksöz, M. N. (1991). Dynamic permeability
  and borehole Stoneley waves: A simplified Biot-Rosenbaum model.
  *J. Acoust. Soc. Am.* 90(3), 1632-1646.
- Kostek, S., & Johnson, D. L. (1992). The interaction of tube waves
  with borehole fractures, Part I: Numerical models. *Geophysics*
  57(6), 784-795.
- Tang, X.-M., & Cheng, A. (2004). *Quantitative Borehole Acoustic
  Methods*, Section 5.1. Elsevier.

### C. Fully-joint Viterbi extensions

**Status**: closed in the [Unreleased] cycle. Both sub-items from
the original 0.4.0 roadmap are now shipped:

1. **Variable candidate budget** (done): the trellis builder
   automatically tightens per-mode top-K when the raw tuple count
   ``prod(n_i + 1)`` would exceed ``max_triples_per_depth``,
   preferring high-coherence candidates within each mode. Replaces
   the earlier hard-fail-on-overflow with graceful degradation.
   Helper ``_auto_fallback_k`` computes the largest K that fits
   the budget; ``logger.debug`` records the per-depth fallback for
   diagnostic visibility.

2. **4-mode joint Viterbi** (done): ``viterbi_pick_joint`` and
   ``viterbi_posterior_marginals`` are now N-mode generic.
   Default priors changed from the (P, S, Stoneley) subset to the
   full ``DEFAULT_PRIORS`` (4 modes); explicit subsets via
   ``priors=`` are supported for users who prefer the prior
   3-mode behaviour. The wider 4-mode trellis is kept tractable
   by the variable-candidate-budget machinery from sub-item 1.

### D. Conda-forge recipe

The package is ready for PyPI (0.4.0 builds cleanly, wheels ship
`py.typed`). A conda-forge recipe (`meta.yaml` + CI setup) can be
submitted to [staged-recipes](
https://github.com/conda-forge/staged-recipes) once the first PyPI
release is live. Reversible, low-risk; one afternoon's work.

### E. `ruff format` sweep

**Status**: closed in the [Unreleased] cycle. Tree-wide ``ruff format``
sweep applied (42 files reformatted, behaviour-preserving). New
``.pre-commit-config.yaml`` registers the ``ruff-format`` hook so
drift is prevented automatically once contributors run
``pre-commit install``. The follow-up is also done: the lint debt
(B023, B007, I001) is cleared and ``ruff-check`` now sits in the
pre-commit list alongside ``ruff-format``, together with the standard
hygiene hooks (trailing whitespace, end-of-file, YAML/TOML syntax,
merge-conflict markers, mixed line endings, large-file guard).

### F. Real-data test fixtures

Ship an anonymised reference dataset (single LAS + single SEG-Y
pulled from a public well) so the test suite includes a genuine
real-data integration test, not just synthetics. Would need
permission for redistribution; the USGS open-file datasets are
likely candidates.

**Status (partially closed)**: the *harness* now exists, and adding a
dataset is a one-entry change. `scripts/fetch_real_data.py` holds a
registry of third-party files with URL, SHA-256, provenance and
licence; `tests/test_real_data.py` runs against them and skips with an
actionable message when they are absent, so CI stays hermetic. Two
files are registered: a real Kansas Geological Survey well log (a
wrapped LAS with 26 service-company curves, which our own writer would
never emit) and a SEG-Y written by `segyio` (so a reader/writer
disagreement cannot hide behind a round-trip through our own writer).

Nothing is vendored, deliberately: the files are published by others
under their own terms -- the KGS log carries a third-party copyright
notice in its own header -- and `tests/data/real/` is git-ignored with
a test asserting it, so the no-redistribution property is enforced
rather than intended.

**What is still open, and it is the important half**: neither
registered file is a full-waveform sonic gather, because no openly
redistributable one is known to exist. So the sonic processing chain
is still validated only against synthetics.

**Priority note**: this remains the highest-value open item, and its
value grew when the `sonic_ml` layer landed (section G). Every
number that layer reports -- including the headline that a learned
inverse beats classical processing by roughly an order of magnitude
in the open hole -- is measured on data drawn from the *same forward
model* that generated the training set. That measures identifiability,
not field accuracy, and no amount of additional synthetic work can
close the gap. A single real gather with trustworthy reference picks
would say more about whether any of this transfers than another
milestone of modelling.

### G. `sonic_ml` -- the machine-learning layer

**Status**: shipped through milestones M0-M5f; see `sonic_ml.rst`
for the narrative overview and `sonic_ml/` for the package. In brief:
a torch-free spine (schema-versioned `.npz` loader, provenance,
regime-stratified splits, determinism), a model-agnostic benchmark
harness with bootstrap CIs, classical baselines, and models -- a
forward dispersion surrogate, a DL-FWI inverse net with a
heteroscedastic head, a low-latency LWD variant, in-house FNO /
DeepONet operator primitives, and a cased-hole forward operator plus
cement-bond inverse.

The layer is deliberately isolated: it is a sibling package excluded
from the core wheel and the core CI gate, running in its own
non-required workflow, and `import fwap` never pulls in PyTorch.

**Two results, and the honest gap between them.** In the open hole the
learned inverse recovers V_S roughly an order of magnitude more
accurately than classical slowness-time processing on identical
held-out gathers. Behind casing, the cement-bond inverse reaches only
about twice the skill of predicting the mean -- because a forward
sensitivity sweep shows cement stiffness moves the cased Stoneley
curve ~7% across its prior while formation V_S moves it ~1.5%, so the
problem is only partially identifiable. The uncertainty head reports
calibrated error bars that say so. Publishing only the first number
would be advertising rather than measuring.

**What's open:**

1. **Real-data evaluation** -- see section F. The binding constraint
   on every claim above.
2. **Free-pipe / leaky cased regime.** The cased dataset spans the
   *bonded* regime, where the cased Stoneley stays bound, so the bond
   inverse grades cement quality and is explicitly not a free-pipe
   detector. Reaching the debonded regime needs a leaky-mode cased
   forward model, not a planted wavetrain -- and it is also the regime
   where a CBL-amplitude baseline would finally be a fair comparison
   rather than a strawman.
3. **Two-mode cased datasets**, gated on the cased-flexural
   bracketing in section A.
4. **Surrogate-in-the-loop inversion** -- *closed*.
   `sonic_ml.models.inversion` puts the differentiable forward
   surrogate inside a multi-start gradient optimisation
   (`invert_with_surrogate`), and ships the control alongside it
   (`invert_with_solver`, the same inversion through fwap's real
   modal solver). The measured verdict is a trade, not a win: about
   ten times faster per sample, and less accurate on three of four
   parameters -- badly so for density, whose curve signature is the
   weakest of the four and so is the first casualty of the
   surrogate's own forward error. The module docstring records the
   rule that predicts this, which is reusable: expected error is
   roughly `(forward error / parameter signature) x parameter
   range`.
5. **Joint multi-depth inversion** -- *closed*.
   `sonic_ml.models.joint` uses the surrogate's gradients across a
   whole logged interval at once, penalising frame-to-frame change.
   The answer is conditional. Noise-free it buys nothing: the best
   available penalty is no penalty on almost every profile and
   parameter. The mechanism the first draft proposed -- that coupling
   averages away the surrogate's forward error -- is false, because
   inside a bed every frame has identical parameters and so draws the
   identical surrogate error, perfectly correlated with nothing to
   cancel. What coupling actually averages is observation noise, so
   at a realistic 2 us/ft of picking scatter it removes 31-45% of the
   error on all four parameters. It also beats the boring control
   that ships beside it (`smooth_independent`, a moving average over
   an independently inverted log) on all four -- by 38 points on `vs`,
   where smoothing cannot help at all, and by under 2 on `rho`, where
   the two tie. Tuned the way a user would have to tune it, the gap
   widens rather than closes: cross-validation keeps 17-29% for
   coupling and *nothing* for smoothing, which it cannot tune at all.
   The same selector fails in the other direction on a quiet log,
   over-coupling badly enough to lose 18-29% -- so the honest summary
   is that this pays on noisy picks, costs on clean ones, and cannot
   reliably tell you which you have.
6. **Bed-boundary-aware penalty** -- *closed*. `invert_joint` takes
   `penalty="tv"`, a pseudo-Huber cost that is linear beyond `tv_eps`
   and so nearly indifferent to how a given amount of change is
   distributed down the log. It exists because the squared-difference
   penalty had a measurable defect: at the cross-validated weight it
   improved a noisy log overall (`vp` MAE 506 to 420) while making the
   bed contacts *worse than not coupling at all* (486 to 500), which
   is what squaring a transition buys you. `"tv"` improves both, to
   388 and 406, and raises contact-localisation precision from 0.83 to
   0.91 against a 0.36 no-skill bar (`contact_precision` /
   `no_skill_contact_precision`).
   The finding is scoped rather than universal, and deliberately so.
   A piecewise-constant test bed is the friendliest possible setting
   for a contact-preserving prior, so `synthesize_profile` grew
   `gradation_frames` to build the hostile one; with contacts ramped
   over four frames the advantage narrows and partly inverts --- `"tv"`
   is then *worse* at the transition frames and worse at finding them.
   The default stays `"l2"` and the choice is a question about the
   rock: bedded log, use `"tv"`; gradational log, do not.
   What remains open: whether `"tv"` should become the default is a
   judgement the current evidence does not force, because it turns on
   how bedded a typical target log is --- which is a real-data
   question (section F), not one more synthetic sweep can settle.
   Coupling across *mode* as well as depth is untouched.

**Deliberately not planned**: shipping trained weights in the repo.
Checkpoints are git-ignored and the committed artefact is the small
JSON model card that binds a checkpoint to its fwap version, config
and training-data hash. Weights are cheap to regenerate and expensive
to keep honest.

## Non-goals

These have come up in reviews and been deliberately deferred:

- **GUI / plotting app**. `fwap.plotting` exposes `wiggle_plot` and
  `save_figure` for use in notebooks and scripts. A dedicated GUI is
  out of scope; integrate with Jupyter or your own plotting stack.
- **Production multi-well log management**. `fwap.io.read_las` /
  `write_las` are single-file helpers. A database / catalog layer
  belongs in a separate package.
- **Time-frequency analysis beyond the STC surface**. Wavelet
  transforms, short-time Fourier, spectrogram picking -- all useful,
  all out of scope for a reference implementation of the 1994 book.
