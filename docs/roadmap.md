# fwap roadmap

A living document of open items that would meaningfully extend fwap
beyond the 0.4.0 release, ordered by estimated effort × user-value.

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

## Open items

### A. Full cylindrical-Biot dispersion solver

**Status**: fwap currently ships a phenomenological flexural
dispersion (`fwap.synthetic.dipole_flexural_dispersion`) and a
Rayleigh-speed-asymptote improvement
(`fwap.cylindrical.flexural_dispersion_physical`). Neither solves the
full 3×3 modal determinant. A production sonic-processing package
needs the Schmitt / Paillet–Cheng / Tang solver.

**What to build**:

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

### B. Quantitative Stoneley permeability

**Status**: fwap ships `stoneley_permeability_indicator` which
returns the fractional Stoneley slowness shift; the docstring is
explicit that this is a dimensionless indicator that needs
calibration to convert to SI permeability.

**What to build**:

The closed-form low-frequency inversion of the Stoneley slowness
shift into formation permeability, from Tang–Cheng–Toksöz (1991).

*Starting point*: the observed slowness shift ``α_ST`` (dimensionless
fractional shift vs a tight reference, already computed by
:func:`fwap.rockphysics.stoneley_permeability_indicator`) and a set
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
fwap.rockphysics.stoneley_permeability_tang_cheng(
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

`fwap.picker.viterbi_pick_joint` already does joint 3-mode Viterbi
across depths, ``viterbi_posterior_marginals`` runs the
forward-backward pass for per-mode posterior marginals, and the
``soft_time_order`` keyword on both turns the strict ordering
constraint into a soft penalty -- so the only remaining sub-item
from the original 0.4.0 roadmap is:

1. **Variable candidate budget**: current implementation caps valid
   triples per depth at ``max_triples_per_depth`` (default 2000) and
   raises on overflow. A smarter top-K pruning per mode (keep the K
   most coherent candidates per mode before triple enumeration)
   would handle pathological peak-heavy STC surfaces gracefully
   without raising.

A natural follow-on now that the picker supports four default modes
(P / S / pseudo-Rayleigh / Stoneley) is:

2. **4-mode joint Viterbi**: currently both `viterbi_pick_joint`
   and `viterbi_posterior_marginals` are hardcoded to the (P, S,
   Stoneley) triple and subset `DEFAULT_PRIORS` accordingly; full
   4-mode picking still falls back to the per-mode sequential
   `viterbi_pick`. Extending the trellis to quadruples squares its
   width but is otherwise mechanical.

**Scope**: each is a few-hours patch. Lower leverage than the
algorithms above because the current picker already handles the
hard-to-handle cases.

### D. Conda-forge recipe

The package is ready for PyPI (0.4.0 builds cleanly, wheels ship
`py.typed`). A conda-forge recipe (`meta.yaml` + CI setup) can be
submitted to [staged-recipes](
https://github.com/conda-forge/staged-recipes) once the first PyPI
release is live. Reversible, low-risk; one afternoon's work.

### E. `ruff format` sweep

Run `ruff format .` once across the tree as a standalone formatting
commit. Reformats ~34 files. Not enabled in pre-commit today because
the existing hand-formatted style is consistent but differs from
ruff-format's defaults (trailing semicolons, specific indent
conventions). After the sweep, add `ruff-format` to the pre-commit
hook list so drift is prevented automatically.

### F. Real-data test fixtures

Ship an anonymised reference dataset (single LAS + single SEG-Y
pulled from a public well) so the test suite includes a genuine
real-data integration test, not just synthetics. Would need
permission for redistribution; the USGS open-file datasets are
likely candidates.

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
