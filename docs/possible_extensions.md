# Possible extensions of fwap

Companion to [`docs/roadmap_old.m`](roadmap_old.m), the archived
roadmap. That file tracks open items already scoped against the four
book chapters and the post-Tang 2004 literature; this file is a wider,
more speculative list of directions the package could grow in. Items
are grouped by theme and ordered roughly by how much new physics they
introduce.

## 1. Cylindrical-Biot solver completions

The bound-mode n=0 (Stoneley) and n=1 (flexural) solvers ship today.
Natural follow-ons:

- **Leaky-mode regime** (already roadmap item A): complex-`k_z`
  Mueller iteration, outgoing Hankel BCs, branch tracking across the
  pseudo-Rayleigh / leaky-flexural cutoffs.
- **n=2 quadrupole bound-mode solver**: 4×4 modal determinant; the
  `fwap.lwd` quadrupole pipeline currently relies on a phenomenological
  prior, and a real solver would let LWD slowness be inverted rather
  than picked.
- **Layered (cased / multi-layer-tool) extension**: replace the single
  fluid-solid interface with a stack of annular regions
  (mud → casing → cement → formation), propagator-matrix style.
  Required for any cased-hole interpretation.
- **Anisotropic borehole modes**: dispersion in a TI / orthorhombic
  formation. Requires the Tsvankin 2011/2012 stiffness machinery
  (section 3 below) inside the modal determinant.

## 2. Poroelastic forward modelling (Carcione 2022)

`ideas/Carcione2022.docx` covers wave propagation in poroelastic media
beyond the simplified Biot-Rosenbaum form already shipped as
`stoneley_permeability_tang_cheng`.

- **Full Biot fast-/slow-P + S synthesis** in a homogeneous porous
  half-space, returning the three poroelastic body-wave slownesses
  and amplitudes versus frequency.
- **Biot-Rosenbaum borehole Stoneley** (no low-frequency
  approximation): full complex-α inversion that keeps the imaginary
  part of the slowness shift, recovering both permeability and
  poroelastic Q in one pass.
- **Squirt-flow / BISQ correction layer**: empirical Dvorkin-Nur
  squirt term on top of Biot, addresses the well-known
  underprediction of attenuation at sonic frequencies.

## 3. Anisotropic rock physics (Thomsen 2014, Tsvankin 2011/2012)

`fwap.anisotropy` today covers Alford rotation, the Thomsen-γ
estimator from dipole + Stoneley, the full VTI moduli summary
(C_33 / C_44 / C_66 + γ) via `vti_moduli_from_logs`, Thomsen ε / δ
from walkaway-VSP slowness-polarization, Backus averaging, and the
Christoffel phase / group velocities (qP, qSV, SH). The themes below
extend that toolkit toward the full Thomsen 2014 + Tsvankin 2011/2012
view of borehole anisotropy.

- **VTI stiffness matrix utilities**: explicit Thomsen ε, δ, γ ↔ C_ij
  conversions (the Backus, Christoffel, and ε/δ inverters all carry
  pieces of this; pulling it into a single public converter would
  remove duplication), the energy-velocity surface alongside the
  shipped phase / group surfaces, and an exact-vs-weak-anisotropy
  comparison helper so callers can audit when the linearised inverters
  break down.
- **η anellipticity attribute**: Thomsen 2014 calls
  `η = (ε − δ) / (1 + 2δ)` the *truly seismic-relevant* combination
  and Tsvankin 2012 ch. 6 shows P-wave time processing in VTI media
  needs only `(Vnmo, η)`. Trivial derived quantity; expose it as a
  property on `ThomsenEpsilonDeltaResult` and `VtiModuli`.
- **Anisotropic Gassmann fluid substitution** (Brown-Korringa /
  Thomsen 2018): Thomsen 2014 Lecture 3 explicitly flags that the
  textbook isotropic Biot-Gassmann formula is *formally* wrong in
  anisotropic rocks because the bulk modulus does not appear naturally
  in the anisotropic P-wave velocity. `fwap.rockphysics.gassmann_fluid_substitution`
  silently returns wrong numbers when handed a VTI rock; add the
  anisotropic counterpart and a docstring warning on the isotropic
  function.
- **TTI (tilted transverse isotropy)**: VTI assumes the symmetry axis
  is aligned with the well. For deviated wells in dipping shales --
  the routine case in industrial depth imaging since ~2010 (Tsvankin
  2012) -- the axis is tilted, and a single rotation parameter
  (`tilt_deg`, `azimuth_deg`) plus a 3×3 stiffness rotation in front
  of the existing Christoffel evaluator and the cylindrical-solver TI
  variants would close the gap.
- **HTI inversion from cross-dipole** beyond the binary
  isotropic / intrinsic / stress-induced classifier: solve for
  Thomsen-style HTI parameters (ε^(V), δ^(V), γ) from the two
  flexural dispersion curves plus Stoneley.
- **Fracture effective-media bridge** (Schoenberg linear-slip;
  Hudson penny-shaped crack; Kachanov non-interaction): Tsvankin &
  Grechka 2011 argue this is the missing link between *inverted*
  Thomsen parameters and *physical* fracture characterisation (crack
  density, normal/tangential compliance, fluid type). The 2011 book
  also makes the durable point that realistic non-aligned,
  non-identical fracture sets produce orthorhombic effective media
  even when each set is HTI. New `fwap.anisotropy.fractures`
  submodule with `linear_slip_compliances_from_hti`,
  `hudson_crack_density`, `kachanov_non_interaction`, sitting
  downstream of the HTI inverter.
- **Orthorhombic synthesis**: nine-stiffness forward model and the
  three-mode (qP, qSV, qSH) phase-velocity calculator. Would let
  cross-dipole + monopole + Stoneley be jointly inverted in
  fractured reservoirs; pairs naturally with the fracture
  effective-media bridge above.
- **Polar vs azimuthal docs separation**: Thomsen 2014 Lecture 1
  insists that polar (VTI/TTI) and azimuthal (HTI/orthorhombic)
  anisotropy arise from different physical mechanisms and respond to
  different acquisition geometries, so blurring them under a generic
  "anisotropic" label is unhelpful. fwap's docs currently use
  "anisotropy" generically; a short narrative section in
  `docs/chapter_map.rst` (or the `fwap.anisotropy` package docstring)
  labelling each shipped symbol with its anisotropy family would help
  new users avoid mis-applying e.g. Alford rotation to a VTI shale.

## 4. Attenuation / Q processing depth

`fwap.attenuation` ships centroid-shift and spectral-ratio Q. Useful
extensions:

- **Frequency-dependent Q(f)**: replace the constant-Q fit with a
  Kjartansson / SLS power-law model.
- **Q tomography**: reuse the `fwap.tomography` intercept-time
  scaffolding to invert per-layer Q from the spectral-ratio surface,
  not just per-receiver-pair Q.
- **Joint amplitude + dispersion Q**: Kramers-Kronig-consistent
  inversion that uses the slight velocity dispersion implied by the
  measured attenuation, improving Q stability in noisy gathers.

## 5. Dispersion-curve inversion utilities

The dispersion module computes forward curves; an inversion layer
would round it out:

- **Dipole-flexural inversion** for V_S(depth) given the measured
  dispersion (currently picked off the STC surface with no formal
  inversion).
- **Stoneley dispersion inversion** for combined permeability +
  fluid bulk modulus given two-frequency band picks.
- **Joint multi-mode inversion**: simultaneous fit of Stoneley +
  flexural + pseudo-Rayleigh dispersion curves to a single
  (V_P, V_S, ρ, a_borehole, V_fluid) tuple. The bound-mode solver
  already provides the forward operator.

## 6. Walkaway-VSP anisotropy inversion

`ideas/Articles.docx` reviews the modern walkaway-VSP anisotropy
literature as a coordinated three-paper set (Grechka & Mateeva 2007
Geophysics; Grechka & Mateeva et al. 2007 The Leading Edge; Liu et al.
2014 Applied Geophysics) and frames it as two complementary inversion
families. fwap ships the slowness-polarization half (Horne & Leaney
2000 weak-anisotropy linearisation via
`thomsen_epsilon_delta_from_walkaway_vsp`); the rest is open.

- **Hodogram polarization preprocessing**: the inversion takes the
  P-wave polarization unit vector as an input the caller must
  compute. `Articles.docx` calls polarization-angle estimation "the
  method's Achilles' heel" (Danek et al. 2018 Wysin-1 case study).
  Add `fwap.dispersion.hodogram_polarization` (or a new `fwap.vsp`
  package) that takes a 3-component VSP gather, picks first-break
  windows, computes the eigenvector of the particle-motion covariance
  matrix, and returns per-shot `(slowness, polarization)` pairs ready
  for the existing inverter.
- **Traveltime-based VSP inversion** (Liu et al. 2014, Applied
  Geophysics 11(1)): an independent route to ε, δ that uses only
  first-arrival traveltimes (the cleanest measurement on a VSP
  record) and is robust to noisy polarizations but depends on the
  overburden velocity model. Near-offset NMO velocity correction
  fixes δ, far-offset Thomsen ray-tracing fixes ε. Add
  `thomsen_epsilon_delta_from_walkaway_vsp_traveltime`; reuses the
  `fwap.tomography` least-squares scaffolding.
- **Wide-azimuth orthorhombic VSP inversion** (Grechka & Mateeva
  2007; Abedi et al. 2019 South Pars): two perpendicular walkaway
  lines recover up to seven Tsvankin-style orthorhombic parameters
  at the receiver level. Stretch goal; depends on the orthorhombic
  forward model and the fracture effective-media bridge in section 3.
- **Exact-Christoffel slowness-polarization inverter**: the shipped
  function uses the Thomsen 1986 weak-anisotropy linearisation. Abedi
  et al. 2019 compare to the exact (q, ψ) curve from Christoffel and
  report differences worth quantifying. Adds an
  `exact_christoffel=True` mode alongside the linearised one.

## 7. Time-frequency picker improvements

`fwap.coherence` plus the Viterbi pickers cover STC-based picking.
Possible additions inside the existing scope (note `roadmap_old.m`
non-goal #3 lists "general TF picking" as out of scope; items here
are picker-targeted, not general TF analysis).

- **CWT-augmented STC**: use a continuous-wavelet ridge as a
  candidate-seeding prior fed into the existing Viterbi trellis.
- **Frequency-banded STC**: multiple narrow-band STC surfaces
  combined with mode-dependent priors, helps in dispersive-mode
  separation where broad-band STC smears the ridge.
- **Weighted spectral semblance**: Tang & Cheng 2004 ch. 3 develops
  this as a less-biased alternative to uniform-weight STC when the
  picked mode is dispersive (dipole flexural in particular). Variant
  of STC with a per-frequency amplitude weighting derived from the
  array spectrum. Worth benchmarking against `dispersive_stc` to see
  whether it subsumes one or both.
- **Mode inventory / classifier**: Paillet & Cheng 1991's organising
  claim is that every borehole-acoustic interpretation reduces to
  identifying which mode carries the diagnostic information. The
  pieces are in fwap (`fwap.picker`,
  `fwap.dispersion.classify_flexural_anisotropy`) but no top-level
  diagnostic exists. New `fwap.coherence.mode_inventory(stc, prior)`
  that labels each STC ridge as P-head / S-head / pseudo-Rayleigh /
  Stoneley / leaky-Rayleigh / collar-mode; doubles as a teaching
  artefact for the cylindrical-Biot notebook.
- **Cross-mode QC bundle**: Mari et al. 1994's running quality
  philosophy is *cross-consistency between modes* (does the dipole γ
  match the Stoneley-derived γ? does Vp/Vs from monopole agree with
  dipole shear?). fwap has the per-mode QC flags but no unified
  report. `fwap.picker.quality.cross_mode_consistency_report(track)`
  returns a structured per-depth card.
- **Auto-DT QC layer**: per-depth confidence score based on
  posterior marginals from `viterbi_posterior_marginals`,
  exported to LAS / DLIS as a companion uncertainty curve.

## 8. Stoneley fracture and permeability processing

Quantitative Stoneley applications already shipped: `fwap.stoneley`
covers slowness / amplitude indicators, the Tang-Cheng-Toksoz
permeability inversion, the Hornby aperture inversion from a single
reflection coefficient, and the unified `stoneley_fracture_density`
combiner. `ideas/Tang2004.docx` flags an event-detection layer that
the per-event aperture inverter does not provide.

- **Stoneley fracture locator from reflections**: Tang & Cheng 2004
  ch. 4 develops the picture in which Stoneley waves reflect off
  individual open fractures and produce characteristic
  time-distance loci that let each fracture be located along the
  wellbore *and* assigned an aperture. fwap inverts an aperture from
  one |R| value but has no sweep that detects the events in the
  first place. New `fwap.stoneley.stoneley_fracture_picker`
  searches a depth × time Stoneley gather for reflection
  hyperbolae and returns per-fracture
  `(depth, aperture, reliability)`; pairs naturally with the
  existing Hornby aperture inverter.
- **Mudcake / elastic-property correction layer**: Tang & Cheng 2004
  ch. 4 emphasises that *apparent* Stoneley attenuation has three
  causes (true permeability, mudcake / borehole irregularity,
  elastic-property contrast) and that practical inversions need a
  decomposition step before the Biot-Rosenbaum inverter runs. fwap
  applies the Biot-Rosenbaum form directly; an upstream classifier
  that separates the contributions would tighten the
  permeability-only output.

## 9. Imaging / inversion beyond Part 3

The intercept-time + dipole-flexural pipeline can be deepened:

- **Full-waveform inversion (FWI) for the altered zone**: replace
  the closed-form delay-to-thickness inversion with a 1-D radial
  FWI driven by the existing synthetic-gather forward model.
- **Refraction tomography in 2-D** along the well: lateral V_P /
  V_S variation over a measured interval, not just the per-depth
  layer-cake from `fwap.tomography`.
- **Reflection imaging from the monopole gather**: the late-time
  energy past Stoneley contains weak P-P and P-SV reflections from
  bed boundaries within ~1 m of the wellbore. Migration produces
  a thin near-wellbore image (BARS / BHTV-style).

## 10. Cased-hole / completion-aware processing

All current modules assume open hole. A completion layer would let
the package handle the majority of real production logs:

- **Cement-bond log (CBL) amplitude + variable-density log (VDL)**
  from the same array gather.
- **Sector-bond / radial cement evaluation** using the dip-azimuth
  scaffolding from `fwap.dip` adapted to azimuthal amplitude rather
  than azimuthal time.
- **Through-tubing flexural** processing: dispersion solver in a
  three-layer (tubing-fluid-formation) geometry.

## 11. I/O & ecosystem

- **WITSML / OSDU** read path. Several operators have moved log
  archives off LAS/DLIS into WITSML 2.x or OSDU; a thin reader on
  top of `fwap.io` would broaden adoption.
- **xarray-backed gather container**: optional `fwap.io.read_xarray`
  that wraps a SEG-Y or DLIS gather as a labelled `Dataset`
  (offset, depth, time, frequency dims) without forcing xarray as a
  hard dependency.
- **Parquet curve store** for batch processing across a field;
  preserves dtypes/units that LAS lossily flattens.
- **CLI batch mode**: `fwap process --batch wells/*.sgy
  --out parquet/` with parallel-per-well dispatch.

## 12. Performance / packaging

- **Numba / Cython hot paths** for `stc`, `tau_p_*`, and the
  cylindrical-Biot determinant evaluator. Profiles show STC is the
  dominant per-depth cost.
- **GPU back-end (CuPy)** behind a `fwap.set_backend("cupy")`
  switch, drop-in for the NumPy array operations in `coherence`,
  `wavesep`, and `tomography`.
- **Conda-forge recipe** (roadmap item D) — the path of least
  resistance for downstream packaging.
- **Wheel/PyPI release** of 0.4.0, prerequisite for the conda
  recipe.

## 13. Validation, fixtures, and reproducibility

- **Real-data fixtures** (roadmap item F): a USGS or Volve-style
  open-license well, single LAS + single SEG-Y, wired into a
  dedicated `tests/test_real_data.py`.
- **Reference-figure regeneration scripts**: each demo writes its
  diagnostic figure and the docs link to the produced image; the
  Sphinx build could regenerate these on every commit so the docs
  cannot drift from the implementation.
- **Cross-validation against published curves**: notebook that
  reproduces Paillet & Cheng 1991 fig 4.5, Schmitt 1988 fig 4,
  Tang & Cheng 2004 figs 3.4 and 5.3; would double as the
  acceptance test for the leaky-mode solver work.

## 14. Out-of-scope but worth noting

Listed here so they are not silently re-proposed. Already declared
non-goals in `roadmap_old.m`:

- GUI / plotting application.
- Production multi-well log management / catalog layer.
- General time-frequency analysis beyond STC-based picking
  (wavelet families, spectrograms, etc.).
- **PS converted-wave processing**: Thomsen 2014 Lecture 5 and
  Tsvankin 2012 ch. 5 develop PS / C-waves at length, but the
  geometry (surface source, surface receiver, mode-converted at
  depth) is squarely surface seismic. Borehole PS (multi-component
  VSP recording downgoing P + upgoing converted S) would be in
  scope via the VSP path of section 6, but generic surface PS
  processing is not.
- **Nonhyperbolic surface-seismic moveout / η picking from CMP
  gathers**: Tsvankin 2012 ch. 4 is the canonical reference but the
  acquisition is surface seismic, not borehole. The η attribute on
  the borehole-derived `ThomsenEpsilonDeltaResult` (section 3) is
  the in-scope subset.
