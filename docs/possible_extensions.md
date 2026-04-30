# Possible extensions of fwap

Companion to [`docs/roadmap.md`](roadmap.md). The roadmap tracks open
items already scoped against the four book chapters and the post-Tang
2004 literature; this file is a wider, more speculative list of
directions the package could grow in. Items are grouped by theme and
ordered roughly by how much new physics they introduce.

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
  (item 3 below) inside the modal determinant.

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

Today `fwap.anisotropy` covers Alford rotation and a Thomsen-γ
estimator from dipole + Stoneley. The `ideas/` references support
extending this from a single Thomsen parameter to a full anisotropy
toolkit.

- **VTI stiffness matrix utilities**: Thomsen ε, δ, γ ↔ C_ij
  conversions, group / phase / energy velocity surfaces, and an
  exact-vs-weak-anisotropy comparison helper.
- **HTI inversion from cross-dipole** beyond the binary
  isotropic / intrinsic / stress-induced classifier: solve for
  Thomsen-style HTI parameters (ε^(V), δ^(V), γ) from the two
  flexural dispersion curves plus Stoneley.
- **Orthorhombic synthesis**: nine-stiffness forward model and the
  three-mode (qP, qSV, qSH) phase-velocity calculator. Would let
  cross-dipole + monopole + Stoneley be jointly inverted in
  fractured reservoirs.
- **Backus averaging** for thin-layered TI from a fine-scale
  Vp/Vs/rho log; pairs naturally with `fwap.io.read_las`.

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

## 6. Time-frequency picker improvements

`fwap.coherence` plus the Viterbi pickers cover STC-based picking.
Possible additions inside the existing scope (note `roadmap.md`
non-goal #3 lists "general TF picking" as out of scope; items here
are picker-targeted, not general TF analysis).

- **CWT-augmented STC**: use a continuous-wavelet ridge as a
  candidate-seeding prior fed into the existing Viterbi trellis.
- **Frequency-banded STC**: multiple narrow-band STC surfaces
  combined with mode-dependent priors, helps in dispersive-mode
  separation where broad-band STC smears the ridge.
- **Auto-DT QC layer**: per-depth confidence score based on
  posterior marginals from `viterbi_posterior_marginals`,
  exported to LAS / DLIS as a companion uncertainty curve.

## 7. Imaging / inversion beyond Part 3

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

## 8. Cased-hole / completion-aware processing

All current modules assume open hole. A completion layer would let
the package handle the majority of real production logs:

- **Cement-bond log (CBL) amplitude + variable-density log (VDL)**
  from the same array gather.
- **Sector-bond / radial cement evaluation** using the dip-azimuth
  scaffolding from `fwap.dip` adapted to azimuthal amplitude rather
  than azimuthal time.
- **Through-tubing flexural** processing: dispersion solver in a
  three-layer (tubing-fluid-formation) geometry.

## 9. I/O & ecosystem

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

## 10. Performance / packaging

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

## 11. Validation, fixtures, and reproducibility

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

## 12. Out-of-scope but worth noting

Listed here so they are not silently re-proposed. Already declared
non-goals in `roadmap.md`:

- GUI / plotting application.
- Production multi-well log management / catalog layer.
- General time-frequency analysis beyond STC-based picking
  (wavelet families, spectrograms, etc.).
