# Changelog

All notable changes to this project are documented here. The format
loosely follows [Keep a Changelog](https://keepachangelog.com/), and
the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- **The crack-wave ceiling is 84 kHz, not ~240** (roadmap A.5 residue). The 240
  figure came from arithmetic on a constant — `_BESSEL_ARG_MAX * V_f / (2 pi r)`
  — rather than from running the solver, and the test that recorded it asserted
  that arithmetic and then checked a frequency well above the real ceiling, so
  both halves passed while measuring nothing.

  `_BESSEL_ARG_MAX` is also not what holds the ceiling down: raised fourfold, it
  moves from 84 kHz to 84 kHz. What binds is the **product**. The determinant
  turns non-finite over the bottom ~16 % of the scan window while every input is
  still fine — parts finite, fluid Wronskian exact to 2e-16 — and that floor
  climbs with frequency faster than the crack root does. At 84 kHz the root sits
  0.3 % above it; two kilohertz later it is underneath. `|E_form|` reaches
  **1.15e150** against a `sqrt(DBL_MAX)` headroom of 1.34e154, and widening the
  window makes `_layer_propagator_n0` overflow in `matmul` outright.

  The cancellation the grid-stability filter works around is real and separate:
  `cond(P_outer)` runs **1e35–1e40**.

  The reformulation A.5 prescribes still stands — a compound-matrix form
  addresses overflow and conditioning together. What is now ruled out is the
  cheap version: equilibrating to dodge the overflow alone would widen the
  spurious-root zone rather than the usable band, because overflow is currently
  acting as a safety net.

### Added
- **The fixture registry now checks its own unverified claims** (roadmap F.4).
  Two entries — `forge_dsi_las` and `iodp_u1347a_dsi` — carry digests computed
  from copies that never came down their canonical URLs, and that was recorded
  only as a `CHECKSUM CAVEAT` paragraph in prose. It is now a
  `checksum_confirmed` field as well, with `check_registry_caveats()` raising if
  the flag and the paragraph disagree, `--list` printing
  `CHECKSUM UNCONFIRMED AGAINST URL` on the header line, and the unconfirmed set
  pinned in `tests/test_real_data.py`. A successful fetch of an unconfirmed entry
  now prints the three edits that clear it, because that download succeeds and
  matches and otherwise looks like any other green run.

  The guarded failure is the quiet one: someone confirms a digest, clears the
  flag, leaves the paragraph — and the registry then reads as verified where it
  is not.

  **The fetches themselves remain open, and the blocker was re-measured rather
  than assumed.** `gdr.openei.org` and `zenodo.org` are both refused at the
  network gateway (403 to CONNECT, policy denial) while ordinary HTTPS works, so
  this is the same obstruction that created F.4. What could be settled was: all
  three digests match the copies in hand — 606 016 251 bytes of the IODP
  archive, 21 435 504 of its member, 3 001 504 of the FORGE LAS — so none is a
  transcription error. What is unverified is the provenance of the bytes, not
  the arithmetic over them.

- **The cased hole is tied to published curves for the first time** (roadmap A.1).
  Every cased-hole number in the suite had been scored against `fwap` itself —
  A.9's leaky branch was validated against the bound solver it takes over from
  because no published cased curve had been read, and A.7's screw path had been
  silent. Schmitt & Cheng's figures 20 and 21 are cased-hole dispersion for the
  dipole and the screw, and `plans/guides.md` §11 had listed them as unread.

  Digitised at 600 dpi, with Table 1's own casing and cement rows read from the
  page (casing 6098/3354/7500, cement 1 2823/1729/1920, cement 2 2823/1555/1730 —
  the cased fixtures elsewhere in the suite use invented values):

  | figure | case | median | worst | coverage |
  | --- | --- | --- | --- | --- |
  | 20a flexural | open hole (anchor) | 0.28 % | 0.63 % | 18/18, 6.5–15 kHz |
  | 20a flexural | casing + 1 cm cement | **0.21 %** | 0.72 % | 23/23, 4–15 kHz |
  | 20a flexural | casing + 3 cm cement | **0.23 %** | 0.50 % | 23/23, 4–15 kHz |
  | 21a screw | open hole (anchor) | 0.26 % | 1.67 % | 13/13, 8–20 kHz |
  | 21a screw | casing + 1 cm cement | **0.82 %** | 1.76 % | 10/10, 8–20 kHz |
  | 21a screw | casing + 3 cm cement | **0.27 %** | 1.04 % | 10/10, 8–20 kHz |

  The flexural tie is at the digitisation floor — the residual is the same size
  as the open-hole anchor's. Figure 21 is the one §11 called *"the only external
  measure of how wrong that path was"* for A.7; before A.7 the configuration
  returned nothing at all, so there was no number to take.

  **The geometry is the thing to get right**, and it is quoted rather than
  inferred: p. 230 says the inner borehole radius is *decreased* by the casing
  and cement, so the 10 cm radius is the formation contact and the 3 cm-cement
  case has `a = 5.98 cm`.

  **The anchor earned its keep.** The first trace of figure 20's open-hole curve
  jumped onto a steeper neighbour through the knee. It showed as −10 % against
  the open-hole solver, and an independent kink test — looking for the slope
  discontinuity a curve-jump leaves — put the jump at 6.32–6.35 kHz without
  reference to `fwap` at all. That curve is recorded only above 6.5 kHz.

  What still needs the books is **VTI flexural**, and it needs a different paper:
  Schmitt (1989). This one is isotropic throughout.

### Added
- **`flexural_dispersion_vti` has an external check for the first time**, and
  the validation notebook's section 5 is a real overlay rather than a
  picture. Three curves traced from Ellefsen, Cheng & Schmitt (1988), MIT
  ERL ([DSpace](https://dspace.mit.edu/handle/1721.1/75100)):
  - `..._fig4_flexural_vti_soft.csv` — **0.30 % RMS**, worst 1.29 %, 70/73
    points over 2.25-19.5 kHz, against `flexural_dispersion_vti`.
  - `..._fig4_flexural_iso_soft.csv` — **0.17 % RMS**, worst 0.34 %, 73/73
    points, against `flexural_dispersion`.
  - `..._fig2_flexural_iso_hard.csv` — **0.45 % RMS**, worst 0.91 %, but
    only **17/73** points over 2.0-6.0 kHz: the fast-formation path returns
    `NaN` outside that band, so the overlay is *silent* there, not green.
  Each figure plots a TI formation against an *equivalent isotropic* one
  defined to share its vertical velocities, which is why one trace scores
  two different solvers.
  **The geometry was assembled from three documents and checked before
  use**, because the Ellefsen report states no numbers at all — no elastic
  constants, no borehole radius, no fluid properties, its fig 1 a labelled
  schematic. The rocks come from Thomsen (1986) table 1: Green River shale
  (Schock et al. row) 3292 / 1768 m/s, 2075 kg/m^3, `eps` 0.195,
  `delta` -0.220, `gamma` 0.180; shale (5000) (Jones & Wang row)
  3048 / 1490 m/s, 2420 kg/m^3, `eps` 0.255, `delta` -0.050,
  `gamma` 0.480.
  **Two independent confirmations, both set up before the table was
  opened.** The curves were traced first, and their low-frequency limits —
  where the flexural branch tends to `V_S` — read **1775** and **1488** m/s
  against the table's 1768 and 1490: **+0.4 %** and **+0.13 %**. That also
  discriminates between table 1's two Green River shale entries, the other
  (Podio et al.) sitting at `V_S0` = 2432 m/s. Tsvankin's figure 1.12
  gives the same row's `eps` = 0.195 and `delta` = -0.22 from a third
  document. **And the borehole radius was a prediction**: `a` = 0.10 m
  appears nowhere in the report, was taken from Schmitt's companion ERL
  reports, used untuned, and all three overlays then landed under 0.5 %.
  A wrong radius shifts the knee in frequency and would have failed loudly.
- **The fourth branch is blocked on the solver, not the reference**, and is
  recorded that way. `flexural_dispersion_vti` raises
  `NotImplementedError` for fast-formation TI (`V_Sv` > `V_f`), which is
  exactly Green River shale at 1768 against a 1500 m/s fluid, so fig 2's TI
  branch cannot be scored. Its traced curve and its now-verified geometry
  both wait in `docs/notebooks/_data/pending/`; promoting it needs no new
  digitising and no new literature, only the H.d follow-up. Worth knowing
  before that starts: the *isotropic* fast path is itself sparse — 17 of 73
  points on the same rock — so a TI path inheriting that behaviour would be
  scored just as thinly.

### Added
- **Cased-hole Stoneley is tied, and `pseudo_rayleigh_dispersion` fails its
  first external check.** Six curves traced from one figure — Tubman, K. M.,
  Cheng, C. H., & Toksoz, M. N. (1984), *Synthetic full waveform acoustic
  logs in cased boreholes*, *Geophysics* **49**(7), 1051-1059,
  [10.1190/1.1441720](https://doi.org/10.1190/1.1441720), fig 4: phase
  velocity for the open and cased geometries, Stoneley plus two
  pseudo-Rayleigh modes each.
  - `..._fig4a_stoneley_open.csv` — **2.23 % RMS**, 67/67 pts, against
    `stoneley_dispersion`.
  - `..._fig4b_stoneley_cased.csv` — **2.34 % RMS**, 43/43 pts, against
    `stoneley_dispersion_layered`. **This closes the last cased mode with no
    external tie.**
  - `..._fig4a_pseudo_rayleigh{1,2}_open.csv` — **FAIL at 35.96 % and
    50.99 %**, marked `known_defect=`.
  - The two cased pseudo-Rayleigh curves are traced and parked in
    `_data/pending/`; the package exposes no cased pseudo-Rayleigh API.
  **The pseudo-Rayleigh failure is unambiguous and is the solver.** For this
  geometry `pseudo_rayleigh_dispersion` returns phase velocities of
  **1.65-2.24 `V_f` against a `V_S / V_f` = 1.551 bound** — a
  pseudo-Rayleigh mode is trapped between the fluid and shear speeds by
  definition, so a root above `V_S` is not a guided mode at all. Both
  branches also return the same value at 10 kHz, so branch selection is
  suspect too. **The control is that both Stoneley overlays pass on exactly
  the same parameters**, which rules out the geometry. The `known_defect=`
  marker is back, and still inverts the assertion rather than relaxing it.
  **The ~2.3 % on the Stoneley ties is expected, not slack digitising.**
  Tubman's table 1 carries `Q`, the fluid at `Q_alpha` = 20, so the published
  curves include intrinsic attenuation while these solvers are elastic. An
  elastic solver runs faster than a `Q` = 20 medium here, and both overlays
  come in ~2.3 % high with the same sign.
  **Four candidate references were checked and rejected before this one** —
  Schmitt 1988.13 figs 59/66 (TI *poroelastic*), Xie 2018 (right geometry
  and a full parameter table, but the figure is 256x237 px native, so
  tracing error would rival the 5 % budget) and Karpfinger 2010 (no casing
  or cement anywhere in it). Tubman's panel is 986x583 px in a 300 dpi scan.
  **It also settles two things the withdrawn Tang & Cheng citation had left
  wrong.** The radius convention: the fluid "thickness" *is* the fluid
  radius, and the cased layers stack outward to the same 4.0 in formation
  contact as the open hole (1.85 + 0.4 + 1.75 = 4.0), where the withdrawn
  geometry had put `a` = 0.10 m *inside* the casing and cement. And the
  casing values: Tubman's steel (6096/3352.8/7500) and cement
  (2822.4/1728.2/1920) are Schmitt 1988.13 table 8 to the digit, and the
  formation (4876.8/2599.9/2160) is Schmitt & Cheng 1987's fast sandstone —
  so the withdrawn 5860/3140/7800 casing disagreed with three independent
  sources.
  **One trim worth recording**: below ~13.5 kHz the open and cased Stoneley
  curves overlap within line width in the printed figure, and the traced
  values agree to within +-3 m/s there. The cased CSV therefore starts at
  14 kHz rather than carrying open-hole values under a cased label.

### Fixed
- **The Tang & Cheng (2004) figure numbers are withdrawn: that book has six
  chapters, so "fig 7.1" never existed.** Confirmed against a physical copy.
  The book is *Quantitative Borehole Acoustic Methods*, Handbook of
  Geophysical Exploration vol. 24, Elsevier, 274 pp., and its contents are
  1 Overview, 2 Elastic Wave Propagation in Boreholes, 3 Velocity and
  Attenuation Estimation from Array Acoustic Waveform Data, 4 Permeability
  Estimation, 5 Anisotropic Formations, 6 Summary. Two separate claims fail:
  - **"fig 7.1 — cased-hole Stoneley"** and the geometry quoted from
    **"sect. 7.2"** cannot exist. Validation-notebook section 4's formation,
    casing, cement and radius are therefore *entirely unsourced* — and its
    casing (5860/3140/7800) matches neither real casing row in the Schmitt
    ERL reports (6098/3354/7500, 6096/3352/7500). That is the **third**
    invented geometry found in this notebook, after the "shale at
    2740/1280/2400" of section 2 and the "Schmitt 1989 sect. 4 example" of
    section 5.
  - **"figs 3.7 / 3.10 — quadrupole slow + fast"**: both figures exist and
    neither is a dispersion curve. **Fig 3.7 is waveform matching, fig 3.10
    is acoustic time delay**, and chapter 3 is a *processing* chapter.
  **The nearest candidate for the quadrupole section has since been located
  and is not usable either.** Chapter 2 carries **fig 2.11, "Analysis of
  dipole and quadrupole waves in the logging-while-drilling
  configuration"** — a **figure of principle**, schematic rather than a
  quantitative dispersion curve, so nothing in the
  `freq_hz, slowness_s_per_m` schema can come out of it. Quadrupole
  dispersion therefore needs a source outside Tang & Cheng (2004)
  entirely; that search is closed rather than merely open. For cased-hole
  Stoneley no candidate has been identified in that book at all.
  Withdrawn in nineteen places across `fwap/cylindrical_solver/`
  (`_cased.py`, `_n0_layered.py`, `_n1_layered.py`, `_n2_quadrupole.py`),
  `tests/test_cylindrical_solver.py`, `plans/roadmap.md`, four
  `docs/plans/cylindrical_biot*.md`, `docs/notebooks/_data/README.md` and
  the notebook. **No replacement figure numbers are asserted**, because
  nobody has read the relevant chapters — inventing a plausible pointer is
  how this started.
  **The correct chapter list was already in the tree.** `docs/ideas/Tang2004.md`
  says "The book is organized into six chapters" and lists them accurately.
  That is the second time the right answer was already present while other
  files carried a wrong one; the Schmitt 1988 page range was the first,
  correct in `fwap/validation.py` and wrong in thirteen other places.
  **And the discrepancy had already been noticed once, then resolved the
  wrong way.** `plans/roadmap.md` recorded that the roadmap said "Fig. 3.4"
  while the notebook said "figs 3.7 / 3.10 ... 7.1", and concluded "the
  notebook is the accurate list". Both were guesses; the disagreement was
  the signal and it was spent on picking a side instead of opening the book.
  That note now says so.
  **Consequence for coverage.** Cased-hole Stoneley is the one cased mode
  with no external tie: roadmap A.1 tied cased flexural and screw from
  Schmitt & Cheng 1987 figs 20/21, the monopole case is not in that report,
  and the cased Stoneley figures in Schmitt 1988.13 (figs 59, 66) are
  transversely isotropic *poroelastic*, which `stoneley_dispersion_layered`
  cannot reproduce. Section 3's quadrupole geometry is likewise unverified.
  Neither section scores anything today, so no published number moves.
  **Still unchecked, and left alone deliberately**: references to chapters
  that do exist — `sect. 2.4-2.5` (LWD, in `fwap/__init__.py`, `lwd.py`,
  `demos/_extensions.py`, `docs/index.rst`, `docs/chapter_map.rst`),
  `sect. 5.3-5.4` (anisotropy, in `anisotropy/_thomsen.py`,
  `_vti_inversion.py`, `dispersion.py`), `ch. 3`, `fig 3.4` and `fig 5.3`.
  These are plausible but unverified, and are not being mass-edited on
  suspicion.

- **"Schmitt 1989 fig 5" is a monopole shot gather, and three other things
  the VTI validation section claimed are also untrue.** Checked against the
  open-access ERL precursor (Schmitt, 1988.13, *Transversely Isotropic
  Saturated Porous Formations II*, [DSpace](https://dspace.mit.edu/handle/1721.1/75108)):
  - **Fig 5 is not a dispersion figure.** Four monopole microseismograms
    against time (0-6 ms, `z` = 5 m, 1 kHz source) comparing permeability
    cases at an impermeable wall. The flexural dispersion in that report is
    fig 22. This is the *second* time a figure number in this repository
    turned out to name a shot gather; the first was "Schmitt 1988 fig 4".
  - **There is no qP/qSV branch pair.** `flexural_dispersion_vti` returns
    one `BoreholeMode`, and the literature comparison is TI against
    *equivalent isotropic*, a different axis entirely. The
    `..._qSV.csv` row in `_data/README.md` was an orphan no cell read.
  - **There is no flexural splitting.** A vertical borehole in a VTI medium
    whose symmetry axis is parallel to the borehole is azimuthally
    isotropic, so the flexural mode has nothing to split into. Splitting
    needs azimuthal anisotropy, which this geometry does not have.
  - **Schmitt 1989 is poroelastic** — Biot two-phase with a permeability
    tensor — while `flexural_dispersion_vti` is elastic. Scoring one
    against the other would be a category error even with the right figure.
  The geometry the section carried (C_11 23.2, C_13 9.0, C_33 18.0, C_44
  4.0, C_66 6.0 GPa, rho 2400) was attributed to "Schmitt 1989, sect. 4
  example"; that attribution is withdrawn and the numbers are now marked
  unverified and illustrative, exactly as the invented shale/limestone
  geometry in section 2 was.
  **The elastic reference is Ellefsen, Cheng & Schmitt (1988)**, MIT ERL
  ([DSpace](https://dspace.mit.edu/handle/1721.1/75100)), figs 2 (hard) and
  4 (soft) — elastic VTI, symmetry axis parallel to the borehole, each
  figure plotting the TI formation against its equivalent isotropic one, so
  a single trace would tie both `flexural_dispersion_vti` and
  `flexural_dispersion`.
  **Still blocked, and recorded as blocked rather than skipped.** That
  report states no numbers at all: constants deferred to Thomsen (1986)
  (Green River shale; shale (5000)), and no borehole radius or fluid
  properties anywhere — its fig 1 is a labelled schematic. The radius could
  be inferred from the figure itself, and deliberately was not: fitting the
  geometry to the curve the overlay then scores against is the silent refit
  `_data/README.md` exists to prevent.

### Added
- **Four flexural curves traced from Ellefsen, Cheng & Schmitt (1988) and
  parked unscored** in `docs/notebooks/_data/pending/`: both branches of
  fig 2 (hard) and fig 4 (soft), 73 points each over 1.5-19.5 kHz, at
  400 dpi. They are deliberately *not* in `_data/` proper, because a CSV
  there is picked up by `check_overlay` on sight and asserted against the
  5 % budget — and scoring against a geometry nobody has verified produces
  a number that looks like validation and is not. Nothing reads
  `pending/`; its README states what is missing and how to promote them.
  Each figure plots the TI formation against an *equivalent isotropic* one
  defined to share its vertical P and S velocities, so each pair ties two
  solvers from one trace — the TI branch scores `flexural_dispersion_vti`,
  the isotropic branch `flexural_dispersion`.
  **The trace carries its own consistency check.** On each figure the two
  branches must converge to the same low-frequency limit, because that
  limit is `V_S` and the equivalent isotropic formation is defined to share
  it. Traced independently — one solid curve, one dashed — they agree to
  **0.1 m/s** on fig 2 (1775.0 against 1775.1) and to **0.0 m/s** on fig 4
  (1488.0 both). Those two numbers are also the anchors that will identify
  the right Thomsen rows: if Green River shale does not give `V_S0` ≈ 1775
  m/s, the hard formation is not Green River shale. Tsvankin's fig 1.12
  supplies a second check for the hard rock alone, `epsilon` = 0.195 and
  `delta` = -0.22 — his monograph turns out not to reproduce Thomsen's
  table (no "shale (5000)" anywhere in its 436 pages, and its only tables
  are 6.1, 7.1 and 8.1), so it does not unblock the overlay.
- **A.9's recorded gap is closed, and its recorded description was wrong in both
  halves.** Over `V_S_layer / V_S` in [1.3, 1.5] at `ka = 2.5` the slow-formation
  cased dipole returned `NaN`. The note said the real-axis scan finds nothing and
  the mode sits within ~1e-3 of the shear branch point. The scan does find its one
  crossing — at 1006, 978 and 956 m/s — and the mode is 6–7 % above `V_S`, at 855,
  851 and 859. From that far away the complex tracker runs instead to the layer's
  own shear speed (1040.00, 1120.00, 1200.00 m/s to the digit), which is the
  degeneracy `exclude` names and rejects. Correctly rejected, and nothing left.

  `_march_leaky_cased_branch` now falls back to seeding **off** the real axis,
  which converges immediately and coarsely — a single level at 5 % of `Re(k_z)`
  over 12 points already finds all three roots to 1e-4. The swept stiffness family
  is unbroken from 1.28 to 1.62, with one turning point in each of phase velocity
  and attenuation. The values were counted before they were computed: an
  argument-principle winding number over the window identified them independently
  of the marcher, which is the job A.10's continuity fix made possible.

  **The sweep is gated on the branch never starting, and the gate is the
  load-bearing part.** Its extra reach also finds a family of zeros just above
  `V_S` that are sharp to 1e-13 and carry winding number +1 — so no root-quality
  test rejects them — and are not modes. They fail both things a guided mode must
  do:

  | | across the annulus (`V_S_layer / V_S` 1.2 → 2.0) | across frequency (3 → 15 kHz) |
  | --- | --- | --- |
  | the branch-point family | 807–810 m/s, ignores the casing | 1.004–1.017 `V_S`, non-monotone |
  | the flexural branch | falls with the annulus in step | 953 → 888 → 868 → 834 m/s |

  Ungated, the sweep seeds off that family at a frequency where the mode has left
  the window, and the monotone-descent rule walks it down the whole band: **every**
  already-converged frequency moved, by 17 % at 3.5 kHz, ending 0.23 % above `V_S`
  instead of 1.3 %. Gated, production output is **bit-identical** — 0 values
  changed, 0 frequencies gained or lost on both the dipole and the screw — and the
  fixture that had nothing converges. A fixed dead band cannot separate the two
  families, because the flexural branch descends into the same neighbourhood at
  the top of the band, so the floor applies to *seeding* only; continuation, which
  arrives along a dispersion curve, stays unrestricted.

  **Cost, because the sweep is paid where it cannot help.** A stack with no mode
  anywhere fails pass one at every frequency, which is exactly when pass two
  runs, so an uncapped sweep does its full seed grid at all of them — and the
  surrogate generators reject such stacks by the hundred. Three tests in
  `tests/test_gen_surrogate_dataset.py` went 41 s → 828 s and the CI job
  422 s → 1331 s. The sweep now tries at most 5 frequencies, spread across the
  band rather than taken from its start, and the grid is 16x2 rather than 24x3
  (12 seeds already found all three gap roots to 1e-4). Pass two also skips the
  real-axis scan while no root has been found, where it provably returns what it
  returned in pass one. Those three tests are back to 82 s and the suite to 4:48;
  the recovered values are identical at every setting tried.

  **The 7 % step at the bound/leaky crossing is settled: three objects, not one.**
  The step is real — 798.91 m/s at ratio 1.26 against 857.39 at 1.28 — and it is a
  **handover between two different modes** rather than a break in one.

  | | measured |
  | --- | --- |
  | the bound mode | **absorbed at the shear branch point.** Climbs to 799.96 m/s at ratio 1.275; past 1.2775 a 1500-point scan of the proper-sheet determinant finds no root anywhere in the bound window. It ends, rather than being lost by the solver. |
  | the leaky branch reported above | **already exists below.** At ratio 1.20 it is at 868.30 m/s while the bound mode is still alive at 786.67. Not a sequel. |
  | the branch-point pole | **continuous through the whole crossing.** On the improper sheet it descends past `V_S`, runs *below* it over `1.285 ≤ V_S_layer / V_S ≤ 1.315` (798.86 m/s at 1.295) and climbs back — one turning point, `\|det\|` sharp to 1e-13 at every step. |

  The third row is what the open question was about, and the answer is that the
  mode is not annihilated: the production search cannot see that stretch because
  its window is floored at `V_S`. Searching only above `V_S` made it appear to
  vanish at 1.28 and return at 1.32 with the attenuation jumping 0.98 → 1.47. The
  gap was the floor, not the physics.

  **One claim withdrawn.** The branch-point pole was written up as ignoring the
  casing and "not a mode". Sampled coarsely at stiffnesses far from the crossing
  it barely moves, which is where that came from; swept finely it runs
  814.80 → 798.86 → 808.11 m/s over ratios 1.10 → 1.295 → 1.41 and is one
  continuous object. The seed floor stands on the claim that survived — it is not
  the flexural branch, and seeding on it destroys the answer by 17 %. The crossing
  test's "no jump across the boundary" comment was never what its assertions
  checked (they allow 1.15 `V_S`) and is withdrawn.

- **The leaky determinant was two functions glued along the real `k_z` axis**
  (roadmap A.10). Every complex root search seeds on that axis and then steps
  off it, so the determinant has to be one analytic function there. It was not.

  `numpy.sqrt` selects `Re(alpha) >= 0`, which is the *decay* condition and the
  right rule for a bound branch. The radiation condition is a different one,
  `Im(alpha) > 0`, and the principal root carries
  `sign(Im(alpha)) = sign(2 Re(k_z) Im(k_z))` — so it is outgoing only while
  `Im(k_z) >= 0`, and **incoming** below the axis. **14 %** of the leaky Bessel
  evaluations in the A.9 cased dipole run, and 3 % of the screw's, were on that
  incoming branch. Correcting the root alone is not enough: with `Re(alpha) < 0`
  the argument `i alpha r` crosses `hankel2`'s cut, so `_k_or_hankel` also
  evaluates its leaky branch through
  `(pi/2) i^{n+1} H_n^{(2)}(i z) = -K_n(z) + i pi (-1)^n I_n(z)`, whose `kv`/`iv`
  cut lies on the negative real axis where `z` never goes.

  Measured at 12 kHz in the fast sandstone, as one-sided limits against the
  value on the axis:

  | branch that flips | `det(k_z - i0) / det(k_z + i0)` before | jump before | jump after |
  | --- | --- | --- | --- |
  | none (all bound) | `+1` | 7e-12 | 7e-12 |
  | fluid only, oscillatory | `-1` exactly — an overall factor, so roots were never moved | 2 | 2.5e-11 |
  | formation S leaky | `-0.238 + 0.020i` — `k_z`-dependent, a different function with different roots | 1.24 | 3.4e-11 |
  | formation P and S leaky | `+0.279 - 0.091i` — likewise | 0.73 | 3.4e-11 |

  ("jump" is `abs(det(k_z - i0) - det(k_z))` over `abs(det(k_z))`; after the fix
  both one-sided limits converge on the axis value, and linearly in the offset.)

  **No published or returned value changes.** The new leaky evaluation agrees
  with the previous `hankel2` one to **1.5e-16** over every argument the solvers
  actually reached, the A.9 dipole and screw branches return bit-identical
  velocities and attenuations, and the incoming-branch evaluations go
  **457 → 0** and **119 → 0**. The A.10 order-consistency invariant
  (`d/dx K_n = -K_{n+1} + (n/x) K_n` on the returned pair) now also holds at
  second-quadrant `alpha`, which did not exist before.

  Layer blocks need no such rule and are left alone: they keep both `I_n` and
  `K_n`, so negating a layer wavenumber is a change of basis that cancels in
  `E(r_out) E(r_in)^{-1}` — measured at **1.1e-12** while the E-matrix itself
  moves by 312 %.

  **It answers A.9's open question.** A.9 recorded a gap at
  `V_S_layer / V_S` in [1.3, 1.5] (`ka = 2.5`) where the real-axis scan has
  nothing to seed from, and noted that an argument-principle search would be
  needed to say whether a root is there at all. That search needs a
  single-valued analytic function on and inside the contour, which is what this
  supplies — before it, a contour dipping below the axis crossed the
  discontinuity and its winding number meant nothing. The count is **exactly one
  root** at each of 1.3, 1.4 and 1.5, and they are ordinary members of the same
  branch (855.1, 850.5 and 859.3 m/s, positive attenuation, `|det|` sharp to
  1e-13). Seeding the driver from it is A.9 driver work and is not done here.

- **The `n = 2` fast-formation solvers were tracking round-off** (roadmap A.7).
  The modal determinant evaluated at real `k_z` is not complex in any useful
  sense: it is a real quantity times a phase that does not depend on `k_z`,
  contributed by the fixed powers of `i` the oscillatory fluid Bessel functions
  and the row/column rescale introduce. **The parity of that phase flips with
  azimuthal order.** At `n = 1` the determinant is imaginary, so `Im(det) = 0` is
  the root condition; at `n = 2` it is real, and the marcher was tracking
  `Im(det)` there — which is round-off at ~1e-16 of `|det|`.

  Measured over 600 velocities at 12 kHz in the fast sandstone, the open-hole
  `n = 2` determinant has **one** sign change in `Re` and **212** in `Im`. Both
  drivers now pick the component that carries the signal by measuring it
  (`_real_root_function`), so a change of convention upstream cannot silently
  reintroduce the defect.

  What this closes, all previously recorded as separate defects:

  | | before | after |
  | --- | --- | --- |
  | figure 5a screw, fast sandstone | 8 % median, "not one within 5 %" originally | **0.16 % median, 0.43 % worst, 12/12 points** |
  | figure 7b screw, granite | 2.60 % median, 14/72 converged | **1.63 %, 72/72** |
  | figure 7b screw, limestone | 12.80 % median, 1/30 converged | **1.38 %, 30/30** |
  | screw cutoff (figures 6, 14) | 8.3 kHz vs published 6.29 — 32 % high | **6.39 kHz, +1.6 %** |
  | figure 6(b) ring band, 6.5–8.2 kHz | empty | **fully covered** |
  | cased `n = 2`, layer = formation | NaN (marcher declined among ~90 crossings) | **reproduces the open hole to 1e-13** |
  | grid reproducibility | two last-bit-identical grids gave different coverage | **identical coverage and values** |

  **The recorded diagnosis was wrong, and so was the fix it pointed at.** A.7
  attributed this to catastrophic cancellation in the propagator chain and named
  the delta-matrix / Abo-Zena reformulation as the only route. The propagator is
  fine: at `N = 1` it reproduces `E(b)` from `P E(a)` to **1e-16** in a row-scaled
  norm, and the same 430 `Im(det)` sign changes appear in the **open-hole**
  determinant, which has no propagator at all. The `plans/guides.md` §10b claim
  that A.7 was governed by the dimensionless group `|s_layer| h` and therefore
  immune to rescaling is withdrawn there.

  `tests/data/cylindrical_solver_golden.npz` regenerated for
  `n2_quadrupole_fast`, which is now reference quality rather than
  characterisation: both values are sharp zeros of `Re(det)` (2.6e-11 and
  1.8e-11 relative to 0.1 % away), inside `(V_f, V_S)`, descending, and
  reproduced exactly by an independent 50 Hz grid.

- **Slow-formation cased holes get their `n >= 1` modes back, as leaky ones**
  (roadmap A.9). Fixing A.8 removed a spurious bound root behind stiff annuli and
  left `flexural_dispersion_layered` / `quadrupole_dispersion_layered` returning
  all-`NaN` for slow formations behind casing. The mode was never bound there: a
  steel casing raises the composite bending stiffness until the dipole outruns
  the formation shear speed, so it radiates into the rock and the real-valued
  determinant — which only describes fields evanescent everywhere outside the
  fluid — has no root for it.

  Both layered drivers now fall back to a complex-`k_z` search at exactly the
  frequencies the bound path could not answer, over
  `(V_S, min(V_f, min layer V_S))`, with the formation's leaky flags coming from
  `_detect_leaky_branches` and the root refined by `_track_complex_root`. The
  marcher (`_march_leaky_cased_branch`) is shared between `n = 1` and `n = 2` so
  the two cannot drift, seeds from the slowest real-axis `Im(det)` crossing, and
  enforces the same monotone-descent rule A.2 established. Results carry a real
  `attenuation_per_meter`; a purely bound curve still reports `None`.

  For the standard steel + cement stack on a slow sandstone the dipole branch
  runs 989.5 → 810.4 m/s over 3.5–12 kHz with attenuation 2.51 → 0.35 /m, and the
  screw branch 948.3 → 817.3 m/s over 6–15 kHz. Both descend monotonically toward
  the formation shear speed from above, as a guided mode must, and
  `Im(k_z)/Re(k_z)` falls from 11 % to 0.4 % across the band.

  There is no published cased-hole dispersion curve in Schmitt & Cheng, so this is
  validated against internal oracles rather than a figure: the complex determinant
  reproduces the bound solver's root to 1e-9 relative with zero imaginary part
  wherever the mode is still bound (two formulations, same answer); the branch is
  continuous across the shear-speed crossing as the annulus stiffens; the
  determinant vanishes at every returned root to 1e-12–1e-14 relative; and grids
  of 9 to 65 points agree at 8 kHz to 6e-13 m/s. **A.7 does not block this** —
  its noise is the fast-formation `n = 2` window, whereas this slow-formation
  window carries one to six crossings, a mode spectrum rather than cancellation.

  Also fixed, in the bound layered loop this sits next to: the bracket-expansion
  step could walk past the mode into the determinant's far tail and return
  physically impossible phase velocities (26.8 m/s at 16 kHz on the cased screw
  fixture). Converged roots below the Scholte speed of the borehole fluid against
  the softest solid in the stack are now rejected.

- **The leaky Bessel branch's docstring described a different function**
  (roadmap A.10). `_k_or_hankel(leaky=True)` claimed to reduce to `K_n` at a bound
  `alpha`; it does not — expanding the Hankel form gives
  `(pi/2) i^{n+1} H_n^{(2)}(i z) = (-1)^{n+1} K_n(z e^{i pi})`, the next sheet, so
  the two differ by factors of 2 to 3e3 there. The *implementation* is correct and
  unchanged: it returns two consecutive orders of one solution, which is the
  property every caller depends on when it forms radial derivatives from the pair,
  and it is outgoing at a leaky `alpha` (phase slope `+Im(alpha)`, against
  `-Im(alpha)` for plain `K_n`). Neither property was tested; both are now, along
  with a note that an attempt to "restore" the false claim by negating `alpha`
  passes every finiteness and asymptotic check while breaking order consistency
  with a residual of order 1.

- **The SV column of every `n >= 1` cylindrical determinant is now a solution of
  the elastodynamic equations** (roadmap A.8). fwap represented the SV field as a
  vector potential with only an azimuthal component. The cylindrical vector
  Laplacian couples the radial and azimuthal components through a term
  proportional to `n`, which that ansatz has no term to cancel, so it is not a
  solution for `n >= 1` — the radial equation becomes Bessel of order
  `sqrt(n^2 + 1)` rather than `n`. At `n = 0` the coupling vanishes, which is why
  `stoneley_dispersion` was never affected.

  The defect was found by transcribing Schmitt & Cheng's appendix (pp. 235-236),
  which prints all 36 elements of the layer matrix in closed form, and expressing
  fwap's columns in that basis: the P and SH columns came out exact
  (r-independent to ~1e-13), the SV columns drifted 24-86 % over a 4 cm change in
  radius. The replacement is the Hansen form `u = curl curl(chi z)`, which is the
  appendix's own column; at general `n`, with `sigma = +1` for the `I` family and
  `-1` for `K`, and `B` the corresponding Bessel function,

      P1 = sigma s B_{n-1} - (n/r) B_n            (= d_r B_n)
      P2 = s^2 B_n + n(n+1) B_n/r^2 - sigma s B_{n-1}/r
      P3 = (n/r) [ (1+n) B_n/r - sigma s B_{n-1} ]

  giving `u_r = kz P1`, `u_z = -s^2 B_n`, `u_theta = -kz (n/r) B_n`,
  `sigma_rr = 2 kz mu P2`, `sigma_rz = -(2 kz^2 - kS^2) mu P1` and
  `sigma_r_theta = 2 kz mu P3`. Verified against the appendix to 3e-16 at `n = 1`
  and `n = 2`, both Bessel families.

  Applied to every `n >= 1` path: the open-hole determinants at `n = 1` and
  `n = 2` (real and complex), the layer E-matrices and their complex twins, the
  formation half-space columns of the complex cased determinants, the hand-coded
  10x10 single-layer determinant, and the VTI qSV column. `flexural_dispersion_layered`
  now routes every layer count through the cased determinant rather than keeping
  a second implementation of the same boundary-value problem in step.

  Measured against Schmitt & Cheng's published curves:

  | tie | before | after |
  | --- | --- | --- |
  | figure 8a Stoneley (`n = 0` control) | 0.033 % rms | 0.033 % rms |
  | figure 8a flexural | 1.29 % rms | **0.063 %** |
  | figure 8a screw | 0.94 % rms | **0.058 %** |
  | figure 2a fast flexural | 0.78 % median | **0.16 %** |
  | figure 7a flexural, granite / limestone | 1.24 / 1.39 % median | **0.45 / 0.31 %** |
  | figure 15(b) screw, virgin / 8 cm / 16 cm | 1.29 / 0.58 / 2.12 % rms | **0.055 / 0.136 / 0.197 %** |
  | figure 15(a) 16 cm invaded, vs the appendix's own assembly | 1.5 % low at every point | agrees to **0.011 %** |

  Several long-standing "solver limitations" recorded in the test suite were this
  defect and have closed with it. The near-cutoff gap — 1.48 kHz at `n = 1` and
  1.52 kHz at `n = 2` in the slow sandstone, which read like a quantity set by the
  hole — is now 0.00 and 0.12 kHz. Figure 11(b)'s screw arrival, which the solver
  reported as absent, is found at 1180.2 m/s against the published 1179.0
  (**+0.10 %**). Figure 8a's Airy frequency, 25 % low when the phase curve was
  differentiated, is now within 1 %. The `n = 1` fast-formation flexural solver
  resolves from 0.99 kHz instead of 2.5.

  **One capability narrowed, honestly.** Behind a steel casing the cased dipole
  mode is faster than a slow formation's shear speed and therefore leaky, so the
  real-valued bound-regime determinant has no root for it: `flexural_dispersion_layered`
  and `quadrupole_dispersion_layered` now return `NaN` for slow-formation cased
  stacks where they used to return a spurious bound root that *rose* with
  frequency (199, 445, 755 m/s at 6, 9, 12 kHz — backwards for a flexural mode,
  and below every wave speed in the problem). The mode is still there: `Im(det)`
  of the complex cased determinant with the leaky formation branch crosses zero
  at 830 m/s (3 kHz) and 877 (6 kHz), just above `V_S = 800`. Reaching it needs a
  leaky search, which the layered path only runs for fast formations — where the
  cased dipole curve is unaffected and descends 2600 -> 1506 m/s over 2-6.5 kHz.
  Invaded-zone stacks, the regime figure 15 validates, are unaffected at any
  layer count. `generate_slow_two_mode_cased_dataset` consequently passes
  `require_all_modes=True` (a new option on `generate_sample` /
  `generate_dataset` / `generate_cased_dataset`) so its samples still carry both
  modes, though the flexural one now covers a contiguous upper sub-band rather
  than the whole grid.

  The `n = 2` fast-formation open-hole path did not improve: granite went
  2.04 -> 2.60 % median with coverage 22 -> 14 of 72, limestone 8.6 -> 12.8 % with
  11 -> 1 of 30. That is consistent with roadmap A.7 rather than at odds with it —
  that determinant is noise-dominated, so which spurious crossings the marcher
  finds changes with any change to the determinant.

  `tests/data/cylindrical_solver_golden.npz` regenerated for all seven `n >= 1`
  arrays, with each replacement verified as a root of its own modal determinant
  (5e-14 - 2e-9 relative), inside its physical window, monotonically descending,
  and reproduced by an independent 50 Hz grid.

- **The fast-formation flexural and screw solvers no longer return an overtone**
  (roadmap A.2). `_flexural_dispersion_fast_formation` and its `n = 2` and cased
  siblings searched phase velocity in `(V_R, V_S)`. **`V_R` is not a limit of
  these modes**: the branch descends from `V_S` toward the *Scholte* speed and
  crosses `V_R` partway through the band (4.45 kHz for the fast sandstone of
  Schmitt & Cheng figure 2a). The window therefore lost the fundamental over most
  of the band while still containing higher trapped modes, and returned one of
  those — silently, since they are ordinary bound roots.
  Two changes, both required. The window is now `(V_f, V_S)`, `V_f` being the
  real floor (below it `F^2 > 0` and the branch flags stop describing the field).
  And the fundamental is selected: the marcher walks **up** in frequency and
  keeps the slowest root no faster than the previous one. Widening alone is not
  a fix — it swaps a 65 %-high answer for a 14-39 %-high one.
  Against the published curves, open-hole `n = 1` now lands at **0.78 % / 1.03 %
  / 0.87 % median error** for figure 2a's sandstone and figure 7a's limestone and
  granite, which is those figures' digitisation floor. It was on the right branch
  at **2 of 115** samples, and granite had **no** correct sample at all. On
  figure 7a's merged band the error goes from **+124 % (granite) and +69 %
  (limestone) to +0.7 % and −1.7 %**, and the "error grows with formation
  stiffness" ordering is gone — granite is now the closest. Coverage is
  contiguous and monotone, and group velocity is never negative.
  **Confirmed by a figure that played no part in designing it**: differentiating
  the corrected branch predicts figure 3's observed Airy arrival at 5 m to
  **+8 %**, where the old bracket implied a wave **2.2× too early**.
  The cased `n = 1` path shares the same marcher, so figure 12a's phase band goes
  from **+30-55 % to −3.8 to −2.5 %**. Open-hole `n = 2` is on the fundamental
  too (granite 1.6 % median against figure 7b); its remaining residual is the
  separate near-cutoff onset delay, not the bracket.
  **What this does not fix**, and returns NaN for rather than guessing: below the
  `V_R` crossing the mode is leaky and has no real-`k_z` root at all (a
  20 000-point scan of `(2000, V_S)` finds no sign change at 2.5, 3.0 or
  4.0 kHz), and above the `V_f` crossing it has left this regime. Both need
  complex-`k_z` continuation. The layered `n = 2` path is a separate matter — see
  Known issues.
  **A caveat that survives at `n = 2`.** The open-hole `n = 2` answer is on the
  fundamental now, but it still moves by **1.7 % across grid densities and 3.2 %
  across grid start points**, and vanishes on some — a solver's answer at one
  frequency should not depend on which others were requested. That is the `n = 2`
  root-finding instability figure 6 recorded independently (two grids differing
  by last-bit rounding giving 47 and 42 converged points of 71), and correcting
  the bracket was never going to remove it. `n = 1` is grid-independent to
  **0.000 %** over the same checks. So `n = 1` fast-formation results are
  quotable and `n = 2` ones are not yet.
  **The same wrong bound existed in a second place.** `sonic_ml`'s
  `test_solver_flexural_asymptotes_bracket_oracles` asserted that the flexural
  mode stays between `1/vs` and `1/V_R`, and passed only because the solver had
  the identical bound built into its search window and so could not produce a
  counterexample. It now checks the real bracket and asserts the branch is *seen*
  to descend past `1/V_R`. `flexural_hf_slowness` itself is unchanged — it
  returns the Rayleigh slowness, which is what it says — but its docstring
  claimed the Scholte limit is "a few percent slower"; for a 4000/2300 m/s
  formation it is 1470 against 2116 m/s, about 30 %, and that understatement is
  what made the bound look safe to assert.
  Roughly 30 tests changed meaning with this fix, including every one written to
  pin the defect. Several had asserted `V_R` as a floor for these modes, which
  was itself a statement of the bug. The `n1_flexural_fast` and
  `n2_quadrupole_fast` golden arrays were regenerated: the old file pinned
  2591.9 m/s at `n = 1` (essentially `V_S`) and 2390.4 at `n = 2` (`V_R` to four
  figures) — the defect itself. The replacements were verified against the
  determinant before being committed, as that file's precedent requires.

- **`quadrupole_dispersion_layered` no longer rejects a single invaded zone**
  (roadmap A.6). The slow-formation branch applied the per-layer constraint
  `layer.vs >= vs` for *every* layer count, so any annulus slower in shear than
  the formation was refused with `ValueError` before the solver ran — and an
  invaded zone is by definition slower than the rock it replaces. The whole
  invaded-zone family was therefore unrepresentable at `n = 2` in a slow
  formation.
  **It was an implementation/docstring mismatch, not a scoping decision.** The
  function's own `Raises` section has always said the constraint applies
  "(multi-layer only)", and `flexural_dispersion_layered` has always enforced it
  that way — for two or more layers, with the single-layer path left to the
  caller. `n = 2` now matches both its docstring and its sister path. **The
  multi-layer guard is unchanged**, and two soft layers still raise.
  **Validated against the figure that plots these curves**, not assumed. Schmitt
  & Cheng figure 15(b) is the screw mode for exactly this configuration; the
  figure-15 work digitised only panel (a), which is why this went unnoticed.
  Against the digitised curves the newly-unblocked path returns **0.58 % rms**
  for an 8 cm invaded zone (median +0.29 %) — *better* than the same solver's
  **1.29 %** on the virgin rock of the same figure, which is the control that
  prices the digitisation. A path that was refusing to run computes its own
  figure more accurately than the path that was allowed to.
  Figure 17 goes from **2 of its 12** plotted waveforms computable to **6**.
  The fix does **not** touch the mode's onset: fwap still resolves the 8 cm
  model only from 5.6 kHz against a published 3.4 kHz, which is the slow screw
  mode's near-cutoff gap and a separate item. The six waveforms still
  unreachable are all below cutoff.
  One existing test changed meaning: `..._rejects_softer_layer` asserted the old
  single-layer rejection and now asserts the documented contract instead — one
  soft layer accepted, two rejected.

### Known issues
- **The cased `n = 2` determinant is noise-dominated in the fast-formation
  window** (roadmap A.7). Exposed by fixing A.2 — the narrow bracket had been
  hiding it. Scanned across `(V_f, V_S)`,
  `_modal_determinant_n2_cased_complex` produces about **90 sign changes at
  12 kHz** on figure 14's model where the physics supports a handful, and
  **10-33** even with the single layer set identical to the formation, a
  configuration that is physically the open hole and where the un-cased 4x4 gives
  one clean root. They arrive as near-duplicate pairs straddling the true value
  (2084.0 and 2085.0 against the open hole's 2084.9) — catastrophic cancellation
  in the propagator chain, not a mode spectrum.
  `quadrupole_dispersion_layered` therefore now returns **NaN in the fast
  formation rather than a root drawn from noise**: quiet, not fixed. This is a
  deliberate trade and a behaviour change — the path previously returned finite
  values, and they were wrong. `flexural_dispersion_layered` (`n = 1` cased) is
  unaffected. The intended route is the delta-matrix / Abo-Zena reformulation
  already tracked as the A.5 residue.
  **A.8 was not the cause, contrary to the hypothesis recorded here.** With the
  SV column corrected, the same layer-equals-formation scan gives an identical
  count — 430 sign changes on a 1200-point grid at 12 kHz, before and after, to
  the sample. The noise is the propagator chain's cancellation, not the column
  fed into it, so fixing the formulation left A.7 exactly where it was.

- **`flexural_dispersion` returns a flexural overtone in fast formations above
  roughly 15 kHz** (roadmap A.2). Measured, pinned by three tests, and not yet
  fixed.
  `_flexural_dispersion_fast_formation` searches phase velocity in
  `(V_R, V_S)`. **`V_R` is not a limit of this mode** — the flexural branch
  asymptotes to the *Scholte* speed, which is 30 % lower (1472.6 against
  2115.8 m/s for `vp/vs/rho = 4000/2300/2500`). So the fundamental leaves the
  search window entirely, and what the window still contains is an overtone.
  At 19.5 kHz the determinant's roots over `(Scholte, V_S)` are 1853 and
  2269 m/s; the fundamental is 1853 and **the call returns 2269**, with nothing
  to mark it as a different branch. Over 10-30 kHz the returned velocity
  descends 2295 → 2162, goes `NaN`, **jumps back up** to 2283, descends to
  2145, goes `NaN`, jumps to 2275 — successive overtones crossing the window.
  A guided mode never speeds up with frequency, so this is not a sparse curve
  with gaps; it is not a curve. Callers who interpolate across the gaps get a
  plausible-looking result assembled from several modes.
  Widening the bracket to `(Scholte, V_S)` roughly doubles coverage over
  1-12 kHz on three fast rocks (32 → 72 %, 16 → 40 %, 20 → 68 %), but it is not
  a one-line fix: the window then holds several roots and the fundamental has
  to be identified. Taking the highest seeds onto an overtone; taking the
  lowest is non-monotone on two of three rocks; seeding at cutoff with a
  "velocity must decrease" guard is monotone but drops coverage to 4/28, 16/28
  and 10/28. A mode-identification criterion is needed, and shipping a
  half-validated change to a physics solver is worse than shipping the
  measurement.
  **This entry is scoped to its own rock and stays open on that basis.** On
  table 1's fast sandstone (4878/2601/2160) the defect was measured against
  the digitised Schmitt & Cheng fig 2(a) curve at **36.79 % RMS** — and then
  A.7/A.8 landed and the same comparison came back at **0.37 %**. So that
  configuration is fixed. The rock this item is written against
  (4000/2300/2500) is a different one, its three tests still pass, and
  nothing here has been measured on it since; do not read the fig 2(a)
  result as closing the item.
  Slow formations are unaffected — this is the fast-formation path only.
  **This corrects the item's own diagnosis**, which said "a fix means
  complex-plane root tracking". Neither defect above involves complex `k_z` at
  all. The below-cutoff sparseness, separately, still does.

  **Now checked against a published curve, which sharpens two of the claims
  above and corrects one.** Schmitt & Cheng figure 2a plots flexural
  dispersion for a fast sandstone the paper specifies (`V_P` 4878, `V_S` 2601,
  `rho` 2160; fluid 1500 m/s; hole radius 0.10 m). Digitised at 600 dpi to
  about ±1 %, it runs from the formation shear speed (2596 read against 2601)
  to the Scholte speed (1493 at 24.9 kHz against 1484, still descending), and
  crosses `V_R` at **4.45 kHz** — so the `(V_R, V_S)` window holds the true
  root over just **10 %** of the plotted band.
  On that rock the solver answers at 5 of 13 tabulated frequencies, every
  answer inside `(V_R, V_S)` and every one **62-73 % too fast**. On a finer
  0.2 kHz grid it lands on the right branch at exactly two frequencies, 4.2 and
  4.4 kHz — **2 of 115 samples**, +2.8 % and +1.5 % against a ±1.2 % reading
  uncertainty — which is precisely the sliver below the 4.45 kHz crossing where
  the true curve is still inside the bracket. Nothing in the returned object
  distinguishes those two from the 47 overtones.
  The Scholte-edged bracket is worth more than "doubles coverage" — it
  recovers **4.4-16.4 kHz at 0.66 % median error** (worst 1.7 %). Outside that
  window it recovers nothing, because no real-axis root exists: below 4.4 kHz
  and above 16.4 kHz a 4001-point scan finds `Im(det)` sign changes only at the
  `s = 0` and `F = 0` endpoints, with the determinant finite and
  `|Re|/|Im| ~ 1e-16` throughout.
  **The correction**: the residue is not "the below-cutoff half". It is two
  disjoint intervals, one at each end of the band, with a working middle
  between them. Figure 2b agrees — the flexural mode's `1/Q × 100` runs 1.70 at
  2.3 kHz to 5.34 at 5 kHz to 3.27 by 25 kHz, non-zero throughout — so the pole
  is off the real axis everywhere and the real-axis root is an approximation
  that happens to be excellent in the middle. Its quality does not track `1/Q`:
  it fails where attenuation is lowest and works where it is highest, so
  proximity to the real axis is not the criterion.
  The digitised reference table is checked in as `_FIG2A_FLEXURAL_PHASE` in
  `tests/test_cylindrical_solver.py`, with an end-anchor test of its own so a
  bad digitisation cannot silently become the reference.

  **Figure 7a then measured the same defect against formation stiffness.** It
  plots the flexural mode for granite, limestone and the same fast sandstone on
  one axis over 0-15 kHz. Digitised with the axes least-squares fitted to the
  ticks (15 x-ticks residual to 0.018 kHz, 4 y-ticks to 0.0004 normalised):
  - **Three anchors, not one.** Every plateau lands on its own shear speed —
    3749.6 / 2768.7 / 2597.7 against 3750 / 2771 / 2601.
  - **The bracket empties at the same frequency whatever the rock.** All three
    cross `V_R` between **4.43 and 4.45 kHz**, though `V_R` spans 2413 to
    3388 m/s. Figure 2a gave 4.45 kHz for the sandstone from a different page
    with a different axis range — four consistent readings. Not self-similarity
    in `v/V_S` (which reads 0.690 / 0.818 / 0.838 at 5 kHz); two things vary and
    cancel. Measured, not explained.
  - **The error grows with stiffness.** At 11-13 kHz, where all three published
    curves have converged to one line near 1570 m/s, the solver returns
    2551-2442 in sandstone (+57 to +64 %), 2738-2628 in limestone (+69 to
    +73 %) and **3740-3483 in granite (+124 to +137 %)**. The `(V_R, V_S)`
    window rides further above the true curve the faster the rock.
  - **Over the band figure 7a resolves, granite returns nothing** — NaN at all
    13 tabulated frequencies from 3 to 10 kHz. Its "10 % coverage" is a
    sawtooth at 11-13 kHz, outside the resolved region entirely.
  - **Cross-check owing nothing to fwap**: the fast sandstone is plotted in both
    figures, on different pages with different axis ranges. Over 2.50-5.50 kHz
    the two independent reads agree to **−0.25 % to +0.38 %** across thirteen
    consecutive samples. Above 5.75 kHz figure 7a reads 0.8-1.9 % high, because
    its limestone and sandstone curves become a single plotted line at exactly
    that frequency (granite joins them at 10.25 kHz). Nothing past those points
    is tabulated.

  **Figure 7b then checked the `n=2` claim** this item has been asserting since
  its re-diagnosis — "affects `n=2` identically, so one fix repairs two
  solvers". It holds, with one difference that makes the quadrupole solver the
  more dangerous of the two. Against the published screw-mode curves for the
  same three rocks: coverage **75 / 66 / 65 %** (granite / limestone / fast
  sandstone), median error **+102 / +57 / +46 %**, **one** point within 5 %
  across all three, every finite value inside `(V_R, V_S)` and in fact sweeping
  that window end to end. The bracket empties at 7.53 / 7.61 / 7.69 kHz — again
  essentially rock-independent, and mode-specific rather than shared with
  `n=1`'s 4.4 kHz.
  Coverage is the difference that matters: **65-75 % here against `n=1`'s
  21-36 %**, so a caller filtering on `NaN` keeps two to three times as many
  wrong answers from `quadrupole_dispersion` than from `flexural_dispersion`.

  **Figure 12 adds the layered path, and inverts the health metric.** It is the
  first published check of `flexural_dispersion_layered` /
  `quadrupole_dispersion_layered`: the flexural and screw modes with (1) a 16 cm
  invaded zone, (2) an 8 cm one, (3) virgin rock only, (4) invaded rock only.
  Table 1's two fast rows are virgin 4878/2601/2160 and invaded 4390/2341/2360,
  and the figure's own plateaus confirm that transcription — 1.7357 against
  2601/1500 (+0.10 %) and 1.5630 against 2341/1500 (+0.15 %).
  All eight runs — two modes × four models — return values strictly inside their
  own `(V_R, V_S)` window and sawtooth, with upward jumps of +121 to +185 m/s
  where a guided mode's phase velocity can only fall. Against figure 12a's
  merged phase band the layered flexural solver reads **+31 % at 6 kHz rising to
  +53 % by 9.8 kHz**.
  The new part is the coverage: 73 % / 38 % (flexural, 16 cm / 8 cm) and
  74 % / 77 % (screw) against 9 % / 10 % and 50 % / 35 % for the corresponding
  homogeneous models. **An altered zone raises coverage four- to eightfold while
  the answers stay wrong**, so on the layered path coverage is not a weak health
  signal but an inverted one — the configuration that returns the most answers
  is the furthest from having any.
  Stated limit: figure 12a draws eight curves in a 1.2-wide window and resolves
  all eight only across 3.5-5.0 kHz, which is also where they cross. No
  per-model curve was traced there and none is tabulated.

  **Figure 11 finds the screw mode where the solver is silent — and one gap the
  solver is right to have.** The slow-formation quadrupole gather at a 6 kHz
  source rings at **4.68 kHz**, above the 3.74 kHz published cutoff: envelope
  moveout 1166 m/s at r^2 = 0.982, `fwap.stc` phase **1139.6 m/s** against
  figure 8a's traced screw curve at 1179 (**−3.3 %**) — while
  `quadrupole_dispersion` returns `NaN`, its first root for this rock being
  5.25 kHz.
  At a 1 kHz source the packet sits at 1.83 kHz, below any trapped screw mode;
  figure 8a draws no curve there either, so the solver's silence is **correct**
  and the arrival is a leaky or head-wave contribution.
  **This also reframes the figure-6 result.** The near-cutoff gap is
  1.48 / 1.51 / 2.00 kHz for flexural-slow, screw-slow and screw-fast — a
  **1.5-2.0 kHz absolute onset delay** across two modes and two formations. As
  percentages the same offsets read 142 %, 40 % and 32 %, which describes the
  cutoff frequencies rather than the solver. Corrected at the figure-6 site.

  **Figure 10 closes the processing chain on published waveforms.** The
  slow-formation dipole shot gather (14 traces, r = 2.40-5.00 m) digitises
  cleanly — an envelope-peak moveout fit has r^2 = 0.995 — and yields two
  velocities that must be kept apart: envelope moveout is *group*
  (1009 / 1037 m/s), `fwap.stc` alignment is *phase* (1205 / 1156 m/s).
  In panel (a) the packet sits at 0.86 kHz, the flexural low-frequency limit,
  and `stc` returns **1205 m/s at 0.960 coherence against `V_S` = 1201**
  (+0.3 %). In panel (b), at 2.77 kHz, `stc` gives 1156 against figure 8a's
  traced curve at 1172 (−1.3 %) and `flexural_dispersion` at 1187 (−2.6 %) —
  **published waveforms, through this package's processing, landing on this
  package's forward model**.
  Panel (a) also settles the near-cutoff gap: `flexural_dispersion` is silent
  below ~2.5 kHz, yet the waveforms show a coherent arrival at 0.86 kHz moving
  at the shear speed. The gap is a solver limitation, not a physical absence.

  **Figure 9 prices the slow-formation residual in the group domain.** It is
  figure 3's counterpart for the slow sandstone at a 4 m offset. Every trace
  from 2.0 kHz up carries an Airy packet at **4.068 +- 0.045 ms**, drifting only
  −1.8 % across a fivefold change in source centre frequency — 983 m/s.
  Differentiating figure 8a's traced phase curve gives a group minimum of
  **992 +- 4 m/s at 5.1-5.5 kHz**, so the paper's time-domain and
  frequency-domain figures agree to **0.9 %**.
  Differentiating fwap's phase output gives **960.4 m/s at 3.89 kHz** — 3 % low
  in value but **25 % low in frequency**, from a phase curve only 1.3 % off. The
  slow-flexural residual figure 8a found is a tilt rather than an offset, and a
  tilt moves the stationary point: a synthetic waveform built from fwap's slow
  flexural curve places its Airy phase at the wrong frequency while the phase
  velocities still look right.
  Coverage on the slow path is 100 % across every grid step tried, showing none
  of the `n=2` grid instability below.

  **Figure 6 shows the `n=2` cutoff itself is 32 % too high, and that coverage
  is not reproducible.** It plots quadrupole shot gathers at 1.5 kHz and 6 kHz
  source centre frequencies, 14 traces at r = 2.40-5.00 m.
  The gather does not survive digitisation well enough to measure a moveout —
  self-normalised traces, overlapping bands, two dashed guide lines drawn
  through every trace; `fwap.stc` over the reconstruction gives 0.4-0.88
  coherence with no stable peak, so **no velocity is quoted from it**.
  What is solid is the ringing *frequency*, since zero crossings survive
  amplitude clipping: median **7.19 kHz** (7.00-7.38 across twelve traces) for a
  **6.0 kHz** source. A received ring above the source frequency is the
  signature of a mode with a cutoff, and it matches figure 5a's 6.29 kHz cutoff
  with figure 5c's excitation switching on at ~6.3 kHz.
  **`quadrupole_dispersion`'s first root for this rock is at 8.29 kHz** — 32 %
  above the published cutoff — and it returns `NaN` at every single-frequency
  call from 6.5 to 8.4 kHz. The solver is empty at the frequency where the
  paper's own waveforms ring hardest, so the `n=2` defect includes a misplaced
  onset, not just overtones above it.
  **And a reproducibility problem qualifying every coverage number in this
  entry.** `np.arange(6.0, 20.01, 0.2) * 1e3` and
  `np.arange(6.0e3, 20.01e3, 200.0)` are the same 71 frequencies to within
  1.5e-11 Hz (relative 8e-16), and give **47 and 42 converged points**
  respectively, disagreeing at four of five probe frequencies. The continuation
  marcher walks high to low, so a missed root at one step changes everything
  downstream. Coverage is therefore a property of how the caller built the
  array, not only of the rock and band. Every coverage figure here was measured
  on a stated grid and is reproducible on it; a test pins the instability and is
  phrased to start failing if the marcher is ever made grid-stable.

  **Figure 1a supplies the pseudo-Rayleigh tie A.1 said did not exist.** It
  plots the Stoneley and the first two pseudo-Rayleigh modes for the fast
  sandstone — three modes, three fwap entry points:
  `stoneley_dispersion` 36/36 at **0.90 % rms**;
  `trapped_pseudo_rayleigh_dispersion(branch=0)` 97 % at **1.01 %**;
  `trapped_pseudo_rayleigh_dispersion(branch=1)` 96 % at **0.80 %**. At this
  figure's resolution a plotted line is 12.7 m/s — 0.87 % at the Stoneley,
  0.5-0.7 % at the pseudo-Rayleigh modes — so all three sit at one to
  one-and-a-half line widths. A small consistent negative bias is present and is
  *not* claimed as a real offset, because the figure cannot resolve it.
  The `branch` index is validated too: 0 lands on the first mode, 1 on the
  second. Anchors: the Stoneley extrapolates to 1398.3 m/s against
  `tube_wave_speed`'s 1396.3 (+0.14 %), and both pseudo-Rayleigh modes cut on at
  the formation shear speed and descend toward the **fluid** velocity rather
  than Scholte — the trapped family's own asymptote, never previously checked.
  A trap worth recording: in this panel the **group curve is drawn above the
  phase curve** for the Stoneley (correct here, since its phase velocity rises
  with frequency, but the opposite of every other panel in the report).
  Comparing against the wrong branch gives a spurious −2.5 %; the overlay check
  is what caught it.
  **Separately, `fwap.synthetic.pseudo_rayleigh_dispersion` is 37 % slow near
  cutoff**, easing to 6 % by 25 kHz: its cutoff scale is `vs / (2 pi a)` =
  4140 Hz against a true cutoff of 7.71 kHz, 1.9× too low. The docstring already
  says "phenomenological"; this pins how much that word is carrying, which
  matters because `fwap.synthetic` uses it to place an arrival a user may pick.

  **Figure 5a is the screw mode's own panel, and bounds the digitisation
  method.** Figure 7b measured `n=2` across three rocks but only resolves the
  fast sandstone below about 10 kHz; figure 5a plots the same mode alone on
  figure 2a's axes, 0-25 kHz. Traced in two overlapping passes, because mode 2's
  group curve crosses mode 1's phase near 18 kHz and a single pass follows the
  steeper branch down.
  Cutoff value 1.7385 against `V_S/V_f` 1.7340 (**+0.26 %**); 1522.6 m/s at
  24.87 kHz against Scholte 1484.4 (**+2.57 %**, still descending); crosses
  `V_R` at **7.58 kHz** where figure 7b independently gave 7.69; and never
  crosses `V_f` inside the plotted band. So **the screw mode approaches Scholte
  more slowly than the flexural one** — +2.6 % at 25 kHz where the flexural mode
  was +0.6 %, and it never drops below the fluid velocity where the flexural
  mode crossed it at 17.9 kHz.
  The cross-figure agreement is an error bar on the method itself, obtained
  without reference to fwap: the same rock read off two pages with different
  axis ranges agrees to **+0.4 % to +1.8 %** across 7-12 kHz, figure 7b
  systematically about 1 % high — looser than the ±0.4 % figures 2a and 7a
  managed for the flexural mode, and the number to quote for readings off the
  crowded three-rock panels.
  fwap over 6.4-25 kHz: **72 % coverage, not one point within 5 %**, every value
  inside `(V_R, V_S)` and sweeping it end to end, errors +15 % to +67 % with
  median +53 %.

  **Figure 3 restates the defect as a traveltime, and cross-checks two
  figures against each other.** It plots 21 synthetic dipole waveforms at a 5 m
  offset in the rock of figure 2a, source centre frequency 0.5 to 10.5 kHz.
  Digitised from the 21 baselines (155.5 px apart, uniform) with the time axis
  fitted to the seven label decimal points — 303.4 px per ms, residual
  ±0.010 ms.
  Every trace from 3.0 kHz up carries a large late packet at **4.35 ± 0.07 ms**
  whose arrival drifts by only −4.4 % while the source centre frequency changes
  by 250 %. That is an **Airy phase**, pinned to the stationary point of the
  group-velocity curve, and it implies an apparent group velocity of
  **1150 m/s** against the **1109.7 m/s** minimum of the group curve digitised
  from figure 2a — **agreement to +3.7 %** between a time-domain and a
  frequency-domain reading of two different figures.
  Over the same band `flexural_dispersion` answers at 3 of 16 frequencies, at
  2414-2597 m/s, which over the figure's own 5 m offset is 1.92-2.07 ms against
  4.35 ms of published waveform: **2.2× too early**.
  Not used: the printed scaling factors, which would give the excitation curve.
  At this scan quality the glyphs are not reliably legible ("0.0014" and
  "0.0019" cannot be told apart).

  **Figure 13 measures how little a dipole sees invasion at 1 kHz.** Panel (a)
  extracts cleanly: the 8 cm model lags the virgin waveform by **+0.1 us** and the
  16 cm model by **+1.2 us** at 5 m, correlating at 0.992 and 0.981 — **under
  0.1 % of the traveltime**. That is the time-domain form of figure 12's shared
  low-frequency plateau.
  **Corrected while working figure 14**: this entry previously said only panel (a)
  was measurable and that nothing in (b)-(d) cleared r = 0.8. That was an artefact
  of the extraction, not the figure — the half-window was narrower than the widest
  trace's excursion, clipping the *virgin* trace in panel (b) to 68 % coverage so
  every correlation there ran against a truncated reference. Widened, **panel (b)
  measures**: at 3 kHz the 8 cm model lags by **+54.6 us** and the 16 cm model by
  **+99.0 us**, at r = 0.930 and 0.848, invariant to +-0.01 us across 36 crop and
  window choices. So the separation figure 12 predicts above 2 kHz is measured
  after all, and it is steep — the 16 cm delay grows **79x** between 1 and 3 kHz.
  Panels (c) and (d) remain refused, now positively: their components merge
  (coverage stuck at 0.76-0.78 whatever the window) and panel (d)'s lags are
  +264 / +319 us regardless of window, the constant-lag signature of cycle
  hopping.

  **Figure 15 then exonerates the layered code.** It is figure 12's slow
  counterpart — same four models, same solver calls, table 1's slow sandstone
  2751/1201/2100 and its invaded zone 2338/1081/2000 — and it separates two
  explanations figure 12 alone could not. Its group curves are *dashed*, so they
  fragment under connected-component labelling and leave the four solid phase
  curves readable; calibration is the best of the six figures (16 x-ticks
  residual to 0.019 kHz, 4 y-ticks to 0.00024).
  Two anchors, both to 0.02 %: the virgin curves leave the axis at 1200.7
  against `V_S` = 1201, the invaded-only curve at 1081.2 against 1081 — which
  also confirms table 1's slow invaded-zone row.
  Coverage / rms / median against the published curves: virgin *(open hole)*
  91 % / 1.43 % / −1.34 %; **8 cm invaded *(layered)* 84 % / 1.47 % / −1.22 %**;
  **16 cm invaded *(layered)* 92 % / 1.48 % / −1.49 %**; invaded only *(open
  hole)* 67 % / 1.01 % / −0.07 %.
  **The layered solver is as accurate as the open-hole one**, so figure 12's
  31-53 % overshoot is the fast-formation bracket and not the layered
  machinery — one fix repairs both paths, and rewriting the propagator is ruled
  out. *Qualified by figure 16*: that holds for **phase** velocity and does not
  survive differentiation — predicting figure 16's Airy arrival from the group
  minimum is +3.0 % late for the virgin rock against +6.3 % and +8.0 % for the
  two layered models, about twice the error. The exoneration stands; the phrase
  describes the plotted curve, not the wave that arrives.
  It also narrows figure 8a's unexplained ~1.3 % slow-flexural offset: it is
  present in all three `n=1` configurations at the same size and shape, open
  hole and layered alike, while the Stoneley on the same rock was 0.04 %. So it
  is `n=1`-specific and geometry-independent.
  Two limits: the invaded-only curve could not be followed past about 4 kHz (a
  dashed group segment crosses it) and is used only for its anchor; and the
  near-cutoff gap is **not** the single width figure 8a suggested — 1.44 kHz
  virgin, 2.44 with an 8 cm zone, 1.19 with a 16 cm zone, 0.92 for the invaded
  rock alone. That claim covered two modes in one homogeneous rock and does not
  extend to layered models; corrected at its site.

  **Figure 14 marks the boundary of what this defect can be blamed for.** It is
  figure 13's quadrupole counterpart — same fast sandstone, same three models,
  same 5 m offset, source centre frequencies 1.5/3/6/7.5 kHz. Its ringing
  wavetrains defeat cross-correlation in panels (b)-(d), but **panel (a) is a
  compact wavelet and does measure**: the 8 cm model lags the virgin waveform by
  **+9.7 us** at r = 0.924 and the 16 cm model by **+36.5 us** at r = 0.797, both
  invariant (+-0.06 us) across 36 combinations of crop start, crop end and
  extraction half-window. Panels (b)-(d) are refused for a positive reason rather
  than a threshold: their 8 cm lags are 237.7 / 238.9 / 235.1 us at 3 / 6 /
  7.5 kHz, constant to +-2 us across a 2.5x change in source frequency and with
  negative zero-lag correlations — the signature of cycle hopping, not of a delay.
  **That 36.5 us must not be read against figure 13(a)'s 1.2 us as a
  dipole/quadrupole gap**: the panels are at different source frequencies, and
  the delay is a steep function of frequency (see the figure-13 correction
  below). The two figures share no source frequency where both are measurable.
  Also legible is the printed peak-amplitude scale factor on all twelve traces,
  transcribed and then checked
  independently by measuring the plotted ink, the two agreeing to within **0.027**
  in the worst panel. The published claim holds: the quadrupole's amplitude
  spread across invasion thickness is **2.90x** at its lowest source frequency
  against the dipole's **1.25x**, and while the dipole goes flat to 1 % at 6 and
  7.5 kHz the quadrupole never drops below **1.29x**. Panel (c) is genuinely
  non-monotone on both readings.
  **But that content is out of scope for a dispersion solver, not merely wrong
  in one.** Peak amplitude at a fixed offset is excitation times propagation;
  `BoreholeMode` has no excitation field, and `attenuation_per_meter` is `None`
  from both the plain and the layered quadrupole path here. A corrected bracket
  would not reproduce figure 14.
  What it would fix is the rest: of the twelve plotted (model, frequency) pairs
  the solver returns a phase velocity for **three**, the virgin rock giving no
  root at any of the four source frequencies (onset 8.4 kHz, above the whole
  figure); all **194** converged samples across the three models sit strictly
  inside `(V_R, V_S)`; coverage again inverts with invasion thickness (49 / 63 /
  82 of 141, onsets 8.40 / 4.10 / 3.40 kHz); and the one dispersion claim the
  paragraph makes — an Airy-phase group velocity rising with thickness — emerges
  **with the wrong sign**, the sawtooth ramps driving `v_g = 1/(d(f*s)/df)`
  negative on 18 of 48 adjacent virgin samples. There is no usable
  group-velocity curve to be in error.
  It also bounds figure 6's reproducibility caveat: repeating that
  two-ways-of-building-a-grid check on this fast-formation model gave identical
  coverage every time, so the instability is model-specific rather than a
  property of the `n=2` marcher everywhere.

  **Figure 16 checks the slow-formation dipole, and it is the first waveform
  figure whose solver path was already known good.** Same experiment as figure
  13 with the rock swapped. Its caption states outright what figures 13 and 14
  left to inference — "each series is normalized with respect to its own maximum
  denoted by 1.00" — confirming the figure-14 amplitude reading from the authors'
  own words.
  **Twelve drawn arrows calibrate it**: read through the time axis at 5 m, the
  four virgin arrows give 1198.0 m/s against table 1's V_S = 1201 (-0.25 %) and
  the eight invaded ones 1083.0 against 1081 (+0.18 %), with no overlap between
  the families. That confirms the slow invaded-zone row a second time — figure 15
  anchored it at 1081.2 from a different figure — and owes nothing to fwap.
  **Invasion is visible here where figure 13 found it invisible.** Panel spreads
  are 1.63 / 1.55 / 2.21 / 1.42 against the fast sandstone's 1.25 / 1.03 / 1.00 /
  1.00; where the fast formation goes flat at and above 3 kHz the slow one never
  drops below 1.42x. Digits and ink agree to 0.018, settling several two-way
  glyphs (0.754 not 0.734, 0.644 not 0.699).
  **The mechanism is measured, not just its size.** Splitting each trace at its
  own arrow, the P-wavetrain-to-shear ratio rises monotonically with thickness at
  every frequency at or above 3 kHz and with frequency at every thickness —
  0.03/0.15/0.22 at 3 kHz, 0.10/0.96/1.53 at 6 kHz, 0.21/1.95/2.76 at 7.5 kHz —
  and at the top end the P wavetrain becomes the largest event in the trace, the
  series maximum jumping from ~5.0 ms to ~2.35 ms. That is conclusion C as a
  number.
  **A like-for-like delay comparison**, which figure 14 could not supply: figures
  13(a) and 16(a) share source, frequency, offset and thicknesses, so the ratio
  means something. The 16 cm delay is +1.2 us in the fast rock and +117.3 us in
  the slow one — 0.06 % against 2.82 % of traveltime, **45x larger**.
  **And a forward prediction that lands.** The virgin shear packet peaks within
  0.10 ms of 5.05 ms across a 7.5x change in source frequency — frequency-
  independent, so it is the Airy phase — giving 989.6 m/s against figure 8a's
  published group minimum of 992.0: **two independent figures 0.24 % apart**.
  fwap predicts 5.21 ms against 5.05 measured (+3.0 %), which is figure 9's
  "3 % low in value" reached from another figure and another domain.
  Unlike figure 14's fast quadrupole these curves are structurally sound — one
  contiguous run per model, monotone phase, group velocity never changing sign —
  so here the defect is accuracy, and A.2's bracket is not implicated. At 1 kHz,
  though, fwap resolves none of the three models (onsets 2.52 / 3.51 / 2.94 kHz):
  the panel that measures best is entirely outside coverage.

  **Figure 17 checks the slow-formation quadrupole, and its headline was a
  refusal** — `quadrupole_dispersion_layered` raised on every invaded zone,
  making eight of the figure's twelve waveforms unrepresentable. That is filed as
  **A.6** and is now **fixed** (see the Fixed section above); with the guard
  corrected, figure 17 goes from 2 computable waveforms to 6.
  The six still unreachable are all below the screw mode's onset, which the fix
  does not touch: the virgin mode resolves only from 5.25 kHz, above the 1 and
  3 kHz panels. That curve is structurally sound (no interior gaps, group velocity
  never negative), and predicting the virgin Airy arrival from it gives 5.24 ms
  against 4.96 measured, **+5.6 %**, against the flexural mode's +3.0 % on the
  same rock.
  **The published data stands whatever fwap does**, and it is the tightest
  external agreement in the series: the eight invaded arrows read **1081.3 m/s**
  against table 1's 1081 (**+0.03 %**), the four virgin ones 1193.6 against 1201
  (-0.61 %). Finding them needed a stricter discriminator than figure 16 used —
  the arrow is the arrow-shaped component *not connected to the trace* — and
  re-running figure 16 that way reproduces its twelve values exactly, so its
  record needed no correction.
  Panel (a)'s amplitude spread is **6.41x**, the largest in the four waveform
  figures, with the *virgin* trace the smallest at 0.156: a slow-formation
  quadrupole at 1 kHz is barely excited, and invasion brings the screw mode's
  useful starting energy down into the source band. One glyph was genuinely
  ambiguous (0.156 against 0.186, which the ink cannot separate on a 39-pixel
  excursion) and is settled by comparing it against known 5s and 8s in the same
  figure. The report's claim for panels (c) and (d) holds as written — P/S grows
  from virgin to 16 cm by 26x and 69x against the dipole's 15x and 13x — though
  read as absolute level rather than growth it would look false.

  Ninety tests now pin the item, not three, and every reference table carries
  its own shear-speed anchor test.

### Validated
- **`stoneley_dispersion` tied to a published curve at 0.04 % rms**, the
  project's first external tie better than 1 %. Schmitt & Cheng figure 8a plots
  the Stoneley, flexural and screw modes for table 1's slow sandstone (`V_P`
  2751, `V_S` 1201, `rho` 2100) on one axis over 0-15 kHz. Digitised — the three
  curves are disjoint connected components there, so no branch tracking was
  needed, and the narrow 0.650-0.850 axis makes the plotted line worth about
  ±3 m/s, or ±0.3 %.
  Three anchors, none needing a solver: the Stoneley's low-frequency limit reads
  1135.6 against `tube_wave_speed`'s 1136.2 (−0.06 %), and both shear modes
  leave the axis at 1201.4 against `V_S` = 1201 (+0.02 %).
  Over 0.1-14.9 kHz at 0.25 kHz: **Stoneley 59/59 finite, rms 0.04 %, worst
  0.08 %** — below what the figure can resolve, so fwap and the published curve
  cannot be told apart. Flexural 49/55, rms 1.29 %. Screw 38/44, rms 0.94 %.
  **The borehole radius is now measured rather than assumed.** Table 1 gives no
  hole radius; the Stoneley misfit is 0.05 % rms at `a` = 0.100 m and degrades
  either side (0.13 % at 0.095, 0.14 % at 0.105).
  **Two things this also found.** The flexural mode carries a real systematic —
  zero near 3.3 kHz, −1.8 % at 5-6 kHz, recovering to −0.8 % by 14 kHz — which
  is four times the reading uncertainty that the Stoneley on the same panel
  bounds at 0.08 %, and which no radius removes. A candidate is that the paper's
  model is viscoelastic (table 1 carries `Q_alpha`/`Q_beta`; figure 8's own
  attenuation panel gives every mode `1/Q` ≈ 0.02) where fwap's open-hole
  solvers are elastic — but that should move the Stoneley too, and it does not.
  Measured and unexplained.
  And both shear solvers lose **the same 1.5 kHz above cutoff**: the published
  flexural curve starts at 1.04 kHz and fwap's first root is at 2.52; the screw
  curve starts at 3.74 and fwap's first root is at 5.26. One gap width for two
  modes whose cutoffs are 2.7 kHz apart. Above it both are contiguous — the
  benign form of the failure that swallows the whole band in fast formations.
  Ten tests, including one that pins the radius and one that keeps the
  Stoneley's tie an order of magnitude tighter than the shear modes'.

### Fixed
- **The "Schmitt (1988) eqs. 24-26" pointer in `_n1_isotropic.py` is
  withdrawn.** Three places said those equations give the high-frequency
  reduction of the dipole modal determinant to the Rayleigh secular
  equation. In the companion ERL report (Schmitt & Cheng 1987, pp. 220-221)
  they are the Thomson-Haskell propagator product (24), the 6x6 `H` assembly
  (25) and the borehole-wall boundary conditions (26) — layer propagation
  and matrix assembly, with no Rayleigh secular equation anywhere near them.
  That report is the document every other "fig N" in this repository turned
  out to mean, and this pointer arrived carrying the same wrong page range,
  so the likeliest reading is that the equation numbers are mis-sourced the
  same way.
  **Not proven, and deliberately not "corrected".** The JASA article is
  paywalled and its own numbering is unverified, so the numbers are dropped
  rather than replaced with a guess. The reduction itself is not in doubt —
  Paillet & Cheng (1991) sect. 4.2 carries it and remains the pointer. This
  was the last item the earlier citation fix had to leave open.
- **Validation-notebook section 2 cited a figure that is not a dispersion
  curve, and a geometry that does not exist.** It named "Schmitt 1988
  fig 4" and quoted parameters attributed to that paper's table 1 — a
  "shale" at 2740/1280/2400 and a "limestone" at 4900/2840/2700. Opening
  the reference: **fig 4 is a time-domain shot gather** (dipole, fast
  sandstone, 1 kHz and 6 kHz source centre frequency), and table 1 has no
  shale row at all. Its rocks are fast sandstone 4878/2601/2160, slow
  sandstone 2751/1201/2100, limestone 5081/2771/2160 and granite
  5881/3750/2160 — so both quoted formations were approximations of rows
  that were already there, off by up to 8 % in `V_S` and 25 % in density.
  The flexural dispersion curves are **fig 2(a)** (fast sandstone) and
  **fig 8(a)** (slow sandstone). Section 2 now uses those, under table 1's
  own numbers, and is named for the document actually consulted —
  Schmitt & Cheng (1987) — since the 1988 JASA numbering is paywalled and
  remains unverified.
  **The notebook had already drifted from the roadmap on this.** A.1 knew
  fig 4 was a shot gather and had identified fig 2a; section 2 carried the
  first half of that ("the overlay it asks for cannot exist") and not the
  second, telling the reader "which Schmitt (1988) figure carries the
  flexural dispersion curves is **not known**" and keeping the invented
  geometry underneath it. Two documents, opposite answers, same repository.
  Corrected in the notebook, in
  `docs/notebooks/_data/README.md` and in `docs/plans/cylindrical_biot.md`
  (both the plan-B validation bullet and the plan-I reference list).
  The plan-B bullet also predicted "the leaky bend just above the geometric
  cutoff" in that figure. There is no bend and no cutoff; see the Added
  entry above.
- **The Schmitt (1988) citation carried another paper's page range.** Every
  reference to "Shear wave logging in elastic formations" outside
  `fwap/validation.py` gave it as *J. Acoust. Soc. Am.* 84(6), **2230-2244**.
  That is a different article: JASA 84(6) carries three Schmitt papers back to
  back — 2200-2214 *Effects of radial layering when logging in saturated porous
  formations* (Schmitt), **2215-2229 *Shear wave logging in elastic
  formations*** (Schmitt, the one this package builds on, DOI 10.1121/1.397015),
  and 2230-2244 *Shear wave logging in semi-infinite saturated porous
  formations* (Schmitt, Zhu & Cheng). So the citation paired the elastic paper's
  title and author with the porous paper's pages, pointing anyone chasing the
  `n=1` modal determinant at a Biot two-phase paper instead.
  Corrected in all thirteen places (`fwap/synthetic.py`, `fwap/dispersion.py`,
  `fwap/cylindrical.py`, `fwap/cylindrical_solver/{__init__,_n1_isotropic,`
  `_leaky}.py`, `scripts/gen_surrogate_dataset.py`, `plans/roadmap.md`,
  `docs/plans/cylindrical_biot_G_prime.md`); `fwap/validation.py` was already
  right. The title is also unhyphenated as published ("Shear wave logging", not
  "Shear-wave logging") so all fourteen sites now read identically.
  **Not verified:** `_n1_isotropic.py` attributes the high-frequency Rayleigh
  reduction to "Eqs. 24-26" of that paper. The paper is paywalled and
  unavailable here, so the equation numbers stand as they were — if they were
  copied from the same source as the page range they may point into the wrong
  article too.
- **The flexural high-frequency test was anchored to the wrong reference**
  (roadmap A.1). `test_flexural_high_f_slowness_above_inverse_rayleigh` compared
  the modal `n=1` slowness against `rayleigh_speed` with `rel=0.10`. The
  flexural mode does not approach the vacuum-loaded Rayleigh speed: it settles
  at **0.908 V_R** and stays there, so the tolerance was absorbing a 9 %
  reference error rather than bounding the solver — the test was passing on
  17 % of its own margin, and a real regression of a few percent would not have
  moved it. The test now asserts only the inequality, which is genuine physics
  (fluid loading holds a surface wave below its vacuum-loaded speed), and the
  quantitative claim moved to a correct reference.
  The docstring had named the right target all along — "positive **Scholte** /
  fluid-loading offset" — and used Rayleigh as a proxy because `scholte_speed`
  did not exist yet.

### Added
- **The validation notebook's overlay path has reference data in it for the
  first time.** A.1 digitised figures 1a, 2a, 7a, 8a, 12, 20 and 21 into
  test constants; none of them reached `docs/notebooks/_data/`, so
  `check_overlay` had still never scored anything. Two curves now live
  there, traced independently at 400 dpi from Schmitt & Cheng (1987) 1987.8
  ([DSpace](https://dspace.mit.edu/handle/1721.1/121148)), geometry from
  that report's table 1 verbatim:
  - `schmitt_cheng_1987_fig8a_flexural_slow.csv` — slow sandstone
    (2751/1201/2100), 55 points over 1.25-14.75 kHz. **PASS at 0.04 % RMS**,
    worst point 0.15 %, 55/55 points scored.
  - `schmitt_cheng_1987_fig2_flexural_fast.csv` — fast sandstone
    (4878/2601/2160), 89 points over 2.5-24.5 kHz. **PASS at 0.37 % RMS**,
    worst point 1.37 %, over 2.75-17.75 kHz. Above 17.75 kHz the solver
    returns `NaN` rather than a wrong root, which leaves 28 of the 89
    reference points unscored — the overlay is silent there, not green.
  Both are **independent of A.1's reads**, which is the point of shipping
  them rather than exporting the test constants: different session,
  different resolution, different tracer. They agree — this trace puts fig
  2(a) at 1494 m/s at 24.5 kHz against A.1's 1493 at 24.9 kHz.
  Tracing was calibrated on the plot frame and axis ticks alone, and both
  curves' low-frequency limits then landed on table 1's `V_S` to **+0.01 %**
  (fast) and **+0.06 %** (slow) — an independent check, since `V_S` was not
  used in the tracing. That the slow curve also matches with the borehole
  radius left at `a` = 0.10 m corroborates the geometry, which the figure
  captions give only in passing.
- **What Schmitt & Cheng fig 2(a) adds to A.2's settled question.** A.1
  established from fig 4 — the shot gather — that a fast formation carries a
  strong coherent dipole arrival at 1 kHz where `flexural_dispersion`
  returned `NaN`. Fig 2(a) says the same thing in the frequency domain and
  more sharply: the fundamental flexural branch has **no cutoff**, running
  continuously from 2.4 kHz (the lowest frequency plotted) at `V_S` = 1.734
  `V_f` down to 0.996 `V_f` at 24.5 kHz, while the *first trapped* mode,
  curve (2), is the one that begins abruptly at ~8 kHz. The alternative an
  earlier entry floated — "possibly there is no pole at all, the mode
  existing only above its cutoff with the low-frequency dipole energy
  travelling as a shear head wave" — is **wrong**.
  Still not settled: whether the pole leaves the real axis. Every layer in
  table 1 carries a finite `Q`, so the published attenuation mixes intrinsic
  with radiation loss and cannot separate them. Suggestive only — flexural
  attenuation sits at the `Q_beta` = 60 intrinsic floor (1/Q x 100 = 1.67)
  at low frequency and rises to ~5.3 near 5 kHz, roughly 3x the floor. And
  the curve is truncated at 2.4 kHz rather than continuing to zero; whether
  that is Schmitt's plotting choice or his own root-finder's limit is not
  visible from the figure.
- **A `known_defect=` marker was added to `check_overlay` and removed again
  in the same branch, having done its job.** The fast overlay shipped marked
  as an expected failure: at the time it missed fig 2(a) by 36.79 % RMS,
  which was roadmap A.2 measured against published data instead of against
  itself. The marker *inverted* the budget assertion rather than relaxing
  it — the cell failed if the curve started passing — on the reasoning that
  raising the budget to 40 % would hide the eventual fix as effectively as
  it hides the bug. Merging A.7/A.8 fixed that configuration, the marker
  tripped exactly as designed, and it is gone along with the exemption.
  Recorded because the alternative would have left a stale 40 % budget
  sitting in the notebook, quietly passing, for however long it took anyone
  to re-check.
- **The flexural mode is tied to the plane Scholte speed** (roadmap A.1). The
  argument the `n=2` block already rests on — at short wavelength the borehole
  wall looks flat to *every* azimuthal order — had never been applied to `n=1`,
  which is the mode the package sells. `scholte_speed` solves a plane
  fluid/solid interface problem: no Bessel functions, no borehole radius, no
  azimuthal order, so agreement is an external check rather than the solver
  confirming itself.
  Measured on a slow formation at `a` = 0.10 m, flexural velocity over the
  Scholte speed runs **1.0166 → 1.00025** across 10–400 kHz, monotone, and the
  new test asserts convergence to 1e-3 rather than proximity. All three
  azimuthal orders agree at 400 kHz to **6e-6**, which no per-mode check can
  see — a branch error in any single order shows up there as a disagreement.
  Scoped to slow formations deliberately: in fast ones `n=1` is leaky and the
  real-axis search returns scatter or `NaN`, which is roadmap A.2 and the same
  failure the `n=2` block records.
  **This re-scoped A.1 rather than adding to it.** The item asked for five
  digitised figures on the grounds that they were "the only checks that tie the
  solver to literature rather than to itself" — a sentence that had stopped
  being true as the analytic oracles accumulated, and that nobody re-read
  against them. An overlay is scored against a 5 % RMS budget, loose on purpose
  because tracing a printed log-axis figure costs a couple of percent by itself;
  Stoneley is tied at 1e-8 and 0.1 %, quadrupole and now flexural at 1e-3. Three
  of the five figures were the weaker instrument and are dropped. What still
  needs the books is the **pseudo-Rayleigh curve**, **cased-hole Stoneley** and
  **VTI flexural**, none of which any analytic oracle reaches.
- **A real full-waveform sonic gather is in the fixture registry** (roadmap
  F.2), which had been the highest-value open item since the project started.
  `iodp_u1347a_dsi` is an eight-receiver Schlumberger **DSI** monopole run from
  IODP Expedition 324, Hole U1347A — 1307 depths, 512 samples at 10 µs on
  0.1524 m receiver spacing, published by the Lamont-Doherty Borehole Research
  Group on Zenodo under **CC0**.
  **The claim that blocked this was false.** Both `scripts/fetch_real_data.py`
  and `tests/test_real_data.py` said no openly redistributable full-waveform
  sonic gather was known to exist. What had actually been established was that
  none had been *found* — a different claim, and one that went stale the moment
  anyone looked again. Both docstrings now say so explicitly rather than
  quietly dropping the sentence.
  **`read_ldeo_waveforms` reads the format** (`fwap.io._ldeo`, with
  `LdeoWaveforms`, `LDEO_TOOL_NAMES` and `LDEO_MODE_NAMES`). The archive
  publishes every sonic run twice — the original service-company DLIS, and a
  plain binary export about a fifth the size carrying a short self-describing
  header. For a fixture the export is the better file.
  The reader **verifies rather than trusts**, for one specific reason: the
  format is big-endian, and read little-endian its header decodes to enormous
  garbage rather than to nothing, so a trusting reader would allocate wildly or
  return silently wrong samples. `4·(1 + n_receiver·n_sample)·(1 + n_depth)`
  must equal the file size exactly, and the sample interval must be
  sonic-plausible, both before a single sample is read.
  It also **declines to invent the transmitter offsets**, which are in neither
  the export nor the DLIS it came from. `ArrayGeometry` is the caller's to
  build; `plans/roadmap.md` F.5 records how the equivalent question was settled
  for another hole.
  **Measured on the real gathers**: `stc` over 50 gathers spanning 3575–3636 m
  returns a median peak coherence of **0.948**, with 96 % above 0.6. Slowness
  ranges over a factor of four across the interval, which is the lithology
  (chert, chalk, basalt) rather than noise — so the test asserts that no pick
  sits on a *search-band edge* instead of asserting a value. That check earned
  itself immediately: the first band tried, (5e-5, 6e-4) s/m, returned 10 % of
  its picks pinned at 6e-4, which look like measurements and are not.
  One registry change came with it: `RealDataset` gained `member` and
  `member_sha256`, because Zenodo publishes the hole's whole 578 MB logging
  archive as a single file and the digest that matters is the extracted
  member's — recompressing a zip changes the archive's hash and leaves the
  member's alone.
  The entry is still **fetched rather than vendored**, but the reason has
  changed and is recorded: CC0 removes the licensing objection, and 578 MB is
  now the only one.
- **The debond inverse is scored by `sonic_ml.bench`** (roadmap G.6), which
  was the last open piece of G.2. `sonic_ml.bench.debond` adds
  `evaluate_thickness`, the `ThicknessPredictor` protocol, `gap_regime_labels`,
  `format_thickness_scorecard`, and two predictors — `KrauklisThicknessPredictor`
  (the closed form, torch-free) and `MeanThicknessPredictor` (the no-skill
  reference). A trained `TrainedDebondInverse` satisfies the protocol directly,
  so both rivals go through one harness on identical held-out indices, with the
  same bootstrap CIs and per-regime rows every other predictor in the layer gets.
  **Two things differ from the Vs and bond harnesses, both deliberately.**
  Errors are in **log10 metres**, because the gap spans two decades by
  construction and a median error in metres would be set by the widest samples
  alone; `format_thickness_scorecard` also prints them as a percentage of
  thickness, which is the readable form. And the protocol takes **no
  `ArrayGeometry`**: a gap-width estimator reads the crack-wave dispersion
  curve, not the gather, so passing a geometry it cannot use would misdescribe
  what is being measured.
  **The per-regime rows immediately showed something the by-hand comparison
  could not.** Scored over all 240 samples — legitimate, since the closed form
  has no fitted state — the Krauklis estimator is not uniformly ~5 % off; its
  error is **six times worse on wide gaps than tight ones**:

  | krauklis_closed_form | n | medAE (decades) | error in *h* | 95 % CI |
  |---|---|---|---|---|
  | all | 240 | 0.0215 | 5.1 % | [0.0152, 0.0268] |
  | tight (< 100 µm) | 142 | 0.0106 | **2.5 %** | [0.0076, 0.0125] |
  | wide (≥ 100 µm) | 98 | 0.0664 | **16.5 %** | [0.0552, 0.0814] |

  That is the direction the physics predicts and the reason the residual model
  has anything to learn: a wider gap carries a faster crack wave and so a
  longer wavelength, which makes the 10 mm casing and 45 mm cement look thinner
  relative to it, and the half-space assumption fail harder. On the held-out
  split the learned inverse flattens the contrast to 0.7 % tight / 1.3 % wide —
  it does not merely lower the average, it removes the regime dependence.
  The harness reproduces the recorded G.2 figures exactly (log RMSE 0.0721 /
  18.1 % classical, 0.0107 / 2.5 % learned); it reports a **median** absolute
  error rather than an RMS, which on these errors is about a third of it — the
  gap between the two is the heavy tail, and both are now visible instead of
  one.
  One incidental fix: the drawn gap width was read out of `layer_params` in
  three places. `baselines.debond.gap_thickness` is now the single reader and
  the other two delegate to it.
- **`read_dlis_waveforms` falls back to a vendor parameter when a file
  declares no AXIS**, which a second real file showed is necessary. ODP Leg
  157 Hole 952A (LDEO-BRG, SDT tool, 1994) carries **zero AXIS objects**, and
  its `DSI0` parameter is the only record of the 10 µs sample interval — so
  the AXIS-only reader could read its waveforms but not say how they were
  sampled.
  That partly overturns the reasoning shipped with the reader. Preferring the
  RP66 standard record over a vendor naming convention is still right, but one
  file made it look sufficient and two show it is not.
  The fallback is deliberately timid, because a parameter carries no declared
  unit and guessing wrong is a factor-of-1000 error. It fires **only** when
  the file declares no time-unit axis *and* its `DSI*` parameters agree on one
  value; where they disagree — as on the FORGE file, which carries 40, 40, 40,
  10, 40 — deciding which belongs to a given channel is a vendor question and
  it raises instead, naming every candidate. The microsecond convention is
  *checked* rather than trusted: the implied record length must be
  sonic-plausible, so a value that would mean a 5 s record is refused.
  `sample_interval_source()` reports which route answered, since one is a
  unit-bearing standard record and the other rests on a convention.
  Verified on both files: ODP resolves to 10 µs via `DSI0` — independently
  confirmed by the archive's binary header, which came through a different
  conversion path entirely — and FORGE still resolves to 10 µs via its AXIS,
  the route that protects it from its own disagreeing parameters.
  A related diagnostics regression was caught and fixed in the same change: a
  channel with *two* time axes now keeps its "2 axes with a time unit" error
  instead of falling through and reporting the file as silent about something
  it declared twice.
- **A learned gap-width inverse for the debonded regime** (roadmap G.2).
  `sonic_ml.models.debond` adds `debond_features`, `DebondResidualNet` and
  `train_debond_inverse`.
  **It is not asked to predict the gap from the waveform, and that is the
  point.** The cased Stoneley mode is the only debonded branch that reaches the
  receivers, and it moves 0.05 % across a 100x change in gap against 1.0-1.5 %
  from the formation alone — so a waveform model would be fitting noise while
  scoring respectably against a careless metric. The input is the crack-wave
  dispersion curve, and the bar is the closed-form baseline above.
  **It predicts the residual** `log10(h_true) - log10(h_krauklis)` rather than
  the thickness. The output head is zero-initialised, so an untrained model
  reproduces the classical answer exactly: training starts at a known-good
  answer, a broken run degrades to it rather than to noise, and any gain is
  attributable. The residual also has a physical name — the Krauklis law
  assumes half-space walls, while the dataset has ~10 mm of casing against a
  comparable crack wavelength, so what there is to learn is the finite-layer
  correction. The features expose exactly what the baseline lacks: it sees the
  bounding moduli only, through the compliance `C`, and never the layer
  *thicknesses*.
  **Leakage is guarded explicitly**, because the gap thickness sits in
  `layer_params` one column from features that are legitimately used:
  `debond_features` drops that column, and a test perturbs the stored gap by
  7.3x while holding the dispersion curve fixed and asserts that not one
  feature moves.
  **Weights are selected on the validation split**, which is not a formality
  here. Run against real solver output the training loss reached exactly zero
  while validation loss rose, with the best validation epoch at **6 of 400** —
  a debonded dataset costs ~14 s a sample, so the feature count is comparable
  to the sample count it can afford, and keeping the last epoch returns a
  memorised model. `history` is now `(train, val)` pairs so that divergence is
  visible rather than inferred.
  **Measured on 240 samples** (192 train / 24 val / 24 test), gap 10-961 µm:

  | held-out test | log RMSE | error in *h* | median ratio | ratio IQR |
  |---|---|---|---|---|
  | classical (Krauklis) | 0.0721 | **18.1 %** | 0.978 | 0.104 |
  | learned residual | 0.0107 | **2.5 %** | 0.998 | 0.018 |

  About **7x better**, and not memorisation: best validation at epoch 88 of
  400 with validation loss falling throughout (0.00152 → 0.00047), and
  held-out 2.5 % against whole-dataset 2.3 %. The classical figure reproduces
  the independent 24-sample measurement exactly.
  **What that does and does not mean.** The residual model learned the
  finite-layer correction, which is what it was built to learn and what the
  layer-thickness features expose. But the dispersion curves it learned from
  are **noiseless** — deterministic solver output, no measurement noise and no
  picking error — so 2.5 % is the ceiling against a perfect forward model,
  not a field expectation. On real data the crack wave would first have to be
  *detected*, and at 63-620 m/s it arrives outside a normal record. The bar
  this clears is a modelling bar.
- **A closed-form microannulus-thickness baseline** (roadmap G.2, the `sonic_ml`
  consumer). `sonic_ml.baselines.CrackWaveThicknessBaseline` inverts the
  Krauklis law, `h = c^3 C rho_f / omega` with `C = sum (1-nu)/mu` over the two
  solids bounding the gap, to read the gap width straight off the crack-wave
  dispersion curve.
  It is a genuinely independent estimator rather than a circular one: the
  dataset's curves are numerical roots of the full modal determinant, and this
  law is the analytic asymptote that validated that determinant to 0.02 %. And
  it needs **no fitted calibration**, unlike the bonded
  `StoneleyBondBaseline` — so it is a harder bar for a learned model, spending
  none of the training split.
  A CBL-amplitude baseline is still not available and would still be a
  strawman: these gathers carry no casing-ring arrival, and
  `CasingRingAugmentation` deliberately draws ring amplitude independently of
  bond. The crack wave is the signal that is genuinely present.
  Scored in the ratio domain (`median_ratio`, `ratio_iqr`, `log_rmse`,
  `rank_correlation`) because the gap spans two decades, so an RMS in metres
  would be set by the widest samples alone — and because that separates a
  constant bias, which a recalibration fixes, from scatter, which it does not.
  The Krauklis law treats the bounding solids as half-spaces while the dataset
  has ~10 mm of casing and ~45 mm of cement against a comparable crack
  wavelength, so a systematic bias is expected; the score reports it rather
  than absorbing it. Measured on 24 generated samples spanning 11-837 µm (a
  76x range): **rank correlation 0.991**, median ratio 0.935 — so the
  half-space bias is only ~6.5 % — and `log_rmse` 0.085, about **21 % in h**,
  falling to **18.1 %** once that single constant is removed. A learned model
  has to beat ~18 % across two decades, from an estimator that spent no
  training data at all.
  The debonded bundle needed **no loader change**: `DatasetBundle` already
  reads `mode_names` and `layer_params` from the file, and `cased_features`
  was already generic over layer count, so a 3-layer 2-mode bundle loads as
  `is_cased` schema v4 unmodified.
- **A debonded cased-hole dataset generator** (roadmap G.2), and a measurement
  that changed what it should be. `scripts/gen_surrogate_dataset.py` gains
  `MicroannulusPriors`, `DEBONDED_MODES`, `generate_debonded_dataset` and a
  `--debonded` CLI flag, drawing a fluid microannulus between casing and cement
  — the standard debonding model, and a bound-mode problem now that A.5 has
  shipped.
  **The obvious build would not have been invertible.** The item was framed as
  the cased dataset in the debonded regime: same Stoneley mode, gap width as
  the label. Measured over 1–12 kHz, the cased Stoneley curve moves **0.05 %**
  when the gap goes from 10 µm to 1 mm — a 100× range — while the formation
  shear velocity alone moves it 1.0–1.5 %. The mode responds to the *slip
  interface*, not its width: bonded → debonded is a **4.14 %** shift and it is
  the same shift at every thickness.
  **The crack wave carries the width, at roughly 100:1.** Over that same range
  its velocity moves **+301 %** (4.78× measured, against 4.64× from the
  Krauklis `h^(1/3)` law) while the formation moves it 0.03 %. So the dataset
  carries both branches: Stoneley for a bonded/debonded state, crack wave for
  the gap. Thickness is sampled log-uniformly, which is uniform-in-observable
  for a cube-root law.
  The crack wave is **recorded but not injected** — `ModeSpec` gained
  `inject=True` for it. At 63–620 m/s it reaches the 3 m near offset between
  4.8 ms and 47.6 ms against a 5.12 ms record, so a planted arrival would be
  fiction. Its dispersion curve is the product.
  No schema change: the gap is written into `layer_params` as an ordinary
  layer with `vs = 0`, so its thickness is already carried by v4.
  `bond_index` keeps its range and its direction (1 = best bond) but is driven
  by gap width here and cement stiffness in `generate_cased_dataset`, so the
  two datasets **must not be pooled**.
  Cost is the reason `--debonded` defaults to a 32-point grid: ~14 s a sample
  against ~0.5 s bonded, since the microannulus solvers run ~0.45 s per
  frequency for the two branches together.
- **`fwap.io.read_dlis_waveforms` reads the per-receiver waveforms in a DLIS**
  (roadmap F.3). `read_dlis` returns one value per depth and skips everything
  else, which is exactly where a full-waveform sonic record lives — so until
  now the package could not reach, through its own API, the data it was
  measured on. Every real-data number in this changelog was produced by calling
  `dlisio` directly.
  The new reader returns `DlisWaveforms`: the channel as
  `(n_depth, n_receiver, n_sample)`, the depth axis, and one `DlisAxis` per
  trailing dimension. **The acquisition geometry comes out of the file**, not
  out of a constant: `sample_interval()` and `offsets()` read the RP66 v1 AXIS
  records and convert from whatever unit is declared there. On the Utah FORGE
  DSI file that is 10 µs and eight receivers 6 in apart starting at 7.874 m.
  Which axis is which is decided by the declared **unit**, never by the
  AXIS-ID string, because AXIS-ID values are producer-defined and units are
  not. An axis list that does not match the channel's dimensions is reported
  as no axes at all rather than guessed at.
  Only the requested channel and the index channel are read, so this does not
  pay for the rest of the frame: one monopole channel out of the 88 MB FORGE
  pass takes **1.1 s**, against ~100 s to materialise the frame.
  `DlisCurves` gained `waveform_channels`, a `{name: shape}` map of the
  channels `read_dlis` skipped, so they are discoverable rather than invisible.
  End-to-end on the real log through the public API only — no `dlisio` import
  at the call site — `stc` + `track_modes` reproduce the previously
  hand-assembled result exactly (compressional 86 % agreement, median −0.57 %),
  including with the file's true 7.874 m first offset rather than the 2.7432 m
  that had been assumed. Slowness depends on receiver *spacing*, so the two
  agree; arrival times do not, and the file's value is the right one.

### Fixed
- **`write_dlis` allocated a 4 GiB buffer for every file, whatever its size.**
  `dliswriter`'s `output_chunk_size` defaults to `2**32`; fwap now passes 8 MiB.
  Measured on a 9124-byte output: **59.16 s and ~8.3 GB peak RSS before, 0.34 s
  and ~89 MB after**, byte-identical. On a memory-constrained machine the old
  path could fail outright rather than merely crawl. `tests/test_io.py` drops
  from minutes to **2 s**.

### Documentation
- **The Sphinx build renders correctly again.** Six docstrings produced wrong
  output rather than merely warnings: `track_to_log_curves`'s VTI table was
  malformed (a cell overflowed its column, so the table silently lost its
  shape); `viterbi_pick_joint`, `viterbi_posterior_marginals` and
  `synthesize_lwd_gather` wrapped a comma-separated parameter-name list across
  lines, which breaks the numpydoc field list and dropped the descriptions
  after it; `estimate_dip`'s custom `Strategy` section was swallowed into
  `Parameters`, rendering its numbered steps as fake parameters
  (`:param 1. Coarse grid search over ``(alpha:`); and `fwap.wavesep`'s module
  docstring had an over-indented continuation plus a bare `|f|` that RST read
  as an undefined substitution. Three section underlines in `fwap.cylindrical`
  were a character short.
- **Twenty-six documents under `docs/` were built but unreachable** — the
  solver design plans, the book reading notes, the notebooks' data README and
  `possible_extensions`. They are now listed in hidden toctrees, which keeps
  the links the validation notebook and `roadmap.rst` make into them working.
- One `myst` link in `docs/plans/cylindrical_biot_F_2.md` still pointed at
  `fwap/cylindrical_solver.py`, which became a package directory.

### Added
- **The compressional-pick defect real data exposed is diagnosed, documented
  and reproduced in CI** (roadmap F.1). It is mode confusion, not imprecision:
  on 143 of 150 bad depths `track_modes` assigned the *same* STC peak to P and
  to S, reporting shear slowness (91.6 us/ft median) as compressional (52.0
  true). The error histogram is sharply bimodal — one cluster at 0 %, one at
  +77 %, nothing between +15 % and +55 %.
  The mechanism is structural. Mode ordering is enforced on arrival *time*,
  never on slowness, so nothing requires P to be faster than S; and the P prior
  window (40-140 us/ft) contains the shear arrival of most formations. When
  shear is the more coherent of the two — 0.946 against 0.791 here — the
  `scored` rule's `time_penalty` cannot cover the gap, and both modes take the
  same peak.
  **`viterbi_pick_joint` already avoids it**: on identical STC surfaces, in the
  same runtime, it confuses 34 rather than 143 and raises compressional
  agreement with the vendor from 62 % to 89 % of depths, with shear unchanged
  at 96 %. So the package contained the answer and said nothing at the call
  site; `track_modes` now carries a warning with these numbers and a pointer.
  Two new tests reproduce both behaviours on a seeded synthetic — a weak
  compressional arrival under a strong shear one — so the finding survives
  without the 808 MB fixture.
  Retuning `time_penalty` is measured to be the wrong lever: the value that
  would flip those depths has median 0.18 and 90th percentile 0.43 against a
  default of 0.1, and raising it that far would bias every late mode, which is
  what the `max_coherence` rule exists to prevent. The repair that *was* made
  is structural instead — see the next entry.
- **The greedy picker no longer assigns one arrival to two modes** (roadmap
  F.1), which repairs the confusion above at its cause.
  `fwap.picker.pick_modes` and `fwap.picker.track_modes` gained
  `resolve_mode_collisions=True`: after the greedy pass, when two modes have
  selected the same STC peak the faster-labelled one re-picks from its own
  candidate pool with that slowness as a strict upper bound.
  **Which label is wrong is not decidable in general, and the rule does not
  guess.** Both directions occur in real data — on the DSI log the shared peak
  is the shear arrival and P is the mislabel; on a slow-formation synthetic it
  is the compressional arrival and S is. So a mode with no admissible faster
  candidate is left exactly as it was, on the reasoning that "nowhere faster to
  go" is evidence it holds the *right* arrival. Nothing is ever dropped and no
  mode is ever moved to a slower candidate, so a depth can never come out worse
  than the greedy result.
  On the same 400 DSI depths, agreement with the vendor's `DTCO` rises from
  62 % to **95 %**, and the count of depths where P is not strictly faster than
  S falls from 143 to **5**. It beats `viterbi_pick_joint` on this log (89 %)
  at the same runtime and with two more depths picked. The rule changed the P
  pick at 138 depths, every one a collision, made 129 of them correct, left the
  shear pick **bit-identical at all 400** (96 % throughout), and damaged **none**
  of the 250 depths that were already right.
  Confirmed on an independent second logging pass of the same well, over a
  different depth interval: 70 % → 86 %, 72 unordered depths → 2, and again no
  damage to any of the 283 depths that were already right.
  `viterbi_pick_joint` is still the better tool where the confusion is not an
  exact collision — a global cost can reject an assignment a local rule cannot
  see — and 7 of these 400 depths are of that kind, plus 3 that end up a single
  slowness cell apart, which is deliberately not treated as a collision.
  **This changes shipped picker output.** Pass `resolve_mode_collisions=False`
  for the previous behaviour; the tests that pin it are kept under that flag.
- **The first real sonic log in the test registry, and the first time this
  package has been scored against a vendor's answers** (roadmap F). A
  Schlumberger DSI run from Utah FORGE well ME-ESW1 is registered as
  `forge_dsi_las` (CC BY 4.0, fetched not vendored), with tests covering its
  curve set and the physical ordering of its reference picks.
  The registry now accepts `kind="dlis"` alongside `las` and `segy`.
  **What it found.** The companion 808 MB DLIS carries the per-receiver
  waveforms — `PWF1`-`PWF4`, each `(10839, 8, 512)`: eight receivers, 512
  samples, for lower dipole, upper dipole, monopole Stoneley and monopole P&S.
  Acquisition parameters were read from the file rather than assumed (10 us
  sampling on the monopole P&S, 6 in receiver spacing, 9 ft to the first
  receiver, zero firing delay), and `DTCO`/`DTSM` agree between the LAS and the
  DLIS to 5e-5 us/ft over ~10 800 depths, so the two are the same processing run
  and the data is scoreable.
  Running `fwap.stc` + `track_modes` over 400 contiguous frames against
  Schlumberger's own picks: **shear matches `DTSM` to a median +0.12 %** (MAD
  2.6 %, 96 % of depths within 10 %). **Compressional does not** — median
  +2.29 % but mean 27 % high and only 62 % within 10 %, a bimodal failure in
  which about a third of depths pick a later arrival as P. That defect is
  invisible to every synthetic test in the suite, because the synthetics are
  generated by the same forward model the picker is scored against. Recorded in
  `plans/log_output.md` and open in the roadmap; not yet diagnosed.
  The waveform comparison is **not** a CI test: the fixture is a 471 MB zip
  containing an 808 MB DLIS, and `fwap.io.read_dlis` skips multi-dimensional
  channels so it cannot read the waveforms at all. Both are recorded as open.
- **`crack_wave_dispersion` — the second root family, and the more sensitive
  debonding indicator.** Where the Stoneley-like mode shifts ~1 % on debonding
  and then barely moves, the crack (Krauklis) wave is *guided by the gap* and
  scales as `(f h)^{1/3}`: 68 m/s at a 1 um gap to 620 m/s at 1 mm, at 8 kHz.
  That closed form is invertible, so a measured crack-wave velocity gives a gap
  thickness directly.
  Reproduces the analytic speed to **0.02 % at a 1 um gap** through the public
  surface, and the cube-root exponent is measured in both variables rather than
  assumed — 1/3 to within 0.02 in `h` and in `f`. The scan window is not
  derived from that formula (it runs from the determinant's representability
  limit to the bound floor), so the check stays independent rather than
  self-confirming.
  **It needed a spurious-root filter, and building one is most of this
  change.** `stoneley_dispersion_microannulus` stops at the first sign change
  above the bound floor and never reaches the phase velocities where the
  elastic propagators lose precision; this function scans down to them
  deliberately, and sign changes there get read as roots — the defect this
  module has shipped twice. Over 270 sampled configurations one produced a
  duplicated pair near 4 m/s. The filter is grid stability: the scan runs twice
  at different resolutions and lower endpoints, and only roots common to both
  survive. On that configuration the spurious pair appeared in one grid of six
  while the genuine roots appeared in all six and agreed to 1e-9. Across the
  same sweep the API now returns no sub-20 m/s value anywhere, and the filter
  holds at every resolution from 60 samples up.
  A detail worth knowing: the artefact's **existence** is platform-dependent,
  not just its value. The configuration that produces a duplicated pair near
  4 m/s on the development machine produces only the two genuine roots on CI.
  That argues for the filter rather than against it — no caller can rely on the
  artefact being absent on their machine — and it is why the tests assert the
  API's answer rather than the artefact's presence. The filter's own contract
  is tested directly instead, on a synthetic determinant with a root confined
  to a needle that one scan grid samples and the other does not.
  The alternative filter was measured and rejected first: the elastic
  propagator's determinant identity is violated by 1e232 at operating points
  where the crack root is correct to 1e-9, so gating on it would have removed
  the capability entirely.
- **`stoneley_dispersion_microannulus` and `FluidAnnulus` — the first public
  entry point for the debonded regime.** Stoneley dispersion for a stack of the
  form `borehole fluid | casing | microannulus | cement | formation`, wrapping
  the 11x11 assembly below.
  **The selection rule is the substance of this, not the wrapper.** The
  determinant carries two families of bound root, and a bracket that assumes
  one is exactly the `n=0` defect that shipped once already. The rule used is
  structural rather than tuned: the Stoneley-like mode is the fastest bound n=0
  mode, so the first sign change above the bound floor is it whatever else the
  stack supports. Verified against an independent scan, and pinned as
  independent of both the caller's frequency grid and the scan resolution —
  each frequency is solved on its own, with no frequency marching.
  `FluidAnnulus` is a distinct type from `BoreholeLayer` rather than a layer
  with `vs = 0`, because a gap is not a limiting case of an elastic layer here:
  a compliant solid drags the bound-mode bracket floor down with its shear
  velocity, while a gap's floor is its acoustic velocity. That separation is
  what makes the configuration reachable, so it is load-bearing.
  The crack wave is a separate entry point rather than a `branch` argument
  here, because the two families are qualitatively different and only one of
  them needs the spurious-root filter described above: this function's window
  never reaches the velocities where that matters.
  Also documented at the API surface: a thin gap does **not** converge to the
  bonded stack. It converges to a frictionless slip interface, 1383.45 m/s
  against 1400.04 m/s bonded at 8 kHz, a 1.2 % offset that does not close.
- **A determinant identity for the elastic layer propagator, found while
  building the above.** `det P = (r_inner / r_outer)^2`, with no frequency,
  velocity, density or `k_z` in it — the 4x4 counterpart of the fluid
  element's Bessel Wronskian, arising the same way (each of the two `(I, K)`
  pairs in `E` contributes one factor of `1/r`).
  `_layer_propagator_n0` has shipped for a long time with no check on its
  *value*: the existing tests pin the group law and the round trip, both of
  which a systematically wrong `E` would still satisfy. This one is arithmetic
  from outside the module and breaks on any swapped Bessel order or sign slip.
  Its accuracy range is measured on the same axis as the fluid element's — the
  dimensionless span `s * dr` — at machine precision below 2, ~1e-9 by 5, and
  no significant digits by 20.
- **Global assembly for the fluid microannulus
  (`_modal_determinant_n0_microannulus`) — the n=0 modal determinant for
  `borehole fluid | casing | microannulus | cement | formation`.** Builds on the
  fluid-annulus element below. The stack splits into two elastic blocks joined
  by a two-component state `(u_r, sigma_rr)`, giving an 11x11 system rather than
  the all-elastic 7x7: 1 borehole + 4 + 4 + 2 unknowns against 3 + 1 + 3 + 4
  interface conditions. The annulus amplitudes are folded out through the fluid
  propagator, exactly as layer-internal amplitudes are in the 7x7, so extra
  layers in either block leave the size unchanged.
  **Validated against the Krauklis crack wave**, the analytic phase velocity of
  a wave guided by a thin fluid gap between elastic walls,
  `c = (omega h / (C rho_f))^{1/3}` with `C` the sum of the two wall compliances
  `(1 - nu)/mu`. That formula comes from lubrication flow plus quasi-static
  half-space compliance — no Bessel functions, no cylindrical geometry, no
  shared code — and it fixes an *absolute* velocity, not just a scaling. The
  solver reproduces it to 0.02 % at a 1 um gap, 0.2 % at 10 um and 1.7 % at
  100 um, departing as `k h` stops being small, and follows its cube-root
  scaling in frequency and gap thickness and its dependence on wall stiffness
  and gap-fluid density. Where the oracle stops applying is measured too: the
  mode is confined within `~1/k_z` of the gap, so walls thinner than that fall
  away from the analytic value (0.64 of it for a 2 mm casing against a 6 mm
  decay length).
  Also checked against an independently assembled 13x13 form that keeps the gap
  amplitudes explicit, and for invariance under subdivision of either elastic
  block.
  **There is no reduction to the existing solver, and that is the physics.**
  The `annulus_thickness -> 0` limit is a frictionless *slip* interface, not the
  bonded stack: shear traction stays zero on both faces and `u_z` stays free
  however thin the gap. Measured at 8 kHz, the Stoneley-like root converges as
  `O(h)` to 1383.45 m/s against 1400.04 m/s bonded — a 1.2 % offset that does
  not close. That is why the validation had to come from outside the module.
  The assembly carries **two root families**, a Stoneley-like mode and the slow
  gap mode, which is recorded as a trap for the root finder that comes next: the
  n=0 branch-selection defect fixed earlier came from a bracket that assumed a
  single root. The root set is checked to be independent of scan grid and
  window. Still private, for that reason — choosing which family a public
  dispersion curve follows is a separate decision.
- **Fluid-annulus propagator element for n=0 (`_fluid_layer_e_matrix_n0`,
  `_fluid_layer_propagator_n0`) — the first piece of the microannulus model for
  the debonded regime.** A fluid annulus differs from an elastic one in ways
  that change the shape of the problem rather than its numbers: two wave
  amplitudes rather than four, shear traction identically zero, and axial slip
  permitted at both faces, so the propagated state is the reduced pair
  `(u_r, sigma_rr)`.
  Verified against an identity that comes from outside the module. The Bessel
  Wronskian `I0(x)K1(x) + I1(x)K0(x) = 1/x` collapses the determinant to
  `det E_f(r) = -1/(rho omega^2 r)` and hence `det P_f = r_inner/r_outer`,
  with **no dependence on frequency, velocity, density or `k_z`** — so a sign
  slip or a swapped Bessel order breaks it immediately. The state matrix is also
  checked against a numerical derivative of the pressure, testing it against the
  momentum equation it encodes rather than against the algebra used to derive
  it.
  Its accuracy range is characterised rather than assumed: error tracks the
  Bessel span `F * (r_outer - r_inner)` — machine precision to about 2, ~1e-11
  by 7, no significant digits by 20. A debonding gap is microns to millimetres,
  putting the span below 0.1, so this is a documented limit rather than a
  practical one. Recorded because the same exponential-range failure produced
  spurious roots elsewhere in this module.
  **Not reachable from the public API yet, deliberately.** `BoreholeLayer`
  cannot express a fluid and the global assembly changes shape when one is
  present. Shipping a public layer type no solver accepts would be worse than
  shipping nothing; the element is verified in isolation first so that any later
  failure is attributable to the assembly.

### Fixed
- **The suite's last two warnings.** `demo_stc_picker` and
  `demo_pseudo_rayleigh` each drew a legend over a dense `pcolormesh`
  coherence map with matplotlib's default `loc="best"`, which searches the
  plotted data for the emptiest corner; matplotlib warns that this is slow.
  Both now pass an explicit `loc`, with a comment saying why so it does not get
  tidied back. Only these two of the seventeen bare `legend()` calls in the
  demos were affected -- the rest draw line plots and stay below matplotlib's
  threshold -- so the other fifteen are left alone.
- **A near-miss recorded rather than a defect: the propagator identity above is
  *not* a valid gate on root quality, and using it as one would have silently
  removed the crack-wave capability.** The obvious next move after finding it
  was to reject roots wherever it fails. Measured instead: at a 1 um gap and
  8 kHz the crack root is fixed to 1.5e-9 across a tenfold range of cement
  thickness over which the identity degrades from 1e0 to **1e232**. The mode is
  confined within `~1/k_z` of the gap (1.35 mm here), so once the block is much
  thicker than that its far field cannot influence the root, and the
  catastrophic error lives entirely in the growing branch the root condition
  never sees. A test pins both halves.
  The general form is worth keeping: a conditioning measure on an intermediate
  quantity bounds *that quantity*, and says nothing on its own about a root
  computed from it.
- **Two ways a determinant sweep could escape its own contract, found while
  building the microannulus assembly.** A determinant a root finder scans must
  return `NaN` where it cannot be formed — never warn, never raise. Neither held
  at low trial phase velocity. The unscaled `I_n` in the layer state matrices
  overflows once the Bessel argument passes ~709, and because that happens
  *inside* the helper, checking the result for finiteness cleaned up the value
  but not the `RuntimeWarning`. Separately, `numpy.linalg.solve` inside the
  fluid propagator hit an exactly singular matrix — `K_n` underflowing to zero
  while `I_n` overflowed — and raised `LinAlgError` out of the sweep.
  `_modal_determinant_n0_microannulus` now bounds every Bessel argument before
  building any state matrix, using `kz * r_outermost` (an upper bound on all of
  them) against `log(sqrt(DBL_MAX))` — the same square-root-of-double-max
  headroom rule the product guard already used, expressed in the exponent. It
  also gates the fluid propagator on its own exact determinant identity
  `det P_f = r_inner/r_outer`, which fails well before the entries stop being
  finite; that turns the element's documented validity range into an enforced
  one. Both guards are exercised, and a 1600-point sweep asserts no
  `RuntimeWarning` escapes. Existing assemblies are untouched.
- **Compliant layers in a cased stack returned spurious roots instead of
  `NaN`.** Found while starting the free-pipe / debonded item. A very compliant
  elastic layer drives the propagator's dynamic range past double precision; the
  7x7 determinant then becomes meaningless and the bracket search finds sign
  changes in it and reports them as roots — finite slownesses corresponding to
  phase velocities of **3-12 m/s against a 1500 m/s fluid**. Some configurations
  produced these with **no warning at all**, so a warning filter would not have
  caught them.
  `_modal_determinant_n0_cased` now checks that the propagator product can be
  formed in double precision before forming it, and returns `NaN` otherwise.
  Checking the *result* for finiteness is not enough: the overflow is raised by
  the matmul itself, so a post-hoc test cleans up the determinant but leaves the
  warning. The bonded regime is bit-identical — cement stiffer than the fluid
  converges across the whole band at unchanged velocities, pinned by a test.
  This also removes the intermittent `overflow encountered in matmul` and
  `invalid value encountered in det` warnings that had been appearing in
  coverage runs.

### Changed
- **`docs/roadmap.md` becomes `plans/roadmap.md`, carrying only the open
  items.** The old file had grown to about a thousand lines of which the great
  majority described work that had already shipped, so the open items were
  merged into a new `plans/roadmap.md` and the closed ones dropped: the 0.4.0
  release notes and three post-0.4.0 completeness sweeps, the closed A.3 / A.4
  solver items, sections B / C / E in full, three closed `sonic_ml` items, and
  the pre-implementation problem statements for two solvers that now ship.
  Nothing is lost — `CHANGELOG.md` is the record of what shipped and when, the
  deleted file is in git history, and the new file ends with a table mapping
  each dropped section to where its detail lives.
  What is kept is deliberately verbatim where it carries measured numbers, and
  the section labels (`A.1`, `A.2`, `A.5`, `D`, `F`, `G`) are unchanged, because
  code comments in `fwap/`, `scripts/` and `tests/` cite them.
  `docs/roadmap.rst` becomes a short pointer page: the roadmap is no longer part
  of the built documentation. That stub previously never rendered at all — it
  shared a document name with `roadmap.md`, so Sphinx resolved `roadmap` to the
  Markdown file, ignored the stub, and emitted `multiple files found for the
  document "roadmap"` on every build. Measured against the build before this
  change: exactly one warning removed, none introduced.
  References repointed in `docs/possible_extensions.md`, `plans/roadmap_1.md`,
  `plans/learning.md`, `plans/log_output.md`, `.pre-commit-config.yaml`,
  `pyproject.toml` and four code comments
  (`fwap/cylindrical_solver/_leaky.py`, `fwap/anisotropy/_vti_inversion.py`,
  `scripts/gen_surrogate_dataset.py`, `tests/test_cylindrical_solver.py`).
  Historical `CHANGELOG.md` entries and the archived session notes in
  `plans/log_output.md` keep the old paths, since both record the tree as it
  stood at the time.
- **The free-pipe / debonded item (roadmap G.2) is re-diagnosed and decoupled
  from the `n=1` leaky-mode work.** It had been filed behind that item on the
  grounds that "reaching debonding needs a leaky-mode cased forward model".
  Measurement splits the question in two. Modelling debonding as *soft cement*
  is genuinely blocked, and the documented restriction is correct: the cased
  Stoneley converges over the whole band down to `cement_vs = V_f`, is partial
  just below, and is gone by 1200 m/s, because `_stoneley_kz_bracket_cased`
  takes its bound-regime floor from the softest shear velocity anywhere in the
  stack. But a **fluid microannulus** — the standard debonding model in
  cement-bond logging — is a different configuration that argument does not
  exclude, since a fluid contributes no shear floor. It also cannot be
  approximated by a compliant elastic layer, precisely because an elastic layer
  does contribute one; measured, that fails at any thickness down to 0.2 mm.
  So the blocker is not a Riemann-sheet derivation but the absence of a
  fluid-annulus element in the propagator, which is an implementation task of
  known shape. Recorded in `plans/roadmap_1.md` with the scope.

### Added
- **`trapped_pseudo_rayleigh_dispersion` — the bound half of the
  pseudo-Rayleigh family, which no public function reached before.** The family
  splits by phase velocity: for `V_f < c < V_S` both formation waves are
  evanescent while the fluid field oscillates radially, so the mode is a genuine
  trapped resonance with a real `k_z` and no attenuation; above `V_S` the shear
  wave propagates and `pseudo_rayleigh_dispersion` takes over with a complex
  one. `stoneley_dispersion` cannot return these either — it brackets from
  `omega/min(V_S, V_f)` upward and so covers only `c < V_f`.
  Several coexist: three at 30 kHz in a 0.10 m hole through a 4000/2300/2500
  formation, six by 50 kHz, each above its own cutoff. A `branch` argument
  selects the radial order using the same convention as the leaky function
  (descending `k_z`, so the fundamental is the slowest of the trapped modes),
  and frequencies below a branch's cutoff return `NaN`.
  They were found while building the biorthogonality oracle, and that oracle now
  validates them: a test checks Auld's relation across the three trapped modes
  *and* the Stoneley mode at the same frequency, so the new function is tied to
  physics rather than to itself.
  Unlike the leaky sister function this one needs no frequency marching — the
  roots are real and simple, so each frequency is solved independently. A test
  pins the consequence, that the result does not depend on the frequency grid at
  all, and another pins that the scan resolution does not decide how many
  branches exist.

### Changed
- **Kramers-Kronig checked; it does not apply to the modal solver, and the
  place it does apply is the attenuation module's test synthetic.** The
  candidate was listed on the grounds that KK relates `Re(k_z)` and `Im(k_z)`
  across frequencies and so cannot be satisfied by a single root. One line of
  data disproves it for modal dispersion: a subtracted KK relation says zero
  attenuation at every frequency forces zero dispersion, and the bound Stoneley
  mode is exactly lossless (`attenuation_per_meter is None`) while its phase
  velocity moves **8.26 %** between the tube-wave and Scholte limits. KK follows
  from causality of the *constitutive* relation; waveguide dispersion is
  geometric, and a lossless hollow waveguide is dispersive for the same reason.
- **The attenuation tests' synthetic gather is acausal, and a causal
  counterpart is now covered alongside it.** `_attenuated_gather` multiplies the
  spectrum by `exp(-pi f t / Q)` and leaves the phase alone; constant-Q
  amplitude loss without the Kolsky-Futterman velocity dispersion violates KK,
  and the result carries energy arriving before the geometric arrival —
  pre-arrival energy fraction 1.5e-7, against 4.9e-12 for the causal version.
  This is **not a bug**: both estimators read `|S(f)|` only, so the missing
  phase cannot bias them directly. But they window in time, and dispersion
  reshapes the waveform inside the window, so the route is real. On the causal
  gather the centroid estimate moves from 62 to 41 against a planted Q of 50 and
  the spectral-ratio estimate from 117 to 81 — both *closer* to truth, so the
  existing tests understate the estimators rather than flattering them. Four new
  tests cover the causal case, including one asserting the two gathers differ in
  causality and one asserting they give materially different Q.
  The dispersion sign was wrong on the first attempt, which makes the signal
  more acausal rather than less and produced a spurious "causality doubles the
  recovered Q" result. It was caught by measuring pre-arrival energy rather than
  re-deriving the algebra; both the wrong numbers and the method that caught
  them are recorded in `plans/log_output.md`.
- **Modal biorthogonality checked; it holds to ~1e-13 and is the first oracle
  here that needs two solutions at once.** The conservation-law survey predicted
  this one would work, on the criterion that a check evaluated on a single mode
  in a region where that mode already satisfies the governing equations must
  come back exact and mean nothing. Auld's waveguide reciprocity relation
  couples two *different* eigenvectors and so escapes that trap.
  The test set is richer than expected: in a fast formation the n=0 bound
  spectrum holds the Stoneley mode (`c < V_f`) **and** the trapped
  pseudo-Rayleigh modes (`V_f < c < V_S`) — four bound modes at 30 kHz, six at
  50 kHz, all azimuthal order 0, so orthogonality among them is not the trivial
  angular-integral kind. (`stoneley_dispersion` returns only the first: its
  bracket stops at `omega/V_f`, so the trapped modes are not exposed by any
  public function.)
  Three tests: the eigenfunctions satisfy the boundary conditions they were
  built from, `S_mn - conj(S_nm)` vanishes to 1e-13 off-diagonal while the
  diagonal stays O(1), and the *wrong* bilinear form — one term of the pairing
  instead of the difference — leaves off-diagonals near 1e-2, ten orders worse.
  That last one makes the tolerance evidence rather than a fitted constant.
  Two measurement traps were hit and are recorded in `plans/learning.md`: a sign
  convention mismatch between the determinant's matrix rows and the field
  expressions, which the boundary-condition check caught before it could be
  mistaken for a failed orthogonality relation; and adaptive quadrature
  manufacturing a 1e-4 residual that *grew* with integration span, the tell that
  the error was numerical rather than physical.
- **`plans/learning.md` gains a section on choosing what to measure**, specific
  and general. The specific part is six questions in the order they have
  actually produced findings here — which return *fields* nothing has looked at;
  what the closed form leaves out; what the check would do to a wrong answer;
  whether the answer depends on something it must not; where the check itself
  expires; and whether a quantity is a property of the system or of the run. The
  general part is one test: what would have to be true of the *world*, rather
  than of the program, for this measurement to come out right? If the answer is
  "nothing in particular", it is a tautology however elaborate — which is what
  killed the fluid-only energy balance, the momentum balance and interface flux
  continuity, and what marked biorthogonality as worth attempting.
- **The n=1 / n=2 rigid-pipe cutoff candidate does not work, and the reason is
  structural.** `plans/learning.md` proposed checking the flexural and
  quadrupole cutoffs against their rigid-pipe closed forms, as PR #61 did for
  n=0 using the appropriate Bessel zeros. The zeros are the easy part
  (`j'_{n,1}` = 3.8317 / 1.8412 / 3.0542); the premise is wrong. The n=0 check
  applies to `pseudo_rayleigh_dispersion`, a *fluid-column* resonance, whereas
  `flexural_dispersion` and `quadrupole_dispersion` return the *fundamental*
  interface modes at their orders. The solver exposes no n=1 or n=2 counterpart
  of pseudo-Rayleigh, so there is nothing for the formula to describe.
  Measurement settles it independently. The cutoff does scale cleanly as `1/a`
  (`f_c * a` constant to 1.4 % and 0.5 % over a 3.3x span of radius), but its
  log-log sensitivities are **0.87-0.89 on `V_S` and 0.08-0.13 on `V_f`** — a
  58 % change in `V_f` moves the cutoff 4 %. A fluid-column cutoff is
  fluid-controlled; these are shear-controlled. And the rigid-pipe form is only
  defined for `V_S > V_f`, where both solvers are separately known to be
  defective (roadmap A.2), so the comparison cannot be rescued by changing
  regime either. Three tests pin the `1/a` scaling, the shear-control, and the
  mismatch with the closed form.
- **`plans/learning.md`** gains the full energy-balance derivation (fluxes,
  the `2 Im(k_z) P_z = P_r` balance, and why the amplitude cancels), and a
  survey of which *other* conservation laws are worth trying. Linear momentum
  is not: for a single mode the axial momentum flux is the energy flux times
  `|k_z|^2 / (omega Re(k_z))`, a constant across the cross-section, verified to
  six digits — so its balance reduces to the same identity. Angular momentum
  and interface flux continuity fail the same way. Modal biorthogonality and
  Kramers-Kronig causality survive the filter, because both involve more than
  one solution.

### Added
- **`plans/log_output.md`** — the measured tables behind the numbers quoted in
  the plans, the roadmap and this changelog, with provenance stated: which
  values are asserted by tests (and so trustworthy), which are one-time
  readings, and which are machine-specific. Withdrawn numbers are kept and
  marked rather than deleted, so they are not re-derived and re-believed.

### Changed
- **Layer-subdivision invariance added as an oracle for the layered propagator;
  the layer-order candidate it replaced was simply wrong.** `plans/learning.md`
  listed "swapping layer order should leave the dispersion invariant" as a
  candidate. That is false for a cylindrical stack — the layers sit at
  *different radii*, so exchanging them moves material from one radius to
  another and changes the medium, by ~1 % here. The premise came from
  plane-layered intuition and had never been checked.
  What does hold is **subdivision**: relabelling one homogeneous annulus as
  several adjacent layers with the same properties is exact to ~1e-15, across
  n=0/n=1/n=2, open-hole mudcake and cased steel-plus-cement stacks, splitting
  the inner or the outer layer, in slow and fast formations. It reaches what the
  order-swap idea was aiming at — interface matching and propagator composition
  across more than one boundary, which no single-layer test exercises. Verified
  non-vacuous: a thickness error of one part in ten thousand moves the answer
  nine orders above the 1e-15 floor. It remains a *consistency* oracle, so an
  error common to every interface would cancel; that is stated rather than
  glossed.
- **A redundant layer is not always transparent — the neighbouring invariance
  has a validity range, now documented.** Appending a layer whose properties
  equal the formation is physically a no-op, and the solver treats it as one
  only while the radial dynamic range across that layer stays moderate. A
  0.15 m formation-equal layer shifts the 100 kHz answer by **14 %**, a 0.05 m
  one fails at 400 kHz, and both calls return finite, plausible slownesses.
  Which side is wrong was settled from outside the layered solver, with
  `scholte_speed`: at 100 kHz the wavelength in the 2 cm mudcake is ~1.6 cm, so
  the mode rides the innermost layer and must approach *that* layer's Scholte
  speed. The plain stack does, to 0.05 %; the padded stack does not. Where the
  padded answer lands is not stable, and neither is how far off it is — the same
  stack has returned 289 m/s and 1095 m/s, disagreeing with the plain answer by
  7 % on one machine and by a factor of four on another. That is itself the
  diagnosis: a root search returning somewhere different for identical physics
  has lost precision rather than found another branch. The test asserts only
  that transparency is lost somewhere in the range, which is the one stable
  claim.
  **Calibration, because the first reading was too alarming:** genuine layers
  with real contrast keep converging correctly at every thickness tried and fail
  cleanly to `NaN` rather than to a wrong number. The defect belongs to a
  construction used to *verify* the solver, not to configurations it exists to
  model. The existing transparency tests use a 0.005 m layer over 0.5-8 kHz,
  comfortably inside the safe window, which is why this went unnoticed rather
  than being a regression. `stoneley_dispersion_layered` now documents both
  facts.

### Fixed
- **`pseudo_rayleigh_dispersion` no longer returns a different mode depending on
  the caller's frequency window.** The seed is now *enumerated* rather than
  guessed: `_enumerate_leaky_roots_n0` sweeps the leaky-S window at the highest
  requested frequency, keeps the points where the determinant genuinely dips
  against its own neighbourhood, and orders them by descending `Re(k_z)` —
  ascending radial order, so index 0 is the fundamental. A new `branch: int = 0`
  argument selects the order.
  This closes all three defects reported in the previous release cycle, with the
  same measurements now running the other way round. `branch=0` on a 0.10 m hole
  in a 4000/2300/2500 formation returns **c = 2486.16 m/s at 30 kHz for every
  grid top from 32 kHz to 100 kHz** (it used to switch silently to 2952 m/s
  somewhere between 55 and 60 kHz); a 0.07 m hole over 4-30 kHz now returns
  **60/60 finite samples on the 60-point grid that used to return nothing**; and
  a 24-32 kHz request returns **81/81** and matches the wide-grid values to 1e-9
  instead of returning nothing. `branch=1` reaches the other root — addressable
  now rather than arrived at by accident.
  Requesting a branch that has not yet passed its cutoff raises with the number
  of branches actually found, rather than returning an empty curve that would be
  indistinguishable from "this mode does not propagate here".
  The enumeration costs about 0.25 s per call, paid once per call regardless of
  grid size. The one place that adds up is `PSEUDO_RAYLEIGH_MODE` in
  `scripts/gen_surrogate_dataset.py`, which calls the solver once per draw — an
  offline generator, so the margin was kept rather than traded for speed.
  Its root *count* is unchanged
  from a 24x5 seed grid up to 80x16 across radii 0.07-0.15 m, `V_S` 1700-2800 m/s
  and 15-60 kHz, so the scan resolution does not decide how many modes exist —
  pinned by a test, since otherwise `branch=1` would mean different things at
  different densities.
  The five tests that pinned the old behaviour as defects are rewritten as
  guarantees, keeping their measured numbers so the direction of the change is
  visible in the diff.

### Changed
- **The leaky-mode energy-balance oracle was attempted and withdrawn; the
  negative result is pinned by tests.** `plans/learning.md` listed it as the
  most promising remaining candidate, on the reasoning that radiated power over
  axial power reproduces `Im(k_z)` with no free geometry in it and might
  therefore *explain* the ~0.6 offset `leaky_radiation_attenuation` leaves open.
  The derivation works and the agreement is exact — ratio 1.000 at every
  frequency, which briefly looked like the strongest confirmation in the
  repository. It is an identity. Fed eight arbitrary complex `k_z` values that
  are not roots of anything, it returns their imaginary parts too, to ratio
  1.0000: closing the balance inside the fluid is the divergence theorem applied
  to a source-free Helmholtz solution, and no property of the formation enters.
  Extending the balance into the formation does not rescue it either — the
  leaky-S field *grows* with radius (0.996 at `r` = 0.1 m to 1.6e86 at 30 m,
  using the solver's own radial evaluator), so the axial power integral has no
  finite value to divide by. Three tests pin all of it: that the balance
  reproduces `Im(k_z)` at roots, that it does so at non-roots too, and that the
  formation field grows. **Nothing was added to the public API** — a check that
  cannot fail is worse than no check.

### Added
- **`tube_wave_speed` — the low-frequency oracle, completing the pair with
  `scholte_speed`.** The White (1983) closed form
  `S_T^2 = 1/V_f^2 + rho_f/mu` is the `f -> 0` limit of the borehole Stoneley
  mode, as `scholte_speed` is the `f -> infinity` limit. Both ends of the
  dispersion curve are now pinned to closed forms. Verified to 1.3e-8-1.5e-7
  relative across five media including a doubled fluid density, and the
  radius-independence the formula predicts (no `a` appears in it) holds across
  `a` = 0.05-0.30 m to 5e-8.
  **The independence is qualified, and the qualification is the interesting
  part.** Unlike Scholte, this formula is already inside the solver:
  `_stoneley_kz_bracket` uses it to place the upper end of its search bracket.
  A test that went through `stoneley_dispersion` would have been partly the
  solver confirming itself. The tests therefore locate the root by scanning
  **40x wider than the solver's factor-of-two bracket**, taking the estimate
  out of the loop; the docstring says plainly that this is a weaker tie than
  the Scholte one rather than presenting it as fully independent.
- **A validity floor on the slow-formation `V_S` estimator, previously
  undocumented.** A tube wave is a bound mode, so it must be slower than the
  formation shear wave; requiring that gives
  `V_S > V_f*sqrt(1 - rho_f/rho)`, equivalently
  `S_ST < (1/V_f)*sqrt(rho/(rho - rho_f))` on the measured slowness. Below it
  **no bound Stoneley root exists at all** — confirmed by scanning the modal
  determinant across a window far wider than the solver's bracket and finding
  no sign change, rather than by observing that `stoneley_dispersion` returns
  NaN. The closed form predicts where the solver stops converging to within
  1 % across seven (rho, rho_f, V_f) combinations spanning floors from 960 to
  1255 m/s.
  This bites in practice: for brine in a 2200 kg/m^3 formation the floor is
  1108 m/s, an ordinary slow formation and squarely inside the range
  `vs_from_stoneley_slow_formation` exists to serve. `tube_wave_speed` raises
  below the floor; the estimator documents it but deliberately does **not**
  enforce it, because a noisy field pick should be screened in QC rather than
  hard-failing a whole log — a choice now stated at the point of use.
- **`plans/learning.md`** — a retrospective on the five analytic oracles added
  between PRs #50 and #63, written to change how the next batch of work is
  chosen rather than to record status. Covers what distinguishes an oracle from
  another test, the five specific ways a check looked convincing and was not
  (a limit that cannot discriminate, a grid that shares the scaling under test,
  statistics over the wrong population, sampling structure mistaken for noise, a
  false test premise), why a systematic offset must be reported rather than
  fitted away, and the planning consequences — chiefly that claims about
  *absence* are the ones this repository's plans keep getting wrong.

### Added
- **`leaky_radiation_attenuation` — an independent oracle for the leaky-mode
  solver's attenuation.** A borehole leaky mode is a fluid wave bouncing from
  wall to wall through the axis, losing energy at each reflection to the shear
  wave it radiates into the formation. That picture gives
  `Im(k_z) = -ln|R| k_f / (2 a k_z)` from nothing but the textbook plane-wave
  fluid/solid reflection coefficient — no Bessel functions, no modal
  determinant — so comparing it with `pseudo_rayleigh_modal_dispersion` checks
  the solver against different physics rather than against itself. This is the
  same move `scholte_speed` made for the bound Stoneley mode.

### Changed
- **Leaky-mode attenuation checked; the estimate holds at the order-of-magnitude
  level and the check turned up three branch-selection defects.** Over 4-30 kHz,
  borehole radii 0.07-0.15 m and fast formations with `V_S` 1700-2800 m/s, the
  solver-to-estimate ratio stays inside 0.37-1.91. The scatter is not random:
  the median ratio is **0.57-0.71 in every one of those cases**, a stable
  systematic offset near 0.6, with the residual an oscillation whose peak
  spacing satisfies `spacing * a = const` to about 6 % — the same `2a`
  transverse round trip the estimate assumes, recovered independently from the
  solver's own output. So the geometry is confirmed and the scale is right to
  within a factor of two. The offset is **reported, not corrected**: folding an
  empirical constant into the formula would turn an oracle into a fit.
- **`pseudo_rayleigh_dispersion`'s result depends on the caller's frequency
  window, which was undocumented.** The march is seeded at the highest requested
  frequency, and more than one leaky root lives near that seed. Measured on a
  0.10 m hole in a 4000/2300/2500 formation: the mode reported at 30 kHz has
  `c = 2486 m/s` for a grid stopping at 40 kHz and `c = 2952 m/s` for one
  stopping at 80 kHz — a 19 % difference, and **both are genuine roots of the
  determinant**, verified by residual against the neighbourhood. Two related
  failure modes are worse because they are silent: a 0.07 m hole recovers the
  4-30 kHz band at 80 samples but returns **all-NaN at 60**, and requesting only
  24-32 kHz returns nothing where a 2-40 kHz grid converges throughout. Where it
  does converge the tracking is sound — halving the step reproduces the
  attenuation to 1e-10 relative — so this is about *which* root is followed, not
  accuracy. The docstring now states all of this and advises treating an
  all-NaN result as "not found on this grid" rather than "no such mode".
  Five tests pin the behaviour so a future fix surfaces here rather than
  passing quietly.
- **Quadrupole high-frequency asymptote checked; it validates the slow-formation
  solver and exposes a fast-formation defect.** At short wavelength the borehole
  wall looks flat to every azimuthal order, so the n=2 quadrupole must approach
  the same plane-interface Scholte speed the n=0 Stoneley does. In slow and
  intermediate formations it does: monotone convergence to better than 0.1 % at
  400 kHz, and n=0 and n=2 agree with each other to 1e-4 there.
  In **fast** formations it does not. `quadrupole_dispersion` returns a
  *non-monotone* scatter between the Rayleigh and shear speeds — finite values,
  which is the hazard, because a caller filtering on `NaN` keeps them. This is
  the same leaky-mode limitation as the n=1 flexural case (roadmap A.2), so that
  item now covers two solvers rather than one.
  It also corrects a comment in `scripts/gen_surrogate_dataset.py`, which said
  fast-formation quadrupole draws "often fall below `min_finite`" and would
  therefore be filtered out downstream. Measured over the default mixed prior
  they usually are not: **19 of 31 fast draws cleared `min_finite`, and 18 of
  those 19 were non-monotone** — they would be marked present in
  `mode_in_gather` and injected into the gather. `QUADRUPOLE_MODE` should be
  paired with a slow-formation prior, which the comment now says. Slow draws are
  unaffected (11 of 11 monotone in the same sample).
  Five tests, including one pinning the fast-formation misbehaviour so that a
  future fix to the leaky-mode search surfaces here rather than passing
  unnoticed.
- **The pseudo-Rayleigh rigid-pipe cutoff estimate is checked against the
  solver for the first time, and its documented use was wrong.**
  `_J1_FIRST_ZERO` and the closed form
  `f_c ~ j_{1,1} V_f V_S / (2 pi a sqrt(V_S^2 - V_f^2))` have been in
  `_leaky.py` since the leaky work landed, with a docstring offering them to
  "callers that want to guard against requesting frequencies below the cutoff".
  Nothing had compared them to what the solver does. Comparing them splits the
  claim in two.
  **What holds:** the geometric `1/a` scaling is reproduced exactly. Measured on
  a *fixed* frequency grid across a 3.3x range of borehole radius, the ratio of
  solver cutoff to closed form is constant to about 1 part in 300. (Tying the
  grid to the estimate would have produced a constant ratio for free, since both
  scale as `1/a` — the first version of this measurement did exactly that and
  had to be redone.) A test pins it, and would catch a radius/diameter
  confusion, which is invisible to any single-radius check.
  **What does not:** the estimate overshoots badly as an absolute cutoff. At
  `V_S = 2600`, `V_f = 1500`, `a = 0.10` it gives 11.2 kHz while the solver
  converges to about 4.1 kHz, so guarding with it discards a valid band nearly
  3 kHz wide — the opposite of the docstring's advice, which is corrected. The
  offset is not a constant that could be folded in: it varies strongly with
  formation velocity, and for some parameter sets the marcher's termination is
  not stable at all (a 1.1 % change in `vp` moved it 20 %; one case never
  converged on a reasonable grid). The docstring now says to treat the `NaN`
  boundary as this implementation's convergence limit rather than as a physical
  cutoff.

### Added
- **`fwap.scholte_speed`, and with it the first literature tie the validation
  notebook actually makes** (roadmap A.1). A.1's remaining half was blocked on
  digitising published figures, which needs the books. This is the part that was
  reachable without them.
  `scholte_speed` solves the classical secular equation for an interface wave on
  a **plane** fluid/solid boundary — a different equation from the cylindrical
  modal determinant, with no Bessel functions and no borehole radius in it. As
  the wavelength shortens the borehole wall looks flat, so `stoneley_dispersion`
  must approach it. It does: agreement better than **0.1 % at 400 kHz**,
  converging monotonically, from below in a fast formation and from above in a
  slow one. That is a cross-check between two independent calculations, not the
  solver confirming itself.
  The oracle is validated in turn by its own light-fluid limit, where it
  collapses to the Rayleigh equation and reproduces `rayleigh_speed` to 1e-9 —
  a third, separate implementation.
  Two properties of the equation are easy to get wrong and are documented
  because both were hit while deriving it: the sign of the fluid-loading term is
  **not** determined by the light-fluid limit (both signs reduce to Rayleigh, so
  that check cannot discriminate them — the sign is fixed instead by requiring a
  root to exist below `min(vs, vf)`), and the root is not generally near the
  Rayleigh speed, which can fall outside the admissible range entirely when
  `vf < vs`.
  Six tests, including one asserting the limit check *can fail*: scored against
  two plausible wrong references — the fluid velocity, and a rock with fluid
  density off by 20 % — both must be at least 5x further from the solver's
  answer than the correct value. Notebook section 6 runs the comparison and
  asserts it, and the notebook's summary no longer claims nothing in it is
  validated.

### Changed
- **Two candidate sources for a real full-waveform sonic gather identified;
  roadmap F's "none is known to exist" withdrawn.** Section F asserted that no
  openly redistributable full-waveform gather was known to exist. A search says
  otherwise, so the claim is retracted and replaced with a shortlist:
  **Utah FORGE** via the DOE Geothermal Data Repository (Schlumberger dipole
  sonic in DLIS, which `fwap.io.read_dlis` already reads, from an
  eight-receiver array with monopole and two dipole sources, licensed
  **CC BY 4.0**), and **IODP/ODP** via the LDEO Borehole Research Group (sonic
  waveforms for many holes, DLIS plus a Python-friendly binary export,
  documented as eight waveforms x 512 samples at 10/40 us every 15.24 cm —
  close to this package's own defaults).
  Neither file has been downloaded or opened, and the entry is explicit that it
  is a shortlist from published metadata rather than a verified result.
  Fetching was then attempted, which narrowed the handoff and corrected one
  claim: the shortlist originally said Utah FORGE is "also mirrored on AWS Open
  Data", implying an S3 route to the logs. That is wrong and is removed. The
  AWS buckets `gdr-data-lake` and `oedi-data-lake` *are* reachable from this
  sandbox and object downloads work, but they carry only bulk monitoring data
  (DAS, geophone, CASSM, magnetotellurics) — no DLIS, no LAS, nothing wireline.
  The hosts that do serve the log submissions (`gdr.openei.org`,
  `data.openei.org`, `brg.ldeo.columbia.edu`, `osti.gov`, `iodp.tamu.edu`) all
  refuse to connect. The obstacle is which host serves the file, not the data,
  so a session with ordinary web egress could fetch it directly. No registry entry is added,
  because the SHA-256 cannot be computed without the file and an unverified
  checksum would defeat the registry's purpose. The remaining steps are
  recorded in order, and only the first — opening a file to confirm it holds
  per-receiver waveforms rather than processed curves — is real work.

### Added
- **Two-mode cased dataset** (`generate_slow_two_mode_cased_dataset`,
  `CASED_TWO_MODES`, `CASED_FLEXURAL_MODE`, `SLOW_TWO_MODE_PRIORS` in
  `scripts/gen_surrogate_dataset.py`): a cased-hole dataset carrying **both**
  the Stoneley and the flexural mode, fully bound across the band, where the
  default cased dataset has been single-mode.
  The interesting part is the prior it needs. The two cased modes fail in
  opposite directions — flexural is sparse in fast formations (leakage, see
  roadmap A.2), and the Stoneley stops being bound as the formation slows away
  from the fluid velocity — so the window where both hold is `V_S` in
  1420-1495 m/s, about **80 m/s wide**. Measured both-modes-bound fraction
  across the annulus prior: 0.00 at 1350 m/s, 0.42 at 1380, 0.92 at 1400, and
  1.00 from 1420 up.
  That window is **disjoint from the default cased prior** (1700-3000 m/s), so
  this is a *different* dataset rather than a subset of the usual one, and the
  two must not be pooled — a property asserted by a test rather than left to a
  comment. It suits cement-bond work, where the label is the bond index and
  formation `V_S` is a nuisance parameter (cement stiffness moves the cased
  Stoneley ~7 %, formation `V_S` ~1.5 %); it is the wrong dataset for anything
  needing formation-property variety, and says so at the point of use.
  No schema bump: mode count is read from `mode_names`, so a two-mode cased
  file is schema v4 like any other.

### Changed
- **Leaky-mode root tracking for `n=1` attempted; recorded as blocked on a
  derivation rather than on code.** The complex-plane machinery already exists
  and is proven for `n=0` (`_track_complex_root`,
  `_march_complex_dispersion`, `pseudo_rayleigh_dispersion`), so extending it to
  the `n=1` cased determinant looks like wiring. It is not. Three approaches all
  fail, and each is now written down so the next attempt starts further along:
  continuation from high frequency reproduces the real-axis branch to
  floating-point noise (`Im(k_z) ~ 1e-16`) and then stops exactly at the cutoff;
  fresh leaky-S seeding below the cutoff converges only sporadically and to
  incoherent values (phase velocity 2681/2918/2789 m/s at 6/4/3 kHz, attenuation
  ~0.6 Np/m — artefacts of the Hankel formulation, not a branch); and strict
  fine-step continuation from the cutoff has its nudged seed fall back onto the
  real axis and then fails on the first step below.
  A fourth observation constrains any future attempt: even *above* the cutoff,
  continuation across 1 kHz steps can hop to a root below the formation Rayleigh
  speed, so the extension needs the validated marcher's regime checks rather
  than the bare tracker. A new test pins the composition that does hold —
  seeded at each frequency, the complex tracker and the `n=1` cased determinant
  reproduce the real-axis solver to 1e-6 — which is the prerequisite any leaky
  work would build on.
  No leaky `n=1` API is shipped, deliberately. The roots found below the cutoff
  are not physical, and publishing them would produce numbers that look like
  dispersion data and are not. What is missing is which Riemann sheet the `n=1`
  pole occupies below the cutoff — and possibly there is no pole at all, the
  mode existing only above its cutoff with the low-frequency dipole energy
  travelling as a shear head wave. Schmitt 1988 fig 4 would settle it, which
  puts this behind the same literature access A.1 needs.
- **Roadmap A.2 re-diagnosed: the cased flexural sparsity is not a cased-hole
  problem.** The item was filed as layered-solver bracketing — "root-finding
  stays sparse for a typical casing + cement stack" — and measuring it says
  otherwise. A fast formation behind casing does converge over only ~38 % of a
  1-12 kHz band, all of it above ~5 kHz, but **stripping the casing and cement
  away leaves the identical formation just as sparse in an open hole, over the
  same lower part of the band**. No amount of work on layered bracketing would
  have fixed it.
  The cause is leakage: for `V_S > V_f` the flexural root leaves the real `k_z`
  axis, and the real-axis `Im(det)` sign change the solver hunts for survives
  only beside the shear branch point at high frequency. Widening the real
  bracket cannot recover it — fine scans find no sign change below the cutoff in
  any of the three sub-windows, and the middle one is singular for the
  propagator-matrix formulation anyway. A fix needs complex-plane root tracking,
  which is the machinery the free-pipe/leaky item (G.2) also needs, so the two
  are now planned as one piece of work.
  Measured (50 draws): fast formations average **28 %** band coverage (5/47
  fully converged), slow formations converge fully. This entry originally added
  "only ~15 % of draws are slow"; that was measured over the *default*
  `FormationPriors` (1200-3200 m/s) rather than the one the cased generator
  uses (1700-3000 m/s, i.e. 100 % fast), so it described the wrong distribution
  and is withdrawn — see the two-mode cased entry below for what replaced the
  conclusion.
  Four tests pin the comparison (slow converges fully; fast is sparse and
  high-frequency-only; the open hole is no better off; the branch that is found
  is formation-controlled and bounded by `V_S`), so the attribution cannot drift
  back. `flexural_dispersion_layered` now documents the limitation where a user
  meets it.

### Added
- **`fwap.validation`: scoring dispersion curves against digitised reference
  figures** (roadmap A.1, the machinery half). The validation notebook had five
  sections that plotted an fwap curve and described a 5 % RMS gate in prose;
  the gate did not exist, and the `OVERLAY_AVAILABLE` flag its documentation
  referred to was never implemented. Now `load_reference_curve` /
  `score_against_reference` / `format_overlay_score` do the comparison, and each
  notebook section calls `check_overlay(...)`, which prints an RMS verdict and
  **asserts** the budget. Verified in both directions: with a reference the
  notebook passes, and with that reference perturbed 12 % it fails with
  `FAIL ... RMS 10.71% (budget 5.0%)`.
  Most of the module is input validation, because the reference data is
  hand-traced out of printed figures and every likely mistake produces a
  plausible-looking file. A slowness column in µs/ft or µs/m, a *velocity* axis
  digitised in place of a slowness one, a frequency axis left in kHz, click-order
  rows, duplicate frequencies — each is detected and refused with a message
  naming the suspected error. Units are never converted on the caller's behalf:
  a reference silently rescaled to fit would agree with a wrong solver as
  readily as with a right one. `OverlayScore` also reports the worst single
  point and the fraction of the figure actually compared, since an RMS over
  three points of a forty-point curve is not the check it appears to be.
  **No reference CSV is shipped**, so nothing in the notebook is yet validated
  against literature — and its closing cell now says exactly that rather than
  letting a page of green plots imply otherwise. Adding a figure is a drop-in:
  put the CSV in `docs/notebooks/_data/` under the documented name, with no
  code to edit. 23 tests.
- **Bed-boundary-aware coupling for joint inversion**: `invert_joint` gains
  `penalty="tv"`, a pseudo-Huber roughness cost, alongside the existing
  squared-difference `"l2"`. The motivation is a defect the previous release
  measured but did not fix: at the weight cross-validation picks, `"l2"`
  coupling improves a 2 us/ft log overall (`vp` MAE 506 -> 420) while making the
  **bed contacts worse than not coupling at all** (486 -> 500). Squaring a
  transition makes it four times cheaper to spread a given change over four
  frames than to deliver it as one contact, so a real boundary is the most
  expensive feature in the log and the optimiser pays to erase it. `"tv"` is
  linear beyond `tv_eps` and so is nearly *indifferent* to how change is
  distributed -- it does not prefer jumps, it stops paying to remove them --
  which improves both numbers at once: 388 overall and 406 at the contacts.
  `contact_precision` and `no_skill_contact_precision` make the boundary claim
  checkable rather than rhetorical: can you still find the beds in the recovered
  log? `"tv"` finds 91% against `"l2"`'s 83%, over a 36% no-skill bar that is
  reported alongside because on a log with many contacts blind guessing already
  scores well.
  Because a piecewise-constant test bed is the most favourable possible setting
  for a contact-preserving penalty, `synthesize_profile` gains
  `gradation_frames` to build the unfavourable one. With contacts ramped over
  four frames the advantage narrows and partly **inverts**: `"tv"` stays ahead
  on overall error but is worse at the transition frames (262 against 241) and
  worse at locating them (0.72 against 0.84). The default therefore stays
  `"l2"`, and the recommendation is about the rock rather than the algorithm --
  bedded log, use `"tv"`; gradational log, do not.
  Also fixes a latent numerical bug found while testing the new penalty: the
  pseudo-Huber was written as `sqrt(d^2 + eps^2) - eps`, which loses most of its
  significant digits to cancellation in float32 once `eps` exceeds the
  transition size. It is now evaluated as the algebraically identical
  `d^2 / (sqrt(d^2 + eps^2) + eps)`, which is stable for every ratio.
- **Joint multi-depth inversion** (`sonic_ml.models.joint`), the follow-on
  named when surrogate-in-the-loop inversion closed: `invert_joint` solves a
  whole logged interval as one problem, penalising frame-to-frame change in the
  standardised parameters, and `synthesize_profile` builds the bedded synthetic
  log to test it on (drawn from the *dataset's own* observed ranges, so the
  profile is inside the surrogate's training support by construction rather
  than by hope).
  The result is conditional, and the condition is the interesting part.
  **Noise-free, depth coupling buys nothing** -- the best available penalty is
  no penalty on almost every profile and parameter. The first draft of the
  module claimed the opposite mechanism (that coupling averages away the
  surrogate's forward error); the noise-free run falsified it, and the reason is
  that inside a bed every frame has *identical* parameters, so the surrogate
  makes the *identical* error, perfectly correlated down the bed with nothing to
  cancel. The mechanism is ordinary noise averaging, so with 2 us/ft of
  dispersion-picking scatter the same sweep removes **31-45%** of the error on
  all four parameters.
  Because the prior is one a moving average could also apply, the boring control
  ships with it: `smooth_independent` post-smooths an independently inverted
  log, and both arms are tuned by the same rule. At their oracle settings
  coupling wins on all four but unevenly -- 38% against **0%** on `vs`, and
  44.6% against 42.8% on `rho`, a tie. At the setting cross-validation actually
  picks, coupling keeps 17-29% while smoothing keeps **nothing**: CV chose a
  one-frame window on all five profiles, because post-hoc averaging degrades
  held-out data misfit monotonically and so cannot be tuned from data at all.
  That asymmetry is the argument for coupling inside the objective rather than
  after it, and it also shows at bed contacts (421 m/s against 535 on `vp`),
  where a well-determined frame can resist its neighbours and a moving average
  cannot.
  `select_lambda` picks the penalty weight by cross-validation on held-out
  frequencies, using no truth, and its failure is reported rather than omitted:
  on a noise-free log it **over-couples** -- non-zero penalty on four of five
  profiles when zero is correct -- leaving the result **18-29% worse than not
  coupling**. Withholding 30% of the frequencies makes each frame less
  determined during selection than at inference, so the prior looks more
  valuable than it is. Noted alongside: on a clean log the untunable control is
  the safer method, since being impossible to talk into a bad setting saves it
  exactly that 18-29%.
  `bed_vs_boundary_mae` scores inside beds and across contacts separately so the
  bias-for-variance trade cannot hide inside one average. 28 tests,
  mechanism-focused.
- **Core open-hole tutorial notebooks + documentation pass**: the library's own
  workflows had no tutorial -- only a solver-validation notebook and two ML
  ones -- so this adds them.
  ``docs/notebooks/open_hole_processing.ipynb`` runs Parts 1-2 end to end
  (synthesize a monopole gather -> STC coherence surface -> pick P/S/Stoneley
  against planted truth -> f-k and tau-p wave separation -> track across depth
  -> LAS-ready log curves), and
  ``docs/notebooks/open_hole_petrophysics.ipynb`` runs Part 3 and the extension
  layer (flexural dispersion bias, elastic moduli, Gassmann substitution,
  Stoneley permeability, and a drilling-decision mud-weight window). Both use
  **only** ``fwap`` -- no torch -- so they are validated by the **core**
  ``ci.yml`` gate via a new ``--nbval-lax`` step (12 cells, ~15 s), which also
  proves the tutorials work without the ML layer installed.
  Two teaching points came out of writing them and are reported rather than
  smoothed over: an eight-receiver array gives the f-k filter almost no
  wavenumber resolution, so tau-p separates the Stoneley ~2.8x better while f-k
  barely moves the ratio; and the Lacy UCS correlation is quadratic in Young's
  modulus and was fitted on *static* core moduli, so feeding it a *dynamic*
  sonic modulus returns 360 MPa where a 2x static correction gives a plausible
  109 MPa -- the notebook prints both.
  ``README.md`` gains a ``sonic_ml`` section (layer table, the two headline
  results with the identifiability gap between them intact, install steps) and a
  tutorial-notebook index; ``docs/`` gains a narrative ``sonic_ml.rst`` page
  covering the isolation guarantees, the versioned ``.npz`` contract, and the
  honest-measurement helpers, with the notebooks split into their own toctree.

### Fixed
- **Stale cased-flexural comment in the surrogate generator**: the
  ``CASED_STONELEY_MODE`` note claimed fwap's layered n=1 solver "covers the
  slow-formation bound regime only", which stopped being true when
  fast-formation cased-hole flexural landed. The solver no longer refuses a
  fast formation; its root-finding is simply still sparse there (a few
  frequencies converge for a typical casing + cement stack), which is why the
  cased dataset stays single-mode. Comment corrected to say that.
- **`sonic_ml` re-gridding evaluation + casing-ring augmentation (M5f)**: two
  robustness slices, both of which produced *negative* results worth shipping.
  ``sonic_ml.models.regrid`` measures the operator's off-grid claim properly:
  ``true_curves_on_grid`` re-runs fwap's layered solver at arbitrary frequencies
  (exact by construction -- on the dataset's own grid it reproduces the stored
  labels bit for bit, which a test asserts), and ``evaluate_regridding`` scores
  the operator's zero-shot prediction on an unseen grid against **the
  interpolation control**: the same operator's training-grid prediction simply
  interpolated onto that grid. **Finding: on these smooth dispersion curves,
  zero-shot re-gridding does not beat interpolation** (measured 0.352 vs 0.254
  us/ft on a 24 -> 93 point refinement, against 0.256 us/ft on the training grid).
  M5c's "self-consistent to 0.14%" is true but measures agreement, not accuracy;
  a model can be smoothly and confidently wrong at every new frequency and still
  agree with itself. ``format_regrid_score`` therefore always prints the control
  and an explicit verdict, so the super-resolution claim cannot be reported
  without the number that can refute it.
  ``CasingRingAugmentation`` plants a phenomenological steel-casing ring arrival
  (the same modelling stance as ``fwap.lwd.lwd_collar_mode``) on cased gathers.
  Its amplitude is drawn **independently of the bond index by design**: a
  bond-coupled ring would manufacture a CBL-like signal, and a model "recovering"
  bond from it would only be recovering a hard-coded relationship. **Finding: at
  realistic ring amplitudes (0.2-1.5x RMS) the M5d bond inverse is already
  robust** (1.0x degradation, so there is nothing to fix); only when the ring
  dominates the record (20-50x RMS) does the plain net degrade 2.4x, and training
  with the augmentation removes that degradation entirely (0.9x) for a small
  clean-accuracy cost. A new ``Augmentation`` Protocol lets it drop into the
  existing ``augment=`` slot with no dataset changes (``GatherAugmentation``
  satisfies it unchanged). This is a robustness probe, **not** free-pipe
  detection -- that needs a leaky-mode forward model, not a planted wavetrain.
  Torch-gated; spine stays torch-free.
- **Cased-hole tutorial notebook (M5e)**:
  ``docs/notebooks/cased_hole_tutorial.ipynb`` walks the cased-hole path end to
  end -- generate a schema-v4 cased dataset, train the M5c forward operator and
  evaluate it on a 3x finer frequency grid it never saw, train the M5d bond
  inverse, and score it against the classical Stoneley baseline and a
  predict-the-mean reference through the bond harness. Its spine is a **forward
  sensitivity sweep run before any training**: sweeping each parameter across its
  prior shows cement stiffness dominates the cased Stoneley curve while formation
  Vs moves it less than the nuisance cement-thickness variation, which *predicts*
  the inverse result that follows (bond recoverable, behind-casing Vs weak) and
  frames the closing section on calibrated uncertainty. Runs in ~75 s, added to
  the Sphinx toctree, and validated on every PR by the ``ml.yml`` ``--nbval-lax``
  step (now covering both tutorials). Closes with an explicit statement that this
  is **not a free-pipe detector** and that the ~2x skill here, unlike the
  open-hole tutorial's ~25x, reflects a partially-identifiable problem rather
  than modelling effort.
  Writing it exposed a gap in the M5d API, fixed here: the M3
  ``residual_zscore_std`` resolves truth through ``bundle.param()`` and so raises
  on ``bond_index``; ``sonic_ml.models.cased_inverse.cased_residual_zscore_std``
  is the cased-aware counterpart (mirroring ``cased_target_mae``), with a test
  pinning that it agrees with the M3 helper on a formation column and reaches the
  bond target where the M3 helper cannot.
- **`sonic_ml` cement-bond inverse (M5d)**: the cased-hole counterpart of the M3
  inverse -- a cased waveform gather is inverted for the two quantities a cement
  evaluation wants, ``(behind-casing Vs, bond index)``, reusing the M3
  ``InverseNet`` (1-D CNN + heteroscedastic mean/log-variance head) unchanged and
  only swapping the targets. Adds ``sonic_ml.models.cased_inverse``
  (``cased_targets``, ``CasedInverseDataset`` -- gather-only input, so the
  dispersion label cannot leak -- ``train_cased_inverse``, ``NeuralBondPredictor``,
  ``cased_target_mae``), a bond scoring harness ``sonic_ml.bench.bond``
  (``BondPredictor`` protocol, ``bond_regime_labels`` stratifying by *bond
  quality* rather than the Vs harness's slow/fast regime, ``evaluate_bond``
  reusing the existing ``Scorecard`` and bootstrap machinery, ``MeanBondPredictor``
  as the no-skill reference), and a classical reference
  ``sonic_ml.baselines.bond.StoneleyBondBaseline``. ``format_scorecard`` gains
  backward-compatible ``row_order`` / ``precision`` arguments so a ``[0, 1]``
  target renders legibly.
  **The baseline is deliberately not a CBL amplitude indicator**: these
  synthetics are built from the cased Stoneley dispersion alone and contain no
  casing-ring arrival, so an amplitude gate would be measuring noise and beating
  it would prove nothing. The honest analogue uses the signal that is present --
  an STC Stoneley pick mapped to bond through a calibration fitted on the
  training split.
  **Measured (300-sample dataset, held-out split):** bond-index MAE 0.134 for the
  net vs 0.220 for the classical baseline and 0.247 for predicting the mean
  (~1.8x skill, ~1.6x over classical); behind-casing Vs MAE 224 m/s (~1.5x over
  the mean). Both targets' predicted sigma is well calibrated (residual z-score
  std 1.10 and 1.12). Scaling to 800 samples lifts both to ~2.1x skill with
  calibration holding (z-std 1.1-1.2), so this is a partially-identifiable
  problem rather than an under-trained one -- deliberately unlike the M3
  open-hole result (~25x), and reported as such. That asymmetry is expected, not a defect: a forward
  sensitivity sweep shows cement stiffness moves the cased Stoneley curve ~7%
  across its prior while formation Vs moves it only ~1.5% -- less than the
  nuisance cement-thickness variation -- so the uncertainty head reporting a wide
  sigma on Vs is the model being honest about a weakly identifiable target.
  **Scope:** the M5a dataset spans the *bonded* regime only, so this inverts
  graded bond quality and is **not** a free-pipe detector. Torch-gated; spine
  stays torch-free.
- **`sonic_ml` cased-hole forward operator (M5c)**: wires the M5b operator
  primitives to the M5a cased-hole dataset, learning
  ``(formation, casing, cement, bond index) -> Stoneley slowness curve s(f)`` --
  a surrogate for the layered modal-determinant root-finding in
  ``fwap.stoneley_dispersion_layered``. Adds ``sonic_ml.models.cased``
  (``cased_features`` assembles the 15-column feature matrix and
  ``Standardizer`` drops the constant fluid/casing-density columns;
  ``CasedForwardOperator`` selects either the ``"fno"`` or ``"deeponet"``
  backbone behind one ``(B, M, F)`` signature; ``TrainedCasedOperator`` predicts
  in raw units and round-trips through a ``weights_only``-safe checkpoint) and
  ``sonic_ml.models.cased_train`` (``CasedDataset``, ``train_cased_operator``
  with a masked loss, ``slowness_mae_us_per_ft``, ``resolution_transfer_error``).
  **The operator payoff:** the frequency axis is a *coordinate*, not an array
  index, so a trained model can be evaluated on a grid it never saw via
  ``predict(..., freq=...)`` -- normalized against the stored training band so a
  sub-band query is not silently rescaled. On a 200-sample dataset both
  backbones reach ~0.15-0.18 us/ft test MAE against a 213-240 us/ft curve range,
  and re-gridding is self-consistent to 0.14% (FNO) and to floating-point
  precision (DeepONet, whose pointwise trunk cannot let a node depend on the
  rest of the grid -- the FNO couples the grid globally through the FFT). Tests
  score skill against a predict-the-training-mean baseline rather than asserting
  an absolute MAE. Same honesty framing as the M2 open-hole surrogate: a
  methodology and validation baseline, not a speed claim. Torch-gated; spine
  stays torch-free.
- **`sonic_ml` operator-learning primitives: FNO + DeepONet (M5b)**: a new
  ``sonic_ml.models.operator`` module adding the two building blocks the
  cased-hole operator surrogate needs, both **implemented in-house on plain
  PyTorch** (no ``neuraloperator`` or other new dependency). ``SpectralConv1d``
  multiplies the lowest retained Fourier modes by learned complex weights
  (stored as a real ``(..., 2)`` tensor so ordinary optimizers and
  ``weights_only=True`` checkpoints keep working); ``FNO1d`` stacks
  ``GELU(spectral + pointwise)`` blocks; ``DeepONet`` factors the map into a
  branch net over parameters and a trunk net over a *query coordinate*; and
  ``params_on_grid`` lifts a parameter vector onto a coordinate grid with the
  standard trailing coordinate channel. Unlike the M2/M3 point networks, these
  learn a map between *functions*: the frequency axis is a coordinate rather
  than an array index, so a grid-trained ``FNO1d`` runs on any grid length
  (verified zero-shot on a 2x finer grid) and a ``DeepONet`` can be queried at
  arbitrary off-grid frequencies. Tests pin the exact Fourier semantics
  (DC-only convolution equals the grid mean; energy above the cutoff is
  rejected; the trunk is pointwise in the query) and assert only that learning
  *happens*, never a specific accuracy. Torch-gated; the spine stays torch-free.
- **Cased-hole surrogate dataset: schema v4 (M5a)**: the generator can now
  synthesize *cased-hole* samples (a steel casing + cement annulus between the
  borehole fluid and the formation) by wrapping fwap's existing layered modal
  solver ``fwap.stoneley_dispersion_layered``. ``scripts/gen_surrogate_dataset.py``
  gains a ``CasingCementPriors`` (draws the two annular ``BoreholeLayer``s plus a
  normalized cement **bond index** in ``[0, 1]``), a ``CASED_STONELEY_MODE`` /
  ``CASED_MODES``, and a ``generate_cased_dataset`` convenience that pins the
  cased configuration (fast-formation prior + stiff-cement bound regime, where
  the Stoneley tube wave -- the classic cement-bond-evaluation mode -- stays a
  clean bound curve). Schema bumps to **v4**: every dataset now stores
  ``layer_params`` ``(N, L, 4)`` = per-layer ``[vp, vs, rho, thickness]``,
  ``layer_names`` ``(L,)``, and ``bond_index`` ``(N,)``; an *open-hole* dataset
  carries an empty ``(N, 0, 4)`` stack and ``NaN`` bond, so open-hole data is
  unchanged bit-for-bit apart from the new metadata keys (the frozen contract
  test is updated in lockstep). ``sonic_ml``'s loader reads the cased arrays into
  ``DatasetBundle.layer_params`` / ``layer_names`` / ``bond_index`` (with an
  ``is_cased`` helper; ``None`` for schema v1/v2/v3 files) and accepts schema
  v1--v4. Honest scope: the dataset covers the *bonded* Stoneley regime;
  free-pipe / leaky bond states and cased flexural (slow-formation-only in fwap)
  are deferred. This is the data foundation for the M5 cased-hole operator
  surrogate + cement-bond inverse.
- **`sonic_ml` low-latency LWD inverse variant (M4f)**: a compact,
  depthwise-separable configuration of the M3 inverse net for logging-while-
  drilling latency/power budgets. ``InverseNet`` gains a backward-compatible
  ``separable=True`` flag that swaps its dense conv blocks for
  depthwise-separable ones (``O(in*out*kernel)`` conv weights ->
  ``O(in*kernel + in*out)``); ``train_inverse`` and the checkpoint round-trip
  thread it through (older checkpoints without the key load as dense). A new
  ``sonic_ml.models.lwd`` packages the compact preset
  (``build_lwd_inverse_net`` / ``train_lwd_inverse`` -- two separable blocks, a
  narrow head; ~40x fewer parameters than the full default net) and a benchmark
  (``count_parameters``, ``measure_latency_ms``, ``latency_accuracy_report`` /
  ``format_latency_accuracy``) that *measures* the parameter / latency /
  accuracy trade-off rather than asserting it. The compact net is a plain
  ``InverseNet``, so it reuses ``TrainedInverseNet`` and the ``InversePredictor``
  harness adapter unchanged. Honest framing: the hardware-independent win is the
  parameter/FLOP reduction (power/latency on a downhole DSP); desktop-CPU
  wall-clock latency for models this small is runtime-dependent and can even
  favour the dense net, so the report shows both numbers. Collar-mode
  contamination stays the domain of the phenomenological ``fwap.lwd`` layer;
  pairing it with this net is future work. Torch-gated; spine stays torch-free.
- **`sonic_ml` tutorial notebook (M4e)**:
  ``docs/notebooks/sonic_ml_tutorial.ipynb`` walks the full ML loop end-to-end
  on a small synthetic dataset -- generate a surrogate dataset from the fwap
  modal solver, load it through the ``.npz`` contract, regime-stratify the
  split, score the classical STC baseline, train the DL-FWI inverse net, compare
  the two head-to-head on identical held-out samples (the inverse net beats
  classical by ~an order of magnitude on Vs, including the fast-formation
  regime), and save the trained model with a provenance model card (M4d). Runs
  in ~30 s. Added to the Sphinx toctree (rendered via ``myst-nb``, execution
  off) and validated on every PR by a new ``--nbval-lax`` step in the
  non-required ``ml.yml`` job (which has torch); the notebook closes with an
  explicit statement of what a same-forward-model synthetic benchmark does and
  does not demonstrate.
- **`sonic_ml` model cards + checkpoint hygiene (M4d)**: a torch-free
  ``ModelCard`` (in ``sonic_ml.models.weights``) binds a trained checkpoint to
  its provenance -- model type, hyper-parameters, held-out metrics, the fwap
  version + git SHA at training time (reused from ``sonic_ml.provenance``), and
  the ``content_hash`` of the training dataset, so a checkpoint can be tied back
  to the exact ``.npz`` (and thus the exact fwap solver output) it learned from.
  ``save_with_card`` writes a model's own ``.save()`` checkpoint plus a small
  ``<ckpt>.card.json`` sidecar beside it; ``card_for`` / ``read_card`` build and
  read the card. The card is duck-typed on the trained wrapper
  (``.model.hparams`` + ``.save``), so it stays torch-free and works for both
  ``TrainedInverseNet`` and ``TrainedForwardSurrogate``. A root ``.gitignore``
  rule now excludes ``*.pt`` / ``*.ckpt`` (large binaries) while keeping the
  committable JSON card -- the durable record of what each checkpoint is.
- **Surrogate schema v3: leaky-mode attenuation channel + pseudo-Rayleigh mode
  (M4c)**: the generator now stores each mode's spatial attenuation rate
  (1/m) in an ``attenuation`` array alongside ``slowness`` -- a free extra
  label the modal solver already produces for leaky modes -- and bumps
  ``SCHEMA_VERSION`` to ``3`` (``SurrogateSample`` gains an ``attenuation``
  field; the core contract test updated in lockstep). Bound modes (Stoneley,
  slow flexural) leave it ``NaN``; the new opt-in ``PSEUDO_RAYLEIGH_MODE`` (a
  leaky ``fwap.pseudo_rayleigh_modal_dispersion`` mode, fast formations only)
  populates it. ``sonic_ml``'s loader reads it into
  ``DatasetBundle.attenuation`` (``None`` for schema v1/v2 files) and accepts
  schema v1/v2/v3. Fully backward compatible.
- **Surrogate generator: optional n=2 quadrupole mode (M4b)**:
  ``scripts/gen_surrogate_dataset.py`` gains a ``QUADRUPOLE_MODE`` -- a
  signature-compatible ``ModeSpec`` wrapping ``fwap.quadrupole_dispersion``.
  It is kept **out** of ``DEFAULT_MODES`` (the default dataset stays two-mode
  and lean, so the schema is unchanged), but passing
  ``modes=(*DEFAULT_MODES, QUADRUPOLE_MODE)`` to ``generate_dataset`` yields a
  three-mode dataset. This exercises the mode-count-agnostic pipeline
  end-to-end: the loader reads ``M`` from ``mode_names`` and both the forward
  surrogate and the inverse net handle ``M = 3`` without change (the quadrupole
  is bound mainly in slow formations; in fast formations it is largely leaky
  and often absent from ``mode_in_gather``). No schema-version bump.
- **`sonic_ml` waveform augmentation for sim-to-real robustness (M4a)**:
  ``GatherAugmentation`` perturbs each *training* gather on the fly -- an SNR
  sweep (additive noise to a random signal-to-noise ratio) plus optional
  amplitude jitter -- wired into ``InverseDataset`` (stochastic per access,
  training split only) and ``train_inverse`` via an ``augment=`` argument. On a
  noise-shifted held-out set the augmented inverse net degrades far less than
  the un-augmented one (Vs MAE ~164 vs ~934 m/s) at no cost to clean accuracy.
  Carries an explicit caveat that this narrows the *synthetic* generalization
  gap and is not a real-world deployment claim. Non-augmented behaviour (and
  the ``InverseDataset.x`` view) is unchanged; torch-gated, spine stays
  torch-free.
- **`sonic_ml` DL-FWI inverse net (M3)** -- the headline: a 1-D CNN over the
  multi-receiver ``gather`` (receivers as channels) that regresses the varying
  formation parameters with a **heteroscedastic head** (mean + log-variance per
  parameter, Gaussian NLL), so weakly identifiable parameters get calibrated
  error bars rather than false precision. Adds ``InverseNet`` /
  ``TrainedInverseNet`` (weights-only-safe checkpoints), an ``InverseDataset``
  with per-gather amplitude normalization (gather-only input -- the dispersion
  label is never fed in), a reproducible ``train_inverse`` loop, and an
  ``InversePredictor`` that satisfies the M1 ``Predictor`` protocol so the
  inverse net is scored by the *same* benchmark harness as the classical
  baselines. On a held-out set the inverse net beats the classical STC baseline
  by ~an order of magnitude on Vs (with non-overlapping bootstrap CIs),
  including the fast-formation regime where classical processing fails. Tests
  ``importorskip`` torch and cover shapes, checkpoint round-trip,
  overfit-a-batch, reproducibility, harness integration, and an anti-leakage
  check that the prediction is invariant to the slowness label. Runs in the
  non-required ``ml.yml`` job; the pure-NumPy spine stays torch-free.
- **`sonic_ml` forward dispersion surrogate (M2)**: the first ML model in the
  layer -- a residual-MLP surrogate mapping the varying standardized formation
  parameters to per-mode slowness curves and per-mode presence logits (a fast,
  differentiable stand-in for the modal-determinant root-finding). Lives in a
  new torch-gated ``sonic_ml.models`` subpackage (not imported by the pure-NumPy
  spine, so ``import fwap`` / ``import sonic_ml`` still need no torch):
  ``ForwardSurrogate`` (model), ``ForwardDataset`` + ``SlownessNormalizer``
  (tensor adapter with a finite-mask that keeps ``NaN`` slowness out of the
  loss), a masked Huber slowness loss + presence BCE, ``train_forward`` with a
  reproducible loop (determinism engaged up front) plus ``slowness_rmse`` /
  ``presence_auc`` metrics, and a ``train`` CLI that checkpoints a
  ``TrainedForwardSurrogate`` (weights + normalizers, weights-only-safe
  round-trip). Covered by the non-required ``ml.yml`` job (torch present);
  the tests ``importorskip`` torch so a torch-free dev install still runs the
  spine suite.
- **Surrogate-dataset schema v2 -- self-describing geometry**:
  ``scripts/gen_surrogate_dataset.py`` now persists the acquisition geometry
  in the ``.npz`` -- three 0-d scalars ``dt`` / ``tr_offset`` / ``dr`` which,
  together with the ``gather``'s ``(n_rec, n_samples)``, reconstruct the
  :class:`fwap.ArrayGeometry` -- and bumps ``SCHEMA_VERSION`` to ``2``. The
  ``SurrogateSample`` gains a ``geom`` field; the core schema-contract test is
  updated in lockstep. On the consumer side ``sonic_ml`` reads the geometry
  into ``DatasetBundle.geometry`` (``None`` for legacy v1 files) and
  ``sonic_ml.geometry.default_geometry`` now returns the stored geometry,
  falling back to the default reconstruction only for v1. This removes the
  "assume default geometry" caveat the M1 harness carried and makes the
  waveform self-describing ahead of the inverse-net work. ``load_npz`` accepts
  both schema v1 and v2.
- **`sonic_ml` classical-baseline harness (M1)**: the no-ML benchmark layer
  that establishes the bar a later ML model must beat. Adds a model-agnostic
  scoring harness (`sonic_ml.bench`) built on a `Predictor` protocol -- any
  object mapping a dataset + geometry to a per-sample Vs estimate -- with an
  `evaluate` that reports median absolute Vs error split by slow/fast regime
  plus a bootstrap 95% CI, and a `format_scorecard` text report. Two classical
  `Predictor`s (`sonic_ml.baselines`) estimate Vs from the waveform with only
  fwap processing: `ClassicalSTCBaseline` (regime-split dispersion-corrected
  STC for slow / pseudo-Rayleigh STC for fast) and `FKDispersionBaseline` (f-k
  phase-slowness reduced to a low-frequency shear slowness); both return `NaN`
  on failure rather than raising. `sonic_ml.oracles` vendors the closed-form
  physics limits (White Stoneley low-frequency slowness; flexural `1/vs` and
  Rayleigh high-frequency asymptotes), validated against the fwap modal solver.
  `sonic_ml.geometry.default_geometry` reconstructs the generator's default
  acquisition geometry (the `.npz` does not yet store it -- a schema v2
  follow-up would make gathers self-describing). Pure NumPy/SciPy + fwap, no
  torch; covered by the non-required `ml.yml` job.
- **`sonic_ml` sibling package (M0 spine)**: a new in-repo, top-level
  `sonic_ml/` package (its own `pyproject.toml`, depends on `fwap` + PyTorch)
  that consumes the surrogate-dataset `.npz` and will host the ML surrogate /
  inverse models tracked in issue #22. It is intentionally **not** part of the
  `fwap` distribution -- the setuptools `include = ["fwap*"]` glob excludes the
  non-`fwap` name, so `mypy fwap`, `ruff`, the public-API guard, and the core
  wheel are all unaffected, and `import fwap` still needs no ML dependency.
  This first drop is the pre-model spine: a defensive `.npz` loader
  (`allow_pickle=False`, asserts `schema_version`, reads `N`/`M`/`F` from
  metadata), a bridge to the core generator, provenance capture (fwap version +
  git SHA + config + content hash, JSON sidecar), regime-stratified
  train/val/test splitting with stored indices, a standardizer with a
  zero-variance guard (drops constant `vf`/`rho_f`), masking helpers (finite
  mask + authoritative `mode_in_gather` presence + imbalance weights), and
  torch-guarded determinism helpers. A new non-required `.github/workflows/
  ml.yml` (installs the in-PR core + a CPU torch wheel + `sonic_ml`, then runs
  ruff/mypy/pytest scoped to `sonic_ml`) covers it without touching the core CI
  gate. Install editable with `pip install -e ./sonic_ml`; a `fwap[ml]` alias
  is deferred until `sonic_ml` is ever published to an index.
- **Surrogate-dataset schema guard**: ``scripts/gen_surrogate_dataset.py``
  now stamps its ``.npz`` output with a ``schema_version`` key
  (``SCHEMA_VERSION = 1``), and a new ``tests/test_npz_schema_contract.py``
  pins the on-disk contract -- the exact key set, ``PARAM_NAMES`` column
  order, per-array dtypes and shapes, and the version. The generator is a
  path-imported script outside the public-API guard and its ``.npz`` is the
  sole hand-off to any downstream ML consumer, so a breaking change (a
  reordered/renamed key or parameter column, a changed dtype) now fails core
  CI here rather than silently mislabelling training data. Pure NumPy; runs
  in the default ``.[dev]`` test job. Bump ``SCHEMA_VERSION`` and update the
  contract test in lockstep for any intentional layout change.
- **Surrogate-model data generator**: ``scripts/gen_surrogate_dataset.py``
  wraps the cylindrical-Biot forward modal solver
  (``stoneley_dispersion`` / ``flexural_dispersion``) as a labelled-pair
  factory for machine-learning surrogate and inverse models -- the
  borehole-acoustic analog of the seismic DL-FWI / neural-operator
  training loop. Each sample carries the per-mode phase-slowness curve
  (forward-surrogate label) and a synthetic multi-receiver gather from
  ``synthesize_gather`` (inverse-net input), with formation parameters
  drawn from a ``FormationPriors`` prior. NumPy/SciPy-only (no ML
  dependencies); a CLI writes a compressed ``.npz``. Model training
  stays out of the core package by design.
- **Continuous integration**: ``.github/workflows/ci.yml`` runs ruff
  (lint + format check), mypy, the public-API guard, and pytest on
  Python 3.11 and 3.12 for every push to ``main`` and every pull
  request. Concurrent runs on the same ref are cancelled.
- **Public-API guard**: ``scripts/check_public_api.py`` asserts that
  every name in a sealed frozen list is still exposed on the top-level
  ``fwap`` package. Step 0 of the planned module-splitting refactor;
  it catches accidental drops from ``fwap/__init__.py``'s re-export
  list when modules move into subpackages. Update the
  ``FROZEN_PUBLIC_API`` tuple in the same commit when a public name is
  intentionally added or removed.

### Fixed
- **CI mypy breaks from modern dependency stubs**: mypy parses installed
  third-party stubs against its configured ``python_version``, and
  recent typed dependencies ship stubs using newer-Python syntax that
  the old ``python_version = "3.9"`` target could not parse, aborting
  ``mypy fwap`` with ``[syntax]`` errors that were unrelated to fwap's
  own code:
    - matplotlib 3.11 (now shipping ``py.typed``) uses 3.10+ ``match``
      statements in ``matplotlib/_afm.py``;
    - numpy 2.5 uses a PEP 695 ``type`` statement (3.12+) in
      ``numpy/__init__.pyi``.
  Bumped the mypy ``python_version`` to ``"3.12"`` (the newest version
  the CI matrix runs; the 3.9 runtime floor is upstream-EOL and slated
  for removal) so modern stubs parse. matplotlib is additionally scoped
  with ``follow_imports = "skip"`` in a ``[[tool.mypy.overrides]]``
  entry, since fwap only uses it in the demos / plotting helpers and
  never type-checks against it. numpy stays fully followed (fwap relies
  on its typing). No fwap runtime or type-annotation change.
- **``CONTRIBUTING.md``** referenced the wrong clone URL and non-existent
  ``[io,segy]`` install extras; the documented ``pip install -e
  ".[dev,docs]"`` invocation now matches what ``pyproject.toml``
  actually defines. The "CI runs..." paragraph now matches the
  workflow that just landed.
- **Mypy backfill for the numpy-2.x stub migration**: the new CI
  surfaced 45 latent mypy errors that the local "host" mypy had been
  hiding (it ran in an isolated tool environment with no numpy
  installed, so imports silently degraded to ``Any``). Fixed without
  any runtime behaviour change:
    - 37 ``[var-annotated]`` sites in ``fwap/cylindrical_solver/``
      (``_cased.py``, ``_vti.py``, ``_n0_layered.py``,
      ``_n1_layered.py``), ``fwap/lwd.py``, ``fwap/picker/quality.py``,
      and ``fwap/demos.py`` get explicit ``np.ndarray`` annotations on
      ``np.zeros``/``np.empty``/``np.full`` initializations.
    - 6 ``[assignment]`` errors in ``fwap.geomechanics.vertical`` and
      ``fwap.geomechanics.indices`` come from signatures like
      ``mud_pressure: np.ndarray = 0.0`` that have always accepted
      scalars or arrays; the annotation widens to ``float |
      np.ndarray`` to match.
    - 2 ``[return-value]`` errors in
      ``fwap.cylindrical_solver._cased._formation_at_b`` come from a
      ``-> tuple[float, float, float]`` annotation on a function that
      indexes into a complex-dtype matrix. The matrix entries are
      real-valued by physics and the destination matrix ``M`` is
      real-dtype, so each value is wrapped in ``float(... .real)`` to
      drop the zero imaginary part explicitly. This silences the
      ``ComplexWarning`` that numpy previously emitted on each
      assignment into ``M``.

### Refactored
- **``fwap.io`` package**: the 762-LoC monolith splits cleanly by
  file format into a four-submodule package. Public surface is
  preserved via re-exports from ``fwap/io/__init__.py`` (and the
  ``from fwap import read_las`` aliases continue to work unchanged).
    - ``_common`` -- the ``_FWAP_UNITS`` mnemonic-to-unit map shared
      by the LAS and DLIS writers.
    - ``_las``    -- ``LasCurves``, ``read_las``, ``write_las``
      (``lasio``).
    - ``_dlis``   -- ``DlisCurves``, ``read_dlis``, ``write_dlis``
      plus the DLIS-only helpers ``_suppress_fd``,
      ``_DLIS_TO_LAS_WELL``, ``_LAS_TO_DLIS_WELL`` (``dlisio`` +
      ``dliswriter``).
    - ``_segy``   -- ``SegyGather``, ``read_segy``, ``write_segy``
      (``segyio``).
  Per-submodule import block pruned via ``ruff check --fix`` (32
  unused imports auto-removed). Largest submodule is ``_dlis`` at
  327 LoC.
- **``fwap.demos`` package**: the 1550-LoC monolith becomes a
  five-submodule package, grouped by book-chapter theme. The package
  ``__init__`` re-exports all 13 ``demo_*`` functions so
  ``fwap.cli``'s ``_DEMOS`` dispatch table (``_demos.demo_X``) and
  ``tests/test_demos.py`` keep working unchanged. Submodules:
    - ``_common``       -- canonical synthetic gather shared by the
      picker, separation, tau-p, and SEG-Y round-trip demos
      (``_CANONICAL_VP/VS/VST``, ``_canonical_monopole_gather``).
    - ``_signal``       -- Part 1 + 2 demos (``demo_stc_picker``,
      ``demo_pseudo_rayleigh``, ``demo_wave_separation``,
      ``demo_tau_p_separation``); imports the canonical-gather
      helper from ``_common``.
    - ``_inversion``    -- Part 3 + 4 demos (``demo_intercept_time``,
      ``demo_dipole``, ``demo_dip``).
    - ``_extensions``   -- Q / anisotropy / LWD demos
      (``demo_attenuation``, ``demo_alford``, ``demo_lwd``).
    - ``_io_roundtrip`` -- LAS / DLIS / SEG-Y round-trip demos
      (``demo_las_roundtrip``, ``demo_dlis_roundtrip``,
      ``demo_segy_roundtrip``); imports the canonical-gather helper
      from ``_common``. Largest submodule is ``_extensions`` at 521
      LoC; the per-submodule import block was pruned via ``ruff
      --fix`` (108 unused imports auto-removed).
- **``fwap.anisotropy`` package**: the 1867-LoC monolith becomes a
  four-submodule package with the public surface re-exported by
  ``fwap/anisotropy/__init__.py`` (so ``from fwap.anisotropy import
  alford_rotation`` and the ``fwap.anisotropy.X``-style ``:func:``
  cross-references in other modules' docstrings keep resolving).
  Submodules:
    - ``_alford`` (cross-dipole rotation: :class:`AlfordResult`,
      :func:`alford_rotation`, :func:`alford_rotation_from_tensor`,
      :class:`StressAnisotropyEstimate`,
      :func:`stress_anisotropy_from_alford`);
    - ``_thomsen`` (Thomsen :math:`\gamma` and Stoneley
      :math:`\to C_{66}`: :class:`ThomsenGammaResult`,
      :func:`stoneley_horizontal_shear_modulus[_corrected]`,
      :func:`thomsen_gamma`, :func:`thomsen_gamma_from_logs`);
    - ``_vti_inversion`` (vertical-well VTI moduli summary + walkaway-
      VSP :math:`\epsilon, \delta` inversion: :func:`c33_from_p_pick`,
      :class:`VtiModuli`, :func:`vti_moduli_from_logs`,
      :class:`ThomsenEpsilonDeltaResult`,
      :func:`thomsen_epsilon_delta_from_walkaway_vsp`; depends on
      ``_thomsen`` for :func:`thomsen_gamma`,
      :func:`thomsen_gamma_from_logs`, and
      :func:`stoneley_horizontal_shear_modulus_corrected`);
    - ``_vti_dispersion`` (Backus averaging + Christoffel phase / group
      velocities: :class:`BackusResult`, :func:`backus_average`,
      :func:`vti_phase_velocities`, :class:`VtiGroupVelocities`,
      :func:`vti_group_velocities`).
  Largest submodule is ``_vti_dispersion`` at 615 LoC. All 21 public
  names ride out of the package ``__init__`` so the public-API guard
  (``scripts/check_public_api.py``) passes unchanged.
- **``fwap.rockphysics`` split**: the seven Stoneley-wave petrophysical
  estimators (slowness / amplitude permeability indicators,
  Tang-Cheng-Toksoz inversion, Hornby aperture, the
  ``stoneley_fracture_density`` combiner, and the slow-formation
  ``vs_from_stoneley_slow_formation``) move into a new
  :mod:`fwap.stoneley` module. ``fwap.rockphysics`` shrinks from
  1374 to 428 LoC and now contains only elastic-moduli core plus
  Gassmann / Reuss / Voigt / Hill. Public API is preserved via the
  package re-exports in ``fwap/__init__.py``; ``from fwap import
  stoneley_permeability_indicator`` etc. still resolves.
- **``fwap.picker`` package**: the 2260-LoC monolith becomes a
  six-submodule package (``_types``, ``greedy``, ``viterbi``,
  ``posterior``, ``shape``, ``quality``) with the shared trellis
  primitives in ``viterbi.py`` and ``posterior.py`` importing them.
  Largest submodule is 676 LoC.
- **``fwap.geomechanics`` package**: the 2197-LoC monolith becomes
  a four-submodule package (``indices``, ``pressures``, ``vertical``,
  ``inclined``); the sole cross-submodule dependency is
  ``inclined.py`` importing ``MudWeightWindow`` from ``vertical.py``.
- **Helper consolidations**: new ``m_per_s_to_us_per_ft`` in
  :mod:`fwap._common` replaces six inline ``1.0e6 / v * 0.3048``
  expressions in ``cli.py`` / ``demos.py``. New private
  ``_mohr_coulomb_q`` (in ``fwap.geomechanics.vertical``) labels the
  ``(1+sin(phi))/(1-sin(phi))`` stress ratio used by both the
  vertical and inclined breakout calculators. New private
  ``_principal_stresses_at_pw`` (in ``fwap.geomechanics.inclined``)
  composes ``inclined_wellbore_wall_stresses`` with
  ``_wall_principal_stresses`` for a single candidate mud pressure,
  shared by ``inclined_breakout_pressure`` and
  ``inclined_breakdown_pressure``.

### Fixed
- **Cylindrical-solver characterisation test now tolerates SciPy
  rounding**: the six ``test_dispersion_matches_golden[…]`` cases
  were drifting 1-5 ULPs (max relative error ~1e-15) on
  SciPy 1.17 against goldens captured on an older SciPy -- pure
  IEEE round-off in ``scipy.optimize.brentq`` /
  ``scipy.special``, not a regression. Loosened the comparison
  from ``np.array_equal`` to ``np.allclose(rtol=1e-12)`` while
  still asserting the NaN-mask is preserved exactly.
- **``test_semblance_scale_invariant`` hypothesis strategy** no
  longer explores inputs whose squares flush to subnormal
  float64. Below ~1.5e-150 the squared sums in
  :func:`fwap.coherence.semblance` underflow and the scale
  identity ``semblance(alpha * x) == semblance(x)`` breaks for
  purely numerical reasons; the strategy now keeps ``|x|`` above
  that threshold (with zero still explicitly allowed).

### Changed
- **``quadrupole_dispersion`` now auto-dispatches to a fast-formation
  path when ``V_S > V_f``** (Roadmap A, plan item E in
  ``docs/plans/cylindrical_biot.md``). Direct n=2 sister of the
  plan item B work on ``flexural_dispersion``: previously
  fast-formation inputs returned NaN throughout (with a
  documented "plan item E follow-up" caveat), now they dispatch
  to a new private ``_quadrupole_dispersion_fast_formation`` that
  brentq's the imaginary part of
  :func:`_modal_determinant_n2_complex` along the real-``k_z``
  axis in the ``(omega/V_S, omega/V_R)`` bracket, with
  continuation across frequencies. Slow-formation behaviour is
  unchanged bit-for-bit.

  As with the n=1 case, the converged ``k_z`` is real to
  floating-point precision: the formation P/S branches stay
  bound, so the mode is bound; the only effect of ``F^2 < 0`` is
  an overall ``i^k`` phase that makes the determinant
  predominantly imaginary at real ``k_z``, reducing the root
  condition to ``Im(det) = 0``. The n=2 determinant magnitudes
  are about 15 orders larger than the n=1 sister, so the
  absolute residual at the converged root sits at ~10^8 rather
  than ~10^4; the relative residual ``|Im(det)|/|det|`` is at
  machine precision in both cases.

### Added
- **``_modal_determinant_n2_complex``** (Roadmap A, plan item E
  scaffolding). Complex-``k_z`` n=2 quadrupole modal determinant
  with optional ``leaky_p`` / ``leaky_s`` flags, structurally
  identical to :func:`_modal_determinant_n2` with K-Bessel
  evaluations swapped for the Hankel analytic continuation in
  the leaky regime via :func:`_k_or_hankel`. Fluid I-Bessel
  handles complex ``F`` transparently via ``scipy.special.iv``.
  In the fully-bound regime (real ``kz``, both flags False) the
  result agrees with the real-only sister to floating-point
  precision -- the regression invariant tested in
  ``tests/test_cylindrical_solver.py::test_complex_n2_matches_real_in_bound_regime``.

  Four new tests added for the leaky-quadrupole deliverable:
  bound-regime regression, slow-formation bit-identical guard
  (``quadrupole_dispersion`` reproduces an open-coded brentq +
  bracket-helper reference value), fast-formation regime sanity
  (velocities in ``(V_R, V_S)``, attenuation_per_meter is None),
  ``Im(det)/|det|`` machine-precision check at converged
  fast-formation roots, and frequency-order invariance.

- **``quadrupole_dispersion`` public API** (Roadmap A, plan item D
  in ``docs/plans/cylindrical_biot.md``). Real-valued n=2 modal-
  determinant solver for the slow-formation (``V_S < V_f``) bound
  regime, the LWD-quadrupole mode framed in Tang & Cheng 2004
  sect. 2.5. Tracks the lowest-``k_z`` zero of a 4x4
  :func:`_modal_determinant_n2` across the input frequency grid
  with the same ``brentq`` + bracket-expansion pattern as
  ``stoneley_dispersion`` and ``flexural_dispersion``. Returns
  ``BoreholeMode(name="quadrupole", azimuthal_order=2)``.

  Implementation: extends the existing n=1 derivation by the rules
  ``(I_0, I_1, K_0, K_1) -> (I_{n-1}, I_n, K_{n-1}, K_n)`` and
  ``azimuthal-derivative factor 1 -> n``, with two structural
  factors that are zero at n=1 but finite at n>=2: a
  ``2 n(n+1)`` overall-rank coefficient that turns the
  ``+ 4 K_1(pa)/a^2`` 1/r^2 correction in M22 into
  ``+ 12 K_2(pa)/a^2`` at n=2, and an ``(n^2-1)/a^2`` correction
  to the sigma_rz C-coefficient that vanishes at n=1 but adds
  a ``+ 3/a^2`` term to M43 at n=2. Specialised to n=2 the
  matrix uses the ``(K_1, K_2)`` and ``(I_1, I_2)`` Bessel
  index pairs.

  **Slow-formation only in this release**: the fast-formation
  (``V_S > V_f``) leaky-quadrupole regime needs the same complex-
  modal-determinant scaffolding that plan item B used for
  fast-flexural and is plan item E. Fast formations return
  all-NaN.

  ``fwap.lwd.lwd_quadrupole_priors`` now points at the new
  ``quadrupole_dispersion`` for callers that have full formation
  properties; the rectangular-window prior factory is retained
  as a Viterbi seed for the rough-V_S case where the full set
  of formation parameters is not available.

  8 new tests cover the dataclass contract, slow-formation
  finite-output + velocity-window sanity, fast-formation
  all-NaN guard, below-cutoff NaN, the ``slowness > 1/V_S``
  invariant, the local-zero property of the modal determinant
  at converged roots, and input validation
  (non-positive scalars, ``vp <= vs``, non-positive freq).

### Changed
- **``flexural_dispersion`` now auto-dispatches to a fast-formation
  path when ``V_S > V_f``** (Roadmap A, plan item B in
  ``docs/plans/cylindrical_biot.md``). Previously the public
  ``flexural_dispersion`` returned NaN throughout for any fast
  formation -- a documented limitation. The function now detects
  ``V_S > V_f`` at call time and dispatches to a new private
  ``_flexural_dispersion_fast_formation`` that brentq's the
  imaginary part of :func:`_modal_determinant_n1_complex` along
  the real-``k_z`` axis. Slow-formation behaviour is unchanged
  bit-for-bit (the dispatch is purely additive).

  **Empirical finding that informed the implementation**: in the
  canonical ``(V_R, V_S)`` velocity window the converged ``k_z``
  is real to floating-point precision rather than complex. The
  earlier "fast-formation flexural is leaky and needs complex-
  ``k_z`` Mueller iteration" framing in the Roadmap-A comments
  was over-stated for this particular root: the formation P/S
  branches stay bound in this regime, so the mode is also bound.
  The complex modal determinant is needed only because ``F^2 < 0``
  introduces an overall ``i^k`` phase that makes the determinant
  predominantly imaginary at real ``k_z``; the root condition
  reduces to ``Im(det) = 0`` and brentq along the real axis is
  the natural tool. Truly leaky n=1 modes with non-trivial
  ``Im(k_z) > 0`` (higher-order leaky flexural, fast-formation
  pseudo-flexural) need the complex marcher and remain out of
  scope for this routine.

  Backward compatibility note: callers that explicitly relied on
  ``flexural_dispersion`` returning all-NaN for fast formations
  must now check ``np.isfinite`` per element. The previous
  "all-NaN sentinel" was documented as a stop-gap pending plan
  item B, so the change is in the spirit of the original API
  rather than against it.

### Added
- **``_modal_determinant_n1_complex``** (Roadmap A, plan item B
  scaffolding). Complex-``k_z`` n=1 dipole modal determinant with
  optional ``leaky_p`` / ``leaky_s`` flags, structurally
  identical to the real-valued :func:`_modal_determinant_n1`
  with K-Bessel evaluations swapped for the Hankel analytic
  continuation in the leaky regime. Fluid I-Bessel handles
  complex ``F`` transparently via ``scipy.special.iv``. In the
  fully-bound regime (real ``kz``, both flags False) the result
  agrees with the real-only sister to floating-point precision
  -- the regression invariant tested in
  ``tests/test_cylindrical_solver.py::test_complex_n1_matches_real_in_bound_regime``.

  Five new tests added for the leaky-flexural deliverable:
  bound-regime regression, slow-formation bit-identical guard
  (``flexural_dispersion`` reproduces an open-coded brentq +
  bracket-helper reference value), fast-formation finite-output
  + velocity-window check (``V_R < v < V_S``), local-zero
  property of ``Im(det)`` at converged fast-formation roots,
  and frequency-order invariance of the fast-formation marcher
  (ascending and descending input grids produce identical output).

- **Cutoff handling + branch tracker** (Roadmap A, plan item C
  in ``docs/plans/cylindrical_biot.md``). Adds a validator-aware
  marcher that distinguishes a converged-but-out-of-regime root
  from a root-finder failure, tolerates a small budget of
  consecutive bad steps before giving up, and recovers from
  one-off branch hops by resuming the march from the last good
  step. Three new symbols in ``fwap.cylindrical_solver``:

  * ``_classify_marcher_step(kz_root, omega, validator) -> str`` --
    private classifier returning ``"ok"``, ``"regime_exit"``, or
    ``"convergence_failure"``. Validator exceptions
    (``ValueError`` / ``ArithmeticError``) collapse to
    ``"regime_exit"`` so a numerically ill-conditioned step does
    not abort the march.

  * ``BranchSegment`` -- public dataclass representing a
    contiguous stretch of finite samples in a dispersion curve
    (``start_idx``, ``end_idx``, ``freq``, ``kz``). Re-exported
    at top level. ``len(segment)`` returns the inclusive sample
    count.

  * ``segments_from_kz_curve(freq_grid, kz_curve)
    -> list[BranchSegment]`` -- public splitter that walks a
    marcher output and emits one ``BranchSegment`` per maximal
    run of finite ``kz``. Re-exported at top level.

  * ``_march_complex_dispersion_validated(det_fn, freq_grid,
    kz_start, *, validator, max_consecutive_invalid, xtol)`` --
    private validator-aware marcher. ``validator(kz, omega) ->
    bool`` says whether a converged step belongs to the regime
    the caller wants to track; failed steps stay NaN, do not
    update the continuation seed, and count against
    ``max_consecutive_invalid``. Setting that to ``0`` recovers
    the strict-stop semantics of the original
    ``_march_complex_dispersion``.

  ``pseudo_rayleigh_dispersion`` is refactored to drive the new
  marcher with a leaky-S-regime validator (``Im(k_z) > 0`` and
  ``1/V_P < slowness < 1/V_S``). On the standard fast-formation
  parameter set the refactor recovers steps that the previous
  step-by-step loop dropped to single-step root hops, returning
  one contiguous segment over the supported band. 12 new tests
  cover the classifier verdicts (each return value, plus
  exception-as-regime-exit), the dataclass contract,
  ``segments_from_kz_curve`` (NaN-gap split, all-NaN, mismatched
  inputs), the validated marcher (skip-and-continue, budget
  exhaustion, empty grid, zero-budget = strict semantics), and
  the pseudo-Rayleigh single-segment regression.

- **``pseudo_rayleigh_dispersion`` public API** (Roadmap A,
  plan item A in ``docs/plans/cylindrical_biot.md``). First
  leaky-mode product on top of the L1-L3 scaffolding. Tracks the
  n=0 leaky root with the formation S wave radiating outward
  (``s``-branch leaky) while the fluid pressure and the formation
  P wave stay bound. Mode exists in fast formations only
  (``V_S > V_f``) above a low-frequency cutoff where it merges
  with the body S head wave.

  Implementation: walks the input frequency grid from high to low
  internally. Seeds at the highest frequency with slowness
  ``0.95 / V_S`` (5% inside the leaky-S regime) plus a small
  positive imaginary part; subsequent steps use the previous
  step's converged ``k_z`` rescaled to the next ``omega`` as the
  seed (constant-slowness extrapolation). The marcher stops as
  soon as

  1. ``scipy.optimize.root`` fails to converge, or
  2. the converged ``Im(k_z)`` is non-positive (mode merged with
     the bound regime, or root finder drifted to a non-physical
     growing branch), or
  3. the converged slowness falls outside ``(1/V_P, 1/V_S)``
     (root hopped to a different physical regime).

  Remaining low-frequency samples stay NaN; branch-stitching
  across the cutoff is plan item C. Returns
  :class:`BoreholeMode` with ``slowness = Re(k_z)/omega`` and the
  newly-populated ``attenuation_per_meter = Im(k_z)`` field.

  Re-exported at top level as ``pseudo_rayleigh_modal_dispersion``
  to disambiguate from the existing
  :func:`fwap.synthetic.pseudo_rayleigh_dispersion`
  phenomenological callable-factory model
  (kept unchanged for backward compatibility). Both names remain
  accessible by their fully-qualified module paths.

  10 tests cover input validation (slow-formation rejection,
  non-positive inputs, invalid frequencies), output contract
  (``BoreholeMode`` shape and ``attenuation_per_meter``
  population), regime sanity (slowness strictly inside
  ``(1/V_P, 1/V_S)``; ``Im(k_z) > 0`` everywhere finite;
  velocity strictly between ``V_S`` and ``V_P``), the
  frequency-order invariance of the marcher (ascending and
  descending input grids produce identical per-frequency
  output), the empty-frequency-array no-op, and the local-zero
  property of the determinant at converged roots
  (``|det(root)| < 1% * |det(off-root)|``).

- **``BoreholeMode.attenuation_per_meter`` field** (Roadmap A
  continuation, dataclass extension for upcoming leaky-mode
  solvers). Adds an optional
  ``attenuation_per_meter: ndarray | None`` field to the
  :class:`BoreholeMode` dataclass, default ``None`` for backward
  compatibility with the existing bound-mode solvers (Stoneley
  and slow-formation flexural). Future leaky-mode solvers will
  populate the field with ``Im(k_z)`` to expose the spatial
  attenuation rate in 1/m. 3 tests cover the field contract:
  default-None, accepts an ndarray, and the existing Stoneley
  solver continues to return ``None`` (bound mode -> no
  attenuation).

- **Complex-``k_z`` root finder + frequency-marching tracker for
  the leaky-mode solver** (Roadmap A continuation, phase L3). Two
  new private helpers in ``fwap.cylindrical_solver``:

  * ``_track_complex_root(det_fn, kz_start, *, xtol=1e-12)`` --
    single-frequency complex root finder. Wraps
    ``scipy.optimize.root(method='hybr')`` on the (Re, Im) split
    of the complex residual. Catches det-evaluation exceptions
    and converts them to large penalty residuals so the iterator
    backs off rather than aborting. Returns the converged
    complex ``k_z`` or ``None`` on failure.

  * ``_march_complex_dispersion(det_fn, freq_grid, kz_start, *,
    xtol)`` -- frequency-marching loop with **scale-invariant
    continuation**: the next step's initial guess is
    ``k_z_prev * (f / f_prev)``, which keeps the seed on the
    constant-slowness extrapolation of the previous step. This
    handles the multiplicative ``k_z`` jumps typical of bound-
    mode dispersion (Stoneley ``k_z`` doubles when frequency
    doubles) without losing the local-quadratic convergence of
    the per-step solver. Returns a NaN-padded complex array;
    once a step fails, the remaining steps stay NaN.

  Branch tracking across leaky-vs-bound transitions is the
  caller's responsibility (the marcher just walks the grid as
  given). Standard pattern: ``det_fn`` internally calls
  ``_detect_leaky_branches`` from L2 to re-classify the regime
  at each evaluation, OR the caller splits the frequency grid
  at the cutoff and calls the marcher separately on each side.

  This phase is purely the root-finding mechanics. The leaky-
  mode public APIs (``pseudo_rayleigh_dispersion``, fast-
  formation flexural, quadrupole) build on top in phases L4-L6.

  7 new tests cover: linear synthetic root (exact recovery);
  closest-root selection on a quadratic; exception-safety of the
  tracker; synthetic linear dispersion (constant complex
  slowness); large-multiplicative-frequency-jump continuation
  (Stoneley-like ``k_z`` scaling); smoothly drifting complex
  dispersion (both Re and Im of slowness drifting with
  frequency); end-to-end regression -- the marcher composed with
  ``_modal_determinant_n0_complex`` recovers the existing
  ``stoneley_dispersion`` result to ~1e-10 relative precision.

- **Leaky-mode scaffolding for the cylindrical-Biot solver**
  (Roadmap A continuation, phases L1 + L2). Mathematical
  scaffolding -- complex-``k_z`` sign conventions, Hankel-
  function ansatz for outgoing-wave BCs, branch-cut handling --
  plus a complex-aware n=0 modal determinant
  ``_modal_determinant_n0_complex(kz, omega, vp, vs, rho, vf,
  rho_f, a, *, leaky_p=False, leaky_s=False)`` that supports
  complex ``k_z`` and switchable K-Bessel / Hankel evaluators
  per radial branch. Plus two helpers:
  ``_detect_leaky_branches(kz, omega, vp, vs, vf)`` classifies
  ``(F, p, s)`` as bound or leaky from the sign of
  ``Re(alpha^2)``;
  ``_k_or_hankel(n, alpha, r, *, leaky)`` returns
  ``(K_n, K_{n+1})`` either as standard modified Bessels (bound)
  or as the Hankel-via-analytic-continuation
  ``(pi/2) i^{n+1} H_n^{(2)}(i alpha r)`` (leaky). The whole
  family is private (underscore-prefixed) because the public
  leaky-mode APIs (pseudo-Rayleigh, fast-formation flexural,
  quadrupole) require the L3 complex root finder which is the
  next planned PR. Regression invariant: in the bound regime
  (real ``kz``, both leaky flags ``False``) the complex
  evaluator agrees with the existing real ``_modal_determinant_n0``
  to floating-point precision (rel < 1e-12; imaginary part
  identically zero) -- this is the test guard that lets future
  L3+ work refactor confidently. 9 new tests cover: real-vs-
  complex agreement at multiple ``kz``; sign-change preservation
  across the Stoneley root; branch-detector classification in
  three regimes (bound, pseudo-Rayleigh, fast-flexural); Bessel-
  vs-Hankel helper agreement on the bound branch; finiteness of
  Hankel-branch evaluations and complex-``kz`` evaluations. The
  ``[Unreleased]`` section in ``docs/roadmap.md`` (item A) gets
  the L1-L7 sequencing detail; the bound-mode half of A remains
  shipped, the leaky-mode half is now mid-flight.

- **Tensile-strength rock-physics correlation**
  (``fwap.geomechanics.tensile_strength_from_ucs``). One-line
  convenience function returning ``T = ratio * UCS`` with default
  ``ratio = 0.10`` (typical sandstones). Documented as a Hoek-Brown-
  style "tension cutoff" rather than the Mohr-Coulomb linear
  extrapolation -- the latter overestimates real-rock tensile
  strength by ~3x and is a commonly-flagged geomechanical pitfall.
  Provides published lithology-specific ratio ranges (sandstones
  0.07-0.12, shales 0.04-0.08, limestones 0.08-0.15, crystalline
  rocks 0.10-0.20) so users can re-tune. Closes the last item on
  the original session-1 list of possible extensions; round-trip
  use is documented (compute UCS via
  ``unconfined_compressive_strength``, T via this function, feed
  T into ``tensile_breakdown_pressure``). 7 new tests cover the
  closed-form linearity, broadcasting, zero-UCS edge case,
  round-trip into the breakdown pressure, and input validation.

- **Inclined tensile-breakdown pressure + inclined safe mud-weight
  window** -- completes the wellbore-stability symmetry between
  vertical and inclined wells. Two new public functions in
  ``fwap.geomechanics``:

  * ``inclined_breakdown_pressure(...)``: Mohr-style tensile-
    failure scan around the wall of an inclined well. Diagonalises
    the (theta, z) 2x2 sub-block at each azimuth, finds the
    smallest eigenvalue lambda_-(theta, P_w), and bisects on the
    worst-azimuth tensile-failure margin
    ``min_theta lambda_- - alpha P_p + T``. Convention follows the
    vertical ``tensile_breakdown_pressure``: ignores the radial
    principal stress sigma_rr (which would always be most tensile
    under positive pore pressure and would not match the standard
    Hubbert-Willis fracture-initiation interpretation).

  * ``inclined_safe_mud_weight_window(...)``: convenience wrapper
    that combines ``inclined_breakout_pressure`` and
    ``inclined_breakdown_pressure`` and returns the same
    :class:`MudWeightWindow` dataclass used by the vertical
    counterpart, with ``width`` and ``is_drillable`` properties.

  Vertical-well consistency: at ``well_inclination_deg = 0`` both
  functions match the vertical closed forms to within the
  azimuth-grid resolution (verified by test). For a typical
  drillable scenario, the safe window narrows from 31.25 MPa
  (vertical) to 13.75 MPa (horizontal) -- breakout rises, breakdown
  falls, net width drops; the well remains drillable but with
  much less mud-weight margin. 10 new tests cover: vertical
  consistency for both bounds; monotonicity in inclination,
  tensile strength, pore pressure; the not-drillable-in-tension-
  at-zero-mud edge case; ``MudWeightWindow`` dataclass contract;
  vertical-window equivalence; window narrowing with inclination;
  horizontal-well drillability; input validation.

- **Inclined-wellbore stability** -- generalized Kirsch wall
  stresses (Hiramatsu-Oka 1962, Fairhurst 1968) and Mohr-Coulomb
  shear-breakout pressure for arbitrarily oriented wells in
  ``fwap.geomechanics``. Two new functions:
  ``inclined_wellbore_wall_stresses(sigma_v, sigma_H, sigma_h, *,
  well_inclination_deg, well_azimuth_deg,
  azimuth_around_wall_deg, mud_pressure, poisson)`` returns the
  four wall stress components ``(sigma_theta, sigma_z,
  sigma_theta_z, sigma_r)`` after rotating the principal-stress
  tensor into well-aligned coordinates;
  ``inclined_breakout_pressure(...)`` finds the critical mud
  pressure by scanning over wall azimuth, computing principal
  stresses (via 2x2 eigenvalue decomposition of the (theta, z)
  sub-block plus the trivial radial principal stress), applying
  Mohr-Coulomb at each azimuth, and bisecting on the worst-
  azimuth failure margin.

  Vertical-well consistency: at ``well_inclination_deg = 0`` the
  wall-stress formulas reduce exactly to the existing
  ``kirsch_wall_stresses``, and the breakout pressure agrees
  with the closed-form ``mohr_coulomb_breakout_pressure`` to
  within the azimuth grid resolution. Inclined wells in normal-
  fault stress regimes need progressively more mud-pressure
  support (verified by test on a drillable scenario: 33.75 MPa
  vertical -> ~40 MPa horizontal).

  Documented assumptions: principal-stress-aligned far-field
  stresses (sigma_v vertical, sigma_H/sigma_h horizontal); no
  shear stresses in the un-rotated frame (the rotation introduces
  them in the well frame). The wall is assumed to fail in shear
  per Mohr-Coulomb; tensile-breakdown for inclined wells is a
  follow-up. The function raises informatively when the wall is
  unconditionally unstable (no mud pressure can stabilise the
  geometry) or when ``friction_angle_deg`` is out of range.
  10 new tests cover: vertical-well wall-stress and breakout-
  pressure consistency with the closed forms; inclination
  monotonicity; horizontal-well azimuth dependence; periodicity
  and symmetry of the wall stresses; sigma_r = mud pressure
  identity; isotropic-horizontal-stress vertical-well limit;
  input validation; not-drillable-geometry error message.

- **Bowers (1995) sonic pore-pressure with unloading branch**
  (``fwap.geomechanics.pore_pressure_bowers``). Closes the
  Bowers-method follow-up flagged in PR #27's CHANGELOG.
  Velocity-effective-stress closed form
  ``V = V_ml + A * sigma_eff^B`` with two branches:

  * **Loading (virgin curve)**: pore pressure from
    ``sigma_eff = ((V - V_ml) / A)^(1/B)``. Selected when
    ``sigma_max_pa`` is None.
  * **Unloading**: pore pressure from
    ``sigma_eff = sigma_max * ((V - V_ml) / (A * sigma_max^B))^(U/B)``,
    selected when ``sigma_max_pa`` is supplied. The unloading
    exponent ``U > B`` makes the curve steeper than loading,
    which is the physical signature of unloading-driven
    overpressure (gas generation, clay diagenesis,
    hydrocarbon expulsion) that Eaton's method
    (``pore_pressure_eaton``) under-estimates.

  Both branches close in closed form with no numerical inversion.
  Default calibration ``(V_ml, A, B, U) = (1524, 14.02, 0.673,
  3.13)`` is Bowers' (1995) Gulf of Mexico shale fit; users
  should re-calibrate against well data for other basins.
  Unit convention: SI throughout (Pa for stresses, m/s for
  velocity); ``A`` is in (m/s) / MPa^B with the Pa↔MPa conversion
  internal. Loading/unloading branch selection is the user's
  responsibility -- the function does not auto-detect the regime
  because that requires burial-history information not on the log.
  11 new tests cover: round-trip recovery on both branches; mudline-
  velocity edge case (V = V_ml gives sigma_eff = 0); monotonicity
  in V; unloading > loading prediction at the same V (the
  Eaton-fix signature); end-to-end pipeline with closure_stress;
  and input validation (V < mudline, non-positive calibration
  constants and sigma_max).

- **VTI group velocities** (``fwap.anisotropy.vti_group_velocities``,
  ``VtiGroupVelocities``). Closes the wavefront-modelling deliverable
  flagged as a follow-up in PR #30: group velocity (the speed of
  energy / wavefront propagation) and group angle (the direction of
  energy propagation, generally different from the phase-angle
  direction in anisotropic media) for the three VTI modes
  (qP, qSV, SH). Tsvankin (2001) sect. 1.3 closed forms:

      v_g_x = v_p sin(theta) + (dv_p/dtheta) cos(theta)
      v_g_z = v_p cos(theta) - (dv_p/dtheta) sin(theta)
      |v_g| = sqrt(v_p^2 + (dv_p/dtheta)^2)
      tan(psi) = v_g_x / v_g_z

  The dv_p/dtheta derivative is computed numerically via
  np.gradient (central differences in the interior, one-sided at
  the grid endpoints); avoids the algebraic complexity of the
  closed-form Tsvankin derivatives at minimal accuracy cost. Output
  is a ``VtiGroupVelocities`` dataclass with three velocities and
  three group angles. Wavefront-plotting use:
  ``x = v_g * sin(psi); z = v_g * cos(psi)``. 12 new tests cover:
  isotropic limit (group exactly equals phase, psi exactly equals
  theta to floating-point precision); dataclass contract; group =
  phase at theta = 0 and pi/2; psi = 0 at theta = 0 and psi = pi/2
  at theta = pi/2 (symmetry-axis-aligned wavefronts); group angle
  differs from phase angle off-axis (qSV refracts toward symmetry
  for the Berea-VTI fixture, SH refracts away); all velocities
  positive and qP > qSV everywhere; input validation rejecting
  one-point and non-increasing grids; end-to-end Cartesian-
  wavefront monotonicity check on a Backus-derived medium.

- **VTI phase velocities** (``fwap.anisotropy.vti_phase_velocities``).
  Christoffel-determinant solution for the three plane-wave modes
  (quasi-P, quasi-SV, SH) in a transversely-isotropic medium with
  vertical symmetry axis, propagating at phase angle ``theta`` from
  the symmetry axis. Tsvankin (2001) eq. 1.41 in standard form;
  closed-form quadratic with the standard ``+/- sqrt(D)``
  discriminant for qP/qSV plus the decoupled SH formula
  ``v_SH^2 = (C44 cos^2 + C66 sin^2)/rho``. Natural consumer of
  ``backus_average`` output (the new function takes the five VTI
  elastic constants directly). Useful for forward-modelling
  wavefronts, qP/qSV crossover analysis, and Thomsen-anisotropy
  consistency checks. 12 new tests cover: vertical and horizontal
  limits (each velocity recovers the corresponding C-modulus
  square root); isotropic limit (constant in theta, qSV=SH);
  qSV / SH degeneracy at vertical; Thomsen-anisotropy signatures
  (epsilon > 0 -> v_qP at horizontal > vertical; gamma > 0 ->
  v_SH at horizontal > vertical); shape and broadcasting; round-
  trip with ``backus_average``; input validation. Pure phase
  velocity; group velocity is a planned follow-up.

- **Tensile-breakdown pressure + safe mud-weight window**
  (``fwap.geomechanics.tensile_breakdown_pressure``,
  ``MudWeightWindow``, ``safe_mud_weight_window``). Closes the
  drilling-decision pipeline by adding the upper bound of the
  safe mud-weight range -- the Hubbert-Willis (1957) fracture-
  initiation pressure
  ``P_w_break = 3 sigma_h - sigma_H + T - alpha P_p`` -- to
  complement the Mohr-Coulomb breakout (lower bound) shipped in
  the same release. The ``MudWeightWindow`` dataclass packages
  both bounds plus convenience properties ``width``
  (= breakdown - breakout) and ``is_drillable`` (= width > 0)
  for diagnostic output. Strong negative-width result on the
  PR #28 smoke-test scenario flagged it as "not drillable in
  this geometry without intervention" -- exactly the kind of
  immediate diagnostic this combiner is meant to produce. 11
  new tests cover: Hubbert-Willis closed-form match;
  at-critical-pressure inverse check (Kirsch hoop stress at
  theta=0 equals -T after effective-stress correction);
  monotonicity in tensile strength, horizontal-stress
  anisotropy, pore pressure; biot_alpha=0 limit; window
  dataclass contract; pure-pass-through equivalence with the
  individual primitives; per-depth drillability flag on vector
  input.

- **Wellbore-stability analysis** — Kirsch (1898) wall-stress
  primitive plus Mohr-Coulomb shear-breakout pressure
  (``fwap.geomechanics.kirsch_wall_stresses`` and
  ``mohr_coulomb_breakout_pressure``). Extends the geomechanics
  module from indices to a drilling-decision deliverable: the
  critical mud pressure below which the borehole wall fails in
  shear at the breakout azimuth (perpendicular to the maximum
  horizontal stress). Combined with the existing
  ``overburden_stress`` -> ``pore_pressure_eaton`` ->
  ``closure_stress`` -> ``unconfined_compressive_strength``
  pipeline, the geomechanics module now produces the full
  drilling stress-state log from a sonic + density acquisition
  alone. Closed-form derivation:
  ``P_w_crit = (3 sigma_H - sigma_h + (q-1) alpha P_p - UCS)
  / (1 + q)`` with ``q = (1 + sin phi) / (1 - sin phi)``.
  Documented assumptions: vertical well, normal-fault stress
  regime (sigma_v as the intermediate principal stress is
  assumed to be safe), no tensile-failure check
  (the upper bound of the safe mud-weight window is a planned
  follow-up). 14 new tests cover: Kirsch hand-derived values at
  the breakout and breakdown azimuths, isotropic-horizontal-
  stress azimuth-independence, mud-pressure linearity,
  Poisson-and-deviator coupling for sigma_z; MC at-critical-
  pressure inverse check, monotonicity in UCS / friction angle /
  pore pressure / horizontal-stress anisotropy, Tresca
  (zero-friction) and dry-rock (alpha=0) limits, friction-angle
  validation; and an end-to-end pipeline test that chains
  overburden -> pore pressure -> closure -> breakout from a
  synthetic 30-depth log.

- **Eaton (1975) sonic pore-pressure prediction**
  (``fwap.geomechanics.pore_pressure_eaton``). Closed-form
  pore-pressure log from a sonic-slowness log, an overburden-
  stress log, and a normal-compaction-trend slowness:
  ``P_p = sigma_v - (sigma_v - P_hydro) * (Dt_normal / Dt_obs)^n``
  with the standard Eaton exponent ``n = 3.0``. Plus a helper
  ``hydrostatic_pressure(depth, fluid_density=1000.0)`` that
  computes :math:`P_\mathrm{hydro} = \rho_w \, g \, z`. Closes
  a missing-input gap in the existing ``closure_stress``
  function: callers can now produce the full
  ``overburden -> pore -> closure`` stress-state pipeline from
  a sonic + density log alone. Documented limitations: the
  sonic Eaton method is calibrated for shales and undercompaction-
  driven overpressure; unloading mechanisms (gas generation,
  diagenesis) need Bowers' method, which is left as a follow-up.
  17 new tests cover: hydrostatic linearity and density scaling;
  Eaton's normal-compaction reduction to ``P_hydro``; severe-
  overpressure approach to ``sigma_v``; sub-hydrostatic /
  depleted-zone case; depth-vs-explicit-hydrostatic agreement;
  Eaton-exponent sensitivity; round-trip test with
  ``overburden_stress`` on a synthetic 30-depth log; input
  validation.

- **Unified Stoneley fracture-density log**
  (``fwap.rockphysics.stoneley_fracture_density``). Pure combiner
  that mixes the four primitive Stoneley indicators
  (``stoneley_permeability_indicator``,
  ``stoneley_amplitude_fracture_indicator``,
  ``stoneley_permeability_tang_cheng``,
  ``hornby_fracture_aperture``) into a single per-depth fracture-
  intensity score in ``[0, 1]``. The matrix-permeability output
  is used as a binary partitioning flag: depths where the TCT
  inversion returned NaN (out-of-model = simplified Biot-Rosenbaum
  cannot account for the observed slowness shift) keep the full
  slowness contribution; depths with finite kappa (matrix-explained)
  have the slowness contribution suppressed. Aperture term uses
  a tanh saturation with a 1 mm reference scale; weights and
  scales are tunable via keyword arguments. Heuristic combiner,
  not a calibrated geomechanical fracture density -- documented
  as such. 12 new tests cover: zero-indicator tight zone, score
  clipping to [0, 1], default-weight slowness-only and
  amplitude-only paths, matrix-partitioning logic, tanh-saturated
  aperture, NaN-aperture handling, partial monotonicity, and
  input validation.

### Added
- **Backus (1962) layered-medium averaging**
  (``fwap.anisotropy.backus_average``). Long-wavelength
  homogenisation of a stack of N isotropic layers into a single
  effective transversely-isotropic (VTI) elastic tensor with
  vertical symmetry axis. Returns a ``BackusResult`` with the
  five independent VTI elastic constants
  ``c11, c13, c33, c44, c66`` (Pa) plus the volume-weighted
  effective density. Layer-parallel components ``c11, c66`` are
  Voigt-like arithmetic averages; layer-perpendicular
  ``c33, c44`` are Reuss-like harmonic averages; ``c13`` is the
  standard Backus combination of ``lambda / (lambda + 2 mu)``
  averages. Useful for upscaling thinly-bedded sonic-log
  intervals to seismic resolution. 12 new tests cover: isotropic-
  limit recovery (single layer or uniform stack -> exact
  per-layer moduli), thickness-scale invariance (only volume
  fractions matter), Voigt-Reuss inequalities (``c66 >= c44`` and
  ``c11 >= c33`` always hold), positive-definiteness of the
  resulting tensor, hand-derived two-layer numerical check, and
  input validation.

### Changed
- **Repository-wide ``ruff format`` sweep**: 42 files reformatted to
  ruff-format defaults. Behaviour-preserving (full test suite still
  passes; same 433 / 1 skipped count as before the sweep). Closes
  Open Item E on the roadmap.
- **``ruff check`` lint debt cleanup**: 56 pre-existing lint
  warnings cleared from the tree. 52 ``I001`` (import-block
  ordering) auto-fixed by ``ruff check --fix`` -- mostly local
  ``import pytest`` blocks in tests that needed a blank line
  between ``import pytest`` and the subsequent ``from fwap.x
  import y``. 2 ``B023`` (loop-variable-not-bound) instances in
  ``stoneley_dispersion`` and ``flexural_dispersion`` fixed by
  binding ``omega`` as a default argument
  (``def _det(kz, omega=omega): ...``); the closure was always
  safe (the inner function was only called within the same loop
  iteration via ``brentq``) but explicit binding silences the
  warning and removes a footgun. 1 ``B007`` (unused loop
  variable ``lbl`` in ``demos.py``) renamed to ``_lbl``. 1
  ``F841`` (unused ``Vst`` in a Stoneley-omitted test) removed.

### Added
- **Pre-commit config** (``.pre-commit-config.yaml``) with both
  the ``ruff-check`` (with ``--fix``) and ``ruff-format`` hooks.
  Run ``pre-commit install`` after cloning to prevent format and
  lint drift on future commits.
- **Variable candidate budget for joint Viterbi picker**: when the
  raw per-depth tuple count ``prod(n_i + 1)`` would exceed
  ``max_triples_per_depth``, the trellis builder now automatically
  tightens per-mode top-K (preferring high-coherence candidates
  within each mode) to fit the budget rather than raising. Replaces
  the earlier hard-fail-on-overflow with graceful degradation;
  pathological peak-heavy STC surfaces no longer kill the sweep.
- **4-mode joint Viterbi**: ``viterbi_pick_joint`` and
  ``viterbi_posterior_marginals`` are now N-mode generic. Default
  priors changed from the (P, S, Stoneley) subset to the full
  ``DEFAULT_PRIORS`` (4 modes including PseudoRayleigh). Pass an
  explicit 3-mode subset to ``priors=`` if the previous default
  behavior is preferred. The 4-mode trellis is kept tractable by
  the variable-candidate-budget machinery above. Closes Open Item
  C on the roadmap (both sub-items).

### Changed
- ``viterbi_pick_joint`` and ``viterbi_posterior_marginals`` no
  longer reject 4-mode prior dicts. Empty prior dicts now raise
  ``ValueError`` with a clear message instead of silently producing
  empty picks.

### Added
- **Quantitative Stoneley permeability** via the Tang-Cheng-Toksoz
  (1991) simplified Biot-Rosenbaum closed form
  (`fwap.rockphysics.stoneley_permeability_tang_cheng`).
  Calibrated complement to the dimensionless rank-ordering
  returned by `stoneley_permeability_indicator`: takes the
  observed Stoneley slowness, a tight reference, and the standard
  set of Biot / fluid parameters (frequency, K_f, eta, rho_f,
  porosity, frame K_phi); returns absolute formation permeability
  in m^2 (multiply by ~9.87e-13 for darcies). Real-valued
  inversion (slowness shift only); the imaginary-part
  (attenuation) inversion is a follow-up. Out-of-model handling:
  `alpha_ST <= 0` (tight or noise-driven negative) clipped to
  `kappa = 0`; `alpha_ST >= K_f / (2 K_phi)` returns NaN
  (typical cause: open fractures requiring the
  `hornby_fracture_aperture` model rather than the matrix-flow
  Biot-Rosenbaum model). 11 new tests including a round-trip
  check against a Tang & Cheng (2004) fig 5.3-style synthetic
  (1-2 darcy permeable bed bracketed by 0.01-0.1 mD tight
  limestone). Closes Open Item B on the roadmap.
- **n=1 dipole flexural modal-determinant solver**
  (`fwap.cylindrical_solver.flexural_dispersion`). Companion to the
  existing n=0 Stoneley solver (Schmitt 1988); root-finds the
  zeros of the 4x4 isotropic-elastic dipole modal determinant in
  the bound-mode regime to produce the dipole flexural dispersion
  curve directly from the underlying boundary-value problem,
  replacing the rational-interpolation phenomenology in
  `fwap.cylindrical.flexural_dispersion_physical` with the
  cylindrical-Biot physics. Public surface: `flexural_dispersion(
  freq, *, vp, vs, rho, vf, rho_f, a)` returns a `BoreholeMode`
  with `name="flexural"` and `azimuthal_order=1`. Bound-mode
  regime only -- requires slow formations (`V_S < V_f`); fast
  formations and below-cutoff frequencies return NaN, matching
  the existing `stoneley_dispersion` out-of-regime convention.
  In a typical slow formation the recovered slowness sits at
  `1 / V_S` just above the geometric cutoff and rises toward
  slightly above `1 / V_R` at high frequency (the few-percent
  Scholte / fluid-loading offset that the phenomenological model
  does not capture). 12 new tests (slow-formation asymptotes,
  monotonicity, qualitative agreement with the phenomenological
  model, fast-formation NaN behavior, modal-determinant zero
  structure, input validation, dataclass contract). The full
  algebraic derivation -- field ansatz, displacements, stresses
  with the Lame reduction, BC strip, phase rescaling to a real
  4x4, low-f and high-f asymptotic cross-checks -- is documented
  in line in `cylindrical_solver.py` (substeps 1.1 through
  1.6.e). Closes the dipole half of the Open Item A
  ("Full cylindrical-Biot dispersion solver") on the roadmap;
  leaky-mode pseudo-Rayleigh and quadrupole n=2 remain open for
  follow-up via complex-`k_z` Mueller iteration with outgoing
  Hankel-function boundary conditions.
- **LWD (logging-while-drilling) phenomenological layer**
  (`fwap.lwd`). Models the steel-drill-collar contamination Tang &
  Cheng (2004) sect. 2.4-2.5 frame as the defining processing
  problem of the LWD era, plus the two practical responses:
  collar-band notching and quadrupole-source / receiver geometry.
  Public surface: `lwd_collar_mode(...)` returns a pre-configured
  Gabor `Mode` at the published 80-130 us/ft collar band;
  `synthesize_lwd_gather(...)` plants the collar on top of the
  formation modes; `notch_slowness_band(...)` is a slowness-band-
  stop filter via tau-p forward + cosine-tapered band-pass mask +
  adjoint **+ subtract-the-in-band** (the subtraction route
  preserves signals at slownesses outside the tau-p grid, e.g.
  Stoneley at ~217 us/ft survives a notch at ~92 us/ft); on the
  quadrupole side, `QuadrupoleRingGather`,
  `synthesize_quadrupole_lwd_gather(...)` builds a ring of n_rec
  >= 4 receivers with a `cos(2(theta - phi))` source pattern and
  `quadrupole_stack(data, azimuths, ...)` projects the ring onto
  m=2, rejecting m=0 / m=1 patterns by orthogonality;
  `lwd_quadrupole_priors()` returns a tool-aware picker priors
  dict. `fwap.demos.demo_lwd` and `fwap lwd` CLI provide a
  worked-example end-to-end. **Not** a layered cylindrical-Biot
  solver -- that is research-grade work and remains future work.
- **Stress-vs-intrinsic anisotropy classifier from a fast / slow
  flexural dispersion-curve crossover.** Sinha & Kostek (1996)
  showed that the two cross-dipole flexural dispersion curves of a
  stress-anisotropic formation cross over in frequency: the
  low-frequency mode samples the far-field rock fabric and the
  high-frequency mode samples the near-wellbore stress
  concentration, so a Δs(f) sign flip between the bands flags
  stress-induced anisotropy (intrinsic anisotropy shows no such
  crossover). New `classify_flexural_anisotropy(curve_a, curve_b,
  ...)` returns a `FlexuralDispersionDiagnosis` with a
  classification in `{"isotropic", "intrinsic", "stress_induced",
  "ambiguous"}`, the per-band Δs averages, the interpolated
  crossover frequency (when present, restricted to the bracket
  spanning the two band means), and a tuple of human-readable
  reasons for QC.
- **Slow-formation Vs from low-frequency Stoneley slowness**
  (`fwap.rockphysics.vs_from_stoneley_slow_formation`). Inverts the
  White (1983) tube-wave formula `S_ST^2 = 1/V_f^2 + rho_f / mu`
  for the formation shear velocity. Primary sonic-only Vs
  estimator for slow formations (V_S < V_fluid) where the
  formation has no critically-refracted S head wave on a monopole
  gather (Paillet & Cheng 1991, Ch. 3) and pseudo-Rayleigh does
  not exist. Same physics as
  `stoneley_horizontal_shear_modulus` (which returns C_66 for
  VTI), divided by rho and square-rooted; the difference is
  interpretation.
- **Hornby et al. (1989) Stoneley reflection-coefficient fracture-
  aperture inversion.** Quantitative complement to the existing
  Stoneley indicators. New `stoneley_reflection_coefficient(A_inc,
  A_refl)` builds `|R|` from incident / reflected pulse
  amplitudes; new `hornby_fracture_aperture(R, frequency_hz,
  V_T, ...)` inverts the rigid-frame, low-frequency, single-
  fracture closed form
  `|R(omega)| = omega L_0 / sqrt(V_T^2 + omega^2 L_0^2)` for the
  fracture aperture L_0 (m). Saturates at +inf for `|R| -> 1`; an
  optional small-amplitude approximation `L_0 ~ V_T |R| / omega`
  is < 5 % off for |R| <= 0.3.
- **Stoneley amplitude fracture indicator**
  (`fwap.rockphysics.stoneley_amplitude_fracture_indicator`).
  Companion to the existing slowness-shift permeability indicator.
  Returns `1 - A_obs / A_ref` -- the fractional Stoneley amplitude
  deficit relative to a tight reference. Detects the same
  fractures / permeable zones via the loss of acoustic energy
  rather than via the dynamic-poroelastic delay; complementary
  noise characteristics, so a coincidence flag is more robust than
  either indicator alone.
- **Dispersive STC for the pseudo-Rayleigh / guided trapped mode**
  (`fwap.dispersion.dispersive_pseudo_rayleigh_stc`). Direct
  pseudo-Rayleigh analogue of `dispersive_stc`: scans formation
  shear slowness, applies the per-frequency phase-slowness
  correction from `pseudo_rayleigh_dispersion`, returns an
  `STCResult` whose slowness axis is the formation `1 / V_S`.
  Removes the high-frequency bias that plain STC produces on
  guided arrivals. Enforces the fast-formation existence
  constraint (`shear_slowness_range[1] < 1 / v_fluid`).
- **Geomechanics indices on top of `ElasticModuli`** (new module
  `fwap.geomechanics`). Closes the gap between the elastic-moduli
  output of `fwap.rockphysics.elastic_moduli` and the
  Workflow-3 deliverables Mari et al. (1994) Part 3 lists --
  *sanding prediction* and *hydraulic-fracture design*. Public
  surface: `brittleness_index_rickman(E, nu, ...)` (Rickman et al.
  2008 BI in `[0, 1]`); `fracability_index(...)` (alias of BI for
  HF-design call sites); `closure_stress(nu, sigma_v, P_p,
  alpha)` (Eaton 1969 uniaxial-strain closure stress, with
  validation rejecting both `nu >= 1` singularity and `nu < 0`
  auxetic regime); `unconfined_compressive_strength(E,
  model='lacy_sandstone')` (Lacy 1997 / Chang et al. 2006);
  `sand_stability_indicator(mu, threshold)` (Bratli & Risnes 1981
  / 5 GPa shear-modulus rule); `overburden_stress(z, rho)`
  (trapezoidal density-log integration). One-call wrapper
  `geomechanics_indices(moduli, ...)` returns a
  `GeomechanicsIndices` dataclass with `brittleness`,
  `fracability`, `ucs`, `sand_stability` and (when
  `sigma_v_pa` is supplied) `closure_stress`. Module-level
  constants `RICKMAN_E_MIN_PA`, `RICKMAN_E_MAX_PA`,
  `RICKMAN_NU_MIN`, `RICKMAN_NU_MAX`,
  `SAND_STABILITY_SHEAR_THRESHOLD_PA` expose the published
  defaults. Six new LAS / DLIS mnemonics in `_FWAP_UNITS`:
  `BRIT`, `FRAC`, `UCS`, `SH`, `SV`, `SAND`.
- **Thomsen-gamma from combined dipole + Stoneley sonic logs**
  (`fwap.anisotropy`). VTI shear-anisotropy parameter
  `gamma = (C_66 - C_44) / (2 C_44)` from two complementary
  measurements: the dipole shear log gives `C_44 = rho * V_Sv^2`,
  and the Stoneley low-frequency tube-wave inversion (White 1983
  / Norris 1990) gives `C_66 = rho_f / (S_ST^2 - 1/V_f^2)`. New
  `stoneley_horizontal_shear_modulus(s_ST, rho_fluid, v_fluid)`,
  `thomsen_gamma(c44, c66)`, and a one-call
  `thomsen_gamma_from_logs(s_dipole, s_stoneley, rho, ...)`
  returning a `ThomsenGammaResult` with `c44`, `c66`, `gamma`.
  New LAS mnemonics: `C44`, `C66`, `GAMMA`.
- **Picker -> log-curve bridge**
  (`fwap.picker.track_to_log_curves`). Converts a
  `Sequence[DepthPicks]` from `track_modes` / `viterbi_pick` /
  `viterbi_pick_joint` into a `(depths, curves)` tuple where
  `curves` is a `{mnemonic: ndarray}` dict suitable to pass
  straight to `write_las` / `write_dlis`. Standard fwap mnemonics
  (`DTP`, `DTS`, `DTST`, `DTPR` / `COHP`, `COHS`, `COHST`,
  `COHPR` / `AMP*` / `TIM*` / `VPVS`); slowness in us/ft;
  missing picks become NaN by default with an optional numeric
  sentinel via `null_value`. `_FWAP_UNITS` extended to carry the
  new mnemonics.

### Fixed
- **WLS in attenuation Q estimators** (`centroid_frequency_shift_Q`
  and `spectral_ratio_Q`): the previous implementations multiplied
  both `A` and `y` by `W = diag(w)` before passing to `lstsq`,
  which makes the solver minimise `sum(w_i^2 * r_i^2)` instead of
  the documented `sum(w_i * r_i^2)`. Switched to `sqrt(W) @ A`,
  `sqrt(W) @ y` so the per-trace weights match the docstring intent
  ("weights = total power") and the residual variance / standard
  error formulas now match the system actually being solved.
- **`read_las` depth-curve detection**: skipped the depth curve
  via mnemonic equality with `las.curves[0].mnemonic`, which was
  fragile when a non-depth curve happened to share the depth
  mnemonic. Now skips by index instead (always first).
- **`viterbi_pick` doc bug**: the comment claimed the
  no-previous-mode sentinel was `+inf`; the code (correctly) uses
  `-inf`. Comment fixed.
- **`anisotropy_strength` docstring**: claimed the metric reaches 1
  for orthogonal waveforms, but actually reaches `1 / sqrt(2)` for
  orthogonal equal-energy waveforms and only saturates at 1 as
  `s -> -f`. Updated formula and verbal description; the metric
  itself was correct.

### Changed
- **`closure_stress` validation tightened** to reject negative
  Poisson's ratio. The Eaton uniaxial-strain model is calibrated
  for the positive-Poisson regime of typical sedimentary rocks;
  negative inputs would produce negative effective horizontal
  stresses. Auxetic materials are out of scope.
- **`classify_flexural_anisotropy`**: band-overlap guard tightened
  from `>` to `>=` (touching bands now rejected); crossover-
  frequency search restricted to the bracket
  `[f_low_band[0], f_high_band[1]]` so a spurious noise zero-
  crossing outside the bands is not reported as the band-to-band
  transition.
- **`track_to_log_curves`**: float-coerce `null_value` at function
  entry (`null_value = float(null_value)`) so passing `None` or
  any non-float type raises `TypeError` cleanly instead of
  slipping through the NaN check and producing object-dtype
  curves.
- **`dispersive_stc`**: rename misleading internal variable
  `tau_of_f` -> `s_of_f`. The variable holds slowness, not a time
  τ; no behaviour change.
- **`sand_stability_indicator`**: docstring now explicit that the
  `mu == threshold_pa` boundary is treated as stable.

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

### Changed
- **Roadmap brought back in line with the tree** (`docs/roadmap.md`). Several
  status claims had gone stale as work landed:
  - Section A (cylindrical-Biot solver) still said the leaky-mode extension was
    "what remains" long after leaky n=0/n=1/n=2, quadrupole, layered/cased-hole
    and VTI solvers had all shipped. It now records the family as essentially
    complete and names the two things genuinely left: extending the validation
    notebook to the full set of published reference figures, and improving the
    cased-flexural bracketing (which is what currently forces the cased dataset
    to stay single-mode).
  - Section E said `ruff-check` was not yet hooked pending a lint-debt cleanup;
    both have since happened.
  - A new **section G** documents the `sonic_ml` layer, which did not exist when
    the roadmap was written -- what shipped across M0-M5f, the isolation
    guarantees, the two headline results *with the identifiability gap between
    them stated*, and its four open follow-ons.
  - Section F (real-data fixtures) is flagged as the highest-value open item,
    because every quantitative claim in the repo -- `sonic_ml`'s included -- is
    currently measured against the same forward model that generated the data.
  A new summary table at the top gives the real remaining scope, and a note
  records that status is checked against the tree rather than from memory: the
  stale section A is exactly the failure mode it exists to prevent.

### Added
- **Real-data integration tests, fetch-on-demand** (roadmap item F, partially
  closed). The suite otherwise runs entirely on synthetics with planted answers,
  which is what makes it assertable and also what bounds it: a synthetic file is
  produced by the same assumptions the reader holds, so it cannot catch a
  convention the reader failed to anticipate. `tests/test_real_data.py` closes
  that gap using files written by *other* software, and
  `scripts/fetch_real_data.py` holds the registry -- URL, SHA-256, provenance
  and licence per entry, with `--list` / `--fetch` / `--verify`.
  Two files are registered: a real Kansas Geological Survey well log (AMOCO
  Collingwood 1-28; a wrapped LAS with 26 service-company mnemonics and vendor
  comment blocks, none of which `write_las` emits) and a SEG-Y written by
  `segyio` (so a reader/writer disagreement about header layout or sample format
  cannot hide behind a round-trip through our own writer -- there is a test that
  reads the foreign file, writes it with fwap, and reads it back).
  **Nothing is vendored**, deliberately: the files are published by third
  parties under their own terms -- the KGS log carries a
  "DATA COPYRIGHT - RILEYS DATASHARE INTERNATIONAL" notice in its own header --
  so they are downloaded into a git-ignored `tests/data/real/` and verified
  against a recorded checksum. A test asserts that directory is git-ignored, so
  the no-redistribution property is enforced rather than intended. Without the
  files the integration tests skip with an actionable message, leaving a normal
  `pytest` run and CI hermetic.
  **Still open, and it is the important half**: neither file is a full-waveform
  sonic gather, because no openly redistributable one is known to exist. The
  sonic processing chain -- and every quantitative claim built on it, including
  `sonic_ml`'s -- remains validated only against the same forward model that
  generated its data. Adding one is a one-entry change to the registry.

### Added
- **`sonic_ml` surrogate-in-the-loop inversion** (roadmap G.4). The M2 forward
  surrogate has always been differentiable; nothing used that, which was the
  original motivation for building it. ``sonic_ml.models.inversion`` closes the
  loop: ``invert_with_surrogate`` optimises formation parameters through the
  surrogate by gradient descent in standardised parameter space, multi-start to
  survive a non-convex landscape, with ``NaN`` frequencies excluded from the
  misfit rather than imputed. ``InversionResult`` carries the recovered values,
  the parameters actually solved for, and the **final data misfit** -- which is
  the number to check before trusting any recovered value.
  **The control ships with the method.** ``invert_with_solver`` runs the same
  inversion through fwap's real modal solver, with the same multi-start budget
  and the same bounds, because "inversion through a learned forward model works"
  means nothing without the exact-forward-model number beside it.
  ``inversion_mae`` and ``no_skill_mae`` score against the training-mean
  reference, so a parameter that was not recovered cannot look as though it was.
  **Measured (400-sample open-hole dataset, held-out, four active parameters):**
  the surrogate route is ~10x faster per sample (~0.36 s vs ~3.5 s) and *less*
  accurate on three of four parameters -- vp 182 vs 236 m/s (surrogate better),
  vs 39 vs 31 m/s, rho 91 vs 3.4 kg/m3, a 0.0038 vs 0.0010 m. The trade is speed
  for accuracy, not a free win. An earlier draft claimed the opposite because
  its control was given one start and a smaller budget than the surrogate got;
  the fair control reversed the conclusion.
  The error pattern turns out to be predictable, and the module docstring
  records it: a surrogate cannot resolve a parameter more finely than its own
  forward error allows, so expected error is roughly
  ``(forward error / parameter signature) x parameter range``. At a measured
  1.5 us/ft forward error that predicts 191 m/s for vp (observed 182),
  103 kg/m3 for rho (observed 91) and 0.0035 m for a (observed 0.0038) --
  within ~12% each. Density has the weakest curve signature of the four and is
  therefore the first casualty of surrogate error; fixing it needs a better
  surrogate, not a better optimiser.
  Tests pin mechanism rather than accuracy (the CI fixture trains on ~32
  samples): that optimisation actually reduces the misfit, that extra starts
  never worsen it, that both routes solve for the same parameters, that results
  are reproducible, and that the answer changes when the surrogate's weights do
  -- so it cannot be coming from the normalisers instead of the learned model.

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
