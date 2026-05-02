# Plan: cylindrical-Biot solver completions

Splits Theme 1 of [`docs/possible_extensions.md`](../possible_extensions.md)
into nine subthemes, each scoped so a single Claude task can land it.
Ordered by dependency and increasing difficulty; A-C are the
leaky-mode finish of `fwap.cylindrical_solver`, D-E add the n=2
azimuthal order, F-G add radial layering, H adds anisotropy, and I
is a cross-cutting validation deliverable.

**Status snapshot.** A, B, C, D, E, F, G (n=0 Stoneley), G'
(n=1 cased-hole flexural), G'' (n=2 cased-hole quadrupole),
H — ✅ DONE. **I (validation notebook) — partial:** the
notebook scaffolding ships at
[`docs/notebooks/cylindrical_biot_validation.ipynb`](../notebooks/cylindrical_biot_validation.ipynb)
with five sections producing the fwap dispersion curves, plus
TODO overlay cells that auto-render once digitised reference
CSVs land in `docs/notebooks/_data/`. The 5 % RMS-deviation
gate activates only after the first CSV.

## Existing scaffolding to reuse

`fwap.cylindrical_solver` already ships, as private helpers:

- `_modal_determinant_n0_complex(kz, omega, ..., leaky_p, leaky_s)`
  — n=0 modal matrix that accepts complex `k_z` and per-wave leaky
  flags (K-Bessel ↔ Hankel selector via `_k_or_hankel`).
- `_detect_leaky_branches(kz, omega, vp, vs, vf)` — sign-of-`Re(α²)`
  classifier that returns `(leaky_F, leaky_p, leaky_s)`.
- `_track_complex_root(det_fn, kz_start)` — Powell hybrid 2-D root
  finder over `(Re kz, Im kz)`.
- `_march_complex_dispersion(det_fn, freq_grid, kz_start)` —
  frequency marcher with constant-slowness extrapolation between
  steps.

The bound-mode public APIs `stoneley_dispersion` (n=0) and
`flexural_dispersion` (n=1) are shipped and validated. The
`BoreholeMode` return type already has an `attenuation_per_meter`
field for leaky-mode use.

That means subthemes A-C were a public-API + branch-tracking job
on top of existing private helpers, *not* a from-scratch solver.
All three have shipped (see status entries below); this section
is retained for historical context.

---

## A. Pseudo-Rayleigh leaky-mode dispersion (n=0) — ✅ DONE

**Status.** Shipped in `d79bd69`. Public
`pseudo_rayleigh_dispersion` exported from `fwap`. Cutoff
regression and Paillet & Cheng 1991 fig 4.5 match validated in
`tests/test_cylindrical_solver.py`.



**Why tractable.** All four pieces of the n=0 complex pipeline
exist as private helpers. The work is wiring a public function and
proving the answer matches Paillet & Cheng 1991 fig 4.5.

**What to build.**

```python
fwap.cylindrical_solver.pseudo_rayleigh_dispersion(
    vp, vs, rho, vf, rho_f, a_borehole, frequency_hz,
) -> BoreholeMode
```

- Detect the cutoff `f_c` where `1/V_S < 1/V_f` (mode exists only
  for fast formations, `V_S > V_f`).
- For each `f > f_c`, seed `_march_complex_dispersion` from the
  high-frequency limit `k_z ≈ ω / V_S - i·ε` (Paillet-Cheng
  asymptote, small positive imaginary attenuation).
- March downward in frequency to the cutoff; stop when
  `_track_complex_root` fails or `Im(kz)` crosses zero.
- Populate `BoreholeMode.slowness = Re(kz)/ω` and
  `attenuation_per_meter = Im(kz)`.

**Validation.**

1. Pure-bound limit check: at `f → ∞`, slowness → `1/V_S` to
   plotting accuracy.
2. Reproduce Paillet & Cheng 1991 fig 4.5 (limestone formation:
   `V_P=5.5 km/s`, `V_S=3.1 km/s`, `V_f=1.5 km/s`, `a=0.1 m`)
   within ~2% of the published curve.
3. Cutoff regression: cutoff frequency matches the closed form
   `f_c = (j_{1,1} V_f V_S) / (2π a sqrt(V_S² − V_f²))`.

**Scope.** ~120 lines of solver code + 6-8 tests. One focused day.

---

## B. Leaky flexural mode (n=1) in fast formations — ✅ DONE

**Status.** Shipped in `6e76fae`. `_modal_determinant_n1_complex`
plus auto-dispatch in `flexural_dispersion`: when `V_S > V_f` the
public API routes to `_flexural_dispersion_fast_formation`, which
brentq's `Im(_modal_determinant_n1_complex)` along the real-`k_z`
axis. Slow-formation regression bit-identical to the existing
bound-mode answer.



**Why tractable.** Mirror of subtheme A but at azimuthal order 1.
The real-valued `_modal_determinant_n1` exists; the work is
adding a complex-aware twin and a public marcher.

**What to build.**

1. `_modal_determinant_n1_complex(kz, omega, ..., leaky_p,
   leaky_s)` — same row/column structure as the real version
   (4×4 in n=1 conventions), with `_k_or_hankel` substituted for
   the bound `K_n` / `K_{n+1}` evaluations.
2. Public `flexural_dispersion(..., regime="auto")` extension:
   when `V_S > V_f` and the bound solver fails, fall back to the
   complex marcher seeded by the Scholte-speed asymptote.
   (Today the public API hardcodes the slow-formation regime.)

**Validation.**

1. Schmitt 1988 fig 4: dispersion curve with the leaky bend just
   above the geometric cutoff.
2. Slow-formation regression: when `V_S < V_f`, the new code path
   is bypassed and the existing bound-mode solver answer is
   bit-identical.

**Scope.** ~200 lines (the n=1 matrix is bigger than n=0) + 6
tests. Two days.

**Depends on.** A (proves the complex pipeline; reuses the same
marcher and seed strategy).

---

## C. Cutoff handling + branch tracker — ✅ DONE

**Status.** Shipped in `b3334c2`. `BranchSegment` dataclass plus
`_march_complex_dispersion_validated` route the marcher through
cutoff and branch-flip events without operator intervention.



**Why tractable.** The marcher today fails silently at cutoffs
(the "cannot continue without a fresh seed" branch in
`_march_complex_dispersion`). The fix is a small per-step
classifier.

**What to build.**

- `_classify_step_failure(kz_prev, kz_attempt, omega, ...)` that
  decides between "hit cutoff" (return NaN, *continue* the
  marcher with a re-seed from the bound-mode side), "branch
  flipped" (re-detect leaky flags via `_detect_leaky_branches`
  and retry), and "genuine convergence failure" (NaN out).
- A small `BranchSegment` dataclass returned by `_march_*` so the
  public APIs can splice multiple branches into one
  `BoreholeMode.slowness` array with NaN gaps at the transition.

**Validation.**

- Synthetic two-branch test: stitched bound + leaky n=0 curve
  matches an analytic concatenation to within tolerance.
- Regression: the same fast-formation pseudo-Rayleigh curve from
  subtheme A runs continuously across the cutoff without operator
  intervention.

**Scope.** ~100 lines + 5 tests. One day.

**Depends on.** A (the failure modes only manifest once a leaky
public API exists).

---

## D. Quadrupole bound-mode dispersion (n=2) — ✅ DONE

**Status.** Shipped in `c383f50`. `_modal_determinant_n2` plus
public `quadrupole_dispersion` exported from `fwap`. Tang &
Cheng 2004 fig 3.7 LWD-slow-formation regression validated.



**Why tractable.** Same Helmholtz-decomposition machinery as n=0
and n=1, only the matrix entries change. Tang & Cheng 2004
sect. 2.5 lists the n=2 modal determinant explicitly. Replaces
the phenomenological `lwd_quadrupole_priors` already in
`fwap.lwd`.

**What to build.**

1. `_modal_determinant_n2(kz, omega, vp, vs, rho, vf, rho_f, a)`
   — real-valued 4×4 (slow-formation regime; bound only).
2. Public `quadrupole_dispersion(...)` returning `BoreholeMode`,
   structured as the n=0/n=1 sisters.
3. A short note in `fwap.lwd` pointing the prior consumer at the
   real solver.

**Validation.**

- Tang & Cheng 2004 fig 3.7 LWD slow formation: low-frequency
  limit `s → 1/V_S`; geometric cutoff at the Scholte-speed
  intercept.
- Closed-form check at the long-wavelength limit
  (`ω a / V_S → 0` ⇒ `s = 1/V_S`).

**Scope.** ~180 lines + 6 tests + one dispersion plot in the
demo. Two days.

**Depends on.** Independent of A-C (bound regime).

---

## E. Quadrupole leaky-mode (n=2, fast formations) — ✅ DONE

**Status.** Shipped in `338684f`. `_modal_determinant_n2_complex`
plus auto-regime dispatch in `quadrupole_dispersion`. Fast-
formation Tang & Cheng 2004 fig 3.10 regression validated;
slow-formation bit-equivalence with the n=2 bound solver
preserved.



**Why tractable.** Same upgrade as B applied to D: lift the n=2
real determinant to complex with leaky flags, reuse
`_track_complex_root` / `_march_complex_dispersion`.

**What to build.**

1. `_modal_determinant_n2_complex(...)` (the real version's twin
   with `_k_or_hankel`).
2. Auto-regime dispatch in the public `quadrupole_dispersion`.

**Validation.**

- Fast-formation quadrupole regression against Tang & Cheng 2004
  fig 3.10.
- Slow-formation bit-equivalence with subtheme D output.

**Scope.** ~150 lines + 5 tests. One day.

**Depends on.** A, B, C (proven complex pipeline) and D (n=2
matrix).

---

## F. Single-extra-layer extension (mudcake or altered zone) — ✅ DONE

**Status.** Shipped across PRs #43, #45, #48, #49 (and the F.3
docs PR landing this update). Detailed sub-plans:
[`cylindrical_biot_F.md`](cylindrical_biot_F.md) for F.1 (n=0
layered Stoneley) and
[`cylindrical_biot_F_2.md`](cylindrical_biot_F_2.md) for F.2 (n=1
layered flexural).

**Why tractable.** Adds one annular region between fluid and
formation (`fluid → mudcake → formation`). At n=0 the
elastodynamic structure stays single-block (7x7 modal matrix
combining 3 r=a BCs + 4 r=b BCs and 1+4+2 amplitudes). At n=1 the
matrix is dense 10x10 -- the d_θ operations cross-couple every
amplitude family into BCs of either azimuthal sector (cf. F.2.a
errata: an earlier draft incorrectly claimed cos/sin block
decoupling).

**What was built.**

1. `_modal_determinant_n0_layered` -- 7x7 layered Stoneley
   determinant (PR #45; F.1.a math scaffolding + F.1.b.{1,2,3,4}
   per-row builders + assembly).
2. `_modal_determinant_n1_layered` -- 10x10 layered flexural
   determinant (PR #48; F.2.a + F.2.b.{1-7} cos-sector + F.2.c.{1-3}
   sin-sector + F.2.d assembly).
3. Public `stoneley_dispersion_layered(..., layers=...)` and
   `flexural_dispersion_layered(..., layers=...)`. Both reuse the
   `BoreholeLayer` dataclass for parameter packaging. Single-
   element layer stacks supported; multi-layer raises
   `NotImplementedError` pointing at plan item G.
4. Slow-formation regime constraint for n=1 (`layer.vs >= vs`)
   documented; fast-formation layered flexural is future work.

**Validation actually achieved.**

- Layer=formation regression for both n=0 and n=1 to `rtol=1e-8`
  across the test frequency grids -- the floating-point oracle
  for the entire row-builder chain.
- Thickness → 0 limit verified (n=0 and n=1 both recover the
  unlayered answer continuously).
- Thickness → ∞ limit verified for n=0 (slowness approaches
  unlayered with layer-as-formation). Not tractable for n=1 in
  the bound regime; documented as omission in F.2.e.
- Determinant vanishes at converged root for any non-trivial
  layer (`|det_at_root| < |det_off_root| × 1e-6`).
- Multi-frequency monotonicity smoke across 100 Hz - 20 kHz
  (n=0) and 3-15 kHz (n=1).
- Headline physics: softer mudcake slows Stoneley by ~0.6-0.8 %
  (n=0); harder layer speeds up flexural by ~1-1.3 % (n=1) at
  typical frequencies.
- ~115 plan-F-specific tests (51 for F.1, 67 for F.2 incl.
  hardening) on top of the 86 pre-existing.

**Scope (actual).** ~3500 lines of solver code + ~3000 lines of
tests, distributed across ~25 mergeable commits in PRs #43, #45,
#48, #49. Significantly larger than the original ~250-line / 8-test
estimate, primarily because (a) the n=1 layered determinant is
genuinely 10x10 dense (not block-diagonal as initially hoped --
F.2.a.6 erratum), and (b) per-row builders with isolated tests
gave ~115 tests instead of ~8.

**Schmitt 1988 fig 6 quantitative match.** Deferred to plan
item I (validation notebook), where the digitised reference
data lives.

**Depends on.** Independent of A-E (bound-mode only;
generalisation in radius, not in `kz`).

---

## G. Cased-hole multi-layer extension (propagator matrix) — ✅ DONE (n=0, n=1)

**Status.** n=0 shipped via plan
[`cylindrical_biot_G.md`](cylindrical_biot_G.md), G.0 through
G.f. n=1 shipped via plan
[`cylindrical_biot_G_prime.md`](cylindrical_biot_G_prime.md),
G'.0 through G'.f. Both cased-hole solvers are wired through
the existing `*_dispersion_layered` public APIs: passing a
multi-layer `layers=(casing, cement, ...)` tuple now dispatches
to the Thomson-Haskell propagator-matrix path
(`_modal_determinant_n0_cased` for Stoneley,
`_modal_determinant_n1_cased` for flexural); single-layer and
unlayered paths remain bit-unchanged. Validation covers two-
formation-layers collapse to unlayered (`rtol=1e-6`), thin-
trivial-layer collapse to N=1, layer-permutation distinctness,
multi-frequency det-at-root self-consistency, and cement-
stiffness physics oracles for both Stoneley and flexural. The
n=1 path additionally enforces the slow-formation per-layer
constraint (`layer.vs >= vs`) at validation time.

**Deferred follow-ups** (separate plans, not yet scheduled):

- **Tang & Cheng 2004 fig 7.1 digitised reproduction**:
  flagged in the G.e / G'.e / G''.e plans; deferred to
  per-figure CSV ingestion work (one figure per landed
  solver, n=0 / n=1 / n=2).
- **Fast-formation cased-hole quadrupole** (deferred from
  G''): the n=2 quadrupole has both bound and leaky variants
  in the unlayered case (E auto-dispatch); the cased-hole
  leaky variant is a separate plan analogous to the future
  fast-formation layered flexural follow-up to F.2.
- **Knopoff / Kennett delta-matrix** for thick-layer / very-
  high-frequency conditioning: only needed if `kz · thickness >
  30` regime emerges.
- **Bracket-helper refinement** for cased-hole flexural in
  slow-formation, thick-cement geometries, where the brentq-
  expansion loop in `flexural_dispersion_layered` may converge
  to a tube-wave-like mode at very large slowness rather than
  the perturbed formation flexural. Documented in the G'.e
  cement-stiffness test docstring.

---

## H. VTI formation (transversely isotropic, vertical symmetry axis) — ✅ DONE

**Why tractable.** The five-parameter TI stiffness tensor (C11,
C13, C33, C44, C66) drops in cleanly: the borehole wall normal is
horizontal, so the symmetry axis aligns with `z` and the modal
determinant decouples into the same n=0/n=1/n=2 azimuthal
orders. The `flexural_dispersion_vti_physical` phenomenological
model in `fwap.cylindrical` already shows the expected
qualitative behaviour and can be used as a sanity prior.

**What to build.**

1. `_radial_wavenumbers_vti(kz, omega, c11, c13, c33, c44, c66,
   rho, mode)` — replaces the isotropic
   `(p² = kz² − ω²/V_P²; s² = kz² − ω²/V_S²)` formulae with the
   Christoffel-equation roots in TI media (qP and qSV in n=0,1;
   qSH for n=2 quadrupole).
2. `_modal_determinant_n{0,1}_vti(...)` mirroring the isotropic
   versions with C-matrix entries instead of Lamé λ, μ.
3. Public `stoneley_dispersion_vti` and `flexural_dispersion_vti`
   with the same `BoreholeMode` return type.

**Validation.**

- Isotropic-collapse regression: feed `C11 = C33 = λ + 2μ`,
  `C44 = C66 = μ`, `C13 = λ` and recover the isotropic output to
  floating-point precision.
- Schmitt 1989 fig 5: Thomsen-γ-induced flexural splitting.
- Norris 1990 closed-form Stoneley low-frequency limit
  `S_ST² = 1/V_f² + ρ_f / C66`.

**Scope.** ~350 lines + 10 tests. One week. The hardest pieces
are the qP/qSV root selection in the radial-wavenumber step and
clean isotropic-collapse equality.

**Depends on.** Independent of A-G structurally; easiest to land
after F because the matrix-bookkeeping practice transfers.

---

## I. Validation notebook against published dispersion figures — ⚠ partial

**Status.** Notebook scaffolding (option B) shipped at
[`docs/notebooks/cylindrical_biot_validation.ipynb`](../notebooks/cylindrical_biot_validation.ipynb)
with one section per planned reference figure, each producing
the fwap dispersion curve. Digitised overlays remain `TODO` --
the per-figure CSVs in `docs/notebooks/_data/` are the
remaining deliverable.

**Why tractable.** Standalone deliverable; once any of A, B, D
ship, the notebook compares the fwap output to the digitised
reference curve.

**What to build.**

`docs/notebooks/cylindrical_biot_validation.ipynb` — produces a
two-row figure per published reference:

- top row: fwap dispersion curve;
- bottom row: same curve overlaid on a digitised version of the
  published figure (digitised data shipped in
  `docs/notebooks/_data/`).

References to reproduce, in order of effort:

1. Paillet & Cheng 1991 fig 4.5 — Stoneley + pseudo-Rayleigh on
   limestone (covers A and the existing `stoneley_dispersion`).
2. Schmitt 1988 fig 4 — flexural in slow + fast formations
   (covers B and `flexural_dispersion`).
3. Tang & Cheng 2004 fig 3.7, 3.10 — quadrupole slow + fast
   (covers D and E).
4. Tang & Cheng 2004 fig 7.1 — cased-hole Stoneley (covers G).
5. Schmitt 1989 fig 5 — VTI flexural splitting (covers H).

**Validation.** The notebook itself is the validation; an `nbval`
pytest hook fails if any cell errors or a per-curve RMS deviation
exceeds 5% of the published value.

**Scope.** ~200 cells + ~20 KB of digitised reference points.
Three days to assemble once the underlying solvers ship.

**Depends on.** Whichever solvers it is validating; can ship
incrementally per reference figure.

---

## Suggested order

The shortest viable path to a public leaky-mode product was
A → C → B (about a week, shipped). The shortest path to LWD-
grade processing was A → C → B → D → E (about two weeks,
shipped). Layered and anisotropic work (F, G n=0, G' n=1,
G'' n=2, H) shipped independently of the leaky-mode chain.
Remaining items:

- **I** — validation notebook against published dispersion
  figures. Notebook scaffolding (option B) shipped at
  `docs/notebooks/cylindrical_biot_validation.ipynb` with the
  five planned sections producing fwap curves; digitising the
  reference figures into `docs/notebooks/_data/*.csv` (and
  flipping each section's TODO cell into a real overlay) is the
  outstanding deliverable.
- **Fast-formation cased-hole quadrupole** (deferred follow-up
  to G''). The unlayered n=2 already auto-dispatches between
  the bound and leaky variants; the cased-hole leaky variant
  needs the same complex-determinant treatment that B / D
  applied to the unlayered solvers.
