# What the oracles taught us

A retrospective on the analytic cross-checks added to `fwap` between PRs #50
and #64, written to change how the next batch of work is chosen. It is not a
status file — `docs/roadmap.md` is that, and `plans/roadmap_1.md` is the
priority snapshot. This one is about *method*.

## The short version

Five analytic oracles have been built. **Every one found something, and four
of the five overturned a written claim rather than confirming it.** That hit
rate is far better than the alternative that kept being proposed in planning
notes — "more tests against existing behaviour" — and it held even in an
environment with no access to published figures or field data.

| Oracle | PR | What it did |
|---|---|---|
| Scholte speed vs cylindrical Stoneley high-f limit | #59 | Confirmed the solver; first literature tie in the repo. Also **overturned** the planning claim that only tests and documentation remained |
| Rigid-pipe pseudo-Rayleigh cutoff | #61 | **Corrected** docstring advice that would have discarded valid data |
| Quadrupole high-frequency asymptote | #62 | Confirmed slow formations; **exposed** a fast-formation defect |
| Ray radiation estimate vs leaky attenuation | #63 | Confirmed scale and geometry; **exposed** window-dependent branch selection, fixed in the follow-up |
| White tube-wave speed vs Stoneley `f -> 0` limit | #64 | Confirmed the limit to ~1e-7; **exposed** an undocumented validity floor on the repo's slow-formation `V_S` estimator |

`fwap.validation` (#50) is deliberately not in that table. It is
the *machinery* for scoring against digitised figures, not an oracle: with no
reference CSV shipped it has found nothing and cannot, and the notebook says so
rather than letting green plots imply otherwise. Counting it among the wins
would be the same overclaim this document is about.

## What makes something an oracle rather than another test

The distinction that matters is **independent formulation**, not independent
code. A second implementation of the same equations catches typos. A different
equation for the same physics catches errors in the model.

The ones that worked share a shape:

- **`scholte_speed`** is a *plane*-interface secular equation. No Bessel
  functions, no borehole radius. The cylindrical solver must approach it at
  short wavelength because the wall looks flat there.
- **The rigid-pipe cutoff** is a geometric resonance condition with the
  formation removed entirely.
- **The quadrupole asymptote** reuses the Scholte oracle at a different
  azimuthal order — the wall looks flat to *every* order, so one oracle covers
  n=0 and n=2.
- **The ray radiation estimate** is a plane-wave reflection coefficient times a
  bounce rate. It reaches the imaginary part of the answer, which nothing else
  did.
- **The White tube-wave speed** is the opposite end of the same curve Scholte
  pins: a quasi-static long-wavelength limit in which the borehole radius drops
  out of the algebra entirely. That absence is the sharper test — the radius is
  the one parameter the solver has and the formula does not.

Four generative questions, in the order they have actually paid off:

1. **Is there a degenerate limit where the geometry disappears?** Flat wall,
   rigid wall, light fluid, zero contrast. Three of the five came from here.
2. **Does the same limit apply to a sibling solver?** The Scholte check cost
   almost nothing to point at n=2 and immediately found a defect.
3. **Is there a quantity the existing checks never touch?** Attenuation had
   three tests and none of them looked at its magnitude. Ask which *fields* of
   a return value are unvalidated, not which functions.
4. **Can the oracle itself be validated by a third implementation?**
   `scholte_speed` collapses to the Rayleigh equation in the light-fluid limit
   and reproduces `rayleigh_speed`. Chains like that are worth building.

## The failure modes, all of which happened

These are the specific ways a check looked convincing and was not. Each cost a
redo.

**A limit that cannot discriminate.** The Scholte secular equation passed its
light-fluid → Rayleigh check to 6e-13 *with the sign of the loading term
wrong*, because both signs reduce to Rayleigh. A passing limit test proves
nothing unless the wrong answer would fail it. The sign was fixed instead by
requiring a root to exist below `min(V_s, V_f)`. **Always ask what the check
would do to a wrong implementation** — and where practical, ship that question
as a test: several now assert that a deliberately wrong reference scores
noticeably worse than the right one.

**Measuring against a grid that shares the scaling you are testing.** The
pseudo-Rayleigh cutoff's `1/a` scaling came out perfect to three decimals on
the first attempt, because the frequency grid was derived from the closed-form
estimate — and both scale as `1/a`, so any grid-determined stop returns a
constant ratio for free. Redone on a fixed absolute grid it was 0.363 ± 0.001,
which is the real result. **If the measurement apparatus is built from the
prediction, the agreement is an artifact.**

**Statistics quoted over the wrong population.** "About 15 % of draws are slow"
was measured over the default `FormationPriors`, not the prior the cased
generator actually pins, where the true figure is 0 %. Correcting it changed a
*conclusion*, not just a number: the two-mode dataset turned out to need a
disjoint prior rather than being a subset. **Check which distribution a
percentage was taken over before repeating it.**

**Sampling structure as noise.** The leaky attenuation looked jagged and
grid-dependent on a coarse grid; the curve is smooth and reproducible to 6e-14,
and the jaggedness was undersampling of real resonance structure. The
convincing test was not refinement but varying the grid *endpoints*, which is
what actually exposed the branch-selection bug. **When something looks like
numerical noise, find the parameter it truly depends on before calling it
noise.**

**An "oracle" the code under test already contains.** The White tube-wave
formula looked like a clean independent closed form for the Stoneley
low-frequency limit — until a grep found `s_st_lf = sqrt(1/vf**2 + rho_f/mu)`
inside `_stoneley_kz_bracket`, where it sets the solver's own search bracket.
Agreement would then have been partly the solver confirming itself, and the
claim "independent oracle" was one commit from being written down. The fix was
to take the bracket out of the loop: the test locates the root by scanning 40×
wider than the solver's factor-of-two bracket. **Before calling anything
independent, grep for it in the implementation.** Independence is a property of
the code, not of the derivation.

**A test premise that is simply false.** A test asserted TV regularisation
"prefers a single contact"; true TV is exactly indifferent, and the smoothing
offset tips it the other way. The measurement was fine and the physics
statement was wrong. **Write the assertion after the measurement, not before.**

## Report offsets; do not fit them

The ray radiation estimate sits a stable factor of ~0.6 from the solver across
every geometry and formation tried. Multiplying by 1/0.6 would have made the
agreement look excellent and destroyed the check: an oracle with a fitted
constant in it can no longer disagree. The offset is documented as unexplained
and the test brackets it.

The same discipline applies to the rigid-pipe cutoff, where the 2.8× offset
varies with formation velocity and so could not have been folded in even if one
wanted to. **An oracle's value is exactly its ability to disagree.** Anything
tuned to reduce disagreement should be treated as a fit and labelled one.

Corollary: **state what the check cannot catch.** The attenuation oracle would
catch a wrong power of frequency or a radius/diameter confusion, and would not
catch a 30 % error. Saying so keeps a green test from being read as a precision
guarantee.

## How this should change planning

**Claims about absence age worst.** `plans/roadmap_1.md` has now been wrong
four times, and every time the error was a statement that something did not
exist or could not be done here: no open sonic gather exists (false, two were
found); Utah FORGE is mirrored on AWS (false); egress reaches GitHub only
(false); only tests and documentation remain (false, three times over). Every
one of these was reasoned rather than measured. **Before writing "there is no
X" or "X is impossible here", spend the ten minutes it takes to check.**

**Budget for the finding, not the confirmation.** Three of five oracles found a
defect. Planning that assumes an oracle will pass and treats it as a
box-ticking exercise will consistently under-budget, because the valuable
outcome is the one that generates follow-up work. Plan an oracle as an
investigation with an open-ended tail.

**Prefer breadth of oracle to depth of test.** Adding a sixth test to a
function that already has five checks the same code against the same
assumptions. Adding one oracle to a function with none can invalidate all five.
When choosing between "more coverage here" and "any independent check there",
take the latter.

**Distinguish "blocked on data" from "blocked on truth".** The environment's
real constraint is *external data* — published figures, field gathers — not
external truth. Closed-form results, asymptotic limits, conservation arguments
and independent formulations of the same physics are all reachable from theory
alone. Conflating the two produced the repeated "nothing left to do here"
conclusion, which was wrong every time it was written.

**Defects found by an oracle should be pinned as defects, not fixed silently or
left implicit.** The convention now in `tests/test_cylindrical_solver.py` is a
comment block saying the tests pin defects, that a future fix will make them
fail, and that they should then be rewritten rather than worked around. This
keeps a known limitation from quietly becoming a guarantee.

## Candidate oracles not yet attempted

Kept concrete so the next session does not have to re-derive the list. Whether
any of these bites is a measurement, not a promise — which is exactly what the
last such list said, and it went four for four.

Add one screening step before starting any of them, learned the hard way on the
tube-wave check: **grep the implementation for the formula first.** If the
solver already uses it — as a bracket, a seed, an initial guess — the check is
not independent, and the test has to be built to route around that use.

- **Energy balance for the leaky modes.** Radiated power computed from the
  far-field Hankel amplitude, divided by axial energy flux, should reproduce
  `Im(k_z)` — and unlike the ray estimate it has no free geometry in it, so it
  might explain the 0.6 offset rather than merely bracketing it.
- **Reciprocity / symmetry of the layered solver.** Swapping layer order in a
  stack where the physics is symmetric should leave the dispersion invariant.
- **The `n=1` and `n=2` cutoffs against their rigid-pipe forms**, the same
  check #61 ran for `n=0`, using the appropriate Bessel zeros.
- **Attenuation vs the bound-mode limit.** `Im(k_z)` must go to zero
  continuously as a mode approaches its trapping boundary; a discontinuity
  there would indicate a branch error.
