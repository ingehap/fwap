# What the oracles taught us

A retrospective on the analytic cross-checks added to `fwap` between PRs #50
and #64, written to change how the next batch of work is chosen. It is not a
status file — `docs/roadmap.md` is that, and `plans/roadmap_1.md` is the
priority snapshot. This one is about *method*.

## The short version

Five analytic oracles have been built. **Every one found something. Four of the
five exposed a defect in code or documentation; the fifth overturned a claim in
the plans.** That hit rate is far better than the alternative that kept being
proposed in planning notes — "more tests against existing behaviour" — and it
held in an environment with no access to published figures or field data.

A sixth candidate — the leaky-mode energy balance — was attempted and turned
out to be **vacuous**, reproducing the right answer for wrong inputs. It is
written up below rather than quietly dropped, because the reason it fails
generalises further than any of the successes.

Worth separating the two kinds of finding, because they are budgeted
differently. A *code* defect (#62, #63) generates follow-up work and may not be
fixable in the same sitting. A *documentation* defect (#61, #64) is usually
cheap to fix and dangerous to leave: both were advice that would have led a
careful user to a wrong answer — one telling them to discard valid data, the
other silently permitting an inversion outside its own validity range.

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

**A conservation law closed over the wrong region.** The leaky-mode energy
balance reproduced `Im(k_z)` to ratio 1.000 at every frequency — and to ratio
1.0000 for arbitrary `k_z` values that are not roots of anything, because
closing the balance inside the fluid is a mathematical identity. A conservation
law tests something only if the control volume contains the constraint you are
testing. Full treatment below, under *Attempted and withdrawn*.

**A check transplanted across a boundary its mechanism does not cross.** The
most common failure in this programme: two of seven candidates. "Swapping layer
order leaves the dispersion invariant" is true of nothing in a *cylindrical*
stack — the layers sit at different radii, so exchanging them changes the
medium, by about 1 % here; the premise came from plane-layered intuition. And
"check the n=1/n=2 cutoffs against their rigid-pipe forms" carried a check from
one azimuthal order to another without asking whether the *kind of mode*
carried with it — at n=0 it is a fluid-column resonance, at n=1 and n=2 the
solver returns interface modes, and measurement confirms the cutoffs there are
shear-controlled rather than fluid-controlled. **Ask what physical mechanism
makes the check true, then ask whether that mechanism survives the move** —
across geometry, azimuthal order, mode family, or regime.

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
five times, and every time the error was a statement that something did not
exist or could not be done here:

1. no openly redistributable sonic gather exists — false, two were found;
2. Utah FORGE is mirrored on AWS Open Data — false, those buckets carry no
   wireline logs;
3. this sandbox's egress reaches GitHub only — false, the S3 buckets are
   reachable and downloads work;
4. only more tests and documentation remain — false, and written three times;
5. no large piece of work can be carried to completion here unaided — false
   within one revision, when the `n=0` branch-selection defect turned out to
   need no derivation and no literature.

Every one was reasoned rather than measured. **Before writing "there is no X"
or "X is impossible here", spend the ten minutes it takes to check.** Note the
asymmetry: no claim of the form "X exists and works" has had to be withdrawn.
Absence is the failure mode, because it is the claim you can make without
looking.

**Budget for the finding, not the confirmation.** Four of five oracles found a
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

**When the fix lands, re-run the defect's own measurements and keep the
numbers.** The branch-selection defect was documented with three measurements
(a silent 2486 → 2952 m/s switch, 0/60 finite samples, 0/81 finite samples);
the fix reports the same three the other way round (one value for every grid
top, 60/60, 81/81). Carrying the numbers across rather than writing fresh ones
makes the direction of the change visible in the diff, and makes it obvious if
a "fix" only moved the problem. This is cheap and it is the main reason the
defect tests were worth writing in the first place.

## Attempted and withdrawn: energy balance for the leaky modes

This list previously led with it, on the reasoning that radiated power over
axial power must reproduce `Im(k_z)` with no free geometry in it, and so might
*explain* the ~0.6 offset the ray estimate leaves open rather than merely
bracketing it. **It does neither, and the way it fails is more instructive than
another success would have been.**

### The derivation, in full

Kept because the algebra is reusable even though the conclusion is negative —
the same fluxes appear in any energy argument about these modes.

Time convention `exp(i(k_z z - omega t))`. The time-averaged energy flux
(Poynting vector) for an elastic medium is

    I_j = -(1/2) Re(sigma_jk conj(v_k)),   v = -i omega u

and in a fluid, where `sigma_jk = -P delta_jk`, that reduces to
`I_j = -(omega/2) Im(P conj(u_j))`.

The fluid field for the n=0 mode is

    P   = A I0(F r),        F^2 = k_z^2 - (omega/V_f)^2
    u   = grad P / (rho_f omega^2)

so `u_r = A F I1(F r) / (rho_f omega^2)` and
`u_z = i k_z A I0(F r) / (rho_f omega^2)`. Substituting:

    axial flux density   I_z = Re(k_z) |A|^2 |I0(F r)|^2 / (2 rho_f omega)
    axial power          P_z = (pi Re(k_z) |A|^2 / (rho_f omega))
                                 INT_0^a |I0(F r)|^2 r dr
    radial power at r=a  P_r = -(pi a / (rho_f omega)) |A|^2
                                 Im(I0(Fa) conj(F I1(Fa)))

`P_r` needs no formation fields at all: at the wall `sigma_rz = 0` and
`sigma_rr = -P`, and `u_r` is continuous, so the flux through the wall can be
evaluated entirely from the fluid side. The mode's power decays as
`exp(-2 Im(k_z) z)`, so energy balance over the fluid cylinder is

    2 Im(k_z) P_z = P_r

and `|A|^2` cancels, leaving

    Im(k_z) = -a Im(I0(Fa) conj(F I1(Fa))) / (2 Re(k_z) INT_0^a |I0(F r)|^2 r dr)

Measured against the solver it reproduces `Im(k_z)` to ratio 1.000 at every
frequency. That looked like the cleanest confirmation yet — for about ten
minutes, until the question this document already insists on: *what would it do
to a wrong answer?* Fed eight arbitrary complex `k_z` values that are not roots
of anything, it returns their imaginary parts too, to ratio 1.0000.

It is an identity, not a check. Closing the balance inside the fluid is just the
divergence theorem applied to a source-free Helmholtz solution, true of any
field `A I0(Fr) exp(i k_z z)` with `F^2 = k_z^2 - (omega/V_f)^2`. No property of
the formation enters, so nothing about the eigenvalue condition is being tested.

The obvious repair — extend the balance into the formation, which would bring
the outgoing-wave condition in — is not available either. The leaky-S field
*grows* with radius (the standard leaky-mode divergence: 0.996 at `r = 0.1` m to
1.6e86 at `r = 30` m, using the solver's own evaluator), so the axial power
integral has no finite value to divide by.

**The lesson, which is new and general: a conservation law only tests something
if the region you close it over contains the constraint you are testing.** Pick
a control volume that excludes the physics determining the answer and the law
becomes a tautology — one that will reproduce the right number perfectly and
tell you nothing. Before building a balance-based check, ask which surface
carries the condition that fixes the eigenvalue, and make sure the control
volume straddles it.

A near-miss worth recording separately: "ratio 1.000 at every frequency" was
one commit from being written up as the strongest confirmation in the
repository. What caught it was mechanically applying failure mode 1 to a result
that looked too good, rather than only to results that look suspicious. The
rule earns its place by being applied when it feels unnecessary.

`tests/test_cylindrical_solver.py` pins all three facts — that the balance
reproduces `Im(k_z)` at roots, that it does so at non-roots too, and that the
formation field grows — so the next attempt starts from the result rather than
from the derivation. Nothing was added to the public API: shipping a check that
cannot fail would be worse than shipping none.

## Which conservation laws are worth trying

Prompted by the energy-balance failure: if energy did not bite, does anything
else? The useful filter is the one that section ends on — a conservation law
tests something only if the control volume contains the constraint that fixes
the eigenvalue. A second filter falls out of the same algebra and is worth
stating separately, because it disqualifies a whole family at once.

**Linear momentum adds nothing.** For a single mode every quadratic flux
carries the same `exp(-2 Im(k_z) z)`, so every balance has the form
`(flux out) = 2 Im(k_z) x (flux carried)` and `Im(k_z)` divides out the same
way. Momentum is worse than merely parallel: for the fluid column the axial
momentum flux density is `rho_f <v_z v_z> = |k_z|^2 |A|^2 |I0|^2 /
(2 rho_f omega^2)`, which is the energy flux density above multiplied by
`|k_z|^2 / (omega Re(k_z))` — a constant across the cross-section. The momentum
balance is therefore the energy balance times a constant, and reduces to the
same identity. (For real `k_z` that factor is `1/c`, the familiar
momentum-equals-energy-over-phase-velocity result.) **The general rule:
a second conserved density built from the same single mode gives an
independent check only if it is not proportional to the first.** The factor above was checked numerically rather
than left as algebra: the ratio of the two integrated fluxes matches
`|k_z|^2 / (omega Re(k_z))` to six digits at every frequency tried.

That rules out most of the obvious list:

- **Angular momentum** — for azimuthal order `n` it is `n/omega` times the
  energy flux. Proportional; no information.
- **Energy flux continuity across an interface** — that *is* the boundary
  condition the determinant already enforces, so it is satisfied by
  construction. Same failure as the fluid-only balance, in a thinner disguise.
- **Momentum flux continuity across an interface** — likewise the traction
  boundary condition.

Two survive the filter and are worth attempting, because each brings in
something the single mode does not already contain:

- **Modal orthogonality.** Two *different* modes at the same frequency satisfy
  a biorthogonality relation (Auld's reciprocity form for waveguides). This
  involves two eigenvectors, so it cannot be satisfied by construction from
  one of them, and it fails if either is wrong. This is the strongest
  remaining conservation-flavoured candidate.
- **Causality (Kramers-Kronig).** `k_z(omega)` for a causal medium ties the
  frequency dependence of `Re(k_z)` to that of `Im(k_z)` non-locally. It
  relates the solver's output *across* frequencies rather than at one, so
  nothing about a single root can satisfy it automatically. Harder to set up,
  and the dispersion relation needs care about branch structure, but it is a
  genuine constraint rather than an identity.

Note what both survivors have in common: they involve more than one solution.
Any law evaluated on a single mode in a region where that mode already
satisfies the governing equations will come back exact and mean nothing.

## Attempted: layered-solver invariance — misstated, with a working replacement

This list said "swapping layer order in a stack where the physics is symmetric
should leave the dispersion invariant". **That is false for a cylindrical
stack**, and not marginally: the layers sit at *different radii*, so exchanging
two of them moves material from one radius to another and changes the medium.
Measured, it shifts the Stoneley slowness by about 1 %. There is no symmetry to
exploit — the premise was borrowed from plane-layered intuition and never
checked.

The invariance that does hold is **subdivision**: relabelling one homogeneous
annulus as several adjacent layers with the same properties changes the
description and not the medium. It holds to 1e-15 across n=0, n=1 and n=2, for
an open-hole mudcake and a cased steel-plus-cement stack, splitting the inner
or the outer layer, in slow and fast formations. It is a good oracle for the
reason the order-swap idea was reaching for: it exercises interface matching
and propagator composition across more than one boundary, which no
single-layer test can. It is also demonstrably not vacuous — a thickness error
of one part in ten thousand moves the answer nine orders above the noise floor.

Two caveats worth carrying forward. It is a *consistency* oracle: an error
common to every interface cancels and goes undetected. And a prior screening
pass mattered again — the suite already had twenty-plus tests for the
neighbouring invariance (a layer whose properties equal the formation reducing
to the unlayered solver), so most of what looked like new ground was already
covered.

**What it found.** That neighbouring invariance turns out to hold only in a
window. Appending a formation-equal layer — physically nothing at all — is
transparent while the added layer is thin, and stops being transparent when it
is not, somewhere above 0.1 m at 100 kHz, with both calls returning finite,
plausible slownesses. Which side is wrong was settled with an oracle from
outside the layered solver entirely:
`scholte_speed`. At 100 kHz the wavelength in the 2 cm mudcake is ~1.6 cm, so
the mode rides the innermost layer and must approach *that* layer's Scholte
speed. The plain stack does, to 0.05 %; the padded stack does not.

Neither the location nor the size of the error is stable, and that turned out
to be the sharper part of the finding. The padded answer has been seen at
289 m/s and at 1095 m/s for the same stack, disagreeing with the plain answer
by 7 % on one machine and by a factor of four on another. Two successive
versions of the test failed in CI — the first pinned where the spurious root
lands, the second how far off it is — before the assertion was reduced to the
only stable claim: that transparency is lost somewhere in the range.
That is failure mode "write the assertion after the measurement" recurring in a
subtler form, and recurring *twice*: the measurement was done both times, on one
machine, and the quantity measured simply had no stable value to record.
**Before asserting a number, ask not only whether it was measured but whether it
is a property of the system or of the run** — and when the answer is "of the
run", assert the structural claim instead of a tighter threshold. Chasing the
threshold is how a flaky test gets written. The instability is also the
diagnosis: a root search that lands somewhere different for identical physics
has lost precision rather than found another branch.

Calibration matters here and the first reading of it was too alarming. Genuine
thick layers, with real contrast, keep converging correctly at every thickness
tried and fail cleanly to NaN rather than to a wrong number. The defect belongs
to a *redundant* layer — a construction used to verify the solver rather than
one it exists to model — so the honest headline is that the verification
technique has a validity range, not that the solver is wrong for realistic
stacks. The existing transparency tests use a 0.005 m layer over 0.5-8 kHz,
comfortably inside that range, which is why this went unnoticed rather than
being a regression.

**The general lesson: an invariance used to verify a solver has a validity
range of its own, and it is not the solver's.** Establish where the check
itself stops working before reading a failure as the code's fault — and before
reading a pass as coverage.

## Attempted: the n=1 / n=2 rigid-pipe cutoffs — the premise was wrong again

This list proposed checking the `n=1` and `n=2` cutoffs against their
rigid-pipe closed forms, "the same check #61 ran for `n=0`, using the
appropriate Bessel zeros". The Bessel zeros are the easy part
(`j'_{n,1}` = 3.8317, 1.8412, 3.0542 for n = 0, 1, 2, and `_J1_FIRST_ZERO`
is indeed the first of those). The premise underneath is wrong.

**The n=0 check applies to a fluid-column mode; the n=1 and n=2 solvers do not
return one.** `pseudo_rayleigh_dispersion` returns a *higher-order* mode of the
borehole fluid perturbed by the wall, which is exactly what a rigid-pipe
resonance describes. `flexural_dispersion` and `quadrupole_dispersion` return
the *fundamental* modes at their orders, which are interface modes. The solver
exposes no n=1 or n=2 counterpart of pseudo-Rayleigh, so there is nothing for
the formula to be compared against. The candidate transplanted a check across
azimuthal order without asking whether the *kind of mode* transplanted with it.

Measurement says so independently, which is what settled it. The cutoff does
scale cleanly as `1/a` — so a geometric cutoff exists — but its log-log
sensitivities are about **0.87 on `V_S` and 0.10 on `V_f`**. A fluid-column
cutoff is fluid-controlled; these are shear-controlled, and by roughly an order
of magnitude in exponent. Changing `V_f` by 58 % moves the cutoff 4 %.

There is a second, independent reason the comparison cannot be rescued. The
rigid-pipe form is only defined for `V_S > V_f`, and in fast formations both
solvers are separately known to be defective (roadmap A.2, and the quadrupole
finding in #62) — 1086 and 1179 of 2000 samples converge. A "cutoff" read off
a defective band is a numerical artefact, not a physical frequency.

Three tests pin the outcome: that the `1/a` scaling holds, that the cutoff is
shear- rather than fluid-controlled, and that the rigid-pipe form does not
match — with the ratios for the two orders differing enough that no single
constant reconciles them, unlike the fixed offset #61 was able to document for
`n=0`.

**The lesson: when transplanting a check from one mode to another, verify that
the physical mechanism transplants too.** Both this and the layer-order
candidate failed the same way — a check that works somewhere nearby, carried
across a boundary (azimuthal order; plane-to-cylindrical geometry) that the
underlying mechanism does not cross. Two of the seven candidates died of it,
which makes it the most common failure mode in this list, ahead of any
numerical issue.

## Candidate oracles not yet attempted

Kept concrete so the next session does not have to re-derive the list. Whether
any of these bites is a measurement, not a promise. Of the seven candidates
tried so far, four became working oracles; one was vacuous (energy balance);
and two failed because a check was transplanted across a boundary its mechanism
does not cross (layer order, and the n=1/n=2 cutoffs) — one of those had a
working replacement nearby, the other did not. All three misses were caught by
measuring and none was obvious from the armchair.

Add one screening step before starting any of them, learned the hard way on the
tube-wave check: **grep the implementation for the formula first.** If the
solver already uses it — as a bracket, a seed, an initial guess — the check is
not independent, and the test has to be built to route around that use.

- **Attenuation vs the bound-mode limit.** `Im(k_z)` must go to zero
  continuously as a mode approaches its trapping boundary; a discontinuity
  there would indicate a branch error.
