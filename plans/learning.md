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

## Choosing what to measure

Every entry above began with a decision about *which number to compute*, and in
hindsight that decision mattered more than the physics or the code. What
follows is the specific recipe this work converged on, then the general
principle underneath it.

### The specific recipe

Given a solver and a claim about it, these are the questions that have actually
produced findings here, in the order they paid off.

1. **Which return values has nothing ever looked at?** Not which functions are
   untested — which *fields*. `attenuation_per_meter` had three tests proving
   it was present, finite and positive, and none that looked at its magnitude;
   that gap was the whole of PR #63. Enumerate the components of every returned
   object and ask what each one is checked against.

2. **What does the closed form leave out?** When comparing against an analytic
   limit, the sharpest measurement is usually of a parameter the *formula does
   not contain*. The tube-wave form has no borehole radius in it, so the
   radius-independence of the solver's limit is a stronger statement than the
   value: agreement on the value could be a coincidence of one geometry, but
   `a` = 0.05-0.30 m agreeing to 5e-8 cannot. Look for the variable the two
   sides disagree about the existence of.

3. **What would this look like if it were wrong?** Compute the check on a
   deliberately wrong input and record the number. If it comes back the same,
   stop — that is the energy-balance outcome, and it cost ten minutes to find
   and would have cost a false claim to miss. Cheap enough to do every time.

4. **Does the answer depend on something it must not?** Grid density, grid
   endpoints, scan resolution, platform. Two defects (#63, #64) were found by
   varying a parameter that carries no physics. The rule generalises: for every
   argument the caller controls that *should not* affect the answer, vary it.
   Grid *endpoints* found the leaky branch defect after grid *density* had
   already come back clean, so vary each independently rather than assuming one
   stands for the family.

5. **Where does the check itself expire?** An invariance used to verify a
   solver has its own validity range. Measure it before reading a failure as
   the code's fault — the transparent-layer breakdown in #65 is a defect in the
   verification technique, not in the configurations the solver models, and
   only measuring the boundary made that distinction available.

6. **Is the quantity a property of the system or of the run?** Before asserting
   any number, ask whether it would survive a different machine. Where a
   spurious root lands, and how far off it is, both failed that test in #65 and
   produced two CI failures in a row. When the answer is "of the run", assert
   the structural claim instead: not *how much* the invariance is violated, but
   *that* it is.

### The general principle

The measurements that found something all share a shape: **they compare two
things that are supposed to agree for a reason that lives outside the code
being tested.** Everything else is a restatement of the implementation.

That gives a single test for whether a proposed measurement is worth making —
*what would have to be true of the world, rather than of the program, for this
to come out right?* If the answer is "nothing in particular; it follows from
the definitions", the measurement is a tautology however elaborate. That is
what killed the fluid-only energy balance, the momentum balance, and interface
flux continuity, and it is what makes modal biorthogonality and
Kramers-Kronig worth attempting: both require a second, independently computed
solution, so neither can be satisfied by construction.

Three corollaries earn their keep:

- **Prefer a measurement that can fail for exactly one reason.** Comparing a
  solver against a closed form in a regime where three approximations overlap
  gives a number nobody can act on. The Scholte check works because at short
  wavelength there is precisely one thing the borehole solver must reduce to.
- **Measure the boundary, not the interior.** Where something stops working is
  more informative than that it works, because it is falsifiable in a direction
  the happy path is not. Every defect in this list was found at a boundary: a
  cutoff, a validity floor, a grid endpoint, a thickness limit.
- **Quantify the discrimination, not just the agreement.** "The check passes"
  is worth little without "and here is how badly a wrong answer would fail it".
  The subdivision oracle is credible because a thickness error of one part in
  ten thousand moves the result nine orders above the noise floor; that ratio,
  not the 1e-15 agreement, is what makes it a test.

The through-line is that a good measurement is *chosen adversarially* — picked
because it could embarrass the code, not because it is convenient to compute.
Measurements chosen for convenience mostly confirm, and confirmation from a
check that could not have failed is the most expensive kind of nothing,
because it is indistinguishable from the real thing until someone relies on it.

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
- **Causality (Kramers-Kronig) — attempted; it does not apply to the modal
  solver, and where it does apply is somewhere else entirely.** See below.

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

## Attempted: modal biorthogonality — the first oracle needing two solutions

The prediction from the conservation-law survey held. Auld's waveguide
reciprocity relation couples two *different* eigenvectors, so unlike every
earlier check it cannot be satisfied by construction from the solution being
tested.

**The test set exists and is richer than expected.** In a fast formation the
n=0 bound spectrum contains the Stoneley mode (`c < V_f`) *and* the trapped
pseudo-Rayleigh modes (`V_f < c < V_S`) — four bound modes at 30 kHz, six at
50 kHz, all azimuthal order 0, so orthogonality among them is not the trivial
angular-integral kind. Worth noting in passing: `stoneley_dispersion` returns
only the first of these, because its bracket stops at `omega/V_f`. The trapped
modes are not exposed by any public function.

**Result: the relation holds to ~1e-13** across every pair, with the diagonal
O(1) — so it is orthogonality, not everything being small. Three tests pin it.

Three things about getting there are worth keeping.

*Check the eigenfunctions before trusting the integral.* The first
boundary-condition check returned `|du_r| / |u_r| = 2.00` exactly — the
signature of equal and opposite, and a sign convention mismatch between the
determinant's matrix rows and the field expressions derived from potentials.
Had that gone unnoticed it would have surfaced as a *failed orthogonality
relation*, i.e. as a fabricated finding about the solver. The BC check is now
a test in its own right, and the general rule is: **when a check is built from
machinery the code under test does not provide, validate the machinery
separately first, or its bugs will be attributed to the code.**

*The first bilinear form was wrong, and wrong in an instructive direction.*
Using one term of the pairing rather than Auld's difference `S_mn - conj(S_nm)`
leaves off-diagonals around 1e-2 — small enough to look like a tolerance
problem and invite a loosened threshold, rather than a wrong formula. What
distinguished them was that the correct form improved matters by ten orders,
not by a factor of two. A test now asserts the wrong form fails, so the
tolerance in the real test is evidence rather than a fitted constant.

*Adaptive quadrature manufactured a false residual.* With `scipy.quad` the
off-diagonals sat at 1e-4 and **grew** with integration span — the tell that
the error was numerical rather than physical, since a genuine truncation error
shrinks. The integrand underflows in the evanescent tail and the adaptive rule
spends its error budget there. Fixed-node Gauss-Legendre over a modest span
gives 1e-13. **When a residual moves the wrong way as you refine, the
refinement parameter is the bug.**

## Attempted: Kramers-Kronig — wrong target, but it found something next door

Listed as the other candidate the "needs more than one solution" criterion
endorsed, on the grounds that Kramers-Kronig relates `Re(k_z)` and `Im(k_z)`
across frequencies and so cannot be satisfied by a single root. The criterion
was right; the target was wrong.

**One line of data disproves it.** A subtracted Kramers-Kronig relation on
complex slowness says zero attenuation at every frequency forces zero
dispersion — the dispersion integrand vanishes identically. The bound Stoneley
mode is exactly lossless: the solver returns no attenuation field for it at
all. Its phase velocity nevertheless moves **8.26 %** across the band, from the
tube-wave limit to the Scholte speed. If modal slowness were KK-constrained
that would be impossible.

The physics is not subtle in hindsight. Kramers-Kronig follows from causality
of the **constitutive relation** — a frequency-dependent modulus. Here the
medium is perfectly elastic and non-dispersive, and every bit of the frequency
dependence comes from the boundary conditions. Waveguide dispersion is
*geometric*. A hollow metallic waveguide is the textbook case: strongly
dispersive with perfectly lossless walls, and nobody expects KK to constrain
its cutoff. **This is the transplant failure again — a check carried from
material response to geometric dispersion, which is a boundary its mechanism
does not cross.** Three of nine candidates have now died of that, and it is
worth noticing that "needs more than one solution" is necessary but not
sufficient: it screens out tautologies, not misapplications.

**Where Kramers-Kronig does bite here.** Not the modal solver — the attenuation
module, and not as an oracle for the code but as a correction to how it is
tested. `tests/test_attenuation.py` builds its gather by multiplying the
spectrum by `exp(-pi f t / Q)` and leaving the phase untouched. That waveform
is acausal: constant-Q amplitude loss without the accompanying Kolsky-Futterman
velocity dispersion violates KK, and the result carries energy arriving *before*
the geometric arrival — measured at a pre-arrival energy fraction of 1.5e-7,
against 4.9e-12 for the causal counterpart.

It is not a bug. Both estimators read `|S(f)|` only, so the missing phase
cannot bias them directly. But they *window in time*, and dispersion reshapes
the waveform inside the window, so the route is real: on the causal gather the
centroid estimate moves from 62 to 41 against a planted Q of 50, and the
spectral-ratio estimate from 117 to 81. Both estimators look **better** on the
physical signal than on the one the tests use — so the existing accuracy claims
understate them rather than flattering them, which is the benign direction but
still not what the tests say they are measuring. Four tests now cover the
causal case alongside the original.

**And the sign was wrong on the first attempt.** The Kolsky phase applied with
the opposite sign makes high frequencies arrive *later*, which is the
anti-causal direction; it raised pre-arrival energy from 1.5e-7 to 8.4e-7 and
produced a spurious "causality doubles the recovered Q" result that was one
step from being written up. What caught it was refusing to settle the sign by
algebra and instead **measuring the property the sign is supposed to produce** —
pre-arrival energy. The general form of that rule is already in this document
(validate machinery you built before attributing its bugs to the code under
test); this is the second time it has paid, and the second time the tell was a
number moving in the wrong direction.

## Candidate oracles not yet attempted

Kept concrete so the next session does not have to re-derive the list. Whether
any of these bites is a measurement, not a promise. Of the eight candidates
tried so far, five became working oracles; one was vacuous (energy balance);
and two failed because a check was transplanted across a boundary its mechanism
does not cross (layer order, and the n=1/n=2 cutoffs) — one of those had a
working replacement nearby, the other did not. All three misses were caught by
measuring and none was obvious from the armchair.

The survey above earned its keep once and misfired once. Its criterion — a
check needs more than one solution — correctly predicted biorthogonality would
work. It also endorsed Kramers-Kronig, which turned out not to apply to the
modal solver at all, because the criterion screens for tautology and not for
misapplication. Both filters are needed: *does this need more than one
solution?* and *does the mechanism that makes it true survive the move?*

Add one screening step before starting any of them, learned the hard way on the
tube-wave check: **grep the implementation for the formula first.** If the
solver already uses it — as a bracket, a seed, an initial guess — the check is
not independent, and the test has to be built to route around that use.

- **Attenuation vs the bound-mode limit.** `Im(k_z)` must go to zero
  continuously as a mode approaches its trapping boundary; a discontinuity
  there would indicate a branch error.
