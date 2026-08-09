# What remains to be done

A prioritised reading of the open items in `docs/roadmap.md`, current through
the analytic-oracle programme (PRs #59-#66) and the fluid-microannulus work
(PRs #67-#69).
`docs/roadmap.md` stays the authoritative status file; this is a snapshot of
*priority and reasoning* at one point in time, so check it against the tree
before acting on it.

## The shape of it

Five things are open, and they fall into three kinds — which matters more than
their ordering, because the kinds differ in whether they can be worked on from a
coding session at all. The struck-out rows are kept because how they closed is
the useful part of the story: each came loose from a larger item by measurement
rather than by planning.

| # | Item | Kind | Blocked on |
|---|------|------|-----------|
| ~~1a~~ | ~~Leaky-mode branch selection, n=0 (A.3)~~ | **closed** (#64) | — |
| ~~1c~~ | ~~Trapped pseudo-Rayleigh modes unexposed (A.4)~~ | **closed** (#66) | — |
| 1b | Leaky-mode root tracking, n=1 (A.2) | modelling *and* derivation | a Riemann-sheet analysis — possibly literature access |
| 2 | A real full-waveform sonic gather (F) | sourcing | fetching one **named** file from a host this sandbox cannot reach |
| 3 | Digitised validation figures (A.1, curve shapes) | sourcing | access to the books |
| 4 | Conda-forge recipe (D) | packaging | a PyPI release |
| 5 | Fluid microannulus, third piece (G.2) | **implementation** | nothing — two of three pieces are on `main` |

**Item 5 is new in kind, and it changes how the rest of this file should be
read.** For several revisions everything open was blocked on something outside
the session: a file behind an unreachable host, a book, a derivation, a release.
Item 5 is not. It is ordinary implementation work, with real user value, that
can be finished from here — see section 5.

Items 2 and 3 cannot be closed by writing code here. Note the qualifier: an
earlier revision said this sandbox's egress "reaches GitHub only", which probing
disproved — the AWS Open Data S3 buckets are reachable and downloads from them
work. They simply do not host the files in question. The obstacle is which host
serves a file, not a blanket network wall.

**Items 1a and 1c both came and went inside a revision, and neither was on any
list beforehand.** Both were found by an oracle aimed at something else. The
`n=0` branch-selection defect surfaced while checking leaky attenuation against
a radiation estimate; the trapped pseudo-Rayleigh modes surfaced while building
a biorthogonality check that needed several coexisting bound modes and turned up
some no public function returned. Neither needed a derivation or literature
access. That is now four claims in this file about what is *not* available here
overturned by measuring instead of reasoning.

**The previous revision's headline was that the oracle programme had run its
course. It was wrong, and it said so itself.** That revision concluded the
endorsed candidate list was empty, and closed with the warning that this claim
"is the same shape" as four earlier absence claims the file had already had to
withdraw, "and should be read with the same suspicion."

One more arrived two revisions later, and it is the strongest of the set: the
**Krauklis crack wave**, which validates the microannulus assembly to 0.02 % on
an absolute velocity (section 5). So the tally is now **twelve candidates: eight
working oracles, one vacuous, three transplant failures** (the previous
revision's "nine / five" was an arithmetic slip against its own list — see the
correction further down), and the file's own prediction about its pessimism has
been borne out for the fifth time.

The useful part is *how* it arrived, because it was not by working the list. It
was not on the candidate list and was not reasoned out in advance; it appeared
because an unexplained extra root in a new determinant was characterised instead
of dismissed, and its measured scaling identified it. `plans/learning.md` argues
that the best oracles come from asking what a check would do to a wrong answer.
This one is the other route — **measure the surprise first, and the oracle is
what explains it** — and that route is not exhausted by an empty candidate list,
because it is fed by new code rather than by a list.

What this means for prioritisation is the opposite of what the last revision
said. It claimed item 2 was "very nearly the only one that can move". That is no
longer true: item 5 can move, from here, today. Item 2 remains the most
*valuable* — it is what makes every quantitative claim in the repository mean
something — but it is an errand blocked on a host, and there is now real work
available beside it rather than instead of it.

## 1. Leaky-mode root tracking (A.2), and the three items that split off it

One roadmap item still needs the machinery; two smaller ones turned out not to
need it at all and are closed; and a fourth (G.2, the debonded regime) turned
out not to belong here either and is now **item 5**.

**A.2, the fast-formation flexural sparsity.** A fast formation behind casing
converges over only ~38 % of a 1-12 kHz band. It was filed as a *cased-hole
bracketing* problem; it is not. Strip the casing and cement away and the
identical formation is just as sparse in an open hole, over the same
frequencies. The cause is leakage: for `V_S > V_f` the flexural root leaves the
real `k_z` axis, and the real-axis sign change the solver hunts for survives
only beside the shear branch point at high frequency. Widening a real bracket
cannot recover it — no sign change exists below the cutoff in any sub-window,
and the middle window is singular for the propagator formulation anyway. Tests
pin the open-hole-vs-cased comparison so the attribution cannot drift back.

**Attempted; it is not a wiring job.** The complex machinery already exists and
works for `n=0`, so pointing it at the `n=1` determinant looks like an
afternoon. It is not. Continuation from high frequency reproduces the real
branch to floating-point noise and stops dead at the cutoff; fresh leaky-S
seeding below the cutoff yields incoherent spurious roots; strict fine-step
continuation from the cutoff fails on its first step. Even above the cutoff,
1 kHz continuation steps can hop to a different branch, so the extension needs
the validated marcher's regime checks and not just the tracker.

**A narrower piece split off and is now closed.** Checking the leaky-mode
attenuation turned up a defect in `n=0` — the part of the leaky solver that is
supposed to work. `pseudo_rayleigh_dispersion` seeded its march with a guess
near `1/V_S` at the highest requested frequency, and several leaky roots live
near that seed, so the mode it returned depended on the caller's grid: 2486 m/s
at 30 kHz for a grid ending at 40 kHz, 2952 m/s for one ending at 80 kHz, both
genuine roots. Worse, a merely-coarse grid returned all-NaN rather than a coarse
answer, with no warning.

Unlike the `n=1` problem this needed no Riemann-sheet derivation — the roots sit
on the principal sheet and are found reliably once seeded — so the fix was to
enumerate them at the seed frequency and expose a `branch` argument. Done; see
`docs/roadmap.md` A item 3. The one non-obvious part was checking that the root
*count* does not depend on the seed-scan density, because otherwise `branch=1`
would silently mean different things at different resolutions.

**A second piece split off and is also closed.** Building the biorthogonality
oracle needed several coexisting bound modes at one frequency, and finding them
turned up a family no public function returned: the *trapped* pseudo-Rayleigh
modes, `V_f < c < V_S`, where both formation waves are evanescent and the mode
is lossless with real `k_z`. They fell between the two existing functions —
`pseudo_rayleigh_dispersion` covers the leaky half above `V_S`, and
`stoneley_dispersion` brackets from `omega/min(V_S, V_f)` upward and so covers
only `c < V_f`. `trapped_pseudo_rayleigh_dispersion` closes that gap.
It needed no marching at all: the roots are real and simple, so each frequency
solves independently and the result is exactly grid independent — the opposite
of the trouble the leaky half caused. The oracle that found them also validates
them, since Auld's relation must hold across the trapped modes and the Stoneley
mode together.

What is missing *for the `n=1` case* is a derivation rather than code: which
Riemann sheet the `n=1` pole occupies below the cutoff. And there may be no pole to find — the
fast-formation flexural mode may simply exist only above its cutoff, with the
low-frequency dipole energy carried by a shear head wave. Settling that is what
Schmitt 1988 fig 4 is for, which quietly puts this item behind item 3's
literature access too.

## 2. A real full-waveform sonic gather (F) — the one that matters most

The harness shipped; adding a dataset is a one-entry change to
`scripts/fetch_real_data.py`. What is missing is the file. Neither registered
fixture is a sonic gather, so **the entire sonic processing chain is validated
only against synthetics.**

This is the binding constraint on every quantitative claim in the repo, and it
got worse when `sonic_ml` landed. The headline result — a learned inverse beats
classical STC by roughly an order of magnitude on shear velocity — is measured
on data drawn from *the same forward model that generated the training set*.
That measures identifiability, not field accuracy, and no further synthetic work
can close the gap.

**Candidates now exist; an earlier revision of this file was too pessimistic.**
It said "no openly redistributable full-waveform gather with trustworthy
reference picks is known to exist". A search found two credible sources, so that
is withdrawn:

- **Utah FORGE** via the DOE Geothermal Data Repository — Schlumberger dipole
  sonic in **DLIS** (already readable by `fwap.io.read_dlis`), from an
  eight-receiver array with monopole and dipole sources, and **CC BY 4.0**.
- **IODP / ODP** via the LDEO Borehole Research Group — sonic waveforms for many
  holes, in DLIS plus a Python-friendly binary export, documented as eight
  waveforms × 512 samples at 10/40 µs every 15.24 cm. Licence unconfirmed;
  matters less than it looks, since the harness fetches on demand and never
  vendors.

Neither has been downloaded or opened, so this is a shortlist from published
metadata, not a verified result.

Fetching was attempted and the result is more specific than "egress is blocked".
The AWS Open Data buckets `gdr-data-lake` and `oedi-data-lake` **are** reachable
and object downloads work — but they carry only bulk monitoring data (DAS,
geophone, CASSM, magnetotellurics), no wireline logs at all. The hosts that do
serve the log submissions (`gdr.openei.org`, `data.openei.org`,
`brg.ldeo.columbia.edu`, `osti.gov`, `iodp.tamu.edu`) all refuse to connect. So
the obstacle is which host serves the file, not the data: a session with
ordinary web egress could fetch it directly.

The next step is one person opening a file to confirm it holds per-receiver
waveforms rather than processed curves, then a checksum and a one-line registry
entry.

## 3. Validation figures (A.1) — the figures are blocked, the *tie* is not

**The solver is now tied to literature, without any figure.** `scholte_speed`
solves the classical secular equation for an interface wave on a *plane*
fluid/solid boundary — a different equation from the cylindrical modal
determinant, with no Bessel functions and no borehole radius in it. As the
wavelength shortens the borehole wall looks flat, so `stoneley_dispersion` must
approach it, and it does: better than 0.1 % at 400 kHz, converging monotonically
and from opposite sides in fast and slow formations. The oracle is validated in
turn by its own light-fluid limit, where it collapses to Rayleigh's equation and
reproduces `rayleigh_speed` — a third, separate implementation.

So the honest statement about this item changed. It is no longer "nothing ties
the solver to literature"; it is "the *dispersion-curve shapes* are still
untied, only the short-wavelength limit is". That is a smaller gap than it was,
and a differently-shaped one: an asymptote check cannot catch an error in the
middle of a curve.

**Closed.** `fwap.validation` scores an fwap dispersion curve against a
digitised reference, and the validation notebook asserts a 5 % RMS budget per
curve — verified to fail on a 12 %-perturbed reference, so it is a real gate
rather than a described one. Most of that module is input validation, because
hand-tracing a printed figure fails in a handful of ways that all produce
plausible files (µs/ft read as s/m, a velocity axis traced as a slowness one,
kHz left unconverted); each is refused with a named diagnosis, and units are
never silently rescaled.

**Still open.** No reference CSV is shipped, so no *curve shape* is checked
against a published figure — and the notebook says which of its sections are and
are not validated rather than letting green plots imply otherwise. Digitising
needs the books (Paillet & Cheng 1991; Schmitt 1988/1989; Tang & Cheng 2004 figs
3.7/3.10 and 7.1). Once a CSV lands in `docs/notebooks/_data/` under the
documented name, no code changes — the section scores and gates automatically.

## 4. Conda-forge recipe (D)

Packaging only, and unblocked once the first PyPI release is live. Reversible
and low-risk; listed for completeness rather than because it competes with
anything above.

## 5. The fluid microannulus (G.2) — two pieces on `main`, one left

The only open item that is blocked on nothing. It began as part of item 1 and
was re-diagnosed out of it in PR #67.

### Why it was mis-filed, and the two models of debonding

The cased dataset spans only the *bonded* regime, so the bond inverse grades
cement quality and is explicitly not a free-pipe detector. That much always
stood. What was wrong was the next sentence, which used to read "reaching
debonding needs a leaky-mode cased forward model", and so filed the whole item
behind the derivation-blocked `n=1` work. The distinction it missed is between
two different physical models:

* **Soft cement.** The documented restriction is real: the cased Stoneley
  converges over the whole band down to `cement_vs = V_f`, is partial just
  below, and is gone by `1200 m/s`. The mechanism is in
  `_stoneley_kz_bracket_cased`, which sets the bound-regime floor from
  `min(V_S, V_f, *(layer V_S))` — the *softest shear velocity anywhere in the
  stack*. Once that drops below the fluid velocity there is no bound window
  containing the physical Stoneley mode.
* **A fluid microannulus**, the standard model of debonding in cement-bond
  logging, is a different configuration and is *not* excluded by that argument.
  It cannot be approximated by a very compliant elastic layer either — precisely
  because an elastic layer, however soft, does drag the floor down. Measured: a
  compliant layer breaks convergence at any thickness tried, down to 0.2 mm.

One correction to how that was first written here. "A fluid contributes no floor
to that bracket" is not quite right — it contributes one at its *acoustic*
velocity. What matters is that this is ~1500 m/s rather than a near-zero shear
velocity, so it does not drag the floor below the fluid and collapse the window.

Free pipe proper — casing surrounded by fluid, the classic CBL casing-ring
amplitude — remains partly a phenomenological amplitude effect rather than a
modal one, and that part is unchanged by any of this.

### What is built (PRs #68, #69)

1. **The fluid element.** `_fluid_layer_e_matrix_n0` /
   `_fluid_layer_propagator_n0`: two amplitudes rather than four, shear traction
   identically zero, axial slip permitted, so the propagated state is the pair
   `(u_r, sigma_rr)`. Pinned by the Bessel Wronskian, which collapses
   `det E_f` to `-1/(rho omega^2 r)` and `det P_f` to `r_inner/r_outer` — no
   dependence on frequency, velocity, density or `k_z`. Its accuracy range is
   measured rather than assumed (machine precision to a Bessel span of ~2,
   useless by 20; a debonding gap sits below 0.1).
2. **The global assembly.** `_modal_determinant_n0_microannulus`: an 11x11
   determinant for `fluid | casing | microannulus | cement | formation`, with
   the gap amplitudes folded out through the fluid propagator so extra layers in
   either block leave the size unchanged.

The assembly had no reduction to the existing solver to check against — the
`annulus_thickness -> 0` limit is a frictionless *slip* interface, not the
bonded stack, since shear traction stays zero on both faces and `u_z` stays free
however thin the gap. Measured at 8 kHz the Stoneley-like root converges as
`O(h)` to 1383.45 m/s against 1400.04 m/s bonded, a 1.2 % offset that does not
close. What replaced the reduction was the **Krauklis crack wave** — see the
headline note at the top of this file, and `plans/log_output.md` for the
numbers.

### What is left

**The public dispersion function.** Everything below it exists; this is the
wiring plus one real decision.

* **Branch selection is the decision, not the wiring.** The determinant carries
  **two root families** — a Stoneley-like mode just below the fluid velocity,
  and the gap mode at 68-620 m/s over four decades of gap thickness — and they
  move in opposite directions as the gap closes. A bracket that assumes one root
  is exactly the `n=0` defect closed in #64. The root set is already pinned as
  independent of scan grid and window, so a regression would show.
* **Expose the gap mode too, not just the Stoneley shift.** It is a debonding
  indicator in its own right, and it is the *better* one: its speed depends on
  gap thickness as `h^{1/3}` where the Stoneley root barely moves at all. It
  also has a closed form, so a caller can invert it for gap thickness directly.
* **A public way to express the configuration.** `BoreholeLayer` requires
  `vs > 0` and cannot represent a fluid. This needs a type — or an explicit
  annulus argument — and whichever is chosen becomes public API, so it needs the
  three-file lockstep (`fwap/__init__.py`, `docs/api.rst`,
  `scripts/check_public_api.py`).
* **Then the `sonic_ml` consumer**, which is what section G item 2 in
  `docs/roadmap.md` actually wants: a debonded-regime dataset, and with it the
  first fair CBL-amplitude comparison rather than a strawman.

Not required for any of the above, and worth stating so it is not assumed:
`n=1` / `n=2` microannulus assemblies would be needed for *flexural* CBL work,
and those are a separate, larger job. The `n=0` path is self-contained.

## Loose ends

- Whether `penalty="tv"` should be the default in `sonic_ml.models.joint` —
  deliberately unresolved, because it turns on how bedded a real target log is.
  That is item 2, not another synthetic sweep.
- Coupling across *mode* as well as depth in joint inversion: untouched.

## Closed since this file was first written

Kept here because each one moved a number or a conclusion that earlier revisions
of this file got wrong, and the corrections are worth not losing.

- **Two-mode cased dataset** (PR #54).
  `generate_slow_two_mode_cased_dataset` carries both the Stoneley and the
  flexural mode, fully bound. The catch is the prior: the two modes fail in
  opposite directions, so the window where both hold is `V_S` 1420-1495 m/s —
  about 80 m/s, and **disjoint from the default cased prior** (1700-3000 m/s),
  making it a different dataset rather than a subset. Measured both-modes-bound
  fraction: 0.00 at 1350 m/s, 0.42 at 1380, 0.92 at 1400, 1.00 from 1420 up.
  This also withdrew a wrong figure — an earlier revision said "only ~15 % of
  draws are slow", measured over the *default* `FormationPriors` rather than the
  fast prior the cased generator actually pins, where the true figure is 0 %.
- **A.2 re-diagnosed** (PR #51). Filed as cased-hole bracketing; measurement
  showed the open hole is equally sparse, relocating it into item 1.
- **A.1 machinery** (PR #50). See item 3.
- **Scholte analytic oracle** (PR #59) — `scholte_speed`, and with it the first
  literature tie the validation notebook actually makes. Worth noting *how* it
  arrived: the previous revision of this file said the only options left in this
  environment were "more tests against existing behaviour, or documentation".
  That was wrong. A third option existed — find an oracle that needs no
  published figure — and it was found by asking what could be checked rather
  than by re-reading the list of what was blocked.
- **Leaky-mode branch selection** (PR #64). Item 1a; see item 1 above. The
  measurements from the defect were re-run the other way round when the fix
  landed, which is now the convention for defect fixes here.
- **Layer-subdivision invariance, and the limit of the transparency check**
  (PR #65). Subdividing a homogeneous annulus is exact to 1e-15 across every
  azimuthal order and both open-hole and cased stacks. It also established that
  the *neighbouring* invariance — padding a stack with a formation-equal layer —
  holds only while the layer is thin, so that verification technique has a
  validity range and should not be used outside it.
- **Trapped pseudo-Rayleigh modes exposed** (PR #66). Item 1c;
  `trapped_pseudo_rayleigh_dispersion`. Found by an oracle aimed at something
  else, and validated by the same one.
- **G.2 re-diagnosed out of item 1** (PR #67). Filed for revisions behind the
  derivation-blocked `n=1` work on the grounds that "reaching debonding needs a
  leaky-mode cased forward model". It does not: a fluid microannulus is a
  different configuration from soft cement, and is an implementation task. Now
  item 5. The same PR fixed compliant cased layers returning spurious 3-12 m/s
  roots instead of `NaN`, some with no warning at all.
- **The fluid-annulus element** (PR #68) and **the microannulus global
  assembly** (PR #69). Items 5.1 and 5.2. The assembly brought the twelfth
  oracle candidate and the best of them, and fixed two more ways a determinant
  sweep could warn or raise instead of returning `NaN`. Worth keeping for the
  method as much as the code: the oracle was found by characterising an
  unexplained extra root rather than by working a list, which is a route the
  "programme is finished" conclusion did not account for.
- **The attenuation module's test synthetic is acausal** (PR #66). Constant-Q
  amplitude loss with the phase left untouched violates Kramers-Kronig. Not a
  bug — both estimators read `|S(f)|` only — but the recovered Q moves by about
  a third on a causal gather, in the direction that makes the existing tests
  understate the estimators. A causal counterpart is now covered alongside.
- **Sonic-gather candidates found** (PRs #56, #57) — not closed, but item 2
  moved further in these two than in anything before. Withdrew "no openly
  redistributable gather is known to exist", then withdrew the replacement's own
  error that Utah FORGE is "mirrored on AWS Open Data" (the reachable buckets
  carry DAS and geophone data, not wireline logs). Two wrong claims in
  succession on the same item is worth remembering when reading the rest of
  this file: statements about what does *not* exist are the ones that age worst.

## Recommendation

**Do item 2, and do it first.** It is the highest-value item on the list and it
is now the cheapest: fetch the Utah FORGE dipole sonic DLIS from
`gdr.openei.org`, open it, confirm it carries per-receiver waveforms rather than
processed slowness curves, then compute a SHA-256 and add one `RealDataset`
entry. Everything downstream of it — every quantitative claim in the repository,
`sonic_ml`'s headline included — is currently measured against the same forward
model that generated the training data. One real gather changes what those
numbers mean.

**Do item 5 from inside a coding session.** It is the only open item blocked on
nothing, two of its three pieces are already on `main`, and what remains is a
public dispersion function plus one genuine decision about branch selection. It
is also the item most likely to throw off another finding, on the evidence: the
two PRs that built it produced the strongest oracle in the programme, two
determinant-contract defect fixes, and one correction to this file's own
reasoning — none of which were planned.

The two recommendations do not compete. Item 2 is an errand for whoever has
network reach; item 5 is work for whoever has a session. The previous revision
paired "do item 2 first" with "and there is nothing else you can do", which was
the part that was wrong.

After those, item 3 (the digitised figures) for whoever has the books; item 1
needs a derivation before any code is worth writing; item 4 waits on a release.

If work continues in this environment regardless, an earlier revision said the
only options were "more tests against existing behaviour, or documentation".
That turned out to be wrong — the Scholte oracle (PR #59) was neither, and it is
a real check the repository did not have. So the honest version is narrower:
what is unavailable here is *external data*, not external truth. Closed-form
results, asymptotic limits and independent formulations of the same physics are
all reachable, and each one that can be implemented from theory is worth more
than another test of behaviour against itself.

Three candidates were listed in that spirit, with the caveat that "whether any
of these bites is a measurement, not a promise". All three have now been run,
and all three bit:

- **The pseudo-Rayleigh cutoff against its rigid-pipe closed form** (PR #61).
  The `1/a` scaling holds to 1 part in 300; the absolute cutoff overshoots by
  ~2.8x, so the docstring's advice to use it as a band guard would have
  discarded valid data. Corrected.
- **The quadrupole high-frequency asymptote** (PR #62). Validates the
  slow-formation solver and exposed a fast-formation defect, which also
  corrected a generator comment about `min_finite` filtering.
- **The leaky-mode attenuation against a radiation estimate** (PR #63).
  `leaky_radiation_attenuation` confirms the solver's attenuation to within a
  factor of two with the right radius scaling, and turned up something the
  check was not looking for: `pseudo_rayleigh_dispersion` returns a different
  mode depending on the caller's frequency window, and fails silently to
  all-NaN on grids that are merely too coarse. See item 1 above.

Six more were tried after those three, and one arrived later without being
tried at all. The full tally is twelve candidates: **eight working oracles, one
vacuous, three transplant failures.**

*Arithmetic correction.* The previous revision gave this as "nine candidates:
five working oracles, one vacuous, three transplant failures", while listing
seven working ones immediately below it. The list was right and the count was
wrong; 8 + 1 + 3 = 12 is the tally that matches the entries. Recorded rather
than quietly fixed, because a summary figure that disagrees with the list under
it is the kind of error that survives several readings.

- Working: Scholte high-frequency limit (#59), rigid-pipe pseudo-Rayleigh
  cutoff (#61), quadrupole asymptote (#62), ray radiation estimate (#63), White
  tube-wave low-frequency limit (#64), layer-subdivision invariance (#65),
  modal biorthogonality (#66) and the Krauklis crack wave (#69). Between them
  they corrected two pieces of documentation that would have led a careful user
  wrong, exposed four code defects, and turned up one entire unexposed mode
  family.
- Vacuous: the leaky-mode energy balance (#64). It reproduces `Im(k_z)`
  exactly — and does so for `k_z` values that are roots of nothing, because
  closing the balance inside the fluid is an identity.
- Transplant failures: layer-order invariance (plane-layered intuition applied
  to a cylindrical stack), the n=1/n=2 rigid-pipe cutoffs (a fluid-column
  formula applied to interface modes), and Kramers-Kronig (a material-response
  relation applied to geometric waveguide dispersion). All three were written
  into a candidate list before being checked.

Two findings from the later ones bear on other parts of this file. The layered
transparency invariance has a **validity range of its own** — padding a stack
with a formation-equal layer stops being a no-op above ~0.1 m at 100 kHz — so
that verification technique should not be used outside it. And the attenuation
module's test gather is **acausal**: `exp(-pi f t / Q)` with the phase
untouched, which violates Kramers-Kronig. Not a bug, since both estimators read
`|S(f)|` only, but on a causal gather the recovered Q moves by about a third,
and in the direction that makes the existing tests understate the estimators.

This file has now been too pessimistic four times — about whether an open sonic
gather exists, twice about what was left to do here, and about whether a large
piece of work could be carried to completion unaided. Every time the error was a
claim about absence. The claim in *this* revision that the oracle programme is
finished is the same shape, and should be read with the same suspicion: it means
no endorsed candidate remains on the list, not that none exists.
