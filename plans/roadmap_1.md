# What remains to be done

A prioritised reading of the open items in `plans/roadmap.md`, current through
the analytic-oracle programme (PRs #59-#66), the fluid-microannulus work
(PRs #67-#72), and the real-data arrival with the defect it exposed and the
waveform reader it demanded (PRs #74 onward).
`plans/roadmap.md` — which absorbed the old `docs/roadmap.md` — is the fuller
status file; this is a snapshot of *priority and reasoning* at one point in
time, so check both against the tree before acting on either.

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
| ~~2~~ | ~~A real full-waveform sonic gather (F)~~ | **closed** (#74) | — |
| ~~2a~~ | ~~The compressional-pick defect it exposed (F.1)~~ | **closed** (#75, #76) | — |
| ~~2b~~ | ~~A waveform path in `read_dlis` (F.3)~~ | **closed** | — |
| ~~5a~~ | ~~The debonded dataset generator, baseline and inverse (G.2)~~ | **closed** | — |
| ~~6~~ | ~~The debond inverse in `sonic_ml.bench` (G.6)~~ | **closed** | — |
| ~~2d~~ | ~~The ODP file's offsets and its 950-A/952A header (F.5)~~ | **closed** | — |
| ~~2c~~ | ~~A waveform fixture CI can fetch (F.2)~~ | **closed** | — |
| 1b | Leaky root tracking, **n=1 *and* n=2** (A.2) | **two defects, measured against the published curves for three fast rocks**: a bracket anchored to `V_R` instead of Scholte — it empties at 4.43-4.45 kHz whatever the formation, and where it answers it is 62 % fast in sandstone, 72 % in limestone, **134 % in granite** — plus genuine leakiness at **both** ends of the band, not just below cutoff. `n=2` checked against fig 7b and it is the **worse** of the two: 65-75 % coverage against `n=1`'s 21-36 %, so a `NaN` filter keeps more of the wrong answers. Fig 12 adds the layered path: a 16 cm invaded zone takes flexural coverage 9 % → 73 % on the same rock with every extra answer an overtone, so **coverage is an inverted health signal there**. Fig 15 shows the identical call at **1.5 % rms** in a slow formation, so the layered propagator is sound and the bracket is the whole defect — though fig 16 refines this: that holds for *phase* velocity, and the layered *group* velocity is twice as wrong as the open-hole one (+6.3/+8.0 % against +3.0 % on the published Airy arrival) | a Scholte-edged bracket earns 4.4-16.4 kHz at 0.66 % median error; the two ends need a Riemann-sheet analysis, and the selection rule has to ship with the bracket. **One fix repairs the open-hole and layered paths together**. Fig 6 adds two: the `n=2` **cutoff** is 32 % high (first root 8.29 kHz against a published 6.29), and **coverage is not reproducible** — two grids differing by last-bit rounding give 47 vs 42 converged points of 71, though fig 14 shows that instability is model-specific rather than universal. Fig 14 also **bounds what this fix buys**: its twelve waveforms turn on peak amplitude, which is excitation × propagation, and `BoreholeMode` carries neither — so a corrected bracket would not reproduce that figure at all. What it would fix there is the Airy phase, which today comes out with the **wrong sign** (`v_g` negative on 18 of 48 adjacent samples) |
| 3 | Digitised validation figures (A.1) — **3 of 5, re-scoped** | sourcing | the books, for **cased Stoneley / VTI flexural** — pseudo-Rayleigh is tied by fig 1a at ~1 % on both branches. Schmitt & Cheng figs 2a, 7 and 8a are **done**. Fig 8a also refutes the re-scope's premise: traced with care it ties `stoneley_dispersion` at **0.04 % rms**, external, tighter than most of the analytic ties — the "5 % overlay budget" was a choice, not a limit |
| 5 | Confirm two registered checksums (F.4) | one fetch each | egress to `gdr.openei.org` and `zenodo.org` |
| 6b | **A.6 `n=2` layered path refuses every invaded zone** | *new, found by fig 17*: `quadrupole_dispersion_layered` raises on any layer slower in shear than a slow formation — i.e. every invaded zone — so eight of fig 17's twelve waveforms are unrepresentable. `flexural_dispersion_layered` applies the same check only for **two or more** layers, so the identical one-layer model is accepted at `n=1`, and those answers tie the published curves at 1.5 % rms | a scoping decision, not an algorithm: relax `n=2` to match `n=1`, or tighten `n=1` and document invaded zones as out of scope for both. The current split cannot be right in both directions |
| 7 | Delta-matrix / Abo-Zena stack reformulation (A.5 residue) | modelling, optional | nothing |
| 4 | Conda-forge recipe (D) | packaging | a PyPI release |

**The shape has changed since the last revision, and mostly by closing.** G.2
is finished end to end — generator, closed-form baseline at 18.1 % in gap
width, learned residual inverse at 2.5 % held-out — and F.3 and G.6 are
finished with it, and F.5 and F.2 with them.

**What that leaves is five rows, and this table said three.** An audit found two
that had never been listed here at all: F.4, which is two unconfirmed checksums
needing one fetch each, and the delta-matrix reformulation left over from A.5.
Both are small; neither is nothing, and a priority table that silently omits
work is worse than one that ranks it low. Row 1b also understated itself — the
same leakage affects `n=2`, so one fix repairs two solvers.

Of the five, only two are blocked on something outside the project (a book for
3, a release for 4). 1b is a derivation, 7 is optional modelling, and 5 needs
nothing but network access to two hosts.

**F.2 is the one worth reading the post-mortem on**, because it was the
highest-priority row in every revision of this file and it was never actually
blocked. Two docstrings asserted that no openly redistributable full-waveform
sonic gather was known to exist. That was a search result written down as a
fact, and it stopped anyone searching again — for the life of the project. A
CC0 eight-receiver DSI run had been sitting in a public archive the whole time.
The lesson generalises past this repository: a negative claim about the outside
world has a shelf life, and the moment it is stated flatly in code it stops
being re-checked. It should have been written as "searched on <date>, found
none" — which invites the re-check that "none exists" forecloses.

Item 6 is worth a line on the way out, because it was filed as ordinary wiring
and did not behave like it. Scoring the closed form through the harness split
by gap width showed its error is **6× worse on wide gaps than tight** —
16.5 % against 2.5 %, with confidence intervals nowhere near overlapping. A
single averaged number had been hiding a regime dependence that the physics
predicts and that the learned model removes. The general lesson is the one
this file keeps re-learning: an average is a claim about a population, and the
cheapest way to find out it is the wrong claim is to score the sub-populations
you already have labels for.

Two of the closed rows came out of the work rather than off a list. Building the
debonded dataset showed the item **as planned would have been uninvertible**:
the cased Stoneley mode is blind to gap width (0.05 % over a 100× range) while
the crack wave carries it at roughly 100:1. And a second real log — ODP Leg
157 Hole 952A, handed over after F.3 shipped — showed the AXIS-only reader was
right but not sufficient, since that file declares no AXIS at all.

~~**2c is the most valuable row and is no longer blocked on work.**~~ **Closed
since**, and not by the file that paragraph was about. ODP 952A made the row
*actionable* by being small; what closed it was noticing that the premise
underneath — no openly redistributable gather exists — was simply false, and
that a **CC0** eight-receiver DSI run (IODP U1347A) had been public throughout.
The 952A work was not wasted: it produced the waveform reader and the F.5
provenance method that the U1347A entry then used.

Item 3 still cannot be closed by writing code here — but note how much of it
turned out not to need the books at all. It went five figures → three → four,
the reductions coming from analytic ties that already existed or cost nothing to
add, and the one restoration from re-reading A.2 against it. Before calling the
remaining four blocked, the qualifier that survived earlier revisions is worth
keeping in view: the obstacle was always *which host serves a file*, never a
blanket network wall — which is exactly how item 2 eventually closed.

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

What this means for prioritisation kept moving. An earlier revision claimed item
2 was "very nearly the only one that can move"; the next said item 5 could move
too. Item 2 has since closed outright, and its value was exactly what this file
predicted: the first real log immediately exposed a defect that had been shipping
for the project's whole life and that no synthetic could have found. Two of its
consequences were 2b and 2c; 2b turned out to be exactly the sort of small
implementation job that earlier revisions of this file kept assuming did not
exist, and is now closed. 2c is a decision rather than work.

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
`plans/roadmap.md`'s closed list. The one non-obvious part was checking that the
root *count* does not depend on the seed-scan density, because otherwise
`branch=1`
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

## 2. A real full-waveform sonic gather (F) — closed, and what it cost to learn

**This item is closed, and it was worth what this file kept saying it was
worth.** Everything below the first two paragraphs is the record of how. Of the
two consequences it left, 2b is closed and 2c is a decision.

The Utah FORGE dipole sonic arrived: a Schlumberger DSI run from well ME-ESW1,
registered as `forge_dsi_las` in `scripts/fetch_real_data.py`, with the
companion 808 MB DLIS carrying the per-receiver waveforms those picks came from
— eight receivers, 512 samples, monopole and both dipoles. `DTCO` and `DTSM`
agree between the LAS and the DLIS to 5e-5 us/ft over ~10 800 common depths, so
the data is *scoreable*: the package's own picks can be compared against a
vendor's on identical rock.

**What it found, immediately.** Shear matched `DTSM` to a median **+0.12 %**
(96 % of depths within 10 %) — the strongest external evidence this package
has. Compressional did not: 62 % of depths, with a mean 27 % high and a sharply
bimodal error. That became F.1, and it turned out to be mode confusion rather
than imprecision — `track_modes` assigning the *same* STC peak to P and to S at
143 of 400 depths. It is now fixed (`resolve_mode_collisions`, PR #76) and the
same log reads 95 %. **This is the entire argument for the item, realised in
one sitting: a defect that had shipped for the project's whole life, invisible
to every synthetic because the synthetics are generated by the forward model
the picker is scored against.**

**Two prior claims in this file were wrong, and the way they were wrong is the
lesson.** The first: "no openly redistributable full-waveform gather with
trustworthy reference picks is known to exist" — withdrawn when a search turned
up two credible sources. The second, its replacement: that Utah FORGE is
mirrored on AWS Open Data — the reachable buckets carry DAS and geophone data,
no wireline logs. Both were statements about what does not exist, and both aged
badly within a revision or two. The surviving formulation was the narrow one:
the obstacle is *which host serves a file*, never a blanket network wall. That
held. `gdr.openei.org` stayed unreachable throughout; the file arrived by
another route, and the registry entry's SHA-256 is still computed from that
copy rather than from the canonical host (item 2c below).

The other candidate is untouched and remains a lead, not a result:

- **IODP / ODP** via the LDEO Borehole Research Group — sonic waveforms for many
  holes, in DLIS plus a Python-friendly binary export, documented as eight
  waveforms × 512 samples at 10/40 µs every 15.24 cm. Licence unconfirmed;
  matters less than it looks, since the harness fetches on demand and never
  vendors. A second well would test whether F.1's repair generalises further
  than the two logging passes it has been checked on.

### 2b. A waveform path in `read_dlis` (F.3) — closed

`read_dlis` skips multi-dimensional channels by design, so `PWF1`-`PWF4` were
unreachable from the public API and every number in the paragraphs above was
obtained by calling `dlisio` directly. `read_dlis_waveforms` closes that.

The part worth keeping is where the geometry came from. The obvious route was
Schlumberger's own parameter records — `DSI4` (digitizer sample interval) and
`RX1G`..`RX8G` (receiver geometry) are all present and all correct. Using them
would have hard-coded a vendor's naming into the reader. RP66 v1 AXIS objects
carry the same information as part of the *standard*: COORDINATES and SPACING,
each with a declared unit. So the reader selects the time and offset axes by
their **unit** rather than by the AXIS-ID string, which is producer-defined,
and converts to seconds and metres from whatever the file declares.

It also corrected an assumption these notes had inherited: the hand-assembled
runs used a 2.7432 m first offset taken from the tool description, and the file
says 7.874 m. The measured agreement is unchanged, because STC slowness depends
on receiver spacing rather than absolute offset — but that is a reason the
error was invisible, not a reason it was harmless.

### 2c. A waveform fixture CI can fetch (F.2), and the checksum

Two loose ends, both small and neither purely technical:

- The waveforms live in an 808 MB DLIS inside a 471 MB zip, which is not a
  viable fetch-on-demand fixture. A small extracted subset would be — but
  hosting one is redistribution and needs a decision rather than a commit.
  Until then the 0.12 % shear result and the 95 % compressional result are
  measured but not regression-tested, and only seeded synthetics stand behind
  F.1 in CI.
- The registered SHA-256 was computed from a mirror copy because
  `gdr.openei.org` was unreachable from the session that added the entry. It is
  flagged as unconfirmed in the entry's `provenance`, and it is the one
  unverified claim in the fixture registry.

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
was re-diagnosed out of it in PR #67. The **forward model is now complete** --
elements, 11x11 assembly, and both public entry points
(`stoneley_dispersion_microannulus`, `crack_wave_dispersion`) -- so what is
left under this heading is the `sonic_ml` consumer.

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
  `plans/roadmap.md` actually wants: a debonded-regime dataset, and with it
  the first fair CBL-amplitude comparison rather than a strawman.

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
- **Sonic-gather candidates found** (PRs #56, #57) — not closed at the time, but
  item 2 moved further in these two than in anything before. Withdrew "no openly
  redistributable gather is known to exist", then withdrew the replacement's own
  error that Utah FORGE is "mirrored on AWS Open Data" (the reachable buckets
  carry DAS and geophone data, not wireline logs). Two wrong claims in
  succession on the same item is worth remembering when reading the rest of
  this file: statements about what does *not* exist are the ones that age worst.
- **The real sonic log, and the first score against a vendor** (PR #74). Item 2,
  closed. A Schlumberger DSI run from Utah FORGE ME-ESW1 is registered and
  tested; shear matches `DTSM` to **0.12 %** median on real rock. The file
  arrived by a route this file had not considered — its host stayed unreachable
  throughout — which is the fifth time an availability claim here has been
  overturned by trying rather than reasoning.
- **The compressional-pick defect: diagnosed** (PR #75). Item 2a. The
  62 %-of-depths compressional result was mode confusion, not imprecision: P and
  S claiming one STC peak at 143 of 400 depths. Reproduced on a seeded synthetic
  so the finding survives without the 808 MB fixture. The greedy picker was
  deliberately left unrepaired at that point, with the expected gain measured
  rather than guessed — which is what made the next PR a decision rather than an
  exploration.
- **The compressional-pick defect: repaired** (PR #76). Item 2a, closed.
  `resolve_mode_collisions` in `pick_modes` / `track_modes`; agreement
  62 % → **95 %** with coverage unchanged, shear bit-identical, no damage to any
  already-correct depth, confirmed on a second logging pass. Worth keeping for
  the method: **the first version of the rule was wrong and a property test
  proved it.** All 143 real collisions failed in the same direction — the shared
  peak was the shear arrival — so the rule generalised that into "the slower
  mode keeps it", which fits every observation and still breaks a slow formation
  where the collision runs the other way. `tests/test_hypothesis.py` caught it
  dropping a correct P. The shipped rule declines to choose. Written up in
  `plans/learning.md` under the failure modes, where it is the first entry
  contributed by a property test rather than by an analytic oracle.

## Recommendation

**Item 2c is closed, and with it the recommendation this file has carried
since its first revision.** A real eight-receiver DSI gather is registered,
read and exercised. What that changes is narrow and worth stating narrowly: the
processing chain is now bounded against data this repository did not generate,
where before every number in it — the 0.12 % shear agreement, the 95 %
compressional agreement, all of `sonic_ml` — was measured against the forward
model that produced its own input.

**What to do next is a genuine choice rather than a queue**, and the five
remaining rows are not comparable:

* **1b (A.2, leaky root tracking at `n=1` and `n=2`)** is the only one that is
  *modelling*. It is the last physics gap in the solver, and one fix repairs two
  modes. It no longer needs a Riemann-sheet analysis for the *whole* band:
  measured against the digitised figure 2a, a Scholte-edged bracket covers
  4.4-16.4 kHz to 0.66 %, and the sheet analysis is needed only for the two
  intervals either side. What it still needs, and what stops the bracket from
  shipping alone, is a rule for picking the fundamental out of the widened
  window. It is also the reason both Scholte ties added for A.1 are scoped to
  slow formations, so it bounds the validation work as well as the solver —
  though the fast-formation half of that tie now has published confirmation
  from the same figure.
* **3 (A.1, validation figures)** needs a book, for three figures rather than
  five. It ties the solver to published literature rather than to itself, which
  is the same class of argument F.2 just won for the processing.
* **5 (F.4, two unconfirmed checksums)** needs one fetch each from
  `gdr.openei.org` and `zenodo.org`. Trivial anywhere with ordinary egress, and
  it is the only place the fixture registry currently asserts something it has
  not verified.
* **7 (A.5 residue, delta-matrix reformulation)** is optional and blocked on
  nothing. It would raise the crack-wave ceiling above ~240 kHz.
* **4 (D, conda-forge)** needs a PyPI release and is pure packaging.

If one is to be picked on value rather than on cost, it is **1b**: it is the
only row where the answer is not already known to somebody else. If one is to be
picked on cost, it is **5**, which is minutes rather than a project — and
clearing it removes the last unverified claim from the registry, which is a
cheap thing to be able to say.

A caution carried forward from how 2c actually closed. Two of the last three
items — F.2 and F.5 — were closed by *looking again* rather than by building
anything, and both had been filed as blocked on the strength of a negative
claim nobody had re-tested. Before 1b or 3 is called blocked on "literature
access" or "the books", the same question is worth asking: has anyone checked
recently, or is that a search result from a year ago wearing the clothes of a
fact?

**That question was then put to item 3, and it shrank it by two fifths.** A.1
justified five digitised figures with "these are the only checks that tie the
solver to literature rather than to itself" — a sentence that had quietly
stopped being true as the analytic oracles accumulated. Stoneley is now tied at
1e-8 and 0.1 %, quadrupole at 1e-3, and flexural at 1e-3 after this revision,
against an overlay budget of 5 %. Three of the five figures were the weaker
instrument and are dropped. The flexural tie cost nothing but noticing that the
`n=2` argument had never been applied to `n=1`, and it exposed a test anchored
to the wrong reference with a tolerance wide enough to hide the 9 % gap.

The residue is real and does need the books: pseudo-Rayleigh's *curve*, cased
Stoneley, and VTI flexural have no external tie of any kind. But the item is no
longer blocking the package's headline mode, which is what made it feel urgent.

*Amended once the figures were actually traced.* The open-hole Stoneley,
flexural and quadrupole modes now do have external ties — Schmitt & Cheng figs
2a, 7 and 8a. The Stoneley one is at **0.04 % rms in a slow formation**, which
is tighter than most of the analytic ties this item used to prefer, and it pins
the paper's borehole radius as a by-product. The three modes above are still
untied; the difference is that "a digitised figure scores at best 5 %" is no
longer a reason to expect little from them.

*Amended again.* **Pseudo-Rayleigh is off that list.** Figure 1a plots the
Stoneley and the first two pseudo-Rayleigh modes for the fast sandstone, and
`trapped_pseudo_rayleigh_dispersion` follows both branches at **1.01 % and
0.80 % rms** — one to one-and-a-half plotted line widths — with the `branch`
index landing on the modes the API says it should. The residue is **cased
Stoneley and VTI flexural**, two items rather than three.

Item 2d is closed, and how it closed is the more useful part. It had been
filed as sourcing — "the offsets need the SDT tool spec; nothing in the files
settles the 950-A header" — and both halves of that were wrong in the same
way. The hole identity was settled by a table sitting *inside the archive*,
which matched all six logging runs by name and depth interval; the offsets
were settled by picking first breaks on the waveforms and finding the
signature the tool specification predicts, two of four paths coinciding and an
intercept of −0.0 µs. Neither needed anything the project did not already
have.

The pattern is worth naming because it is cheap to repeat: **"I do not know
this" had been written down as "the file does not say this."** They are
different claims, and the second one is a research result that needs the
archive read to the end before it can be made. Every unknown filed against an
artefact deserves one pass of *have I actually opened all of it* before it
becomes a roadmap row — and where a specification supplies a number, the
question stops being sourcing and becomes whether the data agrees with it.

The previous revision of this file recommended item 2, on the grounds that
everything in the repository was measured against the forward model that
generated its own data, and that one real gather would change what those numbers
mean. That recommendation was taken, and it was right — but the interesting part
is *how* it was right. The value did not come from confirmation. It came from
the log disagreeing: 62 % on compressional, a defect that had shipped for the
project's whole life and that no synthetic could expose, because the synthetics
are generated by the model the picker is scored against. **The argument for real
data is not that it validates; it is that it is the only thing in the repository
capable of disagreeing.** Item 2c is what keeps that capability, and it is why a
fixture matters more than the numbers already banked.

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
