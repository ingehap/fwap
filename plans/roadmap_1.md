# What remains to be done

A prioritised reading of the open items in `docs/roadmap.md`, current through
the leaky-mode attenuation check (PRs #61, #62 and this branch).
`docs/roadmap.md` stays the authoritative status file; this is a snapshot of
*priority and reasoning* at one point in time, so check it against the tree
before acting on it.

## The shape of it

Four things are open, and they fall into two kinds — which matters more than
their ordering, because only one kind can be worked on from a coding session.
Item 1 used to be a single row; measurement split it, and the half that came
loose (1a) turned out to be ordinary coding and is now closed. The struck-out
row is kept because the split is the useful part of the story.

| # | Item | Kind | Blocked on |
|---|------|------|-----------|
| ~~1a~~ | ~~Leaky-mode branch selection, n=0 (A.3)~~ | **closed** | — |
| 1b | Leaky-mode root tracking, n=1 (A.2 + G.2) | modelling *and* derivation | a Riemann-sheet analysis — possibly literature access |
| 2 | A real full-waveform sonic gather (F) | sourcing | fetching one **named** file from a host this sandbox cannot reach |
| 3 | Digitised validation figures (A.1, curve shapes) | sourcing | access to the books |
| 4 | Conda-forge recipe (D) | packaging | a PyPI release |

Items 2 and 3 cannot be closed by writing code here. Note the qualifier: an
earlier revision said this sandbox's egress "reaches GitHub only", which probing
disproved — the AWS Open Data S3 buckets are reachable and downloads from them
work. They simply do not host the files in question. The obstacle is which host
serves a file, not a blanket network wall.

**Item 1a came and went inside one revision, which is the point.** The revision
before last said "there is no longer a large piece of work a coding session can
carry to completion unaided". Checking the leaky-mode attenuation disproved
that: the `n=0` leaky solver returned a *different mode* depending on the
caller's frequency window and failed silently to all-NaN on merely-coarse grids.
That needed no derivation and no literature — enumerate the roots at the seed
frequency, order them by radial order, let the caller pick — and it is now
closed. The measured defects run the other way round: `branch=0` returns the
same curve for every grid top from 32 to 100 kHz, and both silent failures
recover their full bands.

So the honest headline is not about what is left but about how it keeps being
found. That was the third claim in this file about what is *not* available here
to be overturned by measuring instead of reasoning, and it was overturned by a
check aimed at something else entirely. `plans/learning.md` is the retrospective
on why; the short version is that oracles find defects at a much better rate
than tests do, and that claims about absence are the ones that age worst.

Item 2 remains the most valuable thing on the list — it went from "no such data
is known to exist" to a named CC BY 4.0 file in a format the library already
reads, and is now an errand rather than a research problem.

## 1. Leaky-mode root tracking (A.2 + G.2), and the n=0 piece that split off

Two roadmap items need the same machinery — and a third, smaller one turned out
not to need it at all, and is closed.

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

**G.2, the free-pipe / debonded regime.** The cased dataset spans only the
*bonded* regime, so the bond inverse grades cement quality and is explicitly not
a free-pipe detector. Reaching debonding needs a leaky-mode cased forward model,
not a planted wavetrain. It is also the regime where a CBL-amplitude baseline
would finally be a fair comparison rather than a strawman.

Both need complex-plane root tracking. Doing them together is the difference
between one hard piece of modelling and two.

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

After that, item 3 (the digitised figures) for whoever has the books; item 1
needs a derivation before any code is worth writing; item 4 waits on a release.

If work continues in this environment regardless, the previous revision said the
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
- **The leaky-mode attenuation against a radiation estimate** (this one).
  `leaky_radiation_attenuation` confirms the solver's attenuation to within a
  factor of two with the right radius scaling, and turned up something the
  check was not looking for: `pseudo_rayleigh_dispersion` returns a different
  mode depending on the caller's frequency window, and fails silently to
  all-NaN on grids that are merely too coarse. See item 1 below.

So the pattern held three times out of three: every analytic oracle that could
be implemented from theory found something, and two of the three found an error
rather than a confirmation. That is a better hit rate than "more tests against
existing behaviour" would have given, and it is the strongest argument for
looking for a fourth oracle rather than concluding this environment is exhausted.

This file has now twice been too pessimistic — about whether an open sonic
gather exists, and about what was left to do here. Both times the error was a
claim about absence. Worth keeping in mind when reading the four items above.
