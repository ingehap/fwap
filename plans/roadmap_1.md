# What remains to be done

A prioritised reading of the open items in `docs/roadmap.md`, kept current
through PR #51. `docs/roadmap.md` stays the authoritative status file; this is a
snapshot of *priority and reasoning* at one point in time, so check it against
the tree before acting on it.

## The shape of it

Four things are open, and they fall into two kinds — which matters more than
their ordering, because only one kind can be worked on from a coding session.

| # | Item | Kind | Blocked on |
|---|------|------|-----------|
| 1 | Leaky-mode root tracking (A.2 + G.2) | modelling *and* derivation | a Riemann-sheet analysis — possibly literature access |
| 2 | A real full-waveform sonic gather (F) | sourcing | licensing + network access |
| 3 | Digitised validation figures (A.1, second half) | sourcing | access to the books |
| 4 | Conda-forge recipe (D) | packaging | a PyPI release |

Items 2 and 3 cannot be closed by writing code, and this sandbox's egress
reaches GitHub only, so neither can even be fetched from here. They are work for
someone with a network and a library.

A note on how this list shrank: A.2 was previously listed as its own item
("cased flexural bracketing"). Investigating it showed it is a symptom of the
same leaky-mode problem as G.2, so the two are now one entry. That is a real
reduction in scope, not a re-labelling.

## 1. Leaky-mode root tracking (A.2 + G.2) — the only substantial coding work

Two roadmap items turned out to need the same machinery.

**A.2, the fast-formation flexural sparsity.** A fast formation behind casing
converges over only ~38 % of a 1-12 kHz band. It was filed as a *cased-hole
bracketing* problem; it is not. Strip the casing and cement away and the
identical formation is just as sparse in an open hole, over the same
frequencies. The cause is leakage: for `V_S > V_f` the flexural root leaves the
real `k_z` axis, and the real-axis sign change the solver hunts for survives
only beside the shear branch point at high frequency. Widening a real bracket
cannot recover it — no sign change exists below the cutoff in any sub-window,
and the middle window is singular for the propagator formulation anyway. Four
tests pin the open-hole-vs-cased comparison so the attribution cannot drift
back.

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

What is missing is a derivation rather than code: which Riemann sheet the `n=1`
pole occupies below the cutoff. And there may be no pole to find — the
fast-formation flexural mode may simply exist only above its cutoff, with the
low-frequency dipole energy carried by a shear head wave. Settling that is what
Schmitt 1988 fig 4 is for, which quietly puts this item behind item 3's
literature access too.

**Measured consequence for the dataset**: fast formations average 28 % band
coverage (5/47 fully converged over 50 draws); slow formations converge fully.

*Correction.* An earlier version of this file said "only ~15 % of draws are
slow". That was measured over the **default** `FormationPriors` (1200-3200 m/s),
not the one the cased generator actually uses — `generate_cased_dataset` pins
1700-3000 m/s, so 100 % of its draws are fast. The figure described the wrong
distribution and is withdrawn; see item 5 for what replaced the conclusion.

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

The blocker is genuine: no openly redistributable full-waveform gather with
trustworthy reference picks is known to exist.

## 3. Validation figures (A.1) — half closed, half blocked

**Closed.** `fwap.validation` scores an fwap dispersion curve against a
digitised reference, and the validation notebook asserts a 5 % RMS budget per
curve — verified to fail on a 12 %-perturbed reference, so it is a real gate
rather than a described one. Most of that module is input validation, because
hand-tracing a printed figure fails in a handful of ways that all produce
plausible files (µs/ft read as s/m, a velocity axis traced as a slowness one,
kHz left unconverted); each is refused with a named diagnosis, and units are
never silently rescaled.

**Still open.** No reference CSV is shipped, so the solver is *not* yet tied to
any published figure, and the notebook says so rather than letting green plots
imply otherwise. Digitising needs the books (Paillet & Cheng 1991; Schmitt
1988/1989; Tang & Cheng 2004 figs 3.7/3.10 and 7.1). Once a CSV lands in
`docs/notebooks/_data/` under the documented name, no code changes — the section
scores and gates automatically.

## 4. Conda-forge recipe (D)

Packaging only, and unblocked once the first PyPI release is live. Reversible
and low-risk; listed for completeness rather than because it competes with
anything above.

## 5. Two-mode cased dataset — done, and narrower than expected

`generate_slow_two_mode_cased_dataset` ships a cased dataset carrying both the
Stoneley and the flexural mode, fully bound across the band.

The catch is the prior it needs. The two cased modes fail in opposite
directions — flexural is sparse in fast formations, the Stoneley stops being
bound as the formation slows away from the fluid — so the window where both hold
is `V_S` in 1420-1495 m/s, about 80 m/s wide. Measured both-modes-bound fraction
across the annulus prior: 0.00 at 1350 m/s, 0.42 at 1380, 0.92 at 1400, 1.00
from 1420 up.

That window is **disjoint from the default cased prior** (1700-3000 m/s), so
this is a different dataset rather than a subset, and the two must not be
pooled. It suits cement-bond work, where the label is the bond index and
formation `V_S` is a nuisance parameter; it is the wrong dataset for anything
needing formation-property variety. The restriction is documented at the point
of use and pinned by a test.

## Loose ends from the joint-inversion work

- Whether `penalty="tv"` should be the default in `sonic_ml.models.joint` —
  deliberately unresolved, because it turns on how bedded a real target log is.
  That is item 2, not another synthetic sweep.
- Coupling across *mode* as well as depth: untouched.

## Recommendation

Item 1 was the only substantial thing that looked buildable from here, and
attempting it moved it: the coding part is ready, the *derivation* part is not,
and it may itself depend on the literature access items 2 and 3 need. So there
is currently **no large piece of work that a coding session can carry to
completion unaided** — which is worth saying plainly rather than discovering
again.

The bounded near-term dataset improvement has now been done, and it is smaller
than this file previously implied — see item 5.

Items 2 and 3 should be queued for whoever has the books and a network. They are
worth more than anything on this list, and no amount of coding will close them.
