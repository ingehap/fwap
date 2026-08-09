# What remains to be done

A prioritised reading of the open items in `docs/roadmap.md`, current through
PR #54. `docs/roadmap.md` stays the authoritative status file; this is a
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

**The headline, as of this revision: there is no longer a large piece of work a
coding session can carry to completion unaided.** Item 1 was the candidate;
attempting it moved the blocker from code to derivation. The bounded fallback
that stood behind it has now been built and closed. What is left is one hard
derivation and three things that need something this environment does not have.

## 1. Leaky-mode root tracking (A.2 + G.2)

Two roadmap items need the same machinery.

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

What is missing is a derivation rather than code: which Riemann sheet the `n=1`
pole occupies below the cutoff. And there may be no pole to find — the
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

## Recommendation

Nothing here can be finished from a coding session alone. Ranked by value rather
than by feasibility, items 2 and 3 dominate everything else and should be queued
for whoever has the books and a network; item 1 needs a derivation before any
code is worth writing; item 4 waits on a release.

If work continues here regardless, the honest options are small: more tests
against existing behaviour, or documentation. Both are worth less than the
sourcing work, and this file should not pretend otherwise.
