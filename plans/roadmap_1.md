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
| 1 | Leaky-mode root tracking (A.2 + G.2) | modelling | nothing — this is the work |
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

**Measured consequence for the dataset**, over the generator's own cased priors
(50 draws): fast formations average 28 % band coverage (5/47 fully converged);
slow formations converge fully (3/3). With `V_S` drawn from 1200-3200 m/s
against a 1500 m/s fluid, only ~15 % of draws are slow.

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

## Loose ends from the joint-inversion work

- Whether `penalty="tv"` should be the default in `sonic_ml.models.joint` —
  deliberately unresolved, because it turns on how bedded a real target log is.
  That is item 2, not another synthetic sweep.
- Coupling across *mode* as well as depth: untouched.

## Recommendation

**Item 1 is the only substantial thing that can be built from here**, and it now
buys two roadmap entries instead of one.

If a near-term dataset improvement is wanted without that modelling work, the
bounded option is a two-mode cased dataset restricted to slow formations, with
the restriction stated in the schema rather than left implicit — roughly 15 % of
the current prior, and honest about why.

Items 2 and 3 should be queued for whoever has the books and a network. They are
worth more than anything on this list, and no amount of coding will close them.
