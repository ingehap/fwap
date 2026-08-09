# What remains to be done

A prioritised reading of the open items in `docs/roadmap.md`, first captured
after the joint-inversion work (PRs #46-#48) and updated after the A.1
validation machinery landed (PR #50). `docs/roadmap.md` stays the authoritative
status file; this is a snapshot of priority and reasoning at one point in time,
so check it against the tree before acting on it.

## 1. A real full-waveform sonic gather (F) — the one that matters

The harness shipped; adding a dataset is a one-entry change to
`scripts/fetch_real_data.py`. What is missing is the file. Neither registered
fixture is a sonic gather, so **the entire sonic processing chain is validated
only against synthetics.**

This is the binding constraint on every quantitative claim in the repo, and it
got worse when `sonic_ml` landed. The headline result — a learned inverse beats
classical STC by roughly an order of magnitude on shear velocity — is measured
on data drawn from *the same forward model that generated the training set*.
That measures identifiability, not field accuracy. No further synthetic work
can close the gap. It also silently gates items 3 and 5 below.

The blocker is genuine: no openly redistributable full-waveform gather with
trustworthy reference picks is known to exist. Finding one is a licensing hunt,
not a coding task. Note also that this sandbox's egress policy reaches GitHub
only, so even a freely-licensed file cannot be fetched from here — sourcing it
is work for a human with network access.

## 2. Fast-formation flexural leakage (was "cased flexural bracketing", A.2)

Investigated, and the diagnosis in the earlier version of this file was wrong.
The sparsity is real — a fast formation behind casing converges over about 38 %
of a 1-12 kHz band — but **the layer stack is not the cause**: strip the casing
and cement away and the identical formation is just as sparse in an open hole,
over the same lower part of the band.

The cause is leakage. In a fast formation the flexural root leaves the real
`k_z` axis, and the real-axis sign change the solver hunts for survives only
beside the shear branch point at high frequency. No widening of a real bracket
recovers it; scanning finds no sign change below the cutoff in any sub-window.
A fix means complex-plane root tracking — the same machinery item 4 (G.2) needs,
so they should be planned as one piece of work rather than two.

Measured over the generator's own cased priors: fast formations average 28 %
band coverage (5/47 fully converged), slow formations converge fully. Since the
priors put `V_S` in 1200-3200 m/s against a 1500 m/s fluid, only ~15 % of draws
are slow — so a two-mode cased dataset is reachable today only on the
slow-formation subset. That restriction, not bracketing work, is the honest
near-term option.

The comparison is pinned by tests, so the attribution cannot drift back.

## 3. Validation figures (A.1) — half closed, half blocked

**Closed:** `fwap.validation` scores an fwap dispersion curve against a
digitised reference, and the validation notebook asserts a 5 % RMS budget per
curve. Verified to fail on a 12 %-perturbed reference, so it is a real gate
rather than a described one. Most of that module is input validation, because
hand-tracing a printed figure fails in a handful of ways that all produce
plausible files (µs/ft read as s/m, a velocity axis traced as a slowness one,
kHz left unconverted); each is refused with a named diagnosis, and units are
never silently rescaled.

**Still open:** no reference CSV is shipped, so the solver is *not* yet tied to
any published figure. Digitising needs the books (Paillet & Cheng 1991; Schmitt
1988/1989; Tang & Cheng 2004 figs 3.7/3.10 and 7.1). Once a CSV lands in
`docs/notebooks/_data/` under the documented name, no code changes — the section
scores and gates automatically. Like item 1, this is now a sourcing problem
rather than an engineering one.

## 4. Free-pipe / leaky cased regime (G.2)

The cased dataset spans only the *bonded* regime, so the bond inverse grades
cement quality and is explicitly not a free-pipe detector. Reaching debonding
needs a leaky-mode cased forward model, not a planted wavetrain. It is also the
regime where a CBL-amplitude baseline would finally be a fair comparison
instead of a strawman. Largest scope of anything here.

## 5. Loose ends from the joint-inversion work

- Whether `penalty="tv"` should be the default in `sonic_ml.models.joint` —
  deliberately unresolved, because it turns on how bedded a real target log is.
  That is item 1, not another synthetic sweep.
- Coupling across *mode* as well as depth: untouched.

## Recommendation

Item 2 turned out to be a symptom of item 4 rather than a slice of its own, so
the shape of the remaining work is simpler than it looked: **one substantial
piece of modelling (complex-plane leaky-mode root tracking, covering both), and
two sourcing problems that no amount of coding will close** (a real gather, and
the digitised figures). A.1's remaining half and item 1 should be queued for
whoever has the books and a network.

If a near-term dataset improvement is wanted without that modelling work, the
bounded option is a two-mode cased dataset restricted to slow formations, with
the restriction stated in the schema rather than left implicit — roughly 15 % of
the current prior, and honest about why.
