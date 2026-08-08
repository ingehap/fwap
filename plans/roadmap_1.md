# What remains to be done

A prioritised reading of the open items in `docs/roadmap.md`, captured after
the joint-inversion work (PRs #46-#48) landed. `docs/roadmap.md` stays the
authoritative status file; this is a snapshot of priority and reasoning at one
point in time, so check it against the tree before acting on it.

Five things remain, and they are not equally worth doing.

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
can close the gap. It also silently gates two other items below.

The blocker is genuine: no openly redistributable full-waveform gather with
trustworthy reference picks is known to exist. Finding one is a licensing hunt,
not a coding task.

## 2. Cased flexural bracketing (A.2)

The layered `n=1` solver accepts fast formations, but its root-finding stays
sparse — only a few frequencies converge for a typical casing + cement stack.
That is why the cased dataset is single-mode. Fixing it unlocks the two-mode
cased dataset (G.3), so it is one piece of numerical work with a downstream
payoff.

## 3. Validation figures (A.1)

Extend the validation notebook to the remaining reference figures (Paillet &
Cheng 1991 Fig. 4.5; Schmitt 1988 Fig. 4; Tang & Cheng 2004 Fig. 3.4). Bounded,
unglamorous, and the only checks that tie the solver to literature rather than
to itself. Fully actionable right now — the best next slice if the goal is
something that will definitely finish.

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

A.1 for guaranteed progress; A.2 to unblock the most downstream work. Item 1
dominates in value but is a data-licensing problem that cannot be solved by
writing code — though candidate datasets can be searched for and their terms
evaluated.
