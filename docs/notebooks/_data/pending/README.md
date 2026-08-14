# Traced but not yet scoreable

CSVs here are finished digitisations that **cannot be scored yet**. They
are deliberately not in `_data/` proper: a CSV there is picked up by
`check_overlay` on sight and asserted against the 5 % budget, so parking a
curve the notebook cannot evaluate would fail the notebook rather than
record a gap.

Nothing in the notebook reads this directory.

**It is currently empty.** Every curve that has ever been parked here has
since been promoted, and the entries below are kept as a record of what
each one was waiting for. In every case the solver was what was missing,
not the digitising -- which is the pattern worth carrying forward: a
reference finished at the fidelity of the rest of the repository is worth
committing before the code that can score it exists, because it is then
already in the tree the day that code lands.

## Contents

| File | Source | Blocked on |
|------|--------|------------|
| *(none)* | | |

## 1. ~~Fast-formation TI flexural~~ — promoted

*Closed.* `flexural_dispersion_vti` now implements the fast-formation TI
regime, so `ellefsen_cheng_schmitt_1988_fig2_flexural_vti_hard.csv` moved
up to `_data/` and scores **0.55 % RMS over 20 of 73 points**. It was
parked here from the day it was traced; nothing about the reference
changed, only the solver.

Its coverage is thin for the same reason its equivalent-isotropic sibling
is (17 of 73): the fast-formation marcher returns `NaN` outside a narrow
band rather than a wrong root. That was predicted in this file before the
solver existed, and it held.

## 2. ~~Cased-hole pseudo-Rayleigh, two branches~~ — promoted

*Closed.* `trapped_pseudo_rayleigh_dispersion_layered` now exists, so both
Tubman fig 4(b) curves moved up to `_data/` and score **3.12 %** and
**3.84 %** RMS. As predicted here, it needed a new public function rather
than a new argument to an existing one.

What was not predicted here is *why* it was hard. It needed a **complex**
n=0 cased determinant, because the real one refuses twice over this
window — once for the oscillatory fluid, and once because the cement's
`V_S` (1728 m/s) sits below the modes' 1906-2464 m/s. Neither refusal
means the mode is unbound: only the formation half-space governs that,
and it stays evanescent throughout.

## 3. ~~Slow-formation leaky compressional~~ — promoted

*Closed.* `leaky_compressional_dispersion` now exists, so
`sinha_asvadurov_2004_fig11a_leaky_compressional_slow.csv` moved up to
`_data/` and scores **0.03 % RMS over 107 of 107 points** — the tightest
tie in the repository.

**Writing the solver exposed a defect in code that already existed**,
which is the thing this entry is worth keeping for. The mode radiates
shear, so it uses `_k_or_hankel`'s leaky branch, and that branch was
`-K_n(z) + i pi (-1)^n I_n(z)` — a solution of the modified Bessel
equation, but decomposed against the Hankel pair it is
`(pi/2) i [H1 + 2 H2]`, two thirds **incoming**. It passed every test it
had, because a solution of the same equation satisfies all of them.

The first scoring run against this curve gave 0.39 % RMS with a spurious
±0.8 % sawtooth breaking down every ~2.6 kHz, and a root at
`Im(k_z) < 0` — growing along the borehole. Correcting the sign gave
0.02 % RMS, smooth, and moved the root into the upper half plane. Neither
symptom is visible without a figure to score against: this is what a
parked reference buys.

## 4. ~~Paillet & Cheng 1986 fig 12(a) — leaky compressional, two branches~~ — promoted

*Closed.* Both curves moved up to `_data/`. The fundamental scores
**1.81 % over 136 of 170 points** and the first mode **0.35 % over 65 of
84**, using `leaky_compressional_dispersion` with `tool_radius=0.05`.

Both of this entry's blockers are gone: the tool geometry arrived first,
the mode second. The pair is now the **only external evidence the rigid
logging tool has** — the same fundamental scores 10.66 % with the tool
left out, so the 5 cm inner boundary is doing real work.

Two things recorded here held up and are worth repeating.

**Its fidelity really is lower than the rest.** It is a raster scan of a
1986 journal page: the y axis is good to 0.25 % against the figure's own
dotted lines, but the x axis has six unevenly spaced ticks whose
least-squares residuals reach 0.36 kHz, about **1.4 % of full scale**.
The fundamental's 1.81 % is close to the floor this reference can
support, and the first mode's 0.35 % is better than the axis deserves.

**The phase/group pairing check was right.** With no score available,
the assignment was confirmed by differentiating the phase curve and
recovering its group partner — an Airy minimum of 1.207 km/s at 12.1 kHz
against 1.235 at 10.7 read off the traced group curve. The subsequent
score agrees with that identification.

What this entry did *not* anticipate: the paper's "first mode" is
`branch=2`, not `branch=1`. Branch indices order by `Re(k_z)` alone, and
at 24.8 kHz a cut-off mode two orders more attenuated than its
neighbours sits between the two propagating branches. The first mode
also stops at its own cut-off near 16 kHz, where `Im(k_z) -> 0` and the
phase velocity reaches `V_P`, while the traced reference continues below
that along the figure's C line — which is why 65 of 84 points are scored
rather than all of them, and why the fastest traced point reads 0.5 %
faster than `1/V_P`. The CSV is still shipped as traced rather than
clipped to the physical bound.
