# fwap roadmap

Open items that would meaningfully extend fwap beyond the 0.4.0 release.

This file supersedes `docs/roadmap.md` (latterly `docs/roadmap_old.m`), which
carried the same open items buried in about nine hundred lines of closed ones.
The closed material is not reproduced here — `CHANGELOG.md` is the record of
what shipped and when, and the deleted file remains in git history for anything
this merge dropped. See "Closed, and where the detail lives" at the end for the
map.

Two companion files:

* `plans/roadmap_1.md` — a *prioritised* reading of the same items, with the
  reasoning about what can and cannot be worked on from a coding session. This
  file is status; that one is priority.
* `plans/learning.md` — method, not status: what the analytic-oracle programme
  taught about choosing the next piece of work.

Section labels (`A.1`, `A.2`, `A.5`, `D`, `F`, `G`) are load-bearing. Code
comments in `fwap/`, `scripts/` and `tests/` cite them, so they are kept
verbatim across this merge rather than renumbered.

## Where things stand

Most of what the original roadmap was written to track has shipped. The book's
four Parts and the extension layer were complete by the end of the post-0.4.0
cycle; the cylindrical-Biot solver family has since closed out through leaky
modes, quadrupole, layered / cased-hole and VTI; and a machine-learning layer
that was not contemplated when the file was written now sits alongside the
package.

**The headline for this revision is that four items closed without anyone
building what they asked for, because the claim blocking each one was wrong.**
That is not a run of luck; it is a pattern this file now has enough instances of
to name.

* **F.2** asked for an openly redistributable full-waveform sonic gather, and
  two docstrings recorded that none was known to exist. A **CC0** eight-receiver
  Schlumberger DSI run had been in a public archive the whole time. It is now
  registered as `iodp_u1347a_dsi`, read by `read_ldeo_waveforms`, and `stc`
  returns **0.948** median peak coherence on it.
* **F.5** asked for two facts "not resolvable from the files". Both were
  resolvable — one from a run table inside the archive nobody had opened, the
  other from first-break moveout on the waveforms themselves.
* **A.1** justified five digitised figures as "the only checks that tie the
  solver to literature rather than to itself", a sentence that had stopped being
  true as the analytic oracles accumulated. Three figures were dropped, then one
  restored on a second look. Then the figures were actually digitised, and the
  re-scope's *own* premise turned out to be the shaky one: it dropped them by
  comparing a 1e-3 analytic tie against a 5 % overlay budget, and the budget was
  a choice rather than a limit — a carefully read figure ties the Stoneley at
  **0.04 %**, and unlike the analytic ties it is external.
* **A.2** said "a fix means complex-plane root tracking". Measuring instead of
  reading showed the larger half needs no complex machinery at all — and turned
  up something nobody was looking for: in fast formations above ~15 kHz
  `flexural_dispersion` returns a flexural **overtone** labelled as the mode.
  Not a missing answer, a wrong one. Then the published curve was digitised and
  corrected the *correction*: on the paper's own rock there is no frequency at
  which the solver returns the flexural mode at all, and the real-axis residue
  is two intervals rather than one.

The common shape: **a negative claim about the outside world, or about what a
file contains, written down once as a fact and never re-tested.** Stated flatly
in prose or in a docstring it forecloses the re-check that would overturn it.
F.2 sat blocked for the project's entire life behind one such sentence.

What has genuinely been *built* rather than re-measured, in the same span: a
learned microannulus inverse at **2.5 %** held-out and its benchmark harness
(G.2, G.6), a reader for the LDEO waveform format, and an analytic tie for the
flexural mode at **1e-3**.

Real data now bounds the processing chain. It does not bound `sonic_ml`, whose
numbers are still measured against the forward model that generated their
training data — one hole of one tool does not change that.

| Open item | Why it matters |
|-----------|----------------|
| ~~**F.2 A waveform fixture CI can use**~~ | *Closed, with one thing it does not mean.* `iodp_u1347a_dsi` — an eight-receiver DSI monopole run, **CC0** on Zenodo — is registered, read by `read_ldeo_waveforms`, and exercised by `stc` at **0.948** median peak coherence. CI *can* use it but does not: the default run stays hermetic and skips it. The blocking claim ("no openly redistributable gather is known to exist") turned out to be false. Kept below. |
| ~~**F.5 The ODP file's unknowns**~~ | *Closed.* Both answered, and the item was filed on a wrong premise — "not resolvable from the files" meant "nobody had opened the rest of the archive". The hole identity is settled by the archive's own run table (six runs matched by name and depth interval, six for six); the offsets are the LSS **8/10/10/12 ft** set, confirmed by first-break moveout to an intercept of −0.0 µs. Kept below. |
| **A.2 Fast-formation leakage, `n=1` and `n=2`** | **Measured against the published curves for four rocks, and it is two defects.** Figures 2a and 7a plot this exact quantity for stated formations. Digitised, the flexural branch runs `V_S` → Scholte and crosses `V_R` at **4.43-4.45 kHz in all three fast rocks** — although `V_R` spans 2413 to 3388 m/s. The solver's `(V_R, V_S)` window therefore empties at the same frequency whatever the formation, and holds the true root over **10 %** of the band. *Wrong answers*: overtones labelled as the mode, sawtoothing where a guided mode cannot, and **worse the harder the rock** — +62 % median in the fast sandstone, +72 % in limestone, **+134 % in granite**, which returns nothing at all across the 3-10 kHz band figure 7a resolves. *Missing answers*: at both ends of the band no real-axis root exists. Moving the bracket edge to the Scholte speed recovers **4.4-16.4 kHz at 0.66 % median error** and nothing outside it, so the fix is scoped rather than guessed: a bracket edit for the middle, complex `k_z` for the two ends. Pinned by fourteen tests, not fixed. **`n=2` checked against figure 7b too, and it is the worse of the two**: same mechanism and stiffness ordering, but 65-75 % coverage against `n=1`'s 21-36 %, so a `NaN` filter keeps two to three times as many wrong answers. One fix repairs both. |
| **A.1 Validation figures** | *Re-scoped from five to three* (five → three → four → three: fig 4 was restored, then the figure itself was seen and turned out **not to be a dispersion figure at all** — it is a dipole shot gather, so it cannot be scored in the overlay schema. It did settle A.2's yes/no question, which is why it was worth fetching). Which figure carries the flexural dispersion curves is now known — **figure 2a of Schmitt & Cheng**, since digitised, which also gives A.1's flexural-Scholte tie its first *external* confirmation (1493 m/s read at 24.9 kHz against 1484 computed). **The "figures are the weaker instrument" premise is now measurably wrong.** It compared a 1e-3 analytic tie with a *5 % overlay budget* — but the budget was a choice, not a limit of the method. Digitised carefully, figure 8a ties `stoneley_dispersion` to a published slow-formation curve at **0.04 % rms, below what the figure can resolve**, which is the project's first external tie better than 1 %, and it pins the borehole radius as a by-product. The analytic ties are tighter numerically and are not external at all. What still needs the books is the **pseudo-Rayleigh curve**, **cased Stoneley** and **VTI flexural**, which have no external tie of any kind. |
| **F.4 Two unconfirmed checksums** | Also body-only until this revision, which is the same failure as A.2's. `forge_dsi_las` and `iodp_u1347a_dsi` both carry digests computed from copies that did not come down their canonical URLs, because those hosts were blocked from the sessions that added them. One successful fetch each clears it. It is the only place the fixture registry asserts something it has not verified. |
| **A.5 residue: delta-matrix reformulation** | Optional and blocked on nothing, which is why it kept falling off the list. A delta-matrix / Abo-Zena form of the elastic stack would remove the cancellation the grid-stability filter exists to work around, and raise the crack-wave ceiling above the ~240 kHz where the propagators stop being representable. |
| **D. Conda-forge recipe** | Packaging only; unblocked once a PyPI release is live. |
| ~~**G.6 The debond inverse in the benchmark harness**~~ | *Closed.* `sonic_ml.bench.debond` scores both rivals on identical held-out indices. It paid for itself immediately: the per-regime rows show the closed form is **6× worse on wide gaps than tight** (16.5 % vs 2.5 %), which the single averaged number had hidden. Kept below. |
| ~~**F. A real sonic log**~~ | *Largely closed.* A Schlumberger DSI log is registered and tested; the package's shear picks match the vendor's to **0.12 %** median on real rock. |
| ~~**G.2 Debonded regime**~~ | *Closed.* Generator, closed-form baseline (**18.1 %** in gap width) and learned residual inverse (**2.5 %** held-out) are all on `main`. Kept below for the measurements, which reshaped the item, and for what the result does *not* claim. |
| ~~**F.3 A waveform path in `read_dlis`**~~ | *Closed.* `read_dlis_waveforms` reads a multi-dimensional channel and recovers the sample interval from the RP66 AXIS records, falling back to a vendor parameter for files that declare none. |
| ~~**F.1 The compressional-pick defect**~~ | *Closed.* It was mode confusion, not imprecision. `track_modes` and `pick_modes` now refuse to assign one arrival to two modes (`resolve_mode_collisions`); vendor agreement went 62 % → **95 %**, with shear bit-identical and nothing dropped. Kept below for the reasoning and the residual limits. |
| ~~**A.5 Fluid microannulus**~~ | *Forward model complete.* Elements, assembly and both public APIs are on `main`; kept below for the reasoning and the measured limits. |

A note on how this file is kept honest: items are marked closed only when the
code and its tests are on `main`, and status claims are checked against the tree
rather than against memory. The old roadmap carried a "leaky modes are still
open" note for some time after the leaky solvers had actually shipped; that is
the failure mode this heading exists to prevent.

**That guard was one-directional, and every failure since has gone the other
way.** It stops an item being marked closed early. It does nothing about an item
that stays open after shipping, or one that never gets a row at all — and an
audit of this file found four at once: `A.2` was discussed at length below and
tracked in `roadmap_1.md` but had **no row in the table above**, so a skim said
the last physics gap was closed; `F.4` claimed to be "the one unverified claim"
after a second appeared; and both `A.5`'s and section G's "what is left" lists
still named work that had shipped as G.2 and G.6. Three of those four say
*there is work here* when there is not, which is the more comfortable error and
therefore the one that survives longer — nobody is embarrassed by a roadmap that
under-claims.

Two rules follow, and they are cheap. **Read the table against the body**, not
just each row against the tree: three of the four were visible only in the gap
between the two. And when a section says "what is left", **check every bullet on
every pass**, because a list that was right when written is the easiest kind of
sentence to stop reading.

**A third rule, which the two above did not prevent and which cost five more
instances to learn: correct at the site of the claim, then explain below it.**
Every one of those five was a correction written *beneath* a stale sentence
rather than applied *to* it — an A.2 table row with new text prepended to
contradictory old text, an original A.2 entry still ending "a fix means
complex-plane root tracking" with the correction in a different section, and
A.1 still opening with "the only checks that tie the solver to literature" while
its own re-scope seventy lines down said otherwise. Each *reads* as
conscientious: the correction exists, it is dated, it is argued. But a reader
meets the original sentence first and has nothing to tell them it has been
overturned, so the stale claim keeps working exactly as before.

Appending is not correcting. The fix is mechanical — strike the sentence where
it stands, or hang a blockquote off it saying which half survives — and the tell
that it was skipped is a correction whose text describes a claim the reader has
not yet been warned about.

## A. Cylindrical-Biot dispersion solver

**Shipped.** Plan items A through H in `plans/cylindrical_biot.md` are closed:
bound-mode `n=0` Stoneley (`stoneley_dispersion`) and `n=1` flexural
(`flexural_dispersion`); leaky modes at `n=0`, `n=1` and `n=2` with complex-`k_z`
root finding, outgoing-wave boundary conditions and branch tracking across the
leaky cutoff, carrying a spatial attenuation rate alongside the phase slowness;
quadrupole (`quadrupole_dispersion`), bound and leaky; layered / cased hole over
a `BoreholeLayer` stack (`stoneley_dispersion_layered`,
`flexural_dispersion_layered`, `quadrupole_dispersion_layered`), including
fast-formation cased flexural; VTI formations (`stoneley_dispersion_vti`,
`flexural_dispersion_vti`); and trapped pseudo-Rayleigh modes
(`trapped_pseudo_rayleigh_dispersion`).

The phenomenological models stay shipped
(`fwap.synthetic.dipole_flexural_dispersion`,
`fwap.cylindrical.flexural_dispersion_physical`) for callers who want a
closed-form smoothed-step dispersion curve without solving the determinant per
frequency.

What is still open here is narrow: items **A.1** and **A.2** below, plus the
optional delta-matrix residue of **A.5** — whose forward model is complete and
struck in the table above, so the bare "A.5" this line used to carry overstated
it.

### A.1 Validation-figure coverage

Plan item I, marked partial. ~~These are the only checks that tie the solver to
literature rather than to itself~~ — **that was true when written and is not
now**; see "The item was over-scoped" below, which is the correction this
sentence used to sit seventy lines above without carrying any mark of. The
analytic ties *are* checks against literature, several of them tighter than a
traced figure can score. What is true is the split: the analytic ties are done,
the digitised figures are not.

> *Corrected again by measurement.* "Tighter than a traced figure can score" was
> itself an assumption, carried over from the 5 % overlay budget. Traced with
> care, figure 8a scores `stoneley_dispersion` at **0.04 % rms** — below what
> the figure resolves, and tighter than four of the six analytic ties. It is
> also the only *external* number in the list: an analytic tie compares fwap to
> a formula fwap evaluates. The split that actually matters is not
> tight-vs-loose, it is internal-vs-external. See the figure-8a section under
> A.2.

**Six analytic ties are described below**, and the two most recent are the ones
the re-scope below rests on — so they are written out here rather than only in
its mode table, which is where they lived for a revision. The table carries one
more that has no prose entry, the flexural long-wavelength limit `1/V_S`
(Ellefsen-Cheng-Toksöz), so the total is **seven**. That discrepancy is stated
rather than left for a reader to trip over: a count in prose and a count in a
table drifting apart is how this section went wrong the first time.

*Flexural and quadrupole, at the high-frequency end.* Both approach the same
plane-interface Scholte speed the `n=0` Stoneley does, because at short
wavelength the borehole wall looks flat to *every* azimuthal order. Checked to
**1e-3** against `scholte_speed`, and all three azimuthal orders agree with each
other at 400 kHz to **6e-6** — which no per-mode check can see, since a branch
error in one order shows up only as a disagreement between them. Slow formations
only: in fast ones `n=1` and `n=2` both go leaky, which is A.2.

The four that came earlier:

*Scholte, at the high-frequency end.* The validation notebook's section 6 checks
the cylindrical Stoneley solver against `fwap.scholte_speed`, which solves the
classical secular equation for an interface wave on a **plane** fluid/solid
boundary — a different equation, with no Bessel functions and no borehole radius
in it. As the wavelength shortens the borehole wall looks flat, so the two must
agree; they do, to better than 0.1 % at 400 kHz, converging monotonically and
from opposite sides in fast and slow formations. The oracle is itself validated
by its light-fluid limit, where it collapses to the Rayleigh equation and
reproduces `rayleigh_speed` — a third, independent implementation.

*The rigid-pipe pseudo-Rayleigh cutoff, and a correction it produced.* The
formula had sat in `_leaky.py` unchecked, with a docstring recommending it as a
guard on the requested frequency band. Comparing it against the solver splits in
two: the geometric `1/a` scaling is reproduced to about 1 part in 300 over a
3.3x range of radius (pinned by a test, and enough to catch a radius/diameter
confusion), but as an *absolute* cutoff it overshoots by ~2.8x, so the
documented use would have discarded a valid band. The docstring is corrected.
The offset is not a constant that could be folded in — it varies strongly with
formation velocity, and for some parameter sets the marcher's termination
frequency is not stable at all, which is now recorded as a caveat on reading the
`NaN` boundary as physics.

*The leaky modes' attenuation.* The `attenuation_per_meter` field had tests
proving it was present, finite and positive, but nothing checking its *size*
against any independent physics. `fwap.leaky_radiation_attenuation` supplies
that: a leaky mode is a fluid wave bouncing wall-to-wall through the borehole
axis and shedding energy into the shear wave it radiates, giving
`Im(k_z) = -ln|R| k_f / (2 a k_z)` from the textbook plane-wave fluid/solid
reflection coefficient alone — no Bessel functions, no modal determinant. Over
4-30 kHz, radii 0.07-0.15 m and fast formations with `V_S` 1700-2800 m/s the
solver-to-estimate ratio stays inside 0.37-1.91, and the median is 0.57-0.71 in
*every* case. Two things follow. The scale and the geometry are confirmed: the
residual scatter is an oscillation whose peak spacing satisfies
`spacing * a = const` to about 6 %, which is the same `2a` transverse round trip
the estimate assumes, recovered independently from the solver's own output. And
there is a stable systematic offset near 0.6 that no derivation here accounts
for; it is reported rather than folded into the formula, since an empirical
constant would convert an oracle into a fit. This is an order-of-magnitude and
scaling check — it would catch a wrong power of frequency or a radius/diameter
confusion, not a 30 % error.

*The tube wave, closing the other end of the Stoneley curve.* `scholte_speed`
ties the solver's `f -> infinity` limit; `tube_wave_speed` ties `f -> 0`. The
White (1983) closed form `S_T^2 = 1/V_f^2 + rho_f/mu` matches the modal
determinant's low-frequency root to 1.3e-8-1.5e-7 relative across five media,
and the radius-independence it predicts — no `a` appears in the formula — holds
across `a` = 0.05-0.30 m to 5e-8, which is the sharper of the two checks.

**Its independence is qualified**, unlike Scholte's. The formula is already used
inside `_stoneley_kz_bracket` to place the solver's search bracket, so a check
routed through `stoneley_dispersion` would be partly self-confirming. The tests
locate the root by scanning 40x wider than that bracket instead, and both the
docstring and `plans/learning.md` record this as a weaker tie rather than
presenting it as a clean one.

*What it found.* A validity floor that was not written down anywhere: a tube
wave is a bound mode, so `V_S > V_f sqrt(1 - rho_f/rho)`, equivalently
`S_ST < (1/V_f) sqrt(rho/(rho - rho_f))` on the measured slowness. Below it no
bound Stoneley root exists — verified by scanning the determinant far outside
the solver's own bracket, not merely by observing NaN — and the closed form
predicts where the solver stops converging to within 1 % across seven media with
floors from 960 to 1255 m/s. For brine in a 2200 kg/m^3 formation that floor is
1108 m/s, which sits inside the operating range of
`vs_from_stoneley_slow_formation`, the package's primary slow-formation `V_S`
estimator. It is now documented there, in terms of the slowness a caller
actually measures, and deliberately not enforced — a noisy pick belongs in QC
rather than hard-failing a log.

**The machinery is done.** `fwap.validation` scores an fwap curve against a
digitised reference and the notebook asserts a 5 % RMS budget per curve,
verified to fail on a 12 %-perturbed reference. Most of that module is input
validation, because hand-tracing a printed figure fails in a handful of ways
that all produce plausible files (µs/ft read as s/m, a velocity axis traced as a
slowness one, kHz left unconverted); each is refused with a named diagnosis, and
units are never silently rescaled, since a reference adjusted to fit would agree
with a wrong solver too.

**The item was over-scoped, and re-measuring shrank it.** It used to ask for
five digitised figures and to justify them with "these are the only checks that
tie the solver to literature rather than to itself". That sentence stopped being
true as the analytic ties accumulated, and nobody re-read it against them.

*Correction, one revision later.* This re-scope dropped **Schmitt 1988 fig 4**
outright. Wrong for its fast-formation half: the new flexural tie is
slow-formation **only**, because fast formations are exactly where `n=1` goes
leaky, and A.2 records that this figure is what would settle whether a leaky
continuation exists at all. **Fig 4 is back in the ask, for its fast curve**;
the slow half stays superseded. The count is four figures, not three. This was
the "read the table against the body" failure the honesty note warns about,
committed by the session that wrote the note.

> **Correction to the correction: fig 4 is not a dispersion figure, and this
> file has described it wrongly throughout.** The figure has now been seen. It
> is a dipole **shot gather** — waveform traces against time at 14 receiver
> offsets, `r` = 2.40-5.00 m — in a **fast sandstone**, with panel (a) at a
> 1 kHz source centre frequency and panel (b) at 6 kHz.
>
> Every part of the repository's attribution was wrong: not a dispersion curve
> but a time-domain gather; not two formations (a slow shale and a fast
> limestone) but one; and the two panels are two *source frequencies*, not two
> formations. It therefore **cannot be digitised into the
> `freq_hz, slowness_s_per_m` schema at all** — the ask was never going to work
> as written, whoever fetched the paper.
>
> It is still worth having, for a different reason than the one recorded: it
> settles A.2's yes/no question. See the A.2 section.
>
> **A.1's remaining ask is therefore three figures, not four**, and gains an
> unknown: which figure in Schmitt (1988) carries the flexural dispersion
> curves is no longer known. It is not fig 4. Nobody should go looking for
> "fig 4" again on this file's say-so.
>
> *That unknown lasted one revision.* It is **figure 2a of Schmitt & Cheng**,
> and it has since been digitised — see the figure-2a section under A.2. The
> ask stays at three figures, because fig 2a was spent on A.2 rather than
> added to A.1's overlay set.
>
> This is the third thing this file got wrong about one paper — pages, title,
> and now figure content — and all three were recorded with the same
> confidence. The common cause is that none had been checked against the
> paper itself.
>
> **Then the paper arrived and there were two of them.** The document that
> contains this fig 4 is *Schmitt, D. P., & Cheng, C. H., "Shear Wave Logging
> In (Multilayered) Elastic Formations: An Overview", MIT Earth Resources
> Laboratory, pp. 213-246* — two authors, different title, different venue,
> **and its own figure numbering**. The JASA article this file cites is the
> single-author *Shear wave logging in elastic formations*, 84(6), 2215-2229.
> They are closely related — same method, near-identical abstract — but they
> are not the same document, and figure numbers do not carry across. The
> repository has been citing one and numbering figures from the other.
>
> **Its table 1 also has no "shale".** The notebook cites "Schmitt 1988,
> table 1, 'shale'" at 2740/1280/2400 and "'limestone'" at 4900/2840/2700.
> The actual table has no shale at all, and its limestone is
> **5081/2771/2160**. The real entries are:
>
> | layer | `alpha` m/s | `beta` m/s | `rho` kg/m³ | `Q_alpha` | `Q_beta` |
> |---|---|---|---|---|---|
> | water | 1500 | 0 | 1000 | 30 | — |
> | fast sandstone | 4878 | 2601 | 2160 | 60 | 60 |
> | invaded zone | 4390 | 2341 | 2360 | 40 | 40 |
> | limestone | 5081 | 2771 | 2160 | 60 | 60 |
> | granite | 5881 | 3750 | 2160 | 60 | 60 |
> | slow sandstone | 2751 | 1201 | 2100 | 50 | 50 |
> | invaded zone | 2338 | 1081 | 2000 | 40 | 35 |
> | casing | 6098 | 3354 | 7500 | 1000 | 1000 |
> | cement 1 | 2823 | 1729 | 1920 | 40 | 30 |
> | cement 2 | 2823 | 1555 | 1730 | 40 | 30 |
>
> **The correct figure list** (Schmitt & Cheng, ERL):
>
> 1. monopole, fast sandstone — Stoneley + first two **pseudo-Rayleigh** modes
> 2. dipole, fast sandstone — **flexural** + first trapped mode
> 3. dipole, fast sandstone — source-centre-frequency effects at 5 m
> 4. dipole, fast sandstone — shot gathers at 1 and 6 kHz *(the one supplied)*
> 5. quadrupole, fast sandstone — screw + first trapped mode
> 6. quadrupole, fast sandstone — shot gathers at 1.5 and 6 kHz
> 7. **flexural and screw dispersion for granite, limestone and fast sandstone**
> 8. slow sandstone — Stoneley, flexural and screw
> 9. dipole, slow sandstone — source-centre-frequency effects
>
> **This changes A.1's residue.** Figure 1 is a **pseudo-Rayleigh dispersion
> curve for a fast sandstone** — one of the three overlays A.1 still wants, and
> it is in a document already in hand. Figures 2 and 7 are the fast-formation
> flexural curves A.2 needs. Only **cased Stoneley** and **VTI flexural** now
> have no identified source.

Mode by mode:

| Mode | f → 0 | f → ∞ | Tightness |
|---|---|---|---|
| Stoneley `n=0` | tube wave (White 1983) | `scholte_speed` | 1e-8 / 0.1 % |
| Flexural `n=1` | 1/V_S (Ellefsen-Cheng-Toksöz) | `scholte_speed` | 1e-3 |
| Quadrupole `n=2` | — | `scholte_speed` | 1e-3 |
| Leaky | — | radiation attenuation | order of magnitude |
| Pseudo-Rayleigh | — | cutoff `1/a` scaling only | curve untied |
| Cased Stoneley | — | — | **untied** |
| VTI flexural | — | — | **untied** |

A digitised overlay is scored against a **5 % RMS** budget, and that budget is
loose on purpose because tracing a printed log-axis figure costs a couple of
percent by itself. So for Stoneley, flexural and quadrupole an overlay is three
to seven orders of magnitude weaker than the tie already in place: it cannot
fail unless the solver is catastrophically broken, in which case the tie fails
first and louder. **Figures 4.5 (Stoneley half), 3.7/3.10 and Schmitt 1988
fig 4 are dropped from the ask.**

**The flexural row is new, and it was free.** The argument the `n=2` block
rests on — at short wavelength the wall looks flat to *every* azimuthal order —
had never been applied to `n=1`, the mode the package sells. Measured on a slow
formation at `a` = 0.10 m, flexural velocity over the plane Scholte speed runs
1.0166 → 1.00025 across 10–400 kHz, monotone, and all three azimuthal orders
agree at 400 kHz to 6e-6.

*What that exposed.* `test_flexural_high_f_slowness_above_inverse_rayleigh`
was anchored to `rayleigh_speed` with `rel=0.10`. The flexural mode does **not**
approach the vacuum-loaded Rayleigh speed — it settles at 0.908 V_R and stays
there — so the tolerance was absorbing a 9 % reference error rather than
bounding the solver, and the test was passing on 17 % of its own margin. Its
docstring named the right target ("positive **Scholte** / fluid-loading
offset") and used Rayleigh as a proxy because `scholte_speed` did not exist
yet. It does now, and it was already wired to two other modes. The test keeps
the inequality, which is real physics, and the quantitative claim moved to the
Scholte check.

*Scoped to slow formations, deliberately.* In fast ones `n=1` is leaky and the
real-axis search returns scatter or `NaN` — roadmap A.2, and the same failure
the `n=2` block already records.

**What is genuinely left, and it does need the books.** Three overlays, none
of which any analytic oracle reaches: the **pseudo-Rayleigh curve** (Paillet &
Cheng 1991 fig 4.5 — its cutoff has a `1/a` scaling check, the dispersion does
not), **cased-hole Stoneley** (Tang & Cheng 2004 fig 7.1) and **VTI flexural**
(Schmitt 1989 fig 5). Cased Stoneley may yield to White's tube-wave formula
generalised with the casing and cement compliances in series, which would be a
derivation rather than a lookup; nothing comparable suggests itself for the
other two. Once a CSV lands in `docs/notebooks/_data/` under the documented
name, no code changes: the section scores and gates automatically.

Note the figure numbering: this list previously cited "Tang & Cheng 2004
Fig. 3.4", which does not match the notebook's sections (figs 3.7 and 3.10 for
quadrupole, 7.1 for cased Stoneley). The notebook is the accurate list.

### A.2 — measured, and it is two defects rather than one

**The diagnosis below is incomplete, and the correction matters more than the
detail.** A.2 says "a fix means complex-plane root tracking". Measuring the
fast-formation `n=1` solver instead of reading it shows that a large part of the
failure has nothing to do with leakiness, and the more serious part is not a
coverage problem at all.

`_flexural_dispersion_fast_formation` searches phase velocity in `(V_R, V_S)`.
**`V_R` is not a limit of this mode.** The flexural branch asymptotes to the
*Scholte* speed — the same result A.1's new tie rests on — and Scholte is far
below `V_R`: 1472.6 against 2115.8 m/s for `vp/vs/rho = 4000/2300/2500`. So the
window excludes 30 % of the velocity axis, and the fundamental leaves it
entirely at about 15 kHz.

**Defect 1 — truncation.** Below the crossing the mode is found; above it the
window no longer holds the fundamental. Continuing the branch from 2138.1 m/s at
14.5 kHz, where the current bracket still works and only one root exists, gives
a monotone curve running to 1525.2 m/s at 59.5 kHz — within 3.6 % of Scholte,
still descending. None of it is reachable through the present bracket. Widening
to `(Scholte, V_S)` roughly doubles coverage on three fast rocks over the
1-12 kHz band A.2 complains about: **32 → 72 %, 16 → 40 %, 20 → 68 %**.

**Defect 2 — the window still returns something, and it is the wrong mode.**
Overtones enter near `V_S` as frequency rises, so `(V_R, V_S)` is not empty above
the crossing; it holds an overtone. At 19.5 kHz the determinant's roots over
`(Scholte, V_S)` are 1853 and 2269 m/s. The fundamental is 1853. **The solver
returns 2269**, labelled as the flexural mode, with nothing to indicate it is a
different branch. Over 10-30 kHz the returned velocity descends 2295 → 2162,
goes `NaN`, **jumps back up** to 2283, descends to 2145, goes `NaN`, jumps to
2275. A guided mode never speeds up with frequency: this is not a sparse curve,
it is a sawtooth stitched from successive overtones. A caller interpolating
across the gaps gets a plausible-looking answer assembled from different modes.

That is the same hazard the `n=2` block records — finite values a `NaN` filter
keeps — with the mechanism now identified.

**Why this is not fixed here.** Widening the bracket is one line; identifying
the fundamental among several roots is not, and both naive rules fail. Taking
the highest root seeds onto an overtone. Taking the lowest is non-monotone on
two of three test rocks. Seeding at cutoff and marching upward with a
"velocity must decrease" guard is monotone on all three but drops coverage to
4/28, 16/28 and 10/28 — the guard cannot recover once a step is missed. A
correct fix needs a mode-identification criterion rather than a bracket edit,
and shipping a half-validated change to a physics solver is worse than shipping
the measurement. Three tests pin the defect in
`tests/test_cylindrical_solver.py` so a fix shows up as a failure.

**What survives of the original diagnosis.** The below-cutoff sparseness — the
`NaN`s under about 10 kHz in the rock above — is untouched by any of this, and
is still the leaky problem described below. Complex-plane tracking is needed for
*that* half. It is not needed for either defect above.

> *Corrected by the figure-2 measurement below.* "That half" is not one half.
> The real axis fails at **both** ends of the band, not only under the cutoff:
> above the frequency where the branch drops below the fluid velocity there is
> likewise no `Im(det)` sign change. Complex `k_z` is needed for two disjoint
> intervals with a working middle between them, which is a different — and
> smaller — job than the sentence above implies.

*A correction to A.1, made in the same revision that created it.* The A.1
re-scope dropped Schmitt 1988 fig 4 as "superseded" on the strength of the new
flexural-Scholte tie. That tie is **slow-formation only**, because of exactly
the leakiness this item is about. The figure's fast-formation curve is not
superseded, and the paragraph below explains why it is load-bearing: it is what
would settle whether a leaky continuation exists at all. Fig 4 is restored to
the A.1 ask for its fast half. Dropping it was the "read the table against the
body" failure that the honesty note added two revisions ago exists to catch —
committed by the same session that wrote the note.

> *Superseded, twice.* Fig 4 turned out to be a shot gather rather than a
> dispersion figure, and the fast-formation curve this paragraph wanted is
> **figure 2a**, now digitised — see the next section. Fig 4 is out of the A.1
> ask again, for a better reason than the first time: not "superseded by an
> internal tie" but "it is not the figure". The restoration was still correct
> at the moment it was made, on the evidence then available.

### A.2 — checked against figure 2a, and the fix is now scoped rather than guessed

Everything above is fwap arguing with itself: the bracket is anchored to the
wrong speed, therefore the roots are overtones. The argument rests on fwap's own
determinant, which is also the thing under suspicion. Schmitt & Cheng figure 2a
plots the same quantity for a rock the paper states, so it settles the question
from outside.

**The reference.** Figure 2a, p. 239 of the bound volume: *"Dipole source.
Dispersion (a) … of the flexural mode (1) and the first trapped mode (2) in the
presence of a fast sandstone. The velocities are normalized with respect to the
bore fluid velocity."* Rock from the paper's table 1: `V_P` 4878, `V_S` 2601,
`rho` 2160; fluid 1500 m/s, 1000 kg/m³; hole radius 0.10 m. For that model
`V_R` = 2412.8 and Scholte = 1484.4 m/s.

**How it was read.** Page rendered at 600 dpi, plot frame located from the axis
rules, curve 1's phase branch followed column by column — 1484 samples over
2.20-24.87 kHz. The 26 x-ticks land on integer kHz to within 0.06 kHz and the
1.400 / 1.000 y-ticks read 1.3978 / 0.9981, so axis calibration costs about
±3 m/s. The plotted line is 9-12 px thick, which dominates: **±20 m/s, roughly
±1 %**. That is a scan read carefully, not a validation-grade overlay, and
nothing below leans on better than 1 %.

**What the curve says, before fwap enters.** It starts at 2596 m/s — the
formation shear speed, 2601, to −0.2 %. It ends at 1493 m/s at 24.9 kHz, still
descending, against a Scholte speed of 1484.4: **+0.6 %**. Between those it
crosses `V_R` at **4.45 kHz** and the fluid velocity at **17.9 kHz**.

Two things follow immediately, and the second is the point of the item:

* A.1's flexural-Scholte tie now has an **external** confirmation. Until now it
  rested on fwap converging to a number fwap also computed. A published curve
  for a *fast* formation lands on it, and A.1's tie is a *slow*-formation
  result — so this is a check across the regime boundary, not the same claim
  twice.
* The solver's search window `(V_R, V_S)` contains the true root **only up to
  4.45 kHz — 10 % of the plotted band**. Above that the root is outside the
  bracket by construction. No tolerance, no seeding, no continuation strategy
  can recover it, because there is nothing in the interval to find.

**fwap against the figure**, same rock, 2.2-25 kHz on a 0.2 kHz grid:

| | |
|---|---|
| coverage | **43 %** (49 of 115) |
| every finite value | inside `(V_R, V_S)` — all bracket interior |
| on the right branch | 2 points, at 4.2 and 4.4 kHz (+2.8 %, +1.5 %) |
| the other 47 | two sawtooth ramps, 10.0-13.8 and 17.4-22.6 kHz, each running 2597 → ~2420 m/s |
| error on those 47 | **+58.6 % to +72.2 %, median +65.0 %** |

The two good points are exactly the ones the bracket argument predicts: below
4.45 kHz the true curve has not yet left `(V_R, V_S)`, so the solver finds it.
That they are the fundamental and not another coincidence is confirmed by the
widened-bracket scan, which finds a *single* root at those frequencies. Their
+1.5 % / +2.8 % is close to but not inside the reading uncertainty here — the
curve falls about 350 m/s per kHz through the plunge, so the ±0.06 kHz tick
calibration contributes ±0.9 % on top of the ±0.8 % vertical, for about ±1.2 %
combined. Right branch, agreement good, exactness not demonstrable.

That is the whole of it: **2 of 115 samples, 1.7 % of the band.** Everywhere
else the call either returns nothing or returns an overtone, and nothing in the
returned object distinguishes those two points from the 47 wrong ones.

**The bracket edit, now measured rather than estimated.** Replacing the `1/V_R`
edge with `1/Scholte` and enumerating every `Im(det)` root in the widened
window, against the digitised curve:

| band | result |
|---|---|
| below 4.4 kHz | **no real-axis root exists.** A 4001-point scan at 3.0 and 4.0 kHz finds sign changes only at the two endpoints — `V_S`, where `s = 0`, and `V_f`, the `F = 0` branch point. The determinant is finite throughout and `\|Re\|/\|Im\| ~ 1e-16`, so this is the analytic structure, not a numerical failure. The true root (2534-2595 m/s) is simply off the real axis. |
| **4.4-16.4 kHz** | **recovered: median \|error\| 0.66 %, worst 1.74 %, 18 of 31 sampled frequencies under 1 %** |
| above 16.4 kHz | the branch has dropped below `V_f`; again no real-axis root, and the widened bracket instead returns a different branch running 2095 → 1708 m/s, 14-39 % high |

So the earlier claim that widening "roughly doubles coverage" was right about
the direction and understated what it buys: 12 kHz of the 22.7 kHz plotted band
— **53 %** — comes back correct to better than 1.8 %, median 0.66 %. And that
is coverage of the *right* mode, which is not what the earlier figure counted.
The earlier claim that the residue is "the below-cutoff
half" was wrong — the residue is two intervals, one at each end, straddling a
working middle. Figure 2b is consistent with that: read the same way, the
flexural mode's `1/Q × 100` runs 1.70 at 2.3 kHz → 5.34 at 5 kHz → 3.27 by
25 kHz, so it is **non-zero across the whole band** — best `Q` about 59. The
pole is off the real axis everywhere, and the real-axis root is an
approximation, excellent in the middle and absent at the ends.

Worth noticing that the approximation's quality does **not** track `1/Q`. It
fails at 2.3-4.4 kHz, where attenuation is at its *lowest* (1.70), and works to
0.66 % at 5-16 kHz, where attenuation is at its *highest* (5.34 falling to
3.6). So "the pole is close to the real axis" is not the criterion, and a fix
that assumes it is will be tuned on the wrong quantity.

What the three regions line up with instead is **the phase velocity itself**.
The real-axis root exists over 4.4-16.4 kHz, which is where the curve runs from
2432 down to 1509 m/s — that is, from just under `V_R` to just over `V_f`. It
fails above `V_R` and it fails at `V_f`. The lower edge matches the `V_R`
crossing (4.45 kHz) to better than the grid; the upper edge is softer, because
the curve is nearly flat there — 16 m/s over the last 8.5 kHz — so "where it
reaches `V_f`" cannot be located to better than a kilohertz or two.
`V_f` being a boundary is expected: it is the `F = 0` branch point. `V_R`
being the other one is not explained here, and is the first thing worth
checking when the fix is attempted, since it is what a mode-selection rule
would have to key on.

**Still not fixed here, and now for a stated reason rather than a hedge.** The
measurement scopes the work: a bracket edit earns 4.4-16.4 kHz, and the two ends
need complex `k_z`. But a bracket edit alone would ship a solver that answers
confidently over the full band while being 14-39 % wrong above 16.4 kHz — the
same class of defect as the overtones, traded for a different wrong branch. The
selection problem has to be solved with the bracket, not after it. Two more
tests in `tests/test_cylindrical_solver.py` pin the disagreement against the
digitised table, which is checked in with its own end-anchor test so a bad
digitisation cannot silently become the reference.

### A.2 — figure 7a, and the defect measured against formation stiffness

Figure 2a settles one rock. **Figure 7a** (p. 244) plots the flexural mode for
**granite (1), limestone (2) and the same fast sandstone (3)** on one axis,
0-15 kHz, so the same measurement can be made against stiffness instead of at a
point. Same digitisation method, with the axes least-squares fitted to the tick
marks rather than to the frame rules: **15 x-ticks residual to 0.018 kHz, 4
y-ticks residual to 0.0004 normalised (0.5 m/s)**. Axis calibration is
negligible here; line thickness still gives about ±20 m/s.

**Three anchors instead of one.** Every plateau lands on its own formation shear
speed:

| | plateau read | `V_S` | |
|---|---|---|---|
| granite | 3749.6 | 3750 | **−0.01 %** |
| limestone | 2768.7 | 2771 | **−0.08 %** |
| fast sandstone | 2597.7 | 2601 | **−0.13 %** |

**The bracket empties at the same frequency for every fast formation.** All
three curves cross `V_R` between **4.43 and 4.45 kHz**, although `V_R` spans
2413 to 3388 m/s and `V_S` spans 2601 to 3750. Figure 2a gave 4.45 kHz for the
sandstone from a different page with a different axis range, so that is four
consistent readings. This is *not* self-similarity in `v/V_S` — at 5 kHz the
three read 0.690, 0.818, 0.838 — so two things vary and cancel. Recorded as
measured, not explained; it is the first thing to check when the fix is
attempted, because a mode-selection rule would want to key on it.

The practical form of that: the frequency at which `flexural_dispersion` stops
being able to contain the mode is **not rock-dependent**, so a caller cannot
reason their way around it from the formation properties they have.

**The error grows with stiffness.** By 11 kHz all three published curves have
converged to one line near 1570 m/s. Against it:

| | fwap at 11-13 kHz | error |
|---|---|---|
| fast sandstone | 2551-2442 | +57 % to +64 % |
| limestone | 2738-2628 | **+69 % to +73 %** |
| granite | 3740-3483 | **+124 % to +137 %** |

The `(V_R, V_S)` window rides further above the true curve the faster the rock,
so the defect is worst exactly where dipole logging most needs the answer.

**And over the band figure 7a actually resolves, granite returns nothing at
all** — NaN at all 13 tabulated frequencies from 3 to 10 kHz. Limestone answers
at 2 of 9 (4.25 and 4.5 kHz, +3.3 % and +1.5 %), both in the same sliver at the
crossing that figure 2a already identified. The wrong answers live higher up,
where the figure has merged the curves; the resolved band is simply empty.

**A check on the digitisation that owes nothing to fwap.** The fast sandstone is
plotted twice in this report — figure 2a (0.600-1.800 over 0-25 kHz) and figure
7a (0.500-2.600 over 0-15 kHz). Two pages, two axis ranges, two independent
calibrations of one physical curve. Over 2.50-5.50 kHz, thirteen consecutive
readings, they agree to within **−0.25 % to +0.38 %**. Above 5.75 kHz figure 7a
reads 0.8-1.9 % high, and that is explained rather than mysterious: the
limestone and sandstone curves become a single plotted line at exactly
5.75 kHz, so a column trace returns the band centre. Granite joins them at
10.25 kHz. Nothing above those points is tabulated.

Five more tests pin this, and the reference tables carry their own shear-speed
anchor test.

### Figure 8a — the slow formation, and the first external tie better than 1 %

Everything above measures a defect. Figure 8a (p. 245) is the other kind of
check: *"Slow sandstone. Dispersion and attenuation of the Stoneley wave (0),
the flexural (1) and screw (2) modes excited by a monopole, dipole, and
quadrupole source respectively."* One panel, three published curves, three fwap
solvers, on the path this project has always said works — and had never checked
against anything but itself. Rock: table 1's slow sandstone, `V_P` 2751,
`V_S` 1201, `rho` 2100.

**A trap in the axis, recorded because a careless read costs 0.5 %.** The y
labels print as 0.850 / 0.783 / 0.71? / 0.650, and the scan degrades the third
to something like "0.713". It is **0.71667**: the four tick rows are evenly
spaced (393.5, 395.0, 396.5 px), and fitting evenly divided values gives a
residual of ±0.00013 against ±0.0026 for the literal reading — twenty times
worse and structured. The same package prints 0.667 / 0.783 for an evenly
divided 0.550-0.900 axis in the panel next to it.

**Three anchors, none of which needs a solver.** The Stoneley's low-frequency
limit is the tube-wave speed, a one-line formula; both shear modes leave the
axis at the formation shear speed:

| | read | closed form | |
|---|---|---|---|
| Stoneley at f → 0 | 1135.6 | `tube_wave_speed` 1136.2 | **−0.06 %** |
| flexural onset | 1201.4 | `V_S` 1201 | **+0.02 %** |
| screw onset | 1201.4 | `V_S` 1201 | **+0.02 %** |

The three curves also resolve as three *disjoint* connected components, so no
branch tracking was needed — the median ink row per column is the curve. And the
narrow 0.650-0.850 axis makes this the most precise of the four figures: the
plotted line is worth about **±3 m/s, or ±0.3 %**.

**The result, over 0.1-14.9 kHz at 0.25 kHz:**

| mode | coverage | rms error | worst |
|---|---|---|---|
| **Stoneley** | **59/59** | **0.04 %** | **0.08 %** |
| flexural | 49/55 | 1.29 % | −1.84 % at 5.2 kHz |
| screw | 38/44 | 0.94 % | −0.56 % (+3.1 % near cutoff) |

**The Stoneley agreement is below the resolution of the figure.** fwap and the
published curve cannot be told apart. That is the project's first external tie
for `stoneley_dispersion` and it sits **60× inside the 5 % overlay budget A.1
set** — which changes A.1's own accounting, since the Stoneley was listed as
tied only analytically, to fwap's own asymptote.

**The borehole radius, which every comparison here leans on, is pinned by the
same curve.** Table 1 gives velocities and densities but no hole radius, so
`a` = 0.10 m was an assumption. The Stoneley misfit is 0.05 % rms at 0.100 m and
degrades either side — 0.13 % at 0.095, 0.14 % at 0.105 — so the assumption is
now a measurement, good to a few millimetres.

**And a small systematic that is not the radius.** The flexural offset is real:
zero near 3.3 kHz, −1.8 % at 5-6 kHz, recovering to −0.8 % by 14 kHz. It is
four times the reading uncertainty, and the Stoneley curve *on the same panel,
read with the same calibration* bounds that uncertainty at 0.08 %. No radius
removes it either — the best flexural fit is at 0.090 m, where the Stoneley is
already five times worse, and even there the flexural misfit is ~1 %. One
candidate: the paper's model is **viscoelastic** where fwap's open-hole solvers
are elastic — table 1 carries `Q_alpha` and `Q_beta`, and figure 8's own
attenuation panel gives all three modes `1/Q` ≈ 0.02. But that should move the
Stoneley too, and it does not. Recorded as measured and unexplained.

**The near-cutoff gap, at one width for both shear modes.** The published
flexural curve starts at 1.04 kHz and fwap's first root is at 2.52; the screw
curve starts at 3.74 and fwap's first root is at 5.26. **1.48 and 1.52 kHz** —
the same gap for two modes whose cutoffs are 2.7 kHz apart, so another quantity
set by the hole rather than by the mode, alongside the 4.4 kHz bracket-emptying
frequency in fast formations. Above the gap both solvers are contiguous. This is
the benign form of the failure that swallows the whole band in fast formations.

Ten more tests, including one that pins the radius and one that keeps the
Stoneley's tie an order of magnitude tighter than the shear modes'.

#### And figure 7b checks the `n=2` claim this file has been asserting

The table row and the item body have both said *"affects `n=2` identically, so
one fix repairs two solvers"* since the re-diagnosis, on the strength of a
non-monotone scatter between `V_R` and `V_S`. Figure 7b plots the **screw
(quadrupole) mode** for the same three formations over 4-20 kHz, so it can be
checked rather than asserted.

**It holds** — same mechanism, same stiffness ordering — **with one difference
that makes `n=2` the more dangerous solver of the two.** Measured over each
rock's plotted band at 0.2 kHz:

| rock | coverage | within 5 % | the rest |
|---|---|---|---|
| granite | **75 %** | 1 point | +5 % to +136 %, median **+102 %** |
| limestone | **66 %** | none | +11 % to +66 %, median **+57 %** |
| fast sandstone | **65 %** | none | +13 % to +56 %, median **+46 %** |

Every finite value again lies strictly inside `(V_R, V_S)`; in fact the returned
values sweep that window end to end — 3389-3750, 2565-2771, 2413-2601. And the
bracket empties at a mode-specific but again rock-independent frequency:
**7.53 / 7.61 / 7.69 kHz**, against 4.45 / 4.43 / 4.43 for the flexural mode,
with `V_R` spanning 2413 to 3388 in both.

The difference is coverage: **65-75 % here against 21-36 % at `n=1`**. A caller
filtering on `NaN` keeps two to three times as many wrong answers from
`quadrupole_dispersion` as from `flexural_dispersion`. So of the two solvers the
quadrupole is the one that fails most quietly, and "one fix repairs two solvers"
understates which of them needs it more.

### A.2 Fast-formation flexural leakage — the original entry

Was filed as "cased flexural bracketing", which measurement showed to be the
wrong diagnosis. The layered `n=1` solver no longer refuses fast formations, but
its root-finding stays sparse: on a typical casing + cement stack a fast
formation converges over roughly 38 % of a 1-12 kHz band, and only above about
5 kHz. That sparseness is why `scripts/gen_surrogate_dataset.py` keeps the cased
dataset single-mode.

**It affects `n=2` as well.** Checking the quadrupole's high-frequency asymptote
turned up the same signature: in slow formations `quadrupole_dispersion`
converges cleanly to the plane-interface Scholte speed (better than 0.1 % at
400 kHz), but in fast formations it returns a *non-monotone* scatter between the
Rayleigh and shear speeds — finite values, so a caller filtering on `NaN` keeps
them. Over the default mixed prior, 19 of 31 fast draws cleared `min_finite` and
18 of those 19 were non-monotone, which corrects a comment in the generator
claiming such draws "often fall below `min_finite`". So this item is not only
about the flexural mode; a fix would repair two solvers.

> *Checked against figure 7b, and the "finite values a `NaN` filter keeps" half
> is the understatement here.* Against the published screw-mode curves the
> quadrupole solver covers **65-75 %** of the band on the three fast rocks,
> against 21-36 % for the flexural solver on the same rocks — and is +46 % to
> +102 % wrong at the median, with **1 correct point across all three**. Of the
> two solvers the quadrupole is the one that fails most quietly. See the
> figure-7b subsection above.

**It is not caused by the layer stack.** Removing the casing and cement entirely
leaves the identical formation just as sparse in an *open* hole, over the same
lower part of the band — so no amount of work on layered bracketing will fix it.
`tests/test_cylindrical_solver.py` pins this comparison so the attribution
cannot quietly drift back.

The real cause is that in a fast formation the flexural mode is **leaky**: its
root leaves the real `k_z` axis, and the real-axis `Im(det)` sign change the
solver searches for survives only in a sliver beside the shear branch point at
high frequency. Widening the real bracket cannot recover it — scanning finds no
sign change below the cutoff in any of the three sub-windows (below the slowest
layer shear, between that and the formation Rayleigh speed, or between that and
the formation shear), and the middle window is in any case singular for the
propagator-matrix formulation. A fix means complex-plane root tracking.

> **Superseded in part — read the section above first.** The paragraph as
> written is true only *below the cutoff*, and it was applied to the whole
> item. Above the cutoff there is no leakiness involved: the search window is
> anchored to `V_R`, which is not this mode's limit, and widening it to the
> Scholte speed recovers a monotone branch running to within 3.6 % of Scholte.
> "Widening the real bracket cannot recover it" is a statement about the
> below-cutoff band, and it does not generalise. Nor does "a fix means
> complex-plane root tracking": that is the fix for one of the two defects,
> and not the one that returns wrong answers.

*Correction.* An earlier version of this paragraph continued "which is the same
machinery item G.2 needs, so the two should be planned together rather than as
separate efforts." That is wrong, and it kept the debonded-regime work filed
behind this one for several revisions. The debonded regime's standard model — a
fluid microannulus — is a **bound**-mode problem and needs no complex-plane
tracking at all; it is A.5 below, and two of its three pieces have shipped while
this item is still waiting on a derivation.

**Attempted, and it is not a wiring job.** The complex-plane machinery already
exists and is proven for `n=0` (`_track_complex_root`,
`_march_complex_dispersion`, `pseudo_rayleigh_dispersion`), so the obvious move
is to point it at the `n=1` determinant. Three approaches were tried and all
fail, which is worth recording so the next attempt starts further along:

1. *Continuation from high frequency.* Reproduces the real-axis branch to
   floating-point noise (`Im(k_z) ~ 1e-16`) and then stops exactly at the
   cutoff. The step never leaves the real axis, so it cannot follow a root that
   does.
2. *Fresh leaky-S seeding below the cutoff* (the trick the `n=0` code uses: seed
   above `V_S` with a positive imaginary part). Converges only sporadically and
   to incoherent values — phase velocity jumping 2681, 2918, 2789 m/s at 6, 4,
   3 kHz with attenuations of order 0.6 Np/m. These are numerical artefacts of
   the Hankel formulation, not a branch.
3. *Strict fine-step continuation from the cutoff* with an imaginary nudge. The
   nudged seed converges back onto the real axis, and the first step below the
   cutoff fails outright.

A fourth observation constrains any future attempt: even *above* the cutoff,
continuation across 1 kHz steps can hop to a different root (one below the
formation Rayleigh speed), so the leaky extension needs the validated marcher's
regime checks rather than the bare tracker.

What is missing is not code but a derivation: which Riemann sheet the `n=1` pole
occupies below the cutoff, and a determinant formulated consistently on it. Note
also the possibility that there is simply no leaky continuation to find — that
the fast-formation flexural mode exists only above its cutoff and the
low-frequency dipole energy travels as a shear head wave instead.
Distinguishing those two cases is exactly what Schmitt 1988 fig 4 would settle,
which puts this item behind the same literature access A.1 needs.

> **The whole paper has since been read. It is a different paper from the one
> cited, and it answers more than the figure did.** The document is
>
> > Schmitt, D. P., & **Cheng, C. H.** *Shear Wave Logging In (Multilayered)
> > Elastic Formations: An Overview.* MIT Earth Resources Laboratory, pp.
> > 213-246.
>
> — **two** authors, a different title, and an ERL report rather than the JASA
> article this file cites. Its figure numbering is its own, which is why "fig 4"
> resolved to a shot gather.
>
> **Its abstract settles A.2's open question in the authors' own words**:
> *"Whatever the formation (fast or slow) and the configuration, the low
> frequency part of both the flexural and screw modes follows the virgin
> formation shear wave characteristics."* The low-frequency flexural branch
> exists in fast formations and tracks `V_S`. There is no head-wave escape
> hatch, and no missing physics — only a solver that cannot find it.
>
> **Measured on the paper's own table-1 formations**, `flexural_dispersion`
> over 1-20 kHz:
>
> | formation (table 1) | `V_P` | `V_S` | `rho` | fwap coverage |
> |---|---|---|---|---|
> | fast sandstone | 4878 | 2601 | 2160 | **20 %** |
> | limestone | 5081 | 2771 | 2160 | **20 %** |
> | granite | 5881 | 3750 | 2160 | **10 %** — one point in ten |
> | slow sandstone | 2751 | 1201 | 2100 | 80 %, clean descent to Scholte |
>
> The paper plots continuous curves for all four. That is the size of A.2,
> stated against the reference it was always meant to be checked against.
>
> **Figure 2 is the one this item needed**: *"Dipole source. Dispersion (a),
> attenuation (b), and excitation (c) of the flexural mode (1) and the first
> trapped mode (2) in the presence of a fast sandstone."* Figure 7 adds
> flexural and screw dispersion for granite, limestone and fast sandstone
> together. Both are fast-formation flexural dispersion over the band fwap
> returns almost nothing for.
>
> *Both have since been digitised — see the figure-2a and figure-7a sections
> above.* The coverage column here is the weaker statement, and it should not
> be quoted on its own: coverage counts answers, and figure 7a shows the
> answers are 57-137 % wrong. Granite in particular returns **nothing at all**
> over the 3-10 kHz band figure 7a resolves; its "10 %" is a sawtooth at
> 11-13 kHz, outside the resolved region entirely.
>
> **Settled — the figure has been seen, and it refutes the second case.**
> Schmitt 1988 fig 4 is a dipole **shot gather** in a *fast sandstone*: 14
> traces at `r` = 2.40-5.00 m in 0.20 m steps, panel (a) at a 1 kHz source
> centre frequency and panel (b) at 6 kHz. Both panels show a strong, coherent,
> slowly-decaying dipole arrival. `flexural_dispersion` returns **`NaN` at both
> 1 and 6 kHz** for a fast formation of this kind.
>
> So the energy is there and the solver cannot find it. "No leaky continuation
> to find" is out; the gap is a solver limitation, not an absence of physics.
>
> The figure also shows the branch is *dispersive in the expected direction*.
> Reading peak moveout by eye across the 2.60 m aperture — ±15-20 %, it is a
> scan — panel (a) gives roughly **2400 m/s**, near the formation shear
> velocity, and panel (b) roughly **1450 m/s**, near the Scholte speed. That is
> the same descent the A.1 flexural tie pins to 1e-3 in *slow* formations,
> now visible in a fast one across the band the solver returns nothing for.
>
> What it does **not** give: a usable number. Eyeball moveout at ±20 % cannot
> validate anything, and a shot gather is not a dispersion curve. Its value is
> that it answers a yes/no question that had been blocking the item.
>
> **Figure 2a has since been digitised and does give the numbers** — the same
> descent, `V_S` → Scholte, resolved to ±1 % across 2.2-24.9 kHz. See the
> figure-2a section above. Fig 4 keeps only its yes/no role; it is no longer
> the best instrument for anything in this item, and A.1 should not carry it.

Scale of the consequence: fast formations average **28 %** band coverage (5/47
fully converged over 50 draws), while slow formations converge fully.

*Correction.* An earlier version of this entry added "only about 15 % of draws
are slow", measured over the **default** `FormationPriors` (1200-3200 m/s). That
is not the prior the cased generator uses: `generate_cased_dataset` pins
1700-3000 m/s, so **100 %** of its draws are fast and none are slow. The 15 %
figure described the wrong distribution and is withdrawn.

The correction changes the conclusion rather than just the number. A two-mode
cased dataset is not a *subset* of the existing one; it needs a different,
disjoint prior, because the two cased modes fail in opposite directions —
flexural is sparse in fast formations, and the Stoneley stops being bound as the
formation slows away from the fluid. Measuring both together across the annulus
prior gives a both-modes-bound fraction of 0.00 at `V_S` = 1350 m/s, 0.42 at
1380, 0.92 at 1400, and 1.00 from 1420 up to the 1500 m/s fluid. That ~80 m/s
window is shipped as `SLOW_TWO_MODE_PRIORS` /
`generate_slow_two_mode_cased_dataset`, with the restriction stated at the point
of use: it suits cement-bond work, where the label is the bond index and
formation `V_S` is a nuisance parameter, and is the wrong dataset for anything
needing formation-property variety.

### A.5 Fluid microannulus — the debonded-regime forward model

The forward model is complete and on `main`; what remains of this item is its
`sonic_ml` consumer, tracked as G.2. It arrived here from section G, where it had been filed as
needing a leaky-mode cased forward model — see the correction under A.2. A
microannulus is a **bound**-mode problem.

Debonding has two candidate models and they are not interchangeable. *Soft
cement* is genuinely out of reach: `_stoneley_kz_bracket_cased` takes its
bound-regime floor from the softest shear velocity anywhere in the stack, so
once a layer's `V_S` falls below the fluid velocity there is no bound window
left containing the Stoneley mode — measured, the cased Stoneley converges fully
down to `cement_vs = V_f`, partially just below, and not at all by 1200 m/s. A
*fluid microannulus* — the standard model in cement-bond logging — is not
excluded by that argument, because its floor is set by its acoustic velocity
(~1500 m/s) rather than by a near-zero shear velocity. It also cannot be
approximated by a very compliant elastic layer, precisely because an elastic
layer does drag the floor down: measured, that fails at every thickness tried,
down to 0.2 mm.

**Shipped.** `_fluid_layer_e_matrix_n0` / `_fluid_layer_propagator_n0` (a fluid
annulus carries two amplitudes rather than four, imposes no shear traction, and
permits axial slip, so its state is the pair `(u_r, sigma_rr)`);
`_modal_determinant_n0_microannulus`, an 11x11 assembly for
`fluid | casing | microannulus | cement | formation`; and the public
`stoneley_dispersion_microannulus` / `FluidAnnulus`.

The assembly has **no reduction to the existing solver** to check against: the
`annulus_thickness -> 0` limit is a frictionless slip interface, not the bonded
stack, so at 8 kHz the Stoneley-like root converges to 1383.45 m/s against
1400.04 m/s bonded and the 1.2 % offset does not close. It is validated instead
against the **Krauklis crack wave**, `c = (omega h / (C rho_f))^{1/3}` with `C`
the sum of the wall compliances `(1 - nu)/mu` — an analytic result with no
Bessel functions and no cylindrical geometry in it, reproduced to 0.02 % at a
1 um gap.

Both public entry points and the `FluidAnnulus` type are now on `main`.
`stoneley_dispersion_microannulus` selects structurally — the Stoneley-like
mode is the fastest bound n=0 mode, so the first sign change above the bound
floor is it — and `crack_wave_dispersion` returns the second family, the mode
guided by the gap itself. Both are pinned as independent of the caller's
frequency grid and of the scan resolution.

The crack wave needed a **spurious-root filter**, and the obvious candidate was
measured and rejected on the way. Over 270 sampled configurations the bound
window held exactly two roots in 269; the exception produced a duplicated pair
near 4 m/s. The natural gate — the elastic propagator's determinant identity
`det P = (r_inner/r_outer)^2`, found while building the Stoneley API — does
**not** work: at a 1 um gap the genuine crack root is fixed to 1.5e-9 across a
tenfold range of cement thickness over which that identity degrades to 1e232,
because the mode is confined within `~1/k_z` of the gap and the error lives in
the growing branch the root never sees. What shipped instead is grid-stability
filtering, the technique that exposed the `n=0` defect: two scans at different
resolutions and lower endpoints, keeping only the intersection. The spurious
pair appears in one grid of six; the genuine roots in all six.

**What is left:**

- ~~The `sonic_ml` consumer: a debonded-regime dataset, and with it the first
  fair CBL-amplitude comparison.~~ **Shipped** as G.2 — generator, closed-form
  baseline and learned inverse — and scored through the harness as G.6. This
  bullet outlived the work by several revisions. Note the CBL half of it was
  separately withdrawn: these gathers carry no casing-ring arrival, so a
  CBL-amplitude comparison would still be a strawman. See G.2.
- Optional, and the only genuinely open item in this section: a delta-matrix / Abo-Zena
  reformulation of the elastic stack would remove the cancellation that makes
  the filter necessary at all, and would raise the frequency ceiling — the
  crack-wave window collapses above ~240 kHz on a typical stack purely because
  the propagators stop being representable.

`n=1` / `n=2` microannulus assemblies would be needed for *flexural* CBL work
and are a separate, larger job. The `n=0` path is self-contained and does not
depend on them.

### References for section A

- Schmitt, D. P. (1988). Shear wave logging in elastic formations. *J. Acoust.
  Soc. Am.* 84(6), 2215-2229. https://doi.org/10.1121/1.397015
- Schmitt, D. P., & Cheng, C. H. *Shear Wave Logging In (Multilayered) Elastic
  Formations: An Overview.* MIT Earth Resources Laboratory, pp. 213-246.
  **A second, closely related document, and the one every "fig N" in this
  file actually refers to.** Same method and a near-identical abstract, but two
  authors, a different title and its own figure numbering. Its table 1 and
  figure list are transcribed under A.1. Distinguish the two before citing a
  figure number: they do not carry across.
  <br>*Corrected:* every citation of this paper outside `fwap/validation.py`
  gave the pages as **2230-2244** and hyphenated the title as "Shear-wave" —
  thirteen page ranges and twelve titles across nine files, all propagated from
  one another rather than checked. `validation.py` had it right the whole time,
  so the correct value was **already in the repository**. That matters because
  A.1's remaining ask is "find figure 4 in this paper", and a wrong page range
  is exactly what wastes the trip. The DOI is added so the next check is a click
  rather than a search.
- Paillet, F. L., & Cheng, C. H. (1991). *Acoustic Waves in Boreholes*,
  Chapter 4. CRC Press.
- Tang, X.-M., & Cheng, A. (2004). *Quantitative Borehole Acoustic Methods*,
  Chapter 3. Elsevier.
- Kurkjian, A. L., & Chang, S.-K. (1986). Acoustic multipole sources in
  fluid-filled boreholes. *Geophysics* 51(1), 148-163 (most explicit derivation
  of the 3x3 dipole determinant).
- Ellefsen, K. J., Cheng, C. H., & Toksoz, M. N. (1991). Applications of
  perturbation theory to acoustic logging. *J. Geophys. Res.* 96(B1), 537-549
  (starting-guess strategy for the dipole root-finder).
- White, J. E. (1983). *Underground Sound: Application of Seismic Waves*.
  Elsevier (the tube-wave low-frequency form).

## D. Conda-forge recipe

The package is ready for PyPI (0.4.0 builds cleanly, wheels ship `py.typed`). A
conda-forge recipe (`meta.yaml` + CI setup) can be submitted to
[staged-recipes](https://github.com/conda-forge/staged-recipes) once the first
PyPI release is live. Reversible, low-risk; one afternoon's work.

## F. Real-data test fixtures

**Status (partially closed)**: the *harness* now exists, and adding a dataset is
a one-entry change. `scripts/fetch_real_data.py` holds a registry of third-party
files with URL, SHA-256, provenance and licence; `tests/test_real_data.py` runs
against them and skips with an actionable message when they are absent, so CI
stays hermetic. Two files are registered: a real Kansas Geological Survey well
log (a wrapped LAS with 26 service-company curves, which our own writer would
never emit) and a SEG-Y written by `segyio` (so a reader/writer disagreement
cannot hide behind a round-trip through our own writer).

Nothing is vendored, deliberately: the files are published by others under their
own terms — the KGS log carries a third-party copyright notice in its own header
— and `tests/data/real/` is git-ignored with a test asserting it, so the
no-redistribution property is enforced rather than intended.

**Substantially advanced.** A real Schlumberger DSI sonic log from Utah FORGE
well ME-ESW1 is now registered (`forge_dsi_las`) and covered by tests, and the
companion DLIS carrying the per-receiver waveforms has been opened and measured.
See `plans/log_output.md` for the full reading. In brief:

* The waveforms exist and are the geometry this package models -- `PWF1`-`PWF4`,
  each `(10839, 8, 512)`, eight receivers and 512 samples, for lower dipole,
  upper dipole, monopole Stoneley and monopole P&S. Acquisition parameters were
  read from the file, not assumed: 10 us sampling on the monopole P&S, 6 in
  receiver spacing, 9 ft to the first receiver, zero firing delay.
* The LAS is the processed export of exactly those waveforms: `DTCO` and `DTSM`
  agree between the two to 5e-5 us/ft over all ~10 800 common depths. So the
  data is *scoreable* -- the package's picks can be compared against a vendor's
  on identical rock.
* **Measured, and this is what the item existed to find.** Over 400 contiguous
  frames, `fwap.stc` + `track_modes`: shear matches `DTSM` to a median
  **+0.12 %** (MAD 2.6 %, 96 % within 10 %). Compressional did not -- median
  +2.29 % but mean 27 % high and only 62 % of depths within 10 %, a bimodal
  failure rather than noise, with about a third of depths picking a later
  arrival as P. That became item F.1, and it is now fixed.

**F.1, closed: the compressional-pick defect.**

* **It was mode confusion, not imprecision.** On 143 of the 150 bad depths
  `track_modes` assigned the *same* STC peak to P and to S. Mode ordering was
  enforced on arrival time, never on slowness, and the P prior window
  (40-140 us/ft) contains the shear arrival; when shear is the more coherent of
  the two, the `scored` rule's `time_penalty` cannot overcome the 0.139
  coherence deficit.
* **The repair refuses to give one arrival two labels.** `pick_modes` and
  `track_modes` now take `resolve_mode_collisions=True`: when two modes have
  selected the same STC peak, the faster-labelled one re-picks from its own
  candidate pool with that slowness as a strict upper bound.
* **It deliberately does not decide which label is wrong.** That is not
  decidable in general, and both directions occur. On the DSI log the shared
  peak is the shear arrival and P is the mislabel; on a slow-formation
  synthetic (Vp/Vs = 2, so S lands at 174 us/ft inside P's window) it is the
  compressional arrival and S is. A rule that always trusted the slower mode
  would be right on the log and wrong on the synthetic -- an earlier version
  did exactly that, and `tests/test_hypothesis.py` caught it dropping a
  correct P. So a mode with no admissible faster candidate is left exactly as
  it was, on the reasoning that "nowhere faster to go" is evidence it holds
  the right arrival. Nothing is dropped, nothing moves to a slower candidate,
  and no depth can come out worse than the greedy result.
* **Measured on the same 400 depths.** Vendor agreement 62 % -> **95 %**, with
  coverage unchanged at 400/400; depths where P is not strictly faster than S,
  143 -> **5**. The rule changed the P pick at 138 depths, every one a
  collision, made 129 of them correct, left the shear pick **bit-identical at
  all 400** (96 % throughout), and damaged **none** of the 250 depths that
  were already right. Of the 150 wrong depths 21 still are: 14 collisions it
  could not resolve or re-picked onto an intermediate peak, and 7 that were
  never collisions.
* **Confirmed on a second logging pass, which is what stops this being tuned
  to one dataset.** The same well's 25-September run, a different depth
  interval (7267-7466 ft): agreement 70 % -> **86 %**, unordered depths
  72 -> **2**, 63 of 117 bad depths repaired, none dropped, and again **no
  damage to any** of the 283 depths that were already right. Shear was
  unchanged there too (66 % on that interval, before and after). There is no
  constant to tune in the rule, which is the point: it transfers.
* **Retuning `time_penalty` was the wrong lever**, and this is why the fix is
  structural: the value that would flip those depths has median 0.18 and 90th
  percentile 0.43 against a default of 0.1, and raising it that far biases
  every late mode.
* **`viterbi_pick_joint` is still the better tool on the hard residue.** It
  reaches 89 % on identical surfaces in the same runtime, by a different
  mechanism — a global cost over the mode tuple rather than a local rule — so
  it also repairs confusions that are not exact collisions, of which this log
  has 7. The collision rule by construction leaves those alone, as it does the
  3 depths where P and S end up one slowness cell apart.
* **The ceiling was known in advance and was hit.** 13 of the 150 bad depths
  have a true-P peak below `coherence_min` and 8 have none at all, so no
  selection rule reaches beyond about 95 % of all depths here. The repair
  reaches 95 %.
* Seeded synthetics reproduce the pre-repair failure, the repair, and the
  case the rule declines to guess, all in CI without the 808 MB fixture; the
  old behaviour stays reachable, and tested, via
  `resolve_mode_collisions=False`.

**F.3, closed: the waveforms are reachable from the public API.**

* `read_dlis` returns one value per depth and skips everything else, which is
  where a full-waveform record lives. `read_dlis_waveforms` reads one such
  channel as `(n_depth, n_receiver, n_sample)`, and `DlisCurves` now reports
  the names and shapes of what it skipped so they are discoverable at all.
* **The acquisition geometry comes from the file.** RP66 v1 AXIS objects carry
  COORDINATES and SPACING *with a declared unit*, so `sample_interval()` and
  `offsets()` return seconds and metres without a constant anywhere: 10 us and
  eight receivers 6 in apart from 7.874 m on this tool. Which axis is which is
  decided by the declared unit, never by the AXIS-ID string, since AXIS-ID
  values are producer-defined.
* It also corrected an assumption. The hand-assembled runs used a 2.7432 m
  first offset read off the tool description; the file says 7.874 m. Slowness
  depends on receiver *spacing*, so the earlier numbers stand unchanged — 86 %
  compressional agreement either way — but arrival times do not, and the file's
  value is the right one.
* Reading one channel of the 88 MB pass takes 1.1 s, against ~100 s to
  materialise the whole frame, because only the requested channel and the index
  channel are read.

**F.3's second file, and what it corrected.**

ODP Leg 157 Hole 952A arrived after F.3 shipped, and immediately found the
limit of the design. `read_dlis_waveforms` had been built to read the
acquisition geometry from RP66 v1 AXIS objects rather than from a service
company's PARAMETER naming, on the grounds that AXIS carries a declared unit
and vendor names carry a convention. **That is right and it is not
sufficient**: this file declares *zero* AXIS objects, and its `DSI0` parameter
is the only record of the 10 us sample interval anywhere in the DLIS.

One file made the standards-purist choice look complete. Two showed it left a
real file unreadable. The reader now falls back to the vendor parameter, and
the fallback is deliberately timid because a parameter has no declared unit and
guessing is a factor-of-1000 error:

* it fires only when no axis carries a time unit **and** the file's `DSI*`
  parameters agree on one value. Where they disagree — the FORGE file carries
  40, 40, 40, 10, 40 — deciding which belongs to a channel is a vendor question
  and it raises, naming them all;
* the microsecond convention is checked rather than trusted: the implied record
  length must be sonic-plausible;
* `sample_interval_source()` reports which route answered, because one is a
  standard unit-bearing record and the other rests on a convention.

The ODP fallback value is independently confirmed: the archive's binary header,
which reached the numbers through an entirely different conversion path
(DLIS → GeoFrame → ASCII → binary), also says 10 us.

**What is still open:**

* ~~**F.5 — the ODP file's two unknowns.**~~ **Both answered.** See
  "F.5 — what the ODP archive turned out to know about itself" below. The
  claim that neither was "resolvable from the files" was wrong: one was
  resolvable from the archive alone, and the other was resolvable from the
  waveforms themselves once a specification supplied the numbers to test.
* ~~**F.2 — a waveform fixture the CI can actually use.**~~ **Closed.** See
  "F.2 — the fixture, and the claim that was holding it up" below. The short
  version: the item was blocked for its whole life on a sentence that was
  wrong.
* **F.4 — confirming the registered checksums.** **Two** of them now, not one;
  this entry said "the one unverified claim in the fixture registry" for a
  revision after the second appeared.
  `forge_dsi_las`: `gdr.openei.org` was unreachable from the session that added
  it, so the SHA-256 came from a mirror copy.
  `iodp_u1347a_dsi`: `zenodo.org` was blocked from the session that added it, so
  both the archive and member digests came from a copy that arrived by another
  route, and the canonical URL's shape is inferred from Zenodo's standard file
  layout rather than exercised.
  Both are flagged in their entries' `provenance`. Each needs one successful
  fetch from its canonical host to clear, and the entry should then be
  corrected rather than left carrying a stale caveat.

*This entry used to add "because no openly redistributable one is known to
exist". That is withdrawn — a search turned up two credible candidate sources,
and the claim was too strong.* Neither has been downloaded or opened, so what
follows is a shortlist assembled from published metadata rather than from
inspected files. Treat it as a lead, not a result.

1. **Utah FORGE**, via the DOE Geothermal Data Repository (`gdr.openei.org`).
   Wells 58-32 and 16A(78)-32 carry Schlumberger dipole sonic logs in **DLIS**,
   which `fwap.io.read_dlis` already reads. The tool described for the site
   (DSST-B) is an **eight-receiver array with a monopole and two dipole
   sources** — the geometry this package models. GDR data from DOE Geothermal
   Technologies Office projects is **CC BY 4.0**, so this one is
   redistributable, not merely fetchable. Formation is granite, which is fast —
   useful, and a reminder that it exercises the regime where the flexural solver
   is sparse (A.2).
2. **IODP / ODP**, via the LDEO Borehole Research Group
   (`brg.ldeo.columbia.edu`). Per-hole pages publish sonic waveform data for
   many expeditions, in DLIS *and* in a binary export intended for import into
   Python. The documented layout is close to this package's defaults: eight
   waveforms of 512 samples at 10 us (monopole) or 40 us (dipole), logged every
   15.24 cm. The licence could not be confirmed from here; IODP data are open
   access after moratorium, but whether that permits redistribution is the open
   question. Note this matters less than it looks — the harness fetches on
   demand and never vendors, which is exactly how the KGS log with its
   third-party copyright is already handled.

**Fetching was attempted from here, and the result narrows the handoff.** An
earlier version of this entry added that Utah FORGE is "also mirrored on AWS
Open Data", implying the logs could be pulled from S3. That is wrong and is
removed. What was measured:

* `gdr-data-lake.s3.amazonaws.com` and `oedi-data-lake.s3.amazonaws.com` **are**
  reachable from this sandbox, and object downloads work (a ranged GET returned
  real bytes). So S3-hosted open data is fetchable in principle.
* Those buckets do **not** carry wireline logs. The GDR lake holds bulk
  monitoring data only — FORGE has `DAS/`, `Geophone/` and a stimulation prefix
  (a complete listing, not a truncated one); the other prefixes are CASSM,
  magnetotellurics and DAS. No DLIS, no LAS, nothing from a wireline sonic tool.
* Every route that *does* host the log submissions is blocked: `gdr.openei.org`,
  `data.openei.org`, `catalog.data.gov`, `brg.ldeo.columbia.edu`, `www.osti.gov`
  and `iodp.tamu.edu` all fail to connect.

So the files are not reachable from here, and the reason is which host serves
them rather than anything about the data. A session with ordinary web egress
could fetch them directly. An earlier claim that this sandbox's egress "reaches
GitHub only" is also withdrawn by the measurements above.

**What a person with network access needs to do next**, in order: open one file
and confirm it contains per-receiver waveforms rather than only processed
slowness curves; confirm the licence permits at least fetch-on-demand use;
compute a SHA-256 and add one `RealDataset` entry to
`scripts/fetch_real_data.py`. Only the first of those is real work. No registry
entry is added here because the checksum cannot be computed without the file,
and a registry entry without a verified checksum would defeat the point of the
registry.

**Priority note**: this remains the highest-value open item, and its value grew
when the `sonic_ml` layer landed (section G). Every number that layer reports —
including the headline that a learned inverse beats classical processing by
roughly an order of magnitude in the open hole — is measured on data drawn from
the *same forward model* that generated the training set. That measures
identifiability, not field accuracy, and no amount of additional synthetic work
can close the gap. A single real gather with trustworthy reference picks would
say more about whether any of this transfers than another milestone of
modelling.

## F.2 — the fixture, and the claim that was holding it up

`iodp_u1347a_dsi` is registered: IODP Expedition 324, Hole U1347A, an
eight-receiver Schlumberger **DSI** monopole run over 3575–3774 mbrf. 1307
depths, 512 samples at 10 µs, 0.1524 m receiver spacing, 0.1524 m depth
increment — which is `ArrayGeometry`'s default geometry, arrived at
independently years earlier. Published by LDEO's Borehole Research Group on
Zenodo (record 3939555) under **CC0**.

**The item was blocked on a false sentence, not on a missing file.** Both
`scripts/fetch_real_data.py` and `tests/test_real_data.py` asserted that no
openly redistributable full-waveform sonic gather was known to exist. What had
been established was that none had been *found*. Those are different claims,
and the second one is a research result with a shelf life — it goes stale the
moment someone searches again. Written into a docstring as a flat statement, it
stopped anyone re-checking, and the item sat open for the project's whole life
behind it. Both docstrings now say what was wrong rather than quietly dropping
the sentence.

This is the same failure as F.5's, two items apart: *I do not know this*
recorded as *this is not knowable*. F.5's version was about one archive; this
one was about the entire published record of scientific ocean drilling.

**What had to be built.** The archive publishes every run twice — the original
service-company DLIS, and a plain binary export about a fifth the size with a
short self-describing header. `read_ldeo_waveforms` reads the export, and it
**verifies rather than trusts**: `4·(1 + n_receiver·n_sample)·(1 + n_depth)`
must equal the file size exactly, and the sample interval must be
sonic-plausible, both before a single sample is read. That is not defensive
programming for its own sake — the format is big-endian, and read
little-endian its header decodes to enormous garbage rather than to nothing, so
a trusting reader would allocate wildly or return silently wrong data. It also
declines to invent transmitter offsets, which the export does not carry.

**What it measures.** `stc` over 50 real gathers returns a median peak
coherence of **0.948**, with 96 % above 0.6. The test does *not* assert a
slowness: over this interval the lithology runs chert, chalk and basalt, so
slowness genuinely ranges over a factor of four and pinning a number would be
pinning the geology. It asserts instead that **no pick sits on a search-band
edge**, which is the check that the band measured the formation rather than its
own boundary. That earned itself at once — the first band tried, (5e-5, 6e-4)
s/m, returned 10 % of picks pinned at 6e-4.

**What this does and does not buy.** It bounds how wrong the *processing* can
be against data this repository did not generate, which nothing here could do
before. Three things it does **not** buy, stated plainly because the item's
title ("a waveform fixture CI can use") invites over-reading:

* **CI does not run it.** The real-data suite skips unless the file has been
  fetched, and the default run stays hermetic by design. Fetching is one command
  and every assertion is checksummed, but nothing runs it automatically, so what
  defends the F.1 picker fix on each push is still a seeded synthetic. Making
  CI fetch 578 MB is a separate decision, and a worse one than it sounds.
* **It is not the log that found F.1.** That was FORGE. U1347A is a second,
  independent hole — better in that respect, and not a regression test for the
  original defect.
* **It does not touch `sonic_ml`.** Those results are still measured against the
  forward model that generated their training data. One hole of one tool bounds
  the processing chain; it does not convert an identifiability study into a
  field measurement.

The entry is still fetched rather than vendored, but the reason has changed and
is recorded: CC0 removes the licensing objection, and 578 MB is the only one
left.

## F.5 — what the ODP archive turned out to know about itself

Both unknowns are answered, and the framing they were filed under was wrong.
F.5 said the receiver offsets "need the SDT tool specification" and that the
950-A header "cannot be resolved from the files alone". The first was half
right and the second was simply false. The general lesson is the one worth
keeping: *unknown to me* had been recorded as *not in the file*, and nobody
had opened the rest of the archive to check.

### The hole identity: resolved from the archive alone

The DLIS carries **seven** logical files under **two** origins, not one:

| origin | logical files | header | `well_name` | latitude / longitude | depth units |
|---|---|---|---|---|---|
| 3 | 6, **all with frames** | `PHASOR INDUCTION/LSS` | `ODP HOLE 950-A LEG 157` | 31°09.015′N 25°36.039′N *(sic)* | feet |
| 12 | 1, **no frames** | `NEUTRON/DENSITY POROSITY LOG` | `ODP HOLE 952-A LEG 157` | 30°47.413′N 24°30.588′W | metres |

So the waveforms live in the logical files that name the *wrong* hole. Three
independent checks say the well-name and coordinate fields are stale and
everything else in those origins is 952A's:

1. **Both** origins declare `file_set_name = 'ODP/952-A'`. The file set knows
   what it is even where the well header does not.
2. The archive's own Leg 157 hole table lists `157-952A_1.dlis` (Inv. #421) as
   holding exactly six runs — `DITE.003/.004/.005/.006/.007/.009`, with
   intervals 5503.01–5617.62, 5642.15–5516.73, 5599.33–5498.29,
   5612.28–5708.45, 5738.93–5604.08 and 5743.50–5702.35 m. The six framed
   logical files match those names and intervals **six for six, to the
   decimetre**. Hole 950A's own data is a different file (`157-950A_1.dlis`,
   Inv. #416) with different run names (`DITE.011/.012/.014`).
3. The toolstrings differ and ours matches 952A: the table gives
   950A as `DIT/LSS/NGT` and 952A as `DIT/LSS/HLDT/CNTG/NGT`, and this file's
   tool records include `HLDTA` and `CNTG`. Its own remark line reads
   `Toolstring DITE/HLDT/CNTG/LSS/NGTC.`, and its `BSDF`/`BSDT` (5442.70 m /
   5868.60 m) are 952A's documented sea floor and driller's TD.

A stale header is also the mundane explanation: same leg, same logging unit
(#718), same engineer, same witnesses, so a location block left over from an
earlier hole is the ordinary kind of wellsite mistake. Note the limit of that
last sentence — it is a *story*, not a finding. Whether 31°09.015′N
25°36.039′W is in fact Site 950's position was not confirmed; the LDEO and
ODP publication hosts are blocked from here (see below). What is established
is that those coordinates are **not** 952A's while the three checks above say
the data is, which is all the identification needs.

### The receiver offsets: specification plus a measurement that tests it

The tool is the **Long Spacing Sonic (LSS)** — stated in the remark line and
in the hole table, run on SDT-C hardware (sonde `SLS-ZA`, serial 542, 220 in
long, 3.375 in OD, from the DLIS `EQUIPMENT` records) in firing mode `LDDB`,
which is the depth-derived borehole-compensated long-spacing mode the archive's
info page describes in words. The LSS geometry is two sources 2 ft apart
sitting 8 ft below a pair of receivers also 2 ft apart, giving source–receiver
offsets of **8, 10, 10 and 12 ft**.

That number came from a specification, so it was tested against the waveforms
rather than assumed. First-break moveout across `WF1`–`WF4`, over all 532
depths of the upper section:

| | WF2 | WF1 | WF4 | WF3 |
|---|---|---|---|---|
| median first break (µs) | 1530 | 1910 | 1920 | 2300 |
| implied offset (ft) | 8 | 10 | 10 | 12 |

The signature the specification predicts is present and is not subtle: **two of
the four paths coincide** (WF1 and WF4 differ by one 10 µs sample) and the two
outer gaps are equal at 380 µs each. Regressing the four picks against
8/10/10/12 ft gives a slope of **190.0 µs/ft** and an intercept of
**−0.0 µs** (IQR −25 to +23). Two things follow:

* The slope reproduces the vendor's own `DTLN` over the same interval —
  median **190.0 µs/ft** — exactly. That is an independent confirmation from
  Schlumberger's processing of the same trip.
* The near-zero intercept pins the *absolute* offsets, not just their spacing.
  A base offset wrong by Δ ft would shift the intercept by −190·Δ µs, so
  ±25 µs bounds Δ at about **±0.04 m**. Moveout alone could only ever give the
  2 ft increment; it is the intercept that rules out a different base.

The lower section agrees: slope 187.5 µs/ft, intercept 2.5 µs.

**Mapping for a registry entry:** `WF2` → 2.4384 m, `WF1` → 3.0480 m,
`WF4` → 3.0480 m, `WF3` → 3.6576 m; 500 samples at 10 µs; depth increment
0.1524 m; monopole.

### Three smaller things the check turned up

* **The archive's row counts are wrong.** Its info page states 539 depths for
  the upper section and 592 for the lower. The binary headers say **532** and
  **591**, and the file sizes settle it: at `4·(1 + 4·500) = 8004` bytes per
  record, 8004·(1+532) and 8004·(1+591) reproduce 4 266 132 and 4 738 368 bytes
  exactly. Total is **1123** depths, not 1131.
* **Two hemisphere letters are wrong**, in two different files. The info page
  writes the longitude as 24°30.570′**E** for a site on the Madeira Abyssal
  Plain, which is west; and origin 3 in the DLIS writes its longitude as
  25°36.039′**N**.
* **The two nominally-10 ft paths are not interchangeable.** `WF1` − `WF4` runs
  −10 µs on the upper section and −20 µs on the lower — one to two samples,
  systematic. Small, unexplained, and worth carrying in a registry entry rather
  than smoothing over.

### What is left

Nothing that blocks a fixture entry; F.5's remaining content is provenance
wording rather than research. Two honest caveats to carry:

* The 8 ft base offset rests on the LSS specification *and* on the intercept
  measurement agreeing with it, not on any statement in the file. That is an
  inference, and a registry entry should say so.
* `mlp.ldeo.columbia.edu`, `brg.ldeo.columbia.edu` and `www-odp.tamu.edu` are
  all blocked by this sandbox's egress proxy, so the LSS geometry was
  corroborated through search-result text and the archive's own copies of the
  LDEO pages rather than by fetching LDEO's tool page directly. The numbers
  agree across two independent search sources and the waveforms, which is why
  this is recorded as settled rather than pending — but the primary page has
  not been read from here.

## G.2 The debonded regime — measured, and what it changed

The generator shipped (`MicroannulusPriors`, `DEBONDED_MODES`,
`generate_debonded_dataset`, `--debonded`). The measurements that shaped it are
the durable part, because the obvious build would not have been invertible.

**The item was framed wrongly, and measurement caught it.** The plan was "the
cased dataset, in the debonded regime": same Stoneley mode, gap width as the
label. Over 1-12 kHz on a representative stack, holding everything else fixed:

| quantity varied | Stoneley curve | crack wave |
|---|---|---|
| gap 10 → 1000 µm (100×) | **0.05 %** | **+301 %** |
| formation `vs` across its prior | 1.0-1.5 % | 0.03 % |
| cement `vs` across its prior | 0.48 % | 1.0-3.3 % |
| bonded → debonded (any gap) | **4.14 %** | n/a |

* **The cased Stoneley mode is blind to gap width.** It responds to the slip
  interface — shear traction is zero on both faces of a fluid layer however
  thin — and that response is the same at 10 µm as at 1 mm. It supports a
  bonded/debonded *state* at roughly 3:1 over the nuisance parameters, and not
  a thickness regression. A regressor trained on it would fit noise.
* **The crack wave carries the width, at roughly 100:1.** 4.78× measured over
  the same range against 4.64× from the Krauklis `h^(1/3)` law. So the dataset
  carries both branches, and the gap is sampled log-uniformly — uniform in log
  is uniform in the observable for a cube-root law.
* **The crack wave is recorded, never injected.** At 63-620 m/s it reaches the
  3 m near offset between 4.8 ms and 47.6 ms, against a 5.12 ms record. Only
  the widest gap would even enter the window, so a planted arrival would be
  fiction; `ModeSpec.inject` exists for exactly this.

**A caution for the `sonic_ml` work, and the reason this is the interesting
half of the item.** A 100 µm gap cuts the cement-stiffness sensitivity of the
Stoneley curve from 3.22 % to 0.48 % — about sevenfold. The shipped M5d bond
inverse keys on precisely that sensitivity. It is therefore not merely untested
in the debonded regime: the signal it reads has largely gone there, which is a
different and worse problem than a domain shift. Whatever is built on this
dataset should be scored against that, not around it.

**The classical bar is now in place, and it is a strict one.**
`sonic_ml.baselines.CrackWaveThicknessBaseline` inverts the Krauklis law in
closed form for the gap width. Two things make it a harder baseline than the
bonded `StoneleyBondBaseline` rather than an easier one: it needs no fitted
calibration, so it spends none of the training split; and it is genuinely
independent of the data it scores, since the curves are numerical roots of the
full determinant and the law is the analytic asymptote that validated that
determinant to 0.02 %. Its known weakness is stated rather than hidden — the
law assumes half-space walls, while the stack has ~10 mm of casing and ~45 mm
of cement against a comparable crack wavelength, so the score reports a median
ratio (the bias) separately from the spread (what a recalibration could not
fix).

**Measured, on 24 generated samples spanning 11-837 um:** rank correlation
**0.991**, median ratio 0.935 — the half-space bias is only ~6.5 %, smaller
than expected — and a log RMSE of 0.085, about **21 % in gap width**, falling
to **18.1 %** after removing that one constant. So the closed-form estimator
recovers the gap to under a fifth across two decades having spent no training
data, which is the bar the learned model inherits. It also confirms the
identifiability prediction that reshaped this item.

The bundle needed **no loader change**: `DatasetBundle` reads `mode_names` and
`layer_params` from the file and `cased_features` was already generic over
layer count, so a three-layer two-mode debonded set loads as `is_cased` schema
v4 unmodified.

A CBL-amplitude baseline is *still* not available here, which corrects a
long-standing expectation in this file. The hope was that the debonded regime
would make one fair. It does not: these gathers carry no casing-ring arrival at
all, and `CasingRingAugmentation` deliberately draws ring amplitude
independently of bond precisely so that no model can recover a planted
relationship. What changed is that a better classical estimator now exists —
one reading a signal the physics actually puts in the data.

**The learned model exists; whether it is worth having is not yet measured.**
`sonic_ml.models.debond` predicts the *residual* of the closed-form estimate,
with a zero-initialised head so an untrained model reproduces the classical
answer exactly. That makes any gain attributable — the residual is the
finite-layer correction the half-space law cannot express, and the features
expose exactly what the baseline lacks, the layer thicknesses.

**Measured on 240 samples** (192 train / 24 val / 24 test, gaps 10-961 um):
on the held-out split the closed form scores **18.1 %** in gap width and the
learned residual **2.5 %** — about sevenfold better. It is not memorisation:
best validation lands at epoch 88 of 400 with validation loss falling
throughout, and held-out 2.5 % agrees with whole-dataset 2.3 %. An earlier
24-sample trial had been uninformative and, read carelessly, would have said
the opposite; it is what forced validation-based weight selection, without
which this run would have produced a convincing illusion at larger scale.

**The caveat is the size of the claim, not its direction.** These dispersion
curves are noiseless solver output — no measurement noise, no picking error —
so 2.5 % is a ceiling against a perfect forward model rather than a field
expectation. And on real data the crack wave has to be *detected* first, which
at 63-620 m/s means it arrives outside a normal record. What is established is
that the finite-layer correction the half-space law discards is learnable from
the geometry, which is a modelling result.

**G.6, which was expected to be ordinary wiring and was not.** The model and
the baseline used to be compared by hand in a script; `sonic_ml.bench.debond`
now scores both on identical held-out indices with the same bootstrap and the
same regime rows every other predictor in the layer gets. Two things had to
differ from its sibling harnesses, and both are about not misdescribing the
measurement: errors are reported in **log10 metres**, because a median error
in metres across a two-decade prior is set by the widest samples alone; and
the protocol takes **no `ArrayGeometry`**, because a gap-width estimator reads
the dispersion curve rather than the gather, and handing it a geometry it
cannot use would imply otherwise.

The wiring then found something the by-hand comparison could not. Split by gap
width, the closed form is not uniformly ~5 % off — it is **2.5 % on gaps below
100 µm and 16.5 % above** (n = 142 / 98 over the whole 240-sample set, which
the baseline may legitimately be scored on since it fits nothing; CIs
1.8-2.9 % and 13.6-20.6 %, nowhere near overlapping). That is the direction the
physics predicts, and it is the clearest statement yet of what the residual
model is for: a wider gap carries a faster crack wave and a longer wavelength,
which makes the 10 mm casing and 45 mm cement look thinner relative to it, so
the half-space assumption fails harder exactly where the gap is widest. On the
held-out split the learned inverse reads 0.7 % tight and 1.3 % wide — it
removes the regime dependence rather than just lowering the average.

A note on the two numbers, since both are now in this file. The harness reports
a **median** absolute error; the 18.1 % / 2.5 % figures above are **RMS** in
log units. On these errors the median is about a third of the RMS, and the
difference is the heavy tail — the same errors, two statistics, and the point
of having both is that neither hides what the other shows.

**Costs, because they bound what is practical.** A debonded sample runs ~14 s
against ~0.5 s bonded (the microannulus solvers are ~0.45 s per frequency for
both branches), so `--debonded` defaults to a 32-point grid and a useful set is
a batch job of hours, not a CI artefact.

**No schema change was needed.** The gap is written into `layer_params` as an
ordinary layer with `vs = 0`, so v4 already carries its thickness.
`bond_index` keeps its range and direction but is driven by gap width here and
cement stiffness when bonded — same column, different question, so the two
datasets must not be pooled.

## G. `sonic_ml` — the machine-learning layer

**Status**: shipped through milestones M0-M5f; see `sonic_ml.rst` for the
narrative overview and `sonic_ml/` for the package. In brief: a torch-free spine
(schema-versioned `.npz` loader, provenance, regime-stratified splits,
determinism), a model-agnostic benchmark harness with bootstrap CIs, classical
baselines, and models — a forward dispersion surrogate, a DL-FWI inverse net
with a heteroscedastic head, a low-latency LWD variant, in-house FNO / DeepONet
operator primitives, and a cased-hole forward operator plus cement-bond inverse.

The layer is deliberately isolated: it is a sibling package excluded from the
core wheel and the core CI gate, running in its own non-required workflow, and
`import fwap` never pulls in PyTorch.

**Two results, and the honest gap between them.** In the open hole the learned
inverse recovers `V_S` roughly an order of magnitude more accurately than
classical slowness-time processing on identical held-out gathers. Behind casing,
the cement-bond inverse reaches only about twice the skill of predicting the
mean — because a forward sensitivity sweep shows cement stiffness moves the
cased Stoneley curve ~7 % across its prior while formation `V_S` moves it
~1.5 %, so the problem is only partially identifiable. The uncertainty head
reports calibrated error bars that say so. Publishing only the first number
would be advertising rather than measuring.

**What's open:**

1. **Real-data evaluation** — see section F. Still the binding constraint on
   every claim above, and note what F.2 closing did *not* change: a real DSI
   gather now bounds the *processing chain*, but every number in this section is
   still measured against the forward model that generated its own training
   data. Section F is largely closed; this item is not.
2. **Free-pipe / debonded cased regime.** The cased dataset spans the *bonded*
   regime, where the cased Stoneley stays bound, so the bond inverse grades
   cement quality and is explicitly not a free-pipe detector. The debonded
   generator and its classical baseline have since shipped — see section G.2
   above for both, and for the measurements that reshaped them. ~~What is left
   here is the learned model and its benchmark entry.~~ **Both have since
   shipped too** — the learned residual inverse at 2.5 % held-out (G.2) and its
   scoring through `sonic_ml.bench.debond` (G.6) — so nothing in this entry is
   open except item 1 above. It stayed here for a revision after the fact.

   *Correction, second one on this entry.* It used to add that the debonded
   regime "is also where a CBL-amplitude baseline would finally be a fair
   comparison rather than a strawman". Withdrawn: these gathers carry no
   casing-ring arrival whatever the bond, and `CasingRingAugmentation` draws
   ring amplitude independently of bond on purpose, so a CBL gate would still
   be measuring nothing. The debonded regime supplies a *different* honest
   baseline instead — the crack-wave gap inversion — rather than rehabilitating
   that one.

   *Correction.* This entry used to continue "reaching the debonded regime needs
   a leaky-mode cased forward model, not a planted wavetrain", which filed it
   behind the derivation-blocked `n=1` leaky work in section A. The first half
   is right — a planted wavetrain would not do — and the second half is wrong.
   The standard debonding model is a **fluid microannulus**, which is a
   bound-mode problem needing no complex-plane tracking; two of its three pieces
   have since shipped. This item is therefore gated on **A.5**, not on A.2 --
   and as of this revision A.5's forward model is complete, so what remained of
   that gate is gone: `stoneley_dispersion_microannulus` and
   `crack_wave_dispersion` are both public. This is now the open item blocked
   on nothing.

   Free pipe *proper* — casing surrounded by fluid, the classic CBL casing-ring
   amplitude — remains partly a phenomenological amplitude effect rather than a
   modal one, and that part is unchanged by A.5.
3. **Two-mode cased datasets**, gated on the cased-flexural bracketing in A.2.
4. **Whether `penalty="tv"` should be the default in `sonic_ml.models.joint`.**
   Left open deliberately. `invert_joint` takes `penalty="tv"`, a pseudo-Huber
   cost that is nearly indifferent to how a given amount of change is
   distributed down the log; on a bedded synthetic it beats the squared
   difference on both overall error and bed contacts, and raises
   contact-localisation precision from 0.83 to 0.91 against a 0.36 no-skill bar.
   But a piecewise-constant test bed is the friendliest possible setting for a
   contact-preserving prior, and with contacts ramped over four frames the
   advantage narrows and partly inverts. The default stays `"l2"` because the
   choice turns on how bedded a *typical real* log is — a section F question,
   not one more synthetic sweep can settle it.
5. **Coupling across mode as well as depth** in joint inversion: untouched.

**Deliberately not planned**: shipping trained weights in the repo. Checkpoints
are git-ignored and the committed artefact is the small JSON model card that
binds a checkpoint to its fwap version, config and training-data hash. Weights
are cheap to regenerate and expensive to keep honest.

## Non-goals

These have come up in reviews and been deliberately deferred:

- **GUI / plotting app**. `fwap.plotting` exposes `wiggle_plot` and
  `save_figure` for use in notebooks and scripts. A dedicated GUI is out of
  scope; integrate with Jupyter or your own plotting stack.
- **Production multi-well log management**. `fwap.io.read_las` / `write_las` are
  single-file helpers. A database / catalog layer belongs in a separate package.
- **Time-frequency analysis beyond the STC surface**. Wavelet transforms,
  short-time Fourier, spectrogram picking — all useful, all out of scope for a
  reference implementation of the 1994 book.

`docs/possible_extensions.md` is the companion list of speculative directions,
and it cites these three by number.

## Closed, and where the detail lives

Dropped from this file in the merge, because each is finished and recorded
elsewhere. Nothing here is lost: `CHANGELOG.md` carries the shipped-work entries
and the old roadmap remains in git history.

| Was | Now |
|-----|-----|
| `0.4.0` release notes, and the three post-0.4.0 completeness sweeps | `CHANGELOG.md` |
| **A.3** Leaky-mode branch selection (`branch` argument, `_enumerate_leaky_roots_n0`) | `CHANGELOG.md`; `plans/roadmap_1.md` closed list |
| **A.4** Trapped pseudo-Rayleigh modes (`trapped_pseudo_rayleigh_dispersion`) | as above |
| **B** Quantitative Stoneley permeability (`stoneley_permeability_tang_cheng`) | `CHANGELOG.md` |
| **C** Fully-joint Viterbi extensions (N-mode `viterbi_pick_joint`, variable candidate budget) | `CHANGELOG.md` |
| **E** `ruff format` sweep and the pre-commit hooks | `CHANGELOG.md` |
| **G.4 / G.5 / G.6** surrogate-in-the-loop, joint multi-depth, and bed-boundary-aware inversion | `CHANGELOG.md`; the open residue of G.6 is item 4 above |
| Section A's original from-scratch problem statement, and section B's | git history; both describe solvers that now ship |
