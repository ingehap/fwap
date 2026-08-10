# Using a published paper to constrain a numerical code

A method note, written after taking Schmitt & Cheng (1988) through `fwap`'s
cylindrical solvers figure by figure. It is not a status file — `plans/roadmap.md`
is that, and `plans/roadmap_1.md` is the priority snapshot. `plans/learning.md`
is the sister document for *analytic* oracles; this one is about **published
figures** as oracles, which have different failure modes and a different payoff.

The work it describes found three defects, fixed two of them, and moved the
project's best external agreement from "no tie better than 5 %" to **0.04 % rms**.
It also produced six corrections to its own earlier conclusions, which is the
part worth reading.

---

## 1. Why this paper

**It was supplied, not selected.** The PDF arrived as an upload at the start of
the session; no literature search chose it. That matters for honesty about the
method — the criteria below are reconstructed from why it *worked*, not from a
selection process that happened. They are written as criteria because that is
the reusable part: the next paper should be chosen by them.

What made this one the right paper:

**It is the code's own cited authority.** `fwap` cited Schmitt (1988) in nine
files as the reference for the cylindrical solver. A paper the code already
claims to implement is the strongest possible oracle, because any disagreement
is unambiguously a defect in one of them — there is no "different model, different
answer" escape. (The citation was itself wrong in a way that mattered; see §9.)

**It plots the exact quantity the code returns.** Phase velocity against
frequency, for named rocks, on labelled axes. Not a derived quantity, not a
processed log — the solver's own output. `BoreholeMode.slowness` inverts directly
onto the published `y`-axis.

**It states its inputs completely.** Table 1 gives `V_P`, `V_S`, `ρ`, `Q_α`, `Q_β`
for every medium, and each caption names the borehole radius and fluid. There is
nothing to guess, so a mismatch cannot be blamed on unknown parameters. **This is
the single most important criterion.** A figure without its inputs is a picture,
not an oracle.

**It spans regimes the code treats differently.** Five rocks (fast sandstone,
limestone, granite, slow sandstone, invaded zone), four modes (Stoneley,
pseudo-Rayleigh, flexural, screw), open hole and layered, frequency domain and
time domain. That breadth is what turned a single anomaly into a diagnosis: the
same defect appearing in five independent forms across three rocks is a
mechanism, whereas one bad curve is a bug report.

**It contains internal redundancy.** The same rock appears in several figures;
arrows drawn on waveform panels encode velocities that also appear in Table 1;
group-velocity curves are plotted beside the phase curves they derive from. That
redundancy is what makes the *digitisation* checkable independently of the code
(§5), and it is worth weighting heavily when choosing a paper.

**It is old enough to be independent.** 1988, computed with methods that share no
code with `fwap`. A modern paper using the same open-source solver would be
circular.

### Criteria, condensed

For the next paper, in priority order:

1. Complete, tabulated inputs.
2. Plots the code's own output quantity.
3. Already cited by the code as its authority.
4. Multiple regimes and multiple rocks.
5. Internal redundancy that lets the digitisation be checked without the code.
6. Computationally independent of the code under test.

A paper failing (1) is not usable. A paper failing (5) is usable but every
conclusion carries digitisation risk you cannot bound.

---

## 2. The principle

**A solver's own convergence is not evidence that it converged to the right
thing.** Every defect found here produced finite, plausible, smoothly-varying
numbers. `flexural_dispersion` returned a bound root of the correct determinant,
inside its declared search window, at 43 % of the frequencies asked for — and it
was a different mode from the one requested, on the right branch at **2 of 115**
samples. Nothing internal to the code could have caught that, and nothing did:
the defect had a roadmap entry, a diagnosis, and a proposed fix, all of which
were wrong, and it survived because the diagnosis was written by asking the code
about itself.

The whole method follows from that. It does not mean the internal checks were
worthless — several found real defects — but there is a class of error none of
them can reach, and §3 is about where each instrument is blind.

---

## 3. Four instruments, and what each one cannot see

`plans/learning.md` is the record of the analytic-oracle programme; this section
only places those instruments beside published figures, because the interesting
result is where each is *blind*.

| instrument | catches | structurally blind to |
|---|---|---|
| exact solutions of limiting cases | wrong model, wrong asymptote, wrong scaling | the interior of the band |
| conservation laws | wrong fields for a given mode | **which mode you handed it** |
| structural invariants | non-physical output, branch hopping | absolute accuracy |
| published figures | the interior, mode identity, absolute error | what the code cannot represent |

### Exact solutions are strongest as a constraint on the algorithm, not a check on its output

`scholte_speed`, `rayleigh_speed` and the White tube-wave speed are exact
solutions of limiting cases — a plane interface, a vacuum-loaded half-space, a
quasi-static long wavelength. They anchor every digitised table in this work
(§5), and figure 2a's calibration could not have been verified without them: its
two ends *are* the shear speed and the Scholte speed.

But the sharper use is the one that was missed for a year. `scholte_speed` was
already in the repository, already validated, and A.2's own diagnosis already
named Scholte as the flexural mode's high-frequency limit. What nobody drew was
the next line: **if the branch asymptotes to Scholte, the search window must
contain Scholte.** It did not — the window stopped at `V_R`, some 30 % above —
and that single unmade inference is the whole of A.2.

An exact solution that bounds the answer should be used to bound the *domain the
solver searches*, not merely to test the values it returns. Those are different
tests, and the second passes while the first is failing: the returned overtone
lay comfortably inside the window, so every value-level check was green.

### Conservation laws cannot see mode substitution, and this is structural

`plans/learning.md` reached the key result by deriving the leaky-mode energy
balance in full and then finding that it reproduces `Im(k_z)` **at non-roots as
well as at roots** — so it constrains the fields but not the eigenvalue — and
that momentum is the same balance times a constant.

This work supplies the consequence. `flexural_dispersion` was returning a
*genuine root of the correct determinant*: an ordinary bound trapped mode, just
not the one asked for. It therefore satisfies every conservation law exactly, for
the same reason the energy balance holds at non-roots — a conserved quantity
constrains whatever fields you hand it and says nothing about which solution you
handed it.

So **"right equation, wrong root" is invisible to every instrument that asks the
code about itself**, and no tightening of tolerances changes that. It is not a
gap in the conservation-law programme; it is outside its domain. Recognising this
is what makes an external reference necessary rather than merely desirable.

### Structural invariants are the cheap middle, and they became the fix

Between exact solutions and full references sit facts that are model-independent
but are not conservation laws:

- a guided mode's phase velocity never increases with frequency;
- its group velocity is positive;
- a thicker slow annulus cannot speed the mode up;
- the answer at one frequency must not depend on which *other* frequencies were
  requested in the same call.

These cost nothing, have no tolerance to tune, and did more work here than any
other class. The old solver violated the first two flagrantly — a sawtooth
jumping upward by more than 100 m/s, and `v_g` negative on 18 of 48 adjacent
samples — and those violations are what established the output was not a
dispersion curve at all, before any published number was consulted.

Then they became the repair. **The corrected marcher is an invariant turned into
an algorithm**: "walk up in frequency and keep the slowest root no faster than
the previous one" is monotonicity used as a *selection rule* rather than as a
test. That is the move worth stealing — an invariant strong enough to detect a
defect is often strong enough to prevent it.

The fourth invariant priced what the fix did *not* buy. Afterwards `n=1` returns
the same 10 kHz value across grids of 5 to 161 points and across five different
grid start points, to **0.000 %**; `n=2` still moves by 1.7 % and 3.2 % and
vanishes on some grids. Same fix, same window, same selection rule — so the
`n=2` residual is a root-finding instability rather than the bracket, which is
why `n=1` results are quotable and `n=2` ones are not yet.

### The empirical result of the hierarchy

A.2 survived an analytic-oracle programme, an energy-balance derivation, a
Scholte asymptote check, a rigid-pipe cutoff, biorthogonality and Kramers-Kronig
attempts, and about twelve hundred passing tests. It fell to one digitised
figure. Not because the oracles were weak — several found real defects — but
because they were all asking the code about itself, and this defect was a choice
*between* solutions rather than an error *within* one.

The converse is equally true and less obvious: **the figure work could not have
run without the oracles.** Every digitised table is anchored on an exact
solution, figure 15(b)'s calibration was settled by two shear speeds from
Table 1, and figure 2a's reference is trustworthy only because its two ends are
independently computable. The pairing to aim for is an external reference
*anchored* by exact solutions; neither instrument is sufficient alone.

---

## 4. Hypothesis testing: prefer measurements that relocate the defect

The move that did most of the diagnostic work was not measuring, it was choosing
what to measure. State two explanations, find the observation that must come out
differently under each, and prefer the one that *moves* the defect over the one
that merely confirms it.

The decisive example: the layered flexural path was 31–53 % wrong on figure 12's
fast rock. Two explanations — the propagator chain is broken, or the bracket it
inherits is. The discriminating measurement was to run **the identical call on a
slow rock**, figure 15, where the same propagator tracks the published curves at
**1.47 % rms**. That relocated the defect from the layered code to the bracket in
a single measurement, ruled out rewriting the propagator, and made a prediction —
correcting the bracket alone should bring the fast half in without disturbing the
slow half — which is exactly what happened.

Others that paid, each stated as its discriminating question:

- *Is the sparsity caused by the casing?* Remove the casing: the open hole is
  just as sparse. Not the layer stack.
- *Is the 16 cm trace curve 3 or curve 4?* Run-ordering, not rms — see §5.
- *Is the `n=2` residual a wrong branch or a frequency offset?* Re-score against
  the reference shifted by 1.1 kHz: 5.84 % → 1.35 %. An offset, and one that
  belongs to a different defect.
- *Are the values just above `V_f` real roots or the boundary artefact?* Require
  a true sign change, a smooth descent toward Scholte, and survival of a grid
  change. Real.
- *Is the `n=2` coverage instability universal?* Try a second model: identical
  coverage every time. Model-specific, which bounds the claim.
- *Does widening the bracket alone fix A.2?* No — above 16.4 kHz it returns a
  different wrong branch, 14–39 % high. That answer is why the selection rule
  shipped *with* the bracket rather than after it.
- *Is the new noise guard over-firing, or is the determinant really that bad?*
  Scan it directly: 10–33 sign changes with the layer set identical to the
  formation. The determinant.

Two disciplines make these worth running. **State what the wrong hypothesis
predicts before measuring**, or the result gets read as confirming whatever you
expected. And **check the measurement can discriminate at all**:
`plans/learning.md` records a Scholte sign check that passed with the sign wrong
because both signs reduce to Rayleigh, and this work added its own — an rms
comparison asked to choose between two curves converging to within 0.6–0.8 %,
below the solver's own error. It chose wrongly, and confidently.

---

## 5. Digitisation, and why it needs its own quality control

The reference values are read off a 1988 scan. If the digitisation is wrong the
comparison is worthless *and confidently so*, so the reading needs as much
scepticism as the code.

**Calibrate from ticks, and report the residual in physical units.** Fit tick
positions to their labelled values by least squares; quote the worst residual as
frequency and velocity, not pixels. Figure 15(b): `x` to 0.008 kHz, `y` to
0.00006 normalised. If that budget is not small against the disagreement you are
about to claim, you have no result.

**Let physics confirm the calibration, independently of the ticks.** Figure
15(b)'s `x`-axis runs **2–10 kHz, not 0–10** — nine evenly spaced ticks with "5"
under the fourth. Reading it as 0–10 would have shifted every frequency by 2 kHz
and looked entirely reasonable. What settled it was that curves 1–3 leave the
axis at 0.80 and curve 4 at 0.72 — exactly `1201/1500` and `1081/1500`, the two
shear speeds from Table 1. The physics fixed the axis.

**Anchor every table at a value computable without the solver.** Figure 2a's
low-frequency plateau must be the formation shear speed (2593 read against 2601
exact) and its high-frequency end the Scholte speed (1493 at 24.9 kHz against
1484). Both ends, 1117 m/s apart, so neither is a weak test. Every digitised
table in this work carries such a test, so a bad reading cannot silently become
the reference.

**Overlay the trace back onto the scan and look at it.** This caught two traps
that no numerical check would have:

- figure 8a's `y`-label prints as `0.713` at 600 dpi and is **0.71667**;
- figure 1a draws the Stoneley **group** curve *above* its phase curve, so
  comparing against the upper line gives a plausible −2.5 % that is entirely an
  artefact of tracing the wrong branch.

**Identify curves by structure, not by appearance.** Where curves merge, follow
each from its own start with slope prediction and stop where a neighbour comes
within two line widths. When figure 15(b)'s 16 cm curve was checked, an rms
comparison *preferred the wrong hypothesis* — the data fitted an invaded-only
prediction (1.37 %) better than the layered one (2.12 %). Run-ordering settled
it: the trace sits on run 3 of 4 at every sampled frequency, so it is curve 3.
The apparent preference was only that curves 3 and 4 converge to within 0.6–0.8 %
there, below the solver's own error. **When two hypotheses are closer than your
instrument's precision, rms cannot choose between them and should not be asked
to.**

**Refuse to measure what the scan cannot resolve, and refuse for a positive
reason.** Not "the correlation was low" but "the best-fit lags are +868.9, +867.8
and +862.4 µs at 3, 6 and 7.5 kHz — constant to ±3 µs across a 2.5× change in
source frequency, which is the cycle-hopping signature, not a delay". A refusal
with a mechanism behind it is a finding; a refusal on a threshold is a shrug.
Also refused here: figure 12a's crowded middle (eight curves in a 1.2-wide
window), figure 3's scaling digits ("0.0014" and "0.0019" are indistinguishable
at this scan quality), and figure 6's velocities (self-normalised overlapping
traces).

**Record ambiguities instead of resolving them silently.** Figure 17's 1 kHz
scale factor reads 0.156 or 0.186 and the plotted ink (0.184 ± 0.02 on a
39-pixel excursion) cannot separate them. Comparing the glyph against known 5s
and 8s elsewhere in the same figure settles it — the 8s are two closed bowls and
this is the open-topped 5 of `0.455` — and the ambiguity is recorded in the test
either way, with assertions that hold for both readings.

---

## 6. Comparing, without flattering either side

**Distinguish "wrong" from "missing", and treat them differently.** A `NaN` is a
caller-visible refusal; a wrong number is a silent error. Both were present here,
and conflating them delayed the diagnosis: A.2 had been filed as *sparseness*
(the `NaN`s) when the real defect was the *answers*.

**Coverage is not a health metric, and can invert.** Figure 12 found that adding
a 16 cm invaded zone — strictly harder physics — took `flexural_dispersion`'s
coverage from 9 % to 73 %, with every extra answer an overtone. A caller checking
coverage to decide whether to trust a curve was reading the metric backwards.

**Check the paper's own claims as the authors wrote them.** Page 229 says the
P-wavetrain growth is "especially true with the quadrupole source (Figure 17c,
d)". Read as *absolute level* that is false — the dipole's P/S is larger at
6 kHz. Read as *growth with invaded-zone thickness*, which is what the sentence
says, it holds: 26× and 69× against the dipole's 15× and 13×. Checking the
paraphrase rather than the claim would have produced a spurious "the paper is
wrong".

**Cross-check between figures, owing nothing to the code.** Twenty-four arrows
drawn on waveform panels across figures 16 and 17, read through each panel's own
time axis, give 1198.0 and 1193.6 m/s against Table 1's 1201, and 1083.0 and
1081.3 against 1081 — the latter to **+0.03 %**. That validates the digitisation
pipeline end-to-end using only the paper. Similarly figure 16's Airy arrival
gives 989.6 m/s against figure 8a's differentiated group minimum of 992.0: two
figures, two domains, 0.24 % apart.

**Know what the code cannot represent, and stop there.** Figure 14's content is
peak amplitude, which is excitation × propagation; `BoreholeMode` carries
neither. No amount of solver work reproduces that figure, and saying so is a
result — it bounds what fixing A.2 could buy. The same applies to the excitation
panels (2c, 5c) and, as it turned out, to the attenuation panels: the solvers
take no `Q`, and `attenuation_per_meter` is `None` for every bound mode, so the
published attenuation curves have nothing on the `fwap` side to compare against.

---

## 7. From measurement to fix

**Pin the defect as a test before fixing it**, phrased so it fails when the
defect is repaired. Roughly thirty tests changed meaning when A.2 was fixed, and
that was the design working, not churn. Several had asserted `V_R` as a floor for
the flexural mode — an assertion that was *itself a statement of the bug*.

**Scope the fix from the measurement, not from the diagnosis.** Before touching
A.2: widening the bracket to the Scholte speed recovers 4.4–16.4 kHz at 0.66 %
median error and nothing outside it, because no real-axis root exists there
(4001-point scans find sign changes only at the `s = 0` and `F = 0` endpoints).
That told us three things in advance — a bracket edit for the middle, complex
`k_z` for the ends, and **the selection rule must ship with the bracket**,
because widening alone trades the overtones for a different wrong branch 14–39 %
high. All three held.

**Re-validate against a figure that played no part in designing the fix.**
Figure 3 is a time-domain figure; nothing about it entered the bracket work.
Differentiating the corrected branch predicts its observed Airy arrival to
**+8 %**, where the old bracket implied a wave 2.2× too early. That is the
strongest single piece of evidence the fix is right, and it was available only
because the paper spans two domains.

**Expect the fix to expose the next defect.** Widening the window revealed that
the cased `n=2` determinant produces ~90 sign changes at 12 kHz where physics
supports a handful — and 10–33 even with the layer set identical to the
formation, which is physically the open hole. That is catastrophic cancellation
in the propagator chain (new item A.7), and the narrow bracket had been hiding it
by only ever looking at a sliver. It also exposed a per-layer degeneracy that
pinned an entire curve to 2341.0 m/s. **A fix that reveals nothing new is
suspicious**; these were caught only because pinned tests were watching.

**When the honest answer is silence, ship silence.** The cased `n=2` path now
returns `NaN` rather than a root drawn from noise. That is a behaviour change —
it used to return finite values, and they were wrong — and it is the right trade.

**Search the whole surface for the same wrong belief.** CI caught the A.2 fix
against an oracle in `sonic_ml` asserting the flexural mode stays above `V_R`.
The identical wrong bound, in a second package, passing only because the solver
had the same bound built into its search window and therefore could not produce a
counterexample: **a bug and its test agreeing with each other.** After fixing a
conceptual error, grep for the concept, not the symbol.

---

## 8. Honesty rules this work needed

The two existing rules in `plans/roadmap.md` (state what was measured; do not let
a plan outlive its premise) did not prevent these, so:

**Correct at the site of the claim, then explain below it. Appending is not
correcting.** A reader meets the original sentence first and has nothing to tell
them it was overturned. Applied five times here.

**A figure is not "done" because one of its panels was.** Figure 15(b) sat
undigitised through the figure-15, 16 and 17 work because 15(a) had been read. It
turned out to be the reference that validates the A.6 fix — and A.6 was reachable
from it all along.

**Inventory the whole source before working through it.** Seventeen figures were
digitised on the assumption that seventeen was all of them. The paper has
**twenty-five** and runs to p. 268; figures 18–25 are a cased-hole section that
none of this work touched. The cost was not wasted effort but wrong prioritisation
— cased-hole figures would have been worth more than several that were done.

**A golden file records behaviour, not truth.** Two golden arrays here pinned the
defect: 2591.9 m/s at `n=1` (essentially `V_S`) and 2390.4 at `n=2` (`V_R` to
four figures). Regenerate only with the same verification a new reference gets —
each root checked against the determinant, monotonicity, grid-independence — and
document why.

---

## 9. Failures of this method, all of which happened here

- **A refusal that was wrong.** Figure 13's delays were refused because of a
  clipped extraction window — narrower than the widest trace's own excursion, so
  every correlation ran against a truncated reference. Figure 14 caught it;
  widened, the panel measures. *Check the instrument before trusting a null
  result.*
- **A headline retracted before publishing.** Figure 14(a)'s 36.5 µs against
  figure 13(a)'s 1.2 µs suggests the quadrupole is 30× more delay-sensitive. The
  panels are at different source frequencies. Figure 16 later supplied the
  like-for-like comparison: 45×. *An arresting ratio is a reason to check the
  axes.*
- **A conclusion stated too broadly.** "The layered path is as accurate as the
  open-hole one" held for *phase* velocity and did not survive differentiation —
  the layered group velocity is twice as wrong. Corrected at its site.
- **A defect misfiled.** A.6 was first written up as a scoping decision about
  what the API should support. Reading the docstring showed the code had been
  stricter than its own documented contract all along. *Read the contract before
  theorising about the design.*
- **The citation fix that got the citation wrong.** A commit devoted to
  correcting this paper's citation across nine files fixed the authorship, title
  and venue and left the page range wrong — p. 246 is figure 9, with sixteen
  figures after it.
- **A causal claim that did not survive its own test.** A.8 was recorded as
  "almost certainly also A.7's cause", on the reasonable argument that a
  propagator built from a non-solution has no reason to produce a clean
  determinant. Fixing A.8 refuted it: the layer-equals-formation scan gives an
  identical 430 sign changes at 12 kHz before and after, to the sample. The
  cancellation is in the chain, not in the column. *A mechanism that would
  explain a second defect is a hypothesis, not a diagnosis — and the cheapest
  time to test it is the moment the first defect is fixed.*
- **A conclusion inverted by its own repair.** "The 8 cm layered tie is better
  than its own virgin control" was the measurement that turned A.6 from an
  argument into evidence. After A.8 the control is the tighter of the two
  (0.055 % against 0.136 %), which is what one should expect — the open-hole
  solve carries one homogeneous half-space, the layered one also carries an
  invaded-zone row transcribed from a scan. The A.6 conclusion is unaffected;
  the number that carried it is not. *A comparison between two quantities both
  dominated by the same error says less than it appears to.*

---

## 10. What it produced

Fifty-two digitised reference tables and twelve scalar anchors, each with
provenance, an uncertainty budget and a stated resolution limit, and **90 tests**
referencing them. Every table carries its own anchor test. Three tests were
phrased to start failing if the defect they describe was ever repaired; A.8
repaired all three, and they now assert the agreement instead.

| | before | after A.2/A.6 | after A.8 |
|---|---|---|---|
| best external tie | none better than 5 % | 0.04 % rms (Stoneley, fig 8a) | **0.033 %** |
| slow flexural, fig 8a | 1.29 % rms | 1.29 % rms | **0.063 %** |
| slow screw, fig 8a | 0.94 % rms | 0.94 % rms | **0.058 %** |
| fast flexural, fig 2a | right branch at 2 of 115 | 0.78 % median | **0.16 %** |
| fast flexural, granite | **no correct sample** | 0.87 % median | **0.45 %** |
| invaded zone at `n=2` | `ValueError` | 0.58 % rms | **0.136 %** |
| `n=1` near-cutoff gap | 1.48 kHz | 1.48 kHz | **0.00 kHz** |
| defects known | 1, misdiagnosed | 3 | 5, three fixed |

Two properties of the paper carried this, and they are different in kind. The
figures — five rocks, two domains — made a scattered set of anomalies resolve
into **one mechanism**: a search window anchored to a speed that is not a limit
of the mode it was searching for. The **appendix** then did something the
figures could not, because it prints the matrices rather than their consequences:
it let the formulation be checked term by term, with no root-finding, no
digitisation and no tolerance to argue about. That is what found A.8, and A.8
turned out to be the larger of the two — it moved every `n >= 1` tie in the table
above by an order of magnitude, and closed four defects that had been recorded as
separate solver limitations.

---

## 11. What is left in this paper

Recorded here because §8 says to inventory first, and because the next person
should not have to rediscover it.

1. **Figures 20 and 21** — cased-hole dispersion and attenuation: cement
   thickness effects, cement shear-velocity effects, and the same for the screw
   mode. Figure 21 covers exactly the configuration A.7 forced into silence, so
   it is the only external measure of how wrong that path was.
2. **Table 1's casing and cement rows** (6098/3354/7500; 2823/1729/1920;
   2823/1555/1730) — the repo's cased-hole fixtures use invented values.
3. ~~**The Appendix (pp. 235–236)** gives the layer matrix `T(n, j, r)`
   explicitly.~~ **Done, and it found more than it was sent for.** It was listed
   here as the only route that could *resolve* A.7 rather than characterise it.
   It did something better: checking fwap's **formulation** against it — rather
   than its output — showed that the SV column of the layer matrices is not a
   solution of the elastodynamic equations at all (roadmap A.8). That cost about
   1.5 % on the layered `n=1` output and an order of magnitude on every open-hole
   `n >= 1` tie; it is now fixed, and the fix closed five further defects the
   suite had recorded as solver limitations. It was **not** A.7's cause, which
   the same instrument settled: with the column corrected the layer-equals-
   formation scan gives an identical 430 sign changes at 12 kHz, so the
   cancellation is in the propagator chain rather than in what is fed to it. A
   prediction that fails cleanly is still a result — see §9.
   The method is worth generalising: **a paper that prints its matrices lets you
   test the formulation term by term, with no root-finding, no digitisation and
   no tolerance to argue about.** The decisive test needed no reference values —
   only the observation that a solution must be a *fixed* combination of any
   basis for the solution space, so its coefficients cannot drift with radius.
   That is a stronger instrument than anything in §3, and it belongs in the table
   there: it catches errors in the model itself, which is the one thing output
   comparison can only ever see indirectly.
4. **Figure 8(b)** — a ±10 % `V_S` sensitivity study, constraining the solver's
   derivative rather than its value. Directly relevant to inversion.
5. **Figure 12(b)** — the fast-formation invaded screw mode, twin of the 15(b)
   panel that validated A.6.
6. **Figures 18, 19, 22–25** — cased-hole and invaded-zone waveform panels.

Not usable against the current code, and why: the **attenuation panels** (1b, 2b,
5b, 7, 8, 12, 15) are computed from Table 1's `Q` values, and no `fwap` dispersion
API accepts `Q`; the **excitation panels** (2c, 5c) need a quantity
`BoreholeMode` does not carry.
