# Digitised reference data for the validation notebook

`cylindrical_biot_validation.ipynb` overlays `fwap` dispersion
output on top of digitised reference curves. The reference CSVs
live in this directory.

## Status

**Thirty-four curves are shipped, and all thirty-four pass.**

| File | Solver | Score |
|------|--------|-------|
| `schmitt_cheng_1987_fig8a_flexural_slow.csv` | `flexural_dispersion` | **0.04 %** RMS, 55/55 pts |
| `schmitt_cheng_1987_fig2_flexural_fast.csv`  | `flexural_dispersion` | **0.37 %** RMS, 61/89 pts |
| `ellefsen_cheng_schmitt_1988_fig4_flexural_vti_soft.csv` | `flexural_dispersion_vti` | **0.30 %** RMS, 70/73 pts |
| `ellefsen_cheng_schmitt_1988_fig4_flexural_iso_soft.csv` | `flexural_dispersion` | **0.17 %** RMS, 73/73 pts |
| `ellefsen_cheng_schmitt_1988_fig2_flexural_iso_hard.csv` | `flexural_dispersion` | **0.45 %** RMS, 17/73 pts |
| `ellefsen_cheng_schmitt_1988_fig2_flexural_vti_hard.csv` | `flexural_dispersion_vti` (fast-formation TI) | **0.55 %** RMS, 20/73 pts |
| `tubman_cheng_toksoz_1984_fig4a_stoneley_open.csv` | `stoneley_dispersion` | **2.23 %** RMS, 67/67 pts |
| `tubman_cheng_toksoz_1984_fig4b_stoneley_cased.csv` | `stoneley_dispersion_layered` | **2.34 %** RMS, 43/43 pts |
| `tubman_cheng_toksoz_1984_fig4a_pseudo_rayleigh1_open.csv` | `trapped_pseudo_rayleigh_dispersion` (branch 0) | **2.81 %** RMS, 59/59 pts |
| `tubman_cheng_toksoz_1984_fig4a_pseudo_rayleigh2_open.csv` | `trapped_pseudo_rayleigh_dispersion` (branch 1) | **3.20 %** RMS, 26/26 pts |
| `tubman_cheng_toksoz_1984_fig4b_pseudo_rayleigh1_cased.csv` | `trapped_pseudo_rayleigh_dispersion_layered` (branch 0) | **3.12 %** RMS, 39/39 pts |
| `tubman_cheng_toksoz_1984_fig4b_pseudo_rayleigh2_cased.csv` | `trapped_pseudo_rayleigh_dispersion_layered` (branch 1) | **3.84 %** RMS, 24/26 pts |
| `sinha_asvadurov_2004_fig10a_quadrupole_fast.csv` | `quadrupole_dispersion` | **0.01 %** RMS, 223/257 pts |
| `sinha_asvadurov_2004_fig19a_quadrupole_slow.csv` | `quadrupole_dispersion` | **0.01 %** RMS, 267/299 pts |
| `sinha_asvadurov_2004_fig2a_stoneley_fast.csv` | `stoneley_dispersion` | **0.01 %** RMS, 245/245 pts |
| `sinha_asvadurov_2004_fig2a_pseudo_rayleigh_fast.csv` | `trapped_pseudo_rayleigh_dispersion` (branch 0) | **0.01 %** RMS, 161/162 pts |
| `sinha_asvadurov_2004_fig6a_flexural_fast.csv` | `flexural_dispersion` | **0.01 %** RMS, 64/114 pts |
| `sinha_asvadurov_2004_fig15a_flexural_slow.csv` | `flexural_dispersion` | **0.01 %** RMS, 238/238 pts |
| `sinha_asvadurov_2004_fig11a_stoneley_slow.csv` | `stoneley_dispersion` | **0.01 %** RMS, 204/257 pts |
| `sinha_asvadurov_2004_fig11a_leaky_compressional_slow.csv` | `leaky_compressional_dispersion` | **0.03 %** RMS, 107/107 pts |
| `paillet_cheng_1986_fig12a_leaky_compressional_fundamental.csv` | `leaky_compressional_dispersion` (branch 0, `tool_radius` 0.05) | **1.81 %** RMS, 136/170 pts |
| `paillet_cheng_1986_fig12a_leaky_compressional_first.csv` | `leaky_compressional_dispersion` (branch 2, `tool_radius` 0.05) | **0.35 %** RMS, 65/84 pts |
| `sinha_asvadurov_2004_fig11b_leaky_compressional_group_slow.csv` | `leaky_compressional_dispersion` (group slowness) | **0.59 %** RMS, 51/51 pts |
| `sinha_asvadurov_2004_fig11c_leaky_compressional_attenuation_slow.csv` | `leaky_compressional_dispersion` (**attenuation**) | **0.32 %** RMS, 93/99 pts |
| `sinha_asvadurov_2004_fig2a_leaky_compressional_fast.csv` | `pseudo_rayleigh_dispersion` (branch 1) | **1.06 %** RMS, 154/161 pts |
| `sinha_asvadurov_2004_fig2c_leaky_compressional_attenuation_fast.csv` | `pseudo_rayleigh_dispersion` (**attenuation**, branch 1) | **4.51 %** RMS, 153/160 pts |
| `schmitt_cheng_1987_fig20a_flexural_cased_cement1_1cm.csv` | `flexural_dispersion_layered` | **0.39 %** RMS, 82/93 pts |
| `schmitt_cheng_1987_fig20a_flexural_cased_cement1_3cm.csv` | `flexural_dispersion_layered` | **0.52 %** RMS, 82/93 pts |
| `schmitt_cheng_1987_fig20b_flexural_cased_cement2_3cm.csv` | `flexural_dispersion_layered` | **0.55 %** RMS, 83/93 pts |
| `schmitt_cheng_1987_fig21a_screw_cased_cement1_1cm.csv` | `quadrupole_dispersion_layered` | **0.86 %** RMS, 89/93 pts |
| `schmitt_cheng_1987_fig21b_screw_cased_cement1_3cm.csv` | `quadrupole_dispersion_layered` | **0.18 %** RMS, 89/94 pts |
| `schmitt_cheng_1987_fig21b_screw_cased_cement2_3cm.csv` | `quadrupole_dispersion_layered` | **0.26 %** RMS, 92/94 pts |
| `yang_lv_2022_fig2a_flexural_cased_hard.csv` | `flexural_dispersion_layered` | **0.38 %** RMS, 71/105 pts |
| `yang_lv_2022_fig2b_flexural_cased_soft.csv` | `flexural_dispersion_layered` (**slow formation**) | **0.017 %** RMS, 12/12 pts |

**The twelve Sinha & Asvadurov rows are extracted, not traced**, and that is why
they score two orders of magnitude tighter than everything else here.
Sinha & Asvadurov's figures are *vector* artwork, so the curve
coordinates come out of the PDF drawing operators exactly; there is no
pixel-tracing error to speak of. The calibration was checked against the
figures' own dashed reference lines before any curve was read -- fig
10(a) puts L / S / C at 666.75 / 492.08 / 273.31 us/m against the
1e6/1500 = 666.67, 1e6/2032 = 492.13 and 1e6/3658 = 273.37 the paper's
own Table 1 implies, agreeing to 0.02 %. For these twelve rows the 5 %
budget is therefore measuring the solver rather than the digitising,
which is not true of the raster-traced rows above. Fig 2(a) carries its
own anchor: the trapped pseudo-Rayleigh branch begins at 492.1 us/m
against the 1e6/2032 = 492.13 Table 1 implies, so the cut-off lands on
the shear line to three digits with nothing fitted to make it.

Their coverage gaps are the reference overhanging the solver, not the
solver failing. Fig 10(a)'s m=1 curve is drawn from 3.2 kHz, and its
first couple of kilohertz sit marginally *faster* than V_S (483 us/m
against the 492 us/m S line -- a real feature of the published curve at
that calibration accuracy). A bound quadrupole cannot be faster than
V_S, so `quadrupole_dispersion` returns `NaN` there rather than a wrong
root. The CSVs ship whole; they are not trimmed to the band the solver
likes.

**The Paillet & Cheng rows are the exception in the other direction.**
They are a raster scan of a 1986 journal page. The y axis is good --
the figure's own dotted lines come out at 2.0011 and 1.5039 km/s against
Table 1's 2.0 and 1.5 -- but the x axis has six unevenly spaced ticks
whose least-squares residuals reach 0.36 kHz, about **1.4 % of full
scale**. That is scan distortion and it is not removable by re-tracing,
so the fundamental's 1.81 % is close to the floor this reference can
support. They are shipped because they are the **only** external
evidence the rigid logging tool has: the same fundamental scores 10.66 %
with `tool_radius` left out.

**One row is not a slowness at all.** The fig 11(c) row holds a
radiation attenuation in **dB/m**, and it is loaded with
`load_reference_curve(..., quantity="attenuation")` so it gets its own
unit guard rather than the slowness one. That guard is much weaker than
its slowness sibling and deliberately so: attenuation has no tight prior,
and it cannot tell dB/m from nepers/m -- the factor 8.686 this figure's
whole convention question turns on -- nor reject a slowness column,
because the two bands overlap and fig 11(c)'s own values start at
0.0025 dB/m. The `quantity` argument is a *declaration*, not a
detection.

**The dB convention had to be recovered, because the paper never states
one.** All six of Sinha & Asvadurov's attenuation panels are labelled
only "Attenuation (dB/m)" with no defining equation anywhere in the
text. Read naively as `8.686 Im(k_z)`, fwap comes out about 2.2x high,
drifting 2.30 to 2.14 across the band. The relation that fits is

    Sinha dB/m = 8.686 * Im(k_z) * (V_g / V_p) / 2

and it was confirmed the falsifiable way: inverting the ratio implies a
group slowness of about 681 us/m, nearly flat above 8 kHz, and that
prediction was written down *before* fig 11(b) was opened. Fig 11(b),
calibrated independently on its own gridlines, reads 681.7 -- agreement
to **0.65 % RMS over 21 points**, against a panel that played no part in
deriving the relation.

That doubles as fig 11(c)'s calibration check, which it could not
otherwise have: attenuation panels carry no dashed reference lines to
verify against. A y-scale wrong by any factor would have thrown the
implied group slowness off by the same factor. Fig 11(b) gets its own
check from fig 11(a): the Stoneley mode is nearly non-dispersive at the
top of the band, and the gap between its group and phase slownesses
closes 1.04 % -> 0.42 % from 8 to 15 kHz, which a scale error could not
produce.

**The two fig 2 rows closed the last gap in the table.** Until they
landed, `pseudo_rayleigh_dispersion` was the only leaky solver in the
package with no published curve behind it — before or after the
radiation-branch correction, which is exactly how that defect survived
so long. Sinha calls the curve a *leaky compressional* mode, which is
why it sat unclaimed: in a **fast** formation that window,
`1/V_P < s < 1/V_S`, is the one `pseudo_rayleigh_dispersion` tracks, and
the curve is `branch=1` because this formation's trapped branches cut
off at 7.45 and 15.6 kHz and m=3 is the second one's continuation.

An earlier attempt at exactly this comparison returned **11.3 %** and
was rejected as the wrong mode. It was the wrong branch index, on a
contaminated radiation branch, with grid-dependent seeding; with all
three fixed the same comparison lands at 1.06 %. The lesson is not that
the rejection was wrong — it was right on the evidence available — but
that "wrong mode" was one of three things wrong at once.

Both rows degrade at the cut-on, where fwap reaches `1/V_P` at 9.17 kHz
against the figure's 8.95 — a 2.5 % offset in frequency, on a curve that
is near-vertical there. That offset is reported as measured rather than
absorbed: near cut-on the traced curve's *frequency* is the
well-determined coordinate, so this is not digitising error. Its
derivative is what could not be shipped — see `pending/README.md`.

**The fig 11(a) leaky compressional row is the one that found a bug**,
and it is worth saying which kind. It is the first *leaky* mode in this
table -- every other row was a bound mode, and every analytic oracle in
the notebook looks at a bound mode's real part. The fig 11(c) row above
is the first reference of any kind here that scores an **imaginary**
part. `_k_or_hankel`'s
radiation branch had been two thirds *incoming* since it was written,
passing every local test it had because those tests are satisfied by
the incoming wave too. Scored against this curve it gave 0.39 % RMS
with a spurious +-0.8 % sawtooth and a root growing along the borehole;
corrected, 0.02 % and smooth. No amount of adding bound-mode overlays
would have surfaced it.

**One panel needed a different calibration method, and finding out why
matters.** Deriving the axis scale from the panel's frame rectangle works
to 0.02 % on figs 2(a), 10(a) and 6(a), but puts fig 15(a)'s C reference
line **0.9 % low** — that panel's frame rect is not exactly its axes box.
Fig 15(a) is therefore calibrated from its **gridlines**, which are
independent of Table 1, and checked afterwards against its dashed lines:
1968.7 / 666.45 / 528.91 us/m against the 1968.5 / 666.67 / 529.13 Table 1
implies, agreeing to 0.04 %. Note the gridline layouts differ — fig 6(a)
runs nine lines over 0-800, fig 15(a) five over 500-2500 — and assuming a
common layout is exactly what produced the 0.9 % error before it was
caught. **Calibrate each panel, check it, and do not carry an assumption
between panels of the same figure.**

**Fast-formation TI is implemented, and the curve that was waiting for it
is now scored.** `flexural_dispersion_vti` raised `NotImplementedError`
above `V_Sv > V_f` from the day it shipped, so
`..._fig2_flexural_vti_hard.csv` sat in `pending/` unusable. The complex
VTI determinant now covers that regime and the curve scores **0.55 %**
over 20 of 73 points. It is the only overlay here that exercises the VTI
determinant with an *oscillatory* fluid field — every other VTI tie is
slow-formation, where the fluid Bessels decay.

**Four of the passes are scored thinly** and say so: the fast-formation
flexural path returns `NaN` outside a narrow band rather than a wrong root,
so `..._fig2_flexural_iso_hard.csv` covers 17 of 73 points,
`schmitt_cheng_1987_fig2_flexural_fast.csv` 61 of 89, and
`sinha_asvadurov_2004_fig6a_flexural_fast.csv` 64 of 114 (NaN above
9.65 kHz), and `..._fig2_flexural_vti_hard.csv` 20 of 73 (2.6-7.7 kHz).
Outside those bands **the overlay is silent, not green**. The four
together bracket where that band limit falls across four different fast
geometries — and the TI one lands in the same place as its
equivalent-isotropic sibling, which is the expected behaviour rather than
a coincidence: the limit is a property of the fast-formation marcher, not
of the anisotropy.

**Fig 11(a)'s coverage gap is a mode boundary, exactly.** Its published
m=1 branch runs to `f` -> 0 and ends at 1527.8 us/m — *faster* than `V_S`
(1968.5). `stoneley_dispersion` scores 204 of 257 points, and the 53 it
skips are precisely the 53 lying faster than `V_S`. Above `V_S` the mode
radiates shear into the formation: it is leaky, not bound, and Sinha plots
leaky branches (panel (c) of that figure gives their radiation
attenuation). The solver returns bound roots only and declines there.

**That figure also supplies external evidence for a regime the package
previously only called invalid.** The White tube-wave expression
`V_f / sqrt(1 + rho_f V_f^2 / (rho V_S^2))` was evaluated for formation B
*before* the figure was opened and gave 1526.8 us/m; the traced curve's
`f` -> 0 end reads 1527.8, agreeing to **0.07 %**. So below the validity
floor the formula still predicts the branch's low-frequency asymptote —
what changes is that the branch is leaky rather than bound.
`fwap.tube_wave_speed` still raises there, correctly, since a speed above
`V_S` is not a bound mode; its message no longer says the wave "ceases to
exist at low frequency", which that figure contradicts. The surrounding
docstrings were already precise — they scope the claim to *bound* roots,
and the modal-determinant scan behind it is a real measurement — so they
are unchanged.

**`flexural_dispersion` now has two independent ties.** Sections 2a and 5
both rest on raster traces of Schmitt-lineage figures; a systematic bias in
that lineage would not show up by agreeing with it twice. Sinha & Asvadurov
figs 6(a) and 15(a) come from different artwork, a different group and a
different decade, and are extracted from vector paths rather than traced.
They agree at 0.01 %.

**The package has two pseudo-Rayleigh functions and they are different
modes.** `pseudo_rayleigh_dispersion` tracks the **leaky** n=0 root, phase
velocity between `V_S` and `V_P` (slowness in `(1/V_P, 1/V_S)`), formation
S wave radiating outward. `trapped_pseudo_rayleigh_dispersion` tracks the
classical **trapped** mode, `V_f < c < V_S`. Tubman's figure plots the
trapped mode — its curves run from `V_S` at cutoff down toward `V_f` — so
these overlays use the trapped function. Scoring the figure against the
leaky one gives 36 % and 51 %, which is a category error rather than a
solver defect; that mistake was made and corrected in this branch.

**Branch identity is measured, not assumed.** Pairing figure curve 1 with
branch 1, or curve 2 with branch 0, scores **25.9 %** and **27.3 %** against
the 2.81 % and 3.20 % of the correct pairing. The match picks out the
branch. The cased pair was checked the same way, and the diagonal is just
as clean: curve 1 scores **3.12 %** against branch 0 and 20.2 % against
branch 1; curve 2 scores **3.84 %** against branch 1 and 16.7 % against
branch 0.

**Fig 4(b)'s two pseudo-Rayleigh curves needed a new solver, and the
reason is worth stating.** They sat in `pending/` because the package had
no cased pseudo-Rayleigh entry point, and adding one meant a **complex**
n=0 cased determinant. The real one returns NaN across this entire window
for two independent reasons: `F_f^2 <= 0` at every phase velocity above
`V_f`, and any layer with `s^2 <= 0` — which the cement is, since its
`V_S` is 1728 m/s while the modes run at 1906-2464 m/s.

Neither refusal means the mode is unbound. **Boundedness is set by the
formation half-space alone**, because only it extends to infinity; an
annulus of finite thickness may oscillate freely. The formation stays
evanescent throughout (all 65 points sit below its 2600 m/s `V_S`), so
`k_z` is real and nothing radiates. The search window is `V_f < c < V_S`
on the formation, exactly as in the open hole.

**The 2-3 % on all four Tubman ties is expected, not slack digitising.**
Table 1 carries `Q` — the fluid is `Q_alpha` = 20 — so the published curves
include intrinsic attenuation while these solvers are elastic. An elastic
solver runs faster than a `Q` = 20 medium here, and all four overlays come
in 2-3 % high with the same sign. Read them as ties with a physical floor.

These are deliberately *independent* of the reads recorded under roadmap
A.1, which live as constants in `tests/test_cylindrical_solver.py` rather
than as CSVs — different session, different resolution (400 dpi here),
different tracer. They agree: this trace puts fig 2(a) at 1494 m/s at
24.5 kHz against A.1's 1493 m/s at 24.9 kHz. Exporting A.1's constants
into this directory instead would have made the overlay a restatement of
a check that already exists.

**The cased dipole and quadrupole are tied for the first time**, by
Schmitt & Cheng figs 20 and 21 — six curves, 0.18-0.86 % RMS. Until
these, `flexural_dispersion_layered` and `quadrupole_dispersion_layered`
had only internal consistency checks behind them: N=1 agreeing with the
single-interface determinant, two half-thickness layers agreeing with
one, layer order mattering. All true, none of it evidence that the
curve is the published one.

Three things about that set are worth recording.

**The digitising floor is measured rather than assumed.** Cement 1 at
3 cm is plotted in *both* panels of both figures, so it was traced twice
from independent artwork with independent calibration. The two fig 20
renderings agree to **0.23 % RMS** and the two fig 21 renderings to
**0.55 %** over 7.2-13.5 kHz, above which the panel-(a) copy runs into
the figure's own "/phase" annotation and stops being traceable. Only the
cleaner panel is shipped for that case. Read the 0.18-0.86 % scores
against those numbers: most of the residual is the 1987 scan.

**One traced curve was rejected rather than shipped.** Each panel also
carries the open hole as curve (1). Fig 20's copy crosses case (3)'s
*group* branch inside its steep segment and the trace hops there,
reading **3.8 % RMS** against the already-shipped fig 2(a) rendering of
the identical curve — which `flexural_dispersion` matches at 0.37 %. A
trace that disagrees with its own sibling rendering by ten times the
solver's error is a digitising failure; shipping it would have scored
the tracer. Fig 21's open-hole copy traces cleanly (0.71 %) but is
redundant with fig 2(a), so neither open-hole curve is in the table.

**The attenuation panels are deliberately not scored.** Figs 20 and 21
plot `(1/Q) x 100`, and that `Q` is table 1's *intrinsic* attenuation
(casing 1000, cement `Q_beta` = 30, sandstone 60). These solvers are
elastic and return no intrinsic loss, so the lower panels are a
category mismatch rather than a failing overlay — unlike Sinha fig
11(c), where the attenuation being scored is radiation damping.

**The cased *leaky* dipole still has no curve, and now has a citation.**
The same report states the behaviour outright on p. 231: behind casing
in a slow formation "the high frequency part of the fundamental modes
excited either by a dipole or a quadrupole source will then also be
leaky", travelling "with a velocity higher than that of the formation
shear wave". Schmitt & Cheng illustrate that case with **waveforms**
(figs 24 and 25), not a dispersion curve, so there is nothing to trace.
`tests/test_cylindrical_solver.py` carries the claim as a test instead,
and scanning the real cased determinant over the whole bound window at
their parameters finds no sign change at any frequency from 0.5 to
14 kHz — to within 1e-9 of `V_S`, so not a resolution artefact.

That geometry also exposes a search-window limit worth a number.
`_march_leaky_cased_branch` looks in `(V_S, min(V_f, min layer V_S))`.
For Schmitt & Cheng's slow sandstone behind 1.02 cm of steel and 3 cm of
cement 1 that ceiling is the *fluid*, 1500 m/s, and the branch runs
above it — leaving `V_S` near 1.4 kHz, peaking near 1710 m/s at 5.5 kHz
just under the cement's 1729, and coming back down through `V_f` near
13.8 kHz.

**That gap had two causes, and the seeding one is now fixed.** Above
3 kHz the root is outside the window and the marcher is right not to
find it. At 1.5–2.5 kHz it is *inside* the window and used to be missed
anyway. Rebuilding the seeding recovers that leg in full — 1235.9,
1300.0, 1358.3, 1412.0, 1461.2 m/s — and an argument-principle contour
confirms it is exactly the window's contents: one root at each of those
five frequencies and none at 1.00, 1.25, 2.75 or 3.00. Letting the
marcher **re-acquire** after a gap rather than stopping two misses into
one then extended the upper leg down from 14.00 to 13.25 kHz; a
**downward pass** now walks each leg back from wherever it was entered;
and withdrawing a dead band that had been held off the window ceiling
takes it to 13.00.

What remains is the ceiling itself. Between roughly 3 and 13 kHz the branch is
above `V_f`, outside the window the marcher searches at all; a contour
still counts one root there at 3.0, 5.5 and 8.0 kHz. Closing that half
needs the borehole fluid field handled as oscillatory rather than
evanescent, which is a determinant question rather than a search one.

**The cased dipole is tied twice, and the second tie reaches a slow
formation.** Yang et al. (2022) fig 2 plots the same mode from a
different group, a different decade and *vector* artwork. Its table 1
gives `V_P` and density for two of the eight formations it sweeps, so two
curves ship: the hard one at **0.38 %** over 71/105 and the soft one --
`V_S` = 1450 m/s, below the borehole fluid's 1500 — at **0.017 % over
12/12**, the tightest cased-hole tie here by two orders of magnitude.

Three things distinguish it from section 4b's traces.

**It is extraction, not tracing.** The source is vector, so the figure is
rendered at 600 dpi and the curves separated by ink level and mark size:
the thick grey line by its fill colour, the dotted line by its marks
being 7×8 px squares where the dashed and dash-dot lines run 16–31 px.
Two calibration checks fall out of the artwork itself — the gridlines
recover to **0.0005** in normalised velocity, and the hard curve's flat
top reads 2999.8 m/s against table 1's 3000, **0.007 %**, with nothing
fitted.

**It is a modal root, not a semblance pick.** The paper states
`D₁(k_z, ω) = 0` and `v = ω/k_z`, so it is the same object this solver
computes — unlike the 3DFD dispersion analyses in the neighbouring
literature.

**It still does not tie the cased leaky dipole.** Yang et al. stop at
their mode's cutoff, 15.04 kHz, and the first traced dot sits at 15.13;
all twelve points are below `V_S/V_f` = 0.9667, so the published branch
is **bound**. Below the cutoff `fwap` continues the same branch as a
*leaky* root over 12.10–14.75 kHz, and there is no published curve there.
What this figure adds is a hard tie on the bound side of a cutoff whose
leaky side is still cited rather than scored.

**Nothing is outstanding.** Every section of the validation notebook now
has a reference. The `TODO: digitise <FIGURE>` path in `check_overlay`
still exists and still fires for a missing CSV — it is how a deleted or
renamed reference announces itself rather than silently passing — but no
section reaches it today.

**The scoring is already wired.** Dropping a CSV here needs no notebook
edit: the section's `check_overlay(...)` call switches from printing
the TODO to plotting the overlay, printing an RMS verdict, and
asserting the 5 % budget. The machinery lives in `fwap.validation` and
is covered by `tests/test_validation.py`, so it is known to work before
any real reference exists.

## Schema

One CSV per published figure, two columns, no header rows.

```
freq_hz, slowness_s_per_m
1000.0,  0.000597
2000.0,  0.000601
...
```

Suggested filenames (matching the notebook section titles):

| File                                  | Reference                                                    | Mode                       |
|---------------------------------------|--------------------------------------------------------------|----------------------------|
| `sinha_asvadurov_2004_fig2a_stoneley_fast.csv` | Sinha & Asvadurov 2004 fig 2(a) *(shipped)* | Stoneley, fast formation |
| `sinha_asvadurov_2004_fig2a_pseudo_rayleigh_fast.csv` | Sinha & Asvadurov 2004 fig 2(a) *(shipped)* | trapped pseudo-Rayleigh, fast |
| `sinha_asvadurov_2004_fig2a_leaky_compressional_fast.csv` | Sinha & Asvadurov 2004 fig 2(a) *(shipped)* | leaky m=3, fast formation |
| `sinha_asvadurov_2004_fig2c_leaky_compressional_attenuation_fast.csv` | Sinha & Asvadurov 2004 fig 2(c) *(shipped)* | same mode's attenuation, **dB/m** |
| `schmitt_cheng_1987_fig8a_flexural_slow.csv` | Schmitt & Cheng 1987 fig 8(a) *(shipped)*             | flexural, slow sandstone   |
| `schmitt_cheng_1987_fig2_flexural_fast.csv` | Schmitt & Cheng 1987 fig 2(a) *(shipped)*             | flexural, fast sandstone   |
| `schmitt_cheng_1987_fig20a_flexural_cased_cement1_1cm.csv` | Schmitt & Cheng 1987 fig 20(a) *(shipped)* | cased flexural, 1 cm cement 1 |
| `schmitt_cheng_1987_fig20a_flexural_cased_cement1_3cm.csv` | Schmitt & Cheng 1987 fig 20(a) *(shipped)* | cased flexural, 3 cm cement 1 |
| `schmitt_cheng_1987_fig20b_flexural_cased_cement2_3cm.csv` | Schmitt & Cheng 1987 fig 20(b) *(shipped)* | cased flexural, 3 cm cement 2 |
| `schmitt_cheng_1987_fig21a_screw_cased_cement1_1cm.csv` | Schmitt & Cheng 1987 fig 21(a) *(shipped)* | cased screw, 1 cm cement 1 |
| `schmitt_cheng_1987_fig21b_screw_cased_cement1_3cm.csv` | Schmitt & Cheng 1987 fig 21(b) *(shipped)* | cased screw, 3 cm cement 1 |
| `schmitt_cheng_1987_fig21b_screw_cased_cement2_3cm.csv` | Schmitt & Cheng 1987 fig 21(b) *(shipped)* | cased screw, 3 cm cement 2 |
| `yang_lv_2022_fig2a_flexural_cased_hard.csv` | Yang et al. 2022 fig 2(a) *(shipped)* | cased flexural, hard formation |
| `yang_lv_2022_fig2b_flexural_cased_soft.csv` | Yang et al. 2022 fig 2(b) *(shipped)* | cased flexural, **slow** formation |
| `sinha_asvadurov_2004_fig11a_stoneley_slow.csv` | Sinha & Asvadurov 2004 fig 11(a) *(shipped)* | Stoneley, slow formation |
| `sinha_asvadurov_2004_fig11a_leaky_compressional_slow.csv` | Sinha & Asvadurov 2004 fig 11(a) *(shipped)* | leaky compressional m=3, slow |
| `sinha_asvadurov_2004_fig11b_leaky_compressional_group_slow.csv` | Sinha & Asvadurov 2004 fig 11(b) *(shipped)* | same mode's group slowness |
| `sinha_asvadurov_2004_fig11c_leaky_compressional_attenuation_slow.csv` | Sinha & Asvadurov 2004 fig 11(c) *(shipped)* | same mode's radiation attenuation, **dB/m** |
| `paillet_cheng_1986_fig12a_leaky_compressional_fundamental.csv` | Paillet & Cheng 1986 fig 12(a) *(shipped)* | leaky compressional, shale B + 5 cm tool |
| `paillet_cheng_1986_fig12a_leaky_compressional_first.csv` | Paillet & Cheng 1986 fig 12(a) *(shipped)* | leaky compressional first mode, same |
| `sinha_asvadurov_2004_fig6a_flexural_fast.csv` | Sinha & Asvadurov 2004 fig 6(a) *(shipped)* | flexural, fast formation |
| `sinha_asvadurov_2004_fig15a_flexural_slow.csv` | Sinha & Asvadurov 2004 fig 15(a) *(shipped)* | flexural, slow formation |
| `sinha_asvadurov_2004_fig10a_quadrupole_fast.csv` | Sinha & Asvadurov 2004 fig 10(a) *(shipped)* | quadrupole, fast formation |
| `sinha_asvadurov_2004_fig19a_quadrupole_slow.csv` | Sinha & Asvadurov 2004 fig 19(a) *(shipped)* | quadrupole, slow formation |
| `tubman_cheng_toksoz_1984_fig4a_stoneley_open.csv` | Tubman/Cheng/Toksoz 1984 fig 4a *(shipped)*     | Stoneley, open hole        |
| `tubman_cheng_toksoz_1984_fig4b_stoneley_cased.csv` | Tubman/Cheng/Toksoz 1984 fig 4b *(shipped)*    | Stoneley, cased hole       |
| `tubman_cheng_toksoz_1984_fig4a_pseudo_rayleigh1_open.csv` | Tubman/Cheng/Toksoz 1984 fig 4a *(shipped)* | pseudo-Rayleigh 1, open |
| `tubman_cheng_toksoz_1984_fig4a_pseudo_rayleigh2_open.csv` | Tubman/Cheng/Toksoz 1984 fig 4a *(shipped)* | pseudo-Rayleigh 2, open |
| `ellefsen_cheng_schmitt_1988_fig2_flexural_vti_hard.csv` | Ellefsen/Cheng/Schmitt 1988 fig 2 | elastic VTI flexural, hard |
| `ellefsen_cheng_schmitt_1988_fig2_flexural_iso_hard.csv` | Ellefsen/Cheng/Schmitt 1988 fig 2 | equivalent isotropic, hard |
| `ellefsen_cheng_schmitt_1988_fig4_flexural_vti_soft.csv` | Ellefsen/Cheng/Schmitt 1988 fig 4 | elastic VTI flexural, soft |
| `ellefsen_cheng_schmitt_1988_fig4_flexural_iso_soft.csv` | Ellefsen/Cheng/Schmitt 1988 fig 4 | equivalent isotropic, soft |

The two flexural rows were `schmitt_1988_fig4_flexural_{slow,fast}.csv`
until the reference was actually opened. The source consulted is the
open-access MIT ERL precursor, Schmitt & Cheng (1987) report 1987.8
([DSpace](https://dspace.mit.edu/handle/1721.1/121148)), whose **fig 4
is a time-domain shot gather, not a dispersion curve**. Its flexural
dispersion lives in fig 2(a) (fast sandstone) and fig 8(a) (slow
sandstone). The 1988 JASA article is paywalled and was not consulted, so
its own figure numbering is unverified — hence filenames that name the
document actually traced.

**Figs 20 and 21 of that report are now shipped**, six curves, and they
are the first external evidence `flexural_dispersion_layered` and
`quadrupole_dispersion_layered` have ever had. One further figure in it
is still worth digitising and is not in the table above because no
notebook section covers it: **fig 7** (flexural and screw for granite,
limestone and fast sandstone — three fast formations in one figure).

The VTI rows were `schmitt_1989_fig5_flexural_vti_{qP,qSV}.csv` until the
reference was opened. In the open-access ERL precursor (Schmitt 1988.13)
**fig 5 is a monopole microseismogram**, not a dispersion curve; that
report's flexural dispersion is fig 22, and it is *poroelastic* while
`flexural_dispersion_vti` is elastic. There is also no qP/qSV branch pair
and no flexural splitting — a vertical borehole in a VTI medium with a
vertical symmetry axis is azimuthally isotropic. The `_qSV` row was an
orphan no notebook cell read.

The elastic reference is **Ellefsen, Cheng & Schmitt (1988)**, MIT ERL
([DSpace](https://dspace.mit.edu/handle/1721.1/75100)), figs 2 (hard) and
4 (soft). Each plots the TI formation against its *equivalent isotropic*
formation, so one figure ties both `flexural_dispersion_vti` and
`flexural_dispersion`.

That report states no numbers — constants deferred to Thomsen (1986), no
borehole radius, no fluid properties, its fig 1 a labelled schematic — so
the geometry had to be assembled from elsewhere and **checked before use**:

* **The rocks.** Thomsen (1986) table 1: Green River shale (Schock et al.
  1974 row) `V_P0` 3292, `V_S0` 1768 m/s, `rho` 2075, `eps` 0.195,
  `delta` -0.220, `gamma` 0.180; shale (5000) (Jones & Wang 1981 row)
  `V_P0` 3048, `V_S0` 1490 m/s, `rho` 2420, `eps` 0.255, `delta` -0.050,
  `gamma` 0.480.
* **How the rows were confirmed.** The curves were traced *first*, and
  their low-frequency limits — where the flexural branch tends to `V_S` —
  read **1775** and **1488** m/s against the table's 1768 and 1490:
  **+0.4 %** and **+0.13 %**. That also settles which of table 1's two
  Green River shale entries is meant, since the other (Podio et al.) has
  `V_S0` = 2432 m/s. Tsvankin's figure 1.12 gives the same row's
  `eps` = 0.195 and `delta` = -0.22 independently.
* **The radius was a prediction, not a fit.** `a` = 0.10 m is Schmitt's
  value in the companion ERL reports; it was used untuned, and all three
  scoreable overlays then landed under 0.5 % RMS. A wrong radius shifts the
  knee in frequency and would have shown up at once. Deriving it from the
  figures instead would have been the silent refit this directory exists to
  prevent.

**One of the four branches is still unscored**, and the blocker is the
solver rather than the reference: `flexural_dispersion_vti` raises
`NotImplementedError` for fast-formation TI, which is exactly Green River
shale at `V_Sv` 1768 > `V_f` 1500. That curve waits in
[`pending/`](pending/README.md) with its geometry already verified.

**The Tang & Cheng 2004 rows are withdrawn, not renamed.** Sections 3 and
4 cited that book for figures it does not contain:

* **"fig 7.1 — cased-hole Stoneley"** and the geometry quoted from
  **"sect. 7.2"** cannot exist: the book has **six chapters** (1 Overview,
  2 Elastic Wave Propagation in Boreholes, 3 Velocity and Attenuation
  Estimation, 4 Permeability Estimation, 5 Anisotropic Formations,
  6 Summary). So section 4's formation, casing, cement and radius are all
  unsourced — and its casing (5860/3140/7800) matches neither real casing
  row in the Schmitt ERL reports (6098/3354/7500, 6096/3352/7500).
* **"figs 3.7 / 3.10 — quadrupole slow + fast"**: both figures exist and
  neither is a dispersion curve. Fig 3.7 is waveform matching, fig 3.10 is
  acoustic time delay; chapter 3 is a *processing* chapter.

Both were confirmed against a physical copy.

**The nearest candidate for the quadrupole section has since been located,
and it is not usable either**: fig 2.11, *"Analysis of dipole and
quadrupole waves in the logging-while-drilling configuration"*, in
chapter 2. It is a **figure of principle** — schematic, not a quantitative
dispersion curve — so nothing in the `freq_hz, slowness_s_per_m` schema can
come out of it. Quadrupole dispersion needs a source other than Tang &
Cheng (2004). Beyond that, no replacement figure numbers are asserted,
because nobody has read the remaining chapters — inventing a
plausible-looking pointer is how this started.

~~**Quadrupole dispersion needs a source other than Tang & Cheng.**~~
*Closed.* It is tied by **Sinha & Asvadurov (2004) figs 10(a) and
19(a)**, both at 0.01 % RMS — a fast and a slow formation from one
paper, which is exactly the pair section 3 needs. What made it usable
where most of the quadrupole literature is not: `quadrupole_dispersion`
models **no drill collar**, and quadrupole is *the* LWD shear mode, so
nearly every candidate figure includes one. Sinha & Asvadurov compute an
open hole with no tool, and say so. Rejected on the collar for that
reason: Zheng, Huang & Toksöz (2004); Chi, Zhu & Rao (2005); Ji & Wang
(2024); and Sinha's own *Influence of a pipe tool on borehole modes*.

The repository already contained the correct chapter list, in
[`docs/ideas/Tang2004.md`](../../ideas/Tang2004.md): *"The book is
organized into six chapters."* That is the second time the right answer was
already in the tree while several other files carried a wrong one — the
Schmitt 1988 page range was the first, correct in `fwap/validation.py` and
wrong in thirteen other places.

**A sweep of the rest of the tree found a third instance of the same
pattern.** Ten citations in `fwap/stoneley.py`,
`tests/test_rockphysics.py` and `docs/possible_extensions.md` put the
Stoneley-permeability physics — simplified Biot-Rosenbaum, matrix
transmission loss, mudcake corrections — in "sect. 5.1" / "sect. 5.2",
i.e. in the *anisotropy* chapter. Permeability is chapter 4. Again
`docs/ideas/Tang2004.md` had it right (*"Chapter 4 treats Stoneley-wave
permeability logging… The physical basis is Biot–Rosenbaum theory"*).
They are corrected to **chapter level only**: the section numbering
*inside* chapter 4 could not be verified from any accessible source, and
asserting a `4.x` that merely looks right is the mistake this file
exists to record. Two more unverifiable figure pointers — "fig 5.3" for
the permeability round-trip test and "fig. 3.4" for fast-formation
dipole flexural — were removed rather than renumbered. The `sect. 5.3` /
`sect. 5.4` anisotropy citations elsewhere in the package are correct and
were left alone.

~~**Cased-hole Stoneley is now the one cased mode with no external tie.**~~
*Closed.* It is tied by **Tubman, Cheng & Toksoz (1984) fig 4b**, at 2.34 %
RMS. Four candidates were checked and rejected first — Schmitt 1988.13 figs
59/66 (TI poroelastic), Xie 2018 (right geometry, figure only 256x237 px
native) and Karpfinger 2010 (no casing or cement in it at all). Tubman's
figure is a 986x583 px panel in a 300 dpi scan, and its table 1 is indexed
by figure with rows tagged `4a` / `4b`.

**Paillet & Cheng 1991 fig 4.5 is retired, not merely unfulfilled.**
*Acoustic Waves in Boreholes* (CRC/Telford 1991, Routledge reissue) is
in-print and copyrighted; no accessible copy was found and every host
that might carry a preview is refused by this environment's egress
proxy. But the deciding reason is not access: **neither the figure
number nor the geometry attributed to it had ever been checked against
the book.** Both were inherited from the plan that seeded this notebook,
which is the same provenance that produced three invented geometries and
two shot-gather-for-dispersion mix-ups elsewhere in this file.

Section 1 now points at **Sinha & Asvadurov (2004) fig 2(a)**, which
plots the same two modes — Stoneley and trapped pseudo-Rayleigh — on a
fast formation, from a source whose parameters come from its own Table 1
and whose calibration is checkable against its own dashed reference
lines. Both curves land at 0.01 % RMS with full coverage. The only thing
given up is the rock: Sinha's fast formation instead of Paillet &
Cheng's limestone. The solvers and the physics are unchanged.

**Nothing in this notebook now depends on a source that cannot be
obtained.**

**`pseudo_rayleigh_dispersion` (the leaky n=0 root) still has no external
tie, and two on-topic papers were checked and rejected.** What it needs is
a phase-velocity dispersion curve for an **n=0 leaky mode in a fast
formation (`V_S` > `V_f`), open hole, no logging tool**, with velocity
between `V_S` and `V_P`. That combination is rarer in print than it looks:

* **Paillet & Cheng (1986)**, *A numerical investigation of head waves and
  leaky modes in fluid-filled boreholes*, Geophysics 51(7), 1438-1449 — the
  obvious candidate, and it does treat exactly this mode. But its
  fast-formation figures are complex-plane singularity trajectories (figs
  1-3), pressure-function amplitudes against wavenumber (figs 4, 5) and
  head-wave spectra (figs 6, 7) — no velocity-against-frequency plot. The
  only phase-velocity figures, 12 and 13, are **slow** formations
  (shales A/B/C, `V_S` 800-1100 m/s) **with a 5 cm logging tool**, which is
  neither the regime nor the geometry this solver models.
* **Zhang, Zhang & Wang (2009)**, *Leaky modes and their contributions to
  the compressional head wave in a borehole excited by a dipole source* — a
  **dipole (n=1)** study of **slow** formations (`V_P` 2000, `V_S` 800-900
  m/s) plotting complex poles rather than phase velocity. Wrong azimuthal
  order and wrong regime.

**Sinha & Asvadurov does not tie it either, and the near miss is worth
recording.** Fig 2(a)'s m=3 branch sits in the right slowness window,
`(1/V_P, 1/V_S)`, and scoring `pseudo_rayleigh_dispersion` against it gives
**11.3 %** RMS — which looks like a solver defect until the paper's own text
is read: *"it shows the presence of two cut-off modes (m = 3 and 4)"*. It is
an anharmonic cut-off mode, not a leaky one. That is the same
right-window-wrong-mode trap as the trapped-versus-leaky error above, caught
this time before anything was claimed. **The 11.3 % is not evidence about
the solver.**

One open question this search did not settle: the docstring says the mode's
low-frequency cutoff approaches `1/V_S`, while the solver trends toward
`1/V_P` at low frequency and returns roots down to a few hundred Hz. That is
recorded as a question, not a defect — adjudicating it is precisely what an
external tie would do.

## Workflow for adding an overlay

1. Digitise the figure (e.g. WebPlotDigitizer) into a CSV with the
   schema above.
2. Drop the CSV here under the documented name.
3. Re-run the notebook. That is the whole procedure — there is no code
   to edit.

## Units are checked, not assumed

The loader (`fwap.load_reference_curve`) refuses input whose magnitude
indicates a units mistake, and names the suspected error:

| Symptom in the CSV                | What it usually means            |
|-----------------------------------|----------------------------------|
| slowness around 60–700            | axis read in µs/ft or µs/m       |
| slowness around 1500–5000         | a *velocity* axis traced instead |
| frequency around 1–30             | axis left in kHz                 |

It also sorts click-order output by frequency, and rejects duplicate
frequencies, non-positive values and curves shorter than three points.

It never rescales the data to make it fit. That is deliberate: a
reference quietly adjusted to match would agree with a wrong solver
exactly as readily as with a right one, which would turn this whole
directory into decoration.

## Validation gate

`pytest --nbval-lax docs/notebooks/cylindrical_biot_validation.ipynb`
re-executes every cell and fails on any error — including the
`assert score.passed` inside `check_overlay`. So once a reference CSV
is present here, a solver regression that moves the curve by more than
the 5 % RMS budget fails the notebook.

With no CSVs present the notebook still runs end to end, and its
closing cell states plainly that nothing in it is validated against
literature yet. Green plots are not evidence.
