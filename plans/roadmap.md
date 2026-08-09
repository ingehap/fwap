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

**The headline for this revision is that real data arrived, found a defect, and
the defect is fixed.** For most of this project's life the binding constraint
was that every quantitative claim was measured against the same forward model
that generated its data. A real Schlumberger sonic log now sits in the registry,
and scoring the package against the vendor's own picks split cleanly in two:
shear to **0.12 %** median, and compressional on only 62 % of depths. That
second number was mode confusion rather than imprecision, and closing it (F.1)
took compressional to **95 %**. No synthetic could have found it, because the
synthetics are produced by the forward model the picker is scored against.

The package can now do all of that through its own API: `read_dlis_waveforms`
(F.3) reads the per-receiver waveforms and recovers the acquisition geometry
from the file's AXIS records. What remains is an asymmetry worth naming — **the
results exist but CI cannot defend them**, and the one row that would close that
is blocked on a decision rather than on work. (A third piece of F, confirming
the registered checksum against its canonical host, is tracked as F.4 in the
section itself.)

| Open item | Why it matters |
|-----------|----------------|
| **F.2 A waveform fixture CI can use** | The waveforms live in an 808 MB DLIS inside a 471 MB zip. A small extracted subset would work, but hosting one is redistribution and needs a decision rather than a commit. Until then what defends F.1 in CI is a seeded synthetic, not the log that found it. |
| **G.2 The `sonic_ml` consumer of the debonded dataset** | The generator is on `main`; what is open is the model and benchmark work on top of it. The measurements below changed what that work should be, and are worth reading before starting: the shipped cement-bond inverse keys on a signal a microannulus largely removes. |
| **A.1 Validation figures** | Ties the solver to published literature rather than to itself. Still needs the books. |
| **D. Conda-forge recipe** | Packaging only; unblocked once a PyPI release is live. |
| ~~**F. A real sonic log**~~ | *Largely closed.* A Schlumberger DSI log is registered and tested; the package's shear picks match the vendor's to **0.12 %** median on real rock. |
| ~~**F.3 A waveform path in `read_dlis`**~~ | *Closed.* `read_dlis_waveforms` reads a multi-dimensional channel and recovers sample interval and receiver offsets from the RP66 AXIS records, so the processing chain runs on a real log without `dlisio` at the call site. |
| ~~**F.1 The compressional-pick defect**~~ | *Closed.* It was mode confusion, not imprecision. `track_modes` and `pick_modes` now refuse to assign one arrival to two modes (`resolve_mode_collisions`); vendor agreement went 62 % → **95 %**, with shear bit-identical and nothing dropped. Kept below for the reasoning and the residual limits. |
| ~~**A.5 Fluid microannulus**~~ | *Forward model complete.* Elements, assembly and both public APIs are on `main`; kept below for the reasoning and the measured limits. |

A note on how this file is kept honest: items are marked closed only when the
code and its tests are on `main`, and status claims are checked against the tree
rather than against memory. The old roadmap carried a "leaky modes are still
open" note for some time after the leaky solvers had actually shipped; that is
the failure mode this heading exists to prevent.

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

What is still open here is narrow, and is items A.1, A.2 and A.5 below.

### A.1 Validation-figure coverage

Plan item I, marked partial. These are the only checks that tie the solver to
literature rather than to itself, and the item splits cleanly in two: the
analytic ties, which are done, and the digitised figures, which are not.

**Four analytic ties now exist that need no figure.**

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

**The data is not, and cannot be from here.** No reference CSV is shipped, so
the notebook currently validates nothing against literature — its closing cell
says so rather than letting green plots imply otherwise. The remaining work is
digitising three figures (Paillet & Cheng 1991 fig 4.5; Schmitt 1988 fig 4; Tang
& Cheng 2004 figs 3.7/3.10, 7.1; Schmitt 1989 fig 5), which needs the published
figures themselves — a task for a human with the books, not a coding session.
Once a CSV lands in `docs/notebooks/_data/` under the documented name, no code
changes: the section scores and gates automatically.

Note the figure numbering: this list previously cited "Tang & Cheng 2004
Fig. 3.4", which does not match the notebook's sections (figs 3.7 and 3.10 for
quadrupole, 7.1 for cased Stoneley). The notebook is the accurate list.

### A.2 Fast-formation flexural leakage

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

- The `sonic_ml` consumer: a debonded-regime dataset, and with it the first
  fair CBL-amplitude comparison. The forward model it needs now exists.
- Optional, and not required by the above: a delta-matrix / Abo-Zena
  reformulation of the elastic stack would remove the cancellation that makes
  the filter necessary at all, and would raise the frequency ceiling — the
  crack-wave window collapses above ~240 kHz on a typical stack purely because
  the propagators stop being representable.

`n=1` / `n=2` microannulus assemblies would be needed for *flexural* CBL work
and are a separate, larger job. The `n=0` path is self-contained and does not
depend on them.

### References for section A

- Schmitt, D. P. (1988). Shear-wave logging in elastic formations. *J. Acoust.
  Soc. Am.* 84(6), 2230-2244.
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

**What is still open:**

* **F.2 — a waveform fixture the CI can actually use.** The waveforms live in
  an 808 MB DLIS inside a 471 MB zip, which is not a viable fetch-on-demand
  test fixture. A small extracted subset would be, but hosting one is
  redistribution and needs a decision. Until then the results above are
  measured but not regression-tested, and what defends F.1 in CI is a seeded
  synthetic rather than the log that found it.
* **F.4 — confirming the registered checksum.** `gdr.openei.org` was
  unreachable from the session that added the entry, so the SHA-256 was
  computed from a mirror copy and is flagged as unconfirmed in the entry's
  `provenance`. It is the one unverified claim in the fixture registry.

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

What is *not* yet established is whether it beats the baseline at a usable
sample count. A 24-sample trial found the training loss reaching exactly zero
with best validation at **epoch 6 of 400**, and on its 3 held-out samples the
learned model scored *worse* than the closed form. That trial settled nothing
except that validation-based weight selection was necessary, which is now in
place. Both outcomes remain live, and the null one is a real result rather
than a failure: an analytic law leaving only ~18 % residual is a hard thing to
beat, and "the classical estimator is sufficient here" would be worth knowing.
The comparison needs a few hundred samples, which is an hour of generation.

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

1. **Real-data evaluation** — see section F. The binding constraint on every
   claim above.
2. **Free-pipe / debonded cased regime.** The cased dataset spans the *bonded*
   regime, where the cased Stoneley stays bound, so the bond inverse grades
   cement quality and is explicitly not a free-pipe detector. The debonded
   generator and its classical baseline have since shipped — see section G.2
   above for both, and for the measurements that reshaped them. What is left
   here is the learned model and its benchmark entry.

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
