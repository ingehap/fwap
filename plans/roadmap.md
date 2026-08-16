# fwap roadmap

Open items that would meaningfully extend fwap beyond the 0.4.0 release.

Method notes live beside this file: `plans/learning.md` for analytic oracles,
`plans/guides.md` for using a published paper as an oracle — the latter written
after taking Schmitt & Cheng through the cylindrical solvers figure by figure,
and carrying the inventory of what is still unused in that paper.

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
| ~~**A.2 Fast-formation leakage, `n=1` and `n=2`**~~ | *Fixed for three of the four fast-formation paths; the fourth is now silent instead of wrong.* The search window was phase velocity in `(V_R, V_S)`. **`V_R` is not a limit of these modes** — the branch descends from `V_S` toward *Scholte*, crossing `V_R` partway through the band — so the window lost the fundamental and returned whatever trapped mode it still contained. Corrected to `(V_f, V_S)`, with the fundamental selected by walking **up** in frequency and keeping the slowest root no faster than the last. Both halves were needed: widening alone swaps a 65 %-high answer for a 14-39 %-high one. Open-hole `n=1` now tracks the published curves at **0.78 % / 1.03 % / 0.87 % median** (fig 2a sandstone, fig 7a limestone and granite) — the digitisation floor — where it was on the right branch at 2 of 115 samples and granite had **no** correct sample at all. Figure 7a's merged band goes from **+124 % granite / +69 % limestone to +0.7 % / −1.7 %**, and the stiffness ordering is gone. Coverage is contiguous and monotone; the sawtooth is gone; group velocity is never negative. **Independent confirmation from a figure that played no part in the fix**: differentiating the corrected branch predicts figure 3's Airy arrival at **+8 %**, where the old bracket implied a wave 2.2× too early. Layered `n=1` follows (figure 12a's band **+30-55 % → −3.8 to −2.5 %**), and open-hole `n=2` is on the fundamental (granite 1.6 % median; the residual elsewhere is the separate near-cutoff onset delay, not the bracket). **The one path not fixed is layered `n=2`** — see A.7. Both ends of the band remain NaN by design: below the `V_R` crossing the mode is leaky and has no real-`k_z` root (a 20 000-point scan finds none), above the `V_f` crossing it has left the regime. Complex-`k_z` continuation is what those need. |
| ~~**A.6 The `n=2` layered path refuses every invaded zone**~~ | *Fixed, and it was an implementation/docstring mismatch rather than a scoping call.* `quadrupole_dispersion_layered` raised `ValueError` on any layer slower in shear than a slow formation — which is what an invaded zone **is** — while its own `Raises` section had always said the constraint applies **"(multi-layer only)"**. `flexural_dispersion_layered` enforced it for two-or-more layers only, as documented; `n=2` enforced it for every layer count. The single-layer guard is removed, so `n=2` now matches both its docstring and its sister path; the multi-layer guard is untouched. **Validated rather than assumed**: figure 15(b) plots the screw mode for exactly these models, and against the digitised curves the newly-unblocked path returns **0.58 % rms** for an 8 cm invaded zone — *better* than the same solver's 1.29 % on the virgin rock of the same figure. Figure 17 goes from 2 of its 12 waveforms computable to **6**. The fix does not touch the onset, which stays late by the near-cutoff margin already recorded (5.6 kHz against a published 3.4 kHz for the 8 cm model). |
| ~~**A.8 The SV column is not a solution — every `n >= 1` determinant**~~ | *Fixed everywhere, and it closed five separate defects the test suite had recorded as solver limitations.* Schmitt & Cheng's appendix (pp. 235-268, matrix on 235-236) prints all 36 elements of the layer matrix; transcribed and self-consistent under Hooke's law to 1e-11. Against it, fwap's P and SH columns were exact (r-independent to 1e-13) and the **SV columns drifted 24-86 %**, so they were not solutions. The cause was the ansatz: an azimuthal-only vector potential, which the cylindrical vector Laplacian does not admit for `n >= 1` (the coupling term carries a factor `n`; at `n = 0` it vanishes, and `n = 0` was always clean). The replacement is the Hansen form `u = curl curl(chi z)` — the appendix's own column — with `P1 = sigma s B_{n-1} - (n/r) B_n`, `P2 = s^2 B_n + n(n+1) B_n/r^2 - sigma s B_{n-1}/r`, `P3 = (n/r)[(1+n) B_n/r - sigma s B_{n-1}]`, giving `u_r = kz P1`, `u_z = -s^2 B_n`, `u_theta = -kz (n/r) B_n`, `sigma_rr = 2 kz mu P2`, `sigma_rz = -(2 kz^2 - kS^2) mu P1`, `sigma_r_theta = 2 kz mu P3`; verified against the appendix to 3e-16 at both orders and both Bessel families. **Landed on every `n >= 1` path**: open-hole `n=1`/`n=2` (real and complex), the layer E-matrices and their complex twins, the formation half-space columns of the complex cased determinants, the hand-coded 10×10 single-layer determinant, and the VTI qSV column; `flexural_dispersion_layered` now routes every layer count through the cased determinant instead of keeping a second implementation in step. **Ties**: figure 8a flexural **1.29 % → 0.063 %** and screw **0.94 % → 0.058 %** against a 0.033 % Stoneley control; figure 2a **0.78 % → 0.16 % median**; figure 7a flexural granite/limestone **1.24/1.39 % → 0.45/0.31 %**; figure 15(b) screw virgin/8 cm/16 cm **1.29/0.58/2.12 % → 0.055/0.136/0.197 %**; and on figure 15(a)'s 16 cm invaded curve fwap and an assembly built only from the appendix now agree to **0.011 %** where fwap was 1.5 % low at every point. **Closed as side-effects**: the 1.48/1.52 kHz near-cutoff gaps at `n=1`/`n=2` (now 0.00 and 0.12 kHz); figure 11(b)'s 'missing' screw arrival (found, +0.10 % against the published curve); figure 8a's Airy frequency (25 % low → within 1 %); the `n=1` fast-formation onset (2.5 → 0.99 kHz); and the `n=2` fast-formation grid dependence (now bit-identical wherever it converges). **One narrowing**: behind a steel casing the cased dipole mode is faster than a slow formation's shear speed and therefore leaky, so the bound-regime determinant has no root — the layered solvers return NaN for slow-formation cased stacks where they used to return a spurious branch that rose with frequency. `Im(det)` of the complex cased determinant crosses zero at 830 m/s (3 kHz) and 877 (6 kHz), just above `V_S = 800`, so a leaky search would recover it; that dispatch is the follow-up, tracked at A.9. Fast-formation cased and invaded-zone stacks are unaffected. **A.7 was not caused by this**, contrary to the hypothesis recorded here: the layer-equals-formation scan gives 430 sign changes at 12 kHz before and after, identical to the sample. Pinned by thirteen tests in `tests/test_schmitt_appendix.py`, now inverted to assert the agreement. |
| ~~**A.9 The slow-formation cased `n >= 1` dispatch has no leaky search**~~ | *Fixed.* Behind a steel casing the cased dipole and screw modes are faster than a slow formation's shear speed, so they radiate into it and the real-valued bound-regime determinant has no root. Both layered drivers now fall back to a complex-`k_z` search at exactly the frequencies the bound path could not answer, over `(V_S, min(V_f, min layer V_S))`, with the formation's leaky flags from `_detect_leaky_branches` and the root refined by `_track_complex_root`. `_march_leaky_cased_branch` is shared between `n=1` and `n=2`, seeds from the slowest real-axis `Im(det)` crossing and keeps A.2's monotone-descent rule; results carry a real `attenuation_per_meter`. Standard steel + cement on slow sandstone: dipole **989.5 -> 810.4 m/s over 3.5-12 kHz**, attenuation 2.51 -> 0.35 /m; screw **948.3 -> 817.3 m/s over 6-15 kHz**. `Im/Re(k_z)` falls from 11 % to 0.4 %. **Validated without a published curve**: the complex determinant reproduces the bound solver's root to 1e-9 with zero imaginary part wherever the mode is still bound (two formulations, one answer); the branch is continuous across the crossing as the annulus stiffens; the determinant vanishes at every returned root to 1e-12 - 1e-14; grids of 9 to 65 points agree at 8 kHz to 6e-13 m/s. A.7 does not block it -- that is the *fast*-formation `n=2` window; this one carries one to six crossings, a mode spectrum. ~~**Known gap**: around `V_S_layer / V_S` in [1.3, 1.5] at `ka = 2.5`~~ *-- closed, and the recorded description of it was wrong in both halves.* The scan does find a crossing there, and the mode is nowhere near the branch point: it sits 6-7 % above `V_S` at 855 / 851 / 859 m/s, while the crossing is at 1006 / 978 / 956. From that far away the tracker runs instead to the layer's own shear speed -- 1040.00 / 1120.00 / 1200.00 m/s to the digit -- which is the degeneracy `exclude` names and correctly rejects, leaving nothing. Seeded **off** the real axis it converges immediately: a single level at 5 % of `Re(k_z)` over 12 points already finds all three to 1e-4. `_march_leaky_cased_branch` gained that sweep, and the swept stiffness family is now unbroken from 1.28 to 1.62 with one turning point in each of phase velocity and attenuation. The argument principle did the job A.10 made it able to do, but as the **oracle** rather than the search: it counted the roots first and the marcher's answers were checked against that count. **The sweep is gated on the branch never starting, and the gate is the load-bearing part.** Its extra reach also finds a family of zeros just above `V_S` that are sharp to 1e-13 and carry winding number +1 -- so no root-quality test rejects them -- and are not modes: over an annulus sweep from `V_S_layer / V_S` 1.2 to 2.0 they sit at 807-810 m/s and ignore the casing, and over 3-15 kHz they sit at 1.004-1.017 `V_S` instead of dispersing, where the flexural branch falls 953 -> 888 -> 868 -> 834 m/s over 3 -> 8 kHz. Ungated, the sweep seeds off that family at a frequency where the mode has left the window and the monotone rule walks it down the whole band: **every** already-converged frequency moved, by 17 % at 3.5 kHz, ending 0.23 % above `V_S` instead of 1.3 %. Gated, production output is **bit-identical** and the fixture that had nothing now converges. A fixed dead band cannot separate the two families -- the flexural branch descends into the same neighbourhood at the top of the band -- so the floor is on *seeding* only; continuation, which arrives along a dispersion curve, stays free. ~~**Left open**: the 7 % step at the bound/leaky crossing.~~ *Settled, and it is three objects rather than one.* The step is real -- 798.91 m/s at ratio 1.26 against 857.39 at 1.28 -- and it is a **handover between two different modes**, not a break in one. (i) The **bound mode is absorbed at the shear branch point**: it climbs to 799.96 m/s at ratio 1.275 and past 1.2775 a 1500-point scan of the proper-sheet determinant finds **no root anywhere** in the bound window, so it ends rather than being lost. (ii) The **leaky branch the driver reports above the crossing already exists below it** -- at ratio 1.20 it is at 868.30 m/s while the bound mode is still alive at 786.67, so they are not sequels. (iii) The **branch-point pole is continuous through the whole crossing**, and this is what the open question was really about: tracked on the improper sheet with `leaky_s` forced rather than detected, it descends past `V_S`, runs **below** it over `1.285 <= V_S_layer / V_S <= 1.315` -- 798.86 m/s at 1.295 -- and climbs back, one turning point, `|det|` sharp to 1e-13 at every step. It is not annihilated; the production search cannot see that stretch because its window is floored at `V_S`. Searching only above `V_S` had made it appear to vanish at 1.28 and return at 1.32 with the attenuation jumping 0.98 -> 1.47; the gap was the floor. **One claim withdrawn on the way**: the branch-point pole was written up as ignoring the casing and 'not a mode'. Sampled coarsely far from the crossing it barely moves, which is where that came from; swept finely it runs 814.80 -> 798.86 -> 808.11 m/s over ratios 1.10 -> 1.295 -> 1.41 and is one continuous object. The seed floor stands on the claim that survived -- it is not the flexural branch, and seeding on it destroys the answer by 17 %. The crossing test's 'no jump across the boundary' comment was never what its assertions checked and is withdrawn. Pinned by three tests. Also fixed alongside: the bound layered bracket expansion could return 26.8 m/s phase velocities, now floored at the Scholte speed. Pinned by three tests. |
| **A.12 Cased VTI — assembly and leaky path both built and tied exactly; no driver** | *Where A.11 phase 5 pointed.* Phase 5 established there is **no open-hole leaky dipole mode** to compute, isotropic or VTI; the place the repo does record one is behind casing (A.9), where steel lifts the mode above a slow formation's shear speed. **The target is real, and that was measured first rather than assumed**: A.9's isotropic cased leaky dipole converges **13/13** over 3-15 kHz on slow formations matching the vertical velocities of Thomsen's slow media, at **1.37-1.69 `V_S`** with attenuation falling 2.5 -> 0.66 /m. **And anisotropy would move it by enough to matter.** Bracketing a VTI answer by running the isotropic cased solver at `V_Sv` and again at `V_Sh` gives **+1.6 % to +8.9 %** (Dog Creek `gamma = 0.345`, `V_Sh/V_Sv = 1.30`, peaking at 8.9 % near 5 kHz; Pierre `gamma = 0.165`, up to 5.0 %) -- an order of magnitude above the **0.21-0.27 %** at which the cased curves are tied to Schmitt & Cheng figures 20 and 21, so a cased VTI solver would resolve a real effect rather than a rounding difference. **The oracle is the best available anywhere in this work**: the isotropic limit of a cased VTI determinant must reproduce `flexural_dispersion_layered`, which is itself externally tied to two published figures -- unlike A.11's conjugate regime, which the isotropic limit could not reach at all. **What blocks it is a derivation, not plumbing.** The layered `n = 1` determinant is a 10x10 assembled from ten hand-written row builders, each carrying the isotropic formation inline; making it VTI means supplying the formation half-space columns at `r = b`. A.11 phase 4 built and validated VTI formation columns, but only the four quantities the open-hole problem needs -- `u_r`, `sigma_rr`, `sigma_rz`, `sigma_r_theta`. The layered stack needs continuity of **six**. ~~`u_theta` and `u_z` do not exist anywhere in the codebase and have to be derived.~~ **Derived and landed.** For fields going as `e^{i(n theta + k_z z)}` the coupled qP / qSV pair takes the potential form `u_r = d(phi)/dr`, `u_theta = (i n / r) phi`, `u_z = i k_z gamma phi` with `phi = K_n(alpha r)`, and the axial equation of motion leaves one condition fixing `gamma = -alpha^2 (C13 + C44) / (rho omega^2 + C44 alpha^2 - C33 k_z^2)`; SH is decoupled as `u = curl(psi z)`, so its `u_z` vanishes identically. `_vti_polarisation_ratio` and `_formation_displacements_n1_vti` implement it. **Checked three ways, none of them circular.** The isotropic limit pins `gamma` with no freedom: exactly **1** at the P root (recovering `u = grad phi`) and exactly **`alpha^2 / k_z^2`** at the S root (the Hansen form the isotropic assembly already uses), both to 1e-12. The `u_r` row reproduces the already-validated `_modal_row1_at_a_n1_vti` formation entries **exactly**, which also pinned the per-column normalisation the assembly uses -- `-1` for qP, `-k_z` for qSV, `+i` for SH. And the full field satisfies the VTI equations of motion in cylindrical coordinates to **~1e-9 relative** under fourth-order finite differences, with the strains, constitutive law and divergence written out independently of the module. **The 10x10 assembly is built.** `_modal_matrix_n1_layered_vti` and `_modal_determinant_n1_layered_vti` take the fluid and layer blocks from the isotropic stack unchanged and substitute VTI formation columns. **The substitution was justified by measurement, not by reading the derivation**: building the isotropic matrix for two very different formations shows the formation occupies **columns 6, 7, 10 and appears only in rows 5-10**, with every other entry *bit-identical* -- so the substep-F.2.a.5 phase rescale does not couple the layer block to formation parameters. The row-to-quantity map was calibrated the same way, in the isotropic limit: rows 5-10 are `u_r`, `u_theta`, `u_z`, `sigma_rr`, `sigma_rz`, `sigma_r_theta` with factors `1, i, i, 1, -1, -1`, matched to **~1e-15**. The factors are **per row, constant across the three columns**, which is the load-bearing part -- it says the two assemblies already share a per-column normalisation, so no column rescale is applied. **The oracle is exact and externally anchored**: at isotropic stiffnesses the VTI 10x10 *is* the isotropic 10x10, determinant ratio **`1 + 0j` to 6e-14** rather than merely proportional, over fast and slow formations alike -- and `_modal_determinant_n1_layered` is itself tied to Schmitt & Cheng figures 20 and 21 at 0.21-0.27 %, so this inherits an external tie no open-hole VTI path has. With the recombination in place the determinant differs by that Jacobian and the adjusted ratio is exactly **-1**; the roots coincide with the isotropic ones to four decimals on a fast formation. **What is still missing is a driver, and one boundary is wrong.** On a *slow* formation the isotropic determinant correctly returns nothing above `V_S` -- the mode is leaky and outside its bound regime -- while this one still reports a sign change (984.5 m/s at 9 kHz on `V_P = 2500`, `V_S = 900`). It evaluates there with `alpha^2 < 0` on a branch chosen by the bound rule, without the radiating flags being set, so that crossing is not a certified mode and the slow-formation window needs the leaky flags wired through before it means anything. ~~Bound fast-formation cased VTI is what this currently supports.~~ *The radiating flags are wired through, by a different route than the layered one.* The real-`k_z` layered path cannot express a complex `k_z` at all -- its rows are typed `float` and its Bessels are bound-branch -- so the leaky case goes through `_modal_determinant_n1_cased_complex` instead, which already carries A.9's leaky machinery, with the VTI formation half-space **injected** as a `(6, 3)` block. `_formation_state_vector_n1_vti` builds it and `_modal_determinant_n1_cased_vti_complex` calls through; everything else -- fluid, every layer, the propagator, the real-axis branch handling -- is the isotropic code untouched. **Only the formation takes radiating branches, and that is physical rather than a shortcut**: the fluid and the layers occupy bounded annuli and carry both Bessel families, so their condition is regularity at their inner radius; the half-space is the only part that can carry energy away. **The oracle holds at genuinely complex `k_z`**: over 12 samples with `Im(k_z)` up to 1.0 and `leaky_s` active -- the configuration the slow-formation cased dipole actually sits in -- the VTI determinant reproduces the isotropic one to `max |ratio - 1| = 2.9e-14`. **A mislabelling was found and fixed on the way, and it is the kind that survives a passing test**: the cased block's `sigma_rz` row is fed by `_modal_row4_at_a_n1_vti` and its `sigma_r_theta` row by `_modal_row3_at_a_n1_vti`, the opposite of what the open-hole row numbering suggests. The layered calibration had matched *values* into the correct slots, so the determinant was right while the names on it were backwards; settling it needed both stresses built directly from the displacements via the constitutive law, where the correct pairing gives a clean per-column ratio of `-i` and the swapped one varies by a factor of sixty. **What is still missing is only the driver** -- seeding, marching and continuation over the slow-formation window -- plus a tie to the A.12 bracket (+1.6 % to +8.9 %) once a curve exists. That is a build on the scale of A.11 phases 2-4 combined, and half of it is worth nothing: this codebase has produced a plausible-looking wrong dispersion curve at four separate points in that item alone. |
| **A.11 Leaky VTI — determinant done and validated; phase 5 finds no open-hole mode for it to drive** | *Scoped after #134 made the gap explicit. Phase 0 is measured and it re-ordered the rest; nothing else implemented.* **Phase 0's result changed the justification for this item.** Complexifying `_radial_wavenumbers_vti` is not only a leaky prerequisite — it repairs a live truncation of the **bound** VTI flexural branch. The solve returns `(nan, nan, nan)` wherever the Christoffel discriminant goes negative, on the stated grounds that complex roots are *"not physical in the bound regime"*; complex-conjugate `alpha^2` pairs are a real feature of TI media rather than an error state, and that early return costs **77 % of the bound window on Mesaverde shale(5) and 57 % on Mesaverde sandstone** (the other four table-1 media are clean). Visible in the public solver: `flexural_dispersion_vti` descends 2071 -> 1783 m/s over 3-6 kHz on Mesaverde shale(5) and returns `NaN` from 7 kHz up, stopping just above the discriminant cutoff at 1759 and far above every physical edge — `V_f` is 1500, and on media without the gap the branch runs well below that. A mode ending on its own approaches a limit; this one is cut off mid-descent. So **phase 2 is worth doing on its own merits, before any leaky work**, and it has an internal target rather than a published one. Two tests in `tests/test_anisotropy.py` pin the size of the gap; they deliberately do *not* assert the mode continues past it, because showing that needs the conjugate-pair handling itself. **Phases 1 and 2 are done.** Phase 1 corrected the `_radial_wavenumbers_vti` docstring, which said `alpha_qP` was the *smaller* root in two places while the code took `max` and was right. Phase 2 added `_radial_wavenumbers_vti_complex`, which solves the same quadratic in complex arithmetic and so keeps answering where the real one returns `NaN`. **It is additive on purpose**: changing the existing function's return type would turn today's `NaN` into a `TypeError` inside the row builders' `float(special.kv(...))` casts, which is a regression, so the existing solver is untouched and phase 3 flips the consumers. Validated against the *governing equation* rather than against the solver that could not produce these values: Christoffel residual **<= 1.9e-13** for both roots across the conjugate region, agreement with the real solver **1.8e-13** over 2898 samples where both are defined, `Re(alpha) >= 0` in all 3600 samples, continuous through the `disc = 0` branch point (roots merge at 18.03 then split as `18.0327 +/- 0.0550i`), isotropic limit reducing to `(p, s)` at **1e-14**, and complex `k_z` arithmetic sound at **2e-16**. The radiating branch is deliberately **not** offered: no caller exercises it yet, and an unexercised branch rule is what produced three rootless determinants on this branch; it belongs with the phase-4 driver. **Phase 3 recovered the window.** The row builders now consume the complex solve (7 call sites, 30 Bessel casts relaxed) and the qP / qSV columns are put on a real basis by `_recombine_conjugate_columns`. **The mechanical half alone is wrong and fails silently**, which is the part worth recording: feeding complex columns through the existing `det(M.real)` drops the imaginary halves of two independent solutions, and on Mesaverde shale(5) it moved the 3 kHz root from **2070.94 to 1500.33** and scattered the band between 750 and 1470 m/s -- a plausible-looking dispersion curve made of spurious sign changes. A plain conjugate split fixes that side but leaves the determinant vanishing **at the branch point**, where the two columns merge; the finder then locks onto the degeneracy and returns the cutoff velocity (1760.58) at every frequency whose true root lies above it -- which is what it did, at 3, 4, 5 and 6 kHz. The fix is the symmetric divided difference `(f(a_qP) +/- f(a_qSV))` over `1` and `(a_qP - a_qSV)`, real in both regimes and tending to `df/da` at the merge, i.e. the secular solution a repeated root calls for. It rests on a structural identity checked to **1e-12**: the two builders share one functional form and differ by a factor `k_z`, so `col_qSV = k_z conj(col_qP)` where the roots are conjugate. **Result: all six Thomsen table-1 media now converge 34/34 over 3-20 kHz and monotonically**, against 7/34 on the two truncated ones before; Mesaverde shale(5) runs 2070.94 -> 1474.71 m/s over 3-19 kHz with the four bound-regime values unchanged to the digit, and the four clean media are unchanged over their whole range. **The oracle had to be replaced, and that is the main lesson.** The isotropic limit *cannot* test this: there the discriminant is identically the perfect square `A^2 (p^2 - s^2)^2`, so the conjugate regime never arises and the instrument that anchored #131, #132 and phase 2 has nothing to say (pinned by a test over 2000 random isotropic media: `disc < 0` in **zero** of them). What replaces it is a **homotopy** -- scaling Thomsen's parameters from 0 to their Mesaverde values, anchored at `t = 0` by the independent `flexural_dispersion` to **0.00e+00** and required to stay smooth across `t ~ 0.55`, where the conjugate region swallows the mode and the family used to end. It does: 21/21 finite, steps falling monotonically 2.014 -> 1.313 with max second difference **0.053**, and no kink at the crossing. A different mode, or the right mode on the wrong branch, would have shown as one there. **Phase 4 is part-done: the leaky determinant exists and is validated, the driver is not written.** The formation columns now take radiating branches -- `_radial_wavenumbers_vti_complex` gained a per-wave `radiating=(qP, qSV, SH)` flag and the seven row builders evaluate through `_k_or_hankel` instead of `K_n` (30 sites). **The oracle the original scope promised does work here**, unlike in the conjugate regime: the leaky regime *is* reachable isotropically, so the VTI determinant can be checked against `_modal_determinant_n1_complex`, an independent implementation. It agrees **exactly** -- `det_vti * (alpha_qP - alpha_qSV) * k_z / det_iso = -1` at every sampled velocity, spread `1.0000000000`, the factor being the recombination's own non-vanishing Jacobian, which moves no root. **Three errors were caught by building it, two of them mine.** (i) A first cut forced qP and qSV onto a shared branch; that is wrong and would have blocked the ordinary leaky case, since over `V_Sv < c < V_P` the qSV wave radiates while qP stays evanescent. (ii) The phase-2 labelling was **swapping the two waves throughout the leaky window**: it ordered on `Re(alpha)` / `|alpha|`, but once a square goes negative the corresponding `alpha` is imaginary, so the ordering has to be on `alpha^2`. Phase 2's tests never entered that window; the isotropic limit went from **23.1 to 2.1e-14** on the fix. (iii) `_detect_leaky_branches` returns `(leaky_F, leaky_p, leaky_s)` -- **fluid first** -- and mapping it positionally put qP on the radiating branch and qSV on the bound one, which the oracle rejected before anything was believed. **Phase 0's deferred question is closed**: over the real leaky window `(V_Sv, V_P0)` with `Im(k_z)` up to 1, all six table-1 media keep the pair well separated, so continuity assignment suffices and no polarisation criterion is needed. ~~**What is left is the fluid column.**~~ *Done, and "write the outgoing fluid form" was the wrong framing.* The fluid occupies `0 <= r <= a`, so its condition is **regularity at the origin, not radiation at infinity**, and `I_n` -- entire for integer `n` -- supplies that at any complex argument. There is no outgoing fluid form to write: all the leaky case needed was for `F_f^2` to be allowed to go complex, which is one comparison. The branch of `F_f` is immaterial to the modes as well, and that was checked rather than assumed -- the fluid enters only rows 1 and 2, as `(F_f I_0 - I_1/a)` and `-I_1`, both **odd** in `F_f`, so flipping it negates the whole column and the determinant without moving a root; and over the search region `Im(k_z) >= 0` the argument stays in the closed upper half plane where the principal root is continuous. **A second labelling defect surfaced, in the same line phase 4 had already fixed once.** With a complex `k_z` neither the real nor the conjugate rule applied, so the code fell to a `|alpha|` fallback -- and above roughly `1.2 V_Sv` the *radiating* root has the larger magnitude, so qP and qSV swapped. The real-`k_z` oracle passed throughout that, which is exactly why it was not enough. The rule is now `Re(alpha^2)` everywhere, and it is **exact rather than heuristic**: in the isotropic limit `alpha_p^2 - alpha_s^2 = (omega/V_s)^2 - (omega/V_p)^2`, a positive real constant **independent of `k_z`**. **The oracle now passes over the regime a driver would search**: 30 samples over `c` in [2200, 3600] and `Im(k_z)` in [0, 1.5], `max|ratio + 1| = 1.4e-14`, against **3.4** before the fix. Branch selection also now routes through the shared `_radial_wavenumber` rather than a local copy, which is A.10's lesson applied. **Phase 5 was meant to validate the driver and instead found that the driver has no subject.** Counted by the argument principle, the open-hole window `V_Sv < c < V_P0` at `Im(k_z) > 0` holds **no `n = 1` root at all** -- and holds none for the **isotropic** determinant either, on fast and slow formations alike. So this is a fact about the open-hole dipole problem, not something the VTI assembly is missing, and a marcher pointed at that window would have nothing to march. **The seeding survey says otherwise and is wrong**, which is the trap worth recording: real-axis `Im(det)` sign changes show exactly one crossing per frequency on Green River across 3-18 kHz, at velocities that rise (1770 -> 1934 over 3-6 kHz), jump to 2769, descend to 2616, climb to 3098 and then split in two -- no coherent branch, and `|det|` vanishing at none of them. **The instrument had to be validated before its null result meant anything, and the first version was broken**: it unwrapped the phase and then summed differences around the closed cycle, which telescopes to exactly zero for any input whatsoever. It returned `0` for a box drawn around a root whose position was already known, and the VTI survey was briefly read as "no leaky mode exists" on that basis. Closing the loop before unwrapping gives `+1.0000` on the known root and `0.0000` on an empty box, and only then is the survey evidence. **One exception, not certified**: Dog Creek shale carries winding `+1` over 12-14 kHz only, a sharp zero at `c = 1839.12` m/s (`0.98 V_P0`) with `Im(k_z) = 0.889` and `|det|` down to 5e-15 relative. A narrow band hard against the P cutoff with no isotropic counterpart is the signature A.9 recorded for zeros that are **not** modes, and nothing here can tell the two apart. **Where leaky dipole modes are actually recorded is behind casing** (A.9), where the steel lifts the mode above a slow formation's shear speed -- that is the cased VTI path, which is not built. So phases 4-5 close as: determinant done and validated to 1.4e-14 against an independent implementation, **no driver, and no open-hole leaky VTI dispersion curve, because there is no open-hole leaky dipole mode to compute**. *Original scope follows.* `flexural_dispersion_vti` has no leaky path: the VTI stack is real-`k_z` **by construction, not by convention**, so this is a build rather than a flag flip. `_radial_wavenumbers_vti` returns `tuple[float, float, float]`, and the seven row builders carry **32 `float(special.kv(...))` casts across ~1000 lines** — patching the fluid helper alone just relocates the crash into those casts, which is what #134 established before declining to go further. **The hard part is not the Bessel branches, it is telling qP from qSV.** In the isotropic case P and S have independent square roots that never mix; in VTI both come from one Christoffel quadratic in `alpha^2`, and the code distinguishes them by `max`/`min` on the two real roots — an ordering that does not exist once `k_z` is complex and the roots are a conjugate-free complex pair. The assignment must then be made by identity (continuity from a known-good neighbour, or polarisation), and the two roots can **exchange** where the discriminant passes near zero. **Measured, not assumed**: over Thomsen (1986) table 1, scanning `c` in `[0.6 V_S, 1.6 V_P]` and `Im(k_z)` in `[0, 1.5]` at 3/6/10/16 kHz, Green River, Mesaverde shale(5) and Mesaverde sandstone all show `Re(disc)` sign changes with `min|disc| / median|disc|` down to **3.3e-4**, while Pierre shale, Taylor sandstone and Dog Creek stay clean at 0.55-0.84. Restricted to the narrower window a flexural leaky search actually visits (`c` in `[1200, 2600]` at 8 kHz on Green River) the discriminant stays at **0.128** of its median with no sign change at all — so the exchange region appears to sit at high phase velocity, plausibly outside the physical window, and **establishing that is phase 0 rather than an assumption to build on**. **Staging.** ~~(0) Map the branch-exchange region against the physical leaky window per medium.~~ *Done.* The bound flexural window sits at 1406-1766 (Green River), 777-845 (Pierre), 751-812 (Dog Creek) and 1435-1828 (Taylor) — all well below the high-`c` region where `Re(disc)` changes sign — so for the four clean media continuity assignment suffices with a guard on approach. On the two truncated media the question does not arise in the current window because the solve never gets there; it becomes live only once phase 2 opens that window, and should be re-measured then rather than assumed to transfer. (1) Fix the docstring bug found while scoping — `_radial_wavenumbers_vti` says "`alpha_qP` the smaller root" and "Sorted so `alpha_qP <= alpha_qSV`" in two places, while the code takes `max` and is **right** (isotropic limit: `V_P > V_S` gives `k_z^2 - (omega/V_P)^2 > k_z^2 - (omega/V_S)^2`, and it returns 19.6 against 12.1). Anyone implementing this from the docstring gets the labels backwards, and with complex roots there is no `max` to fall back on. Independent of the rest; do it first. (2) Complexify `_radial_wavenumbers_vti`: identity-preserving qP/qSV assignment plus a per-wave radiation branch, the same shape as `leaky_p` / `leaky_s`. (3) Complexify the four `n=1` row builders (32 casts; the three `n=0` ones only if leaky Stoneley is wanted too). (4) A driver, reusing `_track_complex_root` and the A.9 marcher rather than growing a third one. (5) Validation. **The validation story is unusually strong for this codebase, and is the reason to attempt it at all.** There is no published leaky-VTI curve — A.1 records that even *bound* VTI flexural is untied and needs Schmitt (1989) — but the isotropic limit supplies an independent answer where no figure can: a leaky VTI determinant at isotropic constants must reproduce `_modal_determinant_n1_complex(kz, ..., leaky_p, leaky_s)` at **complex** `k_z`, and the Sinha appendix matrix already in `tests/test_cylindrical_solver.py` is validated on leaky open-hole roots and applies at complex `k_z` directly. That is the #132 technique extended past the real axis, and it would make this the rare branch-rule change that is checkable before it is believed. **Risk.** Wrong-branch selection has produced a determinant with no root three times in this codebase (#129, #131, #132), each arriving looking like a discovery about the solver rather than a bug in the caller; this task is nothing but branch selection, for three wave types at once, with an ordering ambiguity the isotropic case does not have. Phases 0-2 are the research; phase 3 is mechanical but wide. |
| ~~**A.10 The leaky Bessel branch's docstring described a different function**~~ | *Fixed (documentation and tests only; the code was right).* `_k_or_hankel(leaky=True)` claimed its Hankel form reduces to `K_n` at a bound `alpha`. It does not: expanding it gives `(pi/2) i^{n+1} H_n^{(2)}(i z) = (-1)^{n+1} K_n(z e^{i pi})` -- the next sheet -- so at real positive `alpha` the two differ by factors of **2 to 3e3**. Found while building A.9, and worth recording because the false claim nearly caused a real defect: 'restoring' it by negating `alpha` on the leaky branch gives a function that matches `K_n` in the bound limit, is a pure outgoing travelling wave, passes every finiteness check -- and breaks the property callers actually rely on, that the two returned orders are consecutive orders of ONE solution evaluated at the SAME `alpha` the caller uses. That change makes the derivative identity `d/dx K_n = -K_{n+1} + (n/x) K_n` fail with a residual of order 1 and would corrupt every column built from the pair; it also destroys `pseudo_rayleigh_dispersion`, whose fluid energy balance the original branch satisfies to 1e-7 at every frequency. The real invariants -- order consistency at bound, radiating and complex `alpha`, and outgoing phase slope `+Im(alpha)` -- were untested and now are. **A second pass found the substantive defect the docstring had been hiding.** Knowing what the leaky branch *is* did not settle which root of `alpha^2` to feed it: `numpy.sqrt` selects `Re(alpha) >= 0`, the *decay* condition, while the radiation condition is `Im(alpha) > 0`, and the principal root carries `sign(Im(alpha)) = sign(2 Re(k_z) Im(k_z))` -- outgoing only while `Im(k_z) >= 0`, **incoming** below the real axis. **14 %** of the leaky Bessel evaluations in A.9's cased dipole run were on the incoming branch. Correcting the root is half of it: with `Re(alpha) < 0` the argument `i alpha r` crosses `hankel2`'s cut, so the leaky branch is now evaluated through `-K_n(z) + i pi (-1)^n I_n(z)`, whose cut `z` never reaches. The determinant is one analytic function across the seeding axis for the first time: the jump goes from **1.24** (formation S leaky, and `k_z`-dependent -- a different function with different roots) and **exactly -1** (fluid only -- an overall factor, which never moved a root) to **3e-11** in both. **Nothing published moves**: the new evaluation matches the old `hankel2` one to **1.5e-16** over every argument the solvers reached, A.9's velocities and attenuations are bit-identical, and incoming-branch evaluations go 457 -> 0 and 119 -> 0. Layer blocks keep both `I_n` and `K_n`, so a sign flip there is a change of basis that cancels in the propagator -- measured at 1.1e-12 against a 312 % change in `E` -- and they are left alone. **It answers A.9's open question**: the argument principle needs exactly the single-valued analytic function this supplies, and over A.9's recorded gap (`V_S_layer / V_S` in [1.3, 1.5] at `ka = 2.5`) the winding number is **one root**, at 855.1 / 850.5 / 859.3 m/s with positive attenuation and `|det|` sharp to 1e-13. Seeding the driver from that is A.9 driver work and is not done here. Pinned by seven tests. |
| ~~**A.7 The cased `n=2` determinant is noise-dominated**~~ | *Fixed, and both the diagnosis and the prescribed remedy on this line were wrong.* The determinant at real `k_z` is a real quantity times a phase that does not depend on `k_z`, and **the parity of that phase flips with azimuthal order**: imaginary at `n=1`, real at `n=2`. The fast-formation marcher tracked `Im(det)` at both, so at `n=2` it was seeding off round-off. Over 600 velocities at 12 kHz in the fast sandstone the open-hole `n=2` determinant has **one** sign change in `Re` and **212** in `Im`; the layer-equals-formation cased scan gives **1 in `Re` against the recorded 430 in `Im`**. Both drivers now measure which component carries the signal (`_real_root_function`) rather than assuming it. **Closes, as one defect**: figure 5a screw 8 % -> **0.16 % median** (12/12 points); figure 7b granite 2.60 -> **1.63 %** (14/72 -> 72/72) and limestone 12.80 -> **1.38 %** (1/30 -> 30/30); the screw cutoff of figures 6 and 14, 8.3 kHz against a published 6.29 (32 % high) -> **6.39 kHz, +1.6 %**; figure 6(b)'s ring band, empty -> fully covered; the cased layer-equals-formation case, NaN -> **reproduces the open hole to 1e-13**; and the grid irreproducibility, where two last-bit-identical grids gave different coverage -> identical. **It was not the propagator chain.** At `N=1` the chain reproduces `E(b)` from `P E(a)` to **1e-16** in a row-scaled norm, and the same 430 sign changes appear in the open-hole determinant, which has no propagator. So the delta-matrix / Abo-Zena reformulation named here is not needed for this, and returns to being the optional A.5 residue it was. `_FAST_FLEXURAL_MAX_CASED_ROOTS` is retained as a guard but is no longer load-bearing. Goldens regenerated; `n2_quadrupole_fast` is now reference quality. Pinned by nine tests, all of which previously recorded the defect. |
| **A.1 Validation figures** | *Re-scoped from five to three* (five → three → four → three: fig 4 was restored, then the figure itself was seen and turned out **not to be a dispersion figure at all** — it is a dipole shot gather, so it cannot be scored in the overlay schema. It did settle A.2's yes/no question, which is why it was worth fetching). Which figure carries the flexural dispersion curves is now known — **figure 2a of Schmitt & Cheng**, since digitised, which also gives A.1's flexural-Scholte tie its first *external* confirmation (1493 m/s read at 24.9 kHz against 1484 computed). **The "figures are the weaker instrument" premise is now measurably wrong.** It compared a 1e-3 analytic tie with a *5 % overlay budget* — but the budget was a choice, not a limit of the method. Digitised carefully, figure 8a ties `stoneley_dispersion` to a published slow-formation curve at **0.04 % rms, below what the figure can resolve**, which is the project's first external tie better than 1 %, and it pins the borehole radius as a by-product. The analytic ties are tighter numerically and are not external at all. **The pseudo-Rayleigh curve is now tied too** — figure 1a, both branches, ~1 %. **The cased hole is now tied as well, and the figures were in the paper all along.** `plans/guides.md` §11 had listed figures 20 and 21 — cased-hole dispersion for the dipole and the screw — as unread; they are the first external measurement of anything behind casing, where every previous cased number was scored against fwap itself. Read at 600 dpi with Table 1's own casing and cement rows (the cased fixtures elsewhere use invented values), the flexural mode ties at **0.21 % and 0.23 % median** for 1 cm and 3 cm cement, 45/45 points over the whole 4-15 kHz band, against a 0.28 % open-hole anchor — at the digitisation floor. The screw mode ties at **0.82 % and 0.27 %** over 8-20 kHz, 39/39 points: that is figure 21, which §11 called *"the only external measure of how wrong that path was"* for A.7, and before A.7 the configuration returned nothing at all. The geometry is the thing to get right and is quoted from p. 230 — the 10 cm radius is the **formation contact**, so the fluid column shrinks and the 3 cm-cement case has `a = 5.98 cm`. **The anchor earned its keep**: the first trace of figure 20's open-hole curve had jumped onto a steeper neighbour through the knee, which showed as -10 % against the open-hole solver, and an independent kink test put the jump at 6.32-6.35 kHz without reference to fwap. So what still needs the books is **VTI flexural** alone, and it needs a different paper — Schmitt (1989); this one is isotropic throughout. |
| **F.4 Two unconfirmed checksums** | *Still open on the network, no longer body-only.* `forge_dsi_las` and `iodp_u1347a_dsi` carry digests computed from copies that did not come down their canonical URLs. **The blocker is unchanged and was re-measured**: `gdr.openei.org` and `zenodo.org` are both refused at the network gateway (403 to CONNECT, policy denial) from this environment too, while ordinary HTTPS works — so it is the same obstruction that created the item, not a lapsed attempt. One successful fetch each still clears it. **What did get settled**: the digests match the copies in hand, re-checked over all 606 016 251 bytes of the IODP archive, all 21 435 504 of its member, and the 3 001 504-byte FORGE LAS — so none is a transcription error, and what is unverified is the provenance of the bytes rather than the arithmetic over them. **And the record is now checked rather than narrated**, which is the half of this item that was actionable: `checksum_confirmed` is a field, the `CHECKSUM CAVEAT` prose is kept in step with it by `check_registry_caveats`, `--list` prints `CHECKSUM UNCONFIRMED AGAINST URL` on the header line, a successful fetch of an unconfirmed entry prints the three edits that clear it, and the unconfirmed set is pinned in `tests/test_real_data.py` so confirming one cannot silently leave the other. The failure that guards against is the quiet one — flag cleared, paragraph left behind, registry now looking verified where it is not. Pinned by three tests. |
| **A.5 residue: delta-matrix reformulation** | Optional and blocked on nothing, which is why it kept falling off the list. *Still open — but measured, and the entry was wrong twice over.* **The ceiling is 84 kHz, not ~240.** The 240 came from arithmetic on a constant (`_BESSEL_ARG_MAX * V_f / (2 pi r)`), never from running the solver; the crack-wave API returns a root at 84 kHz and NaN at 86. **And `_BESSEL_ARG_MAX` is not what holds it down** — raised fourfold, the ceiling goes from 84 kHz to 84 kHz. What binds is the **product**: the determinant turns non-finite over the bottom ~16 % of the scan window while every input is still fine (parts finite, fluid Wronskian exact to 2e-16), and that floor climbs with frequency faster than the crack root does — at 84 kHz the root sits 0.3 % above it, two kilohertz later it is underneath. The magnitudes say why: `|E_form|` reaches **1.15e150** against a `sqrt(DBL_MAX)` headroom of 1.34e154, and widening the window makes `_layer_propagator_n0` overflow in `matmul` outright. The **cancellation is real and separate**: `cond(P_outer)` runs **1e35–1e40**, which is what the grid-stability filter is for. **So the prescribed remedy stands even though the diagnosis did not** — a Dunkin / Abo-Zena compound form factors out the growing exponentials and addresses overflow and conditioning together. **What is ruled out is the cheap version**: equilibrating to dodge the overflow alone would make things worse, because overflow is currently acting as a safety net — at 90 and 120 kHz, where the determinant is representable but ill-conditioned, the scans already return several grid-unstable roots near 1486–1499 m/s that the filter has to discard, and removing the overflow would widen that zone rather than the usable band. Pinned by two tests, one of which is the raise-the-bound experiment. |
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
> Laboratory, pp. 213-268* — two authors, different title, different venue,
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
not), **cased-hole Stoneley** (no reference identified -- this said "Tang & Cheng
2004 fig 7.1", and that book has six chapters) and **VTI flexural**
(now tied: Ellefsen/Cheng/Schmitt 1988 fig 4, 0.30 % RMS). Cased Stoneley may yield to White's tube-wave formula
generalised with the casing and cement compliances in series, which would be a
derivation rather than a lookup; nothing comparable suggests itself for the
other two. Once a CSV lands in `docs/notebooks/_data/` under the documented
name, no code changes: the section scores and gates automatically.

Note the figure numbering: this list previously cited "Tang & Cheng 2004
Fig. 3.4", disagreed with the notebook's "figs 3.7 and 3.10 for quadrupole,
7.1 for cased Stoneley", and resolved the conflict by declaring the notebook
accurate. **That was the wrong call, and the discrepancy was the signal.**
Checked against a physical copy: the book has **six chapters**, so fig 7.1
cannot exist; figs 3.7 and 3.10 do exist but are waveform matching and
acoustic time delay, not dispersion curves. Both notebook sections have since
had their references withdrawn. No Tang & Cheng figure number in this
repository should be trusted without opening the book.

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

### Figure 11 — the screw mode where fwap is silent, and one gap it is right to have

*"Quadrupole source, slow sandstone. Shot point obtained with a 1 kHz (a) and a
6 kHz (b) source center frequency."* Fourteen traces at r = 2.40-5.00 m, in the
rock of figure 8a.

**Panel (b) is the finding.** A 6 kHz source produces a ringing wavetrain whose
energy sits at **4.68 kHz** — above the 3.74 kHz screw cutoff figure 8a gives,
below the source. Envelope moveout is **1166 m/s with r² = 0.982**, and
`fwap.stc` puts the phase velocity at **1139.6 m/s** against figure 8a's traced
screw curve at 1179 — **−3.3 %**. `quadrupole_dispersion` returns **`NaN`**
there, because its first root for this rock is at 5.25 kHz. The mode
demonstrably propagates, coherently, at a velocity the published curve predicts,
in a band the solver reports as empty. Same argument figure 10 made for the
flexural gap, now for the screw mode.

**Panel (a) is the balancing case, and it matters.** At a 1 kHz source the packet
sits at 1.83 kHz and `quadrupole_dispersion` is silent there too — but so is the
paper: figure 8a draws no screw curve below 3.74 kHz. There is no trapped mode
at 1.83 kHz, so the arrival (moveout ~1580 m/s, r² 0.878, faster than `V_S`) is
a leaky or head-wave contribution a modal solver is not meant to produce, and
**the `NaN` is correct**. Not every gap is a defect, and this item should not
leave the impression that it is.

**The unification, and a correction to how figure 6 was framed here.** Figure 6
reported the fast screw cutoff as "32 % too high" and figure 8a reported a
"1.5 kHz near-cutoff gap". Figure 11 supplies the third point that shows they
are one phenomenon:

| case | published onset | fwap first root | gap | as % |
|---|---|---|---|---|
| flexural, slow | 1.04 kHz | 2.52 | **1.48 kHz** | +142 % |
| screw, slow | 3.74 | 5.25 | **1.51 kHz** | +40 % |
| screw, fast | 6.29 | 8.29 | **2.00 kHz** | +32 % |

The onset is late by **1.5-2.0 kHz in absolute terms** across two modes and two
formations. The percentages differ only because the cutoffs do — 32 % and 142 %
describe the same 1.5 kHz. Quoting it as a percentage, as the figure-6 entry
did, says more about the cutoff frequency than about the solver.

Three more tests.

### Figure 10 — the processing chain closed on published waveforms

*"Dipole source, slow sandstone. Shot point obtained with a 1 kHz (a) and a
3 kHz (b) source center frequency."* Fourteen traces at r = 2.40-5.00 m, in the
rock of figures 8a and 9.

Unlike figure 6 this gather **does** digitise: the packets are compact and the
moveout is strong, so once the crop starts past the scale-factor brackets a
straight-line fit to the envelope peaks has **r² = 0.995**. Two velocities come
out of it, and keeping them apart is the point — an envelope moveout is a
*group* velocity, a coherent alignment across the array is a *phase* velocity,
and on this branch they differ by 15-20 %.

| panel | dominant f | group (moveout) | phase (`stc`) | coherence |
|---|---|---|---|---|
| (a) 1 kHz | 0.86 kHz | 1009 m/s | **1205 m/s** | **0.960** |
| (b) 3 kHz | 2.77 kHz | 1037 m/s | **1156 m/s** | 0.717 |

**The chain closes.** In panel (a) the packet sits at 0.86 kHz, where the
flexural mode is at its low-frequency limit and its phase velocity is the
formation shear speed: `stc` returns 1205 against `V_S` = 1201, **+0.3 %**. In
panel (b), at 2.77 kHz, `stc` returns 1156 against figure 8a's traced phase
curve at 1172 (**−1.3 %**) and against `flexural_dispersion` at 1187
(**−2.6 %**). Published synthetic waveforms, through this package's processing,
land on this package's forward model — the first time the two halves have been
checked against each other on anything external.

The group numbers agree too: 1009 and 1037 m/s sit just above the 992 m/s
minimum that figure 8a's differentiated curve and figure 9's Airy phase both
give, which is right, since neither packet is at the 5.2 kHz where that minimum
sits.

**And panel (a) settles what the near-cutoff gap is.** `flexural_dispersion`
finds no root for this rock below about 2.5 kHz — the gap figure 8a measured at
1.5 kHz wide. At 0.86 kHz it returns `NaN`, yet the paper's own waveforms show a
coherent arrival there, picked at **0.960** and sitting at the shear speed. **The
gap is a solver limitation, not a physical absence**, and this is the waveform
evidence. Figure 4 made the same argument for fast formations from a shot
gather; figure 10 makes it for slow ones with a number attached.

Four more tests.

### Figure 9 — the slow-formation waveforms, and what differentiation costs

*"Dipole source, slow sandstone. Source center frequency effects. The offset is
equal to 4 m."* Figure 3's counterpart in the rock of figure 8a — and this time
in the regime where fwap **works**, so it is a prediction test rather than a
defect measurement.

Digitised from the 21 baselines (155.5 px apart) with the time axis fitted to
the seven label decimal points: 304.9 px per ms, residual ±0.011 ms.

Every trace from 2.0 kHz up carries a compact late packet at **4.068 ±
0.045 ms**, drifting only **−1.8 %** while the source centre frequency changes
fivefold — an Airy phase, and a tighter one than figure 3's −4.4 %. At the
figure's own 4 m offset that is a group velocity of **983 m/s** (960-1009).

| source | group minimum | at |
|---|---|---|
| **figure 9**, measured in the time domain | **983 m/s** | — |
| **figure 8a** phase curve, differentiated | **992 ± 4 m/s** | 5.1-5.5 kHz |
| **fwap** phase output, differentiated | **960.4 m/s** | **3.89 kHz** |

The two readings of the paper agree to **0.9 %** — a time-domain figure against
a frequency-domain one — which also validates the differentiation. fwap is
3.2 % low on the value.

**The finding is the frequency, not the value.** fwap puts the group minimum at
3.89 kHz where the figure puts it near 5.2 — **25 % low** — from a phase curve
that was only 1.3 % off. Differentiation amplifies a phase residual that is a
*distortion* rather than an offset, and figure 8a already showed the shape: zero
near 3.3 kHz, −1.8 % at 5-6 kHz, back to −0.8 % by 14 kHz. A tilt like that
moves the stationary point. So **anyone using fwap's slow flexural curve to
predict a waveform will place the Airy phase at the wrong frequency even though
the phase velocities look fine** — which is the practical cost of the
unexplained residual figure 8a found, now priced.

*Method, since both group curves come from differentiation.* The figure-8a
minimum is stable at 992-996 m/s for boxcar widths 41-121 (21 is undersmoothed
and finds a spurious minimum at 9.7 kHz), while the *frequency* of the minimum
moves over 5.07-5.47 kHz — so the value is good to about 0.5 % and the frequency
to about ±0.4 kHz. fwap's is stable to 0.1 m/s and 0.1 kHz across grid steps
0.02-0.2 kHz, and coverage is 100 % on every one of them: **the slow path shows
none of the grid instability figure 6 found at `n=2`.**

Three more tests.

### Figure 6 — a cutoff 32 % too high, and a reproducibility problem

*"Quadrupole source, fast sandstone. Shot point obtained with a 1.5 kHz (a) and
a 6 kHz (b) source center frequency."* Fourteen traces at r = 2.40-5.00 m, in
the rock of figure 5a.

**What this figure could not be used for, first.** The gather does not survive
digitisation well enough to measure a moveout: each trace is normalised to its
own peak, the wavetrains overlap their neighbours' bands, and the authors drew
two dashed guide lines through every trace. Reconstructing the 14 waveforms and
running `fwap.stc` over them gives coherence scattered between 0.4 and 0.88 with
no stable slowness peak, so **no velocity is quoted from it**. For contrast,
`stc` on the real IODP U1347A gather returns 0.948 median coherence — the
difference is the data, not the tool.

**What it does give is immune to all of that.** Zero crossings survive amplitude
clipping, so the *frequency* of the ringing wavetrain is solid: twelve of the
fourteen traces agree closely, median **7.19 kHz** (7.00-7.38), for a source
centred at **6.0 kHz**.

The ring sitting *above* the source frequency is the signature of a mode with a
cutoff — energy below cutoff cannot propagate in the mode, so the wavetrain is
pushed up to where the excitation switches on. Figure 5a puts the screw cutoff
at 6.29 kHz and figure 5c's excitation is zero below about 6.3 kHz, peaking near
9; a 6 kHz source folded against that lands at about 7.2. It also explains panel
(a): at 1.5 kHz, far below cutoff, there is no ring at all.

**The finding: the cutoff itself is wrong, not just the branch above it.**
`quadrupole_dispersion`'s first root for this rock is at **8.29 kHz** — 32 %
above the published 6.29 kHz — and it returns `NaN` at every single-frequency
call from 6.5 to 8.4 kHz.

> *Reframed by figure 11.* "32 % too high" is the wrong way to quote this. The
> same solver is 40 % high in the slow rock and 142 % high on the slow flexural
> mode — and all three are the **same 1.5-2.0 kHz absolute onset delay**. The
> percentage tracks the cutoff frequency, not the error. See the figure-11
> section. So the solver is empty at the frequency where the
paper's own synthetic waveforms show the screw mode ringing hardest. The `n=2`
defect is not only that the values above cutoff are overtones; the onset is
misplaced, and a 2 kHz band where the mode demonstrably exists and is strongly
excited returns nothing.

**And a reproducibility problem that qualifies every coverage number in this
file, including the ones measured here.**

```
np.arange(6.0, 20.01, 0.2) * 1e3      ->  47 of 71 frequencies converge
np.arange(6.0e3, 20.01e3, 200.0)      ->  42 of 71
```

Those two arrays are the same 71 frequencies to within **1.5e-11 Hz** — a
relative difference of 8e-16, last-bit floating-point rounding. They disagree
about whether four of five probe frequencies converge at all. The cause is the
continuation marcher: it walks high to low, so missing a root at one step
changes everything downstream.

The consequence is worth stating plainly: **coverage is a property of how the
caller happened to build the frequency array, not only of the rock and the
band.** Every coverage figure quoted in this item was measured on a stated grid
and is reproducible on that grid — but two callers writing the same sweep two
ways will get different `NaN`s. A test pins this, phrased so that it starts
failing if the marcher is ever made grid-stable.

**Figure 14 later bounded this.** The same two-ways-of-building-a-grid check on
figure 14's *fast* sandstone, virgin and both invaded models, gave identical
coverage every time. The instability is a property of this slow-formation model,
not of the `n=2` marcher everywhere, so coverage numbers elsewhere in this file
are not all suspect — only ones measured on a model shown to exhibit it.

Three more tests.

### Figure 1a — the pseudo-Rayleigh tie A.1 said did not exist

*"Monopole source. Dispersion (a) and attenuation (b) of the Stoneley wave (1)
and the first two pseudo-Rayleigh modes ((2) and (3)) in the presence of a fast
sandstone."* Three modes, three fwap entry points, on figure 2a's axes.

**A.1 lists the pseudo-Rayleigh curve among three items with "no external tie of
any kind."** Figure 1a supplies one — for both branches — and validates the
`branch` index along the way.

**A trap, caught by overlaying the traces back onto the scan.** In this panel
**the group curve is drawn above the phase curve** for the Stoneley, and the
labels say so. That is correct physics here — the Stoneley phase velocity rises
with frequency in a fast formation, so the group velocity exceeds it — but it is
the opposite of every other panel in this report. Comparing
`stoneley_dispersion` against the upper curve gives a spurious **−2.5 %**
systematic; against the right one it is **−0.8 %**. The overlay is what caught
it, which is the argument for doing one every time.

| curve | fwap entry point | coverage | rms |
|---|---|---|---|
| Stoneley phase | `stoneley_dispersion` | 36/36 | **0.90 %** |
| pseudo-Rayleigh 1 | `trapped_pseudo_rayleigh_dispersion(branch=0)` | 97 % | **1.01 %** |
| pseudo-Rayleigh 2 | `trapped_pseudo_rayleigh_dispersion(branch=1)` | 96 % | **0.80 %** |

At this figure's resolution 1 px is 1.41 m/s, so a plotted line is about
12.7 m/s — 0.87 % at the Stoneley, 0.5-0.7 % at the pseudo-Rayleigh modes. All
three sit at **one to one-and-a-half line widths**, so this is a pass at what
the figure can resolve. There is a consistent small negative bias — fwap reads
low on all three — that the figure cannot resolve into a real offset, and it is
not claimed as one.

**Anchors**: the Stoneley extrapolates to 1398.3 m/s against `tube_wave_speed`'s
1396.3 (**+0.14 %**), and both pseudo-Rayleigh modes cut on at the formation
shear speed (7.71 kHz and 12.89 kHz, against fwap's first roots at 7.96 and
13.39). Both descend toward the **fluid velocity**, not Scholte — the trapped
family's own asymptote — which is the qualitative behaviour the API name
promises and had never been checked.

**Separately, the phenomenological model is not the modal solver.**
`fwap.synthetic.pseudo_rayleigh_dispersion` places the guided arrival in
synthetic wavetrains from a closed form whose cutoff scale is `vs / (2 pi a)` =
**4140 Hz**, against a true cutoff of **7.71 kHz** — 1.9× too low. Measured
against this figure it is **37 % slow near cutoff**, easing to 6 % by 25 kHz.
Its docstring already says "phenomenological"; this pins how much that word is
carrying, which matters because `fwap.synthetic` uses it to place an arrival a
user may then pick.

Five more tests. **A.1's remaining ask drops from three figures to two.**

### Figure 5a — the screw mode's own figure, and a bound on the method

Figure 7b measured `n=2` across three rocks, but its curves merge and it only
resolves the fast sandstone below about 10 kHz. **Figure 5a** (p. 242) is the
screw mode's own panel — *"Quadrupole source. Dispersion (a) … of the screw mode
(1) and the first trapped mode (2) in the presence of a fast sandstone"* — on
figure 2a's axes, 0-25 kHz, with only two modes on it. It is the direct `n=2`
counterpart of figure 2a.

Traced in two overlapping passes: a wide window through the plunge, then a
narrow one with a small slope cap for the flat tail, because mode 2's group
curve crosses mode 1's phase near 18 kHz and a single pass follows the steeper
branch down. Monotone to +0.002 normalised over the whole span, inside the line
width.

| | read | computed | |
|---|---|---|---|
| cutoff value | 1.7385 | `V_S/V_f` 1.7340 | **+0.26 %** |
| at 24.87 kHz | 1522.6 | Scholte 1484.4 | **+2.57 %** |
| crosses `V_R` | **7.58 kHz** | figure 7b gave 7.69 | 1.4 % apart |
| crosses `V_f` | never in the plotted band | | |

**The screw mode approaches Scholte more slowly than the flexural one** — still
+2.6 % at 25 kHz where the flexural mode was +0.6 %, and it never drops below
the fluid velocity at all, where the flexural mode crossed it at 17.9 kHz.

**And the cross-figure agreement bounds the digitisation method itself.** The
same rock's screw mode is drawn twice: figure 5a on a 0-25 kHz axis with two
curves, figure 7b on a 4-20 kHz axis with six. Nine frequencies from 7 to 12 kHz
agree to **+0.4 % to +1.8 %**, with figure 7b systematically about 1 % high —
the expected direction for a reading off the more crowded panel. That is looser
than the ±0.4 % figures 2a and 7a managed for the flexural mode, and it is the
honest error bar to quote for numbers traced off the three-rock panels. It is
also a *consistency* check on the merge-bias explanation already recorded for
figure 7.

**fwap over 6.4-25 kHz**: **72 % coverage**, every value inside `(V_R, V_S)` and
sweeping it end to end (2413-2598), **not one point within 5 %**, errors +15 %
to +67 % with median **+53 %**, and upward jumps of +102 m/s. The screw mode is
never returned for this rock. Same shape as figure 12's finding — high coverage,
no correct answers.

Three more tests.

### Figure 3 — the same defect as a traveltime, and a cross-domain check

Figure 3 (p. 240) is not a dispersion plot: *"Dipole source, fast sandstone.
Source center frequency effects. The offset is equal to 5 m. The source center
frequency varies from .5 kHz to 10.5 kHz by steps of .5 kHz."* Twenty-one
synthetic waveforms in the rock of figure 2a — so it checks the same mode in
the time domain.

Digitised by locating the 21 baselines (155.5 px apart, uniform) and timing each
trace's largest late excursion. The time axis is fitted to the seven label
decimal points: **303.4 px per ms, residual ±0.010 ms**, with any constant
offset between a decimal point and its tick bounded at about 10 px = 0.03 ms.

**Every trace from 3.0 kHz up carries a large late packet at 4.35 ± 0.07 ms**,
and its arrival drifts by only **−4.4 % while the source centre frequency
changes by 250 %**. That is the signature of an **Airy phase**: energy piling up
at the stationary point of the group-velocity curve, where the arrival is a
property of the formation rather than of the source. It is what licenses reading
one group velocity off the whole series.

| | |
|---|---|
| measured apparent group velocity | **1150 m/s** (range 1124-1181) |
| figure 2a's group-curve minimum | **1109.7 m/s** at 5.24 kHz |
| agreement | **+3.7 %** |

**Two figures of the same paper, one in frequency and one in time, agree on the
group-velocity minimum to under 4 %.** The measurement is slightly fast, as
expected — the largest half cycle of an attenuating Airy packet precedes the
envelope centre. This is also the first use made of figure 2a's *group* branch,
which was traced alongside the phase branch and had sat unused.

**And the defect restated as a traveltime, which is how a user would meet it.**
Over 3.0-10.5 kHz `flexural_dispersion` answers at 3 of 16 frequencies, at
2414-2597 m/s. Propagated over the figure's own 5 m offset that is
**1.92-2.07 ms**, against **4.35 ms** of published waveform. The returned mode
would arrive before half the true traveltime had elapsed — **2.2× too early**.

*Not used.* The printed scaling factors down the left edge would give the
excitation curve and let figure 2c be checked too. At this scan quality the
glyphs are not reliably legible — "0.0014" and "0.0019" cannot be told apart —
so they were left alone rather than transcribed into a number the repository
would then trust.

Three more tests.

### Figure 13 — how little a dipole sees invasion at 1 kHz

*"Dipole source. Invaded zone effects in the presence of a fast sandstone.
Iso-offset (z = 5 m) … the only virgin formation (1), and a 8 cm (2) and 16 cm
(3) invaded zone. The source center frequency is successively equal to 1 kHz
(a), 3 kHz (b), 6 kHz (c), and 7.5 kHz (d)."*

Panel (a) extracts cleanly, and the answer is worth having. Cross-correlated
against the virgin trace, the 8 cm model lags by **+0.1 µs** and the 16 cm model
by **+1.2 µs**, at correlations of **0.992** and **0.981**. Against a ~2 ms
traveltime at 5 m that is **under 0.1 %**.

**A 16 cm invaded zone is undetectable at 1 kHz.** That is the time-domain form
of what figure 12 shows in the frequency domain — all three models share a
plotted plateau at `V_S` below about 2 kHz, which figure 12's own reading put at
1.7357 for the whole group. Figure 13(a) says how far apart they actually are
there: a microsecond.

**Corrected while working figure 14.** This section used to say that only panel
(a) was measurable and that nothing in panels (b)-(d) cleared r = 0.8. That was
an artefact of my own extraction rather than a property of the figure: the
half-window was narrower than the widest trace's own excursion, so the *virgin*
trace in panel (b) was clipped to 68 % coverage and every correlation there was
computed against a truncated reference. Widened, **panel (b) measures** — at
3 kHz the 8 cm model lags by **+54.6 µs** and the 16 cm model by **+99.0 µs**,
at correlations of **0.930** and **0.848**, invariant to ±0.01 µs across 36
combinations of crop start, crop end and half-window.

So the growth figure 12 predicts above 2 kHz **is** measured here after all, and
it is steep: the 16 cm delay goes from 1.2 µs at 1 kHz to 99.0 µs at 3 kHz, a
factor of **79** for a 3× change in source frequency.

Panels (c) and (d) are still refused, but now for a positive reason rather than
a threshold. Their traces overlap so the components merge — coverage sticks at
0.76-0.78 whatever the window — and panel (d)'s best-fit lags are +264 and
+319 µs regardless of window choice, the constant-lag signature of a
cross-correlation hopping cycles rather than measuring a delay.

Three more tests.

### Figure 14 — the quadrupole invaded zone, where the effect is amplitude

*"Quadrupole source. Invaded zone effects in the presence of a fast sandstone.
Iso-offset (z = 5 m) … the only virgin formation (1), and a 8 cm (2) and 16 cm
(3) invaded zone. The source center frequency is successively equal to 1.5 kHz
(a), 3 kHz (b), 6 kHz (c), and 7.5 kHz (d)."*

I predicted this figure would hit the ringing-wavetrain wall that stopped
figures 6 and 13(b)-(d). **That was half wrong, and the half that was wrong is
the useful part.** Panel (a) is not a wavetrain at all — it is a compact
three-to-four-cycle wavelet, and it extracts cleanly across the full band. My
first pass reported it as unmeasurable because the extraction half-window was
narrower than the 16 cm trace's own excursion, which clipped it to 60 % coverage;
the overlay check caught that, as it has caught every other error in this series.

With the window widened, **panel (a) gives the quadrupole's invasion delay**:
the 8 cm model lags the virgin waveform by **+9.7 µs** at r = 0.924, the 16 cm
model by **+36.5 µs** at r = 0.797. Both lags are *invariant* — across 36
combinations of crop start (±110 px), crop end (±180 px) and half-window
(±15 px), the spread in both is **zero**. The 16 cm correlation sits at the 0.8
bar not because the measurement is unstable but because the waveform genuinely
changes shape: invasion adds cycles, which is the "more energy in the low
frequency shear event" the report describes on p. 228.

It is tempting to set that 36.5 µs against figure 13(a)'s 1.2 µs and call the
quadrupole thirty times more delay-sensitive. **That comparison does not hold,
and checking it is what turned up the figure-13 error above.** The two panels
are at different source frequencies — 1.5 against 1 kHz — and figure 13's own
panels show how steeply that matters: the dipole's 16 cm delay runs 1.2 µs at
1 kHz to 99.0 µs at 3 kHz. The quadrupole's 1.5 kHz value lands *between* those,
about where interpolating the dipole would put it. No source frequency is shared
between the two figures where both panels are measurable, so **these figures do
not support a like-for-like delay comparison** — only the peak-amplitude one,
which is the comparison the report itself makes. A test pins that distinction.

Panels (b)-(d) remain beyond the method, and there is now a positive reason to
say so rather than a correlation threshold. Their best-fit 8 cm lags are +237.7,
+238.9 and +235.1 µs at 3, 6 and 7.5 kHz — constant to ±2 µs across a 2.5×
change in source frequency, with *negative* zero-lag correlations. A physical
invasion delay does not do that; a cycle-hopping cross-correlation does. **No
delays are quoted from panels (b)-(d).**

Also legible is the printed peak-amplitude scale factor on all twelve traces,
and that is where the rest of this figure's content lives. The report says so on
p. 228 — *"the variations of the peak amplitude as a function of the invaded
zone thickness are more pronounced with low source center frequencies than
previously (Figure 14a, b relative to 1.5 kHz and 3 kHz)"* — and it names the
mechanism: *"due to a higher frequency location of the useful starting energy
of the screw mode"*.

Digits transcribed, then checked independently by measuring the ink: the
plotted excursions reproduce the printed numbers to within **0.027** in the
worst panel and under 0.01 typically, the residual being the finite line width,
which inflates a small trace relative to a large one. Both readings agree that
panel (c) is genuinely non-monotone in thickness — the 8 cm trace sits *below*
the virgin one there.

The dipole/quadrupole contrast, both measured the same way:

| source `f_c` | dipole (fig 13) | quadrupole (fig 14) |
|---|---|---|
| lowest plotted | 1.25× (1 kHz) | **2.90×** (1.5 kHz) |
| 3 kHz | 1.03× | 1.68× |
| 6 kHz | 1.00× | 1.30× |
| 7.5 kHz | 1.00× | 1.65× |

The dipole is flat to 3 % at every frequency at or above 3 kHz; the quadrupole
never drops below 1.29×. The published claim holds.

**fwap cannot be checked against the substance of this figure, and the reason
is worth stating exactly rather than filing as another A.2 miss.** Peak
amplitude at a fixed offset is excitation times propagation, and `BoreholeMode`
carries neither for this model: there is no excitation field on it at all, and
`attenuation_per_meter` comes back `None` from both the plain and the layered
quadrupole path. The figure's main effect is outside the API surface — **a
correct dispersion solver would not reproduce it either.** That is a scope
limit, not a defect, and it is the first time in this series that the
distinction has mattered.

The dispersion the figure implies, fwap mostly does not return. Of the twelve
(model, source frequency) pairs plotted, the quadrupole solver produces a phase
velocity for **three**. The virgin fast sandstone — the reference every panel is
normalised against — gives **no root at any of 1.5, 3, 6 or 7.5 kHz**, its onset
sitting at **8.4 kHz**, above the entire figure. A.2 is again the whole story:
all **194** converged samples across the three runs lie strictly inside
`(V_R, V_S)`, not one outside.

And the single dispersion claim in the figure-14 paragraph — *"the increase of
the group velocity of the Airy phase"* at 6 and 7.5 kHz — does not come out
inaccurate here, it comes out **with the wrong sign**. The overtone sawtooth
ramps at roughly 0.5 (m/s)/Hz, steep enough that `v_g = 1/(d(f·s)/df)` goes
**negative on 18 of 48 adjacent virgin pairs**. No guided mode has a negative
group velocity, so this is not a small error in the Airy velocity — there is no
usable group-velocity curve to be in error.

Coverage inverts with invasion thickness, exactly as figure 12 found at `n=1`:
virgin **49** of 141 samples (first root 8.40 kHz), 8 cm **63** (4.10 kHz),
16 cm **82** (3.40 kHz). The three-medium problem converges further, and lower,
than the one-medium problem contained inside it.

One caveat retired: unlike figure 6's slow-formation model, these counts were
**stable** across bit-identical grids built two ways and across repeat calls.
The grid sensitivity recorded at figure 6 is model-specific, not a universal
property of the `n=2` marcher. The tests still assert bands rather than exact
counts.

Twelve more tests.

### Figure 16 — the slow formation, where invasion finally shows up

*"Dipole source. Invaded zone effects in the presence of a slow sandstone.
Iso-offset (z = 5 m) … the only virgin formation (1), and a 8 cm (2) and 16 cm
(3) invaded zone. The source center frequency is successively equal to 1 kHz
(a), 3 kHz (b), 6 kHz (c), and 7.5 kHz (d). **Each series is normalized with
respect to its own maximum denoted by 1.00.**"*

**That last sentence settles a convention this series had been inferring.**
Figures 13 and 14 print the same scale factors without saying what they mean;
figure 16's caption says it outright. The printed number is the trace's peak
amplitude relative to the largest trace in its panel. The figure-14 reading was
right, and it now has the authors' own words behind it rather than an inference
from ink.

This is figure 13's experiment with the rock swapped, and it is where the
invaded zone stops being invisible.

**Twelve arrows, and they calibrate the figure.** Every trace carries a drawn
arrowhead at a shear arrival — the virgin formation's on trace 1, the invaded
zone's own on traces 2 and 3. Detected as filled blobs and read through the time
axis at 5 m, the four virgin arrows give **1198.0 m/s** against table 1's
`V_S` = 1201 (**−0.25 %**) and the eight invaded ones **1083.0 m/s** against 1081
(**+0.18 %**). Twelve independent detections landing on two distinct published
velocities, with no overlap between the families. This is the calibration check
for everything else taken off the figure, it owes nothing to fwap, and it
confirms table 1's slow invaded-zone row a second time — figure 15 anchored it
at 1081.2 from a different figure entirely.

**The amplitudes**, digits read and then confirmed by measuring the ink to
within **0.018**, which settles several glyphs that could be read two ways
(0.754 not 0.734; 0.644 not 0.699; 0.452 confirmed to 0.002):

| `f_c` | virgin | 8 cm | 16 cm | spread |
|---|---|---|---|---|
| 1 kHz | 0.612 | 0.754 | 1.000 | 1.63× |
| 3 kHz | 1.000 | 0.644 | 0.672 | 1.55× |
| 6 kHz | 1.000 | 0.452 | 0.672 | **2.21×** |
| 7.5 kHz | 0.881 | 0.706 | 1.000 | 1.42× |

Figure 13's fast sandstone, measured the same way, gives 1.25 / 1.03 / 1.00 /
1.00. **Where the fast formation goes flat at and above 3 kHz, the slow one
never drops below 1.42×.**

**And the mechanism is measurable, not just its size.** Splitting each trace at
its own arrow separates the P wavetrain from the shear packet:

| P/S | virgin | 8 cm | 16 cm |
|---|---|---|---|
| 1 kHz | 0.03 | 0.07 | 0.05 |
| 3 kHz | 0.03 | 0.15 | 0.22 |
| 6 kHz | 0.10 | 0.96 | **1.53** |
| 7.5 kHz | 0.21 | 1.95 | **2.76** |

Monotone in thickness at every frequency at or above 3 kHz, and monotone in
frequency at every thickness. At 6 kHz with 16 cm, and at 7.5 kHz with either
thickness, **the P wavetrain becomes the largest event in the trace** — the
series maximum jumps from ~5.0 ms to ~2.35 ms. That is the report's conclusion C
— *"small velocity contrasts can modify the internal dynamics of the waveforms
more easily in slow formations, through an increase of the P wavetrain"* — as a
number, and it explains the non-monotone scale factors in panels (b) and (c):
the virgin trace wins those panels because its shear packet is the biggest thing
in them, not because invasion made anything smaller in absolute terms.

**A like-for-like delay comparison, which figure 14 could not offer.** Figures
13(a) and 16(a) share source type, source frequency, offset and both
thicknesses; only the rock differs. The 16 cm delay is **+1.2 µs** in the fast
sandstone and **+117.3 µs** in the slow one, at correlations of 0.981 and 0.879.
As a fraction of traveltime, 0.06 % against 2.82 % — **45 times larger**.

**The fwap check, on the one path this series has shown to be good.** Figure 15
tied these exact three models' phase velocity at 1.47-1.48 % rms, so a forward
prediction is fair: take the group-velocity minimum, divide 5 m by it, compare
with the measured arrival of the shear packet.

First, the packet is the Airy phase. The virgin trace peaks at 5.01-5.11 ms
across a 7.5× change in source frequency — frequency-independent, the signature
figures 3 and 9 relied on — giving **989.6 m/s** against figure 8a's published
group minimum of **992.0 m/s**. Two independent figures, two domains, **0.24 %
apart**.

| model | measured | fwap | error |
|---|---|---|---|
| virgin | 5.05 ms (n=4) | 5.21 ms | **+3.0 %** |
| 8 cm | 5.56 ms (n=3) | 5.91 ms | **+6.3 %** |
| 16 cm | 5.64 ms (n=2) | 6.09 ms | **+8.0 %** |

The virgin +3.0 % is figure 9's "3 % low in value" reached from another figure
and another domain. **The new result is what a layer does to it.** The invaded
arrivals drift with source frequency (5.37-5.68 and 5.45-5.84), so measured
against the latest — most charitable — end of each range the errors are +2.0 %,
+4.0 % and +4.3 %. Either way the layered error is about **twice** the open-hole
one. The 8 cm versus 16 cm difference sits inside the measurement spread and is
**not** claimed.

So figure 15's verdict needs one qualification, added at its site below.
*"The layered solver is as accurate as the open-hole one"* holds for **phase**
velocity. It does not survive differentiation: the group velocity a waveform
actually arrives at is twice as wrong on the layered path. Both remain far
better than anything the fast formation produces, so this refines the figure-15
conclusion rather than overturning it.

Two smaller things. **At 1 kHz fwap returns nothing for any of the three
models** (onsets 2.52, 3.51, 2.94 kHz) — the panel that measures best is
entirely outside coverage, the near-cutoff gap of figure 10 now confirmed on the
layered path. And unlike figure 14's fast-formation quadrupole, **these curves
are structurally sound**: one contiguous run per model, phase monotone
throughout, group velocity never changing sign, nothing pinned against a bracket
ceiling. Here the defect is accuracy; there it was the absence of a curve. The
group minimum is grid-converged — 241, 591 and 1181-point grids agree to
0.02 m/s — so the coarse grid the tests use is not a shortcut.

Ten more tests.

### Figure 17 — the slow-formation quadrupole, which fwap will not compute

*"Quadrupole source. Invaded zone effects in the presence of a slow sandstone.
Iso-offset (z = 5 m) … the only virgin formation (1), and a 8 cm (2) and 16 cm
(3) invaded zone. The source center frequency is successively equal to 1 kHz
(a), 3 kHz (b), 6 kHz (c), and 7.5 kHz (d)."*

**The headline was a refusal, not a number — and it has since been fixed.**
`quadrupole_dispersion_layered` used to raise `ValueError` on this model before
computing anything: the slow-formation branch required every layer to be at
least as fast in shear as the formation, and an invaded zone is by definition
slower. **Eight of the figure's twelve waveforms were not inaccurate — they were
unrepresentable.**

That was filed as A.6 and is now closed. The short version: the check ran for
every layer count, while the function's own `Raises` section had always said
**"(multi-layer only)"** and `flexural_dispersion_layered` had always enforced
it that way. So it was an implementation/docstring mismatch, not a scoping
call. The single-layer guard is gone; the multi-layer guard is untouched.
Digitising **figure 15(b)** — the panel the figure-15 work skipped, which plots
exactly these curves — showed the newly-unblocked path returns **0.58 % rms**
for the 8 cm model, better than the same solver's virgin control. Figure 17 goes
from **2 of 12** computable waveforms to **6**. Details in the figure-15(b)
section below.

**What fwap still cannot do here**, after the fix. The virgin screw mode
resolves from **5.25 kHz** up — 196 of 297 samples, no interior gaps, group
velocity never negative, so structurally sound like figure 16 and unlike figure
14 — and the invaded models from 5.0-5.6 kHz. Panels (a) and (b) sit at 1 and
3 kHz, below every one of those onsets. **Six of the twelve plotted waveforms
still have no computable phase velocity, and all six are below cutoff** — the
near-cutoff gap of figures 10 and 11, which A.6 never touched.

The one forward prediction left: the virgin screw packet peaks at 4.91-4.99 ms
across all four source frequencies — frequency-independent, so it is the Airy
phase — giving **1008.1 m/s**, distinct from the flexural mode's 989.6 m/s on
the same rock as it should be. fwap's group minimum of 954.2 m/s puts the
arrival at 5.24 ms, **+5.6 %**, against the flexural's +3.0 % in figure 16 —
same direction, larger.

**The published data, which stands whatever fwap does.**

Twelve arrows again, and this is the tightest external agreement in the series:
the four virgin arrows give **1193.6 m/s** against `V_S` = 1201 (**−0.61 %**)
and the eight invaded ones **1081.3** against 1081 (**+0.03 %**). Finding them
needed a better discriminator than figure 16's — this figure's dense
high-frequency wavetrains produce blobs that pass a shape test, so the arrow is
now identified as the arrow-shaped component **not connected to the trace**.
Re-running figure 16 with the stricter method reproduces its twelve values
exactly, so nothing there needed correcting.

| `f_c` | virgin | 8 cm | 16 cm | spread |
|---|---|---|---|---|
| 1 kHz | 0.156 | 0.496 | 1.000 | **6.41×** |
| 3 kHz | 0.918 | 0.838 | 1.000 | 1.19× |
| 6 kHz | 1.000 | 0.455 | 0.546 | 2.20× |
| 7.5 kHz | 0.757 | 0.312 | 1.000 | 3.21× |

Digits and ink agree to 0.028. **One real ambiguity, and it is recorded rather
than hidden**: the 1 kHz virgin glyph could read 0.156 or 0.186, and the ink
(0.184 ± 0.02 on a 39-pixel excursion) cannot separate them. Comparing the glyph
against known 5s and 8s elsewhere in the same figure settles it — the 8s in this
font are two closed bowls (0.918, 0.838) and this is the open-topped 5 of 0.455.

**Panel (a)'s 6.41× is the largest spread in the four waveform figures, and the
virgin trace is the smallest one.** A slow-formation quadrupole at 1 kHz is
barely excited; invasion brings the screw mode's useful starting energy down
into the source band and the response grows more than six-fold. Same mechanism
figure 14 named in a fast formation, with a much larger effect.

**The report's claim for these panels, checked as written.** Page 229 says the
P-wavetrain growth with invaded-zone thickness is *"especially true with the
quadrupole source (Figure 17c, d)"*. Splitting each trace at its own arrow, P/S
grows from virgin to 16 cm by **26×** at 6 kHz and **69×** at 7.5 kHz, against
the dipole's 15× and 13× in figure 16. Read as absolute level instead of growth
the claim would look false — the dipole's P/S is *larger* at 6 kHz — so the
wording matters, and growth is what the authors wrote.

**No delays are quoted from this figure.** Panels (b)-(d) give 8 cm lags of
+868.9, +867.8 and +862.4 µs at 3, 6 and 7.5 kHz — constant to ±3 µs across a
2.5× change in source frequency, the cycle-hopping signature figure 14
established. Panel (a) clears r = 0.8 on both traces, but its 8 cm lag
(+357 µs) contradicts the peak-time shift of the same trace (+100 µs), so it is
not quoted either.

Eight more tests.

### Figure 15(b) — the panel that was skipped, and the fix it validated

Figure 15's caption covers two panels; the figure-15 work digitised only (a),
the flexural. Panel **(b)** is the screw mode for the same four models — and it
plots precisely the curves `quadrupole_dispersion_layered` was refusing to
compute. A.6 was reachable from this figure and was missed for want of one
panel. Worth naming as a pattern: **a figure is not "done" because one of its
panels was.**

**Calibration, with a trap in it.** Panel (b)'s x-axis runs **2-10 kHz, not
0-10** — nine evenly spaced ticks with the "5" label under the fourth and "10"
under the ninth. Reading it as 0-10 would have shifted every frequency by
2 kHz. Physics confirms the reading independently: curves 1, 2 and 3 leave the
axis at v = 0.80 (= 1201/1500, the **virgin** shear speed, exactly as p. 228
says) and curve 4 at 0.72 (= 1081/1500, the invaded one). Tick fits: x to
0.008 kHz, y to 0.00006 normalised.

**Curve identification, checked rather than assumed.** The four solid curves
merge, so each is followed from its own start with slope prediction, stopping
where a neighbour comes within two line widths. That check earned its keep: on
rms alone the 16 cm data fits an *invaded-only* prediction (1.37 %) slightly
better than a layered one (2.12 %), which would have looked like evidence the
trace had hopped branches. Run-ordering settles it — the 16 cm trace sits on
**run 3 of 4** at every sampled frequency, so it is curve 3. The apparent
preference is just that curves 3 and 4 converge to within **0.6-0.8 %** over
the overlap band, below fwap's own error. Curve 2 is **4.2 %** from its nearest
neighbour there, which is why the 8 cm tie carries the weight and the 16 cm one
only corroborates.

**The result.** With the single-layer guard removed:

| model | band | rms | median |
|---|---|---|---|
| virgin *(control, open hole)* | 5.25-9.75 kHz | 1.29 % | +0.35 % |
| **8 cm invaded** *(layered)* | 5.75-8.00 kHz | **0.58 %** | +0.29 % |
| 16 cm invaded *(layered)* | 5.00-5.50 kHz | 2.12 % | +2.07 % *(3 points)* |

The virgin row is a control: it uses the open-hole solver that figure 8a
already tied at 0.94 % rms on this rock, so it prices the digitisation itself.
**The 8 cm layered result is better than that control.** A path that was
refusing to run computes its own figure more accurately than the path that was
allowed to. That is what turned "the guard is over-strict" from an argument
into a measurement, and it is the whole justification for the fix.

What the fix does **not** do: the onset stays where it was. fwap resolves the
8 cm model from 5.6 kHz against a published 3.4 kHz — the slow screw mode's
near-cutoff gap, already recorded, and untouched by any of this.

Six more tests.

### Figure 12 — the invaded zone, and coverage turns out to be an inverted signal

Figure 12 (p. 249) is the first published check of the **layered** solvers:
*"Invaded zone effects with a fast sandstone. Dispersion and attenuation of the
flexural (a) and screw (b) modes in the presence of: (1) a 16 cm thick invaded
zone; (2) a 8 cm thick invaded zone; (3) the only virgin formation; (4) the only
invaded zone."* Table 1's two fast rows: virgin 4878 / 2601 / 2160, invaded
4390 / 2341 / 2360 — slower and denser than the rock it replaces.

**What this panel can be read for, and what it cannot.** Eight curves are drawn
in a 1.2-wide normalised window. The column-line count runs **1 line at 2.0 kHz,
4 at 2.5, 8 only across 3.5-5.0 kHz, then 3 from 6 kHz up** — so the plunge
region where the four models separate is exactly where they also cross. No
per-model curve was traced there and none is tabulated. Stated rather than
papered over: this is the one figure of the four whose middle is beyond the
method.

**What is readable is worth having anyway.** The two plateaus confirm table 1's
invaded-zone row, which every comparison here depends on and which this
repository transcribed from a scan:

| | read | `V_S/V_f` | |
|---|---|---|---|
| upper plateau, 1.80-2.05 kHz | 1.7357 ± 0.0016 | 2601/1500 = 1.7340 | **+0.10 %** |
| lower plateau, 2.20-2.60 kHz | 1.5630 ± 0.0015 | 2341/1500 = 1.5607 | **+0.15 %** |

**The layered path inherits A.2 whole.** All eight fwap runs — two modes × four
models — return values strictly inside their own `(V_R, V_S)` window and
sawtooth, with upward jumps of **+121 to +185 m/s** where a guided mode's phase
velocity can only fall. Against figure 12a's merged phase band the layered
flexural solver reads **+31 % at 6 kHz rising to +53 % by 9.8 kHz**.

**And the new thing, which inverts a habit this file has had throughout.**

| model | flexural | screw |
|---|---|---|
| 1: 16 cm invaded | **73 %** | 74 % |
| 2: 8 cm invaded | 38 % | **77 %** |
| 3: virgin only | 9 % | 50 % |
| 4: invaded only | 10 % | 35 % |

Adding an altered zone raises coverage **four- to eightfold while the answers
stay wrong**. Sparseness has been read as A.2's symptom since the item was
filed — "converges over roughly 38 % of a 1-12 kHz band" is how the original
entry states the problem. On the layered path that reading is backwards: the
configuration returning the most answers is the one furthest from having any. A
caller who checks coverage to decide whether to trust an altered-zone dispersion
curve is reading the metric upside down.

Three more tests, including one that pins the coverage inversion.

#### Figure 15 — the same layered code in a slow formation, and it works

Figure 12 leaves an ambiguity it cannot resolve on its own: is the layered
overshoot the fast-formation bracket, or is the layered propagator itself
wrong? Figure 15 (p. 255) is the slow counterpart — same four models, same
solver calls, table 1's slow sandstone **2751 / 1201 / 2100** and its invaded
zone **2338 / 1081 / 2000** — and it answers it.

This panel reads cleanly where figure 12a did not: its **group curves are
dashed**, so they fragment under connected-component labelling and leave the
four solid phase curves behind. It also has the best calibration of the six
figures — 16 x-ticks residual to 0.019 kHz, 4 y-ticks to 0.00024 (0.36 m/s) —
and the 0.550-0.850 window puts the plotted line at about ±4 m/s.

**Two anchors, both to 0.02 %**: the virgin curves leave the axis at 1200.7
against `V_S` = 1201, the invaded-only curve at 1081.2 against 1081. That also
confirms table 1's *slow* invaded-zone row, the counterpart of what figure 12
did for the fast one.

| model | fwap coverage | rms | median |
|---|---|---|---|
| 1 virgin only *(open hole)* | 91 % | 1.43 % | −1.34 % |
| 2 8 cm invaded *(**layered**)* | 84 % | **1.47 %** | −1.22 % |
| 3 16 cm invaded *(**layered**)* | 92 % | **1.48 %** | −1.49 % |
| 4 invaded only *(open hole)* | 67 % | 1.01 % | −0.07 % |

**The layered solver is as accurate as the open-hole one.** So figure 12's
31-53 % overshoot is the fast-formation bracket — A.2 — and not the layered
machinery. The layered code is exonerated, and the same bracket fix repairs both
paths. That is worth having explicitly: it rules out a whole class of work
(rewriting the propagator) that figure 12 on its own would have left open.

*Qualified by figure 16.* That sentence is about **phase** velocity, and it does
not survive differentiation. Figure 16 is these same three models in the time
domain, and predicting its Airy arrival from the group-velocity minimum is
**+3.0 %** late for the virgin rock but **+6.3 %** and **+8.0 %** for the two
layered ones — about twice the error, on the most charitable reading of the
measurement spread as well as on the means. The layered path is still far better
than anything the fast formation produces, and the exoneration above stands; but
"as accurate as the open-hole one" should be read as a statement about the
plotted curve, not about the wave that arrives.

**And it narrows figure 8a's unexplained offset.** The ~1.3 % slow-flexural
systematic is here in all three `n=1` configurations at the same size and the
same shape — best near 3 kHz, worst about −2 % at 5-6 kHz, recovering by
14 kHz — open hole and layered alike, while the Stoneley on the same rock was
0.04 %. So it is **`n=1`-specific and geometry-independent**: not the layered
code, not the bracket, not the radius, not the reading. The candidate list is
materially shorter than it was.

**Two limits, stated.** The invaded-only curve could not be followed past about
4 kHz — a dashed group segment crosses it — so it is used only for its anchor.
And the near-cutoff gap is **not** the single width figure 8a suggested:
1.44 kHz (virgin), 2.44 (8 cm), 1.19 (16 cm), 0.92 (invaded only). See the
scope correction under figure 8a.

Five more tests, one of which states the fast/slow separation as an assertion so
that a "fix" to the propagator alone would break it.

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

> *Scope corrected by figure 15.* "Set by the hole rather than by the mode"
> holds for the two modes in **one homogeneous rock**, which is all that was
> measured here, and it does not generalise. Put an invaded zone against the
> same formation and the gap moves: 1.44 kHz virgin, **2.44** with an 8 cm
> zone, 1.19 with a 16 cm zone, 0.92 for the invaded rock alone. The sentence
> above was a fair reading of two numbers; four numbers make it a coincidence
> of the homogeneous case.

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

> *Corrected by figure 12: sparseness is the wrong thing to be watching.* The
> 38 % is real, but it is not the defect and it is not even monotone with it.
> Against the published invaded-zone curves, a 16 cm altered layer takes the
> flexural solver from **9 % coverage to 73 %** on the same rock — and every one
> of the extra answers is an overtone, 31-53 % above the figure. On the layered
> path high coverage is the *worse* sign, not the better one. Keeping the cased
> dataset single-mode is still right; the reason to state is "the second mode is
> wrong where it is returned", not "it is rarely returned".

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
> > 213-268.
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
  Formations: An Overview.* MIT Earth Resources Laboratory, pp. 213-268.
  *(Page range corrected from 213-246 while inventorying what else the
  paper offers: p. 246 is figure 9, with sixteen more figures after it.
  The paper carries **25** figures and runs to p. 268 — the commit that
  fixed this citation got the author, title and venue right and the end
  page wrong.)*
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
