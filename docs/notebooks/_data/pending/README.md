# Traced but not yet scoreable

CSVs here are finished digitisations that **cannot be scored yet**. They
are deliberately not in `_data/` proper: a CSV there is picked up by
`check_overlay` on sight and asserted against the 5 % budget, so parking a
curve the notebook cannot evaluate would fail the notebook rather than
record a gap.

Nothing in the notebook reads this directory.

**In every case here the solver is what is missing, not the digitising.**
Entries 2 and 3 are finished references at the fidelity of the rest of the
repository. Entry 4 is the exception worth naming: its reference is
complete, but it is a raster scan whose x-axis calibration carries about
1.4 % of scan distortion, so it is a lower-grade curve than anything in
`_data/` — see its own section before promoting it.

## Contents

| File | Source | Blocked on |
|------|--------|------------|
| `tubman_..._fig4b_pseudo_rayleigh1_cased.csv` | Tubman/Cheng/Toksoz 1984 fig 4(b) | no cased pseudo-Rayleigh API |
| `tubman_..._fig4b_pseudo_rayleigh2_cased.csv` | Tubman/Cheng/Toksoz 1984 fig 4(b) | no cased pseudo-Rayleigh API |
| `sinha_..._fig11a_leaky_compressional_slow.csv` | Sinha & Asvadurov 2004 fig 11(a) | no slow-formation leaky-compressional solver |
| `paillet_cheng_1986_fig12a_..._fundamental.csv` | Paillet & Cheng 1986 fig 12(a) | same, **and** no logging-tool geometry |
| `paillet_cheng_1986_fig12a_..._first.csv` | Paillet & Cheng 1986 fig 12(a) | same, **and** no logging-tool geometry |

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

## 2. Cased-hole pseudo-Rayleigh, two branches

From Tubman, K. M., Cheng, C. H., & Toksoz, M. N. (1984), *Geophysics*
**49**(7), 1051-1059, fig 4(b). Same figure and same verified geometry as
the cased Stoneley curve that scores 2.34 % in `_data/` — fluid radius
1.85 in, 0.4 in steel, 1.75 in cement, formation contact at 4.0 in.

**The package has no cased pseudo-Rayleigh entry point.**
`stoneley_dispersion_layered`, `flexural_dispersion_layered` and
`quadrupole_dispersion_layered` cover n = 0, 1 and 2 for cased geometries,
but `trapped_pseudo_rayleigh_dispersion` takes no `layers` argument, so
there is nothing to call. Promoting these needs a new public function, not
a new argument to an existing one — so it is API work, and belongs behind
an issue rather than being bolted on.

## 3. Slow-formation leaky compressional mode

From Sinha & Asvadurov (2004) fig 11(a), curve 3 — 107 points, 2.2-15.0
kHz, extracted from vector paths rather than traced, on the same Table 1
geometry as the six scored Sinha curves (slow formation B: 1890 / 508 /
2054, water 1500 / 1000, `a` = 0.1016 m).

The paper identifies it directly: *"This mode is also referred to as a
leaky compressional mode as it exists between the formation compressional
and borehole-liquid slownesses."* The traced curve runs 529.0 to 639.1
us/m against a C line at 529.1 and an L line at 666.7 — inside that window
throughout, which is the check that it is the right curve.

**No fwap function computes this mode.** `pseudo_rayleigh_dispersion`
tracks the n=0 leaky root of a **fast** formation, between `1/V_P` and
`1/V_S`, and requires `V_S > V_f`; formation B is slow. The two are
different modes in different regimes, and scoring one against the other
would repeat the trapped-versus-leaky category error recorded in
`../README.md`.

It is parked here because the reference is finished and verified, so if a
slow-formation leaky-compressional solver is ever written, the tie is
already in the tree.

## 4. Paillet & Cheng 1986 fig 12(a) — leaky compressional, two branches

From Paillet, F. L., & Cheng, C. H. (1986), *A numerical investigation of
head waves and leaky modes in fluid-filled boreholes*, **Geophysics**
51(7), 1438-1449. Fig 12(a), *"group and phase velocities for fundamental
and first mode"*, shale B.

**Phase-velocity branches only.** The fundamental is 170 points over
1.5-24.9 kHz (497.4-630.1 us/m); the first mode is 84 points over
13.3-24.8 kHz (498.7-548.7 us/m). Both lie between the figure's own C and
L reference lines (1/V_P = 500.0, 1/V_f = 666.7 us/m), which is the window
a leaky compressional mode occupies.

**Geometry** — Table 1, shale B row: `V_P` 2000, `V_S` 1000 m/s, `rho`
2300 kg/m^3; borehole fluid `V_f` 1500 m/s, `rho_f` 1380 kg/m^3;
`R_hole` 12.5 cm; **`R_tool` 5 cm**.

### Why it is doubly unscoreable

* **Slow formation.** `V_S` 1000 < `V_f` 1500, so
  `pseudo_rayleigh_dispersion` — which requires `V_S > V_f` — will not run
  on it, and it is a different mode from that function's fast-formation
  leaky root in any case.
* **Logging tool.** Table 1 gives shale B a 5 cm centralised tool. fwap
  models no inner tool anywhere: `BoreholeLayer` stacks *outward* from the
  fluid, and `FluidAnnulus` is a debonding gap between casing and cement,
  not an inner cylinder. Even a slow-formation leaky-compressional solver
  would not reproduce this figure without tool support.

### Fidelity: lower than everything else parked here, and why

This is a **raster scan of a 1986 journal page**, not vector artwork.

* **The y axis is sound.** Calibrated from the frame (rows 139 = 2.5, 1041
  = 1.0 km/s) and checked against the figure's own dotted reference lines,
  which come out at **2.0011** and **1.5039** km/s against Table 1's `V_P`
  2.0 and `V_f` 1.5 — 0.25 % or better.
* **The x axis is the weak link.** Its six tick marks are not evenly
  spaced (256-302 px between neighbours). After a least-squares fit the
  residuals still reach **0.36 kHz**, about 1.4 % of full scale. That is
  scan distortion, and it is not removable by re-tracing.
* Each traced curve reproduces six independently-read columns to **0.8
  m/s**, so the tracing itself is not the limiting error — the axis
  calibration is.
* The fundamental's fastest point reads 497.4 us/m, about 0.5 % *faster*
  than `1/V_P` = 500.0. A leaky compressional mode should not exceed
  `V_P`; this is calibration and line-width error at the cut-off, not
  physics. **The CSV is shipped as traced rather than clipped to the
  physical bound.**

### How the phase/group pairing was checked without a score

Fig 12(a) draws four curves — phase and group for two modes — and they
cross. Every other curve in this repository has its identification
confirmed by the overlay score itself; here there is no score, so the
pairing was checked another way: **differentiating the phase curve must
reproduce its group partner.**

Using `v_g = v_p / (1 - (omega/v_p) dv_p/domega)` on the traced
fundamental phase curve gives an Airy minimum of **1.207 km/s at 12.1
kHz**, against **1.235 km/s at 10.7 kHz** read off the traced group curve.
The dip is reproduced in both depth and position, which is what confirms
the assignment. Above ~15 kHz the derivative is too noisy to be
quantitative — the phase curve moves only about two pixels over
18-25 kHz, so `dv_p/domega` there is dominated by trace noise.

That check is why the two *phase* curves are shipped and the two *group*
curves are not: the group curves were traced only to verify the pairing.
