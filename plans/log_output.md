# Measurement log

Raw and near-raw outputs from the oracle work (PRs #59-#65 and the n=1/n=2
cutoff investigation), kept so the numbers quoted in `plans/learning.md`,
`docs/roadmap.md` and the changelog can be traced to something.

## Provenance and how far to trust this

**This is a transcript log, not a reproducible artefact.** Every table below
was produced by a throwaway script during the session that made the
corresponding change, and those scripts were not committed. The numbers are
copied from the session's output, not re-derived. Two consequences:

* Anything here that a test also asserts is trustworthy, because the test runs
  in CI. Anything here that no test asserts is a one-time reading and should be
  re-measured before being relied on.
* Values marked *machine-specific* are known to differ across platforms. They
  are recorded because the *fact of the variation* is itself a result, not
  because the numbers are properties of the physics.

Where a measurement was superseded, both the old and new values are kept and
the reason is stated. Numbers withdrawn as wrong are marked as such rather than
deleted.

## Leaky-mode radiation attenuation (PR #63)

Solver `attenuation_per_meter` divided by the ray radiation estimate
`Im(k_z) = -ln|R| k_f / (2 a k_z)`, over 4-30 kHz. **Asserted by tests.**

| case | median | min | max |
|---|---|---|---|
| base, a=0.10, V_S=2300 | 0.592 | 0.449 | 1.555 |
| wide hole, a=0.15 | 0.574 | 0.373 | 1.554 |
| narrow, a=0.07 | 0.654 | 0.540 | 1.566 |
| faster shear, V_S=2800 | 0.634 | 0.460 | 1.909 |
| near-slow, V_S=1700 | 0.710 | 0.457 | 1.457 |
| light fluid, rho_f=300 | 1.861 | 1.426 | 7.951 |

The light-fluid row is a gas-filled hole and sits outside the range the tests
assert; it is kept because it shows the offset is not universal.

Transverse-resonance peak spacing, supporting the `2a` round trip the estimate
assumes:

| a (m) | mean peak spacing (kHz) | spacing x a |
|---|---|---|
| 0.07 | 13.95 | 976 |
| 0.10 | 10.88 | 1088 |
| 0.15 | 6.80 | 1020 |

## Leaky-mode branch selection (PR #64)

Mode reported at 30 kHz for a 0.10 m hole in a 4000/2300/2500 formation, as a
function of where the frequency grid stops. **Before the fix:**

| grid top (kHz) | c at 30 kHz (m/s) | attenuation (1/m) | lowest converged (Hz) |
|---|---|---|---|
| 35-55 | 2486.2 | 2.4353 | 2000 |
| 60-80 | 2951.7 | 3.0610 | ~21400 |
| 100 | (does not reach 30 kHz) | — | 34421 |

**After enumerating the seeds:** `branch=0` gives 2486.16 m/s for every grid
top from 32 to 100 kHz; `branch=1` gives 2951.74 m/s for every grid top tested.
**Asserted by tests.**

Root count vs seed-scan density, 18 cases (6 media x 3 frequencies), scan
grids 24x5 through 80x16: **zero disagreements**. **Asserted by tests.**

## Stoneley low-frequency limit (PR #64)

Bracket-free determinant root at 1 Hz against the White tube-wave form.
**Asserted by tests.**

| case | relative difference |
|---|---|
| V_S=1300 | +1.54e-07 |
| V_S=1800 | +4.11e-08 |
| V_S=2300 | +1.32e-08 |
| V_S=3000 | +2.51e-09 |
| a=0.20 m | +4.52e-08 |
| heavy mud (V_f=1400, rho_f=1400) | +1.32e-08 |

Validity floor `V_S > V_f sqrt(1 - rho_f/rho)` against where the solver stops
converging:

| rho | rho_f | V_f | floor (m/s) | first converging V_S | ratio |
|---|---|---|---|---|---|
| 2200 | 1000 | 1500 | 1107.8 | 1110 | 1.002 |
| 2400 | 1000 | 1500 | 1145.6 | 1150 | 1.004 |
| 2000 | 1000 | 1500 | 1060.7 | 1070 | 1.009 |
| 2200 | 1200 | 1500 | 1011.3 | 1020 | 1.009 |
| 2200 | 800 | 1500 | 1196.6 | 1200 | 1.003 |
| 2200 | 1000 | 1300 | 960.1 | 970 | 1.010 |
| 2600 | 1000 | 1600 | 1255.1 | 1260 | 1.004 |

Approach to the limit is **not** one-sided; the relative error changes sign at
a crossover that moves with `V_S`:

| case | sign pattern, 2 kHz down to 1 Hz | \|error\| at 1 Hz |
|---|---|---|
| V_S=1300 | `+++++++++++` | 1.54e-07 |
| V_S=1600 | `-++++++++++` | 6.90e-08 |
| V_S=2300 | `----+++++++` | 1.23e-08 |
| V_S=3000 | `-------++++` | 2.53e-09 |

## Leaky-mode energy balance (PR #64, withdrawn)

Balance reproduces `Im(k_z)` to ratio 1.000 at every frequency at genuine
roots — and to ratio 1.0000 for eight arbitrary non-root `k_z` values, which is
why it was withdrawn. **Both facts asserted by tests.**

Leaky-S field magnitude vs radius, using the solver's own evaluator (the wrong
Hankel kind decays and reverses the conclusion):

| r (m) | 0.1 | 0.5 | 1.0 | 2.0 | 5.0 | 10 | 30 |
|---|---|---|---|---|---|---|---|
| \|K0_leaky(s r)\| | 0.996 | 9.66 | 196 | 1.15e5 | 4.15e13 | 1.15e28 | 1.57e86 |

Momentum-to-energy flux ratio against the predicted
`|k_z|^2 / (omega Re(k_z))`: agrees to six digits at all twelve frequencies
tried. *Not currently asserted by a test* — it supports a documentation claim
in `learning.md` only.

## Layered-solver invariances (PR #65)

Subdivision of a homogeneous annulus, maximum relative difference.
**Asserted by tests.**

| case | max rel diff |
|---|---|
| n=0 Stoneley, mudcake | 1.44e-15 |
| n=1 flexural, mudcake | 1.22e-15 |
| n=2 quadrupole, mudcake | 2.89e-15 |
| n=0 cased, split casing | 2.24e-14 |
| n=0 cased, split cement | 4.67e-14 |
| n=0 fast formation | 6.66e-16 |

Sensitivity of the same check to a stack that does *not* preserve the medium:

| perturbation | max rel diff in slowness |
|---|---|
| none | 1.11e-15 |
| thickness +0.01 % | 3.49e-06 |
| thickness +0.1 % | 3.49e-05 |
| thickness +1 % | 3.48e-04 |
| V_P of one half +0.01 % | 5.03e-07 |

Layer-order swap (A-then-B vs B-then-A): **1.19 %** — not an invariance.

Redundant formation-equal layer appended to a mudcake stack. ***Machine-specific
below 1e-10 — see the warning above.***

| added thickness (m) | this machine | CI |
|---|---|---|
| 0.01-0.10 | < 1e-10 (transparent) | transparent |
| 0.12 | c = 289 m/s | 7.5 % from plain |
| 0.15 | c = 1095 m/s (14 % off) | c = 289 m/s |
| 0.18 | c = 289 m/s | — |
| 0.20 | NaN | — |

Correct high-frequency limit for that stack is the *mudcake's* Scholte speed,
1272.503 m/s; the plain stack reaches it to 0.052 % at 100 kHz, 0.024 % at
200 kHz, 0.011 % at 400 kHz. The formation's Scholte speed is 1094.846 m/s.
Only the plain-stack convergence is asserted by tests; the padded values are
not, because they have no stable value.

## n=1 / n=2 cutoffs (this change)

Lowest converging frequency, slow formation (2200/1000/2200, V_f=1500).
**`1/a` scaling asserted by tests.**

| a (m) | n=1 f_c (Hz) | f_c x a | n=2 f_c (Hz) | f_c x a |
|---|---|---|---|---|
| 0.06 | 3599.5 | 215.97 | 7535.7 | 452.14 |
| 0.08 | 2704.9 | 216.39 | 5647.1 | 451.77 |
| 0.10 | 2168.1 | 216.81 | 4533.8 | 453.38 |
| 0.14 | 1551.8 | 217.26 | 3241.6 | 453.83 |
| 0.20 | 1094.6 | 218.92 | 2267.5 | 453.50 |

Dependence on the two velocities, at a = 0.10 m:

| V_S | n=1 f_c x a | n=2 f_c x a | | V_f | n=1 f_c x a | n=2 f_c x a |
|---|---|---|---|---|---|---|
| 700 | 155.71 | 325.37 | | 1200 | 211.60 | 437.14 |
| 850 | 187.64 | 391.24 | | 1350 | 215.59 | 447.12 |
| 1000 | 217.58 | 453.11 | | 1500 | 217.58 | 453.11 |
| 1150 | 245.53 | 511.00 | | 1700 | 219.58 | 459.10 |
| 1300 | 273.47 | 562.89 | | 1900 | 219.58 | 463.09 |
| 1450 | 297.42 | 608.80 | | | | |

Log-log sensitivities — the discriminating result. **Asserted by tests.**

| mode | exponent on V_S | exponent on V_f |
|---|---|---|
| n=1 flexural | 0.889 | 0.081 |
| n=2 quadrupole | 0.860 | 0.125 |

A fluid-column (rigid-pipe) cutoff would be ~1 on `V_f` and ~0 on `V_S`.

## Withdrawn numbers

Kept so they are not re-derived and re-believed.

* **"~15 % of draws are slow"** — measured over the default `FormationPriors`
  rather than the prior the cased generator pins, where the true figure is
  0 %. Withdrawn in PR #54.
* **Pseudo-Rayleigh cutoff ratio "0.362 to three decimals"** — an artefact of a
  frequency grid derived from the closed form under test, which shares its
  `1/a` scaling. Replaced by 0.363 +/- 0.001 on a fixed absolute grid (PR #61).
  Note also that PR #64's seed enumeration changed which frequencies converge,
  so this ratio should be re-read as a property of its own pinned grid and
  formation, not a universal constant.
* **"Utah FORGE is mirrored on AWS Open Data"** — the reachable buckets carry
  DAS, geophone, CASSM and magnetotelluric data, no wireline logs (PR #57).
* **Padded-stack error "14 %"** — holds on this machine at one thickness only;
  CI gives 7.5 % at a different thickness and a factor of four elsewhere.
