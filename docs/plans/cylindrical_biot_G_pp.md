# Plan G'': cased-hole multi-layer quadrupole (n=2, propagator matrix)

Detailed expansion of the deferred-follow-up note in
[item G of `docs/plans/cylindrical_biot.md`](cylindrical_biot.md)
and the G' / G'' notes in [`cylindrical_biot_G.md`](cylindrical_biot_G.md)
and [`cylindrical_biot_G_prime.md`](cylindrical_biot_G_prime.md).
Generalises the existing unlayered n=2 quadrupole solver
(`_modal_determinant_n2`, `quadrupole_dispersion`) to N stacked
annular layers via a Thomson-Haskell-style 6x6 propagator matrix
at azimuthal order n=2. This is the n=2 sister of plans G (n=0)
and G' (n=1).

## Status of plan G'' overall

- ⏳ G''.0 — public-API foundation (introduces
  `quadrupole_dispersion_layered`)
- ⏳ G''.a — math scaffolding (substep blocks, comments only)
- ⏳ G''.b — n=2 per-layer propagator helper (G''.b.1 + G''.b.2)
- ⏳ G''.c — n=2 stacked modal determinant (assembly + collapse oracles)
- ⏳ G''.d — n=2 public-API hook + multi-layer regression
- ⏳ G''.e — n=2 hardening
- ⏳ G''.f — cross-cutting docs

Plan G'' depends on the propagator-matrix scaffolding from plans
G (4x4 at n=0) and G' (6x6 at n=1); it inherits the same
six-component state vector and 10x10 final-form structure as G'.
The only structural difference is the Bessel-function index shift
``(I_{n-1}, I_n) -> (I_1, I_2)`` and the explicit ``n = 2``
factors that come out of the ``d_theta cos(n theta) = -n sin(n
theta)`` step (per `_modal_determinant_n2` substep).

## Scope note (no F.3-equivalent single-layer oracle)

Plans G (n=0) and G' (n=1) had **per-element oracles** for the
E(r) helper via the existing F.1 / F.2 hand-coded layered
determinants (G.b.1 used `_layered_n0_row{1..7}_at_{a,b}`;
G'.b.1 used `_layered_n1_row{1..10}_at_{a,b}`, all 36 entries
pinned to ``rtol=1e-12``). For G'' there is no F.3-equivalent
hand-coded single-layer n=2 form. The G''.b.1 per-element
oracle is therefore **weaker**: it relies on:

1. **Direct transcription** from the substep G''.a.2 block
   (internal consistency).
2. **Layer = formation N=1 collapse** to the existing 4x4
   unlayered `_modal_determinant_n2` (via G''.c brentq root
   match; root-level oracle, not entry-level).
3. **Propagator group-law oracles** in G''.b.2 (identity,
   composition, state-vector continuity).
4. **Two-formation-layers collapse to unlayered** in G''.e
   (multi-layer root match).

A separate **F.3 plan** (single-extra-layer hand-coded n=2 with
10x10 row builders) is a tractable optional prerequisite if
per-element pinning becomes important; it would cost ~600 lines
+ 25 tests and would mirror F.2 with the Bessel-index shift.
**Not required** to land G'' MVP; flagged here for future
work.

## Parameterisation

Reuses `BoreholeLayer` from F unchanged. New public API:

```python
def quadrupole_dispersion_layered(
    freq: np.ndarray,
    *,
    vp: float, vs: float, rho: float,
    vf: float, rho_f: float, a: float,
    layers: tuple[BoreholeLayer, ...] = (),
) -> BoreholeMode
```

Structurally identical to `stoneley_dispersion_layered` /
`flexural_dispersion_layered`. With `layers=()` it dispatches
to `quadrupole_dispersion`. With `len(layers) >= 1` it routes
through the propagator-matrix path.

Slow-formation regime constraint (`layer.vs >= vs` per layer)
inherits from the n=1 case and the LWD-quadrupole physics
(soft-formation analogue of Sinha-Norris-Chang). Validated by
a new `_validate_quadrupole_layers_stacked` helper or by reuse
of `_validate_flexural_layers_stacked` (the constraint is the
same).

## G''.0 — public-API foundation (~150 lines + 6 tests)

Introduces `quadrupole_dispersion_layered` from scratch (vs G.0
/ G'.0 which only sharpened existing NIEs):

* `quadrupole_dispersion_layered(...)` public API. Exports from
  `fwap.cylindrical_solver` and `fwap`.
* With `layers=()` dispatches bit-equivalently to
  `quadrupole_dispersion`.
* With `len(layers) >= 1` raises `NotImplementedError` pointing
  at G''.c / G''.d (the scaffolding that ships in this plan).
* Validation rules: positivity, ``vp > vs``, slow-formation
  constraint per layer (reuses `_validate_flexural_layers_stacked`
  or introduces `_validate_quadrupole_layers_stacked`).
* Fast-formation case (``V_S > V_f``): for the single-interface
  `quadrupole_dispersion` this is auto-dispatched to the leaky
  complex-determinant variant; for the layered case the
  fast-formation propagator path is deferred (analogous to F.2's
  fast-formation NIE).

**Tests (6):**

- ``layers=()`` regression matches `quadrupole_dispersion`
  bit-equivalently across a frequency grid.
- ``len(layers) == 1`` raises `NotImplementedError` pointing at
  G''.c / G''.d (sharpens the message after subsequent sub-
  units land).
- BoreholeMode return-type contract (``name = "quadrupole"``,
  ``azimuthal_order = 2``).
- Validation rejection: per-layer slow-formation constraint
  (`layer.vs < vs` rejected with index in error).
- Validation rejection: positivity / `vp <= vs` / non-positive
  `freq`.
- Fast-formation NIE: with `V_S > V_f` and a non-empty layer,
  raises `NotImplementedError` pointing at the deferred fast-
  formation layered-quadrupole follow-up.

## G''.a — math scaffolding (~300 lines comments-only)

Substep blocks G''.a.1 through G''.a.7 mirroring G'.a with the
Bessel-index shift ``(I_1, I_2)`` and the n=2 azimuthal
factors.

* **G''.a.1** — State vector at quadrupole order n=2:
  ``v(r) = (u_r, u_z, u_theta, sigma_rr, sigma_rz,
  sigma_r_theta)^T``, six components (same as n=1). At
  azimuthal order n=2 the cos / sin sectors carry the
  ``cos(2 theta)`` / ``sin(2 theta)`` factors; each
  ``d_theta`` brings out a factor of ``n = 2`` rather than
  ``1``. The state-vector-component continuity at every layer/
  layer interface is unchanged from n=1.
* **G''.a.2** — 6x6 mode-amplitude-to-state-vector matrix
  ``E_n2(r)``. Six columns: I/K flavours of the P scalar
  (``B_I I_2(p r) + B_K K_2(p r)``), SV vector
  (``C_I I_2(s r) + C_K K_2(s r)``), and SH vector
  (``D_I I_2(s r) + D_K K_2(s r)``) at azimuthal order n=2.
  Six rows: the six state-vector components.

  **Bessel-derivative identities at n=2** (transcribed from
  the existing `_modal_determinant_n2`):

  * ``d_r I_2(p r) = p [I_1(p r) - (2/(pr)) I_2(p r)]
                   = p I_1(p r) - 2 I_2(p r) / r``
  * ``d_r K_2(p r) = -p K_1(p r) - 2 K_2(p r) / r``
  * ``(1/r) d_r [r I_1(s r)]  = ...`` (carried over from F.2)
  * Similar for SV and SH amplitudes.

  **Explicit n=2 factors** appearing in the stress entries
  (per `_modal_determinant_n2`):

  * ``2 n (n + 1) = 12`` in the ``B_K K_2 / r^2`` term of
    sigma_rr.
  * ``n^2 - 1 = 3`` in the SV column of sigma_rz.

* **G''.a.3** — Layer propagator
  ``P_j = E_j(r_outer) E_j(r_inner)^{-1}`` (6x6). Composition
  ``P_total = P_N ... P_2 P_1`` -- identical to G'.a.3 with the
  Bessel-index shift baked into ``E_j``.
* **G''.a.4** — Boundary conditions and 10x10 modal determinant.
  Same BC structure as G'.a.4: 4 BCs at r=a (u_r continuity,
  sigma_rr balance, sigma_rtheta = 0, sigma_rz = 0; the inviscid
  fluid imposes no constraint on u_z or u_theta) + 6 BCs at r=b
  (full state-vector continuity). Final 10x10 form:
  ``[A | B_I, B_K, C_I, C_K | B_form, C_form | D_I, D_K |
  D_form]``. Reduces at N=0 to the unlayered `_modal_determinant_n2`
  (4x4 form) after the propagator chain is bypassed.
* **G''.a.5** — Numerical conditioning. Same disparate-magnitude
  story (``cond(E_n2) ~ mu ~ 1e10``); the state-vector form
  for round-trip oracles in G''.b.2 mitigates the
  raw-matrix-equality footgun (G.b.2 / G'.b.2 lesson).

  **n=2 specific concern**: the ``I_2(p r)`` Bessel function
  grows as ``(p r / 2)^2 / 2`` at small ``p r`` and as
  ``e^{p r} / sqrt(2 pi p r)`` at large ``p r``. The small-
  argument behavior makes the I-flavour columns disproportion-
  ately small near ``r = a``, contributing to slightly worse
  conditioning at low frequencies. Likely still within double
  precision for typical cased-hole geometries; flagged as a
  potential issue in G''.b.2 round-trip tests.

* **G''.a.6** — Layer = formation collapse identity (G''.c
  oracle). With the lack of an F.3-equivalent hand-coded form,
  the strongest collapse oracle is:

  ``layer.{vp, vs, rho} == formation.{vp, vs, rho}``
  ``=>`` cased-hole ``_modal_determinant_n2_cased`` brentq
  root in ``k_z`` matches the unlayered
  ``_modal_determinant_n2`` brentq root in ``k_z``.

  This is the master-plan G''-equivalent of the G validation
  bullet 1 (two-formation-layers collapse to unlayered). The
  determinant magnitudes will differ (different overall
  scaling factors) but the brentq root is invariant.

* **G''.a.7** — Self-check protocol:
  * (a) Identity propagator: ``r_outer = r_inner``.
  * (b) Composition group law.
  * (c) State-vector continuity.
  * (d) N=0 dispatch to `quadrupole_dispersion`.
  * (e) Layer = formation N=1 collapse to unlayered root.
  * (f) Order-matters at N=2.
  * (g) LWD-quadrupole physics smoke: cement-bond signature
    distinct from the unlayered case.

## G''.b — n=2 per-layer propagator (~250 lines + 11 tests)

Two sub-units mirroring G'.b.

### G''.b.1 — `_layer_e_matrix_n2` (~180 lines + 6 tests)

```python
def _layer_e_matrix_n2(
    kz: float, omega: float,
    *, vp: float, vs: float, rho: float, r: float,
) -> np.ndarray
```

Returns the 6x6 mode-amplitude-to-state-vector matrix E(r) for
n=2. Direct transcription of substep G''.a.2.

**Tests (6).** Without an F.3-equivalent oracle the per-element
pinning is internal:

- **Real-valued in bound regime.** All 36 entries finite real
  post-rescale.
- **NaN below bound floor.**
- **Non-zero determinant in bound regime** (precondition for
  G''.b.2 inverse).
- **Sparsity pattern** (analogous to G'.b.1):
  * Row 1 (``u_z``) cols 4, 5 (``D_I``, ``D_K``): zero (SH no
    u_z contribution).
  * Row 2 (``u_theta``) cols 2, 3 (``C_I``, ``C_K``): zero (SV
    no u_theta contribution).
- **Cross-check vs unlayered `_modal_determinant_n2`** at
  layer = formation: build E_layer(r=a) and E_form(r=a) for
  identical params; verify the 4 K-flavour cols of E (cols
  ``B_K, C_K, D_K`` and the 4 state-rows used at r=a) match
  the rows of `_modal_determinant_n2` at the same kz/omega/
  layer params. Indirect per-element oracle (via the
  formation amplitudes).
- **n=2-specific scaling**: at small ``p r``, ``I_2(p r) ~
  (p r)^2 / 8``. Confirm the B_I col entries scale quadratically
  with ``p`` near the small-argument limit (catches a sign or
  Bessel-order error in the ``I_2`` derivative formulas).

### G''.b.2 — `_layer_propagator_n2` (~70 lines + 5 tests)

```python
def _layer_propagator_n2(
    kz: float, omega: float,
    *, vp: float, vs: float, rho: float,
    r_inner: float, r_outer: float,
) -> np.ndarray
```

Mirror of G'.b.2 with `_layer_e_matrix_n2`. Uses
``np.linalg.solve(E_inner.T, E_outer.T).T``.

**Tests (5).** Same group-law oracles as G'.b.2:

- Identity at ``r_inner == r_outer``.
- Round-trip via state-vector identity (state-vector form to
  dodge the ``cond(E) ~ mu`` issue).
- Composition group law.
- State-vector continuity end-to-end.
- NaN propagation below bound floor.

## G''.c — n=2 stacked modal determinant (~300 lines + 6 tests)

```python
def _modal_determinant_n2_cased(
    kz: float, omega: float,
    *, vp: float, vs: float, rho: float,
    vf: float, rho_f: float, a: float,
    layers: tuple[BoreholeLayer, ...],
) -> float
```

Algorithm identical to G'.c with the dimension change
``(n=1 -> n=2)``. The 10x10 final form has the same column
packing as G'.c:
``[A | B_I, B_K, C_I, C_K | B_form, C_form | D_I, D_K | D_form]``.

The state-row to BC-row mapping at r=a (4 BCs) and r=b (6 BCs)
is also unchanged from G'.c.

**Tests (6).** Without F.3-equivalent the N=1 numerical-equality
oracle of G'.c (matching F.2 to ``rtol=1e-10``) is **not
available**. Replace with:

- **Layer = formation N=1 root match**: at layer params equal
  to formation params, the N=1 cased determinant has the same
  brentq root in kz as the unlayered `_modal_determinant_n2`.
  Tested via the brentq pipeline in G''.d (det-at-root oracle).
  Internal: ``|det_at|`` is many orders of magnitude smaller
  than ``|det_off|`` at the unlayered-recovered root.
- **NaN propagation below bound floor.**
- **Two-identical-layers ≡ single-double-thickness** via the
  propagator group law (mirrors G.c / G'.c).
- **Order-matters at N=2** (distinct casing-inside-cement vs
  cement-inside-casing slownesses).
- **N=2 casing + cement smoke**.
- **N=0 dispatch to unlayered**: ``len(layers) == 0`` returns
  the unlayered slowness without going through `_modal_determinant_n2_cased`.

## G''.d — n=2 public-API hook (~80 lines + 6 tests)

Replaces the G''.0 ``len(layers) >= 1 -> NotImplementedError``
raise in `quadrupole_dispersion_layered` with a brentq loop on
`_modal_determinant_n2_cased`. New `_quadrupole_kz_bracket_cased`
helper.

The brentq bracket for the n=2 quadrupole follows the same
strategy as G'.d's flexural bracket: lower bound at the slowest-
body-wave floor across the entire stack, upper bound at the
formation Rayleigh-speed slowness with a 10 % cushion.

**Tests (6).**

- N=1 dispatch regression (slow-formation root match vs the
  unlayered `quadrupole_dispersion` when layer = formation).
- N=2 casing + cement smoke across the LWD-relevant band
  (10-25 kHz).
- BoreholeMode return-type contract.
- Layer permutation distinctness.
- Two-layer-collapse-to-N=1 via thin trivial outer layer.
- N=3 (casing + cement + mudcake) smoke.

## G''.e — n=2 hardening (~80 lines + 4 tests)

Mirror of G.e / G'.e for the quadrupole cased-hole solver.

* **Multi-frequency det-at-root self-consistency** at every
  brentq-converged kz across 8-25 kHz (LWD band).
* **Thin-inner-layer collapse** to N=1 outer-only.
* **Two-formation-layers collapse to unlayered** (master-plan
  G'' validation bullet, ``rtol=1e-6``).
* **LWD-quadrupole cement-bond physics**: stiffer cement gives
  a slowness shift in a measurable direction relative to softer
  cement at the same casing. Direct test of the LWD cement-bond
  signature (qualitative; the direction is the empirical
  observation pinned in the test).

## G''.f — Cross-cutting docs (~30 lines)

- Mark plan G'' done in `docs/plans/cylindrical_biot.md`
  (update G section status to "✅ DONE (n=0, n=1, n=2)";
  refresh status snapshot).
- Update plan G doc to point at G'' as completed.
- Update plan G' doc similarly.
- Update module docstring scope to mention multi-layer cased-
  hole quadrupole support.
- Commit pointers in this plan doc.

## Total scope

~1100 lines of solver code + ~700 lines of tests + ~33 tests,
distributed across ~7 mergeable PRs (G''.0; G''.a; G''.b.1;
G''.b.2; G''.c; G''.d; G''.e + G''.f bundled). Conservative
estimate: 5-7 days of focused work (similar to G').

Risk concentrated in:

- **No F.3-equivalent per-element oracle for G''.b.1**. The
  six-test suite pins the matrix structurally (sparsity,
  finiteness, Bessel-scaling, layer=formation cross-check) but
  not entry-by-entry as G/G' did. Mitigation: write the
  transcription very carefully against `_modal_determinant_n2`'s
  formulas; rely on G''.c / G''.d / G''.e for the
  determinant-level checks. Optional follow-up: a F.3 plan that
  hand-codes the single-extra-layer 10x10 with per-row
  builders, then retrofits G''.b.1 with per-element oracles
  (~600 lines + 25 tests).
- **n=2 Bessel-function scaling at low frequencies**. ``I_2(p
  r)`` is small near ``p r = 0`` (proportional to ``p r``
  squared), making the I-flavour columns of E(r) small near
  ``r = a``. Conditioning of ``E(r_inner)`` may degrade
  slightly compared to n=0/1. The state-vector round-trip
  oracle in G''.b.2 catches the practical impact.
- **LWD-band parameter selection**. The quadrupole cutoff
  shifts with the cased-hole geometry; the G''.d smoke band
  may need tuning to stay above the cased-cutoff while
  remaining within the LWD-relevant 5-25 kHz window.

## Deferred follow-ups (separate plans)

- **F.3 — single-extra-layer hand-coded n=2** (~600 lines + 25
  tests). Optional prerequisite that adds per-element oracles
  for G''.b.1; only needed if G''.b.1's structural-only oracle
  proves insufficient. Mirrors F.2.
- **Tang & Cheng 2004 fig 7 / 8 quadrupole reproduction** for
  cased-hole LWD-quadrupole regression (digitised CSV; deferred
  from G''.e analogous to G.e / G'.e fig 7.1 deferral).
- **Fast-formation cased-hole quadrupole**: the n=2 quadrupole
  has both bound and leaky variants in the unlayered case
  (E auto-dispatch); the cased-hole leaky variant is a
  separate plan analogous to the future fast-formation layered
  flexural follow-up to F.2.

## References

- Tang, X. M., & Cheng, A. (2004). *Quantitative Borehole
  Acoustic Methods*. Elsevier. Sect. 2.5 (LWD quadrupole modal
  determinant); ch. 7 (cased-hole logging).
- Kurkjian, A. L., & Chang, S.-K. (1986). Acoustic multipole
  sources in fluid-filled boreholes. *Geophysics* 51(1), 148-
  163. General-n derivation (eqs. 8 and 9).
- Schmitt, D. P., & Bouchon, M. (1985). Full-wave acoustic
  logging: synthetic microseismograms and frequency-wavenumber
  analysis. *Geophysics* 50(11), 1756-1778. Cylindrical-
  geometry propagator matrix at general n.

## Execution order

1. **G''.0** (foundation: introduce
   `quadrupole_dispersion_layered` API surface) -- anchors the
   floating-point regression test (`layers=()` ≡ unlayered).
2. **G''.a** (math scaffolding) -- comments only; pins the n=2
   ansatz, Bessel-index shift, and 6x6 E_n2(r) transcription.
3. **G''.b.1** (E_n2(r) helper) -- structural oracles plus
   layer=formation cross-check vs unlayered `_modal_determinant_n2`.
4. **G''.b.2** (propagator) -- group-law oracles via state-
   vector identities (mirrors G.b.2 / G'.b.2).
5. **G''.c** (stacked determinant) -- N=0 dispatch + layer=
   formation N=1 root match + multi-layer collapse oracles.
6. **G''.d** (public-API hook + multi-layer regression) -- first
   physically-meaningful G'' output.
7. **G''.e** (hardening + LWD cement-bond physics).
8. **G''.f** (docs).
