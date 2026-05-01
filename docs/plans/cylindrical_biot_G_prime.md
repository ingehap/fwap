# Plan G': cased-hole multi-layer flexural (n=1, propagator matrix)

Detailed expansion of the deferred-follow-up note in
[item G of `docs/plans/cylindrical_biot.md`](cylindrical_biot.md)
and [`cylindrical_biot_G.md`](cylindrical_biot_G.md). Generalises
F.2's single-extra-layer 10x10 hand-coded determinant
(`_modal_determinant_n1_layered`) to N stacked annular layers
via a Thomson-Haskell-style 6x6 propagator-matrix at azimuthal
order n=1 (dipole flexural). This is the n=1 sister of plan G
(n=0 Stoneley), shipped via PRs #57 and #58.

## Status of plan G' overall

- ✅ G'.0 — public-API foundation
- ✅ G'.a — math scaffolding (substep blocks, comments only)
- ✅ G'.b — n=1 per-layer propagator helper (G'.b.1 + G'.b.2)
- ✅ G'.c — n=1 stacked modal determinant (assembly + collapse oracles)
- ✅ G'.d — n=1 public API + multi-layer regression
- ✅ G'.e — n=1 hardening
- ✅ G'.f — cross-cutting docs

Plan G' is the immediate follow-up to plan G; the propagator-
matrix scaffolding from G transfers directly with the substitutions
``4 -> 6`` (state vector / propagator size) and ``7 -> 10`` (final
modal determinant size). Slow-formation regime constraint
(``layer.vs >= vs`` for every layer) carries over from F.2.

## Scope note

This sub-plan covers **n=1 only** (flexural in cased holes), the
highest-priority deferred follow-up after G. The n=2 (quadrupole)
cased-hole counterpart is deferred to plan G''; it reuses the same
6x6 propagator scaffolding with 8x8 per-layer blocks (the n=2
problem has more potential coupling channels but the same
Helmholtz decomposition).

## Parameterisation

Reuses `BoreholeLayer` from F unchanged. The public API takes a
tuple of layers ordered **inside-out** (same convention as G):

```python
def flexural_dispersion_layered(
    freq: np.ndarray,
    *,
    vp: float, vs: float, rho: float,
    vf: float, rho_f: float, a: float,
    layers: tuple[BoreholeLayer, ...] = (),
) -> BoreholeMode
```

`flexural_dispersion_layered` is already wired with `len(layers)
== 0` and `len(layers) == 1` dispatches; G'.d widens the path to
`len(layers) >= 2`.

### Slow-formation constraint

F.2 enforces `layer.vs >= vs` (layer must be at least as fast as
the formation in shear) so the wave is bound in the annulus.
G' inherits this constraint per layer:

> For every layer in the stack, `layer.vs >= vs` (formation V_S).

Validated by extending the existing `_validate_borehole_layers`
or adding a thin `_validate_borehole_layers_stacked_n1(layers, a, vs)`
helper that includes the constraint.

The `_validate_borehole_layers_stacked` helper from G.0 covers
geometry-only checks; G'.0 either layers a slow-formation check
on top or inlines it in `flexural_dispersion_layered`.

## G'.0 — public-API foundation (~50 lines + 4 tests)

The dispatch hook is already in place from G.0: the
`flexural_dispersion_layered` NIE message currently points at
plan G'. G'.0 sharpens the message to G'.c / G'.d (specific
sub-units), and adds the per-layer slow-formation validator.

* `_validate_flexural_layers_stacked(layers, a, vs)` — wraps
  `_validate_borehole_layers_stacked` with the per-layer
  `layer.vs >= vs` check (matches F.2's existing single-layer
  check).
* `flexural_dispersion_layered` retains the `len(layers) == 0`
  and `len(layers) == 1` paths (bit-equivalent to existing) and
  routes `len(layers) >= 2` through the new propagator path. The
  existing single-layer hand-coded F.2 path (10x10) is preserved
  as the floating-point oracle.

**Tests** (4):

- Validator accepts a typical multi-layer stack with all layers
  >= formation V_S.
- Validator rejects a stack with one layer slower than V_S,
  identifying the index in the error message.
- `len(layers) == 2` NIE message points at G'.c / G'.d (sharpens
  the G.0 wording).
- N=0 / N=1 dispatch regressions remain bit-unchanged.

## G'.a — math scaffolding (~300 lines comments-only)

Inline substep blocks in `cylindrical_solver.py` mirroring G.a but
at n=1. The state vector and propagator are larger (6x6) and the
SH polarisation is genuinely coupled into the cos/sin sectors at
n=1 (per the F.2.a.6 erratum).

* **G'.a.1** — State vector at azimuthal order n=1:
  ``v(r) = (u_r, u_z, u_theta, sigma_rr, sigma_rz, sigma_rtheta)^T``,
  six components. Versus n=0's four: u_theta and sigma_rtheta
  are non-trivial at n=1 (the SH polarisation couples through
  the d_theta operations at order 1).
* **G'.a.2** — Mode-amplitude to state-vector matrix E(r) (6x6).
  Six columns: I/K flavours of the P scalar potential
  (``B_I I_n(p r) + B_K K_n(p r)``), the SV vector potential
  theta-component (``C_I I_n(s r) + C_K K_n(s r)``), and the SH
  vector potential z-component (``D_I I_n(s r) + D_K K_n(s r)``)
  at azimuthal order n=1. Six rows: the six state-vector
  components, with the F.2.a.5 phase rescale (row * i for
  z-derivative-bearing rows; col * -i for the SV columns;
  potentially also the SH columns -- pin during G'.a.2
  derivation).
* **G'.a.3** — Layer propagator P_j = E_j(r_outer) E_j(r_inner)^{-1}
  (6x6). Same composition law as G.a.3:
  ``P_total = P_N ... P_2 P_1`` maps the 6-vector at r=a to the
  6-vector at r=b across the entire stack.
* **G'.a.4** — Boundary conditions and 10x10 modal determinant:
  - **At r=a (fluid-solid interface)**: 4 BCs --
    BC1 ``u_r^(f) = u_r^(m)`` (cos sector),
    BC2 ``sigma_rr^(m) = -P^(f)`` (cos),
    BC3 ``sigma_r_theta^(m) = 0`` (sin),
    BC4 ``sigma_rz^(m) = 0`` (cos).
    The fluid imposes no constraint on ``u_z^(m)`` or
    ``u_theta^(m)`` (inviscid), so two of the six state-vector
    rows at r=a are unused.
  - **At r=b (annulus-formation interface for the outermost
    layer)**: 6 BCs -- continuity of all six state components
    ``u_r``, ``u_theta``, ``u_z``, ``sigma_rr``, ``sigma_rz``,
    ``sigma_r_theta``.
  - Total: 4 + 6 = 10 BC rows. With unknowns
    ``(A | c_1 [6 amps] | B_form_K, C_form_K, D_form_K)`` =
    1 + 6 + 3 = 10, the system is square.
* **G'.a.5** — Numerical conditioning. Same disparate-magnitude
  story as n=0 (cond(E_n1) ~ mu ~ 1e10); the round-trip /
  composition oracles in G'.b.2 must use the state-vector
  formulation (mirroring the G.b.2 footgun called out there) to
  avoid spurious off-diagonal residuals in raw matrix-equality
  tests.
* **G'.a.6** — Single-layer collapse identity (G'.b -> F.2
  oracle). At N=1, P_1 @ E_1(a) = E_1(b), so the propagator-
  chain G'.c form should match F.2's hand-coded
  ``_modal_determinant_n1_layered`` exactly (no overall scale
  factor; same brentq root in k_z). Direct mirror of G.a.6.
* **G'.a.7** — Self-check protocol for G'.b / G'.c / G'.d:
  identity propagator, composition (state-vector form),
  state-vector continuity, zero-/single-layer collapse,
  order-matters at N=2, slow-formation V_R-bracketed slowness
  smoke at N>=2.

## G'.b — n=1 per-layer propagator helper (~200 lines + 11 tests)

Two sub-units mirroring G.b.

### G'.b.1 — `_layer_e_matrix_n1` (~140 lines + 6 tests)

```python
def _layer_e_matrix_n1(
    kz: float, omega: float,
    *, vp: float, vs: float, rho: float, r: float,
) -> np.ndarray
```

Returns the 6x6 mode-amplitude-to-state-vector matrix E(r) for
n=1. Direct transcription of substep G'.a.2.

**Tests (6).**

- **Per-element match vs F.2.b at r=a** (rows 1, 2, 3, 4): the
  layer-amplitude columns of `_layered_n1_row{1,2,3,4}_at_a`
  with explicit BC sign factors match the corresponding rows
  of E(a) to ``rtol=1e-12``. Covers four of six rows of E.
- **Per-element match vs F.2.b at r=b** (rows 5, 6, 7, 8, 9, 10):
  all six state components are constrained at r=b, so the layer-
  amplitude columns of `_layered_n1_row{5,6,7,8,9,10}_at_b` cover
  every row of E(b). Exhaustive per-element pin against F.2.
- **Real-valued in bound regime.**
- **NaN below bound floor.**
- **Non-zero determinant in bound regime** (precondition for
  G'.b.2 inverse).
- **Cos / sin sector decoupling** (where it holds): at n=1 the
  cos/sin sectors are **not** block-diagonal (per F.2.a.6
  erratum) -- but specific row/col pairs in E(r) do decouple
  (e.g. the SH amplitudes only contribute to u_theta and
  sigma_r_theta, not to sigma_rr / sigma_rz). Pin those
  zero-entries explicitly.

### G'.b.2 — `_layer_propagator_n1` (~60 lines + 5 tests)

```python
def _layer_propagator_n1(
    kz: float, omega: float,
    *, vp: float, vs: float, rho: float,
    r_inner: float, r_outer: float,
) -> np.ndarray
```

Same body as G.b.2 with the dimension change ``4 -> 6``:
``np.linalg.solve(E_inner.T, E_outer.T).T``.

**Tests (5).**

- Identity at ``r_inner == r_outer``.
- Round-trip via state-vector identity (mirrors G.b.2; raw
  matrix-equality fails on the ``cond(E) ~ mu`` issue).
- Composition group law.
- State-vector continuity end-to-end.
- NaN propagation below bound floor.

## G'.c — n=1 stacked modal determinant (~250 lines + 6 tests)

```python
def _modal_determinant_n1_cased(
    kz: float, omega: float,
    *, vp: float, vs: float, rho: float,
    vf: float, rho_f: float, a: float,
    layers: tuple[BoreholeLayer, ...],
) -> float
```

Algorithm:
1. Compute layer radii r_0 = a, r_j = r_{j-1} + thickness_j.
2. Build per-layer 6x6 propagators P_j via G'.b.2 and compose
   ``P_total = P_N ... P_2 P_1``.
3. Build E_1(a) for the innermost layer (6x6) and E_form(b) for
   the formation half-space (6x6). Pick K-flavour columns of
   E_form (3 of 6).
4. Compose ``v_at_b = P_total @ E_1(a)``: the 6x6 map from c_1
   to v(b).
5. Assemble the 10x10 modal matrix:
   - Rows 0-3: BC1-4 at r=a (4 of 6 state components: u_r,
     sigma_rr, sigma_r_theta, sigma_rz). Layer cols 1-6 use
     E_1(a) with BC-specific sign factors. Fluid col 0 carries
     the fluid contribution to BC1 / BC2; formation cols 7-9
     are zero at r=a.
   - Rows 4-9: BC5-10 at r=b (all 6 state components). Layer
     cols 1-6 use ``v_at_b``. Formation cols 7-9 use the K-flavour
     columns of E_form(b) (negated, BC5-10 are layer - formation).
6. Take ``det(M)``.

**Tests (6).**

- **N=1 floating-point oracle**: G'.c determinant matches F.2's
  ``_modal_determinant_n1_layered`` to ``rtol=1e-10`` off-root
  (mirror of the G.c -> F.1 collapse).
- **N=1 brentq-root oracle**: |det_at| << |det_off| at the F.2-
  recovered flexural root.
- **NaN propagation below bound floor.**
- **Two-identical-layers ≡ single-double-thickness** via
  propagator group law.
- **Order-matters at N=2.**
- **N=2 casing + cement smoke** at a representative bound-regime
  ``kz``.

## G'.d — n=1 public API + multi-layer regression (~80 lines + 6 tests)

Replaces the `len(layers) >= 2` `NotImplementedError` in
`flexural_dispersion_layered` with a brentq loop on
`_modal_determinant_n1_cased`. Adds a
`_flexural_kz_bracket_cased` helper generalising F.2's bracket
to N layers (same pattern as G.d's `_stoneley_kz_bracket_cased`).

**Tests (6).**

- **N=1 dispatch regression** (F.2 path unchanged).
- **N=2 casing + cement smoke** across the dipole-sonic band
  (3-12 kHz) with a smoothness fence.
- **BoreholeMode contract** (`name = "flexural"`,
  `azimuthal_order = 1`, `attenuation_per_meter is None`).
- **Layer permutation distinctness**.
- **Two-layer-collapse-to-N=1** via thin trivial outer layer.
- **N=3 (casing + cement + mudcake) smoke**.

Removes the obsolete G'-deferred NIE test for n=1 multi-layer.

## G'.e — n=1 hardening (~80 lines + 4 tests)

Mirror of G.e for the flexural cased-hole solver.

* **Multi-frequency det-at-root self-consistency** at every
  brentq-converged kz across 4-15 kHz.
* **Thin-inner-layer collapse to N=1 outer-only**.
* **Two-formation-layers collapse to F.2 single-layer** (or to
  unlayered if `flexural_dispersion` is called with no layer):
  master-plan G' validation bullet equivalent to G.e bullet 3.
* **Cement-bond physics for flexural**: stiffer cement gives a
  flexural slowness closer to the formation's V_S (less casing
  / fluid contamination), softer cement gives slowness closer
  to the casing-extensional speed. Direct test of the qualitative
  cement-bond signature for the dipole sonic.

## G'.f — Cross-cutting docs (~30 lines)

- Mark plan G' done in `docs/plans/cylindrical_biot.md` (update
  the "✅ DONE (n=0)" status to include n=1; refresh status
  snapshot and Suggested-order section).
- Update plan G doc `cylindrical_biot_G.md` to point at the G'
  follow-up as completed.
- Update module docstring scope to mention multi-layer cased-hole
  flexural support alongside the existing G n=0 description.
- Commit pointers below.

### Landing commits (branch `claude/plan-cylindrical-biot-f-JgjGH`)

- `514bde4` — G'.0  public-API foundation
  (`_validate_flexural_layers_stacked` + sharpened NIE)
- `2075534` — G'.a  math scaffolding (substep blocks)
- `d05f00c` — G'.b.1  6×6 `_layer_e_matrix_n1` (36/36 entries
  pinned vs F.2 row builders to `rtol=1e-12`)
- `edc2758` — G'.b.2  6×6 `_layer_propagator_n1` (state-vector
  round-trip oracle)
- `7f31e22` — G'.c  10×10 `_modal_determinant_n1_cased` (N=1
  numerical match with F.2 to `rtol=1e-10`)
- `b6a4888` — G'.d  wire `flexural_dispersion_layered` for
  multi-layer + `_flexural_kz_bracket_cased`
- `ff6de6d` — G'.e  hardening (multi-frequency det-at-root,
  layer-collapse oracles, cement-stiffness physics)

## Total scope

~960 lines of solver code + ~750 lines of tests + ~31 tests,
distributed across ~7 mergeable PRs (G'.0 + G'.a; G'.b.1; G'.b.2;
G'.c; G'.d; G'.e + G'.f bundled). Conservative estimate: 5-7
days of focused work.

Risk concentrated in:
- **6x6 matrix transcription**. 36 entries per E(r); F.2.b/c row
  builders are split across cos/sin sectors, so per-element
  oracle bookkeeping needs careful sign-factor tracking. Mitigation:
  build E(r) directly from G'.a.2, then write per-row tests one
  F.2.b/c builder at a time.
- **SH-coupling at n=1**. The F.2.a.6 erratum (cos/sin sectors are
  NOT block-diagonal at n=1) carries over. The 6x6 E(r) is
  fully populated except for known-zero entries that the G'.b.1
  decoupling test enforces.
- **Conditioning at flexural root**. The flexural root in slow
  formations sits very close to ``omega / V_S``, so the radial
  Bessel arguments ``s a`` and ``s b`` are small in the LF limit;
  E(r) entries blow up. The G.b.2 round-trip-via-state-vector
  oracle pattern still works; multi-frequency coverage in G'.e
  catches any numerical instability.
- **Slow-formation constraint**. Different from G's "any
  formation" coverage at n=0: every layer must satisfy
  ``layer.vs >= vs``. The G'.0 validator rejection test pins
  this; the G'.d smoke / regression tests use a typical "harder
  layer" cased-hole geometry.

## Deferred follow-ups (separate plans)

- **G'' — n=2 cased-hole quadrupole** (~700 lines + 25 tests).
  Sister of G' at azimuthal order 2; mirrors the G + G'
  scaffolding with 8x8 per-layer propagators (n=2 has stronger
  P / SV / SH coupling). LWD relevance.
- **Tang & Cheng 2004 fig 7.1 reproduction** for the cased-
  hole flexural (digitised CSV; deferred from G.e).

## References

- Schmitt, D. P. (1988). Shear-wave logging in elastic
  formations. *J. Acoust. Soc. Am.* 84(6), 2230-2244 (n=1
  modal determinant).
- Schmitt, D. P., & Bouchon, M. (1985). Full-wave acoustic
  logging: synthetic microseismograms and frequency-wavenumber
  analysis. *Geophysics* 50(11), 1756-1778 (cylindrical-
  geometry propagator matrix).
- Tang, X. M., & Cheng, C. H. (2004). *Quantitative Borehole
  Acoustic Methods*. Elsevier. Ch. 7 (cased-hole flexural; the
  validation target for G'.e once digitised).
- Knopoff, L. (1964). A matrix method for elastic wave problems.
  *Bull. Seismol. Soc. Am.* 54(1), 431-438. Delta-matrix
  formulation referenced in conditioning notes.

## Execution order

1. **G'.0** (foundation: validator + sharpened NIE message) --
   anchors the floating-point regression test.
2. **G'.a** (math scaffolding) -- comments only; pins the 6x6
   E(r) transcription and the 4-at-r=a / 6-at-r=b BC partition.
3. **G'.b.1** (E(r) helper) -- per-element oracle vs F.2.b/c row
   builders. Strongest unit-level pin.
4. **G'.b.2** (propagator) -- group-law oracles via state-vector
   identities (mirroring the G.b.2 conditioning lesson).
5. **G'.c** (stacked determinant) -- N=1 numerical match with
   F.2 to ``rtol=1e-10``; multi-layer collapse oracles.
6. **G'.d** (public API + multi-layer regression) -- first
   physically-meaningful G' output.
7. **G'.e** (hardening + cement-bond physics).
8. **G'.f** (docs).
