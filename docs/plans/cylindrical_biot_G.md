# Plan G: cased-hole multi-layer (propagator matrix)

Detailed expansion of [item G in
`docs/plans/cylindrical_biot.md`](cylindrical_biot.md). Generalises
F's single-extra-layer infrastructure to ``N`` stacked annular
layers (``fluid → casing → cement → [mudcake] → formation``) by
replacing the hand-coded per-row builders with a Thomson-Haskell-
style propagator-matrix.

## Status of plan G overall

- ✅ G.0 — public-API foundation
- ✅ G.a — math scaffolding (substep blocks, comments only)
- ✅ G.b — n=0 per-layer propagator helper (G.b.1 + G.b.2)
- ✅ G.c — n=0 stacked modal determinant (assembly + collapse oracles)
- ✅ G.d — n=0 public API + cement-bond regression
- ✅ G.e — n=0 hardening (Tang & Cheng 2004 fig 7.1 deferred)
- ✅ G.f — cross-cutting docs

Plan G is independent of plans A-E (leaky-mode chain) and H (VTI).
It depends on F (single-extra-layer infrastructure: `BoreholeLayer`,
the layered public APIs, and the F.1 / F.2 hand-coded determinants
that anchor the single-layer collapse oracle).

**Scope note.** This sub-plan covers **n=0 only** (Stoneley in cased
holes), the highest-priority deliverable. The n=1 (flexural) and
n=2 (quadrupole) cased-hole counterparts are deferred to follow-up
plans G' / G''; they reuse the same propagator-matrix scaffolding
with bigger per-layer blocks (6×6 at n=1, 8×8 at n=2).

## Parameterisation

Reuses `BoreholeLayer` from F unchanged. The public API takes a
tuple of layers ordered **inside-out**:

```python
def stoneley_dispersion_layered(
    freq: np.ndarray,
    *,
    vp: float, vs: float, rho: float,
    vf: float, rho_f: float, a: float,
    layers: tuple[BoreholeLayer, ...] = (),
) -> BoreholeMode
```

For a typical cased hole:

```python
casing = BoreholeLayer(vp=5860, vs=3140, rho=7800, thickness=0.01)
cement = BoreholeLayer(vp=2300, vs=1300, rho=1900, thickness=0.05)
result = stoneley_dispersion_layered(
    freq, vp=vp, vs=vs, rho=rho, vf=vf, rho_f=rho_f, a=a,
    layers=(casing, cement),
)
```

The existing `stoneley_dispersion_layered(..., layers=(single,))`
hand-coded path (F.1) and `layers=()` unlayered path
(`stoneley_dispersion`) remain untouched as the **floating-point
oracles** for the propagator path.

## G.0 — public-API foundation (~80 lines + 5 tests)

The public API hook is already in place from F: both
`stoneley_dispersion_layered` and `flexural_dispersion_layered`
accept arbitrary `layers` tuples and raise `NotImplementedError`
when `len(layers) > 1`. G.0 widens the path:

* `_validate_borehole_layers_stacked(layers, a)` — adds inter-layer
  geometry checks on top of F's per-layer validation:
  - All `thickness > 0` (already in F).
  - Stack does not extend to negative radius
    (`a + sum(thicknesses) > 0`, trivially true).
  - Optional: warn if any individual layer has `kz · thickness > 30`
    at any test frequency, since that's where the propagator's
    `E(r_in)^{-1}` becomes ill-conditioned (Knopoff / delta-matrix
    territory; see G.a.5 substep block).
* `stoneley_dispersion_layered` retains the `len(layers) == 0` and
  `len(layers) == 1` dispatch (bit-equivalent to existing) and
  routes `len(layers) >= 2` through the new propagator path. The
  existing single-layer hand-coded F.1 path is preserved as the
  oracle and continues to handle `len(layers) == 1` directly.

**Tests** (5):

- `len(layers) == 2` no longer raises NotImplementedError.
- `len(layers) == 0` and `len(layers) == 1` paths bit-equivalent
  to pre-G behaviour (regression).
- BoreholeMode return-type contract for two-layer stacks.
- Validation rejection of zero-thickness layers in a multi-stack.
- Layer-order-matters: swapping `(casing, cement)` to
  `(cement, casing)` produces a different slowness curve (sanity
  check that the propagator chain respects ordering).

## G.a — math scaffolding (~200 lines comments-only)

Inline substep blocks in `cylindrical_solver.py` mirroring F.1.a /
H.a. Establishes the propagator-matrix derivation in cylindrical
geometry and pins sign conventions and Bessel-function selectors.

* **G.a.1** — State vector at azimuthal order n=0:
  ``v(r) = (u_r, u_z, σ_rr, σ_rz)^T``. Four components because
  the Helmholtz decomposition gives four Bessel-mode amplitudes
  per layer (P-up, P-down, SV-up, SV-down → I_0(p r), K_0(p r),
  I_1(s r), K_1(s r) for n=0 with the standard sign conventions).
* **G.a.2** — Mode-amplitude → state-vector matrix ``E(r)``:
  ``v(r) = E(r) c`` where ``c = (B_P^+, B_P^-, C_S^+, C_S^-)^T``
  collects the four amplitudes in the layer. Explicit form of
  E(r) for n=0 (4×4):
  ```
       [ p I_0(pr)−I_1(pr)/r,  −p K_0(pr)−K_1(pr)/r,  k_z I_1(sr),     k_z K_1(sr)]
       [ k_z I_0(pr),          k_z K_0(pr),          s I_0(sr),       −s K_0(sr) ]
       [ ...row for σ_rr...                                                    ]
       [ ...row for σ_rz...                                                    ]
  ```
  Substep block transcribes the four rows in full with the Lamé
  reduction `−λ k_P² + 2 μ p² = μ (2 k_z² − k_S²)` already
  validated in F's row builders.
* **G.a.3** — Layer propagator ``P_j = E_j(r_outer_j) E_j(r_inner_j)^{-1}``.
  Maps the state vector at the inner boundary of layer j to the
  outer boundary. Composition for a stack:
  ``v(r_N_outer) = P_N · ... · P_1 · v(r_0_inner)``.
* **G.a.4** — Boundary conditions and modal determinant:
  - Fluid side at ``r = a``: continuity of ``u_r`` and ``σ_rr``;
    vanishing of ``σ_rz`` on the formation/layer side. Encoded
    as a 3×4 matrix ``B_fluid`` extracting the three rows
    ``(u_r − u_r^{(f)}, σ_rr + P^{(f)}, σ_rz)`` from ``v(a)``;
    plus the 4×1 column for the fluid amplitude ``A``.
  - Formation half-space at ``r = b = a + Σ thickness``: only
    the decaying I_n / K_n combination survives; encoded as the
    F.1 row 4-7 BCs evaluated against ``v(b)`` with two
    formation amplitudes ``(B_form, C_form)``.
  - Modal determinant: blocks assembled into a single
    ``(3 + 1) × (4 + 2) = 4×6`` system per layer, with the
    propagator chain folding the per-layer 4×4 blocks into a
    single (3 + N·4 + 2) × (1 + N·4 + 2) matrix that reduces
    by the propagator chain to a final ``(3 + 4 + 2) − ?`` form.
    **TODO during implementation**: fix the exact final size
    (4×6 vs 6×6 vs 7×7 — depends on how the BC rows / amplitude
    columns count). Resolved at G.b.1.
* **G.a.5** — Numerical conditioning. For ``kz · thickness > ~ 30``
  the layer's I_0(p·r_outer) overflows while K_0(p·r_outer)
  underflows; ``E(r_inner)^{-1}`` becomes ill-conditioned. Two
  mitigations:
  - **Typical cased-hole geometries** (1 cm casing, 5 cm cement,
    sonic band 1-15 kHz): ``kz · thickness`` stays well below 1.
    The naive propagator works.
  - **Thick layers / high f**: defer the Knopoff / Kennett delta-
    matrix re-formulation (composes minor determinants instead of
    full propagators; Tang & Cheng 2004 sect. 7.2). Out of scope
    for G's first pass; substep block flags it.
* **G.a.6** — Single-layer collapse identity (G.b → F.1 oracle).
  When ``len(layers) == 1`` the propagator-matrix determinant
  must agree with F's `_modal_determinant_n0_layered` up to an
  overall scale factor (different intermediate factors). Pin the
  scaling in the substep block; the brentq root in `k_z` is
  identical.
* **G.a.7** — Self-check protocol: zero-layer collapse → unlayered;
  single-layer → F.1; identity propagator (`r_outer == r_inner`)
  → identity matrix. Each becomes a test in G.b / G.c.

## G.b — n=0 per-layer propagator helper (~120 lines + 6 tests)

```python
def _layer_propagator_n0(
    kz: float, omega: float,
    *, vp: float, vs: float, rho: float,
    r_inner: float, r_outer: float,
) -> np.ndarray:
    """Returns the 4x4 layer propagator P_j(r_outer | r_inner)
    mapping the (u_r, u_z, sigma_rr, sigma_rz) state vector from
    r_inner to r_outer within a uniform elastic layer."""
```

Internal builders:

```python
def _layer_e_matrix_n0(
    kz: float, omega: float,
    *, vp: float, vs: float, rho: float, r: float,
) -> np.ndarray:
    """4x4 mode-amplitude-to-state-vector matrix E(r)."""
```

Then `_layer_propagator_n0 = E(r_outer) @ inv(E(r_inner))`.

**Tests** (6):

- **Identity propagator**: `r_outer == r_inner` → `eye(4)` to
  floating-point precision.
- **Inverse pair**: `P(r2 | r1) @ P(r1 | r2) ≈ eye(4)` for any
  `r1`, `r2` in the bound regime. The two-step round-trip is the
  cleanest oracle for the inversion implementation.
- **Composition**: `P(r3 | r1) ≈ P(r3 | r2) @ P(r2 | r1)` for
  any intermediate `r2 ∈ (r1, r3)` (within a few ULPs). Validates
  that the propagator is genuinely a one-parameter group in the
  radial coordinate.
- **State-vector continuity**: pick an arbitrary amplitude vector
  `c`, compute `v(r1) = E(r1) @ c` and `v(r2) = P(r2 | r1) @ v(r1)`,
  verify `v(r2) ≈ E(r2) @ c` (the propagator does what it claims).
- **NaN propagation**: `kz` below the bound floor → at least one
  Bessel argument imaginary → propagator entries NaN (brentq-safe).
- **Per-element check at thin-layer limit**: in the limit
  `r_outer → r_inner + ε`, `P ≈ I + ε · M(r_inner)` where M is
  the radial-derivative matrix from the elastodynamic ODE
  system. Cross-checks E(r) construction against the underlying
  PDE.

## G.c — n=0 stacked modal determinant (~150 lines + 6 tests)

```python
def _modal_determinant_n0_cased(
    kz: float, omega: float,
    *, vp: float, vs: float, rho: float,
    vf: float, rho_f: float, a: float,
    layers: tuple[BoreholeLayer, ...],
) -> float:
    """Real-valued n=0 modal determinant for fluid + N annular
    layers + formation half-space. Returns a real scalar after
    the F.1-style row/column phase rescaling."""
```

Algorithm:
1. Compute layer radii: `r_0 = a; r_j = r_{j-1} + layers[j-1].thickness`.
2. Build per-layer propagators `P_j` for j=1..N via G.b.
3. Compose: `P_total = P_N @ ... @ P_2 @ P_1`.
4. Build `B_fluid` (3×4 row block extracting the three fluid-side
   BCs at r=a; reuses F.1.a.{1,2,3} symbolic forms).
5. Build `B_outer` (3×4 row block extracting the three half-space
   BCs at r=b=r_N; reuses F.1.b.{1,2,3} symbolic forms).
6. Assemble the (3 + 3) × (1 + 4 + 2) = 6×7... wait no. Final
   form: the 3 fluid BCs go on the left columns (1 fluid amp +
   2 formation amps adjusted), the 3 outer BCs go on the right.
   Concretely: stack
   `M = [[B_fluid_at_a · [I, P_total^{-1}]], [B_outer_at_b · [P_total, I]]]`
   into a square matrix, take determinant. **TODO at G.c.1**:
   fix the exact dimension and assembly. Most likely a 6×6 form
   that reduces correctly to F.1's 7×7 with one row/column
   redundancy that the propagator chain absorbs.
7. Apply the F.1 phase rescale (row × i for z-derivative-bearing
   rows; col × −i for SV columns) so the determinant is real.

**Tests** (6):

- **Zero-layer collapse**: `len(layers) == 0` → bit-equivalent to
  `_modal_determinant_n0` (within floating-point precision; same
  brentq root).
- **Single-layer collapse**: `len(layers) == 1` → root in `k_z`
  matches `_modal_determinant_n0_layered` to ``rtol=1e-10``. The
  determinants differ by an overall scale (different
  factorisation conventions); the root is invariant.
- **Identity-thin-layer**: `layers[0].thickness → 0` → root in
  `k_z` matches the unlayered solver. Continuity of the
  propagator chain across the trivial-layer limit.
- **Order matters**: swapping two distinct layers
  `(L_a, L_b) ↔ (L_b, L_a)` produces different slowness (the
  same physical materials at different radii give different
  Stoneley dispersion).
- **Bound-regime real**: in the slow-formation bound regime, the
  propagator-matrix determinant is real to ``imag/real < 1e-10``
  after the phase rescale.
- **NaN propagation**: outside the bound regime (kz below the
  fluid floor `omega/V_f`), the determinant returns NaN.

## G.d — n=0 public API + cement-bond regression (~80 lines + 6 tests)

Replaces the `len(layers) >= 2` `NotImplementedError` raise with a
brentq loop on `_modal_determinant_n0_cased`. Reuses the F.1
bracket helper `_stoneley_kz_bracket_layered` directly (the
bracket is set by the slowest shear in the stack and the fluid
floor — both well-defined for arbitrary layer counts).

**Tests** (6):

- **Two-layer regression**: typical casing + cement geometry
  produces a finite, smoothly-dispersive Stoneley curve across
  1-12 kHz.
- **Free-pipe contamination**: cement layer with `vs ≈ 0` (a
  fluid-like cement) admits the casing extensional mode as a
  near-degenerate root that contaminates the Stoneley. The
  test does NOT need to resolve the contamination cleanly (that
  requires the multi-root tracking from plan C); it just confirms
  that the Stoneley root still exists and is shifted by a known
  amount toward the casing-extensional speed.
- **Cement-bond synthetic**: well-bonded vs free-pipe scenarios
  produce distinct Stoneley curves at the same casing geometry,
  with the well-bonded slower (fluid-coupled into a softer
  composite layer). Direct test of the cement-bond logging
  signature.
- **Layer permutation**: swapping casing ↔ cement positions
  produces an unphysical-but-finite curve that differs from the
  physical (casing-inside) curve by a measurable amount.
- **Three-layer extension**: `(casing, cement, mudcake)` runs
  end-to-end and produces a finite slowness curve. No tight
  oracle; smoke + monotonic-frequency-step.
- **BoreholeMode contract**: `name = "Stoneley"`,
  `azimuthal_order = 0`, `freq` echoed.

## G.e — n=0 hardening (~80 lines + 4 tests)

Mirror of F.2.e / H.e: tighter validation against the propagator
chain's self-consistency.

* **Tang & Cheng 2004 fig 7.1**: digitised reference curve for a
  classical free-pipe vs well-bonded cement-bond synthetic.
  Match within 2-3 % across 1-10 kHz. The hardest of the G
  tests; defer to a digitised CSV in `tests/data/` if needed.
* **Multi-frequency det-at-root self-consistency**: at every
  brentq-converged `k_z` across a 6-point geomspaced band, the
  propagator-matrix determinant is at least 6 orders of
  magnitude smaller than its value 1 % off the root.
* **Two-layer-collapse-to-single-layer**: a casing of
  `thickness → 0` reduces to the cement-only single-layer
  solver answer.
* **Casing-only-collapse-to-unlayered**: a casing with formation
  properties + cement with formation properties reduces to the
  unlayered solver to ``rtol=1e-10`` (master-plan G validation
  bullet 1).

## G.f — Cross-cutting docs (~30 lines)

- Mark item G done in `docs/plans/cylindrical_biot.md`.
- Update module docstring scope to mention multi-layer cased-hole
  support.
- Commit pointers below.
- Note that n=1 / n=2 cased-hole counterparts are deferred to
  follow-up plans G' / G''.

### Landing commits (branch `claude/plan-cylindrical-biot-f-JgjGH`)

- `881efcd` — G.0  public-API foundation
  (`_validate_borehole_layers_stacked`, sharpened NIE messages)
- `154a4f3` — G.a  math scaffolding (state vector, E(r),
  propagator chain, BC bookkeeping, conditioning)
- `bad8650` — G.a  sigma_rz sign erratum
- `eabf884` — G.b.1  `_layer_e_matrix_n0` (4 per-element oracles
  vs F.1.b row builders)
- `123e634` — G.b.2  `_layer_propagator_n0` (round-trip via
  state-vector identity, composition group law)
- `dd3db3e` — G.c  `_modal_determinant_n0_cased` (N=1 numerical
  match with F.1 to `rtol=1e-10`)
- `9e6b1ed` — G.d  `stoneley_dispersion_layered` multi-layer
  brentq path; `_stoneley_kz_bracket_cased` helper
- `9df7a78` — G.e  hardening (multi-frequency det-at-root,
  layer-collapse oracles, cement-bond physics sanity)

## Total scope

~660 lines of solver code + ~400 lines of tests + ~27 tests,
distributed across ~6 mergeable PRs (G.0, G.a, G.b, G.c, G.d,
G.e + G.f bundled). Conservative estimate: 4-6 days of focused
work.

Risk concentrated in:
- **G.b inverse**: `inv(E(r_inner))` is the dense linear-algebra
  step; double-check conditioning and use `solve` rather than
  explicit `inv` where possible. Keep the round-trip oracle
  tight (1e-12).
- **G.c assembly dimension**: getting the 6×6 vs 7×7 final size
  right requires care. Strategy: build the explicit 7×7 matrix
  for `len(layers) == 1` first, verify against
  `_modal_determinant_n0_layered`, then generalise to N layers.
  Falling back to the explicit hand-coded form for the smallest
  case is the cleanest debug strategy.
- **G.d free-pipe contamination**: the casing extensional mode
  competes with the Stoneley near the cutoff. brentq with a
  fixed bracket may converge to the wrong root. Mitigation:
  bracket from below (slower than V_f) where only Stoneley
  exists, and verify the root sits in the slow-Stoneley window.
- **G.e Tang & Cheng fig 7.1**: digitising published figures
  is per-pixel work; budget a half-day for the CSV. The other
  three G.e tests are self-consistency / cross-validation,
  cheap to write.

## Deferred follow-ups (separate plans)

- **G' — n=1 cased-hole flexural** — ✅ DONE. Shipped via
  [`cylindrical_biot_G_prime.md`](cylindrical_biot_G_prime.md);
  the cased-hole flexural is the headline cement-bond / through-
  tubing evaluation tool.
- **G'' — n=2 cased-hole quadrupole** (~550 lines + 20 tests).
  Mirror of G' with 8×8 propagator blocks (stronger P / SV / SH
  coupling at n=2). LWD relevance.
- **Knopoff / Kennett delta-matrix** for thick-layer / high-f
  conditioning (~200 lines + 6 tests). Only needed if G's user
  base hits the kz·thickness > 30 regime; defer indefinitely.

## References

- Thomson, W. T. (1950). Transmission of elastic waves through
  a stratified solid medium. *J. Appl. Phys.* 21(2), 89-93.
- Haskell, N. A. (1953). The dispersion of surface waves on
  multilayered media. *Bull. Seismol. Soc. Am.* 43(1), 17-34.
- Schmitt, D. P., & Bouchon, M. (1985). Full-wave acoustic
  logging: synthetic microseismograms and frequency-wavenumber
  analysis. *Geophysics* 50(11), 1756-1778. Cylindrical-geometry
  propagator matrix for cased-hole synthetics.
- Tang, X. M., & Cheng, C. H. (2004). *Quantitative Borehole
  Acoustic Methods*. Elsevier. Ch. 7 (cased-hole and cement-
  bond logging) — fig 7.1 is the validation target for G.e.
- Knopoff, L. (1964). A matrix method for elastic wave problems.
  *Bull. Seismol. Soc. Am.* 54(1), 431-438. Delta-matrix
  formulation referenced in G.a.5 conditioning notes.
- Kennett, B. L. N. (1983). *Seismic Wave Propagation in
  Stratified Media*. Cambridge UP. Reflection / transmission
  matrix re-formulation (cleaner numerics for thick stacks).

## Execution order

1. **G.0** (foundation: widen the multi-layer dispatch) —
   anchors the floating-point regression test.
2. **G.a** (math scaffolding) — comments only; pins the
   propagator-matrix derivation and BC bookkeeping.
3. **G.b** (per-layer propagator) — pure plumbing with strong
   unit oracles (identity, round-trip, composition).
4. **G.c** (stacked determinant + zero/single-layer collapse)
   — smallest viable G product; the `len(layers) == 1` collapse
   to F.1 is the floating-point oracle.
5. **G.d** (public API + cement-bond regression) — first
   physically-meaningful G output; free-pipe / well-bonded
   distinction is the cement-bond logging signature.
6. **G.e** (hardening + Tang & Cheng fig 7.1) — full external
   validation.
7. **G.f** (docs) — close out plan G; flag G' / G'' follow-ups.
