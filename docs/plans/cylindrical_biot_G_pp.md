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

| Substep | Status | Lands at | Tests |
|---------|--------|----------|-------|
| G''.0 — public-API foundation | ✅ | `fwap/cylindrical_solver.py:4865` (`quadrupole_dispersion_layered`) | 6 / 6 |
| G''.a — math scaffolding (comments only) | ✅ | `fwap/cylindrical_solver.py:10518-10756` | n/a |
| G''.b.1 — `_layer_e_matrix_n2` | ✅ | `fwap/cylindrical_solver.py:10786` | 6 / 6 |
| G''.b.2 — `_layer_propagator_n2` | ✅ | `fwap/cylindrical_solver.py:10951` | 5 / 5 |
| G''.c — `_modal_determinant_n2_cased` | ⏳ | not yet shipped | 0 / 6 planned |
| G''.d — public-API hook + brentq loop | ⏳ | currently raises `NotImplementedError` at `fwap/cylindrical_solver.py:4961` | 0 / 6 planned |
| G''.e — n=2 hardening | ⏳ | not yet shipped | 0 / 4 planned |
| G''.f — cross-cutting docs | ⏳ | not yet shipped | n/a |

**Shipped so far:** G''.0 + G''.a + G''.b.1 + G''.b.2 — 17 of the
~33 planned tests, covering the public API surface
(`layers=()` dispatch + validation + future-NIE), the n=2 math
scaffolding, the per-layer `E_n2(r)` helper, and the per-layer
propagator. The `len(layers) >= 1` cased-hole path still raises
`NotImplementedError` until G''.c + G''.d land.

**Next up:** G''.c (`_modal_determinant_n2_cased`) replaces the NIE
raise with a real determinant; G''.d wires it into
`quadrupole_dispersion_layered` via brentq. G''.e adds hardening
(multi-layer collapse oracles + LWD cement-bond physics).

Plan G'' depends on the propagator-matrix scaffolding from plans
G (4x4 at n=0) and G' (6x6 at n=1); it inherits the same
six-component state vector and 10x10 final-form structure as G'.
The only structural difference is the Bessel-function index shift
``(I_{n-1}, I_n) -> (I_1, I_2)`` and the explicit ``n = 2``
factors that come out of the ``d_theta cos(n theta) = -n sin(n
theta)`` step (per `_modal_determinant_n2` substep).

## Predecessor mapping (G / G' equivalents)

Many G'' substeps inherit their structure unchanged from
plans G (n=0, 4x4) and G' (n=1, 6x6). Quick reference for
reviewers — each G'' substep below lists its direct
predecessor in [`cylindrical_biot_G.md`](cylindrical_biot_G.md)
and / or [`cylindrical_biot_G_prime.md`](cylindrical_biot_G_prime.md):

| G'' substep | n=0 (plan G)        | n=1 (plan G')       |
|-------------|---------------------|---------------------|
| G''.0       | G.0                 | G'.0                |
| G''.a.1     | G.a.1               | G'.a.1              |
| G''.a.2     | G.a.2 (4x4)         | G'.a.2 (6x6)        |
| G''.a.3     | G.a.3 (4x4 P)       | G'.a.3 (6x6 P)      |
| G''.a.4     | G.a.4 (7x7 final)   | G'.a.4 (10x10)      |
| G''.a.5     | G.a.5               | G'.a.5              |
| G''.a.6     | G.a.6               | G'.a.6              |
| G''.a.7     | G.a.7               | G'.a.7              |
| G''.b.1     | G.b.1               | G'.b.1              |
| G''.b.2     | G.b.2               | G'.b.2              |
| G''.c       | G.c                 | G'.c                |
| G''.d       | G.d                 | G'.d                |
| G''.e       | G.e                 | G'.e                |
| G''.f       | G.f                 | G'.f                |

The structural delta from G' is the Bessel-function index
shift ``(I_0, I_1) -> (I_1, I_2)`` and the explicit ``n = 2``
azimuthal factors (``2 n (n+1) = 12``, ``n^2 - 1 = 3``); the
state-vector layout, propagator group law, and 10x10 final
form are pixel-identical to G'. The structural delta from G
is the additional sin-sector (SH) coupling (4x4 -> 6x6 state
vector and 7x7 -> 10x10 final form).

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

## G''.0 — public-API foundation (~150 lines + 6 tests) ✅ shipped

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

## G''.a — math scaffolding (~300 lines comments-only) ✅ shipped

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
  Six rows: the six state-vector components ``v(r) = (u_r,
  u_z, u_theta, sigma_rr, sigma_rz, sigma_r_theta)``.

  **Bessel-derivative identities at n=2** (transcribed from
  the existing `_modal_determinant_n2`; the n=2 case is the
  ``nu = 2`` instance of ``d_r F_nu(x r) = (x/2) [F_{nu-1}(x r)
  + sigma F_{nu+1}(x r)] = x F_{nu-1}(x r) - (nu/r) F_nu(x r)``,
  with ``sigma = +1`` for ``F = I`` and ``sigma = -1`` for
  ``F = K``):

  * ``d_r I_2(p r) = p I_1(p r) - 2 I_2(p r) / r``
  * ``d_r K_2(p r) = -p K_1(p r) - 2 K_2(p r) / r``
  * ``(1/r) d_r [r I_2(s r)] = s I_1(s r) - I_2(s r) / r``
  * ``(1/r) d_r [r K_2(s r)] = -s K_1(s r) - K_2(s r) / r``

  The Bessel-index shift ``(I_1, I_2)`` (vs ``(I_0, I_1)`` at
  n=1) propagates through every entry; the ``-(n - 1) F_n / r``
  correction in ``(1/r) d_r [r F_n]`` becomes ``-F_2 / r`` at
  n=2 (vs no correction at n=1, since ``-(n - 1) = 0`` there).
  The same identities apply to the SV and SH columns by
  substituting ``p -> s``.

  **Explicit n=2 azimuthal factors** appearing in the stress
  entries (per `_modal_determinant_n2`; from the ``cos(n
  theta)`` / ``sin(n theta)`` derivative chain ``d_theta cos(n
  theta) = -n sin(n theta)``):

  * ``2 n (n + 1) = 12`` in the ``B_{I,K} F_2 / r^2`` term of
    sigma_rr (and in the SH column of sigma_r_theta).
  * ``n^2 - 1 = 3`` in the SV column of sigma_rz (multiplying
    the ``F_2(s r) / r^2`` term).
  * Standalone ``n = 2`` factor wherever a ``d_theta`` is
    consumed (sigma_r_theta cross-derivative, u_theta SV
    contribution).

  **E_n2(r) row map** (transcribed in
  `fwap/cylindrical_solver.py:10866-10930`; the F.2.a.5 phase
  rescale — row * i for ``u_z`` and ``sigma_rz``, col * -i for
  the SV columns ``C_I`` / ``C_K`` — is absorbed so all 36
  entries are real-valued in the bound regime):

  | Row | State component | B_I, B_K cols | C_I, C_K cols | D_I, D_K cols |
  |-----|-----------------|---------------|---------------|---------------|
  | 0 | ``u_r`` | ``±p F_1(p r) - 2 F_2(p r) / r`` | ``-k_z F_2(s r)`` | ``+2 F_2(s r) / r`` |
  | 1 | ``u_z`` | ``-k_z F_2(p r)`` | ``±s F_1(s r) - F_2(s r) / r`` | 0 (sparsity) |
  | 2 | ``u_theta`` | ``-2 F_2(p r) / r`` | 0 (sparsity) | ``∓s F_1(s r) + 2 F_2(s r) / r`` |
  | 3 | ``sigma_rr`` | ``mu [(2 k_z^2 - k_S^2) F_2 ∓ 2 p F_1 / r + 12 F_2 / r^2]`` | ``∓2 k_z mu (s F_1 ∓ 2 F_2 / r)`` | ``±4 mu (s F_1 / r ∓ 3 F_2 / r^2)`` |
  | 4 | ``sigma_rz`` | ``∓2 k_z mu (p F_1 ∓ 2 F_2 / r)`` | ``+mu (2 k_z^2 - k_S^2 + 3/r^2) F_2`` | ``-2 k_z mu F_2 / r`` |
  | 5 | ``sigma_r_theta`` | ``+4 mu (∓p F_1 / r + 3 F_2 / r^2)`` | ``+2 k_z mu F_2 / r`` | ``-mu [(s^2 + 12/r^2) F_2 ∓ 2 s F_1 / r]`` |

  The ``±`` / ``∓`` signs select the I-flavour (upper) or
  K-flavour (lower) entry within each two-column block;
  ``F_1``, ``F_2`` denote the appropriate ``I`` or ``K`` Bessel
  function; ``mu = rho * V_S^2`` and ``k_S = omega / V_S``.

  **Sparsity pattern** (same as n=1, pinned by G''.b.1 tests):

  * Row 1 (``u_z``) cols 4, 5 (``D_I``, ``D_K``): zero —
    the SH potential ``psi_z`` doesn't contribute to ``u_z``
    at any ``n >= 1``.
  * Row 2 (``u_theta``) cols 2, 3 (``C_I``, ``C_K``): zero —
    the SV potential ``psi_theta`` doesn't contribute to
    ``u_theta`` at any ``n >= 1``.

* **G''.a.3** — Layer propagator
  ``P_j = E_j(r_outer) E_j(r_inner)^{-1}`` (6x6). Composition
  ``P_total = P_N ... P_2 P_1`` -- identical to G'.a.3 with the
  Bessel-index shift baked into ``E_j``.
* **G''.a.4** — Boundary conditions and 10x10 modal determinant.
  Same BC structure as G'.a.4 (the n=2 change is purely in the
  Bessel-function index used inside ``E_n2``; the BC bookkeeping
  is structurally identical). Reduces at N=0 to the unlayered
  `_modal_determinant_n2` (4x4 form) after the propagator chain
  is bypassed.

  **Ten unknowns, ten BCs.** The system is square:

  * **1 fluid amplitude** ``A`` — the inviscid borehole-fluid
    acoustic-pressure scalar; couples to the wall through the
    radial-displacement and pressure-balance BCs at ``r = a``.
  * **6 innermost-layer amplitudes** ``c_1 = (B_I, B_K, C_I,
    C_K, D_I, D_K)`` — both I- and K-flavour P / SV / SH; the
    layer is finite-thickness so both flavours survive the
    decay floor.
  * **3 formation amplitudes** ``(B_form, C_form, D_form)`` —
    K-flavour only; the formation half-space is unbounded and
    the I-flavour Bessel functions diverge as ``r -> infinity``,
    so the radiation condition kills the I-flavour columns.

  **Column packing** of the 10x10 final form (matches the F.2
  / G' convention; column index 0..9):

  ``[A | B_I, B_K, C_I, C_K | B_form, C_form | D_I, D_K | D_form]``

  **Row layout — 4 BCs at r=a** (fluid-innermost-layer
  interface; the inviscid fluid carries only ``u_r`` and a
  pressure scalar, so only the radial-displacement and stress
  rows of the layer participate, with the layer side entered
  negated where the BC is a difference):

  | BC | Equation | E_n2 row used | Sector |
  |----|----------|---------------|--------|
  | BC1 | ``u_r^(f) = u_r^(m)`` | row 0 (``u_r``), layer side negated | cos |
  | BC2 | ``sigma_rr^(m) + P^(f) = 0`` | row 3 (``sigma_rr``), layer side negated | cos |
  | BC3 | ``sigma_r_theta^(m) = 0`` | row 5 (``sigma_r_theta``), layer positive | sin |
  | BC4 | ``sigma_rz^(m) = 0`` | row 4 (``sigma_rz``), layer positive | cos |

  No BC on ``u_z^(m)`` (row 1) or ``u_theta^(m)`` (row 2) at
  ``r = a``: the inviscid fluid cannot grip the wall
  tangentially. Two of the six layer state-vector rows at
  ``r = a`` are therefore unused.

  **Row layout — 6 BCs at r=b** (outermost-layer / formation
  half-space interface; full state-vector continuity, written
  as ``layer - formation = 0``):

  | BC  | Equation                       | E_n2 row used                     |
  |-----|--------------------------------|-----------------------------------|
  | BC5 | ``u_r^(m) = u_r^(form)``       | row 0 (``u_r``)                   |
  | BC6 | ``u_theta^(m) = u_theta^(form)`` | row 2 (``u_theta``)             |
  | BC7 | ``u_z^(m) = u_z^(form)``       | row 1 (``u_z``)                   |
  | BC8 | ``sigma_rr^(m) = sigma_rr^(form)`` | row 3 (``sigma_rr``)          |
  | BC9 | ``sigma_r_theta^(m) = sigma_r_theta^(form)`` | row 5 (``sigma_r_theta``) |
  | BC10 | ``sigma_rz^(m) = sigma_rz^(form)`` | row 4 (``sigma_rz``)          |

  **How each unknown enters the matrix.** For an ``N``-layer
  stack the layer-side state vector at ``r = b`` is
  ``v_layer(b) = P_total @ E_n2(a) c_1`` with
  ``P_total = P_N ... P_2 P_1`` (G''.a.3). The matrix entries
  by column:

  * **Col 0 (``A``)** — populates BC1 (fluid ``u_r``
    contribution) and BC2 (fluid pressure). Zero at all r=b
    rows (the fluid lives only at ``r <= a``).
  * **Cols 1-4 (``B_I, B_K, C_I, C_K``)** — innermost-layer
    P / SV amplitudes. At r=a the relevant rows of ``E_n2(a)``
    fill BC1-BC4 with the layer sign convention from the
    table above. At r=b the same columns are picked off
    ``P_total @ E_n2(a)`` and fill BC5-BC10.
  * **Cols 7-8 (``D_I, D_K``)** — same as cols 1-4 but for
    the SH amplitudes. Fill BC1-BC4 at r=a (rows 0, 3, 4, 5
    of ``E_n2`` are all non-zero in the D columns) and
    BC5-BC10 at r=b.
  * **Cols 5, 6, 9 (``B_form, C_form, D_form``)** — formation
    K-only amplitudes. Zero at all r=a rows (BC1-BC4); at r=b
    they fill BC5-BC10 with ``-E_form_K(b)`` (the minus sign
    comes from the ``layer - formation`` convention).

  This pins the entire ``M[10, 10]`` sparsity pattern: cols
  5, 6, 9 are zero in BC1-BC4; col 0 is zero in BC5-BC10. The
  determinant is a polynomial-style mix of I/K Bessel
  functions evaluated at ``p a``, ``s a``, ``p b``, ``s b``,
  and ``p_form b``, ``s_form b``.

  **Reduction at N=0** — with no annular layers,
  ``P_total = I_6`` and ``v_layer = E_n2(a) c_1``. The
  outermost-layer / formation interface collapses onto the
  fluid / formation interface at ``r = a``; cols 1-4 and 7-8
  drop out (the innermost layer is the formation), only cols
  5, 6, 9 survive, and the 10x10 form reduces to the unlayered
  4x4 ``_modal_determinant_n2``.

  **G''.c reference implementation point**: the row/col
  bookkeeping above is the assembly target; see the source-
  side comment block at
  `fwap/cylindrical_solver.py:10653-10687` for the exact
  state-row-to-BC-row map G''.c will transcribe.
* **G''.a.5** — Numerical conditioning. Same disparate-magnitude
  story (``cond(E_n2) ~ mu ~ 1e10``); the state-vector form
  for round-trip oracles in G''.b.2 mitigates the
  raw-matrix-equality footgun (G.b.2 / G'.b.2 lesson).

  **n=2 specific concern**: the ``I_2(p r)`` Bessel function
  grows as ``(p r / 2)^2 / 2`` at small ``p r`` and as
  ``e^{p r} / sqrt(2 pi p r)`` at large ``p r``. Small-argument
  behaviour makes the I-flavour columns disproportionately
  small near ``r = a`` while the K-flavour columns blow up;
  large-argument behaviour drives an exponential split between
  I and K columns. Both effects degrade ``cond(E_n2)`` and
  worsen with ``n`` (the n=2 small-argument coefficient
  ``(p r / 2)^2 / 2`` is two orders below the n=0 ``1`` and
  one below the n=1 ``(p r) / 2``).

  **Quantification (slow-formation test geometry**, ``vp =
  3000`` m/s, ``vs = 1600`` m/s, ``rho = 2200`` kg/m^3, ``r =
  a = 0.10`` m, ``kz = 1.05 * omega / vs`` -- bound-mode just
  above the floor):

  | f (kHz) | ``cond(E_n0)`` | ``cond(E_n1)`` | ``cond(E_n2)`` |
  |---------|----------------|----------------|----------------|
  | 1       | ``1.9e+13``    | ``4.3e+14``    | ``2.7e+18``    |
  | 3       | ``2.8e+12``    | ``8.7e+12``    | ``2.3e+15``    |
  | 5       | ``1.8e+12``    | ``2.7e+12``    | ``1.3e+14``    |
  | 8       | ``3.0e+13``    | ``2.5e+13``    | ``1.6e+13``    |
  | 12      | ``8.9e+14``    | ``7.5e+14``    | ``4.7e+14``    |
  | 15      | ---            | ---            | ``5.7e+15``    |
  | 20      | ``4.9e+17``    | ``4.3e+17``    | ``3.1e+17``    |

  Two regimes show up:

  * **Below ~3 kHz** the n=2 small-``p r`` blowup dominates --
    ``cond(E_n2)`` is 2-4 orders worse than n=0 / n=1. This
    is the regime the existing scope note flagged.
  * **Above ~12 kHz** the large-``p r`` exponential split
    dominates and all three orders blow up together.

  **Practical implication for G''.b / G''.c.** Double precision
  carries ~16 decimal digits (``eps ~ 2e-16``). The LWD
  quadrupole band of interest is ~5-20 kHz; in that window
  ``cond(E_n2)`` stays at or below ~1e16 except for very low f
  (rare in LWD) and the very top of the band. A raw
  ``np.linalg.solve(E_inner, E_outer)`` round-trip oracle
  written naively against ``E_n2`` would lose 13-15 digits at
  20 kHz -- below the ~``rtol=1e-10`` pinning that G / G' use.
  The state-vector formulation in G''.b.2 (apply
  ``v_outer = P @ v_inner`` and check ``v_outer`` directly
  rather than equating raw 6x6 matrices) sidesteps the worst
  cases by operating in the well-conditioned coordinate
  system. Recommended ``rtol`` budget for G''.b.2 round-trip
  oracles: ``rtol=1e-8`` at <= 12 kHz, relax to ``rtol=1e-6``
  in 12-20 kHz (vs ``rtol=1e-10`` for n=0 / n=1 across the
  same band).

  Numbers above were generated by evaluating
  ``_layer_e_matrix_n2`` directly; they are reproducible from
  ``fwap/cylindrical_solver.py:10786`` and serve as the
  conditioning oracle for G''.b.2.

* **G''.a.6** — Layer = formation collapse identity (G''.c
  oracle). With the lack of an F.3-equivalent hand-coded form,
  the strongest collapse oracle is:

  ``layer.{vp, vs, rho} == formation.{vp, vs, rho}``
  ``=>`` cased-hole ``_modal_determinant_n2_cased`` brentq
  root in ``k_z`` matches the unlayered
  ``_modal_determinant_n2`` brentq root in ``k_z``.

  **Why the roots match exactly.** With ``layer = formation``,
  ``E_layer(r) = E_form(r)`` for all ``r``, so the per-layer
  propagator ``P_layer = E_form(b) E_form(a)^{-1}`` and the
  layer-side state vector at ``r = b`` simplify:

  ``v_layer(b) = P_layer @ E_form(a) @ c_layer
              = E_form(b) E_form(a)^{-1} E_form(a) c_layer
              = E_form(b) c_layer.``

  The six BCs at ``r = b`` (BC5-BC10, full state-vector
  continuity) then read

  ``E_form(b) c_layer - E_form_K(b) c_form_K = 0``,

  where ``c_form_K = (B_form, C_form, D_form)``. Splitting
  the layer amplitudes by Bessel flavour
  ``c_layer = (c_layer_I, c_layer_K)`` and using the
  block factorisation ``E_form(b) = [E_form_I(b) |
  E_form_K(b)]``,

  ``E_form_I(b) c_layer_I + E_form_K(b) (c_layer_K -
  c_form_K) = 0``.

  ``E_form(b)`` is the full 6x6 ``E_n2`` and is non-singular
  in the bound regime, so the unique solution is

  ``c_layer_I = 0`` and ``c_layer_K = c_form_K``.

  Substituting back into the four ``r = a`` BCs (BC1-BC4)
  recovers the unlayered system: the layer-side state vector
  at ``r = a`` becomes ``E_form_K(a) c_form_K`` -- identical
  to ``v_form(a)`` in the unlayered formulation, with the
  same column index packing for ``A`` and ``c_form_K``. So
  the four ``r = a`` rows of the 10x10 reduce to the four
  rows of the unlayered 4x4 ``_modal_determinant_n2`` system
  on ``(A, B_form, C_form, D_form)``.

  Determinant relation. The six BC5-BC10 rows decouple
  cleanly from the ``r = a`` block once the
  ``c_layer_I = 0`` / ``c_layer_K = c_form_K`` substitution
  is performed; their contribution to ``det(M_10)`` is the
  determinant of an explicit 6x6 sub-block built from
  ``E_form(b)``, independent of ``A``. Therefore

  ``det(M_10)(k_z) = det(E_form(b)) * det(M_4)(k_z)``

  up to a sign from the column reordering. ``det(E_form(b))``
  is non-zero in the bound regime (a precondition of
  G''.b.1's "non-singular E" test), so the ``k_z`` zeros of
  ``det(M_10)`` are exactly the zeros of ``det(M_4)``. The
  brentq root match is bit-for-bit, modulo the brentq
  bracket and tolerance (the determinant magnitudes differ
  by the ``det(E_form(b))`` factor; the *roots* are
  invariant).

  This is the master-plan G''-equivalent of the G validation
  bullet 1 (two-formation-layers collapse to unlayered). The
  algebra carries over directly to ``N >= 2`` formation-equal
  layers: each ``P_j`` is built from the same ``E_form``, and
  the inner / outer ``E_form`` factors telescope across layer
  boundaries (``r_outer_{j-1} = r_inner_j``), leaving
  ``P_total = E_form(b) E_form(a)^{-1}`` regardless of how
  many formation-equal layers are stacked. The
  ``v_layer(b) = E_form(b) c_layer`` step then reduces to the
  same ``c_layer_I = 0``, ``c_layer_K = c_form_K`` solution.

* **G''.a.7** — Self-check protocol:
  * (a) Identity propagator: ``r_outer = r_inner``.
  * (b) Composition group law.
  * (c) State-vector continuity.
  * (d) N=0 dispatch to `quadrupole_dispersion`.
  * (e) Layer = formation N=1 collapse to unlayered root.
  * (f) Order-matters at N=2.
  * (g) LWD-quadrupole physics smoke: cement-bond signature
    distinct from the unlayered case.

## G''.b — n=2 per-layer propagator (~250 lines + 11 tests) ✅ shipped

Two sub-units mirroring G'.b.

### G''.b.1 — `_layer_e_matrix_n2` (~180 lines + 6 tests) ✅ shipped

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

### G''.b.2 — `_layer_propagator_n2` (~70 lines + 5 tests) ✅ shipped

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

## G''.c — n=2 stacked modal determinant (~300 lines + 6 tests) ⏳ pending

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

## G''.d — n=2 public-API hook (~80 lines + 6 tests) ⏳ pending

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

## G''.e — n=2 hardening (~80 lines + 4 tests) ⏳ pending

Mirror of G.e / G'.e for the quadrupole cased-hole solver.

* **Multi-frequency det-at-root self-consistency** at every
  brentq-converged kz across 8-25 kHz (LWD band).
* **Thin-inner-layer collapse** to N=1 outer-only.
* **Two-formation-layers collapse to unlayered** (master-plan
  G'' validation bullet, ``rtol=1e-6``).
* **LWD-quadrupole cement-bond physics**: stiffer cement gives
  a slowness shift in a measurable direction relative to softer
  cement at the same casing. Direct test of the LWD cement-bond
  signature.

  **Predicted direction.** In a slow formation (``vs_form <
  vf_borehole``) the LWD-quadrupole low-frequency asymptote is
  the formation shear slowness; the cement layer mediates
  acoustic coupling between casing and formation. Stiffer
  cement (higher ``vs_cement``) couples the casing more
  effectively to the formation, so the guided mode samples
  more of the formation and its slowness moves *toward* the
  formation-shear asymptote. Softer cement (lower
  ``vs_cement``) acoustically decouples the casing from the
  formation; the mode becomes more sensitive to the
  faster-shear casing + fluid system and its slowness moves
  *away* from the formation-shear asymptote (i.e., toward the
  casing-shear-dominated regime, lower slowness).

  Concretely, for a slow-sandstone formation
  (``vs_form ~ 1600`` m/s, slowness ~625 us/m) at ~10-15 kHz:
  cement at ``vs_cement ~ 1700`` m/s ("good cement") should
  give a quadrupole slowness within a few % of the formation
  asymptote, while ``vs_cement ~ 800`` m/s ("light cement /
  partial bond") should give a noticeably lower slowness
  (faster mode). The G''.e test pins the *sign* of
  ``slow(stiff) - slow(soft) > 0`` and a magnitude window of a
  few percent; the exact magnitude is the empirical
  observation from the test, not a closed-form prediction.

## G''.f — Cross-cutting docs (~30 lines) ⏳ pending

- Mark plan G'' done in `docs/plans/cylindrical_biot.md`
  (update G section status to "✅ DONE (n=0, n=1, n=2)";
  refresh status snapshot).
- Update plan G doc to point at G'' as completed.
- Update plan G' doc similarly.
- Update module docstring scope to mention multi-layer cased-
  hole quadrupole support.
- Commit pointers in this plan doc.

## Total scope

Originally estimated: ~1100 lines of solver code + ~700 lines
of tests + ~33 tests, distributed across ~7 mergeable PRs
(G''.0; G''.a; G''.b.1; G''.b.2; G''.c; G''.d; G''.e + G''.f
bundled). Conservative estimate: 5-7 days of focused work
(similar to G').

**Shipped (G''.0 + G''.a + G''.b.1 + G''.b.2):** 17 of the
~33 planned tests; 4 of the 7 PRs. **Remaining (G''.c +
G''.d + G''.e + G''.f bundled):** ~3 PRs, ~16 tests, 2-3
days estimated based on G' / G actuals.

Risk concentrated in:

- **No F.3-equivalent per-element oracle for G''.b.1**. The
  six-test suite pins the matrix structurally (sparsity,
  finiteness, Bessel-scaling, layer=formation cross-check) but
  not entry-by-entry as G/G' did. Mitigation: the G''.a.2 row
  map (cross-checked numerically against
  ``_layer_e_matrix_n2``) and the G''.a.6 collapse-algebra
  proof (``c_layer_I = 0``, ``c_layer_K = c_form_K``) give
  paper-side oracles G/G' lacked; G''.c / G''.d / G''.e add
  determinant-level checks. Optional follow-up: a F.3 plan
  that hand-codes the single-extra-layer 10x10 with per-row
  builders, then retrofits G''.b.1 with per-element oracles
  (~600 lines + 25 tests).
- **n=2 Bessel-function scaling at low frequencies.**
  Quantified in G''.a.5: ``cond(E_n2)`` reaches ~3e18 at 1
  kHz vs ~1e13 for ``cond(E_n0)`` (small-``p r`` blowup), and
  ~3e17 at 20 kHz across all orders (large-``p r``
  exponential split). The state-vector round-trip oracle in
  G''.b.2 sidesteps the worst cases by working in the
  well-conditioned coordinate system; the G''.b.2 ``rtol``
  budget recommended in G''.a.5 is ``1e-8`` at <= 12 kHz and
  ``1e-6`` at 12-20 kHz.
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

Done:

1. ✅ **G''.0** (foundation: introduce
   `quadrupole_dispersion_layered` API surface) -- anchors the
   floating-point regression test (`layers=()` ≡ unlayered).
2. ✅ **G''.a** (math scaffolding) -- comments only; pins the
   n=2 ansatz, Bessel-index shift, and 6x6 E_n2(r)
   transcription.
3. ✅ **G''.b.1** (E_n2(r) helper) -- structural oracles plus
   layer=formation cross-check vs unlayered `_modal_determinant_n2`.
4. ✅ **G''.b.2** (propagator) -- group-law oracles via
   state-vector identities (mirrors G.b.2 / G'.b.2).

Remaining:

5. ⏳ **G''.c** (stacked determinant) -- N=0 dispatch +
   layer=formation N=1 root match + multi-layer collapse
   oracles. Implements the 10x10 assembly per G''.a.4 and
   verifies the determinant relation
   ``det(M_10) = det(E_form(b)) * det(M_4)`` from G''.a.6.
6. ⏳ **G''.d** (public-API hook + multi-layer regression) --
   first physically-meaningful G'' output. Replaces the
   `NotImplementedError` raise in
   `quadrupole_dispersion_layered` with a brentq loop.
7. ⏳ **G''.e** (hardening + LWD cement-bond physics --
   directional prediction in the plan, sign pinned by test).
8. ⏳ **G''.f** (docs).
