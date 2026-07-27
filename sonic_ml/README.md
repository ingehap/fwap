# sonic_ml

Machine-learning surrogate & inverse models for
[`fwap`](../README.md) borehole-acoustic dispersion — the ML layer tracked in
issue #22.

`sonic_ml` is an **in-repo sibling package**: it lives alongside `fwap/` but has
its own `pyproject.toml` and is *not* part of the `fwap` distribution, so the
pure-NumPy/SciPy physics core never takes on an ML dependency. Its sole input is
the `.npz` produced by `scripts/gen_surrogate_dataset.py`, versioned by the
`schema_version` key.

## Install

`sonic_ml` is not published to an index; install it editable from this
checkout. Install the core first so its `fwap` dependency is satisfied locally,
and use the CPU torch wheel:

```bash
pip install -e ".[dev]"                                    # fwap core + dev tools
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e "./sonic_ml[dev]"
```

The pure-NumPy spine (loader, split, normalize, mask, provenance) can be
developed without torch via `pip install -e ./sonic_ml --no-deps`; only the
`determinism` torch hooks and the (later) models require it.

> A `fwap[ml]` install alias is intentionally **not** wired yet: it would only
> resolve once `sonic_ml` is published to an index. Use the editable install
> above.

## What's here (M0 — pre-model spine)

| Module | Purpose |
|---|---|
| `loader` | Load the `.npz` (`allow_pickle=False`), assert `schema_version`, read `N`/`M`/`F` from metadata |
| `gen_shim` | Bridge to the core generator for custom modes/geometry/priors |
| `provenance` | fwap version + git SHA + config + content hash, JSON sidecar |
| `split` | Regime-stratified (slow/fast) train/val/test split with stored indices |
| `normalize` | Standardizer with a zero-variance guard (drops constant `vf`/`rho_f`) |
| `mask` | Finite-mask + authoritative `mode_in_gather` presence + imbalance weights |
| `determinism` | Seed/thread pinning for reproducible training (torch-guarded) |

## Quick start

```python
from sonic_ml import gen_shim, load_npz, stratified_split, Standardizer

gen_shim.build_npz("demo.npz", n=64, seed=0)      # generate via the fwap engine
bundle = load_npz("demo.npz")                      # -> DatasetBundle
split = stratified_split(bundle, seed=0)           # regime-balanced partition
std = Standardizer.fit(bundle.params, bundle.param_names)
z = std.transform(bundle.params)                   # constant vf/rho_f dropped
```

## CI

A non-required `ml.yml` workflow installs the core, a CPU torch wheel, and
`sonic_ml`, then runs `ruff`, `mypy`, and `pytest` scoped to this package. It is
kept off the core gate so an ML/torch failure can never block a `fwap` PR.
