# fwap — Claude Code project guide

`fwap` (Full-Waveform Acoustic Processing) is a NumPy/SciPy reference
implementation of Mari, Coppens, Gavin & Wicquart (1994), *Full Waveform
Acoustic Data Processing*, plus borehole-acoustic extensions (anisotropy,
attenuation, rock physics, Stoneley petrophysics, geomechanics, the
cylindrical-Biot modal solver, and LAS/DLIS/SEG-Y I/O).

Full chapter-to-module map and usage: @README.md.
Full contributor detail: @CONTRIBUTING.md.

## Environment
- Install (dev): `pip install -e ".[dev]"` (add `,docs` for the Sphinx build).
- Python: runtime target is 3.9+, but CI runs **3.11 and 3.12** — test against
  those. (3.9 is upstream-EOL and slated for removal in the next major release.)

## CI gate — all of these must pass before a change is "done"
Run them locally exactly as CI does (`.github/workflows/ci.yml`):
```bash
ruff check fwap/ tests/ scripts/
ruff format --check fwap/ tests/ scripts/
mypy fwap
python scripts/check_public_api.py      # public-API guard
pytest -x                               # bench suite auto-excluded
```
- Benchmarks run separately: `pytest tests/test_bench.py` (excluded from the
  default run via `addopts` in `pyproject.toml`).
- Notebook validation. The two **open-hole tutorials** are torch-free and run in
  the core `ci.yml` gate (they need `ipykernel` alongside `nbval`):
  `pytest --nbval-lax docs/notebooks/open_hole_processing.ipynb docs/notebooks/open_hole_petrophysics.ipynb`.
  The **`sonic_ml` tutorials** need torch and are validated in the `ml.yml` job:
  `pytest --nbval-lax docs/notebooks/sonic_ml_tutorial.ipynb docs/notebooks/cased_hole_tutorial.ipynb`.
  The solver-validation notebook is run on demand:
  `pytest --nbval-lax docs/notebooks/cylindrical_biot_validation.ipynb`.

## Conventions
- **Type hints**: PEP 604/585 (`int | None`, `tuple[int, int]`).
  `from __future__ import annotations` is in every module.
- **Docstrings**: NumPy style (Parameters/Returns/Notes/References); always
  state **units and array shapes**.
- **Lint/format**: ruff, line length 88, double quotes. Matplotlib is used
  only in the demos and plotting helpers (`fwap.demos`, `fwap._plotting`).
- **Tests**: add a test for every new function. Core algorithm modules also
  get an end-to-end `demo_*` path covered by `tests/test_demos.py`.
- **Cylindrical-solver tests live in six modules**, split by what they
  constrain: `test_solver_open_hole` (n=0/1/2 open hole), `test_solver_layered`
  (plan F), `test_solver_vti` (plan H), `test_solver_cased` (plans G/G'/G''),
  `test_solver_branches` (where a branch exists, stops, and survives a change
  of grid), `test_solver_figures` (ties to published curves). Media shared
  across the seams are in `tests/_solver_media.py`. Leaky-VTI work goes in
  `tests/test_anisotropy.py`.
- Private modules are `_`-prefixed (e.g. `fwap/_common.py`,
  `fwap/cylindrical_solver/_bessel.py`); the public surface is re-exported
  from `fwap/__init__.py`.

## ⚠️ Public-API changes happen in lockstep (the easy thing to get wrong)
Adding, removing, or renaming a public name requires editing **all three** in
the same PR, or `python scripts/check_public_api.py` fails CI:
1. the chapter-to-module map in `fwap/__init__.py`
2. the autosummary list in `docs/api.rst`
3. the `FROZEN_PUBLIC_API` tuple in `scripts/check_public_api.py`

## Pull requests
- Branch from `main`; keep diffs to one feature/fix.
- Update `CHANGELOG.md` under the `## [Unreleased]` header.
- Scope-widening (new domain, heavy dependency, GUI) → open an issue first.

## Decisions & Lessons
<!-- Append durable, generalizable lessons here — especially ones learned from
     a mistake. Keep each terse. Promote anything reusable out of chat into
     this section; capture it live with `#<note>`. -->
- Dependency upper bounds are intentional (guard against NumPy 2.0 / SciPy 2.x
  breaks). Bump only in a release commit after a smoke test.
- The perf suite (`tests/test_bench.py`) uses wall-clock budgets tuned on a
  dev laptop and is excluded from `pytest` by default — don't "fix" a local
  bench failure by editing budgets; it belongs in the dedicated CI bench job.
