# Contributing to fwap

Thanks for your interest. This document is short; if something is
unclear, open an issue asking for clarification.

## Quick start

```bash
git clone https://github.com/ingehap/B_Mari_Full-Waveform-Acoustic-Data-Processing
cd B_Mari_Full-Waveform-Acoustic-Data-Processing
pip install -e ".[dev,docs]"
pre-commit install          # optional but recommended
pytest                      # main test suite (excludes benchmarks)
pytest tests/test_bench.py  # perf benches, separately
```

## Scope

`fwap` is a reference implementation of the algorithms in

> Mari, J.-L., Coppens, F., Gavin, P., & Wicquart, E. (1994).
> *Full Waveform Acoustic Data Processing.* Editions Technip.
> ISBN 978-2-7108-0664-6.

plus a few additions (LAS/SEG-Y I/O, rock physics, cross-dipole
anisotropy, Q attenuation). Contributions that stay inside that scope
(new estimators, more-faithful physics models, better synthetics,
improved docs) are welcome. Contributions that would substantially
widen the scope -- e.g. a whole new seismic domain, adding a heavy
dependency, or introducing GUI code -- should be discussed in an
issue before you start writing.

## Coding conventions

- **NumPy 1.22+, SciPy 1.8+, Python 3.9+.** Matplotlib is only used
  in :mod:`fwap.demos` and :mod:`fwap.plotting`.

  Python 3.9 reached upstream end-of-life in October 2025; the next
  major release of `fwap` will drop it from the support matrix.
  New PRs may use 3.10+-only syntax behind
  `from __future__ import annotations` (already in every module),
  but runtime branches that depend on 3.10+ features should be
  guarded with `sys.version_info` until 3.9 is removed from CI.
- **Type hints**: PEP 604 / PEP 585 style (`int | None`,
  `tuple[int, int]`). `from __future__ import annotations` is in
  every module.
- **Docstrings**: NumPy style (Parameters / Returns / Notes /
  References). Every public function and dataclass should have
  units and array shapes where applicable.
- **Tests**: add a test for every new function. The core algorithm
  modules should also have one end-to-end `demo_*` invocation
  covered by `tests/test_demos.py`.
- **Lint**: `ruff check fwap/ tests/` must pass. `pre-commit install`
  hooks this in automatically.

## Pull requests

- Branch from `main`.
- Keep diffs focused -- one feature or fix per PR.
- Update `CHANGELOG.md` under the `## [Unreleased]` header.
- If you add a new public API, update `fwap/__init__.py`'s
  chapter-to-module map and add it to the autosummary list in
  `docs/api.rst`.
- CI runs pytest on Linux / macOS / Windows + Python 3.9 / 3.11 /
  3.12 / 3.13 and builds the docs. All jobs must go green.

## Bug reports and feature requests

Open a GitHub issue. The two templates (`Bug report` and
`Feature request`) nudge you through the information maintainers
usually need.

## Security

See `SECURITY.md`.

## License

By contributing you agree that your contributions will be licensed
under the terms of `LICENSE` (MIT).
