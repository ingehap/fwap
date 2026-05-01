# Repository Improvements

A prioritised list of improvements for the `fwap` repository. Items are
grouped by theme; within each group the highest-leverage items come first.

## 1. Fix broken promises in the README / CONTRIBUTING

These are user-visible claims that don't match the repo's current state.

- **CI badge points to a workflow that does not exist.** `README.md`
  links `.github/workflows/ci.yml`, but there is no `.github/`
  directory. Either add the workflow (see §2) or remove the badge.
- **codecov badge with no CI to publish coverage.** Same root cause —
  remove or implement.
- **`.readthedocs.yaml` is advertised but missing.** README says "A
  `.readthedocs.yaml` is included so the repository can be connected
  to ReadTheDocs without further configuration." Add the file or
  delete the sentence.
- **`docs/notebooks/workflow.ipynb` is referenced but missing.** Both
  the README and the `[docs]` extra (which installs `myst-nb` +
  `ipykernel`) imply it exists. Either add the notebook or drop the
  reference and the unused dependencies.
- **`CONTRIBUTING.md` installs non-existent extras.** It instructs
  `pip install -e ".[dev,io,segy,docs]"`, but `pyproject.toml` only
  defines `dev` and `docs`. Decide whether to split I/O deps into
  optional extras or fix the docs.

## 2. Continuous integration

- **Add a GitHub Actions workflow** (`.github/workflows/ci.yml`) that
  runs on push + PR with a Python matrix (3.10 / 3.11 / 3.12 /
  3.13), executing: `ruff check`, `ruff format --check`, `mypy`,
  `pytest`, and `sphinx-build -W`.
- **Add a separate `bench.yml`** that runs `pytest tests/test_bench.py`
  on a schedule (and on demand) so the perf-regression suite that
  `pyproject.toml` deliberately excludes from `pytest` actually runs
  somewhere.
- **Publish coverage to Codecov** so the badge means something, or
  remove the badge.
- **Add Dependabot or Renovate** to keep `numpy`/`scipy`/`segyio` upper
  bounds (and the pre-commit `ruff` rev) from rotting silently.

## 3. Project metadata & supported Python versions

- **Drop Python 3.9.** It is past EOL (Oct 2025) and forces
  `from __future__ import annotations` and `Union[…]` workarounds.
  Bump `requires-python = ">=3.10"`, retarget mypy/ruff to `py310`,
  and modernise type hints (`X | None`, PEP 604/585 throughout).
- **Add `Programming Language :: Python :: 3.10/3.11/3.12/3.13`
  classifiers** (currently only the generic "Python :: 3" classifier
  is listed).
- **Pin or document the build backend's behaviour for `package-data`.**
  The current setuptools build picks up `py.typed` correctly but a
  release sanity check (`python -m build && twine check dist/*`) is
  not part of CI.
- **Add `project.urls`** (Homepage, Documentation, Issues, Changelog)
  to make the PyPI sidebar useful when the package is published.

## 4. Code organisation

- **Split `fwap/cylindrical_solver.py` (13 236 lines).** A single
  module that large hurts navigation, review, and import-time. Suggested
  decomposition: `cylindrical_solver/` package with submodules for
  isotropic n=0, isotropic n=1, VTI n=0, VTI n=1, layered/propagator
  matrix, and shared root-finding utilities. Keep the public API
  re-exported from `cylindrical_solver/__init__.py` so call sites
  don't move.
- **Several other large modules deserve a second look** (`picker.py`
  2266, `geomechanics.py` 2189, `anisotropy.py` 1857, `demos.py`
  1550, `rockphysics.py` 1375). Audit for dead code paths and
  candidates for extraction into focused submodules.
- **Trim the per-file ruff ignore list in `pyproject.toml`.** Each
  ignore is a small invariant violation (module-level code between
  imports, `E402` waivers); fixing them once is cheaper than carrying
  the exception forever.

## 5. Type checking

- **Replace global `ignore_missing_imports = true` with per-module
  overrides.** The config's own comment already flags this as
  technical debt. Restrict the ignore to `numpy.*`, `scipy.*`,
  `matplotlib.*`, `lasio.*`, `dlisio.*`, `segyio.*`, then turn on
  `disallow_untyped_defs` for `fwap.*`.
- **Run mypy in CI** (it isn't run today because there's no CI).

## 6. Pre-commit hygiene

- **Add the pre-commit-hooks essentials** (`trailing-whitespace`,
  `end-of-file-fixer`, `check-yaml`, `check-toml`, `check-added-large-files`,
  `mixed-line-ending`).
- **Add a `mypy` hook** so type errors show up locally before CI.
- **Bump `ruff-pre-commit` from v0.6.9** (≈ a year old) and let
  Dependabot keep it current.

## 7. `.gitignore`

- Add: `.venv/`, `venv/`, `.env`, `.mypy_cache/`, `.ruff_cache/`,
  `.coverage`, `coverage.xml`, `htmlcov/`, `.tox/`, `.idea/`,
  `.vscode/`, `.DS_Store`, `figures/` (the demo output directory the
  README tells users to generate), and `*.ipynb_checkpoints/` once
  notebooks land.

## 8. Repository contents

- **Move `ideas/` and `docs/*.pdf` out of the repo or into Git LFS.**
  ~800 KB of PDFs and proprietary `.docx` reference notes (potentially
  copyrighted) are checked in as binaries; every clone pays for them
  forever, and the PDFs appear to be Sphinx build artifacts that
  should be generated, not committed.
- **Confirm the `ideas/*.docx` files are licence-clean** before keeping
  them in a public MIT-licensed repo; if they're personal notes from
  copyrighted books, they should not ship in a public repo.
- **Replace `docs/plans/*.md`** (internal design notes) with either a
  `docs/dev/` section explicitly marked as developer-only or move
  them out of the published Sphinx tree.

## 9. Community files

- **Add `CODE_OF_CONDUCT.md`** (Contributor Covenant 2.1 is the
  standard one-liner reference).
- **Add `.github/ISSUE_TEMPLATE/`** with bug-report and
  feature-request forms; the CONTRIBUTING file already steers users
  to "open an issue", so giving them structure costs nothing.
- **Add `.github/PULL_REQUEST_TEMPLATE.md`** with a checklist
  matching the CONTRIBUTING expectations (tests pass, docs updated,
  changelog entry).
- **Add a `FUNDING.yml`** if the maintainers want sponsorship; trivial
  and easy to remove.

## 10. Testing

- **Add a coverage gate** (e.g. fail-under 85%) so `pytest-cov` —
  already a dev dep — does something useful.
- **Run hypothesis examples in CI with a fixed deadline** so flaky
  property tests fail loudly instead of timing out locally.
- **Document how to regenerate any "golden" arrays** the tests rely
  on; a quick scan suggests several tests compare against numeric
  literals whose provenance isn't recorded.

## 11. Documentation

- **Build the Sphinx docs in CI with `-W` (warnings as errors).** The
  README claims this happens already; once §2 lands, wire it up.
- **Generate the API reference with `sphinx-autogen` / `autosummary`
  templates** rather than hand-maintained `api.rst` entries (if any
  drift exists between `__all__` and the API page, this prevents it).
- **Add a "Citing fwap" snippet** that points at `CITATION.cff` from
  the README.

## 12. Release engineering

- **Add a release workflow** (`.github/workflows/release.yml`) that
  builds sdist + wheel on tag push and uploads to PyPI via Trusted
  Publishing — no token in repo secrets.
- **Adopt `setuptools-scm` or single-source the version** so
  `fwap/__init__.py:__version__` and `pyproject.toml:version` cannot
  drift (today both have to be edited by hand on every release).

---

_This list is intentionally specific so each item can be picked up as
its own PR. None of the items are blocking; together they bring the
repository up to the conventions implied by its own README._
