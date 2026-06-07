---
paths:
  - "fwap/**/*.py"
---

# Rules for library code under `fwap/`

These apply whenever you edit package source.

- **Public API is frozen by a guard.** If this change adds, removes, or
  renames a name exported from `fwap/__init__.py`, update all three in the
  same PR — `fwap/__init__.py` (chapter-to-module map), `docs/api.rst`
  (autosummary list), and the `FROZEN_PUBLIC_API` tuple in
  `scripts/check_public_api.py`. Otherwise `python scripts/check_public_api.py`
  fails CI.
- **Docstrings + types.** Every new public function/dataclass needs a
  NumPy-style docstring stating units and array shapes, plus PEP 604/585
  type hints (`int | None`, `tuple[int, int]`).
- **Tests.** Add a test under `tests/` for every new function. Core algorithm
  modules also need a `demo_*` path covered by `tests/test_demos.py`.
- **Before claiming done**, run the full CI gate from CLAUDE.md:
  `ruff check fwap/ tests/ scripts/`, `ruff format --check fwap/ tests/ scripts/`,
  `mypy fwap`, `python scripts/check_public_api.py`, and `pytest -x`.
