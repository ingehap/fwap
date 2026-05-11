"""Sphinx configuration for fwap.

Run from the repo root::

    pip install -e .[docs]
    sphinx-build -b html docs docs/_build/html
"""

from __future__ import annotations

import os
import sys

# Make the package importable for autodoc without requiring install.
sys.path.insert(0, os.path.abspath(".."))

import fwap  # noqa: E402

project = "fwap"
author = "fwap contributors"
copyright = "2026, fwap contributors"
release = fwap.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # NumPy-style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_nb",  # Jupyter notebook rendering
]

# myst-nb: don't re-execute notebooks at build time (too expensive
# for CI); rely on the committed cell outputs.
nb_execution_mode = "off"

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"

napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# PDF / LaTeX build: use xelatex so docstrings with Unicode glyphs
# (⇔ in fwap.dispersion's "<=>" notation, etc.) compile without
# pdflatex's "Unicode character not set up for use" failure. The
# xelatex driver also gives us a Unicode-capable text-mode by
# default, removing the need for inputenc / utf8x shims.
latex_engine = "xelatex"

# Per-section PDFs in addition to the all-in-one reference. Each
# entry produces a standalone LaTeX document; ``make`` in the
# build/latex directory then turns each .tex file into its own .pdf.
# Splitting like this lets GitHub's blob viewer render the shorter
# sections inline (it lazy-loads PDFs and caps the inline view at a
# few pages, so a single 139-page reference is awkward to read on
# the web).
latex_documents = [
    # All-in-one reference.
    (
        "index",
        "fwap.tex",
        "fwap -- Full-Waveform Acoustic Processing",
        author,
        "manual",
    ),
    # Per-top-level-section PDFs (~5-10 pages each).
    ("quickstart", "fwap-quickstart.tex", "fwap -- Quick start", author, "howto"),
    (
        "chapter_map",
        "fwap-chapter-map.tex",
        "fwap -- Chapter-to-module map",
        author,
        "howto",
    ),
    ("roadmap", "fwap-roadmap.tex", "fwap -- Roadmap", author, "howto"),
    ("changelog", "fwap-changelog.tex", "fwap -- Changelog", author, "howto"),
]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = f"fwap {release}"
