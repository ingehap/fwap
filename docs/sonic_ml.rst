``sonic_ml`` -- the machine-learning layer
==========================================

``sonic_ml`` is an **in-repo sibling package** that treats the :mod:`fwap`
forward solver as a labelled-data factory and learns surrogate and inverse
models on top of it. It lives in :file:`sonic_ml/` with its own
:file:`pyproject.toml` and is deliberately *not* part of the :mod:`fwap`
distribution: the physics package stays pure NumPy/SciPy, and installing or
importing :mod:`fwap` never pulls in PyTorch.

The separation is enforced, not merely intended:

* the package name is not ``fwap*``, so setuptools' ``include=["fwap*"]``
  excludes it from the core wheel;
* the core CI gate (lint, mypy, public-API guard, tests) never sees it;
* ``sonic_ml`` runs in its own **non-required** workflow, so an ML failure
  cannot block a physics change;
* the pure-NumPy *spine* (``import sonic_ml``) is torch-free -- only
  ``import sonic_ml.models`` pulls the dependency, and a test asserts it.

Installation
------------

.. code-block:: bash

   pip install -e ".[dev]"           # core fwap
   pip install torch                 # the CPU wheel is sufficient
   pip install -e "./sonic_ml[dev]"  # the ML layer

Layers
------

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - Layer
     - What it does
     - Entry points
   * - Data
     - Cylindrical-Biot forward solves turned into a versioned ``.npz`` of
       (parameters, dispersion, waveform) triples -- open-hole and cased-hole
     - :file:`scripts/gen_surrogate_dataset.py`
   * - Spine (torch-free)
     - Defensive loader with schema versioning, provenance capture,
       regime-stratified splits, standardization, determinism
     - ``sonic_ml.loader``, ``sonic_ml.provenance``, ``sonic_ml.split``
   * - Benchmark
     - Model-agnostic scoring of any shear-velocity or cement-bond predictor,
       with bootstrap confidence intervals and per-regime rows
     - ``sonic_ml.bench``
   * - Baselines
     - The classical bar an ML model must clear: dispersion-corrected STC, f-k
       dispersion, and a Stoneley-slowness cement-bond calibration
     - ``sonic_ml.baselines``
   * - Models
     - Forward surrogate, DL-FWI inverse net with calibrated uncertainty, a
       low-latency LWD variant, FNO / DeepONet operators, the cased-hole
       operator plus cement-bond inverse, and surrogate-in-the-loop inversion
       (single-frame and depth-coupled)
     - ``sonic_ml.models``

The ``.npz`` contract
---------------------

The dataset file is the sole hand-off between the physics and the ML layer, so
it is versioned and frozen by a test in the *core* suite
(:file:`tests/test_npz_schema_contract.py`). A breaking change to the layout
fails core CI rather than silently mislabelling training data downstream.

============  ==========================================================
Version       Adds
============  ==========================================================
v1            ``schema_version``
v2            acquisition geometry (``dt`` / ``tr_offset`` / ``dr``), making
              the stored gathers self-describing
v3            per-mode leaky-mode ``attenuation``
v4            cased-hole annulus (``layer_params`` / ``layer_names``) and the
              cement ``bond_index``
============  ==========================================================

The loader accepts every version and exposes the newer fields as ``None`` on
older files, so old datasets keep working.

Two results, and the gap between them
-------------------------------------

**Open hole.** The DL-FWI inverse net recovers :math:`V_S` roughly an order of
magnitude more accurately than classical slowness-time processing on identical
held-out gathers -- including the fast-formation regime, where the default
two-mode gather carries no clean pseudo-Rayleigh arrival and classical STC
starves.

**Cased hole.** The cement-bond inverse reaches only about twice the skill of
predicting the training mean. That is a property of the problem rather than of
the model: sweeping cement stiffness across its prior moves the cased Stoneley
curve by roughly 7%, while sweeping the formation shear velocity moves it about
1.5% -- less than the nuisance variation from cement thickness alone. The
heteroscedastic head reports calibrated error bars (residual *z*-score standard
deviation near 1) that say exactly this, which is what makes a modest number
usable instead of misleading.

Reporting both, with the reason for the difference, is the point. A library
that only published the first number would be advertising rather than
measuring.

Honest-measurement helpers
--------------------------

Three utilities exist specifically to keep attractive claims checkable:

``sonic_ml.models.joint``
   Inverting a whole logged interval with a frame-to-frame smoothness penalty
   improves accuracy -- but so, potentially, does the far simpler act of
   inverting each frame alone and running a moving average afterwards, so
   ``smooth_independent`` ships as the control and is scored on every result.
   Depth coupling does win, unevenly: 38% against 0% on :math:`V_S`, and a tie
   on density. Tuned as a user would have to tune it -- by cross-validation
   rather than against the answer -- coupling keeps 17-29% and smoothing keeps
   nothing, because averaging after the fact degrades data misfit monotonically
   and so cannot be tuned from data at all. The module also reports where its
   own selector fails: on a noise-free log it over-couples badly enough to lose
   18-29%, so the method is quoted with the condition that it pays on noisy
   picks and costs on clean ones.

``sonic_ml.models.regrid``
   An operator can be evaluated on a frequency grid it never trained on. That
   is only *useful* if it beats the boring alternative of predicting on the
   training grid and interpolating, so ``evaluate_regridding`` always computes
   that control alongside the zero-shot number, and the formatter always prints
   a verdict. On smooth dispersion curves the control currently wins.

``sonic_ml.bench``
   Every reported skill figure is quoted against an explicit no-skill reference
   (``StubPredictor`` / ``MeanBondPredictor``) rather than as an absolute error,
   and confidence intervals come from a seeded bootstrap.

Tutorials
---------

* :doc:`notebooks/sonic_ml_tutorial` -- the open-hole loop end to end:
  generate, train a DL-FWI inverse, and score it against the classical
  baseline.
* :doc:`notebooks/cased_hole_tutorial` -- cement-bond evaluation behind casing,
  opening with the forward sensitivity sweep that predicts the result.

Both are re-executed in CI, so their numbers are reproducible rather than
decorative.
