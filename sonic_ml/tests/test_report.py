"""Tests for sonic_ml.bench.report."""

from __future__ import annotations

from sonic_ml.bench import StubPredictor, evaluate, format_scorecard


def test_format_scorecard_contains_header_and_rows(bundle, geom):
    sc = evaluate(StubPredictor(2000.0, name="mymodel"), bundle, geom, n_boot=50)
    text = format_scorecard(sc)
    assert "Scorecard: mymodel" in text
    assert "parameter: vs" in text
    assert "all" in text
    for regime in ("slow", "fast"):
        if regime in sc.per_regime:
            assert regime in text
    # One line per present regime, plus header/columns/rule.
    n_regime_rows = sum(1 for r in ("all", "slow", "fast") if r in sc.per_regime)
    assert len(text.splitlines()) == 3 + n_regime_rows


def test_format_scorecard_is_string(bundle, geom):
    sc = evaluate(StubPredictor(None), bundle, geom, n_boot=50)
    assert isinstance(format_scorecard(sc), str)
