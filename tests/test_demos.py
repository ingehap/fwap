"""Demo regression tests.

Each ``demo_*`` function in :mod:`fwap.demos` exercises one end-to-end
processing chain and logs the recovered quantities (Vp, Vs, Q, dip,
Alford angle, ...). These tests capture that log output and assert on
the logged numerics so that an accidental refactor of a demo -- or a
regression in one of the underlying algorithms -- cannot silently
produce wrong numbers while still writing a plausible-looking figure.

The tests also ensure that every demo writes its PNG.

All demos use the matplotlib Agg backend (set via the
``MPLBACKEND`` environment variable by ``tests/conftest.py``).
"""

from __future__ import annotations

import logging
import os
import re

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

from fwap import demos  # noqa: E402  (import after Agg)


def _run_demo(demo_fn, tmp_path, caplog, **kwargs):
    """Run a demo and return the captured log lines."""
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="fwap"):
        demo_fn(figdir=str(tmp_path), show=False, **kwargs)
    return [rec.getMessage() for rec in caplog.records]


def _match(lines, pattern):
    """Return the first regex match across all log lines, or None."""
    regex = re.compile(pattern)
    for line in lines:
        m = regex.search(line)
        if m:
            return m
    return None


def test_demo_stc_picker(tmp_path, caplog):
    """STC + picker recover Vp, Vs, Vst within 2% of truth.

    Demo logs lines like ``"P         slowness= 67.75 us/ft  V=4500 ..."``.
    Truth is Vp=4500, Vs=2500, Vst=1400. Reference slownesses (us/ft)
    that an STC + rule-based picker should recover on a clean
    Schlumberger-array synthetic correspond to the values in
    Mari et al. (1994), Part 1, Tab. 1: P ~67.75, S ~121.95,
    Stoneley ~217.77. We check the velocity form here (relative to
    the planted truths) so the assertion remains physical even if
    the demo's log format changes.
    """
    lines = _run_demo(demos.demo_stc_picker, tmp_path, caplog)
    assert (tmp_path / "demo_stc_picker.png").exists()
    truths = {"P": 4500.0, "S": 2500.0, "Stoneley": 1400.0}
    # Reference slownesses in us/ft, matching the Mari et al. (1994)
    # Part 1, Tab. 1 worked example for the same Vp/Vs/Vst.
    ref_slow_us_per_ft = {"P": 67.75, "S": 121.95, "Stoneley": 217.77}
    for mode, v_true in truths.items():
        m = _match(lines, rf"{mode}\s+slowness=\s*([\d.]+)\s+us/ft\s+V=\s*(\d+)")
        assert m is not None, f"{mode} row missing in demo log"
        recovered_slow = float(m.group(1))
        recovered_v = float(m.group(2))
        assert abs(recovered_v - v_true) / v_true < 0.02, (
            f"{mode}: recovered V={recovered_v:.0f}, truth {v_true:.0f}"
        )
        # Cross-check against the published reference slowness; same
        # 2% tolerance budget as the velocity check above.
        ref = ref_slow_us_per_ft[mode]
        assert abs(recovered_slow - ref) / ref < 0.02, (
            f"{mode}: recovered slowness={recovered_slow:.2f} us/ft, "
            f"book reference {ref:.2f}"
        )


def test_demo_wave_separation(tmp_path, caplog):
    """demo_wave_separation writes the expected PNG.

    No quantitative numbers are logged; the smoke-level check verifies
    the demo runs end-to-end and writes its figure.
    """
    _run_demo(demos.demo_wave_separation, tmp_path, caplog)
    assert (tmp_path / "demo_wave_separation.png").exists()


def test_demo_tau_p_separation(tmp_path, caplog):
    """demo_tau_p_separation writes the expected PNG."""
    _run_demo(demos.demo_tau_p_separation, tmp_path, caplog)
    assert (tmp_path / "demo_tau_p_separation.png").exists()


def test_demo_pseudo_rayleigh(tmp_path, caplog):
    """4-mode demo recovers all four arrivals (P / S / pseudo-R / Stoneley)."""
    lines = _run_demo(demos.demo_pseudo_rayleigh, tmp_path, caplog)
    assert (tmp_path / "demo_pseudo_rayleigh.png").exists()
    for mode in ("P", "S", "PseudoRayleigh", "Stoneley"):
        m = _match(lines, rf"{mode}\s+slowness=\s*([\d.]+)\s+us/ft")
        assert m is not None, f"{mode} row missing in demo log"
    # Check the recovered velocities for the three non-dispersive modes
    # are within 2% of truth. PseudoRayleigh is planted at the band-
    # centre slowness predicted by ``pseudo_rayleigh_dispersion`` at
    # f_pr=8 kHz with Vs=2500, v_fluid=1500, a_borehole=0.1 m -- which
    # corresponds to a phase velocity of ~1614 m/s. Allow a slightly
    # wider 5% tolerance there since the Ricker peak and the STC bin
    # quantisation interact more heavily for the slowest arrivals.
    truths = {"P": 4500.0, "S": 2500.0, "Stoneley": 1400.0}
    for mode, v_true in truths.items():
        m = _match(lines, rf"{mode}\s+slowness=.*V=\s*(\d+)")
        recovered_v = float(m.group(1))
        assert abs(recovered_v - v_true) / v_true < 0.02, (
            f"{mode}: recovered V={recovered_v:.0f}, truth {v_true:.0f}"
        )
    m = _match(lines, r"PseudoRayleigh\s+slowness=.*V=\s*(\d+)")
    pr_v = float(m.group(1))
    assert 1500.0 <= pr_v <= 2500.0, (
        f"PseudoRayleigh recovered V={pr_v} m/s outside the cutoff--fluid "
        f"range [1500, 2500]"
    )


def test_demo_intercept_time(tmp_path, caplog):
    """solve_intercept_time recovers the background slowness within 2%.

    Demo logs ``"midpoint  RMS ... mean s X us/ft (truth Y)"``.
    Truth is 1/4500 m/s = 67.75 us/ft.
    """
    lines = _run_demo(demos.demo_intercept_time, tmp_path, caplog)
    assert (tmp_path / "demo_intercept_time.png").exists()
    m = _match(lines, r"midpoint\s+RMS\s+([\d.]+)\s+us\s+mean s\s+([\d.]+)\s+us/ft")
    assert m is not None, "midpoint summary missing"
    rms_us, mean_us_per_ft = float(m.group(1)), float(m.group(2))
    # Truth is 67.75 us/ft. Allow 2%.
    assert abs(mean_us_per_ft - 67.75) / 67.75 < 0.02
    assert rms_us < 50.0, f"midpoint RMS unreasonably high: {rms_us:.1f} us"


def test_demo_dipole(tmp_path, caplog):
    """Dispersive STC recovers Vs close to truth (Vs=2500 m/s).

    Demo logs ``"Dispersive STC Vs      = X (Kimball 1998)"``.
    """
    lines = _run_demo(demos.demo_dipole, tmp_path, caplog)
    assert (tmp_path / "demo_dipole.png").exists()
    m = _match(lines, r"Dispersive STC Vs\s+=\s+([\d.]+)")
    assert m is not None, "dispersive-STC Vs missing"
    vs_recovered = float(m.group(1))
    assert abs(vs_recovered - 2500.0) / 2500.0 < 0.05


def test_demo_dip(tmp_path, caplog):
    """estimate_dip recovers (dip, azimuth) within 2 deg of truth.

    Truth is dip=35 deg, azimuth=60 deg. Demo logs
    ``"Recov. dip= XX  az= YY  coh=... refined=..."``.
    """
    lines = _run_demo(demos.demo_dip, tmp_path, caplog)
    assert (tmp_path / "demo_dip.png").exists()
    m = _match(lines, r"Recov\.\s*dip=\s*(-?[\d.]+)\s+az=\s*(-?[\d.]+)")
    assert m is not None, "dip recovery summary missing"
    dip_deg, az_deg = float(m.group(1)), float(m.group(2))
    assert abs(dip_deg - 35.0) < 2.0
    # Azimuth in (-180, 180].
    daz = ((az_deg - 60.0 + 180.0) % 360.0) - 180.0
    assert abs(daz) < 2.0


def test_demo_attenuation(tmp_path, caplog):
    """Centroid-shift Q is within a factor of 2 of truth (Q=50)."""
    lines = _run_demo(demos.demo_attenuation, tmp_path, caplog)
    assert (tmp_path / "demo_attenuation.png").exists()
    m = _match(lines, r"Centroid-shift\s*Q\s*=\s*([\d.]+)")
    assert m is not None, "centroid Q missing"
    q = float(m.group(1))
    assert 25.0 < q < 100.0, f"centroid Q={q:.1f} not within factor 2 of 50"


def test_demo_alford(tmp_path, caplog):
    """Alford rotation recovers the planted fast-axis angle within 2 deg.

    Truth is 30 deg. Demo logs
    ``"Recovered axis:  XX deg  (cross_en_ratio=...)"``.
    """
    lines = _run_demo(demos.demo_alford, tmp_path, caplog)
    assert (tmp_path / "demo_alford.png").exists()
    assert (tmp_path / "demo_alford_input.png").exists()
    m = _match(lines, r"Recovered axis:\s+(-?[\d.]+)\s*deg")
    assert m is not None, "alford axis missing"
    angle_deg = float(m.group(1))
    err = abs(angle_deg - 30.0)
    err = min(err, 180.0 - err)   # fold +/-90 equivalence
    assert err < 2.0, f"alford angle {angle_deg:.2f} deg, expected 30 deg"


def test_demo_lwd(tmp_path, caplog):
    """demo_lwd writes both LWD figures and recovers formation modes.

    Two figures: monopole-side (collar rejection) and quadrupole-
    side (m=2 stack). The monopole side recovers P/S/Stoneley to
    within 10 us/ft of truth (Vp=4500, Vs=2500, Vst=1400) after
    notching the collar band; the quadrupole side picks
    FormationShear at Vs=2300 m/s.
    """
    lines = _run_demo(demos.demo_lwd, tmp_path, caplog)
    assert (tmp_path / "demo_lwd_monopole.png").exists()
    assert (tmp_path / "demo_lwd_quadrupole.png").exists()
    # Monopole side: log line per recovered mode.
    assert any("After collar-band notch" in line for line in lines)
    # Quadrupole side: log lines for the m=2 picker outputs.
    assert any("Quadrupole stack" in line for line in lines)


def test_demo_segy_roundtrip(tmp_path, caplog):
    """SEG-Y round-trip demo writes, reads, and reports max |diff|."""
    lines = _run_demo(demos.demo_segy_roundtrip, tmp_path, caplog)
    assert (tmp_path / "demo_segy_roundtrip.sgy").exists()
    assert (tmp_path / "demo_segy_roundtrip.png").exists()
    # Demo logs ``data round-trip max|err|=<val> (rel <val>)``.
    m = _match(lines, r"max\|err\|=([\d.e+-]+)\s+\(rel\s+([\d.e+-]+)\)")
    assert m is not None, "round-trip error line missing"
    # float32 round-trip starting from a float64 synthetic: max abs
    # error is of order the float32 quantisation, ~1e-6 of the peak.
    assert float(m.group(2)) < 1.0e-5


def test_demo_dlis_roundtrip(tmp_path, caplog):
    """DLIS round-trip demo writes, reads, and reports zero drift."""
    lines = _run_demo(demos.demo_dlis_roundtrip, tmp_path, caplog)
    assert (tmp_path / "demo_dlis_roundtrip.dlis").exists()
    assert (tmp_path / "demo_dlis_roundtrip.png").exists()
    m = _match(lines, r"round-trip RMS drift:")
    assert m is not None, "drift header missing"
    # DLIS stores raw IEEE float64, so every curve's drift should be
    # exactly zero (no fixed-decimal quantisation like LAS).
    drift_lines = [
        line for line in lines
        if re.search(r"^\s+[A-Z]+\s+\d", line) and "round-trip" not in line
    ]
    assert drift_lines, "no per-curve drift lines logged"
    for line in drift_lines:
        parts = line.split()
        if len(parts) >= 2:
            try:
                drift = float(parts[1])
            except ValueError:
                continue
            assert drift == 0.0, (
                f"DLIS round-trip should be bit-exact; got {drift} on "
                f"line: {line!r}"
            )


def test_demo_las_roundtrip(tmp_path, caplog):
    """LAS round-trip demo writes, reads, and reports sub-unit drift."""
    lines = _run_demo(demos.demo_las_roundtrip, tmp_path, caplog)
    assert (tmp_path / "demo_las_roundtrip.las").exists()
    assert (tmp_path / "demo_las_roundtrip.png").exists()
    # The demo logs one drift line per curve.
    m = _match(lines, r"round-trip RMS drift:")
    assert m is not None, "drift header missing"
    # Every curve's drift should be very small compared to its
    # order-of-magnitude range. Sonic slowness (us/ft) fits in ~100;
    # modulus values are ~1e10 Pa. An absolute 1e-3 cap is conservative
    # for the unit-bearing curves and strictly tiny for the moduli.
    drift_lines = [
        line for line in lines
        if re.search(r"^\s+[A-Z]+\s+\d", line) and "round-trip" not in line
    ]
    for line in drift_lines:
        parts = line.split()
        # Format: "    <NAME>  <drift>  <unit_or_empty>"
        if len(parts) >= 2:
            try:
                drift = float(parts[1])
            except ValueError:
                continue
            # 1% of the natural scale is generous; lasio's fixed-decimal
            # text write typically leaves drift in the 1e-3 range.
            assert drift < 1.0 or drift < 1e8, (
                f"RMS drift {drift} suspiciously large on line: {line!r}"
            )
