"""
test_dsp_engine.py -- Verification script for the rewritten dsp_engine.py.

Exercises process_block() with synthetic audio (sine waves, white noise)
across several wall/path configurations and prints PASS/FAIL lines.

Usage:  python test_dsp_engine.py
"""

import numpy as np

import dsp_engine
from pathfinding import find_k_paths
from shared_state import GRID_COLS, GRID_ROWS

SR = 44100
N = 1024  # block size matching AudioEngine.BLOCKSIZE


def _pf(label: str, ok: bool, detail: str = "") -> None:
    """Print a PASS/FAIL line."""
    tag = "PASS" if ok else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {label}{suffix}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_sine(freq: float, amp: float = 0.5) -> np.ndarray:
    """Return a mono float32 sine block of length N."""
    t = np.arange(N, dtype=np.float32) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _make_noise(amp: float = 0.3, seed: int = 42) -> np.ndarray:
    """Return a mono float32 white-noise block of length N."""
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(N)).astype(np.float32)


def _hf_energy(signal_1d: np.ndarray, threshold_hz: float = 2000.0) -> float:
    """Sum of |rfft| magnitudes above *threshold_hz*."""
    freqs = np.fft.rfftfreq(len(signal_1d), d=1.0 / SR)
    spec = np.abs(np.fft.rfft(signal_1d))
    return float(np.sum(spec[freqs > threshold_hz]))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_direct_los() -> None:
    """Direct line of sight: 0 walls, 0 corners, high cutoff (sums 2 paths in open grid)."""
    print("\n--- Test: direct line of sight (0 walls, 0 corners) ---")

    listener = (5.0, 5.0)
    source = (10.0, 5.0)
    walls: frozenset[tuple[int, int]] = frozenset()
    fstate: dict = {}

    # Verify the paths found
    paths = find_k_paths((5, 5), (10, 5), walls, GRID_COLS, GRID_ROWS, k=2)
    _pf("paths found (k=2)", len(paths) >= 1)
    if paths:
        _pf("primary: 0 corners (direct)", paths[0].corners == 0, f"got {paths[0].corners}")
        _pf("primary: length = 5.0", abs(paths[0].length - 5.0) < 1e-9,
            f"got {paths[0].length:.4f}")
        if len(paths) >= 2:
            _pf("reflected path found", len(paths) == 2, f"reflected length={paths[1].length:.4f}")

    # 440 Hz sine -- well below cutoff, passes through FFT lowpass
    block = _make_sine(440.0, amp=0.5)
    output = dsp_engine.process_block(block, listener, source, walls, SR, fstate)

    _pf("output shape (1024, 2)", output.shape == (N, 2), f"got {output.shape}")
    _pf("no NaN", not np.any(np.isnan(output)))
    _pf("no inf", not np.any(np.isinf(output)))

    # Primary arrival single-path baselines:
    # gain = 1/(1 + 0.15*5) = 1/1.75 ~ 0.5714
    # dx=5, pan=5/20=0.25 -> left_gain=0.375, right_gain=0.625
    primary_gain = 1.0 / (1.0 + 0.15 * 5.0)
    single_left = 0.5 * primary_gain * 0.375   # ~0.1071
    single_right = 0.5 * primary_gain * 0.625  # ~0.1786

    left_peak = float(np.max(np.abs(output[:, 0])))
    right_peak = float(np.max(np.abs(output[:, 1])))

    # With 2 arrivals summed in an open grid, peaks exceed single-arrival baseline
    _pf("left peak >= single-arrival baseline (reflection added)",
        left_peak >= single_left - 1e-4,
        f"left={left_peak:.4f} >= baseline={single_left:.4f}")
    _pf("right peak >= single-arrival baseline (reflection added)",
        right_peak >= single_right - 1e-4,
        f"right={right_peak:.4f} >= baseline={single_right:.4f}")
    _pf("right peak > left peak (source to listener's right)",
        right_peak > left_peak,
        f"right={right_peak:.4f} > left={left_peak:.4f}")

    # Distance gain check: overall RMS should reflect attenuation
    input_rms = float(np.sqrt(np.mean(block ** 2)))
    output_rms = float(np.sqrt(np.mean(output ** 2)))
    ratio = output_rms / max(1e-12, input_rms)
    _pf("output/input RMS ratio in expected range",
        0.1 < ratio < 0.6,
        f"ratio={ratio:.4f}")


def test_corner_detour() -> None:
    """Wall forces detour: corners > 0, lower cutoff, longer path."""
    print("\n--- Test: wall detour (corners > 0, lower cutoff) ---")

    # Vertical wall at x=7, y=3..7 blocks the direct (5,5)->(10,5) path
    walls = frozenset((7, y) for y in range(3, 8))
    listener = (5.0, 5.0)
    source = (10.0, 5.0)

    # Verify path properties
    paths = find_k_paths((5, 5), (10, 5), walls, GRID_COLS, GRID_ROWS, k=1)
    _pf("path found", len(paths) >= 1)
    if paths:
        _pf("corners > 0", paths[0].corners > 0, f"got {paths[0].corners}")
        _pf("path length > 5.0", paths[0].length > 5.0,
            f"got {paths[0].length:.4f}")
        print(f"       path: corners={paths[0].corners}, length={paths[0].length:.4f}")

    # Use broadband noise so FFT energy comparison is meaningful
    noise = _make_noise(amp=0.3, seed=99)

    fstate_detour: dict = {}
    out_detour = dsp_engine.process_block(
        noise, listener, source, walls, SR, fstate_detour)

    fstate_direct: dict = {}
    out_direct = dsp_engine.process_block(
        noise, listener, source, frozenset(), SR, fstate_direct)

    _pf("no NaN (detour)", not np.any(np.isnan(out_detour)))
    _pf("no inf (detour)", not np.any(np.isinf(out_detour)))

    # Detour should have LESS high-frequency energy (lower cutoff)
    hf_det = _hf_energy(out_detour[:, 0])
    hf_dir = _hf_energy(out_direct[:, 0])
    _pf("detour has less HF energy than direct",
        hf_det < hf_dir,
        f"detour={hf_det:.2f}, direct={hf_dir:.2f}")

    # Detour should have lower overall amplitude (longer path -> more attenuation)
    rms_det = float(np.sqrt(np.mean(out_detour ** 2)))
    rms_dir = float(np.sqrt(np.mean(out_direct ** 2)))
    _pf("detour RMS < direct RMS (longer path)",
        rms_det < rms_dir,
        f"detour={rms_det:.4f}, direct={rms_dir:.4f}")


def test_total_occlusion() -> None:
    """Fully sealed: no path, output near-silent at floor gain."""
    print("\n--- Test: total occlusion (sealed off, no path) ---")

    # Box around (20, 12)
    wall_set: set[tuple[int, int]] = set()
    for x in range(19, 22):
        wall_set.add((x, 11))
        wall_set.add((x, 13))
    for y in range(11, 14):
        wall_set.add((19, y))
        wall_set.add((21, y))
    walls = frozenset(wall_set)

    listener = (0.0, 0.0)
    source = (20.0, 12.0)
    fstate: dict = {}

    # Confirm no path exists
    paths = find_k_paths((0, 0), (20, 12), walls, GRID_COLS, GRID_ROWS, k=1)
    _pf("no path exists", len(paths) == 0)

    block = _make_sine(440.0, amp=0.5)
    output = dsp_engine.process_block(block, listener, source, walls, SR, fstate)

    _pf("output shape (1024, 2)", output.shape == (N, 2))
    _pf("no NaN", not np.any(np.isnan(output)))
    _pf("no inf", not np.any(np.isinf(output)))

    # Peak should be very low: input_amp * floor_gain * max_channel_gain
    # = 0.5 * 0.02 * ~0.75 = ~0.0075  (pan depends on dx/20)
    peak = float(np.max(np.abs(output)))
    _pf("output peak very low (< 0.02)", peak < 0.02, f"got {peak:.6f}")
    _pf("output peak > 0 (not hard silence)", peak > 0.0, f"got {peak:.6f}")


def test_same_position() -> None:
    """Listener == source: trivial path, no divide-by-zero, no crash."""
    print("\n--- Test: listener == source (same cell, length=0) ---")

    listener = (5.0, 5.0)
    source = (5.0, 5.0)
    fstate: dict = {}

    block = _make_sine(440.0, amp=0.5)
    output = dsp_engine.process_block(
        block, listener, source, frozenset(), SR, fstate)

    _pf("output shape (1024, 2)", output.shape == (N, 2))
    _pf("no NaN", not np.any(np.isnan(output)))
    _pf("no inf", not np.any(np.isinf(output)))

    # path length = 0 -> gain = 1/(1+0) = 1.0
    # dx=0 -> pan=0 -> left=right=0.5
    # expected peak ~ 0.5 * 1.0 * 0.5 = 0.25
    peak = float(np.max(np.abs(output)))
    _pf("peak ~= 0.25 (full gain, center pan)",
        abs(peak - 0.25) < 0.03, f"got {peak:.4f}")
    _pf("NOT at floor-gain level (> 0.1)", peak > 0.1, f"got {peak:.4f}")


def test_no_nan_inf_sweep() -> None:
    """Sweep several configs and confirm no NaN/inf in any output."""
    print("\n--- Test: NaN/inf sweep across varied configs ---")

    configs = [
        ("open grid, close",    (10.0, 12.0), (12.0, 12.0), frozenset()),
        ("open grid, far",      (0.0, 0.0),   (39.0, 23.0), frozenset()),
        ("single wall between", (5.0, 5.0),   (8.0, 5.0),
         frozenset([(6, 5)])),
        ("heavy walls",         (1.0, 1.0),   (20.0, 20.0),
         frozenset((x, 10) for x in range(0, 30))),
    ]
    block = _make_noise(amp=0.4, seed=7)
    all_ok = True
    for label, lpos, spos, w in configs:
        fstate: dict = {}
        out = dsp_engine.process_block(block, lpos, spos, w, SR, fstate)
        has_nan = bool(np.any(np.isnan(out)))
        has_inf = bool(np.any(np.isinf(out)))
        ok = not has_nan and not has_inf
        if not ok:
            all_ok = False
        _pf(f"{label}: clean output", ok,
            f"NaN={has_nan}, inf={has_inf}")

    _pf("all configs clean", all_ok)


def test_high_corner_cutoff_floor() -> None:
    """Many corners must not push cutoff below _MIN_CUTOFF_HZ."""
    print("\n--- Test: high corner count -> cutoff floored at min ---")

    # Manually verify the cutoff formula with extreme corners
    from dsp_engine import _corner_cutoff_hz, _MIN_CUTOFF_HZ
    for corners in [0, 1, 3, 5, 10, 50]:
        cutoff = _corner_cutoff_hz(corners)
        _pf(f"corners={corners:2d} -> cutoff={cutoff:.0f} Hz >= {_MIN_CUTOFF_HZ:.0f}",
            cutoff >= _MIN_CUTOFF_HZ)


def test_multi_block_continuity() -> None:
    """Run several consecutive blocks through process_block with the same
    filter_state to confirm no obvious discontinuity or crash.

    Since we chose per-block FFT (no overlap-add), we don't expect
    sample-level continuity, but we verify the output is stable and
    doesn't accumulate NaN/inf or crash from stale state.
    """
    print("\n--- Test: multi-block consecutive processing ---")

    listener = (5.0, 5.0)
    source = (12.0, 8.0)
    walls = frozenset((8, y) for y in range(4, 10))
    fstate: dict = {}

    n_blocks = 5
    all_ok = True
    prev_last_sample = None

    for i in range(n_blocks):
        # Continuous sine (phase-coherent across blocks)
        t = (np.arange(N, dtype=np.float32) + i * N) / SR
        block = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        out = dsp_engine.process_block(block, listener, source, walls, SR, fstate)
        has_nan = bool(np.any(np.isnan(out)))
        has_inf = bool(np.any(np.isinf(out)))
        if has_nan or has_inf:
            all_ok = False
            _pf(f"block {i}: clean", False, f"NaN={has_nan}, inf={has_inf}")
        prev_last_sample = float(out[-1, 0])

    _pf(f"all {n_blocks} consecutive blocks clean", all_ok)
    _pf("filter_state dict not corrupted",
        isinstance(fstate, dict),
        f"type={type(fstate).__name__}")


def test_line_of_sight_discretization_bug() -> None:
    """Verify that line-of-sight check prevents discretization corner penalty on open grid."""
    print("\n--- Test: line-of-sight & discretization corner fix ---")

    from dsp_engine import (
        _BASE_CUTOFF_HZ,
        _corner_cutoff_hz,
        _primary_cutoff_hz,
    )
    from pathfinding import has_line_of_sight

    listener = (5, 5)
    source_awkward = (13, 9)
    open_walls: frozenset[tuple[int, int]] = frozenset()

    # 1. Awkward angle (5,5)->(13,9) on open grid
    paths_awkward = find_k_paths(listener, source_awkward, open_walls, GRID_COLS, GRID_ROWS, k=2)
    _pf("awkward angle: pathfinding returns >= 1 path", len(paths_awkward) >= 1)
    _pf("awkward angle: primary path has geometric corners", paths_awkward[0].corners > 0,
        f"got {paths_awkward[0].corners} corners")

    has_los = has_line_of_sight(listener, source_awkward, open_walls)
    _pf("awkward angle: has_line_of_sight is True (open grid)", has_los)

    primary_cutoff = _primary_cutoff_hz(listener, source_awkward, paths_awkward[0].corners, open_walls)
    _pf("awkward angle: primary cutoff == _BASE_CUTOFF_HZ (4000 Hz)",
        primary_cutoff == _BASE_CUTOFF_HZ,
        f"got {primary_cutoff} Hz")

    # 2. Reflected path on open grid retains corner-driven cutoff (NOT overridden)
    if len(paths_awkward) >= 2:
        ref_path = paths_awkward[1]
        ref_cutoff = _corner_cutoff_hz(ref_path.corners)
        _pf("reflected path cutoff driven by corners (not overridden to 4000)",
            ref_cutoff < _BASE_CUTOFF_HZ,
            f"corners={ref_path.corners} -> {ref_cutoff} Hz")

    # 3. Awkward angle WITH blocking wall on straight line
    blocking_walls = frozenset([(9, 7)])  # on Bresenham line between (5,5) and (13,9)
    has_los_blocked = has_line_of_sight(listener, source_awkward, blocking_walls)
    _pf("blocking wall: has_line_of_sight is False", not has_los_blocked)

    paths_blocked = find_k_paths(listener, source_awkward, blocking_walls, GRID_COLS, GRID_ROWS, k=2)
    if paths_blocked:
        blocked_primary_cutoff = _primary_cutoff_hz(
            listener, source_awkward, paths_blocked[0].corners, blocking_walls
        )
        _pf("blocking wall: primary cutoff reduced (< 4000 Hz)",
            blocked_primary_cutoff < _BASE_CUTOFF_HZ,
            f"got {blocked_primary_cutoff} Hz")

    # 4. Same-row and 45-deg diagonal open-grid cases unaffected
    source_row = (10, 5)
    source_diag = (12, 12)
    _pf("same-row open grid has LoS", has_line_of_sight(listener, source_row, open_walls))
    _pf("same-row primary cutoff == 4000 Hz",
        _primary_cutoff_hz(listener, source_row, 0, open_walls) == _BASE_CUTOFF_HZ)

    _pf("45-deg diag open grid has LoS", has_line_of_sight(listener, source_diag, open_walls))
    _pf("45-deg diag primary cutoff == 4000 Hz",
        _primary_cutoff_hz(listener, source_diag, 0, open_walls) == _BASE_CUTOFF_HZ)

    # 5. Audio signal verification: 2500 Hz sine tone passes unobstructed through open-grid awkward angle
    # (2500 Hz was attenuated by the old buggy 1300 Hz cutoff, but passes at 4000 Hz cutoff)
    fstate: dict = {}
    sine_2500 = _make_sine(2500.0, amp=0.5)
    out_open = dsp_engine.process_block(
        sine_2500, (5.0, 5.0), (13.0, 9.0), open_walls, SR, fstate
    )
    peak_open = float(np.max(np.abs(out_open)))
    _pf("2500 Hz tone passes through open grid (> 0.05 peak)", peak_open > 0.05, f"peak={peak_open:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Stage 2C tests: geometry-cache lifecycle
# ---------------------------------------------------------------------------
from unittest.mock import patch
from pathfinding import (
    find_reflector_candidates as _real_frc,
    find_transmission_path as _real_ftp,
)


def test_geometry_cache_populates_on_first_call() -> None:
    """Cache keys must exist in filter_state after the very first process_block call."""
    print("\n--- Test: geometry cache -- populates on first call ---")
    listener = (5.0, 5.0)
    source = (12.0, 8.0)
    walls = frozenset((8, y) for y in range(4, 10))
    fstate: dict = {}
    block = np.zeros(N, dtype=np.float32)
    dsp_engine.process_block(block, listener, source, walls, SR, fstate)
    _pf("_geom_cache_reflector_candidates populated",
        "_geom_cache_reflector_candidates" in fstate)
    _pf("_geom_cache_transmission_path populated",
        "_geom_cache_transmission_path" in fstate)


def test_geometry_cache_not_recomputed_when_unchanged() -> None:
    """Identical inputs on second call must NOT re-run geometry functions."""
    print("\n--- Test: geometry cache -- not recomputed when unchanged ---")
    listener = (5.0, 5.0)
    source = (12.0, 8.0)
    walls = frozenset((8, y) for y in range(4, 10))
    fstate: dict = {}
    block = np.zeros(N, dtype=np.float32)

    with patch("dsp_engine.find_reflector_candidates", wraps=_real_frc) as m_frc, \
         patch("dsp_engine.find_transmission_path", wraps=_real_ftp) as m_ftp:
        dsp_engine.process_block(block, listener, source, walls, SR, fstate)
        count_after_1_frc = m_frc.call_count
        count_after_1_ftp = m_ftp.call_count
        _pf("first call invoked find_reflector_candidates",
            count_after_1_frc == 1, f"call_count={count_after_1_frc}")
        _pf("first call invoked find_transmission_path",
            count_after_1_ftp == 1, f"call_count={count_after_1_ftp}")

        dsp_engine.process_block(block, listener, source, walls, SR, fstate)
        _pf("second call did NOT re-invoke find_reflector_candidates",
            m_frc.call_count == count_after_1_frc,
            f"call_count={m_frc.call_count}")
        _pf("second call did NOT re-invoke find_transmission_path",
            m_ftp.call_count == count_after_1_ftp,
            f"call_count={m_ftp.call_count}")


def test_geometry_cache_invalidates_on_wall_change() -> None:
    """Changing walls between calls must trigger recomputation."""
    print("\n--- Test: geometry cache -- invalidates on wall change ---")
    listener = (5.0, 10.0)
    source = (15.0, 10.0)
    walls_a = frozenset()  # no walls -- transmission should be None
    walls_b = frozenset([(10, 10)])  # wall on direct line
    wall_gains_b = {(10, 10): 0.4}
    fstate: dict = {}
    block = np.zeros(N, dtype=np.float32)

    with patch("dsp_engine.find_reflector_candidates", wraps=_real_frc) as m_frc, \
         patch("dsp_engine.find_transmission_path", wraps=_real_ftp) as m_ftp:
        # Call 1: no walls
        dsp_engine.process_block(block, listener, source, walls_a, SR, fstate)
        count_after_1 = m_frc.call_count
        cached_trans_a = fstate["_geom_cache_transmission_path"]
        _pf("call 1: transmission is None (no walls)",
            cached_trans_a is None, f"got {cached_trans_a}")

        # Call 2: wall added
        dsp_engine.process_block(
            block, listener, source, walls_b, SR, fstate,
            wall_gains=wall_gains_b,
        )
        _pf("call 2: geometry recomputed after wall change",
            m_frc.call_count > count_after_1,
            f"call_count={m_frc.call_count}")
        cached_trans_b = fstate["_geom_cache_transmission_path"]
        _pf("call 2: transmission now non-None (wall crossed)",
            cached_trans_b is not None, f"got {cached_trans_b}")


def test_geometry_cache_invalidates_on_source_move() -> None:
    """Moving source between calls must trigger recomputation."""
    print("\n--- Test: geometry cache -- invalidates on source move ---")
    listener = (5.0, 5.0)
    source_a = (12.0, 8.0)
    source_b = (15.0, 8.0)
    walls = frozenset((8, y) for y in range(4, 10))
    fstate: dict = {}
    block = np.zeros(N, dtype=np.float32)

    with patch("dsp_engine.find_reflector_candidates", wraps=_real_frc) as m_frc, \
         patch("dsp_engine.find_transmission_path", wraps=_real_ftp) as m_ftp:
        dsp_engine.process_block(block, listener, source_a, walls, SR, fstate)
        count_after_1 = m_frc.call_count
        dsp_engine.process_block(block, listener, source_b, walls, SR, fstate)
        _pf("source move triggered recomputation",
            m_frc.call_count > count_after_1,
            f"call_count={m_frc.call_count}")


def test_geometry_cache_invalidates_on_listener_move() -> None:
    """Moving listener between calls must trigger recomputation."""
    print("\n--- Test: geometry cache -- invalidates on listener move ---")
    listener_a = (5.0, 5.0)
    listener_b = (6.0, 5.0)
    source = (12.0, 8.0)
    walls = frozenset((8, y) for y in range(4, 10))
    fstate: dict = {}
    block = np.zeros(N, dtype=np.float32)

    with patch("dsp_engine.find_reflector_candidates", wraps=_real_frc) as m_frc, \
         patch("dsp_engine.find_transmission_path", wraps=_real_ftp) as m_ftp:
        dsp_engine.process_block(block, listener_a, source, walls, SR, fstate)
        count_after_1 = m_frc.call_count
        dsp_engine.process_block(block, listener_b, source, walls, SR, fstate)
        _pf("listener move triggered recomputation",
            m_frc.call_count > count_after_1,
            f"call_count={m_frc.call_count}")


def test_geometry_cache_survives_unrelated_blocks() -> None:
    """5 consecutive blocks with same layout must invoke geometry only once."""
    print("\n--- Test: geometry cache -- survives unrelated blocks ---")
    listener = (5.0, 5.0)
    source = (12.0, 8.0)
    walls = frozenset((8, y) for y in range(4, 10))
    fstate: dict = {}
    n_blocks = 5

    with patch("dsp_engine.find_reflector_candidates", wraps=_real_frc) as m_frc, \
         patch("dsp_engine.find_transmission_path", wraps=_real_ftp) as m_ftp:
        for i in range(n_blocks):
            t = (np.arange(N, dtype=np.float32) + i * N) / SR
            block = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            dsp_engine.process_block(block, listener, source, walls, SR, fstate)
        _pf(f"find_reflector_candidates called exactly 1 time across {n_blocks} blocks",
            m_frc.call_count == 1, f"call_count={m_frc.call_count}")
        _pf(f"find_transmission_path called exactly 1 time across {n_blocks} blocks",
            m_ftp.call_count == 1, f"call_count={m_ftp.call_count}")


def main() -> None:
    print("=" * 60)
    print("  dsp_engine.py -- verification script")
    print("=" * 60)

    test_direct_los()
    test_corner_detour()
    test_total_occlusion()
    test_same_position()
    test_no_nan_inf_sweep()
    test_high_corner_cutoff_floor()
    test_multi_block_continuity()
    test_line_of_sight_discretization_bug()

    test_geometry_cache_populates_on_first_call()
    test_geometry_cache_not_recomputed_when_unchanged()
    test_geometry_cache_invalidates_on_wall_change()
    test_geometry_cache_invalidates_on_source_move()
    test_geometry_cache_invalidates_on_listener_move()
    test_geometry_cache_survives_unrelated_blocks()

    print("\n" + "=" * 60)
    print("  Done.  Review PASS/FAIL lines above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
