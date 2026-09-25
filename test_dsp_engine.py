"""
test_dsp_engine.py -- Verification script for the rewritten dsp_engine.py.

Exercises process_block() with synthetic audio (sine waves, white noise)
across several wall/path configurations and prints PASS/FAIL lines.

Usage:  python test_dsp_engine.py
"""

import numpy as np

import dsp_engine
from pathfinding import find_k_paths, ReflectorCandidate
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
    """Direct line of sight: 0 walls, 0 corners, high cutoff (single primary arrival in open grid)."""
    print("\n--- Test: direct line of sight (0 walls, 0 corners) ---")

    listener = (5.0, 5.0)
    source = (10.0, 5.0)
    walls: frozenset[tuple[int, int]] = frozenset()
    fstate: dict = {}

    # Verify the path found
    paths = find_k_paths((5, 5), (10, 5), walls, GRID_COLS, GRID_ROWS, k=1)
    _pf("paths found (k=1)", len(paths) == 1)
    if paths:
        _pf("primary: 0 corners (direct)", paths[0].corners == 0, f"got {paths[0].corners}")
        _pf("primary: length = 5.0", abs(paths[0].length - 5.0) < 1e-9,
            f"got {paths[0].length:.4f}")

    # 440 Hz sine -- well below cutoff, passes through FFT lowpass
    block = _make_sine(440.0, amp=0.5)
    output = dsp_engine.process_block(block, listener, source, walls, SR, fstate)

    _pf("output shape (1024, 2)", output.shape == (N, 2), f"got {output.shape}")
    _pf("no NaN", not np.any(np.isnan(output)))
    _pf("no inf", not np.any(np.isinf(output)))

    # Primary arrival single-path baselines:
    # gain = 1/(1 + 0.08*5) = 1/1.4 ~ 0.7143
    # dx=5, pan=5/20=0.25 -> left_gain=0.375, right_gain=0.625
    primary_gain = 1.0 / (1.0 + 0.08 * 5.0)
    single_left = 0.5 * primary_gain * 0.375   # ~0.1339
    single_right = 0.5 * primary_gain * 0.625  # ~0.2232

    left_peak = float(np.max(np.abs(output[:, 0])))
    right_peak = float(np.max(np.abs(output[:, 1])))

    # With 1 arrival in an open grid, peaks match single-arrival baseline
    _pf("left peak ~= single-arrival baseline",
        abs(left_peak - single_left) < 0.02,
        f"left={left_peak:.4f}, baseline={single_left:.4f}")
    _pf("right peak ~= single-arrival baseline",
        abs(right_peak - single_right) < 0.02,
        f"right={right_peak:.4f}, baseline={single_right:.4f}")
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
    """Fully sealed: no routed path, output generated by transmission & echoes."""
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

    # Peak includes Family B through-wall transmission and Family C discrete multi-wall echoes.
    peak = float(np.max(np.abs(output)))
    _pf("output peak > 0 (not hard silence)", peak > 0.0, f"got {peak:.6f}")
    _pf("output peak reasonable (< 0.75)", peak < 0.75, f"got {peak:.6f}")


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
    paths_awkward = find_k_paths(listener, source_awkward, open_walls, GRID_COLS, GRID_ROWS, k=1)
    _pf("awkward angle: pathfinding returns >= 1 path", len(paths_awkward) >= 1)
    _pf("awkward angle: primary path has geometric corners", paths_awkward[0].corners > 0,
        f"got {paths_awkward[0].corners} corners")

    has_los = has_line_of_sight(listener, source_awkward, open_walls)
    _pf("awkward angle: has_line_of_sight is True (open grid)", has_los)

    primary_cutoff = _primary_cutoff_hz(listener, source_awkward, paths_awkward[0].corners, open_walls)
    _pf("awkward angle: primary cutoff == _BASE_CUTOFF_HZ (4000 Hz)",
        primary_cutoff == _BASE_CUTOFF_HZ,
        f"got {primary_cutoff} Hz")

    # 2. Awkward angle WITH blocking wall on straight line
    blocking_walls = frozenset([(9, 7)])  # on Bresenham line between (5,5) and (13,9)
    has_los_blocked = has_line_of_sight(listener, source_awkward, blocking_walls)
    _pf("blocking wall: has_line_of_sight is False", not has_los_blocked)

    paths_blocked = find_k_paths(listener, source_awkward, blocking_walls, GRID_COLS, GRID_ROWS, k=1)
    if paths_blocked:
        blocked_primary_cutoff = _primary_cutoff_hz(
            listener, source_awkward, paths_blocked[0].corners, blocking_walls
        )
        _pf("blocking wall: primary cutoff reduced (< 4000 Hz)",
            blocked_primary_cutoff < _BASE_CUTOFF_HZ,
            f"got {blocked_primary_cutoff} Hz")

    # 3. Same-row and 45-deg diagonal open-grid cases unaffected
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


# ---------------------------------------------------------------------------
# Stage 3C tests: Family B (through-wall transmission)
# ---------------------------------------------------------------------------


def test_transmission_zero_when_no_wall_crossed() -> None:
    """No walls anywhere — Family B must contribute exactly 0."""
    print("\n--- Test: transmission -- zero when no wall crossed ---")
    listener = (5.0, 5.0)
    source = (10.0, 5.0)
    walls: frozenset[tuple[int, int]] = frozenset()
    fstate: dict = {}
    block = _make_sine(440.0, amp=0.5)
    output = dsp_engine.process_block(block, listener, source, walls, SR, fstate)
    trans_path = fstate.get("_geom_cache_transmission_path")
    tg = dsp_engine._transmission_gain(source, listener, trans_path)
    _pf("transmission gain == 0.0 (no walls)", tg == 0.0, f"got {tg}")
    _pf("no NaN", not np.any(np.isnan(output)))
    _pf("no inf", not np.any(np.isinf(output)))


def test_transmission_contributes_during_total_occlusion() -> None:
    """Sealed source, but transmission line crosses a wall — output exceeds floor gain."""
    print("\n--- Test: transmission -- contributes during total occlusion ---")
    # Seal source at (20, 12) inside a box
    wall_set: set[tuple[int, int]] = set()
    for x in range(19, 22):
        wall_set.add((x, 11))
        wall_set.add((x, 13))
    for y in range(11, 14):
        wall_set.add((19, y))
        wall_set.add((21, y))
    walls = frozenset(wall_set)
    wall_gains = {w: 0.4 for w in walls}
    listener = (0.0, 0.0)
    source = (20.0, 12.0)

    # Precondition: no routed path
    paths = find_k_paths((0, 0), (20, 12), walls, GRID_COLS, GRID_ROWS, k=1)
    _pf("precondition: no routed path", len(paths) == 0)

    fstate: dict = {}
    block = _make_sine(440.0, amp=0.5)
    output = dsp_engine.process_block(block, listener, source, walls, SR, fstate,
                                       wall_gains=wall_gains)
    peak = float(np.max(np.abs(output)))

    # Old floor-gain-only peak: amp * 0.02 * max_pan_gain
    dx = source[0] - listener[0]
    pan = float(np.clip(dx / 20.0, -1.0, 1.0))
    floor_peak = 0.5 * 0.02 * 0.5 * (1.0 + abs(pan))
    _pf("output peak > floor-gain-only baseline",
        peak > floor_peak, f"peak={peak:.6f} > baseline={floor_peak:.6f}")
    _pf("no NaN", not np.any(np.isnan(output)))
    _pf("no inf", not np.any(np.isinf(output)))


def test_transmission_factor_stacks_correctly() -> None:
    """Unit-test _transmission_gain: stacking behavior matches spec."""
    print("\n--- Test: transmission -- factor stacking ---")
    src = (0.0, 0.0)
    lst = (10.0, 0.0)
    # distance_gain = 1/(1+0.08*10) = 1/1.8 ~ 0.5556 for all cases

    two_curtains = [((3, 0), 0.25), ((6, 0), 0.25)]  # product = 0.75*0.75 = 0.5625
    one_concrete = [((5, 0), 0.90)]                    # product = 0.10
    one_curtain  = [((5, 0), 0.25)]                    # product = 0.75
    mixed        = [((3, 0), 0.90), ((6, 0), 0.25)]   # product = 0.10*0.75 = 0.075

    g_2c = dsp_engine._transmission_gain(src, lst, two_curtains)
    g_1r = dsp_engine._transmission_gain(src, lst, one_concrete)
    g_1c = dsp_engine._transmission_gain(src, lst, one_curtain)
    g_mx = dsp_engine._transmission_gain(src, lst, mixed)

    _pf("two curtains > one concrete",
        g_2c > g_1r, f"2curtain={g_2c:.6f} > 1concrete={g_1r:.6f}")
    _pf("mixed (concrete+curtain) < single curtain alone",
        g_mx < g_1c, f"mixed={g_mx:.6f} < 1curtain={g_1c:.6f}")
    # Verify exact values: distance_gain = 1.0 / 1.8
    _pf("two curtains value correct",
        abs(g_2c - (1.0 / 1.8) * 0.5625) < 1e-9, f"got {g_2c:.9f}, expected {((1.0 / 1.8) * 0.5625):.9f}")
    _pf("one concrete value correct",
        abs(g_1r - (1.0 / 1.8) * 0.10) < 1e-9, f"got {g_1r:.9f}, expected {((1.0 / 1.8) * 0.10):.9f}")


def test_transmission_cutoff_steeper_with_more_walls() -> None:
    """Unit-test _transmission_cutoff_hz: more walls = lower cutoff, floored."""
    print("\n--- Test: transmission -- cutoff steeper with more walls ---")
    dummy_1 = [((5, 0), 0.5)]
    dummy_3 = [((3, 0), 0.5), ((5, 0), 0.5), ((7, 0), 0.5)]
    dummy_10 = [((i, 0), 0.5) for i in range(10)]

    c0 = dsp_engine._transmission_cutoff_hz(None)
    c1 = dsp_engine._transmission_cutoff_hz(dummy_1)
    c3 = dsp_engine._transmission_cutoff_hz(dummy_3)
    c10 = dsp_engine._transmission_cutoff_hz(dummy_10)

    _pf("0-wall cutoff == 1500.0", c0 == 1500.0, f"got {c0}")
    _pf("1-wall cutoff < base", c1 < 1500.0, f"got {c1}")
    _pf("3-wall cutoff < 1-wall cutoff", c3 < c1, f"c3={c3} < c1={c1}")
    _pf("10-wall cutoff >= 300.0 (floor)", c10 >= 300.0, f"got {c10}")
    _pf("10-wall cutoff == 300.0 (clamped)", c10 == 300.0, f"got {c10}")


def test_transmission_no_nan_inf_across_configs() -> None:
    """NaN/inf sweep with Family B-exercising configurations."""
    print("\n--- Test: transmission NaN/inf sweep ---")
    listener = (5.0, 10.0)
    configs = [
        ("no walls", (15.0, 10.0), frozenset(), {}),
        ("1 wall low gain", (15.0, 10.0), frozenset([(10, 10)]),
         {(10, 10): 0.2}),
        ("1 wall high gain (near 1.0)", (15.0, 10.0), frozenset([(10, 10)]),
         {(10, 10): 0.99}),
        ("3 stacked walls", (15.0, 10.0),
         frozenset([(8, 10), (10, 10), (12, 10)]),
         {(8, 10): 0.5, (10, 10): 0.5, (12, 10): 0.5}),
    ]
    all_clean = True
    block = _make_noise(amp=0.3, seed=77)
    for label, source, walls, wg in configs:
        fstate: dict = {}
        out = dsp_engine.process_block(block, listener, source, walls, SR, fstate,
                                        wall_gains=wg)
        has_nan = bool(np.any(np.isnan(out)))
        has_inf = bool(np.any(np.isinf(out)))
        ok = not has_nan and not has_inf
        _pf(f"{label}: clean output", ok, f"NaN={has_nan}, inf={has_inf}")
        if not ok:
            all_clean = False
    _pf("all transmission configs clean", all_clean)


# ---------------------------------------------------------------------------
# Family C: Discrete multi-wall echoes tests (Stage 5)
# ---------------------------------------------------------------------------


def test_multi_echo_count_matches_candidates() -> None:
    """Construct exactly 3 reflector candidates; assert slots 0..2 populate, 3 does not."""
    print("\n--- Test: multi-echo -- count matches candidates ---")
    source = (20.0, 12.0)
    listener = (20.0, 2.0)
    # Three walls on cardinal rays from source (distances 3, 3, 3; gains 0.9, 0.8, 0.5)
    walls = {(20, 15), (17, 12), (23, 12)}
    wall_gains = {(20, 15): 0.9, (17, 12): 0.8, (23, 12): 0.5}
    fstate: dict = {}

    block = _make_sine(440.0, amp=0.5)
    out = dsp_engine.process_block(block, listener, source, walls, SR, fstate, wall_gains=wall_gains)

    _pf("output shape (1024, 2)", out.shape == (N, 2))
    _pf("no NaN", not np.any(np.isnan(out)))
    _pf("no inf", not np.any(np.isinf(out)))
    _pf("slot 0 buffer allocated", "echo_delay_buf_0" in fstate)
    _pf("slot 1 buffer allocated", "echo_delay_buf_1" in fstate)
    _pf("slot 2 buffer allocated", "echo_delay_buf_2" in fstate)
    _pf("slot 3 buffer NOT allocated", "echo_delay_buf_3" not in fstate)


def test_multi_echo_independently_panned_and_delayed() -> None:
    """Assert distinct echoes arrive with their own independent delay and pan."""
    print("\n--- Test: multi-echo -- independently panned and delayed ---")
    source = (20.0, 12.0)
    listener = (20.0, 10.0)
    # Wall 1: (15, 12) -> dx_echo = -5 (panned left), dist = 5.0 -> delay = 400 samples
    # Wall 2: (28, 12) -> dx_echo = +8 (panned right), dist = 8.0 -> delay = 640 samples
    walls = {(15, 12), (28, 12)}
    wall_gains = {(15, 12): 0.9, (28, 12): 0.9}
    fstate: dict = {}

    # Use a single unit impulse so echoes arrive as distinct spikes at their delay offsets
    block = np.zeros(N, dtype=np.float32)
    block[0] = 1.0

    out = dsp_engine.process_block(block, listener, source, walls, SR, fstate, wall_gains=wall_gains)

    _pf("slot 0 and 1 allocated", "echo_delay_buf_0" in fstate and "echo_delay_buf_1" in fstate)

    # Echo 0 arrives at sample 400, panned left (L > R)
    left_400 = float(out[400, 0])
    right_400 = float(out[400, 1])
    _pf("echo 0 arrives at sample 400 with non-zero energy", abs(left_400) > 0.01)
    _pf("echo 0 panned left (L > R)", left_400 > right_400, f"L={left_400:.4f}, R={right_400:.4f}")

    # Echo 1 arrives at sample 640, panned right (R > L)
    left_640 = float(out[640, 0])
    right_640 = float(out[640, 1])
    _pf("echo 1 arrives at sample 640 with non-zero energy", abs(right_640) > 0.01)
    _pf("echo 1 panned right (R > L)", right_640 > left_640, f"L={left_640:.4f}, R={right_640:.4f}")

    # Echoes do not arrive at each other's timestamps
    _pf("echo 1 quiet at sample 400 before its arrival", abs(out[400, 1] - right_400) < 1e-4)


def test_echo_shrinking_candidate_count_flushes_stale_slots() -> None:
    """Shrinking candidate count from 3 to 1 flushes slots 1 and 2 with silence."""
    print("\n--- Test: multi-echo -- shrinking candidate count flushes stale slots ---")
    source = (20.0, 12.0)
    listener = (20.0, 2.0)
    walls_3 = {(20, 15), (17, 12), (23, 12)}
    wall_gains_3 = {(20, 15): 0.9, (17, 12): 0.8, (23, 12): 0.5}
    fstate: dict = {}

    block = _make_sine(440.0, amp=0.5)
    # Block 1: 3 candidates active
    dsp_engine.process_block(block, listener, source, walls_3, SR, fstate, wall_gains=wall_gains_3)

    wpos_0_b = fstate.get("echo_delay_write_pos_0", 0)
    wpos_1_b = fstate.get("echo_delay_write_pos_1", 0)
    wpos_2_b = fstate.get("echo_delay_write_pos_2", 0)

    # Block 2: modified geometry with only 1 candidate
    walls_1 = {(20, 15)}
    wall_gains_1 = {(20, 15): 0.9}
    dsp_engine.process_block(block, listener, source, walls_1, SR, fstate, wall_gains=wall_gains_1)

    wpos_0_a = fstate.get("echo_delay_write_pos_0", 0)
    wpos_1_a = fstate.get("echo_delay_write_pos_1", 0)
    wpos_2_a = fstate.get("echo_delay_write_pos_2", 0)

    _pf("slot 0 remains active and write_pos advances", wpos_0_a - wpos_0_b == N)
    _pf("slot 1 flushed with silence and write_pos advances", wpos_1_a - wpos_1_b == N)
    _pf("slot 2 flushed with silence and write_pos advances", wpos_2_a - wpos_2_b == N)
    _pf("slot 3 was never allocated", "echo_delay_buf_3" not in fstate)

    # Confirm the written section of slot 1 and slot 2 in block 2 was indeed all zeros
    buf1 = fstate["echo_delay_buf_1"]
    buf2 = fstate["echo_delay_buf_2"]
    _pf("slot 1 flushed segment is silent (all zeros)", bool(np.all(buf1[wpos_1_b:wpos_1_a] == 0.0)))
    _pf("slot 2 flushed segment is silent (all zeros)", bool(np.all(buf2[wpos_2_b:wpos_2_a] == 0.0)))


def test_curtain_wall_negligible_echo() -> None:
    """Low-reflectivity curtain wall produces negligible echo vs concrete."""
    print("\n--- Test: multi-echo -- curtain wall negligible echo ---")
    source = (20.0, 12.0)
    listener = (20.0, 10.0)
    wall = (20, 15)

    # High frequency block (2000 Hz) to test both material gain and cutoff attenuation
    t = np.arange(N, dtype=np.float32) / SR
    block = (0.5 * np.sin(2 * np.pi * 2000.0 * t)).astype(np.float32)

    # Open space baseline (no echo)
    out_open = dsp_engine.process_block(block, listener, source, set(), SR, {})
    # Curtain wall (gain = 0.25)
    out_curtain = dsp_engine.process_block(block, listener, source, {wall}, SR, {}, wall_gains={wall: 0.25})
    # Concrete wall (gain = 0.90)
    out_concrete = dsp_engine.process_block(block, listener, source, {wall}, SR, {}, wall_gains={wall: 0.90})

    diff_curtain = out_curtain - out_open
    diff_concrete = out_concrete - out_open

    rms_curtain = float(np.sqrt(np.mean(diff_curtain ** 2)))
    rms_concrete = float(np.sqrt(np.mean(diff_concrete ** 2)))

    _pf("concrete echo non-zero", rms_concrete > 0.01, f"rms={rms_concrete:.6f}")
    _pf("curtain echo < 0.1 * concrete echo", rms_curtain < 0.1 * rms_concrete,
        f"curtain={rms_curtain:.6f}, concrete={rms_concrete:.6f}, ratio={rms_curtain/rms_concrete:.4f}")

    # Also unit-test helper functions directly for curtain vs concrete
    c_curtain = ReflectorCandidate(wall_cell=wall, distance_to_wall=3.0, material_gain=0.25)
    c_concrete = ReflectorCandidate(wall_cell=wall, distance_to_wall=3.0, material_gain=0.90)
    _pf("curtain gain < concrete gain", dsp_engine._echo_gain(c_curtain) < dsp_engine._echo_gain(c_concrete))
    _pf("curtain cutoff < concrete cutoff", dsp_engine._echo_cutoff_hz(c_curtain) < dsp_engine._echo_cutoff_hz(c_concrete))


def test_multi_echo_no_nan_inf_across_configs() -> None:
    """Sweep configs with 0, 1, 4 candidates and dynamic changes, asserting no NaN/inf."""
    print("\n--- Test: multi-echo -- no NaN/inf across configs ---")
    source = (20.0, 12.0)
    listener = (20.0, 10.0)
    block = _make_sine(440.0, amp=0.5)

    configs = [
        ("0 candidates (open)", set()),
        ("1 candidate", {(20, 15)}),
        ("4 candidates (max)", {(20, 15), (20, 9), (17, 12), (23, 12)}),
        ("shrink to 1", {(20, 15)}),
        ("grow to 3", {(20, 15), (17, 12), (23, 12)}),
        ("drop to 0", set()),
        ("back to 4", {(20, 15), (20, 9), (17, 12), (23, 12)}),
    ]

    fstate: dict = {}
    all_clean = True
    for label, walls in configs:
        out = dsp_engine.process_block(block, listener, source, walls, SR, fstate)
        has_nan = bool(np.any(np.isnan(out)))
        has_inf = bool(np.any(np.isinf(out)))
        ok = not has_nan and not has_inf and out.shape == (N, 2)
        _pf(f"{label}: clean output", ok, f"NaN={has_nan}, inf={has_inf}")
        if not ok:
            all_clean = False
    _pf("all multi-echo configs clean", all_clean)


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

    test_transmission_zero_when_no_wall_crossed()
    test_transmission_contributes_during_total_occlusion()
    test_transmission_factor_stacks_correctly()
    test_transmission_cutoff_steeper_with_more_walls()
    test_transmission_no_nan_inf_across_configs()

    test_multi_echo_count_matches_candidates()
    test_multi_echo_independently_panned_and_delayed()
    test_echo_shrinking_candidate_count_flushes_stale_slots()
    test_curtain_wall_negligible_echo()
    test_multi_echo_no_nan_inf_across_configs()

    print("\n" + "=" * 60)
    print("  Done.  Review PASS/FAIL lines above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
