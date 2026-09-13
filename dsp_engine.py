from __future__ import annotations
import numpy as np
from pathfinding import find_k_paths
from shared_state import GRID_COLS, GRID_ROWS
_TOTAL_OCCLUSION_FLOOR_GAIN: float = 0.02
_BASE_CUTOFF_HZ: float = 4000.0
_CUTOFF_DROP_PER_CORNER_HZ: float = 900.0
_MIN_CUTOFF_HZ: float = 300.0
_TRANSITION_BW_HZ: float = 500.0
_SAMPLES_PER_GRID_UNIT: int = 40
_MAX_DELAY_SAMPLES: int = 8000

def _has_line_of_sight(p0: tuple[int, int], p1: tuple[int, int], walls: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> bool:
    if not walls:
        return True
    x0, y0 = p0
    x1, y1 = p1
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = (x0, y0)
    while True:
        if (x, y) != (x0, y0) and (x, y) != (x1, y1):
            if (x, y) in walls:
                return False
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return True

def _corner_cutoff_hz(corners: int) -> float:
    return max(_MIN_CUTOFF_HZ, _BASE_CUTOFF_HZ - corners * _CUTOFF_DROP_PER_CORNER_HZ)

def _primary_cutoff_hz(listener_cell: tuple[int, int], source_cell: tuple[int, int], corners: int, walls: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> float:
    if _has_line_of_sight(listener_cell, source_cell, walls):
        return _BASE_CUTOFF_HZ
    return _corner_cutoff_hz(corners)

def _fft_lowpass(mono: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
    n = len(mono)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    spectrum = np.fft.rfft(mono)
    half_bw = _TRANSITION_BW_HZ / 2.0
    low = max(0.0, cutoff_hz - half_bw)
    high = cutoff_hz + half_bw
    bw = high - low
    gains = np.ones(len(freqs), dtype=np.float64)
    transition_mask = (freqs > low) & (freqs < high)
    if np.any(transition_mask):
        gains[transition_mask] = 0.5 * (1.0 + np.cos(np.pi * (freqs[transition_mask] - low) / max(1e-09, bw)))
    gains[freqs >= high] = 0.0
    spectrum *= gains
    return np.fft.irfft(spectrum, n=n).astype(np.float32)

def _delay_line_process(samples: np.ndarray, delay_samples: int, filter_state: dict, n_frames: int) -> np.ndarray:
    buf_key = 'reflection_delay_buf'
    wpos_key = 'reflection_delay_write_pos'
    if buf_key not in filter_state:
        buf_size = _MAX_DELAY_SAMPLES + max(n_frames, 2048)
        filter_state[buf_key] = np.zeros(buf_size, dtype=np.float32)
        filter_state[wpos_key] = 0
    buf = filter_state[buf_key]
    buf_size = len(buf)
    write_pos = filter_state[wpos_key]
    write_idx = (np.arange(n_frames) + write_pos) % buf_size
    buf[write_idx] = samples[:n_frames]
    read_idx = (np.arange(n_frames) + write_pos - delay_samples) % buf_size
    delayed = buf[read_idx].copy()
    filter_state[wpos_key] = (write_pos + n_frames) % buf_size
    return delayed

def process_block(input_block: np.ndarray, listener_pos: tuple[float, float], source_pos: tuple[float, float], walls: set[tuple[int, int]] | frozenset[tuple[int, int]], sample_rate: int, filter_state: dict) -> np.ndarray:
    n_frames = input_block.shape[0]
    mono = input_block.astype(np.float32, copy=True)
    listener_cell = (int(round(listener_pos[0])), int(round(listener_pos[1])))
    source_cell = (int(round(source_pos[0])), int(round(source_pos[1])))
    paths = find_k_paths(start=listener_cell, end=source_cell, walls=walls, grid_cols=GRID_COLS, grid_rows=GRID_ROWS, k=2)
    if not paths:
        mono *= _TOTAL_OCCLUSION_FLOOR_GAIN
        if 'reflection_delay_buf' in filter_state:
            _delay_line_process(np.zeros(n_frames, dtype=np.float32), 0, filter_state, n_frames)
        dx = source_pos[0] - listener_pos[0]
        max_offset = 20.0
        pan = np.clip(dx / max_offset, -1.0, 1.0)
        right_gain = 0.5 * (1.0 + pan)
        left_gain = 0.5 * (1.0 - pan)
        out = np.empty((n_frames, 2), dtype=np.float32)
        out[:, 0] = mono * left_gain
        out[:, 1] = mono * right_gain
        return out
    nyquist = sample_rate / 2.0
    primary_path = paths[0]
    has_reflection = len(paths) >= 2
    if has_reflection:
        primary_mono = mono.copy()
    else:
        primary_mono = mono
    primary_cutoff = _primary_cutoff_hz(listener_cell, source_cell, primary_path.corners, walls)
    if primary_cutoff < nyquist:
        primary_mono = _fft_lowpass(primary_mono, primary_cutoff, sample_rate)
    primary_gain = 1.0 / (1.0 + 0.15 * primary_path.length)
    primary_mono *= primary_gain
    primary_cells = primary_path.cells
    if len(primary_cells) >= 2:
        lookahead = min(4, len(primary_cells) - 1)
        dx_primary = float(primary_cells[lookahead][0] - primary_cells[0][0])
        primary_scale = float(lookahead)
    else:
        dx_primary = 0.0
        primary_scale = 1.0
    pan = np.clip(dx_primary / max(1.0, primary_scale), -1.0, 1.0)
    primary_right = 0.5 * (1.0 + pan)
    primary_left = 0.5 * (1.0 - pan)
    out = np.empty((n_frames, 2), dtype=np.float32)
    out[:, 0] = primary_mono * primary_left
    out[:, 1] = primary_mono * primary_right
    if has_reflection:
        ref_path = paths[1]
        ref_mono = mono
        ref_cutoff = _corner_cutoff_hz(ref_path.corners)
        if ref_cutoff < nyquist:
            ref_mono = _fft_lowpass(ref_mono, ref_cutoff, sample_rate)
        ref_gain_val = 1.0 / (1.0 + 0.15 * ref_path.length)
        ref_mono = ref_mono * ref_gain_val
        delay_samples = int(round((ref_path.length - primary_path.length) * _SAMPLES_PER_GRID_UNIT))
        delay_samples = max(0, min(delay_samples, _MAX_DELAY_SAMPLES))
        ref_delayed = _delay_line_process(ref_mono, delay_samples, filter_state, n_frames)
        cells = ref_path.cells
        if len(cells) >= 2:
            lookahead = min(4, len(cells) - 1)
            dx_ref = float(cells[lookahead][0] - cells[0][0])
            ref_scale = float(lookahead)
        else:
            dx_ref = 0.0
            ref_scale = 1.0
        ref_pan = np.clip(dx_ref / max(1.0, ref_scale), -1.0, 1.0)
        ref_right = 0.5 * (1.0 + ref_pan)
        ref_left = 0.5 * (1.0 - ref_pan)
        out[:, 0] += ref_delayed * ref_left
        out[:, 1] += ref_delayed * ref_right
    elif 'reflection_delay_buf' in filter_state:
        _delay_line_process(np.zeros(n_frames, dtype=np.float32), 0, filter_state, n_frames)
    return out
