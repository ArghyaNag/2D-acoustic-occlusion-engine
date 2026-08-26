"""
dsp_engine.py — The DSP black-box module.

This module is COMPLETELY ISOLATED from Pygame and sounddevice.
Its only dependencies are NumPy and SciPy.

It implements the spatial-audio transformation for a single source:
  1. Distance-based gain attenuation (inverse-distance law).
  2. Stereo panning based on relative x-offset (linear pan).
  3. Wall-obstruction lowpass filtering (Bresenham ray + biquad filter).

Processing order (documented choice):
  input → wall lowpass filter → distance gain → stereo pan → output
  Filtering before panning means the lowpass is applied once on the mono
  signal rather than twice on the stereo pair — cheaper and equivalent
  because the filter is linear.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


# ---------------------------------------------------------------------------
# Bresenham's line algorithm (grid-cell traversal)
# ---------------------------------------------------------------------------
def _bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Return the list of integer grid cells on the line from (x0,y0) to (x1,y1)."""
    cells: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return cells


def _line_intersects_walls(
    listener: tuple[float, float],
    source: tuple[float, float],
    walls: set[tuple[int, int]] | frozenset[tuple[int, int]],
) -> bool:
    """Check whether the straight line between *listener* and *source*
    passes through any cell in *walls* (excluding the endpoints
    themselves, so standing on a wall doesn't self-muffle)."""
    lx, ly = int(round(listener[0])), int(round(listener[1]))
    sx, sy = int(round(source[0])), int(round(source[1]))
    cells = _bresenham(lx, ly, sx, sy)
    # Exclude the listener and source cells themselves
    endpoints = {(lx, ly), (sx, sy)}
    for cell in cells:
        if cell not in endpoints and cell in walls:
            return True
    return False


# ---------------------------------------------------------------------------
# Biquad filter design (cached per sample_rate to avoid recomputation)
# ---------------------------------------------------------------------------
_cached_sos: dict[int, np.ndarray] = {}

# Wall-obstruction lowpass cutoff frequency.
_WALL_LPF_CUTOFF_HZ: int = 800
_WALL_LPF_ORDER: int = 2


def _get_wall_sos(sample_rate: int) -> np.ndarray:
    """Return the second-order-section coefficients for the wall lowpass filter."""
    if sample_rate not in _cached_sos:
        _cached_sos[sample_rate] = butter(
            _WALL_LPF_ORDER, _WALL_LPF_CUTOFF_HZ, btype="low", fs=sample_rate, output="sos"
        ).astype(np.float32)
    return _cached_sos[sample_rate]


# ---------------------------------------------------------------------------
# Public API — the swap boundary
# ---------------------------------------------------------------------------
def process_block(
    input_block: np.ndarray,          # (n_frames,) mono float32
    listener_pos: tuple[float, float],
    source_pos: tuple[float, float],
    walls: set[tuple[int, int]] | frozenset[tuple[int, int]],
    sample_rate: int,
    filter_state: dict,               # persistent per-source state, mutated in-place
) -> np.ndarray:                      # (n_frames, 2) stereo float32
    """Transform a mono source block into a stereo output block with
    distance attenuation, stereo pan, and wall-obstruction filtering.

    Parameters
    ----------
    input_block : np.ndarray
        Mono float32 audio chunk for one source, shape ``(n_frames,)``.
    listener_pos, source_pos : tuple[float, float]
        Grid-coordinate positions.
    walls : set or frozenset of (int, int)
        Set of grid cells that are walls.
    sample_rate : int
        Audio sample rate in Hz.
    filter_state : dict
        Mutable dict persisted across calls for this source.  Used to
        store biquad filter memory (``sos_zi``) so the lowpass filter
        doesn't click/pop when toggled.

    Returns
    -------
    np.ndarray
        Stereo float32 output, shape ``(n_frames, 2)``.
    """
    n_frames = input_block.shape[0]
    mono = input_block.astype(np.float32, copy=True)

    # ---- 1. Wall-obstruction lowpass filter ---------------------------
    sos = _get_wall_sos(sample_rate)
    obstructed = _line_intersects_walls(listener_pos, source_pos, walls)

    # Ensure filter state exists
    if "sos_zi" not in filter_state:
        # sosfilt_zi returns shape (n_sections, 2); scale by 0 for silence-start
        zi_template = sosfilt_zi(sos).astype(np.float32)
        filter_state["sos_zi"] = zi_template * 0.0

    if obstructed:
        # Apply the lowpass filter with persisted state
        mono, filter_state["sos_zi"] = sosfilt(
            sos, mono, zi=filter_state["sos_zi"]
        )
        mono = mono.astype(np.float32)
    else:
        # Even when unobstructed, run the filter at unity to keep zi in
        # sync, preventing clicks if a wall appears next block.
        # We accomplish this by decaying the filter state toward zero
        # over this block length so the transition is smooth.
        # Simple approach: run a "bypass" by feeding the signal through
        # an allpass (identity) — but we can't change the SOS coeffs
        # without redesigning.  Instead, we just reset the zi gradually:
        # zero it out.  Since the filter wasn't active, a cold-start
        # next time is acceptable — the first filtered block will ramp
        # in naturally because sosfilt_zi provides a steady-state init.
        zi_template = sosfilt_zi(sos).astype(np.float32)
        filter_state["sos_zi"] = zi_template * 0.0

    # ---- 2. Distance gain (inverse-distance law) ----------------------
    dx = source_pos[0] - listener_pos[0]
    dy = source_pos[1] - listener_pos[1]
    distance = math.sqrt(dx * dx + dy * dy)
    gain = 1.0 / (1.0 + 0.15 * distance)
    mono *= gain

    # ---- 3. Stereo pan (linear, based on relative x-offset) -----------
    # TODO: upgrade to equal-power panning
    # Normalize dx to [-1, 1] range.  We use a reasonable max range (half
    # the grid width) so that sources at the edge are fully panned.
    max_offset = 20.0  # half of GRID_COLS=40
    pan = np.clip(dx / max_offset, -1.0, 1.0)  # -1 = full left, +1 = full right

    # Convert to per-channel gain: center → (0.5, 0.5)
    right_gain = 0.5 * (1.0 + pan)
    left_gain = 0.5 * (1.0 - pan)

    # ---- 4. Combine into stereo output --------------------------------
    out = np.empty((n_frames, 2), dtype=np.float32)
    out[:, 0] = mono * left_gain
    out[:, 1] = mono * right_gain

    return out
