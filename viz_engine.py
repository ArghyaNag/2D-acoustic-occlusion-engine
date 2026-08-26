"""
viz_engine.py — Isolated waveform visualization utilities.

This module is COMPLETELY ISOLATED from Pygame, sounddevice, and threading.
Its only dependency is NumPy.

It provides fixed-multiplier display scaling and width-downsampling for graph plotting.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Display multiplier constants
# ---------------------------------------------------------------------------
# Raw audio loaded from files stays unscaled (1.0).
RAW_DISPLAY_MULTIPLIER: float = 1.0

# Fixed multiplier chosen by ear/eye during testing, not derived from the DSP formula.
PROCESSED_DISPLAY_MULTIPLIER: float = 2.5


def compute_display_samples(
    data: np.ndarray,
    multiplier: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Scale `data` by a flat multiplier and clip to [-1.0, 1.0] for plotting.

    Return the scaled array and the actual pre-scaling peak amplitude for numeric display.
    """
    if data is None or len(data) == 0:
        return np.array([], dtype=np.float32), 0.0

    data_arr = np.asarray(data, dtype=np.float32)
    raw_peak = float(np.abs(data_arr).max())
    scaled = np.clip(data_arr * float(multiplier), -1.0, 1.0)

    return scaled, raw_peak


def downsample_for_width(
    data: np.ndarray,
    target_width: int,
) -> np.ndarray:
    """Downsample a 1D float32 array to at most `target_width` points for plotting."""
    if data is None or len(data) == 0 or target_width <= 0:
        return np.array([], dtype=np.float32)

    data_arr = np.asarray(data, dtype=np.float32)
    n = len(data_arr)
    if n <= target_width:
        return data_arr

    step = max(1, n // target_width)
    return data_arr[::step][:target_width]
