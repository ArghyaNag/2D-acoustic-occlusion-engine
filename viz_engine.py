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


def compute_spectrum(
    data: np.ndarray,
    sample_rate: int,
    num_bins: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the magnitude spectrum of ``data`` bucketed into frequency bands.

    Uses ``numpy.fft.rfft`` (real FFT), matching the approach used in
    dsp_engine.py's ``_fft_lowpass``.

    Bucketing
    ---------
    Frequency bands are **logarithmically spaced** between 40 Hz and the
    Nyquist frequency (``sample_rate / 2``).  Log spacing concentrates
    resolution in the low-to-mid frequency range where dsp_engine.py's
    corner-driven lowpass cutoff (300–4000 Hz) operates, making the
    filter's effect visually obvious in a bar graph.

    Within each bucket the **peak** (maximum) FFT-bin magnitude is taken.
    This preserves tonal peaks that would otherwise be diluted by
    averaging across a wide bucket.

    Normalization
    -------------
    Raw FFT magnitudes are divided by ``N / 2`` (where N = len(data)) so
    that a unit-amplitude sine wave produces a peak bucket value of
    approximately 1.0.  Returned magnitudes are non-negative float32
    values suitable for direct bar-height mapping (clip to [0, 1] in the
    renderer if needed).

    Parameters
    ----------
    data : np.ndarray
        Mono float32 audio block, shape ``(n_frames,)``.
    sample_rate : int
        Audio sample rate in Hz.
    num_bins : int
        Number of output frequency buckets (default 32).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(bin_centers_hz, magnitudes)`` — each a 1-D float32 array of
        length ``num_bins``.  Returns two empty float32 arrays if
        ``data`` is None or empty.
    """
    if data is None or len(data) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    data_arr = np.asarray(data, dtype=np.float32)
    n = len(data_arr)
    if n < 2 or num_bins <= 0 or sample_rate <= 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    nyquist = float(sample_rate) / 2.0

    # Full rfft magnitude, normalized so amp-1.0 sine ≈ 1.0 peak
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    fft_mag = np.abs(np.fft.rfft(data_arr)) / (n / 2.0)

    # Log-spaced bucket edges from 40 Hz to Nyquist
    min_freq = 40.0
    edges = np.geomspace(min_freq, nyquist, num_bins + 1)
    # Geometric-mean centre of each bucket
    bin_centers = np.sqrt(edges[:-1] * edges[1:]).astype(np.float32)
    magnitudes = np.zeros(num_bins, dtype=np.float32)

    for i in range(num_bins):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        if np.any(mask):
            magnitudes[i] = float(np.max(fft_mag[mask]))
        else:
            # Bucket narrower than FFT resolution — interpolate
            magnitudes[i] = float(np.interp(bin_centers[i], freqs, fft_mag))

    return bin_centers, magnitudes

