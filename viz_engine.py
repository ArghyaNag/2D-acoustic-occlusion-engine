from __future__ import annotations
import numpy as np
RAW_DISPLAY_MULTIPLIER: float = 1.0
PROCESSED_DISPLAY_MULTIPLIER: float = 2.5

def compute_display_samples(data: np.ndarray, multiplier: float=1.0) -> tuple[np.ndarray, float]:
    if data is None or len(data) == 0:
        return (np.array([], dtype=np.float32), 0.0)
    data_arr = np.asarray(data, dtype=np.float32)
    raw_peak = float(np.abs(data_arr).max())
    scaled = np.clip(data_arr * float(multiplier), -1.0, 1.0)
    return (scaled, raw_peak)

def downsample_for_width(data: np.ndarray, target_width: int) -> np.ndarray:
    if data is None or len(data) == 0 or target_width <= 0:
        return np.array([], dtype=np.float32)
    data_arr = np.asarray(data, dtype=np.float32)
    n = len(data_arr)
    if n <= target_width:
        return data_arr
    step = max(1, n // target_width)
    return data_arr[::step][:target_width]

def compute_spectrum(data: np.ndarray, sample_rate: int, num_bins: int=32) -> tuple[np.ndarray, np.ndarray]:
    if data is None or len(data) == 0:
        return (np.array([], dtype=np.float32), np.array([], dtype=np.float32))
    data_arr = np.asarray(data, dtype=np.float32)
    n = len(data_arr)
    if n < 2 or num_bins <= 0 or sample_rate <= 0:
        return (np.array([], dtype=np.float32), np.array([], dtype=np.float32))
    nyquist = float(sample_rate) / 2.0
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    fft_mag = np.abs(np.fft.rfft(data_arr)) / (n / 2.0)
    min_freq = 40.0
    edges = np.geomspace(min_freq, nyquist, num_bins + 1)
    bin_centers = np.sqrt(edges[:-1] * edges[1:]).astype(np.float32)
    magnitudes = np.zeros(num_bins, dtype=np.float32)
    for i in range(num_bins):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        if np.any(mask):
            magnitudes[i] = float(np.max(fft_mag[mask]))
        else:
            magnitudes[i] = float(np.interp(bin_centers[i], freqs, fft_mag))
    return (bin_centers, magnitudes)
