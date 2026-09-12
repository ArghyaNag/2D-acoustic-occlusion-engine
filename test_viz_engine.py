"""
test_viz_engine.py -- verification script for viz_engine.py

Plain print() PASS/FAIL style, matching this project's test conventions.
"""

import numpy as np
from viz_engine import compute_spectrum

SAMPLE_RATE = 44100


def main() -> None:
    print("=" * 60)
    print("  viz_engine.py -- verification script")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Test 1: None input
    # ------------------------------------------------------------------
    print("\n--- Test: None input ---")
    c, m = compute_spectrum(None, SAMPLE_RATE, 32)
    ok = len(c) == 0 and len(m) == 0
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] None input returns empty arrays  (len c={len(c)}, m={len(m)})")

    # ------------------------------------------------------------------
    # Test 2: Empty input
    # ------------------------------------------------------------------
    print("\n--- Test: empty input ---")
    c, m = compute_spectrum(np.array([], dtype=np.float32), SAMPLE_RATE, 32)
    ok = len(c) == 0 and len(m) == 0
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] empty input returns empty arrays  (len c={len(c)}, m={len(m)})")

    # ------------------------------------------------------------------
    # Test 3: Pure sine tone — peak bin near tone frequency
    # ------------------------------------------------------------------
    print("\n--- Test: sine tone 1000 Hz ---")
    n_frames = 2048
    t = np.arange(n_frames, dtype=np.float32) / SAMPLE_RATE
    tone_hz = 1000.0
    sine = np.sin(2 * np.pi * tone_hz * t).astype(np.float32)
    c, m = compute_spectrum(sine, SAMPLE_RATE, 32)

    ok = len(c) == 32 and len(m) == 32
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] returns 32 bins  (len c={len(c)}, m={len(m)})")

    peak_idx = int(np.argmax(m))
    peak_freq = float(c[peak_idx])
    peak_mag = float(m[peak_idx])

    # Peak bin should be close to 1000 Hz (within a factor of ~2 given
    # log-spaced bucketing)
    ok = 500.0 < peak_freq < 2000.0
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] peak bin near 1000 Hz  (got {peak_freq:.1f} Hz)")

    # Peak magnitude should be meaningfully higher than the median of
    # other bins (docstring says unit-amplitude sine ≈ 1.0 peak)
    other_mags = np.delete(m, peak_idx)
    median_other = float(np.median(other_mags))
    ok = peak_mag > median_other * 3.0
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] peak magnitude >> median of others  "
          f"(peak={peak_mag:.4f}, median_others={median_other:.4f})")

    # Magnitude ~= 1.0 for unit-amplitude sine (within reasonable tolerance;
    # log-bucket boundaries may split energy, yielding ~0.7 for short blocks)
    ok = 0.3 < peak_mag < 1.5
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] peak magnitude ~= 1.0  (got {peak_mag:.4f})")

    # ------------------------------------------------------------------
    # Test 4: White noise — no NaN/inf, energy spread across bins
    # ------------------------------------------------------------------
    print("\n--- Test: white noise ---")
    rng = np.random.default_rng(42)
    noise = rng.uniform(-1.0, 1.0, 4096).astype(np.float32)
    c, m = compute_spectrum(noise, SAMPLE_RATE, 32)

    ok = len(c) == 32 and len(m) == 32
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] returns 32 bins  (len c={len(c)}, m={len(m)})")

    has_nan = bool(np.any(np.isnan(m)))
    tag = "PASS" if not has_nan else "FAIL"
    print(f"  [{tag}] no NaN in magnitudes  (NaN={has_nan})")

    has_inf = bool(np.any(np.isinf(m)))
    tag = "PASS" if not has_inf else "FAIL"
    print(f"  [{tag}] no inf in magnitudes  (inf={has_inf})")

    # White noise should have energy in most bins (at least 80% nonzero)
    nonzero_frac = float(np.count_nonzero(m)) / len(m)
    ok = nonzero_frac >= 0.8
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] energy spread across bins  "
          f"(nonzero fraction={nonzero_frac:.2f})")

    # ------------------------------------------------------------------
    # Test 5: bin_centers_hz range — near 40 Hz to near Nyquist
    # ------------------------------------------------------------------
    print("\n--- Test: bin_centers_hz range ---")
    c, m = compute_spectrum(noise, SAMPLE_RATE, 32)
    min_center = float(c[0])
    max_center = float(c[-1])
    nyquist = SAMPLE_RATE / 2.0

    # Lowest bin center should be near 40 Hz (within factor of ~2)
    ok = 30.0 < min_center < 120.0
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] lowest bin center near 40 Hz  (got {min_center:.1f} Hz)")

    # Highest bin center should approach Nyquist (within factor of ~2)
    ok = max_center > nyquist * 0.5
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] highest bin center near Nyquist ({nyquist:.0f} Hz)  "
          f"(got {max_center:.1f} Hz)")

    # Bin centers should be monotonically increasing
    ok = bool(np.all(np.diff(c) > 0))
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] bin centers monotonically increasing")

    print("\n" + "=" * 60)
    print("  Done.  Review PASS/FAIL lines above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
