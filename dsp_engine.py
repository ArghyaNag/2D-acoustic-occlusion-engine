"""
dsp_engine.py -- The DSP black-box module.

This module is COMPLETELY ISOLATED from Pygame and sounddevice.
Its only dependency beyond the standard library is NumPy (used for FFT
and array maths).  Pathfinding is delegated to the project's own
pathfinding module (pure Python).

It implements the spatial-audio transformation for a single source using
up to TWO sound arrivals per block:
  1. PRIMARY ARRIVAL (shortest path): path-aware FFT lowpass filtering
     (cutoff driven by corner count), path-length-based gain attenuation
     (inverse-distance law), and stereo panning based on the straight-
     line source-to-listener x-offset.
  2. REFLECTED ARRIVAL (second-shortest path, when one exists): same
     filtering and gain formulas parameterized by the alternate path's
     own corner count and length, a delay proportional to the extra path
     length beyond the primary, and stereo panning based on the final
     segment direction of the reflected path (representing the direction
     the reflected sound approaches the source from -- a simplified
     directional approximation).

The two arrivals are independently filtered, gained, delayed, panned,
and then summed into the final stereo output.  No additional global
rescaling is applied after summing -- per-arrival gains already attenuate
each contribution by distance.  For a single source very close to the
listener with two nearly-equal-length paths, the summed output could
theoretically approach ~2x the old single-arrival amplitude; in practice
the reflected path is always longer and more filtered, so the addition
is modest.  audio_engine.py's downstream tanh soft-clipper handles
multi-source mixing, but only activates for multiple SOURCES (not
multiple arrivals within one source), so extreme single-source levels
are possible but unlikely in normal gameplay.

When no path exists (total occlusion), a small floor gain is applied.
When only one path exists, only the primary arrival is produced.

Processing order per arrival:
  input -> FFT lowpass (corner-driven cutoff) -> distance gain
  -> [delay, reflected only] -> stereo pan -> sum both -> output

FFT filtering approach -- per-block with smooth taper, no overlap:
  Each mono block is transformed with numpy's rfft, multiplied by a
  raised-cosine gain curve that tapers smoothly from 1.0 (passband) to
  0.0 (stopband) around the cutoff frequency, then inverse-transformed
  with irfft.  This is applied independently to each block WITHOUT
  overlap-add / overlap-save.

  Justification for omitting overlap handling:
    - The raised-cosine taper (500 Hz transition bandwidth) produces a
      short effective impulse response.  The smoother the frequency-
      domain rolloff, the faster the time-domain decay, so wrap-around
      from circular convolution (inherent in per-block FFT processing)
      affects only a small number of samples at each block boundary.
    - With 1024-sample blocks at 44100 Hz (~23 ms per block) and a
      500 Hz transition, the effective filter length is roughly
      2 / 500 * 44100 ~ 176 samples -- short relative to the 1024-
      sample block, so wrap-around artifacts are small in magnitude.
    - The cutoff frequency changes gradually (only when the listener,
      source, or walls move between frames), not abruptly every block.
    - This is a course project prioritising correctness-of-concept over
      broadcast-quality audio.
    - The dramatically simpler implementation (no inter-block state for
      overlap buffers) reduces bug surface area and is easier to explain
      in a viva.

Tunable constants (all empirical -- adjust by ear, not derived from any
acoustic model, matching the existing convention in this file where the
0.15 distance-gain constant and 20.0 max_offset pan constant were also
documented as empirical / tunable):
  _TOTAL_OCCLUSION_FLOOR_GAIN  -- gain when no path exists (~-34 dB)
  _BASE_CUTOFF_HZ              -- lowpass cutoff at 0 corners (direct LoS)
  _CUTOFF_DROP_PER_CORNER_HZ   -- cutoff reduction per path corner
  _MIN_CUTOFF_HZ               -- floor cutoff to prevent total HF silence
  _TRANSITION_BW_HZ            -- raised-cosine rolloff width in the FFT
  _SAMPLES_PER_GRID_UNIT       -- delay samples per unit of extra path length
  _MAX_DELAY_SAMPLES           -- maximum delay readback (bounds buffer size)
"""

from __future__ import annotations

import numpy as np

from pathfinding import find_k_paths
from shared_state import GRID_COLS, GRID_ROWS


# ---------------------------------------------------------------------------
# Tunable constants (empirical -- adjust by ear, not from an acoustic model)
# ---------------------------------------------------------------------------

# Total occlusion: when no path exists between listener and source, apply
# this floor gain (~-34 dB) to represent minor sound leakage through / around
# a sealed boundary.  Small non-zero value rather than hard zero.
_TOTAL_OCCLUSION_FLOOR_GAIN: float = 0.02

# Corner-driven lowpass cutoff parameters (all in Hz).
# Formula: cutoff = max(_MIN_CUTOFF_HZ,
#                       _BASE_CUTOFF_HZ - corners * _CUTOFF_DROP_PER_CORNER_HZ)
_BASE_CUTOFF_HZ: float = 4000.0           # cutoff at 0 corners (direct LoS)
_CUTOFF_DROP_PER_CORNER_HZ: float = 900.0  # cutoff reduction per corner
_MIN_CUTOFF_HZ: float = 300.0             # floor -- never go below this

# Width of the raised-cosine transition band for the FFT lowpass (Hz).
# Wider = smoother rolloff = shorter impulse response = fewer artifacts.
_TRANSITION_BW_HZ: float = 500.0

# Reflection delay parameters.
# _SAMPLES_PER_GRID_UNIT: at 44100 Hz, 40 samples ~ 0.9 ms per grid unit of
# extra path length.  This gives roughly 1-2 ms of delay per extra grid unit,
# producing a small-room echo feel.  Empirical / tunable.
_SAMPLES_PER_GRID_UNIT: int = 40

# _MAX_DELAY_SAMPLES: upper bound on the reflection delay to cap buffer
# memory in pathological mazes.  8000 samples ~ 181 ms at 44100 Hz.
_MAX_DELAY_SAMPLES: int = 8000


# ---------------------------------------------------------------------------
# Corner-driven cutoff & Line-of-sight computation
# ---------------------------------------------------------------------------
def _has_line_of_sight(
    p0: tuple[int, int],
    p1: tuple[int, int],
    walls: set[tuple[int, int]] | frozenset[tuple[int, int]],
) -> bool:
    """Check if the direct straight line between p0 and p1 is unobstructed.

    Uses Bresenham's algorithm to walk grid cells between p0 and p1.
    Endpoints p0 and p1 are excluded from the wall check so standing on
    a wall coordinate is not self-blocking.
    """
    if not walls:
        return True

    x0, y0 = p0
    x1, y1 = p1
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0
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
    """Compute the lowpass cutoff frequency from the path's corner count.

    Returns a value in Hz, clamped to [_MIN_CUTOFF_HZ, _BASE_CUTOFF_HZ].
    """
    return max(_MIN_CUTOFF_HZ, _BASE_CUTOFF_HZ - corners * _CUTOFF_DROP_PER_CORNER_HZ)


def _primary_cutoff_hz(
    listener_cell: tuple[int, int],
    source_cell: tuple[int, int],
    corners: int,
    walls: set[tuple[int, int]] | frozenset[tuple[int, int]],
) -> float:
    """Determine the lowpass cutoff for the primary arrival.

    If direct line-of-sight between listener and source is unobstructed,
    the full _BASE_CUTOFF_HZ is returned regardless of grid-discretization
    corners.  If obstructed by a wall, the corner-driven formula is used.
    """
    if _has_line_of_sight(listener_cell, source_cell, walls):
        return _BASE_CUTOFF_HZ
    return _corner_cutoff_hz(corners)


# ---------------------------------------------------------------------------
# FFT-based lowpass filter (per-block, no overlap)
# ---------------------------------------------------------------------------
def _fft_lowpass(mono: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
    """Apply a block-based FFT lowpass filter with smooth rolloff.

    Uses numpy's rfft / irfft on real-valued input.  Applies a raised-
    cosine (Hann-window-shaped) taper in the frequency domain around
    *cutoff_hz* rather than a brick-wall, keeping the effective impulse
    response short and minimising circular-convolution wrap-around
    artifacts at block boundaries.

    Parameters
    ----------
    mono : np.ndarray
        Real-valued mono audio block, shape ``(n_frames,)``.
    cutoff_hz : float
        Lowpass cutoff frequency in Hz.
    sample_rate : int
        Audio sample rate in Hz.

    Returns
    -------
    np.ndarray
        Filtered mono block, same length as input, dtype float32.
    """
    n = len(mono)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    spectrum = np.fft.rfft(mono)

    # Build gain curve: 1.0 below cutoff, raised-cosine taper, 0.0 above.
    half_bw = _TRANSITION_BW_HZ / 2.0
    low = max(0.0, cutoff_hz - half_bw)   # full-pass edge (clamped >= 0)
    high = cutoff_hz + half_bw             # full-stop edge
    bw = high - low                        # actual transition width (may differ
                                           # from _TRANSITION_BW_HZ if low was clamped)

    gains = np.ones(len(freqs), dtype=np.float64)

    # Transition band: raised-cosine taper (gain goes 1 -> 0 over [low, high])
    transition_mask = (freqs > low) & (freqs < high)
    if np.any(transition_mask):
        gains[transition_mask] = 0.5 * (
            1.0 + np.cos(np.pi * (freqs[transition_mask] - low) / max(1e-9, bw))
        )

    # Stopband: zero
    gains[freqs >= high] = 0.0

    spectrum *= gains
    return np.fft.irfft(spectrum, n=n).astype(np.float32)


# ---------------------------------------------------------------------------
# Reflection delay line (circular buffer, persistent in filter_state)
# ---------------------------------------------------------------------------
def _delay_line_process(
    samples: np.ndarray,
    delay_samples: int,
    filter_state: dict,
    n_frames: int,
) -> np.ndarray:
    """Write *samples* into the reflection delay buffer and return the
    delayed readback.

    The buffer is a circular (ring) buffer stored in *filter_state* under
    keys ``"reflection_delay_buf"`` and ``"reflection_delay_write_pos"``.
    It is initialized to zeros on first call, so the first few blocks of
    reflected output will ramp in from silence as the buffer primes --
    this is standard delay-line behaviour.

    If *delay_samples* changes between calls (e.g. listener / source
    moved, changing the reflected path length), the new delay is applied
    immediately.  This may produce a minor discontinuity, which is
    acceptable for this game-audio approximation -- smooth delay-length
    interpolation is not implemented.

    Parameters
    ----------
    samples : np.ndarray
        Filtered / gained reflected-arrival mono block to write in.
    delay_samples : int
        Number of samples to delay the readback by.  Clamped externally
        to [0, _MAX_DELAY_SAMPLES] before this function is called.
    filter_state : dict
        Persistent per-source state dict (mutated in place).
    n_frames : int
        Block length in samples.

    Returns
    -------
    np.ndarray
        Delayed mono block, shape ``(n_frames,)``, dtype float32.
    """
    buf_key = "reflection_delay_buf"
    wpos_key = "reflection_delay_write_pos"

    # Allocate on first use.  Size = max delay + generous block padding.
    if buf_key not in filter_state:
        buf_size = _MAX_DELAY_SAMPLES + max(n_frames, 2048)
        filter_state[buf_key] = np.zeros(buf_size, dtype=np.float32)
        filter_state[wpos_key] = 0

    buf = filter_state[buf_key]
    buf_size = len(buf)
    write_pos = filter_state[wpos_key]

    # Write current block (write-first so delay=0 means "no delay").
    write_idx = (np.arange(n_frames) + write_pos) % buf_size
    buf[write_idx] = samples[:n_frames]

    # Read delayed output.
    read_idx = (np.arange(n_frames) + write_pos - delay_samples) % buf_size
    delayed = buf[read_idx].copy()

    # Advance write position.
    filter_state[wpos_key] = (write_pos + n_frames) % buf_size

    return delayed


# ---------------------------------------------------------------------------
# Public API -- the swap boundary
# ---------------------------------------------------------------------------
def process_block(
    input_block: np.ndarray,          # (n_frames,) mono float32
    listener_pos: tuple[float, float],
    source_pos: tuple[float, float],
    walls: set[tuple[int, int]] | frozenset[tuple[int, int]],
    sample_rate: int,
    filter_state: dict,               # persistent per-source state, mutated in-place
) -> np.ndarray:                      # (n_frames, 2) stereo float32
    """Transform a mono source block into a stereo output block.

    Produces up to two independently processed arrivals (primary +
    reflected) whose panned stereo contributions are summed into the
    returned output.

    Parameters
    ----------
    input_block : np.ndarray
        Mono float32 audio chunk for one source, shape ``(n_frames,)``.
    listener_pos, source_pos : tuple[float, float]
        Grid-coordinate positions.  Float-typed due to audio_engine.py's
        calling convention; rounded to int internally for pathfinding.
    walls : set or frozenset of (int, int)
        Set of grid cells that are walls.
    sample_rate : int
        Audio sample rate in Hz.
    filter_state : dict
        Mutable dict persisted across calls for this source.  Used to
        store the reflection delay buffer state:
          - ``"reflection_delay_buf"`` : np.ndarray -- circular buffer
          - ``"reflection_delay_write_pos"`` : int -- write head position
        These keys are created automatically on first call where a
        reflected arrival exists.  Callers should continue passing the
        same dict instance across calls as before.

    Returns
    -------
    np.ndarray
        Stereo float32 output, shape ``(n_frames, 2)``.
    """
    n_frames = input_block.shape[0]
    mono = input_block.astype(np.float32, copy=True)

    # ---- Round float positions to integer grid cells for pathfinding ----
    listener_cell = (int(round(listener_pos[0])), int(round(listener_pos[1])))
    source_cell = (int(round(source_pos[0])), int(round(source_pos[1])))

    # ---- 1. Find up to 2 shortest paths via pathfinding -----------------
    paths = find_k_paths(
        start=listener_cell,
        end=source_cell,
        walls=walls,
        grid_cols=GRID_COLS,
        grid_rows=GRID_ROWS,
        k=2,
    )

    if not paths:
        # ---- Total occlusion: no path exists ---------------------------
        mono *= _TOTAL_OCCLUSION_FLOOR_GAIN

        # Flush delay buffer with silence so stale reflected audio doesn't
        # produce a jarring artifact if a second path reappears later.
        if "reflection_delay_buf" in filter_state:
            _delay_line_process(
                np.zeros(n_frames, dtype=np.float32),
                0, filter_state, n_frames,
            )

        # Pan and return (unchanged total-occlusion behavior).
        dx = source_pos[0] - listener_pos[0]
        max_offset = 20.0
        pan = np.clip(dx / max_offset, -1.0, 1.0)
        right_gain = 0.5 * (1.0 + pan)
        left_gain = 0.5 * (1.0 - pan)

        out = np.empty((n_frames, 2), dtype=np.float32)
        out[:, 0] = mono * left_gain
        out[:, 1] = mono * right_gain
        return out

    # -- At least one path exists -----------------------------------------
    nyquist = sample_rate / 2.0
    primary_path = paths[0]
    has_reflection = len(paths) >= 2

    # === PRIMARY ARRIVAL (paths[0]) ======================================
    if has_reflection:
        primary_mono = mono.copy()  # independent copy; mono reused below
    else:
        primary_mono = mono         # no reflection, safe to modify in place

    # Corner-driven FFT lowpass (with line-of-sight check to ignore grid discretization corners)
    primary_cutoff = _primary_cutoff_hz(
        listener_cell, source_cell, primary_path.corners, walls
    )
    if primary_cutoff < nyquist:
        primary_mono = _fft_lowpass(primary_mono, primary_cutoff, sample_rate)

    # Path-length distance gain
    primary_gain = 1.0 / (1.0 + 0.15 * primary_path.length)
    primary_mono *= primary_gain

    # Pan (straight-line x-offset, unchanged from single-arrival behavior)
    dx = source_pos[0] - listener_pos[0]
    max_offset = 20.0
    pan = np.clip(dx / max_offset, -1.0, 1.0)
    primary_right = 0.5 * (1.0 + pan)
    primary_left = 0.5 * (1.0 - pan)

    # Build stereo output starting with primary arrival.
    out = np.empty((n_frames, 2), dtype=np.float32)
    out[:, 0] = primary_mono * primary_left
    out[:, 1] = primary_mono * primary_right

    # === REFLECTED ARRIVAL (paths[1], when present) ======================
    if has_reflection:
        ref_path = paths[1]

        # -- Filter (same function, second path's corners) ----------------
        ref_mono = mono  # reuse original copy (primary_mono is independent)
        ref_cutoff = _corner_cutoff_hz(ref_path.corners)
        if ref_cutoff < nyquist:
            ref_mono = _fft_lowpass(ref_mono, ref_cutoff, sample_rate)

        # -- Distance gain ------------------------------------------------
        ref_gain_val = 1.0 / (1.0 + 0.15 * ref_path.length)
        ref_mono = ref_mono * ref_gain_val  # new array (avoids aliasing mono)

        # -- Delay --------------------------------------------------------
        delay_samples = int(round(
            (ref_path.length - primary_path.length) * _SAMPLES_PER_GRID_UNIT
        ))
        delay_samples = max(0, min(delay_samples, _MAX_DELAY_SAMPLES))

        ref_delayed = _delay_line_process(
            ref_mono, delay_samples, filter_state, n_frames,
        )

        # -- Pan (final-segment direction) --------------------------------
        # The path is stored listener-first: cells[0] = listener,
        # cells[-1] = source.  We use the direction of the last segment
        # (cells[-2] -> cells[-1]) as a simplified directional cue for
        # the reflected arrival's panning.
        cells = ref_path.cells
        if len(cells) >= 2:
            dx_ref = float(cells[-1][0] - cells[-2][0])
        else:
            dx_ref = 0.0  # degenerate path -> center pan
        ref_pan = np.clip(dx_ref / max_offset, -1.0, 1.0)
        ref_right = 0.5 * (1.0 + ref_pan)
        ref_left = 0.5 * (1.0 - ref_pan)

        # -- Add reflected arrival to output (summed, no rescaling) -------
        out[:, 0] += ref_delayed * ref_left
        out[:, 1] += ref_delayed * ref_right
    else:
        # Single path only.  Flush delay buffer with silence so stale
        # reflected audio doesn't leak if a second path appears later.
        if "reflection_delay_buf" in filter_state:
            _delay_line_process(
                np.zeros(n_frames, dtype=np.float32),
                0, filter_state, n_frames,
            )

    return out
