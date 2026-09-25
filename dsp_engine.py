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
0.08 distance-gain constant and 20.0 max_offset pan constant were also
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

import math

import numpy as np

from pathfinding import (
    find_k_paths, has_line_of_sight,
    ReflectorCandidate,
    find_reflector_candidates, find_transmission_path,
)
from shared_state import GRID_COLS, GRID_ROWS, WALL_GAIN_MEDIUM


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

# Through-wall transmission cutoff parameters (all in Hz) -- empirical / by-ear starting values.
_TRANSMISSION_BASE_CUTOFF_HZ: float = 1500.0
_TRANSMISSION_CUTOFF_DROP_PER_WALL_HZ: float = 500.0

# Discrete echo parameters (all in Hz / counts) -- empirical / by-ear starting values.
# _MAX_SIMULTANEOUS_ECHOES matches find_reflector_candidates's max_candidates=4 cap by design.
_MAX_SIMULTANEOUS_ECHOES: int = 4
_ECHO_BASE_CUTOFF_HZ: float = 3500.0
_ECHO_CUTOFF_MUFFLE_RANGE_HZ: float = 3000.0

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

# Default reflectivity gain used when no matching wall cell is found next to a
# corner (should not occur in a well-formed reflected path, but guards against
# edge cases).  Matches WALL_GAIN_MEDIUM so behaviour is consistent with the
# material system's neutral setting.
_DEFAULT_MATERIAL_GAIN: float = WALL_GAIN_MEDIUM


# ---------------------------------------------------------------------------
# Corner-driven cutoff computation
# ---------------------------------------------------------------------------


def _corner_cutoff_hz(corners: int) -> float:
    """Compute the lowpass cutoff frequency from the path's corner count.

    Returns a value in Hz, clamped to [_MIN_CUTOFF_HZ, _BASE_CUTOFF_HZ].
    """
    return max(_MIN_CUTOFF_HZ, _BASE_CUTOFF_HZ - corners * _CUTOFF_DROP_PER_CORNER_HZ)


# 8 neighbour offsets used for the wall-proximity search at each corner
# cell.  ORDER MATTERS: when a corner touches two different-material
# walls at once, the neighbour earliest in this list wins the tie-break
# (see _corner_material_gain's docstring).  Currently N/S/W/E before
# diagonals, so orthogonal neighbours are always checked first.
_NEIGHBOUR_OFFSETS: tuple[tuple[int, int], ...] = (
    ( 0, -1), ( 0,  1), (-1,  0), ( 1,  0),
    (-1, -1), ( 1, -1), (-1,  1), ( 1,  1),
)


def _corner_material_gain(
    corner_cell: tuple[int, int],
    wall_gains: dict[tuple[int, int], float],
) -> float:
    """Return the reflectivity gain for the wall adjacent to *corner_cell*.

    Searches the 8 immediate neighbours of *corner_cell* for a wall entry
    in *wall_gains*, in a fixed priority order (N, S, W, E, NW, NE, SW, SE
    -- see _NEIGHBOUR_OFFSETS).  Returns the gain of the FIRST neighbour
    found in that order, or _DEFAULT_MATERIAL_GAIN as a safe fallback if
    none is present.

    KNOWN SIMPLIFICATION: if a corner cell happens to be adjacent to two
    or more walls of DIFFERENT materials at once, the neighbour earliest
    in the fixed priority order wins (in practice, this almost always
    means North wins), regardless of which wall the reflected path is
    actually turning against geometrically.  This is an intentional,
    acknowledged trade-off given project time constraints -- a fully
    correct version would determine the gain from whichever wall cell is
    on the path's actual turn direction, using ref_path.cells to infer
    which side the corner bends toward.  Left as a documented limitation
    rather than implemented, since single-material corners (the common
    case) are unaffected.
    """
    cx, cy = corner_cell
    for dx, dy in _NEIGHBOUR_OFFSETS:
        gain = wall_gains.get((cx + dx, cy + dy))
        if gain is not None:
            return gain
    return _DEFAULT_MATERIAL_GAIN


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
    if has_line_of_sight(listener_cell, source_cell, walls):
        return _BASE_CUTOFF_HZ
    return _corner_cutoff_hz(corners)


# ---------------------------------------------------------------------------
# Through-wall transmission helpers (Family B)
# ---------------------------------------------------------------------------


def _transmission_gain(
    source_pos: tuple[float, float],
    listener_pos: tuple[float, float],
    transmission_path: list[tuple[tuple[int, int], float]] | None,
) -> float:
    """Compute through-wall transmission gain (Family B).

    Straight-line distance gain times the product of each crossed wall's
    transmission factor (1.0 - material_gain), so two low-reflectivity walls
    (e.g. curtains) in series transmit more than one, and any single
    high-reflectivity wall (e.g. concrete) sharply reduces it.
    """
    if not transmission_path:
        return 0.0

    dx = source_pos[0] - listener_pos[0]
    dy = source_pos[1] - listener_pos[1]
    dist = math.sqrt(dx * dx + dy * dy)
    distance_gain = 1.0 / (1.0 + 0.08 * dist)

    transmission_product = 1.0
    for _, material_gain in transmission_path:
        transmission_product *= (1.0 - material_gain)

    return distance_gain * transmission_product


def _transmission_cutoff_hz(
    transmission_path: list[tuple[tuple[int, int], float]] | None,
) -> float:
    """Compute the lowpass cutoff frequency for through-wall transmission.

    Formula: cutoff = max(_MIN_CUTOFF_HZ, _TRANSMISSION_BASE_CUTOFF_HZ - wall_count * _TRANSMISSION_CUTOFF_DROP_PER_WALL_HZ).
    This is deliberately steeper than the routed-path cutoff to model that
    walls preferentially attenuate high frequencies even for direct penetration.
    """
    if not transmission_path:
        return _TRANSMISSION_BASE_CUTOFF_HZ

    wall_count = len(transmission_path)
    return max(
        _MIN_CUTOFF_HZ,
        _TRANSMISSION_BASE_CUTOFF_HZ - wall_count * _TRANSMISSION_CUTOFF_DROP_PER_WALL_HZ,
    )


# ---------------------------------------------------------------------------
# Discrete echo helpers (Family C)
# ---------------------------------------------------------------------------


def _echo_gain(candidate: ReflectorCandidate) -> float:
    """Compute discrete echo gain (Family C).

    Predicted echo loudness based on two-way distance attenuation and
    the reflecting wall's material reflectivity.  Matches the exact
    formula used by find_reflector_candidates() to rank candidates:
    gain = (1.0 / (1.0 + 0.08 * 2.0 * distance_to_wall)) * material_gain.
    """
    return (1.0 / (1.0 + 0.08 * 2.0 * candidate.distance_to_wall)) * candidate.material_gain


def _echo_cutoff_hz(candidate: ReflectorCandidate) -> float:
    """Compute the lowpass cutoff frequency for a discrete echo reflection.

    Formula: cutoff = max(_MIN_CUTOFF_HZ, _ECHO_BASE_CUTOFF_HZ - (1.0 - candidate.material_gain) * _ECHO_CUTOFF_MUFFLE_RANGE_HZ).
    Softer materials (lower material_gain) produce a larger muffle term and
    therefore a lower cutoff frequency, modeling higher high-frequency absorption
    upon wall reflection.
    """
    return max(
        _MIN_CUTOFF_HZ,
        _ECHO_BASE_CUTOFF_HZ - (1.0 - candidate.material_gain) * _ECHO_CUTOFF_MUFFLE_RANGE_HZ,
    )


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
# Geometry-cache staleness detection (Stage 2A)
# ---------------------------------------------------------------------------
_GEOM_CACHE_SOURCE_POS_KEY = "_geom_cache_source_pos"
_GEOM_CACHE_LISTENER_POS_KEY = "_geom_cache_listener_pos"
_GEOM_CACHE_WALLS_KEY = "_geom_cache_walls"


def _geometry_cache_is_stale(
    filter_state: dict,
    source_pos: tuple[float, float],
    listener_pos: tuple[float, float],
    walls: set[tuple[int, int]] | frozenset[tuple[int, int]],
) -> bool:
    """Return True if the geometry cache needs recomputing this block.

    Compares the current source_pos, listener_pos, and walls against the
    values stored in filter_state by _geometry_cache_store_snapshot on the
    previous block.  Returns True (stale) when any value differs or when
    no cached snapshot exists yet (first call for this source).

    Stage 2B will use the return value to decide whether to re-run
    find_reflector_candidates / find_transmission_path.
    """
    if (
        _GEOM_CACHE_SOURCE_POS_KEY not in filter_state
        or _GEOM_CACHE_LISTENER_POS_KEY not in filter_state
        or _GEOM_CACHE_WALLS_KEY not in filter_state
    ):
        return True
    if filter_state[_GEOM_CACHE_SOURCE_POS_KEY] != source_pos:
        return True
    if filter_state[_GEOM_CACHE_LISTENER_POS_KEY] != listener_pos:
        return True
    if filter_state[_GEOM_CACHE_WALLS_KEY] != walls:
        return True
    return False


def _geometry_cache_store_snapshot(
    filter_state: dict,
    source_pos: tuple[float, float],
    listener_pos: tuple[float, float],
    walls: set[tuple[int, int]] | frozenset[tuple[int, int]],
) -> None:
    """Record the current geometry inputs so future staleness checks work.

    Called by Stage 2B right after recomputing cached reflector/transmission
    data, to mark the cache as valid for this combination of positions and
    walls.  Walls are stored as a frozenset for consistent equality checks.
    """
    filter_state[_GEOM_CACHE_SOURCE_POS_KEY] = source_pos
    filter_state[_GEOM_CACHE_LISTENER_POS_KEY] = listener_pos
    filter_state[_GEOM_CACHE_WALLS_KEY] = frozenset(walls)


# ---------------------------------------------------------------------------
# Reflection delay line (circular buffer, persistent in filter_state)
# ---------------------------------------------------------------------------
def _delay_line_process(
    samples: np.ndarray,
    delay_samples: int,
    filter_state: dict,
    n_frames: int,
    buf_key: str = "reflection_delay_buf",
    wpos_key: str = "reflection_delay_write_pos",
) -> np.ndarray:
    """Write *samples* into the reflection delay buffer and return the
    delayed readback.

    The buffer is a circular (ring) buffer stored in *filter_state* under
    keys ``"reflection_delay_buf"`` and ``"reflection_delay_write_pos"``
    (or custom keys specified by *buf_key* and *wpos_key*).
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
    buf_key : str
        Key in *filter_state* for the delay buffer array.
    wpos_key : str
        Key in *filter_state* for the integer write head position.

    Returns
    -------
    np.ndarray
        Delayed mono block, shape ``(n_frames,)``, dtype float32.
    """
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
    wall_gains: dict[tuple[int, int], float] | None = None,  # pos -> gain for DSP
) -> np.ndarray:                      # (n_frames, 2) stereo float32
    """Transform a mono source block into a stereo output block.

    Produces independently processed arrivals (primary, reflected,
    through-wall transmission, and discrete echo) whose panned stereo
    contributions are summed into the returned output.  Transmission
    contributes whenever the direct source-listener line crosses walls, and
    up to _MAX_SIMULTANEOUS_ECHOES discrete echo arrivals (Family C)
    contribute whenever reflector candidates are cached, both independently
    of whether a routed path exists (present in both open and totally-occluded
    layouts).

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
        store the reflection and echo delay buffer states:
          - ``"reflection_delay_buf"`` : np.ndarray -- circular buffer
          - ``"reflection_delay_write_pos"`` : int -- write head position
          - ``"echo_delay_buf_{i}"`` : np.ndarray -- circular buffer for echo slot i (0..3)
          - ``"echo_delay_write_pos_{i}"`` : int -- write head position for echo slot i (0..3)
        These keys are created automatically on first call where a
        reflected or echo arrival exists.  Callers should continue passing the
        same dict instance across calls as before.
    wall_gains : dict or None
        Optional mapping of wall positions to their reflectivity gain
        scalars, as produced by SharedState.get_snapshot()["wall_gains"].
        When provided, the reflected arrival's amplitude is multiplied by
        the product of each corner cell's adjacent-wall gain.  Defaults
        to ``None`` (treated as empty dict), which falls back to
        ``_DEFAULT_MATERIAL_GAIN`` per corner and is backward-compatible
        with existing tests that don't pass this argument.

    Returns
    -------
    np.ndarray
        Stereo float32 output, shape ``(n_frames, 2)``.
    """
    if wall_gains is None:
        wall_gains = {}

    n_frames = input_block.shape[0]
    nyquist = sample_rate / 2.0
    mono = input_block.astype(np.float32, copy=True)

    # ---- Round float positions to integer grid cells for pathfinding ----
    listener_cell = (int(round(listener_pos[0])), int(round(listener_pos[1])))
    source_cell = (int(round(source_pos[0])), int(round(source_pos[1])))

    # ---- Geometry-cache: recompute reflector/transmission data if stale --
    if _geometry_cache_is_stale(filter_state, source_pos, listener_pos, walls):
        filter_state["_geom_cache_reflector_candidates"] = (
            find_reflector_candidates(
                source_cell, listener_cell, walls, wall_gains,
                GRID_COLS, GRID_ROWS,
            )
        )
        filter_state["_geom_cache_transmission_path"] = (
            find_transmission_path(
                source_cell, listener_cell, walls, wall_gains,
                GRID_COLS, GRID_ROWS,
            )
        )
        _geometry_cache_store_snapshot(
            filter_state, source_pos, listener_pos, walls,
        )

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

        # Pan and write baseline floor-gain output.
        dx = source_pos[0] - listener_pos[0]
        max_offset = 20.0
        pan = np.clip(dx / max_offset, -1.0, 1.0)
        right_gain = 0.5 * (1.0 + pan)
        left_gain = 0.5 * (1.0 - pan)

        out = np.empty((n_frames, 2), dtype=np.float32)
        out[:, 0] = mono * left_gain
        out[:, 1] = mono * right_gain

        # === THROUGH-WALL TRANSMISSION ARRIVAL (Family B) =================
        transmission_path = filter_state.get("_geom_cache_transmission_path")
        trans_gain = _transmission_gain(source_pos, listener_pos, transmission_path)
        if trans_gain > 0.0:
            trans_mono = input_block.astype(np.float32, copy=True)
            trans_cutoff = _transmission_cutoff_hz(transmission_path)
            if trans_cutoff < nyquist:
                trans_mono = _fft_lowpass(trans_mono, trans_cutoff, sample_rate)
            trans_mono *= trans_gain
            out[:, 0] += trans_mono * left_gain
            out[:, 1] += trans_mono * right_gain

        # === DISCRETE ECHO ARRIVALS (Family C, up to _MAX_SIMULTANEOUS_ECHOES) =
        candidates = filter_state.get("_geom_cache_reflector_candidates") or []
        for i in range(_MAX_SIMULTANEOUS_ECHOES):
            buf_key = f"echo_delay_buf_{i}"
            wpos_key = f"echo_delay_write_pos_{i}"
            if i < len(candidates):
                echo_candidate = candidates[i]
                echo_mono = input_block.astype(np.float32, copy=True)
                echo_cutoff = _echo_cutoff_hz(echo_candidate)
                if echo_cutoff < nyquist:
                    echo_mono = _fft_lowpass(echo_mono, echo_cutoff, sample_rate)
                echo_mono *= _echo_gain(echo_candidate)
                round_trip_samples = int(round(
                    2.0 * echo_candidate.distance_to_wall * _SAMPLES_PER_GRID_UNIT
                ))
                delay_samples = max(0, min(round_trip_samples, _MAX_DELAY_SAMPLES))
                echo_delayed = _delay_line_process(
                    echo_mono, delay_samples, filter_state, n_frames,
                    buf_key=buf_key, wpos_key=wpos_key,
                )
                dx_echo = echo_candidate.wall_cell[0] - listener_pos[0]
                pan = np.clip(dx_echo / max_offset, -1.0, 1.0)
                echo_right = 0.5 * (1.0 + pan)
                echo_left = 0.5 * (1.0 - pan)
                out[:, 0] += echo_delayed * echo_left
                out[:, 1] += echo_delayed * echo_right
            else:
                if buf_key in filter_state:
                    _delay_line_process(
                        np.zeros(n_frames, dtype=np.float32),
                        0, filter_state, n_frames,
                        buf_key=buf_key, wpos_key=wpos_key,
                    )

        return out

    # -- At least one path exists -----------------------------------------
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
    primary_gain = 1.0 / (1.0 + 0.08 * primary_path.length)
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
        ref_gain_val = 1.0 / (1.0 + 0.08 * ref_path.length)
        ref_mono = ref_mono * ref_gain_val  # new array (avoids aliasing mono)

        # -- Material gain (per-corner reflectivity) ----------------------
        # Identify the corner cells of the reflected path using the same
        # direction-change logic as pathfinding._count_corners, so the set
        # of corner cells is identical to what produced ref_path.corners.
        #
        # Corner identification (mirrors pathfinding._count_corners exactly):
        #   prev direction = cells[1] - cells[0]
        #   for i in range(2, len(cells)):
        #       if direction(cells[i]-cells[i-1]) != prev: corner at cells[i-1]
        #
        # This guarantees: len(corner_cells) == ref_path.corners.
        cells = ref_path.cells
        ref_material_gain: float = 1.0
        if len(cells) >= 3 and wall_gains:
            prev_dx = cells[1][0] - cells[0][0]
            prev_dy = cells[1][1] - cells[0][1]
            corner_count_check: int = 0
            for i in range(2, len(cells)):
                dx = cells[i][0] - cells[i - 1][0]
                dy = cells[i][1] - cells[i - 1][1]
                if dx != prev_dx or dy != prev_dy:
                    # cells[i-1] is the corner cell — look up adjacent wall
                    ref_material_gain *= _corner_material_gain(cells[i - 1], wall_gains)
                    corner_count_check += 1
                prev_dx = dx
                prev_dy = dy
            # Assertion (debug-mode only): our loop counts must agree with
            # ref_path.corners as computed by pathfinding._count_corners.
            # Both use identical direction-change logic so they must match.
            assert corner_count_check == ref_path.corners, (
                f"Corner count mismatch: loop counted {corner_count_check}, "
                f"ref_path.corners={ref_path.corners}"
            )
        elif len(cells) >= 3:
            # wall_gains is empty (backward-compat / tests): apply the
            # default material gain for each corner so tests still reflect
            # the material system semantics consistently.
            ref_material_gain = _DEFAULT_MATERIAL_GAIN ** ref_path.corners

        ref_mono = ref_mono * ref_material_gain

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
            # Use the segment(s) LEAVING the listener, not arriving at the source —
            # panning represents how the listener perceives arrival direction.
            # Average over a short run to avoid single-step jitter from grid
            # discretization changing frame to frame.
            lookahead = min(4, len(cells) - 1)
            dx_ref = float(cells[lookahead][0] - cells[0][0])
            ref_scale = float(lookahead)
        else:
            dx_ref = 0.0
            ref_scale = 1.0
        ref_pan = np.clip(dx_ref / max(1.0, ref_scale), -1.0, 1.0)        
        
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

    # === THROUGH-WALL TRANSMISSION ARRIVAL (Family B) =====================
    transmission_path = filter_state.get("_geom_cache_transmission_path")
    trans_gain = _transmission_gain(source_pos, listener_pos, transmission_path)
    if trans_gain > 0.0:
        trans_mono = input_block.astype(np.float32, copy=True)
        trans_cutoff = _transmission_cutoff_hz(transmission_path)
        if trans_cutoff < nyquist:
            trans_mono = _fft_lowpass(trans_mono, trans_cutoff, sample_rate)
        trans_mono *= trans_gain
        out[:, 0] += trans_mono * primary_left
        out[:, 1] += trans_mono * primary_right

    # === DISCRETE ECHO ARRIVALS (Family C, up to _MAX_SIMULTANEOUS_ECHOES) =
    candidates = filter_state.get("_geom_cache_reflector_candidates") or []
    for i in range(_MAX_SIMULTANEOUS_ECHOES):
        buf_key = f"echo_delay_buf_{i}"
        wpos_key = f"echo_delay_write_pos_{i}"
        if i < len(candidates):
            echo_candidate = candidates[i]
            echo_mono = input_block.astype(np.float32, copy=True)
            echo_cutoff = _echo_cutoff_hz(echo_candidate)
            if echo_cutoff < nyquist:
                echo_mono = _fft_lowpass(echo_mono, echo_cutoff, sample_rate)
            echo_mono *= _echo_gain(echo_candidate)
            round_trip_samples = int(round(
                2.0 * echo_candidate.distance_to_wall * _SAMPLES_PER_GRID_UNIT
            ))
            delay_samples = max(0, min(round_trip_samples, _MAX_DELAY_SAMPLES))
            echo_delayed = _delay_line_process(
                echo_mono, delay_samples, filter_state, n_frames,
                buf_key=buf_key, wpos_key=wpos_key,
            )
            dx_echo = echo_candidate.wall_cell[0] - listener_pos[0]
            pan = np.clip(dx_echo / max_offset, -1.0, 1.0)
            echo_right = 0.5 * (1.0 + pan)
            echo_left = 0.5 * (1.0 - pan)
            out[:, 0] += echo_delayed * echo_left
            out[:, 1] += echo_delayed * echo_right
        else:
            if buf_key in filter_state:
                _delay_line_process(
                    np.zeros(n_frames, dtype=np.float32),
                    0, filter_state, n_frames,
                    buf_key=buf_key, wpos_key=wpos_key,
                )

    return out
