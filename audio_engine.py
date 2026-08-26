"""
audio_engine.py — Real-time audio output via sounddevice.

Owns the ``sounddevice.OutputStream`` and its callback.  The callback
runs on a PortAudio thread and must NEVER block, allocate heavily, do
file I/O, or call print().

Audio files are loaded eagerly into memory as float32 NumPy arrays when
``load_audio_for_source`` is called (from the main thread).  The callback
reads the shared state snapshot ONCE per invocation and iterates over
active sources, slicing audio, calling the DSP black-box, and summing
the results into the output buffer.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

import dsp_engine
from shared_state import SharedState


class AudioEngine:
    """Manages the real-time audio output stream and per-source playback state."""

    # TUNE: blocksize controls latency vs. CPU load.  1024 @ 44100 Hz ≈ 23 ms.
    BLOCKSIZE: int = 1024
    SAMPLE_RATE: int = 44100
    CHANNELS: int = 2  # stereo output

    def __init__(self, shared_state: SharedState) -> None:
        self._shared_state = shared_state

        # --- Per-source persistent data (main-thread writes, callback reads) ---
        self._lock = threading.Lock()

        # source_id -> np.ndarray (mono float32, full file)
        self._audio_buffers: dict[int, np.ndarray] = {}

        # source_id -> int (current read position in samples)
        self._cursors: dict[int, int] = {}

        # source_id -> dict (filter_state for dsp_engine.process_block)
        self._filter_states: dict[int, dict] = {}

        # --- Visualisation ring-buffer: last raw/processed blocks per source ---
        # NOTE: These are plain numpy array references swapped atomically by the
        # callback.  A single reference swap of a Python object is thread-safe
        # on CPython (the GIL guarantees pointer-width writes are atomic), so
        # the render thread can read these without a lock.  The worst that can
        # happen is reading a stale frame, which is acceptable for visualisation.
        self._last_raw: dict[int, Optional[np.ndarray]] = {}
        self._last_processed: dict[int, Optional[np.ndarray]] = {}
        self._last_mix: Optional[np.ndarray] = None

        self._stream: Optional[sd.OutputStream] = None

    # ------------------------------------------------------------------
    # Audio file loading (called from the main / UI thread)
    # ------------------------------------------------------------------
    def load_audio_for_source(self, source_id: int, audio_path: str) -> None:
        """Load an audio file from disk into an in-memory float32 buffer.

        Must be called from the main thread — file I/O is forbidden inside
        the audio callback.
        """
        data, file_sr = sf.read(audio_path, dtype="float32", always_2d=True)
        # Mix down to mono if stereo
        if data.shape[1] > 1:
            mono = data.mean(axis=1)
        else:
            mono = data[:, 0]
        # Resample if necessary (simple skip/repeat — good enough for skeleton)
        if file_sr != self.SAMPLE_RATE:
            ratio = self.SAMPLE_RATE / file_sr
            indices = np.round(np.arange(0, len(mono), 1.0 / ratio)).astype(int)
            indices = indices[indices < len(mono)]
            mono = mono[indices]
        mono = mono.astype(np.float32)

        with self._lock:
            self._audio_buffers[source_id] = mono
            self._cursors[source_id] = 0
            self._filter_states.setdefault(source_id, {})

    def remove_source(self, source_id: int) -> None:
        """Clean up buffers for a removed source."""
        with self._lock:
            self._audio_buffers.pop(source_id, None)
            self._cursors.pop(source_id, None)
            self._filter_states.pop(source_id, None)
            self._last_raw.pop(source_id, None)
            self._last_processed.pop(source_id, None)

    # ------------------------------------------------------------------
    # Visualisation accessors (called from the render thread)
    # ------------------------------------------------------------------
    def get_last_raw(self, source_id: int) -> Optional[np.ndarray]:
        """Return the last raw mono chunk for *source_id*, or None."""
        return self._last_raw.get(source_id)

    def get_last_processed(self, source_id: int) -> Optional[np.ndarray]:
        """Return the last processed stereo chunk for *source_id*, or None."""
        return self._last_processed.get(source_id)

    def get_last_mix(self) -> Optional[np.ndarray]:
        """Return the last mixed stereo output block, or None."""
        return self._last_mix

    # ------------------------------------------------------------------
    # Audio callback (runs on PortAudio thread — NO blocking/alloc/IO!)
    # ------------------------------------------------------------------
    def _callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """sounddevice output callback.  Fills *outdata* with the mixed
        stereo signal of all active sources after DSP processing."""

        # Read shared state ONCE
        snapshot = self._shared_state.get_snapshot()

        listener_pos = snapshot["listener_pos"]
        sources = snapshot["sources"]
        walls = snapshot["walls"]

        # Start with silence
        mix = np.zeros((frames, 2), dtype=np.float32)
        active_count = 0

        if listener_pos is not None:
            for sid, src_info in sources.items():
                if not src_info["playing"]:
                    continue

                # ---- Fetch the pre-loaded buffer & cursor ----
                # We access _audio_buffers / _cursors without the lock
                # inside the callback to avoid blocking.  Writes from the
                # main thread are infrequent and atomic-enough on CPython.
                buf = self._audio_buffers.get(sid)
                if buf is None:
                    continue

                cursor = self._cursors.get(sid, 0)
                buf_len = len(buf)

                # ---- Slice the next `frames` samples ----
                if src_info["loop"]:
                    # Looping: wrap around
                    chunk = np.empty(frames, dtype=np.float32)
                    remaining = frames
                    write_pos = 0
                    c = cursor
                    while remaining > 0:
                        available = min(remaining, buf_len - c)
                        chunk[write_pos : write_pos + available] = buf[c : c + available]
                        write_pos += available
                        remaining -= available
                        c = (c + available) % buf_len
                    self._cursors[sid] = c
                else:
                    # Non-looping: read what's available, zero-pad the rest
                    available = min(frames, buf_len - cursor)
                    if available <= 0:
                        # Source exhausted — mark not-playing
                        # (safe to write to shared_state from here because
                        #  set_source_playing just grabs the lock briefly)
                        self._shared_state.set_source_playing(sid, False)
                        continue
                    chunk = np.zeros(frames, dtype=np.float32)
                    chunk[:available] = buf[cursor : cursor + available]
                    self._cursors[sid] = cursor + available

                # ---- Store raw chunk for visualisation ----
                self._last_raw[sid] = chunk.copy()

                # ---- DSP processing ----
                fstate = self._filter_states.get(sid, {})
                self._filter_states[sid] = fstate

                stereo_block = dsp_engine.process_block(
                    input_block=chunk,
                    listener_pos=(float(listener_pos[0]), float(listener_pos[1])),
                    source_pos=(float(src_info["pos"][0]), float(src_info["pos"][1])),
                    walls=walls,
                    sample_rate=self.SAMPLE_RATE,
                    filter_state=fstate,
                )

                # ---- Store processed chunk for visualisation ----
                self._last_processed[sid] = stereo_block.copy()

                mix += stereo_block
                active_count += 1

        # ---- Soft-clip to prevent distortion when sources overlap ----
        if active_count > 1:
            mix = np.tanh(mix)

        self._last_mix = mix.copy()
        outdata[:] = mix

    # ------------------------------------------------------------------
    # Stream lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Open and start the audio output stream."""
        self._stream = sd.OutputStream(
            samplerate=self.SAMPLE_RATE,
            blocksize=self.BLOCKSIZE,  # TUNE: adjust for latency
            channels=self.CHANNELS,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop and close the audio output stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
