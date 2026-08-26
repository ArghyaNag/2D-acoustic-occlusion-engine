"""
live_input.py — Live microphone ring-buffer capture.

Opens a sounddevice InputStream on the default mic, captures mono audio
at the given sample rate, and stores it in a fixed-size circular ring
buffer that can be read without blocking by the output audio callback.

Ring buffer sizing rationale
----------------------------
Buffer length = 1 second (SAMPLE_RATE samples = 44100 @ 44100 Hz).
This is ~172 KiB of float32 — negligible memory.  The output callback
requests BLOCKSIZE=1024 frames (~23 ms) per invocation, so 1 second of
buffer provides >40× headroom.  Even if the UI thread stalls for
several hundred milliseconds (e.g. file-dialog open), the ring buffer
won't overrun in practice.  If it *does* overrun, the write pointer
simply wraps and overwrites the oldest data — no crash, no block.

Overrun handling:  The input callback always writes; the write pointer
wraps around and overwrites oldest unread data.  This is the correct
trade-off for real-time audio — losing stale samples is preferable to
blocking or crashing.

Underrun handling:  read_recent() returns zeros for any portion of the
requested window that hasn't been written yet (e.g. immediately after
start, or if the consumer requests more frames than are available).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import sounddevice as sd


class LiveInputStream:
    """Captures microphone audio into a lock-free circular ring buffer.

    The ring buffer is written by the sounddevice input callback (PortAudio
    thread) and read by the output callback (another PortAudio thread) via
    ``read_recent()``.  Both are real-time threads — neither may block.

    Thread-safety note
    ------------------
    ``_write_pos`` is a plain Python int.  On CPython, int reads/writes are
    atomic (GIL guarantees pointer-width stores are atomic).  The reader
    (``read_recent``) reads ``_write_pos`` once and computes the read
    window from that snapshot; the writer only ever increments it.  Worst
    case is reading a slightly stale ``_write_pos`` — yielding one extra
    old sample, which is inaudible.  No lock is needed.
    """

    def __init__(self, sample_rate: int = 44100, buffer_seconds: float = 1.0) -> None:
        self._sample_rate = sample_rate
        self._buf_len = int(sample_rate * buffer_seconds)

        # Pre-allocated ring buffer — written by the input callback.
        self._buffer = np.zeros(self._buf_len, dtype=np.float32)

        # Monotonically increasing write position (total samples written).
        # The actual index into _buffer is  _write_pos % _buf_len.
        self._write_pos: int = 0

        # Monotonically increasing read position (total samples read).
        # The actual index into _buffer is  _read_pos % _buf_len.
        self._read_pos: int = 0

        # Safety/pre-rolling threshold to handle jitter/latency (2 blocks = 2048 samples)
        self._safety_threshold: int = 2048
        self._pre_rolling: bool = True

        self._stream: Optional[sd.InputStream] = None
        self._active: bool = False

    # ------------------------------------------------------------------
    # Input callback (PortAudio thread — NO blocking / alloc / IO!)
    # ------------------------------------------------------------------
    def _input_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Copy incoming mic frames into the ring buffer."""
        # indata is (frames, channels); take channel 0 as mono.
        mono = indata[:, 0]
        n = len(mono)
        pos = self._write_pos
        idx = pos % self._buf_len

        # Two-part copy if the write wraps around the buffer boundary.
        first = min(n, self._buf_len - idx)
        self._buffer[idx : idx + first] = mono[:first]
        if first < n:
            self._buffer[0 : n - first] = mono[first:]

        # Advance the write pointer (single atomic int store on CPython).
        self._write_pos = pos + n

    # ------------------------------------------------------------------
    # Reader API (called from the *output* callback — also real-time)
    # ------------------------------------------------------------------
    def read_recent(self, num_frames: int) -> np.ndarray:
        """Read continuous samples from the ring buffer.

        Maintains an internal read pointer to avoid repeating or skipping
        samples. Handles overrun and underrun with zero padding and safety
        pre-rolling.

        Returns
        -------
        np.ndarray
            Mono float32 array of shape ``(num_frames,)``.
        """
        out = np.zeros(num_frames, dtype=np.float32)

        write_pos = self._write_pos  # snapshot (atomic read on CPython)
        read_pos = self._read_pos

        # 1. Overrun check: Has the write pointer wrapped and overtaken our read pointer?
        if write_pos - read_pos > self._buf_len - num_frames:
            # Skip read pointer forward to catch up, leaving safety margin.
            read_pos = max(0, write_pos - self._safety_threshold)
            self._pre_rolling = True

        # 2. Check pre-rolling: Wait until we accumulate a safety cushion.
        if self._pre_rolling:
            if write_pos - read_pos >= self._safety_threshold:
                self._pre_rolling = False
            else:
                # Still pre-rolling, return silence (zeros).
                self._read_pos = read_pos
                return out

        # 3. Read up to num_frames samples.
        available = write_pos - read_pos
        if available <= 0:
            # Complete underrun.
            self._pre_rolling = True
            return out

        to_read = min(num_frames, available)
        start_idx = read_pos % self._buf_len
        end_idx = (read_pos + to_read) % self._buf_len

        if start_idx < end_idx:
            out[:to_read] = self._buffer[start_idx:end_idx]
        else:
            first_part = self._buf_len - start_idx
            out[:first_part] = self._buffer[start_idx:]
            out[first_part:to_read] = self._buffer[:end_idx]

        # 4. If we had an underrun, trigger pre-rolling to rebuild safety cushion.
        if to_read < num_frames:
            self._pre_rolling = True

        # Advance persistent read position.
        self._read_pos = read_pos + to_read
        return out

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Open the default mic and start capturing."""
        if self._active:
            return
        # Reset positions and state on start
        self._write_pos = 0
        self._read_pos = 0
        self._pre_rolling = True
        try:
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                callback=self._input_callback,
            )
            self._stream.start()
            self._active = True
        except Exception:
            # Mic not available / permission denied — degrade gracefully.
            # read_recent() will keep returning zeros, which is fine.
            self._stream = None
            self._active = False

    def stop(self) -> None:
        """Stop and close the input stream."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active
