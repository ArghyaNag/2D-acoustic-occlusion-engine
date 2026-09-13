from __future__ import annotations
from typing import Optional
import numpy as np
import sounddevice as sd

class LiveInputStream:

    def __init__(self, sample_rate: int=44100, buffer_seconds: float=1.0) -> None:
        self._sample_rate = sample_rate
        self._buf_len = int(sample_rate * buffer_seconds)
        self._buffer = np.zeros(self._buf_len, dtype=np.float32)
        self._write_pos: int = 0
        self._read_pos: int = 0
        self._safety_threshold: int = 2048
        self._pre_rolling: bool = True
        self._stream: Optional[sd.InputStream] = None
        self._active: bool = False

    def _input_callback(self, indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
        mono = indata[:, 0]
        n = len(mono)
        pos = self._write_pos
        idx = pos % self._buf_len
        first = min(n, self._buf_len - idx)
        self._buffer[idx:idx + first] = mono[:first]
        if first < n:
            self._buffer[0:n - first] = mono[first:]
        self._write_pos = pos + n

    def read_recent(self, num_frames: int) -> np.ndarray:
        out = np.zeros(num_frames, dtype=np.float32)
        write_pos = self._write_pos
        read_pos = self._read_pos
        if write_pos - read_pos > self._buf_len - num_frames:
            read_pos = max(0, write_pos - self._safety_threshold)
            self._pre_rolling = True
        if self._pre_rolling:
            if write_pos - read_pos >= self._safety_threshold:
                self._pre_rolling = False
            else:
                self._read_pos = read_pos
                return out
        available = write_pos - read_pos
        if available <= 0:
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
        if to_read < num_frames:
            self._pre_rolling = True
        self._read_pos = read_pos + to_read
        return out

    def start(self) -> None:
        if self._active:
            return
        self._write_pos = 0
        self._read_pos = 0
        self._pre_rolling = True
        try:
            self._stream = sd.InputStream(samplerate=self._sample_rate, channels=1, dtype='float32', callback=self._input_callback)
            self._stream.start()
            self._active = True
        except Exception:
            self._stream = None
            self._active = False

    def stop(self) -> None:
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
