from __future__ import annotations
import threading
from typing import Optional
import numpy as np
import sounddevice as sd
import soundfile as sf
import dsp_engine
from live_input import LiveInputStream
from network_input import NetworkInputServer
from shared_state import SharedState

class AudioEngine:
    BLOCKSIZE: int = 1024
    SAMPLE_RATE: int = 44100
    CHANNELS: int = 2

    def __init__(self, shared_state: SharedState) -> None:
        self._shared_state = shared_state
        self._lock = threading.Lock()
        self._audio_buffers: dict[int, np.ndarray] = {}
        self._cursors: dict[int, int] = {}
        self._filter_states: dict[int, dict] = {}
        self._last_raw: dict[int, Optional[np.ndarray]] = {}
        self._last_processed: dict[int, Optional[np.ndarray]] = {}
        self._last_mix: Optional[np.ndarray] = None
        self._stream: Optional[sd.OutputStream] = None
        self._live_input = LiveInputStream(sample_rate=self.SAMPLE_RATE)
        self._network_input = NetworkInputServer(sample_rate=self.SAMPLE_RATE)

    def load_audio_for_source(self, source_id: int, audio_path: str) -> None:
        data, file_sr = sf.read(audio_path, dtype='float32', always_2d=True)
        if data.shape[1] > 1:
            mono = data.mean(axis=1)
        else:
            mono = data[:, 0]
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
        with self._lock:
            self._audio_buffers.pop(source_id, None)
            self._cursors.pop(source_id, None)
            self._filter_states.pop(source_id, None)
            self._last_raw.pop(source_id, None)
            self._last_processed.pop(source_id, None)

    def get_last_raw(self, source_id: int) -> Optional[np.ndarray]:
        return self._last_raw.get(source_id)

    def get_last_processed(self, source_id: int) -> Optional[np.ndarray]:
        return self._last_processed.get(source_id)

    def get_last_mix(self) -> Optional[np.ndarray]:
        return self._last_mix

    def _callback(self, outdata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
        snapshot = self._shared_state.get_snapshot()
        listener_pos = snapshot['listener_pos']
        sources = snapshot['sources']
        walls = snapshot['walls']
        mix = np.zeros((frames, 2), dtype=np.float32)
        active_count = 0
        if listener_pos is not None:
            for sid, src_info in sources.items():
                if not src_info['playing']:
                    continue
                input_mode = src_info.get('input_mode', 'file')
                if input_mode == 'mic':
                    chunk = self._live_input.read_recent(frames)
                elif input_mode == 'network':
                    net_client = src_info.get('network_client')
                    if net_client is None:
                        continue
                    chunk = self._network_input.read_recent(net_client, frames)
                else:
                    buf = self._audio_buffers.get(sid)
                    if buf is None:
                        continue
                    cursor = self._cursors.get(sid, 0)
                    buf_len = len(buf)
                    if src_info['loop']:
                        chunk = np.empty(frames, dtype=np.float32)
                        remaining = frames
                        write_pos = 0
                        c = cursor
                        while remaining > 0:
                            available = min(remaining, buf_len - c)
                            chunk[write_pos:write_pos + available] = buf[c:c + available]
                            write_pos += available
                            remaining -= available
                            c = (c + available) % buf_len
                        self._cursors[sid] = c
                    else:
                        available = min(frames, buf_len - cursor)
                        if available <= 0:
                            self._shared_state.set_source_playing(sid, False)
                            continue
                        chunk = np.zeros(frames, dtype=np.float32)
                        chunk[:available] = buf[cursor:cursor + available]
                        self._cursors[sid] = cursor + available
                self._last_raw[sid] = chunk.copy()
                fstate = self._filter_states.get(sid, {})
                self._filter_states[sid] = fstate
                stereo_block = dsp_engine.process_block(input_block=chunk, listener_pos=(float(listener_pos[0]), float(listener_pos[1])), source_pos=(float(src_info['pos'][0]), float(src_info['pos'][1])), walls=walls, sample_rate=self.SAMPLE_RATE, filter_state=fstate)
                self._last_processed[sid] = stereo_block.copy()
                mix += stereo_block
                active_count += 1
        mix = np.tanh(mix)
        self._last_mix = mix.copy()
        outdata[:] = mix

    @property
    def network_input(self) -> NetworkInputServer:
        return self._network_input

    def start(self) -> None:
        self._live_input.start()
        self._network_input.start()
        self._stream = sd.OutputStream(samplerate=self.SAMPLE_RATE, blocksize=self.BLOCKSIZE, channels=self.CHANNELS, dtype='float32', callback=self._callback)
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._live_input.stop()
        self._network_input.stop()
