from __future__ import annotations
import socket
import threading
import time
from typing import Optional
import numpy as np

class _ClientBuffer:

    def __init__(self, buf_len: int) -> None:
        self._buf_len = buf_len
        self._buffer = np.zeros(buf_len, dtype=np.float32)
        self._write_pos: int = 0
        self._read_pos: int = 0
        self._safety_threshold: int = 4096
        self._pre_rolling: bool = True
        self.last_packet_time: float = time.monotonic()

    def write(self, samples: np.ndarray) -> None:
        n = len(samples)
        if n == 0:
            return
        pos = self._write_pos
        idx = pos % self._buf_len
        first = min(n, self._buf_len - idx)
        self._buffer[idx:idx + first] = samples[:first]
        if first < n:
            self._buffer[0:n - first] = samples[first:]
        self._write_pos = pos + n
        self.last_packet_time = time.monotonic()

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

class NetworkInputServer:
    DEFAULT_PORT: int = 50007
    BUFFER_SECONDS: float = 2.0
    CLIENT_TIMEOUT: float = 5.0

    def __init__(self, sample_rate: int=44100, port: int=DEFAULT_PORT, client_timeout: float=CLIENT_TIMEOUT) -> None:
        self._sample_rate = sample_rate
        self._port = port
        self._client_timeout = client_timeout
        self._buf_len = int(sample_rate * self.BUFFER_SECONDS)
        self._clients: dict[tuple[str, int], _ClientBuffer] = {}
        self._clients_lock = threading.Lock()
        self._socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False

    def _recv_loop(self) -> None:
        sock = self._socket
        if sock is None:
            return
        while self._running:
            try:
                data, addr = sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(data) < 4:
                continue
            usable = len(data) - len(data) % 4
            samples = np.frombuffer(data[:usable], dtype=np.float32).copy()
            client_key = (addr[0], addr[1])
            buf = self._clients.get(client_key)
            if buf is None:
                buf = _ClientBuffer(self._buf_len)
                with self._clients_lock:
                    self._clients[client_key] = buf
            buf.write(samples)

    def read_recent(self, client_key: tuple[str, int], num_frames: int) -> np.ndarray:
        buf = self._clients.get(client_key)
        if buf is None:
            return np.zeros(num_frames, dtype=np.float32)
        return buf.read_recent(num_frames)

    def get_active_clients(self) -> list[tuple[str, int]]:
        now = time.monotonic()
        active: list[tuple[str, int]] = []
        expired: list[tuple[str, int]] = []
        with self._clients_lock:
            for key, buf in self._clients.items():
                if now - buf.last_packet_time <= self._client_timeout:
                    active.append(key)
                else:
                    expired.append(key)
            for key in expired:
                del self._clients[key]
        return active

    def start(self) -> None:
        if self._running:
            return
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(('0.0.0.0', self._port))
        self._socket.settimeout(0.1)
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, name='network-audio-recv', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._clients_lock:
            self._clients.clear()

    @property
    def port(self) -> int:
        return self._port
