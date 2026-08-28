"""
network_input.py — UDP network audio receiver with per-client ring buffers.

Receives live mic audio streamed over UDP from remote clients on the same
LAN, each identified by its (ip, port) tuple.  Each client gets its own
ring buffer, identical in design to LiveInputStream's (monotonic pointers,
lock-free read/write hot path, overrun/underrun handling with safety
pre-rolling).

==========================================================================
WIRE PROTOCOL — reference for both server (this file) and client (Prompt B)
==========================================================================

Transport:  Raw UDP (Python ``socket`` module).  No TCP / WebSocket / WebRTC.

Server address:  Binds to 0.0.0.0:<PORT> (default PORT = 50007).

Audio packets
-------------
Each UDP datagram is a PURE PAYLOAD of mono float32 little-endian PCM
samples.  NO header bytes, NO length prefix, NO magic number.

    ┌─────────────────────────────────────┐
    │  N × float32 PCM samples (N × 4 B) │
    └─────────────────────────────────────┘

    - Sample rate: 44100 Hz (must match server).
    - Byte order:  little-endian (numpy default on x86).
    - Recommended chunk size: 1024 samples = 4096 bytes ≈ 23 ms.
      Any size is accepted; the receiver handles variable packet sizes.
    - Maximum practical size: ~16000 samples (64 KB UDP datagram limit),
      but smaller is better for latency.

Client identity
---------------
The client is identified by the UDP source (ip, port) tuple returned by
``recvfrom()``.  No explicit client-ID field is needed in the payload.

Trade-off: a client that restarts on a different ephemeral port appears
as a new client.  This is acceptable — it IS functionally a new session.

Join / handshake
----------------
IMPLICIT.  There is no separate "HELLO" or handshake packet.

The first audio packet received from a previously-unseen (ip, port) is
the join signal.  The server lazily allocates a ring buffer and registers
the client as active.  The main thread detects the new client on its next
``get_active_clients()`` poll and spawns a grid source for it.

Disconnect / leave
------------------
TIMEOUT-BASED.  UDP is connectionless; there is no "BYE" packet.

If no packet is received from a known (ip, port) for 5 seconds, the
server considers the client disconnected.  The main thread detects this
via ``get_active_clients()`` and removes the corresponding grid source.

    Timeout = 5 seconds
    - Long enough to survive brief Wi-Fi hiccups.
    - Short enough to clean up dead clients promptly.

A client that wants to "gracefully" leave can simply stop sending.  It
will be cleaned up after the timeout.
==========================================================================
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Per-client ring buffer (internal helper — not part of the public API)
# ---------------------------------------------------------------------------
class _ClientBuffer:
    """Ring buffer for one network client, same design as LiveInputStream.

    Ring buffer sizing rationale (network)
    --------------------------------------
    Buffer length = 2 seconds (2 × SAMPLE_RATE samples = 88200 @ 44100 Hz).
    Doubled vs. local mic's 1 second because network jitter (Wi-Fi
    retransmits, OS socket-buffer delays) is generally worse than local
    audio-thread jitter.  Memory cost: ~689 KiB of float32 per client —
    negligible even with dozens of clients.

    Safety threshold = 4096 samples ≈ 93 ms.
    Doubled vs. local mic's 2048 to absorb typical Wi-Fi jitter spikes.
    The output callback won't start reading until this many samples have
    accumulated, preventing choppy audio at connection start.
    """

    def __init__(self, buf_len: int) -> None:
        self._buf_len = buf_len
        self._buffer = np.zeros(buf_len, dtype=np.float32)

        # Monotonically increasing positions (total samples written/read).
        # Actual index = pos % buf_len.
        self._write_pos: int = 0
        self._read_pos: int = 0

        self._safety_threshold: int = 4096
        self._pre_rolling: bool = True

        # Timestamp of last received packet (time.monotonic).
        self.last_packet_time: float = time.monotonic()

    def write(self, samples: np.ndarray) -> None:
        """Append samples to the ring buffer.  Called from the recv thread."""
        n = len(samples)
        if n == 0:
            return

        pos = self._write_pos
        idx = pos % self._buf_len

        # Two-part copy if write wraps around the buffer boundary.
        first = min(n, self._buf_len - idx)
        self._buffer[idx : idx + first] = samples[:first]
        if first < n:
            self._buffer[0 : n - first] = samples[first:]

        # Advance write pointer (single atomic int store on CPython).
        self._write_pos = pos + n
        self.last_packet_time = time.monotonic()

    def read_recent(self, num_frames: int) -> np.ndarray:
        """Read continuous samples — same contract as LiveInputStream.read_recent.

        Safe for the audio callback: no blocking, no I/O, no heap allocation
        beyond the returned array.
        """
        out = np.zeros(num_frames, dtype=np.float32)

        write_pos = self._write_pos  # snapshot (atomic read on CPython)
        read_pos = self._read_pos

        # 1. Overrun: write pointer lapped the read pointer.
        if write_pos - read_pos > self._buf_len - num_frames:
            read_pos = max(0, write_pos - self._safety_threshold)
            self._pre_rolling = True

        # 2. Pre-rolling: wait until safety cushion is accumulated.
        if self._pre_rolling:
            if write_pos - read_pos >= self._safety_threshold:
                self._pre_rolling = False
            else:
                self._read_pos = read_pos
                return out

        # 3. Read available samples.
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class NetworkInputServer:
    """UDP server that receives audio from multiple remote clients.

    Each client is identified by its ``(ip, port)`` tuple and gets its
    own ring buffer.  The receive loop runs on a dedicated background
    thread — never on the PortAudio callback thread.

    Parameters
    ----------
    port : int
        UDP port to bind to.  Default 50007.
    sample_rate : int
        Expected sample rate of incoming audio.  Used to size ring buffers.
    client_timeout : float
        Seconds of silence before a client is considered disconnected.
    """

    DEFAULT_PORT: int = 50007
    BUFFER_SECONDS: float = 2.0
    CLIENT_TIMEOUT: float = 5.0

    def __init__(
        self,
        sample_rate: int = 44100,
        port: int = DEFAULT_PORT,
        client_timeout: float = CLIENT_TIMEOUT,
    ) -> None:
        self._sample_rate = sample_rate
        self._port = port
        self._client_timeout = client_timeout
        self._buf_len = int(sample_rate * self.BUFFER_SECONDS)

        # (ip, port) -> _ClientBuffer.  Protected by _clients_lock for
        # dict-level mutations (add/remove).  Individual buffer read/write
        # is lock-free (same as LiveInputStream).
        self._clients: dict[tuple[str, int], _ClientBuffer] = {}
        self._clients_lock = threading.Lock()

        self._socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False

    # ------------------------------------------------------------------
    # Receive loop (runs on its own background thread)
    # ------------------------------------------------------------------
    def _recv_loop(self) -> None:
        """Blocking receive loop.  Runs until ``_running`` is set False."""
        sock = self._socket
        if sock is None:
            return

        while self._running:
            try:
                data, addr = sock.recvfrom(65536)  # max UDP datagram
            except socket.timeout:
                continue  # lets us check _running periodically
            except OSError:
                # Socket closed during stop() — exit cleanly.
                break

            if len(data) < 4:
                # Too small to contain even one float32 — ignore.
                continue

            # Interpret payload as raw float32 PCM (little-endian).
            # Truncate any trailing bytes that don't form a complete float32.
            usable = len(data) - (len(data) % 4)
            samples = np.frombuffer(data[:usable], dtype=np.float32).copy()

            client_key = (addr[0], addr[1])

            # Lazy client creation (first packet = implicit join).
            buf = self._clients.get(client_key)
            if buf is None:
                buf = _ClientBuffer(self._buf_len)
                with self._clients_lock:
                    self._clients[client_key] = buf

            buf.write(samples)

    # ------------------------------------------------------------------
    # Public API — audio callback (real-time safe)
    # ------------------------------------------------------------------
    def read_recent(
        self, client_key: tuple[str, int], num_frames: int
    ) -> np.ndarray:
        """Read ``num_frames`` from a specific client's ring buffer.

        Safe to call from the PortAudio audio callback — non-blocking,
        no I/O, no heap allocation beyond the returned array.

        Returns zeros if the client is unknown or has no data yet.
        """
        buf = self._clients.get(client_key)
        if buf is None:
            return np.zeros(num_frames, dtype=np.float32)
        return buf.read_recent(num_frames)

    # ------------------------------------------------------------------
    # Public API — main thread
    # ------------------------------------------------------------------
    def get_active_clients(self) -> list[tuple[str, int]]:
        """Return currently-connected client keys (not timed out).

        Also prunes fully timed-out clients from the internal dict so
        their ring-buffer memory is freed.

        Called from the main/render thread — not real-time-critical but
        kept cheap (dict snapshot + timestamp comparison).
        """
        now = time.monotonic()
        active: list[tuple[str, int]] = []
        expired: list[tuple[str, int]] = []

        # Snapshot the client dict (cheap — just dict iteration).
        with self._clients_lock:
            for key, buf in self._clients.items():
                if now - buf.last_packet_time <= self._client_timeout:
                    active.append(key)
                else:
                    expired.append(key)

            for key in expired:
                del self._clients[key]

        return active

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Open the UDP socket and start the receive thread."""
        if self._running:
            return

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(("0.0.0.0", self._port))
        # Short timeout so the recv loop can check _running periodically.
        self._socket.settimeout(0.1)

        self._running = True
        self._thread = threading.Thread(
            target=self._recv_loop, name="network-audio-recv", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the receive thread and close the socket."""
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
        """The UDP port this server is bound to."""
        return self._port
