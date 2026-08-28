"""
remote_client.py — Mode 3 remote mic-streaming client.

Standalone program that runs on a second PC/laptop on the same LAN.
Captures live microphone audio and streams it over UDP to the
positional-audio sandbox's NetworkInputServer.

Wire protocol (must match network_input.py exactly):
  - Transport: raw UDP (SOCK_DGRAM) to server IP:port (default 50007).
  - Payload: raw mono float32 little-endian PCM samples, NO header.
  - Sample rate: 44100 Hz (server assumes this, no resampling).
  - Chunk size: 1024 samples (4096 bytes, ~23 ms) recommended.
  - Identity: server identifies client by UDP source (ip, port).
  - Join: implicit on first packet (no handshake).
  - Disconnect: server times out after 5s of silence.

Usage:
    python remote_client.py --server-ip 192.168.1.100
    python remote_client.py --server-ip 127.0.0.1 --server-port 50007
    python remote_client.py --server-ip 192.168.1.100 --chunk-size 512
"""

from __future__ import annotations

import argparse
import queue
import socket
import sys
import threading
import time

import numpy as np
import sounddevice as sd

SAMPLE_RATE: int = 44100


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mode 3 remote client — streams live mic audio over UDP"
    )
    parser.add_argument(
        "--server-ip",
        required=True,
        help="IP address of the listener server (e.g. 192.168.1.100)",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=50007,
        help="UDP port of the listener server (default: 50007)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Samples per packet (default: 1024 = ~23 ms)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dest = (args.server_ip, args.server_port)
    chunk = args.chunk_size

    print(f"Remote client — streaming mic to {dest[0]}:{dest[1]}")
    print(f"Sample rate: {SAMPLE_RATE} Hz, chunk: {chunk} samples "
          f"(~{chunk / SAMPLE_RATE * 1000:.1f} ms/packet)")

    # --- UDP socket (non-blocking send, no bind needed) ---
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # --- Thread-safe queue: callback -> sender thread ---
    audio_q: queue.Queue[bytes] = queue.Queue(maxsize=200)

    # Shared counters for status display (written by sender thread,
    # read by main thread — plain int, atomic on CPython).
    packets_sent = 0
    sender_running = True

    # --- Sender thread: drains the queue and does socket I/O ---
    def sender_loop() -> None:
        nonlocal packets_sent
        while sender_running or not audio_q.empty():
            try:
                payload = audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                sock.sendto(payload, dest)
                packets_sent += 1
            except OSError:
                # Socket closed during shutdown — exit quietly.
                break

    sender_thread = threading.Thread(
        target=sender_loop, name="udp-sender", daemon=True
    )

    # --- sounddevice input callback (PortAudio thread) ---
    # No blocking I/O here — just enqueue the raw bytes.
    def input_callback(
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        # indata is (frames, 1) float32 mono.  Send as raw bytes.
        try:
            audio_q.put_nowait(indata[:, 0].tobytes())
        except queue.Full:
            # Drop the oldest chunk to avoid unbounded growth, then retry.
            try:
                audio_q.get_nowait()
            except queue.Empty:
                pass
            try:
                audio_q.put_nowait(indata[:, 0].tobytes())
            except queue.Full:
                pass  # truly jammed — drop this chunk silently

    # --- Open mic ---
    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=chunk,
            channels=1,
            dtype="float32",
            callback=input_callback,
        )
    except (sd.PortAudioError, Exception) as exc:
        print(f"\nError: could not open microphone — {exc}")
        print("Check that a mic is connected and permissions are granted.")
        sock.close()
        sys.exit(1)

    # --- Start everything ---
    sender_thread.start()
    stream.start()
    print("Mic opened — streaming.  Press Ctrl+C to stop.\n")

    # --- Status heartbeat (main thread, ~1 Hz) ---
    try:
        start_time = time.monotonic()
        while True:
            time.sleep(1.0)
            elapsed = time.monotonic() - start_time
            print(f"  [{elapsed:6.1f}s]  packets sent: {packets_sent}")
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        # Clean shutdown: stop mic first, then drain queue, then socket.
        stream.stop()
        stream.close()
        sender_running = False
        sender_thread.join(timeout=2.0)
        sock.close()
        print(f"Done. Sent {packets_sent} packets total.")


if __name__ == "__main__":
    main()
