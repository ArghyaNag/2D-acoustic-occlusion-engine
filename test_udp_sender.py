"""
test_udp_sender.py — Throwaway test script for verifying network audio.

Sends sine-wave float32 PCM packets from TWO simulated clients (different
local ports) to localhost:50007.  Run alongside main.py to verify:
  1. Two network sources auto-spawn on the grid.
  2. Audio is audible with correct positional DSP.
  3. Stopping this script → sources disappear after ~5s timeout.

Usage:  python test_udp_sender.py
        (Ctrl+C to stop)
"""

import socket
import time
import sys

import numpy as np

SERVER = ("127.0.0.1", 50007)
SAMPLE_RATE = 44100
CHUNK = 1024  # samples per packet
SEND_INTERVAL = CHUNK / SAMPLE_RATE  # ~23 ms


def make_sine_chunk(freq: float, phase: float, chunk: int, sr: int):
    """Generate one chunk of a sine wave, returning (samples, new_phase)."""
    t = (np.arange(chunk) + phase) / sr
    samples = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return samples, phase + chunk


def main():
    # Two sockets = two different source ports = two "clients".
    sock1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind to ephemeral ports (OS picks) so each has a distinct (ip, port).
    sock1.bind(("127.0.0.1", 0))
    sock2.bind(("127.0.0.1", 0))

    port1 = sock1.getsockname()[1]
    port2 = sock2.getsockname()[1]
    print(f"Client 1: 127.0.0.1:{port1}  (440 Hz sine)")
    print(f"Client 2: 127.0.0.1:{port2}  (660 Hz sine)")
    print(f"Sending to {SERVER[0]}:{SERVER[1]}  —  Ctrl+C to stop")

    phase1 = 0.0
    phase2 = 0.0

    try:
        while True:
            samples1, phase1 = make_sine_chunk(440.0, phase1, CHUNK, SAMPLE_RATE)
            sock1.sendto(samples1.tobytes(), SERVER)

            samples2, phase2 = make_sine_chunk(660.0, phase2, CHUNK, SAMPLE_RATE)
            sock2.sendto(samples2.tobytes(), SERVER)

            time.sleep(SEND_INTERVAL)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sock1.close()
        sock2.close()


if __name__ == "__main__":
    main()
