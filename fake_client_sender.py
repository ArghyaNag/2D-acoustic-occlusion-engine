"""
fake_client_sender.py — Throwaway test script to simulate a remote
Mode 3 client sending mic-like audio over UDP.

This is NOT the real Prompt B client. It's just enough to verify that
the server's network_input.py receiver (from Prompt A) actually works:
receives packets, ring-buffers them, and the main app auto-spawns a
source and plays the audio positionally.

USAGE
-----
    python fake_client_sender.py
    python fake_client_sender.py --port 50007 --source-port 0
    python fake_client_sender.py --tone 440          # send a sine tone
    python fake_client_sender.py --wav myfile.wav    # send a WAV file instead

Run TWO copies at once (different terminals) with different
--source-port values (or just leave source-port at 0 / default, since
the OS will pick different ephemeral ports automatically per process)
to simulate two simultaneous "clients."

BEFORE RUNNING: check Opus's Prompt-A summary for the ACTUAL port
number and chunk size it chose, and pass them via --port / --chunk if
they differ from the defaults below (50007 / 1024 — these are just
placeholders matching what was suggested in the Prompt A spec).
"""

from __future__ import annotations

import argparse
import socket
import sys
import time

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None  # only needed if --wav is used


SAMPLE_RATE = 44100


def generate_tone(freq: float, seconds: float = 3600.0) -> np.ndarray:
    """Generate a mono float32 sine wave. Defaults to a long (1hr) buffer
    that will just loop the underlying array; we stream it in chunks."""
    n = int(SAMPLE_RATE * 2.0)  # generate 2 seconds, loop it while sending
    t = np.arange(n) / SAMPLE_RATE
    wave = 0.3 * np.sin(2 * np.pi * freq * t)  # 0.3 amplitude, not full-scale
    return wave.astype(np.float32)


def load_wav(path: str) -> np.ndarray:
    if sf is None:
        print("soundfile is not installed — pip install soundfile, or use --tone instead.")
        sys.exit(1)
    data, file_sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if file_sr != SAMPLE_RATE:
        # crude resample, good enough for a test sender
        ratio = SAMPLE_RATE / file_sr
        idx = np.round(np.arange(0, len(mono), 1.0 / ratio)).astype(int)
        idx = idx[idx < len(mono)]
        mono = mono[idx]
    return mono.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fake Mode 3 UDP client sender")
    parser.add_argument("--host", default="127.0.0.1", help="Server IP (default: localhost)")
    parser.add_argument("--port", type=int, default=50007, help="Server UDP port (check Opus's summary!)")
    parser.add_argument("--source-port", type=int, default=0, help="Local port to bind to (0 = OS picks)")
    parser.add_argument("--chunk", type=int, default=1024, help="Samples per packet (check Opus's summary!)")
    parser.add_argument("--tone", type=float, default=440.0, help="Sine tone frequency in Hz (default 440)")
    parser.add_argument("--wav", type=str, default=None, help="Path to a WAV file to send instead of a tone")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run (0 = run forever until Ctrl+C)")
    args = parser.parse_args()

    if args.wav:
        print(f"Loading {args.wav} ...")
        audio = load_wav(args.wav)
    else:
        print(f"Generating a {args.tone} Hz test tone ...")
        audio = generate_tone(args.tone)

    n_samples = len(audio)
    chunk = args.chunk
    block_duration = chunk / SAMPLE_RATE  # seconds per packet, for real-time pacing

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.source_port))
    bound_port = sock.getsockname()[1]

    dest = (args.host, args.port)
    print(f"Sending to {dest} from local port {bound_port}")
    print(f"Chunk size: {chunk} samples (~{block_duration*1000:.1f} ms/packet)")
    print("Press Ctrl+C to stop.\n")

    pos = 0
    packets_sent = 0
    start_time = time.time()

    try:
        while True:
            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                break

            # Slice the next chunk, wrapping around (loop the tone/wav)
            end = pos + chunk
            if end <= n_samples:
                block = audio[pos:end]
                pos = end % n_samples
            else:
                first = audio[pos:n_samples]
                remaining = chunk - len(first)
                second = audio[0:remaining]
                block = np.concatenate([first, second])
                pos = remaining

            packet_bytes = block.astype(np.float32).tobytes()
            sock.sendto(packet_bytes, dest)
            packets_sent += 1

            if packets_sent % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  sent {packets_sent} packets ({elapsed:.1f}s elapsed)")

            time.sleep(block_duration)  # real-time pacing, mimics a live mic stream

    except KeyboardInterrupt:
        print(f"\nStopped after {packets_sent} packets.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
