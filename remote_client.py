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
    parser = argparse.ArgumentParser(description='Mode 3 remote client — streams live mic audio over UDP')
    parser.add_argument('--server-ip', required=True, help='IP address of the listener server (e.g. 192.168.1.100)')
    parser.add_argument('--server-port', type=int, default=50007, help='UDP port of the listener server (default: 50007)')
    parser.add_argument('--chunk-size', type=int, default=1024, help='Samples per packet (default: 1024 = ~23 ms)')
    return parser.parse_args()

def main() -> None:
    args = _parse_args()
    dest = (args.server_ip, args.server_port)
    chunk = args.chunk_size
    print(f'Remote client — streaming mic to {dest[0]}:{dest[1]}')
    print(f'Sample rate: {SAMPLE_RATE} Hz, chunk: {chunk} samples (~{chunk / SAMPLE_RATE * 1000:.1f} ms/packet)')
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    audio_q: queue.Queue[bytes] = queue.Queue(maxsize=200)
    packets_sent = 0
    sender_running = True

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
                break
    sender_thread = threading.Thread(target=sender_loop, name='udp-sender', daemon=True)

    def input_callback(indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
        try:
            audio_q.put_nowait(indata[:, 0].tobytes())
        except queue.Full:
            try:
                audio_q.get_nowait()
            except queue.Empty:
                pass
            try:
                audio_q.put_nowait(indata[:, 0].tobytes())
            except queue.Full:
                pass
    try:
        stream = sd.InputStream(samplerate=SAMPLE_RATE, blocksize=chunk, channels=1, dtype='float32', callback=input_callback)
    except (sd.PortAudioError, Exception) as exc:
        print(f'\nError: could not open microphone — {exc}')
        print('Check that a mic is connected and permissions are granted.')
        sock.close()
        sys.exit(1)
    sender_thread.start()
    stream.start()
    print('Mic opened — streaming.  Press Ctrl+C to stop.\n')
    try:
        start_time = time.monotonic()
        while True:
            time.sleep(1.0)
            elapsed = time.monotonic() - start_time
            print(f'  [{elapsed:6.1f}s]  packets sent: {packets_sent}')
    except KeyboardInterrupt:
        print('\n\nShutting down...')
    finally:
        stream.stop()
        stream.close()
        sender_running = False
        sender_thread.join(timeout=2.0)
        sock.close()
        print(f'Done. Sent {packets_sent} packets total.')
if __name__ == '__main__':
    main()
