"""
main.py — Entry point for the 3D Positional Audio Sandbox (Mode 1).

Instantiates the shared state, audio engine, and render engine, wires
them together, starts the audio stream, runs the Pygame loop, and
cleanly shuts down on window close.
"""

from shared_state import SharedState
from audio_engine import AudioEngine
from render_engine import RenderEngine


def main() -> None:
    state = SharedState()
    audio = AudioEngine(state)
    renderer = RenderEngine(state, audio)

    audio.start()
    try:
        renderer.run()  # blocks until the Pygame window is closed
    finally:
        audio.stop()


if __name__ == "__main__":
    main()
