from shared_state import SharedState
from audio_engine import AudioEngine
from render_engine import RenderEngine

def main() -> None:
    state = SharedState()
    audio = AudioEngine(state)
    renderer = RenderEngine(state, audio)
    audio.start()
    try:
        renderer.run()
    finally:
        audio.stop()
if __name__ == '__main__':
    main()
