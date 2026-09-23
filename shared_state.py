"""
shared_state.py — Thread-safe state container.

This module is the ONLY communication channel between the Pygame render
loop (main thread) and the sounddevice audio callback (PortAudio thread).
All reads from the audio side go through get_snapshot(), which returns a
deep-copied, read-only dict so the audio thread never touches mutable
internals owned by the UI thread.
"""

from __future__ import annotations

import copy
import threading
from typing import Optional

# ---------------------------------------------------------------------------
# Grid constants (shared across the whole project)
# ---------------------------------------------------------------------------
GRID_COLS: int = 40
GRID_ROWS: int = 24
CELL_SIZE_PX: int = 24

# ---------------------------------------------------------------------------
# Wall material reflectivity-gain constants
# These are the per-corner amplitude multipliers applied to reflected paths.
# ---------------------------------------------------------------------------
WALL_GAIN_HARD: float   = 0.90  # Concrete / Brick — high reflectivity
WALL_GAIN_MEDIUM: float = 0.60  # Wood / Plaster  — moderate reflectivity
WALL_GAIN_SOFT: float   = 0.25  # Curtain / Carpet — high absorption
DEFAULT_WALL_GAIN: float = WALL_GAIN_MEDIUM  # applied when no gain specified

# ---------------------------------------------------------------------------
# Source record type (plain dict for simplicity in this skeleton)
# ---------------------------------------------------------------------------
# Each source is stored as:
#   {
#       "pos": (grid_x, grid_y),
#       "audio_path": str | None,
#       "loop": bool,
#       "playing": bool,
#       "input_mode": "file" | "mic" | "network",
#       "network_client": (ip, port) | None,
#   }


class SharedState:
    """Thread-safe state container for the positional-audio sandbox."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._listener_pos: Optional[tuple[int, int]] = None
        self._sources: dict[int, dict] = {}
        # Maps each wall cell (col, row) → its reflectivity gain scalar.
        self._walls: dict[tuple[int, int], float] = {}
        self._next_source_id: int = 1

    # ------------------------------------------------------------------
    # Snapshot (audio thread reads ONLY through this)
    # ------------------------------------------------------------------
    def get_snapshot(self) -> dict:
        """Return a deep-copied, read-only snapshot of the entire state.

        The audio callback calls this ONCE at the top of each invocation
        so it never holds the lock for more than the copy duration.
        """
        with self._lock:
            return {
                "listener_pos": self._listener_pos,
                "sources": copy.deepcopy(self._sources),
                # frozenset of positions — pathfinding.py receives this
                # unchanged; zero changes needed there.
                "walls": frozenset(self._walls.keys()),
                # flat dict copy for the DSP thread's per-corner gain lookup
                "wall_gains": dict(self._walls),
            }

    # ------------------------------------------------------------------
    # Listener
    # ------------------------------------------------------------------
    def set_listener_pos(self, pos: Optional[tuple[int, int]]) -> None:
        with self._lock:
            self._listener_pos = pos

    def get_listener_pos(self) -> Optional[tuple[int, int]]:
        with self._lock:
            return self._listener_pos

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------
    def add_source(self, pos: tuple[int, int]) -> int:
        """Add a new source at *pos* and return its source_id."""
        with self._lock:
            sid = self._next_source_id
            self._next_source_id += 1
            self._sources[sid] = {
                "pos": pos,
                "audio_path": None,
                "loop": True,
                "playing": False,
                "input_mode": "file",
                "network_client": None,
            }
            return sid

    def move_source(self, source_id: int, pos: tuple[int, int]) -> None:
        with self._lock:
            if source_id in self._sources:
                self._sources[source_id]["pos"] = pos

    def set_source_audio(self, source_id: int, audio_path: str) -> None:
        with self._lock:
            if source_id in self._sources:
                self._sources[source_id]["audio_path"] = audio_path
                self._sources[source_id]["playing"] = True

    def set_source_loop(self, source_id: int, loop: bool) -> None:
        with self._lock:
            if source_id in self._sources:
                self._sources[source_id]["loop"] = loop

    def set_source_playing(self, source_id: int, playing: bool) -> None:
        with self._lock:
            if source_id in self._sources:
                self._sources[source_id]["playing"] = playing

    def set_source_input_mode(self, source_id: int, mode: str) -> None:
        """Set the input mode for *source_id* to 'file', 'mic', or 'network'."""
        with self._lock:
            if source_id in self._sources:
                self._sources[source_id]["input_mode"] = mode

    def set_source_network_client(
        self, source_id: int, client_key: tuple[str, int] | None
    ) -> None:
        """Associate *source_id* with a network client (ip, port) tuple."""
        with self._lock:
            if source_id in self._sources:
                self._sources[source_id]["network_client"] = client_key

    def remove_source(self, source_id: int) -> None:
        with self._lock:
            self._sources.pop(source_id, None)

    def get_sources(self) -> dict[int, dict]:
        """Return a deep copy of all sources (for the UI thread)."""
        with self._lock:
            return copy.deepcopy(self._sources)

    def get_source(self, source_id: int) -> Optional[dict]:
        with self._lock:
            src = self._sources.get(source_id)
            return copy.deepcopy(src) if src is not None else None

    # ------------------------------------------------------------------
    # Walls
    # ------------------------------------------------------------------
    def add_wall(self, pos: tuple[int, int], gain: float = DEFAULT_WALL_GAIN) -> None:
        """Place a wall at *pos* with the given reflectivity *gain*.

        If a wall already exists at *pos*, its gain is overwritten with
        the new value (allows re-painting with a different material).
        """
        with self._lock:
            self._walls[pos] = gain

    def remove_wall(self, pos: tuple[int, int]) -> None:
        """Remove the wall at *pos*, silently ignoring missing cells."""
        with self._lock:
            self._walls.pop(pos, None)  # pop avoids KeyError during erase-drag

    def get_walls(self) -> set[tuple[int, int]]:
        """Return a copy of all wall positions (positions only, no gains).

        Callers that only need occupancy/passability (render paths, LoS
        checks, pathfinding) use this.  Audio-thread callers should read
        get_snapshot()["wall_gains"] instead.
        """
        with self._lock:
            return set(self._walls.keys())

    def get_wall_gains(self) -> dict[tuple[int, int], float]:
        """Return a shallow copy of the full wall→gain mapping (UI thread use)."""
        with self._lock:
            return dict(self._walls)

    def has_wall(self, pos: tuple[int, int]) -> bool:
        with self._lock:
            return pos in self._walls
