"""
render_engine.py — Pygame-based UI for the positional-audio sandbox.

Runs entirely on the main thread.  Draws the grid, handles mouse
interaction (drag-drop from palette, wall painting, source selection),
and renders waveform graphs in the right sidebar.

Layout:
  ┌──────────┬────────────────────────┬──────────────┐
  │  LEFT    │       GRID AREA        │    RIGHT     │
  │ SIDEBAR  │ GRID_COLS * cell_size  │   SIDEBAR    │
  │  250 px  │                        │  (flexible)  │
  └──────────┴────────────────────────┴──────────────┘
"""

from __future__ import annotations
from time import thread_time


import os
import sys
from typing import Optional

import numpy as np
import pygame

from shared_state import (
    SharedState, GRID_COLS, GRID_ROWS, CELL_SIZE_PX,
    WALL_GAIN_HARD, WALL_GAIN_MEDIUM, WALL_GAIN_SOFT,
)
from audio_engine import AudioEngine
from pathfinding import find_k_paths
from viz_engine import (
    compute_display_samples,
    compute_spectrum,
    downsample_for_width,
    RAW_DISPLAY_MULTIPLIER,
    PROCESSED_DISPLAY_MULTIPLIER,
)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
LEFT_SIDEBAR_W: int = 250
RIGHT_SIDEBAR_W: int = 300
GRID_W: int = GRID_COLS * CELL_SIZE_PX
GRID_H: int = GRID_ROWS * CELL_SIZE_PX
WINDOW_W: int = LEFT_SIDEBAR_W + GRID_W + RIGHT_SIDEBAR_W
WINDOW_H: int = GRID_H

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COL_BG          = (24, 24, 30)
COL_SIDEBAR_BG  = (30, 30, 38)
COL_GRID_LINE   = (50, 50, 60)
COL_WALL        = (70, 70, 80)
COL_LISTENER    = (70, 140, 255)
COL_SOURCE      = (255, 160, 50)
COL_SOURCE_SEL  = (255, 220, 80)
COL_TEXT        = (210, 210, 220)
COL_TEXT_DIM    = (130, 130, 145)
COL_BUTTON      = (55, 55, 70)
COL_BUTTON_ACT  = (80, 100, 180)
COL_GRAPH_BG    = (20, 20, 28)
COL_GRAPH_RAW   = (100, 200, 100)
COL_GRAPH_PROC  = (100, 160, 255)
COL_GRAPH_SPECTRUM = (255, 180, 80)
COL_PALETTE_LISTENER = (55, 100, 200)
COL_PALETTE_SOURCE   = (200, 120, 30)
COL_PATH_PRIMARY     = (50, 200, 180)
COL_PATH_REFLECTED   = (200, 100, 220)

# ---------------------------------------------------------------------------
# Wall material definitions (gain, display color, label)
# ---------------------------------------------------------------------------
_WALL_MATERIALS: list[dict] = [
    {"gain": WALL_GAIN_HARD,   "color": (160, 165, 175), "label": "Hard  (Concrete)"},
    {"gain": WALL_GAIN_MEDIUM, "color": (185, 155, 110), "label": "Med   (Wood)"},
    {"gain": WALL_GAIN_SOFT,   "color": ( 80, 155, 145), "label": "Soft  (Curtain)"},
]
# Map gain -> swatch color for fast lookup in _draw_walls()
_GAIN_TO_COLOR: dict[float, tuple[int, int, int]] = {
    m["gain"]: m["color"] for m in _WALL_MATERIALS
}


# ---------------------------------------------------------------------------
# Helper: tkinter file dialog (imported lazily)
# ---------------------------------------------------------------------------
def _open_file_dialog() -> Optional[str]:
    """Open a native file dialog and return the selected path, or None."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio files", "*.wav *.flac *.ogg *.mp3"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return path if path else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Wall tool modes
# ---------------------------------------------------------------------------
WALL_OFF  = 0
WALL_DRAW = 1
WALL_ERASE = 2
WALL_LABELS = {WALL_OFF: "Wall: OFF", WALL_DRAW: "Wall: DRAW", WALL_ERASE: "Wall: ERASE"}


class RenderEngine:
    """Pygame render loop for the 2D positional-audio sandbox."""

    def __init__(
        self,
        shared_state: SharedState,
        audio_engine: AudioEngine,
    ) -> None:
        self._state = shared_state
        self._audio = audio_engine

        pygame.init()
        pygame.display.set_caption("3D Positional Audio Sandbox — Mode 1")
        info = pygame.display.Info()

        # FIX 2: Subtract margin so window has visible title bar and chrome
        window_w = info.current_w - 100
        window_h = info.current_h - 100
        self._screen = pygame.display.set_mode((window_w, window_h), pygame.RESIZABLE)

        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("consolas", 14)
        self._font_sm = pygame.font.SysFont("consolas", 12)
        self._font_lg = pygame.font.SysFont("consolas", 16, bold=True)

        # FIX 1: Dynamic cell size attribute
        self._cell_size: int = CELL_SIZE_PX
        self._update_layout()

        # ---- UI state ----
        self._running: bool = True
        self._dragging: Optional[str] = None  # "listener", "source", or None (palette drag)
        self._dragging_source_id: Optional[int] = None  # when dragging an existing source
        self._drag_mouse_pos: tuple[int, int] = (0, 0)
        self._wall_tool: int = WALL_OFF
        self._wall_painting: bool = False  # True while mouse is held for wall drawing
        self._wall_material: float = WALL_GAIN_MEDIUM  # currently selected material gain
        self._mat_swatch_rects: list[pygame.Rect] = []  # populated in _draw_left_sidebar
        self._selected_source_id: Optional[int] = None

        # Control rects
        self._load_audio_btn_rect: Optional[pygame.Rect] = None
        self._loop_toggle_rect: Optional[pygame.Rect] = None
        self._pause_btn_rect: Optional[pygame.Rect] = None
        self._input_mode_btn_rect: Optional[pygame.Rect] = None

    def _update_layout(self) -> None:
        """Recompute dynamic cell size based on current window dimensions."""
        win_w, win_h = self._screen.get_size()
        avail_w = max(1, win_w - LEFT_SIDEBAR_W - RIGHT_SIDEBAR_W)
        avail_h = max(1, win_h)
        self._cell_size = max(1, min(avail_w // GRID_COLS, avail_h // GRID_ROWS))

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    def _grid_origin(self) -> tuple[int, int]:
        """Top-left pixel of the grid area."""
        return (LEFT_SIDEBAR_W, 0)

    def _pixel_to_grid(self, px: int, py: int) -> Optional[tuple[int, int]]:
        """Convert pixel coords to grid cell, or None if outside the grid."""
        gx_origin, gy_origin = self._grid_origin()
        gx = (px - gx_origin) // self._cell_size
        gy = (py - gy_origin) // self._cell_size
        if 0 <= gx < GRID_COLS and 0 <= gy < GRID_ROWS:
            return (gx, gy)
        return None

    def _grid_to_pixel_center(self, gx: int, gy: int) -> tuple[int, int]:
        """Return the pixel center of grid cell (gx, gy)."""
        ox, oy = self._grid_origin()
        return (
            ox + gx * self._cell_size + self._cell_size // 2,
            oy + gy * self._cell_size + self._cell_size // 2,
        )

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def _draw_grid(self) -> None:
        ox, oy = self._grid_origin()
        grid_draw_w = GRID_COLS * self._cell_size
        grid_draw_h = GRID_ROWS * self._cell_size

        # Background
        pygame.draw.rect(self._screen, COL_BG, (ox, oy, grid_draw_w, grid_draw_h))

        # Vertical lines
        for c in range(GRID_COLS + 1):
            x = ox + c * self._cell_size
            pygame.draw.line(self._screen, COL_GRID_LINE, (x, oy), (x, oy + grid_draw_h))

        # Horizontal lines
        for r in range(GRID_ROWS + 1):
            y = oy + r * self._cell_size
            pygame.draw.line(self._screen, COL_GRID_LINE, (ox, y), (ox + grid_draw_w, y))

    def _draw_walls(self) -> None:
        ox, oy = self._grid_origin()
        wall_gains = self._state.get_wall_gains()  # {pos: gain}
        for (wx, wy), gain in wall_gains.items():
            # Map the stored gain to its swatch color; fall back to the
            # neutral COL_WALL if the gain value is somehow unrecognised.
            color = _GAIN_TO_COLOR.get(gain, COL_WALL)
            rect = pygame.Rect(
                ox + wx * self._cell_size + 1,
                oy + wy * self._cell_size + 1,
                self._cell_size - 1,
                self._cell_size - 1,
            )
            pygame.draw.rect(self._screen, color, rect)

    def _draw_listener(self) -> None:
        lp = self._state.get_listener_pos()
        if lp is None:
            return
        cx, cy = self._grid_to_pixel_center(*lp)
        radius = max(2, self._cell_size // 2 - 2)
        pygame.draw.circle(self._screen, COL_LISTENER, (cx, cy), radius)

        # Forward-axis indicator (small line pointing +x)
        pygame.draw.line(
            self._screen,
            (200, 220, 255),
            (cx, cy),
            (cx + self._cell_size // 2 + 2, cy),
            2,
        )
        label = self._font_sm.render("L", True, (255, 255, 255))
        self._screen.blit(label, (cx - label.get_width() // 2, cy - label.get_height() // 2))

    def _draw_sources(self) -> None:
        sources = self._state.get_sources()
        for sid, info in sources.items():
            px, py = self._grid_to_pixel_center(*info["pos"])
            col = COL_SOURCE_SEL if sid == self._selected_source_id else COL_SOURCE
            radius = max(2, self._cell_size // 2 - 2)
            pygame.draw.circle(self._screen, col, (px, py), radius)
            label = self._font_sm.render(f"S{sid}", True, (0, 0, 0))
            self._screen.blit(label, (px - label.get_width() // 2, py - label.get_height() // 2))

    # ------------------------------------------------------------------
    # Path visualization (drawn BEFORE listener/source circles)
    # ------------------------------------------------------------------
    def _draw_dashed_polyline(self, points, color, width=2, dash_len=6.0, gap_len=4.0):
        if len(points) < 2:
            return
        dash_on = True
        remaining = dash_len
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
            seg_dx, seg_dy = x1 - x0, y1 - y0
            seg_len = (seg_dx ** 2 + seg_dy ** 2) ** 0.5
            if seg_len == 0:
                continue
            travelled = 0.0
            while travelled < seg_len:
                step = min(remaining, seg_len - travelled)
                t0, t1 = travelled / seg_len, (travelled + step) / seg_len
                if dash_on:
                    pygame.draw.line(
                        self._screen, color,
                        (x0 + seg_dx * t0, y0 + seg_dy * t0),
                        (x0 + seg_dx * t1, y0 + seg_dy * t1),
                        width,
                    )
                travelled += step
                remaining -= step
                if remaining <= 0:
                    dash_on = not dash_on
                    remaining = dash_len if dash_on else gap_len   
   
   
    def _draw_source_paths(self) -> None:
        """Draw pathfinding routes from each source to the listener on the grid.

        Calls find_k_paths independently on the render thread (safe — pathfinding
        is pure and stateless).  This duplicates the computation already done on
        the audio thread inside process_block, which is intentional: sharing
        results across threads would violate this project's shared_state
        architecture.  A single k=2 pathfinding call is <1ms on a 40x24 grid,
        so even several sources at 60fps is negligible.
        """
        listener_pos = self._state.get_listener_pos()
        if listener_pos is None:
            return

        sources = self._state.get_sources()
        walls = self._state.get_walls()

        for sid in sorted(sources.keys()):
            source_pos = sources[sid]["pos"]
            paths = find_k_paths(
                listener_pos, source_pos, walls, GRID_COLS, GRID_ROWS, k=2,
            )
            if not paths:
                continue  # Total occlusion — no path to draw.

            # --- Primary path: solid teal line ---
            primary_cells = paths[0].cells
            if len(primary_cells) >= 2:
                points = [
                    self._grid_to_pixel_center(*cell)
                    for cell in primary_cells
                ]
                pygame.draw.lines(
                    self._screen, COL_PATH_PRIMARY, False, points, 2,
                )

            # --- Reflected path: dotted magenta line ---
            if len(paths) >= 2:
                reflected_cells = paths[1].cells
                if len(reflected_cells) >= 2:
                    points = [
                        self._grid_to_pixel_center(*cell)
                        for cell in reflected_cells
                    ]
                    # Draw every other segment to create a dashed/dotted effect.
                    # for i in range(0, len(points) - 1, 2):
                    #     pygame.draw.line(
                    #         self._screen,
                    #         COL_PATH_REFLECTED,
                    #         points[i],
                    #         points[i + 1],
                    #         2,
                    #     )
                    self._draw_dashed_polyline(points, COL_PATH_REFLECTED)

    # ------------------------------------------------------------------
    # Left sidebar
    # ------------------------------------------------------------------
    def _draw_left_sidebar(self) -> None:
        win_h = self._screen.get_height()
        sidebar_rect = pygame.Rect(0, 0, LEFT_SIDEBAR_W, win_h)
        pygame.draw.rect(self._screen, COL_SIDEBAR_BG, sidebar_rect)

        y = 12
        # Title
        title = self._font_lg.render("PALETTE", True, COL_TEXT)
        self._screen.blit(title, (16, y)); y += 28

        # ---- Listener palette item ----
        self._listener_palette_rect = pygame.Rect(16, y, LEFT_SIDEBAR_W - 32, 36)
        pygame.draw.rect(self._screen, COL_PALETTE_LISTENER, self._listener_palette_rect, border_radius=6)
        lt = self._font.render("🎧  Listener", True, (255, 255, 255))
        self._screen.blit(lt, (self._listener_palette_rect.x + 10, self._listener_palette_rect.y + 9))
        y += 46

        # ---- Source palette item ----
        self._source_palette_rect = pygame.Rect(16, y, LEFT_SIDEBAR_W - 32, 36)
        pygame.draw.rect(self._screen, COL_PALETTE_SOURCE, self._source_palette_rect, border_radius=6)
        st = self._font.render("🔊  Source", True, (255, 255, 255))
        self._screen.blit(st, (self._source_palette_rect.x + 10, self._source_palette_rect.y + 9))
        y += 56

        # ---- Wall tool toggle ----
        self._wall_btn_rect = pygame.Rect(16, y, LEFT_SIDEBAR_W - 32, 32)
        btn_col = COL_BUTTON_ACT if self._wall_tool != WALL_OFF else COL_BUTTON
        pygame.draw.rect(self._screen, btn_col, self._wall_btn_rect, border_radius=6)
        wt = self._font.render(WALL_LABELS[self._wall_tool], True, COL_TEXT)
        self._screen.blit(wt, (self._wall_btn_rect.x + 10, self._wall_btn_rect.y + 7))
        y += 42

        # ---- Material swatches (only visible in DRAW mode) ----
        self._mat_swatch_rects = []
        if self._wall_tool == WALL_DRAW:
            swatch_label = self._font_sm.render("Material:", True, COL_TEXT_DIM)
            self._screen.blit(swatch_label, (16, y))
            y += 16
            swatch_w = (LEFT_SIDEBAR_W - 32)
            swatch_h = 26
            for mat in _WALL_MATERIALS:
                rect = pygame.Rect(16, y, swatch_w, swatch_h)
                self._mat_swatch_rects.append(rect)
                # Highlight if this material is selected
                is_selected = (mat["gain"] == self._wall_material)
                border_col = (255, 255, 255) if is_selected else (80, 80, 95)
                pygame.draw.rect(self._screen, mat["color"], rect, border_radius=5)
                pygame.draw.rect(self._screen, border_col, rect, width=2, border_radius=5)
                lbl = self._font_sm.render(mat["label"], True, (20, 20, 20))
                self._screen.blit(lbl, (rect.x + 7, rect.y + 5))
                y += swatch_h + 4
            y += 6  # small gap below swatches
        else:
            y += 6  # same gap so layout below is stable

        # ---- Separator ----
        pygame.draw.line(self._screen, COL_GRID_LINE, (16, y), (LEFT_SIDEBAR_W - 16, y))
        y += 12

        # ---- Selected source panel ----
        if self._selected_source_id is not None:
            src = self._state.get_source(self._selected_source_id)
            if src is not None:
                header = self._font_lg.render(f"Source {self._selected_source_id}", True, COL_SOURCE_SEL)
                self._screen.blit(header, (16, y)); y += 24

                input_mode = src.get("input_mode", "file")
                is_mic = (input_mode == "mic")
                is_network = (input_mode == "network")

                if is_network:
                    # Network sources are auto-managed — show read-only label,
                    # hide all manual controls.
                    net_label_rect = pygame.Rect(16, y, LEFT_SIDEBAR_W - 32, 28)
                    pygame.draw.rect(self._screen, (40, 80, 60), net_label_rect, border_radius=6)
                    net_client = src.get("network_client")
                    if net_client:
                        nl_text = f"Net: {net_client[0]}:{net_client[1]}"
                    else:
                        nl_text = "Input: Network (auto)"
                    nl = self._font.render(nl_text, True, COL_TEXT)
                    self._screen.blit(nl, (net_label_rect.x + 10, net_label_rect.y + 5))
                    y += 38

                    # Playing indicator (always playing for network sources)
                    playing = src.get("playing", False)
                    status_col = (80, 220, 80) if playing else (180, 60, 60)
                    status_text = "▶ Playing" if playing else "■ Stopped"
                    st = self._font.render(status_text, True, status_col)
                    self._screen.blit(st, (16, y)); y += 24

                    # No other controls for network sources.
                    self._input_mode_btn_rect = None
                    self._load_audio_btn_rect = None
                    self._loop_toggle_rect = None
                    self._pause_btn_rect = None
                else:
                    # Input mode toggle button (file ↔ mic only)
                    self._input_mode_btn_rect = pygame.Rect(16, y, LEFT_SIDEBAR_W - 32, 28)
                    mode_col = COL_BUTTON_ACT if is_mic else COL_BUTTON
                    pygame.draw.rect(self._screen, mode_col, self._input_mode_btn_rect, border_radius=6)
                    mode_label = "Input: Mic" if is_mic else "Input: File"
                    ml = self._font.render(mode_label, True, COL_TEXT)
                    self._screen.blit(ml, (self._input_mode_btn_rect.x + 10, self._input_mode_btn_rect.y + 5))
                    y += 38

                if not is_network:
                    if not is_mic:
                        # File name
                        ap = src.get("audio_path")
                        fname = os.path.basename(ap) if ap else "No file"
                        ft = self._font_sm.render(fname, True, COL_TEXT_DIM)
                        self._screen.blit(ft, (16, y)); y += 20

                        # Load audio button
                        self._load_audio_btn_rect = pygame.Rect(16, y, LEFT_SIDEBAR_W - 32, 30)
                        pygame.draw.rect(self._screen, COL_BUTTON, self._load_audio_btn_rect, border_radius=6)
                        la = self._font.render("Load Audio File", True, COL_TEXT)
                        self._screen.blit(la, (self._load_audio_btn_rect.x + 10, self._load_audio_btn_rect.y + 6))
                        y += 40

                        # Loop toggle
                        loop_on = src.get("loop", False)
                        self._loop_toggle_rect = pygame.Rect(16, y, LEFT_SIDEBAR_W - 32, 28)
                        loop_col = COL_BUTTON_ACT if loop_on else COL_BUTTON
                        pygame.draw.rect(self._screen, loop_col, self._loop_toggle_rect, border_radius=6)
                        loop_label = "Loop: ON" if loop_on else "Loop: OFF"
                        ll = self._font.render(loop_label, True, COL_TEXT)
                        self._screen.blit(ll, (self._loop_toggle_rect.x + 10, self._loop_toggle_rect.y + 5))
                        y += 38
                    else:
                        ap = src.get("audio_path")
                        self._load_audio_btn_rect = None
                        self._loop_toggle_rect = None

                    # Pause/Play button — visible for mic-mode (always) and
                    # file-mode when audio_path is set.
                    show_pause = is_mic or src.get("audio_path")
                    if show_pause:
                        playing = src.get("playing", False)
                        self._pause_btn_rect = pygame.Rect(16, y, LEFT_SIDEBAR_W - 32, 28)
                        pause_col = COL_BUTTON_ACT if playing else COL_BUTTON
                        pygame.draw.rect(self._screen, pause_col, self._pause_btn_rect, border_radius=6)
                        pause_label = "⏸ Pause" if playing else "▶ Play"
                        pl = self._font.render(pause_label, True, COL_TEXT)
                        self._screen.blit(pl, (self._pause_btn_rect.x + 10, self._pause_btn_rect.y + 5))
                        y += 38
                    else:
                        self._pause_btn_rect = None

                    # Playing indicator
                    playing = src.get("playing", False)
                    status_col = (80, 220, 80) if playing else (180, 60, 60)
                    status_text = "▶ Playing" if playing else "■ Stopped"
                    st = self._font.render(status_text, True, status_col)
                    self._screen.blit(st, (16, y)); y += 24
            else:
                self._selected_source_id = None
                self._load_audio_btn_rect = None
                self._loop_toggle_rect = None
                self._pause_btn_rect = None
                self._input_mode_btn_rect = None
        else:
            self._load_audio_btn_rect = None
            self._loop_toggle_rect = None
            self._pause_btn_rect = None
            self._input_mode_btn_rect = None

    # ------------------------------------------------------------------
    # Right sidebar — waveform graphs
    # ------------------------------------------------------------------
    def _draw_right_sidebar(self) -> None:
        grid_draw_w = GRID_COLS * self._cell_size
        rx = LEFT_SIDEBAR_W + grid_draw_w
        win_w, win_h = self._screen.get_size()
        right_sidebar_w = max(RIGHT_SIDEBAR_W, win_w - rx)
        sidebar_rect = pygame.Rect(rx, 0, right_sidebar_w, win_h)
        pygame.draw.rect(self._screen, COL_SIDEBAR_BG, sidebar_rect)

        y = 12
        title = self._font_lg.render("WAVEFORMS", True, COL_TEXT)
        self._screen.blit(title, (rx + 12, y)); y += 28

        graph_w = right_sidebar_w - 24
        graph_h = 50

        # ---- FIX 3: FINAL OUTPUT (what you hear) graph at top ----
        mix_data = self._audio.get_last_mix()
        if mix_data is not None and mix_data.ndim == 2:
            mix_data = mix_data[:, 0]  # left channel

        mix_rect = pygame.Rect(rx + 12, y + 16, graph_w, graph_h)
        pygame.draw.rect(self._screen, COL_GRAPH_BG, mix_rect, border_radius=3)
        pygame.draw.rect(self._screen, (100, 180, 255), mix_rect, width=1, border_radius=3)
        mix_peak = self._draw_waveform(mix_rect, mix_data, (120, 220, 255), PROCESSED_DISPLAY_MULTIPLIER)

        mix_label_str = f"FINAL OUTPUT (L)  peak: {mix_peak:.4f}"
        mix_label = self._font_sm.render(mix_label_str, True, (150, 220, 255))
        self._screen.blit(mix_label, (rx + 12, y))
        y += graph_h + 24

        # Separator line
        pygame.draw.line(self._screen, COL_GRID_LINE, (rx + 12, y), (rx + right_sidebar_w - 12, y))
        y += 12

        # ---- Per-source graphs ----
        sources = self._state.get_sources()
        for sid in sorted(sources.keys()):
            if y + graph_h * 3 + 68 > win_h:
                break  # no room for more (raw + processed + spectrum)

            label = self._font_sm.render(f"Source {sid}", True, COL_SOURCE)
            self._screen.blit(label, (rx + 12, y)); y += 18

            # --- Raw waveform graph ---
            raw_rect = pygame.Rect(rx + 12, y + 14, graph_w, graph_h)
            pygame.draw.rect(self._screen, COL_GRAPH_BG, raw_rect, border_radius=3)
            raw_data = self._audio.get_last_raw(sid)
            raw_peak = self._draw_waveform(raw_rect, raw_data, COL_GRAPH_RAW, RAW_DISPLAY_MULTIPLIER)

            raw_label_str = f"raw  peak: {raw_peak:.4f}"
            raw_label = self._font_sm.render(raw_label_str, True, COL_TEXT_DIM)
            self._screen.blit(raw_label, (rx + 12, y))
            y += graph_h + 18

            # --- Processed waveform graph (left channel) ---
            proc_data = self._audio.get_last_processed(sid)
            if proc_data is not None and proc_data.ndim == 2:
                proc_data = proc_data[:, 0]  # left channel only

            proc_rect = pygame.Rect(rx + 12, y + 14, graph_w, graph_h)
            pygame.draw.rect(self._screen, COL_GRAPH_BG, proc_rect, border_radius=3)
            proc_peak = self._draw_waveform(proc_rect, proc_data, COL_GRAPH_PROC, PROCESSED_DISPLAY_MULTIPLIER)

            proc_label_str = f"processed (L)  peak: {proc_peak:.4f}"
            proc_label = self._font_sm.render(proc_label_str, True, COL_TEXT_DIM)
            self._screen.blit(proc_label, (rx + 12, y))
            y += graph_h + 18

            # --- Spectrum bar-graph panel ---
            spec_label = self._font_sm.render("spectrum", True, COL_TEXT_DIM)
            self._screen.blit(spec_label, (rx + 12, y))
            spec_rect = pygame.Rect(rx + 12, y + 14, graph_w, graph_h)
            pygame.draw.rect(self._screen, COL_GRAPH_BG, spec_rect, border_radius=3)

            bin_centers, magnitudes = compute_spectrum(
                proc_data, self._audio.SAMPLE_RATE, num_bins=32,
            )
            if len(magnitudes) > 0:
                num_bars = len(magnitudes)
                bar_w = max(1, spec_rect.width // num_bars)
                for bi in range(num_bars):
                    mag = float(min(1.0, max(0.0, magnitudes[bi])))
                    bar_h = int(mag * spec_rect.height)
                    bar_x = spec_rect.x + bi * bar_w
                    bar_y = spec_rect.y + spec_rect.height - bar_h
                    if bar_h > 0:
                        pygame.draw.rect(
                            self._screen,
                            COL_GRAPH_SPECTRUM,
                            (bar_x, bar_y, max(1, bar_w - 1), bar_h),
                        )
            y += graph_h + 22

    def _draw_waveform(
        self,
        rect: pygame.Rect,
        data: Optional[np.ndarray],
        color: tuple[int, int, int],
        multiplier: float = 1.0,
    ) -> float:
        """Render a simple line-plot waveform inside *rect* using viz_engine.

        Returns the actual pre-scaling peak amplitude computed by viz_engine.
        """
        if data is None or len(data) == 0:
            return 0.0

        w = rect.width
        h = rect.height

        # Downsample for target width using viz_engine
        ds = downsample_for_width(data, w)
        norm_data, peak = compute_display_samples(ds, multiplier)

        if len(norm_data) == 0:
            return peak

        n = len(norm_data)
        points: list[tuple[int, int]] = []
        for i in range(n):
            x = rect.x + (int(i * w / max(1, n - 1)) if n > 1 else 0)
            val = float(norm_data[i])
            y = rect.y + int((1.0 - val) * 0.5 * h)
            points.append((x, y))

        if len(points) > 1:
            pygame.draw.lines(self._screen, color, False, points, 1)

        return peak

    # ------------------------------------------------------------------
    # Draw the dragged item ghost
    # ------------------------------------------------------------------
    def _draw_drag_ghost(self) -> None:
        if self._dragging is None:
            return
        mx, my = self._drag_mouse_pos
        radius = max(2, self._cell_size // 2 - 2)
        if self._dragging == "listener":
            pygame.draw.circle(self._screen, (*COL_LISTENER, 160), (mx, my), radius)
        elif self._dragging == "source":
            pygame.draw.circle(self._screen, (*COL_SOURCE, 160), (mx, my), radius)

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------
    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                return

            elif event.type == pygame.VIDEORESIZE:
                self._screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                self._update_layout()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._on_mouse_down(event.pos)

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self._on_mouse_up(event.pos)

            elif event.type == pygame.MOUSEMOTION:
                self._on_mouse_motion(event.pos, event.buttons)

    def _on_mouse_down(self, pos: tuple[int, int]) -> None:
        mx, my = pos

        # --- Left sidebar interactions ---
        if mx < LEFT_SIDEBAR_W:
            # Palette: start drag
            if hasattr(self, '_listener_palette_rect') and self._listener_palette_rect.collidepoint(mx, my):
                self._dragging = "listener"
                self._drag_mouse_pos = pos
                return
            if hasattr(self, '_source_palette_rect') and self._source_palette_rect.collidepoint(mx, my):
                self._dragging = "source"
                self._drag_mouse_pos = pos
                return
            # Wall tool toggle
            if hasattr(self, '_wall_btn_rect') and self._wall_btn_rect.collidepoint(mx, my):
                self._wall_tool = (self._wall_tool + 1) % 3
                return
            # Material swatch selection (only active in DRAW mode)
            for i, rect in enumerate(self._mat_swatch_rects):
                if rect.collidepoint(mx, my):
                    self._wall_material = _WALL_MATERIALS[i]["gain"]
                    return
            # Load audio button (only clickable in file mode)
            if (self._selected_source_id is not None
                    and hasattr(self, '_load_audio_btn_rect')
                    and self._load_audio_btn_rect is not None
                    and self._load_audio_btn_rect.collidepoint(mx, my)):
                path = _open_file_dialog()
                if path:
                    self._audio.load_audio_for_source(self._selected_source_id, path)
                    self._state.set_source_audio(self._selected_source_id, path)
                return
            # Loop toggle (only clickable in file mode)
            if (self._selected_source_id is not None
                    and hasattr(self, '_loop_toggle_rect')
                    and self._loop_toggle_rect is not None
                    and self._loop_toggle_rect.collidepoint(mx, my)):
                src = self._state.get_source(self._selected_source_id)
                if src:
                    self._state.set_source_loop(self._selected_source_id, not src["loop"])
                return
            # Input mode toggle
            if (self._selected_source_id is not None
                    and hasattr(self, '_input_mode_btn_rect')
                    and self._input_mode_btn_rect is not None
                    and self._input_mode_btn_rect.collidepoint(mx, my)):
                src = self._state.get_source(self._selected_source_id)
                if src:
                    cur = src.get("input_mode", "file")
                    new_mode = "mic" if cur == "file" else "file"
                    self._state.set_source_input_mode(self._selected_source_id, new_mode)
                    # When switching to mic, auto-play so user hears output
                    # immediately.  When switching to file, keep current
                    # playing state.
                    if new_mode == "mic":
                        self._state.set_source_playing(self._selected_source_id, True)
                return
            # Pause / Play toggle
            if (self._selected_source_id is not None
                    and hasattr(self, '_pause_btn_rect')
                    and self._pause_btn_rect is not None
                    and self._pause_btn_rect.collidepoint(mx, my)):
                src = self._state.get_source(self._selected_source_id)
                if src:
                    # For mic mode, always allow toggling; for file mode,
                    # only when an audio file is loaded.
                    is_mic = src.get("input_mode", "file") == "mic"
                    if is_mic or src.get("audio_path"):
                        self._state.set_source_playing(
                            self._selected_source_id,
                            not src.get("playing", False),
                        )
                return
            return

        # --- Grid interactions ---
        cell = self._pixel_to_grid(mx, my)
        if cell is None:
            return

        # Wall painting
        if self._wall_tool == WALL_DRAW:
            if not self._cell_occupied_by_entity(cell):  # preserve occupancy check
                self._state.add_wall(cell, gain=self._wall_material)
            self._wall_painting = True
            return
        elif self._wall_tool == WALL_ERASE:
            self._state.remove_wall(cell)
            self._wall_painting = True
            return

        # Click on existing source → select or start drag
        sources = self._state.get_sources()
        for sid, info in sources.items():
            if info["pos"] == cell:
                self._selected_source_id = sid
                self._dragging = "move_source"
                self._dragging_source_id = sid
                self._drag_mouse_pos = pos
                return

        # Click on listener → start drag to move
        lp = self._state.get_listener_pos()
        if lp == cell:
            self._dragging = "listener"
            self._drag_mouse_pos = pos
            return

        # Click on empty cell → deselect
        self._selected_source_id = None

    def _cell_occupied_by_entity(self, cell: tuple[int, int]) -> bool:
        """True if the listener or any source currently sits at *cell*."""
        if self._state.get_listener_pos() == cell:
            return True
        return any(info["pos"] == cell for info in self._state.get_sources().values())

    def _on_mouse_up(self, pos: tuple[int, int]) -> None:
        mx, my = pos

        if self._wall_painting:
            self._wall_painting = False

        if self._dragging is not None:
            cell = self._pixel_to_grid(mx, my)
            if cell is not None and not self._state.has_wall(cell):
                if self._dragging == "listener":
                    self._state.set_listener_pos(cell)
                elif self._dragging == "source":
                    sid = self._state.add_source(cell)
                    self._selected_source_id = sid
                elif self._dragging == "move_source" and self._dragging_source_id is not None:
                    self._state.move_source(self._dragging_source_id, cell)
            # else: drop rejected silently — item snaps back (nothing to reset,
            # since state was never mutated during the drag itself)
            self._dragging = None
            self._dragging_source_id = None

    def _on_mouse_motion(self, pos: tuple[int, int], buttons: tuple[int, ...]) -> None:
        self._drag_mouse_pos = pos

        # Wall painting while dragging
        if self._wall_painting and buttons[0]:
            cell = self._pixel_to_grid(*pos)
            if cell is not None:
                if self._wall_tool == WALL_DRAW:
                    if not self._cell_occupied_by_entity(cell):  # preserve occupancy check
                        self._state.add_wall(cell, gain=self._wall_material)

                elif self._wall_tool == WALL_ERASE:
                    self._state.remove_wall(cell)

    # ------------------------------------------------------------------
    # Main render loop
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Network source auto-management
    # ------------------------------------------------------------------
    def _sync_network_sources(self) -> None:
        """Synchronise network sources with active UDP clients.

        Called once per frame from the main render loop.  Spawns new
        grid sources for newly-connected clients and removes sources
        for clients that have timed out.
        """
        active_clients = self._audio.network_input.get_active_clients()
        sources = self._state.get_sources()

        # Build map: network_client -> source_id for existing network sources.
        client_to_sid: dict[tuple[str, int], int] = {}
        for sid, info in sources.items():
            if info.get("input_mode") == "network":
                net_client = info.get("network_client")
                if net_client is not None:
                    client_to_sid[tuple(net_client)] = sid

        active_set = set(active_clients)

        # --- Spawn sources for new clients ---
        listener_pos = self._state.get_listener_pos()
        base_x = (listener_pos[0] + 2) if listener_pos else 5
        base_y = (listener_pos[1]) if listener_pos else 5
        network_source_count = len(client_to_sid)

        for client_key in active_clients:
            if client_key not in client_to_sid:
                # Stagger spawn positions based on total network sources
                offset = network_source_count * 2
                spawn_x = min(base_x + offset, GRID_COLS - 1)
                spawn_y = min(base_y + offset, GRID_ROWS - 1)

                sid = self._state.add_source((spawn_x, spawn_y))
                self._state.set_source_input_mode(sid, "network")
                self._state.set_source_network_client(sid, client_key)
                self._state.set_source_playing(sid, True)

                client_to_sid[client_key] = sid
                network_source_count += 1

        # --- Remove sources for disconnected clients ---
        for client_key, sid in client_to_sid.items():
            if client_key not in active_set:
                self._state.remove_source(sid)
                self._audio.remove_source(sid)
                if self._selected_source_id == sid:
                    self._selected_source_id = None

    # ------------------------------------------------------------------
    # Main render loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Run the Pygame render loop at ~60 FPS until the window is closed."""
        while self._running:
            self._handle_events()
            if not self._running:
                break

            # Sync network sources before drawing.
            self._sync_network_sources()

            self._screen.fill(COL_BG)
            self._draw_grid()
            self._draw_walls()
            self._draw_source_paths()
            self._draw_listener()
            self._draw_sources()
            self._draw_left_sidebar()
            self._draw_right_sidebar()
            self._draw_drag_ghost()

            pygame.display.flip()
            self._clock.tick(60)

        pygame.quit()