"""
render_engine.py — Pygame-based UI for the positional-audio sandbox.

Runs entirely on the main thread.  Draws the grid, handles mouse
interaction (drag-drop from palette, wall painting, source selection),
and renders waveform graphs in the right sidebar.

Layout:
  ┌──────────┬────────────────────────┬──────────────┐
  │  LEFT    │       GRID AREA        │    RIGHT     │
  │ SIDEBAR  │ GRID_COLS * CELL_SIZE  │   SIDEBAR    │
  │  250 px  │                        │   300 px     │
  └──────────┴────────────────────────┴──────────────┘
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import pygame

from shared_state import SharedState, GRID_COLS, GRID_ROWS, CELL_SIZE_PX
from audio_engine import AudioEngine

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
COL_PALETTE_LISTENER = (55, 100, 200)
COL_PALETTE_SOURCE   = (200, 120, 30)


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
        self._screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("consolas", 14)
        self._font_sm = pygame.font.SysFont("consolas", 12)
        self._font_lg = pygame.font.SysFont("consolas", 16, bold=True)

        # ---- UI state ----
        self._running: bool = True
        self._dragging: Optional[str] = None  # "listener", "source", or None (palette drag)
        self._dragging_source_id: Optional[int] = None  # when dragging an existing source
        self._drag_mouse_pos: tuple[int, int] = (0, 0)
        self._wall_tool: int = WALL_OFF
        self._wall_painting: bool = False  # True while mouse is held for wall drawing
        self._selected_source_id: Optional[int] = None

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    def _grid_origin(self) -> tuple[int, int]:
        """Top-left pixel of the grid area."""
        return (LEFT_SIDEBAR_W, 0)

    def _pixel_to_grid(self, px: int, py: int) -> Optional[tuple[int, int]]:
        """Convert pixel coords to grid cell, or None if outside the grid."""
        gx_origin, gy_origin = self._grid_origin()
        gx = (px - gx_origin) // CELL_SIZE_PX
        gy = (py - gy_origin) // CELL_SIZE_PX
        if 0 <= gx < GRID_COLS and 0 <= gy < GRID_ROWS:
            return (gx, gy)
        return None

    def _grid_to_pixel_center(self, gx: int, gy: int) -> tuple[int, int]:
        """Return the pixel center of grid cell (gx, gy)."""
        ox, oy = self._grid_origin()
        return (ox + gx * CELL_SIZE_PX + CELL_SIZE_PX // 2,
                oy + gy * CELL_SIZE_PX + CELL_SIZE_PX // 2)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def _draw_grid(self) -> None:
        ox, oy = self._grid_origin()
        # Background
        pygame.draw.rect(self._screen, COL_BG, (ox, oy, GRID_W, GRID_H))
        # Vertical lines
        for c in range(GRID_COLS + 1):
            x = ox + c * CELL_SIZE_PX
            pygame.draw.line(self._screen, COL_GRID_LINE, (x, oy), (x, oy + GRID_H))
        # Horizontal lines
        for r in range(GRID_ROWS + 1):
            y = oy + r * CELL_SIZE_PX
            pygame.draw.line(self._screen, COL_GRID_LINE, (ox, y), (ox + GRID_W, y))

    def _draw_walls(self) -> None:
        ox, oy = self._grid_origin()
        walls = self._state.get_walls()
        for (wx, wy) in walls:
            rect = pygame.Rect(ox + wx * CELL_SIZE_PX + 1, oy + wy * CELL_SIZE_PX + 1,
                               CELL_SIZE_PX - 1, CELL_SIZE_PX - 1)
            pygame.draw.rect(self._screen, COL_WALL, rect)

    def _draw_listener(self) -> None:
        lp = self._state.get_listener_pos()
        if lp is None:
            return
        cx, cy = self._grid_to_pixel_center(*lp)
        pygame.draw.circle(self._screen, COL_LISTENER, (cx, cy), CELL_SIZE_PX // 2 - 2)
        # Forward-axis indicator (small line pointing +x)
        pygame.draw.line(self._screen, (200, 220, 255), (cx, cy), (cx + CELL_SIZE_PX // 2 + 2, cy), 2)
        label = self._font_sm.render("L", True, (255, 255, 255))
        self._screen.blit(label, (cx - label.get_width() // 2, cy - label.get_height() // 2))

    def _draw_sources(self) -> None:
        sources = self._state.get_sources()
        for sid, info in sources.items():
            px, py = self._grid_to_pixel_center(*info["pos"])
            col = COL_SOURCE_SEL if sid == self._selected_source_id else COL_SOURCE
            pygame.draw.circle(self._screen, col, (px, py), CELL_SIZE_PX // 2 - 2)
            label = self._font_sm.render(f"S{sid}", True, (0, 0, 0))
            self._screen.blit(label, (px - label.get_width() // 2, py - label.get_height() // 2))

    # ------------------------------------------------------------------
    # Left sidebar
    # ------------------------------------------------------------------
    def _draw_left_sidebar(self) -> None:
        sidebar_rect = pygame.Rect(0, 0, LEFT_SIDEBAR_W, WINDOW_H)
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
        y += 48

        # ---- Separator ----
        pygame.draw.line(self._screen, COL_GRID_LINE, (16, y), (LEFT_SIDEBAR_W - 16, y))
        y += 12

        # ---- Selected source panel ----
        if self._selected_source_id is not None:
            src = self._state.get_source(self._selected_source_id)
            if src is not None:
                header = self._font_lg.render(f"Source {self._selected_source_id}", True, COL_SOURCE_SEL)
                self._screen.blit(header, (16, y)); y += 24

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

                # Playing indicator
                playing = src.get("playing", False)
                status_col = (80, 220, 80) if playing else (180, 60, 60)
                status_text = "▶ Playing" if playing else "■ Stopped"
                st = self._font.render(status_text, True, status_col)
                self._screen.blit(st, (16, y)); y += 24
            else:
                self._selected_source_id = None
        else:
            self._load_audio_btn_rect = None
            self._loop_toggle_rect = None

    # ------------------------------------------------------------------
    # Right sidebar — waveform graphs
    # ------------------------------------------------------------------
    def _draw_right_sidebar(self) -> None:
        rx = LEFT_SIDEBAR_W + GRID_W
        sidebar_rect = pygame.Rect(rx, 0, RIGHT_SIDEBAR_W, WINDOW_H)
        pygame.draw.rect(self._screen, COL_SIDEBAR_BG, sidebar_rect)

        y = 12
        title = self._font_lg.render("WAVEFORMS", True, COL_TEXT)
        self._screen.blit(title, (rx + 12, y)); y += 28

        sources = self._state.get_sources()
        graph_w = RIGHT_SIDEBAR_W - 24
        graph_h = 50

        for sid in sorted(sources.keys()):
            if y + graph_h * 2 + 50 > WINDOW_H:
                break  # no room for more

            label = self._font_sm.render(f"Source {sid}", True, COL_SOURCE)
            self._screen.blit(label, (rx + 12, y)); y += 18

            # --- Raw waveform graph ---
            raw_label = self._font_sm.render("raw", True, COL_TEXT_DIM)
            self._screen.blit(raw_label, (rx + 12, y))
            raw_rect = pygame.Rect(rx + 12, y + 14, graph_w, graph_h)
            pygame.draw.rect(self._screen, COL_GRAPH_BG, raw_rect, border_radius=3)
            raw_data = self._audio.get_last_raw(sid)
            self._draw_waveform(raw_rect, raw_data, COL_GRAPH_RAW)
            y += graph_h + 18

            # --- Processed waveform graph (left channel) ---
            proc_label = self._font_sm.render("processed (L)", True, COL_TEXT_DIM)
            self._screen.blit(proc_label, (rx + 12, y))
            proc_rect = pygame.Rect(rx + 12, y + 14, graph_w, graph_h)
            pygame.draw.rect(self._screen, COL_GRAPH_BG, proc_rect, border_radius=3)
            proc_data = self._audio.get_last_processed(sid)
            if proc_data is not None and proc_data.ndim == 2:
                proc_data = proc_data[:, 0]  # left channel only
            self._draw_waveform(proc_rect, proc_data, COL_GRAPH_PROC)
            y += graph_h + 22

    def _draw_waveform(
        self,
        rect: pygame.Rect,
        data: Optional[np.ndarray],
        color: tuple[int, int, int],
    ) -> None:
        """Render a simple line-plot waveform inside *rect*."""
        if data is None or len(data) == 0:
            return

        w = rect.width
        h = rect.height
        n = len(data)
        step = max(1, n // w)
        # Downsample to fit width
        points: list[tuple[int, int]] = []
        for i in range(0, min(n, w * step), step):
            x = rect.x + int(i / step)
            # Map sample value [-1, 1] to pixel y
            val = float(np.clip(data[i], -1.0, 1.0))
            y = rect.y + int((1.0 - val) * 0.5 * h)
            points.append((x, y))

        if len(points) > 1:
            pygame.draw.lines(self._screen, color, False, points, 1)

    # ------------------------------------------------------------------
    # Draw the dragged item ghost
    # ------------------------------------------------------------------
    def _draw_drag_ghost(self) -> None:
        if self._dragging is None:
            return
        mx, my = self._drag_mouse_pos
        if self._dragging == "listener":
            pygame.draw.circle(self._screen, (*COL_LISTENER, 160), (mx, my), CELL_SIZE_PX // 2 - 2)
        elif self._dragging == "source":
            pygame.draw.circle(self._screen, (*COL_SOURCE, 160), (mx, my), CELL_SIZE_PX // 2 - 2)

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------
    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                return

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
            # Load audio button
            if (self._selected_source_id is not None
                    and hasattr(self, '_load_audio_btn_rect')
                    and self._load_audio_btn_rect is not None
                    and self._load_audio_btn_rect.collidepoint(mx, my)):
                path = _open_file_dialog()
                if path:
                    self._audio.load_audio_for_source(self._selected_source_id, path)
                    self._state.set_source_audio(self._selected_source_id, path)
                return
            # Loop toggle
            if (self._selected_source_id is not None
                    and hasattr(self, '_loop_toggle_rect')
                    and self._loop_toggle_rect is not None
                    and self._loop_toggle_rect.collidepoint(mx, my)):
                src = self._state.get_source(self._selected_source_id)
                if src:
                    self._state.set_source_loop(self._selected_source_id, not src["loop"])
                return
            return

        # --- Grid interactions ---
        cell = self._pixel_to_grid(mx, my)
        if cell is None:
            return

        # Wall painting
        if self._wall_tool == WALL_DRAW:
            self._state.add_wall(cell)
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

    def _on_mouse_up(self, pos: tuple[int, int]) -> None:
        mx, my = pos

        if self._wall_painting:
            self._wall_painting = False

        if self._dragging is not None:
            cell = self._pixel_to_grid(mx, my)
            if cell is not None:
                if self._dragging == "listener":
                    self._state.set_listener_pos(cell)
                elif self._dragging == "source":
                    sid = self._state.add_source(cell)
                    self._selected_source_id = sid
                elif self._dragging == "move_source" and self._dragging_source_id is not None:
                    self._state.move_source(self._dragging_source_id, cell)

            self._dragging = None
            self._dragging_source_id = None

    def _on_mouse_motion(self, pos: tuple[int, int], buttons: tuple[int, ...]) -> None:
        self._drag_mouse_pos = pos

        # Wall painting while dragging
        if self._wall_painting and buttons[0]:
            cell = self._pixel_to_grid(*pos)
            if cell is not None:
                if self._wall_tool == WALL_DRAW:
                    self._state.add_wall(cell)
                elif self._wall_tool == WALL_ERASE:
                    self._state.remove_wall(cell)

    # ------------------------------------------------------------------
    # Main render loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Run the Pygame render loop at ~60 FPS until the window is closed."""
        while self._running:
            self._handle_events()
            if not self._running:
                break

            self._screen.fill(COL_BG)
            self._draw_grid()
            self._draw_walls()
            self._draw_listener()
            self._draw_sources()
            self._draw_left_sidebar()
            self._draw_right_sidebar()
            self._draw_drag_ghost()

            pygame.display.flip()
            self._clock.tick(60)

        pygame.quit()
