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


import math
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
from pathfinding import (
    find_k_paths,
    find_reflector_candidates,
    find_transmission_path,
)
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
# Design tokens — spacing & corner radius
# ---------------------------------------------------------------------------
SPACE_XS: int = 4
SPACE_SM: int = 8
SPACE_MD: int = 16
SPACE_LG: int = 24
CORNER_RADIUS: int = 8

# ---------------------------------------------------------------------------
# Font family fallback strings (pygame.font.SysFont accepts comma-joined)
# ---------------------------------------------------------------------------
_FONT_UI = "Segoe UI,Helvetica Neue,Arial"
_FONT_MONO = "Cascadia Mono,Consolas"

# ---------------------------------------------------------------------------
# Colour palette — layered background system + accent + text hierarchy
# ---------------------------------------------------------------------------
# Background layers (each visibly distinct)
COL_BG_BASE       = (18, 18, 24)     # darkest — window/grid background
COL_BG_SURFACE    = (28, 29, 38)     # sidebar background
COL_BG_CARD       = (42, 44, 58)     # button/card/panel fill
COL_BG_CARD_HOVER = (58, 62, 82)     # hover/active state for cards

# Accent colour (blue family, consistent for all "active/on/selected" states)
COL_ACCENT        = (80, 110, 200)
COL_ACCENT_BRIGHT = (100, 140, 240)

# Text hierarchy
COL_TEXT_PRIMARY   = (235, 235, 245)  # titles, important labels
COL_TEXT_SECONDARY = (190, 192, 205)  # normal labels
COL_TEXT_DIM       = (120, 124, 140)  # captions, file names

# Subtle border for cards/panels (1 px, slightly lighter than card fill)
COL_BORDER         = (60, 63, 78)

# Grid line colour
COL_GRID_LINE      = (40, 42, 52)

# Fallback wall colour (unrecognised gain value)
COL_WALL           = (70, 70, 80)

# Semantic colours (preserved from original — verified readable on new bg)
COL_LISTENER       = (70, 140, 255)
COL_SOURCE         = (255, 160, 50)
COL_SOURCE_SEL     = (255, 220, 80)
COL_GRAPH_BG       = (22, 22, 30)
COL_GRAPH_RAW      = (100, 200, 100)
COL_GRAPH_PROC     = (100, 160, 255)
COL_GRAPH_SPECTRUM = (255, 180, 80)
COL_PATH_PRIMARY      = (50, 200, 180)
COL_PATH_TRANSMISSION = (255, 140, 0)    # Family B through-wall transmission line
COL_PATH_ECHO         = (200, 100, 220)  # Family C discrete echo candidate rays & markers

# Palette item accent colours
COL_PALETTE_LISTENER = (50, 90, 190)
COL_PALETTE_SOURCE   = (190, 110, 25)

# Status colours
COL_STATUS_PLAYING = (80, 220, 80)
COL_STATUS_STOPPED = (180, 60, 60)

# Network info panel
COL_NET_PANEL      = (30, 65, 50)

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
# Vector icon drawing helpers (replace emoji glyphs with pygame.draw)
# ---------------------------------------------------------------------------
def _draw_icon_headphones(surface: pygame.Surface, cx: int, cy: int, size: int, color: tuple) -> None:
    """Draw a simple headphone icon centred at (cx, cy)."""
    r = size // 2
    # Headband arc (top semicircle)
    band_rect = pygame.Rect(cx - r, cy - r, 2 * r, 2 * r)
    pygame.draw.arc(surface, color, band_rect, math.radians(20), math.radians(160), 2)
    # Left earpiece
    ear_w, ear_h = max(2, r // 2), max(3, r)
    pygame.draw.rect(surface, color, (cx - r - 1, cy - ear_h // 3, ear_w, ear_h), border_radius=2)
    # Right earpiece
    pygame.draw.rect(surface, color, (cx + r - ear_w + 1, cy - ear_h // 3, ear_w, ear_h), border_radius=2)


def _draw_icon_speaker(surface: pygame.Surface, cx: int, cy: int, size: int, color: tuple) -> None:
    """Draw a simple speaker cone icon centred at (cx, cy)."""
    r = size // 2
    # Speaker body (small rect on the left)
    body_w = max(2, r // 2)
    body_h = max(3, r)
    bx = cx - r
    by = cy - body_h // 2
    pygame.draw.rect(surface, color, (bx, by, body_w, body_h))
    # Cone (trapezoid/triangle pointing right)
    cone_pts = [
        (bx + body_w, cy - body_h // 2),
        (cx + r, cy - r),
        (cx + r, cy + r),
        (bx + body_w, cy + body_h // 2),
    ]
    pygame.draw.polygon(surface, color, cone_pts)


def _draw_icon_play(surface: pygame.Surface, cx: int, cy: int, size: int, color: tuple) -> None:
    """Draw a solid right-pointing play triangle centred at (cx, cy)."""
    r = size // 2
    pts = [
        (cx - r // 2, cy - r),
        (cx + r, cy),
        (cx - r // 2, cy + r),
    ]
    pygame.draw.polygon(surface, color, pts)


def _draw_icon_pause(surface: pygame.Surface, cx: int, cy: int, size: int, color: tuple) -> None:
    """Draw two vertical pause bars centred at (cx, cy)."""
    r = size // 2
    bar_w = max(2, r // 2)
    gap = max(2, r // 2)
    bar_h = size
    # Left bar
    pygame.draw.rect(surface, color, (cx - gap - bar_w, cy - r, bar_w, bar_h))
    # Right bar
    pygame.draw.rect(surface, color, (cx + gap, cy - r, bar_w, bar_h))


def _draw_icon_stop(surface: pygame.Surface, cx: int, cy: int, size: int, color: tuple) -> None:
    """Draw a filled square stop icon centred at (cx, cy)."""
    r = size // 2
    pygame.draw.rect(surface, color, (cx - r, cy - r, size, size))


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
        self._screen = pygame.display.set_mode(
            (window_w, window_h), pygame.RESIZABLE | pygame.DOUBLEBUF
        )

        self._clock = pygame.time.Clock()

        # ---- Typography (Part 2) ----
        self._font_title = pygame.font.SysFont(_FONT_UI, 20, bold=True)
        self._font_label = pygame.font.SysFont(_FONT_UI, 14)
        self._font_small = pygame.font.SysFont(_FONT_UI, 12)
        self._font_mono  = pygame.font.SysFont(_FONT_MONO, 12)

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
        pygame.draw.rect(self._screen, COL_BG_BASE, (ox, oy, grid_draw_w, grid_draw_h))

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
            COL_ACCENT_BRIGHT,
            (cx, cy),
            (cx + self._cell_size // 2 + 2, cy),
            2,
        )
        label = self._font_small.render("L", True, COL_TEXT_PRIMARY)
        self._screen.blit(label, (cx - label.get_width() // 2, cy - label.get_height() // 2))

    def _draw_sources(self) -> None:
        sources = self._state.get_sources()
        for sid, info in sources.items():
            px, py = self._grid_to_pixel_center(*info["pos"])
            col = COL_SOURCE_SEL if sid == self._selected_source_id else COL_SOURCE
            radius = max(2, self._cell_size // 2 - 2)
            pygame.draw.circle(self._screen, col, (px, py), radius)
            label = self._font_small.render(f"S{sid}", True, (0, 0, 0))
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
        """Draw acoustic arrival paths for all three families on the grid.

        For each source, independently calls pathfinding routines on the render
        thread (safe — pathfinding is pure and stateless):
          - Family A: find_k_paths (k=1) for the primary routed path (solid teal line)
          - Family B: find_transmission_path for through-wall transmission (dashed orange line)
          - Family C: find_reflector_candidates for discrete echoes (thin magenta rays + circle markers)

        This duplicates computation done on the audio thread inside process_block,
        which is intentional: audio and render threads communicate ONLY via
        shared_state.py snapshots/getters to keep thread safety guarantees intact.
        No audio-thread filter_state is read here.
        """
        listener_pos = self._state.get_listener_pos()
        if listener_pos is None:
            return

        sources = self._state.get_sources()
        walls = self._state.get_walls()
        wall_gains = self._state.get_wall_gains()
        listener_px = self._grid_to_pixel_center(*listener_pos)

        for sid in sorted(sources.keys()):
            source_pos = sources[sid]["pos"]
            source_px = self._grid_to_pixel_center(*source_pos)

            # --- Family A: Primary routed path (solid teal line) ---
            paths = find_k_paths(
                listener_pos, source_pos, walls, GRID_COLS, GRID_ROWS, k=1,
            )
            if paths:
                primary_cells = paths[0].cells
                if len(primary_cells) >= 2:
                    points = [
                        self._grid_to_pixel_center(*cell)
                        for cell in primary_cells
                    ]
                    pygame.draw.lines(
                        self._screen, COL_PATH_PRIMARY, False, points, 2,
                    )

            # --- Family B: Through-wall transmission (dashed orange line) ---
            transmission_path = find_transmission_path(
                source_pos, listener_pos, walls, wall_gains, GRID_COLS, GRID_ROWS,
            )
            if transmission_path is not None:
                self._draw_dashed_polyline(
                    [source_px, listener_px], COL_PATH_TRANSMISSION,
                )

            # --- Family C: Discrete multi-wall echoes (thin rays + bounce markers) ---
            reflector_candidates = find_reflector_candidates(
                source_pos, listener_pos, walls, wall_gains, GRID_COLS, GRID_ROWS,
            )
            for cand in reflector_candidates:
                wall_px = self._grid_to_pixel_center(*cand.wall_cell)
                pygame.draw.line(
                    self._screen, COL_PATH_ECHO, source_px, wall_px, 1,
                )
                pygame.draw.circle(
                    self._screen, COL_PATH_ECHO, wall_px, 4,
                )

    # ------------------------------------------------------------------
    # Sidebar drawing helper: card with border
    # ------------------------------------------------------------------
    def _draw_card(self, rect: pygame.Rect, fill: tuple, border: tuple = COL_BORDER) -> None:
        """Draw a rounded-rect card with fill and 1px border."""
        pygame.draw.rect(self._screen, fill, rect, border_radius=CORNER_RADIUS)
        pygame.draw.rect(self._screen, border, rect, width=1, border_radius=CORNER_RADIUS)

    # ------------------------------------------------------------------
    # Left sidebar
    # ------------------------------------------------------------------
    def _draw_left_sidebar(self) -> None:
        win_h = self._screen.get_height()
        sidebar_rect = pygame.Rect(0, 0, LEFT_SIDEBAR_W, win_h)
        pygame.draw.rect(self._screen, COL_BG_SURFACE, sidebar_rect)

        pad = SPACE_MD  # horizontal padding from sidebar edge
        content_w = LEFT_SIDEBAR_W - 2 * pad
        y = SPACE_MD

        # Title
        title = self._font_title.render("PALETTE", True, COL_TEXT_PRIMARY)
        self._screen.blit(title, (pad, y))
        y += title.get_height() + SPACE_SM

        # ---- Listener palette item ----
        item_h = 36
        self._listener_palette_rect = pygame.Rect(pad, y, content_w, item_h)
        self._draw_card(self._listener_palette_rect, COL_PALETTE_LISTENER)
        # Icon
        icon_cx = self._listener_palette_rect.x + SPACE_MD + 6
        icon_cy = self._listener_palette_rect.y + item_h // 2
        _draw_icon_headphones(self._screen, icon_cx, icon_cy, 14, COL_TEXT_PRIMARY)
        # Label
        lt = self._font_label.render("Listener", True, COL_TEXT_PRIMARY)
        self._screen.blit(lt, (icon_cx + 14, self._listener_palette_rect.y + (item_h - lt.get_height()) // 2))
        y += item_h + SPACE_SM

        # ---- Source palette item ----
        self._source_palette_rect = pygame.Rect(pad, y, content_w, item_h)
        self._draw_card(self._source_palette_rect, COL_PALETTE_SOURCE)
        # Icon
        icon_cx = self._source_palette_rect.x + SPACE_MD + 6
        icon_cy = self._source_palette_rect.y + item_h // 2
        _draw_icon_speaker(self._screen, icon_cx, icon_cy, 14, COL_TEXT_PRIMARY)
        # Label
        st = self._font_label.render("Source", True, COL_TEXT_PRIMARY)
        self._screen.blit(st, (icon_cx + 14, self._source_palette_rect.y + (item_h - st.get_height()) // 2))
        y += item_h + SPACE_MD

        # ---- Wall tool toggle ----
        btn_h = 32
        self._wall_btn_rect = pygame.Rect(pad, y, content_w, btn_h)
        btn_fill = COL_ACCENT if self._wall_tool != WALL_OFF else COL_BG_CARD
        self._draw_card(self._wall_btn_rect, btn_fill)
        wt = self._font_label.render(WALL_LABELS[self._wall_tool], True, COL_TEXT_PRIMARY)
        self._screen.blit(wt, (self._wall_btn_rect.x + SPACE_SM + 2, self._wall_btn_rect.y + (btn_h - wt.get_height()) // 2))
        y += btn_h + SPACE_SM

        # ---- Material swatches (only visible in DRAW mode) ----
        self._mat_swatch_rects = []
        if self._wall_tool == WALL_DRAW:
            swatch_label = self._font_small.render("Material:", True, COL_TEXT_DIM)
            self._screen.blit(swatch_label, (pad, y))
            y += swatch_label.get_height() + SPACE_XS
            swatch_h = 26
            for mat in _WALL_MATERIALS:
                rect = pygame.Rect(pad, y, content_w, swatch_h)
                self._mat_swatch_rects.append(rect)
                # Highlight if this material is selected
                is_selected = (mat["gain"] == self._wall_material)
                border_col = COL_ACCENT_BRIGHT if is_selected else COL_BORDER
                pygame.draw.rect(self._screen, mat["color"], rect, border_radius=CORNER_RADIUS)
                pygame.draw.rect(self._screen, border_col, rect, width=2, border_radius=CORNER_RADIUS)
                lbl = self._font_small.render(mat["label"], True, (20, 20, 20))
                self._screen.blit(lbl, (rect.x + SPACE_SM, rect.y + (swatch_h - lbl.get_height()) // 2))
                y += swatch_h + SPACE_XS
            y += SPACE_SM
        else:
            y += SPACE_SM

        # ---- Separator ----
        pygame.draw.line(self._screen, COL_GRID_LINE, (pad, y), (LEFT_SIDEBAR_W - pad, y))
        y += SPACE_MD

        # ---- Selected source panel ----
        if self._selected_source_id is not None:
            src = self._state.get_source(self._selected_source_id)
            if src is not None:
                header = self._font_title.render(f"Source {self._selected_source_id}", True, COL_SOURCE_SEL)
                self._screen.blit(header, (pad, y))
                y += header.get_height() + SPACE_SM

                input_mode = src.get("input_mode", "file")
                is_mic = (input_mode == "mic")
                is_network = (input_mode == "network")

                if is_network:
                    # Network sources are auto-managed — show read-only label,
                    # hide all manual controls.
                    net_label_rect = pygame.Rect(pad, y, content_w, 28)
                    self._draw_card(net_label_rect, COL_NET_PANEL)
                    net_client = src.get("network_client")
                    if net_client:
                        nl_text = f"Net: {net_client[0]}:{net_client[1]}"
                    else:
                        nl_text = "Input: Network (auto)"
                    nl = self._font_mono.render(nl_text, True, COL_TEXT_SECONDARY)
                    self._screen.blit(nl, (net_label_rect.x + SPACE_SM, net_label_rect.y + (28 - nl.get_height()) // 2))
                    y += 28 + SPACE_SM

                    # Playing indicator (always playing for network sources)
                    playing = src.get("playing", False)
                    status_col = COL_STATUS_PLAYING if playing else COL_STATUS_STOPPED
                    icon_y = y + 3
                    if playing:
                        _draw_icon_play(self._screen, pad + 6, icon_y + 5, 8, status_col)
                    else:
                        _draw_icon_stop(self._screen, pad + 6, icon_y + 5, 8, status_col)
                    status_text = "Playing" if playing else "Stopped"
                    st_surf = self._font_label.render(status_text, True, status_col)
                    self._screen.blit(st_surf, (pad + 18, y))
                    y += st_surf.get_height() + SPACE_SM

                    # No other controls for network sources.
                    self._input_mode_btn_rect = None
                    self._load_audio_btn_rect = None
                    self._loop_toggle_rect = None
                    self._pause_btn_rect = None
                else:
                    # Input mode toggle button (file ↔ mic only)
                    mode_btn_h = 28
                    self._input_mode_btn_rect = pygame.Rect(pad, y, content_w, mode_btn_h)
                    mode_fill = COL_ACCENT if is_mic else COL_BG_CARD
                    self._draw_card(self._input_mode_btn_rect, mode_fill)
                    mode_label = "Input: Mic" if is_mic else "Input: File"
                    ml = self._font_label.render(mode_label, True, COL_TEXT_PRIMARY)
                    self._screen.blit(ml, (self._input_mode_btn_rect.x + SPACE_SM + 2, self._input_mode_btn_rect.y + (mode_btn_h - ml.get_height()) // 2))
                    y += mode_btn_h + SPACE_SM

                if not is_network:
                    if not is_mic:
                        # File name
                        ap = src.get("audio_path")
                        fname = os.path.basename(ap) if ap else "No file"
                        ft = self._font_small.render(fname, True, COL_TEXT_DIM)
                        self._screen.blit(ft, (pad, y))
                        y += ft.get_height() + SPACE_XS

                        # Load audio button
                        load_btn_h = 30
                        self._load_audio_btn_rect = pygame.Rect(pad, y, content_w, load_btn_h)
                        self._draw_card(self._load_audio_btn_rect, COL_BG_CARD)
                        la = self._font_label.render("Load Audio File", True, COL_TEXT_SECONDARY)
                        self._screen.blit(la, (self._load_audio_btn_rect.x + SPACE_SM + 2, self._load_audio_btn_rect.y + (load_btn_h - la.get_height()) // 2))
                        y += load_btn_h + SPACE_SM

                        # Loop toggle
                        loop_on = src.get("loop", False)
                        loop_btn_h = 28
                        self._loop_toggle_rect = pygame.Rect(pad, y, content_w, loop_btn_h)
                        loop_fill = COL_ACCENT if loop_on else COL_BG_CARD
                        self._draw_card(self._loop_toggle_rect, loop_fill)
                        loop_label = "Loop: ON" if loop_on else "Loop: OFF"
                        ll = self._font_label.render(loop_label, True, COL_TEXT_PRIMARY)
                        self._screen.blit(ll, (self._loop_toggle_rect.x + SPACE_SM + 2, self._loop_toggle_rect.y + (loop_btn_h - ll.get_height()) // 2))
                        y += loop_btn_h + SPACE_SM
                    else:
                        ap = src.get("audio_path")
                        self._load_audio_btn_rect = None
                        self._loop_toggle_rect = None

                    # Pause/Play button — visible for mic-mode (always) and
                    # file-mode when audio_path is set.
                    show_pause = is_mic or src.get("audio_path")
                    if show_pause:
                        playing = src.get("playing", False)
                        pause_btn_h = 28
                        self._pause_btn_rect = pygame.Rect(pad, y, content_w, pause_btn_h)
                        pause_fill = COL_ACCENT if playing else COL_BG_CARD
                        self._draw_card(self._pause_btn_rect, pause_fill)
                        # Icon
                        icon_cx = self._pause_btn_rect.x + SPACE_SM + 6
                        icon_cy = self._pause_btn_rect.y + pause_btn_h // 2
                        if playing:
                            _draw_icon_pause(self._screen, icon_cx, icon_cy, 10, COL_TEXT_PRIMARY)
                        else:
                            _draw_icon_play(self._screen, icon_cx, icon_cy, 10, COL_TEXT_PRIMARY)
                        pause_label = "Pause" if playing else "Play"
                        pl = self._font_label.render(pause_label, True, COL_TEXT_PRIMARY)
                        self._screen.blit(pl, (icon_cx + 14, self._pause_btn_rect.y + (pause_btn_h - pl.get_height()) // 2))
                        y += pause_btn_h + SPACE_SM
                    else:
                        self._pause_btn_rect = None

                    # Playing indicator
                    playing = src.get("playing", False)
                    status_col = COL_STATUS_PLAYING if playing else COL_STATUS_STOPPED
                    icon_y = y + 3
                    if playing:
                        _draw_icon_play(self._screen, pad + 6, icon_y + 5, 8, status_col)
                    else:
                        _draw_icon_stop(self._screen, pad + 6, icon_y + 5, 8, status_col)
                    status_text = "Playing" if playing else "Stopped"
                    st_surf = self._font_label.render(status_text, True, status_col)
                    self._screen.blit(st_surf, (pad + 18, y))
                    y += st_surf.get_height() + SPACE_SM
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
        pygame.draw.rect(self._screen, COL_BG_SURFACE, sidebar_rect)

        pad = SPACE_MD - 4  # inner padding
        y = SPACE_MD
        title = self._font_title.render("WAVEFORMS", True, COL_TEXT_PRIMARY)
        self._screen.blit(title, (rx + pad, y))
        y += title.get_height() + SPACE_SM

        graph_w = right_sidebar_w - 2 * pad
        graph_h = 50

        # ---- FINAL OUTPUT (what you hear) graph at top ----
        mix_data = self._audio.get_last_mix()
        if mix_data is not None and mix_data.ndim == 2:
            mix_data = mix_data[:, 0]  # left channel

        mix_label_str = "FINAL OUTPUT (L)"
        mix_label = self._font_small.render(mix_label_str, True, COL_ACCENT_BRIGHT)
        self._screen.blit(mix_label, (rx + pad, y))
        y += mix_label.get_height() + SPACE_XS

        mix_rect = pygame.Rect(rx + pad, y, graph_w, graph_h)
        self._draw_card(mix_rect, COL_GRAPH_BG, COL_ACCENT)
        mix_peak = self._draw_waveform(mix_rect, mix_data, COL_ACCENT_BRIGHT, PROCESSED_DISPLAY_MULTIPLIER)

        # Peak readout (monospace)
        peak_str = f"peak: {mix_peak:.4f}"
        peak_surf = self._font_mono.render(peak_str, True, COL_TEXT_DIM)
        self._screen.blit(peak_surf, (rx + pad + graph_w - peak_surf.get_width(), y - peak_surf.get_height() - 1))
        y += graph_h + SPACE_MD

        # Separator line
        pygame.draw.line(self._screen, COL_GRID_LINE, (rx + pad, y), (rx + right_sidebar_w - pad, y))
        y += SPACE_MD

        # ---- Per-source graphs ----
        sources = self._state.get_sources()
        for sid in sorted(sources.keys()):
            if y + graph_h * 3 + 68 > win_h:
                break  # no room for more (raw + processed + spectrum)

            label = self._font_label.render(f"Source {sid}", True, COL_SOURCE)
            self._screen.blit(label, (rx + pad, y))
            y += label.get_height() + SPACE_XS

            # --- Raw waveform graph ---
            raw_label = self._font_small.render("raw", True, COL_TEXT_DIM)
            self._screen.blit(raw_label, (rx + pad, y))

            raw_rect = pygame.Rect(rx + pad, y + raw_label.get_height() + 2, graph_w, graph_h)
            self._draw_card(raw_rect, COL_GRAPH_BG)
            raw_data = self._audio.get_last_raw(sid)
            raw_peak = self._draw_waveform(raw_rect, raw_data, COL_GRAPH_RAW, RAW_DISPLAY_MULTIPLIER)

            raw_peak_surf = self._font_mono.render(f"peak: {raw_peak:.4f}", True, COL_TEXT_DIM)
            self._screen.blit(raw_peak_surf, (rx + pad + graph_w - raw_peak_surf.get_width(), y))
            y += raw_label.get_height() + 2 + graph_h + SPACE_SM

            # --- Processed waveform graph (left channel) ---
            proc_data = self._audio.get_last_processed(sid)
            if proc_data is not None and proc_data.ndim == 2:
                proc_data = proc_data[:, 0]  # left channel only

            proc_label = self._font_small.render("processed (L)", True, COL_TEXT_DIM)
            self._screen.blit(proc_label, (rx + pad, y))

            proc_rect = pygame.Rect(rx + pad, y + proc_label.get_height() + 2, graph_w, graph_h)
            self._draw_card(proc_rect, COL_GRAPH_BG)
            proc_peak = self._draw_waveform(proc_rect, proc_data, COL_GRAPH_PROC, PROCESSED_DISPLAY_MULTIPLIER)

            proc_peak_surf = self._font_mono.render(f"peak: {proc_peak:.4f}", True, COL_TEXT_DIM)
            self._screen.blit(proc_peak_surf, (rx + pad + graph_w - proc_peak_surf.get_width(), y))
            y += proc_label.get_height() + 2 + graph_h + SPACE_SM

            # --- Spectrum bar-graph panel ---
            spec_label = self._font_small.render("spectrum", True, COL_TEXT_DIM)
            self._screen.blit(spec_label, (rx + pad, y))
            spec_rect = pygame.Rect(rx + pad, y + spec_label.get_height() + 2, graph_w, graph_h)
            self._draw_card(spec_rect, COL_GRAPH_BG)

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
            y += spec_label.get_height() + 2 + graph_h + SPACE_MD

    def _draw_waveform(
        self,
        rect: pygame.Rect,
        data: Optional[np.ndarray],
        color: tuple[int, int, int],
        multiplier: float = 1.0,
    ) -> float:
        """Render a smooth anti-aliased waveform inside *rect* using viz_engine.

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
            # Use anti-aliased lines for smoother waveform rendering
            pygame.draw.aalines(self._screen, color, False, points)

        return peak

    # ------------------------------------------------------------------
    # Draw the dragged item ghost (Part 6: fixed alpha via SRCALPHA surface)
    # ------------------------------------------------------------------
    def _draw_drag_ghost(self) -> None:
        if self._dragging is None:
            return
        mx, my = self._drag_mouse_pos
        radius = max(2, self._cell_size // 2 - 2)

        if self._dragging == "listener":
            ghost_color = (*COL_LISTENER, 120)
        elif self._dragging == "source":
            ghost_color = (*COL_SOURCE, 120)
        else:
            return

        # Draw onto a temporary SRCALPHA surface so alpha actually works
        diameter = radius * 2 + 4
        ghost_surf = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
        pygame.draw.circle(ghost_surf, ghost_color, (diameter // 2, diameter // 2), radius)
        self._screen.blit(ghost_surf, (mx - diameter // 2, my - diameter // 2))

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------
    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                return

            elif event.type == pygame.VIDEORESIZE:
                self._screen = pygame.display.set_mode(
                    (event.w, event.h), pygame.RESIZABLE | pygame.DOUBLEBUF
                )
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

            self._screen.fill(COL_BG_BASE)
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