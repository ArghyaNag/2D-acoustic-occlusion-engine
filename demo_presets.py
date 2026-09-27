"""
demo_presets.py — Declarative preset definitions for live presentation demos.

Each preset is a dictionary with:
  - title:       Short name shown in the UI banner / dropdown.
  - description: One-liner explaining the acoustic phenomenon.
  - listener:    (grid_x, grid_y) for the listener placement.
  - sources:     List of {"pos": (x, y), "file": str} dicts.
  - walls:       {(col, row): reflectivity_gain, ...} mapping.

Materials:
  WALL_GAIN_HARD   = 0.90  (Concrete)
  WALL_GAIN_MEDIUM = 0.60  (Wood)
  WALL_GAIN_SOFT   = 0.25  (Curtain)
"""

from shared_state import WALL_GAIN_HARD, WALL_GAIN_MEDIUM, WALL_GAIN_SOFT

# Default audio file used across most presets
_DEFAULT_AUDIO = "audio_files/file_example_WAV_1MG.wav"

PRESETS = {
    1: {
        "title": "Distance & Panning",
        "description": "Pure 1/d attenuation and stereo azimuthal panning on an open grid.",
        "listener": (20, 12),
        "sources": [
            {"pos": (10, 12), "file": _DEFAULT_AUDIO},
        ],
        "walls": {},
    },

    2: {
        "title": "Obstruction Lowpass",
        "description": "Vertical concrete wall obstructs path — hear treble collapse in real time.",
        "listener": (25, 12),
        "sources": [
            {"pos": (15, 12), "file": _DEFAULT_AUDIO},
        ],
        "walls": {
            (20, r): WALL_GAIN_HARD for r in range(7, 18)
        },
    },

    3: {
        "title": "Corner Diffraction",
        "description": "Compare corner diffraction vs open line-of-sight.",
        "listener": (26, 16),
        "sources": [
            {"pos": (14, 8), "file": _DEFAULT_AUDIO},   # Source 1: behind L-wall (diffracted, muffled)
            {"pos": (32, 6), "file": _DEFAULT_AUDIO},   # Source 2: direct LoS (further, less masked)
        ],
        "walls": {
            # Horizontal segment: row 12, cols 18..26
            **{(c, 12): WALL_GAIN_HARD for c in range(18, 27)},
            # Vertical segment: col 18, rows 6..12
            **{(18, r): WALL_GAIN_HARD for r in range(6, 12)},
        },
    },

    4: {
        "title": "Delay Line Echo",
        "description": "Concrete wall on the right produces a distinct slapback echo via ring buffer.",
        "listener": (16, 12),
        "sources": [
            {"pos": (12, 12), "file": _DEFAULT_AUDIO},
        ],
        "walls": {
            (32, r): WALL_GAIN_HARD for r in range(4, 21)
        },
    },

    5: {
        "title": "Through-Wall Transmission",
        "description": "Source sealed inside a concrete room — sound bleeds through without snapping.",
        "listener": (26, 12),
        "sources": [
            {"pos": (14, 12), "file": _DEFAULT_AUDIO},
        ],
        "walls": {
            # Top wall: row 8, cols 10..18
            **{(c, 8): WALL_GAIN_HARD for c in range(10, 19)},
            # Bottom wall: row 16, cols 10..18
            **{(c, 16): WALL_GAIN_HARD for c in range(10, 19)},
            # Left wall: col 10, rows 8..16
            **{(10, r): WALL_GAIN_HARD for r in range(8, 17)},
            # Right wall: col 18, rows 8..16
            **{(18, r): WALL_GAIN_HARD for r in range(8, 17)},
        },
    },

    6: {
        "title": "Multi-Echo Network",
        "description": "Two side walls produce spatially separated echoes with independent panning.",
        "listener": (20, 14),
        "sources": [
            {"pos": (20, 10), "file": _DEFAULT_AUDIO},
        ],
        "walls": {
            # Left concrete wall: col 6, rows 4..20 (~70 ms, panned left, bright)
            **{(6, r): WALL_GAIN_HARD for r in range(4, 21)},
            # Right wood wall: col 34, rows 4..20 (~110 ms, panned right, warmer)
            **{(34, r): WALL_GAIN_MEDIUM for r in range(4, 21)},
        },
    },

    7: {
        "title": "Grand Superposition",
        "description": "Multi-room scene: diffraction + bleed + echoes — all three families live.",
        "listener": (30, 12),
        "sources": [
            {"pos": (12, 12), "file": _DEFAULT_AUDIO},
        ],
        "walls": {
            # Room enclosure (with doorway gap at rows 11-13 on the right wall)
            **{(c, 6): WALL_GAIN_HARD for c in range(8, 22)},    # top
            **{(c, 18): WALL_GAIN_HARD for c in range(8, 22)},   # bottom
            **{(8, r): WALL_GAIN_HARD for r in range(6, 19)},    # left
            # Right wall with doorway gap (rows 11, 12, 13 open)
            **{(22, r): WALL_GAIN_HARD for r in range(6, 11)},
            **{(22, r): WALL_GAIN_HARD for r in range(14, 19)},
            # Exterior side reflector
            **{(36, r): WALL_GAIN_MEDIUM for r in range(8, 17)},
        },
    },

    8: {
        "title": "Material Diversity",
        "description": "Concrete vs. curtain side-by-side — hear contrasting transmission & reflection.",
        "listener": (20, 16),
        "sources": [
            {"pos": (20, 8), "file": _DEFAULT_AUDIO},
        ],
        "walls": {
            # Left barrier: Concrete (high reflectivity, low transmission)
            **{(14, r): WALL_GAIN_HARD for r in range(10, 15)},
            # Right barrier: Curtain (low reflectivity, high transmission)
            **{(26, r): WALL_GAIN_SOFT for r in range(10, 15)},
        },
    },
}

# Total number of presets
NUM_PRESETS = len(PRESETS)
