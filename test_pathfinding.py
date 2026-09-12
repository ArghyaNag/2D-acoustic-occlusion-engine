"""
test_pathfinding.py -- Verification script for pathfinding.py.

Exercises the find_k_paths() public API with several hand-crafted grid
scenarios and prints clear PASS/FAIL lines a human can skim in a terminal.

Usage:  python test_pathfinding.py
"""

from pathfinding import find_k_paths, PathResult
from shared_state import GRID_COLS, GRID_ROWS


def _pf(label: str, ok: bool, detail: str = "") -> None:
    """Print a PASS/FAIL line."""
    tag = "PASS" if ok else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {label}{suffix}")


def test_trivial_start_equals_end() -> None:
    print("\n--- Test: start == end (trivial) ---")
    results = find_k_paths((5, 5), (5, 5), set(), GRID_COLS, GRID_ROWS, k=1)
    _pf("returns exactly 1 result", len(results) == 1, f"got {len(results)}")
    if results:
        r = results[0]
        _pf("path == [start]", r.cells == [(5, 5)], f"got {r.cells}")
        _pf("length == 0.0", r.length == 0.0, f"got {r.length}")
        _pf("corners == 0", r.corners == 0, f"got {r.corners}")


def test_clear_grid_straight_line() -> None:
    print("\n--- Test: clear grid, straight horizontal line ---")
    start = (2, 5)
    end = (10, 5)
    results = find_k_paths(start, end, set(), GRID_COLS, GRID_ROWS, k=1)
    _pf("returns >= 1 result", len(results) >= 1, f"got {len(results)}")
    if results:
        r = results[0]
        _pf("path starts at start", r.cells[0] == start, f"got {r.cells[0]}")
        _pf("path ends at end", r.cells[-1] == end, f"got {r.cells[-1]}")
        # Straight horizontal: length should be 8.0, corners should be 0
        _pf("length == 8.0 (8 horizontal steps)", abs(r.length - 8.0) < 1e-9, f"got {r.length:.6f}")
        _pf("corners == 0 (straight line)", r.corners == 0, f"got {r.corners}")
        print(f"       path: {r.cells}")


def test_clear_grid_diagonal() -> None:
    print("\n--- Test: clear grid, diagonal line ---")
    start = (0, 0)
    end = (5, 5)
    results = find_k_paths(start, end, set(), GRID_COLS, GRID_ROWS, k=1)
    _pf("returns >= 1 result", len(results) >= 1, f"got {len(results)}")
    if results:
        r = results[0]
        import math
        expected_len = 5 * math.sqrt(2)
        _pf(
            f"length ~= {expected_len:.4f} (5 diagonal steps)",
            abs(r.length - expected_len) < 1e-9,
            f"got {r.length:.6f}",
        )
        _pf("corners == 0 (straight diagonal)", r.corners == 0, f"got {r.corners}")
        print(f"       path: {r.cells}")


def test_wall_detour() -> None:
    print("\n--- Test: single wall segment forces detour ---")
    # Place a vertical wall at x=5, y=3..7, with start at (3,5) and end at (8,5).
    # The shortest path must go around the wall.
    walls = {(5, y) for y in range(3, 8)}
    start = (3, 5)
    end = (8, 5)
    results = find_k_paths(start, end, walls, GRID_COLS, GRID_ROWS, k=1)
    _pf("returns >= 1 result", len(results) >= 1, f"got {len(results)}")
    if results:
        r = results[0]
        _pf("path starts at start", r.cells[0] == start, f"got {r.cells[0]}")
        _pf("path ends at end", r.cells[-1] == end, f"got {r.cells[-1]}")
        _pf("corners > 0 (had to detour)", r.corners > 0, f"got {r.corners}")
        _pf("length > 5.0 (longer than straight)", r.length > 5.0, f"got {r.length:.4f}")
        # Verify no wall cell appears in the path (except possibly endpoints,
        # but our endpoints are not on walls here).
        path_hits_wall = any(c in walls for c in r.cells)
        _pf("path does not pass through walls", not path_hits_wall)
        print(f"       path ({len(r.cells)} cells): {r.cells}")


def test_fully_walled_off() -> None:
    print("\n--- Test: end completely sealed off -> empty list ---")
    # Box around (20, 12) -- seal it with a ring of walls.
    walls = set()
    for x in range(19, 22):
        walls.add((x, 11))
        walls.add((x, 13))
    for y in range(11, 14):
        walls.add((19, y))
        walls.add((21, y))

    start = (0, 0)
    end = (20, 12)
    results = find_k_paths(start, end, walls, GRID_COLS, GRID_ROWS, k=1)
    _pf("returns empty list (no path)", len(results) == 0, f"got {len(results)}")


def test_k_exceeds_available() -> None:
    print("\n--- Test: k=3 requested, only 1 sensible route exists ---")
    # Very constrained corridor: a narrow channel.
    # Walls block everything except a 1-cell-wide corridor from (0,0) to (5,0).
    walls = set()
    for x in range(0, 6):
        walls.add((x, 1))  # wall below the corridor

    start = (0, 0)
    end = (5, 0)
    results = find_k_paths(start, end, walls, GRID_COLS, GRID_ROWS, k=3)
    _pf("returns <= 3 results (no error)", len(results) <= 3, f"got {len(results)}")
    _pf("returns >= 1 result", len(results) >= 1, f"got {len(results)}")
    if results:
        _pf("first result is shortest", all(r.length >= results[0].length for r in results))
        for i, r in enumerate(results):
            print(f"       path {i+1}: length={r.length:.4f}, corners={r.corners}, cells={r.cells}")


def test_k_multiple_paths() -> None:
    print("\n--- Test: k=3 in open area, verify multiple distinct paths ---")
    start = (5, 5)
    end = (10, 5)
    results = find_k_paths(start, end, set(), GRID_COLS, GRID_ROWS, k=3)
    _pf("returns >= 1 result", len(results) >= 1, f"got {len(results)}")
    if len(results) > 1:
        # Verify paths are distinct
        path_sets = [tuple(r.cells) for r in results]
        unique = len(set(path_sets))
        _pf("all returned paths are distinct", unique == len(results), f"{unique} unique of {len(results)}")
        # Verify sorted by length
        lengths = [r.length for r in results]
        _pf("paths sorted by length", all(lengths[i] <= lengths[i+1] + 1e-9 for i in range(len(lengths)-1)),
            f"lengths: {[f'{l:.4f}' for l in lengths]}")
    for i, r in enumerate(results):
        print(f"       path {i+1}: length={r.length:.4f}, corners={r.corners}")


def test_start_on_wall() -> None:
    print("\n--- Test: start cell is itself a wall (not self-blocking) ---")
    walls = {(3, 3)}
    start = (3, 3)
    end = (6, 3)
    results = find_k_paths(start, end, walls, GRID_COLS, GRID_ROWS, k=1)
    _pf("returns >= 1 result (not blocked by standing on wall)", len(results) >= 1, f"got {len(results)}")
    if results:
        r = results[0]
        _pf("path starts at start", r.cells[0] == start)
        _pf("path ends at end", r.cells[-1] == end)
        _pf("length == 3.0 (straight line)", abs(r.length - 3.0) < 1e-9, f"got {r.length:.4f}")


def test_float_inputs_rounded() -> None:
    print("\n--- Test: float inputs are rounded to int correctly ---")
    # Pass floats instead of ints (simulating dsp_engine.py's float() casts)
    results = find_k_paths((2.7, 5.1), (7.3, 5.0), set(), GRID_COLS, GRID_ROWS, k=1)
    _pf("returns >= 1 result", len(results) >= 1, f"got {len(results)}")
    if results:
        r = results[0]
        # (2.7, 5.1) rounds to (3, 5); (7.3, 5.0) rounds to (7, 5)
        _pf("start rounded to (3,5)", r.cells[0] == (3, 5), f"got {r.cells[0]}")
        _pf("end rounded to (7,5)", r.cells[-1] == (7, 5), f"got {r.cells[-1]}")
        _pf("length == 4.0", abs(r.length - 4.0) < 1e-9, f"got {r.length:.4f}")


def main() -> None:
    print("=" * 60)
    print("  pathfinding.py -- verification script")
    print("=" * 60)

    test_trivial_start_equals_end()
    test_clear_grid_straight_line()
    test_clear_grid_diagonal()
    test_wall_detour()
    test_fully_walled_off()
    test_k_exceeds_available()
    test_k_multiple_paths()
    test_start_on_wall()
    test_float_inputs_rounded()

    print("\n" + "=" * 60)
    print("  Done.  Review PASS/FAIL lines above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
