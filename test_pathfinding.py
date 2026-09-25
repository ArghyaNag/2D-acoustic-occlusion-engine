"""
test_pathfinding.py -- Verification script for pathfinding.py.

Exercises the find_k_paths() public API with several hand-crafted grid
scenarios and prints clear PASS/FAIL lines a human can skim in a terminal.

Usage:  python test_pathfinding.py
"""

from pathfinding import (
    find_k_paths, PathResult,
    find_reflector_candidates, find_transmission_path,
    ReflectorCandidate, has_line_of_sight,
)
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


# ---------------------------------------------------------------------------
# Stage 1C tests: reflector candidates, transmission path
# ---------------------------------------------------------------------------


def test_reflector_single_candidate() -> None:
    """Single wall cell as a plausible reflector -- should return exactly one candidate."""
    print("\n--- Test: reflector -- single candidate ---")
    source = (20, 12)
    listener = (20, 5)
    # Wall directly south of source, on ray index 3 (angle = pi/2).
    # Step 5: round(20 + 5*0, 12 + 5*1) = (20, 17).
    walls = {(20, 17)}
    wall_gains = {(20, 17): 0.7}
    results = find_reflector_candidates(
        source, listener, walls, wall_gains, GRID_COLS, GRID_ROWS,
    )
    _pf("returns exactly 1 candidate", len(results) == 1, f"got {len(results)}")
    if results:
        _pf("wall_cell == (20, 17)", results[0].wall_cell == (20, 17),
            f"got {results[0].wall_cell}")
        _pf("material_gain == 0.7", results[0].material_gain == 0.7,
            f"got {results[0].material_gain}")


def test_reflector_multiple_sorted_by_loudness() -> None:
    """Three wall cells at varying distances/gains -- verify loudness sort order."""
    print("\n--- Test: reflector -- multiple candidates, sorted by loudness ---")
    source = (20, 12)
    listener = (20, 2)
    # Three walls on cardinal rays from source:
    #   (20, 15) -- south ray (i=3), distance 3, gain 0.9
    #   (17, 12) -- west ray  (i=6), distance 3, gain 0.8
    #   (23, 12) -- east ray  (i=0), distance 3, gain 0.5
    walls = {(20, 15), (17, 12), (23, 12)}
    wall_gains = {(20, 15): 0.9, (17, 12): 0.8, (23, 12): 0.5}

    # Hand-computed loudness = (1/(1 + 0.08*2*d)) * gain:
    #   (20,15): d=3 -> (1/1.48)*0.9 = 0.60811...
    #   (17,12): d=3 -> (1/1.48)*0.8 = 0.54054...
    #   (23,12): d=3 -> (1/1.48)*0.5 = 0.33784...
    # Expected order (descending loudness): (20,15), (17,12), (23,12)
    expected_order = [(20, 15), (17, 12), (23, 12)]

    results = find_reflector_candidates(
        source, listener, walls, wall_gains, GRID_COLS, GRID_ROWS,
    )
    actual_order = [c.wall_cell for c in results]
    _pf("returns 3 candidates", len(results) == 3, f"got {len(results)}")
    _pf("wall_cells in correct loudness order",
        actual_order == expected_order,
        f"expected {expected_order}, got {actual_order}")


def test_reflector_none_in_radius() -> None:
    """No walls at all -- should return empty list without error."""
    print("\n--- Test: reflector -- no walls in radius ---")
    source = (20, 12)
    listener = (20, 5)
    walls: set[tuple[int, int]] = set()
    wall_gains: dict[tuple[int, int], float] = {}
    results = find_reflector_candidates(
        source, listener, walls, wall_gains, GRID_COLS, GRID_ROWS,
    )
    _pf("returns empty list", results == [], f"got {results}")


def test_reflector_sealed_source() -> None:
    """Source fully sealed by walls -- no routed path, but reflectors still found."""
    print("\n--- Test: reflector -- sealed source still finds candidates ---")
    source = (20, 12)
    listener = (10, 12)
    # Seal source with a ring of walls at distance 1 (8 cells surrounding it).
    seal = set()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            seal.add((20 + dx, 12 + dy))
    walls = seal
    wall_gains = {cell: 0.6 for cell in walls}

    # Precondition: no routed path exists through the seal.
    routed = find_k_paths(source, listener, walls, GRID_COLS, GRID_ROWS, k=1)
    _pf("precondition: no routed path (source sealed)", len(routed) == 0,
        f"got {len(routed)} paths")

    # Despite the seal, ray i=6 (angle=pi, west) hits (19,12) at step 1.
    # source->(19,12) LoS: Bresenham walk from (20,12) to (19,12) has only
    # 1 step and both endpoints are excluded from wall check -> True.
    # (19,12)->(10,12) LoS: straight horizontal line, no walls between -> True.
    _pf("precondition: source->wall(19,12) LoS clear",
        has_line_of_sight(source, (19, 12), walls))
    _pf("precondition: wall(19,12)->listener LoS clear",
        has_line_of_sight((19, 12), listener, walls))

    results = find_reflector_candidates(
        source, listener, walls, wall_gains, GRID_COLS, GRID_ROWS,
    )
    _pf("returns >= 1 candidate", len(results) >= 1, f"got {len(results)}")
    found_cells = [c.wall_cell for c in results]
    _pf("(19, 12) is among candidates", (19, 12) in found_cells,
        f"got {found_cells}")


def test_transmission_zero_walls() -> None:
    """No walls on direct line -- should return None."""
    print("\n--- Test: transmission -- zero walls crossed ---")
    source = (5, 10)
    listener = (15, 10)
    walls: set[tuple[int, int]] = set()
    wall_gains: dict[tuple[int, int], float] = {}
    result = find_transmission_path(
        source, listener, walls, wall_gains, GRID_COLS, GRID_ROWS,
    )
    _pf("returns None", result is None, f"got {result}")


def test_transmission_one_wall() -> None:
    """One wall cell on the direct line -- returns list of length 1."""
    print("\n--- Test: transmission -- one wall crossed ---")
    source = (5, 10)
    listener = (15, 10)
    # (10, 10) is directly on the horizontal Bresenham line between them.
    walls = {(10, 10)}
    wall_gains = {(10, 10): 0.4}
    result = find_transmission_path(
        source, listener, walls, wall_gains, GRID_COLS, GRID_ROWS,
    )
    _pf("returns a list (not None)", result is not None, f"got {result}")
    if result is not None:
        _pf("length == 1", len(result) == 1, f"got {len(result)}")
        _pf("cell == (10, 10)", result[0][0] == (10, 10), f"got {result[0][0]}")
        _pf("gain == 0.4", result[0][1] == 0.4, f"got {result[0][1]}")


def test_transmission_multiple_walls() -> None:
    """Two walls on the direct line -- returns both in source-to-listener order."""
    print("\n--- Test: transmission -- multiple walls crossed ---")
    source = (5, 10)
    listener = (15, 10)
    # (8, 10) and (12, 10) are both on the horizontal Bresenham line.
    # Source is at x=5 going toward x=15, so (8,10) is crossed first.
    walls = {(8, 10), (12, 10)}
    wall_gains = {(8, 10): 0.3, (12, 10): 0.7}
    result = find_transmission_path(
        source, listener, walls, wall_gains, GRID_COLS, GRID_ROWS,
    )
    expected = [((8, 10), 0.3), ((12, 10), 0.7)]
    _pf("returns a list (not None)", result is not None, f"got {result}")
    if result is not None:
        _pf("length == 2", len(result) == 2, f"got {len(result)}")
        _pf("full list matches expected order and gains",
            result == expected,
            f"expected {expected}, got {result}")


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

    test_reflector_single_candidate()
    test_reflector_multiple_sorted_by_loudness()
    test_reflector_none_in_radius()
    test_reflector_sealed_source()
    test_transmission_zero_walls()
    test_transmission_one_wall()
    test_transmission_multiple_walls()

    print("\n" + "=" * 60)
    print("  Done.  Review PASS/FAIL lines above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
